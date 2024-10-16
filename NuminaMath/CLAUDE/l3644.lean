import Mathlib

namespace NUMINAMATH_CALUDE_hexagonal_tile_difference_l3644_364432

theorem hexagonal_tile_difference (initial_blue : ℕ) (initial_green : ℕ) (border_tiles : ℕ) : 
  initial_blue = 20 → initial_green = 15 → border_tiles = 18 →
  (initial_green + 2 * border_tiles) - initial_blue = 31 := by
sorry

end NUMINAMATH_CALUDE_hexagonal_tile_difference_l3644_364432


namespace NUMINAMATH_CALUDE_trigonometric_roots_problem_l3644_364422

open Real

theorem trigonometric_roots_problem (α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : (tan α)^2 - 5*(tan α) + 6 = 0) (h4 : (tan β)^2 - 5*(tan β) + 6 = 0) :
  (α + β = 3*π/4) ∧ (¬ ∃ (x : ℝ), tan (2*(α + β)) = x) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_roots_problem_l3644_364422


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l3644_364434

theorem triangle_angle_difference (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  b = a * Real.tan B →
  A > π / 2 →
  A - B = π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_difference_l3644_364434


namespace NUMINAMATH_CALUDE_coin_grid_intersection_probability_l3644_364423

/-- Probability of a coin intersecting grid lines -/
theorem coin_grid_intersection_probability
  (grid_edge_length : ℝ)
  (coin_diameter : ℝ)
  (h_grid : grid_edge_length = 6)
  (h_coin : coin_diameter = 2) :
  (1 : ℝ) - (grid_edge_length - coin_diameter)^2 / grid_edge_length^2 = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_coin_grid_intersection_probability_l3644_364423


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3644_364439

/-- The slope of a line tangent to a circle --/
theorem tangent_line_slope (k : ℝ) : 
  (∀ x y : ℝ, y = k * (x - 2) + 2 → x^2 + y^2 - 2*x - 2*y = 0 → 
   ∃! x y : ℝ, y = k * (x - 2) + 2 ∧ x^2 + y^2 - 2*x - 2*y = 0) → 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3644_364439


namespace NUMINAMATH_CALUDE_factorization_equality_l3644_364445

theorem factorization_equality (a b : ℝ) : a^2 * b - 6 * a * b + 9 * b = b * (a - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3644_364445


namespace NUMINAMATH_CALUDE_blue_parrots_count_l3644_364427

theorem blue_parrots_count (total : ℕ) (green_fraction : ℚ) : 
  total = 108 →
  green_fraction = 5/6 →
  (total : ℚ) * (1 - green_fraction) = 18 := by
  sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l3644_364427


namespace NUMINAMATH_CALUDE_solution_to_equation_l3644_364493

theorem solution_to_equation : ∃ x y : ℤ, 5 * x + 4 * y = 14 ∧ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3644_364493


namespace NUMINAMATH_CALUDE_crickets_found_later_l3644_364441

theorem crickets_found_later (initial : ℝ) (final : ℕ) : initial = 7.0 → final = 18 → final - initial = 11 := by
  sorry

end NUMINAMATH_CALUDE_crickets_found_later_l3644_364441


namespace NUMINAMATH_CALUDE_basketball_substitutions_l3644_364406

/-- Represents the number of ways to make exactly n substitutions -/
def b (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 5 * (11 - n) * b n

/-- The total number of ways to make substitutions -/
def m : ℕ := (b 0) + (b 1) + (b 2) + (b 3) + (b 4) + (b 5)

theorem basketball_substitutions :
  m % 1000 = 301 := by sorry

end NUMINAMATH_CALUDE_basketball_substitutions_l3644_364406


namespace NUMINAMATH_CALUDE_taehyung_candies_l3644_364482

def total_candies : ℕ := 6
def seokjin_eats : ℕ := 4

theorem taehyung_candies : total_candies - seokjin_eats = 2 := by
  sorry

end NUMINAMATH_CALUDE_taehyung_candies_l3644_364482


namespace NUMINAMATH_CALUDE_rick_ironing_l3644_364451

/-- The number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := 4

/-- The number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := 3

/-- The number of hours Rick spends ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- The number of hours Rick spends ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- The total number of pieces of clothing Rick has ironed -/
def total_pieces : ℕ := shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants

theorem rick_ironing :
  total_pieces = 27 := by sorry

end NUMINAMATH_CALUDE_rick_ironing_l3644_364451


namespace NUMINAMATH_CALUDE_median_circumradius_inequality_l3644_364452

/-- A triangle with medians and circumradius -/
structure Triangle where
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  R : ℝ

/-- Theorem about the relationship between medians and circumradius of a triangle -/
theorem median_circumradius_inequality (t : Triangle) :
  t.m_a^2 + t.m_b^2 + t.m_c^2 ≤ 27 * t.R^2 / 4 ∧
  t.m_a + t.m_b + t.m_c ≤ 9 * t.R / 2 :=
by sorry

end NUMINAMATH_CALUDE_median_circumradius_inequality_l3644_364452


namespace NUMINAMATH_CALUDE_quadratic_solutions_l3644_364489

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 5

-- State the theorem
theorem quadratic_solutions (b : ℝ) :
  (∀ x, f b x = x^2 + b*x - 5) →  -- Definition of f
  (-b/(2:ℝ) = 2) →               -- Axis of symmetry condition
  (∀ x, f b x = 2*x - 13 ↔ x = 2 ∨ x = 4) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_solutions_l3644_364489


namespace NUMINAMATH_CALUDE_big_fifteen_games_l3644_364454

/-- Represents the Big Fifteen Basketball Conference -/
structure BigFifteenConference where
  numDivisions : Nat
  teamsPerDivision : Nat
  intraDivisionGames : Nat
  interDivisionGames : Nat
  nonConferenceGames : Nat

/-- Calculates the total number of games in the conference -/
def totalGames (conf : BigFifteenConference) : Nat :=
  let intraDivisionTotal := conf.numDivisions * (conf.teamsPerDivision.choose 2) * conf.intraDivisionGames
  let interDivisionTotal := conf.numDivisions * conf.teamsPerDivision * (conf.numDivisions - 1) * conf.teamsPerDivision / 2
  let nonConferenceTotal := conf.numDivisions * conf.teamsPerDivision * conf.nonConferenceGames
  intraDivisionTotal + interDivisionTotal + nonConferenceTotal

/-- Theorem stating that the total number of games in the Big Fifteen Conference is 270 -/
theorem big_fifteen_games :
  totalGames {
    numDivisions := 3,
    teamsPerDivision := 5,
    intraDivisionGames := 3,
    interDivisionGames := 1,
    nonConferenceGames := 2
  } = 270 := by sorry


end NUMINAMATH_CALUDE_big_fifteen_games_l3644_364454


namespace NUMINAMATH_CALUDE_percent_democrats_voters_l3644_364431

theorem percent_democrats_voters (d r : ℝ) : 
  d + r = 100 →
  0.75 * d + 0.2 * r = 53 →
  d = 60 :=
by sorry

end NUMINAMATH_CALUDE_percent_democrats_voters_l3644_364431


namespace NUMINAMATH_CALUDE_squares_in_unit_square_l3644_364463

/-- Two squares with side lengths a and b contained in a unit square without sharing interior points have a + b ≤ 1 -/
theorem squares_in_unit_square (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) 
  (contained : a ≤ 1 ∧ b ≤ 1) 
  (no_overlap : ∃ (x y x' y' : ℝ), 
    0 ≤ x ∧ x + a ≤ 1 ∧ 
    0 ≤ y ∧ y + a ≤ 1 ∧
    0 ≤ x' ∧ x' + b ≤ 1 ∧ 
    0 ≤ y' ∧ y' + b ≤ 1 ∧
    (x + a ≤ x' ∨ x' + b ≤ x ∨ y + a ≤ y' ∨ y' + b ≤ y)) : 
  a + b ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_squares_in_unit_square_l3644_364463


namespace NUMINAMATH_CALUDE_max_volume_container_l3644_364481

/-- Represents the dimensions and volume of a rectangular container --/
structure Container where
  shorter_side : ℝ
  longer_side : ℝ
  height : ℝ
  volume : ℝ

/-- Calculates the volume of a container given its dimensions --/
def calculate_volume (c : Container) : ℝ :=
  c.shorter_side * c.longer_side * c.height

/-- Defines the constraints for the container based on the problem --/
def is_valid_container (c : Container) : Prop :=
  c.longer_side = c.shorter_side + 0.5 ∧
  c.height = 3.2 - 2 * c.shorter_side ∧
  4 * (c.shorter_side + c.longer_side + c.height) = 14.8 ∧
  c.volume = calculate_volume c

/-- Theorem stating the maximum volume and corresponding height --/
theorem max_volume_container :
  ∃ (c : Container), is_valid_container c ∧
    c.volume = 1.8 ∧
    c.height = 1.2 ∧
    ∀ (c' : Container), is_valid_container c' → c'.volume ≤ c.volume :=
  sorry

end NUMINAMATH_CALUDE_max_volume_container_l3644_364481


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_l3644_364437

theorem ac_squared_gt_bc_squared (a b c : ℝ) : a > b → a * c^2 > b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_l3644_364437


namespace NUMINAMATH_CALUDE_expression_evaluation_l3644_364494

theorem expression_evaluation (x y : ℝ) 
  (h : |x + 1/2| + (y - 1)^2 = 0) : 
  5 * x^2 * y - (6 * x * y - 2 * (x * y - 2 * x^2 * y) - x * y^2) + 4 * x * y = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3644_364494


namespace NUMINAMATH_CALUDE_x_gt_1_necessary_not_sufficient_l3644_364469

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := ![x - 3, 2]
def b : Fin 2 → ℝ := ![1, 1]

-- Define the dot product of two 2D vectors
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

-- Define what it means for the angle between two vectors to be acute
def is_acute_angle (u v : Fin 2 → ℝ) : Prop := dot_product u v > 0

-- State the theorem
theorem x_gt_1_necessary_not_sufficient :
  (∀ x : ℝ, is_acute_angle (a x) b → x > 1) ∧
  ¬(∀ x : ℝ, x > 1 → is_acute_angle (a x) b) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_1_necessary_not_sufficient_l3644_364469


namespace NUMINAMATH_CALUDE_inequality_proof_l3644_364486

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b * c) / (a^2 + b * c) + (c * a) / (b^2 + c * a) + (a * b) / (c^2 + a * b) ≤
  a / (b + c) + b / (c + a) + c / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3644_364486


namespace NUMINAMATH_CALUDE_cross_number_puzzle_l3644_364410

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def second_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem cross_number_puzzle : 
  ∃! d : ℕ, d < 10 ∧ 
  (∃ m : ℕ, is_three_digit (3^m) ∧ second_digit (3^m) = d) ∧
  (∃ n : ℕ, is_three_digit (7^n) ∧ second_digit (7^n) = d) ∧
  d = 4 :=
sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_l3644_364410


namespace NUMINAMATH_CALUDE_score_well_defined_and_nonnegative_l3644_364457

theorem score_well_defined_and_nonnegative (N C : ℕ+) 
  (h1 : N ≤ 20) (h2 : C ≥ 1) : 
  ∃ (score : ℕ), score = ⌊(N : ℝ) / (C : ℝ)⌋ ∧ score ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_score_well_defined_and_nonnegative_l3644_364457


namespace NUMINAMATH_CALUDE_bill_sunday_saturday_difference_l3644_364479

theorem bill_sunday_saturday_difference (bill_sat bill_sun julia_sun : ℕ) : 
  bill_sun > bill_sat →
  julia_sun = 2 * bill_sun →
  bill_sat + bill_sun + julia_sun = 32 →
  bill_sun = 9 →
  bill_sun - bill_sat = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_bill_sunday_saturday_difference_l3644_364479


namespace NUMINAMATH_CALUDE_triangle_theorem_l3644_364450

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) :
  (t.c * Real.cos t.A + t.a * Real.cos t.C = 2 * t.b * Real.cos t.A) →
  Real.cos t.A = 1 / 2 ∧
  (t.a = Real.sqrt 7 ∧ t.b + t.c = 4) →
  (1 / 2 : ℝ) * t.b * t.c * Real.sqrt (1 - (1 / 2)^2) = 3 * Real.sqrt 3 / 4 :=
by sorry

-- Note: The area formula is expanded as (1/2) * b * c * sin A,
-- where sin A is written as sqrt(1 - cos^2 A)

end NUMINAMATH_CALUDE_triangle_theorem_l3644_364450


namespace NUMINAMATH_CALUDE_max_digit_sum_for_special_fraction_l3644_364456

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the decimal 0.abc where a, b, c are digits -/
def DecimalABC (a b c : Digit) : ℚ := (a.val * 100 + b.val * 10 + c.val : ℕ) / 1000

/-- The theorem statement -/
theorem max_digit_sum_for_special_fraction :
  ∃ (a b c : Digit) (y : ℕ+),
    DecimalABC a b c = (1 : ℚ) / y ∧
    y ≤ 12 ∧
    ∀ (a' b' c' : Digit) (y' : ℕ+),
      DecimalABC a' b' c' = (1 : ℚ) / y' →
      y' ≤ 12 →
      a.val + b.val + c.val ≥ a'.val + b'.val + c'.val ∧
      a.val + b.val + c.val = 8 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_for_special_fraction_l3644_364456


namespace NUMINAMATH_CALUDE_mike_total_spent_l3644_364459

def trumpet_price : Float := 267.35
def song_book_price : Float := 12.95
def trumpet_case_price : Float := 74.50
def cleaning_kit_price : Float := 28.99
def valve_oils_price : Float := 18.75

theorem mike_total_spent : 
  trumpet_price + song_book_price + trumpet_case_price + cleaning_kit_price + valve_oils_price = 402.54 := by
  sorry

end NUMINAMATH_CALUDE_mike_total_spent_l3644_364459


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3644_364484

/-- Given a sum P put at simple interest for 15 years, if increasing the interest rate by 8% 
    results in 2,750 more interest, then P = 2,291.67 -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * R * 15 / 100 + 2750 = P * (R + 8) * 15 / 100) → 
  P = 2291.67 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3644_364484


namespace NUMINAMATH_CALUDE_a_plus_b_equals_seven_thirds_l3644_364409

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -3 * x + 2

-- State the theorem
theorem a_plus_b_equals_seven_thirds 
  (a b : ℝ) 
  (h : ∀ x, g (f a b x) = -2 * x - 3) : 
  a + b = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_seven_thirds_l3644_364409


namespace NUMINAMATH_CALUDE_system_solution_l3644_364416

theorem system_solution :
  let solutions : List (ℝ × ℝ) := [
    (-Real.sqrt (2/5), 2 * Real.sqrt (2/5)),
    (Real.sqrt (2/5), 2 * Real.sqrt (2/5)),
    (Real.sqrt (2/5), -2 * Real.sqrt (2/5)),
    (-Real.sqrt (2/5), -2 * Real.sqrt (2/5))
  ]
  ∀ x y : ℝ,
    (x^2 + y^2 ≤ 2 ∧
     x^4 - 8*x^2*y^2 + 16*y^4 - 20*x^2 - 80*y^2 + 100 = 0) ↔
    (x, y) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3644_364416


namespace NUMINAMATH_CALUDE_jennifer_spending_l3644_364407

theorem jennifer_spending (initial_amount : ℚ) : 
  initial_amount - (initial_amount / 5 + initial_amount / 6 + initial_amount / 2) = 24 →
  initial_amount = 180 := by
sorry

end NUMINAMATH_CALUDE_jennifer_spending_l3644_364407


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3644_364444

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 3}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3644_364444


namespace NUMINAMATH_CALUDE_pam_has_ten_bags_l3644_364425

/-- Represents the number of apples in one of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- Represents the total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- Represents the number of Gerald's bags equivalent to one of Pam's bags -/
def bags_ratio : ℕ := 3

/-- Calculates the number of bags Pam has -/
def pams_bag_count : ℕ := pams_total_apples / (bags_ratio * geralds_bag_count)

/-- Theorem stating that Pam has 10 bags of apples -/
theorem pam_has_ten_bags : pams_bag_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_pam_has_ten_bags_l3644_364425


namespace NUMINAMATH_CALUDE_article_cost_price_l3644_364449

theorem article_cost_price : ∃ C : ℝ, C = 400 ∧ 
  (1.05 * C - (0.95 * C + 0.1 * (0.95 * C)) = 2) := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l3644_364449


namespace NUMINAMATH_CALUDE_equation_equivalence_l3644_364433

theorem equation_equivalence (x : ℝ) :
  (x - 2)^5 + (x - 6)^5 = 32 →
  let z := x - 4
  z^5 + 40*z^3 + 80*z - 32 = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3644_364433


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l3644_364461

-- Define a type for lines
variable (Line : Type)

-- Define a property for line intersection
variable (intersect : Line → Line → Prop)

-- Define a property for lines passing through a common point
variable (pass_through_common_point : (Set Line) → Prop)

-- Define a property for lines lying in one plane
variable (lie_in_one_plane : (Set Line) → Prop)

-- The main theorem
theorem line_intersection_theorem (S : Set Line) :
  (∀ l1 l2 : Line, l1 ∈ S → l2 ∈ S → l1 ≠ l2 → intersect l1 l2) →
  (pass_through_common_point S ∨ lie_in_one_plane S) :=
by sorry


end NUMINAMATH_CALUDE_line_intersection_theorem_l3644_364461


namespace NUMINAMATH_CALUDE_percentage_problem_l3644_364492

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 100 → 
  (P / 100) * N = (3 / 5) * N - 10 → 
  P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3644_364492


namespace NUMINAMATH_CALUDE_triangle_properties_l3644_364499

-- Define the triangle ABC
def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Condition 1: 2c = a + 2b*cos(A)
  2 * c = a + 2 * b * Real.cos A ∧
  -- Condition 2: Area of triangle ABC is √3
  (1 / 2) * a * c * Real.sin B = Real.sqrt 3 ∧
  -- Condition 3: b = √13
  b = Real.sqrt 13

-- Theorem statement
theorem triangle_properties (a b c A B C : ℝ) 
  (h : triangle a b c A B C) : 
  B = Real.pi / 3 ∧ 
  a + b + c = 5 + Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3644_364499


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3644_364403

/-- An arithmetic sequence with its partial sums. -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Partial sums
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem. -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.S 4 = 8)
    (h2 : seq.S 8 = 20) :
    seq.a 11 + seq.a 12 + seq.a 13 + seq.a 14 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3644_364403


namespace NUMINAMATH_CALUDE_marbles_problem_l3644_364465

theorem marbles_problem (total_marbles : ℕ) (marbles_per_boy : ℕ) (h1 : total_marbles = 99) (h2 : marbles_per_boy = 9) :
  total_marbles / marbles_per_boy = 11 :=
by sorry

end NUMINAMATH_CALUDE_marbles_problem_l3644_364465


namespace NUMINAMATH_CALUDE_x_and_y_negative_l3644_364477

theorem x_and_y_negative (x y : ℝ) 
  (h1 : 2 * x - 3 * y > x) 
  (h2 : x + 4 * y < 3 * y) : 
  x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_and_y_negative_l3644_364477


namespace NUMINAMATH_CALUDE_ben_whitewashed_length_l3644_364402

theorem ben_whitewashed_length (total_length : ℝ) (remaining_length : ℝ)
  (h1 : total_length = 100)
  (h2 : remaining_length = 48)
  (h3 : ∃ x : ℝ, 
    remaining_length = total_length - x - 
    (1/5) * (total_length - x) - 
    (1/3) * (total_length - x - (1/5) * (total_length - x))) :
  ∃ x : ℝ, x = 10 ∧ 
    remaining_length = total_length - x - 
    (1/5) * (total_length - x) - 
    (1/3) * (total_length - x - (1/5) * (total_length - x)) :=
by sorry

end NUMINAMATH_CALUDE_ben_whitewashed_length_l3644_364402


namespace NUMINAMATH_CALUDE_evaluate_expression_power_sum_given_equation_l3644_364458

-- Problem 1
theorem evaluate_expression (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5*y) * (-x - 5*y) - (-x + 5*y)^2 = -5.5 := by sorry

-- Problem 2
theorem power_sum_given_equation (a b : ℝ) (h : a^2 - 2*a + b^2 + 4*b + 5 = 0) :
  (a + b)^2013 = -1 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_power_sum_given_equation_l3644_364458


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3644_364460

theorem absolute_value_inequality (x : ℝ) : |5 - 2*x| ≥ 3 ↔ x < 1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3644_364460


namespace NUMINAMATH_CALUDE_sin_minus_cos_value_l3644_364440

theorem sin_minus_cos_value (θ : Real) 
  (h1 : π / 4 < θ) (h2 : θ < π / 2) 
  (h3 : Real.sin θ + Real.cos θ = 5 / 4) : 
  Real.sin θ - Real.cos θ = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_value_l3644_364440


namespace NUMINAMATH_CALUDE_log_equation_solutions_are_integers_l3644_364417

theorem log_equation_solutions_are_integers : ∃ (a b : ℤ), 
  (Real.log (a^2 - 8*a + 20) = 3) ∧ 
  (Real.log (b^2 - 8*b + 20) = 3) ∧ 
  (a ≠ b) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solutions_are_integers_l3644_364417


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3644_364420

/-- Given an arithmetic sequence {aₙ} with common difference d and a₃₀, find a₁ -/
theorem arithmetic_sequence_first_term
  (a : ℕ → ℚ)  -- The arithmetic sequence
  (d : ℚ)      -- Common difference
  (h1 : d = 3/4)
  (h2 : a 30 = 63/4)  -- a₃₀ = 15 3/4 = 63/4
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  : a 1 = -6 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3644_364420


namespace NUMINAMATH_CALUDE_airline_expansion_l3644_364400

theorem airline_expansion (n k : ℕ) : 
  (n * (n - 1)) / 2 + 76 = ((n + k) * (n + k - 1)) / 2 → 
  ((n = 6 ∧ k = 8) ∨ (n = 76 ∧ k = 1)) := by
  sorry

end NUMINAMATH_CALUDE_airline_expansion_l3644_364400


namespace NUMINAMATH_CALUDE_optimal_solution_l3644_364438

/-- Represents the prices and quantities of agricultural products A and B --/
structure AgriProducts where
  price_A : ℝ
  price_B : ℝ
  quantity_A : ℝ
  quantity_B : ℝ

/-- Defines the conditions given in the problem --/
def satisfies_conditions (p : AgriProducts) : Prop :=
  2 * p.price_A + 3 * p.price_B = 690 ∧
  p.price_A + 4 * p.price_B = 720 ∧
  p.quantity_A + p.quantity_B = 40 ∧
  p.price_A * p.quantity_A + p.price_B * p.quantity_B ≤ 5400 ∧
  p.quantity_A ≤ 3 * p.quantity_B

/-- Calculates the profit given the prices and quantities --/
def profit (p : AgriProducts) : ℝ :=
  (160 - p.price_A) * p.quantity_A + (200 - p.price_B) * p.quantity_B

/-- Theorem stating the optimal solution --/
theorem optimal_solution :
  ∃ (p : AgriProducts),
    satisfies_conditions p ∧
    p.price_A = 120 ∧
    p.price_B = 150 ∧
    p.quantity_A = 20 ∧
    p.quantity_B = 20 ∧
    ∀ (q : AgriProducts), satisfies_conditions q → profit q ≤ profit p :=
  sorry


end NUMINAMATH_CALUDE_optimal_solution_l3644_364438


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l3644_364475

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (hr₁ : r₁ = 10)
  (hr₂ : r₂ = 6)
  (hcd : contact_distance = 30) :
  let center_distance := Real.sqrt ((contact_distance ^ 2) + ((r₁ - r₂) ^ 2))
  center_distance = 2 * Real.sqrt 229 := by sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l3644_364475


namespace NUMINAMATH_CALUDE_exists_determining_question_l3644_364435

/-- Represents the type of being (Human or Zombie) --/
inductive Being
| Human
| Zombie

/-- Represents a possible response to a question --/
inductive Response
| Bal
| Yes
| No

/-- Represents a question that can be asked --/
def Question := Being → Response

/-- A function that determines the type of being based on a response --/
def DetermineBeing := Response → Being

/-- Humans always tell the truth, zombies always lie --/
axiom truth_telling (q : Question) :
  ∀ (b : Being), 
    (b = Being.Human → q b = Response.Bal) ∧
    (b = Being.Zombie → q b ≠ Response.Bal)

/-- There exists a question that can determine the type of being --/
theorem exists_determining_question :
  ∃ (q : Question) (d : DetermineBeing),
    ∀ (b : Being), d (q b) = b :=
sorry

end NUMINAMATH_CALUDE_exists_determining_question_l3644_364435


namespace NUMINAMATH_CALUDE_quadratic_polynomial_proof_l3644_364436

theorem quadratic_polynomial_proof :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = (1/3) * (2*x^2 - 4*x + 9)) ∧
    q (-2) = 8 ∧
    q 1 = 2 ∧
    q 3 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_proof_l3644_364436


namespace NUMINAMATH_CALUDE_dividend_in_terms_of_a_l3644_364464

theorem dividend_in_terms_of_a (a : ℝ) :
  let divisor := 25 * quotient
  let divisor' := 7 * remainder
  let quotient_minus_remainder := 15
  let remainder := 3 * a
  let dividend := divisor * quotient + remainder
  dividend = 225 * a^2 + 1128 * a + 5625 := by
  sorry

end NUMINAMATH_CALUDE_dividend_in_terms_of_a_l3644_364464


namespace NUMINAMATH_CALUDE_range_of_a_for_P_and_Q_l3644_364466

theorem range_of_a_for_P_and_Q (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) ↔ 
  a ≤ -2 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_P_and_Q_l3644_364466


namespace NUMINAMATH_CALUDE_equation_solution_l3644_364496

theorem equation_solution : ∃ x : ℝ, 2 * x - 3 = 6 - x :=
  by
    use 3
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3644_364496


namespace NUMINAMATH_CALUDE_delivery_driver_boxes_l3644_364473

/-- Calculates the total number of boxes a delivery driver has -/
def total_boxes (num_stops : ℕ) (boxes_per_stop : ℕ) : ℕ :=
  num_stops * boxes_per_stop

/-- Proves that a delivery driver with 3 stops and 9 boxes per stop has 27 boxes in total -/
theorem delivery_driver_boxes :
  total_boxes 3 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_delivery_driver_boxes_l3644_364473


namespace NUMINAMATH_CALUDE_double_windows_count_l3644_364447

/-- Represents the number of glass panels in each window -/
def panels_per_window : ℕ := 4

/-- Represents the number of single windows upstairs -/
def single_windows : ℕ := 8

/-- Represents the total number of glass panels in the house -/
def total_panels : ℕ := 80

/-- Represents the number of double windows downstairs -/
def double_windows : ℕ := 12

/-- Theorem stating that the number of double windows downstairs is 12 -/
theorem double_windows_count : 
  panels_per_window * double_windows + panels_per_window * single_windows = total_panels :=
by sorry

end NUMINAMATH_CALUDE_double_windows_count_l3644_364447


namespace NUMINAMATH_CALUDE_second_meal_cost_l3644_364414

/-- The cost of a meal consisting of burgers, shakes, and cola. -/
structure MealCost where
  burger : ℝ
  shake : ℝ
  cola : ℝ

/-- The theorem stating the cost of the second meal given the costs of two other meals. -/
theorem second_meal_cost 
  (meal1 : MealCost) 
  (meal2 : MealCost) 
  (h1 : 3 * meal1.burger + 7 * meal1.shake + meal1.cola = 120)
  (h2 : meal2.burger + meal2.shake + meal2.cola = 39)
  (h3 : meal1 = meal2) :
  4 * meal1.burger + 10 * meal1.shake + meal1.cola = 160.5 := by
  sorry

end NUMINAMATH_CALUDE_second_meal_cost_l3644_364414


namespace NUMINAMATH_CALUDE_leonards_age_l3644_364453

theorem leonards_age (leonard nina jerome peter natasha : ℕ) : 
  nina = leonard + 4 →
  nina = jerome / 2 →
  peter = 2 * leonard →
  natasha = peter - 3 →
  leonard + nina + jerome + peter + natasha = 75 →
  leonard = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_leonards_age_l3644_364453


namespace NUMINAMATH_CALUDE_olivia_won_five_games_l3644_364419

/-- Represents a contestant in the math quiz competition -/
inductive Contestant
| Liam
| Noah
| Olivia

/-- The number of games won by a contestant -/
def games_won (c : Contestant) : ℕ :=
  match c with
  | Contestant.Liam => 6
  | Contestant.Noah => 4
  | Contestant.Olivia => 5  -- This is what we want to prove

/-- The number of games lost by a contestant -/
def games_lost (c : Contestant) : ℕ :=
  match c with
  | Contestant.Liam => 3
  | Contestant.Noah => 4
  | Contestant.Olivia => 4

/-- The total number of games played by each contestant -/
def total_games (c : Contestant) : ℕ := games_won c + games_lost c

/-- Each win gives 1 point -/
def points (c : Contestant) : ℕ := games_won c

/-- Theorem stating that Olivia won 5 games -/
theorem olivia_won_five_games :
  (∀ c1 c2 : Contestant, c1 ≠ c2 → total_games c1 = total_games c2) →
  games_won Contestant.Olivia = 5 := by sorry

end NUMINAMATH_CALUDE_olivia_won_five_games_l3644_364419


namespace NUMINAMATH_CALUDE_f_3_equals_3_l3644_364442

def f (x : ℝ) : ℝ := 2 * (x - 1) - 1

theorem f_3_equals_3 : f 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_3_l3644_364442


namespace NUMINAMATH_CALUDE_towel_rate_proof_l3644_364424

/-- Proves that given the specified towel purchases and average price, the unknown rate must be 300. -/
theorem towel_rate_proof (price1 price2 avg_price : ℕ) (count1 count2 count_unknown : ℕ) :
  price1 = 100 →
  price2 = 150 →
  avg_price = 165 →
  count1 = 3 →
  count2 = 5 →
  count_unknown = 2 →
  ∃ (unknown_rate : ℕ),
    (count1 * price1 + count2 * price2 + count_unknown * unknown_rate) / (count1 + count2 + count_unknown) = avg_price ∧
    unknown_rate = 300 :=
by sorry

end NUMINAMATH_CALUDE_towel_rate_proof_l3644_364424


namespace NUMINAMATH_CALUDE_region_location_l3644_364405

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x - 2*y + 6 < 0

-- Theorem statement
theorem region_location :
  ∀ (x y : ℝ), region x y → 
  ∃ (x₀ y₀ : ℝ), line x₀ y₀ ∧ x < x₀ ∧ y > y₀ :=
sorry

end NUMINAMATH_CALUDE_region_location_l3644_364405


namespace NUMINAMATH_CALUDE_solution_equivalence_l3644_364498

def solution_set : Set ℝ := {x | 1 < x ∧ x ≤ 3}

def inequality_system (x : ℝ) : Prop := 1 - x < 0 ∧ x - 3 ≤ 0

theorem solution_equivalence : 
  ∀ x : ℝ, x ∈ solution_set ↔ inequality_system x := by
  sorry

end NUMINAMATH_CALUDE_solution_equivalence_l3644_364498


namespace NUMINAMATH_CALUDE_john_bought_three_spools_l3644_364468

/-- The number of spools John bought given the conditions of the problem -/
def spools_bought (spool_length : ℕ) (wire_per_necklace : ℕ) (necklaces_made : ℕ) : ℕ :=
  (wire_per_necklace * necklaces_made) / spool_length

/-- Theorem stating that John bought 3 spools of wire -/
theorem john_bought_three_spools :
  spools_bought 20 4 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_three_spools_l3644_364468


namespace NUMINAMATH_CALUDE_same_type_monomials_result_l3644_364476

/-- 
Given two monomials of the same type: -x^3 * y^a and 6x^b * y,
prove that (a - b)^3 = -8
-/
theorem same_type_monomials_result (a b : ℤ) : 
  (∀ x y : ℝ, -x^3 * y^a = 6 * x^b * y) → (a - b)^3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_same_type_monomials_result_l3644_364476


namespace NUMINAMATH_CALUDE_first_class_product_rate_l3644_364490

/-- Given a product with a pass rate and a rate of first-class products among qualified products,
    calculate the overall rate of first-class products. -/
theorem first_class_product_rate
  (pass_rate : ℝ)
  (first_class_rate_qualified : ℝ)
  (h1 : pass_rate = 0.95)
  (h2 : first_class_rate_qualified = 0.2) :
  pass_rate * first_class_rate_qualified = 0.19 :=
by sorry

end NUMINAMATH_CALUDE_first_class_product_rate_l3644_364490


namespace NUMINAMATH_CALUDE_rectangle_point_characterization_l3644_364418

/-- A point in the Cartesian coordinate system representing a rectangle's perimeter and area -/
structure RectanglePoint where
  k : ℝ  -- perimeter
  t : ℝ  -- area

/-- Characterizes the region of valid RectanglePoints -/
def is_valid_rectangle_point (p : RectanglePoint) : Prop :=
  p.k > 0 ∧ p.t > 0 ∧ p.t ≤ (p.k^2) / 16

/-- Theorem stating the characterization of valid rectangle points -/
theorem rectangle_point_characterization (p : RectanglePoint) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ p.k = 2*(x + y) ∧ p.t = x*y) ↔ 
  is_valid_rectangle_point p :=
sorry

end NUMINAMATH_CALUDE_rectangle_point_characterization_l3644_364418


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3644_364430

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x > 3 ∧ y > 3 → x + y > 6) ∧
  (∃ x y : ℝ, x + y > 6 ∧ ¬(x > 3 ∧ y > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3644_364430


namespace NUMINAMATH_CALUDE_total_sandwiches_l3644_364448

/-- Represents the number of sandwiches of each type -/
structure Sandwiches where
  cheese : ℕ
  bologna : ℕ
  peanutButter : ℕ

/-- The ratio of sandwich types -/
def sandwichRatio : Sandwiches :=
  { cheese := 1
    bologna := 7
    peanutButter := 8 }

/-- Theorem: Given the sandwich ratio and the number of bologna sandwiches,
    prove the total number of sandwiches -/
theorem total_sandwiches
    (ratio : Sandwiches)
    (bologna_count : ℕ)
    (h1 : ratio = sandwichRatio)
    (h2 : bologna_count = 35) :
    ratio.cheese * (bologna_count / ratio.bologna) +
    bologna_count +
    ratio.peanutButter * (bologna_count / ratio.bologna) = 80 := by
  sorry

#check total_sandwiches

end NUMINAMATH_CALUDE_total_sandwiches_l3644_364448


namespace NUMINAMATH_CALUDE_cafeteria_pies_l3644_364412

def number_of_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies : number_of_pies 51 41 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l3644_364412


namespace NUMINAMATH_CALUDE_longest_side_is_80_l3644_364485

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 120
  area_eq : length * width = 2400

/-- The longest side of a SpecialRectangle is 80 -/
theorem longest_side_is_80 (rect : SpecialRectangle) : 
  max rect.length rect.width = 80 := by
  sorry

#check longest_side_is_80

end NUMINAMATH_CALUDE_longest_side_is_80_l3644_364485


namespace NUMINAMATH_CALUDE_system_solution_l3644_364487

theorem system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (eq1 : 2*x₁ + x₂ + x₃ + x₄ + x₅ = 6)
  (eq2 : x₁ + 2*x₂ + x₃ + x₄ + x₅ = 12)
  (eq3 : x₁ + x₂ + 2*x₃ + x₄ + x₅ = 24)
  (eq4 : x₁ + x₂ + x₃ + 2*x₄ + x₅ = 48)
  (eq5 : x₁ + x₂ + x₃ + x₄ + 2*x₅ = 96) :
  3*x₄ + 2*x₅ = 181 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3644_364487


namespace NUMINAMATH_CALUDE_fraction_simplification_l3644_364491

theorem fraction_simplification : 
  ((2^1004)^2 - (2^1002)^2) / ((2^1003)^2 - (2^1001)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3644_364491


namespace NUMINAMATH_CALUDE_range_of_a_l3644_364446

/-- Proposition p: The equation x^2 + ax + 1 = 0 has solutions -/
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a*x + 1 = 0

/-- Proposition q: For all x ∈ ℝ, e^(2x) - 2e^x + a ≥ 0 always holds -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, Real.exp (2*x) - 2*(Real.exp x) + a ≥ 0

/-- The range of a given p ∧ q is true -/
theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ∈ Set.Ici (0 : ℝ) := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l3644_364446


namespace NUMINAMATH_CALUDE_y_minus_x_values_l3644_364455

theorem y_minus_x_values (x y : ℝ) 
  (h1 : |x + 1| = 3) 
  (h2 : |y| = 5) 
  (h3 : -y/x > 0) : 
  y - x = -7 ∨ y - x = 9 := by
  sorry

end NUMINAMATH_CALUDE_y_minus_x_values_l3644_364455


namespace NUMINAMATH_CALUDE_max_tan_b_in_triangle_l3644_364428

/-- Given a triangle ABC with AB = 25 and BC = 15, the maximum value of tan B is 4/3 -/
theorem max_tan_b_in_triangle (A B C : ℝ × ℝ) :
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = 25 →
  d B C = 15 →
  ∀ C' : ℝ × ℝ, d A C' ≥ d A C → d B C' = 15 →
  Real.tan (Real.arccos ((d A B)^2 + (d B C)^2 - (d A C)^2) / (2 * d A B * d B C)) ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_max_tan_b_in_triangle_l3644_364428


namespace NUMINAMATH_CALUDE_delta_4_zero_delta_3_nonzero_l3644_364411

def u (n : ℕ) : ℤ := n^3 + n

def delta_1 (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def delta (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0 => u
  | 1 => delta_1 u
  | k + 1 => delta_1 (delta k u)

theorem delta_4_zero_delta_3_nonzero :
  (∀ n, delta 4 u n = 0) ∧ (∃ n, delta 3 u n ≠ 0) := by sorry

end NUMINAMATH_CALUDE_delta_4_zero_delta_3_nonzero_l3644_364411


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l3644_364478

def Q (n : ℕ) : ℚ :=
  (3^(n-1) : ℚ) / ((3*n - 2 : ℕ).factorial * n.factorial)

theorem smallest_n_for_Q_less_than_threshold : 
  ∀ k : ℕ, k > 0 → Q k < 1/1500 ↔ k ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l3644_364478


namespace NUMINAMATH_CALUDE_cubic_roots_conditions_l3644_364495

theorem cubic_roots_conditions (a b : ℝ) 
  (h : ∀ x : ℝ, a * x^3 - x^2 + b * x - 1 = 0 → x > 0) : 
  0 < 3 * a * b ∧ 3 * a * b ≤ 1 ∧ b ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_conditions_l3644_364495


namespace NUMINAMATH_CALUDE_correct_equation_l3644_364474

theorem correct_equation (x y : ℝ) : 3 * x^2 * y - 4 * y * x^2 = -x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l3644_364474


namespace NUMINAMATH_CALUDE_polynomial_real_root_l3644_364408

-- Define the polynomial
def P (b x : ℝ) : ℝ := x^4 + b*x^3 - 3*x^2 + b*x + 1

-- State the theorem
theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, P b x = 0) ↔ b ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l3644_364408


namespace NUMINAMATH_CALUDE_grading_problem_solution_l3644_364470

/-- Represents the grading scenario of Teacher Wang --/
structure GradingScenario where
  initial_rate : ℕ            -- Initial grading rate (assignments per hour)
  new_rate : ℕ                -- New grading rate (assignments per hour)
  change_time : ℕ             -- Time at which the grading rate changed (in hours)
  time_saved : ℕ              -- Time saved due to rate change (in hours)
  total_assignments : ℕ       -- Total number of assignments in the batch

/-- Theorem stating the solution to the grading problem --/
theorem grading_problem_solution (scenario : GradingScenario) : 
  scenario.initial_rate = 6 →
  scenario.new_rate = 8 →
  scenario.change_time = 2 →
  scenario.time_saved = 3 →
  scenario.total_assignments = 84 :=
by sorry

end NUMINAMATH_CALUDE_grading_problem_solution_l3644_364470


namespace NUMINAMATH_CALUDE_trigonometric_inequalities_l3644_364497

theorem trigonometric_inequalities (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  (Real.sqrt (y * z / x) + Real.sqrt (z * x / y) + Real.sqrt (x * y / z) ≥ Real.sqrt 3) ∧
  (Real.sqrt (x * y / (z + x * y)) + Real.sqrt (y * z / (x + y * z)) + Real.sqrt (z * x / (y + z * x)) ≤ 3 / 2) ∧
  (x / (x + y * z) + y / (y + z * x) + z / (z + x * y) ≤ 9 / 4) ∧
  ((x - y * z) / (x + y * z) + (y - z * x) / (y + z * x) + (z - x * y) / (z + x * y) ≤ 3 / 2) ∧
  ((x - y * z) / (x + y * z) * (y - z * x) / (y + z * x) * (z - x * y) / (z + x * y) ≤ 1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequalities_l3644_364497


namespace NUMINAMATH_CALUDE_parabola_vertex_l3644_364471

/-- The vertex of the parabola y = x^2 - 1 has coordinates (0, -1) -/
theorem parabola_vertex : 
  let f : ℝ → ℝ := fun x ↦ x^2 - 1
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = -1 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3644_364471


namespace NUMINAMATH_CALUDE_equipment_prices_l3644_364483

theorem equipment_prices (price_A price_B : ℝ) 
  (h1 : price_A = price_B + 25)
  (h2 : 2000 / price_A = 2 * (750 / price_B)) :
  price_A = 100 ∧ price_B = 75 := by
  sorry

end NUMINAMATH_CALUDE_equipment_prices_l3644_364483


namespace NUMINAMATH_CALUDE_integral_curves_satisfy_differential_equation_l3644_364488

/-- The differential equation in terms of x, y, dx, and dy -/
def differential_equation (x y : ℝ) (dx dy : ℝ) : Prop :=
  x * dx + y * dy + (x * dy - y * dx) / (x^2 + y^2) = 0

/-- The integral curve equation -/
def integral_curve (x y : ℝ) (C : ℝ) : Prop :=
  (x^2 + y^2) / 2 - y * Real.arctan (x / y) = C

/-- Theorem stating that the integral_curve satisfies the differential_equation -/
theorem integral_curves_satisfy_differential_equation :
  ∀ (x y : ℝ) (C : ℝ),
    x^2 + y^2 > 0 →
    integral_curve x y C →
    ∃ (dx dy : ℝ), differential_equation x y dx dy :=
sorry

end NUMINAMATH_CALUDE_integral_curves_satisfy_differential_equation_l3644_364488


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3644_364429

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3644_364429


namespace NUMINAMATH_CALUDE_initial_channels_l3644_364462

theorem initial_channels (x : ℕ) : 
  x - 20 + 12 - 10 + 8 + 7 = 147 → x = 150 := by
  sorry

end NUMINAMATH_CALUDE_initial_channels_l3644_364462


namespace NUMINAMATH_CALUDE_smallest_root_between_3_and_4_l3644_364413

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x + 5 * Real.tan x

def is_smallest_positive_root (s : ℝ) : Prop :=
  s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem smallest_root_between_3_and_4 :
  ∃ s, is_smallest_positive_root s ∧ 3 ≤ s ∧ s < 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_between_3_and_4_l3644_364413


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l3644_364426

theorem circus_ticket_cost (cost_per_ticket : ℕ) (num_tickets : ℕ) (total_cost : ℕ) : 
  cost_per_ticket = 44 → num_tickets = 7 → total_cost = cost_per_ticket * num_tickets →
  total_cost = 308 := by
sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l3644_364426


namespace NUMINAMATH_CALUDE_unique_polynomial_function_l3644_364480

-- Define the polynomial function type
def PolynomialFunction (R : Type) [Ring R] := R → R

-- Define the degree of a polynomial function
def degree (f : PolynomialFunction ℝ) : ℕ := sorry

-- State the conditions for the polynomial function
def satisfiesConditions (f : PolynomialFunction ℝ) : Prop :=
  (degree f ≥ 2) ∧
  (∀ x : ℝ, f (x^2 + 1) = (f x)^2 + 1) ∧
  (∀ x : ℝ, f (x^2 + 1) = f (f x + 1))

-- Theorem statement
theorem unique_polynomial_function :
  ∃! f : PolynomialFunction ℝ, satisfiesConditions f :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_function_l3644_364480


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3644_364404

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + (5 + 1/2)x + 1/2 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := 1/2

theorem quadratic_discriminant :
  discriminant a b c = 81/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3644_364404


namespace NUMINAMATH_CALUDE_computer_off_time_l3644_364472

/-- Represents days of the week -/
inductive Day
  | Friday
  | Saturday

/-- Represents time of day in hours (0-23) -/
def Time := Fin 24

/-- Represents a specific moment (day and time) -/
structure Moment where
  day : Day
  time : Time

/-- Adds hours to a given moment, wrapping to the next day if necessary -/
def addHours (m : Moment) (h : Nat) : Moment :=
  let totalHours := m.time.val + h
  let newDay := if totalHours ≥ 24 then Day.Saturday else m.day
  let newTime := Fin.ofNat (totalHours % 24)
  { day := newDay, time := newTime }

theorem computer_off_time 
  (start : Moment) 
  (h : Nat) 
  (start_day : start.day = Day.Friday)
  (start_time : start.time = ⟨14, sorry⟩)
  (duration : h = 30) :
  addHours start h = { day := Day.Saturday, time := ⟨20, sorry⟩ } := by
  sorry

#check computer_off_time

end NUMINAMATH_CALUDE_computer_off_time_l3644_364472


namespace NUMINAMATH_CALUDE_problem_statement_l3644_364443

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ^ b = 343) (h5 : b ^ c = 10) (h6 : a ^ c = 7) : b ^ b = 1000 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3644_364443


namespace NUMINAMATH_CALUDE_grocer_average_sale_l3644_364421

/-- Given the sales figures for five months, prove that the average sale is 7800 --/
theorem grocer_average_sale
  (sale1 : ℕ) (sale2 : ℕ) (sale3 : ℕ) (sale4 : ℕ) (sale5 : ℕ)
  (h1 : sale1 = 5700)
  (h2 : sale2 = 8550)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 3850)
  (h5 : sale5 = 14045) :
  (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = 7800 := by
  sorry

#check grocer_average_sale

end NUMINAMATH_CALUDE_grocer_average_sale_l3644_364421


namespace NUMINAMATH_CALUDE_petrol_price_equation_l3644_364415

/-- The original price of petrol per gallon -/
def P : ℝ := 2.11

/-- The reduction rate in price -/
def reduction_rate : ℝ := 0.1

/-- The additional gallons that can be bought after price reduction -/
def additional_gallons : ℝ := 5

/-- The fixed amount of money spent -/
def fixed_amount : ℝ := 200

theorem petrol_price_equation :
  fixed_amount / ((1 - reduction_rate) * P) - fixed_amount / P = additional_gallons := by
sorry

end NUMINAMATH_CALUDE_petrol_price_equation_l3644_364415


namespace NUMINAMATH_CALUDE_complex_square_condition_l3644_364467

theorem complex_square_condition (a b : ℝ) : 
  (∃ a b : ℝ, (Complex.I : ℂ)^2 = -1 ∧ (a + b * Complex.I)^2 = 2 * Complex.I ∧ ¬(a = 1 ∧ b = 1)) ∧
  ((a = 1 ∧ b = 1) → (a + b * Complex.I)^2 = 2 * Complex.I) :=
sorry

end NUMINAMATH_CALUDE_complex_square_condition_l3644_364467


namespace NUMINAMATH_CALUDE_sale_price_calculation_l3644_364401

def ticket_price : ℝ := 25
def discount_rate : ℝ := 0.25

theorem sale_price_calculation :
  ticket_price * (1 - discount_rate) = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l3644_364401
