import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_plus_exponent_l2518_251854

theorem absolute_value_plus_exponent : |-8| + 3^0 = 9 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_exponent_l2518_251854


namespace NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l2518_251850

open Real

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := log x / log 10

-- State the theorem
theorem order_of_logarithmic_expressions (a b : ℝ) (h1 : a > b) (h2 : b > 1) :
  sqrt (lg a * lg b) < (lg a + lg b) / 2 ∧ (lg a + lg b) / 2 < lg ((a + b) / 2) :=
by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l2518_251850


namespace NUMINAMATH_CALUDE_whitney_bookmarks_l2518_251893

/-- Proves that Whitney bought 2 bookmarks given the conditions of the problem --/
theorem whitney_bookmarks :
  ∀ (initial_amount : ℕ) 
    (poster_cost notebook_cost bookmark_cost : ℕ)
    (posters_bought notebooks_bought : ℕ)
    (amount_left : ℕ),
  initial_amount = 2 * 20 →
  poster_cost = 5 →
  notebook_cost = 4 →
  bookmark_cost = 2 →
  posters_bought = 2 →
  notebooks_bought = 3 →
  amount_left = 14 →
  ∃ (bookmarks_bought : ℕ),
    initial_amount = 
      poster_cost * posters_bought + 
      notebook_cost * notebooks_bought + 
      bookmark_cost * bookmarks_bought + 
      amount_left ∧
    bookmarks_bought = 2 :=
by sorry

end NUMINAMATH_CALUDE_whitney_bookmarks_l2518_251893


namespace NUMINAMATH_CALUDE_angle_between_polar_lines_theorem_l2518_251829

/-- The angle between two lines in polar coordinates -/
def angle_between_polar_lines (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- First line in polar coordinates: ρ(2cosθ + sinθ) = 2 -/
def line1 (ρ θ : ℝ) : Prop :=
  ρ * (2 * Real.cos θ + Real.sin θ) = 2

/-- Second line in polar coordinates: ρcosθ = 1 -/
def line2 (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = 1

/-- Theorem stating the angle between the two lines -/
theorem angle_between_polar_lines_theorem :
  angle_between_polar_lines line1 line2 = Real.arctan (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_angle_between_polar_lines_theorem_l2518_251829


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2518_251815

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > x) ↔ (∃ x : ℝ, x^2 ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2518_251815


namespace NUMINAMATH_CALUDE_finance_marketing_specialization_contradiction_l2518_251823

theorem finance_marketing_specialization_contradiction 
  (finance_percent1 : ℝ) 
  (finance_percent2 : ℝ) 
  (marketing_percent : ℝ) 
  (h1 : finance_percent1 = 88) 
  (h2 : marketing_percent = 76) 
  (h3 : finance_percent2 = 90) 
  (h4 : 0 ≤ finance_percent1 ∧ finance_percent1 ≤ 100) 
  (h5 : 0 ≤ finance_percent2 ∧ finance_percent2 ≤ 100) 
  (h6 : 0 ≤ marketing_percent ∧ marketing_percent ≤ 100) :
  finance_percent1 ≠ finance_percent2 := by
  sorry

end NUMINAMATH_CALUDE_finance_marketing_specialization_contradiction_l2518_251823


namespace NUMINAMATH_CALUDE_inequality_proof_l2518_251834

theorem inequality_proof (a b c : ℝ) : a^2 + b^2 + c^2 + 4 ≥ a*b + 3*b + 2*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2518_251834


namespace NUMINAMATH_CALUDE_square_of_negative_product_l2518_251801

theorem square_of_negative_product (m n : ℝ) : (-2 * m * n)^2 = 4 * m^2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l2518_251801


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2518_251864

theorem solve_exponential_equation : ∃ x : ℝ, (1000 : ℝ)^5 = 40^x ∧ x = 15 := by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2518_251864


namespace NUMINAMATH_CALUDE_shelf_capacity_l2518_251845

/-- The number of CDs each rack can hold -/
def cds_per_rack : ℕ := 8

/-- The total number of CDs the shelf can hold -/
def total_cds : ℕ := 32

/-- The number of racks the shelf can hold -/
def num_racks : ℕ := total_cds / cds_per_rack

theorem shelf_capacity : num_racks = 4 := by
  sorry

end NUMINAMATH_CALUDE_shelf_capacity_l2518_251845


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2518_251809

/-- The quadratic equation (k-1)x^2 + 3x - 1 = 0 has real roots if and only if k ≥ -5/4 and k ≠ 1 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -5/4 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2518_251809


namespace NUMINAMATH_CALUDE_root_ratio_equality_l2518_251890

theorem root_ratio_equality (a : ℝ) (h_pos : a > 0) : 
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ 
   x₁^3 + 1 = a*x₁ ∧ x₂^3 + 1 = a*x₂ ∧
   x₂ / x₁ = 2018 ∧
   (∀ x : ℝ, x^3 + 1 = a*x → x = x₁ ∨ x = x₂ ∨ x ≤ 0)) →
  (∃ y₁ y₂ : ℝ, 0 < y₁ ∧ y₁ < y₂ ∧ 
   y₁^3 + 1 = a*y₁^2 ∧ y₂^3 + 1 = a*y₂^2 ∧
   y₂ / y₁ = 2018 ∧
   (∀ y : ℝ, y^3 + 1 = a*y^2 → y = y₁ ∨ y = y₂ ∨ y ≤ 0)) := by
sorry

end NUMINAMATH_CALUDE_root_ratio_equality_l2518_251890


namespace NUMINAMATH_CALUDE_bob_cereal_difference_l2518_251828

/-- Represents the number of sides on Bob's die -/
def dieSides : ℕ := 8

/-- Represents the threshold for eating organic cereal -/
def organicThreshold : ℕ := 5

/-- Represents the number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- Probability of eating organic cereal -/
def probOrganic : ℚ := 4 / 7

/-- Probability of eating gluten-free cereal -/
def probGlutenFree : ℚ := 3 / 7

/-- Expected difference in days between eating organic and gluten-free cereal -/
def expectedDifference : ℚ := daysInYear * (probOrganic - probGlutenFree)

theorem bob_cereal_difference :
  expectedDifference = 365 * (4/7 - 3/7) :=
sorry

end NUMINAMATH_CALUDE_bob_cereal_difference_l2518_251828


namespace NUMINAMATH_CALUDE_add_decimals_l2518_251852

theorem add_decimals : (124.75 : ℝ) + 0.35 = 125.10 := by sorry

end NUMINAMATH_CALUDE_add_decimals_l2518_251852


namespace NUMINAMATH_CALUDE_smallest_multiple_of_112_l2518_251811

theorem smallest_multiple_of_112 (n : ℕ) : (n * 14 % 112 = 0 ∧ n > 0) → n ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_112_l2518_251811


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_squares_l2518_251804

theorem product_of_difference_and_sum_squares (a b : ℝ) 
  (h1 : a - b = 6) 
  (h2 : a^2 + b^2 = 48) : 
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_squares_l2518_251804


namespace NUMINAMATH_CALUDE_not_p_and_p_or_q_implies_q_l2518_251877

theorem not_p_and_p_or_q_implies_q (p q : Prop) : (¬p ∧ (p ∨ q)) → q := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_p_or_q_implies_q_l2518_251877


namespace NUMINAMATH_CALUDE_distance_AB_is_420_main_theorem_l2518_251891

/-- Represents a person with a speed --/
structure Person where
  speed : ℝ

/-- Represents the problem setup --/
structure ProblemSetup where
  distance_AB : ℝ
  person_A : Person
  person_B : Person
  meeting_point : ℝ
  B_remaining_distance : ℝ

/-- The theorem statement --/
theorem distance_AB_is_420 (setup : ProblemSetup) : setup.distance_AB = 420 :=
  by
  have h1 : setup.person_A.speed > setup.person_B.speed := sorry
  have h2 : setup.meeting_point = setup.distance_AB - 240 := sorry
  have h3 : setup.B_remaining_distance = 120 := sorry
  have h4 : 2 * setup.person_A.speed > 2 * setup.person_B.speed := sorry
  sorry

/-- The main theorem --/
theorem main_theorem : ∃ (setup : ProblemSetup), setup.distance_AB = 420 :=
  by sorry

end NUMINAMATH_CALUDE_distance_AB_is_420_main_theorem_l2518_251891


namespace NUMINAMATH_CALUDE_lucy_moved_fish_l2518_251808

/-- The number of fish moved to a different tank -/
def fish_moved (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

/-- Proof that Lucy moved 68 fish to a different tank -/
theorem lucy_moved_fish : fish_moved 212 144 = 68 := by
  sorry

end NUMINAMATH_CALUDE_lucy_moved_fish_l2518_251808


namespace NUMINAMATH_CALUDE_square_of_binomial_l2518_251858

/-- If ax^2 + 18x + 16 is the square of a binomial, then a = 81/16 -/
theorem square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x : ℚ, a * x^2 + 18 * x + 16 = (r * x + s)^2) → 
  a = 81 / 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2518_251858


namespace NUMINAMATH_CALUDE_investment_worth_l2518_251825

def investment_problem (initial_investment : ℚ) (months : ℕ) (monthly_earnings : ℚ) : Prop :=
  let total_earnings := monthly_earnings * months
  let current_worth := initial_investment + total_earnings
  (months = 5) ∧
  (monthly_earnings = 12) ∧
  (total_earnings = 2 * initial_investment) ∧
  (current_worth = 90)

theorem investment_worth :
  ∃ (initial_investment : ℚ), investment_problem initial_investment 5 12 :=
by sorry

end NUMINAMATH_CALUDE_investment_worth_l2518_251825


namespace NUMINAMATH_CALUDE_problem_solution_l2518_251870

/-- The condition p: x^2 - 5ax + 4a^2 < 0 -/
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0

/-- The condition q: 3 < x ≤ 4 -/
def q (x : ℝ) : Prop := 3 < x ∧ x ≤ 4

theorem problem_solution (a : ℝ) (h : a > 0) :
  (a = 1 → ∀ x, p x a ∧ q x ↔ 3 < x ∧ x < 4) ∧
  (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x) ↔ 1 < a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2518_251870


namespace NUMINAMATH_CALUDE_tom_catches_sixteen_trout_l2518_251846

/-- The number of trout Melanie catches -/
def melanie_trout : ℕ := 8

/-- Tom catches twice as many trout as Melanie -/
def tom_multiplier : ℕ := 2

/-- The number of trout Tom catches -/
def tom_trout : ℕ := tom_multiplier * melanie_trout

theorem tom_catches_sixteen_trout : tom_trout = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_catches_sixteen_trout_l2518_251846


namespace NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l2518_251822

/-- Given a square with side length s ≥ 4 containing a 2x2 square, 
    a 2x4 rectangle, and a non-overlapping rectangle R, 
    the area of R is exactly 4. -/
theorem area_of_inscribed_rectangle (s : ℝ) (h_s : s ≥ 4) : 
  s^2 - (2 * 2 + 2 * 4) = 4 := by sorry

end NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l2518_251822


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l2518_251812

/-- If the terminal side of angle α passes through the point (-4, 3), then sin α = 3/5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = -4 ∧ r * Real.sin α = 3) → 
  Real.sin α = 3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l2518_251812


namespace NUMINAMATH_CALUDE_set_operation_proof_l2518_251897

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, 1, 2}

theorem set_operation_proof :
  A ∪ (U \ B) = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_operation_proof_l2518_251897


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l2518_251806

/-- Given a function f(x) = a*tan³(x) - b*sin(3x) + cx + 7 where f(1) = 14, 
    prove that f(-1) = 0 -/
theorem function_value_at_negative_one 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * (Real.tan x)^3 - b * Real.sin (3 * x) + c * x + 7)
  (h2 : f 1 = 14) : 
  f (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l2518_251806


namespace NUMINAMATH_CALUDE_average_age_combined_l2518_251873

theorem average_age_combined (n_students : ℕ) (n_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  n_students = 40 →
  n_parents = 50 →
  avg_age_students = 13 →
  avg_age_parents = 40 →
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents : ℚ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l2518_251873


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2518_251878

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0006 + 0.00007 = 23467 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2518_251878


namespace NUMINAMATH_CALUDE_true_masses_l2518_251857

/-- Represents the uneven lever scale with a linear relationship between left and right sides -/
structure UnevenLeverScale where
  k : ℝ
  b : ℝ
  left_to_right : ℝ → ℝ
  right_to_left : ℝ → ℝ
  hk_pos : k > 0
  hleft_to_right : left_to_right = fun x => k * x + b
  hright_to_left : right_to_left = fun y => (y - b) / k

/-- The equilibrium conditions observed on the uneven lever scale -/
structure EquilibriumConditions (scale : UnevenLeverScale) where
  melon_right : scale.left_to_right 3 = scale.right_to_left 5.5
  melon_left : scale.right_to_left 5.5 = scale.left_to_right 3
  watermelon_right : scale.left_to_right 5 = scale.right_to_left 10
  watermelon_left : scale.right_to_left 10 = scale.left_to_right 5

/-- The theorem stating the true masses of the melon and watermelon -/
theorem true_masses (scale : UnevenLeverScale) (conditions : EquilibriumConditions scale) :
  ∃ (melon_mass watermelon_mass : ℝ),
    melon_mass = 5.5 ∧
    watermelon_mass = 10 ∧
    scale.left_to_right 3 = melon_mass ∧
    scale.right_to_left 5.5 = melon_mass ∧
    scale.left_to_right 5 = watermelon_mass ∧
    scale.right_to_left 10 = watermelon_mass := by
  sorry

end NUMINAMATH_CALUDE_true_masses_l2518_251857


namespace NUMINAMATH_CALUDE_hayden_pants_ironing_time_l2518_251889

/-- Represents the ironing routine of Hayden --/
structure IroningRoutine where
  shirt_time : ℕ  -- Time spent ironing shirt per day (in minutes)
  days_per_week : ℕ  -- Number of days Hayden irons per week
  total_time : ℕ  -- Total time spent ironing over 4 weeks (in minutes)

/-- Calculates the time spent ironing pants per day --/
def pants_ironing_time (routine : IroningRoutine) : ℕ :=
  let total_per_week := routine.total_time / 4
  let shirt_per_week := routine.shirt_time * routine.days_per_week
  let pants_per_week := total_per_week - shirt_per_week
  pants_per_week / routine.days_per_week

/-- Theorem stating that Hayden spends 3 minutes ironing his pants each day --/
theorem hayden_pants_ironing_time :
  pants_ironing_time ⟨5, 5, 160⟩ = 3 := by
  sorry


end NUMINAMATH_CALUDE_hayden_pants_ironing_time_l2518_251889


namespace NUMINAMATH_CALUDE_purely_imaginary_number_l2518_251876

theorem purely_imaginary_number (k : ℝ) : 
  (∃ (z : ℂ), z = (2 * k^2 - 3 * k - 2 : ℝ) + (k^2 - 2 * k : ℝ) * I ∧ z.re = 0 ∧ z.im ≠ 0) → 
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_number_l2518_251876


namespace NUMINAMATH_CALUDE_units_digit_of_power_sum_divided_l2518_251840

/-- The units digit of (4^503 + 6^503) / 10 is 1 -/
theorem units_digit_of_power_sum_divided : ∃ n : ℕ, (4^503 + 6^503) / 10 = 10 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_sum_divided_l2518_251840


namespace NUMINAMATH_CALUDE_subset_condition_implies_p_range_l2518_251885

open Set

theorem subset_condition_implies_p_range (p : ℝ) : 
  let A : Set ℝ := {x | 4 * x + p < 0}
  let B : Set ℝ := {x | x < -1 ∨ x > 2}
  A.Nonempty → B.Nonempty → A ⊆ B → p ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_subset_condition_implies_p_range_l2518_251885


namespace NUMINAMATH_CALUDE_hexagon_and_circle_construction_l2518_251848

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Constructs a hexagon from three non-adjacent vertex projections -/
def construct_hexagon (p1 p2 p3 : Point) : Hexagon :=
  sorry

/-- Constructs an inscribed circle for a given hexagon -/
def construct_inscribed_circle (h : Hexagon) : Circle :=
  sorry

theorem hexagon_and_circle_construction 
  (p1 p2 p3 : Point) 
  (h_not_collinear : ¬ are_collinear p1 p2 p3) :
  ∃ (hex : Hexagon) (circ : Circle), 
    hex = construct_hexagon p1 p2 p3 ∧ 
    circ = construct_inscribed_circle hex :=
  by sorry

end NUMINAMATH_CALUDE_hexagon_and_circle_construction_l2518_251848


namespace NUMINAMATH_CALUDE_all_non_negative_l2518_251859

theorem all_non_negative (a b c d : ℤ) (h : (2 : ℝ)^a + (2 : ℝ)^b = (3 : ℝ)^c + (3 : ℝ)^d) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_all_non_negative_l2518_251859


namespace NUMINAMATH_CALUDE_no_valid_covering_for_6_and_7_l2518_251805

/-- Represents the L-shaped or T-shaped 4-cell figure -/
inductive TetrominoShape
| L
| T

/-- Represents a position on the n×n square -/
structure Position (n : ℕ) where
  x : Fin n
  y : Fin n

/-- Represents a tetromino (4-cell figure) placement on the square -/
structure TetrominoPlacement (n : ℕ) where
  shape : TetrominoShape
  position : Position n
  rotation : Fin 4  -- 0, 90, 180, or 270 degrees

/-- Checks if a tetromino placement is valid within the n×n square -/
def is_valid_placement (n : ℕ) (placement : TetrominoPlacement n) : Prop := sorry

/-- Checks if a set of tetromino placements covers the entire n×n square exactly once -/
def covers_square_once (n : ℕ) (placements : List (TetrominoPlacement n)) : Prop := sorry

theorem no_valid_covering_for_6_and_7 :
  ¬ (∃ (placements : List (TetrominoPlacement 6)), covers_square_once 6 placements) ∧
  ¬ (∃ (placements : List (TetrominoPlacement 7)), covers_square_once 7 placements) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_covering_for_6_and_7_l2518_251805


namespace NUMINAMATH_CALUDE_chocolate_milk_syrup_amount_l2518_251832

/-- Proves that the amount of chocolate syrup in each glass is 1.5 ounces -/
theorem chocolate_milk_syrup_amount :
  let glass_size : ℝ := 8
  let milk_per_glass : ℝ := 6.5
  let total_milk : ℝ := 130
  let total_syrup : ℝ := 60
  let total_mixture : ℝ := 160
  ∃ (num_glasses : ℕ) (syrup_per_glass : ℝ),
    (↑num_glasses : ℝ) * glass_size = total_mixture ∧
    (↑num_glasses : ℝ) * milk_per_glass = total_milk ∧
    (↑num_glasses : ℝ) * syrup_per_glass ≤ total_syrup ∧
    glass_size = milk_per_glass + syrup_per_glass ∧
    syrup_per_glass = 1.5 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_syrup_amount_l2518_251832


namespace NUMINAMATH_CALUDE_number_division_problem_l2518_251871

theorem number_division_problem (x : ℝ) : (x - 5) / 7 = 7 → (x - 2) / 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2518_251871


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2518_251861

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_abc_properties (t : Triangle) 
  (h_acute : 0 < t.C ∧ t.C < Real.pi / 2)
  (h_sine_relation : Real.sqrt 15 * t.a * Real.sin t.A = t.b * Real.sin t.B * Real.sin t.C)
  (h_b_twice_a : t.b = 2 * t.a)
  (h_a_c_sum : t.a + t.c = 6) :
  Real.tan t.C = Real.sqrt 15 ∧ 
  (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2518_251861


namespace NUMINAMATH_CALUDE_sales_composition_l2518_251896

/-- The percentage of sales that are not pens, pencils, or erasers -/
def other_sales_percentage (pen_sales pencil_sales eraser_sales : ℝ) : ℝ :=
  100 - (pen_sales + pencil_sales + eraser_sales)

/-- Theorem stating that the percentage of sales not consisting of pens, pencils, or erasers is 25% -/
theorem sales_composition 
  (pen_sales : ℝ) 
  (pencil_sales : ℝ) 
  (eraser_sales : ℝ) 
  (h1 : pen_sales = 25)
  (h2 : pencil_sales = 30)
  (h3 : eraser_sales = 20) :
  other_sales_percentage pen_sales pencil_sales eraser_sales = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_composition_l2518_251896


namespace NUMINAMATH_CALUDE_car_expense_difference_l2518_251841

/-- Calculates the difference between Alberto's and Samara's car expenses -/
theorem car_expense_difference : 
  let alberto_engine : ℚ := 2457
  let alberto_transmission : ℚ := 374
  let alberto_tires : ℚ := 520
  let alberto_battery : ℚ := 129
  let alberto_exhaust : ℚ := 799
  let alberto_exhaust_discount : ℚ := 0.05
  let alberto_loyalty_discount : ℚ := 0.07
  let samara_oil : ℚ := 25
  let samara_tires : ℚ := 467
  let samara_detailing : ℚ := 79
  let samara_brake_pads : ℚ := 175
  let samara_paint : ℚ := 599
  let samara_stereo : ℚ := 225
  let samara_sales_tax : ℚ := 0.06

  let alberto_total := alberto_engine + alberto_transmission + alberto_tires + alberto_battery + 
                       (alberto_exhaust * (1 - alberto_exhaust_discount))
  let alberto_final := alberto_total * (1 - alberto_loyalty_discount)
  
  let samara_total := samara_oil + samara_tires + samara_detailing + samara_brake_pads + 
                      samara_paint + samara_stereo
  let samara_final := samara_total * (1 + samara_sales_tax)

  alberto_final - samara_final = 2278.12 := by sorry

end NUMINAMATH_CALUDE_car_expense_difference_l2518_251841


namespace NUMINAMATH_CALUDE_same_color_pair_count_l2518_251831

/-- The number of ways to choose a pair of socks of the same color -/
def choose_same_color_pair (green red purple : ℕ) : ℕ :=
  Nat.choose green 2 + Nat.choose red 2 + Nat.choose purple 2

/-- Theorem stating that choosing a pair of socks of the same color from 
    5 green, 6 red, and 4 purple socks results in 31 possibilities -/
theorem same_color_pair_count : choose_same_color_pair 5 6 4 = 31 := by
  sorry

end NUMINAMATH_CALUDE_same_color_pair_count_l2518_251831


namespace NUMINAMATH_CALUDE_hyunseung_outfit_combinations_l2518_251803

/-- The number of types of tops in Hyunseung's closet -/
def num_tops : ℕ := 3

/-- The number of types of bottoms in Hyunseung's closet -/
def num_bottoms : ℕ := 2

/-- The number of types of shoes in Hyunseung's closet -/
def num_shoes : ℕ := 5

/-- The total number of combinations of tops, bottoms, and shoes Hyunseung can wear -/
def total_combinations : ℕ := num_tops * num_bottoms * num_shoes

theorem hyunseung_outfit_combinations : total_combinations = 30 := by
  sorry

end NUMINAMATH_CALUDE_hyunseung_outfit_combinations_l2518_251803


namespace NUMINAMATH_CALUDE_distance_between_cities_l2518_251814

def train_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem distance_between_cities (D : ℝ) : D = 330 :=
  let train1_speed : ℝ := 60
  let train1_time : ℝ := 3
  let train2_speed : ℝ := 75
  let train2_time : ℝ := 2
  let train1_distance := train_distance train1_speed train1_time
  let train2_distance := train_distance train2_speed train2_time
  have h1 : D = train1_distance + train2_distance := by sorry
  have h2 : train1_distance = 180 := by sorry
  have h3 : train2_distance = 150 := by sorry
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l2518_251814


namespace NUMINAMATH_CALUDE_stating_prob_three_students_same_group_l2518_251807

/-- Represents the total number of students -/
def total_students : ℕ := 800

/-- Represents the number of lunch groups -/
def num_groups : ℕ := 4

/-- Represents the size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- Represents the probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- 
Theorem stating that the probability of three specific students 
being assigned to the same lunch group is 1/16
-/
theorem prob_three_students_same_group : 
  (prob_assigned_to_group * prob_assigned_to_group : ℚ) = 1 / 16 := by
  sorry

#check prob_three_students_same_group

end NUMINAMATH_CALUDE_stating_prob_three_students_same_group_l2518_251807


namespace NUMINAMATH_CALUDE_escalator_theorem_l2518_251839

def escalator_problem (stationary_time walking_time : ℝ) : Prop :=
  let s := 1 / stationary_time -- Clea's walking speed
  let d := 1 -- normalized distance of the escalator
  let v := d / walking_time - s -- speed of the escalator
  (d / v) = 50

theorem escalator_theorem : 
  escalator_problem 75 30 := by sorry

end NUMINAMATH_CALUDE_escalator_theorem_l2518_251839


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l2518_251887

/-- The speed of a man rowing a boat in still water, given the speed of the stream
    and the time taken to row a certain distance downstream. -/
theorem mans_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 8)
  (h2 : downstream_distance = 90)
  (h3 : downstream_time = 5)
  : ∃ (mans_speed : ℝ), mans_speed = 10 ∧ 
    (mans_speed + stream_speed) * downstream_time = downstream_distance :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l2518_251887


namespace NUMINAMATH_CALUDE_eight_friends_receive_necklace_l2518_251836

/-- The number of friends receiving a candy necklace -/
def friends_receiving_necklace (pieces_per_necklace : ℕ) (pieces_per_block : ℕ) (blocks_used : ℕ) : ℕ :=
  (blocks_used * pieces_per_block) / pieces_per_necklace - 1

/-- Theorem: Given the conditions, prove that 8 friends receive a candy necklace -/
theorem eight_friends_receive_necklace :
  friends_receiving_necklace 10 30 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_friends_receive_necklace_l2518_251836


namespace NUMINAMATH_CALUDE_age_difference_l2518_251820

/-- Proves that the age difference between a man and his son is 24 years -/
theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 22 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 24 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2518_251820


namespace NUMINAMATH_CALUDE_point_distance_from_x_axis_l2518_251880

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ := |p.y|

theorem point_distance_from_x_axis (a : ℝ) :
  let p : Point := ⟨2, a⟩
  distanceFromXAxis p = 3 → a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_from_x_axis_l2518_251880


namespace NUMINAMATH_CALUDE_dam_building_time_with_reduced_workers_l2518_251895

/-- The time taken to build a dam given the number of workers and their work rate -/
def build_time (workers : ℕ) (rate : ℚ) : ℚ :=
  1 / (workers * rate)

/-- The work rate of a single worker -/
def worker_rate (initial_workers : ℕ) (initial_time : ℚ) : ℚ :=
  1 / (initial_workers * initial_time)

theorem dam_building_time_with_reduced_workers 
  (initial_workers : ℕ) 
  (initial_time : ℚ) 
  (new_workers : ℕ) : 
  initial_workers = 60 → 
  initial_time = 5 → 
  new_workers = 40 → 
  build_time new_workers (worker_rate initial_workers initial_time) = 7.5 := by
sorry

end NUMINAMATH_CALUDE_dam_building_time_with_reduced_workers_l2518_251895


namespace NUMINAMATH_CALUDE_a_less_than_sqrt3b_l2518_251824

theorem a_less_than_sqrt3b (a b : ℤ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : (a + b) ∣ (a * b + 1)) 
  (h4 : (a - b) ∣ (a * b - 1)) : 
  a < Real.sqrt 3 * b := by
sorry

end NUMINAMATH_CALUDE_a_less_than_sqrt3b_l2518_251824


namespace NUMINAMATH_CALUDE_geometric_sum_five_quarters_l2518_251802

def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_five_quarters :
  geometric_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_five_quarters_l2518_251802


namespace NUMINAMATH_CALUDE_canoe_capacity_fraction_l2518_251872

/-- The fraction of people that can fit in a canoe with a dog compared to without --/
theorem canoe_capacity_fraction :
  let max_people : ℕ := 6  -- Maximum number of people without dog
  let person_weight : ℕ := 140  -- Weight of each person in pounds
  let dog_weight : ℕ := person_weight / 4  -- Weight of the dog
  let total_weight : ℕ := 595  -- Total weight with dog and people
  let people_with_dog : ℕ := (total_weight - dog_weight) / person_weight  -- Number of people with dog
  (people_with_dog : ℚ) / max_people = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_canoe_capacity_fraction_l2518_251872


namespace NUMINAMATH_CALUDE_jerry_skit_first_character_lines_l2518_251862

/-- Represents the number of lines for each character in Jerry's skit script -/
structure SkitScript where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of Jerry's skit script -/
def validScript (s : SkitScript) : Prop :=
  s.first = s.second + 8 ∧
  s.third = 2 ∧
  s.second = 3 * s.third + 6

theorem jerry_skit_first_character_lines :
  ∀ s : SkitScript, validScript s → s.first = 20 := by
  sorry

#check jerry_skit_first_character_lines

end NUMINAMATH_CALUDE_jerry_skit_first_character_lines_l2518_251862


namespace NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l2518_251827

theorem subset_sum_divisible_by_2n (n : ℕ) (hn : n ≥ 4) 
  (S : Finset ℕ) (hS : S.card = n) (hS_subset : ∀ x ∈ S, x ∈ Finset.range (2*n)) :
  ∃ T : Finset ℕ, T ⊆ S ∧ (2*n) ∣ (T.sum id) :=
sorry

end NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l2518_251827


namespace NUMINAMATH_CALUDE_cube_arrangement_exists_l2518_251883

/-- Represents the arrangement of numbers on a cube's edges -/
def CubeArrangement := Fin 12 → Fin 12

/-- Checks if the given arrangement is valid (uses all numbers from 1 to 12 exactly once) -/
def is_valid_arrangement (arr : CubeArrangement) : Prop :=
  (∀ i : Fin 12, ∃ j : Fin 12, arr j = i) ∧ 
  (∀ i j : Fin 12, arr i = arr j → i = j)

/-- Returns the product of numbers on the top face -/
def top_face_product (arr : CubeArrangement) : ℕ :=
  (arr 0 + 1) * (arr 1 + 1) * (arr 2 + 1) * (arr 3 + 1)

/-- Returns the product of numbers on the bottom face -/
def bottom_face_product (arr : CubeArrangement) : ℕ :=
  (arr 4 + 1) * (arr 5 + 1) * (arr 6 + 1) * (arr 7 + 1)

/-- Theorem stating that there exists a valid arrangement with equal products on top and bottom faces -/
theorem cube_arrangement_exists : 
  ∃ (arr : CubeArrangement), 
    is_valid_arrangement arr ∧ 
    top_face_product arr = bottom_face_product arr :=
by sorry

end NUMINAMATH_CALUDE_cube_arrangement_exists_l2518_251883


namespace NUMINAMATH_CALUDE_winter_sales_l2518_251865

/-- Proves that the number of pastries sold in winter is 3 million -/
theorem winter_sales (spring summer fall : ℕ) (total : ℝ) : 
  spring = 3 → 
  summer = 6 → 
  fall = 3 → 
  fall = (1/5 : ℝ) * total → 
  total - (spring + summer + fall : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_winter_sales_l2518_251865


namespace NUMINAMATH_CALUDE_inequality_proof_l2518_251863

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2)/(x^2 + y*z) + (y^2 + 2*z^2 + 2*x^2)/(y^2 + z*x) + (z^2 + 2*x^2 + 2*y^2)/(z^2 + x*y) > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2518_251863


namespace NUMINAMATH_CALUDE_sphere_between_inclined_planes_l2518_251884

/-- The distance from the center of a sphere to the horizontal plane when placed between two inclined planes -/
theorem sphere_between_inclined_planes 
  (r : ℝ) 
  (angle1 : ℝ) 
  (angle2 : ℝ) 
  (h_r : r = 2) 
  (h_angle1 : angle1 = π / 3)  -- 60 degrees in radians
  (h_angle2 : angle2 = π / 6)  -- 30 degrees in radians
  : ∃ (d : ℝ), d = Real.sqrt 3 + 1 ∧ d = 
    r * Real.sin ((π / 2 - angle1 - angle2) / 2 + angle2) :=
by sorry

end NUMINAMATH_CALUDE_sphere_between_inclined_planes_l2518_251884


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l2518_251860

theorem mean_of_added_numbers (original_mean original_count new_mean new_count : ℝ) 
  (h1 : original_mean = 65)
  (h2 : original_count = 7)
  (h3 : new_mean = 80)
  (h4 : new_count = 10) :
  let added_count := new_count - original_count
  let added_sum := new_mean * new_count - original_mean * original_count
  added_sum / added_count = 115 := by
sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l2518_251860


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2518_251843

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 2 + a 3 = 6) : 
  3 * a 4 + a 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2518_251843


namespace NUMINAMATH_CALUDE_taehyung_average_problems_l2518_251821

/-- The average number of problems solved per day -/
def average_problems_per_day (total_problems : ℕ) (num_days : ℕ) : ℚ :=
  (total_problems : ℚ) / (num_days : ℚ)

/-- Theorem stating that the average number of problems solved per day is 23 -/
theorem taehyung_average_problems :
  average_problems_per_day 161 7 = 23 := by
  sorry

end NUMINAMATH_CALUDE_taehyung_average_problems_l2518_251821


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l2518_251847

/-- Given n = 3, prove that r = 177136, where r = 3^s - s and s = 2^n + n -/
theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n + n
  let r : ℕ := 3^s - s
  r = 177136 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l2518_251847


namespace NUMINAMATH_CALUDE_specific_combination_probability_is_one_eighth_l2518_251866

/-- A regular tetrahedron with numbers on its faces -/
structure NumberedTetrahedron :=
  (faces : Fin 4 → Fin 4)

/-- The probability of a specific face showing on a regular tetrahedron -/
def face_probability : ℚ := 1 / 4

/-- The number of ways to choose which tetrahedron shows a specific number -/
def ways_to_choose : ℕ := 2

/-- The probability of getting a specific combination of numbers when throwing two tetrahedra -/
def specific_combination_probability (t1 t2 : NumberedTetrahedron) : ℚ :=
  ↑ways_to_choose * face_probability * face_probability

theorem specific_combination_probability_is_one_eighth (t1 t2 : NumberedTetrahedron) :
  specific_combination_probability t1 t2 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_specific_combination_probability_is_one_eighth_l2518_251866


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_one_minus_i_l2518_251813

theorem complex_fraction_equals_neg_one_minus_i :
  let i : ℂ := Complex.I
  (1 + i)^3 / (1 - i)^2 = -1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_one_minus_i_l2518_251813


namespace NUMINAMATH_CALUDE_cannot_row_against_stream_l2518_251817

theorem cannot_row_against_stream (rate_still : ℝ) (speed_with_stream : ℝ) :
  rate_still = 1 →
  speed_with_stream = 6 →
  let stream_speed := speed_with_stream - rate_still
  stream_speed > rate_still →
  ¬∃ (speed_against_stream : ℝ), speed_against_stream > 0 ∧ speed_against_stream = rate_still - stream_speed :=
by
  sorry

end NUMINAMATH_CALUDE_cannot_row_against_stream_l2518_251817


namespace NUMINAMATH_CALUDE_double_burger_cost_l2518_251869

/-- The cost of a double burger given the total spent, number of burgers, single burger cost, and number of double burgers. -/
theorem double_burger_cost 
  (total_spent : ℚ) 
  (total_burgers : ℕ) 
  (single_burger_cost : ℚ) 
  (double_burger_count : ℕ) 
  (h1 : total_spent = 66.5)
  (h2 : total_burgers = 50)
  (h3 : single_burger_cost = 1)
  (h4 : double_burger_count = 33) :
  (total_spent - single_burger_cost * (total_burgers - double_burger_count)) / double_burger_count = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_double_burger_cost_l2518_251869


namespace NUMINAMATH_CALUDE_outfit_count_l2518_251892

/-- The number of outfits that can be made with different colored shirts and hats -/
def num_outfits (red_shirts green_shirts blue_shirts : ℕ) 
                (pants : ℕ) 
                (green_hats red_hats blue_hats : ℕ) : ℕ :=
  red_shirts * pants * (green_hats + blue_hats) +
  green_shirts * pants * (red_hats + blue_hats) +
  blue_shirts * pants * (green_hats + red_hats)

/-- Theorem stating the number of outfits under given conditions -/
theorem outfit_count : 
  num_outfits 7 6 5 6 6 7 5 = 1284 :=
sorry

end NUMINAMATH_CALUDE_outfit_count_l2518_251892


namespace NUMINAMATH_CALUDE_max_naive_number_with_divisible_ratio_l2518_251844

def is_naive_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 % 10 = n % 10 + 6) ∧
  (n / 100 % 10 = n / 10 % 10 + 2)

def P (n : ℕ) : ℕ :=
  3 * (n / 1000 % 10 + n / 100 % 10) + n / 10 % 10 + n % 10

def Q (n : ℕ) : ℕ :=
  n / 1000 % 10 - 5

theorem max_naive_number_with_divisible_ratio :
  ∃ (m : ℕ), is_naive_number m ∧ 
             (∀ n, is_naive_number n → P n / Q n % 10 = 0 → n ≤ m) ∧
             (P m / Q m % 10 = 0) ∧
             m = 9313 :=
sorry

end NUMINAMATH_CALUDE_max_naive_number_with_divisible_ratio_l2518_251844


namespace NUMINAMATH_CALUDE_total_cutlery_after_addition_l2518_251888

/-- Represents the number of each type of cutlery in a drawer -/
structure Cutlery :=
  (forks : ℕ)
  (knives : ℕ)
  (spoons : ℕ)
  (teaspoons : ℕ)

/-- Calculates the total number of cutlery pieces -/
def totalCutlery (c : Cutlery) : ℕ :=
  c.forks + c.knives + c.spoons + c.teaspoons

/-- Represents the initial state of the cutlery drawer -/
def initialCutlery : Cutlery :=
  { forks := 6
  , knives := 6 + 9
  , spoons := 2 * (6 + 9)
  , teaspoons := 6 / 2 }

/-- Represents the final state of the cutlery drawer after adding 2 of each type -/
def finalCutlery : Cutlery :=
  { forks := initialCutlery.forks + 2
  , knives := initialCutlery.knives + 2
  , spoons := initialCutlery.spoons + 2
  , teaspoons := initialCutlery.teaspoons + 2 }

/-- Theorem: The total number of cutlery pieces after adding 2 of each type is 62 -/
theorem total_cutlery_after_addition : totalCutlery finalCutlery = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_cutlery_after_addition_l2518_251888


namespace NUMINAMATH_CALUDE_problem_solution_l2518_251867

/-- Calculates the number of songs per album given the initial number of albums,
    the number of albums removed, and the total number of songs bought. -/
def songs_per_album (initial_albums : ℕ) (removed_albums : ℕ) (total_songs : ℕ) : ℕ :=
  total_songs / (initial_albums - removed_albums)

/-- Proves that given the specific conditions in the problem,
    the number of songs per album is 7. -/
theorem problem_solution :
  songs_per_album 8 2 42 = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2518_251867


namespace NUMINAMATH_CALUDE_certain_number_equation_l2518_251874

theorem certain_number_equation (x : ℝ) : 28 = (4/5) * x + 8 ↔ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2518_251874


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2518_251838

theorem polynomial_divisibility (a : ℤ) : ∃ k : ℤ, (3*a + 5)^2 - 4 = k * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2518_251838


namespace NUMINAMATH_CALUDE_sqrt_1575n_integer_exists_l2518_251851

theorem sqrt_1575n_integer_exists : ∃ n : ℕ+, ∃ k : ℕ, (k : ℝ) ^ 2 = 1575 * n := by
  sorry

end NUMINAMATH_CALUDE_sqrt_1575n_integer_exists_l2518_251851


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l2518_251886

theorem absolute_value_sum_zero (a b : ℝ) :
  |3 + a| + |b - 2| = 0 → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l2518_251886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2518_251818

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2518_251818


namespace NUMINAMATH_CALUDE_total_fleas_is_40_l2518_251899

/-- The number of fleas on Gertrude the chicken -/
def gertrudeFleas : ℕ := 10

/-- The number of fleas on Olive the chicken -/
def oliveFleas : ℕ := gertrudeFleas / 2

/-- The number of fleas on Maud the chicken -/
def maudFleas : ℕ := 5 * oliveFleas

/-- The total number of fleas on all three chickens -/
def totalFleas : ℕ := gertrudeFleas + oliveFleas + maudFleas

theorem total_fleas_is_40 : totalFleas = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_fleas_is_40_l2518_251899


namespace NUMINAMATH_CALUDE_money_left_after_trip_l2518_251849

def initial_savings : ℕ := 6000
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000

theorem money_left_after_trip : 
  initial_savings - (flight_cost + hotel_cost + food_cost) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_trip_l2518_251849


namespace NUMINAMATH_CALUDE_runner_a_race_time_l2518_251833

/-- Runner A in a race scenario --/
structure RunnerA where
  race_distance : ℝ
  head_start_distance : ℝ
  head_start_time : ℝ

/-- Theorem: Runner A completes the race in 200 seconds --/
theorem runner_a_race_time (a : RunnerA) 
  (h1 : a.race_distance = 1000)
  (h2 : a.head_start_distance = 50)
  (h3 : a.head_start_time = 10) : 
  a.race_distance / (a.head_start_distance / a.head_start_time) = 200 := by
  sorry

end NUMINAMATH_CALUDE_runner_a_race_time_l2518_251833


namespace NUMINAMATH_CALUDE_sum_of_qp_is_zero_l2518_251898

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := -abs x

def evaluation_points : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_qp_is_zero :
  (evaluation_points.map (λ x => q (p x))).sum = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_qp_is_zero_l2518_251898


namespace NUMINAMATH_CALUDE_π_approximation_relation_l2518_251835

/-- Approximate value of π obtained with an n-sided inscribed regular polygon -/
noncomputable def π_n (n : ℕ) : ℝ := sorry

/-- Theorem stating the relationship between π_2n and π_n -/
theorem π_approximation_relation (n : ℕ) :
  π_n (2 * n) = π_n n / Real.cos (π / n) := by sorry

end NUMINAMATH_CALUDE_π_approximation_relation_l2518_251835


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2518_251816

theorem polynomial_division_theorem (x : ℝ) :
  let dividend := x^5 - 20*x^3 + 15*x^2 - 18*x + 12
  let divisor := x - 2
  let quotient := x^4 + 2*x^3 - 16*x^2 - 17*x - 52
  let remainder := -92
  dividend = divisor * quotient + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2518_251816


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l2518_251855

def B : Set ℕ := {n | ∃ x : ℕ, x > 0 ∧ n = 4*x + 2}

theorem gcd_of_B_is_two : 
  ∃ (d : ℕ), d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ 
  (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l2518_251855


namespace NUMINAMATH_CALUDE_max_first_term_l2518_251882

/-- A sequence of positive real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, 0 < a n) ∧ 
  (∀ n, n > 0 → (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0) ∧
  (a 1 = a 10)

/-- The theorem stating the maximum possible value of the first term -/
theorem max_first_term (a : ℕ → ℝ) (h : SpecialSequence a) : 
  a 1 ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_max_first_term_l2518_251882


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2518_251842

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + x + 1 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2518_251842


namespace NUMINAMATH_CALUDE_total_oranges_picked_l2518_251819

theorem total_oranges_picked (del_per_day : ℕ) (del_days : ℕ) (juan_oranges : ℕ) :
  del_per_day = 23 →
  del_days = 2 →
  juan_oranges = 61 →
  del_per_day * del_days + juan_oranges = 107 :=
by
  sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l2518_251819


namespace NUMINAMATH_CALUDE_three_number_set_range_l2518_251830

theorem three_number_set_range (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ascending order
  a = 2 ∧  -- Smallest number is 2
  b = 5 ∧  -- Median is 5
  (a + b + c) / 3 = 5 →  -- Mean is 5
  c - a = 6 :=  -- Range is 6
by sorry

end NUMINAMATH_CALUDE_three_number_set_range_l2518_251830


namespace NUMINAMATH_CALUDE_abs_sum_fraction_inequality_l2518_251800

theorem abs_sum_fraction_inequality (a b : ℝ) :
  |a + b| / (1 + |a + b|) ≤ |a| / (1 + |a|) + |b| / (1 + |b|) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_fraction_inequality_l2518_251800


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2518_251856

theorem quadratic_roots_property (a : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + a*x₁ - 2 = 0) → 
  (x₂^2 + a*x₂ - 2 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^3 + 22/x₂ = x₂^3 + 22/x₁) →
  (a = 3 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2518_251856


namespace NUMINAMATH_CALUDE_relationship_abc_l2518_251853

theorem relationship_abc :
  let a : ℝ := Real.sqrt 5
  let b : ℝ := 2
  let c : ℝ := Real.sqrt 3
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2518_251853


namespace NUMINAMATH_CALUDE_sphere_carved_cube_surface_area_l2518_251875

theorem sphere_carved_cube_surface_area :
  let sphere_diameter : ℝ := Real.sqrt 3
  let cube_side_length : ℝ := 1
  let cube_diagonal : ℝ := cube_side_length * Real.sqrt 3
  cube_diagonal = sphere_diameter →
  (6 : ℝ) * cube_side_length ^ 2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_sphere_carved_cube_surface_area_l2518_251875


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_rectangular_solid_l2518_251879

/-- The surface area of a sphere containing a rectangular solid -/
theorem sphere_surface_area_with_rectangular_solid :
  ∀ (a b c : ℝ) (S : ℝ),
    a = 3 →
    b = 4 →
    c = 5 →
    S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
    S = 50 * Real.pi :=
by
  sorry

#check sphere_surface_area_with_rectangular_solid

end NUMINAMATH_CALUDE_sphere_surface_area_with_rectangular_solid_l2518_251879


namespace NUMINAMATH_CALUDE_expected_total_rainfall_l2518_251810

/-- Represents the weather conditions for each day -/
structure DailyWeather where
  sun_prob : Real
  rain_3in_prob : Real
  rain_8in_prob : Real

/-- Calculates the expected rainfall for a single day -/
def expected_daily_rainfall (w : DailyWeather) : Real :=
  w.sun_prob * 0 + w.rain_3in_prob * 3 + w.rain_8in_prob * 8

/-- The weather forecast for the week -/
def weather_forecast : DailyWeather :=
  { sun_prob := 0.3
  , rain_3in_prob := 0.4
  , rain_8in_prob := 0.3 }

/-- The number of days in the forecast -/
def num_days : Nat := 5

/-- Theorem: The expected total rainfall for the week is 18 inches -/
theorem expected_total_rainfall :
  (expected_daily_rainfall weather_forecast) * num_days = 18 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_rainfall_l2518_251810


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l2518_251826

/-- Theorem: Number of adult tickets sold in a movie theater --/
theorem adult_tickets_sold (adult_price child_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100) : 
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_revenue ∧ 
    adult_tickets = 500 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l2518_251826


namespace NUMINAMATH_CALUDE_sheilas_extra_flour_l2518_251868

/-- Given that Katie needs 3 pounds of flour and the total flour needed is 8 pounds,
    prove that Sheila needs 2 pounds more flour than Katie. -/
theorem sheilas_extra_flour (katie_flour sheila_flour total_flour : ℕ) : 
  katie_flour = 3 → 
  total_flour = 8 → 
  sheila_flour = total_flour - katie_flour →
  sheila_flour - katie_flour = 2 := by
  sorry

end NUMINAMATH_CALUDE_sheilas_extra_flour_l2518_251868


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2518_251881

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (x + 1)⁻¹ + y⁻¹ = (1 : ℝ) / 2) :
  ∀ a b : ℝ, a > 0 → b > 0 → (a + 1)⁻¹ + b⁻¹ = (1 : ℝ) / 2 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + 1)⁻¹ + y⁻¹ = (1 : ℝ) / 2 ∧ x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2518_251881


namespace NUMINAMATH_CALUDE_negation_of_p_l2518_251837

def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

theorem negation_of_p (f : ℝ → ℝ) :
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_p_l2518_251837


namespace NUMINAMATH_CALUDE_oplus_calculation_l2518_251894

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a * b + a + b + 1

-- State the theorem
theorem oplus_calculation : oplus (-3) (oplus 4 2) = -32 := by
  sorry

end NUMINAMATH_CALUDE_oplus_calculation_l2518_251894
