import Mathlib

namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l3724_372458

/-- Proves that given a book with a cost price CP, if selling it for 0.9 * CP
    results in Rs. 720, and selling it for Rs. 880 results in a gain,
    then the percentage of gain is 10%. -/
theorem book_sale_gain_percentage
  (CP : ℝ)  -- Cost price of the book
  (h1 : 0.9 * CP = 720)  -- Selling at 10% loss gives Rs. 720
  (h2 : 880 > CP)  -- Selling at Rs. 880 results in a gain
  : (880 - CP) / CP * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l3724_372458


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l3724_372465

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 42 = 0 ∧ x ≠ 6 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l3724_372465


namespace NUMINAMATH_CALUDE_prob_exactly_one_hit_prob_distribution_X_expected_value_X_l3724_372442

-- Define the probabilities and scores
def prob_A_hit : ℝ := 0.8
def prob_B_hit : ℝ := 0.5
def score_A_hit : ℕ := 5
def score_B_hit : ℕ := 10

-- Define the random variable X for the total score
def X : ℕ → ℝ
| 0 => (1 - prob_A_hit)^2 * (1 - prob_B_hit)
| 5 => 2 * prob_A_hit * (1 - prob_A_hit) * (1 - prob_B_hit)
| 10 => prob_A_hit^2 * (1 - prob_B_hit) + (1 - prob_A_hit)^2 * prob_B_hit
| 15 => 2 * prob_A_hit * (1 - prob_A_hit) * prob_B_hit
| 20 => prob_A_hit^2 * prob_B_hit
| _ => 0

-- Theorem for the probability of exactly one hit
theorem prob_exactly_one_hit : 
  2 * prob_A_hit * (1 - prob_A_hit) * (1 - prob_B_hit) + (1 - prob_A_hit)^2 * prob_B_hit = 0.18 := 
by sorry

-- Theorem for the probability distribution of X
theorem prob_distribution_X : 
  X 0 = 0.02 ∧ X 5 = 0.16 ∧ X 10 = 0.34 ∧ X 15 = 0.16 ∧ X 20 = 0.32 := 
by sorry

-- Theorem for the expected value of X
theorem expected_value_X : 
  0 * X 0 + 5 * X 5 + 10 * X 10 + 15 * X 15 + 20 * X 20 = 13.0 := 
by sorry

end NUMINAMATH_CALUDE_prob_exactly_one_hit_prob_distribution_X_expected_value_X_l3724_372442


namespace NUMINAMATH_CALUDE_solve_quadratic_sets_l3724_372489

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 8 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 - q*x + r = 0}

-- State the theorem
theorem solve_quadratic_sets :
  ∃ (p q r : ℝ),
    A p ≠ B q r ∧
    A p ∪ B q r = {-2, 4} ∧
    A p ∩ B q r = {-2} ∧
    p = -2 ∧ q = -4 ∧ r = 4 :=
by sorry

end NUMINAMATH_CALUDE_solve_quadratic_sets_l3724_372489


namespace NUMINAMATH_CALUDE_det_special_matrix_l3724_372406

-- Define the matrix as a function of y
def matrix (y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![y + 1, y, y],
    ![y, y + 1, y],
    ![y, y, y + 1]]

-- State the theorem
theorem det_special_matrix (y : ℝ) :
  Matrix.det (matrix y) = 3 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l3724_372406


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l3724_372429

theorem polynomial_functional_equation (p : ℝ → ℝ) :
  (∀ x : ℝ, p (x^3) - p (x^3 - 2) = (p x)^2 + 18) →
  (∃ a : ℝ, a^2 = 30 ∧ (∀ x : ℝ, p x = 6 * x^3 + a)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l3724_372429


namespace NUMINAMATH_CALUDE_equal_selection_probability_l3724_372440

/-- Represents a two-stage sampling process -/
structure TwoStageSampling where
  initial_count : ℕ
  excluded_count : ℕ
  selected_count : ℕ

/-- Calculates the probability of selection in a two-stage sampling process -/
def selection_probability (sampling : TwoStageSampling) : ℚ :=
  sampling.selected_count / (sampling.initial_count - sampling.excluded_count)

/-- Theorem stating that the selection probability is equal for all students and is 50/2000 -/
theorem equal_selection_probability (sampling : TwoStageSampling) 
  (h1 : sampling.initial_count = 2011)
  (h2 : sampling.excluded_count = 11)
  (h3 : sampling.selected_count = 50) :
  selection_probability sampling = 50 / 2000 := by
  sorry

#eval selection_probability ⟨2011, 11, 50⟩

end NUMINAMATH_CALUDE_equal_selection_probability_l3724_372440


namespace NUMINAMATH_CALUDE_melt_to_spend_ratio_is_80_l3724_372490

/-- The ratio of the value of melted quarters to spent quarters -/
def meltToSpendRatio : ℚ :=
  let quarterWeight : ℚ := 1 / 5
  let meltedValuePerOunce : ℚ := 100
  let spendingValuePerQuarter : ℚ := 1 / 4
  let quartersPerOunce : ℚ := 1 / quarterWeight
  let meltedValuePerQuarter : ℚ := meltedValuePerOunce / quartersPerOunce
  meltedValuePerQuarter / spendingValuePerQuarter

/-- The ratio of the value of melted quarters to spent quarters is 80 -/
theorem melt_to_spend_ratio_is_80 : meltToSpendRatio = 80 := by
  sorry

end NUMINAMATH_CALUDE_melt_to_spend_ratio_is_80_l3724_372490


namespace NUMINAMATH_CALUDE_distance_for_equilateral_hyperbola_locus_l3724_372498

/-- Two circles C1 and C2 with variable tangent t to C1 intersecting C2 at A and B.
    Tangents to C2 through A and B intersect at P. -/
structure TwoCirclesConfig where
  r1 : ℝ  -- radius of C1
  r2 : ℝ  -- radius of C2
  d : ℝ   -- distance between centers of C1 and C2

/-- The locus of P is contained in an equilateral hyperbola -/
def isEquilateralHyperbolaLocus (config : TwoCirclesConfig) : Prop :=
  config.d = config.r1 * Real.sqrt 2

/-- Theorem: The distance between centers for equilateral hyperbola locus -/
theorem distance_for_equilateral_hyperbola_locus 
  (config : TwoCirclesConfig) (h1 : config.r1 > 0) (h2 : config.r2 > 0) :
  isEquilateralHyperbolaLocus config ↔ config.d = config.r1 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_distance_for_equilateral_hyperbola_locus_l3724_372498


namespace NUMINAMATH_CALUDE_parabola_focus_line_intersection_l3724_372450

/-- Parabola struct representing x^2 = 2py -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (par : Parabola) where
  x : ℝ
  y : ℝ
  h : x^2 = 2 * par.p * y

/-- Line passing through the focus of a parabola with slope √3 -/
structure FocusLine (par : Parabola) where
  slope : ℝ
  hslope : slope = Real.sqrt 3
  pass_focus : ℝ → ℝ
  hpass : pass_focus 0 = par.p / 2

theorem parabola_focus_line_intersection (par : Parabola) 
  (l : FocusLine par) (M N : ParabolaPoint par) 
  (hM : M.x > 0) (hMN : M.x ≠ N.x) : 
  (par.p = 2 → M.x * N.x = -4) ∧ 
  (M.y * N.y = 1 → par.p = 2) ∧ 
  (par.p = 2 → Real.sqrt ((M.x - 0)^2 + (M.y - par.p/2)^2) = 8 + 4 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_line_intersection_l3724_372450


namespace NUMINAMATH_CALUDE_max_value_of_f_l3724_372457

def f (x : ℝ) := x^3 - 3*x

theorem max_value_of_f :
  ∃ (m : ℝ), m = 18 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3724_372457


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3724_372404

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3724_372404


namespace NUMINAMATH_CALUDE_fruits_remaining_l3724_372443

theorem fruits_remaining (initial_apples : ℕ) (plum_ratio : ℚ) (picked_ratio : ℚ) : 
  initial_apples = 180 → 
  plum_ratio = 1 / 3 → 
  picked_ratio = 3 / 5 → 
  (initial_apples + (↑initial_apples * plum_ratio)) * (1 - picked_ratio) = 96 := by
sorry

end NUMINAMATH_CALUDE_fruits_remaining_l3724_372443


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3724_372460

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, -x^2 + 2*x + 3 ≥ 0 ↔ x ∈ Set.Icc (-1) 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3724_372460


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3724_372474

theorem fraction_multiplication : (1/2 + 5/6 - 7/12) * (-36) = -27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3724_372474


namespace NUMINAMATH_CALUDE_walking_speed_is_4_l3724_372482

/-- The speed at which Jack and Jill walked -/
def walking_speed : ℝ → ℝ := λ x => x^3 - 5*x^2 - 14*x + 104

/-- The distance Jill walked -/
def jill_distance : ℝ → ℝ := λ x => x^2 - 7*x - 60

/-- The time Jill walked -/
def jill_time : ℝ → ℝ := λ x => x + 7

theorem walking_speed_is_4 :
  ∃ x : ℝ, x ≠ -7 ∧ walking_speed x = (jill_distance x) / (jill_time x) ∧ walking_speed x = 4 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_is_4_l3724_372482


namespace NUMINAMATH_CALUDE_nine_by_n_grid_rectangles_l3724_372478

theorem nine_by_n_grid_rectangles (n : ℕ) : 
  (9 : ℕ) > 1 → n > 1 → (Nat.choose 9 2 * Nat.choose n 2 = 756) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_nine_by_n_grid_rectangles_l3724_372478


namespace NUMINAMATH_CALUDE_sliding_chord_annulus_area_l3724_372425

/-- The area of the annulus formed by a sliding chord on a circle -/
theorem sliding_chord_annulus_area
  (R : ℝ) -- radius of the outer circle
  (a b : ℝ) -- distances from point C to ends A and B of the chord
  (h1 : R > 0) -- radius is positive
  (h2 : a > 0) -- distance a is positive
  (h3 : b > 0) -- distance b is positive
  (h4 : a + b ≤ 2 * R) -- chord length constraint
  : ∃ (r : ℝ), -- radius of the inner circle
    r^2 = R^2 - a * b ∧ 
    π * R^2 - π * r^2 = π * a * b :=
by sorry

end NUMINAMATH_CALUDE_sliding_chord_annulus_area_l3724_372425


namespace NUMINAMATH_CALUDE_unique_m_value_l3724_372439

def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

theorem unique_m_value (a m : ℝ) 
  (h1 : 0 < m) (h2 : m < 1)
  (h3 : ∀ x : ℝ, f x a m ≥ 2)
  (h4 : a ≤ -5 ∨ a ≥ 5) :
  m = 1/5 := by
sorry

end NUMINAMATH_CALUDE_unique_m_value_l3724_372439


namespace NUMINAMATH_CALUDE_san_diego_zoo_tickets_l3724_372426

/-- Given a family of 7 members visiting the San Diego Zoo, prove that 3 adult tickets were purchased. --/
theorem san_diego_zoo_tickets (total_cost : ℕ) (adult_price child_price : ℕ) 
  (h1 : total_cost = 119)
  (h2 : adult_price = 21)
  (h3 : child_price = 14) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = 7 ∧
    adult_tickets * adult_price + child_tickets * child_price = total_cost ∧
    adult_tickets = 3 := by
  sorry

end NUMINAMATH_CALUDE_san_diego_zoo_tickets_l3724_372426


namespace NUMINAMATH_CALUDE_acid_mixing_problem_l3724_372499

/-- The largest integer concentration percentage achievable in the acid mixing problem -/
def largest_concentration : ℕ := 76

theorem acid_mixing_problem :
  ∀ r : ℕ,
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ (2.8 + 0.9 * x) / (4 + x) = r / 100) →
    r ≤ largest_concentration :=
by sorry

end NUMINAMATH_CALUDE_acid_mixing_problem_l3724_372499


namespace NUMINAMATH_CALUDE_sqrt_eleven_between_integers_l3724_372401

theorem sqrt_eleven_between_integers (a : ℤ) : 3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4 →
  (↑a < Real.sqrt 11 ∧ Real.sqrt 11 < ↑a + 1) ↔ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eleven_between_integers_l3724_372401


namespace NUMINAMATH_CALUDE_negation_equivalence_l3724_372456

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 > Real.exp x) ↔ (∀ x : ℝ, x^2 ≤ Real.exp x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3724_372456


namespace NUMINAMATH_CALUDE_v3_equals_55_l3724_372491

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^5 + 8x^4 - 3x^3 + 5x^2 + 12x - 6 -/
def f : List ℤ := [3, 8, -3, 5, 12, -6]

/-- Theorem: V_3 equals 55 when x = 2 for the given polynomial using Horner's method -/
theorem v3_equals_55 : horner f 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_v3_equals_55_l3724_372491


namespace NUMINAMATH_CALUDE_fraction_inequality_l3724_372472

theorem fraction_inequality (a b : ℚ) (h : a / b = 2 / 3) :
  ¬(∀ (x y : ℚ), x / y = 2 / 3 → x / y = (x + 2) / (y + 2)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3724_372472


namespace NUMINAMATH_CALUDE_triangle_side_relation_l3724_372479

theorem triangle_side_relation (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (angle_B : Real.cos (2 * Real.pi / 3) = -1/2) :
  a^2 + a*c + c^2 - b^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l3724_372479


namespace NUMINAMATH_CALUDE_blood_expires_february_5_l3724_372417

def seconds_per_day : ℕ := 24 * 60 * 60

def february_days : ℕ := 28

def blood_expiration_seconds : ℕ := Nat.factorial 9

def days_until_expiration : ℕ := blood_expiration_seconds / seconds_per_day

theorem blood_expires_february_5 :
  days_until_expiration = 4 →
  (1 : ℕ) + days_until_expiration = 5 :=
by sorry

end NUMINAMATH_CALUDE_blood_expires_february_5_l3724_372417


namespace NUMINAMATH_CALUDE_quadratic_function_m_equals_one_l3724_372496

/-- A quadratic function passing through a point with specific x-range constraints -/
def QuadraticFunction (a b m t : ℝ) : Prop :=
  a ≠ 0 ∧
  2 = a * m^2 - b * m ∧
  ∀ x, (a * x^2 - b * x ≥ -1) → (x ≤ t - 1 ∨ x ≥ -3 - t)

/-- The theorem stating that m must equal 1 given the conditions -/
theorem quadratic_function_m_equals_one (a b m t : ℝ) :
  QuadraticFunction a b m t → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_m_equals_one_l3724_372496


namespace NUMINAMATH_CALUDE_bread_baking_time_l3724_372468

theorem bread_baking_time (rise_time bake_time : ℕ) (num_balls : ℕ) : 
  rise_time = 3 → 
  bake_time = 2 → 
  num_balls = 4 → 
  (rise_time * num_balls) + (bake_time * num_balls) = 20 :=
by sorry

end NUMINAMATH_CALUDE_bread_baking_time_l3724_372468


namespace NUMINAMATH_CALUDE_probability_odd_even_function_selection_l3724_372470

theorem probability_odd_even_function_selection :
  let total_functions : ℕ := 7
  let odd_functions : ℕ := 3
  let even_functions : ℕ := 3
  let neither_odd_nor_even : ℕ := 1
  let total_selections : ℕ := total_functions.choose 2
  let favorable_selections : ℕ := odd_functions * even_functions
  favorable_selections / total_selections = 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_probability_odd_even_function_selection_l3724_372470


namespace NUMINAMATH_CALUDE_find_point_c_l3724_372466

/-- Given two points A and B in a 2D plane, and a point C such that 
    vector BC is half of vector BA, find the coordinates of point C. -/
theorem find_point_c (A B : ℝ × ℝ) (C : ℝ × ℝ) : 
  A = (1, 1) → 
  B = (-1, 2) → 
  C - B = (1/2) • (A - B) → 
  C = (0, 3/2) := by
sorry

end NUMINAMATH_CALUDE_find_point_c_l3724_372466


namespace NUMINAMATH_CALUDE_solution_set_transformation_l3724_372464

/-- Given that the solution set of ax^2 + bx + c > 0 is (1, 2),
    prove that the solution set of cx^2 + bx + a > 0 is (1/2, 1) -/
theorem solution_set_transformation (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + c > 0 ↔ 1 < x ∧ x < 2) →
  (∀ x : ℝ, c*x^2 + b*x + a > 0 ↔ 1/2 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_transformation_l3724_372464


namespace NUMINAMATH_CALUDE_complex_number_problem_l3724_372431

theorem complex_number_problem (a : ℝ) (z₁ : ℂ) (h₁ : a > 0) (h₂ : z₁ = 1 + a * I) (h₃ : ∃ b : ℝ, z₁^2 = b * I) :
  z₁ = 1 + I ∧ Complex.abs (z₁ / (1 - I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3724_372431


namespace NUMINAMATH_CALUDE_existence_of_equal_elements_l3724_372476

theorem existence_of_equal_elements
  (p q n : ℕ+)
  (h_sum : p + q < n)
  (x : Fin (n + 1) → ℤ)
  (h_boundary : x 0 = 0 ∧ x (Fin.last n) = 0)
  (h_diff : ∀ i : Fin n, x i.succ - x i = p ∨ x i.succ - x i = -q) :
  ∃ i j : Fin (n + 1), i < j ∧ (i, j) ≠ (0, Fin.last n) ∧ x i = x j :=
by sorry

end NUMINAMATH_CALUDE_existence_of_equal_elements_l3724_372476


namespace NUMINAMATH_CALUDE_find_a_l3724_372480

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- State the theorem
theorem find_a : 
  ∃ (a : ℝ), (∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3724_372480


namespace NUMINAMATH_CALUDE_professors_age_l3724_372449

/-- Represents a four-digit number abac --/
def FourDigitNumber (a b c : Nat) : Nat :=
  1000 * a + 100 * b + 10 * a + c

/-- Represents a two-digit number ab --/
def TwoDigitNumber (a b : Nat) : Nat :=
  10 * a + b

theorem professors_age (a b c : Nat) (x : Nat) 
  (h1 : x^2 = FourDigitNumber a b c)
  (h2 : x = TwoDigitNumber a b + TwoDigitNumber a c) :
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_professors_age_l3724_372449


namespace NUMINAMATH_CALUDE_total_earnings_l3724_372411

def markese_earnings : ℕ := 16
def difference : ℕ := 5

theorem total_earnings : 
  markese_earnings + (markese_earnings + difference) = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l3724_372411


namespace NUMINAMATH_CALUDE_inequality_solution_l3724_372430

theorem inequality_solution (x : ℕ+) : 4 * x - 3 < 2 * x + 1 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3724_372430


namespace NUMINAMATH_CALUDE_exists_all_accessible_l3724_372486

-- Define the type for cities
variable {City : Type}

-- Define the accessibility relation
variable (accessible : City → City → Prop)

-- Define the property that a city can access itself
variable (self_accessible : ∀ c : City, accessible c c)

-- Define the property that for any two cities, there's a third city that can access both
variable (exists_common_accessible : ∀ p q : City, ∃ r : City, accessible p r ∧ accessible q r)

-- The theorem to prove
theorem exists_all_accessible :
  ∃ c : City, ∀ other : City, accessible other c :=
sorry

end NUMINAMATH_CALUDE_exists_all_accessible_l3724_372486


namespace NUMINAMATH_CALUDE_quadratic_above_x_axis_l3724_372422

/-- Given a quadratic function f(x) = ax^2 + x + 5, if f(x) > 0 for all real x, then a > 1/20 -/
theorem quadratic_above_x_axis (a : ℝ) :
  (∀ x : ℝ, a * x^2 + x + 5 > 0) → a > 1/20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_above_x_axis_l3724_372422


namespace NUMINAMATH_CALUDE_tan_two_implies_expression_one_l3724_372403

theorem tan_two_implies_expression_one (x : ℝ) (h : Real.tan x = 2) :
  4 * (Real.sin x)^2 - 3 * (Real.sin x) * (Real.cos x) - 5 * (Real.cos x)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_expression_one_l3724_372403


namespace NUMINAMATH_CALUDE_horner_method_v3_l3724_372437

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_v3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * x + 0
  let v2 := v1 * x + 2
  v2 * x + 3

theorem horner_method_v3 :
  horner_v3 f 3 = 36 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3724_372437


namespace NUMINAMATH_CALUDE_num_winning_configurations_l3724_372492

/-- Represents a 4x4 tic-tac-toe board -/
def Board := Fin 4 → Fin 4 → Option Bool

/-- Represents a 3x3 section of the 4x4 board -/
def Section := Fin 3 → Fin 3 → Option Bool

/-- The number of 3x3 sections in a 4x4 board -/
def numSections : Nat := 4

/-- The number of ways to place X's in a winning 3x3 section for horizontal or vertical wins -/
def numXPlacementsRowCol : Nat := 18

/-- The number of ways to place X's in a winning 3x3 section for diagonal wins -/
def numXPlacementsDiag : Nat := 20

/-- The number of rows or columns in a 3x3 section -/
def numRowsOrCols : Nat := 6

/-- The number of diagonals in a 3x3 section -/
def numDiagonals : Nat := 2

/-- Calculates the total number of winning configurations in one 3x3 section -/
def winsIn3x3Section : Nat :=
  numRowsOrCols * numXPlacementsRowCol + numDiagonals * numXPlacementsDiag

/-- The main theorem: proves that the number of possible board configurations after Carl wins is 592 -/
theorem num_winning_configurations :
  (numSections * winsIn3x3Section) = 592 := by sorry

end NUMINAMATH_CALUDE_num_winning_configurations_l3724_372492


namespace NUMINAMATH_CALUDE_adam_first_half_correct_l3724_372473

/-- Represents the trivia game scenario -/
structure TriviaGame where
  pointsPerQuestion : ℕ
  secondHalfCorrect : ℕ
  finalScore : ℕ

/-- Calculates the number of correctly answered questions in the first half -/
def firstHalfCorrect (game : TriviaGame) : ℕ :=
  (game.finalScore - game.secondHalfCorrect * game.pointsPerQuestion) / game.pointsPerQuestion

/-- Theorem stating that Adam answered 8 questions correctly in the first half -/
theorem adam_first_half_correct :
  let game : TriviaGame := {
    pointsPerQuestion := 8,
    secondHalfCorrect := 2,
    finalScore := 80
  }
  firstHalfCorrect game = 8 := by sorry

end NUMINAMATH_CALUDE_adam_first_half_correct_l3724_372473


namespace NUMINAMATH_CALUDE_watermelon_weight_l3724_372467

theorem watermelon_weight (total_weight : ℝ) (half_removed_weight : ℝ) 
  (h1 : total_weight = 63)
  (h2 : half_removed_weight = 34) :
  let watermelon_weight := total_weight - half_removed_weight * 2
  watermelon_weight = 58 := by
sorry

end NUMINAMATH_CALUDE_watermelon_weight_l3724_372467


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3724_372463

theorem shaded_area_calculation (carpet_side : ℝ) (S T : ℝ) 
  (h1 : carpet_side = 12)
  (h2 : carpet_side / S = 4)
  (h3 : S / T = 4)
  (h4 : carpet_side > 0) : 
  S^2 + 16 * T^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3724_372463


namespace NUMINAMATH_CALUDE_three_digit_factorial_sum_l3724_372428

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem three_digit_factorial_sum : ∃ a b c : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  100 * a + 10 * b + c = factorial a + factorial b + factorial c := by
  sorry

end NUMINAMATH_CALUDE_three_digit_factorial_sum_l3724_372428


namespace NUMINAMATH_CALUDE_EF_length_l3724_372415

/-- Configuration of line segments AB, CD, and EF -/
structure Configuration where
  AB_length : ℝ
  CD_length : ℝ
  EF_start_x : ℝ
  EF_end_x : ℝ
  AB_height : ℝ
  CD_height : ℝ
  EF_height : ℝ

/-- Conditions for the configuration -/
def valid_configuration (c : Configuration) : Prop :=
  c.AB_length = 120 ∧
  c.CD_length = 80 ∧
  c.EF_start_x = c.CD_length / 2 ∧
  c.EF_end_x = c.CD_length ∧
  c.AB_height > c.EF_height ∧
  c.EF_height > c.CD_height ∧
  c.EF_height = (c.AB_height + c.CD_height) / 2

/-- Theorem: The length of EF is 40 cm -/
theorem EF_length (c : Configuration) (h : valid_configuration c) : 
  c.EF_end_x - c.EF_start_x = 40 := by
  sorry

end NUMINAMATH_CALUDE_EF_length_l3724_372415


namespace NUMINAMATH_CALUDE_g_value_proof_l3724_372477

def nabla (g h : ℝ) : ℝ := g^2 - h^2

theorem g_value_proof (g : ℝ) (h_pos : g > 0) (h_eq : nabla g 6 = 45) : g = 9 := by
  sorry

end NUMINAMATH_CALUDE_g_value_proof_l3724_372477


namespace NUMINAMATH_CALUDE_expression_value_l3724_372441

theorem expression_value (x y : ℝ) (h : x = 2*y + 1) : x^2 - 4*x*y + 4*y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3724_372441


namespace NUMINAMATH_CALUDE_negation_of_inequality_implication_is_true_l3724_372451

theorem negation_of_inequality_implication_is_true :
  ∀ (a b c : ℝ), (a ≤ b → a * c^2 ≤ b * c^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_inequality_implication_is_true_l3724_372451


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3724_372427

theorem unique_solution_exponential_equation :
  ∃! y : ℝ, (10 : ℝ)^(2*y) * (100 : ℝ)^y = (1000 : ℝ)^3 * (10 : ℝ)^y :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3724_372427


namespace NUMINAMATH_CALUDE_marble_bag_problem_l3724_372435

theorem marble_bag_problem (r b : ℕ) : 
  (r - 1 : ℚ) / (r + b - 2 : ℚ) = 1/8 →
  (r : ℚ) / (r + b - 3 : ℚ) = 1/4 →
  r + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l3724_372435


namespace NUMINAMATH_CALUDE_f_equals_g_l3724_372483

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 1
def g (t : ℝ) : ℝ := 2 * t - 1

-- Theorem stating that f and g are the same function
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l3724_372483


namespace NUMINAMATH_CALUDE_half_full_one_minute_before_end_l3724_372459

/-- Represents the filling process of a box with marbles -/
def FillingProcess (total_time : ℕ) : Type :=
  ℕ → ℝ

/-- The quantity doubles every minute -/
def DoublesEveryMinute (process : FillingProcess n) : Prop :=
  ∀ t, t < n → process (t + 1) = 2 * process t

/-- The process is complete at the total time -/
def CompleteAtEnd (process : FillingProcess n) : Prop :=
  process n = 1

/-- The box is half full at a given time -/
def HalfFullAt (process : FillingProcess n) (t : ℕ) : Prop :=
  process t = 1/2

theorem half_full_one_minute_before_end 
  (process : FillingProcess 10) 
  (h1 : DoublesEveryMinute process) 
  (h2 : CompleteAtEnd process) :
  HalfFullAt process 9 :=
sorry

end NUMINAMATH_CALUDE_half_full_one_minute_before_end_l3724_372459


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l3724_372416

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l3724_372416


namespace NUMINAMATH_CALUDE_rectangular_table_capacity_l3724_372418

/-- The number of square tables arranged in a row -/
def num_tables : ℕ := 8

/-- The number of people that can sit evenly spaced around one square table -/
def people_per_square_table : ℕ := 12

/-- The number of sides in a square table -/
def sides_per_square : ℕ := 4

/-- Calculate the number of people that can sit on one side of a square table -/
def people_per_side : ℕ := people_per_square_table / sides_per_square

/-- The number of people that can sit on the long side of the rectangular table -/
def long_side_capacity : ℕ := num_tables * people_per_side

/-- The number of people that can sit on the short side of the rectangular table -/
def short_side_capacity : ℕ := 2 * people_per_side

/-- The total number of people that can sit around the rectangular table -/
def total_capacity : ℕ := 2 * long_side_capacity + 2 * short_side_capacity

theorem rectangular_table_capacity :
  total_capacity = 60 := by sorry

end NUMINAMATH_CALUDE_rectangular_table_capacity_l3724_372418


namespace NUMINAMATH_CALUDE_distance_to_plane_l3724_372493

/-- The distance from a point to a plane defined by three points -/
def distancePointToPlane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  -- Implementation details omitted
  sorry

theorem distance_to_plane :
  let M₀ : ℝ × ℝ × ℝ := (1, -6, -5)
  let M₁ : ℝ × ℝ × ℝ := (-1, 2, -3)
  let M₂ : ℝ × ℝ × ℝ := (4, -1, 0)
  let M₃ : ℝ × ℝ × ℝ := (2, 1, -2)
  distancePointToPlane M₀ M₁ M₂ M₃ = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_plane_l3724_372493


namespace NUMINAMATH_CALUDE_total_amount_is_234_l3724_372414

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Theorem stating the total amount given the conditions -/
theorem total_amount_is_234 
  (div : MoneyDivision) 
  (h1 : div.y = div.x * (45/100))  -- y gets 45 paisa for each rupee x gets
  (h2 : div.z = div.x * (50/100))  -- z gets 50 paisa for each rupee x gets
  (h3 : div.y = 54)                -- The share of y is Rs. 54
  : div.x + div.y + div.z = 234 := by
  sorry

#check total_amount_is_234

end NUMINAMATH_CALUDE_total_amount_is_234_l3724_372414


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l3724_372409

-- Define the lines that bound the region
def line1 (x y : ℝ) : Prop := x + y = 6
def line2 (y : ℝ) : Prop := y = 4
def line3 (x : ℝ) : Prop := x = 0
def line4 (y : ℝ) : Prop := y = 0

-- Define the vertices of the quadrilateral
def P : ℝ × ℝ := (6, 0)
def Q : ℝ × ℝ := (2, 4)
def R : ℝ × ℝ := (0, 6)
def O : ℝ × ℝ := (0, 0)

-- Define the area of the quadrilateral
def area_quadrilateral : ℝ := 18

-- Theorem statement
theorem area_of_bounded_region :
  area_quadrilateral = 18 :=
sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l3724_372409


namespace NUMINAMATH_CALUDE_larger_root_of_quadratic_l3724_372461

theorem larger_root_of_quadratic (x : ℝ) : 
  (x - 5/8) * (x - 5/8) + (x - 5/8) * (x - 1/3) = 0 →
  x = 5/8 ∨ x = 23/48 →
  (5/8 : ℝ) > (23/48 : ℝ) →
  x = 5/8 := by sorry

end NUMINAMATH_CALUDE_larger_root_of_quadratic_l3724_372461


namespace NUMINAMATH_CALUDE_brother_age_proof_l3724_372424

/-- Trevor's current age -/
def Trevor_current_age : ℕ := 11

/-- Trevor's future age when the condition is met -/
def Trevor_future_age : ℕ := 24

/-- Trevor's older brother's current age -/
def Brother_current_age : ℕ := 20

theorem brother_age_proof :
  (Trevor_future_age - Trevor_current_age = Brother_current_age - Trevor_current_age) ∧
  (Brother_current_age + (Trevor_future_age - Trevor_current_age) = 3 * Trevor_current_age) :=
by sorry

end NUMINAMATH_CALUDE_brother_age_proof_l3724_372424


namespace NUMINAMATH_CALUDE_trajectory_and_constant_slope_l3724_372455

noncomputable section

-- Define the points A and P
def A : ℝ × ℝ := (3, -6)
def P : ℝ × ℝ := (1, -2)

-- Define the curve
def on_curve (Q : ℝ × ℝ) : Prop :=
  let (x, y) := Q
  (x^2 + y^2) / ((x - 3)^2 + (y + 6)^2) = 1/4

-- Define complementary angles
def complementary_angles (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Define the theorem
theorem trajectory_and_constant_slope :
  -- Part 1: Equation of the curve
  (∀ Q : ℝ × ℝ, on_curve Q ↔ (Q.1 + 1)^2 + (Q.2 - 2)^2 = 20) ∧
  -- Part 2: Constant slope of BC
  (∀ B C : ℝ × ℝ, 
    on_curve B ∧ on_curve C ∧ 
    (∃ m1 m2 : ℝ, 
      complementary_angles m1 m2 ∧
      (B.2 - P.2) = m1 * (B.1 - P.1) ∧
      (C.2 - P.2) = m2 * (C.1 - P.1)) →
    (C.2 - B.2) / (C.1 - B.1) = -1/2) :=
sorry

end

end NUMINAMATH_CALUDE_trajectory_and_constant_slope_l3724_372455


namespace NUMINAMATH_CALUDE_division_of_decimals_l3724_372497

theorem division_of_decimals : (0.05 : ℝ) / 0.0025 = 20 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l3724_372497


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l3724_372434

theorem quadratic_root_sum (m n : ℝ) : 
  m^2 + 2*m - 5 = 0 → n^2 + 2*n - 5 = 0 → m^2 + m*n + 2*m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l3724_372434


namespace NUMINAMATH_CALUDE_john_mileage_conversion_l3724_372407

/-- Converts a base-8 number represented as a list of digits to its base-10 equivalent -/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

/-- The base-8 representation of John's mileage -/
def johnMileageBase8 : List Nat := [3, 4, 5, 2]

/-- Theorem: John's mileage in base-10 is 1834 miles -/
theorem john_mileage_conversion :
  base8ToBase10 johnMileageBase8 = 1834 := by
  sorry

end NUMINAMATH_CALUDE_john_mileage_conversion_l3724_372407


namespace NUMINAMATH_CALUDE_red_then_blue_probability_l3724_372462

/-- The probability of drawing a red marble first and a blue marble second from a jar -/
theorem red_then_blue_probability (red green white blue : ℕ) :
  red = 4 →
  green = 3 →
  white = 10 →
  blue = 2 →
  let total := red + green + white + blue
  let prob_red := red / total
  let prob_blue_after_red := blue / (total - 1)
  prob_red * prob_blue_after_red = 4 / 171 := by
  sorry

end NUMINAMATH_CALUDE_red_then_blue_probability_l3724_372462


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l3724_372408

/-- Given a complex number z satisfying z + |z| = 2 + 8i, prove that |z|² = 289 -/
theorem complex_modulus_squared (z : ℂ) (h : z + Complex.abs z = 2 + 8 * Complex.I) : 
  Complex.abs z ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l3724_372408


namespace NUMINAMATH_CALUDE_coin_game_winning_strategy_l3724_372446

/-- Represents the state of the coin game -/
structure GameState where
  piles : List Nat
  deriving Repr

/-- Checks if a player has a winning strategy given the current game state -/
def hasWinningStrategy (state : GameState) : Prop :=
  let n1 := (state.piles.filter (· = 1)).length
  let evenPiles := state.piles.filter (· % 2 = 0)
  let sumEvenPiles := (evenPiles.map (λ x => x / 2)).sum
  Odd n1 ∨ Odd sumEvenPiles

/-- The main theorem stating the winning condition for the coin game -/
theorem coin_game_winning_strategy (state : GameState) :
  hasWinningStrategy state ↔ 
  Odd (state.piles.filter (· = 1)).length ∨ 
  Odd ((state.piles.filter (· % 2 = 0)).map (λ x => x / 2)).sum :=
by sorry


end NUMINAMATH_CALUDE_coin_game_winning_strategy_l3724_372446


namespace NUMINAMATH_CALUDE_games_played_l3724_372419

-- Define the total points scored
def total_points : ℝ := 120.0

-- Define the points scored per game
def points_per_game : ℝ := 12

-- Theorem to prove
theorem games_played : (total_points / points_per_game : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_games_played_l3724_372419


namespace NUMINAMATH_CALUDE_lazy_kingdom_date_l3724_372448

-- Define the days of the week in the Lazy Kingdom
inductive LazyDay
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Saturday

-- Define a function to calculate the next day
def nextDay (d : LazyDay) : LazyDay :=
  match d with
  | LazyDay.Sunday => LazyDay.Monday
  | LazyDay.Monday => LazyDay.Tuesday
  | LazyDay.Tuesday => LazyDay.Wednesday
  | LazyDay.Wednesday => LazyDay.Thursday
  | LazyDay.Thursday => LazyDay.Saturday
  | LazyDay.Saturday => LazyDay.Sunday

-- Define a function to calculate the day after n days
def dayAfter (start : LazyDay) (n : Nat) : LazyDay :=
  match n with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

-- Theorem statement
theorem lazy_kingdom_date : 
  dayAfter LazyDay.Sunday 374 = LazyDay.Tuesday := by
  sorry


end NUMINAMATH_CALUDE_lazy_kingdom_date_l3724_372448


namespace NUMINAMATH_CALUDE_circle_equation_with_given_endpoints_l3724_372471

/-- The standard equation of a circle with diameter endpoints M(2,0) and N(0,4) -/
theorem circle_equation_with_given_endpoints :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 ↔ 
  ∃ (t : ℝ), x = 2 * (1 - t) + 0 * t ∧ y = 0 * (1 - t) + 4 * t ∧ 0 ≤ t ∧ t ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_given_endpoints_l3724_372471


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3724_372410

-- Define the sets A and B
def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3724_372410


namespace NUMINAMATH_CALUDE_zoo_visit_l3724_372402

/-- The number of children who saw giraffes but not pandas -/
def giraffes_not_pandas (total children_pandas children_giraffes pandas_not_giraffes : ℕ) : ℕ :=
  children_giraffes - (children_pandas - pandas_not_giraffes)

/-- Theorem stating the number of children who saw giraffes but not pandas -/
theorem zoo_visit (total children_pandas children_giraffes pandas_not_giraffes : ℕ) 
  (h1 : total = 50)
  (h2 : children_pandas = 36)
  (h3 : children_giraffes = 28)
  (h4 : pandas_not_giraffes = 15) :
  giraffes_not_pandas total children_pandas children_giraffes pandas_not_giraffes = 7 := by
  sorry


end NUMINAMATH_CALUDE_zoo_visit_l3724_372402


namespace NUMINAMATH_CALUDE_tan_difference_implies_ratio_l3724_372494

theorem tan_difference_implies_ratio (α : Real) 
  (h : Real.tan (α - π/4) = 1/2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_implies_ratio_l3724_372494


namespace NUMINAMATH_CALUDE_unique_solution_k_values_l3724_372488

/-- The set of values for k that satisfy the given conditions -/
def k_values : Set ℝ := {1 + Real.sqrt 2, (1 - Real.sqrt 5) / 2}

/-- The system of inequalities -/
def system (k x : ℝ) : Prop :=
  1 ≤ k * x^2 + 2 ∧ x + k ≤ 2

/-- The main theorem stating that k_values is the correct set of values for k -/
theorem unique_solution_k_values :
  ∀ k : ℝ, (∃! x : ℝ, system k x) ↔ k ∈ k_values := by
  sorry

#check unique_solution_k_values

end NUMINAMATH_CALUDE_unique_solution_k_values_l3724_372488


namespace NUMINAMATH_CALUDE_prob_select_one_from_couple_l3724_372484

/-- The probability of selecting exactly one person from a couple, given their individual selection probabilities -/
theorem prob_select_one_from_couple (p_husband p_wife : ℝ) 
  (h_husband : p_husband = 1/7)
  (h_wife : p_wife = 1/5) :
  p_husband * (1 - p_wife) + p_wife * (1 - p_husband) = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_select_one_from_couple_l3724_372484


namespace NUMINAMATH_CALUDE_equal_sides_implies_rhombus_l3724_372433

-- Define a quadrilateral
structure Quadrilateral :=
  (sides : Fin 4 → ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, q.sides i = q.sides j

-- Theorem statement
theorem equal_sides_implies_rhombus (q : Quadrilateral) :
  (∀ i j : Fin 4, q.sides i = q.sides j) → is_rhombus q :=
by
  sorry


end NUMINAMATH_CALUDE_equal_sides_implies_rhombus_l3724_372433


namespace NUMINAMATH_CALUDE_angle_sum_proof_l3724_372495

theorem angle_sum_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = 2 * Real.sqrt 5 / 5) (h4 : Real.sin β = 3 * Real.sqrt 10 / 10) :
  α + β = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l3724_372495


namespace NUMINAMATH_CALUDE_cos_105_degrees_l3724_372444

theorem cos_105_degrees : Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l3724_372444


namespace NUMINAMATH_CALUDE_color_films_count_l3724_372423

theorem color_films_count (x y : ℝ) (h : x > 0) :
  let total_bw := 40 * x
  let selected_bw := 2 * y / 5
  let fraction_color := 0.9615384615384615
  let color_films := (fraction_color * (selected_bw + color_films)) / (1 - fraction_color)
  color_films = 10 * y := by
  sorry

end NUMINAMATH_CALUDE_color_films_count_l3724_372423


namespace NUMINAMATH_CALUDE_twitch_income_per_subscriber_l3724_372453

/-- Calculates the income per subscriber for a Twitch streamer --/
theorem twitch_income_per_subscriber
  (initial_subscribers : ℕ)
  (gifted_subscribers : ℕ)
  (total_monthly_income : ℕ)
  (h1 : initial_subscribers = 150)
  (h2 : gifted_subscribers = 50)
  (h3 : total_monthly_income = 1800) :
  total_monthly_income / (initial_subscribers + gifted_subscribers) = 9 := by
sorry

end NUMINAMATH_CALUDE_twitch_income_per_subscriber_l3724_372453


namespace NUMINAMATH_CALUDE_max_y_value_l3724_372412

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = 8) : 
  y ≤ 43 ∧ ∃ (x₀ y₀ : ℤ), x₀ * y₀ + 7 * x₀ + 6 * y₀ = 8 ∧ y₀ = 43 :=
by sorry

end NUMINAMATH_CALUDE_max_y_value_l3724_372412


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3724_372405

/-- Represents the speed of a boat in still water and a stream -/
structure BoatAndStreamSpeeds where
  boat : ℝ
  stream : ℝ

/-- Represents the time taken to travel upstream and downstream -/
structure TravelTimes where
  downstream : ℝ
  upstream : ℝ

/-- The problem statement -/
theorem boat_speed_in_still_water 
  (speeds : BoatAndStreamSpeeds)
  (times : TravelTimes)
  (h1 : speeds.stream = 13)
  (h2 : times.upstream = 2 * times.downstream)
  (h3 : (speeds.boat + speeds.stream) * times.downstream = 
        (speeds.boat - speeds.stream) * times.upstream) :
  speeds.boat = 39 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3724_372405


namespace NUMINAMATH_CALUDE_min_value_theorem_l3724_372487

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 24 / 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 24 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3724_372487


namespace NUMINAMATH_CALUDE_batch_composition_l3724_372447

/-- Represents the characteristics of a product type -/
structure ProductType where
  volume : ℝ  -- Volume per unit in m³
  mass : ℝ    -- Mass per unit in tons

/-- Represents a batch of products -/
structure Batch where
  typeA : ProductType
  typeB : ProductType
  totalVolume : ℝ
  totalMass : ℝ

/-- Theorem: Given the specific product characteristics and total volume and mass,
    prove that the batch consists of 5 units of type A and 8 units of type B -/
theorem batch_composition (b : Batch)
    (h1 : b.typeA.volume = 0.8)
    (h2 : b.typeA.mass = 0.5)
    (h3 : b.typeB.volume = 2)
    (h4 : b.typeB.mass = 1)
    (h5 : b.totalVolume = 20)
    (h6 : b.totalMass = 10.5) :
    ∃ (x y : ℝ), x = 5 ∧ y = 8 ∧
    x * b.typeA.volume + y * b.typeB.volume = b.totalVolume ∧
    x * b.typeA.mass + y * b.typeB.mass = b.totalMass :=
  sorry


end NUMINAMATH_CALUDE_batch_composition_l3724_372447


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3724_372413

/-- Given a function f where the tangent line at x=2 is 2x+y-3=0, prove f(2) + f'(2) = -3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : DifferentiableAt ℝ f 2) 
  (h_tangent : ∀ x y, y = f x → (x = 2 → 2*x + y - 3 = 0)) : 
  f 2 + deriv f 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3724_372413


namespace NUMINAMATH_CALUDE_remainder_n_plus_2023_l3724_372454

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 7 = 3) : (n + 2023) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_n_plus_2023_l3724_372454


namespace NUMINAMATH_CALUDE_kaprekar_convergence_l3724_372452

/-- Reverses a four-digit number -/
def reverseNumber (n : Nat) : Nat :=
  sorry

/-- Rearranges digits of a four-digit number from largest to smallest -/
def rearrangeDigits (n : Nat) : Nat :=
  sorry

/-- Applies the Kaprekar transformation to a four-digit number -/
def kaprekarTransform (n : Nat) : Nat :=
  let m := rearrangeDigits n
  let r := reverseNumber m
  m - r

/-- Applies the Kaprekar transformation k times -/
def kaprekarTransformK (n : Nat) (k : Nat) : Nat :=
  sorry

theorem kaprekar_convergence (n : Nat) (h : n = 5298 ∨ n = 4852) :
  ∃ (k : Nat) (t : Nat), k = 7 ∧ t = 6174 ∧
    kaprekarTransformK n k = t ∧
    kaprekarTransform t = t :=
  sorry

end NUMINAMATH_CALUDE_kaprekar_convergence_l3724_372452


namespace NUMINAMATH_CALUDE_cost_of_four_enchiladas_five_tacos_l3724_372420

/-- The price of an enchilada -/
def enchilada_price : ℝ := sorry

/-- The price of a taco -/
def taco_price : ℝ := sorry

/-- The first condition: 5 enchiladas and 2 tacos cost $4.30 -/
axiom condition1 : 5 * enchilada_price + 2 * taco_price = 4.30

/-- The second condition: 4 enchiladas and 3 tacos cost $4.50 -/
axiom condition2 : 4 * enchilada_price + 3 * taco_price = 4.50

/-- The theorem to prove -/
theorem cost_of_four_enchiladas_five_tacos :
  4 * enchilada_price + 5 * taco_price = 6.01 := by sorry

end NUMINAMATH_CALUDE_cost_of_four_enchiladas_five_tacos_l3724_372420


namespace NUMINAMATH_CALUDE_intersection_sum_l3724_372436

/-- Given two lines that intersect at (3,3), prove that a + b = 4 -/
theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 3 + a) → -- First line passes through (3,3)
  (3 = (1/3) * 3 + b) → -- Second line passes through (3,3)
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l3724_372436


namespace NUMINAMATH_CALUDE_coin_problem_l3724_372432

theorem coin_problem (total : ℕ) (difference : ℕ) (tails : ℕ) : 
  total = 1250 →
  difference = 124 →
  tails + (tails + difference) = total →
  tails = 563 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l3724_372432


namespace NUMINAMATH_CALUDE_viewership_difference_l3724_372475

/-- The number of viewers for each game this week -/
structure ViewersThisWeek where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The total number of viewers last week -/
def viewersLastWeek : ℕ := 350

/-- The conditions for this week's viewership -/
def viewershipConditions (v : ViewersThisWeek) : Prop :=
  v.second = 80 ∧
  v.first = v.second - 20 ∧
  v.third = v.second + 15 ∧
  v.fourth = v.third + (v.third / 10)

/-- The theorem to prove -/
theorem viewership_difference (v : ViewersThisWeek) 
  (h : viewershipConditions v) : 
  v.first + v.second + v.third + v.fourth = viewersLastWeek - 10 := by
  sorry

end NUMINAMATH_CALUDE_viewership_difference_l3724_372475


namespace NUMINAMATH_CALUDE_uncool_parents_count_l3724_372445

/-- Proves the number of students with uncool parents in a music class -/
theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total = 40)
  (h2 : cool_dads = 25)
  (h3 : cool_moms = 19)
  (h4 : both_cool = 8) :
  total - (cool_dads + cool_moms - both_cool) = 4 := by
  sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l3724_372445


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3724_372438

theorem floor_equation_solution (n : ℤ) :
  (⌊n^2 / 3⌋ : ℤ) - (⌊n / 2⌋ : ℤ)^2 = 3 ↔ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3724_372438


namespace NUMINAMATH_CALUDE_transaction_result_l3724_372469

theorem transaction_result : 
  ∀ (house_cost store_cost : ℕ),
  (house_cost * 3 / 4 = 15000) →
  (store_cost * 5 / 4 = 10000) →
  (house_cost + store_cost) - (15000 + 10000) = 3000 :=
by
  sorry

end NUMINAMATH_CALUDE_transaction_result_l3724_372469


namespace NUMINAMATH_CALUDE_trajectory_and_no_line_exist_l3724_372421

-- Define the points and vectors
def A : ℝ × ℝ := (8, 0)
def Q : ℝ × ℝ := (-1, 0)

-- Define the conditions
def condition1 (B P : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BP := (P.1 - B.1, P.2 - B.2)
  AB.1 * BP.1 + AB.2 * BP.2 = 0

def condition2 (B C P : ℝ × ℝ) : Prop :=
  (C.1 - B.1, C.2 - B.2) = (P.1 - C.1, P.2 - C.2)

def on_y_axis (B : ℝ × ℝ) : Prop := B.1 = 0
def on_x_axis (C : ℝ × ℝ) : Prop := C.2 = 0

-- Define the trajectory
def trajectory (P : ℝ × ℝ) : Prop := P.2^2 = -4 * P.1

-- Define the line passing through A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k * x - 8 * k

-- Define the dot product condition
def dot_product_condition (M N : ℝ × ℝ) : Prop :=
  let QM := (M.1 - Q.1, M.2 - Q.2)
  let QN := (N.1 - Q.1, N.2 - Q.2)
  QM.1 * QN.1 + QM.2 * QN.2 = 97

-- The main theorem
theorem trajectory_and_no_line_exist :
  ∀ B C P, on_y_axis B → on_x_axis C →
  condition1 B P → condition2 B C P →
  (trajectory P ∧
   ¬∃ k M N, line_through_A k M.1 M.2 ∧ line_through_A k N.1 N.2 ∧
              trajectory M ∧ trajectory N ∧ dot_product_condition M N) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_no_line_exist_l3724_372421


namespace NUMINAMATH_CALUDE_book_sale_loss_l3724_372400

/-- Given that the cost price of 15 books equals the selling price of 20 books,
    prove that there is a 25% loss. -/
theorem book_sale_loss (C S : ℝ) (h : 15 * C = 20 * S) :
  (C - S) / C * 100 = 25 :=
sorry

end NUMINAMATH_CALUDE_book_sale_loss_l3724_372400


namespace NUMINAMATH_CALUDE_restaurant_students_l3724_372481

theorem restaurant_students (burgers hot_dogs pizza_slices sandwiches : ℕ) : 
  burgers = 30 ∧ 
  burgers = 2 * hot_dogs ∧ 
  pizza_slices = hot_dogs + 5 ∧ 
  sandwiches = 3 * pizza_slices → 
  burgers + hot_dogs + pizza_slices + sandwiches = 125 := by
sorry

end NUMINAMATH_CALUDE_restaurant_students_l3724_372481


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3724_372485

theorem inequality_solution_set (x : ℝ) : 
  (1/3 : ℝ) + |x - 11/48| < 1/2 ↔ x ∈ Set.Ioo (1/16 : ℝ) (19/48 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3724_372485
