import Mathlib

namespace NUMINAMATH_CALUDE_constant_value_l422_42299

theorem constant_value (t : ℝ) (x y : ℝ → ℝ) (constant : ℝ) :
  (∀ t, x t = constant - 3 * t) →
  (∀ t, y t = 2 * t - 3) →
  x 0.8 = y 0.8 →
  constant = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_value_l422_42299


namespace NUMINAMATH_CALUDE_gcd_5280_12155_l422_42243

theorem gcd_5280_12155 : Int.gcd 5280 12155 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5280_12155_l422_42243


namespace NUMINAMATH_CALUDE_bisection_second_iteration_l422_42258

def f (x : ℝ) := x^3 + 3*x - 1

theorem bisection_second_iteration
  (h1 : f 0 < 0)
  (h2 : f (1/2) > 0) :
  let second_iteration := (0 + 1/2) / 2
  second_iteration = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_bisection_second_iteration_l422_42258


namespace NUMINAMATH_CALUDE_base_two_rep_of_125_l422_42260

theorem base_two_rep_of_125 : 
  (125 : ℕ).digits 2 = [1, 0, 1, 1, 1, 1, 1] :=
by sorry

end NUMINAMATH_CALUDE_base_two_rep_of_125_l422_42260


namespace NUMINAMATH_CALUDE_weight_of_two_balls_l422_42270

/-- The weight of two balls on a scale is equal to the sum of their individual weights -/
theorem weight_of_two_balls (blue_weight brown_weight : ℝ) :
  let total_weight := blue_weight + brown_weight
  blue_weight = 6 ∧ brown_weight = 3.12 → total_weight = 9.12 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_two_balls_l422_42270


namespace NUMINAMATH_CALUDE_cow_count_l422_42275

theorem cow_count (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 32) → 
  cows = 16 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l422_42275


namespace NUMINAMATH_CALUDE_nine_power_equation_solution_l422_42224

theorem nine_power_equation_solution :
  ∃! n : ℝ, (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n = (81 : ℝ)^4 :=
by
  sorry

end NUMINAMATH_CALUDE_nine_power_equation_solution_l422_42224


namespace NUMINAMATH_CALUDE_brainiac_teaser_ratio_l422_42225

/-- Represents the number of brainiacs who like rebus teasers -/
def R : ℕ := 58

/-- Represents the number of brainiacs who like math teasers -/
def M : ℕ := 38

/-- The total number of brainiacs surveyed -/
def total : ℕ := 100

/-- The number of brainiacs who like both rebus and math teasers -/
def both : ℕ := 18

/-- The number of brainiacs who like neither rebus nor math teasers -/
def neither : ℕ := 4

/-- The number of brainiacs who like math teasers but not rebus teasers -/
def mathOnly : ℕ := 20

theorem brainiac_teaser_ratio :
  R = 58 ∧ M = 38 ∧ 
  total = 100 ∧
  both = 18 ∧
  neither = 4 ∧
  mathOnly = 20 →
  R * 19 = M * 29 := by
  sorry

end NUMINAMATH_CALUDE_brainiac_teaser_ratio_l422_42225


namespace NUMINAMATH_CALUDE_balki_cereal_boxes_l422_42233

theorem balki_cereal_boxes : 
  let total_raisins : ℕ := 437
  let box1_raisins : ℕ := 72
  let box2_raisins : ℕ := 74
  let other_boxes_raisins : ℕ := 97
  let num_other_boxes : ℕ := (total_raisins - box1_raisins - box2_raisins) / other_boxes_raisins
  let total_boxes : ℕ := 2 + num_other_boxes
  total_boxes = 5 ∧ 
  box1_raisins + box2_raisins + num_other_boxes * other_boxes_raisins = total_raisins :=
by sorry

end NUMINAMATH_CALUDE_balki_cereal_boxes_l422_42233


namespace NUMINAMATH_CALUDE_subway_ride_time_l422_42229

theorem subway_ride_time (total_time subway_time train_time bike_time : ℝ) : 
  total_time = 38 →
  train_time = 2 * subway_time →
  bike_time = 8 →
  total_time = subway_time + train_time + bike_time →
  subway_time = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_subway_ride_time_l422_42229


namespace NUMINAMATH_CALUDE_base6_arithmetic_l422_42252

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 * 6^3 + d2 * 6^2 + d3 * 6 + d4

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ :=
  let d1 := n / (6^3)
  let d2 := (n / (6^2)) % 6
  let d3 := (n / 6) % 6
  let d4 := n % 6
  d1 * 1000 + d2 * 100 + d3 * 10 + d4

/-- The main theorem to prove --/
theorem base6_arithmetic : 
  base10ToBase6 (base6ToBase10 4512 - base6ToBase10 2324 + base6ToBase10 1432) = 4020 := by
  sorry

end NUMINAMATH_CALUDE_base6_arithmetic_l422_42252


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l422_42201

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l422_42201


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_parameter_product_l422_42289

/-- Given an ellipse and a hyperbola with specific foci, prove the product of their parameters. -/
theorem ellipse_hyperbola_parameter_product :
  ∀ (p q : ℝ),
  (∀ (x y : ℝ), x^2 / p^2 + y^2 / q^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2 / p^2 - y^2 / q^2 = 1 → (x = 8 ∧ y = 0) ∨ (x = -8 ∧ y = 0)) →
  |p * q| = Real.sqrt 12371 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_parameter_product_l422_42289


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l422_42268

theorem geometric_sequence_proof (a : ℕ → ℝ) (q : ℝ) (h_positive : q > 0) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 2 * a 6 = 9 * a 4 →             -- given condition
  a 2 = 1 →                         -- given condition
  (q = 3 ∧ ∀ n : ℕ, a n = 3^(n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l422_42268


namespace NUMINAMATH_CALUDE_prize_order_count_is_32_l422_42282

/-- Represents a bowling tournament with 6 players and a specific playoff system. -/
structure BowlingTournament where
  players : Fin 6
  /-- The number of matches in the tournament -/
  num_matches : Nat
  /-- Each match has two possible outcomes -/
  match_outcomes : Nat → Bool

/-- The number of different possible prize orders in the tournament -/
def prizeOrderCount (t : BowlingTournament) : Nat :=
  2^t.num_matches

/-- Theorem stating that the number of different prize orders is 32 -/
theorem prize_order_count_is_32 (t : BowlingTournament) :
  prizeOrderCount t = 32 :=
by sorry

end NUMINAMATH_CALUDE_prize_order_count_is_32_l422_42282


namespace NUMINAMATH_CALUDE_arc_length_cos_curve_l422_42286

/-- The length of the arc of the curve ρ = 6 cos φ from φ = 0 to φ = π/3 is equal to 2π. -/
theorem arc_length_cos_curve (φ : Real) :
  let ρ : Real → Real := λ φ ↦ 6 * Real.cos φ
  let L : Real := ∫ φ in (0)..(π/3), Real.sqrt ((ρ φ)^2 + (deriv ρ φ)^2)
  L = 2 * π :=
by
  sorry

end NUMINAMATH_CALUDE_arc_length_cos_curve_l422_42286


namespace NUMINAMATH_CALUDE_work_hours_constant_l422_42267

/-- Represents the work schedule for a week -/
structure WorkSchedule where
  days_per_week : ℕ
  initial_hours_task1 : ℕ
  initial_hours_task2 : ℕ
  hours_reduction_task1 : ℕ

/-- Calculates the total weekly work hours -/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  schedule.days_per_week * (schedule.initial_hours_task1 + schedule.initial_hours_task2)

/-- Theorem stating that the total weekly work hours remain constant after redistribution -/
theorem work_hours_constant (schedule : WorkSchedule) 
  (h1 : schedule.days_per_week = 5)
  (h2 : schedule.initial_hours_task1 = 5)
  (h3 : schedule.initial_hours_task2 = 3)
  (h4 : schedule.hours_reduction_task1 = 5) :
  total_weekly_hours schedule = 40 := by
  sorry

#eval total_weekly_hours { days_per_week := 5, initial_hours_task1 := 5, initial_hours_task2 := 3, hours_reduction_task1 := 5 }

end NUMINAMATH_CALUDE_work_hours_constant_l422_42267


namespace NUMINAMATH_CALUDE_rotate90_matches_optionC_l422_42214

-- Define the plane
def Plane : Type := ℝ × ℝ

-- Define the X-like shape
def XLikeShape : Type := Set Plane

-- Define rotation function
def rotate90Clockwise (shape : XLikeShape) : XLikeShape := sorry

-- Define the original shape
def originalShape : XLikeShape := sorry

-- Define the shape in option C
def optionCShape : XLikeShape := sorry

-- Theorem statement
theorem rotate90_matches_optionC : 
  rotate90Clockwise originalShape = optionCShape := by sorry

end NUMINAMATH_CALUDE_rotate90_matches_optionC_l422_42214


namespace NUMINAMATH_CALUDE_jane_mean_score_l422_42210

def jane_scores : List ℝ := [85, 88, 90, 92, 95, 100]

theorem jane_mean_score : 
  (jane_scores.sum / jane_scores.length : ℝ) = 550 / 6 := by
  sorry

end NUMINAMATH_CALUDE_jane_mean_score_l422_42210


namespace NUMINAMATH_CALUDE_union_of_sets_l422_42251

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 4, 6}
  A ∪ B = {1, 2, 4, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l422_42251


namespace NUMINAMATH_CALUDE_third_chapter_pages_l422_42295

/-- A book with three chapters -/
structure Book where
  chapter1 : ℕ
  chapter2 : ℕ
  chapter3 : ℕ

/-- The theorem stating the number of pages in the third chapter -/
theorem third_chapter_pages (b : Book) 
  (h1 : b.chapter1 = 35)
  (h2 : b.chapter2 = 18)
  (h3 : b.chapter2 = b.chapter3 + 15) :
  b.chapter3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_chapter_pages_l422_42295


namespace NUMINAMATH_CALUDE_jerry_feathers_left_l422_42213

def feathers_left (hawk_feathers : ℕ) (eagle_ratio : ℕ) (given_away : ℕ) : ℕ :=
  let total_feathers := hawk_feathers + eagle_ratio * hawk_feathers
  let remaining_after_gift := total_feathers - given_away
  remaining_after_gift / 2

theorem jerry_feathers_left : feathers_left 6 17 10 = 49 := by
  sorry

end NUMINAMATH_CALUDE_jerry_feathers_left_l422_42213


namespace NUMINAMATH_CALUDE_disjunction_truth_implication_false_l422_42216

theorem disjunction_truth_implication_false : 
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by sorry

end NUMINAMATH_CALUDE_disjunction_truth_implication_false_l422_42216


namespace NUMINAMATH_CALUDE_price_change_after_four_years_l422_42231

theorem price_change_after_four_years (initial_price : ℝ) :
  let price_after_two_increases := initial_price * (1 + 0.2)^2
  let final_price := price_after_two_increases * (1 - 0.2)^2
  final_price = initial_price * (1 - 0.0784) :=
by sorry

end NUMINAMATH_CALUDE_price_change_after_four_years_l422_42231


namespace NUMINAMATH_CALUDE_tiling_condition_l422_42271

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a square -/
inductive Color
  | Black
  | White

/-- Determines the color of a square based on its position -/
def squareColor (s : Square) : Color :=
  if (s.row.val + s.col.val) % 2 = 0 then Color.Black else Color.White

/-- Represents a chessboard with two squares removed -/
structure ChessboardWithRemovedSquares where
  removed1 : Square
  removed2 : Square
  different : removed1 ≠ removed2

/-- Represents the possibility of tiling the chessboard with dominoes -/
def canTile (board : ChessboardWithRemovedSquares) : Prop :=
  squareColor board.removed1 ≠ squareColor board.removed2

/-- Theorem stating the condition for possible tiling -/
theorem tiling_condition (board : ChessboardWithRemovedSquares) :
  canTile board ↔ squareColor board.removed1 ≠ squareColor board.removed2 := by sorry

end NUMINAMATH_CALUDE_tiling_condition_l422_42271


namespace NUMINAMATH_CALUDE_inequality_proof_l422_42277

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l422_42277


namespace NUMINAMATH_CALUDE_tangent_line_min_sum_l422_42276

noncomputable def f (x : ℝ) := x - Real.exp (-x)

theorem tangent_line_min_sum (m n : ℝ) :
  (∃ t : ℝ, (f t = m * t + n) ∧ 
    (∀ x : ℝ, f x ≤ m * x + n)) →
  m + n ≥ 1 - 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_min_sum_l422_42276


namespace NUMINAMATH_CALUDE_smooth_transition_iff_tangent_l422_42211

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define tangency
def isTangent (c : Circle) (l : Line) (p : Point) : Prop :=
  -- The point lies on both the circle and the line
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  p.2 = l.slope * p.1 + l.intercept ∧
  -- The line is perpendicular to the radius at the point of tangency
  l.slope * (p.1 - c.center.1) = -(p.2 - c.center.2)

-- Define smooth transition
def smoothTransition (c : Circle) (l : Line) (p : Point) : Prop :=
  -- The velocity vector is continuous at the transition point
  isTangent c l p

-- Theorem statement
theorem smooth_transition_iff_tangent (c : Circle) (l : Line) (p : Point) :
  smoothTransition c l p ↔ isTangent c l p :=
sorry

end NUMINAMATH_CALUDE_smooth_transition_iff_tangent_l422_42211


namespace NUMINAMATH_CALUDE_solve_for_y_l422_42250

theorem solve_for_y (x y : ℝ) (h1 : 2 * (x - y) = 32) (h2 : x + y = -4) : y = -10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l422_42250


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l422_42262

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

-- Theorem statement
theorem sixth_term_of_geometric_sequence 
  (a₁ a₂ : ℝ) 
  (h₁ : a₁ = 5) 
  (h₂ : a₂ = 15) : 
  geometric_sequence a₁ (a₂ / a₁) 6 = 1215 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l422_42262


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l422_42220

theorem imaginary_part_of_complex_fraction (i : ℂ) : i * i = -1 → Complex.im ((1 + i) / (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l422_42220


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_sin_2beta_zero_l422_42236

theorem sin_2alpha_plus_sin_2beta_zero (α β : ℝ) 
  (h : Real.sin α * Real.sin β + Real.cos α * Real.cos β = 0) : 
  Real.sin (2 * α) + Real.sin (2 * β) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_sin_2beta_zero_l422_42236


namespace NUMINAMATH_CALUDE_cos_660_degrees_l422_42293

theorem cos_660_degrees : Real.cos (660 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_660_degrees_l422_42293


namespace NUMINAMATH_CALUDE_park_conditions_l422_42221

-- Define the conditions
def temperature_at_least_75 : Prop := sorry
def sunny : Prop := sorry
def park_clean : Prop := sorry
def park_crowded : Prop := sorry

-- Define the main theorem
theorem park_conditions :
  (temperature_at_least_75 ∧ sunny ∧ park_clean → park_crowded) →
  (¬park_crowded → ¬temperature_at_least_75 ∨ ¬sunny ∨ ¬park_clean) :=
by sorry

end NUMINAMATH_CALUDE_park_conditions_l422_42221


namespace NUMINAMATH_CALUDE_function_value_comparison_l422_42245

theorem function_value_comparison (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x = x^2 + 2*x*(deriv f 2)) : f (-1) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_comparison_l422_42245


namespace NUMINAMATH_CALUDE_unique_three_digit_number_square_equals_sum_of_digits_power_five_l422_42297

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 243 is the only three-digit number whose square 
    is equal to the sum of its digits raised to the power of 5 -/
theorem unique_three_digit_number_square_equals_sum_of_digits_power_five : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n^2 = (sum_of_digits n)^5 := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_square_equals_sum_of_digits_power_five_l422_42297


namespace NUMINAMATH_CALUDE_equation_solution_l422_42292

theorem equation_solution : 
  ∀ x : ℝ, 4 * x^2 - (x^2 - 2*x + 1) = 0 ↔ x = 1/3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l422_42292


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l422_42254

theorem sufficient_not_necessary (a b : ℝ) :
  (a > b ∧ b > 0) → (1 / a^2 < 1 / b^2) ∧
  ∃ (x y : ℝ), (1 / x^2 < 1 / y^2) ∧ ¬(x > y ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l422_42254


namespace NUMINAMATH_CALUDE_relationship_abcd_l422_42274

theorem relationship_abcd (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abcd_l422_42274


namespace NUMINAMATH_CALUDE_lifeguard_swimming_test_l422_42228

/-- Lifeguard swimming test problem -/
theorem lifeguard_swimming_test
  (total_distance : ℝ)
  (front_crawl_speed : ℝ)
  (total_time : ℝ)
  (front_crawl_time : ℝ)
  (h1 : total_distance = 500)
  (h2 : front_crawl_speed = 45)
  (h3 : total_time = 12)
  (h4 : front_crawl_time = 8) :
  let front_crawl_distance := front_crawl_speed * front_crawl_time
  let breaststroke_distance := total_distance - front_crawl_distance
  let breaststroke_time := total_time - front_crawl_time
  let breaststroke_speed := breaststroke_distance / breaststroke_time
  breaststroke_speed = 35 := by
sorry


end NUMINAMATH_CALUDE_lifeguard_swimming_test_l422_42228


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l422_42296

/-- Given that koalas absorb 30% of the fiber they eat, 
    prove that if a koala absorbed 12 ounces of fiber, 
    it ate 40 ounces of fiber. -/
theorem koala_fiber_consumption 
  (absorption_rate : ℝ) 
  (absorbed_amount : ℝ) 
  (h1 : absorption_rate = 0.3)
  (h2 : absorbed_amount = 12) :
  absorbed_amount / absorption_rate = 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_consumption_l422_42296


namespace NUMINAMATH_CALUDE_product_of_group_at_least_72_l422_42208

theorem product_of_group_at_least_72 (group1 group2 group3 : List Nat) : 
  (group1 ++ group2 ++ group3).toFinset = Finset.range 9 →
  (group1.prod ≥ 72) ∨ (group2.prod ≥ 72) ∨ (group3.prod ≥ 72) := by
  sorry

end NUMINAMATH_CALUDE_product_of_group_at_least_72_l422_42208


namespace NUMINAMATH_CALUDE_exists_four_mutually_acquainted_l422_42287

/-- Represents the acquaintance relation between people --/
def Acquainted (n : ℕ) := Fin n → Fin n → Prop

/-- The property that among every 3 people, at least 2 are acquainted --/
def AtLeastTwoAcquainted (n : ℕ) (acq : Acquainted n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    acq a b ∨ acq b c ∨ acq a c

/-- A subset of 4 mutually acquainted people --/
def FourMutuallyAcquainted (n : ℕ) (acq : Acquainted n) : Prop :=
  ∃ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    acq a b ∧ acq a c ∧ acq a d ∧ acq b c ∧ acq b d ∧ acq c d

/-- The main theorem --/
theorem exists_four_mutually_acquainted :
  ∀ (acq : Acquainted 9),
    AtLeastTwoAcquainted 9 acq →
    FourMutuallyAcquainted 9 acq :=
by
  sorry


end NUMINAMATH_CALUDE_exists_four_mutually_acquainted_l422_42287


namespace NUMINAMATH_CALUDE_student_survey_l422_42255

theorem student_survey (french_and_english : ℕ) (french_not_english : ℕ) 
  (percent_not_french : ℚ) :
  french_and_english = 25 →
  french_not_english = 65 →
  percent_not_french = 55/100 →
  french_and_english + french_not_english = (100 : ℚ) / (100 - percent_not_french) * 100 :=
by sorry

end NUMINAMATH_CALUDE_student_survey_l422_42255


namespace NUMINAMATH_CALUDE_complex_exponentiation_205_deg_72_l422_42235

theorem complex_exponentiation_205_deg_72 :
  (Complex.exp (205 * π / 180 * Complex.I)) ^ 72 = -1/2 - Complex.I * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponentiation_205_deg_72_l422_42235


namespace NUMINAMATH_CALUDE_prob_two_white_balls_prob_one_white_one_black_l422_42215

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 4

/-- Represents the number of black balls in the bag -/
def black_balls : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- Calculates the probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Theorem stating the probability of drawing two white balls -/
theorem prob_two_white_balls : 
  probability (white_balls.choose 2) (total_balls.choose 2) = 2 / 5 := by sorry

/-- Theorem stating the probability of drawing one white ball and one black ball -/
theorem prob_one_white_one_black : 
  probability (white_balls * black_balls) (total_balls.choose 2) = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_prob_two_white_balls_prob_one_white_one_black_l422_42215


namespace NUMINAMATH_CALUDE_coloring_books_total_l422_42265

theorem coloring_books_total (initial : ℕ) (given_away : ℕ) (bought : ℕ) : 
  initial = 34 → given_away = 3 → bought = 48 → 
  initial - given_away + bought = 79 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_total_l422_42265


namespace NUMINAMATH_CALUDE_sqrt_comparison_l422_42232

theorem sqrt_comparison : Real.sqrt 10 - Real.sqrt 6 < Real.sqrt 7 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l422_42232


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l422_42280

theorem rectangle_width_decrease (L W : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let new_L := 1.5 * L
  let new_W := W * (L / new_L)
  (W - new_W) / W = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l422_42280


namespace NUMINAMATH_CALUDE_carla_leaf_collection_l422_42239

/-- Represents the number of items Carla needs to collect each day -/
def daily_items : ℕ := 5

/-- Represents the number of days Carla has to collect items -/
def total_days : ℕ := 10

/-- Represents the number of bugs Carla needs to collect -/
def bugs_to_collect : ℕ := 20

/-- Calculates the total number of items Carla needs to collect -/
def total_items : ℕ := daily_items * total_days

/-- Calculates the number of leaves Carla needs to collect -/
def leaves_to_collect : ℕ := total_items - bugs_to_collect

theorem carla_leaf_collection :
  leaves_to_collect = 30 := by sorry

end NUMINAMATH_CALUDE_carla_leaf_collection_l422_42239


namespace NUMINAMATH_CALUDE_corn_acreage_l422_42207

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l422_42207


namespace NUMINAMATH_CALUDE_max_basketballs_l422_42204

/-- The cost of footballs and basketballs -/
structure BallCosts where
  football : ℕ
  basketball : ℕ

/-- The problem setup -/
structure ProblemSetup where
  costs : BallCosts
  total_balls : ℕ
  max_cost : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (setup : ProblemSetup) : Prop :=
  3 * setup.costs.football + 2 * setup.costs.basketball = 310 ∧
  2 * setup.costs.football + 5 * setup.costs.basketball = 500 ∧
  setup.total_balls = 96 ∧
  setup.max_cost = 5800

/-- The theorem to prove -/
theorem max_basketballs (setup : ProblemSetup) 
  (h : satisfies_conditions setup) : 
  ∃ (x : ℕ), x ≤ setup.total_balls ∧ 
    x * setup.costs.basketball + (setup.total_balls - x) * setup.costs.football ≤ setup.max_cost ∧
    ∀ (y : ℕ), y > x → 
      y * setup.costs.basketball + (setup.total_balls - y) * setup.costs.football > setup.max_cost :=
by
  sorry

end NUMINAMATH_CALUDE_max_basketballs_l422_42204


namespace NUMINAMATH_CALUDE_smallest_valid_number_l422_42253

def is_valid_number (n : ℕ) : Prop :=
  ∃ (chosen : Finset ℕ) (unchosen : Finset ℕ),
    chosen.card = 5 ∧
    unchosen.card = 4 ∧
    chosen ∪ unchosen = Finset.range 9 ∧
    chosen ∩ unchosen = ∅ ∧
    (∀ d ∈ chosen, n % d = 0) ∧
    (∀ d ∈ unchosen, n % d ≠ 0) ∧
    n ≥ 10000 ∧ n < 100000

theorem smallest_valid_number :
  ∃ (n : ℕ), is_valid_number n ∧ 
  (∀ m, is_valid_number m → n ≤ m) ∧
  n = 14728 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l422_42253


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l422_42227

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 40*x^2 + 144 = 0 ∧ 
  (∀ (y : ℝ), y^4 - 40*y^2 + 144 = 0 → x ≤ y) ∧
  x = -6 := by
sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l422_42227


namespace NUMINAMATH_CALUDE_bruce_bank_savings_l422_42200

/-- The amount of money Bruce puts in the bank -/
def money_in_bank (aunt_money grandfather_money : ℕ) : ℚ :=
  (aunt_money + grandfather_money : ℚ) / 5

/-- Theorem stating the amount Bruce put in the bank -/
theorem bruce_bank_savings :
  money_in_bank 75 150 = 45 := by sorry

end NUMINAMATH_CALUDE_bruce_bank_savings_l422_42200


namespace NUMINAMATH_CALUDE_difference_of_squares_problem_solution_l422_42223

theorem difference_of_squares (k : ℝ) : 
  (5 + k) * (5 - k) = 5^2 - k^2 := by sorry

theorem problem_solution : 
  ∃ n : ℝ, (5 + 2) * (5 - 2) = 5^2 - n ∧ n = 2^2 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_problem_solution_l422_42223


namespace NUMINAMATH_CALUDE_function_symmetry_and_piecewise_l422_42237

theorem function_symmetry_and_piecewise (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x > 0, f x = x * |x - 2|) :
  ∀ x < 0, f x = x * |x + 2| := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_and_piecewise_l422_42237


namespace NUMINAMATH_CALUDE_share_ratio_B_to_C_l422_42218

def total_amount : ℕ := 510
def share_A : ℕ := 360
def share_B : ℕ := 90
def share_C : ℕ := 60

theorem share_ratio_B_to_C : 
  (share_B : ℚ) / (share_C : ℚ) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_B_to_C_l422_42218


namespace NUMINAMATH_CALUDE_max_sphere_radius_in_intersecting_cones_l422_42202

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum radius of a sphere that can fit within two intersecting cones -/
def maxSphereRadius (ic : IntersectingCones) : ℝ := sorry

/-- Theorem stating the maximum sphere radius for the given configuration -/
theorem max_sphere_radius_in_intersecting_cones :
  let ic : IntersectingCones := {
    cone1 := { baseRadius := 5, height := 12 },
    cone2 := { baseRadius := 5, height := 12 },
    intersectionDistance := 4
  }
  maxSphereRadius ic = 40 / 13 := by sorry

end NUMINAMATH_CALUDE_max_sphere_radius_in_intersecting_cones_l422_42202


namespace NUMINAMATH_CALUDE_charity_raffle_winnings_l422_42257

theorem charity_raffle_winnings (W : ℝ) : 
  W / 2 - 2 = 55 → W = 114 := by
  sorry

end NUMINAMATH_CALUDE_charity_raffle_winnings_l422_42257


namespace NUMINAMATH_CALUDE_midpoint_theorem_l422_42234

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -4), 
    prove that the other endpoint is (-1, 6) -/
theorem midpoint_theorem :
  let midpoint : ℝ × ℝ := (3, 1)
  let endpoint1 : ℝ × ℝ := (7, -4)
  let endpoint2 : ℝ × ℝ := (-1, 6)
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_theorem_l422_42234


namespace NUMINAMATH_CALUDE_segment_length_l422_42291

/-- Given a line segment AB divided by points P and Q, prove that the length of AB is 135/7 -/
theorem segment_length (A B P Q : ℝ) : 
  (∃ x y : ℝ, 
    A < P ∧ P < Q ∧ Q < B ∧  -- P and Q are between A and B
    P - A = 3*x ∧ B - P = 2*x ∧  -- P divides AB in ratio 3:2
    Q - A = 4*y ∧ B - Q = 5*y ∧  -- Q divides AB in ratio 4:5
    Q - P = 3)  -- Distance between P and Q is 3
  → B - A = 135/7 := by
sorry

end NUMINAMATH_CALUDE_segment_length_l422_42291


namespace NUMINAMATH_CALUDE_f_increasing_and_no_negative_roots_l422_42264

noncomputable section

variable (a : ℝ) (h : a > 1)

def f (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem f_increasing_and_no_negative_roots :
  (∀ x y, -1 < x ∧ x < y → f a x < f a y) ∧
  (∀ x, x < 0 → f a x ≠ 0) := by sorry

end NUMINAMATH_CALUDE_f_increasing_and_no_negative_roots_l422_42264


namespace NUMINAMATH_CALUDE_find_b_l422_42281

/-- Given two functions f and g, and a condition on their composition, prove the value of b. -/
theorem find_b (f g : ℝ → ℝ) (b : ℝ) 
  (hf : ∀ x, f x = (3 * x) / 7 + 4)
  (hg : ∀ x, g x = 5 - 2 * x)
  (h_comp : f (g b) = 10) :
  b = -4.5 := by sorry

end NUMINAMATH_CALUDE_find_b_l422_42281


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l422_42273

theorem polynomial_identity_sum (d₁ d₂ d₃ e₁ e₂ e₃ : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^6 + x^4 - x^2 + 1 = 
    (x^2 + d₁*x + e₁) * (x^2 + d₂*x + e₂) * (x^2 + d₃*x + e₃) * (x^2 + 1)) :
  d₁*e₁ + d₂*e₂ + d₃*e₃ = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l422_42273


namespace NUMINAMATH_CALUDE_marble_selection_probability_l422_42294

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 3

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 3

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles

/-- The number of marbles to be selected -/
def selected_marbles : ℕ := 4

/-- The probability of selecting exactly one marble of each color, with one color being chosen twice -/
theorem marble_selection_probability :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 2) /
  Nat.choose total_marbles selected_marbles = 9 / 14 :=
sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l422_42294


namespace NUMINAMATH_CALUDE_trigonometric_expression_simplification_l422_42206

theorem trigonometric_expression_simplification :
  let expr := (Real.sin (10 * π / 180) + Real.sin (20 * π / 180) + 
               Real.sin (30 * π / 180) + Real.sin (40 * π / 180) + 
               Real.sin (50 * π / 180) + Real.sin (60 * π / 180) + 
               Real.sin (70 * π / 180) + Real.sin (80 * π / 180)) / 
              (Real.cos (5 * π / 180) * Real.cos (10 * π / 180) * Real.cos (20 * π / 180))
  expr = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_simplification_l422_42206


namespace NUMINAMATH_CALUDE_clock_angle_at_3_37_clock_angle_proof_l422_42272

/-- The acute angle between clock hands at 3:37 -/
theorem clock_angle_at_3_37 : ℝ :=
  let hours : ℕ := 3
  let minutes : ℕ := 37
  let total_hours : ℕ := 12
  let degrees_per_hour : ℝ := 30

  let minute_angle : ℝ := (minutes : ℝ) / 60 * 360
  let hour_angle : ℝ := (hours : ℝ) * degrees_per_hour + (minutes : ℝ) / 60 * degrees_per_hour

  let angle_diff : ℝ := |minute_angle - hour_angle|
  let acute_angle : ℝ := min angle_diff (360 - angle_diff)

  113.5

/-- Proof of the clock angle theorem -/
theorem clock_angle_proof : clock_angle_at_3_37 = 113.5 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_37_clock_angle_proof_l422_42272


namespace NUMINAMATH_CALUDE_money_sum_l422_42205

theorem money_sum (a b : ℝ) (h1 : (3/10) * a = (1/5) * b) (h2 : b = 60) : a + b = 100 := by
  sorry

end NUMINAMATH_CALUDE_money_sum_l422_42205


namespace NUMINAMATH_CALUDE_exists_monochromatic_equilateral_triangle_l422_42219

/-- A color type representing red or blue -/
inductive Color
  | Red
  | Blue

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A coloring function that assigns a color to each point in the plane -/
def Coloring := Point → Color

/-- Predicate to check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p1 p2 p3 : Point) : Prop := sorry

/-- Theorem stating that in any coloring of the plane, there exist three points
    of the same color forming an equilateral triangle -/
theorem exists_monochromatic_equilateral_triangle (c : Coloring) :
  ∃ (p1 p2 p3 : Point) (col : Color),
    c p1 = col ∧ c p2 = col ∧ c p3 = col ∧
    IsEquilateralTriangle p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_equilateral_triangle_l422_42219


namespace NUMINAMATH_CALUDE_interest_difference_l422_42241

def initial_amount : ℝ := 1250

def compound_rate_year1 : ℝ := 0.08
def compound_rate_year2 : ℝ := 0.10
def compound_rate_year3 : ℝ := 0.12

def simple_rate_year1 : ℝ := 0.04
def simple_rate_year2 : ℝ := 0.06
def simple_rate_year3 : ℝ := 0.07
def simple_rate_year4 : ℝ := 0.09

def compound_interest (principal : ℝ) (rate1 rate2 rate3 : ℝ) : ℝ :=
  principal * (1 + rate1) * (1 + rate2) * (1 + rate3)

def simple_interest (principal : ℝ) (rate1 rate2 rate3 rate4 : ℝ) : ℝ :=
  principal * (1 + rate1 + rate2 + rate3 + rate4)

theorem interest_difference :
  compound_interest initial_amount compound_rate_year1 compound_rate_year2 compound_rate_year3 -
  simple_interest initial_amount simple_rate_year1 simple_rate_year2 simple_rate_year3 simple_rate_year4 = 88.2 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l422_42241


namespace NUMINAMATH_CALUDE_line_through_P_with_equal_intercepts_l422_42203

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space by its equation ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

-- The given point P(2,3)
def P : Point2D := ⟨2, 3⟩

-- The two possible lines
def line1 : Line2D := ⟨3, -2, 0⟩
def line2 : Line2D := ⟨1, 1, -5⟩

-- The theorem to prove
theorem line_through_P_with_equal_intercepts :
  (pointOnLine P line1 ∧ equalIntercepts line1) ∨
  (pointOnLine P line2 ∧ equalIntercepts line2) := by
  sorry

end NUMINAMATH_CALUDE_line_through_P_with_equal_intercepts_l422_42203


namespace NUMINAMATH_CALUDE_a2_value_l422_42246

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 2

-- Define the geometric sequence property for a_1, a_2, and a_5
def geometric_property (a : ℕ → ℝ) : Prop :=
  (a 2 / a 1) = (a 5 / a 2)

-- Theorem statement
theorem a2_value (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_geom : geometric_property a) : 
  a 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a2_value_l422_42246


namespace NUMINAMATH_CALUDE_academy_skills_l422_42288

theorem academy_skills (total : ℕ) (dancers : ℕ) (calligraphers : ℕ) (both : ℕ) : 
  total = 120 → 
  dancers = 88 → 
  calligraphers = 32 → 
  both = 18 → 
  total - (dancers + calligraphers - both) = 18 := by
sorry

end NUMINAMATH_CALUDE_academy_skills_l422_42288


namespace NUMINAMATH_CALUDE_air_conditioner_costs_and_minimum_cost_l422_42209

/-- Represents the cost and quantity of air conditioners -/
structure AirConditioner :=
  (costA : ℕ) -- Cost of type A
  (costB : ℕ) -- Cost of type B
  (quantityA : ℕ) -- Quantity of type A
  (quantityB : ℕ) -- Quantity of type B

/-- Conditions for air conditioner purchase -/
def satisfiesConditions (ac : AirConditioner) : Prop :=
  ac.costA * 3 + ac.costB * 2 = 39000 ∧
  ac.costA * 4 = ac.costB * 5 + 6000 ∧
  ac.quantityA + ac.quantityB = 30 ∧
  ac.quantityA * 2 ≥ ac.quantityB ∧
  ac.costA * ac.quantityA + ac.costB * ac.quantityB ≤ 217000

/-- Total cost of air conditioners -/
def totalCost (ac : AirConditioner) : ℕ :=
  ac.costA * ac.quantityA + ac.costB * ac.quantityB

/-- Theorem stating the correct costs and minimum total cost -/
theorem air_conditioner_costs_and_minimum_cost :
  ∃ (ac : AirConditioner),
    satisfiesConditions ac ∧
    ac.costA = 9000 ∧
    ac.costB = 6000 ∧
    (∀ (ac' : AirConditioner), satisfiesConditions ac' → totalCost ac ≤ totalCost ac') ∧
    totalCost ac = 210000 :=
  sorry

end NUMINAMATH_CALUDE_air_conditioner_costs_and_minimum_cost_l422_42209


namespace NUMINAMATH_CALUDE_infinitely_many_consecutive_right_triangles_l422_42290

/-- A right triangle with integer sides where the hypotenuse and one side are consecutive. -/
structure ConsecutiveRightTriangle where
  a : ℕ  -- One side of the triangle
  b : ℕ  -- The other side of the triangle
  c : ℕ  -- The hypotenuse
  consecutive : c = a + 1
  pythagorean : a^2 + b^2 = c^2

/-- There exist infinitely many ConsecutiveRightTriangles. -/
theorem infinitely_many_consecutive_right_triangles :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ∃ t : ConsecutiveRightTriangle, t.c = m :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_consecutive_right_triangles_l422_42290


namespace NUMINAMATH_CALUDE_f_of_one_l422_42226

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_of_one (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 4)
  (h_value : f (-5) = 1) : 
  f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_l422_42226


namespace NUMINAMATH_CALUDE_log_sum_greater_than_exp_l422_42242

theorem log_sum_greater_than_exp (x : ℝ) (h : x < 0) :
  Real.log 2 + Real.log 5 > Real.exp x := by sorry

end NUMINAMATH_CALUDE_log_sum_greater_than_exp_l422_42242


namespace NUMINAMATH_CALUDE_elevator_problem_l422_42244

theorem elevator_problem (x y z w v : ℕ) (h : x = 15 ∧ y = 9 ∧ z = 12 ∧ w = 6 ∧ v = 10) :
  x - y + z - w + v = 28 :=
by sorry

end NUMINAMATH_CALUDE_elevator_problem_l422_42244


namespace NUMINAMATH_CALUDE_probability_neither_mix_l422_42284

/-- Represents the set of buyers -/
def Buyers : Type := Unit

/-- The total number of buyers -/
def total_buyers : ℕ := 100

/-- The number of buyers who purchase cake mix -/
def cake_mix_buyers : ℕ := 50

/-- The number of buyers who purchase muffin mix -/
def muffin_mix_buyers : ℕ := 40

/-- The number of buyers who purchase both cake mix and muffin mix -/
def both_mix_buyers : ℕ := 15

/-- The probability of selecting a buyer who purchases neither cake mix nor muffin mix -/
theorem probability_neither_mix (b : Buyers) : 
  (total_buyers - (cake_mix_buyers + muffin_mix_buyers - both_mix_buyers)) / total_buyers = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_mix_l422_42284


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3328_l422_42238

theorem largest_prime_factor_of_3328 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 3328 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 3328 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3328_l422_42238


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_two_l422_42283

theorem points_three_units_from_negative_two (x : ℝ) : 
  (x = 1 ∨ x = -5) ↔ |x + 2| = 3 :=
by sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_two_l422_42283


namespace NUMINAMATH_CALUDE_problem_solution_l422_42230

theorem problem_solution (a b : ℝ) 
  (h1 : a * b = 2 * (a + b) + 14) 
  (h2 : b - a = 3) : 
  b = 8 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l422_42230


namespace NUMINAMATH_CALUDE_test_questions_missed_l422_42263

theorem test_questions_missed (T : ℕ) (X Y : ℝ) : 
  T > 0 → 
  0 ≤ X ∧ X ≤ 100 →
  0 ≤ Y ∧ Y ≤ 100 →
  ∃ (M F : ℕ),
    M = 5 * F ∧
    M + F = 216 ∧
    M = T * (1 - X / 100) ∧
    F = T * (1 - Y / 100) →
  M = 180 := by
sorry

end NUMINAMATH_CALUDE_test_questions_missed_l422_42263


namespace NUMINAMATH_CALUDE_geometric_sequences_l422_42222

def is_geometric (a : ℕ → ℝ) : Prop := ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequences (a : ℕ → ℝ) (h : is_geometric a) :
  (is_geometric (λ n => 1 / a n)) ∧ (is_geometric (λ n => a n * a (n + 1))) := by sorry

end NUMINAMATH_CALUDE_geometric_sequences_l422_42222


namespace NUMINAMATH_CALUDE_commission_for_8000_l422_42256

/-- Represents the commission structure of a bank -/
structure BankCommission where
  /-- Fixed fee for any withdrawal -/
  fixed_fee : ℝ
  /-- Proportional fee rate for withdrawal amount -/
  prop_rate : ℝ

/-- Calculates the commission for a given withdrawal amount -/
def calculate_commission (bc : BankCommission) (amount : ℝ) : ℝ :=
  bc.fixed_fee + bc.prop_rate * amount

theorem commission_for_8000 :
  ∀ (bc : BankCommission),
    calculate_commission bc 5000 = 110 →
    calculate_commission bc 11000 = 230 →
    calculate_commission bc 8000 = 170 := by
  sorry

end NUMINAMATH_CALUDE_commission_for_8000_l422_42256


namespace NUMINAMATH_CALUDE_new_student_weight_l422_42298

/-- Given a group of students and their weights, calculate the weight of a new student
    that changes the average weight of the group. -/
theorem new_student_weight
  (initial_count : ℕ)
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (h1 : initial_count = 19)
  (h2 : initial_avg = 15)
  (h3 : new_avg = 14.8) :
  (initial_count + 1) * new_avg - initial_count * initial_avg = 11 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l422_42298


namespace NUMINAMATH_CALUDE_loop_condition_proof_l422_42266

theorem loop_condition_proof (i₀ S₀ : ℕ) (result : ℕ) : 
  i₀ = 12 → S₀ = 1 → result = 11880 →
  (∃ n : ℕ, result = Nat.factorial n - Nat.factorial (i₀ - 1)) →
  (∀ i S : ℕ, i > 9 ↔ result = S ∧ S = Nat.factorial i - Nat.factorial (i₀ - 1)) :=
by sorry

end NUMINAMATH_CALUDE_loop_condition_proof_l422_42266


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l422_42261

-- Define sets A and B
def A : Set ℝ := {x | 2 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < 3 * a}

-- Theorem 1
theorem union_condition (a : ℝ) : A ∪ B a = {x | 2 < x ∧ x < 6} → a = 2 := by
  sorry

-- Theorem 2
theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → 2/3 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l422_42261


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l422_42259

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 0 → |x| > 0) ∧ (∃ x : ℝ, |x| > 0 ∧ x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l422_42259


namespace NUMINAMATH_CALUDE_soup_feeding_theorem_l422_42279

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remaining_adults_fed (total_cans : ℕ) (can_capacity : SoupCan) (children_fed : ℕ) : ℕ :=
  let cans_used_for_children := children_fed / can_capacity.children
  let remaining_cans := total_cans - cans_used_for_children
  remaining_cans * can_capacity.adults

/-- Theorem: Given 7 cans of soup, where each can feeds 4 adults or 7 children,
    if 21 children are fed, the remaining soup can feed 16 adults -/
theorem soup_feeding_theorem :
  let can_capacity : SoupCan := { adults := 4, children := 7 }
  let total_cans : ℕ := 7
  let children_fed : ℕ := 21
  remaining_adults_fed total_cans can_capacity children_fed = 16 := by
  sorry


end NUMINAMATH_CALUDE_soup_feeding_theorem_l422_42279


namespace NUMINAMATH_CALUDE_union_M_N_complement_intersection_M_N_l422_42247

-- Define the universal set U
def U : Set ℝ := {x | -6 ≤ x ∧ x ≤ 5}

-- Define set M
def M : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem for M ∪ N
theorem union_M_N : M ∪ N = {x | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for ∁U(M ∩ N)
theorem complement_intersection_M_N : 
  (M ∩ N)ᶜ = {x ∈ U | x ≤ 0 ∨ 2 ≤ x} := by sorry

end NUMINAMATH_CALUDE_union_M_N_complement_intersection_M_N_l422_42247


namespace NUMINAMATH_CALUDE_range_of_f_l422_42278

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x + 3) / (x + 2)

theorem range_of_f :
  ∀ y : ℝ, ∃ x : ℝ, x ≠ -2 ∧ f x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l422_42278


namespace NUMINAMATH_CALUDE_three_Z_five_equals_32_l422_42249

/-- The Z operation as defined in the problem -/
def Z (a b : ℝ) : ℝ := b + 12 * a - a^2

/-- Theorem stating that 3 Z 5 equals 32 -/
theorem three_Z_five_equals_32 : Z 3 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_five_equals_32_l422_42249


namespace NUMINAMATH_CALUDE_plate_selection_probability_l422_42285

def red_plates : ℕ := 6
def light_blue_plates : ℕ := 3
def dark_blue_plates : ℕ := 3

def total_plates : ℕ := red_plates + light_blue_plates + dark_blue_plates

def favorable_outcomes : ℕ := 
  (red_plates.choose 2) + 
  (light_blue_plates.choose 2) + 
  (dark_blue_plates.choose 2) + 
  (light_blue_plates * dark_blue_plates)

def total_outcomes : ℕ := total_plates.choose 2

theorem plate_selection_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 11 := by sorry

end NUMINAMATH_CALUDE_plate_selection_probability_l422_42285


namespace NUMINAMATH_CALUDE_log_comparison_l422_42217

theorem log_comparison (a : ℝ) (h : a > 1) : Real.log a / Real.log (a - 1) > Real.log (a + 1) / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l422_42217


namespace NUMINAMATH_CALUDE_prob_exactly_two_ones_value_l422_42269

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def target_outcome : ℕ := 1
def num_target : ℕ := 2

def prob_exactly_two_ones : ℚ :=
  (num_dice.choose num_target) *
  (1 / num_sides) ^ num_target *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_target)

theorem prob_exactly_two_ones_value :
  prob_exactly_two_ones = (66 * 5^10) / 6^12 := by
  sorry

end NUMINAMATH_CALUDE_prob_exactly_two_ones_value_l422_42269


namespace NUMINAMATH_CALUDE_triangle_properties_l422_42212

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  (t.a + t.c)^2 - t.b^2 = 3 * t.a * t.c ∧
  t.b = 6 ∧
  Real.sin t.C = 2 * Real.sin t.A

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 3 ∧ (1/2 * t.a * t.b : ℝ) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l422_42212


namespace NUMINAMATH_CALUDE_problem_2_l422_42240

theorem problem_2 (a : ℤ) (h : a = 67897) : a * (a + 1) - (a - 1) * (a + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_2_l422_42240


namespace NUMINAMATH_CALUDE_system_solution_ratio_l422_42248

theorem system_solution_ratio (a b x y : ℝ) : 
  8 * x - 5 * y = a →
  10 * y - 15 * x = b →
  x ≠ 0 →
  y ≠ 0 →
  b ≠ 0 →
  a / b = 8 / 15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l422_42248
