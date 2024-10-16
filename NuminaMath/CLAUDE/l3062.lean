import Mathlib

namespace NUMINAMATH_CALUDE_smaller_number_problem_l3062_306273

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) :
  min x y = 3 := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3062_306273


namespace NUMINAMATH_CALUDE_cosine_rule_triangle_l3062_306244

theorem cosine_rule_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  a = 3 ∧ b = 4 ∧ c = 6 → cos_B = 29 / 36 := by
  sorry

end NUMINAMATH_CALUDE_cosine_rule_triangle_l3062_306244


namespace NUMINAMATH_CALUDE_special_sequence_sum_2017_l3062_306267

/-- A sequence with special properties -/
def SpecialSequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → S (n + 1) - S n = 3^n / a n

/-- The sum of the first 2017 terms of the special sequence -/
theorem special_sequence_sum_2017 (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : SpecialSequence a S) : S 2017 = 3^1009 - 2 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_sum_2017_l3062_306267


namespace NUMINAMATH_CALUDE_dawn_hourly_income_l3062_306261

/-- Calculates the hourly income for Dawn's painting project -/
theorem dawn_hourly_income (num_paintings : ℕ) 
                            (sketch_time painting_time finish_time : ℝ) 
                            (watercolor_payment sketch_payment finish_payment : ℝ) : 
  num_paintings = 12 ∧ 
  sketch_time = 1.5 ∧ 
  painting_time = 2 ∧ 
  finish_time = 0.5 ∧
  watercolor_payment = 3600 ∧ 
  sketch_payment = 1200 ∧ 
  finish_payment = 300 → 
  (watercolor_payment + sketch_payment + finish_payment) / 
  (num_paintings * (sketch_time + painting_time + finish_time)) = 106.25 := by
sorry

end NUMINAMATH_CALUDE_dawn_hourly_income_l3062_306261


namespace NUMINAMATH_CALUDE_circle_polar_equation_l3062_306272

/-- The polar equation ρ = 2cosθ represents a circle with center at (1,0) and radius 1 -/
theorem circle_polar_equation :
  ∀ (ρ θ : ℝ), ρ = 2 * Real.cos θ ↔
  (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_polar_equation_l3062_306272


namespace NUMINAMATH_CALUDE_expression_evaluation_l3062_306277

theorem expression_evaluation (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (a*b + b*c + c*d + d*a + a*c + b*d)⁻¹ *
  ((a*b)⁻¹ + (b*c)⁻¹ + (c*d)⁻¹ + (d*a)⁻¹ + (a*c)⁻¹ + (b*d)⁻¹) = (a*b*c*d)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3062_306277


namespace NUMINAMATH_CALUDE_abby_emma_weight_l3062_306294

/-- The combined weight of two people given their individual weights -/
def combined_weight (w1 w2 : ℝ) : ℝ := w1 + w2

/-- Proves that Abby and Emma weigh 310 pounds together given the weights of pairs -/
theorem abby_emma_weight
  (a b c d e : ℝ)  -- Individual weights of Abby, Bart, Cindy, Damon, and Emma
  (h1 : combined_weight a b = 270)  -- Abby and Bart
  (h2 : combined_weight b c = 255)  -- Bart and Cindy
  (h3 : combined_weight c d = 280)  -- Cindy and Damon
  (h4 : combined_weight d e = 295)  -- Damon and Emma
  : combined_weight a e = 310 := by
  sorry

#check abby_emma_weight

end NUMINAMATH_CALUDE_abby_emma_weight_l3062_306294


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3062_306274

/-- Proves that for an arithmetic sequence with given properties, m = 2 when S_m is the arithmetic mean of a_m and a_{m+1} -/
theorem arithmetic_sequence_problem (m : ℕ) : 
  let a : ℕ → ℤ := λ n => 2*n - 1
  let S : ℕ → ℤ := λ n => n^2
  (S m = (a m + a (m+1)) / 2) → m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3062_306274


namespace NUMINAMATH_CALUDE_custom_op_neg_two_neg_one_l3062_306223

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := a^2 - abs b

-- Theorem statement
theorem custom_op_neg_two_neg_one :
  customOp (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_neg_two_neg_one_l3062_306223


namespace NUMINAMATH_CALUDE_constant_k_value_l3062_306220

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4) → k = -17 := by
  sorry

end NUMINAMATH_CALUDE_constant_k_value_l3062_306220


namespace NUMINAMATH_CALUDE_sum_of_integers_l3062_306291

theorem sum_of_integers (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120 →
  (Prime a ∨ Prime b ∨ Prime c ∨ Prime d ∨ Prime e) →
  a + b + c + d + e = 34 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3062_306291


namespace NUMINAMATH_CALUDE_prime_divides_sum_of_powers_l3062_306257

theorem prime_divides_sum_of_powers (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → q > 5 → q ∣ (2^p + 3^p) → q > p :=
by sorry

end NUMINAMATH_CALUDE_prime_divides_sum_of_powers_l3062_306257


namespace NUMINAMATH_CALUDE_nathan_bananas_l3062_306221

/-- The number of bananas Nathan has, given the specified bunches -/
def total_bananas (bunches_of_eight : Nat) (bananas_per_bunch_eight : Nat)
                  (bunches_of_seven : Nat) (bananas_per_bunch_seven : Nat) : Nat :=
  bunches_of_eight * bananas_per_bunch_eight + bunches_of_seven * bananas_per_bunch_seven

/-- Proof that Nathan has 83 bananas given the specified bunches -/
theorem nathan_bananas :
  total_bananas 6 8 5 7 = 83 := by
  sorry

end NUMINAMATH_CALUDE_nathan_bananas_l3062_306221


namespace NUMINAMATH_CALUDE_sum_of_digits_next_l3062_306265

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: For a positive integer n where S(n) = 1274, S(n+1) = 1239 -/
theorem sum_of_digits_next (n : ℕ) (h : S n = 1274) : S (n + 1) = 1239 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_next_l3062_306265


namespace NUMINAMATH_CALUDE_longest_working_secretary_hours_l3062_306235

theorem longest_working_secretary_hours (x y z : ℕ) : 
  x + y + z = 120 →  -- Total hours worked
  y = 2 * x →        -- Second secretary worked twice as long as the first
  z = 5 * x →        -- Third secretary worked five times as long as the first
  z = 75             -- The longest working secretary (third) worked 75 hours
:= by sorry

end NUMINAMATH_CALUDE_longest_working_secretary_hours_l3062_306235


namespace NUMINAMATH_CALUDE_absolute_value_sum_l3062_306251

theorem absolute_value_sum : -2 + |(-3)| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l3062_306251


namespace NUMINAMATH_CALUDE_solution_of_equation_l3062_306255

theorem solution_of_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l3062_306255


namespace NUMINAMATH_CALUDE_friends_total_points_l3062_306218

def total_points (darius_points marius_points matt_points : ℕ) : ℕ :=
  darius_points + marius_points + matt_points

theorem friends_total_points :
  ∀ (darius_points marius_points matt_points : ℕ),
    darius_points = 10 →
    marius_points = darius_points + 3 →
    matt_points = darius_points + 5 →
    total_points darius_points marius_points matt_points = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_friends_total_points_l3062_306218


namespace NUMINAMATH_CALUDE_inequality_proof_l3062_306285

theorem inequality_proof (x₃ x₄ : ℝ) (h1 : 1 < x₃) (h2 : x₃ < x₄) :
  x₃ * Real.exp x₄ > x₄ * Real.exp x₃ := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3062_306285


namespace NUMINAMATH_CALUDE_problem_statement_l3062_306266

theorem problem_statement (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4)
  (h2 : c ^ 3 = d ^ 2)
  (h3 : c - a = 19) :
  d - b = 757 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3062_306266


namespace NUMINAMATH_CALUDE_classmate_heights_most_suitable_l3062_306263

-- Define the survey options
inductive SurveyOption
  | LightBulbLifespan
  | WaterQualityGanRiver
  | TVProgramViewership
  | ClassmateHeights

-- Define the characteristic of being suitable for a comprehensive survey
def SuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.ClassmateHeights => True
  | _ => False

-- Theorem statement
theorem classmate_heights_most_suitable :
  SuitableForComprehensiveSurvey SurveyOption.ClassmateHeights ∧
  (∀ option : SurveyOption, option ≠ SurveyOption.ClassmateHeights →
    ¬SuitableForComprehensiveSurvey option) :=
by sorry

end NUMINAMATH_CALUDE_classmate_heights_most_suitable_l3062_306263


namespace NUMINAMATH_CALUDE_nuts_left_l3062_306253

theorem nuts_left (total : ℕ) (eaten_fraction : ℚ) (h1 : total = 30) (h2 : eaten_fraction = 5/6) :
  total - (total * eaten_fraction).floor = 5 := by
  sorry

end NUMINAMATH_CALUDE_nuts_left_l3062_306253


namespace NUMINAMATH_CALUDE_max_ratio_inscribed_circumscribed_sphere_radii_l3062_306225

/-- Given a right square pyramid with circumscribed and inscribed spheres, 
    this theorem states the maximum ratio of their radii. -/
theorem max_ratio_inscribed_circumscribed_sphere_radii 
  (R r d : ℝ) 
  (h_positive : R > 0 ∧ r > 0)
  (h_relation : d^2 + (R + r)^2 = 2 * R^2) :
  ∃ (max_ratio : ℝ), max_ratio = Real.sqrt 2 - 1 ∧ 
    r / R ≤ max_ratio ∧ 
    ∃ (r' d' : ℝ), r' / R = max_ratio ∧ 
      d'^2 + (R + r')^2 = 2 * R^2 := by
sorry

end NUMINAMATH_CALUDE_max_ratio_inscribed_circumscribed_sphere_radii_l3062_306225


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3062_306259

theorem ratio_x_to_y (x y : ℚ) (h : (12*x - 5*y) / (15*x - 4*y) = 4/7) : x/y = 19/24 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3062_306259


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3062_306231

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 5}
def B : Set Nat := {2, 3, 5}

theorem complement_intersection_theorem :
  (U \ B) ∩ A = {1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3062_306231


namespace NUMINAMATH_CALUDE_some_number_problem_l3062_306260

theorem some_number_problem (n : ℝ) :
  (∃ x₁ x₂ : ℝ, |x₁ - n| = 50 ∧ |x₂ - n| = 50 ∧ x₁ + x₂ = 50) →
  n = 25 :=
by sorry

end NUMINAMATH_CALUDE_some_number_problem_l3062_306260


namespace NUMINAMATH_CALUDE_min_c_value_l3062_306226

theorem min_c_value (a b c : ℕ+) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! p : ℝ × ℝ, (2 * p.1 + p.2 = 2010 ∧
    p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - c.val|)) :
  1006 ≤ c.val := by sorry

end NUMINAMATH_CALUDE_min_c_value_l3062_306226


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_M_l3062_306210

theorem polar_coordinates_of_point_M : 
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 3
  let ρ : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arccos (x / ρ)
  (ρ = 2) ∧ (θ = 2 * Real.pi / 3) := by sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_M_l3062_306210


namespace NUMINAMATH_CALUDE_trees_in_yard_l3062_306298

theorem trees_in_yard (yard_length : ℕ) (tree_distance : ℕ) (h1 : yard_length = 350) (h2 : tree_distance = 14) : 
  (yard_length / tree_distance) + 1 = 26 :=
by sorry

end NUMINAMATH_CALUDE_trees_in_yard_l3062_306298


namespace NUMINAMATH_CALUDE_jerry_firecracker_fraction_l3062_306214

/-- Given:
  * Jerry bought 48 firecrackers initially
  * 12 firecrackers were confiscated
  * 1/6 of the remaining firecrackers were defective
  * Jerry set off 15 good firecrackers
Prove that Jerry set off 1/2 of the good firecrackers -/
theorem jerry_firecracker_fraction :
  let initial_firecrackers : ℕ := 48
  let confiscated_firecrackers : ℕ := 12
  let defective_fraction : ℚ := 1/6
  let set_off_firecrackers : ℕ := 15
  let remaining_firecrackers := initial_firecrackers - confiscated_firecrackers
  let good_firecrackers := remaining_firecrackers - (defective_fraction * remaining_firecrackers).num
  (set_off_firecrackers : ℚ) / good_firecrackers = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_firecracker_fraction_l3062_306214


namespace NUMINAMATH_CALUDE_remaining_distance_l3062_306201

theorem remaining_distance (total_distance driven_distance : ℕ) : 
  total_distance = 1200 → driven_distance = 642 → total_distance - driven_distance = 558 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l3062_306201


namespace NUMINAMATH_CALUDE_line_symmetry_l3062_306236

/-- Given two lines in the xy-plane, this function returns true if they are symmetric with respect to the line y = x -/
def are_symmetric_lines (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y ↔ line2 y x

/-- The equation of the original line: 3x + 4y = 2 -/
def original_line (x y : ℝ) : Prop := 3 * x + 4 * y = 2

/-- The equation of the symmetric line: 4x + 3y = 2 -/
def symmetric_line (x y : ℝ) : Prop := 4 * x + 3 * y = 2

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to y = x -/
theorem line_symmetry : are_symmetric_lines original_line symmetric_line := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l3062_306236


namespace NUMINAMATH_CALUDE_expected_games_value_l3062_306219

/-- The expected number of games in a best-of-seven basketball match -/
def expected_games : ℚ :=
  let p : ℚ := 1 / 2  -- Probability of winning each game
  let prob4 : ℚ := 2 * p^4  -- Probability of ending in 4 games
  let prob5 : ℚ := 2 * 4 * p^4 * (1 - p)  -- Probability of ending in 5 games
  let prob6 : ℚ := 2 * 5 * p^3 * (1 - p)^2  -- Probability of ending in 6 games
  let prob7 : ℚ := 20 * p^3 * (1 - p)^3  -- Probability of ending in 7 games
  4 * prob4 + 5 * prob5 + 6 * prob6 + 7 * prob7

/-- Theorem: The expected number of games in a best-of-seven basketball match
    with equal win probabilities is 93/16 -/
theorem expected_games_value : expected_games = 93 / 16 := by
  sorry

end NUMINAMATH_CALUDE_expected_games_value_l3062_306219


namespace NUMINAMATH_CALUDE_spending_problem_l3062_306207

theorem spending_problem (initial_amount : ℚ) : 
  (initial_amount * (5/7) * (10/13) * (4/5) * (8/11) = 5400) → 
  initial_amount = 16890 := by
  sorry

end NUMINAMATH_CALUDE_spending_problem_l3062_306207


namespace NUMINAMATH_CALUDE_johnny_guitar_picks_l3062_306213

theorem johnny_guitar_picks (total_picks : ℕ) (red_picks blue_picks yellow_picks : ℕ) : 
  total_picks = red_picks + blue_picks + yellow_picks →
  2 * red_picks = total_picks →
  3 * blue_picks = total_picks →
  blue_picks = 12 →
  yellow_picks = 6 := by
sorry

end NUMINAMATH_CALUDE_johnny_guitar_picks_l3062_306213


namespace NUMINAMATH_CALUDE_equation_solution_l3062_306289

theorem equation_solution :
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3062_306289


namespace NUMINAMATH_CALUDE_triangle_side_product_l3062_306282

theorem triangle_side_product (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = pi) →
  ((a + b)^2 - c^2 = 4) →
  (C = pi / 3) →
  (a * b = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_product_l3062_306282


namespace NUMINAMATH_CALUDE_john_illustration_time_l3062_306247

/-- Calculates the total time spent on John's illustration project -/
def total_illustration_time (
  num_landscapes : ℕ)
  (num_portraits : ℕ)
  (landscape_draw_time : ℝ)
  (landscape_color_time_ratio : ℝ)
  (portrait_draw_time : ℝ)
  (portrait_color_time_ratio : ℝ)
  (landscape_enhance_time : ℝ)
  (portrait_enhance_time : ℝ) : ℝ :=
  let landscape_time := 
    num_landscapes * (landscape_draw_time + landscape_color_time_ratio * landscape_draw_time + landscape_enhance_time)
  let portrait_time := 
    num_portraits * (portrait_draw_time + portrait_color_time_ratio * portrait_draw_time + portrait_enhance_time)
  landscape_time + portrait_time

/-- Theorem stating the total time John spends on his illustration project -/
theorem john_illustration_time : 
  total_illustration_time 10 15 2 0.7 3 0.75 0.75 1 = 135.25 := by
  sorry

end NUMINAMATH_CALUDE_john_illustration_time_l3062_306247


namespace NUMINAMATH_CALUDE_sum_of_possible_AB_values_l3062_306243

/-- Represents a 7-digit number in the form A568B72 -/
def SevenDigitNumber (A B : Nat) : Nat :=
  A * 1000000 + 568000 + B * 100 + 72

theorem sum_of_possible_AB_values :
  (∀ A B : Nat, A < 10 ∧ B < 10 →
    SevenDigitNumber A B % 9 = 0 →
    (A + B = 8 ∨ A + B = 17)) ∧
  (∃ A₁ B₁ A₂ B₂ : Nat,
    A₁ < 10 ∧ B₁ < 10 ∧ A₂ < 10 ∧ B₂ < 10 ∧
    SevenDigitNumber A₁ B₁ % 9 = 0 ∧
    SevenDigitNumber A₂ B₂ % 9 = 0 ∧
    A₁ + B₁ = 8 ∧ A₂ + B₂ = 17) :=
by sorry

#check sum_of_possible_AB_values

end NUMINAMATH_CALUDE_sum_of_possible_AB_values_l3062_306243


namespace NUMINAMATH_CALUDE_relay_race_probability_l3062_306202

-- Define the set of students
inductive Student : Type
  | A | B | C | D

-- Define the events
def event_A (s : Student) : Prop := s = Student.A
def event_B (s : Student) : Prop := s = Student.B

-- Define the conditional probability
def conditional_probability (A B : Student → Prop) : ℚ :=
  1 / 3

-- Theorem statement
theorem relay_race_probability :
  conditional_probability event_A event_B = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_probability_l3062_306202


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3062_306248

-- Define the parabola function
def parabola (x : ℝ) : ℝ := (x - 6)^2 - 9

-- Define the square
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  sideLength : ℝ

-- Predicate to check if a square is inscribed in the region
def isInscribed (square : InscribedSquare) : Prop :=
  let halfSide := square.sideLength / 2
  let leftX := square.center - halfSide
  let rightX := square.center + halfSide
  leftX ≥ 0 ∧ 
  rightX ≥ 0 ∧ 
  parabola rightX = -square.sideLength

-- Theorem statement
theorem inscribed_square_area :
  ∃ (square : InscribedSquare), 
    isInscribed square ∧ 
    square.sideLength^2 = 40 - 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3062_306248


namespace NUMINAMATH_CALUDE_expression_values_l3062_306216

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (k : ℤ), k ∈ ({5, 2, 1, -2, -3} : Set ℤ) ∧
  (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d| = k) := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l3062_306216


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3062_306292

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  ArithmeticSequence a → a 5 = 3 → a 6 = -2 →
  (a 3) + (a 4) + (a 5) + (a 6) + (a 7) + (a 8) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3062_306292


namespace NUMINAMATH_CALUDE_rhombus_acute_angle_l3062_306208

/-- Given a rhombus, prove that its acute angle is arccos(1/9) when the ratio of volumes of rotation is 1:2√5 -/
theorem rhombus_acute_angle (a : ℝ) (h : a > 0) : 
  let α := Real.arccos (1/9)
  let V₁ := (1/3) * π * (a * Real.sin (α/2))^2 * (2 * a * Real.cos (α/2))
  let V₂ := π * (a * Real.sin α)^2 * a
  V₁ / V₂ = 1 / (2 * Real.sqrt 5) → 
  α = Real.arccos (1/9) := by
sorry

end NUMINAMATH_CALUDE_rhombus_acute_angle_l3062_306208


namespace NUMINAMATH_CALUDE_exam_average_l3062_306252

theorem exam_average (total_candidates : ℕ) (passed_candidates : ℕ) (passed_avg : ℚ) (failed_avg : ℚ) 
  (h1 : total_candidates = 120)
  (h2 : passed_candidates = 100)
  (h3 : passed_avg = 39)
  (h4 : failed_avg = 15) :
  let failed_candidates := total_candidates - passed_candidates
  let total_marks := passed_avg * passed_candidates + failed_avg * failed_candidates
  total_marks / total_candidates = 35 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l3062_306252


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3062_306206

theorem circumscribed_sphere_surface_area (cube_volume : ℝ) (h : cube_volume = 64) :
  let cube_side := cube_volume ^ (1/3)
  let sphere_radius := cube_side * Real.sqrt 3 / 2
  4 * Real.pi * sphere_radius ^ 2 = 48 * Real.pi :=
by
  sorry

#check circumscribed_sphere_surface_area

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3062_306206


namespace NUMINAMATH_CALUDE_girls_at_picnic_l3062_306215

theorem girls_at_picnic (total_students : ℕ) (picnic_attendees : ℕ) 
  (h1 : total_students = 1200)
  (h2 : picnic_attendees = 730)
  (h3 : ∃ (girls boys : ℕ), girls + boys = total_students ∧ 
    2 * girls / 3 + boys / 2 = picnic_attendees) :
  ∃ (girls : ℕ), 2 * girls / 3 = 520 := by
sorry

end NUMINAMATH_CALUDE_girls_at_picnic_l3062_306215


namespace NUMINAMATH_CALUDE_parabola_a_values_l3062_306290

/-- The parabola equation y = ax^2 -/
def parabola (a : ℝ) (x y : ℝ) : Prop := y = a * x^2

/-- The point M with coordinates (2, 1) -/
def point_M : ℝ × ℝ := (2, 1)

/-- The distance from point M to the directrix is 2 -/
def distance_to_directrix : ℝ := 2

/-- The possible values of a -/
def possible_a_values : Set ℝ := {1/4, -1/12}

/-- Theorem stating the possible values of a for the given conditions -/
theorem parabola_a_values :
  ∀ a : ℝ,
  (∃ y : ℝ, parabola a (point_M.1) y) →
  (∃ d : ℝ, d = distance_to_directrix ∧ 
    ((a > 0 ∧ d = point_M.2 + 1/(4*a)) ∨
     (a < 0 ∧ d = -1/(4*a) - point_M.2))) →
  a ∈ possible_a_values :=
sorry

end NUMINAMATH_CALUDE_parabola_a_values_l3062_306290


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3062_306287

/-- Given a parabola and a hyperbola, prove the equation of a circle with specific properties -/
theorem circle_equation_proof (x y : ℝ) : 
  (∃ (p : ℝ × ℝ), y^2 = 20*x ∧ p = (5, 0)) → -- Parabola equation and its focus
  (x^2/9 - y^2/16 = 1) →                    -- Hyperbola equation
  (∃ (c : ℝ × ℝ) (r : ℝ),                   -- Circle properties
    c = (5, 0) ∧                            -- Circle center at parabola focus
    r = 4 ∧                                 -- Circle radius
    (∀ (x' y' : ℝ), (y' = 4*x'/3 ∨ y' = -4*x'/3) →  -- Asymptotes of hyperbola
      (x' - c.1)^2 + (y' - c.2)^2 = r^2)) →  -- Circle tangent to asymptotes
  (x - 5)^2 + y^2 = 16                       -- Equation of the circle
  := by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3062_306287


namespace NUMINAMATH_CALUDE_combined_annual_income_after_expenses_l3062_306271

def brady_income : List ℕ := [150, 200, 250, 300, 200, 150, 180, 220, 240, 270, 300, 350]
def dwayne_income : List ℕ := [100, 150, 200, 250, 150, 120, 140, 190, 180, 230, 260, 300]
def brady_expense : ℕ := 450
def dwayne_expense : ℕ := 300

theorem combined_annual_income_after_expenses :
  (brady_income.sum - brady_expense) + (dwayne_income.sum - dwayne_expense) = 3930 := by
  sorry

end NUMINAMATH_CALUDE_combined_annual_income_after_expenses_l3062_306271


namespace NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_l3062_306295

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  p : ℝ
  vertex : ℝ × ℝ
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Given hyperbola and conditions on a parabola, proves the equation of the parabola -/
theorem parabola_equation_from_hyperbola (h : Hyperbola) (p : Parabola) :
  h.equation = (fun x y => 16 * x^2 - 9 * y^2 = 144) →
  p.vertex = (0, 0) →
  (p.focus = (3, 0) ∨ p.focus = (-3, 0)) →
  (p.equation = (fun x y => y^2 = 24 * x) ∨ p.equation = (fun x y => y^2 = -24 * x)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_l3062_306295


namespace NUMINAMATH_CALUDE_power_mod_seven_l3062_306262

theorem power_mod_seven : 3^2023 % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_power_mod_seven_l3062_306262


namespace NUMINAMATH_CALUDE_twin_brothers_age_product_difference_l3062_306281

theorem twin_brothers_age_product_difference :
  ∀ (current_age : ℕ),
  current_age = 4 →
  (current_age + 1) * (current_age + 1) - current_age * current_age = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_twin_brothers_age_product_difference_l3062_306281


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l3062_306275

/-- Given a line with slope -4 passing through the point (5, 2),
    prove that the sum of the slope and y-intercept is 18. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -4 ∧ 2 = m * 5 + b → m + b = 18 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l3062_306275


namespace NUMINAMATH_CALUDE_m_increasing_range_l3062_306205

def f (x : ℝ) : ℝ := (x - 1)^2

def is_m_increasing (f : ℝ → ℝ) (m : ℝ) (D : Set ℝ) : Prop :=
  m ≠ 0 ∧ ∀ x ∈ D, x + m ∈ D ∧ f (x + m) ≥ f x

theorem m_increasing_range (m : ℝ) :
  is_m_increasing f m (Set.Ici 0) → m ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_m_increasing_range_l3062_306205


namespace NUMINAMATH_CALUDE_total_apples_in_basket_l3062_306278

theorem total_apples_in_basket (red_apples green_apples : ℕ) 
  (h1 : red_apples = 7) 
  (h2 : green_apples = 2) : 
  red_apples + green_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_in_basket_l3062_306278


namespace NUMINAMATH_CALUDE_fifth_number_in_tenth_row_is_68_l3062_306270

/-- Represents a lattice with rows of consecutive integers -/
structure IntegerLattice where
  rowLength : ℕ
  rowCount : ℕ

/-- Gets the last number in a given row of the lattice -/
def lastNumberInRow (lattice : IntegerLattice) (row : ℕ) : ℤ :=
  (lattice.rowLength : ℤ) * row

/-- Gets the nth number in a given row of the lattice -/
def nthNumberInRow (lattice : IntegerLattice) (row : ℕ) (n : ℕ) : ℤ :=
  lastNumberInRow lattice row - (lattice.rowLength - n : ℤ)

/-- The theorem to be proved -/
theorem fifth_number_in_tenth_row_is_68 :
  let lattice : IntegerLattice := { rowLength := 7, rowCount := 10 }
  nthNumberInRow lattice 10 5 = 68 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_in_tenth_row_is_68_l3062_306270


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3062_306299

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -5
  let c : ℝ := 3
  let x₁ : ℝ := 3/2
  let x₂ : ℝ := 1
  (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3062_306299


namespace NUMINAMATH_CALUDE_smallest_valid_graph_size_l3062_306246

/-- A graph representing acquaintances among n people -/
def AcquaintanceGraph (n : ℕ) := Fin n → Fin n → Prop

/-- The property that any two acquainted people have no common acquaintances -/
def NoCommonAcquaintances (n : ℕ) (g : AcquaintanceGraph n) : Prop :=
  ∀ a b c : Fin n, g a b → g a c → g b c → a = b ∨ a = c ∨ b = c

/-- The property that any two non-acquainted people have exactly two common acquaintances -/
def TwoCommonAcquaintances (n : ℕ) (g : AcquaintanceGraph n) : Prop :=
  ∀ a b : Fin n, ¬g a b → ∃! (c d : Fin n), c ≠ d ∧ g a c ∧ g a d ∧ g b c ∧ g b d

/-- The main theorem stating that 11 is the smallest number satisfying the conditions -/
theorem smallest_valid_graph_size :
  (∃ (g : AcquaintanceGraph 11), NoCommonAcquaintances 11 g ∧ TwoCommonAcquaintances 11 g) ∧
  (∀ n : ℕ, 5 ≤ n → n < 11 →
    ¬∃ (g : AcquaintanceGraph n), NoCommonAcquaintances n g ∧ TwoCommonAcquaintances n g) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_graph_size_l3062_306246


namespace NUMINAMATH_CALUDE_annie_total_travel_l3062_306237

def blocks_to_bus_stop : ℕ := 5
def blocks_on_bus : ℕ := 7

def one_way_trip : ℕ := blocks_to_bus_stop + blocks_on_bus

theorem annie_total_travel : one_way_trip * 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_annie_total_travel_l3062_306237


namespace NUMINAMATH_CALUDE_specific_arrangement_double_coverage_l3062_306264

/-- Represents a rectangle on a grid -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the arrangement of rectangles on the grid -/
structure Arrangement where
  rectangles : List Rectangle
  -- Additional properties to describe the specific arrangement could be added here

/-- Counts the number of cells covered by exactly two rectangles in the given arrangement -/
def countDoublyCoveredCells (arr : Arrangement) : ℕ :=
  sorry -- Implementation details would go here

/-- The main theorem stating that for the specific arrangement of three 4x6 rectangles,
    the number of cells covered by exactly two rectangles is 14 -/
theorem specific_arrangement_double_coverage :
  ∃ (arr : Arrangement),
    (arr.rectangles.length = 3) ∧
    (∀ r ∈ arr.rectangles, r.width = 4 ∧ r.height = 6) ∧
    (countDoublyCoveredCells arr = 14) := by
  sorry


end NUMINAMATH_CALUDE_specific_arrangement_double_coverage_l3062_306264


namespace NUMINAMATH_CALUDE_linlins_speed_l3062_306233

/-- Proves that Linlin's speed is 400 meters per minute given the problem conditions --/
theorem linlins_speed (total_distance : ℕ) (time_taken : ℕ) (qingqing_speed : ℕ) :
  total_distance = 3290 →
  time_taken = 7 →
  qingqing_speed = 70 →
  (total_distance / time_taken - qingqing_speed : ℕ) = 400 :=
by sorry

end NUMINAMATH_CALUDE_linlins_speed_l3062_306233


namespace NUMINAMATH_CALUDE_exists_equal_boundary_interior_rectangle_l3062_306256

/-- Represents a rectangle in a triangular lattice grid -/
structure TriLatticeRectangle where
  m : Nat  -- horizontal side length in lattice units
  n : Nat  -- vertical side length in lattice units

/-- Calculates the number of lattice points on the boundary of the rectangle -/
def boundaryPoints (rect : TriLatticeRectangle) : Nat :=
  2 * (rect.m + rect.n)

/-- Calculates the number of lattice points inside the rectangle -/
def interiorPoints (rect : TriLatticeRectangle) : Nat :=
  2 * rect.m * rect.n - rect.m - rect.n + 1

/-- Theorem stating the existence of a rectangle with equal boundary and interior points -/
theorem exists_equal_boundary_interior_rectangle :
  ∃ (rect : TriLatticeRectangle), boundaryPoints rect = interiorPoints rect :=
sorry

end NUMINAMATH_CALUDE_exists_equal_boundary_interior_rectangle_l3062_306256


namespace NUMINAMATH_CALUDE_prime_equation_solution_l3062_306269

theorem prime_equation_solution :
  ∀ p q : ℕ,
  Prime p → Prime q →
  p^2 - 6*p*q + q^2 + 3*q - 1 = 0 →
  (p = 17 ∧ q = 3) :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l3062_306269


namespace NUMINAMATH_CALUDE_cube_root_three_equation_l3062_306284

theorem cube_root_three_equation (t : ℝ) : 
  t = 1 / (1 - Real.rpow 3 (1/3)) → 
  t = -(1 + Real.rpow 3 (1/3) + Real.rpow 3 (2/3)) / 2 := by
sorry

end NUMINAMATH_CALUDE_cube_root_three_equation_l3062_306284


namespace NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l3062_306230

/-- Calculates the percentage of alcohol in a mixture after adding water -/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 11)
  (h2 : initial_alcohol_percentage = 42)
  (h3 : water_added = 3) :
  let initial_alcohol_volume := initial_volume * (initial_alcohol_percentage / 100)
  let final_volume := initial_volume + water_added
  let final_alcohol_percentage := (initial_alcohol_volume / final_volume) * 100
  final_alcohol_percentage = 33 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l3062_306230


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_B_complement_union_A_B_l3062_306224

-- Define the universal set U
def U : Set ℝ := {x | x^2 - 3*x + 2 ≥ 0}

-- Define set A
def A : Set ℝ := {x | |x - 2| > 1}

-- Define set B
def B : Set ℝ := {x | (x - 1) / (x - 2) > 0}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x | x < 1 ∨ x > 3} := by sorry

theorem intersection_A_complement_B : A ∩ (U \ B) = ∅ := by sorry

theorem complement_union_A_B : U \ (A ∪ B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_B_complement_union_A_B_l3062_306224


namespace NUMINAMATH_CALUDE_courtyard_paving_l3062_306222

/-- The number of bricks required to pave a rectangular courtyard -/
def bricks_required (courtyard_length courtyard_width brick_length brick_width : ℚ) : ℚ :=
  (courtyard_length * courtyard_width) / (brick_length * brick_width)

/-- Theorem stating the number of bricks required for the specific courtyard and brick sizes -/
theorem courtyard_paving :
  bricks_required 25 15 0.2 0.1 = 18750 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l3062_306222


namespace NUMINAMATH_CALUDE_refrigerator_cash_price_l3062_306242

/-- The cash price of a refrigerator given installment payment details --/
theorem refrigerator_cash_price 
  (deposit : ℕ) 
  (num_installments : ℕ) 
  (installment_amount : ℕ) 
  (cash_savings : ℕ) : 
  deposit = 3000 →
  num_installments = 30 →
  installment_amount = 300 →
  cash_savings = 4000 →
  deposit + num_installments * installment_amount - cash_savings = 8000 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_cash_price_l3062_306242


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_decimal_l3062_306293

/-- The repeating decimal 0.4444... as a rational number -/
def repeating_decimal : ℚ := 4 / 9

/-- The result of 8 divided by the repeating decimal 0.4444... -/
theorem eight_divided_by_repeating_decimal : 8 / repeating_decimal = 18 := by sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_decimal_l3062_306293


namespace NUMINAMATH_CALUDE_polygon_sides_from_triangles_l3062_306209

/-- Represents a polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add any necessary fields here

/-- Represents a point on a side of a polygon -/
structure PointOnSide (p : Polygon n) where
  -- Add any necessary fields here

/-- The number of triangles formed when connecting a point on a side to all vertices -/
def numTriangles (p : Polygon n) (point : PointOnSide p) : ℕ :=
  n - 1

theorem polygon_sides_from_triangles
  (p : Polygon n) (point : PointOnSide p)
  (h : numTriangles p point = 8) :
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_triangles_l3062_306209


namespace NUMINAMATH_CALUDE_produce_worth_is_630_l3062_306200

/-- The total worth of produce Gary stocked -/
def total_worth (asparagus_bundles asparagus_price grape_boxes grape_price apple_count apple_price : ℝ) : ℝ :=
  asparagus_bundles * asparagus_price + grape_boxes * grape_price + apple_count * apple_price

/-- Proof that the total worth of produce Gary stocked is $630 -/
theorem produce_worth_is_630 :
  total_worth 60 3 40 2.5 700 0.5 = 630 := by
  sorry

#eval total_worth 60 3 40 2.5 700 0.5

end NUMINAMATH_CALUDE_produce_worth_is_630_l3062_306200


namespace NUMINAMATH_CALUDE_smallest_upper_bound_sum_reciprocals_l3062_306241

theorem smallest_upper_bound_sum_reciprocals :
  ∃ (r s : ℕ), r ≠ 0 ∧ s ≠ 0 ∧
  (∀ (k m n : ℕ), k ≠ 0 → m ≠ 0 → n ≠ 0 →
    (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n < 1 →
    (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n ≤ r / s) ∧
  (∀ (p q : ℕ), p ≠ 0 → q ≠ 0 →
    (∀ (k m n : ℕ), k ≠ 0 → m ≠ 0 → n ≠ 0 →
      (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n < 1 →
      (1 : ℚ) / k + (1 : ℚ) / m + (1 : ℚ) / n ≤ p / q) →
    r / s ≤ p / q) ∧
  r / s = 41 / 42 := by
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_sum_reciprocals_l3062_306241


namespace NUMINAMATH_CALUDE_factor_x4_minus_81_l3062_306217

theorem factor_x4_minus_81 : 
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_81_l3062_306217


namespace NUMINAMATH_CALUDE_x_equals_four_l3062_306203

/-- Custom operation € -/
def euro (x y : ℝ) : ℝ := 2 * x * y

/-- Theorem stating that x = 4 given the conditions -/
theorem x_equals_four :
  ∃ x : ℝ, euro 9 (euro x 5) = 720 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_x_equals_four_l3062_306203


namespace NUMINAMATH_CALUDE_james_tennis_balls_l3062_306212

theorem james_tennis_balls (total_containers : Nat) (balls_per_container : Nat) : 
  total_containers = 5 → 
  balls_per_container = 10 → 
  2 * (total_containers * balls_per_container) = 100 := by
sorry

end NUMINAMATH_CALUDE_james_tennis_balls_l3062_306212


namespace NUMINAMATH_CALUDE_fifteenth_term_of_inverse_proportional_sequence_l3062_306250

/-- A sequence where each term after the first is inversely proportional to the preceding term -/
def InverseProportionalSequence (a : ℕ → ℚ) : Prop :=
  ∃ k : ℚ, k ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = k

theorem fifteenth_term_of_inverse_proportional_sequence
  (a : ℕ → ℚ)
  (h_seq : InverseProportionalSequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 4) :
  a 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_inverse_proportional_sequence_l3062_306250


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3062_306297

/-- The line y = kx + 2k + 1 always passes through the point (-2, 1) for all real k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), ((-2 : ℝ) * k + 2 * k + 1 = 1) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3062_306297


namespace NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l3062_306238

theorem consecutive_product_not_perfect_power (a : ℤ) :
  ¬ ∃ (n : ℕ) (k : ℤ), n > 1 ∧ a * (a^2 - 1) = k^n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l3062_306238


namespace NUMINAMATH_CALUDE_product_zero_l3062_306232

theorem product_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l3062_306232


namespace NUMINAMATH_CALUDE_largest_m_satisfying_inequality_l3062_306239

theorem largest_m_satisfying_inequality :
  ∃ m : ℕ, (((1 : ℚ) / 4 + m / 9 < 5 / 2) ∧
            ∀ n : ℕ, (n > m → (1 : ℚ) / 4 + n / 9 ≥ 5 / 2)) ∧
            m = 10 := by
  sorry

end NUMINAMATH_CALUDE_largest_m_satisfying_inequality_l3062_306239


namespace NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l3062_306279

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDurations where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total cycle duration of a traffic light -/
def cycleDuration (d : TrafficLightDurations) : ℕ :=
  d.red + d.yellow + d.green

/-- Calculates the probability of seeing a red light -/
def redLightProbability (d : TrafficLightDurations) : ℚ :=
  d.red / (cycleDuration d)

/-- Theorem stating that the probability of seeing a red light is 2/5 -/
theorem red_light_probability_is_two_fifths (d : TrafficLightDurations)
  (h_red : d.red = 30)
  (h_yellow : d.yellow = 5)
  (h_green : d.green = 40) :
  redLightProbability d = 2/5 := by
  sorry

#eval redLightProbability ⟨30, 5, 40⟩

end NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l3062_306279


namespace NUMINAMATH_CALUDE_secret_spread_theorem_l3062_306286

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day of the week given a number of days after Monday -/
def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Monday"
  | 1 => "Tuesday"
  | 2 => "Wednesday"
  | 3 => "Thursday"
  | 4 => "Friday"
  | 5 => "Saturday"
  | _ => "Sunday"

theorem secret_spread_theorem :
  ∃ n : ℕ, secret_spread n = 3280 ∧ day_of_week n = "Monday" :=
by
  sorry

end NUMINAMATH_CALUDE_secret_spread_theorem_l3062_306286


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l3062_306234

theorem farmer_tomatoes (initial_tomatoes : ℕ) (picked_tomatoes : ℕ) 
  (h1 : initial_tomatoes = 17)
  (h2 : initial_tomatoes - picked_tomatoes = 8) :
  picked_tomatoes = 9 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l3062_306234


namespace NUMINAMATH_CALUDE_hardey_fitness_center_ratio_l3062_306229

theorem hardey_fitness_center_ratio 
  (avg_female : ℝ) 
  (avg_male : ℝ) 
  (avg_child : ℝ) 
  (avg_overall : ℝ) 
  (h1 : avg_female = 35)
  (h2 : avg_male = 30)
  (h3 : avg_child = 10)
  (h4 : avg_overall = 25) :
  ∃ (f m c : ℝ), 
    f > 0 ∧ m > 0 ∧ c > 0 ∧
    (avg_female * f + avg_male * m + avg_child * c) / (f + m + c) = avg_overall ∧
    c / (f + m) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hardey_fitness_center_ratio_l3062_306229


namespace NUMINAMATH_CALUDE_local_min_implies_a_eq_one_l3062_306276

/-- The function f(x) = x³ - 2ax² + a²x + 1 has a local minimum at x = 1 -/
def has_local_min_at_one (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f x ≥ f 1

/-- The function f(x) = x³ - 2ax² + a²x + 1 -/
def f (a x : ℝ) : ℝ := x^3 - 2*a*x^2 + a^2*x + 1

theorem local_min_implies_a_eq_one (a : ℝ) :
  has_local_min_at_one (f a) a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_local_min_implies_a_eq_one_l3062_306276


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3062_306249

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 10 = 3) →
  (a 3 * a 10 = -5) →
  a 5 + a 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3062_306249


namespace NUMINAMATH_CALUDE_total_teachers_count_l3062_306254

/-- Given a school with major and minor departments, calculate the total number of teachers -/
theorem total_teachers_count (total_departments : Nat) (major_departments : Nat) (minor_departments : Nat)
  (teachers_per_major : Nat) (teachers_per_minor : Nat)
  (h1 : total_departments = major_departments + minor_departments)
  (h2 : total_departments = 17)
  (h3 : major_departments = 9)
  (h4 : minor_departments = 8)
  (h5 : teachers_per_major = 45)
  (h6 : teachers_per_minor = 29) :
  major_departments * teachers_per_major + minor_departments * teachers_per_minor = 637 := by
  sorry

#check total_teachers_count

end NUMINAMATH_CALUDE_total_teachers_count_l3062_306254


namespace NUMINAMATH_CALUDE_sixth_doll_size_l3062_306283

def doll_size (n : ℕ) : ℚ :=
  243 * (2/3)^(n-1)

theorem sixth_doll_size : doll_size 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sixth_doll_size_l3062_306283


namespace NUMINAMATH_CALUDE_john_annual_oil_change_cost_l3062_306258

/-- Calculates the annual cost of oil changes for John --/
def annual_oil_change_cost (miles_per_month : ℕ) (miles_per_oil_change : ℕ) (free_oil_changes : ℕ) (cost_per_oil_change : ℕ) : ℕ :=
  let annual_miles := miles_per_month * 12
  let total_oil_changes := annual_miles / miles_per_oil_change
  let paid_oil_changes := total_oil_changes - free_oil_changes
  paid_oil_changes * cost_per_oil_change

theorem john_annual_oil_change_cost :
  annual_oil_change_cost 1000 3000 1 50 = 150 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_oil_change_cost_l3062_306258


namespace NUMINAMATH_CALUDE_M_subset_N_l3062_306227

def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l3062_306227


namespace NUMINAMATH_CALUDE_equal_money_in_five_weeks_l3062_306296

/-- Represents the number of weeks it takes for two people to have the same amount of money -/
def weeks_to_equal_money (carol_initial : ℕ) (carol_weekly : ℕ) (mike_initial : ℕ) (mike_weekly : ℕ) : ℕ :=
  sorry

/-- Theorem stating that it takes 5 weeks for Carol and Mike to have the same amount of money -/
theorem equal_money_in_five_weeks :
  weeks_to_equal_money 60 9 90 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_money_in_five_weeks_l3062_306296


namespace NUMINAMATH_CALUDE_f_properties_l3062_306228

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℤ := floor ((x + 1) / 3 - floor (x / 3))

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f (x + 3) = f x) ∧ 
  (∀ y : ℤ, y ∈ Set.range f → y = 0 ∨ y = 1) ∧
  (∀ y : ℤ, y = 0 ∨ y = 1 → ∃ x : ℝ, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3062_306228


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3062_306245

theorem complex_equation_solution :
  ∀ (z : ℂ), (1 + Complex.I) * z = 2 + Complex.I → z = 3/2 - (1/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3062_306245


namespace NUMINAMATH_CALUDE_expression_value_l3062_306211

theorem expression_value : (40 + 15)^2 - 15^2 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3062_306211


namespace NUMINAMATH_CALUDE_trainers_average_age_l3062_306288

/-- The average age of trainers in a sports club --/
theorem trainers_average_age
  (total_members : ℕ)
  (overall_average : ℚ)
  (num_women : ℕ)
  (num_men : ℕ)
  (num_trainers : ℕ)
  (women_average : ℚ)
  (men_average : ℚ)
  (h_total : total_members = 70)
  (h_overall : overall_average = 23)
  (h_women : num_women = 30)
  (h_men : num_men = 25)
  (h_trainers : num_trainers = 15)
  (h_women_avg : women_average = 20)
  (h_men_avg : men_average = 25)
  (h_sum : total_members = num_women + num_men + num_trainers) :
  (total_members * overall_average - num_women * women_average - num_men * men_average) / num_trainers = 25 + 2/3 :=
by sorry

end NUMINAMATH_CALUDE_trainers_average_age_l3062_306288


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l3062_306204

/-- The distance between Maxwell's and Brad's homes in kilometers -/
def total_distance : ℝ := 40

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 3

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 5

/-- The distance traveled by Maxwell when they meet -/
def maxwell_distance : ℝ := 15

theorem meeting_point_theorem :
  maxwell_distance = total_distance * maxwell_speed / (maxwell_speed + brad_speed) :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l3062_306204


namespace NUMINAMATH_CALUDE_probability_n_less_than_m_plus_one_l3062_306280

/-- The number of balls in the bag -/
def num_balls : ℕ := 4

/-- The set of possible ball numbers -/
def ball_numbers : Finset ℕ := Finset.range num_balls

/-- The sample space of all possible outcomes (m, n) -/
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product ball_numbers ball_numbers

/-- The event where n < m + 1 -/
def event : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => p.2 < p.1 + 1)

/-- The probability of the event -/
noncomputable def probability : ℚ :=
  (event.card : ℚ) / (sample_space.card : ℚ)

theorem probability_n_less_than_m_plus_one :
  probability = 5/8 := by sorry

end NUMINAMATH_CALUDE_probability_n_less_than_m_plus_one_l3062_306280


namespace NUMINAMATH_CALUDE_min_red_balls_l3062_306240

/-- The total number of balls in the circle -/
def total_balls : ℕ := 58

/-- A type representing the color of a ball -/
inductive Color
| Red
| Blue

/-- A function that counts the number of consecutive triplets with a majority of a given color -/
def count_majority_triplets (balls : List Color) (color : Color) : ℕ := sorry

/-- A function that counts the total number of balls of a given color -/
def count_color (balls : List Color) (color : Color) : ℕ := sorry

/-- The main theorem stating the minimum number of red balls -/
theorem min_red_balls (balls : List Color) :
  balls.length = total_balls →
  count_majority_triplets balls Color.Red = count_majority_triplets balls Color.Blue →
  count_color balls Color.Red ≥ 20 := by sorry

end NUMINAMATH_CALUDE_min_red_balls_l3062_306240


namespace NUMINAMATH_CALUDE_frame_ratio_l3062_306268

theorem frame_ratio (x : ℝ) (h : x > 0) : 
  (20 + 2*x) * (30 + 6*x) - 20 * 30 = 20 * 30 →
  (20 + 2*x) / (30 + 6*x) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_frame_ratio_l3062_306268
