import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_number_decimal_sum_l3444_344418

theorem two_digit_number_decimal_sum (a b : ℕ) (h1 : a ≥ 1 ∧ a ≤ 9) (h2 : b ≥ 0 ∧ b ≤ 9) :
  let n := 10 * a + b
  (n : ℚ) + (a : ℚ) + (b : ℚ) / 10 = 869 / 10 → n = 79 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_decimal_sum_l3444_344418


namespace NUMINAMATH_CALUDE_fast_food_order_cost_l3444_344455

/-- Calculates the total cost of a fast-food order with discount and tax --/
theorem fast_food_order_cost
  (burger_cost : ℝ)
  (sandwich_cost : ℝ)
  (smoothie_cost : ℝ)
  (num_smoothies : ℕ)
  (discount_rate : ℝ)
  (discount_threshold : ℝ)
  (tax_rate : ℝ)
  (h1 : burger_cost = 5)
  (h2 : sandwich_cost = 4)
  (h3 : smoothie_cost = 4)
  (h4 : num_smoothies = 2)
  (h5 : discount_rate = 0.15)
  (h6 : discount_threshold = 10)
  (h7 : tax_rate = 0.1) :
  let total_before_discount := burger_cost + sandwich_cost + (smoothie_cost * num_smoothies)
  let discount := if total_before_discount > discount_threshold then total_before_discount * discount_rate else 0
  let total_after_discount := total_before_discount - discount
  let tax := total_after_discount * tax_rate
  let total_cost := total_after_discount + tax
  ∃ (n : ℕ), (n : ℝ) / 100 = total_cost ∧ n = 1590 :=
by sorry


end NUMINAMATH_CALUDE_fast_food_order_cost_l3444_344455


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l3444_344492

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that |g(x)| = 15 for x ∈ {0, 1, 2, 4, 5, 6} -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  ∀ x ∈ ({0, 1, 2, 4, 5, 6} : Set ℝ), |g x| = 15

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g (-1)| = 75 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l3444_344492


namespace NUMINAMATH_CALUDE_sqrt_sum_square_product_l3444_344416

theorem sqrt_sum_square_product (x : ℝ) :
  Real.sqrt (9 + x) + Real.sqrt (25 - x) = 10 →
  (9 + x) * (25 - x) = 1089 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_square_product_l3444_344416


namespace NUMINAMATH_CALUDE_additional_birds_l3444_344491

theorem additional_birds (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 231)
  (h2 : final_birds = 312) :
  final_birds - initial_birds = 81 := by
  sorry

end NUMINAMATH_CALUDE_additional_birds_l3444_344491


namespace NUMINAMATH_CALUDE_geometric_sequence_iff_a_eq_plus_minus_six_l3444_344484

/-- A sequence of three real numbers is geometric if the ratio between consecutive terms is constant. -/
def IsGeometricSequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

/-- The main theorem stating that the sequence 4, a, 9 is geometric if and only if a = 6 or a = -6 -/
theorem geometric_sequence_iff_a_eq_plus_minus_six :
  ∀ a : ℝ, IsGeometricSequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_iff_a_eq_plus_minus_six_l3444_344484


namespace NUMINAMATH_CALUDE_circle_symmetry_l3444_344489

/-- Given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 1 = 0

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

/-- Symmetrical circle -/
def symmetrical_circle (x y : ℝ) : Prop :=
  (x + 7/5)^2 + (y - 6/5)^2 = 2

/-- Theorem stating that the symmetrical circle is indeed symmetrical to the given circle
    with respect to the line of symmetry -/
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    given_circle x₁ y₁ →
    symmetrical_circle x₂ y₂ →
    ∃ (x_mid y_mid : ℝ),
      symmetry_line x_mid y_mid ∧
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3444_344489


namespace NUMINAMATH_CALUDE_sum_of_combinations_equals_total_l3444_344451

/-- The number of people who took each unique combination of drinks at a gathering --/
def drink_combinations : List ℕ := [
  12, 10, 6, 4, 8, 5, 3, 7, 2, 4, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1
]

/-- The total number of people at the gathering --/
def total_people : ℕ := 89

/-- Theorem stating that the sum of people taking each unique combination of drinks equals the total number of people at the gathering --/
theorem sum_of_combinations_equals_total :
  drink_combinations.sum = total_people := by
  sorry


end NUMINAMATH_CALUDE_sum_of_combinations_equals_total_l3444_344451


namespace NUMINAMATH_CALUDE_function_inequality_l3444_344401

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 2) = -f x)
  (h2 : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 → f x₁ < f x₂)
  (h3 : ∀ x : ℝ, f (x + 2) = f (-x + 2)) :
  f 4.5 < f 7 ∧ f 7 < f 6.5 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l3444_344401


namespace NUMINAMATH_CALUDE_min_cost_floppy_cd_l3444_344436

/-- The minimum cost of 3 floppy disks and 9 CDs given price constraints -/
theorem min_cost_floppy_cd (x y : ℝ) 
  (h1 : 4 * x + 5 * y ≥ 20) 
  (h2 : 6 * x + 3 * y ≤ 24) : 
  ∃ (m : ℝ), m = 3 * x + 9 * y ∧ m ≥ 22 ∧ ∀ (n : ℝ), n = 3 * x + 9 * y → n ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_cost_floppy_cd_l3444_344436


namespace NUMINAMATH_CALUDE_device_improvement_l3444_344424

/-- Represents the sample mean and variance of a device's measurements -/
structure DeviceStats where
  mean : ℝ
  variance : ℝ

/-- Determines if there's a significant improvement between two devices -/
def significantImprovement (old new : DeviceStats) : Prop :=
  new.mean - old.mean ≥ 2 * Real.sqrt ((old.variance + new.variance) / 10)

theorem device_improvement (old new : DeviceStats) 
  (h_old : old = ⟨10, 0.036⟩) 
  (h_new : new = ⟨10.3, 0.04⟩) : 
  significantImprovement old new := by
  sorry

#check device_improvement

end NUMINAMATH_CALUDE_device_improvement_l3444_344424


namespace NUMINAMATH_CALUDE_days_to_empty_tube_l3444_344475

-- Define the volume of the gel tube in mL
def tube_volume : ℝ := 128

-- Define the daily usage of gel in mL
def daily_usage : ℝ := 4

-- Theorem statement
theorem days_to_empty_tube : 
  (tube_volume / daily_usage : ℝ) = 32 := by
  sorry

end NUMINAMATH_CALUDE_days_to_empty_tube_l3444_344475


namespace NUMINAMATH_CALUDE_expression_simplification_l3444_344457

def simplify_expression (a b : ℤ) : ℤ :=
  -2 * (10 * a^2 + 2 * a * b + 3 * b^2) + 3 * (5 * a^2 - 4 * a * b)

theorem expression_simplification :
  simplify_expression 1 (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3444_344457


namespace NUMINAMATH_CALUDE_peters_correct_percentage_l3444_344437

theorem peters_correct_percentage (y : ℕ) :
  let total_questions : ℕ := 7 * y
  let missed_questions : ℕ := 2 * y
  let correct_questions : ℕ := total_questions - missed_questions
  (correct_questions : ℚ) / (total_questions : ℚ) * 100 = 500 / 7 :=
by sorry

end NUMINAMATH_CALUDE_peters_correct_percentage_l3444_344437


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l3444_344442

theorem complex_imaginary_part (z : ℂ) (h : z * (3 - 4*I) = 1) : z.im = 4/25 := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l3444_344442


namespace NUMINAMATH_CALUDE_recliner_sales_increase_l3444_344460

theorem recliner_sales_increase 
  (price_reduction : ℝ) 
  (gross_increase : ℝ) 
  (sales_increase : ℝ) : 
  price_reduction = 0.2 → 
  gross_increase = 0.4400000000000003 → 
  sales_increase = (1 + gross_increase) / (1 - price_reduction) - 1 →
  sales_increase = 0.8 := by
sorry

end NUMINAMATH_CALUDE_recliner_sales_increase_l3444_344460


namespace NUMINAMATH_CALUDE_statue_weight_l3444_344448

-- Define the initial weight and cutting percentages
def initial_weight : ℝ := 250
def first_cut : ℝ := 0.30
def second_cut : ℝ := 0.20
def third_cut : ℝ := 0.25

-- Define the final weight calculation
def final_weight : ℝ :=
  initial_weight * (1 - first_cut) * (1 - second_cut) * (1 - third_cut)

-- Theorem statement
theorem statue_weight :
  final_weight = 105 := by sorry

end NUMINAMATH_CALUDE_statue_weight_l3444_344448


namespace NUMINAMATH_CALUDE_beaver_home_fraction_l3444_344483

theorem beaver_home_fraction (total_beavers : ℕ) (swim_percentage : ℚ) :
  total_beavers = 4 →
  swim_percentage = 3/4 →
  (1 : ℚ) - swim_percentage = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_beaver_home_fraction_l3444_344483


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3444_344488

theorem probability_of_white_ball (p_red_or_white p_yellow_or_white : ℝ) 
  (h1 : p_red_or_white = 0.65)
  (h2 : p_yellow_or_white = 0.6) :
  1 - (1 - p_yellow_or_white) - (1 - p_red_or_white) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3444_344488


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3444_344435

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_eq : a 3 * a 9 = 2 * (a 5)^2)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3444_344435


namespace NUMINAMATH_CALUDE_poverty_alleviation_volunteers_l3444_344463

/-- Represents the age group frequencies in the histogram -/
structure AgeDistribution :=
  (f1 f2 f3 f4 f5 : ℝ)
  (sum_to_one : f1 + f2 + f3 + f4 + f5 = 1)

/-- Represents the stratified sample -/
structure StratifiedSample :=
  (total : ℕ)
  (under_35 : ℕ)
  (over_35 : ℕ)
  (sum_equal : under_35 + over_35 = total)

/-- The main theorem -/
theorem poverty_alleviation_volunteers 
  (dist : AgeDistribution) 
  (sample : StratifiedSample) 
  (h1 : dist.f1 = 0.01)
  (h2 : dist.f2 = 0.02)
  (h3 : dist.f3 = 0.04)
  (h5 : dist.f5 = 0.07)
  (h_sample : sample.total = 10 ∧ sample.under_35 = 6 ∧ sample.over_35 = 4) :
  dist.f4 = 0.06 ∧ 
  ∃ (X : Fin 4 → ℝ), 
    X 0 = 1/30 ∧ 
    X 1 = 3/10 ∧ 
    X 2 = 1/2 ∧ 
    X 3 = 1/6 ∧
    (X 0 * 0 + X 1 * 1 + X 2 * 2 + X 3 * 3 = 1.8) := by
  sorry

end NUMINAMATH_CALUDE_poverty_alleviation_volunteers_l3444_344463


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3444_344400

theorem trigonometric_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3444_344400


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3444_344429

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I * (a + Complex.I) = -1 - 2 * Complex.I) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3444_344429


namespace NUMINAMATH_CALUDE_soda_survey_result_l3444_344453

-- Define the total number of people surveyed
def total_surveyed : ℕ := 500

-- Define the central angle of the "Soda" sector in degrees
def soda_angle : ℕ := 198

-- Define the function to calculate the number of people who chose "Soda"
def soda_count : ℕ := (total_surveyed * soda_angle) / 360

-- Theorem statement
theorem soda_survey_result : soda_count = 275 := by
  sorry

end NUMINAMATH_CALUDE_soda_survey_result_l3444_344453


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3444_344405

/-- An arithmetic sequence starting with 2 and ending with 2006 has 502 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    a 0 = 2 → 
    (∃ n : ℕ, a n = 2006) → 
    (∀ i j : ℕ, a (i + 1) - a i = a (j + 1) - a j) → 
    (∃ n : ℕ, a n = 2006 ∧ n + 1 = 502) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3444_344405


namespace NUMINAMATH_CALUDE_cube_sum_property_l3444_344468

/-- A cube is a three-dimensional geometric shape -/
structure Cube where

/-- The number of edges in a cube -/
def Cube.num_edges (c : Cube) : ℕ := 12

/-- The number of corners in a cube -/
def Cube.num_corners (c : Cube) : ℕ := 8

/-- The number of faces in a cube -/
def Cube.num_faces (c : Cube) : ℕ := 6

/-- The theorem stating that the sum of edges, corners, and faces of a cube is 26 -/
theorem cube_sum_property (c : Cube) : 
  c.num_edges + c.num_corners + c.num_faces = 26 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_property_l3444_344468


namespace NUMINAMATH_CALUDE_optimal_division_l3444_344498

theorem optimal_division (a : ℝ) (h : a > 0) :
  let f := fun (x : ℝ) => x / (a - x) + (a - x) / x
  ∃ (x : ℝ), 0 < x ∧ x < a ∧ ∀ (y : ℝ), 0 < y ∧ y < a → f x ≤ f y :=
by sorry

end NUMINAMATH_CALUDE_optimal_division_l3444_344498


namespace NUMINAMATH_CALUDE_min_value_of_sum_min_value_exists_l3444_344411

theorem min_value_of_sum (a b : ℝ) (ha : a > -1) (hb : b > -2) (hab : (a + 1) * (b + 2) = 16) :
  ∀ x y : ℝ, x > -1 → y > -2 → (x + 1) * (y + 2) = 16 → a + b ≤ x + y :=
sorry

theorem min_value_exists (a b : ℝ) (ha : a > -1) (hb : b > -2) (hab : (a + 1) * (b + 2) = 16) :
  ∃ x y : ℝ, x > -1 ∧ y > -2 ∧ (x + 1) * (y + 2) = 16 ∧ x + y = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_min_value_exists_l3444_344411


namespace NUMINAMATH_CALUDE_small_boxes_packed_l3444_344454

/-- Represents the number of feet of tape used for sealing each type of box --/
def seal_tape_large : ℕ := 4
def seal_tape_medium : ℕ := 2
def seal_tape_small : ℕ := 1

/-- Represents the number of feet of tape used for address label on each box --/
def label_tape : ℕ := 1

/-- Represents the number of large boxes packed --/
def num_large : ℕ := 2

/-- Represents the number of medium boxes packed --/
def num_medium : ℕ := 8

/-- Represents the total amount of tape used in feet --/
def total_tape : ℕ := 44

/-- Calculates the number of small boxes packed --/
def num_small : ℕ := 
  (total_tape - 
   (num_large * (seal_tape_large + label_tape) + 
    num_medium * (seal_tape_medium + label_tape))) / 
  (seal_tape_small + label_tape)

theorem small_boxes_packed : num_small = 5 := by
  sorry

end NUMINAMATH_CALUDE_small_boxes_packed_l3444_344454


namespace NUMINAMATH_CALUDE_quadratic_sum_l3444_344447

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), quadratic a b c y ≥ quadratic a b c x) ∧  -- minimum exists
  (quadratic a b c 3 = 0) ∧  -- passes through (3,0)
  (quadratic a b c 7 = 0) ∧  -- passes through (7,0)
  (∀ (x : ℝ), quadratic a b c x ≥ 36) ∧  -- minimum value is 36
  (∃ (x : ℝ), quadratic a b c x = 36)  -- minimum value is achieved
  →
  a + b + c = -108 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3444_344447


namespace NUMINAMATH_CALUDE_no_infinite_line_family_l3444_344445

theorem no_infinite_line_family : ¬ ∃ (k : ℕ → ℝ), 
  (∀ n, k (n + 1) ≥ k n - 1 / k n) ∧ 
  (∀ n, k n * k (n + 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_line_family_l3444_344445


namespace NUMINAMATH_CALUDE_first_round_games_count_l3444_344413

/-- A tennis tournament with specific conditions -/
structure TennisTournament where
  total_rounds : Nat
  second_round_games : Nat
  third_round_games : Nat
  final_games : Nat
  cans_per_game : Nat
  balls_per_can : Nat
  total_balls_used : Nat

/-- The number of games in the first round of the tournament -/
def first_round_games (t : TennisTournament) : Nat :=
  ((t.total_balls_used - (t.second_round_games + t.third_round_games + t.final_games) * 
    t.cans_per_game * t.balls_per_can) / (t.cans_per_game * t.balls_per_can))

/-- Theorem stating the number of games in the first round -/
theorem first_round_games_count (t : TennisTournament) 
  (h1 : t.total_rounds = 4)
  (h2 : t.second_round_games = 4)
  (h3 : t.third_round_games = 2)
  (h4 : t.final_games = 1)
  (h5 : t.cans_per_game = 5)
  (h6 : t.balls_per_can = 3)
  (h7 : t.total_balls_used = 225) :
  first_round_games t = 8 := by
  sorry

end NUMINAMATH_CALUDE_first_round_games_count_l3444_344413


namespace NUMINAMATH_CALUDE_circle_tangent_relation_l3444_344443

/-- Two circles with radii R₁ and R₂ are externally tangent. A line of length d is perpendicular to their common tangent. -/
structure CircleConfiguration where
  R₁ : ℝ
  R₂ : ℝ
  d : ℝ
  R₁_pos : 0 < R₁
  R₂_pos : 0 < R₂
  d_pos : 0 < d
  externally_tangent : R₁ + R₂ > 0

/-- The relation between the radii of two externally tangent circles and the length of a line perpendicular to their common tangent. -/
theorem circle_tangent_relation (c : CircleConfiguration) :
  1 / c.R₁ + 1 / c.R₂ = 2 / c.d := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_relation_l3444_344443


namespace NUMINAMATH_CALUDE_exists_valid_classification_l3444_344423

/-- Represents a team of students -/
structure Team :=
  (members : Finset Nat)
  (size_eq_six : members.card = 6)

/-- Classification of teams as GOOD or OK -/
def TeamClassification := Team → Bool

/-- Partition of students into teams -/
structure Partition :=
  (teams : Finset Team)
  (covers_all_students : (teams.biUnion Team.members).card = 24)
  (team_count_eq_four : teams.card = 4)

/-- Counts the number of GOOD teams in a partition -/
def countGoodTeams (c : TeamClassification) (p : Partition) : Nat :=
  (p.teams.filter (λ t => c t)).card

/-- Theorem stating the existence of a valid team classification -/
theorem exists_valid_classification : ∃ (c : TeamClassification),
  (∀ (p : Partition), countGoodTeams c p = 3 ∨ countGoodTeams c p = 1) ∧
  (∃ (p1 p2 : Partition), countGoodTeams c p1 = 3 ∧ countGoodTeams c p2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_classification_l3444_344423


namespace NUMINAMATH_CALUDE_f_properties_l3444_344419

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_properties :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ < -1 ∧ x₂ < -1 → f x₁ > f x₂) ∧
  (∀ x₁ x₂, -1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (f (-1) = -(1 / Real.exp 1)) ∧
  (∀ x, f x ≥ -(1 / Real.exp 1)) ∧
  (∀ y : ℝ, ∃ x, f x > y) ∧
  (∃ a : ℝ, a ≥ -2 ∧
    ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ →
      (f x₂ - f a) / (x₂ - a) > (f x₁ - f a) / (x₁ - a)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3444_344419


namespace NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l3444_344494

theorem tangent_line_to_x_ln_x (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (∀ x : ℝ, 2 * x + m = x₀ * Real.log x₀ + (Real.log x₀ + 1) * (x - x₀)) ∧
    (∀ x : ℝ, x > 0 → 2 * x + m ≥ x * Real.log x)) →
  m = -Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l3444_344494


namespace NUMINAMATH_CALUDE_box_dimensions_l3444_344414

theorem box_dimensions (x y z : ℝ) 
  (volume : x * y * z = 160)
  (face1 : y * z = 80)
  (face2 : x * z = 40)
  (face3 : x * y = 32) :
  x = 4 ∧ y = 8 ∧ z = 10 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_l3444_344414


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l3444_344469

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Angles are positive
  a + b = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  a / b = 3 / 2 →  -- Ratio of angles is 3:2
  a = 54 ∧ b = 36 := by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l3444_344469


namespace NUMINAMATH_CALUDE_sqrt_of_four_equals_two_l3444_344450

theorem sqrt_of_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_equals_two_l3444_344450


namespace NUMINAMATH_CALUDE_correct_calculation_l3444_344486

theorem correct_calculation (a : ℝ) : (2*a)^2 / (4*a) = a := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3444_344486


namespace NUMINAMATH_CALUDE_product_of_roots_squared_minus_three_l3444_344412

theorem product_of_roots_squared_minus_three (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - y₁^3 + 1 = 0) → 
  (y₂^5 - y₂^3 + 1 = 0) → 
  (y₃^5 - y₃^3 + 1 = 0) → 
  (y₄^5 - y₄^3 + 1 = 0) → 
  (y₅^5 - y₅^3 + 1 = 0) → 
  ((y₁^2 - 3) * (y₂^2 - 3) * (y₃^2 - 3) * (y₄^2 - 3) * (y₅^2 - 3) = -35) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_squared_minus_three_l3444_344412


namespace NUMINAMATH_CALUDE_pauls_lost_crayons_l3444_344446

/-- Given that Paul initially had 110 crayons, gave 90 crayons to his friends,
    and lost 322 more crayons than those he gave to his friends,
    prove that Paul lost 412 crayons. -/
theorem pauls_lost_crayons
  (initial_crayons : ℕ)
  (crayons_given : ℕ)
  (extra_lost_crayons : ℕ)
  (h1 : initial_crayons = 110)
  (h2 : crayons_given = 90)
  (h3 : extra_lost_crayons = 322)
  : crayons_given + extra_lost_crayons = 412 := by
  sorry

end NUMINAMATH_CALUDE_pauls_lost_crayons_l3444_344446


namespace NUMINAMATH_CALUDE_second_investment_rate_l3444_344406

/-- Calculates the interest rate of the second investment given the total investment,
    the amount and rate of the first investment, and the total amount after interest. -/
theorem second_investment_rate (total_investment : ℝ) (first_investment : ℝ) 
  (first_rate : ℝ) (total_after_interest : ℝ) :
  total_investment = 1000 →
  first_investment = 200 →
  first_rate = 0.03 →
  total_after_interest = 1046 →
  (total_after_interest - total_investment - (first_investment * first_rate)) / 
  (total_investment - first_investment) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_rate_l3444_344406


namespace NUMINAMATH_CALUDE_nell_baseball_cards_l3444_344496

/-- Nell's baseball card collection problem -/
theorem nell_baseball_cards :
  ∀ (initial_cards given_cards remaining_cards : ℕ),
  given_cards = 28 →
  remaining_cards = 276 →
  initial_cards = given_cards + remaining_cards →
  initial_cards = 304 :=
by
  sorry

end NUMINAMATH_CALUDE_nell_baseball_cards_l3444_344496


namespace NUMINAMATH_CALUDE_sufficient_condition_for_product_greater_than_four_l3444_344497

theorem sufficient_condition_for_product_greater_than_four (a b : ℝ) :
  a > 2 → b > 2 → a * b > 4 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_product_greater_than_four_l3444_344497


namespace NUMINAMATH_CALUDE_flat_percentage_calculation_l3444_344410

/-- The price of each flat -/
def flat_price : ℚ := 675958

/-- The overall gain from the transaction -/
def overall_gain : ℚ := 144 / 100

/-- The percentage of gain or loss on each flat -/
noncomputable def percentage : ℚ := overall_gain / (2 * flat_price) * 100

theorem flat_percentage_calculation :
  ∃ (ε : ℚ), abs (percentage - 1065 / 100000000) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_flat_percentage_calculation_l3444_344410


namespace NUMINAMATH_CALUDE_discounted_price_theorem_l3444_344422

/-- The selling price of two discounted items -/
def discounted_price (a : ℝ) : ℝ :=
  let original_price := a
  let markup_percentage := 0.5
  let discount_percentage := 0.2
  let marked_up_price := original_price * (1 + markup_percentage)
  let discounted_price := marked_up_price * (1 - discount_percentage)
  2 * discounted_price

/-- Theorem stating that the discounted price of two items is 2.4 times the original price -/
theorem discounted_price_theorem (a : ℝ) : discounted_price a = 2.4 * a := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_theorem_l3444_344422


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l3444_344490

theorem angle_measure_in_triangle (D E F : ℝ) : 
  D = 90 →  -- Angle D is 90 degrees
  E = 4 * F - 10 →  -- Angle E is 10 degrees less than four times angle F
  D + E + F = 180 →  -- Sum of angles in a triangle is 180 degrees
  F = 20 :=  -- Measure of angle F is 20 degrees
by sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l3444_344490


namespace NUMINAMATH_CALUDE_boat_production_three_months_l3444_344444

def boat_production (initial : ℕ) (months : ℕ) : ℕ :=
  if months = 0 then 0
  else if months = 1 then initial
  else initial + boat_production (initial * 3) (months - 1)

theorem boat_production_three_months :
  boat_production 5 3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_boat_production_three_months_l3444_344444


namespace NUMINAMATH_CALUDE_sum_of_double_root_k_values_l3444_344438

/-- The quadratic equation we're working with -/
def quadratic (k x : ℝ) : ℝ := x^2 + 2*k*x + 7*k - 10

/-- The discriminant of our quadratic equation -/
def discriminant (k : ℝ) : ℝ := (2*k)^2 - 4*(7*k - 10)

/-- A value of k for which the quadratic equation has exactly one solution -/
def is_double_root (k : ℝ) : Prop := discriminant k = 0

/-- The theorem stating that the sum of the values of k for which
    the quadratic equation has exactly one solution is 7 -/
theorem sum_of_double_root_k_values :
  ∃ k₁ k₂ : ℝ, k₁ ≠ k₂ ∧ is_double_root k₁ ∧ is_double_root k₂ ∧ k₁ + k₂ = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_double_root_k_values_l3444_344438


namespace NUMINAMATH_CALUDE_A_div_B_equals_37_l3444_344404

-- Define the series A
def A : ℝ := sorry

-- Define the series B
def B : ℝ := sorry

-- Theorem statement
theorem A_div_B_equals_37 : A / B = 37 := by sorry

end NUMINAMATH_CALUDE_A_div_B_equals_37_l3444_344404


namespace NUMINAMATH_CALUDE_chiquita_height_l3444_344485

theorem chiquita_height :
  ∀ (chiquita_height martinez_height : ℝ),
    martinez_height = chiquita_height + 2 →
    chiquita_height + martinez_height = 12 →
    chiquita_height = 5 := by
  sorry

end NUMINAMATH_CALUDE_chiquita_height_l3444_344485


namespace NUMINAMATH_CALUDE_building_floors_l3444_344474

-- Define the number of floors in each building
def alexie_floors : ℕ := sorry
def baptiste_floors : ℕ := sorry

-- Define the total number of bathrooms and bedrooms
def total_bathrooms : ℕ := 25
def total_bedrooms : ℕ := 18

-- State the theorem
theorem building_floors :
  (3 * alexie_floors + 4 * baptiste_floors = total_bathrooms) ∧
  (2 * alexie_floors + 3 * baptiste_floors = total_bedrooms) →
  alexie_floors = 3 ∧ baptiste_floors = 4 := by
  sorry

end NUMINAMATH_CALUDE_building_floors_l3444_344474


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l3444_344407

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance_theorem (A : ℝ × ℝ) :
  parabola A.1 A.2 →  -- A lies on the parabola
  (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = (B.1 - focus.1)^2 + (B.2 - focus.2)^2 →  -- |AF| = |BF|
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8 :=  -- |AB|^2 = 8, which implies |AB| = 2√2
by
  sorry


end NUMINAMATH_CALUDE_parabola_distance_theorem_l3444_344407


namespace NUMINAMATH_CALUDE_technician_average_salary_l3444_344433

/-- Calculates the average salary of technicians in a workshop --/
theorem technician_average_salary
  (total_workers : ℕ)
  (total_average : ℚ)
  (num_technicians : ℕ)
  (non_technician_average : ℚ)
  (h1 : total_workers = 30)
  (h2 : total_average = 8000)
  (h3 : num_technicians = 10)
  (h4 : non_technician_average = 6000)
  : (total_average * total_workers - non_technician_average * (total_workers - num_technicians)) / num_technicians = 12000 := by
  sorry

end NUMINAMATH_CALUDE_technician_average_salary_l3444_344433


namespace NUMINAMATH_CALUDE_integer_roots_fifth_degree_polynomial_l3444_344495

-- Define the set of possible values for m
def PossibleM : Set ℕ := {0, 1, 2, 3, 5}

-- Define a fifth-degree polynomial with integer coefficients
def FifthDegreePolynomial (a b c d e : ℤ) (x : ℤ) : ℤ :=
  x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

-- Define the number of integer roots (counting multiplicity)
def NumberOfIntegerRoots (p : ℤ → ℤ) : ℕ :=
  -- This is a placeholder definition. In reality, this would be more complex.
  0

-- The main theorem
theorem integer_roots_fifth_degree_polynomial 
  (a b c d e : ℤ) : 
  NumberOfIntegerRoots (FifthDegreePolynomial a b c d e) ∈ PossibleM :=
sorry

end NUMINAMATH_CALUDE_integer_roots_fifth_degree_polynomial_l3444_344495


namespace NUMINAMATH_CALUDE_symmetric_point_l3444_344430

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the line x + y = 0 -/
def symmetryLine (p : Point) : Prop :=
  p.x + p.y = 0

/-- Defines the property of two points being symmetric with respect to the line x + y = 0 -/
def isSymmetric (p1 p2 : Point) : Prop :=
  symmetryLine ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

/-- Theorem: The point symmetric to P(2, 5) with respect to the line x + y = 0 has coordinates (-5, -2) -/
theorem symmetric_point : 
  let p1 : Point := ⟨2, 5⟩
  let p2 : Point := ⟨-5, -2⟩
  isSymmetric p1 p2 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_l3444_344430


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l3444_344402

/-- Proves that adding 6 liters of 75% alcohol solution to 6 liters of 25% alcohol solution results in a 50% alcohol solution -/
theorem alcohol_mixture_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.25
  let added_volume : ℝ := 6
  let added_concentration : ℝ := 0.75
  let target_concentration : ℝ := 0.50
  let final_volume : ℝ := initial_volume + added_volume
  let final_alcohol_amount : ℝ := initial_volume * initial_concentration + added_volume * added_concentration
  final_alcohol_amount / final_volume = target_concentration := by
  sorry


end NUMINAMATH_CALUDE_alcohol_mixture_proof_l3444_344402


namespace NUMINAMATH_CALUDE_sam_washing_pennies_l3444_344409

/-- The number of pennies Sam got for washing clothes -/
def pennies_from_washing (total_cents : ℕ) (quarters : ℕ) : ℕ :=
  total_cents - quarters * 25

/-- Proof that Sam got 9 pennies for washing clothes -/
theorem sam_washing_pennies :
  pennies_from_washing 184 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sam_washing_pennies_l3444_344409


namespace NUMINAMATH_CALUDE_seat_difference_l3444_344427

/-- Represents the seating configuration of a bus --/
structure BusSeating where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeat : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- Theorem stating the difference in seats between left and right sides --/
theorem seat_difference (bus : BusSeating) : 
  bus.leftSeats = 15 →
  bus.backSeat = 12 →
  bus.seatCapacity = 3 →
  bus.totalCapacity = 93 →
  bus.leftSeats - bus.rightSeats = 3 := by
  sorry

#check seat_difference

end NUMINAMATH_CALUDE_seat_difference_l3444_344427


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l3444_344432

/-- The number of seats in the row -/
def num_seats : ℕ := 8

/-- The number of people to be seated -/
def num_people : ℕ := 3

/-- A function that calculates the number of valid seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) : ℕ :=
  sorry  -- The actual implementation would go here

/-- Theorem stating that the number of seating arrangements is 24 -/
theorem correct_seating_arrangements :
  seating_arrangements num_seats num_people = 24 := by sorry


end NUMINAMATH_CALUDE_correct_seating_arrangements_l3444_344432


namespace NUMINAMATH_CALUDE_square_root_of_1024_l3444_344458

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l3444_344458


namespace NUMINAMATH_CALUDE_directory_page_numbering_l3444_344464

/-- Calculate the total number of digits needed to number pages in a directory --/
def totalDigits (totalPages : ℕ) : ℕ :=
  let singleDigitPages := min totalPages 9
  let doubleDigitPages := min (max (totalPages - 9) 0) 90
  let tripleDigitPages := max (totalPages - 99) 0
  singleDigitPages * 1 + doubleDigitPages * 2 + tripleDigitPages * 3

/-- Theorem: A directory with 710 pages requires 2022 digits to number all pages --/
theorem directory_page_numbering :
  totalDigits 710 = 2022 := by sorry

end NUMINAMATH_CALUDE_directory_page_numbering_l3444_344464


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l3444_344441

def isGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_sequence_solution (a : ℝ) (h : a > 0) 
  (h_seq : isGeometricSequence 280 a (180/49)) : 
  a = Real.sqrt (50400/49) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l3444_344441


namespace NUMINAMATH_CALUDE_jerry_pool_time_l3444_344403

/-- Represents the time spent in the pool by each person --/
structure PoolTime where
  jerry : ℝ
  elaine : ℝ
  george : ℝ
  kramer : ℝ

/-- The conditions of the problem --/
def poolConditions (t : PoolTime) : Prop :=
  t.elaine = 2 * t.jerry ∧
  t.george = (1/3) * t.elaine ∧
  t.kramer = 0 ∧
  t.jerry + t.elaine + t.george + t.kramer = 11

/-- The theorem to be proved --/
theorem jerry_pool_time (t : PoolTime) :
  poolConditions t → t.jerry = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jerry_pool_time_l3444_344403


namespace NUMINAMATH_CALUDE_diagram_scale_l3444_344473

/-- Represents the scale of a diagram as a ratio of two natural numbers -/
structure Scale where
  numerator : ℕ
  denominator : ℕ

/-- Converts centimeters to millimeters -/
def cm_to_mm (cm : ℕ) : ℕ := cm * 10

theorem diagram_scale (actual_length_mm : ℕ) (diagram_length_cm : ℕ) :
  actual_length_mm = 4 →
  diagram_length_cm = 8 →
  ∃ (s : Scale), s.numerator = 20 ∧ s.denominator = 1 ∧
    cm_to_mm diagram_length_cm * s.denominator = actual_length_mm * s.numerator :=
by sorry

end NUMINAMATH_CALUDE_diagram_scale_l3444_344473


namespace NUMINAMATH_CALUDE_square_side_prime_l3444_344482

/-- Given an integer 'a' representing the side length of a square, if it's impossible to construct 
    a rectangle with the same area as the square where both sides of the rectangle are integers 
    greater than 1, then 'a' must be a prime number. -/
theorem square_side_prime (a : ℕ) (h : a > 1) : 
  (∀ m n : ℕ, m > 1 → n > 1 → m * n ≠ a * a) → Nat.Prime a := by
  sorry

end NUMINAMATH_CALUDE_square_side_prime_l3444_344482


namespace NUMINAMATH_CALUDE_man_downstream_speed_l3444_344466

/-- The speed of a man rowing in a stream -/
structure RowingSpeed :=
  (still : ℝ)        -- Speed in still water
  (upstream : ℝ)     -- Speed upstream
  (downstream : ℝ)   -- Speed downstream

/-- Calculate the downstream speed given still water and upstream speeds -/
def calculate_downstream_speed (s : RowingSpeed) : Prop :=
  s.downstream = s.still + (s.still - s.upstream)

/-- Theorem: The man's downstream speed is 55 kmph -/
theorem man_downstream_speed :
  ∃ (s : RowingSpeed), s.still = 50 ∧ s.upstream = 45 ∧ s.downstream = 55 ∧ calculate_downstream_speed s :=
sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l3444_344466


namespace NUMINAMATH_CALUDE_sarah_flour_amount_l3444_344440

/-- The amount of rye flour Sarah bought in pounds -/
def rye_flour : ℕ := 5

/-- The amount of whole-wheat bread flour Sarah bought in pounds -/
def wheat_bread_flour : ℕ := 10

/-- The amount of chickpea flour Sarah bought in pounds -/
def chickpea_flour : ℕ := 3

/-- The amount of whole-wheat pastry flour Sarah already had at home in pounds -/
def pastry_flour : ℕ := 2

/-- The total amount of flour Sarah now has in pounds -/
def total_flour : ℕ := rye_flour + wheat_bread_flour + chickpea_flour + pastry_flour

theorem sarah_flour_amount : total_flour = 20 := by
  sorry

end NUMINAMATH_CALUDE_sarah_flour_amount_l3444_344440


namespace NUMINAMATH_CALUDE_f_uniqueness_and_fixed_points_l3444_344472

def is_prime (p : ℕ) : Prop := Nat.Prime p

def f_conditions (f : ℕ → ℕ) : Prop :=
  (∀ p, is_prime p → f p = 1) ∧
  (∀ a b, f (a * b) = a * f b + f a * b)

theorem f_uniqueness_and_fixed_points (f : ℕ → ℕ) (h : f_conditions f) :
  (∀ g, f_conditions g → f = g) ∧
  (∀ n, n = f n ↔ ∃ p, is_prime p ∧ n = p^p) :=
sorry

end NUMINAMATH_CALUDE_f_uniqueness_and_fixed_points_l3444_344472


namespace NUMINAMATH_CALUDE_angle_KJG_measure_l3444_344417

-- Define the geometric configuration
structure GeometricConfig where
  -- JKL is a 45-45-90 right triangle
  JKL_is_45_45_90 : Bool
  -- GHIJ is a square
  GHIJ_is_square : Bool
  -- JKLK is a square
  JKLK_is_square : Bool

-- Define the theorem
theorem angle_KJG_measure (config : GeometricConfig) 
  (h1 : config.JKL_is_45_45_90 = true)
  (h2 : config.GHIJ_is_square = true)
  (h3 : config.JKLK_is_square = true) :
  ∃ (angle_KJG : ℝ), angle_KJG = 135 := by
  sorry


end NUMINAMATH_CALUDE_angle_KJG_measure_l3444_344417


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3444_344428

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and the length of the real axis is 4 and the length of the imaginary axis is 6,
    prove that the equation of its asymptotes is y = ±(3/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (real_axis : 2 * a = 4) (imag_axis : 2 * b = 6) :
  ∃ (k : ℝ), k = 3/2 ∧ (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k*x ∨ y = -k*x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3444_344428


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3444_344425

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 3 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3444_344425


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l3444_344408

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_increasing : ∀ x y, x < y → f x < f y
axiom f_0 : f 0 = -1
axiom f_3 : f 3 = 1

-- Define the solution set
def solution_set := {x : ℝ | |f x| < 1}

-- State the theorem
theorem solution_set_is_open_interval :
  solution_set f = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l3444_344408


namespace NUMINAMATH_CALUDE_target_breaking_orders_l3444_344452

/-- The number of targets in the first column -/
def column_A : ℕ := 4

/-- The number of targets in the second column -/
def column_B : ℕ := 3

/-- The number of targets in the third column -/
def column_C : ℕ := 3

/-- The total number of targets -/
def total_targets : ℕ := column_A + column_B + column_C

/-- The number of different orders to break the targets -/
def break_orders : ℕ := (Nat.factorial total_targets) / 
  (Nat.factorial column_A * Nat.factorial column_B * Nat.factorial column_C)

theorem target_breaking_orders : break_orders = 4200 := by
  sorry

end NUMINAMATH_CALUDE_target_breaking_orders_l3444_344452


namespace NUMINAMATH_CALUDE_recurrence_sequence_uniqueness_l3444_344439

/-- A sequence of natural numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

/-- A bounded sequence of natural numbers -/
def BoundedSequence (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n, a n ≤ M

theorem recurrence_sequence_uniqueness (a : ℕ → ℕ) 
  (h_recurrence : RecurrenceSequence a) (h_bounded : BoundedSequence a) :
  ∀ n, a n = 2 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_uniqueness_l3444_344439


namespace NUMINAMATH_CALUDE_gcd_6Tn_nplus1_eq_one_l3444_344478

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := (n * (n + 1)) / 2

/-- Theorem: The GCD of 6T_n and n+1 is always 1 for positive integers n -/
theorem gcd_6Tn_nplus1_eq_one (n : ℕ+) : Nat.gcd (6 * T n) (n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_6Tn_nplus1_eq_one_l3444_344478


namespace NUMINAMATH_CALUDE_statement_equivalence_l3444_344434

theorem statement_equivalence (x y : ℝ) :
  ((x - 1) * (y + 2) ≠ 0 → x ≠ 1 ∧ y ≠ -2) ↔
  (x = 1 ∨ y = -2 → (x - 1) * (y + 2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l3444_344434


namespace NUMINAMATH_CALUDE_valid_seats_29x29_l3444_344471

/-- Represents a grid of seats -/
def Grid (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if two positions in the grid are adjacent -/
def adjacent (n : ℕ) (p q : Fin n × Fin n) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Counts the number of valid seats in an n x n grid -/
def validSeats (n : ℕ) : ℕ := sorry

/-- The main theorem to be proved -/
theorem valid_seats_29x29 :
  validSeats 29 = 421 :=
sorry

end NUMINAMATH_CALUDE_valid_seats_29x29_l3444_344471


namespace NUMINAMATH_CALUDE_soccer_team_starters_l3444_344465

theorem soccer_team_starters (total_players : ℕ) (first_half_subs : ℕ) (players_not_played : ℕ) :
  total_players = 24 →
  first_half_subs = 2 →
  players_not_played = 7 →
  ∃ (starters : ℕ), starters = 11 ∧ 
    starters + first_half_subs + 2 * first_half_subs + players_not_played = total_players :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l3444_344465


namespace NUMINAMATH_CALUDE_min_blue_eyes_and_lunch_box_l3444_344461

theorem min_blue_eyes_and_lunch_box 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 35) 
  (h2 : blue_eyes = 15) 
  (h3 : lunch_box = 25) :
  ∃ (overlap : ℕ), 
    overlap ≥ 5 ∧ 
    overlap ≤ blue_eyes ∧ 
    overlap ≤ lunch_box ∧ 
    (∀ (x : ℕ), x < overlap → 
      x + (total_students - lunch_box) < blue_eyes ∨ 
      x + (total_students - blue_eyes) < lunch_box) :=
by sorry

end NUMINAMATH_CALUDE_min_blue_eyes_and_lunch_box_l3444_344461


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3444_344467

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 5*x + 1) * (y^2 + 5*y + 1) * (z^2 + 5*z + 1) / (x*y*z) ≥ 343 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 1) * (b^2 + 5*b + 1) * (c^2 + 5*c + 1) / (a*b*c) = 343 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3444_344467


namespace NUMINAMATH_CALUDE_expected_boy_girl_pairs_l3444_344415

theorem expected_boy_girl_pairs (n_boys n_girls : ℕ) (h_boys : n_boys = 8) (h_girls : n_girls = 12) :
  let total := n_boys + n_girls
  let inner_boys := n_boys - 2
  let inner_pairs := total - 1
  let inner_prob := (inner_boys * n_girls) / ((inner_boys + n_girls) * (inner_boys + n_girls - 1))
  let end_prob := n_girls / total
  (inner_pairs - 2) * (2 * inner_prob) + 2 * end_prob = 144/17 + 24/19 := by
  sorry

end NUMINAMATH_CALUDE_expected_boy_girl_pairs_l3444_344415


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l3444_344487

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Angles are positive
  a + b = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  a / b = 5 / 4 →  -- Ratio of angles is 5:4
  (a = 50 ∧ b = 40) ∨ (a = 40 ∧ b = 50) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l3444_344487


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3444_344449

/-- 
Given a rectangular prism with edges in the ratio 2:1:1.5 and a total edge length of 72 cm,
prove that its volume is 192 cubic centimeters.
-/
theorem rectangular_prism_volume (x : ℝ) 
  (h1 : x > 0)
  (h2 : 4 * (2*x) + 4 * x + 4 * (1.5*x) = 72) : 
  (2*x) * x * (1.5*x) = 192 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3444_344449


namespace NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l3444_344493

/-- Calculate the amount John paid out of pocket for his new computer setup --/
theorem johns_out_of_pocket_expense :
  let computer_cost : ℚ := 1200
  let computer_discount : ℚ := 0.15
  let chair_cost : ℚ := 300
  let chair_discount : ℚ := 0.10
  let accessories_cost : ℚ := 350
  let sales_tax_rate : ℚ := 0.08
  let playstation_value : ℚ := 500
  let playstation_discount : ℚ := 0.30
  let bicycle_sale : ℚ := 100

  let discounted_computer := computer_cost * (1 - computer_discount)
  let discounted_chair := chair_cost * (1 - chair_discount)
  let total_before_tax := discounted_computer + discounted_chair + accessories_cost
  let total_with_tax := total_before_tax * (1 + sales_tax_rate)
  let sold_items := playstation_value * (1 - playstation_discount) + bicycle_sale
  let out_of_pocket := total_with_tax - sold_items

  out_of_pocket = 1321.20
  := by sorry

end NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l3444_344493


namespace NUMINAMATH_CALUDE_circle_and_intersection_conditions_l3444_344456

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_and_intersection_conditions (m : ℝ) :
  (∀ x y, circle_equation x y m → m < 5) ∧
  (∃ x1 y1 x2 y2, 
    circle_equation x1 y1 m ∧
    circle_equation x2 y2 m ∧
    line_equation x1 y1 ∧
    line_equation x2 y2 ∧
    perpendicular x1 y1 x2 y2 →
    m = 8/5) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_intersection_conditions_l3444_344456


namespace NUMINAMATH_CALUDE_weight_of_A_l3444_344420

-- Define the weights and ages of persons A, B, C, D, and E
variable (W_A W_B W_C W_D W_E : ℝ)
variable (Age_A Age_B Age_C Age_D Age_E : ℝ)

-- State the conditions from the problem
axiom avg_weight_ABC : (W_A + W_B + W_C) / 3 = 84
axiom avg_age_ABC : (Age_A + Age_B + Age_C) / 3 = 30
axiom avg_weight_ABCD : (W_A + W_B + W_C + W_D) / 4 = 80
axiom avg_age_ABCD : (Age_A + Age_B + Age_C + Age_D) / 4 = 28
axiom avg_weight_BCDE : (W_B + W_C + W_D + W_E) / 4 = 79
axiom avg_age_BCDE : (Age_B + Age_C + Age_D + Age_E) / 4 = 27
axiom weight_E : W_E = W_D + 7
axiom age_E : Age_E = Age_A - 3

-- State the theorem to be proved
theorem weight_of_A : W_A = 79 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_A_l3444_344420


namespace NUMINAMATH_CALUDE_james_works_six_hours_l3444_344421

-- Define the cleaning times and number of rooms
def num_bedrooms : ℕ := 3
def num_bathrooms : ℕ := 2
def bedroom_cleaning_time : ℕ := 20 -- in minutes

-- Define the relationships between cleaning times
def living_room_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time
def bathroom_cleaning_time : ℕ := 2 * living_room_cleaning_time
def house_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time + living_room_cleaning_time + num_bathrooms * bathroom_cleaning_time
def outside_cleaning_time : ℕ := 2 * house_cleaning_time
def total_cleaning_time : ℕ := house_cleaning_time + outside_cleaning_time
def num_siblings : ℕ := 3

-- Define James' working time
def james_working_time : ℚ := (total_cleaning_time / num_siblings) / 60

-- Theorem statement
theorem james_works_six_hours : james_working_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_james_works_six_hours_l3444_344421


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3444_344470

theorem right_triangle_third_side : ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  ((a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4)) →
  c = 5 ∨ c = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3444_344470


namespace NUMINAMATH_CALUDE_tennis_ball_price_is_6_l3444_344479

/-- The price of a tennis ball in yuan -/
def tennis_ball_price : ℝ := 6

/-- The price of a tennis racket in yuan -/
def tennis_racket_price : ℝ := tennis_ball_price + 83

/-- The total cost of 2 tennis rackets and 7 tennis balls in yuan -/
def total_cost : ℝ := 220

theorem tennis_ball_price_is_6 :
  (2 * tennis_racket_price + 7 * tennis_ball_price = total_cost) ∧
  (tennis_racket_price = tennis_ball_price + 83) →
  tennis_ball_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_tennis_ball_price_is_6_l3444_344479


namespace NUMINAMATH_CALUDE_correct_sum_after_misreading_l3444_344477

/-- Given a three-digit number ABC where C was misread as 6 instead of 9,
    and the sum of AB6 and 57 is 823, prove that the correct sum of ABC and 57 is 826 -/
theorem correct_sum_after_misreading (A B : Nat) : 
  (100 * A + 10 * B + 6 + 57 = 823) → 
  (100 * A + 10 * B + 9 + 57 = 826) :=
by sorry

end NUMINAMATH_CALUDE_correct_sum_after_misreading_l3444_344477


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3444_344476

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3444_344476


namespace NUMINAMATH_CALUDE_four_tangent_lines_l3444_344426

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Counts the number of common tangent lines to two circles -/
def countCommonTangents (c1 c2 : Circle) : ℕ := sorry

/-- The main theorem -/
theorem four_tangent_lines (c1 c2 : Circle) 
  (h1 : c1.radius = 5)
  (h2 : c2.radius = 2)
  (h3 : Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 10) :
  countCommonTangents c1 c2 = 4 := by sorry

end NUMINAMATH_CALUDE_four_tangent_lines_l3444_344426


namespace NUMINAMATH_CALUDE_parallelogram_area_and_magnitude_l3444_344480

/-- Given a complex number z with positive real part, if the parallelogram formed by 
    0, z, z², and z + z² has area 20/29, then the smallest possible value of |z² + z| 
    is (r² + r), where r³|sin θ| = 20/29 and z = r(cos θ + i sin θ). -/
theorem parallelogram_area_and_magnitude (z : ℂ) (r θ : ℝ) (h1 : z.re > 0) 
  (h2 : z = r * Complex.exp (θ * Complex.I)) 
  (h3 : r > 0) 
  (h4 : r^3 * |Real.sin θ| = 20/29) 
  (h5 : Complex.abs (z * z - z) = 20/29) : 
  ∃ (d : ℝ), d = r^2 + r ∧ 
  ∀ (w : ℂ), Complex.abs (w^2 + w) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_and_magnitude_l3444_344480


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3444_344462

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

theorem inequality_solution_set (x : ℝ) :
  (x ∈ Set.Ioo (Real.exp (-1)) (Real.exp 1)) ↔
  (f (Real.log x) + f (Real.log (1/x)) < 2 * f 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3444_344462


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l3444_344499

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l3444_344499


namespace NUMINAMATH_CALUDE_floor_with_133_black_tiles_has_4489_total_tiles_l3444_344431

/-- Represents a square floor covered with tiles -/
structure TiledFloor where
  side : ℕ
  black_tiles : ℕ

/-- The number of black tiles on the diagonals of a square floor -/
def diagonal_tiles (floor : TiledFloor) : ℕ :=
  2 * floor.side - 1

/-- The total number of tiles on a square floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.side ^ 2

/-- Theorem stating that a square floor with 133 black tiles on its diagonals has 4489 total tiles -/
theorem floor_with_133_black_tiles_has_4489_total_tiles (floor : TiledFloor) 
    (h : floor.black_tiles = 133) : total_tiles floor = 4489 := by
  sorry


end NUMINAMATH_CALUDE_floor_with_133_black_tiles_has_4489_total_tiles_l3444_344431


namespace NUMINAMATH_CALUDE_ratio_problem_l3444_344459

theorem ratio_problem (x : ℝ) : 0.75 / x = 5 / 7 → x = 1.05 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3444_344459


namespace NUMINAMATH_CALUDE_cloth_cost_price_l3444_344481

/-- Calculates the cost price per meter of cloth given total meters, selling price, and profit per meter -/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_meters) / total_meters

/-- Proves that the cost price of one meter of cloth is 5 Rs. given the problem conditions -/
theorem cloth_cost_price :
  cost_price_per_meter 66 660 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l3444_344481
