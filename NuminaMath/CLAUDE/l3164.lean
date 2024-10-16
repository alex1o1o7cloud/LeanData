import Mathlib

namespace NUMINAMATH_CALUDE_prob_non_intersecting_chords_l3164_316446

/-- The probability of non-intersecting chords when pairing 2n points on a circle -/
theorem prob_non_intersecting_chords (n : ℕ) : 
  ∃ (P : ℚ), P = (2^n : ℚ) / (n + 1).factorial := by
  sorry

end NUMINAMATH_CALUDE_prob_non_intersecting_chords_l3164_316446


namespace NUMINAMATH_CALUDE_tree_space_calculation_l3164_316440

/-- Given a road of length 151 feet where 11 trees are planted with 14 feet between each tree,
    prove that each tree occupies 1 square foot of sidewalk space. -/
theorem tree_space_calculation (road_length : ℕ) (num_trees : ℕ) (gap_between_trees : ℕ) :
  road_length = 151 →
  num_trees = 11 →
  gap_between_trees = 14 →
  (road_length - (num_trees - 1) * gap_between_trees) / num_trees = 1 := by
  sorry

end NUMINAMATH_CALUDE_tree_space_calculation_l3164_316440


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3164_316453

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 3 →
  a 1 + a 6 = 12 →
  a 7 + a 8 + a 9 = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3164_316453


namespace NUMINAMATH_CALUDE_fifth_flower_is_e_l3164_316409

def flowers := ['a', 'b', 'c', 'd', 'e', 'f', 'g']

theorem fifth_flower_is_e : flowers[4] = 'e' := by
  sorry

end NUMINAMATH_CALUDE_fifth_flower_is_e_l3164_316409


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3164_316405

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3164_316405


namespace NUMINAMATH_CALUDE_dubblefud_red_balls_l3164_316485

/-- The number of red balls in a Dubblefud game selection -/
def num_red_balls (r b g : ℕ) : Prop :=
  (2 ^ r) * (4 ^ b) * (5 ^ g) = 16000 ∧ b = g ∧ r = 0

/-- Theorem stating that the number of red balls is 0 given the conditions -/
theorem dubblefud_red_balls :
  ∃ (r b g : ℕ), num_red_balls r b g :=
sorry

end NUMINAMATH_CALUDE_dubblefud_red_balls_l3164_316485


namespace NUMINAMATH_CALUDE_sum_of_square_roots_geq_one_l3164_316432

theorem sum_of_square_roots_geq_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a / (a + 3 * b)) + Real.sqrt (b / (b + 3 * a)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_geq_one_l3164_316432


namespace NUMINAMATH_CALUDE_compare_expressions_l3164_316456

theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (a + 1/a) * (b + 1/b) > (Real.sqrt (a*b) + 1/Real.sqrt (a*b))^2 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (a + 1/a) * (b + 1/b) > ((a+b)/2 + 2/(a+b))^2 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (a + 1/a) * (b + 1/b) < ((a+b)/2 + 2/(a+b))^2 :=
by sorry

end NUMINAMATH_CALUDE_compare_expressions_l3164_316456


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l3164_316472

def arithmetic_sequence_sum (a l d : ℤ) : ℤ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_specific :
  arithmetic_sequence_sum (-45) 3 4 = -273 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l3164_316472


namespace NUMINAMATH_CALUDE_slope_height_calculation_l3164_316499

theorem slope_height_calculation (slope_ratio : Real) (distance : Real) (height : Real) : 
  slope_ratio = 1 / 2.4 →
  distance = 130 →
  height ^ 2 + (height * 2.4) ^ 2 = distance ^ 2 →
  height = 50 := by
sorry

end NUMINAMATH_CALUDE_slope_height_calculation_l3164_316499


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3164_316495

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3164_316495


namespace NUMINAMATH_CALUDE_no_integer_solution_l3164_316412

theorem no_integer_solution : ∀ x y : ℤ, x^2 - 37*y ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3164_316412


namespace NUMINAMATH_CALUDE_fraction_division_l3164_316478

theorem fraction_division (x y z : ℚ) (hx : x = 3) (hy : y = 4) (hz : z = 5) :
  (z / y) / (z / x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l3164_316478


namespace NUMINAMATH_CALUDE_problem_solution_l3164_316497

theorem problem_solution (x y z M : ℚ) 
  (sum_eq : x + y + z = 120)
  (x_eq : x - 10 = M)
  (y_eq : y + 10 = M)
  (z_eq : 10 * z = M) :
  M = 400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3164_316497


namespace NUMINAMATH_CALUDE_allen_reading_time_l3164_316488

/-- Calculates the number of days required to finish reading a book -/
def days_to_finish (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Proves that Allen took 12 days to finish reading the book -/
theorem allen_reading_time : days_to_finish 120 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_allen_reading_time_l3164_316488


namespace NUMINAMATH_CALUDE_min_value_expression_l3164_316481

theorem min_value_expression (x y z : ℝ) (hx : x ≥ 3) (hy : y ≥ 3) (hz : z ≥ 3) :
  let A := ((x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3)) / (x*y + y*z + z*x)
  A ≥ 1 ∧ (A = 1 ↔ x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3164_316481


namespace NUMINAMATH_CALUDE_weather_conditions_on_july_15_l3164_316473

/-- Represents the weather conditions at the beach --/
structure WeatherCondition where
  temperature : ℝ
  sunny : Bool
  windSpeed : ℝ

/-- Predicate to determine if the beach is crowded based on weather conditions --/
def isCrowded (w : WeatherCondition) : Prop :=
  w.temperature ≥ 85 ∧ w.sunny ∧ w.windSpeed < 10

/-- Theorem: Given that the beach is not crowded on July 15, prove that the weather conditions
    must satisfy: temperature < 85°F or not sunny or wind speed ≥ 10 mph --/
theorem weather_conditions_on_july_15 (w : WeatherCondition) 
  (h : ¬isCrowded w) : 
  w.temperature < 85 ∨ ¬w.sunny ∨ w.windSpeed ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_weather_conditions_on_july_15_l3164_316473


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3164_316447

theorem solve_exponential_equation :
  ∃ x : ℝ, 5^(3*x) = (125 : ℝ)^(1/3) ∧ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3164_316447


namespace NUMINAMATH_CALUDE_mean_median_difference_l3164_316404

/-- Represents the score distribution in a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_75_percent : ℚ
  score_82_percent : ℚ
  score_87_percent : ℚ
  score_90_percent : ℚ
  score_98_percent : ℚ

/-- Calculates the mean score given a score distribution -/
def mean_score (sd : ScoreDistribution) : ℚ :=
  (75 * sd.score_75_percent + 82 * sd.score_82_percent + 87 * sd.score_87_percent +
   90 * sd.score_90_percent + 98 * sd.score_98_percent) / 100

/-- Calculates the median score given a score distribution -/
def median_score (sd : ScoreDistribution) : ℚ := 87

/-- The main theorem stating the difference between mean and median scores -/
theorem mean_median_difference (sd : ScoreDistribution) 
  (h1 : sd.total_students = 10)
  (h2 : sd.score_75_percent = 15)
  (h3 : sd.score_82_percent = 10)
  (h4 : sd.score_87_percent = 40)
  (h5 : sd.score_90_percent = 20)
  (h6 : sd.score_98_percent = 15) :
  |mean_score sd - median_score sd| = 9 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l3164_316404


namespace NUMINAMATH_CALUDE_ratio_equality_l3164_316476

theorem ratio_equality (x : ℝ) : (0.6 / x = 5 / 8) → x = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3164_316476


namespace NUMINAMATH_CALUDE_liliane_alice_relationship_l3164_316413

/-- Represents the amount of soda each person has -/
structure SodaAmounts where
  jacqueline : ℝ
  liliane : ℝ
  alice : ℝ
  bruno : ℝ

/-- The conditions of the soda problem -/
def SodaProblem (amounts : SodaAmounts) : Prop :=
  amounts.liliane = amounts.jacqueline * 1.6 ∧
  amounts.alice = amounts.jacqueline * 1.4 ∧
  amounts.bruno = amounts.jacqueline * 0.8

/-- The theorem stating the relationship between Liliane's and Alice's soda amounts -/
theorem liliane_alice_relationship (amounts : SodaAmounts) 
  (h : SodaProblem amounts) : 
  ∃ ε > 0, ε < 0.005 ∧ amounts.liliane = amounts.alice * (1 + 0.15 + ε) :=
sorry

end NUMINAMATH_CALUDE_liliane_alice_relationship_l3164_316413


namespace NUMINAMATH_CALUDE_soda_price_ratio_l3164_316461

/-- The ratio of unit prices between two soda brands -/
theorem soda_price_ratio 
  (volume_A : ℝ) (volume_B : ℝ) (price_A : ℝ) (price_B : ℝ)
  (h_volume : volume_A = 1.25 * volume_B)
  (h_price : price_A = 0.85 * price_B)
  (h_positive : volume_B > 0 ∧ price_B > 0) :
  (price_A / volume_A) / (price_B / volume_B) = 17 / 25 := by
sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l3164_316461


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l3164_316452

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 + (m-1)x -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m - 1) * x

theorem even_function_implies_m_equals_one (m : ℝ) :
  IsEven (f m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l3164_316452


namespace NUMINAMATH_CALUDE_root_equation_a_value_l3164_316458

theorem root_equation_a_value 
  (x₁ x₂ x₃ a b : ℚ) : 
  x₁ = -3 - 5 * Real.sqrt 3 → 
  x₂ = -3 + 5 * Real.sqrt 3 → 
  x₃ = 15 / 11 → 
  x₁ * x₂ * x₃ = -90 → 
  x₁^3 + a*x₁^2 + b*x₁ + 90 = 0 → 
  a = -15 / 11 := by
sorry

end NUMINAMATH_CALUDE_root_equation_a_value_l3164_316458


namespace NUMINAMATH_CALUDE_lemon_profit_problem_l3164_316408

theorem lemon_profit_problem (buy_lemons : ℕ) (buy_cost : ℚ) (sell_lemons : ℕ) (sell_price : ℚ) (target_profit : ℚ) : 
  buy_lemons = 4 →
  buy_cost = 15 →
  sell_lemons = 6 →
  sell_price = 25 →
  target_profit = 120 →
  ∃ (n : ℕ), n = 286 ∧ 
    n * (sell_price / sell_lemons - buy_cost / buy_lemons) ≥ target_profit ∧
    (n - 1) * (sell_price / sell_lemons - buy_cost / buy_lemons) < target_profit :=
by sorry

end NUMINAMATH_CALUDE_lemon_profit_problem_l3164_316408


namespace NUMINAMATH_CALUDE_translation_iff_equal_movements_l3164_316465

/-- Represents the movement of a table's legs -/
structure TableMovement where
  leg1 : ℝ
  leg2 : ℝ
  leg3 : ℝ
  leg4 : ℝ

/-- Determines if a table movement represents a translation -/
def isTranslation (m : TableMovement) : Prop :=
  m.leg1 = m.leg2 ∧ m.leg2 = m.leg3 ∧ m.leg3 = m.leg4

/-- Theorem: A table movement is a translation if and only if all leg movements are equal -/
theorem translation_iff_equal_movements (m : TableMovement) :
  isTranslation m ↔ m.leg1 = m.leg2 ∧ m.leg1 = m.leg3 ∧ m.leg1 = m.leg4 := by sorry

end NUMINAMATH_CALUDE_translation_iff_equal_movements_l3164_316465


namespace NUMINAMATH_CALUDE_gcd_4050_12150_l3164_316433

theorem gcd_4050_12150 : Nat.gcd 4050 12150 = 450 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4050_12150_l3164_316433


namespace NUMINAMATH_CALUDE_degenerate_ellipse_c_l3164_316487

/-- The equation of a potentially degenerate ellipse -/
def ellipse_equation (x y c : ℝ) : Prop :=
  2 * x^2 + y^2 + 8 * x - 10 * y + c = 0

/-- A degenerate ellipse is represented by a single point -/
def is_degenerate_ellipse (c : ℝ) : Prop :=
  ∃! (x y : ℝ), ellipse_equation x y c

/-- The value of c for which the ellipse is degenerate -/
theorem degenerate_ellipse_c : 
  ∃! c : ℝ, is_degenerate_ellipse c ∧ c = 33 :=
sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_c_l3164_316487


namespace NUMINAMATH_CALUDE_sin_15_minus_sin_75_l3164_316469

theorem sin_15_minus_sin_75 : 
  Real.sin (15 * π / 180) - Real.sin (75 * π / 180) = -Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_minus_sin_75_l3164_316469


namespace NUMINAMATH_CALUDE_min_value_ratio_l3164_316494

-- Define the arithmetic and geometric sequence properties
def is_arithmetic_sequence (x a b y : ℝ) : Prop :=
  a + b = x + y

def is_geometric_sequence (x c d y : ℝ) : Prop :=
  c * d = x * y

-- State the theorem
theorem min_value_ratio (x y a b c d : ℝ) 
  (hx : x > 0) (hy : y > 0)
  (ha : is_arithmetic_sequence x a b y)
  (hg : is_geometric_sequence x c d y) :
  (a + b)^2 / (c * d) ≥ 4 ∧ 
  ∃ (a b c d : ℝ), (a + b)^2 / (c * d) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_ratio_l3164_316494


namespace NUMINAMATH_CALUDE_y_share_is_63_l3164_316431

/-- Represents the share of each person in rupees -/
structure Share where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount to be divided -/
def total_amount : ℝ := 245

/-- The ratio of y's share to x's share -/
def y_ratio : ℝ := 0.45

/-- The ratio of z's share to x's share -/
def z_ratio : ℝ := 0.30

/-- The share satisfies the given conditions -/
def is_valid_share (s : Share) : Prop :=
  s.x + s.y + s.z = total_amount ∧
  s.y = y_ratio * s.x ∧
  s.z = z_ratio * s.x

theorem y_share_is_63 :
  ∃ (s : Share), is_valid_share s ∧ s.y = 63 := by
  sorry

end NUMINAMATH_CALUDE_y_share_is_63_l3164_316431


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3164_316434

theorem sqrt_product_equality (x y : ℝ) (hx : x ≥ 0) :
  Real.sqrt (3 * x) * Real.sqrt ((1 / 3) * x * y) = x * Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3164_316434


namespace NUMINAMATH_CALUDE_square_difference_1001_999_l3164_316455

theorem square_difference_1001_999 : 1001^2 - 999^2 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_1001_999_l3164_316455


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l3164_316429

/-- A rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- A hexagon inscribed in a rectangle, with vertices touching midpoints of the rectangle's edges -/
structure InscribedHexagon (r : Rectangle) where

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The area of an inscribed hexagon -/
def InscribedHexagon.area (h : InscribedHexagon r) : ℝ := sorry

theorem inscribed_hexagon_area (r : Rectangle) (h : InscribedHexagon r) 
    (h_width : r.width = 5) (h_height : r.height = 4) : 
    InscribedHexagon.area h = 10 := by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l3164_316429


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3164_316466

theorem scientific_notation_equivalence : 
  ∃ (a : ℝ) (n : ℤ), 0.0000002 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.0 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3164_316466


namespace NUMINAMATH_CALUDE_data_analysis_l3164_316435

def data : List ℝ := [11, 10, 11, 13, 11, 13, 15]

def mode (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

theorem data_analysis :
  mode data = 11 ∧
  mean data = 12 ∧
  variance data = 18/7 ∧
  median data = 11 := by sorry

end NUMINAMATH_CALUDE_data_analysis_l3164_316435


namespace NUMINAMATH_CALUDE_chord_length_single_intersection_l3164_316451

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the line with slope k passing through (0, -1)
def line (k x y : ℝ) : Prop := y = k * x - 1

-- Theorem 1: Length of the chord when k = 2
theorem chord_length : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line 2 x₁ y₁ ∧ line 2 x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 75 :=
sorry

-- Theorem 2: Values of k for single intersection point
theorem single_intersection :
  ∀ k : ℝ,
    (∃! (x y : ℝ), parabola x y ∧ line k x y) ↔ (k = 0 ∨ k = -3) :=
sorry

end NUMINAMATH_CALUDE_chord_length_single_intersection_l3164_316451


namespace NUMINAMATH_CALUDE_arithmetic_problem_l3164_316414

theorem arithmetic_problem : (20 * 24) / (2 * 0 + 2 * 4) = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l3164_316414


namespace NUMINAMATH_CALUDE_sum_four_characterization_l3164_316477

/-- Represents the outcome of rolling a single die -/
def DieOutcome := Fin 6

/-- Represents the outcome of rolling two dice -/
def TwoDiceOutcome := DieOutcome × DieOutcome

/-- The sum of points obtained when rolling two dice -/
def sumPoints (outcome : TwoDiceOutcome) : Nat :=
  outcome.1.val + 1 + outcome.2.val + 1

/-- The event where the sum of points is 4 -/
def sumIsFour (outcome : TwoDiceOutcome) : Prop :=
  sumPoints outcome = 4

/-- The event where one die shows 3 and the other shows 1 -/
def threeAndOne (outcome : TwoDiceOutcome) : Prop :=
  (outcome.1.val = 2 ∧ outcome.2.val = 0) ∨ (outcome.1.val = 0 ∧ outcome.2.val = 2)

/-- The event where both dice show 2 -/
def bothTwo (outcome : TwoDiceOutcome) : Prop :=
  outcome.1.val = 1 ∧ outcome.2.val = 1

theorem sum_four_characterization (outcome : TwoDiceOutcome) :
  sumIsFour outcome ↔ threeAndOne outcome ∨ bothTwo outcome := by
  sorry

end NUMINAMATH_CALUDE_sum_four_characterization_l3164_316477


namespace NUMINAMATH_CALUDE_max_k_for_arithmetic_sequences_l3164_316474

/-- An arithmetic sequence -/
def ArithmeticSequence (a d : ℝ) : ℕ → ℝ := fun n ↦ a + (n - 1) * d

theorem max_k_for_arithmetic_sequences (a₁ a₂ d₁ d₂ : ℝ) (k : ℕ) :
  k > 1 →
  (ArithmeticSequence a₁ d₁ (k - 1)) * (ArithmeticSequence a₂ d₂ (k - 1)) = 42 →
  (ArithmeticSequence a₁ d₁ k) * (ArithmeticSequence a₂ d₂ k) = 30 →
  (ArithmeticSequence a₁ d₁ (k + 1)) * (ArithmeticSequence a₂ d₂ (k + 1)) = 16 →
  a₁ = a₂ →
  k ≤ 14 ∧ ∃ (a d₁ d₂ : ℝ), k = 14 ∧
    (ArithmeticSequence a d₁ 13) * (ArithmeticSequence a d₂ 13) = 42 ∧
    (ArithmeticSequence a d₁ 14) * (ArithmeticSequence a d₂ 14) = 30 ∧
    (ArithmeticSequence a d₁ 15) * (ArithmeticSequence a d₂ 15) = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_k_for_arithmetic_sequences_l3164_316474


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3164_316425

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 14 → b = 48 → c^2 = a^2 + b^2 → c = 50 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3164_316425


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l3164_316424

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_value_reciprocal_sum_achieved (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 1 ∧ (1 / m₀ + 1 / n₀ = 3 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l3164_316424


namespace NUMINAMATH_CALUDE_total_students_proof_l3164_316439

/-- Represents the total number of senior students -/
def total_students : ℕ := 300

/-- Represents the number of students who didn't receive scholarships -/
def no_scholarship_students : ℕ := 255

/-- Percentage of students who received full merit scholarships -/
def full_scholarship_percent : ℚ := 5 / 100

/-- Percentage of students who received half merit scholarships -/
def half_scholarship_percent : ℚ := 10 / 100

/-- Theorem stating that the total number of students is 300 given the scholarship distribution -/
theorem total_students_proof :
  (1 - full_scholarship_percent - half_scholarship_percent) * total_students = no_scholarship_students :=
sorry

end NUMINAMATH_CALUDE_total_students_proof_l3164_316439


namespace NUMINAMATH_CALUDE_parabola_y_comparison_l3164_316418

/-- Parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 2

/-- Theorem stating that y₁ < y₂ for the given parabola -/
theorem parabola_y_comparison :
  ∀ (y₁ y₂ : ℝ), f 1 = y₁ → f 3 = y₂ → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_comparison_l3164_316418


namespace NUMINAMATH_CALUDE_quadratic_vertex_l3164_316464

/-- The quadratic function f(x) = (x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := (x - 3)^2 + 1

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex of the quadratic function f(x) = (x-3)^2 + 1 is at the point (3,1) -/
theorem quadratic_vertex : 
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l3164_316464


namespace NUMINAMATH_CALUDE_dogs_in_center_l3164_316460

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  jump : ℕ
  fetch : ℕ
  shake : ℕ
  jumpFetch : ℕ
  fetchShake : ℕ
  jumpShake : ℕ
  allThree : ℕ
  none : ℕ

/-- The total number of dogs in the center -/
def totalDogs (d : DogTricks) : ℕ :=
  d.allThree +
  (d.jumpFetch - d.allThree) +
  (d.fetchShake - d.allThree) +
  (d.jumpShake - d.allThree) +
  (d.jump - d.jumpFetch - d.jumpShake + d.allThree) +
  (d.fetch - d.jumpFetch - d.fetchShake + d.allThree) +
  (d.shake - d.jumpShake - d.fetchShake + d.allThree) +
  d.none

/-- Theorem stating that the total number of dogs in the center is 115 -/
theorem dogs_in_center (d : DogTricks)
  (h_jump : d.jump = 70)
  (h_fetch : d.fetch = 40)
  (h_shake : d.shake = 50)
  (h_jumpFetch : d.jumpFetch = 30)
  (h_fetchShake : d.fetchShake = 20)
  (h_jumpShake : d.jumpShake = 25)
  (h_allThree : d.allThree = 15)
  (h_none : d.none = 15) :
  totalDogs d = 115 := by
  sorry

end NUMINAMATH_CALUDE_dogs_in_center_l3164_316460


namespace NUMINAMATH_CALUDE_solution_value_l3164_316441

theorem solution_value (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 3/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3164_316441


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3164_316436

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 3) (hy : |y| = 5) (hxy : x * y < 0) :
  x + y = 2 ∨ x + y = -2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3164_316436


namespace NUMINAMATH_CALUDE_computer_on_time_l3164_316493

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents time of day in hours and minutes -/
structure Time where
  hour : ℕ
  minute : ℕ
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents a specific moment (day and time) -/
structure Moment where
  day : Day
  time : Time

def computer_on_duration : ℕ := 100

def computer_off_moment : Moment :=
  { day := Day.Friday
  , time := { hour := 17, minute := 0, h_valid := by norm_num, m_valid := by norm_num } }

theorem computer_on_time (on_moment off_moment : Moment) 
  (h : off_moment = computer_off_moment) 
  (duration : ℕ) (h_duration : duration = computer_on_duration) :
  on_moment = 
    { day := Day.Monday
    , time := { hour := 13, minute := 0, h_valid := by norm_num, m_valid := by norm_num } } :=
  sorry

end NUMINAMATH_CALUDE_computer_on_time_l3164_316493


namespace NUMINAMATH_CALUDE_coffee_maker_capacity_l3164_316411

/-- Represents a cylindrical coffee maker -/
structure CoffeeMaker :=
  (capacity : ℝ)

/-- The coffee maker contains 45 cups when it is 36% full -/
def partially_filled (cm : CoffeeMaker) : Prop :=
  0.36 * cm.capacity = 45

/-- Theorem: A cylindrical coffee maker that contains 45 cups when 36% full has a capacity of 125 cups -/
theorem coffee_maker_capacity (cm : CoffeeMaker) 
  (h : partially_filled cm) : cm.capacity = 125 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_capacity_l3164_316411


namespace NUMINAMATH_CALUDE_james_total_cost_l3164_316445

/-- Calculates the total amount James has to pay for adopting a puppy and a kitten -/
def total_cost (puppy_fee kitten_fee multiple_pet_discount friend1_contribution friend2_contribution sales_tax_rate pet_supplies : ℚ) : ℚ :=
  let total_adoption_fees := puppy_fee + kitten_fee
  let discounted_fees := total_adoption_fees * (1 - multiple_pet_discount)
  let friend_contributions := friend1_contribution * puppy_fee + friend2_contribution * kitten_fee
  let fees_after_contributions := discounted_fees - friend_contributions
  let sales_tax := fees_after_contributions * sales_tax_rate
  fees_after_contributions + sales_tax + pet_supplies

/-- The total cost James has to pay is $354.48 -/
theorem james_total_cost :
  total_cost 200 150 0.1 0.25 0.15 0.07 95 = 354.48 := by
  sorry

end NUMINAMATH_CALUDE_james_total_cost_l3164_316445


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3164_316427

theorem rationalize_denominator : 
  Real.sqrt (5 / (2 + Real.sqrt 2)) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3164_316427


namespace NUMINAMATH_CALUDE_square_sum_product_equality_l3164_316421

theorem square_sum_product_equality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 + (a*d - b*c)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_equality_l3164_316421


namespace NUMINAMATH_CALUDE_unique_number_property_l3164_316467

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l3164_316467


namespace NUMINAMATH_CALUDE_cow_ratio_proof_l3164_316406

theorem cow_ratio_proof (initial_cows : ℕ) (added_cows : ℕ) (remaining_cows : ℕ) : 
  initial_cows = 51 → 
  added_cows = 5 → 
  remaining_cows = 42 → 
  (initial_cows + added_cows - remaining_cows) / (initial_cows + added_cows) = 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_cow_ratio_proof_l3164_316406


namespace NUMINAMATH_CALUDE_parkingLotSpaces_l3164_316403

/-- Represents a car parking lot with three sections. -/
structure ParkingLot where
  section1 : ℕ
  section2 : ℕ
  section3 : ℕ

/-- Calculates the total number of spaces in the parking lot. -/
def totalSpaces (lot : ParkingLot) : ℕ :=
  lot.section1 + lot.section2 + lot.section3

/-- Theorem stating the total number of spaces in the parking lot. -/
theorem parkingLotSpaces : ∃ (lot : ParkingLot), 
  lot.section1 = 320 ∧ 
  lot.section2 = 440 ∧ 
  lot.section2 = lot.section3 + 200 ∧
  totalSpaces lot = 1000 := by
  sorry

end NUMINAMATH_CALUDE_parkingLotSpaces_l3164_316403


namespace NUMINAMATH_CALUDE_project_completion_time_l3164_316400

/-- Represents the project completion time given the conditions -/
theorem project_completion_time 
  (total_man_days : ℝ) 
  (initial_workers : ℕ) 
  (workers_left : ℕ) 
  (h1 : total_man_days = 200)
  (h2 : initial_workers = 10)
  (h3 : workers_left = 4)
  : ∃ (x : ℝ), x > 0 ∧ x + (total_man_days - initial_workers * x) / (initial_workers - workers_left) = 40 := by
  sorry

#check project_completion_time

end NUMINAMATH_CALUDE_project_completion_time_l3164_316400


namespace NUMINAMATH_CALUDE_kindergarten_gifts_l3164_316462

theorem kindergarten_gifts :
  ∀ (n : ℕ) (total_gifts : ℕ),
    (2 * 4 + (n - 2) * 3 + 11 = total_gifts) →
    (4 * 3 + (n - 4) * 6 + 10 = total_gifts) →
    total_gifts = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_kindergarten_gifts_l3164_316462


namespace NUMINAMATH_CALUDE_valid_two_digit_numbers_l3164_316426

def digits : Set Nat := {1, 2, 3}

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def has_no_repeated_digits (n : Nat) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens ≠ ones

def is_valid_number (n : Nat) : Prop :=
  is_two_digit n ∧
  has_no_repeated_digits n ∧
  (n / 10 ∈ digits) ∧
  (n % 10 ∈ digits)

theorem valid_two_digit_numbers :
  {n : Nat | is_valid_number n} = {12, 13, 21, 23, 31, 32} := by
  sorry

end NUMINAMATH_CALUDE_valid_two_digit_numbers_l3164_316426


namespace NUMINAMATH_CALUDE_map_scale_problem_l3164_316402

/-- Represents the map scale problem with given conditions -/
theorem map_scale_problem (map_distance_GN : ℝ) (map_distance_NM : ℝ)
  (time_GN : ℝ) (speed_GN : ℝ) (time_NM : ℝ) (speed_NM : ℝ)
  (h1 : map_distance_GN = 3)
  (h2 : map_distance_NM = 4)
  (h3 : time_GN = 2)
  (h4 : speed_GN = 50)
  (h5 : time_NM = 3)
  (h6 : speed_NM = 60) :
  (speed_GN * time_GN / map_distance_GN = speed_NM * time_NM / map_distance_NM) ∧
  (speed_GN * time_GN / map_distance_GN = 45) :=
by sorry

end NUMINAMATH_CALUDE_map_scale_problem_l3164_316402


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3164_316417

theorem stewart_farm_sheep_count :
  -- Definitions
  let sheep_horse_cow_ratio : Fin 3 → ℕ := ![4, 7, 5]
  let food_per_animal : Fin 3 → ℕ := ![150, 230, 300]
  let total_food : Fin 3 → ℕ := ![9750, 12880, 15000]

  -- Conditions
  ∀ (num_animals : Fin 3 → ℕ),
    (∀ i : Fin 3, num_animals i * food_per_animal i = total_food i) →
    (∀ i j : Fin 3, num_animals i * sheep_horse_cow_ratio j = num_animals j * sheep_horse_cow_ratio i) →

  -- Conclusion
  num_animals 0 = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3164_316417


namespace NUMINAMATH_CALUDE_samuel_food_drinks_spending_l3164_316490

def total_budget : ℕ := 20
def ticket_cost : ℕ := 14
def kevin_drinks : ℕ := 2
def kevin_food : ℕ := 4

theorem samuel_food_drinks_spending :
  ∀ (samuel_food_drinks : ℕ),
    samuel_food_drinks = total_budget - ticket_cost →
    kevin_drinks + kevin_food + ticket_cost = total_budget →
    samuel_food_drinks = 6 := by
  sorry

end NUMINAMATH_CALUDE_samuel_food_drinks_spending_l3164_316490


namespace NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l3164_316459

/-- The set of even digits -/
def evenDigits : Finset Nat := {0, 2, 4, 6, 8}

/-- A function to check if a natural number contains all even digits -/
def containsAllEvenDigits (n : Nat) : Prop :=
  ∀ d ∈ evenDigits, ∃ k : Nat, n / (10 ^ k) % 10 = d

/-- A function to check if a natural number is an eight-digit number -/
def isEightDigitNumber (n : Nat) : Prop :=
  10000000 ≤ n ∧ n ≤ 99999999

/-- The theorem stating that 99986420 is the largest eight-digit number containing all even digits -/
theorem largest_eight_digit_with_even_digits :
  (∀ n : Nat, isEightDigitNumber n → containsAllEvenDigits n → n ≤ 99986420) ∧
  isEightDigitNumber 99986420 ∧
  containsAllEvenDigits 99986420 :=
sorry

end NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l3164_316459


namespace NUMINAMATH_CALUDE_board_tiling_divisibility_l3164_316401

/-- Represents a square on the board -/
structure Square where
  row : Nat
  col : Nat

/-- Represents a domino placement -/
inductive Domino
  | horizontal : Square → Domino
  | vertical : Square → Domino

/-- Represents a tiling of the board -/
def Tiling := List Domino

/-- Represents an assignment of integers to squares -/
def Assignment := Square → Int

/-- Checks if a tiling is valid for a 2n × 2n board -/
def is_valid_tiling (n : Nat) (t : Tiling) : Prop := sorry

/-- Checks if an assignment satisfies the neighbor condition -/
def satisfies_neighbor_condition (n : Nat) (red_tiling blue_tiling : Tiling) (assignment : Assignment) : Prop := sorry

theorem board_tiling_divisibility (n : Nat) 
  (red_tiling blue_tiling : Tiling) 
  (h_red_valid : is_valid_tiling n red_tiling)
  (h_blue_valid : is_valid_tiling n blue_tiling)
  (assignment : Assignment)
  (h_nonzero : ∀ s, assignment s ≠ 0)
  (h_satisfies : satisfies_neighbor_condition n red_tiling blue_tiling assignment) :
  3 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_board_tiling_divisibility_l3164_316401


namespace NUMINAMATH_CALUDE_inequality_condition_l3164_316423

theorem inequality_condition (x : ℝ) : 
  (|x - 1| < 1 → x^2 - 5*x < 0) ∧ 
  ¬(∀ x : ℝ, x^2 - 5*x < 0 → |x - 1| < 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l3164_316423


namespace NUMINAMATH_CALUDE_abby_and_damon_weight_l3164_316486

theorem abby_and_damon_weight (a b c d : ℝ) 
  (h1 : a + b = 260)
  (h2 : b + c = 245)
  (h3 : c + d = 270)
  (h4 : a + c = 220) :
  a + d = 285 := by
  sorry

end NUMINAMATH_CALUDE_abby_and_damon_weight_l3164_316486


namespace NUMINAMATH_CALUDE_correct_celsius_to_fahrenheit_conversion_l3164_316471

/-- Conversion function from Celsius to Fahrenheit -/
def celsiusToFahrenheit (c : ℝ) : ℝ := 1.8 * c + 32

/-- Theorem stating the correct conversion from Celsius to Fahrenheit -/
theorem correct_celsius_to_fahrenheit_conversion (c : ℝ) : 
  celsiusToFahrenheit c = 1.8 * c + 32 := by
  sorry

end NUMINAMATH_CALUDE_correct_celsius_to_fahrenheit_conversion_l3164_316471


namespace NUMINAMATH_CALUDE_sum_of_roots_bound_l3164_316438

/-- Given a quadratic equation x^2 - 2(1-k)x + k^2 = 0 with real roots α and β, 
    the sum of these roots α + β is greater than or equal to 1. -/
theorem sum_of_roots_bound (k : ℝ) (α β : ℝ) : 
  (∀ x, x^2 - 2*(1-k)*x + k^2 = 0 ↔ x = α ∨ x = β) →
  α + β ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_bound_l3164_316438


namespace NUMINAMATH_CALUDE_volleyball_team_lineup_count_l3164_316496

def volleyball_team_size : ℕ := 14
def starting_lineup_size : ℕ := 6
def triplet_size : ℕ := 3

-- Define a function to calculate combinations
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem volleyball_team_lineup_count :
  choose volleyball_team_size starting_lineup_size -
  choose (volleyball_team_size - triplet_size) (starting_lineup_size - triplet_size) = 2838 :=
sorry

end NUMINAMATH_CALUDE_volleyball_team_lineup_count_l3164_316496


namespace NUMINAMATH_CALUDE_cupcakes_problem_l3164_316428

/-- Given the initial number of cupcakes, the number of cupcakes eaten, and the number of packages,
    calculate the number of cupcakes in each package. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Theorem: Given 38 initial cupcakes, 14 cupcakes eaten, and 3 packages made,
    the number of cupcakes in each package is 8. -/
theorem cupcakes_problem : cupcakes_per_package 38 14 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_problem_l3164_316428


namespace NUMINAMATH_CALUDE_no_arithmetic_progression_40_terms_l3164_316463

theorem no_arithmetic_progression_40_terms : ¬ ∃ (a d : ℕ) (f : ℕ → ℕ × ℕ),
  (∀ i : ℕ, i < 40 → ∃ (m n : ℕ), f i = (m, n) ∧ a + i * d = 2^m + 3^n) :=
sorry

end NUMINAMATH_CALUDE_no_arithmetic_progression_40_terms_l3164_316463


namespace NUMINAMATH_CALUDE_problem_solution_l3164_316483

theorem problem_solution (x y : ℝ) (h1 : 3*x + y = 5) (h2 : x + 3*y = 8) :
  5*x^2 + 11*x*y + 5*y^2 = 89 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3164_316483


namespace NUMINAMATH_CALUDE_simplify_roots_l3164_316454

theorem simplify_roots : (625 : ℝ)^(1/4) * (125 : ℝ)^(1/3) = 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_roots_l3164_316454


namespace NUMINAMATH_CALUDE_product_of_tripled_numbers_l3164_316489

theorem product_of_tripled_numbers (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ + 1/x₁ = 3*x₁ ∧ x₂ + 1/x₂ = 3*x₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_tripled_numbers_l3164_316489


namespace NUMINAMATH_CALUDE_inverse_composition_problem_l3164_316422

/-- Given functions h and k where k⁻¹ ∘ h = λ z, 7 * z - 4, prove that h⁻¹(k(12)) = 16/7 -/
theorem inverse_composition_problem (h k : ℝ → ℝ) 
  (hk : Function.LeftInverse k⁻¹ h ∧ Function.RightInverse k⁻¹ h) 
  (h_def : ∀ z, k⁻¹ (h z) = 7 * z - 4) : 
  h⁻¹ (k 12) = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_problem_l3164_316422


namespace NUMINAMATH_CALUDE_min_pizzas_cover_expenses_l3164_316482

/-- Represents the minimum number of pizzas John must deliver to cover his expenses -/
def min_pizzas : ℕ := 1063

/-- Represents the cost of the used car -/
def car_cost : ℕ := 8000

/-- Represents the upfront maintenance cost -/
def maintenance_cost : ℕ := 500

/-- Represents the earnings per pizza delivered -/
def earnings_per_pizza : ℕ := 12

/-- Represents the gas cost per delivery -/
def gas_cost_per_delivery : ℕ := 4

/-- Represents the net earnings per pizza (earnings minus gas cost) -/
def net_earnings_per_pizza : ℕ := earnings_per_pizza - gas_cost_per_delivery

theorem min_pizzas_cover_expenses :
  (min_pizzas : ℝ) * net_earnings_per_pizza ≥ car_cost + maintenance_cost :=
sorry

end NUMINAMATH_CALUDE_min_pizzas_cover_expenses_l3164_316482


namespace NUMINAMATH_CALUDE_square_between_endpoints_l3164_316443

theorem square_between_endpoints (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_cd : c * d = 1) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_between_endpoints_l3164_316443


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3164_316449

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ),
    (x₁ * (5 * x₁ - 11) = 2) ∧
    (x₂ * (5 * x₂ - 11) = 2) ∧
    (x₁ = (11 + Real.sqrt 161) / 10) ∧
    (x₂ = (11 - Real.sqrt 161) / 10) ∧
    (Nat.gcd 11 (Nat.gcd 161 10) = 1) ∧
    (11 + 161 + 10 = 182) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3164_316449


namespace NUMINAMATH_CALUDE_cubic_function_not_monotonic_l3164_316484

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

def not_monotonic (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, x < y ∧ y < z ∧ ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

theorem cubic_function_not_monotonic (a : ℝ) :
  not_monotonic (f a) → a ∈ Set.Iio (-Real.sqrt 3) ∪ Set.Ioi (Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_not_monotonic_l3164_316484


namespace NUMINAMATH_CALUDE_prob_same_color_correct_l3164_316420

def total_balls : ℕ := 16
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

def prob_same_color : ℚ := 98 / 256

theorem prob_same_color_correct :
  (green_balls / total_balls)^2 + (red_balls / total_balls)^2 + (blue_balls / total_balls)^2 = prob_same_color :=
by sorry

end NUMINAMATH_CALUDE_prob_same_color_correct_l3164_316420


namespace NUMINAMATH_CALUDE_sphere_cap_cone_volume_equality_l3164_316437

theorem sphere_cap_cone_volume_equality (R : ℝ) (R_pos : R > 0) :
  ∃ x : ℝ, x > 0 ∧ x < R ∧
  (2 / 3 * R^2 * π * (R - x) = 1 / 3 * (R^2 - x^2) * π * x) ∧
  x = R * (Real.sqrt 5 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_cap_cone_volume_equality_l3164_316437


namespace NUMINAMATH_CALUDE_total_children_count_l3164_316475

/-- The number of toy cars given to boys -/
def toy_cars : ℕ := 134

/-- The number of dolls given to girls -/
def dolls : ℕ := 269

/-- The number of board games given to both boys and girls -/
def board_games : ℕ := 87

/-- Every child received only one toy -/
axiom one_toy_per_child : True

/-- The total number of children attending the event -/
def total_children : ℕ := toy_cars + dolls

theorem total_children_count : total_children = 403 := by sorry

end NUMINAMATH_CALUDE_total_children_count_l3164_316475


namespace NUMINAMATH_CALUDE_percent_relation_l3164_316416

theorem percent_relation (x y : ℝ) (h : 0.2 * (x - y) = 0.15 * (x + y)) : y = x / 7 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3164_316416


namespace NUMINAMATH_CALUDE_car_distance_traveled_l3164_316442

/-- Given a train speed and a car's relative speed to the train, 
    calculate the distance traveled by the car in a given time. -/
theorem car_distance_traveled 
  (train_speed : ℝ) 
  (car_relative_speed : ℝ) 
  (time_minutes : ℝ) : 
  train_speed = 90 →
  car_relative_speed = 2/3 →
  time_minutes = 30 →
  (car_relative_speed * train_speed) * (time_minutes / 60) = 30 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l3164_316442


namespace NUMINAMATH_CALUDE_f_properties_l3164_316444

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x - x

theorem f_properties :
  let f := f
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂) ∧
  (∀ x, x > 0 → f x ≥ -1) ∧
  f 1 = -1 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3164_316444


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_3_and_9_l3164_316479

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_digit_divisible_by_3_and_9 : 
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by (528000 + d * 100 + 74) 3 ∧ 
    is_divisible_by (528000 + d * 100 + 74) 9 ∧
    ∀ (d' : ℕ), d' < d → 
      ¬(is_divisible_by (528000 + d' * 100 + 74) 3 ∧ 
        is_divisible_by (528000 + d' * 100 + 74) 9) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_3_and_9_l3164_316479


namespace NUMINAMATH_CALUDE_jerry_lawsuit_percentage_l3164_316419

/-- Calculates the percentage of a lawsuit claim received -/
def lawsuit_claim_percentage (annual_salary : ℕ) (years : ℕ) (medical_bills : ℕ) (received_amount : ℕ) : ℚ :=
  let salary_damages := annual_salary * years
  let punitive_multiplier := 3
  let punitive_damages := punitive_multiplier * (salary_damages + medical_bills)
  let total_claim := salary_damages + medical_bills + punitive_damages
  (received_amount : ℚ) / (total_claim : ℚ) * 100

theorem jerry_lawsuit_percentage :
  let result := lawsuit_claim_percentage 50000 30 200000 5440000
  (result > 79.9 ∧ result < 80.1) :=
by sorry

end NUMINAMATH_CALUDE_jerry_lawsuit_percentage_l3164_316419


namespace NUMINAMATH_CALUDE_fishing_trip_theorem_l3164_316410

def is_small_fish (weight : ℕ) : Bool := 1 ≤ weight ∧ weight ≤ 5
def is_medium_fish (weight : ℕ) : Bool := 6 ≤ weight ∧ weight ≤ 12
def is_large_fish (weight : ℕ) : Bool := weight > 12

def brendan_morning_catch : List ℕ := [1, 3, 4, 7, 7, 13, 15, 17]
def brendan_afternoon_catch : List ℕ := [2, 8, 8, 18, 20]
def emily_catch : List ℕ := [5, 6, 9, 11, 14, 20]
def dad_catch : List ℕ := [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21]

def brendan_morning_keep (weight : ℕ) : Bool := is_medium_fish weight ∨ is_large_fish weight
def brendan_afternoon_keep (weight : ℕ) : Bool := is_medium_fish weight ∨ (is_large_fish weight ∧ weight > 15)
def emily_keep (weight : ℕ) : Bool := is_large_fish weight ∨ weight = 5
def dad_keep (weight : ℕ) : Bool := (is_medium_fish weight ∧ weight ≥ 8 ∧ weight ≤ 11) ∨ (is_large_fish weight ∧ weight > 15 ∧ weight ≠ 21)

theorem fishing_trip_theorem :
  (brendan_morning_catch.filter brendan_morning_keep).length +
  (brendan_afternoon_catch.filter brendan_afternoon_keep).length +
  (emily_catch.filter emily_keep).length +
  (dad_catch.filter dad_keep).length = 18 := by
  sorry

end NUMINAMATH_CALUDE_fishing_trip_theorem_l3164_316410


namespace NUMINAMATH_CALUDE_grade_assignment_count_l3164_316480

theorem grade_assignment_count : 
  (number_of_grades : ℕ) → 
  (number_of_students : ℕ) → 
  number_of_grades = 4 → 
  number_of_students = 12 → 
  number_of_grades ^ number_of_students = 16777216 :=
by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l3164_316480


namespace NUMINAMATH_CALUDE_pressure_functions_exist_l3164_316498

/-- Represents the gas pressure in a vessel as a function of time. -/
def PressureFunction := ℝ → ℝ

/-- Represents the parameters of the gas system. -/
structure GasSystem where
  V₁ : ℝ  -- Volume of vessel 1
  V₂ : ℝ  -- Volume of vessel 2
  P₁ : ℝ  -- Initial pressure in vessel 1
  P₂ : ℝ  -- Initial pressure in vessel 2
  a  : ℝ  -- Flow rate coefficient
  b  : ℝ  -- Pressure change coefficient

/-- Defines the conditions for valid pressure functions in the gas system. -/
def ValidPressureFunctions (sys : GasSystem) (p₁ p₂ : PressureFunction) : Prop :=
  -- Initial conditions
  p₁ 0 = sys.P₁ ∧ p₂ 0 = sys.P₂ ∧
  -- Conservation of mass
  ∀ t, sys.V₁ * p₁ t + sys.V₂ * p₂ t = sys.V₁ * sys.P₁ + sys.V₂ * sys.P₂ ∧
  -- Differential equations
  ∀ t, sys.a * (p₁ t ^ 2 - p₂ t ^ 2) = -sys.b * sys.V₁ * (deriv p₁ t) ∧
  ∀ t, sys.a * (p₁ t ^ 2 - p₂ t ^ 2) = sys.b * sys.V₂ * (deriv p₂ t)

/-- Theorem stating the existence of valid pressure functions for a given gas system. -/
theorem pressure_functions_exist (sys : GasSystem) :
  ∃ (p₁ p₂ : PressureFunction), ValidPressureFunctions sys p₁ p₂ := by
  sorry


end NUMINAMATH_CALUDE_pressure_functions_exist_l3164_316498


namespace NUMINAMATH_CALUDE_find_second_number_l3164_316448

theorem find_second_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 19 + x) / 3) + 7 →
  x = 70 := by
sorry

end NUMINAMATH_CALUDE_find_second_number_l3164_316448


namespace NUMINAMATH_CALUDE_perpendicular_to_countless_lines_iff_perpendicular_to_plane_l3164_316468

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Defines when a line is perpendicular to a plane -/
def Line.perpendicular_to_plane (l : Line) (a : Plane) : Prop :=
  sorry

/-- Defines when a line is perpendicular to countless lines within a plane -/
def Line.perpendicular_to_countless_lines_in_plane (l : Line) (a : Plane) : Prop :=
  sorry

/-- 
  The statement that a line being perpendicular to countless lines within a plane
  is a necessary and sufficient condition for the line being perpendicular to the plane
-/
theorem perpendicular_to_countless_lines_iff_perpendicular_to_plane
  (l : Line) (a : Plane) :
  Line.perpendicular_to_countless_lines_in_plane l a ↔ Line.perpendicular_to_plane l a :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_countless_lines_iff_perpendicular_to_plane_l3164_316468


namespace NUMINAMATH_CALUDE_mixed_fraction_division_subtraction_l3164_316457

theorem mixed_fraction_division_subtraction :
  (1 + 5/6) / (2 + 3/4) - 1/2 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fraction_division_subtraction_l3164_316457


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3164_316415

-- Problem 1
theorem problem_1 : 2013^2 - 2012 * 2014 = 1 := by sorry

-- Problem 2
theorem problem_2 (m n : ℤ) : ((m - n)^6 / (n - m)^4) * (m - n)^3 = (m - n)^5 := by sorry

-- Problem 3
theorem problem_3 (a b c : ℝ) : (a - 2*b + 3*c) * (a - 2*b - 3*c) = a^2 - 4*a*b + 4*b^2 - 9*c^2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3164_316415


namespace NUMINAMATH_CALUDE_ribbon_calculation_l3164_316492

/-- Represents the types of ribbons available --/
inductive RibbonType
  | A
  | B

/-- Represents the wrapping pattern for a gift --/
structure WrappingPattern where
  typeA : Nat
  typeB : Nat

/-- Calculates the number of ribbons needed for a given number of gifts and wrapping pattern --/
def ribbonsNeeded (numGifts : Nat) (pattern : WrappingPattern) : Nat × Nat :=
  (numGifts * pattern.typeA, numGifts * pattern.typeB)

theorem ribbon_calculation (tomSupplyA tomSupplyB : Nat) :
  let oddPattern : WrappingPattern := { typeA := 1, typeB := 2 }
  let evenPattern : WrappingPattern := { typeA := 2, typeB := 1 }
  let (oddA, oddB) := ribbonsNeeded 4 oddPattern
  let (evenA, evenB) := ribbonsNeeded 4 evenPattern
  let totalA := oddA + evenA
  let totalB := oddB + evenB
  tomSupplyA = 10 ∧ tomSupplyB = 12 →
  totalA - tomSupplyA = 2 ∧ totalB - tomSupplyB = 0 := by
  sorry


end NUMINAMATH_CALUDE_ribbon_calculation_l3164_316492


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3164_316450

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ x - 4 = 21 * (1/x) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3164_316450


namespace NUMINAMATH_CALUDE_range_of_f_l3164_316491

def f (x : ℝ) : ℝ := x^2 + 1

theorem range_of_f :
  Set.range f = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3164_316491


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3164_316407

theorem complex_modulus_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z + w) = 5) :
  Complex.abs z = 4 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3164_316407


namespace NUMINAMATH_CALUDE_m_nonpositive_l3164_316430

theorem m_nonpositive (m : ℝ) (h : Real.sqrt (m^2) = -m) : m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_m_nonpositive_l3164_316430


namespace NUMINAMATH_CALUDE_square_side_length_l3164_316470

/-- A square with perimeter 32 cm has sides of length 8 cm -/
theorem square_side_length (s : ℝ) (h₁ : s > 0) (h₂ : 4 * s = 32) : s = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3164_316470
