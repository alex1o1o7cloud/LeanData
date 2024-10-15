import Mathlib

namespace NUMINAMATH_CALUDE_jimin_candies_count_l1687_168772

/-- The number of candies Jimin gave to Yuna -/
def candies_to_yuna : ℕ := 25

/-- The number of candies Jimin gave to her sister -/
def candies_to_sister : ℕ := 13

/-- The total number of candies Jimin had at first -/
def total_candies : ℕ := candies_to_yuna + candies_to_sister

theorem jimin_candies_count : total_candies = 38 := by
  sorry

end NUMINAMATH_CALUDE_jimin_candies_count_l1687_168772


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l1687_168767

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and subset relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (α β : Plane) (l : Line)
  (h1 : α ≠ β)
  (h2 : subset l α)
  (h3 : perpendicular_line_plane l β) :
  perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l1687_168767


namespace NUMINAMATH_CALUDE_decimal_0_04_is_4_percent_l1687_168780

/-- Converts a decimal fraction to a percentage -/
def decimal_to_percentage (d : ℝ) : ℝ := d * 100

/-- The decimal fraction we're working with -/
def given_decimal : ℝ := 0.04

/-- Theorem: The percentage representation of 0.04 is 4% -/
theorem decimal_0_04_is_4_percent : decimal_to_percentage given_decimal = 4 := by
  sorry

end NUMINAMATH_CALUDE_decimal_0_04_is_4_percent_l1687_168780


namespace NUMINAMATH_CALUDE_conjugate_complex_magnitude_l1687_168757

theorem conjugate_complex_magnitude (α β : ℂ) : 
  (∃ (x y : ℝ), α = x + y * Complex.I ∧ β = x - y * Complex.I) →  -- conjugate complex numbers
  (∃ (r : ℝ), α / β^3 = r) →  -- α/β³ is real
  Complex.abs (α - β) = 4 →  -- |α - β| = 4
  Complex.abs α = 4 * Real.sqrt 3 / 3 :=  -- |α| = 4√3/3
by sorry

end NUMINAMATH_CALUDE_conjugate_complex_magnitude_l1687_168757


namespace NUMINAMATH_CALUDE_inner_rectangle_length_l1687_168735

/-- Represents the dimensions of a rectangular region -/
structure Region where
  length : ℝ
  width : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Represents the floor layout with three regions -/
structure FloorLayout where
  inner : Region
  middle : Region
  outer : Region

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem inner_rectangle_length (layout : FloorLayout) : 
  layout.inner.width = 2 →
  layout.middle.length = layout.inner.length + 2 →
  layout.middle.width = layout.inner.width + 2 →
  layout.outer.length = layout.middle.length + 2 →
  layout.outer.width = layout.middle.width + 2 →
  isArithmeticProgression (area layout.inner) (area layout.middle) (area layout.outer) →
  layout.inner.length = 8 := by
  sorry

#check inner_rectangle_length

end NUMINAMATH_CALUDE_inner_rectangle_length_l1687_168735


namespace NUMINAMATH_CALUDE_houses_with_both_features_l1687_168770

theorem houses_with_both_features (total : ℕ) (garage : ℕ) (pool : ℕ) (neither : ℕ)
  (h_total : total = 85)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_neither : neither = 30) :
  ∃ (both : ℕ), both = garage + pool - (total - neither) :=
by
  sorry

end NUMINAMATH_CALUDE_houses_with_both_features_l1687_168770


namespace NUMINAMATH_CALUDE_divide_number_80_l1687_168784

theorem divide_number_80 (smaller larger : ℝ) : 
  smaller + larger = 80 ∧ 
  larger / 2 = smaller + 10 → 
  smaller = 20 ∧ larger = 60 := by
sorry

end NUMINAMATH_CALUDE_divide_number_80_l1687_168784


namespace NUMINAMATH_CALUDE_complex_number_powers_l1687_168799

theorem complex_number_powers (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^97 + z^98 + z^99 + z^100 + z^101 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_powers_l1687_168799


namespace NUMINAMATH_CALUDE_return_probability_limit_l1687_168740

/-- Represents a player in the money exchange game --/
inductive Player : Type
| Alan : Player
| Beth : Player
| Charlie : Player
| Dana : Player

/-- The state of the game is represented by a function from Player to ℕ (amount of money) --/
def GameState : Type := Player → ℕ

/-- The initial state of the game where each player has $1 --/
def initialState : GameState :=
  fun p => 1

/-- A single round of the game where players randomly exchange money --/
def playRound (state : GameState) : GameState :=
  sorry

/-- The probability of returning to the initial state after many rounds --/
def returnProbability (numRounds : ℕ) : ℚ :=
  sorry

/-- The main theorem stating that the probability approaches 1/9 as the number of rounds increases --/
theorem return_probability_limit :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |returnProbability n - 1/9| < ε :=
sorry

end NUMINAMATH_CALUDE_return_probability_limit_l1687_168740


namespace NUMINAMATH_CALUDE_abs_h_value_l1687_168771

theorem abs_h_value (h : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁^4 + 4*h*x₁^2 = 2) ∧ 
    (x₂^4 + 4*h*x₂^2 = 2) ∧ 
    (x₃^4 + 4*h*x₃^2 = 2) ∧ 
    (x₄^4 + 4*h*x₄^2 = 2) ∧ 
    (x₁^2 + x₂^2 + x₃^2 + x₄^2 = 34)) → 
  |h| = 17/4 := by
sorry

end NUMINAMATH_CALUDE_abs_h_value_l1687_168771


namespace NUMINAMATH_CALUDE_walking_competition_analysis_l1687_168753

/-- The Chi-square statistic for a 2x2 contingency table -/
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 90% confidence in a Chi-square test with 1 degree of freedom -/
def critical_value : ℚ := 2706 / 1000

/-- The probability of selecting a Female Walking Star -/
def p_female_walking_star : ℚ := 14 / 70

/-- The number of trials in the binomial distribution -/
def num_trials : ℕ := 3

/-- The expected value of X (number of Female Walking Stars in a sample of 3) -/
def expected_value : ℚ := num_trials * p_female_walking_star

theorem walking_competition_analysis :
  let k_squared := chi_square 24 16 16 14
  k_squared < critical_value ∧ expected_value = 3/5 := by sorry

end NUMINAMATH_CALUDE_walking_competition_analysis_l1687_168753


namespace NUMINAMATH_CALUDE_bundle_promotion_better_l1687_168711

-- Define the prices and discounts
def cellphone_price : ℝ := 800
def earbud_price : ℝ := 150
def case_price : ℝ := 40
def cellphone_discount : ℝ := 0.05
def earbud_discount : ℝ := 0.10
def bundle_discount : ℝ := 0.07
def loyalty_discount : ℝ := 0.03
def sales_tax : ℝ := 0.08

-- Define the total cost before promotions
def total_before_promotions : ℝ :=
  (2 * cellphone_price * (1 - cellphone_discount)) +
  (2 * earbud_price * (1 - earbud_discount)) +
  case_price

-- Define the cost after each promotion
def bundle_promotion_cost : ℝ :=
  total_before_promotions * (1 - bundle_discount)

def loyalty_promotion_cost : ℝ :=
  total_before_promotions * (1 - loyalty_discount)

-- Define the final costs including tax
def bundle_final_cost : ℝ :=
  bundle_promotion_cost * (1 + sales_tax)

def loyalty_final_cost : ℝ :=
  loyalty_promotion_cost * (1 + sales_tax)

-- Theorem statement
theorem bundle_promotion_better :
  bundle_final_cost < loyalty_final_cost :=
sorry

end NUMINAMATH_CALUDE_bundle_promotion_better_l1687_168711


namespace NUMINAMATH_CALUDE_special_sequence_tenth_term_l1687_168795

/-- A sequence satisfying the given condition -/
def SpecialSequence (a : ℕ+ → ℤ) : Prop :=
  ∀ m n : ℕ+, a m + a n = a (m + n) - 2 * (m.val * n.val)

/-- The theorem to be proved -/
theorem special_sequence_tenth_term (a : ℕ+ → ℤ) 
  (h : SpecialSequence a) (h1 : a 1 = 1) : a 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_tenth_term_l1687_168795


namespace NUMINAMATH_CALUDE_orange_juice_percentage_approx_48_l1687_168787

/-- Represents the juice yield from a specific fruit -/
structure JuiceYield where
  fruit : String
  count : Nat
  ounces : Rat

/-- Calculates the juice blend composition and returns the percentage of orange juice -/
def orangeJuicePercentage (appleYield pearYield orangeYield : JuiceYield) : Rat :=
  let appleJuicePerFruit := appleYield.ounces / appleYield.count
  let pearJuicePerFruit := pearYield.ounces / pearYield.count
  let orangeJuicePerFruit := orangeYield.ounces / orangeYield.count
  let totalJuice := appleJuicePerFruit + pearJuicePerFruit + orangeJuicePerFruit
  (orangeJuicePerFruit / totalJuice) * 100

/-- Theorem stating that the percentage of orange juice in the blend is approximately 48% -/
theorem orange_juice_percentage_approx_48 (appleYield pearYield orangeYield : JuiceYield) 
  (h1 : appleYield.fruit = "apple" ∧ appleYield.count = 5 ∧ appleYield.ounces = 9)
  (h2 : pearYield.fruit = "pear" ∧ pearYield.count = 4 ∧ pearYield.ounces = 10)
  (h3 : orangeYield.fruit = "orange" ∧ orangeYield.count = 3 ∧ orangeYield.ounces = 12) :
  ∃ (ε : Rat), abs (orangeJuicePercentage appleYield pearYield orangeYield - 48) < ε ∧ ε < 1 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_percentage_approx_48_l1687_168787


namespace NUMINAMATH_CALUDE_circle_tangent_theorem_l1687_168797

/-- Given two externally tangent circles and a tangent line satisfying certain conditions,
    prove the relationship between r, R, and p, and the length of BC. -/
theorem circle_tangent_theorem (r R p : ℝ) (h_pos_r : 0 < r) (h_pos_R : 0 < R) (h_pos_p : 0 < p) :
  -- Condition for the geometric configuration
  (p^2 / (4 * (p + 1)) < r / R ∧ r / R < p^2 / (2 * (p + 1))) →
  -- Length of BC
  ∃ (BC : ℝ), BC = p / (p + 1) * Real.sqrt (4 * (p + 1) * R * r - p^2 * R^2) := by
  sorry


end NUMINAMATH_CALUDE_circle_tangent_theorem_l1687_168797


namespace NUMINAMATH_CALUDE_angle_value_in_connected_triangles_l1687_168796

theorem angle_value_in_connected_triangles : ∀ x : ℝ,
  (∃ α β : ℝ,
    -- Left triangle
    3 * x + 4 * x + α = 180 ∧
    -- Middle triangle
    α + 5 * x + β = 180 ∧
    -- Right triangle
    β + 2 * x + 6 * x = 180) →
  x = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_in_connected_triangles_l1687_168796


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l1687_168773

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) 
  (hP : P ≠ 0) : 
  M / N = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l1687_168773


namespace NUMINAMATH_CALUDE_faye_finished_problems_l1687_168794

theorem faye_finished_problems (math_problems science_problems left_for_homework : ℕ)
  (h1 : math_problems = 46)
  (h2 : science_problems = 9)
  (h3 : left_for_homework = 15) :
  math_problems + science_problems - left_for_homework = 40 := by
  sorry

end NUMINAMATH_CALUDE_faye_finished_problems_l1687_168794


namespace NUMINAMATH_CALUDE_circle_radius_zero_l1687_168704

theorem circle_radius_zero (x y : ℝ) :
  25 * x^2 - 50 * x + 25 * y^2 + 100 * y + 125 = 0 →
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l1687_168704


namespace NUMINAMATH_CALUDE_nonagon_perimeter_l1687_168725

theorem nonagon_perimeter : 
  let side_lengths : List ℕ := [2, 2, 3, 3, 1, 3, 2, 2, 2]
  List.sum side_lengths = 20 := by sorry

end NUMINAMATH_CALUDE_nonagon_perimeter_l1687_168725


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1687_168768

/-- Given a triangle with sides 9, 12, and 15, its shortest altitude has length 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 ∧ b = 12 ∧ c = 15 ∧ 
  a^2 + b^2 = c^2 ∧
  h * c = 2 * (1/2 * a * b) →
  h = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1687_168768


namespace NUMINAMATH_CALUDE_school_poll_intersection_l1687_168766

theorem school_poll_intersection (T C D : Finset ℕ) (h1 : T.card = 230) 
  (h2 : C.card = 171) (h3 : D.card = 137) 
  (h4 : (T \ C).card + (T \ D).card - T.card = 37) : 
  (C ∩ D).card = 115 := by
  sorry

end NUMINAMATH_CALUDE_school_poll_intersection_l1687_168766


namespace NUMINAMATH_CALUDE_three_good_sets_l1687_168756

-- Define the "good set" property
def is_good_set (C : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ C, ∃ p₂ ∈ C, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the sets
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}
def C₂ : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2 = 9}
def C₃ : Set (ℝ × ℝ) := {p | 2*p.1^2 + p.2^2 = 9}
def C₄ : Set (ℝ × ℝ) := {p | p.1^2 + p.2 = 9}

-- Theorem statement
theorem three_good_sets : 
  (is_good_set C₁ ∧ is_good_set C₃ ∧ is_good_set C₄ ∧ ¬is_good_set C₂) := by
  sorry

end NUMINAMATH_CALUDE_three_good_sets_l1687_168756


namespace NUMINAMATH_CALUDE_tree_calculation_l1687_168775

theorem tree_calculation (T P R : ℝ) (h1 : T = 400) (h2 : P = 0.20) (h3 : R = 5) :
  T - (P * T) + (P * T * R) = 720 :=
by sorry

end NUMINAMATH_CALUDE_tree_calculation_l1687_168775


namespace NUMINAMATH_CALUDE_two_different_color_chips_probability_l1687_168748

/-- The probability of drawing two chips of different colors from a bag containing
    6 green chips, 5 purple chips, and 4 orange chips, when drawing with replacement. -/
theorem two_different_color_chips_probability :
  let total_chips : ℕ := 6 + 5 + 4
  let green_chips : ℕ := 6
  let purple_chips : ℕ := 5
  let orange_chips : ℕ := 4
  let prob_green : ℚ := green_chips / total_chips
  let prob_purple : ℚ := purple_chips / total_chips
  let prob_orange : ℚ := orange_chips / total_chips
  let prob_not_green : ℚ := (purple_chips + orange_chips) / total_chips
  let prob_not_purple : ℚ := (green_chips + orange_chips) / total_chips
  let prob_not_orange : ℚ := (green_chips + purple_chips) / total_chips
  (prob_green * prob_not_green + prob_purple * prob_not_purple + prob_orange * prob_not_orange) = 148 / 225 := by
  sorry

end NUMINAMATH_CALUDE_two_different_color_chips_probability_l1687_168748


namespace NUMINAMATH_CALUDE_equal_grid_values_l1687_168762

/-- Represents a point in the infinite square grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents an admissible polygon on the grid --/
structure AdmissiblePolygon where
  vertices : List GridPoint
  area : ℕ
  area_gt_two : area > 2

/-- The grid of natural numbers --/
def Grid := GridPoint → ℕ

/-- The value of an admissible polygon --/
def value (grid : Grid) (polygon : AdmissiblePolygon) : ℕ := sorry

/-- Two polygons are congruent --/
def congruent (p1 p2 : AdmissiblePolygon) : Prop := sorry

/-- Main theorem --/
theorem equal_grid_values (grid : Grid) :
  (∀ p1 p2 : AdmissiblePolygon, congruent p1 p2 → value grid p1 = value grid p2) →
  (∀ p1 p2 : GridPoint, grid p1 = grid p2) := by sorry

end NUMINAMATH_CALUDE_equal_grid_values_l1687_168762


namespace NUMINAMATH_CALUDE_passengers_taken_at_second_station_is_12_l1687_168736

/-- Represents the number of passengers on a train at different stages --/
structure TrainPassengers where
  initial : Nat
  after_first_drop : Nat
  after_first_pickup : Nat
  after_second_drop : Nat
  final : Nat

/-- Calculates the number of passengers taken at the second station --/
def passengers_taken_at_second_station (train : TrainPassengers) : Nat :=
  train.final - train.after_second_drop

/-- Theorem stating the number of passengers taken at the second station --/
theorem passengers_taken_at_second_station_is_12 :
  ∃ (train : TrainPassengers),
    train.initial = 270 ∧
    train.after_first_drop = train.initial - train.initial / 3 ∧
    train.after_first_pickup = train.after_first_drop + 280 ∧
    train.after_second_drop = train.after_first_pickup / 2 ∧
    train.final = 242 ∧
    passengers_taken_at_second_station train = 12 := by
  sorry

#check passengers_taken_at_second_station_is_12

end NUMINAMATH_CALUDE_passengers_taken_at_second_station_is_12_l1687_168736


namespace NUMINAMATH_CALUDE_lisa_hourly_wage_l1687_168716

/-- Calculates the hourly wage of Lisa given Greta's work hours, hourly rate, and Lisa's equivalent work hours -/
theorem lisa_hourly_wage (greta_hours : ℕ) (greta_rate : ℚ) (lisa_hours : ℕ) : 
  greta_hours = 40 → 
  greta_rate = 12 → 
  lisa_hours = 32 → 
  (greta_hours * greta_rate) / lisa_hours = 15 := by
sorry

end NUMINAMATH_CALUDE_lisa_hourly_wage_l1687_168716


namespace NUMINAMATH_CALUDE_determinant_value_l1687_168776

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_value (m : ℝ) (h : m^2 - 2*m - 3 = 0) : 
  determinant (m^2) (m-3) (1-2*m) (m-2) = 9 := by sorry

end NUMINAMATH_CALUDE_determinant_value_l1687_168776


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1687_168774

/-- Proves that given a price reduction x%, if the sale increases by 80% and the net effect on the sale is 53%, then x = 15. -/
theorem price_reduction_percentage (x : ℝ) : 
  (1 - x / 100) * 1.80 = 1.53 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1687_168774


namespace NUMINAMATH_CALUDE_inequality_proof_l1687_168723

theorem inequality_proof (x y : ℝ) : x^4 + y^4 + 8 ≥ 8*x*y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1687_168723


namespace NUMINAMATH_CALUDE_total_ants_is_twenty_l1687_168721

/-- The number of ants found by Abe -/
def abe_ants : ℕ := 4

/-- The number of ants found by Beth -/
def beth_ants : ℕ := abe_ants + abe_ants / 2

/-- The number of ants found by CeCe -/
def cece_ants : ℕ := 2 * abe_ants

/-- The number of ants found by Duke -/
def duke_ants : ℕ := abe_ants / 2

/-- The total number of ants found by all four children -/
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

theorem total_ants_is_twenty : total_ants = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_ants_is_twenty_l1687_168721


namespace NUMINAMATH_CALUDE_sixteen_pow_six_mod_nine_l1687_168793

theorem sixteen_pow_six_mod_nine : 16^6 ≡ 1 [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_sixteen_pow_six_mod_nine_l1687_168793


namespace NUMINAMATH_CALUDE_nonnegative_rational_function_l1687_168707

theorem nonnegative_rational_function (x : ℝ) :
  (x^2 - 6*x + 9) / (9 - x^3) ≥ 0 ↔ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_nonnegative_rational_function_l1687_168707


namespace NUMINAMATH_CALUDE_calculation_proof_l1687_168710

theorem calculation_proof : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1687_168710


namespace NUMINAMATH_CALUDE_min_m_value_x_range_l1687_168779

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1

-- Part 1: Minimum value of m
theorem min_m_value (a b : ℝ) (h : conditions a b) :
  ∀ m : ℝ, (∀ a b : ℝ, conditions a b → a * b ≤ m) → m ≥ 1/4 :=
sorry

-- Part 2: Range of x
theorem x_range (a b : ℝ) (h : conditions a b) :
  ∀ x : ℝ, (4/a + 1/b ≥ |2*x - 1| - |x + 2|) ↔ -6 ≤ x ∧ x ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_min_m_value_x_range_l1687_168779


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l1687_168777

theorem line_hyperbola_intersection (k : ℝ) : 
  (∀ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 4 → ∃ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 4) ↔ 
  (k = 1 ∨ k = -1 ∨ (-Real.sqrt 5 / 2 ≤ k ∧ k ≤ Real.sqrt 5 / 2)) :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l1687_168777


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1687_168789

theorem arithmetic_computation : 2 + 3^2 * 4 - 5 + 6 / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1687_168789


namespace NUMINAMATH_CALUDE_price_reduction_equation_l1687_168786

theorem price_reduction_equation (x : ℝ) : 
  (100 : ℝ) * (1 - x)^2 = 80 ↔ 
  (∃ (price1 price2 : ℝ), 
    price1 = 100 * (1 - x) ∧ 
    price2 = price1 * (1 - x) ∧ 
    price2 = 80) :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l1687_168786


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1687_168715

theorem min_value_of_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + 8*y - x*y = 0) :
  x + y ≥ 18 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 8*y₀ - x₀*y₀ = 0 ∧ x₀ + y₀ = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1687_168715


namespace NUMINAMATH_CALUDE_mike_lawn_mowing_earnings_l1687_168706

def mower_blade_cost : ℕ := 24
def game_cost : ℕ := 5
def num_games : ℕ := 9

theorem mike_lawn_mowing_earnings :
  ∃ (total_earnings : ℕ),
    total_earnings = mower_blade_cost + (game_cost * num_games) :=
by
  sorry

end NUMINAMATH_CALUDE_mike_lawn_mowing_earnings_l1687_168706


namespace NUMINAMATH_CALUDE_kylie_picked_558_apples_l1687_168746

/-- Represents the number of apples picked in each hour -/
structure ApplesPicked where
  first_hour : ℕ
  second_hour : ℕ
  third_hour : ℕ

/-- Calculates the total number of apples picked -/
def total_apples (ap : ApplesPicked) : ℕ :=
  ap.first_hour + ap.second_hour + ap.third_hour

/-- Represents the first three Fibonacci numbers -/
def first_three_fibonacci : List ℕ := [1, 1, 2]

/-- Represents the first three terms of the arithmetic progression -/
def arithmetic_progression (a₁ d : ℕ) : List ℕ :=
  [a₁, a₁ + d, a₁ + 2*d]

/-- Kylie's apple picking scenario -/
def kylie_apples : ApplesPicked where
  first_hour := 66
  second_hour := (List.sum first_three_fibonacci) * 66
  third_hour := List.sum (arithmetic_progression 66 10)

/-- Theorem stating that Kylie picked 558 apples in total -/
theorem kylie_picked_558_apples :
  total_apples kylie_apples = 558 := by
  sorry


end NUMINAMATH_CALUDE_kylie_picked_558_apples_l1687_168746


namespace NUMINAMATH_CALUDE_total_mechanical_pencils_l1687_168728

/-- Given 4 sets of school supplies with 16 mechanical pencils each, 
    prove that the total number of mechanical pencils is 64. -/
theorem total_mechanical_pencils : 
  let num_sets : ℕ := 4
  let pencils_per_set : ℕ := 16
  num_sets * pencils_per_set = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_mechanical_pencils_l1687_168728


namespace NUMINAMATH_CALUDE_division_problem_l1687_168719

theorem division_problem (x : ℝ) (h : 82.04 / x = 28) : x = 2.93 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1687_168719


namespace NUMINAMATH_CALUDE_equidistant_points_in_quadrants_I_II_l1687_168730

/-- A point (x, y) on the line 4x + 6y = 18 that is equidistant from both coordinate axes -/
def EquidistantPoint (x y : ℝ) : Prop :=
  4 * x + 6 * y = 18 ∧ |x| = |y|

/-- A point (x, y) is in quadrant I -/
def InQuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- A point (x, y) is in quadrant II -/
def InQuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- A point (x, y) is in quadrant III -/
def InQuadrantIII (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- A point (x, y) is in quadrant IV -/
def InQuadrantIV (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem equidistant_points_in_quadrants_I_II :
  ∀ x y : ℝ, EquidistantPoint x y →
  (InQuadrantI x y ∨ InQuadrantII x y) ∧
  ¬(InQuadrantIII x y ∨ InQuadrantIV x y) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_points_in_quadrants_I_II_l1687_168730


namespace NUMINAMATH_CALUDE_gcd_problems_l1687_168720

theorem gcd_problems : 
  (Nat.gcd 120 168 = 24) ∧ (Nat.gcd 459 357 = 51) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l1687_168720


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1687_168722

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 1/5) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1687_168722


namespace NUMINAMATH_CALUDE_tony_water_consumption_l1687_168705

/-- Calculates the daily water consumption given the bottle capacity, number of refills per week, and days in a week. -/
def daily_water_consumption (bottle_capacity : ℕ) (refills_per_week : ℕ) (days_in_week : ℕ) : ℚ :=
  (bottle_capacity * refills_per_week : ℚ) / days_in_week

/-- Proves that given a water bottle capacity of 84 ounces, filled 6 times per week, 
    and 7 days in a week, the daily water consumption is 72 ounces. -/
theorem tony_water_consumption :
  daily_water_consumption 84 6 7 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tony_water_consumption_l1687_168705


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1687_168713

/-- Represents a workshop with its production quantity -/
structure Workshop where
  quantity : ℕ

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  workshops : List Workshop
  sampleSize : ℕ
  sampledFromC : ℕ

/-- Calculates the total production quantity across all workshops -/
def totalQuantity (s : StratifiedSample) : ℕ :=
  s.workshops.foldl (fun acc w => acc + w.quantity) 0

/-- Theorem stating the relationship between sample size and workshop quantities -/
theorem stratified_sample_size 
  (s : StratifiedSample)
  (hWorkshops : s.workshops = [⟨600⟩, ⟨400⟩, ⟨300⟩])
  (hSampledC : s.sampledFromC = 6) :
  s.sampleSize = 26 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l1687_168713


namespace NUMINAMATH_CALUDE_henrys_initial_book_count_l1687_168764

/-- Calculates the initial number of books in Henry's collection --/
def initialBookCount (boxCount : ℕ) (booksPerBox : ℕ) (roomBooks : ℕ) (tableBooks : ℕ) (kitchenBooks : ℕ) (pickedUpBooks : ℕ) (remainingBooks : ℕ) : ℕ :=
  boxCount * booksPerBox + roomBooks + tableBooks + kitchenBooks - pickedUpBooks + remainingBooks

/-- Theorem stating that Henry's initial book count is 99 --/
theorem henrys_initial_book_count :
  initialBookCount 3 15 21 4 18 12 23 = 99 := by
  sorry

end NUMINAMATH_CALUDE_henrys_initial_book_count_l1687_168764


namespace NUMINAMATH_CALUDE_certain_number_proof_l1687_168749

theorem certain_number_proof : 
  ∃ x : ℚ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1687_168749


namespace NUMINAMATH_CALUDE_fly_distance_l1687_168700

/-- Prove that the distance traveled by a fly between two approaching cyclists is 50 km -/
theorem fly_distance (initial_distance : ℝ) (cyclist1_speed cyclist2_speed fly_speed : ℝ) :
  initial_distance = 50 →
  cyclist1_speed = 40 →
  cyclist2_speed = 60 →
  fly_speed = 100 →
  let relative_speed := cyclist1_speed + cyclist2_speed
  let time := initial_distance / relative_speed
  fly_speed * time = 50 := by sorry

end NUMINAMATH_CALUDE_fly_distance_l1687_168700


namespace NUMINAMATH_CALUDE_f_f_zero_equals_three_pi_squared_minus_four_l1687_168718

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

theorem f_f_zero_equals_three_pi_squared_minus_four :
  f (f 0) = 3 * Real.pi^2 - 4 := by sorry

end NUMINAMATH_CALUDE_f_f_zero_equals_three_pi_squared_minus_four_l1687_168718


namespace NUMINAMATH_CALUDE_complex_z_imaginary_part_l1687_168765

theorem complex_z_imaginary_part (z : ℂ) (h : (3 + 4 * Complex.I) * z = Complex.abs (3 - 4 * Complex.I)) : 
  z.im = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_z_imaginary_part_l1687_168765


namespace NUMINAMATH_CALUDE_helen_thanksgiving_desserts_l1687_168763

/-- The number of chocolate chip cookies Helen baked -/
def chocolate_chip_cookies : ℕ := 435

/-- The number of sugar cookies Helen baked -/
def sugar_cookies : ℕ := 139

/-- The number of brownies Helen made -/
def brownies : ℕ := 215

/-- The total number of desserts Helen prepared for Thanksgiving -/
def total_desserts : ℕ := chocolate_chip_cookies + sugar_cookies + brownies

theorem helen_thanksgiving_desserts : total_desserts = 789 := by
  sorry

end NUMINAMATH_CALUDE_helen_thanksgiving_desserts_l1687_168763


namespace NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l1687_168724

-- Part 1: No sequence of positive integers satisfying the condition
theorem no_positive_integer_sequence :
  ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2)) := by sorry

-- Part 2: Existence of a sequence of positive irrational numbers satisfying the condition
theorem exists_positive_irrational_sequence :
  ∃ (a : ℕ → ℝ), (∀ n : ℕ, Irrational (a n) ∧ a n > 0) ∧
    (∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) := by sorry

end NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l1687_168724


namespace NUMINAMATH_CALUDE_first_investment_rate_l1687_168709

/-- Proves that given the conditions, the interest rate of the first investment is 10% --/
theorem first_investment_rate (total_investment : ℝ) (second_investment : ℝ) (second_rate : ℝ) (income_difference : ℝ) :
  total_investment = 2000 →
  second_investment = 650 →
  second_rate = 0.08 →
  income_difference = 83 →
  ∃ (first_rate : ℝ),
    first_rate * (total_investment - second_investment) - second_rate * second_investment = income_difference ∧
    first_rate = 0.10 :=
by sorry

end NUMINAMATH_CALUDE_first_investment_rate_l1687_168709


namespace NUMINAMATH_CALUDE_paint_mixture_theorem_l1687_168742

theorem paint_mixture_theorem (total : ℝ) (blue_added : ℝ) (white_added : ℝ) :
  white_added = 20 →
  blue_added / total = 0.7 →
  white_added / total = 0.1 →
  blue_added = 140 := by
sorry

end NUMINAMATH_CALUDE_paint_mixture_theorem_l1687_168742


namespace NUMINAMATH_CALUDE_birth_date_satisfies_conditions_l1687_168701

/-- Represents a date with year, month, and day components -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Calculates the age of a person at a given year, given their birth date -/
def age (birthDate : Date) (currentYear : ℕ) : ℕ :=
  currentYear - birthDate.year

/-- Represents the problem conditions -/
def satisfiesConditions (birthDate : Date) : Prop :=
  let ageIn1937 := age birthDate 1937
  ageIn1937 * ageIn1937 = 1937 - birthDate.year ∧ 
  ageIn1937 + birthDate.month = birthDate.day * birthDate.day

/-- The main theorem to prove -/
theorem birth_date_satisfies_conditions : 
  satisfiesConditions (Date.mk 1892 5 7) :=
sorry

end NUMINAMATH_CALUDE_birth_date_satisfies_conditions_l1687_168701


namespace NUMINAMATH_CALUDE_chord_bisector_line_l1687_168760

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a parabola y² = 4x -/
def onParabola (p : Point) : Prop := p.y^2 = 4 * p.x

/-- Checks if a point lies on a line -/
def onLine (p : Point) (l : Line) : Prop := l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

/-- The main theorem -/
theorem chord_bisector_line (A B : Point) (P : Point) :
  onParabola A ∧ onParabola B ∧ 
  isMidpoint P A B ∧ 
  P.x = 1 ∧ P.y = 1 →
  ∃ l : Line, l.a = 2 ∧ l.b = -1 ∧ l.c = -1 ∧ onLine A l ∧ onLine B l :=
by sorry

end NUMINAMATH_CALUDE_chord_bisector_line_l1687_168760


namespace NUMINAMATH_CALUDE_line_distance_theorem_l1687_168738

/-- The line equation 4x - 3y + c = 0 -/
def line_equation (x y c : ℝ) : Prop := 4 * x - 3 * y + c = 0

/-- The distance function from (1,1) to (a,b) -/
def distance_squared (a b : ℝ) : ℝ := (a - 1)^2 + (b - 1)^2

/-- The theorem stating the relationship between the line and the minimum distance -/
theorem line_distance_theorem (a b c : ℝ) :
  line_equation a b c →
  (∀ x y, line_equation x y c → distance_squared a b ≤ distance_squared x y) →
  distance_squared a b = 4 →
  c = -11 ∨ c = 9 := by sorry

end NUMINAMATH_CALUDE_line_distance_theorem_l1687_168738


namespace NUMINAMATH_CALUDE_unique_birth_year_exists_l1687_168737

def sumOfDigits (year : Nat) : Nat :=
  (year / 1000) + ((year / 100) % 10) + ((year / 10) % 10) + (year % 10)

theorem unique_birth_year_exists : 
  ∃! year : Nat, 1900 ≤ year ∧ year < 2003 ∧ 2003 - year = sumOfDigits year := by
  sorry

end NUMINAMATH_CALUDE_unique_birth_year_exists_l1687_168737


namespace NUMINAMATH_CALUDE_division_with_remainder_l1687_168783

theorem division_with_remainder : ∃ (q r : ℤ), 1234567 = 131 * q + r ∧ 0 ≤ r ∧ r < 131 ∧ r = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l1687_168783


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1687_168727

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) : 
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d ≠ 0 →  -- not a constant sequence
  (a 2) * (a 6) = (a 3) * (a 3) →  -- 2nd, 3rd, and 6th terms form a geometric sequence
  (a 3) / (a 2) = 3 :=  -- common ratio is 3
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1687_168727


namespace NUMINAMATH_CALUDE_sequence_split_equal_sum_l1687_168731

theorem sequence_split_equal_sum (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ (b : ℕ) (S : ℕ) (splits : List (List ℕ)),
    b > 1 ∧
    splits.length = b ∧
    (∀ l ∈ splits, l.sum = S) ∧
    splits.join = List.range p) ↔ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_split_equal_sum_l1687_168731


namespace NUMINAMATH_CALUDE_simplify_expression_l1687_168798

theorem simplify_expression : 
  (Real.sqrt 308 / Real.sqrt 77) - (Real.sqrt 245 / Real.sqrt 49) = 2 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1687_168798


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l1687_168752

/-- The quadratic equation (a-1)x^2 + x + a^2 - 1 = 0 has 0 as one of its roots if and only if a = -1 -/
theorem quadratic_root_zero (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 + x + a^2 - 1 = 0 ∧ x = 0) ↔ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l1687_168752


namespace NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l1687_168729

theorem circle_radius_from_longest_chord (c : ℝ) (h : c > 0) :
  (∃ (r : ℝ), r > 0 ∧ 2 * r = c) → c / 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l1687_168729


namespace NUMINAMATH_CALUDE_tangent_slope_at_half_l1687_168791

-- Define the function f(x) = x^3 - 2
def f (x : ℝ) : ℝ := x^3 - 2

-- State the theorem
theorem tangent_slope_at_half :
  (deriv f) (1/2) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_half_l1687_168791


namespace NUMINAMATH_CALUDE_lunks_needed_correct_l1687_168743

/-- Exchange rate between lunks and kunks -/
def lunk_to_kunk_rate : ℚ := 1/2

/-- Exchange rate between kunks and apples -/
def kunk_to_apple_rate : ℚ := 5/3

/-- Number of apples to purchase -/
def apples_to_buy : ℕ := 20

/-- The number of lunks needed to purchase the given number of apples -/
def lunks_needed : ℕ := 24

theorem lunks_needed_correct : 
  ↑lunks_needed = ↑apples_to_buy / (kunk_to_apple_rate * lunk_to_kunk_rate) := by
  sorry

end NUMINAMATH_CALUDE_lunks_needed_correct_l1687_168743


namespace NUMINAMATH_CALUDE_no_real_graph_l1687_168714

/-- The equation x^2 + y^2 + 2x + 4y + 6 = 0 does not represent any real graph in the xy-plane. -/
theorem no_real_graph : ¬∃ (x y : ℝ), x^2 + y^2 + 2*x + 4*y + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_graph_l1687_168714


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1687_168782

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1687_168782


namespace NUMINAMATH_CALUDE_min_solution_value_l1687_168751

def system (x y : ℝ) : Prop :=
  3^(-x) * y^4 - 2*y^2 + 3^x ≤ 0 ∧ 27^x + y^4 - 3^x - 1 = 0

def solution_value (x y : ℝ) : ℝ := x^3 + y^3

theorem min_solution_value :
  ∃ (min : ℝ), min = -1 ∧
  (∀ x y : ℝ, system x y → solution_value x y ≥ min) ∧
  (∃ x y : ℝ, system x y ∧ solution_value x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_solution_value_l1687_168751


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_when_f_always_ge_4_l1687_168785

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solution_set_when_a_is_4 :
  {x : ℝ | f 4 x ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} := by sorry

-- Part 2
theorem range_of_a_when_f_always_ge_4 :
  (∀ x : ℝ, f a x ≥ 4) → (a ≤ -3 ∨ a ≥ 5) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_when_f_always_ge_4_l1687_168785


namespace NUMINAMATH_CALUDE_conversion_1_conversion_2_conversion_3_conversion_4_l1687_168741

-- Define conversion rates
def sq_meter_to_sq_decimeter : ℝ := 100
def hectare_to_sq_meter : ℝ := 10000
def sq_decimeter_to_sq_centimeter : ℝ := 100
def sq_kilometer_to_hectare : ℝ := 100

-- Theorem statements
theorem conversion_1 : 3 * sq_meter_to_sq_decimeter = 300 := by sorry

theorem conversion_2 : 2 * hectare_to_sq_meter = 20000 := by sorry

theorem conversion_3 : 5000 / sq_decimeter_to_sq_centimeter = 50 := by sorry

theorem conversion_4 : 8 * sq_kilometer_to_hectare = 800 := by sorry

end NUMINAMATH_CALUDE_conversion_1_conversion_2_conversion_3_conversion_4_l1687_168741


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l1687_168745

theorem relationship_between_x_and_y 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (x : ℝ) 
  (hx : x = Real.sqrt (a + b) - Real.sqrt b) 
  (y : ℝ) 
  (hy : y = Real.sqrt b - Real.sqrt (b - a)) : 
  x < y := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l1687_168745


namespace NUMINAMATH_CALUDE_divisibility_property_l1687_168717

theorem divisibility_property (p : ℕ) (h1 : Even p) (h2 : p > 2) :
  ∃ k : ℤ, (p + 1) ^ (p / 2) - 1 = k * p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1687_168717


namespace NUMINAMATH_CALUDE_sqrt_seven_identities_l1687_168781

theorem sqrt_seven_identities (a b : ℝ) (ha : a = Real.sqrt 7 + 2) (hb : b = Real.sqrt 7 - 2) :
  (a * b = 3) ∧ (a^2 + b^2 - a * b = 19) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_identities_l1687_168781


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1687_168703

/-- The area of a rectangular field with one side of 4 meters and a diagonal of 5 meters is 12 square meters. -/
theorem rectangular_field_area : ∀ (w l : ℝ), 
  w = 4 → 
  w^2 + l^2 = 5^2 → 
  w * l = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1687_168703


namespace NUMINAMATH_CALUDE_expression_expansion_l1687_168733

theorem expression_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (7 / x^2 - 5 * x^3 + 2) = 3 / x^2 - 15 * x^3 / 7 + 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_expansion_l1687_168733


namespace NUMINAMATH_CALUDE_triangle_count_l1687_168761

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of collinear triplets in the given configuration -/
def collinearTriplets : ℕ := 16

/-- The total number of points in the configuration -/
def totalPoints : ℕ := 12

/-- The number of points needed to form a triangle -/
def pointsPerTriangle : ℕ := 3

theorem triangle_count :
  choose totalPoints pointsPerTriangle - collinearTriplets = 204 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l1687_168761


namespace NUMINAMATH_CALUDE_ben_homework_theorem_l1687_168792

/-- The time in minutes Ben has to work on homework -/
def total_time : ℕ := 60

/-- The time taken to solve the i-th problem -/
def problem_time (i : ℕ) : ℕ := i

/-- The sum of time taken to solve the first n problems -/
def total_problem_time (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

/-- The maximum number of problems Ben can solve -/
def max_problems : ℕ := 10

theorem ben_homework_theorem :
  (∀ n : ℕ, n > max_problems → total_problem_time n > total_time) ∧
  total_problem_time max_problems ≤ total_time :=
sorry

end NUMINAMATH_CALUDE_ben_homework_theorem_l1687_168792


namespace NUMINAMATH_CALUDE_triangle_inequality_l1687_168732

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) : 
  3 * (a * b + a * c + b * c) ≤ (a + b + c)^2 ∧ (a + b + c)^2 < 4 * (a * b + a * c + b * c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1687_168732


namespace NUMINAMATH_CALUDE_triangle_angle_values_l1687_168769

theorem triangle_angle_values (a b c A B C : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  Real.cos A * Real.sin C = (Real.sqrt 3 - 1) / 4 →
  -- Conclusions
  B = π / 3 ∧ A = 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_values_l1687_168769


namespace NUMINAMATH_CALUDE_words_per_page_l1687_168754

theorem words_per_page (total_pages : Nat) (words_mod : Nat) (mod_value : Nat) :
  total_pages = 150 →
  words_mod = 210 →
  mod_value = 221 →
  ∃ (words_per_page : Nat),
    words_per_page ≤ 120 ∧
    (total_pages * words_per_page) % mod_value = words_mod ∧
    words_per_page = 195 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l1687_168754


namespace NUMINAMATH_CALUDE_bridge_length_l1687_168790

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 250 →
  train_speed_kmh = 72 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 350 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1687_168790


namespace NUMINAMATH_CALUDE_snowfall_rate_hamilton_l1687_168747

/-- Snowfall rates and depths in Kingston and Hamilton --/
theorem snowfall_rate_hamilton (
  kingston_initial : ℝ) (hamilton_initial : ℝ) 
  (duration : ℝ) (kingston_rate : ℝ) (hamilton_rate : ℝ) :
  kingston_initial = 12.1 →
  hamilton_initial = 18.6 →
  duration = 13 →
  kingston_rate = 2.6 →
  kingston_initial + kingston_rate * duration = hamilton_initial + hamilton_rate * duration →
  hamilton_rate = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_snowfall_rate_hamilton_l1687_168747


namespace NUMINAMATH_CALUDE_envelope_addressing_machines_l1687_168788

theorem envelope_addressing_machines (machine1_time machine2_time combined_time : ℚ) :
  machine1_time = 10 →
  combined_time = 4 →
  (1 / machine1_time + 1 / machine2_time = 1 / combined_time) →
  machine2_time = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_envelope_addressing_machines_l1687_168788


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l1687_168755

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ b < a, (Nat.gcd b 70 = 1 ∨ Nat.gcd b 84 = 1)) → 
  (Nat.gcd a 70 > 1 ∧ Nat.gcd a 84 > 1) → 
  a = 14 := by sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l1687_168755


namespace NUMINAMATH_CALUDE_quarters_count_l1687_168758

/-- Represents the types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a collection of coins --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  total_coins : pennies + nickels + dimes + quarters = 11
  at_least_one : pennies ≥ 1 ∧ nickels ≥ 1 ∧ dimes ≥ 1 ∧ quarters ≥ 1
  total_value : pennies * coinValue Coin.Penny +
                nickels * coinValue Coin.Nickel +
                dimes * coinValue Coin.Dime +
                quarters * coinValue Coin.Quarter = 132

theorem quarters_count (cc : CoinCollection) : cc.quarters = 3 := by
  sorry

end NUMINAMATH_CALUDE_quarters_count_l1687_168758


namespace NUMINAMATH_CALUDE_jana_walking_distance_l1687_168726

/-- Given a walking speed of 1 mile per 24 minutes, prove that the distance walked in 36 minutes is 1.5 miles. -/
theorem jana_walking_distance (speed : ℚ) (time : ℕ) (distance : ℚ) : 
  speed = 1 / 24 → time = 36 → distance = speed * time → distance = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_jana_walking_distance_l1687_168726


namespace NUMINAMATH_CALUDE_largest_percentage_increase_l1687_168744

def students : Fin 6 → ℕ
  | 0 => 50  -- 2010
  | 1 => 55  -- 2011
  | 2 => 60  -- 2012
  | 3 => 72  -- 2013
  | 4 => 75  -- 2014
  | 5 => 90  -- 2015

def percentageIncrease (year : Fin 5) : ℚ :=
  (students (year.succ) - students year : ℚ) / students year * 100

theorem largest_percentage_increase :
  (∀ year : Fin 5, percentageIncrease year ≤ percentageIncrease 2 ∨ percentageIncrease year ≤ percentageIncrease 4) ∧
  percentageIncrease 2 = percentageIncrease 4 :=
sorry

end NUMINAMATH_CALUDE_largest_percentage_increase_l1687_168744


namespace NUMINAMATH_CALUDE_a_in_A_l1687_168739

def A : Set ℝ := {x | x < 2 * Real.sqrt 3}

theorem a_in_A : 2 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_a_in_A_l1687_168739


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l1687_168750

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_products_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l1687_168750


namespace NUMINAMATH_CALUDE_ramsey_3_3_l1687_168734

/-- A complete graph with 6 vertices where each edge is colored either blue or red. -/
def ColoredGraph := Fin 6 → Fin 6 → Bool

/-- The graph is complete and each edge has a color (blue or red). -/
def is_valid_coloring (g : ColoredGraph) : Prop :=
  ∀ i j : Fin 6, i ≠ j → g i j = true ∨ g i j = false

/-- A triangle in the graph with all edges of the same color. -/
def monochromatic_triangle (g : ColoredGraph) : Prop :=
  ∃ i j k : Fin 6, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    ((g i j = true ∧ g j k = true ∧ g i k = true) ∨
     (g i j = false ∧ g j k = false ∧ g i k = false))

/-- The Ramsey theorem for R(3,3) -/
theorem ramsey_3_3 (g : ColoredGraph) (h : is_valid_coloring g) : 
  monochromatic_triangle g := by
  sorry

end NUMINAMATH_CALUDE_ramsey_3_3_l1687_168734


namespace NUMINAMATH_CALUDE_good_quality_sufficient_for_not_cheap_l1687_168708

-- Define the propositions
variable (good_quality : Prop)
variable (not_cheap : Prop)

-- Define the given equivalence
axiom you_get_what_you_pay_for : (good_quality → not_cheap) ↔ (¬not_cheap → ¬good_quality)

-- Theorem to prove
theorem good_quality_sufficient_for_not_cheap : good_quality → not_cheap := by
  sorry

end NUMINAMATH_CALUDE_good_quality_sufficient_for_not_cheap_l1687_168708


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_cube_l1687_168702

theorem sphere_surface_area_of_circumscribed_cube (edge_length : ℝ) 
  (h : edge_length = 2 * Real.sqrt 3) :
  let diagonal := Real.sqrt 3 * edge_length
  let radius := diagonal / 2
  4 * Real.pi * radius ^ 2 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_cube_l1687_168702


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l1687_168778

/-- A circle passing through three given points -/
def circle_through_points (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 + p.2 ^ 2 + 4 * p.1 - 2 * p.2) = 0}

/-- The three given points -/
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (0, 2)

/-- Theorem stating that the defined circle passes through the given points -/
theorem circle_passes_through_points :
  A ∈ circle_through_points A B C ∧
  B ∈ circle_through_points A B C ∧
  C ∈ circle_through_points A B C :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l1687_168778


namespace NUMINAMATH_CALUDE_salary_calculation_l1687_168712

def initial_salary : ℚ := 3000
def raise_percentage : ℚ := 10 / 100
def pay_cut_percentage : ℚ := 15 / 100
def bonus : ℚ := 500

def final_salary : ℚ := 
  (initial_salary * (1 + raise_percentage) * (1 - pay_cut_percentage)) + bonus

theorem salary_calculation : final_salary = 3305 := by sorry

end NUMINAMATH_CALUDE_salary_calculation_l1687_168712


namespace NUMINAMATH_CALUDE_complex_equation_implies_product_l1687_168759

/-- Given that (1+mi)/i = 1+ni where m, n ∈ ℝ and i is the imaginary unit, prove that mn = -1 -/
theorem complex_equation_implies_product (m n : ℝ) : 
  (1 + m * Complex.I) / Complex.I = 1 + n * Complex.I → m * n = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_product_l1687_168759
