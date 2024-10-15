import Mathlib

namespace NUMINAMATH_CALUDE_monic_polynomial_problem_l2179_217928

theorem monic_polynomial_problem (g : ℝ → ℝ) :
  (∃ b c : ℝ, ∀ x, g x = x^2 + b*x + c) →  -- g is a monic polynomial of degree 2
  g 0 = 6 →                               -- g(0) = 6
  g 1 = 12 →                              -- g(1) = 12
  ∀ x, g x = x^2 + 5*x + 6 :=              -- Conclusion: g(x) = x^2 + 5x + 6
by
  sorry

end NUMINAMATH_CALUDE_monic_polynomial_problem_l2179_217928


namespace NUMINAMATH_CALUDE_intersection_M_N_l2179_217930

def M : Set ℝ := {x | x^2 - 1 < 0}

def N : Set ℝ := {y | ∃ x ∈ M, y = Real.log (x + 2)}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2179_217930


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2179_217925

def A : Set ℝ := {x | |x - 1| ≤ 1}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2179_217925


namespace NUMINAMATH_CALUDE_sprinter_target_heart_rate_l2179_217920

/-- Calculates the maximum heart rate given the age --/
def maxHeartRate (age : ℕ) : ℕ := 225 - age

/-- Calculates the target heart rate as a percentage of the maximum heart rate --/
def targetHeartRate (maxRate : ℕ) : ℚ := 0.85 * maxRate

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (x : ℚ) : ℤ := (x + 1/2).floor

theorem sprinter_target_heart_rate :
  let age : ℕ := 30
  let max_rate := maxHeartRate age
  let target_rate := targetHeartRate max_rate
  roundToNearest target_rate = 166 := by sorry

end NUMINAMATH_CALUDE_sprinter_target_heart_rate_l2179_217920


namespace NUMINAMATH_CALUDE_rectangle_perimeter_product_l2179_217950

theorem rectangle_perimeter_product (a b c d : ℝ) : 
  (a + b = 11 ∧ a + b + c = 19.5 ∧ c = d) ∨
  (a + c = 11 ∧ a + b + c = 19.5 ∧ b = d) ∨
  (b + c = 11 ∧ a + b + c = 19.5 ∧ a = d) →
  (2 * (a + b)) * (2 * (a + c)) * (2 * (b + c)) = 15400 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_product_l2179_217950


namespace NUMINAMATH_CALUDE_banana_arrangements_l2179_217977

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  factorial total_letters / (factorial a_count * factorial n_count * factorial b_count) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l2179_217977


namespace NUMINAMATH_CALUDE_positive_sum_and_product_imply_positive_quadratic_root_conditions_quadratic_root_conditions_not_sufficient_l2179_217986

-- Statement 3
theorem positive_sum_and_product_imply_positive (a b : ℝ) :
  a + b > 0 → a * b > 0 → a > 0 ∧ b > 0 := by sorry

-- Statement 4
def has_two_distinct_positive_roots (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0

theorem quadratic_root_conditions (a b c : ℝ) (h : a ≠ 0) :
  has_two_distinct_positive_roots a b c →
  b / a < 0 ∧ c / a > 0 := by sorry

theorem quadratic_root_conditions_not_sufficient (a b c : ℝ) (h : a ≠ 0) :
  b / a < 0 ∧ c / a > 0 →
  ¬(has_two_distinct_positive_roots a b c ↔ True) := by sorry

end NUMINAMATH_CALUDE_positive_sum_and_product_imply_positive_quadratic_root_conditions_quadratic_root_conditions_not_sufficient_l2179_217986


namespace NUMINAMATH_CALUDE_free_throw_probabilities_l2179_217999

/-- Free throw success rates for players A and B -/
structure FreeThrowRates where
  player_a : ℚ
  player_b : ℚ

/-- Calculates the probability of exactly one successful shot when each player takes one free throw -/
def prob_one_success (rates : FreeThrowRates) : ℚ :=
  rates.player_a * (1 - rates.player_b) + rates.player_b * (1 - rates.player_a)

/-- Calculates the probability of at least one successful shot when each player takes two free throws -/
def prob_at_least_one_success (rates : FreeThrowRates) : ℚ :=
  1 - (1 - rates.player_a)^2 * (1 - rates.player_b)^2

/-- Theorem stating the probabilities for the given free throw rates -/
theorem free_throw_probabilities (rates : FreeThrowRates) 
  (h1 : rates.player_a = 1/2) (h2 : rates.player_b = 2/5) : 
  prob_one_success rates = 1/2 ∧ prob_at_least_one_success rates = 91/100 := by
  sorry

#eval prob_one_success ⟨1/2, 2/5⟩
#eval prob_at_least_one_success ⟨1/2, 2/5⟩

end NUMINAMATH_CALUDE_free_throw_probabilities_l2179_217999


namespace NUMINAMATH_CALUDE_john_vowel_learning_days_l2179_217963

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of days John takes to learn one alphabet -/
def days_per_alphabet : ℕ := 3

/-- The total number of days John needs to finish learning all vowels -/
def total_days : ℕ := num_vowels * days_per_alphabet

theorem john_vowel_learning_days : total_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_vowel_learning_days_l2179_217963


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l2179_217952

theorem fraction_zero_implies_x_one (x : ℝ) (h : (x - 1) / x = 0) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l2179_217952


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2179_217981

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 28 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = (11 / 14) * 100 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2179_217981


namespace NUMINAMATH_CALUDE_vehicles_meeting_time_l2179_217939

/-- The time taken for two vehicles to meet when traveling towards each other -/
theorem vehicles_meeting_time (distance : ℝ) (speed1 speed2 : ℝ) (h1 : distance = 480) 
  (h2 : speed1 = 65) (h3 : speed2 = 55) : 
  (distance / (speed1 + speed2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_vehicles_meeting_time_l2179_217939


namespace NUMINAMATH_CALUDE_redo_profit_is_5000_l2179_217934

/-- Calculates the profit for Redo's horseshoe manufacturing --/
def horseshoe_profit (initial_outlay : ℕ) (cost_per_set : ℕ) (price_per_set : ℕ) (num_sets : ℕ) : ℤ :=
  let manufacturing_cost : ℕ := initial_outlay + cost_per_set * num_sets
  let revenue : ℕ := price_per_set * num_sets
  (revenue : ℤ) - (manufacturing_cost : ℤ)

theorem redo_profit_is_5000 :
  horseshoe_profit 10000 20 50 500 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_redo_profit_is_5000_l2179_217934


namespace NUMINAMATH_CALUDE_least_positive_tangent_inverse_l2179_217904

theorem least_positive_tangent_inverse (y p q : ℝ) (h1 : Real.tan y = p / q) (h2 : Real.tan (3 * y) = q / (p + q)) :
  ∃ m : ℝ, m > 0 ∧ y = Real.arctan m ∧ ∀ m' : ℝ, m' > 0 → y = Real.arctan m' → m ≤ m' ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_least_positive_tangent_inverse_l2179_217904


namespace NUMINAMATH_CALUDE_first_patient_sessions_l2179_217903

/-- Given a group of patients and their session requirements, prove the number of sessions for the first patient. -/
theorem first_patient_sessions
  (total_patients : ℕ)
  (total_sessions : ℕ)
  (patient2_sessions : ℕ → ℕ)
  (remaining_patients_sessions : ℕ)
  (h1 : total_patients = 4)
  (h2 : total_sessions = 25)
  (h3 : patient2_sessions x = x + 5)
  (h4 : remaining_patients_sessions = 8 + 8)
  (h5 : x + patient2_sessions x + remaining_patients_sessions = total_sessions) :
  x = 2 :=
by sorry

end NUMINAMATH_CALUDE_first_patient_sessions_l2179_217903


namespace NUMINAMATH_CALUDE_isochronous_growth_law_l2179_217989

theorem isochronous_growth_law (k α : ℝ) (h1 : k > 0) (h2 : α > 0) :
  (∀ (x y : ℝ), y = k * x^α → (16 * x)^α = 8 * y) → α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_isochronous_growth_law_l2179_217989


namespace NUMINAMATH_CALUDE_prob_same_color_l2179_217975

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of black balls in the bag -/
def black_balls : ℕ := 4

/-- The number of red balls in the bag -/
def red_balls : ℕ := 5

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls + red_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- The probability of drawing at least two balls of the same color -/
theorem prob_same_color : 
  (1 : ℚ) - (white_balls * black_balls * red_balls : ℚ) / (total_balls * (total_balls - 1) * (total_balls - 2) / 6) = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_l2179_217975


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2179_217931

theorem average_speed_calculation (local_distance : ℝ) (local_speed : ℝ) 
  (highway_distance : ℝ) (highway_speed : ℝ) : 
  local_distance = 40 ∧ local_speed = 20 ∧ highway_distance = 180 ∧ highway_speed = 60 →
  (local_distance + highway_distance) / ((local_distance / local_speed) + (highway_distance / highway_speed)) = 44 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2179_217931


namespace NUMINAMATH_CALUDE_not_consecutive_odd_beautiful_l2179_217932

def IsBeautiful (g : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, g x = g (a - x)

theorem not_consecutive_odd_beautiful
  (g : ℤ → ℤ)
  (h1 : ∀ x : ℤ, g x ≠ x)
  : ¬∃ a : ℤ, IsBeautiful g a ∧ IsBeautiful g (a + 2) ∧ Odd a :=
by sorry

end NUMINAMATH_CALUDE_not_consecutive_odd_beautiful_l2179_217932


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l2179_217918

theorem units_digit_of_fraction (n : ℕ) : n = 30 * 31 * 32 * 33 * 34 * 35 / 7200 → n % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l2179_217918


namespace NUMINAMATH_CALUDE_sequence_general_term_l2179_217912

/-- Given a sequence {a_n} with n ∈ ℕ, if S_n = 2a_n - 2^n + 1 represents
    the sum of the first n terms, then a_n = n × 2^(n-1) for all n ∈ ℕ. -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = 2 * a n - 2^n + 1) →
  ∀ n : ℕ, a n = n * 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2179_217912


namespace NUMINAMATH_CALUDE_power_multiplication_l2179_217901

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2179_217901


namespace NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l2179_217944

theorem tangent_line_to_x_ln_x (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (2 * x₀ + m = x₀ * Real.log x₀) ∧ 
    (2 = Real.log x₀ + 1)) → 
  m = -Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l2179_217944


namespace NUMINAMATH_CALUDE_probability_not_red_special_cube_l2179_217958

structure Cube where
  total_faces : ℕ
  green_faces : ℕ
  blue_faces : ℕ
  red_faces : ℕ

def probability_not_red (c : Cube) : ℚ :=
  (c.green_faces + c.blue_faces : ℚ) / c.total_faces

theorem probability_not_red_special_cube :
  let c : Cube := {
    total_faces := 6,
    green_faces := 3,
    blue_faces := 2,
    red_faces := 1
  }
  probability_not_red c = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_red_special_cube_l2179_217958


namespace NUMINAMATH_CALUDE_A_inverse_proof_l2179_217911

def A : Matrix (Fin 3) (Fin 3) ℚ := !![2, 5, 6; 1, 2, 5; 1, 2, 3]

def A_inv : Matrix (Fin 3) (Fin 3) ℚ := !![-2, 3/2, 13/2; 1, 0, 2; 0, -1/2, -1/2]

theorem A_inverse_proof : A⁻¹ = A_inv := by sorry

end NUMINAMATH_CALUDE_A_inverse_proof_l2179_217911


namespace NUMINAMATH_CALUDE_unique_recurrence_solution_l2179_217917

/-- A sequence of positive real numbers satisfying the given recurrence relation. -/
def RecurrenceSequence (X : ℕ → ℝ) : Prop :=
  (∀ n, X n > 0) ∧ 
  (∀ n, X (n + 2) = (1 / X (n + 1) + X n) / 2)

/-- The theorem stating that the only sequence satisfying the recurrence relation is the constant sequence of 1. -/
theorem unique_recurrence_solution (X : ℕ → ℝ) :
  RecurrenceSequence X → (∀ n, X n = 1) := by
  sorry

#check unique_recurrence_solution

end NUMINAMATH_CALUDE_unique_recurrence_solution_l2179_217917


namespace NUMINAMATH_CALUDE_sixty_degrees_is_hundred_clerts_l2179_217921

/-- Represents the number of clerts in a full circle in the Martian system -/
def full_circle_clerts : ℕ := 600

/-- Represents the number of degrees in a full circle in the Earth system -/
def full_circle_degrees : ℕ := 360

/-- Converts degrees to clerts -/
def degrees_to_clerts (degrees : ℕ) : ℚ :=
  (degrees : ℚ) * full_circle_clerts / full_circle_degrees

theorem sixty_degrees_is_hundred_clerts :
  degrees_to_clerts 60 = 100 := by sorry

end NUMINAMATH_CALUDE_sixty_degrees_is_hundred_clerts_l2179_217921


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2179_217974

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 6*x + 8 = 0 →
  x + 3 > 6 ∧ x + 6 > 3 ∧ 3 + 6 > x →
  x + 3 + 6 = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2179_217974


namespace NUMINAMATH_CALUDE_one_twenty_million_properties_l2179_217972

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

/-- Counts the number of significant figures in a scientific notation -/
def countSignificantFigures (sn : ScientificNotation) : ℕ := sorry

/-- Determines the place value of accuracy for a number -/
inductive PlaceValue
  | Ones
  | Tens
  | Hundreds
  | Thousands
  | TenThousands
  | HundredThousands
  | Millions
  | TenMillions
  | HundredMillions

def getPlaceValueAccuracy (x : ℝ) : PlaceValue := sorry

theorem one_twenty_million_properties :
  let x : ℝ := 120000000
  let sn := toScientificNotation x
  countSignificantFigures sn = 2 ∧ getPlaceValueAccuracy x = PlaceValue.Millions := by sorry

end NUMINAMATH_CALUDE_one_twenty_million_properties_l2179_217972


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l2179_217995

theorem other_solution_quadratic (h : 49 * (5/7)^2 - 88 * (5/7) + 40 = 0) :
  49 * (8/7)^2 - 88 * (8/7) + 40 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l2179_217995


namespace NUMINAMATH_CALUDE_area_transformation_l2179_217969

-- Define a function representing the area between a curve and the x-axis
noncomputable def area_between_curve_and_xaxis (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem area_transformation (f : ℝ → ℝ) (h : area_between_curve_and_xaxis f = 12) :
  area_between_curve_and_xaxis (λ x => 4 * f (x + 3)) = 48 :=
by sorry

end NUMINAMATH_CALUDE_area_transformation_l2179_217969


namespace NUMINAMATH_CALUDE_parabola_vertex_l2179_217966

/-- The parabola is defined by the equation y = (x + 3)^2 - 1 -/
def parabola (x : ℝ) : ℝ := (x + 3)^2 - 1

/-- The vertex of the parabola y = (x + 3)^2 - 1 is at the point (-3, -1) -/
theorem parabola_vertex : 
  (∃ (a : ℝ), ∀ (x : ℝ), parabola x = a * (x + 3)^2 - 1) → 
  (∀ (x : ℝ), parabola x ≥ parabola (-3)) ∧ parabola (-3) = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2179_217966


namespace NUMINAMATH_CALUDE_geese_in_marsh_l2179_217900

theorem geese_in_marsh (total_birds ducks : ℕ) (h1 : total_birds = 95) (h2 : ducks = 37) :
  total_birds - ducks = 58 := by
  sorry

end NUMINAMATH_CALUDE_geese_in_marsh_l2179_217900


namespace NUMINAMATH_CALUDE_polygon_sides_l2179_217906

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 720 → ∃ n : ℕ, n = 6 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2179_217906


namespace NUMINAMATH_CALUDE_no_valid_box_dimensions_l2179_217949

theorem no_valid_box_dimensions : 
  ¬∃ (a b c : ℕ), 
    (1 ≤ a) ∧ (a ≤ b) ∧ (b ≤ c) ∧ 
    (a * b * c = 3 * (2 * a * b + 2 * b * c + 2 * c * a)) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_box_dimensions_l2179_217949


namespace NUMINAMATH_CALUDE_odd_number_2009_group_l2179_217940

/-- The cumulative sum of odd numbers up to the n-th group -/
def cumulative_sum (n : ℕ) : ℕ := n^2

/-- The size of the n-th group -/
def group_size (n : ℕ) : ℕ := 2*n - 1

/-- The theorem stating that 2009 belongs to the 32nd group -/
theorem odd_number_2009_group : 
  (cumulative_sum 31 < 2009) ∧ (2009 ≤ cumulative_sum 32) := by sorry

end NUMINAMATH_CALUDE_odd_number_2009_group_l2179_217940


namespace NUMINAMATH_CALUDE_choose_3_from_10_l2179_217953

theorem choose_3_from_10 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_3_from_10_l2179_217953


namespace NUMINAMATH_CALUDE_square_from_l_pieces_l2179_217908

/-- Represents a three-cell L-shaped piece -/
structure LPiece :=
  (cells : Fin 3 → Fin 2 → Fin 2)

/-- Represents a square grid -/
structure Square (n : ℕ) :=
  (grid : Fin n → Fin n → Bool)

/-- Checks if a given square is filled completely -/
def is_filled (s : Square n) : Prop :=
  ∀ i j, s.grid i j = true

/-- Defines the ability to place L-pieces on a square grid -/
def can_place_pieces (n : ℕ) (pieces : List LPiece) (s : Square n) : Prop :=
  sorry

/-- The main theorem stating that it's possible to form a square using L-pieces -/
theorem square_from_l_pieces :
  ∃ (n : ℕ) (pieces : List LPiece) (s : Square n),
    can_place_pieces n pieces s ∧ is_filled s :=
  sorry

end NUMINAMATH_CALUDE_square_from_l_pieces_l2179_217908


namespace NUMINAMATH_CALUDE_hexagon_side_count_l2179_217957

-- Define a convex hexagon with two distinct side lengths
structure ConvexHexagon where
  side_length1 : ℕ
  side_length2 : ℕ
  side_count1 : ℕ
  side_count2 : ℕ
  distinct_lengths : side_length1 ≠ side_length2
  total_sides : side_count1 + side_count2 = 6

-- Theorem statement
theorem hexagon_side_count (h : ConvexHexagon) 
  (side_ab : h.side_length1 = 7)
  (side_bc : h.side_length2 = 8)
  (perimeter : h.side_length1 * h.side_count1 + h.side_length2 * h.side_count2 = 46) :
  h.side_count2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_count_l2179_217957


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l2179_217960

/-- Given an arithmetic sequence with 7 terms, first term 10, and common difference 12,
    prove that the average of all terms is 46. -/
theorem arithmetic_sequence_average : 
  let n : ℕ := 7
  let a : ℕ := 10
  let d : ℕ := 12
  let sequence := (fun i => a + d * (i - 1))
  let sum := (sequence 1 + sequence n) * n / 2
  (sum : ℚ) / n = 46 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l2179_217960


namespace NUMINAMATH_CALUDE_line_slope_problem_l2179_217943

/-- Given a line passing through points (-1, -4) and (3, k) with slope k, prove k = 4/3 -/
theorem line_slope_problem (k : ℚ) : 
  (k - (-4)) / (3 - (-1)) = k → k = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l2179_217943


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2179_217919

/-- 
Given that the coefficient of the third term in the binomial expansion 
of (x - 1/(2x))^n is 7, prove that n = 8.
-/
theorem binomial_expansion_coefficient (n : ℕ) : 
  (1/4 : ℚ) * (n.choose 2) = 7 → n = 8 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2179_217919


namespace NUMINAMATH_CALUDE_uncle_fyodor_sandwiches_l2179_217980

theorem uncle_fyodor_sandwiches (sharik matroskin fyodor : ℕ) : 
  matroskin = 3 * sharik →
  fyodor = sharik + 21 →
  fyodor = 2 * (sharik + matroskin) →
  fyodor = 24 := by
sorry

end NUMINAMATH_CALUDE_uncle_fyodor_sandwiches_l2179_217980


namespace NUMINAMATH_CALUDE_final_number_is_172_l2179_217998

/-- Represents the state of the board at any given time -/
structure BoardState where
  numbers : List Nat
  deriving Repr

/-- The operation of erasing two numbers and replacing them with their sum minus 1 -/
def boardOperation (state : BoardState) (i j : Nat) : BoardState :=
  { numbers := 
      (state.numbers.removeNth i).removeNth j ++ 
      [state.numbers[i]! + state.numbers[j]! - 1] }

/-- The invariant of the board state -/
def boardInvariant (state : BoardState) : Int :=
  state.numbers.sum - state.numbers.length

/-- Initial board state with numbers 1 to 20 -/
def initialBoard : BoardState :=
  { numbers := List.range 20 |>.map (· + 1) }

/-- Theorem stating that after 19 operations, the final number on the board is 172 -/
theorem final_number_is_172 : 
  ∃ (operations : List (Nat × Nat)),
    operations.length = 19 ∧
    (operations.foldl 
      (fun state (i, j) => boardOperation state i j) 
      initialBoard).numbers = [172] := by
  sorry

end NUMINAMATH_CALUDE_final_number_is_172_l2179_217998


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2179_217951

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (3 * a^3 - 5 * a^2 + 170 * a - 7 = 0) →
  (3 * b^3 - 5 * b^2 + 170 * b - 7 = 0) →
  (3 * c^3 - 5 * c^2 + 170 * c - 7 = 0) →
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 
  (11/3 - c)^3 + (11/3 - a)^3 + (11/3 - b)^3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2179_217951


namespace NUMINAMATH_CALUDE_john_saturday_earnings_l2179_217938

/-- The amount of money John earned on Saturday -/
def saturday_earnings : ℝ := sorry

/-- The amount of money John earned on Sunday -/
def sunday_earnings : ℝ := sorry

/-- The amount of money John earned the previous weekend -/
def previous_weekend_earnings : ℝ := 20

/-- The cost of the pogo stick -/
def pogo_stick_cost : ℝ := 60

/-- The additional amount John needs to buy the pogo stick -/
def additional_needed : ℝ := 13

theorem john_saturday_earnings :
  saturday_earnings = 18 ∧
  sunday_earnings = saturday_earnings / 2 ∧
  previous_weekend_earnings + saturday_earnings + sunday_earnings = pogo_stick_cost - additional_needed :=
by sorry

end NUMINAMATH_CALUDE_john_saturday_earnings_l2179_217938


namespace NUMINAMATH_CALUDE_function_properties_and_triangle_perimeter_l2179_217971

noncomputable def f (m : ℝ) (θ : ℝ) (x : ℝ) : ℝ := (m + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

theorem function_properties_and_triangle_perimeter
  (m : ℝ)
  (θ : ℝ)
  (h1 : ∀ x, f m θ x = -f m θ (-x))  -- f is an odd function
  (h2 : f m θ (π/4) = 0)
  (h3 : 0 < θ)
  (h4 : θ < π) :
  (∀ x, f m θ x = -1/2 * Real.sin (4*x)) ∧
  (∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    f m θ (Real.arccos (1/2) / 2 + π/24) = -1/2 ∧
    1 = 1 ∧
    a * b = 2 * Real.sqrt 3 ∧
    a + b + 1 = 3 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_and_triangle_perimeter_l2179_217971


namespace NUMINAMATH_CALUDE_perry_vs_phil_l2179_217983

/-- The number of games won by each player -/
structure GolfWins where
  phil : ℕ
  charlie : ℕ
  dana : ℕ
  perry : ℕ

/-- The conditions of the golf game results -/
def golf_conditions (w : GolfWins) : Prop :=
  w.perry = w.dana + 5 ∧
  w.charlie = w.dana - 2 ∧
  w.phil = w.charlie + 3 ∧
  w.phil = 12

theorem perry_vs_phil (w : GolfWins) (h : golf_conditions w) : w.perry = w.phil + 4 :=
sorry

end NUMINAMATH_CALUDE_perry_vs_phil_l2179_217983


namespace NUMINAMATH_CALUDE_unique_solution_l2179_217961

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_solution :
  ∃! x : ℕ, digit_product x = x^2 - 10*x - 22 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2179_217961


namespace NUMINAMATH_CALUDE_shadow_height_calculation_l2179_217968

/-- Given a tree and a person casting shadows, calculate the person's height -/
theorem shadow_height_calculation (tree_height tree_shadow alex_shadow : ℚ) 
  (h1 : tree_height = 50)
  (h2 : tree_shadow = 25)
  (h3 : alex_shadow = 20 / 12) : -- Convert 20 inches to feet
  tree_height / tree_shadow * alex_shadow = 10 / 3 := by
  sorry

#check shadow_height_calculation

end NUMINAMATH_CALUDE_shadow_height_calculation_l2179_217968


namespace NUMINAMATH_CALUDE_average_team_size_l2179_217985

theorem average_team_size (boys girls teams : ℕ) (h1 : boys = 83) (h2 : girls = 77) (h3 : teams = 4) :
  (boys + girls) / teams = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_team_size_l2179_217985


namespace NUMINAMATH_CALUDE_infantry_column_problem_l2179_217942

/-- Given an infantry column of length 1 km, moving at speed x km/h,
    and Sergeant Kim moving at speed 3x km/h, if the infantry column
    covers 2.4 km while Kim travels to the front of the column and back,
    then Kim's total distance traveled is 3.6 km. -/
theorem infantry_column_problem (x : ℝ) (h : x > 0) :
  let column_length : ℝ := 1
  let column_speed : ℝ := x
  let kim_speed : ℝ := 3 * x
  let column_distance : ℝ := 2.4
  let time := column_distance / column_speed
  let kim_distance := kim_speed * time
  kim_distance = 3.6 := by sorry

end NUMINAMATH_CALUDE_infantry_column_problem_l2179_217942


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_fourth_l2179_217948

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_fourth : 12 / (1 / 4) = 48 := by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_fourth_l2179_217948


namespace NUMINAMATH_CALUDE_no_real_solutions_for_x_l2179_217937

theorem no_real_solutions_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x + 1/y = 8) (eq2 : y + 1/x = 7/20) : False :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_x_l2179_217937


namespace NUMINAMATH_CALUDE_regular_polygon_with_140_degree_interior_angles_l2179_217987

theorem regular_polygon_with_140_degree_interior_angles (n : ℕ) 
  (h_regular : n ≥ 3) 
  (h_interior_angle : (n - 2) * 180 / n = 140) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_140_degree_interior_angles_l2179_217987


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2179_217964

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  swimmer : ℝ
  stream : ℝ

/-- Calculates the effective speed of the swimmer given the direction. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water
  (s : SwimmerSpeeds)
  (h_downstream : effectiveSpeed s true * 4 = 32)
  (h_upstream : effectiveSpeed s false * 4 = 24) :
  s.swimmer = 7 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2179_217964


namespace NUMINAMATH_CALUDE_dads_real_age_l2179_217946

theorem dads_real_age (reported_age : ℕ) (h : reported_age = 35) : 
  ∃ (real_age : ℕ), (5 : ℚ) / 7 * real_age = reported_age ∧ real_age = 49 := by
  sorry

end NUMINAMATH_CALUDE_dads_real_age_l2179_217946


namespace NUMINAMATH_CALUDE_count_non_adjacent_placements_correct_l2179_217965

/-- Represents an n × n grid board. -/
structure GridBoard where
  n : ℕ

/-- Counts the number of ways to place X and O on the grid such that they are not adjacent. -/
def countNonAdjacentPlacements (board : GridBoard) : ℕ :=
  board.n^4 - 3 * board.n^2 + 2 * board.n

/-- Theorem stating that countNonAdjacentPlacements gives the correct count. -/
theorem count_non_adjacent_placements_correct (board : GridBoard) :
  countNonAdjacentPlacements board =
    board.n^4 - 3 * board.n^2 + 2 * board.n :=
by sorry

end NUMINAMATH_CALUDE_count_non_adjacent_placements_correct_l2179_217965


namespace NUMINAMATH_CALUDE_snakes_count_l2179_217909

theorem snakes_count (breeding_balls : ℕ) (snakes_per_ball : ℕ) (snake_pairs : ℕ) : 
  breeding_balls * snakes_per_ball + 2 * snake_pairs = 36 :=
by
  sorry

#check snakes_count 3 8 6

end NUMINAMATH_CALUDE_snakes_count_l2179_217909


namespace NUMINAMATH_CALUDE_boat_stream_speed_ratio_l2179_217910

/-- Proves that if rowing against the stream takes twice as long as rowing with the stream,
    then the ratio of boat speed to stream speed is 3:1 -/
theorem boat_stream_speed_ratio
  (D : ℝ) -- Distance rowed
  (B : ℝ) -- Speed of the boat in still water
  (S : ℝ) -- Speed of the stream
  (hD : D > 0) -- Distance is positive
  (hB : B > 0) -- Boat speed is positive
  (hS : S > 0) -- Stream speed is positive
  (hBS : B > S) -- Boat is faster than the stream
  (h_time : D / (B - S) = 2 * (D / (B + S))) -- Time against stream is twice time with stream
  : B / S = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_stream_speed_ratio_l2179_217910


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l2179_217990

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℤ  -- First term
  d : ℤ   -- Common difference

/-- The nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.a₁ + (n - 1 : ℤ) * seq.d

theorem tenth_term_of_specific_sequence :
  ∃ (seq : ArithmeticSequence),
    seq.a₁ = 10 ∧
    nthTerm seq 2 = 7 ∧
    nthTerm seq 3 = 4 ∧
    nthTerm seq 10 = -17 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l2179_217990


namespace NUMINAMATH_CALUDE_range_of_a_l2179_217954

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - a| ≤ 2) ↔ a ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2179_217954


namespace NUMINAMATH_CALUDE_average_car_selections_l2179_217907

theorem average_car_selections (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ) : 
  num_cars = 15 → num_clients = 15 → selections_per_client = 3 →
  (num_clients * selections_per_client) / num_cars = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_car_selections_l2179_217907


namespace NUMINAMATH_CALUDE_maggie_red_packs_l2179_217978

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := sorry

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := 8

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := 160

theorem maggie_red_packs : red_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_maggie_red_packs_l2179_217978


namespace NUMINAMATH_CALUDE_expression_evaluation_l2179_217916

theorem expression_evaluation : -2^3 / (-2) + (-2)^2 * (-5) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2179_217916


namespace NUMINAMATH_CALUDE_student_meeting_probability_l2179_217914

/-- The probability of two students meeting given specific conditions -/
theorem student_meeting_probability (α : ℝ) (h : 0 < α ∧ α < 60) :
  let p := 1 - ((60 - α) / 60)^2
  0 ≤ p ∧ p ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_student_meeting_probability_l2179_217914


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l2179_217933

theorem shaded_region_perimeter (r : ℝ) (h : r = 7) :
  let circle_fraction : ℝ := 3 / 4
  let arc_length := circle_fraction * (2 * π * r)
  let radii_length := 2 * r
  radii_length + arc_length = 14 + (21 / 2) * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l2179_217933


namespace NUMINAMATH_CALUDE_intersection_A_B_l2179_217947

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {x | x^2 - 4 ≤ 0}

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2179_217947


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2179_217984

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (A : ℚ),
    team_size = 11 →
    captain_age = 26 →
    wicket_keeper_age_diff = 5 →
    (team_size : ℚ) * A - (captain_age + (captain_age + wicket_keeper_age_diff)) = 
      (team_size - 2 : ℚ) * (A - 1) →
    A = 24 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l2179_217984


namespace NUMINAMATH_CALUDE_history_book_cost_l2179_217991

/-- Given the following conditions:
  - Total number of books is 90
  - Math books cost $4 each
  - Total price of all books is $396
  - Number of math books bought is 54
  Prove that the cost of a history book is $5 -/
theorem history_book_cost (total_books : ℕ) (math_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 90 →
  math_book_cost = 4 →
  total_price = 396 →
  math_books = 54 →
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 :=
by sorry

end NUMINAMATH_CALUDE_history_book_cost_l2179_217991


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2179_217988

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence {a_n} where a_2 * a_3 = 5 and a_5 * a_6 = 10, prove that a_8 * a_9 = 20. -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_23 : a 2 * a 3 = 5) 
    (h_56 : a 5 * a 6 = 10) : 
  a 8 * a 9 = 20 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2179_217988


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l2179_217905

/-- Defines a circle equation passing through three points -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- Theorem stating that the circle equation passes through the given points -/
theorem circle_passes_through_points :
  CircleEquation 0 0 ∧ CircleEquation 4 0 ∧ CircleEquation (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l2179_217905


namespace NUMINAMATH_CALUDE_colored_paper_purchase_l2179_217922

theorem colored_paper_purchase (total_money : ℝ) (pencil_cost : ℝ) (paper_cost : ℝ) (pencils_bought : ℕ) :
  total_money = 10 →
  pencil_cost = 1.2 →
  paper_cost = 0.2 →
  pencils_bought = 5 →
  (total_money - pencil_cost * (pencils_bought : ℝ)) / paper_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_colored_paper_purchase_l2179_217922


namespace NUMINAMATH_CALUDE_base_2_representation_of_101_l2179_217913

theorem base_2_representation_of_101 : 
  ∃ (a b c d e f g : Nat), 
    (a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 0 ∧ e = 1 ∧ f = 0 ∧ g = 1) ∧
    101 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_101_l2179_217913


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_arithmetic_sequence_l2179_217956

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifth_term_of_specific_arithmetic_sequence :
  let a₁ : ℝ := 1
  let d : ℝ := 2
  arithmeticSequence a₁ d 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_arithmetic_sequence_l2179_217956


namespace NUMINAMATH_CALUDE_scramble_language_word_count_l2179_217967

/-- The number of letters in the Scramble alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum word length in the Scramble language -/
def max_word_length : ℕ := 5

/-- Calculates the number of words of a given length that contain at least one 'B' -/
def words_with_b (length : ℕ) : ℕ :=
  alphabet_size ^ length - (alphabet_size - 1) ^ length

/-- The total number of valid words in the Scramble language -/
def total_valid_words : ℕ :=
  words_with_b 1 + words_with_b 2 + words_with_b 3 + words_with_b 4 + words_with_b 5

theorem scramble_language_word_count :
  total_valid_words = 1863701 :=
by sorry

end NUMINAMATH_CALUDE_scramble_language_word_count_l2179_217967


namespace NUMINAMATH_CALUDE_richard_david_age_difference_l2179_217979

/-- Represents the ages of the three sons -/
structure Ages where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def family_conditions (ages : Ages) : Prop :=
  ages.richard > ages.david ∧
  ages.david = ages.scott + 8 ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- The theorem to prove -/
theorem richard_david_age_difference (ages : Ages) 
  (h : family_conditions ages) : ages.richard - ages.david = 6 := by
  sorry

end NUMINAMATH_CALUDE_richard_david_age_difference_l2179_217979


namespace NUMINAMATH_CALUDE_walking_distance_calculation_l2179_217923

-- Define the speeds and additional distances
def speed1_original : ℝ := 5
def speed1_alternative : ℝ := 15
def speed2_original : ℝ := 10
def speed2_alternative : ℝ := 20
def additional_distance1 : ℝ := 45
def additional_distance2 : ℝ := 30

-- Define the theorem
theorem walking_distance_calculation :
  ∃ (t1 t2 : ℝ),
    t1 > 0 ∧ t2 > 0 ∧
    (speed1_alternative * t1 - speed1_original * t1 = additional_distance1) ∧
    (speed2_alternative * t2 - speed2_original * t2 = additional_distance2) ∧
    (speed1_original * t1 = 22.5) ∧
    (speed2_original * t2 = 30) :=
  sorry

end NUMINAMATH_CALUDE_walking_distance_calculation_l2179_217923


namespace NUMINAMATH_CALUDE_angle_triple_supplement_l2179_217996

theorem angle_triple_supplement (x : ℝ) : 
  x = 3 * (180 - x) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_l2179_217996


namespace NUMINAMATH_CALUDE_units_digit_of_p_plus_five_l2179_217926

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

def has_positive_units_digit (n : ℕ) : Prop := n % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_p_plus_five (p : ℕ) 
  (h_even : is_positive_even p)
  (h_pos_digit : has_positive_units_digit p)
  (h_cube_square : units_digit (p^3) = units_digit (p^2)) :
  units_digit (p + 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_p_plus_five_l2179_217926


namespace NUMINAMATH_CALUDE_banana_distribution_l2179_217997

/-- The number of bananas each child would normally receive -/
def normal_bananas : ℕ := 2

/-- The number of absent children -/
def absent_children : ℕ := 330

/-- The number of extra bananas each child received due to absences -/
def extra_bananas : ℕ := 2

/-- The actual number of children in the school -/
def actual_children : ℕ := 660

theorem banana_distribution (total_bananas : ℕ) :
  (total_bananas = normal_bananas * actual_children) ∧
  (total_bananas = (normal_bananas + extra_bananas) * (actual_children - absent_children)) →
  actual_children = 660 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l2179_217997


namespace NUMINAMATH_CALUDE_set_A_is_correct_l2179_217935

-- Define the universe set U
def U : Set ℝ := {x | x > 0}

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := {x | 0 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | x ≥ 3}

-- Theorem statement
theorem set_A_is_correct : A = U \ complement_A_in_U := by sorry

end NUMINAMATH_CALUDE_set_A_is_correct_l2179_217935


namespace NUMINAMATH_CALUDE_mens_haircut_time_is_correct_l2179_217902

/-- The time it takes to cut a man's hair -/
def mens_haircut_time : ℕ := 15

/-- The time it takes to cut a woman's hair -/
def womens_haircut_time : ℕ := 50

/-- The time it takes to cut a kid's hair -/
def kids_haircut_time : ℕ := 25

/-- The number of women's haircuts Joe performed -/
def num_womens_haircuts : ℕ := 3

/-- The number of men's haircuts Joe performed -/
def num_mens_haircuts : ℕ := 2

/-- The number of kids' haircuts Joe performed -/
def num_kids_haircuts : ℕ := 3

/-- The total time Joe spent cutting hair -/
def total_time : ℕ := 255

theorem mens_haircut_time_is_correct :
  num_womens_haircuts * womens_haircut_time +
  num_mens_haircuts * mens_haircut_time +
  num_kids_haircuts * kids_haircut_time = total_time := by
sorry

end NUMINAMATH_CALUDE_mens_haircut_time_is_correct_l2179_217902


namespace NUMINAMATH_CALUDE_system_is_linear_l2179_217959

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and x and y are variables. -/
def IsLinearEquation (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, eq x y ↔ a * x + b * y = c

/-- A system of two linear equations is a pair of linear equations in two variables. -/
def IsSystemOfTwoLinearEquations (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  IsLinearEquation eq1 ∧ IsLinearEquation eq2

/-- The given system of equations. -/
def System : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) :=
  (fun x y ↦ x + y = 2, fun _ y ↦ y = 3)

theorem system_is_linear : IsSystemOfTwoLinearEquations System.1 System.2 := by
  sorry

#check system_is_linear

end NUMINAMATH_CALUDE_system_is_linear_l2179_217959


namespace NUMINAMATH_CALUDE_patrol_theorem_l2179_217993

/-- The number of streets patrolled by an officer in one hour -/
def streets_per_hour (streets : ℕ) (hours : ℕ) : ℚ := streets / hours

/-- The total number of streets patrolled by all officers in one hour -/
def total_streets_per_hour (rate_A rate_B rate_C : ℚ) : ℚ := rate_A + rate_B + rate_C

theorem patrol_theorem (a x b y c z : ℕ) 
  (h1 : streets_per_hour a x = 9/1)
  (h2 : streets_per_hour b y = 11/1)
  (h3 : streets_per_hour c z = 7/1) :
  total_streets_per_hour (streets_per_hour a x) (streets_per_hour b y) (streets_per_hour c z) = 27 := by
  sorry

end NUMINAMATH_CALUDE_patrol_theorem_l2179_217993


namespace NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l2179_217927

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- The slope of the tangent line at a point on the parabola -/
def m (a : ℝ) : ℝ := 2*(a - 1)

/-- The y-coordinate of the intersection point of tangent lines -/
def y_intersection (a b : ℝ) : ℝ := a*b - a - b + 2

theorem tangent_intersection_y_coordinate 
  (a b : ℝ) 
  (ha : f a = a^2 - 2*a - 3) 
  (hb : f b = b^2 - 2*b - 3) 
  (h_perp : m a * m b = -1) : 
  y_intersection a b = -1/4 := by
  sorry

#check tangent_intersection_y_coordinate

end NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l2179_217927


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2179_217962

def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 12*x^2 + 18*x - 22

theorem polynomial_remainder (x : ℝ) : 
  ∃ q : ℝ → ℝ, f x = (x - 4) * q x + 114 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2179_217962


namespace NUMINAMATH_CALUDE_min_value_expression_l2179_217929

theorem min_value_expression (a b c : ℕ+) :
  ∃ (x y z : ℕ+), 
    (⌊(8 * (x + y) : ℚ) / z⌋ + ⌊(8 * (x + z) : ℚ) / y⌋ + ⌊(8 * (y + z) : ℚ) / x⌋ = 46) ∧
    ∀ (a b c : ℕ+), 
      ⌊(8 * (a + b) : ℚ) / c⌋ + ⌊(8 * (a + c) : ℚ) / b⌋ + ⌊(8 * (b + c) : ℚ) / a⌋ ≥ 46 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2179_217929


namespace NUMINAMATH_CALUDE_circumcircle_fixed_point_l2179_217982

/-- A parabola that intersects the coordinate axes at three different points -/
structure AxisIntersectingParabola where
  a : ℝ
  b : ℝ
  x₁ : ℝ
  x₂ : ℝ
  intersectsAxes : x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ b ≠ 0
  onParabola₁ : 0 = x₁^2 + a * x₁ + b
  onParabola₂ : 0 = x₂^2 + a * x₂ + b
  onParabola₃ : b = 0^2 + a * 0 + b

/-- The circumcircle of a triangle formed by the intersection points of a parabola with the coordinate axes passes through (0, 1) -/
theorem circumcircle_fixed_point (p : AxisIntersectingParabola) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center.1 - 0)^2 + (center.2 - 1)^2 = radius^2 ∧
    (center.1 - p.x₁)^2 + center.2^2 = radius^2 ∧
    (center.1 - p.x₂)^2 + center.2^2 = radius^2 ∧
    center.1^2 + (center.2 - p.b)^2 = radius^2 :=
sorry

end NUMINAMATH_CALUDE_circumcircle_fixed_point_l2179_217982


namespace NUMINAMATH_CALUDE_burger_cost_is_100_cents_l2179_217973

/-- The cost of items in cents -/
structure ItemCosts where
  burger : ℕ
  soda : ℕ
  fries : ℕ

/-- Alice's purchase -/
def alice_purchase (costs : ItemCosts) : ℕ :=
  4 * costs.burger + 3 * costs.soda + costs.fries

/-- Bob's purchase -/
def bob_purchase (costs : ItemCosts) : ℕ :=
  3 * costs.burger + 2 * costs.soda + 2 * costs.fries

/-- Theorem stating that the cost of a burger is 100 cents -/
theorem burger_cost_is_100_cents :
  ∃ (costs : ItemCosts),
    alice_purchase costs = 540 ∧
    bob_purchase costs = 580 ∧
    costs.burger = 100 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_is_100_cents_l2179_217973


namespace NUMINAMATH_CALUDE_system_solution_unique_l2179_217970

theorem system_solution_unique :
  ∃! (x y : ℝ), 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2179_217970


namespace NUMINAMATH_CALUDE_combined_tax_rate_l2179_217994

theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.3)
  (h2 : mindy_rate = 0.2)
  (h3 : income_ratio = 3) :
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.225 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l2179_217994


namespace NUMINAMATH_CALUDE_sequence_equality_l2179_217976

theorem sequence_equality (C : ℝ) (a : ℕ → ℝ) 
  (hC : C > 1)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2)
  (h3 : ∀ m n : ℕ, m > 0 ∧ n > 0 → a (m * n) = a m * a n)
  (h4 : ∀ m n : ℕ, m > 0 ∧ n > 0 → a (m + n) ≤ C * (a m + a n))
  : ∀ n : ℕ, n > 0 → a n = n := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l2179_217976


namespace NUMINAMATH_CALUDE_not_always_fifteen_different_l2179_217955

/-- Represents a student with a t-shirt color and a pants color -/
structure Student :=
  (tshirt : Fin 15)
  (pants : Fin 15)

/-- The theorem stating that it's not always possible to find 15 students
    with all different t-shirt and pants colors -/
theorem not_always_fifteen_different (n : Nat) (h : n = 30) :
  ∃ (students : Finset Student),
    students.card = n ∧
    ∀ (subset : Finset Student),
      subset ⊆ students →
      subset.card = 15 →
      ∃ (s1 s2 : Student),
        s1 ∈ subset ∧ s2 ∈ subset ∧ s1 ≠ s2 ∧
        (s1.tshirt = s2.tshirt ∨ s1.pants = s2.pants) :=
by sorry

end NUMINAMATH_CALUDE_not_always_fifteen_different_l2179_217955


namespace NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_2_mod_25_l2179_217915

theorem largest_four_digit_negative_congruent_to_2_mod_25 :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n % 25 = 2 → n ≤ -1023 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_2_mod_25_l2179_217915


namespace NUMINAMATH_CALUDE_friends_team_assignment_l2179_217941

theorem friends_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k^n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l2179_217941


namespace NUMINAMATH_CALUDE_round_984530_to_nearest_ten_thousand_l2179_217936

-- Define a function to round to the nearest ten thousand
def roundToNearestTenThousand (n : ℤ) : ℤ :=
  (n + 5000) / 10000 * 10000

-- State the theorem
theorem round_984530_to_nearest_ten_thousand :
  roundToNearestTenThousand 984530 = 980000 := by
  sorry

end NUMINAMATH_CALUDE_round_984530_to_nearest_ten_thousand_l2179_217936


namespace NUMINAMATH_CALUDE_equation_solutions_count_l2179_217945

theorem equation_solutions_count :
  ∃! (solutions : Finset ℝ),
    Finset.card solutions = 8 ∧
    ∀ θ ∈ solutions,
      0 < θ ∧ θ ≤ 2 * Real.pi ∧
      2 - 4 * Real.sin θ + 6 * Real.cos (2 * θ) + Real.sin (3 * θ) = 0 ∧
    ∀ θ : ℝ,
      0 < θ ∧ θ ≤ 2 * Real.pi ∧
      2 - 4 * Real.sin θ + 6 * Real.cos (2 * θ) + Real.sin (3 * θ) = 0 →
      θ ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l2179_217945


namespace NUMINAMATH_CALUDE_triangular_square_l2179_217924

/-- Triangular numbers -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Main theorem -/
theorem triangular_square (m n : ℕ) (h : 2 * triangular m = triangular n) :
  triangular (2 * m - n) = (m - n) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangular_square_l2179_217924


namespace NUMINAMATH_CALUDE_jacket_cost_ratio_l2179_217992

theorem jacket_cost_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_ratio : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_ratio)
  let cost_ratio : ℝ := 2/3
  let cost : ℝ := selling_price * cost_ratio
  cost / marked_price = 1/2 := by
sorry

end NUMINAMATH_CALUDE_jacket_cost_ratio_l2179_217992
