import Mathlib

namespace NUMINAMATH_CALUDE_cos_pi_third_minus_2theta_l2178_217832

theorem cos_pi_third_minus_2theta (θ : ℝ) 
  (h : Real.sin (θ - π / 6) = Real.sqrt 3 / 3) : 
  Real.cos (π / 3 - 2 * θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_2theta_l2178_217832


namespace NUMINAMATH_CALUDE_sequence_properties_l2178_217829

def a (i : ℕ+) : ℕ := (7^(2^i.val) - 1) / 6

theorem sequence_properties :
  ∀ i : ℕ+,
    (∀ j : ℕ+, (a (j + 1)) % (a j) = 0) ∧
    (a i) % 3 ≠ 0 ∧
    (a i) % (2^(i.val + 2)) = 0 ∧
    (a i) % (2^(i.val + 3)) ≠ 0 ∧
    ∃ p : ℕ, ∃ n : ℕ, Prime p ∧ 6 * (a i) + 1 = p^n ∧
    ∃ x y : ℕ, a i = x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2178_217829


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2178_217896

theorem rational_equation_solution :
  ∃! x : ℚ, (x ≠ 2/3) ∧ (x ≠ -3) ∧
  ((7*x + 3) / (3*x^2 + 7*x - 6) = (5*x) / (3*x - 2)) ∧
  x = 1/5 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2178_217896


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l2178_217824

theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a / b = 2 / 5 →    -- Given ratio of legs
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r + s = c →        -- r and s are parts of hypotenuse
  r / s = 4 / 25 :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l2178_217824


namespace NUMINAMATH_CALUDE_chocolate_milk_syrup_amount_l2178_217839

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

end NUMINAMATH_CALUDE_chocolate_milk_syrup_amount_l2178_217839


namespace NUMINAMATH_CALUDE_min_detectors_for_cross_l2178_217845

/-- The size of the board --/
def boardSize : Nat := 5

/-- The number of cells in the cross pattern --/
def crossSize : Nat := 5

/-- The number of possible positions for the cross on the board --/
def possiblePositions : Nat := 3 * 3

/-- Function to calculate the number of possible detector states --/
def detectorStates (n : Nat) : Nat := 2^n

/-- Theorem stating the minimum number of detectors needed --/
theorem min_detectors_for_cross :
  ∃ (n : Nat), (n = 4) ∧ 
  (∀ (k : Nat), detectorStates k ≥ possiblePositions → k ≥ n) ∧
  (detectorStates n ≥ possiblePositions) := by
  sorry

end NUMINAMATH_CALUDE_min_detectors_for_cross_l2178_217845


namespace NUMINAMATH_CALUDE_triangle_acute_iff_sum_squares_gt_8R_squared_l2178_217890

theorem triangle_acute_iff_sum_squares_gt_8R_squared 
  (a b c R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R > 0)
  (h_R_def : 4 * R * (R - a) * (R - b) * (R - c) = a * b * c) :
  (∀ (A B C : ℝ), A + B + C = π → 
    0 < A ∧ A < π/2 ∧ 
    0 < B ∧ B < π/2 ∧ 
    0 < C ∧ C < π/2) ↔ 
  a^2 + b^2 + c^2 > 8 * R^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_acute_iff_sum_squares_gt_8R_squared_l2178_217890


namespace NUMINAMATH_CALUDE_smallest_possible_d_l2178_217887

theorem smallest_possible_d : ∃ (d : ℝ), d ≥ 0 ∧
  (∀ (d' : ℝ), d' ≥ 0 → (4 * Real.sqrt 3) ^ 2 + (d' - 2) ^ 2 = (4 * d') ^ 2 → d ≤ d') ∧
  (4 * Real.sqrt 3) ^ 2 + (d - 2) ^ 2 = (4 * d) ^ 2 ∧
  d = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l2178_217887


namespace NUMINAMATH_CALUDE_x_twelve_equals_one_l2178_217899

theorem x_twelve_equals_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelve_equals_one_l2178_217899


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_l2178_217874

theorem x_cubed_coefficient (p q : Polynomial ℤ) (hp : p = 3 * X ^ 4 - 2 * X ^ 3 + X ^ 2 - 3) 
  (hq : q = 2 * X ^ 2 + 5 * X - 4) : 
  (p * q).coeff 3 = 13 := by sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_l2178_217874


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_y_eq_one_seventh_l2178_217836

theorem matrix_not_invertible_iff_y_eq_one_seventh :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2 + y, 5; 4 - y, 9]
  ¬(IsUnit (Matrix.det A)) ↔ y = (1 : ℝ) / 7 :=
by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_y_eq_one_seventh_l2178_217836


namespace NUMINAMATH_CALUDE_victors_friend_wins_checkers_game_wins_l2178_217833

theorem victors_friend_wins (victor_wins : ℕ) (ratio_victor : ℕ) (ratio_friend : ℕ) : ℕ :=
  let friend_wins := (victor_wins * ratio_friend) / ratio_victor
  friend_wins

theorem checkers_game_wins : victors_friend_wins 36 9 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_victors_friend_wins_checkers_game_wins_l2178_217833


namespace NUMINAMATH_CALUDE_julia_watch_collection_l2178_217888

theorem julia_watch_collection :
  let silver_watches : ℕ := 20
  let bronze_watches : ℕ := 3 * silver_watches
  let platinum_watches : ℕ := 2 * bronze_watches
  let gold_watches : ℕ := (silver_watches + platinum_watches) / 5
  let total_watches : ℕ := silver_watches + bronze_watches + platinum_watches + gold_watches
  total_watches = 228 :=
by sorry

end NUMINAMATH_CALUDE_julia_watch_collection_l2178_217888


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2178_217862

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2178_217862


namespace NUMINAMATH_CALUDE_pokemon_cards_total_l2178_217893

theorem pokemon_cards_total (jason_left : ℕ) (jason_gave : ℕ) (lisa_left : ℕ) (lisa_gave : ℕ) :
  jason_left = 4 → jason_gave = 9 → lisa_left = 7 → lisa_gave = 15 →
  (jason_left + jason_gave) + (lisa_left + lisa_gave) = 35 :=
by sorry

end NUMINAMATH_CALUDE_pokemon_cards_total_l2178_217893


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2178_217822

theorem polynomial_expansion (x : ℝ) : 
  (5*x^2 + 3*x - 4) * (2*x^3 + x^2 - x + 1) = 
  10*x^5 + 11*x^4 - 10*x^3 - 2*x^2 + 7*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2178_217822


namespace NUMINAMATH_CALUDE_total_distance_walked_l2178_217867

-- Define the walking rate in miles per hour
def walking_rate : ℝ := 4

-- Define the total time in hours
def total_time : ℝ := 2

-- Define the break time in hours
def break_time : ℝ := 0.5

-- Define the effective walking time
def effective_walking_time : ℝ := total_time - break_time

-- Theorem to prove
theorem total_distance_walked :
  walking_rate * effective_walking_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l2178_217867


namespace NUMINAMATH_CALUDE_not_arithmetic_sequence_x_i_l2178_217877

/-- Given real constants a and b, a geometric sequence {c_i} with common ratio ≠ 1,
    and the line ax + by + c_i = 0 intersecting the parabola y^2 = 2px (p > 0)
    forming chords with midpoints M_i(x_i, y_i), prove that {x_i} cannot be an arithmetic sequence. -/
theorem not_arithmetic_sequence_x_i 
  (a b : ℝ) 
  (c : ℕ+ → ℝ) 
  (p : ℝ) 
  (hp : p > 0)
  (hc : ∃ (r : ℝ), r ≠ 1 ∧ ∀ (i : ℕ+), c (i + 1) = r * c i)
  (x y : ℕ+ → ℝ)
  (h_intersect : ∀ (i : ℕ+), ∃ (t : ℝ), a * t + b * y i + c i = 0 ∧ (y i)^2 = 2 * p * t)
  (h_midpoint : ∀ (i : ℕ+), ∃ (t₁ t₂ : ℝ), 
    a * t₁ + b * (y i) + c i = 0 ∧ (y i)^2 = 2 * p * t₁ ∧
    a * t₂ + b * (y i) + c i = 0 ∧ (y i)^2 = 2 * p * t₂ ∧
    x i = (t₁ + t₂) / 2 ∧ y i = (y i + y i) / 2) :
  ¬ (∃ (d : ℝ), ∀ (i : ℕ+), x (i + 1) - x i = d) :=
sorry

end NUMINAMATH_CALUDE_not_arithmetic_sequence_x_i_l2178_217877


namespace NUMINAMATH_CALUDE_guilty_pair_is_B_and_C_l2178_217823

/-- Represents the guilt status of a defendant -/
inductive GuiltStatus
| Guilty
| Innocent

/-- Represents a defendant -/
inductive Defendant
| A
| B
| C

/-- The guilt status of all defendants -/
def GuiltStatusSet := Defendant → GuiltStatus

/-- At least one of the defendants is guilty -/
def atLeastOneGuilty (gs : GuiltStatusSet) : Prop :=
  ∃ d : Defendant, gs d = GuiltStatus.Guilty

/-- If A is guilty and B is innocent, then C is innocent -/
def conditionalInnocence (gs : GuiltStatusSet) : Prop :=
  (gs Defendant.A = GuiltStatus.Guilty ∧ gs Defendant.B = GuiltStatus.Innocent) →
  gs Defendant.C = GuiltStatus.Innocent

/-- The main theorem stating that B and C are the two defendants such that one of them is definitely guilty -/
theorem guilty_pair_is_B_and_C :
  ∀ gs : GuiltStatusSet,
  atLeastOneGuilty gs →
  conditionalInnocence gs →
  (gs Defendant.B = GuiltStatus.Guilty ∨ gs Defendant.C = GuiltStatus.Guilty) :=
sorry

end NUMINAMATH_CALUDE_guilty_pair_is_B_and_C_l2178_217823


namespace NUMINAMATH_CALUDE_pet_store_heads_count_l2178_217847

theorem pet_store_heads_count :
  ∀ (num_dogs num_parakeets : ℕ),
    num_dogs = 9 →
    num_dogs * 4 + num_parakeets * 2 = 42 →
    num_dogs + num_parakeets = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_heads_count_l2178_217847


namespace NUMINAMATH_CALUDE_remaining_cube_height_l2178_217891

/-- The height of the remaining portion of a cube after cutting off a corner -/
theorem remaining_cube_height (cube_side : Real) (cut_distance : Real) : 
  cube_side = 2 → 
  cut_distance = 1 → 
  (cube_side - (Real.sqrt 3) / 3) = (5 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cube_height_l2178_217891


namespace NUMINAMATH_CALUDE_principal_amount_proof_l2178_217898

/-- Proves that for a principal amount P, with an interest rate of 5% per annum over 2 years,
    if the difference between compound interest and simple interest is 17, then P equals 6800. -/
theorem principal_amount_proof (P : ℝ) : 
  P * (1 + 0.05)^2 - P - (P * 0.05 * 2) = 17 → P = 6800 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l2178_217898


namespace NUMINAMATH_CALUDE_scale_length_difference_l2178_217876

/-- Proves that a 7 ft scale divided into 4 equal parts of 24 inches each has 12 additional inches -/
theorem scale_length_difference : 
  let scale_length_ft : ℕ := 7
  let num_parts : ℕ := 4
  let part_length_inches : ℕ := 24
  let inches_per_foot : ℕ := 12
  
  (num_parts * part_length_inches) - (scale_length_ft * inches_per_foot) = 12 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_difference_l2178_217876


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_expression_tight_l2178_217812

theorem min_value_expression (a b c : ℝ) (hb : b > 0) (hc : c > 0) (hbc : b > c) (ha : a ≠ 0) :
  ((2*a + b)^2 + (b - c)^2 + (c - 2*a)^2) / b^2 ≥ 4/3 :=
sorry

theorem min_value_expression_tight (a b c : ℝ) (hb : b > 0) (hc : c > 0) (hbc : b > c) (ha : a ≠ 0) :
  ∃ (ε : ℝ), ε > 0 ∧ ((2*a + b)^2 + (b - c)^2 + (c - 2*a)^2) / b^2 < 4/3 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_expression_tight_l2178_217812


namespace NUMINAMATH_CALUDE_sequence_sum_l2178_217803

theorem sequence_sum (A B C D E F G H I J : ℤ) : 
  D = 7 ∧ 
  A + B + C = 24 ∧ 
  B + C + D = 24 ∧ 
  C + D + E = 24 ∧ 
  D + E + F = 24 ∧ 
  E + F + G = 24 ∧ 
  F + G + H = 24 ∧ 
  G + H + I = 24 ∧ 
  H + I + J = 24 → 
  A + J = 105 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l2178_217803


namespace NUMINAMATH_CALUDE_equation_solution_range_l2178_217875

theorem equation_solution_range (b : ℝ) : 
  (∀ x : ℝ, x = -2 → x^2 - b*x - 5 = 5) →
  (∀ x : ℝ, x = -1 → x^2 - b*x - 5 = -1) →
  (∀ x : ℝ, x = 4 → x^2 - b*x - 5 = -1) →
  (∀ x : ℝ, x = 5 → x^2 - b*x - 5 = 5) →
  ∃ x y : ℝ, 
    (-2 < x ∧ x < -1 ∧ x^2 - b*x - 5 = 0) ∧
    (4 < y ∧ y < 5 ∧ y^2 - b*y - 5 = 0) ∧
    (∀ z : ℝ, z^2 - b*z - 5 = 0 → ((-2 < z ∧ z < -1) ∨ (4 < z ∧ z < 5))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2178_217875


namespace NUMINAMATH_CALUDE_trig_identity_l2178_217818

theorem trig_identity (x : Real) (h : Real.sin x - 2 * Real.cos x = 0) :
  2 * Real.sin x ^ 2 + Real.cos x ^ 2 + 1 = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2178_217818


namespace NUMINAMATH_CALUDE_marble_count_l2178_217837

-- Define the number of marbles for each person
def allison_marbles : ℕ := 28
def angela_marbles : ℕ := allison_marbles + 8
def albert_marbles : ℕ := 3 * angela_marbles
def addison_marbles : ℕ := 2 * albert_marbles

-- Define the total number of marbles
def total_marbles : ℕ := allison_marbles + angela_marbles + albert_marbles + addison_marbles

-- Theorem to prove
theorem marble_count : total_marbles = 388 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l2178_217837


namespace NUMINAMATH_CALUDE_fraction_simplification_l2178_217852

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 4 / 6) (hy : y = 8 / 10) : 
  (6 * x + 8 * y) / (48 * x * y) = 13 / 32 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2178_217852


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2178_217853

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 2 + a 3 = 6) : 
  3 * a 4 + a 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2178_217853


namespace NUMINAMATH_CALUDE_library_book_loan_l2178_217858

theorem library_book_loan (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) : 
  initial_books = 75 → 
  return_rate = 4/5 → 
  final_books = 64 → 
  (initial_books : ℚ) - final_books = (1 - return_rate) * 55 := by
  sorry

end NUMINAMATH_CALUDE_library_book_loan_l2178_217858


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2178_217838

/-- The range of k values for which the line y = kx + 2 intersects the ellipse 2x^2 + 3y^2 = 6 at two distinct points -/
theorem line_ellipse_intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + 2 ∧ y₂ = k * x₂ + 2 ∧
    2 * x₁^2 + 3 * y₁^2 = 6 ∧ 
    2 * x₂^2 + 3 * y₂^2 = 6) ↔ 
  k < -Real.sqrt (2/3) ∨ k > Real.sqrt (2/3) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2178_217838


namespace NUMINAMATH_CALUDE_accidental_calculation_l2178_217884

theorem accidental_calculation (x : ℝ) : (x + 12) / 8 = 8 → (x - 12) * 9 = 360 := by
  sorry

end NUMINAMATH_CALUDE_accidental_calculation_l2178_217884


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l2178_217831

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_t_value :
  ∀ t : ℝ, 
  let a : ℝ × ℝ := (1, t)
  let b : ℝ × ℝ := (t, 9)
  parallel a b → t = 3 ∨ t = -3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l2178_217831


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l2178_217855

theorem triangle_angle_inequality (X Y Z : ℝ) 
  (h_positive : X > 0 ∧ Y > 0 ∧ Z > 0) 
  (h_sum : 2 * X + 2 * Y + 2 * Z = π) : 
  (Real.sin X / Real.cos (Y - Z)) + 
  (Real.sin Y / Real.cos (Z - X)) + 
  (Real.sin Z / Real.cos (X - Y)) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l2178_217855


namespace NUMINAMATH_CALUDE_place_value_difference_power_l2178_217801

/-- Given a natural number, returns the count of a specific digit in it. -/
def countDigit (n : ℕ) (digit : ℕ) : ℕ := sorry

/-- Given a natural number, returns a list of place values for specific digits. -/
def getPlaceValues (n : ℕ) (digits : List ℕ) : List ℕ := sorry

/-- Calculates the sum of differences between consecutive place values. -/
def sumOfDifferences (placeValues : List ℕ) : ℕ := sorry

/-- The main theorem to prove. -/
theorem place_value_difference_power (n : ℕ) (h : n = 58219435) :
  let placeValues := getPlaceValues n [1, 5, 8]
  let diffSum := sumOfDifferences placeValues
  let numTwos := countDigit n 2
  diffSum ^ numTwos = 420950000 := by sorry

end NUMINAMATH_CALUDE_place_value_difference_power_l2178_217801


namespace NUMINAMATH_CALUDE_difference_of_squares_312_308_l2178_217879

theorem difference_of_squares_312_308 : 312^2 - 308^2 = 2480 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_312_308_l2178_217879


namespace NUMINAMATH_CALUDE_probability_of_winning_reward_l2178_217846

/-- The number of different types of blessing cards -/
def num_card_types : ℕ := 3

/-- The number of red envelopes Xiao Ming has -/
def num_envelopes : ℕ := 4

/-- The probability of winning the reward -/
def win_probability : ℚ := 4/9

/-- Theorem stating the probability of winning the reward -/
theorem probability_of_winning_reward :
  (num_card_types = 3) →
  (num_envelopes = 4) →
  (win_probability = 4/9) := by
  sorry

end NUMINAMATH_CALUDE_probability_of_winning_reward_l2178_217846


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l2178_217835

theorem ratio_of_numbers (a b : ℕ) (h1 : a > b) (h2 : a + b = 96) (h3 : a = 64) (h4 : b = 32) : a / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l2178_217835


namespace NUMINAMATH_CALUDE_valid_paths_count_l2178_217805

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Represents a vertical line segment -/
structure VerticalSegment where
  x : Nat
  y_start : Nat
  y_end : Nat

/-- Definition of the grid and forbidden segments -/
def grid_height : Nat := 5
def grid_width : Nat := 8
def forbidden_segment1 : VerticalSegment := { x := 3, y_start := 1, y_end := 3 }
def forbidden_segment2 : VerticalSegment := { x := 4, y_start := 2, y_end := 5 }

/-- Function to calculate the number of valid paths -/
def count_valid_paths (height width : Nat) (forbidden1 forbidden2 : VerticalSegment) : Nat :=
  sorry

/-- Theorem stating the number of valid paths -/
theorem valid_paths_count :
  count_valid_paths grid_height grid_width forbidden_segment1 forbidden_segment2 = 838 := by
  sorry

end NUMINAMATH_CALUDE_valid_paths_count_l2178_217805


namespace NUMINAMATH_CALUDE_solution_set_l2178_217809

theorem solution_set (x y z : Real) : 
  x + y + z = Real.pi ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  ((x = Real.pi/6 ∧ y = Real.pi/3 ∧ z = Real.pi/2) ∨
   (x = Real.pi ∧ y = 0 ∧ z = 0) ∨
   (x = 0 ∧ y = Real.pi ∧ z = 0) ∨
   (x = 0 ∧ y = 0 ∧ z = Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l2178_217809


namespace NUMINAMATH_CALUDE_turkey_weight_ratio_l2178_217802

/-- The weight of the first turkey in kilograms -/
def first_turkey_weight : ℝ := 6

/-- The weight of the second turkey in kilograms -/
def second_turkey_weight : ℝ := 9

/-- The cost of turkey per kilogram in dollars -/
def cost_per_kg : ℝ := 2

/-- The total amount spent on all turkeys in dollars -/
def total_spent : ℝ := 66

/-- The number of turkeys bought -/
def num_turkeys : ℕ := 3

theorem turkey_weight_ratio :
  let total_weight := total_spent / cost_per_kg
  let third_turkey_weight := total_weight - (first_turkey_weight + second_turkey_weight)
  third_turkey_weight / second_turkey_weight = 2 := by
sorry

end NUMINAMATH_CALUDE_turkey_weight_ratio_l2178_217802


namespace NUMINAMATH_CALUDE_disprove_propositions_l2178_217866

open Set

/-- Definition of an M point -/
def is_M_point (f : ℝ → ℝ) (c : ℝ) (a b : ℝ) : Prop :=
  ∃ I : Set ℝ, IsOpen I ∧ c ∈ I ∩ Icc a b ∧
  ∀ x ∈ I ∩ Icc a b, x ≠ c → f x < f c

/-- Main theorem stating the existence of a function that disproves both propositions -/
theorem disprove_propositions : ∃ f : ℝ → ℝ,
  (∃ a b x₀ : ℝ, x₀ ∈ Icc a b ∧ 
    (∀ x ∈ Icc a b, f x ≤ f x₀) ∧ 
    ¬is_M_point f x₀ a b) ∧
  (∀ a b : ℝ, a < b → is_M_point f b a b) ∧
  ¬StrictMono f :=
sorry

end NUMINAMATH_CALUDE_disprove_propositions_l2178_217866


namespace NUMINAMATH_CALUDE_earth_land_area_scientific_notation_l2178_217804

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a given number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The land area of the Earth in km² -/
def earthLandArea : ℝ := 149000000

/-- The number of significant figures to retain -/
def sigFiguresRequired : ℕ := 3

theorem earth_land_area_scientific_notation :
  toScientificNotation earthLandArea sigFiguresRequired =
    ScientificNotation.mk 1.49 8 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_earth_land_area_scientific_notation_l2178_217804


namespace NUMINAMATH_CALUDE_min_value_implies_m_range_l2178_217851

-- Define the piecewise function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (x - m)^2 - 2 else 2*x^3 - 3*x^2

-- State the theorem
theorem min_value_implies_m_range (m : ℝ) :
  (∀ x, f m x ≥ -1) ∧ (∃ x, f m x = -1) → m ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_implies_m_range_l2178_217851


namespace NUMINAMATH_CALUDE_cubic_roots_cube_l2178_217870

theorem cubic_roots_cube (u v w : ℂ) :
  (u^3 + v^3 + w^3 = 54) →
  (u^3 * v^3 + v^3 * w^3 + w^3 * u^3 = -89) →
  (u^3 * v^3 * w^3 = 27) →
  (u + v + w = 5) →
  (u * v + v * w + w * u = 4) →
  (u * v * w = 3) →
  (u^3 - 5 * u^2 + 4 * u - 3 = 0) →
  (v^3 - 5 * v^2 + 4 * v - 3 = 0) →
  (w^3 - 5 * w^2 + 4 * w - 3 = 0) →
  ∀ (x : ℂ), x^3 - 54 * x^2 - 89 * x - 27 = 0 ↔ (x = u^3 ∨ x = v^3 ∨ x = w^3) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_cube_l2178_217870


namespace NUMINAMATH_CALUDE_adjacent_probability_four_people_l2178_217895

def num_people : ℕ := 4

def total_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

def favorable_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 2)

def probability_adjacent (n : ℕ) : ℚ :=
  (favorable_arrangements n : ℚ) / (total_arrangements n : ℚ)

theorem adjacent_probability_four_people :
  probability_adjacent num_people = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_probability_four_people_l2178_217895


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2178_217842

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 6 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 6 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 6 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -32/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2178_217842


namespace NUMINAMATH_CALUDE_integral_equals_ln_80_over_23_l2178_217869

open Real MeasureTheory

theorem integral_equals_ln_80_over_23 :
  ∫ x in (1 : ℝ)..2, (9 * x + 4) / (x^5 + 3 * x^2 + x) = Real.log (80 / 23) := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_ln_80_over_23_l2178_217869


namespace NUMINAMATH_CALUDE_equation_root_l2178_217828

theorem equation_root : ∃ x : ℝ, (2 / x = 3 / (x + 1)) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_l2178_217828


namespace NUMINAMATH_CALUDE_runner_a_race_time_l2178_217840

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

end NUMINAMATH_CALUDE_runner_a_race_time_l2178_217840


namespace NUMINAMATH_CALUDE_al_ben_weight_difference_l2178_217830

theorem al_ben_weight_difference :
  ∀ (al_weight ben_weight carl_weight : ℕ),
    ben_weight = carl_weight - 16 →
    al_weight = 146 + 38 →
    carl_weight = 175 →
    al_weight - ben_weight = 25 := by
  sorry

end NUMINAMATH_CALUDE_al_ben_weight_difference_l2178_217830


namespace NUMINAMATH_CALUDE_inequality_proof_l2178_217841

theorem inequality_proof (a b c : ℝ) : a^2 + b^2 + c^2 + 4 ≥ a*b + 3*b + 2*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2178_217841


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l2178_217894

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 3 → 
  (10 * x + y) - (10 * y + x) = 27 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l2178_217894


namespace NUMINAMATH_CALUDE_january_salary_is_2900_l2178_217872

/-- Calculates the salary for January given the average salaries and May's salary -/
def january_salary (avg_jan_to_apr avg_feb_to_may may_salary : ℚ) : ℚ :=
  4 * avg_jan_to_apr - (4 * avg_feb_to_may - may_salary)

/-- Theorem stating that the salary for January is 2900 given the provided conditions -/
theorem january_salary_is_2900 :
  january_salary 8000 8900 6500 = 2900 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_is_2900_l2178_217872


namespace NUMINAMATH_CALUDE_circles_intersect_l2178_217807

def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

theorem circles_intersect : ∃ (x y : ℝ), circle_C1 x y ∧ circle_C2 x y := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l2178_217807


namespace NUMINAMATH_CALUDE_quadratic_domain_range_existence_l2178_217863

/-- 
Given a quadratic function f(x) = -1/2 * x^2 + x + a, where a is a constant,
there exist real numbers m and n (with m < n) such that the domain of f is [m, n]
and the range is [3m, 3n] if and only if -2 < a ≤ 5/2.
-/
theorem quadratic_domain_range_existence (a : ℝ) :
  (∃ (m n : ℝ), m < n ∧
    (∀ x, x ∈ Set.Icc m n ↔ -1/2 * x^2 + x + a ∈ Set.Icc (3*m) (3*n)) ∧
    (∀ y, y ∈ Set.Icc (3*m) (3*n) → ∃ x ∈ Set.Icc m n, y = -1/2 * x^2 + x + a)) ↔
  -2 < a ∧ a ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_domain_range_existence_l2178_217863


namespace NUMINAMATH_CALUDE_train_speed_problem_l2178_217827

theorem train_speed_problem (length1 length2 speed1 time : ℝ) 
  (h1 : length1 = 210)
  (h2 : length2 = 260)
  (h3 : speed1 = 40)
  (h4 : time = 16.918646508279338)
  (h5 : length1 > 0)
  (h6 : length2 > 0)
  (h7 : speed1 > 0)
  (h8 : time > 0) :
  ∃ speed2 : ℝ, 
    speed2 > 0 ∧ 
    (length1 + length2) / 1000 = (speed1 + speed2) * (time / 3600) ∧
    speed2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2178_217827


namespace NUMINAMATH_CALUDE_inequality_range_l2178_217817

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (1 + a) * x^2 + a * x + a < x^2 + 1) → 
  a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2178_217817


namespace NUMINAMATH_CALUDE_genetic_material_distribution_l2178_217848

/-- Represents a cell in a diploid organism -/
structure DiploidCell where
  genetic_material : Set (α : Type)
  cytoplasm : Set (α : Type)

/-- Represents the process of cell division -/
def cell_division (parent : DiploidCell) : DiploidCell × DiploidCell :=
  sorry

/-- Predicate to check if the distribution is random and unequal -/
def is_random_unequal_distribution (parent : DiploidCell) (daughter1 daughter2 : DiploidCell) : Prop :=
  sorry

/-- Theorem stating that genetic material in cytoplasm is distributed randomly and unequally during cell division -/
theorem genetic_material_distribution (cell : DiploidCell) :
  let (daughter1, daughter2) := cell_division cell
  is_random_unequal_distribution cell daughter1 daughter2 := by
  sorry

end NUMINAMATH_CALUDE_genetic_material_distribution_l2178_217848


namespace NUMINAMATH_CALUDE_scale_model_height_l2178_217808

/-- The scale ratio of the model to the actual skyscraper -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the skyscraper in feet -/
def actual_height : ℕ := 1250

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The height of the scale model in inches -/
def model_height_inches : ℕ := 600

/-- Theorem stating that the height of the scale model in inches is 600 -/
theorem scale_model_height :
  (actual_height : ℚ) * scale_ratio * inches_per_foot = model_height_inches := by
  sorry

end NUMINAMATH_CALUDE_scale_model_height_l2178_217808


namespace NUMINAMATH_CALUDE_unique_solution_is_four_l2178_217886

theorem unique_solution_is_four :
  ∃! x : ℝ, 2 * x + 20 = 8 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_four_l2178_217886


namespace NUMINAMATH_CALUDE_vendor_first_day_sale_percentage_l2178_217857

/-- Represents the percentage of apples sold on the first day -/
def first_day_sale_percentage : ℝ := sorry

/-- Represents the total number of apples initially -/
def total_apples : ℝ := sorry

/-- Represents the number of apples remaining after the first day's sale -/
def apples_after_first_sale : ℝ := total_apples * (1 - first_day_sale_percentage)

/-- Represents the number of apples thrown away on the first day -/
def apples_thrown_first_day : ℝ := 0.2 * apples_after_first_sale

/-- Represents the number of apples remaining after throwing away on the first day -/
def apples_remaining_first_day : ℝ := apples_after_first_sale - apples_thrown_first_day

/-- Represents the number of apples sold on the second day -/
def apples_sold_second_day : ℝ := 0.5 * apples_remaining_first_day

/-- Represents the number of apples thrown away on the second day -/
def apples_thrown_second_day : ℝ := apples_remaining_first_day - apples_sold_second_day

/-- Represents the total number of apples thrown away -/
def total_apples_thrown : ℝ := apples_thrown_first_day + apples_thrown_second_day

theorem vendor_first_day_sale_percentage :
  first_day_sale_percentage = 0.5 ∧
  total_apples_thrown = 0.3 * total_apples :=
sorry

end NUMINAMATH_CALUDE_vendor_first_day_sale_percentage_l2178_217857


namespace NUMINAMATH_CALUDE_cyclic_sum_nonnegative_l2178_217861

theorem cyclic_sum_nonnegative (a b c k : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hk_lower : k ≥ 0) (hk_upper : k < 2) : 
  (a^2 - b*c)/(b^2 + c^2 + k*a^2) + 
  (b^2 - a*c)/(a^2 + c^2 + k*b^2) + 
  (c^2 - a*b)/(a^2 + b^2 + k*c^2) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_nonnegative_l2178_217861


namespace NUMINAMATH_CALUDE_dog_roaming_area_l2178_217815

/-- The area a dog can roam when tied to a circular pillar -/
theorem dog_roaming_area (leash_length : ℝ) (pillar_radius : ℝ) (roaming_area : ℝ) : 
  leash_length = 10 →
  pillar_radius = 2 →
  roaming_area = π * (leash_length + pillar_radius)^2 →
  roaming_area = 144 * π :=
by sorry

end NUMINAMATH_CALUDE_dog_roaming_area_l2178_217815


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2178_217897

theorem complex_magnitude_problem (t : ℝ) (h : t > 0) :
  Complex.abs (Complex.mk (-3) t) = 5 * Real.sqrt 5 → t = 2 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2178_217897


namespace NUMINAMATH_CALUDE_candy_sold_tuesday_l2178_217825

/-- Theorem: Candy sold on Tuesday --/
theorem candy_sold_tuesday (initial_candy : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) :
  initial_candy = 80 →
  sold_monday = 15 →
  remaining_wednesday = 7 →
  initial_candy - sold_monday - remaining_wednesday = 58 := by
  sorry

end NUMINAMATH_CALUDE_candy_sold_tuesday_l2178_217825


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2178_217849

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -3/2 ∧ 2*x₁^2 - 4*x₁ = 6 - 3*x₁) ∧
  (x₂ = 2 ∧ 2*x₂^2 - 4*x₂ = 6 - 3*x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2178_217849


namespace NUMINAMATH_CALUDE_bahs_to_yahs_conversion_l2178_217811

/-- The number of bahs in one rah -/
def bahs_per_rah : ℚ := 18 / 30

/-- The number of yahs in one rah -/
def yahs_per_rah : ℚ := 10 / 6

/-- Proves that 432 bahs are equal to 1200 yahs -/
theorem bahs_to_yahs_conversion : 
  432 * bahs_per_rah = 1200 / yahs_per_rah := by sorry

end NUMINAMATH_CALUDE_bahs_to_yahs_conversion_l2178_217811


namespace NUMINAMATH_CALUDE_dog_bones_problem_l2178_217854

theorem dog_bones_problem (initial_bones final_bones : ℕ) 
  (h1 : initial_bones = 493)
  (h2 : final_bones = 860) :
  final_bones - initial_bones = 367 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_problem_l2178_217854


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_sixth_l2178_217860

theorem sin_alpha_minus_pi_sixth (α : Real) 
  (h : Real.cos (α - π/3) - Real.cos α = 1/3) : 
  Real.sin (α - π/6) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_sixth_l2178_217860


namespace NUMINAMATH_CALUDE_shelter_dogs_count_l2178_217871

/-- Given an animal shelter with dogs and cats, prove the number of dogs. -/
theorem shelter_dogs_count (d c : ℕ) : 
  d * 7 = c * 15 →  -- Initial ratio of dogs to cats is 15:7
  d * 11 = (c + 16) * 15 →  -- Ratio after adding 16 cats is 15:11
  d = 60 :=  -- The number of dogs is 60
by sorry

end NUMINAMATH_CALUDE_shelter_dogs_count_l2178_217871


namespace NUMINAMATH_CALUDE_parabola_coefficient_l2178_217892

/-- Given a parabola y = ax^2 + bx + c with vertex (q, 2q) and y-intercept (0, -2q), where q ≠ 0, 
    the value of b is 8/q. -/
theorem parabola_coefficient (a b c q : ℝ) (hq : q ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (2 * q = a * q^2 + b * q + c) →
  (-2 * q = c) →
  b = 8 / q := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l2178_217892


namespace NUMINAMATH_CALUDE_cyclist_speed_is_25_l2178_217821

/-- The speed of the motorcyclist in km/h -/
def V_M : ℝ := 50

/-- The system of equations for the cyclist and motorcyclist problem -/
def equations (x y : ℝ) : Prop :=
  (20 / x - 20 / V_M = y) ∧ (70 - 8 / 3 * x = V_M * (7 / 15 - y))

/-- Theorem stating that x = 25 km/h satisfies the system of equations for some y -/
theorem cyclist_speed_is_25 : ∃ y : ℝ, equations 25 y := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_is_25_l2178_217821


namespace NUMINAMATH_CALUDE_sin_angle_A_is_sqrt3_div_2_l2178_217873

/-- An isosceles trapezoid with specific measurements -/
structure IsoscelesTrapezoid where
  -- Side lengths
  AB : ℝ
  CD : ℝ
  AD : ℝ
  -- Angle A in radians
  angleA : ℝ
  -- Conditions
  isIsosceles : AD = AD  -- AD = BC
  isParallel : AB < CD  -- AB parallel to CD implies AB < CD
  angleValue : angleA = 2 * Real.pi / 3  -- 120° in radians
  sideAB : AB = 160
  sideCD : CD = 240
  perimeter : AB + CD + 2 * AD = 800

/-- The sine of angle A in the isosceles trapezoid is √3/2 -/
theorem sin_angle_A_is_sqrt3_div_2 (t : IsoscelesTrapezoid) : Real.sin t.angleA = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_angle_A_is_sqrt3_div_2_l2178_217873


namespace NUMINAMATH_CALUDE_inequality_one_min_value_min_point_l2178_217881

-- Define the variables and conditions
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hab : a + b = 4)

-- Theorem 1
theorem inequality_one : 1/a + 1/(b+1) ≥ 4/5 := by sorry

-- Theorem 2
theorem min_value : ∃ (min_val : ℝ), min_val = (1 + Real.sqrt 5) / 2 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 4 → 4/(x*y) + x/y ≥ min_val := by sorry

-- Theorem for the values of a and b at the minimum point
theorem min_point : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 4 ∧
  4/(a*b) + a/b = (1 + Real.sqrt 5) / 2 ∧
  a = Real.sqrt 5 - 1 ∧ b = 5 - Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_inequality_one_min_value_min_point_l2178_217881


namespace NUMINAMATH_CALUDE_ratio_squares_l2178_217814

theorem ratio_squares (a b c d : ℤ) 
  (h1 : b * c + a * d = 1)
  (h2 : a * c + 2 * b * d = 1) :
  (a^2 + c^2 : ℚ) / (b^2 + d^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_ratio_squares_l2178_217814


namespace NUMINAMATH_CALUDE_no_month_with_five_mondays_and_thursdays_l2178_217843

/-- Represents the possible number of days in a month -/
inductive MonthDays : Type where
  | days28 : MonthDays
  | days29 : MonthDays
  | days30 : MonthDays
  | days31 : MonthDays

/-- Converts MonthDays to a natural number -/
def monthDaysToNat (md : MonthDays) : Nat :=
  match md with
  | MonthDays.days28 => 28
  | MonthDays.days29 => 29
  | MonthDays.days30 => 30
  | MonthDays.days31 => 31

/-- Represents a day of the week -/
inductive Weekday : Type where
  | monday : Weekday
  | tuesday : Weekday
  | wednesday : Weekday
  | thursday : Weekday
  | friday : Weekday
  | saturday : Weekday
  | sunday : Weekday

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- Counts the number of occurrences of a specific weekday in a month -/
def countWeekday (startDay : Weekday) (monthLength : MonthDays) (day : Weekday) : Nat :=
  sorry  -- Implementation details omitted

theorem no_month_with_five_mondays_and_thursdays :
  ∀ (md : MonthDays) (start : Weekday),
    ¬(countWeekday start md Weekday.monday = 5 ∧ countWeekday start md Weekday.thursday = 5) :=
by sorry


end NUMINAMATH_CALUDE_no_month_with_five_mondays_and_thursdays_l2178_217843


namespace NUMINAMATH_CALUDE_students_liking_neither_l2178_217883

theorem students_liking_neither (total : ℕ) (chinese : ℕ) (math : ℕ) (both : ℕ)
  (h_total : total = 62)
  (h_chinese : chinese = 37)
  (h_math : math = 49)
  (h_both : both = 30) :
  total - (chinese + math - both) = 6 :=
by sorry

end NUMINAMATH_CALUDE_students_liking_neither_l2178_217883


namespace NUMINAMATH_CALUDE_base_8_first_digit_350_l2178_217882

def base_8_first_digit (n : ℕ) : ℕ :=
  (n / 64) % 8

theorem base_8_first_digit_350 :
  base_8_first_digit 350 = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_8_first_digit_350_l2178_217882


namespace NUMINAMATH_CALUDE_train_ticket_types_l2178_217850

/-- The number of ticket types needed for a train route -/
def ticket_types (stops_between : ℕ) : ℕ :=
  let total_stops := stops_between + 2
  total_stops * (total_stops - 1)

/-- Theorem: For a train route with 3 stops between two end cities, 
    the number of ticket types needed is 20 -/
theorem train_ticket_types : ticket_types 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_ticket_types_l2178_217850


namespace NUMINAMATH_CALUDE_binomial_multiplication_l2178_217880

theorem binomial_multiplication (x : ℝ) : (4 * x - 3) * (x + 7) = 4 * x^2 + 25 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_multiplication_l2178_217880


namespace NUMINAMATH_CALUDE_factor_sum_l2178_217813

theorem factor_sum (P Q : ℤ) : 
  (∃ b c : ℤ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2178_217813


namespace NUMINAMATH_CALUDE_half_oz_mixture_bubbles_l2178_217834

/-- The number of bubbles that can be made from one ounce of Dawn liquid soap -/
def dawn_bubbles_per_oz : ℕ := 200000

/-- The number of bubbles that can be made from one ounce of Dr. Bronner's liquid soap -/
def bronner_bubbles_per_oz : ℕ := 2 * dawn_bubbles_per_oz

/-- The number of bubbles that can be made from one ounce of an equal mixture of Dawn and Dr. Bronner's liquid soaps -/
def mixture_bubbles_per_oz : ℕ := (dawn_bubbles_per_oz + bronner_bubbles_per_oz) / 2

/-- Theorem: One half ounce of an equal mixture of Dawn and Dr. Bronner's liquid soaps can make 150,000 bubbles -/
theorem half_oz_mixture_bubbles : mixture_bubbles_per_oz / 2 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_half_oz_mixture_bubbles_l2178_217834


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l2178_217810

-- Define the parabola and hyperbola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def hyperbola (x y : ℝ) : Prop := x^2/7 - y^2/9 = 1

-- Define the focus of the hyperbola
def hyperbola_focus : ℝ × ℝ := (4, 0)

-- Define the parameter p of the parabola
def p : ℝ := 8

-- Define point K
def K : ℝ × ℝ := (-4, 0)

-- Define the relationship between |AK| and |AF|
def AK_AF_relation (A F K : ℝ × ℝ) : Prop :=
  (A.1 - K.1)^2 + (A.2 - K.2)^2 = 2 * ((A.1 - F.1)^2 + (A.2 - F.2)^2)

theorem parabola_triangle_area 
  (A : ℝ × ℝ)
  (h1 : parabola p A.1 A.2)
  (h2 : AK_AF_relation A hyperbola_focus K) :
  (1/2) * ((K.1 - hyperbola_focus.1)^2 + (K.2 - hyperbola_focus.2)^2) = 32 :=
sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l2178_217810


namespace NUMINAMATH_CALUDE_parentheses_removal_correctness_l2178_217826

theorem parentheses_removal_correctness :
  let A := (x y : ℝ) → 5*x - (x - 2*y) = 5*x - x + 2*y
  let B := (a b : ℝ) → 2*a^2 + (3*a - b) = 2*a^2 + 3*a - b
  let C := (x y : ℝ) → (x - 2*y) - (x^2 - y^2) = x - 2*y - x^2 + y^2
  let D := (x : ℝ) → 3*x^2 - 3*(x + 6) = 3*x^2 - 3*x - 6
  A ∧ B ∧ C ∧ ¬D := by sorry

end NUMINAMATH_CALUDE_parentheses_removal_correctness_l2178_217826


namespace NUMINAMATH_CALUDE_max_angle_at_tangent_points_l2178_217859

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Checks if a point is strictly inside a circle -/
def is_inside_circle (p : Point) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

/-- Checks if a point is on a circle -/
def is_on_circle (p : Point) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) = c.radius

/-- Calculates the angle ABC given three points A, B, and C -/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- Defines the tangent points of circles passing through two points and tangent to a given circle -/
noncomputable def tangent_points (A B : Point) (Ω : Circle) : Point × Point := sorry

theorem max_angle_at_tangent_points (Ω : Circle) (A B : Point) :
  is_inside_circle A Ω →
  is_inside_circle B Ω →
  A ≠ B →
  let (C₁, C₂) := tangent_points A B Ω
  ∀ C : Point, is_on_circle C Ω →
    angle A C B ≤ max (angle A C₁ B) (angle A C₂ B) :=
by sorry

end NUMINAMATH_CALUDE_max_angle_at_tangent_points_l2178_217859


namespace NUMINAMATH_CALUDE_min_value_expression_l2178_217856

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a > b) (hbc : b > c) : 
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2178_217856


namespace NUMINAMATH_CALUDE_translation_right_3_units_l2178_217819

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateRight (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

theorem translation_right_3_units :
  let A : Point := { x := 2, y := -1 }
  let A' : Point := translateRight A 3
  A'.x = 5 ∧ A'.y = -1 := by
sorry

end NUMINAMATH_CALUDE_translation_right_3_units_l2178_217819


namespace NUMINAMATH_CALUDE_sailing_speed_calculation_l2178_217889

/-- The sailing speed of a ship in still water, given the following conditions:
  * Two ships (Knight and Warrior) depart from ports A and B at 8 a.m.
  * They travel towards each other, turn around at the opposite port, and return to their starting points.
  * Both ships return to their starting points at 10 a.m.
  * The time it takes for the ships to travel in the same direction is 10 minutes.
  * The speed of the current is 0.5 meters per second.
-/
def sailing_speed : ℝ := 6

/-- The speed of the current in meters per second. -/
def current_speed : ℝ := 0.5

/-- The time it takes for the ships to travel in the same direction, in seconds. -/
def same_direction_time : ℝ := 600

/-- The total travel time for each ship, in seconds. -/
def total_travel_time : ℝ := 7200

theorem sailing_speed_calculation :
  let v := sailing_speed
  let c := current_speed
  let t := same_direction_time
  let T := total_travel_time
  (v + c) * t + (v - c) * t = v * T ∧
  2 * ((v + c)⁻¹ + (v - c)⁻¹) * (v * t) = T :=
by sorry

#check sailing_speed_calculation

end NUMINAMATH_CALUDE_sailing_speed_calculation_l2178_217889


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2178_217820

theorem algebraic_simplification (a b : ℝ) : 2*a - 3*(a-b) = -a + 3*b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2178_217820


namespace NUMINAMATH_CALUDE_cute_5digit_integer_count_l2178_217844

/-- A function that checks if a list of digits forms a palindrome -/
def isPalindrome (digits : List Nat) : Prop :=
  digits = digits.reverse

/-- A function that checks if the first k digits of a number are divisible by k -/
def firstKDigitsDivisibleByK (digits : List Nat) (k : Nat) : Prop :=
  let firstK := digits.take k
  let num := firstK.foldl (fun acc d => acc * 10 + d) 0
  num % k = 0

/-- A function that checks if a list of digits satisfies all conditions -/
def isCute (digits : List Nat) : Prop :=
  digits.length = 5 ∧
  digits.toFinset = {1, 2, 3, 4, 5} ∧
  isPalindrome digits ∧
  ∀ k, 1 ≤ k ∧ k ≤ 5 → firstKDigitsDivisibleByK digits k

theorem cute_5digit_integer_count :
  ∃! digits : List Nat, isCute digits :=
sorry

end NUMINAMATH_CALUDE_cute_5digit_integer_count_l2178_217844


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l2178_217865

theorem complex_simplification_and_multiplication :
  ((4 - 3 * Complex.I) - (7 - 5 * Complex.I)) * (1 + 2 * Complex.I) = -7 - 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l2178_217865


namespace NUMINAMATH_CALUDE_trebled_result_is_69_l2178_217816

theorem trebled_result_is_69 :
  let x : ℕ := 7
  let doubled_plus_nine := 2 * x + 9
  let trebled_result := 3 * doubled_plus_nine
  trebled_result = 69 := by
sorry

end NUMINAMATH_CALUDE_trebled_result_is_69_l2178_217816


namespace NUMINAMATH_CALUDE_yoojung_notebooks_l2178_217806

theorem yoojung_notebooks :
  ∀ (initial : ℕ), 
  (initial ≥ 5) →
  (initial - 5) % 2 = 0 →
  ((initial - 5) / 2 - (initial - 5) / 2 / 2 = 4) →
  initial = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_yoojung_notebooks_l2178_217806


namespace NUMINAMATH_CALUDE_plane_properties_l2178_217868

structure Plane

-- Define parallel and perpendicular relations for planes
def parallel (p q : Plane) : Prop := sorry
def perpendicular (p q : Plane) : Prop := sorry

-- Define line as intersection of two planes
def line_intersection (p q : Plane) : Type := sorry

-- Define parallel relation for lines
def line_parallel (l m : Type) : Prop := sorry

theorem plane_properties (α β γ : Plane) (hd : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (parallel α β ∧ parallel β γ → parallel α γ) ∧
  (parallel α β → ∀ (a b : Type), a = line_intersection α γ → b = line_intersection β γ → line_parallel a b) ∧
  (parallel α β ∧ perpendicular β γ → perpendicular α γ) ∧
  ¬(∀ α β γ : Plane, perpendicular α β ∧ perpendicular β γ → perpendicular α γ) := by
  sorry

end NUMINAMATH_CALUDE_plane_properties_l2178_217868


namespace NUMINAMATH_CALUDE_simplify_expression_l2178_217885

theorem simplify_expression : -(-3) - 4 + (-5) = 3 - 4 - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2178_217885


namespace NUMINAMATH_CALUDE_students_in_all_activities_l2178_217878

theorem students_in_all_activities (total : ℕ) (chess : ℕ) (music : ℕ) (art : ℕ) (at_least_two : ℕ) :
  total = 25 →
  chess = 12 →
  music = 15 →
  art = 11 →
  at_least_two = 11 →
  ∃ (only_chess only_music only_art chess_music chess_art music_art all_three : ℕ),
    only_chess + only_music + only_art + chess_music + chess_art + music_art + all_three = total ∧
    only_chess + chess_music + chess_art + all_three = chess ∧
    only_music + chess_music + music_art + all_three = music ∧
    only_art + chess_art + music_art + all_three = art ∧
    chess_music + chess_art + music_art + all_three = at_least_two ∧
    all_three = 4 :=
by sorry

end NUMINAMATH_CALUDE_students_in_all_activities_l2178_217878


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2178_217800

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base_9_number := [4, 2, 3]  -- 324 in base 9 (least significant digit first)
  let base_6_number := [1, 2, 2]  -- 221 in base 6 (least significant digit first)
  (to_base_10 base_9_number 9) - (to_base_10 base_6_number 6) = 180 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2178_217800


namespace NUMINAMATH_CALUDE_dodecagon_area_l2178_217864

/-- Given a square with side length a, prove that the area of a regular dodecagon
    constructed outside the square, where the upper bases of trapezoids on each side
    of the square and their lateral sides form the dodecagon, is equal to (3*a^2)/2. -/
theorem dodecagon_area (a : ℝ) (a_pos : a > 0) :
  let square_side := a
  let dodecagon_area := (3 * a^2) / 2
  dodecagon_area = (3 * square_side^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_dodecagon_area_l2178_217864
