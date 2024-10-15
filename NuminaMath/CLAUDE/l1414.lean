import Mathlib

namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1414_141493

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d = 12 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1414_141493


namespace NUMINAMATH_CALUDE_sector_central_angle_l1414_141464

theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 10) (h2 : area = 100) :
  (2 * area) / (r^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1414_141464


namespace NUMINAMATH_CALUDE_least_jumps_to_19999_l1414_141485

/-- Represents the total distance jumped after n jumps -/
def totalDistance (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the distance of the nth jump -/
def nthJump (n : ℕ) : ℕ := n

theorem least_jumps_to_19999 :
  ∀ k : ℕ, (totalDistance k ≥ 19999 → k ≥ 201) ∧
  (∃ (adjustedJump : ℤ), 
    totalDistance 201 + nthJump 201 + adjustedJump = 19999 ∧ 
    adjustedJump.natAbs < nthJump 201) := by
  sorry

end NUMINAMATH_CALUDE_least_jumps_to_19999_l1414_141485


namespace NUMINAMATH_CALUDE_yellow_teams_count_l1414_141425

theorem yellow_teams_count (blue_students yellow_students total_students total_teams blue_teams : ℕ)
  (h1 : blue_students = 70)
  (h2 : yellow_students = 84)
  (h3 : total_students = blue_students + yellow_students)
  (h4 : total_teams = 77)
  (h5 : total_students = 2 * total_teams)
  (h6 : blue_teams = 30) :
  ∃ yellow_teams : ℕ, yellow_teams = 37 ∧ 
    yellow_teams = total_teams - blue_teams - (blue_students + yellow_students - 2 * blue_teams) / 2 :=
by sorry

end NUMINAMATH_CALUDE_yellow_teams_count_l1414_141425


namespace NUMINAMATH_CALUDE_residue_mod_35_l1414_141488

theorem residue_mod_35 : ∃ r : ℤ, 0 ≤ r ∧ r < 35 ∧ (-963 + 100) ≡ r [ZMOD 35] ∧ r = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_35_l1414_141488


namespace NUMINAMATH_CALUDE_prime_and_multiple_of_5_probability_l1414_141446

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is a multiple of 5 -/
def isMultipleOf5 (n : ℕ) : Prop := sorry

/-- The set of cards numbered from 1 to 75 -/
def cardSet : Finset ℕ := sorry

/-- The probability of an event occurring when selecting from the card set -/
def probability (event : ℕ → Prop) : ℚ := sorry

theorem prime_and_multiple_of_5_probability :
  probability (fun n => n ∈ cardSet ∧ isPrime n ∧ isMultipleOf5 n) = 1 / 75 := by sorry

end NUMINAMATH_CALUDE_prime_and_multiple_of_5_probability_l1414_141446


namespace NUMINAMATH_CALUDE_ara_height_ara_current_height_ara_height_is_59_l1414_141491

theorem ara_height (shea_initial : ℝ) (shea_final : ℝ) (ara_initial : ℝ) : ℝ :=
  let shea_growth := shea_final - shea_initial
  let ara_growth := shea_growth / 3
  ara_initial + ara_growth

theorem ara_current_height : ℝ :=
  let shea_final := 64
  let shea_initial := shea_final / 1.25
  let ara_initial := shea_initial + 4
  ara_height shea_initial shea_final ara_initial

theorem ara_height_is_59 : ⌊ara_current_height⌋ = 59 := by
  sorry

end NUMINAMATH_CALUDE_ara_height_ara_current_height_ara_height_is_59_l1414_141491


namespace NUMINAMATH_CALUDE_sample_variance_estimates_stability_l1414_141473

-- Define the type for sample statistics
inductive SampleStatistic
  | Mean
  | Median
  | Variance
  | Maximum

-- Define a function that determines if a statistic estimates population stability
def estimatesStability (stat : SampleStatistic) : Prop :=
  match stat with
  | SampleStatistic.Variance => True
  | _ => False

-- Theorem statement
theorem sample_variance_estimates_stability :
  ∃ (stat : SampleStatistic), estimatesStability stat ∧
  (stat = SampleStatistic.Mean ∨
   stat = SampleStatistic.Median ∨
   stat = SampleStatistic.Variance ∨
   stat = SampleStatistic.Maximum) :=
by
  sorry

end NUMINAMATH_CALUDE_sample_variance_estimates_stability_l1414_141473


namespace NUMINAMATH_CALUDE_fresh_fruit_water_content_l1414_141463

theorem fresh_fruit_water_content
  (dried_water_content : ℝ)
  (dried_weight : ℝ)
  (fresh_weight : ℝ)
  (h1 : dried_water_content = 0.15)
  (h2 : dried_weight = 12)
  (h3 : fresh_weight = 101.99999999999999) :
  (fresh_weight - dried_weight * (1 - dried_water_content)) / fresh_weight = 0.9 := by
sorry

end NUMINAMATH_CALUDE_fresh_fruit_water_content_l1414_141463


namespace NUMINAMATH_CALUDE_triangle_angle_a_value_l1414_141403

theorem triangle_angle_a_value (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = Real.sin B + Real.cos B ∧
  a = Real.sqrt 2 ∧
  b = 2 ∧
  a / Real.sin A = b / Real.sin B ∧
  a < b →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_a_value_l1414_141403


namespace NUMINAMATH_CALUDE_annas_initial_candies_l1414_141482

/-- Given that Anna receives some candies from Larry and ends up with a total number of candies,
    this theorem proves how many candies Anna started with. -/
theorem annas_initial_candies
  (candies_from_larry : ℕ)
  (total_candies : ℕ)
  (h1 : candies_from_larry = 86)
  (h2 : total_candies = 91)
  : total_candies - candies_from_larry = 5 := by
  sorry

end NUMINAMATH_CALUDE_annas_initial_candies_l1414_141482


namespace NUMINAMATH_CALUDE_car_impact_suitable_for_sampling_l1414_141466

/-- Characteristics of a suitable sampling survey scenario -/
structure SamplingSurveyCharacteristics where
  large_population : Bool
  impractical_full_survey : Bool
  representative_sample_possible : Bool

/-- Options for the survey scenario -/
inductive SurveyOption
  | A  -- Understanding the height of students in Class 7(1)
  | B  -- Companies recruiting and interviewing job applicants
  | C  -- Investigating the impact resistance of a batch of cars
  | D  -- Selecting the fastest runner in our school for competition

/-- Determine if an option is suitable for sampling survey -/
def is_suitable_for_sampling (option : SurveyOption) : Bool :=
  match option with
  | SurveyOption.C => true
  | _ => false

/-- Characteristics of the car impact resistance scenario -/
def car_impact_scenario : SamplingSurveyCharacteristics :=
  { large_population := true,
    impractical_full_survey := true,
    representative_sample_possible := true }

/-- Theorem stating that investigating car impact resistance is suitable for sampling survey -/
theorem car_impact_suitable_for_sampling :
  is_suitable_for_sampling SurveyOption.C ∧
  car_impact_scenario.large_population ∧
  car_impact_scenario.impractical_full_survey ∧
  car_impact_scenario.representative_sample_possible :=
sorry

end NUMINAMATH_CALUDE_car_impact_suitable_for_sampling_l1414_141466


namespace NUMINAMATH_CALUDE_pet_store_combinations_l1414_141407

def num_puppies : ℕ := 15
def num_kittens : ℕ := 6
def num_hamsters : ℕ := 8
def num_friends : ℕ := 3

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * Nat.factorial num_friends = 4320 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l1414_141407


namespace NUMINAMATH_CALUDE_nines_count_to_thousand_l1414_141423

/-- Count of digit 9 appearances in a single integer -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Count of digit 9 appearances in all integers from 1 to n (inclusive) -/
def total_nines (n : ℕ) : ℕ := sorry

/-- Theorem: The count of digit 9 appearances in all integers from 1 to 1000 (inclusive) is 301 -/
theorem nines_count_to_thousand : total_nines 1000 = 301 := by sorry

end NUMINAMATH_CALUDE_nines_count_to_thousand_l1414_141423


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1414_141431

/-- Given an arithmetic sequence, prove that if the difference of the average of the first 2016 terms
    and the average of the first 16 terms is 100, then the common difference is 1/10. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (d : ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) = a n + d) 
  (h_sum : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) 
  (h_condition : S 2016 / 2016 - S 16 / 16 = 100) :
  d = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1414_141431


namespace NUMINAMATH_CALUDE_factor_expression_l1414_141448

theorem factor_expression (c : ℝ) : 210 * c^2 + 35 * c = 35 * c * (6 * c + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1414_141448


namespace NUMINAMATH_CALUDE_inequality_solution_set_minimum_value_minimum_value_equality_l1414_141405

-- Problem 1
theorem inequality_solution_set (x : ℝ) :
  (2 * x + 1) / (3 - x) ≥ 1 ↔ (2/3 ≤ x ∧ x < 2) ∨ x > 2 :=
sorry

-- Problem 2
theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) ≥ 25 :=
sorry

theorem minimum_value_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) = 25 ↔ x = 2/5 ∧ y = 3/5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_minimum_value_minimum_value_equality_l1414_141405


namespace NUMINAMATH_CALUDE_mandy_toys_count_l1414_141454

theorem mandy_toys_count (mandy anna amanda : ℕ) 
  (h1 : anna = 3 * mandy)
  (h2 : amanda = anna + 2)
  (h3 : mandy + anna + amanda = 142) :
  mandy = 20 := by
sorry

end NUMINAMATH_CALUDE_mandy_toys_count_l1414_141454


namespace NUMINAMATH_CALUDE_expected_value_twelve_sided_die_l1414_141427

/-- A twelve-sided die with faces numbered from 1 to 12 -/
def TwelveSidedDie := Finset.range 12

/-- The expected value of a roll of the twelve-sided die -/
def expectedValue : ℚ := (TwelveSidedDie.sum (λ i => i + 1)) / 12

/-- Theorem: The expected value of a roll of a twelve-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_twelve_sided_die : expectedValue = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_twelve_sided_die_l1414_141427


namespace NUMINAMATH_CALUDE_barbi_weight_loss_duration_l1414_141462

/-- Proves that Barbi lost weight for 12 months given the conditions -/
theorem barbi_weight_loss_duration :
  let barbi_monthly_loss : ℝ := 1.5
  let luca_total_loss : ℝ := 9 * 11
  let loss_difference : ℝ := 81
  
  ∃ (months : ℝ), 
    months * barbi_monthly_loss = luca_total_loss - loss_difference ∧ 
    months = 12 := by
  sorry

end NUMINAMATH_CALUDE_barbi_weight_loss_duration_l1414_141462


namespace NUMINAMATH_CALUDE_sqrt_two_squared_l1414_141404

theorem sqrt_two_squared : Real.sqrt 2 * Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_l1414_141404


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_range_of_a_for_A_intersect_C_eq_C_l1414_141422

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x | (x-4)/(x+1) < 0}
def C (a : ℝ) : Set ℝ := {x | 2-a < x ∧ x < 2+a}

-- Statement for (∁_R A) ∩ B = (3, 4)
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = Set.Ioo 3 4 := by sorry

-- Statement for the range of a when A ∩ C = C
theorem range_of_a_for_A_intersect_C_eq_C :
  ∀ a : ℝ, (A ∩ C a = C a) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_range_of_a_for_A_intersect_C_eq_C_l1414_141422


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l1414_141481

/-- Calculates the profit percentage of a dishonest dealer. -/
theorem dishonest_dealer_profit_percentage 
  (actual_weight : ℝ) 
  (claimed_weight : ℝ) 
  (actual_weight_positive : 0 < actual_weight)
  (claimed_weight_positive : 0 < claimed_weight)
  (h_weights : actual_weight = 575 ∧ claimed_weight = 1000) :
  (claimed_weight - actual_weight) / claimed_weight * 100 = 42.5 :=
by
  sorry

#check dishonest_dealer_profit_percentage

end NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l1414_141481


namespace NUMINAMATH_CALUDE_plane_through_point_and_line_l1414_141418

def line_equation (x y z : ℝ) : Prop :=
  (x - 2) / 4 = (y + 1) / (-5) ∧ (y + 1) / (-5) = (z - 3) / 2

def plane_equation (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 4 * z - 14 = 0

def point_on_plane (x y z : ℝ) : Prop :=
  x = 2 ∧ y = -3 ∧ z = 5

def coefficients_conditions (A B C D : ℤ) : Prop :=
  A > 0 ∧ Nat.gcd (Nat.gcd (Nat.gcd (A.natAbs) (B.natAbs)) (C.natAbs)) (D.natAbs) = 1

theorem plane_through_point_and_line :
  ∀ (x y z : ℝ),
    (∃ (t : ℝ), line_equation (x + t) (y + t) (z + t)) →
    point_on_plane 2 (-3) 5 →
    coefficients_conditions 3 4 4 (-14) →
    plane_equation x y z :=
sorry

end NUMINAMATH_CALUDE_plane_through_point_and_line_l1414_141418


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1414_141426

theorem unique_triple_solution :
  ∃! (a b c : ℕ), b > 1 ∧ 2^c + 2^2016 = a^b ∧ a = 3 * 2^1008 ∧ b = 2 ∧ c = 2019 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1414_141426


namespace NUMINAMATH_CALUDE_largest_base5_to_decimal_l1414_141439

/-- Converts a base-5 digit to its decimal (base-10) value --/
def base5ToDecimal (digit : Nat) : Nat := digit

/-- Calculates the value of a base-5 digit in its positional notation --/
def digitValue (digit : Nat) (position : Nat) : Nat := 
  base5ToDecimal digit * (5 ^ position)

/-- Represents a five-digit base-5 number --/
structure FiveDigitBase5Number where
  digit1 : Nat
  digit2 : Nat
  digit3 : Nat
  digit4 : Nat
  digit5 : Nat
  all_digits_valid : digit1 < 5 ∧ digit2 < 5 ∧ digit3 < 5 ∧ digit4 < 5 ∧ digit5 < 5

/-- Converts a five-digit base-5 number to its decimal (base-10) equivalent --/
def toDecimal (n : FiveDigitBase5Number) : Nat :=
  digitValue n.digit1 4 + digitValue n.digit2 3 + digitValue n.digit3 2 + 
  digitValue n.digit4 1 + digitValue n.digit5 0

/-- The largest five-digit base-5 number --/
def largestBase5 : FiveDigitBase5Number where
  digit1 := 4
  digit2 := 4
  digit3 := 4
  digit4 := 4
  digit5 := 4
  all_digits_valid := by simp

theorem largest_base5_to_decimal : 
  toDecimal largestBase5 = 3124 := by sorry

end NUMINAMATH_CALUDE_largest_base5_to_decimal_l1414_141439


namespace NUMINAMATH_CALUDE_fruit_platter_grapes_l1414_141437

theorem fruit_platter_grapes :
  ∀ (b r g c : ℚ),
  b + r + g + c = 360 →
  r = 3 * b →
  g = 4 * c →
  c = 5 * r →
  g = 21600 / 79 := by
sorry

end NUMINAMATH_CALUDE_fruit_platter_grapes_l1414_141437


namespace NUMINAMATH_CALUDE_find_M_l1414_141414

theorem find_M : ∃ M : ℚ, (5 + 7 + 10) / 3 = (2020 + 2021 + 2022) / M ∧ M = 827 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l1414_141414


namespace NUMINAMATH_CALUDE_inscribed_hexagon_side_length_l1414_141472

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  BAC : ℝ

/-- Regular hexagon UVWXYZ inscribed in triangle ABC -/
structure InscribedHexagon where
  triangle : Triangle
  sideLength : ℝ

/-- Theorem stating the side length of the inscribed hexagon -/
theorem inscribed_hexagon_side_length (t : Triangle) (h : InscribedHexagon) 
  (h1 : t.AB = 5)
  (h2 : t.AC = 8)
  (h3 : t.BAC = π / 3)
  (h4 : h.triangle = t)
  (h5 : ∃ (U V W X Z : ℝ × ℝ), 
    U.1 + V.1 = t.AB ∧ 
    W.2 + X.2 = t.AC ∧ 
    Z.1^2 + Z.2^2 = t.AB^2 + t.AC^2 - 2 * t.AB * t.AC * Real.cos t.BAC) :
  h.sideLength = 35 / 19 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_side_length_l1414_141472


namespace NUMINAMATH_CALUDE_inscribed_circumscribed_ratio_l1414_141455

/-- Given a right-angled triangle with perpendicular sides of 6 and 8,
    prove that the ratio of the radius of the inscribed circle
    to the radius of the circumscribed circle is 2:5 -/
theorem inscribed_circumscribed_ratio (a b c r R : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → 
  r = (a + b - c) / 2 → R = c / 2 → 
  r / R = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circumscribed_ratio_l1414_141455


namespace NUMINAMATH_CALUDE_basketball_probability_l1414_141469

/-- A sequence of basketball shots where the probability of hitting each shot
    after the first two is equal to the proportion of shots hit so far. -/
def BasketballSequence (n : ℕ) : Type :=
  Fin n → Bool

/-- The probability of hitting exactly k shots out of n in a BasketballSequence. -/
def hitProbability (n k : ℕ) : ℚ :=
  if k = 0 ∨ k = n then 0
  else if k = 1 ∧ n = 2 then 1
  else 1 / (n - 1)

/-- The theorem stating the probability of hitting exactly k shots out of n. -/
theorem basketball_probability (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
    hitProbability n k = 1 / (n - 1) := by
  sorry

#eval hitProbability 100 50

end NUMINAMATH_CALUDE_basketball_probability_l1414_141469


namespace NUMINAMATH_CALUDE_possible_values_of_b_over_a_l1414_141459

theorem possible_values_of_b_over_a (a b : ℝ) (h : a > 0) :
  (∀ a b, a > 0 → Real.log a + b - a * Real.exp (b - 1) ≥ 0) →
  (b / a = Real.exp (-1) ∨ b / a = Real.exp (-2) ∨ b / a = -Real.exp (-2)) :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_b_over_a_l1414_141459


namespace NUMINAMATH_CALUDE_correct_people_left_l1414_141486

/-- Calculates the number of people left on a train after two stops -/
def peopleLeftOnTrain (initialPeople : ℕ) (peopleGotOff : ℕ) (peopleGotOn : ℕ) : ℕ :=
  initialPeople - peopleGotOff + peopleGotOn

theorem correct_people_left : peopleLeftOnTrain 123 58 37 = 102 := by
  sorry

end NUMINAMATH_CALUDE_correct_people_left_l1414_141486


namespace NUMINAMATH_CALUDE_square_overlap_areas_l1414_141450

/-- Given a square with side length 3 cm cut along its diagonal, prove the areas of overlap in specific arrangements --/
theorem square_overlap_areas :
  let square_side : ℝ := 3
  let small_square_side : ℝ := 1
  let triangle_area : ℝ := square_side ^ 2 / 2
  let small_square_area : ℝ := small_square_side ^ 2
  
  -- Area of overlap when a 1 cm × 1 cm square is placed inside one of the resulting triangles
  let overlap_area_b : ℝ := small_square_area / 4
  
  -- Area of overlap when the two triangles are arranged to form a rectangle of 1 cm × 3 cm with an additional overlap
  let overlap_area_c : ℝ := triangle_area / 2
  
  (overlap_area_b = 0.25 ∧ overlap_area_c = 2.25) := by
  sorry


end NUMINAMATH_CALUDE_square_overlap_areas_l1414_141450


namespace NUMINAMATH_CALUDE_two_digit_addition_proof_l1414_141430

theorem two_digit_addition_proof (A B C : ℕ) : 
  A ≠ B → A ≠ C → B ≠ C →
  A < 10 → B < 10 → C < 10 →
  A > 0 → B > 0 → C > 0 →
  (10 * A + B) + (10 * B + C) = 100 * B + 10 * C + B →
  A = 9 := by
sorry

end NUMINAMATH_CALUDE_two_digit_addition_proof_l1414_141430


namespace NUMINAMATH_CALUDE_graphs_intersect_once_l1414_141412

/-- The value of b for which the graphs of y = bx^2 + 5x + 3 and y = -2x - 3 intersect at exactly one point -/
def b : ℚ := 49 / 24

/-- The first equation -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 3

/-- The second equation -/
def g (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem stating that the graphs intersect at exactly one point when b = 49/24 -/
theorem graphs_intersect_once :
  ∃! x : ℝ, f x = g x :=
sorry

end NUMINAMATH_CALUDE_graphs_intersect_once_l1414_141412


namespace NUMINAMATH_CALUDE_function_characterization_l1414_141432

theorem function_characterization (f : ℕ+ → ℕ+ → ℕ+) :
  (∀ x : ℕ+, f x x = x) →
  (∀ x y : ℕ+, f x y = f y x) →
  (∀ x y : ℕ+, (x + y) * (f x y) = y * (f x (x + y))) →
  (∀ x y : ℕ+, f x y = Nat.lcm x y) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l1414_141432


namespace NUMINAMATH_CALUDE_vertical_angles_equal_l1414_141477

-- Define a line as a type
def Line := ℝ → ℝ → Prop

-- Define an angle as a pair of lines
def Angle := Line × Line

-- Define vertical angles
def VerticalAngles (a b : Angle) : Prop :=
  ∃ (l1 l2 : Line), l1 ≠ l2 ∧ 
    ((a.1 = l1 ∧ a.2 = l2) ∨ (a.1 = l2 ∧ a.2 = l1)) ∧
    ((b.1 = l1 ∧ b.2 = l2) ∨ (b.1 = l2 ∧ b.2 = l1))

-- Define angle measure
def AngleMeasure (a : Angle) : ℝ := sorry

-- Theorem: Vertical angles are always equal
theorem vertical_angles_equal (a b : Angle) :
  VerticalAngles a b → AngleMeasure a = AngleMeasure b := by
  sorry

end NUMINAMATH_CALUDE_vertical_angles_equal_l1414_141477


namespace NUMINAMATH_CALUDE_cone_height_ratio_l1414_141409

/-- Theorem about the ratio of heights in a cone with reduced height --/
theorem cone_height_ratio (original_height : ℝ) (base_circumference : ℝ) (new_volume : ℝ) :
  original_height = 20 →
  base_circumference = 18 * Real.pi →
  new_volume = 270 * Real.pi →
  ∃ (new_height : ℝ),
    (1 / 3 : ℝ) * Real.pi * (base_circumference / (2 * Real.pi))^2 * new_height = new_volume ∧
    new_height / original_height = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l1414_141409


namespace NUMINAMATH_CALUDE_water_displaced_squared_volume_l1414_141461

/-- The volume of water displaced by a cube in a cylindrical tank -/
def water_displaced (cube_side : ℝ) (tank_radius : ℝ) (tank_height : ℝ) : ℝ :=
  -- Definition left abstract
  sorry

/-- The main theorem stating the squared volume of water displaced -/
theorem water_displaced_squared_volume :
  let cube_side : ℝ := 10
  let tank_radius : ℝ := 5
  let tank_height : ℝ := 12
  (water_displaced cube_side tank_radius tank_height) ^ 2 = 79156.25 := by
  sorry

end NUMINAMATH_CALUDE_water_displaced_squared_volume_l1414_141461


namespace NUMINAMATH_CALUDE_max_probability_two_color_balls_l1414_141468

def p (n : ℕ+) : ℚ :=
  (10 * n) / ((n + 5) * (n + 4))

theorem max_probability_two_color_balls :
  ∀ n : ℕ+, p n ≤ 5/9 :=
by sorry

end NUMINAMATH_CALUDE_max_probability_two_color_balls_l1414_141468


namespace NUMINAMATH_CALUDE_stacked_cubes_volume_l1414_141417

/-- Calculates the total volume of stacked cubes -/
def total_volume (cube_dim : ℝ) (rows cols floors : ℕ) : ℝ :=
  (cube_dim ^ 3) * (rows * cols * floors)

/-- The problem statement -/
theorem stacked_cubes_volume :
  let cube_dim : ℝ := 1
  let rows : ℕ := 7
  let cols : ℕ := 5
  let floors : ℕ := 3
  total_volume cube_dim rows cols floors = 105 := by
  sorry

end NUMINAMATH_CALUDE_stacked_cubes_volume_l1414_141417


namespace NUMINAMATH_CALUDE_rational_function_value_l1414_141444

theorem rational_function_value (x : ℝ) (h : x ≠ 5) :
  x = 4 → (x^2 - 3*x - 10) / (x - 5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l1414_141444


namespace NUMINAMATH_CALUDE_power_of_power_l1414_141438

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1414_141438


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1414_141494

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1414_141494


namespace NUMINAMATH_CALUDE_ranch_minimum_animals_l1414_141406

theorem ranch_minimum_animals : ∀ (ponies horses : ℕ),
  ponies > 0 →
  horses = ponies + 4 →
  (3 * ponies) % 10 = 0 →
  (5 * ((3 * ponies) / 10)) % 8 = 0 →
  ponies + horses ≥ 36 :=
by
  sorry

end NUMINAMATH_CALUDE_ranch_minimum_animals_l1414_141406


namespace NUMINAMATH_CALUDE_solve_for_y_l1414_141413

theorem solve_for_y (x y : ℝ) (hx : x = 99) (heq : x^3*y - 2*x^2*y + x*y = 970200) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1414_141413


namespace NUMINAMATH_CALUDE_molecular_weight_Al2S3_l1414_141411

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Sulfur in g/mol -/
def atomic_weight_S : ℝ := 32.06

/-- The number of Aluminum atoms in Al2S3 -/
def num_Al : ℕ := 2

/-- The number of Sulfur atoms in Al2S3 -/
def num_S : ℕ := 3

/-- The number of moles of Al2S3 -/
def num_moles : ℝ := 10

/-- Theorem: The molecular weight of 10 moles of Al2S3 is 1501.4 grams -/
theorem molecular_weight_Al2S3 : 
  num_moles * (num_Al * atomic_weight_Al + num_S * atomic_weight_S) = 1501.4 := by
  sorry


end NUMINAMATH_CALUDE_molecular_weight_Al2S3_l1414_141411


namespace NUMINAMATH_CALUDE_peters_fish_catch_l1414_141421

theorem peters_fish_catch (n : ℕ) : (3 * n = n + 24) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_peters_fish_catch_l1414_141421


namespace NUMINAMATH_CALUDE_conference_handshakes_l1414_141487

/-- Represents a conference with two groups of people -/
structure Conference :=
  (total : ℕ)
  (group_a : ℕ)
  (group_b : ℕ)
  (h_total : total = group_a + group_b)

/-- Calculates the number of handshakes in a conference -/
def handshakes (c : Conference) : ℕ :=
  c.group_a * c.group_b + (c.group_b.choose 2)

/-- Theorem stating the number of handshakes in the specific conference -/
theorem conference_handshakes :
  ∃ (c : Conference), c.total = 40 ∧ c.group_a = 25 ∧ c.group_b = 15 ∧ handshakes c = 480 :=
by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1414_141487


namespace NUMINAMATH_CALUDE_projection_matrix_values_l1414_141447

/-- A projection matrix is idempotent (P² = P) -/
def IsProjectionMatrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific form of our projection matrix -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 21/76],
    ![c, 55/76]]

theorem projection_matrix_values :
  ∃ (a c : ℚ), IsProjectionMatrix (P a c) ∧ a = 7/19 ∧ c = 21/76 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l1414_141447


namespace NUMINAMATH_CALUDE_polynomial_identity_l1414_141419

theorem polynomial_identity : 
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 1051012301 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1414_141419


namespace NUMINAMATH_CALUDE_library_comic_books_l1414_141443

theorem library_comic_books (fairy_tale_books : ℕ) (science_tech_books : ℕ) (comic_books : ℕ) : 
  fairy_tale_books = 305 →
  science_tech_books = fairy_tale_books + 115 →
  comic_books = 4 * (fairy_tale_books + science_tech_books) →
  comic_books = 2900 := by
sorry

end NUMINAMATH_CALUDE_library_comic_books_l1414_141443


namespace NUMINAMATH_CALUDE_class_average_score_l1414_141428

theorem class_average_score (total_students : ℕ) (score1 score2 score3 : ℕ) (rest_average : ℚ) :
  total_students = 35 →
  score1 = 93 →
  score2 = 83 →
  score3 = 87 →
  rest_average = 76 →
  (score1 + score2 + score3 + (total_students - 3) * rest_average) / total_students = 77 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l1414_141428


namespace NUMINAMATH_CALUDE_power_division_rule_l1414_141471

theorem power_division_rule (x : ℝ) : x^6 / x^3 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1414_141471


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_when_f2_less_than_4_l1414_141489

noncomputable section

variable (a : ℝ)
variable (h : a > 0)

def f (x : ℝ) : ℝ := |x + 1/a| + |x - a|

theorem f_lower_bound : ∀ x : ℝ, f a x ≥ 2 :=
sorry

theorem a_range_when_f2_less_than_4 : 
  f a 2 < 4 → 1 < a ∧ a < 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_when_f2_less_than_4_l1414_141489


namespace NUMINAMATH_CALUDE_mango_seller_profit_l1414_141496

/-- Proves that given a fruit seller who loses 15% when selling mangoes at Rs. 6 per kg,
    if they want to sell at Rs. 7.411764705882353 per kg, their desired profit percentage is 5%. -/
theorem mango_seller_profit (loss_price : ℝ) (loss_percentage : ℝ) (desired_price : ℝ) :
  loss_price = 6 →
  loss_percentage = 15 →
  desired_price = 7.411764705882353 →
  let cost_price := loss_price / (1 - loss_percentage / 100)
  let profit_percentage := (desired_price / cost_price - 1) * 100
  profit_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_mango_seller_profit_l1414_141496


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1414_141497

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 8*x + 12 = 0 →
  x > 0 →
  4 + x > 7 ∧ 7 + x > 4 ∧ 4 + 7 > x →
  4 + 7 + x = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1414_141497


namespace NUMINAMATH_CALUDE_race_cars_l1414_141420

theorem race_cars (p_x p_y p_z p_total : ℚ) : 
  p_x = 1/8 → p_y = 1/12 → p_z = 1/6 → p_total = 375/1000 → 
  p_x + p_y + p_z = p_total → 
  ∀ p_other : ℚ, p_other ≥ 0 → p_x + p_y + p_z + p_other = p_total → p_other = 0 := by
  sorry

end NUMINAMATH_CALUDE_race_cars_l1414_141420


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l1414_141476

theorem polynomial_coefficient_B (A : ℤ) :
  ∃ (r₁ r₂ r₃ r₄ : ℕ+),
    (r₁ : ℤ) + r₂ + r₃ + r₄ = 7 ∧
    ∀ (z : ℂ), z^4 - 7*z^3 + A*z^2 + (-12)*z + 24 = (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l1414_141476


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l1414_141433

theorem factor_implies_d_value (d : ℝ) : 
  (∀ x : ℝ, (2 * x + 5) ∣ (8 * x^3 + 27 * x^2 + d * x + 55)) → d = 39.5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l1414_141433


namespace NUMINAMATH_CALUDE_base_4_addition_l1414_141478

/-- Convert a base 10 number to base 4 --/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Convert a base 4 number (represented as a list of digits) to base 10 --/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

/-- Add two base 4 numbers (represented as lists of digits) --/
def addBase4 (a b : List ℕ) : List ℕ :=
  sorry

theorem base_4_addition :
  addBase4 (toBase4 45) (toBase4 28) = [1, 0, 2, 1] ∧ fromBase4 [1, 0, 2, 1] = 45 + 28 := by
  sorry

end NUMINAMATH_CALUDE_base_4_addition_l1414_141478


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1414_141480

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The standard equation of a hyperbola -/
def standardEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The asymptotic line equations of a hyperbola -/
def asymptoticLines (h : Hyperbola) (x y : ℝ) : Prop :=
  y = h.b / h.a * x ∨ y = -h.b / h.a * x

/-- Theorem stating that the given standard equation implies the asymptotic lines,
    but not necessarily vice versa -/
theorem hyperbola_asymptotes (h : Hyperbola) :
  (h.a = 4 ∧ h.b = 3 → ∀ x y, standardEquation h x y → asymptoticLines h x y) ∧
  ¬(∀ h : Hyperbola, (∀ x y, asymptoticLines h x y → standardEquation h x y)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1414_141480


namespace NUMINAMATH_CALUDE_alabama_theorem_l1414_141452

/-- The number of letters in the word "ALABAMA" -/
def total_letters : ℕ := 7

/-- The number of 'A's in the word "ALABAMA" -/
def number_of_as : ℕ := 4

/-- The number of unique arrangements of the letters in "ALABAMA" -/
def alabama_arrangements : ℕ := total_letters.factorial / number_of_as.factorial

theorem alabama_theorem : alabama_arrangements = 210 := by
  sorry

end NUMINAMATH_CALUDE_alabama_theorem_l1414_141452


namespace NUMINAMATH_CALUDE_unique_k_value_l1414_141453

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The quadratic equation x^2 - 63x + k = 0 with prime roots -/
def quadratic_equation (k : ℕ) (x : ℝ) : Prop :=
  x^2 - 63*x + k = 0

/-- The roots of the quadratic equation are prime numbers -/
def roots_are_prime (k : ℕ) : Prop :=
  ∃ (a b : ℕ), (is_prime a ∧ is_prime b) ∧
  (∀ x : ℝ, quadratic_equation k x ↔ (x = a ∨ x = b))

theorem unique_k_value : ∃! k : ℕ, roots_are_prime k ∧ k = 122 :=
sorry

end NUMINAMATH_CALUDE_unique_k_value_l1414_141453


namespace NUMINAMATH_CALUDE_shelby_rain_time_l1414_141415

/-- Represents the driving scenario for Shelby -/
structure DrivingScenario where
  speed_sun : ℝ  -- Speed when not raining (miles per hour)
  speed_rain : ℝ  -- Speed when raining (miles per hour)
  total_distance : ℝ  -- Total distance driven (miles)
  total_time : ℝ  -- Total time driven (minutes)

/-- Calculates the time driven in rain given a DrivingScenario -/
def time_in_rain (scenario : DrivingScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, Shelby drove 16 minutes in the rain -/
theorem shelby_rain_time :
  let scenario : DrivingScenario := {
    speed_sun := 40,
    speed_rain := 25,
    total_distance := 20,
    total_time := 36
  }
  time_in_rain scenario = 16 := by
  sorry

end NUMINAMATH_CALUDE_shelby_rain_time_l1414_141415


namespace NUMINAMATH_CALUDE_cookies_left_l1414_141451

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens John bought -/
def dozens_bought : ℕ := 2

/-- The number of cookies John ate -/
def cookies_eaten : ℕ := 3

/-- Theorem: John has 21 cookies left -/
theorem cookies_left : dozens_bought * dozen - cookies_eaten = 21 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l1414_141451


namespace NUMINAMATH_CALUDE_composite_function_problem_l1414_141424

-- Definition of composite function for linear functions
def composite_function (k₁ b₁ k₂ b₂ : ℝ) : ℝ → ℝ :=
  λ x => (k₁ + k₂) * x + b₁ * b₂

theorem composite_function_problem :
  -- 1. Composite of y=3x+2 and y=-4x+3
  (∀ x, composite_function 3 2 (-4) 3 x = -x + 6) ∧
  -- 2. If composite of y=ax-2 and y=-x+b is y=3x+2, then a=4 and b=-1
  (∀ a b, (∀ x, composite_function a (-2) (-1) b x = 3 * x + 2) → a = 4 ∧ b = -1) ∧
  -- 3. Conditions for passing through first, second, and fourth quadrants
  (∀ k b, (∀ x, (composite_function (-1) b k (-3) x > 0 ∧ x > 0) ∨
                (composite_function (-1) b k (-3) x < 0 ∧ x > 0) ∨
                (composite_function (-1) b k (-3) x > 0 ∧ x < 0)) →
    k < 1 ∧ b < 0) ∧
  -- 4. Fixed point of composite of y=-2x+m and y=3mx-6
  (∀ m, composite_function (-2) m (3*m) (-6) 2 = -4) := by
  sorry

end NUMINAMATH_CALUDE_composite_function_problem_l1414_141424


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1414_141465

theorem sqrt_product_simplification (y : ℝ) (h : y > 0) :
  Real.sqrt (48 * y) * Real.sqrt (3 * y) * Real.sqrt (50 * y) = 30 * y * Real.sqrt (2 * y) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1414_141465


namespace NUMINAMATH_CALUDE_farm_animals_l1414_141440

theorem farm_animals (total_heads : ℕ) (total_feet : ℕ) (hen_heads : ℕ) (hen_feet : ℕ) (cow_heads : ℕ) (cow_feet : ℕ) : 
  total_heads = 60 →
  total_feet = 200 →
  hen_heads = 1 →
  hen_feet = 2 →
  cow_heads = 1 →
  cow_feet = 4 →
  ∃ (num_hens : ℕ) (num_cows : ℕ),
    num_hens + num_cows = total_heads ∧
    num_hens * hen_feet + num_cows * cow_feet = total_feet ∧
    num_hens = 20 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_l1414_141440


namespace NUMINAMATH_CALUDE_sequence_range_l1414_141470

/-- Given a sequence {a_n} with the following properties:
  1) a_1 = a > 0
  2) a_(n+1) = -a_n^2 + t*a_n for n ∈ ℕ*
  3) There exists a real number t that makes {a_n} monotonically increasing
  Then the range of a is (0,1) -/
theorem sequence_range (a : ℝ) (t : ℝ) (a_n : ℕ → ℝ) :
  a > 0 →
  (∀ n : ℕ, n > 0 → a_n (n + 1) = -a_n n ^ 2 + t * a_n n) →
  (∃ t : ℝ, ∀ n : ℕ, n > 0 → a_n (n + 1) > a_n n) →
  a_n 1 = a →
  0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_range_l1414_141470


namespace NUMINAMATH_CALUDE_wedge_volume_l1414_141495

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (θ : ℝ) (h : ℝ) : 
  d = 16 → θ = 30 → h = d → (π * d^2 * h) / 8 = 512 * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l1414_141495


namespace NUMINAMATH_CALUDE_expression_value_l1414_141435

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1414_141435


namespace NUMINAMATH_CALUDE_library_loan_availability_l1414_141449

-- Define the universe of books in the library
variable (Book : Type)

-- Define a predicate for books available for loan
variable (available_for_loan : Book → Prop)

-- Theorem statement
theorem library_loan_availability (h : ¬∀ (b : Book), available_for_loan b) :
  (∃ (b : Book), ¬available_for_loan b) ∧ (¬∀ (b : Book), available_for_loan b) :=
by sorry

end NUMINAMATH_CALUDE_library_loan_availability_l1414_141449


namespace NUMINAMATH_CALUDE_min_value_sqrt_expression_l1414_141484

theorem min_value_sqrt_expression (x : ℝ) (hx : x > 0) :
  2 * Real.sqrt x + 3 / Real.sqrt x ≥ 2 * Real.sqrt 6 ∧
  (2 * Real.sqrt x + 3 / Real.sqrt x = 2 * Real.sqrt 6 ↔ x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_expression_l1414_141484


namespace NUMINAMATH_CALUDE_square_area_error_l1414_141445

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := 1.19 * x
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.4161 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1414_141445


namespace NUMINAMATH_CALUDE_walters_sticker_distribution_l1414_141400

/-- Miss Walter's sticker distribution problem -/
theorem walters_sticker_distribution 
  (gold : ℕ) 
  (silver : ℕ) 
  (bronze : ℕ) 
  (students : ℕ) 
  (h1 : gold = 50)
  (h2 : silver = 2 * gold)
  (h3 : bronze = silver - 20)
  (h4 : students = 5) :
  (gold + silver + bronze) / students = 46 := by
  sorry

end NUMINAMATH_CALUDE_walters_sticker_distribution_l1414_141400


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l1414_141490

/-- Represents the length of an edge in the pyramid --/
inductive EdgeLength
| ten : EdgeLength
| twenty : EdgeLength
| twentyFive : EdgeLength

/-- Represents a triangular face of the pyramid --/
structure TriangularFace where
  edge1 : EdgeLength
  edge2 : EdgeLength
  edge3 : EdgeLength

/-- Represents the pyramid WXYZ --/
structure Pyramid where
  faces : List TriangularFace
  edge_length_condition : ∀ f ∈ faces, f.edge1 ∈ [EdgeLength.ten, EdgeLength.twenty, EdgeLength.twentyFive] ∧
                                       f.edge2 ∈ [EdgeLength.ten, EdgeLength.twenty, EdgeLength.twentyFive] ∧
                                       f.edge3 ∈ [EdgeLength.ten, EdgeLength.twenty, EdgeLength.twentyFive]
  not_equilateral : ∀ f ∈ faces, f.edge1 ≠ f.edge2 ∨ f.edge1 ≠ f.edge3 ∨ f.edge2 ≠ f.edge3

/-- The surface area of a pyramid --/
def surfaceArea (p : Pyramid) : ℝ := sorry

/-- Theorem stating the surface area of the pyramid WXYZ --/
theorem pyramid_surface_area (p : Pyramid) : surfaceArea p = 100 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_l1414_141490


namespace NUMINAMATH_CALUDE_one_true_proposition_l1414_141442

-- Define propositions p and q
def p : Prop := ∀ a b : ℝ, a > b → (1 / a < 1 / b)
def q : Prop := ∀ a b : ℝ, (1 / (a * b) < 0) → (a * b < 0)

-- State the theorem
theorem one_true_proposition (h1 : ¬p) (h2 : q) :
  (p ∧ q) = false ∧ (p ∨ q) = true ∧ ((¬p) ∧ (¬q)) = false :=
sorry

end NUMINAMATH_CALUDE_one_true_proposition_l1414_141442


namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l1414_141458

/-- Given a rectangle with dimensions 6 x 9, prove that the ratio of the volume of the larger cylinder
    to the volume of the smaller cylinder formed by rolling the rectangle is 3/2. -/
theorem rectangle_cylinder_volume_ratio :
  let width : ℝ := 6
  let length : ℝ := 9
  let volume1 : ℝ := π * (width / (2 * π))^2 * length
  let volume2 : ℝ := π * (length / (2 * π))^2 * width
  volume2 / volume1 = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l1414_141458


namespace NUMINAMATH_CALUDE_student_survey_l1414_141436

theorem student_survey (total : ℕ) (mac_preference : ℕ) (both_preference : ℕ) :
  total = 210 →
  mac_preference = 60 →
  both_preference = mac_preference / 3 →
  total - (mac_preference + both_preference) = 130 := by
  sorry

end NUMINAMATH_CALUDE_student_survey_l1414_141436


namespace NUMINAMATH_CALUDE_library_books_count_l1414_141479

theorem library_books_count :
  ∃ (n : ℕ), 
    500 < n ∧ n < 650 ∧ 
    ∃ (r : ℕ), n = 12 * r + 7 ∧
    ∃ (l : ℕ), n = 25 * l - 5 ∧
    n = 595 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l1414_141479


namespace NUMINAMATH_CALUDE_parabola_unique_values_l1414_141474

/-- Parabola passing through (1, 1) and tangent to y = x - 3 at (2, -1) -/
def parabola_conditions (a b c : ℝ) : Prop :=
  -- Passes through (1, 1)
  a + b + c = 1 ∧
  -- Passes through (2, -1)
  4*a + 2*b + c = -1 ∧
  -- Derivative at x = 2 equals slope of y = x - 3
  4*a + b = 1

/-- Theorem stating the unique values of a, b, and c satisfying the conditions -/
theorem parabola_unique_values :
  ∃! (a b c : ℝ), parabola_conditions a b c ∧ a = 3 ∧ b = -11 ∧ c = 9 :=
by sorry

end NUMINAMATH_CALUDE_parabola_unique_values_l1414_141474


namespace NUMINAMATH_CALUDE_number_ratio_l1414_141460

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 5) = 117) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l1414_141460


namespace NUMINAMATH_CALUDE_jerry_ring_toss_games_l1414_141402

/-- The number of games Jerry played in the ring toss game -/
def games_played (total_rings : ℕ) (rings_per_game : ℕ) : ℕ :=
  total_rings / rings_per_game

/-- Theorem: Jerry played 8 games of ring toss -/
theorem jerry_ring_toss_games : games_played 48 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerry_ring_toss_games_l1414_141402


namespace NUMINAMATH_CALUDE_savings_calculation_l1414_141401

/-- Calculates savings given income and income-to-expenditure ratio --/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem: Given the specified conditions, the savings is 2000 --/
theorem savings_calculation :
  let income := 18000
  let income_ratio := 9
  let expenditure_ratio := 8
  calculate_savings income income_ratio expenditure_ratio = 2000 := by
  sorry

#eval calculate_savings 18000 9 8

end NUMINAMATH_CALUDE_savings_calculation_l1414_141401


namespace NUMINAMATH_CALUDE_tangent_sum_l1414_141441

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_l1414_141441


namespace NUMINAMATH_CALUDE_unique_x_value_l1414_141416

theorem unique_x_value : ∃! (x : ℝ), x^2 ∈ ({1, 0, x} : Set ℝ) ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_value_l1414_141416


namespace NUMINAMATH_CALUDE_series_sum_is_36118_l1414_141499

/-- The sign of a term in the series based on its position -/
def sign (n : ℕ) : ℤ :=
  if n ≤ 8 then 1
  else if n ≤ 35 then -1
  else if n ≤ 80 then 1
  else if n ≤ 143 then -1
  -- Continue this pattern up to 10003
  else if n ≤ 9801 then -1
  else 1

/-- The nth term of the series -/
def term (n : ℕ) : ℤ := sign n * n

/-- The sum of the series from 1 to 10003 -/
def seriesSum : ℤ := (List.range 10003).map term |>.sum

theorem series_sum_is_36118 : seriesSum = 36118 := by
  sorry

#eval seriesSum

end NUMINAMATH_CALUDE_series_sum_is_36118_l1414_141499


namespace NUMINAMATH_CALUDE_gcd_nine_factorial_six_factorial_squared_l1414_141429

theorem gcd_nine_factorial_six_factorial_squared : 
  Nat.gcd (Nat.factorial 9) ((Nat.factorial 6)^2) = 43200 := by
  sorry

end NUMINAMATH_CALUDE_gcd_nine_factorial_six_factorial_squared_l1414_141429


namespace NUMINAMATH_CALUDE_turnip_bag_options_l1414_141457

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (o c : ℕ),
    o + c = (bag_weights.sum - t) ∧
    c = 2 * o

theorem turnip_bag_options :
  ∀ t : ℕ, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 :=
by sorry

end NUMINAMATH_CALUDE_turnip_bag_options_l1414_141457


namespace NUMINAMATH_CALUDE_set_of_naturals_less_than_three_l1414_141467

theorem set_of_naturals_less_than_three :
  {x : ℕ | x < 3} = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_of_naturals_less_than_three_l1414_141467


namespace NUMINAMATH_CALUDE_boys_in_class_l1414_141410

theorem boys_in_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) (boys : ℕ) : 
  total = 35 →
  girls_ratio = 4 →
  boys_ratio = 3 →
  girls_ratio * boys = boys_ratio * (total - boys) →
  boys = 15 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l1414_141410


namespace NUMINAMATH_CALUDE_square_root_fourth_power_l1414_141408

theorem square_root_fourth_power (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fourth_power_l1414_141408


namespace NUMINAMATH_CALUDE_cricket_team_matches_l1414_141475

/-- Proves that the total number of matches played by a cricket team in August is 250,
    given the initial and final winning percentages and the number of matches won during a winning streak. -/
theorem cricket_team_matches : 
  ∀ (initial_win_percent : ℝ) (final_win_percent : ℝ) (streak_wins : ℕ),
    initial_win_percent = 0.20 →
    final_win_percent = 0.52 →
    streak_wins = 80 →
    ∃ (total_matches : ℕ),
      total_matches = 250 ∧
      (initial_win_percent * total_matches + streak_wins) / total_matches = final_win_percent :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_matches_l1414_141475


namespace NUMINAMATH_CALUDE_complex_problem_l1414_141498

-- Define complex numbers z₁ and z₂
def z₁ : ℂ := Complex.mk 1 2
def z₂ : ℂ := Complex.mk (-1) 1

-- Define the equation for z
def equation (a : ℝ) (z : ℂ) : Prop :=
  2 * z^2 + a * z + 10 = 0

-- Define the relationship between z and z₁z₂
def z_condition (z : ℂ) : Prop :=
  z.re = (z₁ * z₂).im

-- Main theorem
theorem complex_problem :
  ∃ (a : ℝ) (z : ℂ),
    Complex.abs (z₁ - z₂) = Real.sqrt 5 ∧
    a = 4 ∧
    (z = Complex.mk (-1) 2 ∨ z = Complex.mk (-1) (-2)) ∧
    equation a z ∧
    z_condition z := by sorry

end NUMINAMATH_CALUDE_complex_problem_l1414_141498


namespace NUMINAMATH_CALUDE_piston_experiment_l1414_141456

variable (l d P q π : ℝ)
variable (x y : ℝ)

-- Conditions
variable (h1 : l > 0)
variable (h2 : d > 0)
variable (h3 : P > 0)
variable (h4 : q > 0)
variable (h5 : π > 0)

-- Theorem statement
theorem piston_experiment :
  -- First experiment
  (P * x^2 + 2*q*l*π*x - P*l^2 = 0) ∧
  -- Pressure in AC region
  (l*π / (l + x) = P * (l - x) / q) ∧
  -- Second experiment
  (y = l*P / (q*π - P)) ∧
  -- Condition for piston not falling to bottom
  (P < q*π/2) :=
by sorry

end NUMINAMATH_CALUDE_piston_experiment_l1414_141456


namespace NUMINAMATH_CALUDE_tower_height_l1414_141434

/-- The height of a tower given specific angles and distance -/
theorem tower_height (distance : ℝ) (elevation_angle depression_angle : ℝ) 
  (h1 : distance = 20)
  (h2 : elevation_angle = 30 * π / 180)
  (h3 : depression_angle = 45 * π / 180) :
  ∃ (height : ℝ), height = 20 * (1 + Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_tower_height_l1414_141434


namespace NUMINAMATH_CALUDE_sister_share_is_49_50_l1414_141483

/-- Calculates the amount each sister receives after Gina's spending and investments --/
def sister_share (initial_amount : ℚ) : ℚ :=
  let mom_share := initial_amount * (1 / 4)
  let clothes_share := initial_amount * (1 / 8)
  let charity_share := initial_amount * (1 / 5)
  let groceries_share := initial_amount * (15 / 100)
  let remaining_before_stocks := initial_amount - mom_share - clothes_share - charity_share - groceries_share
  let stocks_investment := remaining_before_stocks * (1 / 10)
  let final_remaining := remaining_before_stocks - stocks_investment
  final_remaining / 2

/-- Theorem stating that each sister receives $49.50 --/
theorem sister_share_is_49_50 :
  sister_share 400 = 49.50 := by sorry

end NUMINAMATH_CALUDE_sister_share_is_49_50_l1414_141483


namespace NUMINAMATH_CALUDE_domino_arrangements_count_l1414_141492

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a domino with length and width -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the number of distinct arrangements of dominoes on a grid -/
def count_arrangements (g : Grid) (d : Domino) (num_dominoes : ℕ) : ℕ :=
  Nat.choose (g.width + g.height - 2) (g.width - 1)

/-- Theorem stating the number of distinct arrangements -/
theorem domino_arrangements_count (g : Grid) (d : Domino) (num_dominoes : ℕ) :
  g.width = 6 →
  g.height = 4 →
  d.length = 2 →
  d.width = 1 →
  num_dominoes = 5 →
  count_arrangements g d num_dominoes = 56 :=
by sorry

end NUMINAMATH_CALUDE_domino_arrangements_count_l1414_141492
