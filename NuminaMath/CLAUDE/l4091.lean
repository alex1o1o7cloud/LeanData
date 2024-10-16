import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l4091_409107

/-- An arithmetic sequence with given third and seventeenth terms -/
structure ArithmeticSequence where
  a₃ : ℚ
  a₁₇ : ℚ
  is_arithmetic : ∃ d, a₁₇ = a₃ + 14 * d

/-- The properties we want to prove about this arithmetic sequence -/
def ArithmeticSequenceProperties (seq : ArithmeticSequence) : Prop :=
  ∃ (a₁₀ : ℚ),
    (seq.a₃ = 11/15) ∧
    (seq.a₁₇ = 2/3) ∧
    (a₁₀ = 7/10) ∧
    (seq.a₃ + a₁₀ + seq.a₁₇ = 21/10)

/-- The main theorem stating that our arithmetic sequence has the desired properties -/
theorem arithmetic_sequence_theorem (seq : ArithmeticSequence) 
    (h₁ : seq.a₃ = 11/15) (h₂ : seq.a₁₇ = 2/3) : 
    ArithmeticSequenceProperties seq := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l4091_409107


namespace NUMINAMATH_CALUDE_quadratic_root_l4091_409148

theorem quadratic_root (m : ℝ) : 
  (2 : ℝ)^2 + m * 2 - 6 = 0 → (-3 : ℝ)^2 + m * (-3) - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_l4091_409148


namespace NUMINAMATH_CALUDE_ad_arrangement_count_l4091_409128

-- Define the number of original ads
def original_ads : Nat := 5

-- Define the number of ads to be kept
def kept_ads : Nat := 2

-- Define the number of new ads to be added
def new_ads : Nat := 1

-- Define the number of PSAs to be added
def psas : Nat := 2

-- Define the function to calculate the number of arrangements
def num_arrangements (n m : Nat) : Nat :=
  (n.choose m) * (m + 1) * 2

-- Theorem statement
theorem ad_arrangement_count :
  num_arrangements original_ads kept_ads = 120 :=
by sorry

end NUMINAMATH_CALUDE_ad_arrangement_count_l4091_409128


namespace NUMINAMATH_CALUDE_consecutive_sum_iff_not_power_of_two_l4091_409197

/-- A positive integer n cannot be represented as a sum of two or more consecutive integers
    if and only if n is a power of 2. -/
theorem consecutive_sum_iff_not_power_of_two (n : ℕ) (hn : n > 0) :
  (∃ (k m : ℕ), k ≥ 2 ∧ n = (k * (2 * m + k + 1)) / 2) ↔ ¬(∃ i : ℕ, n = 2^i) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_iff_not_power_of_two_l4091_409197


namespace NUMINAMATH_CALUDE_brownie_cost_l4091_409172

/-- The cost of each brownie at Tamara's bake sale -/
theorem brownie_cost (total_revenue : ℚ) (num_pans : ℕ) (pieces_per_pan : ℕ) 
  (h1 : total_revenue = 32)
  (h2 : num_pans = 2)
  (h3 : pieces_per_pan = 8) :
  total_revenue / (num_pans * pieces_per_pan) = 2 := by
  sorry

end NUMINAMATH_CALUDE_brownie_cost_l4091_409172


namespace NUMINAMATH_CALUDE_gcd_of_special_powers_l4091_409108

theorem gcd_of_special_powers : Nat.gcd (2^2024 - 1) (2^2015 - 1) = 2^9 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_special_powers_l4091_409108


namespace NUMINAMATH_CALUDE_value_of_x_l4091_409165

theorem value_of_x (z y x : ℚ) : z = 48 → y = z / 4 → x = y / 3 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l4091_409165


namespace NUMINAMATH_CALUDE_skateboard_distance_l4091_409117

theorem skateboard_distance (scooter_speed : ℝ) (skateboard_speed_ratio : ℝ) (time_minutes : ℝ) :
  scooter_speed = 50 →
  skateboard_speed_ratio = 2 / 5 →
  time_minutes = 45 →
  skateboard_speed_ratio * scooter_speed * (time_minutes / 60) = 15 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_distance_l4091_409117


namespace NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l4091_409147

open Real

theorem function_inequality_implies_positive_a (a : ℝ) :
  (∃ x₀ ∈ Set.Icc 1 (Real.exp 1), a * (x₀ - 1 / x₀) - 2 * log x₀ > -a / x₀) →
  a > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l4091_409147


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l4091_409193

def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![2 + x, 9; 4 - x, 10]

theorem matrix_not_invertible (x : ℝ) : 
  ¬(IsUnit (A x).det) ↔ x = 16/19 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l4091_409193


namespace NUMINAMATH_CALUDE_inverse_variation_solution_l4091_409121

/-- Inverse variation constant -/
def k : ℝ := 9

/-- The relation between x and y -/
def inverse_variation (x y : ℝ) : Prop := x = k / (y ^ 2)

theorem inverse_variation_solution :
  ∀ x y : ℝ,
  inverse_variation x y →
  inverse_variation 1 3 →
  inverse_variation 0.1111111111111111 y →
  y = 9 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_solution_l4091_409121


namespace NUMINAMATH_CALUDE_base_eight_unique_for_729_l4091_409115

/-- Represents a number in base b with digits d₃d₂d₁d₀ --/
def BaseRepresentation (b : ℕ) (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * b^3 + d₂ * b^2 + d₁ * b + d₀

/-- Checks if a number is in XYXY format --/
def IsXYXY (d₃ d₂ d₁ d₀ : ℕ) : Prop :=
  d₃ = d₁ ∧ d₂ = d₀ ∧ d₃ ≠ d₂

theorem base_eight_unique_for_729 :
  ∃! b : ℕ, 6 ≤ b ∧ b ≤ 9 ∧
    ∃ X Y : ℕ, X ≠ Y ∧
      BaseRepresentation b X Y X Y = 729 ∧
      IsXYXY X Y X Y :=
by sorry

end NUMINAMATH_CALUDE_base_eight_unique_for_729_l4091_409115


namespace NUMINAMATH_CALUDE_zoo_visitors_l4091_409109

/-- Proves that the number of adults who went to the zoo is 51, given the total number of people,
    ticket prices, and total sales. -/
theorem zoo_visitors (total_people : ℕ) (adult_price kid_price : ℕ) (total_sales : ℕ)
    (h_total : total_people = 254)
    (h_adult_price : adult_price = 28)
    (h_kid_price : kid_price = 12)
    (h_sales : total_sales = 3864) :
    ∃ (adults : ℕ), adults = 51 ∧
    ∃ (kids : ℕ), adults + kids = total_people ∧
    adult_price * adults + kid_price * kids = total_sales :=
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l4091_409109


namespace NUMINAMATH_CALUDE_tetrahedron_formation_condition_l4091_409170

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a square with side length 2 -/
structure Square where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Condition for forming a tetrahedron by folding triangles in a square -/
def canFormTetrahedron (s : Square) (x : ℝ) : Prop :=
  let E : Point2D := { x := (s.A.x + s.B.x) / 2, y := (s.A.y + s.B.y) / 2 }
  let F : Point2D := { x := s.B.x + x, y := s.B.y }
  let EA' := 1
  let EF := Real.sqrt (1 + x^2)
  let FA' := 2 - x
  EA' + EF > FA' ∧ EF + FA' > EA' ∧ FA' + EA' > EF

theorem tetrahedron_formation_condition (s : Square) :
  (∀ x, canFormTetrahedron s x ↔ 0 < x ∧ x < 4/3) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_formation_condition_l4091_409170


namespace NUMINAMATH_CALUDE_point_A_coordinates_l4091_409105

/-- A translation that moves any point (a,b) to (a+2,b-6) -/
def translation (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2, p.2 - 6)

/-- The point A₁ after translation -/
def A1 : ℝ × ℝ := (4, -3)

theorem point_A_coordinates :
  ∃ A : ℝ × ℝ, translation A = A1 ∧ A = (2, 3) := by sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l4091_409105


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_one_l4091_409137

theorem at_least_one_not_less_than_one (a b c : ℝ) (h : a + b + c = 3) :
  ¬(a < 1 ∧ b < 1 ∧ c < 1) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_one_l4091_409137


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l4091_409103

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 4) : y = 4 - 3 * x := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l4091_409103


namespace NUMINAMATH_CALUDE_max_min_sum_of_f_l4091_409131

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_min_sum_of_f :
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧
               (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧
               M + m = 2 :=
sorry

end NUMINAMATH_CALUDE_max_min_sum_of_f_l4091_409131


namespace NUMINAMATH_CALUDE_cos_difference_value_l4091_409133

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2)
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_value_l4091_409133


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l4091_409189

theorem complex_fraction_equality : 
  (((4.625 - 13/18 * 9/26) / (9/4) + 2.5 / 1.25 / 6.75) / (1 + 53/68)) /
  ((1/2 - 0.375) / 0.125 + (5/6 - 7/12) / (0.358 - 1.4796 / 13.7)) = 17/27 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l4091_409189


namespace NUMINAMATH_CALUDE_x_range_l4091_409119

theorem x_range (x : ℝ) 
  (h1 : 1 / x < 3) 
  (h2 : 1 / x > -4) 
  (h3 : x^2 - 1 > 0) : 
  x > 1 ∨ x < -1 := by
sorry

end NUMINAMATH_CALUDE_x_range_l4091_409119


namespace NUMINAMATH_CALUDE_joe_journey_time_l4091_409104

/-- Represents the scenario of Joe's journey from home to school -/
structure JourneyScenario where
  d : ℝ  -- Total distance from home to school
  walk_speed : ℝ  -- Joe's walking speed
  run_speed : ℝ  -- Joe's running speed
  walk_time : ℝ  -- Time Joe spent walking
  walk_distance : ℝ  -- Distance Joe walked
  run_distance : ℝ  -- Distance Joe ran

/-- The theorem stating the total time of Joe's journey -/
theorem joe_journey_time (scenario : JourneyScenario) :
  scenario.walk_distance = scenario.d / 3 ∧
  scenario.run_distance = 2 * scenario.d / 3 ∧
  scenario.run_speed = 4 * scenario.walk_speed ∧
  scenario.walk_time = 9 ∧
  scenario.walk_distance = scenario.walk_speed * scenario.walk_time ∧
  scenario.run_distance = scenario.run_speed * (13.5 - scenario.walk_time) →
  13.5 = scenario.walk_time + (scenario.run_distance / scenario.run_speed) :=
by sorry


end NUMINAMATH_CALUDE_joe_journey_time_l4091_409104


namespace NUMINAMATH_CALUDE_average_of_first_four_l4091_409134

theorem average_of_first_four (numbers : Fin 6 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 30)
  (h2 : (numbers 3 + numbers 4 + numbers 5) / 3 = 35)
  (h3 : numbers 3 = 25) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 18.75 := by
sorry

end NUMINAMATH_CALUDE_average_of_first_four_l4091_409134


namespace NUMINAMATH_CALUDE_uncertain_relationship_l4091_409184

/-- Represents a line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- Represents the possible relationships between two lines -/
inductive LineRelationship
  | Parallel
  | Perpendicular
  | Skew

/-- Perpendicularity of two lines -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- The relationship between two lines -/
def relationship (l1 l2 : Line3D) : LineRelationship := sorry

theorem uncertain_relationship 
  (l1 l2 l3 l4 : Line3D) 
  (h_distinct : l1 ≠ l2 ∧ l1 ≠ l3 ∧ l1 ≠ l4 ∧ l2 ≠ l3 ∧ l2 ≠ l4 ∧ l3 ≠ l4)
  (h12 : perpendicular l1 l2)
  (h23 : perpendicular l2 l3)
  (h34 : perpendicular l3 l4) :
  ∃ (r : LineRelationship), relationship l1 l4 = r ∧ 
    (r = LineRelationship.Parallel ∨ 
     r = LineRelationship.Perpendicular ∨ 
     r = LineRelationship.Skew) :=
by sorry

end NUMINAMATH_CALUDE_uncertain_relationship_l4091_409184


namespace NUMINAMATH_CALUDE_japanese_students_fraction_l4091_409161

theorem japanese_students_fraction (j : ℕ) (s : ℕ) : 
  s = 2 * j →
  (3 * s / 8 + j / 4) / (j + s) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_japanese_students_fraction_l4091_409161


namespace NUMINAMATH_CALUDE_equal_spaced_roots_value_l4091_409106

theorem equal_spaced_roots_value (k : ℝ) : 
  (∃ a b c d : ℝ, 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (a^2 - 1) * (a^2 - 4) = k ∧
    (b^2 - 1) * (b^2 - 4) = k ∧
    (c^2 - 1) * (c^2 - 4) = k ∧
    (d^2 - 1) * (d^2 - 4) = k ∧
    b - a = c - b ∧ c - b = d - c) →
  k = 7/4 := by
sorry

end NUMINAMATH_CALUDE_equal_spaced_roots_value_l4091_409106


namespace NUMINAMATH_CALUDE_min_max_values_of_f_l4091_409178

def f (x : ℝ) := -2 * x + 1

theorem min_max_values_of_f :
  ∀ x ∈ Set.Icc 0 5,
    (∃ y ∈ Set.Icc 0 5, f y ≤ f x) ∧
    (∃ z ∈ Set.Icc 0 5, f x ≤ f z) ∧
    f 5 = -9 ∧
    f 0 = 1 ∧
    (∀ w ∈ Set.Icc 0 5, -9 ≤ f w ∧ f w ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_of_f_l4091_409178


namespace NUMINAMATH_CALUDE_tree_planting_problem_l4091_409120

theorem tree_planting_problem (x : ℝ) : 
  (∀ y : ℝ, y = x + 5 → 60 / y = 45 / x) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_problem_l4091_409120


namespace NUMINAMATH_CALUDE_franks_oranges_l4091_409143

/-- Proves that Frank has 1260 oranges given the conditions of the problem -/
theorem franks_oranges :
  let betty_oranges : ℕ := 12
  let sandra_oranges : ℕ := 3 * betty_oranges
  let emily_oranges : ℕ := 7 * sandra_oranges
  let frank_oranges : ℕ := 5 * emily_oranges
  frank_oranges = 1260 := by
  sorry

end NUMINAMATH_CALUDE_franks_oranges_l4091_409143


namespace NUMINAMATH_CALUDE_total_molecular_weight_theorem_l4091_409132

/-- Calculates the total molecular weight of given compounds -/
def totalMolecularWeight (Al_weight S_weight H_weight O_weight C_weight : ℝ) : ℝ :=
  let Al2S3_weight := 2 * Al_weight + 3 * S_weight
  let H2O_weight := 2 * H_weight + O_weight
  let CO2_weight := C_weight + 2 * O_weight
  7 * Al2S3_weight + 5 * H2O_weight + 4 * CO2_weight

/-- The total molecular weight of 7 moles of Al2S3, 5 moles of H2O, and 4 moles of CO2 is 1317.12 grams -/
theorem total_molecular_weight_theorem :
  totalMolecularWeight 26.98 32.06 1.01 16.00 12.01 = 1317.12 := by
  sorry

end NUMINAMATH_CALUDE_total_molecular_weight_theorem_l4091_409132


namespace NUMINAMATH_CALUDE_apple_basket_count_apple_basket_theorem_l4091_409122

theorem apple_basket_count : ℕ → Prop :=
  fun total_apples =>
    (total_apples : ℚ) * (12 : ℚ) / 100 + 66 = total_apples ∧ total_apples = 75

-- Proof
theorem apple_basket_theorem : ∃ n : ℕ, apple_basket_count n := by
  sorry

end NUMINAMATH_CALUDE_apple_basket_count_apple_basket_theorem_l4091_409122


namespace NUMINAMATH_CALUDE_ten_point_circle_chords_l4091_409186

/-- The number of chords that can be drawn between n points on a circle's circumference,
    where no two adjacent points can be connected. -/
def restricted_chords (n : ℕ) : ℕ :=
  Nat.choose n 2 - n

theorem ten_point_circle_chords :
  restricted_chords 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ten_point_circle_chords_l4091_409186


namespace NUMINAMATH_CALUDE_expected_adjacent_face_pairs_standard_deck_l4091_409151

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (face_cards : Nat)

/-- The expected number of pairs of adjacent face cards in a circular deal -/
def expected_adjacent_face_pairs (d : Deck) : Rat :=
  (d.face_cards : Rat) * (d.face_cards - 1) / (d.total_cards - 1)

/-- Theorem: The expected number of pairs of adjacent face cards in a standard 52-card deck dealt in a circle is 44/17 -/
theorem expected_adjacent_face_pairs_standard_deck :
  expected_adjacent_face_pairs ⟨52, 12⟩ = 44 / 17 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_face_pairs_standard_deck_l4091_409151


namespace NUMINAMATH_CALUDE_remainder_x_squared_mod_30_l4091_409174

theorem remainder_x_squared_mod_30 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 30]) 
  (h2 : 7 * x ≡ 13 [ZMOD 30]) : 
  x^2 ≡ 21 [ZMOD 30] := by
  sorry

end NUMINAMATH_CALUDE_remainder_x_squared_mod_30_l4091_409174


namespace NUMINAMATH_CALUDE_square_sum_xy_l4091_409157

theorem square_sum_xy (x y a b : ℝ) 
  (h1 : x * y = b^2) 
  (h2 : 1 / x^2 + 1 / y^2 = a) : 
  (x + y)^2 = a * b^4 + 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l4091_409157


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l4091_409138

theorem imaginary_sum_zero (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l4091_409138


namespace NUMINAMATH_CALUDE_concatenated_seven_digit_divisible_by_239_l4091_409126

/-- Represents a sequence of seven-digit numbers -/
def SevenDigitSequence := List Nat

/-- Concatenates a list of natural numbers -/
def concatenate (seq : SevenDigitSequence) : Nat :=
  seq.foldl (fun acc n => acc * 10000000 + n) 0

/-- The sequence of all seven-digit numbers -/
def allSevenDigitNumbers : SevenDigitSequence :=
  List.range 10000000

theorem concatenated_seven_digit_divisible_by_239 :
  ∃ k : ℕ, concatenate allSevenDigitNumbers = 239 * k :=
sorry

end NUMINAMATH_CALUDE_concatenated_seven_digit_divisible_by_239_l4091_409126


namespace NUMINAMATH_CALUDE_smallest_square_sum_12_consecutive_l4091_409100

/-- The sum of 12 consecutive integers starting from n -/
def sum_12_consecutive (n : ℕ) : ℕ := 6 * (2 * n + 11)

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_square_sum_12_consecutive :
  (∀ n : ℕ, n > 0 → sum_12_consecutive n < 150 → ¬ is_perfect_square (sum_12_consecutive n)) ∧
  is_perfect_square 150 ∧
  (∃ n : ℕ, n > 0 ∧ sum_12_consecutive n = 150) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_sum_12_consecutive_l4091_409100


namespace NUMINAMATH_CALUDE_fraction_subtraction_l4091_409112

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l4091_409112


namespace NUMINAMATH_CALUDE_deadlift_percentage_increase_l4091_409198

/-- Bobby's initial deadlift at age 13 in pounds -/
def initial_deadlift : ℝ := 300

/-- Bobby's annual deadlift increase in pounds -/
def annual_increase : ℝ := 110

/-- Number of years between age 13 and 18 -/
def years : ℕ := 5

/-- Bobby's deadlift at age 18 in pounds -/
def deadlift_at_18 : ℝ := initial_deadlift + (annual_increase * years)

/-- The percentage increase we're looking for -/
def P : ℝ := sorry

/-- Theorem stating the relationship between Bobby's deadlift at 18 and the percentage increase -/
theorem deadlift_percentage_increase : deadlift_at_18 * (1 + P / 100) = deadlift_at_18 + 100 := by
  sorry

end NUMINAMATH_CALUDE_deadlift_percentage_increase_l4091_409198


namespace NUMINAMATH_CALUDE_probability_of_target_plate_l4091_409144

structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char

def vowels : List Char := ['A', 'E', 'I', 'O', 'U', 'Y']
def non_vowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z']
def hex_digits : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']

def is_valid_plate (plate : LicensePlate) : Prop :=
  plate.first ∈ vowels ∧
  plate.second ∈ non_vowels ∧
  plate.third ∈ non_vowels ∧
  plate.second ≠ plate.third ∧
  plate.fourth ∈ hex_digits

def total_valid_plates : ℕ := vowels.length * non_vowels.length * (non_vowels.length - 1) * hex_digits.length

def target_plate : LicensePlate := ⟨'E', 'Y', 'B', '5'⟩

theorem probability_of_target_plate :
  (1 : ℚ) / total_valid_plates = 1 / 44352 :=
sorry

end NUMINAMATH_CALUDE_probability_of_target_plate_l4091_409144


namespace NUMINAMATH_CALUDE_worker_count_l4091_409124

theorem worker_count (total_money : ℤ) (num_workers : ℤ) 
  (h1 : total_money - 5 * num_workers = 30)
  (h2 : total_money - 7 * num_workers = -30) : 
  num_workers = 30 := by
sorry

end NUMINAMATH_CALUDE_worker_count_l4091_409124


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l4091_409187

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 25 := by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ (2 / a₀ + 3 / b₀) = 25 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l4091_409187


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l4091_409182

theorem radical_conjugate_sum_product (a b : ℝ) :
  (a + Real.sqrt b) + (a - Real.sqrt b) = 0 ∧
  (a + Real.sqrt b) * (a - Real.sqrt b) = 4 →
  a + b = -4 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l4091_409182


namespace NUMINAMATH_CALUDE_max_cookies_in_class_l4091_409139

/-- Represents the maximum number of cookies one student could have taken in a class. -/
def max_cookies_for_one_student (num_students : ℕ) (avg_cookies : ℕ) (min_cookies : ℕ) : ℕ :=
  num_students * avg_cookies - (num_students - 1) * min_cookies

/-- Theorem stating the maximum number of cookies one student could have taken. -/
theorem max_cookies_in_class (num_students : ℕ) (avg_cookies : ℕ) (min_cookies : ℕ)
    (h_num_students : num_students = 20)
    (h_avg_cookies : avg_cookies = 6)
    (h_min_cookies : min_cookies = 2) :
    max_cookies_for_one_student num_students avg_cookies min_cookies = 82 := by
  sorry

#eval max_cookies_for_one_student 20 6 2

end NUMINAMATH_CALUDE_max_cookies_in_class_l4091_409139


namespace NUMINAMATH_CALUDE_calories_in_box_is_1600_l4091_409194

/-- Represents the number of cookies in a bag -/
def cookies_per_bag : ℕ := 20

/-- Represents the number of bags in a box -/
def bags_per_box : ℕ := 4

/-- Represents the number of calories in a cookie -/
def calories_per_cookie : ℕ := 20

/-- Calculates the total number of calories in a box of cookies -/
def total_calories_in_box : ℕ := cookies_per_bag * bags_per_box * calories_per_cookie

/-- Theorem stating that the total calories in a box of cookies is 1600 -/
theorem calories_in_box_is_1600 : total_calories_in_box = 1600 := by
  sorry

end NUMINAMATH_CALUDE_calories_in_box_is_1600_l4091_409194


namespace NUMINAMATH_CALUDE_quadratic_negative_value_condition_l4091_409116

/-- Given a quadratic function f(x) = x^2 + mx + 1, 
    this theorem states that there exists a positive x₀ such that f(x₀) < 0 
    if and only if m < -2 -/
theorem quadratic_negative_value_condition (m : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + m*x₀ + 1 < 0) ↔ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_negative_value_condition_l4091_409116


namespace NUMINAMATH_CALUDE_sum_of_even_factors_720_l4091_409191

def sum_of_even_factors (n : ℕ) : ℕ := sorry

theorem sum_of_even_factors_720 : sum_of_even_factors 720 = 2340 := by sorry

end NUMINAMATH_CALUDE_sum_of_even_factors_720_l4091_409191


namespace NUMINAMATH_CALUDE_equation_solutions_l4091_409171

noncomputable def solution_equation (a b c d x : ℝ) : Prop :=
  (a*x + b) / (a + b*x) + (c*x + d) / (c + d*x) = 
  (a*x - b) / (a - b*x) + (c*x - d) / (c - d*x)

theorem equation_solutions 
  (a b c d : ℝ) 
  (h1 : a*d + b*c ≠ 0) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : d ≠ 0) :
  (∀ x : ℝ, x ≠ a/b ∧ x ≠ -a/b ∧ x ≠ c/d ∧ x ≠ -c/d →
    (x = 1 ∨ x = -1 ∨ x = Real.sqrt (a*c/(b*d)) ∨ x = -Real.sqrt (a*c/(b*d))) ↔ 
    solution_equation a b c d x) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l4091_409171


namespace NUMINAMATH_CALUDE_total_commission_is_42000_l4091_409195

def sale_price : ℝ := 10000
def commission_rate_first_100 : ℝ := 0.03
def commission_rate_after_100 : ℝ := 0.04
def total_machines_sold : ℕ := 130

def commission_first_100 : ℝ := 100 * sale_price * commission_rate_first_100
def commission_after_100 : ℝ := (total_machines_sold - 100) * sale_price * commission_rate_after_100

theorem total_commission_is_42000 :
  commission_first_100 + commission_after_100 = 42000 := by
  sorry

end NUMINAMATH_CALUDE_total_commission_is_42000_l4091_409195


namespace NUMINAMATH_CALUDE_garden_length_proof_l4091_409140

/-- Represents a rectangular garden with its dimensions. -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden. -/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.breadth)

theorem garden_length_proof (g : RectangularGarden) 
  (h1 : perimeter g = 600) 
  (h2 : g.breadth = 95) : 
  g.length = 205 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_proof_l4091_409140


namespace NUMINAMATH_CALUDE_polynomial_expansion_l4091_409176

theorem polynomial_expansion (x : ℝ) : 
  (5 * x - 3) * (2 * x^3 + 7 * x - 1) = 10 * x^4 - 6 * x^3 + 35 * x^2 - 26 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l4091_409176


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l4091_409118

/-- The hyperbola equation -/
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x + 3*y - 1 = 0

/-- The condition for intersection based on slope comparison -/
def intersection_condition (b : ℝ) : Prop := b > 2/3

/-- The theorem stating that b > 1 is sufficient but not necessary for intersection -/
theorem hyperbola_line_intersection (b : ℝ) (h : b > 0) :
  (∀ x y, hyperbola x y b → line x y → intersection_condition b) ∧
  ¬(∀ b, intersection_condition b → b > 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l4091_409118


namespace NUMINAMATH_CALUDE_line_not_in_plane_necessary_not_sufficient_l4091_409130

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- State the theorem
theorem line_not_in_plane_necessary_not_sufficient
  (a b : Line) (α : Plane)
  (h : contained_in a α) :
  (¬ contained_in b α ∧ ¬ (∀ b, ¬ contained_in b α → skew a b)) ∧
  (∀ b, skew a b → ¬ contained_in b α) := by
sorry

end NUMINAMATH_CALUDE_line_not_in_plane_necessary_not_sufficient_l4091_409130


namespace NUMINAMATH_CALUDE_deductive_reasoning_not_always_correct_l4091_409167

/-- Represents a deductive argument --/
structure DeductiveArgument where
  premises : List Prop
  conclusion : Prop

/-- Represents the form of a deductive argument --/
structure DeductiveForm where
  form : DeductiveArgument → Prop

/-- Defines when a deductive argument conforms to a deductive form --/
def conformsToForm (arg : DeductiveArgument) (form : DeductiveForm) : Prop :=
  form.form arg

/-- Defines when a deductive argument is valid --/
def isValid (arg : DeductiveArgument) : Prop :=
  ∀ (form : DeductiveForm), conformsToForm arg form → arg.conclusion

/-- Theorem: A deductive argument that conforms to a deductive form is not always valid --/
theorem deductive_reasoning_not_always_correct :
  ∃ (arg : DeductiveArgument) (form : DeductiveForm),
    conformsToForm arg form ∧ ¬isValid arg := by
  sorry

end NUMINAMATH_CALUDE_deductive_reasoning_not_always_correct_l4091_409167


namespace NUMINAMATH_CALUDE_relationship_abcd_l4091_409155

theorem relationship_abcd :
  let a : ℝ := 10 / 7
  let b : ℝ := Real.log 3
  let c : ℝ := 2 * Real.sqrt 3 / 3
  let d : ℝ := Real.exp 0.3
  a > d ∧ d > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_relationship_abcd_l4091_409155


namespace NUMINAMATH_CALUDE_smallest_b_for_composite_polynomial_l4091_409168

theorem smallest_b_for_composite_polynomial : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), ¬ Nat.Prime (x^4 + x^3 + b^2 + 5).natAbs) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ∃ (x : ℤ), Nat.Prime (x^4 + x^3 + b'^2 + 5).natAbs) ∧
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_for_composite_polynomial_l4091_409168


namespace NUMINAMATH_CALUDE_solve_for_C_l4091_409156

theorem solve_for_C : ∃ C : ℤ, 80 - (5 - (6 + 2 * (7 - C - 5))) = 89 ∧ C = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_C_l4091_409156


namespace NUMINAMATH_CALUDE_equal_tire_usage_l4091_409145

/-- Represents the usage of tires on a car -/
structure TireUsage where
  total_tires : ℕ
  active_tires : ℕ
  total_miles : ℕ
  tire_miles : ℕ

/-- Theorem stating the correct tire usage for the given scenario -/
theorem equal_tire_usage (usage : TireUsage) 
  (h1 : usage.total_tires = 5)
  (h2 : usage.active_tires = 4)
  (h3 : usage.total_miles = 45000)
  (h4 : usage.tire_miles = usage.total_miles * usage.active_tires / usage.total_tires) :
  usage.tire_miles = 36000 := by
  sorry

#check equal_tire_usage

end NUMINAMATH_CALUDE_equal_tire_usage_l4091_409145


namespace NUMINAMATH_CALUDE_handshake_arrangement_remainder_l4091_409173

/-- The number of people in the group -/
def n : ℕ := 10

/-- The number of handshakes each person makes -/
def k : ℕ := 3

/-- Two arrangements are considered different if at least two people who shake hands
    in one arrangement don't in the other -/
def different_arrangement : Prop := sorry

/-- The total number of possible handshaking arrangements -/
def M : ℕ := sorry

/-- The theorem stating the main result -/
theorem handshake_arrangement_remainder :
  M % 500 = 84 := by sorry

end NUMINAMATH_CALUDE_handshake_arrangement_remainder_l4091_409173


namespace NUMINAMATH_CALUDE_wind_velocity_problem_l4091_409188

/-- Represents the relationship between pressure, area, and velocity -/
def pressure_relation (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^3

theorem wind_velocity_problem (k : ℝ) :
  (pressure_relation k 1 8 = 1) →
  (pressure_relation k 9 12 = 27) :=
by sorry

end NUMINAMATH_CALUDE_wind_velocity_problem_l4091_409188


namespace NUMINAMATH_CALUDE_running_track_area_l4091_409110

theorem running_track_area (r : ℝ) (w : ℝ) (h1 : r = 50) (h2 : w = 3) :
  π * ((r + w)^2 - r^2) = 309 * π := by
  sorry

end NUMINAMATH_CALUDE_running_track_area_l4091_409110


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l4091_409127

theorem product_of_three_numbers (a b c m : ℚ) : 
  a + b + c = 200 ∧ 
  8 * a = m ∧ 
  b = m + 10 ∧ 
  c = m - 10 →
  a * b * c = 505860000 / 4913 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l4091_409127


namespace NUMINAMATH_CALUDE_davids_math_marks_l4091_409177

/-- Calculates David's marks in Mathematics given his marks in other subjects and the average --/
theorem davids_math_marks (english physics chemistry biology : ℕ) (average : ℕ) (h1 : english = 86) (h2 : physics = 82) (h3 : chemistry = 87) (h4 : biology = 85) (h5 : average = 85) :
  (english + physics + chemistry + biology + (5 * average - (english + physics + chemistry + biology))) / 5 = average :=
sorry

end NUMINAMATH_CALUDE_davids_math_marks_l4091_409177


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l4091_409114

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 - 2*x + 3 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l4091_409114


namespace NUMINAMATH_CALUDE_expression_value_l4091_409129

theorem expression_value : 
  Real.sqrt ((16^12 + 8^15) / (16^5 + 8^16)) = (3 * Real.sqrt 2) / 4 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l4091_409129


namespace NUMINAMATH_CALUDE_inequality_theorem_l4091_409185

theorem inequality_theorem :
  (∀ (x y z : ℝ), x^2 + 2*y^2 + 3*z^2 ≥ Real.sqrt 3 * (x*y + y*z + z*x)) ∧
  (∃ (k : ℝ), k > Real.sqrt 3 ∧ ∀ (x y z : ℝ), x^2 + 2*y^2 + 3*z^2 ≥ k * (x*y + y*z + z*x)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l4091_409185


namespace NUMINAMATH_CALUDE_eventually_all_play_all_l4091_409113

/-- Represents a player in the tournament -/
inductive Player
  | Mathematician (id : ℕ)
  | Humanitarian (id : ℕ)

/-- Represents the state of the tournament -/
structure TournamentState where
  n : ℕ  -- number of humanities students
  m : ℕ  -- number of mathematicians
  queue : List Player
  table : Player × Player
  h_different_sizes : n ≠ m

/-- Represents a game played between two players -/
def Game := Player × Player

/-- Simulates the tournament for a given number of steps -/
def simulateTournament (initial : TournamentState) (steps : ℕ) : List Game := sorry

/-- Checks if all mathematicians have played with all humanitarians -/
def allPlayedAgainstAll (games : List Game) : Prop := sorry

/-- The main theorem stating that eventually all mathematicians will play against all humanitarians -/
theorem eventually_all_play_all (initial : TournamentState) :
  ∃ k : ℕ, allPlayedAgainstAll (simulateTournament initial k) := by
  sorry

end NUMINAMATH_CALUDE_eventually_all_play_all_l4091_409113


namespace NUMINAMATH_CALUDE_jack_journey_time_l4091_409154

/-- Represents the time spent in a country during Jack's journey --/
structure CountryTime where
  customs : ℕ
  quarantine_days : ℕ

/-- Represents a layover during Jack's journey --/
structure Layover where
  duration : ℕ

/-- Calculates the total time spent in a country in hours --/
def total_country_time (ct : CountryTime) : ℕ :=
  ct.customs + ct.quarantine_days * 24

/-- Calculates the total time of Jack's journey in hours --/
def total_journey_time (canada : CountryTime) (australia : CountryTime) (japan : CountryTime)
                       (to_australia : Layover) (to_japan : Layover) : ℕ :=
  total_country_time canada + total_country_time australia + total_country_time japan +
  to_australia.duration + to_japan.duration

theorem jack_journey_time :
  let canada : CountryTime := ⟨20, 14⟩
  let australia : CountryTime := ⟨15, 10⟩
  let japan : CountryTime := ⟨10, 7⟩
  let to_australia : Layover := ⟨12⟩
  let to_japan : Layover := ⟨5⟩
  total_journey_time canada australia japan to_australia to_japan = 806 :=
by sorry

end NUMINAMATH_CALUDE_jack_journey_time_l4091_409154


namespace NUMINAMATH_CALUDE_perfect_squares_existence_l4091_409136

theorem perfect_squares_existence (a : ℕ) (h1 : Odd a) (h2 : a > 17) 
  (h3 : ∃ x : ℕ, 3 * a - 2 = x^2) : 
  ∃ b c : ℕ, b ≠ c ∧ b > 0 ∧ c > 0 ∧ 
    (∃ w x y z : ℕ, a + b = w^2 ∧ a + c = x^2 ∧ b + c = y^2 ∧ a + b + c = z^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_existence_l4091_409136


namespace NUMINAMATH_CALUDE_algorithms_should_be_simple_and_convenient_l4091_409142

/-- Represents an algorithm --/
structure Algorithm where
  steps : List String
  task : String

/-- Characteristic of an algorithm being simple and convenient --/
def is_simple_and_convenient (a : Algorithm) : Prop := sorry

/-- Theorem stating that algorithms should be designed to be simple and convenient --/
theorem algorithms_should_be_simple_and_convenient : 
  ∀ (a : Algorithm), is_simple_and_convenient a :=
sorry

end NUMINAMATH_CALUDE_algorithms_should_be_simple_and_convenient_l4091_409142


namespace NUMINAMATH_CALUDE_power_two_geq_square_l4091_409149

theorem power_two_geq_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_geq_square_l4091_409149


namespace NUMINAMATH_CALUDE_square_sum_simplification_l4091_409146

theorem square_sum_simplification : 99^2 + 202 * 99 + 101^2 = 40000 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_simplification_l4091_409146


namespace NUMINAMATH_CALUDE_smallest_sum_of_exponents_l4091_409181

theorem smallest_sum_of_exponents (m n : ℕ+) (h1 : m > n) 
  (h2 : 2012^(m.val) % 1000 = 2012^(n.val) % 1000) : 
  ∃ (k l : ℕ+), k.val + l.val = 104 ∧ k > l ∧ 
  2012^(k.val) % 1000 = 2012^(l.val) % 1000 ∧
  ∀ (p q : ℕ+), p > q → 2012^(p.val) % 1000 = 2012^(q.val) % 1000 → 
  p.val + q.val ≥ 104 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_exponents_l4091_409181


namespace NUMINAMATH_CALUDE_triangle_angle_inequalities_l4091_409190

theorem triangle_angle_inequalities (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) : 
  (Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) ≤ 1/8) ∧ 
  (Real.cos α * Real.cos β * Real.cos γ ≤ 1/8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequalities_l4091_409190


namespace NUMINAMATH_CALUDE_f_min_value_l4091_409166

/-- The function f(x) defined as (x+1)(x+2)(x+3)(x+4) + 35 -/
def f (x : ℝ) : ℝ := (x+1)*(x+2)*(x+3)*(x+4) + 35

/-- Theorem stating that the minimum value of f(x) is 34 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ 34 ∧ ∃ x₀ : ℝ, f x₀ = 34 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l4091_409166


namespace NUMINAMATH_CALUDE_inverse_of_three_mod_243_l4091_409159

theorem inverse_of_three_mod_243 : ∃ x : ℕ, x < 243 ∧ (3 * x) % 243 = 1 :=
by
  use 324
  sorry

end NUMINAMATH_CALUDE_inverse_of_three_mod_243_l4091_409159


namespace NUMINAMATH_CALUDE_externally_tangent_circles_l4091_409199

theorem externally_tangent_circles (m : ℝ) : 
  let C₁ := {(x, y) : ℝ × ℝ | (x - m)^2 + (y + 2)^2 = 9}
  let C₂ := {(x, y) : ℝ × ℝ | (x + 1)^2 + (y - m)^2 = 4}
  (∃ (p : ℝ × ℝ), p ∈ C₁ ∧ p ∈ C₂ ∧ 
    (∀ (q : ℝ × ℝ), q ∈ C₁ ∧ q ∈ C₂ → q = p) ∧
    (∀ (r : ℝ × ℝ), r ∈ C₁ → ∃ (s : ℝ × ℝ), s ∈ C₂ ∧ s ≠ r)) →
  m = 2 ∨ m = -5 :=
by sorry

end NUMINAMATH_CALUDE_externally_tangent_circles_l4091_409199


namespace NUMINAMATH_CALUDE_average_string_length_l4091_409135

theorem average_string_length :
  let string1 : ℝ := 2
  let string2 : ℝ := 5
  let string3 : ℝ := 3
  let total_length : ℝ := string1 + string2 + string3
  let num_strings : ℕ := 3
  (total_length / num_strings) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_string_length_l4091_409135


namespace NUMINAMATH_CALUDE_mike_passing_percentage_l4091_409150

/-- The percentage Mike needs to pass, given his score, shortfall, and maximum possible marks. -/
theorem mike_passing_percentage
  (mike_score : ℕ)
  (shortfall : ℕ)
  (max_marks : ℕ)
  (h1 : mike_score = 212)
  (h2 : shortfall = 16)
  (h3 : max_marks = 760) :
  (((mike_score + shortfall : ℚ) / max_marks) * 100 : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_mike_passing_percentage_l4091_409150


namespace NUMINAMATH_CALUDE_smaller_angle_measure_l4091_409153

/-- A parallelogram with one angle exceeding the other by 70 degrees -/
structure SpecialParallelogram where
  /-- The measure of the smaller angle in degrees -/
  smaller_angle : ℝ
  /-- The measure of the larger angle in degrees -/
  larger_angle : ℝ
  /-- The larger angle exceeds the smaller angle by 70 degrees -/
  angle_difference : larger_angle = smaller_angle + 70
  /-- The sum of adjacent angles is 180 degrees -/
  angle_sum : smaller_angle + larger_angle = 180

/-- The measure of the smaller angle in a special parallelogram is 55 degrees -/
theorem smaller_angle_measure (p : SpecialParallelogram) : p.smaller_angle = 55 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_measure_l4091_409153


namespace NUMINAMATH_CALUDE_line_parallel_contained_in_plane_l4091_409158

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_contained_in_plane 
  (a b : Line) (α : Plane) :
  parallel a b → containedIn b α → 
  parallelToPlane a α ∨ containedIn a α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_contained_in_plane_l4091_409158


namespace NUMINAMATH_CALUDE_factorial_100_trailing_zeros_l4091_409101

-- Define a function to count trailing zeros in a factorial
def trailingZeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem factorial_100_trailing_zeros :
  trailingZeros 100 = 24 := by sorry

end NUMINAMATH_CALUDE_factorial_100_trailing_zeros_l4091_409101


namespace NUMINAMATH_CALUDE_down_payment_calculation_l4091_409162

def cash_price : ℕ := 400
def monthly_payment : ℕ := 30
def num_months : ℕ := 12
def cash_savings : ℕ := 80

theorem down_payment_calculation : 
  cash_price + cash_savings - monthly_payment * num_months = 120 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_calculation_l4091_409162


namespace NUMINAMATH_CALUDE_roboducks_order_l4091_409180

theorem roboducks_order (shelves_percentage : ℚ) (storage_count : ℕ) : 
  shelves_percentage = 30 / 100 →
  storage_count = 140 →
  ∃ total : ℕ, total = 200 ∧ (1 - shelves_percentage) * total = storage_count :=
by
  sorry

end NUMINAMATH_CALUDE_roboducks_order_l4091_409180


namespace NUMINAMATH_CALUDE_not_all_even_P_true_l4091_409152

-- Define the proposition P
def P : ℕ → Prop := sorry

-- Theorem statement
theorem not_all_even_P_true :
  (∀ n : ℕ, n ≤ 1000 → P (2 * n)) →  -- P holds for n = 1, 2, ..., 1000
  P (2 * 1001) →                     -- P holds for n = 1001
  ¬ (∀ k : ℕ, Even k → P k) :=       -- It's not true that P holds for all even k
by
  sorry

end NUMINAMATH_CALUDE_not_all_even_P_true_l4091_409152


namespace NUMINAMATH_CALUDE_special_sequence_eventually_periodic_l4091_409192

/-- A sequence of positive integers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (a 1 < a 2) ∧
  (∀ n ≥ 3,
    (a n > a (n-1)) ∧
    (∃! (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n-1 ∧ a n = a i + a j) ∧
    (∀ m < n, (∃ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ m-1 ∧ a m = a i + a j) → a n > a m))

/-- The set of even numbers in the sequence is finite -/
def FinitelyManyEven (a : ℕ → ℕ) : Prop :=
  ∃ (S : Finset ℕ), ∀ n, Even (a n) → n ∈ S

/-- The sequence of differences is eventually periodic -/
def EventuallyPeriodic (s : ℕ → ℕ) : Prop :=
  ∃ (k p : ℕ), p > 0 ∧ ∀ n ≥ k, s (n + p) = s n

/-- The main theorem -/
theorem special_sequence_eventually_periodic (a : ℕ → ℕ)
  (h1 : SpecialSequence a) (h2 : FinitelyManyEven a) :
  EventuallyPeriodic (fun n => a (n+1) - a n) :=
sorry

end NUMINAMATH_CALUDE_special_sequence_eventually_periodic_l4091_409192


namespace NUMINAMATH_CALUDE_twelve_by_twelve_grid_intersections_l4091_409196

/-- Represents a square grid -/
structure SquareGrid :=
  (n : ℕ)

/-- Number of interior vertical or horizontal lines in a square grid -/
def interior_lines (g : SquareGrid) : ℕ := g.n - 1

/-- Number of interior intersection points in a square grid -/
def interior_intersections (g : SquareGrid) : ℕ :=
  (interior_lines g) * (interior_lines g)

/-- Theorem: The number of interior intersection points on a 12 by 12 grid of squares is 121 -/
theorem twelve_by_twelve_grid_intersections :
  ∃ (g : SquareGrid), g.n = 12 ∧ interior_intersections g = 121 := by
  sorry

end NUMINAMATH_CALUDE_twelve_by_twelve_grid_intersections_l4091_409196


namespace NUMINAMATH_CALUDE_exponential_function_inequality_range_l4091_409164

/-- The range of k for which the given inequality holds for all real x₁ and x₂ -/
theorem exponential_function_inequality_range :
  ∀ (k : ℝ), 
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → 
    |((Real.exp x₁) - (Real.exp x₂)) / (x₁ - x₂)| < |k| * ((Real.exp x₁) + (Real.exp x₂))) 
  ↔ 
  (k ≤ -1/2 ∨ k ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_inequality_range_l4091_409164


namespace NUMINAMATH_CALUDE_methodC_cannot_eliminate_variables_l4091_409141

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := 5 * x + 2 * y = 4
def equation2 (x y : ℝ) : Prop := 2 * x - 3 * y = 10

-- Define the method C
def methodC (x y : ℝ) : Prop := 1.5 * (5 * x + 2 * y) - (2 * x - 3 * y) = 1.5 * 4 - 10

-- Theorem stating that method C cannot eliminate variables
theorem methodC_cannot_eliminate_variables :
  ∀ x y : ℝ, methodC x y ↔ (5.5 * x + 6 * y = -4) :=
by sorry

end NUMINAMATH_CALUDE_methodC_cannot_eliminate_variables_l4091_409141


namespace NUMINAMATH_CALUDE_range_of_f_l4091_409169

def f (x : ℝ) := |x + 5| - |x - 3|

theorem range_of_f :
  ∀ y ∈ Set.range f, -8 ≤ y ∧ y ≤ 8 ∧
  ∀ z, -8 ≤ z ∧ z ≤ 8 → ∃ x, f x = z :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l4091_409169


namespace NUMINAMATH_CALUDE_probability_club_after_removal_l4091_409111

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)

/-- Represents the deck after removing spade cards -/
structure ModifiedDeck :=
  (remaining_cards : ℕ)
  (club_cards : ℕ)

/-- The probability of drawing a club card from the modified deck -/
def probability_club (d : ModifiedDeck) : ℚ :=
  d.club_cards / d.remaining_cards

theorem probability_club_after_removal (standard_deck : Deck) (modified_deck : ModifiedDeck) :
  standard_deck.total_cards = 52 →
  standard_deck.ranks = 13 →
  standard_deck.suits = 4 →
  modified_deck.remaining_cards = 48 →
  modified_deck.club_cards = 13 →
  probability_club modified_deck = 13 / 48 := by
  sorry

#eval (13 : ℚ) / 48

end NUMINAMATH_CALUDE_probability_club_after_removal_l4091_409111


namespace NUMINAMATH_CALUDE_representatives_count_l4091_409163

/-- The number of ways to select representatives from a group with females and males -/
def select_representatives (num_females num_males num_representatives : ℕ) : ℕ :=
  Nat.choose num_females 1 * Nat.choose num_males 2 + 
  Nat.choose num_females 2 * Nat.choose num_males 1

/-- Theorem stating that selecting 3 representatives from 3 females and 4 males, 
    with at least one of each, results in 30 different ways -/
theorem representatives_count : select_representatives 3 4 3 = 30 := by
  sorry

#eval select_representatives 3 4 3

end NUMINAMATH_CALUDE_representatives_count_l4091_409163


namespace NUMINAMATH_CALUDE_texas_california_plate_difference_l4091_409160

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^3 * num_digits^3

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_letters^3 * num_digits^4

/-- The difference in the number of possible license plates between Texas and California -/
def plate_difference : ℕ := texas_plates - california_plates

theorem texas_california_plate_difference :
  plate_difference = 158184000 := by
  sorry

end NUMINAMATH_CALUDE_texas_california_plate_difference_l4091_409160


namespace NUMINAMATH_CALUDE_sum_of_remaining_numbers_l4091_409125

theorem sum_of_remaining_numbers
  (n : ℕ)
  (total_sum : ℝ)
  (subset_sum : ℝ)
  (h1 : n = 5)
  (h2 : total_sum / n = 20)
  (h3 : subset_sum / 2 = 26) :
  total_sum - subset_sum = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remaining_numbers_l4091_409125


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4091_409183

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4091_409183


namespace NUMINAMATH_CALUDE_proportionality_coefficient_l4091_409179

/-- Given variables x, y, z and a positive integer k, satisfying the following conditions:
    1. z - y = k * x
    2. x - z = k * y
    3. z = 5/3 * (x - y)
    Prove that k = 3 -/
theorem proportionality_coefficient (x y z : ℝ) (k : ℕ+) 
  (h1 : z - y = k * x)
  (h2 : x - z = k * y)
  (h3 : z = 5/3 * (x - y)) :
  k = 3 := by sorry

end NUMINAMATH_CALUDE_proportionality_coefficient_l4091_409179


namespace NUMINAMATH_CALUDE_coin_toss_heads_l4091_409123

theorem coin_toss_heads (total_tosses : ℕ) (tail_count : ℕ) (head_count : ℕ) :
  total_tosses = 10 →
  tail_count = 7 →
  head_count = total_tosses - tail_count →
  head_count = 3 := by
sorry

end NUMINAMATH_CALUDE_coin_toss_heads_l4091_409123


namespace NUMINAMATH_CALUDE_remainder_problem_l4091_409175

theorem remainder_problem (n : ℕ) (a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : b < 102) 
  (h3 : n = 103 * c + d) 
  (h4 : d < 103) 
  (h5 : a + d = 20) : 
  b = 20 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l4091_409175


namespace NUMINAMATH_CALUDE_percent_relation_l4091_409102

theorem percent_relation (x y z w : ℝ) 
  (hx : x = 1.2 * y) 
  (hy : y = 0.7 * z) 
  (hw : w = 1.5 * z) : 
  x / w = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l4091_409102
