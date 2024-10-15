import Mathlib

namespace NUMINAMATH_CALUDE_chocolate_distribution_l1251_125129

theorem chocolate_distribution (total : ℕ) (michael_share : ℕ) (paige_share : ℕ) (mandy_share : ℕ) : 
  total = 60 →
  michael_share = total / 2 →
  paige_share = (total - michael_share) / 2 →
  mandy_share = total - michael_share - paige_share →
  mandy_share = 15 := by
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l1251_125129


namespace NUMINAMATH_CALUDE_fruit_basket_count_l1251_125196

/-- The number of fruit baskets -/
def num_baskets : ℕ := 4

/-- The number of apples in each of the first three baskets -/
def apples_per_basket : ℕ := 9

/-- The number of oranges in each of the first three baskets -/
def oranges_per_basket : ℕ := 15

/-- The number of bananas in each of the first three baskets -/
def bananas_per_basket : ℕ := 14

/-- The number of fruits reduced in the fourth basket -/
def reduction : ℕ := 2

/-- The total number of fruits in all baskets -/
def total_fruits : ℕ := 146

theorem fruit_basket_count :
  (3 * (apples_per_basket + oranges_per_basket + bananas_per_basket)) +
  ((apples_per_basket - reduction) + (oranges_per_basket - reduction) + (bananas_per_basket - reduction)) = total_fruits :=
by sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l1251_125196


namespace NUMINAMATH_CALUDE_infinitely_many_coprime_binomials_l1251_125143

theorem infinitely_many_coprime_binomials (k l : ℕ+) :
  ∃ (S : Set ℕ), (∀ (m : ℕ), m ∈ S → m ≥ k) ∧
                 (Set.Infinite S) ∧
                 (∀ (m : ℕ), m ∈ S → Nat.gcd (Nat.choose m k) l = 1) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_coprime_binomials_l1251_125143


namespace NUMINAMATH_CALUDE_solution_satisfies_conditions_l1251_125132

-- Define the function y(x)
def y (x : ℝ) : ℝ := (x + 1)^2

-- State the theorem
theorem solution_satisfies_conditions :
  (∀ x, (deriv^[2] y) x = 2) ∧ 
  y 0 = 1 ∧ 
  (deriv y) 0 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_solution_satisfies_conditions_l1251_125132


namespace NUMINAMATH_CALUDE_binomial_sum_problem_l1251_125197

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

theorem binomial_sum_problem : 
  (binomial 8 5 + binomial 100 98 * binomial 7 7 = 5006) ∧ 
  (binomial 5 0 + binomial 5 1 + binomial 5 2 + binomial 5 3 + binomial 5 4 + binomial 5 5 = 32) :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_problem_l1251_125197


namespace NUMINAMATH_CALUDE_round_trip_distance_boy_school_distance_l1251_125140

/-- Calculates the distance between two points given the speeds and total time of a round trip -/
theorem round_trip_distance (outbound_speed return_speed : ℝ) (total_time : ℝ) : 
  outbound_speed > 0 → return_speed > 0 → total_time > 0 →
  (1 / outbound_speed + 1 / return_speed) * (outbound_speed * return_speed * total_time / (outbound_speed + return_speed)) = total_time := by
  sorry

/-- The distance between the boy's house and school -/
theorem boy_school_distance : 
  let outbound_speed : ℝ := 3
  let return_speed : ℝ := 2
  let total_time : ℝ := 5
  (outbound_speed * return_speed * total_time) / (outbound_speed + return_speed) = 6 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_boy_school_distance_l1251_125140


namespace NUMINAMATH_CALUDE_batsman_average_proof_l1251_125114

/-- Calculates the average runs for a batsman given two sets of matches with different averages -/
def calculateAverageRuns (matches1 : ℕ) (average1 : ℚ) (matches2 : ℕ) (average2 : ℚ) : ℚ :=
  ((matches1 : ℚ) * average1 + (matches2 : ℚ) * average2) / ((matches1 + matches2) : ℚ)

/-- Theorem: Given a batsman's performance in two sets of matches, prove the overall average -/
theorem batsman_average_proof (matches1 matches2 : ℕ) (average1 average2 : ℚ) :
  matches1 = 20 ∧ matches2 = 10 ∧ average1 = 30 ∧ average2 = 15 →
  calculateAverageRuns matches1 average1 matches2 average2 = 25 := by
  sorry

#eval calculateAverageRuns 20 30 10 15

end NUMINAMATH_CALUDE_batsman_average_proof_l1251_125114


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l1251_125161

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4*b = 8) :
  ab ≤ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4*b₀ = 8 ∧ a₀*b₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l1251_125161


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_with_square_root_l1251_125157

theorem unique_solution_quadratic_with_square_root :
  ∃! x : ℝ, x^2 + 6*x + 6*x * Real.sqrt (x + 4) = 31 :=
by
  -- The unique solution is (11 - 3√5) / 2
  use (11 - 3 * Real.sqrt 5) / 2
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_with_square_root_l1251_125157


namespace NUMINAMATH_CALUDE_cut_square_theorem_l1251_125151

/-- Represents the dimensions of the original square -/
def original_size : ℕ := 8

/-- Represents the total length of cuts -/
def total_cut_length : ℕ := 54

/-- Represents the width of a rectangular piece -/
def rect_width : ℕ := 1

/-- Represents the length of a rectangular piece -/
def rect_length : ℕ := 4

/-- Represents the side length of a square piece -/
def square_side : ℕ := 2

/-- Represents the perimeter of the original square -/
def original_perimeter : ℕ := 4 * original_size

/-- Represents the total number of cells in the original square -/
def total_cells : ℕ := original_size * original_size

/-- Represents the number of cells covered by each piece (both rectangle and square) -/
def cells_per_piece : ℕ := square_side * square_side

theorem cut_square_theorem (num_rectangles num_squares : ℕ) :
  (num_rectangles + num_squares = total_cells / cells_per_piece) ∧
  (2 * total_cut_length + original_perimeter = 
   num_rectangles * (2 * (rect_width + rect_length)) + 
   num_squares * (4 * square_side)) →
  num_rectangles = 6 ∧ num_squares = 10 := by
  sorry

end NUMINAMATH_CALUDE_cut_square_theorem_l1251_125151


namespace NUMINAMATH_CALUDE_wilted_flowers_count_l1251_125162

def flower_problem (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (remaining_bouquets : ℕ) : ℕ :=
  initial_flowers - (remaining_bouquets * flowers_per_bouquet)

theorem wilted_flowers_count :
  flower_problem 53 7 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_wilted_flowers_count_l1251_125162


namespace NUMINAMATH_CALUDE_clubsuit_computation_l1251_125153

-- Define the ♣ operation
def clubsuit (a c b : ℚ) : ℚ := (2 * a + c) / b

-- State the theorem
theorem clubsuit_computation : 
  clubsuit (clubsuit 6 1 (clubsuit 4 2 3)) 2 2 = 49 / 10 := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_computation_l1251_125153


namespace NUMINAMATH_CALUDE_sin_cos_sum_equivalent_l1251_125118

theorem sin_cos_sum_equivalent (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * (x + π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equivalent_l1251_125118


namespace NUMINAMATH_CALUDE_least_multiple_of_15_greater_than_520_l1251_125137

theorem least_multiple_of_15_greater_than_520 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n > 520 → n ≥ 525 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_15_greater_than_520_l1251_125137


namespace NUMINAMATH_CALUDE_compute_expression_l1251_125100

theorem compute_expression : 2⁻¹ + |-5| - Real.sin (30 * π / 180) + (π - 1)^0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1251_125100


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l1251_125127

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / y ∧ y = (3 + Real.sqrt 29) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l1251_125127


namespace NUMINAMATH_CALUDE_gcd_102_238_l1251_125168

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l1251_125168


namespace NUMINAMATH_CALUDE_two_digit_integer_less_than_multiple_l1251_125142

theorem two_digit_integer_less_than_multiple (n : ℕ) : n = 83 ↔ 
  (10 ≤ n ∧ n < 100) ∧ 
  (∃ k : ℕ, n + 1 = 3 * k) ∧ 
  (∃ k : ℕ, n + 1 = 4 * k) ∧ 
  (∃ k : ℕ, n + 1 = 7 * k) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_integer_less_than_multiple_l1251_125142


namespace NUMINAMATH_CALUDE_chris_earnings_june_l1251_125194

/-- Chris's earnings for the first two weeks of June --/
def chrisEarnings (hoursWeek1 hoursWeek2 : ℕ) (extraEarnings : ℚ) : ℚ :=
  let hourlyWage := extraEarnings / (hoursWeek2 - hoursWeek1)
  hourlyWage * (hoursWeek1 + hoursWeek2)

/-- Theorem stating Chris's earnings for the first two weeks of June --/
theorem chris_earnings_june :
  chrisEarnings 18 30 (65.40 : ℚ) = (261.60 : ℚ) := by
  sorry

#eval chrisEarnings 18 30 (65.40 : ℚ)

end NUMINAMATH_CALUDE_chris_earnings_june_l1251_125194


namespace NUMINAMATH_CALUDE_tenth_term_of_a_sum_of_2023rd_terms_l1251_125130

/-- Sequence a_n defined as (-2)^n -/
def a (n : ℕ) : ℤ := (-2) ^ n

/-- Sequence b_n defined as (-2)^n + (n+1) -/
def b (n : ℕ) : ℤ := (-2) ^ n + (n + 1)

/-- The 10th term of sequence a_n is (-2)^10 -/
theorem tenth_term_of_a : a 10 = (-2) ^ 10 := by sorry

/-- The sum of the 2023rd terms of sequences a_n and b_n is -2^2024 + 2024 -/
theorem sum_of_2023rd_terms : a 2023 + b 2023 = -2 ^ 2024 + 2024 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_a_sum_of_2023rd_terms_l1251_125130


namespace NUMINAMATH_CALUDE_garden_ratio_l1251_125126

/-- Given a rectangular garden with perimeter 180 yards and length 60 yards,
    prove that the ratio of length to width is 2:1 -/
theorem garden_ratio (perimeter : ℝ) (length : ℝ) (width : ℝ)
    (h1 : perimeter = 180)
    (h2 : length = 60)
    (h3 : perimeter = 2 * length + 2 * width) :
    length / width = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l1251_125126


namespace NUMINAMATH_CALUDE_max_correct_answers_l1251_125133

theorem max_correct_answers (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) :
  total_questions = 25 →
  correct_score = 5 →
  incorrect_score = -3 →
  total_score = 57 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct ≤ 12 ∧
    ∀ (c i u : ℕ),
      c + i + u = total_questions →
      correct_score * c + incorrect_score * i = total_score →
      c ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1251_125133


namespace NUMINAMATH_CALUDE_comic_book_arrangement_l1251_125193

def arrange_comic_books (batman : Nat) (superman : Nat) (wonder_woman : Nat) (flash : Nat) : Nat :=
  (Nat.factorial batman) * (Nat.factorial superman) * (Nat.factorial wonder_woman) * (Nat.factorial flash) * (Nat.factorial 4)

theorem comic_book_arrangement :
  arrange_comic_books 8 7 6 5 = 421275894176000 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_arrangement_l1251_125193


namespace NUMINAMATH_CALUDE_exponential_equation_sum_of_reciprocals_l1251_125170

theorem exponential_equation_sum_of_reciprocals (x y : ℝ) 
  (h1 : 3^x = Real.sqrt 12) 
  (h2 : 4^y = Real.sqrt 12) : 
  1/x + 1/y = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_sum_of_reciprocals_l1251_125170


namespace NUMINAMATH_CALUDE_marks_deck_cost_l1251_125117

/-- The total cost of a rectangular deck with sealant -/
def deck_cost (length width base_cost sealant_cost : ℝ) : ℝ :=
  let area := length * width
  let total_cost_per_sqft := base_cost + sealant_cost
  area * total_cost_per_sqft

/-- Theorem: The cost of Mark's deck is $4800 -/
theorem marks_deck_cost :
  deck_cost 30 40 3 1 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_marks_deck_cost_l1251_125117


namespace NUMINAMATH_CALUDE_W_min_value_l1251_125119

/-- The function W defined on real numbers x and y -/
def W (x y : ℝ) : ℝ := 5 * x^2 - 4 * x * y + y^2 - 2 * y + 8 * x + 3

/-- Theorem stating that W has a minimum value of -2 -/
theorem W_min_value :
  (∀ x y : ℝ, W x y ≥ -2) ∧ (∃ x y : ℝ, W x y = -2) := by
  sorry

end NUMINAMATH_CALUDE_W_min_value_l1251_125119


namespace NUMINAMATH_CALUDE_complex_fraction_imaginary_l1251_125164

theorem complex_fraction_imaginary (a : ℝ) : 
  (∃ (b : ℝ), b ≠ 0 ∧ (a - Complex.I) / (2 + Complex.I) = Complex.I * b) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_imaginary_l1251_125164


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l1251_125105

/-- Given a geometric sequence with positive terms and common ratio not equal to 1,
    prove that the arithmetic mean of the 3rd and 9th terms is greater than
    the geometric mean of the 5th and 7th terms. -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  q ≠ 1 →           -- Common ratio is not 1
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence property
  (a 3 + a 9) / 2 > Real.sqrt (a 5 * a 7) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l1251_125105


namespace NUMINAMATH_CALUDE_simplified_root_expression_l1251_125165

theorem simplified_root_expression : 
  ∃ (a b : ℕ), (a > 0 ∧ b > 0) ∧ 
  (3^5 * 5^4)^(1/4) = a * b^(1/4) ∧ 
  a + b = 18 := by sorry

end NUMINAMATH_CALUDE_simplified_root_expression_l1251_125165


namespace NUMINAMATH_CALUDE_units_digit_of_product_division_l1251_125149

theorem units_digit_of_product_division : 
  (30 * 31 * 32 * 33 * 34 * 35) / 2500 % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_division_l1251_125149


namespace NUMINAMATH_CALUDE_min_pieces_for_horizontal_four_l1251_125110

/-- Represents a chessboard as a list of 8 rows, each containing 8 cells --/
def Chessboard := List (List Bool)

/-- Checks if a row contains 4 consecutive true values --/
def hasFourConsecutive (row : List Bool) : Bool :=
  sorry

/-- Checks if any row in the chessboard has 4 consecutive pieces --/
def hasHorizontalFour (board : Chessboard) : Bool :=
  sorry

/-- Generates all possible arrangements of n pieces on a chessboard --/
def allArrangements (n : Nat) : List Chessboard :=
  sorry

theorem min_pieces_for_horizontal_four :
  ∀ n : Nat, (n ≥ 49 ↔ ∀ board ∈ allArrangements n, hasHorizontalFour board) :=
by sorry

end NUMINAMATH_CALUDE_min_pieces_for_horizontal_four_l1251_125110


namespace NUMINAMATH_CALUDE_total_money_l1251_125101

def jack_money : ℕ := 26
def ben_money : ℕ := jack_money - 9
def eric_money : ℕ := ben_money - 10

theorem total_money : eric_money + ben_money + jack_money = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l1251_125101


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1251_125107

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 - 3*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                                a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1251_125107


namespace NUMINAMATH_CALUDE_combined_list_size_l1251_125134

def combined_friends_list (james_friends john_friends shared_friends : ℕ) : ℕ :=
  james_friends + john_friends - shared_friends

theorem combined_list_size :
  let james_friends : ℕ := 75
  let john_friends : ℕ := 3 * james_friends
  let shared_friends : ℕ := 25
  combined_friends_list james_friends john_friends shared_friends = 275 := by
  sorry

end NUMINAMATH_CALUDE_combined_list_size_l1251_125134


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1251_125124

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (0 < a ∧ a < b → 1 / a > 1 / b) ∧
  ¬(1 / a > 1 / b → 0 < a ∧ a < b) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1251_125124


namespace NUMINAMATH_CALUDE_locus_of_C1_l1251_125189

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define a chord parallel to x-axis
structure Chord :=
  (a : ℝ)
  (property : parabola a = parabola (-a))

-- Define a point on the parabola
structure ParabolaPoint :=
  (x : ℝ)
  (y : ℝ)
  (on_parabola : y = parabola x)

-- Define the circumcircle of a triangle
def circumcircle (A B C : ParabolaPoint) : Set (ℝ × ℝ) := sorry

-- Define a point on the circumcircle with the same x-coordinate as C
def C1 (C : ParabolaPoint) (circle : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- The main theorem
theorem locus_of_C1 (AB : Chord) (C : ParabolaPoint) 
  (hC : C.x ≠ AB.a ∧ C.x ≠ -AB.a) :
  let A := ⟨AB.a, parabola AB.a, rfl⟩
  let B := ⟨-AB.a, parabola (-AB.a), rfl⟩
  let circle := circumcircle A B C
  let c1 := C1 C circle
  c1.2 = 1 + AB.a^2 := by sorry

end NUMINAMATH_CALUDE_locus_of_C1_l1251_125189


namespace NUMINAMATH_CALUDE_prob_two_even_in_six_dice_l1251_125112

/-- A fair 10-sided die with faces numbered from 1 to 10 -/
def TenSidedDie : Type := Fin 10

/-- The probability of rolling an even number on a 10-sided die -/
def probEven : ℚ := 1/2

/-- The probability of rolling an odd number on a 10-sided die -/
def probOdd : ℚ := 1/2

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The number of dice that should show an even number -/
def numEven : ℕ := 2

/-- The probability of rolling exactly two even numbers when rolling six fair 10-sided dice -/
theorem prob_two_even_in_six_dice : 
  (numDice.choose numEven : ℚ) * probEven ^ numEven * probOdd ^ (numDice - numEven) = 15/64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_even_in_six_dice_l1251_125112


namespace NUMINAMATH_CALUDE_special_function_property_l1251_125187

/-- A function satisfying the given property for all real numbers -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ c d : ℝ, c^2 * g d = d^2 * g c

theorem special_function_property (g : ℝ → ℝ) (h1 : SatisfiesProperty g) (h2 : g 4 ≠ 0) :
  (g 7 - g 3) / g 4 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l1251_125187


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l1251_125195

/-- The shaded area of a square with side length 36 inches containing 9 tangent circles -/
theorem shaded_area_square_with_circles : 
  let square_side : ℝ := 36
  let num_circles : ℕ := 9
  let circle_radius : ℝ := square_side / 6

  let square_area : ℝ := square_side ^ 2
  let total_circles_area : ℝ := num_circles * Real.pi * circle_radius ^ 2
  let shaded_area : ℝ := square_area - total_circles_area

  shaded_area = 1296 - 324 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l1251_125195


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1251_125135

/-- The y-coordinate of the point on the y-axis that is equidistant from A(-3, 0) and B(2, 5) is 2. -/
theorem equidistant_point_y_coordinate : ∃ y : ℝ, 
  ((-3 - 0)^2 + (0 - y)^2 = (2 - 0)^2 + (5 - y)^2) ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1251_125135


namespace NUMINAMATH_CALUDE_geese_percentage_among_non_herons_l1251_125158

theorem geese_percentage_among_non_herons :
  let total_birds : ℝ := 100
  let geese_percentage : ℝ := 30
  let swans_percentage : ℝ := 25
  let herons_percentage : ℝ := 20
  let ducks_percentage : ℝ := 25
  let non_heron_percentage : ℝ := total_birds - herons_percentage
  geese_percentage / non_heron_percentage * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_geese_percentage_among_non_herons_l1251_125158


namespace NUMINAMATH_CALUDE_divisibility_problem_l1251_125159

theorem divisibility_problem (a b c : ℕ+) 
  (h1 : a ∣ b^4) 
  (h2 : b ∣ c^4) 
  (h3 : c ∣ a^4) : 
  (a * b * c) ∣ (a + b + c)^21 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1251_125159


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1251_125152

theorem quadratic_equation_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x, x^2 + a*x + b = 0 ↔ x = -2*a ∨ x = b) →
  (b = -2*(-2*a) ∨ -2*a = -2*b) →
  a = -1/2 ∧ b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1251_125152


namespace NUMINAMATH_CALUDE_smallest_perfect_square_sum_of_20_consecutive_integers_l1251_125173

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def sum_of_consecutive_integers (start : ℕ) (count : ℕ) : ℕ :=
  count * (2 * start + count - 1) / 2

theorem smallest_perfect_square_sum_of_20_consecutive_integers :
  ∀ n : ℕ, 
    (∃ start : ℕ, sum_of_consecutive_integers start 20 = n ∧ is_perfect_square n) →
    n ≥ 490 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_sum_of_20_consecutive_integers_l1251_125173


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1251_125163

-- Define the equation
def equation (x : ℝ) : Prop :=
  2 * Real.cos (2 * x) * (Real.cos (2 * x) - Real.cos (2000 * Real.pi ^ 2 / x)) = Real.cos (4 * x) - 1

-- Define the set of all positive real solutions
def solution_set : Set ℝ := {x | x > 0 ∧ equation x}

-- State the theorem
theorem sum_of_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, x ∈ solution_set) ∧
                    (∀ x ∈ solution_set, x ∈ S) ∧
                    (Finset.sum S id = 136 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1251_125163


namespace NUMINAMATH_CALUDE_justin_age_proof_l1251_125150

/-- Angelina's age in 5 years -/
def angelina_future_age : ℕ := 40

/-- Number of years until Angelina reaches her future age -/
def years_until_future : ℕ := 5

/-- Age difference between Angelina and Justin -/
def age_difference : ℕ := 4

/-- Justin's current age -/
def justin_current_age : ℕ := angelina_future_age - years_until_future - age_difference

theorem justin_age_proof : justin_current_age = 31 := by
  sorry

end NUMINAMATH_CALUDE_justin_age_proof_l1251_125150


namespace NUMINAMATH_CALUDE_two_propositions_are_true_l1251_125177

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations
def parallel : Line → Line → Prop := sorry
def perpendicular : Line → Line → Prop := sorry
def planeParallel : Plane → Plane → Prop := sorry
def planePerpendicular : Plane → Plane → Prop := sorry
def lineParallelToPlane : Line → Plane → Prop := sorry
def linePerpendicularToPlane : Line → Plane → Prop := sorry

-- Define the propositions
def prop1 (α β : Plane) (c : Line) : Prop :=
  planeParallel α β ∧ linePerpendicularToPlane c α → linePerpendicularToPlane c β

def prop2 (α : Plane) (b : Line) (γ : Plane) : Prop :=
  lineParallelToPlane b α ∧ planePerpendicular α γ → linePerpendicularToPlane b γ

def prop3 (a : Line) (β γ : Plane) : Prop :=
  lineParallelToPlane a β ∧ linePerpendicularToPlane a γ → planePerpendicular β γ

-- The main theorem
theorem two_propositions_are_true :
  ∃ (α β γ : Plane) (a b c : Line),
    (prop1 α β c ∧ prop3 a β γ) ∧ ¬prop2 α b γ :=
sorry

end NUMINAMATH_CALUDE_two_propositions_are_true_l1251_125177


namespace NUMINAMATH_CALUDE_discontinuous_when_limit_not_equal_value_l1251_125181

-- Define a multivariable function type
def MultivariableFunction (α : Type*) (β : Type*) := α → β

-- Define the concept of a limit for a multivariable function
def HasLimit (f : MultivariableFunction ℝ ℝ) (x₀ : ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - L) < ε

-- Define continuity for a multivariable function
def IsContinuousAt (f : MultivariableFunction ℝ ℝ) (x₀ : ℝ) : Prop :=
  ∃ L, HasLimit f x₀ L ∧ f x₀ = L

-- Theorem statement
theorem discontinuous_when_limit_not_equal_value
  (f : MultivariableFunction ℝ ℝ) (x₀ : ℝ) (L : ℝ) :
  HasLimit f x₀ L → f x₀ ≠ L → ¬(IsContinuousAt f x₀) :=
sorry

end NUMINAMATH_CALUDE_discontinuous_when_limit_not_equal_value_l1251_125181


namespace NUMINAMATH_CALUDE_farm_feet_count_l1251_125166

/-- Given a farm with hens and cows, calculate the total number of feet -/
theorem farm_feet_count (total_heads : ℕ) (hen_count : ℕ) : 
  total_heads = 46 → hen_count = 22 → (hen_count * 2 + (total_heads - hen_count) * 4 = 140) :=
by
  sorry

#check farm_feet_count

end NUMINAMATH_CALUDE_farm_feet_count_l1251_125166


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1251_125115

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = -1) (h2 : x * y = -1) : x^3 + y^3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1251_125115


namespace NUMINAMATH_CALUDE_infinitely_many_primes_l1251_125154

theorem infinitely_many_primes : ∀ S : Finset Nat, (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_l1251_125154


namespace NUMINAMATH_CALUDE_josh_wallet_amount_l1251_125171

def calculate_final_wallet_amount (initial_wallet : ℝ) (investment : ℝ) (debt : ℝ)
  (stock_a_percent : ℝ) (stock_b_percent : ℝ) (stock_c_percent : ℝ)
  (stock_a_change : ℝ) (stock_b_change : ℝ) (stock_c_change : ℝ) : ℝ :=
  let stock_a_value := investment * stock_a_percent * (1 + stock_a_change)
  let stock_b_value := investment * stock_b_percent * (1 + stock_b_change)
  let stock_c_value := investment * stock_c_percent * (1 + stock_c_change)
  let total_stock_value := stock_a_value + stock_b_value + stock_c_value
  let remaining_after_debt := total_stock_value - debt
  initial_wallet + remaining_after_debt

theorem josh_wallet_amount :
  calculate_final_wallet_amount 300 2000 500 0.4 0.3 0.3 0.2 0.3 (-0.1) = 2080 := by
  sorry

end NUMINAMATH_CALUDE_josh_wallet_amount_l1251_125171


namespace NUMINAMATH_CALUDE_problem_solution_l1251_125102

theorem problem_solution (m n : ℝ) : 
  (∃ k : ℝ, k^2 = m + 3 ∧ (k = 1 ∨ k = -1)) →
  (2*n - 12)^(1/3) = 4 →
  m = -2 ∧ n = 38 ∧ Real.sqrt (m + n) = 6 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1251_125102


namespace NUMINAMATH_CALUDE_janet_walk_time_l1251_125106

-- Define Janet's walking pattern
def blocks_north : ℕ := 3
def blocks_west : ℕ := 7 * blocks_north
def blocks_south : ℕ := 8
def blocks_east : ℕ := 2 * blocks_south

-- Define Janet's walking speed
def blocks_per_minute : ℕ := 2

-- Calculate net distance from home
def net_south : ℤ := blocks_south - blocks_north
def net_west : ℤ := blocks_west - blocks_east

-- Total distance to walk home
def total_distance : ℕ := (net_south.natAbs + net_west.natAbs : ℕ)

-- Time to walk home
def time_to_home : ℚ := total_distance / blocks_per_minute

-- Theorem to prove
theorem janet_walk_time : time_to_home = 5 := by
  sorry

end NUMINAMATH_CALUDE_janet_walk_time_l1251_125106


namespace NUMINAMATH_CALUDE_fraction_calculation_l1251_125108

theorem fraction_calculation : 
  (7/6) / ((1/6) - (1/3)) * (3/14) / (3/5) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1251_125108


namespace NUMINAMATH_CALUDE_divisibility_implication_l1251_125192

theorem divisibility_implication (n : ℕ) (h : n > 0) :
  (13 ∣ n^2 + 3*n + 51) → (169 ∣ 21*n^2 + 89*n + 44) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1251_125192


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1251_125156

-- Define the propositions
def proposition_A (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0

def proposition_B (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, proposition_B a → proposition_A a) ∧
  (∃ a : ℝ, proposition_A a ∧ ¬proposition_B a) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1251_125156


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l1251_125144

theorem quadratic_roots_properties (b c x₁ x₂ : ℝ) 
  (h_eq : ∀ x, x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂)
  (h_distinct : x₁ ≠ x₂)
  (h_order : x₁ < x₂)
  (h_x₁_range : -1 < x₁ ∧ x₁ < 0) :
  (x₂ > 0 → c < 0) ∧
  (|x₂ - x₁| = 2 → |1 - b + c| - |1 + b + c| > 2*|4 + 2*b + c| - 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l1251_125144


namespace NUMINAMATH_CALUDE_root_product_equals_27_l1251_125176

theorem root_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_27_l1251_125176


namespace NUMINAMATH_CALUDE_olivia_earnings_l1251_125147

def hourly_wage : ℕ := 9
def monday_hours : ℕ := 4
def wednesday_hours : ℕ := 3
def friday_hours : ℕ := 6

theorem olivia_earnings : 
  hourly_wage * (monday_hours + wednesday_hours + friday_hours) = 117 := by
  sorry

end NUMINAMATH_CALUDE_olivia_earnings_l1251_125147


namespace NUMINAMATH_CALUDE_min_distance_sum_l1251_125148

theorem min_distance_sum (x y z : ℝ) :
  Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2) ≥ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1251_125148


namespace NUMINAMATH_CALUDE_adjacent_probability_l1251_125155

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a type for arrangements
def Arrangement := List Person

-- Define a function to check if A and B are adjacent in an arrangement
def areAdjacent (arr : Arrangement) : Prop :=
  ∃ i, (arr.get? i = some Person.A ∧ arr.get? (i+1) = some Person.B) ∨
       (arr.get? i = some Person.B ∧ arr.get? (i+1) = some Person.A)

-- Define the set of all possible arrangements
def allArrangements : Finset Arrangement :=
  sorry

-- Define the set of arrangements where A and B are adjacent
def adjacentArrangements : Finset Arrangement :=
  sorry

-- State the theorem
theorem adjacent_probability :
  (adjacentArrangements.card : ℚ) / (allArrangements.card : ℚ) = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_adjacent_probability_l1251_125155


namespace NUMINAMATH_CALUDE_divisible_by_42_l1251_125186

theorem divisible_by_42 (n : ℕ) : ∃ k : ℤ, (n ^ 3 * (n ^ 6 - 1) : ℤ) = 42 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_42_l1251_125186


namespace NUMINAMATH_CALUDE_probability_multiple_of_three_l1251_125109

def is_multiple_of_three (n : ℕ) : Bool :=
  n % 3 = 0

def count_multiples_of_three (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_three |>.length

theorem probability_multiple_of_three : 
  (count_multiples_of_three 24 : ℚ) / 24 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_of_three_l1251_125109


namespace NUMINAMATH_CALUDE_problem_statement_l1251_125185

theorem problem_statement (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : a^2 * b^2 / (a^4 - 2 * b^4) = 1) :
  (a^2 - b^2) / (a^2 + b^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1251_125185


namespace NUMINAMATH_CALUDE_more_sad_players_left_l1251_125121

/-- Represents the state of a player in the game -/
inductive PlayerState
| Sad
| Cheerful

/-- Represents the game with its rules and initial state -/
structure Game where
  initialPlayers : Nat
  remainingPlayers : Nat
  sadPlayers : Nat
  cheerfulPlayers : Nat

/-- Definition of a valid game state -/
def validGameState (g : Game) : Prop :=
  g.initialPlayers = 36 ∧
  g.remainingPlayers + g.sadPlayers + g.cheerfulPlayers = g.initialPlayers ∧
  g.remainingPlayers ≥ 1

/-- The game ends when only one player remains -/
def gameEnded (g : Game) : Prop :=
  g.remainingPlayers = 1

/-- Theorem stating that more sad players have left the game than cheerful players when the game ends -/
theorem more_sad_players_left (g : Game) 
  (h1 : validGameState g) 
  (h2 : gameEnded g) : 
  g.sadPlayers > g.cheerfulPlayers :=
sorry

end NUMINAMATH_CALUDE_more_sad_players_left_l1251_125121


namespace NUMINAMATH_CALUDE_probability_of_either_test_l1251_125120

theorem probability_of_either_test (p_math p_english : ℚ) 
  (h_math : p_math = 5/8)
  (h_english : p_english = 1/4)
  (h_independent : True) -- We don't need to express independence in the theorem statement
  : 1 - (1 - p_math) * (1 - p_english) = 23/32 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_either_test_l1251_125120


namespace NUMINAMATH_CALUDE_function_value_at_negative_two_l1251_125188

/-- Given a function f(x) = ax^4 + bx^2 - x + 1 where a and b are real numbers,
    if f(2) = 9, then f(-2) = 13 -/
theorem function_value_at_negative_two
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^4 + b * x^2 - x + 1)
  (h2 : f 2 = 9) :
  f (-2) = 13 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_two_l1251_125188


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1251_125146

theorem sqrt_sum_equality : Real.sqrt 1 + Real.sqrt 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1251_125146


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l1251_125174

theorem at_least_one_non_negative (a b c d e f g h : ℝ) :
  (max (a*c + b*d) (max (a*e + b*f) (max (a*g + b*h) (max (c*e + d*f) (max (c*g + d*h) (e*g + f*h)))))) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l1251_125174


namespace NUMINAMATH_CALUDE_puzzle_solution_l1251_125191

theorem puzzle_solution :
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    1448 = 282 * a + 10 * a + b ∧
    423 * (c / 3) = 282 ∧
    47 * 9 = 423 ∧
    423 * (2 / 3) = 282 ∧
    282 * 5 = 1410 ∧
    1410 + 38 = 1448 ∧
    705 + 348 = 1053 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1251_125191


namespace NUMINAMATH_CALUDE_inequality_range_l1251_125113

theorem inequality_range (m : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) → 
  -6 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1251_125113


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1251_125131

theorem arithmetic_calculation : 8 + (-2)^3 / (-4) * (-7 + 5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1251_125131


namespace NUMINAMATH_CALUDE_gcd_increase_l1251_125125

theorem gcd_increase (m n : ℕ) (h : Nat.gcd (m + 6) n = 9 * Nat.gcd m n) :
  Nat.gcd m n = 3 ∨ Nat.gcd m n = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_increase_l1251_125125


namespace NUMINAMATH_CALUDE_inequality_proof_l1251_125179

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1251_125179


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l1251_125175

theorem arithmetic_progression_sum (x y z d k : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  y * (z - x) - x * (y - z) = d ∧
  z * (x - y) - y * (z - x) = d ∧
  x * (y - z) + y * (z - x) + z * (x - y) = k
  → d = k / 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l1251_125175


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_C_subset_B_implies_m_range_l1251_125104

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
def B : Set ℝ := {x | x^2 - 4*x < 0}
def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 4} := by sorry

theorem complement_union_A_B : 
  (Set.univ : Set ℝ) \ (A ∪ B) = {x : ℝ | x ≤ 0 ∨ x > 6} := by sorry

theorem C_subset_B_implies_m_range (m : ℝ) : 
  C m ⊆ B → m ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_C_subset_B_implies_m_range_l1251_125104


namespace NUMINAMATH_CALUDE_max_value_implies_a_f_leq_g_implies_a_range_l1251_125145

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + 2 * a * x + 1
def g (x : ℝ) : ℝ := x * (Real.exp x + 1)

-- Part (Ⅰ)
theorem max_value_implies_a (a : ℝ) :
  (∃ x > 0, ∀ y > 0, f a x ≥ f a y) ∧ (∃ x > 0, f a x = 0) →
  a = -1/2 :=
sorry

-- Part (Ⅱ)
theorem f_leq_g_implies_a_range (a : ℝ) :
  (∀ x > 0, f a x ≤ g x) →
  a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_f_leq_g_implies_a_range_l1251_125145


namespace NUMINAMATH_CALUDE_sum_of_squares_divisibility_l1251_125128

theorem sum_of_squares_divisibility (a b c : ℤ) :
  9 ∣ (a^2 + b^2 + c^2) → 
  (9 ∣ (a^2 - b^2)) ∨ (9 ∣ (a^2 - c^2)) ∨ (9 ∣ (b^2 - c^2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisibility_l1251_125128


namespace NUMINAMATH_CALUDE_decimal_rep_denominators_num_possible_denominators_l1251_125172

-- Define a digit as a natural number between 0 and 9
def Digit := {n : ℕ // n ≤ 9}

-- Define the decimal representation
def DecimalRep (a b c : Digit) : ℚ := (100 * a.val + 10 * b.val + c.val : ℕ) / 999

-- Define the condition that not all digits are nine
def NotAllNine (a b c : Digit) : Prop :=
  ¬(a.val = 9 ∧ b.val = 9 ∧ c.val = 9)

-- Define the condition that not all digits are zero
def NotAllZero (a b c : Digit) : Prop :=
  ¬(a.val = 0 ∧ b.val = 0 ∧ c.val = 0)

-- Define the set of possible denominators
def PossibleDenominators : Finset ℕ :=
  {3, 9, 27, 37, 111, 333, 999}

-- The main theorem
theorem decimal_rep_denominators (a b c : Digit) 
  (h1 : NotAllNine a b c) (h2 : NotAllZero a b c) :
  (DecimalRep a b c).den ∈ PossibleDenominators := by
  sorry

-- The final result
theorem num_possible_denominators :
  Finset.card PossibleDenominators = 7 := by
  sorry

end NUMINAMATH_CALUDE_decimal_rep_denominators_num_possible_denominators_l1251_125172


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1251_125169

/-- An arithmetic sequence with first term 1 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  1 + 2 * (n - 1)

/-- The general formula for the n-th term of the arithmetic sequence -/
theorem arithmetic_sequence_formula (n : ℕ) :
  arithmetic_sequence n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1251_125169


namespace NUMINAMATH_CALUDE_roy_daily_sports_hours_l1251_125183

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of hours Roy spends on sports in a week when he misses 2 days -/
def sports_hours_with_missed_days : ℕ := 6

/-- The number of days Roy misses in a week -/
def missed_days : ℕ := 2

/-- The number of hours Roy spends on sports activities in school every day -/
def daily_sports_hours : ℚ := 2

theorem roy_daily_sports_hours :
  daily_sports_hours = sports_hours_with_missed_days / (school_days_per_week - missed_days) :=
by sorry

end NUMINAMATH_CALUDE_roy_daily_sports_hours_l1251_125183


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_leg_length_l1251_125198

/-- An isosceles right triangle with a median to the hypotenuse of length 15 units has legs of length 15√2 units. -/
theorem isosceles_right_triangle_leg_length :
  ∀ (a b c m : ℝ),
  a = b →                          -- The triangle is isosceles
  a^2 + b^2 = c^2 →                -- The triangle is right-angled (Pythagorean theorem)
  m = 15 →                         -- The median to the hypotenuse is 15 units
  m = c / 2 →                      -- The median to the hypotenuse is half the hypotenuse length
  a = 15 * Real.sqrt 2 :=           -- The leg length is 15√2
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_leg_length_l1251_125198


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1251_125141

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n
  prop1 : a 7 * a 11 = 6
  prop2 : a 4 + a 14 = 5

/-- The main theorem stating the possible values of a_20 / a_10 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 20 / seq.a 10 = 2/3 ∨ seq.a 20 / seq.a 10 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1251_125141


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l1251_125178

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x + 1|

-- Theorem for the solution set of f(x) > 3
theorem solution_set_f (x : ℝ) : f x > 3 ↔ x > 1 ∨ x < -2 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m^2 + 3*m + 2*f x ≥ 0) ↔ (m ≤ -2 ∨ m ≥ -1) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l1251_125178


namespace NUMINAMATH_CALUDE_billy_coin_count_l1251_125122

/-- Represents the number of piles for each coin type -/
structure PileCount where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Represents the number of coins in each pile for each coin type -/
structure CoinsPerPile where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total number of coins given the pile counts and coins per pile -/
def totalCoins (piles : PileCount) (coinsPerPile : CoinsPerPile) : Nat :=
  piles.quarters * coinsPerPile.quarters +
  piles.dimes * coinsPerPile.dimes +
  piles.nickels * coinsPerPile.nickels +
  piles.pennies * coinsPerPile.pennies

/-- Billy's coin sorting problem -/
theorem billy_coin_count :
  let piles : PileCount := { quarters := 3, dimes := 2, nickels := 4, pennies := 6 }
  let coinsPerPile : CoinsPerPile := { quarters := 5, dimes := 7, nickels := 3, pennies := 9 }
  totalCoins piles coinsPerPile = 95 := by
  sorry

end NUMINAMATH_CALUDE_billy_coin_count_l1251_125122


namespace NUMINAMATH_CALUDE_marley_fruit_count_l1251_125190

/-- Represents the number of fruits a person has -/
structure FruitCount where
  oranges : ℕ
  apples : ℕ

/-- Calculates the total number of fruits -/
def totalFruits (fc : FruitCount) : ℕ :=
  fc.oranges + fc.apples

/-- The problem statement -/
theorem marley_fruit_count :
  let louis : FruitCount := ⟨5, 3⟩
  let samantha : FruitCount := ⟨8, 7⟩
  let marley : FruitCount := ⟨2 * louis.oranges, 3 * samantha.apples⟩
  totalFruits marley = 31 := by
  sorry


end NUMINAMATH_CALUDE_marley_fruit_count_l1251_125190


namespace NUMINAMATH_CALUDE_triangle_angle_ranges_l1251_125160

def triangle_angles (α β γ : Real) : Prop :=
  α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0

theorem triangle_angle_ranges (α β γ : Real) (h : triangle_angles α β γ) :
  60 ≤ max α (max β γ) ∧ max α (max β γ) < 180 ∧
  0 < min α (min β γ) ∧ min α (min β γ) ≤ 60 ∧
  0 < (max (min α β) (min (max α β) γ)) ∧ (max (min α β) (min (max α β) γ)) < 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_ranges_l1251_125160


namespace NUMINAMATH_CALUDE_cakes_left_is_two_l1251_125136

def cakes_baked_yesterday : ℕ := 3
def cakes_baked_lunch : ℕ := 5
def cakes_sold_dinner : ℕ := 6

def cakes_left : ℕ := cakes_baked_yesterday + cakes_baked_lunch - cakes_sold_dinner

theorem cakes_left_is_two : cakes_left = 2 := by
  sorry

end NUMINAMATH_CALUDE_cakes_left_is_two_l1251_125136


namespace NUMINAMATH_CALUDE_problem_solution_l1251_125103

theorem problem_solution (a b c : ℚ) 
  (h1 : a + b + c = 72)
  (h2 : a + 4 = b - 8)
  (h3 : a + 4 = 4 * c) : 
  a = 236 / 9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1251_125103


namespace NUMINAMATH_CALUDE_linear_equation_exponent_l1251_125167

theorem linear_equation_exponent (k : ℕ) : 
  (∀ x, ∃ a b, x^(k-1) + 3 = a*x + b) → k = 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_l1251_125167


namespace NUMINAMATH_CALUDE_quadratic_with_real_roots_l1251_125180

/-- 
Given a quadratic equation with complex coefficients that has real roots, 
prove that the value of the real parameter m is 1/12.
-/
theorem quadratic_with_real_roots (i : ℂ) :
  (∃ x : ℝ, x^2 - (2*i - 1)*x + 3*m - i = 0) → m = 1/12 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_with_real_roots_l1251_125180


namespace NUMINAMATH_CALUDE_nina_walking_distance_l1251_125138

/-- Proves that Nina's walking distance to school is 0.4 miles, given John's distance and the difference between their distances. -/
theorem nina_walking_distance
  (john_distance : ℝ)
  (difference : ℝ)
  (h1 : john_distance = 0.7)
  (h2 : difference = 0.3)
  (h3 : john_distance = nina_distance + difference)
  : nina_distance = 0.4 :=
by
  sorry

end NUMINAMATH_CALUDE_nina_walking_distance_l1251_125138


namespace NUMINAMATH_CALUDE_martha_initial_blocks_l1251_125184

/-- Given that Martha finds 80 blocks and ends up with 84 blocks, 
    prove that she initially had 4 blocks. -/
theorem martha_initial_blocks : 
  ∀ (initial_blocks found_blocks final_blocks : ℕ),
    found_blocks = 80 →
    final_blocks = 84 →
    final_blocks = initial_blocks + found_blocks →
    initial_blocks = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_martha_initial_blocks_l1251_125184


namespace NUMINAMATH_CALUDE_student_congress_sample_size_l1251_125182

/-- Calculates the sample size for a Student Congress given the number of classes and students selected per class. -/
def sampleSize (numClasses : ℕ) (studentsPerClass : ℕ) : ℕ :=
  numClasses * studentsPerClass

/-- Theorem stating that for a school with 40 classes, where each class selects 3 students
    for the Student Congress, the sample size is 120 students. -/
theorem student_congress_sample_size :
  sampleSize 40 3 = 120 := by
  sorry

#eval sampleSize 40 3

end NUMINAMATH_CALUDE_student_congress_sample_size_l1251_125182


namespace NUMINAMATH_CALUDE_system_solution_l1251_125111

theorem system_solution (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : ℝ),
    (z + a*y + a^2*x + a^3 = 0) ∧
    (z + b*y + b^2*x + b^3 = 0) ∧
    (z + c*y + c^2*x + c^3 = 0) ∧
    (x = -(a+b+c)) ∧
    (y = a*b + a*c + b*c) ∧
    (z = -a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1251_125111


namespace NUMINAMATH_CALUDE_find_a_l1251_125199

theorem find_a (b w : ℝ) (h1 : b = 2120) (h2 : w = 0.5) : ∃ a : ℝ, w = a / b ∧ a = 1060 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1251_125199


namespace NUMINAMATH_CALUDE_losing_candidate_percentage_approx_33_percent_l1251_125139

/-- Calculates the percentage of votes received by a losing candidate -/
def losingCandidatePercentage (totalVotes : ℕ) (lossMargin : ℕ) : ℚ :=
  let candidateVotes := (totalVotes - lossMargin) / 2
  (candidateVotes : ℚ) / totalVotes * 100

/-- Theorem stating that given the conditions, the losing candidate's vote percentage is approximately 33% -/
theorem losing_candidate_percentage_approx_33_percent 
  (totalVotes : ℕ) (lossMargin : ℕ) 
  (h1 : totalVotes = 2450) 
  (h2 : lossMargin = 833) : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |losingCandidatePercentage totalVotes lossMargin - 33| < ε :=
sorry

end NUMINAMATH_CALUDE_losing_candidate_percentage_approx_33_percent_l1251_125139


namespace NUMINAMATH_CALUDE_min_value_theorem_l1251_125123

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  arithmetic_sequence a →
  (∀ k : ℕ, a k > 0) →
  a 7 = a 6 + 2 * a 5 →
  Real.sqrt (a m * a n) = 2 * Real.sqrt 2 * a 1 →
  (2 : ℝ) / m + 8 / n ≥ 18 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1251_125123


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_denominator_l1251_125116

theorem quadratic_rational_root_even_denominator
  (a b c : ℤ)  -- Coefficients are integers
  (h_even_sum : Even (a + b))  -- Sum of a and b is even
  (h_odd_c : Odd c)  -- c is odd
  (p q : ℤ)  -- p/q is a rational root in simplest form
  (h_coprime : Nat.Coprime p.natAbs q.natAbs)  -- p and q are coprime
  (h_root : a * p^2 + b * p * q + c * q^2 = 0)  -- p/q is a root
  : Even q  -- q is even
:= by sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_denominator_l1251_125116
