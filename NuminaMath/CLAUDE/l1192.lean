import Mathlib

namespace NUMINAMATH_CALUDE_train_length_proof_l1192_119221

def train_problem (length1 : ℝ) (speed1 : ℝ) (speed2 : ℝ) (clear_time : ℝ) : Prop :=
  let relative_speed : ℝ := (speed1 + speed2) * (1000 / 3600)
  let total_length : ℝ := relative_speed * clear_time
  let length2 : ℝ := total_length - length1
  length2 = 180

theorem train_length_proof :
  train_problem 110 80 65 7.199424046076314 :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l1192_119221


namespace NUMINAMATH_CALUDE_wire_service_reporters_l1192_119201

theorem wire_service_reporters (total_reporters : ℝ) 
  (local_politics_percentage : ℝ) (non_local_politics_percentage : ℝ) :
  local_politics_percentage = 18 / 100 →
  non_local_politics_percentage = 40 / 100 →
  total_reporters > 0 →
  (total_reporters - (total_reporters * local_politics_percentage / (1 - non_local_politics_percentage))) / total_reporters = 70 / 100 := by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l1192_119201


namespace NUMINAMATH_CALUDE_A_equals_two_three_l1192_119273

def A : Set ℤ := {x | (3 : ℚ) / (x - 1) > 1}

theorem A_equals_two_three : A = {2, 3} := by sorry

end NUMINAMATH_CALUDE_A_equals_two_three_l1192_119273


namespace NUMINAMATH_CALUDE_ten_bags_of_bags_l1192_119225

/-- The number of ways to create a "bag of bags" structure with n identical bags. -/
def bagsOfBags : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => sorry  -- The actual recursive definition would go here

/-- The number of ways to create a "bag of bags" structure with 10 identical bags is 719. -/
theorem ten_bags_of_bags : bagsOfBags 10 = 719 := by sorry

end NUMINAMATH_CALUDE_ten_bags_of_bags_l1192_119225


namespace NUMINAMATH_CALUDE_prime_sequence_l1192_119285

def is_increasing (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

theorem prime_sequence (a p : ℕ → ℕ) 
  (h_inc : is_increasing a)
  (h_prime : ∀ n, Nat.Prime (p n))
  (h_div : ∀ n, (p n) ∣ (a n))
  (h_diff : ∀ n k, a n - a k = p n - p k) :
  ∀ n, a n = p n :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_l1192_119285


namespace NUMINAMATH_CALUDE_change_calculation_l1192_119296

def initial_amount : ℕ := 20
def num_items : ℕ := 3
def cost_per_item : ℕ := 2

theorem change_calculation :
  initial_amount - (num_items * cost_per_item) = 14 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l1192_119296


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1192_119270

theorem boys_to_girls_ratio (S : ℕ) (G : ℕ) (h1 : S > 0) (h2 : G > 0) 
  (h3 : 2 * G = 3 * (S / 5)) : 
  (S - G : ℚ) / G = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1192_119270


namespace NUMINAMATH_CALUDE_distinct_numbers_count_l1192_119242

/-- Represents the possible states of a matchstick (present or removed) --/
inductive MatchstickState
| Present
| Removed

/-- Represents the configuration of matchsticks in the symbol --/
structure MatchstickConfiguration :=
(top : MatchstickState)
(bottom : MatchstickState)
(left : MatchstickState)
(right : MatchstickState)

/-- Defines the set of valid number representations --/
def ValidNumberRepresentations : Set MatchstickConfiguration := sorry

/-- Counts the number of distinct valid number representations --/
def CountDistinctNumbers : Nat := sorry

/-- Theorem stating that the number of distinct numbers obtainable is 5 --/
theorem distinct_numbers_count :
  CountDistinctNumbers = 5 := by sorry

end NUMINAMATH_CALUDE_distinct_numbers_count_l1192_119242


namespace NUMINAMATH_CALUDE_triangle_cosine_value_l1192_119207

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b - c = 1/4 * a and 2 * sin B = 3 * sin C, then cos A = -1/4 -/
theorem triangle_cosine_value (a b c A B C : ℝ) :
  b - c = (1/4) * a →
  2 * Real.sin B = 3 * Real.sin C →
  Real.cos A = -(1/4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_cosine_value_l1192_119207


namespace NUMINAMATH_CALUDE_total_juice_boxes_for_school_year_l1192_119213

-- Define the structure for a child
structure Child where
  name : String
  juiceBoxesPerWeek : ℕ
  schoolWeeks : ℕ

-- Define Peyton's children
def john : Child := { name := "John", juiceBoxesPerWeek := 10, schoolWeeks := 16 }
def samantha : Child := { name := "Samantha", juiceBoxesPerWeek := 5, schoolWeeks := 14 }
def heather : Child := { name := "Heather", juiceBoxesPerWeek := 11, schoolWeeks := 15 }

def children : List Child := [john, samantha, heather]

-- Function to calculate total juice boxes for a child
def totalJuiceBoxes (child : Child) : ℕ :=
  child.juiceBoxesPerWeek * child.schoolWeeks

-- Theorem to prove
theorem total_juice_boxes_for_school_year :
  (children.map totalJuiceBoxes).sum = 395 := by
  sorry

end NUMINAMATH_CALUDE_total_juice_boxes_for_school_year_l1192_119213


namespace NUMINAMATH_CALUDE_sqrt_1575n_integer_exists_l1192_119255

theorem sqrt_1575n_integer_exists : ∃ n : ℕ+, ∃ k : ℕ, (k : ℝ) ^ 2 = 1575 * n := by
  sorry

end NUMINAMATH_CALUDE_sqrt_1575n_integer_exists_l1192_119255


namespace NUMINAMATH_CALUDE_min_books_borrowed_l1192_119216

/-- Represents the minimum number of books borrowed by the remaining students -/
def min_books_remaining : ℕ := 4

theorem min_books_borrowed (total_students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℚ) (h1 : total_students = 38) (h2 : no_books = 2) (h3 : one_book = 12) 
  (h4 : two_books = 10) (h5 : avg_books = 2) : 
  min_books_remaining = 4 := by
  sorry

#check min_books_borrowed

end NUMINAMATH_CALUDE_min_books_borrowed_l1192_119216


namespace NUMINAMATH_CALUDE_triangle_inequality_equivalence_l1192_119286

theorem triangle_inequality_equivalence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_equivalence_l1192_119286


namespace NUMINAMATH_CALUDE_product_of_roots_l1192_119266

theorem product_of_roots (x : ℝ) : 
  ((x + 3) * (x - 4) = 22) → 
  (∃ y : ℝ, ((y + 3) * (y - 4) = 22) ∧ (x * y = -34)) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l1192_119266


namespace NUMINAMATH_CALUDE_complement_is_acute_l1192_119257

-- Define an angle as a real number between 0 and 180 degrees
def Angle := {x : ℝ // 0 ≤ x ∧ x ≤ 180}

-- Define an acute angle
def isAcute (a : Angle) : Prop := a.val < 90

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  ⟨90 - a.val, by sorry⟩  -- The proof that this is a valid angle is omitted

-- Theorem statement
theorem complement_is_acute (a : Angle) (h : a.val < 90) : isAcute (complement a) := by
  sorry


end NUMINAMATH_CALUDE_complement_is_acute_l1192_119257


namespace NUMINAMATH_CALUDE_add_decimals_l1192_119256

theorem add_decimals : (124.75 : ℝ) + 0.35 = 125.10 := by sorry

end NUMINAMATH_CALUDE_add_decimals_l1192_119256


namespace NUMINAMATH_CALUDE_kevin_cards_problem_l1192_119232

/-- Given that Kevin finds 47 cards and ends up with 54 cards, prove that he started with 7 cards. -/
theorem kevin_cards_problem (found_cards : ℕ) (total_cards : ℕ) (h1 : found_cards = 47) (h2 : total_cards = 54) :
  total_cards - found_cards = 7 := by
sorry

end NUMINAMATH_CALUDE_kevin_cards_problem_l1192_119232


namespace NUMINAMATH_CALUDE_orange_juice_profit_l1192_119281

/-- Represents the number of orange trees each sister has -/
def trees_per_sister : ℕ := 110

/-- Represents the number of oranges Gabriela's trees produce per tree -/
def gabriela_oranges_per_tree : ℕ := 600

/-- Represents the number of oranges Alba's trees produce per tree -/
def alba_oranges_per_tree : ℕ := 400

/-- Represents the number of oranges Maricela's trees produce per tree -/
def maricela_oranges_per_tree : ℕ := 500

/-- Represents the number of oranges needed to make one cup of juice -/
def oranges_per_cup : ℕ := 3

/-- Represents the price of one cup of juice in dollars -/
def price_per_cup : ℕ := 4

/-- Theorem stating the total money earned from selling orange juice -/
theorem orange_juice_profit : 
  (trees_per_sister * gabriela_oranges_per_tree + 
   trees_per_sister * alba_oranges_per_tree + 
   trees_per_sister * maricela_oranges_per_tree) / oranges_per_cup * price_per_cup = 220000 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_profit_l1192_119281


namespace NUMINAMATH_CALUDE_unknown_number_l1192_119212

theorem unknown_number (x n : ℝ) : 
  (5 * x + n = 10 * x - 17) → (x = 4) → (n = 3) := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_l1192_119212


namespace NUMINAMATH_CALUDE_shoes_outside_library_l1192_119245

/-- The number of shoes for a group of people, given the number of people and shoes per person. -/
def total_shoes (num_people : ℕ) (shoes_per_person : ℕ) : ℕ :=
  num_people * shoes_per_person

/-- Theorem: For a group of 10 people, where each person wears 2 shoes,
    the total number of shoes when everyone takes them off is 20. -/
theorem shoes_outside_library :
  total_shoes 10 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shoes_outside_library_l1192_119245


namespace NUMINAMATH_CALUDE_like_terms_exponent_relation_l1192_119295

theorem like_terms_exponent_relation (m n : ℕ) : 
  (∀ (x y : ℝ), ∃ (k : ℝ), 3 * x^(3*m) * y^2 = k * x^6 * y^n) → m^n = 4 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_relation_l1192_119295


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l1192_119282

def numbers : List ℝ := [1030, 1560, 1980, 2025, 2140, 2250, 2450, 2600, 2780, 2910]

theorem mean_of_remaining_numbers :
  let total_sum := numbers.sum
  let seven_mean := 2300
  let seven_sum := 7 * seven_mean
  let remaining_sum := total_sum - seven_sum
  (remaining_sum / 3 : ℝ) = 2108.33 := by
sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l1192_119282


namespace NUMINAMATH_CALUDE_candle_height_after_burning_l1192_119277

-- Define the initial height of the candle
def initial_height : ℕ := 150

-- Define the burning time for odd-numbered centimeters
def burn_time_odd (k : ℕ) : ℕ := 10 * k

-- Define the burning time for even-numbered centimeters
def burn_time_even (k : ℕ) : ℕ := 15 * k

-- Define the total elapsed time
def elapsed_time : ℕ := 80000

-- Function to calculate the remaining height after a given time
def remaining_height (t : ℕ) : ℕ :=
  sorry

-- Theorem stating that after 80,000 seconds, the remaining height is 70 cm
theorem candle_height_after_burning :
  remaining_height elapsed_time = 70 :=
sorry

end NUMINAMATH_CALUDE_candle_height_after_burning_l1192_119277


namespace NUMINAMATH_CALUDE_function_positive_iff_a_greater_half_l1192_119292

/-- The function f(x) = ax² - 2x + 2 is positive for all x in (1, 4) if and only if a > 1/2 -/
theorem function_positive_iff_a_greater_half (a : ℝ) :
  (∀ x : ℝ, 1 < x → x < 4 → a * x^2 - 2*x + 2 > 0) ↔ a > 1/2 :=
by sorry

end NUMINAMATH_CALUDE_function_positive_iff_a_greater_half_l1192_119292


namespace NUMINAMATH_CALUDE_f_1994_4_l1192_119223

def f (x : ℚ) : ℚ := (2 + x) / (2 - 2*x)

def f_n : ℕ → (ℚ → ℚ)
| 0 => id
| (n+1) => f ∘ (f_n n)

theorem f_1994_4 : f_n 1994 4 = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_1994_4_l1192_119223


namespace NUMINAMATH_CALUDE_max_x_for_perfect_square_l1192_119261

theorem max_x_for_perfect_square : 
  ∀ x : ℕ, x > 1972 → ¬(∃ y : ℕ, 4^27 + 4^1000 + 4^x = y^2) ∧ 
  ∃ y : ℕ, 4^27 + 4^1000 + 4^1972 = y^2 :=
by sorry

end NUMINAMATH_CALUDE_max_x_for_perfect_square_l1192_119261


namespace NUMINAMATH_CALUDE_correct_distribution_l1192_119200

/-- The number of ways to distribute men and women into groups --/
def distribute_people (num_men num_women : ℕ) : ℕ :=
  let group1 := Nat.choose num_men 2 * Nat.choose num_women 1
  let group2 := Nat.choose (num_men - 2) 1 * Nat.choose (num_women - 1) 2
  let group3 := Nat.choose 1 1 * Nat.choose 2 2
  (group1 * group2 * group3) / 2

/-- Theorem stating the correct number of distributions --/
theorem correct_distribution : distribute_people 4 5 = 180 := by
  sorry

end NUMINAMATH_CALUDE_correct_distribution_l1192_119200


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1192_119250

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_y : ℝ), ∀ (y : ℝ), y = 2 * x^2 + 8 * x + 18 → y ≥ min_y ∧ min_y = 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1192_119250


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l1192_119259

theorem largest_integer_inequality : ∀ x : ℤ, x ≤ 4 ↔ (x : ℚ) / 4 - 3 / 7 < 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l1192_119259


namespace NUMINAMATH_CALUDE_no_valid_covering_for_6_and_7_l1192_119284

/-- Represents the L-shaped or T-shaped 4-cell figure -/
inductive TetrominoShape
| L
| T

/-- Represents a position on the n×n square -/
structure Position (n : ℕ) where
  x : Fin n
  y : Fin n

/-- Represents a tetromino (4-cell figure) placement on the square -/
structure TetrominoPlacement (n : ℕ) where
  shape : TetrominoShape
  position : Position n
  rotation : Fin 4  -- 0, 90, 180, or 270 degrees

/-- Checks if a tetromino placement is valid within the n×n square -/
def is_valid_placement (n : ℕ) (placement : TetrominoPlacement n) : Prop := sorry

/-- Checks if a set of tetromino placements covers the entire n×n square exactly once -/
def covers_square_once (n : ℕ) (placements : List (TetrominoPlacement n)) : Prop := sorry

theorem no_valid_covering_for_6_and_7 :
  ¬ (∃ (placements : List (TetrominoPlacement 6)), covers_square_once 6 placements) ∧
  ¬ (∃ (placements : List (TetrominoPlacement 7)), covers_square_once 7 placements) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_covering_for_6_and_7_l1192_119284


namespace NUMINAMATH_CALUDE_intersection_when_m_eq_2_sufficient_not_necessary_condition_l1192_119291

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | (x-1+m)*(x-1-m) ≤ 0}

-- Theorem for part (1)
theorem intersection_when_m_eq_2 : 
  A ∩ B 2 = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (2)
theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) ↔ m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_eq_2_sufficient_not_necessary_condition_l1192_119291


namespace NUMINAMATH_CALUDE_symmetry_of_lines_l1192_119252

/-- Given two lines in a 2D plane, this function returns true if they are symmetric with respect to a third line. -/
def are_symmetric_lines (line1 line2 axis : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, line1 x y ↔ line2 (y - 1) (x + 1)

/-- The line y = 2x + 3 -/
def line1 (x y : ℝ) : Prop := y = 2 * x + 3

/-- The line y = x + 1 (the axis of symmetry) -/
def axis (x y : ℝ) : Prop := y = x + 1

/-- The line x = 2y (which is equivalent to x - 2y = 0) -/
def line2 (x y : ℝ) : Prop := x = 2 * y

theorem symmetry_of_lines : are_symmetric_lines line1 line2 axis := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_lines_l1192_119252


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1192_119247

theorem scientific_notation_equivalence : 
  1400000000 = (1.4 : ℝ) * (10 : ℝ) ^ 9 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1192_119247


namespace NUMINAMATH_CALUDE_viviana_vanilla_chips_l1192_119203

/-- Given the following conditions:
    - Viviana has 5 more chocolate chips than Susana
    - Susana has 3/4 as many vanilla chips as Viviana
    - Susana has 25 chocolate chips
    - They have a total of 90 chips together
    Prove that Viviana has 20 vanilla chips. -/
theorem viviana_vanilla_chips 
  (viviana_chocolate : ℕ) 
  (susana_chocolate : ℕ) 
  (viviana_vanilla : ℕ) 
  (susana_vanilla : ℕ) 
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : susana_chocolate = 25)
  (h4 : viviana_chocolate + susana_chocolate + viviana_vanilla + susana_vanilla = 90) :
  viviana_vanilla = 20 := by
  sorry

end NUMINAMATH_CALUDE_viviana_vanilla_chips_l1192_119203


namespace NUMINAMATH_CALUDE_ram_price_calculation_ram_price_theorem_l1192_119238

theorem ram_price_calculation (initial_price : ℝ) 
  (increase_percentage : ℝ) (decrease_percentage : ℝ) : ℝ :=
  let increased_price := initial_price * (1 + increase_percentage)
  let final_price := increased_price * (1 - decrease_percentage)
  final_price

theorem ram_price_theorem : 
  ram_price_calculation 50 0.3 0.2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ram_price_calculation_ram_price_theorem_l1192_119238


namespace NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l1192_119228

-- Define a rectangle
structure Rectangle where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

-- Define an inscribed rectangle
structure InscribedRectangle (R : Rectangle) where
  x : ℝ
  y : ℝ
  h_x_bounds : 0 ≤ x ∧ x ≤ R.a
  h_y_bounds : 0 ≤ y ∧ y ≤ R.b

-- Define the area of a rectangle
def area (R : Rectangle) : ℝ := R.a * R.b

-- Define the area of an inscribed rectangle
def inscribed_area (R : Rectangle) (IR : InscribedRectangle R) : ℝ :=
  IR.x * IR.y + (R.a - IR.x) * (R.b - IR.y)

-- Theorem statement
theorem inscribed_rectangles_area_sum (R : Rectangle) 
  (IR1 IR2 : InscribedRectangle R) (h : IR1.x = IR2.x) :
  inscribed_area R IR1 + inscribed_area R IR2 = area R := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l1192_119228


namespace NUMINAMATH_CALUDE_remainder_problem_l1192_119254

theorem remainder_problem (n : ℤ) (h : n ≡ 16 [ZMOD 30]) : 2 * n ≡ 2 [ZMOD 15] := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1192_119254


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l1192_119244

theorem two_numbers_with_given_means (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  Real.sqrt (a * b) = Real.sqrt 5 ∧ 
  (a + b) / 2 = 4 → 
  (a = 4 + Real.sqrt 11 ∧ b = 4 - Real.sqrt 11) ∨ 
  (a = 4 - Real.sqrt 11 ∧ b = 4 + Real.sqrt 11) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l1192_119244


namespace NUMINAMATH_CALUDE_workers_read_all_three_l1192_119290

/-- Represents the number of workers who have read books by different authors -/
structure BookReaders where
  total : ℕ
  saramago : ℕ
  kureishi : ℕ
  atwood : ℕ
  saramagoKureishi : ℕ
  allThree : ℕ

/-- The theorem to prove -/
theorem workers_read_all_three (r : BookReaders) : r.allThree = 6 :=
  by
  have h1 : r.total = 75 := by sorry
  have h2 : r.saramago = r.total / 2 := by sorry
  have h3 : r.kureishi = r.total / 4 := by sorry
  have h4 : r.atwood = r.total / 5 := by sorry
  have h5 : r.total - (r.saramago + r.kureishi + r.atwood - (r.saramagoKureishi + r.allThree)) = 
            r.saramago - (r.saramagoKureishi + r.allThree) - 1 := by sorry
  have h6 : r.saramagoKureishi = 2 * r.allThree := by sorry
  sorry

#check workers_read_all_three

end NUMINAMATH_CALUDE_workers_read_all_three_l1192_119290


namespace NUMINAMATH_CALUDE_mean_of_five_numbers_l1192_119269

theorem mean_of_five_numbers (a b c d e : ℚ) 
  (sum_condition : a + b + c + d + e = 3/4) :
  (a + b + c + d + e) / 5 = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_five_numbers_l1192_119269


namespace NUMINAMATH_CALUDE_factorization_equality_l1192_119230

theorem factorization_equality (x y z : ℝ) :
  x^2 - 4*y^2 - z^2 + 4*y*z = (x + 2*y - z) * (x - 2*y + z) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1192_119230


namespace NUMINAMATH_CALUDE_divisibility_condition_l1192_119262

theorem divisibility_condition (a b : ℕ+) : 
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) →
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1192_119262


namespace NUMINAMATH_CALUDE_distance_between_points_l1192_119215

theorem distance_between_points :
  let A : ℝ × ℝ := (8, -5)
  let B : ℝ × ℝ := (0, 10)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 17 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l1192_119215


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1192_119224

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x - 1 = 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1192_119224


namespace NUMINAMATH_CALUDE_log_one_fifth_25_l1192_119288

-- Define the logarithm function for an arbitrary base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem statement
theorem log_one_fifth_25 : log (1/5) 25 = -2 := by sorry

end NUMINAMATH_CALUDE_log_one_fifth_25_l1192_119288


namespace NUMINAMATH_CALUDE_no_double_apply_add_2015_l1192_119267

theorem no_double_apply_add_2015 : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := by
  sorry

end NUMINAMATH_CALUDE_no_double_apply_add_2015_l1192_119267


namespace NUMINAMATH_CALUDE_remainder_after_adding_2010_l1192_119220

theorem remainder_after_adding_2010 (n : ℤ) (h : n % 6 = 1) : (n + 2010) % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2010_l1192_119220


namespace NUMINAMATH_CALUDE_expression_evaluation_l1192_119236

theorem expression_evaluation : 12 - (-18) + (-7) - 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1192_119236


namespace NUMINAMATH_CALUDE_complex_cube_root_sum_l1192_119264

theorem complex_cube_root_sum (z : ℂ) (h1 : z^3 = 1) (h2 : z ≠ 1) :
  z^103 + z^104 + z^105 + z^106 + z^107 + z^108 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_sum_l1192_119264


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1192_119214

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: For a geometric sequence with first term a₁ and common ratio q, 
    the general term a_n is equal to a₁qⁿ⁻¹. -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) (q : ℝ) (a₁ : ℝ) (h : GeometricSequence a q) (h₁ : a 1 = a₁) :
  ∀ n : ℕ, a n = a₁ * q ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1192_119214


namespace NUMINAMATH_CALUDE_product_equals_one_l1192_119289

theorem product_equals_one (x₁ x₂ x₃ : ℝ) 
  (h_nonneg₁ : x₁ ≥ 0) (h_nonneg₂ : x₂ ≥ 0) (h_nonneg₃ : x₃ ≥ 0)
  (h_sum : x₁ + x₂ + x₃ = 1) :
  (x₁ + 3*x₂ + 5*x₃) * (x₁ + x₂/3 + x₃/5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l1192_119289


namespace NUMINAMATH_CALUDE_polar_equation_circle_l1192_119209

theorem polar_equation_circle (ρ : ℝ → ℝ → ℝ) (x y : ℝ) :
  (ρ = λ _ _ => 5) → (x^2 + y^2 = 25) :=
sorry

end NUMINAMATH_CALUDE_polar_equation_circle_l1192_119209


namespace NUMINAMATH_CALUDE_min_quadratic_function_l1192_119239

theorem min_quadratic_function :
  (∀ x : ℝ, x^2 - 2*x ≥ -1) ∧ (∃ x : ℝ, x^2 - 2*x = -1) := by
  sorry

end NUMINAMATH_CALUDE_min_quadratic_function_l1192_119239


namespace NUMINAMATH_CALUDE_winner_ate_15_ounces_l1192_119231

-- Define the weights of each ravioli type
def meat_ravioli_weight : ℝ := 1.5
def pumpkin_ravioli_weight : ℝ := 1.25
def cheese_ravioli_weight : ℝ := 1

-- Define the quantities eaten by Javier
def javier_meat_count : ℕ := 5
def javier_pumpkin_count : ℕ := 2
def javier_cheese_count : ℕ := 4

-- Define the quantity eaten by Javier's brother
def brother_pumpkin_count : ℕ := 12

-- Calculate total weight eaten by Javier
def javier_total_weight : ℝ := 
  meat_ravioli_weight * javier_meat_count +
  pumpkin_ravioli_weight * javier_pumpkin_count +
  cheese_ravioli_weight * javier_cheese_count

-- Calculate total weight eaten by Javier's brother
def brother_total_weight : ℝ := pumpkin_ravioli_weight * brother_pumpkin_count

-- Theorem: The winner ate 15 ounces of ravioli
theorem winner_ate_15_ounces : 
  max javier_total_weight brother_total_weight = 15 := by sorry

end NUMINAMATH_CALUDE_winner_ate_15_ounces_l1192_119231


namespace NUMINAMATH_CALUDE_twenty_fourth_digit_is_8_l1192_119294

-- Define the decimal representations of 1/7 and 1/9
def decimal_1_7 : ℚ := 1 / 7
def decimal_1_9 : ℚ := 1 / 9

-- Define the sum of the decimal representations
def sum_decimals : ℚ := decimal_1_7 + decimal_1_9

-- Function to get the nth digit after the decimal point
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem twenty_fourth_digit_is_8 :
  nth_digit_after_decimal sum_decimals 24 = 8 := by sorry

end NUMINAMATH_CALUDE_twenty_fourth_digit_is_8_l1192_119294


namespace NUMINAMATH_CALUDE_bus_car_ratio_l1192_119299

theorem bus_car_ratio (num_cars : ℕ) (num_buses : ℕ) : 
  num_cars = 85 →
  num_buses = num_cars - 80 →
  (num_buses : ℚ) / (num_cars : ℚ) = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_bus_car_ratio_l1192_119299


namespace NUMINAMATH_CALUDE_custom_operation_solution_l1192_119275

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 2 * a^2 - b

-- State the theorem
theorem custom_operation_solution :
  ∃ x : ℝ, (star 3 (star 4 x) = 8) ∧ (x = 22) := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_solution_l1192_119275


namespace NUMINAMATH_CALUDE_spider_permutations_l1192_119243

/-- Represents the number of legs a spider has -/
def num_legs : ℕ := 8

/-- Represents the number of items per leg -/
def items_per_leg : ℕ := 3

/-- Represents the total number of items -/
def total_items : ℕ := num_legs * items_per_leg

/-- Represents the number of valid orderings per leg -/
def valid_orderings_per_leg : ℕ := 3

/-- Represents the total number of orderings per leg -/
def total_orderings_per_leg : ℕ := 6

/-- Represents the probability of a valid ordering for one leg -/
def prob_valid_ordering : ℚ := 1 / 2

/-- Theorem: The number of valid permutations for a spider to put on its items
    with the given constraints is equal to 24! / 2^8 -/
theorem spider_permutations :
  (Nat.factorial total_items) / (2 ^ num_legs) =
  (Nat.factorial total_items) * (prob_valid_ordering ^ num_legs) :=
sorry

end NUMINAMATH_CALUDE_spider_permutations_l1192_119243


namespace NUMINAMATH_CALUDE_expression_simplification_l1192_119206

theorem expression_simplification (x y : ℝ) (h : x = -3) :
  x * (x - 4) * (x + 4) - (x + 3) * (x^2 - 6*x + 9) + 5*x^3*y^2 / (x^2*y^2) = -66 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1192_119206


namespace NUMINAMATH_CALUDE_monkeys_eating_birds_l1192_119246

theorem monkeys_eating_birds (initial_monkeys initial_birds : ℕ) 
  (h1 : initial_monkeys = 6)
  (h2 : initial_birds = 6)
  (h3 : ∃ (monkeys_ate : ℕ), 
    (initial_monkeys : ℚ) / (initial_monkeys + initial_birds - monkeys_ate) = 3/5) :
  ∃ (monkeys_ate : ℕ), monkeys_ate = 2 := by
sorry

end NUMINAMATH_CALUDE_monkeys_eating_birds_l1192_119246


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1192_119204

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1192_119204


namespace NUMINAMATH_CALUDE_periodic_function_l1192_119297

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f 1 :=
sorry

end NUMINAMATH_CALUDE_periodic_function_l1192_119297


namespace NUMINAMATH_CALUDE_max_socks_pulled_correct_l1192_119249

/-- Represents the state of socks in the drawer and pulled out -/
structure SockState where
  white_in_drawer : ℕ
  black_in_drawer : ℕ
  white_pulled : ℕ
  black_pulled : ℕ

/-- The initial state of socks -/
def initial_state : SockState :=
  { white_in_drawer := 8
  , black_in_drawer := 15
  , white_pulled := 0
  , black_pulled := 0 }

/-- Predicate to check if more black socks than white socks have been pulled -/
def more_black_than_white (state : SockState) : Prop :=
  state.black_pulled > state.white_pulled

/-- The maximum number of socks that can be pulled -/
def max_socks_pulled : ℕ := 17

/-- Theorem stating the maximum number of socks that can be pulled -/
theorem max_socks_pulled_correct :
  ∀ (state : SockState),
    state.white_in_drawer + state.black_in_drawer + state.white_pulled + state.black_pulled = 23 →
    state.white_pulled + state.black_pulled ≤ max_socks_pulled →
    ¬(more_black_than_white state) :=
  sorry

#check max_socks_pulled_correct

end NUMINAMATH_CALUDE_max_socks_pulled_correct_l1192_119249


namespace NUMINAMATH_CALUDE_perpendicular_lines_line_slope_l1192_119287

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
theorem perpendicular_lines (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) (hB₁ : B₁ ≠ 0) (hB₂ : B₂ ≠ 0) :
  (A₁ * x + B₁ * y + C₁ = 0 ∧ A₂ * x + B₂ * y + C₂ = 0) →
  ((-A₁ / B₁) * (-A₂ / B₂) = -1 ↔ (A₁ * A₂ + B₁ * B₂ = 0)) :=
by sorry

/-- The slope of a line Ax + By + C = 0 is -A/B -/
theorem line_slope (A B C : ℝ) (hB : B ≠ 0) :
  (A * x + B * y + C = 0) → (y = (-A / B) * x - C / B) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_line_slope_l1192_119287


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1192_119222

/-- Represents the time to fill a cistern with two taps -/
def time_to_fill_cistern (fill_rate : ℚ) (empty_rate : ℚ) : ℚ :=
  1 / (fill_rate - empty_rate)

/-- Theorem: The time to fill the cistern is 12 hours -/
theorem cistern_fill_time :
  let fill_rate : ℚ := 1/6
  let empty_rate : ℚ := 1/12
  time_to_fill_cistern fill_rate empty_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1192_119222


namespace NUMINAMATH_CALUDE_johns_pill_cost_l1192_119210

/-- Calculates the out-of-pocket cost for pills in a 30-day month given the following conditions:
  * Daily pill requirement
  * Cost per pill
  * Insurance coverage percentage
  * Number of days in a month
-/
def outOfPocketCost (dailyPills : ℕ) (costPerPill : ℚ) (insuranceCoverage : ℚ) (daysInMonth : ℕ) : ℚ :=
  let totalPills := dailyPills * daysInMonth
  let totalCost := totalPills * costPerPill
  let insuranceAmount := totalCost * insuranceCoverage
  totalCost - insuranceAmount

/-- Proves that given the specified conditions, John's out-of-pocket cost for pills in a 30-day month is $54 -/
theorem johns_pill_cost :
  outOfPocketCost 2 (3/2) (2/5) 30 = 54 := by
  sorry

end NUMINAMATH_CALUDE_johns_pill_cost_l1192_119210


namespace NUMINAMATH_CALUDE_side_significant_digits_l1192_119218

-- Define the area of the square
def square_area : Real := 3.0625

-- Define the precision of the area measurement
def area_precision : Real := 0.001

-- Define a function to calculate the number of significant digits
def count_significant_digits (x : Real) : Nat :=
  sorry

-- Theorem statement
theorem side_significant_digits :
  let side := Real.sqrt square_area
  count_significant_digits side = 3 :=
sorry

end NUMINAMATH_CALUDE_side_significant_digits_l1192_119218


namespace NUMINAMATH_CALUDE_sequence_non_positive_l1192_119219

theorem sequence_non_positive
  (n : ℕ)
  (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k : ℕ, 1 ≤ k ∧ k < n → a (k - 1) + a (k + 1) - 2 * a k ≥ 0) :
  ∀ k : ℕ, k ≤ n → a k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l1192_119219


namespace NUMINAMATH_CALUDE_garden_perimeter_l1192_119283

/-- Represents a rectangular shape with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Proves that the perimeter of the garden is 56 meters -/
theorem garden_perimeter (garden : Rectangle) (playground : Rectangle) : 
  garden.width = 16 → 
  playground.length = 16 → 
  garden.area = playground.area → 
  garden.perimeter = 56 → 
  garden.perimeter = 56 := by
  sorry

#check garden_perimeter

end NUMINAMATH_CALUDE_garden_perimeter_l1192_119283


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l1192_119248

theorem largest_gcd_of_sum_1008 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 1008 ∧ Nat.gcd x y = 504 ∧ 
  ∀ (c d : ℕ), c > 0 → d > 0 → c + d = 1008 → Nat.gcd c d ≤ 504 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l1192_119248


namespace NUMINAMATH_CALUDE_stating_sticks_at_100th_stage_l1192_119233

/-- 
Given a sequence where:
- The first term is 4
- Each subsequent term increases by 4
This function calculates the nth term of the sequence
-/
def sticksAtStage (n : ℕ) : ℕ := 4 + 4 * (n - 1)

/-- 
Theorem stating that the 100th stage of the stick pattern contains 400 sticks
-/
theorem sticks_at_100th_stage : sticksAtStage 100 = 400 := by sorry

end NUMINAMATH_CALUDE_stating_sticks_at_100th_stage_l1192_119233


namespace NUMINAMATH_CALUDE_squared_sum_ge_double_product_l1192_119260

theorem squared_sum_ge_double_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b ∧ (a^2 + b^2 = 2*a*b ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_squared_sum_ge_double_product_l1192_119260


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1192_119241

theorem smallest_k_no_real_roots :
  ∀ k : ℤ, (∀ x : ℝ, 3*x*(k*x-5) - 2*x^2 + 8 ≠ 0) → k ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1192_119241


namespace NUMINAMATH_CALUDE_two_distinct_values_of_T_l1192_119208

theorem two_distinct_values_of_T (n : ℤ) : 
  let i : ℂ := Complex.I
  let T : ℂ := i^(2*n) + i^(-2*n) + Real.cos (n * Real.pi)
  ∃ (a b : ℂ), ∀ (m : ℤ), 
    (let T_m : ℂ := i^(2*m) + i^(-2*m) + Real.cos (m * Real.pi)
     T_m = a ∨ T_m = b) ∧ a ≠ b :=
sorry

end NUMINAMATH_CALUDE_two_distinct_values_of_T_l1192_119208


namespace NUMINAMATH_CALUDE_blueberry_picking_l1192_119293

theorem blueberry_picking (annie kathryn ben : ℕ) 
  (h1 : kathryn = annie + 2)
  (h2 : ben = kathryn - 3)
  (h3 : annie + kathryn + ben = 25) :
  annie = 8 := by
sorry

end NUMINAMATH_CALUDE_blueberry_picking_l1192_119293


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1192_119279

theorem max_value_of_expression (x y : ℝ) (h : x * y > 0) :
  (∃ (z : ℝ), ∀ (a b : ℝ), a * b > 0 → x / (x + y) + 2 * y / (x + 2 * y) ≤ z) ∧
  (x / (x + y) + 2 * y / (x + 2 * y) ≤ 4 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1192_119279


namespace NUMINAMATH_CALUDE_sphere_tangent_planes_properties_l1192_119274

/-- Given a sphere with radius r, this theorem proves various geometric properties related to
    tangent planes, spherical caps, and conical frustums. -/
theorem sphere_tangent_planes_properties (r : ℝ) (hr : r > 0) :
  ∃ (locus_radius : ℝ) (cap_area conical_area : ℝ),
    -- The locus of points P forms a sphere with radius r√3
    locus_radius = r * Real.sqrt 3 ∧
    -- The surface area of the smaller spherical cap
    cap_area = 2 * Real.pi * r^2 * (1 - Real.sqrt (2/3)) ∧
    -- The surface area of the conical frustum
    conical_area = Real.pi * r^2 * (2 * Real.sqrt 3 / 3) ∧
    -- The ratio of the two surface areas
    cap_area / conical_area = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_tangent_planes_properties_l1192_119274


namespace NUMINAMATH_CALUDE_count_divisible_sum_l1192_119258

theorem count_divisible_sum : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (10 * n) % (n * (n + 1) / 2) = 0) ∧ 
    (∀ n ∉ S, n > 0 → (10 * n) % (n * (n + 1) / 2) ≠ 0) ∧ 
    Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_sum_l1192_119258


namespace NUMINAMATH_CALUDE_orange_apple_cost_l1192_119205

/-- The cost of oranges and apples given specific quantities and prices -/
theorem orange_apple_cost (orange_price apple_price : ℕ) 
  (h1 : 6 * orange_price + 5 * apple_price = 419)
  (h2 : orange_price = 29)
  (h3 : apple_price = 29) :
  5 * orange_price + 7 * apple_price = 348 := by
  sorry

#check orange_apple_cost

end NUMINAMATH_CALUDE_orange_apple_cost_l1192_119205


namespace NUMINAMATH_CALUDE_exist_n_points_with_integer_distances_l1192_119253

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Check if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Main theorem statement -/
theorem exist_n_points_with_integer_distances (n : ℕ) (h : n ≥ 2) :
  ∃ (points : Fin n → Point),
    (∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬areCollinear (points i) (points j) (points k)) ∧
    (∀ (i j : Fin n), i ≠ j → ∃ (d : ℤ), squaredDistance (points i) (points j) = d^2) :=
by sorry

end NUMINAMATH_CALUDE_exist_n_points_with_integer_distances_l1192_119253


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l1192_119276

theorem smallest_lcm_with_gcd_five (k ℓ : ℕ) : 
  k ≥ 1000 → k < 10000 → ℓ ≥ 1000 → ℓ < 10000 → Nat.gcd k ℓ = 5 → 
  Nat.lcm k ℓ ≥ 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l1192_119276


namespace NUMINAMATH_CALUDE_count_multiples_of_four_l1192_119240

theorem count_multiples_of_four : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n > 50 ∧ n < 300) (Finset.range 300)).card = 62 :=
by sorry

end NUMINAMATH_CALUDE_count_multiples_of_four_l1192_119240


namespace NUMINAMATH_CALUDE_vector_sum_collinear_points_l1192_119217

/-- Given points A, B, C are collinear, O is not on their line, and 
    p⃗OA + q⃗OB + r⃗OC = 0⃗, then p + q + r = 0 -/
theorem vector_sum_collinear_points 
  (O A B C : EuclideanSpace ℝ (Fin 3))
  (p q r : ℝ) :
  Collinear ℝ ({A, B, C} : Set (EuclideanSpace ℝ (Fin 3))) →
  O ∉ affineSpan ℝ {A, B, C} →
  p • (A - O) + q • (B - O) + r • (C - O) = 0 →
  p + q + r = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_collinear_points_l1192_119217


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1192_119268

theorem polynomial_coefficient_sum (a b c d : ℤ) : 
  (∀ x : ℚ, (3*x + 2) * (2*x - 3) * (x - 4) = a*x^3 + b*x^2 + c*x + d) →
  a - b + c - d = 25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1192_119268


namespace NUMINAMATH_CALUDE_perfect_squares_exist_l1192_119280

theorem perfect_squares_exist : ∃ (a b c d : ℤ),
  (∃ (x : ℤ), (a + b) = x^2) ∧
  (∃ (y : ℤ), (a + c) = y^2) ∧
  (∃ (z : ℤ), (a + d) = z^2) ∧
  (∃ (w : ℤ), (b + c) = w^2) ∧
  (∃ (v : ℤ), (b + d) = v^2) ∧
  (∃ (u : ℤ), (c + d) = u^2) ∧
  (∃ (t : ℤ), (a + b + c + d) = t^2) :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_exist_l1192_119280


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1192_119211

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | 2*x - 3 < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 3/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1192_119211


namespace NUMINAMATH_CALUDE_right_triangle_area_l1192_119202

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 10 * Real.sqrt 2 →
  angle = 45 * π / 180 →
  let area := (hypotenuse^2 / 4)
  area = 50 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1192_119202


namespace NUMINAMATH_CALUDE_egg_weight_probability_l1192_119298

theorem egg_weight_probability (p_less_than_30 : ℝ) (p_between_30_and_40 : ℝ) 
  (h1 : p_less_than_30 = 0.3)
  (h2 : p_between_30_and_40 = 0.5) :
  1 - p_less_than_30 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_egg_weight_probability_l1192_119298


namespace NUMINAMATH_CALUDE_expand_expression_l1192_119278

theorem expand_expression (x : ℝ) : 2 * (5 * x^2 - 3 * x + 4 - x^3) = -2 * x^3 + 10 * x^2 - 6 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1192_119278


namespace NUMINAMATH_CALUDE_geometric_sequence_308th_term_l1192_119229

theorem geometric_sequence_308th_term
  (a₁ : ℝ)
  (a₂ : ℝ)
  (h₁ : a₁ = 10)
  (h₂ : a₂ = -10) :
  let r := a₂ / a₁
  let aₙ := a₁ * r^(308 - 1)
  aₙ = -10 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_308th_term_l1192_119229


namespace NUMINAMATH_CALUDE_max_gcd_sum_1729_l1192_119227

theorem max_gcd_sum_1729 :
  ∃ (x y : ℕ+), x + y = 1729 ∧ 
  ∀ (a b : ℕ+), a + b = 1729 → Nat.gcd x y ≥ Nat.gcd a b ∧
  Nat.gcd x y = 247 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1729_l1192_119227


namespace NUMINAMATH_CALUDE_triangle_properties_l1192_119235

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The radius of the circumcircle of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

theorem triangle_properties (t : Triangle) 
  (h1 : t.c ≥ t.a ∧ t.c ≥ t.b) 
  (h2 : t.b = Real.sqrt 3 * circumradius t)
  (h3 : t.b * Real.sin t.B = (t.a + t.c) * Real.sin t.A)
  (h4 : 0 < t.A ∧ t.A < Real.pi / 2)
  (h5 : 0 < t.B ∧ t.B < Real.pi / 2)
  (h6 : 0 < t.C ∧ t.C < Real.pi / 2) :
  t.B = Real.pi / 3 ∧ t.A = Real.pi / 6 ∧ t.C = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1192_119235


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_t_for_f_geq_t_squared_minus_t_l1192_119272

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 1|

-- Theorem for part I
theorem solution_set_f_leq_x : 
  {x : ℝ | f x ≤ x} = {x : ℝ | x ≥ 1/4} :=
sorry

-- Theorem for part II
theorem range_of_t_for_f_geq_t_squared_minus_t : 
  {t : ℝ | ∀ x ∈ Set.Icc (-2) (-1), f x ≥ t^2 - t} = 
  Set.Icc ((1 - Real.sqrt 5) / 2) ((1 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_t_for_f_geq_t_squared_minus_t_l1192_119272


namespace NUMINAMATH_CALUDE_star_one_neg_three_l1192_119234

-- Define the ※ operation
def star (a b : ℝ) : ℝ := 2 * a * b - b^2

-- Theorem statement
theorem star_one_neg_three : star 1 (-3) = -15 := by sorry

end NUMINAMATH_CALUDE_star_one_neg_three_l1192_119234


namespace NUMINAMATH_CALUDE_arrangements_with_adjacent_pair_l1192_119263

-- Define the number of students
def total_students : ℕ := 5

-- Define the function to calculate permutations
def permutations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

-- Define the theorem
theorem arrangements_with_adjacent_pair :
  permutations 4 4 * permutations 2 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_adjacent_pair_l1192_119263


namespace NUMINAMATH_CALUDE_third_grade_girls_sample_l1192_119271

theorem third_grade_girls_sample (total_students : ℕ) (first_grade : ℕ) (second_grade : ℕ) (third_grade : ℕ)
  (first_boys : ℕ) (first_girls : ℕ) (second_boys : ℕ) (second_girls : ℕ) (third_boys : ℕ) (third_girls : ℕ)
  (sample_size : ℕ) :
  total_students = 3000 →
  first_grade = 800 →
  second_grade = 1000 →
  third_grade = 1200 →
  first_boys = 500 →
  first_girls = 300 →
  second_boys = 600 →
  second_girls = 400 →
  third_boys = 800 →
  third_girls = 400 →
  sample_size = 150 →
  first_grade + second_grade + third_grade = total_students →
  first_boys + first_girls = first_grade →
  second_boys + second_girls = second_grade →
  third_boys + third_girls = third_grade →
  (third_grade : ℚ) / (total_students : ℚ) * (sample_size : ℚ) * (third_girls : ℚ) / (third_grade : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_grade_girls_sample_l1192_119271


namespace NUMINAMATH_CALUDE_circle_equation_k_range_l1192_119265

/-- Proves that for the equation x^2 + y^2 - 2x + 2k + 3 = 0 to represent a circle,
    k must be in the range (-∞, -1). -/
theorem circle_equation_k_range :
  ∀ (k : ℝ), (∃ (x y : ℝ), x^2 + y^2 - 2*x + 2*k + 3 = 0 ∧ 
    ∃ (h r : ℝ), ∀ (x' y' : ℝ), (x' - h)^2 + (y' - r)^2 = (x - h)^2 + (y - r)^2) 
  ↔ k < -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_k_range_l1192_119265


namespace NUMINAMATH_CALUDE_pigeonhole_principle_sports_l1192_119237

theorem pigeonhole_principle_sports (n : ℕ) (h : n = 50) :
  ∃ (same_choices : ℕ), same_choices ≥ 3 ∧
  (∀ (choices : Fin n → Fin 4 × Fin 3 × Fin 2),
   ∃ (subset : Finset (Fin n)),
   subset.card = same_choices ∧
   ∀ (i j : Fin n), i ∈ subset → j ∈ subset → choices i = choices j) :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_sports_l1192_119237


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l1192_119226

theorem least_positive_integer_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 5 = 2 ∧
  n % 4 = 2 ∧
  n % 3 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m % 5 = 2 ∧ m % 4 = 2 ∧ m % 3 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l1192_119226


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l1192_119251

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) →
  a = b^n :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l1192_119251
