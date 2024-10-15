import Mathlib

namespace NUMINAMATH_CALUDE_three_letter_words_with_E_count_l3466_346692

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E'}
def word_length : Nat := 3

def total_words : Nat := alphabet.card ^ word_length
def words_without_E : Nat := (alphabet.card - 1) ^ word_length

theorem three_letter_words_with_E_count :
  total_words - words_without_E = 61 := by
  sorry

end NUMINAMATH_CALUDE_three_letter_words_with_E_count_l3466_346692


namespace NUMINAMATH_CALUDE_prob_sum_15_equals_11_663_l3466_346669

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards of each rank (2 through 10) in a standard deck -/
def cards_per_rank : ℕ := 4

/-- The set of possible ranks that can sum to 15 -/
def valid_ranks : Finset ℕ := {6, 7, 8}

/-- The probability of selecting two number cards that sum to 15 from a standard deck -/
def prob_sum_15 : ℚ :=
  (cards_per_rank * cards_per_rank * 2 + cards_per_rank * (cards_per_rank - 1)) / (deck_size * (deck_size - 1))

theorem prob_sum_15_equals_11_663 : prob_sum_15 = 11 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_15_equals_11_663_l3466_346669


namespace NUMINAMATH_CALUDE_trevor_coin_count_l3466_346673

/-- Given that Trevor counted 29 quarters and has 48 more coins in total than quarters,
    prove that the total number of coins Trevor counted is 77. -/
theorem trevor_coin_count :
  let quarters : ℕ := 29
  let extra_coins : ℕ := 48
  quarters + extra_coins = 77
  := by sorry

end NUMINAMATH_CALUDE_trevor_coin_count_l3466_346673


namespace NUMINAMATH_CALUDE_octadecagon_diagonals_l3466_346614

/-- The number of sides in an octadecagon -/
def octadecagon_sides : ℕ := 18

/-- Formula for the number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in an octadecagon is 135 -/
theorem octadecagon_diagonals : 
  num_diagonals octadecagon_sides = 135 := by
  sorry

end NUMINAMATH_CALUDE_octadecagon_diagonals_l3466_346614


namespace NUMINAMATH_CALUDE_quadratic_min_diff_l3466_346689

/-- The quadratic function f(x) = ax² - 2020x + 2021 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2020 * x + 2021

/-- The theorem stating that if the minimum difference between max and min values
    of f on any 2-unit interval is 2, then a must be 2 -/
theorem quadratic_min_diff (a : ℝ) (h_pos : a > 0) :
  (∀ t : ℝ, ∃ M N : ℝ,
    (∀ x ∈ Set.Icc (t - 1) (t + 1), f a x ≤ M) ∧
    (∀ x ∈ Set.Icc (t - 1) (t + 1), N ≤ f a x) ∧
    (∀ K L : ℝ,
      (∀ x ∈ Set.Icc (t - 1) (t + 1), f a x ≤ K) →
      (∀ x ∈ Set.Icc (t - 1) (t + 1), L ≤ f a x) →
      2 ≤ K - L)) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_min_diff_l3466_346689


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3466_346645

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (x + 1)}

-- State the theorem
theorem intersection_complement_theorem : A ∩ (U \ B) = {x | -2 ≤ x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3466_346645


namespace NUMINAMATH_CALUDE_probability_one_or_two_pascal_l3466_346639

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) := sorry

/-- Counts the occurrences of a specific value in Pascal's Triangle up to n rows -/
def countOccurrences (n : ℕ) (value : ℕ) : ℕ := sorry

/-- Calculates the total number of elements in Pascal's Triangle up to n rows -/
def totalElements (n : ℕ) : ℕ := sorry

/-- The main theorem stating the probability of selecting 1 or 2 from the first 20 rows of Pascal's Triangle -/
theorem probability_one_or_two_pascal : 
  (countOccurrences 20 1 + countOccurrences 20 2) / totalElements 20 = 37 / 105 := by sorry

end NUMINAMATH_CALUDE_probability_one_or_two_pascal_l3466_346639


namespace NUMINAMATH_CALUDE_selections_with_former_eq_2850_l3466_346668

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of coordinators to be selected -/
def num_coordinators : ℕ := 4

/-- The total number of members -/
def total_members : ℕ := 18

/-- The number of former coordinators -/
def former_coordinators : ℕ := 8

/-- The number of selections including at least one former coordinator -/
def selections_with_former : ℕ :=
  choose total_members num_coordinators - choose (total_members - former_coordinators) num_coordinators

theorem selections_with_former_eq_2850 : selections_with_former = 2850 := by sorry

end NUMINAMATH_CALUDE_selections_with_former_eq_2850_l3466_346668


namespace NUMINAMATH_CALUDE_factorization_equality_l3466_346685

theorem factorization_equality (a b : ℝ) : (a^2 + b^2)^2 - 4*a^2*b^2 = (a + b)^2 * (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3466_346685


namespace NUMINAMATH_CALUDE_square_plot_side_length_l3466_346691

theorem square_plot_side_length (area : ℝ) (side : ℝ) : 
  area = 2550.25 → side * side = area → side = 50.5 := by sorry

end NUMINAMATH_CALUDE_square_plot_side_length_l3466_346691


namespace NUMINAMATH_CALUDE_prism_18_edges_8_faces_l3466_346633

/-- A prism is a polyhedron with two congruent parallel faces (bases) and other faces (lateral faces) that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  2 + (p.edges / 3)

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ p : Prism, p.edges = 18 → num_faces p = 8 := by
  sorry


end NUMINAMATH_CALUDE_prism_18_edges_8_faces_l3466_346633


namespace NUMINAMATH_CALUDE_lizzy_initial_money_l3466_346634

def loan_amount : ℝ := 15
def interest_rate : ℝ := 0.20
def final_amount : ℝ := 33

theorem lizzy_initial_money :
  ∃ (initial_money : ℝ),
    initial_money = loan_amount ∧
    final_amount = initial_money + loan_amount + (interest_rate * loan_amount) :=
by sorry

end NUMINAMATH_CALUDE_lizzy_initial_money_l3466_346634


namespace NUMINAMATH_CALUDE_parabola_point_order_l3466_346636

/-- Theorem: For a parabola y = ax² - 2ax + 3 with a > 0, and points A(-1, y₁), B(2, y₂), C(4, y₃) on the parabola, prove that y₂ < y₁ < y₃ -/
theorem parabola_point_order (a : ℝ) (y₁ y₂ y₃ : ℝ) 
  (ha : a > 0)
  (hy₁ : y₁ = a * (-1)^2 - 2 * a * (-1) + 3)
  (hy₂ : y₂ = a * 2^2 - 2 * a * 2 + 3)
  (hy₃ : y₃ = a * 4^2 - 2 * a * 4 + 3) :
  y₂ < y₁ ∧ y₁ < y₃ :=
sorry

end NUMINAMATH_CALUDE_parabola_point_order_l3466_346636


namespace NUMINAMATH_CALUDE_minimum_translation_l3466_346680

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.sin x + Real.cos x

theorem minimum_translation (a : ℝ) :
  (∀ x, f a (x - π/4) = f a (π/4 + (π/4 - x))) →
  ∃ φ : ℝ, φ > 0 ∧
    (∀ x, f a (x - φ) = f a (-x)) ∧
    (∀ ψ, ψ > 0 ∧ (∀ x, f a (x - ψ) = f a (-x)) → φ ≤ ψ) ∧
    φ = 3*π/4 :=
sorry

end NUMINAMATH_CALUDE_minimum_translation_l3466_346680


namespace NUMINAMATH_CALUDE_product_sum_relation_l3466_346679

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 14 → b = 8 → b - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3466_346679


namespace NUMINAMATH_CALUDE_negative_one_minus_two_times_negative_two_l3466_346607

theorem negative_one_minus_two_times_negative_two : -1 - 2 * (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_minus_two_times_negative_two_l3466_346607


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l3466_346670

theorem repeating_decimal_subtraction : 
  let a : ℚ := 234 / 999
  let b : ℚ := 567 / 999
  let c : ℚ := 891 / 999
  a - b - c = -1224 / 999 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l3466_346670


namespace NUMINAMATH_CALUDE_janet_total_distance_l3466_346608

/-- Represents Janet's training schedule for a week --/
structure WeekSchedule where
  running_days : Nat
  running_miles : Nat
  cycling_days : Nat
  cycling_miles : Nat
  swimming_days : Nat
  swimming_miles : Nat
  hiking_days : Nat
  hiking_miles : Nat

/-- Calculates the total distance for a given week schedule --/
def weekTotalDistance (schedule : WeekSchedule) : Nat :=
  schedule.running_days * schedule.running_miles +
  schedule.cycling_days * schedule.cycling_miles +
  schedule.swimming_days * schedule.swimming_miles +
  schedule.hiking_days * schedule.hiking_miles

/-- Janet's training schedule for three weeks --/
def janetSchedule : List WeekSchedule := [
  { running_days := 5, running_miles := 8, cycling_days := 3, cycling_miles := 7, swimming_days := 0, swimming_miles := 0, hiking_days := 0, hiking_miles := 0 },
  { running_days := 4, running_miles := 10, cycling_days := 0, cycling_miles := 0, swimming_days := 2, swimming_miles := 2, hiking_days := 0, hiking_miles := 0 },
  { running_days := 5, running_miles := 6, cycling_days := 0, cycling_miles := 0, swimming_days := 0, swimming_miles := 0, hiking_days := 2, hiking_miles := 3 }
]

/-- Theorem: Janet's total training distance is 141 miles --/
theorem janet_total_distance :
  (janetSchedule.map weekTotalDistance).sum = 141 := by
  sorry

end NUMINAMATH_CALUDE_janet_total_distance_l3466_346608


namespace NUMINAMATH_CALUDE_complex_quadrant_l3466_346622

theorem complex_quadrant (z : ℂ) (h : (z - 1) * Complex.I = 1 + 2 * Complex.I) :
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3466_346622


namespace NUMINAMATH_CALUDE_infinite_equal_terms_l3466_346640

/-- An infinite sequence with two ends satisfying the given recurrence relation -/
def InfiniteSequence := ℤ → ℝ

/-- The recurrence relation for the sequence -/
def SatisfiesRecurrence (a : InfiniteSequence) : Prop :=
  ∀ k : ℤ, a k = (1/4) * (a (k-1) + a (k+1))

theorem infinite_equal_terms
  (a : InfiniteSequence)
  (h_recurrence : SatisfiesRecurrence a)
  (h_equal : ∃ k p : ℤ, k < p ∧ a k = a p) :
  ∀ n : ℕ, ∃ k p : ℤ, k < p ∧ a (k - n) = a (p + n) :=
sorry

end NUMINAMATH_CALUDE_infinite_equal_terms_l3466_346640


namespace NUMINAMATH_CALUDE_no_solution_iff_a_less_than_one_l3466_346684

theorem no_solution_iff_a_less_than_one (a : ℝ) :
  (∀ x : ℝ, |x - 1| + x > a) ↔ a < 1 := by
sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_less_than_one_l3466_346684


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3466_346654

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (1 + i) / (1 + i^3) = i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3466_346654


namespace NUMINAMATH_CALUDE_common_divisors_84_90_l3466_346661

theorem common_divisors_84_90 : 
  (Finset.filter (λ x => x ∣ 84 ∧ x ∣ 90) (Finset.range (max 84 90 + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_84_90_l3466_346661


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l3466_346623

theorem abs_inequality_solution_set (x : ℝ) : 
  |3*x + 1| > 2 ↔ x > 1/3 ∨ x < -1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l3466_346623


namespace NUMINAMATH_CALUDE_coefficient_proof_l3466_346656

theorem coefficient_proof (x : ℕ) (some_number : ℕ) :
  x = 13 →
  (2^x) - (2^(x-2)) = some_number * (2^11) →
  some_number = 3 := by
sorry

end NUMINAMATH_CALUDE_coefficient_proof_l3466_346656


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l3466_346637

/-- Parabola defined by y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Line with slope √3 passing through a point -/
def line_with_slope_sqrt3 (x y x0 y0 : ℝ) : Prop :=
  y - y0 = Real.sqrt 3 * (x - x0)

/-- Focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Definition of the length of a line segment on the parabola -/
def segment_length (x1 x2 : ℝ) : ℝ := x1 + x2 + 2

theorem parabola_intersection_length :
  ∀ A B : ℝ × ℝ,
  (∃ x1 x2 y1 y2 : ℝ,
    A = (x1, y1) ∧ B = (x2, y2) ∧
    parabola x1 y1 ∧ parabola x2 y2 ∧
    line_with_slope_sqrt3 x1 y1 focus.1 focus.2 ∧
    line_with_slope_sqrt3 x2 y2 focus.1 focus.2) →
  segment_length A.1 B.1 = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l3466_346637


namespace NUMINAMATH_CALUDE_triangle_problem_l3466_346606

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (0 < A) ∧ (A < Real.pi) ∧
  (0 < B) ∧ (B < Real.pi) ∧
  (0 < C) ∧ (C < Real.pi) ∧
  (A + B + C = Real.pi) ∧
  (Real.sin A)^2 - (Real.sin B)^2 - (Real.sin C)^2 = Real.sin B * Real.sin C ∧
  a = 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  A = 2 * Real.pi / 3 ∧
  (a + b + c) ≤ 3 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3466_346606


namespace NUMINAMATH_CALUDE_unique_perfect_square_l3466_346688

def f (k : ℕ) : ℕ := 2^k + 8*k + 5

theorem unique_perfect_square : ∃! k : ℕ, ∃ n : ℕ, f k = n^2 ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_l3466_346688


namespace NUMINAMATH_CALUDE_angle_Y_is_50_l3466_346609

-- Define the angles in the geometric figure
def angle_X : ℝ := 120
def angle_Y : ℝ := 50
def angle_Z : ℝ := 180 - angle_X

-- Theorem statement
theorem angle_Y_is_50 : 
  angle_X = 120 →
  angle_Y = 50 →
  angle_Z = 180 - angle_X →
  angle_Y = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_Y_is_50_l3466_346609


namespace NUMINAMATH_CALUDE_roots_of_polynomials_l3466_346663

theorem roots_of_polynomials (r : ℝ) : 
  r^2 - 2*r - 1 = 0 → r^5 - 29*r - 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomials_l3466_346663


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3466_346660

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3466_346660


namespace NUMINAMATH_CALUDE_dogs_not_doing_anything_l3466_346662

theorem dogs_not_doing_anything (total : ℕ) (running : ℕ) (playing : ℕ) (barking : ℕ) : 
  total = 88 → 
  running = 12 → 
  playing = total / 2 → 
  barking = total / 4 → 
  total - (running + playing + barking) = 10 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_doing_anything_l3466_346662


namespace NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_l3466_346676

/-- The probability of rolling a 1 on a fair six-sided die -/
def p_one : ℚ := 1/6

/-- The probability of not rolling a 1 on a fair six-sided die -/
def p_not_one : ℚ := 5/6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of times we want to roll a 1 -/
def num_ones : ℕ := 4

/-- The number of ways to choose the positions for the non-1 roll -/
def num_arrangements : ℕ := 5

theorem probability_four_ones_in_five_rolls :
  num_arrangements * p_one^num_ones * p_not_one^(num_rolls - num_ones) = 25/7776 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_l3466_346676


namespace NUMINAMATH_CALUDE_reads_two_days_per_week_l3466_346620

/-- A person's reading habits over a period of weeks -/
structure ReadingHabits where
  booksPerDay : ℕ
  totalBooks : ℕ
  totalWeeks : ℕ

/-- Calculate the number of days per week a person reads based on their reading habits -/
def daysPerWeek (habits : ReadingHabits) : ℚ :=
  (habits.totalBooks / habits.booksPerDay : ℚ) / habits.totalWeeks

/-- Theorem: Given the specific reading habits, prove that the person reads 2 days per week -/
theorem reads_two_days_per_week (habits : ReadingHabits)
  (h1 : habits.booksPerDay = 4)
  (h2 : habits.totalBooks = 48)
  (h3 : habits.totalWeeks = 6) :
  daysPerWeek habits = 2 := by
  sorry

end NUMINAMATH_CALUDE_reads_two_days_per_week_l3466_346620


namespace NUMINAMATH_CALUDE_inequality_proof_l3466_346694

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3466_346694


namespace NUMINAMATH_CALUDE_larger_number_in_ratio_l3466_346651

theorem larger_number_in_ratio (a b : ℕ+) : 
  a.val * 5 = b.val * 2 →  -- ratio condition
  Nat.lcm a.val b.val = 160 →  -- LCM condition
  b = 160 := by  -- conclusion: larger number is 160
sorry

end NUMINAMATH_CALUDE_larger_number_in_ratio_l3466_346651


namespace NUMINAMATH_CALUDE_playground_students_count_l3466_346674

/-- Represents the seating arrangement on the playground -/
structure PlaygroundSeating where
  left : Nat
  right : Nat
  front : Nat
  back : Nat

/-- Calculates the total number of students on the playground -/
def totalStudents (s : PlaygroundSeating) : Nat :=
  ((s.left + s.right - 1) * (s.front + s.back - 1))

/-- Theorem stating the total number of students on the playground -/
theorem playground_students_count (yujeong : PlaygroundSeating) 
  (h1 : yujeong.left = 12)
  (h2 : yujeong.right = 11)
  (h3 : yujeong.front = 18)
  (h4 : yujeong.back = 8) :
  totalStudents yujeong = 550 := by
  sorry

#check playground_students_count

end NUMINAMATH_CALUDE_playground_students_count_l3466_346674


namespace NUMINAMATH_CALUDE_binary_to_base5_conversion_l3466_346667

-- Define the binary number 1101₂
def binary_num : ℕ := 13

-- Define the base-5 number 23₅
def base5_num : ℕ := 2 * 5 + 3

-- Theorem stating the equality of the two representations
theorem binary_to_base5_conversion :
  binary_num = base5_num := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base5_conversion_l3466_346667


namespace NUMINAMATH_CALUDE_air_conditioner_price_l3466_346659

/-- The selling price per unit of the air conditioner fan before the regulation. -/
def price_before : ℝ := 880

/-- The subsidy amount per unit after the regulation. -/
def subsidy : ℝ := 80

/-- The total amount spent on purchases after the regulation. -/
def total_spent : ℝ := 60000

/-- The ratio of units purchased after the regulation to before. -/
def purchase_ratio : ℝ := 1.1

theorem air_conditioner_price :
  (total_spent / (price_before - subsidy) = (total_spent / price_before) * purchase_ratio) ∧
  (price_before > 0) ∧ 
  (price_before > subsidy) := by sorry

end NUMINAMATH_CALUDE_air_conditioner_price_l3466_346659


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3466_346671

theorem rectangle_diagonal (x y a b : ℝ) (h1 : π * x^2 * y = a) (h2 : π * y^2 * x = b) :
  (x^2 + y^2).sqrt = ((a^2 + b^2) / (a * b)).sqrt * ((a * b) / π^2)^(1/6) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3466_346671


namespace NUMINAMATH_CALUDE_problem_solution_l3466_346628

theorem problem_solution (a b : ℝ) : (a + b)^2 + Real.sqrt (2 * b - 4) = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3466_346628


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_x_eq_one_l3466_346601

theorem purely_imaginary_iff_x_eq_one (x : ℝ) : 
  let z : ℂ := (x^2 - 1) + (x + 1)*I
  (∃ y : ℝ, z = y*I) ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_x_eq_one_l3466_346601


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3466_346682

/-- Given two pipes A and B that can fill a tank, this theorem proves the time
    it takes for pipe B to fill the tank alone, given the times for pipe A
    and both pipes together. -/
theorem pipe_filling_time (time_A time_both : ℝ) (h1 : time_A = 10)
    (h2 : time_both = 20 / 3) : 
    (1 / time_A + 1 / 20 = 1 / time_both) := by
  sorry

#check pipe_filling_time

end NUMINAMATH_CALUDE_pipe_filling_time_l3466_346682


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l3466_346686

theorem inequality_implies_upper_bound (m : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) →
  m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l3466_346686


namespace NUMINAMATH_CALUDE_black_go_stones_l3466_346665

theorem black_go_stones (total : ℕ) (difference : ℕ) (black : ℕ) (white : ℕ) : 
  total = 1256 → 
  difference = 408 → 
  total = black + white → 
  white = black + difference → 
  black = 424 := by
sorry

end NUMINAMATH_CALUDE_black_go_stones_l3466_346665


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l3466_346605

/-- Given three lines that intersect at a single point, prove the value of k -/
theorem intersection_of_three_lines (k : ℚ) : 
  (∃! p : ℚ × ℚ, 
    (p.2 = 4 * p.1 + 2) ∧ 
    (p.2 = -2 * p.1 - 8) ∧ 
    (p.2 = 2 * p.1 + k)) → 
  k = -4/3 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l3466_346605


namespace NUMINAMATH_CALUDE_smallest_odd_abundant_number_l3466_346653

def is_abundant (n : ℕ) : Prop :=
  n < (Finset.sum (Finset.filter (λ x => x < n ∧ n % x = 0) (Finset.range n)) id)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem smallest_odd_abundant_number :
  (∀ n : ℕ, n < 945 → ¬(is_odd n ∧ is_abundant n ∧ is_composite n)) ∧
  (is_odd 945 ∧ is_abundant 945 ∧ is_composite 945) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_abundant_number_l3466_346653


namespace NUMINAMATH_CALUDE_semicircle_area_ratio_l3466_346649

theorem semicircle_area_ratio (R : ℝ) (h : R > 0) :
  let r := (3 : ℝ) / 5 * R
  (π * r^2 / 2) / (π * R^2 / 2) = 9 / 25 := by sorry

end NUMINAMATH_CALUDE_semicircle_area_ratio_l3466_346649


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3466_346646

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_property
  (a : ℕ → ℝ) (q : ℝ)
  (h_geo : geometric_sequence a q)
  (h_neg : a 1 * a 2 < 0) :
  a 1 * a 5 > 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3466_346646


namespace NUMINAMATH_CALUDE_second_encounter_correct_l3466_346695

/-- Represents the highway with speed limit signs and monitoring devices -/
structure Highway where
  speed_limit_start : ℕ := 3
  speed_limit_interval : ℕ := 4
  monitoring_start : ℕ := 10
  monitoring_interval : ℕ := 9
  first_encounter : ℕ := 19

/-- The kilometer mark of the second simultaneous encounter -/
def second_encounter (h : Highway) : ℕ := 55

/-- Theorem stating that the second encounter occurs at 55 km -/
theorem second_encounter_correct (h : Highway) : 
  second_encounter h = 55 := by sorry

end NUMINAMATH_CALUDE_second_encounter_correct_l3466_346695


namespace NUMINAMATH_CALUDE_angle_conversions_correct_l3466_346626

theorem angle_conversions_correct :
  let deg_to_rad (d : ℝ) := d * (π / 180)
  let rad_to_deg (r : ℝ) := r * (180 / π)
  (deg_to_rad 60 = π / 3) ∧
  (rad_to_deg (-10 * π / 3) = -600) ∧
  (deg_to_rad (-150) = -5 * π / 6) ∧
  (rad_to_deg (π / 12) = 15) := by
  sorry

end NUMINAMATH_CALUDE_angle_conversions_correct_l3466_346626


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3466_346621

theorem smallest_solution_abs_equation :
  let f : ℝ → ℝ := λ x => x * |x| - (2 * x^2 + 3 * x + 1)
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y ∧ x = (3 + Real.sqrt 13) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3466_346621


namespace NUMINAMATH_CALUDE_parallelogram_may_not_have_symmetry_l3466_346617

-- Define the basic geometric shapes
inductive GeometricShape
  | LineSegment
  | Rectangle
  | Angle
  | Parallelogram

-- Define a property for having an axis of symmetry
def has_axis_of_symmetry (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.LineSegment => True
  | GeometricShape.Rectangle => True
  | GeometricShape.Angle => True
  | GeometricShape.Parallelogram => sorry  -- This can be True or False

-- Theorem: Only parallelograms may not have an axis of symmetry
theorem parallelogram_may_not_have_symmetry :
  ∀ (shape : GeometricShape),
    ¬(has_axis_of_symmetry shape) → shape = GeometricShape.Parallelogram :=
by sorry

-- Note: The actual proof is omitted and replaced with 'sorry'

end NUMINAMATH_CALUDE_parallelogram_may_not_have_symmetry_l3466_346617


namespace NUMINAMATH_CALUDE_valid_numbers_l3466_346629

def is_valid_number (N : ℕ) : Prop :=
  ∃ (a b : ℕ) (q : ℚ),
    N = 10 * a + b ∧
    0 < a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    b = a * q ∧
    N = 3 * (a * q^2)

theorem valid_numbers :
  {N : ℕ | is_valid_number N} = {12, 24, 36, 48} :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l3466_346629


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composite_odd_l3466_346600

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = n

/-- A function that checks if a number is odd -/
def isOdd (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2 * k + 1

/-- A function that checks if a number can be expressed as the sum of two composite odd positive integers -/
def isSumOfTwoCompositeOdd (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ isOdd a ∧ isOdd b ∧ isComposite a ∧ isComposite b ∧ a + b = n

/-- The main theorem stating that 38 is the largest even positive integer that cannot be expressed as the sum of two composite odd positive integers -/
theorem largest_non_sum_of_composite_odd :
  (∀ (n : ℕ), n > 38 → n % 2 = 0 → isSumOfTwoCompositeOdd n) ∧
  ¬(isSumOfTwoCompositeOdd 38) :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composite_odd_l3466_346600


namespace NUMINAMATH_CALUDE_lucy_total_cost_l3466_346698

/-- The total cost Lucy paid for a lamp and a table, given specific pricing conditions. -/
theorem lucy_total_cost : 
  ∀ (lamp_original_price lamp_discounted_price table_price : ℝ),
  lamp_discounted_price = 20 →
  lamp_discounted_price = (1/5) * (0.6 * lamp_original_price) →
  table_price = 2 * lamp_original_price →
  lamp_discounted_price + table_price = 353.34 := by
  sorry

#check lucy_total_cost

end NUMINAMATH_CALUDE_lucy_total_cost_l3466_346698


namespace NUMINAMATH_CALUDE_club_equation_solution_l3466_346603

def club (A B : ℝ) : ℝ := 3 * A + 2 * B + 5

theorem club_equation_solution :
  ∃! A : ℝ, club A 4 = 58 ∧ A = 15 := by sorry

end NUMINAMATH_CALUDE_club_equation_solution_l3466_346603


namespace NUMINAMATH_CALUDE_symmetric_points_product_l3466_346643

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The problem statement -/
theorem symmetric_points_product (a b : ℝ) 
    (h : symmetric_wrt_origin (a + 2) 2 4 (-b)) : a * b = -12 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_product_l3466_346643


namespace NUMINAMATH_CALUDE_sum_odd_implies_difference_odd_l3466_346693

theorem sum_odd_implies_difference_odd (a b : ℤ) : 
  Odd (a + b) → Odd (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_implies_difference_odd_l3466_346693


namespace NUMINAMATH_CALUDE_hexagon_longest_side_range_l3466_346697

/-- Given a hexagon formed by wrapping a line segment of length 20,
    the length of its longest side is between 10/3 and 10 (exclusive). -/
theorem hexagon_longest_side_range :
  ∀ x : ℝ,
    (∃ a b c d e f : ℝ,
      a + b + c + d + e + f = 20 ∧
      x = max a (max b (max c (max d (max e f)))) ∧
      a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0) →
    (10 / 3 : ℝ) ≤ x ∧ x < 10 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_longest_side_range_l3466_346697


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3466_346627

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = -9 ∧ x₂ = -3 ∧ 
  (∀ x : ℝ, x^2 + 12*x + 27 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3466_346627


namespace NUMINAMATH_CALUDE_least_number_divisible_by_first_five_primes_l3466_346624

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def is_divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  ∀ m ∈ list, n % m = 0

theorem least_number_divisible_by_first_five_primes :
  ∃ n : Nat, n > 0 ∧ is_divisible_by_all n first_five_primes ∧
  ∀ k : Nat, k > 0 ∧ is_divisible_by_all k first_five_primes → n ≤ k :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_first_five_primes_l3466_346624


namespace NUMINAMATH_CALUDE_binomial_sum_theorem_l3466_346696

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the left-hand side of the first equation
def lhs1 (n : ℕ) (x : ℝ) : ℝ := sorry

-- Define the right-hand side of the first equation
def rhs1 (n : ℕ) (x : ℝ) : ℝ := sorry

-- Define the left-hand side of the second equation
def lhs2 (n : ℕ) : ℕ := sorry

-- Define the right-hand side of the second equation
def rhs2 (n : ℕ) : ℕ := sorry

-- State the theorem
theorem binomial_sum_theorem (n : ℕ) (hn : n ≥ 1) :
  (∀ x : ℝ, lhs1 n x = rhs1 n x) ∧ (lhs2 n = rhs2 n) := by sorry

end NUMINAMATH_CALUDE_binomial_sum_theorem_l3466_346696


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3466_346690

theorem sum_of_reciprocals (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1/x + 1/y = 4) (h4 : 1/x - 1/y = -8) : 
  x + y = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3466_346690


namespace NUMINAMATH_CALUDE_marias_trip_distance_l3466_346615

/-- The total distance of Maria's trip -/
def total_distance : ℝ := 480

/-- Theorem stating that the total distance of Maria's trip is 480 miles -/
theorem marias_trip_distance :
  ∃ (D : ℝ),
    D / 2 + (D / 2) / 4 + 180 = D ∧
    D = total_distance :=
by sorry

end NUMINAMATH_CALUDE_marias_trip_distance_l3466_346615


namespace NUMINAMATH_CALUDE_function_identity_l3466_346616

def f_condition (f : ℝ → ℝ) : Prop :=
  f 0 = 1 ∧ ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2

theorem function_identity (f : ℝ → ℝ) (h : f_condition f) :
  ∀ x : ℝ, f x = x + 1 :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l3466_346616


namespace NUMINAMATH_CALUDE_remainder_8437_div_9_l3466_346612

theorem remainder_8437_div_9 : 8437 % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8437_div_9_l3466_346612


namespace NUMINAMATH_CALUDE_fern_bushes_needed_l3466_346681

/-- The number of bushes needed to produce a given amount of perfume -/
def bushes_needed (petals_per_ounce : ℕ) (petals_per_rose : ℕ) (roses_per_bush : ℕ) 
                  (ounces_per_bottle : ℕ) (num_bottles : ℕ) : ℕ :=
  (petals_per_ounce * ounces_per_bottle * num_bottles) / (petals_per_rose * roses_per_bush)

/-- Theorem stating the number of bushes Fern needs to harvest -/
theorem fern_bushes_needed : 
  bushes_needed 320 8 12 12 20 = 800 := by
  sorry

end NUMINAMATH_CALUDE_fern_bushes_needed_l3466_346681


namespace NUMINAMATH_CALUDE_no_solution_for_inequality_l3466_346666

theorem no_solution_for_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 - x + 2 < 0 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_inequality_l3466_346666


namespace NUMINAMATH_CALUDE_area_difference_square_rectangle_l3466_346604

theorem area_difference_square_rectangle :
  ∀ (square_side : ℝ) (rect_length rect_width : ℝ),
  square_side * 4 = 52 →
  rect_length = 15 →
  rect_length * 2 + rect_width * 2 = 52 →
  square_side * square_side - rect_length * rect_width = 4 := by
sorry

end NUMINAMATH_CALUDE_area_difference_square_rectangle_l3466_346604


namespace NUMINAMATH_CALUDE_debora_has_twelve_more_dresses_l3466_346632

/-- The number of dresses each person has -/
structure Dresses where
  emily : ℕ
  melissa : ℕ
  debora : ℕ

/-- The conditions of the problem -/
def problem_conditions (d : Dresses) : Prop :=
  d.emily = 16 ∧
  d.melissa = d.emily / 2 ∧
  d.debora > d.melissa ∧
  d.emily + d.melissa + d.debora = 44

/-- The theorem to prove -/
theorem debora_has_twelve_more_dresses (d : Dresses) 
  (h : problem_conditions d) : d.debora = d.melissa + 12 := by
  sorry

#check debora_has_twelve_more_dresses

end NUMINAMATH_CALUDE_debora_has_twelve_more_dresses_l3466_346632


namespace NUMINAMATH_CALUDE_statement_I_statement_II_statement_III_l3466_346625

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Statement I
theorem statement_I : ∀ x : ℝ, floor (x + 1) = floor x + 1 := by sorry

-- Statement II (negation)
theorem statement_II : ∃ x y : ℝ, ∃ k : ℤ, floor (x + y + k) ≠ floor x + floor y + k := by sorry

-- Statement III (negation)
theorem statement_III : ∃ x y : ℝ, floor (x * y) ≠ floor x * floor y := by sorry

end NUMINAMATH_CALUDE_statement_I_statement_II_statement_III_l3466_346625


namespace NUMINAMATH_CALUDE_volume_of_extended_parallelepiped_with_caps_l3466_346642

/-- The volume of a set of points that are inside or within one unit of a rectangular parallelepiped
    with semi-spherical caps on the longest side vertices. -/
theorem volume_of_extended_parallelepiped_with_caps : ℝ := by
  -- Define the dimensions of the parallelepiped
  let length : ℝ := 6
  let width : ℝ := 3
  let height : ℝ := 2

  -- Define the radius of the semi-spherical caps
  let cap_radius : ℝ := 1

  -- Define the number of semi-spherical caps
  let num_caps : ℕ := 4

  -- Calculate the volume
  have volume : ℝ := (324 + 8 * Real.pi) / 3

  sorry

#check volume_of_extended_parallelepiped_with_caps

end NUMINAMATH_CALUDE_volume_of_extended_parallelepiped_with_caps_l3466_346642


namespace NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l3466_346638

/-- Given real t, the point (x, y) satisfies both equations -/
def satisfies_equations (x y t : ℝ) : Prop :=
  2 * t * x - 3 * y - 4 * t = 0 ∧ 2 * x - 3 * t * y + 4 = 0

/-- The locus of points (x, y) satisfying the equations for all t forms a hyperbola -/
theorem intersection_points_form_hyperbola :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, (∃ t : ℝ, satisfies_equations x y t) →
  x^2 / a^2 - y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l3466_346638


namespace NUMINAMATH_CALUDE_evas_shoes_l3466_346641

def total_laces : ℕ := 52
def laces_per_pair : ℕ := 2

theorem evas_shoes : 
  total_laces / laces_per_pair = 26 := by sorry

end NUMINAMATH_CALUDE_evas_shoes_l3466_346641


namespace NUMINAMATH_CALUDE_largest_c_value_l3466_346635

theorem largest_c_value (c : ℝ) : 
  (∀ x : ℝ, -2*x^2 + 8*x - 6 ≥ 0 → x ≤ c) ↔ c = 3 := by sorry

end NUMINAMATH_CALUDE_largest_c_value_l3466_346635


namespace NUMINAMATH_CALUDE_l1_fixed_point_min_distance_intersection_l3466_346650

-- Define the lines and circle
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x - (m + 1) * y - 2 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y + 1 = 0
def l3 (x y : ℝ) : Prop := y = x - 2

def circle_C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 12}

-- Theorem 1: l1 always passes through (-2, -2)
theorem l1_fixed_point (m : ℝ) : l1 m (-2) (-2) := by sorry

-- Theorem 2: Minimum distance between intersection points
theorem min_distance_intersection :
  let center := (1, -1)  -- Intersection of l2 and l3
  ∃ (m : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ circle_C center → B ∈ circle_C center →
    l1 m A.1 A.2 → l1 m B.1 B.2 →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ 8 := by sorry

end NUMINAMATH_CALUDE_l1_fixed_point_min_distance_intersection_l3466_346650


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3466_346658

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, a^2+1, 2*a-1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-3} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3466_346658


namespace NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_probability_l3466_346678

/-- The probability that Xavier and Yvonne solve a problem but Zelda does not, 
    given their individual probabilities of success. -/
theorem xavier_yvonne_not_zelda_probability 
  (p_xavier : ℚ) (p_yvonne : ℚ) (p_zelda : ℚ)
  (h_xavier : p_xavier = 1/6)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8) :
  p_xavier * p_yvonne * (1 - p_zelda) = 1/32 :=
sorry

end NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_probability_l3466_346678


namespace NUMINAMATH_CALUDE_bills_final_money_bills_final_money_is_3180_l3466_346618

/-- Calculates Bill's final amount of money after Frank and Bill's pizza purchase --/
theorem bills_final_money (initial_money : ℝ) (pizza_cost : ℝ) (num_pizzas : ℕ) 
  (topping1_cost : ℝ) (topping2_cost : ℝ) (discount_rate : ℝ) (bills_initial_money : ℝ) : ℝ :=
  let total_pizza_cost := pizza_cost * num_pizzas
  let total_topping_cost := (topping1_cost + topping2_cost) * num_pizzas
  let total_cost_before_discount := total_pizza_cost + total_topping_cost
  let discount := discount_rate * total_pizza_cost
  let final_cost := total_cost_before_discount - discount
  let remaining_money := initial_money - final_cost
  bills_initial_money + remaining_money

/-- Proves that Bill's final amount of money is $31.80 --/
theorem bills_final_money_is_3180 : 
  bills_final_money 42 11 3 1.5 2 0.1 30 = 31.80 := by
  sorry

end NUMINAMATH_CALUDE_bills_final_money_bills_final_money_is_3180_l3466_346618


namespace NUMINAMATH_CALUDE_fraction_inequality_l3466_346664

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : d > c) (h4 : c > 0) : 
  a / c > b / d := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3466_346664


namespace NUMINAMATH_CALUDE_max_term_is_9_8_l3466_346602

/-- The sequence defined by n^2 / 2^n for n ≥ 1 -/
def a (n : ℕ) : ℚ := (n^2 : ℚ) / 2^n

/-- The maximum term of the sequence occurs at n = 3 -/
def max_term_index : ℕ := 3

/-- The maximum value of the sequence -/
def max_term_value : ℚ := 9/8

/-- Theorem stating that the maximum term of the sequence a(n) is 9/8 -/
theorem max_term_is_9_8 :
  (∀ n : ℕ, n ≥ 1 → a n ≤ max_term_value) ∧ 
  (∃ n : ℕ, n ≥ 1 ∧ a n = max_term_value) :=
sorry

end NUMINAMATH_CALUDE_max_term_is_9_8_l3466_346602


namespace NUMINAMATH_CALUDE_second_discount_percentage_l3466_346652

theorem second_discount_percentage
  (normal_price : ℝ)
  (first_discount_rate : ℝ)
  (final_price : ℝ)
  (h1 : normal_price = 174.99999999999997)
  (h2 : first_discount_rate = 0.1)
  (h3 : final_price = 126) :
  let price_after_first_discount := normal_price * (1 - first_discount_rate)
  let second_discount_rate := (price_after_first_discount - final_price) / price_after_first_discount
  second_discount_rate = 0.2 := by
sorry

#eval (174.99999999999997 * 0.9 - 126) / (174.99999999999997 * 0.9)

end NUMINAMATH_CALUDE_second_discount_percentage_l3466_346652


namespace NUMINAMATH_CALUDE_triangle_area_from_sides_and_median_l3466_346672

/-- Given a triangle PQR with side lengths and median, calculate its area -/
theorem triangle_area_from_sides_and_median 
  (PQ PR PM : ℝ) 
  (h_PQ : PQ = 8) 
  (h_PR : PR = 18) 
  (h_PM : PM = 12) : 
  ∃ (area : ℝ), area = Real.sqrt 2975 ∧ area > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_sides_and_median_l3466_346672


namespace NUMINAMATH_CALUDE_election_win_margin_l3466_346655

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) : 
  winner_votes = (62 * total_votes) / 100 →
  winner_votes = 1054 →
  winner_votes - ((38 * total_votes) / 100) = 408 :=
by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l3466_346655


namespace NUMINAMATH_CALUDE_initial_pc_cost_l3466_346648

/-- Proves that the initial cost of a gaming PC is $1200, given the conditions of the video card upgrade and total spent. -/
theorem initial_pc_cost (old_card_sale : ℕ) (new_card_cost : ℕ) (total_spent : ℕ) 
  (h1 : old_card_sale = 300)
  (h2 : new_card_cost = 500)
  (h3 : total_spent = 1400) :
  total_spent - (new_card_cost - old_card_sale) = 1200 := by
  sorry

#check initial_pc_cost

end NUMINAMATH_CALUDE_initial_pc_cost_l3466_346648


namespace NUMINAMATH_CALUDE_max_product_of_externally_tangent_circles_l3466_346611

/-- Circle C₁ with center (a, -2) and radius 2 -/
def C₁ (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y + 2)^2 = 4

/-- Circle C₂ with center (-b, -2) and radius 1 -/
def C₂ (b : ℝ) (x y : ℝ) : Prop := (x + b)^2 + (y + 2)^2 = 1

/-- Circles C₁ and C₂ are externally tangent -/
def externally_tangent (a b : ℝ) : Prop := (a + b)^2 = 3^2

theorem max_product_of_externally_tangent_circles (a b : ℝ) 
  (h : externally_tangent a b) : 
  a * b ≤ 9/4 := by sorry

end NUMINAMATH_CALUDE_max_product_of_externally_tangent_circles_l3466_346611


namespace NUMINAMATH_CALUDE_k_range_l3466_346647

theorem k_range (x y k : ℝ) : 
  3 * x + y = k + 1 →
  x + 3 * y = 3 →
  0 < x + y →
  x + y < 1 →
  -4 < k ∧ k < 0 := by
sorry

end NUMINAMATH_CALUDE_k_range_l3466_346647


namespace NUMINAMATH_CALUDE_h_degree_three_iff_c_eq_three_l3466_346619

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 3 - 8*x + 2*x^2 - 7*x^3 + 6*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 2 - 3*x + x^3 - 2*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * (g x)

/-- Theorem stating that h(x) has degree 3 if and only if c = 3 -/
theorem h_degree_three_iff_c_eq_three :
  ∃! c : ℝ, (∀ x : ℝ, h c x = 3 - 8*x + 2*x^2 - 4*x^3) ∧ c = 3 :=
sorry

end NUMINAMATH_CALUDE_h_degree_three_iff_c_eq_three_l3466_346619


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3466_346675

theorem ellipse_foci_distance (x y : ℝ) :
  (x^2 / 45 + y^2 / 5 = 9) → (∃ f : ℝ, f = 12 * Real.sqrt 10 ∧ f = distance_between_foci) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3466_346675


namespace NUMINAMATH_CALUDE_greatest_unachievable_scores_l3466_346683

def score_system : List ℕ := [19, 9, 8]

def is_achievable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 19 * a + 9 * b + 8 * c

theorem greatest_unachievable_scores :
  (¬ is_achievable 31) ∧
  (¬ is_achievable 39) ∧
  (∀ m, m > 39 → is_achievable m) ∧
  (31 * 39 = 1209) := by
  sorry

#check greatest_unachievable_scores

end NUMINAMATH_CALUDE_greatest_unachievable_scores_l3466_346683


namespace NUMINAMATH_CALUDE_circle_symmetry_about_origin_l3466_346657

/-- Given a circle with equation (x-1)^2+(y+2)^2=5, 
    prove that (x+1)^2+(y-2)^2=5 is its symmetric about the origin -/
theorem circle_symmetry_about_origin :
  let original_circle := (fun (x y : ℝ) => (x - 1)^2 + (y + 2)^2 = 5)
  let symmetric_circle := (fun (x y : ℝ) => (x + 1)^2 + (y - 2)^2 = 5)
  ∀ (x y : ℝ), original_circle (-x) (-y) ↔ symmetric_circle x y :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_about_origin_l3466_346657


namespace NUMINAMATH_CALUDE_symmetry_problem_l3466_346677

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Given point P -/
def P : Point3D := { x := -2, y := 1, z := 4 }

/-- Given point A -/
def A : Point3D := { x := 1, y := 0, z := 2 }

/-- Reflect a point about the xOy plane -/
def reflectXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Find the point symmetric to a given point about another point -/
def symmetricPoint (p q : Point3D) : Point3D :=
  { x := 2 * p.x - q.x,
    y := 2 * p.y - q.y,
    z := 2 * p.z - q.z }

theorem symmetry_problem :
  reflectXOY P = { x := -2, y := 1, z := -4 } ∧
  symmetricPoint P A = { x := -5, y := 2, z := 6 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_problem_l3466_346677


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l3466_346630

theorem factorial_equation_solutions :
  ∀ x y z : ℕ+, 2^x.val + 3^y.val - 7 = Nat.factorial z.val →
    ((x = 2 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 4)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l3466_346630


namespace NUMINAMATH_CALUDE_sum_of_squares_representation_l3466_346631

theorem sum_of_squares_representation (n : ℕ) :
  ∃ (m : ℤ), (∃ (representations : Finset (ℤ × ℤ)), 
    (∀ (pair : ℤ × ℤ), pair ∈ representations → m = pair.1^2 + pair.2^2) ∧
    representations.card ≥ n) ∧ 
  m = 5^(2*n) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_representation_l3466_346631


namespace NUMINAMATH_CALUDE_determinant_evaluation_l3466_346687

theorem determinant_evaluation (x z : ℝ) : 
  Matrix.det !![1, x, z; 1, x + z, z; 1, x, x + z] = x * z + 2 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_evaluation_l3466_346687


namespace NUMINAMATH_CALUDE_problem_proof_l3466_346613

theorem problem_proof (x v : ℝ) (hx : x = 2) (hv : v = 3 * x) :
  (2 * v - 5) - (2 * x - 5) = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l3466_346613


namespace NUMINAMATH_CALUDE_jessica_seashells_l3466_346644

/-- Given that Joan found 6 seashells and the total number of seashells found by Joan and Jessica is 14, prove that Jessica found 8 seashells. -/
theorem jessica_seashells (joan_seashells : ℕ) (total_seashells : ℕ) (h1 : joan_seashells = 6) (h2 : total_seashells = 14) :
  total_seashells - joan_seashells = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_l3466_346644


namespace NUMINAMATH_CALUDE_calculation_proof_l3466_346699

theorem calculation_proof : (((15 - 2 + 4) / 1) / 2) * 8 = 68 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3466_346699


namespace NUMINAMATH_CALUDE_range_of_a_l3466_346610

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 2| - |x - 1| ≥ a^3 - 4*a^2 - 3) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3466_346610
