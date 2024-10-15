import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2247_224798

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2247_224798


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l2247_224712

/-- Calculates the desired gain percentage for a book sale --/
theorem book_sale_gain_percentage 
  (loss_price : ℝ) 
  (loss_percentage : ℝ) 
  (desired_price : ℝ) : 
  loss_price = 800 ∧ 
  loss_percentage = 20 ∧ 
  desired_price = 1100 → 
  (desired_price - loss_price / (1 - loss_percentage / 100)) / 
  (loss_price / (1 - loss_percentage / 100)) * 100 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l2247_224712


namespace NUMINAMATH_CALUDE_circle_radius_tangent_to_square_sides_l2247_224714

open Real

theorem circle_radius_tangent_to_square_sides (a : ℝ) :
  a = Real.sqrt (2 + Real.sqrt 2) →
  ∃ (R : ℝ),
    R = Real.sqrt 2 + Real.sqrt (2 - Real.sqrt 2) ∧
    (Real.sin (π / 8) = Real.sqrt (2 - Real.sqrt 2) / 2) ∧
    (∃ (O : ℝ × ℝ) (C : ℝ × ℝ),
      -- O is the center of the circle, C is a vertex of the square
      -- The distance between O and C is related to R and the sine of 22.5°
      Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2) = 4 * R / Real.sqrt (2 - Real.sqrt 2) ∧
      -- The angle between the tangents from C is 45°
      Real.arctan (R / (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2) - R)) = π / 8) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_tangent_to_square_sides_l2247_224714


namespace NUMINAMATH_CALUDE_nigels_money_theorem_l2247_224745

/-- Represents the amount of money Nigel has at different stages --/
structure NigelsMoney where
  initial : ℕ
  afterFirstGiveaway : ℕ
  afterMotherGift : ℕ
  final : ℕ

/-- Theorem stating Nigel's final amount is $10 more than twice his initial amount --/
theorem nigels_money_theorem (n : NigelsMoney) (h1 : n.initial = 45)
  (h2 : n.afterMotherGift = n.afterFirstGiveaway + 80)
  (h3 : n.final = n.afterMotherGift - 25)
  (h4 : n.afterFirstGiveaway < n.initial) :
  n.final = 2 * n.initial + 10 := by
  sorry

end NUMINAMATH_CALUDE_nigels_money_theorem_l2247_224745


namespace NUMINAMATH_CALUDE_expected_rainfall_theorem_l2247_224718

/-- Represents the daily rainfall probabilities and amounts -/
structure DailyRainfall where
  sun_prob : ℝ
  light_rain_prob : ℝ
  heavy_rain_prob : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculates the expected total rainfall over a given number of days -/
def expected_total_rainfall (daily : DailyRainfall) (days : ℕ) : ℝ :=
  days * (daily.light_rain_prob * daily.light_rain_amount + daily.heavy_rain_prob * daily.heavy_rain_amount)

/-- The main theorem stating the expected total rainfall over 10 days -/
theorem expected_rainfall_theorem (daily : DailyRainfall)
  (h1 : daily.sun_prob = 0.5)
  (h2 : daily.light_rain_prob = 0.3)
  (h3 : daily.heavy_rain_prob = 0.2)
  (h4 : daily.light_rain_amount = 3)
  (h5 : daily.heavy_rain_amount = 6)
  : expected_total_rainfall daily 10 = 21 := by
  sorry

end NUMINAMATH_CALUDE_expected_rainfall_theorem_l2247_224718


namespace NUMINAMATH_CALUDE_complex_simplification_l2247_224733

theorem complex_simplification :
  (-5 + 3 * Complex.I) - (2 - 7 * Complex.I) + (1 + 2 * Complex.I) = -6 + 12 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2247_224733


namespace NUMINAMATH_CALUDE_wedge_volume_l2247_224763

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (θ : ℝ) : 
  d = 12 →  -- diameter of the log
  θ = π/4 →  -- angle between the two cuts (45° in radians)
  (1/2) * π * (d/2)^2 * d = 216 * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l2247_224763


namespace NUMINAMATH_CALUDE_triangle_properties_l2247_224730

open Real

theorem triangle_properties (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C ∧
  1 + (tan C / tan B) = 2 * a / b ∧
  cos (B + π/6) = 1/3 ∧
  (a + b)^2 - c^2 = 4 →
  C = π/3 ∧ 
  sin A = (2 * sqrt 6 + 1) / 6 ∧
  ∀ x y, x > 0 ∧ y > 0 ∧ (x + y)^2 - c^2 = 4 → 3*x + y ≥ 4 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2247_224730


namespace NUMINAMATH_CALUDE_cubic_factorization_l2247_224785

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2247_224785


namespace NUMINAMATH_CALUDE_triangular_display_total_l2247_224722

/-- Represents a triangular display of cans -/
structure CanDisplay where
  bottom_layer : ℕ
  second_layer : ℕ
  top_layer : ℕ

/-- Calculates the total number of cans in the display -/
def total_cans (d : CanDisplay) : ℕ :=
  sorry

/-- Theorem stating that the specific triangular display contains 165 cans -/
theorem triangular_display_total (d : CanDisplay) 
  (h1 : d.bottom_layer = 30)
  (h2 : d.second_layer = 27)
  (h3 : d.top_layer = 3) :
  total_cans d = 165 := by
  sorry

end NUMINAMATH_CALUDE_triangular_display_total_l2247_224722


namespace NUMINAMATH_CALUDE_a_minus_b_value_l2247_224780

/-- Given an equation y = a + b/x, where a and b are constants, 
    prove that a - b = 19/2 when y = 2 for x = 2 and y = 7 for x = -2 -/
theorem a_minus_b_value (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 2 ↔ x = 2) ∧ (a + b / x = 7 ↔ x = -2)) → 
  a - b = 19 / 2 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l2247_224780


namespace NUMINAMATH_CALUDE_a_alone_time_equals_b_alone_time_l2247_224795

/-- Two workers finishing a job -/
structure WorkerPair where
  total_time : ℝ
  b_alone_time : ℝ
  work : ℝ

/-- The time it takes for worker a to finish the job alone -/
def a_alone_time (w : WorkerPair) : ℝ :=
  w.b_alone_time

theorem a_alone_time_equals_b_alone_time (w : WorkerPair)
  (h1 : w.total_time = 10)
  (h2 : w.b_alone_time = 20) :
  a_alone_time w = w.b_alone_time :=
by
  sorry

#check a_alone_time_equals_b_alone_time

end NUMINAMATH_CALUDE_a_alone_time_equals_b_alone_time_l2247_224795


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_negation_l2247_224723

theorem quadratic_inequality_and_negation :
  (∀ x : ℝ, x^2 + 2*x + 3 > 0) ∧
  (¬(∀ x : ℝ, x^2 + 2*x + 3 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_negation_l2247_224723


namespace NUMINAMATH_CALUDE_product_65_35_l2247_224746

theorem product_65_35 : 65 * 35 = 2275 := by
  sorry

end NUMINAMATH_CALUDE_product_65_35_l2247_224746


namespace NUMINAMATH_CALUDE_combined_weight_theorem_l2247_224715

/-- The combined weight that Rodney, Roger, and Ron can lift -/
def combinedWeight (rodney roger ron : ℕ) : ℕ := rodney + roger + ron

/-- Theorem stating the combined weight that Rodney, Roger, and Ron can lift -/
theorem combined_weight_theorem :
  ∀ (ron : ℕ),
  let roger := 4 * ron - 7
  let rodney := 2 * roger
  rodney = 146 →
  combinedWeight rodney roger ron = 239 := by
sorry

end NUMINAMATH_CALUDE_combined_weight_theorem_l2247_224715


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l2247_224741

/-- Converts a base 5 number (represented as a list of digits) to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The loot values in base 5 --/
def silverware : List Nat := [3, 1, 2, 4]
def diamondTiaras : List Nat := [1, 0, 1, 3]
def silkScarves : List Nat := [2, 0, 2]

/-- The theorem to prove --/
theorem pirate_loot_sum :
  base5ToBase10 silverware + base5ToBase10 diamondTiaras + base5ToBase10 silkScarves = 1011 := by
  sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l2247_224741


namespace NUMINAMATH_CALUDE_cheaper_lens_price_l2247_224782

theorem cheaper_lens_price (original_price : ℝ) (discount_rate : ℝ) (savings : ℝ) : 
  original_price = 300 →
  discount_rate = 0.2 →
  savings = 20 →
  original_price * (1 - discount_rate) - savings = 220 := by
sorry

end NUMINAMATH_CALUDE_cheaper_lens_price_l2247_224782


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l2247_224757

theorem parallel_lines_b_value (b : ℝ) : 
  (∃ (m₁ m₂ : ℝ), (∀ x y : ℝ, 3 * y - 4 * b = 9 * x ↔ y = m₁ * x + (4 * b / 3)) ∧
                   (∀ x y : ℝ, y - 2 = (b + 10) * x ↔ y = m₂ * x + 2) ∧
                   m₁ = m₂) →
  b = -7 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l2247_224757


namespace NUMINAMATH_CALUDE_equal_sum_product_square_diff_l2247_224783

theorem equal_sum_product_square_diff : ∃ (x y : ℝ),
  (x + y = x * y) ∧ (x + y = x^2 - y^2) ∧
  ((x = (3 + Real.sqrt 5) / 2 ∧ y = (1 + Real.sqrt 5) / 2) ∨
   (x = (3 - Real.sqrt 5) / 2 ∧ y = (1 - Real.sqrt 5) / 2) ∨
   (x = 0 ∧ y = 0)) :=
by sorry


end NUMINAMATH_CALUDE_equal_sum_product_square_diff_l2247_224783


namespace NUMINAMATH_CALUDE_gold_alloy_percentage_l2247_224759

/-- Proves that adding pure gold to an alloy results in a specific gold percentage -/
theorem gold_alloy_percentage 
  (original_weight : ℝ) 
  (original_percentage : ℝ) 
  (added_gold : ℝ) 
  (h1 : original_weight = 48) 
  (h2 : original_percentage = 0.25) 
  (h3 : added_gold = 12) : 
  (original_percentage * original_weight + added_gold) / (original_weight + added_gold) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_gold_alloy_percentage_l2247_224759


namespace NUMINAMATH_CALUDE_nonreal_cube_root_sum_l2247_224752

/-- Given ω is a nonreal complex cube root of unity, 
    prove that (1 - ω + ω^2)^4 + (1 + ω - ω^2)^4 = -16 -/
theorem nonreal_cube_root_sum (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (1 - ω + ω^2)^4 + (1 + ω - ω^2)^4 = -16 := by
  sorry

end NUMINAMATH_CALUDE_nonreal_cube_root_sum_l2247_224752


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2247_224778

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  (2 / (1 - i)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2247_224778


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achieved_l2247_224764

theorem quadratic_minimum (x : ℝ) : x^2 - 4*x - 2019 ≥ -2023 := by
  sorry

theorem quadratic_minimum_achieved : ∃ x : ℝ, x^2 - 4*x - 2019 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achieved_l2247_224764


namespace NUMINAMATH_CALUDE_sandwich_combinations_l2247_224734

theorem sandwich_combinations (salami_types : Nat) (cheese_types : Nat) (sauce_types : Nat) :
  salami_types = 8 →
  cheese_types = 7 →
  sauce_types = 3 →
  (salami_types * Nat.choose cheese_types 2 * sauce_types) = 504 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l2247_224734


namespace NUMINAMATH_CALUDE_boltons_class_size_l2247_224726

theorem boltons_class_size :
  ∀ (S : ℚ),
  (2 / 5 : ℚ) * S + (1 / 3 : ℚ) * ((3 / 5 : ℚ) * S) + ((3 / 5 : ℚ) * S - (1 / 3 : ℚ) * ((3 / 5 : ℚ) * S)) = S →
  (2 / 5 : ℚ) * S + ((3 / 5 : ℚ) * S - (1 / 3 : ℚ) * ((3 / 5 : ℚ) * S)) = 20 →
  S = 25 := by
  sorry

end NUMINAMATH_CALUDE_boltons_class_size_l2247_224726


namespace NUMINAMATH_CALUDE_lines_intersection_l2247_224790

/-- Two lines intersect at a unique point (-2/7, 5/7) -/
theorem lines_intersection :
  ∃! (p : ℝ × ℝ), 
    (∃ s : ℝ, p = (2 + 3*s, 3 + 4*s)) ∧ 
    (∃ v : ℝ, p = (-1 + v, 2 - v)) ∧
    p = (-2/7, 5/7) := by
  sorry


end NUMINAMATH_CALUDE_lines_intersection_l2247_224790


namespace NUMINAMATH_CALUDE_system_solution_l2247_224756

theorem system_solution : 
  ∃ (x y z : ℝ), 
    (x = 1/2 ∧ y = 0 ∧ z = 0) ∧
    (2*x + 3*y + z = 1) ∧
    (4*x - y + 2*z = 2) ∧
    (8*x + 5*y + 3*z = 4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2247_224756


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l2247_224738

/-- A permutation of integers from 1 to n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Check if a number is divisible by either 4 or 7 -/
def isDivisibleBy4Or7 (n : ℕ) : Prop := n % 4 = 0 ∨ n % 7 = 0

/-- Check if a permutation satisfies the adjacency condition when arranged in a circle -/
def isValidCircularArrangement (p : Permutation 2015) : Prop :=
  ∀ i : Fin 2015, isDivisibleBy4Or7 ((p i).val + (p (i + 1)).val)

theorem exists_valid_arrangement : ∃ p : Permutation 2015, isValidCircularArrangement p := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l2247_224738


namespace NUMINAMATH_CALUDE_total_paper_clips_l2247_224799

/-- The number of boxes used to distribute paper clips -/
def num_boxes : ℕ := 9

/-- The number of paper clips in each box -/
def clips_per_box : ℕ := 9

/-- Theorem: The total number of paper clips collected is 81 -/
theorem total_paper_clips : num_boxes * clips_per_box = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_paper_clips_l2247_224799


namespace NUMINAMATH_CALUDE_sum_extension_terms_l2247_224701

theorem sum_extension_terms (k : ℕ) (hk : k > 1) : 
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k :=
sorry

end NUMINAMATH_CALUDE_sum_extension_terms_l2247_224701


namespace NUMINAMATH_CALUDE_a_minus_b_value_l2247_224793

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 4) (h2 : b^2 = 9) (h3 : a/b > 0) :
  a - b = 1 ∨ a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l2247_224793


namespace NUMINAMATH_CALUDE_laptop_price_proof_l2247_224737

theorem laptop_price_proof (sticker_price : ℝ) : 
  (0.9 * sticker_price - 100 = 0.8 * sticker_price - 20) → 
  sticker_price = 800 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_proof_l2247_224737


namespace NUMINAMATH_CALUDE_starting_lineup_count_l2247_224765

def team_size : ℕ := 15
def lineup_size : ℕ := 7
def all_stars : ℕ := 3
def guards : ℕ := 5

theorem starting_lineup_count :
  (Finset.sum (Finset.range 3) (λ i =>
    Nat.choose guards (i + 2) * Nat.choose (team_size - all_stars - guards) (lineup_size - all_stars - (i + 2)))) = 285 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l2247_224765


namespace NUMINAMATH_CALUDE_not_both_odd_with_equal_product_l2247_224748

/-- Represents a mapping of letters to digits -/
def DigitMapping := Char → Fin 10

/-- Represents a number as a string of letters -/
def NumberWord := String

/-- Calculate the product of digits in a number word given a digit mapping -/
def digitProduct (mapping : DigitMapping) (word : NumberWord) : ℕ :=
  word.foldl (λ acc c => acc * (mapping c).val.succ) 1

/-- Check if a number word represents an odd number given a digit mapping -/
def isOdd (mapping : DigitMapping) (word : NumberWord) : Prop :=
  (mapping word.back).val % 2 = 1

theorem not_both_odd_with_equal_product (mapping : DigitMapping) 
    (word1 word2 : NumberWord) 
    (h_distinct : ∀ (c1 c2 : Char), c1 ≠ c2 → mapping c1 ≠ mapping c2)
    (h_equal_product : digitProduct mapping word1 = digitProduct mapping word2) :
    ¬(isOdd mapping word1 ∧ isOdd mapping word2) := by
  sorry

end NUMINAMATH_CALUDE_not_both_odd_with_equal_product_l2247_224748


namespace NUMINAMATH_CALUDE_calculation_proof_l2247_224724

theorem calculation_proof :
  ((-3/4 - 5/8 + 9/12) * (-24) = 15) ∧
  (-1^6 + |(-2)^3 - 10| - (-3) / (-1)^2023 = 14) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2247_224724


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l2247_224742

theorem jake_and_sister_weight (jake_weight : ℕ) (sister_weight : ℕ) : 
  jake_weight = 108 →
  jake_weight - 12 = 2 * sister_weight →
  jake_weight + sister_weight = 156 :=
by sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l2247_224742


namespace NUMINAMATH_CALUDE_impossible_heart_and_club_l2247_224775

-- Define a standard deck of cards
def StandardDeck : Type := Fin 52

-- Define suits
inductive Suit
| Hearts
| Clubs
| Diamonds
| Spades

-- Define a function to get the suit of a card
def getSuit : StandardDeck → Suit := sorry

-- Theorem: The probability of drawing a card that is both Hearts and Clubs is 0
theorem impossible_heart_and_club (card : StandardDeck) : 
  ¬(getSuit card = Suit.Hearts ∧ getSuit card = Suit.Clubs) := by
  sorry

end NUMINAMATH_CALUDE_impossible_heart_and_club_l2247_224775


namespace NUMINAMATH_CALUDE_negation_equivalence_l2247_224755

/-- An exponential function -/
def ExponentialFunction (f : ℝ → ℝ) : Prop := sorry

/-- A monotonic function -/
def MonotonicFunction (f : ℝ → ℝ) : Prop := sorry

/-- The statement "All exponential functions are monotonic functions" -/
def AllExponentialAreMonotonic : Prop :=
  ∀ f : ℝ → ℝ, ExponentialFunction f → MonotonicFunction f

/-- The negation of "All exponential functions are monotonic functions" -/
def NegationAllExponentialAreMonotonic : Prop :=
  ∃ f : ℝ → ℝ, ExponentialFunction f ∧ ¬MonotonicFunction f

/-- Theorem: The negation of "All exponential functions are monotonic functions"
    is equivalent to "There exists at least one exponential function that is not a monotonic function" -/
theorem negation_equivalence :
  ¬AllExponentialAreMonotonic ↔ NegationAllExponentialAreMonotonic :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2247_224755


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2247_224779

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2247_224779


namespace NUMINAMATH_CALUDE_ball_color_difference_l2247_224753

theorem ball_color_difference (m n : ℕ) (h1 : m > n) (h2 : n > 0) :
  (m * (m - 1) + n * (n - 1) : ℚ) / ((m + n) * (m + n - 1)) = 
  (2 * m * n : ℚ) / ((m + n) * (m + n - 1)) →
  ∃ a : ℕ, a > 1 ∧ m - n = a :=
sorry

end NUMINAMATH_CALUDE_ball_color_difference_l2247_224753


namespace NUMINAMATH_CALUDE_part_one_part_two_l2247_224702

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Part I
theorem part_one : ∃ (m n : ℝ), a = (m • b.1 + n • c.1, m • b.2 + n • c.2) := by sorry

-- Part II
theorem part_two : 
  ∃ (d : ℝ × ℝ), 
    (∃ (k : ℝ), (d.1 - c.1, d.2 - c.2) = k • (a.1 + b.1, a.2 + b.2)) ∧ 
    (d.1 - c.1)^2 + (d.2 - c.2)^2 = 5 ∧
    (d = (3, -1) ∨ d = (5, 3)) := by sorry


end NUMINAMATH_CALUDE_part_one_part_two_l2247_224702


namespace NUMINAMATH_CALUDE_divisibility_problem_l2247_224761

theorem divisibility_problem (n : ℕ) (h : n = (List.range 2001).foldl (· * ·) 1) :
  ∃ k : ℤ, n + (4003 * n - 4002) = 4003 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2247_224761


namespace NUMINAMATH_CALUDE_fruit_punch_theorem_l2247_224731

/-- Calculates the total amount of fruit punch given the amount of orange punch -/
def total_fruit_punch (orange_punch : ℝ) : ℝ :=
  let cherry_punch := 2 * orange_punch
  let apple_juice := cherry_punch - 1.5
  orange_punch + cherry_punch + apple_juice

/-- Theorem stating that given 4.5 liters of orange punch, the total fruit punch is 21 liters -/
theorem fruit_punch_theorem : total_fruit_punch 4.5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fruit_punch_theorem_l2247_224731


namespace NUMINAMATH_CALUDE_shower_water_reduction_l2247_224700

theorem shower_water_reduction 
  (original_time original_rate : ℝ) 
  (new_time : ℝ := 3/4 * original_time) 
  (new_rate : ℝ := 3/4 * original_rate) : 
  1 - (new_time * new_rate) / (original_time * original_rate) = 7/16 := by
sorry

end NUMINAMATH_CALUDE_shower_water_reduction_l2247_224700


namespace NUMINAMATH_CALUDE_combinations_count_l2247_224713

/-- Represents the cost of a pencil in cents -/
def pencil_cost : ℕ := 5

/-- Represents the cost of an eraser in cents -/
def eraser_cost : ℕ := 10

/-- Represents the cost of a notebook in cents -/
def notebook_cost : ℕ := 20

/-- Represents the total amount Mrs. Hilt has in cents -/
def total_amount : ℕ := 50

/-- Counts the number of valid combinations of items that can be purchased -/
def count_combinations : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ =>
    pencil_cost * t.1 + eraser_cost * t.2.1 + notebook_cost * t.2.2 = total_amount)
    (Finset.product (Finset.range (total_amount / pencil_cost + 1))
      (Finset.product (Finset.range (total_amount / eraser_cost + 1))
        (Finset.range (total_amount / notebook_cost + 1))))).card

theorem combinations_count :
  count_combinations = 12 := by sorry

end NUMINAMATH_CALUDE_combinations_count_l2247_224713


namespace NUMINAMATH_CALUDE_brenda_skittles_l2247_224747

theorem brenda_skittles (x : ℕ) : x + 8 = 15 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_brenda_skittles_l2247_224747


namespace NUMINAMATH_CALUDE_willow_catkin_diameter_scientific_notation_l2247_224707

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem willow_catkin_diameter_scientific_notation :
  toScientificNotation 0.0000105 = ScientificNotation.mk 1.05 (-5) (by sorry) :=
sorry

end NUMINAMATH_CALUDE_willow_catkin_diameter_scientific_notation_l2247_224707


namespace NUMINAMATH_CALUDE_luke_connor_sleep_difference_l2247_224735

theorem luke_connor_sleep_difference (connor_sleep : ℕ) (puppy_sleep : ℕ) : 
  connor_sleep = 6 →
  puppy_sleep = 16 →
  puppy_sleep = 2 * (connor_sleep + (puppy_sleep / 2 - connor_sleep)) →
  puppy_sleep / 2 - connor_sleep = 2 :=
by sorry

end NUMINAMATH_CALUDE_luke_connor_sleep_difference_l2247_224735


namespace NUMINAMATH_CALUDE_gilbert_parsley_count_l2247_224704

/-- Represents the number of herb plants Gilbert had at different stages of spring. -/
structure HerbCount where
  initial_basil : ℕ
  initial_parsley : ℕ
  initial_mint : ℕ
  final_basil : ℕ
  final_total : ℕ

/-- The conditions of Gilbert's herb garden during spring. -/
def spring_garden_conditions : HerbCount where
  initial_basil := 3
  initial_parsley := 0  -- We'll prove this is 1
  initial_mint := 2
  final_basil := 4
  final_total := 5

/-- Theorem stating that Gilbert planted 1 parsley plant initially. -/
theorem gilbert_parsley_count :
  spring_garden_conditions.initial_parsley = 1 :=
by sorry

end NUMINAMATH_CALUDE_gilbert_parsley_count_l2247_224704


namespace NUMINAMATH_CALUDE_cereal_box_servings_l2247_224721

def cereal_box_problem (total_cups : ℕ) (cups_per_serving : ℕ) : ℕ :=
  total_cups / cups_per_serving

theorem cereal_box_servings :
  cereal_box_problem 18 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_servings_l2247_224721


namespace NUMINAMATH_CALUDE_triangle_345_not_triangle_123_not_triangle_384_not_triangle_5510_l2247_224786

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that line segments of lengths 3, 4, and 5 can form a triangle -/
theorem triangle_345 : can_form_triangle 3 4 5 := by sorry

/-- Theorem stating that line segments of lengths 1, 2, and 3 cannot form a triangle -/
theorem not_triangle_123 : ¬can_form_triangle 1 2 3 := by sorry

/-- Theorem stating that line segments of lengths 3, 8, and 4 cannot form a triangle -/
theorem not_triangle_384 : ¬can_form_triangle 3 8 4 := by sorry

/-- Theorem stating that line segments of lengths 5, 5, and 10 cannot form a triangle -/
theorem not_triangle_5510 : ¬can_form_triangle 5 5 10 := by sorry

end NUMINAMATH_CALUDE_triangle_345_not_triangle_123_not_triangle_384_not_triangle_5510_l2247_224786


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l2247_224773

theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 16) 
  (h2 : stream_speed = 5) 
  (h3 : time = 4) : 
  boat_speed + stream_speed * time = 84 :=
by sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l2247_224773


namespace NUMINAMATH_CALUDE_odd_function_with_conditions_l2247_224772

def f (a b c x : ℤ) : ℚ := (a * x^2 + 1) / (b * x + c)

theorem odd_function_with_conditions (a b c : ℤ) :
  (∀ x, f a b c (-x) = -f a b c x) →  -- f is an odd function
  f a b c 1 = 2 →                     -- f(1) = 2
  f a b c 2 < 3 →                     -- f(2) < 3
  a = 1 ∧ b = 1 ∧ c = 0 :=            -- conclusion: a = 1, b = 1, c = 0
by sorry

end NUMINAMATH_CALUDE_odd_function_with_conditions_l2247_224772


namespace NUMINAMATH_CALUDE_octagon_side_length_l2247_224784

theorem octagon_side_length (square_side : ℝ) (h : square_side = 1) :
  let octagon_side := square_side - 2 * ((square_side * (1 - 1 / Real.sqrt 2)) / 2)
  octagon_side = 1 - Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_side_length_l2247_224784


namespace NUMINAMATH_CALUDE_function_bounds_l2247_224710

theorem function_bounds (x y z : ℝ) 
  (h1 : -1 ≤ 2*x + y - z ∧ 2*x + y - z ≤ 8)
  (h2 : 2 ≤ x - y + z ∧ x - y + z ≤ 9)
  (h3 : -3 ≤ x + 2*y - z ∧ x + 2*y - z ≤ 7) :
  -6 ≤ 7*x + 5*y - 2*z ∧ 7*x + 5*y - 2*z ≤ 47 := by
sorry

end NUMINAMATH_CALUDE_function_bounds_l2247_224710


namespace NUMINAMATH_CALUDE_triangle_with_pi_power_sum_is_acute_l2247_224788

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

-- Define the property of being an acute triangle
def IsAcute (t : Triangle) : Prop :=
  t.a^2 + t.b^2 > t.c^2

-- State the theorem
theorem triangle_with_pi_power_sum_is_acute (t : Triangle) 
  (h : t.a^Real.pi + t.b^Real.pi = t.c^Real.pi) : IsAcute t := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_pi_power_sum_is_acute_l2247_224788


namespace NUMINAMATH_CALUDE_lillian_candy_count_l2247_224717

theorem lillian_candy_count (initial_candies : ℕ) (additional_candies : ℕ) : 
  initial_candies = 88 → additional_candies = 5 → initial_candies + additional_candies = 93 := by
  sorry

end NUMINAMATH_CALUDE_lillian_candy_count_l2247_224717


namespace NUMINAMATH_CALUDE_projectile_meeting_time_l2247_224754

theorem projectile_meeting_time (initial_distance : ℝ) (speed1 speed2 : ℝ) :
  initial_distance = 1182 →
  speed1 = 460 →
  speed2 = 525 →
  (initial_distance / (speed1 + speed2)) * 60 = 72 := by
  sorry

end NUMINAMATH_CALUDE_projectile_meeting_time_l2247_224754


namespace NUMINAMATH_CALUDE_magic_square_d_plus_e_l2247_224771

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  sum_eq_row1 : sum = 30 + e + 24
  sum_eq_row2 : sum = 15 + c + d
  sum_eq_row3 : sum = a + 28 + b
  sum_eq_col1 : sum = 30 + 15 + a
  sum_eq_col2 : sum = e + c + 28
  sum_eq_col3 : sum = 24 + d + b
  sum_eq_diag1 : sum = 30 + c + b
  sum_eq_diag2 : sum = a + c + 24

theorem magic_square_d_plus_e (sq : MagicSquare) : sq.d + sq.e = 48 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_d_plus_e_l2247_224771


namespace NUMINAMATH_CALUDE_distance_to_line_l2247_224796

/-- The smallest distance from (0, 0) to the line y = 4/3 * x - 100 -/
def smallest_distance : ℝ := 60

/-- The equation of the line in the form Ax + By + C = 0 -/
def line_equation (x y : ℝ) : Prop := -4 * x + 3 * y + 300 = 0

/-- The point from which we're measuring the distance -/
def origin : ℝ × ℝ := (0, 0)

theorem distance_to_line :
  smallest_distance = 
    (‖-4 * origin.1 + 3 * origin.2 + 300‖ : ℝ) / Real.sqrt ((-4)^2 + 3^2) :=
sorry

end NUMINAMATH_CALUDE_distance_to_line_l2247_224796


namespace NUMINAMATH_CALUDE_probabilities_sum_to_one_l2247_224744

def p₁ : ℝ := 0.22
def p₂ : ℝ := 0.31
def p₃ : ℝ := 0.47

theorem probabilities_sum_to_one : p₁ + p₂ + p₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_probabilities_sum_to_one_l2247_224744


namespace NUMINAMATH_CALUDE_rectangle_partition_into_L_shapes_rectangle_1985_1987_not_partitionable_rectangle_1987_1989_partitionable_l2247_224732

/-- An L-shape is a figure composed of 3 unit squares -/
def LShape : Nat := 3

/-- Checks if a number is divisible by 3 -/
def isDivisibleBy3 (n : Nat) : Prop := n % 3 = 0

/-- Checks if a number leaves a remainder of 2 when divided by 3 -/
def hasRemainder2 (n : Nat) : Prop := n % 3 = 2

/-- Theorem: A rectangle can be partitioned into L-shapes iff
    1) Its area is divisible by 3, and
    2) At least one side is divisible by 3, or both sides have remainder 2 when divided by 3 -/
theorem rectangle_partition_into_L_shapes (m n : Nat) :
  (isDivisibleBy3 (m * n)) ∧ 
  (isDivisibleBy3 m ∨ isDivisibleBy3 n ∨ (hasRemainder2 m ∧ hasRemainder2 n)) ↔ 
  ∃ (k : Nat), m * n = k * LShape := by sorry

/-- Corollary: 1985 × 1987 rectangle cannot be partitioned into L-shapes -/
theorem rectangle_1985_1987_not_partitionable :
  ¬ ∃ (k : Nat), 1985 * 1987 = k * LShape := by sorry

/-- Corollary: 1987 × 1989 rectangle can be partitioned into L-shapes -/
theorem rectangle_1987_1989_partitionable :
  ∃ (k : Nat), 1987 * 1989 = k * LShape := by sorry

end NUMINAMATH_CALUDE_rectangle_partition_into_L_shapes_rectangle_1985_1987_not_partitionable_rectangle_1987_1989_partitionable_l2247_224732


namespace NUMINAMATH_CALUDE_grid_rectangles_l2247_224794

/-- The number of rectangles formed in a grid of parallel lines -/
def rectangles_in_grid (lines1 : ℕ) (lines2 : ℕ) : ℕ :=
  (lines1 - 1) * (lines2 - 1)

/-- Theorem: In a grid formed by 8 parallel lines intersected by 10 parallel lines, 
    the total number of rectangles formed is 63 -/
theorem grid_rectangles :
  rectangles_in_grid 8 10 = 63 := by
  sorry

end NUMINAMATH_CALUDE_grid_rectangles_l2247_224794


namespace NUMINAMATH_CALUDE_condition_relationship_l2247_224708

theorem condition_relationship : 
  let A := {x : ℝ | 0 < x ∧ x < 5}
  let B := {x : ℝ | |x - 2| < 3}
  (∀ x ∈ A, x ∈ B) ∧ (∃ x ∈ B, x ∉ A) := by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2247_224708


namespace NUMINAMATH_CALUDE_min_value_of_f_l2247_224720

/-- The polynomial function in two variables -/
def f (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

/-- The minimum value of the polynomial function -/
theorem min_value_of_f :
  ∀ x y : ℝ, f x y ≥ -18 ∧ ∃ a b : ℝ, f a b = -18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2247_224720


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2247_224781

theorem simple_interest_rate_calculation
  (P : ℝ)  -- Principal amount
  (t : ℝ)  -- Time period in years
  (SI : ℝ) -- Simple interest
  (h1 : P = 2800)
  (h2 : t = 5)
  (h3 : SI = P - 2240)
  (h4 : SI = (P * r * t) / 100) -- r is the interest rate
  : r = 4 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2247_224781


namespace NUMINAMATH_CALUDE_infinite_perfect_squares_l2247_224758

theorem infinite_perfect_squares (k : ℕ+) : 
  ∃ n : ℕ+, ∃ m : ℕ, (n * 2^k.val - 7 : ℤ) = m^2 :=
sorry

end NUMINAMATH_CALUDE_infinite_perfect_squares_l2247_224758


namespace NUMINAMATH_CALUDE_intersection_implies_x_zero_l2247_224709

def A : Set ℝ := {0, 1, 2, 4, 5}
def B (x : ℝ) : Set ℝ := {x-2, x, x+2}

theorem intersection_implies_x_zero (x : ℝ) (h : A ∩ B x = {0, 2}) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_x_zero_l2247_224709


namespace NUMINAMATH_CALUDE_sequence_properties_l2247_224739

def sequence_a (n : ℕ) : ℝ := 2 * n - 1

def sum_S (n : ℕ) : ℝ := n^2

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → 4 * (sum_S n) = (sequence_a n + 1)^2) →
  (∀ n : ℕ, n > 0 → sequence_a n = 2 * n - 1) ∧
  (sequence_a 1 = 1) ∧
  (sum_S 20 = 400) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2247_224739


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_45_4050_l2247_224749

theorem gcd_lcm_sum_45_4050 : Nat.gcd 45 4050 + Nat.lcm 45 4050 = 4095 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_45_4050_l2247_224749


namespace NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l2247_224716

/-- The number of positive integer divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly 12 divisors -/
def has_twelve_divisors (n : ℕ+) : Prop :=
  num_divisors n = 12

theorem smallest_with_twelve_divisors :
  ∃ (n : ℕ+), has_twelve_divisors n ∧ ∀ (m : ℕ+), has_twelve_divisors m → n ≤ m :=
by
  use 72
  sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l2247_224716


namespace NUMINAMATH_CALUDE_unique_albums_count_l2247_224719

/-- The number of albums in either Andrew's, John's, or Sarah's collection, but not in all three -/
def unique_albums (shared_albums andrew_total john_unique sarah_unique : ℕ) : ℕ :=
  (andrew_total - shared_albums) + john_unique + sarah_unique

/-- Theorem stating the number of unique albums across the three collections -/
theorem unique_albums_count :
  unique_albums 10 20 5 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_albums_count_l2247_224719


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l2247_224770

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 7800 →
  candidate_percentage = 35 / 100 →
  (total_votes : ℚ) * candidate_percentage < (total_votes : ℚ) * (1 - candidate_percentage) →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 2340 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l2247_224770


namespace NUMINAMATH_CALUDE_blue_lights_l2247_224766

/-- The number of blue lights on a Christmas tree -/
theorem blue_lights (total : ℕ) (red : ℕ) (yellow : ℕ) 
  (h1 : total = 95)
  (h2 : red = 26)
  (h3 : yellow = 37) :
  total - (red + yellow) = 32 := by
  sorry

end NUMINAMATH_CALUDE_blue_lights_l2247_224766


namespace NUMINAMATH_CALUDE_second_quadrant_trig_l2247_224705

theorem second_quadrant_trig (α : Real) (h : π / 2 < α ∧ α < π) : 
  Real.tan α + Real.sin α < 0 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_trig_l2247_224705


namespace NUMINAMATH_CALUDE_y_derivative_l2247_224762

noncomputable def y (x : ℝ) : ℝ := 
  (3 / (8 * Real.sqrt 2)) * Real.log ((Real.sqrt 2 + Real.tanh x) / (Real.sqrt 2 - Real.tanh x)) - 
  (Real.tanh x) / (4 * (2 - Real.tanh x ^ 2))

theorem y_derivative (x : ℝ) : 
  deriv y x = 1 / (2 + Real.cosh x ^ 2) ^ 2 :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l2247_224762


namespace NUMINAMATH_CALUDE_water_level_change_time_correct_l2247_224767

noncomputable def water_level_change_time (S H h s V g : ℝ) : ℝ :=
  let a := S / (0.6 * s * Real.sqrt (2 * g))
  let b := V / (0.6 * s * Real.sqrt (2 * g))
  2 * a * (Real.sqrt H - Real.sqrt (H - h) + b * Real.log (abs ((Real.sqrt H - b) / (Real.sqrt (H - h) - b))))

theorem water_level_change_time_correct
  (S H h s V g : ℝ)
  (h_S : S > 0)
  (h_H : H > 0)
  (h_h : 0 < h ∧ h < H)
  (h_s : s > 0)
  (h_V : V ≥ 0)
  (h_g : g > 0) :
  ∃ T : ℝ, T = water_level_change_time S H h s V g ∧ T > 0 :=
sorry

end NUMINAMATH_CALUDE_water_level_change_time_correct_l2247_224767


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2247_224740

theorem floor_ceil_sum : ⌊(-3.75 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ + (1/2 : ℝ) = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2247_224740


namespace NUMINAMATH_CALUDE_larger_triangle_perimeter_l2247_224706

/-- Given an isosceles triangle with side lengths 7, 7, and 12, and a similar triangle
    with longest side 36, the perimeter of the larger triangle is 78. -/
theorem larger_triangle_perimeter (a b c : ℝ) (d : ℝ) : 
  a = 7 ∧ b = 7 ∧ c = 12 ∧ d = 36 ∧ 
  (a = b) ∧ (c ≥ a) ∧ (c ≥ b) ∧
  (d / c = 36 / 12) →
  d + (d * a / c) + (d * b / c) = 78 := by
  sorry


end NUMINAMATH_CALUDE_larger_triangle_perimeter_l2247_224706


namespace NUMINAMATH_CALUDE_mike_pens_l2247_224789

/-- The number of pens Mike gave -/
def M : ℕ := sorry

/-- The initial number of pens -/
def initial_pens : ℕ := 5

/-- The number of pens given away -/
def pens_given_away : ℕ := 19

/-- The final number of pens -/
def final_pens : ℕ := 31

theorem mike_pens : 
  2 * (initial_pens + M) - pens_given_away = final_pens ∧ M = 20 := by sorry

end NUMINAMATH_CALUDE_mike_pens_l2247_224789


namespace NUMINAMATH_CALUDE_m_range_l2247_224703

-- Define propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the range of m
def range_m (m : ℝ) : Prop := m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3

-- Theorem statement
theorem m_range : 
  ∀ m : ℝ, (¬(P m ∧ Q m) ∧ (P m ∨ Q m)) → range_m m :=
sorry

end NUMINAMATH_CALUDE_m_range_l2247_224703


namespace NUMINAMATH_CALUDE_max_ellipse_area_in_rectangle_l2247_224776

/-- The maximum area of an ellipse inside a rectangle -/
theorem max_ellipse_area_in_rectangle (π : ℝ) (rectangle_length rectangle_width : ℝ) :
  rectangle_length = 18 ∧ rectangle_width = 14 →
  let semi_major_axis := rectangle_length / 2
  let semi_minor_axis := rectangle_width / 2
  let max_area := π * semi_major_axis * semi_minor_axis
  max_area = π * 63 :=
by sorry

end NUMINAMATH_CALUDE_max_ellipse_area_in_rectangle_l2247_224776


namespace NUMINAMATH_CALUDE_new_encoding_correct_l2247_224728

-- Define the encoding function
def encode (c : Char) : String :=
  match c with
  | 'A' => "21"
  | 'B' => "122"
  | 'C' => "1"
  | _ => ""

-- Define the decoding function (simplified for this problem)
def decode (s : String) : String :=
  if s = "011011010011" then "ABCBA" else ""

-- Theorem statement
theorem new_encoding_correct : 
  let original := "011011010011"
  let decoded := decode original
  String.join (List.map encode decoded.data) = "211221121" := by
  sorry


end NUMINAMATH_CALUDE_new_encoding_correct_l2247_224728


namespace NUMINAMATH_CALUDE_min_sum_with_constraints_l2247_224711

theorem min_sum_with_constraints (x y z w : ℕ+) (h : (4 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val ∧ (6 : ℕ) * z.val = (7 : ℕ) * w.val) :
  x.val + y.val + z.val + w.val ≥ 319 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraints_l2247_224711


namespace NUMINAMATH_CALUDE_expression_simplification_l2247_224750

theorem expression_simplification (a b c : ℝ) 
  (ha : a ≠ 2) (hb : b ≠ 3) (hc : c ≠ 6) : 
  ((a - 2) / (6 - c)) * ((b - 3) / (2 - a)) * ((c - 6) / (3 - b)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2247_224750


namespace NUMINAMATH_CALUDE_parallel_vectors_solution_l2247_224760

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

theorem parallel_vectors_solution (a : ℝ) :
  let m : Vector2D := ⟨a, -2⟩
  let n : Vector2D := ⟨1, 1 - a⟩
  parallel m n → a = 2 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_solution_l2247_224760


namespace NUMINAMATH_CALUDE_line_through_quadrants_l2247_224787

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point (x, y) is in the first quadrant -/
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Predicate to check if a point (x, y) is in the second quadrant -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Predicate to check if a point (x, y) is in the fourth quadrant -/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Predicate to check if a line passes through a given quadrant -/
def passes_through_quadrant (l : Line) (quad : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, quad x y ∧ l.a * x + l.b * y + l.c = 0

/-- Main theorem: If a line passes through the first, second, and fourth quadrants,
    then ac > 0 and bc < 0 -/
theorem line_through_quadrants (l : Line) :
  passes_through_quadrant l in_first_quadrant ∧
  passes_through_quadrant l in_second_quadrant ∧
  passes_through_quadrant l in_fourth_quadrant →
  l.a * l.c > 0 ∧ l.b * l.c < 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_quadrants_l2247_224787


namespace NUMINAMATH_CALUDE_max_area_of_triangle_l2247_224751

noncomputable section

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law holds for the triangle. -/
axiom sine_law (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The given condition in the problem. -/
axiom problem_condition (t : Triangle) : Real.sin t.A / t.a = Real.sqrt 3 * Real.cos t.B / t.b

/-- The side b is equal to 2. -/
axiom side_b_is_2 (t : Triangle) : t.b = 2

/-- The area of a triangle. -/
def triangle_area (t : Triangle) : ℝ := 1/2 * t.a * t.c * Real.sin t.B

/-- The theorem to be proved. -/
theorem max_area_of_triangle (t : Triangle) : 
  (∀ t' : Triangle, triangle_area t' ≤ triangle_area t) → triangle_area t = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_area_of_triangle_l2247_224751


namespace NUMINAMATH_CALUDE_puzzle_e_count_l2247_224725

/-- Represents the types of puzzle pieces -/
inductive PieceType
| A  -- Corner piece
| B  -- Edge piece
| C  -- Special edge piece
| D  -- Internal piece with 3 indentations
| E  -- Internal piece with 2 indentations

/-- Structure representing a rectangular puzzle -/
structure Puzzle where
  width : ℕ
  height : ℕ
  total_pieces : ℕ
  a_count : ℕ
  b_count : ℕ
  c_count : ℕ
  d_count : ℕ
  balance_equation : 2 * a_count + b_count + c_count + 3 * d_count = 2 * b_count + 2 * c_count + d_count

/-- Theorem stating the number of E-type pieces in the puzzle -/
theorem puzzle_e_count (p : Puzzle) 
  (h_dim : p.width = 23 ∧ p.height = 37)
  (h_total : p.total_pieces = 851)
  (h_a : p.a_count = 4)
  (h_b : p.b_count = 108)
  (h_c : p.c_count = 4)
  (h_d : p.d_count = 52) :
  p.total_pieces - (p.a_count + p.b_count + p.c_count + p.d_count) = 683 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_e_count_l2247_224725


namespace NUMINAMATH_CALUDE_queen_then_diamond_probability_l2247_224729

/-- Standard deck of cards --/
def standard_deck : ℕ := 52

/-- Number of Queens in a standard deck --/
def num_queens : ℕ := 4

/-- Number of diamonds in a standard deck --/
def num_diamonds : ℕ := 13

/-- Probability of drawing a Queen as the first card and a diamond as the second --/
def prob_queen_then_diamond : ℚ := 52 / 221

theorem queen_then_diamond_probability :
  prob_queen_then_diamond = (num_queens / standard_deck) * (num_diamonds / (standard_deck - 1)) :=
sorry

end NUMINAMATH_CALUDE_queen_then_diamond_probability_l2247_224729


namespace NUMINAMATH_CALUDE_angle_C_is_120_degrees_l2247_224792

theorem angle_C_is_120_degrees 
  (A B : ℝ) 
  (m : ℝ × ℝ) 
  (n : ℝ × ℝ) 
  (h1 : m = (Real.sqrt 3 * Real.sin A, Real.sin B))
  (h2 : n = (Real.cos B, Real.sqrt 3 * Real.cos A))
  (h3 : m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B))
  : ∃ C : ℝ, C = 2 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_angle_C_is_120_degrees_l2247_224792


namespace NUMINAMATH_CALUDE_min_y_over_x_on_ellipse_l2247_224736

/-- The minimum value of y/x for points on the ellipse 4(x-2)^2 + y^2 = 4 -/
theorem min_y_over_x_on_ellipse :
  ∃ (min : ℝ), min = -2 * Real.sqrt 3 / 3 ∧
  ∀ (x y : ℝ), 4 * (x - 2)^2 + y^2 = 4 →
  y / x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_y_over_x_on_ellipse_l2247_224736


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2247_224791

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 4) :
  1/x + 1/y ≥ 1 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 4 ∧ 1/a + 1/b = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2247_224791


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l2247_224768

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 6)
  (h₃ : contact_distance = 26) :
  let center_distance := Real.sqrt ((contact_distance ^ 2) + ((r₁ - r₂) ^ 2))
  center_distance = 2 * Real.sqrt 173 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l2247_224768


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2247_224743

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0) : 
  (s + 2*y = 3*s) →  -- Outer square side length
  (x + s = 3*s) →    -- Perpendicular arrangement
  ((3*s)^2 = 9*s^2)  -- Area ratio
  → x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2247_224743


namespace NUMINAMATH_CALUDE_chicken_farm_theorem_l2247_224774

/-- Represents the chicken farm problem -/
structure ChickenFarm where
  totalChicks : ℕ
  costA : ℕ
  costB : ℕ
  survivalRateA : ℚ
  survivalRateB : ℚ

/-- The solution to the chicken farm problem -/
def chickenFarmSolution (farm : ChickenFarm) : Prop :=
  -- Total number of chicks is 2000
  farm.totalChicks = 2000 ∧
  -- Cost of type A chick is 2 yuan
  farm.costA = 2 ∧
  -- Cost of type B chick is 3 yuan
  farm.costB = 3 ∧
  -- Survival rate of type A chicks is 94%
  farm.survivalRateA = 94/100 ∧
  -- Survival rate of type B chicks is 99%
  farm.survivalRateB = 99/100 ∧
  -- Question 1
  (∃ (x y : ℕ), x + y = farm.totalChicks ∧ 
    farm.costA * x + farm.costB * y = 4500 ∧
    x = 1500 ∧ y = 500) ∧
  -- Question 2
  (∃ (x : ℕ), x ≥ 1300 ∧
    ∀ (y : ℕ), y + x = farm.totalChicks →
      farm.costA * x + farm.costB * y ≤ 4700) ∧
  -- Question 3
  (∃ (x y : ℕ), x + y = farm.totalChicks ∧
    farm.survivalRateA * x + farm.survivalRateB * y ≥ 96/100 * farm.totalChicks ∧
    x = 1200 ∧ y = 800 ∧
    farm.costA * x + farm.costB * y = 4800 ∧
    ∀ (x' y' : ℕ), x' + y' = farm.totalChicks →
      farm.survivalRateA * x' + farm.survivalRateB * y' ≥ 96/100 * farm.totalChicks →
      farm.costA * x' + farm.costB * y' ≥ 4800)

theorem chicken_farm_theorem (farm : ChickenFarm) : chickenFarmSolution farm := by
  sorry

end NUMINAMATH_CALUDE_chicken_farm_theorem_l2247_224774


namespace NUMINAMATH_CALUDE_sons_ages_l2247_224797

theorem sons_ages (x y : ℕ+) (h1 : x < y) (h2 : y ≤ 4) 
  (h3 : ∃ (a b : ℕ+), a ≠ x ∧ b ≠ y ∧ a * b = x * y)
  (h4 : x ≠ y → (x = 1 ∧ y = 4)) :
  x = 1 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_sons_ages_l2247_224797


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l2247_224727

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x + 4}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the interval [4, +∞)
def interval_four_to_inf : Set ℝ := {x | x ≥ 4}

-- Theorem statement
theorem intersection_equals_interval : A_intersect_B = interval_four_to_inf := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l2247_224727


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l2247_224769

/-- Given a point with rectangular coordinates (-5, -7, 4) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, θ + π, -φ) has rectangular coordinates (5, 7, 4). -/
theorem spherical_coordinate_transformation (ρ θ φ : ℝ) :
  (ρ * Real.sin φ * Real.cos θ = -5) →
  (ρ * Real.sin φ * Real.sin θ = -7) →
  (ρ * Real.cos φ = 4) →
  (ρ * Real.sin (-φ) * Real.cos (θ + π) = 5) ∧
  (ρ * Real.sin (-φ) * Real.sin (θ + π) = 7) ∧
  (ρ * Real.cos (-φ) = 4) :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l2247_224769


namespace NUMINAMATH_CALUDE_remainder_sum_of_powers_l2247_224777

theorem remainder_sum_of_powers (n : ℕ) : (9^4 + 8^5 + 7^6) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_of_powers_l2247_224777
