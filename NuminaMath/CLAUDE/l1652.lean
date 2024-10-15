import Mathlib

namespace NUMINAMATH_CALUDE_determinant_of_geometric_sequence_l1652_165200

-- Define a geometric sequence of four terms
def is_geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r

-- State the theorem
theorem determinant_of_geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) :
  is_geometric_sequence a₁ a₂ a₃ a₄ → a₁ * a₄ - a₂ * a₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_geometric_sequence_l1652_165200


namespace NUMINAMATH_CALUDE_income_average_difference_l1652_165224

theorem income_average_difference (n : ℕ) (min_income max_income error_income : ℝ) :
  n > 0 ∧
  min_income = 8200 ∧
  max_income = 98000 ∧
  error_income = 980000 ∧
  n = 28 * 201000 →
  (error_income - max_income) / n = 882 :=
by sorry

end NUMINAMATH_CALUDE_income_average_difference_l1652_165224


namespace NUMINAMATH_CALUDE_share_a_correct_l1652_165283

/-- Calculates the share of profit for partner A given the investment details and total profit -/
def calculate_share_a (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) (total_months : ℕ) (change_month : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_a := initial_a * change_month + (initial_a - withdraw_a) * (total_months - change_month)
  let investment_months_b := initial_b * change_month + (initial_b + advance_b) * (total_months - change_month)
  let total_investment_months := investment_months_a + investment_months_b
  (investment_months_a * total_profit) / total_investment_months

theorem share_a_correct (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) (total_months : ℕ) (change_month : ℕ) (total_profit : ℕ) :
  initial_a = 3000 →
  initial_b = 4000 →
  withdraw_a = 1000 →
  advance_b = 1000 →
  total_months = 12 →
  change_month = 8 →
  total_profit = 840 →
  calculate_share_a initial_a initial_b withdraw_a advance_b total_months change_month total_profit = 320 :=
by sorry

end NUMINAMATH_CALUDE_share_a_correct_l1652_165283


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l1652_165274

theorem bobby_candy_problem (initial_candy : ℕ) (eaten_later : ℕ) (remaining_candy : ℕ)
  (h1 : initial_candy = 36)
  (h2 : eaten_later = 15)
  (h3 : remaining_candy = 4) :
  initial_candy - remaining_candy - eaten_later = 17 :=
by sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l1652_165274


namespace NUMINAMATH_CALUDE_wxyz_equals_mpwy_l1652_165221

/-- Assigns a numeric value to each letter of the alphabet -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14
  | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _ => 0

/-- The product of a four-letter list -/
def four_letter_product (a b c d : Char) : ℕ :=
  letter_value a * letter_value b * letter_value c * letter_value d

theorem wxyz_equals_mpwy :
  four_letter_product 'W' 'X' 'Y' 'Z' = four_letter_product 'M' 'P' 'W' 'Y' :=
by sorry

end NUMINAMATH_CALUDE_wxyz_equals_mpwy_l1652_165221


namespace NUMINAMATH_CALUDE_wine_consumption_problem_l1652_165251

/-- Represents the wine consumption problem from the Ming Dynasty's "The Great Compendium of Mathematics" -/
theorem wine_consumption_problem (x y : ℚ) : 
  (x + y = 19 ∧ 3 * x + (1/3) * y = 33) ↔ 
  (x ≥ 0 ∧ y ≥ 0 ∧ 
   ∃ (good_wine weak_wine guests : ℕ),
     good_wine = x ∧
     weak_wine = y ∧
     guests = 33 ∧
     good_wine + weak_wine = 19 ∧
     (3 * good_wine + (weak_wine / 3 : ℚ)) = guests) :=
by sorry

end NUMINAMATH_CALUDE_wine_consumption_problem_l1652_165251


namespace NUMINAMATH_CALUDE_segment_distance_sum_l1652_165247

/-- Represents a line segment with a midpoint -/
structure Segment where
  length : ℝ
  midpoint : ℝ

/-- The function relating distances from midpoints -/
def distance_relation (x y : ℝ) : Prop := y / x = 5 / 3

theorem segment_distance_sum 
  (ab : Segment) 
  (a'b' : Segment) 
  (h1 : ab.length = 3) 
  (h2 : a'b'.length = 5) 
  (x : ℝ) 
  (y : ℝ) 
  (h3 : distance_relation x y) 
  (h4 : x = 2) : 
  x + y = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_segment_distance_sum_l1652_165247


namespace NUMINAMATH_CALUDE_total_books_and_magazines_l1652_165281

def books_per_shelf : ℕ := 23
def magazines_per_shelf : ℕ := 61
def number_of_shelves : ℕ := 29

theorem total_books_and_magazines :
  books_per_shelf * number_of_shelves + magazines_per_shelf * number_of_shelves = 2436 := by
  sorry

end NUMINAMATH_CALUDE_total_books_and_magazines_l1652_165281


namespace NUMINAMATH_CALUDE_parabola_translation_l1652_165276

/-- Given a parabola y = -2x^2, prove that translating it upwards by 1 unit
    and to the right by 2 units results in the equation y = -2(x-2)^2 + 1 -/
theorem parabola_translation (x y : ℝ) :
  (y = -2 * x^2) →
  (y + 1 = -2 * ((x - 2)^2) + 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l1652_165276


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1652_165250

theorem complex_equation_solution :
  ∀ y : ℝ,
  let z₁ : ℂ := 3 + y * Complex.I
  let z₂ : ℂ := 2 - Complex.I
  z₁ / z₂ = 1 + Complex.I →
  y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1652_165250


namespace NUMINAMATH_CALUDE_sum_abs_roots_quadratic_l1652_165294

theorem sum_abs_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * r₁^2 + b * r₁ + c = 0 ∧ 
  a * r₂^2 + b * r₂ + c = 0 →
  |r₁| + |r₂| = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_abs_roots_quadratic_l1652_165294


namespace NUMINAMATH_CALUDE_library_bookshelf_selection_l1652_165291

/-- Represents a bookshelf with three tiers -/
structure Bookshelf :=
  (tier1 : ℕ)
  (tier2 : ℕ)
  (tier3 : ℕ)

/-- The number of ways to select a book from a bookshelf -/
def selectBook (b : Bookshelf) : ℕ := b.tier1 + b.tier2 + b.tier3

/-- Theorem: The number of ways to select a book from the given bookshelf is 16 -/
theorem library_bookshelf_selection :
  ∃ (b : Bookshelf), b.tier1 = 3 ∧ b.tier2 = 5 ∧ b.tier3 = 8 ∧ selectBook b = 16 := by
  sorry


end NUMINAMATH_CALUDE_library_bookshelf_selection_l1652_165291


namespace NUMINAMATH_CALUDE_remaining_three_digit_numbers_l1652_165292

/-- The number of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The number of three-digit numbers where the first and last digits are the same
    but the middle digit is different -/
def excluded_numbers : ℕ := 81

/-- The number of valid three-digit numbers after exclusion -/
def valid_numbers : ℕ := total_three_digit_numbers - excluded_numbers

theorem remaining_three_digit_numbers : valid_numbers = 819 := by
  sorry

end NUMINAMATH_CALUDE_remaining_three_digit_numbers_l1652_165292


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_coefficient_l1652_165228

theorem quadratic_equal_roots_coefficient (k : ℝ) (h : k = 1.7777777777777777) : 
  let eq := fun x : ℝ => 2 * k * x^2 + 3 * k * x + 2
  let discriminant := (3 * k)^2 - 4 * (2 * k) * 2
  discriminant = 0 → 3 * k = 5.333333333333333 :=
by
  sorry

#eval (3 : Float) * 1.7777777777777777

end NUMINAMATH_CALUDE_quadratic_equal_roots_coefficient_l1652_165228


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_union_of_A_and_B_range_of_p_l1652_165267

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}
def C (p : ℝ) : Set ℝ := {x | 4*x + p < 0}

-- State the theorems
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} := by sorry

theorem union_of_A_and_B : A ∪ B = Set.univ := by sorry

theorem range_of_p (p : ℝ) : C p ⊆ A → p ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_union_of_A_and_B_range_of_p_l1652_165267


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l1652_165275

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  9*x + 40*y + 18 = 0 ∨ x = -2

-- Theorem statement
theorem circle_and_tangent_line :
  -- Given conditions
  (circle_C 0 0) ∧
  (circle_C 6 0) ∧
  (circle_C 0 8) ∧
  -- Line l passes through (-2, 0)
  (line_l (-2) 0) ∧
  -- Line l is tangent to circle C
  (∃ (x y : ℝ), circle_C x y ∧ line_l x y ∧
    (∀ (x' y' : ℝ), line_l x' y' → (x' - x)^2 + (y' - y)^2 > 0 ∨ (x' = x ∧ y' = y))) →
  -- Conclusion: The equations of C and l are correct
  (∀ (x y : ℝ), circle_C x y ↔ (x - 3)^2 + (y - 4)^2 = 25) ∧
  (∀ (x y : ℝ), line_l x y ↔ (9*x + 40*y + 18 = 0 ∨ x = -2)) :=
by sorry


end NUMINAMATH_CALUDE_circle_and_tangent_line_l1652_165275


namespace NUMINAMATH_CALUDE_problem_solution_l1652_165218

def sequence1 (n : ℕ) : ℤ := (-1)^n * (2*n - 1)
def sequence2 (n : ℕ) : ℤ := (-1)^n * (2*n - 1) - 2
def sequence3 (n : ℕ) : ℤ := 3 * (2*n - 1) * (-1)^(n+1)

theorem problem_solution :
  (sequence1 10 = 19) ∧
  (sequence2 15 = -31) ∧
  (∀ n : ℕ, sequence2 n + sequence2 (n+1) + sequence2 (n+2) ≠ 1001) ∧
  (∃! k : ℕ, k % 2 = 1 ∧ sequence1 k + sequence2 k + sequence3 k = 599 ∧ k = 301) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1652_165218


namespace NUMINAMATH_CALUDE_tangent_line_of_odd_function_l1652_165207

/-- Given function f(x) = (a-1)x^2 - a*sin(x) is odd, 
    prove that its tangent line at (0,0) is y = -x -/
theorem tangent_line_of_odd_function (a : ℝ) :
  (∀ x, ((a - 1) * x^2 - a * Real.sin x) = -((a - 1) * (-x)^2 - a * Real.sin (-x))) →
  (∃ f : ℝ → ℝ, (∀ x, f x = (a - 1) * x^2 - a * Real.sin x) ∧ 
    (∃ f' : ℝ → ℝ, (∀ x, HasDerivAt f (f' x) x) ∧ f' 0 = -1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_of_odd_function_l1652_165207


namespace NUMINAMATH_CALUDE_probability_of_king_is_one_thirteenth_l1652_165241

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (kings : ℕ)

/-- The probability of drawing a specific card type from a deck -/
def probability_of_draw (deck : Deck) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / deck.total_cards

/-- Theorem: The probability of drawing a King from a standard deck is 1/13 -/
theorem probability_of_king_is_one_thirteenth (deck : Deck) 
  (h1 : deck.total_cards = 52)
  (h2 : deck.ranks = 13)
  (h3 : deck.suits = 4)
  (h4 : deck.kings = 4) :
  probability_of_draw deck deck.kings = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_king_is_one_thirteenth_l1652_165241


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1652_165289

/-- The total cost of sandwiches and sodas -/
def total_cost (sandwich_price : ℚ) (soda_price : ℚ) (sandwich_quantity : ℕ) (soda_quantity : ℕ) : ℚ :=
  sandwich_price * sandwich_quantity + soda_price * soda_quantity

/-- Theorem: The total cost of 2 sandwiches at $2.49 each and 4 sodas at $1.87 each is $12.46 -/
theorem total_cost_calculation :
  total_cost (249/100) (187/100) 2 4 = 1246/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1652_165289


namespace NUMINAMATH_CALUDE_davids_physics_marks_l1652_165257

/-- Calculates the marks in Physics given marks in other subjects and the average --/
def physics_marks (english : ℕ) (math : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + math + chemistry + biology)

/-- Proves that David's marks in Physics are 82 given his other marks and average --/
theorem davids_physics_marks :
  physics_marks 61 65 67 85 72 = 82 := by
  sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l1652_165257


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1652_165202

/-- The number of distinct intersection points of diagonals in the interior of a regular decagon -/
def diagonal_intersections (n : ℕ) : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  diagonal_intersections 10 = 210 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1652_165202


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l1652_165298

theorem loss_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1500 → 
  selling_price = 1335 → 
  (cost_price - selling_price) / cost_price * 100 = 11 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l1652_165298


namespace NUMINAMATH_CALUDE_trapezoid_median_equals_nine_inches_l1652_165260

/-- Given a triangle and a trapezoid with equal areas and the same altitude,
    if the base of the triangle is 18 inches, then the median of the trapezoid is 9 inches. -/
theorem trapezoid_median_equals_nine_inches 
  (triangle_area trapezoid_area : ℝ) 
  (triangle_altitude trapezoid_altitude : ℝ) 
  (triangle_base : ℝ) 
  (trapezoid_median : ℝ) :
  triangle_area = trapezoid_area →
  triangle_altitude = trapezoid_altitude →
  triangle_base = 18 →
  triangle_area = (1/2) * triangle_base * triangle_altitude →
  trapezoid_area = trapezoid_median * trapezoid_altitude →
  trapezoid_median = 9 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_median_equals_nine_inches_l1652_165260


namespace NUMINAMATH_CALUDE_equation_solution_l1652_165246

theorem equation_solution (x : ℝ) : 
  |x - 3| + x^2 = 10 ↔ 
  x = (-1 + Real.sqrt 53) / 2 ∨ 
  x = (1 + Real.sqrt 29) / 2 ∨ 
  x = (1 - Real.sqrt 29) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1652_165246


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1652_165212

theorem sqrt_equation_solution :
  ∃ y : ℝ, (Real.sqrt (4 - 2 * y) = 9) ∧ (y = -38.5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1652_165212


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l1652_165226

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 9 10))) = 630 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l1652_165226


namespace NUMINAMATH_CALUDE_surf_festival_attendance_l1652_165273

/-- The number of additional surfers on the second day of the Rip Curl Myrtle Beach Surf Festival --/
def additional_surfers : ℕ := 600

theorem surf_festival_attendance :
  let first_day : ℕ := 1500
  let third_day : ℕ := (2 : ℕ) * first_day / (5 : ℕ)
  let total_surfers : ℕ := first_day + (first_day + additional_surfers) + third_day
  let average_surfers : ℕ := 1400
  total_surfers / 3 = average_surfers :=
by sorry

end NUMINAMATH_CALUDE_surf_festival_attendance_l1652_165273


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l1652_165206

def scores : List ℝ := [84, 90, 87, 93, 88, 92]

theorem arithmetic_mean_of_scores : 
  (scores.sum / scores.length : ℝ) = 89 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l1652_165206


namespace NUMINAMATH_CALUDE_cube_of_square_of_second_smallest_prime_l1652_165236

-- Define the second smallest prime number
def second_smallest_prime : Nat := 3

-- Theorem statement
theorem cube_of_square_of_second_smallest_prime : 
  (second_smallest_prime ^ 2) ^ 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_square_of_second_smallest_prime_l1652_165236


namespace NUMINAMATH_CALUDE_quadratic_function_ellipse_l1652_165248

/-- Given a quadratic function y = ax^2 + bx + c where ac ≠ 0,
    with vertex (-b/(2a), -1/(4a)),
    and intersections with x-axis on opposite sides of y-axis,
    prove that (b, c) lies on the ellipse b^2 + c^2/4 = 1 --/
theorem quadratic_function_ellipse (a b c : ℝ) (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (∃ (p q : ℝ), p < 0 ∧ q > 0 ∧ a * p^2 + b * p + c = 0 ∧ a * q^2 + b * q + c = 0) →
  (∃ (m : ℝ), m = -4 ∧ (b / (2 * a))^2 + m^2 = ((b^2 - 4 * a * c) / (4 * a^2))) →
  b^2 + c^2 / 4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_ellipse_l1652_165248


namespace NUMINAMATH_CALUDE_polly_mirror_rate_l1652_165239

/-- Polly's tweeting behavior -/
structure PollyTweets where
  happy_rate : ℕ      -- tweets per minute when happy
  hungry_rate : ℕ     -- tweets per minute when hungry
  mirror_rate : ℕ     -- tweets per minute when watching mirror
  happy_time : ℕ      -- time spent being happy (in minutes)
  hungry_time : ℕ     -- time spent being hungry (in minutes)
  mirror_time : ℕ     -- time spent watching mirror (in minutes)
  total_tweets : ℕ    -- total number of tweets

/-- Theorem about Polly's tweeting rate when watching the mirror -/
theorem polly_mirror_rate (p : PollyTweets)
  (h1 : p.happy_rate = 18)
  (h2 : p.hungry_rate = 4)
  (h3 : p.happy_time = 20)
  (h4 : p.hungry_time = 20)
  (h5 : p.mirror_time = 20)
  (h6 : p.total_tweets = 1340)
  (h7 : p.total_tweets = p.happy_rate * p.happy_time + p.hungry_rate * p.hungry_time + p.mirror_rate * p.mirror_time) :
  p.mirror_rate = 45 := by
  sorry

end NUMINAMATH_CALUDE_polly_mirror_rate_l1652_165239


namespace NUMINAMATH_CALUDE_correct_calculation_l1652_165261

theorem correct_calculation (x : ℤ) (h : x - 749 = 280) : x + 479 = 1508 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1652_165261


namespace NUMINAMATH_CALUDE_complex_equality_l1652_165216

theorem complex_equality (a : ℝ) (z : ℂ) : 
  z = (a + 3*I) / (1 + 2*I) → z.re = z.im → a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_equality_l1652_165216


namespace NUMINAMATH_CALUDE_badArrangementsCount_l1652_165285

-- Define a type for circular arrangements
def CircularArrangement := List ℕ

-- Define what it means for an arrangement to be valid
def isValidArrangement (arr : CircularArrangement) : Prop :=
  arr.length = 6 ∧ arr.toFinset = {1, 2, 3, 4, 5, 6}

-- Define consecutive subsets in a circular arrangement
def consecutiveSubsets (arr : CircularArrangement) : List (List ℕ) :=
  sorry

-- Define what it means for an arrangement to be "bad"
def isBadArrangement (arr : CircularArrangement) : Prop :=
  ∃ n : ℕ, n ≥ 1 ∧ n ≤ 20 ∧ ∀ subset ∈ consecutiveSubsets arr, (subset.sum ≠ n)

-- Define equivalence of arrangements under rotation and reflection
def areEquivalentArrangements (arr1 arr2 : CircularArrangement) : Prop :=
  sorry

-- The main theorem
theorem badArrangementsCount :
  ∃ badArrs : List CircularArrangement,
    badArrs.length = 3 ∧
    (∀ arr ∈ badArrs, isValidArrangement arr ∧ isBadArrangement arr) ∧
    (∀ arr, isValidArrangement arr → isBadArrangement arr →
      ∃ badArr ∈ badArrs, areEquivalentArrangements arr badArr) :=
  sorry

end NUMINAMATH_CALUDE_badArrangementsCount_l1652_165285


namespace NUMINAMATH_CALUDE_chord_length_in_circle_l1652_165271

/-- The length of the chord cut by the line x = 1/2 from the circle (x-1)^2 + y^2 = 1 is √3 -/
theorem chord_length_in_circle (x y : ℝ) : 
  (x = 1/2) → ((x - 1)^2 + y^2 = 1) → 
  ∃ (y1 y2 : ℝ), y1 ≠ y2 ∧ 
    ((1/2 - 1)^2 + y1^2 = 1) ∧ 
    ((1/2 - 1)^2 + y2^2 = 1) ∧
    ((1/2 - 1/2)^2 + (y1 - y2)^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_in_circle_l1652_165271


namespace NUMINAMATH_CALUDE_notebook_cost_l1652_165227

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buying_students : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
  total_students = 42 ∧
  buying_students > total_students / 2 ∧
  notebooks_per_student > 1 ∧
  cost_per_notebook > notebooks_per_student ∧
  buying_students * notebooks_per_student * cost_per_notebook = total_cost ∧
  total_cost = 3213 ∧
  cost_per_notebook = 17 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l1652_165227


namespace NUMINAMATH_CALUDE_travel_distance_proof_l1652_165295

theorem travel_distance_proof (total_distance : ℝ) (plane_fraction : ℝ) (train_to_bus_ratio : ℝ) 
  (h1 : total_distance = 1800)
  (h2 : plane_fraction = 1/3)
  (h3 : train_to_bus_ratio = 2/3) : 
  ∃ (bus_distance : ℝ), 
    bus_distance = 720 ∧ 
    plane_fraction * total_distance + train_to_bus_ratio * bus_distance + bus_distance = total_distance :=
by sorry

end NUMINAMATH_CALUDE_travel_distance_proof_l1652_165295


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l1652_165282

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℚ, 2 * x + 4 * y = 5 ∧ x = 1 - y) →
  (∃ x y : ℚ, 2 * x + 4 * y = 5 ∧ x = 1 - y ∧ x = -1/2 ∧ y = 3/2) ∧
  -- System 2
  (∃ x y : ℚ, 5 * x + 6 * y = 4 ∧ 3 * x - 4 * y = 10) →
  (∃ x y : ℚ, 5 * x + 6 * y = 4 ∧ 3 * x - 4 * y = 10 ∧ x = 2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l1652_165282


namespace NUMINAMATH_CALUDE_smallest_n_for_real_power_l1652_165234

def complex_i : ℂ := Complex.I

def is_real (z : ℂ) : Prop := z.im = 0

theorem smallest_n_for_real_power :
  ∃ (n : ℕ), n > 0 ∧ is_real ((1 + complex_i) ^ n) ∧
  ∀ (m : ℕ), 0 < m → m < n → ¬ is_real ((1 + complex_i) ^ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_real_power_l1652_165234


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l1652_165263

noncomputable def f (x : ℝ) : ℝ := (2^x) / (2 * (Real.log 2 - 1) * x)

theorem f_derivative_at_one :
  deriv f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l1652_165263


namespace NUMINAMATH_CALUDE_roses_in_vase_l1652_165238

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 7

/-- The number of roses added to the vase -/
def added_roses : ℕ := 16

/-- The total number of roses after addition -/
def total_roses : ℕ := 23

theorem roses_in_vase : initial_roses + added_roses = total_roses := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l1652_165238


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1652_165262

theorem circle_area_ratio (R_C R_D : ℝ) (h : R_C > 0 ∧ R_D > 0) :
  (60 / 360 * (2 * Real.pi * R_C) = 2 * (40 / 360 * (2 * Real.pi * R_D))) →
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1652_165262


namespace NUMINAMATH_CALUDE_daughters_age_is_twelve_l1652_165293

/-- Proves that the daughter's age is 12 given the conditions about the father and daughter's ages -/
theorem daughters_age_is_twelve (D : ℕ) (F : ℕ) : 
  F = 3 * D →  -- Father's age is three times daughter's age this year
  F + 12 = 2 * (D + 12) →  -- After 12 years, father's age will be twice daughter's age
  D = 12 :=  -- Daughter's current age is 12
by
  sorry


end NUMINAMATH_CALUDE_daughters_age_is_twelve_l1652_165293


namespace NUMINAMATH_CALUDE_monthly_interest_rate_equation_l1652_165245

/-- The monthly interest rate that satisfies the compound interest equation for a loan of $200 with $22 interest charged in the second month. -/
theorem monthly_interest_rate_equation : ∃ r : ℝ, 200 * (1 + r)^2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_monthly_interest_rate_equation_l1652_165245


namespace NUMINAMATH_CALUDE_base_8_to_10_conversion_l1652_165213

theorem base_8_to_10_conversion : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0 : ℕ) = 2394 := by
  sorry

end NUMINAMATH_CALUDE_base_8_to_10_conversion_l1652_165213


namespace NUMINAMATH_CALUDE_john_distance_l1652_165217

/-- Calculates the total distance John travels given his speeds and running times -/
def total_distance (solo_speed : ℝ) (dog_speed : ℝ) (time_with_dog : ℝ) (time_solo : ℝ) : ℝ :=
  dog_speed * time_with_dog + solo_speed * time_solo

/-- Proves that John travels 5 miles given the specified conditions -/
theorem john_distance :
  let solo_speed : ℝ := 4
  let dog_speed : ℝ := 6
  let time_with_dog : ℝ := 0.5
  let time_solo : ℝ := 0.5
  total_distance solo_speed dog_speed time_with_dog time_solo = 5 := by
  sorry

#eval total_distance 4 6 0.5 0.5

end NUMINAMATH_CALUDE_john_distance_l1652_165217


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l1652_165215

theorem power_tower_mod_500 : 7^(7^(7^7)) ≡ 343 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l1652_165215


namespace NUMINAMATH_CALUDE_abc_congruence_l1652_165214

theorem abc_congruence (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 →
  (2 * a + 3 * b + c) % 7 = 1 →
  (3 * a + b + 2 * c) % 7 = 2 →
  (a + b + c) % 7 = 3 →
  (2 * a * b * c) % 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_congruence_l1652_165214


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_existence_of_547_smallest_k_is_547_l1652_165225

theorem smallest_k_with_remainder_one (k : ℕ) : k > 1 ∧ 
  k % 13 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 ∧ k % 2 = 1 → k ≥ 547 := by
  sorry

theorem existence_of_547 : 
  547 > 1 ∧ 547 % 13 = 1 ∧ 547 % 7 = 1 ∧ 547 % 3 = 1 ∧ 547 % 2 = 1 := by
  sorry

theorem smallest_k_is_547 : ∃! k : ℕ, k > 1 ∧ 
  k % 13 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 ∧ k % 2 = 1 ∧ 
  ∀ m : ℕ, (m > 1 ∧ m % 13 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 ∧ m % 2 = 1) → k ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_existence_of_547_smallest_k_is_547_l1652_165225


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1652_165240

theorem smallest_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 124)
  (h_prop : a = 2 ∧ b = 1/2 ∧ c = 1/4) :
  let x := total / (a + b + c)
  min (a * x) (min (b * x) (c * x)) = 124 / 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1652_165240


namespace NUMINAMATH_CALUDE_apples_per_box_l1652_165255

theorem apples_per_box (total_apples : ℕ) (num_boxes : ℕ) (leftover_apples : ℕ) 
  (h1 : total_apples = 32) 
  (h2 : num_boxes = 7) 
  (h3 : leftover_apples = 4) : 
  (total_apples - leftover_apples) / num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_box_l1652_165255


namespace NUMINAMATH_CALUDE_no_valid_operation_l1652_165270

def basic_op (x y : ℝ) : Set ℝ :=
  {x + y, x - y, x * y, x / y}

theorem no_valid_operation :
  ∀ op ∈ basic_op 9 2, (op * 3 + (4 * 2) - 6) ≠ 21 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_operation_l1652_165270


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l1652_165279

theorem consecutive_integers_product_812_sum_57 (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l1652_165279


namespace NUMINAMATH_CALUDE_mikes_ride_length_l1652_165220

/-- Represents the taxi ride problem --/
structure TaxiRide where
  startingAmount : ℝ
  costPerMile : ℝ
  anniesMiles : ℝ
  bridgeToll : ℝ

/-- The theorem stating that Mike's ride was 46 miles long --/
theorem mikes_ride_length (ride : TaxiRide) 
  (h1 : ride.startingAmount = 2.5)
  (h2 : ride.costPerMile = 0.25)
  (h3 : ride.anniesMiles = 26)
  (h4 : ride.bridgeToll = 5) :
  ∃ (mikesMiles : ℝ), 
    mikesMiles = 46 ∧ 
    ride.startingAmount + ride.costPerMile * mikesMiles = 
    ride.startingAmount + ride.bridgeToll + ride.costPerMile * ride.anniesMiles :=
by
  sorry


end NUMINAMATH_CALUDE_mikes_ride_length_l1652_165220


namespace NUMINAMATH_CALUDE_profit_calculation_l1652_165259

-- Define the variables
def charge_per_lawn : ℕ := 12
def lawns_mowed : ℕ := 3
def gas_expense : ℕ := 17
def extra_income : ℕ := 10

-- Define Tom's profit
def toms_profit : ℕ := charge_per_lawn * lawns_mowed + extra_income - gas_expense

-- Theorem statement
theorem profit_calculation : toms_profit = 29 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l1652_165259


namespace NUMINAMATH_CALUDE_photo_frame_perimeter_l1652_165223

theorem photo_frame_perimeter (frame_width : ℝ) (frame_area : ℝ) (outer_edge : ℝ) :
  frame_width = 2 →
  frame_area = 48 →
  outer_edge = 10 →
  ∃ (photo_length photo_width : ℝ),
    photo_length = outer_edge - 2 * frame_width ∧
    photo_width * (outer_edge - 2 * frame_width) = outer_edge * (frame_area / outer_edge) - frame_area ∧
    2 * (photo_length + photo_width) = 16 :=
by sorry

end NUMINAMATH_CALUDE_photo_frame_perimeter_l1652_165223


namespace NUMINAMATH_CALUDE_book_pricing_and_cost_theorem_l1652_165258

/-- Represents the price and quantity of books --/
structure BookInfo where
  edu_price : ℝ
  ele_price : ℝ
  edu_quantity : ℕ
  ele_quantity : ℕ

/-- Calculates the total cost of books --/
def total_cost (info : BookInfo) : ℝ :=
  info.edu_price * info.edu_quantity + info.ele_price * info.ele_quantity

/-- Checks if the quantity constraint is satisfied --/
def quantity_constraint (info : BookInfo) : Prop :=
  info.edu_quantity ≤ 3 * info.ele_quantity ∧ info.edu_quantity ≥ 70

/-- The main theorem to be proven --/
theorem book_pricing_and_cost_theorem (info : BookInfo) : 
  (total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := 2, ele_quantity := 3} = 126) →
  (total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := 3, ele_quantity := 2} = 109) →
  (info.edu_price = 15 ∧ info.ele_price = 32) ∧
  (∀ m : ℕ, m + info.ele_quantity = 200 → quantity_constraint {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} →
    total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} ≥ 3850) ∧
  (∃ m : ℕ, m + info.ele_quantity = 200 ∧ 
    quantity_constraint {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} ∧
    total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} = 3850) :=
by sorry

end NUMINAMATH_CALUDE_book_pricing_and_cost_theorem_l1652_165258


namespace NUMINAMATH_CALUDE_negation_of_all_birds_can_fly_l1652_165286

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (bird : U → Prop)
variable (can_fly : U → Prop)

-- State the theorem
theorem negation_of_all_birds_can_fly :
  (¬ ∀ (x : U), bird x → can_fly x) ↔ (∃ (x : U), bird x ∧ ¬ can_fly x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_birds_can_fly_l1652_165286


namespace NUMINAMATH_CALUDE_solve_for_x_l1652_165233

theorem solve_for_x (x y z : ℝ) 
  (eq1 : x + y = 75)
  (eq2 : (x + y) + y + z = 130)
  (eq3 : z = y + 10) :
  x = 52.5 := by sorry

end NUMINAMATH_CALUDE_solve_for_x_l1652_165233


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l1652_165208

def point : ℝ × ℝ := (2, -3)

theorem distance_to_x_axis : abs (point.2) = 3 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l1652_165208


namespace NUMINAMATH_CALUDE_f_at_4_l1652_165231

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 3x^4 - 5x^3 + 7x^2 - 9x + 11 -/
def f : List ℤ := [11, -9, 7, -5, 3, 1]

/-- Theorem: The value of f(4) is 1559 -/
theorem f_at_4 : horner f 4 = 1559 := by
  sorry

end NUMINAMATH_CALUDE_f_at_4_l1652_165231


namespace NUMINAMATH_CALUDE_expression_value_l1652_165201

theorem expression_value (a b c d x : ℝ) : 
  a = -b → cd = 1 → abs x = 2 → 
  x^2 - (a + b + cd) * x + (a + b)^2021 + (-cd)^2022 = 3 ∨
  x^2 - (a + b + cd) * x + (a + b)^2021 + (-cd)^2022 = 7 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1652_165201


namespace NUMINAMATH_CALUDE_range_of_m_for_root_in_interval_l1652_165253

/-- Given a function f(x) = 2x - m with a root in the interval (1, 2), 
    prove that the range of m is 2 < m < 4 -/
theorem range_of_m_for_root_in_interval 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = 2 * x - m) 
  (h2 : ∃ x ∈ Set.Ioo 1 2, f x = 0) : 
  2 < m ∧ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_root_in_interval_l1652_165253


namespace NUMINAMATH_CALUDE_three_digit_sum_condition_l1652_165280

/-- Represents a three-digit number abc in decimal form -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_range : a ≤ 9
  b_range : b ≤ 9
  c_range : c ≤ 9
  a_nonzero : a ≠ 0

/-- The value of a three-digit number abc in decimal form -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of all two-digit numbers formed by the digits a, b, c -/
def ThreeDigitNumber.sumTwoDigit (n : ThreeDigitNumber) : Nat :=
  2 * (10 * n.a + 10 * n.b + 10 * n.c)

/-- A three-digit number satisfies the condition if its value equals the sum of all two-digit numbers formed by its digits -/
def ThreeDigitNumber.satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  n.value = n.sumTwoDigit

/-- The theorem stating that only 132, 264, and 396 satisfy the condition -/
theorem three_digit_sum_condition :
  ∀ n : ThreeDigitNumber, n.satisfiesCondition ↔ (n.value = 132 ∨ n.value = 264 ∨ n.value = 396) := by
  sorry


end NUMINAMATH_CALUDE_three_digit_sum_condition_l1652_165280


namespace NUMINAMATH_CALUDE_man_speed_against_current_l1652_165211

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specified speeds, 
    the man's speed against the current is 10 km/hr. -/
theorem man_speed_against_current :
  speed_against_current 15 2.5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_against_current_l1652_165211


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l1652_165242

theorem complex_power_magnitude (z : ℂ) :
  z = (1 / Real.sqrt 2 : ℂ) + (Complex.I / Real.sqrt 2) →
  Complex.abs (z^8) = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l1652_165242


namespace NUMINAMATH_CALUDE_alan_roof_weight_l1652_165244

/-- The number of pine trees in Alan's backyard -/
def num_trees : ℕ := 8

/-- The number of pine cones each tree drops -/
def cones_per_tree : ℕ := 200

/-- The percentage of pine cones that fall on the roof -/
def roof_percentage : ℚ := 30 / 100

/-- The weight of each pine cone in ounces -/
def cone_weight : ℚ := 4

/-- The total weight of pine cones on Alan's roof in ounces -/
def roof_weight : ℚ := num_trees * cones_per_tree * roof_percentage * cone_weight

theorem alan_roof_weight : roof_weight = 1920 := by sorry

end NUMINAMATH_CALUDE_alan_roof_weight_l1652_165244


namespace NUMINAMATH_CALUDE_integer_property_l1652_165284

theorem integer_property (k : ℕ) : k ≥ 3 → (
  (∃ m n : ℕ, 1 < m ∧ m < k ∧
              1 < n ∧ n < k ∧
              Nat.gcd m k = 1 ∧
              Nat.gcd n k = 1 ∧
              m + n > k ∧
              k ∣ (m - 1) * (n - 1))
  ↔ (k = 15 ∨ k = 30)
) := by
  sorry

end NUMINAMATH_CALUDE_integer_property_l1652_165284


namespace NUMINAMATH_CALUDE_descending_order_of_powers_l1652_165203

theorem descending_order_of_powers : 
  2^(2/3) > (-1.8)^(2/3) ∧ (-1.8)^(2/3) > (-2)^(1/3) := by sorry

end NUMINAMATH_CALUDE_descending_order_of_powers_l1652_165203


namespace NUMINAMATH_CALUDE_subtracted_value_proof_l1652_165265

theorem subtracted_value_proof (n : ℝ) (x : ℝ) : 
  n = 15.0 → 3 * n - x = 40 → x = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_proof_l1652_165265


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l1652_165243

-- Define the hyperbola equation
def hyperbola_equation (x y m n : ℝ) : Prop :=
  x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

-- Define the condition for the distance between foci
def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

-- Define the range of n
def n_range (n : ℝ) : Prop :=
  -1 < n ∧ n < 3

-- Theorem statement
theorem hyperbola_n_range :
  ∀ m n : ℝ,
  (∃ x y : ℝ, hyperbola_equation x y m n) →
  foci_distance m n →
  n_range n :=
sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l1652_165243


namespace NUMINAMATH_CALUDE_total_amount_is_80000_l1652_165277

/-- Represents the problem of dividing money between two investments with different interest rates -/
def MoneyDivisionProblem (total_profit interest_10_amount : ℕ) : Prop :=
  ∃ (total_amount interest_20_amount : ℕ),
    -- Total amount is the sum of both investments
    total_amount = interest_10_amount + interest_20_amount ∧
    -- Profit calculation
    total_profit = (interest_10_amount * 10 / 100) + (interest_20_amount * 20 / 100)

/-- Theorem stating that given the problem conditions, the total amount is 80000 -/
theorem total_amount_is_80000 :
  MoneyDivisionProblem 9000 70000 → ∃ total_amount : ℕ, total_amount = 80000 :=
sorry

end NUMINAMATH_CALUDE_total_amount_is_80000_l1652_165277


namespace NUMINAMATH_CALUDE_power_of_prime_squared_minus_one_l1652_165278

theorem power_of_prime_squared_minus_one (n : ℕ) : 
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ n^2 - 1 = p^k) ↔ n = 2 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_of_prime_squared_minus_one_l1652_165278


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l1652_165222

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (sum_to_180 : a + b + c = 180)

-- Define the problem
theorem triangle_angle_problem (t : Triangle) 
  (h1 : t.a = 70)
  (h2 : t.b = 40)
  (h3 : 180 - t.c = 130) :
  t.c = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l1652_165222


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1652_165219

theorem sum_with_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1652_165219


namespace NUMINAMATH_CALUDE_park_outer_diameter_l1652_165232

/-- Represents the structure of a circular park with concentric regions -/
structure CircularPark where
  fountain_diameter : ℝ
  garden_width : ℝ
  inner_path_width : ℝ
  outer_path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.fountain_diameter + 2 * (park.garden_width + park.inner_path_width + park.outer_path_width)

/-- Theorem stating that for a park with given measurements, the outer boundary diameter is 48 feet -/
theorem park_outer_diameter :
  let park : CircularPark := {
    fountain_diameter := 10,
    garden_width := 12,
    inner_path_width := 3,
    outer_path_width := 4
  }
  outer_boundary_diameter park = 48 := by sorry

end NUMINAMATH_CALUDE_park_outer_diameter_l1652_165232


namespace NUMINAMATH_CALUDE_polygon_diagonals_l1652_165204

theorem polygon_diagonals (n : ℕ) (h : n ≥ 3) :
  (n - 3 ≤ 6) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l1652_165204


namespace NUMINAMATH_CALUDE_audit_options_second_week_l1652_165237

def remaining_OR : ℕ := 13 - 2
def remaining_GTU : ℕ := 15 - 3

theorem audit_options_second_week : 
  (remaining_OR.choose 2) * (remaining_GTU.choose 3) = 12100 := by
  sorry

end NUMINAMATH_CALUDE_audit_options_second_week_l1652_165237


namespace NUMINAMATH_CALUDE_geometric_sequence_s4_l1652_165252

/-- A geometric sequence with partial sums S_n -/
structure GeometricSequence where
  S : ℕ → ℝ
  is_geometric : ∀ n : ℕ, S (n + 2) - S (n + 1) = (S (n + 1) - S n) * (S (n + 1) - S n) / (S n - S (n - 1))

/-- Theorem: In a geometric sequence where S_2 = 7 and S_6 = 91, S_4 = 28 -/
theorem geometric_sequence_s4 (seq : GeometricSequence) 
  (h2 : seq.S 2 = 7) (h6 : seq.S 6 = 91) : seq.S 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_s4_l1652_165252


namespace NUMINAMATH_CALUDE_function_composition_sqrt2_l1652_165205

/-- Given a function f(x) = a * x^2 - √2, where a is a constant,
    if f(f(√2)) = -√2, then a = √2 / 2 -/
theorem function_composition_sqrt2 (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = a * x^2 - Real.sqrt 2) →
  f (f (Real.sqrt 2)) = -Real.sqrt 2 →
  a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_sqrt2_l1652_165205


namespace NUMINAMATH_CALUDE_base2_to_base4_conversion_l1652_165249

def base2_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List (Fin 4) :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

theorem base2_to_base4_conversion :
  decimal_to_base4 (base2_to_decimal [true, false, true, true, false, true, true, true, false]) =
  [1, 1, 2, 3, 2] :=
by sorry

end NUMINAMATH_CALUDE_base2_to_base4_conversion_l1652_165249


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1652_165209

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 + 2*m - 5 = 0 → n^2 + 2*n - 5 = 0 → m^2 + m*n + 2*m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1652_165209


namespace NUMINAMATH_CALUDE_special_op_nine_ten_l1652_165287

-- Define the ⊕ operation
def special_op (A B : ℚ) : ℚ := 1 / (A * B) + 1 / ((A + 1) * (B + 2))

-- State the theorem
theorem special_op_nine_ten :
  special_op 9 10 = 7 / 360 :=
by
  -- The proof goes here
  sorry

-- Additional fact given in the problem
axiom special_op_one_two : special_op 1 2 = 5 / 8

end NUMINAMATH_CALUDE_special_op_nine_ten_l1652_165287


namespace NUMINAMATH_CALUDE_abs_product_plus_four_gt_abs_sum_l1652_165254

def f (x : ℝ) := |x - 1| + |x + 1|

def M : Set ℝ := {x | f x < 4}

theorem abs_product_plus_four_gt_abs_sum {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M) :
  |a * b + 4| > |a + b| := by
  sorry

end NUMINAMATH_CALUDE_abs_product_plus_four_gt_abs_sum_l1652_165254


namespace NUMINAMATH_CALUDE_dhoni_spending_l1652_165230

theorem dhoni_spending (total_earnings : ℝ) (rent_percent dishwasher_percent leftover_percent : ℝ) :
  rent_percent = 25 →
  leftover_percent = 52.5 →
  dishwasher_percent = 100 - rent_percent - leftover_percent →
  (rent_percent - dishwasher_percent) / rent_percent * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_dhoni_spending_l1652_165230


namespace NUMINAMATH_CALUDE_total_fruits_l1652_165210

/-- The number of pieces of fruit in three buckets -/
structure FruitBuckets where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The conditions for the fruit bucket problem -/
def fruit_bucket_conditions (fb : FruitBuckets) : Prop :=
  fb.A = fb.B + 4 ∧ fb.B = fb.C + 3 ∧ fb.C = 9

/-- The theorem stating that the total number of fruits in all buckets is 37 -/
theorem total_fruits (fb : FruitBuckets) 
  (h : fruit_bucket_conditions fb) : fb.A + fb.B + fb.C = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_l1652_165210


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1652_165297

def U : Finset ℕ := {0,1,2,3,4,5,6,7,8,9}
def A : Finset ℕ := {0,1,3,5,8}
def B : Finset ℕ := {2,4,5,6,8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7,9} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1652_165297


namespace NUMINAMATH_CALUDE_quadratic_sum_l1652_165296

/-- Given a quadratic function f(x) = -2x^2 + 16x - 72, prove that when expressed
    in the form a(x+b)^2 + c, the sum of a, b, and c is equal to -46. -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a b c : ℝ), 
    (∀ x, -2 * x^2 + 16 * x - 72 = a * (x + b)^2 + c) ∧
    (a + b + c = -46) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1652_165296


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l1652_165272

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1

/-- Two vectors have the same direction if their corresponding components have the same sign -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 ≥ 0) ∧ (a.2 * b.2 ≥ 0)

theorem collinear_vectors_m_value :
  ∀ (m : ℝ),
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4, m)
  collinear a b → same_direction a b → m = 2 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l1652_165272


namespace NUMINAMATH_CALUDE_comic_collection_problem_l1652_165266

/-- The number of comic books that are in either Andrew's or John's collection, but not both -/
def exclusive_comics (shared comics_andrew comics_john_exclusive : ℕ) : ℕ :=
  (comics_andrew - shared) + comics_john_exclusive

theorem comic_collection_problem (shared comics_andrew comics_john_exclusive : ℕ) 
  (h1 : shared = 15)
  (h2 : comics_andrew = 22)
  (h3 : comics_john_exclusive = 10) :
  exclusive_comics shared comics_andrew comics_john_exclusive = 17 := by
  sorry

end NUMINAMATH_CALUDE_comic_collection_problem_l1652_165266


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l1652_165264

theorem mean_equality_implies_z_value : 
  let mean1 := (8 + 14 + 24) / 3
  let mean2 := (16 + z) / 2
  mean1 = mean2 → z = 44 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l1652_165264


namespace NUMINAMATH_CALUDE_expected_winnings_is_four_thirds_l1652_165288

/-- Represents the faces of the coin -/
inductive Face
  | one
  | two
  | three
  | four

/-- The probability of each face appearing -/
def probability (f : Face) : ℚ :=
  match f with
  | Face.one => 5/12
  | Face.two => 1/3
  | Face.three => 1/6
  | Face.four => 1/12

/-- The winnings associated with each face -/
def winnings (f : Face) : ℤ :=
  match f with
  | Face.one => 2
  | Face.two => 0
  | Face.three => -2
  | Face.four => 10

/-- The expected winnings when tossing the coin -/
def expectedWinnings : ℚ :=
  (probability Face.one * winnings Face.one) +
  (probability Face.two * winnings Face.two) +
  (probability Face.three * winnings Face.three) +
  (probability Face.four * winnings Face.four)

theorem expected_winnings_is_four_thirds :
  expectedWinnings = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_winnings_is_four_thirds_l1652_165288


namespace NUMINAMATH_CALUDE_village_news_spread_l1652_165290

/-- Represents the village and its news spreading dynamics -/
structure Village where
  inhabitants : Finset Nat
  acquaintances : Nat → Finset Nat
  news_spreads : ∀ (n : Nat), n ∈ inhabitants → ∀ (m : Nat), m ∈ acquaintances n → m ∈ inhabitants

/-- A village satisfies the problem conditions -/
def ValidVillage (v : Village) : Prop :=
  v.inhabitants.card = 1000 ∧
  (∀ (news : Nat → Prop) (start : Nat → Prop),
    (∃ (d : Nat), ∀ (n : Nat), n ∈ v.inhabitants → news n))

/-- Represents the spread of news over time -/
def NewsSpread (v : Village) (informed : Finset Nat) (days : Nat) : Finset Nat :=
  sorry

/-- The main theorem to be proved -/
theorem village_news_spread (v : Village) (h : ValidVillage v) :
  ∃ (informed : Finset Nat),
    informed.card = 90 ∧
    ∀ (n : Nat), n ∈ v.inhabitants →
      n ∈ NewsSpread v informed 10 :=
sorry

end NUMINAMATH_CALUDE_village_news_spread_l1652_165290


namespace NUMINAMATH_CALUDE_johns_outfit_cost_l1652_165268

/-- The cost of John's outfit given the cost of his pants and the relative cost of his shirt. -/
theorem johns_outfit_cost (pants_cost : ℝ) (shirt_relative_cost : ℝ) : 
  pants_cost = 50 →
  shirt_relative_cost = 0.6 →
  pants_cost + (pants_cost + pants_cost * shirt_relative_cost) = 130 := by
  sorry

end NUMINAMATH_CALUDE_johns_outfit_cost_l1652_165268


namespace NUMINAMATH_CALUDE_remainder_and_smallest_integer_l1652_165269

theorem remainder_and_smallest_integer (n : ℤ) : n % 20 = 11 →
  ((n % 4 + n % 5 = 4) ∧
   (∀ m : ℤ, m > 50 ∧ m % 20 = 11 → m ≥ 51) ∧
   (51 % 20 = 11)) :=
by sorry

end NUMINAMATH_CALUDE_remainder_and_smallest_integer_l1652_165269


namespace NUMINAMATH_CALUDE_calculate_expression_l1652_165299

theorem calculate_expression : 6 * (1/3 - 1/2) - 3^2 / (-12) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1652_165299


namespace NUMINAMATH_CALUDE_pencil_difference_proof_l1652_165229

def pencil_distribution (total : ℕ) (kept : ℕ) (given_to_manny : ℕ) : Prop :=
  let given_away := total - kept
  let given_to_nilo := given_away - given_to_manny
  given_to_nilo - given_to_manny = 10

theorem pencil_difference_proof :
  pencil_distribution 50 20 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_difference_proof_l1652_165229


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_8_l1652_165235

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) (A : ℝ) : ℝ := 2 - A * 2^(n - 1)

/-- The geometric sequence {a_n} -/
def a (n : ℕ) (A : ℝ) : ℝ := S n A - S (n-1) A

/-- Theorem stating that S_8 equals -510 for the given geometric sequence -/
theorem geometric_sequence_sum_8 (A : ℝ) (h1 : ∀ n : ℕ, n ≥ 1 → S n A = 2 - A * 2^(n - 1))
  (h2 : ∀ k : ℕ, k ≥ 1 → a (k+1) A / a k A = a (k+2) A / a (k+1) A) :
  S 8 A = -510 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_8_l1652_165235


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1652_165256

/-- The coefficient of x^2 in the expansion of (x - 2/x)^6 -/
def coefficient_x_squared : ℕ :=
  let n : ℕ := 6
  let k : ℕ := 2
  ((-2)^k : ℤ) * (Nat.choose n k) |>.natAbs

theorem expansion_coefficient :
  coefficient_x_squared = 60 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1652_165256
