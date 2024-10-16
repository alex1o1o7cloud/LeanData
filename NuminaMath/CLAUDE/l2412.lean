import Mathlib

namespace NUMINAMATH_CALUDE_d₁_d₂_not_divisible_by_3_l2412_241210

-- Define d₁ and d₂ as functions of a
def d₁ (a : ℕ) : ℕ := a^3 + 3^a + a * 3^((a+1)/2)
def d₂ (a : ℕ) : ℕ := a^3 + 3^a - a * 3^((a+1)/2)

-- Define the main theorem
theorem d₁_d₂_not_divisible_by_3 :
  ∀ a : ℕ, 1 ≤ a → a ≤ 251 → ¬(3 ∣ (d₁ a * d₂ a)) :=
by sorry

end NUMINAMATH_CALUDE_d₁_d₂_not_divisible_by_3_l2412_241210


namespace NUMINAMATH_CALUDE_ac_plus_bd_equals_23_l2412_241203

theorem ac_plus_bd_equals_23 
  (a b c d : ℝ) 
  (h1 : a + b + c = 6)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 0)
  (h4 : b + c + d = -9) :
  a * c + b * d = 23 := by
sorry

end NUMINAMATH_CALUDE_ac_plus_bd_equals_23_l2412_241203


namespace NUMINAMATH_CALUDE_min_time_30_seconds_l2412_241226

/-- Represents a person moving along the perimeter of a square -/
structure Person where
  start_position : ℕ  -- Starting vertex (0 = A, 1 = B, 2 = C, 3 = D)
  speed : ℕ           -- Speed in meters per second

/-- Calculates the minimum time for two people to be on the same side of a square -/
def min_time_same_side (side_length : ℕ) (person_a : Person) (person_b : Person) : ℕ :=
  sorry

/-- Theorem stating that the minimum time for the given scenario is 30 seconds -/
theorem min_time_30_seconds (side_length : ℕ) (person_a person_b : Person) :
  side_length = 50 ∧ 
  person_a = { start_position := 0, speed := 5 } ∧
  person_b = { start_position := 2, speed := 3 } →
  min_time_same_side side_length person_a person_b = 30 :=
sorry

end NUMINAMATH_CALUDE_min_time_30_seconds_l2412_241226


namespace NUMINAMATH_CALUDE_three_cards_different_suits_l2412_241290

/-- The number of suits in a standard deck of cards -/
def num_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- The number of cards to choose -/
def cards_to_choose : ℕ := 3

/-- The total number of ways to choose 3 cards from a standard deck of 52 cards,
    where all three cards are of different suits and the order doesn't matter -/
def ways_to_choose : ℕ := num_suits.choose cards_to_choose * cards_per_suit ^ cards_to_choose

theorem three_cards_different_suits :
  ways_to_choose = 8788 := by sorry

end NUMINAMATH_CALUDE_three_cards_different_suits_l2412_241290


namespace NUMINAMATH_CALUDE_convexity_condition_l2412_241259

/-- A plane curve C defined by r = a - b cos θ, where a and b are positive reals and a > b -/
structure PlaneCurve where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- The curve C is convex -/
def is_convex (C : PlaneCurve) : Prop := sorry

/-- Main theorem: C is convex if and only if b/a ≤ 1/2 -/
theorem convexity_condition (C : PlaneCurve) : 
  is_convex C ↔ C.b / C.a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_convexity_condition_l2412_241259


namespace NUMINAMATH_CALUDE_transformation_C_not_equivalent_l2412_241260

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := 2 * x + y = 5
def equation2 (x y : ℝ) : Prop := 3 * x + 4 * y = 7

-- Define the incorrect transformation
def transformation_C (x y : ℝ) : Prop := x = (7 + 4 * y) / 3

-- Theorem stating that the transformation is not equivalent to equation2
theorem transformation_C_not_equivalent :
  ∃ x y : ℝ, equation2 x y ∧ ¬(transformation_C x y) :=
sorry

end NUMINAMATH_CALUDE_transformation_C_not_equivalent_l2412_241260


namespace NUMINAMATH_CALUDE_alice_stops_l2412_241206

/-- Represents the coefficients of a quadratic equation ax² + bx + c = 0 -/
structure QuadraticCoeffs where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Transformation rule for the quadratic coefficients -/
def transform (q : QuadraticCoeffs) : QuadraticCoeffs :=
  { a := q.b + q.c
  , b := q.c + q.a
  , c := q.a + q.b }

/-- Sequence of quadratic coefficients after n transformations -/
def coeff_seq (q₀ : QuadraticCoeffs) : ℕ → QuadraticCoeffs
  | 0 => q₀
  | n + 1 => transform (coeff_seq q₀ n)

/-- Predicate to check if a quadratic equation has real roots -/
def has_real_roots (q : QuadraticCoeffs) : Prop :=
  q.b ^ 2 ≥ 4 * q.a * q.c

/-- Main theorem: Alice will stop after a finite number of moves -/
theorem alice_stops (q₀ : QuadraticCoeffs)
  (h₁ : (q₀.a + q₀.c) * q₀.b > 0) :
  ∃ k : ℕ, ¬(has_real_roots (coeff_seq q₀ k)) := by
  sorry

end NUMINAMATH_CALUDE_alice_stops_l2412_241206


namespace NUMINAMATH_CALUDE_largest_number_l2412_241231

/-- Represents a real number with a repeating decimal expansion -/
structure RepeatingDecimal where
  integerPart : ℤ
  nonRepeatingPart : List ℕ
  repeatingPart : List ℕ

/-- Convert a RepeatingDecimal to a real number -/
def toReal (d : RepeatingDecimal) : ℝ :=
  sorry

/-- Define the five numbers in the problem -/
def a : ℝ := 8.23456
def b : RepeatingDecimal := ⟨8, [2, 3, 4], [5]⟩
def c : RepeatingDecimal := ⟨8, [2, 3], [4, 5]⟩
def d : RepeatingDecimal := ⟨8, [2], [3, 4, 5]⟩
def e : RepeatingDecimal := ⟨8, [], [2, 3, 4, 5]⟩

theorem largest_number :
  toReal b > a ∧
  toReal b > toReal c ∧
  toReal b > toReal d ∧
  toReal b > toReal e :=
sorry

end NUMINAMATH_CALUDE_largest_number_l2412_241231


namespace NUMINAMATH_CALUDE_lines_symmetric_about_y_axis_l2412_241272

/-- Two lines are symmetric about the y-axis if and only if their coefficients satisfy a specific relation -/
theorem lines_symmetric_about_y_axis 
  (a b c p q m : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hm : m ≠ 0) :
  (∃ (k : ℝ), k ≠ 0 ∧ -a = k*p ∧ b = k*q ∧ c = k*m) ↔ 
  (∀ (x y : ℝ), a*x + b*y + c = 0 ↔ p*(-x) + q*y + m = 0) :=
sorry

end NUMINAMATH_CALUDE_lines_symmetric_about_y_axis_l2412_241272


namespace NUMINAMATH_CALUDE_B_power_101_l2412_241283

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 0, 0, 0; 0, 1, 0]

theorem B_power_101 : B^101 = !![0, 0, 0; 0, 0, 0; 0, 0, 1] := by sorry

end NUMINAMATH_CALUDE_B_power_101_l2412_241283


namespace NUMINAMATH_CALUDE_system_solutions_l2412_241216

def equation1 (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

def equation2 (x y : ℝ) : ℝ := 
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|)

theorem system_solutions :
  ∀ (x y : ℝ), 
    (equation1 x = 0 ∧ equation2 x y = 0) ↔ 
    ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2412_241216


namespace NUMINAMATH_CALUDE_problem_statement_l2412_241286

theorem problem_statement (a b : ℕ) (m : ℝ) 
  (h1 : a > 1) 
  (h2 : b > 1) 
  (h3 : a * (b + Real.sin m) = b + Real.cos m) : 
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2412_241286


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l2412_241271

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (new_player1_weight : ℕ) 
  (new_player2_weight : ℕ) 
  (new_average_weight : ℕ) 
  (h1 : original_players = 7)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_average_weight = 78) :
  (original_players * (original_players + 2) * new_average_weight - 
   (original_players * new_player1_weight + original_players * new_player2_weight)) / 
  (original_players * original_players) = 76 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l2412_241271


namespace NUMINAMATH_CALUDE_min_value_of_complex_expression_l2412_241285

theorem min_value_of_complex_expression (z : ℂ) (h : Complex.abs (z - 3 + 3*I) = 3) :
  ∃ (min_val : ℝ), min_val = 19 - 6 * Real.sqrt 2 ∧
    ∀ (w : ℂ), Complex.abs (w - 3 + 3*I) = 3 →
      Complex.abs (w + 2 - I)^2 + Complex.abs (w - 4 + 2*I)^2 ≥ min_val :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_complex_expression_l2412_241285


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l2412_241248

/-- Represents the sum of elements in the nth set of a specific sequence of sets of consecutive integers -/
def S (n : ℕ) : ℕ :=
  let first := 1 + (n * (n - 1)) / 2
  let last := first + n - 1
  n * (first + last) / 2

/-- The theorem stating that the sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l2412_241248


namespace NUMINAMATH_CALUDE_smallest_n_satisfies_conditions_count_non_seven_divisors_l2412_241215

def is_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, x = m^2

def is_perfect_cube (x : ℕ) : Prop := ∃ m : ℕ, x = m^3

def is_perfect_seventh (x : ℕ) : Prop := ∃ m : ℕ, x = m^7

def smallest_n : ℕ := 2^6 * 3^10 * 7^14

theorem smallest_n_satisfies_conditions :
  is_perfect_square (smallest_n / 2) ∧
  is_perfect_cube (smallest_n / 3) ∧
  is_perfect_seventh (smallest_n / 7) := by sorry

theorem count_non_seven_divisors :
  (Finset.filter (fun d => ¬(d % 7 = 0)) (Nat.divisors smallest_n)).card = 77 := by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfies_conditions_count_non_seven_divisors_l2412_241215


namespace NUMINAMATH_CALUDE_prime_sum_special_equation_l2412_241298

theorem prime_sum_special_equation (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → q^5 - 2*p^2 = 1 → p + q = 14 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_special_equation_l2412_241298


namespace NUMINAMATH_CALUDE_restaurant_budget_allocation_l2412_241244

/-- Given a restaurant's budget allocation, prove that the fraction of
    remaining budget spent on food and beverages is 1/4. -/
theorem restaurant_budget_allocation (B : ℝ) (B_pos : B > 0) :
  let rent : ℝ := (1 / 4) * B
  let remaining : ℝ := B - rent
  let food_and_beverages : ℝ := 0.1875 * B
  food_and_beverages / remaining = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_restaurant_budget_allocation_l2412_241244


namespace NUMINAMATH_CALUDE_music_club_ratio_l2412_241291

theorem music_club_ratio :
  ∀ (total girls boys : ℕ) (p_girl p_boy : ℝ),
    total = girls + boys →
    total > 0 →
    p_girl + p_boy = 1 →
    p_girl = (3 / 5 : ℝ) * p_boy →
    (girls : ℝ) / total = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_music_club_ratio_l2412_241291


namespace NUMINAMATH_CALUDE_math_competition_scores_l2412_241294

/-- Represents the scoring system for a math competition. -/
structure ScoringSystem where
  num_questions : ℕ
  correct_points : ℕ
  no_answer_points : ℕ
  wrong_answer_deduction : ℕ

/-- Calculates the number of different possible scores for a given scoring system. -/
def num_different_scores (s : ScoringSystem) : ℕ :=
  sorry

/-- Theorem stating that for the given scoring system, there are 35 different possible scores. -/
theorem math_competition_scores :
  let s : ScoringSystem := {
    num_questions := 10,
    correct_points := 4,
    no_answer_points := 0,
    wrong_answer_deduction := 1
  }
  num_different_scores s = 35 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_scores_l2412_241294


namespace NUMINAMATH_CALUDE_four_digit_divisor_cyclic_iff_abab_l2412_241230

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def cyclic_shift (n : ℕ) : ℕ := 
  let d := n % 10
  let r := n / 10
  d * 1000 + r

def is_divisor_of_cyclic (n : ℕ) : Prop :=
  ∃ k, k * n = cyclic_shift n ∨ k * n = cyclic_shift (cyclic_shift n) ∨ k * n = cyclic_shift (cyclic_shift (cyclic_shift n))

def is_abab_form (n : ℕ) : Prop :=
  ∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ n = a * 1000 + b * 100 + a * 10 + b

theorem four_digit_divisor_cyclic_iff_abab (n : ℕ) :
  is_four_digit n ∧ is_divisor_of_cyclic n ↔ is_abab_form n := by sorry

end NUMINAMATH_CALUDE_four_digit_divisor_cyclic_iff_abab_l2412_241230


namespace NUMINAMATH_CALUDE_farmer_horses_count_l2412_241293

/-- Calculates the number of horses a farmer owns based on hay production and consumption --/
def farmer_horses (last_year_bales : ℕ) (last_year_acres : ℕ) (additional_acres : ℕ) 
                  (bales_per_horse_per_day : ℕ) (remaining_bales : ℕ) : ℕ :=
  let total_acres := last_year_acres + additional_acres
  let bales_per_month := (last_year_bales / last_year_acres) * total_acres
  let feeding_months := 4  -- September to December
  let total_bales := bales_per_month * feeding_months + remaining_bales
  let feeding_days := 122  -- Total days from September 1st to December 31st
  let bales_per_horse := bales_per_horse_per_day * feeding_days
  total_bales / bales_per_horse

/-- Theorem stating the number of horses owned by the farmer --/
theorem farmer_horses_count : 
  farmer_horses 560 5 7 3 12834 = 49 := by
  sorry

end NUMINAMATH_CALUDE_farmer_horses_count_l2412_241293


namespace NUMINAMATH_CALUDE_third_quiz_score_l2412_241256

theorem third_quiz_score (score1 score2 score3 : ℕ) : 
  score1 = 91 → 
  score2 = 92 → 
  (score1 + score2 + score3) / 3 = 91 → 
  score3 = 90 := by
sorry

end NUMINAMATH_CALUDE_third_quiz_score_l2412_241256


namespace NUMINAMATH_CALUDE_function_lower_bound_l2412_241204

open Real

/-- A function satisfying the given inequality for all real x -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x, Real.sqrt (2 * f x) - Real.sqrt (2 * f x - f (2 * x)) ≥ 2

/-- The main theorem to be proved -/
theorem function_lower_bound
  (f : ℝ → ℝ) (h : SatisfiesInequality f) :
  ∀ x, f x ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l2412_241204


namespace NUMINAMATH_CALUDE_match_triangle_formation_l2412_241299

theorem match_triangle_formation (n : ℕ) : 
  (n = 100 → ¬(3 ∣ (n * (n + 1) / 2))) ∧ 
  (n = 99 → (3 ∣ (n * (n + 1) / 2))) := by
  sorry

end NUMINAMATH_CALUDE_match_triangle_formation_l2412_241299


namespace NUMINAMATH_CALUDE_square_perimeter_area_l2412_241221

/-- Theorem: A square with a perimeter of 24 inches has an area of 36 square inches. -/
theorem square_perimeter_area : 
  ∀ (side : ℝ), 
  (4 * side = 24) → (side * side = 36) :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_area_l2412_241221


namespace NUMINAMATH_CALUDE_cube_root_of_four_solution_l2412_241262

theorem cube_root_of_four_solution (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_four_solution_l2412_241262


namespace NUMINAMATH_CALUDE_final_ratio_is_four_to_one_l2412_241297

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents the transformations applied to the rectangle -/
def transform (r : Rectangle) : Rectangle :=
  let r1 := Rectangle.mk (2 * r.length) (r.width / 2)
  let r2 := 
    if 2 * r1.length > r1.width
    then Rectangle.mk (r1.length + 1) (r1.width - 4)
    else Rectangle.mk (r1.length - 4) (r1.width + 1)
  let r3 := 
    if r2.length > r2.width
    then Rectangle.mk r2.length (r2.width - 1)
    else Rectangle.mk r2.length (r2.width - 1)
  r3

/-- The theorem stating that after transformations, the ratio of sides is 4:1 -/
theorem final_ratio_is_four_to_one (r : Rectangle) :
  let final := transform r
  (final.length : ℚ) / final.width = 4 :=
sorry

end NUMINAMATH_CALUDE_final_ratio_is_four_to_one_l2412_241297


namespace NUMINAMATH_CALUDE_fountain_area_l2412_241223

theorem fountain_area (AB DC : ℝ) (h1 : AB = 24) (h2 : DC = 14) : 
  let AD : ℝ := AB / 3
  let R : ℝ := Real.sqrt (AD^2 + DC^2)
  π * R^2 = 260 * π := by sorry

end NUMINAMATH_CALUDE_fountain_area_l2412_241223


namespace NUMINAMATH_CALUDE_circle_and_inscribed_square_l2412_241201

/-- Given a circle with circumference 72π and an inscribed square with vertices touching the circle,
    prove that the radius is 36 and the side length of the square is 36√2. -/
theorem circle_and_inscribed_square (C : ℝ) (r : ℝ) (s : ℝ) :
  C = 72 * Real.pi →  -- Circumference of the circle
  C = 2 * Real.pi * r →  -- Relation between circumference and radius
  s^2 * 2 = (2 * r)^2 →  -- Relation between square side and circle diameter
  r = 36 ∧ s = 36 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_and_inscribed_square_l2412_241201


namespace NUMINAMATH_CALUDE_trucks_distance_l2412_241222

-- Define the speeds of trucks A and B in km/h
def speed_A : ℝ := 54
def speed_B : ℝ := 72

-- Define the time elapsed in seconds
def time_elapsed : ℝ := 30

-- Define the conversion factor from km to meters
def km_to_meters : ℝ := 1000

-- Define the conversion factor from hours to seconds
def hours_to_seconds : ℝ := 3600

-- Theorem statement
theorem trucks_distance :
  let speed_A_mps := speed_A * km_to_meters / hours_to_seconds
  let speed_B_mps := speed_B * km_to_meters / hours_to_seconds
  let distance_A := speed_A_mps * time_elapsed
  let distance_B := speed_B_mps * time_elapsed
  distance_A + distance_B = 1050 :=
by sorry

end NUMINAMATH_CALUDE_trucks_distance_l2412_241222


namespace NUMINAMATH_CALUDE_distribute_five_books_four_bags_l2412_241254

/-- The number of ways to distribute n distinct objects into k identical containers --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 41 ways to distribute 5 books into 4 bags --/
theorem distribute_five_books_four_bags : distribute 5 4 = 41 := by sorry

end NUMINAMATH_CALUDE_distribute_five_books_four_bags_l2412_241254


namespace NUMINAMATH_CALUDE_sum_of_tens_equal_hundred_to_ten_l2412_241269

theorem sum_of_tens_equal_hundred_to_ten (n : ℕ) : 
  (n * 10 = 100^10) → (n = 10^19) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tens_equal_hundred_to_ten_l2412_241269


namespace NUMINAMATH_CALUDE_ab_minus_three_l2412_241239

theorem ab_minus_three (a b : ℤ) (h : a - b = -2) : a - b - 3 = -5 := by
  sorry

end NUMINAMATH_CALUDE_ab_minus_three_l2412_241239


namespace NUMINAMATH_CALUDE_fixed_point_on_quadratic_graph_l2412_241237

/-- The fixed point on the graph of y = 9x^2 + mx - 5m for any real m -/
theorem fixed_point_on_quadratic_graph :
  ∀ (m : ℝ), 9 * (5 : ℝ)^2 + m * 5 - 5 * m = 225 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_quadratic_graph_l2412_241237


namespace NUMINAMATH_CALUDE_interval_sum_l2412_241243

/-- The theorem states that for an interval [a, b] satisfying the given inequality,
    the sum of its endpoints is 12. -/
theorem interval_sum (a b : ℝ) : 
  (∀ x ∈ Set.Icc a b, |3*x - 80| ≤ |2*x - 105|) → a + b = 12 := by
  sorry

#check interval_sum

end NUMINAMATH_CALUDE_interval_sum_l2412_241243


namespace NUMINAMATH_CALUDE_min_constant_for_sqrt_inequality_l2412_241253

theorem min_constant_for_sqrt_inequality :
  (∃ (a : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ∧
  (∀ (a : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) → a ≥ Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_min_constant_for_sqrt_inequality_l2412_241253


namespace NUMINAMATH_CALUDE_ron_book_picks_l2412_241275

/-- Represents the number of times a person gets to pick a book in a year -/
def picks_per_year (total_members : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weeks_per_year / total_members

/-- The book club scenario -/
theorem ron_book_picks :
  let couples := 3
  let singles := 5
  let ron_and_wife := 2
  let total_members := couples * 2 + singles + ron_and_wife
  let weeks_per_year := 52
  picks_per_year total_members weeks_per_year = 4 := by
sorry

end NUMINAMATH_CALUDE_ron_book_picks_l2412_241275


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l2412_241251

-- Define the point
def point : ℝ × ℝ := (12, -5)

-- Theorem statement
theorem distance_from_origin_to_point :
  Real.sqrt ((point.1 - 0)^2 + (point.2 - 0)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l2412_241251


namespace NUMINAMATH_CALUDE_fraction_multiplication_addition_l2412_241288

theorem fraction_multiplication_addition : (1/3 : ℚ) * (2/5 : ℚ) + (1/4 : ℚ) = 23/60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_addition_l2412_241288


namespace NUMINAMATH_CALUDE_simplify_fraction_l2412_241219

theorem simplify_fraction : (18 : ℚ) * (8 / 12) * (1 / 9) * 4 = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2412_241219


namespace NUMINAMATH_CALUDE_two_pipes_fill_time_l2412_241277

def fill_time (num_pipes : ℕ) (time : ℝ) : Prop :=
  num_pipes > 0 ∧ time > 0 ∧ num_pipes * time = 36

theorem two_pipes_fill_time :
  fill_time 3 12 → fill_time 2 18 :=
by
  sorry

end NUMINAMATH_CALUDE_two_pipes_fill_time_l2412_241277


namespace NUMINAMATH_CALUDE_grunters_win_all_games_l2412_241274

/-- The number of games played between the Grunters and the Screamers -/
def num_games : ℕ := 6

/-- The probability of the Grunters winning a game that doesn't go to overtime -/
def p_win_no_overtime : ℝ := 0.6

/-- The probability of the Grunters winning a game that goes to overtime -/
def p_win_overtime : ℝ := 0.5

/-- The probability of a game going to overtime -/
def p_overtime : ℝ := 0.1

/-- The theorem stating the probability of the Grunters winning all games -/
theorem grunters_win_all_games : 
  (((1 - p_overtime) * p_win_no_overtime + p_overtime * p_win_overtime) ^ num_games : ℝ) = 
  (823543 : ℝ) / 10000000 := by sorry

end NUMINAMATH_CALUDE_grunters_win_all_games_l2412_241274


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_l2412_241233

theorem sqrt_six_times_sqrt_two : Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_l2412_241233


namespace NUMINAMATH_CALUDE_problem_solution_l2412_241292

theorem problem_solution (x y : ℝ) : 
  x + y = 150 ∧ 1.20 * y - 0.80 * x = 0.75 * (x + y) → x = 33.75 ∧ y = 116.25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2412_241292


namespace NUMINAMATH_CALUDE_remainder_problem_l2412_241214

theorem remainder_problem : (123456789012 : ℕ) % 252 = 228 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2412_241214


namespace NUMINAMATH_CALUDE_tan_sum_three_angles_l2412_241228

theorem tan_sum_three_angles (α β γ : ℝ) : 
  Real.tan (α + β + γ) = (Real.tan α + Real.tan β + Real.tan γ - Real.tan α * Real.tan β * Real.tan γ) / 
                         (1 - Real.tan α * Real.tan β - Real.tan β * Real.tan γ - Real.tan γ * Real.tan α) :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_three_angles_l2412_241228


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_condition_l2412_241261

theorem quadratic_complex_roots_condition (a : ℝ) :
  (∃ x : ℂ, x^2 + a*x + 1 = 0 ∧ x.im ≠ 0) →
  a < 2 ∧
  ¬(a < 2 → ∃ x : ℂ, x^2 + a*x + 1 = 0 ∧ x.im ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_condition_l2412_241261


namespace NUMINAMATH_CALUDE_equation_solution_l2412_241218

theorem equation_solution :
  ∃ x : ℚ, -8 * (2 - x)^3 = 27 ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2412_241218


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_arithmetic_sum_l2412_241270

/-- The sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
def arithmetic_sum (n : ℕ) (a d : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

/-- The sum of the first 12 terms of an arithmetic sequence with first term a and common difference d -/
def sum_12_terms (a d : ℤ) : ℤ := arithmetic_sum 12 a d

theorem greatest_common_divisor_of_arithmetic_sum :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (a d : ℕ), (sum_12_terms a d).natAbs % k = 0) ∧
  (∀ (m : ℕ), m > k → ∃ (a d : ℕ), (sum_12_terms a d).natAbs % m ≠ 0) ∧
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_arithmetic_sum_l2412_241270


namespace NUMINAMATH_CALUDE_number_of_students_in_section_B_l2412_241276

/-- Given a class with two sections A and B, prove the number of students in section B -/
theorem number_of_students_in_section_B 
  (students_A : ℕ) 
  (avg_weight_A : ℚ) 
  (avg_weight_B : ℚ) 
  (avg_weight_total : ℚ) 
  (h1 : students_A = 50)
  (h2 : avg_weight_A = 50)
  (h3 : avg_weight_B = 70)
  (h4 : avg_weight_total = 61.67) :
  ∃ (students_B : ℕ), students_B = 70 := by
sorry

end NUMINAMATH_CALUDE_number_of_students_in_section_B_l2412_241276


namespace NUMINAMATH_CALUDE_hexagon_side_sum_l2412_241246

/-- Given a hexagon PQRSTU with the following properties:
  * The area of PQRSTU is 68
  * PQ = 10
  * QR = 7
  * TU = 6
  Prove that RS + ST = 3 -/
theorem hexagon_side_sum (PQRSTU : Set ℝ × ℝ) (area : ℝ) (PQ QR TU : ℝ) :
  area = 68 → PQ = 10 → QR = 7 → TU = 6 →
  ∃ (RS ST : ℝ), RS + ST = 3 := by
  sorry

#check hexagon_side_sum

end NUMINAMATH_CALUDE_hexagon_side_sum_l2412_241246


namespace NUMINAMATH_CALUDE_betty_garden_ratio_l2412_241212

/-- Represents a herb garden with oregano and basil plants -/
structure HerbGarden where
  total_plants : ℕ
  basil_plants : ℕ
  oregano_plants : ℕ
  total_eq : total_plants = oregano_plants + basil_plants

/-- The ratio of oregano to basil plants in Betty's garden is 12:5 -/
theorem betty_garden_ratio (garden : HerbGarden) 
    (h1 : garden.total_plants = 17)
    (h2 : garden.basil_plants = 5) :
    garden.oregano_plants / garden.basil_plants = 12 / 5 := by
  sorry

#check betty_garden_ratio

end NUMINAMATH_CALUDE_betty_garden_ratio_l2412_241212


namespace NUMINAMATH_CALUDE_calculate_expression_l2412_241281

theorem calculate_expression : (3.242 * 12) / 100 = 0.38904 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2412_241281


namespace NUMINAMATH_CALUDE_sin_sum_of_angles_l2412_241289

theorem sin_sum_of_angles (θ φ : ℝ) 
  (h1 : Complex.exp (θ * Complex.I) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I)
  (h2 : Complex.exp (φ * Complex.I) = -(5/13 : ℂ) + (12/13 : ℂ) * Complex.I) : 
  Real.sin (θ + φ) = 33/65 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_of_angles_l2412_241289


namespace NUMINAMATH_CALUDE_largest_a_for_fibonacci_sum_l2412_241211

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Property: Fₐ, Fᵦ, Fᶜ form an increasing arithmetic sequence -/
def is_arithmetic_seq (a b c : ℕ) : Prop :=
  fib b - fib a = fib c - fib b ∧ fib a < fib b ∧ fib b < fib c

/-- Main theorem -/
theorem largest_a_for_fibonacci_sum (a b c : ℕ) :
  is_arithmetic_seq a b c →
  a + b + c ≤ 3000 →
  a ≤ 998 :=
by sorry

end NUMINAMATH_CALUDE_largest_a_for_fibonacci_sum_l2412_241211


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l2412_241278

theorem rectangular_box_volume (l w h : ℝ) (area1 area2 area3 : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  area1 = l * w →
  area2 = w * h →
  area3 = l * h →
  area1 = 30 →
  area2 = 18 →
  area3 = 10 →
  l * w * h = 90 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l2412_241278


namespace NUMINAMATH_CALUDE_cube_sum_integer_l2412_241247

theorem cube_sum_integer (a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℤ, (a + 1/a : ℝ) = k → ∃ m : ℤ, (a^3 + 1/a^3 : ℝ) = m := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_integer_l2412_241247


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_3_area_is_three_sqrt_three_over_two_l2412_241232

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 + t.b * t.c

-- Theorem 1
theorem angle_A_is_pi_over_3 (t : Triangle) (h : satisfiesCondition t) :
  t.A = π / 3 := by sorry

-- Theorem 2
theorem area_is_three_sqrt_three_over_two (t : Triangle) 
  (h1 : satisfiesCondition t) (h2 : t.a = Real.sqrt 7) (h3 : t.b = 2) :
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_3_area_is_three_sqrt_three_over_two_l2412_241232


namespace NUMINAMATH_CALUDE_solve_sales_problem_l2412_241208

def sales_problem (sales1 sales2 sales4 sales5 desired_average : ℕ) : Prop :=
  let total_months : ℕ := 5
  let known_sales : ℕ := sales1 + sales2 + sales4 + sales5
  let total_desired : ℕ := desired_average * total_months
  let sales3 : ℕ := total_desired - known_sales
  sales3 = 7570 ∧ 
  (sales1 + sales2 + sales3 + sales4 + sales5) / total_months = desired_average

theorem solve_sales_problem : 
  sales_problem 5420 5660 6350 6500 6300 := by
  sorry

end NUMINAMATH_CALUDE_solve_sales_problem_l2412_241208


namespace NUMINAMATH_CALUDE_solution_count_3x_4y_815_l2412_241255

theorem solution_count_3x_4y_815 : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 4 * p.2 = 815 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 816) (Finset.range 816))).card = 68 :=
by sorry

end NUMINAMATH_CALUDE_solution_count_3x_4y_815_l2412_241255


namespace NUMINAMATH_CALUDE_point_on_x_axis_l2412_241264

/-- Given a point A with coordinates (m+1, 2m-4) that is moved up by 2 units
    and lands on the x-axis, prove that m = 1. -/
theorem point_on_x_axis (m : ℝ) : 
  let initial_y := 2*m - 4
  let moved_y := initial_y + 2
  moved_y = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l2412_241264


namespace NUMINAMATH_CALUDE_trackball_mice_count_l2412_241296

theorem trackball_mice_count (total : ℕ) (wireless_ratio : ℚ) (optical_ratio : ℚ) :
  total = 80 →
  wireless_ratio = 1/2 →
  optical_ratio = 1/4 →
  (wireless_ratio + optical_ratio + (1 - wireless_ratio - optical_ratio) : ℚ) = 1 →
  ↑total * (1 - wireless_ratio - optical_ratio) = 20 :=
by sorry

end NUMINAMATH_CALUDE_trackball_mice_count_l2412_241296


namespace NUMINAMATH_CALUDE_fraction_division_result_l2412_241266

theorem fraction_division_result : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_result_l2412_241266


namespace NUMINAMATH_CALUDE_abc_inequality_l2412_241295

theorem abc_inequality (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a + b + c = 6) 
  (h4 : a * b + b * c + a * c = 9) : 
  0 < a ∧ a < 1 ∧ 1 < b ∧ b < 3 ∧ 3 < c ∧ c < 4 :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l2412_241295


namespace NUMINAMATH_CALUDE_g_72_value_l2412_241202

-- Define the properties of function g
def PositiveInteger (n : ℕ) : Prop := n > 0

def g_properties (g : ℕ → ℕ) : Prop :=
  (∀ n, PositiveInteger n → PositiveInteger (g n)) ∧
  (∀ n, PositiveInteger n → g (n + 1) > g n) ∧
  (∀ m n, PositiveInteger m → PositiveInteger n → g (m * n) = g m * g n) ∧
  (∀ m n, m ≠ n → m^n = n^m → (g m = 2*n ∨ g n = 2*m))

-- Theorem statement
theorem g_72_value (g : ℕ → ℕ) (h : g_properties g) : g 72 = 294912 := by
  sorry

end NUMINAMATH_CALUDE_g_72_value_l2412_241202


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2412_241268

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2412_241268


namespace NUMINAMATH_CALUDE_complement_intersection_empty_l2412_241236

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {2, 4, 5}

theorem complement_intersection_empty :
  (U \ A) ∩ (U \ B) = ∅ := by sorry

end NUMINAMATH_CALUDE_complement_intersection_empty_l2412_241236


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l2412_241242

theorem number_puzzle_solution :
  ∃ x : ℚ, x^2 + 100 = (x - 12)^2 ∧ x = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l2412_241242


namespace NUMINAMATH_CALUDE_parabola_hyperbola_shared_focus_l2412_241229

/-- The value of p for which the focus of the parabola y^2 = 2px (p > 0) 
    is also a focus of the hyperbola x^2 - y^2 = 8 -/
theorem parabola_hyperbola_shared_focus (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2 - y^2 = 8 ∧ 
    ((x - p)^2 + y^2 = p^2 ∨ (x + p)^2 + y^2 = p^2)) → 
  p = 8 := by
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_shared_focus_l2412_241229


namespace NUMINAMATH_CALUDE_tuesday_max_hours_l2412_241235

/-- Represents the days of the week from Monday to Friday -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Returns the number of hours Gabe spent riding his bike on a given day -/
def hours_spent (day : Weekday) : ℕ :=
  match day with
  | Weekday.Monday => 3
  | Weekday.Tuesday => 4
  | Weekday.Wednesday => 2
  | Weekday.Thursday => 3
  | Weekday.Friday => 1

/-- Theorem: Tuesday is the day when Gabe spent the greatest number of hours riding his bike -/
theorem tuesday_max_hours :
  ∀ (day : Weekday), hours_spent Weekday.Tuesday ≥ hours_spent day :=
by sorry

end NUMINAMATH_CALUDE_tuesday_max_hours_l2412_241235


namespace NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l2412_241209

theorem factorization_of_2a_squared_minus_8 (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l2412_241209


namespace NUMINAMATH_CALUDE_power_sum_equality_l2412_241282

theorem power_sum_equality (x y : ℕ+) :
  x^(y:ℕ) + y^(x:ℕ) = x^(x:ℕ) + y^(y:ℕ) ↔ x = y :=
sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2412_241282


namespace NUMINAMATH_CALUDE_ball_max_height_l2412_241280

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 45

/-- Theorem stating the maximum height reached by the ball -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 69.5 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l2412_241280


namespace NUMINAMATH_CALUDE_correct_distribution_l2412_241224

/-- Represents the amount of logs contributed by each person -/
structure Contribution where
  troykina : ℕ
  pyatorkina : ℕ
  bestoplivny : ℕ

/-- Represents the payment made by Bestoplivny in kopecks -/
def bestoplivny_payment : ℕ := 80

/-- Calculates the fair distribution of the payment -/
def calculate_distribution (c : Contribution) : ℕ × ℕ := sorry

/-- Theorem stating the correct distribution of the payment -/
theorem correct_distribution (c : Contribution) 
  (h1 : c.troykina = 3)
  (h2 : c.pyatorkina = 5)
  (h3 : c.bestoplivny = 0) :
  calculate_distribution c = (10, 70) := by sorry

end NUMINAMATH_CALUDE_correct_distribution_l2412_241224


namespace NUMINAMATH_CALUDE_pet_store_profit_l2412_241265

-- Define Brandon's selling price
def brandon_price : ℕ := 100

-- Define the pet store's pricing strategy
def pet_store_price (brandon_price : ℕ) : ℕ := 3 * brandon_price + 5

-- Define the profit calculation
def profit (selling_price cost_price : ℕ) : ℕ := selling_price - cost_price

-- Theorem to prove
theorem pet_store_profit :
  profit (pet_store_price brandon_price) brandon_price = 205 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_profit_l2412_241265


namespace NUMINAMATH_CALUDE_max_median_amount_l2412_241273

/-- Represents the initial amounts of money for each person -/
def initial_amounts : List ℕ := [28, 72, 98]

/-- The total amount of money after pooling -/
def total_amount : ℕ := initial_amounts.sum

/-- The number of people -/
def num_people : ℕ := initial_amounts.length

theorem max_median_amount :
  ∃ (distribution : List ℕ),
    distribution.length = num_people ∧
    distribution.sum = total_amount ∧
    (∃ (median : ℕ), median ∈ distribution ∧ 
      (distribution.filter (λ x => x ≤ median)).length ≥ num_people / 2 ∧
      (distribution.filter (λ x => x ≥ median)).length ≥ num_people / 2) ∧
    (∀ (other_distribution : List ℕ),
      other_distribution.length = num_people →
      other_distribution.sum = total_amount →
      (∃ (other_median : ℕ), other_median ∈ other_distribution ∧ 
        (other_distribution.filter (λ x => x ≤ other_median)).length ≥ num_people / 2 ∧
        (other_distribution.filter (λ x => x ≥ other_median)).length ≥ num_people / 2) →
      ∃ (median : ℕ), median ∈ distribution ∧ 
        (distribution.filter (λ x => x ≤ median)).length ≥ num_people / 2 ∧
        (distribution.filter (λ x => x ≥ median)).length ≥ num_people / 2 ∧
        median ≥ other_median) ∧
    (∃ (median : ℕ), median ∈ distribution ∧ 
      (distribution.filter (λ x => x ≤ median)).length ≥ num_people / 2 ∧
      (distribution.filter (λ x => x ≥ median)).length ≥ num_people / 2 ∧
      median = 196) := by
  sorry


end NUMINAMATH_CALUDE_max_median_amount_l2412_241273


namespace NUMINAMATH_CALUDE_function_value_at_two_l2412_241240

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = x^2 + y^2

/-- The main theorem stating that g(2) = 5 for any function satisfying the functional equation -/
theorem function_value_at_two (g : ℝ → ℝ) (h : FunctionalEquation g) : g 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l2412_241240


namespace NUMINAMATH_CALUDE_mahi_share_l2412_241250

structure Friend where
  name : String
  age : ℕ
  distance : ℕ
  removed_amount : ℚ
  ratio : ℕ

def total_amount : ℚ := 2200

def friends : List Friend := [
  ⟨"Neha", 25, 5, 5, 2⟩,
  ⟨"Sabi", 32, 8, 8, 8⟩,
  ⟨"Mahi", 30, 7, 4, 6⟩,
  ⟨"Ravi", 28, 10, 6, 4⟩,
  ⟨"Priya", 35, 4, 10, 10⟩
]

def distance_bonus : ℚ := 10

theorem mahi_share (mahi : Friend) 
  (h1 : mahi ∈ friends)
  (h2 : mahi.name = "Mahi")
  (h3 : ∀ f : Friend, f ∈ friends → 
    f.age * (mahi.ratio * (total_amount - (friends.map Friend.removed_amount).sum) / (friends.map Friend.ratio).sum + mahi.removed_amount + mahi.distance * distance_bonus) = 
    mahi.age * (f.ratio * (total_amount - (friends.map Friend.removed_amount).sum) / (friends.map Friend.ratio).sum + f.removed_amount + f.distance * distance_bonus)) :
  mahi.ratio * (total_amount - (friends.map Friend.removed_amount).sum) / (friends.map Friend.ratio).sum + mahi.removed_amount + mahi.distance * distance_bonus = 507.38 := by
  sorry

end NUMINAMATH_CALUDE_mahi_share_l2412_241250


namespace NUMINAMATH_CALUDE_min_value_expression_l2412_241200

theorem min_value_expression (a θ : ℝ) : 
  (a - 2 * Real.cos θ)^2 + (a - 5 * Real.sqrt 2 - 2 * Real.sin θ)^2 ≥ 9 ∧
  ∃ a θ : ℝ, (a - 2 * Real.cos θ)^2 + (a - 5 * Real.sqrt 2 - 2 * Real.sin θ)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2412_241200


namespace NUMINAMATH_CALUDE_abs_neg_three_equals_three_l2412_241287

theorem abs_neg_three_equals_three : abs (-3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_equals_three_l2412_241287


namespace NUMINAMATH_CALUDE_alternating_seating_card_sum_l2412_241279

theorem alternating_seating_card_sum (n : ℕ) (h : n ≥ 3) :
  ∃ (m : ℕ),
    (∀ (i : ℕ) (hi : i ≤ n), ∃ (b : ℕ), b ≤ n ∧ b = i) ∧  -- Boys' cards
    (∀ (j : ℕ) (hj : n < j ∧ j ≤ 2*n), ∃ (g : ℕ), n < g ∧ g ≤ 2*n ∧ g = j) ∧  -- Girls' cards
    (∀ (k : ℕ) (hk : k ≤ n),
      ∃ (b g₁ g₂ : ℕ),
        b ≤ n ∧ n < g₁ ∧ g₁ ≤ 2*n ∧ n < g₂ ∧ g₂ ≤ 2*n ∧
        b + g₁ + g₂ = m) ↔
  Odd n :=
by sorry

end NUMINAMATH_CALUDE_alternating_seating_card_sum_l2412_241279


namespace NUMINAMATH_CALUDE_january_oil_bill_l2412_241205

theorem january_oil_bill (january february : ℝ) 
  (h1 : february / january = 5 / 4)
  (h2 : (february + 30) / january = 3 / 2) :
  january = 120 := by
sorry

end NUMINAMATH_CALUDE_january_oil_bill_l2412_241205


namespace NUMINAMATH_CALUDE_xoxoxox_probability_l2412_241220

def total_tiles : ℕ := 7
def x_tiles : ℕ := 4
def o_tiles : ℕ := 3

theorem xoxoxox_probability :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = (1 : ℚ) / 35 :=
sorry

end NUMINAMATH_CALUDE_xoxoxox_probability_l2412_241220


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l2412_241257

theorem final_sum_after_operations (S a b : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 7) = 3 * S + 36 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l2412_241257


namespace NUMINAMATH_CALUDE_average_weight_increase_l2412_241213

/-- Proves that replacing a sailor weighing 56 kg with a sailor weighing 64 kg
    in a group of 8 sailors increases the average weight by 1 kg. -/
theorem average_weight_increase (initial_average : ℝ) : 
  let total_weight := 8 * initial_average
  let new_total_weight := total_weight - 56 + 64
  let new_average := new_total_weight / 8
  new_average - initial_average = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2412_241213


namespace NUMINAMATH_CALUDE_problem_statement_l2412_241217

theorem problem_statement : (2112 - 2021)^2 / 169 = 49 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2412_241217


namespace NUMINAMATH_CALUDE_geometric_sum_equals_5592404_l2412_241207

/-- The sum of a geometric series with 11 terms, first term 4, and common ratio 4 -/
def geometricSum : ℕ :=
  4 * (1 - 4^11) / (1 - 4)

/-- Theorem stating that the geometric sum is equal to 5592404 -/
theorem geometric_sum_equals_5592404 : geometricSum = 5592404 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_equals_5592404_l2412_241207


namespace NUMINAMATH_CALUDE_bowling_ball_surface_area_l2412_241263

/-- The surface area of a sphere with diameter 9 inches is 81π square inches. -/
theorem bowling_ball_surface_area :
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 81 * Real.pi := by sorry

end NUMINAMATH_CALUDE_bowling_ball_surface_area_l2412_241263


namespace NUMINAMATH_CALUDE_triangle_ratio_specific_l2412_241234

noncomputable def triangle_ratio (BC AC : ℝ) (angle_C : ℝ) : ℝ :=
  let AB := Real.sqrt (BC^2 + AC^2 - 2*BC*AC*(Real.cos angle_C))
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  let AD := 2 * area / BC
  let BD := Real.sqrt (BC^2 - AD^2)
  let AH := AD - BD / 2
  let HD := BD / 2
  AH / HD

theorem triangle_ratio_specific : 
  triangle_ratio 6 (3 * Real.sqrt 3) (π / 4) = (2 * Real.sqrt 6 - 4) / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_specific_l2412_241234


namespace NUMINAMATH_CALUDE_complex_number_properties_l2412_241238

open Complex

theorem complex_number_properties (z : ℂ) (h : z = 2 - 5*I) : 
  z.im = -5 ∧ abs (z + I) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2412_241238


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_comparison_l2412_241245

theorem geometric_arithmetic_sequence_comparison 
  (a b : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) 
  (h_arith : ∀ n : ℕ, ∃ d : ℝ, b (n + 1) = b n + d)
  (h_pos : a 1 > 0)
  (h_eq1 : a 1 = b 1)
  (h_eq3 : a 3 = b 3)
  (h_neq : a 1 ≠ a 3) :
  a 5 > b 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_comparison_l2412_241245


namespace NUMINAMATH_CALUDE_fraction_transformation_l2412_241225

theorem fraction_transformation (x : ℚ) : 
  (3 + 2*x) / (4 + 3*x) = 5/9 → x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2412_241225


namespace NUMINAMATH_CALUDE_regular_milk_students_l2412_241249

/-- Proof that the number of students who selected regular milk is 3 -/
theorem regular_milk_students (chocolate_milk : ℕ) (strawberry_milk : ℕ) (total_milk : ℕ) :
  chocolate_milk = 2 →
  strawberry_milk = 15 →
  total_milk = 20 →
  total_milk - (chocolate_milk + strawberry_milk) = 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_milk_students_l2412_241249


namespace NUMINAMATH_CALUDE_fraction_equality_l2412_241241

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hxy : x ≠ y) :
  (x * y) / (x^2 - x * y) = y / (x - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2412_241241


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l2412_241267

/-- Given a class with four more girls than boys and 30 total students, 
    prove the ratio of girls to boys is 17/13 -/
theorem girls_to_boys_ratio (g b : ℕ) : 
  g = b + 4 → g + b = 30 → g / b = 17 / 13 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l2412_241267


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2412_241258

-- Define the inequality
def inequality (x : ℝ) : Prop := 4 * x - 5 < 3

-- Define the solution set
def solution_set : Set ℝ := {x | x < 2}

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2412_241258


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l2412_241252

theorem integer_fraction_characterization (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℤ) = k * (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1)) ↔
  (∃ l : ℕ+, (a = 2 * l ∧ b = 1) ∨
             (a = l ∧ b = 2 * l) ∨
             (a = 8 * l.val ^ 4 - l ∧ b = 2 * l)) :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l2412_241252


namespace NUMINAMATH_CALUDE_choose_two_cooks_from_eight_l2412_241284

theorem choose_two_cooks_from_eight (n : ℕ) (k : ℕ) :
  n = 8 ∧ k = 2 → Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_cooks_from_eight_l2412_241284


namespace NUMINAMATH_CALUDE_subtract_largest_3digit_from_smallest_5digit_l2412_241227

def largest_3digit : ℕ := 999
def smallest_5digit : ℕ := 10000

theorem subtract_largest_3digit_from_smallest_5digit :
  smallest_5digit - largest_3digit = 9001 := by
  sorry

end NUMINAMATH_CALUDE_subtract_largest_3digit_from_smallest_5digit_l2412_241227
