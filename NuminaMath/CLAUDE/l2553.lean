import Mathlib

namespace NUMINAMATH_CALUDE_combined_value_of_a_and_b_l2553_255334

/-- Given that 0.5% of a equals 95 paise and b is three times a minus 50,
    prove that the combined value of a and b is 710 rupees. -/
theorem combined_value_of_a_and_b (a b : ℝ) 
  (h1 : 0.005 * a = 95 / 100)  -- 0.5% of a equals 95 paise
  (h2 : b = 3 * a - 50)        -- b is three times a minus 50
  : a + b = 710 := by sorry

end NUMINAMATH_CALUDE_combined_value_of_a_and_b_l2553_255334


namespace NUMINAMATH_CALUDE_exists_counterexample_l2553_255315

/-- A function is strictly monotonically increasing -/
def StrictlyIncreasing (f : ℚ → ℚ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The range of a function is the entire set of rationals -/
def SurjectiveOnRationals (f : ℚ → ℚ) : Prop :=
  ∀ y, ∃ x, f x = y

/-- The main theorem -/
theorem exists_counterexample : ∃ (f g : ℚ → ℚ),
  StrictlyIncreasing f ∧ StrictlyIncreasing g ∧
  SurjectiveOnRationals f ∧ SurjectiveOnRationals g ∧
  ¬SurjectiveOnRationals (λ x => f x + g x) := by
  sorry

end NUMINAMATH_CALUDE_exists_counterexample_l2553_255315


namespace NUMINAMATH_CALUDE_simple_pairs_l2553_255389

theorem simple_pairs (n : ℕ) (h : n > 3) :
  ∃ (p₁ p₂ : ℕ), Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Odd p₁ ∧ Odd p₂ ∧ (p₂ ∣ (2 * n - p₁)) :=
sorry

end NUMINAMATH_CALUDE_simple_pairs_l2553_255389


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l2553_255356

theorem sufficient_condition_for_inequality (a : ℝ) :
  0 < a ∧ a < (1/5) → (1/a) > 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l2553_255356


namespace NUMINAMATH_CALUDE_union_equals_reals_implies_a_is_negative_one_l2553_255327

-- Define the sets S and P
def S : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 2}
def P (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem union_equals_reals_implies_a_is_negative_one (a : ℝ) :
  S ∪ P a = Set.univ → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_reals_implies_a_is_negative_one_l2553_255327


namespace NUMINAMATH_CALUDE_rectilinear_polygon_odd_area_l2553_255398

/-- A rectilinear polygon with integer vertex coordinates and odd side lengths -/
structure RectilinearPolygon where
  vertices : List (Int × Int)
  sides_parallel_to_axes : Bool
  all_sides_odd_length : Bool

/-- The area of a rectilinear polygon -/
noncomputable def area (p : RectilinearPolygon) : ℝ :=
  sorry

/-- A predicate to check if a number is odd -/
def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem rectilinear_polygon_odd_area
  (p : RectilinearPolygon)
  (h_sides : p.vertices.length = 100)
  (h_parallel : p.sides_parallel_to_axes = true)
  (h_odd_sides : p.all_sides_odd_length = true) :
  is_odd (Int.floor (area p)) :=
sorry

end NUMINAMATH_CALUDE_rectilinear_polygon_odd_area_l2553_255398


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l2553_255392

theorem no_function_satisfies_conditions :
  ¬∃ (f : ℚ → ℝ),
    (f 0 = 0) ∧
    (∀ a : ℚ, a ≠ 0 → f a > 0) ∧
    (∀ x y : ℚ, f (x + y) = f x * f y) ∧
    (∀ x y : ℚ, x ≠ 0 → y ≠ 0 → f (x + y) ≤ max (f x) (f y)) ∧
    (∃ x : ℤ, f x ≠ 1) ∧
    (∀ n : ℕ, n > 0 → ∀ x : ℤ, f (1 + x + x^2 + (x^n - 1) / (x - 1)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l2553_255392


namespace NUMINAMATH_CALUDE_probability_of_selecting_letter_from_word_l2553_255332

/-- The number of characters in the extended alphabet -/
def alphabet_size : ℕ := 30

/-- The word from which we're checking letters -/
def word : String := "MATHEMATICS"

/-- The number of unique letters in the word -/
def unique_letters : ℕ := (word.toList.eraseDups).length

/-- The probability of selecting a letter from the word -/
def probability : ℚ := unique_letters / alphabet_size

theorem probability_of_selecting_letter_from_word :
  probability = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_probability_of_selecting_letter_from_word_l2553_255332


namespace NUMINAMATH_CALUDE_total_fireworks_count_l2553_255380

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of boxes Cherie has -/
def cherie_boxes : ℕ := 1

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box) + 
  cherie_boxes * (cherie_sparklers + cherie_whistlers)

theorem total_fireworks_count : total_fireworks = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_fireworks_count_l2553_255380


namespace NUMINAMATH_CALUDE_range_of_m_l2553_255347

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m^2 + 3*m - 3

/-- Proposition p: The minimum value of f(x) is less than 0 -/
def p (m : ℝ) : Prop := ∃ x, f m x < 0

/-- Proposition q: The equation represents an ellipse with foci on the x-axis -/
def q (m : ℝ) : Prop := 5*m - 1 > 0 ∧ m - 2 < 0 ∧ 5*m - 1 > -(m - 2)

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) (h1 : ¬(p m ∨ q m)) (h2 : ¬(p m ∧ q m)) : 
  m ≤ -4 ∨ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2553_255347


namespace NUMINAMATH_CALUDE_linear_function_through_origin_l2553_255352

theorem linear_function_through_origin (k : ℝ) : 
  (∀ x y : ℝ, y = (k - 2) * x + (k^2 - 4)) →  -- Definition of the linear function
  ((0 : ℝ) = (k - 2) * (0 : ℝ) + (k^2 - 4)) →  -- The function passes through the origin
  (k - 2 ≠ 0) →  -- Ensure the function remains linear
  k = -2 := by sorry

end NUMINAMATH_CALUDE_linear_function_through_origin_l2553_255352


namespace NUMINAMATH_CALUDE_hiking_rate_ratio_l2553_255375

/-- Proves that the ratio of the rate down to the rate up is 1.5 given the hiking conditions -/
theorem hiking_rate_ratio 
  (time_equal : ℝ) -- The time for both routes is the same
  (rate_up : ℝ) -- The rate up the mountain
  (time_up : ℝ) -- The time to go up the mountain
  (distance_down : ℝ) -- The distance of the route down the mountain
  (h_rate_up : rate_up = 5) -- The rate up is 5 miles per day
  (h_time_up : time_up = 2) -- It takes 2 days to go up
  (h_distance_down : distance_down = 15) -- The route down is 15 miles long
  : (distance_down / time_equal) / rate_up = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_hiking_rate_ratio_l2553_255375


namespace NUMINAMATH_CALUDE_exam_score_ratio_l2553_255363

theorem exam_score_ratio (total_questions : ℕ) (lowella_percentage : ℚ) 
  (pamela_additional_percentage : ℚ) (mandy_score : ℕ) : 
  total_questions = 100 →
  lowella_percentage = 35 / 100 →
  pamela_additional_percentage = 20 / 100 →
  mandy_score = 84 →
  ∃ (k : ℚ), k * (lowella_percentage * total_questions + 
    pamela_additional_percentage * (lowella_percentage * total_questions)) = mandy_score ∧ 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_ratio_l2553_255363


namespace NUMINAMATH_CALUDE_flagpole_break_height_l2553_255330

/-- 
Given a flagpole of height 8 meters that breaks such that the upper part touches the ground 3 meters from the base, 
the height from the ground to the break point is √73/2 meters.
-/
theorem flagpole_break_height (h : ℝ) (d : ℝ) (x : ℝ) 
  (h_height : h = 8) 
  (d_distance : d = 3) 
  (x_def : x = h - (h^2 - d^2).sqrt / 2) : 
  x = Real.sqrt 73 / 2 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l2553_255330


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2553_255322

theorem complex_equation_solution (a : ℝ) (i : ℂ) (hi : i * i = -1) :
  (a + 2 * i) / (2 + i) = i → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2553_255322


namespace NUMINAMATH_CALUDE_arun_speed_l2553_255388

theorem arun_speed (arun_speed : ℝ) (anil_speed : ℝ) : 
  (30 / arun_speed = 30 / anil_speed + 2) →
  (30 / (2 * arun_speed) = 30 / anil_speed - 1) →
  arun_speed = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_arun_speed_l2553_255388


namespace NUMINAMATH_CALUDE_quadratic_root_square_condition_l2553_255309

theorem quadratic_root_square_condition (p q r : ℝ) (α β : ℝ) : 
  (p * α^2 + q * α + r = 0) →  -- α is a root of the quadratic equation
  (p * β^2 + q * β + r = 0) →  -- β is a root of the quadratic equation
  (β = α^2) →                  -- one root is the square of the other
  (p - 4*q ≥ 0) :=             -- the relationship between coefficients
by sorry

end NUMINAMATH_CALUDE_quadratic_root_square_condition_l2553_255309


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l2553_255341

/-- Tax calculation problem -/
theorem tax_rate_calculation (total_value : ℝ) (tax_free_threshold : ℝ) (tax_paid : ℝ) :
  total_value = 1720 →
  tax_free_threshold = 600 →
  tax_paid = 112 →
  (tax_paid / (total_value - tax_free_threshold)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_calculation_l2553_255341


namespace NUMINAMATH_CALUDE_simplify_expression_l2553_255303

theorem simplify_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = 3*(x + y)) : 
  x/y + y/x - 3/(x*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2553_255303


namespace NUMINAMATH_CALUDE_multiplication_and_division_l2553_255300

theorem multiplication_and_division : 
  (8 * 4 = 32) ∧ (36 / 9 = 4) := by sorry

end NUMINAMATH_CALUDE_multiplication_and_division_l2553_255300


namespace NUMINAMATH_CALUDE_f_not_in_first_quadrant_l2553_255313

/-- A linear function defined by y = -3x + 2 -/
def f (x : ℝ) : ℝ := -3 * x + 2

/-- Definition of the first quadrant -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Theorem stating that the function f does not pass through the first quadrant -/
theorem f_not_in_first_quadrant :
  ∀ x : ℝ, ¬(first_quadrant x (f x)) :=
sorry

end NUMINAMATH_CALUDE_f_not_in_first_quadrant_l2553_255313


namespace NUMINAMATH_CALUDE_inequality_proof_l2553_255342

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 / (b * (a + b)) + 2 / (c * (b + c)) + 2 / (a * (c + a)) ≥ 27 / (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2553_255342


namespace NUMINAMATH_CALUDE_birds_on_fence_l2553_255366

theorem birds_on_fence (num_birds : ℕ) (h : num_birds = 20) : 
  2 * num_birds + 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2553_255366


namespace NUMINAMATH_CALUDE_committee_selection_l2553_255346

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committee_selection :
  choose 20 3 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l2553_255346


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2553_255348

-- Define the sequence a_n
def a (n : ℕ+) : ℕ := 2 * n - 1

-- Define the sum of the first n terms
def S (n : ℕ+) : ℕ := n * n

-- State the theorem
theorem arithmetic_sequence_inequality (m k p : ℕ+) (h : m + p = 2 * k) :
  1 / S m + 1 / S p ≥ 2 / S k := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2553_255348


namespace NUMINAMATH_CALUDE_add_same_power_of_x_l2553_255371

theorem add_same_power_of_x (x : ℝ) : x^3 + x^3 = 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_add_same_power_of_x_l2553_255371


namespace NUMINAMATH_CALUDE_feb_29_is_sunday_l2553_255382

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February of a leap year -/
structure FebruaryDate :=
  (day : Nat)
  (isLeapYear : Bool)

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Advances the day of the week by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Main theorem: If February 11th is a Wednesday in a leap year, then February 29th is a Sunday -/
theorem feb_29_is_sunday (d : FebruaryDate) (dow : DayOfWeek) :
  d.day = 11 → d.isLeapYear = true → dow = DayOfWeek.Wednesday →
  advanceDays dow 18 = DayOfWeek.Sunday :=
by
  sorry


end NUMINAMATH_CALUDE_feb_29_is_sunday_l2553_255382


namespace NUMINAMATH_CALUDE_rectangle_area_calculation_l2553_255317

/-- Rectangle with known side and area -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle similar to Rectangle1 with known diagonal -/
structure Rectangle2 where
  diagonal : ℝ

/-- The theorem to be proved -/
theorem rectangle_area_calculation
  (R1 : Rectangle1)
  (R2 : Rectangle2)
  (h1 : R1.side = 4)
  (h2 : R1.area = 32)
  (h3 : R2.diagonal = 10 * Real.sqrt 2)
  : ∃ (a : ℝ), a * (2 * a) = 80 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_calculation_l2553_255317


namespace NUMINAMATH_CALUDE_janet_action_figures_l2553_255383

/-- Calculates the final number of action figures Janet has -/
def final_action_figure_count (initial_count : ℕ) (sold_count : ℕ) (bought_count : ℕ) : ℕ :=
  let remaining_count := initial_count - sold_count
  let after_purchase_count := remaining_count + bought_count
  after_purchase_count + 2 * after_purchase_count

theorem janet_action_figures :
  final_action_figure_count 10 6 4 = 24 := by
  sorry

#eval final_action_figure_count 10 6 4

end NUMINAMATH_CALUDE_janet_action_figures_l2553_255383


namespace NUMINAMATH_CALUDE_fisher_algebra_eligibility_l2553_255372

/-- Determines if a student is eligible for algebra based on their quarterly scores -/
def isEligible (q1 q2 q3 q4 : ℚ) : Prop :=
  (q1 + q2 + q3 + q4) / 4 ≥ 83

/-- Fisher's minimum required score for the 4th quarter -/
def fisherMinScore : ℚ := 98

theorem fisher_algebra_eligibility :
  ∀ q4 : ℚ,
  isEligible 82 77 75 q4 ↔ q4 ≥ fisherMinScore :=
by sorry

#check fisher_algebra_eligibility

end NUMINAMATH_CALUDE_fisher_algebra_eligibility_l2553_255372


namespace NUMINAMATH_CALUDE_force_for_10_inch_screwdriver_l2553_255321

/-- Represents the force-length relationship for screwdrivers -/
structure ScrewdriverForce where
  force : ℝ
  length : ℝ
  constant : ℝ

/-- The force-length relationship is inverse and constant -/
axiom force_length_relation (sf : ScrewdriverForce) : sf.force * sf.length = sf.constant

/-- Given conditions for the 6-inch screwdriver -/
def initial_screwdriver : ScrewdriverForce :=
  { force := 60
    length := 6
    constant := 60 * 6 }

/-- Theorem stating the force required for a 10-inch screwdriver -/
theorem force_for_10_inch_screwdriver :
  ∃ (sf : ScrewdriverForce), sf.length = 10 ∧ sf.constant = initial_screwdriver.constant ∧ sf.force = 36 :=
by sorry

end NUMINAMATH_CALUDE_force_for_10_inch_screwdriver_l2553_255321


namespace NUMINAMATH_CALUDE_joker_probability_l2553_255353

/-- A deck of cards with Jokers -/
structure DeckWithJokers where
  total_cards : ℕ
  joker_cards : ℕ
  unique_cards : Prop
  shuffled : Prop

/-- The probability of drawing a specific card type from a deck -/
def probability_of_draw (deck : DeckWithJokers) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / deck.total_cards

/-- Our specific deck configuration -/
def our_deck : DeckWithJokers := {
  total_cards := 54
  joker_cards := 2
  unique_cards := True
  shuffled := True
}

/-- Theorem: The probability of drawing a Joker from our deck is 1/27 -/
theorem joker_probability :
  probability_of_draw our_deck our_deck.joker_cards = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_joker_probability_l2553_255353


namespace NUMINAMATH_CALUDE_zeros_in_Q_l2553_255367

/-- R_k is an integer whose base-ten representation is a sequence of k ones -/
def R (k : ℕ+) : ℕ := (10^k.val - 1) / 9

/-- Q is the quotient of R_28 divided by R_8 -/
def Q : ℕ := R 28 / R 8

/-- count_zeros counts the number of zeros in the base-ten representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 21 := by sorry

end NUMINAMATH_CALUDE_zeros_in_Q_l2553_255367


namespace NUMINAMATH_CALUDE_connie_calculation_l2553_255381

theorem connie_calculation (x : ℤ) : x + 2 = 80 → x - 2 = 76 := by
  sorry

end NUMINAMATH_CALUDE_connie_calculation_l2553_255381


namespace NUMINAMATH_CALUDE_factorial_product_not_square_l2553_255304

theorem factorial_product_not_square (n : ℕ) : 
  ∃ (m : ℕ), (n.factorial ^ 2 * (n + 1).factorial * (2 * n + 9).factorial * (2 * n + 10).factorial) ≠ m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_product_not_square_l2553_255304


namespace NUMINAMATH_CALUDE_degree_of_P_l2553_255319

/-- The degree of a monomial in two variables --/
def monomialDegree (a b : ℕ) : ℕ := a + b

/-- The degree of a polynomial is the maximum degree of its monomials --/
def polynomialDegree (degrees : List ℕ) : ℕ := List.foldl max 0 degrees

/-- The polynomial -3a²b + 7a²b² - 2ab --/
def P (a b : ℝ) : ℝ := -3 * a^2 * b + 7 * a^2 * b^2 - 2 * a * b

theorem degree_of_P : 
  polynomialDegree [monomialDegree 2 1, monomialDegree 2 2, monomialDegree 1 1] = 4 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_P_l2553_255319


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l2553_255379

theorem quadratic_roots_expression (x₁ x₂ : ℝ) 
  (h1 : x₁^2 + 5*x₁ + 1 = 0) 
  (h2 : x₂^2 + 5*x₂ + 1 = 0) : 
  (x₁*Real.sqrt 6 / (1 + x₂))^2 + (x₂*Real.sqrt 6 / (1 + x₁))^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l2553_255379


namespace NUMINAMATH_CALUDE_x_twenty_percent_greater_than_52_l2553_255391

theorem x_twenty_percent_greater_than_52 (x : ℝ) : 
  x = 52 * (1 + 20 / 100) → x = 62.4 := by
sorry

end NUMINAMATH_CALUDE_x_twenty_percent_greater_than_52_l2553_255391


namespace NUMINAMATH_CALUDE_lloyd_earnings_correct_l2553_255378

/-- Calculates Lloyd's earnings for the given work days -/
def lloyd_earnings (regular_rate : ℚ) (normal_hours : ℚ) (overtime_rate : ℚ) (saturday_rate : ℚ) 
  (monday_hours : ℚ) (tuesday_hours : ℚ) (saturday_hours : ℚ) : ℚ :=
  let monday_earnings := 
    min normal_hours monday_hours * regular_rate + 
    max 0 (monday_hours - normal_hours) * regular_rate * overtime_rate
  let tuesday_earnings := 
    min normal_hours tuesday_hours * regular_rate + 
    max 0 (tuesday_hours - normal_hours) * regular_rate * overtime_rate
  let saturday_earnings := saturday_hours * regular_rate * saturday_rate
  monday_earnings + tuesday_earnings + saturday_earnings

theorem lloyd_earnings_correct : 
  lloyd_earnings 5 8 (3/2) 2 (21/2) 9 6 = 665/4 := by
  sorry

end NUMINAMATH_CALUDE_lloyd_earnings_correct_l2553_255378


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2553_255385

theorem negation_of_proposition :
  (¬ (∀ n : ℕ, ¬ (Nat.Prime (2^n - 2)))) ↔ (∃ n : ℕ, Nat.Prime (2^n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2553_255385


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_sum_achieved_l2553_255329

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  x + 2*y ≥ 3 + 8*Real.sqrt 2 := by
sorry

theorem min_value_sum_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀ + 2*y₀ = 3 + 8*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_sum_achieved_l2553_255329


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2553_255318

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {-1, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2553_255318


namespace NUMINAMATH_CALUDE_double_reflection_of_D_l2553_255331

/-- Reflects a point across the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point across the line y = -x + 1 -/
def reflect_diagonal (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2 + 1, -p.1 + 1)

/-- The main theorem -/
theorem double_reflection_of_D (D : ℝ × ℝ) (hD : D = (5, 2)) :
  (reflect_diagonal ∘ reflect_x_axis) D = (-3, -4) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_of_D_l2553_255331


namespace NUMINAMATH_CALUDE_series_sum_equals_one_l2553_255362

theorem series_sum_equals_one : 
  ∑' n : ℕ+, (1 : ℝ) / (n * (n + 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_l2553_255362


namespace NUMINAMATH_CALUDE_absolute_value_of_c_l2553_255390

def complex (a b : ℝ) := a + b * Complex.I

theorem absolute_value_of_c (a b c : ℤ) : 
  (∃ (z : ℂ), z = complex 3 1 ∧ a * z^4 + b * z^3 + c * z^2 + b * z + a = 0) →
  (Int.gcd a (Int.gcd b c) = 1) →
  abs c = 109 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_c_l2553_255390


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l2553_255384

theorem gcd_special_numbers : Nat.gcd 33333333 666666666 = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l2553_255384


namespace NUMINAMATH_CALUDE_exists_permutation_multiple_of_seven_l2553_255337

/-- A function that generates all permutations of a list -/
def permutations (l : List ℕ) : List (List ℕ) :=
  sorry

/-- A function that converts a list of digits to a natural number -/
def list_to_number (l : List ℕ) : ℕ :=
  sorry

/-- The theorem stating that there exists a permutation of digits 1, 3, 7, 9 that forms a multiple of 7 -/
theorem exists_permutation_multiple_of_seven :
  ∃ (perm : List ℕ), perm ∈ permutations [1, 3, 7, 9] ∧ (list_to_number perm) % 7 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_multiple_of_seven_l2553_255337


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_and_b_l2553_255397

/-- Given a curve f(x) = ax - b/x, prove that if its tangent line at (2, f(2)) 
    is 7x - 4y - 12 = 0, then a = 1 and b = 3 -/
theorem tangent_line_implies_a_and_b (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x - b / x
  let f' : ℝ → ℝ := λ x => a + b / (x^2)
  let tangent_slope : ℝ := f' 2
  let point_on_curve : ℝ := f 2
  (7 * 2 - 4 * point_on_curve - 12 = 0 ∧ 
   7 - 4 * tangent_slope = 0) →
  a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_and_b_l2553_255397


namespace NUMINAMATH_CALUDE_complement_of_complement_l2553_255311

theorem complement_of_complement (α : ℝ) (h : α = 35) :
  90 - (90 - α) = α := by sorry

end NUMINAMATH_CALUDE_complement_of_complement_l2553_255311


namespace NUMINAMATH_CALUDE_cyclic_matrix_squared_identity_l2553_255344

/-- A 4x4 complex matrix with a cyclic structure -/
def CyclicMatrix (a b c d : ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  !![a, b, c, d;
     b, c, d, a;
     c, d, a, b;
     d, a, b, c]

theorem cyclic_matrix_squared_identity
  (a b c d : ℂ)
  (h1 : (CyclicMatrix a b c d) ^ 2 = 1)
  (h2 : a * b * c * d = 1) :
  a ^ 4 + b ^ 4 + c ^ 4 + d ^ 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_matrix_squared_identity_l2553_255344


namespace NUMINAMATH_CALUDE_product_selection_events_l2553_255345

structure ProductSelection where
  total : Nat
  genuine : Nat
  defective : Nat
  selected : Nat

def is_random_event (ps : ProductSelection) (event : Nat → Prop) : Prop :=
  ∃ (outcome : Nat), event outcome ∧
  ∃ (outcome : Nat), ¬event outcome

def is_impossible_event (ps : ProductSelection) (event : Nat → Prop) : Prop :=
  ∀ (outcome : Nat), ¬event outcome

def is_certain_event (ps : ProductSelection) (event : Nat → Prop) : Prop :=
  ∀ (outcome : Nat), event outcome

def all_genuine (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome = ps.genuine.choose ps.selected

def at_least_one_defective (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome > 0

def all_defective (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome = ps.defective.choose ps.selected

def at_least_one_genuine (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome < ps.selected

theorem product_selection_events (ps : ProductSelection) 
  (h1 : ps.total = 12)
  (h2 : ps.genuine = 10)
  (h3 : ps.defective = 2)
  (h4 : ps.selected = 3)
  (h5 : ps.total = ps.genuine + ps.defective) :
  is_random_event ps (all_genuine ps) ∧
  is_random_event ps (at_least_one_defective ps) ∧
  is_impossible_event ps (all_defective ps) ∧
  is_certain_event ps (at_least_one_genuine ps) := by
  sorry

end NUMINAMATH_CALUDE_product_selection_events_l2553_255345


namespace NUMINAMATH_CALUDE_smallest_two_digit_integer_with_property_l2553_255335

theorem smallest_two_digit_integer_with_property : ∃ n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (let a := n / 10; let b := n % 10; 10 * b + a + 5 = 2 * n) ∧
  (∀ m : ℕ, m ≥ 10 ∧ m < 100 → 
    (let x := m / 10; let y := m % 10; 10 * y + x + 5 = 2 * m) → 
    m ≥ n) ∧
  n = 69 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_integer_with_property_l2553_255335


namespace NUMINAMATH_CALUDE_typists_for_180_letters_l2553_255355

/-- The number of typists needed to type a certain number of letters in a given time, 
    given a known typing rate. -/
def typists_needed 
  (known_typists : ℕ) 
  (known_letters : ℕ) 
  (known_minutes : ℕ) 
  (target_letters : ℕ) 
  (target_minutes : ℕ) : ℕ :=
  sorry

theorem typists_for_180_letters 
  (h1 : typists_needed 20 40 20 180 60 = 30) : 
  typists_needed 20 40 20 180 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_typists_for_180_letters_l2553_255355


namespace NUMINAMATH_CALUDE_theater_ticket_cost_l2553_255340

/-- The cost of tickets at a theater -/
theorem theater_ticket_cost 
  (adult_price : ℝ) 
  (h1 : adult_price > 0)
  (h2 : 4 * adult_price + 3 * (adult_price / 2) + 2 * (3 * adult_price / 4) = 42) :
  6 * adult_price + 5 * (adult_price / 2) + 4 * (3 * adult_price / 4) = 69 := by
  sorry


end NUMINAMATH_CALUDE_theater_ticket_cost_l2553_255340


namespace NUMINAMATH_CALUDE_cubic_difference_l2553_255312

theorem cubic_difference (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 2 * x + y = 16) : 
  x^3 - y^3 = -448 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l2553_255312


namespace NUMINAMATH_CALUDE_parallelogram_area_specific_vectors_l2553_255358

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogram_area (v w : Fin 2 → ℝ) : ℝ :=
  |v 0 * w 1 - v 1 * w 0|

theorem parallelogram_area_specific_vectors :
  let v : Fin 2 → ℝ := ![7, -4]
  let w : Fin 2 → ℝ := ![12, -1]
  parallelogram_area v w = 41 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_specific_vectors_l2553_255358


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2553_255338

def U : Set ℤ := {x | -4 < x ∧ x < 4}
def A : Set ℤ := {-1, 0, 2, 3}
def B : Set ℤ := {-2, 0, 1, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {-3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2553_255338


namespace NUMINAMATH_CALUDE_unique_point_in_S_l2553_255310

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ Real.log (p.1^3 + (1/3)*p.2^3 + 1/9) = Real.log p.1 + Real.log p.2}

theorem unique_point_in_S : ∃! p : ℝ × ℝ, p ∈ S := by sorry

end NUMINAMATH_CALUDE_unique_point_in_S_l2553_255310


namespace NUMINAMATH_CALUDE_favorite_numbers_sum_l2553_255314

/-- Given the favorite numbers of Misty, Glory, and Dawn, prove their sum is 1500 -/
theorem favorite_numbers_sum (glory_fav : ℕ) (misty_fav : ℕ) (dawn_fav : ℕ) 
  (h1 : glory_fav = 450)
  (h2 : misty_fav * 3 = glory_fav)
  (h3 : dawn_fav = glory_fav * 2) :
  misty_fav + glory_fav + dawn_fav = 1500 := by
  sorry

end NUMINAMATH_CALUDE_favorite_numbers_sum_l2553_255314


namespace NUMINAMATH_CALUDE_complement_of_A_is_closed_ray_l2553_255395

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as the domain of log(2-x)
def A : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem complement_of_A_is_closed_ray :
  Set.compl A = Set.Ici (2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_is_closed_ray_l2553_255395


namespace NUMINAMATH_CALUDE_randolph_is_55_l2553_255374

def sherry_age : ℕ := 25

def sydney_age : ℕ := 2 * sherry_age

def randolph_age : ℕ := sydney_age + 5

theorem randolph_is_55 : randolph_age = 55 := by
  sorry

end NUMINAMATH_CALUDE_randolph_is_55_l2553_255374


namespace NUMINAMATH_CALUDE_g_of_three_value_l2553_255394

/-- The function g satisfies 4g(x) - 3g(1/x) = x^2 for all x ≠ 0 -/
def g : ℝ → ℝ :=
  fun x => sorry

/-- The main theorem stating that g(3) = 36.333/7 -/
theorem g_of_three_value : g 3 = 36.333 / 7 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_value_l2553_255394


namespace NUMINAMATH_CALUDE_root_sum_square_l2553_255387

theorem root_sum_square (a b : ℝ) : 
  a ≠ b →
  (a^2 + 2*a - 2022 = 0) → 
  (b^2 + 2*b - 2022 = 0) → 
  a^2 + 4*a + 2*b = 2018 := by
sorry

end NUMINAMATH_CALUDE_root_sum_square_l2553_255387


namespace NUMINAMATH_CALUDE_ceiling_cube_fraction_plus_one_l2553_255328

theorem ceiling_cube_fraction_plus_one :
  ⌈(-5/3)^3 + 1⌉ = -3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_cube_fraction_plus_one_l2553_255328


namespace NUMINAMATH_CALUDE_even_function_property_l2553_255301

def evenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property
  (f : ℝ → ℝ)
  (h_even : evenFunction f)
  (h_increasing : ∀ x y, x < y → x < 0 → f x < f y)
  (x₁ x₂ : ℝ)
  (h_x₁_neg : x₁ < 0)
  (h_x₂_pos : x₂ > 0)
  (h_abs : |x₁| < |x₂|) :
  f (-x₁) > f (-x₂) := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l2553_255301


namespace NUMINAMATH_CALUDE_magnitude_v_l2553_255339

variable (u v : ℂ)

theorem magnitude_v (h1 : u * v = 20 - 15 * Complex.I) (h2 : Complex.abs u = 5) : 
  Complex.abs v = 5 := by
sorry

end NUMINAMATH_CALUDE_magnitude_v_l2553_255339


namespace NUMINAMATH_CALUDE_prime_square_mod_504_l2553_255325

theorem prime_square_mod_504 (p : Nat) (h_prime : Nat.Prime p) (h_gt_7 : p > 7) :
  ∃! (s : Finset Nat), 
    (∀ r ∈ s, r < 504 ∧ ∃ q : Nat, p^2 = 504 * q + r) ∧ 
    s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_504_l2553_255325


namespace NUMINAMATH_CALUDE_milk_price_increase_percentage_l2553_255364

def lowest_price : ℝ := 16
def highest_price : ℝ := 22

theorem milk_price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_increase_percentage_l2553_255364


namespace NUMINAMATH_CALUDE_fourth_number_in_proportion_l2553_255365

-- Define the proportion
def proportion (a b c d : ℝ) : Prop := a / b = c / d

-- State the theorem
theorem fourth_number_in_proportion : 
  proportion 0.75 1.35 5 9 := by sorry

end NUMINAMATH_CALUDE_fourth_number_in_proportion_l2553_255365


namespace NUMINAMATH_CALUDE_probability_point_closer_to_center_l2553_255326

theorem probability_point_closer_to_center (R : Real) (r : Real) : 
  R = 3 → r = 1.5 → (π * r^2) / (π * R^2) = 1/4 := by sorry

end NUMINAMATH_CALUDE_probability_point_closer_to_center_l2553_255326


namespace NUMINAMATH_CALUDE_invertible_function_theorem_l2553_255320

noncomputable section

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem invertible_function_theorem (c d : ℝ) 
  (h1 : Function.Injective g) 
  (h2 : g c = d) 
  (h3 : g d = 5) : 
  c - d = -2 := by sorry

end NUMINAMATH_CALUDE_invertible_function_theorem_l2553_255320


namespace NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l2553_255351

theorem largest_of_seven_consecutive_integers (n : ℕ) 
  (h1 : n > 0)
  (h2 : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 2222) : 
  n + 6 = 320 :=
sorry

end NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l2553_255351


namespace NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_six_fifths_l2553_255399

theorem tan_theta_two_implies_expression_equals_six_fifths (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / 
  (Real.sqrt 2 * Real.cos (θ - π / 4)) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_six_fifths_l2553_255399


namespace NUMINAMATH_CALUDE_range_of_m_l2553_255350

-- Define the propositions p and q
def p (x : ℝ) : Prop := |2*x - 1| ≤ 5
def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - 9*m^2 ≤ 0

-- Define the set of x that satisfies ¬p
def not_p_set : Set ℝ := {x | ¬(p x)}

-- Define the set of x that satisfies ¬q
def not_q_set (m : ℝ) : Set ℝ := {x | ¬(q x m)}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, x ∈ not_p_set → x ∈ not_q_set m) →
  (∃ x : ℝ, x ∈ not_q_set m ∧ x ∉ not_p_set) →
  m ∈ Set.Ioo 0 (1/3) ∪ {1/3} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2553_255350


namespace NUMINAMATH_CALUDE_range_of_m_l2553_255361

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function has a minimum positive period of p if f(x + p) = f(x) for all x,
    and p is the smallest positive number with this property -/
def HasMinPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  (∀ x, f (x + p) = f x) ∧ ∀ q, 0 < q → q < p → ∃ x, f (x + q) ≠ f x

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
    (h_odd : IsOdd f)
    (h_period : HasMinPeriod f 3)
    (h_2015 : f 2015 > 1)
    (h_1 : f 1 = (2*m + 3)/(m - 1)) :
    -2/3 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2553_255361


namespace NUMINAMATH_CALUDE_fifteen_dogs_like_neither_l2553_255377

/-- Represents the number of dogs in different categories -/
structure DogCounts where
  total : Nat
  likesChicken : Nat
  likesBeef : Nat
  likesBoth : Nat

/-- Calculates the number of dogs that don't like either chicken or beef -/
def dogsLikingNeither (counts : DogCounts) : Nat :=
  counts.total - (counts.likesChicken + counts.likesBeef - counts.likesBoth)

/-- Theorem stating that 15 dogs don't like either chicken or beef -/
theorem fifteen_dogs_like_neither (counts : DogCounts)
  (h1 : counts.total = 75)
  (h2 : counts.likesChicken = 13)
  (h3 : counts.likesBeef = 55)
  (h4 : counts.likesBoth = 8) :
  dogsLikingNeither counts = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_dogs_like_neither_l2553_255377


namespace NUMINAMATH_CALUDE_ratio_AB_BC_l2553_255373

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The diagram configuration -/
structure Diagram where
  rectangles : Fin 5 → Rectangle
  x : ℝ
  h1 : ∀ i, (rectangles i).width = x
  h2 : ∀ i, (rectangles i).length = 3 * x

/-- AB is the sum of two widths and one length -/
def AB (d : Diagram) : ℝ := 2 * d.x + 3 * d.x

/-- BC is the length of one rectangle -/
def BC (d : Diagram) : ℝ := 3 * d.x

theorem ratio_AB_BC (d : Diagram) : AB d / BC d = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_AB_BC_l2553_255373


namespace NUMINAMATH_CALUDE_allison_wins_probability_l2553_255349

-- Define the faces of each cube
def allison_cube : Finset Nat := {4}
def charlie_cube : Finset Nat := {1, 2, 3, 4, 5, 6}
def eve_cube : Finset Nat := {3, 3, 4, 4, 4, 5}

-- Define the probability of rolling each face
def prob_roll (cube : Finset Nat) (face : Nat) : ℚ :=
  (cube.filter (· = face)).card / cube.card

-- Define the event of rolling less than 4
def roll_less_than_4 (cube : Finset Nat) : ℚ :=
  (cube.filter (· < 4)).card / cube.card

-- Theorem statement
theorem allison_wins_probability :
  prob_roll allison_cube 4 * roll_less_than_4 charlie_cube * roll_less_than_4 eve_cube = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_allison_wins_probability_l2553_255349


namespace NUMINAMATH_CALUDE_train_distance_difference_l2553_255305

theorem train_distance_difference (v : ℝ) (h1 : v > 0) : 
  let d_ab := 7 * v
  let d_bc := 5 * v
  6 = d_ab + d_bc →
  d_ab - d_bc = 1 := by
sorry

end NUMINAMATH_CALUDE_train_distance_difference_l2553_255305


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2553_255336

/-- Given vectors a, b, and c in ℝ², prove that if a is perpendicular to (b - c), then x = 4/3 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (3, 1)) 
  (h2 : b = (x, -2)) 
  (h3 : c = (0, 2)) 
  (h4 : a • (b - c) = 0) : 
  x = 4/3 := by
  sorry

#check perpendicular_vectors

end NUMINAMATH_CALUDE_perpendicular_vectors_l2553_255336


namespace NUMINAMATH_CALUDE_jerry_shelf_difference_l2553_255307

/-- The difference between action figures and books on Jerry's shelf -/
def shelf_difference (initial_figures : ℕ) (initial_books : ℕ) (added_books : ℕ) : ℤ :=
  (initial_figures : ℤ) - ((initial_books : ℤ) + (added_books : ℤ))

/-- Theorem stating the difference between action figures and books on Jerry's shelf -/
theorem jerry_shelf_difference :
  shelf_difference 7 2 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_difference_l2553_255307


namespace NUMINAMATH_CALUDE_triangle_area_l2553_255376

/-- Prove that the area of the triangle formed by the lines x = -5, y = x, and the x-axis is 12.5 -/
theorem triangle_area : 
  let line1 : ℝ → ℝ := λ x => -5
  let line2 : ℝ → ℝ := λ x => x
  let intersection_x : ℝ := -5
  let intersection_y : ℝ := line2 intersection_x
  let base : ℝ := abs intersection_x
  let height : ℝ := abs intersection_y
  (1/2) * base * height = 12.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2553_255376


namespace NUMINAMATH_CALUDE_sin_five_pi_thirds_l2553_255324

theorem sin_five_pi_thirds : Real.sin (5 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_thirds_l2553_255324


namespace NUMINAMATH_CALUDE_james_tin_collection_l2553_255370

/-- The number of tins James collects in a week -/
def total_tins : ℕ := 500

/-- The number of tins James collects on the first day -/
def first_day_tins : ℕ := 50

/-- The number of tins James collects on the second day -/
def second_day_tins : ℕ := 3 * first_day_tins

/-- The number of tins James collects on each of the remaining days (4th to 7th) -/
def remaining_days_tins : ℕ := 50

/-- The total number of tins James collects on the remaining days (4th to 7th) -/
def total_remaining_days_tins : ℕ := 4 * remaining_days_tins

/-- The number of tins James collects on the third day -/
def third_day_tins : ℕ := total_tins - first_day_tins - second_day_tins - total_remaining_days_tins

theorem james_tin_collection :
  second_day_tins - third_day_tins = 50 :=
sorry

end NUMINAMATH_CALUDE_james_tin_collection_l2553_255370


namespace NUMINAMATH_CALUDE_function_f_form_l2553_255368

/-- A function from positive integers to non-negative integers satisfying the given property -/
def FunctionF (f : ℕ+ → ℕ) : Prop :=
  f ≠ 0 ∧ ∀ a b : ℕ+, 2 * f (a * b) = (↑b + 1) * f a + (↑a + 1) * f b

/-- The main theorem stating the existence of c such that f(n) = c(n-1) -/
theorem function_f_form (f : ℕ+ → ℕ) (hf : FunctionF f) :
  ∃ c : ℕ, ∀ n : ℕ+, f n = c * (↑n - 1) :=
sorry

end NUMINAMATH_CALUDE_function_f_form_l2553_255368


namespace NUMINAMATH_CALUDE_a_months_is_seven_l2553_255396

/-- Represents the rental arrangement for a pasture -/
structure PastureRental where
  a_oxen : ℕ
  b_oxen : ℕ
  c_oxen : ℕ
  b_months : ℕ
  c_months : ℕ
  total_rent : ℕ
  c_share : ℕ

/-- Calculates the number of months a put his oxen for grazing -/
def calculate_a_months (rental : PastureRental) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, a put his oxen for 7 months -/
theorem a_months_is_seven (rental : PastureRental)
  (h1 : rental.a_oxen = 10)
  (h2 : rental.b_oxen = 12)
  (h3 : rental.c_oxen = 15)
  (h4 : rental.b_months = 5)
  (h5 : rental.c_months = 3)
  (h6 : rental.total_rent = 140)
  (h7 : rental.c_share = 36) :
  calculate_a_months rental = 7 :=
sorry

end NUMINAMATH_CALUDE_a_months_is_seven_l2553_255396


namespace NUMINAMATH_CALUDE_intersection_A_B_l2553_255316

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {2, 3}

theorem intersection_A_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2553_255316


namespace NUMINAMATH_CALUDE_rotated_angle_intersection_l2553_255386

/-- 
Given an angle α, when its terminal side is rotated clockwise by π/2,
the intersection of the new angle with the unit circle centered at the origin
has coordinates (sin α, -cos α).
-/
theorem rotated_angle_intersection (α : Real) : 
  let rotated_angle := α - π / 2
  let x := Real.cos rotated_angle
  let y := Real.sin rotated_angle
  (x, y) = (Real.sin α, -Real.cos α) := by
sorry

end NUMINAMATH_CALUDE_rotated_angle_intersection_l2553_255386


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l2553_255308

theorem quadratic_form_k_value : ∃ (a h : ℝ), ∀ x : ℝ, 
  x^2 - 6*x = a*(x - h)^2 + (-9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l2553_255308


namespace NUMINAMATH_CALUDE_japanese_students_fraction_l2553_255360

theorem japanese_students_fraction (J : ℚ) (h1 : J > 0) : 
  let S := 3 * J
  let seniors_studying := (1/3) * S
  let juniors_studying := (3/4) * J
  let total_students := S + J
  (seniors_studying + juniors_studying) / total_students = 7/16 := by
sorry

end NUMINAMATH_CALUDE_japanese_students_fraction_l2553_255360


namespace NUMINAMATH_CALUDE_problem_statement_l2553_255369

theorem problem_statement : 
  let p := ∀ a b c : ℝ, a > b → a * c^2 > b * c^2
  let q := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 + Real.log x₀ = 0
  (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2553_255369


namespace NUMINAMATH_CALUDE_five_balls_two_boxes_l2553_255306

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_balls + 1

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 2 distinguishable boxes -/
theorem five_balls_two_boxes : distribute_balls 5 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_two_boxes_l2553_255306


namespace NUMINAMATH_CALUDE_prism_intersection_probability_l2553_255323

/-- A rectangular prism with dimensions 2, 3, and 5 units. -/
structure RectangularPrism where
  length : ℕ := 2
  width : ℕ := 3
  height : ℕ := 5

/-- The probability that three randomly chosen vertices of the prism
    form a plane intersecting the prism's interior. -/
def intersectionProbability (p : RectangularPrism) : ℚ :=
  11/14

/-- Theorem stating that the probability of three randomly chosen vertices
    forming a plane that intersects the interior of the given rectangular prism is 11/14. -/
theorem prism_intersection_probability (p : RectangularPrism) :
  intersectionProbability p = 11/14 := by
  sorry

end NUMINAMATH_CALUDE_prism_intersection_probability_l2553_255323


namespace NUMINAMATH_CALUDE_subset_condition_l2553_255359

def P : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def S (a : ℝ) : Set ℝ := {x | a*x + 2 = 0}

theorem subset_condition (a : ℝ) : S a ⊆ P ↔ a = 0 ∨ a = 2 ∨ a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l2553_255359


namespace NUMINAMATH_CALUDE_third_group_draw_l2553_255393

/-- Represents a systematic sampling sequence -/
def SystematicSampling (first second : ℕ) : ℕ → ℕ := fun n => first + (n - 1) * (second - first)

/-- Theorem: In a systematic sampling where the first group draws 2 and the second group draws 12,
    the third group will draw 22 -/
theorem third_group_draw (first second : ℕ) (h1 : first = 2) (h2 : second = 12) :
  SystematicSampling first second 3 = 22 := by
  sorry

#eval SystematicSampling 2 12 3

end NUMINAMATH_CALUDE_third_group_draw_l2553_255393


namespace NUMINAMATH_CALUDE_ramu_profit_percent_l2553_255357

/-- Calculates the profit percent given the cost of a car, repair costs, and selling price --/
def profit_percent (car_cost repair_cost selling_price : ℚ) : ℚ :=
  let total_cost := car_cost + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that under the given conditions, the profit percent is 18% --/
theorem ramu_profit_percent :
  profit_percent 42000 13000 64900 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ramu_profit_percent_l2553_255357


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l2553_255354

theorem power_equality_implies_exponent (p : ℕ) : 16^6 = 4^p → p = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l2553_255354


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2553_255333

/-- Represents a chess tournament with the given property --/
structure ChessTournament where
  n : ℕ  -- Total number of players
  half_points_from_last_three : Prop  -- Property that each player scored half their points against the last three

/-- Theorem stating that a chess tournament satisfying the given condition has 9 participants --/
theorem chess_tournament_participants (t : ChessTournament) : t.n = 9 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2553_255333


namespace NUMINAMATH_CALUDE_correct_subtraction_l2553_255302

-- Define the polynomials
def original_poly (x : ℝ) := 2*x^2 - x + 3
def mistaken_poly (x : ℝ) := x^2 + 14*x - 6

-- Theorem statement
theorem correct_subtraction :
  ∀ x : ℝ, original_poly x - mistaken_poly x = -29*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l2553_255302


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_three_l2553_255343

/-- Given two linear functions f and g defined by real parameters A and B,
    proves that if f(g(x)) - g(f(x)) = 2(B - A) and A ≠ B, then A + B = 3. -/
theorem sum_of_coefficients_is_three
  (A B : ℝ)
  (hne : A ≠ B)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x + B)
  (hg : ∀ x, g x = B * x + A)
  (h : ∀ x, f (g x) - g (f x) = 2 * (B - A)) :
  A + B = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_three_l2553_255343
