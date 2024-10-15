import Mathlib

namespace NUMINAMATH_CALUDE_correct_num_arrangements_l2534_253444

/-- The number of different arrangements of 5 boys and 2 girls in a row,
    where one boy (A) must stand in the center and the two girls must stand next to each other. -/
def num_arrangements : ℕ :=
  Nat.choose 4 1 * Nat.factorial 2 * Nat.factorial 4

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_num_arrangements :
  num_arrangements = Nat.choose 4 1 * Nat.factorial 2 * Nat.factorial 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_num_arrangements_l2534_253444


namespace NUMINAMATH_CALUDE_star_inequality_equivalence_l2534_253472

-- Define the * operation
def star (a b : ℝ) : ℝ := (a + 3*b) - a*b

-- State the theorem
theorem star_inequality_equivalence :
  ∀ x : ℝ, star 5 x < 13 ↔ x > -4 :=
by sorry

end NUMINAMATH_CALUDE_star_inequality_equivalence_l2534_253472


namespace NUMINAMATH_CALUDE_stream_speed_l2534_253433

/-- Given a boat traveling downstream and upstream, prove the speed of the stream. -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 60) 
  (h2 : upstream_distance = 30) 
  (h3 : time = 3) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    stream_speed = 5 := by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_stream_speed_l2534_253433


namespace NUMINAMATH_CALUDE_circle_condition_l2534_253474

def is_circle (m : ℤ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), 
    x^2 + y^2 + m*x - m*y + 2 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

theorem circle_condition (m : ℤ) : 
  m ∈ ({0, 1, 2, 3} : Set ℤ) →
  (is_circle m ↔ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l2534_253474


namespace NUMINAMATH_CALUDE_only_opening_window_is_translational_l2534_253454

-- Define the type for phenomena
inductive Phenomenon
  | wipingCarWindows
  | openingClassroomDoor
  | openingClassroomWindow
  | swingingOnSwing

-- Define the property of being a translational motion
def isTranslationalMotion (p : Phenomenon) : Prop :=
  match p with
  | .wipingCarWindows => False
  | .openingClassroomDoor => False
  | .openingClassroomWindow => True
  | .swingingOnSwing => False

-- Theorem statement
theorem only_opening_window_is_translational :
  ∀ (p : Phenomenon), isTranslationalMotion p ↔ p = Phenomenon.openingClassroomWindow :=
by sorry

end NUMINAMATH_CALUDE_only_opening_window_is_translational_l2534_253454


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l2534_253414

theorem average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 6 →
  initial_average = 28 →
  leaving_age = 22 →
  remaining_people = 5 →
  (initial_people * initial_average - leaving_age) / remaining_people = 29.2 := by
  sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l2534_253414


namespace NUMINAMATH_CALUDE_rectangle_area_l2534_253431

theorem rectangle_area : 
  ∀ (square_side : ℝ) (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ),
    square_side^2 = 625 →
    circle_radius = square_side →
    rectangle_length = (2/5) * circle_radius →
    rectangle_breadth = 10 →
    rectangle_length * rectangle_breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2534_253431


namespace NUMINAMATH_CALUDE_negative_twenty_seven_to_five_thirds_l2534_253492

theorem negative_twenty_seven_to_five_thirds :
  (-27 : ℝ) ^ (5/3) = -243 := by
  sorry

end NUMINAMATH_CALUDE_negative_twenty_seven_to_five_thirds_l2534_253492


namespace NUMINAMATH_CALUDE_cos_equation_solutions_l2534_253470

theorem cos_equation_solutions :
  ∃! (S : Finset ℝ), 
    (∀ x ∈ S, x ∈ Set.Icc 0 Real.pi ∧ Real.cos (7 * x) = Real.cos (5 * x)) ∧
    S.card = 7 :=
sorry

end NUMINAMATH_CALUDE_cos_equation_solutions_l2534_253470


namespace NUMINAMATH_CALUDE_div_power_equals_power_diff_l2534_253442

theorem div_power_equals_power_diff (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_div_power_equals_power_diff_l2534_253442


namespace NUMINAMATH_CALUDE_sector_area_l2534_253495

/-- The area of a sector of a circle with radius 5 cm and arc length 4 cm is 10 cm². -/
theorem sector_area (r : ℝ) (arc_length : ℝ) (h1 : r = 5) (h2 : arc_length = 4) :
  (arc_length / (2 * π * r)) * (π * r^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l2534_253495


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2534_253437

theorem smallest_number_with_given_remainders :
  ∃ (x : ℕ), x > 0 ∧
    x % 11 = 9 ∧
    x % 13 = 11 ∧
    x % 15 = 13 ∧
    (∀ y : ℕ, y > 0 ∧ y % 11 = 9 ∧ y % 13 = 11 ∧ y % 15 = 13 → x ≤ y) ∧
    x = 2143 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2534_253437


namespace NUMINAMATH_CALUDE_center_is_nine_l2534_253436

def Grid := Fin 3 → Fin 3 → Nat

def is_valid_arrangement (g : Grid) : Prop :=
  (∀ n : Nat, n ∈ Finset.range 9 → ∃ i j, g i j = n + 1) ∧
  (∀ i j, g i j ∈ Finset.range 9 → g i j ≤ 9) ∧
  (∀ n : Nat, n ∈ Finset.range 8 → 
    ∃ i j k l, g i j = n + 1 ∧ g k l = n + 2 ∧ 
    ((i = k ∧ (j = l + 1 ∨ j + 1 = l)) ∨ 
     (j = l ∧ (i = k + 1 ∨ i + 1 = k))))

def top_edge_sum (g : Grid) : Nat :=
  g 0 0 + g 0 1 + g 0 2

theorem center_is_nine (g : Grid) 
  (h1 : is_valid_arrangement g) 
  (h2 : top_edge_sum g = 15) : 
  g 1 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_center_is_nine_l2534_253436


namespace NUMINAMATH_CALUDE_root_existence_l2534_253410

theorem root_existence : ∃ x : ℝ, x ∈ (Set.Ioo (-1) (-1/2)) ∧ 2^x + x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_existence_l2534_253410


namespace NUMINAMATH_CALUDE_positive_expressions_l2534_253462

theorem positive_expressions (U V W X Y : ℝ) 
  (h1 : U < V) (h2 : V < 0) (h3 : 0 < W) (h4 : W < X) (h5 : X < Y) : 
  (0 < U * V) ∧ 
  (0 < (X / V) * U) ∧ 
  (0 < W / (U * V)) ∧ 
  (0 < (X - Y) / W) := by
  sorry

end NUMINAMATH_CALUDE_positive_expressions_l2534_253462


namespace NUMINAMATH_CALUDE_max_consecutive_digit_sums_l2534_253412

/-- Given a natural number, returns the sum of its digits. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Returns true if the given list of natural numbers contains n consecutive numbers. -/
def isConsecutive (l : List ℕ) (n : ℕ) : Prop := sorry

/-- Theorem: 18 is the maximum value of n for which there exists a sequence of n consecutive 
    natural numbers whose digit sums form another sequence of n consecutive numbers. -/
theorem max_consecutive_digit_sums : 
  ∀ n : ℕ, n > 18 → 
  ¬∃ (start : ℕ), 
    let numbers := List.range n |>.map (λ i => start + i)
    let digitSums := numbers.map sumOfDigits
    isConsecutive numbers n ∧ isConsecutive digitSums n :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_digit_sums_l2534_253412


namespace NUMINAMATH_CALUDE_combined_tax_rate_l2534_253434

/-- The combined tax rate problem -/
theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (julie_rate : ℝ) 
  (mindy_income : ℝ → ℝ) 
  (julie_income : ℝ → ℝ) 
  (h1 : mork_rate = 0.45)
  (h2 : mindy_rate = 0.25)
  (h3 : julie_rate = 0.35)
  (h4 : ∀ m, mindy_income m = 4 * m)
  (h5 : ∀ m, julie_income m = 2 * m)
  (h6 : ∀ m, julie_income m = (mindy_income m) / 2) :
  ∀ m : ℝ, m > 0 → 
    (mork_rate * m + mindy_rate * (mindy_income m) + julie_rate * (julie_income m)) / 
    (m + mindy_income m + julie_income m) = 2.15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l2534_253434


namespace NUMINAMATH_CALUDE_three_zeros_range_of_a_l2534_253478

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*a^2*x - 4*a

-- State the theorem
theorem three_zeros_range_of_a (a : ℝ) :
  a > 0 ∧ (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  a > Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_three_zeros_range_of_a_l2534_253478


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2534_253455

theorem trigonometric_identities (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 3 * Real.sin (π - α) = -2 * Real.cos (π + α)) : 
  ((4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 2/21) ∧ 
  (Real.cos (2*α) + Real.sin (α + π/2) = (5 + 3 * Real.sqrt 13) / 13) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2534_253455


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2534_253428

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) :
  1 / x + 1 / y = 8 / 75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2534_253428


namespace NUMINAMATH_CALUDE_committee_selection_theorem_l2534_253484

/-- The number of candidates nominated for the committee -/
def total_candidates : ℕ := 20

/-- The number of candidates who have previously served on the committee -/
def past_members : ℕ := 9

/-- The number of positions available in the new committee -/
def committee_size : ℕ := 6

/-- The number of ways to select the committee with at least one past member -/
def selections_with_past_member : ℕ := 38298

theorem committee_selection_theorem :
  (Nat.choose total_candidates committee_size) - 
  (Nat.choose (total_candidates - past_members) committee_size) = 
  selections_with_past_member :=
sorry

end NUMINAMATH_CALUDE_committee_selection_theorem_l2534_253484


namespace NUMINAMATH_CALUDE_factorization_problem_triangle_shape_l2534_253476

-- Problem 1
theorem factorization_problem (a b : ℝ) :
  a^2 - 6*a*b + 9*b^2 - 36 = (a - 3*b - 6) * (a - 3*b + 6) := by sorry

-- Problem 2
theorem triangle_shape (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq : a^2 + c^2 + 2*b^2 - 2*a*b - 2*b*c = 0) :
  a = b ∧ b = c := by sorry

end NUMINAMATH_CALUDE_factorization_problem_triangle_shape_l2534_253476


namespace NUMINAMATH_CALUDE_valid_words_count_l2534_253483

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum length of a word -/
def max_word_length : ℕ := 5

/-- The number of words of length n that do not contain the letter A -/
def words_without_a (n : ℕ) : ℕ := (alphabet_size - 1) ^ n

/-- The total number of possible words of length n -/
def total_words (n : ℕ) : ℕ := alphabet_size ^ n

/-- The number of words of length n that contain the letter A at least once -/
def words_with_a (n : ℕ) : ℕ := total_words n - words_without_a n

/-- The total number of valid words -/
def total_valid_words : ℕ :=
  words_with_a 1 + words_with_a 2 + words_with_a 3 + words_with_a 4 + words_with_a 5

theorem valid_words_count : total_valid_words = 1863701 := by
  sorry

end NUMINAMATH_CALUDE_valid_words_count_l2534_253483


namespace NUMINAMATH_CALUDE_nail_polish_drying_time_l2534_253465

theorem nail_polish_drying_time (total_time color_coat_time top_coat_time : ℕ) 
  (h1 : total_time = 13)
  (h2 : color_coat_time = 3)
  (h3 : top_coat_time = 5) :
  total_time - (2 * color_coat_time + top_coat_time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nail_polish_drying_time_l2534_253465


namespace NUMINAMATH_CALUDE_gcd_459_357_l2534_253407

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2534_253407


namespace NUMINAMATH_CALUDE_percentage_problem_l2534_253416

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 1280 = (20 / 100) * 650 + 190 → P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2534_253416


namespace NUMINAMATH_CALUDE_engagement_treats_ratio_l2534_253473

def total_value : ℕ := 158000
def hotel_cost_per_night : ℕ := 4000
def nights_stayed : ℕ := 2
def car_value : ℕ := 30000

theorem engagement_treats_ratio :
  let hotel_total := hotel_cost_per_night * nights_stayed
  let non_house_total := hotel_total + car_value
  let house_value := total_value - non_house_total
  house_value / car_value = 4 := by
sorry

end NUMINAMATH_CALUDE_engagement_treats_ratio_l2534_253473


namespace NUMINAMATH_CALUDE_six_ronna_grams_scientific_notation_l2534_253489

/-- Represents the number of zeros after a number for the 'ronna' prefix --/
def ronna_zeros : ℕ := 27

/-- Converts a number with the 'ronna' prefix to its scientific notation --/
def ronna_to_scientific (n : ℝ) : ℝ := n * (10 ^ ronna_zeros)

/-- Theorem stating that 6 ronna grams is equal to 6 × 10^27 grams --/
theorem six_ronna_grams_scientific_notation :
  ronna_to_scientific 6 = 6 * (10 ^ 27) := by sorry

end NUMINAMATH_CALUDE_six_ronna_grams_scientific_notation_l2534_253489


namespace NUMINAMATH_CALUDE_total_days_on_orbius5_l2534_253490

/-- Definition of the Orbius-5 calendar system -/
structure Orbius5Calendar where
  daysPerYear : Nat := 250
  regularSeasonDays : Nat := 49
  leapSeasonDays : Nat := 51
  regularSeasonsPerYear : Nat := 2
  leapSeasonsPerYear : Nat := 3
  cycleYears : Nat := 10

/-- Definition of the astronaut's visits -/
structure AstronautVisits where
  firstVisitRegularSeasons : Nat := 1
  secondVisitRegularSeasons : Nat := 2
  secondVisitLeapSeasons : Nat := 3
  thirdVisitYears : Nat := 3
  fourthVisitCycles : Nat := 1

/-- Function to calculate total days spent on Orbius-5 -/
def totalDaysOnOrbius5 (calendar : Orbius5Calendar) (visits : AstronautVisits) : Nat :=
  sorry

/-- Theorem stating the total days spent on Orbius-5 -/
theorem total_days_on_orbius5 (calendar : Orbius5Calendar) (visits : AstronautVisits) :
  totalDaysOnOrbius5 calendar visits = 3578 := by
  sorry

end NUMINAMATH_CALUDE_total_days_on_orbius5_l2534_253490


namespace NUMINAMATH_CALUDE_not_perfect_square_l2534_253466

theorem not_perfect_square : 
  (∃ x : ℕ, 6^2040 = x^2) ∧ 
  (∀ y : ℕ, 7^2041 ≠ y^2) ∧ 
  (∃ z : ℕ, 8^2042 = z^2) ∧ 
  (∃ w : ℕ, 9^2043 = w^2) ∧ 
  (∃ v : ℕ, 10^2044 = v^2) :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2534_253466


namespace NUMINAMATH_CALUDE_inequality_proof_largest_constant_l2534_253421

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≤ (Real.sqrt 6 / 2) * Real.sqrt (x + y + z) :=
sorry

theorem largest_constant :
  ∀ k, (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
    x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≤ k * Real.sqrt (x + y + z)) →
  k ≤ Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_largest_constant_l2534_253421


namespace NUMINAMATH_CALUDE_max_value_is_110003_l2534_253481

/-- The set of given integers --/
def given_integers : Finset ℤ := {100004, 110003, 102002, 100301, 100041}

/-- Theorem stating that 110003 is the maximum value in the given set of integers --/
theorem max_value_is_110003 : 
  ∀ x ∈ given_integers, x ≤ 110003 ∧ 110003 ∈ given_integers := by
  sorry

#check max_value_is_110003

end NUMINAMATH_CALUDE_max_value_is_110003_l2534_253481


namespace NUMINAMATH_CALUDE_desired_depth_calculation_desired_depth_is_50_l2534_253425

/-- Calculates the desired depth to be dug given the initial and changed conditions -/
theorem desired_depth_calculation (initial_men : ℕ) (initial_hours : ℕ) (initial_depth : ℕ) 
  (extra_men : ℕ) (new_hours : ℕ) : ℕ :=
  let total_men : ℕ := initial_men + extra_men
  let initial_man_hours : ℕ := initial_men * initial_hours
  let new_man_hours : ℕ := total_men * new_hours
  let desired_depth : ℕ := (new_man_hours * initial_depth) / initial_man_hours
  desired_depth

/-- Proves that the desired depth to be dug is 50 meters -/
theorem desired_depth_is_50 : 
  desired_depth_calculation 45 8 30 55 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_desired_depth_calculation_desired_depth_is_50_l2534_253425


namespace NUMINAMATH_CALUDE_cubic_sum_identity_l2534_253477

theorem cubic_sum_identity 
  (x y z a b c : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^3 + y^3 + z^3 = (a^3 + b^3 + c^3) / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_identity_l2534_253477


namespace NUMINAMATH_CALUDE_courtyard_path_ratio_l2534_253401

theorem courtyard_path_ratio :
  ∀ (t p : ℝ),
  t > 0 →
  p > 0 →
  (400 * t^2) / (400 * (t + 2*p)^2) = 1/4 →
  p/t = 1/2 := by
sorry

end NUMINAMATH_CALUDE_courtyard_path_ratio_l2534_253401


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2534_253458

theorem arithmetic_calculation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2534_253458


namespace NUMINAMATH_CALUDE_tom_read_six_books_in_june_l2534_253430

/-- The number of books Tom read in May -/
def books_may : ℕ := 2

/-- The number of books Tom read in July -/
def books_july : ℕ := 10

/-- The total number of books Tom read -/
def total_books : ℕ := 18

/-- The number of books Tom read in June -/
def books_june : ℕ := total_books - (books_may + books_july)

theorem tom_read_six_books_in_june : books_june = 6 := by
  sorry

end NUMINAMATH_CALUDE_tom_read_six_books_in_june_l2534_253430


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2534_253435

-- Define the two line equations
def line1 (x y : ℝ) : Prop := 3 * y = -2 * x + 6
def line2 (x y : ℝ) : Prop := 4 * y = 7 * x - 8

-- Define the intersection point
def intersection_point : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem intersection_point_is_unique :
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) ∧
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = intersection_point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2534_253435


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2534_253475

theorem polynomial_evaluation : 
  ∃ (x : ℝ), x > 0 ∧ x^2 - 3*x - 10 = 0 ∧ x^4 - 3*x^3 + 2*x^2 + 5*x - 7 = 318 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2534_253475


namespace NUMINAMATH_CALUDE_container_problem_l2534_253441

theorem container_problem :
  ∃! (x y : ℕ), 130 * x + 160 * y = 3000 ∧ x = 12 ∧ y = 9 :=
by sorry

end NUMINAMATH_CALUDE_container_problem_l2534_253441


namespace NUMINAMATH_CALUDE_greatest_consecutive_even_sum_180_l2534_253445

/-- The sum of n consecutive even integers starting from 2a is n(2a + n - 1) -/
def sumConsecutiveEvenIntegers (n : ℕ) (a : ℤ) : ℤ := n * (2 * a + n - 1)

/-- 45 is the greatest number of consecutive even integers whose sum is 180 -/
theorem greatest_consecutive_even_sum_180 :
  ∀ n : ℕ, n > 45 → ¬∃ a : ℤ, sumConsecutiveEvenIntegers n a = 180 ∧
  ∃ a : ℤ, sumConsecutiveEvenIntegers 45 a = 180 :=
by sorry

#check greatest_consecutive_even_sum_180

end NUMINAMATH_CALUDE_greatest_consecutive_even_sum_180_l2534_253445


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l2534_253497

theorem log_equality_implies_golden_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.log a / Real.log 4 = Real.log b / Real.log 6 ∧ 
       Real.log a / Real.log 4 = Real.log (a + b) / Real.log 9) : 
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l2534_253497


namespace NUMINAMATH_CALUDE_refrigerator_temperature_l2534_253494

/-- Given an initial temperature, a temperature decrease rate, and elapsed time,
    calculate the final temperature inside a refrigerator. -/
def final_temperature (initial_temp : ℝ) (decrease_rate : ℝ) (elapsed_time : ℝ) : ℝ :=
  initial_temp - decrease_rate * elapsed_time

/-- Theorem stating that under the given conditions, the final temperature is -8°C. -/
theorem refrigerator_temperature : 
  final_temperature 12 5 4 = -8 := by
  sorry

#eval final_temperature 12 5 4

end NUMINAMATH_CALUDE_refrigerator_temperature_l2534_253494


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2534_253459

/-- An arithmetic sequence with specific terms -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a7_eq : a 7 = -2
  a20_eq : a 20 = -28

/-- The general term of the arithmetic sequence -/
def generalTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  14 - 2 * n

/-- The sum of the first n terms of the arithmetic sequence -/
def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧
  (∃ n, sumFirstN seq n = 42 ∧ ∀ m, sumFirstN seq m ≤ 42) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2534_253459


namespace NUMINAMATH_CALUDE_stone_distribution_fractions_l2534_253422

/-- Number of indistinguishable stones -/
def n : ℕ := 12

/-- Number of distinguishable boxes -/
def k : ℕ := 4

/-- Total number of ways to distribute n stones among k boxes -/
def total_distributions : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Number of ways to distribute stones with even number in each box -/
def even_distributions : ℕ := Nat.choose ((n / 2) + k - 1) (k - 1)

/-- Number of ways to distribute stones with odd number in each box -/
def odd_distributions : ℕ := Nat.choose ((n - k) / 2 + k - 1) (k - 1)

theorem stone_distribution_fractions :
  (even_distributions : ℚ) / total_distributions = 12 / 65 ∧
  (odd_distributions : ℚ) / total_distributions = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_stone_distribution_fractions_l2534_253422


namespace NUMINAMATH_CALUDE_inequality_solution_l2534_253413

noncomputable def solution_set : Set ℝ :=
  { x | x ∈ Set.Ioo (-4) (-14/3) ∪ Set.Ioi (6 + 3 * Real.sqrt 2) }

theorem inequality_solution :
  { x : ℝ | (2*x + 3) / (x + 4) > (5*x + 6) / (3*x + 14) } = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2534_253413


namespace NUMINAMATH_CALUDE_average_birds_per_site_l2534_253409

-- Define the data for each day
def monday_sites : ℕ := 5
def monday_avg : ℕ := 7
def tuesday_sites : ℕ := 5
def tuesday_avg : ℕ := 5
def wednesday_sites : ℕ := 10
def wednesday_avg : ℕ := 8

-- Define the total number of sites
def total_sites : ℕ := monday_sites + tuesday_sites + wednesday_sites

-- Define the total number of birds
def total_birds : ℕ := monday_sites * monday_avg + tuesday_sites * tuesday_avg + wednesday_sites * wednesday_avg

-- Theorem to prove
theorem average_birds_per_site :
  total_birds / total_sites = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_birds_per_site_l2534_253409


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2534_253469

theorem decimal_to_fraction : 
  (1.45 : ℚ) = 29 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2534_253469


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l2534_253405

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → perpendicularPlanes α β := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l2534_253405


namespace NUMINAMATH_CALUDE_interval_of_increase_l2534_253438

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 5*x + 6) / Real.log (1/4)

def domain (x : ℝ) : Prop := x < 2 ∨ x > 3

theorem interval_of_increase :
  ∀ x y, domain x → domain y → x < y → x < 2 → f x > f y := by sorry

end NUMINAMATH_CALUDE_interval_of_increase_l2534_253438


namespace NUMINAMATH_CALUDE_outfits_count_l2534_253406

/-- The number of different outfits that can be created given a specific number of shirts, pants, and ties. --/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) : ℕ :=
  shirts * pants * (ties + 1)

/-- Theorem stating that with 8 shirts, 5 pants, and 6 ties, the number of possible outfits is 280. --/
theorem outfits_count : number_of_outfits 8 5 6 = 280 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2534_253406


namespace NUMINAMATH_CALUDE_binary_101101_is_45_l2534_253429

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_is_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_is_45_l2534_253429


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2534_253456

/-- Represents the problem of a train crossing a platform --/
structure TrainProblem where
  train_speed_kmph : ℝ
  train_speed_ms : ℝ
  platform_length : ℝ
  time_to_cross_man : ℝ

/-- The theorem stating the time taken for the train to cross the platform --/
theorem train_platform_crossing_time (p : TrainProblem)
  (h1 : p.train_speed_kmph = 72)
  (h2 : p.train_speed_ms = p.train_speed_kmph / 3.6)
  (h3 : p.platform_length = 300)
  (h4 : p.time_to_cross_man = 15)
  : p.train_speed_ms * p.time_to_cross_man + p.platform_length = p.train_speed_ms * 30 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2534_253456


namespace NUMINAMATH_CALUDE_tangent_lines_parallel_to_4x_minus_1_l2534_253415

/-- The curve function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_lines_parallel_to_4x_minus_1 :
  ∃! (a b : ℝ), 
    (∃ (x : ℝ), f' x = 4 ∧ 
      (∀ y : ℝ, y = 4 * x + a ↔ y - f x = f' x * (y - x))) ∧
    (∃ (x : ℝ), f' x = 4 ∧ 
      (∀ y : ℝ, y = 4 * x + b ↔ y - f x = f' x * (y - x))) ∧
    a ≠ b ∧ 
    ({a, b} : Set ℝ) = {-4, 0} :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_parallel_to_4x_minus_1_l2534_253415


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2534_253450

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_2 + a_3 = 15 and a_3 + a_4 = 20,
    prove that a_4 + a_5 = 25. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum1 : a 2 + a 3 = 15)
    (h_sum2 : a 3 + a 4 = 20) :
  a 4 + a 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2534_253450


namespace NUMINAMATH_CALUDE_triangle_area_l2534_253423

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c^2 = (a - b)^2 + 6 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2534_253423


namespace NUMINAMATH_CALUDE_x_equals_two_l2534_253432

/-- The sum of digits for all four-digit numbers formed by 1, 4, 5, and x -/
def sumOfDigits (x : ℕ) : ℕ :=
  if x = 0 then
    24 * (1 + 4 + 5)
  else
    24 * (1 + 4 + 5 + x)

/-- Theorem stating that x must be 2 given the conditions -/
theorem x_equals_two :
  ∃! x : ℕ, x ≤ 9 ∧ sumOfDigits x = 288 :=
sorry

end NUMINAMATH_CALUDE_x_equals_two_l2534_253432


namespace NUMINAMATH_CALUDE_tax_revenue_change_l2534_253493

theorem tax_revenue_change (T C : ℝ) (T_new C_new R_new : ℝ) : 
  T_new = T * 0.9 →
  C_new = C * 1.1 →
  R_new = T_new * C_new →
  R_new = T * C * 0.99 := by
sorry

end NUMINAMATH_CALUDE_tax_revenue_change_l2534_253493


namespace NUMINAMATH_CALUDE_complex_arithmetic_proof_l2534_253464

theorem complex_arithmetic_proof :
  let A : ℂ := 3 + 2*Complex.I
  let B : ℂ := -5
  let C : ℂ := 2*Complex.I
  let D : ℂ := 1 + 3*Complex.I
  A - B + C - D = 7 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_proof_l2534_253464


namespace NUMINAMATH_CALUDE_robert_ride_time_l2534_253498

/-- The time taken for Robert to ride along a semicircular path on a highway section -/
theorem robert_ride_time :
  let highway_length : ℝ := 1 -- mile
  let highway_width : ℝ := 40 -- feet
  let robert_speed : ℝ := 5 -- miles per hour
  let feet_per_mile : ℝ := 5280
  let path_shape := Semicircle
  let time_taken := 
    (highway_length * feet_per_mile / highway_width) * (π * highway_width / 2) / 
    (robert_speed * feet_per_mile)
  time_taken = π / 10
  := by sorry

end NUMINAMATH_CALUDE_robert_ride_time_l2534_253498


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l2534_253400

/-- Given a circular arrangement of n people, this function calculates the clockwise distance between two positions -/
def clockwise_distance (n : ℕ) (a b : ℕ) : ℕ :=
  (b - a + n) % n

/-- Given a circular arrangement of n people, this function calculates the counterclockwise distance between two positions -/
def counterclockwise_distance (n : ℕ) (a b : ℕ) : ℕ :=
  (a - b + n) % n

theorem circular_seating_arrangement (n : ℕ) (h1 : n > 31) :
  clockwise_distance n 31 7 = counterclockwise_distance n 31 14 → n = 41 := by
  sorry

#eval clockwise_distance 41 31 7
#eval counterclockwise_distance 41 31 14

end NUMINAMATH_CALUDE_circular_seating_arrangement_l2534_253400


namespace NUMINAMATH_CALUDE_dolls_count_l2534_253467

/-- The total number of toys given -/
def total_toys : ℕ := 403

/-- The number of toy cars given to boys -/
def cars_to_boys : ℕ := 134

/-- The number of dolls given to girls -/
def dolls_to_girls : ℕ := total_toys - cars_to_boys

theorem dolls_count : dolls_to_girls = 269 := by
  sorry

end NUMINAMATH_CALUDE_dolls_count_l2534_253467


namespace NUMINAMATH_CALUDE_star_value_of_a_l2534_253417

-- Define the star operation
def star (a b : ℝ) : ℝ := 3 * a - b^2

-- State the theorem
theorem star_value_of_a : ∃ a : ℝ, star a 4 = 14 ∧ a = 10 := by
  sorry

end NUMINAMATH_CALUDE_star_value_of_a_l2534_253417


namespace NUMINAMATH_CALUDE_solution_set_fraction_inequality_l2534_253460

theorem solution_set_fraction_inequality :
  {x : ℝ | (x - 2) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_fraction_inequality_l2534_253460


namespace NUMINAMATH_CALUDE_monthly_salary_calculation_l2534_253451

/-- Proves that a man's monthly salary is 5750 Rs. given the specified conditions -/
theorem monthly_salary_calculation (savings_rate : ℝ) (expense_increase : ℝ) (new_savings : ℝ) : 
  savings_rate = 0.20 →
  expense_increase = 0.20 →
  new_savings = 230 →
  ∃ (salary : ℝ), salary = 5750 ∧ 
    (1 - savings_rate - expense_increase * (1 - savings_rate)) * salary = new_savings :=
by sorry

end NUMINAMATH_CALUDE_monthly_salary_calculation_l2534_253451


namespace NUMINAMATH_CALUDE_same_combination_probability_l2534_253439

/-- Represents the number of candies of each color in the jar -/
structure JarContents where
  red : Nat
  blue : Nat
  green : Nat

/-- Calculates the probability of two people picking the same color combination -/
def probability_same_combination (jar : JarContents) : ℚ :=
  sorry

/-- The main theorem stating the probability for the given jar contents -/
theorem same_combination_probability :
  let jar : JarContents := { red := 12, blue := 12, green := 6 }
  probability_same_combination jar = 2783 / 847525 := by
  sorry

end NUMINAMATH_CALUDE_same_combination_probability_l2534_253439


namespace NUMINAMATH_CALUDE_hoseok_has_least_paper_l2534_253482

def jungkook_paper : ℕ := 10
def hoseok_paper : ℕ := 7
def seokjin_paper : ℕ := jungkook_paper - 2

theorem hoseok_has_least_paper : 
  hoseok_paper < jungkook_paper ∧ hoseok_paper < seokjin_paper := by
sorry

end NUMINAMATH_CALUDE_hoseok_has_least_paper_l2534_253482


namespace NUMINAMATH_CALUDE_rachel_essay_time_l2534_253443

/-- Represents the time spent on various activities of essay writing -/
structure EssayTime where
  research_time : ℕ  -- in minutes
  writing_rate : ℕ  -- pages per 30 minutes
  total_pages : ℕ
  editing_time : ℕ  -- in minutes

/-- Calculates the total time spent on an essay in hours -/
def total_essay_time (et : EssayTime) : ℚ :=
  let writing_time := (et.total_pages * 30) / 60  -- convert to hours
  let other_time := (et.research_time + et.editing_time) / 60  -- convert to hours
  writing_time + other_time

/-- Theorem stating that Rachel's total essay time is 5 hours -/
theorem rachel_essay_time :
  let rachel_essay := EssayTime.mk 45 1 6 75
  total_essay_time rachel_essay = 5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_essay_time_l2534_253443


namespace NUMINAMATH_CALUDE_shares_to_buy_l2534_253491

def wife_weekly_savings : ℕ := 100
def husband_monthly_savings : ℕ := 225
def weeks_per_month : ℕ := 4
def savings_period_months : ℕ := 4
def stock_price : ℕ := 50

def total_savings : ℕ :=
  (wife_weekly_savings * weeks_per_month + husband_monthly_savings) * savings_period_months

def investment_amount : ℕ := total_savings / 2

theorem shares_to_buy : investment_amount / stock_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_shares_to_buy_l2534_253491


namespace NUMINAMATH_CALUDE_tan_alpha_implies_c_equals_five_l2534_253452

theorem tan_alpha_implies_c_equals_five (α : Real) (c : Real) 
  (h1 : Real.tan α = -1/2) 
  (h2 : c = (2 * Real.cos α - Real.sin α) / (Real.sin α + Real.cos α)) : 
  c = 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_implies_c_equals_five_l2534_253452


namespace NUMINAMATH_CALUDE_jimin_remaining_distance_l2534_253403

/-- Calculates the remaining distance to travel given initial conditions. -/
def remaining_distance (speed : ℝ) (time : ℝ) (total_distance : ℝ) : ℝ :=
  total_distance - speed * time

/-- Proves that given the initial conditions, the remaining distance is 180 km. -/
theorem jimin_remaining_distance :
  remaining_distance 60 2 300 = 180 := by
  sorry

end NUMINAMATH_CALUDE_jimin_remaining_distance_l2534_253403


namespace NUMINAMATH_CALUDE_parallelogram_angle_difference_l2534_253447

theorem parallelogram_angle_difference (a b : ℝ) : 
  a = 65 → -- smaller angle is 65 degrees
  a + b = 180 → -- adjacent angles in a parallelogram are supplementary
  b - a = 50 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_angle_difference_l2534_253447


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2534_253485

theorem necessary_but_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x^2 + y^2 ≥ 2) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2534_253485


namespace NUMINAMATH_CALUDE_pizza_cost_is_9_60_l2534_253488

/-- The cost of a single box of pizza -/
def pizza_cost : ℝ := sorry

/-- The cost of a single can of soft drink -/
def soft_drink_cost : ℝ := 2

/-- The cost of a single hamburger -/
def hamburger_cost : ℝ := 3

/-- The number of pizza boxes Robert buys -/
def robert_pizza_boxes : ℕ := 5

/-- The number of soft drink cans Robert buys -/
def robert_soft_drinks : ℕ := 10

/-- The number of hamburgers Teddy buys -/
def teddy_hamburgers : ℕ := 6

/-- The number of soft drink cans Teddy buys -/
def teddy_soft_drinks : ℕ := 10

/-- The total amount spent by Robert and Teddy -/
def total_spent : ℝ := 106

theorem pizza_cost_is_9_60 :
  pizza_cost = 9.60 ∧
  (robert_pizza_boxes : ℝ) * pizza_cost +
  (robert_soft_drinks : ℝ) * soft_drink_cost +
  (teddy_hamburgers : ℝ) * hamburger_cost +
  (teddy_soft_drinks : ℝ) * soft_drink_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_pizza_cost_is_9_60_l2534_253488


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2534_253411

/-- Given a line L1 with equation 3x - y = 6 and a point P (-2, 3),
    prove that the line L2 with equation y = 3x + 9 is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - y = 6
  let P : ℝ × ℝ := (-2, 3)
  let L2 : ℝ → ℝ → Prop := λ x y => y = 3 * x + 9
  (∀ x y, L1 x y ↔ y = 3 * x - 6) →  -- L1 in slope-intercept form
  L2 P.1 P.2 ∧                      -- L2 passes through P
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → y₂ - y₁ = 3 * (x₂ - x₁)) →  -- Slope of L1 is 3
  (∀ x₁ y₁ x₂ y₂, L2 x₁ y₁ → L2 x₂ y₂ → y₂ - y₁ = 3 * (x₂ - x₁))    -- Slope of L2 is 3
  :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2534_253411


namespace NUMINAMATH_CALUDE_carnation_percentage_is_67_point_5_l2534_253419

/-- Represents a flower display with pink and red flowers, either roses or carnations -/
structure FlowerDisplay where
  total : ℝ
  pink_ratio : ℝ
  red_carnation_ratio : ℝ
  pink_rose_ratio : ℝ

/-- Calculates the percentage of carnations in the flower display -/
def carnation_percentage (display : FlowerDisplay) : ℝ :=
  let red_ratio := 1 - display.pink_ratio
  let pink_carnation_ratio := display.pink_ratio * (1 - display.pink_rose_ratio)
  let red_carnation_ratio := red_ratio * display.red_carnation_ratio
  (pink_carnation_ratio + red_carnation_ratio) * 100

/-- Theorem stating that under given conditions, 67.5% of flowers are carnations -/
theorem carnation_percentage_is_67_point_5
  (display : FlowerDisplay)
  (h_pink_ratio : display.pink_ratio = 7/10)
  (h_red_carnation_ratio : display.red_carnation_ratio = 1/2)
  (h_pink_rose_ratio : display.pink_rose_ratio = 1/4) :
  carnation_percentage display = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_carnation_percentage_is_67_point_5_l2534_253419


namespace NUMINAMATH_CALUDE_mango_count_l2534_253418

/-- The number of mangoes in all boxes -/
def total_mangoes (boxes : ℕ) (dozen_per_box : ℕ) : ℕ :=
  boxes * dozen_per_box * 12

/-- Proof that there are 4320 mangoes in 36 boxes with 10 dozen mangoes each -/
theorem mango_count : total_mangoes 36 10 = 4320 := by
  sorry

end NUMINAMATH_CALUDE_mango_count_l2534_253418


namespace NUMINAMATH_CALUDE_domain_equivalence_l2534_253487

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_x_plus_1 (f : ℝ → ℝ) : Set ℝ := {x | -2 < x ∧ x < 0}

-- Define the domain of f(2x-1)
def domain_f_2x_minus_1 (f : ℝ → ℝ) : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem statement
theorem domain_equivalence (f : ℝ → ℝ) :
  (∀ x, x ∈ domain_f_x_plus_1 f ↔ f (x + 1) ≠ 0) →
  (∀ x, x ∈ domain_f_2x_minus_1 f ↔ f (2 * x - 1) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_domain_equivalence_l2534_253487


namespace NUMINAMATH_CALUDE_roger_lawn_mowing_earnings_l2534_253402

theorem roger_lawn_mowing_earnings :
  ∀ (total_lawns : ℕ) (forgotten_lawns : ℕ) (total_earnings : ℕ),
    total_lawns = 14 →
    forgotten_lawns = 8 →
    total_earnings = 54 →
    (total_earnings : ℚ) / ((total_lawns - forgotten_lawns) : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_roger_lawn_mowing_earnings_l2534_253402


namespace NUMINAMATH_CALUDE_gravelling_cost_l2534_253468

/-- The cost of gravelling intersecting roads on a rectangular lawn. -/
theorem gravelling_cost 
  (lawn_length lawn_width road_width gravel_cost_per_sqm : ℝ)
  (h_lawn_length : lawn_length = 70)
  (h_lawn_width : lawn_width = 30)
  (h_road_width : road_width = 5)
  (h_gravel_cost : gravel_cost_per_sqm = 4) :
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * gravel_cost_per_sqm = 1900 :=
by sorry

end NUMINAMATH_CALUDE_gravelling_cost_l2534_253468


namespace NUMINAMATH_CALUDE_abby_and_damon_weight_l2534_253440

/-- The combined weight of Abby and Damon given the weights of other pairs -/
theorem abby_and_damon_weight
  (a b c d : ℝ)
  (h1 : a + b = 280)
  (h2 : b + c = 265)
  (h3 : c + d = 290)
  (h4 : b + d = 275) :
  a + d = 305 :=
by sorry

end NUMINAMATH_CALUDE_abby_and_damon_weight_l2534_253440


namespace NUMINAMATH_CALUDE_square_sum_plus_double_sum_squares_l2534_253427

theorem square_sum_plus_double_sum_squares : (5 + 7)^2 + (5^2 + 7^2) * 2 = 292 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_plus_double_sum_squares_l2534_253427


namespace NUMINAMATH_CALUDE_joans_grilled_cheese_sandwiches_l2534_253453

/-- Calculates the number of grilled cheese sandwiches Joan makes given the conditions -/
theorem joans_grilled_cheese_sandwiches 
  (total_cheese : ℕ) 
  (ham_sandwiches : ℕ) 
  (cheese_per_ham : ℕ) 
  (cheese_per_grilled : ℕ) 
  (h1 : total_cheese = 50)
  (h2 : ham_sandwiches = 10)
  (h3 : cheese_per_ham = 2)
  (h4 : cheese_per_grilled = 3) :
  (total_cheese - ham_sandwiches * cheese_per_ham) / cheese_per_grilled = 10 := by
  sorry

end NUMINAMATH_CALUDE_joans_grilled_cheese_sandwiches_l2534_253453


namespace NUMINAMATH_CALUDE_initial_group_size_l2534_253463

/-- The number of men in the initial group -/
def initial_men_count : ℕ := sorry

/-- The average age increase when two women replace two men -/
def avg_age_increase : ℕ := 6

/-- The age of the first replaced man -/
def man1_age : ℕ := 18

/-- The age of the second replaced man -/
def man2_age : ℕ := 22

/-- The average age of the women -/
def women_avg_age : ℕ := 50

theorem initial_group_size : initial_men_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_group_size_l2534_253463


namespace NUMINAMATH_CALUDE_fountain_pen_price_l2534_253471

theorem fountain_pen_price (num_fountain_pens : ℕ) (num_mechanical_pencils : ℕ)
  (total_cost : ℚ) (avg_price_mechanical_pencil : ℚ) :
  num_fountain_pens = 450 →
  num_mechanical_pencils = 3750 →
  total_cost = 11250 →
  avg_price_mechanical_pencil = 2.25 →
  (total_cost - (num_mechanical_pencils : ℚ) * avg_price_mechanical_pencil) / (num_fountain_pens : ℚ) = 6.25 := by
sorry

end NUMINAMATH_CALUDE_fountain_pen_price_l2534_253471


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2534_253448

-- Define the curve
def x (t : ℝ) : ℝ := t^3 - t - 2
def y (t : ℝ) : ℝ := t^3 - t^2 - 9*t + 5

-- Define the self-intersection point
def intersection_point : ℝ × ℝ := (22, -4)

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ 
    x a = x b ∧ 
    y a = y b ∧ 
    (x a, y a) = intersection_point :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2534_253448


namespace NUMINAMATH_CALUDE_probability_of_specific_dice_outcome_l2534_253479

def num_dice : ℕ := 5
def num_sides : ℕ := 5
def target_number : ℕ := 3
def num_target : ℕ := 2

theorem probability_of_specific_dice_outcome :
  (num_dice.choose num_target *
   (1 / num_sides) ^ num_target *
   ((num_sides - 1) / num_sides) ^ (num_dice - num_target) : ℚ) =
  640 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_dice_outcome_l2534_253479


namespace NUMINAMATH_CALUDE_als_original_portion_l2534_253446

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1000 →
  a - 100 + 2*b + 2*c = 1500 →
  a = 400 :=
by sorry

end NUMINAMATH_CALUDE_als_original_portion_l2534_253446


namespace NUMINAMATH_CALUDE_four_point_theorem_l2534_253449

/-- Given four points A, B, C, D in a plane, if for any point P the inequality 
    PA + PD ≥ PB + PC holds, then B and C lie on the segment AD and AB = CD. -/
theorem four_point_theorem (A B C D : EuclideanSpace ℝ (Fin 2)) :
  (∀ P : EuclideanSpace ℝ (Fin 2), dist P A + dist P D ≥ dist P B + dist P C) →
  (∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ 
    B = (1 - t₁) • A + t₁ • D ∧ 
    C = (1 - t₂) • A + t₂ • D) ∧
  dist A B = dist C D := by
  sorry

end NUMINAMATH_CALUDE_four_point_theorem_l2534_253449


namespace NUMINAMATH_CALUDE_sum_of_digits_5mul_permutation_l2534_253480

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if two natural numbers are permutations of each other's digits -/
def isDigitPermutation (a b : ℕ) : Prop := sorry

/-- Theorem: If A is a permutation of B's digits, then sum of digits of 5A equals sum of digits of 5B -/
theorem sum_of_digits_5mul_permutation (A B : ℕ) :
  isDigitPermutation A B → sumOfDigits (5 * A) = sumOfDigits (5 * B) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_5mul_permutation_l2534_253480


namespace NUMINAMATH_CALUDE_max_value_implies_a_range_l2534_253461

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 8

-- State the theorem
theorem max_value_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, f x ≤ f a) →
  a ∈ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_range_l2534_253461


namespace NUMINAMATH_CALUDE_square_side_length_l2534_253457

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s > 0 ∧ s * s = d * d / 2 ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2534_253457


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2534_253420

theorem system_solution_ratio (x y a b : ℝ) (h1 : 4*x - 2*y = a) (h2 : 6*y - 12*x = b) (h3 : b ≠ 0) :
  a / b = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2534_253420


namespace NUMINAMATH_CALUDE_soccer_team_math_enrollment_l2534_253408

theorem soccer_team_math_enrollment (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 25 →
  physics_players = 15 →
  both_subjects = 6 →
  ∃ (math_players : ℕ), math_players = 16 ∧ 
    total_players = physics_players + math_players - both_subjects :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_team_math_enrollment_l2534_253408


namespace NUMINAMATH_CALUDE_maria_trip_distance_l2534_253404

/-- Given a total trip distance of 400 miles, with stops at 1/2 of the total distance
    and 1/4 of the remaining distance after the first stop, the distance traveled
    after the second stop is 150 miles. -/
theorem maria_trip_distance : 
  let total_distance : ℝ := 400
  let first_stop_fraction : ℝ := 1/2
  let second_stop_fraction : ℝ := 1/4
  let distance_to_first_stop := total_distance * first_stop_fraction
  let remaining_after_first_stop := total_distance - distance_to_first_stop
  let distance_to_second_stop := remaining_after_first_stop * second_stop_fraction
  let distance_after_second_stop := remaining_after_first_stop - distance_to_second_stop
  distance_after_second_stop = 150 := by
sorry

end NUMINAMATH_CALUDE_maria_trip_distance_l2534_253404


namespace NUMINAMATH_CALUDE_train_length_l2534_253426

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 36 → ∃ (length_m : ℝ), 
  (abs (length_m - 600.12) < 0.01) ∧ (length_m = speed_kmh * (5/18) * time_s) := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2534_253426


namespace NUMINAMATH_CALUDE_jill_cookie_sales_l2534_253424

def cookie_sales (goal : ℕ) (first second third fourth fifth : ℕ) : Prop :=
  let total_sold := first + second + third + fourth + fifth
  goal - total_sold = 75

theorem jill_cookie_sales :
  cookie_sales 150 5 20 10 30 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jill_cookie_sales_l2534_253424


namespace NUMINAMATH_CALUDE_converse_statement_l2534_253499

theorem converse_statement (x : ℝ) : 
  (∀ x, x ≥ 1 → x^2 + 3*x - 2 ≥ 0) →
  (∀ x, x^2 + 3*x - 2 < 0 → x < 1) :=
by sorry

end NUMINAMATH_CALUDE_converse_statement_l2534_253499


namespace NUMINAMATH_CALUDE_runners_meet_again_l2534_253496

def track_length : ℝ := 600

def runner_speeds : List ℝ := [3.6, 4.2, 5.4, 6.0]

def meeting_time : ℝ := 1000

theorem runners_meet_again :
  ∀ (speed : ℝ), speed ∈ runner_speeds →
  ∃ (n : ℕ), speed * meeting_time = n * track_length :=
by sorry

end NUMINAMATH_CALUDE_runners_meet_again_l2534_253496


namespace NUMINAMATH_CALUDE_root_product_zero_l2534_253486

theorem root_product_zero (α β c : ℝ) : 
  (α^2 - 4*α + c = 0) → 
  (β^2 - 4*β + c = 0) → 
  ((-α)^2 + 4*(-α) - c = 0) → 
  α * β = 0 := by
sorry

end NUMINAMATH_CALUDE_root_product_zero_l2534_253486
