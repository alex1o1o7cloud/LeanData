import Mathlib

namespace NUMINAMATH_CALUDE_largest_even_number_with_sum_20_l2369_236922

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits in a natural number are different -/
def has_different_digits (n : ℕ) : Prop := sorry

/-- The theorem stating that 86420 is the largest even number with all different digits whose digits add up to 20 -/
theorem largest_even_number_with_sum_20 : 
  ∀ n : ℕ, 
    n % 2 = 0 ∧ 
    has_different_digits n ∧ 
    sum_of_digits n = 20 → 
    n ≤ 86420 := by sorry

end NUMINAMATH_CALUDE_largest_even_number_with_sum_20_l2369_236922


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2369_236917

theorem cube_volume_problem (a : ℝ) : 
  a > 0 →  -- Ensuring the side length is positive
  (a + 2) * a * (a - 2) = a^3 - 8 → 
  a^3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2369_236917


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l2369_236989

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem gcd_factorial_seven_eight : 
  Nat.gcd (factorial 7) (factorial 8) = factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l2369_236989


namespace NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l2369_236950

theorem triangle_area_with_cosine_root (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → b = 5 → 
  5 * cos_theta^2 - 7 * cos_theta - 6 = 0 →
  cos_theta ≤ 1 →
  (1/2) * a * b * Real.sqrt (1 - cos_theta^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l2369_236950


namespace NUMINAMATH_CALUDE_power_of_five_l2369_236997

theorem power_of_five (m : ℕ) : 5^m = 5 * 25^2 * 125^3 → m = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_l2369_236997


namespace NUMINAMATH_CALUDE_labourerPayCorrect_l2369_236915

/-- Calculates the total amount received by a labourer given the engagement conditions and absence -/
def labourerPay (totalDays : ℕ) (payRate : ℚ) (fineRate : ℚ) (absentDays : ℕ) : ℚ :=
  let workedDays := totalDays - absentDays
  let totalEarned := (workedDays : ℚ) * payRate
  let totalFine := (absentDays : ℚ) * fineRate
  totalEarned - totalFine

/-- The labourer's pay calculation is correct for the given conditions -/
theorem labourerPayCorrect :
  labourerPay 25 2 0.5 5 = 37.5 := by
  sorry

#eval labourerPay 25 2 0.5 5

end NUMINAMATH_CALUDE_labourerPayCorrect_l2369_236915


namespace NUMINAMATH_CALUDE_bowl_glass_pairings_l2369_236981

/-- The number of bowls -/
def num_bowls : ℕ := 5

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The number of unique bowl colors -/
def num_bowl_colors : ℕ := 5

/-- The number of unique glass colors -/
def num_glass_colors : ℕ := 3

/-- The number of red glasses -/
def num_red_glasses : ℕ := 2

/-- The number of blue glasses -/
def num_blue_glasses : ℕ := 2

/-- The number of yellow glasses -/
def num_yellow_glasses : ℕ := 1

theorem bowl_glass_pairings :
  num_bowls * num_glasses = 25 :=
sorry

end NUMINAMATH_CALUDE_bowl_glass_pairings_l2369_236981


namespace NUMINAMATH_CALUDE_santanas_brothers_birthdays_l2369_236924

/-- The number of Santana's brothers -/
def total_brothers : ℕ := 7

/-- The number of brothers with birthdays in March -/
def march_birthdays : ℕ := 3

/-- The number of brothers with birthdays in November -/
def november_birthdays : ℕ := 1

/-- The number of brothers with birthdays in December -/
def december_birthdays : ℕ := 2

/-- The difference in presents between the second and first half of the year -/
def present_difference : ℕ := 8

/-- The number of brothers with birthdays in October -/
def october_birthdays : ℕ := total_brothers - (march_birthdays + november_birthdays + december_birthdays)

theorem santanas_brothers_birthdays :
  october_birthdays = 1 :=
by sorry

end NUMINAMATH_CALUDE_santanas_brothers_birthdays_l2369_236924


namespace NUMINAMATH_CALUDE_paul_initial_books_l2369_236900

/-- Represents the number of books and pens Paul has -/
structure PaulsItems where
  books : ℕ
  pens : ℕ

/-- Represents the change in Paul's items after the garage sale -/
structure GarageSale where
  booksRemaining : ℕ
  pensRemaining : ℕ
  booksSold : ℕ

def initialItems : PaulsItems where
  books := 0  -- Unknown initial number of books
  pens := 55

def afterSale : GarageSale where
  booksRemaining := 66
  pensRemaining := 59
  booksSold := 42

theorem paul_initial_books :
  initialItems.books = afterSale.booksRemaining + afterSale.booksSold :=
by sorry

end NUMINAMATH_CALUDE_paul_initial_books_l2369_236900


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2369_236941

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 2 ↔ (m - 1) * x < Real.sqrt (4 * x - x^2)) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2369_236941


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_upper_bound_l2369_236947

open Real

theorem function_inequality_implies_a_upper_bound 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h_f : ∀ x, f x = x - a * x * log x) :
  (∃ x₀ ∈ Set.Icc (exp 1) (exp 2), f x₀ ≤ (1/4) * log x₀) →
  a ≤ 1 - 1 / (4 * exp 1) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_upper_bound_l2369_236947


namespace NUMINAMATH_CALUDE_x_equals_4n_l2369_236974

/-- Given that x is 3 times larger than n, and 2n + 3 is some percentage of 25, prove that x = 4n -/
theorem x_equals_4n (n x : ℝ) (p : ℝ) 
  (h1 : x = n + 3 * n) 
  (h2 : 2 * n + 3 = p / 100 * 25) : 
  x = 4 * n := by
sorry

end NUMINAMATH_CALUDE_x_equals_4n_l2369_236974


namespace NUMINAMATH_CALUDE_diamond_commutative_l2369_236945

-- Define the set T of all non-zero integers
def T : Set Int := {x : Int | x ≠ 0}

-- Define the binary operation ◇
def diamond (a b : T) : Int := 3 * a * b + a + b

-- Theorem statement
theorem diamond_commutative : ∀ (a b : T), diamond a b = diamond b a := by
  sorry

end NUMINAMATH_CALUDE_diamond_commutative_l2369_236945


namespace NUMINAMATH_CALUDE_solution_set_of_quadratic_inequality_l2369_236939

theorem solution_set_of_quadratic_inequality :
  ∀ x : ℝ, 3 * x^2 + 7 * x < 6 ↔ -3 < x ∧ x < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_quadratic_inequality_l2369_236939


namespace NUMINAMATH_CALUDE_three_x_squared_y_squared_l2369_236962

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 3*x^2*y^2 = 588 := by
  sorry

end NUMINAMATH_CALUDE_three_x_squared_y_squared_l2369_236962


namespace NUMINAMATH_CALUDE_rectangle_locus_l2369_236946

/-- Given a rectangle with length l and width w, and a fixed number b,
    this theorem states that the locus of all points P(x, y) in the plane of the rectangle
    such that the sum of the squares of the distances from P to the four vertices
    of the rectangle equals b is a circle if and only if b > l^2 + w^2. -/
theorem rectangle_locus (l w b : ℝ) :
  (∃ (c : ℝ × ℝ) (r : ℝ),
    ∀ (x y : ℝ),
      (x - 0)^2 + (y - 0)^2 + (x - l)^2 + (y - 0)^2 +
      (x - l)^2 + (y - w)^2 + (x - 0)^2 + (y - w)^2 = b ↔
      (x - c.1)^2 + (y - c.2)^2 = r^2) ↔
  b > l^2 + w^2 := by sorry

end NUMINAMATH_CALUDE_rectangle_locus_l2369_236946


namespace NUMINAMATH_CALUDE_a_less_than_2_necessary_not_sufficient_l2369_236969

theorem a_less_than_2_necessary_not_sufficient :
  (∀ a : ℝ, a^2 < 2*a → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 2*a) :=
by sorry

end NUMINAMATH_CALUDE_a_less_than_2_necessary_not_sufficient_l2369_236969


namespace NUMINAMATH_CALUDE_multiple_compounds_same_weight_l2369_236943

/-- Represents a chemical compound -/
structure Compound where
  molecular_weight : ℕ

/-- Represents the set of all possible compounds -/
def AllCompounds : Set Compound := sorry

/-- The given molecular weight -/
def given_weight : ℕ := 391

/-- Compounds with the given molecular weight -/
def compounds_with_given_weight : Set Compound :=
  {c ∈ AllCompounds | c.molecular_weight = given_weight}

/-- Theorem stating that multiple compounds can have the same molecular weight -/
theorem multiple_compounds_same_weight :
  ∃ (c1 c2 : Compound), c1 ≠ c2 ∧ c1 ∈ compounds_with_given_weight ∧ c2 ∈ compounds_with_given_weight :=
sorry

end NUMINAMATH_CALUDE_multiple_compounds_same_weight_l2369_236943


namespace NUMINAMATH_CALUDE_final_digit_independent_of_sequence_l2369_236925

/-- Represents the count of each digit on the blackboard -/
structure DigitCount where
  zeros : Nat
  ones : Nat
  twos : Nat

/-- Represents a single step of the digit replacement operation -/
def replaceDigits (count : DigitCount) : DigitCount :=
  sorry

/-- Determines if the operation can continue (more than one digit type remains) -/
def canContinue (count : DigitCount) : Bool :=
  sorry

/-- Performs the digit replacement operations until only one digit type remains -/
def performOperations (initial : DigitCount) : Nat :=
  sorry

theorem final_digit_independent_of_sequence (initial : DigitCount) :
  ∀ (seq1 seq2 : List (DigitCount → DigitCount)),
    (seq1.foldl (fun acc f => f acc) initial).zeros +
    (seq1.foldl (fun acc f => f acc) initial).ones +
    (seq1.foldl (fun acc f => f acc) initial).twos = 1 →
    (seq2.foldl (fun acc f => f acc) initial).zeros +
    (seq2.foldl (fun acc f => f acc) initial).ones +
    (seq2.foldl (fun acc f => f acc) initial).twos = 1 →
    (seq1.foldl (fun acc f => f acc) initial) = (seq2.foldl (fun acc f => f acc) initial) :=
  sorry

end NUMINAMATH_CALUDE_final_digit_independent_of_sequence_l2369_236925


namespace NUMINAMATH_CALUDE_leaf_gusts_count_l2369_236986

/-- A gust of wind that moves a leaf -/
structure Gust where
  forward : ℕ
  backward : ℕ

/-- The problem of determining the number of gusts -/
def LeafProblem (g : Gust) (total_distance : ℕ) : Prop :=
  g.forward = 5 ∧ g.backward = 2 ∧ ∃ (n : ℕ), n * (g.forward - g.backward) = total_distance

theorem leaf_gusts_count :
  ∀ (g : Gust) (total_distance : ℕ),
    LeafProblem g total_distance →
    (∃ (n : ℕ), n * (g.forward - g.backward) = total_distance) →
    (∃ (n : ℕ), n = 11 ∧ n * (g.forward - g.backward) = total_distance) :=
by sorry

end NUMINAMATH_CALUDE_leaf_gusts_count_l2369_236986


namespace NUMINAMATH_CALUDE_not_square_sum_divisor_l2369_236942

theorem not_square_sum_divisor (n : ℕ) (d : ℕ) (h : d ∣ 2 * n^2) :
  ¬∃ (x : ℕ), n^2 + d = x^2 := by
sorry

end NUMINAMATH_CALUDE_not_square_sum_divisor_l2369_236942


namespace NUMINAMATH_CALUDE_inverse_of_i_minus_two_i_inv_l2369_236951

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem inverse_of_i_minus_two_i_inv (h : i^2 = -1) :
  (i - 2 * i⁻¹)⁻¹ = -i / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_i_minus_two_i_inv_l2369_236951


namespace NUMINAMATH_CALUDE_probability_of_drawing_specific_balls_l2369_236931

theorem probability_of_drawing_specific_balls (red white blue black : ℕ) : 
  red = 5 → white = 4 → blue = 3 → black = 6 →
  (red * white * blue : ℚ) / ((red + white + blue + black) * (red + white + blue + black - 1) * (red + white + blue + black - 2)) = 5 / 408 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_specific_balls_l2369_236931


namespace NUMINAMATH_CALUDE_expansion_coefficient_l2369_236944

/-- Given that in the expansion of (ax+1)(x+1/x)^6, the coefficient of x^3 is 30, prove that a = 2 -/
theorem expansion_coefficient (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 6 k) * a = 30 ∧ 6 - 2*k + 1 = 3) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l2369_236944


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_seven_l2369_236961

theorem consecutive_integers_around_sqrt_seven (a b : ℤ) : 
  a < Real.sqrt 7 ∧ Real.sqrt 7 < b ∧ b = a + 1 → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_seven_l2369_236961


namespace NUMINAMATH_CALUDE_derivative_of_even_is_odd_l2369_236963

/-- If a real-valued function is even, then its derivative is odd. -/
theorem derivative_of_even_is_odd (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_even : ∀ x, f (-x) = f x) :
  ∀ x, deriv f (-x) = -deriv f x := by sorry

end NUMINAMATH_CALUDE_derivative_of_even_is_odd_l2369_236963


namespace NUMINAMATH_CALUDE_smallest_perimeter_acute_triangle_arithmetic_sides_l2369_236932

theorem smallest_perimeter_acute_triangle_arithmetic_sides (a b c : ℝ) :
  -- Triangle ABC is acute
  0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2 →
  -- Side lengths form an arithmetic sequence
  b - a = c - b →
  -- Area is an integer
  (∃ n : ℕ, n = (1/4) * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))) →
  -- Perimeter is at least 18
  a + b + c ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_acute_triangle_arithmetic_sides_l2369_236932


namespace NUMINAMATH_CALUDE_chocolate_difference_l2369_236967

/-- The number of chocolates Nick has -/
def nick_chocolates : ℕ := 10

/-- The factor by which Alix's chocolates exceed Nick's -/
def alix_factor : ℕ := 3

/-- The number of chocolates mom took from Alix -/
def mom_took : ℕ := 5

/-- The number of chocolates Alix has after mom took some -/
def alix_chocolates : ℕ := alix_factor * nick_chocolates - mom_took

theorem chocolate_difference : alix_chocolates - nick_chocolates = 15 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_l2369_236967


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l2369_236983

/-- Given a cylinder with volume 72π cm³, prove that a cone with the same height and radius has a volume of 24π cm³ -/
theorem cone_volume_from_cylinder_volume (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 72 * π → (1/3) * π * r^2 * h = 24 * π := by
  sorry

#check cone_volume_from_cylinder_volume

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l2369_236983


namespace NUMINAMATH_CALUDE_minimum_additional_weeks_to_win_l2369_236920

def puppy_cost : ℕ := 1000
def weekly_prize : ℕ := 100
def additional_wins_needed : ℕ := 8

theorem minimum_additional_weeks_to_win (current_savings : ℕ) : 
  (current_savings + additional_wins_needed * weekly_prize = puppy_cost) → 
  additional_wins_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_minimum_additional_weeks_to_win_l2369_236920


namespace NUMINAMATH_CALUDE_valid_integers_count_l2369_236949

/-- The number of permutations of 6 distinct elements -/
def total_permutations : ℕ := 720

/-- The number of permutations satisfying the first condition (1 left of 2) -/
def permutations_condition1 : ℕ := total_permutations / 2

/-- The number of permutations satisfying both conditions (1 left of 2 and 3 left of 4) -/
def permutations_both_conditions : ℕ := permutations_condition1 / 2

/-- Theorem stating the number of valid 6-digit integers -/
theorem valid_integers_count : permutations_both_conditions = 180 := by sorry

end NUMINAMATH_CALUDE_valid_integers_count_l2369_236949


namespace NUMINAMATH_CALUDE_vins_bike_distance_l2369_236996

/-- Calculates the total distance ridden in a week given daily distances and number of days -/
def total_distance (to_school : ℕ) (from_school : ℕ) (days : ℕ) : ℕ :=
  (to_school + from_school) * days

/-- Proves that given the specific distances and number of days, the total distance is 65 miles -/
theorem vins_bike_distance : total_distance 6 7 5 = 65 := by
  sorry

end NUMINAMATH_CALUDE_vins_bike_distance_l2369_236996


namespace NUMINAMATH_CALUDE_first_digit_is_two_l2369_236988

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  isThreeDigit : 100 ≤ value ∧ value < 1000

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

/-- The main theorem -/
theorem first_digit_is_two
  (n : ThreeDigitNumber)
  (h1 : ∃ d : ℕ, d * 2 = n.value)
  (h2 : isDivisibleBy n.value 6)
  (h3 : ∃ d : ℕ, d * 2 = n.value ∧ d = 2)
  : n.value / 100 = 2 := by
  sorry

#check first_digit_is_two

end NUMINAMATH_CALUDE_first_digit_is_two_l2369_236988


namespace NUMINAMATH_CALUDE_henry_birthday_money_l2369_236927

theorem henry_birthday_money (initial_amount spent_amount final_amount : ℕ) 
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 10)
  (h3 : final_amount = 19) :
  final_amount + spent_amount - initial_amount = 18 := by
  sorry

end NUMINAMATH_CALUDE_henry_birthday_money_l2369_236927


namespace NUMINAMATH_CALUDE_three_W_seven_equals_thirteen_l2369_236968

/-- Definition of operation W -/
def W (x y : ℤ) : ℤ := y + 5*x - x^2

/-- Theorem: 3W7 equals 13 -/
theorem three_W_seven_equals_thirteen : W 3 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_three_W_seven_equals_thirteen_l2369_236968


namespace NUMINAMATH_CALUDE_hannah_movie_remaining_time_l2369_236910

/-- Calculates the remaining movie time given the total duration and watched duration. -/
def remaining_movie_time (total_duration watched_duration : ℕ) : ℕ :=
  total_duration - watched_duration

/-- Proves that for a 3-hour movie watched for 2 hours and 24 minutes, 36 minutes remain. -/
theorem hannah_movie_remaining_time :
  let total_duration : ℕ := 3 * 60  -- 3 hours in minutes
  let watched_duration : ℕ := 2 * 60 + 24  -- 2 hours and 24 minutes
  remaining_movie_time total_duration watched_duration = 36 := by
  sorry

#eval remaining_movie_time (3 * 60) (2 * 60 + 24)

end NUMINAMATH_CALUDE_hannah_movie_remaining_time_l2369_236910


namespace NUMINAMATH_CALUDE_intersection_M_N_l2369_236972

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - x^2)}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2369_236972


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l2369_236971

/-- Proves that given a round trip with specified conditions, 
    the average speed for the upward journey is 1.125 km/h -/
theorem hill_climbing_speed 
  (up_time : ℝ) 
  (down_time : ℝ) 
  (avg_speed : ℝ) 
  (h1 : up_time = 4) 
  (h2 : down_time = 2) 
  (h3 : avg_speed = 1.5) : 
  (avg_speed * (up_time + down_time)) / (2 * up_time) = 1.125 := by
  sorry

#check hill_climbing_speed

end NUMINAMATH_CALUDE_hill_climbing_speed_l2369_236971


namespace NUMINAMATH_CALUDE_textbook_reading_time_l2369_236923

/-- Calculates the total reading time in hours for a textbook with given parameters. -/
def totalReadingTime (totalChapters : ℕ) (readingTimePerChapter : ℕ) : ℚ :=
  let chaptersRead := totalChapters - (totalChapters / 3)
  (chaptersRead * readingTimePerChapter : ℚ) / 60

/-- Proves that the total reading time for the given textbook is 7 hours. -/
theorem textbook_reading_time :
  totalReadingTime 31 20 = 7 := by
  sorry

#eval totalReadingTime 31 20

end NUMINAMATH_CALUDE_textbook_reading_time_l2369_236923


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l2369_236990

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^(x-2)

theorem zero_point_in_interval :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ 1 < x₀ ∧ x₀ < 2 :=
sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l2369_236990


namespace NUMINAMATH_CALUDE_jie_is_tallest_l2369_236940

-- Define a type for the people
inductive Person : Type
  | Igor : Person
  | Jie : Person
  | Faye : Person
  | Goa : Person
  | Han : Person

-- Define a relation for "taller than"
def taller_than : Person → Person → Prop := sorry

-- Define the conditions
axiom igor_shorter_jie : taller_than Person.Jie Person.Igor
axiom faye_taller_goa : taller_than Person.Faye Person.Goa
axiom jie_taller_faye : taller_than Person.Jie Person.Faye
axiom han_shorter_goa : taller_than Person.Goa Person.Han

-- Define what it means to be the tallest
def is_tallest (p : Person) : Prop :=
  ∀ q : Person, p ≠ q → taller_than p q

-- State the theorem
theorem jie_is_tallest : is_tallest Person.Jie := by
  sorry

end NUMINAMATH_CALUDE_jie_is_tallest_l2369_236940


namespace NUMINAMATH_CALUDE_journey_distance_l2369_236966

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 20)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) : 
  ∃ (distance : ℝ), distance = 448 ∧ 
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2369_236966


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l2369_236936

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 11 * p - 20 = 0) → 
  (3 * q^2 + 11 * q - 20 = 0) → 
  (5 * p - 4) * (3 * q - 2) = -89/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l2369_236936


namespace NUMINAMATH_CALUDE_inscribed_square_area_ratio_l2369_236995

/-- Given a square ABCD with side length s, and an inscribed square A'B'C'D' where each vertex
    of A'B'C'D' is on a diagonal of ABCD and equidistant from the center of ABCD,
    the area of A'B'C'D' is 1/5 of the area of ABCD. -/
theorem inscribed_square_area_ratio (s : ℝ) (h : s > 0) :
  let abcd_area := s^2
  let apbpcpdp_side := s / Real.sqrt 5
  let apbpcpdp_area := apbpcpdp_side^2
  apbpcpdp_area / abcd_area = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_square_area_ratio_l2369_236995


namespace NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l2369_236908

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem fifty_billion_scientific_notation :
  toScientificNotation 50000000000 = ScientificNotation.mk 5 10 sorry :=
sorry

end NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l2369_236908


namespace NUMINAMATH_CALUDE_remainder_4x_mod_7_l2369_236954

theorem remainder_4x_mod_7 (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4x_mod_7_l2369_236954


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2369_236928

theorem pure_imaginary_complex_number (m : ℝ) :
  (((m^2 + 2*m - 3) : ℂ) + (m - 1)*I = (0 : ℂ) + ((m - 1)*I : ℂ)) → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2369_236928


namespace NUMINAMATH_CALUDE_periodic_and_zeros_l2369_236973

-- Define a periodic function
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ∀ x, f (x + T) = f x

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_and_zeros (f : ℝ → ℝ) (a : ℝ) :
  (a ≠ 0 ∧ ∀ x, f (x + a) = -f x) →
  IsPeriodic f (2 * a) ∧
  (IsOdd f → (∀ x, f (x + 1) = -f x) →
    ∃ (zeros : Finset ℝ), zeros.card ≥ 4035 ∧
      (∀ x ∈ zeros, -2017 ≤ x ∧ x ≤ 2017 ∧ f x = 0)) :=
by sorry


end NUMINAMATH_CALUDE_periodic_and_zeros_l2369_236973


namespace NUMINAMATH_CALUDE_sin_960_degrees_l2369_236909

theorem sin_960_degrees : Real.sin (960 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_960_degrees_l2369_236909


namespace NUMINAMATH_CALUDE_three_tangent_lines_l2369_236965

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define a line passing through (1, 0)
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 1)

-- Define the condition for a line to intersect the hyperbola at only one point
def single_intersection (m : ℝ) : Prop :=
  ∃! (x y : ℝ), hyperbola x y ∧ line_through_P m x y

-- Main theorem
theorem three_tangent_lines :
  ∃ (m₁ m₂ m₃ : ℝ), 
    single_intersection m₁ ∧ 
    single_intersection m₂ ∧ 
    single_intersection m₃ ∧
    m₁ ≠ m₂ ∧ m₁ ≠ m₃ ∧ m₂ ≠ m₃ ∧
    ∀ (m : ℝ), single_intersection m → m = m₁ ∨ m = m₂ ∨ m = m₃ :=
sorry

end NUMINAMATH_CALUDE_three_tangent_lines_l2369_236965


namespace NUMINAMATH_CALUDE_largest_divisible_by_thirtyseven_with_decreasing_digits_l2369_236980

/-- 
A function that checks if a natural number's digits are in strictly decreasing order.
-/
def isStrictlyDecreasing (n : ℕ) : Prop :=
  sorry

/-- 
A function that finds the largest natural number less than or equal to n 
that is divisible by 37 and has strictly decreasing digits.
-/
def largestDivisibleByThirtySevenWithDecreasingDigits (n : ℕ) : ℕ :=
  sorry

theorem largest_divisible_by_thirtyseven_with_decreasing_digits :
  largestDivisibleByThirtySevenWithDecreasingDigits 9876543210 = 987654 :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_by_thirtyseven_with_decreasing_digits_l2369_236980


namespace NUMINAMATH_CALUDE_polynomial_property_l2369_236985

def Q (x d e f : ℝ) : ℝ := 3 * x^4 + d * x^3 + e * x^2 + f * x - 27

theorem polynomial_property (d e f : ℝ) :
  (∀ x₁ x₂ x₃ x₄ : ℝ, Q x₁ d e f = 0 ∧ Q x₂ d e f = 0 ∧ Q x₃ d e f = 0 ∧ Q x₄ d e f = 0 →
    x₁ + x₂ + x₃ + x₄ = x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄) ∧
  (x₁ + x₂ + x₃ + x₄ = 3 + d + e + f - 27) ∧
  e = 0 →
  f = -12 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l2369_236985


namespace NUMINAMATH_CALUDE_registration_theorem_l2369_236921

/-- The number of possible ways for students to register for events. -/
def registration_combinations (num_students : ℕ) (num_events : ℕ) : ℕ :=
  num_events ^ num_students

/-- Theorem stating that with 4 students and 3 events, there are 81 possible registration combinations. -/
theorem registration_theorem :
  registration_combinations 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_registration_theorem_l2369_236921


namespace NUMINAMATH_CALUDE_original_number_exists_l2369_236999

theorem original_number_exists : ∃ x : ℝ, 3 * (2 * x + 5) = 123 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_l2369_236999


namespace NUMINAMATH_CALUDE_prob_not_sold_is_one_fifth_expected_profit_four_batches_l2369_236970

-- Define the probability of not passing each round
def prob_fail_first : ℚ := 1 / 9
def prob_fail_second : ℚ := 1 / 10

-- Define the profit/loss values
def profit_if_sold : ℤ := 400
def loss_if_not_sold : ℤ := 800

-- Define the number of batches
def num_batches : ℕ := 4

-- Define the probability of a batch being sold
def prob_sold : ℚ := (1 - prob_fail_first) * (1 - prob_fail_second)

-- Define the probability of a batch not being sold
def prob_not_sold : ℚ := 1 - prob_sold

-- Define the expected profit for a single batch
def expected_profit_single : ℚ := prob_sold * profit_if_sold - prob_not_sold * loss_if_not_sold

-- Theorem: Probability of a batch not being sold is 1/5
theorem prob_not_sold_is_one_fifth : prob_not_sold = 1 / 5 := by sorry

-- Theorem: Expected profit from 4 batches is 640 yuan
theorem expected_profit_four_batches : num_batches * expected_profit_single = 640 := by sorry

end NUMINAMATH_CALUDE_prob_not_sold_is_one_fifth_expected_profit_four_batches_l2369_236970


namespace NUMINAMATH_CALUDE_machine_sale_price_l2369_236992

def selling_price (purchase_price repair_cost transport_cost profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

theorem machine_sale_price :
  selling_price 11000 5000 1000 50 = 25500 := by
  sorry

end NUMINAMATH_CALUDE_machine_sale_price_l2369_236992


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l2369_236911

theorem smallest_angle_in_special_triangle : 
  ∀ (a b c : ℝ), 
    a + b + c = 180 →  -- Sum of angles is 180 degrees
    c = 5 * a →        -- Largest angle is 5 times the smallest
    b = 3 * a →        -- Middle angle is 3 times the smallest
    a = 20 :=          -- Smallest angle is 20 degrees
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l2369_236911


namespace NUMINAMATH_CALUDE_vehicle_value_last_year_l2369_236913

theorem vehicle_value_last_year 
  (value_this_year : ℝ) 
  (value_ratio : ℝ) 
  (h1 : value_this_year = 16000)
  (h2 : value_ratio = 0.8)
  (h3 : value_this_year = value_ratio * value_last_year) :
  value_last_year = 20000 :=
by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_last_year_l2369_236913


namespace NUMINAMATH_CALUDE_sum_interior_angles_limited_diagonal_polygon_l2369_236975

/-- A polygon where at most 6 diagonals can be drawn from any vertex -/
structure LimitedDiagonalPolygon where
  vertices : ℕ
  diagonals_limit : vertices - 3 = 6

/-- The sum of interior angles of a polygon -/
def sum_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

/-- Theorem: The sum of interior angles of a LimitedDiagonalPolygon is 1260° -/
theorem sum_interior_angles_limited_diagonal_polygon (p : LimitedDiagonalPolygon) :
  sum_interior_angles p.vertices = 1260 := by
  sorry

#eval sum_interior_angles 9  -- Expected output: 1260

end NUMINAMATH_CALUDE_sum_interior_angles_limited_diagonal_polygon_l2369_236975


namespace NUMINAMATH_CALUDE_strings_needed_is_302_l2369_236901

/-- Calculates the total number of strings needed for a set of instruments, including extra strings due to machine malfunction --/
def total_strings_needed (num_basses : ℕ) (strings_per_bass : ℕ) (guitar_multiplier : ℕ) 
  (strings_per_guitar : ℕ) (eight_string_guitar_reduction : ℕ) (strings_per_eight_string_guitar : ℕ)
  (strings_per_twelve_string_guitar : ℕ) (nylon_strings_per_eight_string_guitar : ℕ)
  (nylon_strings_per_twelve_string_guitar : ℕ) (malfunction_rate : ℕ) : ℕ :=
  let num_guitars := num_basses * guitar_multiplier
  let num_eight_string_guitars := num_guitars - eight_string_guitar_reduction
  let num_twelve_string_guitars := num_basses
  let total_strings := 
    num_basses * strings_per_bass +
    num_guitars * strings_per_guitar +
    num_eight_string_guitars * strings_per_eight_string_guitar +
    num_twelve_string_guitars * strings_per_twelve_string_guitar
  let extra_strings := (total_strings + malfunction_rate - 1) / malfunction_rate
  total_strings + extra_strings

/-- Theorem stating that given the specific conditions, the total number of strings needed is 302 --/
theorem strings_needed_is_302 : 
  total_strings_needed 5 4 3 6 2 8 12 2 6 10 = 302 := by
  sorry

end NUMINAMATH_CALUDE_strings_needed_is_302_l2369_236901


namespace NUMINAMATH_CALUDE_pizza_toppings_l2369_236926

/-- Given a pizza with 24 slices, where every slice has at least one topping,
    if exactly 15 slices have ham and exactly 17 slices have cheese,
    then the number of slices with both ham and cheese is 8. -/
theorem pizza_toppings (total : Nat) (ham : Nat) (cheese : Nat) (both : Nat) :
  total = 24 →
  ham = 15 →
  cheese = 17 →
  both + (ham - both) + (cheese - both) = total →
  both = 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_toppings_l2369_236926


namespace NUMINAMATH_CALUDE_trip_time_calculation_l2369_236906

/-- Represents the time for a trip given two different speeds -/
def trip_time (initial_speed initial_time new_speed : ℚ) : ℚ :=
  (initial_speed * initial_time) / new_speed

theorem trip_time_calculation (initial_speed initial_time new_speed : ℚ) :
  initial_speed = 80 →
  initial_time = 16/3 →
  new_speed = 50 →
  trip_time initial_speed initial_time new_speed = 128/15 := by
  sorry

#eval trip_time 80 (16/3) 50

end NUMINAMATH_CALUDE_trip_time_calculation_l2369_236906


namespace NUMINAMATH_CALUDE_point_classification_l2369_236987

-- Define the region D
def D (x y : ℝ) : Prop := y < x ∧ x + y ≤ 1 ∧ y ≥ -3

-- Define points P and Q
def P : ℝ × ℝ := (0, -2)
def Q : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem point_classification :
  D P.1 P.2 ∧ ¬D Q.1 Q.2 := by sorry

end NUMINAMATH_CALUDE_point_classification_l2369_236987


namespace NUMINAMATH_CALUDE_expression_evaluation_l2369_236955

theorem expression_evaluation (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 2) / (3 * x^3))^2) = 
  (Real.sqrt ((x^6 + 4) * (x^6 + 1))) / (3 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2369_236955


namespace NUMINAMATH_CALUDE_rohans_salary_l2369_236903

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 10000

/-- Percentage of salary spent on food -/
def food_percentage : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_percentage : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_percentage : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_percentage : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 2000

theorem rohans_salary :
  monthly_salary * (1 - (food_percentage + rent_percentage + entertainment_percentage + conveyance_percentage) / 100) = savings := by
  sorry

#check rohans_salary

end NUMINAMATH_CALUDE_rohans_salary_l2369_236903


namespace NUMINAMATH_CALUDE_pencil_count_problem_l2369_236905

/-- The number of pencils in a drawer after a series of additions and removals. -/
def final_pencil_count (initial : ℕ) (sara_adds : ℕ) (john_adds : ℕ) (ben_removes : ℕ) : ℕ :=
  initial + sara_adds + john_adds - ben_removes

/-- Theorem stating that given the initial number of pencils and the changes made by Sara, John, and Ben, the final number of pencils is 245. -/
theorem pencil_count_problem : final_pencil_count 115 100 75 45 = 245 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_problem_l2369_236905


namespace NUMINAMATH_CALUDE_cubic_coefficient_sum_l2369_236953

theorem cubic_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) : 
  (∀ x : ℝ, (5*x + 4)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) → 
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_coefficient_sum_l2369_236953


namespace NUMINAMATH_CALUDE_circle_equation_l2369_236930

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the points M and N
def M : ℝ × ℝ := (-2, 2)
def N : ℝ × ℝ := (-1, -1)

-- Define the line equation x - y - 1 = 0
def LineEquation (p : ℝ × ℝ) : Prop := p.1 - p.2 - 1 = 0

-- Theorem statement
theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    LineEquation center ∧
    M ∈ Circle center radius ∧
    N ∈ Circle center radius ∧
    center = (3, 2) ∧
    radius = 5 :=
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2369_236930


namespace NUMINAMATH_CALUDE_square_perimeter_47_20_l2369_236991

-- Define the side length of the square
def side_length : ℚ := 47 / 20

-- Define the perimeter of a square
def square_perimeter (s : ℚ) : ℚ := 4 * s

-- Theorem statement
theorem square_perimeter_47_20 : 
  square_perimeter side_length = 47 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_47_20_l2369_236991


namespace NUMINAMATH_CALUDE_marj_wallet_remaining_l2369_236956

/-- Calculates the remaining money in Marj's wallet after expenses --/
def remaining_money (initial_usd : ℚ) (initial_euro : ℚ) (initial_pound : ℚ) 
  (euro_to_usd : ℚ) (pound_to_usd : ℚ) (cake_cost : ℚ) (gift_cost : ℚ) (donation : ℚ) : ℚ :=
  initial_usd + initial_euro * euro_to_usd + initial_pound * pound_to_usd - cake_cost - gift_cost - donation

/-- Theorem stating that Marj will have $64.40 left in her wallet after expenses --/
theorem marj_wallet_remaining : 
  remaining_money 81.5 10 5 1.18 1.32 17.5 12.7 5.3 = 64.4 := by
  sorry

end NUMINAMATH_CALUDE_marj_wallet_remaining_l2369_236956


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2369_236952

/-- A color type representing red, green, or blue -/
inductive Color
| Red
| Green
| Blue

/-- A type representing a 4 × 82 grid where each cell is colored -/
def Grid := Fin 4 → Fin 82 → Color

/-- A function to check if four points form a rectangle with the same color -/
def isMonochromaticRectangle (g : Grid) (x1 y1 x2 y2 : ℕ) : Prop :=
  x1 < x2 ∧ y1 < y2 ∧
  g ⟨x1, by sorry⟩ ⟨y1, by sorry⟩ = g ⟨x1, by sorry⟩ ⟨y2, by sorry⟩ ∧
  g ⟨x1, by sorry⟩ ⟨y1, by sorry⟩ = g ⟨x2, by sorry⟩ ⟨y1, by sorry⟩ ∧
  g ⟨x1, by sorry⟩ ⟨y1, by sorry⟩ = g ⟨x2, by sorry⟩ ⟨y2, by sorry⟩

/-- Theorem: In any 3-coloring of a 4 × 82 grid, there exists a rectangle whose vertices are all the same color -/
theorem monochromatic_rectangle_exists (g : Grid) :
  ∃ (x1 y1 x2 y2 : ℕ), isMonochromaticRectangle g x1 y1 x2 y2 :=
sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2369_236952


namespace NUMINAMATH_CALUDE_hyperbola_dot_product_nonnegative_l2369_236912

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- The left vertex of the hyperbola -/
def A : ℝ × ℝ := (-2, 0)

/-- The right vertex of the hyperbola -/
def B : ℝ × ℝ := (2, 0)

/-- The dot product of vectors PA and PB -/
def dot_product (P : ℝ × ℝ) : ℝ :=
  let (m, n) := P
  ((-2 - m) * (2 - m)) + (n * n)

theorem hyperbola_dot_product_nonnegative :
  ∀ P : ℝ × ℝ, hyperbola P.1 P.2 → dot_product P ≥ 0 := by sorry

end NUMINAMATH_CALUDE_hyperbola_dot_product_nonnegative_l2369_236912


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l2369_236929

/-- The probability of selecting two non-defective pens from a box of 8 pens, where 2 are defective -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) :
  total_pens = 8 →
  defective_pens = 2 →
  selected_pens = 2 →
  (total_pens - defective_pens : ℚ) / total_pens *
  ((total_pens - defective_pens - 1 : ℚ) / (total_pens - 1)) = 15 / 28 :=
by sorry

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l2369_236929


namespace NUMINAMATH_CALUDE_set_union_problem_l2369_236933

theorem set_union_problem (a b : ℕ) : 
  let A : Set ℕ := {5, 2^a}
  let B : Set ℕ := {a, b}
  A ∩ B = {8} →
  A ∪ B = {3, 5, 8} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l2369_236933


namespace NUMINAMATH_CALUDE_matrix_transformation_l2369_236978

theorem matrix_transformation (N : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : N.mulVec ![1, 2] = ![4, 1])
  (h2 : N.mulVec ![2, -3] = ![1, 4]) :
  N.mulVec ![7, -2] = ![(84:ℚ)/7, (81:ℚ)/7] := by
  sorry

end NUMINAMATH_CALUDE_matrix_transformation_l2369_236978


namespace NUMINAMATH_CALUDE_sum_of_smallest_multiples_l2369_236904

def smallest_two_digit_multiple_of_3 : ℕ → Prop :=
  λ n => n ≥ 10 ∧ n < 100 ∧ 3 ∣ n ∧ ∀ m, m ≥ 10 ∧ m < 100 ∧ 3 ∣ m → n ≤ m

def smallest_three_digit_multiple_of_4 : ℕ → Prop :=
  λ n => n ≥ 100 ∧ n < 1000 ∧ 4 ∣ n ∧ ∀ m, m ≥ 100 ∧ m < 1000 ∧ 4 ∣ m → n ≤ m

theorem sum_of_smallest_multiples : 
  ∀ a b : ℕ, smallest_two_digit_multiple_of_3 a → smallest_three_digit_multiple_of_4 b → 
  a + b = 112 := by
sorry

end NUMINAMATH_CALUDE_sum_of_smallest_multiples_l2369_236904


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l2369_236993

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  4 * (x + 1)^2 - 25 = 0 ↔ x = 3/2 ∨ x = -7/2 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (x + 10)^3 = -125 ↔ x = -15 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l2369_236993


namespace NUMINAMATH_CALUDE_range_of_a_l2369_236937

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x - 4 ≤ 0}

-- Define the theorem
theorem range_of_a (a : ℝ) : B a ⊆ A ↔ 0 ≤ a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2369_236937


namespace NUMINAMATH_CALUDE_coat_price_reduction_l2369_236938

theorem coat_price_reduction (original_price reduction : ℝ) 
  (h1 : original_price = 500)
  (h2 : reduction = 250) :
  (reduction / original_price) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_reduction_l2369_236938


namespace NUMINAMATH_CALUDE_n_pointed_star_value_l2369_236948

/-- Represents an n-pointed star. -/
structure PointedStar where
  n : ℕ
  segment_length : ℝ
  angle_a : ℝ
  angle_b : ℝ

/-- Theorem stating the properties of the n-pointed star and the value of n. -/
theorem n_pointed_star_value (star : PointedStar) :
  star.segment_length = 2 * star.n ∧
  star.angle_a = star.angle_b - 10 ∧
  star.n > 2 →
  star.n = 36 := by
  sorry

end NUMINAMATH_CALUDE_n_pointed_star_value_l2369_236948


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l2369_236916

theorem empty_solution_set_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 3| + |x - a| < 1)) ↔ (a ≤ 2 ∨ a ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l2369_236916


namespace NUMINAMATH_CALUDE_fraction_of_shaded_hexagons_l2369_236935

/-- Given a set of hexagons, some of which are shaded, prove that the fraction of shaded hexagons is correct. -/
theorem fraction_of_shaded_hexagons 
  (total : ℕ) 
  (shaded : ℕ) 
  (h1 : total = 9) 
  (h2 : shaded = 5) : 
  (shaded : ℚ) / total = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_shaded_hexagons_l2369_236935


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2369_236964

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁^2 - 4*x₁ - 5 = 0 ∧ 
  x₂^2 - 4*x₂ - 5 = 0 ∧ 
  x₁ = 5 ∧ 
  x₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2369_236964


namespace NUMINAMATH_CALUDE_probability_three_odd_less_than_eighth_l2369_236976

def range_size : ℕ := 2023
def odd_count : ℕ := (range_size + 1) / 2

theorem probability_three_odd_less_than_eighth :
  (odd_count : ℚ) / range_size *
  ((odd_count - 1) : ℚ) / (range_size - 1) *
  ((odd_count - 2) : ℚ) / (range_size - 2) <
  1 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_three_odd_less_than_eighth_l2369_236976


namespace NUMINAMATH_CALUDE_decimal_period_equals_number_period_l2369_236958

/-- The length of the repeating period in the decimal representation of a fraction -/
def decimal_period_length (n p : ℕ) : ℕ := sorry

/-- The length of the period of a number in decimal representation -/
def number_period_length (p : ℕ) : ℕ := sorry

/-- Theorem stating that for a natural number n and a prime number p, 
    where n ≤ p - 1, the length of the repeating period in the decimal 
    representation of n/p is equal to the length of the period of p -/
theorem decimal_period_equals_number_period (n p : ℕ) 
  (h_prime : Nat.Prime p) (h_n_le_p_minus_one : n ≤ p - 1) : 
  decimal_period_length n p = number_period_length p := by
  sorry

end NUMINAMATH_CALUDE_decimal_period_equals_number_period_l2369_236958


namespace NUMINAMATH_CALUDE_solve_for_a_l2369_236994

theorem solve_for_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : a * x + 3 * y = 13) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2369_236994


namespace NUMINAMATH_CALUDE_range_of_a_l2369_236982

theorem range_of_a (a : ℝ) : 
  (∃ x₀ ∈ Set.Icc 1 3, |x₀^2 - a*x₀ + 4| ≤ 3*x₀) → 1 ≤ a ∧ a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2369_236982


namespace NUMINAMATH_CALUDE_difference_of_squares_l2369_236977

theorem difference_of_squares (a b : ℝ) : a^2 - 9*b^2 = (a + 3*b) * (a - 3*b) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2369_236977


namespace NUMINAMATH_CALUDE_product_of_primes_in_final_positions_l2369_236914

-- Define the colors
inductive Color
| Red
| Yellow
| Green
| Blue

-- Define the positions in a 2x2 grid
inductive Position
| TopLeft
| TopRight
| BottomLeft
| BottomRight

-- Define the transformation function
def transform (c : Color) : Position → Position
| Position.TopLeft => 
    match c with
    | Color.Red => Position.TopRight
    | Color.Yellow => Position.TopRight
    | Color.Green => Position.BottomLeft
    | Color.Blue => Position.BottomRight
| Position.TopRight => 
    match c with
    | Color.Red => Position.TopRight
    | Color.Yellow => Position.TopRight
    | Color.Green => Position.BottomRight
    | Color.Blue => Position.BottomRight
| Position.BottomLeft => 
    match c with
    | Color.Red => Position.TopLeft
    | Color.Yellow => Position.TopLeft
    | Color.Green => Position.BottomLeft
    | Color.Blue => Position.BottomLeft
| Position.BottomRight => 
    match c with
    | Color.Red => Position.TopLeft
    | Color.Yellow => Position.TopLeft
    | Color.Green => Position.BottomRight
    | Color.Blue => Position.BottomRight

-- Define the numbers in Figure 4
def figure4 (p : Position) : Nat :=
  match p with
  | Position.TopLeft => 6
  | Position.TopRight => 7
  | Position.BottomLeft => 5
  | Position.BottomRight => 8

-- Define primality
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem product_of_primes_in_final_positions : 
  let finalRedPosition := transform Color.Red (transform Color.Red Position.TopLeft)
  let finalYellowPosition := transform Color.Yellow (transform Color.Yellow Position.TopRight)
  (isPrime (figure4 finalRedPosition) ∧ isPrime (figure4 finalYellowPosition)) →
  figure4 finalRedPosition * figure4 finalYellowPosition = 55 := by
  sorry


end NUMINAMATH_CALUDE_product_of_primes_in_final_positions_l2369_236914


namespace NUMINAMATH_CALUDE_total_wheels_l2369_236907

/-- The number of bikes that can be assembled in the garage -/
def bikes_assemblable : ℕ := 10

/-- The number of wheels required for each bike -/
def wheels_per_bike : ℕ := 2

/-- Theorem: The total number of wheels in the garage is 20 -/
theorem total_wheels : bikes_assemblable * wheels_per_bike = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_l2369_236907


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2369_236998

theorem triangle_angle_proof (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = c →
  C = π / 5 →
  B = 3 * π / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2369_236998


namespace NUMINAMATH_CALUDE_sqrt_77_plus_28sqrt3_l2369_236959

theorem sqrt_77_plus_28sqrt3 :
  ∃ (x y z : ℤ), 
    (∀ (k : ℕ), k > 1 → ¬ (∃ (m : ℕ), z = k^2 * m)) →
    (x + y * Real.sqrt z : ℝ) = Real.sqrt (77 + 28 * Real.sqrt 3) ∧
    x = 7 ∧ y = 2 ∧ z = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_77_plus_28sqrt3_l2369_236959


namespace NUMINAMATH_CALUDE_blue_string_length_l2369_236919

/-- Given three strings (red, white, and blue) with the following properties:
  - The length of the red string is 8 metres.
  - The length of the white string is 5 times the length of the red string.
  - The white string is 8 times as long as the blue string.
  Prove that the length of the blue string is 5 metres. -/
theorem blue_string_length 
  (red : ℝ) 
  (white : ℝ) 
  (blue : ℝ) 
  (h1 : red = 8) 
  (h2 : white = 5 * red) 
  (h3 : white = 8 * blue) : 
  blue = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_string_length_l2369_236919


namespace NUMINAMATH_CALUDE_max_product_constrained_l2369_236918

theorem max_product_constrained (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : x^(Real.log x / Real.log y) * y^(Real.log y / Real.log z) * z^(Real.log z / Real.log x) = 10) :
  x * y * z ≤ 10 ∧ ∃ (a b c : ℝ), a > 1 ∧ b > 1 ∧ c > 1 ∧
    a^(Real.log a / Real.log b) * b^(Real.log b / Real.log c) * c^(Real.log c / Real.log a) = 10 ∧
    a * b * c = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_product_constrained_l2369_236918


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2369_236960

theorem smallest_candy_count : ∃ n : ℕ,
  (100 ≤ n ∧ n < 1000) ∧
  (n + 7) % 9 = 0 ∧
  (n - 9) % 6 = 0 ∧
  (∀ m : ℕ, 100 ≤ m ∧ m < n → (m + 7) % 9 ≠ 0 ∨ (m - 9) % 6 ≠ 0) ∧
  n = 137 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2369_236960


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2369_236902

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval [0, 3]
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem f_max_min_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∀ x ∈ interval, f x ≤ 5) ∧
  (∀ x ∈ interval, -15 ≤ f x) ∧
  (∃ x ∈ interval, f x = 5) ∧
  (∃ x ∈ interval, f x = -15) :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2369_236902


namespace NUMINAMATH_CALUDE_no_rectangular_parallelepiped_sum_866_l2369_236957

theorem no_rectangular_parallelepiped_sum_866 :
  ¬∃ (x y z : ℕ+), x * y * z + 2 * (x * y + x * z + y * z) + 4 * (x + y + z) = 866 := by
sorry

end NUMINAMATH_CALUDE_no_rectangular_parallelepiped_sum_866_l2369_236957


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2369_236979

/-- An arithmetic sequence with a positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_positive : d > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence where a_2, a_6, and a_12 form a geometric sequence,
    the ratio of a_12 to a_2 is 9/4 -/
theorem arithmetic_geometric_ratio
  (seq : ArithmeticSequence)
  (h_geometric : (seq.a 6) ^ 2 = (seq.a 2) * (seq.a 12)) :
  (seq.a 12) / (seq.a 2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2369_236979


namespace NUMINAMATH_CALUDE_factorization_equality_l2369_236934

theorem factorization_equality (x y : ℝ) : 
  4 * (x - y + 1) + y * (y - 2 * x) = (y - 2) * (y - 2 - 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2369_236934


namespace NUMINAMATH_CALUDE_base_conversion_314_to_1242_l2369_236984

/-- Converts a natural number from base 10 to base 6 --/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 6 to a natural number in base 10 --/
def fromBase6 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_314_to_1242 :
  toBase6 314 = [1, 2, 4, 2] ∧ fromBase6 [1, 2, 4, 2] = 314 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_314_to_1242_l2369_236984
