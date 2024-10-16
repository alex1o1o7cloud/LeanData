import Mathlib

namespace NUMINAMATH_CALUDE_squirrel_is_red_l3455_345574

-- Define the color of the squirrel
inductive SquirrelColor
  | Red
  | Gray

-- Define the state of a hollow
inductive HollowState
  | Empty
  | HasNuts

-- Define the structure for the two hollows
structure Hollows :=
  (first : HollowState)
  (second : HollowState)

-- Define the statements made by the squirrel
def statement1 (h : Hollows) : Prop :=
  h.first = HollowState.Empty

def statement2 (h : Hollows) : Prop :=
  h.first = HollowState.HasNuts ∨ h.second = HollowState.HasNuts

-- Define the truthfulness of the squirrel based on its color
def isTruthful (c : SquirrelColor) : Prop :=
  match c with
  | SquirrelColor.Red => True
  | SquirrelColor.Gray => False

-- Theorem: The squirrel must be red
theorem squirrel_is_red (h : Hollows) :
  (isTruthful SquirrelColor.Red → statement1 h ∧ statement2 h) ∧
  (isTruthful SquirrelColor.Gray → ¬(statement1 h) ∧ ¬(statement2 h)) →
  ∃ (h : Hollows), statement1 h ∧ statement2 h →
  SquirrelColor.Red = SquirrelColor.Red :=
by sorry

end NUMINAMATH_CALUDE_squirrel_is_red_l3455_345574


namespace NUMINAMATH_CALUDE_weightlifting_ratio_l3455_345555

theorem weightlifting_ratio (total weight_first weight_second : ℕ) 
  (h1 : total = weight_first + weight_second)
  (h2 : weight_first = 700)
  (h3 : 2 * weight_first = weight_second + 300)
  (h4 : total = 1800) : 
  weight_first * 11 = weight_second * 7 := by
  sorry

end NUMINAMATH_CALUDE_weightlifting_ratio_l3455_345555


namespace NUMINAMATH_CALUDE_solution_set_part_i_value_of_a_l3455_345578

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Part I
theorem solution_set_part_i (a : ℝ) (h : a = 2) :
  {x : ℝ | f a x ≥ 3 * x + 2} = {x : ℝ | x ≥ 4 ∨ x ≤ 0} :=
sorry

-- Part II
theorem value_of_a (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -1}) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_value_of_a_l3455_345578


namespace NUMINAMATH_CALUDE_donut_selection_count_l3455_345515

/-- The number of types of donuts available -/
def num_donut_types : ℕ := 3

/-- The number of donuts Pat wants to buy -/
def num_donuts_to_buy : ℕ := 4

/-- The number of ways to select donuts -/
def num_selections : ℕ := (num_donuts_to_buy + num_donut_types - 1).choose (num_donut_types - 1)

theorem donut_selection_count : num_selections = 15 := by
  sorry

end NUMINAMATH_CALUDE_donut_selection_count_l3455_345515


namespace NUMINAMATH_CALUDE_sara_payment_l3455_345552

/-- The amount Sara gave to the seller -/
def amount_given (book1_price book2_price change : ℝ) : ℝ :=
  book1_price + book2_price + change

/-- Theorem stating the amount Sara gave to the seller -/
theorem sara_payment (book1_price book2_price change : ℝ) 
  (h1 : book1_price = 5.5)
  (h2 : book2_price = 6.5)
  (h3 : change = 8) :
  amount_given book1_price book2_price change = 20 := by
sorry

end NUMINAMATH_CALUDE_sara_payment_l3455_345552


namespace NUMINAMATH_CALUDE_sum_13_eq_26_l3455_345556

/-- An arithmetic sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The sum of the first n terms of an arithmetic sequence. -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (List.range n).map seq.a |>.sum

theorem sum_13_eq_26 (seq : ArithmeticSequence) 
    (h : seq.a 3 + seq.a 7 + seq.a 11 = 6) : 
  sum_n seq 13 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_13_eq_26_l3455_345556


namespace NUMINAMATH_CALUDE_system_integer_solutions_determinant_l3455_345551

theorem system_integer_solutions_determinant (a b c d : ℤ) :
  (∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) →
  (a * d - b * c = 1 ∨ a * d - b * c = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_integer_solutions_determinant_l3455_345551


namespace NUMINAMATH_CALUDE_kids_played_monday_tuesday_l3455_345593

/-- The number of kids Julia played with on Monday, Tuesday, and Wednesday -/
structure KidsPlayedWith where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Theorem: The sum of kids Julia played with on Monday and Tuesday is 33 -/
theorem kids_played_monday_tuesday (k : KidsPlayedWith) 
  (h1 : k.monday = 15)
  (h2 : k.tuesday = 18)
  (h3 : k.wednesday = 97) : 
  k.monday + k.tuesday = 33 := by
  sorry


end NUMINAMATH_CALUDE_kids_played_monday_tuesday_l3455_345593


namespace NUMINAMATH_CALUDE_race_head_start_l3455_345545

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (20 / 14) * Vb) :
  ∃ H : ℝ, H = (3 / 10) * L ∧ L / Va = (L - H) / Vb :=
sorry

end NUMINAMATH_CALUDE_race_head_start_l3455_345545


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_times_111_not_five_digit_palindrome_l3455_345564

-- Define a function to check if a number is a three-digit palindrome
def isThreeDigitPalindrome (n : Nat) : Prop :=
  n ≥ 100 ∧ n ≤ 999 ∧ (n / 100 = n % 10)

-- Define a function to check if a number is a five-digit palindrome
def isFiveDigitPalindrome (n : Nat) : Prop :=
  n ≥ 10000 ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

-- Theorem statement
theorem smallest_three_digit_palindrome_times_111_not_five_digit_palindrome :
  isThreeDigitPalindrome 515 ∧
  ¬(isFiveDigitPalindrome (515 * 111)) ∧
  ∀ n : Nat, n < 515 → isThreeDigitPalindrome n → isFiveDigitPalindrome (n * 111) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_times_111_not_five_digit_palindrome_l3455_345564


namespace NUMINAMATH_CALUDE_sum_last_two_digits_lfs_l3455_345591

/-- Lucas Factorial Series function -/
def lucasFactorialSeries : ℕ → ℕ
| 0 => 2
| 1 => 1
| 2 => 3
| 3 => 4
| 4 => 7
| 5 => 11
| _ => 0

/-- Calculate factorial -/
def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

/-- Get last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ :=
  n % 100

/-- Sum of last two digits of factorials in Lucas Factorial Series -/
def sumLastTwoDigitsLFS : ℕ :=
  let series := List.range 6
  series.foldl (fun acc i => acc + lastTwoDigits (factorial (lucasFactorialSeries i))) 0

/-- Main theorem -/
theorem sum_last_two_digits_lfs :
  sumLastTwoDigitsLFS = 73 := by
  sorry


end NUMINAMATH_CALUDE_sum_last_two_digits_lfs_l3455_345591


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_meaningful_l3455_345575

theorem sqrt_x_plus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_meaningful_l3455_345575


namespace NUMINAMATH_CALUDE_min_value_theorem_l3455_345531

theorem min_value_theorem (a : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  (4 / a) + (1 / (2 - a)) ≥ (9 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3455_345531


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l3455_345557

/-- Two similar right triangles with legs 12 and 9 in the first triangle, 
    and y and 6 in the second triangle, have y equal to 8 -/
theorem similar_triangles_leg_length : ∀ y : ℝ,
  (12 : ℝ) / y = 9 / 6 → y = 8 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l3455_345557


namespace NUMINAMATH_CALUDE_robert_second_trade_l3455_345500

def cards_traded_problem (padma_initial : ℕ) (robert_initial : ℕ) (padma_traded_first : ℕ) 
  (robert_traded_first : ℕ) (padma_traded_second : ℕ) (total_traded : ℕ) : Prop :=
  padma_initial = 75 ∧
  robert_initial = 88 ∧
  padma_traded_first = 2 ∧
  robert_traded_first = 10 ∧
  padma_traded_second = 15 ∧
  total_traded = 35 ∧
  total_traded = padma_traded_first + robert_traded_first + padma_traded_second + 
    (total_traded - padma_traded_first - robert_traded_first - padma_traded_second)

theorem robert_second_trade (padma_initial robert_initial padma_traded_first robert_traded_first 
  padma_traded_second total_traded : ℕ) :
  cards_traded_problem padma_initial robert_initial padma_traded_first robert_traded_first 
    padma_traded_second total_traded →
  total_traded - padma_traded_first - robert_traded_first - padma_traded_second = 25 :=
by sorry

end NUMINAMATH_CALUDE_robert_second_trade_l3455_345500


namespace NUMINAMATH_CALUDE_product_of_primes_l3455_345501

def smallest_one_digit_primes : List Nat := [2, 3]
def largest_three_digit_prime : Nat := 997

theorem product_of_primes :
  (smallest_one_digit_primes.prod * largest_three_digit_prime) = 5982 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_l3455_345501


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3455_345580

def U : Finset Nat := {0, 1, 2, 3, 4}
def A : Finset Nat := {0, 1, 3}
def B : Finset Nat := {2, 3}

theorem intersection_complement_equality : A ∩ (U \ B) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3455_345580


namespace NUMINAMATH_CALUDE_logarithmic_function_problem_l3455_345547

open Real

theorem logarithmic_function_problem (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  let f := fun x => |log x|
  (f a = f b) →
  (∀ x ∈ Set.Icc (a^2) b, f x ≤ 2) →
  (∃ x ∈ Set.Icc (a^2) b, f x = 2) →
  2 * a + b = 2 / Real.exp 1 + Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_logarithmic_function_problem_l3455_345547


namespace NUMINAMATH_CALUDE_dot_product_equals_negative_49_l3455_345532

def vector1 : Fin 4 → ℝ := ![4, -5, 2, -1]
def vector2 : Fin 4 → ℝ := ![-6, 3, -4, 2]

theorem dot_product_equals_negative_49 :
  (Finset.sum Finset.univ (λ i => vector1 i * vector2 i)) = -49 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equals_negative_49_l3455_345532


namespace NUMINAMATH_CALUDE_some_number_solution_l3455_345512

theorem some_number_solution : 
  ∃ x : ℝ, 4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470 ∧ x = 13.26 := by
  sorry

end NUMINAMATH_CALUDE_some_number_solution_l3455_345512


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_parameters_l3455_345508

/-- Represents a hyperbola with given eccentricity and foci -/
structure Hyperbola where
  eccentricity : ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- The equation of a hyperbola given its parameters -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 / 4 - y^2 / 12 = 1

/-- Theorem stating that a hyperbola with eccentricity 2 and foci at (-4, 0) and (4, 0)
    has the equation x^2/4 - y^2/12 = 1 -/
theorem hyperbola_equation_from_parameters :
  ∀ h : Hyperbola,
    h.eccentricity = 2 ∧
    h.focus1 = (-4, 0) ∧
    h.focus2 = (4, 0) →
    hyperbola_equation h = fun x y => x^2 / 4 - y^2 / 12 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_parameters_l3455_345508


namespace NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l3455_345585

theorem mrs_sheridan_fish_count (initial_fish : Nat) (fish_from_sister : Nat) : 
  initial_fish = 22 → fish_from_sister = 47 → initial_fish + fish_from_sister = 69 := by
  sorry

end NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l3455_345585


namespace NUMINAMATH_CALUDE_complex_sum_of_sixth_powers_l3455_345536

theorem complex_sum_of_sixth_powers : 
  (((1 : ℂ) + Complex.I * Real.sqrt 3) / 2) ^ 6 + 
  (((1 : ℂ) - Complex.I * Real.sqrt 3) / 2) ^ 6 = 
  (1 : ℂ) / 2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_of_sixth_powers_l3455_345536


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3455_345503

/-- Given a hyperbola with specified asymptotes and a point it passes through,
    prove that the distance between its foci is 2√(13.5). -/
theorem hyperbola_foci_distance (x₀ y₀ : ℝ) :
  let asymptote1 : ℝ → ℝ := λ x => 2 * x + 2
  let asymptote2 : ℝ → ℝ := λ x => -2 * x + 4
  let point : ℝ × ℝ := (2, 6)
  (∀ x, y₀ = asymptote1 x ∨ y₀ = asymptote2 x → x₀ = x) →
  (y₀ = asymptote1 x₀ ∨ y₀ = asymptote2 x₀) →
  point.1 = 2 ∧ point.2 = 6 →
  ∃ (center : ℝ × ℝ) (a b : ℝ),
    (∀ x y, ((y - center.2)^2 / a^2) - ((x - center.1)^2 / b^2) = 1 →
      y = asymptote1 x ∨ y = asymptote2 x) ∧
    ((point.2 - center.2)^2 / a^2) - ((point.1 - center.1)^2 / b^2) = 1 ∧
    2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 13.5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3455_345503


namespace NUMINAMATH_CALUDE_projection_theorem_l3455_345559

def proj_vector (a b : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_theorem (a b : ℝ × ℝ) (angle : ℝ) :
  angle = 2 * Real.pi / 3 →
  norm a = 10 →
  b = (3, 4) →
  proj_vector a b = (-3, -4) := by sorry

end NUMINAMATH_CALUDE_projection_theorem_l3455_345559


namespace NUMINAMATH_CALUDE_power_eight_sum_ratio_l3455_345598

theorem power_eight_sum_ratio (x y k : ℝ) 
  (h : (x^2 + y^2)/(x^2 - y^2) + (x^2 - y^2)/(x^2 + y^2) = k) :
  (x^8 + y^8)/(x^8 - y^8) + (x^8 - y^8)/(x^8 + y^8) = (k^4 + 24*k^2 + 16)/(4*k^3 + 16*k) :=
by sorry

end NUMINAMATH_CALUDE_power_eight_sum_ratio_l3455_345598


namespace NUMINAMATH_CALUDE_tims_bodyguard_cost_l3455_345530

/-- Calculate the total weekly cost for bodyguards --/
def total_weekly_cost (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

/-- Prove that the total weekly cost for Tim's bodyguards is $2240 --/
theorem tims_bodyguard_cost :
  total_weekly_cost 2 20 8 7 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_tims_bodyguard_cost_l3455_345530


namespace NUMINAMATH_CALUDE_full_price_store_a_is_125_l3455_345533

/-- The full price of a smartphone at Store A, given discount information for two stores. -/
def full_price_store_a : ℝ :=
  let discount_a : ℝ := 0.08
  let price_b : ℝ := 130
  let discount_b : ℝ := 0.10
  let price_difference : ℝ := 2

  -- Define the equation based on the given conditions
  let equation : ℝ → Prop := fun p =>
    p * (1 - discount_a) = price_b * (1 - discount_b) - price_difference

  -- Assert that 125 satisfies the equation
  125

theorem full_price_store_a_is_125 :
  full_price_store_a = 125 := by sorry

end NUMINAMATH_CALUDE_full_price_store_a_is_125_l3455_345533


namespace NUMINAMATH_CALUDE_triangle_circle_area_ratio_l3455_345526

theorem triangle_circle_area_ratio : 
  let a : ℝ := 8
  let b : ℝ := 15
  let c : ℝ := 17
  let s : ℝ := (a + b + c) / 2
  let triangle_area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let circle_radius : ℝ := c / 2
  let circle_area : ℝ := π * circle_radius^2
  let semicircle_area : ℝ := circle_area / 2
  let outside_triangle_area : ℝ := semicircle_area - triangle_area
  abs ((outside_triangle_area / semicircle_area) - 0.471) < 0.001 := by
sorry

end NUMINAMATH_CALUDE_triangle_circle_area_ratio_l3455_345526


namespace NUMINAMATH_CALUDE_exists_n_sum_of_digits_square_eq_2002_l3455_345577

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a positive integer n such that the sum of the digits of n^2 is 2002 -/
theorem exists_n_sum_of_digits_square_eq_2002 : ∃ n : ℕ+, sumOfDigits (n^2) = 2002 := by sorry

end NUMINAMATH_CALUDE_exists_n_sum_of_digits_square_eq_2002_l3455_345577


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l3455_345544

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ)
  (h1 : total_children = 660)
  (h2 : absent_children = 330)
  (h3 : extra_bananas = 2) :
  ∃ (initial_bananas : ℕ),
    initial_bananas * total_children = (initial_bananas + extra_bananas) * (total_children - absent_children) ∧
    initial_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l3455_345544


namespace NUMINAMATH_CALUDE_a_10_value_l3455_345543

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem a_10_value (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 7 = 10 → q = -2 → a 10 = -80 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l3455_345543


namespace NUMINAMATH_CALUDE_min_k_for_inequality_l3455_345516

theorem min_k_for_inequality (k : ℝ) : 
  (∀ x : ℝ, x > 0 → k * x ≥ (Real.sin x) / (2 + Real.cos x)) ↔ k ≥ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_min_k_for_inequality_l3455_345516


namespace NUMINAMATH_CALUDE_study_time_for_average_75_l3455_345584

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  studyTime : ℝ
  score : ℝ
  ratio : ℝ
  rel : score = ratio * studyTime

/-- Proves that 4.5 hours of study will result in a score of 90, given the initial condition -/
theorem study_time_for_average_75 
  (initial : StudyScoreRelation) 
  (h_initial : initial.studyTime = 3 ∧ initial.score = 60) :
  ∃ (second : StudyScoreRelation), 
    second.studyTime = 4.5 ∧ 
    second.score = 90 ∧ 
    (initial.score + second.score) / 2 = 75 ∧
    second.ratio = initial.ratio := by
  sorry

end NUMINAMATH_CALUDE_study_time_for_average_75_l3455_345584


namespace NUMINAMATH_CALUDE_logarithmic_equality_implies_zero_product_l3455_345573

theorem logarithmic_equality_implies_zero_product (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a - b) * Real.log c + (b - c) * Real.log a + (c - a) * Real.log b = 0) :
  (a - b) * (b - c) * (c - a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equality_implies_zero_product_l3455_345573


namespace NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l3455_345571

theorem phoenix_airport_on_time_rate (late_flights : ℕ) (on_time_flights : ℕ) (additional_on_time_flights : ℕ) :
  late_flights = 1 →
  on_time_flights = 3 →
  additional_on_time_flights = 2 →
  (on_time_flights + additional_on_time_flights : ℚ) / (late_flights + on_time_flights + additional_on_time_flights) > 83.33 / 100 := by
sorry

end NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l3455_345571


namespace NUMINAMATH_CALUDE_max_squares_covered_two_inch_card_l3455_345579

/-- Represents a square card -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard square -/
structure BoardSquare where
  side_length : ℝ

/-- Calculates the maximum number of board squares that can be covered by a card -/
def max_squares_covered (card : Card) (board_square : BoardSquare) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a board of 1-inch squares -/
theorem max_squares_covered_two_inch_card :
  let card := Card.mk 2
  let board_square := BoardSquare.mk 1
  max_squares_covered card board_square = 16 := by
    sorry

end NUMINAMATH_CALUDE_max_squares_covered_two_inch_card_l3455_345579


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2310_l3455_345592

theorem smallest_prime_factor_of_2310 : Nat.minFac 2310 = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2310_l3455_345592


namespace NUMINAMATH_CALUDE_tangent_line_to_exponential_l3455_345554

/-- If the line y = x + t is tangent to the curve y = e^x, then t = 1 -/
theorem tangent_line_to_exponential (t : ℝ) : 
  (∃ x₀ : ℝ, (x₀ + t = Real.exp x₀) ∧ 
             (1 = Real.exp x₀)) → 
  t = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_exponential_l3455_345554


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_l3455_345527

theorem polynomial_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 3001*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 118 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_l3455_345527


namespace NUMINAMATH_CALUDE_function_inequality_l3455_345521

theorem function_inequality (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ x, (x - 1) * (deriv f x) < 0) : 
  f 0 + f 2 < 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3455_345521


namespace NUMINAMATH_CALUDE_gcd_176_88_l3455_345506

theorem gcd_176_88 : Nat.gcd 176 88 = 88 := by
  sorry

end NUMINAMATH_CALUDE_gcd_176_88_l3455_345506


namespace NUMINAMATH_CALUDE_polygon_sides_l3455_345502

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : 
  n > 2 ∧ sum_angles = 2190 ∧ sum_angles = (n - 3) * 180 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3455_345502


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3455_345583

theorem polynomial_division_theorem (x : ℝ) :
  ∃ r : ℝ, (5 * x^2 - 5 * x + 3) * (2 * x + 4) + r = 10 * x^3 + 20 * x^2 - 9 * x + 6 ∧ 
  (∃ c : ℝ, r = c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3455_345583


namespace NUMINAMATH_CALUDE_max_ratio_of_two_digit_integers_with_mean_45_l3455_345599

theorem max_ratio_of_two_digit_integers_with_mean_45 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y : ℚ) / 2 = 45 →
  ∀ z : ℚ,
  (z : ℚ) = x / y →
  z ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_of_two_digit_integers_with_mean_45_l3455_345599


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3455_345513

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3455_345513


namespace NUMINAMATH_CALUDE_cookies_packs_l3455_345586

theorem cookies_packs (total packs_cake packs_chocolate : ℕ) 
  (h1 : total = 42)
  (h2 : packs_cake = 22)
  (h3 : packs_chocolate = 16) :
  total - packs_cake - packs_chocolate = 4 := by
  sorry

end NUMINAMATH_CALUDE_cookies_packs_l3455_345586


namespace NUMINAMATH_CALUDE_fraction_sum_equals_seven_l3455_345540

theorem fraction_sum_equals_seven : 
  let U := (1 / (4 - Real.sqrt 15)) - (1 / (Real.sqrt 15 - Real.sqrt 14)) + 
           (1 / (Real.sqrt 14 - Real.sqrt 13)) - (1 / (Real.sqrt 13 - Real.sqrt 12)) + 
           (1 / (Real.sqrt 12 - 3))
  U = 7 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_seven_l3455_345540


namespace NUMINAMATH_CALUDE_operation_equality_l3455_345524

-- Define a custom type for the allowed operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_equality (star mul : Operation) :
  (applyOp star 20 5) / (applyOp mul 15 5) = 1 →
  (applyOp star 8 4) / (applyOp mul 10 2) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_operation_equality_l3455_345524


namespace NUMINAMATH_CALUDE_base8_to_base5_conversion_l3455_345517

-- Define a function to convert from base 8 to base 10
def base8_to_base10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from base 10 to base 5
def base10_to_base5 (n : Nat) : Nat :=
  let thousands := n / 625
  let hundreds := (n % 625) / 125
  let tens := ((n % 625) % 125) / 25
  let ones := (((n % 625) % 125) % 25) / 5
  thousands * 1000 + hundreds * 100 + tens * 10 + ones

theorem base8_to_base5_conversion :
  base10_to_base5 (base8_to_base10 653) = 3202 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base5_conversion_l3455_345517


namespace NUMINAMATH_CALUDE_distance_between_points_l3455_345558

/-- The distance between points (0,15,5) and (8,0,12) in 3D space is √338. -/
theorem distance_between_points : Real.sqrt 338 = Real.sqrt ((8 - 0)^2 + (0 - 15)^2 + (12 - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3455_345558


namespace NUMINAMATH_CALUDE_tank_capacity_l3455_345568

/-- Represents a cylindrical tank with a given capacity and current fill level. -/
structure CylindricalTank where
  capacity : ℝ
  fill_percentage : ℝ
  current_volume : ℝ

/-- 
Theorem: Given a cylindrical tank that contains 60 liters of water when it is 40% full, 
the total capacity of the tank when it is completely full is 150 liters.
-/
theorem tank_capacity (tank : CylindricalTank) 
  (h1 : tank.fill_percentage = 0.4)
  (h2 : tank.current_volume = 60) : 
  tank.capacity = 150 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l3455_345568


namespace NUMINAMATH_CALUDE_simplify_expression_l3455_345511

theorem simplify_expression (t : ℝ) (h : t ≠ 0) :
  (t^5 * t^3) / t^4 = t^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3455_345511


namespace NUMINAMATH_CALUDE_adjusted_equilateral_triangle_l3455_345567

/-- Given a triangle XYZ that was originally equilateral, prove that if angle X is decreased by 5 degrees, 
    then angles Y and Z will each measure 62.5 degrees. -/
theorem adjusted_equilateral_triangle (X Y Z : ℝ) : 
  X + Y + Z = 180 →  -- Sum of angles in a triangle is 180°
  X = 55 →           -- Angle X after decrease
  Y = Z →            -- Angles Y and Z remain equal
  Y = 62.5 ∧ Z = 62.5 := by
sorry

end NUMINAMATH_CALUDE_adjusted_equilateral_triangle_l3455_345567


namespace NUMINAMATH_CALUDE_clothing_factory_payment_theorem_l3455_345529

/-- Represents the payment calculation for two discount plans in a clothing factory. -/
def ClothingFactoryPayment (x : ℕ) : Prop :=
  let suitPrice : ℕ := 400
  let tiePrice : ℕ := 80
  let numSuits : ℕ := 20
  let y₁ : ℕ := suitPrice * numSuits + (x - numSuits) * tiePrice
  let y₂ : ℕ := (suitPrice * numSuits + tiePrice * x) * 9 / 10
  (x > 20) →
  (y₁ = 80 * x + 6400) ∧
  (y₂ = 72 * x + 7200) ∧
  (x = 30 → y₁ < y₂)

theorem clothing_factory_payment_theorem :
  ∀ x : ℕ, ClothingFactoryPayment x :=
sorry

end NUMINAMATH_CALUDE_clothing_factory_payment_theorem_l3455_345529


namespace NUMINAMATH_CALUDE_sams_remaining_pennies_l3455_345561

/-- Given an initial amount of pennies and an amount spent, calculate the remaining pennies -/
def remaining_pennies (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem: Sam's remaining pennies -/
theorem sams_remaining_pennies :
  remaining_pennies 98 93 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_pennies_l3455_345561


namespace NUMINAMATH_CALUDE_no_solution_factorial_power_l3455_345596

theorem no_solution_factorial_power (n k : ℕ) (hn : n > 5) (hk : k > 0) :
  (Nat.factorial (n - 1) + 1 ≠ n ^ k) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_factorial_power_l3455_345596


namespace NUMINAMATH_CALUDE_basketball_tryouts_l3455_345565

theorem basketball_tryouts (girls boys called_back : ℕ) 
  (h1 : girls = 42)
  (h2 : boys = 80)
  (h3 : called_back = 25) :
  girls + boys - called_back = 97 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l3455_345565


namespace NUMINAMATH_CALUDE_solution_interval_l3455_345562

theorem solution_interval (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9 ↔ 63 / 26 < x ∧ x ≤ 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l3455_345562


namespace NUMINAMATH_CALUDE_lathe_processing_time_l3455_345520

/-- Given that 3 lathes can process 180 parts in 4 hours,
    prove that 5 lathes will process 600 parts in 8 hours. -/
theorem lathe_processing_time
  (initial_lathes : ℕ)
  (initial_parts : ℕ)
  (initial_hours : ℕ)
  (target_lathes : ℕ)
  (target_parts : ℕ)
  (h1 : initial_lathes = 3)
  (h2 : initial_parts = 180)
  (h3 : initial_hours = 4)
  (h4 : target_lathes = 5)
  (h5 : target_parts = 600)
  : (target_parts : ℚ) / (target_lathes : ℚ) * (initial_lathes : ℚ) / (initial_parts : ℚ) * (initial_hours : ℚ) = 8 := by
  sorry


end NUMINAMATH_CALUDE_lathe_processing_time_l3455_345520


namespace NUMINAMATH_CALUDE_bread_products_wasted_l3455_345519

/-- Calculates the pounds of bread products wasted in a food fight scenario -/
theorem bread_products_wasted (minimum_wage hours_worked meat_pounds meat_price 
  fruit_veg_pounds fruit_veg_price bread_price janitor_hours janitor_normal_pay : ℝ) 
  (h1 : minimum_wage = 8)
  (h2 : hours_worked = 50)
  (h3 : meat_pounds = 20)
  (h4 : meat_price = 5)
  (h5 : fruit_veg_pounds = 15)
  (h6 : fruit_veg_price = 4)
  (h7 : bread_price = 1.5)
  (h8 : janitor_hours = 10)
  (h9 : janitor_normal_pay = 10) : 
  (minimum_wage * hours_worked - 
   (meat_pounds * meat_price + 
    fruit_veg_pounds * fruit_veg_price + 
    janitor_hours * janitor_normal_pay * 1.5)) / bread_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_bread_products_wasted_l3455_345519


namespace NUMINAMATH_CALUDE_circle_ellipse_tangent_l3455_345504

-- Define the circle M
def circle_M (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*m*x - 3 = 0

-- Define the ellipse C
def ellipse_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2/a^2 + y^2/3 = 1

-- Define the line l
def line_l (c : ℝ) (x y : ℝ) : Prop :=
  x = -c

theorem circle_ellipse_tangent (m c a : ℝ) :
  m < 0 →  -- m is negative
  (∀ x y, circle_M m x y → (x + m)^2 + y^2 = 4) →  -- radius of M is 2
  (∃ x y, ellipse_C a x y ∧ x = -c ∧ y = 0) →  -- left focus of C is F(-c, 0)
  (∀ x y, line_l c x y → (x - 1)^2 = 4) →  -- l is tangent to M
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_ellipse_tangent_l3455_345504


namespace NUMINAMATH_CALUDE_dirt_pile_volume_decomposition_l3455_345542

/-- Represents the dimensions of a rectangular storage bin -/
structure BinDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the parameters of a dirt pile around the storage bin -/
structure DirtPileParams where
  slantDistance : ℝ

/-- Calculates the volume of the dirt pile around a storage bin -/
def dirtPileVolume (bin : BinDimensions) (pile : DirtPileParams) : ℝ :=
  sorry

theorem dirt_pile_volume_decomposition (bin : BinDimensions) (pile : DirtPileParams) :
  bin.length = 10 ∧ bin.width = 12 ∧ bin.height = 3 ∧ pile.slantDistance = 4 →
  ∃ (m n : ℕ), dirtPileVolume bin pile = m + n * Real.pi ∧ m + n = 280 :=
sorry

end NUMINAMATH_CALUDE_dirt_pile_volume_decomposition_l3455_345542


namespace NUMINAMATH_CALUDE_range_of_a_l3455_345522

-- Define the inequalities p and q
def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the theorem
theorem range_of_a :
  (∃ x : ℝ, (¬(p x) ∧ q x a) ∨ (p x ∧ ¬(q x a))) →
  (a ∈ Set.Icc (0 : ℝ) (1/2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3455_345522


namespace NUMINAMATH_CALUDE_money_distribution_l3455_345595

/-- Given three people A, B, and C with a total of 1000 rupees between them,
    where B and C together have 600 rupees, and C has 300 rupees,
    prove that A and C together have 700 rupees. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 1000 →
  B + C = 600 →
  C = 300 →
  A + C = 700 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3455_345595


namespace NUMINAMATH_CALUDE_gcd_128_144_256_l3455_345507

theorem gcd_128_144_256 : Nat.gcd 128 (Nat.gcd 144 256) = 128 := by sorry

end NUMINAMATH_CALUDE_gcd_128_144_256_l3455_345507


namespace NUMINAMATH_CALUDE_calculate_ants_monroe_ants_l3455_345518

/-- Given a collection of spiders and ants, calculate the number of ants -/
theorem calculate_ants (num_spiders : ℕ) (total_legs : ℕ) (spider_legs : ℕ) (ant_legs : ℕ) : ℕ :=
  let num_ants := (total_legs - num_spiders * spider_legs) / ant_legs
  num_ants

/-- Prove that Monroe has 12 ants in his collection -/
theorem monroe_ants : 
  let num_spiders : ℕ := 8
  let total_legs : ℕ := 136
  let spider_legs : ℕ := 8
  let ant_legs : ℕ := 6
  calculate_ants num_spiders total_legs spider_legs ant_legs = 12 := by
  sorry

end NUMINAMATH_CALUDE_calculate_ants_monroe_ants_l3455_345518


namespace NUMINAMATH_CALUDE_expression_simplification_l3455_345590

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (((-2 * x + y)^2 - (2 * x - y) * (y + 2 * x) - 6 * y) / (2 * y)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3455_345590


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l3455_345594

/-- Calculate the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The problem statement -/
theorem ball_bounce_distance :
  let initialHeight : ℝ := 150
  let reboundFactor : ℝ := 3/4
  let bounces : ℕ := 5
  totalDistance initialHeight reboundFactor bounces = 765.703125 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l3455_345594


namespace NUMINAMATH_CALUDE_fuel_cost_per_liter_l3455_345570

-- Define constants
def service_cost : ℝ := 2.10
def mini_vans : ℕ := 3
def trucks : ℕ := 2
def total_cost : ℝ := 299.1
def mini_van_tank : ℝ := 65
def truck_tank_multiplier : ℝ := 2.2  -- 120% bigger means 2.2 times the size

-- Define functions
def total_service_cost : ℝ := service_cost * (mini_vans + trucks)
def truck_tank : ℝ := mini_van_tank * truck_tank_multiplier
def total_fuel_volume : ℝ := mini_vans * mini_van_tank + trucks * truck_tank
def fuel_cost : ℝ := total_cost - total_service_cost

-- Theorem to prove
theorem fuel_cost_per_liter : fuel_cost / total_fuel_volume = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_per_liter_l3455_345570


namespace NUMINAMATH_CALUDE_seryozha_breakfast_impossibility_l3455_345535

theorem seryozha_breakfast_impossibility :
  ¬∃ (x y z : ℕ), x + 2*y + 3*z = 100 ∧ 3*x + 4*y + 5*z = 166 :=
by sorry

end NUMINAMATH_CALUDE_seryozha_breakfast_impossibility_l3455_345535


namespace NUMINAMATH_CALUDE_biotech_job_count_l3455_345510

/-- Represents the class of 2000 biotechnology graduates --/
structure BiotechClass :=
  (total : ℕ)
  (secondDegree : ℕ)
  (bothJobAndDegree : ℕ)
  (neither : ℕ)

/-- Calculates the number of graduates who found a job --/
def graduatesWithJob (c : BiotechClass) : ℕ :=
  c.total - c.neither - (c.secondDegree - c.bothJobAndDegree)

/-- Theorem: In the given biotech class, 32 graduates found a job --/
theorem biotech_job_count (c : BiotechClass) 
  (h1 : c.total = 73)
  (h2 : c.secondDegree = 45)
  (h3 : c.bothJobAndDegree = 13)
  (h4 : c.neither = 9) :
  graduatesWithJob c = 32 := by
sorry

end NUMINAMATH_CALUDE_biotech_job_count_l3455_345510


namespace NUMINAMATH_CALUDE_point_in_plane_region_l3455_345576

def plane_region (x y : ℝ) : Prop := 2 * x + y - 6 < 0

theorem point_in_plane_region :
  plane_region 0 1 ∧
  ¬ plane_region 5 0 ∧
  ¬ plane_region 0 7 ∧
  ¬ plane_region 2 3 :=
by sorry

end NUMINAMATH_CALUDE_point_in_plane_region_l3455_345576


namespace NUMINAMATH_CALUDE_statue_cost_l3455_345553

theorem statue_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 670 ∧ 
  profit_percentage = 35 ∧ 
  selling_price = original_cost * (1 + profit_percentage / 100) →
  original_cost = 496.30 := by
sorry

end NUMINAMATH_CALUDE_statue_cost_l3455_345553


namespace NUMINAMATH_CALUDE_kyoko_balls_correct_l3455_345550

/-- The number of balls Kyoko bought -/
def num_balls : ℕ := 3

/-- The cost of each ball in dollars -/
def cost_per_ball : ℚ := 154/100

/-- The total amount Kyoko paid in dollars -/
def total_paid : ℚ := 462/100

/-- Theorem stating that the number of balls Kyoko bought is correct -/
theorem kyoko_balls_correct : 
  (cost_per_ball * num_balls : ℚ) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_kyoko_balls_correct_l3455_345550


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l3455_345560

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_expression :
  6 * (2 - i) + 4 * i * (6 - i) = 16 + 18 * i :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l3455_345560


namespace NUMINAMATH_CALUDE_cubic_difference_l3455_345581

theorem cubic_difference (x y : ℚ) 
  (h1 : x + y = 10) 
  (h2 : 2 * x - y = 16) : 
  x^3 - y^3 = 17512 / 27 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l3455_345581


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3455_345589

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_geometric : is_geometric_sequence a q)
  (h_arithmetic : is_arithmetic_sequence (λ n => match n with
    | 0 => a 3
    | 1 => 3 * a 2
    | 2 => 5 * a 1
    | _ => 0))
  (h_increasing : ∀ n : ℕ, a n < a (n + 1)) :
  q = 5 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3455_345589


namespace NUMINAMATH_CALUDE_streamer_earnings_l3455_345587

/-- Calculates the weekly earnings of a streamer given their schedule and hourly rate. -/
def weekly_earnings (days_off : ℕ) (hours_per_stream : ℕ) (hourly_rate : ℕ) : ℕ :=
  (7 - days_off) * hours_per_stream * hourly_rate

/-- Theorem stating that a streamer with the given schedule earns $160 per week. -/
theorem streamer_earnings :
  weekly_earnings 3 4 10 = 160 := by
  sorry

end NUMINAMATH_CALUDE_streamer_earnings_l3455_345587


namespace NUMINAMATH_CALUDE_final_selling_price_theorem_l3455_345569

/-- The final selling price of a batch of computers -/
def final_selling_price (a : ℝ) : ℝ :=
  a * (1 + 0.2) * (1 - 0.09)

/-- Theorem stating the final selling price calculation -/
theorem final_selling_price_theorem (a : ℝ) :
  final_selling_price a = a * (1 + 0.2) * (1 - 0.09) :=
by sorry

end NUMINAMATH_CALUDE_final_selling_price_theorem_l3455_345569


namespace NUMINAMATH_CALUDE_game_result_l3455_345528

def g (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 5 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2]
def betty_rolls : List ℕ := [10, 3, 3, 2]

theorem game_result : 
  (List.sum (List.map g allie_rolls)) * (List.sum (List.map g betty_rolls)) = 66 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l3455_345528


namespace NUMINAMATH_CALUDE_two_red_two_blue_probability_l3455_345514

def total_marbles : ℕ := 15 + 9

def red_marbles : ℕ := 15

def blue_marbles : ℕ := 9

def marbles_selected : ℕ := 4

theorem two_red_two_blue_probability :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles marbles_selected = 108 / 361 :=
by sorry

end NUMINAMATH_CALUDE_two_red_two_blue_probability_l3455_345514


namespace NUMINAMATH_CALUDE_intersection_of_specific_lines_l3455_345546

/-- The x-coordinate of the intersection point of two lines -/
def intersection_x (m₁ b₁ a₂ b₂ c₂ : ℚ) : ℚ :=
  (c₂ + b₁) / (m₁ + a₂)

/-- Theorem: The x-coordinate of the intersection point of y = 4x - 25 and 2x + y = 100 is 125/6 -/
theorem intersection_of_specific_lines :
  intersection_x 4 (-25) 2 1 100 = 125 / 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_specific_lines_l3455_345546


namespace NUMINAMATH_CALUDE_star_one_one_eq_neg_eleven_l3455_345549

/-- Definition of the * operation for rational numbers -/
def star (a b c : ℚ) (x y : ℚ) : ℚ := a * x + b * y + c

/-- Theorem: Given the conditions, 1 * 1 = -11 -/
theorem star_one_one_eq_neg_eleven 
  (a b c : ℚ) 
  (h1 : star a b c 3 5 = 15) 
  (h2 : star a b c 4 7 = 28) : 
  star a b c 1 1 = -11 := by
  sorry

end NUMINAMATH_CALUDE_star_one_one_eq_neg_eleven_l3455_345549


namespace NUMINAMATH_CALUDE_smallest_n_for_183_div_11_l3455_345538

theorem smallest_n_for_183_div_11 :
  ∃! n : ℕ, (183 + n) % 11 = 0 ∧ ∀ m : ℕ, m < n → (183 + m) % 11 ≠ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_183_div_11_l3455_345538


namespace NUMINAMATH_CALUDE_sector_central_angle_l3455_345505

/-- Given a circular sector with area 6 cm² and radius 2 cm, prove its central angle is 3 radians. -/
theorem sector_central_angle (area : ℝ) (radius : ℝ) (h1 : area = 6) (h2 : radius = 2) :
  (2 * area) / (radius ^ 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3455_345505


namespace NUMINAMATH_CALUDE_solve_food_bank_problem_l3455_345572

def food_bank_problem (first_week_donation : ℝ) (second_week_multiplier : ℝ) (remaining_food : ℝ) : Prop :=
  let total_donation := first_week_donation + (second_week_multiplier * first_week_donation)
  let food_given_out := total_donation - remaining_food
  let percentage_given_out := (food_given_out / total_donation) * 100
  percentage_given_out = 70

theorem solve_food_bank_problem :
  food_bank_problem 40 2 36 := by
  sorry

end NUMINAMATH_CALUDE_solve_food_bank_problem_l3455_345572


namespace NUMINAMATH_CALUDE_prob_two_boys_from_three_boys_one_girl_l3455_345523

/-- The probability of selecting 2 boys from a group of 3 boys and 1 girl is 1/2 -/
theorem prob_two_boys_from_three_boys_one_girl :
  let total_students : ℕ := 4
  let num_boys : ℕ := 3
  let num_girls : ℕ := 1
  let students_to_select : ℕ := 2
  (Nat.choose num_boys students_to_select : ℚ) / (Nat.choose total_students students_to_select) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_prob_two_boys_from_three_boys_one_girl_l3455_345523


namespace NUMINAMATH_CALUDE_time_difference_l3455_345566

theorem time_difference (brian_time todd_time : ℕ) 
  (h1 : brian_time = 96) 
  (h2 : todd_time = 88) : 
  brian_time - todd_time = 8 := by
sorry

end NUMINAMATH_CALUDE_time_difference_l3455_345566


namespace NUMINAMATH_CALUDE_angle_with_special_complement_supplement_l3455_345588

theorem angle_with_special_complement_supplement : 
  ∀ x : ℝ, 
  (0 ≤ x) ∧ (x ≤ 180) →
  (180 - x = 3 * (90 - x)) →
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_angle_with_special_complement_supplement_l3455_345588


namespace NUMINAMATH_CALUDE_expected_prize_money_l3455_345509

theorem expected_prize_money (a₁ : ℝ) : 
  a₁ > 0 →  -- Probability should be positive
  a₁ + 2 * a₁ + 4 * a₁ = 1 →  -- Sum of probabilities is 1
  700 * a₁ + 560 * (2 * a₁) + 420 * (4 * a₁) = 500 := by
  sorry

end NUMINAMATH_CALUDE_expected_prize_money_l3455_345509


namespace NUMINAMATH_CALUDE_sum_of_m_and_n_is_zero_l3455_345525

theorem sum_of_m_and_n_is_zero 
  (h1 : ∃ p : ℝ, m * n + p^2 + 4 = 0) 
  (h2 : m - n = 4) : 
  m + n = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_m_and_n_is_zero_l3455_345525


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l3455_345541

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem f_derivative_at_2 : 
  (deriv f) 2 = 6 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l3455_345541


namespace NUMINAMATH_CALUDE_light_path_length_in_cube_l3455_345563

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℝ

/-- Represents a light path in the cube -/
structure LightPath where
  cube : Cube
  startPoint : Point3D
  reflectionPoint : Point3D

/-- Calculate the length of the light path -/
def calculateLightPathLength (path : LightPath) : ℝ := sorry

/-- Theorem stating the light path length in the given cube scenario -/
theorem light_path_length_in_cube (c : Cube) (p : Point3D) :
  c.edgeLength = 10 →
  p.x = 4 →
  p.y = 3 →
  p.z = 10 →
  let path := LightPath.mk c (Point3D.mk 0 0 0) p
  calculateLightPathLength path = 50 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_light_path_length_in_cube_l3455_345563


namespace NUMINAMATH_CALUDE_ball_count_l3455_345534

theorem ball_count (white green yellow red purple : ℕ) 
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 5)
  (h4 : red = 6)
  (h5 : purple = 9)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 3/4) :
  white + green + yellow + red + purple = 60 := by sorry

end NUMINAMATH_CALUDE_ball_count_l3455_345534


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l3455_345539

/-- Two angles with parallel sides --/
structure ParallelAngles where
  α : ℝ
  β : ℝ
  x : ℝ
  parallel : Bool
  α_eq : α = 2 * x + 10
  β_eq : β = 3 * x - 20

/-- The possible values for α in the parallel angles scenario --/
def possible_α_values (angles : ParallelAngles) : Set ℝ :=
  {70, 86}

/-- Theorem stating that the possible values for α are 70° or 86° --/
theorem parallel_angles_theorem (angles : ParallelAngles) :
  angles.α ∈ possible_α_values angles :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_angles_theorem_l3455_345539


namespace NUMINAMATH_CALUDE_sandwich_cookie_cost_l3455_345537

theorem sandwich_cookie_cost (s c : ℝ) 
  (eq1 : 3 * s + 4 * c = 4.20)
  (eq2 : 4 * s + 3 * c = 4.50) : 
  4 * s + 5 * c = 5.44 := by
sorry

end NUMINAMATH_CALUDE_sandwich_cookie_cost_l3455_345537


namespace NUMINAMATH_CALUDE_binomial_expansion_arithmetic_progression_l3455_345548

theorem binomial_expansion_arithmetic_progression (n : ℕ) : 
  (∃ (a d : ℚ), 
    (1 : ℚ) = a ∧ 
    (n : ℚ) / 2 = a + d ∧ 
    (n * (n - 1) : ℚ) / 8 = a + 2 * d) ↔ 
  n = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_arithmetic_progression_l3455_345548


namespace NUMINAMATH_CALUDE_jesses_room_length_l3455_345597

theorem jesses_room_length (area : ℝ) (width : ℝ) (h1 : area = 12.0) (h2 : width = 8) :
  area / width = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_jesses_room_length_l3455_345597


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3455_345582

def on_x_axis (p : ℝ × ℝ × ℝ) : Prop :=
  p.2 = 0 ∧ p.2.2 = 0

theorem point_on_x_axis : on_x_axis (5, 0, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3455_345582
