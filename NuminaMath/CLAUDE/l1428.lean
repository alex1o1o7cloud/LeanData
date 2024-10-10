import Mathlib

namespace group_collection_theorem_l1428_142898

/-- Calculates the total collection amount in rupees for a group where each member
    contributes as many paise as there are members in the group. -/
def total_collection (group_size : ℕ) : ℚ :=
  (group_size * group_size : ℚ) / 100

/-- Proves that for a group of 99 members, where each member contributes as many
    paise as there are members, the total collection amount is 98.01 rupees. -/
theorem group_collection_theorem :
  total_collection 99 = 98.01 := by
  sorry

end group_collection_theorem_l1428_142898


namespace events_independent_l1428_142814

/-- Represents the outcome of a single coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Represents the outcome of tossing a coin twice -/
def DoubleToss := CoinToss × CoinToss

/-- Event A: the first toss is heads -/
def event_A (toss : DoubleToss) : Prop :=
  toss.1 = CoinToss.Heads

/-- Event B: the second toss is tails -/
def event_B (toss : DoubleToss) : Prop :=
  toss.2 = CoinToss.Tails

/-- The probability of an event occurring -/
def probability (event : DoubleToss → Prop) : ℝ :=
  sorry

/-- Theorem: Events A and B are mutually independent -/
theorem events_independent :
  probability (fun toss ↦ event_A toss ∧ event_B toss) =
  probability event_A * probability event_B :=
sorry

end events_independent_l1428_142814


namespace divisibility_np_minus_n_l1428_142894

theorem divisibility_np_minus_n (p : Nat) (n : Int) (h : p = 3 ∨ p = 7 ∨ p = 13) :
  ∃ k : Int, n^p - n = k * p := by
  sorry

end divisibility_np_minus_n_l1428_142894


namespace shared_foci_implies_a_equals_one_l1428_142820

-- Define the ellipse equation
def ellipse (x y a : ℝ) : Prop := x^2 / 4 + y^2 / a^2 = 1

-- Define the hyperbola equation
def hyperbola (x y a : ℝ) : Prop := x^2 / a - y^2 / 2 = 1

-- Theorem statement
theorem shared_foci_implies_a_equals_one :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, ellipse x y a ↔ hyperbola x y a) →
  a = 1 :=
by sorry

end shared_foci_implies_a_equals_one_l1428_142820


namespace simple_interest_problem_l1428_142810

theorem simple_interest_problem (P R : ℝ) : 
  P > 0 → R > 0 → 
  P * (R + 3) * 3 / 100 - P * R * 3 / 100 = 90 → 
  P = 1000 := by
  sorry

end simple_interest_problem_l1428_142810


namespace f_of_one_l1428_142802

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_of_one (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 4)
  (h_value : f (-5) = 1) : 
  f 1 = -1 := by
  sorry

end f_of_one_l1428_142802


namespace octal_number_check_l1428_142819

def is_octal_digit (d : Nat) : Prop := d < 8

def is_octal_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 8 → is_octal_digit d

theorem octal_number_check :
  ¬ is_octal_number 8102 ∧
  ¬ is_octal_number 793 ∧
  is_octal_number 214 ∧
  ¬ is_octal_number 998 := by sorry

end octal_number_check_l1428_142819


namespace second_to_last_term_l1428_142863

-- Define the sequence type
def Sequence := Fin 201 → ℕ

-- Define the properties of the sequence
def ValidSequence (a : Sequence) : Prop :=
  (a 0 = 19999) ∧ 
  (a 200 = 19999) ∧
  (∃ t : ℕ+, ∀ n : Fin 199, 
    a (n + 1) + t = (a n + a (n + 2)) / 2)

-- Theorem statement
theorem second_to_last_term (a : Sequence) 
  (h : ValidSequence a) : a 199 = 19800 := by
  sorry

end second_to_last_term_l1428_142863


namespace five_fourths_of_three_and_one_third_l1428_142822

theorem five_fourths_of_three_and_one_third (x : ℚ) :
  x = 3 + 1 / 3 → (5 / 4 : ℚ) * x = 25 / 6 := by
  sorry

end five_fourths_of_three_and_one_third_l1428_142822


namespace min_upper_bound_fraction_l1428_142871

theorem min_upper_bound_fraction (a₁ a₂ a₃ : ℝ) (h : a₁ ≠ 0 ∨ a₂ ≠ 0 ∨ a₃ ≠ 0) :
  ∃ M : ℝ, M = Real.sqrt 2 / 2 ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = 2 →
    (x * a₁ * a₂ + y * a₂ * a₃) / (a₁^2 + a₂^2 + a₃^2) ≤ M) ∧
  ∀ ε > 0, ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^2 + y^2 = 2 ∧
    (x * a₁ * a₂ + y * a₂ * a₃) / (a₁^2 + a₂^2 + a₃^2) > M - ε :=
by sorry

end min_upper_bound_fraction_l1428_142871


namespace no_snow_probability_l1428_142862

theorem no_snow_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end no_snow_probability_l1428_142862


namespace divisibility_of_forms_l1428_142840

/-- Represents a six-digit number in the form ABCDEF --/
def SixDigitNumber (A B C D E F : ℕ) : ℕ := 
  100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F

/-- The form PQQPQQ --/
def FormA (P Q : ℕ) : ℕ := SixDigitNumber P Q Q P Q Q

/-- The form PQPQPQ --/
def FormB (P Q : ℕ) : ℕ := SixDigitNumber P Q P Q P Q

/-- The form QPQPQP --/
def FormC (P Q : ℕ) : ℕ := SixDigitNumber Q P Q P Q P

/-- The form PPPPPP --/
def FormD (P : ℕ) : ℕ := SixDigitNumber P P P P P P

/-- The form PPPQQQ --/
def FormE (P Q : ℕ) : ℕ := SixDigitNumber P P P Q Q Q

theorem divisibility_of_forms (P Q : ℕ) :
  (∃ (k : ℕ), FormA P Q = 7 * k) ∧
  (∃ (k : ℕ), FormB P Q = 7 * k) ∧
  (∃ (k : ℕ), FormC P Q = 7 * k) ∧
  (∃ (k : ℕ), FormD P = 7 * k) ∧
  ¬(∀ (P Q : ℕ), ∃ (k : ℕ), FormE P Q = 7 * k) := by
  sorry

end divisibility_of_forms_l1428_142840


namespace remainder_2519_div_6_l1428_142897

theorem remainder_2519_div_6 : 2519 % 6 = 5 := by
  sorry

end remainder_2519_div_6_l1428_142897


namespace factorial_8_divisors_l1428_142821

def factorial_8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

theorem factorial_8_divisors :
  (factorial_8 = 2^7 * 3^2 * 5 * 7) →
  (∃ (even_divisors : Finset ℕ) (even_divisors_multiple_2_3 : Finset ℕ),
    (∀ d ∈ even_divisors, d ∣ factorial_8 ∧ 2 ∣ d) ∧
    (∀ d ∈ even_divisors_multiple_2_3, d ∣ factorial_8 ∧ 2 ∣ d ∧ 3 ∣ d) ∧
    even_divisors.card = 84 ∧
    even_divisors_multiple_2_3.card = 56) :=
by sorry

end factorial_8_divisors_l1428_142821


namespace element_in_complement_l1428_142828

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {1, 5}

-- Define set P
def P : Set Nat := {2, 4}

-- Theorem statement
theorem element_in_complement : 3 ∈ (U \ (M ∪ P)) := by sorry

end element_in_complement_l1428_142828


namespace birthday_month_l1428_142808

def is_valid_day (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 31

def is_valid_month (m : ℕ) : Prop := 1 ≤ m ∧ m ≤ 12

theorem birthday_month (d m : ℕ) (h1 : is_valid_day d) (h2 : is_valid_month m) 
  (h3 : d * m = 248) : m = 8 := by
  sorry

end birthday_month_l1428_142808


namespace quadratic_equation_solution_l1428_142874

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 2*x₁ - 3 = 0) ∧ 
  (x₂^2 - 2*x₂ - 3 = 0) ∧ 
  x₁ = 3 ∧ 
  x₂ = -1 :=
by sorry

end quadratic_equation_solution_l1428_142874


namespace event_probability_l1428_142865

theorem event_probability (P_B P_AB P_AorB : ℝ) 
  (hB : P_B = 0.4)
  (hAB : P_AB = 0.25)
  (hAorB : P_AorB = 0.6) :
  ∃ P_A : ℝ, P_A = 0.45 ∧ P_AorB = P_A + P_B - P_AB :=
sorry

end event_probability_l1428_142865


namespace exists_integer_term_l1428_142806

def sequence_rule (x : ℚ) : ℚ := x + 1 / (Int.floor x)

def is_valid_sequence (x : ℕ → ℚ) : Prop :=
  x 1 > 1 ∧ ∀ n : ℕ, x (n + 1) = sequence_rule (x n)

theorem exists_integer_term (x : ℕ → ℚ) (h : is_valid_sequence x) :
  ∃ k : ℕ, ∃ m : ℤ, x k = m :=
sorry

end exists_integer_term_l1428_142806


namespace sin_in_M_l1428_142826

/-- The set of functions f that satisfy f(x + T) = T * f(x) for some non-zero constant T -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ (T : ℝ), T ≠ 0 ∧ ∀ x, f (x + T) = T * f x}

/-- Theorem stating the condition for sin(kx) to be in set M -/
theorem sin_in_M (k : ℝ) : 
  (fun x ↦ Real.sin (k * x)) ∈ M ↔ ∃ m : ℤ, k = m * Real.pi :=
sorry

end sin_in_M_l1428_142826


namespace apples_handed_out_to_students_l1428_142815

/-- Given a cafeteria with apples, prove the number of apples handed out to students. -/
theorem apples_handed_out_to_students 
  (initial_apples : ℕ) 
  (apples_per_pie : ℕ) 
  (pies_made : ℕ) 
  (h1 : initial_apples = 51)
  (h2 : apples_per_pie = 5)
  (h3 : pies_made = 2) :
  initial_apples - (apples_per_pie * pies_made) = 41 :=
by sorry

end apples_handed_out_to_students_l1428_142815


namespace center_value_of_arithmetic_array_l1428_142892

/-- Represents a 4x4 array where each row and column is an arithmetic sequence -/
def ArithmeticArray := Fin 4 → Fin 4 → ℚ

/-- The common difference of an arithmetic sequence given its first and last terms -/
def commonDifference (a₁ a₄ : ℚ) : ℚ := (a₄ - a₁) / 3

/-- Checks if a sequence is arithmetic -/
def isArithmeticSequence (seq : Fin 4 → ℚ) : Prop :=
  ∀ i j : Fin 4, i.val < j.val → seq j - seq i = commonDifference (seq 0) (seq 3) * (j - i)

/-- Properties of our specific arithmetic array -/
def isValidArray (arr : ArithmeticArray) : Prop :=
  (∀ i : Fin 4, isArithmeticSequence (λ j => arr i j)) ∧  -- Each row is arithmetic
  (∀ j : Fin 4, isArithmeticSequence (λ i => arr i j)) ∧  -- Each column is arithmetic
  arr 0 0 = 3 ∧ arr 0 3 = 21 ∧                            -- First row conditions
  arr 3 0 = 15 ∧ arr 3 3 = 45                             -- Fourth row conditions

theorem center_value_of_arithmetic_array (arr : ArithmeticArray) 
  (h : isValidArray arr) : arr 1 1 = 14 + 1/3 := by
  sorry

end center_value_of_arithmetic_array_l1428_142892


namespace meaningful_reciprocal_range_l1428_142867

theorem meaningful_reciprocal_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x + 2)) ↔ x ≠ -2 := by
  sorry

end meaningful_reciprocal_range_l1428_142867


namespace initial_crayons_l1428_142855

theorem initial_crayons (initial : ℕ) : initial + 3 = 12 → initial = 9 := by
  sorry

end initial_crayons_l1428_142855


namespace book_arrangement_count_book_arrangement_theorem_l1428_142845

theorem book_arrangement_count : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let german_books : ℕ := 3
  let spanish_books : ℕ := 5
  let arabic_group : ℕ := 1  -- Treat Arabic books as one unit
  let spanish_group : ℕ := 1  -- Treat Spanish books as one unit
  let german_group : ℕ := 1  -- Treat German books as one ordered unit
  let total_groups : ℕ := arabic_group + spanish_group + german_group
  let group_arrangements : ℕ := Nat.factorial total_groups
  let arabic_arrangements : ℕ := Nat.factorial arabic_books
  let spanish_arrangements : ℕ := Nat.factorial spanish_books

  group_arrangements * arabic_arrangements * spanish_arrangements

-- Prove that book_arrangement_count equals 4320
theorem book_arrangement_theorem : book_arrangement_count = 4320 := by
  sorry

end book_arrangement_count_book_arrangement_theorem_l1428_142845


namespace average_visitors_is_290_l1428_142883

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def average_visitors_per_day (sunday_visitors : ℕ) (other_day_visitors : ℕ) : ℚ :=
  let total_sundays := 5
  let total_other_days := 25
  let total_visitors := sunday_visitors * total_sundays + other_day_visitors * total_other_days
  total_visitors / 30

/-- Theorem stating that the average number of visitors per day is 290 -/
theorem average_visitors_is_290 :
  average_visitors_per_day 540 240 = 290 := by
  sorry

end average_visitors_is_290_l1428_142883


namespace equal_chords_length_squared_l1428_142878

/-- Two circles with radii 10 and 8, centers 15 units apart -/
structure CircleConfiguration where
  center_distance : ℝ
  radius1 : ℝ
  radius2 : ℝ
  center_distance_eq : center_distance = 15
  radius1_eq : radius1 = 10
  radius2_eq : radius2 = 8

/-- Point of intersection of the two circles -/
def IntersectionPoint (config : CircleConfiguration) : Type :=
  { p : ℝ × ℝ // 
    (p.1 - 0)^2 + p.2^2 = config.radius1^2 ∧ 
    (p.1 - config.center_distance)^2 + p.2^2 = config.radius2^2 }

/-- Line through intersection point creating equal chords -/
structure EqualChordsLine (config : CircleConfiguration) where
  p : IntersectionPoint config
  q : ℝ × ℝ
  r : ℝ × ℝ
  on_circle1 : (q.1 - 0)^2 + q.2^2 = config.radius1^2
  on_circle2 : (r.1 - config.center_distance)^2 + r.2^2 = config.radius2^2
  equal_chords : (q.1 - p.val.1)^2 + (q.2 - p.val.2)^2 = (r.1 - p.val.1)^2 + (r.2 - p.val.2)^2

/-- Theorem: The square of the length of QP is 164 -/
theorem equal_chords_length_squared 
  (config : CircleConfiguration) 
  (line : EqualChordsLine config) : 
  (line.q.1 - line.p.val.1)^2 + (line.q.2 - line.p.val.2)^2 = 164 := by
  sorry

end equal_chords_length_squared_l1428_142878


namespace f_no_real_roots_l1428_142877

/-- Defines the polynomial f(x) for a given positive integer n -/
def f (n : ℕ+) (x : ℝ) : ℝ :=
  (2 * n.val + 1) * x^(2 * n.val) - 2 * n.val * x^(2 * n.val - 1) + 
  (2 * n.val - 1) * x^(2 * n.val - 2) - 3 * x^2 + 2 * x - 1

/-- Theorem stating that f(x) has no real roots for any positive integer n -/
theorem f_no_real_roots (n : ℕ+) : ∀ x : ℝ, f n x ≠ 0 := by
  sorry

end f_no_real_roots_l1428_142877


namespace perpendicular_vector_of_parallel_lines_l1428_142868

/-- Given two parallel lines l and m in 2D space, this theorem proves that
    the vector perpendicular to both lines, normalized such that its
    components sum to 7, is (2, 5). -/
theorem perpendicular_vector_of_parallel_lines :
  ∀ (l m : ℝ → ℝ × ℝ),
  (∃ (k : ℝ), k ≠ 0 ∧ (l 0).1 - (l 1).1 = k * ((m 0).1 - (m 1).1) ∧
                    (l 0).2 - (l 1).2 = k * ((m 0).2 - (m 1).2)) →
  ∃ (v : ℝ × ℝ),
    v.1 + v.2 = 7 ∧
    v.1 * ((l 0).1 - (l 1).1) + v.2 * ((l 0).2 - (l 1).2) = 0 ∧
    v = (2, 5) :=
by sorry


end perpendicular_vector_of_parallel_lines_l1428_142868


namespace conversion_factor_feet_to_miles_l1428_142873

/-- Conversion factor from feet to miles -/
def feet_per_mile : ℝ := 5280

/-- Speed of the object in miles per hour -/
def speed_mph : ℝ := 68.18181818181819

/-- Distance traveled by the object in feet -/
def distance_feet : ℝ := 400

/-- Time taken by the object in seconds -/
def time_seconds : ℝ := 4

/-- Theorem stating that the conversion factor from feet to miles is 5280 -/
theorem conversion_factor_feet_to_miles :
  feet_per_mile = (distance_feet / time_seconds) / (speed_mph / 3600) := by
  sorry

#check conversion_factor_feet_to_miles

end conversion_factor_feet_to_miles_l1428_142873


namespace circle_area_ratio_after_tripling_radius_l1428_142872

theorem circle_area_ratio_after_tripling_radius (r : ℝ) (h : r > 0) :
  (π * r^2) / (π * (3*r)^2) = 1 / 9 := by
  sorry

end circle_area_ratio_after_tripling_radius_l1428_142872


namespace permutation_of_two_equals_twelve_l1428_142880

theorem permutation_of_two_equals_twelve (n : ℕ) : n * (n - 1) = 12 → n = 4 := by
  sorry

end permutation_of_two_equals_twelve_l1428_142880


namespace james_distance_l1428_142831

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: James rode 80 miles -/
theorem james_distance : distance 16 5 = 80 := by
  sorry

end james_distance_l1428_142831


namespace coeff_x3_sum_l1428_142891

/-- The coefficient of x^3 in the expansion of (1-x)^n -/
def coeff_x3 (n : ℕ) : ℤ := (-1)^3 * Nat.choose n 3

/-- The sum of coefficients of x^3 in the expansion of (1-x)^5 + (1-x)^6 + (1-x)^7 + (1-x)^8 -/
def total_coeff : ℤ := coeff_x3 5 + coeff_x3 6 + coeff_x3 7 + coeff_x3 8

theorem coeff_x3_sum : total_coeff = -121 := by sorry

end coeff_x3_sum_l1428_142891


namespace ellipse_foci_distance_l1428_142847

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 36 is 8√2 -/
theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + y^2 = 36) → (∃ f₁ f₂ : ℝ × ℝ, 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 128) :=
by sorry

end ellipse_foci_distance_l1428_142847


namespace unpainted_area_45_degree_cross_l1428_142861

/-- The area of the unpainted region when two boards cross at 45 degrees -/
theorem unpainted_area_45_degree_cross (board_width : ℝ) (cross_angle : ℝ) : 
  board_width = 5 → cross_angle = 45 → 
  (board_width * (board_width * Real.sqrt 2)) = 25 * Real.sqrt 2 := by
  sorry

#check unpainted_area_45_degree_cross

end unpainted_area_45_degree_cross_l1428_142861


namespace max_value_fraction_max_value_achievable_l1428_142870

theorem max_value_fraction (x y : ℝ) : 
  (2 * x + Real.sqrt 2 * y) / (2 * x^4 + 4 * y^4 + 9) ≤ 1/4 :=
by sorry

theorem max_value_achievable : 
  ∃ x y : ℝ, (2 * x + Real.sqrt 2 * y) / (2 * x^4 + 4 * y^4 + 9) = 1/4 :=
by sorry

end max_value_fraction_max_value_achievable_l1428_142870


namespace kafelnikov_served_first_l1428_142899

/-- Represents a tennis player -/
inductive Player : Type
| Kafelnikov : Player
| Becker : Player

/-- Represents the result of a tennis match -/
structure MatchResult :=
  (winner : Player)
  (winner_games : Nat)
  (loser_games : Nat)

/-- Represents the serving pattern in a tennis match -/
structure ServingPattern :=
  (server_wins : Nat)
  (receiver_wins : Nat)

/-- Determines who served first in a tennis match -/
def first_server (result : MatchResult) (serving : ServingPattern) : Player :=
  sorry

/-- Theorem stating that Kafelnikov served first -/
theorem kafelnikov_served_first 
  (result : MatchResult) 
  (serving : ServingPattern) :
  result.winner = Player.Kafelnikov ∧
  result.winner_games = 6 ∧
  result.loser_games = 3 ∧
  serving.server_wins = 5 ∧
  serving.receiver_wins = 4 →
  first_server result serving = Player.Kafelnikov :=
by sorry

end kafelnikov_served_first_l1428_142899


namespace opposite_gender_officers_l1428_142836

theorem opposite_gender_officers (boys girls : ℕ) (h1 : boys = 18) (h2 : girls = 12) :
  boys * girls + girls * boys = 432 := by
  sorry

end opposite_gender_officers_l1428_142836


namespace james_basketball_score_l1428_142817

/-- Calculates the total points scored by James in a basketball game --/
def jamesScore (threePointers twoPointers freeThrows missedFreeThrows : ℕ) : ℤ :=
  3 * threePointers + 2 * twoPointers + freeThrows - missedFreeThrows

theorem james_basketball_score :
  jamesScore 13 20 5 2 = 82 := by
  sorry

end james_basketball_score_l1428_142817


namespace side_a_is_one_max_perimeter_is_three_max_perimeter_when_b_equals_c_l1428_142864

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.b * Real.cos t.A + t.a * Real.cos t.B = t.a * t.c

-- Theorem for part 1
theorem side_a_is_one (t : Triangle) (h : satisfiesCondition t) : t.a = 1 := by
  sorry

-- Theorem for part 2
theorem max_perimeter_is_three (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.A = Real.pi / 3) :
  t.a + t.b + t.c ≤ 3 := by
  sorry

-- Theorem for the maximum perimeter occurring when b = c
theorem max_perimeter_when_b_equals_c (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.A = Real.pi / 3) :
  ∃ (t' : Triangle), satisfiesCondition t' ∧ t'.A = Real.pi / 3 ∧ t'.b = t'.c ∧ t'.a + t'.b + t'.c = 3 := by
  sorry

end side_a_is_one_max_perimeter_is_three_max_perimeter_when_b_equals_c_l1428_142864


namespace mathildas_debt_l1428_142843

/-- Mathilda's debt problem -/
theorem mathildas_debt (initial_payment : ℝ) (remaining_percentage : ℝ) (original_debt : ℝ) : 
  initial_payment = 125 ∧ 
  remaining_percentage = 75 ∧ 
  initial_payment = (100 - remaining_percentage) / 100 * original_debt →
  original_debt = 500 := by
  sorry

end mathildas_debt_l1428_142843


namespace regular_polygon_sides_l1428_142825

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular polygon satisfying the given condition has 3 sides -/
theorem regular_polygon_sides : ∃ (n : ℕ), n ≥ 3 ∧ n - num_diagonals n = 3 ∧ n = 3 := by
  sorry

end regular_polygon_sides_l1428_142825


namespace quadruple_sequence_no_repetition_l1428_142842

/-- Transformation function for quadruples -/
def transform (q : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := q
  (a * b, b * c, c * d, d * a)

/-- Generates the sequence of quadruples starting from an initial quadruple -/
def quadruple_sequence (initial : ℝ × ℝ × ℝ × ℝ) : ℕ → ℝ × ℝ × ℝ × ℝ
  | 0 => initial
  | n + 1 => transform (quadruple_sequence initial n)

theorem quadruple_sequence_no_repetition (a₀ b₀ c₀ d₀ : ℝ) :
  (a₀, b₀, c₀, d₀) ≠ (1, 1, 1, 1) →
  ∀ i j : ℕ, i ≠ j →
    quadruple_sequence (a₀, b₀, c₀, d₀) i ≠ quadruple_sequence (a₀, b₀, c₀, d₀) j :=
by sorry

end quadruple_sequence_no_repetition_l1428_142842


namespace congruence_solution_l1428_142860

theorem congruence_solution (m : ℤ) : 
  (13 * m) % 47 = 8 % 47 ↔ m % 47 = 20 % 47 :=
by sorry

end congruence_solution_l1428_142860


namespace base_seven_65432_equals_16340_l1428_142895

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ (digits.length - 1 - i))) 0

theorem base_seven_65432_equals_16340 :
  base_seven_to_decimal [6, 5, 4, 3, 2] = 16340 := by
  sorry

end base_seven_65432_equals_16340_l1428_142895


namespace triangle_abc_properties_l1428_142827

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < C → C < π →
  0 < A → A < π / 3 →
  (2 * a + b) / Real.cos B = -c / Real.cos C →
  (C = 2 * π / 3 ∧ 
   ∀ A' B', 0 < A' → A' < π / 3 → 
             Real.sin A' * Real.sin B' ≤ 1 / 4) := by
  sorry

end triangle_abc_properties_l1428_142827


namespace total_pieces_l1428_142824

/-- Represents the number of small pieces in Figure n of Nair's puzzle -/
def small_pieces (n : ℕ) : ℕ := 4 * n

/-- Represents the number of large pieces in Figure n of Nair's puzzle -/
def large_pieces (n : ℕ) : ℕ := n^2 - n

/-- Theorem stating that the total number of pieces in Figure n is n^2 + 3n -/
theorem total_pieces (n : ℕ) : small_pieces n + large_pieces n = n^2 + 3*n := by
  sorry

#eval small_pieces 20 + large_pieces 20  -- Should output 460

end total_pieces_l1428_142824


namespace inequality_system_solutions_l1428_142857

def has_exactly_two_integer_solutions (m : ℝ) : Prop :=
  ∃ (x y : ℤ), x ≠ y ∧
    (x < 1 ∧ x > m - 1) ∧
    (y < 1 ∧ y > m - 1) ∧
    ∀ (z : ℤ), (z < 1 ∧ z > m - 1) → (z = x ∨ z = y)

theorem inequality_system_solutions (m : ℝ) :
  has_exactly_two_integer_solutions m ↔ -1 ≤ m ∧ m < 0 :=
by sorry

end inequality_system_solutions_l1428_142857


namespace triangle_perimeter_bound_l1428_142882

theorem triangle_perimeter_bound (a b c : ℝ) : 
  a = 7 → b = 23 → a + b > c → a + c > b → b + c > a → a + b + c < 60 := by sorry

end triangle_perimeter_bound_l1428_142882


namespace krakozyabr_population_is_32_l1428_142884

structure Krakozyabr where
  hasHorns : Bool
  hasWings : Bool

def totalKrakozyabrs (population : List Krakozyabr) : Nat :=
  population.length

theorem krakozyabr_population_is_32 
  (population : List Krakozyabr) 
  (all_have_horns_or_wings : ∀ k ∈ population, k.hasHorns ∨ k.hasWings)
  (horns_with_wings_ratio : (population.filter (λ k => k.hasHorns ∧ k.hasWings)).length = 
    (population.filter (λ k => k.hasHorns)).length / 5)
  (wings_with_horns_ratio : (population.filter (λ k => k.hasHorns ∧ k.hasWings)).length = 
    (population.filter (λ k => k.hasWings)).length / 4)
  (population_range : 25 < totalKrakozyabrs population ∧ totalKrakozyabrs population < 35) :
  totalKrakozyabrs population = 32 := by
  sorry

end krakozyabr_population_is_32_l1428_142884


namespace multiplier_problem_l1428_142844

theorem multiplier_problem (x : ℝ) (h1 : x = 11) (h2 : 3 * x = (26 - x) + 18) :
  ∃ m : ℝ, m * x = (26 - x) + 18 ∧ m = 3 := by
  sorry

end multiplier_problem_l1428_142844


namespace quadratic_equation_m_value_l1428_142853

theorem quadratic_equation_m_value : 
  ∀ m : ℤ, 
  (∀ x : ℝ, ∃ a b c : ℝ, (m - 1) * x^(m^2 + 1) + 2*x - 3 = a*x^2 + b*x + c) →
  m = -1 := by
sorry

end quadratic_equation_m_value_l1428_142853


namespace quadratic_equation_roots_l1428_142809

theorem quadratic_equation_roots (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - (2*m - 1)*x + m^2
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) ∧ 
  (x₁ ≠ x₂) ∧
  ((x₁ + 1) * (x₂ + 1) = 3) →
  m = -3 := by sorry

end quadratic_equation_roots_l1428_142809


namespace arithmetic_sequence_third_term_l1428_142850

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_20th : a 20 = 15)
  (h_21st : a 21 = 18) :
  a 3 = -36 := by
  sorry

end arithmetic_sequence_third_term_l1428_142850


namespace negation_equivalence_l1428_142885

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by sorry

end negation_equivalence_l1428_142885


namespace cricket_match_average_l1428_142881

/-- Given five cricket match scores x, y, a, b, and c, prove that their average is 36 -/
theorem cricket_match_average (x y a b c : ℝ) : 
  (x + y) / 2 = 30 →
  (a + b + c) / 3 = 40 →
  x ≤ 60 ∧ y ≤ 60 ∧ a ≤ 60 ∧ b ≤ 60 ∧ c ≤ 60 →
  (x + y ≥ 100 ∨ a + b + c ≥ 100) →
  (x + y + a + b + c) / 5 = 36 := by
  sorry

end cricket_match_average_l1428_142881


namespace perfect_square_power_of_two_plus_256_l1428_142869

theorem perfect_square_power_of_two_plus_256 (n : ℕ) :
  (∃ k : ℕ+, 2^n + 256 = k^2) → n = 11 := by
  sorry

end perfect_square_power_of_two_plus_256_l1428_142869


namespace no_universal_divisibility_l1428_142849

def concatenate_two_digits (a b : Nat) : Nat :=
  10 * a + b

def concatenate_three_digits (a n b : Nat) : Nat :=
  100 * a + 10 * n + b

theorem no_universal_divisibility :
  ∀ n : Nat, ∃ a b : Nat,
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    ¬(concatenate_two_digits a b ∣ concatenate_three_digits a n b) := by
  sorry

end no_universal_divisibility_l1428_142849


namespace quadratic_function_property_l1428_142830

theorem quadratic_function_property (b c : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  (f 1 = 0) → (f 3 = 0) → (f (-1) = 8) := by
  sorry

end quadratic_function_property_l1428_142830


namespace cubic_root_sum_cubes_l1428_142875

/-- Given roots r, s, and t of the equation 10x³ + 500x + 1500 = 0,
    prove that (r+s)³ + (t+s)³ + (r+t)³ = -450 -/
theorem cubic_root_sum_cubes (r s t : ℝ) :
  (10 * r^3 + 500 * r + 1500 = 0) →
  (10 * s^3 + 500 * s + 1500 = 0) →
  (10 * t^3 + 500 * t + 1500 = 0) →
  (r + s)^3 + (t + s)^3 + (r + t)^3 = -450 := by
  sorry


end cubic_root_sum_cubes_l1428_142875


namespace nth_root_equation_l1428_142813

theorem nth_root_equation (n : ℕ) : n = 3 →
  (((17 * Real.sqrt 5 + 38) ^ (1 / n : ℝ)) + ((17 * Real.sqrt 5 - 38) ^ (1 / n : ℝ))) = Real.sqrt 20 :=
by sorry

end nth_root_equation_l1428_142813


namespace smallest_solution_quartic_equation_l1428_142803

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 40*x^2 + 144 = 0 ∧ 
  (∀ (y : ℝ), y^4 - 40*y^2 + 144 = 0 → x ≤ y) ∧
  x = -6 := by
sorry

end smallest_solution_quartic_equation_l1428_142803


namespace eggs_used_for_omelet_l1428_142852

theorem eggs_used_for_omelet (initial_eggs : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (final_eggs : ℕ) : 
  initial_eggs = 10 →
  chickens = 2 →
  eggs_per_chicken = 3 →
  final_eggs = 11 →
  initial_eggs + chickens * eggs_per_chicken - final_eggs = 7 :=
by
  sorry

#check eggs_used_for_omelet

end eggs_used_for_omelet_l1428_142852


namespace keith_pears_l1428_142888

/-- Given that Jason picked 46 pears, Mike picked 12 pears, and the total number of pears picked was 105, prove that Keith picked 47 pears. -/
theorem keith_pears (jason_pears mike_pears total_pears : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : mike_pears = 12)
  (h3 : total_pears = 105) :
  total_pears - (jason_pears + mike_pears) = 47 :=
by sorry

end keith_pears_l1428_142888


namespace fifth_inequality_l1428_142835

theorem fifth_inequality (n : ℕ) (h : n = 6) : 
  1 + (1 / 2^2) + (1 / 3^2) + (1 / 4^2) + (1 / 5^2) + (1 / 6^2) < (2 * n - 1) / n :=
by sorry

end fifth_inequality_l1428_142835


namespace probability_no_shaded_square_l1428_142834

/-- Represents a rectangular grid with shaded squares -/
structure ShadedGrid :=
  (rows : Nat)
  (cols : Nat)
  (shaded_cols : Finset Nat)

/-- Calculates the total number of rectangles in the grid -/
def total_rectangles (grid : ShadedGrid) : Nat :=
  (grid.rows * Nat.choose grid.cols 2)

/-- Calculates the number of rectangles containing a shaded square -/
def shaded_rectangles (grid : ShadedGrid) : Nat :=
  grid.rows * (grid.shaded_cols.card * (grid.cols - grid.shaded_cols.card))

/-- Theorem stating the probability of selecting a rectangle without a shaded square -/
theorem probability_no_shaded_square (grid : ShadedGrid) 
  (h1 : grid.rows = 2)
  (h2 : grid.cols = 2005)
  (h3 : grid.shaded_cols = {1003}) :
  (total_rectangles grid - shaded_rectangles grid) / total_rectangles grid = 1002 / 2005 := by
  sorry

end probability_no_shaded_square_l1428_142834


namespace equation_has_real_root_when_K_zero_l1428_142801

/-- The equation x = K³(x³ - 3x² + 2x + 1) has at least one real root when K = 0 -/
theorem equation_has_real_root_when_K_zero :
  ∃ x : ℝ, x = 0^3 * (x^3 - 3*x^2 + 2*x + 1) :=
by
  sorry

end equation_has_real_root_when_K_zero_l1428_142801


namespace planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l1428_142838

-- Define the concept of a plane
variable (Plane : Type)

-- Define the concept of a line
variable (Line : Type)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the property of being non-coincident
variable (non_coincident : Plane → Plane → Prop)

-- Theorem 1: Two non-coincident planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel 
  (α β γ : Plane) 
  (h1 : parallel α γ) 
  (h2 : parallel β γ) 
  (h3 : non_coincident α β) : 
  parallel α β :=
sorry

-- Theorem 2: Two non-coincident planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_same_line_are_parallel 
  (α β : Plane) 
  (a : Line) 
  (h1 : perpendicular a α) 
  (h2 : perpendicular a β) 
  (h3 : non_coincident α β) : 
  parallel α β :=
sorry

end planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l1428_142838


namespace inequality_proof_l1428_142832

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum : a + b + c = 1) :
  (a - b*c)/(a + b*c) + (b - c*a)/(b + c*a) + (c - a*b)/(c + a*b) ≤ 3/2 := by
  sorry

end inequality_proof_l1428_142832


namespace bird_feeder_theorem_l1428_142846

/-- Given a bird feeder with specified capacity and feeding rate, and accounting for stolen seed, 
    calculate the number of birds fed weekly. -/
theorem bird_feeder_theorem (feeder_capacity : ℚ) (birds_per_cup : ℕ) (stolen_amount : ℚ) : 
  feeder_capacity = 2 → 
  birds_per_cup = 14 → 
  stolen_amount = 1/2 → 
  (feeder_capacity - stolen_amount) * birds_per_cup = 21 := by
  sorry

end bird_feeder_theorem_l1428_142846


namespace tom_lasagna_noodles_l1428_142879

/-- The number of packages of noodles Tom needs to buy for his lasagna -/
def noodle_packages_needed (beef_amount : ℕ) (noodle_ratio : ℕ) (existing_noodles : ℕ) (package_size : ℕ) : ℕ :=
  let total_noodles_needed := beef_amount * noodle_ratio
  let additional_noodles_needed := total_noodles_needed - existing_noodles
  (additional_noodles_needed + package_size - 1) / package_size

theorem tom_lasagna_noodles : noodle_packages_needed 10 2 4 2 = 8 := by
  sorry

end tom_lasagna_noodles_l1428_142879


namespace difference_between_shares_l1428_142858

/-- Represents the distribution of money among three people -/
structure MoneyDistribution where
  ratio : Fin 3 → ℕ
  vasimInitialShare : ℕ
  farukTaxRate : ℚ
  vasimTaxRate : ℚ
  ranjithTaxRate : ℚ

/-- Calculates the final share after tax for a given initial share and tax rate -/
def finalShareAfterTax (initialShare : ℕ) (taxRate : ℚ) : ℚ :=
  (1 - taxRate) * initialShare

/-- Theorem stating the difference between Ranjith's and Faruk's final shares -/
theorem difference_between_shares (d : MoneyDistribution) 
  (h1 : d.ratio 0 = 3) 
  (h2 : d.ratio 1 = 5) 
  (h3 : d.ratio 2 = 8) 
  (h4 : d.vasimInitialShare = 1500) 
  (h5 : d.farukTaxRate = 1/10) 
  (h6 : d.vasimTaxRate = 3/20) 
  (h7 : d.ranjithTaxRate = 3/25) : 
  finalShareAfterTax (d.ratio 2 * d.vasimInitialShare / d.ratio 1) d.ranjithTaxRate -
  finalShareAfterTax (d.ratio 0 * d.vasimInitialShare / d.ratio 1) d.farukTaxRate = 1302 := by
  sorry

end difference_between_shares_l1428_142858


namespace percentage_of_blue_flowers_l1428_142887

/-- Given a set of flowers with specific colors, calculate the percentage of blue flowers. -/
theorem percentage_of_blue_flowers (total : ℕ) (red : ℕ) (white : ℕ) (h1 : total = 10) (h2 : red = 4) (h3 : white = 2) :
  (total - red - white) / total * 100 = 40 := by
sorry

end percentage_of_blue_flowers_l1428_142887


namespace talia_father_current_age_talia_future_age_talia_mom_current_age_talia_father_future_age_l1428_142807

-- Define the current year as a reference point
def current_year : ℕ := 0

-- Define Talia's age
def talia_age : ℕ → ℕ
  | year => 13 + year

-- Define Talia's mom's age
def talia_mom_age : ℕ → ℕ
  | year => 3 * talia_age current_year + year

-- Define Talia's father's age
def talia_father_age : ℕ → ℕ
  | year => talia_mom_age current_year + (year - 3)

-- State the theorem
theorem talia_father_current_age :
  talia_father_age current_year = 36 :=
by
  sorry

-- Conditions as separate theorems
theorem talia_future_age :
  talia_age 7 = 20 :=
by
  sorry

theorem talia_mom_current_age :
  talia_mom_age current_year = 3 * talia_age current_year :=
by
  sorry

theorem talia_father_future_age :
  talia_father_age 3 = talia_mom_age current_year :=
by
  sorry

end talia_father_current_age_talia_future_age_talia_mom_current_age_talia_father_future_age_l1428_142807


namespace bowl_game_points_ratio_l1428_142856

theorem bowl_game_points_ratio :
  ∀ (noa_points phillip_points : ℕ) (multiple : ℚ),
    noa_points = 30 →
    phillip_points = noa_points * multiple →
    noa_points + phillip_points = 90 →
    phillip_points / noa_points = 2 := by
  sorry

end bowl_game_points_ratio_l1428_142856


namespace classroom_shirts_problem_l1428_142893

theorem classroom_shirts_problem (total_students : ℕ) 
  (striped_ratio : ℚ) (shorts_difference : ℕ) : 
  total_students = 81 →
  striped_ratio = 2 / 3 →
  shorts_difference = 19 →
  let striped := (total_students : ℚ) * striped_ratio
  let checkered := total_students - striped.floor
  let shorts := checkered + shorts_difference
  striped.floor - shorts = 8 := by
sorry

end classroom_shirts_problem_l1428_142893


namespace solution_set_is_singleton_l1428_142896

def solution_set : Set (ℝ × ℝ) := {(x, y) | 2*x + y = 0 ∧ x - y + 3 = 0}

theorem solution_set_is_singleton : solution_set = {(-1, 2)} := by sorry

end solution_set_is_singleton_l1428_142896


namespace trajectory_equation_l1428_142841

/-- Given points A(-1,1) and B(1,-1) symmetrical about the origin,
    prove that a point P(x,y) with x ≠ ±1 satisfies x^2 + 3y^2 = 4
    if the product of slopes of AP and BP is -1/3 -/
theorem trajectory_equation (x y : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  ((y - 1) / (x + 1)) * ((y + 1) / (x - 1)) = -1/3 →
  x^2 + 3*y^2 = 4 := by
sorry

end trajectory_equation_l1428_142841


namespace function_property_l1428_142866

/-- Given a function f and a real number a, if f(a) + f(1) = 0, then a = -3 -/
theorem function_property (f : ℝ → ℝ) (a : ℝ) (h : f a + f 1 = 0) : a = -3 := by
  sorry

end function_property_l1428_142866


namespace sum_of_digits_of_calculation_l1428_142823

def calculation : ℕ := 100 * 1 + 50 * 2 + 25 * 4 + 2010

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10 + sum_of_digits (n / 10))

theorem sum_of_digits_of_calculation :
  sum_of_digits calculation = 303 := by sorry

end sum_of_digits_of_calculation_l1428_142823


namespace polynomial_division_theorem_l1428_142854

theorem polynomial_division_theorem (x : ℝ) :
  (x - 3) * (x^3 - 19*x^2 - 45*x - 148) + (-435) = x^4 - 22*x^3 + 12*x^2 - 13*x + 9 := by
  sorry

end polynomial_division_theorem_l1428_142854


namespace function_value_implies_a_value_l1428_142800

noncomputable def f (a x : ℝ) : ℝ := x - Real.log (x + 2) + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem function_value_implies_a_value :
  ∃ (a x₀ : ℝ), f a x₀ = 3 → a = -Real.log 2 - 1 := by
  sorry

end function_value_implies_a_value_l1428_142800


namespace circle_area_l1428_142886

theorem circle_area (x y : ℝ) :
  (3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
  (∃ (center_x center_y radius : ℝ),
    ∀ (x' y' : ℝ), (x' - center_x)^2 + (y' - center_y)^2 = radius^2 ↔
    3 * x'^2 + 3 * y'^2 - 9 * x' + 12 * y' + 27 = 0) →
  (π * (1/2)^2 : ℝ) = π/4 :=
by sorry

end circle_area_l1428_142886


namespace jellybeans_left_l1428_142812

/-- Calculates the number of jellybeans left in a jar after a class party --/
theorem jellybeans_left (total_jellybeans : ℕ) 
  (kindergarteners first_graders second_graders : ℕ)
  (absent_kindergarteners absent_second_graders : ℕ)
  (present_kindergartener_rate first_grader_rate : ℕ)
  (absent_kindergartener_rate absent_second_grader_rate : ℕ)
  (h1 : total_jellybeans = 500)
  (h2 : kindergarteners = 10)
  (h3 : first_graders = 10)
  (h4 : second_graders = 10)
  (h5 : absent_kindergarteners = 2)
  (h6 : absent_second_graders = 3)
  (h7 : present_kindergartener_rate = 3)
  (h8 : first_grader_rate = 5)
  (h9 : absent_kindergartener_rate = 5)
  (h10 : absent_second_grader_rate = 10) :
  total_jellybeans - 
  ((kindergarteners - absent_kindergarteners) * present_kindergartener_rate +
   first_graders * first_grader_rate +
   (second_graders - absent_second_graders) * (first_graders * first_grader_rate / 2)) = 176 := by
sorry


end jellybeans_left_l1428_142812


namespace lcm_220_504_l1428_142811

theorem lcm_220_504 : Nat.lcm 220 504 = 27720 := by
  sorry

end lcm_220_504_l1428_142811


namespace max_value_of_function_l1428_142889

theorem max_value_of_function (x : ℝ) (h : x < 5/4) :
  (∀ z : ℝ, z < 5/4 → 4*z - 2 + 1/(4*z - 5) ≤ 4*x - 2 + 1/(4*x - 5)) →
  4*x - 2 + 1/(4*x - 5) = 1 :=
by sorry

end max_value_of_function_l1428_142889


namespace B_is_closed_l1428_142851

def ClosedSet (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

def B : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem B_is_closed : ClosedSet B := by
  sorry

end B_is_closed_l1428_142851


namespace max_value_is_70_l1428_142859

/-- Represents the types of rocks available --/
inductive RockType
  | Seven
  | Three
  | Two

/-- The weight of a rock in pounds --/
def weight : RockType → ℕ
  | RockType.Seven => 7
  | RockType.Three => 3
  | RockType.Two => 2

/-- The value of a rock in dollars --/
def value : RockType → ℕ
  | RockType.Seven => 20
  | RockType.Three => 10
  | RockType.Two => 4

/-- The maximum weight Carl can carry in pounds --/
def maxWeight : ℕ := 21

/-- The minimum number of each type of rock available --/
def minAvailable : ℕ := 15

/-- A function to calculate the total value of a combination of rocks --/
def totalValue (combination : RockType → ℕ) : ℕ :=
  (combination RockType.Seven * value RockType.Seven) +
  (combination RockType.Three * value RockType.Three) +
  (combination RockType.Two * value RockType.Two)

/-- A function to calculate the total weight of a combination of rocks --/
def totalWeight (combination : RockType → ℕ) : ℕ :=
  (combination RockType.Seven * weight RockType.Seven) +
  (combination RockType.Three * weight RockType.Three) +
  (combination RockType.Two * weight RockType.Two)

/-- The main theorem stating that the maximum value of rocks Carl can carry is $70 --/
theorem max_value_is_70 :
  ∃ (combination : RockType → ℕ),
    (∀ rock, combination rock ≤ minAvailable) ∧
    totalWeight combination ≤ maxWeight ∧
    totalValue combination = 70 ∧
    (∀ other : RockType → ℕ,
      (∀ rock, other rock ≤ minAvailable) →
      totalWeight other ≤ maxWeight →
      totalValue other ≤ 70) :=
by
  sorry

end max_value_is_70_l1428_142859


namespace goldbach_140_max_diff_l1428_142829

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_140_max_diff :
  ∀ p q : ℕ,
    is_prime p →
    is_prime q →
    p + q = 140 →
    p < q →
    p < 50 →
    q - p ≤ 134 :=
by sorry

end goldbach_140_max_diff_l1428_142829


namespace sqrt_2a_plus_b_is_6_l1428_142837

/-- Given that the square root of (a + 9) is -5 and the cube root of (2b - a) is -2,
    prove that the arithmetic square root of (2a + b) is 6 -/
theorem sqrt_2a_plus_b_is_6 (a b : ℝ) 
  (h1 : Real.sqrt (a + 9) = -5)
  (h2 : (2 * b - a) ^ (1/3 : ℝ) = -2) :
  Real.sqrt (2 * a + b) = 6 := by
  sorry

end sqrt_2a_plus_b_is_6_l1428_142837


namespace min_value_of_expression_l1428_142818

theorem min_value_of_expression (a b c : ℝ) 
  (h : ∀ (x y : ℝ), 3*x + 4*y - 5 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ 3*x + 4*y + 5) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (v : ℝ), (v = a + b - c → v ≥ m) := by
  sorry

end min_value_of_expression_l1428_142818


namespace calculation_proof_l1428_142833

theorem calculation_proof :
  ((-1/4 + 5/6 - 2/9) * (-36) = -13) ∧
  (-1^4 - 1/6 - (3 + (-3)^2) / (-1 - 1/2) = 6 + 5/6) := by
  sorry

end calculation_proof_l1428_142833


namespace total_pumpkin_pies_l1428_142839

theorem total_pumpkin_pies (pinky helen emily : ℕ) 
  (h1 : pinky = 147) 
  (h2 : helen = 56) 
  (h3 : emily = 89) : 
  pinky + helen + emily = 292 := by
  sorry

end total_pumpkin_pies_l1428_142839


namespace square_side_length_sum_l1428_142805

theorem square_side_length_sum : ∃ (a b : ℕ), a^2 + b^2 = 100 ∧ a + b = 14 := by
  sorry

end square_side_length_sum_l1428_142805


namespace oil_purchase_increase_l1428_142848

/-- Calculates the additional amount of oil that can be purchased after a price reduction -/
def additional_oil_purchase (price_reduction : ℚ) (budget : ℚ) (reduced_price : ℚ) : ℚ :=
  let original_price := reduced_price / (1 - price_reduction)
  let original_amount := budget / original_price
  let new_amount := budget / reduced_price
  new_amount - original_amount

/-- Proves that given a 30% price reduction, a budget of 700, and a reduced price of 70,
    the additional amount of oil that can be purchased is 3 -/
theorem oil_purchase_increase :
  additional_oil_purchase (30 / 100) 700 70 = 3 := by
  sorry

end oil_purchase_increase_l1428_142848


namespace point_on_line_expression_l1428_142890

/-- For any point (a,b) on the line y = 4x + 3, the expression 4a - b - 2 equals -5 -/
theorem point_on_line_expression (a b : ℝ) : b = 4 * a + 3 → 4 * a - b - 2 = -5 := by
  sorry

end point_on_line_expression_l1428_142890


namespace unique_n_divisibility_l1428_142816

theorem unique_n_divisibility : ∃! n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 := by
  sorry

end unique_n_divisibility_l1428_142816


namespace opposite_reciprocal_abs_l1428_142876

theorem opposite_reciprocal_abs (x : ℚ) (h : x = -1.5) : 
  (-x = 1.5) ∧ (1 / x = -2/3) ∧ (abs x = 1.5) := by
  sorry

#check opposite_reciprocal_abs

end opposite_reciprocal_abs_l1428_142876


namespace binary_1100_eq_12_l1428_142804

/-- Converts a binary number represented as a list of bits (0 or 1) to its decimal equivalent. -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.reverse.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of 1100 -/
def binary_1100 : List Nat := [1, 1, 0, 0]

/-- Theorem stating that the binary number 1100 is equal to the decimal number 12 -/
theorem binary_1100_eq_12 : binary_to_decimal binary_1100 = 12 := by
  sorry

end binary_1100_eq_12_l1428_142804
