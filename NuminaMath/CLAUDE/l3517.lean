import Mathlib

namespace NUMINAMATH_CALUDE_bird_count_l3517_351732

theorem bird_count (N t : ℚ) : 
  (3 / 5 * N + 1 / 4 * N + 10 * t = N) → 
  (3 / 5 * N = 40 * t) :=
by sorry

end NUMINAMATH_CALUDE_bird_count_l3517_351732


namespace NUMINAMATH_CALUDE_geometric_series_equality_l3517_351771

def C (n : ℕ) : ℚ := (1024 / 3) * (1 - 1 / (4 ^ n))

def D (n : ℕ) : ℚ := (2048 / 3) * (1 - 1 / ((-2) ^ n))

theorem geometric_series_equality (n : ℕ) (h : n ≥ 1) : 
  (C n = D n) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l3517_351771


namespace NUMINAMATH_CALUDE_percentage_of_indian_children_l3517_351791

theorem percentage_of_indian_children 
  (total_men : ℕ) 
  (total_women : ℕ) 
  (total_children : ℕ) 
  (percent_indian_men : ℚ) 
  (percent_indian_women : ℚ) 
  (percent_non_indian : ℚ) :
  total_men = 500 →
  total_women = 300 →
  total_children = 500 →
  percent_indian_men = 10 / 100 →
  percent_indian_women = 60 / 100 →
  percent_non_indian = 55.38461538461539 / 100 →
  (↑(total_men * 10 + total_women * 60 + total_children * 70) / ↑(total_men + total_women + total_children) : ℚ) = 1 - percent_non_indian :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_indian_children_l3517_351791


namespace NUMINAMATH_CALUDE_circles_are_intersecting_l3517_351785

/-- Two circles are intersecting if the distance between their centers is greater than the absolute 
    difference of their radii and less than the sum of their radii. -/
def are_intersecting (r₁ r₂ d : ℝ) : Prop :=
  d > |r₁ - r₂| ∧ d < r₁ + r₂

/-- Given two circles with radii 5 and 8, whose centers are 8 units apart, 
    prove that they are intersecting. -/
theorem circles_are_intersecting : are_intersecting 5 8 8 := by
  sorry

end NUMINAMATH_CALUDE_circles_are_intersecting_l3517_351785


namespace NUMINAMATH_CALUDE_max_prime_difference_l3517_351781

def is_prime (n : ℕ) : Prop := sorry

def are_distinct {α : Type*} (l : List α) : Prop := sorry

theorem max_prime_difference (a b c : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : is_prime a ∧ is_prime b ∧ is_prime c ∧ 
        is_prime (a+b-c) ∧ is_prime (a+c-b) ∧ is_prime (b+c-a) ∧ is_prime (a+b+c))
  (h3 : are_distinct [a, b, c, a+b-c, a+c-b, b+c-a, a+b+c])
  (h4 : (a + b = 800) ∨ (a + c = 800) ∨ (b + c = 800)) :
  ∃ d : ℕ, d ≤ 1594 ∧ 
  d = max (a+b+c) (max a (max b (max c (max (a+b-c) (max (a+c-b) (b+c-a)))))) -
      min (a+b+c) (min a (min b (min c (min (a+b-c) (min (a+c-b) (b+c-a)))))) ∧
  ∀ d' : ℕ, d' ≤ d := by sorry

end NUMINAMATH_CALUDE_max_prime_difference_l3517_351781


namespace NUMINAMATH_CALUDE_largest_n_implies_x_l3517_351774

/-- Binary operation @ defined as n - (n * x) -/
def binary_op (n : ℤ) (x : ℝ) : ℝ := n - (n * x)

/-- Theorem stating that if 5 is the largest positive integer n such that n @ x < 21, then x = -3 -/
theorem largest_n_implies_x (x : ℝ) :
  (∀ n : ℤ, n > 0 → binary_op n x < 21 → n ≤ 5) ∧
  (binary_op 5 x < 21) →
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_implies_x_l3517_351774


namespace NUMINAMATH_CALUDE_lamp_arrangement_probability_l3517_351775

/-- The total number of lamps -/
def total_lamps : ℕ := 8

/-- The number of red lamps -/
def red_lamps : ℕ := 4

/-- The number of blue lamps -/
def blue_lamps : ℕ := 4

/-- The number of lamps to be turned on -/
def lamps_on : ℕ := 4

/-- The probability of the specific arrangement -/
def target_probability : ℚ := 4 / 49

/-- Theorem stating the probability of the specific arrangement -/
theorem lamp_arrangement_probability :
  let total_arrangements := Nat.choose total_lamps red_lamps * Nat.choose total_lamps lamps_on
  let favorable_arrangements := Nat.choose (total_lamps - 2) (red_lamps - 1) * Nat.choose (total_lamps - 2) (lamps_on - 1)
  (favorable_arrangements : ℚ) / total_arrangements = target_probability := by
  sorry


end NUMINAMATH_CALUDE_lamp_arrangement_probability_l3517_351775


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3517_351731

theorem decimal_to_fraction : 
  (3.76 : ℚ) = 94 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3517_351731


namespace NUMINAMATH_CALUDE_P_intersect_M_l3517_351740

def P : Set ℤ := {x : ℤ | 0 ≤ x ∧ x < 3}
def M : Set ℤ := {x : ℤ | x^2 ≤ 9}

theorem P_intersect_M : P ∩ M = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_P_intersect_M_l3517_351740


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l3517_351798

theorem gcd_from_lcm_and_ratio (A B : ℕ+) : 
  A.lcm B = 240 → A.val * 6 = B.val * 5 → A.gcd B = 8 := by sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l3517_351798


namespace NUMINAMATH_CALUDE_remainder_problem_l3517_351748

theorem remainder_problem (n : ℕ) (h1 : (1661 - 10) % n = 0) (h2 : (2045 - 13) % n = 0) (h3 : n = 127) : 
  13 = 2045 % n :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3517_351748


namespace NUMINAMATH_CALUDE_base_8_first_digit_of_395_l3517_351776

def base_8_first_digit (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let p := Nat.log 8 n
    (n / 8^p) % 8

theorem base_8_first_digit_of_395 :
  base_8_first_digit 395 = 6 := by
sorry

end NUMINAMATH_CALUDE_base_8_first_digit_of_395_l3517_351776


namespace NUMINAMATH_CALUDE_pyarelals_loss_is_1800_l3517_351709

/-- Represents the loss incurred by Pyarelal in a business partnership with Ashok -/
def pyarelals_loss (pyarelals_capital : ℚ) (total_loss : ℚ) : ℚ :=
  (pyarelals_capital / (pyarelals_capital + pyarelals_capital / 9)) * total_loss

/-- Theorem stating that Pyarelal's loss is 1800 given the conditions of the problem -/
theorem pyarelals_loss_is_1800 (pyarelals_capital : ℚ) (h : pyarelals_capital > 0) :
  pyarelals_loss pyarelals_capital 2000 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_pyarelals_loss_is_1800_l3517_351709


namespace NUMINAMATH_CALUDE_student_ratio_l3517_351757

theorem student_ratio (total : ℕ) (on_bleachers : ℕ) 
  (h1 : total = 26) (h2 : on_bleachers = 4) : 
  (total - on_bleachers : ℚ) / total = 11 / 13 := by
  sorry

end NUMINAMATH_CALUDE_student_ratio_l3517_351757


namespace NUMINAMATH_CALUDE_octal_127_equals_binary_1010111_l3517_351792

def octal_to_decimal (octal : ℕ) : ℕ := 
  (octal % 10) + 8 * ((octal / 10) % 10) + 64 * (octal / 100)

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem octal_127_equals_binary_1010111 : 
  decimal_to_binary (octal_to_decimal 127) = [1, 0, 1, 0, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_octal_127_equals_binary_1010111_l3517_351792


namespace NUMINAMATH_CALUDE_x_x_minus_one_sufficient_not_necessary_l3517_351715

theorem x_x_minus_one_sufficient_not_necessary (x : ℝ) :
  (∀ x, x * (x - 1) < 0 → x < 1) ∧
  (∃ x, x < 1 ∧ x * (x - 1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_x_minus_one_sufficient_not_necessary_l3517_351715


namespace NUMINAMATH_CALUDE_gretchen_earnings_l3517_351768

/-- Gretchen's earnings from drawing caricatures over a weekend -/
def weekend_earnings (price_per_drawing : ℕ) (saturday_sales : ℕ) (sunday_sales : ℕ) : ℕ :=
  price_per_drawing * (saturday_sales + sunday_sales)

/-- Theorem stating Gretchen's earnings for the given weekend -/
theorem gretchen_earnings :
  weekend_earnings 20 24 16 = 800 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_earnings_l3517_351768


namespace NUMINAMATH_CALUDE_inequality_property_l3517_351723

theorem inequality_property (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l3517_351723


namespace NUMINAMATH_CALUDE_balls_after_500_steps_l3517_351755

/-- Represents the state of boxes after a certain number of steps -/
def BoxState := Nat

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : Nat) : List Nat :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List Nat) : Nat :=
  sorry

/-- Simulates the ball placement process for a given number of steps -/
def simulateSteps (steps : Nat) : BoxState :=
  sorry

theorem balls_after_500_steps :
  simulateSteps 500 = sumDigits (toBase4 500) :=
sorry

end NUMINAMATH_CALUDE_balls_after_500_steps_l3517_351755


namespace NUMINAMATH_CALUDE_multiplication_equation_solution_l3517_351739

theorem multiplication_equation_solution : ∃ x : ℕ, 80641 * x = 806006795 ∧ x = 9995 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_solution_l3517_351739


namespace NUMINAMATH_CALUDE_valid_pairs_l3517_351728

def is_valid_pair (a b : ℕ+) : Prop :=
  (∃ k : ℤ, (a.val ^ 3 * b.val - 1) = k * (a.val + 1)) ∧
  (∃ m : ℤ, (b.val ^ 3 * a.val + 1) = m * (b.val - 1))

theorem valid_pairs :
  ∀ a b : ℕ+, is_valid_pair a b →
    ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l3517_351728


namespace NUMINAMATH_CALUDE_equation_holds_iff_nonpositive_l3517_351765

theorem equation_holds_iff_nonpositive (a b : ℝ) : a = |b| → (a + b = 0 ↔ b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_iff_nonpositive_l3517_351765


namespace NUMINAMATH_CALUDE_jackson_running_distance_l3517_351721

/-- Calculate the final running distance after doubling the initial distance for a given number of weeks -/
def finalDistance (initialDistance : ℕ) (weeks : ℕ) : ℕ :=
  initialDistance * (2 ^ weeks)

/-- Theorem stating that starting with 3 miles and doubling for 4 weeks results in 24 miles -/
theorem jackson_running_distance : finalDistance 3 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_jackson_running_distance_l3517_351721


namespace NUMINAMATH_CALUDE_rectangle_width_l3517_351770

/-- A rectangle with a perimeter of 20 cm and length 2 cm more than its width has a width of 4 cm. -/
theorem rectangle_width (w : ℝ) (h1 : 2 * (w + 2) + 2 * w = 20) : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3517_351770


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3517_351760

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem quadratic_function_properties :
  (∃ x, f x = 1 ∧ ∀ y, f y ≥ f x) ∧
  f 0 = 3 ∧ f 2 = 3 ∧
  (∀ a : ℝ, (0 < a ∧ a < 1/2) ↔ 
    ¬(∀ x y : ℝ, 2*a ≤ x ∧ x < y ∧ y ≤ a+1 → f x < f y ∨ f x > f y)) ∧
  (∀ m : ℝ, m < 1 ↔ ∀ x : ℝ, -3 ≤ x ∧ x ≤ 0 → f x > 2*x + 2*m + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3517_351760


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3517_351713

theorem two_digit_number_property : 
  ∀ n : ℕ, 
    10 ≤ n ∧ n < 100 →
    (n / (n.mod 10 + n / 10) = (n.mod 10 + n / 10) / 3) →
    n = 27 ∨ n = 48 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3517_351713


namespace NUMINAMATH_CALUDE_emily_sixth_score_l3517_351712

def emily_scores : List ℝ := [94, 88, 92, 85, 97]
def target_mean : ℝ := 90
def num_quizzes : ℕ := 6

theorem emily_sixth_score (sixth_score : ℝ) : 
  sixth_score = 84 →
  (emily_scores.sum + sixth_score) / num_quizzes = target_mean := by
  sorry

end NUMINAMATH_CALUDE_emily_sixth_score_l3517_351712


namespace NUMINAMATH_CALUDE_smallest_non_nine_divisible_by_999_l3517_351706

/-- Checks if a natural number contains the digit 9 --/
def containsNine (n : ℕ) : Prop :=
  ∃ (k : ℕ), n / (10^k) % 10 = 9

/-- Checks if a natural number is divisible by 999 --/
def divisibleBy999 (n : ℕ) : Prop :=
  n % 999 = 0

theorem smallest_non_nine_divisible_by_999 :
  ∀ n : ℕ, n > 0 → divisibleBy999 n → ¬containsNine n → n ≥ 112 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_nine_divisible_by_999_l3517_351706


namespace NUMINAMATH_CALUDE_smallest_money_for_pizza_l3517_351788

theorem smallest_money_for_pizza (x : ℕ) : x ≥ 6 ↔ ∃ (a b : ℕ), x - 1 = 5 * a + 7 * b := by
  sorry

end NUMINAMATH_CALUDE_smallest_money_for_pizza_l3517_351788


namespace NUMINAMATH_CALUDE_total_stops_theorem_l3517_351714

def yoojeong_stops : ℕ := 3
def namjoon_stops : ℕ := 2

theorem total_stops_theorem : yoojeong_stops + namjoon_stops = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_stops_theorem_l3517_351714


namespace NUMINAMATH_CALUDE_armands_guessing_game_l3517_351736

theorem armands_guessing_game : ∃ x : ℕ, x = 33 ∧ 3 * x = 2 * 51 - 3 := by
  sorry

end NUMINAMATH_CALUDE_armands_guessing_game_l3517_351736


namespace NUMINAMATH_CALUDE_pin_permutations_l3517_351705

theorem pin_permutations : 
  let n : ℕ := 4
  ∀ (digits : Finset ℕ), Finset.card digits = n → Finset.card (Finset.powersetCard n digits) = n.factorial :=
by
  sorry

end NUMINAMATH_CALUDE_pin_permutations_l3517_351705


namespace NUMINAMATH_CALUDE_multiply_to_target_l3517_351779

theorem multiply_to_target (x : ℕ) : x * 586645 = 5865863355 → x = 9999 := by
  sorry

end NUMINAMATH_CALUDE_multiply_to_target_l3517_351779


namespace NUMINAMATH_CALUDE_partition_S_l3517_351750

def S : Set ℚ := {-5/6, 0, -7/2, 6/5, 6}

theorem partition_S :
  (∃ (A B : Set ℚ), A ∪ B = S ∧ A ∩ B = ∅ ∧
    A = {x ∈ S | x < 0} ∧
    B = {x ∈ S | x ≥ 0} ∧
    A = {-5/6, -7/2} ∧
    B = {0, 6/5, 6}) :=
by sorry

end NUMINAMATH_CALUDE_partition_S_l3517_351750


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l3517_351762

/-- The area of the region in a square of side length 5 bounded by lines from (0,0) to (2.5,5) and from (5,5) to (0,2.5) is half the area of the square. -/
theorem shaded_area_ratio (square_side : ℝ) (h : square_side = 5) : 
  let shaded_area := (1/2 * 2.5 * 2.5) + (2.5 * 2.5) + (1/2 * 2.5 * 2.5)
  shaded_area / (square_side ^ 2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l3517_351762


namespace NUMINAMATH_CALUDE_third_day_temp_is_two_l3517_351783

/-- The temperature on the third day of a sequence of 8 days, given other temperatures and the mean -/
def third_day_temperature (t1 t2 t4 t5 t6 t7 t8 mean : ℚ) : ℚ :=
  let sum := t1 + t2 + t4 + t5 + t6 + t7 + t8
  8 * mean - sum

theorem third_day_temp_is_two :
  let t1 := -6
  let t2 := -3
  let t4 := -6
  let t5 := 2
  let t6 := 4
  let t7 := 3
  let t8 := 0
  let mean := -0.5
  third_day_temperature t1 t2 t4 t5 t6 t7 t8 mean = 2 := by
  sorry

#eval third_day_temperature (-6) (-3) (-6) 2 4 3 0 (-0.5)

end NUMINAMATH_CALUDE_third_day_temp_is_two_l3517_351783


namespace NUMINAMATH_CALUDE_candidate_X_votes_and_result_l3517_351754

-- Define the number of votes for each candidate
def votes_Z : ℕ := 25000
def votes_Y : ℕ := (3 * votes_Z) / 5
def votes_X : ℕ := (3 * votes_Y) / 2

-- Define the winning threshold
def winning_threshold : ℕ := 30000

-- Theorem to prove
theorem candidate_X_votes_and_result : 
  votes_X = 22500 ∧ votes_X < winning_threshold :=
by sorry

end NUMINAMATH_CALUDE_candidate_X_votes_and_result_l3517_351754


namespace NUMINAMATH_CALUDE_variance_best_stability_measure_l3517_351708

/-- A measure of stability for a set of test scores -/
class StabilityMeasure where
  measure : List ℝ → ℝ

/-- Average as a stability measure -/
def average : StabilityMeasure := sorry

/-- Median as a stability measure -/
def median : StabilityMeasure := sorry

/-- Variance as a stability measure -/
def variance : StabilityMeasure := sorry

/-- Mode as a stability measure -/
def mode : StabilityMeasure := sorry

/-- A function that determines if a stability measure is the best for test scores -/
def isBestStabilityMeasure (m : StabilityMeasure) : Prop := sorry

theorem variance_best_stability_measure : isBestStabilityMeasure variance := by
  sorry

end NUMINAMATH_CALUDE_variance_best_stability_measure_l3517_351708


namespace NUMINAMATH_CALUDE_total_crayons_l3517_351738

def number_of_boxes : ℕ := 7
def crayons_per_box : ℕ := 5

theorem total_crayons : number_of_boxes * crayons_per_box = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l3517_351738


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l3517_351769

/-- The minimum distance between a point on the line y = (12/5)x - 5 and a point on the parabola y = x^2 is 89/65 -/
theorem min_distance_line_parabola :
  let line := fun (x : ℝ) => (12/5) * x - 5
  let parabola := fun (x : ℝ) => x^2
  let distance := fun (x₁ x₂ : ℝ) => 
    Real.sqrt ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2)
  (∃ (x₁ x₂ : ℝ), ∀ (y₁ y₂ : ℝ), distance x₁ x₂ ≤ distance y₁ y₂) ∧
  (∃ (x₁ x₂ : ℝ), distance x₁ x₂ = 89/65) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l3517_351769


namespace NUMINAMATH_CALUDE_sum_of_permutations_unique_l3517_351777

/-- Represents a positive integer with at least two digits as a list of its digits. -/
def PositiveInteger := {l : List Nat // l.length ≥ 2 ∧ l.head! ≠ 0}

/-- Calculates the sum of all permutations of a number's digits, excluding the original number. -/
def sumOfPermutations (n : PositiveInteger) : Nat :=
  sorry

/-- Theorem stating that the sum of permutations is unique for each number. -/
theorem sum_of_permutations_unique (x y : PositiveInteger) :
  x ≠ y → sumOfPermutations x ≠ sumOfPermutations y := by
  sorry

end NUMINAMATH_CALUDE_sum_of_permutations_unique_l3517_351777


namespace NUMINAMATH_CALUDE_percent_more_and_less_equal_l3517_351741

theorem percent_more_and_less_equal (x : ℝ) : x = 138.67 →
  (80 + 0.3 * 80 : ℝ) = (x - 0.25 * x) := by sorry

end NUMINAMATH_CALUDE_percent_more_and_less_equal_l3517_351741


namespace NUMINAMATH_CALUDE_center_is_ten_l3517_351780

/-- Represents a 4x4 array of integers -/
def Array4x4 := Fin 4 → Fin 4 → ℕ

/-- Checks if two positions in the array share an edge -/
def share_edge (p q : Fin 4 × Fin 4) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Defines a valid array according to the problem conditions -/
def valid_array (a : Array4x4) : Prop :=
  (∀ n : Fin 16, ∃ i j : Fin 4, a i j = n.val + 1) ∧
  (∀ n : Fin 15, ∃ i j k l : Fin 4, 
    a i j = n.val + 1 ∧ 
    a k l = n.val + 2 ∧ 
    share_edge (i, j) (k, l)) ∧
  (a 0 0 + a 0 3 + a 3 0 + a 3 3 = 34)

/-- The main theorem to prove -/
theorem center_is_ten (a : Array4x4) (h : valid_array a) : 
  a 1 1 = 10 ∨ a 1 2 = 10 ∨ a 2 1 = 10 ∨ a 2 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_center_is_ten_l3517_351780


namespace NUMINAMATH_CALUDE_inequalities_problem_l3517_351786

theorem inequalities_problem (a b c d : ℝ) 
  (ha : a > 0) 
  (hb1 : 0 > b) 
  (hb2 : b > -a) 
  (hc : c < d) 
  (hd : d < 0) : 
  (a / b + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) := by
sorry

end NUMINAMATH_CALUDE_inequalities_problem_l3517_351786


namespace NUMINAMATH_CALUDE_second_integer_problem_l3517_351703

theorem second_integer_problem (x y : ℕ+) (hx : x = 3) (h : x * y + x = 33) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_integer_problem_l3517_351703


namespace NUMINAMATH_CALUDE_cone_volume_ratio_l3517_351751

/-- Two cones sharing a common base on a sphere -/
structure ConePair where
  R : ℝ  -- Radius of the sphere
  r : ℝ  -- Radius of the base of the cones
  h₁ : ℝ  -- Height of the first cone
  h₂ : ℝ  -- Height of the second cone

/-- The conditions of the problem -/
def ConePairConditions (cp : ConePair) : Prop :=
  cp.r^2 = 3 * cp.R^2 / 4 ∧  -- Area of base is 3/16 of sphere area
  cp.h₁ + cp.h₂ = 2 * cp.R ∧  -- Sum of heights equals diameter
  cp.r^2 + (cp.h₁ / 2)^2 = cp.R^2  -- Pythagorean theorem

/-- The theorem to be proved -/
theorem cone_volume_ratio (cp : ConePair) 
  (hc : ConePairConditions cp) : 
  cp.h₁ * cp.r^2 / (cp.h₂ * cp.r^2) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_cone_volume_ratio_l3517_351751


namespace NUMINAMATH_CALUDE_intersection_theorem_l3517_351718

-- Define the sets M and N
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define the intersection of N and the complement of M
def intersection_N_complement_M : Set ℝ := N ∩ (Set.univ \ M)

-- State the theorem
theorem intersection_theorem :
  intersection_N_complement_M = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3517_351718


namespace NUMINAMATH_CALUDE_count_solutions_x_plus_y_plus_z_15_l3517_351726

/-- The number of solutions to x + y + z = 15 where x, y, and z are positive integers -/
def num_solutions : ℕ := 91

/-- Theorem stating that the number of solutions to x + y + z = 15 where x, y, and z are positive integers is 91 -/
theorem count_solutions_x_plus_y_plus_z_15 :
  (Finset.filter (fun t : ℕ × ℕ × ℕ => t.1 + t.2.1 + t.2.2 = 15 ∧ t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0)
    (Finset.product (Finset.range 16) (Finset.product (Finset.range 16) (Finset.range 16)))).card = num_solutions := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_x_plus_y_plus_z_15_l3517_351726


namespace NUMINAMATH_CALUDE_chocolate_bars_left_l3517_351778

theorem chocolate_bars_left (initial_bars : ℕ) (thomas_friends : ℕ) (piper_return : ℕ) (paul_extra : ℕ) : 
  initial_bars = 500 →
  thomas_friends = 7 →
  piper_return = 7 →
  paul_extra = 5 →
  ∃ (thomas_take piper_take paul_take : ℕ),
    thomas_take = (initial_bars / 3 / thomas_friends) * thomas_friends + 2 ∧
    piper_take = initial_bars / 4 - piper_return ∧
    paul_take = piper_take + paul_extra ∧
    initial_bars - (thomas_take + piper_take + paul_take) = 96 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_left_l3517_351778


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l3517_351772

/-- The minimum distance from a point on the circle ρ = 2 to the line ρ(cos(θ) + √3 sin(θ)) = 6 is 1 -/
theorem min_distance_circle_to_line : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let line := {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 = 6}
  ∃ (d : ℝ), d = 1 ∧ ∀ (p : ℝ × ℝ), p ∈ circle → 
    (∀ (q : ℝ × ℝ), q ∈ line → d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l3517_351772


namespace NUMINAMATH_CALUDE_first_tap_fill_time_l3517_351710

/-- Represents the time (in hours) it takes for the first tap to fill the cistern -/
def T : ℝ := 3

/-- Represents the time (in hours) it takes for the second tap to empty the cistern -/
def empty_time : ℝ := 8

/-- Represents the time (in hours) it takes to fill the cistern when both taps are open -/
def both_open_time : ℝ := 4.8

/-- Proves that T is the correct time for the first tap to fill the cistern -/
theorem first_tap_fill_time :
  (1 / T - 1 / empty_time = 1 / both_open_time) ∧ T > 0 := by
  sorry

end NUMINAMATH_CALUDE_first_tap_fill_time_l3517_351710


namespace NUMINAMATH_CALUDE_valid_three_digit_count_correct_l3517_351722

/-- The count of valid three-digit numbers -/
def valid_three_digit_count : ℕ := 819

/-- The total count of three-digit numbers -/
def total_three_digit_count : ℕ := 900

/-- The count of invalid three-digit numbers where the hundreds and units digits
    are the same but the tens digit is different -/
def invalid_three_digit_count : ℕ := 81

/-- Theorem stating that the count of valid three-digit numbers is correct -/
theorem valid_three_digit_count_correct :
  valid_three_digit_count = total_three_digit_count - invalid_three_digit_count :=
by sorry

end NUMINAMATH_CALUDE_valid_three_digit_count_correct_l3517_351722


namespace NUMINAMATH_CALUDE_pencils_added_l3517_351724

theorem pencils_added (initial_pencils final_pencils : ℕ) (h1 : initial_pencils = 41) (h2 : final_pencils = 71) :
  final_pencils - initial_pencils = 30 := by
  sorry

end NUMINAMATH_CALUDE_pencils_added_l3517_351724


namespace NUMINAMATH_CALUDE_poetry_competition_results_l3517_351759

-- Define the contingency table
def a : ℕ := 6
def b : ℕ := 9
def c : ℕ := 4
def d : ℕ := 1
def n : ℕ := 20

-- Define K^2 calculation
def K_squared : ℚ := (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d) : ℚ)

-- Define probabilities for student C
def prob_buzz : ℚ := 3/5
def prob_correct_buzz : ℚ := 4/5

-- Define the score variable X
inductive Score
| neg_one : Score
| zero : Score
| two : Score

-- Define the probability distribution of X
def prob_X : Score → ℚ
| Score.neg_one => prob_buzz * (1 - prob_correct_buzz)
| Score.zero => 1 - prob_buzz
| Score.two => prob_buzz * prob_correct_buzz

-- Define the expected value of X
def E_X : ℚ := -1 * prob_X Score.neg_one + 0 * prob_X Score.zero + 2 * prob_X Score.two

-- Define the condition for p
def p_condition (p : ℚ) : Prop := 
  |3 * p + 2.52 - (4 * p + 1.68)| ≤ 1/10 ∧ 0 < p ∧ p < 1

theorem poetry_competition_results :
  K_squared < 3841/1000 ∧
  prob_X Score.neg_one = 12/100 ∧
  prob_X Score.zero = 2/5 ∧
  prob_X Score.two = 24/50 ∧
  E_X = 21/25 ∧
  ∀ p, p_condition p ↔ 37/50 ≤ p ∧ p ≤ 47/50 :=
sorry

end NUMINAMATH_CALUDE_poetry_competition_results_l3517_351759


namespace NUMINAMATH_CALUDE_car_average_speed_l3517_351701

/-- Proves that the average speed of a car is 36 km/hr given specific uphill and downhill conditions -/
theorem car_average_speed :
  let uphill_speed : ℝ := 30
  let downhill_speed : ℝ := 60
  let uphill_distance : ℝ := 100
  let downhill_distance : ℝ := 50
  let total_distance : ℝ := uphill_distance + downhill_distance
  let uphill_time : ℝ := uphill_distance / uphill_speed
  let downhill_time : ℝ := downhill_distance / downhill_speed
  let total_time : ℝ := uphill_time + downhill_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 36 := by
sorry

end NUMINAMATH_CALUDE_car_average_speed_l3517_351701


namespace NUMINAMATH_CALUDE_sine_graph_transformation_l3517_351746

theorem sine_graph_transformation (x : ℝ) : 
  4 * Real.sin (2 * x + π / 5) = 4 * Real.sin ((2 * x / 2) + π / 5) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_transformation_l3517_351746


namespace NUMINAMATH_CALUDE_scientific_notation_50300_l3517_351795

theorem scientific_notation_50300 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 50300 = a * (10 : ℝ) ^ n ∧ a = 5.03 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_50300_l3517_351795


namespace NUMINAMATH_CALUDE_swiss_cheese_probability_l3517_351745

theorem swiss_cheese_probability :
  let cheddar : ℕ := 22
  let mozzarella : ℕ := 34
  let pepperjack : ℕ := 29
  let swiss : ℕ := 45
  let gouda : ℕ := 20
  let total : ℕ := cheddar + mozzarella + pepperjack + swiss + gouda
  (swiss : ℚ) / (total : ℚ) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_swiss_cheese_probability_l3517_351745


namespace NUMINAMATH_CALUDE_roll_one_probability_l3517_351729

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Fin 6)

/-- The probability of rolling a specific number on a fair six-sided die -/
def roll_probability (d : FairDie) (n : Fin 6) : ℚ := 1 / 6

/-- The independence of die rolls -/
axiom roll_independence (d : FairDie) (n m : Fin 6) : 
  roll_probability d n = roll_probability d n

/-- Theorem: The probability of rolling a 1 on a fair six-sided die is 1/6 -/
theorem roll_one_probability (d : FairDie) : 
  roll_probability d 0 = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_roll_one_probability_l3517_351729


namespace NUMINAMATH_CALUDE_odd_even_sum_reciprocal_l3517_351733

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function g is even if g(-x) = g(x) for all x -/
def IsEven (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

/-- Given f is odd, g is even, and f(x) + g(x) = 1 / (x - 1), prove f(3) = 3/8 -/
theorem odd_even_sum_reciprocal (f g : ℝ → ℝ) 
    (hodd : IsOdd f) (heven : IsEven g) 
    (hsum : ∀ x ≠ 1, f x + g x = 1 / (x - 1)) : 
    f 3 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_reciprocal_l3517_351733


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_neg_one_l3517_351727

theorem at_least_one_not_greater_than_neg_one (a b c d : ℝ) 
  (sum_eq : a + b + c + d = -2)
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 0) :
  min a (min b (min c d)) ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_neg_one_l3517_351727


namespace NUMINAMATH_CALUDE_commute_time_difference_l3517_351734

/-- Proves that the difference in commute time between walking and taking the train is 25 minutes -/
theorem commute_time_difference
  (distance : Real)
  (walking_speed : Real)
  (train_speed : Real)
  (additional_train_time : Real)
  (h1 : distance = 1.5)
  (h2 : walking_speed = 3)
  (h3 : train_speed = 20)
  (h4 : additional_train_time = 0.5 / 60) :
  (distance / walking_speed - (distance / train_speed + additional_train_time)) * 60 = 25 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_difference_l3517_351734


namespace NUMINAMATH_CALUDE_count_non_zero_area_triangles_l3517_351799

/-- The total number of dots in the grid -/
def total_dots : ℕ := 17

/-- The number of collinear dots in each direction (horizontal and vertical) -/
def collinear_dots : ℕ := 9

/-- The number of ways to choose 3 dots from the total dots -/
def total_combinations : ℕ := Nat.choose total_dots 3

/-- The number of ways to choose 3 collinear dots -/
def collinear_combinations : ℕ := Nat.choose collinear_dots 3

/-- The number of lines with collinear dots (horizontal and vertical) -/
def collinear_lines : ℕ := 2

/-- The number of triangles with non-zero area -/
def non_zero_area_triangles : ℕ := total_combinations - collinear_lines * collinear_combinations

theorem count_non_zero_area_triangles : non_zero_area_triangles = 512 := by
  sorry

end NUMINAMATH_CALUDE_count_non_zero_area_triangles_l3517_351799


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3517_351758

-- Problem 1
theorem problem_1 (x y : ℝ) (h1 : x * y = 5) (h2 : x + y = 6) :
  (x - y)^2 = 16 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : (2016 - a) * (2017 - a) = 5) :
  (a - 2016)^2 + (2017 - a)^2 = 11 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3517_351758


namespace NUMINAMATH_CALUDE_chess_club_committee_probability_l3517_351753

def total_members : ℕ := 27
def boys : ℕ := 15
def girls : ℕ := 12
def committee_size : ℕ := 5

theorem chess_club_committee_probability :
  let total_committees := Nat.choose total_members committee_size
  let all_boys_committees := Nat.choose boys committee_size
  let all_girls_committees := Nat.choose girls committee_size
  let favorable_committees := total_committees - (all_boys_committees + all_girls_committees)
  (favorable_committees : ℚ) / total_committees = 76935 / 80730 := by sorry

end NUMINAMATH_CALUDE_chess_club_committee_probability_l3517_351753


namespace NUMINAMATH_CALUDE_watch_correction_l3517_351735

/-- The number of days from April 1 at 12 noon to April 10 at 6 P.M. -/
def days_passed : ℚ := 9 + 6 / 24

/-- The rate at which the watch loses time, in minutes per day -/
def loss_rate : ℚ := 3

/-- The positive correction in minutes to be added to the watch -/
def correction (d : ℚ) (r : ℚ) : ℚ := d * r

theorem watch_correction :
  correction days_passed loss_rate = 27.75 := by
  sorry

end NUMINAMATH_CALUDE_watch_correction_l3517_351735


namespace NUMINAMATH_CALUDE_all_red_raise_hands_eventually_l3517_351744

/-- Represents the color of a stamp -/
inductive StampColor
| Red
| Green

/-- Represents a faculty member -/
structure FacultyMember where
  stamp : StampColor

/-- Represents the state of the game on a given day -/
structure GameState where
  day : ℕ
  faculty : List FacultyMember
  handsRaised : List FacultyMember

/-- Predicate to check if a faculty member raises their hand -/
def raisesHand (member : FacultyMember) (state : GameState) : Prop :=
  member ∈ state.handsRaised

/-- The main theorem to be proved -/
theorem all_red_raise_hands_eventually 
  (n : ℕ) 
  (faculty : List FacultyMember) 
  (h1 : faculty.length = n) 
  (h2 : ∃ m, m ∈ faculty ∧ m.stamp = StampColor.Red) :
  ∃ (finalState : GameState), 
    finalState.day = n ∧ 
    ∀ m, m ∈ faculty → m.stamp = StampColor.Red → raisesHand m finalState :=
  sorry


end NUMINAMATH_CALUDE_all_red_raise_hands_eventually_l3517_351744


namespace NUMINAMATH_CALUDE_revenue_change_with_price_increase_and_quantity_decrease_l3517_351702

/-- Theorem: Effect on revenue when price increases and quantity decreases -/
theorem revenue_change_with_price_increase_and_quantity_decrease 
  (P Q : ℝ) (P_new Q_new R_new : ℝ) :
  P_new = P * (1 + 0.30) →
  Q_new = Q * (1 - 0.20) →
  R_new = P_new * Q_new →
  R_new = P * Q * 1.04 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_with_price_increase_and_quantity_decrease_l3517_351702


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_391_l3517_351742

theorem greatest_prime_factor_of_391 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 391 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 391 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_391_l3517_351742


namespace NUMINAMATH_CALUDE_angle_values_l3517_351764

/-- Given an angle α with terminal side passing through point P(-3, m) and cosα = -3/5,
    prove the values of m, sinα, and tanα. -/
theorem angle_values (α : Real) (m : Real) 
    (h1 : ∃ (x y : Real), x = -3 ∧ y = m ∧ Real.cos α * Real.sqrt (x^2 + y^2) = x)
    (h2 : Real.cos α = -3/5) :
    (m = 4 ∨ m = -4) ∧ 
    ((Real.sin α = 4/5 ∧ Real.tan α = -4/3) ∨ 
     (Real.sin α = -4/5 ∧ Real.tan α = 4/3)) := by
  sorry

end NUMINAMATH_CALUDE_angle_values_l3517_351764


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3517_351790

theorem smallest_x_absolute_value_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3517_351790


namespace NUMINAMATH_CALUDE_factorial_calculation_l3517_351749

theorem factorial_calculation : (4 * Nat.factorial 6 + 36 * Nat.factorial 5) / Nat.factorial 7 = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l3517_351749


namespace NUMINAMATH_CALUDE_mean_equality_problem_l3517_351773

theorem mean_equality_problem (x : ℚ) : 
  (8 + 10 + 22) / 3 = (15 + x) / 2 → x = 35 / 3 :=
by sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l3517_351773


namespace NUMINAMATH_CALUDE_max_nickels_l3517_351787

-- Define the coin types
inductive Coin
| Penny
| Nickel
| Dime
| Quarter

-- Define the wallet as a function from Coin to ℕ (number of each coin type)
def Wallet := Coin → ℕ

-- Define the value of each coin in cents
def coinValue : Coin → ℕ
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10
| Coin.Quarter => 25

-- Function to calculate the total value of coins in the wallet
def totalValue (w : Wallet) : ℕ :=
  (w Coin.Penny) * (coinValue Coin.Penny) +
  (w Coin.Nickel) * (coinValue Coin.Nickel) +
  (w Coin.Dime) * (coinValue Coin.Dime) +
  (w Coin.Quarter) * (coinValue Coin.Quarter)

-- Function to count the total number of coins in the wallet
def coinCount (w : Wallet) : ℕ :=
  (w Coin.Penny) + (w Coin.Nickel) + (w Coin.Dime) + (w Coin.Quarter)

-- Theorem statement
theorem max_nickels (w : Wallet) :
  (totalValue w = 15 * coinCount w) →
  (totalValue w + coinValue Coin.Dime = 16 * (coinCount w + 1)) →
  (w Coin.Nickel = 2) := by
  sorry


end NUMINAMATH_CALUDE_max_nickels_l3517_351787


namespace NUMINAMATH_CALUDE_inequality_proof_l3517_351797

theorem inequality_proof (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) :
  a^2 > a*b ∧ a*b > a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3517_351797


namespace NUMINAMATH_CALUDE_polynomial_value_l3517_351747

def star (x y : ℤ) : ℤ := (x + 1) * (y + 1)

def star_square (x : ℤ) : ℤ := star x x

theorem polynomial_value : 
  let x := 2
  3 * (star_square x) - 2 * x + 1 = 32 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_l3517_351747


namespace NUMINAMATH_CALUDE_unique_three_digit_integer_l3517_351796

theorem unique_three_digit_integer : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  n % 7 = 3 ∧
  n % 8 = 4 ∧
  n % 13 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_integer_l3517_351796


namespace NUMINAMATH_CALUDE_class_configuration_exists_l3517_351784

theorem class_configuration_exists (n : ℕ) (hn : n = 30) :
  ∃ (b g : ℕ),
    b + g = n ∧
    b = g ∧
    (∀ i j : ℕ, i < b → j < b → i ≠ j → ∃ k : ℕ, k < g ∧ (∃ f : ℕ → ℕ → Prop, f i k ≠ f j k)) ∧
    (∀ i j : ℕ, i < g → j < g → i ≠ j → ∃ k : ℕ, k < b ∧ (∃ f : ℕ → ℕ → Prop, f k i ≠ f k j)) :=
by
  sorry

end NUMINAMATH_CALUDE_class_configuration_exists_l3517_351784


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l3517_351794

theorem dogwood_tree_count (current_trees new_trees : ℕ) 
  (h1 : current_trees = 34)
  (h2 : new_trees = 49) :
  current_trees + new_trees = 83 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l3517_351794


namespace NUMINAMATH_CALUDE_huangshan_temperature_difference_l3517_351720

def temperature_difference (lowest highest : ℤ) : ℤ :=
  highest - lowest

theorem huangshan_temperature_difference :
  let lowest : ℤ := -13
  let highest : ℤ := 11
  temperature_difference lowest highest = 24 := by
  sorry

end NUMINAMATH_CALUDE_huangshan_temperature_difference_l3517_351720


namespace NUMINAMATH_CALUDE_remaining_practice_time_l3517_351737

/-- The total practice time in hours for the week -/
def total_practice_hours : ℝ := 7.5

/-- The number of days with known practice time -/
def known_practice_days : ℕ := 2

/-- The practice time in minutes for each of the known practice days -/
def practice_per_known_day : ℕ := 86

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

theorem remaining_practice_time :
  hours_to_minutes total_practice_hours - (known_practice_days * practice_per_known_day) = 278 := by
  sorry

end NUMINAMATH_CALUDE_remaining_practice_time_l3517_351737


namespace NUMINAMATH_CALUDE_spinner_probability_l3517_351763

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 → pB = 1/3 → pD = 1/6 → pA + pB + pC + pD = 1 → pC = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3517_351763


namespace NUMINAMATH_CALUDE_philatelist_stamps_problem_l3517_351704

theorem philatelist_stamps_problem :
  ∃! x : ℕ,
    x % 3 = 1 ∧
    x % 5 = 3 ∧
    x % 7 = 5 ∧
    150 < x ∧
    x ≤ 300 ∧
    x = 208 := by
  sorry

end NUMINAMATH_CALUDE_philatelist_stamps_problem_l3517_351704


namespace NUMINAMATH_CALUDE_parabola_properties_l3517_351782

/-- A parabola is defined by the equation y = -x^2 + 1 --/
def parabola (x : ℝ) : ℝ := -x^2 + 1

theorem parabola_properties :
  (∀ x : ℝ, parabola x ≤ parabola 0) ∧ 
  parabola 0 = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3517_351782


namespace NUMINAMATH_CALUDE_power_sum_equality_l3517_351766

theorem power_sum_equality : 3^(3+4+5) + (3^3 + 3^4 + 3^5) = 531792 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3517_351766


namespace NUMINAMATH_CALUDE_hilt_current_rocks_l3517_351793

/-- The number of rocks Mrs. Hilt needs to complete the border -/
def total_rocks_needed : ℕ := 125

/-- The number of additional rocks Mrs. Hilt needs -/
def additional_rocks_needed : ℕ := 61

/-- The number of rocks Mrs. Hilt currently has -/
def current_rocks : ℕ := total_rocks_needed - additional_rocks_needed

theorem hilt_current_rocks :
  current_rocks = 64 :=
sorry

end NUMINAMATH_CALUDE_hilt_current_rocks_l3517_351793


namespace NUMINAMATH_CALUDE_hyperbola_point_range_l3517_351716

theorem hyperbola_point_range (x₀ y₀ : ℝ) : 
  (x₀^2 / 2 - y₀^2 = 1) →  -- Point on hyperbola
  (((-Real.sqrt 3 - x₀) * (Real.sqrt 3 - x₀) + (-y₀) * (-y₀)) ≤ 0) →  -- Dot product condition
  (-Real.sqrt 3 / 3 ≤ y₀ ∧ y₀ ≤ Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_point_range_l3517_351716


namespace NUMINAMATH_CALUDE_min_disks_theorem_l3517_351761

/-- The number of labels -/
def n : ℕ := 60

/-- The minimum number of disks with the same label we want to guarantee -/
def k : ℕ := 12

/-- The sum of arithmetic sequence from 1 to m -/
def sum_to (m : ℕ) : ℕ := m * (m + 1) / 2

/-- The total number of disks -/
def total_disks : ℕ := sum_to n

/-- The function to calculate the minimum number of disks to draw -/
def min_disks_to_draw : ℕ := sum_to (k - 1) + (n - (k - 1)) * (k - 1) + 1

/-- The theorem stating the minimum number of disks to draw -/
theorem min_disks_theorem : min_disks_to_draw = 606 := by sorry

end NUMINAMATH_CALUDE_min_disks_theorem_l3517_351761


namespace NUMINAMATH_CALUDE_prime_sum_and_seven_sum_squares_l3517_351730

theorem prime_sum_and_seven_sum_squares (p q : ℕ) : 
  Prime p → Prime q → 
  ∃ x y : ℕ, x^2 = p + q ∧ y^2 = p + 7*q → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_and_seven_sum_squares_l3517_351730


namespace NUMINAMATH_CALUDE_stream_speed_proof_l3517_351700

/-- Proves that the speed of the stream is 21 kmph given the conditions of the rowing problem. -/
theorem stream_speed_proof (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 63 →
  (1 / (boat_speed - stream_speed)) = (2 / (boat_speed + stream_speed)) →
  stream_speed = 21 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_proof_l3517_351700


namespace NUMINAMATH_CALUDE_problem_solution_l3517_351707

theorem problem_solution : 
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁^3 - 3*x₁*y₁^2 = 2010) ∧ (y₁^3 - 3*x₁^2*y₁ = 2006) ∧
    (x₂^3 - 3*x₂*y₂^2 = 2010) ∧ (y₂^3 - 3*x₂^2*y₂ = 2006) ∧
    (x₃^3 - 3*x₃*y₃^2 = 2010) ∧ (y₃^3 - 3*x₃^2*y₃ = 2006) ∧
    ((1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 996/1005) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3517_351707


namespace NUMINAMATH_CALUDE_unique_solution_for_system_l3517_351789

theorem unique_solution_for_system :
  ∀ (x y z : ℝ),
  (x^2 + y^2 + z^2 = 2) →
  (x - z = 2) →
  (x = 1 ∧ y = 0 ∧ z = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l3517_351789


namespace NUMINAMATH_CALUDE_expression_evaluation_l3517_351743

theorem expression_evaluation : 2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3517_351743


namespace NUMINAMATH_CALUDE_handshake_theorem_l3517_351767

/-- The number of handshakes for each student in a class where every two students shake hands once. -/
def handshakes_per_student (n : ℕ) : ℕ := n - 1

/-- The total number of handshakes in a class where every two students shake hands once. -/
def total_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a class of 57 students, if every two students shake hands with each other once, 
    then each student shakes hands 56 times, and the total number of handshakes is (57 × 56) / 2. -/
theorem handshake_theorem :
  handshakes_per_student 57 = 56 ∧ total_handshakes 57 = (57 * 56) / 2 := by
  sorry

end NUMINAMATH_CALUDE_handshake_theorem_l3517_351767


namespace NUMINAMATH_CALUDE_problem_statement_l3517_351717

theorem problem_statement (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 4 * Real.sqrt 2) = Q) :
  10 * (6 * x + 8 * Real.sqrt 2 - Real.sqrt 2) = 4 * Q - 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3517_351717


namespace NUMINAMATH_CALUDE_students_taking_none_in_high_school_l3517_351725

/-- The number of students taking neither music, nor art, nor science in a high school -/
def students_taking_none (total : ℕ) (music art science : ℕ) (music_and_art music_and_science art_and_science : ℕ) (all_three : ℕ) : ℕ :=
  total - (music + art + science - music_and_art - music_and_science - art_and_science + all_three)

/-- Theorem stating the number of students taking neither music, nor art, nor science -/
theorem students_taking_none_in_high_school :
  students_taking_none 800 80 60 50 30 25 20 15 = 670 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_none_in_high_school_l3517_351725


namespace NUMINAMATH_CALUDE_purely_imaginary_z_and_z_plus_one_squared_l3517_351752

theorem purely_imaginary_z_and_z_plus_one_squared (z : ℂ) :
  (z.re = 0) → ((z + 1)^2).re = 0 → (z = Complex.I ∨ z = -Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_and_z_plus_one_squared_l3517_351752


namespace NUMINAMATH_CALUDE_numbers_close_to_zero_not_set_l3517_351719

-- Define the property of being "close to 0"
def CloseToZero (x : ℝ) : Prop := sorry

-- Define the criteria for set formation
structure SetCriteria :=
  (definiteness : Prop)
  (distinctness : Prop)
  (unorderedness : Prop)

-- Define a function to check if a collection satisfies set criteria
def SatisfiesSetCriteria (S : Set ℝ) (criteria : SetCriteria) : Prop := sorry

-- Theorem stating that "numbers close to 0" cannot form a set
theorem numbers_close_to_zero_not_set :
  ¬ ∃ (S : Set ℝ) (criteria : SetCriteria), 
    (∀ x ∈ S, CloseToZero x) ∧ 
    SatisfiesSetCriteria S criteria :=
sorry

end NUMINAMATH_CALUDE_numbers_close_to_zero_not_set_l3517_351719


namespace NUMINAMATH_CALUDE_fish_pond_estimation_l3517_351711

def fish_estimation (initial_catch : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) : ℕ :=
  (initial_catch * second_catch) / marked_in_second

theorem fish_pond_estimation :
  let initial_catch := 200
  let second_catch := 200
  let marked_in_second := 8
  fish_estimation initial_catch second_catch marked_in_second = 5000 := by
  sorry

end NUMINAMATH_CALUDE_fish_pond_estimation_l3517_351711


namespace NUMINAMATH_CALUDE_certain_number_proof_l3517_351756

theorem certain_number_proof (h1 : 268 * 74 = 19732) (n : ℝ) (h2 : 2.68 * n = 1.9832) : n = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3517_351756
