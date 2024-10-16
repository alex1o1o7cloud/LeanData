import Mathlib

namespace NUMINAMATH_CALUDE_least_three_digit_8_heavy_l171_17148

def is_8_heavy (n : ℕ) : Prop := n % 8 > 6

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_8_heavy : 
  (∀ n : ℕ, is_three_digit n → is_8_heavy n → 103 ≤ n) ∧ 
  is_three_digit 103 ∧ 
  is_8_heavy 103 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_8_heavy_l171_17148


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l171_17127

theorem concentric_circles_ratio (r₁ r₂ r₃ : ℝ) (h₁ : 0 < r₁) (h₂ : r₁ < r₂) (h₃ : r₂ < r₃) :
  (r₂^2 - r₁^2 = 2 * (r₃^2 - r₂^2)) →
  (r₃^2 = 3 * (r₂^2 - r₁^2)) →
  ∃ (k : ℝ), r₃ = k ∧ r₂ = k * Real.sqrt (5/6) ∧ r₁ = k / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l171_17127


namespace NUMINAMATH_CALUDE_total_tickets_is_84_l171_17164

-- Define the prices of items in tickets
def hat_price : ℕ := 2
def stuffed_animal_price : ℕ := 10
def yoyo_price : ℕ := 2
def keychain_price : ℕ := 3
def poster_price : ℕ := 7
def toy_car_price : ℕ := 5
def puzzle_price : ℕ := 8
def tshirt_price : ℕ := 15
def novelty_pen_price : ℕ := 4

-- Define the special offer price for two posters
def two_posters_special_price : ℕ := 10

-- Define the function to calculate the total tickets spent
def total_tickets_spent : ℕ :=
  -- First trip
  hat_price + stuffed_animal_price + yoyo_price +
  -- Second trip
  keychain_price + poster_price + toy_car_price +
  -- Third trip
  puzzle_price + tshirt_price + novelty_pen_price +
  -- Fourth trip (special offer for posters)
  two_posters_special_price + stuffed_animal_price +
  -- Fifth trip (50% off sale)
  (tshirt_price / 2) + (toy_car_price / 2)

-- Theorem to prove
theorem total_tickets_is_84 : total_tickets_spent = 84 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_is_84_l171_17164


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l171_17132

theorem smallest_n_for_sqrt_difference : 
  ∀ n : ℕ, n > 0 → (Real.sqrt n - Real.sqrt (n - 1) < 0.02 → n ≥ 626) ∧ 
  (Real.sqrt 626 - Real.sqrt 625 < 0.02) := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l171_17132


namespace NUMINAMATH_CALUDE_initial_machines_l171_17193

/-- The number of machines initially working on a job, given completion times with different numbers of machines. -/
theorem initial_machines (M : ℕ) (R : ℝ) : 
  (M * R * 12 = 1) →  -- M machines complete the job in 12 days
  ((M + 6) * R * 8 = 1) →  -- M + 6 machines complete the job in 8 days
  M = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_machines_l171_17193


namespace NUMINAMATH_CALUDE_max_k_value_l171_17109

theorem max_k_value (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a + b + c = a * b + b * c + c * a) :
  ∃ k : ℝ, k = 1 ∧ 
  ∀ k' : ℝ, 
    ((a + b + c) * ((1 / (a + b)) + (1 / (c + b)) + (1 / (a + c)) - k') ≥ k') → 
    k' ≤ k :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l171_17109


namespace NUMINAMATH_CALUDE_max_element_bound_l171_17181

/-- A set of 5 different positive integers -/
def IntegerSet : Type := Fin 5 → ℕ+

/-- The mean of the set is 20 -/
def hasMean20 (s : IntegerSet) : Prop :=
  (s 0 + s 1 + s 2 + s 3 + s 4 : ℚ) / 5 = 20

/-- The median of the set is 18 -/
def hasMedian18 (s : IntegerSet) : Prop :=
  s 2 = 18

/-- The elements of the set are distinct -/
def isDistinct (s : IntegerSet) : Prop :=
  ∀ i j, i ≠ j → s i ≠ s j

/-- The elements are in ascending order -/
def isAscending (s : IntegerSet) : Prop :=
  ∀ i j, i < j → s i < s j

theorem max_element_bound (s : IntegerSet) 
  (h_mean : hasMean20 s)
  (h_median : hasMedian18 s)
  (h_distinct : isDistinct s)
  (h_ascending : isAscending s) :
  s 4 ≤ 60 :=
sorry

end NUMINAMATH_CALUDE_max_element_bound_l171_17181


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l171_17131

theorem cubic_equation_solution (a b c : ℝ) : 
  (a^3 - 7*a^2 + 12*a = 18) ∧ 
  (b^3 - 7*b^2 + 12*b = 18) ∧ 
  (c^3 - 7*c^2 + 12*c = 18) →
  a*b/c + b*c/a + c*a/b = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l171_17131


namespace NUMINAMATH_CALUDE_mothers_age_problem_l171_17143

theorem mothers_age_problem (x : ℕ) : 
  x + 3 * x = 40 → x = 10 := by sorry

end NUMINAMATH_CALUDE_mothers_age_problem_l171_17143


namespace NUMINAMATH_CALUDE_min_value_sum_l171_17179

theorem min_value_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  x + 3 * y + 6 * z ≥ 18 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 8 ∧ x₀ + 3 * y₀ + 6 * z₀ = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l171_17179


namespace NUMINAMATH_CALUDE_fries_popcorn_ratio_is_two_to_one_l171_17157

/-- Represents the movie night scenario with Joseph and his friends -/
structure MovieNight where
  first_movie_length : ℕ
  second_movie_length : ℕ
  popcorn_time : ℕ
  total_time : ℕ

/-- Calculates the ratio of fries-making time to popcorn-making time -/
def fries_to_popcorn_ratio (mn : MovieNight) : ℚ :=
  let total_movie_time := mn.first_movie_length + mn.second_movie_length
  let fries_time := mn.total_time - total_movie_time - mn.popcorn_time
  fries_time / mn.popcorn_time

/-- Theorem stating the ratio of fries-making time to popcorn-making time is 2:1 -/
theorem fries_popcorn_ratio_is_two_to_one (mn : MovieNight)
    (h1 : mn.first_movie_length = 90)
    (h2 : mn.second_movie_length = mn.first_movie_length + 30)
    (h3 : mn.popcorn_time = 10)
    (h4 : mn.total_time = 240) :
    fries_to_popcorn_ratio mn = 2 := by
  sorry

end NUMINAMATH_CALUDE_fries_popcorn_ratio_is_two_to_one_l171_17157


namespace NUMINAMATH_CALUDE_book_reading_time_l171_17152

/-- Given a book with a certain number of pages and initial reading pace,
    calculate the number of days needed to finish the book with an increased reading pace. -/
theorem book_reading_time (total_pages : ℕ) (initial_pages_per_day : ℕ) (initial_days : ℕ) 
    (increase : ℕ) (h1 : total_pages = initial_pages_per_day * initial_days)
    (h2 : initial_pages_per_day = 15) (h3 : initial_days = 24) (h4 : increase = 3) : 
    total_pages / (initial_pages_per_day + increase) = 20 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l171_17152


namespace NUMINAMATH_CALUDE_unique_solution_l171_17197

/-- The set of digits used in the equation -/
def Digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The sum of all digits from 0 to 9 -/
def DigitsSum : Nat := Finset.sum Digits id

/-- The set of digits used on the left side of the equation -/
def LeftDigits : Finset Nat := {0, 1, 2, 4, 5, 7, 8, 9}

/-- The sum of digits on the left side of the equation -/
def LeftSum : Nat := Finset.sum LeftDigits id

/-- The two-digit number on the right side of the equation -/
def RightNumber : Nat := 36

/-- The statement that the equation is a valid solution -/
theorem unique_solution :
  (LeftSum = RightNumber) ∧
  (Digits \ LeftDigits).card = 2 ∧
  (∀ (s : Finset Nat), s ⊂ Digits → s.card = 8 → Finset.sum s id ≠ RightNumber) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l171_17197


namespace NUMINAMATH_CALUDE_abc_equality_l171_17189

theorem abc_equality (a b c : ℕ) 
  (h : ∀ n : ℕ, (a * b * c)^n ∣ ((a^n - 1) * (b^n - 1) * (c^n - 1) + 1)^3) : 
  a = b ∧ b = c :=
sorry

end NUMINAMATH_CALUDE_abc_equality_l171_17189


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l171_17170

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) = 7) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 7) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l171_17170


namespace NUMINAMATH_CALUDE_units_digit_of_result_l171_17141

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of the calculation -/
def result : ℕ := 7 * 18 * 1978 - 7^4

theorem units_digit_of_result : unitsDigit result = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_result_l171_17141


namespace NUMINAMATH_CALUDE_remainder_times_seven_l171_17137

theorem remainder_times_seven (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = quotient * divisor + remainder →
  dividend = 972345 →
  divisor = 145 →
  remainder < divisor →
  remainder * 7 = 840 := by
sorry

end NUMINAMATH_CALUDE_remainder_times_seven_l171_17137


namespace NUMINAMATH_CALUDE_quadratic_equation_root_difference_l171_17199

theorem quadratic_equation_root_difference (k : ℝ) : 
  (∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ 2 * x₁^2 + k * x₁ + 26 = 0 ∧ 2 * x₂^2 + k * x₂ + 26 = 0) →
  Complex.abs (x₁ - x₂) = 6 →
  k = 4 * Real.sqrt 22 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_difference_l171_17199


namespace NUMINAMATH_CALUDE_difference_greater_than_one_l171_17121

theorem difference_greater_than_one (x : ℕ+) :
  (x.val + 3 : ℚ) / 2 - (2 * x.val - 1 : ℚ) / 3 > 1 ↔ x.val < 5 := by
  sorry

end NUMINAMATH_CALUDE_difference_greater_than_one_l171_17121


namespace NUMINAMATH_CALUDE_product_of_sines_equals_one_eighth_l171_17180

theorem product_of_sines_equals_one_eighth :
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) *
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_equals_one_eighth_l171_17180


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l171_17190

/-- For a quadratic equation kx^2 + 2x - 1 = 0 to have two equal real roots, k must equal -1 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0 ∧ 
   ∀ y : ℝ, k * y^2 + 2 * y - 1 = 0 → y = x) → 
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l171_17190


namespace NUMINAMATH_CALUDE_equation_three_holds_l171_17184

theorem equation_three_holds (square : ℚ) (h : square = 3 + 1/20) : 
  ((6.5 - 2/3) / (3 + 1/2) - (1 + 8/15)) * (square + 71.95) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_three_holds_l171_17184


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l171_17120

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 1 / y) ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 1 / y = 4 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l171_17120


namespace NUMINAMATH_CALUDE_gcd_nine_factorial_six_factorial_squared_l171_17178

theorem gcd_nine_factorial_six_factorial_squared : Nat.gcd (Nat.factorial 9) ((Nat.factorial 6)^2) = 51840 := by
  sorry

end NUMINAMATH_CALUDE_gcd_nine_factorial_six_factorial_squared_l171_17178


namespace NUMINAMATH_CALUDE_school_store_sale_l171_17124

/-- The number of pencils sold in a school store sale -/
def pencils_sold (first_two : ℕ) (next_six : ℕ) (last_two : ℕ) : ℕ :=
  2 * first_two + 6 * next_six + 2 * last_two

/-- Theorem: Given the conditions of the pencil sale, 24 pencils were sold -/
theorem school_store_sale : pencils_sold 2 3 1 = 24 := by
  sorry

end NUMINAMATH_CALUDE_school_store_sale_l171_17124


namespace NUMINAMATH_CALUDE_intersection_point_mod17_l171_17133

theorem intersection_point_mod17 :
  ∃ x : ℕ, x < 17 ∧
  (∀ y : ℕ, (y ≡ 7 * x + 3 [MOD 17]) ↔ (y ≡ 13 * x + 4 [MOD 17])) ∧
  x = 14 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_mod17_l171_17133


namespace NUMINAMATH_CALUDE_log_sum_equality_l171_17185

theorem log_sum_equality : 21 * Real.log 2 + Real.log 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l171_17185


namespace NUMINAMATH_CALUDE_no_periodic_sequence_for_factorial_digits_l171_17115

/-- a_n is the first non-zero digit from the right in the decimal representation of n! -/
def first_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The theorem states that for all natural numbers N, the sequence of first non-zero digits
    from the right in the decimal representation of (N+k)! for k ≥ 1 is not periodic. -/
theorem no_periodic_sequence_for_factorial_digits :
  ∀ N : ℕ, ¬ ∃ T : ℕ+, ∀ k : ℕ, first_nonzero_digit (N + k + 1) = first_nonzero_digit (N + k + 1 + T) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_sequence_for_factorial_digits_l171_17115


namespace NUMINAMATH_CALUDE_derek_savings_l171_17100

theorem derek_savings (a₁ a₂ : ℕ) (sum : ℕ) : 
  a₁ = 2 → a₂ = 4 → sum = 4096 → 
  ∃ (r : ℚ), r > 0 ∧ 
    (∀ n : ℕ, n > 0 → n ≤ 12 → a₁ * r^(n-1) = a₂ * r^(n-2)) ∧
    (sum = a₁ * (1 - r^12) / (1 - r)) →
  a₁ * r^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_derek_savings_l171_17100


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l171_17101

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a 10-player round-robin tournament, there are 45 matches -/
theorem ten_player_tournament_matches :
  num_matches 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l171_17101


namespace NUMINAMATH_CALUDE_sin_double_angle_with_tan_three_l171_17158

theorem sin_double_angle_with_tan_three (θ : ℝ) :
  (∃ (x y : ℝ), x > 0 ∧ y = 3 * x ∧ Real.cos θ * x = Real.sin θ * y) →
  Real.sin (2 * θ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_with_tan_three_l171_17158


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_prime_l171_17162

theorem infinitely_many_divisible_by_prime (p : Nat) (hp : Prime p) :
  ∃ f : ℕ → ℕ, ∀ k : ℕ, p ∣ (2^(f k) - f k) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_prime_l171_17162


namespace NUMINAMATH_CALUDE_line_relationship_indeterminate_l171_17105

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- Perpendicularity relation between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- States that two lines have an indeterminate relationship -/
def indeterminate_relationship (l1 l2 : Line3D) : Prop := sorry

theorem line_relationship_indeterminate 
  (l1 l2 l3 l4 : Line3D) 
  (h1 : perpendicular l1 l2)
  (h2 : parallel l2 l3)
  (h3 : perpendicular l3 l4)
  (h4 : l1 ≠ l2 ∧ l1 ≠ l3 ∧ l1 ≠ l4 ∧ l2 ≠ l3 ∧ l2 ≠ l4 ∧ l3 ≠ l4) :
  indeterminate_relationship l1 l4 := by
  sorry

end NUMINAMATH_CALUDE_line_relationship_indeterminate_l171_17105


namespace NUMINAMATH_CALUDE_inequality_proof_l171_17161

theorem inequality_proof (a b c d : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hn : n ≥ 9) :
  a^n + b^n + c^n + d^n ≥ 
  a^(n-9) * b^4 * c^3 * d^2 + 
  b^(n-9) * c^4 * d^3 * a^2 + 
  c^(n-9) * d^4 * a^3 * b^2 + 
  d^(n-9) * a^4 * b^3 * c^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l171_17161


namespace NUMINAMATH_CALUDE_kid_tickets_sold_l171_17175

/-- Prove that the number of kid tickets sold is 75 -/
theorem kid_tickets_sold (total_tickets : ℕ) (total_profit : ℕ) 
  (adult_price kid_price : ℕ) (h1 : total_tickets = 175) 
  (h2 : total_profit = 750) (h3 : adult_price = 6) (h4 : kid_price = 2) : 
  ∃ (adult_tickets kid_tickets : ℕ), 
    adult_tickets + kid_tickets = total_tickets ∧ 
    adult_price * adult_tickets + kid_price * kid_tickets = total_profit ∧
    kid_tickets = 75 :=
sorry

end NUMINAMATH_CALUDE_kid_tickets_sold_l171_17175


namespace NUMINAMATH_CALUDE_function_inequality_l171_17156

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem function_inequality (hf : ∀ x > 0, x * f' x + x^2 < f x) :
  (2 * f 1 > f 2 + 2) ∧ (3 * f 1 > f 3 + 3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l171_17156


namespace NUMINAMATH_CALUDE_noah_age_after_10_years_l171_17134

def joe_age : ℕ := 6
def noah_age : ℕ := 2 * joe_age
def years_passed : ℕ := 10

theorem noah_age_after_10_years :
  noah_age + years_passed = 22 := by
  sorry

end NUMINAMATH_CALUDE_noah_age_after_10_years_l171_17134


namespace NUMINAMATH_CALUDE_m_less_than_one_sufficient_not_necessary_l171_17116

-- Define the function f(x) = x^2 + 2x + m
def f (x m : ℝ) : ℝ := x^2 + 2*x + m

-- Define what it means for f to have a root
def has_root (m : ℝ) : Prop := ∃ x : ℝ, f x m = 0

-- Statement: "m < 1" is a sufficient but not necessary condition for f to have a root
theorem m_less_than_one_sufficient_not_necessary :
  (∀ m : ℝ, m < 1 → has_root m) ∧ 
  (∃ m : ℝ, ¬(m < 1) ∧ has_root m) :=
sorry

end NUMINAMATH_CALUDE_m_less_than_one_sufficient_not_necessary_l171_17116


namespace NUMINAMATH_CALUDE_distance_of_symmetric_points_on_parabola_l171_17154

-- Define the parabola
def parabola (x : ℝ) : ℝ := 3 - x^2

-- Define the symmetry line
def symmetryLine (x y : ℝ) : Prop := x + y = 0

-- Define a point on the parabola
def pointOnParabola (p : ℝ × ℝ) : Prop :=
  p.2 = parabola p.1

-- Define symmetry with respect to the line x + y = 0
def symmetricPoints (p q : ℝ × ℝ) : Prop :=
  q.1 = p.2 ∧ q.2 = p.1

-- The main theorem
theorem distance_of_symmetric_points_on_parabola (A B : ℝ × ℝ) :
  pointOnParabola A →
  pointOnParabola B →
  A ≠ B →
  symmetricPoints A B →
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_distance_of_symmetric_points_on_parabola_l171_17154


namespace NUMINAMATH_CALUDE_base_four_for_64_l171_17107

theorem base_four_for_64 : ∃! b : ℕ, b > 1 ∧ b ^ 3 ≤ 64 ∧ 64 < b ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_base_four_for_64_l171_17107


namespace NUMINAMATH_CALUDE_geometric_sequence_n_l171_17140

theorem geometric_sequence_n (a₁ q aₙ : ℚ) (n : ℕ) : 
  a₁ = 1/2 → q = 1/2 → aₙ = 1/32 → aₙ = a₁ * q^(n-1) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_n_l171_17140


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l171_17139

theorem complex_magnitude_equation (t : ℝ) (h : t > 0) :
  Complex.abs (2 + t * Complex.I) = 4 * Real.sqrt 10 → t = 2 * Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l171_17139


namespace NUMINAMATH_CALUDE_range_of_g_l171_17171

/-- A function g defined on the interval [-1, 1] with g(x) = cx + d, where c < 0 and d > 0 -/
def g (c d : ℝ) (hc : c < 0) (hd : d > 0) : ℝ → ℝ :=
  fun x => c * x + d

/-- The range of g is [c + d, -c + d] -/
theorem range_of_g (c d : ℝ) (hc : c < 0) (hd : d > 0) :
  Set.range (g c d hc hd) = Set.Icc (c + d) (-c + d) := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l171_17171


namespace NUMINAMATH_CALUDE_sought_hyperbola_satisfies_conditions_l171_17118

/-- Given hyperbola equation -/
def given_hyperbola (x y : ℝ) : Prop :=
  x^2 / 5 - y^2 / 4 = 1

/-- Asymptotes of the given hyperbola -/
def given_asymptotes (x y : ℝ) : Prop :=
  y = (2 / Real.sqrt 5) * x ∨ y = -(2 / Real.sqrt 5) * x

/-- The equation of the sought hyperbola -/
def sought_hyperbola (x y : ℝ) : Prop :=
  5 * y^2 / 4 - x^2 = 1

/-- Theorem stating that the sought hyperbola satisfies the required conditions -/
theorem sought_hyperbola_satisfies_conditions :
  (∀ x y : ℝ, given_asymptotes x y ↔ (y = (2 / Real.sqrt 5) * x ∨ y = -(2 / Real.sqrt 5) * x)) ∧
  sought_hyperbola 2 2 :=
sorry


end NUMINAMATH_CALUDE_sought_hyperbola_satisfies_conditions_l171_17118


namespace NUMINAMATH_CALUDE_beef_weight_problem_l171_17183

theorem beef_weight_problem (initial_weight : ℝ) : 
  initial_weight > 0 →
  initial_weight * (1 - 0.3) * (1 - 0.2) * (1 - 0.5) = 315 →
  initial_weight = 1125 := by
sorry

end NUMINAMATH_CALUDE_beef_weight_problem_l171_17183


namespace NUMINAMATH_CALUDE_original_room_population_l171_17145

theorem original_room_population (x : ℚ) : 
  (1 / 2 : ℚ) * x = 18 →
  (2 / 3 : ℚ) * x - (1 / 4 : ℚ) * ((2 / 3 : ℚ) * x) = 18 →
  x = 36 := by sorry

end NUMINAMATH_CALUDE_original_room_population_l171_17145


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l171_17168

theorem solution_set_of_equation (x y : ℝ) : 
  (Real.sqrt (3 * x - 1) + abs (2 * y + 2) = 0) ↔ (x = 1/3 ∧ y = -1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l171_17168


namespace NUMINAMATH_CALUDE_sqrt_of_square_neg_l171_17153

theorem sqrt_of_square_neg (a : ℝ) (h : a < 0) : Real.sqrt (a^2) = -a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_neg_l171_17153


namespace NUMINAMATH_CALUDE_second_store_earns_more_l171_17146

/-- Represents the total value of goods sold by each department store -/
def total_goods_value : ℕ := 1000000

/-- Represents the discount rate offered by the first department store -/
def discount_rate : ℚ := 1/10

/-- Represents the number of lottery tickets given per 100 yuan spent -/
def tickets_per_hundred : ℕ := 1

/-- Represents the total number of lottery tickets -/
def total_tickets : ℕ := 10000

/-- Represents the number of first prizes -/
def first_prize_count : ℕ := 5

/-- Represents the value of each first prize -/
def first_prize_value : ℕ := 1000

/-- Represents the number of second prizes -/
def second_prize_count : ℕ := 10

/-- Represents the value of each second prize -/
def second_prize_value : ℕ := 500

/-- Represents the number of third prizes -/
def third_prize_count : ℕ := 20

/-- Represents the value of each third prize -/
def third_prize_value : ℕ := 200

/-- Represents the number of fourth prizes -/
def fourth_prize_count : ℕ := 40

/-- Represents the value of each fourth prize -/
def fourth_prize_value : ℕ := 100

/-- Represents the number of fifth prizes -/
def fifth_prize_count : ℕ := 1000

/-- Represents the value of each fifth prize -/
def fifth_prize_value : ℕ := 10

/-- Calculates the earnings of the first department store -/
def first_store_earnings : ℚ := total_goods_value * (1 - discount_rate)

/-- Calculates the total prize value for the second department store -/
def total_prize_value : ℕ := 
  first_prize_count * first_prize_value +
  second_prize_count * second_prize_value +
  third_prize_count * third_prize_value +
  fourth_prize_count * fourth_prize_value +
  fifth_prize_count * fifth_prize_value

/-- Calculates the earnings of the second department store -/
def second_store_earnings : ℕ := total_goods_value - total_prize_value

/-- Theorem stating that the second department store earns at least 72,000 yuan more than the first -/
theorem second_store_earns_more :
  (second_store_earnings : ℚ) - first_store_earnings ≥ 72000 := by
  sorry

end NUMINAMATH_CALUDE_second_store_earns_more_l171_17146


namespace NUMINAMATH_CALUDE_square_of_sum_85_7_l171_17163

theorem square_of_sum_85_7 : (85 + 7)^2 = 8464 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_85_7_l171_17163


namespace NUMINAMATH_CALUDE_rectangle_area_l171_17160

theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ l = 3 * w ∧ x^2 = w^2 + l^2 ∧ w * l = (3 * x^2) / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l171_17160


namespace NUMINAMATH_CALUDE_conference_hall_tables_l171_17177

/-- Represents the setup of a conference hall --/
structure ConferenceHall where
  tables : ℕ
  chairs_per_table : ℕ
  chair_legs : ℕ
  table_legs : ℕ
  sofa_legs : ℕ
  total_legs : ℕ

/-- The conference hall setup satisfies the given conditions --/
def valid_setup (hall : ConferenceHall) : Prop :=
  hall.chairs_per_table = 8 ∧
  hall.chair_legs = 4 ∧
  hall.table_legs = 5 ∧
  hall.sofa_legs = 6 ∧
  hall.total_legs = 760

/-- The number of sofas is half the number of tables --/
def sofa_table_relation (hall : ConferenceHall) : Prop :=
  2 * (hall.tables / 2) = hall.tables

/-- The total number of legs is correctly calculated --/
def correct_leg_count (hall : ConferenceHall) : Prop :=
  hall.total_legs = 
    hall.chair_legs * (hall.chairs_per_table * hall.tables) +
    hall.table_legs * hall.tables +
    hall.sofa_legs * (hall.tables / 2)

/-- Theorem stating that given the conditions, there are 19 tables in the hall --/
theorem conference_hall_tables (hall : ConferenceHall) :
  valid_setup hall → sofa_table_relation hall → correct_leg_count hall → hall.tables = 19 := by
  sorry


end NUMINAMATH_CALUDE_conference_hall_tables_l171_17177


namespace NUMINAMATH_CALUDE_outfits_count_l171_17166

/-- The number of possible outfits given a set of clothing items -/
def number_of_outfits (shirts : ℕ) (ties : ℕ) (pants : ℕ) : ℕ :=
  shirts * pants * (ties + 1)

/-- Theorem stating that with 6 shirts, 4 ties, and 3 pairs of pants,
    the number of possible outfits is 90 -/
theorem outfits_count : number_of_outfits 6 4 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l171_17166


namespace NUMINAMATH_CALUDE_deepak_age_l171_17174

/-- Proves that Deepak's present age is 18 years given the conditions -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 2 = 26 →
  deepak_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l171_17174


namespace NUMINAMATH_CALUDE_water_flow_rates_verify_conditions_l171_17150

/-- Represents the water flow model with introducing and removing pipes -/
structure WaterFlowModel where
  /-- Water flow rate of one introducing pipe in m³/h -/
  inlet_rate : ℝ
  /-- Water flow rate of one removing pipe in m³/h -/
  outlet_rate : ℝ

/-- Theorem stating the correct water flow rates given the problem conditions -/
theorem water_flow_rates (model : WaterFlowModel) : 
  (5 * (4 * model.inlet_rate - 3 * model.outlet_rate) = 1000) ∧ 
  (2 * (2 * model.inlet_rate - 2 * model.outlet_rate) = 180) →
  model.inlet_rate = 65 ∧ model.outlet_rate = 20 := by
  sorry

/-- Function to calculate the net water gain in a given time period -/
def net_water_gain (model : WaterFlowModel) (inlet_count outlet_count : ℕ) (hours : ℝ) : ℝ :=
  hours * (inlet_count * model.inlet_rate - outlet_count * model.outlet_rate)

/-- Verifies that the calculated rates satisfy the given conditions -/
theorem verify_conditions (model : WaterFlowModel) 
  (h1 : model.inlet_rate = 65) 
  (h2 : model.outlet_rate = 20) : 
  net_water_gain model 4 3 5 = 1000 ∧ 
  net_water_gain model 2 2 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_rates_verify_conditions_l171_17150


namespace NUMINAMATH_CALUDE_final_inventory_calculation_l171_17187

def initial_inventory : ℕ := 4500
def monday_sales : ℕ := 2445
def tuesday_sales : ℕ := 900
def daily_sales_wed_to_sun : ℕ := 50
def days_wed_to_sun : ℕ := 5
def saturday_delivery : ℕ := 650

theorem final_inventory_calculation :
  initial_inventory - 
  (monday_sales + tuesday_sales + daily_sales_wed_to_sun * days_wed_to_sun) + 
  saturday_delivery = 1555 := by
  sorry

end NUMINAMATH_CALUDE_final_inventory_calculation_l171_17187


namespace NUMINAMATH_CALUDE_ratio_of_tenth_terms_l171_17195

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem ratio_of_tenth_terms 
  (a b : ArithmeticSequence)
  (h : ∀ n, sumFirstN a n / sumFirstN b n = (3 * n - 1) / (2 * n + 3)) :
  a.a 10 / b.a 10 = 57 / 41 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_tenth_terms_l171_17195


namespace NUMINAMATH_CALUDE_fair_die_weighted_coin_l171_17198

theorem fair_die_weighted_coin (n : ℕ) (p_heads : ℚ) : 
  n ≥ 7 →
  (p_heads = 1/3 ∨ p_heads = 2/3) →
  (1/n) * p_heads = 1/15 →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_fair_die_weighted_coin_l171_17198


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l171_17194

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 15 →
  (n * original_mean - n * decrement) / n = 185 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l171_17194


namespace NUMINAMATH_CALUDE_largest_divisible_sum_fourth_powers_l171_17169

/-- A set of n prime numbers greater than 10 -/
def PrimeSet (n : ℕ) := { S : Finset ℕ | S.card = n ∧ ∀ p ∈ S, Nat.Prime p ∧ p > 10 }

/-- The sum of fourth powers of elements in a finite set -/
def SumFourthPowers (S : Finset ℕ) : ℕ := S.sum (λ x => x^4)

/-- The main theorem statement -/
theorem largest_divisible_sum_fourth_powers :
  ∀ n > 240, ∃ S ∈ PrimeSet n, ¬ (n ∣ SumFourthPowers S) ∧
  ∀ m ≤ 240, ∀ T ∈ PrimeSet m, m ∣ SumFourthPowers T :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_sum_fourth_powers_l171_17169


namespace NUMINAMATH_CALUDE_gems_per_dollar_l171_17119

/-- Proves that the number of gems per dollar is 100 given the problem conditions -/
theorem gems_per_dollar (total_spent : ℝ) (bonus_rate : ℝ) (final_gems : ℝ) :
  total_spent = 250 →
  bonus_rate = 0.2 →
  final_gems = 30000 →
  (final_gems / (total_spent * (1 + bonus_rate))) = 100 := by
sorry

end NUMINAMATH_CALUDE_gems_per_dollar_l171_17119


namespace NUMINAMATH_CALUDE_additional_blue_tickets_for_bible_l171_17192

/-- Represents the number of tickets Tom has of each color -/
structure TicketCounts where
  yellow : ℕ
  red : ℕ
  green : ℕ
  blue : ℕ

/-- Represents the conversion rates between ticket colors -/
structure TicketRates where
  yellow_to_red : ℕ
  red_to_green : ℕ
  green_to_blue : ℕ

def calculate_additional_blue_tickets (
  bible_yellow_requirement : ℕ
  ) (rates : TicketRates) (current_tickets : TicketCounts) : ℕ :=
  sorry

theorem additional_blue_tickets_for_bible (
  bible_yellow_requirement : ℕ
  ) (rates : TicketRates) (current_tickets : TicketCounts) :
  bible_yellow_requirement = 20 →
  rates.yellow_to_red = 15 →
  rates.red_to_green = 12 →
  rates.green_to_blue = 10 →
  current_tickets.yellow = 12 →
  current_tickets.red = 8 →
  current_tickets.green = 14 →
  current_tickets.blue = 27 →
  calculate_additional_blue_tickets bible_yellow_requirement rates current_tickets = 13273 :=
by sorry

end NUMINAMATH_CALUDE_additional_blue_tickets_for_bible_l171_17192


namespace NUMINAMATH_CALUDE_prob_at_least_one_2_l171_17167

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The probability of at least one die showing 2 when two fair 8-sided dice are rolled -/
def probAtLeastOne2 : ℚ := 15 / 64

/-- Theorem stating that the probability of at least one die showing 2 
    when two fair 8-sided dice are rolled is 15/64 -/
theorem prob_at_least_one_2 : 
  probAtLeastOne2 = (numSides^2 - (numSides - 1)^2) / numSides^2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_2_l171_17167


namespace NUMINAMATH_CALUDE_entropy_increase_l171_17130

-- Define the temperature in Kelvin
def T : ℝ := 298

-- Define the enthalpy change in kJ/mol
def ΔH : ℝ := 2171

-- Define the entropy change in J/(mol·K)
def ΔS : ℝ := 635.5

-- Theorem to prove that the entropy change is positive
theorem entropy_increase : ΔS > 0 := by
  sorry

end NUMINAMATH_CALUDE_entropy_increase_l171_17130


namespace NUMINAMATH_CALUDE_hcf_problem_l171_17142

theorem hcf_problem (a b h : ℕ) (h_pos : 0 < h) (a_pos : 0 < a) (b_pos : 0 < b) :
  (Nat.gcd a b = h) →
  (∃ k : ℕ, Nat.lcm a b = 10 * 15 * k) →
  (max a b = 450) →
  h = 30 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l171_17142


namespace NUMINAMATH_CALUDE_shaded_fraction_of_large_rectangle_l171_17182

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

theorem shaded_fraction_of_large_rectangle (large : Rectangle) (small : Rectangle) 
  (h1 : large.width = 15)
  (h2 : large.height = 20)
  (h3 : small.area = (1 / 5) * large.area)
  (h4 : small.area > 0) :
  (1 / 2) * small.area / large.area = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_large_rectangle_l171_17182


namespace NUMINAMATH_CALUDE_delta_y_over_delta_x_l171_17136

/-- Given a function f(x) = 2x² + 1, prove that Δy/Δx = 4 + 2Δx for points P(1, 3) and Q(1 + Δx, 3 + Δy) -/
theorem delta_y_over_delta_x (Δx : ℝ) (Δy : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 1
  Δy = f (1 + Δx) - f 1 →
  Δx ≠ 0 →
  Δy / Δx = 4 + 2 * Δx := by
sorry

end NUMINAMATH_CALUDE_delta_y_over_delta_x_l171_17136


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l171_17144

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (50 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 10 * q * Real.sqrt (15 * q) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l171_17144


namespace NUMINAMATH_CALUDE_triangle_problem_l171_17147

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Given conditions
  t.a = 7/2 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = 3/2 * Real.sqrt 3) ∧
  (t.a * Real.sin t.B - Real.sqrt 3 * t.b * Real.cos t.A = 0) →
  -- Conclusions to prove
  (t.A = π/3) ∧ (t.b + t.c = 11/2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l171_17147


namespace NUMINAMATH_CALUDE_house_painting_and_window_washing_l171_17165

/-- Represents the number of people needed to complete a task in a given number of days -/
structure WorkForce :=
  (people : ℕ)
  (days : ℕ)

/-- Calculates the total person-days for a given workforce -/
def personDays (w : WorkForce) : ℕ := w.people * w.days

theorem house_painting_and_window_washing 
  (paint_initial : WorkForce) 
  (paint_target : WorkForce) 
  (wash_initial : WorkForce) 
  (wash_target : WorkForce) :
  paint_initial.people = 8 →
  paint_initial.days = 5 →
  paint_target.days = 3 →
  wash_initial.people = paint_initial.people →
  wash_initial.days = 4 →
  wash_target.people = wash_initial.people + 4 →
  personDays paint_initial = personDays paint_target →
  personDays wash_initial = personDays wash_target →
  paint_target.people = 14 ∧ wash_target.days = 3 := by
  sorry

#check house_painting_and_window_washing

end NUMINAMATH_CALUDE_house_painting_and_window_washing_l171_17165


namespace NUMINAMATH_CALUDE_douglas_county_x_votes_l171_17114

/-- Represents the percentage of votes Douglas won in county X -/
def douglas_x_percent : ℝ := 64

/-- Represents the percentage of votes Douglas won in county Y -/
def douglas_y_percent : ℝ := 46

/-- Represents the ratio of voters in county X to county Y -/
def county_ratio : ℝ := 2

/-- Represents the total percentage of votes Douglas won in both counties -/
def total_percent : ℝ := 58

theorem douglas_county_x_votes :
  douglas_x_percent * county_ratio + douglas_y_percent = total_percent * (county_ratio + 1) :=
by sorry

end NUMINAMATH_CALUDE_douglas_county_x_votes_l171_17114


namespace NUMINAMATH_CALUDE_two_digit_product_4320_l171_17129

theorem two_digit_product_4320 :
  ∃! (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4320 ∧ a = 60 ∧ b = 72 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_product_4320_l171_17129


namespace NUMINAMATH_CALUDE_pencils_per_box_l171_17149

/-- Given Arnel's pencil distribution scenario, prove that each box contains 5 pencils. -/
theorem pencils_per_box (num_boxes : ℕ) (num_friends : ℕ) (pencils_kept : ℕ) (pencils_per_friend : ℕ) :
  num_boxes = 10 →
  num_friends = 5 →
  pencils_kept = 10 →
  pencils_per_friend = 8 →
  (∃ (pencils_per_box : ℕ), 
    pencils_per_box * num_boxes = pencils_kept + num_friends * pencils_per_friend ∧
    pencils_per_box = 5) :=
by sorry

end NUMINAMATH_CALUDE_pencils_per_box_l171_17149


namespace NUMINAMATH_CALUDE_general_drinking_horse_shortest_distance_l171_17191

/-- The shortest distance for the "General Drinking Horse" problem -/
theorem general_drinking_horse_shortest_distance :
  let camp := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 1}
  let A : ℝ × ℝ := (2, 0)
  let riverbank := {p : ℝ × ℝ | p.1 + p.2 = 3}
  ∃ (B : ℝ × ℝ) (C : ℝ × ℝ),
    B ∈ riverbank ∧ C ∈ camp ∧
    ∀ (B' : ℝ × ℝ) (C' : ℝ × ℝ),
      B' ∈ riverbank → C' ∈ camp →
      Real.sqrt 10 - 1 ≤ dist A B' + dist B' C' :=
sorry

end NUMINAMATH_CALUDE_general_drinking_horse_shortest_distance_l171_17191


namespace NUMINAMATH_CALUDE_cone_volume_from_sphere_properties_l171_17159

/-- Given a sphere and a cone with specific properties, prove that the volume of the cone is 12288π cm³ -/
theorem cone_volume_from_sphere_properties (r : ℝ) (h : ℝ) (S_sphere : ℝ) (S_cone : ℝ) (V_cone : ℝ) :
  r = 24 →
  h = 2 * r →
  S_sphere = 4 * π * r^2 →
  S_cone = S_sphere →
  V_cone = (1/3) * π * (S_cone / (2 * π * h))^2 * h →
  V_cone = 12288 * π := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_sphere_properties_l171_17159


namespace NUMINAMATH_CALUDE_billiard_ball_trajectory_l171_17196

theorem billiard_ball_trajectory :
  ∀ (x y : ℚ),
    (x ≥ 0 ∧ y ≥ 0) →  -- Restricting to first quadrant
    (y = x / Real.sqrt 2) →  -- Line equation
    (¬ ∃ (m n : ℤ), (x = ↑m ∧ y = ↑n)) :=  -- No integer coordinate intersection
by sorry

end NUMINAMATH_CALUDE_billiard_ball_trajectory_l171_17196


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l171_17108

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (5 * a^3 + 505 * a + 1010 = 0) →
  (5 * b^3 + 505 * b + 1010 = 0) →
  (5 * c^3 + 505 * c + 1010 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 606 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l171_17108


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l171_17106

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 30) 
  (h2 : n2 = 50) 
  (h3 : avg1 = 30) 
  (h4 : avg2 = 60) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 48.75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l171_17106


namespace NUMINAMATH_CALUDE_smallest_a_for_distinct_roots_in_unit_interval_l171_17176

theorem smallest_a_for_distinct_roots_in_unit_interval :
  ∃ (b c : ℤ), 
    (∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧ 
      5 * x^2 - b * x + c = 0 ∧ 5 * y^2 - b * y + c = 0) ∧
    (∀ (a : ℕ), a < 5 → 
      ¬∃ (b c : ℤ), ∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧ 
        a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_distinct_roots_in_unit_interval_l171_17176


namespace NUMINAMATH_CALUDE_alice_most_dogs_l171_17125

-- Define the number of cats and dogs for each person
variable (Kc Ac Bc Kd Ad Bd : ℕ)

-- Define the conditions
axiom kathy_more_cats : Kc > Ac
axiom kathy_more_dogs : Kd > Bd
axiom alice_more_dogs : Ad > Kd
axiom bruce_more_cats : Bc > Ac

-- Theorem to prove
theorem alice_most_dogs : Ad > Kd ∧ Ad > Bd :=
sorry

end NUMINAMATH_CALUDE_alice_most_dogs_l171_17125


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l171_17155

/-- A sequence {a_n} with sum of first n terms S_n = p^n + q, where p ≠ 0 and p ≠ 1, 
    is geometric if and only if q = -1 -/
theorem geometric_sequence_condition (p : ℝ) (q : ℝ) (h_p_nonzero : p ≠ 0) (h_p_not_one : p ≠ 1) :
  let a : ℕ → ℝ := fun n => (p^n + q) - (p^(n-1) + q)
  let S : ℕ → ℝ := fun n => p^n + q
  (∀ n : ℕ, n ≥ 2 → a (n+1) / a n = a 2 / a 1) ↔ q = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l171_17155


namespace NUMINAMATH_CALUDE_dave_baseball_cards_pages_l171_17135

/-- The number of pages needed to organize baseball cards in a binder -/
def pages_needed (cards_per_page new_cards old_cards : ℕ) : ℕ :=
  (new_cards + old_cards + cards_per_page - 1) / cards_per_page

/-- Proof that Dave needs 2 pages to organize his baseball cards -/
theorem dave_baseball_cards_pages :
  pages_needed 8 3 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_baseball_cards_pages_l171_17135


namespace NUMINAMATH_CALUDE_shortest_rope_length_l171_17117

theorem shortest_rope_length (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a : ℝ) / 4 = (b : ℝ) / 5 ∧ (b : ℝ) / 5 = (c : ℝ) / 6 →
  a + c = b + 100 →
  a = 80 := by
sorry

end NUMINAMATH_CALUDE_shortest_rope_length_l171_17117


namespace NUMINAMATH_CALUDE_first_term_exceeding_thousand_l171_17104

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Predicate to check if a term exceeds 1000 -/
def exceedsThousand (x : ℝ) : Prop :=
  x > 1000

theorem first_term_exceeding_thousand :
  let a₁ := 2
  let d := 3
  (∀ n < 334, ¬(exceedsThousand (arithmeticSequenceTerm a₁ d n))) ∧
  exceedsThousand (arithmeticSequenceTerm a₁ d 334) :=
by sorry

end NUMINAMATH_CALUDE_first_term_exceeding_thousand_l171_17104


namespace NUMINAMATH_CALUDE_storybook_pages_l171_17128

/-- The number of days between two dates (inclusive) -/
def daysBetween (startDate endDate : Nat) : Nat :=
  endDate - startDate + 1

theorem storybook_pages : 
  let startDate := 10  -- March 10th
  let endDate := 20    -- March 20th
  let pagesPerDay := 11
  let readingDays := daysBetween startDate endDate
  readingDays * pagesPerDay = 121 := by
  sorry

end NUMINAMATH_CALUDE_storybook_pages_l171_17128


namespace NUMINAMATH_CALUDE_triathlon_bicycle_speed_triathlon_solution_l171_17113

theorem triathlon_bicycle_speed 
  (total_time : ℝ) 
  (swim_speed swim_distance : ℝ) 
  (run_speed run_distance : ℝ) 
  (bike_distance : ℝ) : ℝ :=
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

theorem triathlon_solution :
  triathlon_bicycle_speed 3 1 0.5 8 4 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_bicycle_speed_triathlon_solution_l171_17113


namespace NUMINAMATH_CALUDE_total_cost_is_180_l171_17186

/-- The cost to fill all planter pots at the corners of a rectangle-shaped pool -/
def total_cost : ℝ :=
  let corners_of_rectangle := 4
  let palm_fern_cost := 15.00
  let creeping_jenny_cost := 4.00
  let geranium_cost := 3.50
  let palm_ferns_per_pot := 1
  let creeping_jennies_per_pot := 4
  let geraniums_per_pot := 4
  let cost_per_pot := palm_fern_cost * palm_ferns_per_pot + 
                      creeping_jenny_cost * creeping_jennies_per_pot + 
                      geranium_cost * geraniums_per_pot
  corners_of_rectangle * cost_per_pot

/-- Theorem stating that the total cost to fill all planter pots is $180.00 -/
theorem total_cost_is_180 : total_cost = 180.00 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_180_l171_17186


namespace NUMINAMATH_CALUDE_rectangles_in_grid_l171_17102

def grid_size : ℕ := 5

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of rectangles in an n x n grid -/
def num_rectangles (n : ℕ) : ℕ := (choose_two n) ^ 2

theorem rectangles_in_grid :
  num_rectangles grid_size = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangles_in_grid_l171_17102


namespace NUMINAMATH_CALUDE_remainder_theorem_l171_17103

theorem remainder_theorem (N : ℤ) (h : ∃ k : ℤ, N = 39 * k + 18) : 
  ∃ m : ℤ, N = 13 * m + 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l171_17103


namespace NUMINAMATH_CALUDE_polynomial_simplification_l171_17188

theorem polynomial_simplification (s : ℝ) : (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 4) = -4 * s + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l171_17188


namespace NUMINAMATH_CALUDE_income_analysis_l171_17111

/-- Represents the income status of a household -/
inductive IncomeStatus
| Above10000
| Below10000

/-- Represents a region with households -/
structure Region where
  totalHouseholds : ℕ
  aboveThreshold : ℕ

/-- Represents the sample data -/
structure SampleData where
  regionA : Region
  regionB : Region
  totalSample : ℕ

/-- The probability of selecting a household with income above 10000 from a region -/
def probAbove10000 (r : Region) : ℚ :=
  r.aboveThreshold / r.totalHouseholds

/-- The expected value of X (number of households with income > 10000 when selecting one from each region) -/
def expectedX (sd : SampleData) : ℚ :=
  (probAbove10000 sd.regionA + probAbove10000 sd.regionB) / 2

/-- The main theorem to be proved -/
theorem income_analysis (sd : SampleData)
  (h1 : sd.regionA.totalHouseholds = 300)
  (h2 : sd.regionA.aboveThreshold = 100)
  (h3 : sd.regionB.totalHouseholds = 200)
  (h4 : sd.regionB.aboveThreshold = 150)
  (h5 : sd.totalSample = 500) :
  probAbove10000 sd.regionA = 1/3 ∧ expectedX sd = 13/12 := by
  sorry

end NUMINAMATH_CALUDE_income_analysis_l171_17111


namespace NUMINAMATH_CALUDE_softball_team_ratio_l171_17138

theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
    women = men + 6 → 
    men + women = 16 → 
    (men : ℚ) / women = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l171_17138


namespace NUMINAMATH_CALUDE_triangle_theorem_l171_17126

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h3 : A + B + C = π)
  (h4 : a * sin A = b * sin B)
  (h5 : b * sin B = c * sin C)
  (h6 : a * sin A - c * sin C = (a - b) * sin B)

/-- The theorem stating the angle C and maximum area of the triangle -/
theorem triangle_theorem (t : Triangle) (h : t.c = sqrt 6) :
  t.C = π / 3 ∧
  ∃ (S : ℝ), S = (3 * sqrt 3) / 2 ∧ ∀ (S' : ℝ), S' ≤ S := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l171_17126


namespace NUMINAMATH_CALUDE_division_problem_l171_17123

theorem division_problem (x : ℝ) (h : 10 / x = 2) : 20 / x = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l171_17123


namespace NUMINAMATH_CALUDE_trapezoid_median_equals_12_l171_17110

/-- Given a triangle and a trapezoid with equal areas and altitudes, where the triangle's base is 24 inches and one base of the trapezoid is twice the other, prove the trapezoid's median is 12 inches. -/
theorem trapezoid_median_equals_12 (h : ℝ) (x : ℝ) : 
  h > 0 →  -- Altitude is positive
  (1/2) * 24 * h = ((x + 2*x) / 2) * h →  -- Equal areas
  (x + 2*x) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_median_equals_12_l171_17110


namespace NUMINAMATH_CALUDE_perfect_squares_theorem_l171_17122

theorem perfect_squares_theorem (x y z : ℕ+) 
  (h_coprime : ∀ d : ℕ, d > 1 → ¬(d ∣ x ∧ d ∣ y ∧ d ∣ z))
  (h_eq : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = (z : ℚ)⁻¹) :
  ∃ (a b : ℕ), 
    (x : ℤ) - (z : ℤ) = a^2 ∧ 
    (y : ℤ) - (z : ℤ) = b^2 ∧ 
    (x : ℤ) + (y : ℤ) = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_theorem_l171_17122


namespace NUMINAMATH_CALUDE_team_point_difference_l171_17151

/-- The difference in points between two teams -/
def pointDifference (beth_score jan_score judy_score angel_score : ℕ) : ℕ :=
  (beth_score + jan_score) - (judy_score + angel_score)

/-- Theorem stating the point difference between the two teams -/
theorem team_point_difference :
  pointDifference 12 10 8 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_team_point_difference_l171_17151


namespace NUMINAMATH_CALUDE_baron_munchausen_contradiction_l171_17112

-- Define the total distance and time of the walk
variable (S : ℝ) -- Total distance
variable (T : ℝ) -- Total time

-- Define the speeds
def speed1 : ℝ := 5 -- Speed for half the distance
def speed2 : ℝ := 6 -- Speed for half the time

-- Theorem: It's impossible to satisfy both conditions
theorem baron_munchausen_contradiction :
  ¬(∃ (S T : ℝ), S > 0 ∧ T > 0 ∧
    (S / 2) / speed1 + (S / 2) / speed2 = T ∧
    (S / 2) + speed2 * (T / 2) = S) :=
sorry

end NUMINAMATH_CALUDE_baron_munchausen_contradiction_l171_17112


namespace NUMINAMATH_CALUDE_point_P_coordinates_l171_17172

def P (m : ℝ) : ℝ × ℝ := (m + 3, 2*m - 1)

theorem point_P_coordinates :
  (∀ m : ℝ, P m = (0, -7) ↔ (P m).1 = 0) ∧
  (∀ m : ℝ, P m = (10, 13) ↔ (P m).2 = (P m).1 + 3) ∧
  (∀ m : ℝ, P m = (5/2, -2) ↔ |(P m).2| = 2 ∧ (P m).1 > 0 ∧ (P m).2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l171_17172


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l171_17173

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 9 * I) * (a + b * I) = y * I) : a / b = -3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l171_17173
