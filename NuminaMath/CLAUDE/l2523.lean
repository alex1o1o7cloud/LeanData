import Mathlib

namespace NUMINAMATH_CALUDE_two_over_x_is_inverse_proportion_l2523_252350

/-- A function f is an inverse proportion function if there exists a constant k such that f(x) = k/x for all non-zero x. -/
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function f(x) = 2/x is an inverse proportion function. -/
theorem two_over_x_is_inverse_proportion :
  is_inverse_proportion (λ x : ℝ => 2 / x) := by
  sorry


end NUMINAMATH_CALUDE_two_over_x_is_inverse_proportion_l2523_252350


namespace NUMINAMATH_CALUDE_percentage_problem_l2523_252349

theorem percentage_problem (x : ℝ) : 
  (15 / 100 * 40 = x / 100 * 16 + 2) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2523_252349


namespace NUMINAMATH_CALUDE_find_larger_number_l2523_252347

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1515) (h2 : L = 16 * S + 15) : L = 1615 := by
  sorry

end NUMINAMATH_CALUDE_find_larger_number_l2523_252347


namespace NUMINAMATH_CALUDE_history_not_statistics_l2523_252381

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ)
  (h_total : total = 90)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_or_statistics : history_or_statistics = 57) :
  history - (history + statistics - history_or_statistics) = 25 := by
  sorry

end NUMINAMATH_CALUDE_history_not_statistics_l2523_252381


namespace NUMINAMATH_CALUDE_largest_multiple_of_45_with_8_and_0_l2523_252387

/-- A function that checks if a natural number consists only of digits 8 and 0 -/
def onlyEightAndZero (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 8 ∨ d = 0

/-- The largest positive multiple of 45 consisting only of digits 8 and 0 -/
def m : ℕ := sorry

theorem largest_multiple_of_45_with_8_and_0 :
  m % 45 = 0 ∧
  onlyEightAndZero m ∧
  (∀ k : ℕ, k > m → k % 45 = 0 → ¬onlyEightAndZero k) ∧
  m / 45 = 197530 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_45_with_8_and_0_l2523_252387


namespace NUMINAMATH_CALUDE_complex_fraction_real_l2523_252348

theorem complex_fraction_real (a : ℝ) : 
  ((-a + Complex.I) / (1 - Complex.I)).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l2523_252348


namespace NUMINAMATH_CALUDE_barkley_buried_bones_l2523_252318

/-- Calculates the number of bones Barkley has buried given the conditions -/
def bones_buried (bones_per_month : ℕ) (months_passed : ℕ) (available_bones : ℕ) : ℕ :=
  bones_per_month * months_passed - available_bones

/-- Theorem stating that Barkley has buried 42 bones under the given conditions -/
theorem barkley_buried_bones : 
  bones_buried 10 5 8 = 42 := by
  sorry

end NUMINAMATH_CALUDE_barkley_buried_bones_l2523_252318


namespace NUMINAMATH_CALUDE_max_value_of_f_l2523_252386

def f (x : ℝ) : ℝ := -3 * x^2 + 18

theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2523_252386


namespace NUMINAMATH_CALUDE_increasing_function_implies_a_leq_neg_two_l2523_252377

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- State the theorem
theorem increasing_function_implies_a_leq_neg_two :
  ∀ a : ℝ, (∀ x y : ℝ, -2 < x ∧ x < y ∧ y < 2 → f a x < f a y) →
  a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_implies_a_leq_neg_two_l2523_252377


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2523_252370

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3/5) 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2523_252370


namespace NUMINAMATH_CALUDE_solution_equivalence_l2523_252313

/-- Given constants m and n where mx + n > 0 is equivalent to x < 1/2, 
    prove that nx - m < 0 is equivalent to x < -2 -/
theorem solution_equivalence (m n : ℝ) 
    (h : ∀ x, mx + n > 0 ↔ x < (1/2)) : 
    ∀ x, nx - m < 0 ↔ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_equivalence_l2523_252313


namespace NUMINAMATH_CALUDE_expression_evaluation_l2523_252376

theorem expression_evaluation : 200 * (200 + 5) - (200 * 200 + 5) = 995 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2523_252376


namespace NUMINAMATH_CALUDE_exists_ten_digit_number_divisible_by_11_with_all_digits_l2523_252397

def is_ten_digit_number (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def contains_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, (n / 10^k) % 10 = d

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem exists_ten_digit_number_divisible_by_11_with_all_digits :
  ∃ n : ℕ, is_ten_digit_number n ∧ contains_all_digits n ∧ is_divisible_by_11 n :=
sorry

end NUMINAMATH_CALUDE_exists_ten_digit_number_divisible_by_11_with_all_digits_l2523_252397


namespace NUMINAMATH_CALUDE_power_24_mod_15_l2523_252395

theorem power_24_mod_15 : 24^2377 % 15 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_24_mod_15_l2523_252395


namespace NUMINAMATH_CALUDE_prize_distribution_l2523_252385

theorem prize_distribution (total_winners : ℕ) (min_award : ℝ) (max_award : ℝ) : 
  total_winners = 20 →
  min_award = 20 →
  max_award = 340 →
  (∃ (prize : ℝ), 
    prize > 0 ∧
    (∀ (winner : ℕ), winner ≤ total_winners → ∃ (award : ℝ), min_award ≤ award ∧ award ≤ max_award) ∧
    (2/5 * prize = 3/5 * total_winners * max_award) ∧
    prize = 10200) :=
by sorry

end NUMINAMATH_CALUDE_prize_distribution_l2523_252385


namespace NUMINAMATH_CALUDE_complex_power_six_l2523_252352

theorem complex_power_six (i : ℂ) (h : i^2 = -1) : (1 + i)^6 = -8*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_six_l2523_252352


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2523_252302

theorem polynomial_factorization (m : ℤ) : 
  (∀ x : ℤ, x^2 + m*x - 35 = (x - 7)*(x + 5)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2523_252302


namespace NUMINAMATH_CALUDE_chord_intersection_parameter_l2523_252357

/-- Given a line and a circle, prove that the parameter a equals 1 when they intersect to form a chord of length √2. -/
theorem chord_intersection_parameter (a : ℝ) : a > 0 → ∃ (x y : ℝ),
  (x + y + a = 0) ∧ (x^2 + y^2 = a) ∧ 
  (∃ (x1 y1 x2 y2 : ℝ), (x1 + y1 + a = 0) ∧ (x2 + y2 + a = 0) ∧
                        (x1^2 + y1^2 = a) ∧ (x2^2 + y2^2 = a) ∧
                        ((x1 - x2)^2 + (y1 - y2)^2 = 2)) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_parameter_l2523_252357


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2523_252358

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 8*x + 1 = 0 ↔ (x - 4)^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2523_252358


namespace NUMINAMATH_CALUDE_milk_addition_rate_l2523_252329

/-- Calculates the rate of milk addition given initial conditions --/
theorem milk_addition_rate
  (initial_milk : ℝ)
  (pump_rate : ℝ)
  (pump_time : ℝ)
  (addition_time : ℝ)
  (final_milk : ℝ)
  (h1 : initial_milk = 30000)
  (h2 : pump_rate = 2880)
  (h3 : pump_time = 4)
  (h4 : addition_time = 7)
  (h5 : final_milk = 28980) :
  let milk_pumped := pump_rate * pump_time
  let milk_before_addition := initial_milk - milk_pumped
  let milk_added := final_milk - milk_before_addition
  milk_added / addition_time = 1500 := by
  sorry

end NUMINAMATH_CALUDE_milk_addition_rate_l2523_252329


namespace NUMINAMATH_CALUDE_students_with_both_fruits_l2523_252382

theorem students_with_both_fruits (apples bananas only_one : ℕ) 
  (h1 : apples = 12)
  (h2 : bananas = 8)
  (h3 : only_one = 10) :
  apples + bananas - only_one = 5 := by
  sorry

end NUMINAMATH_CALUDE_students_with_both_fruits_l2523_252382


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l2523_252359

theorem complementary_angles_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Angles are positive
  a + b = 90 ∧     -- Angles are complementary
  a / b = 5 / 4 →  -- Ratio of angles is 5:4
  a = 50 :=        -- Larger angle is 50°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l2523_252359


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2523_252363

theorem complex_equation_solution (b : ℂ) : (1 + b * Complex.I) * Complex.I = -1 + Complex.I → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2523_252363


namespace NUMINAMATH_CALUDE_exam_results_l2523_252301

/-- Represents the score distribution of students in an examination. -/
structure ScoreDistribution where
  scores : List (Nat × Nat)
  total_students : Nat
  sum_scores : Nat

/-- The given score distribution for the examination. -/
def exam_distribution : ScoreDistribution := {
  scores := [(95, 10), (85, 30), (75, 40), (65, 45), (55, 20), (45, 15)],
  total_students := 160,
  sum_scores := 11200
}

/-- Calculate the average score from a ScoreDistribution. -/
def average_score (d : ScoreDistribution) : Rat :=
  d.sum_scores / d.total_students

/-- Calculate the percentage of students scoring at least 60%. -/
def percentage_passing (d : ScoreDistribution) : Rat :=
  let passing_students := (d.scores.filter (fun p => p.fst ≥ 60)).map (fun p => p.snd) |>.sum
  (passing_students * 100) / d.total_students

theorem exam_results :
  average_score exam_distribution = 70 ∧
  percentage_passing exam_distribution = 78125 / 1000 := by
  sorry

#eval average_score exam_distribution
#eval percentage_passing exam_distribution

end NUMINAMATH_CALUDE_exam_results_l2523_252301


namespace NUMINAMATH_CALUDE_molly_total_swim_distance_l2523_252323

def saturday_distance : ℕ := 45
def sunday_distance : ℕ := 28

theorem molly_total_swim_distance :
  saturday_distance + sunday_distance = 73 :=
by sorry

end NUMINAMATH_CALUDE_molly_total_swim_distance_l2523_252323


namespace NUMINAMATH_CALUDE_comic_book_stacks_theorem_l2523_252304

/-- The number of ways to stack comic books -/
def comic_book_stacks (spiderman : ℕ) (archie : ℕ) (garfield : ℕ) : ℕ :=
  (spiderman.factorial * archie.factorial * garfield.factorial * 2)

/-- Theorem: The number of ways to stack 7 Spiderman, 5 Archie, and 4 Garfield comic books,
    with Archie books on top and each series stacked together, is 29,030,400 -/
theorem comic_book_stacks_theorem :
  comic_book_stacks 7 5 4 = 29030400 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacks_theorem_l2523_252304


namespace NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l2523_252369

/-- Definition of the complex number z as a function of m -/
def z (m : ℝ) : ℂ := Complex.mk (2*m^2 - 7*m + 6) (m^2 - m - 2)

/-- z is purely imaginary iff m = 3/2 -/
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * (z m).im ↔ m = 3/2 := by
  sorry

/-- z is in the fourth quadrant iff -1 < m < 3/2 -/
theorem z_in_fourth_quadrant (m : ℝ) : 
  (z m).re > 0 ∧ (z m).im < 0 ↔ -1 < m ∧ m < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l2523_252369


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_ellipse_equation_l2523_252310

/-- Definition of the ellipse based on the sum of distances from two foci -/
def is_on_ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - 0)^2 + (y + Real.sqrt 3)^2) +
  Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2) = 4

/-- Definition of the line y = kx + √3 -/
def is_on_line (k x y : ℝ) : Prop := y = k * x + Real.sqrt 3

/-- Definition of a point being on the circle with diameter AB passing through origin -/
def is_on_circle (xA yA xB yB x y : ℝ) : Prop :=
  x * (xA + xB) + y * (yA + yB) = xA * xB + yA * yB

theorem ellipse_and_line_intersection :
  ∃ (k : ℝ),
    (∃ (xA yA xB yB : ℝ),
      is_on_ellipse xA yA ∧ is_on_ellipse xB yB ∧
      is_on_line k xA yA ∧ is_on_line k xB yB ∧
      is_on_circle xA yA xB yB 0 0) ∧
    k = Real.sqrt 11 / 2 ∨ k = -Real.sqrt 11 / 2 := by sorry

theorem ellipse_equation :
  ∀ (x y : ℝ), is_on_ellipse x y ↔ x^2 + y^2 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_ellipse_equation_l2523_252310


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l2523_252392

/-- Represents the number of ways to arrange books of two subjects. -/
def arrange_books (total : ℕ) (subject1 : ℕ) (subject2 : ℕ) : ℕ :=
  2 * 2 * 2

/-- Theorem stating that arranging 4 books (2 Chinese and 2 math) 
    such that books of the same subject are not adjacent 
    results in 8 possible arrangements. -/
theorem book_arrangement_theorem :
  arrange_books 4 2 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l2523_252392


namespace NUMINAMATH_CALUDE_three_heads_in_ten_flips_l2523_252309

/-- The probability of flipping exactly k heads in n flips of an unfair coin -/
def unfair_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The main theorem: probability of 3 heads in 10 flips of a coin with 1/3 probability of heads -/
theorem three_heads_in_ten_flips :
  unfair_coin_probability 10 3 (1/3) = 15360 / 59049 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_ten_flips_l2523_252309


namespace NUMINAMATH_CALUDE_puppies_per_cage_l2523_252353

/-- Given a pet store scenario with initial puppies, bought puppies, and cages used,
    calculate the number of puppies per cage. -/
theorem puppies_per_cage
  (initial_puppies : ℝ)
  (bought_puppies : ℝ)
  (cages_used : ℝ)
  (h1 : initial_puppies = 18.0)
  (h2 : bought_puppies = 3.0)
  (h3 : cages_used = 4.2) :
  (initial_puppies + bought_puppies) / cages_used = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l2523_252353


namespace NUMINAMATH_CALUDE_students_liking_food_l2523_252342

theorem students_liking_food (total : ℕ) (dislike : ℕ) (like : ℕ) : 
  total = 814 → dislike = 431 → like = total - dislike → like = 383 := by
sorry

end NUMINAMATH_CALUDE_students_liking_food_l2523_252342


namespace NUMINAMATH_CALUDE_tv_show_watch_time_l2523_252375

/-- Calculates the total watch time for a TV show with regular seasons and a final season -/
def total_watch_time (regular_seasons : ℕ) (episodes_per_regular_season : ℕ) 
  (extra_episodes_final_season : ℕ) (hours_per_episode : ℚ) : ℚ :=
  let total_episodes := regular_seasons * episodes_per_regular_season + 
    (episodes_per_regular_season + extra_episodes_final_season)
  total_episodes * hours_per_episode

/-- Theorem stating that the total watch time for the given TV show is 112 hours -/
theorem tv_show_watch_time : 
  total_watch_time 9 22 4 (1/2) = 112 := by sorry

end NUMINAMATH_CALUDE_tv_show_watch_time_l2523_252375


namespace NUMINAMATH_CALUDE_days_off_per_month_l2523_252383

def total_holidays : ℕ := 36
def months_in_year : ℕ := 12

theorem days_off_per_month :
  total_holidays / months_in_year = 3 := by sorry

end NUMINAMATH_CALUDE_days_off_per_month_l2523_252383


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2523_252380

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2523_252380


namespace NUMINAMATH_CALUDE_unique_B_for_divisible_by_7_l2523_252319

def is_divisible_by_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def four_digit_number (B : ℕ) : ℕ := 4000 + 100 * B + 10 * B + 3

theorem unique_B_for_divisible_by_7 :
  ∀ B : ℕ, B < 10 →
    is_divisible_by_7 (four_digit_number B) →
    B = 0 := by sorry

end NUMINAMATH_CALUDE_unique_B_for_divisible_by_7_l2523_252319


namespace NUMINAMATH_CALUDE_pudding_cups_problem_l2523_252340

theorem pudding_cups_problem (students : ℕ) (additional_cups : ℕ) 
  (h1 : students = 218) 
  (h2 : additional_cups = 121) : 
  ∃ initial_cups : ℕ, 
    initial_cups + additional_cups = students ∧ 
    initial_cups = 97 := by
  sorry

end NUMINAMATH_CALUDE_pudding_cups_problem_l2523_252340


namespace NUMINAMATH_CALUDE_park_length_l2523_252373

theorem park_length (perimeter breadth : ℝ) (h1 : perimeter = 1000) (h2 : breadth = 200) :
  let length := (perimeter - 2 * breadth) / 2
  length = 300 :=
by
  sorry

#check park_length

end NUMINAMATH_CALUDE_park_length_l2523_252373


namespace NUMINAMATH_CALUDE_least_n_satisfying_conditions_l2523_252390

theorem least_n_satisfying_conditions : ∃ n : ℕ,
  n > 1 ∧
  2*n % 3 = 2 ∧
  3*n % 4 = 3 ∧
  4*n % 5 = 4 ∧
  5*n % 6 = 5 ∧
  (∀ m : ℕ, m > 1 ∧ 
    2*m % 3 = 2 ∧
    3*m % 4 = 3 ∧
    4*m % 5 = 4 ∧
    5*m % 6 = 5 → m ≥ n) ∧
  n = 61 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_conditions_l2523_252390


namespace NUMINAMATH_CALUDE_cubic_roots_inequality_l2523_252393

theorem cubic_roots_inequality (a b c : ℝ) : 
  (∃ x y z : ℝ, ∀ t : ℝ, t^3 + a*t^2 + b*t + c = (t - x) * (t - y) * (t - z)) → 
  3*b ≤ a^2 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_inequality_l2523_252393


namespace NUMINAMATH_CALUDE_equal_vector_sums_implies_equilateral_or_equal_l2523_252356

-- Define the circle and points
def Circle := {p : ℂ | ∃ r : ℝ, r > 0 ∧ Complex.abs p = r}

-- Define the property of equal vector sums
def EqualVectorSums (A B C : ℂ) : Prop :=
  Complex.abs (A + B) = Complex.abs (B + C) ∧ 
  Complex.abs (B + C) = Complex.abs (C + A)

-- Define an equilateral triangle
def IsEquilateralTriangle (A B C : ℂ) : Prop :=
  Complex.abs (A - B) = Complex.abs (B - C) ∧
  Complex.abs (B - C) = Complex.abs (C - A)

-- State the theorem
theorem equal_vector_sums_implies_equilateral_or_equal 
  (A B C : ℂ) (hA : A ∈ Circle) (hB : B ∈ Circle) (hC : C ∈ Circle) 
  (hEqual : EqualVectorSums A B C) :
  A = B ∧ B = C ∨ IsEquilateralTriangle A B C := by
  sorry

end NUMINAMATH_CALUDE_equal_vector_sums_implies_equilateral_or_equal_l2523_252356


namespace NUMINAMATH_CALUDE_find_f_one_l2523_252365

/-- A function with the property f(x + y) = f(x) + f(y) + 7xy + 4 -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4

theorem find_f_one (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 + f 5 = 125) :
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_f_one_l2523_252365


namespace NUMINAMATH_CALUDE_nesbitts_inequality_l2523_252374

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_nesbitts_inequality_l2523_252374


namespace NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l2523_252315

theorem fifty_billion_scientific_notation :
  (50000000000 : ℝ) = 5.0 * (10 : ℝ) ^ 9 := by sorry

end NUMINAMATH_CALUDE_fifty_billion_scientific_notation_l2523_252315


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l2523_252344

def initial_amount : ℚ := 360
def sandwich_proportion : ℚ := 3/10
def museum_proportion : ℚ := 1/4
def book_proportion : ℚ := 35/100
def charity_proportion : ℚ := 1/8

theorem jennifer_remaining_money :
  let sandwich_cost := initial_amount * sandwich_proportion
  let museum_cost := initial_amount * museum_proportion
  let book_cost := initial_amount * book_proportion
  let total_spent := sandwich_cost + museum_cost + book_cost
  let remaining_before_charity := initial_amount - total_spent
  let charity_donation := remaining_before_charity * charity_proportion
  let final_remaining := remaining_before_charity - charity_donation
  final_remaining = 63/2 := by
sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l2523_252344


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l2523_252354

/-- Given (√x + 2/√x)^n, where the binomial coefficients of the second, third, and fourth terms 
    in its expansion form an arithmetic sequence, prove that n = 7 and the expansion does not 
    contain a constant term. -/
theorem binomial_expansion_property (x : ℝ) (n : ℕ) 
  (h : (Nat.choose n 2) * 2 = (Nat.choose n 1) + (Nat.choose n 3)) : 
  (n = 7) ∧ (∀ k : ℕ, 2 * k ≠ n) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_property_l2523_252354


namespace NUMINAMATH_CALUDE_calculation_proof_l2523_252362

theorem calculation_proof : 15 * 30 + 45 * 15 - 15 * 10 = 975 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2523_252362


namespace NUMINAMATH_CALUDE_expression_values_l2523_252334

theorem expression_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : (x + y) / z = (y + z) / x) (h2 : (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 := by
sorry

end NUMINAMATH_CALUDE_expression_values_l2523_252334


namespace NUMINAMATH_CALUDE_sleep_difference_l2523_252341

def sleep_pattern (x : ℝ) : Prop :=
  let first_night := 6
  let second_night := x
  let third_night := x / 2
  let fourth_night := 3 * (x / 2)
  first_night + second_night + third_night + fourth_night = 30

theorem sleep_difference : ∃ x : ℝ, sleep_pattern x ∧ x - 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sleep_difference_l2523_252341


namespace NUMINAMATH_CALUDE_max_value_condition_l2523_252333

/-- The function f(x) = kx^2 + kx + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + k * x + 1

/-- The maximum value of f(x) on the interval [-2, 2] is 4 -/
def has_max_4 (k : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) 2, f k x ≤ 4 ∧ ∃ y ∈ Set.Icc (-2) 2, f k y = 4

/-- The theorem stating that k = 1/2 or k = -12 if and only if
    the maximum value of f(x) on [-2, 2] is 4 -/
theorem max_value_condition (k : ℝ) :
  has_max_4 k ↔ k = 1/2 ∨ k = -12 := by sorry

end NUMINAMATH_CALUDE_max_value_condition_l2523_252333


namespace NUMINAMATH_CALUDE_sin_negative_ninety_degrees_l2523_252305

theorem sin_negative_ninety_degrees :
  Real.sin (- π / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_ninety_degrees_l2523_252305


namespace NUMINAMATH_CALUDE_hannah_total_spending_l2523_252351

def hannah_fair_spending (initial_amount : ℝ) (ride_percent : ℝ) (game_percent : ℝ)
  (dessert_cost : ℝ) (cotton_candy_cost : ℝ) (hotdog_cost : ℝ) (keychain_cost : ℝ) : ℝ :=
  (initial_amount * ride_percent) + (initial_amount * game_percent) +
  dessert_cost + cotton_candy_cost + hotdog_cost + keychain_cost

theorem hannah_total_spending :
  hannah_fair_spending 80 0.35 0.25 7 4 5 6 = 70 := by
  sorry

end NUMINAMATH_CALUDE_hannah_total_spending_l2523_252351


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l2523_252336

/-- The number of oranges harvested per day (not discarded) -/
def oranges_harvested (sacks_per_day : ℕ) (sacks_discarded : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  (sacks_per_day - sacks_discarded) * oranges_per_sack

theorem orange_harvest_theorem :
  oranges_harvested 76 64 50 = 600 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l2523_252336


namespace NUMINAMATH_CALUDE_team_a_more_uniform_l2523_252322

/-- Represents a dance team -/
structure DanceTeam where
  name : String
  variance : ℝ

/-- Compares the uniformity of heights between two dance teams -/
def more_uniform_heights (team1 team2 : DanceTeam) : Prop :=
  team1.variance < team2.variance

/-- The problem statement -/
theorem team_a_more_uniform : 
  let team_a : DanceTeam := ⟨"A", 1.5⟩
  let team_b : DanceTeam := ⟨"B", 2.4⟩
  more_uniform_heights team_a team_b := by
  sorry

end NUMINAMATH_CALUDE_team_a_more_uniform_l2523_252322


namespace NUMINAMATH_CALUDE_sum_areas_tangent_circles_l2523_252300

/-- Three mutually externally tangent circles whose centers form a 5-12-13 right triangle -/
structure TangentCircles where
  /-- Radius of the circle centered at the vertex opposite the side of length 5 -/
  a : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 12 -/
  b : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 13 -/
  c : ℝ
  /-- The circles are mutually externally tangent -/
  tangent_5 : a + b = 5
  tangent_12 : a + c = 12
  tangent_13 : b + c = 13

/-- The sum of the areas of three mutually externally tangent circles 
    whose centers form a 5-12-13 right triangle is 113π -/
theorem sum_areas_tangent_circles (circles : TangentCircles) :
  π * (circles.a^2 + circles.b^2 + circles.c^2) = 113 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_areas_tangent_circles_l2523_252300


namespace NUMINAMATH_CALUDE_pet_snake_cost_l2523_252306

def initial_amount : ℕ := 73
def amount_left : ℕ := 18

theorem pet_snake_cost : initial_amount - amount_left = 55 := by sorry

end NUMINAMATH_CALUDE_pet_snake_cost_l2523_252306


namespace NUMINAMATH_CALUDE_flowers_picked_l2523_252335

/-- Proves that if a person can make 7 bouquets with 8 flowers each after 10 flowers have wilted,
    then they initially picked 66 flowers. -/
theorem flowers_picked (bouquets : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) :
  bouquets = 7 →
  flowers_per_bouquet = 8 →
  wilted_flowers = 10 →
  bouquets * flowers_per_bouquet + wilted_flowers = 66 :=
by sorry

end NUMINAMATH_CALUDE_flowers_picked_l2523_252335


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l2523_252394

theorem final_sum_after_transformation (x y S : ℝ) : 
  x + y = S → 3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l2523_252394


namespace NUMINAMATH_CALUDE_largest_fraction_l2523_252303

theorem largest_fraction : 
  (151 : ℚ) / 301 > 3 / 7 ∧
  (151 : ℚ) / 301 > 4 / 9 ∧
  (151 : ℚ) / 301 > 17 / 35 ∧
  (151 : ℚ) / 301 > 100 / 201 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l2523_252303


namespace NUMINAMATH_CALUDE_crackers_per_friend_l2523_252345

/-- Given that Matthew had 23 crackers initially, has 11 crackers left, and gave equal numbers of crackers to 2 friends, prove that each friend ate 6 crackers. -/
theorem crackers_per_friend (initial_crackers : ℕ) (remaining_crackers : ℕ) (num_friends : ℕ) :
  initial_crackers = 23 →
  remaining_crackers = 11 →
  num_friends = 2 →
  (initial_crackers - remaining_crackers) / num_friends = 6 :=
by sorry

end NUMINAMATH_CALUDE_crackers_per_friend_l2523_252345


namespace NUMINAMATH_CALUDE_contrapositive_not_true_l2523_252343

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b ∨ b = k • a

/-- Two vectors have the same direction if they are positive scalar multiples of each other -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ (a = k • b ∨ b = k • a)

/-- The original proposition -/
def original_proposition : Prop :=
  ∀ a b : ℝ × ℝ, collinear a b → same_direction a b

/-- The contrapositive of the original proposition -/
def contrapositive : Prop :=
  ∀ a b : ℝ × ℝ, ¬ same_direction a b → ¬ collinear a b

theorem contrapositive_not_true : ¬ contrapositive := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_not_true_l2523_252343


namespace NUMINAMATH_CALUDE_probability_is_one_l2523_252366

def card_set : Finset ℕ := {1, 3, 4, 6, 7, 9}

def probability_less_than_or_equal_to_9 : ℚ :=
  (card_set.filter (λ x => x ≤ 9)).card / card_set.card

theorem probability_is_one :
  probability_less_than_or_equal_to_9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_l2523_252366


namespace NUMINAMATH_CALUDE_two_digit_sum_product_equality_l2523_252367

/-- P(n) is the product of the digits of n -/
def P (n : ℕ) : ℕ := sorry

/-- S(n) is the sum of the digits of n -/
def S (n : ℕ) : ℕ := sorry

/-- A two-digit number can be represented as 10a + b where a ≠ 0 -/
def isTwoDigit (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 10 * a + b

theorem two_digit_sum_product_equality :
  ∀ n : ℕ, isTwoDigit n → (n = P n + S n ↔ ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 9) :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_product_equality_l2523_252367


namespace NUMINAMATH_CALUDE_flour_needed_for_butter_l2523_252324

/-- Given a recipe with a ratio of butter to flour, calculate the amount of flour needed for a given amount of butter -/
theorem flour_needed_for_butter 
  (original_butter : ℚ) 
  (original_flour : ℚ) 
  (used_butter : ℚ) 
  (h1 : original_butter > 0) 
  (h2 : original_flour > 0) 
  (h3 : used_butter > 0) : 
  (used_butter / original_butter) * original_flour = 30 := by
  sorry

#check flour_needed_for_butter 2 5 12

end NUMINAMATH_CALUDE_flour_needed_for_butter_l2523_252324


namespace NUMINAMATH_CALUDE_average_marks_first_class_l2523_252337

theorem average_marks_first_class 
  (students_first_class : ℕ) 
  (students_second_class : ℕ)
  (average_second_class : ℝ)
  (average_all : ℝ) :
  students_first_class = 35 →
  students_second_class = 55 →
  average_second_class = 65 →
  average_all = 57.22222222222222 →
  (students_first_class * (average_all * (students_first_class + students_second_class) - 
   students_second_class * average_second_class)) / 
   (students_first_class * students_first_class) = 45 := by
sorry

end NUMINAMATH_CALUDE_average_marks_first_class_l2523_252337


namespace NUMINAMATH_CALUDE_concert_tickets_l2523_252327

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem concert_tickets : choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_concert_tickets_l2523_252327


namespace NUMINAMATH_CALUDE_missy_patient_count_l2523_252320

/-- Represents the total number of patients Missy is attending to -/
def total_patients : ℕ := 12

/-- Represents the time (in minutes) it takes to serve all patients -/
def total_serving_time : ℕ := 64

/-- Represents the time (in minutes) to serve a standard care patient -/
def standard_serving_time : ℕ := 5

/-- Represents the fraction of patients with special dietary requirements -/
def special_diet_fraction : ℚ := 1 / 3

/-- Represents the increase in serving time for special dietary patients -/
def special_diet_time_increase : ℚ := 1 / 5

theorem missy_patient_count :
  total_patients = 12 ∧
  (special_diet_fraction * total_patients : ℚ) * 
    (standard_serving_time : ℚ) * (1 + special_diet_time_increase) +
  ((1 - special_diet_fraction) * total_patients : ℚ) * 
    (standard_serving_time : ℚ) = total_serving_time := by
  sorry

end NUMINAMATH_CALUDE_missy_patient_count_l2523_252320


namespace NUMINAMATH_CALUDE_prime_square_mod_504_l2523_252311

theorem prime_square_mod_504 (p : Nat) (h_prime : Nat.Prime p) (h_gt_7 : p > 7) :
  ∃! (s : Finset Nat), 
    (∀ r ∈ s, r < 504 ∧ ∃ q : Nat, p^2 = 504 * q + r) ∧ 
    s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_504_l2523_252311


namespace NUMINAMATH_CALUDE_even_games_player_exists_l2523_252361

/-- Represents a player in the chess tournament -/
structure Player where
  id : Nat
  gamesPlayed : Nat

/-- Represents the state of a round-robin chess tournament -/
structure ChessTournament where
  players : Finset Player
  numPlayers : Nat
  h_numPlayers : numPlayers = 17

/-- The main theorem to prove -/
theorem even_games_player_exists (tournament : ChessTournament) :
  ∃ p ∈ tournament.players, Even p.gamesPlayed :=
sorry

end NUMINAMATH_CALUDE_even_games_player_exists_l2523_252361


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2523_252314

-- Define set A
def A : Set ℝ := Set.univ

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 - 2*x + 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Iic 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2523_252314


namespace NUMINAMATH_CALUDE_isabel_homework_completion_l2523_252316

/-- Given that Isabel had 72.0 homework problems in total, each problem has 5 sub tasks,
    and she has to solve 200 sub tasks, prove that she finished 40 homework problems. -/
theorem isabel_homework_completion (total : ℝ) (subtasks_per_problem : ℕ) (subtasks_solved : ℕ) 
    (h1 : total = 72.0)
    (h2 : subtasks_per_problem = 5)
    (h3 : subtasks_solved = 200) :
    (subtasks_solved : ℝ) / subtasks_per_problem = 40 := by
  sorry

#check isabel_homework_completion

end NUMINAMATH_CALUDE_isabel_homework_completion_l2523_252316


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2523_252364

theorem quadratic_roots_condition (p q : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + q = 0 ∧ 
    x₂^2 + p*x₂ + q = 0 ∧
    x₁ = 2*p ∧ 
    x₂ = p + q) →
  p = 2/3 ∧ q = -8/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2523_252364


namespace NUMINAMATH_CALUDE_cousin_distribution_l2523_252398

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers --/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 cousins and 5 rooms --/
def cousins : ℕ := 5
def rooms : ℕ := 5

/-- The main theorem: there are 52 ways to distribute the cousins into the rooms --/
theorem cousin_distribution : distribute cousins rooms = 52 := by sorry

end NUMINAMATH_CALUDE_cousin_distribution_l2523_252398


namespace NUMINAMATH_CALUDE_complex_division_1_complex_division_2_l2523_252328

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem for the first calculation
theorem complex_division_1 : (1 - i) * (1 + 2*i) / (1 + i) = 2 - i := by sorry

-- Theorem for the second calculation
theorem complex_division_2 : ((1 + 2*i)^2 + 3*(1 - i)) / (2 + i) = 3 - 6/5 * i := by sorry

end NUMINAMATH_CALUDE_complex_division_1_complex_division_2_l2523_252328


namespace NUMINAMATH_CALUDE_coin_difference_is_nine_l2523_252396

def coin_denominations : List Nat := [5, 10, 25, 50]

def amount_to_pay : Nat := 55

def min_coins (denominations : List Nat) (amount : Nat) : Nat :=
  sorry

def max_coins (denominations : List Nat) (amount : Nat) : Nat :=
  sorry

theorem coin_difference_is_nine :
  max_coins coin_denominations amount_to_pay - min_coins coin_denominations amount_to_pay = 9 :=
by sorry

end NUMINAMATH_CALUDE_coin_difference_is_nine_l2523_252396


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2523_252331

theorem quadratic_roots_properties : ∃ (a b : ℝ), 
  (a^2 + a - 2023 = 0) ∧ 
  (b^2 + b - 2023 = 0) ∧ 
  (a * b = -2023) ∧ 
  (a^2 - b = 2024) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2523_252331


namespace NUMINAMATH_CALUDE_probability_point_closer_to_center_l2523_252312

theorem probability_point_closer_to_center (R : Real) (r : Real) : 
  R = 3 → r = 1.5 → (π * r^2) / (π * R^2) = 1/4 := by sorry

end NUMINAMATH_CALUDE_probability_point_closer_to_center_l2523_252312


namespace NUMINAMATH_CALUDE_triangle_max_area_l2523_252379

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(B) + b * cos(A) = √3 and the area of its circumcircle is π,
    then the maximum area of triangle ABC is 3√3/4. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.cos B + b * Real.cos A = Real.sqrt 3 →
  (π * (a / (2 * Real.sin A))^2) = π →
  ∃ (S : ℝ), S = (1/2) * a * b * Real.sin C ∧
              S ≤ (3 * Real.sqrt 3) / 4 ∧
              (∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin C → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2523_252379


namespace NUMINAMATH_CALUDE_total_spent_candy_and_chocolate_l2523_252378

/-- The total amount spent on a candy bar and chocolate -/
def total_spent (candy_bar_cost chocolate_cost : ℕ) : ℕ :=
  candy_bar_cost + chocolate_cost

/-- Theorem: The total amount spent on a candy bar costing $7 and chocolate costing $6 is $13 -/
theorem total_spent_candy_and_chocolate :
  total_spent 7 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_candy_and_chocolate_l2523_252378


namespace NUMINAMATH_CALUDE_contrapositive_example_l2523_252371

theorem contrapositive_example (a b : ℝ) : 
  (∀ a b, a = 0 → a * b = 0) ↔ (∀ a b, a * b ≠ 0 → a ≠ 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l2523_252371


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l2523_252355

/-- The number of herbs available for the wizard's elixir. -/
def num_herbs : ℕ := 4

/-- The number of crystals available for the wizard's elixir. -/
def num_crystals : ℕ := 6

/-- The number of incompatible combinations due to the first problematic crystal. -/
def incompatible_combinations_1 : ℕ := 2

/-- The number of incompatible combinations due to the second problematic crystal. -/
def incompatible_combinations_2 : ℕ := 1

/-- The total number of viable combinations for the wizard's elixir. -/
def viable_combinations : ℕ := num_herbs * num_crystals - (incompatible_combinations_1 + incompatible_combinations_2)

theorem wizard_elixir_combinations :
  viable_combinations = 21 :=
sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l2523_252355


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_positive_square_plus_x_l2523_252332

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) := by sorry

theorem negation_of_positive_square_plus_x :
  (¬ ∀ x > 0, x^2 + x > 0) ↔ (∃ x > 0, x^2 + x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_positive_square_plus_x_l2523_252332


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2523_252339

def C : Set Nat := {66, 68, 71, 73, 75}

theorem smallest_prime_factor_in_C : 
  ∃ (n : Nat), n ∈ C ∧ (∀ m ∈ C, ∀ p q : Nat, Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q) ∧ n = 66 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2523_252339


namespace NUMINAMATH_CALUDE_function_inequality_l2523_252360

theorem function_inequality (a b : ℝ) (f g : ℝ → ℝ) 
  (h₁ : a ≤ b)
  (h₂ : DifferentiableOn ℝ f (Set.Icc a b))
  (h₃ : DifferentiableOn ℝ g (Set.Icc a b))
  (h₄ : ∀ x ∈ Set.Icc a b, deriv f x > deriv g x)
  (h₅ : f a = g a) :
  ∀ x ∈ Set.Icc a b, f x ≥ g x :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l2523_252360


namespace NUMINAMATH_CALUDE_parallel_lines_x_value_l2523_252308

/-- Two points in ℝ² -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in ℝ² defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Check if a line is vertical -/
def isVertical (l : Line) : Prop :=
  l.p1.x = l.p2.x

/-- Two lines are parallel if they are both vertical or have the same slope -/
def areParallel (l1 l2 : Line) : Prop :=
  (isVertical l1 ∧ isVertical l2) ∨
  (¬isVertical l1 ∧ ¬isVertical l2 ∧
    (l1.p2.y - l1.p1.y) / (l1.p2.x - l1.p1.x) = (l2.p2.y - l2.p1.y) / (l2.p2.x - l2.p1.x))

theorem parallel_lines_x_value (x : ℝ) :
  let l1 : Line := { p1 := { x := -1, y := -2 }, p2 := { x := -1, y := 4 } }
  let l2 : Line := { p1 := { x := 2, y := 1 }, p2 := { x := x, y := 6 } }
  areParallel l1 l2 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_x_value_l2523_252308


namespace NUMINAMATH_CALUDE_petes_number_l2523_252372

theorem petes_number : ∃ x : ℚ, 5 * (3 * x + 15) = 200 ∧ x = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l2523_252372


namespace NUMINAMATH_CALUDE_number_equation_solution_l2523_252391

theorem number_equation_solution : ∃ x : ℝ, (0.68 * x - 5) / 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2523_252391


namespace NUMINAMATH_CALUDE_matrix_power_result_l2523_252346

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec ![3, -1] = ![6, -2]) : 
  (B ^ 3).mulVec ![3, -1] = ![24, -8] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_result_l2523_252346


namespace NUMINAMATH_CALUDE_new_group_average_age_l2523_252307

theorem new_group_average_age 
  (initial_count : ℕ) 
  (initial_avg : ℚ) 
  (new_count : ℕ) 
  (final_avg : ℚ) :
  initial_count = 20 →
  initial_avg = 16 →
  new_count = 20 →
  final_avg = 15.5 →
  (initial_count * initial_avg + new_count * (initial_count * final_avg - initial_count * initial_avg) / new_count) / (initial_count + new_count) = 15 :=
by sorry

end NUMINAMATH_CALUDE_new_group_average_age_l2523_252307


namespace NUMINAMATH_CALUDE_plot_length_is_75_l2523_252321

/-- The length of a rectangular plot in meters -/
def length : ℝ := 75

/-- The breadth of a rectangular plot in meters -/
def breadth : ℝ := length - 50

/-- The cost of fencing per meter in rupees -/
def cost_per_meter : ℝ := 26.50

/-- The total cost of fencing in rupees -/
def total_cost : ℝ := 5300

theorem plot_length_is_75 :
  (2 * length + 2 * breadth) * cost_per_meter = total_cost ∧
  length = breadth + 50 ∧
  length = 75 := by sorry

end NUMINAMATH_CALUDE_plot_length_is_75_l2523_252321


namespace NUMINAMATH_CALUDE_vector_problem_l2523_252338

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

theorem vector_problem :
  (∃ k : ℝ, k * a.1 + 2 * b.1 = 14 * (k * a.2 + 2 * b.2) / (-4) ∧ k = -1) ∧
  (∃ c : ℝ × ℝ, (c.1^2 + c.2^2 = 1) ∧
    ((c.1 + 3)^2 + (c.2 - 2)^2 = 20) ∧
    ((c = (5/13, -12/13)) ∨ (c = (1, 0)))) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l2523_252338


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l2523_252388

theorem product_of_sums_equals_difference_of_powers : 
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * 
  (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) = 7^128 - 5^128 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l2523_252388


namespace NUMINAMATH_CALUDE_water_purifier_theorem_l2523_252317

/-- Represents a water purifier type -/
inductive PurifierType
| A
| B

/-- Represents the costs and prices of water purifiers -/
structure PurifierInfo where
  cost_A : ℝ
  cost_B : ℝ
  price_A : ℝ
  price_B : ℝ
  filter_cost_A : ℝ
  filter_cost_B : ℝ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  num_A : ℕ
  num_B : ℕ

/-- The main theorem about water purifier costs and purchasing plans -/
theorem water_purifier_theorem 
  (info : PurifierInfo)
  (h1 : info.cost_B = info.cost_A + 600)
  (h2 : 36000 / info.cost_A = 2 * (27000 / info.cost_B))
  (h3 : info.price_A = 1350)
  (h4 : info.price_B = 2100)
  (h5 : info.filter_cost_A = 400)
  (h6 : info.filter_cost_B = 500) :
  info.cost_A = 1200 ∧ 
  info.cost_B = 1800 ∧
  (∃ (plans : List PurchasePlan), 
    (∀ p ∈ plans, 
      p.num_A * info.cost_A + p.num_B * info.cost_B ≤ 60000 ∧ 
      p.num_B ≤ 8) ∧
    plans.length = 4) ∧
  (∃ (num_filters_A num_filters_B : ℕ),
    num_filters_A + num_filters_B = 6 ∧
    ∃ (p : PurchasePlan), 
      p.num_A * (info.price_A - info.cost_A) + 
      p.num_B * (info.price_B - info.cost_B) - 
      (num_filters_A * info.filter_cost_A + num_filters_B * info.filter_cost_B) = 5250) :=
by sorry

end NUMINAMATH_CALUDE_water_purifier_theorem_l2523_252317


namespace NUMINAMATH_CALUDE_number_difference_l2523_252399

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21308)
  (a_ends_in_5 : a % 10 = 5)
  (b_derivation : b = 50 + (a - 5) / 10) :
  b - a = 17344 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2523_252399


namespace NUMINAMATH_CALUDE_bell_pepper_cost_l2523_252326

/-- The cost of a single bell pepper given the total cost of ingredients for tacos -/
theorem bell_pepper_cost (taco_shells_cost meat_price_per_pound meat_pounds total_spent bell_pepper_count : ℚ) :
  taco_shells_cost = 5 →
  bell_pepper_count = 4 →
  meat_pounds = 2 →
  meat_price_per_pound = 3 →
  total_spent = 17 →
  (total_spent - (taco_shells_cost + meat_price_per_pound * meat_pounds)) / bell_pepper_count = 3/2 := by
sorry

end NUMINAMATH_CALUDE_bell_pepper_cost_l2523_252326


namespace NUMINAMATH_CALUDE_jessica_has_62_marbles_l2523_252389

-- Define the number of marbles each person has
def dennis_marbles : ℕ := 70
def kurt_marbles : ℕ := dennis_marbles - 45
def laurie_marbles : ℕ := kurt_marbles + 12
def jessica_marbles : ℕ := laurie_marbles + 25

-- Theorem to prove
theorem jessica_has_62_marbles : jessica_marbles = 62 := by
  sorry

end NUMINAMATH_CALUDE_jessica_has_62_marbles_l2523_252389


namespace NUMINAMATH_CALUDE_main_diagonal_equals_anti_diagonal_l2523_252330

/-- Represents a square board with side length 2^n -/
structure Board (n : ℕ) where
  size : ℕ := 2^n
  elements : Fin (size * size) → ℕ

/-- Defines the initial arrangement of numbers on the board -/
def initial_board (n : ℕ) : Board n where
  elements := λ i => i.val + 1

/-- Defines the anti-diagonal of a board -/
def anti_diagonal (b : Board n) : List ℕ :=
  List.range b.size |>.map (λ i => b.elements ⟨i + (b.size - 1 - i) * b.size, sorry⟩)

/-- Represents a transformation on the board -/
def transform (b : Board n) : Board n :=
  sorry

/-- Theorem: After transformations, the main diagonal equals the original anti-diagonal -/
theorem main_diagonal_equals_anti_diagonal (n : ℕ) :
  let final_board := (transform^[n] (initial_board n))
  List.range (2^n) |>.map (λ i => final_board.elements ⟨i + i * (2^n), sorry⟩) =
  anti_diagonal (initial_board n) := by
  sorry

end NUMINAMATH_CALUDE_main_diagonal_equals_anti_diagonal_l2523_252330


namespace NUMINAMATH_CALUDE_subset_complement_iff_m_range_l2523_252368

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 28 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2*x^2 - (5+m)*x + 5 ≤ 0}

-- State the theorem
theorem subset_complement_iff_m_range (m : ℝ) :
  B m ⊆ (univ \ A) ↔ m < -5 - 2*Real.sqrt 10 ∨ m > -5 + 2*Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_subset_complement_iff_m_range_l2523_252368


namespace NUMINAMATH_CALUDE_range_of_y_over_x_for_unit_modulus_complex_l2523_252384

theorem range_of_y_over_x_for_unit_modulus_complex (x y : ℝ) :
  (x - 2)^2 + y^2 = 1 →
  y ≠ 0 →
  ∃ k : ℝ, y = k * x ∧ k ∈ Set.Ioo (-Real.sqrt 3 / 3) 0 ∪ Set.Ioo 0 (Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_for_unit_modulus_complex_l2523_252384


namespace NUMINAMATH_CALUDE_new_quadratic_from_roots_sum_product_l2523_252325

theorem new_quadratic_from_roots_sum_product (a b c : ℝ) (ha : a ≠ 0) :
  let original_eq := fun x => a * x^2 + b * x + c
  let new_eq := fun x => a^2 * x^2 + (a*b - a*c) * x - b*c
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  (∀ x, original_eq x = 0 ↔ x = sum_of_roots ∨ x = product_of_roots) →
  (∀ x, new_eq x = 0 ↔ x = sum_of_roots ∨ x = product_of_roots) :=
by sorry

end NUMINAMATH_CALUDE_new_quadratic_from_roots_sum_product_l2523_252325
