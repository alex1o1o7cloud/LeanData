import Mathlib

namespace NUMINAMATH_CALUDE_other_number_proof_l3235_323590

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 8820)
  (h2 : Nat.gcd a b = 36)
  (h3 : a = 360) :
  b = 882 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l3235_323590


namespace NUMINAMATH_CALUDE_square_plus_divisor_not_perfect_square_plus_divisor_perfect_iff_l3235_323518

def is_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, x = m^2

theorem square_plus_divisor_not_perfect (n d : ℕ) (hn : n > 0) (hd : d > 0) (hdiv : d ∣ 2*n^2) :
  ¬ is_perfect_square (n^2 + d) := by sorry

theorem square_plus_divisor_perfect_iff (n d : ℕ) (hn : n > 0) (hd : d > 0) (hdiv : d ∣ 3*n^2) :
  is_perfect_square (n^2 + d) ↔ d = 3*n^2 := by sorry

end NUMINAMATH_CALUDE_square_plus_divisor_not_perfect_square_plus_divisor_perfect_iff_l3235_323518


namespace NUMINAMATH_CALUDE_quadratic_rational_roots_l3235_323576

theorem quadratic_rational_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a = 1 ∧ b = 2 ∧ c = -3 →
  ∃ (x y : ℚ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rational_roots_l3235_323576


namespace NUMINAMATH_CALUDE_better_performance_against_teamB_l3235_323556

/-- Represents the statistics for a team --/
structure TeamStats :=
  (points : List Nat)
  (rebounds : List Nat)
  (turnovers : List Nat)

/-- Calculate the average of a list of numbers --/
def average (l : List Nat) : Rat :=
  (l.sum : Rat) / l.length

/-- Calculate the comprehensive score for a team --/
def comprehensiveScore (stats : TeamStats) : Rat :=
  average stats.points + 1.2 * average stats.rebounds - average stats.turnovers

/-- Xiao Bin's statistics against Team A --/
def teamA : TeamStats :=
  { points := [21, 29, 24, 26],
    rebounds := [10, 10, 14, 10],
    turnovers := [2, 2, 3, 5] }

/-- Xiao Bin's statistics against Team B --/
def teamB : TeamStats :=
  { points := [25, 31, 16, 22],
    rebounds := [17, 15, 12, 8],
    turnovers := [2, 0, 4, 2] }

/-- Theorem: Xiao Bin's comprehensive score against Team B is higher than against Team A --/
theorem better_performance_against_teamB :
  comprehensiveScore teamB > comprehensiveScore teamA :=
by
  sorry


end NUMINAMATH_CALUDE_better_performance_against_teamB_l3235_323556


namespace NUMINAMATH_CALUDE_base9_432_equals_base10_353_l3235_323565

/-- Converts a base 9 number to base 10 --/
def base9_to_base10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 9^2 + d₁ * 9^1 + d₀ * 9^0

/-- The base 9 number 432₉ is equal to 353 in base 10 --/
theorem base9_432_equals_base10_353 :
  base9_to_base10 4 3 2 = 353 := by sorry

end NUMINAMATH_CALUDE_base9_432_equals_base10_353_l3235_323565


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3235_323523

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3235_323523


namespace NUMINAMATH_CALUDE_product_three_consecutive_divisibility_l3235_323520

theorem product_three_consecutive_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 5 * m) →
  (∃ m : ℤ, n = 6 * m) ∧
  (∃ m : ℤ, n = 10 * m) ∧
  (∃ m : ℤ, n = 15 * m) ∧
  (∃ m : ℤ, n = 30 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 60 * m) :=
by sorry

end NUMINAMATH_CALUDE_product_three_consecutive_divisibility_l3235_323520


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_l3235_323534

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the property of an isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

-- Define the angle measure in degrees
def angleMeasure (t : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem isosceles_triangle_angle (t : Triangle) :
  isIsosceles t →
  angleMeasure t t.B = 55 →
  angleMeasure t t.A = 70 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_angle_l3235_323534


namespace NUMINAMATH_CALUDE_f_strictly_increasing_and_odd_l3235_323535

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem f_strictly_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_and_odd_l3235_323535


namespace NUMINAMATH_CALUDE_existence_of_h_l3235_323584

theorem existence_of_h : ∃ h : ℝ, ∀ n : ℕ, 
  ¬(⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) := by sorry

end NUMINAMATH_CALUDE_existence_of_h_l3235_323584


namespace NUMINAMATH_CALUDE_total_wheat_mass_l3235_323507

def wheat_weights : List Float := [90, 91, 91.5, 89, 91.2, 91.3, 89.7, 88.8, 91.8, 91.1]

theorem total_wheat_mass :
  wheat_weights.sum = 905.4 := by
  sorry

end NUMINAMATH_CALUDE_total_wheat_mass_l3235_323507


namespace NUMINAMATH_CALUDE_sum_property_implies_isosceles_l3235_323598

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : a + b + c = π

-- Define a quadrilateral
structure Quadrilateral where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  angle_sum : w + x + y + z = 2 * π

-- Define the property that for any two angles of the triangle, 
-- there is an angle in the quadrilateral equal to their sum
def has_sum_property (t : Triangle) (q : Quadrilateral) : Prop :=
  ∃ (i j : Fin 3) (k : Fin 4), 
    i ≠ j ∧ 
    match i, j with
    | 0, 1 | 1, 0 => q.w = t.a + t.b ∨ q.x = t.a + t.b ∨ q.y = t.a + t.b ∨ q.z = t.a + t.b
    | 0, 2 | 2, 0 => q.w = t.a + t.c ∨ q.x = t.a + t.c ∨ q.y = t.a + t.c ∨ q.z = t.a + t.c
    | 1, 2 | 2, 1 => q.w = t.b + t.c ∨ q.x = t.b + t.c ∨ q.y = t.b + t.c ∨ q.z = t.b + t.c
    | _, _ => False

-- Define what it means for a triangle to be isosceles
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- The theorem to be proved
theorem sum_property_implies_isosceles (t : Triangle) (q : Quadrilateral) :
  has_sum_property t q → is_isosceles t :=
by sorry

end NUMINAMATH_CALUDE_sum_property_implies_isosceles_l3235_323598


namespace NUMINAMATH_CALUDE_binomial_arithmetic_sequence_implies_seven_l3235_323510

def factorial (r : ℕ) : ℕ := Nat.factorial r

def binomial_coefficient (j k : ℕ) : ℕ :=
  if k ≤ j then
    factorial j / (factorial k * factorial (j - k))
  else
    0

theorem binomial_arithmetic_sequence_implies_seven (n : ℕ) 
  (h1 : n > 3)
  (h2 : ∃ d : ℕ, binomial_coefficient n 2 - binomial_coefficient n 1 = d ∧
                 binomial_coefficient n 3 - binomial_coefficient n 2 = d) :
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_arithmetic_sequence_implies_seven_l3235_323510


namespace NUMINAMATH_CALUDE_movies_watched_l3235_323587

theorem movies_watched (total_movies : ℕ) (movies_left : ℕ) (h1 : total_movies = 8) (h2 : movies_left = 4) :
  total_movies - movies_left = 4 := by
  sorry

end NUMINAMATH_CALUDE_movies_watched_l3235_323587


namespace NUMINAMATH_CALUDE_marathon_distance_l3235_323513

theorem marathon_distance (marathon_miles : ℕ) (marathon_yards : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) :
  marathon_miles = 26 →
  marathon_yards = 395 →
  yards_per_mile = 1760 →
  num_marathons = 15 →
  (num_marathons * marathon_yards) % yards_per_mile = 645 := by
  sorry

#check marathon_distance

end NUMINAMATH_CALUDE_marathon_distance_l3235_323513


namespace NUMINAMATH_CALUDE_average_age_of_women_l3235_323514

/-- The average age of four women given the following conditions:
    - There are 15 men initially.
    - The average age of 15 men is 40 years.
    - Four men of ages 26, 32, 41, and 39 years are replaced by four women.
    - The new average age increases by 2.9 years after the replacement. -/
theorem average_age_of_women (
  initial_men : ℕ)
  (initial_avg_age : ℝ)
  (replaced_men_ages : Fin 4 → ℝ)
  (new_avg_increase : ℝ)
  (h1 : initial_men = 15)
  (h2 : initial_avg_age = 40)
  (h3 : replaced_men_ages = ![26, 32, 41, 39])
  (h4 : new_avg_increase = 2.9)
  : (initial_men * initial_avg_age + 4 * new_avg_increase * initial_men - (replaced_men_ages 0 + replaced_men_ages 1 + replaced_men_ages 2 + replaced_men_ages 3)) / 4 = 45.375 := by
  sorry


end NUMINAMATH_CALUDE_average_age_of_women_l3235_323514


namespace NUMINAMATH_CALUDE_product_equals_four_l3235_323580

theorem product_equals_four (a b c : ℝ) 
  (h : ∀ x y z : ℝ, x * y * z = (Real.sqrt ((x + 2) * (y + 3))) / (z + 1)) : 
  6 * 15 * 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_product_equals_four_l3235_323580


namespace NUMINAMATH_CALUDE_ratio_difference_l3235_323558

theorem ratio_difference (a b : ℕ) (ha : a > 5) (hb : b > 5) : 
  (a : ℚ) / b = 6 / 5 → (a - 5 : ℚ) / (b - 5) = 5 / 4 → a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_difference_l3235_323558


namespace NUMINAMATH_CALUDE_find_N_l3235_323561

theorem find_N : ∃ N : ℝ, (0.2 * N = 0.3 * 2500) ∧ (N = 3750) := by
  sorry

end NUMINAMATH_CALUDE_find_N_l3235_323561


namespace NUMINAMATH_CALUDE_second_class_size_l3235_323570

def students_first_class : ℕ := 25
def avg_marks_first_class : ℚ := 50
def avg_marks_second_class : ℚ := 65
def avg_marks_all : ℚ := 59.23076923076923

theorem second_class_size :
  ∃ (x : ℕ), 
    (students_first_class * avg_marks_first_class + x * avg_marks_second_class) / (students_first_class + x) = avg_marks_all ∧
    x = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_class_size_l3235_323570


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3235_323594

theorem smallest_number_with_given_remainders : ∃ b : ℕ, 
  b % 4 = 2 ∧ b % 3 = 2 ∧ b % 5 = 3 ∧
  ∀ n : ℕ, n < b → (n % 4 ≠ 2 ∨ n % 3 ≠ 2 ∨ n % 5 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3235_323594


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_5_6_l3235_323525

theorem greatest_three_digit_divisible_by_3_5_6 :
  ∀ n : ℕ, n < 1000 → n ≥ 100 → n % 3 = 0 → n % 5 = 0 → n % 6 = 0 → n ≤ 990 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_5_6_l3235_323525


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3235_323589

theorem arithmetic_mean_of_fractions : 
  (5 : ℚ) / 6 = ((9 : ℚ) / 12 + (11 : ℚ) / 12) / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3235_323589


namespace NUMINAMATH_CALUDE_books_pages_after_move_l3235_323536

theorem books_pages_after_move (initial_books : ℕ) (pages_per_book : ℕ) (lost_books : ℕ) : 
  initial_books = 10 → pages_per_book = 100 → lost_books = 2 →
  (initial_books - lost_books) * pages_per_book = 800 := by
  sorry

end NUMINAMATH_CALUDE_books_pages_after_move_l3235_323536


namespace NUMINAMATH_CALUDE_range_of_f_l3235_323519

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*x + 2)

theorem range_of_f :
  Set.range f = Set.Ioo 0 (1/2) ∪ {1/2} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3235_323519


namespace NUMINAMATH_CALUDE_hot_dog_contest_l3235_323533

/-- Hot dog eating contest problem -/
theorem hot_dog_contest (first_competitor second_competitor third_competitor : ℕ) : 
  first_competitor = 12 →
  second_competitor = 2 * first_competitor →
  third_competitor = second_competitor - (second_competitor / 4) →
  third_competitor = 18 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_contest_l3235_323533


namespace NUMINAMATH_CALUDE_vector_operation_l3235_323551

/-- Given vectors a and b in ℝ², prove that 2b - a equals the expected result. -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (3, 2)) (h2 : b = (0, -1)) :
  (2 : ℝ) • b - a = (-3, -4) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3235_323551


namespace NUMINAMATH_CALUDE_infinitely_many_decimals_between_3_3_and_3_6_l3235_323501

theorem infinitely_many_decimals_between_3_3_and_3_6 :
  ∃ f : ℕ → ℝ, (∀ n : ℕ, 3.3 < f n ∧ f n < 3.6) ∧ (∀ n m : ℕ, n ≠ m → f n ≠ f m) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_decimals_between_3_3_and_3_6_l3235_323501


namespace NUMINAMATH_CALUDE_dave_final_tickets_l3235_323577

/-- Calculates the final number of tickets Dave has after a series of events at the arcade. -/
def dave_tickets : ℕ :=
  let initial_tickets := 11
  let candy_bar_cost := 3
  let beanie_cost := 5
  let racing_game_win := 10
  let claw_machine_win := 7
  let after_spending := initial_tickets - candy_bar_cost - beanie_cost
  let after_winning := after_spending + racing_game_win + claw_machine_win
  2 * after_winning

/-- Theorem stating that Dave ends up with 40 tickets after all events at the arcade. -/
theorem dave_final_tickets : dave_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_dave_final_tickets_l3235_323577


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3235_323568

theorem max_value_quadratic (x : ℝ) : 
  ∃ (max : ℝ), max = 9 ∧ ∀ y : ℝ, y = -3 * x^2 + 9 → y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3235_323568


namespace NUMINAMATH_CALUDE_circle_center_l3235_323528

/-- The center of a circle given by the equation x^2 - 8x + y^2 + 4y = -3 -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 8*x + y^2 + 4*y = -3) → (∃ r : ℝ, (x - 4)^2 + (y + 2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l3235_323528


namespace NUMINAMATH_CALUDE_smallest_y_l3235_323512

theorem smallest_y (x y : ℕ+) (h1 : x.val - y.val = 8) 
  (h2 : Nat.gcd ((x.val^3 + y.val^3) / (x.val + y.val)) (x.val * y.val) = 16) :
  ∀ z : ℕ+, z.val < y.val → 
    Nat.gcd ((z.val^3 + (z.val + 8)^3) / (z.val + (z.val + 8))) (z.val * (z.val + 8)) ≠ 16 :=
by sorry

#check smallest_y

end NUMINAMATH_CALUDE_smallest_y_l3235_323512


namespace NUMINAMATH_CALUDE_oranges_from_first_tree_l3235_323552

/-- Represents the number of oranges picked from each tree -/
structure OrangesPicked where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The total number of oranges picked is the sum of oranges from all three trees -/
def total_oranges (op : OrangesPicked) : ℕ := op.first + op.second + op.third

/-- Theorem: Given the total oranges and the number from the second and third trees, 
    we can determine the number of oranges from the first tree -/
theorem oranges_from_first_tree (op : OrangesPicked) 
  (h1 : total_oranges op = 260)
  (h2 : op.second = 60)
  (h3 : op.third = 120) :
  op.first = 80 := by
  sorry

end NUMINAMATH_CALUDE_oranges_from_first_tree_l3235_323552


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3235_323522

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = q * a n) 
  (h_arithmetic : a 3 + a 4 = a 5) :
  q = (Real.sqrt 5 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3235_323522


namespace NUMINAMATH_CALUDE_proposition_and_variants_l3235_323511

theorem proposition_and_variants (x y : ℝ) :
  (∀ x y, xy = 0 → x = 0 ∨ y = 0) ∧
  (∀ x y, x = 0 ∨ y = 0 → xy = 0) ∧
  (∀ x y, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  (∀ x y, x ≠ 0 ∧ y ≠ 0 → xy ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_variants_l3235_323511


namespace NUMINAMATH_CALUDE_total_savings_l3235_323537

/-- Represents the savings of Anne and Katherine -/
structure Savings where
  anne : ℝ
  katherine : ℝ

/-- The conditions of the savings problem -/
def SavingsConditions (s : Savings) : Prop :=
  (s.anne - 150 = (1 / 3) * s.katherine) ∧
  (2 * s.katherine = 3 * s.anne)

/-- Theorem stating that under the given conditions, the total savings is $750 -/
theorem total_savings (s : Savings) (h : SavingsConditions s) : 
  s.anne + s.katherine = 750 := by
  sorry

#check total_savings

end NUMINAMATH_CALUDE_total_savings_l3235_323537


namespace NUMINAMATH_CALUDE_sales_solution_l3235_323515

def sales_problem (sale1 sale2 sale3 sale5 sale6 average : ℕ) : Prop :=
  let total_sales := 6 * average
  let known_sales := sale1 + sale2 + sale3 + sale5 + sale6
  total_sales - known_sales = 5730

theorem sales_solution :
  sales_problem 4000 6524 5689 6000 12557 7000 := by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l3235_323515


namespace NUMINAMATH_CALUDE_binomial_variance_four_half_l3235_323545

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ :=
  ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: The variance of a binomial distribution B(4, 1/2) is 1 -/
theorem binomial_variance_four_half :
  ∀ ξ : BinomialDistribution, ξ.n = 4 ∧ ξ.p = 1/2 → variance ξ = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_four_half_l3235_323545


namespace NUMINAMATH_CALUDE_lynne_book_purchase_total_cost_lynne_spent_75_dollars_l3235_323578

theorem lynne_book_purchase_total_cost : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | cat_books, solar_books, magazines, book_cost, magazine_cost =>
    let total_books := cat_books + solar_books
    let book_total_cost := total_books * book_cost
    let magazine_total_cost := magazines * magazine_cost
    book_total_cost + magazine_total_cost

theorem lynne_spent_75_dollars : 
  lynne_book_purchase_total_cost 7 2 3 7 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_lynne_book_purchase_total_cost_lynne_spent_75_dollars_l3235_323578


namespace NUMINAMATH_CALUDE_norris_game_spending_l3235_323548

/-- The amount of money Norris spent on the online game -/
def money_spent (september_savings october_savings november_savings money_left : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - money_left

/-- Theorem stating that Norris spent $75 on the online game -/
theorem norris_game_spending :
  money_spent 29 25 31 10 = 75 := by
  sorry

end NUMINAMATH_CALUDE_norris_game_spending_l3235_323548


namespace NUMINAMATH_CALUDE_square_ends_with_self_l3235_323562

theorem square_ends_with_self (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) → (A^2 ≡ A [ZMOD 1000]) ↔ (A = 376 ∨ A = 625) := by
  sorry

end NUMINAMATH_CALUDE_square_ends_with_self_l3235_323562


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_is_eight_l3235_323573

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (12 - x^4) + x^2) / x^3 + 4

def X : Set ℝ := {x | x ∈ Set.Icc (-1) 0 ∪ Set.Ioc 0 1}

theorem sum_of_max_and_min_is_eight :
  ∃ (A B : ℝ), (∀ x ∈ X, f x ≤ A) ∧ 
               (∃ x ∈ X, f x = A) ∧ 
               (∀ x ∈ X, B ≤ f x) ∧ 
               (∃ x ∈ X, f x = B) ∧ 
               A + B = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_is_eight_l3235_323573


namespace NUMINAMATH_CALUDE_green_paint_amount_l3235_323569

/-- The amount of green paint needed for a treehouse project. -/
def green_paint (total white brown : ℕ) : ℕ :=
  total - (white + brown)

/-- Theorem stating that the amount of green paint is 15 ounces. -/
theorem green_paint_amount :
  green_paint 69 20 34 = 15 := by
  sorry

end NUMINAMATH_CALUDE_green_paint_amount_l3235_323569


namespace NUMINAMATH_CALUDE_no_solution_implies_a_le_8_l3235_323502

theorem no_solution_implies_a_le_8 (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_le_8_l3235_323502


namespace NUMINAMATH_CALUDE_remainder_17_power_77_mod_7_l3235_323530

theorem remainder_17_power_77_mod_7 : 17^77 % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_power_77_mod_7_l3235_323530


namespace NUMINAMATH_CALUDE_closest_root_l3235_323550

def options : List ℤ := [2, 3, 4, 5]

theorem closest_root (x : ℝ) (h : x^3 - 9 = 16) : 
  3 = (options.argmin (λ y => |y - x|)).get sorry :=
sorry

end NUMINAMATH_CALUDE_closest_root_l3235_323550


namespace NUMINAMATH_CALUDE_distance_on_parametric_line_l3235_323547

/-- The distance between two points on a parametric line --/
theorem distance_on_parametric_line :
  let line : ℝ → ℝ × ℝ := λ t ↦ (1 + 3 * t, 1 + t)
  let point1 := line 0
  let point2 := line 1
  (point1.1 - point2.1)^2 + (point1.2 - point2.2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_on_parametric_line_l3235_323547


namespace NUMINAMATH_CALUDE_remainder_theorem_l3235_323564

theorem remainder_theorem (N : ℤ) (h : N % 35 = 25) : N % 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3235_323564


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3235_323504

def geometric_sequence (a : ℕ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem sixth_term_of_geometric_sequence 
  (a₁ : ℕ) (a₅ : ℕ) (h₁ : a₁ = 3) (h₅ : a₅ = 375) :
  ∃ r : ℝ, 
    geometric_sequence a₁ r 5 = a₅ ∧ 
    geometric_sequence a₁ r 6 = 9375 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3235_323504


namespace NUMINAMATH_CALUDE_fourth_year_afforestation_l3235_323557

/-- Calculates the area afforested in a given year, given the initial area and annual increase rate. -/
def area_afforested (initial_area : ℝ) (annual_increase : ℝ) (year : ℕ) : ℝ :=
  initial_area * (1 + annual_increase) ^ (year - 1)

/-- Theorem stating that given an initial area of 10,000 acres and an annual increase of 20%,
    the area afforested in the fourth year is 17,280 acres. -/
theorem fourth_year_afforestation :
  area_afforested 10000 0.2 4 = 17280 := by
  sorry

end NUMINAMATH_CALUDE_fourth_year_afforestation_l3235_323557


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3235_323596

theorem ellipse_eccentricity (k : ℝ) :
  (∃ x y : ℝ, x^2 / 9 + y^2 / (4 + k) = 1) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c / a = 4/5 ∧
    ((a^2 = 9 ∧ b^2 = 4 + k) ∨ (a^2 = 4 + k ∧ b^2 = 9)) ∧
    c^2 = a^2 - b^2) →
  k = -19/25 ∨ k = 21 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3235_323596


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3235_323572

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≤ a (n + 1)

theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) :
  (geometric_sequence a q) →
  (¬(((a 1 * q > 0) → increasing_sequence a) ∧
     (increasing_sequence a → (a 1 * q > 0)))) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3235_323572


namespace NUMINAMATH_CALUDE_vasya_drove_two_fifths_l3235_323588

/-- Represents the fraction of the total distance driven by each person -/
structure DistanceFractions where
  anton : ℝ
  vasya : ℝ
  sasha : ℝ
  dima : ℝ

/-- Conditions of the driving problem -/
def driving_conditions (d : DistanceFractions) : Prop :=
  d.anton + d.vasya + d.sasha + d.dima = 1 ∧  -- Total distance is 1
  d.anton = d.vasya / 2 ∧                     -- Anton drove half of Vasya's distance
  d.sasha = d.anton + d.dima ∧                -- Sasha drove as long as Anton and Dima combined
  d.dima = 1 / 10                             -- Dima drove one-tenth of the distance

/-- Theorem: Under the given conditions, Vasya drove 2/5 of the total distance -/
theorem vasya_drove_two_fifths (d : DistanceFractions) 
  (h : driving_conditions d) : d.vasya = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vasya_drove_two_fifths_l3235_323588


namespace NUMINAMATH_CALUDE_average_students_count_l3235_323571

theorem average_students_count (total : ℕ) (honor average poor : ℕ)
  (first_yes second_yes third_yes : ℕ) :
  total = 30 →
  total = honor + average + poor →
  first_yes = 19 →
  second_yes = 12 →
  third_yes = 9 →
  first_yes = honor + average / 2 →
  second_yes = average →
  third_yes = poor + average / 2 →
  average = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_students_count_l3235_323571


namespace NUMINAMATH_CALUDE_fraction_comparison_l3235_323529

theorem fraction_comparison : 
  (14 / 10 : ℚ) = 7 / 5 ∧ 
  (1 + 2 / 5 : ℚ) = 7 / 5 ∧ 
  (1 + 4 / 20 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 3 / 15 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 2 / 6 : ℚ) ≠ 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3235_323529


namespace NUMINAMATH_CALUDE_sequence_properties_l3235_323585

/-- Given a sequence {a_n} with partial sum S_n satisfying 3a_n - 2S_n = 2 for all n,
    prove the general term formula and a property of partial sums. -/
theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, 3 * a n - 2 * S n = 2) : 
    (∀ n, a n = 2 * 3^(n-1)) ∧ 
    (∀ n, S (n+1)^2 - S n * S (n+2) = 4 * 3^n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3235_323585


namespace NUMINAMATH_CALUDE_three_cubic_yards_to_cubic_feet_l3235_323586

-- Define the conversion factor
def yard_to_foot : ℝ := 3

-- Define the volume in cubic yards
def cubic_yards : ℝ := 3

-- Theorem to prove
theorem three_cubic_yards_to_cubic_feet :
  cubic_yards * (yard_to_foot ^ 3) = 81 := by
  sorry

end NUMINAMATH_CALUDE_three_cubic_yards_to_cubic_feet_l3235_323586


namespace NUMINAMATH_CALUDE_parrot_response_characterization_l3235_323531

def parrot_calc (n : ℤ) : ℚ :=
  (5 * n + 14) / 6 - 1

theorem parrot_response_characterization :
  ∀ n : ℤ, (∃ k : ℤ, parrot_calc n = k) ↔ ∃ m : ℤ, n = 6 * m + 2 :=
sorry

end NUMINAMATH_CALUDE_parrot_response_characterization_l3235_323531


namespace NUMINAMATH_CALUDE_builder_cost_l3235_323542

/-- The cost of hiring builders to construct houses -/
theorem builder_cost (builders_per_floor : ℕ) (days_per_floor : ℕ) (daily_wage : ℕ)
  (num_builders : ℕ) (num_houses : ℕ) (floors_per_house : ℕ) :
  builders_per_floor = 3 →
  days_per_floor = 30 →
  daily_wage = 100 →
  num_builders = 6 →
  num_houses = 5 →
  floors_per_house = 6 →
  (num_houses * floors_per_house * days_per_floor * daily_wage * num_builders) / builders_per_floor = 270000 :=
by sorry

end NUMINAMATH_CALUDE_builder_cost_l3235_323542


namespace NUMINAMATH_CALUDE_player2_is_best_l3235_323575

structure Player where
  id : Nat
  average_time : ℝ
  variance : ℝ

def players : List Player := [
  { id := 1, average_time := 51, variance := 3.5 },
  { id := 2, average_time := 50, variance := 3.5 },
  { id := 3, average_time := 51, variance := 14.5 },
  { id := 4, average_time := 50, variance := 14.4 }
]

def is_better_performer (p1 p2 : Player) : Prop :=
  p1.average_time < p2.average_time ∨ 
  (p1.average_time = p2.average_time ∧ p1.variance < p2.variance)

theorem player2_is_best : 
  ∀ p ∈ players, p.id ≠ 2 → is_better_performer (players[1]) p :=
sorry

end NUMINAMATH_CALUDE_player2_is_best_l3235_323575


namespace NUMINAMATH_CALUDE_point_transformation_l3235_323508

def rotate90 (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (c d : ℝ) :
  let (x₁, y₁) := rotate90 c d 2 3
  let (x₂, y₂) := reflect_y_eq_x x₁ y₁
  (x₂ = 7 ∧ y₂ = -4) → d - c = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3235_323508


namespace NUMINAMATH_CALUDE_fourth_degree_equation_roots_l3235_323541

theorem fourth_degree_equation_roots :
  ∃ (r₁ r₂ r₃ r₄ : ℂ),
    (∀ x : ℂ, 3 * x^4 + 2 * x^3 - 7 * x^2 + 2 * x + 3 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) :=
by sorry

end NUMINAMATH_CALUDE_fourth_degree_equation_roots_l3235_323541


namespace NUMINAMATH_CALUDE_recipe_batches_for_competition_l3235_323532

/-- Calculates the number of full recipe batches needed for a math competition --/
def recipe_batches_needed (total_students : ℕ) (attendance_drop : ℚ) 
  (cookies_per_student : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  let attending_students := (total_students : ℚ) * (1 - attendance_drop)
  let total_cookies_needed := (attending_students * cookies_per_student : ℚ).ceil
  let batches_needed := (total_cookies_needed / cookies_per_batch : ℚ).ceil
  batches_needed.toNat

/-- Proves that 17 full recipe batches are needed for the math competition --/
theorem recipe_batches_for_competition : 
  recipe_batches_needed 144 (30/100) 3 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_recipe_batches_for_competition_l3235_323532


namespace NUMINAMATH_CALUDE_two_thousand_sixteenth_smallest_n_l3235_323599

/-- The number of ways Yang can reach (n,0) under the given movement rules -/
def a (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number satisfies the condition an ≡ 1 (mod 5) -/
def satisfies_condition (n : ℕ) : Prop :=
  a n % 5 = 1

/-- The function that returns the kth smallest positive integer satisfying the condition -/
def kth_smallest (k : ℕ) : ℕ := sorry

theorem two_thousand_sixteenth_smallest_n :
  kth_smallest 2016 = 475756 :=
sorry

end NUMINAMATH_CALUDE_two_thousand_sixteenth_smallest_n_l3235_323599


namespace NUMINAMATH_CALUDE_binary_equals_octal_l3235_323544

-- Define the binary number
def binary_num : List Bool := [true, true, false, true, false, true]

-- Define the octal number
def octal_num : Nat := 65

-- Function to convert binary to decimal
def binary_to_decimal (bin : List Bool) : Nat :=
  bin.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

-- Function to convert decimal to octal
def decimal_to_octal (n : Nat) : Nat :=
  if n < 8 then n
  else 10 * (decimal_to_octal (n / 8)) + (n % 8)

-- Theorem stating that the binary number is equal to the octal number when converted
theorem binary_equals_octal :
  decimal_to_octal (binary_to_decimal binary_num) = octal_num := by
  sorry


end NUMINAMATH_CALUDE_binary_equals_octal_l3235_323544


namespace NUMINAMATH_CALUDE_arrangements_proof_l3235_323517

def boys : ℕ := 4
def girls : ℕ := 3
def total_people : ℕ := boys + girls
def selected_people : ℕ := 3
def tasks : ℕ := 3

def arrangements_with_at_least_one_girl : ℕ :=
  Nat.choose total_people selected_people * Nat.factorial tasks -
  Nat.choose boys selected_people * Nat.factorial tasks

theorem arrangements_proof : arrangements_with_at_least_one_girl = 186 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_proof_l3235_323517


namespace NUMINAMATH_CALUDE_tangent_circles_m_value_l3235_323538

/-- Two externally tangent circles C₁ and C₂ -/
structure TangentCircles where
  /-- Equation of C₁: (x+2)² + (y-m)² = 9 -/
  c1 : ∀ (x y : ℝ), (x + 2)^2 + (y - m)^2 = 9
  /-- Equation of C₂: (x-m)² + (y+1)² = 4 -/
  c2 : ∀ (x y : ℝ), (x - m)^2 + (y + 1)^2 = 4
  /-- m is a real number -/
  m : ℝ

/-- The value of m for externally tangent circles C₁ and C₂ is either 2 or -5 -/
theorem tangent_circles_m_value (tc : TangentCircles) : tc.m = 2 ∨ tc.m = -5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_m_value_l3235_323538


namespace NUMINAMATH_CALUDE_jessica_expense_increase_l3235_323555

/-- Calculates the increase in Jessica's yearly expenses --/
def yearly_expense_increase (
  last_year_rent : ℕ)
  (last_year_food : ℕ)
  (last_year_insurance : ℕ)
  (rent_increase_percent : ℕ)
  (food_increase_percent : ℕ)
  (insurance_multiplier : ℕ) : ℕ :=
  let new_rent := last_year_rent + last_year_rent * rent_increase_percent / 100
  let new_food := last_year_food + last_year_food * food_increase_percent / 100
  let new_insurance := last_year_insurance * insurance_multiplier
  let last_year_total := last_year_rent + last_year_food + last_year_insurance
  let this_year_total := new_rent + new_food + new_insurance
  (this_year_total - last_year_total) * 12

theorem jessica_expense_increase :
  yearly_expense_increase 1000 200 100 30 50 3 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_jessica_expense_increase_l3235_323555


namespace NUMINAMATH_CALUDE_max_age_on_aubrey_eighth_birthday_l3235_323505

/-- Proves that Max's age on Aubrey's 8th birthday is 6 years -/
theorem max_age_on_aubrey_eighth_birthday 
  (max_birth : ℕ) -- Max's birth year
  (luka_birth : ℕ) -- Luka's birth year
  (aubrey_birth : ℕ) -- Aubrey's birth year
  (h1 : max_birth = luka_birth + 4) -- Max born when Luka turned 4
  (h2 : luka_birth = aubrey_birth - 2) -- Luka is 2 years older than Aubrey
  (h3 : aubrey_birth + 8 = max_birth + 6) -- Aubrey's 8th birthday is when Max is 6
  : (aubrey_birth + 8) - max_birth = 6 := by
sorry

end NUMINAMATH_CALUDE_max_age_on_aubrey_eighth_birthday_l3235_323505


namespace NUMINAMATH_CALUDE_arrangement_count_l3235_323591

/-- The number of white pieces -/
def white_pieces : ℕ := 5

/-- The number of black pieces -/
def black_pieces : ℕ := 10

/-- The number of different arrangements of white and black pieces
    satisfying the given conditions -/
def num_arrangements : ℕ := Nat.choose black_pieces white_pieces

theorem arrangement_count :
  num_arrangements = 252 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3235_323591


namespace NUMINAMATH_CALUDE_remainder_14_div_5_l3235_323567

theorem remainder_14_div_5 : 14 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_14_div_5_l3235_323567


namespace NUMINAMATH_CALUDE_pyramid_height_specific_l3235_323539

/-- Represents a pyramid with a square base and identical triangular faces. -/
structure Pyramid where
  base_area : ℝ
  face_area : ℝ

/-- The height of a pyramid given its base area and face area. -/
def pyramid_height (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating that a pyramid with base area 1440 and face area 840 has height 40. -/
theorem pyramid_height_specific : 
  ∀ (p : Pyramid), p.base_area = 1440 ∧ p.face_area = 840 → pyramid_height p = 40 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_specific_l3235_323539


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l3235_323560

theorem fraction_sum_equals_one (x y : ℝ) (h : x + y ≠ 0) :
  x / (x + y) + y / (x + y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l3235_323560


namespace NUMINAMATH_CALUDE_sufficient_condition_for_square_inequality_l3235_323500

theorem sufficient_condition_for_square_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) →
  (a ≥ 5 → (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0)) ∧
  ¬(∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 → a ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_square_inequality_l3235_323500


namespace NUMINAMATH_CALUDE_spent_sixty_four_l3235_323526

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * (victor_decks + friend_decks)

/-- Theorem: Victor and his friend spent $64 on trick decks -/
theorem spent_sixty_four :
  total_spent 8 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_spent_sixty_four_l3235_323526


namespace NUMINAMATH_CALUDE_third_median_length_l3235_323595

/-- Given a triangle with two medians of lengths 5 and 7 inches, and an area of 4√21 square inches,
    the length of the third median is 2√14 inches. -/
theorem third_median_length (m₁ m₂ : ℝ) (area : ℝ) (h₁ : m₁ = 5) (h₂ : m₂ = 7) (h_area : area = 4 * Real.sqrt 21) :
  ∃ (m₃ : ℝ), m₃ = 2 * Real.sqrt 14 ∧ 
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 3 * (m₁^2 + m₂^2 + m₃^2) ∧
                   area = (4 / 3) * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :=
by sorry

end NUMINAMATH_CALUDE_third_median_length_l3235_323595


namespace NUMINAMATH_CALUDE_geometric_sequence_a10_l3235_323521

/-- A geometric sequence with positive common ratio -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a10 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 4 * a 8 = 2 * (a 5)^2 →
  a 2 = 1 →
  a 10 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a10_l3235_323521


namespace NUMINAMATH_CALUDE_max_value_of_f_l3235_323516

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≥ -2) ∧  -- Minimum value is -2
  (∃ x ∈ Set.Icc 0 1, f a x = -2) →  -- Minimum value is achieved
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 1) ∧   -- Maximum value is at most 1
  (∃ x ∈ Set.Icc 0 1, f a x = 1)     -- Maximum value 1 is achieved
  := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3235_323516


namespace NUMINAMATH_CALUDE_total_chips_amount_l3235_323524

def person1_chips : ℕ := 350
def person2_chips : ℕ := 268
def person3_chips : ℕ := 182

theorem total_chips_amount : person1_chips + person2_chips + person3_chips = 800 := by
  sorry

end NUMINAMATH_CALUDE_total_chips_amount_l3235_323524


namespace NUMINAMATH_CALUDE_books_distribution_l3235_323597

/-- Number of ways to distribute books among students -/
def distribute_books (n_books : ℕ) (n_students : ℕ) : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem: Distributing 5 books among 3 students results in 90 different methods -/
theorem books_distribution :
  distribute_books 5 3 = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_books_distribution_l3235_323597


namespace NUMINAMATH_CALUDE_water_remaining_l3235_323593

/-- Given 3 gallons of water and using 5/4 gallons in an experiment, 
    prove that the remaining amount is 7/4 gallons. -/
theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l3235_323593


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l3235_323566

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three : 
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l3235_323566


namespace NUMINAMATH_CALUDE_betty_age_l3235_323554

/-- Given the ages of Albert, Mary, Betty, and Charlie, prove Betty's age --/
theorem betty_age (albert mary betty charlie : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 14)
  (h4 : charlie = 3 * betty)
  (h5 : charlie = mary + 10) :
  betty = 7 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l3235_323554


namespace NUMINAMATH_CALUDE_shirt_discount_problem_l3235_323559

theorem shirt_discount_problem (list_price : ℝ) (final_price : ℝ) (second_discount : ℝ) (first_discount : ℝ) : 
  list_price = 150 →
  final_price = 105 →
  second_discount = 12.5 →
  final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) →
  first_discount = 20 := by
sorry

end NUMINAMATH_CALUDE_shirt_discount_problem_l3235_323559


namespace NUMINAMATH_CALUDE_w_expression_l3235_323581

theorem w_expression (x y z w : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
  (eq : 1/x + 1/y + 1/z = 1/w) : 
  w = x*y*z / (y*z + x*z + x*y) := by
sorry

end NUMINAMATH_CALUDE_w_expression_l3235_323581


namespace NUMINAMATH_CALUDE_truncated_cone_inscribed_sphere_l3235_323574

/-- Given a truncated cone with an inscribed sphere, this theorem relates the ratio of their volumes
    to the angle between the generatrix and the base of the cone, and specifies the allowable values for the ratio. -/
theorem truncated_cone_inscribed_sphere (k : ℝ) (α : ℝ) :
  k > (3/2) →
  (∃ (V_cone V_sphere : ℝ), V_cone > 0 ∧ V_sphere > 0 ∧ V_cone / V_sphere = k) →
  α = Real.arctan (2 / Real.sqrt (2 * k - 3)) ∧
  α = angle_between_generatrix_and_base :=
by sorry

/-- Defines the angle between the generatrix and the base of the truncated cone. -/
def angle_between_generatrix_and_base : ℝ :=
sorry

end NUMINAMATH_CALUDE_truncated_cone_inscribed_sphere_l3235_323574


namespace NUMINAMATH_CALUDE_ecosystem_probability_l3235_323546

theorem ecosystem_probability : ∀ (n : ℕ) (p q r : ℚ),
  n = 7 →
  p = 1 / 5 →
  q = 1 / 10 →
  r = 17 / 20 →
  p + q + r = 1 →
  (Nat.choose n 4 : ℚ) * p^4 * r^3 = 34391 / 1000000 :=
by sorry

end NUMINAMATH_CALUDE_ecosystem_probability_l3235_323546


namespace NUMINAMATH_CALUDE_manager_percentage_problem_l3235_323563

theorem manager_percentage_problem (total_employees : ℕ) 
  (managers_left : ℕ) (final_percentage : ℚ) :
  total_employees = 500 →
  managers_left = 250 →
  final_percentage = 98/100 →
  (total_employees - managers_left) * final_percentage = 
    total_employees - managers_left - 
    ((100 - 99)/100 * total_employees) →
  99/100 * total_employees = total_employees - 
    ((100 - 99)/100 * total_employees) :=
by sorry

end NUMINAMATH_CALUDE_manager_percentage_problem_l3235_323563


namespace NUMINAMATH_CALUDE_unique_integer_pair_l3235_323582

theorem unique_integer_pair : ∃! (x y : ℕ+), 
  (x.val : ℝ) ^ (y.val : ℝ) + 1 = (y.val : ℝ) ^ (x.val : ℝ) ∧ 
  2 * (x.val : ℝ) ^ (y.val : ℝ) = (y.val : ℝ) ^ (x.val : ℝ) + 13 ∧ 
  x.val = 2 ∧ y.val = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_pair_l3235_323582


namespace NUMINAMATH_CALUDE_wheel_radius_increase_wheel_radius_increase_approx_011_l3235_323543

/-- Calculates the increase in wheel radius given the original and new measurements -/
theorem wheel_radius_increase (original_radius : ℝ) (original_distance : ℝ) 
  (new_odometer_distance : ℝ) (new_actual_distance : ℝ) : ℝ :=
  let original_circumference := 2 * Real.pi * original_radius
  let original_rotations := original_distance * 63360 / original_circumference
  let new_radius := new_actual_distance * 63360 / (2 * Real.pi * original_rotations)
  new_radius - original_radius

/-- The increase in wheel radius is approximately 0.11 inches -/
theorem wheel_radius_increase_approx_011 :
  abs (wheel_radius_increase 12 300 310 315 - 0.11) < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_wheel_radius_increase_wheel_radius_increase_approx_011_l3235_323543


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_value_l3235_323540

theorem function_inequality_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 3*x + m ≥ 2*x^2 - 4*x) ↔ (-1 ≤ x ∧ x ≤ 2)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_value_l3235_323540


namespace NUMINAMATH_CALUDE_decimal_multiplication_addition_l3235_323553

theorem decimal_multiplication_addition : (0.3 * 0.7) + (0.5 * 0.4) = 0.41 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_addition_l3235_323553


namespace NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_l3235_323527

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem statement -/
theorem parabola_midpoint_to_directrix 
  (para : Parabola) 
  (A B M : Point) 
  (h_line : (B.y - A.y) / (B.x - A.x) = 1) -- Slope of line AB is 1
  (h_on_parabola : A.y^2 = 2 * para.p * A.x ∧ B.y^2 = 2 * para.p * B.x) -- A and B are on the parabola
  (h_midpoint : M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2) -- M is midpoint of AB
  (h_m_y : M.y = 2) -- y-coordinate of M is 2
  : M.x - (-para.p) = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_l3235_323527


namespace NUMINAMATH_CALUDE_mary_extra_flour_l3235_323592

/-- Given a recipe that calls for a certain amount of flour and the actual amount used,
    calculate the extra amount of flour used. -/
def extra_flour (recipe_amount : ℝ) (actual_amount : ℝ) : ℝ :=
  actual_amount - recipe_amount

/-- Theorem stating that Mary used 2 extra cups of flour -/
theorem mary_extra_flour :
  let recipe_amount : ℝ := 7.0
  let actual_amount : ℝ := 9.0
  extra_flour recipe_amount actual_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_extra_flour_l3235_323592


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3235_323549

def first_ten_integers : Finset ℕ := Finset.range 10

theorem least_common_multiple_first_ten : ∃ (n : ℕ), n > 0 ∧ 
  (∀ i ∈ first_ten_integers, i.succ ∣ n) ∧ 
  (∀ m : ℕ, m > 0 → (∀ i ∈ first_ten_integers, i.succ ∣ m) → n ≤ m) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3235_323549


namespace NUMINAMATH_CALUDE_points_needed_in_next_game_l3235_323579

def last_home_game_score : ℕ := 62

def first_away_game_score : ℕ := last_home_game_score / 2

def second_away_game_score : ℕ := first_away_game_score + 18

def third_away_game_score : ℕ := second_away_game_score + 2

def cumulative_score_goal : ℕ := 4 * last_home_game_score

def current_cumulative_score : ℕ := 
  last_home_game_score + first_away_game_score + second_away_game_score + third_away_game_score

theorem points_needed_in_next_game : 
  cumulative_score_goal - current_cumulative_score = 55 := by
  sorry

end NUMINAMATH_CALUDE_points_needed_in_next_game_l3235_323579


namespace NUMINAMATH_CALUDE_smallest_value_l3235_323583

theorem smallest_value (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/(x+1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_l3235_323583


namespace NUMINAMATH_CALUDE_expression_evaluation_l3235_323503

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 7
  (2*x + 3) * (2*x - 3) - (x + 2)^2 + 4*(x + 3) = 20 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3235_323503


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3235_323509

theorem adult_ticket_cost 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (concession_cost : ℚ) 
  (total_cost : ℚ) 
  (child_ticket_cost : ℚ) 
  (h1 : num_adults = 5) 
  (h2 : num_children = 2) 
  (h3 : concession_cost = 12) 
  (h4 : total_cost = 76) 
  (h5 : child_ticket_cost = 7) :
  ∃ (adult_ticket_cost : ℚ), 
    adult_ticket_cost = 10 ∧ 
    (num_adults : ℚ) * adult_ticket_cost + 
    (num_children : ℚ) * child_ticket_cost + 
    concession_cost = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3235_323509


namespace NUMINAMATH_CALUDE_union_M_N_intersect_N_complement_M_l3235_323506

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

-- Theorem for M ∪ N
theorem union_M_N : M ∪ N = {x | x ≤ 2} := by sorry

-- Theorem for N ∩ (∁ᵤM)
theorem intersect_N_complement_M : N ∩ (U \ M) = {x | x < -2} := by sorry

end NUMINAMATH_CALUDE_union_M_N_intersect_N_complement_M_l3235_323506
