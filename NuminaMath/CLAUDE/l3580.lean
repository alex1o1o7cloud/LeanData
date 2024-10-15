import Mathlib

namespace NUMINAMATH_CALUDE_infinite_inequality_occurrences_l3580_358006

theorem infinite_inequality_occurrences (a : ℕ → ℕ+) : 
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    ∀ n ∈ S, (1 : ℝ) + a n > (a (n-1) : ℝ) * (2 : ℝ) ^ (1 / n) :=
sorry

end NUMINAMATH_CALUDE_infinite_inequality_occurrences_l3580_358006


namespace NUMINAMATH_CALUDE_worker_task_completion_time_l3580_358038

theorem worker_task_completion_time 
  (x y : ℝ) -- x and y represent the time taken by the first and second worker respectively
  (h1 : (1/x) + (2/x + 2/y) = 11/20) -- Work completed in 3 hours
  (h2 : (1/x) + (1/y) = 1/2) -- Each worker completes half the task
  : x = 10 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_worker_task_completion_time_l3580_358038


namespace NUMINAMATH_CALUDE_garden_tulips_count_l3580_358053

/-- Represents the garden scenario with tulips and sunflowers -/
structure Garden where
  tulip_ratio : ℕ
  sunflower_ratio : ℕ
  initial_sunflowers : ℕ
  added_sunflowers : ℕ

/-- Calculates the final number of tulips in the garden -/
def final_tulips (g : Garden) : ℕ :=
  let final_sunflowers := g.initial_sunflowers + g.added_sunflowers
  let ratio_units := final_sunflowers / g.sunflower_ratio
  ratio_units * g.tulip_ratio

/-- Theorem stating that given the garden conditions, the final number of tulips is 30 -/
theorem garden_tulips_count (g : Garden) 
  (h1 : g.tulip_ratio = 3)
  (h2 : g.sunflower_ratio = 7)
  (h3 : g.initial_sunflowers = 42)
  (h4 : g.added_sunflowers = 28) : 
  final_tulips g = 30 := by
  sorry

end NUMINAMATH_CALUDE_garden_tulips_count_l3580_358053


namespace NUMINAMATH_CALUDE_binomial_divisibility_iff_prime_l3580_358012

theorem binomial_divisibility_iff_prime (m : ℕ) (h : m ≥ 2) :
  (∀ n : ℕ, m / 3 ≤ n ∧ n ≤ m / 2 → n ∣ Nat.choose n (m - 2*n)) ↔ Nat.Prime m :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_iff_prime_l3580_358012


namespace NUMINAMATH_CALUDE_total_tax_percentage_l3580_358079

/-- Calculate the total tax percentage given spending percentages and tax rates -/
theorem total_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.60)
  (h2 : food_percent = 0.10)
  (h3 : other_percent = 0.30)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.04)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.08) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.048 := by
sorry

end NUMINAMATH_CALUDE_total_tax_percentage_l3580_358079


namespace NUMINAMATH_CALUDE_black_equals_sum_of_whites_l3580_358046

/-- Definition of a white number -/
def is_white_number (x : ℝ) : Prop :=
  ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ x = Real.sqrt (a + b * Real.sqrt 2)

/-- Definition of a black number -/
def is_black_number (x : ℝ) : Prop :=
  ∃ (c d : ℤ), c ≠ 0 ∧ d ≠ 0 ∧ x = Real.sqrt (c + d * Real.sqrt 7)

/-- Theorem stating that a black number can be equal to the sum of two white numbers -/
theorem black_equals_sum_of_whites :
  ∃ (x y z : ℝ), is_white_number x ∧ is_white_number y ∧ is_black_number z ∧ z = x + y :=
sorry

end NUMINAMATH_CALUDE_black_equals_sum_of_whites_l3580_358046


namespace NUMINAMATH_CALUDE_arthur_reading_time_ben_reading_time_l3580_358097

-- Define the reading speed of the narrator
def narrator_speed : ℝ := 1

-- Define the time it takes the narrator to read the book (in hours)
def narrator_time : ℝ := 3

-- Define Arthur's reading speed relative to the narrator
def arthur_speed : ℝ := 3 * narrator_speed

-- Define Ben's reading speed relative to the narrator
def ben_speed : ℝ := 4 * narrator_speed

-- Theorem for Arthur's reading time
theorem arthur_reading_time :
  (narrator_time * narrator_speed) / arthur_speed = 1 := by sorry

-- Theorem for Ben's reading time
theorem ben_reading_time :
  (narrator_time * narrator_speed) / ben_speed = 3/4 := by sorry

end NUMINAMATH_CALUDE_arthur_reading_time_ben_reading_time_l3580_358097


namespace NUMINAMATH_CALUDE_simplify_expression_l3580_358080

theorem simplify_expression (a b : ℝ) : 2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3580_358080


namespace NUMINAMATH_CALUDE_divisible_by_65_l3580_358090

theorem divisible_by_65 (n : ℕ) : ∃ k : ℤ, 5^n * (2^(2*n) - 3^n) + 2^n - 7^n = 65 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_65_l3580_358090


namespace NUMINAMATH_CALUDE_smallest_positive_translation_l3580_358022

theorem smallest_positive_translation (f : ℝ → ℝ) (φ : ℝ) : 
  (f = λ x => Real.sin (2 * x) + Real.cos (2 * x)) →
  (∀ x, f (x - φ) = f (φ - x)) →
  (∀ ψ, 0 < ψ ∧ ψ < φ → ¬(∀ x, f (x - ψ) = f (ψ - x))) →
  φ = 3 * Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_translation_l3580_358022


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l3580_358008

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 220 * x - 960

/-- Represents the optimal selling price that maximizes profit -/
def optimal_price : ℝ := 11

theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_price :=
sorry

#check optimal_price_maximizes_profit

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l3580_358008


namespace NUMINAMATH_CALUDE_simplify_expression_l3580_358015

theorem simplify_expression (p : ℝ) : 
  ((7 * p - 3) - 3 * p * 2) * 2 + (5 - 2 / 2) * (8 * p - 12) = 34 * p - 54 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3580_358015


namespace NUMINAMATH_CALUDE_addition_problem_l3580_358037

def base_8_to_10 (n : ℕ) : ℕ := 
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem addition_problem (X Y : ℕ) (h : X < 8 ∧ Y < 8) :
  base_8_to_10 (500 + 10 * X + Y) + base_8_to_10 32 = base_8_to_10 (600 + 40 + X) →
  X + Y = 16 := by
  sorry

end NUMINAMATH_CALUDE_addition_problem_l3580_358037


namespace NUMINAMATH_CALUDE_inequality_proof_l3580_358039

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 5/2) (hy : y ≥ 5/2) (hz : z ≥ 5/2) :
  (1 + 1/(2+x)) * (1 + 1/(2+y)) * (1 + 1/(2+z)) ≥ (1 + 1/(2 + (x*y*z)^(1/3)))^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3580_358039


namespace NUMINAMATH_CALUDE_dissimilar_terms_eq_distribution_ways_l3580_358072

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^7 -/
def dissimilar_terms : ℕ := sorry

/-- The number of ways to distribute 7 indistinguishable objects into 4 distinguishable boxes -/
def distribution_ways : ℕ := sorry

/-- Theorem stating that the number of dissimilar terms in (a + b + c + d)^7 
    is equal to the number of ways to distribute 7 objects into 4 boxes -/
theorem dissimilar_terms_eq_distribution_ways : 
  dissimilar_terms = distribution_ways := by sorry

end NUMINAMATH_CALUDE_dissimilar_terms_eq_distribution_ways_l3580_358072


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3580_358093

theorem arithmetic_mean_problem (a b c : ℝ) 
  (h1 : (a + b) / 2 = 80) 
  (h2 : (b + c) / 2 = 180) : 
  a - c = -200 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3580_358093


namespace NUMINAMATH_CALUDE_square_is_one_l3580_358070

/-- Represents a digit in base-7 --/
def Base7Digit := Fin 7

/-- The addition problem in base-7 --/
def addition_problem (square : Base7Digit) : Prop :=
  ∃ (carry1 carry2 carry3 : Nat),
    (square.val + 1 + 3 + 2) % 7 = 0 ∧
    (carry1 + square.val + 5 + square.val + 1) % 7 = square.val ∧
    (carry2 + 4 + carry3) % 7 = 5 ∧
    carry1 = (square.val + 1 + 3 + 2) / 7 ∧
    carry2 = (carry1 + square.val + 5 + square.val + 1) / 7 ∧
    carry3 = (square.val + 5 + 1) / 7

theorem square_is_one :
  ∃ (square : Base7Digit), addition_problem square ∧ square.val = 1 := by sorry

end NUMINAMATH_CALUDE_square_is_one_l3580_358070


namespace NUMINAMATH_CALUDE_train_speed_l3580_358057

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 250.00000000000003)
  (h2 : time = 15) : 
  ∃ (speed : ℝ), abs (speed - 60) < 0.00000000000001 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3580_358057


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_27_l3580_358025

def polynomial (x : ℝ) : ℝ :=
  -3 * (x^7 - 2*x^6 + x^4 - 3*x^2 + 6) + 6 * (x^3 - 4*x + 1) - 2 * (x^5 - 5*x + 7)

theorem sum_of_coefficients_is_negative_27 : 
  (polynomial 1) = -27 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_27_l3580_358025


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_four_l3580_358058

/-- A line in 2D space defined by parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if two lines are parallel --/
def are_parallel (l1 l2 : ParametricLine) : Prop :=
  ∃ (k : ℝ), ∀ (t : ℝ), 
    (l1.x t - l1.x 0) * (l2.y t - l2.y 0) = k * (l1.y t - l1.y 0) * (l2.x t - l2.x 0)

/-- The first line l₁ --/
def l1 : ParametricLine where
  x := λ s => 2 * s + 1
  y := λ s => s

/-- The second line l₂ --/
def l2 (a : ℝ) : ParametricLine where
  x := λ t => a * t
  y := λ t => 2 * t - 1

/-- Theorem: If l₁ and l₂ are parallel, then a = 4 --/
theorem parallel_lines_a_equals_four :
  are_parallel l1 (l2 a) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_four_l3580_358058


namespace NUMINAMATH_CALUDE_average_bowling_score_l3580_358075

-- Define the players and their scores
def gretchen_score : ℕ := 120
def mitzi_score : ℕ := 113
def beth_score : ℕ := 85

-- Define the number of players
def num_players : ℕ := 3

-- Define the total score
def total_score : ℕ := gretchen_score + mitzi_score + beth_score

-- Theorem to prove
theorem average_bowling_score :
  (total_score : ℚ) / num_players = 106 := by
  sorry

end NUMINAMATH_CALUDE_average_bowling_score_l3580_358075


namespace NUMINAMATH_CALUDE_village_seniors_l3580_358029

/-- Proves the number of seniors in a village given the population distribution -/
theorem village_seniors (total_population : ℕ) 
  (h1 : total_population * 60 / 100 = 23040)  -- 60% of population are adults
  (h2 : total_population * 30 / 100 = total_population * 3 / 10) -- 30% are children
  : total_population * 10 / 100 = 3840 := by
  sorry

end NUMINAMATH_CALUDE_village_seniors_l3580_358029


namespace NUMINAMATH_CALUDE_consecutive_sum_39_l3580_358099

theorem consecutive_sum_39 (n : ℕ) : 
  n + (n + 1) = 39 → n = 19 := by
sorry

end NUMINAMATH_CALUDE_consecutive_sum_39_l3580_358099


namespace NUMINAMATH_CALUDE_julie_income_calculation_l3580_358077

/-- Calculates Julie's net monthly income based on given conditions --/
def julies_net_monthly_income (
  starting_pay : ℝ)
  (experience_bonus : ℝ)
  (years_experience : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (biweekly_bonus : ℝ)
  (tax_rate : ℝ)
  (insurance_premium : ℝ)
  (missed_days : ℕ) : ℝ :=
  sorry

/-- Theorem stating that Julie's net monthly income is $963.20 --/
theorem julie_income_calculation :
  julies_net_monthly_income 5 0.5 3 8 6 50 0.12 40 1 = 963.20 :=
by sorry

end NUMINAMATH_CALUDE_julie_income_calculation_l3580_358077


namespace NUMINAMATH_CALUDE_initial_salty_cookies_count_l3580_358055

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := sorry

/-- The number of salty cookies Paco ate -/
def eaten_salty_cookies : ℕ := 3

/-- The number of salty cookies Paco had left after eating -/
def remaining_salty_cookies : ℕ := 3

/-- Theorem stating that the initial number of salty cookies is 6 -/
theorem initial_salty_cookies_count : initial_salty_cookies = 6 := by sorry

end NUMINAMATH_CALUDE_initial_salty_cookies_count_l3580_358055


namespace NUMINAMATH_CALUDE_ratio_problem_l3580_358096

theorem ratio_problem (a b : ℝ) 
  (h1 : b / a = 2) 
  (h2 : b = 15 - 4 * a) : 
  a = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l3580_358096


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3580_358094

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3580_358094


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l3580_358065

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Theorem stating that f is an odd function and an increasing function
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l3580_358065


namespace NUMINAMATH_CALUDE_product_of_decimals_l3580_358010

theorem product_of_decimals : (0.5 : ℝ) * 0.3 = 0.15 := by sorry

end NUMINAMATH_CALUDE_product_of_decimals_l3580_358010


namespace NUMINAMATH_CALUDE_right_triangle_relation_l3580_358056

theorem right_triangle_relation (a b c h : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hh : h > 0)
  (right_triangle : a^2 + b^2 = c^2) (height_relation : 2 * h * c = a * b) :
  1 / a^2 + 1 / b^2 = 1 / h^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_relation_l3580_358056


namespace NUMINAMATH_CALUDE_inequality_proof_l3580_358035

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h_prod : a * b * c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1 / a + 1 / b + 1 / c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3580_358035


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l3580_358002

theorem meaningful_fraction_range (x : ℝ) :
  (|x| - 6 ≠ 0) ↔ (x ≠ 6 ∧ x ≠ -6) := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l3580_358002


namespace NUMINAMATH_CALUDE_marble_distribution_l3580_358045

theorem marble_distribution (total_marbles : ℕ) (num_groups : ℕ) (marbles_per_group : ℕ) :
  total_marbles = 64 →
  num_groups = 32 →
  total_marbles = num_groups * marbles_per_group →
  marbles_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l3580_358045


namespace NUMINAMATH_CALUDE_expression_simplification_l3580_358031

theorem expression_simplification (p : ℝ) 
  (h1 : p^3 - p^2 + 2*p + 16 ≠ 0) 
  (h2 : p^2 + 2*p + 6 ≠ 0) : 
  (p^3 + 4*p^2 + 10*p + 12) / (p^3 - p^2 + 2*p + 16) * 
  (p^3 - 3*p^2 + 8*p) / (p^2 + 2*p + 6) = p := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3580_358031


namespace NUMINAMATH_CALUDE_blanket_folding_ratio_l3580_358064

theorem blanket_folding_ratio (initial_thickness final_thickness : ℝ) 
  (num_folds : ℕ) (ratio : ℝ) 
  (h1 : initial_thickness = 3)
  (h2 : final_thickness = 48)
  (h3 : num_folds = 4)
  (h4 : final_thickness = initial_thickness * ratio ^ num_folds) :
  ratio = 2 := by
sorry

end NUMINAMATH_CALUDE_blanket_folding_ratio_l3580_358064


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_2010_l3580_358018

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_2010 : 
  units_digit (factorial_sum 2010) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_2010_l3580_358018


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3580_358009

-- Define the sets M and N
def M : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def N : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3580_358009


namespace NUMINAMATH_CALUDE_paula_cans_used_l3580_358068

/-- Represents the painting scenario with Paula the painter --/
structure PaintingScenario where
  initial_rooms : ℕ
  lost_cans : ℕ
  final_rooms : ℕ

/-- Calculates the number of cans used for painting given a scenario --/
def cans_used (scenario : PaintingScenario) : ℕ :=
  sorry

/-- Theorem stating the number of cans used in Paula's specific scenario --/
theorem paula_cans_used :
  let scenario : PaintingScenario := {
    initial_rooms := 45,
    lost_cans := 5,
    final_rooms := 35
  }
  cans_used scenario = 18 :=
by sorry

end NUMINAMATH_CALUDE_paula_cans_used_l3580_358068


namespace NUMINAMATH_CALUDE_division_sum_theorem_l3580_358092

theorem division_sum_theorem (quotient divisor remainder : ℝ) :
  quotient = 450 →
  divisor = 350.7 →
  remainder = 287.9 →
  (divisor * quotient) + remainder = 158102.9 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l3580_358092


namespace NUMINAMATH_CALUDE_max_difference_reverse_digits_l3580_358007

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Main theorem -/
theorem max_difference_reverse_digits (q r : ℕ) :
  TwoDigitInt q ∧ TwoDigitInt r ∧
  r = reverseDigits q ∧
  q > r ∧
  q - r < 20 →
  q - r ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_max_difference_reverse_digits_l3580_358007


namespace NUMINAMATH_CALUDE_three_isosceles_triangles_l3580_358073

-- Define a point on the grid
structure GridPoint where
  x : Int
  y : Int

-- Define a triangle on the grid
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : GridPoint) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : GridTriangle) : Bool :=
  let d1 := squaredDistance t.p1 t.p2
  let d2 := squaredDistance t.p2 t.p3
  let d3 := squaredDistance t.p3 t.p1
  d1 = d2 || d2 = d3 || d3 = d1

-- Define the five triangles
def triangle1 : GridTriangle := ⟨⟨0, 8⟩, ⟨4, 8⟩, ⟨2, 5⟩⟩
def triangle2 : GridTriangle := ⟨⟨2, 2⟩, ⟨2, 5⟩, ⟨6, 2⟩⟩
def triangle3 : GridTriangle := ⟨⟨1, 1⟩, ⟨5, 4⟩, ⟨9, 1⟩⟩
def triangle4 : GridTriangle := ⟨⟨7, 7⟩, ⟨6, 9⟩, ⟨10, 7⟩⟩
def triangle5 : GridTriangle := ⟨⟨3, 1⟩, ⟨4, 4⟩, ⟨6, 0⟩⟩

-- List of all triangles
def allTriangles : List GridTriangle := [triangle1, triangle2, triangle3, triangle4, triangle5]

-- Theorem: Exactly 3 out of the 5 given triangles are isosceles
theorem three_isosceles_triangles :
  (allTriangles.filter isIsosceles).length = 3 := by
  sorry


end NUMINAMATH_CALUDE_three_isosceles_triangles_l3580_358073


namespace NUMINAMATH_CALUDE_allocation_theorem_l3580_358000

/-- The number of ways to allocate doctors and nurses to schools -/
def allocation_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  num_doctors * (num_nurses.choose (num_nurses / num_schools))

/-- Theorem: There are 12 ways to allocate 2 doctors and 4 nurses to 2 schools -/
theorem allocation_theorem :
  allocation_methods 2 4 2 = 12 := by
  sorry

#eval allocation_methods 2 4 2

end NUMINAMATH_CALUDE_allocation_theorem_l3580_358000


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3580_358021

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x + 3 = 7 - x
def equation2 (x : ℝ) : Prop := (1/2) * x - 6 = (3/4) * x

-- Theorem for equation 1
theorem solution_equation1 : ∃! x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃! x : ℝ, equation2 x ∧ x = -24 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3580_358021


namespace NUMINAMATH_CALUDE_mode_is_80_l3580_358088

/-- Represents the frequency of each score in the test results -/
def score_frequency : List (Nat × Nat) := [
  (61, 1), (61, 1), (62, 1),
  (75, 1), (77, 1),
  (80, 3), (81, 1), (83, 2),
  (92, 2), (94, 1), (96, 1), (97, 2),
  (105, 2), (109, 1),
  (110, 2)
]

/-- The maximum score possible on the test -/
def max_score : Nat := 120

/-- Definition of the mode: the score that appears most frequently -/
def is_mode (score : Nat) (frequencies : List (Nat × Nat)) : Prop :=
  ∀ other_score, other_score ≠ score →
    (frequencies.filter (λ pair => pair.1 = score)).length ≥
    (frequencies.filter (λ pair => pair.1 = other_score)).length

/-- Theorem stating that 80 is the mode of the scores -/
theorem mode_is_80 : is_mode 80 score_frequency := by
  sorry

end NUMINAMATH_CALUDE_mode_is_80_l3580_358088


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l3580_358071

theorem square_circle_area_ratio :
  ∀ (r : ℝ) (s : ℝ),
    r > 0 →
    s > 0 →
    s = r * Real.sqrt 15 / 2 →
    (s^2) / (π * r^2) = 15 / (4 * π) := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l3580_358071


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l3580_358023

/-- The probability of getting exactly k successes in n independent Bernoulli trials -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The theorem statement -/
theorem coin_flip_probability_difference :
  let p_four_heads := binomial_probability 5 4 (1/2)
  let p_five_heads := binomial_probability 5 5 (1/2)
  |p_four_heads - p_five_heads| = 1/8 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l3580_358023


namespace NUMINAMATH_CALUDE_orange_weight_l3580_358024

theorem orange_weight (apple_weight orange_weight : ℝ) 
  (h1 : orange_weight = 5 * apple_weight) 
  (h2 : apple_weight + orange_weight = 12) : 
  orange_weight = 10 := by
sorry

end NUMINAMATH_CALUDE_orange_weight_l3580_358024


namespace NUMINAMATH_CALUDE_max_product_constrained_max_product_achieved_l3580_358089

theorem max_product_constrained (x y : ℕ+) (h : 7 * x + 4 * y = 140) :
  x * y ≤ 168 := by
  sorry

theorem max_product_achieved : ∃ (x y : ℕ+), 7 * x + 4 * y = 140 ∧ x * y = 168 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_max_product_achieved_l3580_358089


namespace NUMINAMATH_CALUDE_subtraction_of_negative_l3580_358054

theorem subtraction_of_negative : 3 - (-3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_l3580_358054


namespace NUMINAMATH_CALUDE_particle_probability_l3580_358044

/-- The probability of a particle reaching point (2,3) after 5 moves -/
theorem particle_probability (n : ℕ) (k : ℕ) (p : ℝ) : 
  n = 5 → k = 2 → p = 1/2 → 
  Nat.choose n k * p^n = Nat.choose 5 2 * (1/2)^5 :=
by sorry

end NUMINAMATH_CALUDE_particle_probability_l3580_358044


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l3580_358095

theorem cubic_roots_relation (a b c : ℂ) : 
  (a^3 - 3*a^2 + 5*a - 8 = 0) → 
  (b^3 - 3*b^2 + 5*b - 8 = 0) → 
  (c^3 - 3*c^2 + 5*c - 8 = 0) → 
  (∃ r s : ℂ, (a-b)^3 + r*(a-b)^2 + s*(a-b) + 243 = 0 ∧ 
               (b-c)^3 + r*(b-c)^2 + s*(b-c) + 243 = 0 ∧ 
               (c-a)^3 + r*(c-a)^2 + s*(c-a) + 243 = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l3580_358095


namespace NUMINAMATH_CALUDE_smallest_base_for_100_in_three_digits_l3580_358060

theorem smallest_base_for_100_in_three_digits : ∃ (b : ℕ), b = 5 ∧ 
  (∀ (n : ℕ), n^2 ≤ 100 ∧ 100 < n^3 → b ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_100_in_three_digits_l3580_358060


namespace NUMINAMATH_CALUDE_no_rectangle_with_half_perimeter_and_area_l3580_358003

theorem no_rectangle_with_half_perimeter_and_area 
  (a b : ℝ) (h_ab : 0 < a ∧ a < b) : 
  ¬∃ (x y : ℝ), 
    0 < x ∧ x < b ∧
    0 < y ∧ y < b ∧
    x + y = a + b ∧
    x * y = (a * b) / 2 := by
sorry

end NUMINAMATH_CALUDE_no_rectangle_with_half_perimeter_and_area_l3580_358003


namespace NUMINAMATH_CALUDE_january_first_is_monday_l3580_358040

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  days : Nat
  firstDay : DayOfWeek
  mondayCount : Nat
  thursdayCount : Nat

/-- Theorem stating that a month with 31 days, 5 Mondays, and 4 Thursdays must start on a Monday -/
theorem january_first_is_monday (m : Month) :
  m.days = 31 ∧ m.mondayCount = 5 ∧ m.thursdayCount = 4 →
  m.firstDay = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_january_first_is_monday_l3580_358040


namespace NUMINAMATH_CALUDE_population_growth_rate_l3580_358063

/-- Given a population increase of 160 persons in 40 minutes, 
    proves that the time taken for one person to be added is 15 seconds. -/
theorem population_growth_rate (persons : ℕ) (minutes : ℕ) (seconds_per_person : ℕ) : 
  persons = 160 ∧ minutes = 40 → seconds_per_person = 15 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l3580_358063


namespace NUMINAMATH_CALUDE_fraction_value_l3580_358014

theorem fraction_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : y = x / (x + 1)) :
  (x - y + 4 * x * y) / (x * y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3580_358014


namespace NUMINAMATH_CALUDE_prime_sum_product_l3580_358013

theorem prime_sum_product (p₁ p₂ p₃ p₄ : ℕ) 
  (h_prime₁ : Nat.Prime p₁) (h_prime₂ : Nat.Prime p₂) 
  (h_prime₃ : Nat.Prime p₃) (h_prime₄ : Nat.Prime p₄)
  (h_order : p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄)
  (h_sum : p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882) :
  p₁ = 7 ∧ p₂ = 11 ∧ p₃ = 13 ∧ p₄ = 17 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l3580_358013


namespace NUMINAMATH_CALUDE_square_sum_of_special_integers_l3580_358067

theorem square_sum_of_special_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 71)
  (h2 : x^2 * y + x * y^2 = 880) : 
  x^2 + y^2 = 146 := by sorry

end NUMINAMATH_CALUDE_square_sum_of_special_integers_l3580_358067


namespace NUMINAMATH_CALUDE_root_product_theorem_l3580_358076

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^5 + 3*x^2 + 1

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 5

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (hroots : f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = 131 := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3580_358076


namespace NUMINAMATH_CALUDE_x_shape_is_line_segments_l3580_358001

/-- The shape defined by θ = π/4 or θ = 5π/4 within 2 units of the origin -/
def X_shape : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 + p.2^2 ≤ 4) ∧ 
    (p.2 = p.1 ∨ p.2 = -p.1) ∧ 
    (p.1 ≠ 0 ∨ p.2 ≠ 0)}

theorem x_shape_is_line_segments : 
  ∃ (a b c d : ℝ × ℝ), 
    a ≠ b ∧ c ≠ d ∧
    X_shape = {p : ℝ × ℝ | ∃ (t : ℝ), (0 ≤ t ∧ t ≤ 1 ∧ 
      ((p = (1 - t) • a + t • b) ∨ (p = (1 - t) • c + t • d)))} :=
sorry

end NUMINAMATH_CALUDE_x_shape_is_line_segments_l3580_358001


namespace NUMINAMATH_CALUDE_ramsey_three_three_three_l3580_358061

/-- A coloring of edges in a complete graph with three colors -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A monochromatic triangle in a coloring -/
def HasMonochromaticTriangle (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    c i j = c j k ∧ c j k = c i k

/-- The Ramsey theorem R(3,3,3) ≤ 17 -/
theorem ramsey_three_three_three :
  ∀ (c : Coloring 17), HasMonochromaticTriangle 17 c :=
sorry

end NUMINAMATH_CALUDE_ramsey_three_three_three_l3580_358061


namespace NUMINAMATH_CALUDE_circle_max_area_center_l3580_358052

/-- A circle with equation x^2 + y^2 + kx + 2y + k^2 = 0 -/
def Circle (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + k * p.1 + 2 * p.2 + k^2 = 0}

/-- The center of a circle -/
def center (c : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The area of a circle -/
def area (c : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The center of the circle is (0, -1) when its area is maximized -/
theorem circle_max_area_center (k : ℝ) :
  (∀ k' : ℝ, area (Circle k') ≤ area (Circle k)) →
  center (Circle k) = (0, -1) := by sorry

end NUMINAMATH_CALUDE_circle_max_area_center_l3580_358052


namespace NUMINAMATH_CALUDE_students_per_group_l3580_358042

theorem students_per_group 
  (total_students : ℕ) 
  (students_not_picked : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 64) 
  (h2 : students_not_picked = 36) 
  (h3 : num_groups = 4) : 
  (total_students - students_not_picked) / num_groups = 7 := by
sorry

end NUMINAMATH_CALUDE_students_per_group_l3580_358042


namespace NUMINAMATH_CALUDE_power_equation_solution_l3580_358085

theorem power_equation_solution : ∃ m : ℤ, 2^4 - 3 = 5^2 + m ∧ m = -12 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3580_358085


namespace NUMINAMATH_CALUDE_even_function_order_l3580_358098

def f (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem even_function_order (b c : ℝ) 
  (h : ∀ x, f x b c = f (-x) b c) : 
  f 1 b c < f (-2) b c ∧ f (-2) b c < f 3 b c := by
  sorry

end NUMINAMATH_CALUDE_even_function_order_l3580_358098


namespace NUMINAMATH_CALUDE_simplify_expression_l3580_358049

theorem simplify_expression (x : ℝ) : x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -5*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3580_358049


namespace NUMINAMATH_CALUDE_package_weight_problem_l3580_358034

theorem package_weight_problem (x y z w : ℝ) 
  (h1 : x + y + z = 150)
  (h2 : y + z + w = 160)
  (h3 : z + w + x = 170) :
  x + y + z + w = 160 := by
sorry

end NUMINAMATH_CALUDE_package_weight_problem_l3580_358034


namespace NUMINAMATH_CALUDE_joyce_apples_l3580_358026

def initial_apples : ℕ := 75
def apples_given : ℕ := 52
def apples_left : ℕ := 23

theorem joyce_apples : initial_apples = apples_given + apples_left := by
  sorry

end NUMINAMATH_CALUDE_joyce_apples_l3580_358026


namespace NUMINAMATH_CALUDE_find_p_value_l3580_358016

theorem find_p_value (x y z p : ℝ) 
  (h1 : 8 / (x + y) = p / (x + z)) 
  (h2 : p / (x + z) = 12 / (z - y)) : p = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_p_value_l3580_358016


namespace NUMINAMATH_CALUDE_equal_negative_exponents_l3580_358078

theorem equal_negative_exponents : -2^3 = (-2)^3 ∧ 
  -3^2 ≠ -2^3 ∧ 
  (-3 * 2)^2 ≠ -3 * 2^2 ∧ 
  -3^2 ≠ (-3)^2 :=
by sorry

end NUMINAMATH_CALUDE_equal_negative_exponents_l3580_358078


namespace NUMINAMATH_CALUDE_sweater_discount_percentage_l3580_358032

/-- Proves that the discount percentage is approximately 15.5% given the conditions -/
theorem sweater_discount_percentage (markup : ℝ) (profit : ℝ) :
  markup = 0.5384615384615385 →
  profit = 0.3 →
  let normal_price := 1 + markup
  let discounted_price := 1 + profit
  let discount := (normal_price - discounted_price) / normal_price
  abs (discount - 0.155) < 0.001 := by
sorry

end NUMINAMATH_CALUDE_sweater_discount_percentage_l3580_358032


namespace NUMINAMATH_CALUDE_shekar_science_score_l3580_358027

def average_marks : ℝ := 77
def num_subjects : ℕ := 5
def math_score : ℝ := 76
def social_studies_score : ℝ := 82
def english_score : ℝ := 67
def biology_score : ℝ := 95

theorem shekar_science_score :
  ∃ (science_score : ℝ),
    (math_score + social_studies_score + english_score + biology_score + science_score) / num_subjects = average_marks ∧
    science_score = 65 := by
  sorry

end NUMINAMATH_CALUDE_shekar_science_score_l3580_358027


namespace NUMINAMATH_CALUDE_flow_rate_is_twelve_l3580_358074

/-- Represents the flow rate problem described in the question -/
def flow_rate_problem (tub_capacity : ℕ) (leak_rate : ℕ) (fill_time : ℕ) : ℕ :=
  let cycles := fill_time / 2
  let net_fill_per_cycle := (tub_capacity / cycles) + (2 * leak_rate)
  net_fill_per_cycle

/-- Theorem stating that the flow rate is 12 liters per minute under the given conditions -/
theorem flow_rate_is_twelve :
  flow_rate_problem 120 1 24 = 12 := by
  sorry

end NUMINAMATH_CALUDE_flow_rate_is_twelve_l3580_358074


namespace NUMINAMATH_CALUDE_man_travel_distance_l3580_358069

theorem man_travel_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2 → time = 39 → distance = speed * time → distance = 78 := by
  sorry

end NUMINAMATH_CALUDE_man_travel_distance_l3580_358069


namespace NUMINAMATH_CALUDE_linear_function_problem_l3580_358050

-- Define a linear function f
def f (x : ℝ) : ℝ := sorry

-- Define the inverse function of f
def f_inv (x : ℝ) : ℝ := sorry

-- State the theorem
theorem linear_function_problem :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) →  -- f is linear
  (∀ x : ℝ, f x = 3 * f_inv x + 9) →         -- f(x) = 3f^(-1)(x) + 9
  f 3 = 6 →                                  -- f(3) = 6
  f 6 = 10.5 * Real.sqrt 3 - 4.5 :=          -- f(6) = 10.5√3 - 4.5
by sorry

end NUMINAMATH_CALUDE_linear_function_problem_l3580_358050


namespace NUMINAMATH_CALUDE_regular_pentagon_diagonal_l3580_358036

/-- For a regular pentagon with side length a, its diagonal d satisfies d = (√5 + 1)/2 * a -/
theorem regular_pentagon_diagonal (a : ℝ) (h : a > 0) :
  ∃ d : ℝ, d > 0 ∧ d = (Real.sqrt 5 + 1) / 2 * a := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_diagonal_l3580_358036


namespace NUMINAMATH_CALUDE_harrison_croissant_cost_l3580_358066

/-- The cost of croissants for Harrison in a year -/
def croissant_cost (regular_price almond_price : ℚ) (weeks_per_year : ℕ) : ℚ :=
  weeks_per_year * (regular_price + almond_price)

/-- Theorem: Harrison spends $468.00 on croissants in a year -/
theorem harrison_croissant_cost :
  croissant_cost (35/10) (55/10) 52 = 468 :=
sorry

end NUMINAMATH_CALUDE_harrison_croissant_cost_l3580_358066


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l3580_358033

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence. -/
def nthTerm (a : ℕ → ℝ) (n : ℕ) : ℝ := a n

theorem tenth_term_of_arithmetic_sequence
    (a : ℕ → ℝ)
    (h_arith : ArithmeticSequence a)
    (h_4th : nthTerm a 4 = 23)
    (h_6th : nthTerm a 6 = 43) :
  nthTerm a 10 = 83 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l3580_358033


namespace NUMINAMATH_CALUDE_max_product_of_tangent_circles_l3580_358062

/-- Two circles C₁ and C₂ are externally tangent -/
def externally_tangent (a b : ℝ) : Prop :=
  a + b = 3

/-- The product of a and b -/
def product (a b : ℝ) : ℝ := a * b

/-- The theorem stating the maximum value of ab -/
theorem max_product_of_tangent_circles (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_tangent : externally_tangent a b) :
  product a b ≤ 9/4 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_tangent_circles_l3580_358062


namespace NUMINAMATH_CALUDE_lap_time_improvement_l3580_358087

def initial_laps : ℕ := 10
def initial_time : ℕ := 25
def current_laps : ℕ := 12
def current_time : ℕ := 24

def improvement : ℚ := 1/2

theorem lap_time_improvement : 
  (initial_time : ℚ) / initial_laps - (current_time : ℚ) / current_laps = improvement :=
sorry

end NUMINAMATH_CALUDE_lap_time_improvement_l3580_358087


namespace NUMINAMATH_CALUDE_isosceles_triangle_removal_l3580_358086

/-- Given a square with side length x, from which isosceles right triangles
    with leg length s are removed from each corner to form a rectangle
    with longer side 15 units, prove that the combined area of the four
    removed triangles is 225 square units. -/
theorem isosceles_triangle_removal (x s : ℝ) : 
  x > 0 →
  s > 0 →
  x - 2*s = 15 →
  (x - s)^2 + (x - s)^2 = x^2 →
  4 * (1/2 * s^2) = 225 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_removal_l3580_358086


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3580_358030

-- Problem 1
theorem problem_1 (a b : ℝ) :
  a^2 * (2*a*b - 1) + (a - 3*b) * (a + b) = 2*a^3*b - 2*a*b - 3*b^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) :
  (2*x - 3)^2 - (x + 2)^2 = 3*x^2 - 16*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3580_358030


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_two_l3580_358047

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x + 1

theorem tangent_line_at_zero_two :
  let f : ℝ → ℝ := λ x ↦ Real.exp x + 2 * x + 1
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3 * x - y + 2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_two_l3580_358047


namespace NUMINAMATH_CALUDE_tray_trips_l3580_358005

theorem tray_trips (capacity : ℕ) (total_trays : ℕ) (h1 : capacity = 8) (h2 : total_trays = 16) :
  (total_trays + capacity - 1) / capacity = 2 := by
  sorry

end NUMINAMATH_CALUDE_tray_trips_l3580_358005


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l3580_358020

def x : ℕ := 5 * 24 * 36

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube : 
  (∀ y < 50, ¬ is_perfect_cube (x * y)) ∧ is_perfect_cube (x * 50) := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l3580_358020


namespace NUMINAMATH_CALUDE_intersection_equality_l3580_358059

def M : Set ℤ := {-1, 0, 1}

def N (a : ℤ) : Set ℤ := {a, a^2}

theorem intersection_equality (a : ℤ) : M ∩ N a = N a ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l3580_358059


namespace NUMINAMATH_CALUDE_servings_count_l3580_358028

/-- Represents the number of cups of cereal in a box -/
def total_cups : ℕ := 18

/-- Represents the number of cups per serving -/
def cups_per_serving : ℕ := 2

/-- Calculates the number of servings in a cereal box -/
def servings_in_box : ℕ := total_cups / cups_per_serving

/-- Proves that the number of servings in the cereal box is 9 -/
theorem servings_count : servings_in_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_servings_count_l3580_358028


namespace NUMINAMATH_CALUDE_equation_solution_l3580_358043

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x * (x - 2) - 2 * (x + 1)
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 6 ∧ x₂ = 2 - Real.sqrt 6 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3580_358043


namespace NUMINAMATH_CALUDE_units_digit_27_times_36_l3580_358004

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_27_times_36 : units_digit (27 * 36) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_27_times_36_l3580_358004


namespace NUMINAMATH_CALUDE_pool_perimeter_is_20_l3580_358048

/-- Represents the dimensions and properties of a garden with a rectangular pool -/
structure GardenPool where
  garden_length : ℝ
  garden_width : ℝ
  pool_area : ℝ
  walkway_width : ℝ

/-- Calculates the perimeter of the pool given the garden dimensions and pool properties -/
def pool_perimeter (g : GardenPool) : ℝ :=
  2 * ((g.garden_length - 2 * g.walkway_width) + (g.garden_width - 2 * g.walkway_width))

/-- Theorem stating that the perimeter of the pool is 20 meters under the given conditions -/
theorem pool_perimeter_is_20 (g : GardenPool) 
    (h1 : g.garden_length = 8)
    (h2 : g.garden_width = 6)
    (h3 : g.pool_area = 24)
    (h4 : (g.garden_length - 2 * g.walkway_width) * (g.garden_width - 2 * g.walkway_width) = g.pool_area) :
  pool_perimeter g = 20 := by
  sorry

#check pool_perimeter_is_20

end NUMINAMATH_CALUDE_pool_perimeter_is_20_l3580_358048


namespace NUMINAMATH_CALUDE_combined_age_theorem_l3580_358084

/-- The combined age of Jane and John after 12 years -/
def combined_age_after_12_years (justin_age : ℕ) (jessica_age_diff : ℕ) (james_age_diff : ℕ) (julia_age_diff : ℕ) (jane_age_diff : ℕ) (john_age_diff : ℕ) : ℕ :=
  let jessica_age := justin_age + jessica_age_diff
  let james_age := jessica_age + james_age_diff
  let jane_age := james_age + jane_age_diff
  let john_age := jane_age + john_age_diff
  (jane_age + 12) + (john_age + 12)

/-- Theorem stating the combined age of Jane and John after 12 years -/
theorem combined_age_theorem :
  combined_age_after_12_years 26 6 7 8 25 3 = 155 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_theorem_l3580_358084


namespace NUMINAMATH_CALUDE_cross_section_area_unit_cube_l3580_358017

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  edge_length : ℝ

/-- Theorem: Area of cross-section in a unit cube -/
theorem cross_section_area_unit_cube (c : Cube) (X Y Z : Point3D) :
  c.edge_length = 1 →
  X = ⟨1/2, 1/2, 0⟩ →
  Y = ⟨1, 1/2, 1/2⟩ →
  Z = ⟨3/4, 3/4, 3/4⟩ →
  let sphere_radius := Real.sqrt 3 / 2
  let plane_distance := Real.sqrt ((1/4)^2 + (1/4)^2 + (3/4)^2)
  let cross_section_radius := Real.sqrt (sphere_radius^2 - plane_distance^2)
  let cross_section_area := π * cross_section_radius^2
  cross_section_area = 5 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_unit_cube_l3580_358017


namespace NUMINAMATH_CALUDE_train_speed_and_length_l3580_358019

def bridge_length : ℝ := 1260
def bridge_time : ℝ := 60
def tunnel_length : ℝ := 2010
def tunnel_time : ℝ := 90

theorem train_speed_and_length :
  ∃ (speed length : ℝ),
    (bridge_length + length) / bridge_time = (tunnel_length + length) / tunnel_time ∧
    speed = (bridge_length + length) / bridge_time ∧
    speed = 25 ∧
    length = 240 := by sorry

end NUMINAMATH_CALUDE_train_speed_and_length_l3580_358019


namespace NUMINAMATH_CALUDE_mrs_lim_milk_revenue_l3580_358051

/-- Calculates the revenue from milk sales given the conditions of Mrs. Lim's milk production and sales --/
theorem mrs_lim_milk_revenue :
  let yesterday_morning : ℕ := 68
  let yesterday_evening : ℕ := 82
  let this_morning_difference : ℕ := 18
  let remaining_milk : ℕ := 24
  let price_per_gallon : ℚ := 7/2

  let this_morning : ℕ := yesterday_morning - this_morning_difference
  let total_milk : ℕ := yesterday_morning + yesterday_evening + this_morning
  let sold_milk : ℕ := total_milk - remaining_milk
  let revenue : ℚ := price_per_gallon * sold_milk

  revenue = 616 := by sorry

end NUMINAMATH_CALUDE_mrs_lim_milk_revenue_l3580_358051


namespace NUMINAMATH_CALUDE_inspection_arrangements_l3580_358091

/-- Represents the number of liberal arts classes -/
def liberal_arts_classes : ℕ := 2

/-- Represents the number of science classes -/
def science_classes : ℕ := 4

/-- Represents the total number of classes -/
def total_classes : ℕ := liberal_arts_classes + science_classes

/-- Represents the number of ways to choose inspectors from science classes for liberal arts classes -/
def science_to_liberal_arts : ℕ := science_classes * (science_classes - 1)

/-- Represents the number of ways to arrange inspections within science classes -/
def science_arrangements : ℕ := 
  liberal_arts_classes * (liberal_arts_classes - 1) * (science_classes - 2) * (science_classes - 3) +
  liberal_arts_classes * (liberal_arts_classes - 1) +
  liberal_arts_classes * liberal_arts_classes * (science_classes - 2)

/-- The main theorem stating the total number of inspection arrangements -/
theorem inspection_arrangements : 
  science_to_liberal_arts * science_arrangements = 168 := by
  sorry

end NUMINAMATH_CALUDE_inspection_arrangements_l3580_358091


namespace NUMINAMATH_CALUDE_calculation_proof_l3580_358082

theorem calculation_proof : 
  (3 + 1 / 117) * (4 + 1 / 119) - (1 + 116 / 117) * (5 + 118 / 119) - 5 / 119 = 10 / 117 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3580_358082


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l3580_358011

theorem negative_fractions_comparison : -1/3 < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l3580_358011


namespace NUMINAMATH_CALUDE_total_books_is_283_l3580_358081

/-- The number of books borrowed on a given day -/
def books_borrowed (day : Nat) : Nat :=
  match day with
  | 1 => 40  -- Monday
  | 2 => 42  -- Tuesday
  | 3 => 44  -- Wednesday
  | 4 => 46  -- Thursday
  | 5 => 64  -- Friday
  | _ => 0   -- Weekend (handled separately)

/-- The total number of books borrowed during weekdays -/
def weekday_total : Nat :=
  (List.range 5).map books_borrowed |>.sum

/-- The number of books borrowed during the weekend -/
def weekend_books : Nat :=
  (weekday_total / 10) * 2

/-- The total number of books borrowed over the week -/
def total_books : Nat :=
  weekday_total + weekend_books

theorem total_books_is_283 : total_books = 283 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_283_l3580_358081


namespace NUMINAMATH_CALUDE_existence_of_representation_l3580_358041

theorem existence_of_representation (m : ℤ) :
  ∃ (a b k : ℤ), Odd a ∧ Odd b ∧ k ≥ 0 ∧ 2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_representation_l3580_358041


namespace NUMINAMATH_CALUDE_certain_number_value_l3580_358083

theorem certain_number_value : ∃ x : ℝ, 
  (x + 40 + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l3580_358083
