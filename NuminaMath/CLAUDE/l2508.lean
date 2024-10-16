import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l2508_250865

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 3) :
  (9/5 ≤ a^2 + b^2 ∧ a^2 + b^2 < 9) ∧ a^3*b + 4*a*b^3 ≤ 81/16 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2508_250865


namespace NUMINAMATH_CALUDE_circle_C2_equation_l2508_250820

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define circle C1
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define circle C2
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

-- Define symmetry relation
def symmetric_point (x y x' y' : ℝ) : Prop :=
  line_of_symmetry ((x + x') / 2) ((y + y') / 2) ∧
  (x - x')^2 + (y - y')^2 = 2 * ((x - y - 1)^2)

-- Theorem statement
theorem circle_C2_equation :
  ∀ x y : ℝ, circle_C2 x y ↔
  ∃ x' y' : ℝ, circle_C1 x' y' ∧ symmetric_point x y x' y' :=
sorry

end NUMINAMATH_CALUDE_circle_C2_equation_l2508_250820


namespace NUMINAMATH_CALUDE_sqrt_simplification_l2508_250886

theorem sqrt_simplification : Real.sqrt 32 + Real.sqrt 8 - Real.sqrt 50 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l2508_250886


namespace NUMINAMATH_CALUDE_yellow_parrots_count_l2508_250823

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) (yellow_fraction : ℚ) : 
  total = 120 →
  red_fraction = 2/3 →
  yellow_fraction = 1 - red_fraction →
  (yellow_fraction * total : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_yellow_parrots_count_l2508_250823


namespace NUMINAMATH_CALUDE_function_determination_l2508_250859

/-- A continuous monotonic function satisfying the given inequality is x + 1 -/
theorem function_determination (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_mono : Monotone f) 
  (h_init : f 0 = 1) 
  (h_ineq : ∀ x y : ℝ, f (x + y) ≥ f x * f y - f (x * y) + 1) : 
  ∀ x : ℝ, f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_determination_l2508_250859


namespace NUMINAMATH_CALUDE_rectangle_area_divisible_by_12_l2508_250873

theorem rectangle_area_divisible_by_12 (x y z : ℤ) 
  (h : x^2 + y^2 = z^2) : 
  ∃ k : ℤ, x * y = 12 * k := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_divisible_by_12_l2508_250873


namespace NUMINAMATH_CALUDE_greatest_valid_integer_l2508_250863

def is_valid (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 18 = 6

theorem greatest_valid_integer : 
  (∀ m, is_valid m → m ≤ 174) ∧ is_valid 174 := by sorry

end NUMINAMATH_CALUDE_greatest_valid_integer_l2508_250863


namespace NUMINAMATH_CALUDE_hyperbola_focus_l2508_250851

/-- Given a hyperbola with equation ((x-5)^2)/7^2 - ((y-20)^2)/15^2 = 1,
    the focus with the larger x-coordinate has coordinates (5 + √274, 20) -/
theorem hyperbola_focus (x y : ℝ) :
  ((x - 5)^2 / 7^2) - ((y - 20)^2 / 15^2) = 1 →
  ∃ (f_x f_y : ℝ), f_x > 5 ∧ f_y = 20 ∧ f_x = 5 + Real.sqrt 274 ∧
  ∀ (x' y' : ℝ), ((x' - 5)^2 / 7^2) - ((y' - 20)^2 / 15^2) = 1 →
  (x' - 5)^2 / 7^2 + (y' - 20)^2 / 15^2 = (x' - f_x)^2 / 7^2 + (y' - f_y)^2 / 15^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l2508_250851


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l2508_250801

theorem percentage_increase_proof (initial_earnings new_earnings : ℝ) 
  (h1 : initial_earnings = 60)
  (h2 : new_earnings = 110) :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 83.33 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l2508_250801


namespace NUMINAMATH_CALUDE_sugar_amount_proof_l2508_250864

/-- The price of a kilogram of sugar in dollars -/
def sugar_price : ℝ := 1.50

/-- The number of kilograms of sugar bought -/
def sugar_bought : ℝ := 2

/-- The price of a kilogram of salt in dollars -/
def salt_price : ℝ := (5 - 3 * sugar_price)

theorem sugar_amount_proof :
  sugar_bought * sugar_price + 5 * salt_price = 5.50 ∧
  3 * sugar_price + salt_price = 5 →
  sugar_bought = 2 :=
by sorry

end NUMINAMATH_CALUDE_sugar_amount_proof_l2508_250864


namespace NUMINAMATH_CALUDE_quadratic_complex_conjugate_roots_l2508_250806

theorem quadratic_complex_conjugate_roots (a b : ℝ) : 
  (∃ x y : ℝ, (Complex.I * x + y) ^ 2 + (6 + Complex.I * a) * (Complex.I * x + y) + (15 + Complex.I * b) = 0 ∧
               (Complex.I * (-x) + y) ^ 2 + (6 + Complex.I * a) * (Complex.I * (-x) + y) + (15 + Complex.I * b) = 0) →
  a = 0 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_complex_conjugate_roots_l2508_250806


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2508_250866

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_first : a 0 = 23)
  (h_last : a 4 = 53) :
  a 2 = 38 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2508_250866


namespace NUMINAMATH_CALUDE_production_line_b_units_l2508_250853

/-- 
Given a factory with three production lines A, B, and C, prove that production line B 
produced 1000 units under the following conditions:
1. The total number of units produced is 3000
2. The number of units sampled from each production line (a, b, c) form an arithmetic sequence
3. The sum of a, b, and c equals the total number of units produced
-/
theorem production_line_b_units (a b c : ℕ) : 
  (a + b + c = 3000) → 
  (2 * b = a + c) → 
  b = 1000 := by
sorry

end NUMINAMATH_CALUDE_production_line_b_units_l2508_250853


namespace NUMINAMATH_CALUDE_green_green_pairs_l2508_250847

/-- Represents the distribution of students and pairs in a math competition. -/
structure Competition :=
  (total_students : ℕ)
  (blue_shirts : ℕ)
  (green_shirts : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ)

/-- The main theorem about the number of green-green pairs in the competition. -/
theorem green_green_pairs (c : Competition) 
  (h1 : c.total_students = 150)
  (h2 : c.blue_shirts = 68)
  (h3 : c.green_shirts = 82)
  (h4 : c.total_pairs = 75)
  (h5 : c.blue_blue_pairs = 30)
  (h6 : c.total_students = c.blue_shirts + c.green_shirts)
  (h7 : c.total_students = 2 * c.total_pairs) :
  ∃ (green_green_pairs : ℕ), green_green_pairs = 37 ∧ 
    c.total_pairs = c.blue_blue_pairs + green_green_pairs + (c.blue_shirts - 2 * c.blue_blue_pairs) :=
sorry

end NUMINAMATH_CALUDE_green_green_pairs_l2508_250847


namespace NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l2508_250892

theorem one_fourth_in_one_eighth : (1 / 8 : ℚ) / (1 / 4 : ℚ) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l2508_250892


namespace NUMINAMATH_CALUDE_parrots_per_cage_l2508_250807

theorem parrots_per_cage (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_birds : ℕ) :
  num_cages = 8 →
  parakeets_per_cage = 7 →
  total_birds = 72 →
  ∃ (parrots_per_cage : ℕ),
    parrots_per_cage * num_cages + parakeets_per_cage * num_cages = total_birds ∧
    parrots_per_cage = 2 :=
by sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l2508_250807


namespace NUMINAMATH_CALUDE_michael_pet_sitting_cost_l2508_250837

-- Define the number of cats and dogs
def num_cats : ℕ := 2
def num_dogs : ℕ := 3

-- Define the cost per animal per night
def cost_per_animal : ℕ := 13

-- Define the total number of animals
def total_animals : ℕ := num_cats + num_dogs

-- State the theorem
theorem michael_pet_sitting_cost :
  total_animals * cost_per_animal = 65 := by
  sorry

end NUMINAMATH_CALUDE_michael_pet_sitting_cost_l2508_250837


namespace NUMINAMATH_CALUDE_largest_valid_marking_l2508_250887

/-- A marking function that assigns a boolean value to each cell in an n × n grid. -/
def Marking (n : ℕ) := Fin n → Fin n → Bool

/-- Predicate that checks if a rectangle contains a marked cell. -/
def ContainsMarkedCell (m : Marking n) (x y w h : Fin n) : Prop :=
  ∃ i j, i < w ∧ j < h ∧ m (x + i) (y + j) = true

/-- Predicate that checks if a marking satisfies the condition for all rectangles. -/
def ValidMarking (n : ℕ) (m : Marking n) : Prop :=
  ∀ x y w h : Fin n, w * h ≥ n → ContainsMarkedCell m x y w h

/-- The main theorem stating that 7 is the largest n for which a valid marking exists. -/
theorem largest_valid_marking :
  (∃ (m : Marking 7), ValidMarking 7 m) ∧
  (∀ n > 7, ¬∃ (m : Marking n), ValidMarking n m) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_marking_l2508_250887


namespace NUMINAMATH_CALUDE_max_b_value_for_divisible_by_55_l2508_250836

/-- Represents a 7-digit number of the form a2b34c -/
structure SevenDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h_a : a < 10
  h_b : b < 10
  h_c : c < 10

/-- Converts a SevenDigitNumber to its numerical value -/
def SevenDigitNumber.toNat (n : SevenDigitNumber) : Nat :=
  n.a * 1000000 + 200000 + n.b * 10000 + 340 + n.c

/-- Checks if a SevenDigitNumber is divisible by 55 -/
def SevenDigitNumber.isDivisibleBy55 (n : SevenDigitNumber) : Prop :=
  n.toNat % 55 = 0

theorem max_b_value_for_divisible_by_55 :
  ∀ n : SevenDigitNumber, n.isDivisibleBy55 → n.b ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_for_divisible_by_55_l2508_250836


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l2508_250884

-- Define the propositions P and q as functions of a
def P (a : ℝ) : Prop := a ≤ -1 ∨ a ≥ 2

def q (a : ℝ) : Prop := a > 3

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 3)

-- Theorem statement
theorem range_of_a_theorem :
  ∀ a : ℝ, (¬(P a ∧ q a) ∧ (P a ∨ q a)) → range_of_a a :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l2508_250884


namespace NUMINAMATH_CALUDE_fluorescent_tubes_count_l2508_250831

theorem fluorescent_tubes_count :
  ∀ (x y : ℕ),
  x + y = 13 →
  x / 3 + y / 2 = 5 →
  x = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_fluorescent_tubes_count_l2508_250831


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l2508_250802

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, 
  13 * n = 104 ∧ 
  104 ≥ 100 ∧
  104 < 1000 ∧
  ∀ m : ℕ, (13 * m ≥ 100 ∧ 13 * m < 1000) → 13 * m ≥ 104 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l2508_250802


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2508_250857

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (where a > 0, b > 0),
    if one of its asymptotes is tangent to the curve y = √(x - 1),
    then its eccentricity is √5/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y = b / a * x ∧ y = Real.sqrt (x - 1) ∧
   (∀ (x' y' : ℝ), y' = b / a * x' → y' ≠ Real.sqrt (x' - 1) ∨ (x' = x ∧ y' = y))) →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2508_250857


namespace NUMINAMATH_CALUDE_perimeter_of_figure_c_l2508_250890

/-- Given a large rectangle made up of 20 identical small rectangles,
    this theorem proves that if the perimeter of figure A (6x2 small rectangles)
    and figure B (4x6 small rectangles) are both 56 cm,
    then the perimeter of figure C (2x6 small rectangles) is 40 cm. -/
theorem perimeter_of_figure_c (x y : ℝ) 
  (h1 : 6 * x + 2 * y = 56)  -- Perimeter of figure A
  (h2 : 4 * x + 6 * y = 56)  -- Perimeter of figure B
  : 2 * x + 6 * y = 40 :=    -- Perimeter of figure C
by sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_c_l2508_250890


namespace NUMINAMATH_CALUDE_quadratic_sum_l2508_250830

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (QuadraticFunction a b c (-3) = 0) →
  (QuadraticFunction a b c 5 = 0) →
  (∀ x, QuadraticFunction a b c x ≥ -36) →
  (∃ x, QuadraticFunction a b c x = -36) →
  a + b + c = -36 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2508_250830


namespace NUMINAMATH_CALUDE_johnson_calls_l2508_250839

def days_in_year : ℕ := 365

def call_frequencies : List ℕ := [2, 3, 6, 7]

/-- 
Calculates the number of days in a year where no calls are received, 
given a list of call frequencies (in days) for each grandchild.
-/
def days_without_calls (frequencies : List ℕ) (total_days : ℕ) : ℕ :=
  sorry

theorem johnson_calls : 
  days_without_calls call_frequencies days_in_year = 61 := by sorry

end NUMINAMATH_CALUDE_johnson_calls_l2508_250839


namespace NUMINAMATH_CALUDE_six_people_arrangement_l2508_250824

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

def permutations (n : ℕ) (k : ℕ) : ℕ := 
  if k > n then 0
  else factorial n / factorial (n - k)

theorem six_people_arrangement : 
  let total_arrangements := permutations 6 6
  let a_head_b_tail := permutations 4 4
  let a_head_b_not_tail := permutations 4 1 * permutations 4 4
  let a_not_head_b_tail := permutations 4 1 * permutations 4 4
  total_arrangements - a_head_b_tail - a_head_b_not_tail - a_not_head_b_tail = 504 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l2508_250824


namespace NUMINAMATH_CALUDE_greatest_common_factor_72_180_270_l2508_250841

theorem greatest_common_factor_72_180_270 : Nat.gcd 72 (Nat.gcd 180 270) = 18 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_72_180_270_l2508_250841


namespace NUMINAMATH_CALUDE_distance_between_places_l2508_250814

/-- The distance between places A and B in kilometers -/
def distance : ℝ := 150

/-- The speed of bicycling in kilometers per hour -/
def bicycle_speed : ℝ := 15

/-- The speed of walking in kilometers per hour -/
def walking_speed : ℝ := 5

/-- The time difference between return trip and going trip in hours -/
def time_difference : ℝ := 2

theorem distance_between_places : 
  ∃ (return_time : ℝ),
    (distance / 2 / bicycle_speed + distance / 2 / walking_speed = return_time - time_difference) ∧
    (distance = return_time / 3 * bicycle_speed + 2 * return_time / 3 * walking_speed) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_places_l2508_250814


namespace NUMINAMATH_CALUDE_incorrect_spelling_probability_incorrect_spelling_probability_is_59_60_l2508_250805

/-- The probability of spelling "theer" incorrectly -/
theorem incorrect_spelling_probability : ℚ :=
  let total_letters : ℕ := 5
  let repeated_letter : ℕ := 2
  let distinct_letters : ℕ := 3
  let total_arrangements : ℕ := (Nat.choose total_letters repeated_letter) * (Nat.factorial distinct_letters)
  let correct_arrangements : ℕ := 1
  (total_arrangements - correct_arrangements : ℚ) / total_arrangements

/-- Proof that the probability of spelling "theer" incorrectly is 59/60 -/
theorem incorrect_spelling_probability_is_59_60 : 
  incorrect_spelling_probability = 59 / 60 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_spelling_probability_incorrect_spelling_probability_is_59_60_l2508_250805


namespace NUMINAMATH_CALUDE_value_of_M_l2508_250848

theorem value_of_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1500) ∧ (M = 4500) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l2508_250848


namespace NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l2508_250811

/-- The expected number of boy-girl adjacencies in a random arrangement of boys and girls -/
theorem expected_boy_girl_adjacencies
  (num_boys : ℕ)
  (num_girls : ℕ)
  (total : ℕ)
  (h_total : total = num_boys + num_girls)
  (h_boys : num_boys = 7)
  (h_girls : num_girls = 13) :
  (total - 1 : ℚ) * (num_boys * num_girls : ℚ) / (total * (total - 1) / 2 : ℚ) = 91 / 10 :=
sorry

end NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l2508_250811


namespace NUMINAMATH_CALUDE_stating_weaver_production_increase_l2508_250843

/-- Represents the daily increase in fabric production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the initial daily production -/
def initial_production : ℕ := 5

/-- Represents the number of days -/
def days : ℕ := 30

/-- Represents the total production over the given period -/
def total_production : ℕ := 390

/-- 
Theorem stating that given the initial production and total production over a period,
the daily increase in production is as calculated.
-/
theorem weaver_production_increase : 
  initial_production * days + (days * (days - 1) / 2) * daily_increase = total_production := by
  sorry


end NUMINAMATH_CALUDE_stating_weaver_production_increase_l2508_250843


namespace NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l2508_250835

theorem negative_x_squared_times_x_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l2508_250835


namespace NUMINAMATH_CALUDE_f_difference_512_256_l2508_250875

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Define the f function as described in the problem
def f (n : ℕ) : ℚ := (sum_of_divisors n : ℚ) / n

-- Theorem statement
theorem f_difference_512_256 : f 512 - f 256 = 1 / 512 := by sorry

end NUMINAMATH_CALUDE_f_difference_512_256_l2508_250875


namespace NUMINAMATH_CALUDE_nancy_added_pencils_l2508_250813

/-- The number of pencils Nancy placed in the drawer -/
def pencils_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Nancy added 45 pencils to the drawer -/
theorem nancy_added_pencils : pencils_added 27 72 = 45 := by
  sorry

end NUMINAMATH_CALUDE_nancy_added_pencils_l2508_250813


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2508_250882

theorem trigonometric_identity : 4 * Real.sin (20 * π / 180) + Real.tan (20 * π / 180) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2508_250882


namespace NUMINAMATH_CALUDE_first_three_digits_of_quotient_l2508_250844

/-- The dividend a as a real number -/
def a : ℝ := 0.1234567891011

/-- The divisor b as a real number -/
def b : ℝ := 0.51504948

/-- Theorem stating that the first three digits of a/b are 0.239 -/
theorem first_three_digits_of_quotient (ha : a > 0) (hb : b > 0) :
  0.239 * b ≤ a ∧ a < 0.24 * b :=
sorry

end NUMINAMATH_CALUDE_first_three_digits_of_quotient_l2508_250844


namespace NUMINAMATH_CALUDE_max_large_chips_l2508_250819

/-- The smallest composite number -/
def smallest_composite : ℕ := 4

/-- Represents the problem of finding the maximum number of large chips -/
def chip_problem (total : ℕ) (small : ℕ) (large : ℕ) : Prop :=
  total = 60 ∧
  small + large = total ∧
  ∃ c : ℕ, c ≥ smallest_composite ∧ small = large + c

/-- The theorem stating the maximum number of large chips -/
theorem max_large_chips :
  ∀ total small large,
  chip_problem total small large →
  large ≤ 28 :=
sorry

end NUMINAMATH_CALUDE_max_large_chips_l2508_250819


namespace NUMINAMATH_CALUDE_internal_tangent_circles_locus_l2508_250869

/-- Two circles are tangent internally if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  Real.sqrt ((c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2) = r₁ - r₂

/-- The locus of a point is a circle if its distance from a fixed point is constant -/
def is_circle_locus (center : ℝ × ℝ) (radius : ℝ) (locus : Set (ℝ × ℝ)) : Prop :=
  ∀ p ∈ locus, Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) = radius

theorem internal_tangent_circles_locus (O₁ : ℝ × ℝ) :
  let r₁ : ℝ := 7
  let r₂ : ℝ := 4
  let locus : Set (ℝ × ℝ) := {O₂ | ∃ (O₂ : ℝ × ℝ), internally_tangent O₁ O₂ r₁ r₂}
  is_circle_locus O₁ 3 locus := by
  sorry

end NUMINAMATH_CALUDE_internal_tangent_circles_locus_l2508_250869


namespace NUMINAMATH_CALUDE_stream_speed_l2508_250845

/-- Proves that the speed of a stream is 135/14 km/h given the conditions of a boat's travel --/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : downstream_distance = 150)
  (h2 : upstream_distance = 75)
  (h3 : downstream_time = 5)
  (h4 : upstream_time = 7) : 
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 135 / 14 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2508_250845


namespace NUMINAMATH_CALUDE_product_and_sum_of_squares_l2508_250868

theorem product_and_sum_of_squares (x y : ℝ) :
  x * y = 120 ∧ x^2 + y^2 = 289 → x + y = 22 := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_of_squares_l2508_250868


namespace NUMINAMATH_CALUDE_button_probability_l2508_250889

/-- Represents a jar containing buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a red button from a jar -/
def redProbability (j : Jar) : ℚ :=
  j.red / (j.red + j.blue)

theorem button_probability (initialJarA initialJarB finalJarA finalJarB : Jar) :
  initialJarA.red = 5 →
  initialJarA.blue = 10 →
  initialJarB.red = 0 →
  initialJarB.blue = 0 →
  finalJarA.red + finalJarB.red = initialJarA.red →
  finalJarA.blue + finalJarB.blue = initialJarA.blue →
  finalJarB.red = finalJarB.blue / 2 →
  finalJarA.red + finalJarA.blue = (3 * (initialJarA.red + initialJarA.blue)) / 5 →
  redProbability finalJarA = 1/3 ∧ redProbability finalJarB = 1/3 ∧
  redProbability finalJarA * redProbability finalJarB = 1/9 := by
  sorry


end NUMINAMATH_CALUDE_button_probability_l2508_250889


namespace NUMINAMATH_CALUDE_total_spent_is_30_40_l2508_250828

/-- Represents the store's inventory and pricing --/
structure Store where
  barrette_price : ℝ
  comb_price : ℝ
  hairband_price : ℝ
  hair_ties_price : ℝ

/-- Represents a customer's purchase --/
structure Purchase where
  barrettes : ℕ
  combs : ℕ
  hairbands : ℕ
  hair_ties : ℕ

/-- Calculates the total cost of a purchase before discount and tax --/
def purchase_cost (s : Store) (p : Purchase) : ℝ :=
  s.barrette_price * p.barrettes +
  s.comb_price * p.combs +
  s.hairband_price * p.hairbands +
  s.hair_ties_price * p.hair_ties

/-- Applies discount if applicable --/
def apply_discount (cost : ℝ) (item_count : ℕ) : ℝ :=
  if item_count > 5 then cost * 0.85 else cost

/-- Applies sales tax --/
def apply_tax (cost : ℝ) : ℝ :=
  cost * 1.08

/-- Calculates the final cost of a purchase after discount and tax --/
def final_cost (s : Store) (p : Purchase) : ℝ :=
  let initial_cost := purchase_cost s p
  let item_count := p.barrettes + p.combs + p.hairbands + p.hair_ties
  let discounted_cost := apply_discount initial_cost item_count
  apply_tax discounted_cost

/-- The main theorem --/
theorem total_spent_is_30_40 (s : Store) (k_purchase c_purchase : Purchase) :
  s.barrette_price = 4 ∧
  s.comb_price = 2 ∧
  s.hairband_price = 3 ∧
  s.hair_ties_price = 2.5 ∧
  k_purchase = { barrettes := 1, combs := 1, hairbands := 2, hair_ties := 0 } ∧
  c_purchase = { barrettes := 3, combs := 1, hairbands := 0, hair_ties := 2 } →
  final_cost s k_purchase + final_cost s c_purchase = 30.40 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_30_40_l2508_250828


namespace NUMINAMATH_CALUDE_employee_recorder_price_l2508_250815

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the markup percentage
def markup_percentage : ℝ := 0.20

-- Define the employee discount percentage
def employee_discount_percentage : ℝ := 0.05

-- Define the retail price calculation
def retail_price : ℝ := wholesale_cost * (1 + markup_percentage)

-- Define the employee price calculation
def employee_price : ℝ := retail_price * (1 - employee_discount_percentage)

-- Theorem statement
theorem employee_recorder_price : employee_price = 228 := by
  sorry

end NUMINAMATH_CALUDE_employee_recorder_price_l2508_250815


namespace NUMINAMATH_CALUDE_oil_price_reduction_l2508_250817

/-- Proves that given a 10% reduction in the price of oil, if a housewife can obtain 6 kgs more 
    for Rs. 900 after the reduction, then the reduced price per kg of oil is Rs. 15. -/
theorem oil_price_reduction (original_price : ℝ) : 
  let reduced_price := original_price * 0.9
  let original_quantity := 900 / original_price
  let new_quantity := 900 / reduced_price
  new_quantity = original_quantity + 6 →
  reduced_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l2508_250817


namespace NUMINAMATH_CALUDE_interesting_trapezoid_area_interesting_trapezoid_area_range_l2508_250895

/-- An interesting isosceles trapezoid inscribed in a unit square. -/
structure InterestingTrapezoid where
  /-- Parameter determining the position of the trapezoid's vertices. -/
  a : ℝ
  /-- The parameter a is between 0 and 1/2 inclusive. -/
  h_a_range : 0 ≤ a ∧ a ≤ 1/2

/-- The vertices of the trapezoid. -/
def vertices (t : InterestingTrapezoid) : Fin 4 → ℝ × ℝ
  | 0 => (t.a, 0)
  | 1 => (1, t.a)
  | 2 => (1 - t.a, 1)
  | 3 => (0, 1 - t.a)

/-- The area of an interesting isosceles trapezoid. -/
def area (t : InterestingTrapezoid) : ℝ := 1 - 2 * t.a

/-- Theorem: The area of an interesting isosceles trapezoid is 1 - 2a. -/
theorem interesting_trapezoid_area (t : InterestingTrapezoid) :
  area t = 1 - 2 * t.a :=
by sorry

/-- Theorem: The area of an interesting isosceles trapezoid is between 0 and 1 inclusive. -/
theorem interesting_trapezoid_area_range (t : InterestingTrapezoid) :
  0 ≤ area t ∧ area t ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_interesting_trapezoid_area_interesting_trapezoid_area_range_l2508_250895


namespace NUMINAMATH_CALUDE_canal_meeting_participants_l2508_250838

theorem canal_meeting_participants (total : Nat) (greetings : Nat) : total = 12 ∧ greetings = 31 →
  ∃ (egyptians panamanians : Nat),
    egyptians + panamanians = total ∧
    egyptians > panamanians ∧
    egyptians * (egyptians - 1) / 2 + panamanians * (panamanians - 1) / 2 = greetings ∧
    egyptians = 7 ∧
    panamanians = 5 := by
  sorry

end NUMINAMATH_CALUDE_canal_meeting_participants_l2508_250838


namespace NUMINAMATH_CALUDE_expand_polynomial_l2508_250899

theorem expand_polynomial (x : ℝ) : (x + 3) * (x^2 - x + 4) = x^3 + 2*x^2 + x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2508_250899


namespace NUMINAMATH_CALUDE_circles_tangent_m_values_l2508_250871

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y m : ℝ) : Prop := x^2 + y^2 - 8*x + 8*y + m = 0

-- Define tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y m ∧
  (∀ (x' y' : ℝ), C₁ x' y' ∧ C₂ x' y' m → (x' = x ∧ y' = y))

-- Theorem statement
theorem circles_tangent_m_values :
  ∀ m : ℝ, are_tangent m ↔ (m = -4 ∨ m = 16) :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_m_values_l2508_250871


namespace NUMINAMATH_CALUDE_x_gt_y_iff_exp_and_cbrt_l2508_250800

theorem x_gt_y_iff_exp_and_cbrt (x y : ℝ) : 
  x > y ↔ (Real.exp x > Real.exp y ∧ x^(1/3) > y^(1/3)) :=
sorry

end NUMINAMATH_CALUDE_x_gt_y_iff_exp_and_cbrt_l2508_250800


namespace NUMINAMATH_CALUDE_root_sum_power_property_l2508_250878

theorem root_sum_power_property (x₁ x₂ : ℂ) (n : ℤ) : 
  x₁^2 - 6*x₁ + 1 = 0 → 
  x₂^2 - 6*x₂ + 1 = 0 → 
  (∃ m : ℤ, x₁^n + x₂^n = m) ∧ 
  ¬(∃ k : ℤ, x₁^n + x₂^n = 5*k) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_power_property_l2508_250878


namespace NUMINAMATH_CALUDE_betty_total_items_betty_total_cost_l2508_250827

/-- The number of slippers Betty ordered -/
def slippers : ℕ := 6

/-- The number of lipsticks Betty ordered -/
def lipsticks : ℕ := 4

/-- The number of hair colors Betty ordered -/
def hair_colors : ℕ := 8

/-- The cost of each slipper -/
def slipper_cost : ℚ := 5/2

/-- The cost of each lipstick -/
def lipstick_cost : ℚ := 5/4

/-- The cost of each hair color -/
def hair_color_cost : ℚ := 3

/-- The total amount Betty paid -/
def total_paid : ℚ := 44

/-- Theorem stating that Betty ordered 18 items in total -/
theorem betty_total_items : slippers + lipsticks + hair_colors = 18 := by
  sorry

/-- Theorem verifying the total cost matches the amount Betty paid -/
theorem betty_total_cost : 
  slippers * slipper_cost + lipsticks * lipstick_cost + hair_colors * hair_color_cost = total_paid := by
  sorry

end NUMINAMATH_CALUDE_betty_total_items_betty_total_cost_l2508_250827


namespace NUMINAMATH_CALUDE_trailing_zeroes_sum_factorials_l2508_250810

def trailing_zeroes (n : ℕ) : ℕ := sorry

def factorial (n : ℕ) : ℕ := sorry

theorem trailing_zeroes_sum_factorials :
  trailing_zeroes (factorial 60 + factorial 120) = trailing_zeroes (factorial 60) :=
sorry

end NUMINAMATH_CALUDE_trailing_zeroes_sum_factorials_l2508_250810


namespace NUMINAMATH_CALUDE_least_possible_smallest_integer_l2508_250893

theorem least_possible_smallest_integer
  (a b c d : ℤ) -- Four integers
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) -- They are distinct
  (h_average : (a + b + c + d) / 4 = 70) -- Their average is 70
  (h_largest : d = 90 ∧ d ≥ a ∧ d ≥ b ∧ d ≥ c) -- d is the largest and equals 90
  : a ≥ 184 -- The smallest integer is at least 184
:= by sorry

end NUMINAMATH_CALUDE_least_possible_smallest_integer_l2508_250893


namespace NUMINAMATH_CALUDE_rajdhani_speed_calculation_l2508_250833

/-- The speed of Bombay Express in km/h -/
def bombay_speed : ℝ := 60

/-- The time difference between the departures of the two trains in hours -/
def time_difference : ℝ := 2

/-- The distance at which the two trains meet in km -/
def meeting_distance : ℝ := 480

/-- The speed of Rajdhani Express in km/h -/
def rajdhani_speed : ℝ := 80

theorem rajdhani_speed_calculation :
  let distance_covered_by_bombay : ℝ := bombay_speed * time_difference
  let remaining_distance : ℝ := meeting_distance - distance_covered_by_bombay
  let time_to_meet : ℝ := remaining_distance / bombay_speed
  rajdhani_speed = meeting_distance / time_to_meet :=
by sorry

end NUMINAMATH_CALUDE_rajdhani_speed_calculation_l2508_250833


namespace NUMINAMATH_CALUDE_circle_max_sum_l2508_250870

theorem circle_max_sum :
  ∀ x y : ℤ, x^2 + y^2 = 16 → x + y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_sum_l2508_250870


namespace NUMINAMATH_CALUDE_det_specific_matrix_l2508_250821

theorem det_specific_matrix : 
  Matrix.det !![2, 0, 4; 3, -1, 5; 1, 2, 3] = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l2508_250821


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l2508_250855

theorem square_perimeter_sum (x y : ℝ) (h1 : x^2 + y^2 = 130) (h2 : x^2 - y^2 = 42) :
  4*x + 4*y = 4*Real.sqrt 86 + 8*Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l2508_250855


namespace NUMINAMATH_CALUDE_f_zero_at_negative_one_f_negative_iff_a_greater_than_negative_one_l2508_250885

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - a) * abs x + b

-- Part 1: Prove that when a = 2 and b = 3, the only zero of f is x = -1
theorem f_zero_at_negative_one :
  ∃! x : ℝ, f 2 3 x = 0 ∧ x = -1 := by sorry

-- Part 2: Prove that when b = -2, f(x) < 0 for all x ∈ [-1, 1] if and only if a > -1
theorem f_negative_iff_a_greater_than_negative_one :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ [-1, 1] → f a (-2) x < 0) ↔ a > -1 := by sorry

end NUMINAMATH_CALUDE_f_zero_at_negative_one_f_negative_iff_a_greater_than_negative_one_l2508_250885


namespace NUMINAMATH_CALUDE_michael_quiz_score_l2508_250816

theorem michael_quiz_score (existing_scores : List ℕ) (target_mean : ℕ) (required_score : ℕ) : 
  existing_scores = [84, 78, 95, 88, 91] →
  target_mean = 90 →
  required_score = 104 →
  (existing_scores.sum + required_score) / (existing_scores.length + 1) = target_mean :=
by sorry

end NUMINAMATH_CALUDE_michael_quiz_score_l2508_250816


namespace NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l2508_250804

theorem polar_to_rectangular_transformation (x y : ℝ) (h : x = 12 ∧ y = 5) :
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  (r^2 * Real.cos (3 * θ), r^2 * Real.sin (3 * θ)) = (-494004 / 2197, 4441555 / 2197) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l2508_250804


namespace NUMINAMATH_CALUDE_total_spending_calculation_l2508_250849

def shirt_price : ℝ := 13.04
def shirt_tax_rate : ℝ := 0.07
def jacket_price : ℝ := 12.27
def jacket_tax_rate : ℝ := 0.085
def scarf_price : ℝ := 7.90
def hat_price : ℝ := 9.13
def scarf_hat_tax_rate : ℝ := 0.065

def total_cost (price : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 + tax_rate)

theorem total_spending_calculation :
  total_cost shirt_price shirt_tax_rate +
  total_cost jacket_price jacket_tax_rate +
  total_cost scarf_price scarf_hat_tax_rate +
  total_cost hat_price scarf_hat_tax_rate =
  45.4027 := by sorry

end NUMINAMATH_CALUDE_total_spending_calculation_l2508_250849


namespace NUMINAMATH_CALUDE_orange_bucket_theorem_l2508_250881

/-- Represents the number of oranges in each bucket -/
structure OrangeBuckets where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of oranges in all buckets -/
def total_oranges (buckets : OrangeBuckets) : ℕ :=
  buckets.first + buckets.second + buckets.third

/-- Theorem stating the total number of oranges in the given conditions -/
theorem orange_bucket_theorem (buckets : OrangeBuckets) 
  (h1 : buckets.first = 22)
  (h2 : buckets.second = buckets.first + 17)
  (h3 : buckets.third = buckets.second - 11) :
  total_oranges buckets = 89 := by
  sorry

#check orange_bucket_theorem

end NUMINAMATH_CALUDE_orange_bucket_theorem_l2508_250881


namespace NUMINAMATH_CALUDE_university_weighted_average_age_l2508_250822

/-- Calculates the weighted average age of a university given the number of arts and technical classes,
    their respective average ages, and assuming each class has the same number of students. -/
theorem university_weighted_average_age
  (num_arts_classes : ℕ)
  (num_tech_classes : ℕ)
  (avg_age_arts : ℝ)
  (avg_age_tech : ℝ)
  (h1 : num_arts_classes = 8)
  (h2 : num_tech_classes = 5)
  (h3 : avg_age_arts = 21)
  (h4 : avg_age_tech = 18) :
  (num_arts_classes * avg_age_arts + num_tech_classes * avg_age_tech) / (num_arts_classes + num_tech_classes) = 258 / 13 := by
sorry

end NUMINAMATH_CALUDE_university_weighted_average_age_l2508_250822


namespace NUMINAMATH_CALUDE_projection_difference_l2508_250818

/-- Represents a projection type -/
inductive ProjectionType
| Parallel
| Central

/-- Represents the behavior of projection lines -/
inductive ProjectionLineBehavior
| Parallel
| Converging

/-- Defines the projection line behavior for a given projection type -/
def projectionLineBehavior (p : ProjectionType) : ProjectionLineBehavior :=
  match p with
  | ProjectionType.Parallel => ProjectionLineBehavior.Parallel
  | ProjectionType.Central => ProjectionLineBehavior.Converging

/-- Theorem stating the difference between parallel and central projections -/
theorem projection_difference :
  ∀ (p : ProjectionType),
    (p = ProjectionType.Parallel ∧ projectionLineBehavior p = ProjectionLineBehavior.Parallel) ∨
    (p = ProjectionType.Central ∧ projectionLineBehavior p = ProjectionLineBehavior.Converging) :=
by sorry

end NUMINAMATH_CALUDE_projection_difference_l2508_250818


namespace NUMINAMATH_CALUDE_fraction_multiplication_one_half_of_one_third_of_one_sixth_of_72_l2508_250809

theorem fraction_multiplication (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem one_half_of_one_third_of_one_sixth_of_72 :
  (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_one_half_of_one_third_of_one_sixth_of_72_l2508_250809


namespace NUMINAMATH_CALUDE_thirtythree_by_thirtythree_black_count_l2508_250894

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard where
  size : ℕ
  blackInCorners : Bool

/-- Counts the number of black squares on a checkerboard -/
def countBlackSquares (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem: A 33x33 checkerboard with black corners has 545 black squares -/
theorem thirtythree_by_thirtythree_black_count :
  ∀ (board : Checkerboard),
    board.size = 33 ∧ board.blackInCorners = true →
    countBlackSquares board = 545 :=
by sorry

end NUMINAMATH_CALUDE_thirtythree_by_thirtythree_black_count_l2508_250894


namespace NUMINAMATH_CALUDE_sammy_janine_bottle_cap_difference_l2508_250862

/-- Proof that Sammy has 2 more bottle caps than Janine -/
theorem sammy_janine_bottle_cap_difference :
  ∀ (sammy janine billie : ℕ),
    sammy > janine →
    janine = 3 * billie →
    billie = 2 →
    sammy = 8 →
    sammy - janine = 2 := by
  sorry

end NUMINAMATH_CALUDE_sammy_janine_bottle_cap_difference_l2508_250862


namespace NUMINAMATH_CALUDE_product_equals_fraction_l2508_250825

/-- The decimal representation of a real number with digits 1, 4, 5 repeating after the decimal point -/
def repeating_decimal : ℚ := 145 / 999

/-- The product of the repeating decimal and 11 -/
def product : ℚ := 11 * repeating_decimal

theorem product_equals_fraction : product = 1595 / 999 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l2508_250825


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2508_250891

theorem geometric_sequence_middle_term (b : ℝ) (h : b > 0) :
  (∃ s : ℝ, s ≠ 0 ∧ 10 * s = b ∧ b * s = 1/3) → b = Real.sqrt (10/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2508_250891


namespace NUMINAMATH_CALUDE_power_of_negative_square_l2508_250812

theorem power_of_negative_square (a : ℝ) : (-a^2)^3 = -a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l2508_250812


namespace NUMINAMATH_CALUDE_salt_problem_l2508_250840

theorem salt_problem (a x : ℝ) (h : a - x = 2 * (a - 2 * x)) : x = a / 3 := by
  sorry

end NUMINAMATH_CALUDE_salt_problem_l2508_250840


namespace NUMINAMATH_CALUDE_initial_water_percentage_l2508_250860

theorem initial_water_percentage
  (V₁ : ℝ) (V₂ : ℝ) (P_f : ℝ)
  (h₁ : V₁ = 20)
  (h₂ : V₂ = 20)
  (h₃ : P_f = 5)
  : ∃ P_i : ℝ, P_i = 10 ∧ (P_i / 100) * V₁ = (P_f / 100) * (V₁ + V₂) := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l2508_250860


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2508_250898

theorem unique_solution_for_equation : ∃! p n k : ℕ+, 
  Nat.Prime p ∧ 
  k > 1 ∧ 
  (3 : ℕ)^(p : ℕ) + (4 : ℕ)^(p : ℕ) = (n : ℕ)^(k : ℕ) ∧ 
  p = 2 ∧ n = 5 ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2508_250898


namespace NUMINAMATH_CALUDE_john_phone_cost_l2508_250883

theorem john_phone_cost (alan_price : ℝ) (john_percentage : ℝ) : 
  alan_price = 2000 → john_percentage = 0.02 → 
  alan_price * (1 + john_percentage) = 2040 := by
  sorry

end NUMINAMATH_CALUDE_john_phone_cost_l2508_250883


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l2508_250834

/-- Given three positive real numbers X, Y, and Z in the ratio 3:2:6,
    prove that (2X + 3Y) / (5Z - 2X) = 1/2 -/
theorem ratio_fraction_equality (X Y Z : ℝ) (hX : X > 0) (hY : Y > 0) (hZ : Z > 0)
  (h_ratio : X / Y = 3 / 2 ∧ Y / Z = 2 / 6) :
  (2 * X + 3 * Y) / (5 * Z - 2 * X) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l2508_250834


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2508_250872

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I)^2 / z = 1 + Complex.I) : 
  z = -1 - Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2508_250872


namespace NUMINAMATH_CALUDE_correct_num_fruit_baskets_l2508_250867

/-- The number of possible fruit baskets given the conditions -/
def num_fruit_baskets : ℕ := 84

/-- The number of apples available -/
def num_apples : ℕ := 7

/-- The number of oranges available -/
def num_oranges : ℕ := 12

/-- Theorem stating that the number of possible fruit baskets is correct -/
theorem correct_num_fruit_baskets :
  num_fruit_baskets = num_apples * num_oranges :=
by sorry

end NUMINAMATH_CALUDE_correct_num_fruit_baskets_l2508_250867


namespace NUMINAMATH_CALUDE_sky_diving_company_total_amount_l2508_250829

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem sky_diving_company_total_amount :
  individual_bookings + group_bookings - refunds = 26400 := by
  sorry

end NUMINAMATH_CALUDE_sky_diving_company_total_amount_l2508_250829


namespace NUMINAMATH_CALUDE_prime_quadratic_residue_equivalence_l2508_250896

theorem prime_quadratic_residue_equivalence (p : ℕ) (hp : Nat.Prime p) :
  (∃ α : ℕ+, p ∣ α * (α - 1) + 3) ↔ (∃ β : ℕ+, p ∣ β * (β - 1) + 25) := by
  sorry

end NUMINAMATH_CALUDE_prime_quadratic_residue_equivalence_l2508_250896


namespace NUMINAMATH_CALUDE_f_properties_l2508_250879

noncomputable section

-- Define the function f
def f (p : ℝ) (x : ℝ) : ℝ := Real.log (1 - x) + Real.log (p + x)

-- State the theorem
theorem f_properties (p : ℝ) (a : ℝ) (h_p : p > -1) (h_a : 0 < a ∧ a < 1) :
  -- Part 1: Domain of f
  (∀ x, f p x ≠ 0 ↔ -p < x ∧ x < 1) ∧
  -- Part 2: Minimum value of f when p = 1
  (∃ min_val, ∀ x, -a < x ∧ x ≤ a → f 1 x ≥ min_val) ∧
  (f 1 a = Real.log (1 - a^2)) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l2508_250879


namespace NUMINAMATH_CALUDE_maple_trees_after_planting_l2508_250861

/-- The number of maple trees in the park after planting -/
def total_maple_trees (initial_maple_trees planted_maple_trees : ℕ) : ℕ :=
  initial_maple_trees + planted_maple_trees

/-- Theorem: The total number of maple trees after planting is equal to
    the sum of the initial number of maple trees and the number of maple trees being planted -/
theorem maple_trees_after_planting 
  (initial_maple_trees planted_maple_trees : ℕ) : 
  total_maple_trees initial_maple_trees planted_maple_trees = 
  initial_maple_trees + planted_maple_trees := by
  sorry

#eval total_maple_trees 2 9

end NUMINAMATH_CALUDE_maple_trees_after_planting_l2508_250861


namespace NUMINAMATH_CALUDE_years_until_double_age_l2508_250856

/-- Proves the number of years until a man's age is twice his son's age -/
theorem years_until_double_age (son_age : ℕ) (age_difference : ℕ) (years : ℕ) : 
  son_age = 44 → 
  age_difference = 46 → 
  (son_age + age_difference + years) = 2 * (son_age + years) → 
  years = 2 := by
sorry

end NUMINAMATH_CALUDE_years_until_double_age_l2508_250856


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l2508_250880

/-- Proves that the actual distance traveled is 20 km given the conditions -/
theorem actual_distance_traveled (initial_speed time_taken : ℝ) 
  (h1 : initial_speed = 5)
  (h2 : initial_speed * time_taken + 20 = 2 * initial_speed * time_taken) :
  initial_speed * time_taken = 20 := by
  sorry

#check actual_distance_traveled

end NUMINAMATH_CALUDE_actual_distance_traveled_l2508_250880


namespace NUMINAMATH_CALUDE_first_half_total_score_l2508_250808

/-- Represents the scores of a team in a basketball game --/
structure TeamScores where
  q1 : ℝ
  q2 : ℝ
  q3 : ℝ
  q4 : ℝ

/-- The game conditions --/
def GameConditions (alpha : TeamScores) (beta : TeamScores) : Prop :=
  -- Tied after first quarter
  alpha.q1 = beta.q1
  -- Alpha's scores form a geometric sequence
  ∧ ∃ r : ℝ, r > 1 ∧ alpha.q2 = alpha.q1 * r ∧ alpha.q3 = alpha.q2 * r ∧ alpha.q4 = alpha.q3 * r
  -- Beta's scores form an arithmetic sequence
  ∧ ∃ d : ℝ, d > 0 ∧ beta.q2 = beta.q1 + d ∧ beta.q3 = beta.q2 + d ∧ beta.q4 = beta.q3 + d
  -- Alpha won by 3 points
  ∧ alpha.q1 + alpha.q2 + alpha.q3 + alpha.q4 = beta.q1 + beta.q2 + beta.q3 + beta.q4 + 3
  -- No team scored more than 120 points
  ∧ alpha.q1 + alpha.q2 + alpha.q3 + alpha.q4 ≤ 120
  ∧ beta.q1 + beta.q2 + beta.q3 + beta.q4 ≤ 120

/-- The theorem to be proved --/
theorem first_half_total_score (alpha : TeamScores) (beta : TeamScores) 
  (h : GameConditions alpha beta) : 
  alpha.q1 + alpha.q2 + beta.q1 + beta.q2 = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_first_half_total_score_l2508_250808


namespace NUMINAMATH_CALUDE_bicycle_wheel_revolutions_l2508_250888

theorem bicycle_wheel_revolutions 
  (front_radius : ℝ) 
  (back_radius : ℝ) 
  (front_revolutions : ℝ) 
  (h1 : front_radius = 3) 
  (h2 : back_radius = 6 / 12) 
  (h3 : front_revolutions = 150) :
  (2 * Real.pi * front_radius * front_revolutions) / (2 * Real.pi * back_radius) = 900 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheel_revolutions_l2508_250888


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l2508_250826

theorem stratified_sampling_medium_supermarkets 
  (large : ℕ) 
  (medium : ℕ) 
  (small : ℕ) 
  (sample_size : ℕ) 
  (h1 : large = 200) 
  (h2 : medium = 400) 
  (h3 : small = 1400) 
  (h4 : sample_size = 100) :
  (medium : ℚ) * sample_size / (large + medium + small) = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l2508_250826


namespace NUMINAMATH_CALUDE_inequality_implication_l2508_250850

theorem inequality_implication (a b : ℝ) (h : a > b) : a + 2 > b + 1 := by
  sorry

#check inequality_implication

end NUMINAMATH_CALUDE_inequality_implication_l2508_250850


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2508_250876

def f (x : ℝ) := -2 * x^2 + 12 * x - 10

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2508_250876


namespace NUMINAMATH_CALUDE_line_graph_most_suitable_for_aqi_l2508_250874

/-- Represents types of statistical graphs -/
inductive StatGraph
  | LineGraph
  | Histogram
  | BarGraph
  | PieChart

/-- Represents a series of daily AQI values -/
def AQISeries := List Nat

/-- Determines if a graph type is suitable for showing time-based trends -/
def shows_time_trends (graph : StatGraph) : Prop :=
  match graph with
  | StatGraph.LineGraph => true
  | _ => false

/-- Determines if a graph type is suitable for showing continuous data -/
def shows_continuous_data (graph : StatGraph) : Prop :=
  match graph with
  | StatGraph.LineGraph => true
  | _ => false

/-- Theorem: A line graph is the most suitable for describing AQI changes over time -/
theorem line_graph_most_suitable_for_aqi (aqi_data : AQISeries) :
  aqi_data.length = 10 →
  ∃ (g : StatGraph), shows_time_trends g ∧ shows_continuous_data g ∧
  ∀ (g' : StatGraph), (shows_time_trends g' ∧ shows_continuous_data g') → g = g' :=
by sorry

end NUMINAMATH_CALUDE_line_graph_most_suitable_for_aqi_l2508_250874


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2508_250832

theorem sqrt_equation_solution (x : ℚ) :
  (Real.sqrt (6 * x) / Real.sqrt (5 * (x - 2)) = 3) → x = 30 / 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2508_250832


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2508_250854

theorem complex_modulus_equality (n : ℝ) (hn : 0 < n) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 26 → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2508_250854


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2508_250858

/-- A right triangle with medians m₁ and m₂ drawn from the vertices of the acute angles has hypotenuse of length 3√(336/13) when m₁ = 6 and m₂ = √48 -/
theorem right_triangle_hypotenuse (m₁ m₂ : ℝ) (h₁ : m₁ = 6) (h₂ : m₂ = Real.sqrt 48) :
  ∃ (a b c : ℝ), 
    a^2 + b^2 = c^2 ∧  -- right triangle condition
    (b^2 + (3*a/2)^2 = m₁^2) ∧  -- first median condition
    (a^2 + (3*b/2)^2 = m₂^2) ∧  -- second median condition
    c = 3 * Real.sqrt (336/13) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2508_250858


namespace NUMINAMATH_CALUDE_expression_value_l2508_250803

def opposite_numbers (a b : ℝ) : Prop := a = -b ∧ a ≠ 0 ∧ b ≠ 0

def reciprocals (c d : ℝ) : Prop := c * d = 1

def distance_from_one (m : ℝ) : Prop := |m - 1| = 2

theorem expression_value (a b c d m : ℝ) 
  (h1 : opposite_numbers a b) 
  (h2 : reciprocals c d) 
  (h3 : distance_from_one m) : 
  (a + b) * (c / d) + m * c * d + (b / a) = 2 ∨ 
  (a + b) * (c / d) + m * c * d + (b / a) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2508_250803


namespace NUMINAMATH_CALUDE_parallel_perpendicular_lines_l2508_250842

-- Define the lines and points
def l₁ (m : ℝ) := {(x, y) : ℝ × ℝ | (y - m) / (x + 2) = (4 - m) / (m + 2)}
def l₂ := {(x, y) : ℝ × ℝ | 2*x + y - 1 = 0}
def l₃ (n : ℝ) := {(x, y) : ℝ × ℝ | x + n*y + 1 = 0}

def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (m : ℝ) : ℝ × ℝ := (m, 4)

-- Define the theorem
theorem parallel_perpendicular_lines (m n : ℝ) : 
  (A m ∈ l₁ m) → 
  (B m ∈ l₁ m) → 
  (∀ (x y : ℝ), (x, y) ∈ l₁ m ↔ (x, y) ∈ l₂) → 
  (∀ (x y : ℝ), (x, y) ∈ l₂ → (x, y) ∈ l₃ n → x = y) → 
  m + n = -10 := by
  sorry

#check parallel_perpendicular_lines

end NUMINAMATH_CALUDE_parallel_perpendicular_lines_l2508_250842


namespace NUMINAMATH_CALUDE_dangerous_animals_count_l2508_250877

/-- The number of crocodiles pointed out by the teacher -/
def num_crocodiles : ℕ := 22

/-- The number of alligators pointed out by the teacher -/
def num_alligators : ℕ := 23

/-- The number of vipers pointed out by the teacher -/
def num_vipers : ℕ := 5

/-- The total number of dangerous animals pointed out by the teacher -/
def total_dangerous_animals : ℕ := num_crocodiles + num_alligators + num_vipers

theorem dangerous_animals_count : total_dangerous_animals = 50 := by
  sorry

end NUMINAMATH_CALUDE_dangerous_animals_count_l2508_250877


namespace NUMINAMATH_CALUDE_ratio_problem_l2508_250897

theorem ratio_problem (a b c : ℚ) (h1 : b/a = 4) (h2 : c/b = 5) : 
  (a + 2*b) / (3*b + c) = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2508_250897


namespace NUMINAMATH_CALUDE_bill_meets_dexter_at_12_50_l2508_250852

/-- Represents a person or dog in the problem -/
structure Participant where
  speed : ℝ
  startTime : ℝ

/-- Calculates the time when Bill meets Dexter -/
def meetingTime (anna bill dexter : Participant) : ℝ :=
  sorry

/-- Theorem stating that Bill meets Dexter at 12:50 pm -/
theorem bill_meets_dexter_at_12_50 :
  let anna : Participant := { speed := 4, startTime := 0 }
  let bill : Participant := { speed := 3, startTime := 0 }
  let dexter : Participant := { speed := 6, startTime := 0.25 }
  meetingTime anna bill dexter = 0.8333333333 := by
  sorry

end NUMINAMATH_CALUDE_bill_meets_dexter_at_12_50_l2508_250852


namespace NUMINAMATH_CALUDE_super_soup_stores_l2508_250846

/-- The number of stores Super Soup had at the end of 2020 -/
def final_stores : ℕ :=
  let initial_stores : ℕ := 23
  let opened_2019 : ℕ := 5
  let closed_2019 : ℕ := 2
  let opened_2020 : ℕ := 10
  let closed_2020 : ℕ := 6
  initial_stores + (opened_2019 - closed_2019) + (opened_2020 - closed_2020)

/-- Theorem stating that the final number of stores is 30 -/
theorem super_soup_stores : final_stores = 30 := by
  sorry

end NUMINAMATH_CALUDE_super_soup_stores_l2508_250846
