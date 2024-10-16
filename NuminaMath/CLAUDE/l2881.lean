import Mathlib

namespace NUMINAMATH_CALUDE_largest_valid_number_is_valid_853_largest_valid_number_is_853_l2881_288137

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit integer
  (n / 100 = 8) ∧  -- Starts with 8
  (∀ d, d ≠ 0 → d ∣ n → n % d = 0) ∧  -- Divisible by each non-zero digit
  (n % (n / 100 + (n / 10) % 10 + n % 10) = 0)  -- Divisible by sum of digits

theorem largest_valid_number :
  ∀ m, is_valid_number m → m ≤ 853 :=
by sorry

theorem is_valid_853 : is_valid_number 853 :=
by sorry

theorem largest_valid_number_is_853 :
  ∀ n, is_valid_number n ∧ n ≠ 853 → n < 853 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_is_valid_853_largest_valid_number_is_853_l2881_288137


namespace NUMINAMATH_CALUDE_curve_tangent_line_values_l2881_288183

/-- The curve y = x^2 + ax + b -/
def curve (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The tangent line x - y + 1 = 0 -/
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

/-- The theorem stating the values of a and b -/
theorem curve_tangent_line_values (a b : ℝ) :
  (∀ x y, curve a b x = y → tangent_line x y) →
  (curve a b 0 = b) →
  (a = 1 ∧ b = 1) :=
sorry

end NUMINAMATH_CALUDE_curve_tangent_line_values_l2881_288183


namespace NUMINAMATH_CALUDE_arithmetic_sequence_range_l2881_288163

/-- 
For an arithmetic sequence with first term a₁ = -10 and common difference d,
if the 10th term and all subsequent terms are positive,
then 10/9 < d ≤ 5/4.
-/
theorem arithmetic_sequence_range (d : ℝ) : 
  (∀ n : ℕ, n ≥ 10 → -10 + (n - 1) * d > 0) → 
  10/9 < d ∧ d ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_range_l2881_288163


namespace NUMINAMATH_CALUDE_largest_valid_n_l2881_288130

/-- Represents the color of a ball -/
inductive Color
| Black
| White

/-- Represents a coloring function for balls -/
def Coloring := ℕ → Color

/-- Checks if a coloring satisfies the given condition -/
def ValidColoring (c : Coloring) (n : ℕ) : Prop :=
  ∀ a₁ a₂ a₃ a₄ : ℕ,
    a₁ ≤ n ∧ a₂ ≤ n ∧ a₃ ≤ n ∧ a₄ ≤ n →
    a₁ + a₂ + a₃ = a₄ →
    (c a₁ = Color.Black ∨ c a₂ = Color.Black ∨ c a₃ = Color.Black) ∧
    (c a₁ = Color.White ∨ c a₂ = Color.White ∨ c a₃ = Color.White)

/-- The theorem stating that 10 is the largest possible value of n -/
theorem largest_valid_n :
  (∃ c : Coloring, ValidColoring c 10) ∧
  (∀ n > 10, ¬∃ c : Coloring, ValidColoring c n) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_n_l2881_288130


namespace NUMINAMATH_CALUDE_no_two_unique_digit_cubes_l2881_288165

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_unique_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

def no_common_digits (n m : ℕ) : Prop :=
  (n.digits 10).toFinset ∩ (m.digits 10).toFinset = ∅

theorem no_two_unique_digit_cubes (kub : ℕ) 
  (h1 : is_three_digit_number kub)
  (h2 : has_unique_digits kub)
  (h3 : is_cube kub) :
  ¬ ∃ shar : ℕ, 
    is_three_digit_number shar ∧ 
    has_unique_digits shar ∧ 
    is_cube shar ∧ 
    no_common_digits kub shar :=
by sorry

end NUMINAMATH_CALUDE_no_two_unique_digit_cubes_l2881_288165


namespace NUMINAMATH_CALUDE_arrangements_count_is_2880_l2881_288109

/-- The number of arrangements of 4 students and 3 teachers in a row,
    where exactly two teachers are standing next to each other. -/
def arrangements_count : ℕ :=
  let num_students : ℕ := 4
  let num_teachers : ℕ := 3
  let num_units : ℕ := num_students + 1  -- 4 students + 1 teacher pair
  let teacher_pair_permutations : ℕ := 2  -- 2! ways to arrange 2 teachers in a pair
  let remaining_teacher_positions : ℕ := num_students + 1  -- positions for the remaining teacher
  let teacher_pair_combinations : ℕ := 3  -- number of ways to choose 2 teachers out of 3
  (Nat.factorial num_units) * teacher_pair_permutations * remaining_teacher_positions * teacher_pair_combinations

/-- Theorem stating that the number of arrangements is 2880 -/
theorem arrangements_count_is_2880 : arrangements_count = 2880 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_is_2880_l2881_288109


namespace NUMINAMATH_CALUDE_distribute_5_3_l2881_288155

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- There are 5 distinguishable balls -/
def num_balls : ℕ := 5

/-- There are 3 distinguishable boxes -/
def num_boxes : ℕ := 3

theorem distribute_5_3 : distribute num_balls num_boxes = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l2881_288155


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l2881_288125

theorem sqrt_sum_simplification : Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l2881_288125


namespace NUMINAMATH_CALUDE_shorter_wall_area_l2881_288167

/-- Given a rectangular hall with specified dimensions, calculate the area of the shorter wall. -/
theorem shorter_wall_area (floor_area : ℝ) (longer_wall_area : ℝ) (height : ℝ) :
  floor_area = 20 →
  longer_wall_area = 10 →
  height = 40 →
  let length := longer_wall_area / height
  let width := floor_area / length
  width * height = 3200 := by
  sorry

#check shorter_wall_area

end NUMINAMATH_CALUDE_shorter_wall_area_l2881_288167


namespace NUMINAMATH_CALUDE_scientific_notation_of_1040000000_l2881_288156

theorem scientific_notation_of_1040000000 :
  (1040000000 : ℝ) = 1.04 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1040000000_l2881_288156


namespace NUMINAMATH_CALUDE_nine_five_dollar_bills_equal_45_dollars_l2881_288196

/-- The total value in dollars when a person has a certain number of five-dollar bills -/
def total_value (num_bills : ℕ) : ℕ := 5 * num_bills

/-- Theorem: If a person has 9 five-dollar bills, they have a total of 45 dollars -/
theorem nine_five_dollar_bills_equal_45_dollars :
  total_value 9 = 45 := by sorry

end NUMINAMATH_CALUDE_nine_five_dollar_bills_equal_45_dollars_l2881_288196


namespace NUMINAMATH_CALUDE_special_sequence_bound_l2881_288171

def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n m, n < m → a n < a m) ∧ 
  (∀ k : ℕ, k ∈ Set.range a ∨ ∃ i j, k = a i + a j)

theorem special_sequence_bound (a : ℕ → ℕ) (h : SpecialSequence a) : 
  ∀ n : ℕ, n > 0 → a n ≤ n^2 := by
sorry

end NUMINAMATH_CALUDE_special_sequence_bound_l2881_288171


namespace NUMINAMATH_CALUDE_three_nap_simultaneously_l2881_288143

-- Define the type for mathematicians
def Mathematician := Fin 5

-- Define the type for nap times
variable {T : Type*}

-- Define the nap function that assigns two nap times to each mathematician
variable (nap : Mathematician → Fin 2 → T)

-- Define the property that any two mathematicians share a nap time
variable (share_nap : ∀ m1 m2 : Mathematician, m1 ≠ m2 → ∃ t : T, (∃ i : Fin 2, nap m1 i = t) ∧ (∃ j : Fin 2, nap m2 j = t))

-- Theorem statement
theorem three_nap_simultaneously :
  ∃ t : T, ∃ m1 m2 m3 : Mathematician, m1 ≠ m2 ∧ m2 ≠ m3 ∧ m1 ≠ m3 ∧
  (∃ i j k : Fin 2, nap m1 i = t ∧ nap m2 j = t ∧ nap m3 k = t) :=
sorry

end NUMINAMATH_CALUDE_three_nap_simultaneously_l2881_288143


namespace NUMINAMATH_CALUDE_cash_realized_before_brokerage_l2881_288139

/-- The cash realized on selling a stock before brokerage, given the total amount and brokerage rate -/
theorem cash_realized_before_brokerage 
  (total_amount : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : total_amount = 104)
  (h2 : brokerage_rate = 1 / 400) : 
  ∃ (cash_before_brokerage : ℝ), 
    cash_before_brokerage + cash_before_brokerage * brokerage_rate = total_amount ∧ 
    cash_before_brokerage = 41600 / 401 := by
  sorry

end NUMINAMATH_CALUDE_cash_realized_before_brokerage_l2881_288139


namespace NUMINAMATH_CALUDE_quadratic_roots_inequality_l2881_288113

theorem quadratic_roots_inequality (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - a*x₁ + a = 0 → x₂^2 - a*x₂ + a = 0 → x₁ ≠ x₂ → x₁^2 + x₂^2 ≥ 2*(x₁ + x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_inequality_l2881_288113


namespace NUMINAMATH_CALUDE_binary_ones_factorial_divisibility_l2881_288134

-- Define a function to count the number of ones in the binary representation of a natural number
def countOnes (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem binary_ones_factorial_divisibility (n : ℕ) (h : n > 0) (h_ones : countOnes n = 1995) :
  (2^(n - 1995) : ℕ) ∣ n! :=
sorry

end NUMINAMATH_CALUDE_binary_ones_factorial_divisibility_l2881_288134


namespace NUMINAMATH_CALUDE_coefficient_of_x_power_5_l2881_288127

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient function for the expansion of (x - 1/√x)^8
def coefficient (r : ℕ) : ℤ :=
  (-1)^r * (binomial 8 r)

-- Theorem statement
theorem coefficient_of_x_power_5 : coefficient 2 = 28 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_power_5_l2881_288127


namespace NUMINAMATH_CALUDE_book_fraction_is_half_l2881_288142

-- Define the total amount Jennifer had
def total_money : ℚ := 120

-- Define the fraction spent on sandwich
def sandwich_fraction : ℚ := 1 / 5

-- Define the fraction spent on museum ticket
def museum_fraction : ℚ := 1 / 6

-- Define the amount left over
def left_over : ℚ := 16

-- Theorem to prove
theorem book_fraction_is_half :
  let sandwich_cost := total_money * sandwich_fraction
  let museum_cost := total_money * museum_fraction
  let total_spent := total_money - left_over
  let book_cost := total_spent - sandwich_cost - museum_cost
  book_cost / total_money = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_book_fraction_is_half_l2881_288142


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l2881_288176

theorem regular_polygon_interior_angle (n : ℕ) (n_ge_3 : n ≥ 3) :
  (((n - 2) * 180) / n = 150) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l2881_288176


namespace NUMINAMATH_CALUDE_exists_palindromic_product_l2881_288103

/-- A natural number is palindromic in base 10 if it reads the same forward and backward. -/
def IsPalindromic (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), n = digits.foldl (λ acc d => 10 * acc + d) 0 ∧ digits = digits.reverse

/-- For any natural number not divisible by 10, there exists another natural number
    such that their product is palindromic in base 10. -/
theorem exists_palindromic_product (x : ℕ) (hx : ¬ 10 ∣ x) :
  ∃ y : ℕ, IsPalindromic (x * y) := by
  sorry

end NUMINAMATH_CALUDE_exists_palindromic_product_l2881_288103


namespace NUMINAMATH_CALUDE_six_pointed_star_perimeter_l2881_288158

/-- A regular hexagon with perimeter 3 meters -/
structure RegularHexagon :=
  (perimeter : ℝ)
  (is_regular : perimeter = 3)

/-- A six-pointed star formed by extending the sides of a regular hexagon -/
structure SixPointedStar (h : RegularHexagon) :=
  (perimeter : ℝ)

/-- The perimeter of the six-pointed star is 4√3 meters -/
theorem six_pointed_star_perimeter (h : RegularHexagon) (s : SixPointedStar h) :
  s.perimeter = 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_six_pointed_star_perimeter_l2881_288158


namespace NUMINAMATH_CALUDE_subset_implies_max_a_max_a_is_negative_three_l2881_288124

theorem subset_implies_max_a (a : ℝ) : 
  let A : Set ℝ := {x | |x| ≥ 3}
  let B : Set ℝ := {x | x ≥ a}
  A ⊆ B → a ≤ -3 :=
by
  sorry

theorem max_a_is_negative_three :
  ∃ c, c = -3 ∧ 
  (∀ a : ℝ, (let A : Set ℝ := {x | |x| ≥ 3}; let B : Set ℝ := {x | x ≥ a}; A ⊆ B) → a ≤ c) ∧
  (let A : Set ℝ := {x | |x| ≥ 3}; let B : Set ℝ := {x | x ≥ c}; A ⊆ B) :=
by
  sorry

end NUMINAMATH_CALUDE_subset_implies_max_a_max_a_is_negative_three_l2881_288124


namespace NUMINAMATH_CALUDE_specific_pairs_probability_l2881_288145

/-- The probability of two specific pairs forming in a random pairing of students -/
theorem specific_pairs_probability (n : ℕ) (h : n = 32) : 
  (1 : ℚ) / (n - 1) * (1 : ℚ) / (n - 2) = 1 / 930 :=
by sorry

end NUMINAMATH_CALUDE_specific_pairs_probability_l2881_288145


namespace NUMINAMATH_CALUDE_solve_for_z_l2881_288132

theorem solve_for_z : ∃ z : ℝ, ((2^5 : ℝ) * (9^2)) / (z * (3^5)) = 0.16666666666666666 ∧ z = 64 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l2881_288132


namespace NUMINAMATH_CALUDE_equation_real_root_l2881_288108

theorem equation_real_root (x m : ℝ) (i : ℂ) : 
  (∃ x : ℝ, x^2 + (1 - 2*i)*x + 3*m - i = 0) → m = 1/12 :=
by sorry

end NUMINAMATH_CALUDE_equation_real_root_l2881_288108


namespace NUMINAMATH_CALUDE_divisibility_of_square_sum_minus_2017_l2881_288173

theorem divisibility_of_square_sum_minus_2017 (n : ℕ+) : 
  ∃ (x y : ℤ), (n : ℤ) ∣ (x^2 + y^2 - 2017) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_square_sum_minus_2017_l2881_288173


namespace NUMINAMATH_CALUDE_battery_change_month_l2881_288186

/-- Given a 7-month interval between battery changes, starting in January,
    prove that the 15th change will occur in March. -/
theorem battery_change_month :
  let interval := 7  -- months between changes
  let start_month := 1  -- January
  let change_number := 15
  let total_months := interval * (change_number - 1)
  let years_passed := total_months / 12
  let extra_months := total_months % 12
  (start_month + extra_months - 1) % 12 + 1 = 3  -- 3 represents March
  := by sorry

end NUMINAMATH_CALUDE_battery_change_month_l2881_288186


namespace NUMINAMATH_CALUDE_art_museum_exhibits_l2881_288114

/-- The number of exhibits in an art museum --/
def num_exhibits : ℕ := 4

/-- The number of pictures the museum currently has --/
def current_pictures : ℕ := 15

/-- The number of additional pictures needed for equal distribution --/
def additional_pictures : ℕ := 1

theorem art_museum_exhibits :
  (current_pictures + additional_pictures) % num_exhibits = 0 ∧
  current_pictures % num_exhibits ≠ 0 ∧
  num_exhibits > 1 :=
sorry

end NUMINAMATH_CALUDE_art_museum_exhibits_l2881_288114


namespace NUMINAMATH_CALUDE_point_b_value_l2881_288193

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Represents the distance between two points on a number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

/-- Given points A and B on a number line, where A represents -2 and B is 5 units away from A,
    B must represent either -7 or 3. -/
theorem point_b_value (A B : Point) :
  A.value = -2 ∧ distance A B = 5 → B.value = -7 ∨ B.value = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_b_value_l2881_288193


namespace NUMINAMATH_CALUDE_number_of_history_books_l2881_288160

theorem number_of_history_books (total_books geography_books math_books : ℕ) 
  (h1 : total_books = 100)
  (h2 : geography_books = 25)
  (h3 : math_books = 43) :
  total_books - geography_books - math_books = 32 :=
by sorry

end NUMINAMATH_CALUDE_number_of_history_books_l2881_288160


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2881_288105

/-- An isosceles triangle with side lengths 9 and 5 has a perimeter of either 19 or 23 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = 9 ∧ b = 5 ∧ c = 5) ∨ (a = 5 ∧ b = 9 ∧ c = 9) →  -- isosceles with sides 9 and 5
  a + b + c = 19 ∨ a + b + c = 23 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2881_288105


namespace NUMINAMATH_CALUDE_eagle_pairs_count_l2881_288177

/-- The number of nesting pairs of bald eagles in 1963 -/
def pairs_1963 : ℕ := 417

/-- The increase in nesting pairs since 1963 -/
def increase : ℕ := 6649

/-- The current number of nesting pairs of bald eagles in the lower 48 states -/
def current_pairs : ℕ := pairs_1963 + increase

theorem eagle_pairs_count : current_pairs = 7066 := by
  sorry

end NUMINAMATH_CALUDE_eagle_pairs_count_l2881_288177


namespace NUMINAMATH_CALUDE_positive_operation_on_negative_two_l2881_288187

theorem positive_operation_on_negative_two (op : ℝ → ℝ → ℝ) : 
  (op 1 (-2) > 0) → (1 - (-2) > 0) :=
by sorry

end NUMINAMATH_CALUDE_positive_operation_on_negative_two_l2881_288187


namespace NUMINAMATH_CALUDE_area_of_triangle_PAB_l2881_288117

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

-- Define the tangent line
def tangent_line (x y m : ℝ) : Prop := y = Real.sqrt 3 * x + m ∧ m > 0

-- Define the point of tangency P
def point_P (x y : ℝ) : Prop := circle_O x y ∧ ∃ m, tangent_line x y m

-- Define points A and B as intersections of circle O and line y = x
def point_A_B (xa ya xb yb : ℝ) : Prop :=
  circle_O xa ya ∧ line_y_eq_x xa ya ∧
  circle_O xb yb ∧ line_y_eq_x xb yb ∧
  (xa ≠ xb ∨ ya ≠ yb)

-- Theorem statement
theorem area_of_triangle_PAB :
  ∀ (xa ya xb yb xp yp : ℝ),
  point_A_B xa ya xb yb →
  point_P xp yp →
  ∃ (area : ℝ), area = Real.sqrt 6 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_PAB_l2881_288117


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_right_triangle_l2881_288195

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if vectors m = (a+c, b) and n = (b, a-c) are parallel, then ABC is a right triangle -/
theorem parallel_vectors_imply_right_triangle 
  (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_parallel : (a + c) * (a - c) = b^2) :
  a^2 = b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_right_triangle_l2881_288195


namespace NUMINAMATH_CALUDE_line_intersects_segment_iff_a_gt_two_l2881_288150

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on the positive side of a line -/
def positiveSide (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c > 0

/-- Check if a point is on the negative side of a line -/
def negativeSide (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c < 0

/-- Check if two points are on opposite sides of a line -/
def oppositeSides (l : Line) (p1 p2 : Point) : Prop :=
  (positiveSide l p1 ∧ negativeSide l p2) ∨ (negativeSide l p1 ∧ positiveSide l p2)

/-- The main theorem -/
theorem line_intersects_segment_iff_a_gt_two (a : ℝ) :
  let A : Point := ⟨1, a⟩
  let B : Point := ⟨2, 4⟩
  let l : Line := ⟨1, -1, 1⟩
  oppositeSides l A B ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_segment_iff_a_gt_two_l2881_288150


namespace NUMINAMATH_CALUDE_triangle_side_difference_l2881_288166

theorem triangle_side_difference (x : ℤ) : 
  (x > 5 ∧ x < 11) → (11 - 6 = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l2881_288166


namespace NUMINAMATH_CALUDE_counterexample_25_l2881_288174

theorem counterexample_25 : 
  ¬(¬(Nat.Prime 25) → Nat.Prime (25 + 3)) := by sorry

end NUMINAMATH_CALUDE_counterexample_25_l2881_288174


namespace NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_l2881_288191

theorem probability_four_ones_in_five_rolls :
  let n_rolls : ℕ := 5
  let n_desired : ℕ := 4
  let die_sides : ℕ := 6
  let p_success : ℚ := 1 / die_sides
  let p_failure : ℚ := 1 - p_success
  let combinations : ℕ := Nat.choose n_rolls n_desired
  combinations * p_success ^ n_desired * p_failure ^ (n_rolls - n_desired) = 25 / 7776 :=
by sorry

end NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_l2881_288191


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l2881_288197

/-- Systematic sampling from a population -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  groups : ℕ
  first_group_draw : ℕ
  nth_group_draw : ℕ
  nth_group : ℕ

/-- Theorem for systematic sampling -/
theorem systematic_sample_theorem (s : SystematicSample)
  (h1 : s.population = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.groups = 20)
  (h4 : s.population = s.groups * 8)
  (h5 : s.nth_group_draw = 126)
  (h6 : s.nth_group = 16) :
  s.first_group_draw = 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l2881_288197


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2881_288178

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  cos C + (cos A - Real.sqrt 3 * sin A) * cos B = 0 ∧
  b = Real.sqrt 3 ∧
  c = 1 →
  B = π / 3 ∧
  (1 / 2) * a * c * sin B = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2881_288178


namespace NUMINAMATH_CALUDE_matt_current_age_l2881_288179

/-- Matt's current age -/
def matt_age : ℕ := sorry

/-- Kaylee's current age -/
def kaylee_age : ℕ := 8

theorem matt_current_age : matt_age = 5 := by
  have h1 : kaylee_age + 7 = 3 * matt_age := sorry
  sorry

end NUMINAMATH_CALUDE_matt_current_age_l2881_288179


namespace NUMINAMATH_CALUDE_sphere_radius_from_hemisphere_volume_l2881_288101

/-- Given a sphere whose hemisphere has a volume of 36π cm³, prove that the radius of the sphere is 3 cm. -/
theorem sphere_radius_from_hemisphere_volume :
  ∀ r : ℝ, (2 / 3 * π * r^3 = 36 * π) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hemisphere_volume_l2881_288101


namespace NUMINAMATH_CALUDE_unit_conversion_l2881_288168

/-- Conversion rates --/
def hectare_to_square_meter : ℝ := 10000
def meter_to_centimeter : ℝ := 100
def square_kilometer_to_hectare : ℝ := 100
def hour_to_minute : ℝ := 60
def kilogram_to_gram : ℝ := 1000

/-- Unit conversion theorem --/
theorem unit_conversion :
  (360 / hectare_to_square_meter = 0.036) ∧
  (504 / meter_to_centimeter = 5.04) ∧
  (0.06 * square_kilometer_to_hectare = 6) ∧
  (15 / hour_to_minute = 0.25) ∧
  (5.45 = 5 + 450 / kilogram_to_gram) :=
by sorry

end NUMINAMATH_CALUDE_unit_conversion_l2881_288168


namespace NUMINAMATH_CALUDE_fruit_basket_count_l2881_288184

def num_apples : ℕ := 7
def num_oranges : ℕ := 12
def min_fruits_per_basket : ℕ := 2

-- Function to calculate the number of valid fruit baskets
def count_valid_baskets (apples oranges min_fruits : ℕ) : ℕ :=
  (apples + 1) * (oranges + 1) - (1 + apples + oranges)

-- Theorem stating that the number of valid fruit baskets is 101
theorem fruit_basket_count :
  count_valid_baskets num_apples num_oranges min_fruits_per_basket = 101 := by
  sorry

#eval count_valid_baskets num_apples num_oranges min_fruits_per_basket

end NUMINAMATH_CALUDE_fruit_basket_count_l2881_288184


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2881_288185

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ y => |y - 8| + 3 * y
  ∃ (y₁ y₂ : ℝ), y₁ = 23/4 ∧ y₂ = 7/2 ∧ f y₁ = 15 ∧ f y₂ = 15 ∧
    (∀ y : ℝ, f y = 15 → y = y₁ ∨ y = y₂) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2881_288185


namespace NUMINAMATH_CALUDE_fraction_power_multiplication_l2881_288154

theorem fraction_power_multiplication :
  (3 / 5 : ℝ)^4 * (2 / 9 : ℝ)^(1/2) = 81 * Real.sqrt 2 / 1875 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_multiplication_l2881_288154


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2881_288182

theorem roots_sum_of_squares (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) → 
  a^2 + b^2 + c^2 = 4046 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2881_288182


namespace NUMINAMATH_CALUDE_negation_equivalence_l2881_288115

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2881_288115


namespace NUMINAMATH_CALUDE_point_on_line_l2881_288152

/-- Given three points in the plane, this function checks if they are collinear -/
def are_collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that the point (14,7) lies on the line passing through (2,1) and (10,5) -/
theorem point_on_line : are_collinear 2 1 10 5 14 7 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2881_288152


namespace NUMINAMATH_CALUDE_triangle_side_simplification_l2881_288175

theorem triangle_side_simplification (k : ℝ) (h1 : 3 < k) (h2 : k < 5) :
  |2*k - 5| - Real.sqrt (k^2 - 12*k + 36) = 3*k - 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_simplification_l2881_288175


namespace NUMINAMATH_CALUDE_congruence_problem_l2881_288102

theorem congruence_problem (y : ℤ) 
  (h1 : (4 + y) % (4^3) = 3^2 % (4^3))
  (h2 : (6 + y) % (6^3) = 4^2 % (6^3))
  (h3 : (8 + y) % (8^3) = 6^2 % (8^3)) :
  y % 168 = 4 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l2881_288102


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2881_288123

/-- Proves that the rationalization of 1/(√5 + √7 + √11) is equal to (-√5 - √7 + √11 + 2√385)/139 -/
theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) =
    (A * Real.sqrt 5 + B * Real.sqrt 7 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    A = -1 ∧
    B = -1 ∧
    C = 1 ∧
    D = 2 ∧
    E = 385 ∧
    F = 139 ∧
    F > 0 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2881_288123


namespace NUMINAMATH_CALUDE_prob_one_head_in_three_flips_l2881_288190

/-- The probability of getting exactly one head in three flips of a fair coin -/
theorem prob_one_head_in_three_flips :
  let n : ℕ := 3  -- number of flips
  let k : ℕ := 1  -- number of desired heads
  let p : ℚ := 1/2  -- probability of getting heads on a single flip
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_head_in_three_flips_l2881_288190


namespace NUMINAMATH_CALUDE_find_k_l2881_288116

theorem find_k : ∃ k : ℚ, (2 * 2 - 3 * k * (-1) = 1) ∧ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2881_288116


namespace NUMINAMATH_CALUDE_a_5_equals_9_l2881_288104

-- Define the sequence and its sum
def S (n : ℕ) := n^2

-- Define the general term of the sequence
def a (n : ℕ) : ℕ := S n - S (n-1)

-- Theorem statement
theorem a_5_equals_9 : a 5 = 9 := by sorry

end NUMINAMATH_CALUDE_a_5_equals_9_l2881_288104


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2881_288153

/-- Given a principal amount that yields 202.50 interest at 4.5% rate, 
    prove that the rate yielding 225 interest on the same principal is 5% -/
theorem interest_rate_calculation (P : ℝ) : 
  P * 0.045 = 202.50 → P * (5 / 100) = 225 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2881_288153


namespace NUMINAMATH_CALUDE_stratified_sampling_l2881_288119

/-- Stratified sampling problem -/
theorem stratified_sampling 
  (total_items : ℕ) 
  (sample_size : ℕ) 
  (stratum_A_size : ℕ) 
  (h1 : total_items = 600) 
  (h2 : sample_size = 100) 
  (h3 : stratum_A_size = 150) :
  let items_from_A := (sample_size * stratum_A_size) / total_items
  let prob_item_A := sample_size / total_items
  (items_from_A = 25) ∧ (prob_item_A = 1 / 6) := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2881_288119


namespace NUMINAMATH_CALUDE_oil_to_add_l2881_288136

/-- The amount of oil Scarlett needs to add to her measuring cup -/
theorem oil_to_add (current : ℚ) (desired : ℚ) : 
  current = 0.16666666666666666 →
  desired = 0.8333333333333334 →
  desired - current = 0.6666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_oil_to_add_l2881_288136


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_equality_condition_l2881_288169

theorem min_value_reciprocal_sum (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_one : x + y + z = 1) :
  1/x + 4/y + 9/z ≥ 36 := by
  sorry

theorem equality_condition (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_one : x + y + z = 1) :
  1/x + 4/y + 9/z = 36 ↔ x = 1/6 ∧ y = 1/3 ∧ z = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_equality_condition_l2881_288169


namespace NUMINAMATH_CALUDE_larger_root_of_quadratic_l2881_288149

theorem larger_root_of_quadratic (x : ℝ) : 
  x^2 + 17*x - 72 = 0 → x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_larger_root_of_quadratic_l2881_288149


namespace NUMINAMATH_CALUDE_min_omega_value_l2881_288107

theorem min_omega_value (ω : ℝ) (f g : ℝ → ℝ) : 
  (ω > 0) →
  (∀ x, f x = Real.sin (ω * x)) →
  (∀ x, g x = f (x - π / 12)) →
  (∃ k : ℤ, ω * π / 3 - ω * π / 12 = k * π + π / 2) →
  ω ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_value_l2881_288107


namespace NUMINAMATH_CALUDE_probability_greater_than_30_l2881_288131

def numbers : Finset ℕ := {1, 2, 3, 4, 5}

def is_valid_selection (pair : ℕ × ℕ) : Prop :=
  pair.1 ∈ numbers ∧ pair.2 ∈ numbers ∧ pair.1 ≠ pair.2

def to_two_digit (pair : ℕ × ℕ) : ℕ :=
  10 * pair.1 + pair.2

def is_greater_than_30 (n : ℕ) : Prop :=
  n > 30

theorem probability_greater_than_30 :
  Nat.card {pair : ℕ × ℕ | is_valid_selection pair ∧ is_greater_than_30 (to_two_digit pair)} /
  Nat.card {pair : ℕ × ℕ | is_valid_selection pair} = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_greater_than_30_l2881_288131


namespace NUMINAMATH_CALUDE_exam_class_size_l2881_288194

/-- Represents a class of students with their exam marks -/
structure ExamClass where
  total_students : ℕ
  total_marks : ℕ
  average_mark : ℚ
  excluded_students : ℕ
  excluded_average : ℚ
  remaining_average : ℚ

/-- Theorem stating the conditions and the result to be proven -/
theorem exam_class_size (c : ExamClass) 
  (h1 : c.average_mark = 80)
  (h2 : c.excluded_students = 5)
  (h3 : c.excluded_average = 40)
  (h4 : c.remaining_average = 90)
  (h5 : c.total_marks = c.total_students * c.average_mark)
  (h6 : c.total_marks - c.excluded_students * c.excluded_average = 
        (c.total_students - c.excluded_students) * c.remaining_average) :
  c.total_students = 25 := by
  sorry

end NUMINAMATH_CALUDE_exam_class_size_l2881_288194


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l2881_288180

theorem sqrt_sum_equality : Real.sqrt (11 + 6 * Real.sqrt 2) + Real.sqrt (11 - 6 * Real.sqrt 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l2881_288180


namespace NUMINAMATH_CALUDE_tea_cups_filled_l2881_288126

theorem tea_cups_filled (total_tea : ℕ) (tea_per_cup : ℕ) (h1 : total_tea = 1050) (h2 : tea_per_cup = 65) :
  (total_tea / tea_per_cup : ℕ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_tea_cups_filled_l2881_288126


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2881_288199

/-- The first parabola -/
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 7

/-- The second parabola -/
def g (x : ℝ) : ℝ := 2 * x^2 - 5

/-- The intersection points of the two parabolas -/
def intersection_points : Set (ℝ × ℝ) := {(-2, 3), (1/2, -4.5)}

theorem parabolas_intersection :
  ∀ p : ℝ × ℝ, f p.1 = g p.1 ↔ p ∈ intersection_points := by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2881_288199


namespace NUMINAMATH_CALUDE_vectors_are_parallel_l2881_288189

def a : ℝ × ℝ × ℝ := (1, 2, -2)
def b : ℝ × ℝ × ℝ := (-2, -4, 4)

theorem vectors_are_parallel : ∃ k : ℝ, b = k • a := by
  sorry

end NUMINAMATH_CALUDE_vectors_are_parallel_l2881_288189


namespace NUMINAMATH_CALUDE_xy_value_l2881_288112

theorem xy_value (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 44) : x*y = -24 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2881_288112


namespace NUMINAMATH_CALUDE_rational_inequality_l2881_288100

theorem rational_inequality (a b c d : ℚ) 
  (h : a^3 - 2005 = b^3 + 2027 ∧ 
       b^3 + 2027 = c^3 - 2822 ∧ 
       c^3 - 2822 = d^3 + 2820) : 
  c > a ∧ a > b ∧ b > d := by
sorry

end NUMINAMATH_CALUDE_rational_inequality_l2881_288100


namespace NUMINAMATH_CALUDE_bus_dispatch_interval_l2881_288121

/-- Represents the speed of a vehicle or person -/
structure Speed : Type :=
  (value : ℝ)

/-- Represents a time interval -/
structure TimeInterval : Type :=
  (minutes : ℝ)

/-- Represents a distance -/
structure Distance : Type :=
  (value : ℝ)

/-- The speed of Xiao Nan -/
def xiao_nan_speed : Speed := ⟨1⟩

/-- The speed of Xiao Yu -/
def xiao_yu_speed : Speed := ⟨3 * xiao_nan_speed.value⟩

/-- The time interval at which Xiao Nan encounters buses -/
def xiao_nan_encounter_interval : TimeInterval := ⟨10⟩

/-- The time interval at which Xiao Yu encounters buses -/
def xiao_yu_encounter_interval : TimeInterval := ⟨5⟩

/-- The speed of the bus -/
def bus_speed : Speed := ⟨5 * xiao_nan_speed.value⟩

/-- The distance between two consecutive buses -/
def bus_distance (s : Speed) (t : TimeInterval) : Distance :=
  ⟨s.value * t.minutes⟩

/-- The theorem stating that the interval between bus dispatches is 8 minutes -/
theorem bus_dispatch_interval :
  ∃ (t : TimeInterval),
    t.minutes = 8 ∧
    bus_distance bus_speed t = bus_distance (Speed.mk (bus_speed.value - xiao_nan_speed.value)) xiao_nan_encounter_interval ∧
    bus_distance bus_speed t = bus_distance (Speed.mk (bus_speed.value + xiao_yu_speed.value)) xiao_yu_encounter_interval :=
sorry

end NUMINAMATH_CALUDE_bus_dispatch_interval_l2881_288121


namespace NUMINAMATH_CALUDE_new_men_average_age_l2881_288122

/-- Given a group of 8 men, when two men aged 21 and 23 are replaced by two new men,
    and the average age of the group increases by 2 years,
    prove that the average age of the two new men is 30 years. -/
theorem new_men_average_age
  (initial_count : Nat)
  (replaced_age1 replaced_age2 : Nat)
  (age_increase : ℝ)
  (h_count : initial_count = 8)
  (h_replaced1 : replaced_age1 = 21)
  (h_replaced2 : replaced_age2 = 23)
  (h_increase : age_increase = 2)
  : (↑initial_count * age_increase + ↑replaced_age1 + ↑replaced_age2) / 2 = 30 :=
by sorry

end NUMINAMATH_CALUDE_new_men_average_age_l2881_288122


namespace NUMINAMATH_CALUDE_triangle_mass_l2881_288144

-- Define the shapes
variable (Square Circle Triangle : ℝ)

-- Define the scale equations
axiom scale1 : Square + Circle = 8
axiom scale2 : Square + 2 * Circle = 11
axiom scale3 : Circle + 2 * Triangle = 15

-- Theorem to prove
theorem triangle_mass : Triangle = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_mass_l2881_288144


namespace NUMINAMATH_CALUDE_second_largest_of_five_consecutive_odds_l2881_288111

theorem second_largest_of_five_consecutive_odds (a b c d e : ℕ) : 
  (∀ n : ℕ, n ∈ [a, b, c, d, e] → n % 2 = 1) →  -- all numbers are odd
  (b = a + 2 ∧ c = b + 2 ∧ d = c + 2 ∧ e = d + 2) →  -- consecutive
  a + b + c + d + e = 195 →  -- sum is 195
  d = 41 :=  -- 2nd largest (4th in sequence) is 41
by
  sorry

end NUMINAMATH_CALUDE_second_largest_of_five_consecutive_odds_l2881_288111


namespace NUMINAMATH_CALUDE_female_democrats_count_l2881_288170

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 810 →
  female + male = total →
  (female / 2 + male / 4 : ℚ) = total / 3 →
  female / 2 = 135 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l2881_288170


namespace NUMINAMATH_CALUDE_decimal_sum_theorem_l2881_288120

theorem decimal_sum_theorem : 0.03 + 0.004 + 0.009 + 0.0001 = 0.0431 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_theorem_l2881_288120


namespace NUMINAMATH_CALUDE_max_player_salary_l2881_288159

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (total_salary_cap : ℕ) :
  num_players = 18 →
  min_salary = 20000 →
  total_salary_cap = 900000 →
  ∃ (max_salary : ℕ),
    max_salary = 560000 ∧
    max_salary + (num_players - 1) * min_salary ≤ total_salary_cap ∧
    ∀ (s : ℕ), s > max_salary →
      s + (num_players - 1) * min_salary > total_salary_cap :=
by sorry


end NUMINAMATH_CALUDE_max_player_salary_l2881_288159


namespace NUMINAMATH_CALUDE_smallest_colors_l2881_288162

/-- A coloring of an infinite table -/
def InfiniteColoring (n : ℕ) := ℤ → ℤ → Fin n

/-- Predicate to check if a 2x3 or 3x2 rectangle has all different colors -/
def ValidRectangle (c : InfiniteColoring n) : Prop :=
  ∀ i j : ℤ, (
    (c i j ≠ c i (j+1) ∧ c i j ≠ c i (j+2) ∧ c i j ≠ c (i+1) j ∧ c i j ≠ c (i+1) (j+1) ∧ c i j ≠ c (i+1) (j+2)) ∧
    (c i (j+1) ≠ c i (j+2) ∧ c i (j+1) ≠ c (i+1) j ∧ c i (j+1) ≠ c (i+1) (j+1) ∧ c i (j+1) ≠ c (i+1) (j+2)) ∧
    (c i (j+2) ≠ c (i+1) j ∧ c i (j+2) ≠ c (i+1) (j+1) ∧ c i (j+2) ≠ c (i+1) (j+2)) ∧
    (c (i+1) j ≠ c (i+1) (j+1) ∧ c (i+1) j ≠ c (i+1) (j+2)) ∧
    (c (i+1) (j+1) ≠ c (i+1) (j+2))
  ) ∧ (
    (c i j ≠ c i (j+1) ∧ c i j ≠ c (i+1) j ∧ c i j ≠ c (i+2) j ∧ c i j ≠ c (i+1) (j+1) ∧ c i j ≠ c (i+2) (j+1)) ∧
    (c i (j+1) ≠ c (i+1) j ∧ c i (j+1) ≠ c (i+2) j ∧ c i (j+1) ≠ c (i+1) (j+1) ∧ c i (j+1) ≠ c (i+2) (j+1)) ∧
    (c (i+1) j ≠ c (i+2) j ∧ c (i+1) j ≠ c (i+1) (j+1) ∧ c (i+1) j ≠ c (i+2) (j+1)) ∧
    (c (i+2) j ≠ c (i+1) (j+1) ∧ c (i+2) j ≠ c (i+2) (j+1)) ∧
    (c (i+1) (j+1) ≠ c (i+2) (j+1))
  )

/-- The smallest number of colors needed is 8 -/
theorem smallest_colors : (∃ c : InfiniteColoring 8, ValidRectangle c) ∧ 
  (∀ n < 8, ¬∃ c : InfiniteColoring n, ValidRectangle c) :=
sorry

end NUMINAMATH_CALUDE_smallest_colors_l2881_288162


namespace NUMINAMATH_CALUDE_electricity_gasoline_ratio_l2881_288110

theorem electricity_gasoline_ratio (total : ℕ) (both : ℕ) (gas_only : ℕ) (neither : ℕ)
  (h_total : total = 300)
  (h_both : both = 120)
  (h_gas_only : gas_only = 60)
  (h_neither : neither = 24)
  (h_sum : total = both + gas_only + (total - both - gas_only - neither) + neither) :
  (total - both - gas_only - neither) / neither = 4 := by
sorry

end NUMINAMATH_CALUDE_electricity_gasoline_ratio_l2881_288110


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2881_288157

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) * (2 - x) > 0 ↔ 2 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2881_288157


namespace NUMINAMATH_CALUDE_largest_palindrome_divisible_by_15_l2881_288161

/-- A function that checks if a number is a 4-digit palindrome --/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- The largest 4-digit palindromic number divisible by 15 --/
def largest_palindrome : ℕ := 5775

/-- Sum of digits of a natural number --/
def digit_sum (n : ℕ) : ℕ :=
  let rec sum_digits (m : ℕ) (acc : ℕ) : ℕ :=
    if m = 0 then acc else sum_digits (m / 10) (acc + m % 10)
  sum_digits n 0

theorem largest_palindrome_divisible_by_15 :
  is_four_digit_palindrome largest_palindrome ∧
  largest_palindrome % 15 = 0 ∧
  (∀ n : ℕ, is_four_digit_palindrome n → n % 15 = 0 → n ≤ largest_palindrome) ∧
  digit_sum largest_palindrome = 24 := by
  sorry

end NUMINAMATH_CALUDE_largest_palindrome_divisible_by_15_l2881_288161


namespace NUMINAMATH_CALUDE_half_of_third_of_sixth_of_90_l2881_288140

theorem half_of_third_of_sixth_of_90 : (1 / 2 : ℚ) * (1 / 3) * (1 / 6) * 90 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_of_third_of_sixth_of_90_l2881_288140


namespace NUMINAMATH_CALUDE_min_faces_prism_min_vertices_pyramid_l2881_288192

/-- A prism is a three-dimensional shape with two identical ends and flat sides. -/
structure Prism where
  base : Set (ℝ × ℝ)  -- Representing the base as a set of points in 2D
  height : ℝ

/-- A pyramid is a three-dimensional shape with a polygonal base and triangular faces meeting at a point. -/
structure Pyramid where
  base : Set (ℝ × ℝ)  -- Representing the base as a set of points in 2D
  apex : ℝ × ℝ × ℝ    -- The apex point in 3D

/-- The number of faces in a prism. -/
def num_faces_prism (p : Prism) : ℕ := sorry

/-- The number of vertices in a pyramid. -/
def num_vertices_pyramid (p : Pyramid) : ℕ := sorry

/-- The minimum number of faces in any prism is 5. -/
theorem min_faces_prism : ∀ p : Prism, num_faces_prism p ≥ 5 := sorry

/-- The number of vertices in a pyramid with the minimum number of faces is 4. -/
theorem min_vertices_pyramid : ∃ p : Pyramid, num_vertices_pyramid p = 4 ∧ 
  (∀ q : Pyramid, num_vertices_pyramid q ≥ num_vertices_pyramid p) := sorry

end NUMINAMATH_CALUDE_min_faces_prism_min_vertices_pyramid_l2881_288192


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l2881_288133

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^4 - 8 * X^3 - 12 * X^2 + 5 * X - 9
  let divisor : Polynomial ℚ := 3 * X^2 - 2
  let quotient := dividend / divisor
  (quotient.coeff 2 = 10/3) ∧ (quotient.coeff 1 = -8/3) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l2881_288133


namespace NUMINAMATH_CALUDE_impossibility_of_all_powers_of_two_l2881_288106

/-- Represents a card with a natural number -/
structure Card where
  value : ℕ

/-- Represents the state of the table at any given time -/
structure TableState where
  cards : List Card
  oddCount : ℕ

/-- The procedure of creating a new card from existing cards -/
def createNewCard (state : TableState) : Card :=
  sorry

/-- The evolution of the table state over time -/
def evolveTable (initialState : TableState) : ℕ → TableState
  | 0 => initialState
  | n + 1 => let prevState := evolveTable initialState n
              let newCard := createNewCard prevState
              { cards := newCard :: prevState.cards,
                oddCount := if newCard.value % 2 = 1 then prevState.oddCount + 1 else prevState.oddCount }

/-- Checks if a number is divisible by 2^d -/
def isDivisibleByPowerOfTwo (n d : ℕ) : Bool :=
  n % (2^d) = 0

theorem impossibility_of_all_powers_of_two :
  ∀ (initialCards : List Card),
    initialCards.length = 100 →
    (initialCards.filter (λ c => c.value % 2 = 1)).length = 28 →
    ∃ (d : ℕ), ∀ (t : ℕ),
      ¬∃ (card : Card),
        card ∈ (evolveTable { cards := initialCards, oddCount := 28 } t).cards ∧
        isDivisibleByPowerOfTwo card.value d :=
  sorry

end NUMINAMATH_CALUDE_impossibility_of_all_powers_of_two_l2881_288106


namespace NUMINAMATH_CALUDE_sum_divisible_by_three_l2881_288146

theorem sum_divisible_by_three (a : ℤ) : ∃ k : ℤ, a^3 + 2*a = 3*k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_three_l2881_288146


namespace NUMINAMATH_CALUDE_teacher_student_grouping_probability_l2881_288129

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of groups -/
def num_groups : ℕ := 2

/-- The number of teachers per group -/
def teachers_per_group : ℕ := 1

/-- The number of students per group -/
def students_per_group : ℕ := 2

/-- The probability that teacher A and student B are in the same group -/
def prob_same_group : ℚ := 1/2

theorem teacher_student_grouping_probability :
  (num_teachers = 2) →
  (num_students = 4) →
  (num_groups = 2) →
  (teachers_per_group = 1) →
  (students_per_group = 2) →
  prob_same_group = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_teacher_student_grouping_probability_l2881_288129


namespace NUMINAMATH_CALUDE_gnome_distribution_ways_l2881_288147

/-- The number of ways to distribute n identical objects among k recipients,
    with each recipient receiving at least m objects. -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * (m - 1) + k - 1) (k - 1)

/-- The number of gnomes -/
def num_gnomes : ℕ := 3

/-- The total number of stones -/
def total_stones : ℕ := 70

/-- The minimum number of stones each gnome must receive -/
def min_stones : ℕ := 10

theorem gnome_distribution_ways : 
  distribution_ways total_stones num_gnomes min_stones = 946 := by
  sorry

end NUMINAMATH_CALUDE_gnome_distribution_ways_l2881_288147


namespace NUMINAMATH_CALUDE_constant_term_is_165_l2881_288128

-- Define the derivative function
def derivative (q : ℝ → ℝ) : ℝ → ℝ := sorry

-- Define the equation q' = 3q + c
def equation (c : ℝ) (q : ℝ → ℝ) : Prop :=
  ∀ x, derivative q x = 3 * q x + c

-- State the theorem
theorem constant_term_is_165 :
  ∃ (q : ℝ → ℝ) (c : ℝ),
    equation c q ∧
    derivative (derivative q) 6 = 210 ∧
    c = 165 :=
sorry

end NUMINAMATH_CALUDE_constant_term_is_165_l2881_288128


namespace NUMINAMATH_CALUDE_unique_point_exists_l2881_288188

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}

-- Define the diameter endpoints
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the conditions for point P
def IsValidP (p : ℝ × ℝ) : Prop :=
  p ∈ Circle ∧
  (p.1 - A.1)^2 + (p.2 - A.2)^2 + (p.1 - B.1)^2 + (p.2 - B.2)^2 = 10 ∧
  Real.cos (Real.arccos ((p.1 - A.1) * (p.1 - B.1) + (p.2 - A.2) * (p.2 - B.2)) /
    (Real.sqrt ((p.1 - A.1)^2 + (p.2 - A.2)^2) * Real.sqrt ((p.1 - B.1)^2 + (p.2 - B.2)^2))) = 1/2

theorem unique_point_exists : ∃! p, IsValidP p :=
  sorry

end NUMINAMATH_CALUDE_unique_point_exists_l2881_288188


namespace NUMINAMATH_CALUDE_watch_cost_price_l2881_288164

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (C : ℚ),
  (C * 88 / 100 : ℚ) + 140 = C * 104 / 100 ∧ C = 875 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2881_288164


namespace NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_456_l2881_288181

theorem multiplicative_inverse_123_mod_456 :
  ∃ (x : ℕ), x < 456 ∧ (123 * x) % 456 = 1 :=
by
  use 52
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_456_l2881_288181


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_l2881_288148

/-- Given an ellipse ax² + by² = 1 intersecting the line y = 1 - x, 
    if a line through the origin and the midpoint of the intersection points 
    has slope √3/2, then a/b = √3/2 -/
theorem ellipse_intersection_slope (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : 
  (∃ x₁ x₂ : ℝ, 
    a * x₁^2 + b * (1 - x₁)^2 = 1 ∧ 
    a * x₂^2 + b * (1 - x₂)^2 = 1 ∧
    x₁ ≠ x₂ ∧
    (a / (a + b)) / (b / (a + b)) = Real.sqrt 3 / 2) →
  a / b = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_l2881_288148


namespace NUMINAMATH_CALUDE_determinant_sum_l2881_288118

theorem determinant_sum (x y : ℝ) (h1 : x ≠ y) 
  (h2 : Matrix.det ![![2, 6, 12], ![4, x, y], ![4, y, x]] = 0) : 
  x + y = 36 := by
  sorry

end NUMINAMATH_CALUDE_determinant_sum_l2881_288118


namespace NUMINAMATH_CALUDE_parallel_lines_l2881_288141

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- Two lines are coincident if and only if they have the same slope and y-intercept -/
def coincident (m1 b1 m2 b2 : ℝ) : Prop := m1 = m2 ∧ b1 = b2

theorem parallel_lines (a : ℝ) : 
  (parallel (-a/2) (-3/(a-1)) ∧ ¬coincident (-a/2) (-1/2) (-3/(a-1)) (-1/(a-1))) → 
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_l2881_288141


namespace NUMINAMATH_CALUDE_fencing_theorem_l2881_288138

/-- Represents a rectangular field with given dimensions -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- Calculates the fencing required for three sides of a rectangular field -/
def fencing_required (field : RectangularField) : ℝ :=
  2 * field.width + field.length

theorem fencing_theorem (field : RectangularField) 
  (h1 : field.area = 600)
  (h2 : field.uncovered_side = 30)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  fencing_required field = 70 := by
  sorry

#check fencing_theorem

end NUMINAMATH_CALUDE_fencing_theorem_l2881_288138


namespace NUMINAMATH_CALUDE_inequality_always_holds_l2881_288135

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l2881_288135


namespace NUMINAMATH_CALUDE_old_supervisor_salary_is_870_l2881_288172

/-- Represents the salary structure of a factory --/
structure FactorySalary where
  workers : ℕ               -- Number of workers
  oldAverage : ℚ            -- Average salary with old supervisor
  newAverage : ℚ            -- Average salary with new supervisor
  newSupervisorSalary : ℚ   -- Salary of the new supervisor

/-- Calculates the salary of the old supervisor given the factory's salary structure --/
def oldSupervisorSalary (fs : FactorySalary) : ℚ :=
  (fs.workers + 1) * fs.oldAverage - fs.workers * fs.newAverage - fs.newSupervisorSalary + (fs.workers + 1) * fs.newAverage

/-- Theorem stating that the old supervisor's salary was 870 given the problem conditions --/
theorem old_supervisor_salary_is_870 (fs : FactorySalary)
  (h1 : fs.workers = 8)
  (h2 : fs.oldAverage = 430)
  (h3 : fs.newAverage = 420)
  (h4 : fs.newSupervisorSalary = 780) :
  oldSupervisorSalary fs = 870 := by
  sorry

end NUMINAMATH_CALUDE_old_supervisor_salary_is_870_l2881_288172


namespace NUMINAMATH_CALUDE_equal_numbers_l2881_288151

theorem equal_numbers (a : Fin 100 → ℝ)
  (h : ∀ i : Fin 100, a i - 3 * a (i + 1) + 2 * a (i + 2) ≥ 0) :
  ∀ i j : Fin 100, a i = a j :=
by sorry


end NUMINAMATH_CALUDE_equal_numbers_l2881_288151


namespace NUMINAMATH_CALUDE_expression_evaluation_l2881_288198

theorem expression_evaluation (x y : ℚ) (hx : x = 2/3) (hy : y = 5/2) :
  (1/3) * x^8 * y^9 = 5^9 / (2 * 3^9) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2881_288198
