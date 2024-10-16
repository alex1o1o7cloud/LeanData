import Mathlib

namespace NUMINAMATH_CALUDE_recurrence_sequence_property_l3736_373630

/-- An integer sequence satisfying the given recurrence relation -/
def RecurrenceSequence (m : ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n

/-- The main theorem -/
theorem recurrence_sequence_property
  (m : ℤ) (a : ℕ → ℤ) (h_m : |m| ≥ 2)
  (h_nonzero : ¬(a 1 = 0 ∧ a 2 = 0))
  (h_recurrence : RecurrenceSequence m a)
  (r s : ℕ) (h_rs : r > s ∧ s ≥ 2)
  (h_equal : a r = a 1 ∧ a s = a 1) :
  r - s ≥ |m| := by sorry

end NUMINAMATH_CALUDE_recurrence_sequence_property_l3736_373630


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subsequence_l3736_373653

/-- An arithmetic sequence with the property that removing one term results in a geometric sequence -/
def ArithmeticSequenceWithGeometricSubsequence (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : Prop :=
  n ≥ 4 ∧ 
  d ≠ 0 ∧ 
  (∀ i, a i ≠ 0) ∧
  (∀ i, i < n → a (i + 1) = a i + d) ∧
  ∃ k, k < n ∧ 
    (∀ i j, i < j ∧ j < n ∧ i ≠ k ∧ j ≠ k → 
      (a j)^2 = a i * a (if j < k then j + 1 else j))

theorem arithmetic_sequence_with_geometric_subsequence 
  (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : 
  ArithmeticSequenceWithGeometricSubsequence n a d → 
  n = 4 ∧ (a 1 / d = -4 ∨ a 1 / d = 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subsequence_l3736_373653


namespace NUMINAMATH_CALUDE_solution_implies_q_value_l3736_373617

theorem solution_implies_q_value (q : ℚ) (h : 2 * q - 3 = 11) : q = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_q_value_l3736_373617


namespace NUMINAMATH_CALUDE_f_properties_l3736_373614

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - (m / 2) * x^2 + x

theorem f_properties (m : ℝ) :
  (m > 0 ∧ (∀ x > 0, f m x ≤ m * x - 1/2) → m ≥ 1) ∧
  (m = -1 → ∀ x₁ > 0, ∀ x₂ > 0, f m x₁ + f m x₂ = 0 → x₁ + x₂ ≥ Real.sqrt 3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3736_373614


namespace NUMINAMATH_CALUDE_square_of_sum_product_l3736_373668

theorem square_of_sum_product (a b c d A : ℤ) 
  (h1 : a^2 + A = b^2) (h2 : c^2 + A = d^2) : 
  ∃ n : ℕ, 2 * (a + b) * (c + d) * (a * c + b * d - A) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_product_l3736_373668


namespace NUMINAMATH_CALUDE_total_bricks_used_l3736_373608

/-- The number of walls being built -/
def number_of_walls : ℕ := 4

/-- The number of bricks in a single row of a wall -/
def bricks_per_row : ℕ := 60

/-- The number of rows in each wall -/
def rows_per_wall : ℕ := 100

/-- Theorem stating the total number of bricks used for all walls -/
theorem total_bricks_used :
  number_of_walls * bricks_per_row * rows_per_wall = 24000 := by
  sorry

end NUMINAMATH_CALUDE_total_bricks_used_l3736_373608


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l3736_373678

theorem complex_modulus_equation (n : ℝ) (hn : 0 < n) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l3736_373678


namespace NUMINAMATH_CALUDE_translation_theorem_l3736_373602

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- Translate a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

/-- The main theorem -/
theorem translation_theorem (m n : ℝ) :
  let p := Point.mk m n
  let p' := translateVertical (translateHorizontal p 2) 1
  p'.x = m + 2 ∧ p'.y = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l3736_373602


namespace NUMINAMATH_CALUDE_unique_function_solution_l3736_373604

/-- The functional equation f(x + f(y)) = x + y + k has exactly one solution. -/
theorem unique_function_solution :
  ∃! f : ℝ → ℝ, ∃ k : ℝ, ∀ x y : ℝ, f (x + f y) = x + y + k :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l3736_373604


namespace NUMINAMATH_CALUDE_absolute_value_and_exponents_l3736_373694

theorem absolute_value_and_exponents : |-3| + 2^2 - (Real.sqrt 3 - 1)^0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponents_l3736_373694


namespace NUMINAMATH_CALUDE_minimum_balls_to_draw_l3736_373663

theorem minimum_balls_to_draw (red green yellow blue white black : ℕ) 
  (h_red : red = 35) (h_green : green = 25) (h_yellow : yellow = 22) 
  (h_blue : blue = 15) (h_white : white = 14) (h_black : black = 12) :
  let total := red + green + yellow + blue + white + black
  let threshold := 18
  ∃ n : ℕ, n = 93 ∧ 
    (∀ m : ℕ, m < n → 
      ∃ (r g y b w k : ℕ), r + g + y + b + w + k = m ∧
        r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black ∧
        r < threshold ∧ g < threshold ∧ y < threshold ∧ 
        b < threshold ∧ w < threshold ∧ k < threshold) ∧
    (∀ m : ℕ, m ≥ n → 
      ∀ (r g y b w k : ℕ), r + g + y + b + w + k = m →
        r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black →
        r ≥ threshold ∨ g ≥ threshold ∨ y ≥ threshold ∨ 
        b ≥ threshold ∨ w ≥ threshold ∨ k ≥ threshold) :=
by sorry

end NUMINAMATH_CALUDE_minimum_balls_to_draw_l3736_373663


namespace NUMINAMATH_CALUDE_chicken_bucket_capacity_l3736_373632

/-- Represents the cost of a chicken bucket with sides in dollars -/
def bucket_cost : ℚ := 12

/-- Represents the total amount Monty spent in dollars -/
def total_spent : ℚ := 72

/-- Represents the number of family members Monty fed -/
def family_members : ℕ := 36

/-- Represents the number of people one chicken bucket with sides can feed -/
def people_per_bucket : ℕ := 6

/-- Proves that one chicken bucket with sides can feed 6 people -/
theorem chicken_bucket_capacity :
  (total_spent / bucket_cost) * people_per_bucket = family_members :=
by sorry

end NUMINAMATH_CALUDE_chicken_bucket_capacity_l3736_373632


namespace NUMINAMATH_CALUDE_count_decreasing_digit_numbers_l3736_373667

/-- A function that checks if a natural number has strictly decreasing digits. -/
def hasDecreasingDigits (n : ℕ) : Bool :=
  sorry

/-- The count of natural numbers with at least two digits and strictly decreasing digits. -/
def countDecreasingDigitNumbers : ℕ :=
  sorry

/-- Theorem stating that the count of natural numbers with at least two digits 
    and strictly decreasing digits is 1013. -/
theorem count_decreasing_digit_numbers :
  countDecreasingDigitNumbers = 1013 := by
  sorry

end NUMINAMATH_CALUDE_count_decreasing_digit_numbers_l3736_373667


namespace NUMINAMATH_CALUDE_range_of_a_l3736_373660

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, 2 * x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) : ¬(p a ∨ q a) → (a > 2 ∨ (-2 < a ∧ a < 1)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3736_373660


namespace NUMINAMATH_CALUDE_jade_savings_l3736_373659

/-- Calculates Jade's monthly savings given her earnings and spending patterns. -/
theorem jade_savings (monthly_earnings : ℝ) (living_expenses_ratio : ℝ) (insurance_ratio : ℝ) :
  monthly_earnings = 1600 →
  living_expenses_ratio = 0.75 →
  insurance_ratio = 1/5 →
  monthly_earnings * (1 - living_expenses_ratio - insurance_ratio) = 80 :=
by sorry

end NUMINAMATH_CALUDE_jade_savings_l3736_373659


namespace NUMINAMATH_CALUDE_range_of_a_l3736_373684

/-- The range of a given the conditions in the problem -/
theorem range_of_a (p q : Prop) (a : ℝ) 
  (hp : p ↔ ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0)
  (hq : q ↔ ∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0)
  (hpq_or : p ∨ q)
  (hpq_not_and : ¬(p ∧ q)) :
  a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 3 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l3736_373684


namespace NUMINAMATH_CALUDE_fraction_range_l3736_373666

theorem fraction_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hineq : a ≤ 2*b ∧ 2*b ≤ 2*a + b) :
  (4/9 : ℝ) ≤ (2*a*b)/(a^2 + 2*b^2) ∧ (2*a*b)/(a^2 + 2*b^2) ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_range_l3736_373666


namespace NUMINAMATH_CALUDE_monotonicity_when_a_eq_1_extreme_value_two_zero_points_l3736_373639

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2 + (2*a - 1) * x

-- State the theorems
theorem monotonicity_when_a_eq_1 :
  ∀ x y, 0 < x ∧ x < 1 ∧ 0 < y ∧ 1 < y → f 1 x < f 1 1 ∧ f 1 1 > f 1 y := by sorry

theorem extreme_value :
  ∀ a, a > 0 → ∃ x, x > 0 ∧ ∀ y, y > 0 → f a y ≤ f a x ∧ f a x = a * (Real.log a + a - 1) := by sorry

theorem two_zero_points :
  ∀ a, (∃ x y, 0 < x ∧ x < y ∧ f a x = 0 ∧ f a y = 0) ↔ a > 1 := by sorry

end

end NUMINAMATH_CALUDE_monotonicity_when_a_eq_1_extreme_value_two_zero_points_l3736_373639


namespace NUMINAMATH_CALUDE_center_sum_l3736_373612

/-- The center of a circle defined by the equation x^2 + y^2 = 4x - 6y + 9 -/
def circle_center : ℝ × ℝ := sorry

/-- The equation of the circle -/
axiom circle_equation (p : ℝ × ℝ) : p.1^2 + p.2^2 = 4*p.1 - 6*p.2 + 9

theorem center_sum : circle_center.1 + circle_center.2 = -1 := by sorry

end NUMINAMATH_CALUDE_center_sum_l3736_373612


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3736_373609

theorem geometric_sequence_fourth_term :
  ∀ x : ℚ,
  let a₁ := x
  let a₂ := 3*x + 3
  let a₃ := 5*x + 5
  let r := a₂ / a₁
  (a₂ = r * a₁) ∧ (a₃ = r * a₂) →
  let a₄ := r * a₃
  a₄ = -125/12 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3736_373609


namespace NUMINAMATH_CALUDE_system_solutions_l3736_373613

theorem system_solutions :
  ∀ x y : ℝ,
  (y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2) ∨ 
   (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3736_373613


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_44_l3736_373682

theorem smallest_four_digit_divisible_by_44 : 
  ∀ n : ℕ, 1000 ≤ n → n < 10000 → n % 44 = 0 → 1023 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_44_l3736_373682


namespace NUMINAMATH_CALUDE_aaron_earnings_l3736_373690

/-- Calculates the total earnings for Aaron over four days given his work hours and rates -/
theorem aaron_earnings (monday_hours : ℝ) (tuesday_minutes : ℝ) (wednesday_hours : ℝ) (thursday_minutes : ℝ) 
  (normal_rate : ℝ) (h_monday : monday_hours = 1.5) (h_tuesday : tuesday_minutes = 65) 
  (h_wednesday : wednesday_hours = 3) (h_thursday : thursday_minutes = 45) (h_rate : normal_rate = 4) :
  let tuesday_hours := tuesday_minutes / 60
  let thursday_hours := thursday_minutes / 60
  let monday_earnings := monday_hours * normal_rate
  let tuesday_earnings := tuesday_hours * normal_rate
  let wednesday_earnings := wednesday_hours * (2 * normal_rate)
  let thursday_earnings := thursday_hours * normal_rate
  monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings = 37.33 := by
sorry

end NUMINAMATH_CALUDE_aaron_earnings_l3736_373690


namespace NUMINAMATH_CALUDE_only_100_not_sum_of_four_consecutive_odds_l3736_373687

def is_sum_of_four_consecutive_odds (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 4 * k + 12 ∧ k % 2 = 1

theorem only_100_not_sum_of_four_consecutive_odds :
  ¬ is_sum_of_four_consecutive_odds 100 ∧
  (is_sum_of_four_consecutive_odds 16 ∧
   is_sum_of_four_consecutive_odds 40 ∧
   is_sum_of_four_consecutive_odds 72 ∧
   is_sum_of_four_consecutive_odds 200) :=
by sorry

end NUMINAMATH_CALUDE_only_100_not_sum_of_four_consecutive_odds_l3736_373687


namespace NUMINAMATH_CALUDE_ramanujan_number_l3736_373635

theorem ramanujan_number (r h : ℂ) : 
  r * h = 40 + 24 * I ∧ h = 7 + I → r = 28/5 + 64/25 * I :=
by sorry

end NUMINAMATH_CALUDE_ramanujan_number_l3736_373635


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_gt_3_l3736_373619

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1: A ∪ B = {x | 2 < x < 10}
theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

-- Theorem 2: (ℝ \ A) ∩ B = {x | 2 < x < 3 or 7 ≤ x < 10}
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a > 3
theorem intersection_A_C_nonempty_implies_a_gt_3 (a : ℝ) : (A ∩ C a).Nonempty → a > 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_gt_3_l3736_373619


namespace NUMINAMATH_CALUDE_tan_product_30_degrees_l3736_373692

theorem tan_product_30_degrees :
  let A : Real := 30 * π / 180
  let B : Real := 30 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_30_degrees_l3736_373692


namespace NUMINAMATH_CALUDE_min_fruits_in_platter_l3736_373673

/-- Represents the types of fruits --/
inductive Fruit
  | GreenApple
  | RedApple
  | YellowApple
  | RedOrange
  | YellowOrange
  | GreenKiwi
  | PurpleGrape
  | GreenGrape

/-- Represents the fruit platter --/
structure FruitPlatter :=
  (greenApples : ℕ)
  (redApples : ℕ)
  (yellowApples : ℕ)
  (redOranges : ℕ)
  (yellowOranges : ℕ)
  (greenKiwis : ℕ)
  (purpleGrapes : ℕ)
  (greenGrapes : ℕ)

/-- Checks if the platter satisfies all constraints --/
def isValidPlatter (p : FruitPlatter) : Prop :=
  p.greenApples + p.redApples + p.yellowApples ≥ 5 ∧
  p.redOranges + p.yellowOranges ≤ 5 ∧
  p.greenKiwis + p.purpleGrapes + p.greenGrapes ≥ 8 ∧
  p.greenKiwis + p.purpleGrapes + p.greenGrapes ≤ 12 ∧
  p.greenGrapes ≥ 1 ∧
  p.purpleGrapes ≥ 1 ∧
  p.greenApples * 2 = p.redApples ∧
  p.greenApples * 3 = p.yellowApples * 2 ∧
  p.redOranges = 1 ∧
  p.yellowOranges = 2 ∧
  p.greenKiwis = p.purpleGrapes

/-- Calculates the total number of fruits in the platter --/
def totalFruits (p : FruitPlatter) : ℕ :=
  p.greenApples + p.redApples + p.yellowApples +
  p.redOranges + p.yellowOranges +
  p.greenKiwis + p.purpleGrapes + p.greenGrapes

/-- Theorem stating that the minimum number of fruits in a valid platter is 30 --/
theorem min_fruits_in_platter :
  ∀ p : FruitPlatter, isValidPlatter p → totalFruits p ≥ 30 :=
sorry

end NUMINAMATH_CALUDE_min_fruits_in_platter_l3736_373673


namespace NUMINAMATH_CALUDE_second_quadrant_and_modulus_condition_l3736_373681

def complex_i : ℂ := Complex.I

theorem second_quadrant_and_modulus_condition (a : ℝ) : 
  let z₁ : ℂ := a + 2 / (1 - complex_i)
  let z₂ : ℂ := a - complex_i
  (z₁.re < 0 ∧ z₁.im > 0) → Complex.abs z₂ = 2 → a = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_and_modulus_condition_l3736_373681


namespace NUMINAMATH_CALUDE_odd_most_likely_l3736_373699

def box_size : Nat := 30

def is_multiple_of_10 (n : Nat) : Bool :=
  n % 10 = 0

def is_odd (n : Nat) : Bool :=
  n % 2 ≠ 0

def contains_digit_3 (n : Nat) : Bool :=
  ∃ d, d ∈ n.digits 10 ∧ d = 3

def is_multiple_of_5 (n : Nat) : Bool :=
  n % 5 = 0

def contains_digit_2 (n : Nat) : Bool :=
  ∃ d, d ∈ n.digits 10 ∧ d = 2

def count_satisfying (p : Nat → Bool) : Nat :=
  (List.range box_size).filter p |>.length

theorem odd_most_likely :
  count_satisfying is_odd >
  max
    (count_satisfying is_multiple_of_10)
    (max
      (count_satisfying contains_digit_3)
      (max
        (count_satisfying is_multiple_of_5)
        (count_satisfying contains_digit_2))) :=
by sorry

end NUMINAMATH_CALUDE_odd_most_likely_l3736_373699


namespace NUMINAMATH_CALUDE_max_integer_value_of_fraction_l3736_373674

theorem max_integer_value_of_fraction (x : ℝ) : 
  (4 * x^2 + 12 * x + 20) / (4 * x^2 + 12 * x + 8) < 12002 ∧ 
  ∀ ε > 0, ∃ y : ℝ, (4 * y^2 + 12 * y + 20) / (4 * y^2 + 12 * y + 8) > 12001 - ε :=
by sorry

#check max_integer_value_of_fraction

end NUMINAMATH_CALUDE_max_integer_value_of_fraction_l3736_373674


namespace NUMINAMATH_CALUDE_original_triangle_area_l3736_373637

theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ (side : ℝ), new_area = (4 * side)^2 * (original_area / side^2)) →
  new_area = 64 →
  original_area = 4 := by
sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3736_373637


namespace NUMINAMATH_CALUDE_shaded_area_of_square_pattern_l3736_373683

/-- Given a square with side length a, this theorem proves that the area of the shaded region
    formed by connecting vertices to midpoints of opposite sides in a pattern is (3/5) * a^2. -/
theorem shaded_area_of_square_pattern (a : ℝ) (h : a > 0) : ℝ :=
  let square_area := a^2
  let shaded_area := (3/5) * square_area
  shaded_area

#check shaded_area_of_square_pattern

end NUMINAMATH_CALUDE_shaded_area_of_square_pattern_l3736_373683


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3736_373645

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) + f (x - y) = x^2 + y^2) →
  (∀ x : ℝ, f x = x^2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3736_373645


namespace NUMINAMATH_CALUDE_alice_box_height_l3736_373624

/-- The height of the box Alice needs to reach the light bulb -/
def box_height (ceiling_height room_height alice_height alice_reach light_bulb_distance shelf_distance : ℝ) : ℝ :=
  ceiling_height - light_bulb_distance - (alice_height + alice_reach)

/-- Proof that Alice needs a 75 cm box to reach the light bulb -/
theorem alice_box_height :
  let ceiling_height : ℝ := 300  -- cm
  let room_height : ℝ := 300     -- cm
  let alice_height : ℝ := 160    -- cm
  let alice_reach : ℝ := 50      -- cm
  let light_bulb_distance : ℝ := 15  -- cm from ceiling
  let shelf_distance : ℝ := 10   -- cm below light bulb
  box_height ceiling_height room_height alice_height alice_reach light_bulb_distance shelf_distance = 75 := by
  sorry


end NUMINAMATH_CALUDE_alice_box_height_l3736_373624


namespace NUMINAMATH_CALUDE_sue_dogs_walked_l3736_373620

def perfume_cost : ℕ := 50
def christian_initial_savings : ℕ := 5
def sue_initial_savings : ℕ := 7
def yards_mowed : ℕ := 4
def yard_mowing_rate : ℕ := 5
def dog_walking_rate : ℕ := 2
def additional_needed : ℕ := 6

theorem sue_dogs_walked :
  ∃ (dogs_walked : ℕ),
    perfume_cost =
      christian_initial_savings + sue_initial_savings +
      yards_mowed * yard_mowing_rate +
      dogs_walked * dog_walking_rate +
      additional_needed ∧
    dogs_walked = 6 := by
  sorry

end NUMINAMATH_CALUDE_sue_dogs_walked_l3736_373620


namespace NUMINAMATH_CALUDE_cricket_player_average_increase_l3736_373638

/-- 
Theorem: Cricket Player's Average Increase

Given:
- A cricket player has played 10 innings
- The current average is 32 runs per innings
- The player needs to make 76 runs in the next innings

Prove: The increase in average is 4 runs per innings
-/
theorem cricket_player_average_increase 
  (innings : ℕ) 
  (current_average : ℚ) 
  (next_innings_runs : ℕ) 
  (h1 : innings = 10)
  (h2 : current_average = 32)
  (h3 : next_innings_runs = 76) : 
  (((innings : ℚ) * current_average + next_innings_runs) / (innings + 1) - current_average) = 4 := by
  sorry


end NUMINAMATH_CALUDE_cricket_player_average_increase_l3736_373638


namespace NUMINAMATH_CALUDE_smallest_positive_omega_l3736_373661

theorem smallest_positive_omega : ∃ ω : ℝ, ω > 0 ∧
  (∀ x : ℝ, Real.sin (ω * x - Real.pi / 4) = Real.cos (ω * (x - Real.pi / 2))) ∧
  (∀ ω' : ℝ, ω' > 0 → 
    (∀ x : ℝ, Real.sin (ω' * x - Real.pi / 4) = Real.cos (ω' * (x - Real.pi / 2))) → 
    ω ≤ ω') ∧
  ω = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_omega_l3736_373661


namespace NUMINAMATH_CALUDE_cauliflower_increase_l3736_373657

theorem cauliflower_increase (n : ℕ) (h : n^2 = 12544) : n^2 - (n-1)^2 = 223 := by
  sorry

end NUMINAMATH_CALUDE_cauliflower_increase_l3736_373657


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3736_373654

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (m : Line) (α β : Plane) :
  perpendicular m β → parallel m α → perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3736_373654


namespace NUMINAMATH_CALUDE_p_has_four_digits_l3736_373603

-- Define p as given in the problem
def p : ℚ := 125 * 243 * 16 / 405

-- Function to count the number of digits in a rational number
def count_digits (q : ℚ) : ℕ := sorry

-- Theorem stating that p has 4 digits
theorem p_has_four_digits : count_digits p = 4 := by sorry

end NUMINAMATH_CALUDE_p_has_four_digits_l3736_373603


namespace NUMINAMATH_CALUDE_same_solution_implies_a_value_l3736_373633

theorem same_solution_implies_a_value :
  ∀ x a : ℚ,
  (3 * x + 5 = 11) →
  (6 * x + 3 * a = 22) →
  a = 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_value_l3736_373633


namespace NUMINAMATH_CALUDE_three_sequence_inequality_l3736_373656

theorem three_sequence_inequality (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by sorry

end NUMINAMATH_CALUDE_three_sequence_inequality_l3736_373656


namespace NUMINAMATH_CALUDE_smallest_value_abcd_l3736_373655

theorem smallest_value_abcd (a b c d : ℤ) 
  (sum_condition : a + b + c + d < 25)
  (a_condition : a > 8)
  (b_condition : b < 5)
  (c_odd : c % 2 = 1)
  (d_even : d % 2 = 0) :
  (∀ a' b' c' d' : ℤ, 
    a' + b' + c' + d' < 25 → 
    a' > 8 → 
    b' < 5 → 
    c' % 2 = 1 → 
    d' % 2 = 0 → 
    a' - b' + c' - d' ≥ a - b + c - d) →
  a - b + c - d = -4 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_abcd_l3736_373655


namespace NUMINAMATH_CALUDE_games_in_23_team_tournament_l3736_373615

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- The number of games played in a single-elimination tournament -/
def games_played (t : Tournament) : ℕ := t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties, 
    the number of games played is 22 -/
theorem games_in_23_team_tournament (t : Tournament) 
  (h1 : t.num_teams = 23) (h2 : t.no_ties = true) : 
  games_played t = 22 := by
  sorry

end NUMINAMATH_CALUDE_games_in_23_team_tournament_l3736_373615


namespace NUMINAMATH_CALUDE_line_intersection_range_l3736_373618

/-- The line y = e^x + b has at most one common point with both f(x) = e^x and g(x) = ln(x) 
    if and only if b is in the closed interval [-2, 0] -/
theorem line_intersection_range (b : ℝ) : 
  (∀ x : ℝ, (∃! y : ℝ, y = Real.exp x + b ∧ (y = Real.exp x ∨ y = Real.log x)) ∨
            (∀ y : ℝ, y ≠ Real.exp x + b ∨ (y ≠ Real.exp x ∧ y ≠ Real.log x))) ↔ 
  b ∈ Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_range_l3736_373618


namespace NUMINAMATH_CALUDE_unique_solution_l3736_373670

theorem unique_solution (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (eq1 : x*y + y*z + z*x = 12)
  (eq2 : x*y*z = 2 + x + y + z) :
  x = 2 ∧ y = 2 ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l3736_373670


namespace NUMINAMATH_CALUDE_exactly_two_in_favor_l3736_373696

def probability_in_favor : ℝ := 0.6

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_in_favor :
  binomial_probability 4 2 probability_in_favor = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_in_favor_l3736_373696


namespace NUMINAMATH_CALUDE_investment_problem_l3736_373680

/-- Proves that the amount invested in the first account is approximately $2336.36 --/
theorem investment_problem (total_interest : ℝ) (second_account_investment : ℝ) 
  (interest_rate_difference : ℝ) (first_account_rate : ℝ) :
  total_interest = 1282 →
  second_account_investment = 8200 →
  interest_rate_difference = 0.015 →
  first_account_rate = 0.11 →
  ∃ x : ℝ, (x * first_account_rate + 
    second_account_investment * (first_account_rate + interest_rate_difference) = total_interest) ∧ 
    (abs (x - 2336.36) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l3736_373680


namespace NUMINAMATH_CALUDE_middle_group_frequency_l3736_373610

/-- Represents a frequency distribution histogram -/
structure FrequencyHistogram where
  rectangles : Fin 5 → ℝ
  total_sample_size : ℝ
  middle_rectangle_condition : rectangles 2 = (1/3) * (rectangles 0 + rectangles 1 + rectangles 3 + rectangles 4)
  total_area_condition : rectangles 0 + rectangles 1 + rectangles 2 + rectangles 3 + rectangles 4 = total_sample_size

/-- The theorem stating that the frequency of the middle group is 25 -/
theorem middle_group_frequency (h : FrequencyHistogram) (h_sample_size : h.total_sample_size = 100) :
  h.rectangles 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_middle_group_frequency_l3736_373610


namespace NUMINAMATH_CALUDE_abs_value_of_z_l3736_373641

theorem abs_value_of_z (z : ℂ) (h : z = Complex.I * (1 - Complex.I)) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_z_l3736_373641


namespace NUMINAMATH_CALUDE_household_survey_l3736_373665

/-- Proves that the total number of households surveyed is 240 given the specified conditions -/
theorem household_survey (neither_brand : ℕ) (only_A : ℕ) (both_brands : ℕ)
  (h1 : neither_brand = 80)
  (h2 : only_A = 60)
  (h3 : both_brands = 25) :
  neither_brand + only_A + 3 * both_brands + both_brands = 240 := by
sorry

end NUMINAMATH_CALUDE_household_survey_l3736_373665


namespace NUMINAMATH_CALUDE_max_product_constraint_l3736_373698

theorem max_product_constraint (x y : ℝ) : 
  x > 0 → y > 0 → x + 2 * y = 1 → x * y ≤ 1/8 := by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3736_373698


namespace NUMINAMATH_CALUDE_factorization_identity_l3736_373693

theorem factorization_identity (a : ℝ) : 3 * a^2 + 6 * a + 3 = 3 * (a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l3736_373693


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3736_373677

open Real

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ w x y z : ℝ, w > 0 → x > 0 → y > 0 → z > 0 → w * x = y * z →
    (f w)^2 + (f x)^2 / (f (y^2) + f (z^2)) = (w^2 + x^2) / (y^2 + z^2)

/-- The main theorem stating the form of functions satisfying the equation -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (h1 : ∀ x, x > 0 → f x > 0) 
    (h2 : SatisfiesEquation f) : 
    ∀ x, x > 0 → (f x = x ∨ f x = 1 / x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3736_373677


namespace NUMINAMATH_CALUDE_translation_vector_exponential_l3736_373685

/-- Given two functions f and g, where f(x) = 2^x + 1 and g(x) = 2^(x+1),
    prove that the translation vector (h, k) that transforms the graph of f
    into the graph of g is (-1, -1). -/
theorem translation_vector_exponential (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 2^x + 1)
  (hg : ∀ x, g x = 2^(x+1))
  (h k : ℝ)
  (translation : ∀ x, g x = f (x - h) + k) :
  h = -1 ∧ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_translation_vector_exponential_l3736_373685


namespace NUMINAMATH_CALUDE_first_digit_value_l3736_373616

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem first_digit_value (x y : ℕ) : 
  x < 10 → 
  y < 10 → 
  is_divisible_by (653 * 100 + x * 10 + y) 80 → 
  x + y = 2 → 
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_value_l3736_373616


namespace NUMINAMATH_CALUDE_figure_18_to_square_l3736_373611

/-- Represents a figure on a graph paper -/
structure Figure where
  area : ℕ

/-- Represents a cut of the figure -/
structure Cut where
  parts : ℕ

/-- Represents the result of rearranging the cut parts -/
structure Rearrangement where
  is_square : Bool

/-- Function to determine if a figure can be cut and rearranged into a square -/
def can_form_square (f : Figure) (c : Cut) : Prop :=
  ∃ (r : Rearrangement), r.is_square = true

/-- Theorem stating that a figure with area 18 can be cut into 3 parts and rearranged into a square -/
theorem figure_18_to_square :
  ∀ (f : Figure) (c : Cut), 
    f.area = 18 → c.parts = 3 → can_form_square f c :=
by sorry

end NUMINAMATH_CALUDE_figure_18_to_square_l3736_373611


namespace NUMINAMATH_CALUDE_chapters_read_l3736_373688

theorem chapters_read (num_books : ℕ) (chapters_per_book : ℕ) (total_chapters : ℕ) : 
  num_books = 10 → chapters_per_book = 24 → total_chapters = num_books * chapters_per_book →
  total_chapters = 240 :=
by sorry

end NUMINAMATH_CALUDE_chapters_read_l3736_373688


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3736_373679

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, -9)
  parallel a b → x = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3736_373679


namespace NUMINAMATH_CALUDE_log_base_2_derivative_l3736_373669

open Real

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => log x / log 2) x = 1 / (x * log 2) := by
sorry

end NUMINAMATH_CALUDE_log_base_2_derivative_l3736_373669


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3736_373627

theorem absolute_value_inequality (x : ℝ) : 2 ≤ |x - 5| ∧ |x - 5| ≤ 4 ↔ x ∈ Set.Icc 1 3 ∪ Set.Icc 7 9 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3736_373627


namespace NUMINAMATH_CALUDE_range_of_a_l3736_373625

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3736_373625


namespace NUMINAMATH_CALUDE_gcd_1721_1733_l3736_373662

theorem gcd_1721_1733 : Nat.gcd 1721 1733 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1721_1733_l3736_373662


namespace NUMINAMATH_CALUDE_article_cost_price_l3736_373628

theorem article_cost_price (profit_percent : ℝ) (discount_percent : ℝ) (price_reduction : ℝ) (new_profit_percent : ℝ) :
  profit_percent = 25 →
  discount_percent = 20 →
  price_reduction = 8.40 →
  new_profit_percent = 30 →
  ∃ (cost : ℝ), 
    cost > 0 ∧
    (cost + profit_percent / 100 * cost) - price_reduction = 
    (cost * (1 - discount_percent / 100)) * (1 + new_profit_percent / 100) ∧
    cost = 40 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l3736_373628


namespace NUMINAMATH_CALUDE_tangent_line_and_root_condition_l3736_373643

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- State the theorem
theorem tangent_line_and_root_condition (x : ℝ) :
  -- The tangent line at (2, 7)
  (∃ (m b : ℝ), f 2 = 7 ∧ 
    (∀ x, f x = m * x + b) ∧
    m = 12 ∧ b = -17) ∧
  -- Condition for three distinct real roots
  (∀ m : ℝ, (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔
    -3 < m ∧ m < -2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_root_condition_l3736_373643


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l3736_373658

theorem quadratic_equation_proof (m : ℝ) (h1 : m < 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + m
  (∃ x : ℝ, f x = 0 ∧ x = -1) →
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ∧  -- two distinct real roots
  m = -3 ∧                                  -- value of m
  (∃ x : ℝ, f x = 0 ∧ x = 3)                -- other root
:= by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l3736_373658


namespace NUMINAMATH_CALUDE_negation_equivalence_l3736_373675

-- Define the original proposition
def original_proposition : Prop := ∀ x : ℝ, x > Real.sin x

-- Define the negation of the original proposition
def negation_proposition : Prop := ∃ x : ℝ, x ≤ Real.sin x

-- Theorem stating that the negation of the original proposition is equivalent to the negation_proposition
theorem negation_equivalence : ¬original_proposition ↔ negation_proposition := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3736_373675


namespace NUMINAMATH_CALUDE_route_upper_bound_l3736_373672

/-- Represents the number of possible routes in a grid city -/
def f (m n : ℕ) : ℕ := sorry

/-- Theorem: The number of possible routes in a grid city is at most 2^(m*n) -/
theorem route_upper_bound (m n : ℕ) : f m n ≤ 2^(m*n) := by sorry

end NUMINAMATH_CALUDE_route_upper_bound_l3736_373672


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l3736_373644

theorem sum_reciprocal_inequality (u v w : ℝ) (h : u + v + w = 3) :
  1 / (u^2 + 7) + 1 / (v^2 + 7) + 1 / (w^2 + 7) ≤ 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l3736_373644


namespace NUMINAMATH_CALUDE_pyramid_volume_approx_l3736_373652

-- Define the pyramid
structure Pyramid where
  base_area : ℝ
  face1_area : ℝ
  face2_area : ℝ

-- Define the volume function
def pyramid_volume (p : Pyramid) : ℝ :=
  -- The actual calculation is not implemented, as per instructions
  sorry

-- Theorem statement
theorem pyramid_volume_approx (p : Pyramid) 
  (h1 : p.base_area = 144) 
  (h2 : p.face1_area = 72) 
  (h3 : p.face2_area = 54) : 
  ∃ (ε : ℝ), abs (pyramid_volume p - 518.76) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_volume_approx_l3736_373652


namespace NUMINAMATH_CALUDE_alyssa_car_wash_earnings_l3736_373621

/-- The amount Alyssa earned from washing the family car -/
def car_wash_earnings (weekly_allowance : ℝ) (movie_spending_fraction : ℝ) (final_amount : ℝ) : ℝ :=
  final_amount - (weekly_allowance * (1 - movie_spending_fraction))

/-- Theorem: Alyssa earned 8 dollars from washing the family car -/
theorem alyssa_car_wash_earnings :
  car_wash_earnings 8 0.5 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_car_wash_earnings_l3736_373621


namespace NUMINAMATH_CALUDE_difference_of_squares_101_99_l3736_373648

theorem difference_of_squares_101_99 : 101^2 - 99^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_101_99_l3736_373648


namespace NUMINAMATH_CALUDE_line_l_and_symmetrical_line_l3736_373646

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Define line l
def l (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the symmetrical line
def symmetrical_l (x y : ℝ) : Prop := 2 * x + y - 2 = 0

theorem line_l_and_symmetrical_line : 
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = P) → 
  (∀ x y : ℝ, l x y → line3 (y + 2) (-x - 1)) →
  (∀ x y : ℝ, l x y ↔ 2 * x + y + 2 = 0) ∧
  (∀ x y : ℝ, symmetrical_l x y ↔ 2 * x + y - 2 = 0) := by sorry

end NUMINAMATH_CALUDE_line_l_and_symmetrical_line_l3736_373646


namespace NUMINAMATH_CALUDE_fraction_division_simplification_l3736_373622

theorem fraction_division_simplification : (3 / 4) / (5 / 6) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_simplification_l3736_373622


namespace NUMINAMATH_CALUDE_circle_k_range_l3736_373629

-- Define the equation
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 4*k + 1 = 0

-- Define what it means for the equation to represent a circle
def is_circle (k : ℝ) : Prop :=
  ∃ (h r : ℝ), r > 0 ∧ ∀ (x y : ℝ),
    circle_equation x y k ↔ (x + 2)^2 + (y + 1)^2 = r^2

-- Theorem statement
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_k_range_l3736_373629


namespace NUMINAMATH_CALUDE_camping_trip_attendance_l3736_373691

/-- The percentage of students who went to the camping trip -/
def camping_trip_percentage : ℝ := 14

/-- The percentage of students who went to the music festival -/
def music_festival_percentage : ℝ := 8

/-- The percentage of students who participated in the sports league -/
def sports_league_percentage : ℝ := 6

/-- The percentage of camping trip attendees who spent more than $100 -/
def camping_trip_high_cost_percentage : ℝ := 60

/-- The percentage of music festival attendees who spent more than $90 -/
def music_festival_high_cost_percentage : ℝ := 80

/-- The percentage of sports league participants who paid more than $70 -/
def sports_league_high_cost_percentage : ℝ := 75

theorem camping_trip_attendance : 
  camping_trip_percentage = 14 := by sorry

end NUMINAMATH_CALUDE_camping_trip_attendance_l3736_373691


namespace NUMINAMATH_CALUDE_sara_pumpkins_l3736_373695

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The number of pumpkins Sara has left -/
def pumpkins_left : ℕ := 20

/-- The original number of pumpkins Sara grew -/
def original_pumpkins : ℕ := pumpkins_eaten + pumpkins_left

theorem sara_pumpkins : original_pumpkins = 43 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l3736_373695


namespace NUMINAMATH_CALUDE_quadratic_point_m_l3736_373636

/-- Given a quadratic function y = -ax² + 2ax + 3 where a > 0,
    if the point (m, 3) lies on the graph and m ≠ 0, then m = 2. -/
theorem quadratic_point_m (a m : ℝ) (ha : a > 0) (hm : m ≠ 0) :
  (3 = -a * m^2 + 2 * a * m + 3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_m_l3736_373636


namespace NUMINAMATH_CALUDE_compounded_growth_rate_l3736_373626

/-- Given an initial investment P that grows by k% in the first year and m% in the second year,
    the compounded rate of growth R after two years is equal to k + m + (km/100). -/
theorem compounded_growth_rate (P k m : ℝ) (hP : P > 0) (hk : k ≥ 0) (hm : m ≥ 0) :
  let R := k + m + (k * m) / 100
  let growth_factor := (1 + k / 100) * (1 + m / 100)
  R = (growth_factor - 1) * 100 :=
by sorry

end NUMINAMATH_CALUDE_compounded_growth_rate_l3736_373626


namespace NUMINAMATH_CALUDE_total_money_calculation_l3736_373606

theorem total_money_calculation (total_notes : ℕ) 
  (denominations : Fin 3 → ℕ) 
  (h1 : total_notes = 75) 
  (h2 : denominations 0 = 1 ∧ denominations 1 = 5 ∧ denominations 2 = 10) : 
  (total_notes / 3) * (denominations 0 + denominations 1 + denominations 2) = 400 :=
by
  sorry

#check total_money_calculation

end NUMINAMATH_CALUDE_total_money_calculation_l3736_373606


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3736_373697

theorem simplify_square_roots : 
  (Real.sqrt 338 / Real.sqrt 288) + (Real.sqrt 150 / Real.sqrt 96) = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3736_373697


namespace NUMINAMATH_CALUDE_surface_area_reduction_approx_l3736_373671

/-- The number of faces in a single cube -/
def cube_faces : ℕ := 6

/-- The number of faces lost when splicing two cubes into a cuboid -/
def faces_lost : ℕ := 2

/-- The percentage reduction in surface area when splicing two cubes into a cuboid -/
def surface_area_reduction : ℚ :=
  (faces_lost : ℚ) / (2 * cube_faces : ℚ) * 100

theorem surface_area_reduction_approx :
  ∃ ε > 0, abs (surface_area_reduction - 167/10) < ε ∧ ε < 1/10 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_reduction_approx_l3736_373671


namespace NUMINAMATH_CALUDE_race_distance_multiple_of_360_l3736_373686

/-- Represents a race between two contestants A and B -/
structure Race where
  speedRatio : Rat  -- Ratio of speeds of A to B
  headStart : ℕ     -- Head start distance for A in meters
  winMargin : ℕ     -- Distance by which A wins in meters

/-- The total distance of the race is a multiple of 360 meters -/
theorem race_distance_multiple_of_360 (race : Race) 
  (h1 : race.speedRatio = 3 / 4)
  (h2 : race.headStart = 140)
  (h3 : race.winMargin = 20) :
  ∃ (k : ℕ), race.headStart + race.winMargin + k * 360 = 
    race.headStart + (4 * (race.headStart + race.winMargin)) / 3 :=
sorry

end NUMINAMATH_CALUDE_race_distance_multiple_of_360_l3736_373686


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3736_373664

theorem quadratic_equation_properties (k : ℝ) :
  let f (x : ℝ) := x^2 + (2*k - 1)*x - k - 1
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (x₁ + x₂ - 4*x₁*x₂ = 2 → k = -3/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3736_373664


namespace NUMINAMATH_CALUDE_det_of_matrix_l3736_373631

def matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![2, -1, 4;
     0,  6, -3;
     3,  0,  1]

theorem det_of_matrix : Matrix.det matrix = -51 := by
  sorry

end NUMINAMATH_CALUDE_det_of_matrix_l3736_373631


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3736_373607

theorem inequality_solution_set (x : ℝ) :
  (2 < (1 / (x - 1)) ∧ (1 / (x - 1)) < 3 ∧ 0 < x - 1) ↔ (4/3 < x ∧ x < 3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3736_373607


namespace NUMINAMATH_CALUDE_bob_investment_l3736_373623

theorem bob_investment (interest_rate_1 interest_rate_2 total_interest investment_1 : ℝ)
  (h1 : interest_rate_1 = 0.18)
  (h2 : interest_rate_2 = 0.14)
  (h3 : total_interest = 3360)
  (h4 : investment_1 = 7000)
  (h5 : investment_1 * interest_rate_1 + (total_investment - investment_1) * interest_rate_2 = total_interest) :
  ∃ (total_investment : ℝ), total_investment = 22000 := by
sorry

end NUMINAMATH_CALUDE_bob_investment_l3736_373623


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_leq_neg_two_l3736_373601

/-- The function f(x) = x^2 - 2ax + a^2 - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 1

/-- The theorem stating that if the solution set of f[f(x)] < 0 is empty, then a ≤ -2 -/
theorem empty_solution_set_implies_a_leq_neg_two (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) → a ≤ -2 := by
  sorry


end NUMINAMATH_CALUDE_empty_solution_set_implies_a_leq_neg_two_l3736_373601


namespace NUMINAMATH_CALUDE_henrys_cookbooks_l3736_373689

theorem henrys_cookbooks (initial_books : ℕ) (boxes : ℕ) (books_per_box : ℕ)
  (room_books : ℕ) (coffee_table_books : ℕ) (new_books : ℕ) (final_books : ℕ)
  (h1 : initial_books = 99)
  (h2 : boxes = 3)
  (h3 : books_per_box = 15)
  (h4 : room_books = 21)
  (h5 : coffee_table_books = 4)
  (h6 : new_books = 12)
  (h7 : final_books = 23) :
  initial_books - (boxes * books_per_box + room_books + coffee_table_books) - (final_books - new_books) = 18 := by
sorry

end NUMINAMATH_CALUDE_henrys_cookbooks_l3736_373689


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l3736_373650

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The total number of possible triangles formed by choosing 3 vertices from n vertices -/
def total_triangles : ℕ := n.choose 3

/-- The number of triangles with one side being a side of the decagon -/
def triangles_one_side : ℕ := n * (n - 4)

/-- The number of triangles with two sides being sides of the decagon -/
def triangles_two_sides : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side being a side of the decagon) -/
def favorable_outcomes : ℕ := triangles_one_side + triangles_two_sides

/-- The probability of a randomly chosen triangle having at least one side that is also a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l3736_373650


namespace NUMINAMATH_CALUDE_sum_square_gt_four_times_adjacent_products_l3736_373605

theorem sum_square_gt_four_times_adjacent_products 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 > 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

end NUMINAMATH_CALUDE_sum_square_gt_four_times_adjacent_products_l3736_373605


namespace NUMINAMATH_CALUDE_interview_score_is_85_l3736_373676

/-- Calculate the interview score based on individual scores and their proportions -/
def interview_score (basic_knowledge : ℝ) (communication_skills : ℝ) (work_attitude : ℝ) 
  (basic_prop : ℝ) (comm_prop : ℝ) (attitude_prop : ℝ) : ℝ :=
  basic_knowledge * basic_prop + communication_skills * comm_prop + work_attitude * attitude_prop

/-- Theorem: The interview score for the given scores and proportions is 85 points -/
theorem interview_score_is_85 :
  interview_score 85 80 88 0.2 0.3 0.5 = 85 := by
  sorry

#eval interview_score 85 80 88 0.2 0.3 0.5

end NUMINAMATH_CALUDE_interview_score_is_85_l3736_373676


namespace NUMINAMATH_CALUDE_multiplication_division_equality_l3736_373640

theorem multiplication_division_equality : 15 * (1 / 5) * 40 / 4 = 30 := by sorry

end NUMINAMATH_CALUDE_multiplication_division_equality_l3736_373640


namespace NUMINAMATH_CALUDE_log_properties_l3736_373651

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem statement
theorem log_properties (a M N x : ℝ) 
  (ha : a > 0 ∧ a ≠ 1) (hM : M > 0) (hN : N > 0) :
  (log a (a^x) = x) ∧
  (log a (M / N) = log a M - log a N) ∧
  (log a (M * N) = log a M + log a N) := by
  sorry

end NUMINAMATH_CALUDE_log_properties_l3736_373651


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l3736_373600

theorem min_product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≥ 1/18 :=
by sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l3736_373600


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3736_373649

variable (z : ℂ)

theorem complex_equation_solution :
  (1 - Complex.I) * z = 2 * Complex.I →
  z = -1 + Complex.I ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3736_373649


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3736_373642

-- Define an isosceles triangle with sides 4, 8, and 8
def isosceles_triangle (a b c : ℝ) : Prop :=
  a = 4 ∧ b = 8 ∧ c = 8 ∧ b = c

-- Define the perimeter of a triangle
def triangle_perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, isosceles_triangle a b c → triangle_perimeter a b c = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3736_373642


namespace NUMINAMATH_CALUDE_social_gathering_attendance_l3736_373647

theorem social_gathering_attendance
  (num_men : ℕ)
  (women_per_man : ℕ)
  (men_per_woman : ℕ)
  (h_num_men : num_men = 15)
  (h_women_per_man : women_per_man = 4)
  (h_men_per_woman : men_per_woman = 3) :
  (num_men * women_per_man) / men_per_woman = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_social_gathering_attendance_l3736_373647


namespace NUMINAMATH_CALUDE_fraction_simplification_l3736_373634

theorem fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  (9 * x^3 * y^2) / (12 * x^2 * y^4) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3736_373634
