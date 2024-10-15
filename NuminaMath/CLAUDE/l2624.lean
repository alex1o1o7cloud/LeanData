import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_expansion_l2624_262425

theorem polynomial_expansion :
  ∀ x : ℝ, (4 * x^3 - 3 * x^2 + 2 * x + 7) * (5 * x^4 + x^3 - 3 * x + 9) =
    20 * x^7 - 27 * x^5 + 8 * x^4 + 45 * x^3 - 4 * x^2 + 51 * x + 196 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2624_262425


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l2624_262417

/-- The function f(x) = x³ --/
def f (x : ℝ) : ℝ := x^3

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_line_at_x_1 :
  let m := f 1
  let slope := f' 1
  (fun x y => y - m = slope * (x - 1)) = (fun x y => y = 3 * x - 2) := by
    sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l2624_262417


namespace NUMINAMATH_CALUDE_abc_value_l2624_262440

theorem abc_value (a b c : ℂ) 
  (eq1 : 2 * a * b + 3 * b = -21)
  (eq2 : 2 * b * c + 3 * c = -21)
  (eq3 : 2 * c * a + 3 * a = -21) :
  a * b * c = 105.75 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l2624_262440


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2624_262436

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  property1 : a 3 * a 7 = 8
  property2 : a 4 + a 6 = 6

/-- Theorem: For a geometric sequence satisfying the given properties, a_2 + a_8 = 9 -/
theorem geometric_sequence_sum (seq : GeometricSequence) : seq.a 2 + seq.a 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2624_262436


namespace NUMINAMATH_CALUDE_cubic_factorization_l2624_262423

theorem cubic_factorization (x : ℝ) : -3*x + 6*x^2 - 3*x^3 = -3*x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2624_262423


namespace NUMINAMATH_CALUDE_max_value_product_sum_l2624_262405

theorem max_value_product_sum (X Y Z : ℕ) (h : X + Y + Z = 15) :
  (∀ A B C : ℕ, A + B + C = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ A * B * C + A * B + B * C + C * A) →
  X * Y * Z + X * Y + Y * Z + Z * X = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l2624_262405


namespace NUMINAMATH_CALUDE_rectangle_area_l2624_262494

/-- The area of a rectangle with vertices at (-7, 1), (1, 1), (1, -6), and (-7, -6) in a rectangular coordinate system is 56 square units. -/
theorem rectangle_area : ℝ := by
  -- Define the vertices of the rectangle
  let v1 : ℝ × ℝ := (-7, 1)
  let v2 : ℝ × ℝ := (1, 1)
  let v3 : ℝ × ℝ := (1, -6)
  let v4 : ℝ × ℝ := (-7, -6)

  -- Calculate the length and width of the rectangle
  let length : ℝ := v2.1 - v1.1
  let width : ℝ := v1.2 - v4.2

  -- Calculate the area of the rectangle
  let area : ℝ := length * width

  -- Prove that the area is equal to 56
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2624_262494


namespace NUMINAMATH_CALUDE_exists_number_with_2001_trailing_zeros_l2624_262402

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of all divisors of a natural number -/
def productOfDivisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with 2001 trailing zeros in its product of divisors -/
theorem exists_number_with_2001_trailing_zeros : 
  ∃ n : ℕ, trailingZeros (productOfDivisors n) = 2001 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_2001_trailing_zeros_l2624_262402


namespace NUMINAMATH_CALUDE_area_triangle_BCD_l2624_262413

/-- Given a triangle ABC with area 50 square units and base AC of 6 units,
    and an extension of AC to point D such that CD is 36 units long,
    prove that the area of triangle BCD is 300 square units. -/
theorem area_triangle_BCD (h : ℝ) : 
  (1/2 : ℝ) * 6 * h = 50 →  -- Area of triangle ABC
  (1/2 : ℝ) * 36 * h = 300  -- Area of triangle BCD
  := by sorry

end NUMINAMATH_CALUDE_area_triangle_BCD_l2624_262413


namespace NUMINAMATH_CALUDE_peach_difference_l2624_262467

def steven_peaches : ℕ := 13
def jake_peaches : ℕ := 7

theorem peach_difference : steven_peaches - jake_peaches = 6 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l2624_262467


namespace NUMINAMATH_CALUDE_binomial_510_510_l2624_262468

theorem binomial_510_510 : Nat.choose 510 510 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_510_510_l2624_262468


namespace NUMINAMATH_CALUDE_value_of_b_plus_a_l2624_262459

theorem value_of_b_plus_a (a b : ℝ) : 
  (abs a = 8) → 
  (abs b = 2) → 
  (abs (a - b) = b - a) → 
  ((b + a = -6) ∨ (b + a = -10)) := by
sorry

end NUMINAMATH_CALUDE_value_of_b_plus_a_l2624_262459


namespace NUMINAMATH_CALUDE_mike_arcade_time_mike_play_time_l2624_262408

/-- Given Mike's weekly pay and arcade expenses, calculate his play time in minutes -/
theorem mike_arcade_time (weekly_pay : ℕ) (food_cost : ℕ) (hourly_rate : ℕ) : ℕ :=
  let arcade_budget := weekly_pay / 2
  let token_budget := arcade_budget - food_cost
  let play_hours := token_budget / hourly_rate
  play_hours * 60

/-- Prove that Mike can play for 300 minutes given the specific conditions -/
theorem mike_play_time :
  mike_arcade_time 100 10 8 = 300 := by
  sorry

end NUMINAMATH_CALUDE_mike_arcade_time_mike_play_time_l2624_262408


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_solution_l2624_262442

theorem smallest_four_digit_congruence_solution :
  let x : ℕ := 1001
  (∀ y : ℕ, 1000 ≤ y ∧ y < x →
    ¬(11 * y ≡ 33 [ZMOD 22] ∧
      3 * y + 10 ≡ 19 [ZMOD 12] ∧
      5 * y - 3 ≡ 2 * y [ZMOD 36] ∧
      y ≡ 3 [ZMOD 4])) ∧
  (11 * x ≡ 33 [ZMOD 22] ∧
   3 * x + 10 ≡ 19 [ZMOD 12] ∧
   5 * x - 3 ≡ 2 * x [ZMOD 36] ∧
   x ≡ 3 [ZMOD 4]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_solution_l2624_262442


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l2624_262453

theorem least_four_digit_multiple : ∀ n : ℕ,
  (1000 ≤ n) →
  (n % 3 = 0) →
  (n % 4 = 0) →
  (n % 9 = 0) →
  1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_l2624_262453


namespace NUMINAMATH_CALUDE_division_remainder_l2624_262477

theorem division_remainder (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l2624_262477


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2624_262458

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 3}

-- Theorem statement
theorem intersection_A_complement_B : 
  A ∩ (U \ B) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2624_262458


namespace NUMINAMATH_CALUDE_FMF_better_than_MFM_l2624_262460

/-- Represents the probability of winning a tennis match against a parent. -/
structure ParentProbability where
  /-- The probability of winning against the parent. -/
  prob : ℝ
  /-- The probability is between 0 and 1. -/
  prob_between_zero_and_one : 0 ≤ prob ∧ prob ≤ 1

/-- Calculates the probability of winning in a Father-Mother-Father (FMF) sequence. -/
def prob_win_FMF (p q : ParentProbability) : ℝ :=
  2 * p.prob * q.prob - p.prob * q.prob^2

/-- Calculates the probability of winning in a Mother-Father-Mother (MFM) sequence. -/
def prob_win_MFM (p q : ParentProbability) : ℝ :=
  2 * p.prob * q.prob - p.prob^2 * q.prob

/-- 
Theorem: The probability of winning in the Father-Mother-Father (FMF) sequence
is higher than the probability of winning in the Mother-Father-Mother (MFM) sequence,
given that the probability of winning against the father is less than
the probability of winning against the mother.
-/
theorem FMF_better_than_MFM (p q : ParentProbability) 
  (h : p.prob < q.prob) : prob_win_FMF p q > prob_win_MFM p q := by
  sorry


end NUMINAMATH_CALUDE_FMF_better_than_MFM_l2624_262460


namespace NUMINAMATH_CALUDE_parabola_intersects_line_segment_l2624_262437

/-- Parabola C_m: y = x^2 - mx + m + 1 -/
def C_m (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m + 1

/-- Line segment AB with endpoints A(0,4) and B(4,0) -/
def line_AB (x : ℝ) : ℝ := -x + 4

/-- The parabola C_m intersects the line segment AB at exactly two points
    if and only if m is in the range [3, 17/3] -/
theorem parabola_intersects_line_segment (m : ℝ) :
  (∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 ∧
   C_m m x₁ = line_AB x₁ ∧ C_m m x₂ = line_AB x₂ ∧
   ∀ x, 0 ≤ x ∧ x ≤ 4 → C_m m x = line_AB x → (x = x₁ ∨ x = x₂)) ↔
  (3 ≤ m ∧ m ≤ 17/3) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersects_line_segment_l2624_262437


namespace NUMINAMATH_CALUDE_biquadratic_root_negation_l2624_262455

/-- A biquadratic equation is of the form ax^4 + bx^2 + c = 0 -/
def BiquadraticEquation (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^4 + b * x^2 + c = 0

/-- If α is a root of a biquadratic equation, then -α is also a root -/
theorem biquadratic_root_negation (a b c α : ℝ) :
  BiquadraticEquation a b c α → BiquadraticEquation a b c (-α) :=
by sorry

end NUMINAMATH_CALUDE_biquadratic_root_negation_l2624_262455


namespace NUMINAMATH_CALUDE_folded_paper_cut_ratio_l2624_262438

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem folded_paper_cut_ratio :
  let original_side : ℝ := 6
  let folded_paper := Rectangle.mk original_side (original_side / 2)
  let large_rectangle := folded_paper
  let small_rectangle := Rectangle.mk (original_side / 2) (original_side / 2)
  (perimeter small_rectangle) / (perimeter large_rectangle) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_cut_ratio_l2624_262438


namespace NUMINAMATH_CALUDE_min_value_y_l2624_262403

noncomputable def y (x a : ℝ) : ℝ := (Real.exp x - a)^2 + (Real.exp (-x) - a)^2

theorem min_value_y (a : ℝ) (h : a ≠ 0) :
  (∀ x, y x a ≥ (if a ≥ 2 then a^2 - 2 else 2*(a-1)^2)) ∧
  (∃ x, y x a = (if a ≥ 2 then a^2 - 2 else 2*(a-1)^2)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_y_l2624_262403


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l2624_262490

/-- A predicate that determines if three positive real numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating the triangle inequality for forming a triangle -/
theorem triangle_inequality_theorem (a b c : ℝ) :
  can_form_triangle a b c ↔ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l2624_262490


namespace NUMINAMATH_CALUDE_bank_deposit_l2624_262479

theorem bank_deposit (P : ℝ) (interest_rate : ℝ) (years : ℕ) (final_amount : ℝ) : 
  interest_rate = 0.1 →
  years = 2 →
  final_amount = 121 →
  P * (1 + interest_rate) ^ years = final_amount →
  P = 100 := by
sorry

end NUMINAMATH_CALUDE_bank_deposit_l2624_262479


namespace NUMINAMATH_CALUDE_expanded_parallelepiped_volume_l2624_262483

/-- The volume of a set of points inside or within one unit of a rectangular parallelepiped -/
def volume_expanded_parallelepiped (a b c : ℝ) : ℝ :=
  (a + 2) * (b + 2) * (c + 2) - (a * b * c)

/-- Represents the condition that two natural numbers are coprime -/
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem expanded_parallelepiped_volume 
  (m n p : ℕ) 
  (h_positive : m > 0 ∧ n > 0 ∧ p > 0) 
  (h_coprime : coprime n p) 
  (h_volume : volume_expanded_parallelepiped 2 3 4 = (m + n * Real.pi) / p) :
  m + n + p = 262 := by
  sorry

end NUMINAMATH_CALUDE_expanded_parallelepiped_volume_l2624_262483


namespace NUMINAMATH_CALUDE_bird_count_l2624_262470

/-- The number of birds in a crape myrtle tree --/
theorem bird_count (swallows bluebirds cardinals : ℕ) : 
  swallows = 2 →
  bluebirds = 2 * swallows →
  cardinals = 3 * bluebirds →
  swallows + bluebirds + cardinals = 18 := by
sorry

end NUMINAMATH_CALUDE_bird_count_l2624_262470


namespace NUMINAMATH_CALUDE_octagon_theorem_l2624_262484

def is_permutation (l : List ℕ) : Prop :=
  l.length = 8 ∧ l.toFinset = Finset.range 8

def cyclic_shift (l : List ℕ) (k : ℕ) : List ℕ :=
  (l.drop k ++ l.take k).take 8

def product_sum (l1 l2 : List ℕ) : ℕ :=
  List.sum (List.zipWith (· * ·) l1 l2)

theorem octagon_theorem (l1 l2 : List ℕ) (h1 : is_permutation l1) (h2 : is_permutation l2) :
  ∃ k, product_sum l1 (cyclic_shift l2 k) ≥ 162 := by
  sorry

end NUMINAMATH_CALUDE_octagon_theorem_l2624_262484


namespace NUMINAMATH_CALUDE_relay_race_sequences_l2624_262469

/-- Represents the number of athletes in the relay race -/
def numAthletes : ℕ := 4

/-- Represents the set of all possible permutations of athletes -/
def allPermutations : ℕ := Nat.factorial numAthletes

/-- Represents the number of permutations where athlete A runs the first leg -/
def permutationsAFirst : ℕ := Nat.factorial (numAthletes - 1)

/-- Represents the number of permutations where athlete B runs the fourth leg -/
def permutationsBLast : ℕ := Nat.factorial (numAthletes - 1)

/-- Represents the number of permutations where A runs first and B runs last -/
def permutationsAFirstBLast : ℕ := Nat.factorial (numAthletes - 2)

/-- The theorem stating the number of valid sequences in the relay race -/
theorem relay_race_sequences :
  allPermutations - permutationsAFirst - permutationsBLast + permutationsAFirstBLast = 14 := by
  sorry


end NUMINAMATH_CALUDE_relay_race_sequences_l2624_262469


namespace NUMINAMATH_CALUDE_multiple_is_two_l2624_262401

/-- The multiple of Period 2 students compared to Period 1 students -/
def multiple_of_period2 (period1_students period2_students : ℕ) : ℚ :=
  (period1_students + 5) / period2_students

theorem multiple_is_two :
  let period1_students : ℕ := 11
  let period2_students : ℕ := 8
  multiple_of_period2 period1_students period2_students = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_is_two_l2624_262401


namespace NUMINAMATH_CALUDE_triangle_side_length_l2624_262444

theorem triangle_side_length (a b c : ℝ) (A : Real) :
  A = π / 3 →  -- Angle A = 60°
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →  -- Area of triangle = √3
  b + c = 6 →  -- Given condition
  a = 2 * Real.sqrt 6 := by  -- Prove that a = 2√6
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2624_262444


namespace NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l2624_262435

theorem binomial_coeff_not_coprime (n m k : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) :
  ∃ d : ℕ, d > 1 ∧ d ∣ Nat.choose n k ∧ d ∣ Nat.choose n m :=
by sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l2624_262435


namespace NUMINAMATH_CALUDE_fraction_most_compliant_l2624_262410

/-- Represents the compliance of an algebraic expression with standard notation -/
inductive AlgebraicCompliance
  | Compliant
  | NonCompliant

/-- Evaluates the compliance of a mixed number with variable expression -/
def mixedNumberWithVariable (n : ℕ) (m : ℕ) (d : ℕ) (x : String) : AlgebraicCompliance :=
  AlgebraicCompliance.NonCompliant

/-- Evaluates the compliance of a fraction expression -/
def fraction (n : String) (d : String) : AlgebraicCompliance :=
  AlgebraicCompliance.Compliant

/-- Evaluates the compliance of an expression with an attached unit -/
def expressionWithUnit (expr : String) (unit : String) : AlgebraicCompliance :=
  AlgebraicCompliance.NonCompliant

/-- Evaluates the compliance of a multiplication expression -/
def multiplicationExpression (x : String) (n : ℕ) : AlgebraicCompliance :=
  AlgebraicCompliance.NonCompliant

/-- Theorem stating that fraction (b/a) is the most compliant with standard algebraic notation -/
theorem fraction_most_compliant :
  fraction "b" "a" = AlgebraicCompliance.Compliant ∧
  mixedNumberWithVariable 1 1 2 "a" = AlgebraicCompliance.NonCompliant ∧
  expressionWithUnit "3a-1" "个" = AlgebraicCompliance.NonCompliant ∧
  multiplicationExpression "a" 3 = AlgebraicCompliance.NonCompliant :=
by sorry

end NUMINAMATH_CALUDE_fraction_most_compliant_l2624_262410


namespace NUMINAMATH_CALUDE_green_peaches_count_l2624_262443

/-- The number of baskets -/
def num_baskets : ℕ := 7

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 2

/-- The total number of green peaches -/
def total_green_peaches : ℕ := num_baskets * green_peaches_per_basket

theorem green_peaches_count : total_green_peaches = 14 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2624_262443


namespace NUMINAMATH_CALUDE_sphere_volume_in_cube_l2624_262432

/-- The volume of a sphere inscribed in a cube with surface area 6 cm² is (1/6)π cm³ -/
theorem sphere_volume_in_cube (cube_surface_area : ℝ) (sphere_volume : ℝ) :
  cube_surface_area = 6 →
  sphere_volume = (1 / 6) * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_in_cube_l2624_262432


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l2624_262488

/-- The range of k values for which the line y = kx + 1 intersects the right branch of the hyperbola 3x^2 - y^2 = 3 at two distinct points -/
theorem line_hyperbola_intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
    y₁ = k * x₁ + 1 ∧ y₂ = k * x₂ + 1 ∧
    3 * x₁^2 - y₁^2 = 3 ∧ 3 * x₂^2 - y₂^2 = 3) ↔
  -2 < k ∧ k < -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l2624_262488


namespace NUMINAMATH_CALUDE_tank_water_level_l2624_262491

theorem tank_water_level (tank_capacity : ℝ) (initial_level : ℝ) 
  (empty_percentage : ℝ) (fill_percentage : ℝ) (final_volume : ℝ) :
  tank_capacity = 8000 →
  empty_percentage = 0.4 →
  fill_percentage = 0.3 →
  final_volume = 4680 →
  final_volume = initial_level * (1 - empty_percentage) * (1 + fill_percentage) →
  initial_level / tank_capacity = 0.75 := by
sorry

end NUMINAMATH_CALUDE_tank_water_level_l2624_262491


namespace NUMINAMATH_CALUDE_new_year_after_10_years_l2624_262489

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year in the 21st century -/
structure Year21stCentury where
  year : Nat
  is_21st_century : 2001 ≤ year ∧ year ≤ 2100

/-- Function to determine if a year is a leap year -/
def isLeapYear (y : Year21stCentury) : Bool :=
  y.year % 4 = 0 && (y.year % 100 ≠ 0 || y.year % 400 = 0)

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Theorem stating that New Year's Day 10 years after a Friday is a Thursday -/
theorem new_year_after_10_years 
  (start_year : Year21stCentury)
  (h1 : DayOfWeek.Friday = advanceDays DayOfWeek.Friday 0)  -- New Year's Day is Friday in start_year
  (h2 : ∀ d : DayOfWeek, (advanceDays d (5 * 365 + 2)) = d) -- All days occur equally often in 5 years
  : DayOfWeek.Thursday = advanceDays DayOfWeek.Friday (10 * 365 + 3) :=
by sorry


end NUMINAMATH_CALUDE_new_year_after_10_years_l2624_262489


namespace NUMINAMATH_CALUDE_days_without_calls_is_250_l2624_262431

/-- Represents the frequency of calls from each grandchild -/
def call_frequency₁ : ℕ := 5
def call_frequency₂ : ℕ := 7

/-- Represents the number of days in the year -/
def days_in_year : ℕ := 365

/-- Calculates the number of days without calls -/
def days_without_calls : ℕ :=
  days_in_year - (days_in_year / call_frequency₁ + days_in_year / call_frequency₂ - days_in_year / (call_frequency₁ * call_frequency₂))

/-- Theorem stating that there are 250 days without calls -/
theorem days_without_calls_is_250 : days_without_calls = 250 := by
  sorry

end NUMINAMATH_CALUDE_days_without_calls_is_250_l2624_262431


namespace NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l2624_262418

/-- The perimeter of a rectangular garden with length 100 m and breadth 200 m is 600 m. -/
theorem garden_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun length breadth perimeter =>
    length = 100 ∧ 
    breadth = 200 ∧ 
    perimeter = 2 * (length + breadth) →
    perimeter = 600

-- The proof is omitted
theorem garden_perimeter_proof : garden_perimeter 100 200 600 := by sorry

end NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l2624_262418


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2624_262427

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h_seq : ArithmeticSequence a) 
  (h_prod : a 7 * a 11 = 6) 
  (h_sum : a 4 + a 14 = 5) : 
  ∃ d : ℚ, (d = 1/4 ∨ d = -1/4) ∧ ∀ n : ℕ, a (n + 1) = a n + d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2624_262427


namespace NUMINAMATH_CALUDE_lcm_36_90_l2624_262449

theorem lcm_36_90 : Nat.lcm 36 90 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_90_l2624_262449


namespace NUMINAMATH_CALUDE_square_sum_greater_than_quarter_l2624_262409

theorem square_sum_greater_than_quarter (a b : ℝ) (h : a + b = 1) :
  a^2 + b^2 > 1/4 := by
sorry

end NUMINAMATH_CALUDE_square_sum_greater_than_quarter_l2624_262409


namespace NUMINAMATH_CALUDE_polynomial_real_root_iff_b_in_range_l2624_262492

/-- A polynomial of the form x^4 + bx^3 + x^2 + bx + 1 -/
def polynomial (b : ℝ) (x : ℝ) : ℝ :=
  x^4 + b*x^3 + x^2 + b*x + 1

/-- The polynomial has at least one real root -/
def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, polynomial b x = 0

/-- Theorem: The polynomial has at least one real root if and only if b is in [-3/4, 0) -/
theorem polynomial_real_root_iff_b_in_range :
  ∀ b : ℝ, has_real_root b ↔ -3/4 ≤ b ∧ b < 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_iff_b_in_range_l2624_262492


namespace NUMINAMATH_CALUDE_fifteen_power_equals_R_S_power_l2624_262424

theorem fifteen_power_equals_R_S_power (a b : ℤ) (R S : ℝ) 
  (hR : R = 3^a) (hS : S = 5^b) : 15^(a*b) = R^b * S^a := by
  sorry

end NUMINAMATH_CALUDE_fifteen_power_equals_R_S_power_l2624_262424


namespace NUMINAMATH_CALUDE_min_queries_for_parity_l2624_262415

/-- Represents a query about the parity of balls in 15 bags -/
def Query := Fin 100 → Bool

/-- Represents the state of all bags -/
def BagState := Fin 100 → Bool

/-- The result of a query given a bag state -/
def queryResult (q : Query) (s : BagState) : Bool :=
  (List.filter (fun i => q i) (List.range 100)).foldl (fun acc i => acc ≠ s i) false

/-- A set of queries is sufficient if it can determine the parity of bag 1 -/
def isSufficient (qs : List Query) : Prop :=
  ∀ s1 s2 : BagState, (∀ q ∈ qs, queryResult q s1 = queryResult q s2) → s1 0 = s2 0

theorem min_queries_for_parity : 
  (∃ qs : List Query, qs.length = 3 ∧ isSufficient qs) ∧
  (∀ qs : List Query, qs.length < 3 → ¬isSufficient qs) := by
  sorry

end NUMINAMATH_CALUDE_min_queries_for_parity_l2624_262415


namespace NUMINAMATH_CALUDE_product_pricing_l2624_262493

/-- Given three products A, B, and C with unknown prices, prove that if 2A + 3B + 1C costs 295 yuan
    and 4A + 3B + 5C costs 425 yuan, then 1A + 1B + 1C costs 120 yuan. -/
theorem product_pricing (a b c : ℝ) 
    (h1 : 2*a + 3*b + c = 295)
    (h2 : 4*a + 3*b + 5*c = 425) : 
  a + b + c = 120 := by
sorry

end NUMINAMATH_CALUDE_product_pricing_l2624_262493


namespace NUMINAMATH_CALUDE_smallest_common_divisor_l2624_262471

theorem smallest_common_divisor (n : ℕ) (h1 : n = 627) :
  let m := n + 3
  let k := Nat.minFac (Nat.gcd m (Nat.gcd 4590 105))
  k = 105 := by sorry

end NUMINAMATH_CALUDE_smallest_common_divisor_l2624_262471


namespace NUMINAMATH_CALUDE_volleyball_tickets_l2624_262498

def initial_tickets (jude_tickets andrea_tickets sandra_tickets tickets_left : ℕ) : Prop :=
  andrea_tickets = 2 * jude_tickets ∧
  sandra_tickets = jude_tickets / 2 + 4 ∧
  jude_tickets = 16 ∧
  tickets_left = 40 ∧
  jude_tickets + andrea_tickets + sandra_tickets + tickets_left = 100

theorem volleyball_tickets :
  ∃ (jude_tickets andrea_tickets sandra_tickets tickets_left : ℕ),
    initial_tickets jude_tickets andrea_tickets sandra_tickets tickets_left :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_tickets_l2624_262498


namespace NUMINAMATH_CALUDE_complex_power_result_l2624_262464

theorem complex_power_result : 
  (3 * (Complex.cos (Real.pi / 6) + Complex.I * Complex.sin (Real.pi / 6)))^8 = 
  Complex.mk (-3280.5) (-3280.5 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_complex_power_result_l2624_262464


namespace NUMINAMATH_CALUDE_concatenation_product_sum_l2624_262496

theorem concatenation_product_sum : ∃! (n m : ℕ), 
  (10 ≤ n ∧ n < 100) ∧ 
  (100 ≤ m ∧ m < 1000) ∧ 
  (1000 * n + m = 9 * n * m) ∧ 
  (n + m = 126) := by
  sorry

end NUMINAMATH_CALUDE_concatenation_product_sum_l2624_262496


namespace NUMINAMATH_CALUDE_johns_purchase_price_l2624_262447

/-- Calculate the final price after rebate and tax -/
def finalPrice (originalPrice rebatePercent taxPercent : ℚ) : ℚ :=
  let priceAfterRebate := originalPrice * (1 - rebatePercent / 100)
  let salesTax := priceAfterRebate * (taxPercent / 100)
  priceAfterRebate + salesTax

/-- Theorem stating the final price for John's purchase -/
theorem johns_purchase_price :
  finalPrice 6650 6 10 = 6876.1 :=
sorry

end NUMINAMATH_CALUDE_johns_purchase_price_l2624_262447


namespace NUMINAMATH_CALUDE_nickel_count_l2624_262434

/-- Proves that given $4 in quarters, dimes, and nickels, with 10 quarters and 12 dimes, the number of nickels is 6. -/
theorem nickel_count (total : ℚ) (quarters dimes : ℕ) : 
  total = 4 → 
  quarters = 10 → 
  dimes = 12 → 
  ∃ (nickels : ℕ), 
    total = (0.25 * quarters + 0.1 * dimes + 0.05 * nickels) ∧ 
    nickels = 6 := by sorry

end NUMINAMATH_CALUDE_nickel_count_l2624_262434


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2624_262421

theorem initial_money_calculation (clothes_percent grocery_percent electronics_percent dining_percent : ℚ)
  (remaining_money : ℚ) :
  clothes_percent = 20 / 100 →
  grocery_percent = 15 / 100 →
  electronics_percent = 10 / 100 →
  dining_percent = 5 / 100 →
  remaining_money = 15700 →
  ∃ initial_money : ℚ, 
    initial_money * (1 - (clothes_percent + grocery_percent + electronics_percent + dining_percent)) = remaining_money ∧
    initial_money = 31400 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2624_262421


namespace NUMINAMATH_CALUDE_race_probability_l2624_262454

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℚ) : 
  total_cars = 16 →
  prob_Y = 1/12 →
  prob_Z = 1/7 →
  prob_XYZ = 47619047619047616/100000000000000000 →
  ∃ (prob_X : ℚ), 
    prob_X + prob_Y + prob_Z = prob_XYZ ∧
    prob_X = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_race_probability_l2624_262454


namespace NUMINAMATH_CALUDE_average_weight_a_b_l2624_262461

/-- Given three weights a, b, and c, proves that the average of a and b is 40,
    under certain conditions. -/
theorem average_weight_a_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 31 →
  (a + b) / 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_weight_a_b_l2624_262461


namespace NUMINAMATH_CALUDE_max_value_abc_l2624_262406

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  a^2 * b^3 * c ≤ 27/16 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_l2624_262406


namespace NUMINAMATH_CALUDE_polynomial_value_at_five_l2624_262478

theorem polynomial_value_at_five : 
  let x : ℤ := 5
  x^5 - 3*x^3 - 5*x = 2725 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_five_l2624_262478


namespace NUMINAMATH_CALUDE_matrix_power_eigen_l2624_262404

theorem matrix_power_eigen (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B.vecMul (![3, -1]) = ![12, -4] →
  (B^4).vecMul (![3, -1]) = ![768, -256] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_eigen_l2624_262404


namespace NUMINAMATH_CALUDE_infinite_sum_of_squares_with_neighbors_l2624_262412

theorem infinite_sum_of_squares_with_neighbors (k : ℕ) :
  ∃ n : ℕ,
    (∃ a b : ℕ, n = a^2 + b^2) ∧
    (∀ x y : ℕ, (n - 1) ≠ x^2 + y^2) ∧
    (∀ x y : ℕ, (n + 1) ≠ x^2 + y^2) :=
by sorry

end NUMINAMATH_CALUDE_infinite_sum_of_squares_with_neighbors_l2624_262412


namespace NUMINAMATH_CALUDE_exists_proportion_with_means_less_than_extremes_l2624_262482

/-- A proportion is represented by four real numbers a, b, c, d such that a : b = c : d -/
def IsProportion (a b c d : ℝ) : Prop := a * d = b * c

/-- Theorem: There exists a proportion where both means are less than both extremes -/
theorem exists_proportion_with_means_less_than_extremes :
  ∃ (a b c d : ℝ), IsProportion a b c d ∧ b < a ∧ b < d ∧ c < a ∧ c < d := by
  sorry

end NUMINAMATH_CALUDE_exists_proportion_with_means_less_than_extremes_l2624_262482


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2624_262456

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3*a) → 
  a ∈ Set.Icc (-2 : ℝ) 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2624_262456


namespace NUMINAMATH_CALUDE_path_length_2x1x1_block_l2624_262480

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the path traced by a dot on the block -/
def path_length (b : Block) : ℝ := sorry

/-- Theorem stating that the path length for a 2×1×1 block is 4π -/
theorem path_length_2x1x1_block :
  let b : Block := ⟨2, 1, 1⟩
  path_length b = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_path_length_2x1x1_block_l2624_262480


namespace NUMINAMATH_CALUDE_simplify_expression_l2624_262446

theorem simplify_expression : (5^7 + 3^6) * (1^5 - (-1)^4)^10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2624_262446


namespace NUMINAMATH_CALUDE_square_of_1023_l2624_262497

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1023_l2624_262497


namespace NUMINAMATH_CALUDE_larger_sphere_radius_l2624_262411

/-- The radius of a sphere with volume equal to 12 spheres of radius 0.5 inches is ³√3 inches. -/
theorem larger_sphere_radius (r : ℝ) : 
  (4 / 3 * Real.pi * r^3 = 12 * (4 / 3 * Real.pi * (1 / 2)^3)) → r = (3 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_larger_sphere_radius_l2624_262411


namespace NUMINAMATH_CALUDE_banana_change_calculation_emily_banana_change_l2624_262487

/-- Calculates the change received when buying bananas with a discount --/
theorem banana_change_calculation (num_bananas : ℕ) (cost_per_banana : ℚ) 
  (discount_threshold : ℕ) (discount_rate : ℚ) (paid_amount : ℚ) : ℚ :=
  let total_cost := num_bananas * cost_per_banana
  let discounted_cost := if num_bananas > discount_threshold 
    then total_cost * (1 - discount_rate) 
    else total_cost
  paid_amount - discounted_cost

/-- Proves that Emily received $8.65 in change --/
theorem emily_banana_change : 
  banana_change_calculation 5 (30/100) 4 (10/100) 10 = 865/100 := by
  sorry

end NUMINAMATH_CALUDE_banana_change_calculation_emily_banana_change_l2624_262487


namespace NUMINAMATH_CALUDE_eccentricity_conic_sections_l2624_262428

theorem eccentricity_conic_sections : ∃ (e₁ e₂ : ℝ), 
  e₁^2 - 5*e₁ + 1 = 0 ∧ 
  e₂^2 - 5*e₂ + 1 = 0 ∧ 
  (0 < e₁ ∧ e₁ < 1) ∧ 
  (e₂ > 1) := by sorry

end NUMINAMATH_CALUDE_eccentricity_conic_sections_l2624_262428


namespace NUMINAMATH_CALUDE_total_frisbees_sold_l2624_262466

/-- Represents the number of frisbees sold at $3 -/
def x : ℕ := sorry

/-- Represents the number of frisbees sold at $4 -/
def y : ℕ := sorry

/-- The total receipts from frisbee sales is $200 -/
axiom total_sales : 3 * x + 4 * y = 200

/-- The fewest number of $4 frisbees sold is 8 -/
axiom min_four_dollar_frisbees : y ≥ 8

/-- The total number of frisbees sold -/
def total_frisbees : ℕ := x + y

theorem total_frisbees_sold : total_frisbees = 64 := by sorry

end NUMINAMATH_CALUDE_total_frisbees_sold_l2624_262466


namespace NUMINAMATH_CALUDE_hockey_league_games_l2624_262433

/-- Represents the number of games played between two groups of teams -/
def games_between (n m : ℕ) (games_per_pair : ℕ) : ℕ := n * m * games_per_pair

/-- Represents the number of games played within a group of teams -/
def games_within (n : ℕ) (games_per_pair : ℕ) : ℕ := n * (n - 1) * games_per_pair / 2

/-- The total number of games played in the hockey league season -/
def total_games : ℕ :=
  let top5 := 5
  let mid5 := 5
  let bottom5 := 5
  let top_vs_top := games_within top5 12
  let top_vs_rest := games_between top5 (mid5 + bottom5) 8
  let mid_vs_mid := games_within mid5 10
  let mid_vs_bottom := games_between mid5 bottom5 6
  let bottom_vs_bottom := games_within bottom5 8
  top_vs_top + top_vs_rest + mid_vs_mid + mid_vs_bottom + bottom_vs_bottom

theorem hockey_league_games :
  total_games = 850 := by sorry

end NUMINAMATH_CALUDE_hockey_league_games_l2624_262433


namespace NUMINAMATH_CALUDE_a_closed_form_l2624_262441

def a : ℕ → ℤ
  | 0 => -1
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + 3 * a n + 3^(n + 2)

theorem a_closed_form (n : ℕ) :
  a n = (1 / 16) * ((4 * n - 3) * 3^(n + 1) - 7 * (-1)^n) :=
by sorry

end NUMINAMATH_CALUDE_a_closed_form_l2624_262441


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2624_262465

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^5 - x^4 + 2 * x^3 + 5 * x^2 - 3 * x + 7) + 
  (-x^5 + 4 * x^4 + x^3 - 6 * x^2 + 5 * x - 4) - 
  (2 * x^5 - 3 * x^4 + 4 * x^3 - x^2 - x + 2) = 
  6 * x^4 - x^3 + 3 * x + 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2624_262465


namespace NUMINAMATH_CALUDE_diag_diff_octagon_heptagon_l2624_262450

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Number of diagonals in a heptagon -/
def A : ℕ := num_diagonals 7

/-- Number of diagonals in an octagon -/
def B : ℕ := num_diagonals 8

/-- The difference between the number of diagonals in an octagon and a heptagon is 6 -/
theorem diag_diff_octagon_heptagon : B - A = 6 := by sorry

end NUMINAMATH_CALUDE_diag_diff_octagon_heptagon_l2624_262450


namespace NUMINAMATH_CALUDE_cubic_repeated_root_l2624_262452

/-- The cubic equation has a repeated root iff p = 5 or p = -7 -/
theorem cubic_repeated_root (p : ℝ) : 
  (∃ x : ℝ, (3 * x^3 - (p + 1) * x^2 + 4 * x - 12 = 0) ∧ 
   (6 * x^2 - 2 * (p + 1) * x + 4 = 0)) ↔ 
  (p = 5 ∨ p = -7) :=
sorry

end NUMINAMATH_CALUDE_cubic_repeated_root_l2624_262452


namespace NUMINAMATH_CALUDE_cricketer_matches_count_l2624_262474

/-- Proves that a cricketer played 10 matches given the average scores for all matches, 
    the first 6 matches, and the last 4 matches. -/
theorem cricketer_matches_count 
  (total_average : ℝ) 
  (first_six_average : ℝ) 
  (last_four_average : ℝ) 
  (h1 : total_average = 38.9)
  (h2 : first_six_average = 42)
  (h3 : last_four_average = 34.25) : 
  ∃ (n : ℕ), n = 10 ∧ 
    n * total_average = 6 * first_six_average + 4 * last_four_average := by
  sorry

#check cricketer_matches_count

end NUMINAMATH_CALUDE_cricketer_matches_count_l2624_262474


namespace NUMINAMATH_CALUDE_max_female_students_with_four_teachers_min_group_size_exists_min_group_l2624_262457

/-- Represents the composition of a study group --/
structure StudyGroup where
  male_students : ℕ
  female_students : ℕ
  teachers : ℕ

/-- Checks if a study group satisfies the given conditions --/
def is_valid_group (g : StudyGroup) : Prop :=
  g.male_students > g.female_students ∧
  g.female_students > g.teachers ∧
  2 * g.teachers > g.male_students

theorem max_female_students_with_four_teachers :
  ∀ g : StudyGroup,
  is_valid_group g → g.teachers = 4 →
  g.female_students ≤ 6 :=
sorry

theorem min_group_size :
  ∀ g : StudyGroup,
  is_valid_group g →
  g.male_students + g.female_students + g.teachers ≥ 12 :=
sorry

theorem exists_min_group :
  ∃ g : StudyGroup,
  is_valid_group g ∧
  g.male_students + g.female_students + g.teachers = 12 :=
sorry

end NUMINAMATH_CALUDE_max_female_students_with_four_teachers_min_group_size_exists_min_group_l2624_262457


namespace NUMINAMATH_CALUDE_frank_bought_five_chocolates_l2624_262422

-- Define the cost of items
def chocolate_cost : ℕ := 2
def chips_cost : ℕ := 3

-- Define the number of bags of chips
def chips_count : ℕ := 2

-- Define the total amount spent
def total_spent : ℕ := 16

-- Define the function to calculate the number of chocolate bars
def chocolate_bars : ℕ → Prop
  | n => chocolate_cost * n + chips_cost * chips_count = total_spent

-- Theorem statement
theorem frank_bought_five_chocolates : 
  ∃ (n : ℕ), chocolate_bars n ∧ n = 5 := by sorry

end NUMINAMATH_CALUDE_frank_bought_five_chocolates_l2624_262422


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2624_262486

/-- Given plane vectors a and b, if a is parallel to 2b - a, then m = 9/2 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
    (h1 : a = (4, 3))
    (h2 : b = (6, m))
    (h3 : ∃ (k : ℝ), a = k • (2 • b - a)) :
  m = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2624_262486


namespace NUMINAMATH_CALUDE_fish_remaining_l2624_262462

theorem fish_remaining (guppies angelfish tiger_sharks oscar_fish : ℕ)
  (guppies_sold angelfish_sold tiger_sharks_sold oscar_fish_sold : ℕ)
  (h1 : guppies = 94)
  (h2 : angelfish = 76)
  (h3 : tiger_sharks = 89)
  (h4 : oscar_fish = 58)
  (h5 : guppies_sold = 30)
  (h6 : angelfish_sold = 48)
  (h7 : tiger_sharks_sold = 17)
  (h8 : oscar_fish_sold = 24) :
  (guppies - guppies_sold) + (angelfish - angelfish_sold) +
  (tiger_sharks - tiger_sharks_sold) + (oscar_fish - oscar_fish_sold) = 198 :=
by sorry

end NUMINAMATH_CALUDE_fish_remaining_l2624_262462


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2624_262476

-- Define a coloring function type
def ColoringFunction := ℝ × ℝ → Bool

-- Define a property for a valid coloring
def ValidColoring (f : ColoringFunction) : Prop :=
  ∀ A B : ℝ × ℝ, A ≠ B →
    ∃ t : ℝ, 0 < t ∧ t < 1 ∧
      let C := (1 - t) • A + t • B
      f C ≠ f A ∨ f C ≠ f B

-- Theorem statement
theorem exists_valid_coloring : ∃ f : ColoringFunction, ValidColoring f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2624_262476


namespace NUMINAMATH_CALUDE_range_of_a_l2624_262445

-- Define the function f(x) for any real a
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 + a) * x^2 - a * x + 1

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x > 0) ↔ ((-4/3 < a ∧ a < -1) ∨ a = 0) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2624_262445


namespace NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l2624_262416

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) :
  a + b > 0 → a > 0 ∨ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l2624_262416


namespace NUMINAMATH_CALUDE_evaluate_expression_l2624_262451

theorem evaluate_expression (b c : ℕ) (hb : b = 2) (hc : c = 5) :
  b^3 * b^4 * c^2 = 3200 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2624_262451


namespace NUMINAMATH_CALUDE_area_between_circle_and_square_l2624_262407

/-- Given a square with side length 2 and a circle with radius √2 sharing the same center,
    the area inside the circle but outside the square is equal to 2π - 4. -/
theorem area_between_circle_and_square :
  let square_side : ℝ := 2
  let circle_radius : ℝ := Real.sqrt 2
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  circle_area - square_area = 2 * π - 4 := by
  sorry

end NUMINAMATH_CALUDE_area_between_circle_and_square_l2624_262407


namespace NUMINAMATH_CALUDE_draw_one_is_random_event_l2624_262472

/-- A set of cards numbered from 1 to 10 -/
def CardSet : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}

/-- Definition of a random event -/
def IsRandomEvent (event : Set ℕ → Prop) : Prop :=
  ∃ (s : Set ℕ), event s ∧ ∃ (t : Set ℕ), ¬event t

/-- Drawing a card numbered 1 from the set -/
def DrawOne (s : Set ℕ) : Prop := 1 ∈ s

/-- Theorem: Drawing a card numbered 1 from a set of cards numbered 1 to 10 is a random event -/
theorem draw_one_is_random_event : IsRandomEvent DrawOne :=
sorry

end NUMINAMATH_CALUDE_draw_one_is_random_event_l2624_262472


namespace NUMINAMATH_CALUDE_max_sum_of_first_two_l2624_262420

theorem max_sum_of_first_two (a b c d e : ℕ) : 
  a < b → b < c → c < d → d < e → 
  a + 2*b + 3*c + 4*d + 5*e = 300 → 
  a + b ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_first_two_l2624_262420


namespace NUMINAMATH_CALUDE_toy_cost_price_l2624_262495

/-- The cost price of a toy, given the selling conditions -/
def cost_price (selling_price : ℕ) (num_sold : ℕ) (gain_equiv : ℕ) : ℚ :=
  selling_price / (num_sold + gain_equiv)

theorem toy_cost_price :
  let selling_price : ℕ := 25200
  let num_sold : ℕ := 18
  let gain_equiv : ℕ := 3
  cost_price selling_price num_sold gain_equiv = 1200 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_price_l2624_262495


namespace NUMINAMATH_CALUDE_sam_seashells_l2624_262473

/-- The number of seashells Sam found on the beach -/
def total_seashells : ℕ := 35

/-- The number of seashells Sam gave to Joan -/
def seashells_given : ℕ := 18

/-- The number of seashells Sam has now -/
def seashells_remaining : ℕ := 17

/-- Theorem stating that the total number of seashells Sam found is equal to
    the sum of seashells given away and seashells remaining -/
theorem sam_seashells : 
  total_seashells = seashells_given + seashells_remaining := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l2624_262473


namespace NUMINAMATH_CALUDE_bell_peppers_needed_l2624_262439

/-- Represents the number of slices and pieces obtained from one bell pepper -/
def slices_per_pepper : ℕ := 20

/-- Represents the fraction of large slices that are cut into smaller pieces -/
def fraction_cut : ℚ := 1/2

/-- Represents the number of smaller pieces each large slice is cut into -/
def pieces_per_slice : ℕ := 3

/-- Represents the total number of slices and pieces Tamia wants to use -/
def total_slices : ℕ := 200

/-- Proves that 5 bell peppers are needed to produce 200 slices and pieces -/
theorem bell_peppers_needed : 
  (total_slices : ℚ) / ((1 - fraction_cut) * slices_per_pepper + 
  fraction_cut * slices_per_pepper * pieces_per_slice) = 5 := by
sorry

end NUMINAMATH_CALUDE_bell_peppers_needed_l2624_262439


namespace NUMINAMATH_CALUDE_cistern_length_l2624_262475

/-- Given a cistern with specified dimensions, calculate its length -/
theorem cistern_length (width : Real) (water_depth : Real) (wet_area : Real)
  (h1 : width = 4)
  (h2 : water_depth = 1.25)
  (h3 : wet_area = 49)
  : ∃ (length : Real), length = wet_area / (width + 2 * water_depth) :=
by
  sorry

#check cistern_length

end NUMINAMATH_CALUDE_cistern_length_l2624_262475


namespace NUMINAMATH_CALUDE_thirtieth_term_is_119_l2624_262414

/-- An arithmetic sequence is defined by its first term and common difference -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ := fun n => a₁ + (n - 1) * d

/-- The first term of our sequence -/
def a₁ : ℝ := 3

/-- The second term of our sequence -/
def a₂ : ℝ := 7

/-- The third term of our sequence -/
def a₃ : ℝ := 11

/-- The common difference of our sequence -/
def d : ℝ := a₂ - a₁

/-- The 30th term of our sequence -/
def a₃₀ : ℝ := arithmeticSequence a₁ d 30

theorem thirtieth_term_is_119 : a₃₀ = 119 := by sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_119_l2624_262414


namespace NUMINAMATH_CALUDE_composition_value_l2624_262430

-- Define the functions h and j
def h (x : ℝ) : ℝ := 4 * x + 5
def j (x : ℝ) : ℝ := 6 * x - 11

-- State the theorem
theorem composition_value : j (h 5) = 139 := by sorry

end NUMINAMATH_CALUDE_composition_value_l2624_262430


namespace NUMINAMATH_CALUDE_last_digit_2016_octal_l2624_262481

def decimal_to_octal_last_digit (n : ℕ) : ℕ :=
  n % 8

theorem last_digit_2016_octal : decimal_to_octal_last_digit 2016 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_2016_octal_l2624_262481


namespace NUMINAMATH_CALUDE_trigonometric_problem_l2624_262499

theorem trigonometric_problem (α : ℝ) 
  (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1/4) :
  (Real.tan α = -2) ∧ 
  ((Real.sin (2*α) + 1) / (1 + Real.sin (2*α) + Real.cos (2*α)) = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l2624_262499


namespace NUMINAMATH_CALUDE_no_solution_sqrt_plus_one_l2624_262448

theorem no_solution_sqrt_plus_one :
  ∀ x : ℝ, ¬(Real.sqrt (x + 4) + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_sqrt_plus_one_l2624_262448


namespace NUMINAMATH_CALUDE_c_profit_share_l2624_262400

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_profit_share (investment : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * total_profit) / total_investment

theorem c_profit_share :
  let a_investment := 12000
  let b_investment := 16000
  let c_investment := 20000
  let total_investment := a_investment + b_investment + c_investment
  let total_profit := 86400
  calculate_profit_share c_investment total_investment total_profit = 36000 := by
sorry

#eval calculate_profit_share 20000 (12000 + 16000 + 20000) 86400

end NUMINAMATH_CALUDE_c_profit_share_l2624_262400


namespace NUMINAMATH_CALUDE_present_age_of_R_l2624_262429

-- Define the present ages of P, Q, and R
variable (Pp Qp Rp : ℝ)

-- Define the conditions
def condition1 : Prop := Pp - 8 = (1/2) * (Qp - 8)
def condition2 : Prop := Qp - 8 = (2/3) * (Rp - 8)
def condition3 : Prop := Qp = 2 * Real.sqrt Rp
def condition4 : Prop := Pp / Qp = 3/5

-- Theorem statement
theorem present_age_of_R 
  (h1 : condition1 Pp Qp)
  (h2 : condition2 Qp Rp)
  (h3 : condition3 Qp Rp)
  (h4 : condition4 Pp Qp) :
  Rp = 400 := by
  sorry

end NUMINAMATH_CALUDE_present_age_of_R_l2624_262429


namespace NUMINAMATH_CALUDE_father_age_problem_l2624_262485

theorem father_age_problem (father_age son_age : ℕ) : 
  (father_age = 4 * son_age + 4) →
  (father_age + 4 = 2 * (son_age + 4) + 20) →
  father_age = 44 := by
  sorry

end NUMINAMATH_CALUDE_father_age_problem_l2624_262485


namespace NUMINAMATH_CALUDE_student_average_age_l2624_262419

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (avg_increase : ℝ) : 
  num_students = 22 → 
  teacher_age = 44 → 
  avg_increase = 1 → 
  (((num_students : ℝ) * x + teacher_age) / (num_students + 1) = x + avg_increase) → 
  x = 21 :=
by sorry

end NUMINAMATH_CALUDE_student_average_age_l2624_262419


namespace NUMINAMATH_CALUDE_situps_total_l2624_262426

/-- The number of sit-ups Barney can perform in one minute -/
def barney_situps : ℕ := 45

/-- The number of minutes Barney performs sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The number of sit-ups Carrie can perform in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can perform in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := barney_situps * barney_minutes + 
                        carrie_situps * carrie_minutes + 
                        jerrie_situps * jerrie_minutes

theorem situps_total : total_situps = 510 := by
  sorry

end NUMINAMATH_CALUDE_situps_total_l2624_262426


namespace NUMINAMATH_CALUDE_part_one_part_two_l2624_262463

-- Define the propositions p and q
def p (t a : ℝ) : Prop := t^2 - 5*a*t + 4*a^2 < 0

def q (t : ℝ) : Prop := ∃ (x y : ℝ), x^2/(t-2) + y^2/(t-6) = 1 ∧ (t-2)*(t-6) < 0

-- Part I
theorem part_one (t : ℝ) : p t 1 ∧ q t → 2 < t ∧ t < 4 := by sorry

-- Part II
theorem part_two (a : ℝ) : (∀ t : ℝ, q t → p t a) → 3/2 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2624_262463
