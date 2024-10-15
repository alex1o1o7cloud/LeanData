import Mathlib

namespace NUMINAMATH_CALUDE_invertible_function_fixed_point_l2870_287030

/-- Given an invertible function f: ℝ → ℝ, if f(a) = 3 and f(3) = a, then a - 3 = 0 -/
theorem invertible_function_fixed_point 
  (f : ℝ → ℝ) (hf : Function.Bijective f) (a : ℝ) 
  (h1 : f a = 3) (h2 : f 3 = a) : a - 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_invertible_function_fixed_point_l2870_287030


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l2870_287066

theorem imaginary_part_of_complex_product : Complex.im ((1 + Complex.I)^2 * (2 + Complex.I)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l2870_287066


namespace NUMINAMATH_CALUDE_half_dollar_percentage_l2870_287091

theorem half_dollar_percentage : 
  let nickel_count : ℕ := 80
  let half_dollar_count : ℕ := 40
  let nickel_value : ℕ := 5
  let half_dollar_value : ℕ := 50
  let total_value := nickel_count * nickel_value + half_dollar_count * half_dollar_value
  let half_dollar_total := half_dollar_count * half_dollar_value
  (half_dollar_total : ℚ) / total_value = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_half_dollar_percentage_l2870_287091


namespace NUMINAMATH_CALUDE_sector_max_area_l2870_287037

/-- Given a rope of length 20cm forming a sector, the area of the sector is maximized when the central angle is 2 radians. -/
theorem sector_max_area (r l α : ℝ) : 
  0 < r → r < 10 →
  l + 2 * r = 20 →
  l = r * α →
  ∀ r' l' α', 
    0 < r' → r' < 10 →
    l' + 2 * r' = 20 →
    l' = r' * α' →
    r * l ≥ r' * l' →
  α = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_max_area_l2870_287037


namespace NUMINAMATH_CALUDE_steve_answerable_questions_l2870_287095

theorem steve_answerable_questions (total_questions : ℕ) (difference : ℕ) : 
  total_questions = 45 → difference = 7 → total_questions - difference = 38 := by
sorry

end NUMINAMATH_CALUDE_steve_answerable_questions_l2870_287095


namespace NUMINAMATH_CALUDE_smallest_dual_representation_l2870_287040

/-- Represents a number in a given base -/
def represent_in_base (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- Converts a number from a given base to base 10 -/
def to_base_10 (digits : List ℕ) (base : ℕ) : ℕ := sorry

/-- Checks if a number can be represented as 13 in base c and 31 in base d -/
def is_valid_representation (n : ℕ) (c : ℕ) (d : ℕ) : Prop :=
  (represent_in_base n c = [1, 3]) ∧ (represent_in_base n d = [3, 1])

theorem smallest_dual_representation :
  ∃ (n : ℕ) (c : ℕ) (d : ℕ),
    c > 3 ∧ d > 3 ∧
    is_valid_representation n c d ∧
    (∀ (m : ℕ) (c' : ℕ) (d' : ℕ),
      c' > 3 → d' > 3 → is_valid_representation m c' d' → n ≤ m) ∧
    n = 13 := by sorry

#check smallest_dual_representation

end NUMINAMATH_CALUDE_smallest_dual_representation_l2870_287040


namespace NUMINAMATH_CALUDE_time_before_second_rewind_is_45_l2870_287042

/-- Represents the movie watching scenario with rewinds -/
structure MovieWatching where
  totalTime : ℕ
  initialWatchTime : ℕ
  firstRewindTime : ℕ
  secondRewindTime : ℕ
  finalWatchTime : ℕ

/-- Calculates the time watched before the second rewind -/
def timeBeforeSecondRewind (m : MovieWatching) : ℕ :=
  m.totalTime - (m.initialWatchTime + m.firstRewindTime + m.secondRewindTime + m.finalWatchTime)

/-- Theorem stating the time watched before the second rewind is 45 minutes -/
theorem time_before_second_rewind_is_45 (m : MovieWatching)
    (h1 : m.totalTime = 120)
    (h2 : m.initialWatchTime = 35)
    (h3 : m.firstRewindTime = 5)
    (h4 : m.secondRewindTime = 15)
    (h5 : m.finalWatchTime = 20) :
    timeBeforeSecondRewind m = 45 := by
  sorry

end NUMINAMATH_CALUDE_time_before_second_rewind_is_45_l2870_287042


namespace NUMINAMATH_CALUDE_unique_solution_star_l2870_287025

/-- The star operation defined on real numbers -/
def star (x y : ℝ) : ℝ := 3 * x - 2 * y + x^2 * y

/-- Theorem stating that there's exactly one solution to 2 ⋆ y = 9 -/
theorem unique_solution_star :
  ∃! y : ℝ, star 2 y = 9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_star_l2870_287025


namespace NUMINAMATH_CALUDE_cos_sin_sum_equality_l2870_287043

theorem cos_sin_sum_equality : 
  Real.cos (16 * π / 180) * Real.cos (61 * π / 180) + 
  Real.sin (16 * π / 180) * Real.sin (61 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equality_l2870_287043


namespace NUMINAMATH_CALUDE_min_xy_value_l2870_287007

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 9) : 
  (x * y : ℕ) ≥ 108 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l2870_287007


namespace NUMINAMATH_CALUDE_equation_solutions_l2870_287094

theorem equation_solutions (x : ℚ) :
  (x = 2/9 ∧ 81 * x^2 + 220 = 196 * x - 15) →
  (5/9 : ℚ)^2 * 81 + 220 = 196 * (5/9 : ℚ) - 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2870_287094


namespace NUMINAMATH_CALUDE_total_books_proof_l2870_287035

/-- The total number of books on two bookshelves -/
def total_books : ℕ := 30

/-- The number of books moved from the first shelf to the second shelf -/
def books_moved : ℕ := 5

theorem total_books_proof :
  (∃ (initial_books_per_shelf : ℕ),
    initial_books_per_shelf * 2 = total_books ∧
    (initial_books_per_shelf + books_moved) = 2 * (initial_books_per_shelf - books_moved)) :=
by
  sorry

end NUMINAMATH_CALUDE_total_books_proof_l2870_287035


namespace NUMINAMATH_CALUDE_min_value_theorem_l2870_287049

theorem min_value_theorem (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x + y > z) (hyz : y + z > x) (hzx : z + x > y) :
  (x + y + z) * (1 / (x + y - z) + 1 / (y + z - x) + 1 / (z + x - y)) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2870_287049


namespace NUMINAMATH_CALUDE_monomial_like_terms_sum_l2870_287064

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, ∃ c₁ c₂ : ℚ, a x y = c₁ ∧ b x y = c₂

theorem monomial_like_terms_sum (m n : ℕ) :
  like_terms (fun x y => 5 * x^m * y) (fun x y => -3 * x^2 * y^n) →
  m + n = 3 := by
sorry

end NUMINAMATH_CALUDE_monomial_like_terms_sum_l2870_287064


namespace NUMINAMATH_CALUDE_simultaneous_strike_l2870_287006

def cymbal_interval : ℕ := 7
def triangle_interval : ℕ := 2

theorem simultaneous_strike :
  ∃ (n : ℕ), n > 0 ∧ n % cymbal_interval = 0 ∧ n % triangle_interval = 0 ∧
  ∀ (m : ℕ), 0 < m ∧ m < n → (m % cymbal_interval ≠ 0 ∨ m % triangle_interval ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_strike_l2870_287006


namespace NUMINAMATH_CALUDE_char_coeff_pair_example_char_poly_sum_example_char_poly_diff_example_l2870_287027

-- Define the characteristic coefficient pair
def char_coeff_pair (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

-- Define the characteristic polynomial
def char_poly (p : ℝ × ℝ × ℝ) (x : ℝ) : ℝ :=
  let (a, b, c) := p
  a * x^2 + b * x + c

theorem char_coeff_pair_example : char_coeff_pair 3 4 1 = (3, 4, 1) := by sorry

theorem char_poly_sum_example : 
  char_poly (2, 1, 2) x + char_poly (2, -1, 2) x = 4 * x^2 + 4 := by sorry

theorem char_poly_diff_example (m n : ℝ) : 
  (char_poly (1, 2, m) x - char_poly (2, n, 3) x = -x^2 + x - 1) → m * n = 2 := by sorry

end NUMINAMATH_CALUDE_char_coeff_pair_example_char_poly_sum_example_char_poly_diff_example_l2870_287027


namespace NUMINAMATH_CALUDE_perpendicular_circle_radius_l2870_287083

/-- Given two perpendicular lines and a circle of radius R tangent to these lines,
    the radius of a circle that is tangent to the same lines and intersects
    the given circle at a right angle is R(2 ± √3). -/
theorem perpendicular_circle_radius (R : ℝ) (R_pos : R > 0) :
  ∃ x : ℝ, x > 0 ∧ 
  (x = R * (2 + Real.sqrt 3) ∨ x = R * (2 - Real.sqrt 3)) ∧
  (∃ (C C₁ : ℝ × ℝ),
    (C.1 = R ∧ C.2 = R) ∧  -- Center of the given circle
    (C₁.1 > 0 ∧ C₁.2 > 0) ∧  -- Center of the new circle in the first quadrant
    ((C₁.1 - C.1)^2 + (C₁.2 - C.2)^2 = (x + R)^2) ∧  -- Circles intersect at right angle
    (C₁.1 = x ∧ C₁.2 = x))  -- New circle is tangent to the perpendicular lines
:= by sorry

end NUMINAMATH_CALUDE_perpendicular_circle_radius_l2870_287083


namespace NUMINAMATH_CALUDE_sequence_properties_l2870_287001

def sequence_a (n : ℕ) : ℝ := (n + 1 : ℝ) * 2^(n - 1)

def partial_sum (n : ℕ) : ℝ := n * 2^n - 2^n

theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n > 0 → S n = 2 * a n - 2^n) :
  (∀ n : ℕ, n > 0 → a n / 2^n - a (n-1) / 2^(n-1) = 1/2) ∧
  (∀ n : ℕ, n > 0 → a n = sequence_a n) ∧
  (∀ n : ℕ, n > 0 → S n = partial_sum n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2870_287001


namespace NUMINAMATH_CALUDE_percentage_problem_l2870_287029

theorem percentage_problem : ∃ p : ℚ, p = 55/100 ∧ p * 40 = 4/5 * 25 + 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2870_287029


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2870_287018

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  (2 * c - b) / Real.cos B = a / Real.cos A ∧
  a = Real.sqrt 7 ∧
  2 * b = 3 * c

theorem triangle_ABC_properties {a b c A B C : ℝ} 
  (h : triangle_ABC a b c A B C) :
  A = π / 3 ∧ 
  (1/2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2870_287018


namespace NUMINAMATH_CALUDE_days_to_finish_book_l2870_287061

def total_pages : ℕ := 285
def pages_per_day : ℕ := 23

theorem days_to_finish_book :
  ⌈(total_pages : ℝ) / pages_per_day⌉ = 13 := by sorry

end NUMINAMATH_CALUDE_days_to_finish_book_l2870_287061


namespace NUMINAMATH_CALUDE_ball_count_proof_l2870_287048

/-- 
Given a bag with m balls, including 6 red balls, 
if the probability of picking a red ball is 0.3, then m = 20.
-/
theorem ball_count_proof (m : ℕ) (h1 : m > 0) (h2 : 6 ≤ m) : 
  (6 : ℝ) / m = 0.3 → m = 20 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l2870_287048


namespace NUMINAMATH_CALUDE_card_arrangement_exists_l2870_287090

/-- Represents a card with two sides, each containing a natural number -/
structure Card where
  side1 : Nat
  side2 : Nat

/-- Represents the set of n cards -/
def CardSet (n : Nat) := {cards : Finset Card // cards.card = n}

/-- Predicate to check if a set of cards satisfies the problem conditions -/
def ValidCardSet (n : Nat) (cards : CardSet n) : Prop :=
  (∀ i : Nat, i ∈ Finset.range n → (cards.val.filter (λ c => c.side1 = i + 1 ∨ c.side2 = i + 1)).card = 2) ∧
  (∀ c : Card, c ∈ cards.val → c.side1 ≤ n ∧ c.side2 ≤ n)

/-- Represents an arrangement of cards on the table -/
def Arrangement (n : Nat) := Fin n → Bool

/-- Predicate to check if an arrangement is valid (shows numbers 1 to n exactly once) -/
def ValidArrangement (n : Nat) (cards : CardSet n) (arr : Arrangement n) : Prop :=
  ∀ i : Fin n, ∃! c : Card, c ∈ cards.val ∧
    ((arr i = true ∧ c.side1 = i + 1) ∨ (arr i = false ∧ c.side2 = i + 1))

theorem card_arrangement_exists (n : Nat) (cards : CardSet n) 
  (h : ValidCardSet n cards) : ∃ arr : Arrangement n, ValidArrangement n cards arr := by
  sorry

end NUMINAMATH_CALUDE_card_arrangement_exists_l2870_287090


namespace NUMINAMATH_CALUDE_min_value_condition_l2870_287097

open Set

variables {f : ℝ → ℝ} {a b : ℝ}

theorem min_value_condition (h_diff : Differentiable ℝ f) (h_cont : ContinuousOn f (Ioo a b)) :
  (∃ x₀ ∈ Ioo a b, deriv f x₀ = 0) →
  (∃ x_min ∈ Ioo a b, ∀ x ∈ Ioo a b, f x_min ≤ f x) ∧
  ¬ ((∃ x_min ∈ Ioo a b, ∀ x ∈ Ioo a b, f x_min ≤ f x) →
     (∃ x₀ ∈ Ioo a b, deriv f x₀ = 0)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_condition_l2870_287097


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l2870_287093

theorem sum_of_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ 1 + i + i^2 + i^3 = i := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l2870_287093


namespace NUMINAMATH_CALUDE_max_value_of_product_l2870_287055

/-- The function f(x) = 6x^3 - ax^2 - 2bx + 2 -/
def f (a b x : ℝ) : ℝ := 6 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 18 * x^2 - 2 * a * x - 2 * b

theorem max_value_of_product (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f' a b 1 = 0) : 
  a * b ≤ (81 : ℝ) / 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ f' a₀ b₀ 1 = 0 ∧ a₀ * b₀ = (81 : ℝ) / 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_product_l2870_287055


namespace NUMINAMATH_CALUDE_doug_marbles_l2870_287080

theorem doug_marbles (ed_initial : ℕ) (doug_initial : ℕ) (ed_lost : ℕ) (ed_current : ℕ) 
  (h1 : ed_initial = doug_initial + 12)
  (h2 : ed_lost = 20)
  (h3 : ed_current = 17)
  (h4 : ed_initial = ed_current + ed_lost) : 
  doug_initial = 25 := by
sorry

end NUMINAMATH_CALUDE_doug_marbles_l2870_287080


namespace NUMINAMATH_CALUDE_average_weight_increase_l2870_287023

theorem average_weight_increase (W : ℝ) : 
  let original_average := (W + 45) / 10
  let new_average := (W + 75) / 10
  new_average - original_average = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2870_287023


namespace NUMINAMATH_CALUDE_sum_after_transformation_l2870_287021

theorem sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  2 * ((a + 3) + (b + 3)) = 2 * S + 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_transformation_l2870_287021


namespace NUMINAMATH_CALUDE_max_value_of_f_on_S_l2870_287062

/-- The set S of real numbers x where x^4 - 13x^2 + 36 ≤ 0 -/
def S : Set ℝ := {x : ℝ | x^4 - 13*x^2 + 36 ≤ 0}

/-- The function f(x) = x^3 - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- Theorem stating that the maximum value of f(x) on S is 18 -/
theorem max_value_of_f_on_S : ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), x ∈ S → f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_S_l2870_287062


namespace NUMINAMATH_CALUDE_johns_initial_squat_weight_l2870_287092

/-- Calculates John's initial squat weight based on given conditions --/
theorem johns_initial_squat_weight :
  ∀ (initial_bench initial_deadlift new_total : ℝ),
  initial_bench = 400 →
  initial_deadlift = 800 →
  new_total = 1490 →
  ∃ (initial_squat : ℝ),
    initial_squat * 0.7 + initial_bench + (initial_deadlift - 200) = new_total ∧
    initial_squat = 700 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_initial_squat_weight_l2870_287092


namespace NUMINAMATH_CALUDE_sailboat_problem_l2870_287077

theorem sailboat_problem (small_sail_size : ℝ) (small_sail_speed : ℝ) 
  (big_sail_speed : ℝ) (distance : ℝ) (time_difference : ℝ) :
  small_sail_size = 12 →
  small_sail_speed = 20 →
  big_sail_speed = 50 →
  distance = 200 →
  time_difference = 6 →
  distance / small_sail_speed - distance / big_sail_speed = time_difference →
  ∃ big_sail_size : ℝ, 
    big_sail_size = 30 ∧ 
    small_sail_speed / big_sail_speed = small_sail_size / big_sail_size :=
by sorry


end NUMINAMATH_CALUDE_sailboat_problem_l2870_287077


namespace NUMINAMATH_CALUDE_sequence_sum_equals_exp_l2870_287016

/-- Given a positive integer m, y_k is a sequence defined by:
    y_0 = 1
    y_1 = m
    y_{k+2} = ((m+1)y_{k+1} - (m-k)y_k) / (k+1) for k ≥ 0
    This theorem states that the sum of all terms in the sequence equals e^(m+1) -/
theorem sequence_sum_equals_exp (m : ℕ+) : ∃ (y : ℕ → ℝ), 
  y 0 = 1 ∧ 
  y 1 = m ∧ 
  (∀ k : ℕ, y (k + 2) = ((m + 1 : ℝ) * y (k + 1) - (m - k) * y k) / (k + 1)) ∧
  (∑' k, y k) = Real.exp (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_exp_l2870_287016


namespace NUMINAMATH_CALUDE_bees_flew_in_l2870_287022

theorem bees_flew_in (initial_bees final_bees : ℕ) (h1 : initial_bees = 16) (h2 : final_bees = 23) :
  final_bees - initial_bees = 7 := by
sorry

end NUMINAMATH_CALUDE_bees_flew_in_l2870_287022


namespace NUMINAMATH_CALUDE_cans_ratio_theorem_l2870_287056

/-- Represents the number of cans collected by each person -/
structure CansCollected where
  solomon : ℕ
  juwan : ℕ
  levi : ℕ

/-- Represents a ratio between two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- The theorem to be proved -/
theorem cans_ratio_theorem (c : CansCollected) 
  (h1 : c.solomon = 66)
  (h2 : c.solomon + c.juwan + c.levi = 99)
  (h3 : c.levi = c.juwan / 2)
  : Ratio.mk 3 1 = Ratio.mk c.solomon c.juwan := by
  sorry

#check cans_ratio_theorem

end NUMINAMATH_CALUDE_cans_ratio_theorem_l2870_287056


namespace NUMINAMATH_CALUDE_only_set_D_forms_triangle_l2870_287005

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- The given sets of line segments -/
def set_A : Vector ℝ 3 := ⟨[5, 11, 6], by simp⟩
def set_B : Vector ℝ 3 := ⟨[8, 8, 16], by simp⟩
def set_C : Vector ℝ 3 := ⟨[10, 5, 4], by simp⟩
def set_D : Vector ℝ 3 := ⟨[6, 9, 14], by simp⟩

/-- Theorem: Among the given sets, only set D can form a triangle -/
theorem only_set_D_forms_triangle :
  ¬(can_form_triangle set_A[0] set_A[1] set_A[2]) ∧
  ¬(can_form_triangle set_B[0] set_B[1] set_B[2]) ∧
  ¬(can_form_triangle set_C[0] set_C[1] set_C[2]) ∧
  can_form_triangle set_D[0] set_D[1] set_D[2] :=
by sorry

end NUMINAMATH_CALUDE_only_set_D_forms_triangle_l2870_287005


namespace NUMINAMATH_CALUDE_unique_valid_tournament_l2870_287051

/-- Represents the result of a chess game -/
inductive GameResult
  | Win
  | Draw
  | Loss

/-- Represents a player in the chess tournament -/
structure Player where
  id : Fin 5
  score : Rat

/-- Represents the result of a game between two players -/
structure GameOutcome where
  player1 : Fin 5
  player2 : Fin 5
  result : GameResult

/-- Represents the chess tournament -/
structure ChessTournament where
  players : Fin 5 → Player
  games : List GameOutcome

def ChessTournament.isValid (t : ChessTournament) : Prop :=
  -- Each player played exactly once with each other
  (t.games.length = 10) ∧
  -- First-place winner had no draws
  (¬ ∃ g ∈ t.games, g.player1 = 0 ∧ g.result = GameResult.Draw) ∧
  (¬ ∃ g ∈ t.games, g.player2 = 0 ∧ g.result = GameResult.Draw) ∧
  -- Second-place winner did not lose any game
  (¬ ∃ g ∈ t.games, g.player1 = 1 ∧ g.result = GameResult.Loss) ∧
  (¬ ∃ g ∈ t.games, g.player2 = 1 ∧ g.result = GameResult.Win) ∧
  -- Fourth-place player did not win any game
  (¬ ∃ g ∈ t.games, g.player1 = 3 ∧ g.result = GameResult.Win) ∧
  (¬ ∃ g ∈ t.games, g.player2 = 3 ∧ g.result = GameResult.Loss) ∧
  -- Scores of all participants were different
  (∀ i j : Fin 5, i ≠ j → (t.players i).score ≠ (t.players j).score)

/-- The unique valid tournament configuration -/
def uniqueTournament : ChessTournament := sorry

theorem unique_valid_tournament :
  ∀ t : ChessTournament, t.isValid → t = uniqueTournament := by sorry

end NUMINAMATH_CALUDE_unique_valid_tournament_l2870_287051


namespace NUMINAMATH_CALUDE_salad_dressing_weight_l2870_287078

/-- Calculates the total weight of a salad dressing mixture --/
theorem salad_dressing_weight (bowl_capacity : ℝ) (oil_fraction vinegar_fraction : ℝ)
  (oil_density vinegar_density : ℝ) :
  bowl_capacity = 150 ∧
  oil_fraction = 2/3 ∧
  vinegar_fraction = 1/3 ∧
  oil_density = 5 ∧
  vinegar_density = 4 →
  bowl_capacity * oil_fraction * oil_density +
  bowl_capacity * vinegar_fraction * vinegar_density = 700 := by
  sorry

end NUMINAMATH_CALUDE_salad_dressing_weight_l2870_287078


namespace NUMINAMATH_CALUDE_seojun_pizza_problem_l2870_287011

/-- Seojun's pizza problem -/
theorem seojun_pizza_problem (initial_pizza : ℚ) : 
  initial_pizza - 7/3 = 3/2 →
  initial_pizza + 7/3 = 37/6 := by
  sorry

end NUMINAMATH_CALUDE_seojun_pizza_problem_l2870_287011


namespace NUMINAMATH_CALUDE_reflect_F_coordinates_l2870_287096

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The composition of reflecting over y-axis and then x-axis -/
def reflect_yx (p : ℝ × ℝ) : ℝ × ℝ := reflect_x (reflect_y p)

theorem reflect_F_coordinates :
  reflect_yx (6, -4) = (-6, 4) := by sorry

end NUMINAMATH_CALUDE_reflect_F_coordinates_l2870_287096


namespace NUMINAMATH_CALUDE_simple_annual_interest_rate_l2870_287057

/-- Simple annual interest rate calculation -/
theorem simple_annual_interest_rate 
  (monthly_interest : ℝ) 
  (investment_amount : ℝ) 
  (h1 : monthly_interest = 225)
  (h2 : investment_amount = 30000) : 
  (monthly_interest * 12) / investment_amount = 0.09 := by
sorry

end NUMINAMATH_CALUDE_simple_annual_interest_rate_l2870_287057


namespace NUMINAMATH_CALUDE_area_between_parabola_and_line_l2870_287059

theorem area_between_parabola_and_line : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := 1
  let lower_bound := -1
  let upper_bound := 1
  (∫ (x : ℝ) in lower_bound..upper_bound, g x - f x) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_area_between_parabola_and_line_l2870_287059


namespace NUMINAMATH_CALUDE_handshake_count_l2870_287009

theorem handshake_count (n : ℕ) (h : n = 8) : n * (n - 1) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l2870_287009


namespace NUMINAMATH_CALUDE_school_population_l2870_287026

/-- Given a school with boys, girls, and teachers, prove that the total number of people is 61t, where t is the number of teachers. -/
theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 12 * t) : 
  b + g + t = 61 * t := by
  sorry

end NUMINAMATH_CALUDE_school_population_l2870_287026


namespace NUMINAMATH_CALUDE_smallest_class_size_l2870_287076

theorem smallest_class_size (n : ℕ) (h1 : n > 50) 
  (h2 : ∃ x : ℕ, n = 3*x + 2*(x+1)) : n ≥ 52 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2870_287076


namespace NUMINAMATH_CALUDE_meaningful_fraction_l2870_287039

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (x - 3) / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l2870_287039


namespace NUMINAMATH_CALUDE_arithmetic_progression_relatively_prime_l2870_287002

theorem arithmetic_progression_relatively_prime :
  ∃ (a : ℕ → ℕ) (d : ℕ),
    (∀ n, 1 ≤ n → n ≤ 100 → a n > 0) ∧
    (∀ n m, 1 ≤ n → n < m → m ≤ 100 → a m > a n) ∧
    (∀ n, 1 < n → n ≤ 100 → a n - a (n-1) = d) ∧
    (∀ n m, 1 ≤ n → n < m → m ≤ 100 → Nat.gcd (a n) (a m) = 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_relatively_prime_l2870_287002


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2870_287063

/-- The parabola function y = x^2 - 6x + 8 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- The vertex of the parabola -/
def parabola_vertex : ℝ × ℝ := (3, -1)

/-- Predicate to check if a point is on the parabola -/
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2 = parabola p.1

/-- Predicate to check if a point is on the x-axis -/
def on_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 = 0

/-- Definition of a square inscribed in the parabola region -/
structure InscribedSquare :=
  (center : ℝ × ℝ)
  (side_length : ℝ)
  (vertex_on_parabola : on_parabola (center.1 - side_length/2, center.2 - side_length/2))
  (bottom_left_on_x_axis : on_x_axis (center.1 - side_length/2, center.2 + side_length/2))
  (bottom_right_on_x_axis : on_x_axis (center.1 + side_length/2, center.2 + side_length/2))
  (top_right_on_parabola : on_parabola (center.1 + side_length/2, center.2 - side_length/2))

/-- Theorem: The area of the inscribed square is 12 - 8√2 -/
theorem inscribed_square_area :
  ∃ (s : InscribedSquare), s.side_length^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2870_287063


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l2870_287024

theorem x_range_for_quadratic_inequality (x : ℝ) :
  (∀ a : ℝ, -1 ≤ a ∧ a ≤ 1 → x^2 + (a-4)*x + 4-2*a > 0) ↔
  (x < -3 ∨ x > -2) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l2870_287024


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2870_287008

theorem consecutive_integers_sum (x : ℤ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2870_287008


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l2870_287085

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

/-- The tangent line function -/
def g (x b : ℝ) : ℝ := -3*x + b

theorem tangent_line_b_value :
  ∀ b : ℝ, (∃ x : ℝ, f x = g x b ∧ f' x = -3) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l2870_287085


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l2870_287046

theorem correct_quotient_proof (N : ℕ) (h1 : N % 21 = 0) (h2 : N / 12 = 70) : N / 21 = 40 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l2870_287046


namespace NUMINAMATH_CALUDE_egg_collection_total_l2870_287014

/-- The number of dozen eggs Benjamin collects -/
def benjamin_eggs : ℕ := 6

/-- The number of dozen eggs Carla collects -/
def carla_eggs : ℕ := 3 * benjamin_eggs

/-- The number of dozen eggs Trisha collects -/
def trisha_eggs : ℕ := benjamin_eggs - 4

/-- The total number of dozen eggs collected by Benjamin, Carla, and Trisha -/
def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem egg_collection_total : total_eggs = 26 := by
  sorry

end NUMINAMATH_CALUDE_egg_collection_total_l2870_287014


namespace NUMINAMATH_CALUDE_antoinette_weight_l2870_287050

theorem antoinette_weight (rupert_weight : ℝ) : 
  let antoinette_weight := 2 * rupert_weight - 7
  (antoinette_weight + rupert_weight = 98) → antoinette_weight = 63 := by
sorry

end NUMINAMATH_CALUDE_antoinette_weight_l2870_287050


namespace NUMINAMATH_CALUDE_cyrus_remaining_pages_l2870_287031

/-- Represents the number of pages written on each day --/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Calculates the remaining pages to be written --/
def remainingPages (total : ℕ) (daily : DailyPages) : ℕ :=
  total - (daily.day1 + daily.day2 + daily.day3 + daily.day4)

/-- Theorem stating the number of remaining pages Cyrus needs to write --/
theorem cyrus_remaining_pages :
  let total := 500
  let daily := DailyPages.mk 25 (25 * 2) ((25 * 2) * 2) 10
  remainingPages total daily = 315 := by
  sorry

end NUMINAMATH_CALUDE_cyrus_remaining_pages_l2870_287031


namespace NUMINAMATH_CALUDE_number_of_shortest_paths_is_54_l2870_287069

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents the grid configuration -/
structure Grid where
  squareSize : ℕ  -- Side length of each square in km
  refuelDistance : ℕ  -- Distance the car can travel before refueling in km

/-- Calculates the number of shortest paths between two points on the grid -/
def numberOfShortestPaths (g : Grid) (start finish : GridPoint) : ℕ :=
  sorry

/-- The specific grid configuration for the problem -/
def problemGrid : Grid :=
  { squareSize := 10
  , refuelDistance := 30 }

/-- The start point A -/
def pointA : GridPoint :=
  { x := 0, y := 0 }

/-- The end point B -/
def pointB : GridPoint :=
  { x := 6, y := 6 }  -- Assuming a 6x6 grid based on the problem description

theorem number_of_shortest_paths_is_54 :
  numberOfShortestPaths problemGrid pointA pointB = 54 :=
by sorry

end NUMINAMATH_CALUDE_number_of_shortest_paths_is_54_l2870_287069


namespace NUMINAMATH_CALUDE_prob_four_sixes_eq_one_over_1296_l2870_287036

-- Define a fair six-sided die
def fair_six_sided_die : Finset ℕ := Finset.range 6

-- Define the probability of rolling a specific number on a fair six-sided die
def prob_single_roll (n : ℕ) : ℚ :=
  if n ∈ fair_six_sided_die then 1 / 6 else 0

-- Define the probability of rolling four sixes
def prob_four_sixes : ℚ := (prob_single_roll 6) ^ 4

-- Theorem statement
theorem prob_four_sixes_eq_one_over_1296 :
  prob_four_sixes = 1 / 1296 := by sorry

end NUMINAMATH_CALUDE_prob_four_sixes_eq_one_over_1296_l2870_287036


namespace NUMINAMATH_CALUDE_pencil_distribution_ways_l2870_287072

def distribute_pencils (n : ℕ) (k : ℕ) (min_first : ℕ) (min_others : ℕ) : ℕ :=
  Nat.choose (n - (min_first + (k - 1) * min_others) + k - 1) (k - 1)

theorem pencil_distribution_ways : 
  distribute_pencils 8 4 2 1 = 20 := by sorry

end NUMINAMATH_CALUDE_pencil_distribution_ways_l2870_287072


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2870_287052

/-- For all real numbers a and b, a²b - 25b = b(a + 5)(a - 5) -/
theorem polynomial_factorization (a b : ℝ) : a^2 * b - 25 * b = b * (a + 5) * (a - 5) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2870_287052


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2870_287098

def f (x : ℝ) : ℝ := |x - 1|

theorem f_increasing_on_interval : 
  ∀ x y : ℝ, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f x ≤ f y :=
by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2870_287098


namespace NUMINAMATH_CALUDE_base_10_678_to_base_7_l2870_287086

/-- Converts a base-10 integer to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base-7 to a base-10 integer -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_10_678_to_base_7 :
  toBase7 678 = [1, 6, 5, 6] ∧ fromBase7 [1, 6, 5, 6] = 678 := by
  sorry

end NUMINAMATH_CALUDE_base_10_678_to_base_7_l2870_287086


namespace NUMINAMATH_CALUDE_prime_factor_sum_squares_l2870_287032

theorem prime_factor_sum_squares (n : ℕ+) : 
  (∃ p q : ℕ, 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p ∣ n ∧ 
    q ∣ n ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ n → p ≤ r) ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ n → r ≤ q) ∧
    p^2 + q^2 = n + 9) ↔ 
  n = 9 ∨ n = 20 := by
sorry


end NUMINAMATH_CALUDE_prime_factor_sum_squares_l2870_287032


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l2870_287081

theorem quadratic_inequality_always_true :
  ∀ x : ℝ, 3 * x^2 + 9 * x ≥ -12 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l2870_287081


namespace NUMINAMATH_CALUDE_max_min_constrained_optimization_l2870_287088

-- Define the constraint function
def constraint (x y : ℝ) : Prop :=
  Real.sqrt (x - 2) + Real.sqrt (y - 3) = 3

-- Define the objective function
def objective (x y : ℝ) : ℝ :=
  x - 2*y

-- Theorem statement
theorem max_min_constrained_optimization :
  ∃ (x_max y_max x_min y_min : ℝ),
    constraint x_max y_max ∧
    constraint x_min y_min ∧
    (∀ x y, constraint x y → objective x y ≤ objective x_max y_max) ∧
    (∀ x y, constraint x y → objective x_min y_min ≤ objective x y) ∧
    x_max = 11 ∧ y_max = 3 ∧
    x_min = 2 ∧ y_min = 12 ∧
    objective x_max y_max = 5 ∧
    objective x_min y_min = -22 :=
  sorry

end NUMINAMATH_CALUDE_max_min_constrained_optimization_l2870_287088


namespace NUMINAMATH_CALUDE_greatest_integer_in_odd_set_l2870_287065

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def is_consecutive_odd_set (s : Set ℤ) : Prop :=
  ∃ a b : ℤ, a ≤ b ∧ s = {n : ℤ | a ≤ n ∧ n ≤ b ∧ is_odd n ∧ ∀ m : ℤ, a ≤ m ∧ m < n → is_odd m}

def median (s : Set ℤ) : ℤ := sorry

theorem greatest_integer_in_odd_set (s : Set ℤ) :
  is_consecutive_odd_set s →
  155 ∈ s →
  median s = 167 →
  ∃ m : ℤ, m ∈ s ∧ ∀ n ∈ s, n ≤ m ∧ m = 179 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_in_odd_set_l2870_287065


namespace NUMINAMATH_CALUDE_books_bought_l2870_287084

theorem books_bought (initial_books final_books : ℕ) 
  (h1 : initial_books = 50)
  (h2 : final_books = 151) :
  final_books - initial_books = 101 := by
  sorry

end NUMINAMATH_CALUDE_books_bought_l2870_287084


namespace NUMINAMATH_CALUDE_cupcake_net_profit_l2870_287074

/-- Calculates the net profit from selling cupcakes given the specified conditions -/
theorem cupcake_net_profit :
  let cupcake_cost : ℚ := 0.75
  let burnt_cupcakes : ℕ := 24
  let first_batch : ℕ := 24
  let second_batch : ℕ := 24
  let eaten_immediately : ℕ := 5
  let eaten_later : ℕ := 4
  let selling_price : ℚ := 2

  let total_cupcakes : ℕ := burnt_cupcakes + first_batch + second_batch
  let total_cost : ℚ := cupcake_cost * total_cupcakes
  let cupcakes_to_sell : ℕ := total_cupcakes - burnt_cupcakes - eaten_immediately - eaten_later
  let revenue : ℚ := selling_price * cupcakes_to_sell
  let net_profit : ℚ := revenue - total_cost

  net_profit = 72 := by sorry

end NUMINAMATH_CALUDE_cupcake_net_profit_l2870_287074


namespace NUMINAMATH_CALUDE_quadratic_properties_l2870_287087

/-- Given a quadratic function y = (x - m)² - 2(x - m), where m is a constant -/
def f (x m : ℝ) : ℝ := (x - m)^2 - 2*(x - m)

theorem quadratic_properties (m : ℝ) :
  /- The x-intercepts are at x = m and x = m + 2 -/
  (∃ x, f x m = 0 ↔ x = m ∨ x = m + 2) ∧
  /- The vertex is at (m + 1, -1) -/
  (f (m + 1) m = -1 ∧ ∀ x, f x m ≥ -1) ∧
  /- When the graph is shifted 3 units left and 1 unit up to become y = x², m = 2 -/
  (∀ x, f (x + 3) m - 1 = x^2 → m = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2870_287087


namespace NUMINAMATH_CALUDE_river_joe_collection_l2870_287012

/-- Represents the total money collected by River Joe's Seafood Diner --/
def total_money_collected (catfish_price popcorn_shrimp_price : ℚ) 
  (total_orders popcorn_shrimp_orders : ℕ) : ℚ :=
  let catfish_orders := total_orders - popcorn_shrimp_orders
  catfish_price * catfish_orders + popcorn_shrimp_price * popcorn_shrimp_orders

/-- Proves that River Joe collected $133.50 given the specified conditions --/
theorem river_joe_collection : 
  total_money_collected 6 3.5 26 9 = 133.5 := by
  sorry

#eval total_money_collected 6 3.5 26 9

end NUMINAMATH_CALUDE_river_joe_collection_l2870_287012


namespace NUMINAMATH_CALUDE_min_elements_special_relation_l2870_287047

/-- A relation on a set X satisfying the given properties -/
structure SpecialRelation (X : Type) where
  rel : X → X → Prop
  irreflexive : ∀ x, ¬(rel x x)
  trichotomous : ∀ x y, x ≠ y → (rel x y ∨ rel y x) ∧ ¬(rel x y ∧ rel y x)
  transitive_element : ∀ x y, rel x y → ∃ z, rel x z ∧ rel z y

/-- The minimum number of elements in a set with a SpecialRelation is 7 -/
theorem min_elements_special_relation :
  ∀ (X : Type) [Fintype X] (r : SpecialRelation X),
  Fintype.card X ≥ 7 ∧ (∀ (Y : Type) [Fintype Y], SpecialRelation Y → Fintype.card Y < 7 → False) :=
sorry

end NUMINAMATH_CALUDE_min_elements_special_relation_l2870_287047


namespace NUMINAMATH_CALUDE_percent_to_decimal_five_percent_to_decimal_l2870_287058

theorem percent_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem five_percent_to_decimal : (5 : ℚ) / 100 = 0.05 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_five_percent_to_decimal_l2870_287058


namespace NUMINAMATH_CALUDE_sum_of_coefficients_cubic_factorization_l2870_287003

theorem sum_of_coefficients_cubic_factorization :
  ∃ (p q r s t : ℤ), 
    (∀ y, 512 * y^3 + 27 = (p * y + q) * (r * y^2 + s * y + t)) ∧
    p + q + r + s + t = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_cubic_factorization_l2870_287003


namespace NUMINAMATH_CALUDE_opposite_of_three_l2870_287013

theorem opposite_of_three : -(3 : ℤ) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l2870_287013


namespace NUMINAMATH_CALUDE_sqrt_13_parts_sum_l2870_287038

theorem sqrt_13_parts_sum (a b : ℝ) : 
  (a = ⌊Real.sqrt 13⌋) → 
  (b = Real.sqrt 13 - ⌊Real.sqrt 13⌋) → 
  2 * a^2 + b - Real.sqrt 13 = 15 := by
sorry

end NUMINAMATH_CALUDE_sqrt_13_parts_sum_l2870_287038


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2870_287017

theorem complex_fraction_simplification :
  let z : ℂ := (3 - 2*I) / (5 - 2*I)
  z = 19/29 - (4/29)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2870_287017


namespace NUMINAMATH_CALUDE_emily_beads_count_l2870_287082

/-- Given that Emily makes necklaces where each necklace requires 12 beads,
    and she made 7 necklaces, prove that the total number of beads she had is 84. -/
theorem emily_beads_count :
  let beads_per_necklace : ℕ := 12
  let necklaces_made : ℕ := 7
  beads_per_necklace * necklaces_made = 84 :=
by sorry

end NUMINAMATH_CALUDE_emily_beads_count_l2870_287082


namespace NUMINAMATH_CALUDE_sin_75_degrees_l2870_287028

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l2870_287028


namespace NUMINAMATH_CALUDE_bill_division_l2870_287054

/-- Proves that when three people divide a 99-dollar bill evenly, each person pays 33 dollars. -/
theorem bill_division (total_bill : ℕ) (num_people : ℕ) (each_share : ℕ) :
  total_bill = 99 → num_people = 3 → each_share = total_bill / num_people → each_share = 33 := by
  sorry

#check bill_division

end NUMINAMATH_CALUDE_bill_division_l2870_287054


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2870_287019

theorem mean_equality_implies_z_value :
  let mean1 := (8 + 12 + 24) / 3
  let mean2 := (16 + z) / 2
  mean1 = mean2 → z = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2870_287019


namespace NUMINAMATH_CALUDE_product_pricing_l2870_287073

/-- Given a cost per unit, original markup percentage, and current price percentage,
    calculate the current selling price and profit per unit. -/
theorem product_pricing (a : ℝ) (h : a > 0) :
  let original_price := a * (1 + 0.22)
  let current_price := original_price * 0.85
  let profit := current_price - a
  (current_price = 1.037 * a) ∧ (profit = 0.037 * a) := by
  sorry

end NUMINAMATH_CALUDE_product_pricing_l2870_287073


namespace NUMINAMATH_CALUDE_power_sum_integer_l2870_287010

theorem power_sum_integer (x : ℝ) (h : ∃ (k : ℤ), x + 1/x = k) :
  ∀ (n : ℕ), n > 0 → ∃ (m : ℤ), x^n + 1/(x^n) = m :=
by sorry

end NUMINAMATH_CALUDE_power_sum_integer_l2870_287010


namespace NUMINAMATH_CALUDE_smallest_list_size_l2870_287079

theorem smallest_list_size (n a b : ℕ) (h1 : n = a + b) (h2 : 89 * n = 73 * a + 111 * b) : n ≥ 19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_list_size_l2870_287079


namespace NUMINAMATH_CALUDE_soccer_match_players_l2870_287070

/-- The number of players in a soccer match -/
def num_players : ℕ := 11

/-- The total number of socks in the washing machine -/
def total_socks : ℕ := 22

/-- Each player wears exactly two socks -/
def socks_per_player : ℕ := 2

/-- Theorem: The number of players is 11 given the conditions -/
theorem soccer_match_players :
  num_players = total_socks / socks_per_player :=
by sorry

end NUMINAMATH_CALUDE_soccer_match_players_l2870_287070


namespace NUMINAMATH_CALUDE_prob_red_card_standard_deck_l2870_287041

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- Represents the properties of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4,
    red_suits := 2,
    black_suits := 2 }

/-- Calculates the probability of drawing a red card from the deck -/
def prob_red_card (d : Deck) : ℚ :=
  (d.ranks * d.red_suits : ℚ) / d.total_cards

/-- Theorem stating that the probability of drawing a red card from a standard deck is 1/2 -/
theorem prob_red_card_standard_deck : 
  prob_red_card standard_deck = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_red_card_standard_deck_l2870_287041


namespace NUMINAMATH_CALUDE_at_least_one_goes_probability_l2870_287053

def prob_at_least_one_goes (prob_A prob_B : ℚ) : Prop :=
  1 - (1 - prob_A) * (1 - prob_B) = 2/5

theorem at_least_one_goes_probability :
  prob_at_least_one_goes (1/4 : ℚ) (1/5 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_at_least_one_goes_probability_l2870_287053


namespace NUMINAMATH_CALUDE_angle_bac_equals_arcsin_four_fifths_l2870_287000

-- Define the triangle ABC and point O
structure Triangle :=
  (A B C O : ℝ × ℝ)

-- Define the distances OA, OB, OC
def distOA (t : Triangle) : ℝ := 15
def distOB (t : Triangle) : ℝ := 12
def distOC (t : Triangle) : ℝ := 20

-- Define the property that the feet of perpendiculars form an equilateral triangle
def perpendicularsFormEquilateralTriangle (t : Triangle) : Prop := sorry

-- Define the angle BAC
def angleBac (t : Triangle) : ℝ := sorry

-- State the theorem
theorem angle_bac_equals_arcsin_four_fifths (t : Triangle) :
  distOA t = 15 →
  distOB t = 12 →
  distOC t = 20 →
  perpendicularsFormEquilateralTriangle t →
  angleBac t = Real.arcsin (4/5) :=
by sorry

end NUMINAMATH_CALUDE_angle_bac_equals_arcsin_four_fifths_l2870_287000


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2870_287020

def distance : ℝ := 360
def time : ℝ := 4.5

theorem average_speed_calculation : distance / time = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2870_287020


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2870_287068

theorem complex_expression_simplification :
  let a : ℂ := 3 - I
  let b : ℂ := 2 + I
  let c : ℂ := -1 + 2 * I
  3 * a + 4 * b - 2 * c = 19 := by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2870_287068


namespace NUMINAMATH_CALUDE_intersection_when_m_3_m_value_for_given_intersection_l2870_287034

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | -x^2 + 2*x + m > 0}

-- Theorem 1: Intersection of A and B when m = 3
theorem intersection_when_m_3 :
  A ∩ B 3 = {x : ℝ | -1 < x ∧ x < 3} := by sorry

-- Theorem 2: Value of m when A ∩ B = {x | -1 < x < 4}
theorem m_value_for_given_intersection :
  ∃ m : ℝ, A ∩ B m = {x : ℝ | -1 < x ∧ x < 4} ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_m_value_for_given_intersection_l2870_287034


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2870_287045

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 24) :
  |x - y| = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2870_287045


namespace NUMINAMATH_CALUDE_parabola_chord_theorem_l2870_287033

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (s : ℝ) : Prop := parabola s 4

-- Define perpendicular lines
def perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

-- Define a point (x,y) on line AB
def point_on_AB (x y y1 y2 : ℝ) : Prop := y + 4 = (4 / (y1 + y2)) * (x - 8)

theorem parabola_chord_theorem :
  ∀ y1 y2 : ℝ,
  point_on_parabola 4 →
  parabola (y1^2 / 4) y1 →
  parabola (y2^2 / 4) y2 →
  perpendicular ((y1 - 4) / ((y1^2 - 16) / 4)) ((y2 - 4) / ((y2^2 - 16) / 4)) →
  point_on_AB 8 (-4) y1 y2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_chord_theorem_l2870_287033


namespace NUMINAMATH_CALUDE_ratio_problem_l2870_287060

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (x + 2 * y) = 4 / 5) : 
  x / y = 18 / 11 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l2870_287060


namespace NUMINAMATH_CALUDE_coeff_x20_Q_greater_than_P_l2870_287089

-- Define the two expressions
def P (x : ℝ) : ℝ := (1 - x^2 + x^3)^1000
def Q (x : ℝ) : ℝ := (1 + x^2 - x^3)^1000

-- Define a function to get the coefficient of x^20 in a polynomial
noncomputable def coeff_x20 (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem coeff_x20_Q_greater_than_P :
  coeff_x20 Q > coeff_x20 P := by sorry

end NUMINAMATH_CALUDE_coeff_x20_Q_greater_than_P_l2870_287089


namespace NUMINAMATH_CALUDE_function_property_l2870_287044

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y^3 * f x = x^3 * f y

theorem function_property (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 3 ≠ 0) :
  (f 20 - f 2) / f 3 = 296 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2870_287044


namespace NUMINAMATH_CALUDE_r_bounds_for_area_range_l2870_287075

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 + 2

/-- The line function -/
def line (r : ℝ) (x : ℝ) : ℝ := r - 1

/-- The intersection points of the parabola and the line -/
def intersection_points (r : ℝ) : Set ℝ := {x | parabola x = line r x}

/-- The area of the triangle formed by the vertex of the parabola and the intersection points -/
def triangle_area (r : ℝ) : ℝ := (r - 3)^(3/2)

/-- Theorem stating the relationship between r and the area of the triangle -/
theorem r_bounds_for_area_range :
  ∀ r : ℝ, (16 ≤ triangle_area r ∧ triangle_area r ≤ 128) ↔ (7 ≤ r ∧ r ≤ 19) :=
sorry

end NUMINAMATH_CALUDE_r_bounds_for_area_range_l2870_287075


namespace NUMINAMATH_CALUDE_clothing_store_profit_l2870_287071

/-- Profit function for a clothing store --/
def profit_function (x : ℝ) : ℝ := 20 * x + 4000

/-- Maximum profit under cost constraint --/
def max_profit : ℝ := 5500

/-- Discount value for maximum profit under new conditions --/
def discount_value : ℝ := 9

/-- Theorem stating the main results --/
theorem clothing_store_profit :
  (∀ x : ℝ, x ≥ 60 → x ≤ 100 → profit_function x = 20 * x + 4000) ∧
  (∀ x : ℝ, x ≥ 60 → x ≤ 75 → 160 * x + 120 * (100 - x) ≤ 15000 → profit_function x ≤ max_profit) ∧
  (∃ x : ℝ, x ≥ 60 ∧ x ≤ 75 ∧ 160 * x + 120 * (100 - x) ≤ 15000 ∧ profit_function x = max_profit) ∧
  (∀ a : ℝ, 0 < a → a < 20 → 
    (∃ x : ℝ, x ≥ 60 ∧ x ≤ 75 ∧ 
      ((20 - a) * x + 100 * a + 3600 = 4950) → a = discount_value)) :=
by sorry

end NUMINAMATH_CALUDE_clothing_store_profit_l2870_287071


namespace NUMINAMATH_CALUDE_sandys_comic_books_l2870_287099

/-- Sandy's comic book problem -/
theorem sandys_comic_books (initial : ℕ) : 
  (initial / 2 : ℚ) + 6 = 13 → initial = 14 := by
  sorry

end NUMINAMATH_CALUDE_sandys_comic_books_l2870_287099


namespace NUMINAMATH_CALUDE_train_y_completion_time_l2870_287067

/-- Represents the time it takes for Train Y to complete the trip -/
def train_y_time (route_length : ℝ) (train_x_time : ℝ) (train_x_distance : ℝ) : ℝ :=
  4

/-- Theorem stating that Train Y takes 4 hours to complete the trip under the given conditions -/
theorem train_y_completion_time 
  (route_length : ℝ) 
  (train_x_time : ℝ) 
  (train_x_distance : ℝ)
  (h1 : route_length = 180)
  (h2 : train_x_time = 5)
  (h3 : train_x_distance = 80) :
  train_y_time route_length train_x_time train_x_distance = 4 := by
  sorry

#check train_y_completion_time

end NUMINAMATH_CALUDE_train_y_completion_time_l2870_287067


namespace NUMINAMATH_CALUDE_total_seeds_calculation_l2870_287004

/-- The number of seeds planted in each flower bed -/
def seeds_per_bed : ℕ := 6

/-- The number of flower beds -/
def num_beds : ℕ := 9

/-- The total number of seeds planted -/
def total_seeds : ℕ := seeds_per_bed * num_beds

theorem total_seeds_calculation : total_seeds = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_calculation_l2870_287004


namespace NUMINAMATH_CALUDE_subtraction_problem_l2870_287015

theorem subtraction_problem : 444 - 44 - 4 = 396 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2870_287015
