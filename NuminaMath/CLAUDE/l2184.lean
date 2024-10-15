import Mathlib

namespace NUMINAMATH_CALUDE_custom_operation_result_l2184_218499

/-- Custom operation * for non-zero integers -/
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

/-- Theorem: Given the conditions, prove that a * b = 3/8 -/
theorem custom_operation_result (a b : ℤ) 
  (h1 : a + b = 12) 
  (h2 : a * b = 32) 
  (h3 : b = 8) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) : 
  star a b = 3 / 8 := by
  sorry

#check custom_operation_result

end NUMINAMATH_CALUDE_custom_operation_result_l2184_218499


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2184_218402

theorem complex_arithmetic_equality : 
  ((-4 : ℝ) ^ 5) ^ (1/5) - (-5 : ℝ) ^ 2 - 5 + ((-43 : ℝ) ^ 4) ^ (1/4) - (-(3 : ℝ) ^ 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2184_218402


namespace NUMINAMATH_CALUDE_ellipse_constant_product_l2184_218417

def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def focus (x y : ℝ) : Prop := x = -1 ∧ y = 0

def min_distance (d : ℝ) : Prop := d = Real.sqrt 2 - 1

def point_M (x y : ℝ) : Prop := x = -5/4 ∧ y = 0

def line_intersects_ellipse (l : ℝ → ℝ → Prop) (a b : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂

def product_MA_MB (xₐ yₐ xₘ yₘ xb yb : ℝ) : ℝ :=
  ((xₐ - xₘ)^2 + (yₐ - yₘ)^2) * ((xb - xₘ)^2 + (yb - yₘ)^2)

theorem ellipse_constant_product (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  ∀ l : ℝ → ℝ → Prop,
    (∃ x y, focus x y) →
    (∃ d, min_distance d) →
    (∃ xₘ yₘ, point_M xₘ yₘ) →
    line_intersects_ellipse l a b →
    (∃ xₐ yₐ xb yb xₘ yₘ,
      l xₐ yₐ ∧ l xb yb ∧ point_M xₘ yₘ ∧
      product_MA_MB xₐ yₐ xₘ yₘ xb yb = -7/16) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_constant_product_l2184_218417


namespace NUMINAMATH_CALUDE_puzzle_sum_l2184_218458

def is_valid_puzzle (a b c d e f g h i : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i

theorem puzzle_sum (a b c d e f g h i : ℕ) :
  is_valid_puzzle a b c d e f g h i →
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f) + (100 * g + 10 * h + i) = 1665 →
  b + e + h = 15 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_sum_l2184_218458


namespace NUMINAMATH_CALUDE_scientific_notation_87000000_l2184_218497

theorem scientific_notation_87000000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 87000000 = a * (10 : ℝ) ^ n ∧ a = 8.7 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_87000000_l2184_218497


namespace NUMINAMATH_CALUDE_number_division_l2184_218485

theorem number_division (x : ℝ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l2184_218485


namespace NUMINAMATH_CALUDE_family_size_l2184_218431

/-- Given a family where one side has 10 members and the other side is 30% larger,
    the total number of family members is 23. -/
theorem family_size (fathers_side : ℕ) (mothers_side : ℕ) : 
  fathers_side = 10 →
  mothers_side = fathers_side + (fathers_side * 3 / 10) →
  fathers_side + mothers_side = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_family_size_l2184_218431


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_is_8_l2184_218436

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The height of the triangle
  height : ℝ
  -- The ratio of base to side (4:3)
  baseToSideRatio : ℚ
  -- Assumption that the height is 20
  height_is_20 : height = 20
  -- Assumption that the base to side ratio is 4:3
  ratio_is_4_3 : baseToSideRatio = 4 / 3

/-- The radius of the inscribed circle in the isosceles triangle -/
def inscribedCircleRadius (t : IsoscelesTriangle) : ℝ := 8

/-- Theorem stating that the radius of the inscribed circle is 8 -/
theorem inscribed_circle_radius_is_8 (t : IsoscelesTriangle) :
  inscribedCircleRadius t = 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_is_8_l2184_218436


namespace NUMINAMATH_CALUDE_fertilizer_per_acre_l2184_218447

theorem fertilizer_per_acre 
  (num_horses : ℕ) 
  (fertilizer_per_horse_per_day : ℕ) 
  (total_acres : ℕ) 
  (acres_per_day : ℕ) 
  (total_days : ℕ) 
  (h1 : num_horses = 80)
  (h2 : fertilizer_per_horse_per_day = 5)
  (h3 : total_acres = 20)
  (h4 : acres_per_day = 4)
  (h5 : total_days = 25) :
  (num_horses * fertilizer_per_horse_per_day * total_days) / total_acres = 500 := by
  sorry

#check fertilizer_per_acre

end NUMINAMATH_CALUDE_fertilizer_per_acre_l2184_218447


namespace NUMINAMATH_CALUDE_volunteer_allocation_schemes_l2184_218437

theorem volunteer_allocation_schemes (n : ℕ) (m : ℕ) (k : ℕ) : 
  n = 5 → m = 3 → k = 2 →
  (Nat.choose n 1) * (Nat.choose (n - 1) k / 2) * Nat.factorial m = 90 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_allocation_schemes_l2184_218437


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2184_218420

theorem sum_of_solutions (a : ℝ) (h : a > 2) : 
  ∃ x₁ x₂ : ℝ, (Real.sqrt (a - Real.sqrt (a + x₁)) = x₁ + 1) ∧ 
              (Real.sqrt (a - Real.sqrt (a + x₂)) = x₂ + 1) ∧ 
              (x₁ + x₂ = -2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2184_218420


namespace NUMINAMATH_CALUDE_angle_half_in_third_quadrant_l2184_218498

open Real

-- Define the first quadrant
def FirstQuadrant (α : ℝ) : Prop := 0 < α ∧ α < π / 2

-- Define the third quadrant
def ThirdQuadrant (θ : ℝ) : Prop := π < θ ∧ θ < 3 * π / 2

-- State the theorem
theorem angle_half_in_third_quadrant (α : ℝ) 
  (h1 : FirstQuadrant α) 
  (h2 : |cos (α / 2)| = -cos (α / 2)) : 
  ThirdQuadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_half_in_third_quadrant_l2184_218498


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2184_218463

/-- A geometric sequence with given first and fourth terms -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ a 4 = -4 ∧ ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The common ratio of the geometric sequence -/
def common_ratio (a : ℕ → ℚ) : ℚ :=
  (a 2) / (a 1)

/-- Theorem: Properties of the geometric sequence -/
theorem geometric_sequence_properties (a : ℕ → ℚ) 
  (h : geometric_sequence a) : 
  common_ratio a = -2 ∧ ∀ n : ℕ, a n = 1/2 * (-2)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2184_218463


namespace NUMINAMATH_CALUDE_average_math_chemistry_l2184_218446

-- Define the marks for each subject
variable (M P C : ℕ)

-- Define the conditions
axiom total_math_physics : M + P = 70
axiom chemistry_score : C = P + 20

-- Define the theorem to prove
theorem average_math_chemistry : (M + C) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_math_chemistry_l2184_218446


namespace NUMINAMATH_CALUDE_probability_after_removal_l2184_218423

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : ℕ)
  (cards_per_number : ℕ)

/-- The probability of selecting a pair from a deck -/
def probability_of_pair (d : Deck) : ℚ :=
  sorry

/-- The original deck configuration -/
def original_deck : Deck :=
  { total := 52, numbers := 13, cards_per_number := 4 }

/-- The deck after removing a matching pair -/
def remaining_deck : Deck :=
  { total := 48, numbers := 12, cards_per_number := 4 }

theorem probability_after_removal :
  probability_of_pair remaining_deck = 3 / 47 := by
  sorry

end NUMINAMATH_CALUDE_probability_after_removal_l2184_218423


namespace NUMINAMATH_CALUDE_candy_bar_calories_l2184_218478

theorem candy_bar_calories
  (distance : ℕ) -- Total distance walked
  (calories_per_mile : ℕ) -- Calories burned per mile
  (net_deficit : ℕ) -- Net calorie deficit
  (h1 : distance = 3) -- Cary walks 3 miles round-trip
  (h2 : calories_per_mile = 150) -- Cary burns 150 calories per mile
  (h3 : net_deficit = 250) -- Cary's net calorie deficit is 250 calories
  : distance * calories_per_mile - net_deficit = 200 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_calories_l2184_218478


namespace NUMINAMATH_CALUDE_relationship_abc_l2184_218496

theorem relationship_abc (a b c : ℚ) : 
  (2 * a + a = 1) → (2 * b + b = 2) → (3 * c + c = 2) → a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2184_218496


namespace NUMINAMATH_CALUDE_decorations_used_l2184_218455

theorem decorations_used (boxes : Nat) (decorations_per_box : Nat) (given_away : Nat) : 
  boxes = 4 → decorations_per_box = 15 → given_away = 25 →
  boxes * decorations_per_box - given_away = 35 := by
  sorry

end NUMINAMATH_CALUDE_decorations_used_l2184_218455


namespace NUMINAMATH_CALUDE_smallest_K_for_divisibility_l2184_218474

def repeatedDigit (d : ℕ) (K : ℕ) : ℕ :=
  d * (10^K - 1) / 9

theorem smallest_K_for_divisibility (K : ℕ) : 
  (∀ n : ℕ, n < K → ¬(198 ∣ repeatedDigit 2 n)) ∧ 
  (198 ∣ repeatedDigit 2 K) → 
  K = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_K_for_divisibility_l2184_218474


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l2184_218488

/-- The slope of a chord in an ellipse with midpoint (-2, 1) -/
theorem ellipse_chord_slope :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  (x₁^2 / 16 + y₁^2 / 9 = 1) →
  (x₂^2 / 16 + y₂^2 / 9 = 1) →
  (x₁ + x₂ = -4) →
  (y₁ + y₂ = 2) →
  ((y₂ - y₁) / (x₂ - x₁) = 9 / 8) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l2184_218488


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2184_218461

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 3 * x^2 - 5 * x + 20 = 0 ↔ x = a + b * I ∨ x = a - b * I) → 
  a + b^2 = 245/36 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2184_218461


namespace NUMINAMATH_CALUDE_sum_of_products_bound_l2184_218466

theorem sum_of_products_bound (a b c : ℝ) (h : a + b + c = 1) :
  0 ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_bound_l2184_218466


namespace NUMINAMATH_CALUDE_train_distance_difference_l2184_218413

/-- Proves that the difference in distance traveled by two trains meeting each other is 100 km -/
theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 50) 
  (h2 : v2 = 60)
  (h3 : total_distance = 1100) : 
  (v2 * (total_distance / (v1 + v2))) - (v1 * (total_distance / (v1 + v2))) = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_difference_l2184_218413


namespace NUMINAMATH_CALUDE_pages_left_to_read_l2184_218433

theorem pages_left_to_read (total_pages read_pages : ℕ) 
  (h1 : total_pages = 563)
  (h2 : read_pages = 147) :
  total_pages - read_pages = 416 := by
sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l2184_218433


namespace NUMINAMATH_CALUDE_equation_solution_l2184_218467

theorem equation_solution (m n k x : ℝ) 
  (hm : m ≠ 0) (hn : n ≠ 0) (hk : k ≠ 0) (hmn : m ≠ n) :
  (x + m)^2 - (x + n)^2 = k * (m - n)^2 → 
  x = ((k - 1) * (m + n) - 2 * k * n) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2184_218467


namespace NUMINAMATH_CALUDE_euler_conjecture_counterexample_l2184_218401

theorem euler_conjecture_counterexample : ∃! (n : ℕ), n > 0 ∧ n^5 = 133^5 + 110^5 + 84^5 + 27^5 := by
  sorry

end NUMINAMATH_CALUDE_euler_conjecture_counterexample_l2184_218401


namespace NUMINAMATH_CALUDE_expression_simplification_l2184_218451

theorem expression_simplification :
  let a := 16 / 2015
  let b := 17 / 2016
  (6 + a) * (9 + b) - (3 - a) * (18 - b) - 27 * a = 17 / 224 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2184_218451


namespace NUMINAMATH_CALUDE_circle_C_properties_l2184_218441

/-- Definition of the circle C -/
def circle_C (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = 25}

/-- Theorem stating the properties of circle C and its tangent lines -/
theorem circle_C_properties :
  ∃ (a b : ℝ),
    (a + b + 1 = 0) ∧
    ((-2 - a)^2 + (0 - b)^2 = 25) ∧
    ((5 - a)^2 + (1 - b)^2 = 25) ∧
    (circle_C a b = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 3)^2 = 25}) ∧
    (∀ (x y : ℝ), x = -3 → (x, y) ∈ circle_C a b → y = 0 ∨ y ≠ 0) ∧
    (∀ (x y : ℝ), y = (8/15) * (x + 3) → (x, y) ∈ circle_C a b → x = -3 ∨ x ≠ -3) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_C_properties_l2184_218441


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l2184_218444

theorem distance_from_origin_to_point : 
  let x : ℝ := 3
  let y : ℝ := -4
  Real.sqrt (x^2 + y^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l2184_218444


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l2184_218469

/-- Calculates the expected potato yield from a rectangular garden --/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) 
  (usable_percentage : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_feet := length_steps * step_length
  let width_feet := width_steps * step_length
  let total_area := length_feet * width_feet
  let usable_area := total_area * usable_percentage
  usable_area * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden --/
theorem mr_green_potato_yield :
  expected_potato_yield 18 25 3 0.9 0.5 = 1822.5 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l2184_218469


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2184_218440

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {2, 3, 5}

-- Define set B
def B : Set Nat := {1, 3, 4, 6}

-- Theorem statement
theorem intersection_with_complement : A ∩ (U \ B) = {2, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2184_218440


namespace NUMINAMATH_CALUDE_max_large_chips_l2184_218492

theorem max_large_chips :
  ∀ (small large : ℕ),
  small + large = 72 →
  ∃ (p : ℕ), Prime p ∧ small = large + p →
  large ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_max_large_chips_l2184_218492


namespace NUMINAMATH_CALUDE_cos_four_minus_sin_four_equals_cos_double_l2184_218408

theorem cos_four_minus_sin_four_equals_cos_double (θ : ℝ) :
  Real.cos θ ^ 4 - Real.sin θ ^ 4 = Real.cos (2 * θ) := by
  sorry

end NUMINAMATH_CALUDE_cos_four_minus_sin_four_equals_cos_double_l2184_218408


namespace NUMINAMATH_CALUDE_x_intercepts_count_l2184_218442

theorem x_intercepts_count : Nat.card { k : ℤ | 100 < k * Real.pi ∧ k * Real.pi < 1000 } = 286 := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l2184_218442


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l2184_218430

/-- Given a mixture of pure water and salt solution, prove the amount of salt solution needed. -/
theorem salt_solution_mixture (x : ℝ) : 
  (0.30 * x = 0.20 * (x + 1)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l2184_218430


namespace NUMINAMATH_CALUDE_tournament_divisibility_l2184_218453

theorem tournament_divisibility (n : ℕ) 
  (h1 : ∃ (m : ℕ), (n * (n - 1) / 2 + 2 * n^2 - m) = 5 / 4 * (2 * n * (2 * n - 1) + m)) : 
  9 ∣ (3 * n) := by
sorry

end NUMINAMATH_CALUDE_tournament_divisibility_l2184_218453


namespace NUMINAMATH_CALUDE_commute_time_is_120_minutes_l2184_218445

def minutes_in_hour : ℕ := 60

def rise_time : ℕ := 6 * 60  -- 6:00 a.m. in minutes
def leave_time : ℕ := 7 * 60  -- 7:00 a.m. in minutes
def return_time : ℕ := 17 * 60 + 30  -- 5:30 p.m. in minutes

def num_lectures : ℕ := 8
def lecture_duration : ℕ := 45
def lunch_duration : ℕ := 60
def library_duration : ℕ := 90

def total_time_away : ℕ := return_time - leave_time

def total_college_time : ℕ := num_lectures * lecture_duration + lunch_duration + library_duration

theorem commute_time_is_120_minutes :
  total_time_away - total_college_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_is_120_minutes_l2184_218445


namespace NUMINAMATH_CALUDE_exam_combinations_l2184_218428

/-- The number of compulsory subjects -/
def compulsory_subjects : ℕ := 3

/-- The number of subjects to choose from for the "1" part -/
def choose_one_from : ℕ := 2

/-- The number of subjects to choose from for the "2" part -/
def choose_two_from : ℕ := 4

/-- The number of subjects to be chosen in the "2" part -/
def subjects_to_choose : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The maximum number of different combinations in the "3+1+2" plan -/
def max_combinations : ℕ :=
  1 * choose_one_from * combinations choose_two_from subjects_to_choose

theorem exam_combinations :
  max_combinations = 12 :=
sorry

end NUMINAMATH_CALUDE_exam_combinations_l2184_218428


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_nonnegative_l2184_218484

/-- A function f is increasing on an interval [a, +∞) if for all x, y in the interval with x < y, f(x) < f(y) -/
def IncreasingOnInterval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → f x < f y

/-- The function f(x) = x^2 + 2(a-1)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 3

theorem f_increasing_iff_a_nonnegative :
  ∀ a : ℝ, IncreasingOnInterval (f a) 1 ↔ a ∈ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_nonnegative_l2184_218484


namespace NUMINAMATH_CALUDE_time_to_change_tires_l2184_218480

def minutes_to_wash_car : ℕ := 10
def minutes_to_change_oil : ℕ := 15
def cars_washed : ℕ := 9
def cars_oil_changed : ℕ := 6
def sets_of_tires_changed : ℕ := 2
def hours_worked : ℕ := 4

theorem time_to_change_tires :
  let total_minutes : ℕ := hours_worked * 60
  let washing_time : ℕ := cars_washed * minutes_to_wash_car
  let oil_change_time : ℕ := cars_oil_changed * minutes_to_change_oil
  let remaining_time : ℕ := total_minutes - (washing_time + oil_change_time)
  remaining_time / sets_of_tires_changed = 30 := by sorry

end NUMINAMATH_CALUDE_time_to_change_tires_l2184_218480


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2184_218432

/-- The distance traveled by a car under specific conditions -/
theorem car_distance_theorem (actual_speed : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  actual_speed = 20 →
  speed_increase = 10 →
  time_decrease = 0.5 →
  ∃ (distance : ℝ),
    distance = actual_speed * (distance / actual_speed) ∧
    distance = (actual_speed + speed_increase) * (distance / actual_speed - time_decrease) ∧
    distance = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l2184_218432


namespace NUMINAMATH_CALUDE_odd_function_property_l2184_218409

-- Define the function f on the interval [-1, 1]
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property :
  (∀ x ∈ Set.Icc (-1) 1, f x = f x) →  -- f is defined on [-1, 1]
  isOdd f →  -- f is an odd function
  (∀ x ∈ Set.Ioo 0 1, f x = x * (x - 1)) →  -- f(x) = x(x-1) for 0 < x ≤ 1
  (∀ x ∈ Set.Ioc (-1) 0, f x = -x^2 - x) :=  -- f(x) = -x^2 - x for -1 ≤ x < 0
by sorry

end NUMINAMATH_CALUDE_odd_function_property_l2184_218409


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_times_square_l2184_218448

theorem factorization_cubic_minus_linear_times_square (a b : ℝ) :
  a^3 - a*b^2 = a*(a+b)*(a-b) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_times_square_l2184_218448


namespace NUMINAMATH_CALUDE_remainder_3_88_plus_5_mod_7_l2184_218459

theorem remainder_3_88_plus_5_mod_7 : (3^88 + 5) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_88_plus_5_mod_7_l2184_218459


namespace NUMINAMATH_CALUDE_max_min_sum_of_f_l2184_218471

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) - x) + (3 * Real.exp x + 1) / (Real.exp x + 1)

def domain : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem max_min_sum_of_f :
  ∃ (M N : ℝ), (∀ x ∈ domain, f x ≤ M) ∧
               (∀ x ∈ domain, N ≤ f x) ∧
               (∃ x₁ ∈ domain, f x₁ = M) ∧
               (∃ x₂ ∈ domain, f x₂ = N) ∧
               M + N = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_of_f_l2184_218471


namespace NUMINAMATH_CALUDE_base_ten_satisfies_equation_l2184_218404

/-- Given a base b, converts a number in base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + b * acc) 0

/-- Checks if the equation 253_b + 176_b = 431_b holds for a given base b --/
def equationHolds (b : Nat) : Prop :=
  toDecimal [2, 5, 3] b + toDecimal [1, 7, 6] b = toDecimal [4, 3, 1] b

theorem base_ten_satisfies_equation :
  equationHolds 10 ∧ ∀ b : Nat, b ≠ 10 → ¬equationHolds b :=
sorry

end NUMINAMATH_CALUDE_base_ten_satisfies_equation_l2184_218404


namespace NUMINAMATH_CALUDE_complex_multiplication_l2184_218434

theorem complex_multiplication :
  let i : ℂ := Complex.I
  (1 - 2*i) * (2 + i) = 4 - 3*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2184_218434


namespace NUMINAMATH_CALUDE_equation_solution_l2184_218410

theorem equation_solution (z : ℝ) (hz : z ≠ 0) :
  (5 * z)^10 = (20 * z)^5 ↔ z = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2184_218410


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_necessary_condition_range_l2184_218482

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3/2}
def B (m : ℝ) : Set ℝ := {x | 1-m < x ∧ x ≤ 3*m+1}

theorem intersection_when_m_is_one :
  A ∩ B 1 = {x : ℝ | 0 < x ∧ x ≤ 3/2} := by sorry

theorem necessary_condition_range :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ B m → x ∈ A) ↔ m ≤ 1/6 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_necessary_condition_range_l2184_218482


namespace NUMINAMATH_CALUDE_set_equality_l2184_218477

theorem set_equality : {x : ℕ | x - 1 ≤ 2} = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_set_equality_l2184_218477


namespace NUMINAMATH_CALUDE_bush_leaves_theorem_l2184_218491

theorem bush_leaves_theorem (total_branches : ℕ) (leaves_only : ℕ) (leaves_with_flower : ℕ) : 
  total_branches = 10 →
  leaves_only = 5 →
  leaves_with_flower = 2 →
  ∀ (total_leaves : ℕ),
    (∃ (m n : ℕ), m + n = total_branches ∧ total_leaves = m * leaves_only + n * leaves_with_flower) →
    total_leaves ≠ 45 ∧ total_leaves ≠ 39 ∧ total_leaves ≠ 37 ∧ total_leaves ≠ 31 :=
by sorry

end NUMINAMATH_CALUDE_bush_leaves_theorem_l2184_218491


namespace NUMINAMATH_CALUDE_other_people_in_house_l2184_218421

-- Define the given conditions
def cups_per_person_per_day : ℕ := 2
def ounces_per_cup : ℚ := 1/2
def price_per_ounce : ℚ := 5/4
def weekly_spend : ℚ := 35

-- Define the theorem
theorem other_people_in_house :
  let total_ounces : ℚ := weekly_spend / price_per_ounce
  let ounces_per_person_per_week : ℚ := 7 * cups_per_person_per_day * ounces_per_cup
  let total_people : ℕ := Nat.floor (total_ounces / ounces_per_person_per_week)
  total_people - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_other_people_in_house_l2184_218421


namespace NUMINAMATH_CALUDE_probability_no_same_color_boxes_l2184_218476

/-- Represents a person with 4 colored blocks -/
structure Person :=
  (blocks : Fin 4 → Color)

/-- The four possible colors of blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | Black

/-- Represents a placement of blocks in boxes -/
def Placement := Fin 4 → Fin 4

/-- The probability space of all possible placements -/
def PlacementSpace := Person → Placement

/-- Checks if a box has blocks of all the same color -/
def hasSameColorBlocks (p : PlacementSpace) (box : Fin 4) : Prop :=
  ∃ c : Color, ∀ person : Person, (person.blocks ((p person) box)) = c

/-- The event where no box has blocks of all the same color -/
def NoSameColorBoxes (p : PlacementSpace) : Prop :=
  ∀ box : Fin 4, ¬(hasSameColorBlocks p box)

/-- The probability measure on the placement space -/
noncomputable def P : (PlacementSpace → Prop) → ℝ :=
  sorry

theorem probability_no_same_color_boxes :
  P NoSameColorBoxes = 14811 / 65536 :=
sorry

end NUMINAMATH_CALUDE_probability_no_same_color_boxes_l2184_218476


namespace NUMINAMATH_CALUDE_remainder_mod_five_l2184_218449

theorem remainder_mod_five : (9^6 + 8^8 + 7^9) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_five_l2184_218449


namespace NUMINAMATH_CALUDE_abs_5x_minus_3_not_positive_l2184_218429

theorem abs_5x_minus_3_not_positive (x : ℚ) : 
  ¬(|5*x - 3| > 0) ↔ x = 3/5 := by sorry

end NUMINAMATH_CALUDE_abs_5x_minus_3_not_positive_l2184_218429


namespace NUMINAMATH_CALUDE_johns_age_l2184_218472

/-- Proves that John's current age is 39 years old given the problem conditions -/
theorem johns_age (john_age : ℕ) (james_age : ℕ) (james_brother_age : ℕ) : 
  james_brother_age = 16 →
  james_brother_age = james_age + 4 →
  john_age - 3 = 2 * (james_age + 6) →
  john_age = 39 := by
  sorry

#check johns_age

end NUMINAMATH_CALUDE_johns_age_l2184_218472


namespace NUMINAMATH_CALUDE_symmetric_lines_ellipse_intersection_l2184_218450

/-- Given two lines symmetric about y = x + 1 intersecting an ellipse, 
    prove properties about their slopes and intersection points. -/
theorem symmetric_lines_ellipse_intersection 
  (k : ℝ) 
  (h_k_pos : k > 0) 
  (h_k_neq_one : k ≠ 1) 
  (k₁ : ℝ) 
  (h_symmetric : ∀ x y, y = k * x + 1 ↔ y = k₁ * x + 1) 
  (E : Set (ℝ × ℝ)) 
  (h_E : E = {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}) 
  (A M N : ℝ × ℝ) 
  (h_A : A ∈ E ∧ A.2 = k * A.1 + 1 ∧ A.2 = k₁ * A.1 + 1) 
  (h_M : M ∈ E ∧ M.2 = k * M.1 + 1) 
  (h_N : N ∈ E ∧ N.2 = k₁ * N.1 + 1) : 
  k * k₁ = 1 ∧ 
  ∃ t : ℝ, (1 - t) * M.1 + t * N.1 = 0 ∧ (1 - t) * M.2 + t * N.2 = -5/3 := by
  sorry


end NUMINAMATH_CALUDE_symmetric_lines_ellipse_intersection_l2184_218450


namespace NUMINAMATH_CALUDE_blake_lollipops_l2184_218419

def problem (num_lollipops : ℕ) : Prop :=
  let num_chocolate_packs : ℕ := 6
  let lollipop_price : ℕ := 2
  let chocolate_pack_price : ℕ := 4 * lollipop_price
  let total_paid : ℕ := 6 * 10
  let change : ℕ := 4
  let total_spent : ℕ := total_paid - change
  let chocolate_cost : ℕ := num_chocolate_packs * chocolate_pack_price
  let lollipop_cost : ℕ := total_spent - chocolate_cost
  num_lollipops * lollipop_price = lollipop_cost

theorem blake_lollipops : ∃ (n : ℕ), problem n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_blake_lollipops_l2184_218419


namespace NUMINAMATH_CALUDE_clothes_cost_l2184_218460

def total_spent : ℕ := 8000
def adidas_cost : ℕ := 600

theorem clothes_cost (nike_cost : ℕ) (skechers_cost : ℕ) :
  nike_cost = 3 * adidas_cost →
  skechers_cost = 5 * adidas_cost →
  total_spent - (adidas_cost + nike_cost + skechers_cost) = 2600 :=
by sorry

end NUMINAMATH_CALUDE_clothes_cost_l2184_218460


namespace NUMINAMATH_CALUDE_tv_cost_l2184_218465

def lindas_savings : ℚ := 960

theorem tv_cost (furniture_fraction : ℚ) (h1 : furniture_fraction = 3 / 4) :
  (1 - furniture_fraction) * lindas_savings = 240 := by
  sorry

end NUMINAMATH_CALUDE_tv_cost_l2184_218465


namespace NUMINAMATH_CALUDE_f_5_equals_2015_l2184_218406

/-- Horner's method representation of a polynomial --/
def horner_poly (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 - 2x^4 + x^3 + x^2 - x - 5 --/
def f (x : ℤ) : ℤ := horner_poly [-5, -1, 1, 1, -2, 1] x

theorem f_5_equals_2015 : f 5 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_2015_l2184_218406


namespace NUMINAMATH_CALUDE_simplify_expression_l2184_218475

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  a^2 / (Real.sqrt a * 3 * a^2) = 1 / Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2184_218475


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l2184_218473

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^(n^2) + n
  let r : ℕ := 3^s - 2*s
  r = 3^515 - 1030 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l2184_218473


namespace NUMINAMATH_CALUDE_bertolli_farm_corn_count_l2184_218464

theorem bertolli_farm_corn_count :
  ∀ (tomatoes onions corn : ℕ),
    tomatoes = 2073 →
    onions = 985 →
    tomatoes + corn - onions = 5200 →
    corn = 4039 :=
by
  sorry

end NUMINAMATH_CALUDE_bertolli_farm_corn_count_l2184_218464


namespace NUMINAMATH_CALUDE_smallest_a_value_l2184_218427

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x + π)) : 
  a ≥ 17 ∧ (∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x + π)) → a' ≥ a) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2184_218427


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l2184_218468

theorem seven_digit_divisible_by_11 : ∃ (a g : ℕ), ∃ (b c d e : ℕ),
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ g ∧ g ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  b + c + d + e = 18 ∧
  (a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + 7 * 10 + g) % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l2184_218468


namespace NUMINAMATH_CALUDE_factor_expression_l2184_218462

theorem factor_expression (x : ℝ) : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2184_218462


namespace NUMINAMATH_CALUDE_triangle_properties_l2184_218424

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  -- Condition 1
  (Real.sqrt 2 * c = a * Real.sin C + c * Real.cos A →
    A = Real.pi / 4) ∧
  -- Condition 2
  (Real.sin (B + C) = Real.sqrt 2 - 1 + 2 * (Real.sin (A / 2))^2 →
    A = Real.pi / 4) ∧
  -- Condition 3
  (Real.sqrt 2 * Real.cos (Real.pi / 2 - A) = Real.sin (2 * A) →
    A = Real.pi / 4) ∧
  -- Part 2
  (A = Real.pi / 4 ∧ S = 6 ∧ b = 2 * Real.sqrt 2 →
    a = 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2184_218424


namespace NUMINAMATH_CALUDE_perfume_production_l2184_218443

/-- The number of rose petals required to make an ounce of perfume -/
def petals_per_ounce (petals_per_rose : ℕ) (roses_per_bush : ℕ) (bushes_harvested : ℕ) (bottles : ℕ) (ounces_per_bottle : ℕ) : ℕ :=
  (petals_per_rose * roses_per_bush * bushes_harvested) / (bottles * ounces_per_bottle)

/-- Theorem stating the number of rose petals required to make an ounce of perfume under given conditions -/
theorem perfume_production (petals_per_rose roses_per_bush bushes_harvested bottles ounces_per_bottle : ℕ) 
  (h1 : petals_per_rose = 8)
  (h2 : roses_per_bush = 12)
  (h3 : bushes_harvested = 800)
  (h4 : bottles = 20)
  (h5 : ounces_per_bottle = 12) :
  petals_per_ounce petals_per_rose roses_per_bush bushes_harvested bottles ounces_per_bottle = 320 := by
  sorry

end NUMINAMATH_CALUDE_perfume_production_l2184_218443


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2184_218418

def cost_price : ℝ := 900
def selling_price : ℝ := 1170

theorem gain_percent_calculation : 
  (selling_price - cost_price) / cost_price * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2184_218418


namespace NUMINAMATH_CALUDE_sin6_cos2_integral_l2184_218493

theorem sin6_cos2_integral : ∫ x in (0 : ℝ)..(2 * Real.pi), (Real.sin x)^6 * (Real.cos x)^2 = (5 * Real.pi) / 64 := by
  sorry

end NUMINAMATH_CALUDE_sin6_cos2_integral_l2184_218493


namespace NUMINAMATH_CALUDE_cubic_factorization_l2184_218435

theorem cubic_factorization (m : ℝ) : m^3 - 16*m = m*(m+4)*(m-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2184_218435


namespace NUMINAMATH_CALUDE_conic_parametric_to_cartesian_l2184_218422

theorem conic_parametric_to_cartesian (t : ℝ) (x y : ℝ) :
  x = t^2 + 1/t^2 - 2 ∧ y = t - 1/t → y^2 = x :=
by sorry

end NUMINAMATH_CALUDE_conic_parametric_to_cartesian_l2184_218422


namespace NUMINAMATH_CALUDE_path_width_is_three_l2184_218425

/-- Represents a rectangular garden surrounded by a path of constant width. -/
structure GardenWithPath where
  garden_length : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the perimeter of the garden. -/
def garden_perimeter (g : GardenWithPath) : ℝ :=
  2 * (g.garden_length + g.garden_width)

/-- Calculates the perimeter of the outer edge of the path. -/
def outer_perimeter (g : GardenWithPath) : ℝ :=
  2 * ((g.garden_length + 2 * g.path_width) + (g.garden_width + 2 * g.path_width))

/-- Theorem: If the perimeter of the garden is 24 m shorter than the outer perimeter,
    then the path width is 3 m. -/
theorem path_width_is_three (g : GardenWithPath) :
  outer_perimeter g = garden_perimeter g + 24 → g.path_width = 3 := by
  sorry

#check path_width_is_three

end NUMINAMATH_CALUDE_path_width_is_three_l2184_218425


namespace NUMINAMATH_CALUDE_thirtieth_in_base_five_l2184_218426

def to_base_five (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem thirtieth_in_base_five :
  to_base_five 30 = [1, 1, 0] :=
sorry

end NUMINAMATH_CALUDE_thirtieth_in_base_five_l2184_218426


namespace NUMINAMATH_CALUDE_correct_average_calculation_l2184_218452

theorem correct_average_calculation (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ incorrect_avg = 20 ∧ incorrect_num = 26 ∧ correct_num = 86 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 26 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l2184_218452


namespace NUMINAMATH_CALUDE_tan_two_pi_thirds_l2184_218400

theorem tan_two_pi_thirds : Real.tan (2 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_pi_thirds_l2184_218400


namespace NUMINAMATH_CALUDE_larger_gate_width_l2184_218486

/-- Calculates the width of the larger gate for a rectangular garden. -/
theorem larger_gate_width
  (length : ℝ)
  (width : ℝ)
  (small_gate_width : ℝ)
  (total_fencing : ℝ)
  (h1 : length = 225)
  (h2 : width = 125)
  (h3 : small_gate_width = 3)
  (h4 : total_fencing = 687) :
  2 * (length + width) - (small_gate_width + total_fencing) = 10 :=
by sorry

end NUMINAMATH_CALUDE_larger_gate_width_l2184_218486


namespace NUMINAMATH_CALUDE_expression_simplification_l2184_218415

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2) :
  (a^2 - 1) / (a^2 - a) / (2 + (a^2 + 1) / a) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2184_218415


namespace NUMINAMATH_CALUDE_widget_production_theorem_l2184_218405

/-- Represents the widget production difference between Monday and Tuesday -/
def widget_production_difference (t : ℝ) : ℝ :=
  let w := 3 * t  -- Monday's production rate
  let monday_production := w * t
  let tuesday_production := (w + 5) * (t - 3)
  monday_production - tuesday_production

/-- Theorem stating the widget production difference -/
theorem widget_production_theorem (t : ℝ) :
  widget_production_difference t = 4 * t + 15 := by
  sorry

end NUMINAMATH_CALUDE_widget_production_theorem_l2184_218405


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2184_218495

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 5)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2184_218495


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2184_218479

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) - 1
  f 2 = 0 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2184_218479


namespace NUMINAMATH_CALUDE_parabola_directrix_equation_l2184_218438

/-- Proves that for a parabola y^2 = 2px with focus (2, 0) coinciding with the right focus of the ellipse x^2/9 + y^2/5 = 1, the equation of the directrix is x = -2. -/
theorem parabola_directrix_equation (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x → (2 : ℝ) = p/2) → 
  (∀ x y : ℝ, x^2/9 + y^2/5 = 1 → (2 : ℝ) = Real.sqrt (9 - 5)) → 
  (∀ x : ℝ, x = -p/2 ↔ x = -2) := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_equation_l2184_218438


namespace NUMINAMATH_CALUDE_line_tangent_to_circumcircle_l2184_218454

/-- Represents a line in the form x = my + n -/
structure Line where
  m : ℝ
  n : ℝ
  h : n > 0

/-- Checks if a line passes through a given point -/
def Line.passesThrough (l : Line) (x y : ℝ) : Prop :=
  x = l.m * y + l.n

/-- Represents the feasible region with its circumcircle -/
structure FeasibleRegion where
  diameter : ℝ

/-- Main theorem -/
theorem line_tangent_to_circumcircle (l : Line) (fr : FeasibleRegion) :
  l.passesThrough 4 4 → fr.diameter = 8 → l.n = 4 := by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circumcircle_l2184_218454


namespace NUMINAMATH_CALUDE_lucas_pet_beds_lucas_pet_beds_solution_l2184_218494

theorem lucas_pet_beds (initial_beds : ℕ) (beds_per_pet : ℕ) (pets_capacity : ℕ) : ℕ :=
  let total_beds_needed := pets_capacity * beds_per_pet
  let additional_beds := total_beds_needed - initial_beds
  additional_beds

theorem lucas_pet_beds_solution :
  lucas_pet_beds 12 2 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_lucas_pet_beds_lucas_pet_beds_solution_l2184_218494


namespace NUMINAMATH_CALUDE_inequality_solution_l2184_218490

open Set

theorem inequality_solution (x : ℝ) : 
  (x^2 - 1) / (x^2 - 3*x + 2) ≥ 2 ↔ x ∈ Ioo 1 2 ∪ Ioo (3 - Real.sqrt 6) (3 + Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2184_218490


namespace NUMINAMATH_CALUDE_f_one_equals_phi_l2184_218481

noncomputable section

def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define the properties of function f
def IsValidF (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 0 → y > 0 → x < y → f x < f y) ∧ 
  (∀ x, x > 0 → f x * f (f x + 1/x) = 1)

-- State the theorem
theorem f_one_equals_phi (f : ℝ → ℝ) (h : IsValidF f) : f 1 = φ := by
  sorry

end

end NUMINAMATH_CALUDE_f_one_equals_phi_l2184_218481


namespace NUMINAMATH_CALUDE_cube_divisors_count_l2184_218489

-- Define a natural number with exactly two prime divisors
def has_two_prime_divisors (n : ℕ) : Prop :=
  ∃ p q α β : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p^α * q^β

-- Define the number of divisors function
noncomputable def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- State the theorem
theorem cube_divisors_count
  (n : ℕ)
  (h1 : has_two_prime_divisors n)
  (h2 : num_divisors (n^2) = 35) :
  num_divisors (n^3) = 70 := by
  sorry

end NUMINAMATH_CALUDE_cube_divisors_count_l2184_218489


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2184_218456

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_a4 : a 4 = 5) : 
  a 3 * a 5 = 25 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2184_218456


namespace NUMINAMATH_CALUDE_probability_divisible_by_20_l2184_218411

/-- The set of digits used to form the six-digit number -/
def digits : Finset Nat := {1, 2, 3, 4, 5, 8}

/-- The total number of possible six-digit arrangements -/
def total_arrangements : Nat := 720

/-- Predicate to check if a number is divisible by 20 -/
def is_divisible_by_20 (n : Nat) : Prop := n % 20 = 0

/-- The number of arrangements divisible by 20 -/
def arrangements_divisible_by_20 : Nat := 576

theorem probability_divisible_by_20 :
  (arrangements_divisible_by_20 : ℚ) / total_arrangements = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisible_by_20_l2184_218411


namespace NUMINAMATH_CALUDE_unique_number_l2184_218403

theorem unique_number : ∃! x : ℕ, 
  x > 0 ∧ 
  (∃ k : ℕ, 10 * x + 4 = k * (x + 4)) ∧
  (10 * x + 4) / (x + 4) = x + 4 - 27 ∧
  x = 32 := by
sorry

end NUMINAMATH_CALUDE_unique_number_l2184_218403


namespace NUMINAMATH_CALUDE_sanitizer_theorem_l2184_218439

/-- Represents the prices and quantities of hand sanitizer and disinfectant -/
structure SanitizerProblem where
  x : ℚ  -- Price of hand sanitizer
  y : ℚ  -- Price of 84 disinfectant
  eq1 : 100 * x + 150 * y = 1500
  eq2 : 120 * x + 160 * y = 1720
  promotion : ℕ → ℕ
  promotion_def : ∀ n : ℕ, promotion n = n / 150 * 10

/-- The solution to the sanitizer problem -/
def sanitizer_solution (p : SanitizerProblem) : Prop :=
  p.x = 9 ∧ p.y = 4 ∧ 
  9 * 150 + 4 * (60 - p.promotion 150) = 1550

/-- The main theorem stating that the solution is correct -/
theorem sanitizer_theorem (p : SanitizerProblem) : sanitizer_solution p := by
  sorry

end NUMINAMATH_CALUDE_sanitizer_theorem_l2184_218439


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l2184_218414

theorem final_sum_after_operations (a b S : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry


end NUMINAMATH_CALUDE_final_sum_after_operations_l2184_218414


namespace NUMINAMATH_CALUDE_base_seven_sum_of_digits_product_l2184_218412

def to_decimal (n : ℕ) (base : ℕ) : ℕ := sorry

def from_decimal (n : ℕ) (base : ℕ) : ℕ := sorry

def add_base (a b base : ℕ) : ℕ := 
  from_decimal (to_decimal a base + to_decimal b base) base

def mult_base (a b base : ℕ) : ℕ := 
  from_decimal (to_decimal a base * to_decimal b base) base

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem base_seven_sum_of_digits_product : 
  let base := 7
  let a := 35
  let b := add_base 12 16 base
  let product := mult_base a b base
  sum_of_digits product = 7 := by sorry

end NUMINAMATH_CALUDE_base_seven_sum_of_digits_product_l2184_218412


namespace NUMINAMATH_CALUDE_convex_ngon_non_acute_side_l2184_218457

/-- A convex n-gon is a polygon with n sides and n vertices, where all internal angles are less than 180 degrees. -/
def ConvexNGon (n : ℕ) : Type := sorry

/-- An angle is acute if it is less than 90 degrees. -/
def IsAcute (angle : ℝ) : Prop := angle < 90

/-- Given a convex n-gon and a side, returns the two angles at the endpoints of that side. -/
def EndpointAngles (polygon : ConvexNGon n) (side : Fin n) : ℝ × ℝ := sorry

theorem convex_ngon_non_acute_side (n : ℕ) (hn : n ≥ 7) :
  ∀ (polygon : ConvexNGon n), ∃ (side : Fin n),
    let (angle1, angle2) := EndpointAngles polygon side
    ¬(IsAcute angle1 ∨ IsAcute angle2) :=
sorry

end NUMINAMATH_CALUDE_convex_ngon_non_acute_side_l2184_218457


namespace NUMINAMATH_CALUDE_distance_difference_l2184_218407

/-- Given distances between locations, prove the difference in total distances -/
theorem distance_difference (orchard_to_house house_to_pharmacy pharmacy_to_school : ℕ) 
  (h1 : orchard_to_house = 800)
  (h2 : house_to_pharmacy = 1300)
  (h3 : pharmacy_to_school = 1700) :
  (orchard_to_house + house_to_pharmacy) - pharmacy_to_school = 400 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l2184_218407


namespace NUMINAMATH_CALUDE_range_of_k_l2184_218470

/-- Represents an ellipse equation -/
def is_ellipse (k : ℝ) : Prop :=
  2 * k - 1 > 0 ∧ k - 1 > 0

/-- Represents a hyperbola equation -/
def is_hyperbola (k : ℝ) : Prop :=
  (4 - k) * (k - 3) < 0

/-- The main theorem stating the range of k -/
theorem range_of_k :
  (∀ k : ℝ, (is_ellipse k ∨ is_hyperbola k) ∧ ¬(is_ellipse k ∧ is_hyperbola k)) →
  (∀ k : ℝ, k ≤ 1 ∨ (3 ≤ k ∧ k ≤ 4)) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l2184_218470


namespace NUMINAMATH_CALUDE_simplify_expression_l2184_218487

theorem simplify_expression : (27 * (10 ^ 12)) / (9 * (10 ^ 4)) = 300000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2184_218487


namespace NUMINAMATH_CALUDE_sin_decreasing_interval_l2184_218483

theorem sin_decreasing_interval :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2),
    ∀ y ∈ Set.Icc (π / 2) (3 * π / 2),
      x ≤ y → Real.sin x ≥ Real.sin y :=
by sorry

end NUMINAMATH_CALUDE_sin_decreasing_interval_l2184_218483


namespace NUMINAMATH_CALUDE_vector_sum_proof_l2184_218416

/-- Given two vectors a and b in ℝ², prove that their sum is (-1, 5) -/
theorem vector_sum_proof (a b : ℝ × ℝ) (ha : a = (2, 1)) (hb : b = (-3, 4)) :
  a + b = (-1, 5) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l2184_218416
