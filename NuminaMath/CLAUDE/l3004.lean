import Mathlib

namespace NUMINAMATH_CALUDE_staircase_steps_l3004_300428

def jumps (step_size : ℕ) (total_steps : ℕ) : ℕ :=
  (total_steps + step_size - 1) / step_size

theorem staircase_steps : ∃ (n : ℕ), n > 0 ∧ jumps 3 n - jumps 4 n = 10 ∧ n = 120 := by
  sorry

end NUMINAMATH_CALUDE_staircase_steps_l3004_300428


namespace NUMINAMATH_CALUDE_bijective_function_theorem_l3004_300415

theorem bijective_function_theorem (a : ℝ) :
  (∃ f : ℝ → ℝ, Function.Bijective f ∧
    (∀ x : ℝ, f (f x) = x^2 * f x + a * x^2)) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_bijective_function_theorem_l3004_300415


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l3004_300457

theorem max_product_sum_2000 : 
  ∃ (a b : ℤ), a + b = 2000 ∧ 
    ∀ (x y : ℤ), x + y = 2000 → x * y ≤ a * b ∧ 
    a * b = 1000000 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l3004_300457


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l3004_300489

theorem circle_equation_k_value (x y k : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 4*y - k = 0 ↔ (x + 4)^2 + (y + 2)^2 = 64) → 
  k = 44 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l3004_300489


namespace NUMINAMATH_CALUDE_mars_mission_cost_share_l3004_300430

-- Define the given conditions
def cost_in_euros : ℝ := 25e9
def number_of_people : ℝ := 300e6
def exchange_rate : ℝ := 1.2

-- Define the theorem to prove
theorem mars_mission_cost_share :
  (cost_in_euros * exchange_rate) / number_of_people = 100 := by
  sorry

end NUMINAMATH_CALUDE_mars_mission_cost_share_l3004_300430


namespace NUMINAMATH_CALUDE_smallest_perimeter_l3004_300409

/-- A triangle with consecutive integer side lengths, where the smallest side is greater than 2 -/
structure ConsecutiveIntegerTriangle where
  n : ℕ
  gt_two : n > 2

/-- The perimeter of a ConsecutiveIntegerTriangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ := t.n + (t.n + 1) + (t.n + 2)

/-- Predicate to check if a ConsecutiveIntegerTriangle is valid (satisfies triangle inequality) -/
def is_valid_triangle (t : ConsecutiveIntegerTriangle) : Prop :=
  t.n + (t.n + 1) > t.n + 2 ∧
  t.n + (t.n + 2) > t.n + 1 ∧
  (t.n + 1) + (t.n + 2) > t.n

theorem smallest_perimeter :
  ∃ (t : ConsecutiveIntegerTriangle), 
    is_valid_triangle t ∧ 
    perimeter t = 12 ∧ 
    (∀ (t' : ConsecutiveIntegerTriangle), is_valid_triangle t' → perimeter t' ≥ 12) :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l3004_300409


namespace NUMINAMATH_CALUDE_claire_photos_l3004_300454

theorem claire_photos (c : ℕ) 
  (h1 : 3 * c = c + 10) : c = 5 := by
  sorry

#check claire_photos

end NUMINAMATH_CALUDE_claire_photos_l3004_300454


namespace NUMINAMATH_CALUDE_max_sum_after_swap_l3004_300482

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Swaps the first and last digits of a ThreeDigitNumber -/
def ThreeDigitNumber.swap (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.tens
  ones := n.hundreds
  is_valid := by sorry

/-- The main theorem to prove -/
theorem max_sum_after_swap
  (a b c : ThreeDigitNumber)
  (h : a.toNat + b.toNat + c.toNat = 2019) :
  (a.swap.toNat + b.swap.toNat + c.swap.toNat) ≤ 2118 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_after_swap_l3004_300482


namespace NUMINAMATH_CALUDE_race_heartbeats_l3004_300469

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

theorem race_heartbeats :
  let heart_rate : ℕ := 140  -- heartbeats per minute
  let race_distance : ℕ := 30  -- miles
  let pace : ℕ := 6  -- minutes per mile
  total_heartbeats heart_rate race_distance pace = 25200 := by
  sorry

end NUMINAMATH_CALUDE_race_heartbeats_l3004_300469


namespace NUMINAMATH_CALUDE_square_sum_and_reciprocal_square_l3004_300495

theorem square_sum_and_reciprocal_square (x : ℝ) (h : x + 2/x = 6) :
  x^2 + 4/x^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_reciprocal_square_l3004_300495


namespace NUMINAMATH_CALUDE_smallest_value_for_x_between_zero_and_one_l3004_300486

theorem smallest_value_for_x_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x < 1) :
  x^2 < min x (min (2*x) (min (Real.sqrt x) (1/x))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_between_zero_and_one_l3004_300486


namespace NUMINAMATH_CALUDE_distance_product_range_l3004_300477

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 8

-- Define the point P on C₁
def P (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

-- Define the line l passing through P with slope 1
def l (t x : ℝ) : ℝ := x + 2*t - t^2

-- Define the product of distances |PQ||PR|
def distance_product (t : ℝ) : ℝ := (t^2 - 2)^2 + 4

-- Main theorem
theorem distance_product_range :
  ∀ t : ℝ, C₁ (P t).1 (P t).2 →
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
      C₂ x₁ (l t x₁) ∧ C₂ x₂ (l t x₂)) →
    distance_product t ∈ Set.Icc 4 8 ∪ Set.Ioo 8 36 :=
by sorry

end NUMINAMATH_CALUDE_distance_product_range_l3004_300477


namespace NUMINAMATH_CALUDE_vegetable_seedling_price_l3004_300445

theorem vegetable_seedling_price (base_price : ℚ) : 
  (300 / base_price - 300 / (5/4 * base_price) = 3) → base_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_seedling_price_l3004_300445


namespace NUMINAMATH_CALUDE_range_of_x₁_l3004_300494

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the inequality condition
def InequalityCondition (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ + x₂ = 1 → f x₁ + f 0 > f x₂ + f 1

-- Theorem statement
theorem range_of_x₁ (h₁ : IsIncreasing f) (h₂ : InequalityCondition f) :
  ∀ x₁, (∃ x₂, x₁ + x₂ = 1 ∧ f x₁ + f 0 > f x₂ + f 1) → x₁ > 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_x₁_l3004_300494


namespace NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_factors_l3004_300474

theorem larger_number_given_hcf_lcm_factors (a b : ℕ) : 
  a > 0 → b > 0 → 
  Nat.gcd a b = 10 → 
  Nat.lcm a b = 10 * 11 * 15 → 
  max a b = 150 := by
sorry

end NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_factors_l3004_300474


namespace NUMINAMATH_CALUDE_cake_muffin_probability_l3004_300419

theorem cake_muffin_probability (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) 
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_both : both = 16) :
  (total - (cake + muffin - both)) / total = 26 / 100 := by
sorry

end NUMINAMATH_CALUDE_cake_muffin_probability_l3004_300419


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3004_300465

/-- Given a hyperbola with asymptote equation 4x - 3y = 0 and sharing foci with the ellipse x²/30 + y²/5 = 1, 
    its equation is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ k : ℝ, 4 * x - 3 * y = k) ∧ 
  (∃ c : ℝ, c > 0 ∧ c^2 = 25 ∧ (∀ x y : ℝ, x^2 / 30 + y^2 / 5 = 1 → x^2 ≤ c^2)) →
  (x^2 / 9 - y^2 / 16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3004_300465


namespace NUMINAMATH_CALUDE_teds_age_l3004_300435

theorem teds_age (s t j : ℕ) : 
  t = 2 * s - 20 →
  j = s + 6 →
  t + s + j = 90 →
  t = 32 := by
  sorry

end NUMINAMATH_CALUDE_teds_age_l3004_300435


namespace NUMINAMATH_CALUDE_point_comparison_l3004_300443

/-- Given that points (-2, y₁) and (-1, y₂) lie on the line y = -3x + b, prove that y₁ > y₂ -/
theorem point_comparison (b : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : y₁ = -3 * (-2) + b) 
  (h₂ : y₂ = -3 * (-1) + b) : 
  y₁ > y₂ := by
  sorry


end NUMINAMATH_CALUDE_point_comparison_l3004_300443


namespace NUMINAMATH_CALUDE_smallest_shift_l3004_300458

-- Define a function that repeats every 15 units horizontally
def is_periodic_15 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 15) = f x

-- Define the property we're looking for
def satisfies_shift (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f ((x - b) / 3) = f (x / 3)

-- State the theorem
theorem smallest_shift (f : ℝ → ℝ) (h : is_periodic_15 f) :
  ∃ b : ℝ, b > 0 ∧ satisfies_shift f b ∧ ∀ b' : ℝ, b' > 0 ∧ satisfies_shift f b' → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l3004_300458


namespace NUMINAMATH_CALUDE_man_business_ownership_l3004_300446

theorem man_business_ownership (total_value : ℝ) (sold_value : ℝ) (sold_fraction : ℝ) :
  total_value = 150000 →
  sold_value = 75000 →
  sold_fraction = 3/4 →
  ∃ original_fraction : ℝ,
    original_fraction * total_value * sold_fraction = sold_value ∧
    original_fraction = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_man_business_ownership_l3004_300446


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l3004_300423

/-- Two-digit integer -/
def TwoDigitInt (z : ℕ) : Prop := 10 ≤ z ∧ z ≤ 99

/-- Reverse digits of a two-digit number -/
def reverseDigits (x : ℕ) : ℕ := 10 * (x % 10) + (x / 10)

theorem two_digit_reverse_sum (x y n : ℕ) : 
  TwoDigitInt x → TwoDigitInt y → y = reverseDigits x → x^2 - y^2 = n^2 → x + y + n = 154 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l3004_300423


namespace NUMINAMATH_CALUDE_science_fair_students_l3004_300439

theorem science_fair_students (know_it_all : ℕ) (karen : ℕ) (novel_corona : ℕ) (total : ℕ) :
  know_it_all = 50 →
  karen = 3 * know_it_all / 5 →
  total = 240 →
  total = know_it_all + karen + novel_corona →
  novel_corona = 160 := by
sorry

end NUMINAMATH_CALUDE_science_fair_students_l3004_300439


namespace NUMINAMATH_CALUDE_unfair_coin_flip_probability_l3004_300452

/-- The probability of getting heads in a single flip of the unfair coin -/
def p_heads : ℚ := 1/3

/-- The probability of getting tails in a single flip of the unfair coin -/
def p_tails : ℚ := 2/3

/-- The number of coin flips -/
def n : ℕ := 10

/-- The number of heads we want to get -/
def k : ℕ := 3

/-- The probability of getting exactly k heads in n flips of the unfair coin -/
def prob_k_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem unfair_coin_flip_probability :
  prob_k_heads n k p_heads = 512/1969 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_flip_probability_l3004_300452


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l3004_300471

theorem max_value_of_exponential_difference (x : ℝ) : 5^x - 25^x ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l3004_300471


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l3004_300422

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.foldr (fun bit acc => 2 * acc + bit) 0

theorem binary_101_equals_5 : binary_to_decimal [1, 0, 1] = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l3004_300422


namespace NUMINAMATH_CALUDE_expression_simplification_l3004_300444

theorem expression_simplification (x y : ℝ) :
  (1/2) * x - 2 * (x - (1/3) * y^2) + (-3/2 * x + (1/3) * y^2) = -3 * x + y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3004_300444


namespace NUMINAMATH_CALUDE_remaining_two_average_l3004_300400

theorem remaining_two_average (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) :
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 4.60 →
  (n₁ + n₂) / 2 = 3.4 →
  (n₃ + n₄) / 2 = 3.8 →
  (n₅ + n₆) / 2 = 6.6 :=
by sorry

end NUMINAMATH_CALUDE_remaining_two_average_l3004_300400


namespace NUMINAMATH_CALUDE_evaluate_expression_l3004_300467

theorem evaluate_expression : (16^24) / (32^12) = 8^12 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3004_300467


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3004_300499

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (3 - 2 * i) / (1 + 4 * i) = -5/17 - 14/17 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3004_300499


namespace NUMINAMATH_CALUDE_symmetric_point_to_origin_l3004_300414

/-- If |a-3|+(b+4)^2=0, then the point (a,b) is (3,-4) and its symmetric point to the origin is (-3,4) -/
theorem symmetric_point_to_origin (a b : ℝ) : 
  (|a - 3| + (b + 4)^2 = 0) → 
  (a = 3 ∧ b = -4) ∧ 
  ((-a, -b) = (-3, 4)) := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_to_origin_l3004_300414


namespace NUMINAMATH_CALUDE_perimeter_comparison_l3004_300431

-- Define a structure for rectangular parallelepiped
structure RectangularParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  positive_dimensions : 0 < length ∧ 0 < width ∧ 0 < height

-- Define a function to calculate the perimeter of a rectangular parallelepiped
def perimeter (p : RectangularParallelepiped) : ℝ :=
  4 * (p.length + p.width + p.height)

-- Define what it means for one parallelepiped to be contained within another
def contained_within (p q : RectangularParallelepiped) : Prop :=
  p.length ≤ q.length ∧ p.width ≤ q.width ∧ p.height ≤ q.height

-- Theorem statement
theorem perimeter_comparison 
  (p q : RectangularParallelepiped) 
  (h : contained_within p q) : 
  perimeter p ≤ perimeter q :=
sorry

end NUMINAMATH_CALUDE_perimeter_comparison_l3004_300431


namespace NUMINAMATH_CALUDE_coprime_linear_combination_l3004_300412

theorem coprime_linear_combination (m n : ℕ+) (h : Nat.Coprime m n) :
  ∃ N : ℕ, ∀ k : ℕ, k ≥ N → ∃ a b : ℕ, k = a * m + b * n ∧
  (∀ N' : ℕ, (∀ k : ℕ, k ≥ N' → ∃ a b : ℕ, k = a * m + b * n) → N' ≥ N) ∧
  N = m * n - m - n + 1 :=
sorry

end NUMINAMATH_CALUDE_coprime_linear_combination_l3004_300412


namespace NUMINAMATH_CALUDE_isosceles_minimizes_side_l3004_300411

/-- Represents a triangle with sides a, b, c and angle α opposite to side a -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  area : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧ area > 0
  h_angle : α < π
  h_area : area = (1/2) * b * c * Real.sin α

/-- Given a fixed angle α and area S, the triangle that minimizes side a is isosceles with b = c -/
theorem isosceles_minimizes_side (α S : ℝ) (h_α : 0 < α ∧ α < π) (h_S : S > 0) :
  ∃ (t : Triangle), t.α = α ∧ t.area = S ∧ t.b = t.c ∧
  ∀ (u : Triangle), u.α = α → u.area = S → t.a ≤ u.a :=
sorry

end NUMINAMATH_CALUDE_isosceles_minimizes_side_l3004_300411


namespace NUMINAMATH_CALUDE_janet_bird_count_l3004_300438

theorem janet_bird_count (crows hawks : ℕ) : 
  hawks = crows + (crows * 6 / 10) →
  crows + hawks = 78 →
  crows = 30 := by
sorry

end NUMINAMATH_CALUDE_janet_bird_count_l3004_300438


namespace NUMINAMATH_CALUDE_f_fixed_points_l3004_300466

def f (x : ℝ) : ℝ := x^3 - 4*x

theorem f_fixed_points (x : ℝ) : 
  (x = 0 ∨ x = 2 ∨ x = -2) → f (f x) = f x :=
by
  sorry

end NUMINAMATH_CALUDE_f_fixed_points_l3004_300466


namespace NUMINAMATH_CALUDE_trig_simplification_l3004_300488

theorem trig_simplification :
  let x : Real := 40 * π / 180
  let y : Real := 50 * π / 180
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) / (Real.cos x - Real.sqrt (1 - Real.sin y ^ 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3004_300488


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3004_300434

theorem min_value_quadratic (x : ℝ) :
  ∃ (m : ℝ), m = 1438 ∧ ∀ x, 3 * x^2 - 12 * x + 1450 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3004_300434


namespace NUMINAMATH_CALUDE_no_valid_ab_pairs_l3004_300492

theorem no_valid_ab_pairs : 
  ¬∃ (a b : ℝ), ∃ (x y : ℤ), 
    (3 * a * x + 7 * b * y = 3) ∧ 
    (x^2 + y^2 = 85) ∧ 
    (x % 5 = 0 ∨ y % 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_no_valid_ab_pairs_l3004_300492


namespace NUMINAMATH_CALUDE_fraction_product_l3004_300449

theorem fraction_product : (2 : ℚ) / 9 * (4 : ℚ) / 5 = (8 : ℚ) / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3004_300449


namespace NUMINAMATH_CALUDE_two_digit_triple_reverse_difference_l3004_300464

theorem two_digit_triple_reverse_difference (A B : ℕ) : 
  A ≠ 0 → 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  2 ∣ ((30 * B + A) - (10 * B + A)) := by
sorry

end NUMINAMATH_CALUDE_two_digit_triple_reverse_difference_l3004_300464


namespace NUMINAMATH_CALUDE_complex_magnitude_l3004_300484

theorem complex_magnitude (z : ℂ) (h : (1 + 2*I)*z = -3 + 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3004_300484


namespace NUMINAMATH_CALUDE_tetrahedron_intersection_theorem_l3004_300481

/-- Represents a tetrahedron with an inscribed sphere -/
structure TetrahedronWithSphere where
  volume : ℝ
  surface_area : ℝ
  inscribed_sphere_radius : ℝ

/-- Represents a plane intersecting three edges of a tetrahedron -/
structure IntersectingPlane where
  passes_through_center : Bool

/-- Represents the parts of the tetrahedron created by the intersecting plane -/
structure TetrahedronParts where
  volume_ratio : ℝ
  surface_area_ratio : ℝ

/-- The main theorem statement -/
theorem tetrahedron_intersection_theorem 
  (t : TetrahedronWithSphere) 
  (p : IntersectingPlane) 
  (parts : TetrahedronParts) : 
  (parts.volume_ratio = parts.surface_area_ratio) ↔ p.passes_through_center := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_intersection_theorem_l3004_300481


namespace NUMINAMATH_CALUDE_bouquet_cost_45_lilies_l3004_300429

/-- Represents the cost of a bouquet of lilies -/
def bouquet_cost (num_lilies : ℕ) : ℚ :=
  let base_price_per_lily : ℚ := 30 / 15
  let discount_threshold : ℕ := 30
  let discount_rate : ℚ := 1 / 10
  if num_lilies ≤ discount_threshold then
    num_lilies * base_price_per_lily
  else
    num_lilies * (base_price_per_lily * (1 - discount_rate))

theorem bouquet_cost_45_lilies :
  bouquet_cost 45 = 81 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_45_lilies_l3004_300429


namespace NUMINAMATH_CALUDE_plan_C_not_more_expensive_l3004_300480

/-- Represents the number of days required for Team A to complete the project alone -/
def x : ℕ := sorry

/-- Cost per day for Team A -/
def cost_A : ℕ := 10000

/-- Cost per day for Team B -/
def cost_B : ℕ := 6000

/-- Number of days both teams work together in Plan C -/
def days_together : ℕ := 3

/-- Extra days required for Team B to complete the project alone -/
def extra_days_B : ℕ := 4

/-- Equation representing the work done in Plan C -/
axiom plan_C_equation : (days_together : ℝ) / x + x / (x + extra_days_B) = 1

/-- Cost of Plan A -/
def cost_plan_A : ℕ := x * cost_A

/-- Cost of Plan C -/
def cost_plan_C : ℕ := days_together * (cost_A + cost_B) + (x - days_together) * cost_B

/-- Theorem stating that Plan C is not more expensive than Plan A -/
theorem plan_C_not_more_expensive : cost_plan_C ≤ cost_plan_A := by sorry

end NUMINAMATH_CALUDE_plan_C_not_more_expensive_l3004_300480


namespace NUMINAMATH_CALUDE_determinant_inequality_solution_l3004_300485

-- Define the determinant of a 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the solution set
def solution_set : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem determinant_inequality_solution :
  {x : ℝ | det 1 2 x (x^2) < 3} = solution_set :=
sorry

end NUMINAMATH_CALUDE_determinant_inequality_solution_l3004_300485


namespace NUMINAMATH_CALUDE_opposite_of_two_minus_sqrt_five_l3004_300455

theorem opposite_of_two_minus_sqrt_five :
  -(2 - Real.sqrt 5) = Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_two_minus_sqrt_five_l3004_300455


namespace NUMINAMATH_CALUDE_c_rent_share_is_27_l3004_300475

/-- Represents the rental information for a person --/
structure RentalInfo where
  oxen : ℕ
  months : ℕ

/-- Calculates the total rent share for a person --/
def calculateRentShare (totalRent : ℚ) (totalOxenMonths : ℕ) (info : RentalInfo) : ℚ :=
  totalRent * (info.oxen * info.months : ℚ) / totalOxenMonths

theorem c_rent_share_is_27 
  (a b c : RentalInfo)
  (h_a : a = ⟨10, 7⟩)
  (h_b : b = ⟨12, 5⟩)
  (h_c : c = ⟨15, 3⟩)
  (h_total_rent : totalRent = 105)
  (h_total_oxen_months : totalOxenMonths = a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) :
  calculateRentShare totalRent totalOxenMonths c = 27 := by
  sorry


end NUMINAMATH_CALUDE_c_rent_share_is_27_l3004_300475


namespace NUMINAMATH_CALUDE_bacteria_after_three_hours_l3004_300403

/-- Represents the number of bacteria after a given time -/
def bacteria_count (initial_count : ℕ) (split_interval : ℕ) (total_time : ℕ) : ℕ :=
  initial_count * 2 ^ (total_time / split_interval)

/-- Theorem stating that the number of bacteria after 3 hours is 64 -/
theorem bacteria_after_three_hours :
  bacteria_count 1 30 180 = 64 := by
  sorry

#check bacteria_after_three_hours

end NUMINAMATH_CALUDE_bacteria_after_three_hours_l3004_300403


namespace NUMINAMATH_CALUDE_book_arrangement_l3004_300436

theorem book_arrangement (n m : ℕ) (h : n + m = 8) :
  Nat.choose 8 n = 56 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_l3004_300436


namespace NUMINAMATH_CALUDE_total_unique_eagle_types_l3004_300442

/-- The number of unique types of eagles across all sections -/
def uniqueEagleTypes (sectionA sectionB sectionC sectionD sectionE : ℝ)
  (overlapAB overlapBC overlapCD overlapDE overlapACE : ℝ) : ℝ :=
  sectionA + sectionB + sectionC + sectionD + sectionE - 
  (overlapAB + overlapBC + overlapCD + overlapDE - overlapACE)

/-- Theorem stating the total number of unique eagle types -/
theorem total_unique_eagle_types :
  uniqueEagleTypes 12.5 8.3 10.7 14.2 17.1 3.5 2.1 3.7 4.4 1.5 = 51.6 := by
  sorry

end NUMINAMATH_CALUDE_total_unique_eagle_types_l3004_300442


namespace NUMINAMATH_CALUDE_product_from_lcm_hcf_l3004_300417

theorem product_from_lcm_hcf (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750) 
  (h_hcf : Nat.gcd a b = 25) : 
  a * b = 18750 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_hcf_l3004_300417


namespace NUMINAMATH_CALUDE_village_population_l3004_300450

theorem village_population (partial_population : ℕ) (partial_percentage : ℚ) (total_population : ℕ) :
  partial_percentage = 9/10 →
  partial_population = 36000 →
  total_population * partial_percentage = partial_population →
  total_population = 40000 := by
sorry

end NUMINAMATH_CALUDE_village_population_l3004_300450


namespace NUMINAMATH_CALUDE_irrational_pi_only_l3004_300406

theorem irrational_pi_only (a b c d : ℝ) : 
  a = 1 / 7 → b = Real.pi → c = -1 → d = 0 → 
  (¬ Irrational a ∧ Irrational b ∧ ¬ Irrational c ∧ ¬ Irrational d) := by
  sorry

end NUMINAMATH_CALUDE_irrational_pi_only_l3004_300406


namespace NUMINAMATH_CALUDE_cost_per_item_proof_l3004_300476

/-- The cost per item in the first batch of fruits -/
def cost_per_item_first_batch : ℝ := 120

/-- The total cost of the first batch of fruits -/
def total_cost_first_batch : ℝ := 600

/-- The total cost of the second batch of fruits -/
def total_cost_second_batch : ℝ := 1250

/-- The number of items in the second batch is twice the number in the first batch -/
axiom double_items : ∃ n : ℝ, n * cost_per_item_first_batch = total_cost_first_batch ∧
                               2 * n * (cost_per_item_first_batch + 5) = total_cost_second_batch

theorem cost_per_item_proof : 
  cost_per_item_first_batch = 120 :=
sorry

end NUMINAMATH_CALUDE_cost_per_item_proof_l3004_300476


namespace NUMINAMATH_CALUDE_cheerleaders_who_quit_l3004_300441

theorem cheerleaders_who_quit 
  (initial_football_players : Nat) 
  (initial_cheerleaders : Nat)
  (football_players_quit : Nat)
  (total_left : Nat)
  (h1 : initial_football_players = 13)
  (h2 : initial_cheerleaders = 16)
  (h3 : football_players_quit = 10)
  (h4 : total_left = 15)
  (h5 : initial_football_players - football_players_quit + initial_cheerleaders - cheerleaders_quit = total_left)
  : cheerleaders_quit = 4 :=
by
  sorry

#check cheerleaders_who_quit

end NUMINAMATH_CALUDE_cheerleaders_who_quit_l3004_300441


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_geq_one_l3004_300487

theorem quadratic_inequality_implies_a_geq_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_geq_one_l3004_300487


namespace NUMINAMATH_CALUDE_f_geq_6_iff_l3004_300402

def f (x : ℝ) := |x + 1| + |2*x - 4|

theorem f_geq_6_iff (x : ℝ) : f x ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_f_geq_6_iff_l3004_300402


namespace NUMINAMATH_CALUDE_second_number_problem_l3004_300427

theorem second_number_problem (A B C : ℝ) (h_sum : A + B + C = 98) 
  (h_ratio1 : A / B = 2 / 3) (h_ratio2 : B / C = 5 / 8) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) : 
  B = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_problem_l3004_300427


namespace NUMINAMATH_CALUDE_only_negative_three_squared_positive_l3004_300404

theorem only_negative_three_squared_positive :
  let a := 0 * ((-2019) ^ 2018)
  let b := (-3) ^ 2
  let c := -2 / ((-3) ^ 4)
  let d := (-2) ^ 3
  (a ≤ 0 ∧ b > 0 ∧ c < 0 ∧ d < 0) := by sorry

end NUMINAMATH_CALUDE_only_negative_three_squared_positive_l3004_300404


namespace NUMINAMATH_CALUDE_prime_factor_sum_l3004_300479

theorem prime_factor_sum (w x y z k : ℕ) :
  2^w * 3^x * 5^y * 7^z * 11^k = 2520 →
  2*w + 3*x + 5*y + 7*z + 11*k = 24 := by
  sorry

end NUMINAMATH_CALUDE_prime_factor_sum_l3004_300479


namespace NUMINAMATH_CALUDE_hyperbola_parabola_focus_l3004_300473

theorem hyperbola_parabola_focus (a : ℝ) (h1 : a > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/3 = 1) ∧ 
  (∃ (x : ℝ), (2, 0) = (x, 0) ∧ x^2/a^2 - 0^2/3 = 1) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_focus_l3004_300473


namespace NUMINAMATH_CALUDE_sequence_value_proof_l3004_300478

def sequence_sum (n : ℕ) : ℤ := n^2 - 9*n

def sequence_term (n : ℕ) : ℤ := sequence_sum n - sequence_sum (n-1)

theorem sequence_value_proof (k : ℕ) (h : 5 < sequence_term k ∧ sequence_term k < 8) :
  sequence_term k = 6 := by sorry

end NUMINAMATH_CALUDE_sequence_value_proof_l3004_300478


namespace NUMINAMATH_CALUDE_lexie_family_age_ratio_l3004_300437

/-- Proves that given the age relationships in Lexie's family, the ratio of her sister's age to Lexie's age is 2. -/
theorem lexie_family_age_ratio :
  ∀ (lexie_age brother_age sister_age : ℕ),
    lexie_age = 8 →
    lexie_age = brother_age + 6 →
    sister_age - brother_age = 14 →
    ∃ (k : ℕ), sister_age = k * lexie_age →
    sister_age / lexie_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_lexie_family_age_ratio_l3004_300437


namespace NUMINAMATH_CALUDE_man_rowing_speed_l3004_300451

/-- Proves that given a man's speed in still water and his speed rowing downstream,
    his speed rowing upstream can be calculated. -/
theorem man_rowing_speed
  (speed_still : ℝ)
  (speed_downstream : ℝ)
  (h_still : speed_still = 30)
  (h_downstream : speed_downstream = 35) :
  speed_still - (speed_downstream - speed_still) = 25 :=
by sorry

end NUMINAMATH_CALUDE_man_rowing_speed_l3004_300451


namespace NUMINAMATH_CALUDE_boy_squirrel_walnuts_l3004_300491

theorem boy_squirrel_walnuts (initial_walnuts : ℕ) (boy_gathered : ℕ) (girl_brought : ℕ) (girl_ate : ℕ) (final_walnuts : ℕ) :
  initial_walnuts = 12 →
  girl_brought = 5 →
  girl_ate = 2 →
  final_walnuts = 20 →
  final_walnuts = initial_walnuts + boy_gathered - 1 + girl_brought - girl_ate →
  boy_gathered = 6 := by
sorry

end NUMINAMATH_CALUDE_boy_squirrel_walnuts_l3004_300491


namespace NUMINAMATH_CALUDE_chord_equation_through_midpoint_l3004_300460

/-- The equation of a chord passing through a point on an ellipse --/
theorem chord_equation_through_midpoint (x y : ℝ) :
  (4 * x^2 + 9 * y^2 = 144) →  -- Ellipse equation
  (3 : ℝ)^2 * 4 + 1^2 * 9 < 144 →  -- P(3,1) is inside the ellipse
  (∃ (x₁ y₁ x₂ y₂ : ℝ),  -- Existence of chord endpoints
    (4 * x₁^2 + 9 * y₁^2 = 144) ∧  -- A is on the ellipse
    (4 * x₂^2 + 9 * y₂^2 = 144) ∧  -- B is on the ellipse
    (x₁ + x₂ = 6) ∧  -- P is midpoint (x-coordinate)
    (y₁ + y₂ = 2)) →  -- P is midpoint (y-coordinate)
  (4 * x + 3 * y - 15 = 0)  -- Equation of the chord
  := by sorry

end NUMINAMATH_CALUDE_chord_equation_through_midpoint_l3004_300460


namespace NUMINAMATH_CALUDE_history_students_count_l3004_300433

def total_students : ℕ := 86
def math_students : ℕ := 17
def english_students : ℕ := 36
def all_three_classes : ℕ := 3
def exactly_two_classes : ℕ := 3

theorem history_students_count : 
  ∃ (history_students : ℕ), 
    history_students = total_students - math_students - english_students + all_three_classes := by
  sorry

end NUMINAMATH_CALUDE_history_students_count_l3004_300433


namespace NUMINAMATH_CALUDE_M_N_intersection_empty_l3004_300470

def M : Set ℂ :=
  {z | ∃ t : ℝ, t ≠ -1 ∧ t ≠ 0 ∧ z = t / (1 + t) + Complex.I * ((1 + t) / t)}

def N : Set ℂ :=
  {z | ∃ t : ℝ, |t| ≤ 1 ∧ z = Real.sqrt 2 * (Complex.cos (Real.arcsin t) + Complex.I * Complex.cos (Real.arccos t))}

theorem M_N_intersection_empty : M ∩ N = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_N_intersection_empty_l3004_300470


namespace NUMINAMATH_CALUDE_triangle_area_approx_036_l3004_300440

-- Define the slopes and intersection point
def slope1 : ℚ := 3/4
def slope2 : ℚ := 1/3
def intersection : ℚ × ℚ := (3, 3)

-- Define the lines
def line1 (x : ℚ) : ℚ := slope1 * (x - intersection.1) + intersection.2
def line2 (x : ℚ) : ℚ := slope2 * (x - intersection.1) + intersection.2
def line3 (x y : ℚ) : Prop := x + y = 8

-- Define the function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Theorem statement
theorem triangle_area_approx_036 :
  ∃ (p1 p2 p3 : ℚ × ℚ),
    p1 = intersection ∧
    line1 p2.1 = p2.2 ∧
    line2 p3.1 = p3.2 ∧
    line3 p2.1 p2.2 ∧
    line3 p3.1 p3.2 ∧
    abs (triangleArea p1 p2 p3 - 0.36) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_approx_036_l3004_300440


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3004_300407

theorem digit_sum_problem (a b c d : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) →  -- digits are less than 10
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →  -- digits are different
  (c + a = 10) →  -- condition from right column
  (b + c + 1 = 10) →  -- condition from middle column
  (a + d + 1 = 11) →  -- condition from left column
  (a + b + c + d = 19) :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3004_300407


namespace NUMINAMATH_CALUDE_san_diego_zoo_ticket_cost_l3004_300447

/-- Calculates the total cost of zoo tickets for a family -/
def total_cost_zoo_tickets (family_size : ℕ) (adult_price : ℕ) (child_price : ℕ) (adult_tickets : ℕ) : ℕ :=
  let child_tickets := family_size - adult_tickets
  adult_price * adult_tickets + child_price * child_tickets

/-- Theorem: The total cost of zoo tickets for a family of 7 with 4 adult tickets is $126 -/
theorem san_diego_zoo_ticket_cost :
  total_cost_zoo_tickets 7 21 14 4 = 126 := by
  sorry

end NUMINAMATH_CALUDE_san_diego_zoo_ticket_cost_l3004_300447


namespace NUMINAMATH_CALUDE_parenthesized_subtraction_equality_l3004_300410

theorem parenthesized_subtraction_equality :
  1 - 2 - 3 - 4 - (5 - 6 - 7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_parenthesized_subtraction_equality_l3004_300410


namespace NUMINAMATH_CALUDE_trigonometric_expressions_l3004_300416

open Real

theorem trigonometric_expressions :
  (∀ α : ℝ, tan α = 2 →
    (sin (2 * π - α) + cos (π + α)) / (cos (α - π) - cos ((3 * π) / 2 - α)) = -3) ∧
  sin (50 * π / 180) * (1 + Real.sqrt 3 * tan (10 * π / 180)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_l3004_300416


namespace NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l3004_300497

theorem smallest_third_term_geometric_progression 
  (a b c : ℝ) 
  (arithmetic_prog : a = 7 ∧ c - b = b - a) 
  (geometric_prog : ∃ r : ℝ, r > 0 ∧ (b + 3) = a * r ∧ (c + 22) = (b + 3) * r) :
  c + 22 ≥ 23 + 16 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l3004_300497


namespace NUMINAMATH_CALUDE_marathon_average_time_l3004_300462

-- Define the marathon distance in miles
def marathonDistance : ℕ := 24

-- Define the total time in minutes (3 hours and 36 minutes = 216 minutes)
def totalTimeMinutes : ℕ := 3 * 60 + 36

-- Define the average time per mile
def averageTimePerMile : ℚ := totalTimeMinutes / marathonDistance

-- Theorem statement
theorem marathon_average_time :
  averageTimePerMile = 9 := by sorry

end NUMINAMATH_CALUDE_marathon_average_time_l3004_300462


namespace NUMINAMATH_CALUDE_journey_fraction_l3004_300424

theorem journey_fraction (total_distance : ℝ) (bus_fraction : ℚ) (foot_distance : ℝ) :
  total_distance = 130 →
  bus_fraction = 17 / 20 →
  foot_distance = 6.5 →
  ∃ rail_fraction : ℚ,
    rail_fraction + bus_fraction + (foot_distance / total_distance) = 1 ∧
    rail_fraction = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_journey_fraction_l3004_300424


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3004_300496

theorem simplify_square_roots : 
  Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^4) = 225 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3004_300496


namespace NUMINAMATH_CALUDE_lindas_savings_l3004_300461

theorem lindas_savings (savings : ℚ) : 
  (7/13 : ℚ) * savings + (3/13 : ℚ) * savings + 180 = savings ∧ 
  (3/13 : ℚ) * savings = 2 * 180 → 
  savings = 1560 := by sorry

end NUMINAMATH_CALUDE_lindas_savings_l3004_300461


namespace NUMINAMATH_CALUDE_alternating_ball_probability_l3004_300418

-- Define the number of balls of each color
def white_balls : ℕ := 5
def black_balls : ℕ := 5
def red_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := white_balls + black_balls + red_balls

-- Define the function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the function to calculate the number of successful arrangements
def successful_arrangements : ℕ := 
  (binomial (total_balls) red_balls) * (binomial (white_balls + black_balls) white_balls)

-- Define the function to calculate the total number of arrangements
def total_arrangements : ℕ := 
  Nat.factorial total_balls / (Nat.factorial white_balls * Nat.factorial black_balls * Nat.factorial red_balls)

-- State the theorem
theorem alternating_ball_probability : 
  (successful_arrangements : ℚ) / total_arrangements = 123 / 205 := by sorry

end NUMINAMATH_CALUDE_alternating_ball_probability_l3004_300418


namespace NUMINAMATH_CALUDE_max_consecutive_odd_exponents_is_seven_l3004_300401

/-- A natural number has odd prime factor exponents if all exponents in its prime factorization are odd. -/
def has_odd_prime_factor_exponents (n : ℕ) : Prop :=
  ∀ p k, p.Prime → p ^ k ∣ n → k % 2 = 1

/-- The maximum number of consecutive natural numbers with odd prime factor exponents. -/
def max_consecutive_odd_exponents : ℕ := 7

/-- Theorem stating that the maximum number of consecutive natural numbers 
    with odd prime factor exponents is 7. -/
theorem max_consecutive_odd_exponents_is_seven :
  ∀ n : ℕ, ∃ m ∈ Finset.range 7, ¬(has_odd_prime_factor_exponents (n + m)) ∧
  ∃ k : ℕ, ∀ i ∈ Finset.range 7, has_odd_prime_factor_exponents (k + i) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_odd_exponents_is_seven_l3004_300401


namespace NUMINAMATH_CALUDE_log_identity_l3004_300459

theorem log_identity (x y : ℝ) 
  (hx : Real.log 5 / Real.log 4 = x)
  (hy : Real.log 7 / Real.log 5 = y) : 
  Real.log 7 / Real.log 10 = (2 * x * y) / (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l3004_300459


namespace NUMINAMATH_CALUDE_two_copy_machines_output_l3004_300490

/-- Calculates the total number of copies made by two copy machines in a given time -/
def total_copies (rate1 rate2 time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

/-- Proves that two copy machines with given rates produce 3300 copies in 30 minutes -/
theorem two_copy_machines_output : total_copies 35 75 30 = 3300 := by
  sorry

end NUMINAMATH_CALUDE_two_copy_machines_output_l3004_300490


namespace NUMINAMATH_CALUDE_divisible_by_50_l3004_300405

/-- A polygon drawn on a square grid -/
structure GridPolygon where
  area : ℕ
  divisible_by_2 : ∃ (half : ℕ), area = 2 * half
  divisible_by_25 : ∃ (part : ℕ), area = 25 * part

/-- The main theorem -/
theorem divisible_by_50 (p : GridPolygon) (h : p.area = 100) :
  ∃ (small : ℕ), p.area = 50 * small := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_50_l3004_300405


namespace NUMINAMATH_CALUDE_solve_linear_systems_l3004_300453

theorem solve_linear_systems :
  (∃ (x1 y1 : ℝ), x1 + y1 = 3 ∧ 2*x1 + 3*y1 = 8 ∧ x1 = 1 ∧ y1 = 2) ∧
  (∃ (x2 y2 : ℝ), 5*x2 - 2*y2 = 4 ∧ 2*x2 - 3*y2 = -5 ∧ x2 = 2 ∧ y2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_linear_systems_l3004_300453


namespace NUMINAMATH_CALUDE_jills_peaches_l3004_300448

theorem jills_peaches (steven_peaches : ℕ) (jake_fewer : ℕ) (jake_more : ℕ) 
  (h1 : steven_peaches = 14)
  (h2 : jake_fewer = 6)
  (h3 : jake_more = 3)
  : steven_peaches - jake_fewer - jake_more = 5 := by
  sorry

end NUMINAMATH_CALUDE_jills_peaches_l3004_300448


namespace NUMINAMATH_CALUDE_number_operation_l3004_300425

theorem number_operation (x : ℝ) : (x - 5) / 7 = 7 → (x - 34) / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_l3004_300425


namespace NUMINAMATH_CALUDE_investment_ratio_l3004_300493

/-- Represents the business investment scenario -/
structure Investment where
  nandan_amount : ℝ
  nandan_time : ℝ
  krishan_amount : ℝ
  krishan_time : ℝ
  total_gain : ℝ
  nandan_gain : ℝ

/-- The theorem representing the investment problem -/
theorem investment_ratio (i : Investment) 
  (h1 : i.krishan_amount = 4 * i.nandan_amount)
  (h2 : i.total_gain = 26000)
  (h3 : i.nandan_gain = 2000)
  (h4 : ∃ (k : ℝ), i.nandan_gain / i.total_gain = 
       (i.nandan_amount * i.nandan_time) / 
       (i.nandan_amount * i.nandan_time + i.krishan_amount * i.krishan_time)) :
  i.krishan_time / i.nandan_time = 3 := by
  sorry


end NUMINAMATH_CALUDE_investment_ratio_l3004_300493


namespace NUMINAMATH_CALUDE_no_integer_solution_l3004_300421

theorem no_integer_solution : ¬ ∃ (x : ℤ), 7 - 3 * (x^2 - 2) > 19 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3004_300421


namespace NUMINAMATH_CALUDE_triangular_square_triangular_l3004_300483

/-- Definition of triangular number -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Statement: 1 and 6 are the only triangular numbers whose squares are also triangular numbers -/
theorem triangular_square_triangular :
  ∀ n : ℕ, (∃ m : ℕ, (triangular n)^2 = triangular m) ↔ n = 1 ∨ n = 3 := by
sorry

end NUMINAMATH_CALUDE_triangular_square_triangular_l3004_300483


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3004_300498

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 3 → x^3 - 27 > 0) ↔ (∃ x : ℝ, x > 3 ∧ x^3 - 27 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3004_300498


namespace NUMINAMATH_CALUDE_expression_equality_l3004_300456

theorem expression_equality : (50 - (5020 - 520)) + (5020 - (520 - 50)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3004_300456


namespace NUMINAMATH_CALUDE_movie_production_people_l3004_300413

/-- The number of people at the movie production --/
def num_people : ℕ := 50

/-- The cost of hiring actors --/
def actor_cost : ℕ := 1200

/-- The cost of food per person --/
def food_cost_per_person : ℕ := 3

/-- The total cost of the movie production --/
def total_cost : ℕ := 10000 - 5950

/-- The equipment rental cost is twice the combined cost of food and actors --/
def equipment_cost (p : ℕ) : ℕ := 2 * (food_cost_per_person * p + actor_cost)

/-- The total cost calculation based on the number of people --/
def calculated_cost (p : ℕ) : ℕ :=
  actor_cost + food_cost_per_person * p + equipment_cost p

theorem movie_production_people :
  calculated_cost num_people = total_cost :=
by sorry

end NUMINAMATH_CALUDE_movie_production_people_l3004_300413


namespace NUMINAMATH_CALUDE_bills_soaking_time_l3004_300468

/-- Calculates the total soaking time for Bill's clothes -/
def total_soaking_time (grass_stain_time grass_stains marinara_stain_time marinara_stains : ℕ) : ℕ :=
  grass_stain_time * grass_stains + marinara_stain_time * marinara_stains

/-- Proves that the total soaking time for Bill's clothes is 19 minutes -/
theorem bills_soaking_time :
  total_soaking_time 4 3 7 1 = 19 := by
  sorry

#eval total_soaking_time 4 3 7 1

end NUMINAMATH_CALUDE_bills_soaking_time_l3004_300468


namespace NUMINAMATH_CALUDE_average_problem_l3004_300432

theorem average_problem (x y : ℝ) : 
  ((100 + 200300 + x) / 3 = 250) → 
  ((300 + 150100 + x + y) / 4 = 200) → 
  y = -4250 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l3004_300432


namespace NUMINAMATH_CALUDE_inequality_proof_l3004_300408

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3004_300408


namespace NUMINAMATH_CALUDE_fraction_zero_implies_a_negative_two_l3004_300472

theorem fraction_zero_implies_a_negative_two (a : ℝ) : 
  (a^2 - 4) / (a - 2) = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_a_negative_two_l3004_300472


namespace NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l3004_300463

/-- The sum of the series 1/(n(n+2)) from n=1 to infinity equals 3/4 -/
theorem series_sum_equals_three_fourths :
  ∑' n, (1 : ℝ) / (n * (n + 2)) = 3/4 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l3004_300463


namespace NUMINAMATH_CALUDE_five_digit_numbers_count_l3004_300426

/-- The number of odd digits available -/
def num_odd_digits : ℕ := 5

/-- The number of even digits available -/
def num_even_digits : ℕ := 5

/-- The total number of digits in the formed numbers -/
def total_digits : ℕ := 5

/-- Function to calculate the number of ways to form five-digit numbers -/
def count_five_digit_numbers : ℕ :=
  let case1 := Nat.choose num_odd_digits 2 * Nat.choose (num_even_digits - 1) 3 * Nat.factorial total_digits
  let case2 := Nat.choose num_odd_digits 2 * Nat.choose (num_even_digits - 1) 2 * Nat.choose 4 1 * Nat.factorial 4
  case1 + case2

/-- Theorem stating that the number of unique five-digit numbers is 10,560 -/
theorem five_digit_numbers_count : count_five_digit_numbers = 10560 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_numbers_count_l3004_300426


namespace NUMINAMATH_CALUDE_fifth_power_last_digit_l3004_300420

theorem fifth_power_last_digit (n : ℕ) : 10 ∣ (n^5 - n) := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_last_digit_l3004_300420
