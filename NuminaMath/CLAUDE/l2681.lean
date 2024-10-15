import Mathlib

namespace NUMINAMATH_CALUDE_intersection_distance_product_l2681_268176

/-- Given a line L and a circle C, prove that the product of distances from a point on the line to the intersection points of the line and circle is 1/4. -/
theorem intersection_distance_product (P : ℝ × ℝ) (α : ℝ) (C : Set (ℝ × ℝ)) : 
  P = (1/2, 1) →
  α = π/6 →
  C = {(x, y) | x^2 + y^2 = x + y} →
  ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ 
    (∃ (t₁ t₂ : ℝ), 
      A = (1/2 + (Real.sqrt 3)/2 * t₁, 1 + 1/2 * t₁) ∧
      B = (1/2 + (Real.sqrt 3)/2 * t₂, 1 + 1/2 * t₂)) ∧
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) * 
    Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_product_l2681_268176


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2681_268162

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 22) (h2 : x - y = 16) : 
  min x y = 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2681_268162


namespace NUMINAMATH_CALUDE_money_left_after_shopping_l2681_268119

def bread_price : ℝ := 2
def butter_original_price : ℝ := 3
def butter_discount : ℝ := 0.1
def juice_price_multiplier : ℝ := 2
def cookies_original_price : ℝ := 4
def cookies_discount : ℝ := 0.2
def vat_rate : ℝ := 0.05
def initial_money : ℝ := 20

def calculate_discounted_price (original_price discount : ℝ) : ℝ :=
  original_price * (1 - discount)

def calculate_total_cost (bread butter juice cookies : ℝ) : ℝ :=
  bread + butter + juice + cookies

def apply_vat (total_cost vat_rate : ℝ) : ℝ :=
  total_cost * (1 + vat_rate)

theorem money_left_after_shopping :
  let butter_price := calculate_discounted_price butter_original_price butter_discount
  let cookies_price := calculate_discounted_price cookies_original_price cookies_discount
  let juice_price := bread_price * juice_price_multiplier
  let total_cost := calculate_total_cost bread_price butter_price juice_price cookies_price
  let final_cost := apply_vat total_cost vat_rate
  initial_money - final_cost = 7.5 := by sorry

end NUMINAMATH_CALUDE_money_left_after_shopping_l2681_268119


namespace NUMINAMATH_CALUDE_compound_proposition_1_compound_proposition_2_compound_proposition_3_l2681_268175

-- Define the propositions
def smallest_angle_not_greater_than_60 (α : Real) : Prop :=
  (∀ β γ : Real, α + β + γ = 180 → α ≤ β ∧ α ≤ γ) → α ≤ 60

def isosceles_right_triangle (α β γ : Real) : Prop :=
  α + β + γ = 180 ∧ α = 90 ∧ β = 45 ∧ (γ = α ∨ γ = β) ∧ α = 90

def triangle_with_60_degree (α β γ : Real) : Prop :=
  α + β + γ = 180 ∧ (α = 60 ∨ β = 60 ∨ γ = 60)

-- Theorem statements
theorem compound_proposition_1 (α : Real) :
  smallest_angle_not_greater_than_60 α ↔ 
  ¬(∀ β γ : Real, α + β + γ = 180 → α ≤ β ∧ α ≤ γ → α > 60) :=
sorry

theorem compound_proposition_2 (α β γ : Real) :
  isosceles_right_triangle α β γ ↔
  (α + β + γ = 180 ∧ α = 90 ∧ β = 45 ∧ (γ = α ∨ γ = β)) ∧
  (α + β + γ = 180 ∧ α = 90 ∧ β = 45) :=
sorry

theorem compound_proposition_3 (α β γ : Real) :
  triangle_with_60_degree α β γ ↔
  (α + β + γ = 180 ∧ α = 60 ∧ β = 60 ∧ γ = 60) ∨
  (α + β + γ = 180 ∧ (α = 60 ∨ β = 60 ∨ γ = 60) ∧ (α = 90 ∨ β = 90 ∨ γ = 90)) :=
sorry

end NUMINAMATH_CALUDE_compound_proposition_1_compound_proposition_2_compound_proposition_3_l2681_268175


namespace NUMINAMATH_CALUDE_range_of_a_l2681_268161

-- Define the function f(x) = (a^2 - 1)x^2 - (a-1)x - 1
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 1) * x^2 - (a - 1) * x - 1

-- Define the property that f(x) < 0 for all real x
def always_negative (a : ℝ) : Prop := ∀ x : ℝ, f a x < 0

-- Theorem statement
theorem range_of_a : 
  {a : ℝ | always_negative a} = Set.Ioc (- 3/5) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2681_268161


namespace NUMINAMATH_CALUDE_unique_solution_system_l2681_268143

theorem unique_solution_system : 
  ∃! (a b c : ℕ+), 
    (a.val : ℤ)^3 - (b.val : ℤ)^3 - (c.val : ℤ)^3 = 3 * (a.val : ℤ) * (b.val : ℤ) * (c.val : ℤ) ∧ 
    (a.val : ℤ)^2 = 2 * ((b.val : ℤ) + (c.val : ℤ)) ∧
    a.val = 2 ∧ b.val = 1 ∧ c.val = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2681_268143


namespace NUMINAMATH_CALUDE_apples_on_tree_l2681_268157

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := 7

/-- The number of apples Rachel picked -/
def picked_apples : ℕ := 4

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := initial_apples - picked_apples

theorem apples_on_tree : remaining_apples = 3 := by
  sorry

end NUMINAMATH_CALUDE_apples_on_tree_l2681_268157


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2681_268137

theorem cubic_equation_solutions :
  ∀ x y z : ℤ,
  (x + y + z = 2 ∧ x^3 + y^3 + z^3 = -10) ↔
  ((x, y, z) = (3, 3, -4) ∨ (x, y, z) = (3, -4, 3) ∨ (x, y, z) = (-4, 3, 3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2681_268137


namespace NUMINAMATH_CALUDE_trailing_zeros_50_factorial_l2681_268103

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailing_zeros_50_factorial : trailingZeros 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_50_factorial_l2681_268103


namespace NUMINAMATH_CALUDE_intersection_equals_positive_l2681_268185

-- Define sets A and B
def A : Set ℝ := {x | 2 * x^2 + x > 0}
def B : Set ℝ := {x | 2 * x + 1 > 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_positive : A_intersect_B = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_positive_l2681_268185


namespace NUMINAMATH_CALUDE_sum_of_squares_power_of_three_l2681_268183

theorem sum_of_squares_power_of_three (n : ℕ) :
  ∃ x y z : ℤ, (Nat.gcd (Nat.gcd x.natAbs y.natAbs) z.natAbs = 1) ∧
  (x^2 + y^2 + z^2 = 3^(2^n)) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_power_of_three_l2681_268183


namespace NUMINAMATH_CALUDE_special_triangle_properties_l2681_268165

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions
  angle_sum : A + B + C = π
  side_angle_relation : (Real.cos B) / (Real.cos C) = b / (2 * a - c)
  b_value : b = Real.sqrt 7
  area : (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2

/-- Main theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) : t.B = π/3 ∧ t.a + t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l2681_268165


namespace NUMINAMATH_CALUDE_cheap_handcuff_time_is_6_l2681_268194

/-- The time it takes to pick the lock on a cheap pair of handcuffs -/
def cheap_handcuff_time : ℝ := 6

/-- The time it takes to pick the lock on an expensive pair of handcuffs -/
def expensive_handcuff_time : ℝ := 8

/-- The number of friends to rescue -/
def num_friends : ℕ := 3

/-- The total time it takes to free all friends -/
def total_rescue_time : ℝ := 42

/-- Theorem stating that the time to pick a cheap handcuff lock is 6 minutes -/
theorem cheap_handcuff_time_is_6 :
  cheap_handcuff_time = 6 ∧
  num_friends * (cheap_handcuff_time + expensive_handcuff_time) = total_rescue_time :=
by sorry

end NUMINAMATH_CALUDE_cheap_handcuff_time_is_6_l2681_268194


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l2681_268116

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_150_is_identity :
  B ^ 150 = 1 := by sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l2681_268116


namespace NUMINAMATH_CALUDE_f_11_equals_149_l2681_268181

def f (n : ℕ) : ℕ := n^2 + n + 17

theorem f_11_equals_149 : f 11 = 149 := by sorry

end NUMINAMATH_CALUDE_f_11_equals_149_l2681_268181


namespace NUMINAMATH_CALUDE_clock_angle_at_two_thirty_l2681_268144

/-- The measure of the smaller angle formed by the hour-hand and minute-hand of a clock at 2:30 -/
def clock_angle : ℝ := 105

/-- The number of degrees in a full circle on a clock -/
def full_circle : ℝ := 360

/-- The number of hours on a clock -/
def clock_hours : ℕ := 12

/-- The hour component of the time -/
def hour : ℕ := 2

/-- The minute component of the time -/
def minute : ℕ := 30

theorem clock_angle_at_two_thirty :
  clock_angle = min (|hour_angle - minute_angle|) (full_circle - |hour_angle - minute_angle|) :=
by
  sorry
where
  /-- The angle of the hour hand from 12 o'clock position -/
  hour_angle : ℝ := (hour + minute / 60) * (full_circle / clock_hours)
  /-- The angle of the minute hand from 12 o'clock position -/
  minute_angle : ℝ := minute * (full_circle / 60)

#check clock_angle_at_two_thirty

end NUMINAMATH_CALUDE_clock_angle_at_two_thirty_l2681_268144


namespace NUMINAMATH_CALUDE_plane_division_l2681_268133

/-- The number of parts into which n lines divide a plane -/
def f (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem stating that f(n) correctly counts the number of parts for n ≥ 2 -/
theorem plane_division (n : ℕ) (h : n ≥ 2) : 
  f n = 1 + n * (n + 1) / 2 := by
  sorry

/-- Helper lemma for the induction step -/
lemma induction_step (k : ℕ) (h : k ≥ 2) :
  f (k + 1) = f k + (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_plane_division_l2681_268133


namespace NUMINAMATH_CALUDE_area_BXC_specific_trapezoid_l2681_268151

/-- Represents a trapezoid ABCD with intersection point X of diagonals -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  area : ℝ

/-- Calculates the area of triangle BXC in the trapezoid -/
def area_BXC (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of triangle BXC in the specific trapezoid -/
theorem area_BXC_specific_trapezoid :
  let t : Trapezoid := { AB := 20, CD := 30, area := 300 }
  area_BXC t = 72 := by sorry

end NUMINAMATH_CALUDE_area_BXC_specific_trapezoid_l2681_268151


namespace NUMINAMATH_CALUDE_product_of_numbers_l2681_268123

theorem product_of_numbers (x y : ℝ) : 
  x - y = 7 → x^2 + y^2 = 85 → x * y = 18 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2681_268123


namespace NUMINAMATH_CALUDE_minimum_value_inequality_minimum_value_achievable_l2681_268172

theorem minimum_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 5) : 
  1/x + 4/y + 9/z ≥ 36/5 := by
  sorry

theorem minimum_value_achievable : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 5 ∧ 1/x + 4/y + 9/z = 36/5 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_inequality_minimum_value_achievable_l2681_268172


namespace NUMINAMATH_CALUDE_factorial_ratio_l2681_268122

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2681_268122


namespace NUMINAMATH_CALUDE_marked_squares_rearrangement_l2681_268145

/-- Represents a square table with marked cells -/
structure MarkedTable (n : ℕ) where
  marks : Finset ((Fin n) × (Fin n))
  mark_count : marks.card = 110

/-- Represents a permutation of rows and columns -/
structure TablePermutation (n : ℕ) where
  row_perm : Equiv.Perm (Fin n)
  col_perm : Equiv.Perm (Fin n)

/-- Checks if a cell is on or above the main diagonal -/
def is_on_or_above_diagonal {n : ℕ} (i j : Fin n) : Prop :=
  i.val ≤ j.val

/-- Applies a permutation to a marked cell -/
def apply_perm {n : ℕ} (perm : TablePermutation n) (cell : (Fin n) × (Fin n)) : (Fin n) × (Fin n) :=
  (perm.row_perm cell.1, perm.col_perm cell.2)

/-- Theorem: For any 100x100 table with 110 marked squares, there exists a permutation
    that places all marked squares on or above the main diagonal -/
theorem marked_squares_rearrangement :
  ∀ (t : MarkedTable 100),
  ∃ (perm : TablePermutation 100),
  ∀ cell ∈ t.marks,
  is_on_or_above_diagonal (apply_perm perm cell).1 (apply_perm perm cell).2 :=
sorry

end NUMINAMATH_CALUDE_marked_squares_rearrangement_l2681_268145


namespace NUMINAMATH_CALUDE_p_recurrence_l2681_268178

/-- The probability of getting a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℝ := sorry

/-- The recurrence relation for p_{n,k} -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + 1 / 2^k := by sorry

end NUMINAMATH_CALUDE_p_recurrence_l2681_268178


namespace NUMINAMATH_CALUDE_largest_710_double_correct_l2681_268152

/-- Converts a positive integer to its base-7 representation as a list of digits --/
def toBase7 (n : ℕ+) : List ℕ :=
  sorry

/-- Converts a list of digits to a base-10 number --/
def toBase10 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a positive integer is a 7-10 double --/
def is710Double (n : ℕ+) : Prop :=
  toBase10 (toBase7 n) = 2 * n

/-- The largest 7-10 double --/
def largest710Double : ℕ+ := 315

theorem largest_710_double_correct :
  is710Double largest710Double ∧
  ∀ n : ℕ+, n > largest710Double → ¬is710Double n :=
sorry

end NUMINAMATH_CALUDE_largest_710_double_correct_l2681_268152


namespace NUMINAMATH_CALUDE_camera_profit_difference_l2681_268131

/-- Represents the profit calculation for camera sales -/
def camera_profit (num_cameras : ℕ) (buy_price sell_price : ℚ) : ℚ :=
  num_cameras * (sell_price - buy_price)

/-- Represents the problem of calculating the difference in profit between Maddox and Theo -/
theorem camera_profit_difference : 
  let num_cameras : ℕ := 3
  let buy_price : ℚ := 20
  let maddox_sell_price : ℚ := 28
  let theo_sell_price : ℚ := 23
  let maddox_profit := camera_profit num_cameras buy_price maddox_sell_price
  let theo_profit := camera_profit num_cameras buy_price theo_sell_price
  maddox_profit - theo_profit = 15 := by
sorry


end NUMINAMATH_CALUDE_camera_profit_difference_l2681_268131


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2681_268160

theorem parabola_line_intersection (b : ℝ) : 
  (∃! x : ℝ, bx^2 + 5*x + 2 = -2*x - 2) ↔ b = 49/16 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2681_268160


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2681_268190

theorem least_number_divisible_by_five_primes : 
  ∃ n : ℕ, (n > 0) ∧ 
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ q₁ q₂ q₃ q₄ q₅ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0) → 
    m ≥ n) ∧
  n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2681_268190


namespace NUMINAMATH_CALUDE_log_properties_l2681_268124

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem statement
theorem log_properties (a M N x : ℝ) 
  (ha : a > 0 ∧ a ≠ 1) (hM : M > 0) (hN : N > 0) :
  (log a (a^x) = x) ∧
  (log a (M / N) = log a M - log a N) ∧
  (log a (M * N) = log a M + log a N) := by
  sorry

end NUMINAMATH_CALUDE_log_properties_l2681_268124


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_angle_l2681_268158

theorem triangle_arithmetic_sequence_angle (α d : ℝ) :
  (α - d) + α + (α + d) = 180 → α = 60 ∨ (α - d) = 60 ∨ (α + d) = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_angle_l2681_268158


namespace NUMINAMATH_CALUDE_shaded_area_of_divided_square_l2681_268138

theorem shaded_area_of_divided_square (side_length : ℝ) (total_squares : ℕ) (shaded_squares : ℕ) : 
  side_length = 10 ∧ total_squares = 25 ∧ shaded_squares = 5 → 
  (side_length^2 / total_squares) * shaded_squares = 20 := by
  sorry

#check shaded_area_of_divided_square

end NUMINAMATH_CALUDE_shaded_area_of_divided_square_l2681_268138


namespace NUMINAMATH_CALUDE_hot_dog_stand_ketchup_bottles_l2681_268173

/-- Given a ratio of condiment bottles and the number of mayo bottles,
    calculate the number of ketchup bottles -/
def ketchup_bottles (ketchup_ratio mustard_ratio mayo_ratio mayo_count : ℕ) : ℕ :=
  (ketchup_ratio * mayo_count) / mayo_ratio

theorem hot_dog_stand_ketchup_bottles :
  ketchup_bottles 3 3 2 4 = 6 := by sorry

end NUMINAMATH_CALUDE_hot_dog_stand_ketchup_bottles_l2681_268173


namespace NUMINAMATH_CALUDE_max_individual_score_l2681_268106

theorem max_individual_score (n : ℕ) (total : ℕ) (min_score : ℕ) 
  (h1 : n = 12)
  (h2 : total = 100)
  (h3 : min_score = 7)
  (h4 : ∀ p : ℕ, p ≤ n → min_score ≤ p) :
  ∃ max_score : ℕ, 
    (∀ p : ℕ, p ≤ n → p ≤ max_score) ∧ 
    (∃ player : ℕ, player ≤ n ∧ player = max_score) ∧
    max_score = 23 :=
sorry

end NUMINAMATH_CALUDE_max_individual_score_l2681_268106


namespace NUMINAMATH_CALUDE_quadrilateral_classification_l2681_268104

/-- A quadrilateral with angles α, β, γ, δ satisfying certain conditions --/
structure Quadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  angle_sum : α + β + γ + δ = 2 * Real.pi
  cosine_sum : Real.cos α + Real.cos β + Real.cos γ + Real.cos δ = 0

/-- Definition of a parallelogram based on opposite angles --/
def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.α + q.γ = Real.pi) ∧ (q.β + q.δ = Real.pi)

/-- Definition of a cyclic quadrilateral based on opposite angles --/
def is_cyclic (q : Quadrilateral) : Prop :=
  (q.α + q.γ = Real.pi) ∨ (q.β + q.δ = Real.pi)

/-- Definition of a trapezoid based on adjacent angles --/
def is_trapezoid (q : Quadrilateral) : Prop :=
  (q.α + q.β = Real.pi) ∨ (q.β + q.γ = Real.pi) ∨ (q.γ + q.δ = Real.pi) ∨ (q.δ + q.α = Real.pi)

/-- Main theorem: A quadrilateral with the given properties is either a parallelogram, cyclic, or trapezoid --/
theorem quadrilateral_classification (q : Quadrilateral) :
  is_parallelogram q ∨ is_cyclic q ∨ is_trapezoid q := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_classification_l2681_268104


namespace NUMINAMATH_CALUDE_alex_cookies_l2681_268135

theorem alex_cookies (alex sam : ℕ) : 
  alex = sam + 8 → 
  sam = alex / 3 → 
  alex = 12 := by
sorry

end NUMINAMATH_CALUDE_alex_cookies_l2681_268135


namespace NUMINAMATH_CALUDE_no_square_with_seven_lattice_points_l2681_268163

/-- A square in a right-angled coordinate system -/
structure RotatedSquare where
  /-- The center of the square -/
  center : ℝ × ℝ
  /-- The side length of the square -/
  side_length : ℝ
  /-- The angle between the sides of the square and the coordinate axes (in radians) -/
  angle : ℝ

/-- A lattice point in the coordinate system -/
def LatticePoint : Type := ℤ × ℤ

/-- Predicate to check if a point is inside a rotated square -/
def is_inside (s : RotatedSquare) (p : ℝ × ℝ) : Prop := sorry

/-- Count the number of lattice points inside a rotated square -/
def count_lattice_points_inside (s : RotatedSquare) : ℕ := sorry

/-- Theorem stating that no rotated square at 45° contains exactly 7 lattice points -/
theorem no_square_with_seven_lattice_points :
  ¬ ∃ (s : RotatedSquare), s.angle = π / 4 ∧ count_lattice_points_inside s = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_square_with_seven_lattice_points_l2681_268163


namespace NUMINAMATH_CALUDE_bus_passengers_l2681_268127

theorem bus_passengers (initial : ℕ) (got_on : ℕ) (current : ℕ) : 
  got_on = 13 → current = 17 → initial + got_on = current → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l2681_268127


namespace NUMINAMATH_CALUDE_teachers_count_correct_teachers_count_l2681_268168

theorem teachers_count (num_students : ℕ) (ticket_cost : ℕ) (total_cost : ℕ) : ℕ :=
  let total_tickets := total_cost / ticket_cost
  let num_teachers := total_tickets - num_students
  num_teachers

theorem correct_teachers_count :
  teachers_count 20 5 115 = 3 := by
  sorry

end NUMINAMATH_CALUDE_teachers_count_correct_teachers_count_l2681_268168


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2681_268120

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2681_268120


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l2681_268155

/-- A parabola with equation y = x^2 + 4 -/
def parabola (x y : ℝ) : Prop := y = x^2 + 4

/-- A hyperbola with equation y^2 - mx^2 = 1, where m is a parameter -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 1

/-- Two curves are tangent if they intersect at exactly one point -/
def are_tangent (curve1 curve2 : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, curve1 p.1 p.2 ∧ curve2 p.1 p.2

theorem parabola_hyperbola_tangency (m : ℝ) :
  are_tangent (parabola) (hyperbola m) → m = 8 + 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l2681_268155


namespace NUMINAMATH_CALUDE_circle_trajectory_l2681_268169

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- The fixed point A -/
def A : Point := ⟨2, 0⟩

/-- Checks if a circle passes through a given point -/
def passesThrough (c : Circle) (p : Point) : Prop :=
  (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2

/-- Checks if a circle intersects the y-axis forming a chord of length 4 -/
def intersectsYAxis (c : Circle) : Prop :=
  c.radius^2 = c.center.x^2 + 4

/-- The trajectory of the center of the moving circle -/
def trajectory (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- Theorem: The trajectory of the center of a circle that passes through (2,0) 
    and intersects the y-axis forming a chord of length 4 is y² = 4x -/
theorem circle_trajectory : 
  ∀ (c : Circle), 
    passesThrough c A → 
    intersectsYAxis c → 
    trajectory c.center :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_l2681_268169


namespace NUMINAMATH_CALUDE_percentage_problem_l2681_268199

theorem percentage_problem (P : ℝ) : 
  (P ≥ 0 ∧ P ≤ 100) → 
  (P / 100) * 3200 = (20 / 100) * 650 + 190 → 
  P = 10 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l2681_268199


namespace NUMINAMATH_CALUDE_arithmetic_combination_equals_24_l2681_268182

theorem arithmetic_combination_equals_24 : ∃ (expr : ℝ → ℝ → ℝ → ℝ → ℝ), 
  (expr 5 7 8 8 = 24) ∧ 
  (∀ a b c d, expr a b c d = ((b + c) / a) * d ∨ expr a b c d = ((b - a) * c) + d) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_combination_equals_24_l2681_268182


namespace NUMINAMATH_CALUDE_rectangle_length_l2681_268146

theorem rectangle_length (l w : ℝ) (h1 : l = 4 * w) (h2 : l * w = 100) : l = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l2681_268146


namespace NUMINAMATH_CALUDE_cats_dogs_percentage_difference_l2681_268107

/-- Represents the number of animals in a compound -/
structure AnimalCount where
  cats : ℕ
  dogs : ℕ
  frogs : ℕ

/-- The conditions of the animal compound problem -/
def CompoundConditions (count : AnimalCount) : Prop :=
  count.cats < count.dogs ∧
  count.frogs = 2 * count.dogs ∧
  count.cats + count.dogs + count.frogs = 304 ∧
  count.frogs = 160

/-- The percentage difference between dogs and cats -/
def PercentageDifference (count : AnimalCount) : ℚ :=
  (count.dogs - count.cats : ℚ) / count.dogs * 100

/-- Theorem stating the percentage difference between dogs and cats -/
theorem cats_dogs_percentage_difference (count : AnimalCount) 
  (h : CompoundConditions count) : PercentageDifference count = 20 := by
  sorry


end NUMINAMATH_CALUDE_cats_dogs_percentage_difference_l2681_268107


namespace NUMINAMATH_CALUDE_minimum_artists_count_l2681_268102

theorem minimum_artists_count : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 5 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 5 = 1 ∧ m % 6 = 2 ∧ m % 8 = 3 → m ≥ n) ∧
  n = 236 := by
  sorry

end NUMINAMATH_CALUDE_minimum_artists_count_l2681_268102


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l2681_268187

def f (m : ℝ) (x : ℝ) : ℝ := x^4 + (m-1)*x + 1

theorem even_function_implies_m_equals_one (m : ℝ) :
  (∀ x, f m x = f m (-x)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l2681_268187


namespace NUMINAMATH_CALUDE_dans_remaining_money_l2681_268159

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial : ℕ) (cost : ℕ) : ℕ :=
  initial - cost

theorem dans_remaining_money :
  remaining_money 4 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l2681_268159


namespace NUMINAMATH_CALUDE_original_denominator_proof_l2681_268166

theorem original_denominator_proof (d : ℚ) : 
  (5 / d : ℚ) ≠ 0 → -- Ensure the original fraction is well-defined
  ((5 - 3) / (d + 4) : ℚ) = 1 / 3 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l2681_268166


namespace NUMINAMATH_CALUDE_james_bike_ride_l2681_268111

/-- Proves that given the conditions of James' bike ride, the third hour distance is 25% farther than the second hour distance -/
theorem james_bike_ride (second_hour_distance : ℝ) (total_distance : ℝ) :
  second_hour_distance = 18 →
  second_hour_distance = (1 + 0.2) * (second_hour_distance / 1.2) →
  total_distance = 55.5 →
  (total_distance - (second_hour_distance + second_hour_distance / 1.2)) / second_hour_distance = 0.25 := by
  sorry

#check james_bike_ride

end NUMINAMATH_CALUDE_james_bike_ride_l2681_268111


namespace NUMINAMATH_CALUDE_cycle_loss_percentage_l2681_268115

/-- Given a cost price and selling price, calculate the percentage of loss -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice * 100

theorem cycle_loss_percentage :
  let costPrice : ℚ := 2800
  let sellingPrice : ℚ := 2100
  percentageLoss costPrice sellingPrice = 25 := by
  sorry

end NUMINAMATH_CALUDE_cycle_loss_percentage_l2681_268115


namespace NUMINAMATH_CALUDE_average_monthly_balance_l2681_268188

def monthly_balances : List ℝ := [100, 200, 250, 250, 150, 100]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 175 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l2681_268188


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_product_and_gcd_l2681_268136

theorem two_digit_numbers_with_product_and_gcd 
  (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a < 100) 
  (h2 : 10 ≤ b ∧ b < 100) 
  (h3 : a * b = 1728) 
  (h4 : Nat.gcd a b = 12) : 
  (a = 36 ∧ b = 48) ∨ (a = 48 ∧ b = 36) := by
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_product_and_gcd_l2681_268136


namespace NUMINAMATH_CALUDE_greatest_b_value_l2681_268121

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 12 ≥ 0 → x ≤ 6) ∧ 
  (-6^2 + 8*6 - 12 ≥ 0) := by
sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2681_268121


namespace NUMINAMATH_CALUDE_topsoil_cost_l2681_268171

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards : ℝ := 7

/-- The total cost of topsoil for a given number of cubic yards -/
def total_cost (yards : ℝ) : ℝ :=
  yards * cubic_feet_per_cubic_yard * cost_per_cubic_foot

theorem topsoil_cost : total_cost cubic_yards = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l2681_268171


namespace NUMINAMATH_CALUDE_molecular_weight_CaI2_value_l2681_268167

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of Calcium atoms in CaI2 -/
def num_Ca : ℕ := 1

/-- The number of Iodine atoms in CaI2 -/
def num_I : ℕ := 2

/-- The molecular weight of CaI2 in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca * num_Ca + atomic_weight_I * num_I

theorem molecular_weight_CaI2_value : molecular_weight_CaI2 = 293.88 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_CaI2_value_l2681_268167


namespace NUMINAMATH_CALUDE_min_value_expression_l2681_268142

open Real

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min : ℝ), min = Real.sqrt 39 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 →
    (|6*x - 4*y| + |3*(x + y*Real.sqrt 3) + 2*(x*Real.sqrt 3 - y)|) / Real.sqrt (x^2 + y^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2681_268142


namespace NUMINAMATH_CALUDE_equation_solution_l2681_268112

theorem equation_solution : ∃! x : ℝ, 13 + Real.sqrt (-4 + x - 3 * 3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2681_268112


namespace NUMINAMATH_CALUDE_total_cement_is_15_1_l2681_268198

/-- The amount of cement used for Lexi's street in tons -/
def lexiStreetCement : ℝ := 10

/-- The amount of cement used for Tess's street in tons -/
def tessStreetCement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company in tons -/
def totalCement : ℝ := lexiStreetCement + tessStreetCement

/-- Theorem stating that the total cement used is 15.1 tons -/
theorem total_cement_is_15_1 : totalCement = 15.1 := by sorry

end NUMINAMATH_CALUDE_total_cement_is_15_1_l2681_268198


namespace NUMINAMATH_CALUDE_pyramid_volume_approx_l2681_268125

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


end NUMINAMATH_CALUDE_pyramid_volume_approx_l2681_268125


namespace NUMINAMATH_CALUDE_fraction_simplification_l2681_268192

theorem fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  (9 * x^3 * y^2) / (12 * x^2 * y^4) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2681_268192


namespace NUMINAMATH_CALUDE_local_max_condition_l2681_268148

theorem local_max_condition (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ IsLocalMax (fun x => Real.exp x + a * x) x) →
  a < -1 := by sorry

end NUMINAMATH_CALUDE_local_max_condition_l2681_268148


namespace NUMINAMATH_CALUDE_closest_option_is_150000_l2681_268177

/-- Represents the population of the United States in 2020 --/
def us_population : ℕ := 331000000

/-- Represents the total area of the United States in square miles --/
def us_area : ℕ := 3800000

/-- Represents the number of square feet in one square mile --/
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

/-- Calculates the average number of square feet per person --/
def avg_sq_feet_per_person : ℚ :=
  (us_area * sq_feet_per_sq_mile) / us_population

/-- List of given options for the average square feet per person --/
def options : List ℕ := [30000, 60000, 90000, 120000, 150000]

/-- Theorem stating that 150000 is the closest option to the actual average --/
theorem closest_option_is_150000 :
  ∃ (x : ℕ), x ∈ options ∧ 
  ∀ (y : ℕ), y ∈ options → |avg_sq_feet_per_person - x| ≤ |avg_sq_feet_per_person - y| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_option_is_150000_l2681_268177


namespace NUMINAMATH_CALUDE_sum_of_two_and_repeating_third_l2681_268184

-- Define the repeating decimal 0.3333...
def repeating_third : ℚ := 1 / 3

-- Theorem statement
theorem sum_of_two_and_repeating_third :
  2 + repeating_third = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_and_repeating_third_l2681_268184


namespace NUMINAMATH_CALUDE_half_fourth_of_twelve_y_plus_three_l2681_268195

theorem half_fourth_of_twelve_y_plus_three (y : ℝ) :
  (1/2) * (1/4) * (12 * y + 3) = (3 * y) / 2 + 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_half_fourth_of_twelve_y_plus_three_l2681_268195


namespace NUMINAMATH_CALUDE_complement_of_70_degrees_l2681_268153

theorem complement_of_70_degrees :
  let given_angle : ℝ := 70
  let complement_sum : ℝ := 90
  let complement_angle : ℝ := complement_sum - given_angle
  complement_angle = 20 := by sorry

end NUMINAMATH_CALUDE_complement_of_70_degrees_l2681_268153


namespace NUMINAMATH_CALUDE_player_a_wins_l2681_268139

/-- Represents a player in the chocolate bar game -/
inductive Player
| A
| B

/-- Represents a move in the chocolate bar game -/
inductive Move
| Single
| Double

/-- Represents the state of the chocolate bar game -/
structure GameState where
  grid : Fin 7 → Fin 7 → Bool
  current_player : Player

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Checks if a move is valid for the current player and game state -/
def is_valid_move (gs : GameState) (m : Move) : Bool :=
  match gs.current_player, m with
  | Player.A, Move.Single => true
  | Player.B, _ => true
  | _, _ => false

/-- Applies a move to the game state, returning the new state -/
def apply_move (gs : GameState) (m : Move) : GameState :=
  sorry

/-- Counts the number of squares taken by a player -/
def count_squares (gs : GameState) (p : Player) : Nat :=
  sorry

/-- The main theorem stating that Player A can always secure more than half the squares -/
theorem player_a_wins (init_state : GameState) (strategy_a strategy_b : Strategy) :
  ∃ (final_state : GameState),
    count_squares final_state Player.A > 24 :=
  sorry

end NUMINAMATH_CALUDE_player_a_wins_l2681_268139


namespace NUMINAMATH_CALUDE_degree_of_composed_and_multiplied_polynomials_l2681_268140

/-- The degree of a polynomial -/
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

/-- Polynomial composition -/
def polyComp (p q : Polynomial ℝ) : Polynomial ℝ := sorry

/-- Polynomial multiplication -/
def polyMul (p q : Polynomial ℝ) : Polynomial ℝ := sorry

theorem degree_of_composed_and_multiplied_polynomials 
  (f g : Polynomial ℝ) 
  (hf : degree f = 3) 
  (hg : degree g = 7) : 
  degree (polyMul (polyComp f (Polynomial.X^4)) (polyComp g (Polynomial.X^3))) = 33 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_composed_and_multiplied_polynomials_l2681_268140


namespace NUMINAMATH_CALUDE_central_angle_for_given_sector_l2681_268154

/-- A circular sector with given area and perimeter -/
structure CircularSector where
  area : ℝ
  perimeter : ℝ

/-- The central angle of a circular sector in radians -/
def central_angle (s : CircularSector) : ℝ := 
  2 -- We define this as 2, which is what we want to prove

/-- Theorem: For a circular sector with area 1 and perimeter 4, the central angle is 2 radians -/
theorem central_angle_for_given_sector :
  ∀ (s : CircularSector), s.area = 1 ∧ s.perimeter = 4 → central_angle s = 2 := by
  sorry

#check central_angle_for_given_sector

end NUMINAMATH_CALUDE_central_angle_for_given_sector_l2681_268154


namespace NUMINAMATH_CALUDE_volume_of_R_revolution_l2681_268180

-- Define the region R
def R := {(x, y) : ℝ × ℝ | |8 - x| + y ≤ 10 ∧ 3 * y - x ≥ 15}

-- Define the axis of revolution
def axis := {(x, y) : ℝ × ℝ | 3 * y - x = 15}

-- Define the volume of the solid of revolution
def volume_of_revolution (region : Set (ℝ × ℝ)) (axis : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem volume_of_R_revolution :
  volume_of_revolution R axis = (343 * Real.pi) / (12 * Real.sqrt 10) := by sorry

end NUMINAMATH_CALUDE_volume_of_R_revolution_l2681_268180


namespace NUMINAMATH_CALUDE_sequence_c_increasing_l2681_268186

theorem sequence_c_increasing (n : ℕ) : 
  let a : ℕ → ℤ := λ n => 2 * n^2 - 5 * n + 1
  a (n + 1) > a n :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_c_increasing_l2681_268186


namespace NUMINAMATH_CALUDE_same_solution_implies_a_value_l2681_268191

theorem same_solution_implies_a_value :
  ∀ x a : ℚ,
  (3 * x + 5 = 11) →
  (6 * x + 3 * a = 22) →
  a = 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_value_l2681_268191


namespace NUMINAMATH_CALUDE_pool_depth_is_10_feet_l2681_268132

/-- Represents the dimensions and properties of a pool -/
structure Pool where
  width : ℝ
  length : ℝ
  depth : ℝ
  capacity : ℝ
  drainRate : ℝ
  drainTime : ℝ
  initialFillPercentage : ℝ

/-- Calculates the volume of water drained from the pool -/
def volumeDrained (p : Pool) : ℝ := p.drainRate * p.drainTime

/-- Calculates the total capacity of the pool -/
def totalCapacity (p : Pool) : ℝ := p.width * p.length * p.depth

/-- Theorem stating that the depth of the pool is 10 feet -/
theorem pool_depth_is_10_feet (p : Pool) 
  (h1 : p.width = 40)
  (h2 : p.length = 150)
  (h3 : p.drainRate = 60)
  (h4 : p.drainTime = 800)
  (h5 : p.initialFillPercentage = 0.8)
  (h6 : volumeDrained p = p.initialFillPercentage * totalCapacity p) :
  p.depth = 10 := by
  sorry

#check pool_depth_is_10_feet

end NUMINAMATH_CALUDE_pool_depth_is_10_feet_l2681_268132


namespace NUMINAMATH_CALUDE_derivative_pos_implies_increasing_exists_increasing_not_always_pos_derivative_l2681_268134

open Function Real

-- Define a differentiable function f: ℝ → ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for f to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Statement 1: If f'(x) > 0 for all x, then f is increasing
theorem derivative_pos_implies_increasing :
  (∀ x, deriv f x > 0) → IsIncreasing f :=
sorry

-- Statement 2: There exists an increasing f where it's not true that f'(x) > 0 for all x
theorem exists_increasing_not_always_pos_derivative :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ IsIncreasing f ∧ ¬(∀ x, deriv f x > 0) :=
sorry

end NUMINAMATH_CALUDE_derivative_pos_implies_increasing_exists_increasing_not_always_pos_derivative_l2681_268134


namespace NUMINAMATH_CALUDE_typist_salary_problem_l2681_268196

/-- Given a salary that is first increased by 10% and then decreased by 5%,
    resulting in Rs. 2090, prove that the original salary was Rs. 2000. -/
theorem typist_salary_problem (S : ℝ) : 
  S * 1.1 * 0.95 = 2090 → S = 2000 := by
  sorry

#check typist_salary_problem

end NUMINAMATH_CALUDE_typist_salary_problem_l2681_268196


namespace NUMINAMATH_CALUDE_greatest_x_value_l2681_268179

theorem greatest_x_value (x : ℝ) : 
  x ≠ 6 → x ≠ -3 → (x^2 - x - 30) / (x - 6) = 5 / (x + 3) → 
  x ≤ -2 ∧ ∃ y, y = -2 ∧ (y^2 - y - 30) / (y - 6) = 5 / (y + 3) := by
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2681_268179


namespace NUMINAMATH_CALUDE_ahead_of_schedule_l2681_268147

/-- Represents the worker's production plan -/
def WorkerPlan (total_parts : ℕ) (total_days : ℕ) (initial_rate : ℕ) (initial_days : ℕ) (x : ℕ) : Prop :=
  initial_rate * initial_days + (total_days - initial_days) * x > total_parts

/-- Theorem stating the condition for completing the task ahead of schedule -/
theorem ahead_of_schedule (x : ℕ) :
  WorkerPlan 408 15 24 3 x ↔ 24 * 3 + (15 - 3) * x > 408 :=
by sorry

end NUMINAMATH_CALUDE_ahead_of_schedule_l2681_268147


namespace NUMINAMATH_CALUDE_triangle_shape_l2681_268149

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def has_equal_roots (t : Triangle) : Prop :=
  ∃ x : ℝ, t.b * (x^2 + 1) + t.c * (x^2 - 1) - 2 * t.a * x = 0 ∧
  ∀ y : ℝ, t.b * (y^2 + 1) + t.c * (y^2 - 1) - 2 * t.a * y = 0 → y = x

def angle_condition (t : Triangle) : Prop :=
  Real.sin t.C * Real.cos t.A - Real.cos t.C * Real.sin t.A = 0

-- Define an isosceles right-angled triangle
def is_isosceles_right_angled (t : Triangle) : Prop :=
  t.a = t.b ∧ t.A = t.B ∧ t.C = Real.pi / 2

-- State the theorem
theorem triangle_shape (t : Triangle) :
  has_equal_roots t → angle_condition t → is_isosceles_right_angled t := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l2681_268149


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2681_268108

/-- A geometric sequence with a_3 = 2 and a_7 = 8 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ (∀ n, a (n + 1) = r * a n) ∧ a 3 = 2 ∧ a 7 = 8

/-- Theorem: In a geometric sequence where a_3 = 2 and a_7 = 8, a_5 = 4 -/
theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : geometric_sequence a) : a 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2681_268108


namespace NUMINAMATH_CALUDE_identifiable_bulbs_for_two_trips_three_states_max_identifiable_bulbs_is_power_l2681_268118

/-- The maximum number of bulbs and switches that can be identified -/
def max_identifiable_bulbs (n : ℕ) (m : ℕ) : ℕ := m^n

/-- Theorem: With 2 trips and 3 states, 9 bulbs and switches can be identified -/
theorem identifiable_bulbs_for_two_trips_three_states :
  max_identifiable_bulbs 2 3 = 9 := by
  sorry

/-- Theorem: The maximum number of identifiable bulbs is always a power of the number of states -/
theorem max_identifiable_bulbs_is_power (n m : ℕ) :
  ∃ k, max_identifiable_bulbs n m = m^k := by
  sorry

end NUMINAMATH_CALUDE_identifiable_bulbs_for_two_trips_three_states_max_identifiable_bulbs_is_power_l2681_268118


namespace NUMINAMATH_CALUDE_log_equation_solution_l2681_268114

theorem log_equation_solution : 
  ∃ (x : ℝ), (Real.log 729 / Real.log (3 * x) = x) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2681_268114


namespace NUMINAMATH_CALUDE_ramanujan_number_l2681_268193

theorem ramanujan_number (r h : ℂ) : 
  r * h = 40 + 24 * I ∧ h = 7 + I → r = 28/5 + 64/25 * I :=
by sorry

end NUMINAMATH_CALUDE_ramanujan_number_l2681_268193


namespace NUMINAMATH_CALUDE_sum_of_two_5cm_cubes_volume_l2681_268170

/-- The volume of a cube with edge length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The sum of volumes of two cubes with edge length s -/
def sum_of_two_cube_volumes (s : ℝ) : ℝ := 2 * cube_volume s

theorem sum_of_two_5cm_cubes_volume :
  sum_of_two_cube_volumes 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_5cm_cubes_volume_l2681_268170


namespace NUMINAMATH_CALUDE_car_travel_time_fraction_l2681_268101

theorem car_travel_time_fraction (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) 
  (h1 : distance = 432)
  (h2 : original_time = 6)
  (h3 : new_speed = 48) : 
  (distance / new_speed) / original_time = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_time_fraction_l2681_268101


namespace NUMINAMATH_CALUDE_right_triangle_in_sets_l2681_268109

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2

/-- The sets of side lengths given in the problem --/
def side_length_sets : List (ℕ × ℕ × ℕ) :=
  [(5, 4, 3), (1, 2, 3), (5, 6, 7), (2, 2, 3)]

theorem right_triangle_in_sets :
  ∃! (a b c : ℕ), (a, b, c) ∈ side_length_sets ∧ is_right_triangle a b c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_in_sets_l2681_268109


namespace NUMINAMATH_CALUDE_equalizing_amount_is_55_l2681_268110

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The condition when Foma gives Ierema 70 gold coins -/
def condition1 (w : MerchantWealth) : Prop :=
  w.ierema + 70 = w.yuliy

/-- The condition when Foma gives Ierema 40 gold coins -/
def condition2 (w : MerchantWealth) : Prop :=
  w.foma - 40 = w.yuliy

/-- The amount of gold coins Foma should give Ierema to equalize their wealth -/
def equalizingAmount (w : MerchantWealth) : ℕ :=
  (w.foma - w.ierema) / 2

theorem equalizing_amount_is_55 (w : MerchantWealth) 
  (h1 : condition1 w) (h2 : condition2 w) : 
  equalizingAmount w = 55 := by
  sorry

end NUMINAMATH_CALUDE_equalizing_amount_is_55_l2681_268110


namespace NUMINAMATH_CALUDE_projectile_height_l2681_268164

theorem projectile_height (t : ℝ) : 
  t > 0 ∧ -16 * t^2 + 80 * t = 36 ∧ 
  (∀ s, s > 0 ∧ -16 * s^2 + 80 * s = 36 → t ≤ s) → 
  t = 0.5 := by sorry

end NUMINAMATH_CALUDE_projectile_height_l2681_268164


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2681_268150

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 < x + 6 ↔ -2 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2681_268150


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2681_268189

theorem roots_sum_of_squares (m n : ℝ) : 
  (m^2 - 5*m + 3 = 0) → (n^2 - 5*n + 3 = 0) → (m^2 + n^2 = 19) := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2681_268189


namespace NUMINAMATH_CALUDE_largest_ball_radius_largest_ball_touches_plane_largest_ball_on_z_axis_l2681_268174

/-- Represents a torus formed by revolving a circle about the z-axis. -/
structure Torus where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents a spherical ball. -/
structure Ball where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The largest ball that can be positioned on top of the torus. -/
def largest_ball (t : Torus) : Ball :=
  { center := (0, 0, 4),
    radius := 4 }

/-- Theorem stating that the largest ball has radius 4. -/
theorem largest_ball_radius (t : Torus) :
  (t.center = (4, 0, 1) ∧ t.radius = 1) →
  (largest_ball t).radius = 4 := by
  sorry

/-- Theorem stating that the largest ball touches the horizontal plane. -/
theorem largest_ball_touches_plane (t : Torus) :
  (t.center = (4, 0, 1) ∧ t.radius = 1) →
  (largest_ball t).center.2.1 = (largest_ball t).radius := by
  sorry

/-- Theorem stating that the largest ball is centered on the z-axis. -/
theorem largest_ball_on_z_axis (t : Torus) :
  (t.center = (4, 0, 1) ∧ t.radius = 1) →
  (largest_ball t).center.1 = 0 ∧ (largest_ball t).center.2.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_ball_radius_largest_ball_touches_plane_largest_ball_on_z_axis_l2681_268174


namespace NUMINAMATH_CALUDE_center_sum_l2681_268141

/-- The center of a circle defined by the equation x^2 + y^2 = 4x - 6y + 9 -/
def circle_center : ℝ × ℝ := sorry

/-- The equation of the circle -/
axiom circle_equation (p : ℝ × ℝ) : p.1^2 + p.2^2 = 4*p.1 - 6*p.2 + 9

theorem center_sum : circle_center.1 + circle_center.2 = -1 := by sorry

end NUMINAMATH_CALUDE_center_sum_l2681_268141


namespace NUMINAMATH_CALUDE_pagoda_lamps_l2681_268130

theorem pagoda_lamps (n : ℕ) (total : ℕ) (h1 : n = 7) (h2 : total = 381) : 
  (∃ a : ℕ, a * (2^n - 1) = total) → 3 * (2^n - 1) = total := by
sorry

end NUMINAMATH_CALUDE_pagoda_lamps_l2681_268130


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l2681_268100

theorem trig_expression_equals_four : 
  (Real.sqrt 3 * Real.tan (10 * π / 180) + 1) / 
  ((4 * (Real.cos (10 * π / 180))^2 - 2) * Real.sin (10 * π / 180)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l2681_268100


namespace NUMINAMATH_CALUDE_complex_square_equality_l2681_268126

theorem complex_square_equality (a b : ℕ+) :
  (↑a - Complex.I * ↑b) ^ 2 = 15 - 8 * Complex.I →
  ↑a - Complex.I * ↑b = 4 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equality_l2681_268126


namespace NUMINAMATH_CALUDE_polynomial_roots_l2681_268129

theorem polynomial_roots : ∃ (x₁ x₂ x₃ : ℝ), 
  (x₁ = 1 ∧ x₂ = 2 ∧ x₃ = -1) ∧ 
  (∀ x : ℝ, x^4 - 4*x^3 + 3*x^2 + 4*x - 4 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2681_268129


namespace NUMINAMATH_CALUDE_gumballs_last_42_days_l2681_268197

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_pair : ℕ := 9

/-- The number of pairs of earrings Kim brings on day 1 -/
def earrings_day1 : ℕ := 3

/-- The number of pairs of earrings Kim brings on day 2 -/
def earrings_day2 : ℕ := 2 * earrings_day1

/-- The number of pairs of earrings Kim brings on day 3 -/
def earrings_day3 : ℕ := earrings_day2 - 1

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The total number of gumballs Kim receives -/
def total_gumballs : ℕ := 
  gumballs_per_pair * (earrings_day1 + earrings_day2 + earrings_day3)

/-- The number of days the gumballs will last -/
def days_gumballs_last : ℕ := total_gumballs / gumballs_eaten_per_day

theorem gumballs_last_42_days : days_gumballs_last = 42 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_last_42_days_l2681_268197


namespace NUMINAMATH_CALUDE_intersection_is_empty_l2681_268117

def A : Set ℝ := {x | x^2 - 2*x > 0}
def B : Set ℝ := {x | |x + 1| < 0}

theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l2681_268117


namespace NUMINAMATH_CALUDE_s_range_l2681_268128

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def divisible_by_11 (n : ℕ) : Prop := ∃ k, n = 11 * k

def s (n : ℕ) : ℕ := sorry

theorem s_range (n : ℕ) (h_composite : is_composite n) (h_div11 : divisible_by_11 n) :
  ∃ (m : ℕ), m ≥ 11 ∧ s n = m ∧ ∀ (k : ℕ), k ≥ 11 → ∃ (p : ℕ), is_composite p ∧ divisible_by_11 p ∧ s p = k :=
sorry

end NUMINAMATH_CALUDE_s_range_l2681_268128


namespace NUMINAMATH_CALUDE_max_attendees_is_three_tuesday_has_three_friday_has_three_saturday_has_three_no_day_exceeds_three_l2681_268105

-- Define the days of the week
inductive Day
| Mon | Tues | Wed | Thurs | Fri | Sat

-- Define the people
inductive Person
| Amy | Bob | Charlie | Diana | Evan

-- Define the availability function
def available : Person → Day → Bool
| Person.Amy, Day.Mon => false
| Person.Amy, Day.Tues => true
| Person.Amy, Day.Wed => false
| Person.Amy, Day.Thurs => false
| Person.Amy, Day.Fri => true
| Person.Amy, Day.Sat => true
| Person.Bob, Day.Mon => true
| Person.Bob, Day.Tues => false
| Person.Bob, Day.Wed => true
| Person.Bob, Day.Thurs => true
| Person.Bob, Day.Fri => false
| Person.Bob, Day.Sat => true
| Person.Charlie, Day.Mon => false
| Person.Charlie, Day.Tues => false
| Person.Charlie, Day.Wed => false
| Person.Charlie, Day.Thurs => true
| Person.Charlie, Day.Fri => true
| Person.Charlie, Day.Sat => false
| Person.Diana, Day.Mon => true
| Person.Diana, Day.Tues => true
| Person.Diana, Day.Wed => false
| Person.Diana, Day.Thurs => false
| Person.Diana, Day.Fri => true
| Person.Diana, Day.Sat => false
| Person.Evan, Day.Mon => false
| Person.Evan, Day.Tues => true
| Person.Evan, Day.Wed => true
| Person.Evan, Day.Thurs => false
| Person.Evan, Day.Fri => false
| Person.Evan, Day.Sat => true

-- Count the number of available people for a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (λ p => available p d) [Person.Amy, Person.Bob, Person.Charlie, Person.Diana, Person.Evan]).length

-- Find the maximum number of available people across all days
def maxAvailable : Nat :=
  List.foldl max 0 (List.map countAvailable [Day.Mon, Day.Tues, Day.Wed, Day.Thurs, Day.Fri, Day.Sat])

-- Theorem: The maximum number of attendees is 3
theorem max_attendees_is_three : maxAvailable = 3 := by sorry

-- Theorem: Tuesday has 3 attendees
theorem tuesday_has_three : countAvailable Day.Tues = 3 := by sorry

-- Theorem: Friday has 3 attendees
theorem friday_has_three : countAvailable Day.Fri = 3 := by sorry

-- Theorem: Saturday has 3 attendees
theorem saturday_has_three : countAvailable Day.Sat = 3 := by sorry

-- Theorem: No other day has more than 3 attendees
theorem no_day_exceeds_three : ∀ d : Day, countAvailable d ≤ 3 := by sorry

end NUMINAMATH_CALUDE_max_attendees_is_three_tuesday_has_three_friday_has_three_saturday_has_three_no_day_exceeds_three_l2681_268105


namespace NUMINAMATH_CALUDE_prob_three_same_color_l2681_268156

/-- A deck of cards with red and black colors -/
structure Deck :=
  (total : ℕ)
  (red : ℕ)
  (black : ℕ)
  (h1 : red + black = total)

/-- The probability of drawing three cards of the same color -/
def prob_same_color (d : Deck) : ℚ :=
  2 * (d.red.choose 3 / d.total.choose 3)

/-- The specific deck described in the problem -/
def modified_deck : Deck :=
  { total := 60
  , red := 30
  , black := 30
  , h1 := by simp }

/-- The main theorem stating the probability for the given deck -/
theorem prob_three_same_color :
  prob_same_color modified_deck = 406 / 1711 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_same_color_l2681_268156


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2681_268113

theorem complex_fraction_simplification :
  (5 - Complex.I) / (1 - Complex.I) = 3 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2681_268113
