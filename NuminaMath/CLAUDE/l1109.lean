import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_length_l1109_110981

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  A = π/4 →
  2*b*(Real.sin B) - c*(Real.sin C) = 2*a*(Real.sin A) →
  (1/2)*b*c*(Real.sin A) = 3 →
  c = 2*(Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1109_110981


namespace NUMINAMATH_CALUDE_weights_missing_l1109_110911

/-- Represents a set of weights with equal quantities of each type -/
structure WeightSet where
  quantity : ℕ
  total_mass : ℕ

/-- The weight set described in the problem -/
def problem_weights : WeightSet :=
  { quantity := 0,  -- We don't know the exact quantity, so we use 0 as a placeholder
    total_mass := 606060606060 }  -- Assuming the pattern repeats 6 times for illustration

/-- Theorem stating that at least one weight is missing and more than 10 weights are missing -/
theorem weights_missing (w : WeightSet) :
  (w.total_mass % 72 ≠ 0) ∧ 
  (∃ (a b : ℕ), a + b > 10 ∧ 5*a + 43*b ≡ w.total_mass [MOD 24]) :=
by sorry

end NUMINAMATH_CALUDE_weights_missing_l1109_110911


namespace NUMINAMATH_CALUDE_inequality_proof_l1109_110933

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.rpow (a^2 / (b + c)^2) (1/3) + 
  Real.rpow (b^2 / (c + a)^2) (1/3) + 
  Real.rpow (c^2 / (a + b)^2) (1/3) ≥ 
  3 / Real.rpow 4 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1109_110933


namespace NUMINAMATH_CALUDE_line_m_equation_l1109_110919

-- Define the plane
def Plane := ℝ × ℝ

-- Define a line in the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in the plane
def Point := Plane

-- Define the given lines
def ℓ : Line := { a := 2, b := -5, c := 0 }
def m : Line := { a := 5, b := 2, c := 0 }

-- Define the given points
def Q : Point := (3, -2)
def Q'' : Point := (-2, 3)

-- Define the reflection operation
def reflect (p : Point) (L : Line) : Point := sorry

-- State the theorem
theorem line_m_equation :
  ∃ (Q' : Point),
    reflect Q m = Q' ∧
    reflect Q' ℓ = Q'' ∧
    m.a = 5 ∧ m.b = 2 ∧ m.c = 0 := by sorry

end NUMINAMATH_CALUDE_line_m_equation_l1109_110919


namespace NUMINAMATH_CALUDE_smallest_number_l1109_110943

/-- Converts a number from base b to decimal -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The given numbers in their respective bases -/
def number_a : List Nat := [3, 3]
def number_b : List Nat := [0, 1, 1, 1]
def number_c : List Nat := [2, 2, 1]
def number_d : List Nat := [1, 2]

theorem smallest_number :
  to_decimal number_d 5 < to_decimal number_a 4 ∧
  to_decimal number_d 5 < to_decimal number_b 2 ∧
  to_decimal number_d 5 < to_decimal number_c 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1109_110943


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1109_110903

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * Real.sqrt 7 * x

-- Define the asymptote passing through (2, √3)
def asymptote_through_point (a b : ℝ) : Prop :=
  b / a = Real.sqrt 3 / 2

-- Define the focus on the directrix condition
def focus_on_directrix (c : ℝ) : Prop :=
  c = Real.sqrt 7

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : asymptote_through_point a b)
  (h4 : ∃ c, focus_on_directrix c ∧ a^2 + b^2 = c^2) :
  ∀ x y, hyperbola a b x y ↔ x^2 / 4 - y^2 / 3 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1109_110903


namespace NUMINAMATH_CALUDE_seating_theorem_l1109_110936

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to seat athletes from three teams in a row, with teammates seated together -/
def seating_arrangements (team_a : ℕ) (team_b : ℕ) (team_c : ℕ) : ℕ :=
  factorial 3 * factorial team_a * factorial team_b * factorial team_c

theorem seating_theorem :
  seating_arrangements 4 3 3 = 5184 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l1109_110936


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l1109_110942

theorem coefficient_x4_in_expansion : 
  let expansion := (fun x => (2 * x + 1) * (x - 3)^5)
  ∃ (a b c d e f : ℤ), 
    (∀ x, expansion x = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f) ∧
    b = 165 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l1109_110942


namespace NUMINAMATH_CALUDE_percentage_with_both_pets_l1109_110955

def total_students : ℕ := 40
def puppy_percentage : ℚ := 80 / 100
def both_pets : ℕ := 8

theorem percentage_with_both_pets : 
  (both_pets : ℚ) / (puppy_percentage * total_students) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_with_both_pets_l1109_110955


namespace NUMINAMATH_CALUDE_degree_of_h_l1109_110994

/-- Given a polynomial f(x) = -5x^5 + 2x^4 + 7x - 8 and a polynomial h(x) such that
    the degree of f(x) - h(x) is 3, prove that the degree of h(x) is 5. -/
theorem degree_of_h (f h : Polynomial ℝ) : 
  f = -5 * X^5 + 2 * X^4 + 7 * X - 8 →
  Polynomial.degree (f - h) = 3 →
  Polynomial.degree h = 5 :=
by sorry

end NUMINAMATH_CALUDE_degree_of_h_l1109_110994


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l1109_110999

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (γ : ℝ) 
  (ha : a = 7) 
  (hb : b = 8) 
  (hγ : γ = 2 * π / 3) -- 120° in radians
  (hc : c^2 = a^2 + b^2 - 2*a*b*Real.cos γ) : -- Law of Cosines
  c = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l1109_110999


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1109_110941

/-- Given a geometric sequence {a_n} where a_1 = 3 and 4a_1, 2a_2, a_3 form an arithmetic sequence,
    prove that a_3 + a_4 + a_5 = 84. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 3 →  -- First term
  4 * a 1 - 2 * a 2 = 2 * a 2 - a 3 →  -- Arithmetic sequence condition
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1109_110941


namespace NUMINAMATH_CALUDE_angle_Y_value_l1109_110963

-- Define the angles as real numbers
def A : ℝ := 50
def Z : ℝ := 50

-- Define the theorem
theorem angle_Y_value :
  ∀ B X Y : ℝ,
  A + B = 180 →
  X = Y →
  B + Z = 180 →
  B + X + Y = 180 →
  Y = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_Y_value_l1109_110963


namespace NUMINAMATH_CALUDE_victor_score_l1109_110971

/-- 
Given a maximum mark and a percentage score, calculate the actual score.
-/
def calculateScore (maxMark : ℕ) (percentage : ℚ) : ℚ :=
  percentage * maxMark

theorem victor_score :
  let maxMark : ℕ := 300
  let percentage : ℚ := 80 / 100
  calculateScore maxMark percentage = 240 := by
  sorry

end NUMINAMATH_CALUDE_victor_score_l1109_110971


namespace NUMINAMATH_CALUDE_jacob_dimes_l1109_110992

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coin_value "penny" + nickels * coin_value "nickel" + dimes * coin_value "dime"

theorem jacob_dimes (mrs_hilt_pennies mrs_hilt_nickels mrs_hilt_dimes : ℕ)
                    (jacob_pennies jacob_nickels : ℕ)
                    (difference : ℕ) :
  mrs_hilt_pennies = 2 →
  mrs_hilt_nickels = 2 →
  mrs_hilt_dimes = 2 →
  jacob_pennies = 4 →
  jacob_nickels = 1 →
  difference = 13 →
  ∃ jacob_dimes : ℕ,
    total_value mrs_hilt_pennies mrs_hilt_nickels mrs_hilt_dimes -
    total_value jacob_pennies jacob_nickels jacob_dimes = difference ∧
    jacob_dimes = 1 :=
by sorry

end NUMINAMATH_CALUDE_jacob_dimes_l1109_110992


namespace NUMINAMATH_CALUDE_simplify_fraction_l1109_110922

theorem simplify_fraction (x y : ℚ) (hx : x = 2) (hy : y = 5) :
  15 * x^3 * y^2 / (10 * x^2 * y^4) = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1109_110922


namespace NUMINAMATH_CALUDE_smallest_multiple_with_100_divisors_l1109_110956

/-- The number of positive integral divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is a multiple of m -/
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_multiple_with_100_divisors :
  ∃ m : ℕ,
    m > 0 ∧
    is_multiple m 100 ∧
    num_divisors m = 100 ∧
    (∀ k : ℕ, k > 0 → is_multiple k 100 → num_divisors k = 100 → m ≤ k) ∧
    m / 100 = 324 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_100_divisors_l1109_110956


namespace NUMINAMATH_CALUDE_equal_share_theorem_l1109_110959

/-- Represents the number of stickers each person has -/
structure Stickers where
  kate : ℝ
  jenna : ℝ
  ava : ℝ

/-- The ratio of stickers between Kate, Jenna, and Ava -/
def sticker_ratio : Stickers := { kate := 7.5, jenna := 4.25, ava := 5.75 }

/-- Kate's actual number of stickers -/
def kate_stickers : ℝ := 45

/-- Calculates the total number of stickers -/
def total_stickers (s : Stickers) : ℝ := s.kate + s.jenna + s.ava

/-- Theorem stating that when the stickers are equally shared, each person gets 35 stickers -/
theorem equal_share_theorem (s : Stickers) :
  s.kate / sticker_ratio.kate = s.jenna / sticker_ratio.jenna ∧
  s.kate / sticker_ratio.kate = s.ava / sticker_ratio.ava ∧
  s.kate = kate_stickers →
  (total_stickers s) / 3 = 35 := by sorry

end NUMINAMATH_CALUDE_equal_share_theorem_l1109_110959


namespace NUMINAMATH_CALUDE_repeat2016_product_of_palindromes_l1109_110934

/-- A natural number is a palindrome if it reads the same forwards and backwards. -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- Repeats the digits 2016 n times to form a natural number. -/
def repeat2016 (n : ℕ) : ℕ := sorry

/-- Theorem: Any number formed by repeating 2016 n times is the product of two palindromes. -/
theorem repeat2016_product_of_palindromes (n : ℕ) (h : n ≥ 1) :
  ∃ (a b : ℕ), isPalindrome a ∧ isPalindrome b ∧ repeat2016 n = a * b := by sorry

end NUMINAMATH_CALUDE_repeat2016_product_of_palindromes_l1109_110934


namespace NUMINAMATH_CALUDE_point_outside_circle_l1109_110912

/-- A line intersects a circle at two distinct points if and only if 
    the distance from the circle's center to the line is less than the radius -/
axiom line_intersects_circle_iff_distance_lt_radius 
  (a b : ℝ) : (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    a * x₁ + b * y₁ = 4 ∧ x₁^2 + y₁^2 = 4 ∧
    a * x₂ + b * y₂ = 4 ∧ x₂^2 + y₂^2 = 4) ↔
  (4 / Real.sqrt (a^2 + b^2) < 2)

/-- The distance from a point to the origin is greater than 2 
    if and only if the point is outside the circle with radius 2 centered at the origin -/
axiom outside_circle_iff_distance_gt_radius 
  (a b : ℝ) : Real.sqrt (a^2 + b^2) > 2 ↔ (a, b) ∉ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}

theorem point_outside_circle (a b : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    a * x₁ + b * y₁ = 4 ∧ x₁^2 + y₁^2 = 4 ∧
    a * x₂ + b * y₂ = 4 ∧ x₂^2 + y₂^2 = 4) →
  (a, b) ∉ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1109_110912


namespace NUMINAMATH_CALUDE_unique_n_with_conditions_l1109_110914

theorem unique_n_with_conditions :
  ∃! n : ℕ,
    50 ≤ n ∧ n ≤ 150 ∧
    7 ∣ n ∧
    n % 9 = 3 ∧
    n % 6 = 3 ∧
    n % 11 = 5 ∧
    n = 109 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_with_conditions_l1109_110914


namespace NUMINAMATH_CALUDE_line_through_ellipse_midpoint_l1109_110975

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on the given ellipse -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 25 + p.y^2 / 16 = 1

/-- Checks if a point lies on the given line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

theorem line_through_ellipse_midpoint (M A B : Point) (l : Line) :
  isOnLine M l →
  isOnEllipse A →
  isOnEllipse B →
  isOnLine A l →
  isOnLine B l →
  isMidpoint M A B →
  M.x = 1 →
  M.y = 2 →
  l.a = 8 ∧ l.b = 25 ∧ l.c = -58 := by
  sorry


end NUMINAMATH_CALUDE_line_through_ellipse_midpoint_l1109_110975


namespace NUMINAMATH_CALUDE_find_divisor_l1109_110973

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 127)
  (h2 : quotient = 5)
  (h3 : remainder = 2)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 25 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1109_110973


namespace NUMINAMATH_CALUDE_wallet_theorem_l1109_110969

def wallet_problem (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) (twenty_dollar_bills : ℕ) : Prop :=
  let total_amount : ℕ := 150
  let ten_dollar_amount : ℕ := 50
  let twenty_dollar_count : ℕ := 4
  (5 * five_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills = total_amount) ∧
  (10 * ten_dollar_bills = ten_dollar_amount) ∧
  (twenty_dollar_bills = twenty_dollar_count) ∧
  (five_dollar_bills + ten_dollar_bills + twenty_dollar_bills = 13)

theorem wallet_theorem :
  ∃ (five_dollar_bills ten_dollar_bills twenty_dollar_bills : ℕ),
    wallet_problem five_dollar_bills ten_dollar_bills twenty_dollar_bills :=
by
  sorry

end NUMINAMATH_CALUDE_wallet_theorem_l1109_110969


namespace NUMINAMATH_CALUDE_recurring_decimal_division_l1109_110985

theorem recurring_decimal_division :
  let a : ℚ := 36 / 99
  let b : ℚ := 12 / 99
  a / b = 3 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_division_l1109_110985


namespace NUMINAMATH_CALUDE_positive_interval_for_quadratic_l1109_110904

theorem positive_interval_for_quadratic (x : ℝ) :
  (x + 1) * (x - 3) > 0 ↔ x < -1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_interval_for_quadratic_l1109_110904


namespace NUMINAMATH_CALUDE_simplify_fraction_l1109_110977

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1109_110977


namespace NUMINAMATH_CALUDE_tiling_theorem_l1109_110923

/-- Represents a tile on the board -/
inductive Tile
  | SmallTile : Tile  -- 1 x 3 tile
  | LargeTile : Tile  -- 2 x 2 tile

/-- Represents the position of the 2 x 2 tile -/
inductive LargeTilePosition
  | Central : LargeTilePosition
  | Corner : LargeTilePosition

/-- Represents a board configuration -/
structure Board :=
  (size : Nat)
  (largeTilePos : LargeTilePosition)

/-- Checks if a board can be tiled -/
def canBeTiled (b : Board) : Prop :=
  match b.largeTilePos with
  | LargeTilePosition.Central => true
  | LargeTilePosition.Corner => false

/-- The main theorem to be proved -/
theorem tiling_theorem (b : Board) (h : b.size = 10000) :
  canBeTiled b ↔ b.largeTilePos = LargeTilePosition.Central :=
sorry

end NUMINAMATH_CALUDE_tiling_theorem_l1109_110923


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l1109_110901

theorem min_value_parallel_vectors (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![4 - n, m]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  (∀ x y, x > 0 → y > 0 → x / y + y / x ≥ 2) →
  (n / m + 8 / n ≥ 6) ∧ (∃ m₀ n₀, m₀ > 0 ∧ n₀ > 0 ∧ n₀ / m₀ + 8 / n₀ = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l1109_110901


namespace NUMINAMATH_CALUDE_sam_book_purchase_l1109_110905

theorem sam_book_purchase (initial_amount : ℕ) (book_cost : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : book_cost = 7)
  (h3 : remaining_amount = 16) :
  (initial_amount - remaining_amount) / book_cost = 9 := by
  sorry

end NUMINAMATH_CALUDE_sam_book_purchase_l1109_110905


namespace NUMINAMATH_CALUDE_selection_schemes_eq_240_l1109_110982

-- Define the number of people and cities
def total_people : ℕ := 6
def total_cities : ℕ := 4

-- Define the function to calculate the number of selection schemes
def selection_schemes : ℕ :=
  -- Options for city A (excluding person A and B)
  (total_people - 2) *
  -- Options for city B
  (total_people - 1) *
  -- Options for city C
  (total_people - 2) *
  -- Options for city D
  (total_people - 3)

-- Theorem to prove
theorem selection_schemes_eq_240 : selection_schemes = 240 := by
  sorry

end NUMINAMATH_CALUDE_selection_schemes_eq_240_l1109_110982


namespace NUMINAMATH_CALUDE_exponent_division_equality_l1109_110997

theorem exponent_division_equality (a b : ℝ) :
  (a^2 * b)^3 / ((-a * b)^2) = a^4 * b :=
by sorry

end NUMINAMATH_CALUDE_exponent_division_equality_l1109_110997


namespace NUMINAMATH_CALUDE_cost_of_12_pencils_9_notebooks_l1109_110979

/-- The cost of a single pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a single notebook -/
def notebook_cost : ℝ := sorry

/-- The first given condition: 9 pencils and 6 notebooks cost $3.21 -/
axiom condition1 : 9 * pencil_cost + 6 * notebook_cost = 3.21

/-- The second given condition: 8 pencils and 5 notebooks cost $2.84 -/
axiom condition2 : 8 * pencil_cost + 5 * notebook_cost = 2.84

/-- Theorem: The cost of 12 pencils and 9 notebooks is $4.32 -/
theorem cost_of_12_pencils_9_notebooks : 
  12 * pencil_cost + 9 * notebook_cost = 4.32 := by sorry

end NUMINAMATH_CALUDE_cost_of_12_pencils_9_notebooks_l1109_110979


namespace NUMINAMATH_CALUDE_box_volume_l1109_110925

theorem box_volume (x y z : ℝ) 
  (h1 : 2*x + 2*y = 26) 
  (h2 : x + z = 10) 
  (h3 : y + z = 7) : 
  x * y * z = 80 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l1109_110925


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l1109_110962

theorem gift_wrapping_combinations : 
  let wrapping_paper := 8
  let ribbon := 5
  let gift_card := 4
  let gift_sticker := 6
  wrapping_paper * ribbon * gift_card * gift_sticker = 960 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l1109_110962


namespace NUMINAMATH_CALUDE_paper_clip_collection_l1109_110996

theorem paper_clip_collection (num_boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : num_boxes = 9) (h2 : clips_per_box = 9) : 
  num_boxes * clips_per_box = 81 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_collection_l1109_110996


namespace NUMINAMATH_CALUDE_binomial_product_minus_240_l1109_110948

theorem binomial_product_minus_240 : 
  (Nat.choose 10 3) * (Nat.choose 8 3) - 240 = 6480 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_minus_240_l1109_110948


namespace NUMINAMATH_CALUDE_teacher_age_l1109_110930

theorem teacher_age (num_students : ℕ) (student_avg : ℝ) (new_avg : ℝ) : 
  num_students = 50 → 
  student_avg = 14 → 
  new_avg = 15 → 
  (num_students : ℝ) * student_avg + (new_avg * (num_students + 1) - num_students * student_avg) = 65 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l1109_110930


namespace NUMINAMATH_CALUDE_presents_difference_l1109_110966

def ethan_presents : ℝ := 31.0
def alissa_presents : ℕ := 9

theorem presents_difference : ethan_presents - alissa_presents = 22 := by
  sorry

end NUMINAMATH_CALUDE_presents_difference_l1109_110966


namespace NUMINAMATH_CALUDE_red_light_probability_l1109_110953

-- Define the durations of each light
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

-- Define the total cycle time
def total_cycle_time : ℕ := red_duration + yellow_duration + green_duration

-- Define the probability of seeing a red light
def probability_red_light : ℚ := red_duration / total_cycle_time

-- Theorem statement
theorem red_light_probability :
  probability_red_light = 30 / 75 :=
by sorry

end NUMINAMATH_CALUDE_red_light_probability_l1109_110953


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l1109_110908

/-- Proves that for a right circular cone and a sphere with the same radius,
    if the volume of the cone is one-third that of the sphere,
    then the ratio of the cone's altitude to its base radius is 4/3. -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * π * r^2 * h) = (1 / 3 * (4 / 3 * π * r^3)) →
  h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l1109_110908


namespace NUMINAMATH_CALUDE_starting_player_winning_strategy_l1109_110928

/-- Represents the color of a disk -/
inductive DiskColor
| Red
| Blue

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the state of the chessboard -/
def BoardState (n : Nat) := Position → DiskColor

/-- Checks if a position is within the bounds of the board -/
def isValidPosition (n : Nat) (pos : Position) : Prop :=
  pos.row < n ∧ pos.col < n

/-- Represents a move in the game -/
structure Move :=
  (pos : Position)

/-- Applies a move to the board state -/
def applyMove (n : Nat) (state : BoardState n) (move : Move) : BoardState n :=
  sorry

/-- Checks if a player can make a move -/
def canMove (n : Nat) (state : BoardState n) : Prop :=
  ∃ (move : Move), isValidPosition n move.pos ∧ state move.pos = DiskColor.Blue

/-- Defines a winning strategy for the starting player -/
def hasWinningStrategy (n : Nat) (initialState : BoardState n) : Prop :=
  sorry

/-- The main theorem stating the winning condition for the starting player -/
theorem starting_player_winning_strategy (n : Nat) (initialState : BoardState n) :
  hasWinningStrategy n initialState ↔ 
  initialState ⟨n - 1, n - 1⟩ = DiskColor.Blue :=
sorry

end NUMINAMATH_CALUDE_starting_player_winning_strategy_l1109_110928


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1109_110960

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 + 2*a - 3 = 0) → 
  (b^3 - 2*b^2 + 2*b - 3 = 0) → 
  (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1109_110960


namespace NUMINAMATH_CALUDE_number_representation_proof_l1109_110947

theorem number_representation_proof (n a b c : ℕ) : 
  (n = 14^2 * a + 14 * b + c) →
  (n = 15^2 * a + 15 * c + b) →
  (n = 6^3 * a + 6^2 * c + 6 * a + c) →
  (a > 0) →
  (a < 6 ∧ b < 14 ∧ c < 6) →
  (n = 925) := by
sorry

end NUMINAMATH_CALUDE_number_representation_proof_l1109_110947


namespace NUMINAMATH_CALUDE_only_c_is_perfect_square_l1109_110920

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2

def number_a : ℕ := 4^4 * 5^5 * 6^6
def number_b : ℕ := 4^4 * 5^6 * 6^5
def number_c : ℕ := 4^5 * 5^4 * 6^6
def number_d : ℕ := 4^6 * 5^4 * 6^5
def number_e : ℕ := 4^6 * 5^5 * 6^4

theorem only_c_is_perfect_square :
  ¬(is_perfect_square number_a) ∧
  ¬(is_perfect_square number_b) ∧
  is_perfect_square number_c ∧
  ¬(is_perfect_square number_d) ∧
  ¬(is_perfect_square number_e) :=
sorry

end NUMINAMATH_CALUDE_only_c_is_perfect_square_l1109_110920


namespace NUMINAMATH_CALUDE_intersection_M_N_l1109_110900

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1109_110900


namespace NUMINAMATH_CALUDE_possible_x_value_for_simplest_radical_l1109_110907

/-- A number is a simplest quadratic radical if it's of the form √n where n is a positive integer
    and not a perfect square. -/
def is_simplest_quadratic_radical (n : ℝ) : Prop :=
  ∃ (m : ℕ), n = Real.sqrt m ∧ ¬ ∃ (k : ℕ), m = k^2

/-- The proposition states that 2 is a possible value for x that makes √(x+3) 
    the simplest quadratic radical. -/
theorem possible_x_value_for_simplest_radical : 
  ∃ (x : ℝ), is_simplest_quadratic_radical (Real.sqrt (x + 3)) ∧ x = 2 :=
sorry

end NUMINAMATH_CALUDE_possible_x_value_for_simplest_radical_l1109_110907


namespace NUMINAMATH_CALUDE_derivative_at_pi_over_two_l1109_110916

open Real

theorem derivative_at_pi_over_two (f : ℝ → ℝ) (hf : ∀ x, f x = sin x + 2 * x * (deriv f 0)) :
  deriv f (π / 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_over_two_l1109_110916


namespace NUMINAMATH_CALUDE_tan_sum_pi_third_l1109_110913

theorem tan_sum_pi_third (x : ℝ) (h : Real.tan x = 3) :
  Real.tan (x + π / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_third_l1109_110913


namespace NUMINAMATH_CALUDE_polynomial_root_implies_h_value_l1109_110978

theorem polynomial_root_implies_h_value :
  ∀ h : ℝ, ((-2 : ℝ)^3 + h * (-2) - 12 = 0) → h = -10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_h_value_l1109_110978


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1109_110990

/-- Given a quadratic equation kx^2 - 2(k+1)x + k-1 = 0 with two distinct real roots, 
    this theorem proves properties about the range of k and the sum of reciprocals of roots. -/
theorem quadratic_equation_properties (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k*x₁^2 - 2*(k+1)*x₁ + (k-1) = 0 ∧ k*x₂^2 - 2*(k+1)*x₂ + (k-1) = 0) →
  (k > -1/3 ∧ k ≠ 0) ∧
  ¬(∃ k : ℝ, ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → k*x₁^2 - 2*(k+1)*x₁ + (k-1) = 0 → k*x₂^2 - 2*(k+1)*x₂ + (k-1) = 0 → 
    1/x₁ + 1/x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1109_110990


namespace NUMINAMATH_CALUDE_min_young_rank_is_11_l1109_110918

/-- Yuna's rank in the running event -/
def yuna_rank : ℕ := 6

/-- The number of people who finished between Yuna and Min-Young -/
def people_between : ℕ := 5

/-- Min-Young's rank in the running event -/
def min_young_rank : ℕ := yuna_rank + people_between

/-- Theorem stating Min-Young's rank -/
theorem min_young_rank_is_11 : min_young_rank = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_young_rank_is_11_l1109_110918


namespace NUMINAMATH_CALUDE_line_parameter_range_l1109_110970

/-- Given two points on opposite sides of a line, prove the range of the line's parameter. -/
theorem line_parameter_range (m : ℝ) : 
  (∀ (x y : ℝ), 2*x + y + m = 0 → 
    ((x = 1 ∧ y = 3) ∨ (x = -4 ∧ y = -2)) →
    (2*1 + 3 + m) * (2*(-4) + (-2) + m) < 0) →
  -5 < m ∧ m < 10 :=
sorry

end NUMINAMATH_CALUDE_line_parameter_range_l1109_110970


namespace NUMINAMATH_CALUDE_double_area_right_triangle_l1109_110980

/-- The area of a triangle with double the area of a right-angled triangle -/
theorem double_area_right_triangle (a b : ℝ) : 
  let triangle_I_base : ℝ := a + b
  let triangle_I_height : ℝ := a + b
  let triangle_I_area : ℝ := (1 / 2) * triangle_I_base * triangle_I_height
  let triangle_II_area : ℝ := 2 * triangle_I_area
  triangle_II_area = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_double_area_right_triangle_l1109_110980


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l1109_110926

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmingScenario where
  v_m : ℝ  -- Speed of the man in still water (km/h)
  v_s : ℝ  -- Speed of the stream (km/h)

/-- Theorem stating that given the downstream and upstream swimming distances and times,
    the speed of the swimmer in still water is 12 km/h. -/
theorem swimmer_speed_in_still_water 
  (scenario : SwimmingScenario)
  (h_downstream : (scenario.v_m + scenario.v_s) * 3 = 54)
  (h_upstream : (scenario.v_m - scenario.v_s) * 3 = 18) :
  scenario.v_m = 12 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l1109_110926


namespace NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l1109_110906

/-- The probability of getting an odd number on a single roll of a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 6

/-- The number of times we want to get an odd number -/
def target_odd : ℕ := 5

/-- The probability of getting exactly 'target_odd' odd numbers in 'num_rolls' rolls -/
def prob_target_odd : ℚ :=
  (Nat.choose num_rolls target_odd : ℚ) * prob_odd ^ target_odd * (1 - prob_odd) ^ (num_rolls - target_odd)

theorem prob_five_odd_in_six_rolls : prob_target_odd = 3/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l1109_110906


namespace NUMINAMATH_CALUDE_scientific_notation_equals_original_number_l1109_110935

def scientific_notation_value : ℝ := 6.7 * (10 ^ 6)

theorem scientific_notation_equals_original_number : scientific_notation_value = 6700000 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equals_original_number_l1109_110935


namespace NUMINAMATH_CALUDE_power_of_power_l1109_110988

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1109_110988


namespace NUMINAMATH_CALUDE_youngest_son_cookies_value_l1109_110902

/-- The number of cookies in a box -/
def total_cookies : ℕ := 54

/-- The number of days the box lasts -/
def days : ℕ := 9

/-- The number of cookies the oldest son gets each day -/
def oldest_son_cookies : ℕ := 4

/-- The number of cookies the youngest son gets each day -/
def youngest_son_cookies : ℕ := (total_cookies - (oldest_son_cookies * days)) / days

theorem youngest_son_cookies_value : youngest_son_cookies = 2 := by
  sorry

end NUMINAMATH_CALUDE_youngest_son_cookies_value_l1109_110902


namespace NUMINAMATH_CALUDE_no_fibonacci_right_triangle_l1109_110991

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Theorem: No right-angled triangle has all sides as Fibonacci numbers -/
theorem no_fibonacci_right_triangle (n : ℕ) : 
  (fib n)^2 + (fib (n + 1))^2 ≠ (fib (n + 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_no_fibonacci_right_triangle_l1109_110991


namespace NUMINAMATH_CALUDE_complex_norm_squared_l1109_110993

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 5 + 4*Complex.I) : 
  Complex.abs z^2 = 41/10 := by
sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l1109_110993


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1109_110940

theorem quadratic_roots_relation (m p q : ℝ) (hm : m ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -q ∧ s₁ * s₂ = m) ∧
               (3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = p)) →
  p / q = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1109_110940


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1109_110939

theorem magnitude_of_z : ∀ z : ℂ, z = (Complex.abs (2 + Complex.I) + 2 * Complex.I) / Complex.I → Complex.abs z = 3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1109_110939


namespace NUMINAMATH_CALUDE_first_donor_amount_l1109_110983

theorem first_donor_amount (d1 d2 d3 d4 : ℝ) 
  (h1 : d2 = 2 * d1)
  (h2 : d3 = 3 * d2)
  (h3 : d4 = 4 * d3)
  (h4 : d1 + d2 + d3 + d4 = 132) :
  d1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_donor_amount_l1109_110983


namespace NUMINAMATH_CALUDE_green_ball_probability_l1109_110915

/-- Represents a container with balls -/
structure Container where
  green : ℕ
  red : ℕ

/-- The probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.green + c.red)

/-- The containers given in the problem -/
def containers : List Container := [
  ⟨8, 2⟩,  -- Container A
  ⟨6, 4⟩,  -- Container B
  ⟨5, 5⟩,  -- Container C
  ⟨8, 2⟩   -- Container D
]

/-- The number of containers -/
def numContainers : ℕ := containers.length

/-- The theorem stating the probability of selecting a green ball -/
theorem green_ball_probability : 
  (1 / numContainers) * (containers.map greenProbability).sum = 43 / 160 := by
  sorry


end NUMINAMATH_CALUDE_green_ball_probability_l1109_110915


namespace NUMINAMATH_CALUDE_cone_volume_l1109_110931

/-- Given a cone with slant height 3 and lateral area 3√5π, its volume is 10π/3 -/
theorem cone_volume (l : ℝ) (L : ℝ) (r : ℝ) (h : ℝ) (V : ℝ) : 
  l = 3 →
  L = 3 * Real.sqrt 5 * Real.pi →
  L = Real.pi * r * l →
  l^2 = r^2 + h^2 →
  V = (1/3) * Real.pi * r^2 * h →
  V = (10/3) * Real.pi := by
sorry


end NUMINAMATH_CALUDE_cone_volume_l1109_110931


namespace NUMINAMATH_CALUDE_pi_digits_ratio_l1109_110989

/-- The number of digits of pi memorized by Carlos -/
def carlos_digits : ℕ := sorry

/-- The number of digits of pi memorized by Sam -/
def sam_digits : ℕ := sorry

/-- The number of digits of pi memorized by Mina -/
def mina_digits : ℕ := sorry

/-- The ratio of digits memorized by Mina to Carlos -/
def mina_carlos_ratio : ℚ := sorry

theorem pi_digits_ratio :
  sam_digits = carlos_digits + 6 ∧
  mina_digits = 24 ∧
  sam_digits = 10 ∧
  ∃ k : ℕ, mina_digits = k * carlos_digits →
  mina_carlos_ratio = 6 := by sorry

end NUMINAMATH_CALUDE_pi_digits_ratio_l1109_110989


namespace NUMINAMATH_CALUDE_geometric_progression_common_ratio_l1109_110944

theorem geometric_progression_common_ratio 
  (x y z w : ℝ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) 
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0) 
  (h_geom_prog : ∃ (a r : ℝ), r ≠ 0 ∧ 
    x * (y - z) = a ∧ 
    y * (z - x) = a * r ∧ 
    z * (x - y) = a * r^2 ∧ 
    w * (x - y) = a * r^3) : 
  ∃ r : ℝ, r^3 + r^2 + r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_common_ratio_l1109_110944


namespace NUMINAMATH_CALUDE_dice_probability_l1109_110995

def standard_dice : ℕ := 5
def special_dice : ℕ := 5
def standard_sides : ℕ := 6
def special_sides : ℕ := 3  -- Only even numbers (2, 4, 6)

def probability_standard_one : ℚ := 1 / 6
def probability_standard_not_one : ℚ := 5 / 6
def probability_special_four : ℚ := 1 / 3
def probability_special_not_four : ℚ := 2 / 3

theorem dice_probability : 
  (Nat.choose standard_dice 1 : ℚ) * probability_standard_one * probability_standard_not_one ^ 4 *
  (Nat.choose special_dice 1 : ℚ) * probability_special_four * probability_special_not_four ^ 4 =
  250000 / 1889568 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l1109_110995


namespace NUMINAMATH_CALUDE_leg_lengths_are_3_or_4_l1109_110984

/-- Represents an isosceles triangle with integer side lengths --/
structure IsoscelesTriangle where
  base : ℕ
  leg : ℕ
  sum_eq_10 : base + 2 * leg = 10

/-- The set of possible leg lengths for an isosceles triangle formed from a 10cm wire --/
def possible_leg_lengths : Set ℕ :=
  {l | ∃ t : IsoscelesTriangle, t.leg = l}

/-- Theorem stating that the only possible leg lengths are 3 and 4 --/
theorem leg_lengths_are_3_or_4 : possible_leg_lengths = {3, 4} := by
  sorry

#check leg_lengths_are_3_or_4

end NUMINAMATH_CALUDE_leg_lengths_are_3_or_4_l1109_110984


namespace NUMINAMATH_CALUDE_product_of_n_values_product_of_possible_n_values_l1109_110937

-- Define the temperatures at noon
def temp_minneapolis (n : ℝ) (l : ℝ) : ℝ := l + n
def temp_stlouis (l : ℝ) : ℝ := l

-- Define the temperatures at 4:00 PM
def temp_minneapolis_4pm (n : ℝ) (l : ℝ) : ℝ := temp_minneapolis n l - 7
def temp_stlouis_4pm (l : ℝ) : ℝ := temp_stlouis l + 5

-- Define the temperature difference at 4:00 PM
def temp_diff_4pm (n : ℝ) (l : ℝ) : ℝ := |temp_minneapolis_4pm n l - temp_stlouis_4pm l|

-- Theorem statement
theorem product_of_n_values (n : ℝ) (l : ℝ) :
  (temp_diff_4pm n l = 4) → (n = 16 ∨ n = 8) ∧ (16 * 8 = 128) := by
  sorry

-- Main theorem
theorem product_of_possible_n_values : 
  ∃ (n₁ n₂ : ℝ), (n₁ ≠ n₂) ∧ (∀ l : ℝ, temp_diff_4pm n₁ l = 4 ∧ temp_diff_4pm n₂ l = 4) ∧ (n₁ * n₂ = 128) := by
  sorry

end NUMINAMATH_CALUDE_product_of_n_values_product_of_possible_n_values_l1109_110937


namespace NUMINAMATH_CALUDE_constant_term_of_given_equation_l1109_110924

/-- The quadratic equation 2x^2 - 3x - 1 = 0 -/
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 3 * x - 1 = 0

/-- The constant term of a quadratic equation ax^2 + bx + c = 0 is c -/
def constant_term (a b c : ℝ) : ℝ := c

theorem constant_term_of_given_equation :
  constant_term 2 (-3) (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_given_equation_l1109_110924


namespace NUMINAMATH_CALUDE_cost_price_of_toy_cost_price_is_800_l1109_110965

/-- The cost price of a toy given the selling price and gain conditions -/
theorem cost_price_of_toy (total_sale : ℕ) (num_toys : ℕ) (gain_in_toys : ℕ) : ℕ :=
  let selling_price := total_sale / num_toys
  let cost_price := selling_price / (1 + gain_in_toys / num_toys)
  cost_price
  
/-- Proof that the cost price of a toy is 800 given the conditions -/
theorem cost_price_is_800 : cost_price_of_toy 16800 18 3 = 800 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_toy_cost_price_is_800_l1109_110965


namespace NUMINAMATH_CALUDE_teacher_age_l1109_110998

/-- Given a class of students and their teacher, this theorem proves the teacher's age
    based on how the average age changes when including the teacher. -/
theorem teacher_age (num_students : ℕ) (student_avg_age teacher_age : ℝ) 
    (h1 : num_students = 25)
    (h2 : student_avg_age = 26)
    (h3 : (num_students * student_avg_age + teacher_age) / (num_students + 1) = student_avg_age + 1) :
  teacher_age = 52 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l1109_110998


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1109_110932

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (100 - x) = 9 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1109_110932


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l1109_110929

/-- Given an isosceles triangle with perimeter 24 and base 10, prove the leg length is 7 -/
theorem isosceles_triangle_leg_length 
  (perimeter : ℝ) 
  (base : ℝ) 
  (leg : ℝ) 
  (h1 : perimeter = 24) 
  (h2 : base = 10) 
  (h3 : perimeter = base + 2 * leg) : 
  leg = 7 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l1109_110929


namespace NUMINAMATH_CALUDE_no_line_exists_l1109_110961

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the intersection points of the line and the parabola
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ parabola x y ∧ line_through_focus m x y}

-- Define the distance from a point to the line x = -2
def distance_to_line (x y : ℝ) : ℝ := x + 2

-- Statement to prove
theorem no_line_exists :
  ¬ ∃ m : ℝ, ∃ A B : ℝ × ℝ,
    A ∈ intersection_points m ∧
    B ∈ intersection_points m ∧
    A ≠ B ∧
    distance_to_line A.1 A.2 + distance_to_line B.1 B.2 = 5 :=
sorry

end NUMINAMATH_CALUDE_no_line_exists_l1109_110961


namespace NUMINAMATH_CALUDE_total_books_sold_l1109_110921

/-- Represents the sales data for a salesperson over 5 days -/
structure SalesData where
  monday : Float
  tuesday_multiplier : Float
  wednesday_multiplier : Float
  friday_multiplier : Float

/-- Calculates the total books sold by a salesperson over 5 days -/
def total_sales (data : SalesData) : Float :=
  let tuesday := data.monday * data.tuesday_multiplier
  let wednesday := tuesday * data.wednesday_multiplier
  data.monday + tuesday + wednesday + data.monday + (data.monday * data.friday_multiplier)

/-- Theorem stating the total books sold by all three salespeople -/
theorem total_books_sold (matias_data olivia_data luke_data : SalesData) 
  (h_matias : matias_data = { monday := 7, tuesday_multiplier := 2.5, wednesday_multiplier := 3.5, friday_multiplier := 4.2 })
  (h_olivia : olivia_data = { monday := 5, tuesday_multiplier := 1.5, wednesday_multiplier := 2.2, friday_multiplier := 3 })
  (h_luke : luke_data = { monday := 12, tuesday_multiplier := 0.75, wednesday_multiplier := 1.5, friday_multiplier := 0.8 }) :
  total_sales matias_data + total_sales olivia_data + total_sales luke_data = 227.75 := by
  sorry


end NUMINAMATH_CALUDE_total_books_sold_l1109_110921


namespace NUMINAMATH_CALUDE_abc_sum_difference_l1109_110987

theorem abc_sum_difference (a b c : ℝ) 
  (hab : |a - b| = 1)
  (hbc : |b - c| = 1)
  (hca : |c - a| = 2)
  (habc : a * b * c = 60) :
  a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_difference_l1109_110987


namespace NUMINAMATH_CALUDE_water_consumption_rate_l1109_110974

/-- 
Given a person drinks water at a rate of 1 cup every 20 minutes,
prove that they will drink 11.25 cups in 225 minutes.
-/
theorem water_consumption_rate (drinking_rate : ℚ) (time : ℚ) (cups : ℚ) : 
  drinking_rate = 1 / 20 → time = 225 → cups = time * drinking_rate → cups = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_rate_l1109_110974


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1109_110909

theorem solve_linear_equation :
  ∃ x : ℤ, 9773 + x = 13200 ∧ x = 3427 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1109_110909


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l1109_110986

theorem concert_ticket_cost (current_amount : ℕ) (amount_needed : ℕ) (num_tickets : ℕ) : 
  current_amount = 189 →
  amount_needed = 171 →
  num_tickets = 4 →
  (current_amount + amount_needed) / num_tickets = 90 := by
sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l1109_110986


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l1109_110954

def first_n_even_sum (n : ℕ) : ℕ := n * (n + 1)

def first_n_odd_sum (n : ℕ) : ℕ := n^2

theorem even_odd_sum_difference :
  first_n_even_sum 1500 - first_n_odd_sum 1500 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l1109_110954


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1109_110927

theorem divisibility_equivalence (a b c d : ℤ) (h : a ≠ c) :
  (∃ k : ℤ, a * b + c * d = k * (a - c)) ↔ (∃ m : ℤ, a * d + b * c = m * (a - c)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1109_110927


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_interval_l1109_110945

/-- The function f(x) defined in terms of a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * |2 * a - x| + 2 * x

/-- The theorem stating that f(x) is increasing on ℝ if and only if a ∈ [-1, 1] -/
theorem f_increasing_iff_a_in_interval (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (-1 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_interval_l1109_110945


namespace NUMINAMATH_CALUDE_gcf_twenty_pair_l1109_110958

theorem gcf_twenty_pair : ∃! (a b : ℕ), 
  ((a = 200 ∧ b = 2000) ∨ 
   (a = 40 ∧ b = 50) ∨ 
   (a = 20 ∧ b = 40) ∨ 
   (a = 20 ∧ b = 25)) ∧ 
  Nat.gcd a b = 20 :=
by sorry

end NUMINAMATH_CALUDE_gcf_twenty_pair_l1109_110958


namespace NUMINAMATH_CALUDE_cube_sum_is_18_l1109_110952

/-- Represents the arrangement of numbers on a cube's vertices -/
def CubeArrangement := Fin 8 → Fin 9

/-- The sum of numbers on a face of the cube -/
def face_sum (arrangement : CubeArrangement) (face : Finset (Fin 8)) : ℕ :=
  (face.sum fun v => arrangement v).val + 1

/-- Predicate for a valid cube arrangement -/
def is_valid_arrangement (arrangement : CubeArrangement) : Prop :=
  ∀ (face1 face2 : Finset (Fin 8)), face1.card = 4 → face2.card = 4 → 
    face_sum arrangement face1 = face_sum arrangement face2

theorem cube_sum_is_18 :
  ∀ (arrangement : CubeArrangement), is_valid_arrangement arrangement →
    ∃ (face : Finset (Fin 8)), face.card = 4 ∧ face_sum arrangement face = 18 :=
sorry

end NUMINAMATH_CALUDE_cube_sum_is_18_l1109_110952


namespace NUMINAMATH_CALUDE_farmer_feed_cost_l1109_110964

theorem farmer_feed_cost (total_spent : ℝ) (chicken_feed_percent : ℝ) (chicken_discount : ℝ) : 
  total_spent = 35 →
  chicken_feed_percent = 0.4 →
  chicken_discount = 0.5 →
  let chicken_feed_cost := total_spent * chicken_feed_percent
  let goat_feed_cost := total_spent * (1 - chicken_feed_percent)
  let full_price_chicken_feed := chicken_feed_cost / (1 - chicken_discount)
  let full_price_total := full_price_chicken_feed + goat_feed_cost
  full_price_total = 49 := by sorry

end NUMINAMATH_CALUDE_farmer_feed_cost_l1109_110964


namespace NUMINAMATH_CALUDE_libor_lucky_numbers_l1109_110938

theorem libor_lucky_numbers :
  {n : ℕ | n < 1000 ∧ 7 ∣ n^2 ∧ 8 ∣ n^2 ∧ 9 ∣ n^2 ∧ 10 ∣ n^2} = {420, 840} :=
by sorry

end NUMINAMATH_CALUDE_libor_lucky_numbers_l1109_110938


namespace NUMINAMATH_CALUDE_eunji_has_most_marbles_l1109_110910

def minyoung_marbles : ℕ := 4
def yujeong_marbles : ℕ := 2
def eunji_marbles : ℕ := minyoung_marbles + 1

theorem eunji_has_most_marbles :
  eunji_marbles > minyoung_marbles ∧ eunji_marbles > yujeong_marbles :=
by
  sorry

end NUMINAMATH_CALUDE_eunji_has_most_marbles_l1109_110910


namespace NUMINAMATH_CALUDE_truncated_pyramid_surface_area_l1109_110946

/-- The total surface area of a truncated right pyramid with given dimensions --/
theorem truncated_pyramid_surface_area
  (base_side : ℝ)
  (upper_side : ℝ)
  (height : ℝ)
  (h_base : base_side = 15)
  (h_upper : upper_side = 10)
  (h_height : height = 20) :
  let slant_height := Real.sqrt (height^2 + ((base_side - upper_side) / 2)^2)
  let lateral_area := 2 * (base_side + upper_side) * slant_height
  let base_area := base_side^2 + upper_side^2
  lateral_area + base_area = 1332.8 :=
by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_surface_area_l1109_110946


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1109_110976

theorem reciprocal_of_negative_2023 :
  ∃ (x : ℚ), x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1109_110976


namespace NUMINAMATH_CALUDE_range_of_m_for_equation_l1109_110951

theorem range_of_m_for_equation (P : Prop) 
  (h : P ↔ ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x+1) + m = 0) : 
  P → ∀ m : ℝ, (∃ x : ℝ, 4^x - 2^(x+1) + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_equation_l1109_110951


namespace NUMINAMATH_CALUDE_range_of_fraction_l1109_110968

theorem range_of_fraction (x y : ℝ) (h : x^2 + y^2 + 2*x = 0) :
  ∃ (t : ℝ), y / (x - 1) = t ∧ -Real.sqrt 3 / 3 ≤ t ∧ t ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1109_110968


namespace NUMINAMATH_CALUDE_seonyeong_class_size_l1109_110949

/-- The number of rows of students -/
def num_rows : ℕ := 12

/-- The number of students in each row -/
def students_per_row : ℕ := 4

/-- The number of additional students -/
def additional_students : ℕ := 3

/-- The number of students in Jieun's class -/
def jieun_class_size : ℕ := 12

/-- The total number of students -/
def total_students : ℕ := num_rows * students_per_row + additional_students

/-- Theorem: The number of students in Seonyeong's class is 39 -/
theorem seonyeong_class_size : total_students - jieun_class_size = 39 := by
  sorry

end NUMINAMATH_CALUDE_seonyeong_class_size_l1109_110949


namespace NUMINAMATH_CALUDE_discount_card_saves_money_l1109_110917

-- Define the cost of the discount card
def discount_card_cost : ℝ := 100

-- Define the discount percentage
def discount_percentage : ℝ := 0.03

-- Define the cost of cakes
def cake_cost : ℝ := 500

-- Define the number of cakes
def num_cakes : ℕ := 4

-- Define the cost of fruits
def fruit_cost : ℝ := 1600

-- Calculate the total cost without discount
def total_cost_without_discount : ℝ := cake_cost * num_cakes + fruit_cost

-- Calculate the discounted amount
def discounted_amount : ℝ := total_cost_without_discount * discount_percentage

-- Calculate the total cost with discount
def total_cost_with_discount : ℝ := 
  total_cost_without_discount - discounted_amount + discount_card_cost

-- Theorem to prove that buying the discount card saves money
theorem discount_card_saves_money : 
  total_cost_with_discount < total_cost_without_discount :=
by sorry

end NUMINAMATH_CALUDE_discount_card_saves_money_l1109_110917


namespace NUMINAMATH_CALUDE_percentage_equation_l1109_110950

theorem percentage_equation (x : ℝ) : (35 / 100 * 400 = 20 / 100 * x) → x = 700 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_l1109_110950


namespace NUMINAMATH_CALUDE_coin_combinations_eq_20_l1109_110957

/-- The number of combinations of pennies, nickels, and quarters that sum to 50 cents -/
def coin_combinations : Nat :=
  (Finset.filter (fun (p, n, q) => p + 5 * n + 25 * q = 50)
    (Finset.product (Finset.range 51)
      (Finset.product (Finset.range 11) (Finset.range 3)))).card

/-- Theorem stating that the number of coin combinations is 20 -/
theorem coin_combinations_eq_20 : coin_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_coin_combinations_eq_20_l1109_110957


namespace NUMINAMATH_CALUDE_max_value_fraction_l1109_110967

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 ∧ 1 ≤ y' ∧ y' ≤ 3 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = 0.4 := by
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1109_110967


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l1109_110972

theorem sum_of_specific_numbers : 1235 + 2351 + 3512 + 5123 = 12221 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l1109_110972
