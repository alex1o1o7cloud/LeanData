import Mathlib

namespace NUMINAMATH_CALUDE_walter_school_allocation_l1490_149083

/-- Represents Walter's work and allocation details -/
structure WorkDetails where
  daysPerWeek : ℕ
  hourlyWage : ℚ
  hoursPerDay : ℕ
  allocationRatio : ℚ

/-- Calculates the amount allocated for school based on work details -/
def schoolAllocation (w : WorkDetails) : ℚ :=
  w.daysPerWeek * w.hourlyWage * w.hoursPerDay * w.allocationRatio

/-- Theorem stating that Walter's school allocation is $75 -/
theorem walter_school_allocation :
  let w : WorkDetails := {
    daysPerWeek := 5,
    hourlyWage := 5,
    hoursPerDay := 4,
    allocationRatio := 3/4
  }
  schoolAllocation w = 75 := by sorry

end NUMINAMATH_CALUDE_walter_school_allocation_l1490_149083


namespace NUMINAMATH_CALUDE_max_radius_in_wine_glass_l1490_149041

theorem max_radius_in_wine_glass :
  let f : ℝ → ℝ := λ x ↦ x^4
  let max_r : ℝ := (3/4) * Real.rpow 2 (1/3)
  ∀ r > 0,
    (∀ x y : ℝ, (y - r)^2 + x^2 = r^2 → y ≥ f x) ∧
    (0 - r)^2 + 0^2 = r^2 →
    r ≤ max_r :=
by sorry

end NUMINAMATH_CALUDE_max_radius_in_wine_glass_l1490_149041


namespace NUMINAMATH_CALUDE_hexagon_midpoint_area_l1490_149057

-- Define the hexagon
def regular_hexagon (side_length : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the set of line segments
def line_segments (h : Set (ℝ × ℝ)) : Set (ℝ × ℝ × ℝ × ℝ) := sorry

-- Define the midpoints of the line segments
def midpoints (segments : Set (ℝ × ℝ × ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Define the area enclosed by the midpoints
def enclosed_area (points : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem hexagon_midpoint_area :
  let h := regular_hexagon 3
  let s := line_segments h
  let m := midpoints s
  let a := enclosed_area m
  ∃ ε > 0, abs (a - 1.85) < ε := by sorry

end NUMINAMATH_CALUDE_hexagon_midpoint_area_l1490_149057


namespace NUMINAMATH_CALUDE_l₃_is_symmetric_to_l₁_l1490_149050

/-- The equation of line l₁ -/
def l₁ (x y : ℝ) : Prop := x - 2 * y - 2 = 0

/-- The equation of line l₂ -/
def l₂ (x y : ℝ) : Prop := x + y = 0

/-- The equation of line l₃ -/
def l₃ (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- A point is symmetric to another point with respect to l₂ -/
def symmetric_wrt_l₂ (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -y₁ ∧ y₂ = -x₁

theorem l₃_is_symmetric_to_l₁ :
  ∀ x y : ℝ, l₃ x y ↔ ∃ x₁ y₁ : ℝ, l₁ x₁ y₁ ∧ symmetric_wrt_l₂ x y x₁ y₁ :=
sorry

end NUMINAMATH_CALUDE_l₃_is_symmetric_to_l₁_l1490_149050


namespace NUMINAMATH_CALUDE_not_divisible_by_4p_l1490_149066

theorem not_divisible_by_4p (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ¬ (4 * p ∣ (2 * p - 1)^(p - 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_4p_l1490_149066


namespace NUMINAMATH_CALUDE_OM_range_theorem_l1490_149015

-- Define the line equation
def line_eq (m n x y : ℝ) : Prop := 2 * m * x - (4 * m + n) * y + 2 * n = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 6)

-- Define that m and n are not simultaneously zero
def not_zero (m n : ℝ) : Prop := m ≠ 0 ∨ n ≠ 0

-- Define the perpendicular line passing through P
def perp_line (m n : ℝ) (M : ℝ × ℝ) : Prop :=
  line_eq m n M.1 M.2 ∧ 
  (M.1 - point_P.1) * (2 * m) + (M.2 - point_P.2) * (-(4 * m + n)) = 0

-- Define the range of |OM|
def OM_range (x : ℝ) : Prop := 5 - Real.sqrt 5 ≤ x ∧ x ≤ 5 + Real.sqrt 5

-- Theorem statement
theorem OM_range_theorem (m n : ℝ) (M : ℝ × ℝ) :
  not_zero m n →
  perp_line m n M →
  OM_range (Real.sqrt (M.1^2 + M.2^2)) :=
sorry

end NUMINAMATH_CALUDE_OM_range_theorem_l1490_149015


namespace NUMINAMATH_CALUDE_diagonal_intersection_coincidence_l1490_149071

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by its four vertices -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Predicate to check if a quadrilateral is circumscribed around a circle -/
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Function to get the tangency points of a circumscribed quadrilateral -/
def tangency_points (q : Quadrilateral) (c : Circle) : 
  (Point × Point × Point × Point) := sorry

/-- Function to get the intersection point of two diagonals -/
def diagonal_intersection (q : Quadrilateral) : Point := sorry

/-- The main theorem -/
theorem diagonal_intersection_coincidence 
  (q : Quadrilateral) (c : Circle) 
  (h : is_circumscribed q c) : 
  let (E, F, G, K) := tangency_points q c
  let q' := Quadrilateral.mk E F G K
  diagonal_intersection q = diagonal_intersection q' := by sorry

end NUMINAMATH_CALUDE_diagonal_intersection_coincidence_l1490_149071


namespace NUMINAMATH_CALUDE_derivative_implies_limit_l1490_149010

theorem derivative_implies_limit (f : ℝ → ℝ) (x₀ a : ℝ) (h : HasDerivAt f a x₀) :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx, 0 < |Δx| → |Δx| < δ →
    |(f (x₀ + Δx) - f (x₀ - Δx)) / Δx - 2*a| < ε :=
by sorry

end NUMINAMATH_CALUDE_derivative_implies_limit_l1490_149010


namespace NUMINAMATH_CALUDE_common_chord_length_l1490_149007

theorem common_chord_length (r₁ r₂ d : ℝ) (h₁ : r₁ = 8) (h₂ : r₂ = 12) (h₃ : d = 20) :
  let chord_length := 2 * Real.sqrt (r₂^2 - (d/2)^2)
  chord_length = 4 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_l1490_149007


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_l1490_149097

theorem smallest_factorizable_b : ∃ (b : ℕ),
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 2016 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), b' < b →
    ¬(∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b'*x + 2016 = (x + p) * (x + q))) ∧
  b = 90 := by
sorry

end NUMINAMATH_CALUDE_smallest_factorizable_b_l1490_149097


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_divisible_by_12_l1490_149069

theorem consecutive_integers_sum_divisible_by_12 (a b c d : ℤ) :
  (b = a + 1) → (c = b + 1) → (d = c + 1) →
  ∃ k : ℤ, ab + ac + ad + bc + bd + cd + 1 = 12 * k :=
by
  sorry
where
  ab := a * b
  ac := a * c
  ad := a * d
  bc := b * c
  bd := b * d
  cd := c * d

end NUMINAMATH_CALUDE_consecutive_integers_sum_divisible_by_12_l1490_149069


namespace NUMINAMATH_CALUDE_spells_conversion_l1490_149044

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The number of spells in base 9 -/
def spellsBase9 : List Nat := [7, 4, 5]

theorem spells_conversion :
  base9ToBase10 spellsBase9 = 448 := by
  sorry

end NUMINAMATH_CALUDE_spells_conversion_l1490_149044


namespace NUMINAMATH_CALUDE_area_of_region_R_approx_l1490_149012

/-- Represents a rhombus ABCD -/
structure Rhombus :=
  (side_length : ℝ)
  (angle_B : ℝ)

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem statement -/
theorem area_of_region_R_approx (r : Rhombus) :
  r.side_length = 4 ∧ r.angle_B = π/3 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |area (region_R r) - 3| < ε :=
sorry

end NUMINAMATH_CALUDE_area_of_region_R_approx_l1490_149012


namespace NUMINAMATH_CALUDE_megans_work_hours_l1490_149096

/-- Megan's work problem -/
theorem megans_work_hours
  (hourly_rate : ℝ)
  (days_per_month : ℕ)
  (total_earnings : ℝ)
  (h : hourly_rate = 7.5)
  (d : days_per_month = 20)
  (e : total_earnings = 2400) :
  ∃ (hours_per_day : ℝ),
    hours_per_day * hourly_rate * (2 * days_per_month) = total_earnings ∧
    hours_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_megans_work_hours_l1490_149096


namespace NUMINAMATH_CALUDE_least_k_divisible_by_1260_two_ten_divisible_by_1260_least_k_is_210_l1490_149043

theorem least_k_divisible_by_1260 (k : ℕ) : k > 0 ∧ k^4 % 1260 = 0 → k ≥ 210 := by
  sorry

theorem two_ten_divisible_by_1260 : (210 : ℕ)^4 % 1260 = 0 := by
  sorry

theorem least_k_is_210 : ∃ k : ℕ, k > 0 ∧ k^4 % 1260 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m^4 % 1260 = 0) → m ≥ k :=
  ⟨210, by
    sorry⟩

end NUMINAMATH_CALUDE_least_k_divisible_by_1260_two_ten_divisible_by_1260_least_k_is_210_l1490_149043


namespace NUMINAMATH_CALUDE_negate_negative_equals_positive_l1490_149002

theorem negate_negative_equals_positive (n : ℤ) : -(-n) = n := by
  sorry

end NUMINAMATH_CALUDE_negate_negative_equals_positive_l1490_149002


namespace NUMINAMATH_CALUDE_third_month_sale_l1490_149099

def sale_month1 : ℕ := 6435
def sale_month2 : ℕ := 6927
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 7391
def average_sale : ℕ := 6900
def num_months : ℕ := 6

theorem third_month_sale :
  ∃ (sale_month3 : ℕ),
    sale_month3 = num_months * average_sale - (sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month3 = 6855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l1490_149099


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l1490_149036

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) : 
  total = 36 → difference = 6 → 
  ∃ (girls boys : ℕ), 
    girls + boys = total ∧ 
    girls = boys + difference ∧
    girls * 5 = boys * 7 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l1490_149036


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l1490_149051

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = π) :
  8 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l1490_149051


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l1490_149021

/-- Represents a four-digit number --/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Extracts the thousands digit from a four-digit number --/
def thousandsDigit (n : FourDigitNumber) : ℕ := n.val / 1000

/-- Extracts the hundreds digit from a four-digit number --/
def hundredsDigit (n : FourDigitNumber) : ℕ := (n.val / 100) % 10

/-- Extracts the tens digit from a four-digit number --/
def tensDigit (n : FourDigitNumber) : ℕ := (n.val / 10) % 10

/-- Extracts the units digit from a four-digit number --/
def unitsDigit (n : FourDigitNumber) : ℕ := n.val % 10

/-- Checks if a natural number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem unique_four_digit_square : 
  ∃! (n : FourDigitNumber), 
    isPerfectSquare n.val ∧ 
    thousandsDigit n = tensDigit n ∧ 
    hundredsDigit n = unitsDigit n + 1 ∧
    n.val = 8281 :=
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l1490_149021


namespace NUMINAMATH_CALUDE_largest_power_of_ten_in_factorial_l1490_149098

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def count_multiples (n : ℕ) (d : ℕ) : ℕ := n / d

def count_factors_of_five (n : ℕ) : ℕ :=
  (count_multiples n 5) + (count_multiples n 25) + (count_multiples n 125)

theorem largest_power_of_ten_in_factorial :
  (∀ k : ℕ, k ≤ 41 → (factorial 170) % (10^k) = 0) ∧
  ¬((factorial 170) % (10^42) = 0) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_ten_in_factorial_l1490_149098


namespace NUMINAMATH_CALUDE_square_vertex_locus_l1490_149034

/-- Represents a line in 2D plane with equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with vertices A, B, C, D and center O -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point
  O : Point

def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem square_vertex_locus 
  (a c o : Line) 
  (h_not_parallel : a.a * c.b ≠ a.b * c.a) :
  ∃ (F G H : ℝ),
    ∀ (ABCD : Square),
      on_line ABCD.A a → 
      on_line ABCD.C c → 
      on_line ABCD.O o → 
      (on_line ABCD.B ⟨F, G, H⟩ ∧ on_line ABCD.D ⟨F, G, H⟩) :=
sorry

end NUMINAMATH_CALUDE_square_vertex_locus_l1490_149034


namespace NUMINAMATH_CALUDE_triangle_area_l1490_149039

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    prove that its area is 15√3 under certain conditions. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b < c →
  2 * a * c * Real.cos C + 2 * c^2 * Real.cos A = a + c →
  2 * c * Real.sin A - Real.sqrt 3 * a = 0 →
  let S := (1 / 2) * a * b * Real.sin C
  S = 15 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1490_149039


namespace NUMINAMATH_CALUDE_girls_combined_avg_is_76_l1490_149006

-- Define the schools
inductive School
| Cedar
| Dale

-- Define the student types
inductive StudentType
| Boy
| Girl

-- Define the average score function
def avg_score (s : School) (st : StudentType) : ℝ :=
  match s, st with
  | School.Cedar, StudentType.Boy => 65
  | School.Cedar, StudentType.Girl => 70
  | School.Dale, StudentType.Boy => 75
  | School.Dale, StudentType.Girl => 82

-- Define the combined average score function for each school
def combined_avg_score (s : School) : ℝ :=
  match s with
  | School.Cedar => 68
  | School.Dale => 78

-- Define the combined average score for boys at both schools
def combined_boys_avg : ℝ := 73

-- Theorem to prove
theorem girls_combined_avg_is_76 :
  ∃ (c d : ℝ), c > 0 ∧ d > 0 ∧
  (c * avg_score School.Cedar StudentType.Boy + d * avg_score School.Dale StudentType.Boy) / (c + d) = combined_boys_avg ∧
  (c * combined_avg_score School.Cedar + d * combined_avg_score School.Dale) / (c + d) = (c * avg_score School.Cedar StudentType.Girl + d * avg_score School.Dale StudentType.Girl) / (c + d) ∧
  (avg_score School.Cedar StudentType.Girl + avg_score School.Dale StudentType.Girl) / 2 = 76 :=
sorry

end NUMINAMATH_CALUDE_girls_combined_avg_is_76_l1490_149006


namespace NUMINAMATH_CALUDE_jamies_coins_l1490_149068

/-- Represents the number of coins of each type -/
structure CoinCounts where
  quarters : ℚ
  nickels : ℚ
  dimes : ℚ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCounts) : ℚ :=
  25 * coins.quarters + 5 * coins.nickels + 10 * coins.dimes

/-- Theorem stating the solution to Jamie's coin problem -/
theorem jamies_coins :
  ∃ (coins : CoinCounts),
    coins.nickels = 2 * coins.quarters ∧
    coins.dimes = coins.quarters ∧
    totalValue coins = 1520 ∧
    coins.quarters = 304/9 ∧
    coins.nickels = 608/9 ∧
    coins.dimes = 304/9 := by
  sorry

end NUMINAMATH_CALUDE_jamies_coins_l1490_149068


namespace NUMINAMATH_CALUDE_a_100_value_l1490_149014

/-- Sequence S defined recursively -/
def S : ℕ → ℚ
| 0 => 0
| 1 => 3
| (n + 2) => 3 / (3 * n + 1)

/-- Sequence a defined in terms of S -/
def a : ℕ → ℚ
| 0 => 0
| 1 => 3
| (n + 2) => (3 * (S (n + 2))^2) / (3 * S (n + 2) - 2)

/-- Main theorem: a₁₀₀ = -9/84668 -/
theorem a_100_value : a 100 = -9/84668 := by sorry

end NUMINAMATH_CALUDE_a_100_value_l1490_149014


namespace NUMINAMATH_CALUDE_max_right_triangle_area_in_rectangle_l1490_149054

theorem max_right_triangle_area_in_rectangle (a b : ℝ) (ha : a = 12) (hb : b = 15) :
  ∃ (area : ℝ), area = 90 ∧ 
  ∀ (x y z : ℝ), 
    0 ≤ x ∧ x ≤ a ∧ 
    0 ≤ y ∧ y ≤ b ∧ 
    x^2 + y^2 = z^2 ∧ 
    z ≤ (a^2 + b^2)^(1/2) →
    (1/2) * x * y ≤ area :=
sorry

end NUMINAMATH_CALUDE_max_right_triangle_area_in_rectangle_l1490_149054


namespace NUMINAMATH_CALUDE_article_cost_price_l1490_149095

theorem article_cost_price (C S : ℝ) : 
  (S = 1.05 * C) →                    -- Condition 1
  (S - 5 = 1.1 * (0.95 * C)) →        -- Condition 2
  C = 1000 :=                         -- Conclusion
by sorry

end NUMINAMATH_CALUDE_article_cost_price_l1490_149095


namespace NUMINAMATH_CALUDE_least_k_equals_2_pow_q_l1490_149009

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Given an even positive integer n, this function returns the least k₀ such that
    k₀ = f(x) · (x+1)^n + g(x) · (x^n + 1) for some polynomials f(x) and g(x) with integer coefficients -/
noncomputable def least_k (n : ℕ) : ℕ :=
  sorry

theorem least_k_equals_2_pow_q (n : ℕ) (q r : ℕ) (hn : Even n) (hq : Odd q) (hnqr : n = q * 2^r) :
  least_k n = 2^q :=
by sorry

end NUMINAMATH_CALUDE_least_k_equals_2_pow_q_l1490_149009


namespace NUMINAMATH_CALUDE_a_not_zero_l1490_149072

theorem a_not_zero 
  (a b c d : ℝ) 
  (h1 : a / b < -3 * c / d) 
  (h2 : b * d ≠ 0) 
  (h3 : c = 2 * a) : 
  a ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_a_not_zero_l1490_149072


namespace NUMINAMATH_CALUDE_unique_solution_l1490_149047

theorem unique_solution : ∃! x : ℝ, 4 * x - 3 = 9 * (x - 7) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1490_149047


namespace NUMINAMATH_CALUDE_triangle_side_range_l1490_149092

-- Define an acute-angled triangle with side lengths 2, 4, and x
def is_acute_triangle (x : ℝ) : Prop :=
  0 < x ∧ x < 2 + 4 ∧ 2 < 4 + x ∧ 4 < 2 + x ∧
  (2^2 + 4^2 > x^2) ∧ (2^2 + x^2 > 4^2) ∧ (4^2 + x^2 > 2^2)

-- Theorem statement
theorem triangle_side_range :
  ∀ x : ℝ, is_acute_triangle x → (2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1490_149092


namespace NUMINAMATH_CALUDE_power_tower_at_three_l1490_149046

theorem power_tower_at_three : 
  let x : ℕ := 3
  (x^x)^(x^x) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_tower_at_three_l1490_149046


namespace NUMINAMATH_CALUDE_inequality_and_constraint_solution_l1490_149062

-- Define the inequality and its solution set
def inequality (a : ℝ) (x : ℝ) : Prop := 2 * a * x^2 - 8 * x - 3 * a^2 < 0
def solution_set (a b : ℝ) : Set ℝ := {x | -1 < x ∧ x < b ∧ inequality a x}

-- Define the constraint equation
def constraint (a b x y : ℝ) : Prop := a / x + b / y = 1

-- State the theorem
theorem inequality_and_constraint_solution :
  ∃ (a b : ℝ),
    (∀ x, x ∈ solution_set a b ↔ inequality a x) ∧
    a > 0 ∧
    (∀ x y, x > 0 → y > 0 → constraint a b x y →
      3 * x + 2 * y ≥ 24 ∧
      (∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ constraint a b x₀ y₀ ∧ 3 * x₀ + 2 * y₀ = 24)) ∧
    a = 2 ∧
    b = 3 :=
  sorry

end NUMINAMATH_CALUDE_inequality_and_constraint_solution_l1490_149062


namespace NUMINAMATH_CALUDE_remainder_theorem_l1490_149031

theorem remainder_theorem : ∃ q : ℕ, 2^222 + 222 = q * (2^111 + 2^56 + 1) + 218 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1490_149031


namespace NUMINAMATH_CALUDE_coffee_maker_price_l1490_149061

def original_price (sale_price : ℝ) (discount : ℝ) : ℝ :=
  sale_price + discount

theorem coffee_maker_price :
  let sale_price : ℝ := 70
  let discount : ℝ := 20
  original_price sale_price discount = 90 :=
by sorry

end NUMINAMATH_CALUDE_coffee_maker_price_l1490_149061


namespace NUMINAMATH_CALUDE_sqrt_square_iff_abs_l1490_149045

theorem sqrt_square_iff_abs (f g : ℝ → ℝ) :
  (∀ x, Real.sqrt (f x ^ 2) ≥ Real.sqrt (g x ^ 2)) ↔ (∀ x, |f x| ≥ |g x|) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_iff_abs_l1490_149045


namespace NUMINAMATH_CALUDE_soft_drink_cost_l1490_149013

/-- The cost of a soft drink given the total spent and the cost of candy bars. -/
theorem soft_drink_cost (total_spent : ℕ) (candy_bars : ℕ) (candy_bar_cost : ℕ) (soft_drink_cost : ℕ) : 
  total_spent = 27 ∧ candy_bars = 5 ∧ candy_bar_cost = 5 → soft_drink_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_soft_drink_cost_l1490_149013


namespace NUMINAMATH_CALUDE_last_digit_of_3_power_2012_l1490_149074

/-- The last digit of 3^n for any natural number n -/
def lastDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem last_digit_of_3_power_2012 :
  lastDigitOf3Power 2012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_3_power_2012_l1490_149074


namespace NUMINAMATH_CALUDE_power_of_729_l1490_149028

theorem power_of_729 : (729 : ℝ) ^ (4/6 : ℝ) = 81 :=
by
  have h : 729 = 3^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_729_l1490_149028


namespace NUMINAMATH_CALUDE_bananas_permutations_l1490_149029

/-- The number of distinct permutations of a word with repeated letters -/
def permutationsWithRepeats (total : ℕ) (repeats : List ℕ) : ℕ :=
  Nat.factorial total / (repeats.map Nat.factorial).prod

/-- The word "BANANAS" has 7 letters -/
def totalLetters : ℕ := 7

/-- The repetition pattern of letters in "BANANAS" -/
def letterRepeats : List ℕ := [3, 2]  -- 3 'A's and 2 'N's

theorem bananas_permutations :
  permutationsWithRepeats totalLetters letterRepeats = 420 := by
  sorry


end NUMINAMATH_CALUDE_bananas_permutations_l1490_149029


namespace NUMINAMATH_CALUDE_three_letter_sets_count_l1490_149093

/-- The number of permutations of k elements chosen from a set of n distinct elements -/
def permutations (n k : ℕ) : ℕ := sorry

/-- The number of letters available (A through J) -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def set_size : ℕ := 3

theorem three_letter_sets_count : permutations num_letters set_size = 720 := by
  sorry

end NUMINAMATH_CALUDE_three_letter_sets_count_l1490_149093


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_N_l1490_149087

/-- The number N as defined in the problem -/
def N : ℕ := 46 * 46 * 81 * 450

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the ratio of sum of odd divisors to sum of even divisors of N -/
theorem ratio_odd_even_divisors_N :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_N_l1490_149087


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l1490_149027

theorem solve_system_of_equations :
  ∃ (x y : ℚ), 3 * x - 2 * y = 11 ∧ x + 3 * y = 12 ∧ x = 57 / 11 ∧ y = 25 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l1490_149027


namespace NUMINAMATH_CALUDE_jimmy_folders_l1490_149019

-- Define the variables
def pen_cost : ℕ := 1
def notebook_cost : ℕ := 3
def folder_cost : ℕ := 5
def num_pens : ℕ := 3
def num_notebooks : ℕ := 4
def paid_amount : ℕ := 50
def change_amount : ℕ := 25

-- Define the theorem
theorem jimmy_folders :
  (paid_amount - change_amount - (num_pens * pen_cost + num_notebooks * notebook_cost)) / folder_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_jimmy_folders_l1490_149019


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1490_149023

theorem arithmetic_sequence_sum (a₁ aₙ d : ℕ) (n : ℕ+) :
  (a₁ ≤ aₙ) →
  (aₙ = a₁ + (n - 1) * d) →
  3 * (n : ℕ) * (a₁ + aₙ) / 2 = 3774 →
  3 * (Finset.sum (Finset.range n) (λ i => a₁ + i * d)) = 3774 := by
  sorry

#check arithmetic_sequence_sum 50 98 3 17

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1490_149023


namespace NUMINAMATH_CALUDE_circle_equation_l1490_149075

/-- The equation of a circle with center (-2, 1) passing through the point (2, -2) -/
theorem circle_equation :
  let center : ℝ × ℝ := (-2, 1)
  let point : ℝ × ℝ := (2, -2)
  ∀ x y : ℝ,
  (x - center.1)^2 + (y - center.2)^2 = (point.1 - center.1)^2 + (point.2 - center.2)^2 ↔
  (x + 2)^2 + (y - 1)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1490_149075


namespace NUMINAMATH_CALUDE_rent_percentage_is_seven_percent_l1490_149059

/-- Proves that the percentage of monthly earnings spent on rent is 7% -/
theorem rent_percentage_is_seven_percent (monthly_earnings : ℝ) 
  (rent_amount : ℝ) (savings_amount : ℝ) :
  rent_amount = 133 →
  savings_amount = 817 →
  monthly_earnings = rent_amount + savings_amount + (monthly_earnings / 2) →
  (rent_amount / monthly_earnings) * 100 = 7 := by
sorry

end NUMINAMATH_CALUDE_rent_percentage_is_seven_percent_l1490_149059


namespace NUMINAMATH_CALUDE_intersection_sum_l1490_149076

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 4*x + 3
def g (x y : ℝ) : Prop := x + 3*y = 3

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
  f x₁ = y₁ ∧ g x₁ y₁ ∧
  f x₂ = y₂ ∧ g x₂ y₂ ∧
  f x₃ = y₃ ∧ g x₃ y₃

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    f x₁ = y₁ ∧ g x₁ y₁ ∧
    f x₂ = y₂ ∧ g x₂ y₂ ∧
    f x₃ = y₃ ∧ g x₃ y₃ ∧
    x₁ + x₂ + x₃ = 0 ∧
    y₁ + y₂ + y₃ = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l1490_149076


namespace NUMINAMATH_CALUDE_probability_ellipse_x_foci_value_l1490_149049

/-- The probability that x²/m² + y²/n² = 1 represents an ellipse with foci on the x-axis,
    given m ∈ [1,5] and n ∈ [2,4] -/
def probability_ellipse_x_foci (m n : ℝ) : ℝ :=
  sorry

/-- Theorem stating the probability is equal to some value P -/
theorem probability_ellipse_x_foci_value :
  ∃ P, ∀ m n, m ∈ Set.Icc 1 5 → n ∈ Set.Icc 2 4 →
    probability_ellipse_x_foci m n = P :=
  sorry

end NUMINAMATH_CALUDE_probability_ellipse_x_foci_value_l1490_149049


namespace NUMINAMATH_CALUDE_min_value_inequality_l1490_149085

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c + 1) * (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≥ 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1490_149085


namespace NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l1490_149038

theorem three_numbers_in_unit_interval (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x < 1) (hy : 0 ≤ y ∧ y < 1) (hz : 0 ≤ z ∧ z < 1) :
  ∃ a b, (a = x ∨ a = y ∨ a = z) ∧ (b = x ∨ b = y ∨ b = z) ∧ a ≠ b ∧ |b - a| < (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_in_unit_interval_l1490_149038


namespace NUMINAMATH_CALUDE_bruno_pens_l1490_149040

/-- The number of pens in a dozen -/
def pens_per_dozen : ℕ := 12

/-- The total number of pens Bruno will have -/
def total_pens : ℕ := 30

/-- The number of dozens of pens Bruno wants to buy -/
def dozens_to_buy : ℚ := total_pens / pens_per_dozen

/-- Theorem stating that Bruno wants to buy 2.5 dozens of pens -/
theorem bruno_pens : dozens_to_buy = 5/2 := by sorry

end NUMINAMATH_CALUDE_bruno_pens_l1490_149040


namespace NUMINAMATH_CALUDE_sphere_radius_equal_cylinder_surface_area_l1490_149070

/-- The radius of a sphere given that its surface area is equal to the curved surface area of a right circular cylinder with height and diameter both 6 cm -/
theorem sphere_radius_equal_cylinder_surface_area (h : ℝ) (d : ℝ) (r : ℝ) : 
  h = 6 →
  d = 6 →
  4 * Real.pi * r^2 = 2 * Real.pi * (d/2) * h →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_equal_cylinder_surface_area_l1490_149070


namespace NUMINAMATH_CALUDE_thirteenth_term_of_arithmetic_sequence_l1490_149078

/-- An arithmetic sequence is defined by its third and twenty-third terms -/
def arithmetic_sequence (a₃ a₂₃ : ℚ) :=
  ∃ (a : ℕ → ℚ), (∀ n m, a (n + 1) - a n = a (m + 1) - a m) ∧ a 3 = a₃ ∧ a 23 = a₂₃

/-- The thirteenth term of the sequence is the average of the third and twenty-third terms -/
theorem thirteenth_term_of_arithmetic_sequence 
  (h : arithmetic_sequence (2/11) (3/7)) : 
  ∃ (a : ℕ → ℚ), a 13 = 47/154 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_term_of_arithmetic_sequence_l1490_149078


namespace NUMINAMATH_CALUDE_power_division_rule_l1490_149073

theorem power_division_rule (a : ℝ) : a^4 / a^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1490_149073


namespace NUMINAMATH_CALUDE_rogers_money_l1490_149084

theorem rogers_money (x : ℝ) : 
  (x + 28 - 25 = 19) → (x = 16) := by
  sorry

end NUMINAMATH_CALUDE_rogers_money_l1490_149084


namespace NUMINAMATH_CALUDE_binomial_60_3_l1490_149004

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l1490_149004


namespace NUMINAMATH_CALUDE_coffee_fraction_is_37_84_l1490_149077

-- Define the initial conditions
def initial_coffee : ℚ := 5
def initial_cream : ℚ := 7
def cup_size : ℚ := 10

-- Define the transfers
def first_transfer : ℚ := 2
def second_transfer : ℚ := 3
def third_transfer : ℚ := 1

-- Define the function to calculate the final fraction of coffee in cup 1
def final_coffee_fraction (ic : ℚ) (icr : ℚ) (cs : ℚ) (ft : ℚ) (st : ℚ) (tt : ℚ) : ℚ :=
  let coffee_after_first := ic - ft
  let total_after_first := coffee_after_first + icr + ft
  let coffee_ratio_second := ft / total_after_first
  let coffee_returned := st * coffee_ratio_second
  let total_after_second := coffee_after_first + coffee_returned + st * (1 - coffee_ratio_second)
  let coffee_after_second := coffee_after_first + coffee_returned
  let coffee_ratio_third := coffee_after_second / total_after_second
  let coffee_final := coffee_after_second - tt * coffee_ratio_third
  let total_final := total_after_second - tt
  coffee_final / total_final

-- Theorem statement
theorem coffee_fraction_is_37_84 :
  final_coffee_fraction initial_coffee initial_cream cup_size first_transfer second_transfer third_transfer = 37 / 84 := by
  sorry

end NUMINAMATH_CALUDE_coffee_fraction_is_37_84_l1490_149077


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l1490_149048

theorem complex_roots_on_circle : 
  ∃ (r : ℝ), r = 2/3 ∧ 
  ∀ (z : ℂ), (z + 1)^5 = 32 * z^5 → Complex.abs z = r :=
sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l1490_149048


namespace NUMINAMATH_CALUDE_square_difference_quotient_l1490_149026

theorem square_difference_quotient : (245^2 - 205^2) / 40 = 450 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_quotient_l1490_149026


namespace NUMINAMATH_CALUDE_smallest_sum_five_consecutive_primes_l1490_149080

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if five consecutive natural numbers are all prime -/
def fiveConsecutivePrimes (n : ℕ) : Prop :=
  isPrime n ∧ isPrime (n + 1) ∧ isPrime (n + 2) ∧ isPrime (n + 3) ∧ isPrime (n + 4)

/-- The sum of five consecutive natural numbers starting from n -/
def sumFiveConsecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (n + 3) + (n + 4)

/-- The main theorem: 119 is the smallest sum of five consecutive primes divisible by 5 -/
theorem smallest_sum_five_consecutive_primes :
  ∃ n : ℕ, fiveConsecutivePrimes n ∧ 
           sumFiveConsecutive n = 119 ∧
           119 % 5 = 0 ∧
           (∀ m : ℕ, m < n → fiveConsecutivePrimes m → sumFiveConsecutive m % 5 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_five_consecutive_primes_l1490_149080


namespace NUMINAMATH_CALUDE_changgi_weight_l1490_149055

/-- Given the weights of three people with certain relationships, prove Changgi's weight -/
theorem changgi_weight (total_weight chaeyoung_hyeonjeong_diff changgi_chaeyoung_diff : ℝ) 
  (h1 : total_weight = 106.6)
  (h2 : chaeyoung_hyeonjeong_diff = 7.7)
  (h3 : changgi_chaeyoung_diff = 4.8) : 
  ∃ (changgi chaeyoung hyeonjeong : ℝ),
    changgi + chaeyoung + hyeonjeong = total_weight ∧
    chaeyoung = hyeonjeong + chaeyoung_hyeonjeong_diff ∧
    changgi = chaeyoung + changgi_chaeyoung_diff ∧
    changgi = 41.3 := by
  sorry

end NUMINAMATH_CALUDE_changgi_weight_l1490_149055


namespace NUMINAMATH_CALUDE_y_axis_reflection_of_P_l1490_149032

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem y_axis_reflection_of_P :
  let P : ℝ × ℝ := (-1, 2)
  reflect_y_axis P = (1, 2) := by sorry

end NUMINAMATH_CALUDE_y_axis_reflection_of_P_l1490_149032


namespace NUMINAMATH_CALUDE_lattice_points_count_l1490_149016

/-- The number of lattice points on a line segment with given integer endpoints -/
def countLatticePoints (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (5,13) to (47,275) is 3 -/
theorem lattice_points_count : countLatticePoints 5 13 47 275 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_count_l1490_149016


namespace NUMINAMATH_CALUDE_subset_condition_l1490_149079

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x - a^2 < 0}

theorem subset_condition (a : ℝ) : 
  A ⊆ B a ↔ a < -1 - sqrt 5 ∨ a > (1 + sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l1490_149079


namespace NUMINAMATH_CALUDE_martha_lasagna_cheese_amount_l1490_149024

/-- The amount of cheese Martha needs for her lasagna -/
def cheese_amount : ℝ :=
  1.5

/-- The cost of cheese per kilogram in dollars -/
def cheese_cost_per_kg : ℝ :=
  6

/-- The cost of meat per kilogram in dollars -/
def meat_cost_per_kg : ℝ :=
  8

/-- The amount of meat Martha needs in grams -/
def meat_amount_grams : ℝ :=
  500

/-- The total cost of ingredients in dollars -/
def total_cost : ℝ :=
  13

theorem martha_lasagna_cheese_amount :
  cheese_amount * cheese_cost_per_kg +
  (meat_amount_grams / 1000) * meat_cost_per_kg =
  total_cost :=
by sorry

end NUMINAMATH_CALUDE_martha_lasagna_cheese_amount_l1490_149024


namespace NUMINAMATH_CALUDE_calculation_proof_equation_solution_proof_l1490_149052

-- Problem 1
theorem calculation_proof :
  18 + |-(Real.sqrt 2)| - (2012 - Real.pi)^0 - 4 * Real.sin (45 * π / 180) = 17 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem equation_solution_proof :
  ∃! x : ℝ, x ≠ 2 ∧ (4 * x) / (x^2 - 4) - 2 / (x - 2) = 1 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_solution_proof_l1490_149052


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_three_l1490_149065

theorem sqrt_difference_equals_three : Real.sqrt (81 + 49) - Real.sqrt (36 + 25) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_three_l1490_149065


namespace NUMINAMATH_CALUDE_fixed_point_difference_l1490_149094

/-- Given a function f(x) = a^(2x-6) + n, where a > 0 and a ≠ 1,
    and f(m) = 2, prove that m - n = 2 -/
theorem fixed_point_difference (a n m : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  (fun x ↦ a^(2*x - 6) + n) m = 2 → m - n = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_difference_l1490_149094


namespace NUMINAMATH_CALUDE_benny_missed_games_l1490_149042

theorem benny_missed_games (total_games attended_games : ℕ) 
  (h1 : total_games = 39)
  (h2 : attended_games = 14) :
  total_games - attended_games = 25 := by
  sorry

end NUMINAMATH_CALUDE_benny_missed_games_l1490_149042


namespace NUMINAMATH_CALUDE_darla_books_count_l1490_149089

/-- Proves that Darla has 6 books given the conditions of the problem -/
theorem darla_books_count :
  ∀ (d k g : ℕ),
  k = d / 2 →
  g = 5 * (d + k) →
  d + k + g = 54 →
  d = 6 :=
by sorry

end NUMINAMATH_CALUDE_darla_books_count_l1490_149089


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l1490_149058

theorem prime_sum_theorem (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h_eq : 2 * p + 3 * q = 6 * r) : p + q + r = 7 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l1490_149058


namespace NUMINAMATH_CALUDE_last_integer_in_sequence_l1490_149011

def sequence_term (n : ℕ) : ℚ :=
  524288 / 2^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem last_integer_in_sequence :
  ∃ (k : ℕ), (∀ (n : ℕ), n ≤ k → is_integer (sequence_term n)) ∧
             (∀ (m : ℕ), m > k → ¬ is_integer (sequence_term m)) ∧
             sequence_term k = 1 :=
sorry

end NUMINAMATH_CALUDE_last_integer_in_sequence_l1490_149011


namespace NUMINAMATH_CALUDE_additional_fabric_needed_l1490_149053

def yards_to_feet (yards : ℝ) : ℝ := yards * 3

def fabric_needed_for_dresses : ℝ :=
  2 * (yards_to_feet 5.5) +
  2 * (yards_to_feet 6) +
  2 * (yards_to_feet 6.5)

def current_fabric : ℝ := 10

theorem additional_fabric_needed :
  fabric_needed_for_dresses - current_fabric = 98 :=
by sorry

end NUMINAMATH_CALUDE_additional_fabric_needed_l1490_149053


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l1490_149000

theorem quadratic_inequality_implies_a_range :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → x^2 - 2*a*x + a + 2 ≥ 0) →
  a ∈ Set.Icc (-2) 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l1490_149000


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1490_149063

/-- A rectangular parallelepiped with edge lengths 1, 2, and 3 -/
structure Parallelepiped where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  edge1_eq : edge1 = 1
  edge2_eq : edge2 = 2
  edge3_eq : edge3 = 3

/-- A sphere containing all vertices of a rectangular parallelepiped -/
structure Sphere where
  radius : ℝ
  contains_parallelepiped : Parallelepiped → Prop

/-- The surface area of a sphere is 14π given the conditions -/
theorem sphere_surface_area (s : Sphere) (p : Parallelepiped) 
  (h : s.contains_parallelepiped p) : 
  s.radius^2 * (4 * Real.pi) = 14 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1490_149063


namespace NUMINAMATH_CALUDE_k_range_for_equation_solution_l1490_149035

theorem k_range_for_equation_solution (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ k * 4^x - k * 2^(x + 1) + 6 * (k - 5) = 0) →
  k ∈ Set.Icc 5 6 :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_equation_solution_l1490_149035


namespace NUMINAMATH_CALUDE_angle_between_clock_hands_at_9_15_angle_between_clock_hands_at_9_15_is_82_point_5_l1490_149082

/-- The angle between clock hands at 9:15 --/
theorem angle_between_clock_hands_at_9_15 : ℝ :=
  let full_rotation : ℝ := 360
  let hours_on_clock_face : ℕ := 12
  let minutes_on_clock_face : ℕ := 60
  let current_hour : ℕ := 9
  let current_minute : ℕ := 15

  let angle_per_hour : ℝ := full_rotation / hours_on_clock_face
  let angle_per_minute : ℝ := full_rotation / minutes_on_clock_face

  let minute_hand_angle : ℝ := current_minute * angle_per_minute
  let hour_hand_angle : ℝ := current_hour * angle_per_hour + (current_minute * angle_per_hour / minutes_on_clock_face)

  let angle_between_hands : ℝ := abs (minute_hand_angle - hour_hand_angle)

  82.5

theorem angle_between_clock_hands_at_9_15_is_82_point_5 :
  angle_between_clock_hands_at_9_15 = 82.5 := by sorry

end NUMINAMATH_CALUDE_angle_between_clock_hands_at_9_15_angle_between_clock_hands_at_9_15_is_82_point_5_l1490_149082


namespace NUMINAMATH_CALUDE_prism_with_18_edges_has_8_faces_l1490_149003

/-- A prism is a polyhedron with two congruent parallel faces (bases) and whose other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ :=
  let L := p.edges / 3
  2 + L

theorem prism_with_18_edges_has_8_faces (p : Prism) (h : p.edges = 18) :
  num_faces p = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_18_edges_has_8_faces_l1490_149003


namespace NUMINAMATH_CALUDE_sector_area_l1490_149067

/-- The area of a circular sector with central angle 72° and radius 5 is 5π. -/
theorem sector_area (S : ℝ) : S = 5 * Real.pi := by
  -- Given:
  -- Central angle is 72°
  -- Radius is 5
  sorry

end NUMINAMATH_CALUDE_sector_area_l1490_149067


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1490_149020

theorem point_in_fourth_quadrant (a b : ℝ) : 
  let A : ℝ × ℝ := (a^2 + 1, -1 - b^2)
  A.1 > 0 ∧ A.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1490_149020


namespace NUMINAMATH_CALUDE_function_properties_l1490_149001

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1))) 
  (h2 : is_odd (λ x => f (x - 1))) : 
  (∀ x, f (x + 4) = f x) ∧ 
  (is_odd (λ x => f (x + 3))) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l1490_149001


namespace NUMINAMATH_CALUDE_multiplier_value_l1490_149037

theorem multiplier_value (x n : ℚ) : 
  x = 40 → (x / 4) * n + 10 - 12 = 48 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_value_l1490_149037


namespace NUMINAMATH_CALUDE_interest_difference_l1490_149030

/-- Calculate the difference between the principal and simple interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal = 9200 ∧ rate = 12 ∧ time = 3 →
  principal - (principal * rate * time / 100) = 5888 := by
sorry

end NUMINAMATH_CALUDE_interest_difference_l1490_149030


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1490_149090

theorem quadratic_inequality_range (m : ℝ) : 
  (∃ x ∈ Set.Icc 2 4, x^2 - 2*x + 5 - m < 0) ↔ m > 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1490_149090


namespace NUMINAMATH_CALUDE_power_sum_equals_1999_l1490_149086

theorem power_sum_equals_1999 :
  ∃ (a b c d : ℕ), 5^a + 6^b + 7^c + 11^d = 1999 ∧ a = 4 ∧ b = 2 ∧ c = 1 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_1999_l1490_149086


namespace NUMINAMATH_CALUDE_slope_of_sine_at_pi_fourth_l1490_149064

theorem slope_of_sine_at_pi_fourth (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x) :
  deriv f (π/4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_sine_at_pi_fourth_l1490_149064


namespace NUMINAMATH_CALUDE_horner_method_result_l1490_149060

def f (x : ℝ) : ℝ := 9 + 15*x - 8*x^2 - 20*x^3 + 6*x^4 + 3*x^5

theorem horner_method_result : f 4 = 3269 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_result_l1490_149060


namespace NUMINAMATH_CALUDE_triangle_division_into_congruent_parts_l1490_149033

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : c > 0)
  (h₄ : a + b > c)
  (h₅ : b + c > a)
  (h₆ : c + a > b)

-- Define congruence for triangles
def CongruentTriangles (t₁ t₂ : Triangle) : Prop :=
  t₁.a = t₂.a ∧ t₁.b = t₂.b ∧ t₁.c = t₂.c

-- Define a division of a triangle into five smaller triangles
structure TriangleDivision (t : Triangle) :=
  (t₁ t₂ t₃ t₄ t₅ : Triangle)

-- State the theorem
theorem triangle_division_into_congruent_parts (t : Triangle) :
  ∃ (d : TriangleDivision t), 
    CongruentTriangles d.t₁ d.t₂ ∧
    CongruentTriangles d.t₁ d.t₃ ∧
    CongruentTriangles d.t₁ d.t₄ ∧
    CongruentTriangles d.t₁ d.t₅ :=
sorry

end NUMINAMATH_CALUDE_triangle_division_into_congruent_parts_l1490_149033


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1490_149091

theorem product_of_three_numbers (a b c : ℚ) : 
  a + b + c = 30 →
  a = 3 * (b + c) →
  b = 6 * c →
  a * b * c = 10125 / 14 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1490_149091


namespace NUMINAMATH_CALUDE_square_sum_zero_l1490_149022

theorem square_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 0) (h_cubic_heptic : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_l1490_149022


namespace NUMINAMATH_CALUDE_average_temperature_l1490_149005

def temperatures : List ℝ := [52, 62, 55, 59, 50]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 55.6 := by sorry

end NUMINAMATH_CALUDE_average_temperature_l1490_149005


namespace NUMINAMATH_CALUDE_factorial_ten_base_twelve_zeros_l1490_149081

theorem factorial_ten_base_twelve_zeros (n : ℕ) (h : n = 10) :
  ∃ k : ℕ, k = 4 ∧ 12^k ∣ n! ∧ ¬(12^(k+1) ∣ n!) :=
sorry

end NUMINAMATH_CALUDE_factorial_ten_base_twelve_zeros_l1490_149081


namespace NUMINAMATH_CALUDE_solution_value_l1490_149018

theorem solution_value (a b : ℝ) : 
  (a * 1 - b * 2 + 3 = 0) →
  (a * (-1) - b * 1 + 3 = 0) →
  a - 3 * b = -5 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l1490_149018


namespace NUMINAMATH_CALUDE_restaurant_sales_tax_rate_l1490_149088

theorem restaurant_sales_tax_rate 
  (total_bill : ℝ) 
  (striploin_cost : ℝ) 
  (wine_cost : ℝ) 
  (gratuities : ℝ) 
  (h1 : total_bill = 140)
  (h2 : striploin_cost = 80)
  (h3 : wine_cost = 10)
  (h4 : gratuities = 41) :
  (total_bill - striploin_cost - wine_cost - gratuities) / (striploin_cost + wine_cost) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_sales_tax_rate_l1490_149088


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_p_or_q_l1490_149025

theorem not_p_necessary_not_sufficient_for_not_p_or_q :
  (∃ p q : Prop, ¬p ∧ (p ∨ q)) ∧
  (∀ p q : Prop, ¬(p ∨ q) → ¬p) :=
by sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_p_or_q_l1490_149025


namespace NUMINAMATH_CALUDE_smallest_a_value_l1490_149008

/-- Given two quadratic equations with integer roots less than -1, find the smallest possible 'a' -/
theorem smallest_a_value (a b c : ℤ) : 
  (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b*x + a = 0 ∧ y^2 + b*y + a = 0) →
  (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c*z + a = 1 ∧ w^2 + c*w + a = 1) →
  (∀ a' b' c' : ℤ, 
    (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b'*x + a' = 0 ∧ y^2 + b'*y + a' = 0) →
    (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c'*z + a' = 1 ∧ w^2 + c'*w + a' = 1) →
    a' ≥ a) →
  a = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1490_149008


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1490_149056

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) → (0 ≤ a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1490_149056


namespace NUMINAMATH_CALUDE_partnership_profit_l1490_149017

/-- Represents the profit share of a partner in a business partnership. -/
structure ProfitShare where
  investment : ℕ
  share : ℕ

/-- Calculates the total profit of a partnership business given the investments and one partner's profit share. -/
def totalProfit (a b c : ProfitShare) : ℕ :=
  sorry

/-- Theorem stating that given the specific investments and A's profit share, the total profit is 12500. -/
theorem partnership_profit (a b c : ProfitShare) 
  (ha : a.investment = 6300)
  (hb : b.investment = 4200)
  (hc : c.investment = 10500)
  (ha_share : a.share = 3750) :
  totalProfit a b c = 12500 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l1490_149017
