import Mathlib

namespace NUMINAMATH_CALUDE_matrix_equation_holds_l3535_353504

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 2; 1, 0, 1; 2, 1, 0]

theorem matrix_equation_holds :
  let s : ℤ := -2
  let t : ℤ := -6
  let u : ℤ := -14
  let v : ℤ := -13
  A^4 + s • A^3 + t • A^2 + u • A + v • (1 : Matrix (Fin 3) (Fin 3) ℤ) = (0 : Matrix (Fin 3) (Fin 3) ℤ) := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_holds_l3535_353504


namespace NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l3535_353536

/-- The shortest distance between a point and a parabola -/
theorem shortest_distance_point_to_parabola :
  let point := (6, 12)
  let parabola := {(x, y) : ℝ × ℝ | x = y^2 / 2}
  (shortest_distance : ℝ) →
  shortest_distance = 2 * Real.sqrt 17 ∧
  ∀ (p : ℝ × ℝ), p ∈ parabola → 
    Real.sqrt ((point.1 - p.1)^2 + (point.2 - p.2)^2) ≥ shortest_distance :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l3535_353536


namespace NUMINAMATH_CALUDE_limit_s_at_zero_is_infinity_l3535_353554

/-- The x coordinate of the left endpoint of the intersection of y = x^3 and y = m -/
noncomputable def P (m : ℝ) : ℝ := -Real.rpow m (1/3)

/-- The function s defined as [P(-m) - P(m)]/m -/
noncomputable def s (m : ℝ) : ℝ := (P (-m) - P m) / m

theorem limit_s_at_zero_is_infinity :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -2 < m ∧ m < 2 → |s m| > ε :=
sorry

end NUMINAMATH_CALUDE_limit_s_at_zero_is_infinity_l3535_353554


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3535_353599

theorem intersection_of_sets : 
  let A : Set ℤ := {-1, 1, 2, 4}
  let B : Set ℤ := {-1, 0, 2}
  A ∩ B = {-1, 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3535_353599


namespace NUMINAMATH_CALUDE_special_function_at_five_l3535_353530

/-- A function satisfying f(x - y) = f(x) + f(y) for all real x and y, and f(0) = 2 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x - y) = f x + f y) ∧ (f 0 = 2)

/-- Theorem: For any function satisfying the special_function property, f(5) = 1 -/
theorem special_function_at_five (f : ℝ → ℝ) (h : special_function f) : f 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_five_l3535_353530


namespace NUMINAMATH_CALUDE_equation_solution_l3535_353586

theorem equation_solution (x : ℝ) : 
  (x / 6) / 3 = 9 / (x / 3) → x = 9 * Real.sqrt 6 ∨ x = -9 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3535_353586


namespace NUMINAMATH_CALUDE_odd_expression_proof_l3535_353546

theorem odd_expression_proof (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) :
  Odd (4 * p^2 + 2 * q^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_expression_proof_l3535_353546


namespace NUMINAMATH_CALUDE_binomial_26_6_l3535_353540

theorem binomial_26_6 (h1 : Nat.choose 24 4 = 10626)
                      (h2 : Nat.choose 24 5 = 42504)
                      (h3 : Nat.choose 24 6 = 53130) :
  Nat.choose 26 6 = 148764 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l3535_353540


namespace NUMINAMATH_CALUDE_thousandth_term_of_sequence_l3535_353529

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem thousandth_term_of_sequence :
  arithmetic_sequence 1 3 1000 = 2998 := by
  sorry

end NUMINAMATH_CALUDE_thousandth_term_of_sequence_l3535_353529


namespace NUMINAMATH_CALUDE_red_toy_percentage_l3535_353535

/-- Represents a toy production lot -/
structure ToyLot where
  total : ℕ
  red : ℕ
  green : ℕ
  small : ℕ
  large : ℕ
  redSmall : ℕ
  redLarge : ℕ
  greenLarge : ℕ

/-- The conditions of the toy production lot -/
def validToyLot (lot : ToyLot) : Prop :=
  lot.total > 0 ∧
  lot.red + lot.green = lot.total ∧
  lot.small + lot.large = lot.total ∧
  lot.small = lot.large ∧
  lot.redSmall = (lot.total * 10) / 100 ∧
  lot.greenLarge = 40 ∧
  lot.redLarge = 60

/-- The theorem stating the percentage of red toys -/
theorem red_toy_percentage (lot : ToyLot) :
  validToyLot lot → (lot.red : ℚ) / lot.total = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_red_toy_percentage_l3535_353535


namespace NUMINAMATH_CALUDE_expansion_nonzero_terms_l3535_353582

/-- The number of nonzero terms in the expansion of (x^2+5)(3x^3+2x^2+6)-4(x^4-3x^3+8x^2+1) + 2x^3 -/
theorem expansion_nonzero_terms (x : ℝ) : 
  let expanded := (x^2 + 5) * (3*x^3 + 2*x^2 + 6) - 4*(x^4 - 3*x^3 + 8*x^2 + 1) + 2*x^3
  ∃ (a b c d e : ℝ) (n : ℕ), 
    expanded = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_nonzero_terms_l3535_353582


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3535_353517

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 120 → b = 160 → c^2 = a^2 + b^2 → c = 200 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3535_353517


namespace NUMINAMATH_CALUDE_minimum_cans_needed_l3535_353566

/-- The number of ounces in each can -/
def can_capacity : ℕ := 10

/-- The minimum number of ounces required -/
def min_ounces : ℕ := 120

/-- The minimum number of cans needed to provide at least the required ounces -/
def min_cans : ℕ := 12

theorem minimum_cans_needed :
  (min_cans * can_capacity ≥ min_ounces) ∧
  (∀ n : ℕ, n * can_capacity ≥ min_ounces → n ≥ min_cans) :=
by sorry

end NUMINAMATH_CALUDE_minimum_cans_needed_l3535_353566


namespace NUMINAMATH_CALUDE_quadratic_factor_l3535_353559

theorem quadratic_factor (k : ℝ) : 
  (∃ b : ℝ, (X + 5) * (X + b) = X^2 - k*X - 15) → 
  (X - 3) * (X + 5) = X^2 - k*X - 15 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factor_l3535_353559


namespace NUMINAMATH_CALUDE_ellipse_properties_l3535_353584

/-- Given an ellipse C with equation (x^2 / a^2) + (y^2 / b^2) = 1, where a > b > 0,
    eccentricity 1/2, and the area of the quadrilateral formed by its vertices is 4√3,
    we prove properties about its equation and intersecting lines. -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := 1 / 2  -- eccentricity
  let quad_area := 4 * Real.sqrt 3  -- area of quadrilateral formed by vertices
  ∀ x y : ℝ,
    (x^2 / a^2 + y^2 / b^2 = 1) →  -- equation of ellipse C
    (e = Real.sqrt (1 - b^2 / a^2)) →  -- definition of eccentricity
    (quad_area = 4 * a * b) →  -- area of quadrilateral
    (∀ x₁ y₁ x₂ y₂ : ℝ,
      (x₁^2 / a^2 + y₁^2 / b^2 = 1) →  -- P(x₁, y₁) on ellipse
      (x₂^2 / a^2 + y₂^2 / b^2 = 1) →  -- Q(x₂, y₂) on ellipse
      (1/2 * |x₁ * y₂ - x₂ * y₁| = Real.sqrt 3) →  -- area of triangle OPQ is √3
      (x₁^2 / 4 + y₁^2 / 3 = 1) ∧  -- equation of ellipse C
      (x₂^2 / 4 + y₂^2 / 3 = 1) ∧  -- equation of ellipse C
      (x₁^2 + x₂^2 = 4))  -- constant sum of squares
  := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3535_353584


namespace NUMINAMATH_CALUDE_steve_socks_count_l3535_353550

/-- The number of pairs of matching socks Steve has -/
def matching_pairs : ℕ := 4

/-- The number of mismatching socks Steve has -/
def mismatching_socks : ℕ := 17

/-- The total number of socks Steve has -/
def total_socks : ℕ := 2 * matching_pairs + mismatching_socks

theorem steve_socks_count : total_socks = 25 := by
  sorry

end NUMINAMATH_CALUDE_steve_socks_count_l3535_353550


namespace NUMINAMATH_CALUDE_integer_expression_l3535_353562

/-- Binomial coefficient -/
def binomial (m l : ℕ) : ℕ := Nat.choose m l

/-- The main theorem -/
theorem integer_expression (l m : ℤ) (h1 : 1 ≤ l) (h2 : l < m) :
  ∃ (k : ℤ), ((m - 3*l + 2) / (l + 2)) * binomial m.toNat l.toNat = k ↔ 
  ∃ (n : ℤ), m + 8 = n * (l + 2) := by sorry

end NUMINAMATH_CALUDE_integer_expression_l3535_353562


namespace NUMINAMATH_CALUDE_currency_and_unit_comparisons_l3535_353588

-- Define the conversion rates
def yuan_to_jiao : ℚ → ℚ := (· * 10)
def dm_to_cm : ℚ → ℚ := (· * 10)
def hectare_to_m2 : ℚ → ℚ := (· * 10000)
def km2_to_hectare : ℚ → ℚ := (· * 100)

-- Define the theorem
theorem currency_and_unit_comparisons :
  (7 > 5.70) ∧
  (70 > 7) ∧
  (80000 > 70000) ∧
  (1 = 1) ∧
  (34 * 6 * 2 = 34 * 12) ∧
  (3.9 = 3.9) := by
  sorry

end NUMINAMATH_CALUDE_currency_and_unit_comparisons_l3535_353588


namespace NUMINAMATH_CALUDE_min_product_sum_l3535_353568

theorem min_product_sum (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : 
  ({a₁, a₂, a₃, b₁, b₂, b₃} : Finset ℕ) = {1, 2, 3, 4, 5, 6} →
  a₁ * a₂ * a₃ + b₁ * b₂ * b₃ ≥ 56 ∧ 
  ∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℕ), 
    ({x₁, x₂, x₃, y₁, y₂, y₃} : Finset ℕ) = {1, 2, 3, 4, 5, 6} ∧
    x₁ * x₂ * x₃ + y₁ * y₂ * y₃ = 56 :=
by sorry

end NUMINAMATH_CALUDE_min_product_sum_l3535_353568


namespace NUMINAMATH_CALUDE_second_platform_length_l3535_353589

/-- Given a train and two platforms, calculate the length of the second platform -/
theorem second_platform_length 
  (train_length : ℝ) 
  (first_platform_length : ℝ) 
  (first_crossing_time : ℝ) 
  (second_crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : first_platform_length = 200)
  (h3 : first_crossing_time = 15)
  (h4 : second_crossing_time = 20) :
  (second_crossing_time * ((train_length + first_platform_length) / first_crossing_time)) - train_length = 300 :=
by sorry

end NUMINAMATH_CALUDE_second_platform_length_l3535_353589


namespace NUMINAMATH_CALUDE_solve_equations_l3535_353567

theorem solve_equations (t u s : ℝ) : 
  t = 15 * s^2 → 
  u = 5 * s + 3 → 
  t = 3.75 → 
  s = 0.5 ∧ u = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equations_l3535_353567


namespace NUMINAMATH_CALUDE_john_car_profit_l3535_353533

-- Define the given values
def repair_cost : ℝ := 20000
def discount_rate : ℝ := 0.2
def prize_money : ℝ := 70000
def prize_keep_rate : ℝ := 0.9

-- Define the theorem
theorem john_car_profit :
  let discounted_cost := repair_cost * (1 - discount_rate)
  let kept_prize := prize_money * prize_keep_rate
  kept_prize - discounted_cost = 47000 := by
  sorry

end NUMINAMATH_CALUDE_john_car_profit_l3535_353533


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l3535_353563

theorem cheryl_material_usage 
  (bought : ℚ) 
  (left : ℚ) 
  (h1 : bought = 3/8 + 1/3) 
  (h2 : left = 15/40) : 
  bought - left = 1/3 := by
sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l3535_353563


namespace NUMINAMATH_CALUDE_book_sale_revenue_l3535_353511

theorem book_sale_revenue (total_books : ℕ) (sold_fraction : ℚ) (price_per_book : ℚ) (unsold_books : ℕ) : 
  sold_fraction = 2 / 3 →
  price_per_book = 2 →
  unsold_books = 36 →
  unsold_books = (1 - sold_fraction) * total_books →
  sold_fraction * total_books * price_per_book = 144 := by
  sorry

#check book_sale_revenue

end NUMINAMATH_CALUDE_book_sale_revenue_l3535_353511


namespace NUMINAMATH_CALUDE_inequality_proof_l3535_353576

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3535_353576


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3535_353579

/-- Given vectors a, b, c, and the condition that (2a - b) is parallel to c, 
    prove that sin(2θ) = -12/13 --/
theorem sin_2theta_value (θ : ℝ) 
  (a b c : ℝ × ℝ)
  (ha : a = (Real.sin θ, 1))
  (hb : b = (-Real.sin θ, 0))
  (hc : c = (Real.cos θ, -1))
  (h_parallel : ∃ (k : ℝ), (2 • a - b) = k • c) :
  Real.sin (2 * θ) = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3535_353579


namespace NUMINAMATH_CALUDE_four_solutions_l3535_353526

/-- The number of integer pairs (m, n) satisfying (m-1)(n-1) = 2 -/
def count_solutions : ℕ := 4

/-- A pair of integers (m, n) satisfies the equation (m-1)(n-1) = 2 -/
def is_solution (m n : ℤ) : Prop := (m - 1) * (n - 1) = 2

theorem four_solutions :
  (∃ (S : Finset (ℤ × ℤ)), S.card = count_solutions ∧
    (∀ (p : ℤ × ℤ), p ∈ S ↔ is_solution p.1 p.2) ∧
    (∀ (m n : ℤ), is_solution m n → (m, n) ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l3535_353526


namespace NUMINAMATH_CALUDE_counterexample_exists_l3535_353553

def is_in_set (n : ℕ) : Prop := n = 14 ∨ n = 18 ∨ n = 20 ∨ n = 24 ∨ n = 30

theorem counterexample_exists : 
  ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n + 2)) ∧ is_in_set n :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3535_353553


namespace NUMINAMATH_CALUDE_tournament_matches_l3535_353593

/-- Represents the number of matches played by each student -/
structure MatchCounts where
  student1 : Nat
  student2 : Nat
  student3 : Nat
  student4 : Nat
  student5 : Nat
  student6 : Nat

/-- The total number of matches in a tournament with 6 players -/
def totalMatches : Nat := 15

theorem tournament_matches (mc : MatchCounts) : 
  mc.student1 = 5 → 
  mc.student2 = 4 → 
  mc.student3 = 3 → 
  mc.student4 = 2 → 
  mc.student5 = 1 → 
  mc.student1 + mc.student2 + mc.student3 + mc.student4 + mc.student5 + mc.student6 = 2 * totalMatches → 
  mc.student6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tournament_matches_l3535_353593


namespace NUMINAMATH_CALUDE_dividend_calculation_l3535_353581

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : face_value = 100)
  (h3 : premium_rate = 0.20)
  (h4 : dividend_rate = 0.07) :
  let price_per_share := face_value * (1 + premium_rate)
  let num_shares := investment / price_per_share
  let dividend_per_share := face_value * dividend_rate
  num_shares * dividend_per_share = 840 := by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3535_353581


namespace NUMINAMATH_CALUDE_geometric_means_l3535_353534

theorem geometric_means (a b : ℝ) (p : ℕ) (ha : 0 < a) (hb : a < b) :
  let r := (b / a) ^ (1 / (p + 1 : ℝ))
  ∀ k : ℕ, k ≥ 1 → k ≤ p →
    a * r ^ k = a * (b / a) ^ (k / (p + 1 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_geometric_means_l3535_353534


namespace NUMINAMATH_CALUDE_solve_for_y_l3535_353556

theorem solve_for_y (x y p : ℝ) (h : p = (5 * x * y) / (x - y)) : 
  y = (p * x) / (5 * x + p) := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l3535_353556


namespace NUMINAMATH_CALUDE_alternatingArithmeticSequenceSum_l3535_353522

def alternatingArithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |> List.map (λ i => a₁ + i * d * (if i % 2 = 0 then 1 else -1))

theorem alternatingArithmeticSequenceSum :
  let seq := alternatingArithmeticSequence 2 4 26
  seq.sum = -52 := by
  sorry

end NUMINAMATH_CALUDE_alternatingArithmeticSequenceSum_l3535_353522


namespace NUMINAMATH_CALUDE_square_plus_25_divisible_by_2_and_5_l3535_353560

/-- A positive integer with only prime divisors 2 and 5 -/
def HasOnly2And5AsDivisors (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p = 2 ∨ p = 5

theorem square_plus_25_divisible_by_2_and_5 :
  ∀ N : ℕ, N > 0 →
  HasOnly2And5AsDivisors N →
  (∃ M : ℕ, N + 25 = M^2) →
  N = 200 ∨ N = 2000 := by
sorry

end NUMINAMATH_CALUDE_square_plus_25_divisible_by_2_and_5_l3535_353560


namespace NUMINAMATH_CALUDE_sum_of_triangles_34_l3535_353594

/-- The triangle operation defined as a * b - c -/
def triangle_op (a b c : ℕ) : ℕ := a * b - c

/-- Theorem stating that the sum of two specific triangle operations equals 34 -/
theorem sum_of_triangles_34 : triangle_op 3 5 2 + triangle_op 4 6 3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangles_34_l3535_353594


namespace NUMINAMATH_CALUDE_sarah_remaining_pages_l3535_353525

/-- Given the initial number of problems, the number of completed problems,
    and the number of problems per page, calculates the number of remaining pages. -/
def remaining_pages (initial_problems : ℕ) (completed_problems : ℕ) (problems_per_page : ℕ) : ℕ :=
  (initial_problems - completed_problems) / problems_per_page

/-- Proves that Sarah has 5 pages of problems left to do. -/
theorem sarah_remaining_pages :
  remaining_pages 60 20 8 = 5 := by
  sorry

#eval remaining_pages 60 20 8

end NUMINAMATH_CALUDE_sarah_remaining_pages_l3535_353525


namespace NUMINAMATH_CALUDE_annual_growth_rate_l3535_353505

theorem annual_growth_rate (initial_amount final_amount : ℝ) (h : initial_amount * (1 + 0.125)^2 = final_amount) :
  ∃ (rate : ℝ), initial_amount * (1 + rate)^2 = final_amount ∧ rate = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_annual_growth_rate_l3535_353505


namespace NUMINAMATH_CALUDE_six_factorial_divisors_l3535_353516

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define function to count positive divisors
def count_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- Theorem statement
theorem six_factorial_divisors :
  count_divisors (factorial 6) = 30 := by
  sorry

end NUMINAMATH_CALUDE_six_factorial_divisors_l3535_353516


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l3535_353544

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℤ) :
  ∃ k : ℤ, n^3 + (n+1)^3 + (n+2)^3 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l3535_353544


namespace NUMINAMATH_CALUDE_ellipse_and_line_equation_l3535_353541

noncomputable section

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola
def hyperbola (m n : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 / n = 1

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_and_line_equation 
  (a b m n e₁ e₂ : ℝ)
  (h₁ : a > b)
  (h₂ : b > 0)
  (h₃ : m > 0)
  (h₄ : n > 0)
  (h₅ : b = Real.sqrt 3)
  (h₆ : n / m = 3)
  (h₇ : e₁ * e₂ = 1)
  (h₈ : e₂ = Real.sqrt 4)
  (h₉ : e₁ = eccentricity a b)
  (P : ℝ × ℝ)
  (h₁₀ : P = (-1, 3/2))
  (S₁ S₂ : ℝ)
  (h₁₁ : S₁ = 6 * S₂) :
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ k, k = Real.sqrt 6 / 2 ∧ 
    (∀ x y, y = k * x + 1 ∨ y = -k * x + 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ellipse_and_line_equation_l3535_353541


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l3535_353503

def income : ℕ := 10000
def savings : ℕ := 3000
def expenditure : ℕ := income - savings

theorem income_expenditure_ratio :
  (income : ℚ) / (expenditure : ℚ) = 10 / 7 := by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l3535_353503


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l3535_353569

theorem quadrilateral_angle_measure (A B C D : ℝ) : 
  A + B = 180 →  -- ∠A + ∠B = 180°
  C = D →        -- ∠C = ∠D
  A = 40 →       -- ∠A = 40°
  B + C = 160 → -- ∠B + ∠C = 160°
  D = 20 :=      -- Prove that ∠D = 20°
by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l3535_353569


namespace NUMINAMATH_CALUDE_outside_trash_count_l3535_353547

def total_trash : ℕ := 1576
def classroom_trash : ℕ := 344

theorem outside_trash_count : total_trash - classroom_trash = 1232 := by
  sorry

end NUMINAMATH_CALUDE_outside_trash_count_l3535_353547


namespace NUMINAMATH_CALUDE_right_triangle_k_values_l3535_353591

/-- A right-angled triangle in a 2D Cartesian coordinate system. -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angled : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∨
                    (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0 ∨
                    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

/-- The theorem stating the possible values of k in the given right-angled triangle. -/
theorem right_triangle_k_values (triangle : RightTriangle)
  (h1 : triangle.B.1 - triangle.A.1 = 2 ∧ triangle.B.2 - triangle.A.2 = 1)
  (h2 : triangle.C.1 - triangle.A.1 = 3)
  (h3 : ∃ k, triangle.C.2 - triangle.A.2 = k) :
  ∃ k, (k = -6 ∨ k = -1) ∧ triangle.C.2 - triangle.A.2 = k :=
sorry


end NUMINAMATH_CALUDE_right_triangle_k_values_l3535_353591


namespace NUMINAMATH_CALUDE_y_greater_than_one_l3535_353551

theorem y_greater_than_one (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 := by
  sorry

end NUMINAMATH_CALUDE_y_greater_than_one_l3535_353551


namespace NUMINAMATH_CALUDE_mo_drinks_26_cups_l3535_353521

/-- Represents Mo's drinking habits and the weather conditions for a week -/
structure WeeklyDrinks where
  n : ℕ  -- Number of hot chocolate cups on a rainy day
  rainyDays : ℕ  -- Number of rainy days in the week
  teaPerNonRainyDay : ℕ  -- Number of tea cups on a non-rainy day
  teaExcess : ℕ  -- Excess of tea cups over hot chocolate cups

/-- Calculates the total number of cups (tea and hot chocolate) Mo drinks in a week -/
def totalCups (w : WeeklyDrinks) : ℕ :=
  w.n * w.rainyDays + w.teaPerNonRainyDay * (7 - w.rainyDays)

/-- Theorem stating that under the given conditions, Mo drinks 26 cups in total -/
theorem mo_drinks_26_cups (w : WeeklyDrinks)
  (h1 : w.rainyDays = 1)
  (h2 : w.teaPerNonRainyDay = 3)
  (h3 : w.teaPerNonRainyDay * (7 - w.rainyDays) = w.n * w.rainyDays + w.teaExcess)
  (h4 : w.teaExcess = 10) :
  totalCups w = 26 := by
  sorry

#check mo_drinks_26_cups

end NUMINAMATH_CALUDE_mo_drinks_26_cups_l3535_353521


namespace NUMINAMATH_CALUDE_math_competition_theorem_l3535_353598

/-- Represents the number of participants who solved both problem i and problem j -/
def p (i j : Fin 6) (n : ℕ) : ℕ := sorry

/-- Represents the number of participants who solved exactly k problems -/
def n_k (k : Fin 7) (n : ℕ) : ℕ := sorry

theorem math_competition_theorem (n : ℕ) :
  (∀ i j : Fin 6, i < j → p i j n > (2 * n) / 5) →
  (n_k 6 n = 0) →
  (n_k 5 n ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_math_competition_theorem_l3535_353598


namespace NUMINAMATH_CALUDE_pyramid_ball_count_l3535_353543

/-- The number of layers in the pyramid -/
def n : ℕ := 13

/-- The number of balls in the top layer -/
def first_term : ℕ := 4

/-- The number of balls in the bottom layer -/
def last_term : ℕ := 40

/-- The sum of the arithmetic sequence representing the number of balls in each layer -/
def sum_of_sequence : ℕ := n * (first_term + last_term) / 2

theorem pyramid_ball_count :
  sum_of_sequence = 286 := by sorry

end NUMINAMATH_CALUDE_pyramid_ball_count_l3535_353543


namespace NUMINAMATH_CALUDE_stream_rate_l3535_353557

/-- The rate of a stream given boat speed and downstream travel information -/
theorem stream_rate (boat_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  boat_speed = 16 →
  distance = 168 →
  time = 8 →
  (boat_speed + (distance / time - boat_speed)) * time = distance →
  distance / time - boat_speed = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_stream_rate_l3535_353557


namespace NUMINAMATH_CALUDE_largest_common_divisor_414_345_l3535_353538

theorem largest_common_divisor_414_345 : Nat.gcd 414 345 = 69 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_414_345_l3535_353538


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3535_353590

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (4 * x - 6 * y = -14) ∧ (8 * x + 3 * y = -15) ∧ (x = -11/5) ∧ (y = 13/15) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3535_353590


namespace NUMINAMATH_CALUDE_walk_a_thon_earnings_l3535_353587

theorem walk_a_thon_earnings (last_year_rate : ℚ) (last_year_total : ℚ) 
  (extra_miles : ℕ) (this_year_rate : ℚ) : 
  last_year_rate = 4 →
  last_year_total = 44 →
  extra_miles = 5 →
  (last_year_total / last_year_rate + extra_miles) * this_year_rate = last_year_total →
  this_year_rate = 11/4 := by
sorry

#eval (11 : ℚ) / 4  -- To show the decimal representation

end NUMINAMATH_CALUDE_walk_a_thon_earnings_l3535_353587


namespace NUMINAMATH_CALUDE_rajesh_work_time_l3535_353555

/-- The problem of determining Rajesh's work time -/
theorem rajesh_work_time (rahul_rate : ℝ) (rajesh_rate : ℝ → ℝ) (combined_rate : ℝ → ℝ) 
  (total_payment : ℝ) (rahul_share : ℝ) (R : ℝ) :
  rahul_rate = 1/3 →
  (∀ x, rajesh_rate x = 1/x) →
  (∀ x, combined_rate x = (x + 3) / (3*x)) →
  total_payment = 150 →
  rahul_share = 60 →
  R = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_rajesh_work_time_l3535_353555


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3535_353542

-- Define variables
variable (a b x y : ℝ)

-- Theorem 1
theorem simplify_expression_1 : 
  6*a + 7*b^2 - 9 + 4*a - b^2 + 6 = 6*b^2 + 10*a - 3 := by sorry

-- Theorem 2
theorem simplify_expression_2 :
  5*x - 2*(4*x + 5*y) + 3*(3*x - 4*y) = 6*x - 22*y := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3535_353542


namespace NUMINAMATH_CALUDE_robie_cards_count_l3535_353520

/-- The number of cards in each box -/
def cards_per_box : ℕ := 10

/-- The number of cards not placed in a box -/
def cards_outside_box : ℕ := 5

/-- The number of boxes Robie gave away -/
def boxes_given_away : ℕ := 2

/-- The number of boxes Robie has with him -/
def boxes_remaining : ℕ := 5

/-- The total number of cards Robie had in the beginning -/
def total_cards : ℕ := (boxes_given_away + boxes_remaining) * cards_per_box + cards_outside_box

theorem robie_cards_count : total_cards = 75 := by
  sorry

end NUMINAMATH_CALUDE_robie_cards_count_l3535_353520


namespace NUMINAMATH_CALUDE_max_edges_100_vertices_triangle_free_l3535_353507

/-- The maximum number of edges in a triangle-free graph with n vertices -/
def maxEdgesTriangleFree (n : ℕ) : ℕ := n^2 / 4

/-- Theorem: In a graph with 100 vertices and no triangles, the maximum number of edges is 2500 -/
theorem max_edges_100_vertices_triangle_free :
  maxEdgesTriangleFree 100 = 2500 := by
  sorry

#eval maxEdgesTriangleFree 100  -- Should output 2500

end NUMINAMATH_CALUDE_max_edges_100_vertices_triangle_free_l3535_353507


namespace NUMINAMATH_CALUDE_coins_missing_l3535_353572

theorem coins_missing (x : ℚ) (h : x > 0) : 
  let lost := (2 : ℚ) / 3 * x
  let found := (3 : ℚ) / 4 * lost
  let remaining := x - lost + found
  (x - remaining) / x = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_coins_missing_l3535_353572


namespace NUMINAMATH_CALUDE_triangle_prime_angles_l3535_353502

theorem triangle_prime_angles (a b c : ℕ) :
  a + b + c = 180 →
  (Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) →
  a = 2 ∨ b = 2 ∨ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_prime_angles_l3535_353502


namespace NUMINAMATH_CALUDE_number_of_officers_l3535_353545

/-- Prove the number of officers in an office given average salaries and number of non-officers -/
theorem number_of_officers
  (avg_salary : ℝ)
  (avg_salary_officers : ℝ)
  (avg_salary_non_officers : ℝ)
  (num_non_officers : ℕ)
  (h1 : avg_salary = 120)
  (h2 : avg_salary_officers = 430)
  (h3 : avg_salary_non_officers = 110)
  (h4 : num_non_officers = 465) :
  ∃ (num_officers : ℕ),
    avg_salary * (num_officers + num_non_officers) =
    avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers ∧
    num_officers = 15 :=
by sorry

end NUMINAMATH_CALUDE_number_of_officers_l3535_353545


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l3535_353575

-- Define the function f(x) = (2x + 1)³
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 3

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 6 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l3535_353575


namespace NUMINAMATH_CALUDE_power_ends_in_12890625_l3535_353596

theorem power_ends_in_12890625 (a n : ℕ) (h : a % 10^8 = 12890625) :
  (a^n) % 10^8 = 12890625 := by
  sorry

end NUMINAMATH_CALUDE_power_ends_in_12890625_l3535_353596


namespace NUMINAMATH_CALUDE_logarithm_difference_l3535_353519

theorem logarithm_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) : 
  b - d = 93 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_difference_l3535_353519


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l3535_353561

theorem abs_neg_three_eq_three : |(-3 : ℤ)| = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l3535_353561


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3535_353573

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3535_353573


namespace NUMINAMATH_CALUDE_total_tickets_sold_l3535_353509

/-- Proves that the total number of tickets sold is 130 given the specified conditions -/
theorem total_tickets_sold (adult_price child_price total_receipts child_tickets : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_receipts = 840)
  (h4 : child_tickets = 90)
  : ∃ (adult_tickets : ℕ), adult_tickets * adult_price + child_tickets * child_price = total_receipts ∧ 
    adult_tickets + child_tickets = 130 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l3535_353509


namespace NUMINAMATH_CALUDE_min_value_abs_diff_l3535_353512

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the theorem
theorem min_value_abs_diff (x y : ℝ) :
  log 4 (x + 2*y) + log 1 (x - 2*y) = 1 ∧ 
  x + 2*y > 0 ∧ 
  x - 2*y > 0 →
  ∃ (min_val : ℝ), min_val = Real.sqrt 3 ∧ ∀ (a b : ℝ), 
    (log 4 (a + 2*b) + log 1 (a - 2*b) = 1 ∧ a + 2*b > 0 ∧ a - 2*b > 0) →
    |a| - |b| ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_abs_diff_l3535_353512


namespace NUMINAMATH_CALUDE_custom_op_solution_l3535_353549

/-- Custom operation for integers -/
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if y12 = 110 using the custom operation, then y = 11 -/
theorem custom_op_solution :
  ∀ y : ℤ, customOp y 12 = 110 → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l3535_353549


namespace NUMINAMATH_CALUDE_swimming_lane_length_l3535_353518

/-- Represents the length of a swimming lane in meters -/
def lane_length : ℝ := 100

/-- Represents the number of round trips swam -/
def round_trips : ℕ := 4

/-- Represents the total distance swam in meters -/
def total_distance : ℝ := 800

/-- Represents the number of lane lengths in a round trip -/
def lengths_per_round_trip : ℕ := 2

theorem swimming_lane_length :
  lane_length * (round_trips * lengths_per_round_trip) = total_distance :=
sorry

end NUMINAMATH_CALUDE_swimming_lane_length_l3535_353518


namespace NUMINAMATH_CALUDE_ring_binder_price_l3535_353565

/-- Proves that the original price of each ring-binder was $20 given the problem conditions -/
theorem ring_binder_price : 
  ∀ (original_backpack_price backpack_price_increase 
     ring_binder_price_decrease num_ring_binders total_spent : ℕ),
  original_backpack_price = 50 →
  backpack_price_increase = 5 →
  ring_binder_price_decrease = 2 →
  num_ring_binders = 3 →
  total_spent = 109 →
  ∃ (original_ring_binder_price : ℕ),
    original_ring_binder_price = 20 ∧
    (original_backpack_price + backpack_price_increase) + 
    num_ring_binders * (original_ring_binder_price - ring_binder_price_decrease) = total_spent :=
by
  sorry

end NUMINAMATH_CALUDE_ring_binder_price_l3535_353565


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3535_353539

theorem nested_fraction_equality : (1 / (1 + 1 / (4 + 1 / 5))) = 21 / 26 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3535_353539


namespace NUMINAMATH_CALUDE_vasya_no_purchase_days_l3535_353523

theorem vasya_no_purchase_days :
  ∀ (x y z w : ℕ),
  x + y + z + w = 15 →
  9 * x + 4 * z = 30 →
  2 * y + z = 9 →
  w = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_vasya_no_purchase_days_l3535_353523


namespace NUMINAMATH_CALUDE_m_divided_by_8_l3535_353514

theorem m_divided_by_8 (m : ℕ) (h : m = 16^1500) : m / 8 = 2^5997 := by
  sorry

end NUMINAMATH_CALUDE_m_divided_by_8_l3535_353514


namespace NUMINAMATH_CALUDE_number_of_employees_l3535_353570

def average_salary_without_manager : ℝ := 1200
def average_salary_with_manager : ℝ := 1300
def manager_salary : ℝ := 3300

theorem number_of_employees : 
  ∃ (E : ℕ), 
    (E * average_salary_without_manager + manager_salary) / (E + 1) = average_salary_with_manager ∧ 
    E = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_employees_l3535_353570


namespace NUMINAMATH_CALUDE_unique_digit_solution_l3535_353500

theorem unique_digit_solution :
  ∃! (digits : Fin 5 → Nat),
    (∀ i, digits i ≠ 0 ∧ digits i ≤ 9) ∧
    (digits 0 + digits 1 = (digits 2 + digits 3 + digits 4) / 7) ∧
    (digits 0 + digits 3 = (digits 1 + digits 2 + digits 4) / 5) := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l3535_353500


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3535_353578

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (1 + Complex.I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3535_353578


namespace NUMINAMATH_CALUDE_tetrahedron_projection_ratio_l3535_353501

/-- Represents a tetrahedron with edge lengths a, b, c, d, e, f -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f
  h_a_greatest : a ≥ b ∧ a ≥ c ∧ a ≥ d ∧ a ≥ e ∧ a ≥ f

/-- The theorem stating that for any tetrahedron, there exist two projection planes
    such that the ratio of their projection areas is not less than √2 -/
theorem tetrahedron_projection_ratio (t : Tetrahedron) :
  ∃ (area₁ area₂ : ℝ), area₁ > 0 ∧ area₂ > 0 ∧ area₂ / area₁ ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_projection_ratio_l3535_353501


namespace NUMINAMATH_CALUDE_expand_expression_l3535_353583

theorem expand_expression (x : ℝ) : (15 * x^2 + 5 - 3 * x) * 3 * x^3 = 45 * x^5 - 9 * x^4 + 15 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3535_353583


namespace NUMINAMATH_CALUDE_rectangle_area_l3535_353564

theorem rectangle_area (square_area : ℝ) (rectangle_width rectangle_length rectangle_area : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_area = rectangle_width * rectangle_length →
  rectangle_area = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3535_353564


namespace NUMINAMATH_CALUDE_triangle_inequality_variant_l3535_353508

theorem triangle_inequality_variant (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_variant_l3535_353508


namespace NUMINAMATH_CALUDE_complex_square_plus_self_l3535_353577

theorem complex_square_plus_self (z : ℂ) (h : z = 1 + I) : z^2 + z = 1 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_plus_self_l3535_353577


namespace NUMINAMATH_CALUDE_triangle_sphere_distance_l3535_353528

/-- The distance between the plane containing a triangle inscribed on a sphere and the center of the sphere -/
theorem triangle_sphere_distance (a b c R : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) (hR : R = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area / s
  Real.sqrt (R^2 - r^2) = 2 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sphere_distance_l3535_353528


namespace NUMINAMATH_CALUDE_distance_ratio_is_one_to_one_l3535_353574

def walking_speed : ℝ := 4
def running_speed : ℝ := 8
def total_time : ℝ := 1.5
def total_distance : ℝ := 8

theorem distance_ratio_is_one_to_one :
  ∃ (d_w d_r : ℝ),
    d_w / walking_speed + d_r / running_speed = total_time ∧
    d_w + d_r = total_distance ∧
    d_w / d_r = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_is_one_to_one_l3535_353574


namespace NUMINAMATH_CALUDE_triangle_properties_l3535_353506

/-- Given a triangle ABC with specific properties, prove its angle A and area. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) : 
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C →
  -- Sum of angles is π
  A + B + C = π →
  -- Side lengths satisfy triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Given conditions
  Real.sin A + Real.sqrt 3 * Real.cos A = 2 →
  a = 2 →
  B = π / 4 →
  -- Prove angle A and area
  A = π / 6 ∧ 
  (1/2 : ℝ) * a * b * Real.sin C = Real.sqrt 3 + 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3535_353506


namespace NUMINAMATH_CALUDE_emily_contribution_l3535_353532

/-- Proves that Emily needs to contribute 3 more euros to buy the pie -/
theorem emily_contribution (pie_cost : ℝ) (emily_usd : ℝ) (berengere_euro : ℝ) (exchange_rate : ℝ) :
  pie_cost = 15 →
  emily_usd = 10 →
  berengere_euro = 3 →
  exchange_rate = 1.1 →
  ∃ (emily_extra : ℝ), emily_extra = 3 ∧ 
    pie_cost = berengere_euro + (emily_usd / exchange_rate) + emily_extra :=
by sorry

end NUMINAMATH_CALUDE_emily_contribution_l3535_353532


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_zero_l3535_353595

def M : Set ℝ := {x : ℝ | |x| ≤ 2}
def N : Set ℝ := {x : ℝ | x^2 - 3*x = 0}

theorem M_intersect_N_equals_zero : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_zero_l3535_353595


namespace NUMINAMATH_CALUDE_alternating_sequence_sum_l3535_353513

def sequence_sum (first last : ℤ) (step : ℤ) : ℤ :=
  let n := (last - first) / step + 1
  let sum := (first + last) * n / 2
  if n % 2 = 0 then -sum else sum

theorem alternating_sequence_sum : 
  sequence_sum 2 74 4 = 38 := by sorry

end NUMINAMATH_CALUDE_alternating_sequence_sum_l3535_353513


namespace NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l3535_353558

/-- The surface area of a sphere inscribed in a triangular pyramid with all edges of length a -/
theorem inscribed_sphere_surface_area (a : ℝ) (h : a > 0) :
  ∃ (r : ℝ), r > 0 ∧ r = a / (2 * Real.sqrt 6) ∧ 
  4 * Real.pi * r^2 = Real.pi * a^2 / 6 :=
sorry

end NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l3535_353558


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3535_353597

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100000 ∧ n < 1000000 ∧
  ∀ i : Nat, i ∈ [0, 1, 2, 3] →
    let three_digit := (n / 10^i) % 1000
    three_digit % 6 = 0 ∨ three_digit % 7 = 0

theorem smallest_valid_number :
  is_valid_number 112642 ∧
  ∀ m : Nat, m < 112642 → ¬(is_valid_number m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3535_353597


namespace NUMINAMATH_CALUDE_evaluate_expression_l3535_353531

theorem evaluate_expression (a b : ℚ) (h1 : a = 5) (h2 : b = -3) : 3 / (a + b) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3535_353531


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3535_353527

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  fourth_quadrant 4 (-3) := by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3535_353527


namespace NUMINAMATH_CALUDE_minimum_eccentricity_sum_l3535_353585

/-- Given two points F₁ and F₂ that are common foci of an ellipse and a hyperbola,
    and P is their common point. -/
structure CommonFociConfig where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ

/-- The eccentricity of the ellipse -/
def eccentricity_ellipse (config : CommonFociConfig) : ℝ := sorry

/-- The eccentricity of the hyperbola -/
def eccentricity_hyperbola (config : CommonFociConfig) : ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem minimum_eccentricity_sum (config : CommonFociConfig) 
  (h1 : distance config.P config.F₂ > distance config.P config.F₁)
  (h2 : distance config.P config.F₁ = distance config.F₁ config.F₂) :
  (∀ e₁ e₂ : ℝ, e₁ = eccentricity_ellipse config → e₂ = eccentricity_hyperbola config →
    3 / e₁ + e₂ / 3 ≥ 8) ∧ 
  (∃ e₁ e₂ : ℝ, e₁ = eccentricity_ellipse config ∧ e₂ = eccentricity_hyperbola config ∧
    3 / e₁ + e₂ / 3 = 8) :=
sorry

end NUMINAMATH_CALUDE_minimum_eccentricity_sum_l3535_353585


namespace NUMINAMATH_CALUDE_john_shopping_expense_l3535_353515

/-- Given John's shopping scenario, prove the amount spent on pants. -/
theorem john_shopping_expense (tshirt_count : ℕ) (tshirt_price : ℕ) (total_spent : ℕ) 
  (h1 : tshirt_count = 3)
  (h2 : tshirt_price = 20)
  (h3 : total_spent = 110) :
  total_spent - (tshirt_count * tshirt_price) = 50 := by
  sorry

end NUMINAMATH_CALUDE_john_shopping_expense_l3535_353515


namespace NUMINAMATH_CALUDE_odd_function_value_l3535_353510

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the theorem
theorem odd_function_value (f g : ℝ → ℝ) 
  (h_odd : OddFunction f)
  (h_g : ∀ x, g x = f x + 6)
  (h_g_neg_one : g (-1) = 3) :
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l3535_353510


namespace NUMINAMATH_CALUDE_equation_solution_l3535_353537

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  4 - 9 / x + 4 / (x^2) = 0 → (3 / x = 12 ∨ 3 / x = 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3535_353537


namespace NUMINAMATH_CALUDE_rope_folding_l3535_353524

theorem rope_folding (n : ℕ) (original_length : ℝ) (h : n = 3) :
  let num_parts := 2^n
  let part_length := original_length / num_parts
  part_length = (1 / 8) * original_length := by
  sorry

end NUMINAMATH_CALUDE_rope_folding_l3535_353524


namespace NUMINAMATH_CALUDE_positive_root_of_cubic_l3535_353592

theorem positive_root_of_cubic (x : ℝ) :
  x = 2 + Real.sqrt 2 →
  x^3 - 4*x^2 + x - 2*Real.sqrt 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_positive_root_of_cubic_l3535_353592


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3535_353548

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0) ↔ k ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3535_353548


namespace NUMINAMATH_CALUDE_pages_left_to_read_l3535_353580

def total_pages : ℕ := 563
def pages_read : ℕ := 147

theorem pages_left_to_read : total_pages - pages_read = 416 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l3535_353580


namespace NUMINAMATH_CALUDE_num_subsets_eq_two_pow_l3535_353571

/-- The number of subsets of a finite set -/
def num_subsets (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of subsets of a set with n elements is 2^n -/
theorem num_subsets_eq_two_pow (n : ℕ) : num_subsets n = 2^n := by
  sorry

end NUMINAMATH_CALUDE_num_subsets_eq_two_pow_l3535_353571


namespace NUMINAMATH_CALUDE_inequality_proof_l3535_353552

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3535_353552
