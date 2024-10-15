import Mathlib

namespace NUMINAMATH_CALUDE_multiples_of_4_between_80_and_300_l3113_311370

theorem multiples_of_4_between_80_and_300 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n > 80 ∧ n < 300) (Finset.range 300)).card = 54 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_4_between_80_and_300_l3113_311370


namespace NUMINAMATH_CALUDE_peanuts_in_jar_l3113_311379

theorem peanuts_in_jar (initial_peanuts : ℕ) : 
  (initial_peanuts : ℚ) - (1/4 : ℚ) * initial_peanuts - 29 = 82 → 
  initial_peanuts = 148 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_jar_l3113_311379


namespace NUMINAMATH_CALUDE_cubic_roots_expression_l3113_311331

theorem cubic_roots_expression (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p - 6 = 0 →
  q^3 - 6*q^2 + 11*q - 6 = 0 →
  r^3 - 6*r^2 + 11*r - 6 = 0 →
  p^3 + q^3 + r^3 - 3*p*q*r = 18 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_expression_l3113_311331


namespace NUMINAMATH_CALUDE_max_a_for_integer_solutions_l3113_311329

theorem max_a_for_integer_solutions : 
  (∃ (a : ℕ+), ∀ (x : ℤ), x^2 + a*x = -30 → 
    (∀ (b : ℕ+), (∀ (y : ℤ), y^2 + b*y = -30 → b ≤ a))) ∧
  (∃ (x : ℤ), x^2 + 31*x = -30) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_integer_solutions_l3113_311329


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3113_311376

theorem sum_of_a_and_b (a b : ℝ) (h1 : a > b) (h2 : |a| = 9) (h3 : b^2 = 4) :
  a + b = 11 ∨ a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3113_311376


namespace NUMINAMATH_CALUDE_tonya_initial_stamps_proof_l3113_311368

/-- The number of matches equivalent to one stamp -/
def matches_per_stamp : ℕ := 12

/-- The number of matches in each matchbook -/
def matches_per_matchbook : ℕ := 24

/-- The number of matchbooks Jimmy has -/
def jimmy_matchbooks : ℕ := 5

/-- The number of stamps Tonya has left after the trade -/
def tonya_stamps_left : ℕ := 3

/-- The initial number of stamps Tonya had -/
def tonya_initial_stamps : ℕ := 13

theorem tonya_initial_stamps_proof :
  tonya_initial_stamps = 
    (jimmy_matchbooks * matches_per_matchbook / matches_per_stamp) + tonya_stamps_left :=
by sorry

end NUMINAMATH_CALUDE_tonya_initial_stamps_proof_l3113_311368


namespace NUMINAMATH_CALUDE_correct_conclusions_l3113_311327

theorem correct_conclusions :
  (∀ a b : ℝ, a + b < 0 ∧ b / a > 0 → |a + 2*b| = -a - 2*b) ∧
  (∀ m : ℚ, |m| + m ≥ 0) ∧
  (∀ a b c : ℝ, c < 0 ∧ 0 < a ∧ a < b → (a - b)*(b - c)*(c - a) > 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_conclusions_l3113_311327


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3113_311305

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • (B^14) = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3113_311305


namespace NUMINAMATH_CALUDE_triangle_base_length_l3113_311354

/-- Given a square with perimeter 40 and a triangle with height 40 that share a side and have equal areas, 
    the base of the triangle is 5. -/
theorem triangle_base_length : 
  ∀ (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ),
    square_perimeter = 40 →
    triangle_height = 40 →
    (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_base * triangle_height →
    triangle_base = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3113_311354


namespace NUMINAMATH_CALUDE_sum_m_2n_3k_l3113_311307

theorem sum_m_2n_3k (m n k : ℕ+) 
  (sum_mn : m + n = 2021)
  (prime_m_3k : Nat.Prime (m - 3*k))
  (prime_n_k : Nat.Prime (n + k)) :
  m + 2*n + 3*k = 2025 ∨ m + 2*n + 3*k = 4040 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_2n_3k_l3113_311307


namespace NUMINAMATH_CALUDE_m_range_when_M_in_fourth_quadrant_l3113_311316

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point M with coordinates dependent on m -/
def M (m : ℝ) : Point :=
  { x := m + 3, y := m - 1 }

/-- Theorem: If M(m) is in the fourth quadrant, then -3 < m < 1 -/
theorem m_range_when_M_in_fourth_quadrant :
  ∀ m : ℝ, in_fourth_quadrant (M m) → -3 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_when_M_in_fourth_quadrant_l3113_311316


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3113_311383

theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 1200)
  (h2 : interest_paid = 432)
  (h3 : ∀ rate time, interest_paid = principal * rate * time / 100 → rate = time) :
  ∃ rate : ℝ, rate = 6 ∧ interest_paid = principal * rate * rate / 100 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3113_311383


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3113_311310

theorem quadratic_inequality (x : ℝ) : -3 * x^2 + 8 * x + 5 > 0 ↔ x < -1/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3113_311310


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_system_2_l3113_311351

-- Equation (1)
theorem solve_equation_1 : 
  ∃ x : ℚ, (3 * x + 2) / 2 - 1 = (2 * x - 1) / 4 ↔ x = -1/4 := by sorry

-- System of equations (2)
theorem solve_system_2 : 
  ∃ x y : ℚ, (3 * x - 2 * y = 9 ∧ 2 * x + 3 * y = 19) ↔ (x = 5 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_system_2_l3113_311351


namespace NUMINAMATH_CALUDE_alices_preferred_numbers_l3113_311303

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

def preferred_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧ 
  n % 7 = 0 ∧
  ¬(n % 3 = 0) ∧
  is_prime (digit_sum n)

theorem alices_preferred_numbers :
  {n : ℕ | preferred_number n} = {119, 133, 140} := by sorry

end NUMINAMATH_CALUDE_alices_preferred_numbers_l3113_311303


namespace NUMINAMATH_CALUDE_angle_A_measure_l3113_311314

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions to ensure it's a valid triangle
  true

-- Define the measure of an angle
def angle_measure (A B C : ℝ × ℝ) : ℝ :=
  sorry

-- Define the length of a side
def side_length (A B : ℝ × ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem angle_A_measure 
  (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_acute : angle_measure A B C < π / 2 ∧ 
             angle_measure B C A < π / 2 ∧ 
             angle_measure C A B < π / 2)
  (h_BC : side_length B C = 3)
  (h_AB : side_length A B = Real.sqrt 6)
  (h_angle_C : angle_measure B C A = π / 4) :
  angle_measure C A B = π / 3 :=
sorry

end NUMINAMATH_CALUDE_angle_A_measure_l3113_311314


namespace NUMINAMATH_CALUDE_circle_C_equation_l3113_311388

/-- The standard equation of a circle with center (h, k) and radius r -/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Circle C with center (1, 2) and radius 3 -/
def circle_C (x y : ℝ) : Prop :=
  standard_circle_equation x y 1 2 3

theorem circle_C_equation :
  ∀ x y : ℝ, circle_C x y ↔ (x - 1)^2 + (y - 2)^2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_C_equation_l3113_311388


namespace NUMINAMATH_CALUDE_subset_inequality_l3113_311361

-- Define the set S_n
def S_n (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

-- Define the properties of function f
def is_valid_f (n : ℕ) (f : Set ℕ → ℝ) : Prop :=
  (∀ A : Set ℕ, A ⊆ S_n n → f A > 0) ∧
  (∀ A : Set ℕ, ∀ x y : ℕ, A ⊆ S_n n → x ∈ S_n n → y ∈ S_n n → x ≠ y →
    f (A ∪ {x}) * f (A ∪ {y}) ≤ f (A ∪ {x, y}) * f A)

-- State the theorem
theorem subset_inequality (n : ℕ) (f : Set ℕ → ℝ) (h : is_valid_f n f) :
  ∀ A B : Set ℕ, A ⊆ S_n n → B ⊆ S_n n →
    f A * f B ≤ f (A ∪ B) * f (A ∩ B) :=
sorry

end NUMINAMATH_CALUDE_subset_inequality_l3113_311361


namespace NUMINAMATH_CALUDE_line_through_P_intersecting_C_l3113_311302

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y - 5 = 0

-- Define the point P
def point_P : ℝ × ℝ := (5, 0)

-- Define the chord length
def chord_length : ℝ := 8

-- Define the two possible line equations
def line_eq1 (x : ℝ) : Prop := x = 5
def line_eq2 (x y : ℝ) : Prop := 7*x + 24*y - 35 = 0

-- Theorem statement
theorem line_through_P_intersecting_C :
  ∃ (l : ℝ → ℝ → Prop),
    (∀ x y, l x y → (x = point_P.1 ∧ y = point_P.2)) ∧
    (∃ x1 y1 x2 y2, l x1 y1 ∧ l x2 y2 ∧ circle_C x1 y1 ∧ circle_C x2 y2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2) ∧
    ((∀ x y, l x y ↔ line_eq1 x) ∨ (∀ x y, l x y ↔ line_eq2 x y)) :=
sorry

end NUMINAMATH_CALUDE_line_through_P_intersecting_C_l3113_311302


namespace NUMINAMATH_CALUDE_diaries_calculation_l3113_311318

/-- Calculates the number of diaries after doubling and losing a quarter --/
def diaries_after_change (initial : ℕ) : ℕ :=
  let doubled := initial * 2
  let total := initial + doubled
  total - (total / 4)

/-- Theorem stating that starting with 8 diaries results in 18 after changes --/
theorem diaries_calculation : diaries_after_change 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_diaries_calculation_l3113_311318


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3113_311398

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 12 ∧ x ≠ -4 →
    (6 * x + 15) / (x^2 - 8*x - 48) = (87/16) / (x - 12) + (9/16) / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3113_311398


namespace NUMINAMATH_CALUDE_base_conversion_1850_to_base_7_l3113_311377

theorem base_conversion_1850_to_base_7 :
  (5 * 7^3 + 2 * 7^2 + 5 * 7^1 + 2 * 7^0 : ℕ) = 1850 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1850_to_base_7_l3113_311377


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3113_311364

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 + 6*x + 5 = 0 ↔ x = -1 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3113_311364


namespace NUMINAMATH_CALUDE_freddy_age_l3113_311334

/-- Represents the ages of three children --/
structure ChildrenAges where
  matthew : ℕ
  rebecca : ℕ
  freddy : ℕ

/-- The conditions of the problem --/
def problem_conditions (ages : ChildrenAges) : Prop :=
  ages.matthew + ages.rebecca + ages.freddy = 35 ∧
  ages.matthew = ages.rebecca + 2 ∧
  ages.freddy = ages.matthew + 4

/-- The theorem stating that under the given conditions, Freddy is 15 years old --/
theorem freddy_age (ages : ChildrenAges) : 
  problem_conditions ages → ages.freddy = 15 := by
  sorry


end NUMINAMATH_CALUDE_freddy_age_l3113_311334


namespace NUMINAMATH_CALUDE_characterization_of_function_l3113_311333

theorem characterization_of_function (f : ℤ → ℝ) 
  (h1 : ∀ m n : ℤ, m < n → f m < f n)
  (h2 : ∀ m n : ℤ, ∃ k : ℤ, f m - f n = f k) :
  ∃ a : ℝ, ∃ t : ℤ, a > 0 ∧ ∀ n : ℤ, f n = a * (n + t) := by
sorry

end NUMINAMATH_CALUDE_characterization_of_function_l3113_311333


namespace NUMINAMATH_CALUDE_least_five_digit_square_and_cube_l3113_311319

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem least_five_digit_square_and_cube : 
  (is_five_digit 15625 ∧ is_perfect_square 15625 ∧ is_perfect_cube 15625) ∧ 
  (∀ n : ℕ, n < 15625 → ¬(is_five_digit n ∧ is_perfect_square n ∧ is_perfect_cube n)) :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_and_cube_l3113_311319


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3113_311324

/-- Represents a quadrilateral EFGH with given side lengths -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)
  (EG : ℕ)

/-- The specific quadrilateral from the problem -/
def problem_quadrilateral : Quadrilateral :=
  { EF := 7
  , FG := 21
  , GH := 7
  , HE := 13
  , EG := 21 }

/-- Triangle inequality theorem -/
axiom triangle_inequality {a b c : ℝ} : a + b > c

theorem quadrilateral_diagonal_length : 
  ∀ q : Quadrilateral, 
  q.EF = 7 → q.FG = 21 → q.GH = 7 → q.HE = 13 → 
  q.EG = problem_quadrilateral.EG :=
by
  sorry

#check quadrilateral_diagonal_length

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3113_311324


namespace NUMINAMATH_CALUDE_dennis_marbles_l3113_311315

theorem dennis_marbles (laurie kurt dennis : ℕ) 
  (h1 : laurie = kurt + 12)
  (h2 : kurt + 45 = dennis)
  (h3 : laurie = 37) : 
  dennis = 70 := by
sorry

end NUMINAMATH_CALUDE_dennis_marbles_l3113_311315


namespace NUMINAMATH_CALUDE_largest_palindrome_divisible_by_15_l3113_311312

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

end NUMINAMATH_CALUDE_largest_palindrome_divisible_by_15_l3113_311312


namespace NUMINAMATH_CALUDE_distance_to_asymptotes_l3113_311392

/-- The distance from point P(0,1) to the asymptotes of the hyperbola y²/4 - x² = 1 is √5/5 -/
theorem distance_to_asymptotes (x y : ℝ) : 
  let P : ℝ × ℝ := (0, 1)
  let hyperbola := {(x, y) | y^2/4 - x^2 = 1}
  let asymptote (m : ℝ) := {(x, y) | y = m*x}
  let distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) := 
    |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)
  ∃ (m : ℝ), m^2 = 4 ∧ 
    distance_point_to_line P m (-1) 0 = Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_asymptotes_l3113_311392


namespace NUMINAMATH_CALUDE_midpoint_triangle_area_l3113_311359

/-- The area of a triangle formed by midpoints in a square --/
theorem midpoint_triangle_area (s : ℝ) (h : s = 12) :
  let square_area := s^2
  let midpoint_triangle_area := s^2 / 8
  midpoint_triangle_area = 18 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_triangle_area_l3113_311359


namespace NUMINAMATH_CALUDE_quadratic_equation_h_value_l3113_311378

theorem quadratic_equation_h_value (h : ℝ) : 
  (∃ r s : ℝ, r^2 + 2*h*r = 3 ∧ s^2 + 2*h*s = 3 ∧ r^2 + s^2 = 10) → 
  |h| = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_h_value_l3113_311378


namespace NUMINAMATH_CALUDE_sector_area_special_case_l3113_311340

/-- The area of a sector with arc length and central angle both equal to 5 is 5/2 -/
theorem sector_area_special_case :
  ∀ (l α : ℝ), l = 5 → α = 5 → (1/2) * l * (l / α) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_special_case_l3113_311340


namespace NUMINAMATH_CALUDE_newcomer_weight_l3113_311336

/-- Represents the weight of a group of people -/
structure GroupWeight where
  initial : ℝ
  new : ℝ

/-- The problem setup -/
def weightProblem (g : GroupWeight) : Prop :=
  -- Initial weight is between 400 kg and 420 kg
  400 ≤ g.initial ∧ g.initial ≤ 420 ∧
  -- The average weight increase is 3.5 kg
  g.new = g.initial - 47 + 68 ∧
  -- The average weight increases by 3.5 kg
  (g.new / 6) - (g.initial / 6) = 3.5

/-- The theorem to prove -/
theorem newcomer_weight (g : GroupWeight) : 
  weightProblem g → 68 = g.new - g.initial + 47 := by
  sorry


end NUMINAMATH_CALUDE_newcomer_weight_l3113_311336


namespace NUMINAMATH_CALUDE_smallest_n_for_432n_perfect_square_l3113_311352

theorem smallest_n_for_432n_perfect_square :
  ∃ (n : ℕ), n > 0 ∧ (∃ (k : ℕ), 432 * n = k^2) ∧
  (∀ (m : ℕ), m > 0 → m < n → ¬∃ (j : ℕ), 432 * m = j^2) ∧
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_432n_perfect_square_l3113_311352


namespace NUMINAMATH_CALUDE_smallest_colors_l3113_311313

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

end NUMINAMATH_CALUDE_smallest_colors_l3113_311313


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3113_311367

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem stating the properties of the specific arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h1 : seq.S 6 = 51)
    (h2 : seq.a 1 + seq.a 9 = 26) :
  seq.d = 3 ∧ ∀ n, seq.a n = 3 * n - 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3113_311367


namespace NUMINAMATH_CALUDE_log_base_2_derivative_l3113_311395

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_log_base_2_derivative_l3113_311395


namespace NUMINAMATH_CALUDE_solution_comparison_l3113_311306

theorem solution_comparison (p p' q q' : ℕ+) (hp : p ≠ p') (hq : q ≠ q') :
  (-q : ℚ) / p > (-q' : ℚ) / p' ↔ q * p' < p * q' :=
sorry

end NUMINAMATH_CALUDE_solution_comparison_l3113_311306


namespace NUMINAMATH_CALUDE_foggy_day_walk_l3113_311338

/-- Represents a person walking on a straight road -/
structure Walker where
  speed : ℝ
  position : ℝ

/-- The problem setup and solution -/
theorem foggy_day_walk (visibility : ℝ) (alex ben : Walker) (initial_time : ℝ) :
  visibility = 100 →
  alex.speed = 4 →
  ben.speed = 6 →
  initial_time = 60 →
  alex.position = alex.speed * initial_time →
  ben.position = ben.speed * initial_time →
  ∃ (meeting_time : ℝ),
    meeting_time = 50 ∧
    abs (alex.position - alex.speed * meeting_time - (ben.position - ben.speed * meeting_time)) = visibility ∧
    abs (alex.position - alex.speed * meeting_time) = 40 ∧
    abs (ben.position - ben.speed * meeting_time) = 60 :=
by sorry

end NUMINAMATH_CALUDE_foggy_day_walk_l3113_311338


namespace NUMINAMATH_CALUDE_age_height_not_function_l3113_311397

-- Define the relationships
def angle_sine_relation : Set (ℝ × ℝ) := sorry
def square_side_area_relation : Set (ℝ × ℝ) := sorry
def polygon_sides_angles_relation : Set (ℕ × ℝ) := sorry
def age_height_relation : Set (ℕ × ℝ) := sorry

-- Define the property of being a function
def is_function (r : Set (α × β)) : Prop := 
  ∀ x y z, (x, y) ∈ r → (x, z) ∈ r → y = z

-- State the theorem
theorem age_height_not_function :
  is_function angle_sine_relation ∧ 
  is_function square_side_area_relation ∧ 
  is_function polygon_sides_angles_relation → 
  ¬ is_function age_height_relation := by
sorry

end NUMINAMATH_CALUDE_age_height_not_function_l3113_311397


namespace NUMINAMATH_CALUDE_number_of_women_is_six_l3113_311326

/-- The number of women in a group that can color 360 meters of cloth in 3 days,
    given that 5 women can color 100 meters of cloth in 1 day. -/
def number_of_women : ℕ :=
  let meters_per_day := 360 / 3
  let meters_per_woman_per_day := 100 / 5
  meters_per_day / meters_per_woman_per_day

theorem number_of_women_is_six : number_of_women = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_women_is_six_l3113_311326


namespace NUMINAMATH_CALUDE_remainder_of_x_120_divided_by_x2_minus_4x_plus_3_l3113_311373

theorem remainder_of_x_120_divided_by_x2_minus_4x_plus_3 :
  ∀ (x : ℝ), ∃ (Q : ℝ → ℝ),
    x^120 = (x^2 - 4*x + 3) * Q x + ((3^120 - 1)*x + (3 - 3^120)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_x_120_divided_by_x2_minus_4x_plus_3_l3113_311373


namespace NUMINAMATH_CALUDE_complex_modulus_range_l3113_311355

theorem complex_modulus_range (a : ℝ) :
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) →
  -Real.sqrt 5 / 5 ≤ a ∧ a ≤ Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l3113_311355


namespace NUMINAMATH_CALUDE_quadratic_equation_a_range_l3113_311304

/-- The range of values for a in the quadratic equation (a-1)x^2 + √(a+1)x + 2 = 0 -/
theorem quadratic_equation_a_range :
  ∀ a : ℝ, (∃ x : ℝ, (a - 1) * x^2 + Real.sqrt (a + 1) * x + 2 = 0) →
  (a ≥ -1 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_a_range_l3113_311304


namespace NUMINAMATH_CALUDE_smallest_positive_period_sin_cos_l3113_311394

/-- The smallest positive period of f(x) = sin x cos x is π -/
theorem smallest_positive_period_sin_cos (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x * Real.cos x) :
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = π :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_sin_cos_l3113_311394


namespace NUMINAMATH_CALUDE_tea_boxes_problem_l3113_311342

/-- Proves that if there are four boxes of tea, and after removing 9 kg from each box,
    the total remaining quantity equals the original quantity in one box,
    then each box initially contained 12 kg of tea. -/
theorem tea_boxes_problem (x : ℝ) : 
  (4 * (x - 9) = x) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_tea_boxes_problem_l3113_311342


namespace NUMINAMATH_CALUDE_least_positive_integer_for_reducible_fraction_l3113_311325

theorem least_positive_integer_for_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), 0 < k ∧ k < n → ¬(∃ (d : ℕ), d > 1 ∧ d ∣ (k - 20) ∧ d ∣ (7 * k + 2))) ∧
  (∃ (d : ℕ), d > 1 ∧ d ∣ (n - 20) ∧ d ∣ (7 * n + 2)) ∧
  n = 22 :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_reducible_fraction_l3113_311325


namespace NUMINAMATH_CALUDE_min_cone_volume_with_sphere_l3113_311348

/-- The minimum volume of a cone that contains a sphere of radius 1 touching its base -/
theorem min_cone_volume_with_sphere (r : ℝ) (h : r = 1) : 
  ∃ (V : ℝ), V = Real.pi * 8 / 3 ∧ 
  (∀ (cone_volume : ℝ), 
    (∃ (R h : ℝ), 
      cone_volume = Real.pi * R^2 * h / 3 ∧ 
      r^2 + (R - r)^2 = h^2) → 
    V ≤ cone_volume) :=
by sorry

end NUMINAMATH_CALUDE_min_cone_volume_with_sphere_l3113_311348


namespace NUMINAMATH_CALUDE_min_abs_z_min_abs_z_achievable_l3113_311362

open Complex

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 5*I) + Complex.abs (z - 6) = 7) : 
  Complex.abs z ≥ 30 / Real.sqrt 61 := by
  sorry

theorem min_abs_z_achievable : ∃ z : ℂ, 
  (Complex.abs (z - 5*I) + Complex.abs (z - 6) = 7) ∧ 
  (Complex.abs z = 30 / Real.sqrt 61) := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_min_abs_z_achievable_l3113_311362


namespace NUMINAMATH_CALUDE_volume_maximized_at_10cm_l3113_311347

/-- Represents the dimensions of the original sheet --/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the volume of the container given sheet dimensions and cut length --/
def containerVolume (sheet : SheetDimensions) (cutLength : ℝ) : ℝ :=
  (sheet.length - 2 * cutLength) * (sheet.width - 2 * cutLength) * cutLength

/-- Theorem stating that the volume is maximized when cut length is 10cm --/
theorem volume_maximized_at_10cm (sheet : SheetDimensions) 
  (h1 : sheet.length = 90)
  (h2 : sheet.width = 48) :
  ∃ (maxCutLength : ℝ), maxCutLength = 10 ∧ 
  ∀ (x : ℝ), 0 < x → x < sheet.width / 2 → x < sheet.length / 2 → 
  containerVolume sheet x ≤ containerVolume sheet maxCutLength :=
sorry

end NUMINAMATH_CALUDE_volume_maximized_at_10cm_l3113_311347


namespace NUMINAMATH_CALUDE_zeros_of_g_l3113_311396

/-- Given a linear function f(x) = ax + b with a zero at x = 1,
    prove that the zeros of g(x) = bx^2 - ax are 0 and -1 -/
theorem zeros_of_g (a b : ℝ) (h : a + b = 0) (ha : a ≠ 0) :
  let f := λ x : ℝ => a * x + b
  let g := λ x : ℝ => b * x^2 - a * x
  (∀ x : ℝ, g x = 0 ↔ x = 0 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_g_l3113_311396


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l3113_311353

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

end NUMINAMATH_CALUDE_systematic_sample_theorem_l3113_311353


namespace NUMINAMATH_CALUDE_part_one_part_two_l3113_311371

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, f a x < 4 - 2*a ↔ -4 < x ∧ x < 4) → a = 0 := by sorry

-- Part 2
theorem part_two : 
  (∀ x : ℝ, f 1 x - f 1 (-2*x) ≤ x + 2) ∧ 
  (∀ m : ℝ, (∀ x : ℝ, f 1 x - f 1 (-2*x) ≤ x + m) → m ≥ 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3113_311371


namespace NUMINAMATH_CALUDE_first_day_over_500_is_saturday_l3113_311357

def days : List String := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

def pens_on_day (day : Nat) : Nat :=
  if day = 0 then 5
  else if day = 1 then 10
  else 10 * (3 ^ (day - 1))

def first_day_over_500 : String :=
  days[(days.findIdx? (fun d => pens_on_day (days.indexOf d) > 500)).getD 0]

theorem first_day_over_500_is_saturday : first_day_over_500 = "Saturday" := by
  sorry

end NUMINAMATH_CALUDE_first_day_over_500_is_saturday_l3113_311357


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3113_311365

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 400 →
  crossing_time = 45 →
  train_speed = 55.99999999999999 →
  ∃ (bridge_length : ℝ), bridge_length = train_speed * crossing_time - train_length ∧
                         bridge_length = 2120 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3113_311365


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_pow6_l3113_311369

theorem nearest_integer_to_3_plus_sqrt2_pow6 :
  ∃ n : ℤ, n = 7414 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2) ^ 6 - n| ≤ |((3 : ℝ) + Real.sqrt 2) ^ 6 - m| :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_pow6_l3113_311369


namespace NUMINAMATH_CALUDE_karens_paddling_speed_l3113_311374

/-- Karen's canoe paddling problem -/
theorem karens_paddling_speed
  (river_current : ℝ)
  (river_length : ℝ)
  (paddling_time : ℝ)
  (h1 : river_current = 4)
  (h2 : river_length = 12)
  (h3 : paddling_time = 2)
  : ∃ (still_water_speed : ℝ),
    still_water_speed = 10 ∧
    river_length = (still_water_speed - river_current) * paddling_time :=
by sorry

end NUMINAMATH_CALUDE_karens_paddling_speed_l3113_311374


namespace NUMINAMATH_CALUDE_gcd_of_198_and_308_l3113_311375

theorem gcd_of_198_and_308 : Nat.gcd 198 308 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_198_and_308_l3113_311375


namespace NUMINAMATH_CALUDE_real_part_of_reciprocal_l3113_311337

theorem real_part_of_reciprocal (x y : ℝ) (z : ℂ) (h1 : z = x + y * I) (h2 : z ≠ x) (h3 : Complex.abs z = 1) :
  (1 / (2 - z)).re = (2 - x) / (5 - 4 * x) := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_reciprocal_l3113_311337


namespace NUMINAMATH_CALUDE_yan_distance_ratio_l3113_311387

/-- Yan's scenario with distances and speeds -/
structure YanScenario where
  a : ℝ  -- distance from Yan to home
  b : ℝ  -- distance from Yan to mall
  w : ℝ  -- Yan's walking speed
  bike_speed : ℝ -- Yan's bicycle speed

/-- The conditions of Yan's scenario -/
def valid_scenario (s : YanScenario) : Prop :=
  s.a > 0 ∧ s.b > 0 ∧ s.w > 0 ∧
  s.bike_speed = 5 * s.w ∧
  s.b / s.w = s.a / s.w + (s.a + s.b) / s.bike_speed

/-- The theorem stating the ratio of distances -/
theorem yan_distance_ratio (s : YanScenario) (h : valid_scenario s) :
  s.a / s.b = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_yan_distance_ratio_l3113_311387


namespace NUMINAMATH_CALUDE_solve_equation_l3113_311386

theorem solve_equation (x : ℝ) : 2 * x = (26 - x) + 19 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3113_311386


namespace NUMINAMATH_CALUDE_set_relations_l3113_311344

open Set

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

theorem set_relations (m : ℝ) :
  (A m ⊆ B ↔ m < 2 ∨ m > 4) ∧
  (A m ∩ B = ∅ ↔ m ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_set_relations_l3113_311344


namespace NUMINAMATH_CALUDE_translation_downward_3_units_l3113_311300

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Translates a linear function vertically -/
def translate_vertical (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + units }

theorem translation_downward_3_units :
  let original := LinearFunction.mk 3 2
  let translated := translate_vertical original (-3)
  translated = LinearFunction.mk 3 (-1) := by sorry

end NUMINAMATH_CALUDE_translation_downward_3_units_l3113_311300


namespace NUMINAMATH_CALUDE_quadratic_radical_simplification_l3113_311330

theorem quadratic_radical_simplification (a m n : ℕ+) :
  (a : ℝ) + 2 * Real.sqrt 21 = (Real.sqrt (m : ℝ) + Real.sqrt (n : ℝ))^2 →
  a = 10 ∨ a = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_simplification_l3113_311330


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3113_311349

theorem complex_fraction_equality : (4 - 2*Complex.I) / (1 + Complex.I) = 1 - 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3113_311349


namespace NUMINAMATH_CALUDE_quadratic_discriminant_relationship_l3113_311311

/-- The discriminant of a quadratic equation ax^2 + 2bx + c = 0 is 1 -/
def discriminant_is_one (a b c : ℝ) : Prop :=
  (2 * b)^2 - 4 * a * c = 1

/-- The relationship between a, b, and c -/
def relationship (a b c : ℝ) : Prop :=
  b^2 - a * c = 1/4

/-- Theorem: If the discriminant of ax^2 + 2bx + c = 0 is 1, 
    then b^2 - ac = 1/4 -/
theorem quadratic_discriminant_relationship 
  (a b c : ℝ) : discriminant_is_one a b c → relationship a b c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_relationship_l3113_311311


namespace NUMINAMATH_CALUDE_black_friday_sales_l3113_311380

/-- Proves that if a store sells 477 televisions three years from now, 
    and the number of televisions sold increases by 50 each year, 
    then the store sold 327 televisions this year. -/
theorem black_friday_sales (current_sales : ℕ) : 
  (current_sales + 3 * 50 = 477) → current_sales = 327 := by
  sorry

end NUMINAMATH_CALUDE_black_friday_sales_l3113_311380


namespace NUMINAMATH_CALUDE_new_average_production_l3113_311391

theorem new_average_production (n : ℕ) (past_average : ℝ) (today_production : ℝ) 
  (h1 : n = 11)
  (h2 : past_average = 50)
  (h3 : today_production = 110) : 
  (n * past_average + today_production) / (n + 1) = 55 :=
by sorry

end NUMINAMATH_CALUDE_new_average_production_l3113_311391


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3113_311328

theorem quadratic_equation_roots (x : ℝ) : ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ (x^2 - 4*x - 3 = 0 ↔ x = r₁ ∨ x = r₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3113_311328


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l3113_311382

theorem sum_of_reciprocals_positive 
  (a b c d : ℝ) 
  (ha : |a| > 1) 
  (hb : |b| > 1) 
  (hc : |c| > 1) 
  (hd : |d| > 1) 
  (h_sum : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) : 
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l3113_311382


namespace NUMINAMATH_CALUDE_eddie_earnings_l3113_311323

-- Define the work hours for each day
def monday_hours : ℚ := 5/2
def tuesday_hours : ℚ := 7/6
def wednesday_hours : ℚ := 7/4
def saturday_hours : ℚ := 3/4

-- Define the pay rates
def weekday_rate : ℚ := 4
def saturday_rate : ℚ := 6

-- Define the total earnings
def total_earnings : ℚ := 
  monday_hours * weekday_rate + 
  tuesday_hours * weekday_rate + 
  wednesday_hours * weekday_rate + 
  saturday_hours * saturday_rate

-- Theorem to prove
theorem eddie_earnings : total_earnings = 26.17 := by
  sorry

end NUMINAMATH_CALUDE_eddie_earnings_l3113_311323


namespace NUMINAMATH_CALUDE_box_dimensions_theorem_l3113_311309

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  smallest : ℝ
  middle : ℝ
  largest : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (d : BoxDimensions) : Prop :=
  d.smallest + d.largest = 17 ∧
  d.smallest + d.middle = 13 ∧
  d.middle + d.largest = 20

/-- The theorem to prove -/
theorem box_dimensions_theorem (d : BoxDimensions) :
  satisfiesConditions d → d = BoxDimensions.mk 5 8 12 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_theorem_l3113_311309


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3113_311332

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, 3 * x * (x - 1) = 2 * (x + 2) + 8 ↔ a * x^2 + b * x + c = 0) →
  a = 3 ∧ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3113_311332


namespace NUMINAMATH_CALUDE_product_of_roots_l3113_311322

theorem product_of_roots (a b c : ℂ) : 
  (3 * a^3 - 4 * a^2 - 12 * a + 9 = 0) →
  (3 * b^3 - 4 * b^2 - 12 * b + 9 = 0) →
  (3 * c^3 - 4 * c^2 - 12 * c + 9 = 0) →
  a * b * c = -3 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l3113_311322


namespace NUMINAMATH_CALUDE_min_value_expression_l3113_311341

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + 1 / (a + b)^3 ≥ (4 : ℝ)^(1/3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3113_311341


namespace NUMINAMATH_CALUDE_union_of_sets_l3113_311358

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4}
  let B : Set ℕ := {1, 3, 5, 7}
  A ∪ B = {1, 2, 3, 4, 5, 7} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3113_311358


namespace NUMINAMATH_CALUDE_intersection_sum_l3113_311346

/-- Two circles intersecting at (1, 3) and (m, n) with centers on x - y - 2 = 0 --/
structure IntersectingCircles where
  m : ℝ
  n : ℝ
  centers_on_line : ∀ (x y : ℝ), (x - y - 2 = 0) → (∃ (r : ℝ), (x - 1)^2 + (y - 3)^2 = r^2 ∧ (x - m)^2 + (y - n)^2 = r^2)

/-- The sum of coordinates of the second intersection point is 4 --/
theorem intersection_sum (c : IntersectingCircles) : c.m + c.n = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3113_311346


namespace NUMINAMATH_CALUDE_sequence_properties_and_sum_l3113_311321

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₁ + (n - 1) * d

def geometric_sequence (b₁ : ℕ) (q : ℕ) : ℕ → ℕ
  | n => b₁ * q^(n - 1)

def merge_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  sorry

theorem sequence_properties_and_sum :
  ∀ (a b : ℕ → ℕ),
    (a 1 = 1) →
    (∀ n, a (b n) = 2^(n+1) - 1) →
    (∀ n, a n = 2*n - 1) →
    (∀ n, b n = 2^n) →
    merge_and_sum a b 100 = 8903 :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_and_sum_l3113_311321


namespace NUMINAMATH_CALUDE_almost_every_graph_chromatic_number_l3113_311399

-- Define the random graph model
structure RandomGraph (n : ℕ) (p : ℝ) where
  -- Add necessary fields here

-- Define the chromatic number
def chromaticNumber (G : RandomGraph n p) : ℝ := sorry

-- Main theorem
theorem almost_every_graph_chromatic_number 
  (p : ℝ) (ε : ℝ) (n : ℕ) (h_p : 0 < p ∧ p < 1) (h_ε : ε > 0) :
  ∃ (G : RandomGraph n p), 
    chromaticNumber G > (Real.log (1 / (1 - p))) / (2 + ε) * (n / Real.log n) := by
  sorry

end NUMINAMATH_CALUDE_almost_every_graph_chromatic_number_l3113_311399


namespace NUMINAMATH_CALUDE_flight_cost_B_to_C_l3113_311301

/-- Represents a city in a triangular configuration -/
inductive City
| A
| B
| C

/-- Represents the distance between two cities in kilometers -/
def distance (x y : City) : ℝ :=
  match x, y with
  | City.A, City.C => 3000
  | City.B, City.C => 1000
  | _, _ => 0  -- We don't need other distances for this problem

/-- The booking fee for a flight in dollars -/
def bookingFee : ℝ := 100

/-- The cost per kilometer for a flight in dollars -/
def costPerKm : ℝ := 0.1

/-- Calculates the cost of a flight between two cities -/
def flightCost (x y : City) : ℝ :=
  bookingFee + costPerKm * distance x y

/-- States that cities A, B, and C form a right-angled triangle with C as the right angle -/
axiom right_angle_at_C : distance City.A City.B ^ 2 = distance City.A City.C ^ 2 + distance City.B City.C ^ 2

theorem flight_cost_B_to_C :
  flightCost City.B City.C = 200 := by
  sorry

end NUMINAMATH_CALUDE_flight_cost_B_to_C_l3113_311301


namespace NUMINAMATH_CALUDE_nellie_legos_l3113_311372

theorem nellie_legos (initial : ℕ) (lost : ℕ) (given_away : ℕ) :
  initial ≥ lost + given_away →
  initial - (lost + given_away) = initial - lost - given_away :=
by
  sorry

#check nellie_legos 380 57 24

end NUMINAMATH_CALUDE_nellie_legos_l3113_311372


namespace NUMINAMATH_CALUDE_fraction_sign_change_l3113_311350

theorem fraction_sign_change (a b : ℝ) (hb : b ≠ 0) : (-a) / (-b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_sign_change_l3113_311350


namespace NUMINAMATH_CALUDE_computation_proof_l3113_311356

theorem computation_proof : 18 * (216 / 3 + 36 / 6 + 4 / 9 + 2 + 1 / 18) = 1449 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l3113_311356


namespace NUMINAMATH_CALUDE_bill_calculation_correct_l3113_311389

/-- Calculates the final bill amount after late charges and fees --/
def finalBillAmount (originalBill : ℝ) (firstLateChargeRate : ℝ) (secondLateChargeRate : ℝ) (flatFee : ℝ) : ℝ :=
  ((originalBill * (1 + firstLateChargeRate)) * (1 + secondLateChargeRate)) + flatFee

/-- Proves that the final bill amount is correct given the specified conditions --/
theorem bill_calculation_correct :
  finalBillAmount 500 0.01 0.02 5 = 520.1 := by
  sorry

#eval finalBillAmount 500 0.01 0.02 5

end NUMINAMATH_CALUDE_bill_calculation_correct_l3113_311389


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l3113_311390

theorem inequality_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ((a + 1) * x > 2 * a + 2) ↔ (x < 2)) →
  a < -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l3113_311390


namespace NUMINAMATH_CALUDE_platform_length_l3113_311384

/-- The length of a platform given train passing times -/
theorem platform_length (train_length : ℝ) (time_pass_man : ℝ) (time_cross_platform : ℝ) 
  (h1 : train_length = 178)
  (h2 : time_pass_man = 8)
  (h3 : time_cross_platform = 20) :
  let train_speed := train_length / time_pass_man
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 267 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l3113_311384


namespace NUMINAMATH_CALUDE_hotel_towels_l3113_311366

/-- Calculates the total number of towels handed out in a hotel --/
def total_towels (num_rooms : ℕ) (people_per_room : ℕ) (towels_per_person : ℕ) : ℕ :=
  num_rooms * people_per_room * towels_per_person

/-- Proves that a hotel with 10 full rooms, 3 people per room, and 2 towels per person hands out 60 towels --/
theorem hotel_towels : total_towels 10 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_hotel_towels_l3113_311366


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3113_311317

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 3)

theorem perpendicular_vectors (m : ℝ) : 
  (a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3113_311317


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_foci_and_eccentricity_l3113_311393

/-- A hyperbola with given foci and eccentricity -/
structure Hyperbola where
  foci : ℝ × ℝ × ℝ × ℝ  -- Represents (x₁, y₁, x₂, y₂)
  eccentricity : ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 2 = 1

/-- Theorem stating that a hyperbola with given foci and eccentricity has the specified equation -/
theorem hyperbola_equation_from_foci_and_eccentricity (h : Hyperbola)
    (h_foci : h.foci = (-2, 0, 2, 0))
    (h_eccentricity : h.eccentricity = Real.sqrt 2) :
    ∀ x y, hyperbola_equation h x y :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_foci_and_eccentricity_l3113_311393


namespace NUMINAMATH_CALUDE_square_area_percent_difference_l3113_311335

theorem square_area_percent_difference (A B : ℝ) (h : A > B) :
  (A^2 - B^2) / B^2 * 100 = 100 * (A^2 - B^2) / B^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_percent_difference_l3113_311335


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l3113_311360

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome :
  ∃ (n : ℕ), n > 6 ∧ 
    isPalindrome n 2 ∧ 
    isPalindrome n 4 ∧ 
    (∀ m : ℕ, m > 6 ∧ m < n → ¬(isPalindrome m 2 ∧ isPalindrome m 4)) ∧
    n = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l3113_311360


namespace NUMINAMATH_CALUDE_total_cost_to_fill_displays_l3113_311339

-- Define the jewelry types
inductive JewelryType
| Necklace
| Ring
| Bracelet

-- Define the structure for jewelry information
structure JewelryInfo where
  capacity : Nat
  current : Nat
  price : Nat
  discountRules : List (Nat × Nat)

-- Define the jewelry store inventory
def inventory : JewelryType → JewelryInfo
| JewelryType.Necklace => ⟨12, 5, 4, [(4, 10), (6, 15)]⟩
| JewelryType.Ring => ⟨30, 18, 10, [(10, 5), (20, 10)]⟩
| JewelryType.Bracelet => ⟨15, 8, 5, [(7, 8), (10, 12)]⟩

-- Function to calculate the discounted price
def calculateDiscountedPrice (info : JewelryInfo) (quantity : Nat) : Nat :=
  let totalPrice := quantity * info.price
  let applicableDiscount := info.discountRules.foldl
    (fun acc (threshold, discount) => if quantity ≥ threshold then max acc discount else acc)
    0
  totalPrice - totalPrice * applicableDiscount / 100

-- Theorem statement
theorem total_cost_to_fill_displays :
  (calculateDiscountedPrice (inventory JewelryType.Necklace) (12 - 5)) +
  (calculateDiscountedPrice (inventory JewelryType.Ring) (30 - 18)) +
  (calculateDiscountedPrice (inventory JewelryType.Bracelet) (15 - 8)) = 170 := by
  sorry


end NUMINAMATH_CALUDE_total_cost_to_fill_displays_l3113_311339


namespace NUMINAMATH_CALUDE_eight_by_eight_diagonal_shaded_count_l3113_311345

/-- Represents a square grid with a diagonal shading pattern -/
structure DiagonalGrid where
  size : Nat
  shaded_rows : Nat
  shaded_per_row : Nat

/-- Calculates the total number of shaded squares in a DiagonalGrid -/
def total_shaded (grid : DiagonalGrid) : Nat :=
  grid.shaded_rows * grid.shaded_per_row

/-- Theorem stating that an 8×8 grid with 7 shaded rows and 7 shaded squares per row has 49 total shaded squares -/
theorem eight_by_eight_diagonal_shaded_count :
  ∀ (grid : DiagonalGrid),
    grid.size = 8 →
    grid.shaded_rows = 7 →
    grid.shaded_per_row = 7 →
    total_shaded grid = 49 := by
  sorry

end NUMINAMATH_CALUDE_eight_by_eight_diagonal_shaded_count_l3113_311345


namespace NUMINAMATH_CALUDE_shoes_sales_goal_l3113_311363

/-- Given a monthly goal and the number of shoes sold in two weeks, 
    calculate the additional pairs needed to meet the goal -/
def additional_pairs_needed (monthly_goal : ℕ) (sold_week1 : ℕ) (sold_week2 : ℕ) : ℕ :=
  monthly_goal - (sold_week1 + sold_week2)

/-- Theorem: Given the specific values from the problem, 
    the additional pairs needed is 41 -/
theorem shoes_sales_goal :
  additional_pairs_needed 80 27 12 = 41 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sales_goal_l3113_311363


namespace NUMINAMATH_CALUDE_pencil_distribution_l3113_311343

theorem pencil_distribution (total_pencils : ℕ) (pencils_per_row : ℕ) (rows : ℕ) : 
  total_pencils = 12 → 
  pencils_per_row = 4 → 
  total_pencils = rows * pencils_per_row → 
  rows = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3113_311343


namespace NUMINAMATH_CALUDE_ad_length_is_sqrt_397_l3113_311381

/-- A quadrilateral with intersecting diagonals -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)
  (bo : dist B O = 5)
  (od : dist O D = 7)
  (ao : dist A O = 9)
  (oc : dist O C = 4)
  (ab : dist A B = 7)

/-- The length of AD in the quadrilateral -/
def ad_length (q : Quadrilateral) : ℝ := dist q.A q.D

/-- Theorem stating that AD length is √397 -/
theorem ad_length_is_sqrt_397 (q : Quadrilateral) : ad_length q = Real.sqrt 397 := by
  sorry


end NUMINAMATH_CALUDE_ad_length_is_sqrt_397_l3113_311381


namespace NUMINAMATH_CALUDE_largest_consecutive_nonprime_less_than_40_l3113_311320

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_consecutive_nonprime_less_than_40 
  (a b c d e : ℕ) 
  (h1 : a + 1 = b)
  (h2 : b + 1 = c)
  (h3 : c + 1 = d)
  (h4 : d + 1 = e)
  (h5 : 10 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 40)
  (h6 : ¬is_prime a ∧ ¬is_prime b ∧ ¬is_prime c ∧ ¬is_prime d ∧ ¬is_prime e) :
  e = 36 :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_nonprime_less_than_40_l3113_311320


namespace NUMINAMATH_CALUDE_fraction_power_equality_l3113_311385

theorem fraction_power_equality : (72000 ^ 5 : ℕ) / (9000 ^ 5) = 32768 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l3113_311385


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l3113_311308

theorem largest_angle_in_triangle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = (7/5) * 90 →  -- Sum of two angles is 7/5 of a right angle
  β = α + 40 →  -- One angle is 40° larger than the other
  max α (max β γ) = 83 :=  -- The largest angle is 83°
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l3113_311308
