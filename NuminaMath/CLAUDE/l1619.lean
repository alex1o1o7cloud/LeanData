import Mathlib

namespace NUMINAMATH_CALUDE_sin_2012_deg_l1619_161939

theorem sin_2012_deg : Real.sin (2012 * π / 180) = -Real.sin (32 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_2012_deg_l1619_161939


namespace NUMINAMATH_CALUDE_subtraction_digit_sum_l1619_161979

theorem subtraction_digit_sum : ∃ (a b : ℕ), 
  (a < 10) ∧ (b < 10) ∧ 
  (a * 10 + 9) - (1800 + b * 10 + 8) = 1 ∧
  a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_subtraction_digit_sum_l1619_161979


namespace NUMINAMATH_CALUDE_constant_sum_l1619_161908

/-- The number of distinct roots of a rational function -/
noncomputable def distinctRoots (num : ℝ → ℝ) (denom : ℝ → ℝ) : ℕ := sorry

theorem constant_sum (a b : ℝ) : 
  distinctRoots (λ x => (x+a)*(x+b)*(x+10)) (λ x => (x+4)^2) = 3 →
  distinctRoots (λ x => (x+2*a)*(x+4)*(x+5)) (λ x => (x+b)*(x+10)) = 1 →
  100*a + b = 205 := by sorry

end NUMINAMATH_CALUDE_constant_sum_l1619_161908


namespace NUMINAMATH_CALUDE_pauls_new_books_l1619_161986

theorem pauls_new_books (initial_books sold_books current_books : ℕ) : 
  initial_books = 2 → 
  sold_books = 94 → 
  current_books = 58 → 
  current_books = initial_books - sold_books + (sold_books - initial_books + current_books) :=
by
  sorry

end NUMINAMATH_CALUDE_pauls_new_books_l1619_161986


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1619_161906

theorem digit_sum_problem (X Y Z : ℕ) : 
  X < 10 → Y < 10 → Z < 10 →
  100 * X + 10 * Y + Z + 100 * X + 10 * Y + Z + 10 * Y + Z = 1675 →
  X + Y + Z = 15 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1619_161906


namespace NUMINAMATH_CALUDE_polygon_with_20_diagonals_is_octagon_l1619_161962

theorem polygon_with_20_diagonals_is_octagon :
  ∀ n : ℕ, n > 2 → (n * (n - 3)) / 2 = 20 → n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_with_20_diagonals_is_octagon_l1619_161962


namespace NUMINAMATH_CALUDE_count_triangle_points_l1619_161947

/-- The number of integer points in the triangle OAB -/
def triangle_points : ℕ :=
  (Finset.range 99).sum (fun k => 2 * k - 1)

/-- The theorem stating the number of integer points in the triangle OAB -/
theorem count_triangle_points :
  triangle_points = 9801 := by sorry

end NUMINAMATH_CALUDE_count_triangle_points_l1619_161947


namespace NUMINAMATH_CALUDE_part1_part2_l1619_161999

/-- Defines the sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℤ
| 0 => 3  -- We define a₀ = 3 to match a₁ = 3 in the original problem
| n + 1 => 2 * a n + n^2 - 4*n + 1

/-- The arithmetic sequence b_n -/
def b (n : ℕ) : ℤ := -2*n + 3

theorem part1 : ∀ n : ℕ, a n = 2^n - n^2 + 2*n := by sorry

theorem part2 (h : ∀ n : ℕ, a n = (n + 1) * b (n + 1) - n * b n) : 
  a 0 = 1 ∧ ∀ n : ℕ, b n = -2*n + 3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1619_161999


namespace NUMINAMATH_CALUDE_trig_identity_quadratic_equation_solution_l1619_161952

-- Part 1
theorem trig_identity : Real.cos (30 * π / 180) * Real.tan (60 * π / 180) - 2 * Real.sin (45 * π / 180) = 3/2 - Real.sqrt 2 := by
  sorry

-- Part 2
theorem quadratic_equation_solution (x : ℝ) : 3 * x^2 - 1 = -2 * x ↔ x = 1/3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_quadratic_equation_solution_l1619_161952


namespace NUMINAMATH_CALUDE_jonny_stairs_l1619_161980

theorem jonny_stairs :
  ∀ (j : ℕ),
  (j + (j / 3 - 7) = 1685) →
  j = 1521 := by
sorry

end NUMINAMATH_CALUDE_jonny_stairs_l1619_161980


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1619_161938

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 960) (h2 : Nat.gcd a c = 324) :
  ∃ (d : ℕ), d = Nat.gcd b c ∧ d = 12 ∧ ∀ (e : ℕ), e = Nat.gcd b c → e ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1619_161938


namespace NUMINAMATH_CALUDE_rick_sean_money_ratio_l1619_161945

theorem rick_sean_money_ratio :
  ∀ (fritz_money sean_money rick_money : ℝ),
    fritz_money = 40 →
    sean_money = fritz_money / 2 + 4 →
    rick_money + sean_money = 96 →
    rick_money / sean_money = 3 := by
  sorry

end NUMINAMATH_CALUDE_rick_sean_money_ratio_l1619_161945


namespace NUMINAMATH_CALUDE_colorNGon_correct_l1619_161921

/-- The number of ways to color exactly k vertices of an n-gon in red,
    such that no two consecutive vertices are red. -/
def colorNGon (n k : ℕ) : ℕ :=
  Nat.choose (n - k - 1) (k - 1) + Nat.choose (n - k) k

/-- Theorem stating that the number of ways to color exactly k vertices of an n-gon in red,
    such that no two consecutive vertices are red, is equal to ⁽ⁿ⁻ᵏ⁻¹ᵏ⁻¹⁾ + ⁽ⁿ⁻ᵏᵏ⁾. -/
theorem colorNGon_correct (n k : ℕ) (h1 : n > k) (h2 : k > 0) :
  colorNGon n k = Nat.choose (n - k - 1) (k - 1) + Nat.choose (n - k) k := by
  sorry

#eval colorNGon 5 2  -- Example usage

end NUMINAMATH_CALUDE_colorNGon_correct_l1619_161921


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1619_161992

theorem quadratic_factorization (x : ℝ) :
  -x^2 + 4*x - 4 = -(x - 2)^2 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1619_161992


namespace NUMINAMATH_CALUDE_gcd_90_450_l1619_161997

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_450_l1619_161997


namespace NUMINAMATH_CALUDE_competition_score_l1619_161916

theorem competition_score (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) 
  (total_score : ℤ) (score_difference : ℤ) :
  total_questions = 10 →
  correct_points = 5 →
  incorrect_points = -2 →
  total_score = 58 →
  score_difference = 14 →
  ∃ (a_correct : ℕ) (b_correct : ℕ),
    a_correct + b_correct ≤ total_questions ∧
    a_correct * correct_points + (total_questions - a_correct) * incorrect_points +
    b_correct * correct_points + (total_questions - b_correct) * incorrect_points = total_score ∧
    a_correct * correct_points + (total_questions - a_correct) * incorrect_points -
    (b_correct * correct_points + (total_questions - b_correct) * incorrect_points) = score_difference ∧
    a_correct = 8 :=
by sorry

end NUMINAMATH_CALUDE_competition_score_l1619_161916


namespace NUMINAMATH_CALUDE_keith_card_spending_l1619_161936

/-- The amount Keith spent on cards -/
def total_spent (digimon_packs : ℕ) (digimon_price : ℚ) (baseball_price : ℚ) : ℚ :=
  digimon_packs * digimon_price + baseball_price

/-- Proof that Keith spent $23.86 on cards -/
theorem keith_card_spending :
  total_spent 4 (445/100) (606/100) = 2386/100 := by
  sorry

end NUMINAMATH_CALUDE_keith_card_spending_l1619_161936


namespace NUMINAMATH_CALUDE_false_statements_exist_l1619_161915

theorem false_statements_exist : ∃ (a b c d : ℝ),
  (a > b ∧ c ≠ 0 ∧ a * c ≤ b * c) ∧
  (a > b ∧ b > 0 ∧ c > d ∧ a * c ≤ b * d) ∧
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) :=
by sorry

end NUMINAMATH_CALUDE_false_statements_exist_l1619_161915


namespace NUMINAMATH_CALUDE_find_M_l1619_161904

theorem find_M : ∃ M : ℚ, (5 + 7 + 10) / 3 = (2020 + 2021 + 2022) / M ∧ M = 827 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l1619_161904


namespace NUMINAMATH_CALUDE_min_stamps_for_50_cents_l1619_161985

/-- Represents the number of stamps of each denomination -/
structure StampCombination where
  threes : ℕ
  fours : ℕ
  sixes : ℕ

/-- Calculates the total value of stamps in cents -/
def totalValue (s : StampCombination) : ℕ :=
  3 * s.threes + 4 * s.fours + 6 * s.sixes

/-- Calculates the total number of stamps -/
def totalStamps (s : StampCombination) : ℕ :=
  s.threes + s.fours + s.sixes

/-- Checks if a stamp combination is valid (totals 50 cents) -/
def isValid (s : StampCombination) : Prop :=
  totalValue s = 50

/-- Theorem: The minimum number of stamps to make 50 cents is 10 -/
theorem min_stamps_for_50_cents :
  (∃ (s : StampCombination), isValid s ∧ totalStamps s = 10) ∧
  (∀ (s : StampCombination), isValid s → totalStamps s ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_min_stamps_for_50_cents_l1619_161985


namespace NUMINAMATH_CALUDE_triangle_area_range_line_equation_l1619_161927

/-- Ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 / 6 + y^2 / 4 = 1

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- Point on ellipse C₁ -/
def point_on_C₁ (P : ℝ × ℝ) : Prop := C₁ P.1 P.2

/-- Line passing through (-1, 0) -/
def line_through_M (l : ℝ → ℝ) : Prop := l 0 = -1

/-- Intersection points of line l with C₁ and C₂ -/
def intersection_points (l : ℝ → ℝ) (A B C D : ℝ × ℝ) : Prop :=
  point_on_C₁ A ∧ point_on_C₁ D ∧ C₂ B.1 B.2 ∧ C₂ C.1 C.2 ∧
  A.2 > B.2 ∧ B.2 > C.2 ∧ C.2 > D.2 ∧
  (∀ y, l y = A.1 ↔ y = A.2) ∧ (∀ y, l y = B.1 ↔ y = B.2) ∧
  (∀ y, l y = C.1 ↔ y = C.2) ∧ (∀ y, l y = D.1 ↔ y = D.2)

/-- Theorem 1: Range of triangle area -/
theorem triangle_area_range :
  ∀ P : ℝ × ℝ, point_on_C₁ P →
  ∃ S : ℝ, 1 ≤ S ∧ S ≤ Real.sqrt 2 ∧
  (∃ Q : ℝ × ℝ, C₂ Q.1 Q.2 ∧ S = (1/2) * Real.sqrt ((P.1^2 + P.2^2) * 2 - (P.1 * Q.1 + P.2 * Q.2)^2)) :=
sorry

/-- Theorem 2: Equation of line l -/
theorem line_equation :
  ∀ l : ℝ → ℝ, line_through_M l →
  (∃ A B C D : ℝ × ℝ, intersection_points l A B C D ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2) →
  (∀ y, l y = -1) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_range_line_equation_l1619_161927


namespace NUMINAMATH_CALUDE_three_x_plus_four_l1619_161912

theorem three_x_plus_four (x : ℝ) : x = 5 → 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_three_x_plus_four_l1619_161912


namespace NUMINAMATH_CALUDE_marbles_remaining_l1619_161993

theorem marbles_remaining (total_marbles : ℕ) (total_bags : ℕ) (bags_removed : ℕ) : 
  total_marbles = 28 →
  total_bags = 4 →
  bags_removed = 1 →
  total_marbles % total_bags = 0 →
  (total_bags - bags_removed) * (total_marbles / total_bags) = 21 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remaining_l1619_161993


namespace NUMINAMATH_CALUDE_brick_tower_heights_l1619_161917

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of distinct tower heights achievable -/
def distinctTowerHeights (numBricks : ℕ) (dimensions : BrickDimensions) : ℕ :=
  sorry

/-- Theorem stating the number of distinct tower heights for the given problem -/
theorem brick_tower_heights :
  let dimensions : BrickDimensions := ⟨3, 11, 17⟩
  distinctTowerHeights 62 dimensions = 435 := by
  sorry

end NUMINAMATH_CALUDE_brick_tower_heights_l1619_161917


namespace NUMINAMATH_CALUDE_average_of_three_l1619_161946

theorem average_of_three (q₁ q₂ q₃ q₄ q₅ : ℝ) : 
  (q₁ + q₂ + q₃ + q₄ + q₅) / 5 = 12 →
  (q₄ + q₅) / 2 = 24 →
  (q₁ + q₂ + q₃) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_of_three_l1619_161946


namespace NUMINAMATH_CALUDE_inverse_13_mod_1373_l1619_161900

theorem inverse_13_mod_1373 : ∃ x : ℕ, 0 ≤ x ∧ x < 1373 ∧ (13 * x) % 1373 = 1 := by
  use 843
  sorry

end NUMINAMATH_CALUDE_inverse_13_mod_1373_l1619_161900


namespace NUMINAMATH_CALUDE_triangle_properties_l1619_161956

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin t.B - t.b * Real.cos t.A = 0)
  (h2 : t.a = 2 * Real.sqrt 5)
  (h3 : t.b = 2) :
  t.A = π / 4 ∧ (1/2 * t.b * t.c * Real.sin t.A = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1619_161956


namespace NUMINAMATH_CALUDE_polynomial_equality_l1619_161924

theorem polynomial_equality (P : ℝ → ℝ) :
  (∀ m : ℝ, P m - (4 * m^3 + m^2 + 5) = 3 * m^4 - 4 * m^3 - m^2 + m - 8) →
  (∀ m : ℝ, P m = 3 * m^4 + m - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1619_161924


namespace NUMINAMATH_CALUDE_system_solutions_l1619_161941

def is_solution (x y z : ℝ) : Prop :=
  x^3 + y^3 + z^3 = 8 ∧
  x^2 + y^2 + z^2 = 22 ∧
  1/x + 1/y + 1/z + z/(x*y) = 0

theorem system_solutions :
  (is_solution 3 2 (-3)) ∧
  (is_solution (-3) 2 3) ∧
  (is_solution 2 3 (-3)) ∧
  (is_solution 2 (-3) 3) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1619_161941


namespace NUMINAMATH_CALUDE_five_digit_number_probability_l1619_161942

/-- The set of digits to choose from -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- The number of digits to select -/
def num_selected : Nat := 3

/-- The length of the number to form -/
def num_length : Nat := 5

/-- The number of digits that should be used twice -/
def num_twice_used : Nat := 2

/-- The probability of forming a number with two digits each used twice -/
def probability : Rat := 3/5

theorem five_digit_number_probability :
  (Finset.card digits = 5) →
  (num_selected = 3) →
  (num_length = 5) →
  (num_twice_used = 2) →
  (probability = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_number_probability_l1619_161942


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1619_161971

theorem hyperbola_sum (h k a b c : ℝ) : 
  (h = -3) →
  (k = 1) →
  (c = Real.sqrt 41) →
  (a = 4) →
  (c^2 = a^2 + b^2) →
  (h + k + a + b = 7) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1619_161971


namespace NUMINAMATH_CALUDE_M_equals_four_l1619_161911

theorem M_equals_four : 
  let M := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / 
           Real.sqrt (Real.sqrt 8 + 2) - 
           Real.sqrt (5 - 2 * Real.sqrt 6)
  M = 4 := by sorry

end NUMINAMATH_CALUDE_M_equals_four_l1619_161911


namespace NUMINAMATH_CALUDE_triangle_rotation_theorem_l1619_161960

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices O, P, and Q -/
structure Triangle where
  O : Point
  P : Point
  Q : Point

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point) : ℝ := sorry

/-- Rotates a point 90 degrees counterclockwise around the origin -/
def rotate90 (p : Point) : Point := 
  { x := -p.y, y := p.x }

theorem triangle_rotation_theorem (t : Triangle) : 
  t.O = ⟨0, 0⟩ → 
  t.Q = ⟨6, 0⟩ → 
  t.P.x > 0 → 
  t.P.y > 0 → 
  angle ⟨t.P.x - t.Q.x, t.P.y - t.Q.y⟩ ⟨t.O.x - t.Q.x, t.O.y - t.Q.y⟩ = π / 2 →
  angle ⟨t.P.x - t.O.x, t.P.y - t.O.y⟩ ⟨t.Q.x - t.O.x, t.Q.y - t.O.y⟩ = π / 4 →
  t.P = ⟨6, 6⟩ ∧ rotate90 t.P = ⟨-6, 6⟩ := by
sorry

end NUMINAMATH_CALUDE_triangle_rotation_theorem_l1619_161960


namespace NUMINAMATH_CALUDE_sum_of_abs_roots_of_P_l1619_161901

/-- The polynomial P(x) = x^3 - 6x^2 + 5x + 12 -/
def P (x : ℝ) : ℝ := x^3 - 6*x^2 + 5*x + 12

/-- Theorem: The sum of the absolute values of the roots of P(x) is 8 -/
theorem sum_of_abs_roots_of_P :
  ∃ (x₁ x₂ x₃ : ℝ),
    (P x₁ = 0) ∧ (P x₂ = 0) ∧ (P x₃ = 0) ∧
    (∀ x, P x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    |x₁| + |x₂| + |x₃| = 8 :=
sorry

end NUMINAMATH_CALUDE_sum_of_abs_roots_of_P_l1619_161901


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l1619_161968

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve xy = 2 -/
def onCurve (p : Point) : Prop := p.x * p.y = 2

/-- A circle in the 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The theorem statement -/
theorem fourth_intersection_point
  (c : Circle)
  (h1 : onCurve ⟨2, 1⟩ ∧ onCircle ⟨2, 1⟩ c)
  (h2 : onCurve ⟨-4, -1/2⟩ ∧ onCircle ⟨-4, -1/2⟩ c)
  (h3 : onCurve ⟨1/2, 4⟩ ∧ onCircle ⟨1/2, 4⟩ c)
  (h4 : ∃ (p : Point), onCurve p ∧ onCircle p c ∧ p ≠ ⟨2, 1⟩ ∧ p ≠ ⟨-4, -1/2⟩ ∧ p ≠ ⟨1/2, 4⟩) :
  ∃ (p : Point), p = ⟨-1, -2⟩ ∧ onCurve p ∧ onCircle p c :=
sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l1619_161968


namespace NUMINAMATH_CALUDE_new_rectangle_area_l1619_161996

/-- Given a rectangle with sides a and b, construct a new rectangle and calculate its area -/
theorem new_rectangle_area (a b : ℝ) (ha : a = 3) (hb : b = 4) : 
  let d := Real.sqrt (a^2 + b^2)
  let new_length := d + min a b
  let new_breadth := d - max a b
  new_length * new_breadth = 8 := by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l1619_161996


namespace NUMINAMATH_CALUDE_class_size_l1619_161988

theorem class_size (chorus : ℕ) (band : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : chorus = 18)
  (h2 : band = 26)
  (h3 : both = 2)
  (h4 : neither = 8) :
  chorus + band - both + neither = 50 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1619_161988


namespace NUMINAMATH_CALUDE_roses_problem_l1619_161926

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 21

/-- The number of roses Jessica cut from her garden -/
def cut_roses : ℕ := 28

/-- The number of roses Jessica threw away -/
def thrown_roses : ℕ := 34

/-- The number of roses currently in the vase -/
def current_roses : ℕ := 15

theorem roses_problem :
  initial_roses = 21 ∧
  thrown_roses = cut_roses + 6 ∧
  current_roses = initial_roses + cut_roses - thrown_roses :=
by sorry

end NUMINAMATH_CALUDE_roses_problem_l1619_161926


namespace NUMINAMATH_CALUDE_xy_gt_one_necessary_not_sufficient_l1619_161972

theorem xy_gt_one_necessary_not_sufficient :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x * y > 1) ∧
  (∃ x y : ℝ, x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_xy_gt_one_necessary_not_sufficient_l1619_161972


namespace NUMINAMATH_CALUDE_representation_of_2019_representation_of_any_integer_l1619_161983

theorem representation_of_2019 : ∃ (a b c : ℤ), 2019 = a^2 + b^2 - c^2 := by sorry

theorem representation_of_any_integer : ∀ (n : ℤ), ∃ (a b c d : ℤ), n = a^2 + b^2 + c^2 - d^2 := by sorry

end NUMINAMATH_CALUDE_representation_of_2019_representation_of_any_integer_l1619_161983


namespace NUMINAMATH_CALUDE_tangent_line_value_l1619_161955

/-- Proves that if a line is tangent to both y = ln x and x² = ay at the same point, then a = 2e -/
theorem tangent_line_value (a : ℝ) (h : a > 0) : 
  (∃ (x y : ℝ), x > 0 ∧ 
    y = Real.log x ∧ 
    x^2 = a * y ∧ 
    (1 / x) = (2 / a) * x) → 
  a = 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_value_l1619_161955


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1619_161961

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := e = 5/4

-- Define distance from focus to asymptote
def distance_focus_asymptote (d : ℝ) : Prop := d = 3

-- Theorem statement
theorem hyperbola_properties :
  ∃ (e d : ℝ), hyperbola x y ∧ eccentricity e ∧ distance_focus_asymptote d :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1619_161961


namespace NUMINAMATH_CALUDE_jordans_unripe_mangoes_l1619_161935

/-- Proves that Jordan kept 16 unripe mangoes given the conditions of the problem -/
theorem jordans_unripe_mangoes (total_mangoes : ℕ) (ripe_fraction : ℚ) (unripe_fraction : ℚ)
  (mangoes_per_jar : ℕ) (jars_made : ℕ) :
  total_mangoes = 54 →
  ripe_fraction = 1/3 →
  unripe_fraction = 2/3 →
  mangoes_per_jar = 4 →
  jars_made = 5 →
  (unripe_fraction * total_mangoes : ℚ).num - mangoes_per_jar * jars_made = 16 :=
by sorry

end NUMINAMATH_CALUDE_jordans_unripe_mangoes_l1619_161935


namespace NUMINAMATH_CALUDE_n_plus_one_in_terms_of_m_l1619_161965

theorem n_plus_one_in_terms_of_m (m n : ℕ) 
  (h1 : m * n = 121) 
  (h2 : (m + 1) * (n + 1) = 1000) : 
  n + 1 = 879 - m := by
sorry

end NUMINAMATH_CALUDE_n_plus_one_in_terms_of_m_l1619_161965


namespace NUMINAMATH_CALUDE_carol_meets_alice_l1619_161987

/-- Alice's speed in miles per hour -/
def alice_speed : ℝ := 4

/-- Carol's speed in miles per hour -/
def carol_speed : ℝ := 6

/-- Initial distance between Carol and Alice in miles -/
def initial_distance : ℝ := 5

/-- Time in minutes for Carol to meet Alice -/
def meeting_time : ℝ := 30

theorem carol_meets_alice : 
  initial_distance / (alice_speed + carol_speed) * 60 = meeting_time := by
  sorry

end NUMINAMATH_CALUDE_carol_meets_alice_l1619_161987


namespace NUMINAMATH_CALUDE_john_classes_l1619_161995

theorem john_classes (packs_per_student : ℕ) (students_per_class : ℕ) (total_packs : ℕ) 
  (h1 : packs_per_student = 2)
  (h2 : students_per_class = 30)
  (h3 : total_packs = 360) :
  total_packs / (packs_per_student * students_per_class) = 6 := by
  sorry

end NUMINAMATH_CALUDE_john_classes_l1619_161995


namespace NUMINAMATH_CALUDE_area_between_squares_l1619_161977

/-- The area of the region inside a large square but outside a smaller square -/
theorem area_between_squares (large_side : ℝ) (small_side : ℝ) 
  (h_large : large_side = 10)
  (h_small : small_side = 4)
  (h_placement : ∃ (x y : ℝ), x ^ 2 + y ^ 2 = (large_side / 2) ^ 2 ∧ 
                 0 ≤ x ∧ x ≤ small_side ∧ 0 ≤ y ∧ y ≤ small_side) :
  large_side ^ 2 - small_side ^ 2 = 84 := by
sorry

end NUMINAMATH_CALUDE_area_between_squares_l1619_161977


namespace NUMINAMATH_CALUDE_min_employees_needed_l1619_161981

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of working days for each employee per week -/
def working_days : ℕ := 5

/-- The number of rest days for each employee per week -/
def rest_days : ℕ := 2

/-- The minimum number of employees required on duty each day -/
def min_employees_per_day : ℕ := 45

/-- The minimum number of employees needed by the company -/
def min_total_employees : ℕ := 63

theorem min_employees_needed :
  ∀ (total_employees : ℕ),
    (∀ (day : Fin days_in_week),
      (total_employees * working_days) / days_in_week ≥ min_employees_per_day) →
    total_employees ≥ min_total_employees :=
by sorry

end NUMINAMATH_CALUDE_min_employees_needed_l1619_161981


namespace NUMINAMATH_CALUDE_sector_area_l1619_161959

/-- The area of a sector with radius 6 and central angle 60° is 6π. -/
theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = 60 * π / 180) :
  (θ / (2 * π)) * π * r^2 = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1619_161959


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1619_161951

/-- Two points P and Q in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the x-axis -/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The theorem to be proved -/
theorem symmetric_points_sum (a b : ℝ) :
  let p : Point := ⟨a, 1⟩
  let q : Point := ⟨2, b⟩
  symmetricAboutXAxis p q → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1619_161951


namespace NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l1619_161989

theorem gcd_lcm_product_30_75 : Nat.gcd 30 75 * Nat.lcm 30 75 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l1619_161989


namespace NUMINAMATH_CALUDE_equation_a_is_linear_l1619_161984

/-- Definition of a linear equation -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- The equation x = 1 -/
def equation_a (x : ℝ) : ℝ := x - 1

theorem equation_a_is_linear : is_linear_equation equation_a := by
  sorry

#check equation_a_is_linear

end NUMINAMATH_CALUDE_equation_a_is_linear_l1619_161984


namespace NUMINAMATH_CALUDE_digitCubeSequence_1729th_term_l1619_161930

/-- Sum of cubes of digits of a natural number -/
def sumCubesOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence defined by the sum of cubes of digits -/
def digitCubeSequence : ℕ → ℕ
  | 0 => 1729
  | n + 1 => sumCubesOfDigits (digitCubeSequence n)

/-- The 1729th term of the digit cube sequence is 370 -/
theorem digitCubeSequence_1729th_term :
  digitCubeSequence 1728 = 370 := by sorry

end NUMINAMATH_CALUDE_digitCubeSequence_1729th_term_l1619_161930


namespace NUMINAMATH_CALUDE_pascal_21st_number_23_row_l1619_161922

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of elements in the nth row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

theorem pascal_21st_number_23_row : 
  let row := 22
  let position := 21
  pascal_row_length row = 23 → binomial row (row + 1 - position) = 231 := by
  sorry

end NUMINAMATH_CALUDE_pascal_21st_number_23_row_l1619_161922


namespace NUMINAMATH_CALUDE_original_number_proof_l1619_161931

theorem original_number_proof : 
  ∃ N : ℕ, N ≥ 118 ∧ (N - 31) % 87 = 0 ∧ ∀ M : ℕ, M < N → (M - 31) % 87 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l1619_161931


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l1619_161994

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) (h1 : n ≥ 2) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a = k * b) (h5 : k > 0) :
  (n * (a - b)^(n-1) * (-b) + n * (n-1) / 2 * (a - b)^(n-2) * (-b)^2 = 0) → n = 2 * k + 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l1619_161994


namespace NUMINAMATH_CALUDE_A_equals_nine_l1619_161923

/-- A3 is a two-digit number -/
def A3 : ℕ := sorry

/-- A is the tens digit of A3 -/
def A : ℕ := A3 / 10

/-- The ones digit of A3 -/
def B : ℕ := A3 % 10

/-- A3 is a two-digit number -/
axiom A3_two_digit : 10 ≤ A3 ∧ A3 ≤ 99

/-- A3 - 41 = 52 -/
axiom A3_equation : A3 - 41 = 52

theorem A_equals_nine : A = 9 := by
  sorry

end NUMINAMATH_CALUDE_A_equals_nine_l1619_161923


namespace NUMINAMATH_CALUDE_jason_music_store_spending_l1619_161918

/-- The cost of Jason's flute -/
def flute_cost : ℚ := 142.46

/-- The cost of Jason's music tool -/
def music_tool_cost : ℚ := 8.89

/-- The cost of Jason's song book -/
def song_book_cost : ℚ := 7

/-- The total amount Jason spent at the music store -/
def total_spent : ℚ := flute_cost + music_tool_cost + song_book_cost

theorem jason_music_store_spending :
  total_spent = 158.35 := by sorry

end NUMINAMATH_CALUDE_jason_music_store_spending_l1619_161918


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l1619_161932

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  symmetry : ∀ x, a * x^2 + b * x + c = a * (2 - x)^2 + b * (2 - x) + c

/-- The theorem stating the relationship between f(2^x) and f(3^x) -/
theorem quadratic_function_inequality (f : QuadraticFunction) :
  ∀ x : ℝ, f.a * (3^x)^2 + f.b * (3^x) + f.c > f.a * (2^x)^2 + f.b * (2^x) + f.c :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l1619_161932


namespace NUMINAMATH_CALUDE_star_calculation_l1619_161914

-- Define the * operation
def star (a b : ℚ) : ℚ := (a + 2*b) / 3

-- State the theorem
theorem star_calculation : star (star 4 6) 9 = 70 / 9 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l1619_161914


namespace NUMINAMATH_CALUDE_saturday_exclamation_l1619_161910

/-- Represents the alien's exclamation as a string of 'A's and 'U's -/
def Exclamation := String

/-- Transforms a single character in the exclamation -/
def transformChar (c : Char) : Char :=
  match c with
  | 'A' => 'U'
  | 'U' => 'A'
  | _ => c

/-- Transforms the second half of the exclamation -/
def transformSecondHalf (s : String) : String :=
  s.map transformChar

/-- Generates the next day's exclamation based on the current day -/
def nextDayExclamation (current : Exclamation) : Exclamation :=
  let n := current.length
  let firstHalf := current.take (n / 2)
  let secondHalf := current.drop (n / 2)
  firstHalf ++ transformSecondHalf secondHalf

/-- Generates the nth day's exclamation -/
def nthDayExclamation (n : Nat) : Exclamation :=
  match n with
  | 0 => "A"
  | n + 1 => nextDayExclamation (nthDayExclamation n)

theorem saturday_exclamation :
  nthDayExclamation 5 = "АУУАУААУУААУАУААУУААУААААУУААУАА" :=
by sorry

end NUMINAMATH_CALUDE_saturday_exclamation_l1619_161910


namespace NUMINAMATH_CALUDE_muffin_buyers_count_l1619_161950

-- Define the total number of buyers
def total_buyers : ℕ := 100

-- Define the number of buyers who purchase cake mix
def cake_buyers : ℕ := 50

-- Define the number of buyers who purchase both cake and muffin mix
def both_buyers : ℕ := 19

-- Define the probability of selecting a buyer who purchases neither cake nor muffin mix
def prob_neither : ℚ := 29/100

-- Theorem to prove
theorem muffin_buyers_count : 
  ∃ (muffin_buyers : ℕ), 
    muffin_buyers = total_buyers - cake_buyers - (total_buyers * prob_neither).num + both_buyers := by
  sorry

end NUMINAMATH_CALUDE_muffin_buyers_count_l1619_161950


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1619_161974

theorem complex_magnitude_product : Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * (2 * Real.sqrt 3 + 6 * Complex.I)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1619_161974


namespace NUMINAMATH_CALUDE_rectangle_to_square_l1619_161934

theorem rectangle_to_square (l w : ℝ) : 
  (2 * (l + w) = 40) →  -- Perimeter of rectangle is 40cm
  (l - 8 = w + 2) →     -- Rectangle becomes square after changes
  (l - 8 = 7) :=        -- Side length of resulting square is 7cm
by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l1619_161934


namespace NUMINAMATH_CALUDE_solution_difference_l1619_161969

-- Define the equation
def equation (r : ℝ) : Prop :=
  (r^2 - 6*r - 20) / (r + 3) = 3*r + 10

-- Define the solutions
def solutions : Set ℝ :=
  {r : ℝ | equation r ∧ r ≠ -3}

-- Theorem statement
theorem solution_difference :
  ∃ (r₁ r₂ : ℝ), r₁ ∈ solutions ∧ r₂ ∈ solutions ∧ r₁ ≠ r₂ ∧ |r₁ - r₂| = 20 :=
by sorry

end NUMINAMATH_CALUDE_solution_difference_l1619_161969


namespace NUMINAMATH_CALUDE_Q_value_l1619_161902

-- Define the relationship between P, Q, and U
def varies_directly_inversely (P Q U : ℚ) : Prop :=
  ∃ k : ℚ, P = k * Q / U

-- Define the initial conditions
def initial_conditions (P Q U : ℚ) : Prop :=
  P = 12 ∧ Q = 1/2 ∧ U = 16/25

-- Define the final conditions
def final_conditions (P U : ℚ) : Prop :=
  P = 27 ∧ U = 9/49

-- Theorem statement
theorem Q_value :
  ∀ P Q U : ℚ,
  varies_directly_inversely P Q U →
  initial_conditions P Q U →
  final_conditions P U →
  Q = 225/696 :=
by sorry

end NUMINAMATH_CALUDE_Q_value_l1619_161902


namespace NUMINAMATH_CALUDE_rectangle_area_with_equal_perimeter_to_triangle_l1619_161998

/-- The area of a rectangle with equal perimeter to a specific triangle -/
theorem rectangle_area_with_equal_perimeter_to_triangle : 
  ∀ (rectangle_side1 rectangle_side2 : ℝ),
  rectangle_side1 = 12 →
  2 * (rectangle_side1 + rectangle_side2) = 10 + 12 + 15 →
  rectangle_side1 * rectangle_side2 = 78 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_with_equal_perimeter_to_triangle_l1619_161998


namespace NUMINAMATH_CALUDE_new_female_percentage_new_female_percentage_proof_l1619_161948

theorem new_female_percentage (initial_female_percentage : ℝ) 
                               (additional_male_hires : ℕ) 
                               (total_employees_after : ℕ) : ℝ :=
  let initial_employees := total_employees_after - additional_male_hires
  let initial_female_employees := (initial_female_percentage / 100) * initial_employees
  (initial_female_employees / total_employees_after) * 100

#check 
  @new_female_percentage 60 20 240 = 55

theorem new_female_percentage_proof :
  new_female_percentage 60 20 240 = 55 := by
  sorry

end NUMINAMATH_CALUDE_new_female_percentage_new_female_percentage_proof_l1619_161948


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1619_161958

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, 2 * x^2 - 3 * x + (3/2) ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1619_161958


namespace NUMINAMATH_CALUDE_charlies_data_usage_l1619_161907

/-- Represents Charlie's cell phone data usage problem -/
theorem charlies_data_usage 
  (data_limit : ℝ) 
  (extra_cost_per_gb : ℝ)
  (week1_usage : ℝ)
  (week2_usage : ℝ)
  (week3_usage : ℝ)
  (extra_charge : ℝ)
  (h1 : data_limit = 8)
  (h2 : extra_cost_per_gb = 10)
  (h3 : week1_usage = 2)
  (h4 : week2_usage = 3)
  (h5 : week3_usage = 5)
  (h6 : extra_charge = 120)
  : ∃ (week4_usage : ℝ), 
    week4_usage = 10 ∧ 
    (week1_usage + week2_usage + week3_usage + week4_usage - data_limit) * extra_cost_per_gb = extra_charge :=
sorry

end NUMINAMATH_CALUDE_charlies_data_usage_l1619_161907


namespace NUMINAMATH_CALUDE_hockey_cards_count_l1619_161903

theorem hockey_cards_count (hockey : ℕ) (football : ℕ) (baseball : ℕ) : 
  baseball = football - 50 →
  football = 4 * hockey →
  hockey + football + baseball = 1750 →
  hockey = 200 := by
sorry

end NUMINAMATH_CALUDE_hockey_cards_count_l1619_161903


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l1619_161991

theorem unique_two_digit_integer (u : ℕ) : 
  (u ≥ 10 ∧ u ≤ 99) ∧ (17 * u) % 100 = 45 ↔ u = 85 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l1619_161991


namespace NUMINAMATH_CALUDE_pentagon_count_l1619_161943

/-- The number of distinct points on the circumference of a circle -/
def n : ℕ := 15

/-- The number of vertices in each polygon -/
def k : ℕ := 5

/-- The number of distinct convex pentagons that can be formed -/
def num_pentagons : ℕ := Nat.choose n k

theorem pentagon_count :
  num_pentagons = 3003 :=
sorry

end NUMINAMATH_CALUDE_pentagon_count_l1619_161943


namespace NUMINAMATH_CALUDE_parallel_condition_l1619_161975

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the parallel relation for planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation for a line and a plane
variable (line_plane_parallel : Line → Plane → Prop)

-- State the theorem
theorem parallel_condition 
  (α β : Plane) (a : Line) 
  (h_subset : subset a α) : 
  (∀ α β a, plane_parallel α β → line_plane_parallel a β) ∧ 
  (∃ α β a, line_plane_parallel a β ∧ ¬plane_parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l1619_161975


namespace NUMINAMATH_CALUDE_cube_opposite_faces_l1619_161967

/-- Represents a face of a cube --/
inductive Face : Type
| G | H | I | J | S | K

/-- Represents the adjacency relation between faces --/
def adjacent : Face → Face → Prop := sorry

/-- Represents the opposite relation between faces --/
def opposite : Face → Face → Prop := sorry

/-- Theorem: If H and I are adjacent, G is adjacent to both H and I, 
    and J is adjacent to H and I, then J is opposite to G --/
theorem cube_opposite_faces 
  (adj_H_I : adjacent Face.H Face.I)
  (adj_G_H : adjacent Face.G Face.H)
  (adj_G_I : adjacent Face.G Face.I)
  (adj_J_H : adjacent Face.J Face.H)
  (adj_J_I : adjacent Face.J Face.I) :
  opposite Face.G Face.J := by sorry

end NUMINAMATH_CALUDE_cube_opposite_faces_l1619_161967


namespace NUMINAMATH_CALUDE_function_value_at_m_l1619_161909

/-- Given a function f(x) = x³ + ax + 3 where f(-m) = 1, prove that f(m) = 5 -/
theorem function_value_at_m (a m : ℝ) : 
  (fun x : ℝ ↦ x^3 + a*x + 3) (-m) = 1 → 
  (fun x : ℝ ↦ x^3 + a*x + 3) m = 5 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_m_l1619_161909


namespace NUMINAMATH_CALUDE_bus_seating_capacity_bus_total_capacity_l1619_161919

/-- The number of people that can sit in a bus given the seating arrangement --/
theorem bus_seating_capacity 
  (left_seats : ℕ) 
  (right_seats_difference : ℕ) 
  (people_per_seat : ℕ) 
  (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seats_difference
  let left_capacity := left_seats * people_per_seat
  let right_capacity := right_seats * people_per_seat
  left_capacity + right_capacity + back_seat_capacity

/-- The total number of people that can sit in the bus is 90 --/
theorem bus_total_capacity : 
  bus_seating_capacity 15 3 3 9 = 90 := by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_bus_total_capacity_l1619_161919


namespace NUMINAMATH_CALUDE_product_of_differences_divisible_by_12_l1619_161963

theorem product_of_differences_divisible_by_12 (a b c d : ℤ) :
  12 ∣ ((a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_divisible_by_12_l1619_161963


namespace NUMINAMATH_CALUDE_painting_ratio_l1619_161982

theorem painting_ratio (monday : ℝ) (total : ℝ) : 
  monday = 30 →
  total = 105 →
  (total - (monday + 2 * monday)) / monday = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_painting_ratio_l1619_161982


namespace NUMINAMATH_CALUDE_least_abaaba_six_primes_l1619_161925

def is_abaaba_form (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧
  n = a * 100000 + b * 10000 + a * 1000 + a * 100 + b * 10 + a

def is_product_of_six_distinct_primes (n : ℕ) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : ℕ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ Prime p₆ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧
    p₅ ≠ p₆ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅ * p₆

theorem least_abaaba_six_primes :
  (is_abaaba_form 282282 ∧ is_product_of_six_distinct_primes 282282) ∧
  (∀ n : ℕ, n < 282282 → ¬(is_abaaba_form n ∧ is_product_of_six_distinct_primes n)) :=
by sorry

end NUMINAMATH_CALUDE_least_abaaba_six_primes_l1619_161925


namespace NUMINAMATH_CALUDE_triangle_angle_sin_sum_bounds_triangle_angle_sin_sum_equality_condition_l1619_161913

open Real

/-- Triangle interior angles in radians -/
structure TriangleAngles where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_eq_pi : A + B + C = π
  all_positive : 0 < A ∧ 0 < B ∧ 0 < C

theorem triangle_angle_sin_sum_bounds (t : TriangleAngles) :
  -2 < sin (3 * t.A) + sin (3 * t.B) + sin (3 * t.C) ∧
  sin (3 * t.A) + sin (3 * t.B) + sin (3 * t.C) ≤ 3/2 * Real.sqrt 3 :=
sorry

theorem triangle_angle_sin_sum_equality_condition (t : TriangleAngles) :
  sin (3 * t.A) + sin (3 * t.B) + sin (3 * t.C) = 3/2 * Real.sqrt 3 ↔
  t.A = 7*π/18 ∧ t.B = π/9 ∧ t.C = π/9 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_sin_sum_bounds_triangle_angle_sin_sum_equality_condition_l1619_161913


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1619_161929

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  (f 0 = 0) ∧ (f 2 = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 0 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1619_161929


namespace NUMINAMATH_CALUDE_function_inequality_l1619_161966

theorem function_inequality (a x : ℝ) : 
  let f := fun (t : ℝ) => t^2 - t + 13
  |x - a| < 1 → |f x - f a| < 2 * (|a| + 1) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1619_161966


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l1619_161976

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 7, where a, b, and c are constants,
    if f(-2011) = -17, then f(2011) = 31. -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 7
  f (-2011) = -17 → f 2011 = 31 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l1619_161976


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1619_161940

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | Real.log (x - 2) < 1}

def B : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 2 < x ∧ x < 12} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1619_161940


namespace NUMINAMATH_CALUDE_money_problem_l1619_161990

theorem money_problem (a b : ℝ) 
  (eq_condition : 6 * a + b = 66)
  (ineq_condition : 4 * a + b < 48) :
  a > 9 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_money_problem_l1619_161990


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l1619_161928

/-- The area of a stripe on a cylindrical tank -/
theorem stripe_area_on_cylinder (d h w r : ℝ) (h1 : d = 20) (h2 : w = 4) (h3 : r = d / 2) :
  3 * (2 * π * r) * w = 240 * π := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l1619_161928


namespace NUMINAMATH_CALUDE_min_value_f_min_value_expression_l1619_161978

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, f x ≥ m :=
sorry

-- Theorem for the minimum value of 1/(a²+1) + 4/(b²+1)
theorem min_value_expression (a b : ℝ) (h : a^2 + b^2 = 2) :
  ∃ min_val : ℝ, min_val = 9/4 ∧
  1 / (a^2 + 1) + 4 / (b^2 + 1) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_f_min_value_expression_l1619_161978


namespace NUMINAMATH_CALUDE_sum_of_specific_S_values_l1619_161954

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem sum_of_specific_S_values : S 17 + S 33 + S 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_values_l1619_161954


namespace NUMINAMATH_CALUDE_regular_pentagon_angle_l1619_161957

theorem regular_pentagon_angle (n : ℕ) (h : n = 5) :
  let central_angle := 360 / n
  2 * central_angle = 144 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_angle_l1619_161957


namespace NUMINAMATH_CALUDE_license_plate_count_l1619_161937

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of alphanumeric characters (letters + digits) -/
def num_alphanumeric : ℕ := num_letters + num_digits

/-- The number of different license plates that can be formed -/
def num_license_plates : ℕ := num_letters * num_digits * num_alphanumeric

theorem license_plate_count : num_license_plates = 9360 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1619_161937


namespace NUMINAMATH_CALUDE_roses_in_vase_l1619_161933

theorem roses_in_vase (initial_roses : ℕ) (initial_orchids : ℕ) (final_orchids : ℕ) (cut_orchids : ℕ) :
  initial_roses = 16 →
  initial_orchids = 3 →
  final_orchids = 7 →
  cut_orchids = 4 →
  ∃ (cut_roses : ℕ), cut_roses = cut_orchids →
  initial_roses + cut_roses = 24 :=
by sorry

end NUMINAMATH_CALUDE_roses_in_vase_l1619_161933


namespace NUMINAMATH_CALUDE_farm_land_allocation_l1619_161973

theorem farm_land_allocation (total_land : ℕ) (reserved : ℕ) (cattle : ℕ) (crops : ℕ) 
  (h1 : total_land = 150)
  (h2 : reserved = 15)
  (h3 : cattle = 40)
  (h4 : crops = 70) :
  total_land - reserved - cattle - crops = 25 := by
  sorry

end NUMINAMATH_CALUDE_farm_land_allocation_l1619_161973


namespace NUMINAMATH_CALUDE_sasha_questions_per_hour_l1619_161970

theorem sasha_questions_per_hour 
  (initial_questions : ℕ)
  (work_hours : ℕ)
  (remaining_questions : ℕ)
  (h1 : initial_questions = 60)
  (h2 : work_hours = 2)
  (h3 : remaining_questions = 30) :
  (initial_questions - remaining_questions) / work_hours = 15 := by
sorry

end NUMINAMATH_CALUDE_sasha_questions_per_hour_l1619_161970


namespace NUMINAMATH_CALUDE_percentage_problem_l1619_161905

theorem percentage_problem (p : ℝ) : 
  (25 / 100 * 840 = p / 100 * 1500 - 15) → p = 15 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1619_161905


namespace NUMINAMATH_CALUDE_min_value_of_arithmetic_geometric_seq_l1619_161920

/-- A positive arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ q : ℝ, q > 0 ∧ ∀ k, a (k + 1) = a k * q

theorem min_value_of_arithmetic_geometric_seq
  (a : ℕ → ℝ)
  (h_seq : ArithGeomSeq a)
  (h_cond : a 7 = a 6 + 2 * a 5)
  (h_exists : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) :
  (∃ m n : ℕ, 1 / m + 4 / n = 3 / 2) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_arithmetic_geometric_seq_l1619_161920


namespace NUMINAMATH_CALUDE_min_teams_in_championship_l1619_161949

/-- Represents a soccer championship with the given rules --/
structure SoccerChampionship where
  numTeams : ℕ
  /-- Each team plays one match against every other team --/
  totalMatches : ℕ := numTeams * (numTeams - 1) / 2
  /-- Winning team gets 2 points, tie gives 1 point to each team, losing team gets 0 points --/
  pointSystem : List ℕ := [2, 1, 0]

/-- Represents the points of a team --/
structure TeamPoints where
  wins : ℕ
  draws : ℕ
  points : ℕ := 2 * wins + draws

/-- The condition that one team has the most points but fewer wins than any other team --/
def hasUniqueLeader (c : SoccerChampionship) (leader : TeamPoints) (others : List TeamPoints) : Prop :=
  ∀ team ∈ others, leader.points > team.points ∧ leader.wins < team.wins

/-- The main theorem stating the minimum number of teams --/
theorem min_teams_in_championship : 
  ∀ c : SoccerChampionship, 
  ∀ leader : TeamPoints,
  ∀ others : List TeamPoints,
  hasUniqueLeader c leader others →
  c.numTeams ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_teams_in_championship_l1619_161949


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1619_161953

theorem quadratic_discriminant (a b c : ℝ) (h : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = x - 2) ∧
  (∃! x, a * x^2 + b * x + c = 1 - x / 2) →
  b^2 - 4*a*c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1619_161953


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1619_161964

/-- Given a geometric sequence {aₙ} with a₁ = 1 and common ratio q ≠ 1,
    if -3a₁, -a₂, and a₃ form an arithmetic sequence,
    then the sum of the first 4 terms (S₄) equals -20. -/
theorem geometric_sequence_sum (q : ℝ) (h1 : q ≠ 1) : 
  let a : ℕ → ℝ := λ n => q^(n-1)
  ∀ n, a n = q^(n-1)
  → -3 * (a 1) + (a 3) = 2 * (-a 2)
  → (a 1) = 1
  → (a 1) + (a 2) + (a 3) + (a 4) = -20 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1619_161964


namespace NUMINAMATH_CALUDE_noah_sales_revenue_l1619_161944

-- Define constants
def large_painting_price : ℝ := 60
def small_painting_price : ℝ := 30
def last_month_large_sales : ℕ := 8
def last_month_small_sales : ℕ := 4
def large_painting_discount : ℝ := 0.1
def small_painting_commission : ℝ := 0.05
def sales_tax_rate : ℝ := 0.07

-- Define the theorem
theorem noah_sales_revenue :
  let this_month_large_sales := 2 * last_month_large_sales
  let this_month_small_sales := 2 * last_month_small_sales
  let discounted_large_price := large_painting_price * (1 - large_painting_discount)
  let commissioned_small_price := small_painting_price * (1 - small_painting_commission)
  let total_sales_before_tax := 
    this_month_large_sales * discounted_large_price +
    this_month_small_sales * commissioned_small_price
  let sales_tax := total_sales_before_tax * sales_tax_rate
  let total_sales_revenue := total_sales_before_tax + sales_tax
  total_sales_revenue = 1168.44 := by
  sorry

end NUMINAMATH_CALUDE_noah_sales_revenue_l1619_161944
