import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l315_31529

theorem expression_simplification (x : ℝ) :
  x = Real.sqrt 12 + (Real.sqrt 5)^(0 : ℝ) - (1/2)^(-1 : ℝ) →
  (1 / (x + 1) + 1 / (x^2 - 1)) / (x / (x - 1)) = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l315_31529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f1_solution_set_f2_g2_relation_l315_31532

-- Part 1
def f1 (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem f1_solution_set : 
  {x : ℝ | f1 x = 4} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/2} := by sorry

-- Part 2
def f2 (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1|
def g2 (x : ℝ) : ℝ := |x - 2| + 1

theorem f2_g2_relation (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, g2 x₂ = f2 a x₁) →
  a ∈ (Set.Iic (-2) ∪ Set.Ici 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f1_solution_set_f2_g2_relation_l315_31532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_maximum_l315_31534

theorem triangle_area_maximum (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  b = 2 →
  Real.tan C = (Real.sqrt 3 * Real.sin B) / (1 - Real.sqrt 3 * Real.cos B) →
  S = Real.sqrt ((1/4) * (a^2 * c^2 - ((a^2 + c^2 - b^2)/2)^2)) →
  ∃ (S_max : ℝ), S ≤ S_max ∧ S_max = Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_maximum_l315_31534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_properties_l315_31503

-- Define the function f
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Define the theorem
theorem function_and_triangle_properties 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : abs φ < Real.pi / 2) 
  (h4 : ∀ x, f A ω φ x ≤ 2)
  (h5 : ∀ x, f A ω φ (x + Real.pi / (2 * ω)) = f A ω φ x)
  (h6 : f A ω φ (Real.pi / 24) = 0)
  (h7 : ∃ C, f A ω φ (C / 4) = 2)
  (h8 : ∃ a b c A B C, 
    c = Real.sqrt 3 / 2 ∧ 
    a * Real.sin C = b * Real.sin A ∧ 
    b * Real.sin C = c * Real.sin B ∧
    A + B + C = Real.pi) :
  (∀ x, f A ω φ x = 2 * Real.sin (4 * x - Real.pi / 6)) ∧
  (∃ a b, Real.sqrt 3 / 2 < a + 2 * b ∧ a + 2 * b < Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_properties_l315_31503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cord_sufficient_l315_31563

/-- The setup for the pulley system -/
structure PulleySystem where
  r : ℝ  -- radius of each pulley
  d12 : ℝ  -- distance between O₁ and O₂
  d13 : ℝ  -- distance between O₁ and O₃
  h : ℝ  -- perpendicular distance from O₃ to plane O₁O₂

/-- Calculate the length of the belt required for the pulley system -/
noncomputable def beltLength (p : PulleySystem) : ℝ :=
  let d23 := Real.sqrt (p.d12^2 + p.d13^2 - 2 * p.d12 * p.d13 * (p.h / p.d13))
  p.d12 + p.d13 + d23 + 2 * Real.pi * p.r

/-- Theorem stating that a 54 cm cord is sufficient for the given pulley system -/
theorem cord_sufficient (p : PulleySystem) 
    (h1 : p.r = 2)
    (h2 : p.d12 = 12)
    (h3 : p.d13 = 10)
    (h4 : p.h = 8) : 
  beltLength p < 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cord_sufficient_l315_31563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l315_31551

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := 1
noncomputable def line2 (x : ℝ) : ℝ := 1 + (Real.sqrt 3 / 3) * x

-- Define the equilateral triangle
structure EquilateralTriangle where
  -- The third line passing through the origin
  slope : ℝ
  -- Condition that it forms an equilateral triangle
  is_equilateral : ∃ (x y : ℝ), y = slope * x ∧ 
                   (x - 1)^2 + (y - line1 x)^2 = 
                   (x - 1)^2 + (y - line2 x)^2 ∧
                   x^2 + y^2 = 
                   (x - 1)^2 + (y - line2 x)^2

-- Theorem statement
theorem equilateral_triangle_perimeter 
  (triangle : EquilateralTriangle) : 
  ∃ (a b c : ℝ), 
    (a - 1)^2 + (b - line1 a)^2 +
    (a - 1)^2 + (b - line2 a)^2 +
    a^2 + b^2 = (3 + 2 * Real.sqrt 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l315_31551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_half_l315_31584

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / ((2*x - 1) * (x + a))

-- Define what it means for f to be an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_implies_a_eq_half (a : ℝ) : 
  is_odd_function (f a) → a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_half_l315_31584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l315_31569

noncomputable def f (x : ℝ) := (Real.sqrt (2 - x)) / (x - 1)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x | x < 1 ∨ (1 < x ∧ x ≤ 2)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l315_31569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l315_31583

noncomputable def f (x : ℝ) : ℝ := Real.sin (4 * x - Real.pi / 2)

theorem phase_shift_of_f :
  ∃ (φ : ℝ), φ = Real.pi / 8 ∧
  ∀ (x : ℝ), f (x + φ) = Real.sin (4 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l315_31583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_exists_l315_31507

/-- A natural number is prime if it's greater than 1 and its only positive divisors are 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A natural number is a perfect square if it's equal to some integer multiplied by itself. -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The roots of the quadratic equation x^2 - 40x + k = 0 -/
def roots (k : ℕ) : Set ℝ := {x : ℝ | x^2 - 40*x + k = 0}

/-- The theorem stating that there is exactly one value of k satisfying the conditions -/
theorem unique_k_exists : ∃! k : ℕ, 
  ∃ (x y : ℝ), x ∈ roots k ∧ y ∈ roots k ∧ x ≠ y ∧ 
  isPrime (Int.toNat ⌊x⌋) ∧ isPerfectSquare (Int.toNat ⌊y⌋) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_exists_l315_31507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_value_l315_31531

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x * (a * x + b) - x^2 - 4 * x

noncomputable def f' (a b x : ℝ) : ℝ := Real.exp x * (a * x + a + b) - 2 * x - 4

theorem tangent_line_and_max_value (a b : ℝ) :
  (f' a b 0 = 4 ∧ f a b 0 = 4) →
  (a = 4 ∧ b = 4) ∧
  (∀ x : ℝ, f' 4 4 x = 4 * (x + 2) * (Real.exp x - 1/2)) ∧
  (∀ x : ℝ, x < -2 ∨ (-Real.log 2 < x) → f' 4 4 x > 0) ∧
  (∀ x : ℝ, -2 < x ∧ x < -Real.log 2 → f' 4 4 x < 0) ∧
  (f 4 4 (-2) = 4 * (1 - Real.exp (-2))) ∧
  (∀ x : ℝ, f 4 4 x ≤ f 4 4 (-2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_value_l315_31531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_value_l315_31538

-- Define the functions f and g on ℝ
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the range conditions for f and g
axiom f_range : ∀ x, -4 ≤ f x ∧ f x ≤ 3
axiom g_range : ∀ x, -3 ≤ g x ∧ g x ≤ 2

-- Theorem statement
theorem max_product_value :
  ∃ b, ∀ x, f x * g x ≤ b ∧ (∃ y, f y * g y = b) ∧ b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_value_l315_31538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_zero_set_equality_l315_31530

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.cos x

-- Define sets M and N
def M : Set ℝ := {x | f x ≠ 0}
def N : Set ℝ := {x | g x ≠ 0}

-- State the theorem
theorem sine_cosine_zero_set_equality :
  {x : ℝ | f x * g x = 0} = (U \ M) ∪ (U \ N) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_zero_set_equality_l315_31530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_coprime_no_solution_l315_31506

theorem smallest_sum_coprime_no_solution : 
  ∃ (a b : ℕ) (p : ℕ) (c : ℤ), 
    Nat.Coprime a b ∧ 
    Nat.Prime p ∧
    (∀ (x y : ℤ), (x^a + y^b) % p ≠ c % p) ∧
    a + b = 7 ∧
    (∀ (a' b' : ℕ) (p' : ℕ) (c' : ℤ), 
      Nat.Coprime a' b' →
      Nat.Prime p' →
      (∀ (x y : ℤ), (x^a' + y^b') % p' ≠ c' % p') →
      a' + b' ≥ 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_coprime_no_solution_l315_31506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_big_integers_sum_l315_31514

open BigOperators

/-- A "big" integer in a permutation is one that is greater than all elements that come after it. -/
def is_big {n : ℕ} (perm : Fin n → Fin n) (i : Fin n) : Prop :=
  ∀ j, i < j → perm i > perm j

/-- The expected number of "big" integers in a random permutation of {1, 2, ..., n} -/
noncomputable def expected_big_integers (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (1 : ℝ) / (k + 1)

/-- Theorem: The expected number of "big" integers in a random permutation of {1, 2, ..., n}
    is equal to the sum of reciprocals from 1 to n -/
theorem expected_big_integers_sum (n : ℕ) :
  expected_big_integers n = ∑ k in Finset.range n, (1 : ℝ) / (k + 1) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_big_integers_sum_l315_31514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l315_31523

noncomputable section

-- Define the curve C
def curve_C (α : Real) : Real × Real :=
  (Real.sqrt 6 * Real.cos α, Real.sqrt 2 * Real.sin α)

-- Define the line l
def line_l (x y : Real) : Prop :=
  x + Real.sqrt 3 * y + 4 = 0

-- Define the distance function from a point to the line
def distance_to_line (x y : Real) : Real :=
  abs (x + Real.sqrt 3 * y + 4) / 2

-- Theorem statement
theorem max_distance_curve_to_line :
  ∃ (max_dist : Real), max_dist = 2 + Real.sqrt 3 ∧
  ∀ (α : Real), 
    let (x, y) := curve_C α
    distance_to_line x y ≤ max_dist ∧
    ∃ (α₀ : Real), distance_to_line (curve_C α₀).1 (curve_C α₀).2 = max_dist :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l315_31523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_min_distance_l315_31547

noncomputable def area_parallelogram (a b c d : ℂ) : ℝ := 
  abs ((b - a).im * (c - a).re - (b - a).re * (c - a).im)

theorem parallelogram_area_min_distance (z : ℂ) :
  (z.re > 0) →
  (area_parallelogram 0 z (z⁻¹) (z + z⁻¹) = 12 / 13) →
  (∀ w : ℂ, w.re > 0 ∧ area_parallelogram 0 w (w⁻¹) (w + w⁻¹) = 12 / 13 → 
    Complex.normSq (z + z⁻¹) ≤ Complex.normSq (w + w⁻¹)) →
  Complex.normSq (z + z⁻¹) = 36 / 13 :=
by
  sorry

#check parallelogram_area_min_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_min_distance_l315_31547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_per_bottle_l315_31521

/-- Calculates the average price per bottle given the number of large and small bottles,
    their respective prices, discounts, and sales tax rate. -/
theorem average_price_per_bottle
  (large_bottles : ℕ)
  (small_bottles : ℕ)
  (large_price : ℚ)
  (small_price : ℚ)
  (large_discount : ℚ)
  (small_discount : ℚ)
  (sales_tax : ℚ)
  (h_large_bottles : large_bottles = 1375)
  (h_small_bottles : small_bottles = 690)
  (h_large_price : large_price = 175/100)
  (h_small_price : small_price = 135/100)
  (h_large_discount : large_discount = 12/100)
  (h_small_discount : small_discount = 9/100)
  (h_sales_tax : sales_tax = 8/100) :
  abs ((((large_bottles * large_price + small_bottles * small_price) * 
        (1 - large_discount * (large_bottles * large_price) / (large_bottles * large_price + small_bottles * small_price) - 
         small_discount * (small_bottles * small_price) / (large_bottles * large_price + small_bottles * small_price))) * 
        (1 + sales_tax)) / (large_bottles + small_bottles) - 155/100) < 1/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_per_bottle_l315_31521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l315_31544

noncomputable def f (x : ℝ) : ℝ := Real.exp x + (2*x - 5) / (x^2 + 1)

theorem tangent_perpendicular_line (m : ℝ) :
  (∃ (slope : ℝ), (slope = (deriv f) 0) ∧ 
   (slope * (1/m) = -1) ∧ 
   (∀ (x y : ℝ), x - m*y + 4 = 0 → y = (1/m)*x + 4/m)) →
  m = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l315_31544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_exists_l315_31539

/-- A point on the chessboard grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents a chessboard configuration -/
structure Chessboard where
  n : Nat
  pieces : Finset GridPoint

/-- Predicate to check if four points form a parallelogram -/
def IsParallelogram (p1 p2 p3 p4 : GridPoint) : Prop :=
  (p2.x - p1.x = p4.x - p3.x) ∧ (p2.y - p1.y = p4.y - p3.y)

/-- Main theorem: There exists a parallelogram on the chessboard -/
theorem parallelogram_exists (board : Chessboard) 
  (h1 : board.n ≥ 2) 
  (h2 : board.pieces.card = 2 * board.n) 
  (h3 : ∀ p, p ∈ board.pieces → p.x < board.n ∧ p.y < board.n) :
  ∃ p1 p2 p3 p4, p1 ∈ board.pieces ∧ p2 ∈ board.pieces ∧ 
                 p3 ∈ board.pieces ∧ p4 ∈ board.pieces ∧ 
                 IsParallelogram p1 p2 p3 p4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_exists_l315_31539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_zero_at_negative_one_l315_31596

-- Define the quadratic function
def f (a b c : ℤ) : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

-- State the theorem
theorem not_zero_at_negative_one (a b c : ℤ) (h_a : a ≠ 0) :
  (∀ x : ℝ, (deriv (f a b c)) x = 2 * (a : ℝ) * x + (b : ℝ)) →
  (deriv (f a b c) 1 = 0) →
  (f a b c 1 = 3) →
  (f a b c 2 = 8) →
  (f a b c (-1) ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_zero_at_negative_one_l315_31596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l315_31545

noncomputable def f (x : ℝ) : ℝ := 2 * Real.arcsin (2 / (3 * x + 4)) + Real.sqrt (9 * x^2 + 24 * x + 12)

theorem f_derivative (x : ℝ) (h : 3 * x + 4 > 0) :
  deriv f x = (15 * x + 20) / Real.sqrt (9 * x^2 + 24 * x + 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l315_31545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worst_player_is_granddaughter_l315_31565

-- Define the players
inductive Player
| Grandmother
| Daughter
| Grandson
| Granddaughter

-- Define the sex
inductive Sex
| Male
| Female

-- Define the functions
def twin : Player → Player := sorry
def best_player : Player := sorry
def worst_player : Player := sorry
def sex : Player → Sex := sorry
def age : Player → ℕ := sorry

-- Define the axioms
axiom twin_in_players : ∃ p : Player, twin worst_player = p
axiom opposite_sex : sex (twin worst_player) ≠ sex best_player
axiom same_age : age worst_player = age best_player
axiom grandmother_oldest : ∀ p : Player, p ≠ Player.Grandmother → age Player.Grandmother > age p
axiom parent_older_than_children : age Player.Daughter > age Player.Grandson ∧ age Player.Daughter > age Player.Granddaughter

-- Theorem to prove
theorem worst_player_is_granddaughter : worst_player = Player.Granddaughter := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worst_player_is_granddaughter_l315_31565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_five_l315_31562

/-- An arithmetic sequence with common difference d and first term a₁ -/
noncomputable def arithmetic_sequence (d : ℚ) (a₁ : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S_n (d : ℚ) (a₁ : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem max_sum_at_five (d : ℚ) (a₁ : ℚ) (h_d : d < 0) (h_eq : a₁^2 = (arithmetic_sequence d a₁ 10)^2) :
  ∀ n : ℕ, S_n d a₁ 5 ≥ S_n d a₁ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_five_l315_31562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_l315_31564

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (3/2) * x^2 + 2*x + a

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  let a : ℝ := -3/2
  let tangent_line (x y : ℝ) : Prop := 2*x - y - 6 = 0
  tangent_line 3 (f a 3) ∧ 
  ∀ x y, tangent_line x y ↔ y - (f a 3) = f_derivative 3 * (x - 3) := by
  sorry

-- Theorem for the range of a
theorem range_of_a :
  let has_three_distinct_roots (a : ℝ) : Prop := 
    ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0
  ∀ a, has_three_distinct_roots a ↔ -5/6 < a ∧ a < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_l315_31564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_expression_value_l315_31593

theorem tan_expression_value (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_expression_value_l315_31593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determinant_equals_two_l315_31552

-- Define the angles of an obtuse triangle
variable (A B C : ℝ)

-- Define the conditions
axiom obtuse_triangle : 0 < B ∧ 0 < C ∧ B + C < Real.pi
axiom A_obtuse : A = Real.pi - B - C

-- Define the determinant function
noncomputable def triangle_determinant (A B C : ℝ) : ℝ :=
  Real.tan A * (Real.tan B * Real.tan C - 1) - (Real.tan C - 1) + (1 - Real.tan B)

-- State the theorem
theorem triangle_determinant_equals_two :
  triangle_determinant A B C = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determinant_equals_two_l315_31552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_perimeter_l315_31580

/-- Represents a garden enclosed by six straight fences -/
structure Garden where
  fences : Fin 6 → ℝ
  area : ℝ

/-- The perimeter of the garden -/
def Garden.perimeter (g : Garden) : ℝ :=
  (Finset.sum (Finset.range 6) (λ i => g.fences i))

theorem garden_perimeter (g : Garden) (h : g.area = 97) : g.perimeter = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_perimeter_l315_31580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_20_seconds_l315_31509

/-- The time taken for a train to cross a platform -/
noncomputable def train_crossing_time (train_length platform_length man_passing_time : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed := train_length / man_passing_time
  total_distance / train_speed

/-- Theorem stating that a train of length 178 meters passing a man on a 267-meter platform in 8 seconds takes approximately 20 seconds to cross the platform -/
theorem train_crossing_approx_20_seconds :
  let ε := 0.1
  ∃ (t : ℝ), |t - train_crossing_time 178 267 8| < ε ∧ t = 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_20_seconds_l315_31509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_behavior_l315_31548

-- Define a function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the first and second derivatives of f
variable (f' f'' : ℝ → ℝ)

-- Define the conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_positive (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x > 0, HasDerivAt f (f' x) x ∧ f' x > 0

def is_convex_positive (f' : ℝ → ℝ) (f'' : ℝ → ℝ) : Prop :=
  ∀ x > 0, HasDerivAt f' (f'' x) x ∧ f'' x > 0

-- State the theorem
theorem odd_function_behavior
  (h_odd : is_odd f)
  (h_incr : is_increasing_positive f f')
  (h_conv : is_convex_positive f' f'') :
  (∀ x < 0, HasDerivAt f (f' x) x ∧ f' x > 0) ∧
  (∀ x < 0, HasDerivAt f' (f'' x) x ∧ f'' x < 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_behavior_l315_31548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_asymptote_iff_l315_31510

/-- The function g(x) with parameter c -/
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + c) / (x^2 - 3*x + 2)

/-- A vertical asymptote occurs at x if the denominator is zero at x -/
def has_vertical_asymptote (c : ℝ) (x : ℝ) : Prop :=
  x^2 - 3*x + 2 = 0 ∧ x^2 + 2*x + c ≠ 0

/-- The function g has exactly one vertical asymptote -/
def has_exactly_one_vertical_asymptote (c : ℝ) : Prop :=
  ∃! x, has_vertical_asymptote c x

/-- Theorem: g(x) has exactly one vertical asymptote iff c = -3 or c = -8 -/
theorem g_one_asymptote_iff (c : ℝ) :
  has_exactly_one_vertical_asymptote c ↔ c = -3 ∨ c = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_asymptote_iff_l315_31510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_polygons_is_eight_l315_31527

/-- A polygon within a square -/
structure PolygonInSquare where
  vertices : List (ℤ × ℤ)
  is_parallel_to_square : Bool
  is_non_convex : Bool

/-- Check if two polygons are translations of each other -/
def is_translation (p1 p2 : PolygonInSquare) : Bool := sorry

/-- The set of all valid polygons in the square -/
def valid_polygons : Set PolygonInSquare := sorry

/-- The maximum number of equal non-convex polygons dividing the square -/
def max_polygons : ℕ := sorry

theorem max_polygons_is_eight :
  (∀ p, p ∈ valid_polygons → p.is_parallel_to_square ∧ p.is_non_convex) →
  (∀ p1 p2, p1 ∈ valid_polygons → p2 ∈ valid_polygons → p1 ≠ p2 → ¬is_translation p1 p2) →
  max_polygons = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_polygons_is_eight_l315_31527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_odd_sum_l315_31589

theorem always_odd_sum : 
  ∀ (f : Fin 8 → Bool), 
  ∃ (result : ℤ), 
  result = 1 + (if f 0 then 2 else -2) + 
           (if f 1 then 3 else -3) + 
           (if f 2 then 4 else -4) + 
           (if f 3 then 5 else -5) + 
           (if f 4 then 6 else -6) + 
           (if f 5 then 7 else -7) + 
           (if f 6 then 8 else -8) + 
           (if f 7 then 9 else -9) ∧ 
  result % 2 = 1 := by
  sorry

#check always_odd_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_odd_sum_l315_31589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_a_to_b_is_two_thirds_l315_31585

/-- An arithmetic sequence with four initial terms -/
structure ArithmeticSequence where
  a₁ : ℚ
  a₂ : ℚ
  a₃ : ℚ
  a₄ : ℚ

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℚ := (seq.a₄ - seq.a₂) / 2

theorem ratio_of_a_to_b_is_two_thirds (seq₁ seq₂ : ArithmeticSequence)
  (h₁ : seq₁.a₂ = seq₂.a₄)
  (h₂ : seq₁.a₄ = 14)
  (h₃ : seq₂.a₁ = 2)
  (h₄ : seq₂.a₃ = 6) :
  seq₁.a₁ / seq₁.a₃ = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_a_to_b_is_two_thirds_l315_31585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_i_nonzero_l315_31511

/-- A polynomial of degree 5 with five distinct real roots, one of which is 1 -/
def Q (f g h i j : ℝ) : ℝ → ℝ := λ x ↦ x^5 + f*x^4 + g*x^3 + h*x^2 + i*x + j

/-- The theorem stating that the coefficient of x (i) must be non-zero -/
theorem coefficient_i_nonzero (f g h i j : ℝ) :
  (∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ 1 ∧ 
                  b ≠ c ∧ b ≠ d ∧ b ≠ 1 ∧ 
                  c ≠ d ∧ c ≠ 1 ∧ 
                  d ≠ 1 ∧
                  ∀ x, Q f g h i j x = (x - 1) * (x - a) * (x - b) * (x - c) * (x - d)) →
  i ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_i_nonzero_l315_31511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_n_independent_iff_n_2_or_4_l315_31558

/-- An equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : ∀ (X Y : ℝ × ℝ), X ∈ ({A, B, C} : Set (ℝ × ℝ)) → Y ∈ ({A, B, C} : Set (ℝ × ℝ)) → X ≠ Y → 
    (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2

/-- A point on the circumcircle of an equilateral triangle -/
def PointOnCircumcircle (t : EquilateralTriangle) (P : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (X : ℝ × ℝ), X ∈ ({t.A, t.B, t.C} : Set (ℝ × ℝ)) → 
    (P.1 - X.1)^2 + (P.2 - X.2)^2 = r^2

/-- The sum S_n(P) for a given n and point P -/
noncomputable def S_n (t : EquilateralTriangle) (n : ℕ) (P : ℝ × ℝ) : ℝ :=
  ((P.1 - t.A.1)^2 + (P.2 - t.A.2)^2)^(n/2 : ℝ) +
  ((P.1 - t.B.1)^2 + (P.2 - t.B.2)^2)^(n/2 : ℝ) +
  ((P.1 - t.C.1)^2 + (P.2 - t.C.2)^2)^(n/2 : ℝ)

/-- The main theorem -/
theorem S_n_independent_iff_n_2_or_4 (t : EquilateralTriangle) :
  (∀ (P Q : ℝ × ℝ), PointOnCircumcircle t P → PointOnCircumcircle t Q → 
    ∀ (n : ℕ), S_n t n P = S_n t n Q) ↔ n = 2 ∨ n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_n_independent_iff_n_2_or_4_l315_31558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_equation_unique_solution_l315_31581

theorem factorial_sum_equation_unique_solution :
  ∀ k ℓ m n : ℕ,
    k > 0 → ℓ > 0 → m > 0 → n > 0 →
    (1 : ℚ) / k.factorial + (1 : ℚ) / ℓ.factorial + (1 : ℚ) / m.factorial = (1 : ℚ) / n.factorial →
    k = 3 ∧ ℓ = 3 ∧ m = 3 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_equation_unique_solution_l315_31581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_11_eq_66_l315_31574

/-- An arithmetic sequence with a given condition -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  condition : a 2 + a 8 - a 4 = 6

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- Theorem: For the given arithmetic sequence, S₁₁ = 66 -/
theorem sum_11_eq_66 (seq : ArithmeticSequence) : sum_n seq 11 = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_11_eq_66_l315_31574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_statement_l315_31597

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, Real.sin x < 2) ↔ (∃ x : ℝ, Real.sin x ≥ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_statement_l315_31597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_pill_cost_is_25_l315_31549

/-- Represents the duration in days -/
def duration : ℕ := 21

/-- Represents the total cost in dollars -/
def totalCost : ℕ := 966

/-- Represents the price difference between blue and red pills in dollars -/
def priceDifference : ℕ := 4

/-- Represents the daily intake of blue pills -/
def dailyBluePills : ℕ := 1

/-- Represents the daily intake of red pills -/
def dailyRedPills : ℕ := 1

/-- Calculates the cost of a blue pill -/
def bluePillCost : ℕ :=
  let dailyCost : ℕ := totalCost / duration
  dailyCost / 2 + priceDifference / 2

/-- Theorem stating that the blue pill cost is 25 dollars -/
theorem blue_pill_cost_is_25 : bluePillCost = 25 := by
  -- The proof goes here
  sorry

#eval bluePillCost  -- This will evaluate and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_pill_cost_is_25_l315_31549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_transformation_theorem_l315_31592

theorem data_transformation_theorem (data : List ℝ) (n : ℕ) 
  (h_nonempty : data.length = n ∧ n > 0) 
  (h_transformed_avg : (data.map (λ x => 2 * x - 80)).sum / (n : ℝ) = 1.2)
  (h_transformed_var : ((data.map (λ x => 2 * x - 80)).map (λ y => (y - 1.2)^2)).sum / (n : ℝ) = 4.4) :
  data.sum / (n : ℝ) = 40.6 ∧ ((data.map (λ x => (x - 40.6)^2)).sum / (n : ℝ) = 1.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_transformation_theorem_l315_31592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_navigation_time_l315_31505

-- Define the water depth function
noncomputable def water_depth (t : ℝ) : ℝ := 3 * Real.sin (Real.pi * t / 6) + 10

-- Define the safe depth threshold
def safe_depth : ℝ := 11.5

-- Theorem statement
theorem safe_navigation_time :
  ∃ (t₁ t₂ : ℝ), 
    0 ≤ t₁ ∧ t₁ < t₂ ∧ t₂ ≤ 24 ∧
    (∀ t, t₁ ≤ t ∧ t ≤ t₂ → water_depth t ≥ safe_depth) ∧
    t₂ - t₁ = 16 ∧
    (∀ t₃ t₄, 0 ≤ t₃ ∧ t₃ < t₄ ∧ t₄ ≤ 24 ∧
      (∀ t, t₃ ≤ t ∧ t ≤ t₄ → water_depth t ≥ safe_depth) →
      t₄ - t₃ ≤ 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_navigation_time_l315_31505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_five_consecutive_divisible_by_2005_l315_31573

/-- The sequence a_n = 1 + 2^n + 3^n + 4^n + 5^n -/
def a (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

/-- There do not exist five consecutive terms in the sequence a_n that are all divisible by 2005 -/
theorem no_five_consecutive_divisible_by_2005 :
  ¬ ∃ k, ∀ i ∈ Finset.range 5, 2005 ∣ a (k + i) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_five_consecutive_divisible_by_2005_l315_31573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bailing_rate_bounds_l315_31540

/-- Represents the scenario of Steve and LeRoy's fishing trip -/
structure FishingTrip where
  distanceToShore : ℝ
  waterIntakeRate : ℝ
  maxWaterCapacity : ℝ
  rowingSpeed : ℝ

/-- Calculates the minimum bailing rate required to prevent sinking -/
noncomputable def minBailingRate (trip : FishingTrip) : ℝ :=
  (trip.waterIntakeRate * (trip.distanceToShore / trip.rowingSpeed) * 60 - trip.maxWaterCapacity) / 
  (trip.distanceToShore / trip.rowingSpeed * 60)

/-- Theorem stating the bounds on the minimum bailing rate -/
theorem bailing_rate_bounds (trip : FishingTrip) 
  (h1 : trip.distanceToShore = 3)
  (h2 : trip.waterIntakeRate = 15)
  (h3 : trip.maxWaterCapacity = 50)
  (h4 : trip.rowingSpeed = 3) :
  14 < minBailingRate trip ∧ minBailingRate trip < 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bailing_rate_bounds_l315_31540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l315_31590

-- First expression
theorem simplify_expression_1 :
  Real.sqrt ((3 - Real.pi) ^ 2) + (0.008 : ℝ) ^ (1/3 : ℝ) + (0.25 : ℝ) ^ (1/2 : ℝ) * ((1 / Real.sqrt 2) ^ (-4 : ℝ)) - Real.exp (Real.log Real.pi) = -4/5 := by
  sorry

-- Second expression
theorem simplify_expression_2 :
  ∀ x : ℝ, Real.log (Real.sqrt (x^2 + 1) + x) + Real.log (Real.sqrt (x^2 + 1) - x) + (Real.log 2 / Real.log 10)^2 + (1 + Real.log 2 / Real.log 10) * (Real.log 5 / Real.log 10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l315_31590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_log_over_x_minus_one_l315_31508

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (log x) / (x + 1) + 1 / x

-- State the theorem
theorem f_greater_than_log_over_x_minus_one (x : ℝ) 
  (h1 : x > 0) (h2 : x ≠ 1) : f x > log x / (x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_log_over_x_minus_one_l315_31508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_neg_one_f_greater_than_exp_neg_x_implies_k_positive_l315_31588

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (2 : ℝ)^x + k * (2 : ℝ)^(-x)

-- Theorem 1: If f is odd, then k = -1
theorem odd_function_implies_k_neg_one (k : ℝ) :
  (∀ x, f k (-x) = -(f k x)) → k = -1 :=
by sorry

-- Theorem 2: If f(x) > 2^(-x) for all x ≥ 0, then k > 0
theorem f_greater_than_exp_neg_x_implies_k_positive (k : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f k x > (2 : ℝ)^(-x)) → k > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_neg_one_f_greater_than_exp_neg_x_implies_k_positive_l315_31588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_plane_theorem_l315_31501

/-- A color type with two possible values -/
inductive Color
| Red
| Blue

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A coloring of the plane -/
def Coloring := Point → Color

/-- The coloring uses both colors -/
def uses_both_colors (c : Coloring) : Prop :=
  ∃ p q : Point, c p = Color.Red ∧ c q = Color.Blue

theorem two_color_plane_theorem (c : Coloring) (h : uses_both_colors c) :
  (∃ p q : Point, c p = c q ∧ distance p q = 2006) ∧
  (∃ p q : Point, c p ≠ c q ∧ distance p q = 2006) := by
  sorry

#check two_color_plane_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_plane_theorem_l315_31501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wearing_hats_l315_31598

theorem total_wearing_hats (total : ℕ) (men_fraction : ℚ) (men_hat_percent : ℚ) (women_hat_percent : ℚ)
  (h1 : total = 1800)
  (h2 : men_fraction = 2 / 3)
  (h3 : men_hat_percent = 15 / 100)
  (h4 : women_hat_percent = 25 / 100) :
  (men_fraction * ↑total * men_hat_percent).floor + ((1 - men_fraction) * ↑total * women_hat_percent).floor = 330 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wearing_hats_l315_31598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_range_l315_31557

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x) - Real.sqrt 3 * (Real.cos x)^2

noncomputable def g (x : ℝ) : ℝ := f (x/2)

theorem f_properties_and_g_range :
  (∃ T : ℝ, T > 0 ∧ T = π ∧ ∀ x, f (x + T) = f x ∧
   ∀ T' : ℝ, T' > 0 ∧ (∀ x, f (x + T') = f x) → T ≤ T') ∧
  (∀ x, f x ≥ -(2 + Real.sqrt 3)/2) ∧
  (∃ x, f x = -(2 + Real.sqrt 3)/2) ∧
  (∀ y ∈ Set.Icc ((1 - Real.sqrt 3)/2) ((2 - Real.sqrt 3)/2),
    ∃ x ∈ Set.Ioo (π/2) π, g x = y) ∧
  (∀ x ∈ Set.Ioo (π/2) π,
    g x ∈ Set.Icc ((1 - Real.sqrt 3)/2) ((2 - Real.sqrt 3)/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_range_l315_31557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l315_31567

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) / (x - 1)

def DomainOf (f : ℝ → ℝ) : Set ℝ :=
  {x | ∃ y, f x = y}

theorem domain_of_f :
  DomainOf f = {x : ℝ | x ≥ -1 ∧ x ≠ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l315_31567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_properties_l315_31591

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -5 ∧ x ≤ -2 then 3 + (x + 2)
  else if x > -2 ∧ x < 2 then -Real.sqrt (9 - x^2)
  else if x ≥ 2 ∧ x ≤ 5 then (x - 2) - 3
  else 0  -- Define a default value for x outside [-5, 5]

-- State the properties of |g(x)|
theorem abs_g_properties :
  (∀ x ∈ Set.Icc (-5 : ℝ) 5, ContinuousOn (fun x => |g x|) (Set.Icc (-5 : ℝ) 5)) ∧
  (∀ x ∈ Set.Icc (-5 : ℝ) (-2), |g x| = 3 + (x + 2)) ∧
  (∀ x ∈ Set.Ioo (-2 : ℝ) 2, |g x| = Real.sqrt (9 - x^2)) ∧
  (∀ x ∈ Set.Icc (2 : ℝ) 5, |g x| = (x - 2) - 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_properties_l315_31591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_singleton_in_partition_l315_31579

theorem max_singleton_in_partition (n : ℕ) (A : Fin n → Set ℕ) : 
  (∀ i : Fin n, ∀ r s : ℕ, r ∈ A i → s ∈ A i → r ≠ s → r + s ∈ A i) →
  (∃ j : Fin n, ∃ k : ℕ, A j = {k}) →
  (∀ x : ℕ, ∃ i : Fin n, x ∈ A i) →
  (∀ i j : Fin n, i ≠ j → A i ∩ A j = ∅) →
  ∃ j : Fin n, ∃ k : ℕ, A j = {k} ∧ k ≤ n - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_singleton_in_partition_l315_31579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l315_31502

theorem solve_exponential_equation (y : ℝ) : (1000 : ℝ)^4 = 10^y → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l315_31502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l315_31520

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x / (x - 4) + Real.sqrt (x + 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ -2 ∧ x ≠ 4}

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l315_31520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_reflected_arcs_area_l315_31542

noncomputable section

/-- The area of the region bounded by 8 reflected arcs of a circle, where each arc is determined by a side of a regular octagon with side length 1 inscribed in the circle -/
def boundedAreaOctagon : ℝ := 2 + 2 * Real.sqrt 2 - Real.pi / (2 - Real.sqrt 2) + 2 * Real.sqrt (2 - Real.sqrt 2)

/-- The side length of the regular octagon -/
def sideLength : ℝ := 1

/-- The number of sides in the regular octagon -/
def numSides : ℕ := 8

theorem octagon_reflected_arcs_area :
  let r : ℝ := 1 / Real.sqrt (2 - Real.sqrt 2)
  let octagonArea : ℝ := 2 * (1 + Real.sqrt 2) * sideLength ^ 2
  let reflectedArcArea : ℝ := Real.pi / (8 * (2 - Real.sqrt 2)) - Real.sqrt (2 - Real.sqrt 2) / 4
  octagonArea - numSides * reflectedArcArea = boundedAreaOctagon := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_reflected_arcs_area_l315_31542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_twentyninth_415th_digit_l315_31515

/-- The decimal representation of 7/29 as a sequence of digits -/
def decimal_rep : ℕ → ℕ := sorry

/-- The length of the repeating cycle in the decimal representation of 7/29 -/
def cycle_length : ℕ := 58

/-- The 415th digit after the decimal point in the representation of 7/29 -/
def digit_415 : ℕ := decimal_rep 415

theorem seventh_twentyninth_415th_digit :
  digit_415 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_twentyninth_415th_digit_l315_31515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_100_l315_31554

/-- A point satisfying the conditions of the problem -/
structure SpecialPoint where
  x : ℝ
  y : ℝ
  distance_from_line : |y - 15| = 7
  distance_from_point : (x - 10)^2 + (y - 15)^2 = 15^2

/-- The set of all points satisfying the conditions -/
def SpecialPoints : Set SpecialPoint :=
  {p : SpecialPoint | True}

theorem sum_of_coordinates_is_100 (points : Finset SpecialPoint) 
    (h : points.card = 4) (h' : ∀ p, p ∈ points → p ∈ SpecialPoints) : 
    (points.sum (λ p => p.x + p.y)) = 100 := by
  sorry

#check sum_of_coordinates_is_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_100_l315_31554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_plus_area_eq_fifty_thirds_l315_31535

/-- Represents a cube with cuts forming a pyramid at one corner -/
structure CutCube where
  edge_length : ℝ
  is_valid : edge_length > 0

/-- Calculates the volume of the pyramid formed by the cuts -/
noncomputable def pyramid_volume (cube : CutCube) : ℝ :=
  (1/3) * (cube.edge_length/2 * cube.edge_length/2) * cube.edge_length

/-- Calculates the iced surface area of the pyramid -/
noncomputable def iced_surface_area (cube : CutCube) : ℝ :=
  (cube.edge_length/2 * cube.edge_length/2) + 3 * (1/2 * cube.edge_length * cube.edge_length)

/-- Theorem stating that the sum of pyramid volume and iced surface area is 50/3 for a cube with edge length 4 -/
theorem volume_plus_area_eq_fifty_thirds (cube : CutCube) 
  (h : cube.edge_length = 4) : 
  pyramid_volume cube + iced_surface_area cube = 50/3 := by
  sorry

#eval 50/3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_plus_area_eq_fifty_thirds_l315_31535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_on_line_segment_l315_31560

-- Define points A and B on the coordinate plane
variable (A B : ℝ × ℝ)

-- Define the distance between A and B
def distance_AB : ℝ := 2

-- Define point P
variable (P : ℝ × ℝ)

-- Define the sum of distances from P to A and B
noncomputable def sum_distances (P A B : ℝ × ℝ) : ℝ :=
  (((P.1 - A.1)^2 + (P.2 - A.2)^2).sqrt) +
  (((P.1 - B.1)^2 + (P.2 - B.2)^2).sqrt)

-- Theorem statement
theorem point_P_on_line_segment (h1 : sum_distances P A B = distance_AB) :
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_on_line_segment_l315_31560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l315_31571

/-- Represents the state of the game -/
structure GameState where
  coins : ℕ

/-- Represents a valid move in the game -/
def ValidMove (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

/-- Defines the game rules -/
def GameRules (state : GameState) (move : ℕ) (new_state : GameState) : Prop :=
  ValidMove move ∧ new_state.coins = state.coins - move

/-- Defines a winning state -/
def IsWinningState (state : GameState) : Prop :=
  state.coins = 0

/-- Defines a strategy for a player -/
def Strategy := GameState → ℕ

/-- Defines a winning strategy for the second player -/
def SecondPlayerWinningStrategy (s : Strategy) : Prop :=
  ∀ (initial_state : GameState),
    initial_state.coins = 60 →
    ∀ (first_move : ℕ),
      ValidMove first_move →
      ∃ (second_move : ℕ),
        ValidMove second_move ∧
        ∀ (game_sequence : ℕ → GameState),
          game_sequence 0 = initial_state →
          (∀ n : ℕ, GameRules (game_sequence n) (if n % 2 = 0 then first_move else s (game_sequence n)) (game_sequence (n+1))) →
          ∃ k : ℕ, IsWinningState (game_sequence (2*k+1))

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (s : Strategy), SecondPlayerWinningStrategy s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l315_31571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l315_31587

theorem constant_term_expansion (x : ℝ) :
  let expansion := (x^(1/4) - 3/x)^5
  ∃ (c : ℝ), c = -15 ∧ 
    (∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |expansion - c| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l315_31587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_p_l315_31543

noncomputable def p (x : ℝ) : ℝ := 7 + 2*x^2 + 3*Real.sqrt 3 + 12*x^5 + 4*Real.pi*x + 5*x^5 - 3

/-- The degree of a polynomial -/
def polynomial_degree (p : ℝ → ℝ) : ℕ := sorry

theorem degree_of_p :
  polynomial_degree p = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_p_l315_31543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_cylinder_surface_area_l315_31536

theorem hemisphere_cylinder_surface_area 
  (base_area : ℝ) 
  (cylinder_height : ℝ) : 
  base_area = 144 * Real.pi → 
  cylinder_height = 10 → 
  (let r := Real.sqrt (base_area / Real.pi);
   let hemisphere_area := 2 * base_area;
   let cylinder_side_area := 2 * Real.pi * r * cylinder_height;
   let total_area := hemisphere_area + cylinder_side_area + base_area;
   total_area = 672 * Real.pi) := by
  intro h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_cylinder_surface_area_l315_31536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_y_exists_l315_31500

noncomputable def x : ℝ := Real.rpow (4/25) (1/3)

def satisfies_property (y : ℝ) : Prop :=
  0 < y ∧ y < x ∧ Real.rpow x x = Real.rpow y y

theorem unique_y_exists : ∃! y : ℝ, satisfies_property y ∧ y = Real.rpow (32/3125) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_y_exists_l315_31500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l315_31533

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - 1 + Real.sqrt (6 * x - x^2)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-1 : ℝ) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l315_31533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_profit_percentage_approx_30_l315_31524

/-- Represents a trader with faulty weights -/
structure FaultyTrader where
  buyingFactor : ℝ
  sellingFactor : ℝ

/-- Calculates the actual weight received by the trader when buying -/
noncomputable def actualBuyWeight (t : FaultyTrader) (indicatedWeight : ℝ) : ℝ :=
  indicatedWeight * (1 + t.buyingFactor)

/-- Calculates the weight given to the customer -/
noncomputable def weightGivenToCustomer (t : FaultyTrader) (claimedWeight : ℝ) : ℝ :=
  claimedWeight / (1 + t.sellingFactor)

/-- Calculates the profit percentage of the trader -/
noncomputable def profitPercentage (t : FaultyTrader) (indicatedWeight : ℝ) : ℝ :=
  let actualBuy := actualBuyWeight t indicatedWeight
  let givenToCustomer := weightGivenToCustomer t indicatedWeight
  (actualBuy - givenToCustomer) / givenToCustomer * 100

/-- Theorem stating that the trader's profit percentage is approximately 30% -/
theorem trader_profit_percentage_approx_30 (t : FaultyTrader) (w : ℝ) :
  t.buyingFactor = 0.1 → t.sellingFactor = 0.3 → w > 0 →
  ∃ ε > 0, |profitPercentage t w - 30| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_profit_percentage_approx_30_l315_31524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l315_31555

/-- Given an ellipse C with the following properties:
    - Its equation is of the form y²/a² + x²/b² = 1 where a > b > 0
    - Its eccentricity is √2/2
    - F₁ and F₂ are its upper and lower foci respectively
    - A line l passing through F₂ intersects C at points A and B
    - The perimeter of triangle ABF₁ is 4√2
    - There's a point P on the y-axis with coordinates (0, -2)
    - The ratio |F₂A|/|F₂B| is between 1/2 and 1 inclusive

    This theorem proves that:
    1. The standard equation of C is y²/2 + x² = 1
    2. The range of the length of diagonal PQ in parallelogram PAQB is [2, 13√2/8]
-/
theorem ellipse_properties (a b : ℝ) (F₁ F₂ A B P : ℝ × ℝ) (l C : Set (ℝ × ℝ)) :
  a > b ∧ b > 0 ∧
  (∀ x y, (y^2 / a^2 + x^2 / b^2 = 1) ↔ (x, y) ∈ C) ∧
  (Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2) ∧
  F₂.2 < F₁.2 ∧
  F₂ ∈ l ∧ A ∈ l ∩ C ∧ B ∈ l ∩ C ∧ A ≠ B ∧
  dist A F₁ + dist B F₁ + dist A B = 4 * Real.sqrt 2 ∧
  P = (0, -2) ∧
  1/2 ≤ dist F₂ A / dist F₂ B ∧ dist F₂ A / dist F₂ B ≤ 1 →
  ((∀ x y, (y^2 / 2 + x^2 = 1) ↔ (x, y) ∈ C) ∧
   (2 ≤ ‖P - (A + B - P)‖ ∧ ‖P - (A + B - P)‖ ≤ 13 * Real.sqrt 2 / 8)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l315_31555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_girls_count_l315_31577

-- Define a predicate 'collects_stamps' for demonstration purposes
axiom collects_stamps : ℕ → Prop

theorem max_girls_count (total : ℕ) (ussr : ℕ) (africa : ℕ) (america : ℕ)
  (only_ussr : ℕ) (only_africa : ℕ) (only_america : ℕ) (all_three : ℕ)
  (h_total : total = 150)
  (h_ussr : ussr = 67)
  (h_africa : africa = 48)
  (h_america : america = 32)
  (h_only_ussr : only_ussr = 11)
  (h_only_africa : only_africa = 7)
  (h_only_america : only_america = 4)
  (h_all_three : all_three = 1)
  (h_boys_only : ∀ n : ℕ, n ≤ total ∧ n ≠ 0 → (∃ m : ℕ, m ≤ n ∧ m ≠ 0 ∧ collects_stamps m)) :
  ∃ (girls : ℕ), girls = total - (ussr + africa + america - only_ussr - only_africa - only_america - 2 * all_three) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_girls_count_l315_31577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_l315_31528

/-- Triangle ABC with positive integer side lengths, AB = AC, and BI = 10 where I is the 
    intersection of angle bisectors of ∠B and ∠C. One angle is an integer number of degrees. -/
def MinPerimeterTriangle (a b c : ℕ+) (i : ℝ × ℝ) : Prop :=
  a = b ∧ 
  ∃ (angle : ℕ), angle > 0 ∧ angle < 180 ∧
  ∃ (bi : ℝ), bi = 10 ∧
  -- Additional conditions to define the triangle and point I
  True

/-- The smallest possible perimeter of triangle ABC under the given conditions is 40. -/
theorem smallest_perimeter (a b c : ℕ+) (i : ℝ × ℝ) 
  (h : MinPerimeterTriangle a b c i) : 
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 40 := by
  sorry

#check smallest_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_l315_31528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l315_31594

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- The side length of the base -/
  base_side : ℝ
  /-- The distance between a diagonal of the base and a lateral edge skew to it -/
  diagonal_edge_distance : ℝ
  /-- The base side length is positive -/
  base_side_pos : 0 < base_side
  /-- The diagonal to edge distance is 1/2 -/
  diagonal_edge_distance_eq : diagonal_edge_distance = 1/2

/-- The radius of the circumscribed sphere of a regular quadrilateral pyramid -/
noncomputable def circumscribed_sphere_radius (p : RegularQuadrilateralPyramid) : ℝ :=
  2 * p.base_side * Real.sqrt 2 / Real.sqrt 7

/-- Theorem: The radius of the circumscribed sphere of a regular quadrilateral pyramid
    is (2a√2)/√7, where a is the side length of the base -/
theorem circumscribed_sphere_radius_formula (p : RegularQuadrilateralPyramid) :
  circumscribed_sphere_radius p = 2 * p.base_side * Real.sqrt 2 / Real.sqrt 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l315_31594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_pi_l315_31513

-- Define the functions
noncomputable def f₁ (x : ℝ) := Real.sin (abs (2 * x))
noncomputable def f₂ (x : ℝ) := abs (Real.cos x)
noncomputable def f₃ (x : ℝ) := Real.cos (2 * x + Real.pi / 6)
noncomputable def f₄ (x : ℝ) := Real.tan (2 * x - Real.pi / 4)

-- Define the period of a function
def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Define the smallest positive period
def isSmallestPositivePeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ isPeriodic f T ∧ ∀ T' > 0, isPeriodic f T' → T ≤ T'

-- Theorem statement
theorem smallest_period_pi :
  isSmallestPositivePeriod f₂ Real.pi ∧
  isSmallestPositivePeriod f₃ Real.pi ∧
  (¬ isSmallestPositivePeriod f₁ Real.pi) ∧
  (¬ isSmallestPositivePeriod f₄ Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_pi_l315_31513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l315_31576

/-- The maximum area of a triangle ABC where:
    A = (2,0), B = (5,3), C = (p, -p^2 + 8p - 12), and 2 ≤ p ≤ 5 -/
theorem max_triangle_area :
  let A : ℝ × ℝ := (2, 0)
  let B : ℝ × ℝ := (5, 3)
  let C (p : ℝ) : ℝ × ℝ := (p, -p^2 + 8*p - 12)
  let area (p : ℝ) : ℝ := abs ((A.1 * B.2 + B.1 * (C p).2 + (C p).1 * A.2
                               - A.2 * B.1 - B.2 * (C p).1 - (C p).2 * A.1) / 2)
  ∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ 
    (∀ q : ℝ, 2 ≤ q ∧ q ≤ 5 → area q ≤ area p) ∧
    area p = 27/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l315_31576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_access_three_cards_prob_access_all_cards_l315_31578

/-- Represents a credit card with a PIN -/
structure CreditCard where
  pin : ℕ

/-- Represents the thief's attempt to access cards -/
structure AccessAttempt where
  cards : Finset CreditCard
  pins : Finset ℕ
  h_cards : cards.card = 4
  h_pins : pins.card = 4

/-- Represents the success of accessing at least 3 cards -/
def AccessThreeCards (attempt : AccessAttempt) : Prop :=
  ∃ (accessed : Finset CreditCard), accessed.card ≥ 3 ∧ accessed ⊆ attempt.cards

/-- Represents the probability of accessing all 4 cards -/
def ProbAccessAllCards (attempt : AccessAttempt) : ℚ :=
  23 / 24

/-- Theorem stating that it's possible to access at least 3 cards -/
theorem can_access_three_cards (attempt : AccessAttempt) :
  AccessThreeCards attempt :=
sorry

/-- Theorem stating the probability of accessing all 4 cards -/
theorem prob_access_all_cards (attempt : AccessAttempt) :
  ProbAccessAllCards attempt = 23 / 24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_access_three_cards_prob_access_all_cards_l315_31578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_max_value_l315_31541

/-- Given vectors and their relationship, prove the maximum value of a quadratic expression -/
theorem vector_max_value (α : ℝ) :
  let a : Fin 2 → ℝ := ![1, 1]
  let b : Fin 2 → ℝ := ![1, -1]
  let c : Fin 2 → ℝ := ![Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α]
  (∃ (max : ℝ), ∀ (m n : ℝ), m • a + n • b = c → (m - 3)^2 + n^2 ≤ max) ∧
  (∃ (m n : ℝ), m • a + n • b = c ∧ (m - 3)^2 + n^2 = 16) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_max_value_l315_31541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l315_31550

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x⁻¹

-- State the theorem
theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x > 1 → f (x + 2) > f (2 * x + 1)) ∧
  (∃ x : ℝ, x ≤ 1 ∧ f (x + 2) > f (2 * x + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l315_31550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_is_one_range_of_m_l315_31595

/-- The circle equation x^2 + (y-1)^2 = 2 -/
def circleEq (x y : ℝ) : Prop := x^2 + (y-1)^2 = 2

/-- The inequality condition x + y + m ≥ 0 -/
def inequalityCond (x y m : ℝ) : Prop := x + y + m ≥ 0

/-- The minimum value of m that satisfies the conditions is 1 -/
theorem min_m_is_one :
  ∀ m : ℝ, (∀ x y : ℝ, circleEq x y → inequalityCond x y m) → m ≥ 1 :=
by sorry

/-- The range of m is [1, +∞) -/
theorem range_of_m :
  ∀ m : ℝ, (∀ x y : ℝ, circleEq x y → inequalityCond x y m) ↔ m ∈ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_is_one_range_of_m_l315_31595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_origin_l315_31570

theorem parabola_point_distance_to_origin :
  ∀ (x y : ℝ),
  y^2 = 2*x →
  (x + 1/2) / y = 9/4 →
  ((x - 1/2)^2 + y^2)^(1/2) > 2 →
  (x^2 + y^2)^(1/2) = 4 * (5^(1/2)) := by
  intros x y h1 h2 h3
  sorry

#check parabola_point_distance_to_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_origin_l315_31570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_quotient_five_quotient_integer_condition_next_consecutive_set_l315_31525

def consecutive_integers (x : ℤ) : List ℤ := [x - 1, x, x + 1]

def product (l : List ℤ) : ℤ := l.foldl (·*·) 1

def sum (l : List ℤ) : ℤ := l.foldl (·+·) 0

noncomputable def quotient (x : ℤ) : ℚ := (product (consecutive_integers x) : ℚ) / (sum (consecutive_integers x) : ℚ)

theorem consecutive_integers_quotient_five (x : ℤ) :
  quotient x = 5 → x = 4 := by sorry

theorem quotient_integer_condition (x : ℤ) :
  (quotient x).isInt ↔ x % 3 = 1 ∨ x % 3 = 2 := by sorry

theorem next_consecutive_set (x : ℤ) :
  x = 4 ∧ (quotient x).isInt → (quotient (x + 1)).isInt ∧ consecutive_integers (x + 1) = [4, 5, 6] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_quotient_five_quotient_integer_condition_next_consecutive_set_l315_31525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l315_31546

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  d_neg : d < 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

theorem arithmetic_sequence_property
  (seq : ArithmeticSequence)
  (m : ℕ)
  (h_m : m ≥ 3)
  (h_eq : seq.a m = sum_n seq m) :
  ∀ n : ℕ, n > 0 → (sum_n seq n < seq.a n ↔ n > m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l315_31546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l315_31517

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (a + 1) * x + 1 else x^2 - 2 * a * x + 2

-- State the theorem
theorem range_of_a_for_increasing_f (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  -1 < a ∧ a ≤ 1/3 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l315_31517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_lengths_l315_31599

theorem right_triangle_side_lengths : ∀ x : ℝ,
  x > 0 ∧ (x + 1) > 0 ∧ (2*x - 1) > 0 →
  (x^2 + (x+1)^2 = (2*x-1)^2) →
  x = 3 :=
λ x h1 h2 ↦ by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_lengths_l315_31599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_angled_triangle_area_l315_31572

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines an ellipse with semi-major axis 7 and semi-minor axis 2√6 -/
def Ellipse (p : Point) : Prop :=
  p.y^2 / 49 + p.x^2 / 24 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if three points form a right-angled triangle -/
def isRightAngled (p1 p2 p3 : Point) : Prop :=
  (distance p1 p2)^2 + (distance p2 p3)^2 = (distance p1 p3)^2

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)|

/-- The theorem to be proved -/
theorem ellipse_right_angled_triangle_area :
  ∀ (p : Point) (f1 f2 : Point),
    Ellipse p →
    f1 = Point.mk 0 (-5) →
    f2 = Point.mk 0 5 →
    isRightAngled f1 p f2 →
    triangleArea f1 p f2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_angled_triangle_area_l315_31572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_terms_eq_7_5_l315_31512

/-- An arithmetic sequence with given first and third terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = 3
  third_term : a 3 = 2

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * (seq.a 2 - seq.a 1))

/-- Theorem: The sum of the first 10 terms of the given arithmetic sequence is 7.5 -/
theorem sum_10_terms_eq_7_5 (seq : ArithmeticSequence) : sum_n_terms seq 10 = 15/2 := by
  sorry

#eval (15 : ℚ) / 2  -- To verify that 15/2 is indeed 7.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_terms_eq_7_5_l315_31512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_CD_C_l315_31519

-- Define the circles and points
variable (Γ₁ Γ₂ : Set (EuclideanSpace ℝ (Fin 2)))
variable (A B C D C' D' : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
axiom distinct_intersections : A ≠ B ∧ A ∈ Γ₁ ∩ Γ₂ ∧ B ∈ Γ₁ ∩ Γ₂
axiom C_on_Γ₁ : C ∈ Γ₁ ∧ C ≠ A ∧ C ≠ B
axiom D_on_Γ₁ : D ∈ Γ₁ ∧ D ≠ A ∧ D ≠ B ∧ D ≠ C
axiom C'_on_Γ₂ : C' ∈ Γ₂ ∧ C' ≠ A ∧ C' ≠ B
axiom D'_on_Γ₂ : D' ∈ Γ₂ ∧ D' ≠ A ∧ D' ≠ B ∧ D' ≠ C'
axiom C'_on_BC : ∃ t : ℝ, C' = (1 - t) • B + t • C
axiom D'_on_AD : ∃ s : ℝ, D' = (1 - s) • A + s • D

-- Define the theorem
theorem lines_CD_C'D'_parallel :
  ∃ (k : ℝ), C' - D' = k • (C - D) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_CD_C_l315_31519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_ratio_approx_l315_31559

-- Define the side lengths of the square and pentagon
noncomputable def side_square : ℝ := sorry
noncomputable def side_pentagon : ℝ := sorry

-- Define the areas of the square and pentagon
noncomputable def area_square : ℝ := side_square ^ 2
noncomputable def area_pentagon : ℝ := (5 * side_pentagon ^ 2 * Real.tan (54 * Real.pi / 180)) / 4

-- State that the areas are equal
axiom areas_equal : area_square = area_pentagon

-- Define the ratio of side lengths
noncomputable def side_ratio : ℝ := side_square / side_pentagon

-- Theorem to prove
theorem side_ratio_approx :
  ∃ ε > 0, |side_ratio - Real.sqrt ((5 * Real.tan (54 * Real.pi / 180)) / 4)| < ε := by sorry

-- Note: We use ∃ ε > 0, |x - y| < ε to represent "x is approximately equal to y"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_ratio_approx_l315_31559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_between_complex_numbers_l315_31556

theorem smallest_distance_between_complex_numbers (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 4*Complex.I)) = 2) 
  (hw : Complex.abs (w - (-3 + 2*Complex.I)) = 5) : 
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 61 - 7 ∧ 
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 - 4*Complex.I)) = 2 → 
      Complex.abs (w' - (-3 + 2*Complex.I)) = 5 → 
      Complex.abs (z' - w') ≥ min_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_between_complex_numbers_l315_31556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_distance_P_to_midpoint_l315_31566

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-2 - 3*t, 2 - 4*t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (y - 2)^2 - x^2 = 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    A = line_l t₁ ∧ B = line_l t₂ ∧
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    t₁ ≠ t₂

-- Define point P
noncomputable def point_P : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos (3 * Real.pi / 4), 2 * Real.sqrt 2 * Real.sin (3 * Real.pi / 4))

-- Theorem for the length of AB
theorem length_AB (A B : ℝ × ℝ) (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 * Real.sqrt 71 / 7 := by
  sorry

-- Theorem for the distance from P to the midpoint of AB
theorem distance_P_to_midpoint (A B : ℝ × ℝ) (h : intersection_points A B) :
  Real.sqrt ((point_P.1 - (A.1 + B.1)/2)^2 + (point_P.2 - (A.2 + B.2)/2)^2) = 30/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_distance_P_to_midpoint_l315_31566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l315_31522

theorem sin_cos_equation_solution :
  ∃ x ∈ ({10 * Real.pi / 180, 20 * Real.pi / 180, 50 * Real.pi / 180, 70 * Real.pi / 180} : Set ℝ),
    (Real.sin (4 * x) * Real.cos (5 * x) = -Real.cos (4 * x) * Real.sin (5 * x)) ∧
    (∀ y ∈ ({10 * Real.pi / 180, 20 * Real.pi / 180, 50 * Real.pi / 180, 70 * Real.pi / 180} : Set ℝ),
      Real.sin (4 * y) * Real.cos (5 * y) = -Real.cos (4 * y) * Real.sin (5 * y) → y = x) ∧
    x = 20 * Real.pi / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l315_31522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l315_31582

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- Checks if a point (x, y) lies on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_theorem (e : Ellipse) 
    (h_point : e.contains 2 (Real.sqrt 2))
    (h_ecc : e.eccentricity = Real.sqrt 2 / 2) :
  ∃ (m : ℝ),
    (e.a^2 = 8 ∧ e.b^2 = 4) ∧
    (∀ (y : ℝ), y ∈ Set.Icc (-(Real.sqrt 2 / 2)) (Real.sqrt 2 / 2) ↔ 
      ∃ (k : ℝ), y = -2 / ((1 / k) + 2 * k) ∧ k ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l315_31582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_or_line_segment_l315_31586

noncomputable def F₁ : ℝ × ℝ := (2, 0)
noncomputable def F₂ : ℝ × ℝ := (-2, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def is_on_trajectory (P : ℝ × ℝ) (a : ℝ) : Prop :=
  distance P F₁ + distance P F₂ = 4*a + 1/a ∧ a > 0

theorem trajectory_is_ellipse_or_line_segment :
  ∀ P : ℝ × ℝ, ∀ a : ℝ, 
    is_on_trajectory P a → 
    (∃ c : ℝ, c > 2 ∧ distance P F₁ + distance P F₂ = c) ∨
    (distance P F₁ + distance P F₂ = 4) :=
by
  sorry

#check trajectory_is_ellipse_or_line_segment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_or_line_segment_l315_31586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_compound_equals_one_implies_x_inverse_cube_root_l315_31518

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the theorem
theorem log_compound_equals_one_implies_x_inverse_cube_root (x : ℝ) :
  log 5 (log 4 (log 3 x)) = 1 →
  x^(-(1/3 : ℝ)) = 1 / (3^(341 + 1/3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_compound_equals_one_implies_x_inverse_cube_root_l315_31518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slice_properties_l315_31537

/-- Represents a circular pizza -/
structure Pizza where
  thickness : ℝ
  diameter : ℝ
  num_slices : ℕ

/-- Calculates the volume of one slice of pizza -/
noncomputable def slice_volume (p : Pizza) : ℝ :=
  (p.thickness * p.diameter^2 * Real.pi) / (4 * p.num_slices)

/-- Calculates the curved surface area along the crust of one slice -/
noncomputable def slice_crust_area (p : Pizza) : ℝ :=
  (p.thickness * p.diameter * Real.pi) / (2 * p.num_slices)

/-- Theorem stating the volume and curved surface area of a pizza slice -/
theorem pizza_slice_properties (p : Pizza) 
  (h_thickness : p.thickness = 1/2)
  (h_diameter : p.diameter = 10)
  (h_num_slices : p.num_slices = 12) :
  slice_volume p = 25*Real.pi/24 ∧ slice_crust_area p = 5*Real.pi/12 := by
  sorry

#check pizza_slice_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slice_properties_l315_31537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_satisfying_inequality_l315_31568

theorem count_pairs_satisfying_inequality : 
  let pairs := Finset.filter (λ (p : ℕ × ℕ) => 
    let (a, b) := p
    1 ≤ a ∧ a ≤ 10 ∧ 1 ≤ b ∧ b ≤ 10 ∧ (1/4 : ℚ) < (b : ℚ)/(a : ℚ) ∧ (b : ℚ)/(a : ℚ) < (1/3 : ℚ))
    (Finset.product (Finset.range 10) (Finset.range 10))
  Finset.card pairs = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_satisfying_inequality_l315_31568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l315_31504

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2

-- State the theorem
theorem function_properties :
  -- Part 1: Tangent line equation
  (∃ b : ℝ, ∀ x : ℝ, (deriv f 1) * (x - 1) + f 1 = (Real.exp 1 - 2) * x + 1) ∧
  -- Part 2: Maximum value on [0,1]
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → f x ≤ Real.exp 1 - 1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ f x = Real.exp 1 - 1) ∧
  -- Part 3: Inequality for x > 0
  (∀ x : ℝ, x > 0 → Real.exp x + (1 - Real.exp 1) * x - x * Real.log x - 1 ≥ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l315_31504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_sum_properties_arithmetic_sequence_property_l315_31561

def sequence_a (n : ℕ) : ℚ :=
  (1 / 3) ^ n

def sum_S (n : ℕ) : ℚ :=
  (1 / 2) * (1 - (1 / 3) ^ n)

theorem sequence_and_sum_properties :
  ∀ (n : ℕ), n > 0 →
    (sequence_a 1 = 1 / 3) ∧
    (sum_S (n + 1) - sum_S n = (1 / 3) ^ (n + 1)) ∧
    (sequence_a n = (1 / 3) ^ n) ∧
    (sum_S n = (1 / 2) * (1 - (1 / 3) ^ n)) :=
by sorry

def arithmetic_progression (n : ℕ) : ℕ → ℚ := sorry

theorem arithmetic_sequence_property :
  ∃ (t : ℚ),
    (sum_S 1, t * (sum_S 1 + sum_S 2), 3 * (sum_S 2 + sum_S 3)) = 
    (arithmetic_progression 3 0, arithmetic_progression 3 1, arithmetic_progression 3 2) ∧ 
    t = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_sum_properties_arithmetic_sequence_property_l315_31561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l315_31516

/-- Given a hyperbola and an ellipse with equations involving parameters m and n,
    this theorem proves that the eccentricity of the ellipse is √6/3 when the
    eccentricity of the hyperbola is 2. -/
theorem ellipse_eccentricity (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_hyp_ecc : Real.sqrt ((1/m + 1/n) / (1/m)) = 2) : 
  Real.sqrt (1 - n/m) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l315_31516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_is_45_l315_31575

/-- Represents the walking scenario with two speeds -/
structure WalkingScenario where
  slow_speed : ℚ
  fast_speed : ℚ
  extra_distance : ℚ

/-- The actual distance traveled given a walking scenario -/
noncomputable def actual_distance (scenario : WalkingScenario) : ℚ :=
  scenario.slow_speed * (scenario.extra_distance / (scenario.fast_speed - scenario.slow_speed))

/-- Theorem stating that the actual distance traveled is 45 km for the given scenario -/
theorem actual_distance_is_45 :
  let scenario : WalkingScenario := {
    slow_speed := 15,
    fast_speed := 30,
    extra_distance := 45
  }
  actual_distance scenario = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_is_45_l315_31575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_no_solutions_one_solution_two_solutions_l315_31553

noncomputable def solution_count (k : ℝ) : ℕ :=
  if k < -5/4 ∨ (-1 < k ∧ k < 1) ∨ k > 5/4 then 0
  else if k = -5/4 ∨ k = 5/4 then 1
  else if (-5/4 < k ∧ k ≤ -1) ∨ (1 ≤ k ∧ k < 5/4) then 2
  else 0

theorem equation_solutions (k : ℝ) :
  (∃ x : ℝ, |x^2 - 1| = x + k) ↔ solution_count k > 0 := by sorry

theorem no_solutions (k : ℝ) :
  (k < -5/4 ∨ (-1 < k ∧ k < 1) ∨ k > 5/4) → solution_count k = 0 := by sorry

theorem one_solution (k : ℝ) :
  (k = -5/4 ∨ k = 5/4) → solution_count k = 1 := by sorry

theorem two_solutions (k : ℝ) :
  ((-5/4 < k ∧ k ≤ -1) ∨ (1 ≤ k ∧ k < 5/4)) → solution_count k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_no_solutions_one_solution_two_solutions_l315_31553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_cos_sin_identity_l315_31526

theorem tan_alpha_cos_sin_identity (α : Real) (h : Real.tan α = 3/4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_cos_sin_identity_l315_31526
