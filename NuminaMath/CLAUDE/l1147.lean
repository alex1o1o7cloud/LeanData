import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inequality_bound_bound_is_tight_l1147_114724

theorem triangle_inequality_bound (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) (h_cond : a ≥ (b + c) / 3) :
  (a * c + b * c - c^2) / (a^2 + b^2 + 3 * c^2 + 2 * a * b - 4 * b * c) ≤ (2 * Real.sqrt 2 + 1) / 7 :=
sorry

theorem bound_is_tight :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧
  a ≥ (b + c) / 3 ∧
  (a * c + b * c - c^2) / (a^2 + b^2 + 3 * c^2 + 2 * a * b - 4 * b * c) = (2 * Real.sqrt 2 + 1) / 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_bound_bound_is_tight_l1147_114724


namespace NUMINAMATH_CALUDE_range_of_m_l1147_114775

/-- Represents an ellipse with foci on the x-axis -/
def is_ellipse_x_axis (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / m + y^2 / (6 - m) = 1 ∧ m > 6 - m ∧ m > 0

/-- Represents a hyperbola with given eccentricity range -/
def is_hyperbola_with_eccentricity (m : ℝ) : Prop :=
  ∃ x y e : ℝ, y^2 / 5 - x^2 / m = 1 ∧ 
    e^2 = 1 + m / 5 ∧ 
    Real.sqrt 6 / 2 < e ∧ e < Real.sqrt 2 ∧
    m > 0

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) : 
  (is_ellipse_x_axis m ∨ is_hyperbola_with_eccentricity m) ∧ 
  ¬(is_ellipse_x_axis m ∧ is_hyperbola_with_eccentricity m) →
  (5/2 < m ∧ m ≤ 3) ∨ (5 ≤ m ∧ m < 6) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1147_114775


namespace NUMINAMATH_CALUDE_additional_stickers_needed_l1147_114786

def current_stickers : ℕ := 35
def row_size : ℕ := 8

theorem additional_stickers_needed :
  let next_multiple := (current_stickers + row_size - 1) / row_size * row_size
  next_multiple - current_stickers = 5 := by sorry

end NUMINAMATH_CALUDE_additional_stickers_needed_l1147_114786


namespace NUMINAMATH_CALUDE_square_area_calculation_l1147_114779

theorem square_area_calculation (s r l : ℝ) : 
  l = (2/5) * r →
  r = s →
  l * 10 = 200 →
  s^2 = 2500 := by sorry

end NUMINAMATH_CALUDE_square_area_calculation_l1147_114779


namespace NUMINAMATH_CALUDE_product_63_57_l1147_114700

theorem product_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_CALUDE_product_63_57_l1147_114700


namespace NUMINAMATH_CALUDE_circle_tangency_l1147_114776

/-- Two circles are tangent internally if the distance between their centers
    is equal to the absolute difference of their radii -/
def are_tangent_internally (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (r1 - r2)^2

/-- The statement of the problem -/
theorem circle_tangency (m : ℝ) : 
  are_tangent_internally (m, -2) (-1, m) 3 2 ↔ m = -2 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_l1147_114776


namespace NUMINAMATH_CALUDE_parabola_properties_l1147_114701

def parabola (x : ℝ) : ℝ := -3 * x^2

theorem parabola_properties :
  (∀ x : ℝ, parabola x ≤ parabola 0) ∧
  (parabola 0 = 0) ∧
  (∀ x y : ℝ, x > 0 → y > x → parabola y < parabola x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1147_114701


namespace NUMINAMATH_CALUDE_triangle_squares_l1147_114706

theorem triangle_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let petya_square := a * b / (a + b)
  let vasya_square := a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)
  -- Petya's square is larger than Vasya's square
  petya_square > vasya_square ∧ 
  -- Petya's square formula is correct
  (∃ (x : ℝ), x = petya_square ∧ x * (a + b) = a * b) ∧
  -- Vasya's square formula is correct
  (∃ (y : ℝ), y = vasya_square ∧ 
    y * (a^2 / b + b + a) = Real.sqrt (a^2 + b^2) * a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_squares_l1147_114706


namespace NUMINAMATH_CALUDE_sign_of_b_is_negative_l1147_114736

/-- Given that exactly two of a+b, a-b, ab, a/b are positive and the other two are negative, prove that b < 0 -/
theorem sign_of_b_is_negative (a b : ℝ) 
  (h : (a + b > 0 ∧ a - b > 0 ∧ a * b < 0 ∧ a / b < 0) ∨
       (a + b > 0 ∧ a - b < 0 ∧ a * b > 0 ∧ a / b < 0) ∨
       (a + b > 0 ∧ a - b < 0 ∧ a * b < 0 ∧ a / b > 0) ∨
       (a + b < 0 ∧ a - b > 0 ∧ a * b > 0 ∧ a / b < 0) ∨
       (a + b < 0 ∧ a - b > 0 ∧ a * b < 0 ∧ a / b > 0) ∨
       (a + b < 0 ∧ a - b < 0 ∧ a * b > 0 ∧ a / b > 0))
  (h_nonzero : b ≠ 0) : b < 0 := by
  sorry


end NUMINAMATH_CALUDE_sign_of_b_is_negative_l1147_114736


namespace NUMINAMATH_CALUDE_max_value_ln_x_over_x_l1147_114708

/-- The function f(x) = ln(x) / x attains its maximum value at e^(-1) for x > 0 -/
theorem max_value_ln_x_over_x : 
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → (Real.log x) / x ≥ (Real.log y) / y ∧ (Real.log x) / x = Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_max_value_ln_x_over_x_l1147_114708


namespace NUMINAMATH_CALUDE_bowling_team_size_l1147_114705

theorem bowling_team_size (original_avg : ℝ) (new_player1_weight : ℝ) (new_player2_weight : ℝ) (new_avg : ℝ) :
  original_avg = 103 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  new_avg = 99 →
  ∃ n : ℕ, n > 0 ∧ n * original_avg + new_player1_weight + new_player2_weight = (n + 2) * new_avg ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_size_l1147_114705


namespace NUMINAMATH_CALUDE_vector_parallelism_l1147_114781

theorem vector_parallelism (x : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  (∃ (k : ℝ), k ≠ 0 ∧ (a.1 + 2 * b.1, a.2 + 2 * b.2) = k • (2 * a.1 - b.1, 2 * a.2 - b.2)) →
  x = (1 / 2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_vector_parallelism_l1147_114781


namespace NUMINAMATH_CALUDE_max_amount_received_back_l1147_114732

/-- Represents the casino chip denominations -/
inductive ChipDenomination
  | twenty
  | hundred

/-- Calculates the value of a chip -/
def chipValue : ChipDenomination → ℕ
  | ChipDenomination.twenty => 20
  | ChipDenomination.hundred => 100

/-- Represents the number of chips lost for each denomination -/
structure ChipsLost where
  twenty : ℕ
  hundred : ℕ

/-- Calculates the total value of chips lost -/
def totalLost (chips : ChipsLost) : ℕ :=
  chips.twenty * chipValue ChipDenomination.twenty +
  chips.hundred * chipValue ChipDenomination.hundred

/-- Represents the casino scenario -/
structure CasinoScenario where
  totalBought : ℕ
  chipsLost : ChipsLost

/-- Calculates the amount received back -/
def amountReceivedBack (scenario : CasinoScenario) : ℕ :=
  scenario.totalBought - totalLost scenario.chipsLost

/-- The main theorem to prove -/
theorem max_amount_received_back :
  ∀ (scenario : CasinoScenario),
    scenario.totalBought = 3000 ∧
    scenario.chipsLost.twenty + scenario.chipsLost.hundred = 13 ∧
    (scenario.chipsLost.twenty = scenario.chipsLost.hundred + 3 ∨
     scenario.chipsLost.twenty = scenario.chipsLost.hundred - 3) →
    amountReceivedBack scenario ≤ 2340 :=
by
  sorry

#check max_amount_received_back

end NUMINAMATH_CALUDE_max_amount_received_back_l1147_114732


namespace NUMINAMATH_CALUDE_parabola_sum_a_c_l1147_114702

/-- A parabola that intersects the x-axis at x = -1 -/
structure Parabola where
  a : ℝ
  c : ℝ
  intersect_at_neg_one : a * (-1)^2 + (-1) + c = 0

/-- The sum of a and c for a parabola intersecting the x-axis at x = -1 is 1 -/
theorem parabola_sum_a_c (p : Parabola) : p.a + p.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_a_c_l1147_114702


namespace NUMINAMATH_CALUDE_residue_of_negative_935_mod_24_l1147_114713

theorem residue_of_negative_935_mod_24 : 
  ∃ (r : ℤ), 0 ≤ r ∧ r < 24 ∧ -935 ≡ r [ZMOD 24] ∧ r = 1 :=
sorry

end NUMINAMATH_CALUDE_residue_of_negative_935_mod_24_l1147_114713


namespace NUMINAMATH_CALUDE_bernardo_winning_number_l1147_114726

theorem bernardo_winning_number : ∃ N : ℕ, 
  (N ≤ 1999) ∧ 
  (8 * N + 600 < 2000) ∧ 
  (8 * N + 700 ≥ 2000) ∧ 
  (∀ M : ℕ, M < N → 
    (M ≤ 1999 → 8 * M + 700 < 2000) ∨ 
    (8 * M + 600 ≥ 2000)) := by
  sorry

#eval Nat.find bernardo_winning_number

end NUMINAMATH_CALUDE_bernardo_winning_number_l1147_114726


namespace NUMINAMATH_CALUDE_coefficient_x4_sum_binomials_l1147_114743

theorem coefficient_x4_sum_binomials : 
  (Finset.sum (Finset.range 3) (fun i => Nat.choose (i + 5) 4)) = 55 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_sum_binomials_l1147_114743


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1147_114728

theorem consecutive_integers_average (a : ℤ) (b : ℚ) : 
  (a > 0) →
  (b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) / 5) →
  ((b + (b + 10) + (b + 20)) / 3 = a + 12) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1147_114728


namespace NUMINAMATH_CALUDE_line_segment_does_not_intersect_staircase_l1147_114770

/-- Represents a step in the staircase -/
structure Step where
  width : Nat
  height : Nat

/-- Represents the staircase -/
def Staircase : List Step := List.range 2019 |>.map (fun i => { width := i + 1, height := 1 })

/-- The line segment from (0,0) to (2019,2019) -/
def LineSegment : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (2019 * t, 2019 * t)}

/-- Checks if a point is on a step -/
def onStep (p : ℝ × ℝ) (s : Step) : Prop :=
  (s.width - 1 : ℝ) ≤ p.1 ∧ p.1 < s.width ∧
  (s.height - 1 : ℝ) ≤ p.2 ∧ p.2 < s.height

theorem line_segment_does_not_intersect_staircase :
  ∀ p ∈ LineSegment, ∀ s ∈ Staircase, ¬ onStep p s := by
  sorry

end NUMINAMATH_CALUDE_line_segment_does_not_intersect_staircase_l1147_114770


namespace NUMINAMATH_CALUDE_pagoda_lanterns_sum_l1147_114757

/-- Represents a pagoda with lanterns -/
structure Pagoda where
  layers : ℕ
  top_lanterns : ℕ
  total_lanterns : ℕ

/-- Calculates the number of lanterns on the bottom layer of the pagoda -/
def bottom_lanterns (p : Pagoda) : ℕ := p.top_lanterns * 2^(p.layers - 1)

/-- Calculates the sum of lanterns on all layers of the pagoda -/
def sum_lanterns (p : Pagoda) : ℕ := p.top_lanterns * (2^p.layers - 1)

/-- Theorem: For a 7-layer pagoda with lanterns doubling from top to bottom and
    a total of 381 lanterns, the sum of lanterns on the top and bottom layers is 195 -/
theorem pagoda_lanterns_sum :
  ∀ (p : Pagoda), p.layers = 7 → p.total_lanterns = 381 → sum_lanterns p = p.total_lanterns →
  p.top_lanterns + bottom_lanterns p = 195 :=
sorry

end NUMINAMATH_CALUDE_pagoda_lanterns_sum_l1147_114757


namespace NUMINAMATH_CALUDE_tetrahedron_division_ratio_l1147_114762

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a plane -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Calculates the volume of a part of a tetrahedron cut by a plane -/
def partialTetrahedronVolume (t : Tetrahedron) (p : Plane) : ℝ := sorry

/-- Checks if a point lies on a line segment between two other points -/
def isOnLineSegment (p : Point3D) (a : Point3D) (b : Point3D) : Prop := sorry

/-- Checks if a point lies on the extension of a line segment beyond a point -/
def isOnLineExtension (p : Point3D) (a : Point3D) (b : Point3D) : Prop := sorry

/-- Theorem: The plane divides the tetrahedron in the ratio 2:33 -/
theorem tetrahedron_division_ratio (ABCD : Tetrahedron) (K M N : Point3D) (p : Plane) : 
  isOnLineSegment K ABCD.A ABCD.D ∧ 
  isOnLineExtension N ABCD.A ABCD.B ∧ 
  isOnLineExtension M ABCD.A ABCD.C ∧ 
  (ABCD.A.x - K.x) / (K.x - ABCD.D.x) = 3 ∧
  (N.x - ABCD.B.x) = (ABCD.B.x - ABCD.A.x) ∧
  (M.x - ABCD.C.x) / (ABCD.C.x - ABCD.A.x) = 1/3 ∧
  (p.a * K.x + p.b * K.y + p.c * K.z + p.d = 0) ∧
  (p.a * M.x + p.b * M.y + p.c * M.z + p.d = 0) ∧
  (p.a * N.x + p.b * N.y + p.c * N.z + p.d = 0) →
  (partialTetrahedronVolume ABCD p) / (tetrahedronVolume ABCD) = 2/35 := by
sorry

end NUMINAMATH_CALUDE_tetrahedron_division_ratio_l1147_114762


namespace NUMINAMATH_CALUDE_function_composition_l1147_114798

-- Define the function f
def f : ℝ → ℝ := fun x => 2 * x + 7

-- State the theorem
theorem function_composition (x : ℝ) : 
  (fun x => f (x - 1)) = (fun x => 2 * x + 5) → f (x^2) = 2 * x^2 + 7 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l1147_114798


namespace NUMINAMATH_CALUDE_solution_set_part_i_solution_range_part_ii_l1147_114715

-- Define the functions f and g
def f (x : ℝ) := |x - 1|
def g (a x : ℝ) := 2 * |x + a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f x - g 1 x > 1} = {x : ℝ | -1 < x ∧ x < -1/3} :=
sorry

-- Part II
theorem solution_range_part_ii :
  ∀ a : ℝ, (∃ x : ℝ, 2 * f x + g a x ≤ (a + 1)^2) ↔ (a ≤ -3 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_solution_range_part_ii_l1147_114715


namespace NUMINAMATH_CALUDE_unique_three_digit_numbers_l1147_114785

/-- The number of available digits -/
def n : ℕ := 5

/-- The number of digits to be used in each number -/
def r : ℕ := 3

/-- The number of unique three-digit numbers that can be formed without repetition -/
def uniqueNumbers : ℕ := n.choose r * r.factorial

theorem unique_three_digit_numbers :
  uniqueNumbers = 60 := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_numbers_l1147_114785


namespace NUMINAMATH_CALUDE_pocket_money_calculation_l1147_114709

def fifty_cent_coins : ℕ := 6
def twenty_cent_coins : ℕ := 6
def fifty_cent_value : ℚ := 0.5
def twenty_cent_value : ℚ := 0.2

theorem pocket_money_calculation :
  (fifty_cent_coins : ℚ) * fifty_cent_value + (twenty_cent_coins : ℚ) * twenty_cent_value = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_pocket_money_calculation_l1147_114709


namespace NUMINAMATH_CALUDE_quadratic_radical_problem_l1147_114723

/-- A number is a simplest quadratic radical if it cannot be further simplified -/
def IsSimplestQuadraticRadical (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ ∀ m : ℕ, m < n → ¬∃ k : ℕ, n = k^2 * m

/-- Two quadratic radicals are of the same type if their radicands have the same squarefree part -/
def SameTypeRadical (x y : ℝ) : Prop :=
  ∃ a b : ℕ, x = Real.sqrt a ∧ y = Real.sqrt b ∧ ∃ k m n : ℕ, k ≠ 0 ∧ m.Coprime n ∧ a = k * m ∧ b = k * n

theorem quadratic_radical_problem (a : ℝ) :
  IsSimplestQuadraticRadical (Real.sqrt (2 * a + 1)) →
  SameTypeRadical (Real.sqrt (2 * a + 1)) (Real.sqrt 48) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_problem_l1147_114723


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1147_114733

theorem regular_polygon_sides (interior_angle : ℝ) : 
  interior_angle = 140 → (360 / (180 - interior_angle) : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1147_114733


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1147_114787

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + (3*x - 5) + 2*x + 18 + (2*x + 6)) / 5 = 30 → x = 15.125 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1147_114787


namespace NUMINAMATH_CALUDE_leahs_coins_value_l1147_114753

/-- Represents the number of coins of each type --/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value of coins in cents --/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Theorem stating that Leah's coins are worth 88 cents --/
theorem leahs_coins_value :
  ∃ (coins : CoinCount),
    coins.pennies + coins.nickels + coins.dimes = 20 ∧
    coins.pennies = coins.nickels ∧
    coins.pennies = coins.dimes + 4 ∧
    totalValue coins = 88 := by
  sorry

#check leahs_coins_value

end NUMINAMATH_CALUDE_leahs_coins_value_l1147_114753


namespace NUMINAMATH_CALUDE_sequence_formulas_l1147_114707

def S (n : ℕ) : ℝ := sorry

def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 2 * S n + 1

def b (n : ℕ) : ℝ := (3 * n - 1) * a n

def T (n : ℕ) : ℝ := sorry

theorem sequence_formulas :
  (∀ n : ℕ, a n = 3^n) ∧
  (∀ n : ℕ, T n = ((3 * n / 2) - 5 / 4) * 3^n + 5 / 4) :=
sorry

end NUMINAMATH_CALUDE_sequence_formulas_l1147_114707


namespace NUMINAMATH_CALUDE_special_function_a_range_l1147_114766

/-- A function satisfying the given properties -/
structure SpecialFunction where
  f : ℝ → ℝ
  even : ∀ x, f (-x) = f x
  increasing_nonneg : ∀ x₁ x₂, 0 ≤ x₁ ∧ 0 ≤ x₂ ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

/-- The theorem statement -/
theorem special_function_a_range (f : SpecialFunction) :
  {a : ℝ | ∀ x ∈ Set.Icc (1/2 : ℝ) 1, f.f (a * x + 1) ≤ f.f (x - 2)} = Set.Icc (-2 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_a_range_l1147_114766


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1147_114704

theorem quadratic_solution_difference_squared :
  ∀ p q : ℝ,
  (5 * p^2 - 8 * p - 15 = 0) →
  (5 * q^2 - 8 * q - 15 = 0) →
  (p - q)^2 = 14.5924 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1147_114704


namespace NUMINAMATH_CALUDE_prime_divisibility_pairs_l1147_114767

theorem prime_divisibility_pairs (n p : ℕ) : 
  p.Prime → 
  n ≤ 2 * p → 
  (p - 1)^n + 1 ∣ n^(p - 1) → 
  ((n = 1 ∧ p.Prime) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_pairs_l1147_114767


namespace NUMINAMATH_CALUDE_planes_lines_parallelism_l1147_114774

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the necessary relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem planes_lines_parallelism 
  (α β : Plane) (m n : Line) 
  (h_not_coincident : α ≠ β)
  (h_different_lines : m ≠ n)
  (h_parallel_planes : parallel α β)
  (h_n_perp_α : perpendicular n α)
  (h_m_perp_β : perpendicular m β) :
  line_parallel m n :=
sorry

end NUMINAMATH_CALUDE_planes_lines_parallelism_l1147_114774


namespace NUMINAMATH_CALUDE_cosine_equality_l1147_114759

theorem cosine_equality (n : ℤ) : 0 ≤ n ∧ n ≤ 180 → n = 38 → Real.cos (n * π / 180) = Real.cos (758 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l1147_114759


namespace NUMINAMATH_CALUDE_intersection_length_l1147_114784

/-- The length of the circular intersection between a sphere and a plane -/
theorem intersection_length (x y z : ℝ) : 
  x + y + z = 8 → 
  x * y + y * z + x * z = 14 → 
  (2 * Real.pi : ℝ) * (2 * Real.sqrt (11 / 3)) = 4 * Real.pi * Real.sqrt (11 / 3) := by
  sorry

#check intersection_length

end NUMINAMATH_CALUDE_intersection_length_l1147_114784


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l1147_114717

theorem monomial_sum_condition (a b : ℕ) (m n : ℕ) : 
  (∃ k : ℕ, 2 * a^(m+2) * b^(2*n+2) + a^3 * b^8 = k * a^(m+2) * b^(2*n+2)) → 
  m = 1 ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l1147_114717


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1147_114778

def inequality (x : ℝ) := x^2 - 3*x - 10 > 0

theorem inequality_solution_set :
  {x : ℝ | inequality x} = {x : ℝ | x > 5 ∨ x < -2} :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1147_114778


namespace NUMINAMATH_CALUDE_uncovered_fraction_of_plates_l1147_114727

/-- The fraction of a circular plate with diameter 12 inches that is not covered
    by a smaller circular plate with diameter 10 inches placed on top of it is 11/36. -/
theorem uncovered_fraction_of_plates (small_diameter large_diameter : ℝ) 
  (h_small : small_diameter = 10)
  (h_large : large_diameter = 12) :
  (large_diameter^2 - small_diameter^2) / large_diameter^2 = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_fraction_of_plates_l1147_114727


namespace NUMINAMATH_CALUDE_softball_team_ratio_l1147_114711

/-- Represents a co-ed softball team --/
structure Team where
  men : ℕ
  women : ℕ
  total : ℕ
  h1 : women = men + 4
  h2 : men + women = total

/-- The ratio of men to women in a team --/
def menWomenRatio (t : Team) : Rat :=
  t.men / t.women

theorem softball_team_ratio (t : Team) (h : t.total = 14) :
  menWomenRatio t = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l1147_114711


namespace NUMINAMATH_CALUDE_not_prime_sum_l1147_114739

theorem not_prime_sum (a b c d : ℕ+) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_int : ∃ (n : ℤ), (a / (a + b) : ℚ) + (b / (b + c) : ℚ) + (c / (c + d) : ℚ) + (d / (d + a) : ℚ) = n) : 
  ¬ Nat.Prime (a + b + c + d) := by
sorry

end NUMINAMATH_CALUDE_not_prime_sum_l1147_114739


namespace NUMINAMATH_CALUDE_parallel_lines_planes_l1147_114783

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the "not contained in" relation for a line and a plane
variable (not_contained_in : Line → Plane → Prop)

-- State the theorem
theorem parallel_lines_planes 
  (l m : Line) 
  (α β : Plane) 
  (h_distinct_lines : l ≠ m)
  (h_distinct_planes : α ≠ β)
  (h_alpha_beta_parallel : parallel_plane α β)
  (h_l_alpha_parallel : parallel_line_plane l α)
  (h_l_m_parallel : parallel l m)
  (h_m_not_in_beta : not_contained_in m β) :
  parallel_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_planes_l1147_114783


namespace NUMINAMATH_CALUDE_complement_of_A_l1147_114735

def U : Set Nat := {1, 2, 3}
def A : Set Nat := {1, 3}

theorem complement_of_A : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1147_114735


namespace NUMINAMATH_CALUDE_equal_coverings_l1147_114790

/-- Represents a 1993 x 1993 grid -/
def Grid := Fin 1993 × Fin 1993

/-- Represents a 1 x 2 rectangle -/
def Rectangle := Set (Fin 1993 × Fin 1993)

/-- Predicate to check if two squares are on the same edge of the grid -/
def on_same_edge (a b : Grid) : Prop :=
  (a.1 = b.1 ∧ (a.2 = 0 ∨ a.2 = 1992)) ∨
  (a.2 = b.2 ∧ (a.1 = 0 ∨ a.1 = 1992))

/-- Predicate to check if there's an odd number of squares between two squares -/
def odd_squares_between (a b : Grid) : Prop :=
  ∃ n : Nat, n % 2 = 1 ∧
  ((a.1 = b.1 ∧ abs (a.2 - b.2) = n + 1) ∨
   (a.2 = b.2 ∧ abs (a.1 - b.1) = n + 1))

/-- Type representing a covering of the grid with 1 x 2 rectangles -/
def Covering := Set Rectangle

/-- Predicate to check if a covering is valid (covers the entire grid except one square) -/
def valid_covering (c : Covering) (uncovered : Grid) : Prop := sorry

/-- The number of valid coverings that leave a given square uncovered -/
def num_coverings (uncovered : Grid) : Nat := sorry

theorem equal_coverings (A B : Grid)
  (h1 : on_same_edge A B)
  (h2 : odd_squares_between A B) :
  num_coverings A = num_coverings B := by sorry

end NUMINAMATH_CALUDE_equal_coverings_l1147_114790


namespace NUMINAMATH_CALUDE_complex_product_QED_l1147_114788

theorem complex_product_QED (Q E D : ℂ) : 
  Q = 7 + 3 * Complex.I → 
  E = 2 * Complex.I → 
  D = 7 - 3 * Complex.I → 
  Q * E * D = 116 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_QED_l1147_114788


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l1147_114760

theorem sqrt_sum_comparison : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l1147_114760


namespace NUMINAMATH_CALUDE_product_expansion_sum_l1147_114745

theorem product_expansion_sum (a b c d : ℤ) : 
  (∀ x, (5 * x^2 - 8 * x + 3) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  10 * a + 5 * b + 2 * c + d = 60 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l1147_114745


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l1147_114714

/-- Represents the stratified sampling problem -/
structure StratifiedSample where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  sample_size : ℕ
  interview_size : ℕ

/-- Calculates the number of male students in the sample -/
def male_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.male_students) / s.total_students

/-- Calculates the number of female students in the sample -/
def female_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.female_students) / s.total_students

/-- Calculates the probability of selecting exactly one female student for interview -/
def prob_one_female (s : StratifiedSample) : ℚ :=
  let male_count := male_in_sample s
  let female_count := female_in_sample s
  (male_count * female_count : ℚ) / ((s.sample_size * (s.sample_size - 1)) / 2 : ℚ)

/-- The main theorem to be proved -/
theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_students = 50)
  (h2 : s.male_students = 30)
  (h3 : s.female_students = 20)
  (h4 : s.sample_size = 5)
  (h5 : s.interview_size = 2) :
  male_in_sample s = 3 ∧ 
  female_in_sample s = 2 ∧ 
  prob_one_female s = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l1147_114714


namespace NUMINAMATH_CALUDE_range_f_a2_values_of_a_min_3_l1147_114792

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

-- Part 1: Range of f(x) when a = 2 in [1, 2]
theorem range_f_a2 :
  ∀ y ∈ Set.Icc (-2) 2, ∃ x ∈ Set.Icc 1 2, f 2 x = y :=
sorry

-- Part 2: Values of a when minimum of f(x) in [0, 2] is 3
theorem values_of_a_min_3 :
  (∀ x ∈ Set.Icc 0 2, f a x ≥ 3) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 3) →
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_range_f_a2_values_of_a_min_3_l1147_114792


namespace NUMINAMATH_CALUDE_total_daisies_l1147_114731

/-- Calculates the total number of daisies in Jack's flower crowns --/
theorem total_daisies (white pink red : ℕ) : 
  white = 6 ∧ 
  pink = 9 * white ∧ 
  red = 4 * pink - 3 → 
  white + pink + red = 273 := by
sorry


end NUMINAMATH_CALUDE_total_daisies_l1147_114731


namespace NUMINAMATH_CALUDE_a_1_value_l1147_114712

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem a_1_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 1 = -8 := by
sorry

end NUMINAMATH_CALUDE_a_1_value_l1147_114712


namespace NUMINAMATH_CALUDE_cube_difference_l1147_114793

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 59) : 
  a^3 - b^3 = 448 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l1147_114793


namespace NUMINAMATH_CALUDE_absolute_value_square_equivalence_l1147_114718

theorem absolute_value_square_equivalence (m n : ℝ) :
  (|m| > |n| → m^2 > n^2) ∧
  (m^2 > n^2 → |m| > |n|) ∧
  (|m| ≤ |n| → m^2 ≤ n^2) ∧
  (m^2 ≤ n^2 → |m| ≤ |n|) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_equivalence_l1147_114718


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1147_114741

theorem min_value_of_expression (x y : ℝ) (h : 2 * x - y = 4) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (a b : ℝ), 2 * a - b = 4 → 4^a + (1/2)^b ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1147_114741


namespace NUMINAMATH_CALUDE_range_of_f_l1147_114797

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the domain of x
def domain : Set ℝ := Set.Icc 0 2

-- Theorem statement
theorem range_of_f :
  Set.range (fun x => f x) = Set.Icc 0 4 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1147_114797


namespace NUMINAMATH_CALUDE_binary_1101001_plus_14_equals_119_l1147_114744

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 1101001₂ -/
def binary_1101001 : List Bool := [true, false, false, true, false, true, true]

theorem binary_1101001_plus_14_equals_119 :
  binary_to_decimal binary_1101001 + 14 = 119 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101001_plus_14_equals_119_l1147_114744


namespace NUMINAMATH_CALUDE_prob_red_tile_value_l1147_114794

/-- The number of integers from 1 to 100 that are congruent to 3 mod 7 -/
def red_tiles : ℕ := (Finset.filter (fun n => n % 7 = 3) (Finset.range 100)).card

/-- The total number of tiles -/
def total_tiles : ℕ := 100

/-- The probability of selecting a red tile -/
def prob_red_tile : ℚ := red_tiles / total_tiles

theorem prob_red_tile_value :
  prob_red_tile = 7 / 50 := by sorry

end NUMINAMATH_CALUDE_prob_red_tile_value_l1147_114794


namespace NUMINAMATH_CALUDE_lopez_seating_theorem_l1147_114729

/-- Represents the number of family members -/
def family_members : ℕ := 5

/-- Represents the number of front seats in the car -/
def front_seats : ℕ := 2

/-- Represents the number of back seats in the car -/
def back_seats : ℕ := 3

/-- Represents the number of possible drivers (Mr. or Mrs. Lopez) -/
def possible_drivers : ℕ := 2

/-- Calculates the number of possible seating arrangements for the Lopez family -/
def seating_arrangements : ℕ :=
  possible_drivers * (family_members - 1) * Nat.factorial (family_members - 2)

theorem lopez_seating_theorem :
  seating_arrangements = 48 :=
sorry

end NUMINAMATH_CALUDE_lopez_seating_theorem_l1147_114729


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_is_1620_l1147_114749

def arithmeticSeriesSum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_is_1620 :
  arithmeticSeriesSum 10 30 (1/4) = 1620 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_is_1620_l1147_114749


namespace NUMINAMATH_CALUDE_louisa_travel_problem_l1147_114754

/-- Louisa's vacation travel problem -/
theorem louisa_travel_problem (speed : ℝ) (second_day_distance : ℝ) (time_difference : ℝ) :
  speed = 60 →
  second_day_distance = 420 →
  time_difference = 3 →
  ∃ (first_day_distance : ℝ),
    first_day_distance = speed * (second_day_distance / speed - time_difference) ∧
    first_day_distance = 240 :=
by sorry

end NUMINAMATH_CALUDE_louisa_travel_problem_l1147_114754


namespace NUMINAMATH_CALUDE_f_increasing_l1147_114799

def f (x : ℝ) := 2 * x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_l1147_114799


namespace NUMINAMATH_CALUDE_factors_of_M_l1147_114771

/-- The number of natural-number factors of M, where M = 2^4 · 3^3 · 7^1 -/
def num_factors (M : ℕ) : ℕ :=
  (5 : ℕ) * (4 : ℕ) * (2 : ℕ)

/-- M is defined as 2^4 · 3^3 · 7^1 -/
def M : ℕ := 2^4 * 3^3 * 7^1

theorem factors_of_M :
  num_factors M = 40 :=
by sorry

end NUMINAMATH_CALUDE_factors_of_M_l1147_114771


namespace NUMINAMATH_CALUDE_frog_final_position_l1147_114769

def frog_jumps (n : ℕ) : ℕ := n * (n + 1) / 2

theorem frog_final_position :
  ∀ (total_positions : ℕ) (num_jumps : ℕ),
    total_positions = 6 →
    num_jumps = 20 →
    frog_jumps num_jumps % total_positions = 1 := by
  sorry

end NUMINAMATH_CALUDE_frog_final_position_l1147_114769


namespace NUMINAMATH_CALUDE_cookie_price_is_three_l1147_114747

/-- The price of each cookie in Zane's purchase --/
def cookie_price : ℚ := 3

/-- The total number of items (Oreos and cookies) --/
def total_items : ℕ := 65

/-- The ratio of Oreos to cookies --/
def oreo_cookie_ratio : ℚ := 4 / 9

/-- The price of each Oreo --/
def oreo_price : ℚ := 2

/-- The difference in total spent on cookies vs Oreos --/
def cookie_oreo_diff : ℚ := 95

theorem cookie_price_is_three :
  let num_cookies : ℚ := total_items / (1 + oreo_cookie_ratio)
  let num_oreos : ℚ := total_items - num_cookies
  let total_oreo_cost : ℚ := num_oreos * oreo_price
  let total_cookie_cost : ℚ := total_oreo_cost + cookie_oreo_diff
  cookie_price = total_cookie_cost / num_cookies :=
by sorry

end NUMINAMATH_CALUDE_cookie_price_is_three_l1147_114747


namespace NUMINAMATH_CALUDE_trip_time_difference_l1147_114748

def speed : ℝ := 40
def distance1 : ℝ := 360
def distance2 : ℝ := 400

theorem trip_time_difference : 
  (distance2 - distance1) / speed * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l1147_114748


namespace NUMINAMATH_CALUDE_function_inequality_l1147_114752

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * deriv f x ≥ 0) : f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1147_114752


namespace NUMINAMATH_CALUDE_circle_equation_and_intersection_range_l1147_114777

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*Real.sqrt 3*y = 0

-- Define the line l
def line_l (t x y : ℝ) : Prop :=
  x = -1 - (Real.sqrt 3 / 2) * t ∧ y = Real.sqrt 3 + (1 / 2) * t

-- Define the intersection point P
def intersection_point (x y : ℝ) : Prop :=
  ∃ t : ℝ, line_l t x y ∧ circle_C x y

theorem circle_equation_and_intersection_range :
  (∀ ρ θ : ℝ, ρ = 4 * Real.sin (θ - Real.pi / 6) → 
    ∃ x y : ℝ, ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y ∧ circle_C x y) ∧
  (∀ x y : ℝ, intersection_point x y → 
    -2 ≤ Real.sqrt 3 * x + y ∧ Real.sqrt 3 * x + y ≤ 2) := by sorry

end NUMINAMATH_CALUDE_circle_equation_and_intersection_range_l1147_114777


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1147_114773

def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 2016) + y^2 / (k - 2018) = 1

def asymptote_equation (x y : ℝ) : Prop :=
  x + y = 0 ∨ x - y = 0

theorem hyperbola_asymptotes (k : ℤ) :
  (∃ x y : ℝ, hyperbola_equation x y (k : ℝ)) →
  (∀ x y : ℝ, hyperbola_equation x y (k : ℝ) → asymptote_equation x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1147_114773


namespace NUMINAMATH_CALUDE_box_side_face_area_l1147_114763

theorem box_side_face_area (L W H : ℝ) 
  (h1 : W * H = (1/2) * (L * W))
  (h2 : L * W = 1.5 * (H * L))
  (h3 : L * W * H = 648) :
  H * L = 72 := by
  sorry

end NUMINAMATH_CALUDE_box_side_face_area_l1147_114763


namespace NUMINAMATH_CALUDE_cubic_equation_root_range_l1147_114716

theorem cubic_equation_root_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x^3 - 3*x - m = 0) ↔ m ∈ Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_range_l1147_114716


namespace NUMINAMATH_CALUDE_president_savings_l1147_114751

theorem president_savings (total_funds : ℝ) (friends_percentage : ℝ) (family_percentage : ℝ) : 
  total_funds = 10000 →
  friends_percentage = 40 →
  family_percentage = 30 →
  let friends_contribution := (friends_percentage / 100) * total_funds
  let remaining_after_friends := total_funds - friends_contribution
  let family_contribution := (family_percentage / 100) * remaining_after_friends
  let president_savings := remaining_after_friends - family_contribution
  president_savings = 4200 := by
sorry

end NUMINAMATH_CALUDE_president_savings_l1147_114751


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angles_divisible_by_nine_l1147_114758

theorem regular_polygon_interior_angles_divisible_by_nine :
  (∃ (S : Finset ℕ), S.card = 5 ∧
    (∀ n ∈ S, 3 ≤ n ∧ n ≤ 15 ∧ (180 - 360 / n) % 9 = 0) ∧
    (∀ n, 3 ≤ n → n ≤ 15 → (180 - 360 / n) % 9 = 0 → n ∈ S)) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angles_divisible_by_nine_l1147_114758


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l1147_114755

/-- Represents the time (in hours) when two cyclists A and B are 32.5 km apart -/
def time_when_apart (initial_distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (final_distance : ℝ) : Set ℝ :=
  {t : ℝ | t * (speed_A + speed_B) = initial_distance - final_distance ∨ 
           t * (speed_A + speed_B) = initial_distance + final_distance}

/-- Theorem stating that the time when cyclists A and B are 32.5 km apart is either 1 or 3 hours -/
theorem cyclists_meeting_time :
  time_when_apart 65 17.5 15 32.5 = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l1147_114755


namespace NUMINAMATH_CALUDE_better_value_is_16_cents_per_ounce_l1147_114721

/-- Represents a box of macaroni and cheese -/
structure MacaroniBox where
  weight : ℕ  -- weight in ounces
  price : ℕ   -- price in cents

/-- Calculates the price per ounce for a given box -/
def pricePerOunce (box : MacaroniBox) : ℚ :=
  box.price / box.weight

/-- Finds the box with the lowest price per ounce -/
def bestValue (box1 box2 : MacaroniBox) : MacaroniBox :=
  if pricePerOunce box1 ≤ pricePerOunce box2 then box1 else box2

theorem better_value_is_16_cents_per_ounce :
  let largerBox : MacaroniBox := ⟨30, 480⟩
  let smallerBox : MacaroniBox := ⟨20, 340⟩
  pricePerOunce (bestValue largerBox smallerBox) = 16 / 1 := by
  sorry

end NUMINAMATH_CALUDE_better_value_is_16_cents_per_ounce_l1147_114721


namespace NUMINAMATH_CALUDE_fixed_point_of_parabola_l1147_114734

theorem fixed_point_of_parabola (s : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + s * x - 3 * s
  f 3 = 36 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabola_l1147_114734


namespace NUMINAMATH_CALUDE_evaluate_expression_l1147_114746

theorem evaluate_expression : 3000 * (3000^1500 + 3000^1500) = 2 * 3000^1501 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1147_114746


namespace NUMINAMATH_CALUDE_trig_identity_l1147_114710

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 1 / Real.sin (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1147_114710


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l1147_114722

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet, rounding down -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem sculpture_and_base_height :
  let sculpture_height := feet_inches_to_inches 2 10
  let base_height := 2
  let total_height := sculpture_height + base_height
  inches_to_feet total_height = 3 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l1147_114722


namespace NUMINAMATH_CALUDE_remaining_staff_count_l1147_114730

/-- Calculates the remaining staff in a cafe after some leave --/
theorem remaining_staff_count 
  (initial_chefs initial_waiters initial_busboys initial_hostesses : ℕ)
  (leaving_chefs leaving_waiters leaving_busboys leaving_hostesses : ℕ)
  (h1 : initial_chefs = 16)
  (h2 : initial_waiters = 16)
  (h3 : initial_busboys = 10)
  (h4 : initial_hostesses = 5)
  (h5 : leaving_chefs = 6)
  (h6 : leaving_waiters = 3)
  (h7 : leaving_busboys = 4)
  (h8 : leaving_hostesses = 2) :
  (initial_chefs - leaving_chefs) + (initial_waiters - leaving_waiters) + 
  (initial_busboys - leaving_busboys) + (initial_hostesses - leaving_hostesses) = 32 := by
  sorry

end NUMINAMATH_CALUDE_remaining_staff_count_l1147_114730


namespace NUMINAMATH_CALUDE_integral_equation_solution_l1147_114780

theorem integral_equation_solution (k : ℝ) : 
  (∫ x in (0:ℝ)..1, x - k) = (3/2 : ℝ) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_integral_equation_solution_l1147_114780


namespace NUMINAMATH_CALUDE_min_value_of_function_l1147_114719

theorem min_value_of_function (t : ℝ) (h : t > 0) :
  (t^2 - 4*t + 1) / t ≥ -2 ∧ 
  ∀ ε > 0, ∃ t₀ > 0, (t₀^2 - 4*t₀ + 1) / t₀ < -2 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1147_114719


namespace NUMINAMATH_CALUDE_no_three_squares_l1147_114764

theorem no_three_squares (x : ℤ) : ¬(∃ (a b c : ℤ), (2*x - 1 = a^2) ∧ (5*x - 1 = b^2) ∧ (13*x - 1 = c^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_three_squares_l1147_114764


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1147_114765

/-- The number of players in the chess tournament -/
def num_players : ℕ := 7

/-- The total number of games played in the tournament -/
def total_games : ℕ := 42

/-- The number of times each player plays against each opponent -/
def games_per_pair : ℕ := 2

theorem chess_tournament_games :
  (num_players * (num_players - 1) * games_per_pair) / 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1147_114765


namespace NUMINAMATH_CALUDE_pizza_toppings_l1147_114789

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) :
  total_slices = 12 →
  pepperoni_slices = 6 →
  mushroom_slices = 10 →
  pepperoni_slices + mushroom_slices ≥ total_slices →
  ∃ (both_toppings : ℕ),
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 4 :=
by
  sorry

#check pizza_toppings

end NUMINAMATH_CALUDE_pizza_toppings_l1147_114789


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l1147_114742

theorem infinite_solutions_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l1147_114742


namespace NUMINAMATH_CALUDE_on_time_probability_difference_l1147_114795

theorem on_time_probability_difference 
  (p_plane : ℝ) 
  (p_train : ℝ) 
  (p_plane_on_time : ℝ) 
  (p_train_on_time : ℝ)
  (h_p_plane : p_plane = 0.7)
  (h_p_train : p_train = 0.3)
  (h_p_plane_on_time : p_plane_on_time = 0.8)
  (h_p_train_on_time : p_train_on_time = 0.9)
  (h_sum_prob : p_plane + p_train = 1) :
  let p_on_time := p_plane * p_plane_on_time + p_train * p_train_on_time
  let p_plane_given_on_time := (p_plane * p_plane_on_time) / p_on_time
  let p_train_given_on_time := (p_train * p_train_on_time) / p_on_time
  p_plane_given_on_time - p_train_given_on_time = 29 / 83 := by
sorry

end NUMINAMATH_CALUDE_on_time_probability_difference_l1147_114795


namespace NUMINAMATH_CALUDE_jezebel_flower_cost_l1147_114796

/-- Calculates the total cost of flowers with discount and tax --/
def total_cost (red_roses white_lilies sunflowers blue_orchids : ℕ)
  (rose_price lily_price sunflower_price orchid_price : ℚ)
  (discount_rate tax_rate : ℚ) : ℚ :=
  let subtotal := red_roses * rose_price + white_lilies * lily_price +
                  sunflowers * sunflower_price + blue_orchids * orchid_price
  let discount := discount_rate * (red_roses * rose_price + white_lilies * lily_price)
  let after_discount := subtotal - discount
  let tax := tax_rate * after_discount
  after_discount + tax

/-- Theorem stating the total cost for Jezebel's flower purchase --/
theorem jezebel_flower_cost :
  total_cost 24 14 8 10 1.5 2.75 3 4.25 0.1 0.07 = 142.9 := by
  sorry

end NUMINAMATH_CALUDE_jezebel_flower_cost_l1147_114796


namespace NUMINAMATH_CALUDE_expression_simplification_l1147_114768

/-- For x in the open interval (0, 1], the given expression simplifies to ∛((1-x)/(3x)) -/
theorem expression_simplification (x : ℝ) (h : 0 < x ∧ x ≤ 1) :
  1.37 * Real.rpow ((2 * x^2) / (9 + 18*x + 9*x^2)) (1/3) *
  Real.sqrt (((1 + x) * Real.rpow (1 - x) (1/3)) / x) *
  Real.rpow ((3 * Real.sqrt (1 - x^2)) / (2 * x * Real.sqrt x)) (1/3) =
  Real.rpow ((1 - x) / (3 * x)) (1/3) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1147_114768


namespace NUMINAMATH_CALUDE_clock_angle_theorem_l1147_114750

/-- The angle (in degrees) the minute hand moves per minute -/
def minute_hand_speed : ℝ := 6

/-- The angle (in degrees) the hour hand moves per minute -/
def hour_hand_speed : ℝ := 0.5

/-- The current time in minutes past 3:00 -/
def t : ℝ := 23

/-- The position of the minute hand 8 minutes from now -/
def minute_hand_pos : ℝ := minute_hand_speed * (t + 8)

/-- The position of the hour hand 4 minutes ago -/
def hour_hand_pos : ℝ := 90 + hour_hand_speed * (t - 4)

/-- The theorem stating that the time is approximately 23 minutes past 3:00 -/
theorem clock_angle_theorem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  (|minute_hand_pos - hour_hand_pos| = 90 ∨ 
   |minute_hand_pos - hour_hand_pos| = 270) ∧
  t ≥ 0 ∧ t < 60 ∧ 
  |t - 23| < ε :=
sorry

end NUMINAMATH_CALUDE_clock_angle_theorem_l1147_114750


namespace NUMINAMATH_CALUDE_ricky_roses_l1147_114737

def initial_roses : ℕ → ℕ → ℕ → ℕ → Prop
  | total, stolen, people, each =>
    total = stolen + people * each

theorem ricky_roses : initial_roses 40 4 9 4 := by
  sorry

end NUMINAMATH_CALUDE_ricky_roses_l1147_114737


namespace NUMINAMATH_CALUDE_total_students_correct_l1147_114761

/-- The total number of students who appeared for the examination -/
def total_students : ℕ := 840

/-- The percentage of students who passed the examination -/
def pass_percentage : ℚ := 35 / 100

/-- The number of students who failed the examination -/
def failed_students : ℕ := 546

/-- Theorem stating that the total number of students is correct given the conditions -/
theorem total_students_correct : 
  (1 - pass_percentage) * total_students = failed_students := by sorry

end NUMINAMATH_CALUDE_total_students_correct_l1147_114761


namespace NUMINAMATH_CALUDE_aquarium_visitors_l1147_114782

theorem aquarium_visitors (total : ℕ) (ill_percentage : ℚ) : 
  total = 500 → ill_percentage = 40 / 100 → 
  (total : ℚ) * (1 - ill_percentage) = 300 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_visitors_l1147_114782


namespace NUMINAMATH_CALUDE_product_of_roots_l1147_114720

theorem product_of_roots (x y : ℝ) : 
  x = 16^(1/4) → y = 64^(1/2) → x * y = 16 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1147_114720


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1147_114738

theorem complex_equation_sum (x y : ℝ) :
  (x + 2 * Complex.I = y - 1 + y * Complex.I) → x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1147_114738


namespace NUMINAMATH_CALUDE_solution_exists_l1147_114756

theorem solution_exists (x : ℝ) (h1 : x > 0) (h2 : x * 3^x = 3^18) :
  ∃ k : ℕ, k = 15 ∧ k < x ∧ x < k + 1 := by
sorry

end NUMINAMATH_CALUDE_solution_exists_l1147_114756


namespace NUMINAMATH_CALUDE_one_non_prime_expression_l1147_114725

def expressions : List (ℕ → ℕ) := [
  (λ n => n^2 + (n+1)^2),
  (λ n => (n+1)^2 + (n+2)^2),
  (λ n => (n+2)^2 + (n+3)^2),
  (λ n => (n+3)^2 + (n+4)^2),
  (λ n => (n+4)^2 + (n+5)^2)
]

theorem one_non_prime_expression :
  (expressions.filter (λ f => ¬ Nat.Prime (f 1))).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_non_prime_expression_l1147_114725


namespace NUMINAMATH_CALUDE_industrial_lubricants_allocation_l1147_114772

/-- Represents the budget allocation for Megatech Corporation --/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  basic_astrophysics_degrees : ℝ
  total_degrees : ℝ

/-- Theorem stating that the industrial lubricants allocation is 8% --/
theorem industrial_lubricants_allocation
  (budget : BudgetAllocation)
  (h1 : budget.microphotonics = 12)
  (h2 : budget.home_electronics = 24)
  (h3 : budget.food_additives = 15)
  (h4 : budget.genetically_modified_microorganisms = 29)
  (h5 : budget.basic_astrophysics_degrees = 43.2)
  (h6 : budget.total_degrees = 360) :
  100 - (budget.microphotonics + budget.home_electronics + budget.food_additives +
    budget.genetically_modified_microorganisms + budget.basic_astrophysics_degrees *
    100 / budget.total_degrees) = 8 := by
  sorry


end NUMINAMATH_CALUDE_industrial_lubricants_allocation_l1147_114772


namespace NUMINAMATH_CALUDE_sum_of_divisors_360_l1147_114791

-- Define the sum of positive divisors function
def sumOfDivisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_divisors_360 (i j : ℕ) :
  sumOfDivisors (2^i * 3^j) = 360 → i = 3 ∧ j = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_360_l1147_114791


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l1147_114740

-- Define a fair coin
def fair_coin_prob : ℚ := 1/2

-- Define the number of flips
def num_flips : ℕ := 4

-- Define the probability of exactly 3 heads in 4 flips
def prob_3_heads : ℚ := Nat.choose num_flips 3 * fair_coin_prob^3 * (1 - fair_coin_prob)^(num_flips - 3)

-- Define the probability of 4 heads in 4 flips
def prob_4_heads : ℚ := fair_coin_prob^num_flips

-- Theorem statement
theorem coin_flip_probability_difference : 
  |prob_3_heads - prob_4_heads| = 7/16 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l1147_114740


namespace NUMINAMATH_CALUDE_odd_function_property_l1147_114703

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_prop : ∀ x, f (-x + 1) = f (x + 1))
  (h_val : f (-1) = 1) :
  f 2017 = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l1147_114703
