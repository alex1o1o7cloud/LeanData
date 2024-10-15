import Mathlib

namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_three_l739_73945

/-- Given a function f(x) = (x^2 + a) / (x + 1), prove that if f has an extremum at x = 1, then a = 3. -/
theorem extremum_implies_a_equals_three (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x^2 + a) / (x + 1)
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 3 := by
sorry


end NUMINAMATH_CALUDE_extremum_implies_a_equals_three_l739_73945


namespace NUMINAMATH_CALUDE_unique_solution_l739_73959

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (heq : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 11 ∧ y = 10 ∧ z = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l739_73959


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_side_c_equation_l739_73972

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine law for triangle ABC --/
axiom cosine_law (t : Triangle) : t.c^2 = t.a^2 + t.b^2 - 2*t.a*t.b*Real.cos t.C

/-- The given condition c/2 = b - a cos(C) --/
def condition (t : Triangle) : Prop := t.c/2 = t.b - t.a * Real.cos t.C

theorem angle_A_is_60_degrees (t : Triangle) (h : condition t) : Real.cos t.A = 1/2 := by sorry

theorem side_c_equation (t : Triangle) (h : condition t) (ha : t.a = Real.sqrt 15) (hb : t.b = 4) :
  t.c^2 - 4*t.c + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_side_c_equation_l739_73972


namespace NUMINAMATH_CALUDE_linear_function_through_points_l739_73908

/-- A linear function passing through two points -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem linear_function_through_points :
  ∃ k b : ℝ, 
    (linear_function k b 3 = 5) ∧ 
    (linear_function k b (-4) = -9) ∧
    (∀ x : ℝ, linear_function k b x = 2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_points_l739_73908


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l739_73922

/-- If 16x^2 + 32x + a is the square of a binomial, then a = 16 -/
theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b c : ℝ, ∀ x, 16 * x^2 + 32 * x + a = (b * x + c)^2) → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l739_73922


namespace NUMINAMATH_CALUDE_complex_power_eight_l739_73946

theorem complex_power_eight :
  let z : ℂ := 3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180))
  z^8 = -3280.5 - 3280.5 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eight_l739_73946


namespace NUMINAMATH_CALUDE_solve_for_a_l739_73985

theorem solve_for_a (a : ℝ) : (1 + 2 * a = -3) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l739_73985


namespace NUMINAMATH_CALUDE_parallel_vectors_l739_73953

def a (n : ℝ) : ℝ × ℝ := (n, -1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (-1, 2)

theorem parallel_vectors (n : ℝ) : 
  (∃ k : ℝ, a n + b = k • c) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l739_73953


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l739_73952

theorem quadratic_solution_sum (a b c : ℝ) : a ≠ 0 → (∀ x, a * x^2 + b * x + c = 0 ↔ x = 1) → a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l739_73952


namespace NUMINAMATH_CALUDE_part_i_part_ii_l739_73931

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 3| + |2*x - 4| - a

-- Theorem for Part I
theorem part_i :
  ∀ x : ℝ, f x 6 > 0 ↔ x < (1:ℝ)/3 ∨ x > (13:ℝ)/3 := by sorry

-- Theorem for Part II
theorem part_ii :
  ∀ a : ℝ, (∃ x : ℝ, f x a < 0) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_part_i_part_ii_l739_73931


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l739_73900

theorem imaginary_part_of_z (z : ℂ) : 
  z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I →
  z.im = (Real.sqrt 2 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l739_73900


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l739_73976

theorem max_sum_of_squares (p q r s : ℝ) : 
  p + q = 18 →
  p * q + r + s = 85 →
  p * r + q * s = 190 →
  r * s = 120 →
  p^2 + q^2 + r^2 + s^2 ≤ 886 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l739_73976


namespace NUMINAMATH_CALUDE_sum_of_specific_series_l739_73965

def arithmetic_series (a₁ : ℕ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + i * d)

def alternating_sum (l : List ℤ) : ℤ :=
  l.enum.foldl (λ acc (i, x) => acc + (if i % 2 == 0 then x else -x)) 0

theorem sum_of_specific_series :
  let series := arithmetic_series 100 (-2) 50
  alternating_sum series = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_series_l739_73965


namespace NUMINAMATH_CALUDE_min_value_expression_l739_73932

theorem min_value_expression (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x^2 + y^2 + z^2 = 1) :
  (z + 1)^2 / (2 * x * y * z) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l739_73932


namespace NUMINAMATH_CALUDE_solve_equation_l739_73917

theorem solve_equation (x y : ℝ) : y = 2 / (5 * x + 2) → y = 2 → x = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l739_73917


namespace NUMINAMATH_CALUDE_triangle_area_with_median_l739_73915

/-- The area of a triangle with two sides 1 and √15, and a median to the third side equal to 2, is √15/2. -/
theorem triangle_area_with_median (a b c : ℝ) (m : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 15) (h3 : m = 2) :
  (1/2 : ℝ) * a * b = (Real.sqrt 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_median_l739_73915


namespace NUMINAMATH_CALUDE_roots_in_unit_interval_l739_73913

noncomputable def f (q : ℕ → ℝ) : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => x
| (n + 2), x => (1 + q n) * x * f q (n + 1) x - q n * f q n x

theorem roots_in_unit_interval (q : ℕ → ℝ) (h : ∀ n, q n > 0) :
  ∀ n : ℕ, ∀ x : ℝ, |x| > 1 → |f q (n + 1) x| > |f q n x| :=
sorry

end NUMINAMATH_CALUDE_roots_in_unit_interval_l739_73913


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l739_73901

theorem sqrt_sum_equality : Real.sqrt (49 + 81) + Real.sqrt (36 + 49) = Real.sqrt 130 + Real.sqrt 85 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l739_73901


namespace NUMINAMATH_CALUDE_impossible_61_cents_l739_73940

/-- Represents the types of coins available in the piggy bank -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coin_value (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a combination of coins -/
def CoinCombination := List Coin

/-- Calculates the total value of a coin combination in cents -/
def total_value (comb : CoinCombination) : Nat :=
  comb.map coin_value |>.sum

/-- Theorem: It's impossible to make 61 cents with exactly 6 coins -/
theorem impossible_61_cents :
  ¬∃ (comb : CoinCombination), comb.length = 6 ∧ total_value comb = 61 := by
  sorry


end NUMINAMATH_CALUDE_impossible_61_cents_l739_73940


namespace NUMINAMATH_CALUDE_pretzel_price_is_two_l739_73903

/-- Represents the revenue and quantity information for a candy store --/
structure CandyStore where
  fudgePounds : ℕ
  fudgePrice : ℚ
  trufflesDozens : ℕ
  trufflePrice : ℚ
  pretzelsDozens : ℕ
  totalRevenue : ℚ

/-- Calculates the price of each chocolate-covered pretzel --/
def pretzelPrice (store : CandyStore) : ℚ :=
  let fudgeRevenue := store.fudgePounds * store.fudgePrice
  let trufflesRevenue := store.trufflesDozens * 12 * store.trufflePrice
  let pretzelsRevenue := store.totalRevenue - fudgeRevenue - trufflesRevenue
  let pretzelsCount := store.pretzelsDozens * 12
  pretzelsRevenue / pretzelsCount

/-- Theorem stating that the price of each chocolate-covered pretzel is $2 --/
theorem pretzel_price_is_two (store : CandyStore)
  (h1 : store.fudgePounds = 20)
  (h2 : store.fudgePrice = 5/2)
  (h3 : store.trufflesDozens = 5)
  (h4 : store.trufflePrice = 3/2)
  (h5 : store.pretzelsDozens = 3)
  (h6 : store.totalRevenue = 212) :
  pretzelPrice store = 2 := by
  sorry


end NUMINAMATH_CALUDE_pretzel_price_is_two_l739_73903


namespace NUMINAMATH_CALUDE_popsicle_bottle_cost_l739_73998

/-- Represents the cost of popsicle supplies and production -/
structure PopsicleSupplies where
  total_budget : ℚ
  mold_cost : ℚ
  stick_pack_cost : ℚ
  sticks_per_pack : ℕ
  popsicles_per_bottle : ℕ
  remaining_sticks : ℕ

/-- Calculates the cost of each bottle of juice -/
def bottle_cost (supplies : PopsicleSupplies) : ℚ :=
  let money_for_juice := supplies.total_budget - supplies.mold_cost - supplies.stick_pack_cost
  let used_sticks := supplies.sticks_per_pack - supplies.remaining_sticks
  let bottles_used := used_sticks / supplies.popsicles_per_bottle
  money_for_juice / bottles_used

/-- Theorem stating that given the conditions, the cost of each bottle is $2 -/
theorem popsicle_bottle_cost :
  let supplies := PopsicleSupplies.mk 10 3 1 100 20 40
  bottle_cost supplies = 2 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_bottle_cost_l739_73998


namespace NUMINAMATH_CALUDE_cosine_sum_simplification_l739_73950

theorem cosine_sum_simplification :
  let x := Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)
  x = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_simplification_l739_73950


namespace NUMINAMATH_CALUDE_median_determines_top_five_l739_73918

/-- A list of 9 distinct real numbers representing scores -/
def Scores := List ℝ

/-- Predicate to check if a list has exactly 9 distinct elements -/
def has_nine_distinct (s : Scores) : Prop :=
  s.length = 9 ∧ s.Nodup

/-- The median of a list of 9 distinct real numbers -/
def median (s : Scores) : ℝ := sorry

/-- Predicate to check if a score is in the top 5 of a list of scores -/
def in_top_five (score : ℝ) (s : Scores) : Prop := sorry

theorem median_determines_top_five (s : Scores) (score : ℝ) 
  (h : has_nine_distinct s) :
  in_top_five score s ↔ score > median s := by sorry

end NUMINAMATH_CALUDE_median_determines_top_five_l739_73918


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l739_73979

theorem triangle_abc_proof (A B C : Real) (a b c : Real) : 
  A = π / 6 →
  (1 + Real.sqrt 3) * c = 2 * b →
  c * b * Real.cos C = 1 + Real.sqrt 3 →
  C = π / 4 ∧ 
  a = Real.sqrt 2 ∧ 
  b = 1 + Real.sqrt 3 ∧ 
  c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l739_73979


namespace NUMINAMATH_CALUDE_not_always_valid_proof_from_untrue_prop_l739_73904

-- Define the concept of a valid proof
def ValidProof (premises : Prop) (conclusion : Prop) : Prop :=
  premises → conclusion

-- Define the concept of an untrue proposition
def UntrueProp (p : Prop) : Prop :=
  ¬p

-- Theorem stating that it's not generally true that a valid proof
-- can be constructed from an untrue proposition to reach a true conclusion
theorem not_always_valid_proof_from_untrue_prop :
  ¬∀ (p q : Prop), UntrueProp p → ValidProof p q → q :=
sorry

end NUMINAMATH_CALUDE_not_always_valid_proof_from_untrue_prop_l739_73904


namespace NUMINAMATH_CALUDE_building_volume_l739_73943

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular room -/
def volume (d : RoomDimensions) : ℝ := d.length * d.breadth * d.height

/-- Calculates the surface area of the walls of a rectangular room -/
def wallArea (d : RoomDimensions) : ℝ := 2 * (d.length * d.height + d.breadth * d.height)

/-- Calculates the floor area of a rectangular room -/
def floorArea (d : RoomDimensions) : ℝ := d.length * d.breadth

theorem building_volume (firstFloor secondFloor : RoomDimensions)
  (h1 : firstFloor.length = 15)
  (h2 : firstFloor.breadth = 12)
  (h3 : secondFloor.length = 20)
  (h4 : secondFloor.breadth = 10)
  (h5 : secondFloor.height = firstFloor.height)
  (h6 : 2 * floorArea firstFloor = wallArea firstFloor) :
  volume firstFloor + volume secondFloor = 2534.6 := by
  sorry

end NUMINAMATH_CALUDE_building_volume_l739_73943


namespace NUMINAMATH_CALUDE_vertical_complementary_implies_perpendicular_l739_73971

/-- Two angles are vertical if they are opposite each other when two lines intersect. -/
def are_vertical_angles (α β : Real) : Prop := sorry

/-- Two angles are complementary if their sum is 90 degrees. -/
def are_complementary (α β : Real) : Prop := α + β = 90

/-- Two lines are perpendicular if they form a right angle (90 degrees) at their intersection. -/
def are_perpendicular_lines (l1 l2 : Line) : Prop := sorry

theorem vertical_complementary_implies_perpendicular (α β : Real) (l1 l2 : Line) :
  are_vertical_angles α β → are_complementary α β → are_perpendicular_lines l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_vertical_complementary_implies_perpendicular_l739_73971


namespace NUMINAMATH_CALUDE_circle_intersection_and_origin_l739_73923

/-- Given line -/
def given_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

/-- Given circle -/
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0

/-- New circle -/
def new_circle (x y : ℝ) : Prop := x^2 + y^2 + 28*x - 15*y = 0

theorem circle_intersection_and_origin :
  (∀ x y : ℝ, given_line x y ∧ given_circle x y → new_circle x y) ∧
  new_circle 0 0 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_origin_l739_73923


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l739_73958

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - 1)

theorem f_max_min_on_interval :
  let a : ℝ := -3 * Real.pi / 4
  let b : ℝ := 3 * Real.pi / 4
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max = 0 ∧
    min = -(Real.sqrt 2 / 2) * Real.exp (3 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l739_73958


namespace NUMINAMATH_CALUDE_part_one_part_two_l739_73990

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := abs x * (x + a)

-- Part I
theorem part_one (a : ℝ) (h : ∀ x, f a x = -f a (-x)) : a = 0 := by
  sorry

-- Part II
theorem part_two (b : ℝ) (h1 : b > 0) 
  (h2 : ∃ (max min : ℝ), (∀ x ∈ Set.Icc (-b) b, f 0 x ≤ max ∧ min ≤ f 0 x) ∧ max - min = b) :
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l739_73990


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l739_73951

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 9 → b = 12 → c^2 = a^2 + b^2 → c = 15 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l739_73951


namespace NUMINAMATH_CALUDE_integral_3_minus_7x_squared_cos_2x_l739_73939

theorem integral_3_minus_7x_squared_cos_2x (π : ℝ) :
  (∫ x in (0 : ℝ)..(2 * π), (3 - 7 * x^2) * Real.cos (2 * x)) = -7 * π := by
  sorry

end NUMINAMATH_CALUDE_integral_3_minus_7x_squared_cos_2x_l739_73939


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l739_73962

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), (∀ x : ℝ, -x^2 + 9*x - 18 ≥ 0 → x ≤ x_max) ∧ (-x_max^2 + 9*x_max - 18 ≥ 0) ∧ x_max = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l739_73962


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l739_73957

/-- A conic section curve with two foci -/
structure ConicSection where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The eccentricity of a conic section -/
def eccentricity (C : ConicSection) : ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Given a conic section curve C with foci F₁ and F₂, and a point P on C
    such that |PF₁| : |F₁F₂| : |PF₂| = 4 : 3 : 2, the eccentricity of C is either 1/2 or 3/2 -/
theorem conic_section_eccentricity (C : ConicSection) (P : ℝ × ℝ) :
  distance P C.F₁ / distance C.F₁ C.F₂ = 4/3 ∧
  distance C.F₁ C.F₂ / distance P C.F₂ = 3/2 →
  eccentricity C = 1/2 ∨ eccentricity C = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l739_73957


namespace NUMINAMATH_CALUDE_units_digit_of_F_F10_l739_73912

-- Define the sequence F_n
def F : ℕ → ℕ
| 0 => 3
| 1 => 2
| (n + 2) => F (n + 1) + F n

-- Theorem statement
theorem units_digit_of_F_F10 : ∃ k : ℕ, F (F 10) = 10 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_F_F10_l739_73912


namespace NUMINAMATH_CALUDE_curve_C₂_equation_l739_73980

-- Define the parabola C₁
def C₁ (x y : ℝ) : Prop := y = (1/20) * x^2

-- Define the focus F of C₁
def F : ℝ × ℝ := (0, 5)

-- Define point E symmetric to F with respect to the origin
def E : ℝ × ℝ := (0, -5)

-- Define the property of points on C₂
def on_C₂ (x y : ℝ) : Prop :=
  abs (Real.sqrt ((x - E.1)^2 + (y - E.2)^2) - Real.sqrt ((x - F.1)^2 + (y - F.2)^2)) = 6

-- Theorem statement
theorem curve_C₂_equation :
  ∀ x y : ℝ, on_C₂ x y ↔ y^2/9 - x^2/16 = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_C₂_equation_l739_73980


namespace NUMINAMATH_CALUDE_tangent_range_l739_73988

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the circle C
def C (k : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y + 2*k - 1 = 0

-- Define the condition for two tangents
def has_two_tangents (P : ℝ × ℝ) (C : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ k, (P.1^2 + P.2^2 + 2*P.1 + P.2 + 2*k - 1 > 0) ∧
       (4 + 1 - 4*(2*k - 1) > 0)

-- Theorem statement
theorem tangent_range :
  has_two_tangents P C → ∃ k, -4 < k ∧ k < 9/8 :=
sorry

end NUMINAMATH_CALUDE_tangent_range_l739_73988


namespace NUMINAMATH_CALUDE_freshman_percentage_l739_73927

theorem freshman_percentage (total_students : ℝ) (freshmen : ℝ) 
  (h1 : freshmen > 0) (h2 : total_students > 0) :
  let liberal_arts_fraction : ℝ := 0.6
  let psychology_fraction : ℝ := 0.5
  let freshmen_psych_liberal_fraction : ℝ := 0.24
  (liberal_arts_fraction * psychology_fraction * (freshmen / total_students) = 
    freshmen_psych_liberal_fraction) →
  freshmen / total_students = 0.8 := by
sorry

end NUMINAMATH_CALUDE_freshman_percentage_l739_73927


namespace NUMINAMATH_CALUDE_dot_product_problem_l739_73964

/-- Given vectors a and b in ℝ², prove that the dot product of (2a + b) and a is 6. -/
theorem dot_product_problem (a b : ℝ × ℝ) (h1 : a = (2, -1)) (h2 : b = (-1, 2)) :
  (2 • a + b) • a = 6 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_problem_l739_73964


namespace NUMINAMATH_CALUDE_linear_system_solution_l739_73921

theorem linear_system_solution (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47)
  (eq2 : 4 * a + 3 * b = 39) :
  2 * a + 2 * b = 164 / 7 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l739_73921


namespace NUMINAMATH_CALUDE_probability_two_white_and_one_white_one_red_l739_73995

/-- Represents the color of a ball -/
inductive Color
  | White
  | Red

/-- Represents a bag of balls -/
structure Bag :=
  (total : Nat)
  (white : Nat)
  (red : Nat)
  (h_total : total = white + red)

/-- Calculates the probability of drawing two balls of a specific color combination -/
def probability_draw_two (bag : Bag) (first second : Color) : Rat :=
  sorry

theorem probability_two_white_and_one_white_one_red 
  (bag : Bag)
  (h_total : bag.total = 6)
  (h_white : bag.white = 4)
  (h_red : bag.red = 2) :
  (probability_draw_two bag Color.White Color.White = 2/5) ∧
  (probability_draw_two bag Color.White Color.Red = 8/15) :=
sorry

end NUMINAMATH_CALUDE_probability_two_white_and_one_white_one_red_l739_73995


namespace NUMINAMATH_CALUDE_fast_food_order_cost_correct_l739_73942

/-- Calculates the total cost of a fast food order with discount and tax --/
def fastFoodOrderCost (burgerPrice sandwichPrice smoothiePrice : ℚ)
                      (smoothieQuantity : ℕ)
                      (discountRate taxRate : ℚ)
                      (discountThreshold : ℚ)
                      (orderTime : ℕ) : ℚ :=
  let totalBeforeDiscount := burgerPrice + sandwichPrice + smoothiePrice * smoothieQuantity
  let discountedPrice := if totalBeforeDiscount > discountThreshold ∧ orderTime ≥ 1400 ∧ orderTime ≤ 1600
                         then totalBeforeDiscount * (1 - discountRate)
                         else totalBeforeDiscount
  let finalPrice := discountedPrice * (1 + taxRate)
  finalPrice

theorem fast_food_order_cost_correct :
  fastFoodOrderCost 5.75 4.50 4.25 2 0.20 0.12 15 1545 = 16.80 := by
  sorry

end NUMINAMATH_CALUDE_fast_food_order_cost_correct_l739_73942


namespace NUMINAMATH_CALUDE_order_of_expressions_l739_73968

theorem order_of_expressions : 
  let a : ℝ := (1/2)^(3/2)
  let b : ℝ := Real.log π
  let c : ℝ := Real.log (3/2) / Real.log (1/2)
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l739_73968


namespace NUMINAMATH_CALUDE_cow_spots_l739_73963

theorem cow_spots (left_spots : ℕ) : 
  (left_spots + (3 * left_spots + 7) = 71) → left_spots = 16 := by
  sorry

end NUMINAMATH_CALUDE_cow_spots_l739_73963


namespace NUMINAMATH_CALUDE_lucy_fish_count_l739_73997

/-- The number of fish Lucy needs to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy wants to have -/
def total_fish : ℕ := 280

/-- The number of fish Lucy currently has -/
def current_fish : ℕ := total_fish - fish_to_buy

theorem lucy_fish_count : current_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l739_73997


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l739_73905

theorem imaginary_part_of_2_minus_i :
  Complex.im (2 - Complex.I) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l739_73905


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l739_73978

/-- Given a complex number z satisfying (2+i)z = 1+3i, prove that z is in the first quadrant -/
theorem z_in_first_quadrant (z : ℂ) (h : (2 + Complex.I) * z = 1 + 3 * Complex.I) :
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l739_73978


namespace NUMINAMATH_CALUDE_exists_max_volume_l739_73916

/-- A rectangular prism with specific diagonal lengths --/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  h_space_diagonal : a^2 + b^2 + c^2 = 1
  h_face_diagonal : b^2 + c^2 = 2
  h_positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The volume of a rectangular prism --/
def volume (prism : RectangularPrism) : ℝ :=
  prism.a * prism.b * prism.c

/-- There exists a value p that maximizes the volume of the rectangular prism --/
theorem exists_max_volume : 
  ∃ p : ℝ, p > 0 ∧ 
  ∃ prism : RectangularPrism, 
    prism.a = p ∧
    ∀ other : RectangularPrism, volume prism ≥ volume other := by
  sorry


end NUMINAMATH_CALUDE_exists_max_volume_l739_73916


namespace NUMINAMATH_CALUDE_intersection_A_B_l739_73944

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | x = 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l739_73944


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_camp_cedar_ratio_l739_73960

/-- Represents a summer camp with boys, girls, and counselors -/
structure SummerCamp where
  boys : ℕ
  girls : ℕ
  counselors : ℕ
  children_per_counselor : ℕ

/-- Camp Cedar with given conditions -/
def camp_cedar : SummerCamp :=
  { boys := 40,
    girls := 120,  -- This is derived, not given directly
    counselors := 20,
    children_per_counselor := 8 }

/-- The theorem stating the ratio of girls to boys in Camp Cedar -/
theorem girls_to_boys_ratio (c : SummerCamp) (h1 : c = camp_cedar) :
  c.girls / c.boys = 3 := by
  sorry

/-- The main theorem proving the ratio of girls to boys in Camp Cedar -/
theorem camp_cedar_ratio :
  (camp_cedar.girls : ℚ) / camp_cedar.boys = 3 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_camp_cedar_ratio_l739_73960


namespace NUMINAMATH_CALUDE_g_has_no_zeros_l739_73955

noncomputable section

open Real

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x - a * log x - x + exp (x - 1)

-- State the theorem
theorem g_has_no_zeros (a : ℝ) (h : 0 ≤ a ∧ a ≤ exp 1) :
  ∀ x > 0, g a x ≠ 0 := by
  sorry

end

end NUMINAMATH_CALUDE_g_has_no_zeros_l739_73955


namespace NUMINAMATH_CALUDE_frank_candy_total_l739_73902

/-- Given that Frank put candy equally into 2 bags and there are 8 pieces of candy in each bag,
    prove that the total number of pieces of candy is 16. -/
theorem frank_candy_total (num_bags : ℕ) (pieces_per_bag : ℕ) 
    (h1 : num_bags = 2) 
    (h2 : pieces_per_bag = 8) : 
  num_bags * pieces_per_bag = 16 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_total_l739_73902


namespace NUMINAMATH_CALUDE_road_length_l739_73914

theorem road_length (repaired : ℚ) (remaining_extra : ℚ) : 
  repaired = 7/15 → remaining_extra = 2/5 → repaired + (repaired + remaining_extra) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_road_length_l739_73914


namespace NUMINAMATH_CALUDE_smallest_sum_of_distinct_squares_l739_73984

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- The theorem statement -/
theorem smallest_sum_of_distinct_squares (a b c d : ℕ) :
  isPerfectSquare a ∧ isPerfectSquare b ∧ isPerfectSquare c ∧ isPerfectSquare d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a ^ b = c ^ d →
  305 ≤ a + b + c + d :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_distinct_squares_l739_73984


namespace NUMINAMATH_CALUDE_distance_between_points_on_lines_l739_73966

/-- The distance between two points on specific lines -/
theorem distance_between_points_on_lines (a c m k : ℝ) :
  let b := 2 * m * a + k
  let d := -m * c + k
  (((c - a)^2 + (d - b)^2) : ℝ).sqrt = ((1 + m^2 * (c + 2*a)^2) * (c - a)^2 : ℝ).sqrt :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_on_lines_l739_73966


namespace NUMINAMATH_CALUDE_total_notes_count_l739_73909

/-- Proves that the total number of notes is 126 given the conditions -/
theorem total_notes_count (total_amount : ℕ) (note_50_count : ℕ) (note_50_value : ℕ) (note_500_value : ℕ) :
  total_amount = 10350 ∧
  note_50_count = 117 ∧
  note_50_value = 50 ∧
  note_500_value = 500 ∧
  total_amount = note_50_count * note_50_value + (total_amount - note_50_count * note_50_value) / note_500_value * note_500_value →
  note_50_count + (total_amount - note_50_count * note_50_value) / note_500_value = 126 :=
by sorry

end NUMINAMATH_CALUDE_total_notes_count_l739_73909


namespace NUMINAMATH_CALUDE_sodium_chloride_dilution_l739_73949

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% sodium chloride solution
    results in a 25% concentration. -/
theorem sodium_chloride_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 ∧
  initial_concentration = 0.4 ∧
  added_water = 30 ∧
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry

#check sodium_chloride_dilution

end NUMINAMATH_CALUDE_sodium_chloride_dilution_l739_73949


namespace NUMINAMATH_CALUDE_solution_set_for_a_l739_73993

/-- The solution set for parameter a in the given equation with domain restrictions -/
theorem solution_set_for_a (x a : ℝ) : 
  x ≠ 2 → x ≠ 6 → a - 7*x + 39 ≥ 0 →
  (x^2 - 4*x - 21 + ((|x-2|)/(x-2) + (|x-6|)/(x-6) + a)^2 = 0) →
  a ∈ Set.Ioo (-5) (-4) ∪ Set.Ioo (-3) 3 ∪ Set.Ico 5 7 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_l739_73993


namespace NUMINAMATH_CALUDE_nonnegative_real_inequality_l739_73948

theorem nonnegative_real_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^2 * y^2 + x^2 * y + x * y^2 ≤ x^4 * y + x + y^4 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_real_inequality_l739_73948


namespace NUMINAMATH_CALUDE_expression_simplification_l739_73991

theorem expression_simplification (a b : ℝ) (h1 : a = 1) (h2 : b = -2) :
  ((a - 2*b)^2 - (a - 2*b)*(a + 2*b) - 4*b) / (-2*b) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l739_73991


namespace NUMINAMATH_CALUDE_sum_from_true_discount_and_simple_interest_l739_73987

/-- Given a sum, time, and rate, if the true discount is 80 and the simple interest is 88, then the sum is 880. -/
theorem sum_from_true_discount_and_simple_interest
  (S T R : ℝ) 
  (h1 : S > 0) 
  (h2 : T > 0) 
  (h3 : R > 0) 
  (h4 : (S * R * T) / 100 = 88) 
  (h5 : S - S / (1 + R * T / 100) = 80) : 
  S = 880 := by
sorry

end NUMINAMATH_CALUDE_sum_from_true_discount_and_simple_interest_l739_73987


namespace NUMINAMATH_CALUDE_company_manager_fraction_l739_73999

/-- Given a company with female managers, total female employees, and the condition that
    the fraction of managers is the same for all employees and male employees,
    prove that the fraction of employees who are managers is 0.4 -/
theorem company_manager_fraction (total_female_employees : ℕ) (female_managers : ℕ)
    (h1 : female_managers = 200)
    (h2 : total_female_employees = 500)
    (h3 : ∃ (f : ℚ), f * (total_female_employees : ℚ) = (female_managers : ℚ) ∧
                     f * ((total_female_employees : ℚ) - (female_managers : ℚ)) = 
                     (female_managers : ℚ) * ((total_female_employees : ℚ) / (female_managers : ℚ) - 1)) :
  ∃ (f : ℚ), f = 0.4 ∧ 
    f * (total_female_employees : ℚ) = (female_managers : ℚ) ∧
    f * ((total_female_employees : ℚ) - (female_managers : ℚ)) = 
    (female_managers : ℚ) * ((total_female_employees : ℚ) / (female_managers : ℚ) - 1) := by
  sorry


end NUMINAMATH_CALUDE_company_manager_fraction_l739_73999


namespace NUMINAMATH_CALUDE_sphere_quarter_sphere_radius_l739_73982

theorem sphere_quarter_sphere_radius (r : ℝ) (h : r = 2 * Real.rpow 4 (1/3)) :
  ∃ R : ℝ, (4/3 * Real.pi * R^3 = 1/3 * Real.pi * r^3) ∧ R = 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_quarter_sphere_radius_l739_73982


namespace NUMINAMATH_CALUDE_complex_quadrant_l739_73920

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the conditions
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Define the equation
def equation (z a : ℂ) : Prop := (2 + i) * z = 1 + a * i^3

-- Theorem statement
theorem complex_quadrant (z a : ℂ) 
  (h1 : is_pure_imaginary z) 
  (h2 : equation z a) : 
  (a + z).re > 0 ∧ (a + z).im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l739_73920


namespace NUMINAMATH_CALUDE_zero_of_f_l739_73929

def f (x : ℝ) := 2 * x - 3

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = 3/2 :=
sorry

end NUMINAMATH_CALUDE_zero_of_f_l739_73929


namespace NUMINAMATH_CALUDE_exponent_multiplication_l739_73973

theorem exponent_multiplication (x : ℝ) : x^8 * x^2 = x^10 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l739_73973


namespace NUMINAMATH_CALUDE_probability_of_selecting_seven_l739_73938

-- Define the fraction
def fraction : ℚ := 3 / 8

-- Define the decimal representation as a list of digits
def decimal_representation : List ℕ := [3, 7, 5]

-- Define the target digit
def target_digit : ℕ := 7

-- Theorem statement
theorem probability_of_selecting_seven :
  (decimal_representation.filter (· = target_digit)).length / decimal_representation.length = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_seven_l739_73938


namespace NUMINAMATH_CALUDE_sum_of_numbers_l739_73926

theorem sum_of_numbers (a b c : ℝ) (ha : a = 0.8) (hb : b = 1/2) (hc : c = 0.9) :
  (if a ≥ 0.3 then a else 0) + (if b ≥ 0.3 then b else 0) + (if c ≥ 0.3 then c else 0) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l739_73926


namespace NUMINAMATH_CALUDE_triangle_angle_c_l739_73933

theorem triangle_angle_c (A B C : Real) :
  -- ABC is a triangle
  A + B + C = π →
  -- Given condition
  |Real.cos A - Real.sqrt 3 / 2| + (1 - Real.tan B)^2 = 0 →
  -- Conclusion
  C = π * 7 / 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l739_73933


namespace NUMINAMATH_CALUDE_f_x1_gt_f_x2_l739_73981

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

-- Define the theorem
theorem f_x1_gt_f_x2 
  (h_even : is_even f) 
  (h_incr : is_increasing_on_pos f) 
  (x₁ x₂ : ℝ) 
  (h_x1_neg : x₁ < 0) 
  (h_x2_pos : x₂ > 0) 
  (h_abs : abs x₁ > abs x₂) : 
  f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_x1_gt_f_x2_l739_73981


namespace NUMINAMATH_CALUDE_programmer_is_odd_one_out_l739_73977

-- Define the set of professions
inductive Profession
| Dentist
| ElementarySchoolTeacher
| Programmer

-- Define a predicate for having special pension benefits
def has_special_pension_benefits (p : Profession) : Prop :=
  match p with
  | Profession.Dentist => true
  | Profession.ElementarySchoolTeacher => true
  | Profession.Programmer => false

-- Define the odd one out
def is_odd_one_out (p : Profession) : Prop :=
  ¬(has_special_pension_benefits p) ∧
  ∀ q : Profession, q ≠ p → has_special_pension_benefits q

-- Theorem statement
theorem programmer_is_odd_one_out :
  is_odd_one_out Profession.Programmer :=
sorry

end NUMINAMATH_CALUDE_programmer_is_odd_one_out_l739_73977


namespace NUMINAMATH_CALUDE_gcd_3_powers_l739_73936

theorem gcd_3_powers : Nat.gcd (3^1001 - 1) (3^1012 - 1) = 177146 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3_powers_l739_73936


namespace NUMINAMATH_CALUDE_hillary_climbing_rate_l739_73906

/-- Hillary's climbing rate in ft/hr -/
def hillary_rate : ℝ := 800

/-- Eddy's climbing rate in ft/hr -/
def eddy_rate : ℝ := hillary_rate - 500

/-- Distance from base camp to summit in ft -/
def summit_distance : ℝ := 5000

/-- Distance Hillary climbs before stopping in ft -/
def hillary_climb_distance : ℝ := summit_distance - 1000

/-- Hillary's descent rate in ft/hr -/
def hillary_descent_rate : ℝ := 1000

/-- Total time from departure to meeting in hours -/
def total_time : ℝ := 6

theorem hillary_climbing_rate :
  hillary_rate = 800 ∧
  eddy_rate = hillary_rate - 500 ∧
  summit_distance = 5000 ∧
  hillary_climb_distance = summit_distance - 1000 ∧
  hillary_descent_rate = 1000 ∧
  total_time = 6 →
  hillary_rate * (total_time - hillary_climb_distance / hillary_descent_rate) = hillary_climb_distance ∧
  eddy_rate * total_time = hillary_climb_distance - hillary_descent_rate * (total_time - hillary_climb_distance / hillary_descent_rate) :=
by sorry

end NUMINAMATH_CALUDE_hillary_climbing_rate_l739_73906


namespace NUMINAMATH_CALUDE_smallest_ratio_of_equation_l739_73974

theorem smallest_ratio_of_equation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : 18 * x - 4 * x^2 + 2 * x^3 - 9 * y - 10 * x * y - x^2 * y + 6 * y^2 + 2 * x * y^2 - y^3 = 0) :
  ∃ (k : ℝ), k = y / x ∧ k ≥ 4/3 ∧ (∀ (k' : ℝ), k' = y / x → k' ≥ k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_ratio_of_equation_l739_73974


namespace NUMINAMATH_CALUDE_empty_set_subset_of_all_l739_73986

theorem empty_set_subset_of_all (A : Set α) : ∅ ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_empty_set_subset_of_all_l739_73986


namespace NUMINAMATH_CALUDE_files_remaining_l739_73935

theorem files_remaining (m v d : ℕ) (hm : m = 4) (hv : v = 21) (hd : d = 23) :
  (m + v) - d = 2 :=
by sorry

end NUMINAMATH_CALUDE_files_remaining_l739_73935


namespace NUMINAMATH_CALUDE_equation_solution_l739_73970

theorem equation_solution :
  ∃! y : ℚ, 7 * (2 * y - 3) + 4 = 3 * (5 - 9 * y) ∧ y = 32 / 41 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l739_73970


namespace NUMINAMATH_CALUDE_star_calculation_l739_73930

-- Define the ★ operation
def star (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem star_calculation : star (star 3 5) 8 = -1/3 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l739_73930


namespace NUMINAMATH_CALUDE_infinitely_many_winning_positions_l739_73925

/-- The pebble game where players remove square numbers of pebbles -/
def PebbleGame (n : ℕ) : Prop :=
  ∀ (move : ℕ → ℕ), 
    (∀ k, ∃ m : ℕ, move k = m * m) → 
    (∀ k, move k ≤ n * n) →
    (n + 1 ≤ n * n + n + 1 - move (n * n + n + 1))

/-- There are infinitely many winning positions for the second player -/
theorem infinitely_many_winning_positions :
  ∀ n : ℕ, PebbleGame n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_winning_positions_l739_73925


namespace NUMINAMATH_CALUDE_f_properties_l739_73911

-- Define the function f
def f (x : ℝ) : ℝ := -x - x^3

-- State the theorem
theorem f_properties (x₁ x₂ : ℝ) (h : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧ (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l739_73911


namespace NUMINAMATH_CALUDE_strawberry_picking_problem_l739_73992

/-- Calculates the number of pounds of strawberries picked given the problem conditions -/
def strawberries_picked (entrance_fee : ℚ) (price_per_pound : ℚ) (num_people : ℕ) (total_paid : ℚ) : ℚ :=
  (total_paid + num_people * entrance_fee) / price_per_pound

/-- Theorem stating that under the given conditions, 7 pounds of strawberries were picked -/
theorem strawberry_picking_problem :
  let entrance_fee : ℚ := 4
  let price_per_pound : ℚ := 20
  let num_people : ℕ := 3
  let total_paid : ℚ := 128
  strawberries_picked entrance_fee price_per_pound num_people total_paid = 7 := by
  sorry


end NUMINAMATH_CALUDE_strawberry_picking_problem_l739_73992


namespace NUMINAMATH_CALUDE_total_balloons_l739_73910

theorem total_balloons (gold : ℕ) (silver : ℕ) (black : ℕ) : 
  gold = 141 → 
  silver = 2 * gold → 
  black = 150 → 
  gold + silver + black = 573 := by
sorry

end NUMINAMATH_CALUDE_total_balloons_l739_73910


namespace NUMINAMATH_CALUDE_dividend_calculation_l739_73969

theorem dividend_calculation (quotient divisor k : ℕ) 
  (h1 : quotient = 4)
  (h2 : divisor = k)
  (h3 : k = 4) :
  quotient * divisor = 16 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l739_73969


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l739_73919

theorem sufficient_not_necessary : 
  let A := {x : ℝ | 1 < x ∧ x < 2}
  let B := {x : ℝ | x < 2}
  (A ⊂ B) ∧ (B \ A).Nonempty := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l739_73919


namespace NUMINAMATH_CALUDE_min_value_of_function_l739_73961

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  ∃ y : ℝ, y = x + 4 / x ∧ y ≥ 4 ∧ (∀ z : ℝ, z = x + 4 / x → z ≥ y) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l739_73961


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l739_73996

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*Complex.I)*z = 3 - 4*Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l739_73996


namespace NUMINAMATH_CALUDE_number_order_l739_73934

/-- Represents a number in a given base -/
structure BaseNumber where
  digits : List Nat
  base : Nat

/-- Convert a BaseNumber to its decimal representation -/
def toDecimal (n : BaseNumber) : Nat :=
  n.digits.enum.foldl (fun acc (i, d) => acc + d * n.base ^ i) 0

/-- Define the given numbers -/
def a : BaseNumber := ⟨[14, 3], 16⟩
def b : BaseNumber := ⟨[0, 1, 2], 6⟩
def c : BaseNumber := ⟨[0, 0, 0, 1], 4⟩
def d : BaseNumber := ⟨[1, 1, 0, 1, 1, 1], 2⟩

/-- Theorem stating the order of the given numbers -/
theorem number_order :
  toDecimal b > toDecimal c ∧ toDecimal c > toDecimal a ∧ toDecimal a > toDecimal d := by
  sorry

end NUMINAMATH_CALUDE_number_order_l739_73934


namespace NUMINAMATH_CALUDE_certain_number_value_l739_73941

theorem certain_number_value : ∃ x : ℝ, (0.60 * 50 = 0.42 * x + 17.4) ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l739_73941


namespace NUMINAMATH_CALUDE_abhay_sameer_speed_difference_l739_73989

/-- Prove that when Abhay doubles his speed, he takes 1 hour less than Sameer to cover 18 km,
    given that Abhay's original speed is 3 km/h and he initially takes 2 hours more than Sameer. -/
theorem abhay_sameer_speed_difference (distance : ℝ) (abhay_speed : ℝ) (sameer_speed : ℝ) :
  distance = 18 →
  abhay_speed = 3 →
  distance / abhay_speed = distance / sameer_speed + 2 →
  distance / (2 * abhay_speed) = distance / sameer_speed - 1 :=
by sorry

end NUMINAMATH_CALUDE_abhay_sameer_speed_difference_l739_73989


namespace NUMINAMATH_CALUDE_quadratic_equation_d_has_two_distinct_roots_l739_73907

/-- Discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Predicate for a quadratic equation having two distinct real roots -/
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ discriminant a b c > 0

theorem quadratic_equation_d_has_two_distinct_roots :
  has_two_distinct_real_roots 1 2 (-1) ∧
  ¬has_two_distinct_real_roots 1 0 4 ∧
  ¬has_two_distinct_real_roots 4 (-4) 1 ∧
  ¬has_two_distinct_real_roots 1 (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_d_has_two_distinct_roots_l739_73907


namespace NUMINAMATH_CALUDE_tom_fruit_purchase_amount_l739_73928

/-- Represents a fruit purchase with quantity and rate --/
structure FruitPurchase where
  quantity : ℝ
  rate : ℝ

/-- Calculates the total cost of purchases before discount --/
def totalCost (purchases : List FruitPurchase) : ℝ :=
  purchases.foldl (fun acc p => acc + p.quantity * p.rate) 0

/-- Calculates the final amount after discount and tax --/
def finalAmount (purchases : List FruitPurchase) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let total := totalCost purchases
  let discountedPrice := total * (1 - discountRate)
  discountedPrice * (1 + taxRate)

theorem tom_fruit_purchase_amount :
  let purchases := [
    ⟨8, 70⟩,  -- Apples
    ⟨9, 55⟩,  -- Mangoes
    ⟨5, 40⟩,  -- Oranges
    ⟨12, 30⟩, -- Bananas
    ⟨7, 45⟩,  -- Grapes
    ⟨4, 80⟩   -- Cherries
  ]
  finalAmount purchases 0.1 0.05 = 2126.25 := by
  sorry


end NUMINAMATH_CALUDE_tom_fruit_purchase_amount_l739_73928


namespace NUMINAMATH_CALUDE_parabola_tangent_intersection_l739_73937

/-- Two points on a parabola with tangents intersecting at 45° -/
structure ParabolaPoints where
  a : ℝ
  b : ℝ

/-- The parabola y = 4x^2 -/
def parabola (x : ℝ) : ℝ := 4 * x^2

/-- Tangent slope at a point on the parabola -/
def tangentSlope (x : ℝ) : ℝ := 8 * x

/-- Condition for tangents intersecting at 45° -/
def tangentAngle45 (p : ParabolaPoints) : Prop :=
  |((tangentSlope p.a - tangentSlope p.b) / (1 + tangentSlope p.a * tangentSlope p.b))| = 1

/-- Y-coordinate of the intersection point of tangents -/
def intersectionY (p : ParabolaPoints) : ℝ := 4 * p.a * p.b

theorem parabola_tangent_intersection
  (p : ParabolaPoints)
  (h1 : parabola p.a = 4 * p.a^2)
  (h2 : parabola p.b = 4 * p.b^2)
  (h3 : tangentAngle45 p) :
  intersectionY p = -1/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_intersection_l739_73937


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l739_73967

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (sum_squares : a^2 + b^2 + c^2 = 0.1) :
  a^4 + b^4 + c^4 = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l739_73967


namespace NUMINAMATH_CALUDE_extremum_condition_l739_73924

/-- A function f: ℝ → ℝ has an extremum at x₀ if f(x₀) is either a maximum or minimum value of f in some neighborhood of x₀ -/
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x₀ ≤ f x ∨ f x ≤ f x₀

theorem extremum_condition (f : ℝ → ℝ) (x₀ : ℝ) (hf : Differentiable ℝ f) :
  (HasExtremumAt f x₀ → (deriv f) x₀ = 0) ∧
  ¬(((deriv f) x₀ = 0) → HasExtremumAt f x₀) :=
sorry

end NUMINAMATH_CALUDE_extremum_condition_l739_73924


namespace NUMINAMATH_CALUDE_horner_method_v3_l739_73983

def f (x : ℤ) (a b : ℤ) : ℤ := x^5 + a*x^4 - b*x^2 + 1

def horner_v3 (a b : ℤ) : ℤ :=
  let x := -1
  let v0 := 1
  let v1 := v0 * x + a
  let v2 := v1 * x + 0
  v2 * x - b

theorem horner_method_v3 :
  horner_v3 47 37 = 9 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l739_73983


namespace NUMINAMATH_CALUDE_sum_of_integers_l739_73994

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 289) (h2 : x * y = 120) : x + y = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l739_73994


namespace NUMINAMATH_CALUDE_expression_evaluation_l739_73947

theorem expression_evaluation :
  (((3^0 : ℝ) - 1 + 4^2 - 3)^(-1 : ℝ)) * 4 = 4/13 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l739_73947


namespace NUMINAMATH_CALUDE_fraction_of_larger_part_l739_73975

theorem fraction_of_larger_part (total : ℝ) (larger : ℝ) (f : ℝ) : 
  total = 66 →
  larger = 50 →
  f * larger = 0.625 * (total - larger) + 10 →
  f = 0.4 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_larger_part_l739_73975


namespace NUMINAMATH_CALUDE_no_real_roots_iff_m_less_than_neg_one_l739_73954

theorem no_real_roots_iff_m_less_than_neg_one (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - m ≠ 0) ↔ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_m_less_than_neg_one_l739_73954


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l739_73956

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (h_total : total = 120) (h_props : a = 3 ∧ b = 5 ∧ c = 7) :
  let x := total / (a + b + c)
  min (a * x) (min (b * x) (c * x)) = 24 := by
sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l739_73956
