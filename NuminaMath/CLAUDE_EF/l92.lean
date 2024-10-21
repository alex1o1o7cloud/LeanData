import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stuffed_animal_cost_l92_9235

/-- The cost of items at a garage sale -/
structure GarageSale where
  magnet : ℝ
  sticker : ℝ
  stuffed_animal : ℝ
  toy_car : ℝ
  h1 : magnet = 3 * sticker
  h2 : magnet = 1/4 * (2 * stuffed_animal)
  h3 : toy_car = 1/2 * stuffed_animal
  h4 : toy_car = 2 * sticker
  h5 : magnet = 6

/-- The cost of a single stuffed animal is $8 -/
theorem stuffed_animal_cost (sale : GarageSale) : sale.stuffed_animal = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stuffed_animal_cost_l92_9235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_representations_l92_9228

/-- Represents the number of ways to express a positive integer as the sum of consecutive integers --/
def f (n : ℕ+) : ℕ :=
  f₁ n + f₂ n
where
  /-- Counts the odd factors m of n that satisfy m < (1 + √(1 + 8n)) / 2 --/
  f₁ (n : ℕ+) : ℕ := sorry

  /-- Counts the even factors of 2n that are not multiples of 2^(p₀+1), where n = 2^p₀ * q and q is odd --/
  f₂ (n : ℕ+) : ℕ := sorry

/-- Represents the number of ways to express a positive integer as the sum of consecutive integers --/
def NumberOfWaysToExpressAsConsecutiveSum (n : ℕ+) : ℕ := sorry

/-- The main theorem stating that f(n) correctly counts the number of ways to express n
    as the sum of consecutive positive integers --/
theorem consecutive_sum_representations (n : ℕ+) :
  NumberOfWaysToExpressAsConsecutiveSum n = f n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_representations_l92_9228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gilled_mushroom_count_l92_9209

/-- Represents the number of mushrooms with a specific characteristic -/
structure MushroomCount where
  count : Nat

/-- The total number of mushrooms on the log -/
def total_mushrooms : Nat := 30

/-- Theorem: Given the conditions, prove that there are 3 gilled mushrooms -/
theorem gilled_mushroom_count 
  (spotted : MushroomCount) 
  (gilled : MushroomCount) 
  (h1 : spotted.count + gilled.count = total_mushrooms) 
  (h2 : spotted.count = 9 * gilled.count) : 
  gilled.count = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gilled_mushroom_count_l92_9209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_at_745_l92_9293

noncomputable def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  let h : ℝ := hour
  let m : ℝ := minute
  |60 * h - 11 * m| / 2

noncomputable def smallest_angle (angle : ℝ) : ℝ :=
  min angle (360 - angle)

theorem smallest_angle_at_745 : 
  smallest_angle (clock_angle 7 45) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_at_745_l92_9293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_with_area_three_halves_l92_9204

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ
  points : List Point3D

/-- Represents a triangle -/
structure Triangle where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D

/-- Function to calculate the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- Function to check if a point is a vertex or midpoint of a cube -/
def isValidPoint (c : Cube) (p : Point3D) : Prop := sorry

/-- Function to count triangles with a specific area -/
noncomputable def countTrianglesWithArea (c : Cube) (area : ℝ) : ℕ := sorry

/-- Theorem stating the number of triangles with area 3/2 in the given cube -/
theorem triangles_with_area_three_halves (c : Cube) 
  (h1 : c.sideLength = 2)
  (h2 : c.points.length = 20)
  (h3 : ∀ p ∈ c.points, isValidPoint c p) :
  countTrianglesWithArea c (3/2) = 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_with_area_three_halves_l92_9204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l92_9298

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 2, 2]

theorem det_B_squared_minus_3B : Matrix.det (B^2 - 3 • B) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l92_9298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l92_9201

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2

noncomputable def point : ℝ × ℝ := (1, 1/2)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ 
      (y - point.2) = (deriv f point.1) * (x - point.1)) ∧
    a = 2 ∧ b = -2 ∧ c = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l92_9201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l92_9296

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_properties
  (f : ℝ → ℝ)
  (h_even : IsEven f)
  (h_neg : ∀ x, x ∈ Set.Icc 1 2 → f x < 0)
  (h_incr : ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → f x < f y) :
  (∀ x, x ∈ Set.Icc (-2) (-1) → f x < 0) ∧
  (∀ x y, x ∈ Set.Icc (-2) (-1) → y ∈ Set.Icc (-2) (-1) → x < y → f (-y) < f (-x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l92_9296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_theorem_l92_9226

structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0

def altitude (t : RightTriangle) (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2) = 0

def vector_sum_zero (t : RightTriangle) : Prop :=
  ∃ (D E F : ℝ × ℝ),
    altitude t D t.B t.C ∧
    altitude t E t.A t.C ∧
    altitude t F t.A t.B ∧
    6 * (D.1 - t.A.1, D.2 - t.A.2) +
    3 * (E.1 - t.B.1, E.2 - t.B.2) +
    2 * (F.1 - t.C.1, F.2 - t.C.2) = (0, 0)

theorem right_triangle_theorem (t : RightTriangle) (h : vector_sum_zero t) :
  (t.C.1 - t.A.1) * (t.B.1 - t.A.1) + (t.C.2 - t.A.2) * (t.B.2 - t.A.2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_theorem_l92_9226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_gk_divisible_by_3_is_9_14_l92_9277

def set_g : Finset ℕ := {3, 5, 7, 9, 11, 13, 8, 12}
def set_k : Finset ℕ := {2, 4, 6, 10, 7, 21, 9}

def divisible_by_3 (n : ℕ) : Bool := n % 3 = 0

def probability_gk_divisible_by_3 : ℚ :=
  (set_g.filter (λ x => divisible_by_3 x)).card * set_k.card +
  (set_g.filter (λ x => ¬(divisible_by_3 x))).card * (set_k.filter (λ x => divisible_by_3 x)).card /
  (set_g.card * set_k.card)

theorem probability_gk_divisible_by_3_is_9_14 :
  probability_gk_divisible_by_3 = 9 / 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_gk_divisible_by_3_is_9_14_l92_9277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_to_initial_state_probability_l92_9225

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Carol : Player
| Dave : Player

/-- Represents the state of the game as a function from Player to ℕ (their money) -/
def GameState : Type := Player → ℕ

/-- The initial state of the game where each player has 1 dollar -/
def initialState : GameState :=
  fun _ => 1

/-- A single turn of the game -/
def gameTurn (state : GameState) : GameState :=
  sorry  -- Implementation details omitted

/-- The probability of returning to the initial state after any number of turns -/
def returnProbability : ℚ := 8 / 81

theorem return_to_initial_state_probability :
  ∀ n : ℕ, (gameTurn^[n] initialState = initialState) ↔ returnProbability = 8 / 81 :=
sorry

#check return_to_initial_state_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_to_initial_state_probability_l92_9225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_halved_l92_9254

theorem prime_sum_halved (a b c : ℕ) 
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (h1 : Nat.Prime (Nat.factorial a + b + c))
  (h2 : Nat.Prime (Nat.factorial b + c + a))
  (h3 : Nat.Prime (Nat.factorial c + a + b)) :
  Nat.Prime ((a + b + c + 1) / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_halved_l92_9254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_sum_max_l92_9268

theorem parabola_vertex_sum_max (a T : ℤ) (h : T ≠ 0) : 
  let parabola (x y : ℚ) := ∃ b c : ℚ, y = a * x^2 + b * x + c;
  let point_on_parabola (x y : ℚ) := parabola x y;
  let N : ℚ := let x := (2 : ℚ) * T; x + (a * x * (x - 4 * T));
  (point_on_parabola 0 0) ∧ 
  (point_on_parabola (4 * T) 0) ∧ 
  (point_on_parabola (4 * T + 2) 32) →
  N ≤ 14 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_sum_max_l92_9268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l92_9275

theorem triangle_side_lengths 
  (A : ℝ) (a : ℝ) (S : ℝ) 
  (h1 : A = 120 * (π / 180)) 
  (h2 : a = Real.sqrt 21) 
  (h3 : S = Real.sqrt 3) : 
  ∃ b c : ℝ, (b = 4 ∧ c = 1) ∨ (b = 1 ∧ c = 4) := by
  sorry

#check triangle_side_lengths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l92_9275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_must_exceed_40mph_l92_9282

/-- The minimum speed Dan must exceed to arrive before Cara -/
noncomputable def minimum_speed_dan (distance : ℝ) (cara_speed : ℝ) (dan_delay : ℝ) : ℝ :=
  distance / (distance / cara_speed - dan_delay)

theorem dan_must_exceed_40mph 
  (distance : ℝ) 
  (cara_speed : ℝ) 
  (dan_delay : ℝ) 
  (h1 : distance = 120) 
  (h2 : cara_speed = 30) 
  (h3 : dan_delay = 1) : 
  minimum_speed_dan distance cara_speed dan_delay > 40 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval minimum_speed_dan 120 30 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_must_exceed_40mph_l92_9282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_with_arithmetic_progression_roots_l92_9272

/-- The roots of a cubic polynomial form an arithmetic progression if they can be written as r-d, r, r+d for some r and d. -/
def roots_form_arithmetic_progression (p : ℂ → ℂ) : Prop :=
  ∃ (r d : ℂ), (p r) = 0 ∧ (p (r-d)) = 0 ∧ (p (r+d)) = 0

/-- A polynomial has not all real roots if at least one of its roots is non-real. -/
def not_all_real_roots (p : ℂ → ℂ) : Prop :=
  ∃ (z : ℂ), (p z) = 0 ∧ z.im ≠ 0

/-- Extension of a real polynomial to the complex domain -/
def extend_to_complex (p : ℝ → ℝ) : ℂ → ℂ :=
  fun z => Complex.mk (p z.re) 0

theorem cubic_polynomial_with_arithmetic_progression_roots (a : ℝ) :
  roots_form_arithmetic_progression (extend_to_complex (fun x => x^3 - 9*x^2 + 42*x + a)) ∧
  not_all_real_roots (extend_to_complex (fun x => x^3 - 9*x^2 + 42*x + a)) →
  a = -72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_with_arithmetic_progression_roots_l92_9272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checker_arrangement_exists_l92_9285

/-- Represents a checker with a color -/
structure Checker :=
  (color : ℕ)

/-- Represents a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Option Checker

/-- Predicate to check if two checkers have different colors -/
def different_colors (c1 c2 : Checker) : Prop :=
  c1.color ≠ c2.color

/-- Predicate to check if a given arrangement satisfies the condition for every two-cell rectangle -/
def valid_arrangement (board : Chessboard) : Prop :=
  ∀ i j : Fin 8, ∀ d : Fin 2,
    (board i j).isSome ∧ (board (i + d) j).isSome →
    (∀ (h1 : (board i j).isSome) (h2 : (board (i + d) j).isSome),
      different_colors ((board i j).get h1) ((board (i + d) j).get h2)) ∧
    (board i j).isSome ∧ (board i (j + d)).isSome →
    (∀ (h1 : (board i j).isSome) (h2 : (board i (j + d)).isSome),
      different_colors ((board i j).get h1) ((board i (j + d)).get h2))

theorem checker_arrangement_exists :
  ∀ (checkers : Finset Checker),
    checkers.card = 64 →
    (∀ c1 c2, c1 ∈ checkers → c2 ∈ checkers → c1 ≠ c2 → different_colors c1 c2) →
    ∃ (board : Chessboard), valid_arrangement board ∧ 
      (∀ c, c ∈ checkers → ∃ i j : Fin 8, board i j = some c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_checker_arrangement_exists_l92_9285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_strips_odd_l92_9227

/-- Represents a paper strip covering two adjacent unit cells -/
structure Strip where
  cells : Fin 2 → Nat × Nat × Nat

/-- Represents the configuration of strips covering a cube -/
structure CubeConfig where
  size : Nat
  strips : List Strip

/-- Determines if a strip is folded (bent around an edge) -/
def Strip.isFolded (s : Strip) : Bool :=
  -- Implementation details omitted
  sorry

/-- Counts the number of folded strips in a cube configuration -/
def countFoldedStrips (config : CubeConfig) : Nat :=
  (config.strips.filter Strip.isFolded).length

/-- Checks if the strips cover the entire cube without overlaps -/
def CubeConfig.coversCube (config : CubeConfig) : Prop :=
  -- Implementation details omitted
  sorry

/-- Checks if all strips are aligned with the edges of the unit cells -/
def CubeConfig.stripsAligned (config : CubeConfig) : Prop :=
  -- Implementation details omitted
  sorry

theorem folded_strips_odd (config : CubeConfig) 
  (h_size : config.size = 9)
  (h_cover : CubeConfig.coversCube config)
  (h_align : CubeConfig.stripsAligned config) :
  Odd (countFoldedStrips config) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_strips_odd_l92_9227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_lower_limit_l92_9243

/-- Arun's weight in kg -/
def W : ℝ := sorry

/-- Lower limit of Arun's weight according to his own opinion -/
def L : ℝ := sorry

/-- Arun's weight is greater than L but less than 72 kg -/
axiom arun_opinion : L < W ∧ W < 72

/-- Arun's weight is greater than 60 kg but less than 70 kg (brother's opinion) -/
axiom brother_opinion : 60 < W ∧ W < 70

/-- Arun's weight is not greater than 69 kg (mother's opinion) -/
axiom mother_opinion : W ≤ 69

/-- The average of different probable weights of Arun is 68 kg -/
axiom average_weight : (L + 69) / 2 = 68

theorem arun_lower_limit : L = 67 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_lower_limit_l92_9243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_correct_l92_9278

noncomputable def equation (x : ℝ) : Prop :=
  3 * x / (x - 3) + (3 * x^2 - 27) / x = 14

noncomputable def smallest_solution : ℝ := (7 - Real.sqrt 76) / 3

theorem smallest_solution_is_correct :
  equation smallest_solution ∧
  ∀ y : ℝ, equation y → y ≥ smallest_solution :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_correct_l92_9278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_regular_octagon_area_is_option_d_l92_9208

/-- The area of a regular octagon given its longest and shortest diagonals -/
theorem regular_octagon_area (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  ∃ S : ℝ, S = a * b ∧ S > 0 := by
  use a * b
  constructor
  · rfl  -- reflexivity proves a * b = a * b
  · exact mul_pos h_pos_a h_pos_b

/-- The area of the regular octagon is option D (a * b) -/
theorem regular_octagon_area_is_option_d (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  ∃ S : ℝ, S = a * b ∧ S ≠ a^2 + b^2 ∧ S ≠ a^2 - b^2 ∧ S ≠ a + b := by
  obtain ⟨S, hS, hS_pos⟩ := regular_octagon_area a b h_pos_a h_pos_b
  use S
  constructor
  · exact hS
  constructor
  · intro h
    have : S > a^2 + b^2 := by
      rw [hS]
      sorry  -- This requires more involved algebra, so we use sorry for now
    linarith
  constructor
  · intro h
    have : S > a^2 - b^2 := by
      rw [hS]
      sorry  -- This also requires more involved algebra
    linarith
  · intro h
    have : S > a + b := by
      rw [hS]
      sorry  -- This too requires more involved algebra
    linarith

#check regular_octagon_area
#check regular_octagon_area_is_option_d

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_regular_octagon_area_is_option_d_l92_9208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_theorem_l92_9258

/-- Given a principal amount and an interest rate, calculates the simple interest for a given number of years -/
def simpleInterest (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * rate * years

/-- Given a principal amount and an interest rate, calculates the compound interest for a given number of years -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- Theorem stating that if the simple interest for 2 years is $50 and the compound interest for 2 years is $51.25, then the annual interest rate is 5% -/
theorem interest_rate_theorem (P : ℝ) (r : ℝ) 
    (h1 : simpleInterest P r 2 = 50) 
    (h2 : compoundInterest P r 2 = 51.25) : 
    r = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_theorem_l92_9258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_height_specific_l92_9259

noncomputable def right_pyramid_height (base_perimeter : ℝ) (apex_to_vertex : ℝ) : ℝ :=
  let side_length := base_perimeter / 4
  let half_diagonal := (Real.sqrt 2 * side_length) / 2
  Real.sqrt (apex_to_vertex^2 - half_diagonal^2)

theorem right_pyramid_height_specific :
  right_pyramid_height 32 12 = 4 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_height_specific_l92_9259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_equation_solution_l92_9257

theorem arcsin_equation_solution :
  ∃ x : ℝ, x = 0 ∧ Real.arcsin x + Real.arcsin (1 - 2*x) = Real.arccos (2*x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_equation_solution_l92_9257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_for_three_zeros_l92_9239

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 2 then 2^x - 4
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (-x^2 + 2*x)
  else 0  -- We define f as 0 outside its specified domain for completeness

-- Define F in terms of f and k
noncomputable def F (k : ℝ) (x : ℝ) : ℝ := f x - k*x - 3*k

-- State the theorem
theorem range_of_k_for_three_zeros :
  ∃ k_min k_max : ℝ, k_min = 0 ∧ k_max = Real.sqrt 15 / 15 ∧
  ∀ k : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    F k x₁ = 0 ∧ F k x₂ = 0 ∧ F k x₃ = 0) ↔ k_min < k ∧ k < k_max :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_for_three_zeros_l92_9239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l92_9221

theorem problem_solution :
  ∀ x y z : ℤ,
  x ≥ y → y ≥ z →
  x > 0 → y > 0 → z > 0 →
  x^2 - y^2 - z^2 + x*y = 1005 →
  x^2 + 2*y^2 + 2*z^2 - 2*x*y - x*z - y*z = -995 →
  Even x →
  x = 505 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l92_9221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amc_10_2007_scoring_l92_9211

theorem amc_10_2007_scoring (total_problems : Nat) (correct_points : Nat) (incorrect_points : Nat) 
  (unanswered_points : Nat) (attempted_problems : Nat) (unanswered_problems : Nat) 
  (target_score : Nat) :
  total_problems = 25 →
  correct_points = 7 →
  incorrect_points = 0 →
  unanswered_points = 2 →
  attempted_problems = 20 →
  unanswered_problems = 5 →
  target_score = 120 →
  (∃ x : Nat, x * correct_points + (attempted_problems - x) * incorrect_points + 
    unanswered_problems * unanswered_points ≥ target_score ∧
    ∀ y : Nat, y < x → y * correct_points + (attempted_problems - y) * incorrect_points + 
    unanswered_problems * unanswered_points < target_score) →
  16 * correct_points + (attempted_problems - 16) * incorrect_points + 
    unanswered_problems * unanswered_points ≥ target_score :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amc_10_2007_scoring_l92_9211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l92_9210

noncomputable def circle_area_problem (r : ℝ) : Prop :=
  let larger_radius := 4 * r
  let tangent_length := 5
  let smaller_circle_area := Real.pi * r^2
  (r > 0) ∧
  (larger_radius > r) ∧
  (tangent_length > 0) ∧
  (r^2 + tangent_length^2 = (3*r)^2) ∧
  (smaller_circle_area = 25 * Real.pi / 8)

theorem circle_area_theorem :
  ∃ r : ℝ, circle_area_problem r := by
  sorry

#check circle_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l92_9210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_percentage_for_90_percent_l92_9240

/-- Represents the percentage of people -/
def P (x : ℝ) : Prop := x ≥ 0 ∧ x ≤ 100

/-- Represents the percentage of money owned by a group of people -/
def M (x y : ℝ) : Prop := x ≥ 0 ∧ x ≤ 100 ∧ y ≥ 0 ∧ y ≤ 100

/-- 20% of people own at least 80% of all money -/
axiom wealth_distribution : M 20 80

/-- The relationship between percentage of people and their wealth -/
axiom wealth_relation (x y : ℝ) : 
  P x → M x y → P (100 - x) → M (100 - x) (100 - y)

/-- The sum of all percentages should be 100% -/
axiom percentage_sum (x y : ℝ) : P x → P y → x + y = 100

/-- Theorem: The smallest percentage of people that can be guaranteed to own 90% of all money is 60% -/
theorem min_percentage_for_90_percent : 
  (∃ (x : ℝ), P x ∧ M x 90) ∧ (∀ (y : ℝ), y < 60 → ¬(P y ∧ M y 90)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_percentage_for_90_percent_l92_9240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_moves_exist_l92_9246

-- Define the car type and direction
structure Car where
  position : ℕ × ℕ
  direction : Direction

inductive Direction
  | North
  | South
  | East
  | West

-- Define the grid and state types
def Grid := ℕ × ℕ → Option Car
structure GridState where
  grid : Grid
  cars : Finset Car

-- Define a valid move
def ValidMove (state : GridState) (car : Car) : Prop :=
  ∃ (new_pos : ℕ × ℕ),
    state.grid new_pos = none ∧
    (car.direction = Direction.North → new_pos.2 = car.position.2 + 1) ∧
    (car.direction = Direction.South → new_pos.2 = car.position.2 - 1) ∧
    (car.direction = Direction.East  → new_pos.1 = car.position.1 + 1) ∧
    (car.direction = Direction.West  → new_pos.1 = car.position.1 - 1)

-- Define the conditions of the problem
def ValidGridState (state : GridState) : Prop :=
  (∀ c₁ c₂ : Car, c₁ ∈ state.cars → c₂ ∈ state.cars → c₁.position = c₂.position → c₁ = c₂) ∧
  (∀ c : Car, c ∈ state.cars → ValidMove state c) ∧
  (∀ c₁ c₂ : Car, c₁ ∈ state.cars → c₂ ∈ state.cars →
    (c₁.direction = Direction.East ∧ c₂.direction = Direction.West → c₁.position.1 > c₂.position.1) ∧
    (c₁.direction = Direction.North ∧ c₂.direction = Direction.South → c₁.position.2 > c₂.position.2))

-- Define an infinite sequence of moves
def InfiniteSequence (state : GridState) : Prop :=
  ∃ (seq : ℕ → Car), 
    (∀ n : ℕ, seq n ∈ state.cars) ∧
    (∀ c : Car, c ∈ state.cars → ∀ m : ℕ, ∃ n > m, seq n = c) ∧
    (∀ n : ℕ, ValidMove state (seq n))

-- The main theorem
theorem infinite_moves_exist (state : GridState) (h : ValidGridState state) :
  InfiniteSequence state :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_moves_exist_l92_9246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l92_9292

theorem power_equality (y : ℝ) (h : (8 : ℝ)^y - (8 : ℝ)^(y - 1) = 112) : 
  (3*y)^y = (7^(1/3 : ℝ))^7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l92_9292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_tax_rate_problem_l92_9231

/-- Calculates the original sales tax percentage given the market price, new tax rate, and savings. -/
noncomputable def original_tax_rate (market_price : ℝ) (new_tax_rate : ℝ) (savings : ℝ) : ℝ :=
  let new_tax_amount := market_price * new_tax_rate / 100
  let original_tax_amount := new_tax_amount + savings
  (original_tax_amount / market_price) * 100

/-- Theorem stating that given the specified conditions, the original tax rate was 3.5% -/
theorem original_tax_rate_problem :
  let market_price : ℝ := 10800
  let new_tax_rate : ℝ := 10/3  -- 3 1/3%
  let savings : ℝ := 18
  original_tax_rate market_price new_tax_rate savings = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_tax_rate_problem_l92_9231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_song_preference_count_l92_9279

/-- Represents a girl --/
inductive Girl
  | Amy
  | Beth
  | Jo
  | Meg

/-- Represents a song preference configuration --/
def SongPreference := Girl → Fin 6 → Bool

/-- Checks if a song is liked by all girls --/
def allLike (pref : SongPreference) (song : Fin 6) : Prop :=
  ∀ g : Girl, pref g song = true

/-- Checks if a song is liked by exactly three out of four girls --/
def threeOutOfFourLike (pref : SongPreference) (song : Fin 6) : Prop :=
  ∃ g : Girl, (∀ g' : Girl, g' ≠ g → pref g' song = true) ∧ pref g song = false

/-- Checks if the preference configuration satisfies all conditions --/
def validPreference (pref : SongPreference) : Prop :=
  (∀ song : Fin 6, ¬(allLike pref song)) ∧
  (∀ g : Girl, ∃ song : Fin 6, threeOutOfFourLike pref song ∧ pref g song = false)

/-- The number of valid preference configurations --/
def numValidPreferences : ℕ := sorry

theorem song_preference_count : numValidPreferences = 1440 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_song_preference_count_l92_9279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_target_l92_9281

noncomputable def initial_height : ℝ := 2000
noncomputable def bounce_ratio : ℝ := 1/2
noncomputable def target_height : ℝ := 0.5

noncomputable def height_after_bounces (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

def reaches_target (n : ℕ) : Prop :=
  height_after_bounces n < target_height

theorem min_bounces_to_target :
  ∃ (k : ℕ), reaches_target k ∧ ∀ (m : ℕ), m < k → ¬reaches_target m :=
by sorry

#eval Nat.log 2 4000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_target_l92_9281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_difference_l92_9230

variable (a m : ℝ)

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*(a+m)*x + a^2

noncomputable def g (x : ℝ) : ℝ := -x^2 + 2*(a-m)*x - a^2 + 2*m^2

noncomputable def H₁ (x : ℝ) : ℝ := max (f a m x) (g a m x)

noncomputable def H₂ (x : ℝ) : ℝ := min (f a m x) (g a m x)

theorem min_max_difference (a m : ℝ) :
  ∃ A B : ℝ, (∀ x : ℝ, A ≤ H₁ a m x) ∧ 
            (∀ x : ℝ, H₂ a m x ≤ B) ∧ 
            A - B = -4*m^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_difference_l92_9230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_each_person_runs_five_miles_l92_9267

/-- The number of people on the sprint team -/
noncomputable def team_size : ℚ := 150

/-- The total number of miles run by the team -/
noncomputable def total_miles : ℚ := 750

/-- The number of miles each person runs -/
noncomputable def miles_per_person : ℚ := total_miles / team_size

/-- Theorem stating that each person runs 5 miles -/
theorem each_person_runs_five_miles : miles_per_person = 5 := by
  unfold miles_per_person total_miles team_size
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_each_person_runs_five_miles_l92_9267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_complement_A_union_A_complement_B_l92_9295

-- Define the universal set U
def U : Set ℝ := {x : ℝ | x^2 ≤ 25}

-- Define set A
def A : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

-- Theorem for complement of A in U
theorem complement_A : (U \ A) = {x : ℝ | (-5 ≤ x ∧ x ≤ 0) ∨ (3 < x ∧ x ≤ 5)} := by sorry

-- Theorem for A ∪ (complement of B in U)
theorem union_A_complement_B : A ∪ (U \ B) = {x : ℝ | (-5 ≤ x ∧ x < -2) ∨ (0 < x ∧ x ≤ 5)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_complement_A_union_A_complement_B_l92_9295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_roots_m_range_l92_9269

-- Define the piecewise function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then
    2 * x^3 + 3 * x^2 + m
  else if x > 1 then
    m * x + 5
  else
    0  -- Undefined for x < 0, but we set it to 0 for completeness

-- Theorem statement
theorem f_two_roots_m_range (m : ℝ) :
  (∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧ 
   (∀ z, z ≠ x ∧ z ≠ y → f m z ≠ 0)) →
  m > -5 ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_roots_m_range_l92_9269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_determination_of_T_largest_n_for_unique_T_l92_9222

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a₁ : ℚ  -- First term
  d : ℚ   -- Common difference

/-- Sum of first n terms of an arithmetic sequence -/
def Sₙ (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a₁ + (n - 1) * seq.d) / 2

/-- Sum of first n sums of an arithmetic sequence -/
noncomputable def Tₙ (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (fun k => Sₙ seq (k + 1))

/-- The theorem stating that given S₁₀₀₀, T₁₅₀₁ can be uniquely determined -/
theorem unique_determination_of_T (seq : ArithmeticSequence) (S₁₀₀₀ : ℚ) :
  ∃! (T : ℚ), T = Tₙ seq 1501 :=
sorry

/-- The theorem stating that 1501 is the largest n for which Tₙ can be uniquely determined -/
theorem largest_n_for_unique_T (seq : ArithmeticSequence) (S₁₀₀₀ : ℚ) (n : ℕ) :
  (∃! (T : ℚ), T = Tₙ seq n) → n ≤ 1501 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_determination_of_T_largest_n_for_unique_T_l92_9222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_is_two_l92_9299

/-- Expresses a number as a ratio of factorials -/
def factorial_ratio (n : ℕ) (a₁ a₂ b₁ b₂ : ℕ) : Prop :=
  n * (b₁.factorial * b₂.factorial) = a₁.factorial * a₂.factorial

/-- The conditions for the factorial ratio representation -/
def valid_representation (n a₁ a₂ b₁ b₂ : ℕ) : Prop :=
  factorial_ratio n a₁ a₂ b₁ b₂ ∧ a₁ ≥ a₂ ∧ b₁ ≥ b₂

/-- The sum a₁ + b₁ is minimal among all valid representations -/
def minimal_sum (n a₁ a₂ b₁ b₂ : ℕ) : Prop :=
  valid_representation n a₁ a₂ b₁ b₂ ∧
  ∀ a₁' a₂' b₁' b₂', valid_representation n a₁' a₂' b₁' b₂' →
    a₁ + b₁ ≤ a₁' + b₁'

theorem smallest_difference_is_two :
  ∃ a₁ a₂ b₁ b₂ : ℕ,
    minimal_sum 2100 a₁ a₂ b₁ b₂ ∧
    (if a₁ ≥ b₁ then a₁ - b₁ else b₁ - a₁) = 2 ∧
    ∀ a₁' a₂' b₁' b₂', minimal_sum 2100 a₁' a₂' b₁' b₂' →
      (if a₁' ≥ b₁' then a₁' - b₁' else b₁' - a₁') ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_is_two_l92_9299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_sending_theorem_l92_9203

-- Define a simple graph
structure Graph (V : Type) where
  adj : V → V → Prop

-- Define a tree (connected graph with n vertices and n-1 edges)
def IsTree {V : Type} (G : Graph V) (n : ℕ) : Prop :=
  (∃ (vertices : Finset V), vertices.card = n) ∧
  (∀ u v : V, ∃ (path : List V), path.head? = some u ∧ path.getLast? = some v)

-- Define the letter sending function
def LetterFunction {V : Type} (G : Graph V) (f : V → V) : Prop :=
  ∀ u v : V, G.adj u v → (f u = f v ∨ G.adj (f u) (f v))

-- Theorem statement
theorem letter_sending_theorem {V : Type} (G : Graph V) (n : ℕ) (f : V → V) :
  IsTree G n → LetterFunction G f →
  (∃ v : V, f v = v) ∨ (∃ u v : V, G.adj u v ∧ f u = v ∧ f v = u) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_sending_theorem_l92_9203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_r_pi_15_root_l92_9256

theorem tan_r_pi_15_root (r : ℕ) (hr1 : r > 0) (hr2 : r < 15) (hr3 : Nat.Coprime r 15) :
  (Real.tan (r * Real.pi / 15))^8 - 92 * (Real.tan (r * Real.pi / 15))^6 + 
  134 * (Real.tan (r * Real.pi / 15))^4 - 28 * (Real.tan (r * Real.pi / 15))^2 + 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_r_pi_15_root_l92_9256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_parallel_l92_9255

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x^2 else -2 * x^2

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem tangent_lines_parallel :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
  (∀ (x : ℝ), f (-x) = -f x) ∧ 
  (deriv f x₀ = deriv g x₀) ∧
  x₀ = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_parallel_l92_9255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_photo_arrangements_l92_9234

/-- The number of students in the group photo. -/
def num_students : ℕ := 5

/-- The number of teachers in the group photo. -/
def num_teachers : ℕ := 2

/-- The total number of people in the group photo. -/
def total_people : ℕ := num_students + num_teachers

/-- The number of ways to arrange the group for a photo, given the constraints. -/
def seating_arrangements : ℕ := 960

/-- Represents a person in the group photo, either a student or a teacher. -/
inductive Person
  | Student
  | Teacher

/-- Predicate to check if teachers are sitting together in an arrangement. -/
def teachers_together (arrangement : List Person) : Prop :=
  sorry

/-- Predicate to check if teachers are not sitting at the ends in an arrangement. -/
def teachers_not_at_ends (arrangement : List Person) : Prop :=
  sorry

/-- Theorem stating that the number of seating arrangements for the group photo,
    where teachers must sit together and not at the ends, is 960. -/
theorem group_photo_arrangements :
  (num_students = 5) →
  (num_teachers = 2) →
  (total_people = num_students + num_teachers) →
  (∀ arrangement : List Person, 
    (teachers_together arrangement) ∧ 
    (teachers_not_at_ends arrangement)) →
  seating_arrangements = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_photo_arrangements_l92_9234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_625_representation_l92_9265

def is_four_digit (b : ℕ) (n : ℕ) : Prop :=
  b^3 ≤ n ∧ n < b^4

def final_digit (b : ℕ) (n : ℕ) : ℕ :=
  n % b

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem base_625_representation (b : ℕ) (h : b = 6 ∨ b = 7 ∨ b = 8) :
  is_four_digit b 625 ∧ is_odd (final_digit b 625) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_625_representation_l92_9265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_sum_l92_9273

/-- The sum of coordinates of a point in 2D space. -/
def sum_coordinates (p : ℝ × ℝ) : ℝ := p.1 + p.2

theorem midpoint_coordinates_sum :
  let c : ℝ × ℝ := (-6, 2)
  let m : ℝ × ℝ := (-2, 10)
  ∀ d : ℝ × ℝ, ((c.1 + d.1) / 2, (c.2 + d.2) / 2) = m → sum_coordinates d = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_sum_l92_9273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_coverage_l92_9237

/-- An L-triomino covers exactly three squares -/
structure LTriomino where
  covers : Fin 3 → Nat × Nat

/-- Chessboard with alternating colors and black corners -/
def Chessboard (n : Nat) : Type :=
  Σ' (i j : Nat), i < n ∧ j < n

/-- A square is black if the sum of its coordinates is even -/
def isBlack (square : Nat × Nat) : Prop :=
  (square.1 + square.2) % 2 = 0

/-- The number of black squares on an n × n chessboard -/
def blackSquaresCount (n : Nat) : Nat :=
  ((n + 1) / 2) ^ 2

/-- A valid placement of L-triominoes on the chessboard -/
def validPlacement (n : Nat) (placement : List LTriomino) : Prop :=
  (∀ t ∈ placement, ∀ s, (t.covers s).1 < n ∧ (t.covers s).2 < n) ∧
  (∀ t1 ∈ placement, ∀ t2 ∈ placement, t1 ≠ t2 → ∀ s1 s2, t1.covers s1 ≠ t2.covers s2) ∧
  (∀ i j, i < n → j < n → isBlack (i, j) →
    ∃ t ∈ placement, ∃ s, t.covers s = (i, j))

theorem chessboard_coverage (n : Nat) (h1 : n > 1) (h2 : Odd n) (h3 : n ≥ 7) :
  ∃ placement : List LTriomino,
    validPlacement n placement ∧ placement.length = blackSquaresCount n := by
  sorry

#check chessboard_coverage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_coverage_l92_9237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos6_l92_9280

theorem min_sin6_plus_2cos6 :
  ∀ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 2/3 ∧
  ∃ y : ℝ, Real.sin y ^ 6 + 2 * Real.cos y ^ 6 = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos6_l92_9280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_integers_digits_sum_l92_9253

def count_digits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

def sum_digits_even (n : ℕ) : ℕ :=
  (List.range n).filter (λ x => x % 2 = 0 ∧ x > 0)
    |>.map count_digits
    |>.sum

theorem even_integers_digits_sum :
  sum_digits_even 5001 = 9444 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_integers_digits_sum_l92_9253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_weight_probability_l92_9205

/-- The cumulative distribution function of the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ := 
  1 / (σ * Real.sqrt (2 * Real.pi)) * Real.exp (-(x - μ)^2 / (2 * σ^2))

theorem rice_weight_probability :
  let μ : ℝ := 20
  let σ : ℝ := 0.2
  let a : ℝ := 19.6
  let b : ℝ := 20.4
  (∫ x in a..b, normal_pdf μ σ x) = 0.9544 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_weight_probability_l92_9205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_triangle_area_is_correct_l92_9219

/-- The area of the gray triangle formed by bending a corner of a 12 by 15 rectangle -/
noncomputable def gray_triangle_area (rectangle_width : ℝ) (rectangle_height : ℝ) : ℝ :=
  let larger_triangle_leg := Real.sqrt (rectangle_height^2 - rectangle_width^2)
  let smaller_triangle_leg := (rectangle_width * larger_triangle_leg) / rectangle_height
  (smaller_triangle_leg * rectangle_height) / 2

/-- Theorem stating that the area of the gray triangle is 135/4 -/
theorem gray_triangle_area_is_correct :
  gray_triangle_area 12 15 = 135 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_triangle_area_is_correct_l92_9219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l92_9213

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the dot product of PF1 and PF2
noncomputable def dot_product (x y : ℝ) : ℝ :=
  (x - F1.1) * (x - F2.1) + (y - F1.2) * (y - F2.2)

-- Theorem statement
theorem dot_product_range :
  ∀ x y : ℝ, is_on_ellipse x y → -2 ≤ dot_product x y ∧ dot_product x y ≤ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l92_9213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_march_first_friday_l92_9206

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Repr, DecidableEq

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Counts the number of occurrences of a specific day in a sequence of days -/
def countDays (startDay : DayOfWeek) (numDays : Nat) (targetDay : DayOfWeek) : Nat :=
  let rec count (currentDay : DayOfWeek) (daysLeft : Nat) (acc : Nat) : Nat :=
    if daysLeft = 0 then acc
    else count (nextDay currentDay) (daysLeft - 1) (if currentDay = targetDay then acc + 1 else acc)
  count startDay numDays 0

theorem march_first_friday 
  (h1 : countDays DayOfWeek.Friday 31 DayOfWeek.Monday = 4)
  (h2 : countDays DayOfWeek.Friday 31 DayOfWeek.Thursday = 4) :
  DayOfWeek.Friday = DayOfWeek.Friday :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_march_first_friday_l92_9206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_2b_plus_12_l92_9224

theorem divisibility_of_2b_plus_12 (a b : ℤ) (h : 3 * b = 8 - 2 * a) :
  (∀ n : ℕ, n ≤ 6 → (n ∣ Int.natAbs (2 * b + 12) ↔ n = 1 ∨ n = 2 ∨ n = 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_2b_plus_12_l92_9224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l92_9215

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Theorem statement
theorem problem_solution (y : ℝ) (h : y = 2) :
  (floor 6.5) * (floor (2 / 3 : ℝ)) + (floor y) * (7.2 : ℝ) + (floor 8.4) - (6 : ℝ) = 16.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l92_9215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_m_eq_neg_one_l92_9297

/-- A function f is even if f(-x) = f(x) for all x in its domain. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x(10^x + m⋅10^(-x)) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  x * (10^x + m * 10^(-x))

/-- If f(x) = x(10^x + m⋅10^(-x)) is an even function, then m = -1 -/
theorem even_function_implies_m_eq_neg_one (m : ℝ) :
  IsEven (f m) → m = -1 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_m_eq_neg_one_l92_9297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l92_9241

theorem sin_cos_product (α : ℝ) (h : Real.sin α + Real.cos α = 1/2) : 
  Real.sin α * Real.cos α = -3/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l92_9241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_l92_9287

noncomputable def original_curve (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def curve_C (x : ℝ) : ℝ := original_curve (x - 3) - 2

noncomputable def symmetric_curve (x : ℝ) : ℝ := -(2^(2 - x)) - 1

theorem curve_symmetry :
  ∀ (x y : ℝ), curve_C x = y ↔ symmetric_curve (-y) = -x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_l92_9287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_equilateral_triangle_axial_section_area_l92_9252

/-- An equilateral triangle with side length a -/
def Equilateral_triangle (a : ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- A line passing through the center of the triangle and parallel to one of its sides -/
def Line_through_center_parallel_to_side (triangle : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- The body obtained by rotating the triangle around the specified axis -/
def Rotate (triangle : Set (ℝ × ℝ)) (axis : Set (ℝ × ℝ)) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The area of the axial cross-section of a 3D body -/
def Axial_cross_section_area (body : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- The area of the axial cross-section of a rotated equilateral triangle -/
theorem rotated_equilateral_triangle_axial_section_area (a : ℝ) (h : a > 0) :
  let triangle := Equilateral_triangle a
  let rotation_axis := Line_through_center_parallel_to_side triangle
  let rotated_body := Rotate triangle rotation_axis
  Axial_cross_section_area rotated_body = (a^2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_equilateral_triangle_axial_section_area_l92_9252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l92_9223

theorem trigonometric_identity (θ a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + 2*b) →
  (Real.sin θ)^12 / a^3 + (Real.cos θ)^12 / b^3 = (a^3 + b^3) / (a + 2*b)^6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l92_9223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_OA_OB_l92_9236

-- Define the curve C and line l in polar coordinates
noncomputable def curve_C (a : ℝ) (θ : ℝ) : ℝ := 2 * a * Real.cos θ
noncomputable def line_l (θ : ℝ) : ℝ := (3/2) / Real.cos (θ - Real.pi/3)

-- Define the theorem
theorem max_sum_OA_OB :
  ∀ (a : ℝ), a > 0 →
  (∃! p : ℝ × ℝ, curve_C a p.1 = line_l p.1) →
  (∀ θ : ℝ, (|curve_C a θ| + |curve_C a (θ + Real.pi/3)|) ≤ 2 * Real.sqrt 3) ∧
  (∃ θ : ℝ, |curve_C a θ| + |curve_C a (θ + Real.pi/3)| = 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_OA_OB_l92_9236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_polar_sin_curve_l92_9229

open Real

/-- The set of points (r cos θ, r sin θ) where r = sin θ and 0 ≤ θ ≤ t -/
def polarSinCurve (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ p = ((sin θ) * (cos θ), (sin θ) * (sin θ))}

/-- The unit circle -/
def unitCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Theorem: π is the smallest positive real number t such that 
    the polar sin curve for 0 ≤ θ ≤ t forms a complete circle -/
theorem smallest_complete_polar_sin_curve :
  ∀ t : ℝ, t > 0 → (polarSinCurve t = unitCircle ↔ t ≥ π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_polar_sin_curve_l92_9229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_of_gain_l92_9270

/-- Represents the investment of a partner -/
structure Investment where
  amount : ℚ
  duration : ℚ

/-- Calculates the investment ratio based on amount and duration -/
def investmentRatio (i : Investment) : ℚ := i.amount * i.duration

/-- Calculates the share of annual gain based on investment ratio and total ratio -/
def shareOfGain (individualRatio totalRatio annualGain : ℚ) : ℚ :=
  (individualRatio / totalRatio) * annualGain

theorem a_share_of_gain (x : ℚ) (annualGain : ℚ) :
  let a : Investment := ⟨x, 12⟩
  let b : Investment := ⟨2*x, 6⟩
  let c : Investment := ⟨3*x, 4⟩
  let totalRatio := investmentRatio a + investmentRatio b + investmentRatio c
  annualGain = 15000 →
  shareOfGain (investmentRatio a) totalRatio annualGain = (6/11) * 15000 := by
  intro h
  simp [Investment, investmentRatio, shareOfGain]
  -- The actual proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_of_gain_l92_9270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l92_9217

def sequence_property (a : ℕ → ℝ) (A B : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = A * a (n + 1) + B * a n

def is_periodic (a : ℕ → ℝ) (period : ℕ) : Prop :=
  ∀ n : ℕ, a (n + period) = a n

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem sequence_properties (a : ℕ → ℝ) (A B : ℝ) 
    (h : sequence_property a A B) (hA : A ≠ 0) (hB : B ≠ 0) :
  (A = 1 ∧ B = -1 → is_periodic a 6) ∧
  (A = 3 ∧ B = -2 → is_geometric (λ n ↦ a (n + 1) - a n)) ∧
  (A > 0 ∧ B > 1 ∧ A + 1 = B ∧ a 1 = 0 ∧ a 2 = B → is_increasing (λ n ↦ a (2 * n))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l92_9217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_condition_l92_9263

theorem no_function_satisfies_condition : 
  ¬∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x^2 - 1996 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_condition_l92_9263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l92_9276

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem omega_range (ω : ℝ) :
  ω > 0 →
  (∀ x y, -2*π/3 ≤ x ∧ x < y ∧ y ≤ π/3 → f ω x < f ω y) →
  (∃! x, x ∈ Set.Icc 0 π ∧ |f ω x| = 1) →
  1/2 ≤ ω ∧ ω ≤ 3/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l92_9276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l92_9220

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem range_of_x (h : Set.range f = Set.Ici 4) :
  {x : ℝ | ∃ y, f y = x} = Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l92_9220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_CF_length_l92_9286

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define a custom membership relation for Point and Circle
def pointInCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

-- Define the problem setup
def problem_setup (circle1 circle2 : Circle) (A B C D F : Point) : Prop :=
  -- Two circles with radius 7
  circle1.radius = 7 ∧ circle2.radius = 7 ∧
  -- A and B are intersection points of the circles
  pointInCircle A circle1 ∧ pointInCircle A circle2 ∧ 
  pointInCircle B circle1 ∧ pointInCircle B circle2 ∧
  -- C is on the first circle, D is on the second circle
  pointInCircle C circle1 ∧ pointInCircle D circle2 ∧
  -- B lies on segment CD
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B.x = t * C.x + (1 - t) * D.x ∧ B.y = t * C.y + (1 - t) * D.y) ∧
  -- ∠CAD = 90°
  ((C.x - A.x) * (D.x - A.x) + (C.y - A.y) * (D.y - A.y) = 0) ∧
  -- F is on the perpendicular to CD passing through B
  ((D.x - C.x) * (F.x - B.x) + (D.y - C.y) * (F.y - B.y) = 0) ∧
  -- BF = BD
  ((F.x - B.x)^2 + (F.y - B.y)^2 = (D.x - B.x)^2 + (D.y - B.y)^2) ∧
  -- A and F are on opposite sides of line CD
  ((C.x - D.x) * (A.y - D.y) - (C.y - D.y) * (A.x - D.x)) *
  ((C.x - D.x) * (F.y - D.y) - (C.y - D.y) * (F.x - D.x)) < 0

-- Theorem statement
theorem segment_CF_length 
  (circle1 circle2 : Circle) (A B C D F : Point) 
  (h : problem_setup circle1 circle2 A B C D F) : 
  Real.sqrt ((C.x - F.x)^2 + (C.y - F.y)^2) = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_CF_length_l92_9286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_square_is_pi_over_four_l92_9250

/-- The area inside a circle of radius 1 but outside a square of side length 2,
    where the circle is tangent to one side of the square at its midpoint. -/
noncomputable def areaOutsideSquare : ℝ := Real.pi / 4

/-- Proof that the area inside the circle but outside the square is π/4 -/
theorem area_outside_square_is_pi_over_four :
  let squareSide : ℝ := 2
  let circleRadius : ℝ := 1
  areaOutsideSquare = Real.pi / 4 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_square_is_pi_over_four_l92_9250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_land_division_theorem_l92_9214

/-- Represents a square plot of land -/
structure LandPlot where
  side : ℝ
  trees : ℕ
  tree_positions : Finset (ℝ × ℝ)

/-- Represents a division of the land -/
structure LandDivision where
  parts : ℕ
  trees_per_part : ℕ

theorem land_division_theorem (land : LandPlot) (div : LandDivision) :
  land.trees = 24 →
  div.parts = 8 →
  div.trees_per_part = 3 →
  ∃ (partition : Finset (Finset (ℝ × ℝ))),
    partition.card = div.parts ∧
    ∀ part : Finset (ℝ × ℝ), part ∈ partition →
      (part.card = div.trees_per_part) ∧
      (∀ (x y : ℝ), (x, y) ∈ part → x ≥ 0 ∧ x ≤ land.side ∧ y ≥ 0 ∧ y ≤ land.side) ∧
      (∀ other_part : Finset (ℝ × ℝ), other_part ∈ partition → other_part ≠ part → part ∩ other_part = ∅) ∧
      (∀ p q : Finset (ℝ × ℝ), p ∈ partition → q ∈ partition → p.card = q.card) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_land_division_theorem_l92_9214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_cross_section_l92_9244

-- Define a triangular prism
structure TriangularPrism where
  -- We'll leave this empty for now, as we don't need specific fields for this proof
  mk :: -- Empty constructor

-- Define a plane
structure Plane where
  -- We'll leave this empty for now, as we don't need specific fields for this proof
  mk :: -- Empty constructor

-- Define the possible cross-section shapes
inductive CrossSectionShape
  | Triangle
  | Trapezoid

-- Define the function that determines the cross-section shape
def crossSectionShape (prism : TriangularPrism) (plane : Plane) : CrossSectionShape :=
  sorry

-- Define a predicate to check if a plane passes through a base edge
def passesThoughBaseEdge (plane : Plane) (prism : TriangularPrism) : Prop :=
  sorry

-- Theorem statement
theorem triangular_prism_cross_section 
  (prism : TriangularPrism) 
  (plane : Plane) 
  (h : passesThoughBaseEdge plane prism) :
  (crossSectionShape prism plane = CrossSectionShape.Triangle) ∨ 
  (crossSectionShape prism plane = CrossSectionShape.Trapezoid) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_cross_section_l92_9244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_value_l92_9294

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_value (a b c : ℝ) :
  (∀ x, quadratic a b c x ≥ -3) ∧  -- minimum value is -3
  (quadratic a b c (-1) = -3) ∧   -- minimum occurs at x = -1
  (quadratic a b c 1 = 7) →       -- passes through (1, 7)
  quadratic a b c 3 = 37 :=        -- value at x = 3 is 37
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_value_l92_9294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_union_theorem_l92_9264

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2*x + 3) / Real.log 10

noncomputable def g (x : ℝ) : ℝ := 1 / (x^2 - 2*x + 2)

def A : Set ℝ := {x : ℝ | ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f x < f y}

def B : Set ℝ := Set.range g

def C (a : ℝ) : Set ℝ := Set.Icc (a - 1) (2 - a)

theorem intersection_theorem : A ∩ B = Set.Ioo 0 1 := by sorry

theorem union_theorem : ∀ a : ℝ, A ∪ C a = A → a > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_union_theorem_l92_9264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_25_l92_9245

theorem count_multiples_of_25 : 
  (List.range ((125 - 25) / 25 + 1)).length = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_25_l92_9245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_part_zero_l92_9290

theorem complex_real_part_zero (a : ℝ) : 
  (((a + 2 * Complex.I) * (1 + Complex.I)).re = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_part_zero_l92_9290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l92_9283

/-- The angle rotated by the minute hand of a clock after 40 minutes -/
def minute_hand_rotation (full_rotation : ℝ) (fraction_of_hour : ℝ) : ℝ :=
  let clockwise_full_rotation := -full_rotation
  let rotation_after_40_minutes := clockwise_full_rotation * fraction_of_hour
  rotation_after_40_minutes

/-- Main theorem -/
theorem main_theorem : minute_hand_rotation (2 * Real.pi) (2 / 3) = -(4 / 3 * Real.pi) := by
  -- Unfold the definition of minute_hand_rotation
  unfold minute_hand_rotation
  -- Simplify the expression
  simp
  -- Perform algebraic manipulations
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l92_9283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l92_9251

theorem log_problem (x : ℝ) (h1 : x > 1) (h2 : (Real.log x)^2 - Real.log (x^2) = 18 * Real.log 10) :
  (Real.log x)^3 - Real.log (x^3) = 198 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l92_9251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ratios_is_14_l92_9248

/-- Given positive real numbers x and y, and an angle θ not equal to π/2n for any integer n,
    if sin(θ)/x = cos(θ)/y and cos^4(θ)/x^4 + sin^4(θ)/y^4 = 97*sin(2θ)/(x^3*y + y^3*x),
    then x/y + y/x = 14 -/
theorem sum_of_ratios_is_14 
  (x y : ℝ) (θ : ℝ) 
  (hx : x > 0) (hy : y > 0)
  (hθ : ∀ n : ℤ, θ ≠ π / 2 * n)
  (h1 : Real.sin θ / x = Real.cos θ / y)
  (h2 : Real.cos θ^4 / x^4 + Real.sin θ^4 / y^4 = 97 * Real.sin (2 * θ) / (x^3 * y + y^3 * x)) :
  x / y + y / x = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ratios_is_14_l92_9248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recruit_line_solution_l92_9249

/-- Represents the position of a person in a line. -/
structure Position where
  ahead : Nat

/-- Represents a brother in the line of recruits. -/
structure Brother where
  name : String
  position : Position

/-- The line of recruits. -/
structure RecruitLine where
  peter : Brother
  nicholas : Brother
  denis : Brother
  total : Nat

theorem recruit_line_solution (line : RecruitLine) : 
  line.peter.position.ahead = 50 →
  line.nicholas.position.ahead = 100 →
  line.denis.position.ahead = 170 →
  (∃ (b1 b2 : Brother), 
    (b1 = line.peter ∨ b1 = line.nicholas ∨ b1 = line.denis) ∧
    (b2 = line.peter ∨ b2 = line.nicholas ∨ b2 = line.denis) ∧
    (b1 ≠ b2) ∧
    (line.total - 1 - b1.position.ahead = 4 * (line.total - 1 - b2.position.ahead))) →
  line.total = 211 := by
  sorry

#check recruit_line_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recruit_line_solution_l92_9249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_interior_angle_l92_9260

/-- The number of sides in a decagon -/
def n : ℕ := 10

/-- The sum of interior angles of a polygon with n sides -/
noncomputable def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- The measure of each interior angle in a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := (sum_interior_angles n) / n

theorem regular_decagon_interior_angle :
  interior_angle n = 144 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_interior_angle_l92_9260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_min_distance_squared_value_l92_9274

noncomputable def parallelogram_area (z : ℂ) : ℝ :=
  abs (z.im * (z + z⁻¹).re - z.re * (z + z⁻¹).im)

theorem min_distance_squared (z : ℂ) (h1 : z.re > 0) (h2 : parallelogram_area z = 12/13) :
  ∃ (w : ℂ), w.re > 0 ∧ parallelogram_area w = 12/13 ∧
  ∀ (u : ℂ), u.re > 0 → parallelogram_area u = 12/13 → Complex.abs (w - w⁻¹) ≤ Complex.abs (u - u⁻¹) :=
by sorry

theorem min_distance_squared_value (z : ℂ) (h1 : z.re > 0) (h2 : parallelogram_area z = 12/13) :
  ∃ (w : ℂ), w.re > 0 ∧ parallelogram_area w = 12/13 ∧
  Complex.abs (w - w⁻¹) ^ 2 = 6 ∧
  ∀ (u : ℂ), u.re > 0 → parallelogram_area u = 12/13 → Complex.abs (w - w⁻¹) ≤ Complex.abs (u - u⁻¹) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_min_distance_squared_value_l92_9274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_diameter_l92_9289

/-- The volume of a sphere with radius r --/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

/-- The diameter of a sphere with radius r --/
def sphereDiameter (r : ℝ) : ℝ := 2 * r

theorem larger_sphere_diameter (r₁ r₂ : ℝ) (h₁ : r₁ = 9) (h₂ : sphereVolume r₂ = 3 * sphereVolume r₁) :
  sphereDiameter r₂ = 18 * (3 : ℝ) ^ (1/3) := by
  sorry

#check larger_sphere_diameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_diameter_l92_9289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_gt_one_l92_9218

noncomputable def g (x m : ℝ) : ℝ := Real.log x + 1 / (2 * x) - m

theorem sum_of_zeros_gt_one {x₁ x₂ m : ℝ} (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ < x₂)
  (hz₁ : g x₁ m = 0) (hz₂ : g x₂ m = 0) : x₁ + x₂ > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_gt_one_l92_9218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_basketball_score_l92_9266

theorem jordan_basketball_score :
  ∀ (x y : ℕ),
  x + y = 40 →  -- Total shots
  (0.4 * 3 * (x : ℝ) + 0.5 * 2 * (y : ℝ)) = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_basketball_score_l92_9266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gym_member_ratio_l92_9284

theorem gym_member_ratio (male_spend female_spend overall_spend : ℝ) 
  (male_count female_count : ℝ) :
  male_spend = 60 →
  female_spend = 80 →
  overall_spend = 65 →
  (male_spend * male_count + female_spend * female_count) / 
    (male_count + female_count) = overall_spend →
  female_count / male_count = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gym_member_ratio_l92_9284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_l92_9232

theorem sum_a_b : ∀ a b : ℕ,
  (∀ n : ℕ, n ≥ 2 ∧ n ≤ 5 → (n + n / (n^2 - 1 : ℕ)).sqrt = n * (n / (n^2 - 1 : ℕ)).sqrt) →
  (10 + a / b).sqrt = 10 * (a / b).sqrt →
  a + b = 109 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_l92_9232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arrangement_exists_l92_9216

/-- A type representing the arrangement of numbers on a triangle's sides -/
def TriangleArrangement := Array (Array Nat)

/-- Check if an arrangement is valid (uses all numbers from 1 to 9 once) -/
def isValidArrangement (arr : TriangleArrangement) : Prop :=
  arr.size = 3 ∧
  (arr.foldl (init := []) (fun acc side => acc ++ side.toList)).toFinset = Finset.range 9

/-- Calculate the sum of squares for a side of the triangle -/
def sumOfSquares (side : Array Nat) : Nat :=
  side.foldl (init := 0) (fun acc x => acc + x * x)

/-- The main theorem statement -/
theorem triangle_arrangement_exists :
  ∃ (arr : TriangleArrangement),
    isValidArrangement arr ∧
    (∀ side, side ∈ arr.data → sumOfSquares side = 95) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arrangement_exists_l92_9216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_approx_l92_9200

/-- Represents the parameters of a cone constructed from a circular paper --/
structure ConePaper where
  R : ℝ  -- Radius of the original circular paper
  r : ℝ  -- Radius of the cone's base
  V : ℝ  -- Volume of the cone
  h : ℝ  -- Height of the cone
  θ : ℝ  -- Angle of the sector used to form the cone (in radians)

/-- Calculates the angle of the unused sector in degrees --/
noncomputable def unused_sector_angle (cone : ConePaper) : ℝ :=
  360 - (cone.θ * 180 / Real.pi)

/-- Theorem stating the unused sector angle for the given cone parameters --/
theorem unused_sector_angle_approx (cone : ConePaper) 
  (h_r : cone.r = 15)
  (h_V : cone.V = 675 * Real.pi)
  (h_cone_valid : cone.r > 0 ∧ cone.V > 0 ∧ cone.h > 0 ∧ cone.R > cone.r) :
  ∃ ε > 0, |unused_sector_angle cone - 51.43| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_approx_l92_9200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_T_l92_9261

noncomputable def S (n : ℕ) : ℚ :=
  if n % 2 = 0 then -n / 2 else (n + 1) / 2

noncomputable def T (n : ℕ) : ℚ :=
  S n + ⌊(n : ℚ).sqrt⌋

theorem sum_of_specific_T : T 19 + T 21 + T 40 = 15 := by
  -- Unfold definitions and simplify
  simp [T, S]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_T_l92_9261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_path_length_l92_9271

/-- Represents a cylindrical post with a helical path around it. -/
structure CylindricalPost where
  height : ℝ
  circumference : ℝ
  circuitsPerUnit : ℝ

/-- Calculates the length of the helical path around the cylindrical post. -/
noncomputable def helicalPathLength (post : CylindricalPost) : ℝ :=
  let totalCircuits := post.height * post.circuitsPerUnit
  let horizontalDistance := totalCircuits * post.circumference
  Real.sqrt (post.height ^ 2 + horizontalDistance ^ 2)

/-- Theorem stating that the helical path length for the given conditions is 20 feet. -/
theorem squirrel_path_length :
  let post : CylindricalPost := {
    height := 16,
    circumference := 3,
    circuitsPerUnit := 1 / 4
  }
  helicalPathLength post = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_path_length_l92_9271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_l92_9242

theorem cone_height (r : ℝ) (θ : ℝ) (h : ℝ) : 
  r = 6 → θ = π / 3 → h = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_l92_9242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_deformation_l92_9291

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_flat : Prop
  rigid_rods : Prop
  hinged : Prop

/-- Predicate to check if a polygon is a parallelogram -/
def is_parallelogram {n : ℕ} (p : Polygon n) : Prop :=
  n = 4 ∧ sorry

/-- Predicate to check if a polygon can be deformed into a triangle -/
def can_deform_to_triangle {n : ℕ} (p : Polygon n) : Prop :=
  sorry

/-- Theorem stating that any polygon can be deformed into a triangle except for parallelograms when n = 4 -/
theorem polygon_deformation {n : ℕ} (p : Polygon n) :
  can_deform_to_triangle p ↔ (n ≠ 4 ∨ (n = 4 ∧ ¬is_parallelogram p)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_deformation_l92_9291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l92_9247

/-- Given vectors a, b, and c in ℝ², prove that if a - λb is collinear with c, then λ = 4/3 -/
theorem collinear_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 2) →
  b = (2, 1) →
  c = (5, -2) →
  (∃ (k : ℝ), a - lambda • b = k • c) →
  lambda = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l92_9247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_plus_one_solutions_l92_9288

theorem factorial_plus_one_solutions :
  ∀ n m : ℕ, n > 0 → m > 0 → ((n + 1) * m = Nat.factorial n + 1 ↔ 
    (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 4 ∧ m = 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_plus_one_solutions_l92_9288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_effect_l92_9207

/-- Represents the three distinct objects in the circle -/
inductive Object : Type
| Triangle
| SmallCircle
| Rectangle

/-- Represents a position on the circle's perimeter -/
structure Position :=
(angle : ℚ)

/-- Represents the arrangement of objects on the circle -/
def Arrangement := Object → Position

/-- Rotates a position by the given angle (in degrees) clockwise -/
def rotate (p : Position) (angle : ℚ) : Position :=
{ angle := (p.angle - angle + 360) % 360 }

/-- The theorem stating that after a 150° clockwise rotation, each object moves to the position
    previously occupied by the next object in the clockwise direction -/
theorem rotation_effect (initial : Arrangement) :
  ∃ (final : Arrangement),
    (∀ o : Object, final o = rotate (initial o) 150) ∧
    (final Object.Triangle = initial Object.SmallCircle) ∧
    (final Object.SmallCircle = initial Object.Rectangle) ∧
    (final Object.Rectangle = initial Object.Triangle) := by
  sorry

#check rotation_effect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_effect_l92_9207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_dominating_pair_exists_l92_9233

/-- Represents a team in the tournament -/
structure Team where
  id : ℕ

/-- Represents the result of a match between two teams -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a tournament with 9 teams -/
structure Tournament where
  teams : Fin 9 → Team
  result : Fin 9 → Fin 9 → MatchResult

/-- Two teams dominate if every other team loses to at least one of them -/
def dominating_pair (t : Tournament) (a b : Fin 9) : Prop :=
  ∀ c : Fin 9, c ≠ a ∧ c ≠ b → 
    t.result c a = MatchResult.Loss ∨ t.result c b = MatchResult.Loss

/-- Theorem stating that it's not always true that there exists a dominating pair -/
theorem not_always_dominating_pair_exists : 
  ∃ t : Tournament, ¬∃ (a b : Fin 9), a ≠ b ∧ dominating_pair t a b :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_dominating_pair_exists_l92_9233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_correct_l92_9262

-- Define the set of radii
def radii : List ℝ := [2, 4, 6, 8, 10]

-- Define functions for diameter, circumference, and area
def diameter (r : ℝ) : ℝ := 2 * r

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the function to calculate the point for a given radius
noncomputable def calculatePoint (r : ℝ) : ℝ × ℝ × ℝ :=
  (diameter r, circumference r, area r)

-- State the theorem
theorem circle_points_correct :
  List.map calculatePoint radii = [
    (4, 4 * Real.pi, 4 * Real.pi),
    (8, 8 * Real.pi, 16 * Real.pi),
    (12, 12 * Real.pi, 36 * Real.pi),
    (16, 16 * Real.pi, 64 * Real.pi),
    (20, 20 * Real.pi, 100 * Real.pi)
  ] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_correct_l92_9262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_theorem_l92_9202

/-- Represents a chessboard configuration -/
structure ChessBoard where
  pieces : Fin 8 → Fin 8 → ℕ

/-- The number of pieces in a 2x2 square starting at (i, j) -/
def pieces_in_2x2 (board : ChessBoard) (i j : Fin 8) : ℕ :=
  board.pieces i j + board.pieces i (j+1) + board.pieces (i+1) j + board.pieces (i+1) (j+1)

/-- The number of pieces in a 3x1 rectangle starting at (i, j) -/
def pieces_in_3x1 (board : ChessBoard) (i j : Fin 8) : ℕ :=
  board.pieces i j + board.pieces (i+1) j + board.pieces (i+2) j

/-- The total number of pieces on the board -/
def total_pieces (board : ChessBoard) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 8)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 8)) (board.pieces i)

/-- The property that all 2x2 squares have the same number of pieces -/
def uniform_2x2 (board : ChessBoard) : Prop :=
  ∀ i j k l : Fin 7, pieces_in_2x2 board i j = pieces_in_2x2 board k l

/-- The property that all 3x1 rectangles have the same number of pieces -/
def uniform_3x1 (board : ChessBoard) : Prop :=
  ∀ i j k l : Fin 6, pieces_in_3x1 board i j = pieces_in_3x1 board k l

theorem chessboard_theorem (board : ChessBoard) 
  (h1 : uniform_2x2 board) (h2 : uniform_3x1 board) :
  total_pieces board = 0 ∨ total_pieces board = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_theorem_l92_9202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cost_l92_9212

/-- Represents the cost and quantity of an item -/
structure Item where
  quantity : Nat
  cost : Nat
  deriving Repr

/-- Calculates the total cost of an order -/
def totalCost (items : List Item) : Nat :=
  items.foldl (fun acc item => acc + item.quantity * item.cost) 0

/-- Proves that the cost of each ice-cream cup is 25 rupees -/
theorem ice_cream_cost (chapati : Item) (rice : Item) (vegetable : Item) (ice_cream : Item) 
    (h1 : chapati.quantity = 16 ∧ chapati.cost = 6)
    (h2 : rice.quantity = 5 ∧ rice.cost = 45)
    (h3 : vegetable.quantity = 7 ∧ vegetable.cost = 70)
    (h4 : ice_cream.quantity = 6)
    (h5 : totalCost [chapati, rice, vegetable, ice_cream] = 961) :
    ice_cream.cost = 25 := by
  sorry

#eval totalCost [
  { quantity := 16, cost := 6 },
  { quantity := 5, cost := 45 },
  { quantity := 7, cost := 70 },
  { quantity := 6, cost := 25 }
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cost_l92_9212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l92_9238

/-- For a parabola with equation y² = x, its directrix is given by x = -1/4 -/
theorem parabola_directrix (y x : ℝ) : 
  (∀ y, y^2 = x) → (x = -1/4 ↔ x = -(1/4)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l92_9238
