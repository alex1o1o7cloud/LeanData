import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_proof_l1322_132291

theorem base_number_proof (x : ℝ) (n : ℝ) (b : ℝ) : 
  n = x^(15/100) → n^b = 9 → b = 40/3 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_proof_l1322_132291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l1322_132293

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d e f : ℝ} :
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ d * x + e * y + f = 0) → a * e = b * d

/-- Definition of line l₁ -/
def l₁ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (3 - a) * x + (2 * a - 1) * y + 5 = 0

/-- Definition of line l₂ -/
def l₂ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (2 * a + 1) * x + (a + 5) * y - 3 = 0

/-- Theorem: If l₁ is parallel to l₂, then a = 8/5 -/
theorem parallel_lines_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ↔ l₂ a x y) → a = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l1322_132293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_decreasing_on_interval_l1322_132292

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 12)

noncomputable def f' (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 12)

noncomputable def y (x : ℝ) : ℝ := 2 * f x + f' x

theorem y_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, Real.pi / 12 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 7 * Real.pi / 12 → y x₂ < y x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_decreasing_on_interval_l1322_132292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_equals_diameter_l1322_132296

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to calculate distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Helper definitions (not proven, just declared)
def are_externally_tangent (c1 c2 : Circle) : Prop := sorry

def is_tangent_point (p : Point) (c : Circle) : Prop := sorry

def is_diametrically_opposite (p1 p2 : Point) (c : Circle) : Prop := sorry

-- Define the theorem
theorem tangent_equals_diameter 
  (c1 c2 : Circle) 
  (A B : Point) 
  (h1 : are_externally_tangent c1 c2)
  (h2 : is_tangent_point A c1)
  (h3 : is_diametrically_opposite A B c1) :
  ∃ (D : Point), 
    is_tangent_point D c2 ∧ 
    distance B D = distance A B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_equals_diameter_l1322_132296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_g_l1322_132268

-- Define the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^3

-- Theorem statement
theorem monotonicity_of_g (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x > f a y) : 
  ∀ x y : ℝ, x < y → g a x > g a y :=
by
  -- We'll prove this later
  sorry

-- Helper lemma to show that a < 1
lemma a_less_than_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x y : ℝ, x < y → f a x > f a y) : a < 1 :=
by
  -- We'll prove this later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_g_l1322_132268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l1322_132244

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := n

-- Define b_n
noncomputable def b (n : ℕ) : ℝ := a n / 2^n

-- Define the sum of the first n terms of b_n
noncomputable def T (n : ℕ) : ℝ := 2 - (n + 2) / 2^n

-- Define the sum of the first 5 terms of a_n
noncomputable def S_5 : ℝ := (a 1 + a 2 + a 3 + a 4 + a 5)

-- State the theorem
theorem arithmetic_sequence_proof :
  (S_5 = 15) ∧
  (∃ r : ℝ, r > 1 ∧ a 6 = r * (2 * a 2) ∧ (a 8 + 1) = r * a 6) →
  (∀ n : ℕ, a n = n) ∧
  (∀ n : ℕ, T n = 2 - (n + 2) / 2^n) := by
  sorry

#check arithmetic_sequence_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l1322_132244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_20_equals_97_l1322_132240

/-- A function satisfying the given recurrence relation -/
def f : ℕ → ℚ
| 0 => 2
| n + 1 => (2 * f n + n.succ) / 2

/-- The theorem stating that f(20) = 97 -/
theorem f_20_equals_97 : f 19 = 97 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_20_equals_97_l1322_132240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_experimental_pi_properties_l1322_132249

/-- Experimental value of π based on random point simulation in a square --/
noncomputable def experimental_pi (n m : ℕ) (h : m ≤ n) : ℝ :=
  9 * (m : ℝ) / (4 * n)

/-- Properties of the experimental π value --/
theorem experimental_pi_properties 
  (side_length : ℝ) 
  (n m : ℕ) 
  (h1 : side_length = 3)
  (h2 : m ≤ n) : 
  ∃ (square_area circle_area : ℝ),
    square_area = side_length ^ 2 ∧ 
    circle_area = 4 * Real.pi ∧
    experimental_pi n m h2 = (circle_area / square_area) * (m : ℝ) / (n : ℝ) := by
  sorry

#check experimental_pi
#check experimental_pi_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_experimental_pi_properties_l1322_132249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l1322_132279

-- Define the function f(x) = x - ln x
noncomputable def f (x : ℝ) : ℝ := x - Real.log x

-- State the theorem
theorem monotonic_decreasing_interval_f :
  {x : ℝ | x > 0 ∧ x < 1} = {x : ℝ | ∀ y, x < y → f x > f y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l1322_132279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_properties_l1322_132247

noncomputable def series_term (n : ℕ) : ℝ := (((n + 1)^2 - 1).sqrt : ℝ) / n

theorem series_properties :
  (series_term 6 = (2 * Real.sqrt 3) / 3) ∧
  (series_term 1 = Real.sqrt 3) ∧
  (series_term 2 = Real.sqrt 8 / 2) ∧
  (series_term 3 = Real.sqrt 15 / 3) ∧
  (series_term 4 = Real.sqrt 24 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_properties_l1322_132247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jar_marble_difference_l1322_132236

theorem jar_marble_difference (total_green : ℕ) : 
  total_green = 145 →
  ∃ (x y : ℚ), 
    x > 0 ∧ y > 0 ∧
    10 * x = 7 * y ∧
    3 * x + 2 * y = total_green ∧
    (Int.floor (5 * y) : ℤ) - (Int.floor (7 * x) : ℤ) = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jar_marble_difference_l1322_132236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_x_in_triangle_l1322_132273

theorem max_tan_x_in_triangle (X Y Z : Real) :
  (∀ X' Y' Z' : Real, 
    (Y' - X')^2 + Z'^2 = 26^2 → Z' = 18 → 
    Real.tan X' ≤ (9 * Real.sqrt 22) / 44) ∧
  ∃ X₀ Y₀ Z₀ : Real, 
    (Y₀ - X₀)^2 + Z₀^2 = 26^2 ∧ Z₀ = 18 ∧ 
    Real.tan X₀ = (9 * Real.sqrt 22) / 44 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_x_in_triangle_l1322_132273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_least_valid_n_l1322_132237

def S (n : ℕ) : ℚ := (n - 1) * n * (n + 1) * (3 * n + 2) / 24

def is_valid (n : ℕ) : Prop := n ≥ 2 ∧ (S n).num % 5 = 0

def least_valid_n : List ℕ := [5, 6, 10, 11, 15, 16, 20, 21, 25, 26]

theorem sum_of_least_valid_n : 
  least_valid_n.length = 10 ∧ 
  (∀ m : ℕ, m ∈ least_valid_n → is_valid m) ∧
  (∀ m : ℕ, is_valid m → m < least_valid_n.head! → m ∈ least_valid_n) ∧
  least_valid_n.sum = 105 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_least_valid_n_l1322_132237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typed_number_digits_l1322_132229

-- Define the typed number
def typed_number : ℕ := 52115

-- Define the number of possible original numbers
def possible_originals : ℕ := 21

-- Define the number of digits in the original number
def original_digits : ℕ := 7

-- Define the number of missing digits
def missing_digits : ℕ := 2

-- Theorem to prove
theorem typed_number_digits : 
  (String.length (toString typed_number) = original_digits - missing_digits) ∧
  (Nat.choose original_digits missing_digits = possible_originals) := by
  sorry

#eval String.length (toString typed_number)
#eval Nat.choose original_digits missing_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typed_number_digits_l1322_132229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_candy_distribution_l1322_132265

/-- Given the number of candy necklace packs and necklaces per pack,
    calculate the number of packs opened based on the remaining necklaces. -/
def packs_opened (total_packs : ℕ) (necklaces_per_pack : ℕ) (necklaces_left : ℕ) : ℕ :=
  ((total_packs * necklaces_per_pack - necklaces_left) / necklaces_per_pack)

theorem emily_candy_distribution :
  packs_opened 15 12 67 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_candy_distribution_l1322_132265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_and_circumference_l1322_132215

noncomputable def circle_center : ℝ × ℝ := (-4, 5)
noncomputable def point_on_circle : ℝ × ℝ := (10, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def radius : ℝ := distance circle_center point_on_circle

theorem circle_area_and_circumference :
  (π * radius^2 = 245 * π) ∧
  (2 * π * radius = 2 * π * Real.sqrt 245) := by
  sorry

#check circle_area_and_circumference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_and_circumference_l1322_132215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1322_132276

noncomputable def f (x : ℝ) : ℝ := 2 / (abs x - 1)

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 ↔ x ≠ 1 ∧ x ≠ -1) ∧
  (∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ∈ Set.Iic (-2) ∪ Set.Ioi 0) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1322_132276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_justin_and_tim_games_l1322_132297

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of games two specific players play together -/
def games_together : ℕ := 210

/-- Theorem stating that two specific players will play together in 210 games -/
theorem justin_and_tim_games : 
  ∀ (players : Finset (Fin total_players)),
  (players.card = total_players) →
  (∃ (justin tim : Fin total_players), justin ≠ tim) →
  (∀ (game : Finset (Fin total_players)), 
    game.card = players_per_game → 
    (∃! (occurrence : ℕ), occurrence = 1)) →
  (∃ (justin tim : Fin total_players),
    (Finset.filter (fun game => 
      game.card = players_per_game ∧ 
      justin ∈ game ∧ 
      tim ∈ game) (Finset.powerset players)).card = games_together) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_justin_and_tim_games_l1322_132297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_example_l1322_132223

/-- Two monomials are like terms if they have the same variables with the same exponents. -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ i j, (term1 i j ≠ 0 ↔ term2 i j ≠ 0) ∧ (term1 i j ≠ 0 → term1 i j = term2 i j)

/-- Given a = 2 and b = 3, prove that bx^(2a+1)y^4 and ax^5y^(b+1) are like terms. -/
theorem like_terms_example (a b : ℕ) (h1 : a = 2) (h2 : b = 3) :
  are_like_terms 
    (λ i j ↦ if i = 5 ∧ j = 4 then b else 0) 
    (λ i j ↦ if i = 5 ∧ j = 4 then a else 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_example_l1322_132223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1322_132253

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arcsin (x^2))

-- State the theorem about the domain of g
theorem domain_of_g :
  {x : ℝ | g x ≠ 0 ∧ g x ≠ (π/2)} = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1322_132253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_prime_at_one_l1322_132251

-- Define the function f
noncomputable def f (x : ℝ) (θ : ℝ) : ℝ := x^3 + x^2 + Real.tan θ

-- State the theorem
theorem range_of_f_prime_at_one :
  ∀ θ : ℝ, θ ∈ Set.Icc 0 Real.pi →
  ∃ y : ℝ, y ≤ 2 ∧ ∃ θ₀ : ℝ, θ₀ ∈ Set.Icc 0 Real.pi ∧ 
  (deriv (λ x => f x θ₀)) 1 = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_prime_at_one_l1322_132251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_BL_AC_l1322_132295

/-- Type representing a geometric point -/
structure Point : Type := (x y : ℝ)

/-- Type representing a geometric line -/
structure Line : Type := (a b c : ℝ)

/-- Predicate stating that O is the circumcenter of triangle ABC -/
def CircumCenter (O A B C : Point) : Prop := sorry

/-- Predicate stating that M is on the circumcircle of triangle AOC -/
def OnCircumcircle (M A O C : Point) : Prop := sorry

/-- Predicate stating that M is on the line AB -/
def OnLine (M A B : Point) : Prop := sorry

/-- Predicate stating that L is the reflection of K in the line MN -/
def Reflection (L K M N : Point) : Prop := sorry

/-- Predicate stating that two lines are perpendicular -/
def Perpendicular (l1 l2 : Line) : Prop := sorry

/-- Given a triangle ABC with circumcenter O, and K as the circumcenter of triangle AOC,
    where AB and BC meet the circumcircle of AOC at M and N respectively,
    and L is the reflection of K in MN, prove that BL ⊥ AC -/
theorem perpendicular_BL_AC (A B C O K M N L : Point) : 
  CircumCenter O A B C →
  CircumCenter K A O C →
  OnCircumcircle M A O C →
  OnCircumcircle N A O C →
  OnLine M A B →
  OnLine N B C →
  Reflection L K M N →
  Perpendicular (Line.mk 0 0 0) (Line.mk 0 0 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_BL_AC_l1322_132295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1322_132202

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem f_monotone_increasing (ω φ : ℝ) (h1 : ω > 0) (h2 : -π < φ ∧ φ < 0)
  (h3 : ∀ x, f ω φ (x + π) = f ω φ x)
  (h4 : f ω φ (π/3) = 1) :
  StrictMonoOn (f ω φ) (Set.Icc (-π/6) (π/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1322_132202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_coloring_iff_odd_l1322_132264

/-- Represents a convex n-gon -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : Prop := sorry

/-- Represents a coloring of sides and diagonals -/
def Coloring (n : ℕ) := (Fin n × Fin n) → Fin n

/-- Checks if a coloring is valid according to the problem statement -/
def is_valid_coloring (n : ℕ) (P : ConvexPolygon n) (c : Coloring n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∃ (v1 v2 v3 : Fin n), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
      c (v1, v2) = i ∧ c (v2, v3) = j ∧ c (v3, v1) = k

/-- The main theorem -/
theorem valid_coloring_iff_odd (n : ℕ) :
  (∃ (P : ConvexPolygon n) (c : Coloring n), is_valid_coloring n P c) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_coloring_iff_odd_l1322_132264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_conversions_l1322_132285

-- Define conversion factors as noncomputable
noncomputable def cm3_to_L : ℝ := 1 / 1000
noncomputable def L_to_mL : ℝ := 1000
noncomputable def m3_to_dm3 : ℝ := 1000

-- Define the three equalities to be proved
theorem volume_conversions :
  (3500 * cm3_to_L = 3.5) ∧
  (7.2 * L_to_mL = 7200) ∧
  (5 * m3_to_dm3 = 5000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_conversions_l1322_132285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_problem_l1322_132258

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, -1)

theorem vector_dot_product_problem (b : ℝ × ℝ) 
  (h1 : ‖b‖ = Real.sqrt 5)
  (h2 : a • (a - b) = 0) :
  (a + b) • (a - 3 * b) = -19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_problem_l1322_132258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_sums_l1322_132204

theorem chessboard_sums (grid : Fin 8 → Fin 8 → ℕ) 
  (h_grid : ∀ i j, grid i j ∈ Finset.range 65 \ {0}) : 
  ∃ (s : Finset (Fin 7 × Fin 7)), s.card ≥ 3 ∧ 
    ∀ (i j), (i, j) ∈ s → grid i j + grid i (j+1) + grid (i+1) j + grid (i+1) (j+1) > 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_sums_l1322_132204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surjective_function_characterization_l1322_132232

/-- A bijection on the set of primes -/
def PrimeBijection := {g : ℕ → ℕ // Function.Bijective g ∧ ∀ p, Prime p ↔ Prime (g p)}

/-- The theorem statement -/
theorem surjective_function_characterization :
  ∀ (f : ℕ → ℕ),
    Function.Surjective f →
    (∀ m n : ℕ, m ∣ n ↔ f m ∣ f n) →
    ∃ (g : PrimeBijection),
      ∀ n : ℕ,
        f n = (Finset.prod (Nat.factors n).toFinset
          (λ p ↦ (g.val p) ^ (Nat.factorization n p))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surjective_function_characterization_l1322_132232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_midpoint_AB_l1322_132257

/-- The distance from the origin to the midpoint of a line segment -/
noncomputable def distanceOriginToMidpoint (a b : ℝ × ℝ) : ℝ :=
  let midpoint := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  Real.sqrt (midpoint.1 ^ 2 + midpoint.2 ^ 2)

/-- Theorem: The distance from the origin to the midpoint of the line segment
    connecting points A(-1, 5) and B(3, -7) is equal to √2 -/
theorem distance_origin_to_midpoint_AB :
  distanceOriginToMidpoint (-1, 5) (3, -7) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_midpoint_AB_l1322_132257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1322_132252

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(|x - a|)

theorem min_m_value (a : ℝ) :
  (∀ x, f a (1 + x) = f a (1 - x)) →
  (∃ m, ∀ x y, m ≤ x → x < y → f a x < f a y) →
  (∃ m, ∀ x y, m ≤ x → x < y → f a x < f a y ∧ 
    ∀ m', (∀ x y, m' ≤ x → x < y → f a x < f a y) → m ≤ m') :=
by
  intro h1 h2
  -- The proof would go here, but we'll use sorry for now
  sorry

#check min_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1322_132252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_sum_sum_of_extreme_points_l1322_132298

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 / (2 * x)) - a * x^2 + x

theorem extreme_points_and_sum (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, x > 0 → ¬ IsLocalExtr (f a) x) ∨
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ IsLocalExtr (f a) x₁ ∧ IsLocalExtr (f a) x₂) :=
by sorry

theorem sum_of_extreme_points (a : ℝ) (h : a > 0) 
  (x₁ x₂ : ℝ) (h₁ : IsLocalExtr (f a) x₁) (h₂ : IsLocalExtr (f a) x₂) :
  f a x₁ + f a x₂ > 3 - 4 * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_sum_sum_of_extreme_points_l1322_132298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_range_l1322_132213

open Real

-- Define the functions f and g
def f (a x : ℝ) : ℝ := -x^3 + 1 + a
noncomputable def g (x : ℝ) : ℝ := 3 * log x

-- Define the interval [1/e, e]
def I : Set ℝ := { x | 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 }

-- State the theorem
theorem symmetric_points_range (a : ℝ) :
  (∃ x ∈ I, f a x = -g x) ↔ 0 ≤ a ∧ a ≤ (Real.exp 1)^3 - 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_range_l1322_132213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_inflection_point_implies_k_range_l1322_132274

/-- The function f(x) defined for x > 0 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 4) * Real.exp x - k / 20 * x^5 + k / 6 * x^4

/-- The second derivative of f(x) -/
noncomputable def f_second_deriv (k : ℝ) (x : ℝ) : ℝ := (x - 2) * (Real.exp x - k * x^2)

/-- Theorem stating that if f has a unique inflection point at x = 2, then k is in the range (-∞, e²/4] -/
theorem unique_inflection_point_implies_k_range :
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → (f_second_deriv k x = 0 ↔ x = 2)) →
  k ∈ Set.Iic (Real.exp 2 / 4) := by
  sorry

#check unique_inflection_point_implies_k_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_inflection_point_implies_k_range_l1322_132274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_points_existence_l1322_132228

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Check if a point is inside a triangle -/
def insideTriangle (p a b c : Point) : Prop :=
  -- This is a simplified definition; in reality, you'd need to check if p is on the same side of all three edges
  sorry

/-- The main theorem -/
theorem interior_points_existence (n : ℕ) (A : Finset Point) :
  (A.card = n) →
  (∀ p q r, p ∈ A → q ∈ A → r ∈ A → p ≠ q → q ≠ r → p ≠ r → ¬collinear p q r) →
  ∃ (B : Finset Point), 
    B.card = 2*n - 5 ∧ 
    ∀ p q r, p ∈ A → q ∈ A → r ∈ A → p ≠ q → q ≠ r → p ≠ r → 
      ∃ b, b ∈ B ∧ insideTriangle b p q r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_points_existence_l1322_132228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_weight_l1322_132284

/-- Given a mixture with 6 elements and specific percentages for each component,
    prove that the total weight of the mixture is 200/3 pounds. -/
theorem mixture_weight (sand_percent water_percent cement_percent lime_percent additive_percent : ℚ)
                       (gravel_weight : ℚ) :
  sand_percent = 25 / 100 →
  water_percent = 20 / 100 →
  cement_percent = 15 / 100 →
  lime_percent = 10 / 100 →
  additive_percent = 7 / 100 →
  gravel_weight = 12 →
  sand_percent + water_percent + cement_percent + lime_percent + additive_percent + gravel_weight / (200 / 3) = 1 →
  200 / 3 = 200 / 3 := by
  intro h1 h2 h3 h4 h5 h6 h7
  rfl

#check mixture_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_weight_l1322_132284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_inscribed_circumscribed_spheres_l1322_132287

/-- The angle between the generatrix of a cone and its base, given the ratio of inscribed to circumscribed sphere volumes -/
noncomputable def cone_angle (k : ℝ) : ℝ := 
  Real.arccos ((1 + Real.sqrt (1 - 2 * k^(1/3))) / 2)

/-- Theorem stating the relationship between the angle of a cone and the ratio of inscribed to circumscribed sphere volumes -/
theorem cone_inscribed_circumscribed_spheres (k : ℝ) :
  (0 < k ∧ k ≤ 1/8) →
  ∃ (α : ℝ), 
    (α = cone_angle k ∨ α = Real.arccos ((1 - Real.sqrt (1 - 2 * k^(1/3))) / 2)) ∧
    0 < α ∧ α < π/2 ∧
    k = (Real.sin α * (1 - Real.cos α))^3 / (2 * (1 - Real.cos α))^3 := by
  sorry

#check cone_inscribed_circumscribed_spheres

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_inscribed_circumscribed_spheres_l1322_132287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l1322_132200

-- Define the function f(x) = cos(πx) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi * x)

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, x ∈ Set.Icc (1/2 : ℝ) 1 → y ∈ Set.Icc (1/2 : ℝ) 1 → x < y → f y < f x) ∧
  (∀ x, f x + f (1 - x) = 0) := by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l1322_132200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l1322_132203

/-- A hyperbola is defined by its standard equation and asymptotes -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y ↦ x^2 / a^2 - y^2 / b^2 = 1
  asymptote : ℝ → ℝ → Prop := fun x y ↦ y = (b/a) * x ∨ y = -(b/a) * x

/-- The theorem states that a hyperbola passing through (4, √3) with asymptotes y = ±(1/2)x has the standard equation x²/4 - y² = 1 -/
theorem hyperbola_standard_equation :
  ∀ h : Hyperbola,
    (∀ x y, h.asymptote x y ↔ y = (1/2) * x ∨ y = -(1/2) * x) →
    h.equation 4 (Real.sqrt 3) →
    h.a = 2 ∧ h.b = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l1322_132203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l1322_132263

theorem cube_root_equality (a b : ℝ) : (a ^ (1/3)) = (b ^ (1/3)) → a = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l1322_132263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_equals_18_l1322_132261

-- Define the curve and line
noncomputable def curve (x : ℝ) : ℝ := Real.sqrt (2 * x)
def line (x : ℝ) : ℝ := x - 4

-- Define the area function
def area_between_curve_and_line : ℝ := 18

-- Theorem statement
theorem area_enclosed_equals_18 : area_between_curve_and_line = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_equals_18_l1322_132261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_interval_l1322_132254

-- Define the distribution function F(x)
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ -1 then 0
  else if x ≤ 1/3 then 3/4 * x + 3/4
  else 1

-- Define the probability function P(a < X < b)
noncomputable def P (a b : ℝ) : ℝ := F b - F a

-- Theorem statement
theorem probability_in_interval :
  P 0 (1/3) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_interval_l1322_132254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l1322_132260

/-- A power function that passes through the point (2,8) -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

/-- The condition that f passes through (2,8) -/
def f_passes_through_2_8 (α : ℝ) : Prop := f α 2 = 8

theorem power_function_inequality (α : ℝ) (h : f_passes_through_2_8 α) :
  {a : ℝ | f α (a + 1) < f α (3 - 2*a)} = {a : ℝ | a < 2/3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l1322_132260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_specific_complex_l1322_132211

theorem modulus_of_specific_complex : Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_specific_complex_l1322_132211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_cube_plus_nine_l1322_132280

theorem prime_cube_plus_nine (P : ℕ) (h1 : Nat.Prime P) (h2 : Nat.Prime (P^3 + 9)) :
  (P : ℤ)^2 - 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_cube_plus_nine_l1322_132280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_l1322_132212

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def digits_of (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits_of (n / 10)

def five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def all_digits_even (n : ℕ) : Prop :=
  ∀ d ∈ digits_of n, is_even d

def third_digit_largest (n : ℕ) : Prop :=
  let d := digits_of n
  d.length = 5 ∧ ∀ i, i ≠ 2 → d[2]! > d[i]!

def first_double_last (n : ℕ) : Prop :=
  let d := digits_of n
  d.length = 5 ∧ d[4]! = 2 * d[0]!

def second_double_second_last (n : ℕ) : Prop :=
  let d := digits_of n
  d.length = 5 ∧ d[3]! = 2 * d[1]!

theorem unique_number :
  ∃! n : ℕ, five_digit_number n ∧
             all_digits_even n ∧
             third_digit_largest n ∧
             first_double_last n ∧
             second_double_second_last n ∧
             n = 88644 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_l1322_132212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_plane_l1322_132288

def sphere_radius : ℝ := 13

def triangle_side_AB : ℝ := 6
def triangle_side_BC : ℝ := 8
def triangle_side_CA : ℝ := 10

theorem distance_center_to_plane (R AB BC CA : ℝ) 
  (h_R : R = sphere_radius)
  (h_AB : AB = triangle_side_AB)
  (h_BC : BC = triangle_side_BC)
  (h_CA : CA = triangle_side_CA) :
  ∃ (d : ℝ), d = 12 ∧ d^2 + (AB * BC * CA / (4 * Real.sqrt (12 * (12 - AB) * (12 - BC) * (12 - CA))))^2 = R^2 :=
by
  sorry

#eval sphere_radius
#eval triangle_side_AB
#eval triangle_side_BC
#eval triangle_side_CA

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_plane_l1322_132288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_f_max_value_range_l1322_132224

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x - abs (x + 1)

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (a - 1) * x - 1
  else if x = 0 then 0
  else (a - 1) * x + 1

-- Theorem for the properties of g
theorem g_properties (a : ℝ) :
  (∀ x : ℝ, x > 0 → g a x = f a x) ∧
  (∀ x : ℝ, g a (-x) = -g a x) ∧
  (g a 0 = 0) := by sorry

-- Theorem for the range of a where f has a maximum value
theorem f_max_value_range :
  ∀ a : ℝ, (∃ M : ℝ, ∀ x : ℝ, f a x ≤ M) ↔ a ∈ Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_f_max_value_range_l1322_132224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_45_67834_to_hundredth_l1322_132205

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_45_67834_to_hundredth :
  round_to_hundredth 45.67834 = 45.68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_45_67834_to_hundredth_l1322_132205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l1322_132269

/-- Represents a trapezoid ABCD with point E where extended legs meet -/
structure ExtendedTrapezoid where
  -- Length of base AB
  ab : ℝ
  -- Length of base CD
  cd : ℝ
  -- Height from D to line AB
  height : ℝ
  -- Ensure ab and cd are positive
  ab_pos : 0 < ab
  cd_pos : 0 < cd
  height_pos : 0 < height

/-- The ratio of the area of triangle EAB to the area of trapezoid ABCD -/
noncomputable def areaRatio (t : ExtendedTrapezoid) : ℝ :=
  -- Definition of the ratio (to be proved)
  1 / 3

/-- Theorem stating that for a trapezoid with given dimensions, 
    the ratio of the area of triangle EAB to the area of trapezoid ABCD is 1/3 -/
theorem area_ratio_is_one_third (t : ExtendedTrapezoid) 
    (h1 : t.ab = 10) 
    (h2 : t.cd = 20) 
    (h3 : t.height = 5) : 
  areaRatio t = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l1322_132269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_property_l1322_132256

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through a point (x₀, y₀) with slope k
def line (x₀ y₀ k x y : ℝ) : Prop := y - y₀ = k * (x - x₀)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define vector subtraction
def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Define scalar multiplication for vectors
def vec_mul (s : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (s * v.1, s * v.2)

-- Define the theorem
theorem parabola_intersection_property 
  (l_slope : ℝ) 
  (A B M : ℝ × ℝ) 
  (a b : ℝ) :
  parabola A.1 A.2 ∧ 
  parabola B.1 B.2 ∧ 
  A ≠ B ∧
  line focus.1 focus.2 l_slope A.1 A.2 ∧
  line focus.1 focus.2 l_slope B.1 B.2 ∧
  M.1 = 0 ∧
  line focus.1 focus.2 l_slope M.1 M.2 ∧
  vec_sub A M = vec_mul a (vec_sub focus A) ∧
  vec_sub B M = vec_mul b (vec_sub focus B) →
  a + b = -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_property_l1322_132256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_eraser_sum_l1322_132289

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := sorry

/-- The cost of an eraser in cents -/
def eraser_cost : ℕ := sorry

/-- The total cost of 10 pencils and 5 erasers is 200 cents -/
axiom total_cost : 10 * pencil_cost + 5 * eraser_cost = 200

/-- Both pencil and eraser cost at least 3 cents each -/
axiom min_cost : pencil_cost ≥ 3 ∧ eraser_cost ≥ 3

/-- A pencil costs more than an eraser -/
axiom pencil_more_expensive : pencil_cost > eraser_cost

/-- Theorem: The sum of the cost of one pencil and one eraser is 22 cents -/
theorem pencil_eraser_sum : pencil_cost + eraser_cost = 22 := by
  sorry

#check pencil_eraser_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_eraser_sum_l1322_132289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1322_132206

open Real

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - x^2)

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 1 ∧ (∀ y ∈ Set.Ioo 0 1, f y ≥ f x) ∧ f x = 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1322_132206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_probability_l1322_132294

/-- Represents the probability of rolling an odd number on a biased die -/
noncomputable def prob_odd : ℝ := 1 / 3

/-- Represents the probability of rolling an even number on a biased die -/
noncomputable def prob_even : ℝ := 2 / 3

/-- Axiom: The probabilities of rolling odd and even numbers sum to 1 -/
axiom prob_sum : prob_odd + prob_even = 1

/-- Axiom: The probability of rolling an even number is twice the probability of rolling an odd number -/
axiom prob_ratio : prob_even = 2 * prob_odd

/-- The probability of getting an odd sum when rolling the biased die twice -/
noncomputable def prob_odd_sum : ℝ := prob_odd * prob_even + prob_even * prob_odd

theorem odd_sum_probability : prob_odd_sum = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_probability_l1322_132294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mod_3_periodicity_l1322_132243

def f : ℕ → ℕ
| 0 => 3  -- We'll define f(0) as 3 to handle the case of f(1) in the original problem
| 1 => 5
| (n + 2) => f (n + 1) + f n

theorem f_mod_3_periodicity (b : ℕ) :
  f b % 3 = f (b % 8) % 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mod_3_periodicity_l1322_132243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1322_132245

noncomputable def f (x : ℝ) : ℝ := abs (Matrix.det ![![Real.sin x, 2], ![-1, Real.cos x]])

theorem min_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1322_132245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1322_132239

theorem min_value_theorem (m n : ℝ) 
  (h1 : ∀ x : ℝ, m * x^2 - 2 * Real.sqrt 2 * x + n ≥ 0)
  (h2 : m > 0)
  (h3 : n > 0) :
  (2 : ℝ)^m + (4 : ℝ)^n ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1322_132239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_incorrect_mark_l1322_132208

/-- Given a class of students with an initial average mark and a correction to one student's mark,
    calculate the original incorrect mark of that student. -/
theorem calculate_incorrect_mark
  (n : ℕ)                    -- Total number of students
  (initial_avg : ℝ)          -- Initial average mark
  (correct_mark : ℝ)         -- Correct mark for the student
  (final_avg : ℝ)            -- Final average mark after correction
  (h_n : n = 35)             -- There are 35 students
  (h_initial : initial_avg = 72)  -- Initial average was 72
  (h_correct : correct_mark = 56) -- Reema's correct mark is 56
  (h_final : final_avg = 71.71)   -- Final average is 71.71
  : ∃ (incorrect_mark : ℝ),
    n * initial_avg - correct_mark + incorrect_mark = n * final_avg ∧
    abs (incorrect_mark - 46.85) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_incorrect_mark_l1322_132208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1322_132209

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

-- State the theorem
theorem f_inequality (x : ℝ) : f x > f (2*x - 3) ↔ 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1322_132209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1322_132225

-- Define the problem
theorem arithmetic_sequence_ratio (n : ℕ) (S T : ℕ → ℝ) (a b : ℕ → ℝ)
    (h : ∀ n, S n / T n = (7 * n) / (n + 3)) :
    a 5 / b 5 = 21 / 4 := by
  -- Define the sum of arithmetic sequence
  let sum (x : ℕ → ℝ) (n : ℕ) := (n : ℝ) * (x 1 + x n) / 2

  -- Assert that S and T are sums of a and b respectively
  have sum_def : ∀ n, S n = sum a n ∧ T n = sum b n := by sorry
  
  -- Use the given ratio for n = 9
  have ratio_9 : S 9 / T 9 = 7 * 9 / (9 + 3) := h 9
  
  -- Express a_5 and b_5 in terms of a_1, a_9 and b_1, b_9
  have seq_prop : a 5 / b 5 = (a 1 + a 9) / (b 1 + b 9) := by sorry
  
  -- Connect the sequence property with the sum
  calc a 5 / b 5 = (a 1 + a 9) / (b 1 + b 9) := seq_prop
    _ = (S 9 / (9 / 2)) / (T 9 / (9 / 2)) := by sorry
    _ = S 9 / T 9 := by sorry
    _ = 7 * 9 / (9 + 3) := ratio_9
    _ = 21 / 4 := by norm_num

-- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1322_132225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_min_area_l1322_132246

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the point A
def point_A : ℝ × ℝ := (1, 0)

-- Define the condition for P being on curve C
def P_on_C (x y : ℝ) : Prop := curve_C x y

-- Define the condition for Q being on curve C with x₀ ≥ 5
def Q_on_C (x₀ y₀ : ℝ) : Prop := curve_C x₀ y₀ ∧ x₀ ≥ 5

-- Define the tangent lines from Q to circle E
def tangent_lines (x₀ y₀ : ℝ) : Prop :=
  Q_on_C x₀ y₀ → ∃ (M N : ℝ), 
    (∃ (k₁ k₂ : ℝ), 
      y₀ = k₁ * (M - x₀) ∧
      y₀ = k₂ * (N - x₀) ∧
      circle_E M 0 ∧
      circle_E N 0)

-- Define the area of triangle QMN
noncomputable def area_QMN (x₀ y₀ M N : ℝ) : ℝ := 
  (1/2) * |M - N| * y₀

-- Theorem statement
theorem curve_C_and_min_area :
  (∀ x y, P_on_C x y → curve_C x y) ∧
  (∀ x₀ y₀ M N, Q_on_C x₀ y₀ → tangent_lines x₀ y₀ → 
    area_QMN x₀ y₀ M N ≥ 25/2 ∧
    ∃ x₁ y₁ M₁ N₁, Q_on_C x₁ y₁ ∧ tangent_lines x₁ y₁ ∧ 
      area_QMN x₁ y₁ M₁ N₁ = 25/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_min_area_l1322_132246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_12204_l1322_132278

/-- Represents a digit string that can be placed in the grid -/
structure DigitString where
  value : List Nat
  length : Nat
  endsWith0 : Bool

/-- Represents a position in the grid -/
structure GridPosition where
  row : Nat
  col : Nat

/-- Represents the direction of a string placement -/
inductive Direction where
  | Vertical
  | Horizontal

/-- Represents a placement of a digit string in the grid -/
structure Placement where
  string : DigitString
  position : GridPosition
  direction : Direction

/-- Represents the grid -/
def Grid := List (List Nat)

/-- The set of all available digit strings -/
def availableStrings : List DigitString := sorry

/-- The dimensions of the grid -/
def gridRows : Nat := sorry
def gridCols : Nat := sorry

/-- Checks if a placement is valid according to the rules -/
def isValidPlacement (grid : Grid) (placement : Placement) : Bool := sorry

/-- Fills the grid with the given placements -/
def fillGrid (placements : List Placement) : Grid := sorry

/-- Extracts the value of ABCDE from the filled grid -/
def extractABCDE (grid : Grid) : Nat := sorry

/-- The main theorem stating that the unique solution for ABCDE is 12204 -/
theorem unique_solution_is_12204 :
  ∀ (placements : List Placement),
    (∀ s, s ∈ availableStrings → ∃! p, p ∈ placements ∧ p.string = s) →
    (∀ p, p ∈ placements → isValidPlacement (fillGrid placements) p) →
    extractABCDE (fillGrid placements) = 12204 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_12204_l1322_132278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_proposition_l1322_132255

-- Define the basic types
structure GeometricObject where

-- Define the relationships
def perpendicular (a b : GeometricObject) : Prop := sorry

def parallel (a b : GeometricObject) : Prop := sorry

-- Define the different types of geometric objects
def Line : Type := GeometricObject
def Plane : Type := GeometricObject

-- Define the proposition
def proposition (x y z : GeometricObject) : Prop :=
  perpendicular x y ∧ parallel y z → perpendicular x z

-- State the theorem
theorem geometric_proposition :
  (∀ (x y z : Line), proposition x y z) ∧
  (∀ (x y z : Plane), proposition x y z) ∧
  (∀ (x z : Plane) (y : Line), proposition x y z) ∧
  ¬(∀ (x y : Line) (z : Plane), proposition x y z) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_proposition_l1322_132255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_f_3_l1322_132238

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 2*m - 2) * x^(m-1)

theorem power_function_decreasing_f_3 (m : ℝ) :
  (∃ k c : ℝ, ∀ x : ℝ, x > 0 → f m x = c * x^k) →  -- f is a power function
  (∀ x y : ℝ, 0 < x ∧ x < y → f m y < f m x) →  -- f is monotonically decreasing in (0,+∞)
  f m 3 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_f_3_l1322_132238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_one_l1322_132201

-- Define a 5x5 grid
def Grid := Fin 5 → Fin 5 → Fin 5

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ≠ 0) ∧
  (∀ i, Function.Injective (g i)) ∧
  (∀ j, Function.Injective (λ i ↦ g i j))

-- Define the initial configuration
def initial_config (g : Grid) : Prop :=
  g 0 0 = 1 ∧ g 0 4 = 2 ∧
  g 1 1 = 3 ∧
  g 2 0 = 5 ∧ g 2 2 = 4 ∧
  g 3 2 = 1 ∧ g 3 3 = 3

-- Theorem statement
theorem lower_right_is_one (g : Grid) 
  (hvalid : is_valid_grid g) 
  (hinit : initial_config g) : 
  g 4 4 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_one_l1322_132201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1322_132210

def U : Set ℕ := {x | x ≤ 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {4, 5, 6}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1322_132210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surrounding_squares_l1322_132241

/-- A unit square in a 2D plane -/
structure UnitSquare where
  center : ℝ × ℝ

/-- Represents the configuration of unit squares around a fixed square -/
structure SquareConfiguration where
  fixed : UnitSquare
  surrounding : List UnitSquare

/-- Two squares touch but do not overlap if their centers are exactly 1 unit apart -/
def touches (s1 s2 : UnitSquare) : Prop :=
  let (x1, y1) := s1.center
  let (x2, y2) := s2.center
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 1

/-- Two squares do not overlap if their centers are at least 1 unit apart -/
def do_not_overlap (s1 s2 : UnitSquare) : Prop :=
  let (x1, y1) := s1.center
  let (x2, y2) := s2.center
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) ≥ 1

/-- A configuration is valid if all surrounding squares touch the fixed square
    and do not overlap with each other -/
def is_valid_configuration (config : SquareConfiguration) : Prop :=
  (∀ s, s ∈ config.surrounding → touches config.fixed s) ∧
  (∀ s1 s2, s1 ∈ config.surrounding → s2 ∈ config.surrounding → s1 ≠ s2 → do_not_overlap s1 s2)

/-- The main theorem: The maximum number of surrounding squares is 8 -/
theorem max_surrounding_squares (config : SquareConfiguration) :
  is_valid_configuration config → config.surrounding.length ≤ 8 := by
  sorry

#check max_surrounding_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surrounding_squares_l1322_132241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_with_bounded_nonadjacent_edges_l1322_132267

/-- A simple graph with n vertices and k edges, containing no triangles. -/
structure TriangleFreeGraph where
  n : ℕ  -- number of vertices
  k : ℕ  -- number of edges
  no_triangles : True  -- represents the condition that the graph has no triangles

/-- The main theorem statement -/
theorem exists_point_with_bounded_nonadjacent_edges (G : TriangleFreeGraph) :
  ∃ (P : Fin G.n), ∃ (m : ℕ), m ≤ G.k * (1 - 4 * G.k / G.n^2) ∧
    (∀ (Q R : Fin G.n), Q ≠ P → R ≠ P → (∃ (edge_QR : Bool), edge_QR = true) → m = m + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_with_bounded_nonadjacent_edges_l1322_132267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotational_volume_formula_correct_l1322_132233

/-- Represents a circular segment. -/
structure CircularSegment where
  chord_length : ℝ
  projection_length : ℝ
  diameter_non_intersecting : Bool

/-- Calculates the volume of a solid obtained by rotating a circular segment around its diameter. -/
noncomputable def rotational_volume (segment : CircularSegment) : ℝ :=
  (1/6) * Real.pi * segment.chord_length^2 * segment.projection_length

/-- Theorem stating that the rotational volume formula is correct for circular segments. -/
theorem rotational_volume_formula_correct (segment : CircularSegment) 
  (h_positive : segment.chord_length > 0 ∧ segment.projection_length > 0)
  (h_non_intersecting : segment.diameter_non_intersecting = true) :
  rotational_volume segment = (1/6) * Real.pi * segment.chord_length^2 * segment.projection_length := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotational_volume_formula_correct_l1322_132233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_asian_stamps_before_80s_total_european_stamps_90s_l1322_132235

-- Define the countries
inductive Country
| China
| Japan
| Italy
| Germany

-- Define the decades
inductive Decade
| Fifties
| Sixties
| Seventies
| Nineties

-- Define the function for stamp counts
def stampCount : Country → Decade → Nat
| Country.China => λ
  | Decade.Fifties => 5
  | Decade.Sixties => 9
  | Decade.Seventies => 11
  | Decade.Nineties => 8
| Country.Japan => λ
  | Decade.Fifties => 10
  | Decade.Sixties => 6
  | Decade.Seventies => 9
  | Decade.Nineties => 12
| Country.Italy => λ
  | Decade.Fifties => 7
  | Decade.Sixties => 8
  | Decade.Seventies => 5
  | Decade.Nineties => 11
| Country.Germany => λ
  | Decade.Fifties => 8
  | Decade.Sixties => 7
  | Decade.Seventies => 10
  | Decade.Nineties => 9

-- Define the price function
def price : Country → Nat
| Country.China => 7
| Country.Japan => 7
| Country.Italy => 5
| Country.Germany => 6

-- Define Asian and European countries
def isAsian : Country → Bool
| Country.China => true
| Country.Japan => true
| _ => false

def isEuropean : Country → Bool
| Country.Italy => true
| Country.Germany => true
| _ => false

-- Theorem statements
theorem total_cost_asian_stamps_before_80s :
  (List.sum ([Country.China, Country.Japan].map (λ c =>
    (List.sum ([Decade.Fifties, Decade.Sixties, Decade.Seventies].map (stampCount c))) * price c
  ))) = 350 := by sorry

theorem total_european_stamps_90s :
  (List.sum ([Country.Italy, Country.Germany].map (λ c => stampCount c Decade.Nineties))) = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_asian_stamps_before_80s_total_european_stamps_90s_l1322_132235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_less_than_one_l1322_132286

/-- The function f(x) = |e^x - e| + e^x + ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |Real.exp x - Real.exp 1| + Real.exp x + a * x

/-- Theorem: For a < -e, if x₁ and x₂ are the two zeros of f(x) with x₁ < x₂, then x₁x₂ < 1 -/
theorem zeros_product_less_than_one (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a < -Real.exp 1) 
  (hx : x₁ < x₂)
  (hf₁ : f a x₁ = 0)
  (hf₂ : f a x₂ = 0) :
  x₁ * x₂ < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_less_than_one_l1322_132286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raise_calculation_l1322_132234

/-- Calculates the new weekly earnings after a percentage raise -/
noncomputable def new_earnings (original : ℚ) (percentage : ℚ) : ℚ :=
  original * (1 + percentage / 100)

/-- Proves that a 40% raise on $60 results in $84 weekly earnings -/
theorem raise_calculation :
  let original := 60
  let percentage := 40
  new_earnings original percentage = 84 := by
  -- Unfold the definition of new_earnings
  unfold new_earnings
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raise_calculation_l1322_132234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1322_132250

-- Define a type for triangles
variable (Triangle : Type)

-- Define a predicate for having three medians of equal length
variable (has_equal_medians : Triangle → Prop)

-- Define the negation of the original statement
def negation_statement (Triangle : Type) (has_equal_medians : Triangle → Prop) : Prop :=
  ¬ (∃ t : Triangle, has_equal_medians t)

-- Define the equivalent statement
def equivalent_statement (Triangle : Type) (has_equal_medians : Triangle → Prop) : Prop :=
  ∀ t : Triangle, ¬(has_equal_medians t)

-- Theorem stating the equivalence
theorem negation_equivalence (Triangle : Type) (has_equal_medians : Triangle → Prop) :
  negation_statement Triangle has_equal_medians ↔ equivalent_statement Triangle has_equal_medians :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1322_132250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1322_132242

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the area function
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

-- State the theorem
theorem triangle_properties (abc : Triangle) :
  abc.b = 2 ∧ 
  (Real.tan abc.A + Real.tan abc.C = Real.sqrt 3 * (Real.tan abc.A * Real.tan abc.C - 1)) →
  abc.B = π / 3 ∧ 
  (∀ (other : Triangle), other.b = 2 → area other ≤ Real.sqrt 3) ∧
  (∃ (max_triangle : Triangle), max_triangle.b = 2 ∧ area max_triangle = Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1322_132242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_example_l1322_132221

/-- Sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℝ) (aₙ : ℝ) (d : ℝ) : ℝ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- The sum of the given arithmetic sequence is 970.2 -/
theorem arithmetic_sum_example : arithmetic_sum 15 31.2 0.4 = 970.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_example_l1322_132221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_on_y_eq_x_l1322_132219

/-- The distance between two points in R² -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The sum of distances from a point (x, y) to (1, 2) and (-2, 1) -/
noncomputable def m (x y : ℝ) : ℝ :=
  distance x y 1 2 + distance x y (-2) 1

/-- The theorem stating the minimum value of m for points on y = x -/
theorem min_m_on_y_eq_x :
  ∃ (min_m : ℝ), min_m = 4 ∧ ∀ (x y : ℝ), y = x → m x y ≥ min_m := by
  sorry

#check min_m_on_y_eq_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_on_y_eq_x_l1322_132219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_for_500_parts_l1322_132216

-- Define the cost, initial price, and minimum price
def cost : ℚ := 40
def initial_price : ℚ := 70
def min_price : ℚ := 61

-- Define the price decrease per additional part
def price_decrease : ℚ := 2/100

-- Define the threshold for price decrease
def threshold : ℚ := 100

-- Define the ex-factory price function
noncomputable def f (x : ℚ) : ℚ :=
  if x ≤ threshold then initial_price
  else max (initial_price - price_decrease * (x - threshold)) min_price

-- Define the profit function
noncomputable def L (x : ℚ) : ℚ := (f x - cost) * x

-- Theorem to prove
theorem profit_for_500_parts : L 500 = 11000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_for_500_parts_l1322_132216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amelia_score_l1322_132281

/-- Represents the number of points scored by Amelia in a basketball match -/
def ameliaPoints (x y z : ℚ) : ℚ :=
  0.75 * (3 * x) + 0.8 * (2 * y) + 0.5 * z

/-- Theorem stating that Amelia scored 28.5 points given the conditions -/
theorem amelia_score :
  ∃ (x y z : ℚ),
    x + y + z = 40 ∧
    z = 10 ∧
    ameliaPoints x y z = 28.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amelia_score_l1322_132281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_bear_incorrect_l1322_132226

/-- Represents the color of a bear -/
inductive BearColor
| White
| Brown
| Black

/-- Represents a row of 1000 bears -/
def BearRow := Fin 1000 → BearColor

/-- The constraint that among any three consecutive bears, there is at least one of each color -/
def valid_bear_row (row : BearRow) : Prop :=
  ∀ i : Fin 998, ∃ (c1 c2 c3 : BearColor), 
    ({c1, c2, c3} : Set BearColor) = {BearColor.White, BearColor.Brown, BearColor.Black} ∧
    ({row i, row (i + 1), row (i + 2)} : Set BearColor) = {c1, c2, c3}

/-- Iskander's guesses about the colors of specific bears -/
def iskander_guesses (row : BearRow) : Prop :=
  (row 1 = BearColor.White) ∧
  (row 19 = BearColor.Brown) ∧
  (row 399 = BearColor.Black) ∧
  (row 599 = BearColor.Brown) ∧
  (row 799 = BearColor.White)

/-- The theorem stating that the 20th bear's color must be the one incorrectly guessed -/
theorem twentieth_bear_incorrect (row : BearRow) 
  (h1 : valid_bear_row row) 
  (h2 : ∃! i : Fin 5, ¬ iskander_guesses row) : 
  row 19 ≠ BearColor.Brown := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_bear_incorrect_l1322_132226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_piles_mean_weight_l1322_132220

/-- Given three piles of marbles X, Y, and Z, prove that the least possible integer value
for the mean weight of marbles in combined piles Y and Z is 45 grams. -/
theorem marble_piles_mean_weight (X Y Z : Finset ℝ) : 
  (∀ x ∈ X, x > 0) →  -- All weights are positive
  (∀ y ∈ Y, y > 0) →
  (∀ z ∈ Z, z > 0) →
  X.Nonempty →  -- Piles are non-empty
  Y.Nonempty →
  Z.Nonempty →
  (X.sum id / X.card = 30) →  -- Mean weight of X is 30 grams
  (Y.sum id / Y.card = 60) →  -- Mean weight of Y is 60 grams
  ((X.sum id + Y.sum id) / (X.card + Y.card) = 50) →  -- Mean weight of X and Y combined is 50 grams
  ((X.sum id + Z.sum id) / (X.card + Z.card) = 45) →  -- Mean weight of X and Z combined is 45 grams
  ∃ n : ℕ, n ≥ 45 ∧ 
    ∀ m : ℕ, m < 45 → 
      ¬∃ (Y' Z' : Finset ℝ), 
        (Y' ⊆ Y ∧ Z' ⊆ Z) ∧ 
        ((Y'.sum id + Z'.sum id) / (Y'.card + Z'.card) = m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_piles_mean_weight_l1322_132220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_complex_number_l1322_132217

theorem purely_imaginary_complex_number (x : ℝ) : 
  (Complex.im ((x^2 - 1) + (x^2 + 3*x + 2)*Complex.I) = (x^2 + 3*x + 2) ∧ 
   Complex.re ((x^2 - 1) + (x^2 + 3*x + 2)*Complex.I) = 0) → 
  x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_complex_number_l1322_132217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l1322_132259

-- Define the function f
noncomputable def f (x : Real) : Real :=
  (2 * Real.sqrt 3 * Real.sin x - Real.cos x) * Real.cos x + (Real.cos (Real.pi / 2 - x))^2

-- State the theorem
theorem triangle_angle_range (a b c A B C : Real) : 
  -- Conditions
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  (a^2 + c^2 - b^2) / c = (a^2 + b^2 - c^2) / (2 * a - c) ∧
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Conclusion
  1 < f A ∧ f A ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l1322_132259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_p_l1322_132214

theorem max_prime_factors_p (p q : ℕ+) 
  (h_gcd : (Nat.factors (Nat.gcd p q)).length = 5)
  (h_lcm : (Nat.factors (Nat.lcm p q)).length = 20)
  (h_fewer : (Nat.factors p.val).length < (Nat.factors q.val).length) :
  (Nat.factors p.val).length ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_p_l1322_132214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_values_l1322_132299

-- Define the curves C₁ and C₂
noncomputable def C₁ (a t : ℝ) : ℝ × ℝ := (a + Real.sqrt 2 * t, 1 + Real.sqrt 2 * t)

def C₂ (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points A and B
def intersection_points (a : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), 
    C₂ (C₁ a t₁).1 (C₁ a t₁).2 ∧ 
    C₂ (C₁ a t₂).1 (C₁ a t₂).2 ∧ 
    t₁ ≠ t₂

-- Define the condition |PA| = 2|PB|
def distance_condition (a : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), 
    C₂ (C₁ a t₁).1 (C₁ a t₁).2 ∧ 
    C₂ (C₁ a t₂).1 (C₁ a t₂).2 ∧ 
    t₁ ≠ t₂ ∧
    (2 * |t₁| = 4 * |t₂| ∨ 2 * |t₂| = 4 * |t₁|)

-- Theorem statement
theorem intersection_values :
  ∀ a : ℝ, intersection_points a ∧ distance_condition a → a = 1/36 ∨ a = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_values_l1322_132299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_M_and_N_l1322_132272

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 2 > 0}
def N : Set ℝ := {x | Real.exp ((x - 1) * Real.log 2) ≤ 1/2}

-- State the theorem
theorem intersection_complement_M_and_N :
  (Mᶜ ∩ N) = Set.Icc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_M_and_N_l1322_132272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_convoy_l1322_132283

-- Define the problem parameters
noncomputable def tunnel_length : ℝ := 1165
def num_cars : ℕ := 17
noncomputable def car_length : ℝ := 5
noncomputable def speed_limit : ℝ := 200 / 9

-- Define the function for distance between cars
noncomputable def distance_between_cars (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 20
  else if 10 < x ∧ x ≤ speed_limit then (1/4 * x^2 - 1/2 * x)
  else 0

-- Define the function for total time
noncomputable def total_time (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 1570 / x
  else if 10 < x ∧ x ≤ speed_limit then 1250 / x + 4 * x - 8
  else 0

-- State the theorem
theorem min_time_convoy :
  ∃ (min_time : ℝ), min_time = 100 * Real.sqrt 2 - 8 ∧
  ∀ (x : ℝ), 0 < x ∧ x ≤ speed_limit → total_time x ≥ min_time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_convoy_l1322_132283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parameter_l1322_132248

/-- Given two lines l₁ and l₂ in the plane, if they are perpendicular, 
    then the parameter a in l₂'s equation equals 2. -/
theorem perpendicular_lines_parameter :
  ∀ a : ℝ, 
  (∀ x y : ℝ, x + y - 2 = 0 → 2*x + a*y - 3 = 0 → (x + y - 2) * (2*x + a*y - 3) = -1) → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parameter_l1322_132248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_table_sum_difference_l1322_132262

/-- Represents a square table of size n × n filled with integers from 1 to n^2 -/
def SquareTable (n : ℕ) := Fin n → Fin n → Fin (n^2)

/-- Predicate to check if a SquareTable follows the given rules -/
def ValidTable (n : ℕ) (table : SquareTable n) : Prop :=
  ∃ (start_row start_col : Fin n),
    (table start_row start_col = ⟨0, sorry⟩) ∧ 
    (∀ k : Fin (n^2), k.val < n^2 - 1 →
      ∃ (i j : Fin n), table i j = k ∧ 
        table j (⟨(table i j).val, sorry⟩) = ⟨k.val + 1, sorry⟩)

/-- Sum of numbers in a specific row of the table -/
def RowSum (n : ℕ) (table : SquareTable n) (row : Fin n) : ℕ :=
  (Finset.univ.sum (λ col => (table row col).val + 1))

/-- Sum of numbers in a specific column of the table -/
def ColSum (n : ℕ) (table : SquareTable n) (col : Fin n) : ℕ :=
  (Finset.univ.sum (λ row => (table row col).val + 1))

/-- The main theorem to prove -/
theorem square_table_sum_difference (n : ℕ) (table : SquareTable n) 
  (h : ValidTable n table) :
  ∃ (row_with_1 col_with_n2 : Fin n),
    ColSum n table col_with_n2 - RowSum n table row_with_1 = n^2 - n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_table_sum_difference_l1322_132262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_max_sum_squares_l1322_132270

/-- Given a circle with radius R, prove that for any inscribed triangle, 
    the sum of the squares of its side lengths is at most 9R^2, 
    with equality if and only if the triangle is equilateral. -/
theorem inscribed_triangle_max_sum_squares (R : ℝ) (h : R > 0) :
  ∀ A B C : ℝ × ℝ,
  (‖A‖ = R) → (‖B‖ = R) → (‖C‖ = R) →
  ‖A - B‖^2 + ‖B - C‖^2 + ‖C - A‖^2 ≤ 9 * R^2 ∧
  (‖A - B‖^2 + ‖B - C‖^2 + ‖C - A‖^2 = 9 * R^2 ↔ ‖A - B‖ = ‖B - C‖ ∧ ‖B - C‖ = ‖C - A‖) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_max_sum_squares_l1322_132270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l1322_132271

/-- Represents a right circular cone filled with liquid -/
structure Cone where
  surface_radius : ℝ
  marble_radius : ℝ

/-- Calculates the rise in liquid level when a marble is submerged in a cone -/
noncomputable def liquid_rise (cone : Cone) : ℝ :=
  (4 / 3) * Real.pi * cone.marble_radius ^ 3 / (Real.pi * cone.surface_radius ^ 2)

/-- The two cones in the problem -/
def cone1 : Cone := ⟨5, 2⟩
def cone2 : Cone := ⟨10, 3⟩

theorem liquid_rise_ratio :
  liquid_rise cone1 / liquid_rise cone2 = 32 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l1322_132271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_problem_l1322_132282

theorem arithmetic_progression_problem (a d : ℝ) :
  let seq := [a - 2*d, a - d, a, a + d, a + 2*d]
  (∀ i j, i < j → seq.get ⟨i, sorry⟩ < seq.get ⟨j, sorry⟩) →
  (seq.map (λ x => x^3)).sum = 0 →
  (seq.map (λ x => x^2)).sum = 70 →
  seq.head! = -2 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_problem_l1322_132282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_calculation_trigonometric_simplification_l1322_132231

-- Part 1
theorem trigonometric_calculation : 
  Real.sin (120 * π / 180) ^ 2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) - 
  Real.cos (-330 * π / 180) ^ 2 + Real.sin (-210 * π / 180) = 1/2 := by sorry

-- Part 2
theorem trigonometric_simplification (α : ℝ) : 
  (Real.sin (α - 2*π) * Real.cos (α - π/2) * Real.cos (π + α)) / 
  (Real.sin (3*π - α) * Real.sin (-π - α)) = -Real.cos α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_calculation_trigonometric_simplification_l1322_132231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_squared_l1322_132227

theorem root_difference_squared (x₁ x₂ : ℂ) : 
  (Real.sqrt 11 : ℝ) * x₁^2 + (Real.sqrt 180 : ℝ) * x₁ + (Real.sqrt 176 : ℝ) = 0 →
  (Real.sqrt 11 : ℝ) * x₂^2 + (Real.sqrt 180 : ℝ) * x₂ + (Real.sqrt 176 : ℝ) = 0 →
  x₁ ≠ x₂ →
  Complex.abs (1/x₁^2 - 1/x₂^2) = (Real.sqrt 45 : ℝ)/44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_squared_l1322_132227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derangement_one_fixed_point_relation_l1322_132275

/-- Number of permutations of {1, ..., n} with exactly one fixed point -/
def a (n : ℕ) : ℕ := sorry

/-- Number of derangements (permutations with no fixed points) of {1, ..., n} -/
def d (n : ℕ) : ℕ := sorry

/-- The relationship between derangements and permutations with one fixed point -/
theorem derangement_one_fixed_point_relation (n : ℕ) :
  d n = a n + (-1 : ℤ)^n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derangement_one_fixed_point_relation_l1322_132275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1322_132218

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  (Real.cos (2 * C) - 3 * Real.cos (A + B) = 1) →
  (c = 2 * Real.sqrt 3) →
  -- Conclusions
  (C = π / 3) ∧
  (∃ (S : ℝ), S = (1 / 2) * a * b * Real.sin C ∧
    ∀ (S' : ℝ), S' = (1 / 2) * a * b * Real.sin C → S' ≤ 3 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1322_132218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_ratio_l1322_132230

-- Define the ellipse and hyperbola
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def Hyperbola (A B : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / A^2) - (p.2^2 / B^2) = 1}

-- Define the eccentricity of an ellipse
noncomputable def EllipseEccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

-- Define the eccentricity of a hyperbola
noncomputable def HyperbolaEccentricity (A B : ℝ) : ℝ :=
  Real.sqrt (1 + (B^2 / A^2))

-- Theorem statement
theorem ellipse_hyperbola_intersection_ratio
  (a b A B : ℝ) 
  (ha : a > b) (hb : b > 0) (hA : A > 0) (hB : B > 0)
  (h_same_foci : ∃ c, c^2 = a^2 - b^2 ∧ c^2 = A^2 + B^2)
  (h_eccentricity : EllipseEccentricity a b * HyperbolaEccentricity A B = 1)
  (D : ℝ × ℝ)
  (h_D_ellipse : D ∈ Ellipse a b)
  (h_D_hyperbola : D ∈ Hyperbola A B)
  (h_D_first_quadrant : D.1 > 0 ∧ D.2 > 0) :
  (2 * a^2 / b^2) - 1 = 
    ((D.1 - a)^2 + D.2^2).sqrt / ((D.1 + a)^2 + D.2^2).sqrt :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_ratio_l1322_132230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l1322_132207

-- Define the function f(x) = e^x + 2x - 6
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 6

-- Theorem statement
theorem solution_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l1322_132207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_decrease_l1322_132277

theorem card_value_decrease (initial_value : ℝ) (h : initial_value > 0) : 
  (initial_value * (1 - 0.2) * (1 - 0.2) - initial_value) / initial_value = -0.36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_decrease_l1322_132277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_difference_l1322_132266

/-- The difference in amount owed between monthly and annual compounding interest -/
noncomputable def interestDifference (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  let monthlyAmount := principal * (1 + rate / 12) ^ (12 * years)
  let annualAmount := principal * (1 + rate) ^ years
  monthlyAmount - annualAmount

/-- Theorem stating the difference in interest for the given loan conditions -/
theorem loan_interest_difference :
  ∃ (diff : ℝ),
    interestDifference 8000 0.10 5 = diff ∧
    |diff - 376.04| < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_difference_l1322_132266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincidence_l1322_132290

/-- The x-coordinate of the focus of a parabola y² = 2px -/
noncomputable def parabola_focus_x (p : ℝ) : ℝ := p / 2

/-- The x-coordinate of the right focus of a hyperbola x²/3 - y² = 1 -/
noncomputable def hyperbola_right_focus_x : ℝ := 2

/-- 
  If the focus of the parabola y² = 2px coincides with the right focus of 
  the hyperbola x²/3 - y² = 1, then p = 4.
-/
theorem parabola_hyperbola_focus_coincidence (p : ℝ) :
  parabola_focus_x p = hyperbola_right_focus_x → p = 4 := by
  intro h
  have : p / 2 = 2 := h
  have : p = 4 := by
    linarith
  exact this


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincidence_l1322_132290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_earnings_l1322_132222

def earnings_pattern : ℕ → ℕ
| n => match n % 7 with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | 5 => 6
  | _ => 7

def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map earnings_pattern |>.sum

theorem jason_earnings :
  total_earnings 39 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_earnings_l1322_132222
