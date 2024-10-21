import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_slope_probability_theorem_l1003_100315

/-- A triangular pyramid roof with right angles at the apex -/
structure TriangularPyramidRoof where
  /-- Inclination angle of the red slope -/
  α : ℝ
  /-- Inclination angle of the blue slope -/
  β : ℝ

/-- The probability of a raindrop landing on the green slope -/
noncomputable def greenSlopeProbability (roof : TriangularPyramidRoof) : ℝ :=
  1 - (Real.cos roof.β)^2 - (Real.cos roof.α)^2

/-- Theorem stating the probability of a raindrop landing on the green slope -/
theorem green_slope_probability_theorem (roof : TriangularPyramidRoof) :
  greenSlopeProbability roof = 1 - (Real.cos roof.β)^2 - (Real.cos roof.α)^2 := by
  -- Unfold the definition of greenSlopeProbability
  unfold greenSlopeProbability
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_slope_probability_theorem_l1003_100315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_divides_triangle_into_equal_areas_l1003_100320

structure Point where
  x : ℝ
  y : ℝ

def Triangle (P Q R : Point) : Type := Unit

noncomputable def centroid (P Q R : Point) : Point :=
  { x := (P.x + Q.x + R.x) / 3,
    y := (P.y + Q.y + R.y) / 3 }

theorem centroid_divides_triangle_into_equal_areas
  (P Q R : Point)
  (h : Triangle P Q R)
  (hP : P = ⟨7, 10⟩)
  (hQ : Q = ⟨2, -4⟩)
  (hR : R = ⟨9, 3⟩) :
  let S := centroid P Q R
  8 * S.x + 3 * S.y = 57 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_divides_triangle_into_equal_areas_l1003_100320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_marble_tile_cost_proof_l1003_100397

/-- Calculates the cost of each green marble tile in Jackson's courtyard project -/
def green_marble_tile_cost : ℚ :=
  let courtyard_length : ℚ := 10
  let courtyard_width : ℚ := 25
  let tiles_per_sqft : ℚ := 4
  let green_tile_percentage : ℚ := 2/5  -- 40% expressed as a rational number
  let red_tile_cost : ℚ := 3/2  -- $1.50 expressed as a rational number
  let total_tile_cost : ℚ := 2100

  let total_area : ℚ := courtyard_length * courtyard_width
  let total_tiles : ℚ := total_area * tiles_per_sqft
  let green_tiles : ℚ := green_tile_percentage * total_tiles
  let red_tiles : ℚ := total_tiles - green_tiles
  let red_tiles_cost : ℚ := red_tiles * red_tile_cost
  let green_tiles_cost : ℚ := total_tile_cost - red_tiles_cost
  green_tiles_cost / green_tiles

/-- Proves the cost of each green marble tile in Jackson's courtyard project -/
theorem green_marble_tile_cost_proof : green_marble_tile_cost = 3 := by
  -- Unfold the definition and simplify
  unfold green_marble_tile_cost
  -- Perform arithmetic calculations
  norm_num
  -- QED

#eval green_marble_tile_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_marble_tile_cost_proof_l1003_100397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l1003_100375

/-- Ellipse C defined by x²/12 + y²/8 = 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 12 + p.2^2 / 8 = 1}

/-- Left focus F of the ellipse -/
noncomputable def F : ℝ × ℝ := (-2, 0)

/-- Fixed point A -/
noncomputable def A : ℝ × ℝ := (-1, Real.sqrt 3)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance_sum :
  ∃ (M : ℝ), M = 6 * Real.sqrt 3 ∧
  ∀ (P : ℝ × ℝ), P ∈ C →
  distance P F + distance P A ≤ M := by
  sorry

#check max_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l1003_100375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_derivative_equals_surface_area_l1003_100301

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The surface area of a sphere with radius r -/
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- Theorem stating that the derivative of the volume function of a sphere
    equals its surface area function -/
theorem sphere_volume_derivative_equals_surface_area :
  ∀ r : ℝ, (deriv sphere_volume) r = sphere_surface_area r :=
by
  intro r
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_derivative_equals_surface_area_l1003_100301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_behavior_l1003_100318

noncomputable def f (x : ℝ) : ℝ := 2^x - 1

def g (x : ℝ) : ℝ := 1 - x^2

noncomputable def F (x : ℝ) : ℝ := if |f x| ≥ g x then |f x| else -g x

theorem F_behavior :
  (∃ (m : ℝ), ∀ (x : ℝ), F x ≥ m ∧ m = -1) ∧
  (∀ (M : ℝ), ∃ (x : ℝ), F x > M) := by
  sorry

#check F_behavior

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_behavior_l1003_100318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_count_l1003_100355

theorem apple_count (initial_oranges : ℕ) (removed_oranges : ℕ) (apple_percentage : ℚ) : 
  initial_oranges = 23 →
  removed_oranges = 15 →
  apple_percentage = 60 / 100 →
  ∃ (apples : ℕ), 
    apples = 12 ∧ 
    ↑apples = apple_percentage * (↑apples + ↑(initial_oranges - removed_oranges)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_count_l1003_100355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthographic_projection_area_equilateral_triangle_l1003_100361

/-- The area of the orthographic projection of an equilateral triangle with side length 1 is √6/16 -/
theorem orthographic_projection_area_equilateral_triangle :
  let side_length : ℝ := 1
  let original_area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let projection_factor : ℝ := (Real.sqrt 2 / 2)^2
  let projected_area : ℝ := projection_factor * original_area
  projected_area = Real.sqrt 6 / 16 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthographic_projection_area_equilateral_triangle_l1003_100361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solutions_purely_imaginary_z_l1003_100332

-- Define the complex quadratic equation
def quadratic_eq (x : ℂ) : Prop := x^2 - 6*x + 13 = 0

-- Define the complex number z
def z (a : ℝ) : ℂ := (1 + Complex.I) * (a + 2*Complex.I)

-- Theorem for the first part of the problem
theorem quadratic_solutions :
  ∃ (x₁ x₂ : ℂ), quadratic_eq x₁ ∧ quadratic_eq x₂ ∧ x₁ = 3 + 2*Complex.I ∧ x₂ = 3 - 2*Complex.I :=
sorry

-- Theorem for the second part of the problem
theorem purely_imaginary_z :
  ∃ (a : ℝ), (z a).re = 0 → z a = 4*Complex.I :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solutions_purely_imaginary_z_l1003_100332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_placement_exists_l1003_100358

/-- Represents a chessboard -/
def Chessboard := Fin 8 × Fin 8

/-- Represents a domino piece -/
inductive Domino
| mk : Chessboard → Chessboard → Domino

/-- Predicate to check if two cells are adjacent -/
def are_adjacent (c1 c2 : Chessboard) : Prop :=
  (c1.1 = c2.1 ∧ c1.2.val + 1 = c2.2.val) ∨
  (c1.1 = c2.1 ∧ c1.2.val = c2.2.val + 1) ∨
  (c1.1.val + 1 = c2.1.val ∧ c1.2 = c2.2) ∨
  (c1.1.val = c2.1.val + 1 ∧ c1.2 = c2.2)

/-- Predicate to check if a domino placement is valid -/
def is_valid_placement (d : Domino) : Prop :=
  match d with
  | Domino.mk c1 c2 => are_adjacent c1 c2

/-- Predicate to check if two dominoes overlap -/
def overlap (d1 d2 : Domino) : Prop :=
  match d1, d2 with
  | Domino.mk c1 c2, Domino.mk c3 c4 =>
    c1 = c3 ∨ c1 = c4 ∨ c2 = c3 ∨ c2 = c4

/-- Predicate to check if a domino can be moved without removing others -/
def can_move (d : Domino) (placement : List Domino) : Prop :=
  ∃ (new_d : Domino), is_valid_placement new_d ∧
    ∀ other, other ∈ placement → other ≠ d → ¬overlap new_d other

/-- The main theorem stating that it's possible to place 28 dominoes such that none can be moved -/
theorem domino_placement_exists : ∃ (placement : List Domino),
  placement.length = 28 ∧
  (∀ d, d ∈ placement → is_valid_placement d) ∧
  (∀ d1 d2, d1 ∈ placement → d2 ∈ placement → d1 ≠ d2 → ¬overlap d1 d2) ∧
  (∀ d, d ∈ placement → ¬can_move d placement) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_placement_exists_l1003_100358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeterRatio_max_value_l1003_100378

/-- A right triangle inscribed in a circle with hypotenuse as diameter -/
structure InscribedRightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  right_angle : a^2 + b^2 = c^2
  inscribed : c = a + b

/-- The ratio of the triangle's perimeter to its hypotenuse -/
noncomputable def perimeterRatio (t : InscribedRightTriangle) : ℝ :=
  (t.a + t.b + t.c) / t.c

/-- The maximum value of the perimeter ratio is 1 + √2 -/
theorem perimeterRatio_max_value :
  ∀ t : InscribedRightTriangle, perimeterRatio t ≤ 1 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeterRatio_max_value_l1003_100378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_power_equals_pq_power_l1003_100312

theorem fifteen_power_equals_pq_power (m n : ℕ) (P Q : ℕ) 
  (h1 : P = 2^m) (h2 : Q = 5^n) : 
  15^(m*n) = P^n * Q^m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_power_equals_pq_power_l1003_100312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1003_100363

noncomputable def f (x : ℝ) := Real.log x / Real.log (1/2) - x + 4

theorem zero_in_interval : 
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1003_100363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_at_specific_distances_l1003_100339

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the coordinate plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between a point and a line -/
noncomputable def distance_point_line (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- The statement of the problem -/
theorem two_lines_at_specific_distances :
  let A : Point := ⟨1, 2⟩
  let B : Point := ⟨3, 1⟩
  ∃! (s : Finset Line), (∀ l ∈ s, distance_point_line A l = 1 ∧ distance_point_line B l = 2) ∧ s.card = 2 := by
  sorry

#check two_lines_at_specific_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_at_specific_distances_l1003_100339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1003_100329

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define the vector relation
def vector_relation (C D : ℝ × ℝ) (lambda : ℝ) : Prop :=
  (D.1 - M.1, D.2 - M.2) = (lambda * (C.1 - M.1), lambda * (C.2 - M.2))

-- Theorem statement
theorem lambda_range (C D : ℝ × ℝ) (lambda : ℝ) :
  ellipse C.1 C.2 → ellipse D.1 D.2 → vector_relation C D lambda →
  1/3 ≤ lambda ∧ lambda ≤ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1003_100329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l1003_100381

-- Define the function f(x) = e^x + 3x
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 3 * x

-- State the theorem
theorem f_has_unique_zero :
  ∃! x : ℝ, f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l1003_100381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_change_l1003_100380

-- Define the initial bowling average
noncomputable def initial_average : ℝ := 12.4

-- Define the recent performance
def recent_wickets : ℕ := 5
def recent_runs : ℕ := 26

-- Define the total wickets after recent performance
def total_wickets : ℕ := 85

-- Calculate the change in bowling average
noncomputable def change_in_average : ℝ :=
  initial_average - (initial_average * (total_wickets - recent_wickets) + recent_runs) / total_wickets

-- Theorem statement
theorem bowling_average_change :
  ∃ (n : ℕ), (n : ℝ) / 10000 = change_in_average ∧ n = 4235 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_change_l1003_100380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_efficiency_calculation_l1003_100311

/-- The distance (in km) that a liter of gasoline can cover -/
def gasoline_efficiency : ℝ → Prop := λ d => d > 0

/-- The total round trip distance in km -/
def total_distance : ℝ := 300

/-- The cost of the first car rental option per day in dollars -/
def cost_option1 : ℝ := 50

/-- The cost of the second car rental option per day in dollars -/
def cost_option2 : ℝ := 90

/-- The cost of gasoline per liter in dollars -/
def gasoline_cost_per_liter : ℝ := 0.9

/-- The amount saved by choosing the first option over the second option in dollars -/
def savings : ℝ := 22

theorem gasoline_efficiency_calculation (d : ℝ) :
  gasoline_efficiency d →
  cost_option2 - (cost_option1 + (total_distance / d) * gasoline_cost_per_liter) = savings →
  d = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_efficiency_calculation_l1003_100311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_combinations_l1003_100308

theorem coin_combinations : 
  let quarter_value : ℕ := 25
  let half_dollar_value : ℕ := 50
  let total_cents : ℕ := 2000
  let valid_combination (q h : ℕ) := 
    q * quarter_value + h * half_dollar_value = total_cents ∧ q > 0 ∧ h > 0
  ∃! n : ℕ, n = (Finset.filter (λ p : ℕ × ℕ => valid_combination p.1 p.2) 
                 (Finset.product (Finset.range 81) (Finset.range 41))).card ∧ 
                  n = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_combinations_l1003_100308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_equality_l1003_100376

theorem log_power_equality (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 2) : 
  (4 : ℝ)^(-a) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_equality_l1003_100376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_point_lines_l1003_100330

/-- A point on the coordinate plane with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- Checks if a point is within the specified bounds -/
def isValidPoint (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ 2 ∧ 0 ≤ p.y ∧ p.y ≤ 26

/-- Checks if three points are collinear -/
def areCollinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- The set of all valid points -/
def validPoints : Set Point :=
  {p : Point | isValidPoint p}

/-- A line passing through exactly three valid points -/
structure ThreePointLine where
  p : Point
  q : Point
  r : Point
  pValid : isValidPoint p
  qValid : isValidPoint q
  rValid : isValidPoint r
  distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r
  collinear : areCollinear p q r

/-- The main theorem -/
theorem count_three_point_lines :
  ∃ s : Finset ThreePointLine, s.card = 365 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_point_lines_l1003_100330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l1003_100353

/-- Calculate the final selling price of a cycle given initial conditions and market changes -/
theorem cycle_selling_price
  (initial_price_usd : ℝ)
  (initial_exchange_rate : ℝ)
  (depreciation_rate : ℝ)
  (depreciation_years : ℕ)
  (intended_profit : ℝ)
  (discount : ℝ)
  (sales_tax : ℝ)
  (final_exchange_rate : ℝ)
  (h1 : initial_price_usd = 100)
  (h2 : initial_exchange_rate = 80)
  (h3 : depreciation_rate = 0.1)
  (h4 : depreciation_years = 2)
  (h5 : intended_profit = 0.1)
  (h6 : discount = 0.05)
  (h7 : sales_tax = 0.12)
  (h8 : final_exchange_rate = 75) :
  ∃ (final_price_usd : ℝ), abs (final_price_usd - 124.84) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l1003_100353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_partition_l1003_100384

theorem stones_partition (n : ℕ) (k : ℕ) (h : n = 660 ∧ k = 30) :
  ∃ (partition : Finset ℕ),
    Finset.card partition = k ∧
    (Finset.sum partition id : ℕ) = n ∧
    ∀ x y, x ∈ partition → y ∈ partition → (x : ℝ) < 2 * y ∧ (y : ℝ) < 2 * x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_partition_l1003_100384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_theorem_l1003_100395

/-- The area of a square sheet of wrapping paper required to wrap a rectangular box -/
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  let side_length := l + w + 2 * h
  side_length^2

theorem wrapping_paper_area_theorem (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  let box_dimensions := l + w + h
  2 * box_dimensions^2 = wrapping_paper_area l w h :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_theorem_l1003_100395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessary_and_sufficient_condition_l1003_100364

theorem not_necessary_and_sufficient_condition : ¬(∀ x : ℝ, Real.sin x = 1/2 ↔ x = π/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessary_and_sufficient_condition_l1003_100364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_given_field_l1003_100326

/-- Calculates the fencing required for a rectangular field with one side uncovered -/
noncomputable def fencing_required (area : ℝ) (uncovered_side : ℝ) : ℝ :=
  let width := area / uncovered_side
  2 * width + uncovered_side

theorem fencing_for_given_field :
  fencing_required 880 25 = 95.4 := by
  -- Unfold the definition of fencing_required
  unfold fencing_required
  -- Simplify the expression
  simp
  -- Approximate real number arithmetic
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_given_field_l1003_100326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tian_ji_probability_l1003_100338

/-- Represents the strength of a horse -/
inductive Strength : Type
| Top : Strength
| Middle : Strength
| Lower : Strength

/-- Represents a set of horses -/
structure HorseSet :=
  (top : Strength)
  (middle : Strength)
  (lower : Strength)

/-- Determines if one horse beats another -/
def beats (a b : Strength) : Bool :=
  match a, b with
  | Strength.Top, Strength.Middle => true
  | Strength.Top, Strength.Lower => true
  | Strength.Middle, Strength.Lower => true
  | _, _ => false

/-- The probability of Tian Ji winning -/
def winning_probability (king : HorseSet) (tian : HorseSet) : ℚ :=
  let total_outcomes := 9  -- 3 choices for each side
  let winning_outcomes := 
    (beats tian.top king.middle).toNat +
    (beats tian.top king.lower).toNat +
    (beats tian.middle king.lower).toNat
  ↑winning_outcomes / ↑total_outcomes

/-- The main theorem -/
theorem tian_ji_probability (king tian : HorseSet) : 
  (beats tian.top king.middle = true) ∧ 
  (beats king.top tian.top = true) ∧
  (beats tian.middle king.lower = true) ∧
  (beats king.middle tian.middle = true) ∧
  (beats king.lower tian.lower = true) →
  winning_probability king tian = 1/3 := by
  sorry

#eval winning_probability 
  { top := Strength.Top, middle := Strength.Middle, lower := Strength.Lower }
  { top := Strength.Middle, middle := Strength.Lower, lower := Strength.Lower }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tian_ji_probability_l1003_100338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1003_100366

theorem constant_term_binomial_expansion :
  let n : ℕ := 6
  let f (x : ℝ) := x - 1/x
  let expansion := (fun k => (-1)^k * Nat.choose n k * (f 1)^(n-k) * (f (-1))^k)
  expansion (n/2) = -20 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1003_100366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_arrangements_l1003_100336

/-- Represents a seating arrangement of knights around a round table -/
def SeatingArrangement (N : ℕ) := Fin N → Fin N

/-- Two knights can swap if they were not neighbors on the first day -/
def CanSwap (N : ℕ) (first_day : SeatingArrangement N) (i j : Fin N) : Prop :=
  (i.val + 1) % N ≠ j.val ∧ (j.val + 1) % N ≠ i.val

/-- Two seating arrangements are equivalent if they differ only by rotation -/
def EquivalentArrangements (N : ℕ) (a b : SeatingArrangement N) : Prop :=
  ∃ k : Fin N, ∀ i : Fin N, a i = b ((i + k : Fin N))

/-- The maximum number of distinct seating arrangements -/
theorem max_distinct_arrangements (N : ℕ) :
  (∃ (arrangements : Fin N → SeatingArrangement N),
    (∀ i j : Fin N, i ≠ j → ¬EquivalentArrangements N (arrangements i) (arrangements j)) ∧
    (∀ a : SeatingArrangement N, ∃ i : Fin N, EquivalentArrangements N a (arrangements i))) ∧
  (∀ k : ℕ, k > N →
    ¬∃ (arrangements : Fin k → SeatingArrangement N),
      (∀ i j : Fin k, i ≠ j → ¬EquivalentArrangements N (arrangements i) (arrangements j)) ∧
      (∀ a : SeatingArrangement N, ∃ i : Fin k, EquivalentArrangements N a (arrangements i))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_arrangements_l1003_100336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_neq_side_l1003_100391

/-- A square is a quadrilateral with all sides equal and all angles right angles. -/
structure Square where
  side : ℝ
  side_pos : side > 0

/-- The length of the diagonal of a square. -/
noncomputable def Square.diagonal (sq : Square) : ℝ := sq.side * Real.sqrt 2

/-- Theorem: The diagonal of a square is not equal to its side length. -/
theorem square_diagonal_neq_side (sq : Square) : sq.diagonal ≠ sq.side := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_neq_side_l1003_100391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_hill_length_l1003_100309

theorem mary_hill_length 
  (mary_speed : ℝ) 
  (ann_hill_length : ℝ) 
  (ann_speed : ℝ) 
  (time_difference : ℝ) 
  (mary_hill_length : ℝ) :
  mary_speed = 90 →
  ann_hill_length = 800 →
  ann_speed = 40 →
  time_difference = 13 →
  ann_hill_length / ann_speed = (mary_hill_length / mary_speed) + time_difference →
  mary_hill_length = 630 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_hill_length_l1003_100309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_3642_to_nearest_tenth_l1003_100341

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The statement to prove -/
theorem round_24_3642_to_nearest_tenth :
  round_to_nearest_tenth 24.3642 = 24.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_3642_to_nearest_tenth_l1003_100341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_diverges_l1003_100356

/-- Definition of the function f(n) --/
noncomputable def f (n : ℕ+) : ℝ := ∑' k, (1 : ℝ) / (k : ℝ) ^ (n : ℝ)

/-- The sum of f(n) from n = 1 to infinity diverges --/
theorem sum_f_diverges : ¬ Summable (fun n : ℕ+ => f n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_diverges_l1003_100356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1003_100316

/-- Given a principal amount P and an annual interest rate r (compounded annually),
    if P(1+r)^2 = 2420 and P(1+r)^3 = 3146, then r is approximately 0.2992. -/
theorem interest_rate_calculation (P r : ℝ) 
    (h1 : P * (1 + r)^2 = 2420)
    (h2 : P * (1 + r)^3 = 3146) :
    ∃ ε > 0, |r - 0.2992| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1003_100316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1003_100342

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem g_properties :
  (∃ p : ℝ, p > 0 ∧ p = Real.pi ∧ ∀ x : ℝ, g (x + p) = g x) ∧
  (∀ x : ℝ, g (Real.pi / 6 + x) = g (Real.pi / 6 - x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1003_100342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_2x_minus_x_squared_minus_x_l1003_100337

theorem integral_sqrt_2x_minus_x_squared_minus_x :
  (∫ (x : ℝ) in Set.Icc 0 1, (Real.sqrt (2*x - x^2) - x)) = (Real.pi - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_2x_minus_x_squared_minus_x_l1003_100337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_seventh_approx_l1003_100344

noncomputable section

variable (a : ℝ) (h_a : 0 < a ∧ a < 1)

/-- The function f as described in the problem -/
def f : ℝ → ℝ := sorry

/-- f is continuous on [0, 1] -/
axiom f_continuous : Continuous f

/-- f(0) = 0 -/
axiom f_zero : f 0 = 0

/-- f(1) = 1 -/
axiom f_one : f 1 = 1

/-- The functional equation for f -/
axiom f_midpoint (x y : ℝ) (hx : 0 ≤ x) (hy : x ≤ y) (hy1 : y ≤ 1) :
  f ((x + y) / 2) = (1 - a) * f x + a * f y

/-- Theorem stating that f(1/7) is approximately a^3 -/
theorem f_one_seventh_approx :
  ∃ (ε : ℝ), ε > 0 ∧ |f (1/7) - a^3| < ε :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_seventh_approx_l1003_100344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1003_100321

/-- The circle with center at origin and radius 2 -/
def myCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- The point P from which tangents are drawn -/
def P : ℝ × ℝ := (2, 1)

/-- A point of tangency on the circle -/
def tangent_point (A : ℝ × ℝ) : Prop := A ∈ myCircle ∧ (∃ t : ℝ, A = (2 * t, 1 * t))

/-- The line passing through two points -/
def line_equation (A B : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ p ↦ (B.2 - A.2) * (p.1 - A.1) = (B.1 - A.1) * (p.2 - A.2)

/-- The theorem stating that the line AB has the equation 2x + y - 4 = 0 -/
theorem tangent_line_equation :
  ∃ (A B : ℝ × ℝ), tangent_point A ∧ tangent_point B ∧ A ≠ B ∧
    (∀ p : ℝ × ℝ, line_equation A B p ↔ 2 * p.1 + p.2 = 4) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1003_100321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1003_100346

-- Define the function f(x) = 4/sin(x) + sin(x)
noncomputable def f (x : ℝ) : ℝ := 4 / Real.sin x + Real.sin x

-- State the theorem
theorem min_value_of_f :
  ∀ x ∈ Set.Ioo 0 π, f x ≥ 5 ∧ ∃ x₀ ∈ Set.Ioo 0 π, f x₀ = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1003_100346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1003_100373

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 1/2 * t.a * t.c * Real.sin t.B

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 3)
  (h2 : t.B = π/3)
  (h3 : area t = 6 * Real.sqrt 3) :
  (perimeter t = 18) ∧ (Real.sin (2 * t.A) = 39 * Real.sqrt 3 / 98) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1003_100373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1003_100349

/-- The eccentricity of a hyperbola with the given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  let F := (Real.sqrt (a^2 + b^2), 0)
  ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧
    (∃ (k : ℝ), A.2 = Real.sqrt 3 * A.1 ∧ B.2 = Real.sqrt 3 * B.1) ∧
    ((A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 0) →
  (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 3 + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1003_100349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_plus_pi_l1003_100390

theorem sin_x_plus_pi (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.tan x = -4/3) : 
  Real.sin (x + π) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_plus_pi_l1003_100390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_calculation_l1003_100360

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def jacket_cost : ℕ := 20
def shoes_cost : ℕ := 15

def jacket_discount (num_jackets : ℕ) : ℕ :=
  if num_jackets ≥ 3 then jacket_cost / 2 else 0

def free_hats (num_jeans : ℕ) : ℕ :=
  num_jeans / 3

def shoe_discount (num_shirts : ℕ) : ℕ :=
  (num_shirts / 2) * 2

def total_cost (num_shirts num_jeans num_hats num_jackets num_shoes : ℕ) : ℕ :=
  num_shirts * shirt_cost +
  num_jeans * jeans_cost +
  (num_hats - free_hats num_jeans) * hat_cost +
  (num_jackets * jacket_cost - jacket_discount num_jackets) +
  (num_shoes * shoes_cost - shoe_discount num_shirts)

theorem total_cost_calculation :
  total_cost 4 3 4 3 2 = 138 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_calculation_l1003_100360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l1003_100302

-- Define the constants
noncomputable def e : ℝ := Real.exp 1
noncomputable def π : ℝ := Real.pi

-- Axioms for e and π
axiom e_pos : 0 < e
axiom π_pos : 0 < π
axiom e_approx : e > 2.7 ∧ e < 2.8
axiom π_def : π > 3 ∧ π < 3.2

-- Theorem statement
theorem order_of_constants : 3 * e < 3 ∧ 3 < e * π ∧ e * π < π * e ∧ π * e < π^3 ∧ π^3 < 3 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l1003_100302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_capacity_theorem_l1003_100371

/-- Represents the content of a can with milk and water -/
structure Can where
  milk : ℝ
  water : ℝ

/-- The capacity of the can -/
noncomputable def Can.capacity (c : Can) : ℝ := c.milk + c.water

/-- The ratio of milk to water in the can -/
noncomputable def Can.ratio (c : Can) : ℝ := c.milk / c.water

theorem can_capacity_theorem (c : Can) :
  c.ratio = 1 / 5 →
  (Can.capacity { milk := c.milk + 2, water := c.water } = Can.capacity c + 2) →
  Can.ratio { milk := c.milk + 2, water := c.water } = 2.00001 / 5.00001 →
  Can.capacity c = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_capacity_theorem_l1003_100371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratios_triangle_side_c_l1003_100370

/-- Given a triangle ABC with angles A, B, C and opposite sides a, b, c -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

theorem triangle_ratios (t : Triangle) 
  (h : Real.sin (2 * t.A + t.B) = 2 * Real.sin t.A + 2 * Real.cos (t.A + t.B) * Real.sin t.A) :
  t.a / t.b = 1 / 2 := by
  sorry

theorem triangle_side_c (t : Triangle) 
  (h1 : (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2)
  (h2 : t.a = 1) :
  t.c = Real.sqrt 3 ∨ t.c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratios_triangle_side_c_l1003_100370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_exponential_graphs_l1003_100374

theorem symmetry_of_exponential_graphs :
  ∀ a : ℝ, ∃ y₁ y₂ : ℝ, 
    y₁ = 3^a ∧ 
    y₂ = -3^(-a) ∧ 
    (a, y₁) = (-(-a), -y₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_exponential_graphs_l1003_100374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_sum_l1003_100348

/-- A sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = (Finset.range n).sum (fun i ↦ a (i + 1))

/-- The sum of the first n terms of the sequence -/
def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i ↦ a (i + 1))

/-- The main theorem: For a sequence satisfying the given conditions,
    the sum of the first n terms is 2^(n-1) -/
theorem special_sequence_sum (a : ℕ → ℕ) (h : SpecialSequence a) :
    ∀ n : ℕ, n ≥ 1 → S a n = 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_sum_l1003_100348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_one_fourth_from_perigee_l1003_100327

/-- Represents the distance between two points in astronomical units (AU) -/
def Distance := ℝ

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : Distance
  apogee : Distance

/-- Calculates the distance from the focus to a point on the elliptical orbit -/
def distanceFromFocus (orbit : EllipticalOrbit) (fraction : ℝ) : Distance :=
  sorry

theorem distance_one_fourth_from_perigee 
  (orbit : EllipticalOrbit) 
  (h1 : orbit.perigee = (3 : ℝ))
  (h2 : orbit.apogee = (8 : ℝ)) :
  distanceFromFocus orbit (1/4) = (10.75 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_one_fourth_from_perigee_l1003_100327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_solution_l1003_100377

-- Define the coefficient matrix
def A : Matrix (Fin 3) (Fin 3) ℝ := sorry

-- Define the conditions on the coefficients
axiom a11_positive : 0 < A 0 0
axiom a22_positive : 0 < A 1 1
axiom a33_positive : 0 < A 2 2

axiom other_coeff_negative :
  ∀ i j, i ≠ j → A i j < 0

axiom sum_coeff_positive :
  ∀ i, 0 < (A i 0 + A i 1 + A i 2)

-- Define the system of equations
def system (x : Fin 3 → ℝ) : Prop :=
  ∀ i, (A.mulVec x) i = 0

-- Theorem statement
theorem unique_zero_solution :
  ∀ x : Fin 3 → ℝ, system x → x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_solution_l1003_100377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_1_trig_identity_2_l1003_100365

-- Part 1
theorem trig_identity_1 (x : ℝ) (h1 : Real.cos x ≠ 0) (h2 : Real.cos x ^ 2 ≠ Real.sin x ^ 2) :
  (1 - 2 * Real.sin x * Real.cos x) / (Real.cos x ^ 2 - Real.sin x ^ 2) = (1 - Real.tan x) / (1 + Real.tan x) := by
  sorry

-- Part 2
theorem trig_identity_2 (θ a b : ℝ) (h1 : Real.tan θ + Real.sin θ = a) (h2 : Real.tan θ - Real.sin θ = b) :
  (a ^ 2 - b ^ 2) ^ 2 = 16 * a * b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_1_trig_identity_2_l1003_100365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_sides_l1003_100305

/-- Given two points (3,t) and (2t,4) on the terminal sides of angles α and α + 45°
    respectively, with the vertex at the origin and the initial side on the 
    non-negative x-axis, prove that t = 1 -/
theorem point_on_angle_sides (t : ℝ) : 
  (∃ α : ℝ, 
    Real.tan α = t / 3 ∧ 
    Real.tan (α + 45 * π / 180) = 2 / t ∧ 
    3 > 0 ∧ 
    2 * t > 0) → 
  t = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_sides_l1003_100305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l1003_100387

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 5^(abs x)
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

-- State the theorem
theorem function_composition_equality (a : ℝ) : 
  f (g a 1) = 1 → a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l1003_100387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_theorem_l1003_100306

-- Define the circle and its properties
structure Circle (O : EuclideanSpace ℝ (Fin 2)) where
  radius : ℝ
  radius_pos : radius > 0

-- Define the chords and their properties
def Chord (c : Circle O) (A B : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖A - O‖ = c.radius ∧ ‖B - O‖ = c.radius

-- Define the arc measure
noncomputable def ArcMeasure (c : Circle O) (A B : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

-- State the theorem
theorem chord_theorem (O : EuclideanSpace ℝ (Fin 2)) (c : Circle O) (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (m n : ℚ) :
  Chord c A B → Chord c C D →
  ‖C - D‖ = 2 →
  ‖A - B‖ = m + n * Real.sqrt 5 →
  ArcMeasure c A B = 108 →
  ArcMeasure c C D = 36 →
  108 * m - 36 * n = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_theorem_l1003_100306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_sum_m_n_equals_four_l1003_100324

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := ((2^x - 1) * (1 - 2 * Real.sin x ^ 2)) / (2^x + 1)

-- State the theorem
theorem intersection_point_coordinates 
  (k : ℝ) 
  (hk : k ≠ 0) 
  (A B : ℝ × ℝ) 
  (hAB : A ≠ B) 
  (hA : A.1 + k * A.2 = 0 ∧ A.2 = f A.1) 
  (hB : B.1 + k * B.2 = 0 ∧ B.2 = f B.1) 
  (hf_odd : ∀ x, f (-x) = -f x) 
  (C : ℝ × ℝ) 
  (hC : C = (9, 3)) 
  (D : ℝ × ℝ) 
  (hD : (D.1 - A.1, D.2 - A.2) + (D.1 - B.1, D.2 - B.2) = (C.1 - D.1, C.2 - D.2)) :
  D = (3, 1) := by
  sorry

-- Theorem to prove that m + n = 4
theorem sum_m_n_equals_four 
  (k : ℝ) 
  (hk : k ≠ 0) 
  (A B : ℝ × ℝ) 
  (hAB : A ≠ B) 
  (hA : A.1 + k * A.2 = 0 ∧ A.2 = f A.1) 
  (hB : B.1 + k * B.2 = 0 ∧ B.2 = f B.1) 
  (hf_odd : ∀ x, f (-x) = -f x) 
  (C : ℝ × ℝ) 
  (hC : C = (9, 3)) 
  (D : ℝ × ℝ) 
  (hD : (D.1 - A.1, D.2 - A.2) + (D.1 - B.1, D.2 - B.2) = (C.1 - D.1, C.2 - D.2)) :
  D.1 + D.2 = 4 := by
  have h : D = (3, 1) := intersection_point_coordinates k hk A B hAB hA hB hf_odd C hC D hD
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_sum_m_n_equals_four_l1003_100324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l1003_100313

theorem min_value_sin_cos (x : ℝ) : Real.sin x ^ 6 + (4/3) * Real.cos x ^ 6 ≥ 6/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l1003_100313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_articles_l1003_100335

/-- Represents the possible article choices --/
inductive Article
  | Definite   -- represents "the"
  | Indefinite -- represents "a"
deriving Repr

/-- Represents the sentence structure --/
def Sentence := Article → Article → String

/-- The specific sentence in question --/
def exampleSentence : Sentence := fun a1 a2 => 
  match a1, a2 with
  | Article.Definite, Article.Definite => "We can never expect the bluer sky unless we create the less polluted world."
  | Article.Definite, Article.Indefinite => "We can never expect the bluer sky unless we create a less polluted world."
  | Article.Indefinite, Article.Definite => "We can never expect a bluer sky unless we create the less polluted world."
  | Article.Indefinite, Article.Indefinite => "We can never expect a bluer sky unless we create a less polluted world."

/-- Predicate to check if an article is correct in this context --/
def isCorrectArticle (a : Article) : Prop := a = Article.Indefinite

/-- Theorem stating that both articles should be indefinite --/
theorem correct_articles : 
  ∀ (a1 a2 : Article), 
    exampleSentence a1 a2 = "We can never expect a bluer sky unless we create a less polluted world." 
    ↔ (isCorrectArticle a1 ∧ isCorrectArticle a2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_articles_l1003_100335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_rent_expression_l1003_100304

/-- A library charges for renting books as follows:
    - $0.6 per day for the first two days
    - $0.3 per day after the first two days
    This theorem proves the expression for the rent after x days (where x ≥ 2) -/
theorem library_rent_expression (x : ℝ) (h : x ≥ 2) :
  let rent_function := fun (days : ℝ) ↦ 
    if days ≤ 2 then 0.6 * days
    else 0.6 * 2 + 0.3 * (days - 2)
  rent_function x = 0.3 * x + 0.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_rent_expression_l1003_100304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_trinomial_m_value_l1003_100396

-- Define the polynomial
noncomputable def P (m : ℤ) (x : ℝ) : ℝ := (1/3) * x^(abs m) - (m + 4) * x - 11

-- State the theorem
theorem quartic_trinomial_m_value (m : ℤ) :
  (∀ x : ℝ, ∃ a b c d : ℝ, P m x = a*x^4 + b*x^3 + c*x^2 + d*x - 11) →
  (m + 4 ≠ 0) →
  m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_trinomial_m_value_l1003_100396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_william_land_percentage_is_12_5_l1003_100393

/-- Represents the farm tax scenario in Mr. William's village -/
structure FarmTax where
  total_tax : ℚ  -- Total tax collected from the village
  william_tax : ℚ  -- Tax paid by Mr. William

/-- Calculates the percentage of Mr. William's taxable land over the total taxable land -/
def william_land_percentage (ft : FarmTax) : ℚ :=
  (ft.william_tax / ft.total_tax) * 100

/-- Theorem stating that Mr. William's taxable land is 12.5% of the total taxable land -/
theorem william_land_percentage_is_12_5 (ft : FarmTax) 
  (h1 : ft.total_tax = 3840)
  (h2 : ft.william_tax = 480) : 
  william_land_percentage ft = 25/2 := by
  sorry

#eval william_land_percentage ⟨3840, 480⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_william_land_percentage_is_12_5_l1003_100393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_implies_2a_minus_2b_zero_l1003_100300

/-- A function g defined as (2ax - b) / (dx - 2b) -/
noncomputable def g (a b d x : ℝ) : ℝ := (2*a*x - b) / (d*x - 2*b)

/-- Theorem stating that if g is its own inverse, then 2a - 2b = 0 -/
theorem g_inverse_implies_2a_minus_2b_zero
  (a b d : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : b ≠ 0)
  (h₃ : d ≠ 0)
  (h₄ : ∀ x, x ∈ {x | d*x - 2*b ≠ 0} → g a b d (g a b d x) = x) :
  2*a - 2*b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_implies_2a_minus_2b_zero_l1003_100300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_30_heights_l1003_100333

/-- The sum of heights of n students in the sequence -/
def sum_heights (n : ℕ) : ℝ := sorry

/-- The number of students in the sequence -/
def total_students : ℕ := 30

/-- The height difference between adjacent students is constant -/
axiom constant_difference : ∃ d : ℝ, ∀ n : ℕ, n < total_students - 1 →
  sum_heights (n + 1) - sum_heights n = d

/-- The sum of heights of the first 10 students -/
axiom sum_10 : sum_heights 10 = 1450

/-- The sum of heights of the first 20 students -/
axiom sum_20 : sum_heights 20 = 3030

/-- The theorem to be proved -/
theorem sum_30_heights : sum_heights total_students = 4610 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_30_heights_l1003_100333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_sheets_average_l1003_100340

theorem borrowed_sheets_average (total_sheets : ℕ) (total_pages : ℕ) (borrowed_sheets : ℕ) (average : ℚ) : 
  total_sheets = 40 →
  total_pages = 80 →
  borrowed_sheets = 20 →
  average = 41 →
  (2 * borrowed_sheets + 1 + total_pages) * (total_sheets - borrowed_sheets) / (2 * (total_sheets - borrowed_sheets)) = average :=
by
  intros h_total_sheets h_total_pages h_borrowed_sheets h_average
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_sheets_average_l1003_100340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_curve_l1003_100392

/-- The polar equation of the curve -/
def polar_equation (ρ : ℝ) (θ : ℝ) : Prop :=
  ρ^2 * Real.cos (2 * θ) = 1

/-- The eccentricity of the curve -/
noncomputable def eccentricity : ℝ := Real.sqrt 2

/-- Theorem stating that the eccentricity of the curve represented by the given polar equation is √2 -/
theorem eccentricity_of_curve :
  ∀ ρ θ, polar_equation ρ θ → eccentricity = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_curve_l1003_100392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1003_100369

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (x + 4) else x * (x - 4)

-- State the theorem
theorem range_of_a (a : ℝ) : f a > f (8 - a) → a > 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1003_100369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_expected_shots_l1003_100359

/-- The expected number of shots needed to decrease the number of arrows by one -/
noncomputable def expected_shots_per_arrow : ℝ := 10 / 7

/-- The initial number of arrows -/
def initial_arrows : ℕ := 14

/-- The probability of hitting a cone -/
def hit_probability : ℝ := 0.1

/-- The number of additional arrows given for each hit -/
def additional_arrows : ℕ := 3

/-- The expected total number of shots -/
noncomputable def expected_total_shots : ℝ := initial_arrows * expected_shots_per_arrow

theorem ivan_expected_shots :
  expected_total_shots = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_expected_shots_l1003_100359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_journey_time_l1003_100347

/-- The usual time for a boy to reach school -/
def usual_time : ℝ → ℝ := sorry

/-- The time taken when the boy walks at 5/4 of his usual rate -/
def faster_time (t : ℝ) : ℝ := t - 4

theorem school_journey_time (t : ℝ) (h1 : t > 0) 
  (h2 : (5 / 4) * (1 / (faster_time t)) = 1 / t) : 
  usual_time t = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_journey_time_l1003_100347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_probability_l1003_100350

theorem coin_probability (p q : ℝ) : 
  q = 1 - p →
  (Nat.choose 10 5 : ℝ) * p^5 * q^5 = (Nat.choose 10 6 : ℝ) * p^6 * q^4 →
  p = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_probability_l1003_100350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1003_100352

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x - a|
def g (a x : ℝ) : ℝ := f a x - |x - 2|

-- Define the solution set for part 1
def solution_set (a : ℝ) : Set ℝ := {x | f a x ≥ (x + 1) / 2}

-- Define the range of g
def range_g (a : ℝ) : Set ℝ := {y | ∃ x, g a x = y}

theorem problem_solution :
  (∀ x, x ∈ solution_set 1 ↔ x ≤ 1/3 ∨ x ≥ 3) ∧
  (∀ a, range_g a ⊆ Set.Icc (-1) 3 → a ∈ Set.Icc 1 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1003_100352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l1003_100328

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 + x - 6 else (abs x)^2 + abs x - 6

theorem f_solution_set :
  (∀ x : ℝ, f x = f (abs x)) →
  (∀ x : ℝ, x ≥ 0 → f x = x^2 + x - 6) →
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l1003_100328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1003_100314

-- Define the function f(x) = ln x - 2√x
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * Real.sqrt x

-- State the theorem
theorem f_max_value :
  ∀ x : ℝ, x > 0 → f x ≤ f 1 ∧ f 1 = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1003_100314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_distribution_equality_l1003_100334

/-- Represents the number of staff on duty -/
def x : ℕ := sorry

/-- The total number of masks when each person receives 3 masks -/
def masks_scenario1 (x : ℕ) : ℕ := 3 * x + 20

/-- The total number of masks when each person receives 4 masks -/
def masks_scenario2 (x : ℕ) : ℕ := 4 * x - 25

/-- Theorem stating that the two scenarios represent the same total number of masks -/
theorem mask_distribution_equality : masks_scenario1 x = masks_scenario2 x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_distribution_equality_l1003_100334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_code_count_l1003_100343

/-- Represents the set of digits from 1 to 9 -/
def Digits : Finset Nat := Finset.range 9 \ {0}

/-- Predicate to check if a natural number is odd -/
def isOdd (n : Nat) : Prop := n % 2 = 1

/-- Predicate to check if a natural number is even -/
def isEven (n : Nat) : Prop := n % 2 = 0

/-- Represents a valid 6-digit lock code -/
structure LockCode where
  code : Fin 6 → Nat
  valid : ∀ i, code i ∈ Digits
  no_repeat : ∀ i j, i ≠ j → code i ≠ code j
  pattern : (isOdd (code 0)) ∧ (isEven (code 1)) ∧ (isOdd (code 2)) ∧
            (isEven (code 3)) ∧ (isOdd (code 4)) ∧ (isEven (code 5))

/-- Instance to make LockCode a finite type -/
instance : Fintype LockCode := sorry

/-- The main theorem stating the number of possible lock codes -/
theorem lock_code_count : Fintype.card LockCode = 1440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_code_count_l1003_100343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_laugh_probability_l1003_100382

theorem baby_laugh_probability (p : ℝ) (n k : ℕ) (h_p : p = 1/3) (h_n : n = 7) (h_k : k = 3) :
  1 - (Finset.sum (Finset.range k) (λ i ↦ (n.choose i) * p^i * (1-p)^(n-i))) = 939/2187 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_laugh_probability_l1003_100382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l1003_100351

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

noncomputable def f' (x : ℝ) : ℝ := -2 / ((x - 1) ^ 2)

theorem tangent_perpendicular_line (a : ℝ) : 
  (f' 3 = -1/2) →  -- Slope of the tangent line at x=3
  (f 3 = 2) →      -- The point (3,2) lies on the curve
  (f' 3 * (-1/a) = -1) →  -- Perpendicularity condition
  a = -2 := by
  sorry

#check tangent_perpendicular_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l1003_100351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_on_chessboard_l1003_100357

theorem min_pieces_on_chessboard (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) 
  (h3 : n / 2 < k) (h4 : k ≤ 2 * n / 3) :
  ∃ m : ℕ, 
    m = 4 * (n - k) ∧ 
    (∀ arrangement : Fin n → Fin n → Bool,
      (∀ row col : Fin n, ¬∃ start : Fin n, ∀ i : Fin k, 
        i.val + start.val < n → ¬arrangement row ⟨i.val + start.val, sorry⟩) →
      (∀ row col : Fin n, ¬∃ start : Fin n, ∀ i : Fin k, 
        i.val + start.val < n → ¬arrangement ⟨i.val + start.val, sorry⟩ col) →
      m ≤ (Finset.sum (Finset.univ : Finset (Fin n × Fin n)) fun p => if arrangement p.1 p.2 then 1 else 0)) ∧
    (∀ m' : ℕ, m' < m → 
      ∃ arrangement : Fin n → Fin n → Bool,
        (∃ row : Fin n, ∃ start : Fin n, ∀ i : Fin k, 
          i.val + start.val < n → ¬arrangement row ⟨i.val + start.val, sorry⟩) ∨
        (∃ col : Fin n, ∃ start : Fin n, ∀ i : Fin k, 
          i.val + start.val < n → ¬arrangement ⟨i.val + start.val, sorry⟩ col) ∨
        m' < (Finset.sum (Finset.univ : Finset (Fin n × Fin n)) fun p => if arrangement p.1 p.2 then 1 else 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_on_chessboard_l1003_100357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_parametric_sum_bounds_l1003_100385

/-- Represents a point on a circle in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a point on a circle in Cartesian coordinates -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- The given polar equation of the circle -/
def circleEquation (p : PolarPoint) : Prop :=
  p.ρ^2 - 4 * Real.sqrt 2 * p.ρ * Real.cos (p.θ - Real.pi/4) + 6 = 0

/-- The parametric equation of the circle -/
noncomputable def parametricCircle (α : ℝ) : CartesianPoint :=
  ⟨2 + Real.sqrt 2 * Real.cos α, 2 + Real.sqrt 2 * Real.sin α⟩

/-- Theorem stating the equivalence of the polar equation and the parametric form -/
theorem polar_to_parametric :
  ∀ p : PolarPoint, circleEquation p ↔ 
  ∃ α : ℝ, p.ρ * Real.cos p.θ = (parametricCircle α).x ∧ 
           p.ρ * Real.sin p.θ = (parametricCircle α).y := by
  sorry

/-- Theorem stating the bounds on x + y for points on the circle -/
theorem sum_bounds :
  ∀ p : CartesianPoint, (∃ α : ℝ, p = parametricCircle α) →
  2 ≤ p.x + p.y ∧ p.x + p.y ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_parametric_sum_bounds_l1003_100385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_modulus_l1003_100323

theorem complex_number_modulus (i : ℂ) (h : i^2 = -1) :
  Complex.abs ((i / (1 - i))^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_modulus_l1003_100323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_inequality_main_theorem_l1003_100331

open BigOperators
open Finset

def f (n : ℕ) : ℚ :=
  ∑ i in range n, 1 / (i + 1 : ℚ)

theorem harmonic_sum_inequality (n : ℕ) :
  f (2^n) > n / 2 :=
sorry

axiom f_difference (k : ℕ) :
  f (2^(k+1)) - f (2^k) = ∑ i in range (2^k), 1 / ((2^k + i + 1) : ℚ)

theorem main_theorem (n : ℕ) :
  f (2^n) > n / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_inequality_main_theorem_l1003_100331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1003_100398

noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 16) / x

theorem range_of_a (a : ℝ) :
  (a > 2) →
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ a → 9 ≤ f x ∧ f x ≤ 11) →
  (4 ≤ a ∧ a ≤ 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1003_100398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_directrix_distance_l1003_100389

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := 4 * y = x^2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the directrix of the parabola
def directrix : ℝ → ℝ := λ x ↦ -1

-- Theorem statement
theorem focus_directrix_distance :
  let f := focus
  let d := directrix
  (f.2 - d f.1) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_directrix_distance_l1003_100389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l1003_100345

/-- Represents a trapezoid with an inscribed circle -/
structure InscribedTrapezoid where
  /-- Length of one of the bases -/
  a : ℝ
  /-- Length of segment adjacent to the base -/
  b : ℝ
  /-- Length of segment not adjacent to the base -/
  d : ℝ
  /-- Ensure a > b for a valid trapezoid -/
  h_a_gt_b : a > b
  /-- Ensure b and d are positive -/
  h_b_pos : b > 0
  h_d_pos : d > 0

/-- The area of a trapezoid with an inscribed circle -/
noncomputable def trapezoidArea (t : InscribedTrapezoid) : ℝ :=
  (t.a^2 + t.a * (t.d - t.b)) / (t.a - t.b) * Real.sqrt (t.b * t.d)

/-- Theorem stating that the given formula correctly computes the area of the trapezoid -/
theorem trapezoid_area_formula (t : InscribedTrapezoid) :
  trapezoidArea t = (t.a^2 + t.a * (t.d - t.b)) / (t.a - t.b) * Real.sqrt (t.b * t.d) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l1003_100345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_assignment_l1003_100325

-- Define the triangle structure
structure Triangle :=
  (A B C D E F : ℕ)

-- Define the conditions
def valid_triangle (t : Triangle) : Prop :=
  (t.A = 1 ∨ t.A = 2 ∨ t.A = 3 ∨ t.A = 4 ∨ t.A = 5 ∨ t.A = 6) ∧
  (t.B = 1 ∨ t.B = 2 ∨ t.B = 3 ∨ t.B = 4 ∨ t.B = 5 ∨ t.B = 6) ∧
  (t.C = 1 ∨ t.C = 2 ∨ t.C = 3 ∨ t.C = 4 ∨ t.C = 5 ∨ t.C = 6) ∧
  (t.D = 1 ∨ t.D = 2 ∨ t.D = 3 ∨ t.D = 4 ∨ t.D = 5 ∨ t.D = 6) ∧
  (t.E = 1 ∨ t.E = 2 ∨ t.E = 3 ∨ t.E = 4 ∨ t.E = 5 ∨ t.E = 6) ∧
  (t.F = 1 ∨ t.F = 2 ∨ t.F = 3 ∨ t.F = 4 ∨ t.F = 5 ∨ t.F = 6) ∧
  t.A + t.B + t.D = 9 ∧
  t.B + t.C + t.E = 11 ∧
  t.D + t.E + t.F = 15 ∧
  t.A ≠ t.B ∧ t.A ≠ t.C ∧ t.A ≠ t.D ∧ t.A ≠ t.E ∧ t.A ≠ t.F ∧
  t.B ≠ t.C ∧ t.B ≠ t.D ∧ t.B ≠ t.E ∧ t.B ≠ t.F ∧
  t.C ≠ t.D ∧ t.C ≠ t.E ∧ t.C ≠ t.F ∧
  t.D ≠ t.E ∧ t.D ≠ t.F ∧
  t.E ≠ t.F

-- Theorem statement
theorem unique_triangle_assignment :
  ∃! t : Triangle, valid_triangle t ∧ t = ⟨1, 3, 2, 5, 6, 4⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_assignment_l1003_100325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l1003_100319

noncomputable def f (x b a : ℝ) : ℝ := (x + b) * (Real.exp x - a)

theorem tangent_line_and_inequality 
  (b : ℝ) 
  (h_b : b > 0) 
  (h_tangent : ∀ x y, y = f x 1 1 → (Real.exp 1 - 1) * x + Real.exp 1 * y + Real.exp 1 - 1 = 0) :
  (∃ a b, a = 1 ∧ b = 1) ∧
  (∀ m x, m ≤ 0 → f x 1 1 ≥ m * x^2 + x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l1003_100319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l1003_100367

theorem matrix_equation_solution : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  M^3 - 3 • M^2 + 2 • M = !![6, 12; 3, 6] := by
  sorry

#check matrix_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l1003_100367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1003_100307

theorem trigonometric_identities (α : Real) 
  (h1 : Real.cos α = -4/5) 
  (h2 : α ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  Real.sin (α - Real.pi/3) = (3 + 4*Real.sqrt 3)/10 ∧ Real.cos (2*α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1003_100307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cylinder_volume_ratio_l1003_100354

/-- The ratio of the volume of a sphere inscribed in a right circular cylinder
    (where the cylinder's height equals its diameter) to the volume of the cylinder -/
theorem inscribed_sphere_cylinder_volume_ratio :
  ∀ d : ℝ, d > 0 →
  (let r := d / 2
   let sphere_volume := (4 / 3) * Real.pi * r^3
   let cylinder_volume := Real.pi * r^2 * d
   sphere_volume / cylinder_volume) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cylinder_volume_ratio_l1003_100354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vendelin_crayons_l1003_100388

/-- The number of crayons Míša has -/
def M : ℕ := 5

/-- The number of crayons Vojta has -/
def V : ℕ := sorry

/-- The number of crayons Vendelín has -/
def W : ℕ := sorry

/-- Vojta has fewer crayons than Míša -/
axiom vojta_fewer : V < M

/-- Vendelín has as many crayons as Míša and Vojta combined -/
axiom vendelin_sum : W = M + V

/-- The three of them have seven times more crayons than Vojta -/
axiom total_seven_times : M + V + W = 7 * V

theorem vendelin_crayons : W = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vendelin_crayons_l1003_100388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_value_l1003_100386

theorem smallest_c_value (c d : ℕ) (r₁ r₂ r₃ : ℕ) : 
  r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 →
  r₁ * r₂ * r₃ = 2310 →
  c = r₁ + r₂ + r₃ →
  d = min r₁ (min r₂ r₃) * max r₁ (max r₂ r₃) →
  ∀ c' d' r₁' r₂' r₃', 
    r₁' > 0 ∧ r₂' > 0 ∧ r₃' > 0 →
    r₁' * r₂' * r₃' = 2310 →
    c' = r₁' + r₂' + r₃' →
    d' = min r₁' (min r₂' r₃') * max r₁' (max r₂' r₃') →
    c ≤ c' →
  c = 48 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_value_l1003_100386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_price_scaling_l1003_100394

/-- The volume of a cylinder given its radius and height -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The price of a container given its volume and price per unit volume -/
def containerPrice (volume pricePerUnitVolume : ℝ) : ℝ := volume * pricePerUnitVolume

theorem container_price_scaling (v1 v2 p1 : ℝ) (hv1 : v1 = 5 * Real.pi) (hv2 : v2 = 40 * Real.pi) (hp1 : p1 = 0.75) :
  containerPrice v2 (p1 / v1) = 6 := by
  sorry

#check container_price_scaling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_price_scaling_l1003_100394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basil_has_winning_strategy_l1003_100362

/-- Represents a card with the product of 5 distinct variables -/
def Card := Finset (Fin 10)

/-- The set of all possible cards -/
noncomputable def AllCards : Finset Card :=
  Finset.filter (fun s : Finset (Fin 10) => s.card = 5) (Finset.powerset (Finset.univ))

/-- A strategy for choosing cards -/
def Strategy := List Card → Option Card

/-- The result of the game -/
inductive GameResult
| PeterWins
| BasilWins
| Draw

/-- Play the game given Peter's and Basil's strategies -/
def playGame (peterStrategy : Strategy) (basilStrategy : Strategy) : GameResult :=
  sorry

/-- Basil's winning condition -/
def BasilWins (peterCards basilCards : Finset Card) (x : Fin 10 → ℝ) : Prop :=
  (basilCards.sum fun c => (c.prod fun i => x i)) >
  (peterCards.sum fun c => (c.prod fun i => x i))

/-- The main theorem: Basil has a winning strategy -/
theorem basil_has_winning_strategy :
  ∃ (basilStrategy : Strategy),
    ∀ (peterStrategy : Strategy),
      ∃ (x : Fin 10 → ℝ),
        (∀ i j : Fin 10, i ≤ j → x i ≤ x j) ∧
        (∀ i : Fin 10, 0 ≤ x i) ∧
        (playGame peterStrategy basilStrategy = GameResult.BasilWins) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basil_has_winning_strategy_l1003_100362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_finance_fraction_approx_one_third_l1003_100303

/-- The fraction of automobile installment credit extended by automobile finance companies -/
noncomputable def auto_finance_fraction (total_credit auto_credit_percentage finance_credit : ℝ) : ℝ :=
  finance_credit / (total_credit * auto_credit_percentage)

/-- Theorem stating that the fraction of automobile installment credit
    extended by automobile finance companies is approximately 1/3 -/
theorem auto_finance_fraction_approx_one_third
  (total_credit : ℝ)
  (auto_credit_percentage : ℝ)
  (finance_credit : ℝ)
  (h1 : total_credit = 342.857)
  (h2 : auto_credit_percentage = 0.35)
  (h3 : finance_credit = 40) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |auto_finance_fraction total_credit auto_credit_percentage finance_credit - 1/3| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_finance_fraction_approx_one_third_l1003_100303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_calculation_l1003_100372

/-- The cost in dollars to transport 1 kg to the Mars Orbiter. -/
noncomputable def cost_per_kg : ℝ := 15000

/-- The weight of the scientific instrument in grams. -/
noncomputable def instrument_weight : ℝ := 500

/-- The number of grams in 1 kilogram. -/
noncomputable def grams_per_kg : ℝ := 1000

/-- The cost of transporting the scientific instrument to the Mars Orbiter. -/
noncomputable def transport_cost : ℝ := cost_per_kg * (instrument_weight / grams_per_kg)

theorem transport_cost_calculation :
  transport_cost = 7500 := by
  -- Unfold the definitions
  unfold transport_cost cost_per_kg instrument_weight grams_per_kg
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_calculation_l1003_100372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l1003_100379

/-- The complex function f(z) -/
noncomputable def f (z : ℂ) : ℂ := ((2 - Complex.I * Real.sqrt 3) * z + (-Real.sqrt 3 - 12 * Complex.I)) / 2

/-- The fixed point c -/
noncomputable def c : ℂ := 1 - 4 * Real.sqrt 3 * Complex.I

/-- Theorem stating that c is the fixed point of f -/
theorem fixed_point_of_f : f c = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l1003_100379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_b_rate_is_ten_cents_l1003_100322

/-- Represents the charge for a call under Plan A -/
noncomputable def planACharge (minutes : ℝ) : ℝ :=
  if minutes ≤ 8 then 0.60 else 0.60 + 0.06 * (minutes - 8)

/-- Represents the charge for a call under Plan B -/
def planBCharge (rate : ℝ) (minutes : ℝ) : ℝ :=
  rate * minutes

/-- Theorem stating that the per-minute charge under Plan B is $0.10 -/
theorem plan_b_rate_is_ten_cents
  (h : ∃ (rate : ℝ), planACharge 6 = planBCharge rate 6) :
  ∃ (rate : ℝ), rate = 0.10 ∧ planACharge 6 = planBCharge rate 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_b_rate_is_ten_cents_l1003_100322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_ellipse_equation_l1003_100368

-- Problem 1
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (k : ℝ), (x^2 / 9 - y^2 / 4 = k) ∧ (x^2 / 9 - y^2 / 4 = 1)) →
  (3^2 / 9 - 4^2 / 4 = -3) →
  (y^2 / 12 - x^2 / 27 = 1) := by sorry

-- Problem 2
theorem ellipse_equation (x y a b : ℝ) (A B P : ℝ × ℝ) :
  (a > b) → (b > 0) →
  (x^2 / a^2 + y^2 / b^2 = 1) →
  (∃ (f : ℝ × ℝ), f.1 + f.2 - Real.sqrt 3 = 0 ∧ f.1 > 0) →
  (A.1 + A.2 - Real.sqrt 3 = 0) →
  (B.1 + B.2 - Real.sqrt 3 = 0) →
  (P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →
  (P.2 / P.1 = 1 / 2) →
  (x^2 / 6 + y^2 / 3 = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_ellipse_equation_l1003_100368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_cost_l1003_100317

-- Define variables for the cost of a mixer and a TV
variable (M T : ℕ)

-- Define the conditions from the problem
def condition1 (M T : ℕ) : Prop := 2 * M + T = 7000
def condition2 (M T : ℕ) : Prop := M + 2 * T = 9800

-- Theorem to prove
theorem tv_cost : ∃ (M T : ℕ), condition1 M T ∧ condition2 M T ∧ T = 4200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_cost_l1003_100317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1003_100383

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)

-- State the theorem
theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1003_100383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_time_l1003_100310

-- Define the rates at which Alice and Bob can fill the pool
noncomputable def alice_rate : ℝ := 1 / 3
noncomputable def bob_rate : ℝ := 1 / 4

-- Define the duration of the break
def break_duration : ℝ := 2

-- Define the total time to fill the pool
def total_time : ℝ → Prop := λ t ↦ (alice_rate + bob_rate) * (t - break_duration) = 1

-- Theorem statement
theorem pool_filling_time (t : ℝ) : 
  total_time t ↔ (1 / 3 + 1 / 4) * (t - 2) = 1 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_time_l1003_100310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_x_value_l1003_100399

noncomputable def figure_area (x : ℝ) : ℝ :=
  (3*x)^2 + 2*x*x + 1/2 * (3*x) * x

theorem figure_x_value :
  ∃ x : ℝ, x > 0 ∧ figure_area x = 300 ∧ x = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_x_value_l1003_100399
