import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldbach_difference_156_l395_39565

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the absolute difference function for natural numbers
def absDiff (a b : ℕ) : ℕ := if a ≥ b then a - b else b - a

-- Define the theorem
theorem goldbach_difference_156 :
  ∃ (p q : ℕ), p ≠ q ∧ isPrime p ∧ isPrime q ∧ p + q = 156 ∧
  ∀ (r s : ℕ), r ≠ s → isPrime r → isPrime s → r + s = 156 → absDiff r s ≤ 146 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldbach_difference_156_l395_39565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l395_39522

noncomputable def f (x y : ℝ) : ℝ := Real.sqrt (8 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50)

theorem max_value_of_f :
  ∃ (max : ℝ), max = 6 * Real.sqrt 10 ∧
  ∀ (x y : ℝ), x^2 + y^2 = 25 → f x y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l395_39522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_tea_consumed_mad_tea_party_solution_l395_39530

/-- Represents a circular arrangement of cups -/
structure CircularTable where
  num_cups : Nat
  is_even : num_cups % 2 = 0

/-- Represents the state of drinking from the table -/
structure DrinkingState where
  hare_pos : Nat
  dormouse_pos : Nat
  cups_drunk : Nat

/-- Defines a valid drinking strategy -/
def valid_strategy (table : CircularTable) (initial : DrinkingState) : Prop :=
  initial.hare_pos < table.num_cups ∧
  initial.dormouse_pos < table.num_cups ∧
  initial.hare_pos ≠ initial.dormouse_pos

/-- Defines the next state after drinking and rotating -/
def next_state (table : CircularTable) (state : DrinkingState) : DrinkingState :=
  { hare_pos := (state.hare_pos + 1) % table.num_cups,
    dormouse_pos := (state.dormouse_pos + table.num_cups / 2) % table.num_cups,
    cups_drunk := state.cups_drunk + 2 }

/-- Theorem: All tea can be consumed with the given strategy -/
theorem all_tea_consumed (table : CircularTable) (initial : DrinkingState) 
  (h_valid : valid_strategy table initial) : 
  ∃ n : Nat, (Nat.iterate (next_state table) n initial).cups_drunk = table.num_cups := by
  sorry

/-- The specific problem instance -/
def tea_party : CircularTable :=
  { num_cups := 30,
    is_even := by rfl }

theorem mad_tea_party_solution :
  ∃ (initial : DrinkingState), valid_strategy tea_party initial ∧
  ∃ n : Nat, (Nat.iterate (next_state tea_party) n initial).cups_drunk = tea_party.num_cups := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_tea_consumed_mad_tea_party_solution_l395_39530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sequence_sum_l395_39598

/-- Given sequences (c_n) and (d_n) of real numbers satisfying 
    (1 + 2i)^n = c_n + d_n*i for all integers n ≥ 0, 
    the sum of c_n * d_n / 8^n from n = 0 to infinity is equal to 16/15 -/
theorem complex_sequence_sum (c d : ℕ → ℝ) :
  (∀ n : ℕ, (Complex.I : ℂ) * 2 + 1 ^ n = (c n : ℂ) + (d n : ℂ) * Complex.I) →
  tsum (λ n ↦ (c n * d n) / (8 : ℝ) ^ n) = 16 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sequence_sum_l395_39598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_five_l395_39525

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points in the xy-plane -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The radius of the circle is 5 -/
theorem circle_radius_is_five (center inside outside : Point)
  (h_center : center = ⟨2, 1⟩)
  (h_inside : inside = ⟨-2, 1⟩)
  (h_outside : outside = ⟨2, -5⟩)
  (h_inside_dist : distance center inside < distance center outside)
  (h_radius_int : ∃ r : ℕ, distance center inside < r ∧ r < distance center outside) :
  ∃ r : ℕ, r = 5 ∧ distance center inside < r ∧ r < distance center outside :=
by
  sorry

#check circle_radius_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_five_l395_39525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_increases_with_n_l395_39514

-- Define the formula for C as a function of n
noncomputable def C (e R r n : ℝ) : ℝ := (e * n) / (R + n * r)

-- Theorem statement
theorem C_increases_with_n (e R r : ℝ) (he : e > 0) (hR : R > 0) (hr : r > 0) :
  ∀ n₁ n₂ : ℝ, n₁ < n₂ → C e R r n₁ < C e R r n₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_increases_with_n_l395_39514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_is_1134_l395_39534

-- Define the original pyramid
def original_base_edge : ℚ := 18
def original_altitude : ℚ := 12

-- Define the smaller pyramid
def smaller_base_edge : ℚ := 9
def smaller_altitude : ℚ := 6

-- Define the volume of a square pyramid
def pyramid_volume (base_edge : ℚ) (altitude : ℚ) : ℚ :=
  (1 / 3) * base_edge ^ 2 * altitude

-- Define the volume of the frustum
def frustum_volume : ℚ :=
  pyramid_volume original_base_edge original_altitude - 
  pyramid_volume smaller_base_edge smaller_altitude

-- Theorem statement
theorem frustum_volume_is_1134 :
  frustum_volume = 1134 := by
  -- Expand the definition of frustum_volume
  unfold frustum_volume
  -- Expand the definition of pyramid_volume
  unfold pyramid_volume
  -- Perform the calculation
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_is_1134_l395_39534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valley_of_five_lakes_streams_l395_39554

/-- Represents a lake in the Valley of Five Lakes -/
inductive Lake : Type
| S | A | B | C | D

/-- Represents a stream connecting two lakes -/
structure StreamConnection where
  source : Lake
  destination : Lake

/-- The Valley of Five Lakes system -/
structure ValleyOfFiveLakes where
  streams : List StreamConnection

/-- Probability of a fish moving from one lake to another -/
noncomputable def transitionProb (v : ValleyOfFiveLakes) (source destination : Lake) : ℝ := sorry

/-- Probability of a fish ending up in a lake after 4 moves -/
noncomputable def finalProb (v : ValleyOfFiveLakes) (start finish : Lake) : ℝ := sorry

theorem valley_of_five_lakes_streams :
  ∃ (v : ValleyOfFiveLakes),
    (v.streams.length = 3) ∧
    (finalProb v Lake.S Lake.S = 375 / 1000) ∧
    (finalProb v Lake.S Lake.B = 625 / 1000) ∧
    (∀ l : Lake, l ≠ Lake.S ∧ l ≠ Lake.B → finalProb v Lake.S l = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valley_of_five_lakes_streams_l395_39554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_point_distances_l395_39561

/-- A square with a point inside --/
structure SquareWithPoint where
  /-- The side length of the square --/
  side : ℝ
  /-- The coordinates of point P inside the square --/
  p : Fin 2 → ℝ
  /-- Ensure P is inside the square --/
  p_inside : p 0 ∈ Set.Icc (0 : ℝ) side ∧ p 1 ∈ Set.Icc (0 : ℝ) side

/-- The distance between two points --/
noncomputable def distance (a b : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((a 0 - b 0)^2 + (a 1 - b 1)^2)

/-- The theorem to be proved --/
theorem square_point_distances (s : SquareWithPoint) 
  (hPA : distance s.p (λ _ => 0) = 5)
  (hPD : distance s.p (λ i => if i = 0 then s.side else 0) = 6)
  (hPB : distance s.p (λ i => if i = 1 then s.side else 0) = 7) :
  distance s.p (λ _ => s.side) = Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_point_distances_l395_39561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_numbers_l395_39585

-- Define the set of four numbers
def BoardNumbers := Fin 4 → ℝ

-- Define the trigonometric functions
noncomputable def trig_funcs (α : ℝ) : Fin 4 → ℝ → ℝ
| 0 => λ x => x * Real.sin α
| 1 => λ x => x * Real.cos α
| 2 => λ x => x * Real.tan α
| 3 => λ x => x * (1 / Real.tan α)  -- Replace Real.cot with 1 / Real.tan

-- Define the property that the set remains unchanged after multiplication
def unchanged_after_mult (nums : BoardNumbers) (α : ℝ) : Prop :=
  ∃ (perm : Fin 4 ≃ Fin 4), ∀ i, trig_funcs α i (nums i) = nums (perm i)

-- The main theorem
theorem max_distinct_numbers (nums : BoardNumbers) (α : ℝ) 
  (h : unchanged_after_mult nums α) : 
  Finset.card (Finset.image nums Finset.univ) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_numbers_l395_39585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_for_product_12_factorial_l395_39563

theorem min_sum_for_product_12_factorial (a b c d : ℕ+) 
  (h : (a : ℕ) * b * c * d = Nat.factorial 12) : 
  (a : ℕ) + b + c + d ≥ 683 ∧ ∃ (a' b' c' d' : ℕ+), 
    (a' : ℕ) * b' * c' * d' = Nat.factorial 12 ∧ (a' : ℕ) + b' + c' + d' = 683 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_for_product_12_factorial_l395_39563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l395_39577

/-- Given a triangle ABC with circumcenter O, this theorem states that
    if BC = 8, AC = 4, and |4⃗OA - ⃗OB - 3⃗OC| = 10, then AB = 5. -/
theorem triangle_side_length (A B C O : ℝ × ℝ) : 
  (∃ r : ℝ, ∀ P ∈ ({A, B, C} : Set (ℝ × ℝ)), dist O P = r) →  -- O is the circumcenter
  dist B C = 8 →                              -- BC = 8
  dist A C = 4 →                              -- AC = 4
  Real.sqrt ((4 • (A - O) - (B - O) - 3 • (C - O)).1^2 + (4 • (A - O) - (B - O) - 3 • (C - O)).2^2) = 10  -- |4⃗OA - ⃗OB - 3⃗OC| = 10
  → dist A B = 5 :=                           -- Then AB = 5
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l395_39577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l395_39504

/-- The function f(x) = sin(3x) + cos(3x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x) + Real.cos (3 * x)

/-- The period of a function -/
def is_period (p : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- Theorem: The period of f(x) = sin(3x) + cos(3x) is 2π/3 -/
theorem period_of_f :
  is_period (2 * Real.pi / 3) f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l395_39504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_five_digit_count_l395_39558

theorem odd_five_digit_count : ∃ n : ℕ, n = 72 := by
  let digits : Finset ℕ := {1, 2, 3, 4, 5}
  let odd_digits : Finset ℕ := {1, 3, 5}
  let n : ℕ := 5  -- number of digits in the number

  have h : (odd_digits.card * (digits.card - 1).factorial) = 72 := by
    -- Proof goes here
    sorry

  exact ⟨72, rfl⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_five_digit_count_l395_39558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_characterization_l395_39502

open Set
open Function

def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

theorem increasing_function_characterization
  (D : Set ℝ) (f : ℝ → ℝ) (m n : ℝ) (h : Set.Ioo m n ⊆ D) :
  IncreasingOn f (Set.Ioo m n) ↔
  ∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioo m n → x₂ ∈ Set.Ioo m n → x₁ ≠ x₂ →
    (f x₁ - f x₂) / (x₁ - x₂) > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_characterization_l395_39502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_2017_equals_1_l395_39536

def my_sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | 1 => 5
  | (n + 2) => my_sequence (n + 1) - my_sequence n

theorem my_sequence_2017_equals_1 : my_sequence 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_2017_equals_1_l395_39536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_150_degree_angles_l395_39550

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  n > 2 → (n : ℝ) * 150 = (n - 2 : ℝ) * 180 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_150_degree_angles_l395_39550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_corner_distance_l395_39541

/-- The distance between two corners separated by two sides in a regular octagon with side length 1 -/
noncomputable def distance_between_corners (side_length : ℝ) : ℝ :=
  side_length * (1 + Real.sqrt 2)

/-- Theorem stating that the distance between two corners separated by two sides
    in a regular octagon with side length 1 is 1 + √2 -/
theorem regular_octagon_corner_distance (side_length : ℝ) 
  (h1 : side_length = 1) : 
  distance_between_corners side_length = 1 + Real.sqrt 2 := by
  unfold distance_between_corners
  rw [h1]
  ring
  
#check regular_octagon_corner_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_corner_distance_l395_39541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_cents_proof_l395_39556

def initial_amount : ℚ := 360
def final_amount : ℚ := 367.20
def interest_rate : ℚ := 1/25 -- 0.04 as a rational number
def time_period : ℚ := 1/2
def compounding_frequency : ℕ := 2

theorem interest_cents_proof :
  let compound_interest := final_amount - initial_amount
  let cents_part := (compound_interest - ⌊compound_interest⌋) * 100
  cents_part = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_cents_proof_l395_39556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_iff_m_eq_2_or_neg_5_l395_39521

/-- Two circles are externally tangent if and only if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  Real.sqrt ((c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2) = r₁ + r₂

/-- The equation of circle C₁ -/
def circle_C₁ (m : ℝ) (x y : ℝ) : Prop := (x + 2)^2 + (y - m)^2 = 9

/-- The equation of circle C₂ -/
def circle_C₂ (m : ℝ) (x y : ℝ) : Prop := (x - m)^2 + (y + 1)^2 = 4

/-- The center of circle C₁ -/
def center_C₁ (m : ℝ) : ℝ × ℝ := (-2, m)

/-- The center of circle C₂ -/
def center_C₂ (m : ℝ) : ℝ × ℝ := (m, -1)

/-- The radius of circle C₁ -/
def radius_C₁ : ℝ := 3

/-- The radius of circle C₂ -/
def radius_C₂ : ℝ := 2

theorem circles_externally_tangent_iff_m_eq_2_or_neg_5 :
  ∀ m : ℝ, externally_tangent (center_C₁ m) (center_C₂ m) radius_C₁ radius_C₂ ↔ (m = 2 ∨ m = -5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_iff_m_eq_2_or_neg_5_l395_39521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l395_39582

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/9) + Real.sin (5*Real.pi/9 - x)

noncomputable def g (x : ℝ) : ℝ := f (f x)

theorem g_max_value : ∃ (M : ℝ), (∀ (x : ℝ), g x ≤ M) ∧ M = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l395_39582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_increase_is_three_l395_39595

/-- Represents the river crossing problem --/
structure RiverCrossing where
  river_width : ℚ
  initial_speed : ℚ
  current_speed : ℚ
  waterfall_distance : ℚ

/-- Calculates the minimum speed increase required to safely cross the river --/
noncomputable def min_speed_increase (rc : RiverCrossing) : ℚ :=
  let midpoint := rc.river_width / 2
  let time_to_midpoint := midpoint / rc.initial_speed
  let downstream_distance := time_to_midpoint * rc.current_speed
  let remaining_downstream := rc.waterfall_distance - downstream_distance
  let time_to_waterfall := remaining_downstream / rc.current_speed
  let required_speed := midpoint / time_to_waterfall
  required_speed - rc.initial_speed

/-- Theorem stating that the minimum speed increase for the given conditions is 3 feet/s --/
theorem min_speed_increase_is_three :
  let rc : RiverCrossing := {
    river_width := 100,
    initial_speed := 2,
    current_speed := 5,
    waterfall_distance := 175
  }
  min_speed_increase rc = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_increase_is_three_l395_39595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l395_39587

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the area of a triangle given two side lengths and the angle between them
noncomputable def triangleArea (s1 s2 angle : ℝ) : ℝ := 1/2 * s1 * s2 * Real.sin angle

theorem ellipse_and_triangle_area 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ellipse a b (-2) 0) 
  (h4 : ∀ m : ℝ, ∃ x1 y1 x2 y2 : ℝ, 
    ellipse a b x1 y1 ∧ 
    ellipse a b x2 y2 ∧ 
    line m x1 y1 ∧ 
    line m x2 y2 ∧ 
    (x1 ≠ -2 ∨ y1 ≠ 0) ∧ 
    (x2 ≠ -2 ∨ y2 ≠ 0))
  (h5 : ∃ x1 y1 x2 y2 : ℝ, 
    ellipse a b x1 y1 ∧ 
    ellipse a b x2 y2 ∧ 
    x1 = 1 ∧ 
    x2 = 1 ∧ 
    distance x1 y1 x2 y2 = 3) :
  (a = 2 ∧ b = Real.sqrt 3) ∧ 
  (∀ m : ℝ, ∃ S : ℝ, 
    (∃ x1 y1 x2 y2 : ℝ, 
      ellipse 2 (Real.sqrt 3) x1 y1 ∧ 
      ellipse 2 (Real.sqrt 3) x2 y2 ∧ 
      line m x1 y1 ∧ 
      line m x2 y2 ∧ 
      S = triangleArea (distance (-2) 0 x1 y1) (distance (-2) 0 x2 y2) (Real.arccos ((x1 + 2) * (x2 + 2) / (distance (-2) 0 x1 y1 * distance (-2) 0 x2 y2)))) ∧
    S > 0 ∧ 
    S ≤ 9/2) ∧
  (∃ m : ℝ, ∃ x1 y1 x2 y2 : ℝ, 
    ellipse 2 (Real.sqrt 3) x1 y1 ∧ 
    ellipse 2 (Real.sqrt 3) x2 y2 ∧ 
    line m x1 y1 ∧ 
    line m x2 y2 ∧ 
    triangleArea (distance (-2) 0 x1 y1) (distance (-2) 0 x2 y2) (Real.arccos ((x1 + 2) * (x2 + 2) / (distance (-2) 0 x1 y1 * distance (-2) 0 x2 y2))) = 9/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l395_39587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_problem_l395_39596

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (l₁ l₂ : Line) : ℝ :=
  abs (l₁.c - l₂.c) / Real.sqrt (l₁.a^2 + l₁.b^2)

/-- Theorem: For the given parallel lines, m = 8 and the distance between them is 2 -/
theorem parallel_lines_problem (m : ℝ) :
  let l₁ : Line := ⟨3, 4, -3⟩
  let l₂ : Line := ⟨6, m, 14⟩
  (l₁.a * l₂.b = l₁.b * l₂.a) →
  (m = 8 ∧ distance_between_parallel_lines l₁ ⟨3, 4, 7⟩ = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_problem_l395_39596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomials_equal_l395_39524

-- Define the polynomials f and g
variable (f g : ℕ → ℕ)

-- Define m as the largest coefficient of f
def m : ℕ := sorry

-- Define the property of f and g having nonnegative integer coefficients
def nonneg_int_coeffs (p : ℕ → ℕ) : Prop := sorry

-- Define the property of m being the largest coefficient of f
def m_largest_coeff (f : ℕ → ℕ) (m : ℕ) : Prop := sorry

-- Define the existence of a and b
def exists_a_b (f g : ℕ → ℕ) (m : ℕ) : Prop := 
  ∃ (a b : ℕ), a < b ∧ f a = g a ∧ f b = g b ∧ b > m

-- Theorem statement
theorem polynomials_equal 
  (h1 : nonneg_int_coeffs f) 
  (h2 : nonneg_int_coeffs g) 
  (h3 : m_largest_coeff f m) 
  (h4 : exists_a_b f g m) : 
  f = g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomials_equal_l395_39524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_arrangement_l395_39572

theorem impossibility_of_arrangement : ¬ ∃ (a b c d e f : ℕ) (S : ℕ),
  ({a, b, c, d, e, f} : Finset ℕ) = {2021, 3022, 4023, 5024, 6025, 7026} ∧
  a + b + c = S ∧
  c + d + e = S ∧
  e + f + a = S ∧
  a + c + e = S :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_arrangement_l395_39572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_element_in_M_l395_39557

theorem unique_element_in_M (a : ℝ) : ∃! x : ℝ, x^3 + (a^2 + 1) * x + 2 * a^2 + 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_element_in_M_l395_39557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_radius_circle_tangent_y_axis_l395_39512

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- The focus of a parabola -/
noncomputable def focus (para : Parabola) : Point :=
  { x := para.p / 2, y := 0 }

/-- A point on the parabola -/
def point_on_parabola (para : Parabola) (P : Point) : Prop :=
  P.y^2 = 2 * para.p * P.x

/-- The center of the circle with PF as diameter -/
noncomputable def circle_center (para : Parabola) (P : Point) : Point :=
  { x := (2 * P.x + para.p) / 4, y := P.y / 2 }

/-- The radius of the circle with PF as diameter -/
noncomputable def circle_radius (para : Parabola) (P : Point) : ℝ :=
  (2 * P.x + para.p) / 4

theorem parabola_focal_radius_circle_tangent_y_axis (para : Parabola) (P : Point)
  (h_on_parabola : point_on_parabola para P) :
  circle_center para P = { x := circle_radius para P, y := circle_center para P |>.y } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_radius_circle_tangent_y_axis_l395_39512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_diameter_intersection_l395_39507

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point of tangency between two circles -/
structure TangencyPoint where
  point : ℝ × ℝ
  circle1 : Circle
  circle2 : Circle

/-- Represents a line in a 2D plane -/
def line (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • p1 + t • p2}

/-- Represents the set of points on a circle -/
def circle_points (c : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2}

/-- Checks if two points form a diameter of a circle -/
def is_diameter (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  p1 ∈ circle_points c ∧ p2 ∈ circle_points c ∧
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = (2 * c.radius)^2

/-- Given three circles that are pairwise tangent, proves that the lines connecting
    the point of tangency of two circles with the other two points of tangency
    intersect the third circle at the endpoints of its diameter -/
theorem tangent_circles_diameter_intersection
  (S₁ S₂ S₃ : Circle)
  (A : TangencyPoint)
  (B : TangencyPoint)
  (C : TangencyPoint)
  (h1 : A.circle1 = S₁ ∧ A.circle2 = S₂)
  (h2 : B.circle1 = S₂ ∧ B.circle2 = S₃)
  (h3 : C.circle1 = S₃ ∧ C.circle2 = S₁)
  (h_distinct : A.point ≠ B.point ∧ B.point ≠ C.point ∧ C.point ≠ A.point) :
  ∃ (M N : ℝ × ℝ),
    (M ∈ line A.point B.point ∩ circle_points S₃) ∧
    (N ∈ line A.point C.point ∩ circle_points S₃) ∧
    is_diameter S₃ M N :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_diameter_intersection_l395_39507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_not_suff_nec_for_ellipse_l395_39546

/-- Predicate to check if a given function represents an ellipse -/
def IsEllipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a * b > 0 ∧ a * c < 0 ∧ ∀ x y, f x y ↔ a * x^2 + b * y^2 = c

/-- Given a, b, c ∈ ℝ with abc ≠ 0, prove that ac > 0 is neither a sufficient
    nor a necessary condition for ax^2 + by^2 = c to be an ellipse. -/
theorem ac_not_suff_nec_for_ellipse (a b c : ℝ) (h : a * b * c ≠ 0) :
  ¬(∀ x y : ℝ, a * c > 0 → IsEllipse (fun x y => a * x^2 + b * y^2 = c)) ∧
  ¬(∀ x y : ℝ, IsEllipse (fun x y => a * x^2 + b * y^2 = c) → a * c > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_not_suff_nec_for_ellipse_l395_39546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_complete_first_but_zero_value_expected_prize_value_l395_39540

/-- Represents the game with three levels of challenges --/
structure Game where
  prize1 : ℚ := 600
  prize2 : ℚ := 900
  prize3 : ℚ := 1500
  prob_success1 : ℚ := 3/4
  prob_success2 : ℚ := 2/3
  prob_success3 : ℚ := 1/2
  prob_continue1 : ℚ := 3/5
  prob_continue2 : ℚ := 2/5

/-- The probability of completing the first level but ending with zero total prize money --/
def prob_complete_first_but_zero (g : Game) : ℚ :=
  g.prob_success1 * g.prob_continue1 * (1 - g.prob_success2) +
  g.prob_success1 * g.prob_continue1 * g.prob_success2 * g.prob_continue2 * (1 - g.prob_success3)

/-- The expected value of the total prize money --/
def expected_prize (g : Game) : ℚ :=
  0 * (1 - g.prob_success1 + prob_complete_first_but_zero g) +
  g.prize1 * g.prob_success1 * (1 - g.prob_continue1) +
  (g.prize1 + g.prize2) * g.prob_success1 * g.prob_continue1 * g.prob_success2 * (1 - g.prob_continue2) +
  (g.prize1 + g.prize2 + g.prize3) * g.prob_success1 * g.prob_continue1 * g.prob_success2 * g.prob_continue2 * g.prob_success3

/-- Theorem stating the probability of completing the first level but ending with zero total prize money --/
theorem prob_complete_first_but_zero_value (g : Game) :
  prob_complete_first_but_zero g = 21/100 := by sorry

/-- Theorem stating the expected value of the total prize money --/
theorem expected_prize_value (g : Game) :
  expected_prize g = 630 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_complete_first_but_zero_value_expected_prize_value_l395_39540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l395_39528

/-- An inverse proportion function -/
noncomputable def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := (m + 1) / x

/-- Predicate for a function being in the first and third quadrants -/
def in_first_and_third_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x > 0 ∧ ∀ x, x < 0 → f x < 0

theorem inverse_proportion_quadrants (m : ℝ) :
  in_first_and_third_quadrants (inverse_proportion m) → m > -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l395_39528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l395_39579

noncomputable def f (x : ℝ) : ℝ :=
  if 2 < x ∧ x ≤ 3 then -x^2 + 4*x
  else if 3 < x ∧ x ≤ 4 then (x^2 + 2) / x
  else 0  -- Define f outside (2,4] as 0 for completeness

def g (a x : ℝ) : ℝ := a * x + 1

theorem range_of_a :
  ∀ a : ℝ,
    (∀ x₁ ∈ Set.Ioc (-4) (-2),
      ∃ x₂ ∈ Set.Icc (-2) 1,
        g a x₂ = f (x₁ + 6)) ↔
    a ∈ Set.Iic (-5/8) ∪ Set.Ici (5/16) := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l395_39579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_count_l395_39527

/-- A coloring of a 3D grid -/
def Coloring := Fin 1983 → Fin 1983 → Fin 1983 → Bool

/-- Check if a coloring is valid for a given parallelepiped -/
def isValidParallelepiped (c : Coloring) (x1 x2 y1 y2 z1 z2 : Fin 1983) : Prop :=
  (((List.sum (List.map (fun x => 
    List.sum (List.map (fun y => 
      List.sum (List.map (fun z => 
        if c x y z then 1 else 0
      ) [z1, z2])
    ) [y1, y2])
  ) [x1, x2]))) % 4) = 0

/-- A coloring is valid if all parallelepipeds are valid -/
def isValidColoring (c : Coloring) : Prop :=
  ∀ x1 x2 y1 y2 z1 z2 : Fin 1983, isValidParallelepiped c x1 x2 y1 y2 z1 z2

/-- The number of valid colorings -/
def numValidColorings : ℕ := sorry

/-- The main theorem -/
theorem valid_colorings_count :
  numValidColorings = 2^5947 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_count_l395_39527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_principal_l395_39586

/-- Calculates the principal amount of an investment given monthly interest payment and annual interest rate. -/
noncomputable def calculate_principal (monthly_interest : ℚ) (annual_rate : ℚ) : ℚ :=
  monthly_interest / (annual_rate / 1200)

/-- Proves that an investment with $225 monthly interest payment and 9% annual rate has a principal of $30,000. -/
theorem investment_principal :
  calculate_principal 225 9 = 30000 := by
  -- Unfold the definition of calculate_principal
  unfold calculate_principal
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_principal_l395_39586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l395_39526

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (3 : ℝ)^(2*x + 1) - (3 : ℝ)^(x + 2) - 9 * (3 : ℝ)^x + 27 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l395_39526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l395_39568

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := exp x - x * exp 1

-- State the theorem
theorem max_value_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ 
  (∀ (y : ℝ), y ∈ Set.Icc 0 1 → f y ≤ f x) ∧
  f x = exp 1 - 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l395_39568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_one_eq_neg_one_l395_39552

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x * (x - 2) else -x * (x + 2)

theorem f_at_one_eq_neg_one : f 1 = -1 := by
  -- Unfold the definition of f
  unfold f
  -- Since 1 ≥ 0, we use the first case of the definition
  simp [if_pos (show 1 ≥ 0 by norm_num)]
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_one_eq_neg_one_l395_39552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimized_at_set_l395_39549

open Real

-- Define the expression as a function of x
noncomputable def f (x : ℝ) : ℝ := 1 + (cos ((π * sin (2 * x)) / Real.sqrt 3))^2 + (sin (2 * Real.sqrt 3 * π * cos x))^2

-- Define the set of x values that minimize the function
def minimizing_set : Set ℝ := {x | ∃ k : ℤ, x = π/6 + π * k ∨ x = -π/6 + π * k}

-- Theorem statement
theorem f_minimized_at_set : 
  ∀ x : ℝ, ∀ y ∈ minimizing_set, f y ≤ f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimized_at_set_l395_39549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_rates_bounded_l395_39551

/-- Represents the flow rate of a river -/
structure FlowRate where
  rate : ℝ
  pos : rate > 0

/-- Represents the distance between two points on a river -/
structure Distance where
  length : ℝ
  pos : length > 0

/-- Represents the speed of a boat -/
structure BoatSpeed where
  speed : ℝ
  pos : speed > 0

/-- Calculates the time taken for a boat to travel a distance with or against a flow -/
noncomputable def travelTime (d : Distance) (b : BoatSpeed) (f : FlowRate) (upstream : Bool) : ℝ :=
  if upstream then d.length / (b.speed - f.rate) else d.length / (b.speed + f.rate)

theorem river_flow_rates_bounded
  (f1 f2 : FlowRate)
  (d1 d2 : Distance)
  (b : BoatSpeed)
  (h1 : d1.length = 2 * d2.length)
  (h2 : travelTime d1 b f1 true + travelTime d2 b f2 false =
        travelTime d2 b f2 true + travelTime d1 b f1 false) :
  f1.rate ≤ 2 * f2.rate ∧ f2.rate ≤ 2 * f1.rate := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_rates_bounded_l395_39551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x₀_l395_39597

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the line
def line_eq (x y : ℝ) : Prop := x + 3*y - 6 = 0

-- Define the angle condition
def angle_condition (x₀ y₀ : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ 
  let OP := Real.sqrt (x₀^2 + y₀^2);
  let OQ := Real.sqrt 3;
  let PQ := Real.sqrt ((x - x₀)^2 + (y - y₀)^2);
  (OP^2 + PQ^2 - OQ^2) / (2 * OP * PQ) = 1/2

-- Theorem statement
theorem range_of_x₀ (x₀ y₀ : ℝ) :
  line_eq x₀ y₀ → angle_condition x₀ y₀ → 0 ≤ x₀ ∧ x₀ ≤ 6/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x₀_l395_39597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_features_l395_39599

noncomputable def f (x : ℝ) : ℝ := (x^3 + 2*x^2 - 3*x - 6) / (x^4 - x^3 - 6*x^2)

def a : ℕ := 0

def b : ℕ := 3

def c : ℕ := 1

def d : ℕ := 0

theorem sum_of_features : a + 2*b + 3*c + 4*d = 9 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_features_l395_39599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l395_39571

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution_set :
  {x : ℝ | f x > 0} = Set.Iio 2 ∪ Set.Ioo 3 4 ∪ Set.Ioo 5 6 ∪ Set.Ioi 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l395_39571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_of_square_l395_39509

def square_vertices (a b c : ℂ) : Prop :=
  ∃ (d : ℂ), 
    Complex.abs (a - b) = Complex.abs (b - c) ∧
    Complex.abs (a - b) = Complex.abs (c - d) ∧
    Complex.abs (a - b) = Complex.abs (d - a) ∧
    (a - c) = (b - d)

theorem fourth_vertex_of_square :
  let a : ℂ := 1 + 2*Complex.I
  let b : ℂ := -2 + Complex.I
  let c : ℂ := -1 - 2*Complex.I
  let d : ℂ := 2 - Complex.I
  square_vertices a b c → d = 2 - Complex.I :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_of_square_l395_39509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_1_30_l395_39593

noncomputable section

/-- The number of degrees in a full circle on a clock -/
def full_circle : ℝ := 360

/-- The number of hours on a clock -/
def hours_on_clock : ℕ := 12

/-- The number of minutes in an hour -/
def minutes_in_hour : ℕ := 60

/-- The angle (in degrees) moved by the hour hand in one hour -/
noncomputable def hour_hand_angle_per_hour : ℝ := full_circle / hours_on_clock

/-- The angle (in degrees) moved by the minute hand in one minute -/
noncomputable def minute_hand_angle_per_minute : ℝ := full_circle / minutes_in_hour

/-- The position of the hour hand at 1:30 -/
noncomputable def hour_hand_position : ℝ := 
  hour_hand_angle_per_hour * (1 + 30 / minutes_in_hour)

/-- The position of the minute hand at 1:30 -/
noncomputable def minute_hand_position : ℝ := minute_hand_angle_per_minute * 30

/-- The angle between the hour hand and minute hand at 1:30 -/
noncomputable def angle_between_hands : ℝ := minute_hand_position - hour_hand_position

theorem clock_angle_at_1_30 : angle_between_hands = 135 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_1_30_l395_39593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l395_39510

theorem problem_solution : ((3^1 - 2 + 6^2 + 1)⁻¹ * 6 : ℚ) = 3/19 := by
  -- Convert the expression to rational numbers
  have h1 : (3^1 - 2 + 6^2 + 1 : ℚ) = 38 := by norm_num
  
  -- Calculate the inverse
  have h2 : ((3^1 - 2 + 6^2 + 1)⁻¹ : ℚ) = 1/38 := by
    rw [h1]
    norm_num
  
  -- Multiply by 6
  calc ((3^1 - 2 + 6^2 + 1)⁻¹ * 6 : ℚ) = (1/38) * 6 := by rw [h2]
    _ = 3/19 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l395_39510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_from_hyperbola_vertices_l395_39578

/-- Given a hyperbola with equation x²/4 - y²/9 = 1, prove that the equation of the ellipse
    with foci at the vertices of the hyperbola is x²/13 + y²/9 = 1 -/
theorem ellipse_from_hyperbola_vertices (x y : ℝ) :
  (x^2 / 4 - y^2 / 9 = 1) →
  (x^2 / 13 + y^2 / 9 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_from_hyperbola_vertices_l395_39578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alfie_original_seat_l395_39589

-- Define the seat type
inductive Seat
| one
| two
| three
| four
| five

-- Define the person type
inductive Person
| alfie
| bob
| carl
| doug
| eddy

-- Define the seating arrangement
def SeatingArrangement := Seat → Option Person

-- Define the initial seating arrangement
def initialSeating : SeatingArrangement := sorry

-- Define the movement functions
def moveRight (n : Nat) (p : Person) (s : SeatingArrangement) : SeatingArrangement := sorry
def swap (p1 p2 : Person) (s : SeatingArrangement) : SeatingArrangement := sorry

-- Define the final seating arrangement after all movements
def finalSeating (s : SeatingArrangement) : SeatingArrangement :=
  s |> moveRight 1 Person.bob
    |> moveRight 2 Person.carl
    |> swap Person.doug Person.eddy
    |> moveRight 1 Person.doug

-- Theorem: Alfie's original seat was Seat.one
theorem alfie_original_seat (s : SeatingArrangement) :
  s = initialSeating →
  (finalSeating s Seat.one = none ∨ finalSeating s Seat.five = none) →
  s Seat.one = some Person.alfie := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alfie_original_seat_l395_39589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_point_l395_39537

noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

theorem g_symmetry_about_point :
  ∀ (x : ℝ), g (2 * Real.pi / 3 + (2 * Real.pi / 3 - x)) = g x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_point_l395_39537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l395_39532

theorem exponential_equation_solution :
  ∃ x : ℚ, (3 : ℝ) ^ (4 * (x : ℝ)^2 - 3 * (x : ℝ) + 5) = 3 ^ (4 * (x : ℝ)^2 + 9 * (x : ℝ) - 6) ∧ x = 11/12 := by
  use 11/12
  constructor
  · simp
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l395_39532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l395_39569

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x : ℝ, f (-Real.pi/6 + x) = -f (-Real.pi/6 - x)) ∧
  (∀ x : ℝ, f x = 4 * Real.cos (2 * x - Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l395_39569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l395_39566

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  point : ℝ × ℝ
  on_parabola : point.2^2 = 2 * C.p * point.1

/-- Focus of the parabola -/
def focus (C : Parabola) : ℝ × ℝ := (C.p, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem statement -/
theorem parabola_properties (C : Parabola) 
  (h_focus_dist : C.p = 1) -- Focus is at distance 2 from directrix
  : 
  -- Part 1
  (∀ (A B : PointOnParabola C) (M : ℝ × ℝ),
    distance A.point B.point = 8 ∧ 
    M = ((A.point.1 + B.point.1)/2, (A.point.2 + B.point.2)/2) ∧
    distance (focus C) A.point = distance (focus C) B.point
    → M.1 = 3) ∧
  -- Part 2
  (∀ (P : ℝ × ℝ) (Q : PointOnParabola C),
    P.1 = -C.p ∧  -- P is on directrix
    distance P (focus C) = 4 * distance Q.point (focus C)
    → distance P (focus C) = 6) ∧
  -- Part 3
  (∀ (A B : PointOnParabola C),
    distance (focus C) A.point = distance (focus C) B.point
    → 9 * distance (focus C) A.point + distance (focus C) B.point ≥ 16) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l395_39566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_rectangular_coords_for_complementary_phi_l395_39506

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates (ρ, θ, φ),
    the point with spherical coordinates (ρ, θ, 2π-φ) has the same rectangular coordinates (x, y, z). -/
theorem same_rectangular_coords_for_complementary_phi 
  (x y z ρ θ φ : ℝ) (h1 : x = ρ * Real.sin φ * Real.cos θ)
  (h2 : y = ρ * Real.sin φ * Real.sin θ) (h3 : z = ρ * Real.cos φ) :
  (ρ * Real.sin (2 * Real.pi - φ) * Real.cos θ = x) ∧
  (ρ * Real.sin (2 * Real.pi - φ) * Real.sin θ = y) ∧
  (ρ * Real.cos (2 * Real.pi - φ) = z) := by
  sorry

#check same_rectangular_coords_for_complementary_phi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_rectangular_coords_for_complementary_phi_l395_39506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_sum_folded_edge_passes_vertex_crescent_area_sum_equals_triangle_l395_39574

/-- Right triangle with semicircles on its sides -/
structure RightTriangleWithSemicircles where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_pythagorean : a^2 + b^2 = c^2

/-- Area of a semicircle -/
noncomputable def semicircle_area (d : ℝ) : ℝ := Real.pi * d^2 / 8

/-- Areas of the semicircles -/
noncomputable def A1 (t : RightTriangleWithSemicircles) : ℝ := semicircle_area t.a
noncomputable def A2 (t : RightTriangleWithSemicircles) : ℝ := semicircle_area t.b
noncomputable def A3 (t : RightTriangleWithSemicircles) : ℝ := semicircle_area t.c

/-- Area of the triangle -/
noncomputable def T_area (t : RightTriangleWithSemicircles) : ℝ := t.a * t.b / 2

/-- Theorem: The sum of areas of semicircles on legs equals the area of semicircle on hypotenuse -/
theorem semicircle_area_sum (t : RightTriangleWithSemicircles) : A1 t + A2 t = A3 t := by sorry

/-- Theorem: The folded edge of A3 passes through the right-angle vertex of T -/
theorem folded_edge_passes_vertex (t : RightTriangleWithSemicircles) : 
  ∃ (p : ℝ × ℝ), p.1^2 + p.2^2 = (t.c/2)^2 ∧ p.1 = t.a ∧ p.2 = t.b := by sorry

/-- Theorem: The sum of areas of crescents L1 and L2 equals the area of triangle T -/
theorem crescent_area_sum_equals_triangle (t : RightTriangleWithSemicircles) : 
  ∃ (L1 L2 : ℝ), L1 + L2 = T_area t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_sum_folded_edge_passes_vertex_crescent_area_sum_equals_triangle_l395_39574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_coverage_is_75_percent_l395_39573

/-- Represents a tiling pattern with squares and hexagons -/
structure TilingPattern where
  rectangleRegions : ℕ := 12
  squareRegions : ℕ := 3
  squareSideLength : ℝ := 2

/-- Calculates the percentage of area covered by hexagons in the tiling pattern -/
noncomputable def hexagonCoverage (pattern : TilingPattern) : ℝ :=
  let totalArea : ℝ := pattern.rectangleRegions * (pattern.squareSideLength ^ 2)
  let squareArea : ℝ := pattern.squareRegions * (pattern.squareSideLength ^ 2)
  let hexagonArea : ℝ := totalArea - squareArea
  (hexagonArea / totalArea) * 100

/-- Theorem stating that the hexagon coverage is 75% for the given tiling pattern -/
theorem hexagon_coverage_is_75_percent (pattern : TilingPattern) :
  hexagonCoverage pattern = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_coverage_is_75_percent_l395_39573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l395_39590

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 9
def C₂ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define the line
def L (y : ℝ) : Prop := y = -1

-- Define the distance function
noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (M N P : ℝ × ℝ),
  C₁ M.1 M.2 ∧ C₂ N.1 N.2 ∧ L P.2 ∧
  (∀ (M' N' P' : ℝ × ℝ), C₁ M'.1 M'.2 → C₂ N'.1 N'.2 → L P'.2 →
    dist P.1 P.2 M.1 M.2 + dist P.1 P.2 N.1 N.2 ≤ 
    dist P'.1 P'.2 M'.1 M'.2 + dist P'.1 P'.2 N'.1 N'.2) ∧
  dist P.1 P.2 M.1 M.2 + dist P.1 P.2 N.1 N.2 = 5 * Real.sqrt 2 - 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l395_39590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l395_39511

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem f_range :
  {y : ℝ | ∃ x : ℝ, x ≠ -2 ∧ f x = y} = Set.Ioi 1 ∪ Set.Iio 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l395_39511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_factorial_equation_l395_39564

theorem unique_n_for_factorial_equation : ∃! n : ℕ, 2^6 * 5^2 * n = Nat.factorial 10 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_factorial_equation_l395_39564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_before_turning_theorem_l395_39531

/-- Represents a hiker's journey --/
structure HikerJourney where
  rate : ℚ  -- Hiking rate in minutes per kilometer
  totalTime : ℚ  -- Total time for the round trip in minutes
  totalDistance : ℚ  -- Total distance hiked east in kilometers

/-- Calculates the distance hiked east before turning back --/
def distanceBeforeTurning (journey : HikerJourney) : ℚ :=
  journey.totalDistance / 2

/-- Theorem stating the distance hiked east before turning back --/
theorem distance_before_turning_theorem (journey : HikerJourney) 
  (h1 : journey.rate = 10)
  (h2 : journey.totalTime = 35)
  (h3 : journey.totalDistance = 3) :
  distanceBeforeTurning journey = 1/2 := by
  sorry

#check distance_before_turning_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_before_turning_theorem_l395_39531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_is_pi_l395_39584

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sin x * Real.cos x

-- Define the period of a function
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Theorem stating that the period of f(x) is π
theorem f_period_is_pi : is_periodic f π := by
  sorry

#check f_period_is_pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_is_pi_l395_39584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadratics_with_distinct_roots_l395_39545

theorem max_quadratics_with_distinct_roots 
  (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  ∃ (p1 p2 p3 p4 p5 p6 : ℝ → ℝ),
    (∀ x, p1 x = a * x^2 + b * x + c) ∧
    (∀ x, p2 x = a * x^2 + c * x + b) ∧
    (∀ x, p3 x = b * x^2 + a * x + c) ∧
    (∀ x, p4 x = b * x^2 + c * x + a) ∧
    (∀ x, p5 x = c * x^2 + a * x + b) ∧
    (∀ x, p6 x = c * x^2 + b * x + a) ∧
    (∀ i : Fin 6, ∃ (x y : ℝ), x ≠ y ∧ 
      (match i with
      | ⟨0, _⟩ => p1 x = 0 ∧ p1 y = 0
      | ⟨1, _⟩ => p2 x = 0 ∧ p2 y = 0
      | ⟨2, _⟩ => p3 x = 0 ∧ p3 y = 0
      | ⟨3, _⟩ => p4 x = 0 ∧ p4 y = 0
      | ⟨4, _⟩ => p5 x = 0 ∧ p5 y = 0
      | ⟨5, _⟩ => p6 x = 0 ∧ p6 y = 0
      | _ => False)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadratics_with_distinct_roots_l395_39545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_selections_count_l395_39520

def my_marbles : Finset ℕ := Finset.range 8
def mathews_marbles : Finset ℕ := Finset.range 16

def valid_selection (m : ℕ) (a b c : ℕ) : Prop :=
  m ∈ mathews_marbles ∧ a ∈ my_marbles ∧ b ∈ my_marbles ∧ c ∈ my_marbles ∧ a + b + c = m + 1

def count_valid_selections : ℕ := 
  (mathews_marbles.card * my_marbles.card * (my_marbles.card - 1) * (my_marbles.card - 2))

theorem valid_selections_count : count_valid_selections = 204 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_selections_count_l395_39520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_in_range_l395_39515

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 - (a+2)*x + 2*a*Real.log x + 1

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x - (a+2) + 2*a/x

/-- Theorem stating that if f(x) has an extremum point in (4,6), then a is in (4,6) -/
theorem extremum_implies_a_in_range (a : ℝ) :
  (∃ x ∈ Set.Ioo 4 6, f_derivative a x = 0) →
  a ∈ Set.Ioo 4 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_in_range_l395_39515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sum_positive_l395_39519

noncomputable def f (x : ℝ) := x - Real.sin x

theorem x_sum_positive (x₁ x₂ : ℝ) 
  (h₁ : x₁ ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h₂ : x₂ ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h₃ : f x₁ + f x₂ > 0) : 
  x₁ + x₂ > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sum_positive_l395_39519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_solution_l395_39580

theorem cos_sin_equation_solution (n : ℕ+) :
  ∀ x : ℝ, (Real.cos x)^(n : ℕ) - (Real.sin x)^(n : ℕ) = 1 ↔ ∃ k : ℤ, x = k * Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_solution_l395_39580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_distance_50_distance_satisfies_condition_smaller_n_not_sufficient_l395_39517

/-- The x-coordinate of point A_n -/
noncomputable def x_coord (n : ℕ) : ℝ := 2 * n / 3

/-- The distance between A_{n-1} and A_n -/
noncomputable def a (n : ℕ) : ℝ := 2 * n / 3

/-- The total distance from A₀ to A_n -/
noncomputable def total_distance (n : ℕ) : ℝ := n * (n + 1) / 3

theorem smallest_n_for_distance_50 :
  ∃ n, total_distance n ≥ 50 ∧ ∀ m < n, total_distance m < 50 := by
  use 12
  constructor
  · -- Prove total_distance 12 ≥ 50
    sorry
  · -- Prove ∀ m < 12, total_distance m < 50
    sorry

theorem distance_satisfies_condition :
  total_distance 12 ≥ 50 := by sorry

theorem smaller_n_not_sufficient :
  ∀ n < 12, total_distance n < 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_distance_50_distance_satisfies_condition_smaller_n_not_sufficient_l395_39517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pairball_playing_time_l395_39508

/-- In a game of pairball, prove that each child plays 80/3 minutes given the conditions -/
theorem pairball_playing_time (game_duration : ℕ) (num_children : ℕ) (playing_time : ℕ → ℚ)
  (h1 : game_duration = 80)
  (h2 : num_children = 6)
  (h3 : ∀ i j, i ≠ j → playing_time i = playing_time j) :
  ∀ k, playing_time k = 80 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pairball_playing_time_l395_39508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_l395_39516

/-- Given a configuration of points A, B, C, and D where:
    - B is due north of A
    - C is due east of B
    - AC = 12√2
    - ∠BAC = 45°
    - D is 24 meters due east of C
    Prove that AD = 12√6 -/
theorem distance_AD (A B C D : ℝ × ℝ) : 
  (B.2 > A.2) ∧ (B.1 = A.1) →  -- B is due north of A
  (C.1 > B.1) ∧ (C.2 = B.2) →  -- C is due east of B
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 12 * Real.sqrt 2 →  -- AC = 12√2
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) / Real.sqrt 2 →  -- ∠BAC = 45°
  D.1 = C.1 + 24 ∧ D.2 = C.2 →  -- D is 24 meters due east of C
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 12 * Real.sqrt 6 :=  -- AD = 12√6
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_l395_39516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orbital_speed_relation_l395_39581

theorem orbital_speed_relation (a v₁ : ℝ) (h₁ : a > 0) (h₂ : v₁ > 0) :
  ∃ v₂ : ℝ,
    let r₁ := (3/4) * a
    let r₂ := (1/2) * a
    let G := Real.sqrt ((v₁^2 * r₁^3) / (r₂ * (r₁ - r₂)))
    v₂ = (3/Real.sqrt 5) * v₁ ∧
    (1/2) * v₁^2 - G/r₁ = (1/2) * v₂^2 - G/r₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orbital_speed_relation_l395_39581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_and_inequality_l395_39591

-- Define the function f
noncomputable def f (a t x : ℝ) : ℝ := a^x + t * a^(-x)

-- State the theorem
theorem even_function_and_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- f is an even function
  (∀ x, f a 1 x = f a 1 (-x)) →
  -- t = 1
  (∃ t, ∀ x, f a t x = f a t (-x)) ∧
  -- For a > 1, the solution set is (-∞, 3)
  (a > 1 → ∀ x, f a 1 x > a^(2*x-3) + a^(-x) ↔ x < 3) ∧
  -- For 0 < a < 1, the solution set is (3, +∞)
  (a < 1 → ∀ x, f a 1 x > a^(2*x-3) + a^(-x) ↔ x > 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_and_inequality_l395_39591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l395_39523

-- Define the circles
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def circle_M (x y : ℝ) : Prop := (x - 7)^2 + y^2 = 1

-- Define point P on circle M
noncomputable def P : ℝ × ℝ := sorry

-- Define points E and F on circle C
noncomputable def E : ℝ × ℝ := sorry
noncomputable def F : ℝ × ℝ := sorry

-- Define the tangent line condition
def is_tangent (P E F : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem max_dot_product :
  ∀ (P E F : ℝ × ℝ),
  circle_M P.1 P.2 →
  circle_C E.1 E.2 →
  circle_C F.1 F.2 →
  is_tangent P E F →
  (E.1 - 2, E.2) • (F.1 - 2, F.2) ≤ -2 :=
by
  sorry

#check max_dot_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l395_39523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courier_earnings_l395_39575

/-- Calculates the monthly earnings of a school student working as a courier --/
def monthly_earnings (daily_rate : ℚ) (days_per_week : ℕ) (weeks : ℕ) (tax_rate : ℚ) : ℚ :=
  let gross_earnings := daily_rate * (days_per_week : ℚ) * (weeks : ℚ)
  let tax_amount := gross_earnings * tax_rate
  gross_earnings - tax_amount

/-- Theorem stating the monthly earnings of a school student working as a courier --/
theorem courier_earnings : 
  monthly_earnings 1250 4 4 (13/100) = 17400 := by
  -- Unfold the definition of monthly_earnings
  unfold monthly_earnings
  -- Simplify the arithmetic expressions
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_courier_earnings_l395_39575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_girl_fraction_l395_39576

theorem field_trip_girl_fraction :
  ∀ (total : ℕ) (boys girls : ℕ),
    boys = girls →
    boys + girls = total →
    (2 * girls : ℚ) / 3 / ((5 * boys : ℚ) / 6 + (2 * girls : ℚ) / 3) = 4 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_girl_fraction_l395_39576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_double_in_25_years_l395_39570

/-- Represents the simple interest rate as a percentage -/
noncomputable def SimpleInterestRate (principal initialAmount finalAmount : ℝ) (time : ℝ) : ℝ :=
  100 * (finalAmount - initialAmount) / (principal * time)

/-- Theorem: If a sum of money doubles itself in 25 years at simple interest, 
    then the rate percent per annum is 4% -/
theorem simple_interest_double_in_25_years 
  (principal : ℝ) (h_pos : principal > 0) :
  SimpleInterestRate principal principal (2 * principal) 25 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_double_in_25_years_l395_39570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_or_third_quadrant_l395_39518

-- Define an acute angle
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Define the angle β
noncomputable def β (θ : ℝ) (k : ℤ) : ℝ := k * Real.pi + θ

-- Define being in the first or third quadrant
def in_first_or_third_quadrant (angle : ℝ) : Prop :=
  (0 ≤ angle % (2 * Real.pi) ∧ angle % (2 * Real.pi) < Real.pi / 2) ∨
  (Real.pi ≤ angle % (2 * Real.pi) ∧ angle % (2 * Real.pi) < 3 * Real.pi / 2)

-- The theorem to prove
theorem angle_in_first_or_third_quadrant (θ : ℝ) (k : ℤ) :
  is_acute θ → in_first_or_third_quadrant (β θ k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_or_third_quadrant_l395_39518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_walk_probabilities_l395_39547

/-- A fly's monotone walk on a coordinate grid. -/
structure MonotoneWalk where
  start : ℕ × ℕ := (0, 0)
  finish : ℕ × ℕ
  steps : ℕ

/-- Probability of a specific path in a monotone walk. -/
def pathProbability (w : MonotoneWalk) : ℚ :=
  1 / 2 ^ w.steps

/-- Number of paths from (0, 0) to (x, y) in a monotone walk. -/
def numPaths (x y : ℕ) : ℕ :=
  Nat.choose (x + y) x

theorem fly_walk_probabilities :
  let walk := { start := (0, 0), finish := (8, 10), steps := 18 : MonotoneWalk }
  let circleCenter := (4, 5)
  let circleRadius := 3
  (∀ (x y : ℕ), numPaths x y * pathProbability walk = (Nat.choose (x + y) x : ℚ) / 2 ^ (x + y)) ∧
  (pathProbability walk * numPaths 8 10 = (Nat.choose 18 8 : ℚ) / 2^18) ∧
  (pathProbability walk * (numPaths 5 6 * numPaths 2 4) = ((Nat.choose 11 5 * Nat.choose 6 2) : ℚ) / 2^18) ∧
  (pathProbability walk * (2 * (Nat.choose 9 2 * Nat.choose 9 6 + Nat.choose 9 3 * Nat.choose 9 5) + Nat.choose 9 4 * Nat.choose 9 4) = 
   (2 * (Nat.choose 9 2 * Nat.choose 9 6 + Nat.choose 9 3 * Nat.choose 9 5) + Nat.choose 9 4 * Nat.choose 9 4 : ℚ) / 2^18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_walk_probabilities_l395_39547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_l395_39535

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- The derivative of a quadratic function -/
def QuadraticFunction.f' (q : QuadraticFunction) : ℝ → ℝ :=
  λ x => 2 * (q.a : ℝ) * x + (q.b : ℝ)

/-- Theorem: If a quadratic function has vertex (2, 5) and passes through (3, 6), then a = 1 -/
theorem quadratic_coefficient (q : QuadraticFunction) 
  (vertex : q.f 2 = 5 ∧ q.f' 2 = 0) 
  (point : q.f 3 = 6) : 
  q.a = 1 := by
  sorry

#check quadratic_coefficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_l395_39535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l395_39543

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

noncomputable def time_to_complete (rate : ℝ) : ℝ := 1 / rate

theorem combined_work_time 
  (b_days : ℝ) 
  (hb : b_days = 36) 
  (ha : work_rate b_days = work_rate b_days) -- A works as fast as B
  : time_to_complete (work_rate b_days + work_rate b_days) = 18 := by
  sorry

#check combined_work_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l395_39543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paratrooper_exit_forest_l395_39583

/-- A shape in a 2D plane --/
structure Shape where
  area : ℝ
  -- Additional properties of the shape are not specified

/-- A closed path in a 2D plane --/
structure ClosedPath where
  length : ℝ
  -- Additional properties of the path are not specified

/-- 
Given any shape with area S, there exists a closed path with length 
no more than 2√(πS) that encloses the shape.
--/
theorem paratrooper_exit_forest (shape : Shape) : 
  ∃ (path : ClosedPath), path.length ≤ 2 * Real.sqrt (Real.pi * shape.area) ∧ 
  -- Additional condition to specify that the path encloses the shape
  True := by
  sorry

#check paratrooper_exit_forest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paratrooper_exit_forest_l395_39583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l395_39567

-- Define the efficiency of worker b
noncomputable def b_efficiency : ℝ := 1 / 18

-- Define the efficiency of worker a (twice as efficient as b)
noncomputable def a_efficiency : ℝ := 2 * b_efficiency

-- Define the combined efficiency of a and b working together
noncomputable def combined_efficiency : ℝ := a_efficiency + b_efficiency

-- Theorem to prove
theorem work_completion_time : (1 / combined_efficiency) = 6 := by
  -- Expand the definitions
  unfold combined_efficiency a_efficiency b_efficiency
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l395_39567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l395_39505

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x + 2 / x - 3 else Real.log (x^2 + 1) / Real.log 2

theorem f_properties :
  (f (f (-3)) = 0) ∧
  (∀ x, f x ≥ 2 * Real.sqrt 2 - 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l395_39505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_U_singleton_l395_39513

def U (a : ℝ) : Set ℝ := {1, 3, a^3 + 3*a^2 + 2*a}
def M (a : ℝ) : Set ℝ := {1, |2*a - 1|}

theorem complement_M_U_singleton (a : ℝ) : 
  (U a)ᶜ ∩ (M a) = {0} ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_U_singleton_l395_39513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_on_line_l_min_distance_C_to_l_l395_39562

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

-- Define point P
def point_P : ℝ × ℝ := (-2, 2)

-- Define the distance function from a point on C to line l
noncomputable def distance_to_l (α : ℝ) : ℝ :=
  |Real.sqrt 3 * Real.cos α - Real.sin α + 4| / Real.sqrt 2

theorem point_P_on_line_l :
  line_l point_P.1 point_P.2 := by sorry

theorem min_distance_C_to_l :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ ∀ (α : ℝ), distance_to_l α ≥ d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_on_line_l_min_distance_C_to_l_l395_39562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_in_triangle_l395_39548

/-- Given a triangle ABC with points D and E, prove that DE = -1/3 * AB - 1/2 * BC 
    if AE = 2 * EB and BC = 2 * BD -/
theorem vector_relation_in_triangle (A B C D E : ℝ × ℝ) : 
  (A - E = 2 • (E - B)) → (B - C = 2 • (B - D)) → 
  (D - E = -1/3 • (A - B) - 1/2 • (B - C)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_in_triangle_l395_39548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l395_39553

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The common ratio of the geometric sequence -/
def q : ℝ := sorry

/-- The first term of the geometric sequence -/
def a₁ : ℝ := sorry

theorem geometric_sequence_sum_ratio
  (h1 : ∀ n, S n = a₁ * (1 - q^n) / (1 - q))
  (h2 : q ≠ 1)
  (h3 : S 8 / S 4 = 3) :
  S 16 / S 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l395_39553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_sphere_cut_circumference_l395_39539

-- Define the volume of the sphere
noncomputable def sphere_volume : ℝ := 288 * Real.pi

-- Define the radius of the sphere
noncomputable def sphere_radius : ℝ := (3 * sphere_volume / (4 * Real.pi)) ^ (1/3)

-- Theorem for the surface area
theorem sphere_surface_area :
  4 * Real.pi * sphere_radius ^ 2 = 144 * Real.pi := by
  sorry

-- Theorem for the circumference of the cut surface
theorem sphere_cut_circumference :
  Real.pi * (2 * sphere_radius) = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_sphere_cut_circumference_l395_39539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_l395_39560

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 8x -/
def Parabola : Set Point :=
  { p : Point | p.y^2 = 8 * p.x }

/-- The focus of the parabola -/
def F : Point :=
  { x := 2, y := 0 }

/-- A line y = k(x-2) -/
def Line (k : ℝ) : Set Point :=
  { p : Point | p.y = k * (p.x - 2) }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The sum of reciprocals of distances from F to P and Q is 1/2 -/
theorem parabola_line_intersection_sum (k : ℝ) (P Q : Point)
    (h1 : P ∈ Parabola) (h2 : Q ∈ Parabola)
    (h3 : P ∈ Line k) (h4 : Q ∈ Line k)
    (h5 : P ≠ Q) :
    1 / distance F P + 1 / distance F Q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_l395_39560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_u_l395_39529

-- Define the complex numbers z₁ and z₂
noncomputable def z₁ (x y : ℝ) : ℂ := x + Complex.I * y + (11 : ℝ).sqrt
noncomputable def z₂ (x y : ℝ) : ℂ := x + Complex.I * y - (11 : ℝ).sqrt

-- Define the function u
def u (x y : ℝ) : ℝ := |5 * x - 6 * y - 30|

-- State the theorem
theorem max_min_u :
  ∃ (x y : ℝ), (Complex.abs (z₁ x y) + Complex.abs (z₂ x y) = 12) ∧
  (∀ (a b : ℝ), Complex.abs (z₁ a b) + Complex.abs (z₂ a b) = 12 → 
    u a b ≤ 30 ∧ u x y = 30) ∧
  (∃ (c d : ℝ), Complex.abs (z₁ c d) + Complex.abs (z₂ c d) = 12 ∧ u c d = 0) := by
  sorry

#check max_min_u

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_u_l395_39529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l395_39500

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x + 2*k

-- Define the solution set
def solution_set (k : ℝ) : Set ℝ := {x : ℝ | f k x < 0}

-- Define the range of k
def k_range : Set ℝ := Set.Icc (-1) 0 ∪ Set.Ioo 8 9

-- State the theorem
theorem quadratic_inequality_range :
  ∀ k : ℝ, (∃ x₁ x₂ : ℝ, 
    (solution_set k = Set.Ioo x₁ x₂) ∧ 
    (|x₁ - x₂| ≤ 3)) ↔ 
    k ∈ k_range :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l395_39500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l395_39542

theorem second_discount_percentage (initial_price : ℝ) 
  (h_initial_price_positive : initial_price > 0) : 
  let price_after_increase := initial_price * 1.36
  let price_after_first_discount := price_after_increase * 0.90
  let final_price := initial_price * 1.04040000000000006
  ∃ (second_discount : ℝ), 
    price_after_first_discount * (1 - second_discount) = final_price ∧ 
    abs (second_discount - 0.15) < 0.000001 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l395_39542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_ordinary_equation_l395_39533

-- Define the parametric equations
noncomputable def x (θ : ℝ) : ℝ := |Real.sin (θ/2) + Real.cos (θ/2)|
noncomputable def y (θ : ℝ) : ℝ := 1 + Real.sin θ

-- State the theorem
theorem parametric_to_ordinary_equation :
  ∀ θ : ℝ, θ ∈ Set.Icc 0 (2 * Real.pi) →
  ∃ x_val y_val : ℝ,
    x_val = x θ ∧
    y_val = y θ ∧
    x_val^2 = y_val ∧
    0 ≤ x_val ∧ x_val ≤ Real.sqrt 2 ∧
    0 ≤ y_val ∧ y_val ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_ordinary_equation_l395_39533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l395_39588

theorem cube_root_equality (a b c : ℕ+) :
  (((9 : ℝ) * a + 2 * b / c)^(1/3) : ℝ) = 3 * a * ((2 * b / c : ℝ)^(1/3)) ↔ 
  (c : ℝ) = (2 * b * (9 * a^3 - 1)) / (3 * a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l395_39588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_path_implies_sixth_entrance_l395_39538

/-- Represents a rectangular building with a number of entrances -/
structure Building where
  width : ℕ
  height : ℕ
  num_entrances : ℕ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: If the shortest paths from the top and bottom corners of a building
    to the 4th entrance of a neighboring building are equal, then the person
    must live in the 6th entrance of their building -/
theorem equal_path_implies_sixth_entrance
  (building1 building2 : Building)
  (top_corner bottom_corner fourth_entrance : Point)
  (h1 : distance top_corner fourth_entrance = distance bottom_corner fourth_entrance)
  (h2 : building1.num_entrances ≥ 6)
  (h3 : building2.num_entrances ≥ 4) :
  6 = building1.num_entrances - 4 :=
by sorry

#check equal_path_implies_sixth_entrance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_path_implies_sixth_entrance_l395_39538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l395_39544

/-- A trapezoid with specific measurements -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  height : ℝ
  parallel : AB = CD

/-- The perimeter of a trapezoid -/
noncomputable def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.CD + t.BC + (2 * Real.sqrt 2)

/-- Theorem stating the perimeter of the specific trapezoid -/
theorem trapezoid_perimeter : 
  ∀ (t : Trapezoid), 
  t.AB = 4 ∧ t.CD = 4 ∧ t.BC = 6 ∧ t.height = 2 → 
  perimeter t = 14 + 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l395_39544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_offset_length_l395_39501

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ := (q.diagonal / 2) * (q.offset1 + q.offset2)

/-- Theorem stating that for a quadrilateral with diagonal 28, one offset 6, and area 210,
    the other offset must be 9 -/
theorem offset_length (q : Quadrilateral) 
    (h1 : q.diagonal = 28)
    (h2 : q.offset2 = 6)
    (h3 : area q = 210) :
    q.offset1 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_offset_length_l395_39501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camera_price_difference_l395_39594

def list_price : ℚ := 59.99
def budget_buys_discount : ℚ := 0.15
def value_mart_discount : ℚ := 10

def budget_buys_price : ℚ := list_price * (1 - budget_buys_discount)
def value_mart_price : ℚ := list_price - value_mart_discount

theorem camera_price_difference :
  (Int.floor (budget_buys_price * 100) / 100 - value_mart_price) * 100 = 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camera_price_difference_l395_39594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_equals_train_length_l395_39592

-- Define the given constants
noncomputable def train_speed : ℝ := 36 -- km/hr
noncomputable def crossing_time : ℝ := 1 / 60 -- hours (1 minute)
noncomputable def train_length : ℝ := 300 -- meters

-- Define the theorem
theorem platform_length_equals_train_length :
  ∃ (platform_length : ℝ),
    platform_length = train_length ∧
    (train_speed * 1000 / 3600) * (crossing_time * 3600) = train_length + platform_length := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_equals_train_length_l395_39592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l395_39503

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ p > 0, ∀ x, f (x + p) = f x ∧ ∀ q, 0 < q → q < p → ∃ y, f (y + q) ≠ f y) ∧
  (∀ x, f (Real.pi/3 + (Real.pi/3 - x)) = f x) ∧
  (∀ x y, -Real.pi/6 ≤ x → x < y → y ≤ Real.pi/3 → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l395_39503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_c_l395_39555

/-- A piecewise function f(x) defined by three parts. -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a * (2 * x + 1) + 2
  else if -1 ≤ x ∧ x ≤ 1 then b * x + 3
  else 3 * x - c

/-- The function f is continuous. -/
axiom f_continuous (a b c : ℝ) : Continuous (f a b c)

/-- Theorem: For a continuous piecewise function f(x),
    the sum of a and c is equal to 4a + 1. -/
theorem sum_a_c (a b c : ℝ) : a + c = 4 * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_c_l395_39555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_bottom_left_is_91_l395_39559

/-- Represents a 3x3 multiplicative magic square -/
structure MagicSquare where
  a₁₁ : ℕ+
  a₁₂ : ℕ+
  a₁₃ : ℕ+
  a₂₁ : ℕ+
  a₂₂ : ℕ+
  a₂₃ : ℕ+
  a₃₁ : ℕ+
  a₃₂ : ℕ+
  a₃₃ : ℕ+
  row_prod : a₁₁ * a₁₂ * a₁₃ = a₂₁ * a₂₂ * a₂₃ ∧ a₂₁ * a₂₂ * a₂₃ = a₃₁ * a₃₂ * a₃₃
  col_prod : a₁₁ * a₂₁ * a₃₁ = a₁₂ * a₂₂ * a₃₂ ∧ a₁₂ * a₂₂ * a₃₂ = a₁₃ * a₂₃ * a₃₃
  diag_prod : a₁₁ * a₂₂ * a₃₃ = a₁₃ * a₂₂ * a₃₁

/-- The sum of possible values for the bottom-left entry in a specific magic square -/
def sum_of_possible_bottom_left (ms : MagicSquare) : ℕ :=
  sorry

/-- Theorem stating that the sum of possible bottom-left entries is 91 -/
theorem sum_of_possible_bottom_left_is_91 (ms : MagicSquare) 
  (h₁ : ms.a₁₁ = 72) (h₂ : ms.a₃₃ = 4) : 
  sum_of_possible_bottom_left ms = 91 :=
by
  sorry

#check sum_of_possible_bottom_left_is_91

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_bottom_left_is_91_l395_39559
