import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_condition_l462_46206

-- Define the power function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

-- Define the property of being a decreasing function on (0, +∞)
def is_decreasing_on_positive (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → g y < g x

-- State the theorem
theorem power_function_decreasing_condition :
  ∃ m : ℝ, is_decreasing_on_positive (f m) ∧ m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_condition_l462_46206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base9PerfectSquare_l462_46237

-- Define a function to convert base 9 number to decimal
def base9ToDecimal (b a : ℕ) : ℕ := 729 * b + 81 * a + 54

-- Define the property of being a perfect square
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

-- Define the main theorem
theorem base9PerfectSquare (b : ℕ) (h1 : b ≠ 0) (h2 : b < 10) :
  (∃ a : ℕ, a < 9 ∧ isPerfectSquare (base9ToDecimal b a)) ↔ b ∈ Finset.range 10 \ {0} :=
sorry

#check base9PerfectSquare

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base9PerfectSquare_l462_46237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_l462_46203

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - 1

noncomputable def g (a b : ℝ) (x : ℝ) := (a * x + 1) / (b * x - 1)

theorem solution_sets (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ 1 < x ∧ x < 2) →
  (∀ x, g a b x > 0 ↔ 2/3 < x ∧ x < 2) :=
by
  sorry

#check solution_sets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_l462_46203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_product_is_two_l462_46222

-- Define the circle and parabola
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 5
def parabolaEq (x y p : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the intersection point A
def pointA (x₀ : ℝ) : Prop := circleEq x₀ 2 ∧ parabolaEq x₀ 2 (2 : ℝ)

-- Define point B as diametrically opposite to A
def pointB (x₀ : ℝ) : Prop := circleEq (-x₀) (-2)

-- Define the line through B
def lineThroughB (x y m c : ℝ) : Prop := y = m*x + c ∧ -2 = m*(-1) + c

-- Define points D and E on the parabola
def pointsDE (x₁ y₁ x₂ y₂ p : ℝ) : Prop :=
  parabolaEq x₁ y₁ p ∧ parabolaEq x₂ y₂ p ∧
  ∃ m c, lineThroughB x₁ y₁ m c ∧ lineThroughB x₂ y₂ m c

-- Theorem statement
theorem slopes_product_is_two (x₀ x₁ y₁ x₂ y₂ p : ℝ) :
  pointA x₀ →
  pointB x₀ →
  pointsDE x₁ y₁ x₂ y₂ p →
  ((y₁ - 2) / (x₁ - x₀)) * ((y₂ - 2) / (x₂ - x₀)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_product_is_two_l462_46222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_derivative_of_y_l462_46287

noncomputable def y (x : ℝ) : ℝ := (x^2 + 3*x + 1) * Real.exp (3*x + 2)

theorem fifth_derivative_of_y (x : ℝ) :
  (deriv^[5] y) x = 3^3 * (9*x^2 + 57*x + 74) * Real.exp (3*x + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_derivative_of_y_l462_46287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_proof_l462_46266

/-- The distance to Margo's friend's house -/
noncomputable def distance_to_friends_house : ℝ := 0.75

/-- Margo's uphill walking speed in miles per hour -/
noncomputable def uphill_speed : ℝ := 3

/-- Margo's downhill walking speed in miles per hour -/
noncomputable def downhill_speed : ℝ := 6

/-- Time taken for uphill walk in hours -/
noncomputable def uphill_time : ℝ := 15 / 60

/-- Time taken for downhill walk in hours -/
noncomputable def downhill_time : ℝ := 10 / 60

/-- Theorem stating that the distance to Margo's friend's house is 0.75 miles -/
theorem distance_proof : 
  uphill_speed * uphill_time = distance_to_friends_house ∧
  downhill_speed * downhill_time = distance_to_friends_house := by
  sorry

#check distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_proof_l462_46266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l462_46226

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (c : ℝ), a = c • b ∨ b = c • a

theorem vector_parallel_condition (a b : V) (h : a ≠ 0 ∨ b ≠ 0) :
  (∃ (c : ℝ), a = c • b) → parallel a b ∧
  ¬ (parallel a b → ∃ (c : ℝ), a = c • b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l462_46226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_min_of_three_uniform_l462_46293

open Real

variable (a b c : ℝ)

noncomputable def uniform_dist (lower upper : ℝ) : Type := Set ℝ

noncomputable def random_var (dist : Type) : Type := dist → ℝ

noncomputable def expected_value {dist : Type} (X : random_var dist) : ℝ := sorry

noncomputable def min_of_three (x y z : ℝ) : ℝ := min x (min y z)

theorem expected_min_of_three_uniform 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0)
  (x : random_var (uniform_dist 0 a))
  (y : random_var (uniform_dist 0 b))
  (z : random_var (uniform_dist 0 c)) :
  expected_value (λ ω => min_of_three (x ω) (y ω) (z ω)) = 
    (c^2 / 2) * (1/a + 1/b + 1/c) - 
    (2*c^3 / 3) * (1/(a*b) + 1/(b*c) + 1/(c*a)) + 
    (3*c^4) / (4*a*b*c) :=
by sorry

#check expected_min_of_three_uniform

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_min_of_three_uniform_l462_46293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_sixteen_pi_thirds_l462_46218

theorem cos_negative_sixteen_pi_thirds : Real.cos (-16 * π / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_sixteen_pi_thirds_l462_46218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_a_range_l462_46207

theorem log_inequality_implies_a_range (a : ℝ) : 
  (0 < a) → (a ≠ 1) → (Real.log (a^2 + 1) / Real.log a < Real.log (2*a) / Real.log a) → 
  (Real.log (2*a) / Real.log a < 0) → (1/2 < a) ∧ (a < 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_a_range_l462_46207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increased_probability_after_removal_l462_46254

def T : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13}

def sumPairs (s : Finset Nat) : Finset (Nat × Nat) :=
  s.product s |>.filter (fun p => p.1 < p.2 ∧ p.1 + p.2 = 15)

def probability (s : Finset Nat) : Rat :=
  (sumPairs s).card / Nat.choose s.card 2

theorem increased_probability_after_removal (m : Nat) :
  m ∈ T →
  m ∉ (sumPairs T).image Prod.fst →
  m ∉ (sumPairs T).image Prod.snd →
  probability (T.erase m) > probability T := by
  sorry

#eval T.filter (fun m => 
  m ∉ (sumPairs T).image Prod.fst ∧
  m ∉ (sumPairs T).image Prod.snd ∧
  probability (T.erase m) > probability T)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increased_probability_after_removal_l462_46254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_puppy_weight_l462_46201

/-- The weight of the first puppy -/
def p1 : ℚ := sorry

/-- The weight of the second puppy -/
def p2 : ℚ := sorry

/-- The weight of the smaller cat -/
def c1 : ℚ := sorry

/-- The weight of the larger cat -/
def c2 : ℚ := sorry

/-- The total weight of all animals is 36 pounds -/
axiom total_weight : p1 + p2 + c1 + c2 = 36

/-- The first puppy and larger cat weigh three times the smaller cat -/
axiom weight_relation1 : p1 + c2 = 3 * c1

/-- The first puppy and smaller cat weigh the same as the larger cat -/
axiom weight_relation2 : p1 + c1 = c2

/-- The second puppy weighs 1.5 times the first puppy -/
axiom weight_relation3 : p2 = 1.5 * p1

theorem second_puppy_weight : p2 = 108 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_puppy_weight_l462_46201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_pie_ratio_l462_46238

theorem apple_pie_ratio (bonnie_apples samuel_more_apples samuel_left : ℕ) 
  (h1 : bonnie_apples = 8)
  (h2 : samuel_more_apples = 20)
  (h3 : samuel_left = 10) : 
  (((bonnie_apples + samuel_more_apples) - (bonnie_apples + samuel_more_apples) / 2 - samuel_left : ℚ) / 
   (bonnie_apples + samuel_more_apples)) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_pie_ratio_l462_46238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_l462_46298

def M : Finset Nat := {1, 2}
def N : Finset Nat := {2, 3}

theorem number_of_proper_subsets (M N : Finset Nat) (hM : M = {1, 2}) (hN : N = {2, 3}) :
  Finset.card (Finset.powerset (M ∪ N) \ {∅}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_l462_46298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_equals_half_compound_interest_principal_uniqueness_l462_46279

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- The principal that satisfies the given conditions -/
def principal_to_find : ℝ := 1400

theorem simple_interest_equals_half_compound_interest :
  simple_interest principal_to_find 10 3 = 
    (1/2) * compound_interest 4000 10 2 := by
  sorry

/-- The principal satisfying the conditions is unique -/
theorem principal_uniqueness (P : ℝ) :
  simple_interest P 10 3 = (1/2) * compound_interest 4000 10 2 →
  P = principal_to_find := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_equals_half_compound_interest_principal_uniqueness_l462_46279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l462_46228

theorem sin_cos_difference (α : Real) : 
  (α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) → 
  (Real.sin α + Real.cos α = 1/3) → 
  (Real.sin α - Real.cos α = -Real.sqrt 17 / 3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l462_46228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l462_46220

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x : ℝ | x > 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l462_46220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_clearable_iff_div_by_three_l462_46275

/-- Represents a move on the board -/
inductive Move
  | PlaceTromino : Move
  | ClearColumn : Move
  | ClearRow : Move

/-- Represents the state of the board -/
def Board (n : ℕ) := Fin n → Fin n → Bool

/-- Applies a move to the board -/
def applyMove (n : ℕ) (b : Board n) (m : Move) : Board n :=
  sorry

/-- Checks if the board is empty -/
def isEmpty {n : ℕ} (b : Board n) : Prop :=
  ∀ i j, b i j = false

/-- Represents a sequence of moves -/
def MoveSequence (n : ℕ) := List Move

/-- Applies a sequence of moves to the board -/
def applyMoveSequence (n : ℕ) (b : Board n) (ms : MoveSequence n) : Board n :=
  sorry

/-- The main theorem -/
theorem board_clearable_iff_div_by_three {n : ℕ} (h : n ≥ 2) :
  (∃ (ms : MoveSequence n), ms ≠ [] ∧ isEmpty (applyMoveSequence n (λ _ _ ↦ false) ms)) ↔ 3 ∣ n :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_clearable_iff_div_by_three_l462_46275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_distance_l462_46230

/-- Calculates the total distance John travels given his speeds and running times -/
noncomputable def total_distance (solo_speed : ℝ) (dog_speed : ℝ) (solo_time : ℝ) (dog_time : ℝ) : ℝ :=
  solo_speed * solo_time / 60 + dog_speed * dog_time / 60

/-- Proves that John travels 5 miles given the problem conditions -/
theorem john_distance : 
  total_distance 4 6 30 30 = 5 := by
  -- Unfold the definition of total_distance
  unfold total_distance
  -- Simplify the expression
  simp [mul_div_assoc, add_div]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_distance_l462_46230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_person_distance_l462_46209

/-- Two people moving towards each other with different speeds --/
structure MovingPeople where
  total_distance : ℝ
  speed_ratio : ℝ
  meeting_point_offset : ℝ

/-- Calculate the distance of the slower person from the faster person's starting point --/
noncomputable def distance_of_slower_person (p : MovingPeople) : ℝ :=
  p.total_distance - (p.total_distance * (p.speed_ratio / (1 + p.speed_ratio)))

/-- Theorem: When two people move towards each other with speed ratio 6:5 and meet 5 km from midpoint,
    the slower person will be 5/3 km away from the faster person's starting point when they switch positions --/
theorem slower_person_distance (p : MovingPeople) 
  (h1 : p.speed_ratio = 6/5)
  (h2 : p.meeting_point_offset = 5)
  (h3 : p.total_distance = 2 * p.meeting_point_offset) :
  distance_of_slower_person p = 5/3 := by
  sorry

#check slower_person_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_person_distance_l462_46209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_only_maximum_l462_46210

open Real

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/6) * x^3 - (1/2) * m * x^2 + x

-- Define the first derivative of f(x)
noncomputable def f' (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - m * x + 1

-- Define the second derivative of f(x)
noncomputable def f'' (m : ℝ) (x : ℝ) : ℝ := x - m

-- Theorem statement
theorem f_has_only_maximum (m : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 2, f'' m x < 0) →
  ∃ c ∈ Set.Ioo (-1) 2, (∀ x ∈ Set.Ioo (-1) 2, f m x ≤ f m c) ∧
  ¬∃ d ∈ Set.Ioo (-1) 2, ∀ x ∈ Set.Ioo (-1) 2, f m x ≥ f m d :=
by
  sorry

#check f_has_only_maximum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_only_maximum_l462_46210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_jackets_afforded_l462_46249

def budget : ℚ := 350
def shirt_price : ℚ := 12
def pants_price : ℚ := 20
def jacket_price : ℚ := 30
def max_jackets : ℕ := 5

def shirt_cost (quantity : ℕ) (discount : ℚ) : ℚ :=
  (shirt_price * quantity) * (1 - discount)

def pants_cost (quantity : ℕ) : ℚ :=
  pants_price * quantity - (pants_price * (quantity / 2) * (1/2))

def jacket_cost (quantity : ℕ) : ℚ :=
  (jacket_price * quantity) * (11/10)

def total_shirt_cost : ℚ :=
  shirt_cost 5 (1/10) + shirt_cost 7 (3/20) + shirt_cost 3 (1/5)

def total_pants_cost : ℚ :=
  pants_cost 4

def remaining_budget : ℚ :=
  budget - (total_shirt_cost + total_pants_cost)

theorem max_jackets_afforded :
  ∃ n : ℕ, n ≤ max_jackets ∧
  jacket_cost n ≤ remaining_budget ∧
  ∀ m : ℕ, m > n → jacket_cost m > remaining_budget :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_jackets_afforded_l462_46249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l462_46233

/-- Parabola E: x^2 = 4y -/
def parabola_E (x y : ℝ) : Prop := x^2 = 4*y

/-- Circle C: x^2 + (y-3)^2 = 1 -/
def circle_C (x y : ℝ) : Prop := x^2 + (y-3)^2 = 1

/-- Point P on parabola E -/
structure PointP where
  x : ℝ
  y : ℝ
  on_parabola : parabola_E x y

/-- Point Q on circle C -/
structure PointQ where
  x : ℝ
  y : ℝ
  on_circle : circle_C x y

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: Minimum distance between P and Q is 2√2 - 1 -/
theorem min_distance_PQ :
  ∃ (min_dist : ℝ),
    (∀ (P : PointP) (Q : PointQ), distance (P.x, P.y) (Q.x, Q.y) ≥ min_dist) ∧
    min_dist = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l462_46233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_corner_values_l462_46212

-- Define the type for cube faces
def Face := Fin 6

-- Define the function that gives the opposite face
def opposite : Face → Face
| ⟨0, _⟩ => ⟨5, by norm_num⟩
| ⟨1, _⟩ => ⟨4, by norm_num⟩
| ⟨2, _⟩ => ⟨3, by norm_num⟩
| ⟨3, _⟩ => ⟨2, by norm_num⟩
| ⟨4, _⟩ => ⟨1, by norm_num⟩
| ⟨5, _⟩ => ⟨0, by norm_num⟩
| ⟨n+6, h⟩ => absurd h (by norm_num)

-- Define the function that gives the value of a face
def faceValue : Face → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 2
| ⟨2, _⟩ => 3
| ⟨3, _⟩ => 4
| ⟨4, _⟩ => 5
| ⟨5, _⟩ => 6
| ⟨n+6, h⟩ => absurd h (by norm_num)

-- Define the property that opposite faces sum to 7
def oppositeFacesSumToSeven : Prop :=
  ∀ f : Face, faceValue f + faceValue (opposite f) = 7

-- Define a corner as a triple of faces
def Corner := (Face × Face × Face)

-- Define the list of all corners
def allCorners : List Corner :=
  [
    (⟨0, by norm_num⟩, ⟨1, by norm_num⟩, ⟨2, by norm_num⟩),
    (⟨0, by norm_num⟩, ⟨1, by norm_num⟩, ⟨3, by norm_num⟩),
    (⟨0, by norm_num⟩, ⟨2, by norm_num⟩, ⟨4, by norm_num⟩),
    (⟨0, by norm_num⟩, ⟨3, by norm_num⟩, ⟨4, by norm_num⟩),
    (⟨5, by norm_num⟩, ⟨1, by norm_num⟩, ⟨2, by norm_num⟩),
    (⟨5, by norm_num⟩, ⟨1, by norm_num⟩, ⟨3, by norm_num⟩),
    (⟨5, by norm_num⟩, ⟨2, by norm_num⟩, ⟨4, by norm_num⟩),
    (⟨5, by norm_num⟩, ⟨3, by norm_num⟩, ⟨4, by norm_num⟩)
  ]

-- Define the value of a corner
def cornerValue (c : Corner) : ℕ :=
  faceValue c.1 * faceValue c.2.1 * faceValue c.2.2

-- The main theorem
theorem sum_of_corner_values (h : oppositeFacesSumToSeven) :
  (allCorners.map cornerValue).sum = 343 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_corner_values_l462_46212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_intersection_point_l462_46299

/-- A rectangle in the Cartesian plane --/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- A line in the Cartesian plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The area of a region --/
noncomputable def area (r : ℝ) : ℝ := sorry

/-- Checks if a point is above a rectangle --/
def isAbove (p : ℝ × ℝ) (r : Rectangle) : Prop :=
  p.2 > r.y2

/-- Checks if two lines intersect at a given point --/
def intersectAt (l1 l2 : Line) (p : ℝ × ℝ) : Prop :=
  l1.slope * p.1 + l1.intercept = p.2 ∧
  l2.slope * p.1 + l2.intercept = p.2

/-- Checks if two lines divide a rectangle into three equal areas --/
def divideEqualAreas (l1 l2 : Line) (r : Rectangle) : Prop :=
  ∃ a : ℝ, a > 0 ∧ area a = area a ∧ area a = area a

theorem rectangle_division_intersection_point :
  ∀ (r : Rectangle) (l1 l2 : Line),
    r.x1 = 0 ∧ r.y1 = 0 ∧ r.x2 = 10 ∧ r.y2 = 8 →
    l1.slope = -3 ∧ l2.slope = 3 →
    divideEqualAreas l1 l2 r →
    ∃ (p : ℝ × ℝ), intersectAt l1 l2 p ∧ isAbove p r →
    p = (5, 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_intersection_point_l462_46299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_goals_moment_l462_46297

/-- Represents the progress of a hockey match --/
def MatchProgress : Type := { t : ℝ // 0 ≤ t ∧ t ≤ 1 }

/-- Goals scored by Dynamo at time t --/
def dynamoGoals (t : MatchProgress) : ℝ := sorry

/-- Goals scored by Spartak at time t --/
def spartakGoals (t : MatchProgress) : ℝ := sorry

/-- The final score of the match --/
axiom final_score : ∃ (t : MatchProgress), dynamoGoals t = 8 ∧ spartakGoals t = 5

/-- The theorem to prove --/
theorem equal_goals_moment :
  ∃ t : MatchProgress, dynamoGoals t = spartakGoals t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_goals_moment_l462_46297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_in_unit_square_l462_46205

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Definition of a unit square ABCD -/
def UnitSquare : (Point × Point × Point × Point) :=
  (⟨0, 0⟩, ⟨1, 0⟩, ⟨1, 1⟩, ⟨0, 1⟩)

/-- Predicate to check if a point is inside the unit square -/
def isInside (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1

theorem inequality_in_unit_square (P Q : Point) 
  (hP : isInside P) (hQ : isInside Q) : 
  let (A, B, C, D) := UnitSquare
  13 * (distance P A + distance Q C) + 14 * distance P Q + 
  15 * (distance P B + distance Q D) > 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_in_unit_square_l462_46205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_points_theorem_l462_46267

noncomputable section

structure Point where
  x : ℝ
  y : ℝ

def Hyperbola (p : Point) : Prop :=
  p.x^2 / 4 - p.y^2 / 9 = 1

def DotProduct (p q : Point) : ℝ :=
  p.x * q.x + p.y * q.y

noncomputable def Distance (p : Point) : ℝ :=
  Real.sqrt (p.x^2 + p.y^2)

theorem hyperbola_points_theorem (A B P : Point) 
  (hA : Hyperbola A) (hB : Hyperbola B) 
  (h_perp : DotProduct A B = 0)
  (h_P : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P.x = t * A.x + (1 - t) * B.x ∧ P.y = t * A.y + (1 - t) * B.y)
  (h_P_perp : DotProduct P (Point.mk (A.x - B.x) (A.y - B.y)) = 0) :
  (1 / (Distance A)^2 + 1 / (Distance B)^2 = 5 / 36) ∧
  (Distance P = 6 * Real.sqrt 5 / 5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_points_theorem_l462_46267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_in_set_l462_46286

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / x^2 + 1 / x^4

-- Define the set of a values
def A : Set ℝ := Set.Ioo (-3) (-1/2) ∪ Set.Ioo (-1/2) (1/3)

-- Theorem statement
theorem f_inequality_implies_a_in_set (a : ℝ) :
  f (a - 2) < f (2 * a + 1) → a ∈ A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_in_set_l462_46286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l462_46256

/-- In an acute triangle ABC, if a/b + b/a = 6 cos(C), then tan(C)/tan(A) + tan(C)/tan(B) = 4 -/
theorem triangle_tangent_ratio (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive sides
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- acute angles
  a * Real.sin B = b * Real.sin A →  -- sine rule
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →  -- cosine rule
  a/b + b/a = 6 * Real.cos C →
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l462_46256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_range_l462_46245

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

noncomputable def focus_distance : ℝ := Real.sqrt 3

noncomputable def focus1 : ℝ × ℝ := (focus_distance, 0)
noncomputable def focus2 : ℝ × ℝ := (-focus_distance, 0)

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

noncomputable def vector_MF1 (x y : ℝ) : ℝ × ℝ := (focus_distance - x, -y)
noncomputable def vector_MF2 (x y : ℝ) : ℝ × ℝ := (-focus_distance - x, -y)

theorem hyperbola_point_range (x₀ y₀ : ℝ) :
  hyperbola x₀ y₀ →
  vector_dot_product (vector_MF1 x₀ y₀) (vector_MF2 x₀ y₀) < 0 →
  -Real.sqrt 3 / 3 < y₀ ∧ y₀ < Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_range_l462_46245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_n_for_prime_equation_l462_46242

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem product_of_n_for_prime_equation : ∃ (n : ℕ+), 
  (∀ k : ℕ+, is_prime (k.val^2 - 41*k.val + 408) → k = n) ∧ n.val = 406 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_n_for_prime_equation_l462_46242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_inequality_l462_46231

noncomputable section

-- Define the inverse proportion function
def inverse_prop (x : ℝ) : ℝ := -2 / x

-- Define the theorem
theorem inverse_prop_inequality (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : x₁ < x₂) (h2 : x₂ < 0) (h3 : 0 < x₃)
  (h4 : y₁ = inverse_prop x₁)
  (h5 : y₂ = inverse_prop x₂)
  (h6 : y₃ = inverse_prop x₃) :
  y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_inequality_l462_46231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l462_46271

/-- Helper function to calculate the distance ratio -/
noncomputable def distance_ratio (x y line_x point_x point_y : ℝ) : ℝ :=
  |x - line_x| / Real.sqrt ((x - point_x)^2 + (y - point_y)^2)

/-- The trajectory of a point with a given distance ratio to a fixed line and point -/
theorem trajectory_equation (a c : ℝ) (h : c > a) :
  ∃ b : ℝ, b = Real.sqrt (c^2 - a^2) ∧
  ∀ x y : ℝ, (distance_ratio x y (-a^2/c) (-c) 0 = a/c) ↔ 
    (x^2/a^2 - y^2/b^2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l462_46271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_horizontal_asymptote_l462_46234

/-- Represents a rational function f(x) = (ax^5 + bx^4 + cx^3 + dx^2 + ex + f) / (gx^4 + hx^3 + ix^2 + jx + k) -/
noncomputable def RationalFunction (a b c d e f g h i j k : ℝ) : ℝ → ℝ :=
  fun x => (a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) / (g*x^4 + h*x^3 + i*x^2 + j*x + k)

/-- The given function has no horizontal asymptote -/
theorem no_horizontal_asymptote :
  ¬∃(L : ℝ), ∀ε>0, ∃N, ∀x>N, |RationalFunction 18 7 5 3 1 2 4 3 5 2 6 x - L| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_horizontal_asymptote_l462_46234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_flagpole_break_height_l462_46215

/-- 
Given a flagpole of height h that breaks and touches the ground at distance d from its base,
this function calculates the height of the breaking point.
-/
noncomputable def breakingPoint (h d : ℝ) : ℝ := 
  Real.sqrt (h^2 + d^2) / 2

/-- 
Theorem stating that for a 10-meter flagpole that breaks and touches the ground 
3 meters away from its base, the breaking point is at height √109 / 2 meters from the ground.
-/
theorem flagpole_break_height : 
  breakingPoint 10 3 = Real.sqrt 109 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_flagpole_break_height_l462_46215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_output_l462_46269

def MyOption : Type := Prod Nat Nat

def optionA : MyOption := (1, 3)
def optionB : MyOption := (4, 9)
def optionC : MyOption := (4, 12)
def optionD : MyOption := (4, 8)

def is_correct_output (o : MyOption) : Prop :=
  o = optionD

theorem correct_output :
  is_correct_output optionD ∧
  ¬is_correct_output optionA ∧
  ¬is_correct_output optionB ∧
  ¬is_correct_output optionC := by
  sorry

#check correct_output

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_output_l462_46269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_m_sum_a_b_l462_46243

def z (m : ℝ) : ℂ := (m - 1) * (m + 2) + (m - 1) * Complex.I

theorem purely_imaginary_m (m : ℝ) : z m = Complex.I * (z m).im → m = -2 := by sorry

theorem sum_a_b : 
  let m : ℝ := 2
  let w := (z m + Complex.I) / (z m - Complex.I)
  ∃ (a b : ℝ), w = a + b * Complex.I ∧ a + b = 13/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_m_sum_a_b_l462_46243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l462_46232

/-- Represents a person mowing their lawn -/
structure Mower where
  name : String
  lawn_area : ℝ
  mower_speed : ℝ
  break_time : ℝ

/-- Calculates the time taken to mow a lawn -/
noncomputable def mowing_time (m : Mower) : ℝ :=
  m.lawn_area / m.mower_speed + m.break_time

/-- Proves that Beth finishes mowing first -/
theorem beth_finishes_first (andy beth carlos : Mower)
  (h1 : andy.lawn_area = 3 * beth.lawn_area)
  (h2 : andy.lawn_area = 4 * carlos.lawn_area)
  (h3 : beth.mower_speed = andy.mower_speed)
  (h4 : carlos.mower_speed = andy.mower_speed / 2)
  (h5 : carlos.break_time = 10)
  (h6 : andy.break_time = 0)
  (h7 : beth.break_time = 0) :
  mowing_time beth < mowing_time andy ∧ mowing_time beth < mowing_time carlos := by
  sorry

#check beth_finishes_first

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l462_46232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_equal_if_same_length_and_direction_l462_46219

/-- Two vectors are equal if they have the same length and direction -/
theorem vectors_equal_if_same_length_and_direction 
  {V : Type*} [NormedAddCommGroup V] [Module ℝ V] {a b : V} : 
  (‖a‖ = ‖b‖) → (∃ (k : ℝ), k > 0 ∧ a = k • b) → a = b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_equal_if_same_length_and_direction_l462_46219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l462_46277

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) ^ 2 + 2 * Real.cos (x ^ 2)

-- State the theorem
theorem derivative_of_f (x : ℝ) :
  deriv f x = 2 * Real.sin (4 * x) - 4 * x * Real.sin (x ^ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l462_46277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_reciprocal_squares_l462_46260

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- Distance from the origin to the plane -/
  distance : ℝ
  /-- Coordinates of the intersection points -/
  a : ℝ  -- x-coordinate of point A
  b : ℝ  -- y-coordinate of point B
  c : ℝ  -- z-coordinate of point C
  /-- Conditions ensuring the plane intersects the axes away from the origin -/
  ha : a ≠ 0
  hb : b ≠ 0
  hc : c ≠ 0
  /-- Condition for the distance from the origin to the plane -/
  h_distance : distance = 2
  /-- Equation of the plane -/
  h_plane : 1 / a + 1 / b + 1 / c = 1 / distance

/-- Centroid of the triangle formed by the intersection points -/
noncomputable def centroid (p : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (p.a / 3, p.b / 3, p.c / 3)

/-- Theorem stating the result for the sum of reciprocals of squared centroid coordinates -/
theorem centroid_sum_reciprocal_squares (p : IntersectingPlane) :
  let (x, y, z) := centroid p
  1 / x^2 + 1 / y^2 + 1 / z^2 = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_reciprocal_squares_l462_46260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_functions_l462_46200

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n > 0 → f (Nat.factorial n) = Nat.factorial (f n)) ∧
  (∀ m n : ℕ, m > 0 → n > 0 → m ≠ n → (m - n : ℤ) ∣ (f m - f n : ℤ))

theorem characterize_valid_functions :
  ∀ f : ℕ → ℕ, is_valid_function f →
    (∀ n : ℕ, n > 0 → f n = 1) ∨
    (∀ n : ℕ, n > 0 → f n = 2) ∨
    (∀ n : ℕ, n > 0 → f n = n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_functions_l462_46200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_E_points_range_l462_46216

/-- Represents a football team --/
inductive Team : Type
| A | B | C | D | E
deriving Repr, DecidableEq

/-- The result of a match between two teams --/
inductive MatchResult
| Win (winner : Team) (loser : Team)
| Draw (team1 : Team) (team2 : Team)

/-- The tournament consisting of all matches --/
def Tournament := List MatchResult

/-- Calculate the points for a given team based on the tournament results --/
def calculatePoints (team : Team) (tournament : Tournament) : ℕ :=
  tournament.foldl
    (fun acc result =>
      match result with
      | MatchResult.Win winner loser =>
          if winner = team then acc + 3
          else if loser = team then acc
          else acc
      | MatchResult.Draw team1 team2 =>
          if team1 = team then acc + 1
          else if team2 = team then acc + 1
          else acc)
    0

/-- The theorem stating the range of points for team E --/
theorem team_E_points_range (tournament : Tournament) :
  (calculatePoints Team.A tournament = 1) →
  (calculatePoints Team.B tournament = 4) →
  (calculatePoints Team.C tournament = 7) →
  (calculatePoints Team.D tournament = 8) →
  (1 ≤ calculatePoints Team.E tournament) ∧
  (calculatePoints Team.E tournament ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_E_points_range_l462_46216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l462_46221

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  focus1 : Point
  focus2 : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  distance p e.focus1 + distance p e.focus2 = 2 * e.a

/-- The main theorem about the ellipse -/
theorem ellipse_theorem (e : Ellipse) (p : Point) : 
  e.focus1 = ⟨-3, 0⟩ →
  e.focus2 = ⟨3, 0⟩ →
  p = ⟨5, 8⟩ →
  isOnEllipse e p →
  e.a > 0 →
  e.b > 0 →
  e.a = 4 * Real.sqrt 2 + Real.sqrt 17 ∧
  e.b = Real.sqrt (32 + 16 * Real.sqrt 34) :=
by sorry

#check ellipse_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l462_46221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l462_46236

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 3)^2 + (y + 4)^2 = 4

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the directrix l
def directrix (x : ℝ) : Prop := x = -2

-- Define the focus F
def focus : ℝ × ℝ := (2, 0)

-- Define the distance from a point to the directrix
def m (x : ℝ) : ℝ := |x + 2|

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x1 - x2)^2 + (y1 - y2)^2).sqrt

-- Theorem statement
theorem min_distance_sum :
  ∃ (x y : ℝ), parabola x y ∧ 
  ∀ (x' y' : ℝ), parabola x' y' → 
    m x + distance x y (-3) (-4) ≤ m x' + distance x' y' (-3) (-4) ∧
    m x + distance x y (-3) (-4) = (41 : ℝ).sqrt := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l462_46236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l462_46240

noncomputable def g (z : ℂ) : ℂ := ((1 + Complex.I * Real.sqrt 3) * z + (4 * Real.sqrt 3 + 12 * Complex.I)) / 2

theorem rotation_center : 
  ∃ (d : ℂ), g d = d ∧ d = -4 * Real.sqrt 3 * (2 - Complex.I) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l462_46240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l462_46274

def base4_repr (n : ℕ) : ℕ × ℕ :=
  (n / 4, n % 4)

def base7_repr (n : ℕ) : ℕ × ℕ :=
  (n / 7, n % 7)

def misinterpreted_sum (n : ℕ) : ℕ :=
  let (a, b) := base4_repr n
  let (c, d) := base7_repr n
  10 * a + b + 10 * c + d

def digit_sum (n : ℕ) : ℕ :=
  let (a, b) := base4_repr n
  let (c, d) := base7_repr n
  a + b + c + d

def satisfies_condition (n : ℕ) : Bool :=
  misinterpreted_sum n = digit_sum n

theorem count_satisfying_numbers :
  (Finset.filter (λ n => n ≥ 10 ∧ satisfies_condition n) (Finset.range 90)).card = 5 := by
  sorry

#eval (Finset.filter (λ n => n ≥ 10 ∧ satisfies_condition n) (Finset.range 90)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l462_46274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l462_46296

/-- Calculates the average speed of a trip given the total distance,
    speed of the first half, and the time factor for the second half. -/
noncomputable def averageSpeed (totalDistance : ℝ) (firstHalfSpeed : ℝ) (secondHalfTimeFactor : ℝ) : ℝ :=
  let firstHalfDistance := totalDistance / 2
  let firstHalfTime := firstHalfDistance / firstHalfSpeed
  let secondHalfTime := firstHalfTime * secondHalfTimeFactor
  let totalTime := firstHalfTime + secondHalfTime
  totalDistance / totalTime

/-- Theorem stating that for a 640-mile trip with the given conditions,
    the average speed is 40 miles per hour. -/
theorem average_speed_theorem :
  averageSpeed 640 80 3 = 40 := by
  -- Unfold the definition of averageSpeed
  unfold averageSpeed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l462_46296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_100_mod_5_l462_46268

-- Define the sequence S
def S : ℕ → ℕ
  | 0 => 7  -- Add a case for 0 to cover all natural numbers
  | 1 => 7
  | n + 2 => 7^(S (n + 1))

-- State the theorem
theorem s_100_mod_5 : S 100 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_100_mod_5_l462_46268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_d_can_be_same_l462_46213

-- Define the prize types
inductive Prize
| First
| Second

-- Define the students
inductive Student
| A
| B
| C
| D
| E

-- Define a function to represent the prize allocation
def prize_allocation : Student → Prize := fun _ => Prize.Second

-- Define the conditions of the problem
axiom two_first_prizes :
  ∃ (s1 s2 : Student), s1 ≠ s2 ∧ 
  prize_allocation s1 = Prize.First ∧ 
  prize_allocation s2 = Prize.First

axiom three_second_prizes :
  ∃ (s1 s2 s3 : Student), s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3 ∧ 
  prize_allocation s1 = Prize.Second ∧ 
  prize_allocation s2 = Prize.Second ∧ 
  prize_allocation s3 = Prize.Second

axiom a_uncertainty :
  (prize_allocation Student.B = Prize.First ∧ prize_allocation Student.C = Prize.Second) ∨
  (prize_allocation Student.B = Prize.Second ∧ prize_allocation Student.C = Prize.First) ∨
  (prize_allocation Student.B = Prize.Second ∧ prize_allocation Student.C = Prize.Second)

axiom a_e_same_prize :
  prize_allocation Student.A = prize_allocation Student.E

-- Theorem to prove
theorem b_d_can_be_same : 
  ∃ (pa : Student → Prize), 
    pa Student.B = pa Student.D := by
  -- We'll use the existing prize_allocation as our witness
  use prize_allocation
  -- For now, we'll use sorry to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_d_can_be_same_l462_46213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_rate_correct_quadruple_rate_unique_l462_46288

/-- The annual interest rate that causes a sum to quadruple in 10 years with continuous compounding -/
noncomputable def quadruple_rate : ℝ := Real.log 4 / 10

/-- Theorem stating that the quadruple rate causes a sum to quadruple in 10 years -/
theorem quadruple_rate_correct : 
  ∀ (P : ℝ), P > 0 → Real.exp (quadruple_rate * 10) * P = 4 * P := by
  sorry

/-- Theorem stating that the quadruple rate is the unique rate that causes a sum to quadruple in 10 years -/
theorem quadruple_rate_unique : 
  ∀ (r : ℝ), (∀ (P : ℝ), P > 0 → Real.exp (r * 10) * P = 4 * P) → r = quadruple_rate := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_rate_correct_quadruple_rate_unique_l462_46288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l462_46223

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, 7)
def C : ℝ × ℝ := (4, 4)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  distance p1 p2 + distance p2 p3 + distance p3 p1

theorem triangle_perimeter : perimeter A B C = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l462_46223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l462_46247

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem f_monotone_increasing : 
  ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l462_46247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l462_46278

-- Define the function f
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Define the conditions
axiom A_positive : 0 < (2 : ℝ).sqrt
axiom ω_positive : 0 < (2 : ℝ)
axiom φ_range : -Real.pi/2 ≤ -Real.pi/6 ∧ -Real.pi/6 < Real.pi/2
axiom max_value : ∀ x, f (2 : ℝ).sqrt 2 (-Real.pi/6) x ≤ (2 : ℝ).sqrt
axiom symmetry : ∀ x, f (2 : ℝ).sqrt 2 (-Real.pi/6) (Real.pi/3 - x) = f (2 : ℝ).sqrt 2 (-Real.pi/6) (Real.pi/3 + x)
axiom period : ∀ x, f (2 : ℝ).sqrt 2 (-Real.pi/6) (x + Real.pi) = f (2 : ℝ).sqrt 2 (-Real.pi/6) x

-- Define g as a transformation of f
noncomputable def g (x : ℝ) : ℝ := f (2 : ℝ).sqrt 2 (-Real.pi/6) (2 * (x + Real.pi/12))

-- State the theorem to be proved
theorem f_and_g_properties :
  (∀ x, f (2 : ℝ).sqrt 2 (-Real.pi/6) x = (2 : ℝ).sqrt * Real.sin (2 * x - Real.pi/6)) ∧
  (∀ x ∈ Set.Icc 0 1, g x ≥ x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l462_46278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l462_46273

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the line 3y + 2x - 6 = 0 -/
noncomputable def slope₁ : ℝ := -2/3

/-- The slope of the line 4y + bx - 5 = 0 in terms of b -/
noncomputable def slope₂ (b : ℝ) : ℝ := -b/4

/-- Theorem: If the lines 3y + 2x - 6 = 0 and 4y + bx - 5 = 0 are perpendicular, then b = -6 -/
theorem perpendicular_lines (b : ℝ) : perpendicular slope₁ (slope₂ b) → b = -6 := by
  sorry

#check perpendicular_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l462_46273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_is_60_l462_46262

noncomputable def satisfies_equation (x y : ℝ) : Prop :=
  |15 * x| + |8 * y| + |120 - 15 * x - 8 * y| = 120

def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (8, 0)
def point_C : ℝ × ℝ := (0, 15)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

theorem area_of_triangle_is_60 :
  satisfies_equation point_A.1 point_A.2 ∧
  satisfies_equation point_B.1 point_B.2 ∧
  satisfies_equation point_C.1 point_C.2 →
  triangle_area point_A point_B point_C = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_is_60_l462_46262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_problem_l462_46265

-- Define the triangles and their properties
structure Triangle where
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the length of a side
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define that J lies between I and K
def between (I J K : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem triangle_similarity_problem (FGH IJK : Triangle) :
  similar FGH IJK →
  length FGH.G FGH.H = 30 →
  length IJK.F IJK.G = 18 →
  length IJK.G IJK.H = 15 →
  between IJK.F IJK.G IJK.H →
  length FGH.F FGH.G = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_problem_l462_46265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l462_46292

/-- A hyperbola is defined by its equation and the location of its focus -/
structure Hyperbola where
  m : ℝ
  equation : (x y : ℝ) → x^2 / m - y^2 / (3 + m) = 1
  focus : ℝ × ℝ := (2, 0)

/-- The value of m for a hyperbola with the given properties is 1/2 -/
theorem hyperbola_m_value (h : Hyperbola) : h.m = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l462_46292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_area_l462_46263

-- Define the area of a unit square
noncomputable def unit_square_area : ℝ := 1

-- Define the area of a right triangle with legs of 1 unit
noncomputable def right_triangle_area : ℝ := 1/2

-- Define the polygons
noncomputable def area_P : ℝ := 4 * unit_square_area
noncomputable def area_Q : ℝ := 6 * unit_square_area
noncomputable def area_R : ℝ := 3 * unit_square_area + 3 * right_triangle_area
noncomputable def area_S : ℝ := 6 * right_triangle_area
noncomputable def area_T : ℝ := 5 * unit_square_area + 2 * right_triangle_area

theorem largest_area : 
  area_Q = area_T ∧ 
  area_Q > area_P ∧ 
  area_Q > area_R ∧ 
  area_Q > area_S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_area_l462_46263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_three_fourths_l462_46291

/-- Represents a square divided into four congruent squares -/
structure DividedSquare where
  total_area : ℝ
  is_positive : total_area > 0

/-- Represents the shaded area within a DividedSquare -/
noncomputable def shaded_area (s : DividedSquare) : ℝ :=
  -- Two fully shaded squares
  (1/2 : ℝ) * s.total_area +
  -- Two smaller shaded squares within unshaded squares
  2 * ((1/2 : ℝ) * (1/4 : ℝ) * s.total_area)

/-- The fraction of the large square that is shaded -/
noncomputable def shaded_fraction (s : DividedSquare) : ℝ :=
  shaded_area s / s.total_area

/-- Theorem stating that the shaded fraction is three-fourths -/
theorem shaded_fraction_is_three_fourths (s : DividedSquare) :
  shaded_fraction s = 3/4 := by
  sorry

#check shaded_fraction_is_three_fourths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_three_fourths_l462_46291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_d_l462_46272

theorem find_d : ∃ d : ℝ, d = -8.5 ∧
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * n^2 + 19 * n - 70 = 0) ∧
  (let f : ℝ := d - ⌊d⌋; 4 * f^2 - 12 * f + 5 = 0) ∧
  0 ≤ d - ⌊d⌋ ∧ d - ⌊d⌋ < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_d_l462_46272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_two_element_set_l462_46208

theorem proper_subsets_of_two_element_set :
  ∀ (S : Finset ℕ), S.card = 2 → (S.powerset.card : ℕ) - 1 = 3 :=
by
  intro S h
  have h1 : S.powerset.card = 2^2 := by
    rw [Finset.card_powerset, h]
  calc
    (S.powerset.card : ℕ) - 1 = 2^2 - 1 := by rw [h1]
    _ = 4 - 1 := by rfl
    _ = 3 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_two_element_set_l462_46208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l462_46244

theorem triangle_inequality (a b c α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum : α + β + γ = Real.pi)
  (h_sine_law : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ) :
  b / Real.sin (γ + α / 3) + c / Real.sin (β + α / 3) > 2 / 3 * a / Real.sin (α / 3) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l462_46244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deductible_is_50000_l462_46255

/-- Represents an insurance contract with proportional liability system -/
structure InsuranceContract where
  insuredValue : ℚ
  actualValue : ℚ
  actualDamage : ℚ
  compensationPaid : ℚ

/-- Calculates the deductible (franchise) in an insurance contract -/
def calculateDeductible (contract : InsuranceContract) : ℚ :=
  let expectedCompensation := (contract.insuredValue / contract.actualValue) * contract.actualDamage
  expectedCompensation - contract.compensationPaid

/-- Theorem stating that for the given insurance contract, the deductible is 50000 -/
theorem deductible_is_50000 (contract : InsuranceContract) 
  (h1 : contract.insuredValue = 3750000)
  (h2 : contract.actualValue = 7500000)
  (h3 : contract.actualDamage = 2750000)
  (h4 : contract.compensationPaid = 1350000) :
  calculateDeductible contract = 50000 := by
  sorry

#eval calculateDeductible {
  insuredValue := 3750000,
  actualValue := 7500000,
  actualDamage := 2750000,
  compensationPaid := 1350000
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deductible_is_50000_l462_46255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l462_46261

theorem salary_calculation (salary : ℚ) 
  (food_expense : salary * (1/3 : ℚ) = salary - salary * (2/3 : ℚ))
  (rent_expense : salary * (1/4 : ℚ) = salary - salary * (3/4 : ℚ))
  (clothes_expense : salary * (1/5 : ℚ) = salary - salary * (4/5 : ℚ))
  (remaining : salary - salary * (1/3 : ℚ) - salary * (1/4 : ℚ) - salary * (1/5 : ℚ) = 1760) :
  ∃ (rounded_salary : ℚ), abs (rounded_salary - 8123.08) < 0.01 ∧ abs (rounded_salary - salary) < 0.01 := by
  sorry

#eval (1760 : ℚ) / (13/60 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l462_46261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_greater_necessary_not_sufficient_for_log_greater_l462_46282

theorem abs_greater_necessary_not_sufficient_for_log_greater :
  ∃ a b : ℝ, (|a| > |b| ∧ ¬(Real.log a > Real.log b)) ∧
  ∀ x y : ℝ, (Real.log x > Real.log y → |x| > |y|) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_greater_necessary_not_sufficient_for_log_greater_l462_46282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l462_46235

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.rpow 2 x + x^2) + Real.rpow 2 (-x)

-- Part 1: Prove that if f is even, then a = 1
theorem part1 (a : ℝ) (h : ∀ x, f a x = f a (-x)) : a = 1 := by
  sorry

-- Part 2: Prove that if the inequality holds, then m ≤ -1/3
theorem part2 (m : ℝ) (h : ∀ x > 0, m * (f 1 x) ≤ Real.rpow 2 (-x) + m*x^2 + m - 1) : m ≤ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l462_46235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_lemon_count_l462_46214

/-- Represents the number of lemons and oranges at a grocery store --/
structure GroceryInventory where
  lemons : ℚ
  oranges : ℚ

def GroceryInventory.ratio (inventory : GroceryInventory) : ℚ :=
  inventory.lemons / inventory.oranges

/-- Approximate equality for rational numbers --/
def approx_eq (x y : ℚ) (ε : ℚ) : Prop :=
  abs (x - y) < ε

notation x " ≈ " y => approx_eq x y 0.01

theorem grocery_lemon_count 
  (initial : GroceryInventory)
  (final : GroceryInventory)
  (h1 : initial.oranges = 60)
  (h2 : final.lemons = 20)
  (h3 : final.oranges = 40)
  (h4 : (final.ratio / initial.ratio) ≈ 0.6) :
  initial.lemons ≈ 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_lemon_count_l462_46214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_add_1567_minutes_to_3pm_l462_46253

/-- Represents time in 24-hour format -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def add_minutes (t : Time) (m : ℕ) : Time :=
  sorry

theorem add_1567_minutes_to_3pm (
  start_time : Time
) (h_start : start_time.hours = 15 ∧ start_time.minutes = 0) : 
  add_minutes start_time 1567 = ⟨17, 7, by norm_num, by norm_num⟩ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_add_1567_minutes_to_3pm_l462_46253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l462_46217

variable (a b c : ℝ)

def A (a b c : ℝ) : ℝ := 3 * a^2 * b - 2 * a * b^2 + a * b * c
def C (a b c : ℝ) : ℝ := 4 * a^2 * b - 3 * a * b^2 + 4 * a * b * c
def B (a b c : ℝ) : ℝ := C a b c - 2 * A a b c

theorem problem_solution :
  B a b c = -2 * a^2 * b + a * b^2 + 2 * a * b * c ∧
  2 * A a b c - B a b c = 8 * a^2 * b - 5 * a * b^2 ∧
  ∀ c₁ c₂ : ℝ, 2 * A a b c₁ - B a b c₁ = 2 * A a b c₂ - B a b c₂ ∧
  (a = 1/8 ∧ b = 1/5) → 2 * A a b c - B a b c = 0 := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l462_46217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l462_46211

/-- Represents a quadrilateral in 2D space -/
def Quadrilateral (α : Type) := Fin 4 → α × α

/-- Checks if a quadrilateral is a parallelogram -/
def IsParallelogram (q : Quadrilateral ℝ) : Prop := sorry

/-- Calculates the length of a line segment -/
def SegmentLength (p : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- Calculates the length of a diagonal in a quadrilateral -/
def DiagonalLength (q : Quadrilateral ℝ) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def Area (q : Quadrilateral ℝ) : ℝ := sorry

/-- Rotates a quadrilateral -/
def Quadrilateral.rotate (q : Quadrilateral ℝ) : Quadrilateral ℝ := sorry

/-- A parallelogram with specific side and diagonal lengths has an area of 4√5 -/
theorem parallelogram_area (ABCD : Quadrilateral ℝ) (h1 : IsParallelogram ABCD) 
  (h2 : SegmentLength (ABCD 0, ABCD 1) = 3)
  (h3 : DiagonalLength ABCD = 4)
  (h4 : DiagonalLength (Quadrilateral.rotate ABCD) = 2 * Real.sqrt 5) :
  Area ABCD = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l462_46211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_expression_simplification_l462_46258

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  5 * x - 1 < 3 * (x + 1) ∧ (x + 1) / 2 ≥ (2 * x + 1) / 5

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  -3 ≤ x ∧ x < 2

-- Theorem for the inequality system
theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ solution_set x :=
by sorry

-- Define the equation x^2 - 2x - 3 = 0
def equation (x : ℝ) : Prop :=
  x^2 - 2*x - 3 = 0

-- Define the expression
noncomputable def expression (x : ℝ) : ℝ :=
  (1 / (x + 1) + x - 1) / (x^2 / (x^2 + 2*x + 1))

-- Theorem for the expression simplification
theorem expression_simplification :
  ∀ x : ℝ, equation x → x ≠ -1 → expression x = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_expression_simplification_l462_46258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_info_to_determine_b_plus_g_l462_46204

-- Define the variables
variable (B G : ℕ) -- Number of boys and girls
variable (b g : ℝ) -- Average ages of boys and girls

-- Define the conditions
def average_age_boys (b g : ℝ) : ℝ := g
def average_age_girls (b g : ℝ) : ℝ := b
def average_age_all (b g : ℝ) : ℝ := b + g
def teacher_age : ℕ := 42

-- Theorem statement
theorem insufficient_info_to_determine_b_plus_g :
  ∀ (B G : ℕ) (b g : ℝ),
  b > 0 ∧ g > 0 ∧ B > 0 ∧ G > 0 →
  average_age_boys b g = g ∧
  average_age_girls b g = b ∧
  average_age_all b g = b + g →
  ¬∃! (sum : ℝ), sum = b + g :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_info_to_determine_b_plus_g_l462_46204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_equivalence_l462_46295

-- Define the function f(x) = log_a(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the property of a function being decreasing
def isDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem converse_equivalence (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x > 0, isDecreasing (f a) → f a 2 < 0) ↔
  (f a 2 ≥ 0 → ¬(isDecreasing (f a))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_equivalence_l462_46295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_l462_46257

noncomputable section

/-- Definition of the ellipse E -/
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Definition of the line l -/
def line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 2

/-- Definition of the area of triangle OPQ -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  4 * Real.sqrt (4 * k^2 - 3) / (1 + 4 * k^2)

theorem ellipse_and_max_area :
  -- Given conditions
  let A : ℝ × ℝ := (0, -2)
  let e : ℝ := Real.sqrt 3 / 2  -- eccentricity
  let slope_AF : ℝ := 2 * Real.sqrt 3 / 3

  -- Prove the following
  -- (1) The equation of E is x^2/4 + y^2 = 1
  (∀ x y : ℝ, ellipse x y ↔ x^2 / 4 + y^2 = 1) ∧
  -- (2) The maximum area occurs when k = ±√7/2
  (∀ k : ℝ, triangle_area k ≤ triangle_area (Real.sqrt 7 / 2) ∧
             triangle_area k ≤ triangle_area (-Real.sqrt 7 / 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_l462_46257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AC_polar_equation_l462_46284

/-- Given point A and circle C in polar coordinates, prove the equation of line AC -/
theorem line_AC_polar_equation (θ : Real) :
  let A : Real × Real := (2, π/4)
  let C_eq (ρ θ : Real) : Prop := ρ = 4 * Real.sqrt 2 * Real.sin θ
  let AC_eq (ρ θ : Real) : Prop := ρ = (2 * Real.sqrt 2) / (Real.cos θ + Real.sin θ)
  AC_eq 2 (π/4) ∧ ∀ ρ, C_eq ρ θ → AC_eq ρ θ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AC_polar_equation_l462_46284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_h_max_value_h_max_at_set_l462_46283

noncomputable section

/-- The function f(x) defined as cos(π/3 + x) * cos(π/3 - x) -/
def f (x : ℝ) : ℝ := Real.cos (Real.pi/3 + x) * Real.cos (Real.pi/3 - x)

/-- The function g(x) defined as (1/2) * sin(2x) - (1/4) -/
def g (x : ℝ) : ℝ := (1/2) * Real.sin (2*x) - (1/4)

/-- The function h(x) defined as f(x) - g(x) -/
def h (x : ℝ) : ℝ := f x - g x

/-- The smallest positive period of f(x) is π -/
theorem f_period : ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi :=
sorry

/-- The maximum value of h(x) is √2/2 -/
theorem h_max_value : ∃ (M : ℝ), (∀ (x : ℝ), h x ≤ M) ∧ (∃ (x : ℝ), h x = M) ∧ M = Real.sqrt 2 / 2 :=
sorry

/-- The set of x values for which h(x) attains its maximum value -/
def h_max_set : Set ℝ := {x | ∃ (k : ℤ), x = 3*Real.pi/8 + k*Real.pi}

/-- h(x) attains its maximum value at all points in h_max_set -/
theorem h_max_at_set : ∀ (x : ℝ), x ∈ h_max_set ↔ h x = Real.sqrt 2 / 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_h_max_value_h_max_at_set_l462_46283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_f_and_g_l462_46248

-- Define the domain for x
def valid_domain (x : ℝ) : Prop := (x > -1 ∧ x < 0) ∨ x > 0

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x + 1) / x

-- State the theorem
theorem product_of_f_and_g (x : ℝ) (h : valid_domain x) :
  f x * g x = 4 * Real.sqrt (x + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_f_and_g_l462_46248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l462_46264

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_inequality_solution_set (x : ℝ) :
  (floor x)^2 - 3*(floor x) - 10 ≤ 0 ↔ -2 ≤ x ∧ x < 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l462_46264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_lines_l462_46225

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 4

-- Define the centers of the circles
def center₁ : ℝ × ℝ := (-3, 1)
def center₂ : ℝ × ℝ := (4, 5)

-- Define the radius of both circles
def radius : ℝ := 2

-- Define a function to calculate the number of common tangent lines
def number_of_common_tangent_lines (C₁ C₂ : (ℝ → ℝ → Prop)) : ℕ := 4

-- Theorem statement
theorem common_tangent_lines :
  let d := Real.sqrt ((center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2)
  d > 2 * radius →
  (∃ (n : ℕ), n = 4 ∧ n = number_of_common_tangent_lines C₁ C₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_lines_l462_46225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l462_46246

-- Define the function f(x) = 3^x + 3x - 8
noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.log 3 * x) + 3*x - 8

-- State the theorem
theorem root_exists_in_interval :
  Continuous f →
  f 1 < 0 →
  f 1.5 > 0 →
  f 1.25 < 0 →
  ∃ x ∈ Set.Ioo 1.25 1.5, f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l462_46246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l462_46285

noncomputable def original_expr (x : ℝ) : ℝ :=
  ((x - 1) * Real.sqrt (x^2 - 6*x + 8)) / ((x - 2) * Real.sqrt (x^2 - 4*x + 3)) +
  ((x - 5) * Real.sqrt (x^2 - 4*x + 3)) / ((x - 3) * Real.sqrt (x^2 - 7*x + 10))

noncomputable def simplified_expr (x : ℝ) : ℝ :=
  (Real.sqrt (x - 1) * (Real.sqrt (x - 4) + Real.sqrt (x - 5))) / Real.sqrt (x^2 - 5*x + 6)

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 2) 
  (h2 : x ≠ 3) 
  (h3 : x^2 - 6*x + 8 ≥ 0) 
  (h4 : x^2 - 4*x + 3 ≥ 0) 
  (h5 : x^2 - 7*x + 10 ≥ 0) 
  (h6 : x - 1 ≥ 0) 
  (h7 : x - 4 ≥ 0) 
  (h8 : x - 5 ≥ 0) 
  (h9 : x^2 - 5*x + 6 > 0) :
  original_expr x = simplified_expr x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l462_46285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_range_on_interval_g_nonnegative_iff_a_range_l462_46241

noncomputable def f (x : ℝ) : ℝ := -2^x / (2^x + 1)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a/2 + f x

theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ := by sorry

theorem f_range_on_interval : ∀ x : ℝ, x ∈ Set.Icc 1 2 → f x ∈ Set.Icc (-4/5) (-2/3) := by sorry

theorem g_nonnegative_iff_a_range : 
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 2 → g a x ≥ 0) ↔ a ≥ 8/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_range_on_interval_g_nonnegative_iff_a_range_l462_46241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_g_l462_46252

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) / x^2

theorem monotonicity_of_g (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x > f a y) :
  (∀ x y : ℝ, 0 < x ∧ x < y → g a x < g a y) ∧
  (∀ x y : ℝ, x < y ∧ y < 0 → g a x > g a y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_g_l462_46252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l462_46276

noncomputable def f (x : ℝ) : ℝ := Real.log x + x + 1

theorem tangent_line_equation (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2) → 
  ∃ y₀ : ℝ, f x₀ = y₀ ∧ (λ x ↦ 2*x) = (λ x ↦ 2*(x - x₀) + y₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l462_46276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l462_46229

theorem equation_solution : ∃ x ∈ ({2, -2, 1, -1} : Set ℝ), 2*x - 3 = -1 ∧ x = 1 := by
  use 1
  constructor
  · simp [Set.mem_insert, Set.mem_singleton]
  · constructor
    · ring
    · rfl

#check equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l462_46229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_product_l462_46227

theorem simplify_fraction_product (x : ℝ) (h : x ≠ 0) :
  6 / (5 * x^4) * (5 * x^3) / 3 = 2 * x^7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_product_l462_46227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l462_46289

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 6).sum (λ k ↦ (Nat.choose 5 k) * (2^k) * ((-1)^(5-k)) * x^k) =
  -40 * x^2 + (Finset.range 6).sum (λ k ↦ if k ≠ 2 then (Nat.choose 5 k) * (2^k) * ((-1)^(5-k)) * x^k else 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l462_46289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_number_l462_46280

def is_valid_number (n : Nat) : Prop :=
  (n ≥ 10000 ∧ n < 100000) ∧
  (∃ a b c d e : Nat, 
    (a ∈ ({1, 3, 5, 7, 9} : Set Nat)) ∧ 
    (b ∈ ({1, 3, 5, 7, 9} : Set Nat)) ∧ 
    (c ∈ ({1, 3, 5, 7, 9} : Set Nat)) ∧ 
    (d ∈ ({1, 3, 5, 7, 9} : Set Nat)) ∧ 
    (e ∈ ({1, 3, 5, 7, 9} : Set Nat)) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e)

def is_divisible_by_digit (n : Nat) : Prop :=
  ∃ d : Nat, (d ∈ ({3, 5, 7, 9} : Set Nat)) ∧ n % d = 0

theorem smallest_valid_divisible_number :
  ∀ n : Nat, is_valid_number n ∧ is_divisible_by_digit n → n ≥ 13597 :=
by
  intro n
  intro h
  sorry

#check smallest_valid_divisible_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_number_l462_46280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l462_46250

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a*d) - Real.sqrt (b*c)| ≤ 1) : 
  (a*e + b/e) * (c*e + d/e) ≥ (a^2*f^2 - b^2/f^2) * (d^2/f^2 - c^2*f^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l462_46250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_theorem_l462_46202

/-- Represents a journey with three segments -/
structure Journey where
  total_distance : ℝ
  first_speed : ℝ
  second_speed : ℝ
  third_speed : ℝ
  total_time : ℝ

/-- Calculates the distance of the third segment given a journey -/
noncomputable def third_segment_distance (j : Journey) : ℝ :=
  (1/6) * j.total_distance

theorem journey_theorem (j : Journey) 
  (h1 : j.first_speed = 20)
  (h2 : j.second_speed = 10)
  (h3 : j.third_speed = 6)
  (h4 : j.total_time = 70/60)
  (h5 : (1/3 * j.total_distance / j.first_speed) + 
        (1/2 * 2/3 * j.total_distance / j.second_speed) + 
        (1/6 * j.total_distance / j.third_speed) = j.total_time) :
  ∃ ε > 0, |third_segment_distance j - 3.7| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_theorem_l462_46202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_age_sum_of_digits_l462_46270

def birthday_problem (joey_age chloe_age mike_age : ℕ) : Prop :=
  -- Joey is 2 years older than Chloe
  joey_age = chloe_age + 2 ∧
  -- Mike is exactly 2 years old today
  mike_age = 2 ∧
  -- Today is the second of 6 birthdays on which Chloe's age will be an integral multiple of Mike's age
  ∃ k : ℕ, chloe_age = k * mike_age ∧ k > 1 ∧ k < 6

theorem joey_age_sum_of_digits 
  (joey_age chloe_age mike_age : ℕ) 
  (h : birthday_problem joey_age chloe_age mike_age) :
  ∃ n : ℕ, 
    (joey_age + n) % (mike_age + n) = 0 ∧ 
    (Nat.digits 10 (joey_age + n)).sum = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_age_sum_of_digits_l462_46270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersects_y_axis_l462_46239

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

-- Define the point of tangency
noncomputable def point_of_tangency : ℝ × ℝ := (Real.exp 1, 2)

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2 / x

-- Define the slope of the tangent line at the point of tangency
noncomputable def tangent_slope : ℝ := f' (Real.exp 1)

-- Define the equation of the tangent line
noncomputable def tangent_line (x : ℝ) : ℝ := 
  tangent_slope * (x - point_of_tangency.1) + point_of_tangency.2

-- Theorem: The tangent line intersects the y-axis at (0, 0)
theorem tangent_intersects_y_axis : 
  tangent_line 0 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersects_y_axis_l462_46239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_height_ranges_l462_46290

open Real

-- Define the triangle
def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧ 
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- State the theorem
theorem triangle_angle_and_height_ranges 
  (a b c A B C : ℝ) (h : ℝ) 
  (h_triangle : triangle a b c A B C)
  (h_acute : A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2)
  (h_eq : a^2 - b^2 = b*c)
  (h_c : c = 4)
  (h_height : h = a * Real.sin B) :
  Real.pi/6 < B ∧ B < Real.pi/4 ∧ Real.sqrt 3 < h ∧ h < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_height_ranges_l462_46290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l462_46294

/-- The angle of inclination of a line with slope k -/
noncomputable def angle_of_inclination (k : ℝ) : ℝ := Real.arctan k

/-- A line y = kx - √3 intersects with x + y - 3 = 0 in the first quadrant -/
def intersection_in_first_quadrant (k : ℝ) : Prop :=
  let x := (3 + Real.sqrt 3) / (1 + k)
  let y := k * x - Real.sqrt 3
  0 < x ∧ 0 < y

theorem angle_of_inclination_range :
  ∀ k : ℝ, intersection_in_first_quadrant k →
    π / 6 < angle_of_inclination k ∧ angle_of_inclination k < π / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l462_46294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadratic_polynomial_root_l462_46251

theorem exists_quadratic_polynomial_root : ∃ (a b c : ℤ), 
  let f : ℝ → ℝ := λ x ↦ (a * x ^ 2 + b * x + c : ℝ)
  f (f (Real.sqrt 3)) = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadratic_polynomial_root_l462_46251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_f_existence_and_uniqueness_l462_46281

variable (ω v : ℂ)
variable (g : ℂ → ℂ)

-- ω is a cube root of 1 other than 1
axiom ω_cube_root : ω ^ 3 = 1 ∧ ω ≠ 1

-- Define the function f
noncomputable def f (z : ℂ) : ℂ := (1/2) * (g z - g (ω * z + v) + g (ω^2 * z + ω * v + v))

-- State the theorem
theorem unique_f_existence_and_uniqueness :
  (∀ z, f ω v g z + f ω v g (ω * z + v) = g z) ∧
  (∀ h : ℂ → ℂ, (∀ z, h z + h (ω * z + v) = g z) → h = f ω v g) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_f_existence_and_uniqueness_l462_46281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l462_46259

/-- Calculates the time (in seconds) it takes for a train to cross a pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

theorem train_crossing_pole_time :
  let train_length := (400 : ℝ)
  let train_speed := (120 : ℝ)
  let crossing_time := train_crossing_time train_length train_speed
  abs (crossing_time - 12) < 0.1 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_crossing_time 400 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l462_46259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_m_equals_one_tangent_lines_when_m_equals_one_l462_46224

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*m*x - 2*y + 8*m - 7 = 0

-- Define the area of the circle as a function of m
noncomputable def circle_area (m : ℝ) : ℝ :=
  Real.pi * ((2*m - 2)^2 + 4)

-- Statement for the minimum area
theorem min_area_m_equals_one :
  ∃ (m : ℝ), ∀ (m' : ℝ), circle_area m ≤ circle_area m' ∧ m = 1 := by
  sorry

-- Define the tangent line equations
def tangent_line_1 (x y : ℝ) : Prop :=
  3*x + 4*y = 0

def tangent_line_2 (x : ℝ) : Prop :=
  x = 4

-- Statement for the tangent lines
theorem tangent_lines_when_m_equals_one :
  ∀ (x y : ℝ),
    circle_equation x y 1 →
    (tangent_line_1 x y ∨ tangent_line_2 x) →
    ∃ (t : ℝ), x = 4 + t ∧ y = -3 - (3/4)*t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_m_equals_one_tangent_lines_when_m_equals_one_l462_46224
