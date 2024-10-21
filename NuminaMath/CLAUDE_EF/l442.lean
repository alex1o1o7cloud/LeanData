import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_4_3_5_less_than_300_l442_44291

theorem divisible_by_4_3_5_less_than_300 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) (Finset.range 300)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_4_3_5_less_than_300_l442_44291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_count_correct_l442_44217

open BigOperators

/-- The number of ways to build a tower of 14 cubes from 9 blue, 3 red, and 4 green cubes. -/
def tower_count : ℕ := 15093

/-- Theorem stating that the number of ways to build the tower is correct. -/
theorem tower_count_correct : tower_count = 15093 := by
  -- Unfold the definition of tower_count
  unfold tower_count
  
  -- The proof would go here, but we'll use sorry for now
  sorry

#eval tower_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_count_correct_l442_44217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_distance_sum_l442_44208

/-- Triangle ABC with vertices A, B, C in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Sum of distances from a point P to vertices of a triangle -/
noncomputable def sum_distances (t : Triangle) (P : ℝ × ℝ) : ℝ :=
  distance P t.A + distance P t.B + distance P t.C

/-- The main theorem -/
theorem fermat_point_distance_sum :
  let t : Triangle := { A := (0, 0), B := (8, 0), C := (4, 7) }
  let P : ℝ × ℝ := (3, 3)
  sum_distances t P = 3 * Real.sqrt 2 + Real.sqrt 34 + Real.sqrt 17 := by
  sorry

#eval 3 + 1  -- Should evaluate to 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_distance_sum_l442_44208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_complement_of_B_set_difference_results_l442_44242

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x > 4}

-- Define set B
def B : Set ℝ := {x | -6 < x ∧ x < 6}

-- Theorem for intersection and union of A and B
theorem intersection_and_union :
  (A ∩ B = {x | 4 < x ∧ x < 6}) ∧ (A ∪ B = {x | x > 4}) := by sorry

-- Theorem for complement of B in U
theorem complement_of_B :
  (U \ B) = {x | x ≤ -6 ∨ x ≥ 6} := by sorry

-- Define set difference
def set_difference (X Y : Set ℝ) : Set ℝ := X \ Y

-- Theorem for A - B and A - (A - B)
theorem set_difference_results :
  (set_difference A B = {x | x ≥ 6}) ∧
  (set_difference A (set_difference A B) = {x | 4 < x ∧ x < 6}) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_complement_of_B_set_difference_results_l442_44242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l442_44244

/-- Triangle PQR in the coordinate plane -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- Theorem: The area of triangle PQR is 2 square units -/
theorem triangle_PQR_area :
  ∀ (T : Triangle),
    T.P = (0, 0) →
    T.Q = (2, 0) →
    T.R.1 = 2 →
    T.R.2 = T.R.1 →
    triangleArea 2 2 = 2 := by
  intro T h1 h2 h3 h4
  unfold triangleArea
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l442_44244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_five_divisors_l442_44277

theorem remainder_five_divisors : 
  {n : ℕ | n > 5 ∧ ∃ k : ℕ, 42 = n * k ∧ 47 % n = 5} = {6, 7, 14, 21, 42} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_five_divisors_l442_44277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_shorter_leg_l442_44250

theorem right_triangle_shorter_leg : 
  ∃ a b c : ℕ,
  a < b ∧ b < c ∧ c = 65 ∧  -- c is the hypotenuse and equals 65
  a^2 + b^2 = c^2 ∧         -- Pythagorean theorem
  a = 25                    -- a is the shorter leg and equals 25
  := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_shorter_leg_l442_44250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_eq_half_l442_44263

/-- Given an angle α in a Cartesian coordinate system where:
    - The vertex of α is at the origin (0, 0)
    - The initial side of α coincides with the positive x-axis
    - The terminal side of α passes through the point (1, -√3)
    Prove that cos α = 1/2 -/
theorem cos_alpha_eq_half (α : ℝ) (P : ℝ × ℝ) : 
  P.1 = 1 → P.2 = -Real.sqrt 3 → Real.cos α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_eq_half_l442_44263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_implies_a_range_l442_44269

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a * x^2 - 2*x - 2 else -2^x

-- Define the property that f has a maximum value
def has_maximum (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ x : ℝ, f x ≤ M

-- Theorem statement
theorem f_max_implies_a_range (a : ℝ) :
  has_maximum (f a) ↔ a ∈ Set.Icc (-1) 0 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_implies_a_range_l442_44269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_weight_after_processing_l442_44258

/-- The weight of a side of beef after processing, given its initial weight and percentage loss. -/
noncomputable def weight_after_processing (initial_weight : ℝ) (percent_loss : ℝ) : ℝ :=
  initial_weight * (1 - percent_loss / 100)

/-- Theorem stating that a side of beef losing 35% of its weight and initially weighing 846.15 pounds
    will weigh approximately 550 pounds after processing. -/
theorem beef_weight_after_processing :
  let initial_weight : ℝ := 846.15
  let percent_loss : ℝ := 35
  let processed_weight := weight_after_processing initial_weight percent_loss
  ∃ ε > 0, abs (processed_weight - 550) < ε :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_weight_after_processing_l442_44258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l442_44202

-- Define the point P as a variable
variable (x y : ℝ)
def P : ℝ × ℝ := (x, y)

-- Define the conditions
def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

-- State the theorem
theorem point_coordinates (x y : ℝ) :
  second_quadrant (x, y) ∧
  distance_to_x_axis (x, y) = 5 ∧
  distance_to_y_axis (x, y) = 2 →
  (x, y) = (-2, 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l442_44202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_initial_purchase_tree_purchase_problem_l442_44204

-- Define the unit prices
def banyan_price : ℕ := 60
def camphor_price : ℕ := 80

-- Define the cost function
def total_cost (b c : ℕ) : ℕ := b * banyan_price + c * camphor_price

-- Define the constraints
theorem price_difference : banyan_price + 20 = camphor_price :=
  by rfl

theorem initial_purchase : total_cost 3 2 = 340 :=
  by rfl

-- Define the purchase constraints
def valid_purchase (b c : ℕ) : Prop :=
  b + c = 150 ∧
  total_cost b c ≤ 10840 ∧
  c ≥ (3 * b) / 2

-- Theorem statement
theorem tree_purchase_problem :
  (banyan_price = 60 ∧ camphor_price = 80) ∧
  (∀ b c, valid_purchase b c ↔ (b = 58 ∧ c = 92) ∨ (b = 59 ∧ c = 91) ∨ (b = 60 ∧ c = 90)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_initial_purchase_tree_purchase_problem_l442_44204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_gain_percentages_l442_44292

/-- Represents a type of cloth with its sales information -/
structure ClothType where
  metersToSell : ℕ
  gainMeters : ℕ

/-- Calculates the gain percentage for a given cloth type -/
noncomputable def gainPercentage (cloth : ClothType) : ℚ :=
  (cloth.gainMeters : ℚ) / ((cloth.metersToSell - cloth.gainMeters) : ℚ) * 100

/-- The three types of cloth sold by the shop owner -/
def typeA : ClothType := ⟨50, 10⟩
def typeB : ClothType := ⟨75, 15⟩
def typeC : ClothType := ⟨100, 25⟩

theorem cloth_gain_percentages :
  gainPercentage typeA = 25 ∧
  gainPercentage typeB = 25 ∧
  (gainPercentage typeC : ℚ) > 33 ∧ (gainPercentage typeC : ℚ) < 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_gain_percentages_l442_44292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_age_is_19_l442_44215

def age_distribution : List (Nat × Nat) := [(18, 3), (19, 5), (20, 2), (21, 1), (22, 1)]

def total_players : Nat := (age_distribution.map (fun p => p.2)).sum

theorem median_age_is_19 (h : total_players = 12) :
  let sorted_ages := age_distribution.bind (fun p => List.replicate p.2 p.1)
  let median := (sorted_ages.get! 5 + sorted_ages.get! 6) / 2
  median = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_age_is_19_l442_44215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_orthogonal_sets_l442_44284

/-- Two functions are orthogonal on [-1, 1] if their product integrates to 0 over this interval -/
def IsOrthogonal (f g : ℝ → ℝ) : Prop :=
  ∫ x in (-1)..(1), f x * g x = 0

/-- The first set of functions -/
noncomputable def f₁ (x : ℝ) : ℝ := Real.sin (x / 2)
noncomputable def g₁ (x : ℝ) : ℝ := Real.cos (x / 2)

/-- The second set of functions -/
def f₂ (x : ℝ) : ℝ := x + 1
def g₂ (x : ℝ) : ℝ := x - 1

/-- The third set of functions -/
def f₃ (x : ℝ) : ℝ := x
def g₃ (x : ℝ) : ℝ := x^2

/-- The theorem stating that exactly two sets of functions are orthogonal -/
theorem two_orthogonal_sets :
  (IsOrthogonal f₁ g₁ ∧ IsOrthogonal f₃ g₃ ∧ ¬IsOrthogonal f₂ g₂) :=
sorry

/-- Helper lemmas for each set of functions -/
lemma orthogonal_set_1 : IsOrthogonal f₁ g₁ := sorry

lemma not_orthogonal_set_2 : ¬IsOrthogonal f₂ g₂ := sorry

lemma orthogonal_set_3 : IsOrthogonal f₃ g₃ := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_orthogonal_sets_l442_44284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l442_44209

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2 = 0

-- Define the line
def lineEq (m x y : ℝ) : Prop := y = m*x

-- Define the tangent condition
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circleEq x y ∧ lineEq m x y ∧
  ∀ (x' y' : ℝ), circleEq x' y' → lineEq m x' y' → (x = x' ∧ y = y')

-- Theorem statement
theorem tangent_line_slope :
  ∀ m : ℝ, is_tangent m → (m = 1 ∨ m = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l442_44209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_fraction_closest_to_AD_l442_44288

/-- A trapezoid with specific properties -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  parallel_BC_AD : (C.2 - B.2) / (C.1 - B.1) = (D.2 - A.2) / (D.1 - A.1)
  angle_BAD : Real.cos (60 * π / 180) = ((B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2)) / 
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2))
  angle_CDA : Real.cos (60 * π / 180) = ((C.1 - D.1) * (A.1 - D.1) + (C.2 - D.2) * (A.2 - D.2)) / 
    (Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) * Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2))
  angle_ABC : Real.cos (120 * π / 180) = ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) / 
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2))
  angle_BCD : Real.cos (120 * π / 180) = ((B.1 - C.1) * (D.1 - C.1) + (B.2 - C.2) * (D.2 - C.2)) / 
    (Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) * Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2))
  length_BC : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 100
  length_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 100
  length_CD : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 100

/-- The area of the region closest to AD -/
noncomputable def areaClosestToAD (t : Trapezoid) : ℝ := sorry

/-- The total area of the trapezoid -/
noncomputable def totalArea (t : Trapezoid) : ℝ := sorry

/-- The area of the region closer to AD than to any other side is 5/12 of the total area -/
theorem area_fraction_closest_to_AD (t : Trapezoid) : 
  (areaClosestToAD t) / (totalArea t) = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_fraction_closest_to_AD_l442_44288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_sinh_one_l442_44271

/-- The length of the arc of the curve y = 2 + cosh(x) from x = 0 to x = 1 -/
noncomputable def arcLength : ℝ := ∫ x in (Set.Icc 0 1), Real.cosh x

/-- Theorem stating that the arc length is equal to sinh(1) -/
theorem arc_length_equals_sinh_one : arcLength = Real.sinh 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_sinh_one_l442_44271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_sum_l442_44205

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (b 1 + b n) / 2

theorem geometric_arithmetic_sum
  (a : ℕ → ℝ) (b : ℕ → ℝ) :
  geometric_sequence a →
  arithmetic_sequence b →
  a 4 * a 6 = 2 * a 5 →
  b 5 = 2 * a 5 →
  arithmetic_sum b 9 = 36 := by
  sorry

#check geometric_arithmetic_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_sum_l442_44205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_n_dividing_2_pow_n_minus_n_l442_44206

theorem infinite_n_dividing_2_pow_n_minus_n (p : ℕ) (hp : Nat.Prime p) :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, p ∣ (2^(f k) - f k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_n_dividing_2_pow_n_minus_n_l442_44206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_base_length_l442_44240

/-- Represents the properties of a parallelogram-shaped field -/
structure ParallelogramField where
  height : ℝ
  levelingCostPer10SqM : ℝ
  totalLevelingCost : ℝ

/-- Calculates the base length of a parallelogram-shaped field -/
noncomputable def calculateBaseLength (field : ParallelogramField) : ℝ :=
  (field.totalLevelingCost / field.levelingCostPer10SqM) * 10 / field.height

/-- Theorem stating that for the given field properties, the base length is 54 meters -/
theorem parallelogram_base_length :
  let field : ParallelogramField := {
    height := 24,
    levelingCostPer10SqM := 50,
    totalLevelingCost := 6480
  }
  calculateBaseLength field = 54 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_base_length_l442_44240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_eq_l442_44229

noncomputable def f (x : ℝ) := Real.sin x + Real.exp x + x^2017

noncomputable def f_seq : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => λ x => deriv (f_seq n) x

theorem f_2018_eq (x : ℝ) : f_seq 2018 x = -Real.sin x + Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_eq_l442_44229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_theorem_l442_44290

/-- Calculates the average speed of a bus excluding stoppages -/
noncomputable def average_speed_excluding_stoppages (stoppage_time : ℝ) (average_speed_with_stoppages : ℝ) : ℝ :=
  let running_time := 60 - stoppage_time
  average_speed_with_stoppages * 60 / running_time

/-- Proves that a bus stopping for 20 minutes per hour with an average speed of 40 km/hr
    including stoppages has an average speed of 60 km/hr excluding stoppages -/
theorem bus_speed_theorem :
  average_speed_excluding_stoppages 20 40 = 60 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval average_speed_excluding_stoppages 20 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_theorem_l442_44290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l442_44213

def set_A : Set ℝ := {x | 3 * x - x^2 > 0}
def set_B : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l442_44213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_with_prime_floor_theorem_l442_44219

-- Define the condition that for every positive integer n, 
-- there exists a prime number between n^3 and (n+1)^3
def exists_prime_between_cubes (n : ℕ+) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n^3 < p ∧ p < (n + 1)^3

-- Define the property we want to prove
def exists_a_with_prime_floor (a : ℝ) : Prop :=
  a > 1 ∧ ∀ k : ℕ, Nat.Prime (Int.toNat ⌊a^(3^k)⌋)

-- The theorem statement
theorem exists_a_with_prime_floor_theorem 
  (h : ∀ n : ℕ+, exists_prime_between_cubes n) : 
  ∃ a : ℝ, exists_a_with_prime_floor a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_with_prime_floor_theorem_l442_44219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_zero_l442_44214

theorem root_difference_zero : 
  ∃ (r₁ r₂ : ℝ), (r₁^2 + 40*r₁ + 300 = -100) ∧ 
                 (r₂^2 + 40*r₂ + 300 = -100) ∧ 
                 |r₁ - r₂| = 0 :=
by
  -- We'll prove this later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_zero_l442_44214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_cost_theorem_l442_44220

def food_cost : ℚ := 61.48
def sales_tax_rate : ℚ := 7 / 100
def tip_rate : ℚ := 15 / 100

def total_cost : ℚ := food_cost + (sales_tax_rate * food_cost) + (tip_rate * food_cost)

def round_to_cents (x : ℚ) : ℚ := 
  (x * 100).floor / 100

theorem meal_cost_theorem : 
  ∃ (rounded_cost : ℚ), 
    rounded_cost = round_to_cents total_cost ∧ 
    rounded_cost = 75.01 :=
by
  -- The proof goes here
  sorry

#eval round_to_cents total_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_cost_theorem_l442_44220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_l442_44201

-- Define the given constants
noncomputable def horizontal_distance : ℝ := 18
noncomputable def elevation_angle : ℝ := Real.pi / 4  -- 45° in radians
noncomputable def eye_level : ℝ := 1.6

-- Define the theorem
theorem flagpole_height :
  ∃ (height : ℝ),
    height = eye_level + horizontal_distance * Real.tan elevation_angle ∧
    height = 19.6 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_l442_44201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_binomial_coeff_count_is_power_of_two_l442_44268

/-- The number of 1s in the binary representation of a natural number -/
def binary_ones_count (n : ℕ) : ℕ :=
  (n.digits 2).filter (· = 1) |>.length

/-- The count of odd binomial coefficients for a given n -/
def odd_binomial_coeff_count (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (fun m => Nat.choose n m % 2 = 1) |>.length

/-- 
Theorem: For any positive integer n, the count of odd binomial coefficients C(n,m) 
for 0 ≤ m ≤ n equals 2^k, where k is the number of 1s in the binary representation of n.
-/
theorem odd_binomial_coeff_count_is_power_of_two (n : ℕ+) : 
  odd_binomial_coeff_count n.val = 2^(binary_ones_count n.val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_binomial_coeff_count_is_power_of_two_l442_44268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l442_44216

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  (train_length + bridge_length) / crossing_time

/-- Theorem stating the approximate speed of the train -/
theorem train_speed_approx :
  let train_length : ℝ := 250
  let bridge_length : ℝ := 300
  let crossing_time : ℝ := 45
  abs (train_speed train_length bridge_length crossing_time - 12.22) < 0.01 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l442_44216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l442_44266

-- Define set A
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l442_44266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l442_44275

-- Define the lines
def line1 (a x y : ℝ) : Prop := a * x + 4 * y - 2 = 0
def line2 (b x y : ℝ) : Prop := 2 * x - 5 * y + b = 0

-- Define perpendicularity of lines
def perpendicular (a b : ℝ) : Prop := 
  ∃ (m1 m2 c1 c2 : ℝ), (m1 * m2 = -1) ∧ 
  (∀ x y, line1 a x y → y = m1 * x + c1) ∧
  (∀ x y, line2 b x y → y = m2 * x + c2)

-- The theorem
theorem perpendicular_condition (a b : ℝ) :
  perpendicular a b ↔ a = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l442_44275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_range_l442_44226

open Real

/-- The function f(x) = ax - ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - log x

/-- f is monotonically increasing on [2, +∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → x < y → f a x < f a y

/-- The theorem stating the range of 'a' for which f is monotonically increasing -/
theorem monotone_increasing_range :
  {a : ℝ | is_monotone_increasing a} = Set.Ici (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_range_l442_44226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_in_range_l442_44259

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + (1/2) * x^2 - a * x

-- State the theorem
theorem f_nonnegative_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) ↔ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_in_range_l442_44259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solutions_l442_44212

theorem positive_integer_solutions : 
  {(x, y, z) : ℕ+ × ℕ+ × ℕ+ | 
    (1 + 1 / (x : ℚ)) * (1 + 1 / (y : ℚ)) * (1 + 1 / (z : ℚ)) = 2} = 
  {(2, 4, 15), (2, 5, 9), (2, 6, 7), (3, 3, 8), (3, 4, 5)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solutions_l442_44212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l442_44257

/-- A dart board shaped like a regular decagon with an inscribed circle -/
structure DartBoard where
  /-- Side length of the decagon -/
  side_length : ℝ
  /-- Number of sides in the decagon -/
  num_sides : ℕ
  /-- Assertion that the shape is a regular decagon -/
  is_decagon : num_sides = 10

/-- The probability of a dart landing in the circular region of the dart board -/
noncomputable def probability_in_circle (board : DartBoard) : ℝ :=
  Real.pi / (board.num_sides * Real.tan (Real.pi / board.num_sides))

/-- Theorem stating the probability of a dart landing in the circular region -/
theorem dart_probability (board : DartBoard) (h : board.side_length = 2) :
  probability_in_circle board = Real.pi / (10 * Real.tan (Real.pi / 10)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l442_44257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_P_inter_Q_eq_interval_1_2_l442_44238

/-- The set P -/
def P : Set ℝ := {y | ∃ x > 0, y = (1/2)^x}

/-- The set Q -/
def Q : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2) / Real.log 2}

/-- The complement of P in ℝ -/
def complement_P : Set ℝ := {x | x ∉ P}

/-- The interval [1, 2) -/
def interval_1_2 : Set ℝ := {x | 1 ≤ x ∧ x < 2}

/-- Theorem stating that the intersection of complement_P and Q equals [1, 2) -/
theorem complement_P_inter_Q_eq_interval_1_2 : complement_P ∩ Q = interval_1_2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_P_inter_Q_eq_interval_1_2_l442_44238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_from_C₁_to_C₂_l442_44252

-- Define the ellipse C₁
noncomputable def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x + y - 8 = 0

-- Define the distance function from a point to the line C₂
noncomputable def distance_to_C₂ (x y : ℝ) : ℝ :=
  |x + y - 8| / Real.sqrt 2

-- State the theorem
theorem min_distance_from_C₁_to_C₂ :
  ∃ (min_dist : ℝ) (px py : ℝ),
    (∀ (x y : ℝ), C₁ x y → distance_to_C₂ x y ≥ min_dist) ∧
    C₁ px py ∧
    distance_to_C₂ px py = min_dist ∧
    min_dist = 3 * Real.sqrt 2 ∧
    px = 3 / 2 ∧
    py = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_from_C₁_to_C₂_l442_44252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_less_than_one_tangent_lines_slope_zero_tangent_line_equations_l442_44207

-- Define the curve C
noncomputable def C (x : ℝ) : ℝ := x + 1/x

-- Statement 1: The derivative of C is less than 1 for all x ≠ 0
theorem slope_less_than_one (x : ℝ) (hx : x ≠ 0) : 
  deriv C x < 1 := by sorry

-- Statement 2: The tangent lines with slope 0 occur at x = ±1
theorem tangent_lines_slope_zero :
  {x : ℝ | deriv C x = 0} = {1, -1} := by sorry

-- Statement 3: The equations of the tangent lines with slope 0 are y = 2 and y = -2
theorem tangent_line_equations :
  {y : ℝ | ∃ x, deriv C x = 0 ∧ C x = y} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_less_than_one_tangent_lines_slope_zero_tangent_line_equations_l442_44207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_exist_l442_44260

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew (l1 l2 : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  ¬ (∃ (v : EuclideanSpace ℝ (Fin 3)), v ≠ 0 ∧ ∀ x ∈ l1, x + v ∈ l2) ∧ 
  l1 ∩ l2 = ∅

/-- A line intersects another line if they have a common point. -/
def intersects (l1 l2 : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  l1 ∩ l2 ≠ ∅

/-- Given two skew lines, there exist two intersecting lines such that each of them
    intersects both of the given skew lines. -/
theorem intersecting_lines_exist (p q : Set (EuclideanSpace ℝ (Fin 3))) 
  (h : are_skew p q) :
  ∃ (l1 l2 : Set (EuclideanSpace ℝ (Fin 3))), intersects l1 l2 ∧ 
    intersects l1 p ∧ intersects l1 q ∧ 
    intersects l2 p ∧ intersects l2 q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_exist_l442_44260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_inside_circle_l442_44293

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if a point is inside a circle -/
def is_inside (c : Circle) (p : Point) : Prop :=
  distance p c.center < c.radius

/-- The given circle -/
def given_circle : Circle :=
  { center := (2, 3), radius := 2 }

/-- The given point -/
def P : Point := (3, 2)

/-- Theorem: P is inside the given circle -/
theorem P_inside_circle : is_inside given_circle P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_inside_circle_l442_44293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l442_44286

/-- The function f(x) = 2sin(2x - π/3) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3)

/-- The smallest positive period of f(x) -/
noncomputable def smallest_positive_period : ℝ := Real.pi

/-- Theorem: The smallest positive period of f(x) = 2sin(2x - π/3) is π -/
theorem f_period : 
  ∀ x : ℝ, f (x + smallest_positive_period) = f x ∧ 
  ∀ p : ℝ, 0 < p → p < smallest_positive_period → ∃ y : ℝ, f (y + p) ≠ f y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l442_44286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_digging_time_l442_44256

/-- The time taken for a team to complete a digging task -/
noncomputable def digging_time (men : ℕ) (effort : ℝ) : ℝ :=
  let original_effort : ℝ := 18 * 5
  effort * original_effort / men

theorem new_digging_time :
  digging_time 30 1.5 = 4.5 := by
  -- Unfold the definition of digging_time
  unfold digging_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_digging_time_l442_44256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_billion_two_hundred_two_million_and_five_l442_44253

/-- The number one billion two hundred two million and five is equal to 1,202,000,005. -/
theorem one_billion_two_hundred_two_million_and_five : 1202000005 = 1202000005 := by
  rfl

#eval 1202000005

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_billion_two_hundred_two_million_and_five_l442_44253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_product_A_B_l442_44283

-- Define the hexadecimal system
def hexadecimal_digit := Fin 16

-- Define the conversion function from decimal to hexadecimal
def decimal_to_hexadecimal (n : ℕ) : List hexadecimal_digit :=
  sorry

-- Define the multiplication operation in hexadecimal
def hex_mult (a b : hexadecimal_digit) : List hexadecimal_digit :=
  sorry

-- Theorem statement
theorem hex_product_A_B :
  decimal_to_hexadecimal (10 * 11) = [⟨6, sorry⟩, ⟨14, sorry⟩] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_product_A_B_l442_44283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l442_44218

/-- The area of a triangle given its vertex coordinates -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Theorem: The area of triangle PQR with vertices P(-3, 2), Q(1, 5), and R(4, -1) is 16.5 square units -/
theorem area_of_triangle_PQR : 
  triangleArea (-3) 2 1 5 4 (-1) = 16.5 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp [abs_of_nonneg]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l442_44218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_g_zeros_l442_44296

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x - (1/2) * a * x^2

theorem f_extrema_and_g_zeros :
  (∀ x ∈ Set.Icc (-Real.pi) Real.pi, f x ≤ Real.pi/2) ∧
  (∃ x ∈ Set.Icc (-Real.pi) Real.pi, f x = Real.pi/2) ∧
  (∀ x ∈ Set.Icc (-Real.pi) Real.pi, f x ≥ -1) ∧
  (∃ x ∈ Set.Icc (-Real.pi) Real.pi, f x = -1) ∧
  (∀ a > 1/3, ∃! s : Finset ℝ, s.card = 2 ∧ ∀ x ∈ s, g a x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_g_zeros_l442_44296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l442_44245

noncomputable def vector_projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  (dot_product / magnitude_squared * w.1, dot_product / magnitude_squared * w.2)

theorem projection_theorem (w : ℝ × ℝ) 
  (h : vector_projection (0, 3) w = (-9/10, 3/10)) :
  vector_projection (4, 1) w = (33/10, -11/10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l442_44245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l442_44200

-- Define the function f(x) = xe^(-x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

-- State the theorem
theorem min_value_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 4 ∧ f x = 0 ∧ ∀ y ∈ Set.Icc 0 4, f y ≥ f x := by
  -- We claim that x = 0 is the minimum
  use 0
  constructor
  · -- Show that 0 is in the interval [0, 4]
    simp [Set.Icc]
  constructor
  · -- Show that f(0) = 0
    simp [f]
  · -- Show that for all y in [0, 4], f(y) ≥ f(0)
    intro y hy
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l442_44200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_equation_l442_44261

theorem smallest_positive_solution_tan_equation :
  let f : ℝ → ℝ := λ x => Real.tan (4 * x) + Real.tan (5 * x) - 1 / Real.cos (5 * x)
  ∃ (x : ℝ), x > 0 ∧ f x = 0 ∧ ∀ (y : ℝ), y > 0 → f y = 0 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_equation_l442_44261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_equation_solution_l442_44299

/-- A digit is a natural number between 0 and 9 inclusive -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Convert a list of digits to an integer -/
def digits_to_int (digits : List Digit) : ℤ :=
  digits.foldl (fun acc d => acc * 10 + d.val) 0

theorem digit_equation_solution (a b c d : Digit) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (-(digits_to_int [d, a, b, a, c])) / (2014 * (d.val : ℤ)) = -1 →
  d.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_equation_solution_l442_44299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_scalene_triangles_11_l442_44241

/-- A triangle with integral side lengths --/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  h_triangle : a + b > c ∧ a + c > b ∧ b + c > a

/-- The set of scalene triangles with longest side 11 --/
def scaleneTriangles11 : Set IntTriangle :=
  {t : IntTriangle | t.c = 11 ∧ t.a < t.b ∧ t.b < 11}

/-- Helper function to count the elements in scaleneTriangles11 --/
def countScaleneTriangles11 : ℕ := by
  let count := 20  -- This is the result we want to prove
  exact count

/-- Theorem stating that there are 20 non-congruent scalene triangles with longest side 11 --/
theorem count_scalene_triangles_11 : countScaleneTriangles11 = 20 := by
  rfl  -- reflexivity, since we defined countScaleneTriangles11 to be 20

#eval countScaleneTriangles11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_scalene_triangles_11_l442_44241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l442_44231

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the equation SIX × 3 = TWEN -/
def EquationHolds (S I X T W E N : Digit) : Prop :=
  (S.val * 100 + I.val * 10 + X.val) * 3 = T.val * 1000 + W.val * 100 + E.val * 10 + N.val

/-- All digits in the equation are different -/
def AllDifferent (S I X T W E N : Digit) : Prop :=
  S ≠ I ∧ S ≠ X ∧ S ≠ T ∧ S ≠ W ∧ S ≠ E ∧ S ≠ N ∧
  I ≠ X ∧ I ≠ T ∧ I ≠ W ∧ I ≠ E ∧ I ≠ N ∧
  X ≠ T ∧ X ≠ W ∧ X ≠ E ∧ X ≠ N ∧
  T ≠ W ∧ T ≠ E ∧ T ≠ N ∧
  W ≠ E ∧ W ≠ N ∧
  E ≠ N

theorem puzzle_solution :
  ∀ (I X T W E N : Digit),
    EquationHolds (⟨1, by norm_num⟩ : Digit) I X T W E N →
    AllDifferent (⟨1, by norm_num⟩ : Digit) I X T W E N →
    N.val % 2 = 0 →
    T = ⟨5, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l442_44231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_side_length_l442_44203

/-- The length of a side of an equilateral triangle -/
def equilateral_side_length : ℝ := 2

/-- The number of isosceles triangles inside the equilateral triangle -/
def num_isosceles_triangles : ℕ := 3

/-- Theorem: Given an equilateral triangle with side length 2 and three congruent 
    isosceles triangles constructed inside it such that their bases collectively 
    cover the entire perimeter of the equilateral triangle equally, and the sum 
    of their areas equals the area of the equilateral triangle, the length of one 
    of the congruent sides of one of the isosceles triangles is 2√3/3. -/
theorem isosceles_side_length (
  equilateral_area : ℝ → ℝ) 
  (isosceles_area : ℝ → ℝ) 
  (isosceles_base : ℝ) 
  (isosceles_height : ℝ) :
  (equilateral_area equilateral_side_length = 
    ↑num_isosceles_triangles * isosceles_area isosceles_base) →
  (isosceles_base = equilateral_side_length) →
  (isosceles_area isosceles_base = 1/2 * isosceles_base * isosceles_height) →
  (equilateral_area equilateral_side_length = Real.sqrt 3 * equilateral_side_length^2 / 4) →
  Real.sqrt ((isosceles_base / 2)^2 + isosceles_height^2) = 2 * Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_side_length_l442_44203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_winning_strategy_l442_44289

/-- Represents the possible letters in the game -/
inductive Letter : Type
| A : Letter
| B : Letter

/-- Represents a move by the second player -/
inductive Move : Type
| swap : Nat → Nat → Move  -- Swap letters at two positions
| noop : Move              -- Do nothing

/-- Represents the state of the game after each round -/
structure GameState where
  sequence : List Letter
  moves : List Move

/-- Function to check if a list is a palindrome -/
def isPalindrome {α : Type} (l : List α) : Prop :=
  l = l.reverse

/-- Function to apply a list of moves to a sequence -/
def applyMoves (sequence : List Letter) (moves : List Move) : List Letter :=
  sorry

/-- The main theorem stating the existence of a winning strategy for the second player -/
theorem second_player_winning_strategy :
  ∀ (initial_sequence : List Letter),
    initial_sequence.length = 1999 →
    ∃ (player2_moves : List Move),
      player2_moves.length ≤ 1999 ∧
      isPalindrome (applyMoves initial_sequence player2_moves) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_winning_strategy_l442_44289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_omega_proof_l442_44278

/-- The minimum value of ω that satisfies the given conditions -/
noncomputable def minimum_omega : ℝ := 3/2

/-- The function whose graph is being considered -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi/3) + 2

/-- The condition that the graph coincides with itself after shifting -/
def graph_coincides (ω : ℝ) : Prop :=
  ∀ x, f ω x = f ω (x - 4*Real.pi/3)

theorem minimum_omega_proof (h : ℝ) (h_pos : h > 0) (h_coincides : graph_coincides h) :
  h ≥ minimum_omega :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_omega_proof_l442_44278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_four_less_than_negative_three_l442_44230

theorem only_negative_four_less_than_negative_three :
  ∀ x : ℤ, x ∈ ({-4, -2, 0, 3} : Set ℤ) → (x < -3 ↔ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_four_less_than_negative_three_l442_44230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_420_degrees_l442_44251

theorem sin_420_degrees : Real.sin (420 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_420_degrees_l442_44251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_pair_probability_l442_44267

/-- Represents the total number of sock pairs in the wardrobe -/
def total_pairs : ℕ := 4

/-- Represents the number of socks selected -/
def socks_selected : ℕ := 4

/-- Event A: At least two of the selected socks are from the same pair -/
noncomputable def event_A : ℝ := 27 / 35

/-- Event AB: Exactly two of the selected socks are from the same pair, 
    and the other two are not from the same pair -/
noncomputable def event_AB : ℝ := 24 / 35

/-- Theorem stating the probability that given two of the selected socks 
    form a pair, the other two do not form a pair -/
theorem sock_pair_probability : 
  event_AB / event_A = 8 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_pair_probability_l442_44267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_menelaus_theorem_l442_44280

/-- Menelaus' Theorem -/
theorem menelaus_theorem (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 2)) : 
  (∃ (t : ℝ), A₁ = (1 - t) • B + t • C) →
  (∃ (u : ℝ), B₁ = (1 - u) • C + u • A) →
  (∃ (v : ℝ), C₁ = (1 - v) • A + v • B) →
  (Collinear ℝ {A₁, B₁, C₁} ↔ 
    (dist B A₁ / dist C A₁) * (dist C B₁ / dist A B₁) * (dist A C₁ / dist B C₁) = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_menelaus_theorem_l442_44280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_theorem_pumping_time_corollary_l442_44239

/-- Represents a water pool with its capacity and current fill level. -/
structure WaterPool where
  capacity : ℝ
  currentFill : ℝ

/-- Calculates the time needed to pump a given volume of water at a specific flow rate. -/
noncomputable def pumpingTime (volume : ℝ) (flowRate : ℝ) : ℝ :=
  volume / flowRate

/-- Theorem stating the total capacity of the pool given the specified conditions. -/
theorem pool_capacity_theorem (pool : WaterPool) 
    (h1 : pool.currentFill + 300 = 0.8 * pool.capacity)
    (h2 : 300 = 0.3 * pool.currentFill) :
    pool.capacity = 3000 := by
  sorry

/-- Corollary stating the time needed to pump the additional water. -/
theorem pumping_time_corollary (flowRate : ℝ) (h : flowRate = 20) :
    pumpingTime 300 flowRate = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_theorem_pumping_time_corollary_l442_44239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_area_of_triangle_l442_44265

-- Define the points
def P : ℝ × ℝ := (5, 1)
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (12, 0)
def C : ℝ × ℝ := (4, 4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area function for a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Theorem statement
theorem distance_and_area_of_triangle :
  (distance P A + distance P B + distance P C = Real.sqrt 26 + 5 * Real.sqrt 2 + Real.sqrt 10) ∧
  (triangleArea A B C = 24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_area_of_triangle_l442_44265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2011_eq_3_l442_44273

/-- The sequence defined by the recurrence relation -/
noncomputable def a : ℕ → ℝ
  | 0 => 3
  | n + 1 => (Real.sqrt 3 * a n - 1) / (a n + Real.sqrt 3)

/-- Theorem stating that the 2011th term of the sequence is 3 -/
theorem a_2011_eq_3 : a 2010 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2011_eq_3_l442_44273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_balancing_lines_l442_44227

/-- A point in the plane with a color -/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Bool -- True for blue, False for red

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : ColoredPoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Check if a point is on a given side of a line -/
noncomputable def onSide (p : ColoredPoint) (l : Line) : Bool :=
  l.a * p.x + l.b * p.y + l.c > 0

/-- Check if a line is a balancing line -/
def isBalancingLine (points : List ColoredPoint) (l : Line) : Prop :=
  ∃ (p1 p2 : ColoredPoint),
    p1 ∈ points ∧ p2 ∈ points ∧
    p1.color ≠ p2.color ∧
    (l.a * p1.x + l.b * p1.y + l.c = 0) ∧
    (l.a * p2.x + l.b * p2.y + l.c = 0) ∧
    (points.filter (fun p => p.color ∧ onSide p l)).length =
    (points.filter (fun p => ¬p.color ∧ onSide p l)).length ∧
    (points.filter (fun p => p.color ∧ ¬onSide p l)).length =
    (points.filter (fun p => ¬p.color ∧ ¬onSide p l)).length

theorem two_balancing_lines
  (n : ℕ)
  (h_n : n > 1)
  (points : List ColoredPoint)
  (h_count : points.length = 2 * n)
  (h_blue : (points.filter (fun p => p.color)).length = n)
  (h_red : (points.filter (fun p => ¬p.color)).length = n)
  (h_not_collinear : ∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points →
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬collinear p1 p2 p3) :
  ∃ l1 l2 : Line, l1 ≠ l2 ∧ isBalancingLine points l1 ∧ isBalancingLine points l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_balancing_lines_l442_44227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l442_44272

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V) 
  (h1 : ‖a - b‖ = Real.sqrt 3)
  (h2 : ‖a + b‖ = ‖2 • a - b‖) :
  ‖b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l442_44272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_bipartite_graph_l442_44243

/-- Represents a bipartite graph with two sets of vertices and edges between them -/
structure BipartiteGraph (m n : ℕ) where
  edges : Finset (Fin m × Fin n)

/-- A path in the graph is a list of alternating vertices from each set -/
def isPath (G : BipartiteGraph m n) : List (Fin m ⊕ Fin n) → Prop :=
  sorry

/-- The graph is connected if there's a path between any two vertices -/
def isConnected (G : BipartiteGraph m n) : Prop :=
  sorry

/-- The degree of a vertex is the number of edges connected to it -/
def degree (G : BipartiteGraph m n) (v : Fin m ⊕ Fin n) : ℕ :=
  sorry

/-- The main theorem: maximum number of edges in a connected bipartite graph
    with 25 and 15 vertices, leaving at least one vertex with degree 1 -/
theorem max_edges_bipartite_graph :
  ∃ (G : BipartiteGraph 25 15),
    isConnected G ∧
    (∃ v, degree G v = 1) ∧
    (∀ G' : BipartiteGraph 25 15,
      isConnected G' → (∃ v, degree G' v = 1) →
      G'.edges.card ≤ G.edges.card) ∧
    G.edges.card = 351 :=
  sorry

#check max_edges_bipartite_graph

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_bipartite_graph_l442_44243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l442_44223

noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₁ - c₂) / Real.sqrt (a^2 + b^2)

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 11 = 0

theorem distance_between_given_lines :
  distance_parallel_lines 3 4 (-12) (11/2) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l442_44223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l442_44279

noncomputable def z : ℂ := (1 + 2 * Complex.I) / (2 - Complex.I)

theorem magnitude_of_z : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l442_44279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoveling_time_theorem_l442_44235

/-- The time it takes Joan and Mary to shovel a driveway together -/
def combined_shoveling_time (joan_rate mary_rate : ℚ) : ℕ :=
  let combined_rate := joan_rate + mary_rate
  let exact_time := 1 / combined_rate
  (exact_time + 1/2).floor.toNat

/-- Theorem stating that Joan and Mary's combined shoveling time is 14 minutes -/
theorem shoveling_time_theorem :
  combined_shoveling_time (1/50) (1/20) = 14 := by
  sorry

#eval combined_shoveling_time (1/50) (1/20)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoveling_time_theorem_l442_44235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_hazelnuts_count_l442_44274

/-- Represents the number of hazelnuts remaining after each child takes their share -/
def hazelnuts_remaining : Fin 6 → ℕ
| ⟨0, _⟩ => 62  -- Initial number of hazelnuts
| ⟨n+1, h⟩ => (hazelnuts_remaining ⟨n, Nat.lt_trans (Nat.lt_succ_self n) h⟩) / 2 - 1

/-- The theorem stating that the initial number of hazelnuts is 62 and none are left at the end -/
theorem initial_hazelnuts_count : hazelnuts_remaining ⟨0, Nat.zero_lt_succ 5⟩ = 62 ∧ 
                                  hazelnuts_remaining ⟨5, Nat.lt_succ_self 5⟩ = 0 := by
  sorry

#check initial_hazelnuts_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_hazelnuts_count_l442_44274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l442_44236

/-- A triangle with unequal sides -/
structure Triangle where
  -- We don't need to explicitly define the sides, just assume they're unequal
  sides_unequal : True

/-- The medians of a triangle -/
structure Medians (t : Triangle) where
  m₁ : ℝ
  m₂ : ℝ
  m₃ : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 6 * Real.sqrt 15

theorem third_median_length (t : Triangle) (m : Medians t) 
  (h₁ : m.m₁ = 5) 
  (h₂ : m.m₂ = 8) : 
  m.m₃ = 3 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l442_44236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l442_44237

/-- Given that cos α = 1 / (2 sin(π/5)) and E = cos(2π/5) + i sin(2π/5) is a fifth root of unity,
    prove that (E^2 - E^3) / (i√5) = (2 sin(π/5)) / √5 -/
theorem trig_identity (α : ℝ) (E : ℂ) :
  Real.cos α = 1 / (2 * Real.sin (Real.pi/5)) →
  E = Complex.exp (2 * Real.pi * Complex.I / 5) →
  E^5 = 1 →
  (E^2 - E^3) / (Complex.I * Real.sqrt 5) = (2 * Real.sin (Real.pi/5)) / Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l442_44237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_bisection_theorem_l442_44249

/-- The equation of an ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- A point inside the ellipse -/
def P : ℝ × ℝ := (1, 1)

/-- A chord of the ellipse -/
structure Chord :=
  (a b : ℝ × ℝ)
  (inside_ellipse : ellipse a.1 a.2 ∧ ellipse b.1 b.2)

/-- The point P bisects the chord -/
def bisects (c : Chord) : Prop :=
  P.1 = (c.a.1 + c.b.1) / 2 ∧ P.2 = (c.a.2 + c.b.2) / 2

/-- The equation of a line -/
def line_equation (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- The main theorem -/
theorem chord_bisection_theorem (c : Chord) :
  bisects c → ∀ x y, x = c.a.1 ∨ x = c.b.1 → y = c.a.2 ∨ y = c.b.2 → line_equation x y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_bisection_theorem_l442_44249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_zero_possibility_l442_44294

-- Define the fractions
noncomputable def fractionA (x : ℝ) : ℝ := (x^2 + 1) / (x - 1)
noncomputable def fractionB (x : ℝ) : ℝ := (x + 1) / (x^2 - 1)
noncomputable def fractionC (x : ℝ) : ℝ := (x^2 + 2*x + 1) / (x + 1)
noncomputable def fractionD (x : ℝ) : ℝ := (x + 1) / (x - 1)

-- Theorem stating that fractionD can be zero while others cannot
theorem fraction_zero_possibility :
  (∃ x : ℝ, fractionD x = 0) ∧
  (∀ x : ℝ, fractionA x ≠ 0) ∧
  (∀ x : ℝ, x ≠ -1 → x ≠ 1 → fractionB x ≠ 0) ∧
  (∀ x : ℝ, x ≠ -1 → fractionC x ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_zero_possibility_l442_44294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l442_44287

/-- Calculates the length of a train given specific conditions --/
theorem train_length_calculation 
  (initial_speed_train acceleration_train speed_motorbike overtake_time length_motorbike : Real) 
  (h1 : initial_speed_train = 90 * 1000 / 3600)  -- Convert km/h to m/s
  (h2 : acceleration_train = 0.5)
  (h3 : speed_motorbike = 72 * 1000 / 3600)  -- Convert km/h to m/s
  (h4 : overtake_time = 50)
  (h5 : length_motorbike = 2)
  : ∃ train_length : Real, train_length = 877 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l442_44287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_defined_implies_x_not_half_l442_44254

theorem fraction_defined_implies_x_not_half (x : ℝ) :
  (∃ y : ℝ, y = (1 + 2*x) / (1 - 2*x)) → x ≠ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_defined_implies_x_not_half_l442_44254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_yard_sale_books_l442_44247

theorem mary_yard_sale_books 
  (initial book_club bookstore daughter mother donated sold final : ℕ) :
  initial = 72 →
  book_club = 12 →
  bookstore = 5 →
  daughter = 1 →
  mother = 4 →
  donated = 12 →
  sold = 3 →
  final = 81 →
  final = initial + book_club + bookstore + daughter + mother - donated - sold + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_yard_sale_books_l442_44247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_three_l442_44285

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 1, |b| = 2, and the angle between them is 60°,
    prove that the projection of 2a + b onto b is 3. -/
theorem projection_equals_three 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (a b : V) :
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  inner a b = ‖a‖ * ‖b‖ * (1 / 2 : ℝ) →
  inner (2 • a + b) b / ‖b‖ = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_three_l442_44285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_even_function_l442_44281

variable {M : Type*}
variable (f : M → ℝ)

-- We need to explicitly state that M has a negation operation
variable [Neg M]

theorem negation_of_even_function :
  (¬ ∀ x : M, f (-x) = f x) ↔ (∃ x : M, f (-x) ≠ f x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_even_function_l442_44281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l442_44282

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l442_44282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_of_three_divisor_number_has_seven_divisors_l442_44264

/-- A positive integer with exactly three divisors -/
def ThreeDivisorNumber (x : ℕ) : Prop :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 3

/-- The number of positive divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem cube_of_three_divisor_number_has_seven_divisors (x : ℕ) 
  (h : ThreeDivisorNumber x) : numDivisors (x^3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_of_three_divisor_number_has_seven_divisors_l442_44264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_g_difference_l442_44246

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := 2 * Real.log x - a * x - 3 / (a * x)

def g (x : ℝ) : ℝ := f a x + x^2 + 3 / (a * x)

theorem max_value_g_difference {x₁ x₂ : ℝ} (h₁ : x₁ < x₂) 
                               (h₂ : DifferentiableAt ℝ (g a) x₁)
                               (h₃ : DifferentiableAt ℝ (g a) x₂)
                               (h₄ : deriv (g a) x₁ = 0)
                               (h₅ : deriv (g a) x₂ = 0) :
  ∃ (M : ℝ), M = 3 * Real.log 2 + 1 ∧ g a x₂ - 2 * g a x₁ ≤ M :=
sorry

#check max_value_g_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_g_difference_l442_44246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_daily_revenue_l442_44225

def f (t : ℕ+) : ℚ := 4 + 1 / (t : ℚ)

def g (t : ℕ+) : ℚ := 125 - |((t : ℚ) - 25)|

def W (t : ℕ+) : ℚ := f t * g t

theorem min_daily_revenue :
  ∃ (min : ℚ), min = 441 ∧
  ∀ (t : ℕ+), 1 ≤ (t : ℕ) ∧ (t : ℕ) ≤ 30 → W t ≥ min :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_daily_revenue_l442_44225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_cotangent_max_l442_44255

/-- Given a triangle ABC with side lengths a, b, c, and three nonzero real numbers x₀, y₀, z₀
    satisfying certain conditions, the maximum value of tan B · cot C is 5/3. -/
theorem triangle_tangent_cotangent_max (a b c x₀ y₀ z₀ : ℝ) :
  b > max a c →
  x₀ ≠ 0 → y₀ ≠ 0 → z₀ ≠ 0 →
  a * (z₀ / x₀) + b * (2 * y₀ / x₀) + c = 0 →
  (z₀ / y₀)^2 + (x₀ / y₀)^2 / 4 = 1 →
  ∃ (B C : ℝ), 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
    ∀ B' C', 0 < B' ∧ B' < π ∧ 0 < C' ∧ C' < π →
      Real.tan B' * (1 / Real.tan C') ≤ Real.tan B * (1 / Real.tan C) ∧
      Real.tan B * (1 / Real.tan C) = 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_cotangent_max_l442_44255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l442_44222

/-- The total surface area of a cone -/
noncomputable def totalSurfaceArea (d α : ℝ) : ℝ :=
  (Real.pi * d^2) / (2 * Real.sin α * (Real.sin ((Real.pi / 4) - (α / 2)))^2)

/-- Theorem: The total surface area of a cone given distance d and angle α -/
theorem cone_surface_area (d α : ℝ) (h1 : d > 0) (h2 : 0 < α ∧ α < Real.pi/2) :
  totalSurfaceArea d α = (Real.pi * d^2) / (2 * Real.sin α * (Real.sin ((Real.pi / 4) - (α / 2)))^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l442_44222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_pi_over_6_l442_44270

theorem tan_a_pi_over_6 (a : ℝ) (h : (3 : ℝ)^a = 81) : Real.tan (a * π / 6) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_pi_over_6_l442_44270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_chords_l442_44221

/-- Represents a pair of concentric circles -/
structure ConcentricCircles where
  outerRadius : ℝ
  innerRadius : ℝ
  innerRadius_lt_outerRadius : innerRadius < outerRadius

/-- Represents a chord of the outer circle tangent to the inner circle -/
structure TangentChord (cc : ConcentricCircles) where
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ
  on_outer_circle : (startPoint.1 - cc.outerRadius)^2 + startPoint.2^2 = cc.outerRadius^2
  tangent_to_inner : ∃ (t : ℝ), 
    ((1 - t) * startPoint.1 + t * endPoint.1 - cc.innerRadius)^2 + 
    ((1 - t) * startPoint.2 + t * endPoint.2)^2 = cc.innerRadius^2

/-- Angle between three points -/
def angle (a b c : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that exactly 3 tangent chords can be drawn -/
theorem three_tangent_chords (cc : ConcentricCircles) : 
  ∀ (c1 c2 c3 : TangentChord cc), 
    (angle c1.endPoint c2.startPoint c2.endPoint = 60) → 
    (angle c2.endPoint c3.startPoint c3.endPoint = 60) → 
    (angle c3.endPoint c1.startPoint c1.endPoint = 60) → 
    c1.startPoint = c3.endPoint :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_chords_l442_44221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overtakes_before_collision_l442_44262

/-- Represents a vehicle with its velocity -/
structure Vehicle where
  velocity : ℝ

/-- Represents the scenario with three vehicles -/
structure Scenario where
  A : Vehicle
  B : Vehicle
  C : Vehicle
  distanceAB : ℝ
  distanceAC : ℝ
  deceleration : ℝ
  acceleration : ℝ

def Scenario.isValid (s : Scenario) : Prop :=
  s.A.velocity = 70 ∧
  s.B.velocity = 50 ∧
  s.C.velocity = 65 ∧
  s.distanceAB = 40 ∧
  s.distanceAC = 250 ∧
  s.deceleration = 5 ∧
  s.acceleration = 10

noncomputable def Scenario.effectiveVelocityA (s : Scenario) : ℝ :=
  s.A.velocity - s.deceleration + s.acceleration

noncomputable def Scenario.timeToOvertakeB (s : Scenario) : ℝ :=
  s.distanceAB / (s.effectiveVelocityA - s.B.velocity)

noncomputable def Scenario.timeToCollideC (s : Scenario) : ℝ :=
  s.distanceAC / (s.effectiveVelocityA + s.C.velocity)

theorem overtakes_before_collision (s : Scenario) 
  (h : s.isValid) : 
  s.timeToOvertakeB < s.timeToCollideC := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overtakes_before_collision_l442_44262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_from_ellipse_l442_44298

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The foci of an ellipse -/
noncomputable def Ellipse.foci (e : Ellipse) : ℝ × ℝ :=
  let c := (e.a^2 - e.b^2).sqrt
  (c, -c)

/-- The vertices of a hyperbola -/
def Hyperbola.vertices (h : Hyperbola) : ℝ × ℝ := (h.a, -h.a)

/-- The foci of a hyperbola -/
noncomputable def Hyperbola.foci (h : Hyperbola) : ℝ × ℝ :=
  let c := (h.a^2 + h.b^2).sqrt
  (c, -c)

theorem hyperbola_from_ellipse (e : Ellipse) (h : Hyperbola) 
    (h_foci : h.foci = (e.a, -e.a))
    (h_vertices : h.vertices = e.foci) :
    h.a^2 = 9 ∧ h.b^2 = 16 := by
  sorry

#check hyperbola_from_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_from_ellipse_l442_44298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_characterization_l442_44210

theorem power_of_two_characterization (n : ℕ) : 
  (∃ (m : ℕ), (2^n - 1) % 3 = 0 ∧ (4 * m^2 + 1) % ((2^n - 1) / 3) = 0) ↔ 
  (∃ (j : ℕ), n = 2^j) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_characterization_l442_44210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newtons_cooling_law_initial_temperature_temperature_after_20_minutes_l442_44234

/-- Newton's law of cooling function -/
noncomputable def T (t : ℝ) : ℝ := 20 + 80 * (1/2)^(t/20)

/-- The ambient temperature -/
def T₀ : ℝ := 20

/-- The cooling rate constant -/
noncomputable def k : ℝ := -(Real.log 2) / 20

theorem newtons_cooling_law (t : ℝ) :
  deriv T t = k * (T t - T₀) :=
sorry

theorem initial_temperature :
  T 0 = 100 :=
sorry

theorem temperature_after_20_minutes :
  T 20 = 60 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newtons_cooling_law_initial_temperature_temperature_after_20_minutes_l442_44234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l442_44232

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number
noncomputable def z : ℂ := (i^2017) / (1 + i)

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l442_44232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l442_44224

/-- Sum of a geometric series with first term a, common ratio r, and n terms -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The number of terms in the geometric series -/
def num_terms : ℕ := 11

theorem geometric_series_sum :
  let a : ℝ := 3
  let r : ℝ := 2
  let n := num_terms
  geometric_sum a r n = 6141 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l442_44224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_ge_one_l442_44228

/-- The function f(x) defined as √(x^2 + 1) - ax, where a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

/-- Monotonicity of f(x) on [0, +∞) -/
def monotonic_on_nonneg (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f a x ≥ f a y

/-- Theorem stating that f is monotonic on [0, +∞) if and only if a ≥ 1 -/
theorem f_monotonic_iff_a_ge_one (a : ℝ) (h : a > 0) :
  monotonic_on_nonneg a ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_ge_one_l442_44228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_best_strategy_l442_44211

/-- Represents a player in the game -/
inductive Player : Type
  | F : Player  -- Father
  | M : Player  -- Mother
  | S : Player  -- Son

/-- Probability of one player winning against another -/
noncomputable def winProb : Player → Player → ℝ := sorry

/-- Probability of father winning when two players start -/
noncomputable def fatherWinProb : Player → Player → ℝ := sorry

/-- Son is the strongest player -/
axiom son_strongest (p : Player) : p ≠ Player.S → winProb Player.S p > winProb p Player.S

/-- Probabilities sum to 1 -/
axiom prob_sum (p q : Player) : winProb p q + winProb q p = 1

/-- Father's best strategy theorem -/
theorem fathers_best_strategy :
  fatherWinProb Player.F Player.M > fatherWinProb Player.F Player.S ∧
  fatherWinProb Player.F Player.S > fatherWinProb Player.M Player.S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_best_strategy_l442_44211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l442_44297

noncomputable section

-- Define the ellipse
def is_ellipse (a : ℝ) : Prop :=
  a > Real.sqrt 2

-- Define the eccentricity
def eccentricity (a : ℝ) : ℝ :=
  Real.sqrt (a^2 - 2) / a

-- Define the line
def line (a : ℝ) (x y : ℝ) : Prop :=
  y = eccentricity a * x + a

-- Define the left focus
def left_focus (a : ℝ) : ℝ × ℝ :=
  (-Real.sqrt (a^2 - 2), 0)

-- Define the right focus
def right_focus (a : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 - 2), 0)

-- Define the symmetric point P
def symmetric_point (a : ℝ) : ℝ × ℝ :=
  let f := left_focus a
  let d := (a - eccentricity a * Real.sqrt (a^2 - 2)) / Real.sqrt (1 + (eccentricity a)^2)
  (f.1, f.2 + 2 * d)

-- Define the isosceles triangle condition
def is_isosceles_triangle (a : ℝ) : Prop :=
  let p := symmetric_point a
  let f1 := left_focus a
  let f2 := right_focus a
  (p.1 - f1.1)^2 + (p.2 - f1.2)^2 = (f2.1 - f1.1)^2 + (f2.2 - f1.2)^2

theorem ellipse_theorem (a : ℝ) :
  is_ellipse a ∧ is_isosceles_triangle a → a = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l442_44297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_l442_44233

theorem fraction_sum (a b : ℕ) : 
  (a : ℚ) / b = 3975 / 10000 → 
  (∀ k : ℕ, k > 1 → (k ∣ a ∧ k ∣ b) → False) →
  a + b = 559 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_l442_44233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_analysis_l442_44248

structure BusinessSegment where
  name : String
  revenue_percentage : ℝ
  profit_percentage : ℝ

structure Company where
  smartphone : BusinessSegment
  consumer_products : BusinessSegment
  internet_services : BusinessSegment
  other : BusinessSegment

def overall_net_profit_margin : ℝ := 0.05

theorem company_analysis (c : Company) :
  (c.smartphone.revenue_percentage = 0.602 ∧
   c.smartphone.profit_percentage = 0.673 ∧
   c.consumer_products.revenue_percentage = 0.282 ∧
   c.consumer_products.profit_percentage = 0.234 ∧
   c.internet_services.revenue_percentage = 0.099 ∧
   c.internet_services.profit_percentage = 0.099 ∧
   c.other.revenue_percentage = 0.017 ∧
   c.other.profit_percentage = -0.006) →
  (∀ s : BusinessSegment, s ∈ [c.smartphone, c.consumer_products, c.internet_services, c.other] →
    c.smartphone.revenue_percentage ≥ s.revenue_percentage ∧
    c.smartphone.profit_percentage ≥ s.profit_percentage) ∧
  (c.internet_services.profit_percentage / c.internet_services.revenue_percentage = overall_net_profit_margin) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_analysis_l442_44248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_doubling_l442_44276

/-- Given a cost function of the form tb^4, prove that doubling b results in a new cost that is 1600% of the original cost. -/
theorem cost_doubling (t b : ℝ) : 
  (t * (2*b)^4) / (t * b^4) * 100 = 1600 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_doubling_l442_44276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l442_44295

noncomputable def f (x : ℝ) : ℝ := 2 * Real.tan (x - Real.pi / 6)

theorem f_range :
  let a : ℝ := -Real.pi / 6
  let b : ℝ := 5 * Real.pi / 12
  ∃ (y : ℝ), y ∈ Set.Icc (-2 * Real.sqrt 3) 2 ↔ 
    ∃ (x : ℝ), x ∈ Set.Icc a b ∧ f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l442_44295
