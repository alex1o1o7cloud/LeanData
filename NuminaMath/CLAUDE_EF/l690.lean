import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_l690_69088

-- Define the parameter m
variable (m : ℝ) 

-- Define the curves C and E
def C (x y : ℝ) : Prop := x^2/4 + y^2/(4-m) = 1
def E (x y : ℝ) : Prop := x^2 - y^2/(m-1) = 1

-- Define the point P
def P (m : ℝ) : Prop := ∃ x y, x > 0 ∧ y > 0 ∧ C m x y ∧ E m x y

-- Define the foci F₁ and F₂
noncomputable def F₁ (m : ℝ) : ℝ × ℝ := sorry
noncomputable def F₂ (m : ℝ) : ℝ × ℝ := sorry

-- Define the tangent line l at P
noncomputable def l (m : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the incenter M of triangle F₁PF₂
noncomputable def M (m : ℝ) : ℝ × ℝ := sorry

-- Define point N as the intersection of F₁M and l
noncomputable def N (m : ℝ) : ℝ × ℝ := sorry

-- State the theorem
theorem sum_of_x_coordinates (h : 1 < m ∧ m < 4) : (M m).1 + (N m).1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_l690_69088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_from_trisection_l690_69099

theorem hypotenuse_length_from_trisection (α : Real) 
  (h_α_pos : 0 < α) (h_α_lt_pi_half : α < π / 2) : 
  ∃ (a b c : Real),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    c^2 = a^2 + b^2 ∧
    (2/3 * a)^2 + (1/3 * b)^2 = Real.sin α^2 ∧
    (1/3 * a)^2 + (2/3 * b)^2 = Real.cos α^2 ∧
    c = 3 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_from_trisection_l690_69099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l690_69058

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The total volume of n cubes, each with side length s -/
def total_volume (n : ℕ) (s : ℝ) : ℝ := n * (cube_volume s)

theorem problem_solution :
  let jim_cubes := total_volume 7 3
  let laura_cubes := total_volume 4 4
  jim_cubes + laura_cubes = 445 := by
  simp [total_volume, cube_volume]
  ring

#eval (7 : ℕ) * (3 : ℕ)^3 + (4 : ℕ) * (4 : ℕ)^3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l690_69058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l690_69092

/-- Represents a triangle -/
structure Triangle where
  /-- The triangle is equilateral -/
  is_equilateral : Prop
  /-- The length of the base of the triangle -/
  base_length : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ

/-- An equilateral triangle with base length 10 has perimeter 30 -/
theorem equilateral_triangle_perimeter : 
  ∀ (t : Triangle), 
  t.is_equilateral → 
  t.base_length = 10 → 
  t.perimeter = 30 :=
by
  intro t h_equilateral h_base
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l690_69092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_l690_69086

theorem trigonometric_values (α : ℝ) 
  (h1 : Real.cos α = -Real.sqrt 5 / 5) 
  (h2 : α ∈ Set.Ioo π (3 * π / 2)) : 
  Real.tan α = 2 ∧ 
  (3 * Real.sin (π + α) + Real.cos (3 * π - α)) / 
  (Real.sin (3 * π / 2 + α) + 2 * Real.sin (α - 2 * π)) = -7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_l690_69086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_a_l690_69028

def sequence_a : ℕ → ℚ
  | 0 => -1/9  -- Define the base case for n = 0
  | n + 1 => sequence_a n / (8 * sequence_a n + 1)

theorem max_value_of_sequence_a :
  ∃ (M : ℚ), M = 1/7 ∧ ∀ (n : ℕ), sequence_a n ≤ M :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_a_l690_69028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_cardinality_even_l690_69068

/-- The set of complex numbers z satisfying |z| < 20 and e^z = -(z - 1)/(z + 1) -/
def S : Set ℂ :=
  {z : ℂ | Complex.abs z < 20 ∧ Complex.exp z = -(z - 1) / (z + 1)}

/-- The cardinality of S is even -/
theorem S_cardinality_even : ∃ k : ℕ, ↑(Nat.card S) = 2 * k := by
  sorry

#check S_cardinality_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_cardinality_even_l690_69068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circular_region_l690_69027

/-- The area of a region bounded by three circular arcs -/
theorem area_of_circular_region (r : ℝ) (θ : ℝ) : 
  r > 0 → 
  θ = π/4 → 
  3 * (θ * r^2 / 2) - (Real.sqrt 3 / 4 * (r * Real.sqrt (2 - 2 * Real.cos θ))^2) = 
    -25 * Real.sqrt 3 / 4 + 75 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circular_region_l690_69027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_26_l690_69035

-- Define the series A
noncomputable def A : ℝ := ∑' n, if n % 2 ≠ 0 ∧ n % 5 ≠ 0 then (-1)^((n - 1) / 2) / n^2 else 0

-- Define the series B
noncomputable def B : ℝ := ∑' n, if n % 10 = 5 then (-1)^((n - 5) / 10) / n^2 else 0

-- Theorem statement
theorem A_div_B_eq_26 : A / B = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_26_l690_69035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_slope_l690_69098

-- Define the point of light emission
def emission_point : ℝ × ℝ := (-2, -3)

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x + 3)^2 + (y - 2)^2 = 1

-- Define the reflected ray's line equation
def reflected_ray (k : ℝ) (x y : ℝ) : Prop := y + 3 = k * (x - 2)

-- Define the tangency condition
def is_tangent (k : ℝ) : Prop :=
  ∃ x y, circle_equation x y ∧ reflected_ray k x y ∧
  (abs (-3 * k - 2 - 2 * k - 3) / Real.sqrt (k^2 + 1) = 1)

-- Theorem statement
theorem reflected_ray_slope :
  ∃ k, is_tangent k ∧ (k = -4/3 ∨ k = -3/4) := by
  sorry

#check reflected_ray_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_slope_l690_69098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_distance_relation_l690_69017

/-- Predicate to state that O is the intersection of the medians of triangle ABC -/
def IsMedianIntersection (A B C O : ℂ) : Prop :=
  O = (1/3 : ℂ) * (A + B + C)

/-- Given a triangle ABC with centroid O, prove that the sum of the squares of the triangle's sides
    is equal to three times the sum of the squares of the distances from each vertex to the centroid. -/
theorem triangle_centroid_distance_relation (A B C O : ℂ) : 
  IsMedianIntersection A B C O →
  Complex.abs (B - A)^2 + Complex.abs (C - B)^2 + Complex.abs (A - C)^2 = 
  3 * (Complex.abs (O - A)^2 + Complex.abs (O - B)^2 + Complex.abs (O - C)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_distance_relation_l690_69017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_135_l690_69082

/-- The angle of inclination of a line passing through two points -/
noncomputable def angle_of_inclination (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.arctan ((y2 - y1) / (x2 - x1)) * (180 / Real.pi)

/-- Theorem: The angle of inclination of a line passing through (-2, 0) and (-5, 3) is 135° -/
theorem line_angle_135 :
  angle_of_inclination (-2) 0 (-5) 3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_135_l690_69082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l690_69006

/-- The speed of a train given its length and time to cross a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: A train with length 1500 meters that takes 15 seconds to cross a point has a speed of 100 meters per second -/
theorem train_speed_calculation :
  train_speed 1500 15 = 100 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l690_69006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_distance_from_start_l690_69069

/-- Alice's walk from a flagpole base --/
noncomputable def alice_walk (west east north south : ℝ) : ℝ := 
  Real.sqrt ((east - west)^2 + (north - south)^2)

/-- Theorem: Alice's distance from start after her walk --/
theorem alice_distance_from_start : 
  alice_walk 30 80 50 20 = Real.sqrt 3400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_distance_from_start_l690_69069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_second_difference_installment_plan_cost_difference_l690_69014

/-- Represents the cost difference between consecutive years -/
structure YearlyDifference where
  second_first : ℕ  -- Difference between second and first year
  fourth_third : ℕ  -- Difference between fourth and third year

/-- Represents the yearly costs -/
structure YearlyCosts where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Theorem stating the difference between third and second year costs -/
theorem third_second_difference (
  total_payment : ℕ)
  (diff : YearlyDifference)
  (costs : YearlyCosts) : costs.third - costs.second = 2 :=
by
  sorry

/-- Main theorem proving the cost difference -/
theorem installment_plan_cost_difference : ∃ (
  total_payment : ℕ)
  (diff : YearlyDifference)
  (costs : YearlyCosts),
  total_payment = 96 ∧
  diff.second_first = 2 ∧
  diff.fourth_third = 4 ∧
  costs.first = 20 ∧
  costs.second = costs.first + diff.second_first ∧
  costs.fourth = costs.third + diff.fourth_third ∧
  total_payment = costs.first + costs.second + costs.third + costs.fourth ∧
  costs.third - costs.second = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_second_difference_installment_plan_cost_difference_l690_69014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l690_69076

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_properties :
  (∀ x > 1, lg (x + f x) = lg x + lg (f x)) →
  (∀ x₁ x₂, 1 < x₂ ∧ x₂ < x₁ → f x₁ < f x₂) ∧
  (∀ x > 1, x + f x ≥ 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l690_69076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_candy_kinds_l690_69054

/-- Represents the arrangement of candies on a counter. -/
def CandyArrangement := List Nat

/-- Checks if the number of candies between two indices is even. -/
def evenCandiesBetween (arr : CandyArrangement) (i j : Nat) : Prop :=
  (j - i - 1) % 2 = 0

/-- Checks if the arrangement satisfies the condition that between any two candies
    of the same kind, there is an even number of candies. -/
def validArrangement (arr : CandyArrangement) : Prop :=
  ∀ i j, i < j → arr.get? i = arr.get? j → evenCandiesBetween arr i j

/-- The main theorem stating that the minimum number of kinds of candies is 46. -/
theorem min_candy_kinds (arr : CandyArrangement) 
    (h1 : arr.length = 91)
    (h2 : validArrangement arr) : 
  (arr.toFinset).card ≥ 46 := by
  sorry

#check min_candy_kinds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_candy_kinds_l690_69054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l690_69020

/-- The time taken for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

/-- Theorem: A 400 m long train traveling at 72 km/h takes 20 seconds to cross an electric pole -/
theorem train_crossing_theorem :
  train_crossing_time 400 72 = 20 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_crossing_time 400 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l690_69020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_with_hcf_and_lcm_factors_l690_69077

theorem larger_number_with_hcf_and_lcm_factors 
  (a b : ℕ) 
  (hcf_ab : Nat.gcd a b = 23)
  (lcm_factors : ∃ (m n : ℕ), Nat.lcm a b = 23 * 13 * 15 ∧ m * n = 13 * 15 ∧ Nat.Coprime m n)
  : max a b = 345 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_with_hcf_and_lcm_factors_l690_69077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l690_69013

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 3)
noncomputable def b (n : ℝ) : ℝ × ℝ := (n, Real.sqrt 3)
noncomputable def e : ℝ × ℝ := (a.1 / Real.sqrt (a.1^2 + a.2^2), a.2 / Real.sqrt (a.1^2 + a.2^2))

theorem vector_properties (n : ℝ) :
  (∃ (k : ℝ), k > 0 ∧ a = Prod.map (· * k) (· * k) (b n)) → n = 1 ∧
  ((a.1 * (b n).1 + a.2 * (b n).2) / Real.sqrt (a.1^2 + a.2^2) = 3) → n = 3 := by
  sorry

#check vector_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l690_69013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crackers_per_sleeve_l690_69052

/-- The number of crackers Chad uses per sandwich -/
def crackers_per_sandwich : ℕ := 2

/-- The number of sandwiches Chad eats per night -/
def sandwiches_per_night : ℕ := 5

/-- The number of sleeves in a box of crackers -/
def sleeves_per_box : ℕ := 4

/-- The number of boxes of crackers -/
def num_boxes : ℕ := 5

/-- The number of nights 5 boxes of crackers last -/
def nights_lasted : ℕ := 56

/-- Theorem stating the number of crackers in each sleeve -/
theorem crackers_per_sleeve (crackers_per_sleeve : ℕ) : 
  (num_boxes * sleeves_per_box * crackers_per_sleeve = 
   nights_lasted * sandwiches_per_night * crackers_per_sandwich) → 
  crackers_per_sleeve = 28 :=
by
  intro h
  have : crackers_per_sleeve = (nights_lasted * sandwiches_per_night * crackers_per_sandwich) / (num_boxes * sleeves_per_box) := by
    sorry
  rw [this]
  norm_num
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crackers_per_sleeve_l690_69052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_two_ways_l690_69029

/-- Represents the attributes of a block -/
structure BlockAttributes where
  material : Fin 3
  size : Fin 3
  color : Fin 5
  shape : Fin 4
deriving Fintype, DecidableEq

/-- Counts the number of differing attributes between two blocks -/
def countDifferences (b1 b2 : BlockAttributes) : Nat :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0)

/-- The reference block (metal medium purple circle) -/
def referenceBlock : BlockAttributes :=
  { material := 2, size := 1, color := 4, shape := 0 }

/-- Theorem: The number of blocks differing in exactly 2 attributes from the reference block is 40 -/
theorem blocks_differing_in_two_ways :
  (Finset.filter (fun b => countDifferences b referenceBlock = 2) (Finset.univ : Finset BlockAttributes)).card = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_two_ways_l690_69029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_specific_geometric_series_sum_l690_69038

theorem geometric_series_sum (a r : ℝ) (h : |r| < 1) :
  let S := ∑' n, a * r^n
  S = a / (1 - r) := by
  sorry

theorem specific_geometric_series_sum :
  let a : ℝ := 1
  let r : ℝ := (1/4 : ℝ)
  let S := ∑' n, a * r^n
  S = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_specific_geometric_series_sum_l690_69038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_range_of_m_l690_69050

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a - x) * abs x

-- Part I
theorem monotonicity_intervals (x : ℝ) :
  let f₁ := f 1
  (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1/2 → f₁ x₁ < f₁ x₂) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → f₁ x₁ > f₁ x₂) ∧
  (∀ x₁ x₂, 1/2 < x₁ ∧ x₁ < x₂ → f₁ x₁ > f₁ x₂) :=
by
  sorry

-- Part II
theorem range_of_m (m : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → m * x^2 + m > f 0 (f 0 x)) →
  m > 16/5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_range_of_m_l690_69050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alien_abduction_result_l690_69040

/-- The number of people an alien takes to their home planet -/
def alien_abduction (initial_abducted : ℕ) (return_percentage : ℚ) (additional_taken : ℕ) : ℕ :=
  initial_abducted - (return_percentage * initial_abducted).floor.toNat + additional_taken

/-- Theorem: Given the specific scenario, the alien takes 50 people to their home planet -/
theorem alien_abduction_result : 
  alien_abduction 200 (4/5) 10 = 50 := by
  -- Unfold the definition of alien_abduction
  unfold alien_abduction
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alien_abduction_result_l690_69040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_in_interval_l690_69070

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.sin x

-- State the theorem
theorem exists_x0_in_interval :
  ∃ x0 : ℝ, -π/2 < x0 ∧ x0 < 0 ∧ 1 < f x0 ∧ f x0 < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_in_interval_l690_69070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_possible_value_l690_69053

/-- A polynomial with real, nonnegative coefficients -/
def NonNegPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ (n : ℕ) (a : ℕ → ℝ), (∀ i, 0 ≤ a i) ∧
    ∀ x, f x = (Finset.range (n + 1)).sum (λ i ↦ a i * x ^ i)

theorem largest_possible_value
  (f : ℝ → ℝ)
  (h_nonneg : NonNegPolynomial f)
  (h_f5 : f 5 = 25)
  (h_f20 : f 20 = 1024) :
  f 10 ≤ 100 :=
by
  sorry

#check largest_possible_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_possible_value_l690_69053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_bound_l690_69072

theorem subset_intersection_bound (n : ℕ) (k : ℕ) (S : Fin k → Finset (Fin (4 * n))) :
  0 < n →
  (∀ i : Fin k, Finset.card (S i) = 2 * n) →
  (∀ i j : Fin k, i < j → Finset.card (S i ∩ S j) ≤ n) →
  k ≤ 6 ^ ((n + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_bound_l690_69072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_of_squares_l690_69018

def sequence_a : ℕ → ℤ
  | 0 => 5
  | 1 => 25
  | (n + 2) => 7 * sequence_a (n + 1) - sequence_a n - 6

theorem exists_sum_of_squares :
  ∃ (x y : ℤ), sequence_a 2022 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_of_squares_l690_69018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_range_l690_69022

noncomputable def f (x : ℝ) := Real.log (x^2 - 4*x - 5)

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x y, a < x ∧ x < y → f x < f y) ↔ a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_range_l690_69022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_selling_price_l690_69071

/-- The selling price of a book given its cost price and profit percentage -/
theorem book_selling_price (cost_price : ℚ) (profit_percentage : ℚ) :
  cost_price = 208.33 →
  profit_percentage = 20 →
  ∃ (selling_price : ℚ),
    selling_price = cost_price * (1 + profit_percentage / 100) ∧
    (selling_price * 100).floor / 100 = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_selling_price_l690_69071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_A_three_points_team_A_one_point_teams_A_two_B_one_points_l690_69023

-- Define the probabilities for each team member
noncomputable def team_A_prob : ℝ := 1/3
noncomputable def team_B_prob_1 : ℝ := 1/2
noncomputable def team_B_prob_2 : ℝ := 1/3
noncomputable def team_B_prob_3 : ℝ := 1/4

-- Define the number of team members
def team_size : ℕ := 3

-- Define the function to calculate the probability of exactly k successes in n trials
noncomputable def binomial_prob (p : ℝ) (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1-p)^(n-k)

-- Theorem statements
theorem team_A_three_points :
  binomial_prob team_A_prob team_size team_size = 1/27 := by sorry

theorem team_A_one_point :
  binomial_prob team_A_prob team_size 1 = 4/9 := by sorry

theorem teams_A_two_B_one_points :
  binomial_prob team_A_prob team_size 2 *
  (team_B_prob_1 * (1 - team_B_prob_2) * (1 - team_B_prob_3) +
   (1 - team_B_prob_1) * team_B_prob_2 * (1 - team_B_prob_3) +
   (1 - team_B_prob_1) * (1 - team_B_prob_2) * team_B_prob_3) = 11/108 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_A_three_points_team_A_one_point_teams_A_two_B_one_points_l690_69023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_a_value_l690_69047

theorem factor_implies_a_value (a b : ℤ) :
  (∃ p : Polynomial ℤ, a * X^19 + b * X^18 + 1 = (X^2 - 2*X - 1) * p) →
  a = 3571 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_a_value_l690_69047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l690_69008

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem triangle_circumradius (a b c : ℝ) (ha : a = 8) (hb : b = 6) (hc : c = 10) :
  circumradius a b c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l690_69008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_equals_eighteen_l690_69034

theorem root_product_equals_eighteen :
  (27 : ℝ) ^ (1/3) * 81 ^ (1/4) * 64 ^ (1/6) = 18 := by
  -- Rewrite each term as its integer root
  have h1 : (27 : ℝ) ^ (1/3) = 3 := by sorry
  have h2 : (81 : ℝ) ^ (1/4) = 3 := by sorry
  have h3 : (64 : ℝ) ^ (1/6) = 2 := by sorry
  
  -- Rewrite the left side using these equalities
  rw [h1, h2, h3]
  
  -- Simplify
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_equals_eighteen_l690_69034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l690_69036

-- Define the circle
def circleEquation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the point P
noncomputable def point_P : ℝ × ℝ := (1, Real.sqrt 3)

-- Define the tangent line equation
def tangentLineEquation (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 2 = 0

-- Theorem statement
theorem tangent_line_to_circle :
  ∃ (x y : ℝ), circleEquation x y ∧ (x, y) = point_P ∧ tangentLineEquation x y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l690_69036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnny_bought_10000_balls_l690_69084

/-- The number of ping pong balls Johnny bought -/
def num_balls : ℕ := 10000

/-- The original price of each ping pong ball in dollars -/
def original_price : ℚ := 1/10

/-- The discount percentage Johnny receives -/
def discount_percent : ℚ := 30/100

/-- The total amount Johnny pays in dollars -/
def total_paid : ℚ := 700

theorem johnny_bought_10000_balls : 
  (original_price * (1 - discount_percent) * (num_balls : ℚ) = total_paid) ∧
  (∀ n : ℕ, n ≠ num_balls → 
    original_price * (1 - discount_percent) * (n : ℚ) ≠ total_paid) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnny_bought_10000_balls_l690_69084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_tetrahedron_l690_69051

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of triangle ABC in cm² -/
  abc_area : ℝ
  /-- Area of triangle ABD in cm² -/
  abd_area : ℝ
  /-- Angle between planes ABC and ABD in radians -/
  plane_angle : ℝ

/-- The volume of the tetrahedron ABCD -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  (80 * Real.sqrt 2) / 3

/-- Theorem stating the volume of the tetrahedron with given properties -/
theorem volume_of_tetrahedron (t : Tetrahedron) 
  (h1 : t.ab_length = 4)
  (h2 : t.abc_area = 20)
  (h3 : t.abd_area = 16)
  (h4 : t.plane_angle = π/4) :
  tetrahedron_volume t = (80 * Real.sqrt 2) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_tetrahedron_l690_69051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l690_69030

theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.sin α < 0) 
  (h2 : Real.tan α > 0) : 
  π < α ∧ α < 3*π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l690_69030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_queue_time_calculation_l690_69043

/-- Calculates the time required to travel a given distance at a constant rate -/
noncomputable def time_to_travel (initial_distance : ℝ) (initial_time : ℝ) (remaining_distance : ℝ) : ℝ :=
  (remaining_distance * initial_time) / initial_distance

theorem movie_queue_time_calculation (initial_distance : ℝ) (initial_time : ℝ) (remaining_distance : ℝ) 
  (h1 : initial_distance = 100) 
  (h2 : initial_time = 40) 
  (h3 : remaining_distance = 150) :
  time_to_travel initial_distance initial_time remaining_distance = 60 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_travel 100 40 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_queue_time_calculation_l690_69043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_equality_l690_69097

theorem definite_integral_equality : 
  ∫ x in (0:ℝ)..(1:ℝ), (2 + Real.sqrt (1 - x^2)) = π/4 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_equality_l690_69097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_relation_l690_69039

-- Define the triangles and their properties
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the angle function (placeholder)
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem triangle_relation (ABC A'B'C' : Triangle)
  (h1 : angle ABC.B ABC.A ABC.C = angle A'B'C'.B A'B'C'.A A'B'C'.C)
  (h2 : angle ABC.A ABC.B ABC.C + angle A'B'C'.A A'B'C'.B A'B'C'.C = Real.pi) :
  ABC.a * A'B'C'.a = ABC.b * A'B'C'.b + ABC.c * A'B'C'.c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_relation_l690_69039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_production_l690_69078

/-- Given that 6 identical machines produce 330 bottles per minute at a constant rate,
    prove that 10 such machines will produce 2200 bottles in 4 minutes. -/
theorem bottle_production
  (bottles_per_minute : ℕ)
  (machines : ℕ)
  (time : ℕ)
  (h1 : bottles_per_minute = 330)
  (h2 : machines = 6)
  (h4 : time = 4) :
  (bottles_per_minute * 10 * time) / machines = 2200 := by
  sorry

#check bottle_production

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_production_l690_69078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_path_is_critical_path_l690_69057

/-- A workflow diagram is represented as a directed graph. -/
structure WorkflowDiagram where
  vertices : Type
  edges : vertices → vertices → Prop

/-- A path in a workflow diagram is a sequence of connected vertices. -/
def WorkflowPath (w : WorkflowDiagram) (start finish : w.vertices) : Type :=
  List w.vertices

/-- The length of a path is the number of edges it contains. -/
def PathLength (w : WorkflowDiagram) {start finish : w.vertices} (p : WorkflowPath w start finish) : ℕ :=
  p.length - 1

/-- A critical path is a longest path in the workflow diagram. -/
def CriticalPath (w : WorkflowDiagram) (start finish : w.vertices) (p : WorkflowPath w start finish) : Prop :=
  ∀ (q : WorkflowPath w start finish), PathLength w p ≥ PathLength w q

/-- The longest path in a workflow diagram is equivalent to the critical path. -/
theorem longest_path_is_critical_path (w : WorkflowDiagram) (start finish : w.vertices) 
    (p : WorkflowPath w start finish) :
    (∀ (q : WorkflowPath w start finish), PathLength w p ≥ PathLength w q) ↔ CriticalPath w start finish p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_path_is_critical_path_l690_69057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_converges_to_zero_l690_69015

/-- The sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 1 + 2*(a n - 1)^2

/-- The partial product of the sequence up to the nth term -/
def partial_product (n : ℕ) : ℚ :=
  Finset.prod (Finset.range (n+1)) (λ i => a i)

/-- The product of the entire sequence converges to 0 -/
theorem product_converges_to_zero :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |partial_product n| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_converges_to_zero_l690_69015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_f_greater_than_one_max_a_value_l690_69059

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - 5/2| + |x - a|

-- Part I: Prove that ln f(x) > 1 when a = -1/2
theorem ln_f_greater_than_one : ∀ x : ℝ, Real.log (f (-1/2) x) > 1 := by
  sorry

-- Part II: Prove that the maximum value of a such that f(x) ≥ a for all x ∈ ℝ is 5/4
theorem max_a_value : 
  (∃ a : ℝ, ∀ x : ℝ, f a x ≥ a) ∧ 
  (∀ b : ℝ, (∀ x : ℝ, f b x ≥ b) → b ≤ 5/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_f_greater_than_one_max_a_value_l690_69059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_percentage_profit_and_loss_l690_69025

/-- Proves that selling an article for 1280 results in the same percentage loss
as the percentage profit when selling it for 1820, given the cost price and other conditions. -/
theorem same_percentage_profit_and_loss 
  (cost_price : ℚ)
  (h1 : cost_price = 1550)
  (h2 : 1937.5 = cost_price * (1 + 0.25)) :
  (1820 - cost_price) / cost_price * 100 = (cost_price - 1280) / cost_price * 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_percentage_profit_and_loss_l690_69025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l690_69087

noncomputable def f (a x : ℝ) : ℝ := a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)

noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ -Real.sqrt 2 / 2 then Real.sqrt 2
  else if a ≤ -1/2 then -a - 1/(2*a)
  else a + 2

theorem max_value_of_f (a : ℝ) (h : a < 0) :
  ∃ (M : ℝ), ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → f a x ≤ M ∧ M = g a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l690_69087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l690_69042

noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

theorem f_properties :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x ≥ -1/2) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x ≤ 1/2) ∧
  (f (-1) = -1/2) ∧
  (f 1 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l690_69042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_converges_to_four_sevenths_l690_69032

/-- Represents the state of the forest after n animals have arrived -/
structure ForestState where
  n : ℕ  -- number of animals that have arrived
  k : ℕ  -- number of animals in groups of two

/-- The probability that the next animal joins a group of two -/
def prob_join_two (state : ForestState) : ℚ :=
  state.k / (state.n + 4)

/-- The next state of the forest after a new animal arrives -/
def next_state (state : ForestState) : ForestState :=
  { n := state.n + 1,
    k := (state.k * (state.n - 2) + 4) / (state.n + 4) }

/-- The initial state of the forest -/
def initial_state : ForestState :=
  { n := 1, k := 2 }

/-- The theorem stating that the probability converges to 4/7 -/
theorem prob_converges_to_four_sevenths :
  ∀ ε > 0, ∃ N, ∀ n ≥ N,
    |prob_join_two (Nat.iterate next_state n initial_state) - 4/7| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_converges_to_four_sevenths_l690_69032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_curve_l690_69010

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y = -24

-- Define the area of the region enclosed by the curve
noncomputable def enclosed_area : ℝ := Real.pi

-- Theorem statement
theorem area_of_curve :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, curve_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    enclosed_area = Real.pi * radius^2 := by
  sorry

#check area_of_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_curve_l690_69010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l690_69046

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (p q r : Point) : ℝ :=
  (1/2) * abs ((p.x * q.y + q.x * r.y + r.x * p.y) - (q.x * p.y + r.x * q.y + p.x * r.y))

/-- Predicate to check if a point is on the line x + y = 6 -/
def onLine (p : Point) : Prop :=
  p.x + p.y = 6

theorem triangle_PQR_area :
  let p : Point := ⟨2, 1⟩
  let q : Point := ⟨1, 4⟩
  ∀ r : Point, onLine r → triangleArea p q r = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l690_69046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_max_value_f_value_of_a_l690_69019

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 5

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 2

-- Theorem for the minimum value
theorem min_value_f : 
  ∃ (x : ℝ), x ∈ I ∧ f x = ⨅ (y ∈ I), f y ∧ f x = 31/8 ∧ x = 3/4 :=
sorry

-- Theorem for the maximum value
theorem max_value_f :
  ∃ (x : ℝ), x ∈ I ∧ f x = ⨆ (y ∈ I), f y ∧ f x = 19 ∧ x = -2 :=
sorry

-- Define the second quadratic function
def g (a x : ℝ) : ℝ := x^2 + 2*a*x + 1

-- Define the second interval
def J : Set ℝ := Set.Icc (-1) 2

-- Theorem for the value of a
theorem value_of_a :
  ∀ (a : ℝ), (∀ (x : ℝ), x ∈ J → g a x ≤ 4) ∧ (∃ (x : ℝ), x ∈ J ∧ g a x = 4) 
  → a = -1 ∨ a = -1/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_max_value_f_value_of_a_l690_69019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_120_moves_l690_69037

noncomputable def move (z : ℂ) : ℂ := z * Complex.exp (Complex.I * (Real.pi / 6)) + 6

theorem particle_position_after_120_moves :
  (move^[120]) 3 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_120_moves_l690_69037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dacids_weighted_average_l690_69011

-- Define the marks for each subject
noncomputable def english_marks : ℚ := 51
noncomputable def math_marks : ℚ := 65
noncomputable def physics_marks : ℚ := 82
noncomputable def chemistry_marks : ℚ := 67
noncomputable def biology_marks : ℚ := 85
noncomputable def history_marks : ℚ := 63
noncomputable def geography_marks : ℚ := 78
noncomputable def cs_marks : ℚ := 90

-- Define the weights for each subject
def english_weight : ℕ := 2
def math_weight : ℕ := 3
def physics_weight : ℕ := 2
def chemistry_weight : ℕ := 1
def biology_weight : ℕ := 1
def history_weight : ℕ := 1
def geography_weight : ℕ := 1
def cs_weight : ℕ := 3

-- Define the weighted average calculation
noncomputable def weighted_average : ℚ :=
  (english_marks * english_weight +
   math_marks * math_weight +
   physics_marks * physics_weight +
   chemistry_marks * chemistry_weight +
   biology_marks * biology_weight +
   history_marks * history_weight +
   geography_marks * geography_weight +
   cs_marks * cs_weight) /
  (english_weight + math_weight + physics_weight + chemistry_weight +
   biology_weight + history_weight + geography_weight + cs_weight)

-- Theorem statement
theorem dacids_weighted_average :
  (⌊weighted_average * 100⌋ : ℚ) / 100 = 73.14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dacids_weighted_average_l690_69011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_ride_to_bernard_l690_69012

/-- The distance between June and Julia's houses in miles -/
noncomputable def distance_to_julia : ℝ := 1.2

/-- The time it takes June to ride to Julia's house in minutes -/
noncomputable def time_to_julia : ℝ := 4.8

/-- The distance to Bernard's house in miles -/
noncomputable def distance_to_bernard : ℝ := 4.5

/-- The rest time during the journey to Bernard's house in minutes -/
noncomputable def rest_time : ℝ := 2

/-- June's riding speed in miles per minute -/
noncomputable def june_speed : ℝ := distance_to_julia / time_to_julia

theorem june_ride_to_bernard (total_time : ℝ) :
  total_time = distance_to_bernard / june_speed + rest_time →
  total_time = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_ride_to_bernard_l690_69012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_almost_involution_pentagonal_theorem_l690_69007

-- Define a partition as a decreasing sequence of positive integers
def Partition := List Nat

-- Define the function that operates on partitions
def partitionFunction (p : Partition) : Partition :=
  sorry

-- Define the pentagonal numbers
def pentagonal (n : Int) : Int :=
  (n * (3 * n - 1)) / 2

-- State the theorem
theorem almost_involution_pentagonal_theorem :
  ∀ (n : Nat),
    (∃ (p : Partition), partitionFunction (partitionFunction p) ≠ p) ↔
    (∃ (k : Int), n = (pentagonal k).toNat ∨ n = (pentagonal (k + 1)).toNat) :=
by
  sorry

#check almost_involution_pentagonal_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_almost_involution_pentagonal_theorem_l690_69007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_frame_angle_for_specific_radii_v_frame_angle_proof_l690_69075

/-- The angle between the sides of a V-shaped frame containing three externally tangent spheres -/
noncomputable def v_frame_angle (r₁ r₂ r₃ : ℝ) : ℝ :=
  2 * Real.arcsin (Real.sqrt 13 / 6)

/-- Theorem stating the angle of the V-shaped frame for specific sphere radii -/
theorem v_frame_angle_for_specific_radii :
  v_frame_angle 1 2 3 = 2 * Real.arcsin (Real.sqrt 13 / 6) := by
  sorry

/-- Conditions for the spheres and frame -/
structure SpheresInFrame where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  externally_tangent : Bool
  tangent_to_frame : Bool

/-- The specific configuration described in the problem -/
def problem_config : SpheresInFrame :=
  { r₁ := 1
  , r₂ := 2
  , r₃ := 3
  , externally_tangent := true
  , tangent_to_frame := true }

/-- Main theorem proving the angle for the given configuration -/
theorem v_frame_angle_proof (config : SpheresInFrame) 
  (h₁ : config.r₁ = 1)
  (h₂ : config.r₂ = 2)
  (h₃ : config.r₃ = 3)
  (h₄ : config.externally_tangent = true)
  (h₅ : config.tangent_to_frame = true) :
  v_frame_angle config.r₁ config.r₂ config.r₃ = 2 * Real.arcsin (Real.sqrt 13 / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_frame_angle_for_specific_radii_v_frame_angle_proof_l690_69075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l690_69096

/-- The trajectory of the midpoint of a line segment connecting a fixed point and a point moving on a circle. -/
theorem midpoint_trajectory (P : ℝ × ℝ) (A : ℝ × ℝ) :
  (P.1^2 + P.2^2 = 4) →  -- P is on the circle x^2 + y^2 = 4
  (A = (3, 4)) →         -- A is the fixed point (3, 4)
  let M := ((P.1 + A.1) / 2, (P.2 + A.2) / 2)  -- M is the midpoint of AP
  ((M.1 - 3/2)^2 + (M.2 - 2)^2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l690_69096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_over_two_l690_69002

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_gt_b : b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The left focus of an ellipse -/
noncomputable def left_focus (e : Ellipse) : ℝ × ℝ := 
  (-Real.sqrt (e.a^2 - e.b^2), 0)

/-- The upper endpoint of an ellipse -/
def upper_endpoint (e : Ellipse) : ℝ × ℝ := (0, e.b)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_eccentricity_sqrt_two_over_two (e : Ellipse) 
  (A : ℝ × ℝ) -- Point A on the ellipse
  (h_A_on_ellipse : A.1^2 / e.a^2 + A.2^2 / e.b^2 = 1) -- A is on the ellipse
  (h_BF_3AF : distance (upper_endpoint e) (left_focus e) = 
              3 * distance A (left_focus e)) -- |BF| = 3|AF|
  : eccentricity e = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_over_two_l690_69002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l690_69001

/-- A point with integer coordinates on the circle x^2 + y^2 = 16 -/
structure IntegerCirclePoint where
  x : ℤ
  y : ℤ
  on_circle : x^2 + y^2 = 16

/-- Distance between two points -/
noncomputable def distance (p q : IntegerCirclePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Statement of the problem -/
theorem max_ratio_on_circle (A B C D : IntegerCirclePoint) 
  (hAB : Irrational (distance A B))
  (hCD : Irrational (distance C D))
  (hDistinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  (∃ (A' B' C' D' : IntegerCirclePoint), 
    Irrational (distance A' B') ∧ 
    Irrational (distance C' D') ∧
    A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' ∧
    ∀ (X Y Z W : IntegerCirclePoint), 
      Irrational (distance X Y) → 
      Irrational (distance Z W) → 
      X ≠ Y ∧ X ≠ Z ∧ X ≠ W ∧ Y ≠ Z ∧ Y ≠ W ∧ Z ≠ W →
      (distance X Y) / (distance Z W) ≤ (distance A' B') / (distance C' D')) ∧
  (∀ (X Y Z W : IntegerCirclePoint), 
    Irrational (distance X Y) → 
    Irrational (distance Z W) → 
    X ≠ Y ∧ X ≠ Z ∧ X ≠ W ∧ Y ≠ Z ∧ Y ≠ W ∧ Z ≠ W →
    (distance X Y) / (distance Z W) ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l690_69001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_economics_test_correct_answers_l690_69021

theorem economics_test_correct_answers
  (total_enrolled : Finset ℕ)
  (correct_q1 : Finset ℕ)
  (correct_q2 : Finset ℕ)
  (not_taken : Finset ℕ)
  (h1 : total_enrolled.card = 29)
  (h2 : correct_q1.card = 19)
  (h3 : correct_q2.card = 24)
  (h4 : not_taken.card = 5)
  (h5 : correct_q2 = total_enrolled \ not_taken) :
  correct_q1.card = (correct_q1 ∩ correct_q2).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_economics_test_correct_answers_l690_69021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_framing_for_enlarged_picture_l690_69067

/-- Calculates the minimum number of linear feet of framing needed for an enlarged picture with a border -/
def min_framing_needed (original_width original_height border_width : ℚ) : ℕ :=
  let enlarged_width := original_width * 4
  let enlarged_height := original_height * 4
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (Int.ceil (perimeter / 12)).toNat

/-- Theorem stating the minimum number of linear feet of framing needed for the given picture specifications -/
theorem framing_for_enlarged_picture :
  min_framing_needed 4 6 3 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_framing_for_enlarged_picture_l690_69067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_equality_l690_69083

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the membership relation
def Point.belongs_to_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

-- Define the Line type
def Line (P Q : Point) : Set Point :=
  {R : Point | ∃ t : ℝ, R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)}

-- Define the angle between tangents (noncomputable as it involves trigonometry)
noncomputable def angle_between_tangents (P Q : Point) : ℝ := sorry

-- Define the theorem
theorem tangent_angle_equality 
  (k1 k2 : Circle) 
  (D E : Point) 
  (A : Point) 
  (B C : Point) 
  (h1 : Point.belongs_to_circle D k1 ∧ Point.belongs_to_circle D k2) 
  (h2 : Point.belongs_to_circle E k1 ∧ Point.belongs_to_circle E k2) 
  (h3 : Point.belongs_to_circle A k1) 
  (h4 : Point.belongs_to_circle B k2 ∧ Point.belongs_to_circle C k2) 
  (h5 : A ≠ D ∧ A ≠ E) 
  (h6 : B ∈ Line A D) 
  (h7 : C ∈ Line A E) : 
  angle_between_tangents B C = angle_between_tangents D E :=
by sorry

-- Instance for Set membership
instance : Membership Point (Set Point) := by exact inferInstance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_equality_l690_69083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_sum_of_coefficients_l690_69064

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def quadrilateral_vertices : List point := [(1,2), (4,5), (5,4), (4,1)]

noncomputable def perimeter (vertices : List point) : ℝ :=
  let pairs := List.zip vertices (vertices.rotateLeft 1)
  (pairs.map (fun (p1, p2) => distance p1 p2)).sum

theorem quadrilateral_perimeter_sum_of_coefficients :
  ∃ (c d : ℕ), perimeter quadrilateral_vertices = c * Real.sqrt 2 + d * Real.sqrt 10 ∧ c + d = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_sum_of_coefficients_l690_69064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_negative_correlation_l690_69091

/-- Represents a linear relationship between two variables -/
structure LinearRelationship where
  slope : ℝ
  intercept : ℝ

/-- Determines if a linear relationship represents a negative correlation -/
def is_negative_correlation (rel : LinearRelationship) : Prop :=
  rel.slope < 0

/-- Given linear relationship between x and y -/
axiom xy_relationship : LinearRelationship

/-- The specific equation for x and y -/
axiom xy_equation : xy_relationship = { slope := -2, intercept := 3 }

/-- Theorem: The correlation between x and y is negative -/
theorem xy_negative_correlation : is_negative_correlation xy_relationship := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_negative_correlation_l690_69091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_theorem_l690_69056

variable (n : ℕ)
variable (a : Fin 10 → ℕ)

-- Define the conditions
def valid_fishing_distribution (a : Fin 10 → ℕ) : Prop :=
  ∀ i : Fin 10, a i ≥ a (i.succ) ∧ a ⟨9, by norm_num⟩ ≤ a ⟨0, by norm_num⟩

-- Define the total fish caught
def total_fish (a : Fin 10 → ℕ) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 10)) a

-- Theorem statement
theorem fishing_theorem (a : Fin 10 → ℕ) 
  (h : valid_fishing_distribution a) : 
  total_fish a = Finset.sum (Finset.univ : Finset (Fin 10)) a := by
  -- The proof is trivial as it's true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_theorem_l690_69056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l690_69081

-- Define the constants a and b as noncomputable
noncomputable def a : ℝ := Real.log 2020 / Real.log 2019
noncomputable def b : ℝ := Real.log 2019 / Real.log 2020

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := a + x - b^x

-- State the theorem
theorem zero_in_interval :
  ∃ x₀ ∈ Set.Ioo (-1 : ℝ) 0, f x₀ = 0 := by
  sorry

-- Additional lemmas to support the main theorem
lemma a_bounds : 1 < a ∧ a < 2 := by
  sorry

lemma b_bounds : 0 < b ∧ b < 1 := by
  sorry

lemma f_increasing : Monotone f := by
  sorry

lemma f_zero_positive : f 0 > 0 := by
  sorry

lemma f_neg_one_negative : f (-1) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l690_69081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_half_l690_69005

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 2 * x

-- Define the angle of inclination
noncomputable def angle_of_inclination (x : ℝ) : ℝ := Real.arctan (f' x)

-- Theorem statement
theorem tangent_angle_at_half : angle_of_inclination (1/2) = π/4 := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_half_l690_69005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l690_69063

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

theorem g_is_odd : ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l690_69063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petes_walking_distance_l690_69061

/-- Represents a pedometer with a maximum count before resetting --/
structure Pedometer where
  max_count : ℕ
  steps_per_mile : ℕ

/-- Calculates the total steps given the number of resets and final reading --/
def total_steps (p : Pedometer) (resets : ℕ) (final_reading : ℕ) : ℕ :=
  resets * (p.max_count + 1) + final_reading

/-- Converts steps to miles --/
def steps_to_miles (p : Pedometer) (steps : ℕ) : ℚ :=
  steps / p.steps_per_mile

/-- Finds the closest value in a list to a given number --/
def closest_value (target : ℚ) (options : List ℚ) : ℚ :=
  options.foldl (fun acc x => if |x - target| < |acc - target| then x else acc) (options.head!)

theorem petes_walking_distance (p : Pedometer) (resets : ℕ) (final_reading : ℕ) :
  p.max_count = 99999 →
  p.steps_per_mile = 1800 →
  resets = 44 →
  final_reading = 50000 →
  closest_value (steps_to_miles p (total_steps p resets final_reading)) [2500, 3000, 3500, 4000, 4500] = 2500 := by
  sorry

#eval closest_value (2472.2222 : ℚ) [2500, 3000, 3500, 4000, 4500]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petes_walking_distance_l690_69061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l690_69024

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 8 - y^2 / 12 = 1

-- Define the foci coordinates
noncomputable def foci : Set (ℝ × ℝ) := {(-2 * Real.sqrt 5, 0), (2 * Real.sqrt 5, 0)}

-- Define the length of the transverse axis
noncomputable def transverse_axis_length : ℝ := 4 * Real.sqrt 2

-- Define the length of the conjugate axis
noncomputable def conjugate_axis_length : ℝ := 4 * Real.sqrt 3

-- Define the equations of the asymptotes
def asymptote_equations (x y : ℝ) : Prop :=
  y = Real.sqrt 6 / 2 * x ∨ y = -Real.sqrt 6 / 2 * x

-- Theorem stating the properties of the hyperbola
theorem hyperbola_properties :
  (∀ x y, hyperbola x y →
    ((x, y) ∈ foci ∨
    (∃ t, x = t ∧ y = 0 ∧ abs t = transverse_axis_length / 2) ∨
    (∃ t, x = 0 ∧ y = t ∧ abs t = conjugate_axis_length / 2) ∨
    asymptote_equations x y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l690_69024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l690_69062

-- Define the points
noncomputable def A : ℝ × ℝ := (0, 4)
noncomputable def B : ℝ × ℝ := (8, 0)
noncomputable def C : ℝ × ℝ := (6, 3)
noncomputable def D : ℝ × ℝ := (3, 0)

-- Define the lines AB and CD
noncomputable def line_AB (x : ℝ) : ℝ := -1/2 * x + 4
noncomputable def line_CD (x : ℝ) : ℝ := x - 3

-- Define the intersection point E
noncomputable def E : ℝ × ℝ := (7/3, 4/3)

-- Theorem statement
theorem length_of_AE : 
  let dist := Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2)
  dist = Real.sqrt 113 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l690_69062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_comparison_l690_69095

/-- The revenue from Vasya's promotion strategy -/
noncomputable def vasya_revenue (normal_revenue : ℝ) : ℝ := 1.6 * normal_revenue

/-- The revenue from Kolya's promotion strategy -/
noncomputable def kolya_revenue (normal_revenue : ℝ) : ℝ := (8/3) * normal_revenue

theorem promotion_comparison (normal_revenue : ℝ) 
  (h_normal : normal_revenue = 10000) :
  vasya_revenue normal_revenue > kolya_revenue normal_revenue ∧ 
  vasya_revenue normal_revenue - normal_revenue = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_comparison_l690_69095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_profit_distribution_l690_69079

/-- Represents the investment and profit distribution for a partnership --/
structure Partnership where
  investment_li_wei : ℕ
  investment_wang_gang : ℕ
  total_profit : ℕ

/-- Calculates the profit distribution for Li Wei --/
def profit_li_wei (p : Partnership) : ℕ :=
  p.total_profit * p.investment_li_wei / (p.investment_li_wei + p.investment_wang_gang)

/-- Calculates the profit distribution for Wang Gang --/
def profit_wang_gang (p : Partnership) : ℕ :=
  p.total_profit * p.investment_wang_gang / (p.investment_li_wei + p.investment_wang_gang)

/-- Theorem stating the correct profit distribution --/
theorem correct_profit_distribution (p : Partnership) 
  (h1 : p.investment_li_wei = 16000)
  (h2 : p.investment_wang_gang = 12000)
  (h3 : p.total_profit = 14000) :
  profit_li_wei p = 8000 ∧ profit_wang_gang p = 6000 := by
  sorry

#eval profit_li_wei ⟨16000, 12000, 14000⟩
#eval profit_wang_gang ⟨16000, 12000, 14000⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_profit_distribution_l690_69079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l690_69003

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- State the theorem
theorem intersection_complement_equality : A ∩ (Set.univ \ B) = Set.Ioc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l690_69003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_15_value_l690_69000

/-- Arithmetic sequence -/
noncomputable def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of arithmetic sequence -/
noncomputable def SumArithmetic (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

/-- Geometric sequence -/
noncomputable def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b_15_value (a : ℕ → ℝ) (b : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence b →
  SumArithmetic a 9 = -18 →
  SumArithmetic a 13 = -52 →
  b 5 = a 5 →
  b 7 = a 7 →
  b 15 = -64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_15_value_l690_69000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_E_bounds_l690_69041

/-- The expression E(x,y,z) defined for non-negative real numbers x, y, z. -/
noncomputable def E (x y z : ℝ) : ℝ := Real.sqrt (x * (y + 3)) + Real.sqrt (y * (z + 3)) + Real.sqrt (z * (x + 3))

/-- Theorem stating the bounds of E(x,y,z) given the constraints. -/
theorem E_bounds (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 3) :
  3 ≤ E x y z ∧ E x y z ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_E_bounds_l690_69041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_weights_equal_w2_w6_l690_69065

-- Define the vertex set
def V : Finset ℕ := Finset.range 6

-- Define the weight function for the original graph
variable (weight : ℕ → ℝ)

-- Define the neighbor function
variable (neighbors : ℕ → Finset ℕ)

-- Define the weight function for the second graph
def weight_W (i : ℕ) : ℝ := (neighbors i).sum weight

-- State the theorem
theorem sum_weights_equal_w2_w6 :
  V.sum weight = weight_W weight neighbors 1 + weight_W weight neighbors 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_weights_equal_w2_w6_l690_69065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l690_69045

theorem max_value_trig_expression :
  ∀ φ θ : ℝ, 3 * Real.cos φ * Real.cos θ + 3 * Real.sin φ * Real.sin θ ≤ 3 :=
by
  intros φ θ
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l690_69045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l690_69060

-- Define the constants a and b (assuming they are positive real numbers)
variable (a b : ℝ) (ha : a > 0) (hb : b > 0)

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line y = x
def line (x y : ℝ) : Prop := y = x

-- Define the right focus of the ellipse
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

-- Define the distance function
noncomputable def distance_to_line (a b : ℝ) : ℝ :=
  let (fx, fy) := right_focus a b
  (fx - fy) / Real.sqrt 2

-- The theorem to prove
theorem distance_from_focus_to_line (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (k : ℝ), distance_to_line a b = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l690_69060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_altitude_length_l690_69085

-- Define a triangle type
structure Triangle (α : Type*) where
  a : α
  b : α
  c : α

-- Define a triangle with sides 30, 40, and 50
def myTriangle : Triangle ℝ := ⟨30, 40, 50⟩

-- Define the properties of the triangle
axiom triangle_sides : myTriangle.a = 30 ∧ myTriangle.b = 40 ∧ myTriangle.c = 50

-- Define the shortest altitude of the triangle
noncomputable def shortest_altitude (t : Triangle ℝ) : ℝ := sorry

-- Theorem to prove
theorem shortest_altitude_length : shortest_altitude myTriangle = 24 := by
  sorry

#check shortest_altitude_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_altitude_length_l690_69085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_length_is_seven_l690_69033

/-- A box with given dimensions and cube properties -/
structure Box where
  width : ℚ
  height : ℚ
  cubeVolume : ℚ
  totalCubes : ℕ

/-- The length of the box given its properties -/
def boxLength (b : Box) : ℚ :=
  (b.cubeVolume * b.totalCubes) / (b.width * b.height)

/-- Theorem stating that the length of the box is 7 cm -/
theorem box_length_is_seven (b : Box) 
  (h1 : b.width = 18)
  (h2 : b.height = 3)
  (h3 : b.cubeVolume = 9)
  (h4 : b.totalCubes = 42) : 
  boxLength b = 7 := by
  sorry

#eval boxLength { width := 18, height := 3, cubeVolume := 9, totalCubes := 42 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_length_is_seven_l690_69033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_half_l690_69094

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/6) -/
noncomputable def circle_radius : ℝ := 1 / 2

/-- Theorem stating that the radius of the circle is 1/2 -/
theorem circle_radius_is_half :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi →
  let r := Real.sqrt ((1 * Real.sin (Real.pi / 6) * Real.cos θ) ^ 2 + (1 * Real.sin (Real.pi / 6) * Real.sin θ) ^ 2)
  r = circle_radius := by
  sorry

#check circle_radius_is_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_half_l690_69094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_area_of_circle_with_six_arcs_l690_69009

/-- The area of the inner part of a circle enclosed by six equal arcs -/
theorem inner_area_of_circle_with_six_arcs (R : ℝ) (R_pos : R > 0) : 
  (2 * R^2 * (3 * Real.sqrt 3 - π)) / 3 = 
  let circle_area := π * R^2
  let arc_length := 2 * π * R / 6
  let inner_arc_radius := R * (Real.sqrt 3 / 3)
  6 * (
    (R^2 * Real.sqrt 3 / 4) -  -- Area of equilateral triangle
    (1/3 * π * inner_arc_radius^2 - 1/2 * inner_arc_radius^2 * Real.sin (2 * π / 3))  -- Area of circular segment
  ) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_area_of_circle_with_six_arcs_l690_69009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l690_69044

noncomputable section

-- Statement A
noncomputable def f₁ (x : ℝ) : ℝ := |x| / x
noncomputable def g (x : ℝ) : ℝ := if x ≥ 0 then 1 else -1

-- Statement B
noncomputable def f₂ (x : ℝ) : ℝ := x + 1/x

-- Statement C
noncomputable def f₃ (x : ℝ) : ℝ := (x^2 - 1) / x

-- Statement D
noncomputable def f₄ (x : ℝ) : ℝ := |x - 1| - x

theorem problem_solution :
  (¬ (∀ x : ℝ, x ≠ 0 → f₁ x = g x)) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 1 → x₂ > 1 → x₁ < x₂ → f₂ x₁ < f₂ x₂) ∧
  (¬ (∀ x₁ x₂ : ℝ, x₁ ≠ 0 → x₂ ≠ 0 → x₁ < x₂ → f₃ x₁ < f₃ x₂)) ∧
  (f₄ (f₄ (1/2)) = 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l690_69044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_washing_cost_exact_l690_69089

/-- Calculates the cost of white washing a room with given dimensions and openings. -/
def white_washing_cost (length width height : ℝ) (door_width door_height : ℝ) 
  (window_width window_height : ℝ) (num_windows : ℕ) (cost_per_sqft : ℝ) : ℝ :=
  let total_wall_area := 2 * (length * height + width * height)
  let door_area := door_width * door_height
  let window_area := window_width * window_height * (num_windows : ℝ)
  let net_area := total_wall_area - door_area - window_area
  net_area * cost_per_sqft

/-- Theorem stating the cost of white washing the room with given specifications. -/
theorem white_washing_cost_exact : 
  white_washing_cost 25 15 12 6 3 4 3 3 2 = 1812 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_washing_cost_exact_l690_69089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_terms_equal_l690_69074

theorem binomial_expansion_terms_equal (a b : ℝ) (n : ℕ) 
  (h1 : n ≥ 2) 
  (h2 : a * b ≠ 0) 
  (h3 : a = 3 * b) : 
  ((7 * b) ^ n = -(n * (n - 1) / 2 * (7 * b) ^ (n - 2))) ↔ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_terms_equal_l690_69074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_determines_plane_three_lines_determine_at_most_three_planes_l690_69093

-- Define the basic geometric objects
structure Point where

structure Line where

structure Plane where

-- Define the concept of a trapezoid
structure Trapezoid where

-- Define the relation of determining a plane
def determines_plane (t : Trapezoid) (p : Plane) : Prop := sorry

-- Define the concept of lines intersecting in pairs
def lines_intersect_in_pairs (lines : List Line) : Prop := sorry

-- Define the relation of lines determining planes
def lines_determine_planes (lines : List Line) (planes : List Plane) : Prop := sorry

-- Theorem 1: A trapezoid can determine a plane
theorem trapezoid_determines_plane (t : Trapezoid) : ∃ p : Plane, determines_plane t p := by
  sorry

-- Theorem 2: Three lines intersecting in pairs can determine at most three planes
theorem three_lines_determine_at_most_three_planes (l1 l2 l3 : Line) :
  lines_intersect_in_pairs [l1, l2, l3] →
  ∀ planes : List Plane, lines_determine_planes [l1, l2, l3] planes →
  planes.length ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_determines_plane_three_lines_determine_at_most_three_planes_l690_69093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tram_max_income_l690_69090

noncomputable section

/-- Tram passenger capacity as a function of departure interval -/
def p (t : ℝ) : ℝ :=
  if 2 ≤ t ∧ t < 10 then 400 - 2 * (10 - t)^2
  else if 10 ≤ t ∧ t ≤ 20 then 400
  else 0

/-- Net income per minute as a function of departure interval -/
noncomputable def Q (t : ℝ) : ℝ := (6 * p t - 1500) / t - 60

theorem tram_max_income :
  ∀ t : ℝ, 2 ≤ t ∧ t ≤ 20 → Q t ≤ 60 ∧
  ∃ t₀ : ℝ, t₀ = 5 ∧ Q t₀ = 60 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tram_max_income_l690_69090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l690_69026

-- Define M and m functions as noncomputable
noncomputable def M (x y : ℝ) : ℝ := max x y
noncomputable def m (x y : ℝ) : ℝ := min x y

-- State the theorem
theorem problem_solution (p q r s t : ℝ) 
  (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t) :
  M (M p (m q s)) (m r (m p t)) = q := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l690_69026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l690_69055

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x - Real.sin x

theorem min_shift_for_even_function :
  ∃ (n : ℝ), n > 0 ∧ 
  (∀ (x : ℝ), f (x + n) = f (-x - n)) ∧
  (∀ (m : ℝ), m > 0 → (∀ (x : ℝ), f (x + m) = f (-x - m)) → n ≤ m) ∧
  n = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l690_69055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_unit_circle_l690_69080

theorem points_on_unit_circle (t : ℝ) : (Real.cos (2 * t))^2 + (Real.sin (2 * t))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_unit_circle_l690_69080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maurice_bonus_period_l690_69016

/-- Represents Maurice's earnings structure -/
structure MauriceEarnings where
  baseRate : ℕ → ℕ  -- Base rate per task
  bonusAmount : ℕ   -- Bonus amount
  bonusPeriod : ℕ   -- Number of tasks after which bonus is received

/-- Calculates total earnings for a given number of tasks -/
def totalEarnings (e : MauriceEarnings) (tasks : ℕ) : ℕ :=
  e.baseRate tasks + (tasks / e.bonusPeriod) * e.bonusAmount

/-- Theorem stating that Maurice receives a bonus every 10 tasks -/
theorem maurice_bonus_period :
  ∃ e : MauriceEarnings,
    (∀ t, e.baseRate t = 2 * t) ∧
    e.bonusAmount = 6 ∧
    totalEarnings e 30 = 78 ∧
    e.bonusPeriod = 10 := by
  sorry

#check maurice_bonus_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maurice_bonus_period_l690_69016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_inequality_l690_69066

/-- Two lines in 3-space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Distance between a point and a line in 3-space -/
def distance (p : ℝ × ℝ × ℝ) (l : Line3D) : ℝ := sorry

/-- Midpoint of two points in 3-space -/
def midpoint3D (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- Check if a point is on a line -/
def pointOnLine (p : ℝ × ℝ × ℝ) (l : Line3D) : Prop := sorry

theorem distance_inequality (l l' : Line3D) (A B C : ℝ × ℝ × ℝ) 
  (h1 : B = midpoint3D A C)
  (h2 : pointOnLine A l ∧ pointOnLine B l ∧ pointOnLine C l)
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h3 : a = distance A l')
  (h4 : b = distance B l')
  (h5 : c = distance C l') :
  b ≤ Real.sqrt ((a^2 + c^2) / 2) ∧ 
  (parallel l l' → b = Real.sqrt ((a^2 + c^2) / 2)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_inequality_l690_69066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_line_l690_69073

/-- The minimum distance between an ellipse and a line -/
theorem min_distance_ellipse_line :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 16 + p.2^2 / 9 = 1}
  let line := {q : ℝ × ℝ | q.1 + q.2 = 6}
  ∃ d : ℝ, d = Real.sqrt 2 / 2 ∧ 
    ∀ p ∈ ellipse, ∀ q ∈ line, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_line_l690_69073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_real_roots_l690_69004

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x - 1

theorem f_has_three_real_roots : ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_real_roots_l690_69004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_special_triangle_l690_69049

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem angle_measure_in_special_triangle (t : Triangle) 
  (h : t.a / Real.sin t.B + t.b / Real.sin t.A = 2 * t.c) : 
  t.A = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_special_triangle_l690_69049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l690_69048

/-- For a geometric sequence of n terms, P is the product of the terms,
    S is the sum of the terms, and S' is the sum of the reciprocals of the terms. -/
def geometric_sequence (n : ℕ) (P S S' : ℝ) : Prop :=
  ∃ (a r : ℝ) (hr : r ≠ 1),
    P = a^n * r^(n * (n - 1) / 2) ∧
    S = a * (1 - r^n) / (1 - r) ∧
    S' = (r^(1 - n) / a) * (1 - r^n) / (1 - r)

/-- The product of terms in a geometric sequence can be expressed
    in terms of the sum of terms and sum of reciprocals of terms. -/
theorem geometric_sequence_product (n : ℕ) (P S S' : ℝ) 
    (h : geometric_sequence n P S S') : 
    P = (S / S') ^ (n / 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l690_69048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_zero_l690_69031

/-- Given three distinct collinear points A, B, C and a point O not on their line,
    if p*OA + q*OB + r*OC = 0 for some real p, q, r, then p + q + r = 0 -/
theorem sum_of_coefficients_zero
  (O A B C : EuclideanSpace ℝ (Fin 2))  -- 2D space
  (p q r : ℝ)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_collinear : Collinear ℝ {A, B, C})
  (h_not_on_line : ¬ Collinear ℝ {O, A, B})
  (h_vector_sum : p • (A - O) + q • (B - O) + r • (C - O) = 0) :
  p + q + r = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_zero_l690_69031
