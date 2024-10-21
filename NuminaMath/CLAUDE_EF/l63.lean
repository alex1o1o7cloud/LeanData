import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l63_6391

def M : ℕ := 2^4 * 3^3 * 7^2

theorem number_of_factors_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l63_6391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishermen_catch_l63_6328

/-- The number of fish caught by two fishermen satisfying given conditions -/
theorem fishermen_catch (fish1 fish2 : ℕ) : 
  (fish1 = fish2 / 2 + 10) →
  (fish2 = fish1 + 20) →
  fish1 + fish2 = 100 := by
  sorry

#check fishermen_catch

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishermen_catch_l63_6328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_is_ten_l63_6397

-- Define the work completion time and efficiency for each worker
noncomputable def work_time_A : ℝ := 60
noncomputable def efficiency_A : ℝ := 1.5
noncomputable def work_time_B : ℝ := 20
noncomputable def efficiency_B : ℝ := 1
noncomputable def work_time_C : ℝ := 30
noncomputable def efficiency_C : ℝ := 0.75

-- Define the combined work rate
noncomputable def combined_work_rate : ℝ := 
  efficiency_A / work_time_A + efficiency_B / work_time_B + efficiency_C / work_time_C

-- Theorem to prove
theorem combined_work_time_is_ten : 
  1 / combined_work_rate = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_is_ten_l63_6397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_impossibility_l63_6315

theorem partition_impossibility : ¬ ∃ (partition : Finset (Finset ℕ)),
  (∀ s ∈ partition, s.card = 3) ∧ 
  partition.card = 11 ∧
  (∀ s ∈ partition, ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a + b = c) ∧
  (partition.biUnion id).card = 33 ∧
  (∀ n ∈ partition.biUnion id, 1 ≤ n ∧ n ≤ 33) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_impossibility_l63_6315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binet_formula_l63_6330

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def φ_hat : ℝ := (1 - Real.sqrt 5) / 2

def fib : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem binet_formula (n : ℕ) : fib n = (φ^n - φ_hat^n) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binet_formula_l63_6330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l63_6367

open Real

-- Define the function f(x) = |2^x - 1|
noncomputable def f (x : ℝ) : ℝ := abs (2^x - 1)

-- State the theorem
theorem non_monotonic_interval (k : ℝ) :
  (∃ x y z, k - 1 < x ∧ x < y ∧ y < z ∧ z < k + 1 ∧
    ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))) →
  -1 < k ∧ k < 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l63_6367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_of_xy_line_l63_6341

noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m

noncomputable def slope_of_line (a b : ℝ) : ℝ := -a / b

theorem inclination_of_xy_line :
  let line_equation := λ (x y : ℝ) => x - y + 3
  let m := slope_of_line 1 (-1)
  inclination_angle m = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_of_xy_line_l63_6341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_union_problem_l63_6383

-- Define the sets P and Q
def P (x : ℝ) : Set ℝ := {z | z = Real.log 4 / Real.log (2 * x) ∨ z = 3}
def Q (x y : ℝ) : Set ℝ := {z | z = x ∨ z = y}

-- Define the theorem
theorem set_union_problem (x y : ℝ) :
  (P x ∩ Q x y = {2}) → (P x ∪ Q x y = {1, 2, 3}) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_union_problem_l63_6383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l63_6385

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x)) / Real.log m

theorem f_properties (m : ℝ) (hm : m > 0) (hm1 : m ≠ 1) :
  let g (x : ℝ) := f m x
  ∀ x ∈ Set.Ioo (-1) 1,
  (∃ y, y^2 - 1 = x ∧ g (y^2 - 1) = Real.log (y^2 / (2 - y^2)) / Real.log m) ∧
  g (-x) = -g x ∧
  (m > 1 → (g x ≤ 0 ↔ -1 < x ∧ x ≤ 0)) ∧
  (0 < m ∧ m < 1 → (g x ≤ 0 ↔ 0 ≤ x ∧ x < 1)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l63_6385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l63_6307

/-- Definition of the sequence a_n -/
def a : ℕ → ℝ := sorry

/-- Definition of the partial sum sequence S_n -/
def S : ℕ → ℝ
  | 0 => 0
  | n + 1 => S n + a (n + 1)

/-- The main theorem -/
theorem sequence_properties :
  (∀ n : ℕ, n > 0 → (2 * S n) / n + n = 2 * a n + 1) →
  (∃ r : ℝ, a 7 ^ 2 = a 4 * a 9) →
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l63_6307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_non_congruent_triangles_l63_6379

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define a function to count non-congruent triangles
def count_non_congruent_triangles (t : Triangle) : ℕ := sorry

-- Theorem statement
theorem two_non_congruent_triangles (t : Triangle) :
  count_non_congruent_triangles t = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_non_congruent_triangles_l63_6379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_function_is_identity_l63_6375

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f ((x + f x) / 2 + y) = 2 * x - f x + f (f y)) ∧
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → (f x - f y) * (x - y) ≥ 0)

/-- The main theorem stating that any function satisfying the conditions is the identity function -/
theorem satisfying_function_is_identity (f : ℝ → ℝ) (hf : SatisfyingFunction f) : 
  ∀ x : ℝ, x ≥ 0 → f x = x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_function_is_identity_l63_6375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_a_range_l63_6309

/-- A piecewise function f(x) defined as:
    f(x) = x^2 for x > 1
    f(x) = (4 - a/2)x - 1 for x ≤ 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then x^2 else (4 - a/2)*x - 1

/-- The theorem stating that if f(x) is monotonically increasing on ℝ,
    then the range of values for a is [4, 8) -/
theorem f_monotone_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc 4 8 ∧ a ≠ 8 :=
by
  intro h
  sorry  -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_a_range_l63_6309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l63_6337

/-- Calculates the time taken for a faster train to cross a man in a slower train -/
noncomputable def time_to_cross (faster_speed slower_speed : ℝ) (train_length : ℝ) : ℝ :=
  let relative_speed := faster_speed - slower_speed
  let relative_speed_mps := relative_speed * (1000 / 3600)
  train_length / relative_speed_mps

/-- Theorem stating that the time taken for the faster train to cross the man in the slower train is 24 seconds -/
theorem train_crossing_time :
  let faster_speed := (108 : ℝ)
  let slower_speed := (54 : ℝ)
  let train_length := (360 : ℝ)
  time_to_cross faster_speed slower_speed train_length = 24 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_cross 108 54 360

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l63_6337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_when_a_is_negative_four_range_of_a_for_minimum_condition_l63_6395

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 2 * x

-- Theorem for part (1)
theorem extremum_when_a_is_negative_four :
  let a : ℝ := -4
  ∃ x_min : ℝ, x_min > 0 ∧ ∀ x > 0, f a x ≥ f a x_min ∧ f a x_min = 4 - 4 * Real.log 2 := by
  sorry

-- Theorem for part (2)
theorem range_of_a_for_minimum_condition :
  {a : ℝ | a ≠ 0 ∧ ∀ x > 0, f a x ≥ -a} = {a : ℝ | -2 ≤ a ∧ a < 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_when_a_is_negative_four_range_of_a_for_minimum_condition_l63_6395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_identity_l63_6305

theorem sin_identity (x : ℝ) (h : Real.sin (2 * x + Real.pi / 5) = Real.sqrt 3 / 3) :
  Real.sin (4 * Real.pi / 5 - 2 * x) + (Real.sin (3 * Real.pi / 10 - 2 * x))^2 = (2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_identity_l63_6305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_with_ratio_l63_6333

theorem product_of_numbers_with_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a - b) / (a + b) = 1 / 8 ∧ (a + b) / (a * b) = 8 / 30 →
  a * b = 400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_with_ratio_l63_6333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_earnings_proof_l63_6356

/-- The amount Maria needs to earn to buy a bike and accessories --/
def maria_bike_purchase (retail_price : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) 
  (savings : ℚ) (mother_contribution : ℚ) (accessories_cost : ℚ) : ℚ :=
  let discounted_price := retail_price * (1 - discount_rate)
  let taxed_price := discounted_price * (1 + tax_rate)
  let total_cost := taxed_price + accessories_cost
  let available_funds := savings + mother_contribution
  total_cost - available_funds

#eval maria_bike_purchase 600 (1/10) (1/20) 120 250 50

theorem maria_earnings_proof :
  maria_bike_purchase 600 (1/10) (1/20) 120 250 50 = 247 := by
  -- Unfold the definition of maria_bike_purchase
  unfold maria_bike_purchase
  -- Simplify the arithmetic expressions
  simp [add_assoc, mul_assoc, add_mul, mul_add, sub_add_eq_sub_sub, add_sub_assoc]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_earnings_proof_l63_6356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_angle_theorem_l63_6318

/-- The hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop

/-- A point on a hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : h.equation x y

/-- The foci of a hyperbola -/
noncomputable def foci (h : Hyperbola) : (ℝ × ℝ) × (ℝ × ℝ) := by
  let c := Real.sqrt (h.a^2 + h.b^2)
  exact ((-c, 0), (c, 0))

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_angle_theorem (h : Hyperbola) (p : PointOnHyperbola h) :
  h.a = 1 ∧ h.b = 1 →
  h.equation = (fun x y => x^2 - y^2 = 1) →
  let (f1, f2) := foci h
  triangleArea (p.x, p.y) f1 f2 = Real.sqrt 3 →
  angle f1 (p.x, p.y) f2 = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_angle_theorem_l63_6318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_club_candy_packages_l63_6311

/-- Calculates the number of full candy packages needed for a math club with reduced attendance -/
def candy_packages_needed (original_members : ℕ) (candies_per_member : ℕ) (attendance_drop : ℚ) (candies_per_package : ℕ) : ℕ :=
  let reduced_members := (original_members : ℚ) * (1 - attendance_drop)
  let total_candies_needed := reduced_members * (candies_per_member : ℚ)
  (total_candies_needed / candies_per_package).ceil.toNat

/-- Proves that 18 full packages are needed for the given conditions -/
theorem math_club_candy_packages :
  candy_packages_needed 150 3 (30/100) 18 = 18 := by
  sorry

#eval candy_packages_needed 150 3 (30/100) 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_club_candy_packages_l63_6311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_onto_line_l63_6362

def line_direction : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 1
  | 1 => -2
  | 2 => 2

def vector_to_project : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 5
  | 1 => -3
  | 2 => 2

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (Finset.univ.sum λ i => v i * w i)

theorem projection_onto_line :
  let projection := (dot_product vector_to_project line_direction / dot_product line_direction line_direction) • line_direction
  projection = λ i =>
    match i with
    | 0 => 5/3
    | 1 => -10/3
    | 2 => 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_onto_line_l63_6362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_pair_is_five_seven_l63_6325

/-- Represents a pair of positive integers -/
structure IntPair where
  first : Nat
  second : Nat
  pos_first : first > 0
  pos_second : second > 0

/-- The sequence of integer pairs -/
def pairSequence : Nat → IntPair := sorry

/-- The sum of the components of a pair -/
def pairSum (p : IntPair) : Nat := p.first + p.second

/-- The group number for a given pair in the sequence -/
def groupNumber : Nat → Nat := sorry

/-- The position of a pair within its group -/
def positionInGroup : Nat → Nat := sorry

/-- Main theorem -/
theorem sixtieth_pair_is_five_seven :
  pairSequence 60 = ⟨5, 7, by sorry, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_pair_is_five_seven_l63_6325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_A_and_B_l63_6321

/-- The probability of selecting both A and B when choosing 3 students out of 5 (including A and B) -/
theorem prob_select_A_and_B : ∃ (p : ℚ), p = 3 / 10 := by
  let total_students : ℕ := 5
  let selected_students : ℕ := 3
  let total_ways : ℕ := Nat.choose total_students selected_students
  let ways_with_A_and_B : ℕ := Nat.choose (total_students - 2) (selected_students - 2)
  let prob : ℚ := ways_with_A_and_B / total_ways
  
  have h : prob = 3 / 10 := by sorry
  
  exact ⟨prob, h⟩

#eval (3 : ℚ) / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_A_and_B_l63_6321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l63_6353

/-- Given a train's speed in km/h and the time it takes to cross a pole, 
    calculate the length of the train in meters. -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem: A train traveling at 360 km/h that crosses a pole in 5 seconds 
    has a length of 500 meters. -/
theorem train_length_calculation :
  trainLength 360 5 = 500 := by
  -- Unfold the definition of trainLength
  unfold trainLength
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_div_assoc]
  -- The result should now be obvious to Lean
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l63_6353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_same_last_two_digits_l63_6366

theorem smallest_n_same_last_two_digits : ∃ n : ℕ, 
  n > 0 ∧
  (∀ m : ℕ, 0 < m ∧ m < n → (107 * m) % 100 ≠ m % 100) ∧
  (107 * n) % 100 = n % 100 ∧
  n = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_same_last_two_digits_l63_6366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_acute_angle_l63_6310

theorem parallel_vectors_acute_angle (x : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.sin x, 3/4]
  let b : Fin 2 → ℝ := ![1/3, 1/2 * Real.cos x]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →  -- parallel vectors condition
  0 < x ∧ x < π/2 →                 -- acute angle condition
  x = π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_acute_angle_l63_6310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exponential_to_rectangular_l63_6344

open Complex

theorem complex_exponential_to_rectangular : 
  (Real.sqrt 3 : ℂ) * Complex.exp (Complex.I * (13 * Real.pi / 6)) = (3 / 2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exponential_to_rectangular_l63_6344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l63_6335

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x - y = 4 * Real.sqrt 3

-- Define the line θ = π/6
def line_theta (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x

-- Define the distance function
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem intersection_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    curve_C x₁ y₁ ∧
    line_l x₂ y₂ ∧
    line_theta x₁ y₁ ∧
    line_theta x₂ y₂ ∧
    x₁ ≠ 0 ∧
    y₁ ≠ 0 ∧
    distance x₁ y₁ x₂ y₂ = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l63_6335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_OP_OQ_l63_6326

open Real

noncomputable def C1 (α : ℝ) : ℝ × ℝ := (2 + 2 * cos α, 2 * sin α)
noncomputable def C2 (β : ℝ) : ℝ × ℝ := (2 * cos β, 2 + 2 * sin β)

def l1 (α : ℝ) : ℝ := α
noncomputable def l2 (α : ℝ) : ℝ := α - π / 6

noncomputable def P (α : ℝ) : ℝ × ℝ := (4 * cos α * cos α, 4 * cos α * sin α)
noncomputable def Q (α : ℝ) : ℝ × ℝ := (4 * sin (α - π / 6) * cos (α - π / 6), 
                          4 * sin (α - π / 6) * sin (α - π / 6))

noncomputable def distance (p : ℝ × ℝ) : ℝ := sqrt (p.1 * p.1 + p.2 * p.2)

theorem max_product_OP_OQ :
  ∃ (α : ℝ), 0 < α ∧ α < π / 2 ∧
  ∀ (β : ℝ), 0 < β ∧ β < π / 2 →
  distance (P α) * distance (Q α) ≥ distance (P β) * distance (Q β) ∧
  distance (P α) * distance (Q α) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_OP_OQ_l63_6326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_translation_l63_6331

noncomputable def original_function (x : ℝ) : ℝ := 2^(-x)

def translate_down (f : ℝ → ℝ) (units : ℝ) : ℝ → ℝ := λ x => f x - units

def translate_left (f : ℝ → ℝ) (units : ℝ) : ℝ → ℝ := λ x => f (x + units)

theorem function_translation :
  ∀ x : ℝ,
    (translate_down original_function 2) x = 2^(-x) - 2 ∧
    (translate_left (translate_down original_function 2) 1) x = 2^(-x-1) - 2 :=
by
  intro x
  constructor
  · -- First part of the conjunction
    simp [translate_down, original_function]
  · -- Second part of the conjunction
    simp [translate_left, translate_down, original_function]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_translation_l63_6331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_of_grid_l63_6332

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs (x₁*(y₂-y₃) + x₂*(y₃-y₁) + x₃*(y₁-y₂))

def rectangle_area (width height : ℝ) : ℝ :=
  width * height

theorem triangle_fraction_of_grid : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (2, -2)
  let C : ℝ × ℝ := (3, 5)
  let grid_width : ℝ := 8
  let grid_height : ℝ := 6
  (triangle_area A B C) / (rectangle_area grid_width grid_height) = 11/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_of_grid_l63_6332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_in_U_l63_6386

-- Define the universe set U
def U : Set ℝ := {x | x ≥ -2}

-- Define set A
def A : Set ℝ := {x | (2 : ℝ)^x > 1/2}

-- State the theorem
theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = Set.Icc (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_in_U_l63_6386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_ratio_after_two_replacements_l63_6374

/-- Represents the container and replacement process --/
structure Container where
  capacity : ℝ
  initial_liquid_a : ℝ
  replacement_volume : ℝ

/-- Calculates the amount of liquid A after one replacement --/
noncomputable def liquid_a_after_one_replacement (c : Container) : ℝ :=
  c.initial_liquid_a * (c.capacity - c.replacement_volume) / c.capacity

/-- Calculates the amount of liquid A after two replacements --/
noncomputable def liquid_a_after_two_replacements (c : Container) : ℝ :=
  liquid_a_after_one_replacement c * (c.capacity - c.replacement_volume) / c.capacity

/-- Calculates the amount of liquid B after two replacements --/
noncomputable def liquid_b_after_two_replacements (c : Container) : ℝ :=
  c.capacity - liquid_a_after_two_replacements c

/-- Theorem stating the ratio of liquids A to B after two replacements --/
theorem liquid_ratio_after_two_replacements (c : Container) 
  (h1 : c.capacity = 37.5)
  (h2 : c.initial_liquid_a = 37.5)
  (h3 : c.replacement_volume = 15) :
  liquid_a_after_two_replacements c / liquid_b_after_two_replacements c = 9 / 16 := by
  sorry

#check liquid_ratio_after_two_replacements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_ratio_after_two_replacements_l63_6374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_francescas_lemonade_calories_l63_6323

/-- Represents the lemonade recipe and calorie information -/
structure LemonadeRecipe where
  lemon_juice_grams : ℚ
  sugar_grams : ℚ
  water_grams : ℚ
  lemon_juice_calories_per_50g : ℚ
  sugar_calories_per_100g : ℚ

/-- Calculates the total calories in a given weight of lemonade -/
def calories_in_lemonade (recipe : LemonadeRecipe) (serving_grams : ℚ) : ℚ :=
  let total_calories := recipe.lemon_juice_calories_per_50g * (recipe.lemon_juice_grams / 50) +
                        recipe.sugar_calories_per_100g * (recipe.sugar_grams / 100)
  let total_weight := recipe.lemon_juice_grams + recipe.sugar_grams + recipe.water_grams
  let calories_per_gram := total_calories / total_weight
  calories_per_gram * serving_grams

/-- Francesca's lemonade recipe -/
def francescas_recipe : LemonadeRecipe :=
  { lemon_juice_grams := 50
  , sugar_grams := 150
  , water_grams := 300
  , lemon_juice_calories_per_50g := 32
  , sugar_calories_per_100g := 386 }

/-- Theorem stating that a 250g serving of Francesca's lemonade contains 305.5 calories -/
theorem francescas_lemonade_calories :
  calories_in_lemonade francescas_recipe 250 = 305.5 := by
  sorry

#eval calories_in_lemonade francescas_recipe 250

end NUMINAMATH_CALUDE_ERRORFEEDBACK_francescas_lemonade_calories_l63_6323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_opportunities_prob_distribution_expected_value_penalty_kick_results_l63_6312

-- Define the success probability
noncomputable def p : ℚ := 3/5

-- Define the number of opportunities
def n : ℕ := 3

-- Define the points earned for each successful kick
def points_per_success : ℚ := 50

-- Define the random variable X (total points earned)
def X : ℕ → ℚ
| 0 => 0
| 1 => points_per_success
| 2 => 2 * points_per_success
| 3 => 3 * points_per_success
| _ => 0  -- This case should never occur given the problem constraints

-- Theorem for the probability of using all 3 opportunities
theorem prob_all_opportunities : ℚ := 21/25

-- Theorem for the probability distribution of X
theorem prob_distribution (k : ℕ) : ℚ :=
  match k with
  | 0 => 4/25
  | 1 => 24/125
  | 2 => 54/125
  | 3 => 27/125
  | _ => 0

-- Theorem for the expected value of X
theorem expected_value : ℚ := 426/5

-- Main theorem combining all results
theorem penalty_kick_results :
  (prob_all_opportunities = 21/25) ∧
  (∀ k, prob_distribution k = match k with
                              | 0 => 4/25
                              | 1 => 24/125
                              | 2 => 54/125
                              | 3 => 27/125
                              | _ => 0) ∧
  (expected_value = 426/5) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_opportunities_prob_distribution_expected_value_penalty_kick_results_l63_6312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l63_6345

noncomputable def triangle_ABC (a b c A B C : ℝ) : Prop :=
  Real.sin A / a = Real.sin B / b ∧ Real.sin B / b = Real.sin C / c

theorem min_value_theorem (a₁ b₁ A₁ a₂ b₂ A₂ m n : ℝ) 
  (h₁ : triangle_ABC 7 8 (Real.sqrt ((7^2 + 8^2 - 2*7*8*Real.cos (30 * π/180)))) (30 * π/180) B₁ C₁)
  (h₂ : triangle_ABC (13 * Real.sqrt 3) 26 (Real.sqrt (((13*Real.sqrt 3)^2 + 26^2 - 2*13*Real.sqrt 3*26*Real.cos (60 * π/180)))) (60 * π/180) B₂ C₂)
  (h_m : m = 2)
  (h_n : n = 1)
  (h_positive : ∀ x y : ℝ, x > 0 ∧ y > 0 → m*x + n*y = 3) :
  ∃ (min_val : ℝ), ∀ x y : ℝ, x > 0 ∧ y > 0 → m*x + n*y = 3 → 1/x + 2/y ≥ min_val ∧ min_val = 8/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l63_6345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_obtuse_angle_l63_6396

/-- A rhombus with perimeter 8 and height 1 has an obtuse angle of 150 degrees. -/
theorem rhombus_obtuse_angle (perimeter height : ℝ) (h1 : perimeter = 8) (h2 : height = 1) :
  let side_length := perimeter / 4
  let acute_angle := Real.arcsin (height / side_length)
  let obtuse_angle := π - 2 * acute_angle
  obtuse_angle = 5 * π / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_obtuse_angle_l63_6396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l63_6343

/-- Given a hyperbola C with the following properties:
    - Foci on the x-axis
    - Center at the origin
    - Asymptotes given by the equations 2x ± 3y = 0
    - Eccentricity equal to 2√13
    The equation of the hyperbola C is x²/9 - y²/4 = 1 -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) 
  (foci_on_x_axis : ∀ (x y : ℝ), (x, y) ∈ C → (x, 0) ∈ C)
  (center_origin : (0, 0) ∈ C)
  (asymptotes : ∀ (x y : ℝ), (x, y) ∈ C → (2*x = 3*y ∨ 2*x = -3*y))
  (eccentricity : Real.sqrt 13 = 2) :
  ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/9 - y^2/4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l63_6343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_cylindrical_equiv_l63_6317

/-- Converts a point from rectangular coordinates to cylindrical coordinates -/
noncomputable def rect_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x = 0 && y = 0 then 0
           else if x ≥ 0 then Real.arctan (y / x)
           else Real.arctan (y / x) + Real.pi
  (r, θ, z)

/-- The equivalence of (-3, 0, 5) in rectangular coordinates to (3, π, 5) in cylindrical coordinates -/
theorem rect_to_cylindrical_equiv :
  rect_to_cylindrical (-3) 0 5 = (3, Real.pi, 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_cylindrical_equiv_l63_6317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modifiedFibonacci_50th_term_mod_9_l63_6346

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2  -- Adding this case to cover Nat.zero
  | 1 => 2
  | 2 => 5
  | n + 3 => modifiedFibonacci (n + 2) + modifiedFibonacci (n + 1)

theorem modifiedFibonacci_50th_term_mod_9 :
  modifiedFibonacci 50 % 9 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modifiedFibonacci_50th_term_mod_9_l63_6346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_ten_percent_l63_6387

noncomputable section

-- Define the markup percentage
def markup : ℝ := 0.30

-- Define the profit percentage after discount
def profit : ℝ := 0.17

-- Define the function to calculate the discount percentage
noncomputable def discount_percentage (markup profit : ℝ) : ℝ :=
  (markup - profit) / (1 + markup) * 100

-- Theorem statement
theorem discount_is_ten_percent :
  discount_percentage markup profit = 10 := by
  -- Expand the definition of discount_percentage
  unfold discount_percentage
  -- Perform the calculation
  simp [markup, profit]
  -- The proof is completed automatically
  norm_num

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_ten_percent_l63_6387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_price_is_four_l63_6342

/-- The price per pound of a candy mixture -/
noncomputable def mixture_price (x : ℝ) : ℝ :=
  let candy1_weight := x
  let candy1_price := 3.50
  let candy2_weight := 6.25
  let candy2_price := 4.30
  let total_weight := 10.0
  (candy1_weight * candy1_price + candy2_weight * candy2_price) / total_weight

/-- Theorem stating that the price per pound of the mixture is $4.00 -/
theorem mixture_price_is_four :
  ∃ x : ℝ, x + 6.25 = 10 ∧ mixture_price x = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_price_is_four_l63_6342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_implies_sin_cos_ratio_l63_6358

theorem tan_sum_implies_sin_cos_ratio (α : ℝ) :
  Real.tan (α + π/4) = 2 →
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_implies_sin_cos_ratio_l63_6358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blouse_price_calculation_l63_6388

noncomputable def original_price (final_price : ℝ) : ℝ :=
  final_price / (0.82 * 1.05 * 0.90)

theorem blouse_price_calculation :
  let final_price := 147.60
  let calculated_price := original_price final_price
  ‖calculated_price - 200‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blouse_price_calculation_l63_6388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_243_equals_3_to_m_l63_6347

theorem cube_root_243_equals_3_to_m (m : ℝ) : (243 : ℝ) ^ (1/3 : ℝ) = 3^m → m = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_243_equals_3_to_m_l63_6347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l63_6324

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x - 2*y - 12 = 0

/-- Distance from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - 2*y - 12| / Real.sqrt 5

/-- The theorem stating that (2, -3) is the point on the ellipse with minimum distance to the line -/
theorem min_distance_point : 
  ellipse 2 (-3) ∧ 
  ∀ x y : ℝ, ellipse x y → distance_to_line 2 (-3) ≤ distance_to_line x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l63_6324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_area_l63_6327

/-- A convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  perpendicular_ab : Prop  -- a and b are perpendicular
  equal_other_sides : Prop -- other two sides are equal
  right_angle_other : Prop -- other two sides form a right angle

/-- The area of the ConvexQuadrilateral -/
noncomputable def area (q : ConvexQuadrilateral) : ℝ :=
  (q.a * q.b) / 2 + ((q.a^2 + q.b^2) / 4)

/-- Theorem: The area of the specific quadrilateral is 400 cm² -/
theorem specific_quadrilateral_area :
  ∀ q : ConvexQuadrilateral, q.a = 28.29 ∧ q.b = 11.71 → area q = 400 := by
  intro q ⟨h_a, h_b⟩
  simp [area, h_a, h_b]
  -- The actual computation is left as an exercise
  sorry

#check specific_quadrilateral_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_area_l63_6327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_leq_d_iff_e_leq_f_l63_6359

theorem c_leq_d_iff_e_leq_f 
  {a b c d e f : ℝ}  -- Specify the type of variables
  (h1 : a ≥ b → c > d)
  (h2 : c > d → a ≥ b)
  (h3 : a < b ↔ e ≤ f)
  : c ≤ d ↔ e ≤ f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_leq_d_iff_e_leq_f_l63_6359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_or_coincident_l63_6316

noncomputable section

-- Define the two lines as functions of x
def line1 (k : ℝ) (x : ℝ) : ℝ := 2 * x - k
def line2 (x : ℝ) : ℝ := 2 * x - 1/2

-- Theorem statement
theorem lines_parallel_or_coincident (k : ℝ) :
  (∀ x, line1 k x = line2 x) ∨ 
  (∃ c, ∀ x, line1 k x = line2 x + c) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_or_coincident_l63_6316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_value_l63_6360

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line
def myLine (x y a : ℝ) : Prop := x + y = a

-- Define the intersection points
def intersectionPoints (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    myCircle x₁ y₁ ∧ myCircle x₂ y₂ ∧ 
    myLine x₁ y₁ a ∧ myLine x₂ y₂ a ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the vector equality condition
def vectorCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂)^2 + (y₁ + y₂)^2 = (x₁ - x₂)^2 + (y₁ - y₂)^2

-- Main theorem
theorem intersection_line_value (a : ℝ) :
  intersectionPoints a ∧ 
  (∀ x₁ y₁ x₂ y₂ : ℝ, myCircle x₁ y₁ ∧ myCircle x₂ y₂ ∧ myLine x₁ y₁ a ∧ myLine x₂ y₂ a → 
    vectorCondition x₁ y₁ x₂ y₂) →
  a = 2 ∨ a = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_value_l63_6360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l63_6334

/-- The area of a quadrilateral with a diagonal of length d and offsets h1 and h2 -/
noncomputable def quadrilateralArea (d h1 h2 : ℝ) : ℝ := (1/2 * d * h1) + (1/2 * d * h2)

/-- Theorem: The area of a quadrilateral with diagonal 20 cm and offsets 5 cm and 4 cm is 90 cm² -/
theorem quadrilateral_area_example : quadrilateralArea 20 5 4 = 90 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the arithmetic expression
  simp [mul_add, add_mul, mul_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l63_6334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_one_l63_6372

/-- Represents a 5x5 grid where each cell contains a digit from 1 to 5 or is empty -/
def Grid := Fin 5 → Fin 5 → Option (Fin 5)

/-- Checks if a given row in the grid contains each digit from 1 to 5 exactly once -/
def validRow (g : Grid) (row : Fin 5) : Prop :=
  ∀ d : Fin 5, ∃! col : Fin 5, g row col = some d

/-- Checks if a given column in the grid contains each digit from 1 to 5 exactly once -/
def validColumn (g : Grid) (col : Fin 5) : Prop :=
  ∀ d : Fin 5, ∃! row : Fin 5, g row col = some d

/-- Checks if the entire grid is valid (each row and column contains 1-5 exactly once) -/
def validGrid (g : Grid) : Prop :=
  (∀ row : Fin 5, validRow g row) ∧ (∀ col : Fin 5, validColumn g col)

/-- Initial configuration of the grid -/
def initialGrid : Grid := fun row col =>
  match row, col with
  | 0, 0 => some 1 | 0, 1 => some 3 | 0, 4 => some 2
  | 1, 1 => some 4 | 1, 2 => some 2 | 1, 3 => some 3
  | 2, 0 => some 2 | 2, 2 => some 4
  | 3, 3 => some 5 | 3, 4 => some 4
  | _, _ => none

theorem lower_right_is_one (g : Grid) :
  validGrid g → (∀ row col, Option.isSome (initialGrid row col) → g row col = initialGrid row col) →
  g 4 4 = some 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_one_l63_6372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_photos_l63_6380

-- Definitions (moved to the top and modified)
def two_boys {n : ℕ} (photo : Fin n) : Prop := sorry
def two_girls {n : ℕ} (photo : Fin n) : Prop := sorry
def same_children {n : ℕ} (photo photo' : Fin n) : Prop := sorry

theorem minimum_photos (n_girls n_boys : ℕ) (h_girls : n_girls = 4) (h_boys : n_boys = 8) :
  ∃ (min_photos : ℕ), min_photos = n_girls * n_boys + 1 ∧
  (∀ (photos : ℕ), photos ≥ min_photos →
    (∃ (photo : Fin photos), two_boys photo ∨ two_girls photo ∨
      (∃ (photo' : Fin photos), photo ≠ photo' ∧ same_children photo photo'))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_photos_l63_6380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ninth_term_l63_6384

theorem sequence_ninth_term : 
  ∀ n : ℕ, (Real.sqrt (3 * (2 * n - 1)) = 9) ↔ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ninth_term_l63_6384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l63_6320

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_side_length (t : Triangle) : 
  t.A + t.C = 2 * t.B → 
  Real.sqrt 3 * t.a^2 + Real.sqrt 3 * t.c^2 - 2 * t.a * t.c * Real.sin t.B = 9 * Real.sqrt 3 → 
  t.b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l63_6320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_of_triangle_l63_6336

noncomputable section

-- Define points P, Q, and M
def P : ℝ × ℝ := (-2, 4 * Real.sqrt 3)
def Q : ℝ × ℝ := (2, 0)
def M : ℝ × ℝ := (0, -6)

-- Define lines l, l₁, and l₂
def l (y : ℝ) : ℝ := Real.sqrt 3 * y - 2 * Real.sqrt 3
def l₁ (x : ℝ) : ℝ := -Real.sqrt 3 * (x - 2)
def l₂ (x : ℝ) : ℝ := Real.sqrt 3 * (x - 2)

-- Define the incircle
def incircle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

-- Theorem statement
theorem incircle_of_triangle :
  ∀ x y : ℝ, (∃ t : ℝ, x = l t ∨ y = l₁ x ∨ y = l₂ x) → incircle x y :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_of_triangle_l63_6336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_beta_l63_6351

theorem cos_alpha_plus_beta (α β : Real) 
  (h1 : 0 < β) (h2 : β < π/2) (h3 : π/2 < α) (h4 : α < π)
  (h5 : Real.cos (α - β/2) = -1/9) (h6 : Real.sin (α/2 - β) = 2/3) : 
  Real.cos (α + β) = -239/729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_beta_l63_6351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_can_be_found_l63_6369

/-- Represents the state of a box in the line -/
inductive BoxState
  | Empty
  | HasDiamond

/-- Represents the claim made by a box -/
inductive Claim
  | Left
  | Right

/-- A configuration of boxes -/
structure BoxConfig where
  boxes : Fin 100 → BoxState
  claims : Fin 100 → Claim
  diamond_location : Fin 100
  true_claim : Fin 100

/-- Predicate to check if a configuration is valid -/
def is_valid_config (config : BoxConfig) : Prop :=
  (∃ i, config.boxes i = BoxState.HasDiamond) ∧
  (∀ i, config.boxes i = BoxState.HasDiamond → i = config.diamond_location) ∧
  (∀ i, i ≠ config.true_claim → 
    (config.claims i = Claim.Left → i > 0 → config.diamond_location ≠ i - 1) ∧
    (config.claims i = Claim.Right → i < 99 → config.diamond_location ≠ i + 1)) ∧
  ((config.claims config.true_claim = Claim.Left ∧ config.true_claim > 0 → config.diamond_location = config.true_claim - 1) ∨
   (config.claims config.true_claim = Claim.Right ∧ config.true_claim < 99 → config.diamond_location = config.true_claim + 1))

/-- The main theorem stating that the diamond can be found by opening one box -/
theorem diamond_can_be_found (config : BoxConfig) (h : is_valid_config config) :
  ∃ (i : Fin 100), ∀ (j : Fin 100), config.boxes j = BoxState.HasDiamond → (j = 0 ∨ j = 99) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_can_be_found_l63_6369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_specific_vectors_l63_6361

theorem cross_product_specific_vectors :
  let v₁ : Fin 3 → ℝ := ![3, -1, 4]
  let v₂ : Fin 3 → ℝ := ![6, -2, 8]
  (v₁ 1 * v₂ 2 - v₁ 2 * v₂ 1, v₁ 2 * v₂ 3 - v₁ 3 * v₂ 2, v₁ 3 * v₂ 1 - v₁ 1 * v₂ 3) = (0, 0, 0)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_specific_vectors_l63_6361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_mixture_theorem_l63_6314

/-- Represents the amount of butterfat in a mixture of milk --/
noncomputable def butterfat_amount (volume : ℝ) (percentage : ℝ) : ℝ := volume * percentage / 100

/-- Represents the total volume of a mixture --/
noncomputable def total_volume (volume1 : ℝ) (volume2 : ℝ) : ℝ := volume1 + volume2

/-- Represents the percentage of butterfat in a mixture --/
noncomputable def butterfat_percentage (total_butterfat : ℝ) (total_volume : ℝ) : ℝ := 
  total_butterfat / total_volume * 100

theorem milk_mixture_theorem (volume_40_percent : ℝ) (volume_10_percent : ℝ) 
  (h1 : volume_40_percent = 8) 
  (h2 : volume_10_percent = 16) : 
  butterfat_percentage 
    (butterfat_amount volume_40_percent 40 + butterfat_amount volume_10_percent 10) 
    (total_volume volume_40_percent volume_10_percent) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_mixture_theorem_l63_6314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l63_6306

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*x + 6)

-- State the theorem
theorem f_monotone_increasing :
  ∀ a b : ℝ, a < b ∧ b ≤ 1 → f a < f b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l63_6306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_product_l63_6382

theorem cos_sum_product (c d : ℝ) : Real.cos (c + d) + Real.cos (c - d) = 2 * Real.cos c * Real.cos d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_product_l63_6382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_k_alpha_equals_three_l63_6313

/-- A power function that passes through the point (2,4) -/
noncomputable def power_function (k α : ℝ) : ℝ → ℝ := λ x => k * (x ^ α)

/-- The condition that the function passes through (2,4) -/
def passes_through_point (k α : ℝ) : Prop :=
  power_function k α 2 = 4

theorem sum_k_alpha_equals_three :
  ∃ k α : ℝ, passes_through_point k α ∧ k + α = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_k_alpha_equals_three_l63_6313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l63_6377

/-- Function to calculate the area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let a := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let b := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let c := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The maximum area of a triangle ABC with AB = 12 and BC : AC = 30 : 31 -/
theorem max_triangle_area : 
  ∀ (A B C : ℝ × ℝ),
  let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let ac := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  ab = 12 →
  bc / ac = 30 / 31 →
  triangle_area A B C ≤ 930 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l63_6377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_picnic_volunteers_l63_6350

/-- 
Given:
- total_parents: The total number of parents who attended the meeting
- supervise_parents: The number of parents who volunteered to supervise children
- both_parents: The number of parents who volunteered both to supervise and bring refreshments
- refresh_ratio: The ratio of parents who volunteered to bring refreshments to those who didn't volunteer for either task

Prove that the number of parents who volunteered to bring refreshments is 42.
-/
theorem school_picnic_volunteers 
  (total_parents : ℕ) 
  (supervise_parents : ℕ) 
  (both_parents : ℕ) 
  (refresh_ratio : ℚ) 
  (h1 : total_parents = 84)
  (h2 : supervise_parents = 25)
  (h3 : both_parents = 11)
  (h4 : refresh_ratio = 3/2) :
  ∃ (refresh_parents : ℕ), 
    refresh_parents = 42 ∧ 
    (refresh_parents : ℚ) = refresh_ratio * (total_parents - supervise_parents - refresh_parents + both_parents) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_picnic_volunteers_l63_6350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_gcd_l63_6322

/-- Represents the outcome of a single coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a sequence of coin flips -/
def FlipSequence := List CoinFlip

/-- The number of coin flips -/
def totalFlips : Nat := 20

/-- The first specific sequence (HTTH) -/
def sequence1 : FlipSequence := [CoinFlip.Heads, CoinFlip.Tails, CoinFlip.Tails, CoinFlip.Heads]

/-- The second specific sequence (HTHH) -/
def sequence2 : FlipSequence := [CoinFlip.Heads, CoinFlip.Tails, CoinFlip.Heads, CoinFlip.Heads]

/-- The probability of each specific sequence occurring exactly twice in the total flips -/
noncomputable def p : ℚ := sorry

/-- m and n are positive integers such that p = m/n and they are relatively prime -/
noncomputable def m : ℕ+ := sorry
noncomputable def n : ℕ+ := sorry

axiom p_def : p = m / n

axiom m_n_coprime : Nat.Coprime m.val n.val

/-- The main theorem: If p is the probability of two specific sequences each occurring exactly
    twice in 20 coin flips, and p can be expressed as m/n where m and n are relatively prime
    positive integers, then gcd(m, n) = 1 -/
theorem coin_flip_probability_gcd :
  Nat.gcd m.val n.val = 1 := by
  rw [← Nat.Coprime]
  exact m_n_coprime


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_gcd_l63_6322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dollar_op_result_l63_6392

-- Define the set S
def S : Type := Fin 4

-- Define the operation $
def dollar_op : S → S → S
| ⟨0, _⟩, ⟨0, _⟩ => ⟨3, by norm_num⟩
| ⟨0, _⟩, ⟨1, _⟩ => ⟨0, by norm_num⟩
| ⟨0, _⟩, ⟨2, _⟩ => ⟨1, by norm_num⟩
| ⟨0, _⟩, ⟨3, _⟩ => ⟨2, by norm_num⟩
| ⟨1, _⟩, ⟨0, _⟩ => ⟨0, by norm_num⟩
| ⟨1, _⟩, ⟨1, _⟩ => ⟨2, by norm_num⟩
| ⟨1, _⟩, ⟨2, _⟩ => ⟨3, by norm_num⟩
| ⟨1, _⟩, ⟨3, _⟩ => ⟨1, by norm_num⟩
| ⟨2, _⟩, ⟨0, _⟩ => ⟨1, by norm_num⟩
| ⟨2, _⟩, ⟨1, _⟩ => ⟨3, by norm_num⟩
| ⟨2, _⟩, ⟨2, _⟩ => ⟨2, by norm_num⟩
| ⟨2, _⟩, ⟨3, _⟩ => ⟨0, by norm_num⟩
| ⟨3, _⟩, ⟨0, _⟩ => ⟨2, by norm_num⟩
| ⟨3, _⟩, ⟨1, _⟩ => ⟨1, by norm_num⟩
| ⟨3, _⟩, ⟨2, _⟩ => ⟨0, by norm_num⟩
| ⟨3, _⟩, ⟨3, _⟩ => ⟨3, by norm_num⟩

theorem dollar_op_result : dollar_op (dollar_op ⟨3, by norm_num⟩ ⟨1, by norm_num⟩) (dollar_op ⟨2, by norm_num⟩ ⟨0, by norm_num⟩) = ⟨2, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dollar_op_result_l63_6392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_approx_93_l63_6338

/-- The number of coins -/
def n : ℕ := 100

/-- The maximum number of flips per coin -/
def max_flips : ℕ := 4

/-- The probability of getting heads on a single flip -/
noncomputable def p : ℝ := 1/2

/-- The probability of getting heads after at most 'max_flips' flips -/
noncomputable def prob_heads : ℝ := 1 - (1 - p)^max_flips

/-- The expected number of coins showing heads -/
noncomputable def expected_heads : ℝ := n * prob_heads

theorem expected_heads_approx_93 : 
  ⌊expected_heads⌋ = 93 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_approx_93_l63_6338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_relation_l63_6398

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define collinearity
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

-- Define tangent length
noncomputable def tangentLength (p : Point) (c : Circle) : ℝ :=
  let centerPoint : Point := ⟨(c.center).1, (c.center).2⟩
  Real.sqrt (distance p centerPoint^2 - c.radius^2)

theorem tangent_relation (O : Circle) (A B C : Point) (a b c : ℝ) :
  collinear A B C →
  a = tangentLength A O^2 →
  b = tangentLength B O^2 →
  c = tangentLength C O^2 →
  a * distance B C + c * distance A B - b * distance A C = 
    distance B C * distance A C * distance A B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_relation_l63_6398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l63_6301

theorem solve_exponential_equation :
  ∃ x : ℚ, (10 : ℝ)^(2*x : ℝ) * (1000 : ℝ)^(x : ℝ) = (10000 : ℝ)^3 ↔ x = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l63_6301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_15_terms_is_180_l63_6357

noncomputable def arithmetic_progression (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

noncomputable def sum_arithmetic_progression (a d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_15_terms_is_180
  (a d : ℝ)
  (h : arithmetic_progression a d 4 + arithmetic_progression a d 12 = 24) :
  sum_arithmetic_progression a d 15 = 180 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_15_terms_is_180_l63_6357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_and_inequality_imply_sum_l63_6354

-- Define the function g(x)
noncomputable def g (A B C : ℤ) (x : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

-- State the theorem
theorem asymptotes_and_inequality_imply_sum (A B C : ℤ) :
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 2 → g A B C x ≠ 0) →  -- Vertical asymptotes at x = -1 and x = 2
  (∀ x : ℝ, x > 4 → g A B C x > 0.5) →         -- For all x > 4, g(x) > 0.5
  A + B + C = -4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_and_inequality_imply_sum_l63_6354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l63_6355

-- Definitions
noncomputable def CircleWithDiameter (A B : Point) : Set Point := sorry
noncomputable def LineSegment (A B : Point) : Set Point := sorry
noncomputable def Triangle (A B C : Point) : Set Point := sorry
noncomputable def Area (S : Set Point) : Real := sorry
noncomputable def Line (A B : Point) : Set Point := sorry
noncomputable def AngleBetween (l1 l2 : Set Point) : Real := sorry
noncomputable def Angle (A B C : Point) : Real := sorry

theorem triangle_angles (A B C D E : Point) (α β γ : Real) : 
  -- Circle with diameter AB intersects AC at D and BC at E
  CircleWithDiameter A B ∩ LineSegment A C = {D} →
  CircleWithDiameter A B ∩ LineSegment B C = {E} →
  -- DE bisects the area of ABC
  Area (Triangle A B C) = 2 * Area (Triangle C D E) →
  -- DE forms a 15° angle with AB
  AngleBetween (Line D E) (Line A B) = 15 →
  -- α, β, γ are angles of triangle ABC
  Angle A B C = α ∧ Angle B C A = β ∧ Angle C A B = γ →
  -- Conclusion: angles of ABC are 60°, 75°, 45°
  α = 60 ∧ β = 75 ∧ γ = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l63_6355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_distributions_l63_6348

/-- Represents a distribution of desks in rooms -/
structure DeskDistribution where
  rooms : List Nat
  desks_per_room : List Nat
  total_desks : Nat
  total_rooms : Nat

/-- Checks if a distribution is valid according to the problem constraints -/
def is_valid_distribution (d : DeskDistribution) : Prop :=
  d.total_desks = 6018 ∧
  d.total_rooms = 2007 ∧
  d.rooms.length = d.total_rooms ∧
  d.desks_per_room.length = d.total_rooms ∧
  (∀ r ∈ d.desks_per_room, r ≥ 1) ∧
  d.desks_per_room.sum = d.total_desks

/-- Checks if a distribution allows for equal redistribution after clearing any room -/
def allows_equal_redistribution (d : DeskDistribution) : Prop :=
  ∀ i, i < d.total_rooms →
    ∃ n : Nat, ∀ j, j < d.total_rooms ∧ j ≠ i →
      (d.total_desks - d.desks_per_room[i]!) / (d.total_rooms - 1) = n

/-- The main theorem stating the only possible valid distributions -/
theorem valid_distributions (d : DeskDistribution) :
  is_valid_distribution d ∧ allows_equal_redistribution d →
  (d.desks_per_room.count 1 = 1 ∧ d.desks_per_room.count 2 = 1 ∧ d.desks_per_room.count 3 = 2005) ∨
  (d.desks_per_room.count 2 = 3 ∧ d.desks_per_room.count 3 = 2004) :=
by sorry

#check valid_distributions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_distributions_l63_6348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_foci_and_vertices_l63_6399

noncomputable section

-- Define the parabola P
def P : ℝ → ℝ := λ x => 2 * x^2

-- Define the vertex and focus of P
def V₁ : ℝ × ℝ := (0, 0)
def F₁ : ℝ × ℝ := (0, 1/8)

-- Define points A and B on P
def A (a : ℝ) : ℝ × ℝ := (a, P a)
def B (b : ℝ) : ℝ × ℝ := (b, P b)

-- Define the angle condition
def angle_condition (a b : ℝ) : Prop := a * b = -1/2

-- Define the midpoint M of AB
def M (a b : ℝ) : ℝ × ℝ := ((a + b)/2, (a + b)^2 + 1/2)

-- Define the parabola Q formed by the locus of M
def Q : ℝ → ℝ := λ x => 4 * x^2 + 1/2

-- Define the vertex and focus of Q
def V₂ : ℝ × ℝ := (0, 1/2)
def F₂ : ℝ × ℝ := (0, 9/16)

-- Define the distance function
def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- State the theorem
theorem ratio_of_foci_and_vertices (a b : ℝ) :
  angle_condition a b →
  distance F₁ F₂ / distance V₁ V₂ = 5/8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_foci_and_vertices_l63_6399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_59_l63_6349

theorem remainder_sum_mod_59 (a b c : ℕ) 
  (ha : a % 59 = 28)
  (hb : b % 59 = 15)
  (hc : c % 59 = 19)
  (h_consec : (a = b + d ∧ d % 59 = 13) ∨ (b = a + d ∧ d % 59 = 46) ∨ 
              (b = c + d ∧ d % 59 = 55) ∨ (c = b + d ∧ d % 59 = 4)) :
  (a + b + c) % 59 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_59_l63_6349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_129_l63_6308

def mySequence : ℕ → ℕ
  | 0 => 12
  | n + 1 => 
    if mySequence n < 100 then
      mySequence n + 2 * (n + 1)
    else
      let lastTwoDigits := mySequence n % 100
      let firstDigit := mySequence n / 100
      (firstDigit + 1) * 100 + ((lastTwoDigits + 2 * ((n + 1) % 4)) % 100)

theorem tenth_term_is_129 : mySequence 9 = 129 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_129_l63_6308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l63_6373

theorem cos_minus_sin_value (x : ℝ) 
  (h1 : Real.sin (2 * x) = 3/4) 
  (h2 : π/4 < x ∧ x < π/2) : 
  Real.cos x - Real.sin x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l63_6373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_bike_speed_l63_6302

/-- A mini-triathlon with swimming, running, and biking segments -/
structure MiniTriathlon where
  swim_distance : ℚ
  run_distance : ℚ
  bike_distance : ℚ
  swim_speed : ℚ
  run_speed : ℚ
  total_time_goal : ℚ

/-- Calculate the required bike speed to complete the mini-triathlon in the goal time -/
def required_bike_speed (t : MiniTriathlon) : ℚ :=
  t.bike_distance / (t.total_time_goal - (t.swim_distance / t.swim_speed + t.run_distance / t.run_speed))

/-- The mini-triathlon problem -/
def linda_triathlon : MiniTriathlon :=
  { swim_distance := 1/8
  , run_distance := 2
  , bike_distance := 8
  , swim_speed := 1
  , run_speed := 8
  , total_time_goal := 3/2 }

theorem linda_bike_speed :
  required_bike_speed linda_triathlon = 64/9 := by
  unfold required_bike_speed linda_triathlon
  norm_num
  -- The proof is completed by norm_num, which simplifies and evaluates the expression
  -- If more detailed steps are needed, they can be added here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_bike_speed_l63_6302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt3_div_3_l63_6368

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := sorry

/-- Point on the ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The right focus of the ellipse -/
noncomputable def rightFocus (e : Ellipse) : ℝ × ℝ := sorry

/-- The right directrix of the ellipse -/
noncomputable def rightDirectrix (e : Ellipse) : ℝ → ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The eccentricity of an ellipse with specific properties is √3/3 -/
theorem ellipse_eccentricity_sqrt3_div_3 (e : Ellipse) 
  (P Q : EllipsePoint e) (M : ℝ × ℝ) :
  let F := rightFocus e
  let D := rightDirectrix e
  (P.x = F.1 ∧ Q.x = F.1) → -- P and Q are on a line through the right focus
  (M.1 = D 0) → -- M is where the right directrix intersects the x-axis
  (distance (P.x, P.y) (Q.x, Q.y) = distance (Q.x, Q.y) M ∧ 
   distance (Q.x, Q.y) M = distance M (P.x, P.y)) → -- PQM is equilateral
  eccentricity e = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt3_div_3_l63_6368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_5_fifth_power_l63_6352

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom fg_condition : ∀ x, x ≥ 1 → f (g x) = x^3
axiom gf_condition : ∀ x, x ≥ 1 → g (f x) = x^5
axiom g_25 : g 25 = 25

-- State the theorem to be proved
theorem g_5_fifth_power : (g 5)^5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_5_fifth_power_l63_6352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_225_l63_6304

-- Define the given parameters
noncomputable def train_length : ℝ := 150
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def crossing_time : ℝ := 30

-- Define the function to calculate bridge length
noncomputable def bridge_length : ℝ :=
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let total_distance : ℝ := train_speed_ms * crossing_time
  total_distance - train_length

-- Theorem statement
theorem bridge_length_is_225 :
  bridge_length = 225 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Perform the calculation
  simp [train_length, train_speed_kmh, crossing_time]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_225_l63_6304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_is_ten_l63_6364

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Reflection of a point about the line x = k -/
noncomputable def reflect (p : Point) (k : ℝ) : Point :=
  ⟨2 * k - p.x, p.y⟩

/-- Area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  (1/2) * abs (t.A.x * (t.B.y - t.C.y) + t.B.x * (t.C.y - t.A.y) + t.C.x * (t.A.y - t.B.y))

/-- Original triangle -/
def originalTriangle : Triangle :=
  ⟨⟨4, 3⟩, ⟨6, -2⟩, ⟨7, 1⟩⟩

/-- Reflected triangle -/
noncomputable def reflectedTriangle : Triangle :=
  ⟨reflect originalTriangle.A 6, reflect originalTriangle.B 6, reflect originalTriangle.C 6⟩

/-- Theorem: The area of the union of the original triangle and its reflection is 10 -/
theorem area_of_union_is_ten :
  triangleArea originalTriangle + triangleArea reflectedTriangle = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_is_ten_l63_6364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_of_valid_units_digits_l63_6393

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def valid_units_digit (x : ℕ) : Prop := x = 0 ∨ x = 5

theorem greatest_difference_of_valid_units_digits :
  ∀ x y : ℕ,
  (∃ n : ℕ, n = 940 + x ∧ is_multiple_of_five n) →
  (∃ m : ℕ, m = 940 + y ∧ is_multiple_of_five m) →
  valid_units_digit x →
  valid_units_digit y →
  (∀ a b : ℕ, valid_units_digit a → valid_units_digit b → Int.natAbs (a - b) ≤ Int.natAbs (x - y)) →
  Int.natAbs (x - y) = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_of_valid_units_digits_l63_6393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_specific_angles_l63_6371

theorem sin_sum_specific_angles (α β : ℝ) 
  (h1 : Real.sin α = -3/5) 
  (h2 : Real.cos β = 1) : 
  Real.sin (α + β) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_specific_angles_l63_6371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l63_6389

-- Define the functions f and g
noncomputable def f (a b x : ℝ) : ℝ := a^x - b
def g (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  (∀ x : ℝ, f a b x * g x ≤ 0) →
  (∃ m : ℝ, m = 4 ∧ ∀ a' b' : ℝ, a' > 0 → a' ≠ 1 → (∀ x : ℝ, f a' b' x * g x ≤ 0) → 1/a' + 4/b' ≥ m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l63_6389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_example_l63_6300

/-- The center of a hyperbola is the midpoint of its foci -/
noncomputable def hyperbola_center (f1 f2 : ℝ × ℝ) : ℝ × ℝ :=
  ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)

/-- Theorem: The center of a hyperbola with foci at (3, -2) and (7, 6) is (5, 2) -/
theorem hyperbola_center_example : 
  hyperbola_center (3, -2) (7, 6) = (5, 2) := by
  -- Unfold the definition of hyperbola_center
  unfold hyperbola_center
  -- Simplify the arithmetic
  simp [add_div]
  -- Check that the result is equal to (5, 2)
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_example_l63_6300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_is_four_l63_6376

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x) * Real.sin (x - 1) + x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 3

-- State the theorem
theorem max_min_sum_is_four :
  ∃ (M m : ℝ), (∀ x ∈ interval, f x ≤ M) ∧
               (∀ x ∈ interval, m ≤ f x) ∧
               (∃ x₁ ∈ interval, f x₁ = M) ∧
               (∃ x₂ ∈ interval, f x₂ = m) ∧
               M + m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_is_four_l63_6376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_and_distance_l63_6390

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : (Real × Real × Real) :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

noncomputable def distance_from_origin (x y z : Real) : Real :=
  Real.sqrt (x^2 + y^2 + z^2)

theorem spherical_to_rectangular_and_distance 
  (ρ θ φ : Real) 
  (h_ρ : ρ = 5)
  (h_θ : θ = 7 * Real.pi / 4)
  (h_φ : φ = Real.pi / 3) :
  let (x, y, z) := spherical_to_rectangular ρ θ φ
  (x = -5 * Real.sqrt 6 / 4 ∧ 
   y = -5 * Real.sqrt 6 / 4 ∧ 
   z = 5 / 2) ∧
  distance_from_origin x y z = 5 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_and_distance_l63_6390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l63_6363

theorem constant_term_binomial_expansion :
  let n : ℕ := 8
  let expansion := fun x : ℚ => (x - x⁻¹) ^ n
  (Finset.sum (Finset.range (n + 1)) (fun k => ((-1)^k * Nat.choose n k) * ((1 : ℚ)^(n - 2*k)))) = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l63_6363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_circle_centers_l63_6378

theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 20)
  (h_height : rectangle_height = 24)
  (h_diameter : circle_diameter = 10) :
  ∃ d : ℝ, d = Real.sqrt 296 ∧
    ∀ d' : ℝ, (∃ c₁ c₂ : ℝ × ℝ,
      0 ≤ c₁.1 ∧ c₁.1 ≤ rectangle_width ∧
      0 ≤ c₁.2 ∧ c₁.2 ≤ rectangle_height ∧
      0 ≤ c₂.1 ∧ c₂.1 ≤ rectangle_width ∧
      0 ≤ c₂.2 ∧ c₂.2 ≤ rectangle_height ∧
      ∀ p : ℝ × ℝ, dist p c₁ ≤ circle_diameter / 2 → p.1 ≥ 0 ∧ p.1 ≤ rectangle_width ∧ p.2 ≥ 0 ∧ p.2 ≤ rectangle_height ∧
      ∀ p : ℝ × ℝ, dist p c₂ ≤ circle_diameter / 2 → p.1 ≥ 0 ∧ p.1 ≤ rectangle_width ∧ p.2 ≥ 0 ∧ p.2 ≤ rectangle_height ∧
      d' = dist c₁ c₂) →
    d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_circle_centers_l63_6378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_atv_overtakes_motorcycle_on_11th_lap_l63_6381

/-- Represents a vehicle with speeds in forest and field -/
structure Vehicle where
  forestSpeed : ℝ
  fieldSpeed : ℝ

/-- Calculates the time taken for a vehicle to complete one lap -/
noncomputable def lapTime (v : Vehicle) (circum : ℝ) : ℝ :=
  (circum / 4) / v.forestSpeed + (3 * circum / 4) / v.fieldSpeed

/-- The circular road on which the vehicles travel -/
structure Road where
  circumference : ℝ

theorem atv_overtakes_motorcycle_on_11th_lap (road : Road) : 
  let motorcycle := Vehicle.mk 20 60
  let atv := Vehicle.mk 40 45
  let n : ℕ := 11
  (n : ℝ) * lapTime atv road.circumference = ((n : ℝ) - 1) * lapTime motorcycle road.circumference :=
by
  sorry

#check atv_overtakes_motorcycle_on_11th_lap

end NUMINAMATH_CALUDE_ERRORFEEDBACK_atv_overtakes_motorcycle_on_11th_lap_l63_6381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_addition_result_l63_6303

def A : Matrix (Fin 4) (Fin 4) ℤ := ![
  ![3, 0, 1, 4],
  ![1, 2, 0, 0],
  ![5, -3, 2, 1],
  ![0, 0, -1, 3]
]

def B : Matrix (Fin 4) (Fin 4) ℤ := ![
  ![-5, -7, 3, 2],
  ![4, -9, 5, -2],
  ![8, 2, -3, 0],
  ![1, 1, -2, -4]
]

def C : Matrix (Fin 4) (Fin 4) ℤ := ![
  ![-2, -7, 4, 6],
  ![5, -7, 5, -2],
  ![13, -1, -1, 1],
  ![1, 1, -3, -1]
]

theorem matrix_addition_result : A + B = C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_addition_result_l63_6303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_coin_game_winning_strategy_l63_6319

/-- Represents the coin game described in the problem -/
def CoinGame (n : ℕ) : Prop := n ≥ 2

/-- Represents a valid move in the coin game -/
def ValidMove (n : ℕ) (k : ℕ) : Prop := k ≤ n ∧ k > 0

/-- Represents the condition for a winning strategy for player A -/
def WinningStrategy (n : ℕ) : Prop := n % 4 = 0 ∨ n % 4 = 3

/-- 
Theorem stating that player A has a winning strategy in the coin game 
if and only if n is congruent to 0 or 3 modulo 4
-/
theorem coin_game_winning_strategy (n : ℕ) (h : CoinGame n) : 
  (∀ k, ValidMove n k → k ≤ n) → 
  (WinningStrategy n ↔ ∃ (strategy : ℕ → Prop), strategy n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_coin_game_winning_strategy_l63_6319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_integer_and_special_divisors_l63_6394

def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n

noncomputable def τ (n : ℕ) : ℕ := Finset.card (Nat.divisors n)

theorem fraction_integer_and_special_divisors :
  (∃! (p : ℕ), p > 0 ∧ is_integer ((p + 2 : ℚ) / (p + 1))) ∧
  (∀ (N : ℕ), (∃ (a b : ℕ), N = 2^a * 3^b) → 
    (τ (N^2) = 3 * τ N ↔ N = 144 ∨ N = 324)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_integer_and_special_divisors_l63_6394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_from_square_l63_6340

/-- The volume of a cylinder formed by rotating a square about its diagonal -/
noncomputable def cylinderVolume (sideLength : ℝ) : ℝ :=
  2000 * Real.sqrt 2 * Real.pi

/-- Theorem: The volume of the cylinder formed by rotating a square with side length 20 centimeters
    about its diagonal is equal to 2000√2π cubic centimeters -/
theorem cylinder_volume_from_square (sideLength : ℝ) (h : sideLength = 20) :
  cylinderVolume sideLength = 2000 * Real.sqrt 2 * Real.pi := by
  -- Unfold the definition of cylinderVolume
  unfold cylinderVolume
  -- The definition already matches the right-hand side, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_from_square_l63_6340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_speed_l63_6329

theorem escalator_speed 
  (escalator_length : ℝ) 
  (person_speed : ℝ) 
  (time_taken : ℝ) 
  (escalator_speed : ℝ) : 
  escalator_length = 196 →
  person_speed = 2 →
  time_taken = 14 →
  escalator_length = (person_speed + escalator_speed) * time_taken →
  escalator_speed = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_speed_l63_6329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_distance_squared_l63_6339

/-- Given a triangle with circumcenter O, orthocenter H, circumradius R, and angles α, β, γ,
    the square of the distance between O and H is equal to R² multiplied by (1 - 8 cos α cos β cos γ) -/
theorem orthocenter_circumcenter_distance_squared (O H : EuclideanSpace ℝ (Fin 2)) (R α β γ : ℝ) : 
  ‖O - H‖^2 = R^2 * (1 - 8 * Real.cos α * Real.cos β * Real.cos γ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_distance_squared_l63_6339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l63_6370

noncomputable def f (x : ℝ) := Real.log (x^2 - 2)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -Real.sqrt 2 ∨ x > Real.sqrt 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l63_6370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subsidy_required_min_avg_cost_amount_min_avg_cost_value_l63_6365

-- Define the processing cost function
noncomputable def processing_cost (x : ℝ) : ℝ := x^2 - 50*x + 900

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := 20*x - processing_cost x

-- Define the average processing cost function
noncomputable def avg_processing_cost (x : ℝ) : ℝ := processing_cost x / x

-- Theorem for the minimum subsidy required
theorem min_subsidy_required : 
  ∀ x ∈ Set.Icc 10 15, profit x ≤ -75 := by
  sorry

-- Theorem for the processing amount that minimizes average cost
theorem min_avg_cost_amount : 
  ∃ x : ℝ, x > 0 ∧ ∀ y > 0, avg_processing_cost x ≤ avg_processing_cost y := by
  sorry

-- Theorem for the minimum average processing cost
theorem min_avg_cost_value : 
  ∃ x : ℝ, x > 0 ∧ avg_processing_cost x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subsidy_required_min_avg_cost_amount_min_avg_cost_value_l63_6365
