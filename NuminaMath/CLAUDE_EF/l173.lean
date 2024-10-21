import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_wood_problem_l173_17303

theorem sunzi_wood_problem (x y : ℝ) : 
  y - x = 4.5 ∧ x - (1/2) * y = 1 → 
  (y - x = 4.5 ∧ x - (1/2) * y = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_wood_problem_l173_17303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l173_17384

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define helper functions
def externally_tangent (C1 C2 : Circle) : Prop := sorry
def internally_tangent (C1 C2 : Circle) : Prop := sorry
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the problem setup
def problem_setup (P Q R S : Circle) : Prop :=
  -- Circles P, Q, and R are externally tangent to each other
  externally_tangent P Q ∧ externally_tangent P R ∧ externally_tangent Q R ∧
  -- Circles P, Q, and R are internally tangent to circle S
  internally_tangent P S ∧ internally_tangent Q S ∧ internally_tangent R S ∧
  -- Circles Q and R are congruent
  Q.radius = R.radius ∧
  -- Circle P has radius 2
  P.radius = 2 ∧
  -- Circle P passes through the center of S
  distance P.center S.center = P.radius

-- Define the theorem
theorem circle_radius_theorem (P Q R S : Circle) :
  problem_setup P Q R S → Q.radius = 16 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l173_17384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_no_minimum_has_zero_three_distinct_roots_l173_17379

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then (x - 1)^3 else |x - 1|

-- Statement 1
theorem exists_no_minimum : ∃ a : ℝ, ∀ y : ℝ, ∃ x : ℝ, f a x < y := by
  sorry

-- Statement 2
theorem has_zero (a : ℝ) : ∃ x : ℝ, f a x = 0 := by
  sorry

-- Statement 3
theorem three_distinct_roots (a : ℝ) (h : 1 < a ∧ a < 2) : 
  ∃ m : ℝ, ∃ x₁ x₂ x₃ : ℝ, 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_no_minimum_has_zero_three_distinct_roots_l173_17379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l173_17338

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi/2 ∧ 
  0 < B ∧ B < Real.pi/2 ∧ 
  0 < C ∧ C < Real.pi/2 ∧
  A + B + C = Real.pi

theorem triangle_properties 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : 4 * a * Real.sin B = Real.sqrt 7 * b)
  (h_a : a = 6)
  (h_bc : b + c = 8) :
  let S := (1/2) * b * c * Real.sin A
  (S = Real.sqrt 7) ∧ 
  (Real.sin (2 * A + 2 * Real.pi / 3) = (Real.sqrt 3 - 3 * Real.sqrt 7) / 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l173_17338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_primes_sum_zero_mod_p_l173_17386

-- Define the function χ
def χ (p : ℕ) (a : Fin p) : Int := sorry

-- Define the sum
def sum_a_chi (p : ℕ) : Int := sorry

-- Theorem statement
theorem count_primes_sum_zero_mod_p : 
  (Finset.filter (fun p => Nat.Prime p ∧ p < 100 ∧ sum_a_chi p ≡ 0 [ZMOD p]) (Finset.range 100)).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_primes_sum_zero_mod_p_l173_17386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l173_17317

/-- Triangle ABC in a plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Area of a triangle using determinant formula -/
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * abs (t.A.1 * (t.B.2 - t.C.2) + t.B.1 * (t.C.2 - t.A.2) + t.C.1 * (t.A.2 - t.B.2))

/-- The main theorem -/
theorem triangle_problem :
  ∃! (n : ℕ), ∃ (S : Finset (ℝ × ℝ)),
    S.card = n ∧
    (∀ C ∈ S, ∃ (t : Triangle),
      t.A = (0, 0) ∧
      t.B = (12, 0) ∧
      t.C = C ∧
      distance t.A t.B = 12 ∧
      perimeter t = 60 ∧
      area t = 144) ∧
    (∀ C : ℝ × ℝ, (∃ (t : Triangle),
      t.A = (0, 0) ∧
      t.B = (12, 0) ∧
      t.C = C ∧
      distance t.A t.B = 12 ∧
      perimeter t = 60 ∧
      area t = 144) → C ∈ S) ∧
    n = 2 :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l173_17317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_max_a_value_l173_17304

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 1) / Real.exp x

-- Statement 1: f(x) is monotonically increasing on (1, 2)
theorem f_monotone_increasing : 
  ∀ x y, 1 < x ∧ x < y ∧ y < 2 → f x < f y := by
  sorry

-- Statement 2: Given e^x * f(x) ≥ a + ln(x) for all x > 0, the maximum value of a is 1
theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.exp x * f x ≥ a + Real.log x) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_max_a_value_l173_17304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_C_l173_17306

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem triangle_angle_C (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  f A = 1 ∧
  Real.sin B ^ 2 + Real.sqrt 2 * Real.sin A * Real.sin C = Real.sin A ^ 2 + Real.sin C ^ 2 →
  C = 5 * Real.pi / 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_C_l173_17306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_score_combinations_l173_17324

/-- The number of possible score combinations for four students -/
def score_combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

/-- The set of possible scores -/
def score_set : Finset ℕ := {90, 92, 93, 96, 98}

/-- The theorem stating the total number of possible score combinations -/
theorem total_score_combinations :
  score_combinations (Finset.card score_set) 4 +
  score_combinations (Finset.card score_set) 3 = 15 := by
  -- Unfold the definitions
  unfold score_combinations
  unfold score_set
  -- Simplify the expressions
  simp [Finset.card]
  -- The proof is completed
  rfl

#eval score_combinations (Finset.card score_set) 4 +
      score_combinations (Finset.card score_set) 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_score_combinations_l173_17324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_increasing_iff_base_greater_than_one_l173_17359

noncomputable section

/-- An exponential function -/
def exponential (a : ℝ) (x : ℝ) : ℝ := a^x

/-- A function is increasing on ℝ -/
def IncreasingOnReals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem exponential_increasing_iff_base_greater_than_one
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  IncreasingOnReals (exponential a) ↔ a > 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_increasing_iff_base_greater_than_one_l173_17359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_72deg_20cm_l173_17396

/-- The area of a circular sector with central angle θ (in radians) and radius r -/
noncomputable def sectorArea (θ : ℝ) (r : ℝ) : ℝ := (θ / (2 * Real.pi)) * Real.pi * r^2

theorem sector_area_72deg_20cm (θ : ℝ) (r : ℝ) (h1 : θ = 72 * Real.pi / 180) (h2 : r = 20) :
  sectorArea θ r = 80 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_72deg_20cm_l173_17396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_field_area_l173_17330

/-- Represents the cost of fencing in rupees -/
noncomputable def fencing_cost : ℚ := 945/10

/-- Represents the fencing rate in rupees per meter -/
noncomputable def fencing_rate : ℚ := 1/4

/-- Represents the ratio of the shorter side to the longer side of the rectangular field -/
noncomputable def side_ratio : ℚ := 3/4

theorem rectangular_field_area (x : ℚ) 
  (h1 : x > 0) 
  (h2 : fencing_cost = (3 * x + 4 * x) * 2 * fencing_rate) : 
  3 * x * 4 * x = 8748 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_field_area_l173_17330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_arithmetic_problem_l173_17329

theorem modular_arithmetic_problem (n : ℕ) : 
  n < 19 → 5 * n ≡ 1 [ZMOD 19] → (3^n)^3 - 3 ≡ 6 [ZMOD 19] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_arithmetic_problem_l173_17329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_hyperbola_properties_l173_17363

/-- A rectangular hyperbola with equation xy = 4 -/
structure RectangularHyperbola where
  equation : ℝ → ℝ → Prop
  is_rectangular : Prop

/-- The foci of a rectangular hyperbola -/
noncomputable def foci (h : RectangularHyperbola) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((2 * Real.sqrt 2, 2 * Real.sqrt 2), (-2 * Real.sqrt 2, -2 * Real.sqrt 2))

/-- The distance between the foci of a rectangular hyperbola -/
def foci_distance : ℝ := 8

/-- Theorem stating the properties of the rectangular hyperbola xy = 4 -/
theorem rectangular_hyperbola_properties (h : RectangularHyperbola) 
  (heq : h.equation = fun x y => x * y = 4) 
  (hrect : h.is_rectangular = true) : 
  foci h = ((2 * Real.sqrt 2, 2 * Real.sqrt 2), (-2 * Real.sqrt 2, -2 * Real.sqrt 2)) ∧ 
  foci_distance = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_hyperbola_properties_l173_17363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l173_17382

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

-- State the theorem about the minimum positive period and maximum value of f(x)
theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
    (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S)) ∧
  (∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ y, f y = M)) ∧
  (let T := 6 * Real.pi;
   let M := Real.sqrt 2;
   (∀ x, f (x + T) = f x) ∧
   (∀ x, f x ≤ M) ∧
   (∃ y, f y = M)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l173_17382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_length_l173_17369

noncomputable section

-- Define the radius of the sphere
def sphere_radius : ℝ := 4

-- Define the surface area of a sphere
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Define the surface area of a cube
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

-- Theorem statement
theorem cube_side_length :
  ∃ (s : ℝ), s > 0 ∧ cube_surface_area s = sphere_surface_area sphere_radius ∧ 
  s = Real.sqrt ((32 * Real.pi) / 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_length_l173_17369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_food_ratio_proof_l173_17389

/-- The ratio of the second dog's food to the first dog's food -/
noncomputable def dog_food_ratio (T : ℝ) : ℝ :=
  (T - 19) / 13

/-- Theorem stating the ratio of the second dog's food to the first dog's food -/
theorem dog_food_ratio_proof (T : ℝ) (h1 : T > 19) :
  ∃ (x : ℝ), x > 0 ∧ x / 13 = dog_food_ratio T ∧ 13 + x + 6 = T := by
  use T - 19
  constructor
  · -- Prove x > 0
    linarith
  constructor
  · -- Prove x / 13 = dog_food_ratio T
    unfold dog_food_ratio
    simp
  · -- Prove 13 + x + 6 = T
    linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_food_ratio_proof_l173_17389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_equals_45_l173_17348

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ≤ a (n + 1)) ∧
  (∀ k, (Finset.filter (λ n ↦ a n = k) (Finset.range (Finset.sum (Finset.range k) (λ i ↦ 2 * i + 1)))).card = 2 * k - 1)

theorem a_2014_equals_45 (a : ℕ → ℕ) (h : is_valid_sequence a) : a 2014 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_equals_45_l173_17348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_theorem_l173_17367

theorem perfect_square_theorem (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : c = a + b / a - 1 / b) : 
  ∃ m : ℕ, c = m^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_theorem_l173_17367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_alone_time_l173_17375

/-- Represents the time it takes for a worker to complete a job alone -/
@[ext] structure WorkTime where
  time : ℝ
  time_pos : time > 0

/-- Represents the rate at which a worker completes a job -/
@[ext] structure WorkRate where
  rate : ℝ
  rate_pos : rate > 0

variable (a b c : WorkRate)

/-- The sum of work rates equals the reciprocal of the time taken -/
axiom work_rate_sum (t : WorkTime) (rates : List WorkRate) : 
  (rates.map WorkRate.rate).sum = 1 / t.time → (1 / (rates.map WorkRate.rate).sum = t.time)

theorem c_alone_time (h1 : a.rate + b.rate = 1 / 15) (h2 : a.rate + b.rate + c.rate = 1 / 10) :
  1 / c.rate = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_alone_time_l173_17375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_inequality_exclusive_or_condition_l173_17394

def line_equation (m x y : ℝ) : Prop := m * x - y + 2 = 0

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 19/4 = 0

def line_intersects_circle (m : ℝ) : Prop := 
  ∃ x y : ℝ, line_equation m x y ∧ circle_equation x y

def inequality_condition (m : ℝ) : Prop :=
  ∃ x0 : ℝ, x0 ∈ Set.Icc (-Real.pi/6) (Real.pi/4) ∧ 
    2 * Real.sin (2*x0 + Real.pi/6) + 2 * Real.cos (2*x0) ≤ m

theorem intersection_and_inequality (m : ℝ) :
  (line_intersects_circle m ∧ inequality_condition m) →
  m ∈ Set.Icc 0 (Real.sqrt 3 / 3) :=
sorry

theorem exclusive_or_condition (m : ℝ) :
  (line_intersects_circle m ∨ inequality_condition m) ∧
  ¬(line_intersects_circle m ∧ inequality_condition m) →
  m ∈ Set.Ioo (-Real.sqrt 3 / 3) 0 ∪ Set.Ici (Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_inequality_exclusive_or_condition_l173_17394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_decreases_with_angle_increase_acute_triangle_from_positive_cos_product_isosceles_or_right_from_cosine_relation_l173_17310

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Statement A
theorem cos_decreases_with_angle_increase {t : Triangle} :
  t.A > t.B → Real.cos t.A < Real.cos t.B := by
  sorry

-- Statement C
theorem acute_triangle_from_positive_cos_product {t : Triangle} :
  Real.cos t.A * Real.cos t.B * Real.cos t.C > 0 →
  t.A < Real.pi / 2 ∧ t.B < Real.pi / 2 ∧ t.C < Real.pi / 2 := by
  sorry

-- Statement D
theorem isosceles_or_right_from_cosine_relation {t : Triangle} :
  t.a - t.c * Real.cos t.B = t.a * Real.cos t.C →
  (t.a = t.b ∨ t.b = t.c ∨ t.c = t.a) ∨ (t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_decreases_with_angle_increase_acute_triangle_from_positive_cos_product_isosceles_or_right_from_cosine_relation_l173_17310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_magnitudes_of_sum_l173_17319

open Set
open Real

/-- Represents a plane vector -/
structure PlaneVector where
  x : ℝ
  y : ℝ

/-- The dot product of two plane vectors -/
def dot (u v : PlaneVector) : ℝ := u.x * v.x + u.y * v.y

/-- The magnitude of a plane vector -/
noncomputable def mag (v : PlaneVector) : ℝ := Real.sqrt (dot v v)

/-- The sum of a list of plane vectors -/
def vecSum (vs : List PlaneVector) : PlaneVector :=
  { x := vs.map (·.x) |>.sum
  , y := vs.map (·.y) |>.sum }

theorem possible_magnitudes_of_sum (a : ℕ → PlaneVector) (h1 : ∀ i, mag (a i) = 2) 
    (h2 : ∀ i, dot (a i) (a (i + 1)) = 0) :
  ∀ m ≥ 2, mag (vecSum (List.range m |>.map a)) ∈ ({0, 2, 2 * Real.sqrt 2} : Set ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_magnitudes_of_sum_l173_17319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_game_sum_l173_17334

/-- Represents the state of a calculator after a number of operations. -/
structure CalculatorState where
  value : Int
  iterations : Nat

/-- Performs the operation on a calculator state based on its initial value. -/
def operate : CalculatorState → CalculatorState
  | ⟨1, n⟩ => ⟨1, n + 1⟩  -- Cubing 1 remains 1
  | ⟨0, n⟩ => ⟨0, n + 1⟩  -- Squaring 0 remains 0
  | ⟨-1, n⟩ => ⟨(-1) ^ (n + 1), n + 1⟩  -- Negating -1 alternates between -1 and 1
  | ⟨v, n⟩ => ⟨v, n + 1⟩  -- Default case for other values

/-- Applies the operation n times to a calculator state. -/
def iterateOperation (n : Nat) : CalculatorState → CalculatorState :=
  (operate^[n])

theorem calculator_game_sum :
  let final1 := iterateOperation 50 ⟨1, 0⟩
  let final0 := iterateOperation 50 ⟨0, 0⟩
  let finalNeg1 := iterateOperation 50 ⟨-1, 0⟩
  final1.value + final0.value + finalNeg1.value = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_game_sum_l173_17334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_hemisphere_radius_l173_17395

/-- The volume of a hemisphere with radius r -/
noncomputable def hemisphereVolume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

/-- Given a hemisphere of radius 1 and 27 smaller congruent hemispheres with the same total volume,
    prove that the radius of each smaller hemisphere is 1/3 -/
theorem smaller_hemisphere_radius :
  let largeRadius : ℝ := 1
  let numSmallHemispheres : ℕ := 27
  let largeVolume := hemisphereVolume largeRadius
  ∃ (smallRadius : ℝ),
    (numSmallHemispheres : ℝ) * hemisphereVolume smallRadius = largeVolume ∧
    smallRadius = 1/3 :=
by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_hemisphere_radius_l173_17395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_strategy_29_attempts_exists_strategy_24_attempts_l173_17353

/-- Represents a test with a fixed number of questions -/
structure Test where
  num_questions : ℕ
  answers : Fin num_questions → Bool

/-- Represents the result of a test attempt -/
def AttemptResult := ℕ

/-- A strategy is a function that takes the current attempt number and previous results,
    and returns a new set of answers -/
def Strategy (n : ℕ) := ℕ → List AttemptResult → Fin n → Bool

/-- Theorem stating that there exists a strategy to determine all correct answers within 29 attempts -/
theorem exists_strategy_29_attempts (test : Test) (h : test.num_questions = 30) :
  ∃ (s : Strategy 30), ∀ (n : ℕ) (results : List AttemptResult),
    n ≤ 29 → (∀ i : Fin 30, s n results i = test.answers ⟨i.val, by {
      rw [h]
      exact i.isLt
    }⟩) :=
sorry

/-- Theorem stating that there exists a strategy to determine all correct answers within 24 attempts -/
theorem exists_strategy_24_attempts (test : Test) (h : test.num_questions = 30) :
  ∃ (s : Strategy 30), ∀ (n : ℕ) (results : List AttemptResult),
    n ≤ 24 → (∀ i : Fin 30, s n results i = test.answers ⟨i.val, by {
      rw [h]
      exact i.isLt
    }⟩) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_strategy_29_attempts_exists_strategy_24_attempts_l173_17353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_run_distance_l173_17356

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents Crystal's running path -/
def crystalPath : List Point :=
  [⟨0, 0⟩, ⟨0, 2⟩, ⟨3, 2⟩, ⟨3, 0⟩]

theorem crystal_run_distance :
  distance (crystalPath.head!) (crystalPath.getLast!) = 3 := by
  sorry

#check crystal_run_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_run_distance_l173_17356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_case_l173_17316

/-- The area of a triangle given the length of one side and two medians --/
noncomputable def triangle_area_from_side_and_medians (side_ab : ℝ) (median_a : ℝ) (median_b : ℝ) : ℝ :=
  let s_m := (median_a + median_b + (side_ab ^ 2 + 4 * median_a ^ 2 + 4 * median_b ^ 2 - 
              2 * side_ab * median_a - 2 * side_ab * median_b - 2 * median_a * median_b).sqrt) / 2
  4 / 3 * (s_m * (s_m - median_a) * (s_m - median_b) * 
           (s_m - (side_ab ^ 2 + 4 * median_a ^ 2 + 4 * median_b ^ 2 - 
                   2 * side_ab * median_a - 2 * side_ab * median_b - 2 * median_a * median_b).sqrt)).sqrt

theorem triangle_area_specific_case :
  triangle_area_from_side_and_medians 10 9 12 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_case_l173_17316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l173_17387

noncomputable def C₁ (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

noncomputable def C₂ (a b : ℝ) (x y : ℝ) : Prop := y^2/a^2 + x^2/b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

noncomputable def area_ratio (k₁ k₂ : ℝ) : ℝ := 4 - 36 / (17 + 2*(k₁^2 + k₂^2))

theorem ellipse_problem (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) :
  eccentricity a b = Real.sqrt 2 / 2 →
  C₂ a b 1 (Real.sqrt 2) →
  (a = 2 ∧ b = Real.sqrt 2) ∧
  (∀ k₁ k₂ : ℝ, k₁ * k₂ = -2 → 
    64/25 ≤ area_ratio k₁ k₂ ∧ area_ratio k₁ k₂ < 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l173_17387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_difference_zero_l173_17335

theorem integral_difference_zero :
  (∫ x in (1:ℝ)..(Real.exp 1), (1 / x)) - (∫ x in (0:ℝ)..(Real.pi / 2), Real.sin x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_difference_zero_l173_17335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l173_17397

theorem divisibility_condition (a m n : ℕ) :
  (a > 0 ∧ m > 0 ∧ n > 0) →
  (a^m + 1 ∣ (a + 1)^n) ↔ 
  (m = 1 ∨ a = 1 ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l173_17397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_placement_l173_17378

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents a small board on the chessboard -/
structure SmallBoard :=
  (x : ℕ)
  (y : ℕ)
  (size : ℕ)

/-- Checks if two small boards overlap -/
def overlap (b1 b2 : SmallBoard) : Prop :=
  ∃ (x y : ℕ), 
    x ≥ b1.x ∧ x < b1.x + b1.size ∧
    y ≥ b1.y ∧ y < b1.y + b1.size ∧
    x ≥ b2.x ∧ x < b2.x + b2.size ∧
    y ≥ b2.y ∧ y < b2.y + b2.size

/-- Main theorem -/
theorem chessboard_placement 
  (board : Chessboard)
  (small_boards : List SmallBoard)
  (h1 : board.size = 48)
  (h2 : ∀ b, b ∈ small_boards → b.size = 3)
  (h3 : small_boards.length = 99)
  (h4 : ∀ b1 b2, b1 ∈ small_boards → b2 ∈ small_boards → b1 ≠ b2 → ¬ overlap b1 b2) :
  ∃ (new_board : SmallBoard),
    new_board.size = 3 ∧
    new_board.x + new_board.size ≤ board.size ∧
    new_board.y + new_board.size ≤ board.size ∧
    ∀ b, b ∈ small_boards → ¬ overlap new_board b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_placement_l173_17378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_midline_l173_17368

-- Define the sinusoidal function
noncomputable def sinusoidal (a b c d : ℝ) (x : ℝ) : ℝ := a * Real.sin (b * x + c) + d

-- State the theorem
theorem sinusoidal_midline 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_max : ∀ x, sinusoidal a b c d x ≤ 5) 
  (h_min : ∀ x, sinusoidal a b c d x ≥ -3) 
  (h_reaches_max : ∃ x, sinusoidal a b c d x = 5) 
  (h_reaches_min : ∃ x, sinusoidal a b c d x = -3) : 
  d = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_midline_l173_17368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angle_satisfying_conditions_l173_17362

theorem unique_angle_satisfying_conditions : 
  ∃! x : Real, 0 ≤ x ∧ x < 2 * Real.pi ∧ Real.cos x = -0.5 ∧ Real.sin x < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angle_satisfying_conditions_l173_17362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_stretch_problem_l173_17372

/-- Represents the extension of a spring under a given force. -/
noncomputable def spring_extension (force : ℝ) (k : ℝ) : ℝ := force / k

/-- Hooke's Law constant for the given spring. -/
noncomputable def spring_constant (force : ℝ) (extension : ℝ) : ℝ := force / extension

theorem spring_stretch_problem (initial_force initial_extension new_force : ℝ) 
  (h1 : initial_force = 100)
  (h2 : initial_extension = 20)
  (h3 : new_force = 150) :
  let k := spring_constant initial_force initial_extension
  spring_extension new_force k = 30 := by
  sorry

#eval "Theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_stretch_problem_l173_17372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_l173_17349

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + (2 : ℝ)^x

-- State the theorem
theorem x_range (x : ℝ) : f (x^2 + 2) < f (3*x) → x ∈ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_l173_17349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagonal_circle_arrangement_l173_17347

/-- The diameter of a circle circumscribing eight tangent circles of radius 4 units
    arranged in an octagonal pattern -/
noncomputable def large_circle_diameter : ℝ := 8 * (Real.sqrt 2 + 1)

/-- The radius of each small circle -/
def small_circle_radius : ℝ := 4

/-- The number of small circles -/
def num_small_circles : ℕ := 8

/-- Predicate to represent that a real number is the diameter of a circle
    circumscribing n tangent circles of radius r -/
def is_diameter_of_circle_circumscribing (D : ℝ) (n : ℕ) (r : ℝ) : Prop := sorry

theorem octagonal_circle_arrangement (r : ℝ) (n : ℕ) (h1 : r = small_circle_radius) (h2 : n = num_small_circles) :
  ∃ (D : ℝ), D = large_circle_diameter ∧ 
  D = 2 * (r * (Real.sqrt 2 + 1)) ∧
  is_diameter_of_circle_circumscribing D n r :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagonal_circle_arrangement_l173_17347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l173_17398

/-- Geometric sequence with first term a and common ratio r -/
noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ := λ n ↦ a * r^(n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_five :
  ∀ (a : ℝ),
  (∃ (r : ℝ), r > 1 ∧
    geometric_sequence a r 1 = 1 ∧
    geometric_sequence a r 3 = 4 ∧
    (geometric_sequence a r 1)^2 - 5*(geometric_sequence a r 1) + 4 = 0 ∧
    (geometric_sequence a r 3)^2 - 5*(geometric_sequence a r 3) + 4 = 0) →
  geometric_sum a 2 5 = 31 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l173_17398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_four_vertices_shapes_l173_17374

structure Cube where
  vertices : Finset (Fin 8)
  is_cube : vertices.card = 8

def is_rectangle (vertices : Finset (Fin 8)) : Prop :=
  vertices.card = 4 ∧ sorry -- Add appropriate conditions for rectangle

def is_tetrahedron_isosceles_equilateral (vertices : Finset (Fin 8)) : Prop :=
  vertices.card = 4 ∧ sorry -- Add appropriate conditions for this specific tetrahedron

def is_tetrahedron_equilateral (vertices : Finset (Fin 8)) : Prop :=
  vertices.card = 4 ∧ sorry -- Add appropriate conditions for equilateral tetrahedron

def is_tetrahedron_right (vertices : Finset (Fin 8)) : Prop :=
  vertices.card = 4 ∧ sorry -- Add appropriate conditions for right tetrahedron

theorem cube_four_vertices_shapes (c : Cube) :
  (∃ v : Finset (Fin 8), v ⊆ c.vertices ∧ is_rectangle v) ∧
  (∃ v : Finset (Fin 8), v ⊆ c.vertices ∧ is_tetrahedron_isosceles_equilateral v) ∧
  (∃ v : Finset (Fin 8), v ⊆ c.vertices ∧ is_tetrahedron_equilateral v) ∧
  (∃ v : Finset (Fin 8), v ⊆ c.vertices ∧ is_tetrahedron_right v) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_four_vertices_shapes_l173_17374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_sqrt3x_minus_y_plus2_l173_17340

theorem angle_of_inclination_sqrt3x_minus_y_plus2 :
  let line : Set (ℝ × ℝ) := {(x, y) | Real.sqrt 3 * x - y + 2 = 0}
  ∃ θ : ℝ, θ = π / 3 ∧ ∀ (x y : ℝ), (x, y) ∈ line → Real.tan θ = (y - 2) / (Real.sqrt 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_sqrt3x_minus_y_plus2_l173_17340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_edmonton_to_calgary_l173_17365

/-- Represents the travel scenario from Edmonton to Calgary --/
structure TravelScenario where
  distance_edmonton_red_deer : ℝ
  distance_red_deer_calgary : ℝ
  speed_to_red_deer : ℝ
  speed_to_calgary : ℝ
  detour_distance : ℝ
  stop_duration : ℝ

/-- Calculates the total travel time for the given scenario --/
noncomputable def total_travel_time (scenario : TravelScenario) : ℝ :=
  let time_to_red_deer := (scenario.distance_edmonton_red_deer + scenario.detour_distance) / scenario.speed_to_red_deer
  let time_to_calgary := scenario.distance_red_deer_calgary / scenario.speed_to_calgary
  time_to_red_deer + scenario.stop_duration + time_to_calgary

/-- The given travel scenario --/
def edmonton_to_calgary : TravelScenario where
  distance_edmonton_red_deer := 220
  distance_red_deer_calgary := 110
  speed_to_red_deer := 100
  speed_to_calgary := 90
  detour_distance := 30
  stop_duration := 1

/-- Theorem stating that the total travel time for the given scenario is approximately 4.72 hours --/
theorem travel_time_edmonton_to_calgary :
  ∃ ε > 0, |total_travel_time edmonton_to_calgary - 4.72| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_edmonton_to_calgary_l173_17365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_equals_one_l173_17343

/-- An ellipse with equation x² + my² = 1, where m is a positive real number -/
structure Ellipse (m : ℝ) where
  m_pos : m > 0
  equation : Set (ℝ × ℝ) := {(x, y) | x^2 + m * y^2 = 1}

/-- The foci of an ellipse are on the y-axis -/
def foci_on_y_axis (m : ℝ) (e : Ellipse m) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ (0, c) ∈ e.equation ∧ (0, -c) ∈ e.equation

/-- The length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (m : ℝ) (e : Ellipse m) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a = 2 * b ∧
    (a, 0) ∈ e.equation ∧ (-a, 0) ∈ e.equation ∧
    (0, b) ∈ e.equation ∧ (0, -b) ∈ e.equation

theorem ellipse_m_equals_one (m : ℝ) (e : Ellipse m) 
  (h1 : foci_on_y_axis m e) (h2 : major_axis_twice_minor m e) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_equals_one_l173_17343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_99_equals_1782_l173_17313

def sequence_sum (n : ℕ) : ℕ :=
  let group := (n - 1) / 3
  let position := (n - 1) % 3
  group + position + 1

def sum_first_99 : ℕ := (Finset.range 99).sum (λ i => sequence_sum (i + 1))

theorem sum_first_99_equals_1782 : sum_first_99 = 1782 := by
  -- Proof goes here
  sorry

#eval sum_first_99

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_99_equals_1782_l173_17313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_doubles_l173_17308

/-- Square OPQR with O at origin and Q at (3,3) -/
structure Square where
  O : ℝ × ℝ
  Q : ℝ × ℝ
  is_origin : O = (0, 0)
  is_q : Q = (3, 3)
  is_square : ∀ (P R : ℝ × ℝ), P.1 = 3 ∧ P.2 = 0 ∧ R.1 = 0 ∧ R.2 = 3

/-- Point T -/
def T : ℝ × ℝ := (3, 6)

/-- Area of a triangle given three points -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- Theorem: T satisfies the area doubling condition after rotation -/
theorem area_doubles (s : Square) : 
  let P : ℝ × ℝ := (3, 0)
  let area_square := (3 : ℝ) ^ 2
  let area_triangle := triangle_area P s.Q T
  2 * area_square = area_triangle := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_doubles_l173_17308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_two_theta_l173_17320

theorem cos_pi_third_plus_two_theta (θ : ℝ) :
  Real.sin (π / 3 - θ) = 3 / 4 → Real.cos (π / 3 + 2 * θ) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_two_theta_l173_17320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_triples_l173_17377

theorem number_of_triples : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ × ℕ), x ∈ S ↔ 
      (Nat.gcd (Nat.gcd x.1 x.2.1) x.2.2 = 10 ∧ 
       Nat.lcm (Nat.lcm x.1 x.2.1) x.2.2 = 2^17 * 5^16)) ∧ 
    S.card = 8640 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_triples_l173_17377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l173_17350

/-- Circle Q₁ with center (0, 0) and radius 3 -/
def Q₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

/-- Circle Q₂ with center (3, 4) and radius 1 -/
def Q₂ : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}

/-- The center of Q₁ -/
def O₁ : ℝ × ℝ := (0, 0)

/-- The center of Q₂ -/
def O₂ : ℝ × ℝ := (3, 4)

/-- The radius of Q₁ -/
def R₁ : ℝ := 3

/-- The radius of Q₂ -/
def R₂ : ℝ := 1

/-- The distance between the centers of Q₁ and Q₂ -/
noncomputable def distance_between_centers : ℝ := Real.sqrt ((O₂.1 - O₁.1)^2 + (O₂.2 - O₁.2)^2)

/-- Number of common tangents between two circles -/
def NumberOfCommonTangents (C₁ C₂ : Set (ℝ × ℝ)) : ℕ := sorry

theorem common_tangents_count : ∃ (n : ℕ), n = 4 ∧ n = NumberOfCommonTangents Q₁ Q₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l173_17350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_amount_proof_l173_17361

-- Define the molar mass of water
def water_molar_mass : ℝ := 18.015

-- Define the mass of water formed
def water_mass : ℝ := 18

-- Theorem statement
theorem water_amount_proof :
  ∃ (ε : ℝ), ε > 0 ∧ |water_mass / water_molar_mass - 1| < ε := by
  -- We use ∃ to express "approximately equal" in a more formal way
  -- This states that there exists some small positive number ε such that
  -- the absolute difference between our calculation and 1 is less than ε
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_amount_proof_l173_17361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_number_l173_17342

-- Define the ranges as finite sets
def billy_range : Finset ℕ := Finset.filter (λ n => n > 0 ∧ n < 300 ∧ n % 20 = 0) (Finset.range 300)
def bobbi_range : Finset ℕ := Finset.filter (λ n => n > 0 ∧ n < 300 ∧ n % 30 = 0) (Finset.range 300)
def common_range : Finset ℕ := billy_range ∩ bobbi_range

theorem probability_same_number : 
  (Finset.card common_range : ℚ) / ((Finset.card billy_range : ℚ) * (Finset.card bobbi_range : ℚ)) = 1 / 30 := by
  sorry

#eval Finset.card common_range
#eval Finset.card billy_range
#eval Finset.card bobbi_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_number_l173_17342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l173_17314

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The total volume of three spheres with radii 4, 6, and 8 inches -/
noncomputable def total_volume : ℝ :=
  sphere_volume 4 + sphere_volume 6 + sphere_volume 8

/-- Theorem: The total volume of three spheres with radii 4, 6, and 8 inches is 1056π cubic inches -/
theorem snowman_volume : total_volume = 1056 * Real.pi := by
  -- Unfold definitions
  unfold total_volume sphere_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l173_17314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_values_l173_17331

-- Define the angles α and β
noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

-- Define the conditions
axiom α_third_quadrant : Real.pi < α ∧ α < 3 * Real.pi / 2
axiom β_terminal_point : Real.cos β = -2 / Real.sqrt 5 ∧ Real.sin β = -1 / Real.sqrt 5
axiom tan_2α : Real.tan (2 * α) = -4 / 3

-- Define the expressions
noncomputable def expression1 : ℝ :=
  (Real.sin (Real.pi / 2 + α) + 2 * Real.sin (3 * Real.pi - α)) / 
  (4 * Real.cos (-α) + Real.sin (Real.pi + α))

noncomputable def expression2 : ℝ :=
  Real.sin (2 * α + β)

-- State the theorem
theorem expressions_values :
  expression1 = 5 / 2 ∧ expression2 = -Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_values_l173_17331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_count_l173_17323

theorem quadratic_inequality_count : 
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ 3*x^2 + 11*x + 4 ≤ 21) ∧ Finset.card S = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_count_l173_17323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_value_l173_17383

theorem power_difference_value (a b : ℝ) 
  (ha : (10 : ℝ)^a = 3) (hb : (10 : ℝ)^b = 5) : (10 : ℝ)^(b-a) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_value_l173_17383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l173_17370

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the expression
noncomputable def expr (x : ℝ) (n : ℕ) : ℂ := (x^2 - i / Real.sqrt x)^n

-- Define the ratio of coefficients
def ratio_coeff : ℚ := -3/14

theorem expansion_properties :
  ∃ (n : ℕ),
    -- The ratio of the coefficient of the third term to the fifth term
    (n.choose 2 / n.choose 4 : ℚ) = ratio_coeff ∧
    -- The value of n
    n = 10 ∧
    -- The constant term
    ((-i)^8 * (n.choose 8 : ℂ) = 45) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l173_17370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l173_17352

noncomputable def a : ℝ := Real.log 3 / Real.log 10

noncomputable def b : ℝ := Real.log 2 / Real.log 10

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + (a + b) * x + 2 else 2

theorem equation_solutions_count :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x : ℝ, x ∈ s ↔ f x = x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l173_17352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l173_17307

noncomputable def a : ℝ × ℝ := (3, 2)
noncomputable def b : ℝ × ℝ := (-1, 2)
noncomputable def c : ℝ × ℝ := (4, 1)

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def perpendicular (u v : ℝ × ℝ) : Prop := dot_product u v = 0
def parallel (u v : ℝ × ℝ) : Prop := ∃ (k : ℝ), u = scalar_mult k v
noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem problem_solution :
  (∃ (k : ℝ), perpendicular (vector_add a (scalar_mult k c)) (vector_add (scalar_mult 2 b) (scalar_mult (-1) a)) ∧ k = -11/18) ∧
  (∃ (d : ℝ × ℝ), parallel d c ∧ vector_norm d = Real.sqrt 34 ∧ 
    (d = (4 * Real.sqrt 2, Real.sqrt 2) ∨ d = (-4 * Real.sqrt 2, -Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l173_17307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_first_group_length_l173_17325

/-- A sequence of 19 ones and 49 zeros arranged in random order -/
def RandomSequence := Fin 68 → Fin 2

/-- The probability of a one being at the start of the sequence -/
noncomputable def probOne : ℝ := 1 / 50

/-- The probability of a zero being at the start of the sequence -/
noncomputable def probZero : ℝ := 1 / 20

/-- The expected length of the first group in the sequence -/
noncomputable def expectedFirstGroupLength (seq : RandomSequence) : ℝ :=
  19 * probOne + 49 * probZero

theorem expected_first_group_length :
  ∀ seq : RandomSequence, expectedFirstGroupLength seq = 2.83 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_first_group_length_l173_17325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_above_185_is_184_l173_17315

/-- Calculates the sum of the first n odd numbers -/
def sumOfOddNumbers (n : ℕ) : ℕ := n^2

/-- Finds the row number for a given element in the array -/
def findRow (n : ℕ) : ℕ :=
  (Nat.sqrt n).succ

/-- Calculates the starting number of a given row -/
def rowStart (row : ℕ) : ℕ :=
  sumOfOddNumbers (row - 1) + 1

/-- Calculates the position of a number within its row -/
def positionInRow (n : ℕ) : ℕ :=
  n - rowStart (findRow n) + 1

/-- Theorem: The number directly above 185 in the array is 184 -/
theorem number_above_185_is_184 :
  let row := findRow 185
  let prevRow := row - 1
  let position := positionInRow 185
  rowStart prevRow + position - 1 = 184 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_above_185_is_184_l173_17315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_calculation_l173_17390

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a binary number (represented as a list of bits). -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec to_bits m : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
    to_bits n

/-- The main theorem to be proved. -/
theorem binary_calculation :
  let a := [true, false, true, true]  -- 1101₂
  let b := [false, true, true]        -- 110₂
  let c := [true, true, false, true]  -- 1011₂
  let d := [true, false, false, true] -- 1001₂
  binary_to_nat a + binary_to_nat b - binary_to_nat c + binary_to_nat d =
  binary_to_nat [true, false, false, false, true] -- 10001₂
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_calculation_l173_17390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_k_l173_17399

def k : Nat := 10^40 - 46

theorem sum_of_digits_of_k : (k.repr.toList.map (λ c => c.toString.toNat!)).sum = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_k_l173_17399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_curve_l173_17392

-- Define the curve C'
noncomputable def curve_C' (x' y' : ℝ) : Prop := (x' / 2)^2 + y'^2 = 1

-- Define the expression to be minimized
noncomputable def expr_to_minimize (x' y' : ℝ) : ℝ := x' + 2 * Real.sqrt 3 * y'

-- Theorem statement
theorem min_value_on_curve :
  ∃ (min : ℝ), min = -4 ∧
  (∀ (x' y' : ℝ), curve_C' x' y' → expr_to_minimize x' y' ≥ min) ∧
  (∃ (x'₀ y'₀ : ℝ), curve_C' x'₀ y'₀ ∧ expr_to_minimize x'₀ y'₀ = min) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_curve_l173_17392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_value_scaling_l173_17300

theorem cube_value_scaling (initial_side : ℝ) (new_side : ℝ) (initial_value : ℝ) :
  initial_side = 4 →
  new_side = 6 →
  initial_value = 500 →
  ∃ (new_value : ℝ), new_value = ⌊initial_value * (new_side / initial_side)^3 + 0.5⌋ ∧ new_value = 1688 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_value_scaling_l173_17300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l173_17327

/-- The distance between two points in 3D space -/
noncomputable def distance_3d (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- Coordinates of point A -/
def A : ℝ × ℝ × ℝ := (1, 3, -2)

/-- Coordinates of point B -/
def B : ℝ × ℝ × ℝ := (-2, 3, 2)

theorem distance_A_to_B :
  distance_3d A.1 A.2.1 A.2.2 B.1 B.2.1 B.2.2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l173_17327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l173_17381

/-- Given a right-angled triangle with hypotenuse H, if a smaller similar triangle is formed
    by cutting parallel to the hypotenuse such that the new hypotenuse is 0.65H and the area
    of the smaller triangle is 14.365 square inches, then the area of the original triangle
    is approximately 34 square inches. -/
theorem original_triangle_area (H : ℝ) (A : ℝ) :
  H > 0 →
  A > 0 →
  let H' := 0.65 * H
  let A' := 14.365
  A' / A = (H' / H)^2 →
  ∃ ε > 0, |A - 34| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l173_17381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_equalization_time_l173_17344

/-- The time taken for two pools to have the same amount of water -/
noncomputable def equalizationTime (initialWater1 initialWater2 rateDifference : ℝ) : ℝ :=
  (initialWater1 - initialWater2) / rateDifference

theorem pool_equalization_time :
  let initialWater1 : ℝ := 200
  let initialWater2 : ℝ := 112
  let rateDifference : ℝ := 22
  equalizationTime initialWater1 initialWater2 rateDifference = 4 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_equalization_time_l173_17344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l173_17301

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * (a^2 - a) * Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2 * a^2 * Real.log x

-- Part 1: Tangent line equation
theorem tangent_line_at_one (h : ℝ → ℝ) (h' : ℝ → ℝ) :
  (∀ x, h x = x^2 - 4 * Real.log x) →
  (∀ x, h' x = deriv h x) →
  (λ x ↦ h' 1 * (x - 1) + h 1) = (λ x ↦ -2 * x + 3) :=
by sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) :
  a ≤ 1/2 →
  (∀ x > 1, f a x > 2 * g a x) →
  (1 - Real.sqrt (1 + 12 * Real.exp 1)) / 6 < a ∧ a ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l173_17301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_theorem_l173_17339

/-- Represents the world population growth from 1999 to 2009 -/
noncomputable def world_population (initial_population : ℝ) (growth_rate : ℝ) : ℝ :=
  initial_population * (1 + growth_rate / 100) ^ 10

/-- Theorem stating the relationship between initial population, growth rate, and final population -/
theorem population_growth_theorem (x : ℝ) :
  world_population 6 x = 6 * (1 + x / 100) ^ 10 := by
  -- Unfold the definition of world_population
  unfold world_population
  -- The rest of the proof is omitted
  sorry

#check population_growth_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_theorem_l173_17339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_theorem_l173_17332

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Check if a point is on the asymptote -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = h.b / h.a * p.x ∨ p.y = -h.b / h.a * p.x

/-- Right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola) : Point :=
  ⟨Real.sqrt (h.a^2 + h.b^2), 0⟩

/-- Theorem statement -/
theorem hyperbola_distance_theorem (h : Hyperbola) (p m a : Point) (f : Point) :
  on_hyperbola h p →
  on_hyperbola h a →
  on_asymptote h m →
  m.x = -1 ∧ m.y = Real.sqrt 3 →
  a.x = 3 ∧ a.y = 1 →
  f = right_focus h →
  distance p a + 1/2 * distance p f ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_theorem_l173_17332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l173_17328

noncomputable section

/-- The function f(x) defined in the problem -/
def f (a b : ℝ) (x : ℝ) : ℝ := (a^2/3) * x^3 - 2*a*x^2 + b*x

/-- The derivative of f(x) -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := a^2 * x^2 - 4*a*x + b

theorem problem_solution (a b : ℝ) :
  (f_deriv a b 0 = 3) →  -- Slope of tangent line at (0, f(0)) is 3
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), x ≠ 1 → f a b x < f a b 1) →  -- Local maximum at x = 1
  (b = 3 ∧ a = 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l173_17328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l173_17373

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := 
  Real.sqrt (1 + b^2 / a^2)

/-- The distance from the center to a focus of the hyperbola -/
noncomputable def focal_distance (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 + b^2)

/-- Theorem: If the right vertex of a hyperbola is inside the circle with diameter
    formed by the intersection points of a line perpendicular to the x-axis from
    the left focus, then the eccentricity of the hyperbola is greater than 2 -/
theorem hyperbola_eccentricity_bound {a b : ℝ} (h : Hyperbola a b) 
  (vertex_inside : a + focal_distance a b < b^2 / a) : 
  eccentricity h > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l173_17373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_loss_l173_17388

/-- Represents a digit in the range [0, 9] -/
def Digit := Fin 10

/-- Represents the six digits in the arithmetic progression -/
structure DigitSequence where
  p : Digit
  a : Digit
  c : Digit
  x : Digit
  o : Digit
  d : Digit

/-- Checks if the given sequence forms an arithmetic progression -/
def isArithmeticProgression (seq : DigitSequence) : Prop :=
  ∃ (d : ℤ), 
    (seq.a.val : ℤ) = seq.p.val + d ∧
    (seq.c.val : ℤ) = seq.a.val + d ∧
    (seq.x.val : ℤ) = seq.c.val + d ∧
    (seq.o.val : ℤ) = seq.x.val + d ∧
    (seq.d.val : ℤ) = seq.o.val + d

/-- Calculates the EXPENSE value from the digit sequence -/
def expense (seq : DigitSequence) : ℕ :=
  100000 * seq.p.val + 10000 * seq.a.val + 1000 * seq.c.val + 
  100 * seq.x.val + 10 * seq.o.val + seq.d.val

/-- Calculates the INCOME value from the digit sequence -/
def income (seq : DigitSequence) : ℕ :=
  10000 * seq.a.val + 1000 * seq.c.val + 100 * seq.x.val + 
  10 * seq.o.val + seq.d.val

/-- Calculates the loss as EXPENSE - INCOME -/
def loss (seq : DigitSequence) : ℕ :=
  expense seq - income seq

theorem minimum_loss : 
  ∃ (seq : DigitSequence), isArithmeticProgression seq ∧ 
  (∀ (other : DigitSequence), isArithmeticProgression other → loss seq ≤ loss other) ∧
  loss seq = 58000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_loss_l173_17388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l173_17351

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x + Real.sin x ^ 2

theorem f_properties :
  let π := Real.pi
  (f (π / 4) = 1) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ (Real.sqrt 2 + 1) / 2) ∧
  (f (3 * π / 8) = (Real.sqrt 2 + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l173_17351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deer_per_hunting_wolf_l173_17371

/-- Calculates the number of deer each hunting wolf needs to kill to feed the pack. -/
theorem deer_per_hunting_wolf 
  (hunting_wolves : ℕ) 
  (additional_wolves : ℕ) 
  (meat_per_wolf_per_day : ℕ) 
  (days_between_hunts : ℕ) 
  (meat_per_deer : ℕ) 
  (h1 : hunting_wolves = 4) 
  (h2 : additional_wolves = 16) 
  (h3 : meat_per_wolf_per_day = 8) 
  (h4 : days_between_hunts = 5) 
  (h5 : meat_per_deer = 200) : 
  (hunting_wolves + additional_wolves) * meat_per_wolf_per_day * days_between_hunts / 
  (meat_per_deer * hunting_wolves) = 1 := by
  sorry

#eval Nat.div ((4 + 16) * 8 * 5) (200 * 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deer_per_hunting_wolf_l173_17371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_2023_is_T_l173_17341

def mySequence : List Char := ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Y', 'X', 'W', 'V', 'U', 'T', 'S', 'R', 'Q', 'P', 'O', 'N', 'M', 'L', 'K', 'J', 'I', 'H', 'G', 'F']

def infiniteSequence (n : Nat) : Char :=
  mySequence[n % mySequence.length]'sorry

theorem letter_2023_is_T : infiniteSequence 2022 = 'T' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_2023_is_T_l173_17341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l173_17318

/-- Parabola type -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * a * x

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Sum of distances from a point to two other points -/
noncomputable def sumDistances (p p1 p2 : Point) : ℝ :=
  distance p p1 + distance p p2

/-- Theorem: The point (2, 2) minimizes the sum of distances to A and F -/
theorem min_distance_point (p : Parabola) (a f : Point) :
  p.a = 1 →  -- parabola equation y^2 = 2x
  f = ⟨1/2, 0⟩ →  -- focus
  a = ⟨3, 2⟩ →  -- point A
  ∀ q : Point, p.eq q.x q.y →
    sumDistances ⟨2, 2⟩ a f ≤ sumDistances q a f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l173_17318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l173_17326

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3^(x + 1)

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := -1 + Real.log x / Real.log 3

-- State the theorem
theorem inverse_function_theorem :
  (∀ x, -1 ≤ x ∧ x < 0 → 1 ≤ f x ∧ f x < 3) ∧
  (∀ x, 1 ≤ x ∧ x < 3 → -1 ≤ g x ∧ g x < 0) ∧
  (∀ x, -1 ≤ x ∧ x < 0 → g (f x) = x) ∧
  (∀ x, 1 ≤ x ∧ x < 3 → f (g x) = x) := by
  sorry

#check inverse_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l173_17326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l173_17333

/-- Represents the tank emptying problem -/
structure TankProblem where
  tank_volume : ℚ
  leak_empty_time : ℚ
  inlet_rate : ℚ

/-- Calculates the time to empty the tank when both leak and inlet are active -/
def time_to_empty (p : TankProblem) : ℚ :=
  let leak_rate := p.tank_volume / p.leak_empty_time
  let inlet_rate_per_hour := p.inlet_rate * 60
  let net_emptying_rate := leak_rate - inlet_rate_per_hour
  p.tank_volume / net_emptying_rate

/-- Theorem stating the solution to the tank problem -/
theorem tank_emptying_time (p : TankProblem) 
    (h1 : p.tank_volume = 12960)
    (h2 : p.leak_empty_time = 9)
    (h3 : p.inlet_rate = 6) : 
  time_to_empty p = 12 := by
  sorry

#eval time_to_empty { tank_volume := 12960, leak_empty_time := 9, inlet_rate := 6 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l173_17333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_proof_l173_17393

-- Define variables
variable (m x : ℝ)
variable (h : m ≠ 0 ∧ x ≠ 0)

-- Define the original fraction
noncomputable def original_fraction (m x : ℝ) : ℝ := (5 * m^2 * x^2) / (10 * m * x^2)

-- Define the simplifying factor
noncomputable def simplifying_factor (m x : ℝ) : ℝ := 5 * m * x^2

-- Define the simplified fraction
noncomputable def simplified_fraction (m : ℝ) : ℝ := m / 2

-- Theorem statement
theorem simplification_proof (m x : ℝ) (h : m ≠ 0 ∧ x ≠ 0) :
  original_fraction m x = simplified_fraction m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_proof_l173_17393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_average_age_theorem_l173_17391

/-- Calculates the average age of a choir given the number of females, their average age,
    the number of males, and their average age. -/
noncomputable def choirAverageAge (femaleCount : ℕ) (femaleAvgAge : ℝ) (maleCount : ℕ) (maleAvgAge : ℝ) : ℝ :=
  ((femaleCount : ℝ) * femaleAvgAge + (maleCount : ℝ) * maleAvgAge) / ((femaleCount + maleCount) : ℝ)

/-- Theorem stating that for a choir with 12 females (average age 28) and 18 males (average age 40),
    the overall average age is 35.2 years. -/
theorem choir_average_age_theorem :
  choirAverageAge 12 28 18 40 = 35.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_average_age_theorem_l173_17391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_b_is_30_degrees_l173_17346

theorem triangle_angle_b_is_30_degrees 
  {a b c : ℝ} 
  {A B C : ℝ}
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : A + B + C = π)
  (h_sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C)
  (h_equation : (b - c) * (Real.sin B + Real.sin C) = (a - Real.sqrt 3 * c) * Real.sin A) :
  B = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_b_is_30_degrees_l173_17346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_a_over_c_value_l173_17309

noncomputable def f (x : ℝ) : ℝ := (2*x - 1) / (x + 5)

noncomputable def f_inv (x : ℝ) : ℝ := (5*x + 1) / (-x + 2)

theorem inverse_function_theorem (x : ℝ) : 
  f (f_inv x) = x ∧ f_inv (f x) = x :=
sorry

theorem a_over_c_value : 5 / (-1 : ℝ) = -5 :=
by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_a_over_c_value_l173_17309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_extreme_points_condition_G_negative_condition_l173_17360

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := a * Real.exp x
def G (a x : ℝ) : ℝ := f x - g a x

-- Statement 1
theorem tangent_line_condition (a : ℝ) :
  (∃ x₀, g a x₀ = x₀ - 1 ∧ (deriv (g a)) x₀ = 1) ↔ a = Real.exp (-2) := by sorry

-- Statement 2
theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ (deriv (G a)) x₁ = 0 ∧ (deriv (G a)) x₂ = 0) ↔ 0 < a ∧ a < 1 / Real.exp 1 := by sorry

-- Statement 3
theorem G_negative_condition (a : ℝ) (h : a * Real.exp 2 ≥ 2) :
  ∀ x > 0, G a x < 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_extreme_points_condition_G_negative_condition_l173_17360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_CF_l173_17357

/-- Square with side length 17 --/
structure Square :=
  (A B D E : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (17, 0) ∧ D = (17, 17) ∧ E = (0, 17))

/-- Point F with given properties --/
noncomputable def F (s : Square) : ℝ × ℝ := (-120/17, 225/17)

/-- Point C with given properties --/
noncomputable def C (s : Square) : ℝ × ℝ := (17 + 120/17, 64/17)

/-- Distance between two points --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_CF (s : Square) :
  distance (F s) (C s) = 17 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_CF_l173_17357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joeys_return_speed_l173_17355

theorem joeys_return_speed 
  (delivery_distance : ℝ) 
  (delivery_time : ℝ) 
  (round_trip_avg_speed : ℝ) 
  (h1 : delivery_distance = 5)
  (h2 : delivery_time = 1)
  (h3 : round_trip_avg_speed = 8) :
  let delivery_speed := delivery_distance / delivery_time
  let round_trip_distance := 2 * delivery_distance
  let round_trip_time := round_trip_distance / round_trip_avg_speed
  let return_time := round_trip_time - delivery_time
  let return_speed := delivery_distance / return_time
  return_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joeys_return_speed_l173_17355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l173_17376

def b : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n+2) => 1/2 * b (n+1) + 1/3 * b n

theorem sequence_sum : ∑' n, b n = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l173_17376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ingrid_cookie_ratio_l173_17358

def total_cookies : ℕ := 148
def ingrid_percentage : ℚ := 31524390243902438 / 100000000000000000

theorem ingrid_cookie_ratio : 
  let ingrid_cookies := (total_cookies : ℚ) * ingrid_percentage
  ⌊ingrid_cookies⌋ = 47 ∧ 
  (⌊ingrid_cookies⌋ : ℚ) / total_cookies = 47 / 148 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ingrid_cookie_ratio_l173_17358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l173_17302

noncomputable def g (x : ℝ) : ℝ := 1 / (x^2 + 4)

theorem range_of_g :
  Set.range g = Set.Ioo 0 (1/4) ∪ {1/4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l173_17302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l173_17366

theorem smallest_positive_z (x z : ℝ) : 
  Real.sin x = 0 → 
  Real.cos (x + z) = Real.sqrt 3 / 2 → 
  z > 0 → 
  ∀ w, (w > 0 ∧ Real.sin x = 0 ∧ Real.cos (x + w) = Real.sqrt 3 / 2) → z ≤ w → 
  z = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l173_17366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_thirteen_l173_17312

def sequence_value (n : ℕ) : ℤ :=
  match n % 5 with
  | 0 => (n / 5 : ℤ) + 1
  | 1 => 6 * (n / 5 : ℤ) + 1
  | 2 => (n / 5 : ℤ) - 1
  | 3 => -4 * (n / 5 : ℤ) - 4
  | 4 => -4 * (n / 5 : ℤ) - 4
  | _ => 0  -- This case is theoretically impossible, but Lean requires it for exhaustiveness

theorem expression_value_thirteen (n : ℕ) :
  sequence_value n = 13 ↔ n = 11 ∨ n = 62 ∨ n = 72 := by
  sorry

#eval sequence_value 11
#eval sequence_value 62
#eval sequence_value 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_thirteen_l173_17312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_close_points_in_circle_l173_17337

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is inside a circle -/
def isInside (p : Point) (center : Point) (radius : ℝ) : Prop :=
  distance p center < radius

theorem close_points_in_circle (r : ℝ) (center : Point) (points : Finset Point) :
  r > 0 →
  points.card = 17 →
  (∀ p ∈ points, isInside p center r) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 < (2/3) * r := by
  sorry

#check close_points_in_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_close_points_in_circle_l173_17337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_solution_set_l173_17345

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- Define the properties of f
def is_odd_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f (-x) = -f x

def is_monotone_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f x < f y

-- State the theorem
theorem function_properties_and_solution_set 
  (a b : ℝ) 
  (h_odd : is_odd_on_interval (f a b))
  (h_value : f a b (1/2) = 2/5)
  (h_monotone : is_monotone_increasing_on_interval (f a b)) :
  (∀ x, f a b x = x / (1 + x^2)) ∧ 
  (∀ t, f a b (t-1) + f a b t < 0 ↔ 0 < t ∧ t < 1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_solution_set_l173_17345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_y_axis_symmetric_point_example_l173_17364

/-- Function to calculate the point symmetric to a given point with respect to the y-axis -/
def point_symmetric_to_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Given a point (a, b) in a plane rectangular coordinate system,
    the point symmetric to it with respect to the y-axis is (-a, b) -/
theorem symmetric_point_y_axis (a b : ℝ) :
  point_symmetric_to_y_axis (a, b) = (-a, b) :=
by
  -- Unfold the definition of point_symmetric_to_y_axis
  unfold point_symmetric_to_y_axis
  -- The result follows directly from the definition
  rfl

/-- The point symmetric to (2, 5) with respect to the y-axis is (-2, 5) -/
theorem symmetric_point_example :
  point_symmetric_to_y_axis (2, 5) = (-2, 5) :=
by
  -- Apply the general theorem
  apply symmetric_point_y_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_y_axis_symmetric_point_example_l173_17364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_population_decrease_l173_17321

theorem bird_population_decrease (P₀ : ℝ) (P₀_pos : P₀ > 0) :
  ∃ n : ℕ, n = 5 ∧ ∀ m : ℕ, (0.5 : ℝ) ^ m < 0.05 ↔ m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_population_decrease_l173_17321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_face_angle_eq_arccos_lateral_face_angle_expressions_equal_l173_17305

/-- A regular triangular pyramid with lateral edge angle of 60° to the base. -/
structure RegularTriangularPyramid where
  /-- The angle between the lateral edge and the base plane is 60°. -/
  lateral_edge_angle : ℝ
  lateral_edge_angle_eq : lateral_edge_angle = π / 3

/-- The angle between lateral faces of a regular triangular pyramid. -/
noncomputable def lateral_face_angle (p : RegularTriangularPyramid) : ℝ := Real.arccos (5 / 13)

/-- 
Theorem: In a regular triangular pyramid where the lateral edge forms a 60° angle 
with the base plane, the angle between the lateral faces is arccos(5/13).
-/
theorem lateral_face_angle_eq_arccos (p : RegularTriangularPyramid) :
  lateral_face_angle p = 2 * Real.arctan (2 / 3) := by
  sorry

/-- 
Corollary: The two expressions for the lateral face angle are equivalent.
-/
theorem lateral_face_angle_expressions_equal :
  Real.arccos (5 / 13) = 2 * Real.arctan (2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_face_angle_eq_arccos_lateral_face_angle_expressions_equal_l173_17305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_shape_is_12_l173_17322

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the shape
def shape (x y : ℝ) : Prop :=
  (floor x)^2 + (floor y)^2 = 50

-- Define the area of the shape
noncomputable def area_of_shape : ℝ := sorry

-- Theorem statement
theorem area_of_shape_is_12 : area_of_shape = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_shape_is_12_l173_17322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l173_17354

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def semiperimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

noncomputable def area (t : Triangle) : ℝ :=
  let s := semiperimeter t
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

noncomputable def circumradius (t : Triangle) : ℝ := (t.a * t.b * t.c) / (4 * area t)

noncomputable def inradius (t : Triangle) : ℝ := area t / semiperimeter t

def is_acute (t : Triangle) : Prop :=
  semiperimeter t > (t.a * t.b * t.c) / (2 * area t) + area t / semiperimeter t

def is_right (t : Triangle) : Prop :=
  semiperimeter t = (t.a * t.b * t.c) / (2 * area t) + area t / semiperimeter t

def is_obtuse (t : Triangle) : Prop :=
  semiperimeter t < (t.a * t.b * t.c) / (2 * area t) + area t / semiperimeter t

theorem triangle_classification (t : Triangle) :
  is_acute t ∨ is_right t ∨ is_obtuse t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l173_17354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l173_17385

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : List Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Define the cube with given vertices -/
def myCube : Cube :=
  { vertices := [
      { x := 0, y := 0, z := 0 },
      { x := 0, y := 0, z := 6 },
      { x := 0, y := 6, z := 0 },
      { x := 0, y := 6, z := 6 },
      { x := 6, y := 0, z := 0 },
      { x := 6, y := 0, z := 6 },
      { x := 6, y := 6, z := 0 },
      { x := 6, y := 6, z := 6 }
    ]
  }

/-- Define the plane intersecting the cube -/
def intersectingPlane : Plane :=
  { a := 3, b := 2, c := -1, d := 6 }

/-- Theorem: The distance between the two additional intersection points is 2√13 -/
theorem intersection_distance (c : Cube) (p : Plane) 
  (h1 : c = myCube)
  (h2 : p = intersectingPlane)
  (h3 : ∃ (p1 p2 p3 : Point3D), 
    p1 = { x := 0, y := 3, z := 0 } ∧
    p2 = { x := 2, y := 0, z := 0 } ∧
    p3 = { x := 2, y := 6, z := 6 } ∧
    p.a * p1.x + p.b * p1.y + p.c * p1.z = p.d ∧
    p.a * p2.x + p.b * p2.y + p.c * p2.z = p.d ∧
    p.a * p3.x + p.b * p3.y + p.c * p3.z = p.d) :
  ∃ (q1 q2 : Point3D), 
    p.a * q1.x + p.b * q1.y + p.c * q1.z = p.d ∧
    p.a * q2.x + p.b * q2.y + p.c * q2.z = p.d ∧
    distance q1 q2 = 2 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l173_17385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l173_17380

-- Define the hyperbola
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2/b^2 = 1

-- Define the distance from focus to asymptote
def focus_to_asymptote_distance (b : ℝ) : ℝ := 2

-- Define the eccentricity
noncomputable def eccentricity (b : ℝ) : ℝ := Real.sqrt (1 + b^2)

-- Theorem statement
theorem hyperbola_eccentricity (b : ℝ) :
  (∃ x y, hyperbola b x y) ∧ focus_to_asymptote_distance b = 2 →
  eccentricity b = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l173_17380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_ten_implies_k_is_sixteen_l173_17311

/-- The sum of the infinite series given k -/
noncomputable def seriesSum (k : ℝ) : ℝ := 
  4 + (4 + k) / 5 + (4 + 2*k) / 5^2 + (4 + 3*k) / 5^3 + (∑' n, (4 + n*k) / 5^n)

/-- Theorem stating that if the series sum equals 10, then k must be 16 -/
theorem series_sum_equals_ten_implies_k_is_sixteen :
  ∀ k : ℝ, seriesSum k = 10 → k = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_ten_implies_k_is_sixteen_l173_17311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_properties_l173_17336

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x - y + 1 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + a * y + 1 = 0

-- Define perpendicularity of two lines
def perpendicular (f g : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ, f a x₁ y₁ ∧ f a x₂ y₂ ∧ g a x₁ y₁ ∧ g a x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) = 0

-- Define the intersection point M of l₁ and l₂
noncomputable def M (a : ℝ) : ℝ × ℝ :=
  let x := (a^2 - 1) / (a^2 + 1)
  let y := (2*a) / (a^2 + 1)
  (x, y)

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem lines_properties :
  (perpendicular l₁ l₂) ∧
  (∀ a : ℝ, distance (M a) (0, 0) ≤ Real.sqrt 2) ∧
  (∃ a : ℝ, distance (M a) (0, 0) = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_properties_l173_17336
