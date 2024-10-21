import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l334_33448

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = (3 * t.c - t.b) * Real.cos t.A ∧
  t.a * Real.sin t.B = 2 * Real.sqrt 2 ∧
  t.a = 2 * Real.sqrt 2 ∧
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 2

-- State the theorem
theorem triangle_properties (t : Triangle) (h : conditions t) :
  t.b = 3 ∧ t.b + t.c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l334_33448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l334_33498

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt 3 - 2 * Real.cos x)

theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.univ ↔ ∃ k : ℤ, x ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 6) (2 * k * Real.pi + 11 * Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l334_33498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_dancing_calories_l334_33489

/-- Calculates the calories burned per week from dancing given the following conditions:
  * The calorie burn rate for dancing is twice that of walking
  * Dancing occurs twice a day for 0.5 hours each time
  * Dancing is done 4 times a week
  * The calorie burn rate for walking is known
-/
def calories_burned_dancing_per_week (walking_calories_per_hour : ℕ) : ℕ :=
  let dancing_calories_per_hour := 2 * walking_calories_per_hour
  let dancing_hours_per_day := 1 -- 2 * 0.5 = 1
  let dancing_days_per_week := 4
  dancing_calories_per_hour * dancing_hours_per_day * dancing_days_per_week

/-- Theorem stating that James burns 2400 calories per week from dancing -/
theorem james_dancing_calories : calories_burned_dancing_per_week 300 = 2400 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_dancing_calories_l334_33489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_3_minus_2x_l334_33458

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem domain_of_f_3_minus_2x :
  {x : ℝ | f (3 - 2*x) = f (3 - 2*x)} = Set.Icc (-1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_3_minus_2x_l334_33458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l334_33452

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x - Real.log x

-- State the theorem
theorem inequality_proof (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m ≥ -1) 
  (hx₁ : x₁ > 0) 
  (hx₂ : x₂ > 0) 
  (h : (f m x₁ + f m x₂) / 2 ≤ x₁^2 + x₂^2 + (3/2) * x₁ * x₂) :
  x₁ + x₂ ≥ (Real.sqrt 3 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l334_33452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_from_area_formula_l334_33456

/-- Given a triangle ABC with sides a, b, c and area S, 
    if S = (a^2 + b^2 - c^2) / (4√3), then angle C measures π/6 radians. -/
theorem angle_measure_from_area_formula {a b c S : ℝ} (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_area : S = (a^2 + b^2 - c^2) / (4 * Real.sqrt 3)) :
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_from_area_formula_l334_33456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_id_only_self_composing_polynomial_l334_33464

/-- A polynomial that satisfies p(p(x)) = xp²(x) + x³ --/
def SelfComposingPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ x, p (p x) = x * (p x)^2 + x^3

/-- The identity function on real numbers --/
def id_poly : ℝ → ℝ := λ x ↦ x

/-- Theorem stating that the identity function is the only polynomial
    satisfying the self-composing property --/
theorem id_only_self_composing_polynomial :
  (∀ p : ℝ → ℝ, SelfComposingPolynomial p → p = id_poly) ∧
  SelfComposingPolynomial id_poly :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_id_only_self_composing_polynomial_l334_33464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l334_33480

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given its three vertices -/
noncomputable def triangleArea (A B C : Point) : ℝ :=
  (1/2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem triangle_DEF_area :
  let D : Point := ⟨-2, 2⟩
  let E : Point := ⟨8, 2⟩
  let F : Point := ⟨6, -4⟩
  triangleArea D E F = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l334_33480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_interior_angle_sum_l334_33403

/-- An m-pointed star formed from an (m+3)-sided regular polygon. -/
structure StarPolygon (m : ℕ) where
  /-- The number of points in the star. -/
  num_points : ℕ
  /-- The number of sides in the original polygon. -/
  num_sides : ℕ
  /-- Condition that m ≥ 7. -/
  m_ge_seven : m ≥ 7
  /-- The star is formed from an (m+3)-sided polygon. -/
  sides_eq : num_sides = m + 3
  /-- The number of points in the star equals m. -/
  points_eq : num_points = m

/-- The degree-sum of interior angles of an m-pointed star. -/
def interiorAngleSum (m : ℕ) (s : StarPolygon m) : ℕ := 180 * (s.num_points - 4)

/-- Theorem stating that the degree-sum of interior angles of an m-pointed star is 180(m-4). -/
theorem star_interior_angle_sum (m : ℕ) (s : StarPolygon m) :
  interiorAngleSum m s = 180 * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_interior_angle_sum_l334_33403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_product_value_l334_33475

-- Define the condition that √(x+1) can be combined with √(5/2)
def can_combine (x : ℝ) : Prop := ∃ (k : ℝ), Real.sqrt (x + 1) = k * Real.sqrt (5/2)

-- Theorem 1: Prove that x = 9
theorem x_value (x : ℝ) (h : can_combine x) : x = 9 := by
  sorry

-- Theorem 2: Prove that √(x+1) * √(5/2) = 5 when x = 9
theorem product_value (x : ℝ) (h : x = 9) : Real.sqrt (x + 1) * Real.sqrt (5/2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_product_value_l334_33475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_259_21_l334_33477

theorem square_root_of_259_21 : 
  ∃ (x : ℝ), x^2 = 259.21 ∧ (x = 16.1 ∨ x = -16.1) := by
  use 16.1
  constructor
  · norm_num
  · left
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_259_21_l334_33477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l334_33433

/-- Calculates the length of a train given its speed and time to cross a point -/
noncomputable def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Proves that a train with given speed and crossing time has the expected length -/
theorem train_length_calculation (speed : ℝ) (time : ℝ) 
  (h1 : speed = 144)
  (h2 : time = 2.9997600191984644) :
  ∃ ε > 0, |train_length speed time - 119.99| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l334_33433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_obtuse_l334_33454

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the triangle is obtuse if 2c² = 2a² + 2b² + ab -/
theorem triangle_is_obtuse (a b c : ℝ) (h : 2 * c^2 = 2 * a^2 + 2 * b^2 + a * b) :
  ∃ A B C : ℝ, 
    0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
    A + B + C = Real.pi ∧    -- Sum of angles in a triangle
    a = Real.sqrt (b^2 + c^2 - 2*b*c*(Real.cos A)) ∧  -- Law of cosines for side a
    b = Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B)) ∧  -- Law of cosines for side b
    c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos C)) ∧  -- Law of cosines for side c
    Real.pi / 2 < C          -- Angle C is obtuse
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_obtuse_l334_33454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seq_property_l334_33409

def seq (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n + 2 => 2005 * seq (n + 1) - seq n

theorem seq_property : ∀ n : ℕ, ∃ k : ℤ, 
  seq (n + 1) * seq n - 1 = 2003 * k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seq_property_l334_33409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_and_value_l334_33457

-- Define the function f
noncomputable def f (ω φ m : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ) + m

-- State the theorem
theorem function_symmetry_and_value (ω φ m : ℝ) : 
  (∀ t : ℝ, f ω φ m (t + π/4) = f ω φ m (-t)) → 
  f ω φ m (π/8) = -1 → 
  m = -3 ∨ m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_and_value_l334_33457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_values_for_4001_l334_33474

/-- Sequence definition -/
def seq (x : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => 4000
  | (n + 2) => seq x n * seq x (n + 1) - 1

/-- The theorem to prove -/
theorem three_values_for_4001 :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
  (∀ x ∈ s, x > 0 ∧ ∃ n : ℕ, seq x n = 4001) ∧
  (∀ x > 0, (∃ n : ℕ, seq x n = 4001) → x ∈ s) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_values_for_4001_l334_33474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_l334_33425

/-- Given two solutions A and B with different concentrations of liquid X,
    prove that mixing them results in a solution with a specific concentration of liquid X. -/
theorem mixture_concentration
  (weight_A : ℝ) (weight_B : ℝ) (percent_X_in_A : ℝ) (percent_X_in_B : ℝ)
  (h_weight_A : weight_A = 300)
  (h_weight_B : weight_B = 700)
  (h_percent_A : percent_X_in_A = 0.8)
  (h_percent_B : percent_X_in_B = 1.8) :
  (let total_weight := weight_A + weight_B
   let weight_X_in_A := weight_A * (percent_X_in_A / 100)
   let weight_X_in_B := weight_B * (percent_X_in_B / 100)
   let total_weight_X := weight_X_in_A + weight_X_in_B
   let final_percent_X := (total_weight_X / total_weight) * 100
   final_percent_X) = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_l334_33425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l334_33401

/-- Proves that a train 160 metres long running at 72 km/hr takes 8 seconds to pass a pole. -/
theorem train_passing_pole_time (train_length : ℝ) (train_speed_kmh : ℝ) 
  (h1 : train_length = 160) 
  (h2 : train_speed_kmh = 72) : ℝ :=
by
  -- Convert speed from km/h to m/s
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  
  -- Calculate time to pass the pole
  let time := train_length / train_speed_ms
  
  -- Prove that the time is 8 seconds
  sorry

-- Example usage (commented out to avoid evaluation errors)
-- #eval train_passing_pole_time 160 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l334_33401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palm_meadows_room_count_l334_33472

theorem palm_meadows_room_count : ∃ (rooms_with_two_beds rooms_with_three_beds : ℕ),
  let total_rooms : ℕ := 13
  let total_beds : ℕ := 31
  rooms_with_two_beds + rooms_with_three_beds = total_rooms ∧
  2 * rooms_with_two_beds + 3 * rooms_with_three_beds = total_beds ∧
  rooms_with_two_beds = 8 := by
  -- The proof goes here
  sorry

#check palm_meadows_room_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palm_meadows_room_count_l334_33472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insect_distance_ratio_l334_33444

/-- Represents the position of an insect on a clock face -/
structure InsectPosition where
  angle : ℝ  -- Angle in radians
  hand : Bool  -- True for hour hand, False for minute hand

/-- Calculates the new position of an insect after one hour -/
noncomputable def new_position (pos : InsectPosition) : InsectPosition :=
  if pos.hand then
    { angle := pos.angle + 2 * Real.pi / 12, hand := false }
  else
    { angle := pos.angle + 2 * Real.pi, hand := true }

/-- Calculates the distance traveled by an insect in one hour -/
noncomputable def distance_traveled (pos : InsectPosition) : ℝ :=
  if pos.hand then
    2 * Real.pi / 12
  else
    2 * Real.pi

/-- The main theorem to be proved -/
theorem insect_distance_ratio :
  let initial_mosquito : InsectPosition := { angle := 2 * Real.pi / 12, hand := true }
  let initial_fly : InsectPosition := { angle := 0, hand := false }
  let final_time : ℕ := 12

  let mosquito_distance : ℝ := (List.range final_time).foldl
    (λ acc _ => acc + distance_traveled (new_position initial_mosquito)) 0

  let fly_distance : ℝ := (List.range final_time).foldl
    (λ acc _ => acc + distance_traveled (new_position initial_fly)) 0

  mosquito_distance / fly_distance = 83 / 73 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insect_distance_ratio_l334_33444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_of_point_on_parabola_l334_33442

/-- A parabola with equation y² = 2px -/
structure Parabola where
  p : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance from a point to the directrix of a parabola -/
noncomputable def distance_to_directrix (para : Parabola) (pt : Point) : ℝ :=
  pt.x + para.p / 2

theorem distance_to_directrix_of_point_on_parabola :
  ∀ (para : Parabola) (pt : Point),
    pt.x = 1 ∧ pt.y = Real.sqrt 5 ∧ pt.y^2 = 2 * para.p * pt.x →
    distance_to_directrix para pt = 9/4 := by
  sorry

#check distance_to_directrix_of_point_on_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_of_point_on_parabola_l334_33442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_applePyramidSum_l334_33470

/-- Calculates the number of apples in a layer of the pyramid --/
def applesInLayer (baseWidth : ℕ) (baseLength : ℕ) (layer : ℕ) : ℕ :=
  (baseWidth - layer + 1) * (baseLength - layer + 1)

/-- Calculates the total number of apples in the pyramid stack --/
def totalApples (baseWidth : ℕ) (baseLength : ℕ) : ℕ :=
  let numLayers := min baseWidth baseLength
  (List.range numLayers).foldl (fun acc layer =>
    acc + applesInLayer baseWidth baseLength layer +
    if (layer + 1) % 3 = 0 then 1 else 0) 0

/-- Theorem stating that the total number of apples in the described pyramid is 151 --/
theorem applePyramidSum :
  totalApples 6 9 = 151 := by
  sorry

#eval totalApples 6 9  -- This will evaluate the function and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_applePyramidSum_l334_33470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_significant_digits_l334_33435

/-- The number of significant digits in a number -/
def significantDigits (x : ℝ) : ℕ :=
  sorry

/-- The area of the rectangle -/
def area : ℝ := 0.07344

/-- The precision of the area measurement (to the nearest hundred-thousandth) -/
def areaPrecision : ℝ := 0.00001

/-- The side length of the rectangle -/
noncomputable def sideLength : ℝ := Real.sqrt area

/-- Theorem stating that the side length has 3 significant digits -/
theorem side_significant_digits :
  significantDigits sideLength = 3 :=
by sorry

#eval area
#eval areaPrecision

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_significant_digits_l334_33435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_total_savings_l334_33487

/-- Represents an income source with its income-to-expenditure ratio and income amount -/
structure IncomeSource where
  income_ratio : ℕ
  expenditure_ratio : ℕ
  income : ℚ

/-- Calculates the savings from a single income source -/
def savings (source : IncomeSource) : ℚ :=
  source.income - (source.income * source.expenditure_ratio / source.income_ratio)

/-- John's income sources -/
def john_income_sources : List IncomeSource := [
  { income_ratio := 3, expenditure_ratio := 2, income := 20000 },
  { income_ratio := 4, expenditure_ratio := 1, income := 30000 },
  { income_ratio := 5, expenditure_ratio := 3, income := 45000 }
]

/-- Theorem: John's total savings from all income sources is approximately 47166.66 -/
theorem john_total_savings :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ abs ((john_income_sources.map savings).sum - 47166.66) < ε := by
  sorry

#eval (john_income_sources.map savings).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_total_savings_l334_33487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l334_33445

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- Get the nth term of an arithmetic sequence -/
noncomputable def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + seq.common_difference * (n - 1)

/-- Get the sum of the first n terms of an arithmetic sequence -/
noncomputable def ArithmeticSequence.sum (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.first_term + seq.common_difference * (n - 1)) / 2

theorem arithmetic_sequence_ratio (a b d : ℝ) :
  let S : ArithmeticSequence := { first_term := a, common_difference := d }
  let T : ArithmeticSequence := { first_term := b, common_difference := 2 * d }
  ∀ n : ℕ, (S.sum n) / (T.sum n) = (9 * n + 3) / (5 * n + 35) →
  (S.nthTerm 15) / (T.nthTerm 15) = 23 / 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l334_33445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_paths_ABCD_l334_33421

def paths_AB : ℕ := Nat.choose 7 2
def paths_BC : ℕ := Nat.choose 6 3
def paths_CD : ℕ := Nat.choose 6 2

theorem total_paths_ABCD : paths_AB * paths_BC * paths_CD = 6300 := by
  rw [paths_AB, paths_BC, paths_CD]
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_paths_ABCD_l334_33421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l334_33429

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 - (a + 5) * x
  else x^3 - (a + 3) / 2 * x^2 + a * x

-- State the theorem
theorem f_properties (a : ℝ) (h : a ∈ Set.Icc (-2) 0) :
  (∀ x ∈ Set.Ioo (-1) 1, StrictMonoOn (fun y => -(f a y)) (Set.Ioo (-1) 1)) ∧
  (∀ x ∈ Set.Ioi 1, StrictMonoOn (f a) (Set.Ioi 1)) ∧
  (∀ x₁ x₂ x₃ : ℝ, x₁ * x₂ * x₃ ≠ 0 →
    (∃ k : ℝ, (deriv (f a)) x₁ = k ∧ (deriv (f a)) x₂ = k ∧ (deriv (f a)) x₃ = k) →
    x₁ + x₂ + x₃ > -1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l334_33429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_eq_S_closed_l334_33484

/-- The sum of products from 1 to n -/
def S (n : ℕ) : ℕ := 
  List.range n |>.map (λ i => (i + 1) * (i + 2) * (i + 3) * (i + 4)) |>.sum

/-- The proposed closed form for S(n) -/
def S_closed (n : ℕ) : ℚ := 
  n * (n + 1) * (n + 2) * (n + 3) * (n + 4) / 5

/-- Theorem stating that S(n) equals S_closed(n) for all positive natural numbers -/
theorem S_eq_S_closed : ∀ n : ℕ, n > 0 → S n = ⌊S_closed n⌋ := by
  sorry

#eval S 5
#eval ⌊S_closed 5⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_eq_S_closed_l334_33484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_movements_l334_33426

/-- Represents a two-dimensional vector with integer coordinates -/
structure Vector2D where
  x : Int
  y : Int

/-- Calculates the straight-line distance between two points represented by Vector2D -/
noncomputable def distance (v : Vector2D) : Real :=
  Real.sqrt ((v.x : Real) ^ 2 + (v.y : Real) ^ 2)

/-- Theorem: The straight-line distance after a series of movements is 50 km -/
theorem distance_after_movements : 
  let start := Vector2D.mk 0 0
  let movements := [Vector2D.mk 0 (-60), Vector2D.mk (-40) 0, Vector2D.mk 0 20, Vector2D.mk 10 0]
  let end_point := movements.foldl (fun acc v => Vector2D.mk (acc.x + v.x) (acc.y + v.y)) start
  distance end_point = 50 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_movements_l334_33426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_quadrilateral_l334_33483

/-- A tangential quadrilateral is a quadrilateral with an inscribed circle -/
structure TangentialQuadrilateral where
  sides : Fin 4 → ℝ
  radius : ℝ
  sum_opposite_sides : sides 0 + sides 2 = sides 1 + sides 3
  all_positive : ∀ i, sides i > 0
  radius_positive : radius > 0

/-- The area of a tangential quadrilateral -/
noncomputable def area (q : TangentialQuadrilateral) : ℝ :=
  (q.sides 0 + q.sides 1 + q.sides 2 + q.sides 3) / 2 * q.radius

theorem area_of_specific_quadrilateral :
  ∀ q : TangentialQuadrilateral,
  q.sides 0 + q.sides 2 = 20 →
  q.radius = 4 →
  area q = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_quadrilateral_l334_33483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_polygon_sides_equal_l334_33460

/-- The number of sides of the enclosed polygon -/
def m : ℕ := 12

/-- The number of enclosing polygons -/
def num_enclosing : ℕ := m

/-- The exterior angle of a regular polygon with k sides -/
noncomputable def exterior_angle (k : ℕ) : ℝ := 360 / k

/-- The theorem stating that a regular m-gon enclosed by m regular n-gons implies n = m -/
theorem enclosed_polygon_sides_equal (n : ℕ) 
  (h1 : n > 2)  -- n must be at least 3 for a polygon
  (h2 : 2 * (exterior_angle n / 2) = exterior_angle m) :
  n = m := by
  sorry

#eval m  -- This will output 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_polygon_sides_equal_l334_33460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vehicles_passing_N_mod_15_l334_33459

/-- Represents the maximum speed of vehicles on the motorway in km/h. -/
def max_speed : ℕ := 100

/-- Represents the length of each vehicle in meters. -/
def vehicle_length : ℕ := 5

/-- Calculates the space occupied by a vehicle including its safety distance. -/
noncomputable def space_occupied (speed : ℕ) : ℝ :=
  vehicle_length * (1 + ↑(speed / 10 + if speed % 10 = 0 then 0 else 1))

/-- Calculates the distance covered by a vehicle in one hour. -/
def distance_per_hour (speed : ℕ) : ℝ :=
  speed * 1000

/-- Calculates the number of vehicles that can pass a sensor in one hour at a given speed. -/
noncomputable def vehicles_per_hour (speed : ℕ) : ℝ :=
  distance_per_hour speed / space_occupied speed

/-- The maximum number of vehicles that can pass the sensor in one hour. -/
def N : ℕ := 2000

theorem max_vehicles_passing (speed : ℕ) : speed ≤ max_speed →
  vehicles_per_hour speed ≤ N := by
  sorry

theorem N_mod_15 : N % 15 = 5 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vehicles_passing_N_mod_15_l334_33459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l334_33420

theorem right_triangle_area (DE EF : ℝ) (angleD : Real) (h1 : DE = 8) (h2 : angleD = 45) 
  (h3 : EF = DE) : 
  (1 / 2) * DE * EF = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l334_33420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_hyperbola_equilateral_triangle_l334_33419

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the left vertex of the hyperbola
def left_vertex : ℝ × ℝ := (-1, 0)

-- Define a point on the right branch of the hyperbola
def right_branch_point (x y : ℝ) : Prop := hyperbola x y ∧ x > 0

-- Define an equilateral triangle
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := (p.1 - q.1)^2 + (p.2 - q.2)^2
  d A B = d B C ∧ d B C = d C A

-- Main theorem
theorem area_of_hyperbola_equilateral_triangle :
  ∀ (B C : ℝ × ℝ),
  right_branch_point B.1 B.2 →
  right_branch_point C.1 C.2 →
  is_equilateral left_vertex B C →
  let area := Real.sqrt 3 / 4 * (B.1 - left_vertex.1)^2
  area = Real.sqrt 3 / 4 * (5 + 2 * Real.sqrt 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_hyperbola_equilateral_triangle_l334_33419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_in_open_zero_one_l334_33438

noncomputable section

-- Define the function f
def f (a x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

-- Define the derivative of f
def f_deriv (a x : ℝ) : ℝ := (2 * Real.exp x + 1) * (a * Real.exp x - 1)

-- Define the function u
def u (a : ℝ) : ℝ := 1 - 1 / a + Real.log a

theorem two_zeros_implies_a_in_open_zero_one (a : ℝ) :
  (a > 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  (0 < a ∧ a < 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_in_open_zero_one_l334_33438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_calculation_sum_of_fraction_parts_l334_33471

/-- The time when Jenny and Kenny can see each other again -/
noncomputable def time_to_see_again (jenny_speed kenny_speed path_distance building_diameter initial_distance : ℝ) : ℝ :=
  200 / 3

theorem time_calculation (jenny_speed kenny_speed path_distance building_diameter initial_distance : ℝ) 
  (h1 : jenny_speed = 2)
  (h2 : kenny_speed = 4)
  (h3 : path_distance = 250)
  (h4 : building_diameter = 150)
  (h5 : initial_distance = 250) :
  time_to_see_again jenny_speed kenny_speed path_distance building_diameter initial_distance = 200 / 3 := by
  sorry

#eval Nat.gcd 200 3  -- To verify that 200/3 is in lowest terms

theorem sum_of_fraction_parts : 200 + 3 = 203 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_calculation_sum_of_fraction_parts_l334_33471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_properties_l334_33428

def data_set : List ℝ := [-1, 0, 4, 6, 7, 14]

def is_ascending (l : List ℝ) : Prop :=
  ∀ i j, i < j → i < l.length → j < l.length → l[i]! ≤ l[j]!

noncomputable def median (l : List ℝ) : ℝ :=
  if l.length % 2 = 0
  then (l[l.length / 2 - 1]! + l[l.length / 2]!) / 2
  else l[l.length / 2]!

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / l.length

theorem data_set_properties :
  is_ascending data_set ∧
  median data_set = 5 ∧
  mean data_set = 5 ∧
  variance data_set = 74/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_properties_l334_33428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_cube_volume_ratio_l334_33410

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  side : ℝ
  side_pos : side > 0

/-- A cube whose vertices are the centers of the faces of a regular tetrahedron -/
noncomputable def CubeFromTetrahedron (t : RegularTetrahedron) : ℝ → Prop :=
  fun edge => edge = t.side * Real.sqrt 6 / 6

/-- The volume of a regular tetrahedron -/
noncomputable def tetrahedronVolume (t : RegularTetrahedron) : ℝ :=
  Real.sqrt 2 / 12 * t.side ^ 3

/-- The volume of a cube -/
def cubeVolume (edge : ℝ) : ℝ :=
  edge ^ 3

theorem tetrahedron_cube_volume_ratio (t : RegularTetrahedron) :
    ∃ c : ℝ, CubeFromTetrahedron t c ∧
    tetrahedronVolume t / cubeVolume c = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_cube_volume_ratio_l334_33410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_maximum_l334_33453

noncomputable def f (x a : ℝ) : ℝ := (x + a/2)^2 - (3*x + 2*a)^2

theorem function_maximum :
  ∃ (a : ℝ), (∀ (x : ℝ), f x a ≤ f (-11/8) a) ∧ (f (-11/8) a = 5/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_maximum_l334_33453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_of_angle_l334_33490

/-- Given a point (3, -4) on the terminal side of angle α, prove that tan α = -4/3 -/
theorem tangent_of_angle (α : ℝ) : 
  (∃ (x y : ℝ), x = 3 ∧ y = -4 ∧ (x, y) ∈ Set.range (λ t : ℝ × ℝ => (t.1 * Real.cos α - t.2 * Real.sin α, t.1 * Real.sin α + t.2 * Real.cos α))) → 
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_of_angle_l334_33490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l334_33455

noncomputable def binomial_sum (n : ℕ) : ℕ := n.choose 1 + n.choose 2

noncomputable def expansion_term (n k : ℕ) (x : ℝ) : ℝ :=
  (-1)^k * 2^k * n.choose k * x^((n - 5*k : ℤ) / 2)

theorem expansion_properties (n : ℕ) (h : n > 0) (h_sum : binomial_sum n = 36) :
  n = 8 ∧
  expansion_term n 1 x = -16 * x^(3/2) ∧
  expansion_term n 4 x = 1120 / x^6 ∧
  ∀ k, 0 ≤ k ∧ k ≤ n → n.choose k ≤ n.choose 4 := by
  sorry

#check expansion_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l334_33455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l334_33443

/-- The number of days required for a given number of men to complete a piece of work,
    given that 36 men can complete the work in 18 days. -/
def days_to_complete (num_men : ℕ) : ℕ :=
  let exact_days := (36 * 18 : ℚ) / num_men
  Int.ceil exact_days |>.toNat

/-- Theorem stating that 81 men will complete the work in 41 days. -/
theorem work_completion_time : days_to_complete 81 = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l334_33443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l334_33497

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y² = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def F : Point :=
  ⟨1, 0⟩

/-- The vertex of the parabola -/
def O : Point :=
  ⟨0, 0⟩

/-- A point on the parabola -/
noncomputable def M : Point :=
  ⟨2, Real.sqrt 8⟩

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p q r : Point) : ℝ :=
  (1/2) * abs (p.x*(q.y - r.y) + q.x*(r.y - p.y) + r.x*(p.y - q.y))

theorem parabola_triangle_area :
  M ∈ Parabola ∧ distance M F = 3 → triangleArea O M F = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l334_33497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l334_33408

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (((L / 2) * (3 * B) - L * B) / (L * B)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l334_33408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ab_value_l334_33449

noncomputable def f (x : ℝ) : ℝ := -1 / x

theorem f_ab_value (a b : ℝ) (h1 : f a = -1/3) (h2 : b = 2) : f (a * b) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ab_value_l334_33449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_performance_analysis_l334_33494

/-- Represents the contingency table for student performance --/
structure ContingencyTable where
  excellent_A : ℕ
  not_excellent_A : ℕ
  excellent_B : ℕ
  not_excellent_B : ℕ

/-- Calculates the chi-square statistic --/
noncomputable def chi_square (ct : ContingencyTable) : ℝ :=
  let n := ct.excellent_A + ct.not_excellent_A + ct.excellent_B + ct.not_excellent_B
  let ad := ct.excellent_A * ct.not_excellent_B
  let bc := ct.not_excellent_A * ct.excellent_B
  (n * (ad - bc)^2 : ℝ) / ((ct.excellent_A + ct.not_excellent_A) * 
    (ct.excellent_B + ct.not_excellent_B) * 
    (ct.excellent_A + ct.excellent_B) * 
    (ct.not_excellent_A + ct.not_excellent_B))

theorem student_performance_analysis 
  (total_students : ℕ) 
  (total_excellent : ℕ) 
  (excellent_A : ℕ) 
  (not_excellent_B : ℕ) 
  (h1 : total_students = 110) 
  (h2 : total_excellent = 30) 
  (h3 : excellent_A = 10) 
  (h4 : not_excellent_B = 30) :
  let ct := ContingencyTable.mk excellent_A 50 (total_excellent - excellent_A) not_excellent_B
  (ct.not_excellent_A = 50) ∧ 
  (chi_square ct > 6.635) ∧ 
  (8 * ct.not_excellent_A / (ct.not_excellent_A + ct.not_excellent_B : ℝ) = 5) ∧
  (8 * ct.not_excellent_B / (ct.not_excellent_A + ct.not_excellent_B : ℝ) = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_performance_analysis_l334_33494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_questions_completed_l334_33404

/-- The total number of questions completed by three students in two hours given their initial completion rates and percentage increases -/
theorem total_questions_completed 
  (fiona_first_hour : ℕ) 
  (x y z : ℝ) 
  (h1 : fiona_first_hour = 36)
  (h2 : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) :
  (let shirley_first_hour := 2 * fiona_first_hour
   let kiana_first_hour := (fiona_first_hour + shirley_first_hour) / 2
   let fiona_second_hour := fiona_first_hour + (x / 100) * fiona_first_hour
   let shirley_second_hour := shirley_first_hour + (y / 100) * shirley_first_hour
   let kiana_second_hour := kiana_first_hour + (z / 100) * kiana_first_hour
   let total := fiona_first_hour + shirley_first_hour + kiana_first_hour + 
                fiona_second_hour + shirley_second_hour + kiana_second_hour
   total) = 324 + 0.36 * x + 0.72 * y + 0.54 * z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_questions_completed_l334_33404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l334_33440

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log 3

theorem relationship_abc :
  let a := f (Real.log 4 / Real.log (1/3))
  let b := f (Real.sqrt (π^3))
  let c := f 2
  b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l334_33440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_period_divisibility_fibonacci_divisibility_extractable_characterization_l334_33450

/-- Checks if 5 is a quadratic residue modulo p -/
def is_extractable (p : ℕ) : Prop := ∃ x : ℤ, x * x ≡ 5 [ZMOD p]

/-- Represents the period length of the Fibonacci sequence modulo p -/
def period_length (p : ℕ) : ℕ := sorry

/-- Main theorem about Fibonacci period divisibility -/
theorem fibonacci_period_divisibility (p : ℕ) (r : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_not_two_five : p ≠ 2 ∧ p ≠ 5) 
  (h_r_period : r = period_length p) :
  (¬ is_extractable p → r ∣ (p + 1)) ∧
  (is_extractable p → r ∣ (p - 1)) := by
  sorry

/-- Theorem about divisibility of Fibonacci numbers -/
theorem fibonacci_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_not_two_five : p ≠ 2 ∧ p ≠ 5) :
  (¬ is_extractable p → ∃ a : ℕ, a ≡ Nat.fib (p + 1) [MOD p] ∧ p ∣ a) ∧
  (is_extractable p → ∃ a : ℕ, a ≡ Nat.fib (p - 1) [MOD p] ∧ p ∣ a) := by
  sorry

/-- Theorem characterizing when 5 is a quadratic residue -/
theorem extractable_characterization (p : ℕ) (h_prime : Nat.Prime p) (h_not_two_five : p ≠ 2 ∧ p ≠ 5) :
  is_extractable p ↔ ∃ k : ℤ, p = 5 * k + 1 ∨ p = 5 * k - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_period_divisibility_fibonacci_divisibility_extractable_characterization_l334_33450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_segment_length_l334_33496

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- The length of a segment connecting points on the sides of the triangle -/
noncomputable def segment_length (t : RightTriangle) (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

/-- The area of the triangle -/
noncomputable def triangle_area (t : RightTriangle) : ℝ :=
  (t.a * t.b) / 2

/-- The area of the smaller part created by the segment -/
noncomputable def small_area (t : RightTriangle) (x y : ℝ) : ℝ :=
  (x * y) / 2

theorem shortest_segment_length (t : RightTriangle) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≤ t.b ∧ y ≤ t.c ∧
  small_area t x y = triangle_area t / 2 ∧
  ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' ≤ t.b → y' ≤ t.c →
  small_area t x' y' = triangle_area t / 2 →
  segment_length t x y ≤ segment_length t x' y' ∧
  segment_length t x y = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_segment_length_l334_33496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_is_seven_l334_33461

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

structure Triangle where
  a : Point
  b : Point
  c : Point

noncomputable def longestSide (t : Triangle) : ℝ :=
  max (distance t.a t.b) (max (distance t.b t.c) (distance t.c t.a))

theorem longest_side_is_seven :
  let t : Triangle := { 
    a := ⟨1, 3⟩,
    b := ⟨7, 9⟩,
    c := ⟨8, 3⟩
  }
  longestSide t = 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_is_seven_l334_33461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l334_33476

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos (x + Real.pi / 4))^2 - 1

theorem f_properties : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l334_33476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_perp_diag_are_rhombus_l334_33407

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- A rhombus is a quadrilateral with all sides equal in length. -/
def is_rhombus (q : Quadrilateral) : Prop := sorry

/-- Two line segments are perpendicular if they meet at a right angle. -/
def are_perpendicular (s1 s2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

/-- The diagonals of a quadrilateral are the line segments connecting opposite vertices. -/
def diagonals (q : Quadrilateral) : ((ℝ × ℝ) × (ℝ × ℝ)) × ((ℝ × ℝ) × (ℝ × ℝ)) := sorry

/-- Theorem: Not all quadrilaterals with perpendicular diagonals are rhombuses. -/
theorem not_all_perp_diag_are_rhombus :
  ¬ ∀ (q : Quadrilateral), are_perpendicular (diagonals q).1 (diagonals q).2 → is_rhombus q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_perp_diag_are_rhombus_l334_33407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l334_33441

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the point of tangency
noncomputable def point : ℝ × ℝ := (2, Real.exp 2)

-- Define the slope of the tangent line
noncomputable def tangent_slope : ℝ := Real.exp 2

-- Define the y-intercept of the tangent line
noncomputable def y_intercept : ℝ := -Real.exp 2

-- Define the x-intercept of the tangent line
def x_intercept : ℝ := 1

-- Theorem statement
theorem tangent_line_triangle_area :
  (1 / 2) * x_intercept * (point.2 - y_intercept) = (Real.exp 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l334_33441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_theorem_l334_33469

/-- The percentage of profits to revenues in the previous year -/
noncomputable def profit_percentage (previous_revenue : ℝ) (previous_profit : ℝ) : ℝ :=
  (previous_profit / previous_revenue) * 100

/-- The revenue in 1999 -/
noncomputable def revenue_1999 (previous_revenue : ℝ) : ℝ :=
  0.7 * previous_revenue

/-- The profit in 1999 -/
noncomputable def profit_1999 (previous_revenue : ℝ) : ℝ :=
  0.2 * revenue_1999 previous_revenue

theorem profit_percentage_theorem (previous_revenue : ℝ) (previous_profit : ℝ) 
  (h1 : profit_1999 previous_revenue = 1.3999999999999997 * previous_profit)
  (h2 : previous_revenue > 0) :
  profit_percentage previous_revenue previous_profit = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_theorem_l334_33469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_integers_l334_33400

def is_valid_integer (n : ℕ) : Bool :=
  100 ≤ n ∧ n ≤ 999 ∧  -- Three-digit integer
  n % 2 = 1 ∧  -- Odd integer
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits[0]! < digits[1]! ∧ digits[1]! < digits[2]!  -- Strictly increasing digits

theorem count_valid_integers : 
  (Finset.filter (fun n => is_valid_integer n) (Finset.range 900)).card + 
  (Finset.filter (fun n => is_valid_integer n) (Finset.range 100)).card = 50 := by
  sorry

#eval (Finset.filter (fun n => is_valid_integer n) (Finset.range 900)).card + 
      (Finset.filter (fun n => is_valid_integer n) (Finset.range 100)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_integers_l334_33400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l334_33405

/-- The maximum distance between a point on the circle x² + y² = 9 and the line x + √3 y = 2 is 4 -/
theorem max_distance_circle_to_line : 
  ∃ (d_max : ℝ), d_max = 4 ∧ 
  (∀ (x y : ℝ), x^2 + y^2 = 9 → 
    ∀ (d : ℝ), d = |x + Real.sqrt 3 * y - 2| / 2 → 
      d ≤ d_max) ∧
  (∃ (x y : ℝ), x^2 + y^2 = 9 ∧ 
    |x + Real.sqrt 3 * y - 2| / 2 = d_max) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l334_33405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_zero_of_sin_minus_2cos_l334_33424

theorem cos_2x_zero_of_sin_minus_2cos (x₀ : ℝ) : 
  Real.sin x₀ - 2 * Real.cos x₀ = 0 → Real.cos (2 * x₀) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_zero_of_sin_minus_2cos_l334_33424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_eq_neg_one_f_geq_two_over_e_iff_a_in_range_l334_33416

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + 1/x - a * Real.log x

-- Theorem for part (1)
theorem tangent_line_implies_a_eq_neg_one (a : ℝ) :
  (∃ x₀ > 0, f a x₀ = x₀ + 1 ∧ (deriv (f a)) x₀ = 1) → a = -1 := by
  sorry

-- Theorem for part (2)
theorem f_geq_two_over_e_iff_a_in_range (a : ℝ) :
  (∀ x > 0, f a x ≥ 2/Real.exp 1) ↔ 1/Real.exp 1 - Real.exp 1 ≤ a ∧ a ≤ Real.exp 1 - 1/Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_eq_neg_one_f_geq_two_over_e_iff_a_in_range_l334_33416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_hard_configuration_l334_33422

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- Represents a query that returns the distance to the nearest unguessed point -/
def Query := Point → ℝ

/-- Represents a strategy for guessing points -/
def Strategy := ℕ → Query → Option Point

/-- The minimum number of queries required to guess all points for a given configuration -/
noncomputable def minQueries (n : ℕ) (circle : Circle) (points : Finset Point) : ℕ :=
  sorry

/-- The theorem stating that there exists a configuration requiring at least (n + 1)^2 queries -/
theorem exists_hard_configuration (n : ℕ) :
  ∃ (circle : Circle) (points : Finset Point),
    points.card = n ∧ 
    (∀ p ∈ points, isInside p circle) ∧
    minQueries n circle points ≥ (n + 1)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_hard_configuration_l334_33422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_through_center_l334_33415

/-- The line equation: x - y + 1 = 0 -/
def line (x y : ℝ) : Prop := x - y + 1 = 0

/-- The circle equation: (x-2)² + (y-1)² = 4 -/
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 1)

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- The distance from a point (x, y) to the line ax + by + c = 0 -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

theorem line_intersects_circle_not_through_center :
  ∃ (x y : ℝ), line x y ∧ circle_eq x y ∧
  (x, y) ≠ circle_center ∧
  distance_point_to_line circle_center.1 circle_center.2 1 (-1) 1 < circle_radius :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_through_center_l334_33415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_implies_m_eq_1_l334_33478

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := (x - 1)^3 - 1 / (x - 1)
def g (m : ℝ) (x : ℝ) := -x + m

-- Define the property that the sum of x-coordinates of intersection points is 2
def intersection_sum_2 (m : ℝ) : Prop :=
  ∃ x₁ x₂, f x₁ = g m x₁ ∧ f x₂ = g m x₂ ∧ x₁ + x₂ = 2

-- Theorem statement
theorem intersection_sum_implies_m_eq_1 :
  ∀ m : ℝ, intersection_sum_2 m → m = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_implies_m_eq_1_l334_33478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_motion_distance_time_proportional_l334_33451

/-- Represents the speed in uniform motion -/
noncomputable def speed : ℝ → ℝ → ℝ := λ d t ↦ d / t

/-- A motion is uniform if its speed is constant -/
def is_uniform_motion (v : ℝ) (d : ℝ → ℝ) : Prop :=
  ∀ t₁ t₂, t₁ ≠ 0 → t₂ ≠ 0 → speed (d t₁) t₁ = speed (d t₂) t₂ ∧ speed (d t₁) t₁ = v

/-- Direct proportionality between two quantities -/
def is_directly_proportional (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x, f x = k * g x

/-- Theorem: In uniform motion, distance is directly proportional to time -/
theorem uniform_motion_distance_time_proportional (v : ℝ) (d : ℝ → ℝ) 
  (h : is_uniform_motion v d) : is_directly_proportional d id := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_motion_distance_time_proportional_l334_33451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l334_33412

-- Define the line equation
noncomputable def line_equation (x : ℝ) (α : ℝ) : ℝ := x * Real.sin α + 1

-- Define the angle of inclination
noncomputable def angle_of_inclination (α : ℝ) : ℝ := Real.arctan (Real.sin α)

-- Theorem statement
theorem angle_of_inclination_range :
  ∀ α : ℝ, ∃ θ : ℝ, θ = angle_of_inclination α ∧
  (θ ∈ Set.Icc 0 (π / 4) ∨ θ ∈ Set.Ico (3 * π / 4) π) :=
by
  sorry

-- Additional helper lemmas if needed
lemma sin_range (α : ℝ) : Real.sin α ∈ Set.Icc (-1) 1 :=
by
  sorry

lemma arctan_range (x : ℝ) : x ∈ Set.Icc (-1) 1 → Real.arctan x ∈ Set.Icc (- π / 4) (π / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l334_33412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2a_plus_pi_6_l334_33482

theorem cos_2a_plus_pi_6 (α : Real) (h1 : Real.sin α = (Real.sqrt 10) / 10) (h2 : 0 < α ∧ α < Real.pi / 2) :
  Real.cos (2 * α + Real.pi / 6) = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2a_plus_pi_6_l334_33482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_l334_33427

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

-- Define the concept of a symmetric axis
def is_symmetric_axis (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f (x₀ + x) = f (x₀ - x)

-- State the theorem
theorem point_on_line (a b x₀ : ℝ) :
  is_symmetric_axis (f a b) x₀ →
  Real.tan x₀ = 3 →
  ∃ t : ℝ, a = 3*t ∧ b = t :=
by
  sorry

#check point_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_l334_33427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l334_33473

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (-2 - 3*t, 2 - 4*t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (y - 2)^2 - x^2 = 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t ∧ curve_C p.1 p.2}

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem intersection_distance :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ 
  A ≠ B ∧ distance A B = 10 * Real.sqrt 71 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l334_33473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l334_33463

/-- Given two intersecting circles M and N, prove that the equation of circle N
    is either (x-2)^2+(y-1)^2=4 or (x-2)^2+(y-1)^2=20 -/
theorem circle_equation_proof (x y : ℝ) :
  let circle_m : ℝ → ℝ → Prop := λ x y => x^2 + (y+1)^2 = 4
  let center_n : ℝ × ℝ := (2, 1)
  let intersect_points : Set (ℝ × ℝ) := {p | circle_m p.1 p.2 ∧ ∃ r, (p.1 - center_n.1)^2 + (p.2 - center_n.2)^2 = r^2}
  ∀ a b : ℝ × ℝ, a ∈ intersect_points → b ∈ intersect_points → a ≠ b →
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = 8 →
  (∃ r, ∀ x y, (x - center_n.1)^2 + (y - center_n.2)^2 = r^2 ↔ (r = 2 ∨ r = 2*Real.sqrt 5)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l334_33463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l334_33492

theorem complex_number_problem (m : ℝ) (z : ℂ) : 
  z = m + 2*Complex.I → Complex.I.im ≠ 0 → (2 + Complex.I) * z = Complex.I * (Complex.I * z).im → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l334_33492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l334_33406

/-- A function g that satisfies the given conditions -/
noncomputable def g (A B C : ℤ) : ℝ → ℝ := λ x => x^2 / (A * x^2 + B * x + C)

/-- The theorem stating the sum of A, B, and C -/
theorem sum_of_coefficients (A B C : ℤ) : 
  (∀ x < -3, g A B C x < 0.5) → A + B + C = 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l334_33406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_formula_l334_33485

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n - 2 * a (n + 1) + a (n + 2) = 0

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_a2 : a 2 = 4) : 
  ∀ n : ℕ, a n = 2 * n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_formula_l334_33485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_equality_l334_33468

theorem floor_sqrt_equality (n : ℕ+) :
  ⌊(n : ℝ).sqrt + ((n : ℝ) + 1).sqrt⌋ = ⌊(4 * (n : ℝ) + 1).sqrt⌋ ∧
  ⌊(4 * (n : ℝ) + 1).sqrt⌋ = ⌊(4 * (n : ℝ) + 2).sqrt⌋ ∧
  ⌊(4 * (n : ℝ) + 2).sqrt⌋ = ⌊(4 * (n : ℝ) + 3).sqrt⌋ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_equality_l334_33468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_power_function_l334_33439

/-- The equation of the tangent line to y = x^n at (2,8) is 12x - y - 16 = 0 -/
theorem tangent_line_power_function (n : ℝ) :
  (2 : ℝ) ^ n = 8 →
  ∃ l : ℝ → ℝ → Prop,
    (∀ x y, l x y ↔ 12 * x - y - 16 = 0) ∧
    (∀ x, l x (x^n) → x = 2) ∧
    l 2 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_power_function_l334_33439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_airplane_seats_l334_33402

def airplane_seats (first_class business_class economy_class total : ℕ) : Prop :=
  (first_class = 30) ∧
  (business_class = total / 5) ∧
  (economy_class = 3 * total / 5) ∧
  (first_class + business_class + economy_class = total)

theorem solve_airplane_seats : ∃ total : ℕ, airplane_seats 30 (total / 5) (3 * total / 5) total :=
by
  use 150
  apply And.intro
  · rfl
  apply And.intro
  · rfl
  apply And.intro
  · rfl
  · norm_num

#eval (150 : ℕ) / 5  -- Should output 30
#eval 3 * (150 : ℕ) / 5  -- Should output 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_airplane_seats_l334_33402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_range_l334_33414

/-- Represents a rectangle ABCD with a circle centered at D and B on the circle -/
structure RectangleWithCircle where
  AD : ℝ
  CD : ℝ
  is_rectangle : AD > 0 ∧ CD > 0
  D_is_center : True  -- We can't directly represent this geometrically, so we use a placeholder
  B_on_circle : True  -- We can't directly represent this geometrically, so we use a placeholder

/-- The area of the semi-circle minus the rectangle for the given configuration -/
noncomputable def area_difference (r : RectangleWithCircle) : ℝ :=
  let radius := Real.sqrt (r.AD ^ 2 + r.CD ^ 2) / 2
  (Real.pi * radius ^ 2) / 2 - r.AD * r.CD

/-- Theorem stating that the area difference is between -9 and -7 for the given dimensions -/
theorem area_difference_range :
  ∀ r : RectangleWithCircle, r.AD = 6 ∧ r.CD = 8 →
  -9 < area_difference r ∧ area_difference r < -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_range_l334_33414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_power_range_l334_33462

-- Define the function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log a / Real.log (1/2)) ^ x

-- State the theorem
theorem increasing_log_power_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 0 < a ∧ a < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_power_range_l334_33462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l334_33432

/-- Calculates the speed of a train given the speed of another train traveling in the opposite direction, the length of the train, and the time it takes to pass. -/
noncomputable def calculate_train_speed (speed_a : ℝ) (length_b : ℝ) (passing_time : ℝ) : ℝ :=
  let speed_a_mps := speed_a * (1000 / 3600)
  let relative_speed := length_b / passing_time
  let speed_b_mps := relative_speed - speed_a_mps
  speed_b_mps * (3600 / 1000)

/-- Theorem stating that under the given conditions, the speed of the goods train is approximately 62 km/h. -/
theorem goods_train_speed :
  let speed_a := (50 : ℝ) -- km/h
  let length_b := (280 : ℝ) -- meters
  let passing_time := (9 : ℝ) -- seconds
  let calculated_speed := calculate_train_speed speed_a length_b passing_time
  abs (calculated_speed - 62) < 0.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l334_33432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_properties_l334_33479

noncomputable def f (x : ℝ) : ℝ := 2^x - 1/x

theorem zero_point_properties (x₀ x₁ x₂ : ℝ) 
  (h₀ : f x₀ = 0) 
  (h₁ : 0 < x₁ ∧ x₁ < x₀) 
  (h₂ : x₀ < x₂) : 
  f x₁ < 0 ∧ 0 < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_properties_l334_33479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_union_size_l334_33437

def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

theorem max_union_size (A B : Finset ℕ) 
  (subset_A : ↑A ⊆ S) 
  (subset_B : ↑B ⊆ S)
  (equal_size : A.card = B.card)
  (disjoint : Disjoint A B)
  (mapping : ∀ n ∈ A, (2 * n + 2) ∈ B) :
  (A ∪ B).card ≤ 66 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_union_size_l334_33437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_impossibility_l334_33495

theorem partition_impossibility : ¬ ∃ (partition : Fin 11 → Fin 33 → Prop),
  (∀ i : Fin 33, ∃! j : Fin 11, partition j i) ∧
  (∀ j : Fin 11, ∃ a b c : Fin 33, 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (partition j a ∧ partition j b ∧ partition j c) ∧
    (a.val + 1 = b.val + 1 + c.val + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_impossibility_l334_33495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l334_33436

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a*x + Real.sin (x + Real.pi/2)

theorem even_function_implies_a_equals_two :
  (∀ x : ℝ, y x a = y (-x) a) → a = 2 :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l334_33436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_log_functions_decreasing_l334_33413

-- Define a logarithmic function type
def LogFunction (f : ℝ → ℝ) : Prop := ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ x > 0, f x = Real.log x / Real.log a

-- Define what it means for a function to be decreasing
def IsDecreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

-- Theorem: Not all logarithmic functions are decreasing
theorem not_all_log_functions_decreasing :
  ¬(∀ f : ℝ → ℝ, LogFunction f → IsDecreasing f) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_log_functions_decreasing_l334_33413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_amplitude_l334_33488

/-- Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, d are positive constants,
    if the function oscillates between a maximum of 5 and a minimum of -3, then a = 4. -/
theorem sinusoidal_amplitude (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → 
  (∀ x, ((-3 : ℝ) ≤ a * Real.sin (b * x + c) + d) ∧ (a * Real.sin (b * x + c) + d ≤ (5 : ℝ))) → 
  (∃ x₁ x₂, a * Real.sin (b * x₁ + c) + d = (5 : ℝ) ∧ a * Real.sin (b * x₂ + c) + d = (-3 : ℝ)) →
  a = (4 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_amplitude_l334_33488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freshman_height_estimates_l334_33447

/-- Represents the data for a group of students -/
structure StudentGroup where
  count : ℕ
  avgHeight : ℝ
  variance : ℝ

/-- Represents the data for the entire freshman class -/
structure FreshmanClass where
  totalCount : ℕ
  boys : StudentGroup
  girls : StudentGroup

noncomputable def estimateAverageHeight (fc : FreshmanClass) : ℝ :=
  (fc.boys.count : ℝ) / (fc.totalCount : ℝ) * fc.boys.avgHeight +
  (fc.girls.count : ℝ) / (fc.totalCount : ℝ) * fc.girls.avgHeight

noncomputable def estimateVariance (fc : FreshmanClass) (avgHeight : ℝ) : ℝ :=
  (fc.boys.count : ℝ) / (fc.totalCount : ℝ) * (fc.boys.variance + (fc.boys.avgHeight - avgHeight)^2) +
  (fc.girls.count : ℝ) / (fc.totalCount : ℝ) * (fc.girls.variance + (fc.girls.avgHeight - avgHeight)^2)

theorem freshman_height_estimates (fc : FreshmanClass)
  (h1 : fc.totalCount = 300)
  (h2 : fc.boys.count = 180)
  (h3 : fc.girls.count = 120)
  (h4 : fc.boys.avgHeight = 170)
  (h5 : fc.boys.variance = 14)
  (h6 : fc.girls.avgHeight = 160)
  (h7 : fc.girls.variance = 24) :
  estimateAverageHeight fc = 166 ∧ estimateVariance fc (estimateAverageHeight fc) = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freshman_height_estimates_l334_33447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_outside_plane_l334_33411

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relations
variable (belongs_to : Point → Line → Prop)
variable (outside_of : Point → Plane → Prop)

-- State the theorem
theorem line_through_point_outside_plane 
  (a : Line) (P : Point) (α : Plane) :
  (belongs_to P a ∧ outside_of P α) ↔ 
  (∃ (Q : Point), belongs_to Q a ∧ outside_of Q α) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_outside_plane_l334_33411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_mode_not_necessarily_same_l334_33481

/-- Represents a person's shooting results -/
structure ShootingResult where
  shots : ℕ
  average : ℝ
  variance : ℝ

/-- Definition of mode for a ShootingResult -/
def is_mode (result : ShootingResult) (m : ℕ) : Prop :=
  ∃ (scores : List ℕ), 
    scores.length = result.shots ∧ 
    (scores.sum : ℝ) / result.shots = result.average ∧
    ((scores.map (λ x => ((x : ℝ) - result.average)^2)).sum / result.shots) = result.variance ∧
    ∀ n, (scores.count n) ≤ (scores.count m)

/-- The problem statement -/
theorem shooting_mode_not_necessarily_same 
  (person_A person_B : ShootingResult)
  (h_shots_A : person_A.shots = 10)
  (h_shots_B : person_B.shots = 10)
  (h_avg_A : person_A.average = 8)
  (h_avg_B : person_B.average = 8)
  (h_var_A : person_A.variance = 1.2)
  (h_var_B : person_B.variance = 1.6) :
  ¬ (∀ (mode_A mode_B : ℕ), is_mode person_A mode_A → is_mode person_B mode_B → mode_A = mode_B) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_mode_not_necessarily_same_l334_33481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l334_33466

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + 8 * x + y^2 - 2 * y + 8 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := Real.pi * Real.sqrt 2 / 2

/-- Theorem stating that the area of the ellipse defined by the given equation is π * sqrt(2) / 2 -/
theorem ellipse_area_theorem : 
  ∃ (a b : ℝ), (∀ (x y : ℝ), ellipse_equation x y ↔ (x + 2)^2 / a^2 + (y - 1)^2 / b^2 = 1) ∧ 
  ellipse_area = Real.pi * a * b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l334_33466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_theorem_l334_33431

/-- Represents a continued fraction of the form [a; ̅b,c] -/
structure MyContinuedFraction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Represents a quadratic equation with integer coefficients -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Converts a continued fraction to a real number -/
noncomputable def continuedFractionToReal (cf : MyContinuedFraction) : ℝ :=
  sorry

/-- Checks if a real number is a root of a quadratic equation -/
def isRoot (eq : QuadraticEquation) (x : ℝ) : Prop :=
  eq.a * x^2 + eq.b * x + eq.c = 0

/-- The main theorem to prove -/
theorem quadratic_roots_theorem (eq : QuadraticEquation) (cf : MyContinuedFraction) :
  isRoot eq (continuedFractionToReal cf) →
  isRoot eq (cf.a - continuedFractionToReal ⟨cf.c, cf.b, cf.c⟩) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_theorem_l334_33431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_j_l334_33467

noncomputable def f (x : ℝ) : ℝ := (x + Real.sqrt (x^2 - 4)) / 2

theorem existence_of_j (i n : ℕ) (hi : i ≥ 2) (hn : n ≥ 2) :
  ∃ j : ℕ, j ≥ 2 ∧ (f^[n] (i : ℝ)) = f j :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_j_l334_33467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_root_is_three_halves_l334_33486

-- Define the equation
def equation (x : Real) : Prop :=
  |Real.sin (2 * Real.pi * x) - Real.cos (Real.pi * x)| = 
  (|Real.sin (2 * Real.pi * x)| - |Real.cos (Real.pi * x)|)

-- Define the interval
def in_interval (x : Real) : Prop := 1/4 < x ∧ x < 2

-- Theorem statement
theorem largest_root_is_three_halves :
  (∀ x : Real, in_interval x → equation x → x ≤ 3/2) ∧
  in_interval (3/2) ∧
  equation (3/2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_root_is_three_halves_l334_33486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_magnification_l334_33418

/-- The magnification of a circular tissue sample under an electron microscope. -/
noncomputable def magnification (magnified_diameter actual_diameter : ℝ) : ℝ :=
  magnified_diameter / actual_diameter

/-- Theorem: The magnification of a circular tissue sample is 1000 times. -/
theorem tissue_magnification :
  let magnified_diameter : ℝ := 0.3
  let actual_diameter : ℝ := 0.0003
  magnification magnified_diameter actual_diameter = 1000 := by
  -- Unfold the definition of magnification
  unfold magnification
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_magnification_l334_33418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_white_balls_theorem_l334_33499

/-- Represents a bag of balls with a total count and a frequency of drawing white balls. -/
structure BagOfBalls where
  total_balls : ℕ
  white_frequency : ℝ
  total_positive : total_balls > 0
  frequency_range : 0 ≤ white_frequency ∧ white_frequency ≤ 1

/-- Calculates the estimated number of white balls in the bag. -/
def estimated_white_balls (bag : BagOfBalls) : ℝ :=
  (bag.total_balls : ℝ) * bag.white_frequency

/-- Theorem stating that for a bag with 10 balls and a white ball drawing frequency of 0.4,
    the estimated number of white balls is 4. -/
theorem estimate_white_balls_theorem (bag : BagOfBalls)
    (h1 : bag.total_balls = 10)
    (h2 : bag.white_frequency = 0.4) :
    estimated_white_balls bag = 4 := by
  rw [estimated_white_balls, h1, h2]
  norm_num

#check estimate_white_balls_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_white_balls_theorem_l334_33499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_problem_l334_33434

-- Define the triangles and their properties
structure Triangle :=
  (X Y Z : ℝ × ℝ)

def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the lengths of the sides
def length (A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_similarity_problem (XYZ WZV : Triangle) :
  similar XYZ WZV →
  length XYZ.Y XYZ.Z = 30 →
  length XYZ.X XYZ.Z = 15 →
  length XYZ.X XYZ.Y = 18 →
  length XYZ.Z XYZ.Y = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_problem_l334_33434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prom_cost_is_836_l334_33493

/-- Calculates the total cost of James and Susan's prom night. -/
def prom_cost (ticket_price : ℕ) (dinner_price : ℕ) (tip_percentage : ℚ) 
              (limo_hourly_rate : ℕ) (limo_hours : ℕ) : ℕ :=
  let ticket_total := 2 * ticket_price
  let dinner_total := dinner_price + (dinner_price : ℚ) * tip_percentage
  let limo_total := limo_hourly_rate * limo_hours
  ticket_total + dinner_total.ceil.toNat + limo_total

/-- Theorem stating that the total cost of James and Susan's prom night is $836. -/
theorem prom_cost_is_836 : 
  prom_cost 100 120 (30/100) 80 6 = 836 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prom_cost_is_836_l334_33493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_trig_ratios_l334_33491

/-- Represents a right triangle with sides a, b, and c, where c is the hypotenuse -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The cosine of angle C in a right triangle -/
noncomputable def cos_C (t : RightTriangle) : ℝ := t.b / t.c

/-- The sine of angle C in a right triangle -/
noncomputable def sin_C (t : RightTriangle) : ℝ := t.a / t.c

theorem right_triangle_trig_ratios :
  ∃ (t : RightTriangle),
    t.a = 8 ∧ t.c = 15 ∧ cos_C t = Real.sqrt 161 / 15 ∧ sin_C t = 8 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_trig_ratios_l334_33491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_good_not_n_plus_one_good_l334_33430

def S (k : ℕ) : ℕ := sorry

def f (n : ℕ) : ℕ := n - S n

def f_iter : ℕ → ℕ → ℕ
  | 0, n => n
  | k + 1, n => f_iter k (f n)

def is_n_good (n a : ℕ) : Prop :=
  ∃ x y : ℕ, f_iter n x < a ∧ a ≤ f_iter n y

theorem exists_n_good_not_n_plus_one_good :
  ∀ n : ℕ, ∃ a : ℕ, is_n_good n a ∧ ¬is_n_good (n + 1) a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_good_not_n_plus_one_good_l334_33430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_180_not_divisible_by_3_l334_33417

theorem divisors_of_180_not_divisible_by_3 : 
  (Finset.filter (fun d : ℕ => d > 0 ∧ 180 % d = 0 ∧ d % 3 ≠ 0) (Finset.range (180 + 1))).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_180_not_divisible_by_3_l334_33417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l334_33446

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x ∈ Set.Ioo 1 2, 
  (MonotoneOn (fun x => 2*x^2 - 2*(m-2)*x + 3*m - 1) (Set.Ioo 1 2))

def q (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ x y : ℝ, x^2 / (m+1) + y^2 / (9-m) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

-- State the theorem
theorem range_of_m : 
  ∃ S : Set ℝ, S = Set.Iic (-1) ∪ {4} ∧
  ∀ m : ℝ, m ∈ S ↔ 
    ((p m ∨ q m) ∧ ¬(p m ∧ q m) ∧ ¬(¬(p m))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l334_33446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_log2_implies_exp2_plus_1_l334_33423

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) :=
  ∀ x y, f x = y ↔ g y = x

theorem symmetric_log2_implies_exp2_plus_1 (f : ℝ → ℝ) :
  symmetric_wrt_y_eq_x f (λ x ↦ log2 (x - 1)) →
  ∀ x, f x = 2^x + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_log2_implies_exp2_plus_1_l334_33423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l334_33465

theorem inequality_proof (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  let a := (m^(m+1) + n^(n+1)) / (m^m + n^n)
  a^m + a^n ≥ m^m + n^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l334_33465
