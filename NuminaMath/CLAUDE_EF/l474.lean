import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_ratio_l474_47478

theorem square_side_ratio (area_ratio : ℚ) : 
  area_ratio = 192 / 80 →
  ∃ (a b c : ℕ), 
    (a * Real.sqrt b : ℝ) / c = Real.sqrt (area_ratio) ∧
    a = 2 ∧ b = 15 ∧ c = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_ratio_l474_47478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_other_asymptote_l474_47455

/-- Given a hyperbola with one asymptote y = 2x and foci with x-coordinate -4,
    prove that the equation of the other asymptote is y = -1/2x - 10 -/
theorem hyperbola_other_asymptote 
  (h : Set (ℝ × ℝ)) 
  (asymptote1 : Set.range (fun x => (x, 2 * x)) ⊆ h)
  (foci_x : ∀ f ∈ h, f.1 = -4) :
  Set.range (fun x => (x, -1/2 * x - 10)) ⊆ h :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_other_asymptote_l474_47455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_ironing_pants_correct_l474_47409

/-- Calculates the time spent ironing pants given ironing rates and total pieces ironed -/
def time_ironing_pants 
  (shirt_rate : ℕ) -- shirts ironed per hour
  (pants_rate : ℕ) -- pants ironed per hour
  (shirt_time : ℕ) -- hours spent ironing shirts
  (total_pieces : ℕ) -- total pieces of clothing ironed
  : ℕ :=
  let shirts_ironed := shirt_rate * shirt_time
  let pants_ironed := total_pieces - shirts_ironed
  pants_ironed / pants_rate

theorem time_ironing_pants_correct
  (shirt_rate : ℕ) 
  (pants_rate : ℕ) 
  (shirt_time : ℕ) 
  (total_pieces : ℕ)
  (h1 : shirt_rate = 4)
  (h2 : pants_rate = 3)
  (h3 : shirt_time = 3)
  (h4 : total_pieces = 27)
  : time_ironing_pants shirt_rate pants_rate shirt_time total_pieces = 5 := by
  sorry

#eval time_ironing_pants 4 3 3 27

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_ironing_pants_correct_l474_47409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_range_l474_47469

-- Define the curves
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log x

-- Define the condition for common tangent
def has_common_tangent (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
    ((-a / x₁^2) * x₁ + 2*a/x₁ = (2/x₂) * x₁ + 2*Real.log x₂ - 2) ∧
    (-a / x₁^2 = 2/x₂)

-- State the theorem
theorem common_tangent_implies_a_range (a : ℝ) :
  has_common_tangent a → -2/Real.exp 1 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_range_l474_47469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stockholm_uppsala_distance_l474_47480

/-- Represents the scale of a map --/
structure MapScale where
  map_distance : ℚ
  actual_distance : ℚ

/-- Calculates the actual distance given a map distance and a map scale --/
def calculate_actual_distance (map_distance : ℚ) (scale : MapScale) : ℚ :=
  (map_distance * scale.actual_distance) / scale.map_distance

/-- Theorem: The actual distance between Stockholm and Uppsala is 240 km --/
theorem stockholm_uppsala_distance :
  let map_distance : ℚ := 3
  let scale : MapScale := { map_distance := 1, actual_distance := 80 }
  calculate_actual_distance map_distance scale = 240 := by
  -- Unfold the definitions
  unfold calculate_actual_distance
  -- Simplify the arithmetic
  simp [MapScale.map_distance, MapScale.actual_distance]
  -- The proof is complete
  rfl

#eval calculate_actual_distance 3 { map_distance := 1, actual_distance := 80 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stockholm_uppsala_distance_l474_47480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_literally_1434_l474_47448

def is_literally_1434 (n : ℕ) : Bool :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000) % 5 = 1 ∧
  ((n / 100) % 10) % 5 = 4 ∧
  ((n / 10) % 10) % 5 = 3 ∧
  (n % 10) % 5 = 4

theorem sum_of_literally_1434 :
  (Finset.filter (fun n => is_literally_1434 n) (Finset.range 10000)).sum id = 67384 := by
  sorry

#eval (Finset.filter (fun n => is_literally_1434 n) (Finset.range 10000)).sum id

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_literally_1434_l474_47448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l474_47442

theorem tan_double_angle (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.cos x = 3/5) : 
  Real.tan (2 * x) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l474_47442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normalize_eq1_normalize_eq2_l474_47468

-- Define the normalization function
noncomputable def my_normalize (a b c : ℝ) : ℝ × ℝ × ℝ :=
  let m := 1 / Real.sqrt (a^2 + b^2)
  (m * a, m * b, m * c)

-- Theorem for the first equation
theorem normalize_eq1 :
  let (a, b, c) := my_normalize 2 (-3) (-10)
  (a, b, c) = (2 / Real.sqrt 13, -3 / Real.sqrt 13, -10 / Real.sqrt 13) :=
by sorry

-- Theorem for the second equation
theorem normalize_eq2 :
  let (a, b, c) := my_normalize 3 4 0
  (a, b, c) = (3 / 5, 4 / 5, 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normalize_eq1_normalize_eq2_l474_47468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_perpendicular_asymptote_l474_47407

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The slope of the asymptotes of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ :=
  h.b / h.a

theorem hyperbola_eccentricity_with_perpendicular_asymptote 
  (h : Hyperbola) 
  (perpendicular : asymptote_slope h * 2 = -1) : 
  eccentricity h = Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_perpendicular_asymptote_l474_47407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_locations_l474_47498

/-- The distance between two locations A and B, given a car's travel details --/
theorem distance_between_locations (speed time distance_to_midpoint : ℝ) : 
  speed = 42.5 →
  time = 1.5 →
  distance_to_midpoint = 26 →
  2 * (speed * time + distance_to_midpoint) = 179.5 := by
  intros h_speed h_time h_distance
  -- Proof steps would go here
  sorry

#check distance_between_locations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_locations_l474_47498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_sequence_contains_all_solutions_l474_47476

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Fractional part function -/
noncomputable def frac (x : ℝ) : ℝ :=
  x - Int.floor x

/-- The sequence of solutions -/
noncomputable def solution_sequence (n : ℕ) : ℝ :=
  let k := (n + 1) / 2
  if n % 2 = 1 then k else Real.sqrt (k^2 + 1/k)

/-- The main theorem -/
theorem solution_characterization (x : ℝ) (h : x ≥ 1) :
  Real.sqrt (floor x * floor (x^3)) + Real.sqrt (frac x * frac (x^3)) = x^2 ↔
  (∃ k : ℕ, x = k ∨ x = Real.sqrt (k^2 + 1/k)) := by
  sorry

/-- Proof that the sequence contains all solutions -/
theorem sequence_contains_all_solutions (x : ℝ) (h : x ≥ 1) :
  Real.sqrt (floor x * floor (x^3)) + Real.sqrt (frac x * frac (x^3)) = x^2 →
  ∃ n : ℕ, solution_sequence n = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_sequence_contains_all_solutions_l474_47476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l474_47428

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

-- Main theorem
theorem f_properties :
  -- 1. f is an odd function
  (∀ x, f (-x) = -f x) ∧
  -- 2. f is a decreasing function
  (∀ x y, x < y → f x > f y) ∧
  -- 3. If f(t^2 - 2t) < f(-2t^2 + k) for all t, then k < -1/3
  (∀ k, (∀ t, f (t^2 - 2*t) < f (-2*t^2 + k)) → k < -1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l474_47428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l474_47479

theorem symmetry_implies_phi_value (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi) :
  (∀ x : ℝ, Real.sin (2 * (Real.pi/6 + x) + φ) = Real.sin (2 * (Real.pi/6 - x) + φ)) →
  φ = Real.pi/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l474_47479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_transformation_l474_47454

theorem determinant_transformation (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 3 →
  Matrix.det !![x, 5*x + 2*y; z, 5*z + 2*w] = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_transformation_l474_47454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l474_47474

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) ^ 2 - Real.sin (2 * x) ^ 2

theorem f_is_even_and_periodic : 
  (∀ x, f x = f (-x)) ∧ (∀ x, f (x + π / 2) = f x) := by
  constructor
  · intro x
    -- Proof that f is even
    sorry
  · intro x
    -- Proof that f has period π/2
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l474_47474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_directions_l474_47400

-- Define the points and vector
def A : ℝ × ℝ := (1, 2)
def B (x : ℝ) : ℝ × ℝ := (3, x)
def a (x : ℝ) : ℝ × ℝ := (2 - x, -1)

-- Define the vector AB
def AB (x : ℝ) : ℝ × ℝ := ((B x).1 - A.1, (B x).2 - A.2)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

-- Define same direction
def same_direction (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ v = (k * w.1, k * w.2)

-- Define opposite direction
def opposite_direction (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ v = (k * w.1, k * w.2)

-- Main theorem
theorem vector_directions :
  ∀ x : ℝ, parallel (AB x) (a x) →
    (x = 2 - Real.sqrt 2 → same_direction (AB x) (a x)) ∧
    (x = 2 + Real.sqrt 2 → opposite_direction (AB x) (a x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_directions_l474_47400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_equivalence_l474_47471

theorem quadratic_radical_equivalence (a : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ k * Real.sqrt (3*a - 4) = Real.sqrt 2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_equivalence_l474_47471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_of_2011_integers_l474_47416

theorem unique_sum_of_2011_integers :
  ∃! (S : Finset ℕ), 
    S.card = 2011 ∧
    (∀ a b, a ∈ S → b ∈ S → a ≠ b → a < b) ∧
    (S.sum id = 2023066 ∨ S.sum id = 2013062) ∧
    (∀ T : Finset ℕ, T.card = 2011 → (∀ a b, a ∈ T → b ∈ T → a ≠ b → a < b) → T.sum id = S.sum id → T = S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_of_2011_integers_l474_47416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_parallel_implies_angle_l474_47429

/-- Given a triangle ABC with sides a, b, c, if vector (sin B - sin A, √3a + c) is parallel to 
    vector (sin C, a + b), then angle B = 5π/6 -/
theorem triangle_vector_parallel_implies_angle (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ k * (Real.sin B - Real.sin A) = Real.sin C ∧ k * (Real.sqrt 3 * a + c) = a + b) :
  B = 5 * π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_parallel_implies_angle_l474_47429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_owners_without_motorcycle_l474_47494

theorem car_owners_without_motorcycle (total car_owners motorcycle_owners : ℕ)
  (h1 : total = 400)
  (h2 : car_owners = 380)
  (h3 : motorcycle_owners = 60)
  (h4 : ∀ a : Fin total, (a.val < car_owners) ∨ (a.val < motorcycle_owners))
  : (car_owners - (car_owners + motorcycle_owners - total)) = 340 :=
by
  -- Replace this with the actual proof steps
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_owners_without_motorcycle_l474_47494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_subsets_l474_47412

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 * x - 4 ≥ 0}
def B : Set ℝ := {x | 2 ≤ Real.exp (x * Real.log 2) ∧ Real.exp (x * Real.log 2) < 16}
def C : Set ℝ := {0, 1, 2}

-- Define the set M
def M : Set ℝ := (A ∪ B) ∩ C

-- State the theorem
theorem set_operations_and_subsets :
  (Set.univ \ (A ∩ B) = {x : ℝ | x < 2 ∨ x ≥ 4}) ∧
  (Set.powerset M \ {M} = {∅, {1}, {2}}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_subsets_l474_47412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_stratified_sample_l474_47414

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Represents the number of people to be sampled from each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Calculates the total population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Calculates the sampling ratio -/
def samplingRatio (p : Population) (sampleSize : ℕ) : ℚ :=
  sampleSize / (totalPopulation p : ℚ)

/-- Rounds a rational number to the nearest natural number -/
def roundToNat (q : ℚ) : ℕ :=
  (q + 1/2).floor.toNat

/-- Calculates the stratified sample -/
def stratifiedSample (p : Population) (sampleSize : ℕ) : Sample :=
  let ratio := samplingRatio p sampleSize
  { elderly := roundToNat (ratio * p.elderly),
    middleAged := roundToNat (ratio * p.middleAged),
    young := roundToNat (ratio * p.young) }

/-- Theorem: The stratified sample for the given population and sample size is correct -/
theorem correct_stratified_sample :
  let p : Population := { elderly := 55, middleAged := 108, young := 162 }
  let sampleSize : ℕ := 36
  stratifiedSample p sampleSize = { elderly := 6, middleAged := 12, young := 18 } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_stratified_sample_l474_47414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l474_47499

/-- Geometric sequence with given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Arithmetic sequence with given properties -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Theorem stating the properties of sequences a_n, b_n, and c_n -/
theorem sequence_properties (a b : ℕ → ℝ) (h_geom : geometric_sequence a) 
    (h_arith : arithmetic_sequence b) (h_eq : a 1 = b 1) (h_a2 : a 2 = 3) 
    (h_a3 : a 3 = 9) (h_a4b14 : a 4 = b 14) :
  (∀ n : ℕ, b n = 2 * n - 1) ∧ 
  (∀ n : ℕ, (Finset.range n).sum (λ i ↦ a (i + 1) - b (i + 1)) = 
    3^n / 2 - n^2 - 1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l474_47499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_ratio_on_curve_W_l474_47445

-- Define the circle M
noncomputable def circle_M (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 12

-- Define point A
noncomputable def point_A : ℝ × ℝ := (-Real.sqrt 2, 0)

-- Define curve W
def curve_W (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the theorem
theorem slope_ratio_on_curve_W 
  (C D E : ℝ × ℝ) 
  (hC : curve_W C.1 C.2) 
  (hD : curve_W D.1 D.2) 
  (hE : curve_W E.1 E.2) 
  (h_origin : ∃ (k : ℝ), C.2 = k * C.1 ∧ D.2 = k * D.1) 
  (h_perp : (E.2 - C.2) * (D.1 - C.1) = -(E.1 - C.1) * (D.2 - C.2)) 
  (h_distinct : C ≠ D) :
  let F : ℝ × ℝ := (2 * C.1, 0)
  let k₁ := (E.2 - D.2) / (E.1 - D.1)
  let k₂ := -C.2 / C.1
  k₁ / k₂ = -1/3 := by
  sorry

#check slope_ratio_on_curve_W

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_ratio_on_curve_W_l474_47445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_storage_space_l474_47481

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℚ :=
  d.length * d.width * d.height

/-- Represents the storage information for a company -/
structure StorageInfo where
  boxDimensions : BoxDimensions
  costPerBoxPerMonth : ℚ
  totalMonthlyPayment : ℚ

/-- Calculates the total space occupied by boxes given storage information -/
def totalSpaceOccupied (info : StorageInfo) : ℚ :=
  let numberOfBoxes := info.totalMonthlyPayment / info.costPerBoxPerMonth
  numberOfBoxes * boxVolume info.boxDimensions

/-- Theorem stating the total space occupied by the company's boxes -/
theorem company_storage_space (info : StorageInfo)
    (h1 : info.boxDimensions = ⟨15, 12, 10⟩)
    (h2 : info.costPerBoxPerMonth = 2/5)
    (h3 : info.totalMonthlyPayment = 240) :
    totalSpaceOccupied info = 1080000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_storage_space_l474_47481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_pizza_piece_area_smallest_pizza_piece_area_proof_l474_47446

/-- The area of the smallest piece of a circular pizza with radius 2,
    when sliced into 4 equal strips both vertically and horizontally. -/
theorem smallest_pizza_piece_area : ℝ :=
  let r : ℝ := 2  -- radius of the pizza
  let n : ℕ := 4   -- number of strips in each direction
  -- Area of the smallest piece (corner piece)
  Real.pi - 1

-- The proof is omitted
theorem smallest_pizza_piece_area_proof : smallest_pizza_piece_area = Real.pi - 1 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_pizza_piece_area_smallest_pizza_piece_area_proof_l474_47446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_0_range_of_m_l474_47453

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 1|

-- Theorem for the solution set of f(x) ≤ 0
theorem solution_set_f_leq_0 :
  {x : ℝ | f x ≤ 0} = {x : ℝ | x ≥ 1/3 ∨ x ≤ -3} :=
sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x - 2*m^2 ≤ 4*m) ↔ m ≤ -5/2 ∨ m ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_0_range_of_m_l474_47453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sum_of_three_seventh_powers_l474_47430

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 8  -- Add this case to cover Nat.zero
  | 1 => 8
  | 2 => 20
  | (n + 3) => a (n + 2)^2 + 12 * a (n + 2) * a (n + 1) + a (n + 2) + 11 * a (n + 1)

-- Theorem statement
theorem no_sum_of_three_seventh_powers (n : ℕ) :
  ¬ ∃ (x y z : ℤ), a n = x^7 + y^7 + z^7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sum_of_three_seventh_powers_l474_47430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_sphere_ratio_l474_47485

/-- A truncated right circular cone with an inscribed sphere -/
structure TruncatedConeWithSphere where
  R : ℝ  -- radius of the larger base
  r : ℝ  -- radius of the smaller base
  s : ℝ  -- radius of the inscribed sphere
  h : ℝ  -- height of the truncated cone

/-- Volume of a sphere -/
noncomputable def sphereVolume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius^3

/-- Volume of a truncated cone -/
noncomputable def truncatedConeVolume (cone : TruncatedConeWithSphere) : ℝ :=
  (Real.pi * cone.h / 3) * (cone.R^2 + cone.r^2 + cone.R * cone.r)

theorem truncated_cone_sphere_ratio 
  (cone : TruncatedConeWithSphere) 
  (hR : cone.R > 0) 
  (hr : cone.r > 0) 
  (hs : cone.s > 0) 
  (h_inscribed : cone.s = Real.sqrt (cone.R * cone.r)) 
  (h_volume : truncatedConeVolume cone = 3 * sphereVolume cone.s) :
  cone.R / cone.r = (4 + Real.sqrt 13) / 3 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_sphere_ratio_l474_47485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l474_47463

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos (2 * x))
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) :
  f (A / 2) = 2 →
  C = 2 * Real.pi / 3 →
  A + B + C = Real.pi →
  (4 * Real.pi) = Real.pi * (a^2 / (2 * Real.sin A * Real.sin B * Real.sin C)) →
  a + b + c = 4 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l474_47463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearing_time_approx_l474_47484

/-- The time (in seconds) it takes for two trains to be completely clear of each other -/
noncomputable def clearingTime (train1Length train2Length : ℝ) (train1Speed train2Speed : ℝ) : ℝ :=
  (train1Length + train2Length) / ((train1Speed + train2Speed) * 1000 / 3600)

theorem train_clearing_time_approx :
  ∀ (ε : ℝ), ε > 0 →
  ∃ (train1Length train2Length train1Speed train2Speed : ℝ),
    train1Length = 250 ∧
    train2Length = 330 ∧
    train1Speed = 120 ∧
    train2Speed = 95 ∧
    |clearingTime train1Length train2Length train1Speed train2Speed - 9.71| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearing_time_approx_l474_47484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_values_l474_47473

theorem matrix_transformation_values (θ k : ℝ) : 
  0 < θ → θ < 2 * Real.pi →
  0 < k → k < 1 →
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, k]
  B * A = !![0, -1; (1/2), 0] →
  k = 1/2 ∧ θ = Real.pi/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_values_l474_47473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_percentage_l474_47457

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ)
  (germination_rate_plot1 germination_rate_plot2 : ℚ) :
  seeds_plot1 = 500 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 30 / 100 →
  germination_rate_plot2 = 50 / 100 →
  (seeds_plot1 * germination_rate_plot1 + seeds_plot2 * germination_rate_plot2) / (seeds_plot1 + seeds_plot2) = 3571 / 10000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_percentage_l474_47457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_15_l474_47424

theorem remainder_sum_mod_15 (x y z : ℕ) 
  (hx_pos : x > 0) (hy_pos : y > 0) (hz_pos : z > 0)
  (hx : x % 15 = 11) 
  (hy : y % 15 = 13) 
  (hz : z % 15 = 9) : 
  (2 * (x % 15) + y % 15 + z % 15) % 15 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_15_l474_47424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_continuity_l474_47452

-- Define the piecewise function
noncomputable def f (b c : ℝ) : ℝ → ℝ := λ x =>
  if x > 2 then 3 * x + b else 5 * x + c

-- State the theorem
theorem piecewise_function_continuity (b c : ℝ) :
  Continuous (f b c) ↔ b - c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_continuity_l474_47452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_property_l474_47408

/-- Represents a continued fraction with k terms -/
def continued_fraction (k : ℕ) : ℚ :=
  match k with
  | 0 => 0
  | k + 1 => 1 / (1 + continued_fraction k)

/-- The numerator of the continued fraction with k terms -/
def m (k : ℕ) : ℕ := (continued_fraction k).num.natAbs

/-- The denominator of the continued fraction with k terms -/
def n (k : ℕ) : ℕ := (continued_fraction k).den

theorem continued_fraction_property :
  (m 1988 : ℤ) ^ 2 + (m 1988 : ℤ) * (n 1988 : ℤ) - (n 1988 : ℤ) ^ 2 = -1 ∧
  Nat.Coprime (m 1988) (n 1988) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_property_l474_47408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l474_47435

-- Define the trajectory C as a set of points (x, y) satisfying y^2 = 4x
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the point F
def F : ℝ × ℝ := (1, 0)

-- Define the line x = -1
def line_x_neg_1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the point N
def N : ℝ × ℝ := (-1, 0)

-- Define the line l with slope k passing through N
def line_l (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * (p.1 + 1)}

-- Theorem statement
theorem constant_dot_product 
  (k : ℝ) 
  (hk : k ≠ 0 ∧ (k > -1 ∧ k < 0 ∨ k > 0 ∧ k < 1)) 
  (A B : ℝ × ℝ) 
  (hA : A ∈ C ∩ line_l k) 
  (hB : B ∈ C ∩ line_l k) 
  (hAB : A ≠ B) 
  (D : ℝ × ℝ) 
  (hD : D ∈ C ∧ D ≠ A ∧ D ≠ B) :
  ∃ (P Q : ℝ × ℝ), 
    (P.1 = 1 ∧ Q.1 = 1) ∧ 
    (∃ (t : ℝ), P = (1 - t) • F + t • A) ∧
    (∃ (s : ℝ), Q = (1 - s) • F + s • B) ∧
    (P.1 * Q.1 + P.2 * Q.2 = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l474_47435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_theorem_l474_47434

/-- The number of arrangements for 5 people with one person between A and B -/
def arrangements_count : ℕ := 36

/-- The total number of people in the row -/
def total_people : ℕ := 5

/-- The number of people between A and B -/
def people_between : ℕ := 1

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem arrangements_theorem :
  arrangements_count = 
    (total_people - 2) *    -- Choose 1 person to be between A and B
    2 *                     -- A and B can be in either order
    factorial (total_people - 2) :=  -- Permute the remaining people
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_theorem_l474_47434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_seven_l474_47487

theorem largest_multiple_of_seven (n : ℤ) : 
  (n * 7 = 147) ↔ 
  (∀ m : ℤ, m * 7 > 147 → -m * 7 ≤ -150) ∧ 
  (-147 > -150) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_seven_l474_47487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_completion_time_l474_47467

/- Define the work rates and time worked by A -/
noncomputable def work_rate_A : ℝ := 1 / 15
noncomputable def work_rate_B : ℝ := 1 / 14.999999999999996
def days_worked_by_A : ℝ := 5

/- Define the portion of work completed by A and remaining for B -/
noncomputable def work_completed_by_A : ℝ := work_rate_A * days_worked_by_A
noncomputable def work_remaining_for_B : ℝ := 1 - work_completed_by_A

/- Theorem statement -/
theorem B_completion_time :
  work_remaining_for_B / work_rate_B = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_completion_time_l474_47467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_mixture_density_l474_47433

/-- Given three liquids with densities in ratio 6:3:2, prove that the mass ratio
    satisfying the mixture density condition meets the equation 4x + 15y = 7 --/
theorem liquid_mixture_density (ρ₁ ρ₂ ρ₃ m₁ m₂ m₃ : ℝ) 
  (h_ratio : ρ₁ / ρ₂ = 2 ∧ ρ₂ / ρ₃ = 3/2)
  (h_mass : m₁ ≥ 3.5 * m₂) :
  let x := m₂ / m₁
  let y := m₃ / m₁
  let ρ_mix := (m₁ + m₂ + m₃) / (m₁/ρ₁ + m₂/ρ₂ + m₃/ρ₃)
  let ρ_mean := (ρ₁ + ρ₂ + ρ₃) / 3
  ρ_mix = ρ_mean →
  4*x + 15*y = 7 ∧ x ≤ 2/7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_mixture_density_l474_47433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_time_decrease_l474_47470

/-- Represents the rate of decrease in average daily homework time -/
def x : Real := sorry

/-- The initial average daily homework time in minutes -/
def initial_time : Real := 100

/-- The final average daily homework time in minutes -/
def final_time : Real := 70

/-- The number of semesters over which the adjustments were made -/
def num_semesters : Nat := 2

/-- Theorem stating that the equation correctly represents the decrease in homework time -/
theorem homework_time_decrease : initial_time * (1 - x)^num_semesters = final_time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_time_decrease_l474_47470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l474_47404

-- Define the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (4 : ℝ)^x - 2^(x+1) + a

-- Define the function g
noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  (f x a) * (f (x+1) a - (2 : ℝ)^(2*x+1)) + a^2

theorem function_properties (a : ℝ) :
  -- Part 1: Solution set of f(x) < a
  (∀ x : ℝ, f x a < a ↔ x < 1) ∧
  -- Part 2: If min g(x) = g(a) and g(a) = 2a, then a = 0 or a = 2
  ((∀ x : ℝ, g x a ≥ g a a) ∧ g a a = 2*a → a = 0 ∨ a = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l474_47404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_p_is_one_tenth_l474_47420

/-- The number of zero digits in the representation of natural numbers from 1 to n -/
def number_of_zero_digits (n : ℕ) : ℕ :=
  sorry

/-- The total number of digits used to represent natural numbers from 1 to n -/
def total_number_of_digits (n : ℕ) : ℕ :=
  sorry

/-- The probability of drawing a zero digit from the set of digits representing natural numbers from 1 to n -/
noncomputable def p (n : ℕ) : ℝ :=
  (number_of_zero_digits n : ℝ) / (total_number_of_digits n : ℝ)

/-- The theorem stating that the limit of p(n) as n approaches infinity is 1/10 -/
theorem limit_p_is_one_tenth :
  Filter.Tendsto p Filter.atTop (nhds (1/10)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_p_is_one_tenth_l474_47420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_127_32_l474_47486

/-- Represents the side length of a triangle in the sequence -/
def triangleSide (n : ℕ) : ℚ :=
  (1 : ℚ) / 2^n

/-- Represents the perimeter contribution of the nth triangle -/
def trianglePerimeter (n : ℕ) : ℚ :=
  if n = 0 then
    2 * triangleSide 0 + triangleSide 1
  else if n < 5 then
    triangleSide n + triangleSide (n + 1)
  else
    2 * triangleSide 5

/-- The total perimeter of the figure -/
def totalPerimeter : ℚ :=
  Finset.sum (Finset.range 6) trianglePerimeter

theorem perimeter_is_127_32 : totalPerimeter = 127 / 32 := by
  sorry

#eval totalPerimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_127_32_l474_47486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_existence_l474_47444

-- Define the basic structures
structure Plane where

structure Line where

structure Point where

-- Define the planes, line, and points
variable (P Q : Plane)
variable (p : Line)
variable (A B C D E : Point)

-- Define the relationships between planes, lines, and points
axiom intersect : Plane → Plane → Line
axiom lies_in : Point → Plane → Prop
axiom not_in : Point → Plane → Prop
axiom on_line : Point → Line → Prop
axiom parallel : Line → Line → Prop
axiom perpendicular : Line → Line → Prop
axiom distance : Point → Point → ℝ
axiom has_inscribed_circle : Point → Point → Point → Point → Prop

-- State the theorem
theorem quadrilateral_existence 
  (h1 : intersect P Q = p)
  (h2 : lies_in A P ∧ not_in A Q)
  (h3 : lies_in C Q ∧ not_in C P)
  (h4 : ¬on_line A p ∧ ¬on_line C p)
  (h5 : parallel (Line.mk) (Line.mk))
  (h6 : distance A D = distance B C)
  (h7 : lies_in B P)
  (h8 : lies_in D Q)
  (h9 : has_inscribed_circle A B C D)
  (h10 : perpendicular (Line.mk) (Line.mk))
  : ∃ (n : ℕ), n ≤ 2 ∧ 
    (n = 0 ↔ distance A E < distance C E) ∧
    (n = 1 ↔ distance A E = distance C E) ∧
    (n = 2 ↔ distance A E > distance C E) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_existence_l474_47444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_fifteen_percent_l474_47441

/-- Calculates the interest rate given principal, simple interest, and time -/
noncomputable def calculate_interest_rate (principal : ℝ) (simple_interest : ℝ) (time : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that given the specified conditions, the interest rate is 15% -/
theorem interest_rate_is_fifteen_percent 
  (principal : ℝ) 
  (simple_interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 400) 
  (h2 : simple_interest = 120) 
  (h3 : time = 2) : 
  calculate_interest_rate principal simple_interest time = 15 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 400 120 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_fifteen_percent_l474_47441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l474_47464

/-- The number of blue balls in the bin -/
def blue_balls : ℕ := 8

/-- The amount won or lost per ball drawn in dollars -/
def amount : ℚ := 3

/-- The expected amount won in dollars -/
def expected_win : ℚ := 3/4

/-- Calculates the expected value given the number of red balls -/
def expected_value (k : ℕ) : ℚ :=
  (blue_balls : ℚ) / ((blue_balls : ℚ) + k) * amount +
  (k : ℚ) / ((blue_balls : ℚ) + k) * (-amount)

/-- Theorem stating that the number of red balls is closest to 5 -/
theorem red_balls_count :
  ∃ (k : ℕ), expected_value k = expected_win ∧
  ∀ (m : ℕ), m ≠ k → |((m : ℚ) - 5)| > |((k : ℚ) - 5)| := by
  sorry

#eval expected_value 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l474_47464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l474_47427

theorem polynomial_roots : ∃ (f : ℝ → ℝ), 
  (∀ x, f x = x^4 + x^3 - 7*x^2 - x + 6) ∧ 
  (f (-2) = 0) ∧ 
  (f (-1) = 0) ∧ 
  ((deriv f) (-1) = 0) ∧ 
  (f 3 = 0) ∧
  (∀ x, f x = 0 → x = -2 ∨ x = -1 ∨ x = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l474_47427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_condition_l474_47449

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define the origin and points
variable (O A B C D : V)

-- Define the scalar k
variable (k : ℝ)

-- Define coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c d : ℝ), a • (A - D) + b • (B - D) + c • (C - D) = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

-- State the theorem
theorem coplanar_condition (h : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = 0) :
  coplanar A B C D ↔ k = -13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_condition_l474_47449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_no_intersection_l474_47492

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a step in the staircase -/
structure Step where
  width : ℕ
  height : ℕ

/-- Generates the staircase with n steps -/
def generateStaircase (n : ℕ) : List Step :=
  List.range n |>.map (fun i => { width := i + 1, height := 1 })

/-- Calculates the coordinates of the upper right point of the staircase -/
def upperRightPoint (n : ℕ) : Point :=
  { x := n * (n + 1) / 2, y := n }

/-- Checks if a step intersects the line segment between two points -/
def intersects_segment (p1 p2 : Point) (step : Step) : Prop :=
  sorry -- Define the intersection check logic here

/-- Theorem: The segment from (0,0) to the upper right point does not intersect the staircase -/
theorem staircase_no_intersection (n : ℕ) :
  let staircase := generateStaircase n
  let endPoint := upperRightPoint n
  ∀ step ∈ staircase, ¬ intersects_segment ⟨0, 0⟩ endPoint step := by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_no_intersection_l474_47492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_a_votes_l474_47493

def total_votes : ℕ := 560000
def invalid_percentage : ℚ := 15 / 100
def candidate_a_percentage : ℚ := 55 / 100

theorem candidate_a_votes :
  let valid_votes : ℚ := (total_votes : ℚ) * (1 - invalid_percentage)
  let candidate_a_votes : ℚ := valid_votes * candidate_a_percentage
  ⌊candidate_a_votes⌋ = 261800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_a_votes_l474_47493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_value_l474_47472

/-- Given that γ is directly proportional to δ and γ = 5 when δ = -10,
    prove that γ = -25/2 when δ = 25 -/
theorem gamma_value : ∃ (γ δ : ℚ → ℚ) (k : ℚ), 
    (∀ x, γ x = k * δ x) ∧ 
    (γ 5 = -10) ∧ 
    (γ 25 = -25/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_value_l474_47472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l474_47418

theorem divisor_problem (n d : ℕ) : 
  (∃! n, n < 180 ∧ n % d = 5) → d = 175 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l474_47418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_implies_a_value_l474_47411

/-
  Define two circles:
  C₁: (x-a)² + y² = 4
  C₂: x² + (y - √5)² = a²
-/
def C₁ (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 4
def C₂ (a x y : ℝ) : Prop := x^2 + (y - Real.sqrt 5)^2 = a^2

/- Define the distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/- Define external tangency of two circles -/
noncomputable def externally_tangent (a : ℝ) : Prop :=
  distance a 0 0 (Real.sqrt 5) = 2 + |a|

/- Theorem statement -/
theorem circles_tangent_implies_a_value (a : ℝ) :
  externally_tangent a → (a = 1/4 ∨ a = -1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_implies_a_value_l474_47411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_linear_equations_l474_47496

-- Define the set of equations
def equations : List String := [
  "4x - 7 = 0",
  "3x + y = z",
  "x - 7 = x^2",
  "4xy = 3",
  "(x + y) / 2 = y / 3",
  "3 / x = 1",
  "y(y - 1) = y^2 - x"
]

-- Define a predicate for linear equations with two variables
def isLinearTwoVar (eq : String) : Bool :=
  match eq with
  | "(x + y) / 2 = y / 3" => true
  | "y(y - 1) = y^2 - x" => true
  | _ => false

-- Theorem statement
theorem two_linear_equations :
  (equations.filter isLinearTwoVar).length = 2 := by
  -- Evaluate the filter and check the length
  simp [equations, isLinearTwoVar]
  -- The proof is completed by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_linear_equations_l474_47496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_difference_zero_l474_47490

/-- Recurrence relation for the sequence x_n -/
noncomputable def recurrence (x : ℝ) : ℝ := (x + (2 - Real.sqrt 3)) / (1 - (2 - Real.sqrt 3) * x)

/-- The sequence x_n -/
noncomputable def x : ℕ → ℝ
  | 0 => 0  -- arbitrary initial value
  | n + 1 => recurrence (x n)

/-- Theorem stating that x_1001 - x_401 = 0 -/
theorem x_difference_zero : x 1001 - x 401 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_difference_zero_l474_47490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_negative_one_l474_47477

def f (k : ℝ) (x : ℝ) : ℝ := k * x^4 + 2 * x

theorem f_derivative_at_negative_one :
  ∃ (k : ℝ), (∀ x, f k x = k * x^4 + 2 * x) ∧
             deriv (f k) (-1) = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_negative_one_l474_47477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_KLMN_l474_47462

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define the points K, L, N
structure Points :=
  (bk an bl : ℝ)

-- Define the theorem
theorem area_of_quadrilateral_KLMN 
  (abc : Triangle)
  (points : Points)
  (h1 : abc.a = 13)
  (h2 : abc.b = 14)
  (h3 : abc.c = 15)
  (h4 : points.bk = 14/13)
  (h5 : points.an = 10)
  (h6 : points.bl = 1) :
  ∃ (area : ℝ), area = 36503/1183 ∧ 
  (area = (abc.a * abc.b * abc.c) / (4 * (abc.a + abc.b + abc.c))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_KLMN_l474_47462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_period_4_l474_47419

-- Define the last non-zero digit function
def lastNonZeroDigit (n : ℕ) : ℕ := sorry

-- Define the factorial function
def fact (n : ℕ) : ℕ := sorry

-- Define the function f
def f (n : ℕ) : ℕ := lastNonZeroDigit (fact n)

-- Define the function g
def g (a : ℕ) : ℕ := 
  let r : ℕ := sorry
  let a_i : ℕ → ℕ := sorry
  lastNonZeroDigit (fact (Finset.sum (Finset.range r) (fun i => 5^(a_i i))))

-- The theorem to prove
theorem g_period_4 : ∃ (k : ℕ), ∀ (a : ℕ), g (a + 4) = g a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_period_4_l474_47419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_pyramid_in_regular_tetrahedron_l474_47443

/-- The volume of a pyramid PQRS in a regular tetrahedron PQRSTU with edge length 2 -/
noncomputable def volume_pyramid_in_tetrahedron : ℝ :=
  2 * Real.sqrt 6 / 9

/-- A regular tetrahedron PQRSTU with edge length 2 -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_regular : edge_length = 2

/-- Theorem: The volume of pyramid PQRS in a regular tetrahedron PQRSTU with edge length 2 is 2√6/9 -/
theorem volume_pyramid_in_regular_tetrahedron (t : RegularTetrahedron) :
  volume_pyramid_in_tetrahedron = 2 * Real.sqrt 6 / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_pyramid_in_regular_tetrahedron_l474_47443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_rules_correct_l474_47410

noncomputable section

/-- Discount function based on total amount -/
def discount (amount : ℝ) : ℝ :=
  if amount ≤ 500 then amount
  else if amount ≤ 800 then amount * 0.8
  else 800 * 0.8 + (amount - 800) * 0.6

/-- Total payment for two items -/
def total_payment (price1 price2 : ℝ) : ℝ :=
  discount (price1 + price2)

/-- Payment for a single item priced over 800 -/
def single_item_payment (a : ℝ) : ℝ :=
  800 * 0.8 + (a - 800) * 0.6

/-- Total payment for two items with total price 2000 and first item price x -/
def two_items_payment (x : ℝ) : ℝ :=
  if x < 500 then discount x + discount (2000 - x)
  else if x ≤ 800 then discount x + discount (2000 - x)
  else discount x + discount (2000 - x)

theorem discount_rules_correct :
  (total_payment 480 520 = 760) ∧
  (∀ a > 800, single_item_payment a = 0.6 * a + 160) ∧
  (∀ x, x < 500 → two_items_payment x = 1360 + 0.4 * x) ∧
  (∀ x, 500 ≤ x ∧ x ≤ 800 → two_items_payment x = 1360 + 0.2 * x) ∧
  (∀ x, 800 < x ∧ x < 1000 → two_items_payment x = 1520) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_rules_correct_l474_47410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_positions_l474_47426

noncomputable def line (x : ℝ) : ℝ := -Real.sqrt 3 * x + 2 * Real.sqrt 3

noncomputable def point_A : ℝ × ℝ := (-2, 0)
def point_O : ℝ × ℝ := (0, 0)

noncomputable def point_B : ℝ × ℝ := (2, 0)
noncomputable def point_C : ℝ × ℝ := (0, 2 * Real.sqrt 3)

noncomputable def angle_APO (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  Real.arctan ((y - 0) / (x - (-2))) - Real.arctan (y / x)

def is_on_BC (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  y = line x ∧ 0 ≤ x ∧ x ≤ 2

theorem five_positions :
  ∃ (S : Finset (ℝ × ℝ)), 
    Finset.card S = 5 ∧ 
    (∀ P ∈ S, is_on_BC P ∧ angle_APO P = π / 6) ∧
    (∀ P, is_on_BC P ∧ angle_APO P = π / 6 → P ∈ S) :=
by sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_positions_l474_47426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_l474_47406

/-- Proves that the total distance covered is 4 km given the specified conditions -/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) (h1 : walking_speed = 4)
  (h2 : running_speed = 8) (h3 : total_time = 1.5) : ℝ := by
  let total_distance := walking_speed * total_time / 2 + running_speed * total_time / 2
  have h4 : total_distance = 4 := by sorry
  exact total_distance

#check distance_covered

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_l474_47406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_theorem_l474_47491

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the transformation function g
noncomputable def g (x : ℝ) : ℝ := -(f (x + 2))

-- Theorem statement
theorem transformation_theorem :
  ∀ x : ℝ, g x = -(Real.log (x + 2) / Real.log 3) :=
by
  intro x
  -- Unfold the definitions of g and f
  unfold g f
  -- The proof is now obvious by definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_theorem_l474_47491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_park_fencing_cost_l474_47495

/-- The cost of fencing each side of a square park -/
noncomputable def cost_per_side (total_cost : ℝ) : ℝ := total_cost / 4

/-- Theorem: For a square park with a total fencing cost of 288 dollars, 
    the cost of fencing each side is 72 dollars -/
theorem square_park_fencing_cost : 
  cost_per_side 288 = 72 := by
  -- Unfold the definition of cost_per_side
  unfold cost_per_side
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_park_fencing_cost_l474_47495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_triples_l474_47466

/-- The number of ordered triples (a, b, c) satisfying the given conditions --/
def num_triples : ℕ := 2

/-- Predicate for a valid triple --/
def is_valid_triple (a b c : ℕ) : Prop :=
  a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ 
  (Real.log (a : ℝ) = c^2023 * Real.log (b : ℝ)) ∧
  (a + b + c = 2023)

/-- Theorem stating that there are exactly two valid triples --/
theorem two_valid_triples :
  ∃! (s : Finset (ℕ × ℕ × ℕ)), 
    s.card = num_triples ∧ 
    (∀ (t : ℕ × ℕ × ℕ), t ∈ s ↔ is_valid_triple t.1 t.2.1 t.2.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_triples_l474_47466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_10_value_l474_47440

def T (n : ℕ) : ℚ := n * (n + 1) / 2

def P (n : ℕ) : ℚ :=
  if n < 3 then 0
  else (Finset.range (n - 2)).prod (fun k => (T (k + 3)^2) / (T (k + 3) - 1))

theorem P_10_value : P 10 = 121 / 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_10_value_l474_47440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_neg_three_point_seven_l474_47475

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_neg_three_point_seven :
  floor (-3.7) = -4 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_neg_three_point_seven_l474_47475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_ratio_property_locus_forms_hyperbola_l474_47439

/-- Definition of a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  h_e_gt_one : e > 1

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space (vertical line for simplicity) -/
structure Line where
  k : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a vertical line -/
def distance_to_line (p : Point) (l : Line) : ℝ :=
  |p.x - l.k|

/-- Predicate to check if a point is on a hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- Theorem about the distance ratio property of hyperbolas -/
theorem hyperbola_distance_ratio_property (h : Hyperbola) (p : Point) (f : Point) (d : Line) :
  on_hyperbola h p → distance p f / distance_to_line p d = h.e := by sorry

/-- Theorem about the locus of points forming a hyperbola -/
theorem locus_forms_hyperbola (e : ℝ) (h_e_gt_one : e > 1) (f : Point) (l : Line) :
  ∃ h : Hyperbola, ∀ p : Point, 
    distance p f / distance_to_line p l = e → on_hyperbola h p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_ratio_property_locus_forms_hyperbola_l474_47439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_count_l474_47489

theorem division_with_remainder_count : 
  let S := {x : ℕ | x > 0 ∧ 47 % x = 7}
  Finset.card (Finset.filter (λ x => x > 0 ∧ 47 % x = 7) (Finset.range 47)) = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_count_l474_47489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debt_difference_l474_47459

/-- Represents the payments made by each person on the trip -/
structure Payments where
  tom : ℚ
  dorothy : ℚ
  sammy : ℚ
  alice : ℚ

/-- Calculates the total amount paid by all four people -/
def total_paid (p : Payments) : ℚ :=
  p.tom + p.dorothy + p.sammy + p.alice

/-- Calculates the amount each person should have paid for an even split -/
def even_split (p : Payments) : ℚ :=
  (total_paid p) / 4

/-- Theorem stating that the difference between Tom's debt to Sammy and Dorothy's debt to Alice is 20 -/
theorem debt_difference (p : Payments) (h : p = { tom := 180, dorothy := 200, sammy := 240, alice := 280 }) :
  (even_split p - p.tom) - (even_split p - p.dorothy) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debt_difference_l474_47459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l474_47423

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) * (a (n + 1) + 1) = a n + 2023

theorem min_sum_first_two_terms (a : ℕ → ℕ) :
  sequence_property a → (∀ n, a n > 0) → 
  ∃ (a₁ a₂ : ℕ), a 1 = a₁ ∧ a 2 = a₂ ∧ a₁ + a₂ = 136 ∧ 
  (∀ (b₁ b₂ : ℕ), sequence_property (Function.update (Function.update a 1 b₁) 2 b₂) → b₁ + b₂ ≥ 136) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l474_47423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_number_when_one_is_specific_thirty_five_twentyfour_never_appears_l474_47460

def arithmetic_mean (a b : ℚ) : ℚ := (a + b) / 2

def harmonic_mean (a b : ℚ) : ℚ := 2 * a * b / (a + b)

def board_transform (pair : ℚ × ℚ) : ℚ × ℚ :=
  let (a, b) := pair
  (arithmetic_mean a b, harmonic_mean a b)

def initial_board : ℚ × ℚ := (1, 2)

theorem other_number_when_one_is_specific (n : ℕ) :
  let pair := (board_transform^[n] initial_board)
  (pair.1 = 941664 / 665857 ∨ pair.2 = 941664 / 665857) →
  (pair.1 = 665857 / 470832 ∨ pair.2 = 665857 / 470832) :=
sorry

theorem thirty_five_twentyfour_never_appears (n : ℕ) :
  let pair := (board_transform^[n] initial_board)
  pair.1 ≠ 35 / 24 ∧ pair.2 ≠ 35 / 24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_number_when_one_is_specific_thirty_five_twentyfour_never_appears_l474_47460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_parts_three_planes_l474_47413

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this statement

/-- The number of parts that a set of planes divides 3D space into -/
def num_parts (planes : Finset Plane3D) : ℕ :=
  sorry

/-- The maximum number of parts that three planes can divide 3D space into -/
theorem max_parts_three_planes :
  ∃ (planes : Finset Plane3D), (planes.card = 3) ∧ (num_parts planes = 8) ∧
    (∀ (other_planes : Finset Plane3D), (other_planes.card = 3) → (num_parts other_planes ≤ 8)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_parts_three_planes_l474_47413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_z_efficiency_decrease_l474_47415

/-- Represents the efficiency of a car in miles per gallon at a given speed. -/
structure CarEfficiency where
  speed : ℝ  -- Speed in miles per hour
  mpg : ℝ    -- Miles per gallon

/-- Calculates the percentage decrease between two values. -/
noncomputable def percentageDecrease (original : ℝ) (new : ℝ) : ℝ :=
  (original - new) / original * 100

theorem car_z_efficiency_decrease :
  let efficiency45 : CarEfficiency := ⟨45, 45⟩
  let efficiency60 : CarEfficiency := ⟨60, 360 / 10⟩
  percentageDecrease efficiency45.mpg efficiency60.mpg = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_z_efficiency_decrease_l474_47415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_satisfying_functions_l474_47405

/-- A function satisfying the given inequality for all real x, y, and z -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x^2 * y) + f (x * z) - f x * f (y * z) ≥ 2

/-- The set of all functions satisfying the inequality -/
def SatisfyingFunctions : Set (ℝ → ℝ) :=
  {f | SatisfiesInequality f}

/-- There are exactly two functions satisfying the inequality -/
theorem two_satisfying_functions :
  ∃! (s : Finset (ℝ → ℝ)), s.toSet = SatisfyingFunctions ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_satisfying_functions_l474_47405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_in_decade_growth_rate_l474_47450

/-- The annual growth rate for a quantity that quadruples over 10 years -/
theorem quadruple_in_decade_growth_rate :
  ∀ (initial_value : ℝ) (growth_rate : ℝ),
    initial_value > 0 →
    (1 + growth_rate) ^ 10 = 4 →
    growth_rate = (4 : ℝ) ^ (1/10 : ℝ) - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_in_decade_growth_rate_l474_47450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l474_47447

theorem triangle_side_length (A B C : EuclideanSpace ℝ (Fin 2)) (angleB : ℝ) (lengthAB lengthAC : ℝ) :
  angleB = π / 4 →
  dist A B = 100 →
  dist A C = 50 * Real.sqrt 2 →
  dist B C = 100 * Real.sqrt 1.5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l474_47447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_problem_l474_47483

noncomputable def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem diamond_problem : diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * Real.sqrt 2 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_problem_l474_47483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drop_height_l474_47422

/-- The rebound factor of the ball -/
noncomputable def rebound_factor : ℝ := 0.75

/-- The total distance traveled by the ball -/
noncomputable def total_distance : ℝ := 305

/-- The number of times the ball touches the floor -/
def bounce_count : ℕ := 5

/-- Calculates the sum of the geometric series for the ball's trajectory -/
noncomputable def trajectory_sum : ℝ :=
  1 + 2 * (1 - rebound_factor ^ (bounce_count - 1)) / (1 - rebound_factor)

/-- The original height from which the ball was dropped -/
noncomputable def original_height : ℝ := total_distance / trajectory_sum

theorem ball_drop_height :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ abs (original_height - 56.3) < ε := by
  sorry

#eval bounce_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drop_height_l474_47422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l474_47461

-- Define the ellipse parameters
noncomputable def Ellipse (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Helper function to calculate area of quadrilateral OACB
noncomputable def area_OACB (O C A B : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem ellipse_properties (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : b = 1) -- short axis is 2
  (h4 : eccentricity a b = Real.sqrt 3 / 2) :
  (∃ (E : Set (ℝ × ℝ)), E = Ellipse 2 1) ∧ 
  (∃ (max_area : ℝ), max_area = 2 * Real.sqrt 3 / 3 ∧
    ∀ (F C A B : ℝ × ℝ),
      F.1 = Real.sqrt 3 ∧ F.2 = 0 ∧  -- Right focus
      C.1 = 2 ∧ C.2 = 0 ∧            -- Right vertex
      A ∈ Ellipse 2 1 ∧ B ∈ Ellipse 2 1 ∧
      (∃ (m : ℝ), A.1 = m * A.2 + Real.sqrt 3 ∧ 
                  B.1 = m * B.2 + Real.sqrt 3) →
      area_OACB (0, 0) C A B ≤ max_area) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l474_47461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l474_47425

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the foci -/
def foci (c : ℝ) : Prop := ∃ F₁ F₂ : ℝ × ℝ, F₁ = (-c, 0) ∧ F₂ = (c, 0)

/-- Theorem stating the properties of the ellipse and the fixed point -/
theorem ellipse_and_fixed_point :
  ∀ a b c : ℝ,
  a > b ∧ b > 0 ∧
  b = Real.sqrt 3 ∧
  (∃ A B : ℝ × ℝ, A = (a, 0) ∧ B = (0, -b) ∧ 
    (abs (a * b) / Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 21 / 7)) →
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ P Q : ℝ × ℝ,
    P.1 = 4 ∧ Q.1 = 4 ∧ foci c →
    (P.1 - (-c), P.2) • (Q.1 - c, Q.2) = 0 →
    ∃ fixed_point : ℝ × ℝ, 
      fixed_point = (4 + Real.sqrt 15, 0) ∨ fixed_point = (4 - Real.sqrt 15, 0)) :=
by sorry

#check ellipse_and_fixed_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l474_47425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_and_regularity_l474_47482

-- Define a polygon P
structure Polygon where
  n : ℕ
  n_ge_3 : n ≥ 3

-- Define the sum of interior angles S
def SumInteriorAngles (P : Polygon) : ℝ := sorry

-- Define the sum of exterior angles
def SumExteriorAngles (P : Polygon) : ℝ := 360

-- Define interior and exterior angles
def InteriorAngles (P : Polygon) : Set ℝ := sorry
def ExteriorAngles (P : Polygon) : Set ℝ := sorry

-- Define the relationship between corresponding interior and exterior angles
def Corresponding (P : Polygon) (a b : ℝ) : Prop := sorry

-- Define the relationship between interior and exterior angles
axiom interior_exterior_relation (P : Polygon) :
  ∀ (a b : ℝ), (a ∈ InteriorAngles P ∧ b ∈ ExteriorAngles P ∧ Corresponding P a b) → a = 8.5 * b

-- Theorem to prove
theorem sum_interior_angles_and_regularity (P : Polygon) :
  SumInteriorAngles P = 3060 ∧
  ¬(∀ (x y : ℝ), x ∈ InteriorAngles P ∧ y ∈ InteriorAngles P → x = y) ∧
  ¬(∀ (x y : ℝ), x ∈ InteriorAngles P ∧ y ∈ InteriorAngles P → x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_and_regularity_l474_47482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_placement_theorem_l474_47437

/-- The number of ways to place 5 numbered balls into 5 numbered boxes -/
def ball_placement_count : ℕ := 20

/-- The set of ball numbers -/
def ball_numbers : Finset ℕ := Finset.range 5

/-- The set of box numbers -/
def box_numbers : Finset ℕ := Finset.range 5

/-- A placement of balls into boxes -/
def Placement := ball_numbers → box_numbers

/-- Predicate for a valid placement -/
def is_valid_placement (p : Placement) : Prop :=
  Function.Injective p ∧ (∃ (s : Finset ball_numbers), s.card = 2 ∧ ∀ i ∈ s, p i = i)

/-- The set of all valid placements -/
noncomputable def valid_placements : Finset Placement :=
  sorry

theorem ball_placement_theorem :
  valid_placements.card = ball_placement_count := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_placement_theorem_l474_47437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l474_47465

noncomputable def π : ℝ := Real.pi

/-- A function that represents the horizontal stretch transformation -/
def horizontalStretch (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := fun x ↦ f (k * x)

/-- A function that represents the horizontal shift transformation -/
def horizontalShift (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := fun x ↦ f (x - h)

/-- A function that represents the vertical shift transformation -/
def verticalShift (f : ℝ → ℝ) (v : ℝ) : ℝ → ℝ := fun x ↦ f x + v

/-- The main theorem stating the relationship between f and the transformed function -/
theorem function_transformation (f : ℝ → ℝ) :
  (verticalShift (horizontalShift (horizontalStretch f (1/2)) (π/3)) (-1)) = (fun x ↦ 1/2 * Real.sin x) →
  f = (fun x ↦ 1/2 * Real.sin (2*x - π/3) + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l474_47465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_three_l474_47401

/-- Represents a standard six-sided die --/
structure Die where
  faces : Fin 6 → Nat
  valid : ∀ i, faces i ∈ ({1, 2, 3, 4, 5, 6} : Set Nat)
  sum_21 : (Finset.sum (Finset.range 6) (λ i => faces i)) = 21

/-- Represents a throw of the die --/
structure Throw where
  lateral_sum : Nat

/-- Theorem stating the property of the opposite face of three --/
theorem opposite_face_of_three (d : Die) 
  (t1 : Throw) (t2 : Throw)
  (h1 : t1.lateral_sum = 12)
  (h2 : t2.lateral_sum = 15)
  (h3 : ∃ i, d.faces i = 3) :
  ∃ j, d.faces j = 6 ∧ (∀ k, d.faces k = 3 → 
    (k.val + j.val) % 6 = 5 ∨ (k.val + j.val) % 6 = 11) := by
  sorry

#check opposite_face_of_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_three_l474_47401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l474_47421

/-- Represents a hyperbola with parameters a, b, and c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c^2 = a^2 + b^2

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- The slope of the asymptote of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

/-- The slope of the line from focus to imaginary axis end -/
noncomputable def focus_to_imaginary_axis_slope (h : Hyperbola) : ℝ := h.b / h.c

/-- The theorem stating the eccentricity of the hyperbola under given conditions -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_perpendicular : focus_to_imaginary_axis_slope h * asymptote_slope h = -1) :
  eccentricity h = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l474_47421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monic_quadratic_with_complex_root_l474_47436

/-- A monic quadratic polynomial with real coefficients that has 2 - i as a root -/
def target_polynomial (x : ℂ) : ℂ := x^2 - 4*x + 5

/-- Proves that the target polynomial is the unique monic quadratic polynomial 
    with real coefficients that has 2 - i as a root -/
theorem unique_monic_quadratic_with_complex_root :
  ∀ (p : ℂ → ℂ),
    (∃ a b : ℝ, ∀ x : ℂ, p x = x^2 + a*x + b) →  -- p is monic quadratic with real coefficients
    p (Complex.mk 2 (-1)) = 0 →                  -- 2 - i is a root
    ∀ x, p x = target_polynomial x :=            -- p is equal to the target polynomial
by sorry

#check unique_monic_quadratic_with_complex_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monic_quadratic_with_complex_root_l474_47436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_path_varies_l474_47451

-- Define the circle and points
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

def A : ℝ × ℝ := (-10, 0)
def B : ℝ × ℝ := (10, 0)
def O : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (-4, 0)
def D : ℝ × ℝ := (4, 0)

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem broken_line_path_varies :
  ∃ P₁ P₂ : ℝ × ℝ, P₁ ∈ Circle O 10 ∧ P₂ ∈ Circle O 10 ∧
  distance C P₁ + distance P₁ D ≠ distance C P₂ + distance P₂ D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_path_varies_l474_47451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_works_ten_hours_per_day_l474_47497

/-- Jerry's work schedule and earnings --/
structure JerryWork where
  pay_per_task : ℚ
  hours_per_task : ℚ
  days_per_week : ℕ
  weekly_earnings : ℚ

/-- Calculate the number of hours Jerry works per day --/
def hours_per_day (j : JerryWork) : ℚ :=
  (j.weekly_earnings / j.pay_per_task * j.hours_per_task) / j.days_per_week

/-- Theorem: Given Jerry's work conditions, he works 10 hours per day --/
theorem jerry_works_ten_hours_per_day :
  let j : JerryWork := {
    pay_per_task := 40,
    hours_per_task := 2,
    days_per_week := 7,
    weekly_earnings := 1400
  }
  hours_per_day j = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_works_ten_hours_per_day_l474_47497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_point_m_l474_47458

-- Define the parabola and point M
def Parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = -2*p*x ∧ p > 0

def PointM (x y : ℝ) : Prop := x = -9

-- Define the distance from a point to the focus
noncomputable def DistanceToFocus (p x y : ℝ) : ℝ := Real.sqrt ((x + p/2)^2 + y^2)

theorem parabola_equation_and_point_m 
  (p : ℝ) (x y : ℝ) 
  (h1 : Parabola p x y) 
  (h2 : PointM x y) 
  (h3 : DistanceToFocus p x y = 10) : 
  (y^2 = -4*x) ∧ (y = 6 ∨ y = -6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_point_m_l474_47458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_jogging_speed_l474_47432

/-- Represents a track with oval ends -/
structure OvalTrack where
  straightLength : ℝ
  innerMinorAxis : ℝ
  width : ℝ
  timeDifference : ℝ

/-- Calculates the perimeter of an oval track -/
noncomputable def trackPerimeter (t : OvalTrack) (isOuter : Bool) : ℝ :=
  let b := if isOuter then t.innerMinorAxis + t.width else t.innerMinorAxis
  2 * t.straightLength + 2 * Real.pi * Real.sqrt ((4 * b ^ 2 + b ^ 2) / 2)

/-- Calculates the jogging speed given a track -/
noncomputable def joggingSpeed (t : OvalTrack) : ℝ :=
  (trackPerimeter t true - trackPerimeter t false) / t.timeDifference

theorem amy_jogging_speed (t : OvalTrack) 
  (h1 : t.width = 4)
  (h2 : t.timeDifference = 48) :
  joggingSpeed t = Real.pi / 2 := by
  sorry

#check amy_jogging_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_jogging_speed_l474_47432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_k_range_l474_47456

def P (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k}

def Q (a : ℝ) : Set (ℝ × ℝ) := {q : ℝ × ℝ | q.2 = a^q.1 + 1 ∧ a > 0 ∧ a ≠ 1}

theorem intersection_implies_k_range (k : ℝ) :
  (∃ a : ℝ, ∃! x : ℝ, (x, k) ∈ P k ∩ Q a) → k ∈ Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_k_range_l474_47456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_eq_sin_l474_47438

open Real

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => cos
  | n + 1 => deriv (f n)

theorem f_2011_eq_sin : f 2011 = sin := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_eq_sin_l474_47438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_property_l474_47403

/-- Sequence definition for aₖ -/
def a : ℕ → ℤ
  | 0 => 2
  | n + 1 => 2 * (a n)^2 - 1

theorem prime_divisor_property (p n : ℕ) :
  Nat.Prime p →
  Odd p →
  (p : ℤ) ∣ a n →
  2^(n + 3) ∣ p^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_property_l474_47403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_equality_l474_47402

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {1, 3, 4}
def C : Set ℝ := {x | x^2 - 3*x + 2 > 0}

theorem set_intersection_equality :
  (A ∪ B) ∩ C = {0, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_equality_l474_47402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_investment_rate_l474_47431

/-- Proof of the annual simple interest rate for the second investment --/
theorem second_investment_rate 
  (total_income : ℝ) 
  (first_rate : ℝ) 
  (total_invested : ℝ) 
  (first_invested : ℝ) 
  (h1 : total_income = 575)
  (h2 : first_rate = 0.085)
  (h3 : total_invested = 8000)
  (h4 : first_invested = 3000) :
  (let second_invested := total_invested - first_invested
   let first_income := first_invested * first_rate
   let second_income := total_income - first_income
   let second_rate := second_income / second_invested
   second_rate) = 0.064 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_investment_rate_l474_47431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l474_47417

noncomputable section

-- Define the cones and marbles
structure Cone where
  radius : ℝ
  height : ℝ

structure Marble where
  radius : ℝ

-- Define the problem setup
def narrowCone : Cone := { radius := 4, height := 1 }
def wideCone : Cone := { radius := 8, height := 1 }
def narrowMarble : Marble := { radius := 2 }
def wideMarble : Marble := { radius := 1 }

-- Function to calculate the volume of a cone
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

-- Function to calculate the volume of a sphere (marble)
noncomputable def sphereVolume (m : Marble) : ℝ := (4/3) * Real.pi * m.radius^3

-- Function to calculate the height increase of liquid in a cone after adding a marble
noncomputable def heightIncrease (c : Cone) (m : Marble) : ℝ :=
  (sphereVolume m) / (Real.pi * c.radius^2)

-- Theorem statement
theorem liquid_rise_ratio :
  (heightIncrease narrowCone narrowMarble) / (heightIncrease wideCone wideMarble) = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l474_47417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_line_l474_47488

-- Define the original line l1
def l1 (x y : ℝ) : Prop := x - y - 3 = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (3, 0)

-- Define the rotation angle in radians
noncomputable def rotation_angle : ℝ := 15 * (Real.pi / 180)

-- Define the rotated line l2
def l2 (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 3 * Real.sqrt 3 = 0

-- Theorem statement
theorem rotate_line :
  ∀ (x y : ℝ),
  l1 x y ∧ 
  (x - fixed_point.1) * Real.cos rotation_angle - (y - fixed_point.2) * Real.sin rotation_angle = x - fixed_point.1 ∧
  (x - fixed_point.1) * Real.sin rotation_angle + (y - fixed_point.2) * Real.cos rotation_angle = y - fixed_point.2
  → l2 x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_line_l474_47488
