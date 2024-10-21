import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_gain_is_10_over_3_l438_43868

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- The probability of getting heads on a single flip -/
noncomputable def prob_heads : ℝ := 2/3

/-- The probability of getting tails on a single flip -/
noncomputable def prob_tails : ℝ := 1/3

/-- The amount gained from flipping heads -/
def gain_heads : ℝ := 5

/-- The amount lost from flipping two consecutive tails -/
def loss_two_tails : ℝ := 15

/-- The gain from two coin flips -/
def gain (flip1 flip2 : CoinFlip) : ℝ :=
  match flip1, flip2 with
  | CoinFlip.Heads, CoinFlip.Heads => 2 * gain_heads
  | CoinFlip.Heads, CoinFlip.Tails => gain_heads
  | CoinFlip.Tails, CoinFlip.Heads => gain_heads
  | CoinFlip.Tails, CoinFlip.Tails => -loss_two_tails

/-- The expected value of Harris's gain after two independent coin flips -/
noncomputable def expected_gain : ℝ :=
  (prob_heads * prob_heads * gain CoinFlip.Heads CoinFlip.Heads) +
  (prob_heads * prob_tails * gain CoinFlip.Heads CoinFlip.Tails) +
  (prob_tails * prob_heads * gain CoinFlip.Tails CoinFlip.Heads) +
  (prob_tails * prob_tails * gain CoinFlip.Tails CoinFlip.Tails)

theorem expected_gain_is_10_over_3 : expected_gain = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_gain_is_10_over_3_l438_43868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l438_43831

/-- The volume of a cone with given slant height and central angle of unwrapped side. -/
noncomputable def cone_volume (slant_height : ℝ) (central_angle : ℝ) : ℝ :=
  let radius := (slant_height * central_angle) / (2 * Real.pi)
  let height := Real.sqrt (slant_height^2 - radius^2)
  (1/3) * Real.pi * radius^2 * height

/-- Theorem stating that the volume of a cone with slant height 4 and central angle 2π/3 is (128√2)/81 π -/
theorem cone_volume_specific : 
  cone_volume 4 ((2/3) * Real.pi) = (128 * Real.sqrt 2) / 81 * Real.pi := by
  sorry

#check cone_volume_specific

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l438_43831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l438_43886

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (m - 2) + y^2 / (m - 5) = 1) → (m - 2) * (m - 5) < 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x > 0 → x^2 - m*x + 4 ≥ 0

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) ∧ (p m ∧ q m) → m ∈ Set.Iic 2 ∪ Set.Ioo 4 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l438_43886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_probability_l438_43894

/-- The number of different types of cards -/
def num_card_types : ℕ := 3

/-- The number of books bought -/
def num_books : ℕ := 5

/-- The probability of winning the prize -/
def win_probability : ℚ := 50/81

/-- Theorem stating the probability of winning the prize -/
theorem prize_probability :
  (((num_card_types^num_books : ℕ) - (Nat.choose num_card_types 2 * 2^num_books - num_card_types) : ℚ) / 
   (num_card_types^num_books)) = win_probability :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_probability_l438_43894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l438_43843

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- Define the sequence a_n
noncomputable def a (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => y
  | n + 2 => (a x y (n + 1) * a x y n + 1) / (a x y (n + 1) + a x y n)

-- State the theorem
theorem sequence_properties (x y : ℝ) :
  (∃ n₀ : ℕ, ∀ n ≥ n₀, a x y n = a x y n₀) ↔ (|y| = 1 ∧ x ≠ -y) ∧
  ∀ n : ℕ, a x y n = 
    ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) + (x - 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1))) /
    ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) - (x - 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l438_43843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l438_43844

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then (x - 1)^2 else 2/x

-- Define the monotonicity properties
def increasing_on_left_interval : Prop :=
  ∀ x ∈ Set.Icc 1 2, x < 2 → MonotoneOn f (Set.Icc 1 x)

def decreasing_on_right_interval : Prop :=
  MonotoneOn (fun x => -f x) (Set.Ici 2)

-- Theorem statement
theorem f_increasing_interval :
  ∃ a b, Set.Icc a b = {x | ∀ y, x ≤ y → f x ≤ f y} ∧ a = 1 ∧ b = 2 := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma f_continuous : ContinuousOn f (Set.Icc 1 2) := by
  sorry

lemma f_differentiable : DifferentiableOn ℝ f (Set.Icc 1 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l438_43844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_n_l438_43838

theorem not_divisible_by_n (n : ℕ) (h : n ≥ 2) :
  ∃ a b : ℤ, ∀ m : ℤ, ¬(n : ℤ) ∣ (m^3 + a*m + b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_n_l438_43838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_perpendicular_to_countless_lines_not_sufficient_l438_43819

structure Plane where
  -- Define a plane (placeholder)
  dummy : Unit

structure Line where
  -- Define a line (placeholder)
  dummy : Unit

def perpendicular (l : Line) (p : Plane) : Prop :=
  -- Define what it means for a line to be perpendicular to a plane (placeholder)
  True

def perpendicular_to_countless_lines (l : Line) (p : Plane) : Prop :=
  -- Define what it means for a line to be perpendicular to countless lines in a plane (placeholder)
  True

theorem perpendicular_implies_perpendicular_to_countless_lines
  (a : Line) (α : Plane) :
  perpendicular a α → perpendicular_to_countless_lines a α :=
by
  sorry

theorem not_sufficient
  (a : Line) (α : Plane) :
  ∃ a α, perpendicular_to_countless_lines a α ∧ ¬perpendicular a α :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_perpendicular_to_countless_lines_not_sufficient_l438_43819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_l438_43852

open Matrix

variable {n : ℕ}
variable (A B : Matrix (Fin n) (Fin n) ℝ)

theorem det_product (h1 : Matrix.det A = 2) (h2 : Matrix.det B = 12) : Matrix.det (A * B) = 24 := by
  have h3 : Matrix.det (A * B) = Matrix.det A * Matrix.det B := by exact Matrix.det_mul A B
  rw [h3, h1, h2]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_l438_43852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_circumscribed_circle_l438_43817

/-- Given a sector with central angle φ cut from a circle of radius 8,
    the radius of the circle circumscribed about the sector is 8 sec(φ/2). -/
theorem radius_of_circumscribed_circle (φ : Real) :
  let r : Real := 8  -- radius of the original circle
  let R : Real := r / Real.cos (φ / 2)  -- radius of the circumscribed circle
  R = 8 / Real.cos (φ / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_circumscribed_circle_l438_43817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l438_43873

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (x : ℝ) := Real.exp (-x)

theorem intersection_properties (k : ℝ) (h1 : k > 0) : 
  let n := f k
  let m := g k
  n < 2 * m →
  (n + m < (3 * Real.sqrt 2) / 2) ∧
  (n - m < Real.sqrt 2 / 2) ∧
  (n^(m + 1) < (m + 1)^n) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l438_43873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_individual_is_09_l438_43898

/-- Represents a random number table --/
def RandomNumberTable : Type := List (List Nat)

/-- The given random number table --/
def givenTable : RandomNumberTable :=
  [[6667, 4067, 1464, 0571, 9586, 1105, 6509, 6876, 8320, 3790],
   [5716, 0011, 6614, 9084, 4511, 7573, 8805, 9052, 2741, 1486]]

/-- Selects valid numbers from a list based on population size --/
def selectValidNumbers (numbers : List Nat) (populationSize : Nat) : List Nat :=
  numbers.filter (λ n => n > 0 && n ≤ populationSize)

/-- Selects unique numbers from a list --/
def selectUniqueNumbers (numbers : List Nat) : List Nat :=
  numbers.foldl (λ acc n => if n ∉ acc then acc ++ [n] else acc) []

/-- Theorem: The 4th individual selected is 09 --/
theorem fourth_individual_is_09 (populationSize : Nat) (startRow startCol : Nat) :
  populationSize = 50 →
  startRow = 0 →
  startCol = 8 →
  let flattenedTable := givenTable.join
  let validNumbers := selectValidNumbers flattenedTable populationSize
  let uniqueNumbers := selectUniqueNumbers validNumbers
  uniqueNumbers.get? 3 = some 9 := by
  intro h_pop h_row h_col
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_individual_is_09_l438_43898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l438_43892

theorem rationalize_denominator : 
  1 / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l438_43892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_with_distance_l438_43872

/-- The equation of a line passing through the intersection of two given lines
    and having a specified distance from a given point. -/
theorem line_through_intersection_with_distance :
  ∃ (l : Set (ℝ × ℝ)),
    (∀ p : ℝ × ℝ, p ∈ {p | p.1 - 2*p.2 + 3 = 0 ∧ 2*p.1 + 3*p.2 - 8 = 0} → p ∈ l) ∧
    (|(1 : ℝ) * 0 + 0 * 4 + (-1)| / Real.sqrt (1^2 + 0^2) = 1 ∨
     |3 * 0 + 4 * 4 + (-11)| / Real.sqrt (3^2 + 4^2) = 1) ∧
    (l = {p : ℝ × ℝ | p.1 = 1} ∨ l = {p : ℝ × ℝ | 3*p.1 + 4*p.2 - 11 = 0}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_with_distance_l438_43872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_direction_movement_l438_43895

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a wheel of a train -/
structure Wheel where
  center : Point
  radius : ℝ

/-- Represents the motion of a train -/
def TrainMotion := ℝ → Wheel

/-- A function that calculates the position of a point on the wheel's flange -/
noncomputable def flangePointPosition (w : Wheel) (t : ℝ) : Point :=
  { x := w.center.x + w.radius * Real.cos t
    y := w.center.y + w.radius * Real.sin t }

/-- Theorem stating that there exists a point on the wheel moving in the opposite direction -/
theorem opposite_direction_movement 
  (train : TrainMotion) 
  (train_direction : ℝ) 
  (h_positive : train_direction > 0) :
  ∃ (t₁ t₂ : ℝ) (ε : ℝ), 
    t₁ < t₂ ∧ 
    ε > 0 ∧
    (flangePointPosition (train t₂) t₂).x - (flangePointPosition (train t₁) t₁).x < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_direction_movement_l438_43895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_properties_l438_43891

/-- Sphere properties -/
structure Sphere where
  radius : ℝ

/-- Surface area of a sphere -/
noncomputable def surfaceArea (s : Sphere) : ℝ := 4 * Real.pi * s.radius ^ 2

/-- Volume of a sphere -/
noncomputable def volume (s : Sphere) : ℝ := (4 / 3) * Real.pi * s.radius ^ 3

theorem sphere_properties :
  (∃ s : Sphere, surfaceArea s = 64 * Real.pi ∧ volume s = (256 / 3) * Real.pi) ∧
  (∃ s : Sphere, volume s = (500 / 3) * Real.pi ∧ surfaceArea s = 100 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_properties_l438_43891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_function_properties_l438_43820

/-- Definition of an inequality function on [0, 1] -/
def IsInequalityFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

/-- The function g(x) = x^3 -/
def g (x : ℝ) : ℝ := x^3

/-- The function h(x) = 2^x - a -/
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := Real.exp (x * Real.log 2) - a

theorem inequality_function_properties :
  (IsInequalityFunction g) ∧
  (∀ a : ℝ, IsInequalityFunction (h a) ↔ a = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_function_properties_l438_43820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_x_eq_2_parallel_implies_x_eq_neg_10_l438_43809

-- Define the vectors a and b
def a : Fin 3 → ℝ := ![2, -1, 5]
def b (x : ℝ) : Fin 3 → ℝ := ![-4, 2, x]

-- Define dot product
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Define perpendicularity condition
def isPerpendicular (v w : Fin 3 → ℝ) : Prop :=
  dot_product v w = 0

-- Define parallelism condition
def isParallel (v w : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i : Fin 3, v i = k * w i)

-- Theorem for perpendicular case
theorem perpendicular_implies_x_eq_2 :
  isPerpendicular a (b 2) :=
by
  unfold isPerpendicular dot_product a b
  simp
  norm_num

-- Theorem for parallel case
theorem parallel_implies_x_eq_neg_10 :
  isParallel a (b (-10)) :=
by
  unfold isParallel a b
  use (-1/2)
  constructor
  · norm_num
  · intro i
    fin_cases i <;> simp <;> norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_x_eq_2_parallel_implies_x_eq_neg_10_l438_43809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_bounded_l438_43824

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => Real.sqrt (1 + a n)

theorem a_bounded (n : ℕ) : 1 < a n ∧ a n < 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_bounded_l438_43824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l438_43832

/-- Represents a hyperbola with center at the origin and foci on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Condition that an asymptote of the hyperbola passes through the point (4, 2) -/
def asymptote_condition (h : Hyperbola) : Prop :=
  4^2 / h.a^2 - 2^2 / h.b^2 = 1

theorem hyperbola_eccentricity (h : Hyperbola) (asymptote : asymptote_condition h) :
  eccentricity h = Real.sqrt (7/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l438_43832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l438_43875

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 2, 5}

theorem complement_A_intersect_B :
  ((U \ A) ∩ B) = {1, 5} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l438_43875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_set_l438_43810

def sequence_a : ℕ → ℝ := sorry

def sum_S : ℕ → ℝ := sorry

axiom sum_relation : ∀ n : ℕ, sum_S n = 2 * sequence_a n - 1

theorem sequence_inequality_set :
  {n : ℕ | n > 0 ∧ sequence_a n / n ≤ 2} = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_set_l438_43810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annas_deducted_salary_l438_43808

/-- Calculates the deducted salary given the weekly salary, workdays per week, and days absent. -/
def deducted_salary (weekly_salary : ℚ) (workdays_per_week : ℚ) (days_absent : ℚ) : ℚ :=
  weekly_salary - (weekly_salary / workdays_per_week) * days_absent

/-- Proves that Anna's deducted salary is correct given the problem conditions. -/
theorem annas_deducted_salary :
  deducted_salary 1379 5 2 = 827.40 := by
  sorry

#eval deducted_salary 1379 5 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annas_deducted_salary_l438_43808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l438_43862

noncomputable def f (x : ℝ) : ℝ := (x + 6) / Real.sqrt (x^2 - 3*x - 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -1 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l438_43862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_symmetry_and_monotonicity_l438_43882

theorem sine_function_symmetry_and_monotonicity (ω φ : ℝ) (h1 : ω > 0) (h2 : 0 < φ) (h3 : φ ≤ π / 2) : 
  (∀ x : ℝ, Real.sin (ω * x + φ) = Real.sin (ω * (π / 3 - x) + φ)) →  -- symmetry condition
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π / 6 → Real.sin (ω * x + φ) > Real.sin (ω * y + φ)) →  -- monotonicity condition
  ω = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_symmetry_and_monotonicity_l438_43882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_half_l438_43848

theorem divisor_sum_half (a b n : ℕ) : 
  (a ∣ n) → (b ∣ n) → a + b = n / 2 → 
  (∃ k : ℕ, k > 0 ∧ ((a = k ∧ b = 2*k ∧ n = 6*k) ∨ 
                     (a = 2*k ∧ b = k ∧ n = 6*k) ∨ 
                     (a = k ∧ b = k ∧ n = 4*k))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_half_l438_43848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_price_l438_43857

/-- Given a good sold for 6400 after successive discounts of 20%, 10%, and 5%, 
    its original price was approximately 9356.73 -/
theorem discounted_price (original_price : ℝ) : 
  (((original_price * (1 - 0.2)) * (1 - 0.1)) * (1 - 0.05) = 6400) → 
  (abs (original_price - 9356.73) < 0.01) :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_price_l438_43857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l438_43811

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

-- State the theorem
theorem even_function_implies_a_equals_two (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l438_43811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axisymmetric_iff_foldable_coincidable_l438_43835

/-- A shape that can be folded along a straight line. -/
structure FoldableShape :=
  (canFoldAlongLine : Bool)

/-- A shape where parts on both sides of a line can coincide. -/
structure CoincidableParts :=
  (partsCoincide : Bool)

/-- An axisymmetric shape. -/
structure AxisymmetricShape :=
  (isAxisymmetric : Bool)

/-- A shape that is both foldable along a line and has coincidable parts. -/
structure FoldableCoincidableShape extends FoldableShape, CoincidableParts

/-- Theorem: A shape is axisymmetric if and only if it can be folded along a straight line
    such that the parts on both sides of the line coincide with each other. -/
theorem axisymmetric_iff_foldable_coincidable (shape : FoldableCoincidableShape) :
  (∃ a : AxisymmetricShape, a.isAxisymmetric) ↔ (shape.canFoldAlongLine ∧ shape.partsCoincide) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axisymmetric_iff_foldable_coincidable_l438_43835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_eight_percent_l438_43821

/-- Calculate the tax rate given the total value of goods, tax-free limit, and tax paid -/
noncomputable def calculate_tax_rate (total_value : ℝ) (tax_free_limit : ℝ) (tax_paid : ℝ) : ℝ :=
  (tax_paid / (total_value - tax_free_limit)) * 100

/-- The tax rate on the portion of the total value in excess of $500 is 8% -/
theorem tax_rate_is_eight_percent :
  let total_value : ℝ := 730
  let tax_free_limit : ℝ := 500
  let tax_paid : ℝ := 18.40
  calculate_tax_rate total_value tax_free_limit tax_paid = 8 := by
  -- Unfold the definition of calculate_tax_rate
  unfold calculate_tax_rate
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_eight_percent_l438_43821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l438_43815

noncomputable section

-- Define the points A, B, and C
def A (y : ℝ) : ℝ × ℝ := (-2, y)
def B (y : ℝ) : ℝ × ℝ := (0, y/2)
def C (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define vectors AB and BC
def vecAB (y : ℝ) : ℝ × ℝ := (2, -y/2)
def vecBC (x y : ℝ) : ℝ × ℝ := (x, y/2)

-- Define the dot product of two 2D vectors
def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the perpendicularity condition
def arePerp (v w : ℝ × ℝ) : Prop := dotProduct v w = 0

-- State the theorem
theorem trajectory_equation (x y : ℝ) (h : x ≠ 0) :
  arePerp (vecAB y) (vecBC x y) → y^2 = 8*x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l438_43815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l438_43861

theorem triangle_ratio (A B C a b c : Real) :
  A > 0 → B > 0 → C > 0 → A + B + C = Real.pi →
  a > 0 → b > 0 → c > 0 →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  (2 * a + c) * Real.cos B + b * Real.cos C = 0 →
  1/2 * a * c * Real.sin B = 15 * Real.sqrt 3 →
  a + b + c = 30 →
  Real.sin (2*B) / (Real.sin A + Real.sin C) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l438_43861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_Q_l438_43849

def Q : Set ℕ := {x : ℕ | 2 * x^2 - 5 * x ≤ 0}

theorem number_of_subsets_Q : Finset.card (Finset.powerset (Finset.filter (λ x => 2 * x^2 - 5 * x ≤ 0) (Finset.range 3))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_Q_l438_43849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l438_43814

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the state of the coin flipping process -/
inductive FlipState
| Start
| OneTail
| TwoTails
| OneHeadAfterTwoTails

/-- Represents the probability of transitioning between states -/
def transition_prob : FlipState → FlipState → ℚ
| FlipState.Start, FlipState.OneTail => 1/2
| FlipState.OneTail, FlipState.TwoTails => 1/2
| FlipState.TwoTails, FlipState.OneHeadAfterTwoTails => 1/2
| FlipState.OneHeadAfterTwoTails, FlipState.OneHeadAfterTwoTails => 1/2
| _, _ => 0

/-- The probability of reaching the desired outcome -/
def prob_two_heads_after_two_tails : ℚ := 1/24

theorem coin_flip_probability :
  prob_two_heads_after_two_tails =
    transition_prob FlipState.Start FlipState.OneTail *
    transition_prob FlipState.OneTail FlipState.TwoTails *
    transition_prob FlipState.TwoTails FlipState.OneHeadAfterTwoTails *
    transition_prob FlipState.OneHeadAfterTwoTails FlipState.OneHeadAfterTwoTails :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l438_43814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_manipulation_problem_l438_43823

theorem fraction_manipulation_problem :
  ∃! (x y : ℚ),
    (11 : ℚ) + x / (41 + x) = 3/8 ∧
    (37 : ℚ) - y / (63 + y) = 3/17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_manipulation_problem_l438_43823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_analytic_function_theorem_l438_43887

-- Define the complex plane
variable (z : ℂ)

-- Define the function f
variable (f : ℂ → ℂ)

-- Define the real and imaginary parts of z
variable (x y : ℝ)

-- Define the real part of f
noncomputable def u (x y : ℝ) : ℝ := 2 * Real.exp x * Real.cos y

-- State the theorem
theorem analytic_function_theorem (h1 : ∀ z : ℂ, (f z).re = u z.re z.im) 
  (h2 : f 0 = 2) : 
  ∀ z : ℂ, f z = 2 * Complex.exp z := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_analytic_function_theorem_l438_43887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_inequality_l438_43880

theorem sin_squared_sum_inequality (α β : Real) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : 0 < β ∧ β < Real.pi / 2) 
  (h3 : Real.sin α ^ 2 + Real.sin β ^ 2 < 1) : Real.sin α ^ 2 + Real.sin β ^ 2 < Real.sin (α + β) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_inequality_l438_43880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l438_43830

theorem tan_half_sum (a b : ℝ) 
  (h1 : Real.cos a + Real.cos b = 3/5) 
  (h2 : Real.sin a + Real.sin b = 1/2) 
  (h3 : Real.tan a * Real.tan b = 1) :
  Real.tan ((a + b) / 2) = 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l438_43830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_duration_l438_43816

-- Define the usual speed and total distance
variable (x : ℝ) -- usual speed
variable (z : ℝ) -- total distance

-- Define the delay time in hours
noncomputable def delay : ℝ := 1 + 42 / 60

-- Define the theorem
theorem train_journey_duration :
  (0.9 * z / (1.2 * x) + 0.1 * z / (1.25 * x) = z / x - delay) →
  z / x = 10 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_duration_l438_43816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l438_43806

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.sqrt (1 - x) + x

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l438_43806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l438_43874

def set_A : Set ℤ := {x | x^2 - 3*x - 4 ≤ 0}
def set_B : Set ℤ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l438_43874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_sets_not_cover_reals_l438_43890

/-- A set S is harmonious if it's a non-empty subset of real numbers and
    for any a, b ∈ S, both a + b ∈ S and a - b ∈ S -/
def IsHarmonious (S : Set ℝ) : Prop :=
  S.Nonempty ∧ ∀ a b, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a - b) ∈ S

theorem harmonious_sets_not_cover_reals :
  ∃ (S₁ S₂ : Set ℝ), IsHarmonious S₁ ∧ IsHarmonious S₂ ∧ S₁ ≠ S₂ ∧ (S₁ ∪ S₂ ≠ Set.univ) := by
  sorry

#check harmonious_sets_not_cover_reals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_sets_not_cover_reals_l438_43890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_circle_l438_43856

-- Define the fixed circle O
structure FixedCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point P
def Point : Type := ℝ × ℝ

-- Define the moving circle C
structure MovingCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the possible trajectories
inductive Trajectory
  | TwoRays
  | Circle
  | Ellipse

-- State the theorem
theorem trajectory_of_moving_circle (O : FixedCircle) (P : Point) (C : MovingCircle) :
  (P.1 - O.center.1)^2 + (P.2 - O.center.2)^2 ≤ O.radius^2 →  -- P is inside or on O
  (C.center.1 - P.1)^2 + (C.center.2 - P.2)^2 = C.radius^2 →  -- C passes through P
  (C.center.1 - O.center.1)^2 + (C.center.2 - O.center.2)^2 = (O.radius + C.radius)^2 →  -- C is tangent to O
  ∃ t : Trajectory, t ∈ ({Trajectory.TwoRays, Trajectory.Circle, Trajectory.Ellipse} : Set Trajectory) :=
by
  intro h1 h2 h3
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_circle_l438_43856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l438_43879

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 4 * x - 3 * y = -0.5
def equation2 (x y : ℝ) : Prop := 5 * x + 7 * y = 10.3

-- Define the approximate solution
def solution : ℝ × ℝ := (0.6372, 1.0163)

-- Theorem statement
theorem system_solution :
  ∃ (x y : ℝ), equation1 x y ∧ equation2 x y ∧
  (abs (x - solution.1) < 0.0001 ∧ abs (y - solution.2) < 0.0001) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l438_43879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_difference_is_twenty_percent_l438_43871

/-- The exchange rate from Japanese yen to US dollars -/
noncomputable def yen_to_dollar : ℚ := 8 / 1000

/-- Michael's money in US dollars -/
def michael_money : ℚ := 2000

/-- Haruto's money in Japanese yen -/
def haruto_money_yen : ℚ := 300000

/-- Haruto's money converted to US dollars -/
noncomputable def haruto_money_dollar : ℚ := haruto_money_yen * yen_to_dollar

/-- The percentage difference between Haruto's and Michael's money -/
noncomputable def percentage_difference : ℚ := (haruto_money_dollar - michael_money) / michael_money * 100

theorem money_difference_is_twenty_percent :
  percentage_difference = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_difference_is_twenty_percent_l438_43871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_line_chord_length_when_m_zero_l438_43807

/-- Circle C with center (1, 2) and radius 5 -/
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

/-- Line l with parameter m -/
def l (x y m : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y = 7*m + 4

/-- The fixed point that the line always passes through -/
def fixed_point : ℝ × ℝ := (3, 1)

/-- The length of the chord when m = 0 -/
noncomputable def chord_length : ℝ := 7 * Real.sqrt 2

theorem fixed_point_on_line :
  ∀ m : ℝ, l (fixed_point.1) (fixed_point.2) m := by sorry

theorem chord_length_when_m_zero :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  C x₁ y₁ ∧ C x₂ y₂ ∧ 
  l x₁ y₁ 0 ∧ l x₂ y₂ 0 ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_line_chord_length_when_m_zero_l438_43807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l438_43805

/-- A function that represents the puzzle's rule --/
def puzzleRule (n : ℕ) : ℕ := sorry

/-- The puzzle rule applied to 111 gives 9 --/
axiom rule_111 : puzzleRule 111 = 9

/-- The puzzle rule applied to 444 gives 12 --/
axiom rule_444 : puzzleRule 444 = 12

/-- The puzzle rule applied to 888 gives 15 --/
axiom rule_888 : puzzleRule 888 = 15

/-- Theorem: Given the puzzle rules for 111, 444, and 888, the rule applied to 777 gives 24 --/
theorem puzzle_solution : puzzleRule 777 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l438_43805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_value_in_rice_l438_43842

/-- Represents the value of one fish -/
def fish : ℚ := sorry

/-- Represents the value of one loaf of bread -/
def loaf : ℚ := sorry

/-- Represents the value of one bag of rice -/
def rice : ℚ := sorry

/-- Represents the value of one hat -/
def hat : ℚ := sorry

/-- Four fish can be traded for three loaves of bread -/
axiom trade_fish_bread : 4 * fish = 3 * loaf

/-- One loaf of bread can be traded for five bags of rice -/
axiom trade_bread_rice : loaf = 5 * rice

/-- Five fish can be exchanged for seven hats -/
axiom trade_fish_hat : 5 * fish = 7 * hat

/-- Theorem: One hat is worth 75/28 bags of rice -/
theorem hat_value_in_rice : hat = 75/28 * rice := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_value_in_rice_l438_43842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l438_43822

/-- Prove that a train crosses a bridge in approximately 17.28 seconds given the following conditions:
    - The train is 100 meters long
    - The bridge is 150 meters long
    - The initial speed of the train is 53.7 kmph
    - The train's speed decreases by 3% due to wind resistance
    - The train's speed may vary along the travel -/
theorem train_bridge_crossing_time :
  ∀ (train_length bridge_length : ℝ)
    (initial_speed : ℝ)
    (speed_decrease_percentage : ℝ),
  train_length = 100 →
  bridge_length = 150 →
  initial_speed = 53.7 →
  speed_decrease_percentage = 3 →
  ∃ (crossing_time : ℝ),
    (crossing_time ≥ 17.28 ∧ crossing_time ≤ 17.29) ∧
    crossing_time > 0 ∧
    crossing_time * (initial_speed * (1 - speed_decrease_percentage / 100) * 1000 / 3600) ≥
      train_length + bridge_length :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l438_43822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nacl_formation_l438_43834

/-- Represents the number of moles of a chemical substance -/
structure Moles where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents the reaction ratio between NaOH and HCl -/
def ReactionRatio : ℝ := 1

/-- 
Given the number of moles of NaOH and HCl, and assuming a 1:1 reaction ratio,
prove that the number of moles of NaCl formed is equal to the number of moles of NaOH (or HCl).
-/
theorem nacl_formation (naoh hcl : Moles) (h : naoh.value = hcl.value) : 
  let nacl := min naoh.value hcl.value
  nacl = naoh.value :=
by
  rw [h]
  simp [min]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nacl_formation_l438_43834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_0_002_l438_43867

-- Define the payroll threshold
noncomputable def payroll_threshold : ℝ := 200000

-- Define the total payroll
noncomputable def total_payroll : ℝ := 300000

-- Define the total tax paid
noncomputable def total_tax_paid : ℝ := 200

-- Define the taxable payroll
noncomputable def taxable_payroll : ℝ := total_payroll - payroll_threshold

-- Define the tax rate calculation
noncomputable def calculate_tax_rate (taxable : ℝ) (tax_paid : ℝ) : ℝ :=
  tax_paid / taxable

-- Theorem statement
theorem tax_rate_is_0_002 :
  calculate_tax_rate taxable_payroll total_tax_paid = 0.002 := by
  -- Unfold definitions
  unfold calculate_tax_rate taxable_payroll total_tax_paid total_payroll payroll_threshold
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_0_002_l438_43867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_walk_distance_l438_43841

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

theorem carmen_walk_distance :
  let start := Point.mk 0 0
  let turn := Point.mk 0 3
  let end_point := Point.mk (8 * Real.cos (π/4)) (3 + 8 * Real.sin (π/4))
  distance start end_point = Real.sqrt (73 + 24 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_walk_distance_l438_43841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_P_properties_l438_43828

/-- The equation of plane P: 3y + 2z = -1 -/
def plane_P (x y z : ℝ) : Prop := 3 * y + 2 * z = -1

/-- Line L: intersection of two planes -/
def line_L (x y z : ℝ) : Prop := x + 2*y + 3*z = 2 ∧ x - y + z = 3

/-- Distance function between a point and a plane -/
noncomputable def plane_point_distance (A B C D : ℝ) (x₀ y₀ z₀ : ℝ) : ℝ :=
  abs (A*x₀ + B*y₀ + C*z₀ + D) / Real.sqrt (A^2 + B^2 + C^2)

theorem plane_P_properties :
  -- P contains line L
  (∀ x y z, line_L x y z → plane_P x y z) ∧
  -- P is different from the two given planes
  (∃ x y z, plane_P x y z ∧ ¬(x + 2*y + 3*z = 2)) ∧
  (∃ x y z, plane_P x y z ∧ ¬(x - y + z = 3)) ∧
  -- P has a distance of 3/√6 from point (1,0,1)
  (plane_point_distance 0 3 2 (-1) 1 0 1 = 3 / Real.sqrt 6) ∧
  -- A, B, C, and D are integers (implicitly satisfied by the definition)
  -- A > 0 (satisfied as 3 > 0)
  -- gcd(|A|,|B|,|C|,|D|) = 1 (satisfied by 3, 2, 0, 1)
  True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_P_properties_l438_43828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l438_43859

/-- The number of bricks required to pave a rectangular courtyard -/
def bricks_required (courtyard_length courtyard_width brick_length brick_width : ℚ) : ℕ :=
  ⌊(courtyard_length * 100 * courtyard_width * 100) / (brick_length * brick_width)⌋.toNat

/-- Theorem stating the number of bricks required for the given courtyard and brick dimensions -/
theorem courtyard_paving :
  bricks_required 30 16 20 10 = 24000 := by
  -- Proof goes here
  sorry

#eval bricks_required 30 16 20 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l438_43859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_r_eq_pq_l438_43854

-- Define the curve
noncomputable def curve (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q * x^2) / (r * x + s)

-- Define the symmetry condition
def is_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y → f (2 * y) = y

-- Theorem statement
theorem symmetry_implies_r_eq_pq (p q r s : ℝ) :
  is_symmetric (curve p q r s) → r = p * q :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_r_eq_pq_l438_43854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soil_erosion_occurs_no_soil_erosion_before_l438_43802

/-- Represents the timber stock model with given parameters -/
structure TimberModel where
  a : ℝ  -- Initial stock
  b : ℝ  -- Annual harvest
  n : ℕ  -- Number of years

/-- Calculates the timber stock after n years -/
noncomputable def stock_after_n_years (model : TimberModel) : ℝ :=
  (5/4)^model.n * model.a - 4 * ((5/4)^model.n - 1) * model.b

/-- Theorem stating when soil erosion occurs -/
theorem soil_erosion_occurs (model : TimberModel) :
  model.b = 19 * model.a / 72 →
  model.n = 8 →
  stock_after_n_years model < 7/9 * model.a :=
by
  sorry

/-- Theorem stating that soil erosion does not occur before 8 years -/
theorem no_soil_erosion_before (model : TimberModel) :
  model.b = 19 * model.a / 72 →
  model.n < 8 →
  stock_after_n_years model ≥ 7/9 * model.a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soil_erosion_occurs_no_soil_erosion_before_l438_43802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l438_43813

/-- Calculates the speed of a train in km/h given its length in meters and time to pass a fixed point in seconds. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem: A train of length 170 meters that takes 9 seconds to pass an electric pole has a speed of 68 km/h. -/
theorem train_speed_calculation :
  let length : ℝ := 170
  let time : ℝ := 9
  train_speed length time = 68 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l438_43813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_polynomial_roots_l438_43864

-- Define a polynomial with non-negative coefficients
def NonNegativePolynomial (n : ℕ) := {p : Polynomial ℝ // ∀ i, 0 ≤ p.coeff i ∧ p.degree = n}

-- Define a triangle
def IsTriangle (a b c : ℝ) : Prop := 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem triangle_polynomial_roots {n : ℕ} (hn : 2 ≤ n) (P : NonNegativePolynomial n) (a b c : ℝ) 
  (h_triangle : IsTriangle a b c) : 
  IsTriangle ((P.val.eval a) ^ (1 / n : ℝ)) ((P.val.eval b) ^ (1 / n : ℝ)) ((P.val.eval c) ^ (1 / n : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_polynomial_roots_l438_43864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distribution_maximizes_triangles_l438_43845

/-- Represents a distribution of points into groups -/
def Distribution := List Nat

/-- Calculates the total number of triangles formed by the given distribution -/
def triangleCount (d : Distribution) : Nat :=
  d.foldl (fun acc x => acc + x * (d.sum - x)) 0 / 2

/-- The optimal distribution of 1989 points into 30 groups -/
def optimalDistribution : Distribution :=
  List.range 31 |>.map (· + 51)

theorem optimal_distribution_maximizes_triangles :
  ∀ d : Distribution,
    d.length = 30 ∧
    d.sum = 1989 ∧
    d.Nodup ∧
    (∀ x ∈ d.toFinset, x ≤ 81 ∧ x ≥ 51) →
    triangleCount d ≤ triangleCount optimalDistribution := by
  sorry

#eval triangleCount optimalDistribution
#eval optimalDistribution.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distribution_maximizes_triangles_l438_43845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_bottom_vertex_is_one_thirtysixth_l438_43836

structure Cube where
  vertices : Fin 8
  adjacentVertices : Fin 8 → List (Fin 8)
  topVertex : Fin 8
  bottomVertex : Fin 8

/-- A random walk on a cube -/
def randomWalk (cube : Cube) : Fin 8 → ℝ :=
  sorry

/-- The probability of ending at the bottom vertex after three random moves -/
def probabilityBottomVertex (cube : Cube) : ℚ :=
  sorry

theorem probability_bottom_vertex_is_one_thirtysixth (cube : Cube) :
  probabilityBottomVertex cube = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_bottom_vertex_is_one_thirtysixth_l438_43836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_incircle_segments_l438_43870

/-- Predicate stating that a set of points forms a right triangle -/
def IsRightTriangle (T : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate stating that a circle touches the hypotenuse of a triangle -/
def TouchesHypotenuse (T : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate stating that the incircle divides the hypotenuse into segments a and b -/
def DividesHypotenuse (T : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) (a b : ℝ) : Prop := sorry

/-- Function to calculate the area of a triangle -/
noncomputable def Area (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- Given a right triangle with incircle touching the hypotenuse at point P,
    dividing the hypotenuse into segments of length a and b,
    prove that the area of the triangle is ab. -/
theorem right_triangle_area_incircle_segments 
  (T : Set (ℝ × ℝ)) -- T represents the triangle
  (is_right_triangle : IsRightTriangle T)
  (incircle : Set (ℝ × ℝ))
  (touches_hypotenuse : TouchesHypotenuse T incircle)
  (a b : ℝ)
  (divides_hypotenuse : DividesHypotenuse T incircle a b) :
  Area T = a * b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_incircle_segments_l438_43870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_minimum_a_l438_43837

-- Part 1: Prove the inequality
theorem inequality_proof (x : ℝ) (h : x > 0) :
  1 - 1/x ≤ Real.log x ∧ Real.log x ≤ x - 1 := by sorry

-- Part 2: Find the range of values for a
noncomputable def f (a x : ℝ) := a * (1 - x^2) + x^2 * Real.log x

theorem minimum_a :
  (∀ x, 0 < x → x ≤ 1 → f (1/2) x ≥ 0) ∧
  ∀ ε > 0, ∃ x, 0 < x ∧ x ≤ 1 ∧ f ((1/2) - ε) x < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_minimum_a_l438_43837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l438_43883

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + m

theorem function_properties (m : ℝ) :
  (∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x m = 6) →
  (∀ k : ℤ, ∀ x ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (2 * Real.pi / 3 + k * Real.pi), 
    ∀ y ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (2 * Real.pi / 3 + k * Real.pi), 
    x ≤ y → f x m ≥ f y m) ∧
  (∀ x : ℝ, f x m ≤ 3 ↔ 
    ∃ k : ℤ, x ∈ Set.Icc (Real.pi / 2 + k * Real.pi) (5 * Real.pi / 6 + k * Real.pi)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l438_43883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_is_systematic_l438_43850

/-- Represents a sampling method -/
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom
  | RandomNumberTable

/-- Represents a class with students -/
structure ClassInfo where
  students : Finset Nat
  student_count : students.card = 50
  student_range : ∀ s ∈ students, 1 ≤ s ∧ s ≤ 50

/-- Represents a grade with multiple classes -/
structure Grade where
  classes : Finset ClassInfo
  class_count : classes.card = 20

/-- Represents the sampling process -/
def sample_homework (g : Grade) : Finset Nat :=
  g.classes.biUnion (λ c => c.students.filter (λ s => s ∈ [5, 15, 25, 35, 45]))

/-- The main theorem to prove -/
theorem sampling_is_systematic (g : Grade) :
  sample_homework g ≠ ∅ →
  ∃ k : Nat, ∀ s ∈ sample_homework g, ∃ n : Nat, s = k * n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_is_systematic_l438_43850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l438_43869

/-- The circle ⊙C in polar form -/
noncomputable def circleC (θ : Real) : Real := Real.cos θ + Real.sin θ

/-- The line l in polar form -/
noncomputable def lineL (θ : Real) : Real := 2 * Real.sqrt 2 / Real.cos (θ + Real.pi / 4)

/-- The minimum distance from any point on the circle to the line -/
noncomputable def min_distance : Real := 3 * Real.sqrt 2 / 4

theorem min_distance_circle_to_line :
  ∀ θ₁ θ₂ : Real,
  ∃ d : Real,
  d ≥ min_distance ∧
  d = Real.sqrt ((circleC θ₁ * Real.cos θ₁ - lineL θ₂ * Real.cos θ₂)^2 +
                 (circleC θ₁ * Real.sin θ₁ - lineL θ₂ * Real.sin θ₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l438_43869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_point_in_all_circles_l438_43804

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of a point being inside a circle
def isInside (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define the property that no circle contains the center of another
def noCircleContainsCenter (circles : List Circle) : Prop :=
  ∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → ¬(isInside c2.center c1)

-- Theorem statement
theorem no_point_in_all_circles (circles : List Circle) 
  (h1 : circles.length = 6)
  (h2 : noCircleContainsCenter circles) :
  ¬∃ p : ℝ × ℝ, ∀ c, c ∈ circles → isInside p c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_point_in_all_circles_l438_43804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l438_43825

/-- Given propositions p and q, prove that the range of x satisfies the given conditions. -/
theorem range_of_x (x : ℝ) : 
  (¬(Real.log (x^2 - 2*x - 2) ≥ 0) ∧ ¬(0 < x ∧ x < 4)) ∧
  (Real.log (x^2 - 2*x - 2) ≥ 0 ∨ (0 < x ∧ x < 4)) →
  x ≤ -1 ∨ (0 < x ∧ x < 3) ∨ x ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l438_43825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_shaded_areas_l438_43866

-- Define the square
def square_area : ℝ := 64

-- Define the side length of the square
noncomputable def side_length : ℝ := Real.sqrt square_area

-- Define the number of divisions after folding
def num_divisions : ℕ := 4

-- Define the area of one small square after division
noncomputable def small_square_area : ℝ := square_area / (num_divisions * num_divisions)

-- Define the area of one shaded rectangle
noncomputable def shaded_rectangle_area : ℝ := 2 * small_square_area

-- Theorem to prove
theorem sum_of_shaded_areas : 
  2 * shaded_rectangle_area = 16 := by
  -- Proof steps would go here
  sorry

#eval square_area
#eval num_divisions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_shaded_areas_l438_43866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_termite_ridden_homes_l438_43853

/-- The fraction of termite-ridden homes on Gotham Street -/
noncomputable def termite_ridden_fraction : ℝ := sorry

/-- The fraction of termite-ridden homes that are collapsing -/
noncomputable def collapsing_fraction : ℝ := 5/8

/-- The fraction of homes that are termite-ridden but not collapsing -/
noncomputable def termite_not_collapsing : ℝ := 0.125

theorem termite_ridden_homes :
  termite_ridden_fraction = 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_termite_ridden_homes_l438_43853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_in_still_water_l438_43827

/-- The rate of a man rowing in still water, given his downstream and upstream speeds -/
noncomputable def rate_in_still_water (downstream_speed upstream_speed : ℝ) : ℝ :=
  (downstream_speed + upstream_speed) / 2

/-- Theorem: Given a man who can row downstream at 26 kmph and upstream at 12 kmph,
    his rate in still water is 19 kmph -/
theorem mans_rate_in_still_water :
  rate_in_still_water 26 12 = 19 := by
  unfold rate_in_still_water
  norm_num

-- We can't use #eval with noncomputable definitions, so let's use #check instead
#check rate_in_still_water 26 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_in_still_water_l438_43827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elevator_time_approx_l438_43858

/-- Represents the time taken for the elevator to reach the bottom floor. -/
noncomputable def elevator_time : ℝ :=
  let total_floors : ℕ := 40
  let time_first_quarter : ℝ := 25
  let time_second_quarter : ℝ := 10 * 10
  let time_third_quarter : ℝ := 7 * 10
  let time_fourth_quarter_first_half : ℝ := 35
  let time_fourth_quarter_second_half : ℝ := 10 * 5
  let total_minutes : ℝ := time_first_quarter + time_second_quarter + time_third_quarter + 
                            time_fourth_quarter_first_half + time_fourth_quarter_second_half
  total_minutes / 60

/-- Theorem stating that the elevator time is approximately 4.67 hours. -/
theorem elevator_time_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |elevator_time - 4.67| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elevator_time_approx_l438_43858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sqrt2_minus_1_l438_43885

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1 / x

-- State the theorem
theorem f_sqrt2_minus_1 : f (Real.sqrt 2 - 1) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sqrt2_minus_1_l438_43885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_eq_half_final_result_l438_43877

/-- The custom operation ⋆ defined for non-zero real numbers -/
noncomputable def star (a b : ℝ) : ℝ := (3 * a) / (2 * b) * b / (3 * a)

/-- Theorem stating that a ⋆ b = 1/2 for any non-zero real numbers a and b -/
theorem star_eq_half (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : star a b = 1 / 2 := by
  sorry

/-- Theorem proving the final result (8 ⋆ (4 ⋆ 7)) ⋆ 3 = 1/2 -/
theorem final_result : star (star 8 (star 4 7)) 3 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_eq_half_final_result_l438_43877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l438_43899

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle_at_B : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Define point D as the foot of the altitude from B
def altitude_foot (A B C D : ℝ × ℝ) : Prop :=
  (D.1 - A.1) * (C.1 - A.1) + (D.2 - A.2) * (C.2 - A.2) = 0 ∧
  (B.1 - D.1) * (C.1 - A.1) + (B.2 - D.2) * (C.2 - A.2) = 0

-- Define the lengths AD and DC
noncomputable def segment_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem area_of_triangle (A B C D : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : altitude_foot A B C D)
  (h3 : segment_length A D = 5)
  (h4 : segment_length D C = 3) :
  (1/2) * segment_length A C * Real.sqrt 15 = 4 * Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l438_43899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_square_weight_l438_43846

/-- Represents a square piece of wood -/
structure WoodSquare where
  sideLength : ℝ
  weight : ℝ

/-- The density of the wood in ounces per square inch -/
noncomputable def woodDensity (w : WoodSquare) : ℝ :=
  w.weight / (w.sideLength * w.sideLength)

theorem second_square_weight (s1 s2 : WoodSquare)
    (h1 : s1.sideLength = 4)
    (h2 : s1.weight = 16)
    (h3 : s2.sideLength = 6)
    (h4 : woodDensity s1 = woodDensity s2) :
    s2.weight = 36 := by
  sorry

#check second_square_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_square_weight_l438_43846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_range_l438_43865

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (a - 2) * x else -1/8 * x - 1/2

theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ≤ 13/8 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_range_l438_43865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_BC_l438_43801

/-- Parabola defined by y = 4x^2 -/
noncomputable def parabola (x : ℝ) : ℝ := 4 * x^2

/-- Triangle ABC with vertices on the parabola y = 4x^2 -/
structure Triangle where
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_B_on_parabola : B.2 = parabola B.1
  h_C_on_parabola : C.2 = parabola C.1
  h_BC_parallel_x : B.2 = C.2

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

theorem length_of_BC (t : Triangle) (h_area : triangle_area (t.C.1 - t.B.1) t.B.2 = 256) :
  t.C.1 - t.B.1 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_BC_l438_43801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_value_f_max_value_on_interval_smallest_positive_period_l438_43889

noncomputable section

open Real

/-- The function f(x) = sin(ωx + π/4) -/
def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 4)

theorem f_zero_value (ω : ℝ) (h : ω > 0) :
  f ω 0 = sqrt 2 / 2 := by sorry

theorem f_max_value_on_interval :
  let f (x : ℝ) := sin (2 * x + π / 4)
  ∃ (x : ℝ), x ∈ Set.Icc 0 (π / 2) ∧
    (∀ (y : ℝ), y ∈ Set.Icc 0 (π / 2) → f y ≤ f x) ∧
    f x = 1 := by sorry

/-- The smallest positive period of f(x) = sin(ωx + π/4) is π when ω = 2 -/
theorem smallest_positive_period (ω : ℝ) (h : ω > 0) :
  (∀ (x : ℝ), f ω (x + π) = f ω x) ∧
  (∀ (T : ℝ), T > 0 → T < π → ∃ (x : ℝ), f ω (x + T) ≠ f ω x) →
  ω = 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_value_f_max_value_on_interval_smallest_positive_period_l438_43889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_16_5_l438_43839

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculate the area of a trapezoid given its bases and height -/
noncomputable def trapezoidArea (a b h : ℝ) : ℝ :=
  (a + b) * h / 2

/-- Main theorem: Area of trapezoid ABCD is 16.5 cm² -/
theorem trapezoid_area_is_16_5 (ABCD : Trapezoid)
  (hAB : |ABCD.A.x - ABCD.B.x| = 7)
  (hCD : |ABCD.C.x - ABCD.D.x| = 4)
  (hS : ∃ S : Point, S.x = (ABCD.A.x + ABCD.D.x) / 2 ∧ S.y = (ABCD.A.y + ABCD.D.y) / 2)
  (hT : ∃ T : Point, T.x = (ABCD.B.x + ABCD.C.x) / 2 ∧ T.y = (ABCD.B.y + ABCD.C.y) / 2)
  (hX : ∃ X : Point, X.x ∈ Set.Icc ABCD.A.x ABCD.C.x ∧ X.y ∈ Set.Icc ABCD.A.y ABCD.C.y)
  (hY : ∃ Y : Point, Y.x ∈ Set.Icc ABCD.A.x ABCD.B.x ∧ Y.y ∈ Set.Icc ABCD.D.y ABCD.A.y)
  (hAYCD : ∃ Y : Point, trapezoidArea (|ABCD.A.x - Y.x|) (|ABCD.C.x - ABCD.D.x|) 
           (|ABCD.A.y - ABCD.C.y|) = 12) :
  trapezoidArea (|ABCD.A.x - ABCD.B.x|) (|ABCD.C.x - ABCD.D.x|) 
    (|ABCD.A.y - ABCD.C.y|) = 16.5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_16_5_l438_43839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_graph_and_coordinate_sum_l438_43878

/-- Given a function f : ℝ → ℝ such that f(9) = 7, prove that (3, 16/9) is on the graph of 
    3y = f(3x)/3 + 3, and the sum of its coordinates is 43/9. -/
theorem point_on_graph_and_coordinate_sum (f : ℝ → ℝ) (h : f 9 = 7) : 
  (3 * (16/9) = f (3*3) / 3 + 3) ∧ (3 + 16/9 = 43/9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_graph_and_coordinate_sum_l438_43878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_group_size_for_game_l438_43863

-- Define plays_together and plays_round as parameters
theorem largest_group_size_for_game (n : ℕ) 
  (plays_together : ℕ → ℕ → ℕ → Prop)
  (plays_round : ℕ → ℕ → ℕ → ℕ → Prop) : 
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃ (r : ℕ), r < n ∧ plays_together i j r) →
  (∀ (r : ℕ), r < n → ∃! (p1 p2 p3 : ℕ), p1 < n ∧ p2 < n ∧ p3 < n ∧ 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    plays_round p1 p2 p3 r) →
  n ≤ 7 :=
by
  sorry

-- Define the properties separately
def plays_together_prop (n : ℕ) (plays_round : ℕ → ℕ → ℕ → ℕ → Prop) (i j r : ℕ) : Prop :=
  ∃ (k : ℕ), k < n ∧ k ≠ i ∧ k ≠ j ∧ plays_round i j k r

def plays_round_prop (n : ℕ) (plays_together : ℕ → ℕ → ℕ → Prop) (p1 p2 p3 r : ℕ) : Prop :=
  p1 < n ∧ p2 < n ∧ p3 < n ∧ 
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
  (∀ (q : ℕ), q < n → q ≠ p1 → q ≠ p2 → q ≠ p3 → 
    ¬plays_together q p1 r ∧ ¬plays_together q p2 r ∧ ¬plays_together q p3 r)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_group_size_for_game_l438_43863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equation_solution_l438_43897

noncomputable def f (n : ℝ) : ℝ :=
  if n < 0 then n^2 - 2 else 2*n - 20

theorem f_equation_solution :
  ∃ a₁ a₂ : ℝ,
    f (-2) + f 2 + f a₁ = 0 ∧
    f (-2) + f 2 + f a₂ = 0 ∧
    a₁ ≠ a₂ ∧
    (∀ a : ℝ, f (-2) + f 2 + f a = 0 → (a = a₁ ∨ a = a₂)) ∧
    |a₁ - a₂| = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equation_solution_l438_43897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_area_l438_43884

/-- Regular hexagon with apothem 3 -/
structure RegularHexagon where
  apothem : ℝ
  apothem_eq : apothem = 3

/-- Midpoint of a side of the hexagon -/
def midpoint_of_side (h : RegularHexagon) : Type := Unit

/-- Triangle formed by three consecutive midpoints -/
def MidpointTriangle (h : RegularHexagon) :=
  (midpoint_of_side h) × (midpoint_of_side h) × (midpoint_of_side h)

/-- Area of a triangle -/
noncomputable def area (h : RegularHexagon) (t : MidpointTriangle h) : ℝ := 
  9 * Real.sqrt 3 / 4

/-- Theorem: The area of the triangle formed by the midpoints of three consecutive
    sides of a regular hexagon with apothem 3 is 9√3/4 -/
theorem midpoint_triangle_area (h : RegularHexagon) (t : MidpointTriangle h) :
  area h t = 9 * Real.sqrt 3 / 4 := by
  sorry

#check midpoint_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_area_l438_43884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nh4no3_equals_nano3_l438_43893

/-- Represents the number of moles of a chemical substance -/
def Moles : Type := ℝ

/-- Represents the chemical reaction between NH4NO3 and NaOH to form NaNO3 -/
structure Reaction where
  nh4no3 : Moles
  naoh : Moles
  nano3 : Moles
  ratio : nh4no3 = naoh ∧ nh4no3 = nano3

/-- The given reaction with 2 moles of NaOH and 2 moles of NaNO3 formed -/
def given_reaction : Reaction where
  nh4no3 := (2 : ℝ)
  naoh := (2 : ℝ)
  nano3 := (2 : ℝ)
  ratio := by
    constructor
    · rfl
    · rfl

/-- Theorem stating that the number of moles of NH4NO3 needed is equal to the number of moles of NaNO3 formed -/
theorem nh4no3_equals_nano3 (r : Reaction) : r.nh4no3 = r.nano3 := by
  exact r.ratio.2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nh4no3_equals_nano3_l438_43893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_prime_l438_43847

theorem fraction_is_prime (m n : ℕ+) (p : ℕ) (h_p : Prime p) :
  let frac := (7^(m : ℕ) + p * 2^(n : ℕ)) / (7^(m : ℕ) - p * 2^(n : ℕ))
  (∃ k : ℤ, frac = k) →
  frac = 13 ∨ frac = 97 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_prime_l438_43847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_cone_height_ratio_l438_43860

/-- Represents a right cone with a circular base -/
structure RightCone where
  base_circumference : ℝ
  height : ℝ

/-- Calculates the volume of a right cone -/
noncomputable def volume (cone : RightCone) : ℝ :=
  (1/3) * (cone.base_circumference / (2 * Real.pi)) ^ 2 * Real.pi * cone.height

/-- Theorem statement -/
theorem shorter_cone_height_ratio (original_cone : RightCone) 
    (h_base : original_cone.base_circumference = 16 * Real.pi)
    (h_height : original_cone.height = 30)
    (h_shorter_volume : ∃ shorter_height : ℝ, 
      volume { base_circumference := original_cone.base_circumference, 
               height := shorter_height } = 192 * Real.pi) :
  ∃ shorter_height : ℝ, 
    volume { base_circumference := original_cone.base_circumference, 
             height := shorter_height } = 192 * Real.pi ∧
    shorter_height / original_cone.height = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_cone_height_ratio_l438_43860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_geometric_progression_l438_43851

/-- The base of the triangle in terms of height and common ratio. -/
def base (h r : ℝ) : ℝ := h * r

/-- The area of a triangle with base and height in geometric progression. -/
theorem triangle_area_geometric_progression (h r A : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  (base h r = h * r) → (A = (1/2) * base h r * h) → A = (1/2) * r * h^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_geometric_progression_l438_43851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_probability_l438_43888

/-- Pentagon vertices -/
noncomputable def F : ℝ × ℝ := (0, 1)
noncomputable def G : ℝ × ℝ := (3, 0)
noncomputable def H : ℝ × ℝ := (2*Real.pi + 2, 0)
noncomputable def I : ℝ × ℝ := (2*Real.pi + 2, 3)
noncomputable def J : ℝ × ℝ := (0, 3)

/-- Area of the pentagon FGHIJ -/
noncomputable def pentagon_area : ℝ := 6*Real.pi + 3

/-- Probability of ∠FQG being obtuse -/
noncomputable def prob_obtuse_angle : ℝ := 5 / (24*Real.pi + 12)

/-- The main theorem -/
theorem obtuse_angle_probability :
  prob_obtuse_angle = (5*Real.pi/4) / pentagon_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_probability_l438_43888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_matches_match_eliminates_one_matches_64_players_matches_2011_players_l438_43800

/-- The number of matches in a tournament with n players. -/
def number_of_matches (n : ℕ) : ℕ := n - 1

/-- In a single-elimination tournament, the number of matches is one less than the number of players. -/
theorem tournament_matches (n : ℕ) (h : n > 0) : 
  number_of_matches n = n - 1 :=
by rfl

/-- A player is a winner if they are not eliminated. -/
def is_winner (player : ℕ) (total_matches : ℕ) : Prop := 
  player > total_matches

/-- The number of players remaining after m matches. -/
def players_remaining (n : ℕ) (m : ℕ) : ℕ := n - m

/-- There is one winner in the tournament. -/
axiom one_winner (n : ℕ) (h : n > 0) : 
  ∃! player, is_winner player (number_of_matches n)

/-- Each match eliminates one player. -/
theorem match_eliminates_one (n : ℕ) (m : ℕ) (h : n > m) :
  players_remaining n m = n - m :=
by rfl

/-- The number of matches in a tournament with 64 players is 63. -/
theorem matches_64_players : 
  number_of_matches 64 = 63 :=
by rfl

/-- The number of matches in a tournament with 2011 players is 2010. -/
theorem matches_2011_players : 
  number_of_matches 2011 = 2010 :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_matches_match_eliminates_one_matches_64_players_matches_2011_players_l438_43800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_9_30_l438_43855

/-- The rotation rate of the hour hand in degrees per minute -/
def hour_hand_rotation_rate : ℚ := 1/2

/-- The rotation rate of the minute hand in degrees per minute -/
def minute_hand_rotation_rate : ℚ := 6

/-- The angle between the hour and minute hands at 9:00 AM -/
def initial_angle : ℚ := 270

/-- The number of minutes between 9:00 AM and 9:30 AM -/
def time_elapsed : ℕ := 30

/-- The acute angle formed by the hour and minute hands at 9:30 AM -/
def clock_angle (hr : ℚ) (min : ℚ) (init : ℚ) (t : ℕ) : ℚ :=
  let angle := init + hr * t - min * t
  if angle > 180 then 360 - angle else angle

theorem clock_angle_at_9_30 :
  clock_angle hour_hand_rotation_rate minute_hand_rotation_rate initial_angle time_elapsed = 105 := by
  sorry

#eval clock_angle hour_hand_rotation_rate minute_hand_rotation_rate initial_angle time_elapsed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_9_30_l438_43855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cubes_between_powers_of_two_l438_43812

theorem perfect_cubes_between_powers_of_two : 
  let lower_bound := 2^9 + 1
  let upper_bound := 2^17 + 1
  (Finset.filter (fun n : ℕ => lower_bound ≤ n^3 ∧ n^3 ≤ upper_bound) 
    (Finset.range 51)).card = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cubes_between_powers_of_two_l438_43812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l438_43833

-- Define the right triangle DEF
structure RightTriangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ
  is_right_angle : DE^2 + DF^2 = EF^2

-- Define the properties we want to prove
noncomputable def median_length (t : RightTriangle) : ℝ := t.EF / 2

noncomputable def altitude_length (t : RightTriangle) : ℝ := (t.DE * t.DF) / t.EF

-- Theorem statement
theorem right_triangle_properties (t : RightTriangle) 
  (h1 : t.DE = 5)
  (h2 : t.DF = 12)
  (h3 : t.EF = 13) :
  median_length t = 13/2 ∧ altitude_length t = 60/13 := by
  sorry

#check right_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l438_43833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_properties_l438_43896

noncomputable def arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

noncomputable def sum_arithmetic (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

noncomputable def geometric_sequence (b : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ+, b (n + 1) = q * b n

noncomputable def sum_geometric (b : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  let q := b 2 / b 1
  (b 1 * (1 - q^(n : ℝ))) / (1 - q)

theorem arithmetic_geometric_sequence_properties
  (a : ℕ+ → ℝ) (b : ℕ+ → ℝ) (n : ℕ+) :
  arithmetic_sequence a →
  a 3 = 5 →
  sum_arithmetic a 3 = 9 →
  geometric_sequence b →
  (∃ q : ℝ, q > 0 ∧ ∀ n : ℕ+, b (n + 1) = q * b n) →
  b 3 = a 5 →
  sum_geometric b 3 = 13 →
  (∀ n : ℕ+, b n = 1 / (a n * a (n + 1))) →
  (∀ n : ℕ+, a n = 2 * n - 1) ∧
  (∀ n : ℕ+, sum_geometric b n = (3^(n : ℝ) - 1) / 2) ∧
  (∀ n : ℕ+, sum_arithmetic b n = n / (2 * n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_properties_l438_43896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_AOB_l438_43876

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the point P
def P : ℝ × ℝ := (0, -2)

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the chord length
def chord_length (k m : ℝ) : ℝ := 3

-- Define the area of triangle AOB
noncomputable def area_AOB (k m : ℝ) : ℝ := 
  let d := Real.sqrt 3 / 2
  let AB := Real.sqrt (2 * (1 + k^2) * (5 * k^2 + 1)) / (2 * k^2 + 1)
  d * AB / 2

theorem max_area_AOB :
  ∀ k m : ℝ, 
    line_l k m (F.1) (F.2) →
    chord_length k m = 3 →
    (∀ x y : ℝ, line_l k m x y ∧ circle_O x y → ellipse x y) →
    area_AOB k m ≤ Real.sqrt 2 / 2 ∧
    (∃ k₀ m₀ : ℝ, area_AOB k₀ m₀ = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_AOB_l438_43876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l438_43829

-- Define the functions f and g
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := 9^x - t * 3^x
noncomputable def g (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- State the theorem
theorem range_of_t (t : ℝ) : 
  (∃ a b : ℝ, g a + g b = 0 ∧ f t a + f t b = 0) → t ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l438_43829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_to_children_ratio_l438_43803

/-- Represents a group of people with men, women, and children -/
structure PeopleGroup where
  men : ℕ
  women : ℕ
  children : ℕ

/-- Theorem: The ratio of women to children in the group is 3:1 -/
theorem women_to_children_ratio (g : PeopleGroup) : 
  g.men = 2 * g.women →
  g.children = 30 →
  g.men + g.women + g.children = 300 →
  g.women = 3 * g.children := by
  sorry

#check women_to_children_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_to_children_ratio_l438_43803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_concentric_l438_43840

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The function f representing the equation of a circle -/
noncomputable def f : ℝ → ℝ → ℝ := sorry

/-- Circle C1 with equation f(x,y) = 0 -/
noncomputable def C1 : Circle := sorry

/-- Circle C2 with equation f(x,y) = f(x,y) -/
noncomputable def C2 : Circle := sorry

/-- Point P outside of circle C1 -/
noncomputable def P : ℝ × ℝ := sorry

/-- Condition that P is outside C1 -/
axiom P_outside_C1 : f P.1 P.2 > 0

theorem circles_are_concentric (hf : ∀ x y, f x y = 0 ↔ (x - C1.center.1)^2 + (y - C1.center.2)^2 = C1.radius^2) :
  C1.center = C2.center := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_concentric_l438_43840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_h_one_zero_h_two_zeros_h_no_zeros_l438_43826

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x * (x + 1)

-- Define the function h(x, a)
noncomputable def h (x a : ℝ) : ℝ := g x - a * (x^3 + x^2)

-- Theorem for the tangent line
theorem tangent_line_at_zero_one :
  ∃ (m b : ℝ), ∀ x, m * x + b = g x + (deriv g 0) * (x - 0) :=
sorry

-- Theorems for the number of zeros of h(x)
theorem h_one_zero (a : ℝ) :
  a = Real.exp 2 / 4 → ∃! x, x > 0 ∧ h x a = 0 :=
sorry

theorem h_two_zeros (a : ℝ) :
  a > Real.exp 2 / 4 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ h x₁ a = 0 ∧ h x₂ a = 0 ∧
    ∀ x, x > 0 ∧ h x a = 0 → x = x₁ ∨ x = x₂ :=
sorry

theorem h_no_zeros (a : ℝ) :
  0 < a ∧ a < Real.exp 2 / 4 → ∀ x, x > 0 → h x a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_h_one_zero_h_two_zeros_h_no_zeros_l438_43826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_matrix_A_is_zero_l438_43818

noncomputable def matrix_A : Matrix (Fin 3) (Fin 3) ℝ := 
  λ i j => Real.sin ((i.val * 3 + j.val + 1) : ℝ)

theorem det_matrix_A_is_zero : 
  Matrix.det matrix_A = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_matrix_A_is_zero_l438_43818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jiyeol_average_score_l438_43881

/-- Represents the scores for Korean, Math, and English tests -/
structure TestScores where
  korean : ℝ
  math : ℝ
  english : ℝ

/-- Calculates the mean of two real numbers -/
noncomputable def mean (a b : ℝ) : ℝ := (a + b) / 2

/-- Calculates the average of three real numbers -/
noncomputable def average (a b c : ℝ) : ℝ := (a + b + c) / 3

/-- Theorem: Given the conditions, Jiyeol's whole average score is 30 -/
theorem jiyeol_average_score (scores : TestScores) 
  (h1 : mean scores.korean scores.math = 26.5)
  (h2 : mean scores.math scores.english = 34.5)
  (h3 : mean scores.korean scores.english = 29) :
  average scores.korean scores.math scores.english = 30 := by
  sorry

#check jiyeol_average_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jiyeol_average_score_l438_43881
