import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_l865_86507

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem angle_B_value (t : Triangle) : 
  t.A = π/4 ∧ 
  t.a = 2 ∧ 
  t.b * Real.cos t.C - t.c * Real.cos t.B = 2 * Real.sqrt 2 → 
  t.B = 5*π/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_l865_86507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_transmission_properties_l865_86530

/-- A channel with independent signal transmission -/
structure Channel where
  α : ℝ
  β : ℝ
  h_α : 0 < α ∧ α < 1
  h_β : 0 < β ∧ β < 1

/-- Probability of receiving a sequence in single transmission -/
def singleTransmissionProb (c : Channel) (send receive : List Bool) : ℝ :=
  sorry

/-- Probability of receiving a sequence in triple transmission -/
def tripleTransmissionProb (c : Channel) (send receive : List Bool) : ℝ :=
  sorry

/-- Probability of decoding as a specific signal in triple transmission -/
def tripleTransmissionDecodeProb (c : Channel) (send decode : Bool) : ℝ :=
  sorry

theorem channel_transmission_properties (c : Channel) :
  let send101 := [true, false, true]
  let recv101 := [true, false, true]
  let send111 := [true, true, true]
  (singleTransmissionProb c send101 recv101 = (1 - c.α) * (1 - c.β)^2) ∧
  (tripleTransmissionProb c send111 recv101 = c.β * (1 - c.β)^2) ∧
  (tripleTransmissionDecodeProb c true true = 3 * c.β * (1 - c.β)^2 + (1 - c.β)^3) ∧
  (∀ α, 0 < α → α < 0.5 → 
    tripleTransmissionDecodeProb {α := α, β := c.β, h_α := ⟨sorry, sorry⟩, h_β := c.h_β} false false >
    singleTransmissionProb {α := α, β := c.β, h_α := ⟨sorry, sorry⟩, h_β := c.h_β} [false] [false]) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_transmission_properties_l865_86530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_sine_to_cosine_l865_86535

theorem min_shift_sine_to_cosine (φ : ℝ) :
  (∀ x, Real.sin (2 * (x + φ)) = Real.cos (2 * x)) ∧ φ > 0 →
  φ ≥ π / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_sine_to_cosine_l865_86535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_equality_l865_86516

theorem expression_one_equality : (1 : ℝ) - 2^(2 : ℝ) + 3^(0 : ℝ) - (-1/2)^(-(1 : ℝ)) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_equality_l865_86516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l865_86541

noncomputable section

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  (cos B) / (cos C) = b / (2 * a - c) →
  -- b = √7
  b = sqrt 7 →
  -- Area of triangle
  (1/2) * a * c * (sin B) = (3 * sqrt 3) / 2 →
  -- Prove:
  B = π/3 ∧ a + c = 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l865_86541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l865_86536

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a^2 - 5*a - 3 ≥ 3) ∧ 
  (∀ x : ℝ, x^2 + a*x + 2 ≥ 0) → 
  -2 * Real.sqrt 2 ≤ a ∧ a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l865_86536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_l865_86592

-- Define the function f(x) = 2^(-x)
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x * Real.log 2)

-- State the theorem
theorem f_monotonically_decreasing :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_l865_86592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smith_family_probability_l865_86576

/-- Represents the number of children in the family -/
def total_children : ℕ := 8

/-- Represents the number of twins in the family -/
def num_twins : ℕ := 2

/-- Represents the probability of a child being male or female -/
noncomputable def gender_probability : ℚ := 1/2

/-- Calculates the probability of having an unequal number of sons and daughters -/
noncomputable def unequal_gender_probability : ℚ := 27/32

/-- Theorem stating the probability of having an unequal number of sons and daughters in the Smith family -/
theorem smith_family_probability :
  let independent_children := total_children - num_twins
  let total_outcomes := 2^independent_children
  let equal_gender_outcomes := (Nat.choose independent_children (independent_children / 2)) * 2
  (total_outcomes - equal_gender_outcomes) / total_outcomes = unequal_gender_probability := by
  sorry

#check smith_family_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smith_family_probability_l865_86576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_I_expression_II_l865_86554

-- Expression I
theorem expression_I : 
  Real.sqrt (9/4) - ((-2):ℝ)^(0:ℝ) - (27/8:ℝ)^(-(2/3):ℝ) + (3/2:ℝ)^(-2:ℝ) = 1/2 := by sorry

-- Expression II
theorem expression_II :
  (Real.log 5 / Real.log 10) + Real.log 2 - (Real.log 8 / Real.log 4) + (3:ℝ)^(Real.log 2 / Real.log 3) = Real.log 5 + 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_I_expression_II_l865_86554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_number_28_l865_86505

/-- A positive integer is perfect if the sum of its proper divisors equals the number itself. -/
def IsPerfect (n : ℕ) : Prop :=
  (Finset.sum (Finset.filter (· < n) (Finset.range n)) id) = n

/-- The sum of reciprocals of positive factors of a number. -/
def SumReciprocalFactors (n : ℕ) : ℚ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => (1 : ℚ) / x)

/-- The sum of positive factors of a number. -/
def SumFactors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem perfect_number_28 :
  IsPerfect 28 ∧ SumReciprocalFactors 28 = 2 → SumFactors 28 = 56 := by
  sorry

#eval SumFactors 28

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_number_28_l865_86505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_product_l865_86587

def sequence_a : ℕ → ℚ
| 0 => 1/2
| n + 1 => sequence_a n / (3 * sequence_a n + 1)

def sequence_product (n : ℕ) : ℚ := sequence_a n * sequence_a (n + 1)

theorem sum_of_sequence_product (n : ℕ) :
  (Finset.range n).sum (λ i => sequence_product i) = n / (6 * n + 4) := by
  sorry

#eval sequence_a 1
#eval sequence_product 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_product_l865_86587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_polar_line_l865_86599

/-- The distance from the origin to a line given in polar form -/
theorem distance_to_polar_line :
  let polar_eq : ℝ → ℝ → Prop := λ ρ θ ↦ ρ * (Real.cos θ + Real.sin θ) = Real.sqrt 3
  let distance : ℝ := Real.sqrt 6 / 2
  (∃ (ρ θ : ℝ), polar_eq ρ θ) →
  (∀ (x y : ℝ), x + y = Real.sqrt 3 ↔ ∃ (ρ θ : ℝ), polar_eq ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  distance = |-(Real.sqrt 3)| / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_polar_line_l865_86599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l865_86564

/-- Curve C₁ in parametric form -/
noncomputable def C₁ (t : ℝ) : ℝ × ℝ :=
  ((Real.sqrt 5 / 5) * t, (2 * Real.sqrt 5 / 5) * t - 1)

/-- Curve C₂ in polar form -/
noncomputable def C₂ (θ : ℝ) : ℝ :=
  2 * Real.cos θ - 4 * Real.sin θ

/-- The distance between the two intersection points of C₁ and C₂ -/
theorem intersection_distance : ∃ (p₁ p₂ : ℝ × ℝ),
  (∃ (t : ℝ), C₁ t = p₁) ∧
  (∃ (θ : ℝ), (C₂ θ * Real.cos θ, C₂ θ * Real.sin θ) = p₁) ∧
  (∃ (t : ℝ), C₁ t = p₂) ∧
  (∃ (θ : ℝ), (C₂ θ * Real.cos θ, C₂ θ * Real.sin θ) = p₂) ∧
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) = 8 * Real.sqrt 5 / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l865_86564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_same_ending_digits_l865_86559

/-- A three-digit number is between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A number N has the property that any integer power of it ends with the same three digits as the original number. -/
def HasSameEndingDigits (N : ℕ) : Prop := ∀ k : ℕ, k > 0 → N^k % 1000 = N

theorem three_digit_same_ending_digits :
  ∀ N : ℕ, ThreeDigitNumber N → HasSameEndingDigits N → (N = 625 ∨ N = 376) :=
by sorry

#check three_digit_same_ending_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_same_ending_digits_l865_86559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l865_86585

theorem simplify_trig_expression (θ : ℝ) (h : π / 2 < θ ∧ θ < π) :
  (Real.sin θ / Real.sqrt (1 - Real.sin θ ^ 2)) + (Real.sqrt (1 - Real.cos θ ^ 2) / Real.cos θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l865_86585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_term_difference_l865_86528

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  length : ℕ
  sum : ℚ
  min_value : ℚ
  max_value : ℚ

/-- The 40th term of an arithmetic sequence -/
noncomputable def fortieth_term (seq : ArithmeticSequence) (d : ℚ) : ℚ :=
  (seq.sum / seq.length) - 111 * d

/-- The theorem to be proved -/
theorem fortieth_term_difference (seq : ArithmeticSequence) : 
  seq.length = 150 → 
  seq.sum = 9000 → 
  seq.min_value = 20 → 
  seq.max_value = 90 → 
  ∃ (L G : ℚ), 
    (∀ d : ℚ, L ≤ fortieth_term seq d ∧ fortieth_term seq d ≤ G) ∧
    G - L = 6660 / 149 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_term_difference_l865_86528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_division_l865_86598

-- Define a cube
structure Cube where
  side_length : ℝ
  surface_area : ℕ
  num_faces : ℕ

-- Define the properties of our cube
def paper_cube : Cube :=
  { side_length := 1,  -- Arbitrary side length
    surface_area := 6, -- 6 * side_length^2
    num_faces := 6 }

-- Theorem statement
theorem cube_surface_division (c : Cube) :
  (c.surface_area % 6 = 0) → (c.surface_area % 12 = 0) :=
by
  intro h
  sorry

#check cube_surface_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_division_l865_86598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_estimators_approx_l865_86544

/-- Gamma distribution parameters -/
structure GammaParams where
  α : ℝ
  β : ℝ
  α_gt_neg_one : α > -1
  β_pos : β > 0

/-- Sample data for floods -/
structure FloodData where
  xi : List ℝ
  ni : List ℕ
  total_samples : Nat
  sample_mean : ℝ
  sample_variance : ℝ

/-- Method of moments estimators for Gamma distribution -/
noncomputable def method_of_moments (data : FloodData) : GammaParams where
  α := (data.sample_mean^2 / data.sample_variance) - 1
  β := data.sample_variance / data.sample_mean
  α_gt_neg_one := by sorry
  β_pos := by sorry

/-- Given flood data -/
def flood_data : FloodData where
  xi := [37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 250, 350]
  ni := [1, 3, 6, 7, 7, 5, 4, 8, 4]
  total_samples := 45
  sample_mean := 166
  sample_variance := 6782

/-- Theorem stating the approximate values of the estimators -/
theorem gamma_estimators_approx (ε : ℝ) (ε_pos : ε > 0) :
  let estimators := method_of_moments flood_data
  |estimators.α - 3.06| < ε ∧ |estimators.β - 40.86| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_estimators_approx_l865_86544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_numbering_exists_l865_86522

/-- A type representing a point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a line on a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Check if two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

/-- The main theorem -/
theorem intersection_numbering_exists (points : Finset Point) 
  (h1 : points.card = 200)
  (h2 : ∀ p q r, p ∈ points → q ∈ points → r ∈ points → 
       p ≠ q → q ≠ r → p ≠ r → ¬collinear p q r) :
  ∃ f : Fin 200 → Point,
    (∀ i, f i ∈ points) ∧ 
    (∀ i j : Fin 100, i ≠ j → 
      intersect (Line.mk 1 0 0) (Line.mk 1 0 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_numbering_exists_l865_86522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l865_86545

theorem expression_value : 
  let x : ℝ := 0.96
  let y : ℝ := 0.1
  abs ((x^3 - y^3 / x^2 + 0.096 + y^2) - 0.989651) < 1e-6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l865_86545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_y_l865_86525

theorem find_y (y : ℝ) (h : (3 : ℝ)^(y-2) = (9 : ℝ)^3) : y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_y_l865_86525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l865_86527

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / x + 1 / (2 * x + 1 / x)

/-- Theorem stating that 5√2/3 is the minimum value of f(x) for x > 0 -/
theorem f_min_value : ∀ x : ℝ, x > 0 → f x ≥ (5 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l865_86527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l865_86518

/-- The length of each train given the conditions of the problem -/
noncomputable def train_length (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  let relative_speed := (v1 - v2) * 1000 / 3600
  (relative_speed * t) / 2

/-- Theorem stating the length of each train under the given conditions -/
theorem train_length_calculation :
  let v1 := (75 : ℝ) -- speed of faster train in km/hr
  let v2 := (55 : ℝ) -- speed of slower train in km/hr
  let t := (210 : ℝ) -- time taken to pass in seconds
  abs (train_length v1 v2 t - 583.33) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l865_86518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_5pi_6_f_range_l865_86575

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

-- Theorem 1: f(5π/6) = 0
theorem f_at_5pi_6 : f (5 * Real.pi / 6) = 0 := by sorry

-- Theorem 2: Range of f(x) when x ∈ [0, π/4] is [1, 2]
theorem f_range : ∀ y, y ∈ Set.range (fun x => f x) → x ∈ Set.Icc 0 (Real.pi / 4) → 1 ≤ y ∧ y ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_5pi_6_f_range_l865_86575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_hiker_catch_up_time_l865_86571

/-- Calculates the waiting time for a cyclist to be caught by a hiker -/
noncomputable def cyclist_wait_time (hiker_speed cyclist_speed : ℝ) (cyclist_wait : ℝ) : ℝ :=
  let cyclist_distance := (cyclist_speed / 60) * cyclist_wait
  cyclist_distance / (hiker_speed / 60)

theorem cyclist_hiker_catch_up_time :
  cyclist_wait_time 5 25 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_hiker_catch_up_time_l865_86571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_communities_count_l865_86556

/-- The number of boys in the school -/
def total_boys : ℕ := 1987

/-- The percentage of Muslim boys -/
def muslim_percent : ℚ := 367/1000

/-- The percentage of Hindu boys -/
def hindu_percent : ℚ := 243/1000

/-- The percentage of Sikh boys -/
def sikh_percent : ℚ := 95/1000

/-- The percentage of Christian boys -/
def christian_percent : ℚ := 53/1000

/-- The percentage of Buddhist boys -/
def buddhist_percent : ℚ := 21/1000

/-- The number of boys belonging to other communities -/
def other_boys : ℕ := 419

theorem other_communities_count :
  (1 - (muslim_percent + hindu_percent + sikh_percent + christian_percent + buddhist_percent)) * (total_boys : ℚ) = other_boys := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_communities_count_l865_86556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l865_86504

theorem sin_plus_cos_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : -Real.cos α + Real.sin α = 2/3) : 
  Real.sin α + Real.cos α = Real.sqrt 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l865_86504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paperboy_delivery_ways_l865_86532

/-- Represents the number of ways to deliver newspapers to n houses, 
    where no four consecutive houses can be skipped. -/
def D : ℕ → ℕ
| 0 => 1  -- Added case for 0
| 1 => 2
| 2 => 4
| 3 => 8
| 4 => 15
| (n+5) => D (n+4) + D (n+3) + D (n+2) + D (n+1)

/-- The number of ways to deliver newspapers to 12 houses, 
    where no four consecutive houses can be skipped, is 2873. -/
theorem paperboy_delivery_ways : D 12 = 2873 := by
  -- Proof goes here
  sorry

#eval D 12  -- This line will evaluate D 12 and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paperboy_delivery_ways_l865_86532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_launch_work_l865_86568

-- Define constants
noncomputable def P : ℝ := 1.5 * 1000  -- 1.5 tons in kg
noncomputable def H : ℝ := 2000000     -- 2000 km in meters
noncomputable def R : ℝ := 6400000     -- 6400 km in meters

-- Define gravitational force function
noncomputable def F (x : ℝ) : ℝ := (P * R^2) / x^2

-- Theorem statement
theorem rocket_launch_work :
  ∃ Q : ℝ, Q = (P * R * H) / (R + H) ∧
    Q = ∫ x in R..(R + H), F x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_launch_work_l865_86568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_root_product_l865_86520

-- Define the quadratic equation
noncomputable def quadratic (x m : ℝ) : ℝ := 2 * x^2 - 6 * x + m

-- Define the discriminant
noncomputable def discriminant (m : ℝ) : ℝ := (-6)^2 - 4 * 2 * m

-- Define the condition for real roots
def has_real_roots (m : ℝ) : Prop := discriminant m ≥ 0

-- Define the product of roots
noncomputable def root_product (m : ℝ) : ℝ := m / 2

-- Theorem statement
theorem max_root_product :
  ∃ (m : ℝ), has_real_roots m ∧ 
  (∀ (m' : ℝ), has_real_roots m' → root_product m' ≤ root_product m) ∧
  m = 4.5 := by sorry

#check max_root_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_root_product_l865_86520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_set_l865_86517

def S : Finset ℕ := {8, 88, 888, 8888, 88888, 888888, 8888888}

theorem arithmetic_mean_of_set :
  let N := (S.sum id) / S.card
  N = 1269841 ∧ ¬ (∃ d : ℕ, d < 10 ∧ d = 3 ∧ (N / 10^d % 10 = 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_set_l865_86517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l865_86529

/-- The solution function to the Cauchy problem -/
def solution (x : ℝ) : ℝ := x^2

/-- The differential equation -/
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, x * (x - 1) * (deriv y x) + y x = x^2 * (2 * x - 1)

/-- The initial condition -/
def initial_condition (y : ℝ → ℝ) : Prop :=
  y 2 = 4

theorem cauchy_problem_solution :
  differential_equation solution ∧ initial_condition solution := by
  sorry

#check cauchy_problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l865_86529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_pattern_ratio_l865_86515

/-- Represents the square pattern of tiles -/
structure TilePattern where
  black : ℕ
  white : ℕ

/-- Extends the tile pattern with alternating layers -/
def extendPattern (original : TilePattern) : TilePattern :=
  let originalSize := Nat.sqrt (original.black + original.white)
  let firstLayerSize := originalSize + 2
  let secondLayerSize := originalSize + 4
  let newBlack := firstLayerSize ^ 2 - originalSize ^ 2
  let newWhite := secondLayerSize ^ 2 - firstLayerSize ^ 2
  { black := original.black + newBlack
    white := original.white + newWhite }

/-- The theorem to be proved -/
theorem extended_pattern_ratio (original : TilePattern) 
  (h1 : original.black = 9)
  (h2 : original.white = 16) :
  let extended := extendPattern original
  (extended.black : ℚ) / extended.white = 33 / 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_pattern_ratio_l865_86515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_speed_approx_l865_86547

-- Define the structure for a trip segment
structure Segment where
  distance : ℝ
  speed : ℝ

-- Define the trip
def trip : List Segment := [
  { distance := 45, speed := 60 },
  { distance := 75, speed := 50 },
  { distance := 105, speed := 80 },
  { distance := 55, speed := 40 }
]

-- Calculate total distance
noncomputable def totalDistance : ℝ := (trip.map (λ s => s.distance)).sum

-- Calculate total time
noncomputable def totalTime : ℝ := (trip.map (λ s => s.distance / s.speed)).sum

-- Calculate average speed
noncomputable def averageSpeed : ℝ := totalDistance / totalTime

-- Theorem statement
theorem trip_average_speed_approx :
  |averageSpeed - 56.72| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_speed_approx_l865_86547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trigonometric_expression_cosine_sum_and_square_l865_86583

-- Part 1
theorem simplify_trigonometric_expression (x : ℝ) :
  (Real.sin x)^2 / (Real.sin x - Real.cos x) - (Real.sin x + Real.cos x) / (Real.tan x^2 - 1) - Real.sin x = Real.cos x :=
by sorry

-- Part 2
theorem cosine_sum_and_square (α : ℝ) 
  (h : Real.cos (π/6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5*π/6 + α) + (Real.cos (4*π/3 + α))^2 = (2 - Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trigonometric_expression_cosine_sum_and_square_l865_86583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l865_86552

def point := ℝ × ℝ

def reflect_x (p : point) : point :=
  (p.1, -p.2)

noncomputable def distance (p q : point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem reflection_distance (A : point) :
  reflect_x A = (-1, 4) → distance A (-1, 4) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l865_86552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_infinite_operations_l865_86589

/-- The smallest number of integers for infinite operations -/
theorem smallest_n_for_infinite_operations
  (p q : ℕ+) -- Two positive integers p and q
  (h_coprime : Nat.Coprime p.val q.val) -- Assumption that p and q are coprime
  : (n : ℕ) → -- For any natural number n
    (∀ (a : ℕ) (m : ℕ), m < n → a ∈ (Finset.range n).val → (a + p) ∈ (Finset.range n).val ∧ (a + q) ∈ (Finset.range n).val) -- If for all a in the range [0, n), both a+p and a+q are also in this range
    → n ≥ p + q -- Then n is greater than or equal to p + q
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_infinite_operations_l865_86589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l865_86509

/-- Represents the side lengths of the nine squares and the rectangle dimensions -/
structure SquareSides where
  a : Fin 9 → ℕ
  l : ℕ
  w : ℕ

/-- The conditions for the square side lengths and rectangle dimensions -/
def satisfiesConditions (s : SquareSides) : Prop :=
  s.a 1 + s.a 2 = s.a 3 ∧
  s.a 1 + s.a 3 = s.a 4 ∧
  s.a 3 + s.a 4 = s.a 5 ∧
  s.a 4 + s.a 5 = s.a 6 ∧
  s.a 2 + 2 * s.a 3 = s.a 7 ∧
  s.a 2 + s.a 7 = s.a 8 ∧
  s.a 1 + s.a 4 + s.a 6 = s.a 9 ∧
  s.a 6 + s.a 9 = s.a 7 + s.a 8 ∧
  s.l = s.a 9 ∧
  s.w = s.a 6 + s.a 8 ∧
  Nat.Coprime s.l s.w

theorem rectangle_perimeter (s : SquareSides) (h : satisfiesConditions s) :
  2 * (s.l + s.w) = 220 := by
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l865_86509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_implies_slope_l865_86526

/-- A line with equation ax + y + 2 - a = 0 -/
structure Line where
  a : ℝ

/-- The x-intercept of the line -/
noncomputable def x_intercept (l : Line) : ℝ := (l.a - 2) / l.a

/-- The y-intercept of the line -/
def y_intercept (l : Line) : ℝ := l.a - 2

/-- The slope of the line -/
def line_slope (l : Line) : ℝ := -l.a

theorem equal_intercepts_implies_slope (l : Line) :
  x_intercept l = y_intercept l → line_slope l = -1 ∨ line_slope l = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_implies_slope_l865_86526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_set_is_reals_l865_86549

/-- A set of real numbers is complete if for any real numbers a and b,
    if a + b is in the set, then a * b is also in the set. -/
def IsCompleteSet (A : Set ℝ) : Prop :=
  ∀ a b : ℝ, (a + b) ∈ A → (a * b) ∈ A

/-- The only complete set of real numbers is ℝ itself. -/
theorem complete_set_is_reals (A : Set ℝ) (hA : A.Nonempty) (hComplete : IsCompleteSet A) :
  A = Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_set_is_reals_l865_86549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_value_in_quadratic_sequence_l865_86546

/-- A sequence of function values generated by a quadratic equation -/
def QuadraticSequence (seq : List ℤ) : Prop :=
  ∃ (a b c : ℚ), ∀ (i : ℕ), i < seq.length →
    seq[i]! = Int.floor (a * i^2 + b * i + c)

/-- The second differences of a sequence -/
def SecondDifferences (seq : List ℤ) : List ℤ :=
  let firstDiffs := seq.zipWith (·-·) (seq.tail!)
  firstDiffs.zipWith (·-·) (firstDiffs.tail!)

/-- A proposition stating that all elements in a list are equal except one -/
def AllEqualExceptOne (lst : List ℤ) : Prop :=
  ∃ (common : ℤ) (index : ℕ), index < lst.length ∧
    ∀ (i : ℕ), i < lst.length → i ≠ index → lst[i]! = common

/-- The main theorem -/
theorem incorrect_value_in_quadratic_sequence
  (seq : List ℤ)
  (h1 : QuadraticSequence seq)
  (h2 : AllEqualExceptOne (SecondDifferences seq)) :
  ∃ (index : ℕ), index < seq.length ∧
    ¬∃ (a b c : ℚ), seq[index]! = Int.floor (a * index^2 + b * index + c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_value_in_quadratic_sequence_l865_86546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l865_86512

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α - Real.pi/4) = 7*Real.sqrt 2/10) 
  (h2 : Real.cos (2*α) = 7/25) : 
  Real.sin α = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l865_86512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l865_86574

/-- Represents a parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y ↦ y^2 = 2 * p * x

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_equation (para : Parabola) 
  (h1 : ∃ (P Q : Point), para.eq P.x P.y ∧ para.eq Q.x Q.y ∧ 
                         P.y = 2 * P.x + 1 ∧ Q.y = 2 * Q.x + 1)
  (h2 : ∃ (P Q : Point), para.eq P.x P.y ∧ para.eq Q.x Q.y ∧ 
                         distance P Q = Real.sqrt 15) :
  para.p = -2 ∨ para.p = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l865_86574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_boys_theorem_max_min_apples_theorem_l865_86595

/-- The maximum number of boys that can receive different quantities of apples from a total of 99 apples -/
def max_boys : ℕ := 13

/-- The maximum number of apples received by the boy who got the fewest, given 10 boys and 99 total apples -/
def max_min_apples : ℕ := 5

/-- The total number of apples -/
def total_apples : ℕ := 99

/-- Function to calculate the sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating the maximum number of boys that can receive different quantities of apples -/
theorem max_boys_theorem :
  sum_first_n max_boys ≤ total_apples ∧
  sum_first_n (max_boys + 1) > total_apples := by
  sorry

/-- Theorem stating the maximum number of apples received by the boy who got the fewest, given 10 boys -/
theorem max_min_apples_theorem (num_boys : ℕ) (h : num_boys = 10) :
  ∃ (distribution : Fin num_boys → ℕ),
    (∀ i j, i ≠ j → distribution i ≠ distribution j) ∧
    (∀ i, distribution i ≥ max_min_apples) ∧
    (Finset.sum Finset.univ (fun i => distribution i)) = total_apples := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_boys_theorem_max_min_apples_theorem_l865_86595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_of_triangle_l865_86503

def sequence_sum (n : ℕ) : ℕ := n^2

def triangle_side_2 : ℕ := sequence_sum 2 - sequence_sum 1
def triangle_side_3 : ℕ := sequence_sum 3 - sequence_sum 2
def triangle_side_4 : ℕ := sequence_sum 4 - sequence_sum 3

theorem largest_angle_of_triangle : 
  let a : ℝ := triangle_side_2
  let b : ℝ := triangle_side_3
  let c : ℝ := triangle_side_4
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_of_triangle_l865_86503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l865_86534

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem problem_solution :
  (∃ a : ℝ, 
    (a = -4 → 
      (A ∩ B a = {x : ℝ | 1/2 ≤ x ∧ x < 2}) ∧
      (A ∪ B a = {x : ℝ | -2 < x ∧ x ≤ 3})) ∧
    ((Set.compl A ∩ B a = B a) → a ≥ -1/4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l865_86534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l865_86565

theorem polynomial_value_bound {n : ℕ} (p : Polynomial ℝ) (x : Fin (n + 1) → ℝ) :
  p.degree = n →
  (∀ i j : Fin (n + 1), i < j → x i < x j) →
  ∃ i : Fin (n + 1), |p.eval (x i)| ≥ (n.factorial : ℝ) / 2^n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l865_86565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_directness_l865_86590

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  p1 : Point
  p2 : Point

structure Plane where
  p1 : Point
  p2 : Point
  p3 : Point

structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the necessary functions
def is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

def is_perpendicular (l1 l2 : Line) : Prop := sorry

def directness (t : Tetrahedron) : ℕ := sorry

-- The main theorem
theorem tetrahedron_directness (ABCD : Tetrahedron) 
  (h1 : is_perpendicular_to_plane (Line.mk ABCD.A ABCD.D) (Plane.mk ABCD.A ABCD.B ABCD.C))
  (h2 : is_perpendicular (Line.mk ABCD.A ABCD.C) (Line.mk ABCD.B ABCD.C)) :
  directness ABCD = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_directness_l865_86590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_length_l865_86537

/-- Two circles with radii 4 and 8 intersecting at a right angle have a common tangent of length 8 -/
theorem common_tangent_length (O O₁ A M N : ℝ × ℝ) : 
  let r₁ : ℝ := 4
  let r₂ : ℝ := 8
  let d : ℝ := Real.sqrt (r₁^2 + r₂^2)
  ∀ (circle₁ : Set (ℝ × ℝ)) (circle₂ : Set (ℝ × ℝ)),
  (∀ P, P ∈ circle₁ ↔ dist O P = r₂) →
  (∀ P, P ∈ circle₂ ↔ dist O₁ P = r₁) →
  A ∈ circle₁ ∧ A ∈ circle₂ →
  dist O O₁ = d →
  (O.1 - A.1) * (O₁.1 - A.1) + (O.2 - A.2) * (O₁.2 - A.2) = 0 →
  M ∈ circle₁ ∧ N ∈ circle₂ →
  (M.1 - N.1) * (O.1 - A.1) + (M.2 - N.2) * (O.2 - A.2) = 0 →
  (M.1 - N.1) * (O₁.1 - A.1) + (M.2 - N.2) * (O₁.2 - A.2) = 0 →
  dist M N = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_length_l865_86537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l865_86555

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (1 + m * x) + x^2 / 2 - m * x

theorem f_properties (m : ℝ) (h_m : 0 < m ∧ m ≤ 1) :
  (∀ x, -1 < x ∧ x ≤ 0 → f 1 x ≤ x^3 / 3) ∧
  (∃! x, x > -1 ∧ f 1 x = 0) ∧
  (m < 1 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0 ∧
    ∀ x, f m x = 0 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l865_86555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_for_f_l865_86553

open Real

-- Define the function f(x) = 2sin(2x + π/3)
noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)

-- Define the set of x-values for the axis of symmetry
def axisOfSymmetry : Set ℝ := {x | ∃ k : ℤ, x = k * π / 2 + π / 12}

-- Theorem stating that the defined set is indeed the axis of symmetry for f
theorem axis_of_symmetry_for_f :
  ∀ x ∈ axisOfSymmetry, ∀ y : ℝ, f (x - y) = f (x + y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_for_f_l865_86553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_a_closed_a_1_eq_2_a_recurrence_l865_86550

/-- A sequence defined recursively -/
def a : ℕ → ℕ
  | 0 => 2  -- Add this case to cover Nat.zero
  | 1 => 2
  | n + 1 => 2 * (n + 1 + a n)

/-- The proposed closed form of the sequence -/
def a_closed (n : ℕ) : ℕ := 2^(n+2) - 2*n - 4

/-- Theorem stating that the closed form equals the recursive definition -/
theorem a_equals_a_closed : ∀ n : ℕ, n ≥ 1 → a n = a_closed n := by
  sorry

/-- Proof that the sequence starts at 2 for n = 1 -/
theorem a_1_eq_2 : a 1 = 2 := by
  rfl

/-- Proof of the recurrence relation -/
theorem a_recurrence (n : ℕ) (h : n ≥ 1) : a (n + 1) = 2 * (n + 1 + a n) := by
  cases n
  · contradiction
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_a_closed_a_1_eq_2_a_recurrence_l865_86550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_of_g_l865_86573

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem sum_of_max_min_of_g (a : ℝ) (f : ℝ → ℝ) (h_odd : IsOdd f) :
  let g := fun x ↦ f x + 2
  ∃ (m M : ℝ), (∀ x ∈ Set.Icc (-a) a, m ≤ g x ∧ g x ≤ M) ∧ m + M = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_of_g_l865_86573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_to_plane_l865_86584

-- Define the types for planes and lines
variable (Plane Line : Type*)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpLine : Line → Line → Prop)
variable (perpLineToPlane : Line → Plane → Prop)

-- Theorem statement
theorem perpendicular_line_to_plane
  (α β : Plane) (m n : Line)
  (h_distinct : α ≠ β)
  (h_perp_planes : perpendicular α β)
  (h_intersect : intersect α β m)
  (h_n_in_α : contains α n)
  (h_perp_lines : perpLine m n) :
  perpLineToPlane n β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_to_plane_l865_86584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_pi_over_two_plus_four_l865_86566

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + Real.sqrt (2 * x - x^2)

-- State the theorem
theorem integral_f_equals_pi_over_two_plus_four :
  ∫ x in (0 : ℝ)..2, f x = π / 2 + 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_pi_over_two_plus_four_l865_86566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_real_iff_a_or_b_zero_l865_86539

theorem complex_square_real_iff_a_or_b_zero (a b : ℝ) :
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  let z : ℂ := Complex.ofReal a + Complex.I * Complex.ofReal b
  (z^2).im = 0 ↔ a = 0 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_real_iff_a_or_b_zero_l865_86539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_expression_l865_86542

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2 / 2) * Real.cos (2 * x + Real.pi / 4) + Real.sin x ^ 2

noncomputable def g (x : ℝ) : ℝ := 1 / 2 - f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_period_and_g_expression :
  (∃ T > 0, is_periodic f T ∧ ∀ S, 0 < S → is_periodic f S → T ≤ S) ∧
  (∀ x, g (x + Real.pi / 2) = g x) ∧
  (∀ x, 0 ≤ x → x ≤ Real.pi / 2 → g x = 1 / 2 - f x) →
  (∃ T > 0, is_periodic f T ∧ ∀ S, 0 < S → is_periodic f S → T ≤ S ∧ T = Real.pi) ∧
  (∀ x, -Real.pi ≤ x → x ≤ 0 → 
    g x = if -Real.pi / 2 ≤ x then -1 / 2 * Real.sin (2 * x) else 1 / 2 * Real.sin (2 * x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_expression_l865_86542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_properties_l865_86597

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 2) + 1

theorem sinusoidal_properties :
  let amplitude : ℝ := 2
  let phase_shift : ℝ := -Real.pi / 8
  let vertical_shift : ℝ := 1
  (∀ x, f x = amplitude * Real.sin (4 * (x - phase_shift)) + vertical_shift) ∧
  (amplitude > 0) ∧
  (phase_shift ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4)) ∧
  (vertical_shift = f 0 - amplitude * Real.sin (-4 * phase_shift)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_properties_l865_86597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l865_86510

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (2^x + b) / (2^x + a)

theorem function_properties (a b : ℝ) :
  (f a b 1 = 1/3 ∧ f a b 0 = 0) →
  (∀ x, f a b x = (2^x - 1) / (2^x + 1)) ∧
  (∀ x, f a b x = -f a b (-x)) ∧
  (∀ m, (∀ x ∈ Set.Icc 0 2, f a b x * (2^x + 1) < m * 4^x) ↔ m > 1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l865_86510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_coins_l865_86560

def possible_one_yuan_coins (total_coins : ℕ) (min_one_yuan : ℕ) (max_value : ℚ) : Set ℕ :=
  {x : ℕ | x ≥ min_one_yuan ∧ 
           x ≤ total_coins ∧ 
           (x : ℚ) + 0.5 * ((total_coins : ℚ) - (x : ℚ)) < max_value}

theorem xiaoming_coins : 
  possible_one_yuan_coins 15 2 10 = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_coins_l865_86560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excenter_incenter_orthocenter_inequality_l865_86588

/-- Predicate stating that a triangle ABC is acute -/
def IsAcute (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

/-- Predicate stating that H is the orthocenter of triangle ABC -/
def IsOrthocenter (H A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

/-- Predicate stating that I is the incenter of triangle ABC -/
def IsIncenter (I A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

/-- Predicate stating that E is an excenter of triangle ABC -/
def IsExcenter (E A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

/-- Given an acute triangle ABC with orthocenter H, incenter I, and excenters I_A, I_B, I_C,
    the product of the distances from I to the excenters is greater than or equal to
    8 times the product of the distances from H to the vertices of the triangle. -/
theorem excenter_incenter_orthocenter_inequality 
  (A B C H I I_A I_B I_C : EuclideanSpace ℝ (Fin 2)) 
  (h1 : IsAcute A B C)
  (h2 : IsOrthocenter H A B C)
  (h3 : IsIncenter I A B C)
  (h4 : IsExcenter I_A A B C)
  (h5 : IsExcenter I_B A B C)
  (h6 : IsExcenter I_C A B C) :
  dist I I_A * dist I I_B * dist I I_C ≥ 8 * dist A H * dist B H * dist C H := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excenter_incenter_orthocenter_inequality_l865_86588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l865_86514

def parabola (x : ℝ) : ℝ := x^2 + 4

def line (r : ℝ) (x : ℝ) : ℝ := r

def triangle_area (r : ℝ) : ℝ := (r - 4)^(3/2)

theorem triangle_area_bounds (r : ℝ) :
  (16 ≤ triangle_area r ∧ triangle_area r ≤ 128) ↔ (8 ≤ r ∧ r ≤ 20) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l865_86514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_three_digit_numbers_less_than_500_l865_86594

/-- The set of digits that can be used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A function that returns true if a number is even, false otherwise -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- A function that returns true if a number is less than 500, false otherwise -/
def lessThan500 (n : Nat) : Bool := n < 500

/-- A function that returns true if a number is a three-digit number, false otherwise -/
def isThreeDigit (n : Nat) : Bool := n ≥ 100 ∧ n ≤ 999

/-- A function to generate all three-digit numbers from the given digits -/
def threeDigitNumbers : Finset Nat :=
  Finset.filter (fun n => isThreeDigit n)
    (Finset.image (fun x => 100 * x.1 + 10 * x.2.1 + x.2.2)
      (digits.product (digits.product digits)))

/-- The main theorem -/
theorem count_even_three_digit_numbers_less_than_500 :
  (Finset.filter (fun n => isEven n ∧ lessThan500 n) threeDigitNumbers).card = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_three_digit_numbers_less_than_500_l865_86594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l865_86551

/-- The fraction of the amount A gets compared to B and C together -/
noncomputable def fraction_A (A B C : ℝ) : ℝ := A / (B + C)

/-- The problem statement -/
theorem problem_solution 
  (A B C : ℝ) 
  (h1 : ∃ x : ℝ, A = x * (B + C))
  (h2 : B = (2/7) * (A + C))
  (h3 : A = B + 30)
  (h4 : A + B + C = 1080) :
  fraction_A A B C = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l865_86551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_l865_86561

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y = 4
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, Real.sin θ)

-- Define the ray l
noncomputable def l (α : ℝ) : ℝ → ℝ × ℝ := λ ρ => (ρ * Real.cos α, ρ * Real.sin α)

-- Define the intersection points A and B
noncomputable def A (α : ℝ) : ℝ × ℝ := (4 / (Real.cos α + Real.sin α) * Real.cos α, 4 / (Real.cos α + Real.sin α) * Real.sin α)
noncomputable def B (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

-- Define the ratio |OB|/|OA|
noncomputable def ratio (α : ℝ) : ℝ := Real.sqrt ((B α).1^2 + (B α).2^2) / Real.sqrt ((A α).1^2 + (A α).2^2)

theorem max_ratio :
  ∃ (α : ℝ), ∀ (β : ℝ), -π/4 < β ∧ β < π/2 → ratio β ≤ (Real.sqrt 2 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_l865_86561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_proof_l865_86586

/-- Calculates the molecular weight of one mole of a compound -/
noncomputable def molecular_weight_per_mole (total_weight : ℝ) (num_moles : ℝ) : ℝ :=
  total_weight / num_moles

/-- Proves that the molecular weight of one mole of the compound is 76 g/mol -/
theorem molecular_weight_proof (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 228)
  (h2 : num_moles = 3) :
  molecular_weight_per_mole total_weight num_moles = 76 := by
  -- Unfold the definition of molecular_weight_per_mole
  unfold molecular_weight_per_mole
  -- Substitute the known values
  rw [h1, h2]
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_proof_l865_86586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l865_86591

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 2*x - 3}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2*x + 13}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-4) 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l865_86591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_4pi_l865_86582

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4*y + 9 = 0

-- Define the area of the circle
noncomputable def circle_area : ℝ := 4 * Real.pi

-- Theorem statement
theorem circle_area_is_4pi :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    circle_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry

#check circle_area_is_4pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_4pi_l865_86582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l865_86521

/-- An equilateral triangle with average side length 12 -/
structure EquilateralTriangle where
  side_length : ℝ
  average_length_eq : side_length = 12

/-- The perimeter of an equilateral triangle -/
noncomputable def perimeter (t : EquilateralTriangle) : ℝ :=
  3 * t.side_length

/-- The area of an equilateral triangle -/
noncomputable def area (t : EquilateralTriangle) : ℝ :=
  (Real.sqrt 3 / 4) * t.side_length ^ 2

theorem equilateral_triangle_properties (t : EquilateralTriangle) :
  perimeter t = 36 ∧ area t = 36 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l865_86521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angleCalculation_equationSolution_l865_86501

-- Part 1: Angle calculation
def degreesToSeconds (d : ℕ) (m : ℕ) (s : ℕ) : ℕ := d * 3600 + m * 60 + s

def secondsToDegreeMinutes (s : ℕ) : ℕ × ℕ :=
  let degrees := s / 3600
  let remainingSeconds := s % 3600
  let minutes := remainingSeconds / 60
  (degrees, minutes)

theorem angleCalculation :
  let angle1 := degreesToSeconds 34 25 20
  let angle2 := degreesToSeconds 35 42 0
  let result := secondsToDegreeMinutes (3 * angle1 + angle2)
  result = (138, 58) := by sorry

-- Part 2: Equation solving
theorem equationSolution (x : ℚ) :
  (x + 1) / 2 - 1 = (2 - 3 * x) / 3 → x = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angleCalculation_equationSolution_l865_86501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_diagonal_l865_86562

/-- A rectangular solid with edge lengths a, b, and c. -/
structure RectangularSolid where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The total surface area of a rectangular solid. -/
def totalSurfaceArea (s : RectangularSolid) : ℝ :=
  2 * (s.a * s.b + s.b * s.c + s.a * s.c)

/-- The total length of all edges of a rectangular solid. -/
def totalEdgeLength (s : RectangularSolid) : ℝ :=
  4 * (s.a + s.b + s.c)

/-- The length of an interior diagonal of a rectangular solid. -/
noncomputable def interiorDiagonal (s : RectangularSolid) : ℝ :=
  Real.sqrt (s.a^2 + s.b^2 + s.c^2)

theorem rectangular_solid_diagonal (s : RectangularSolid) 
  (h1 : totalSurfaceArea s = 34)
  (h2 : totalEdgeLength s = 28) :
  interiorDiagonal s = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_diagonal_l865_86562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_y_intercept_of_given_line_l865_86567

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
noncomputable def y_intercept (a b c : ℝ) : ℝ := -c / b

/-- The line equation is in the form ax + by + c = 0 -/
theorem y_intercept_of_line (a b c : ℝ) (hb : b ≠ 0) :
  y_intercept a b c = -c / b := by
  sorry

theorem y_intercept_of_given_line :
  y_intercept 1 (-1) 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_y_intercept_of_given_line_l865_86567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_in_Al2O3_l865_86500

-- Define the atomic masses
noncomputable def atomic_mass_Al : ℝ := 26.98
noncomputable def atomic_mass_O : ℝ := 16.00

-- Define the number of atoms in Al2O3
def num_Al_atoms : ℕ := 2
def num_O_atoms : ℕ := 3

-- Define the molar mass of Al2O3
noncomputable def molar_mass_Al2O3 : ℝ := num_Al_atoms * atomic_mass_Al + num_O_atoms * atomic_mass_O

-- Define the mass of Al in Al2O3
noncomputable def mass_Al_in_Al2O3 : ℝ := num_Al_atoms * atomic_mass_Al

-- Define the mass percentage calculation
noncomputable def mass_percentage (mass_element : ℝ) (total_mass : ℝ) : ℝ := (mass_element / total_mass) * 100

-- Theorem statement
theorem mass_percentage_Al_in_Al2O3 :
  abs (mass_percentage mass_Al_in_Al2O3 molar_mass_Al2O3 - 52.91) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_in_Al2O3_l865_86500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l865_86543

-- Define the hyperbola equation
noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y - 1)^2) - Real.sqrt ((x + 3)^2 + (y - 1)^2) = 4

-- Define the positive slope of the asymptote
noncomputable def positive_asymptote_slope : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem hyperbola_asymptote_slope :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ positive_asymptote_slope = 2 * Real.sqrt 2 := by
  sorry

#check hyperbola_asymptote_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l865_86543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l865_86513

/-- Given a line with equation 3x - 4y = 12, prove its y-intercept and slope -/
theorem line_properties :
  let line := {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 = 12}
  (∃ y : ℝ, (0, y) ∈ line ∧ y = -3) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ line → (x₂, y₂) ∈ line → x₁ ≠ x₂ →
    (y₂ - y₁) / (x₂ - x₁) = 3 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l865_86513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_largest_least_l865_86538

def digits : List Nat := [9, 2, 1, 5]

def largest_number (digits : List Nat) : Nat :=
  digits.toArray.qsort (fun a b => b < a)
    |> Array.foldl (fun acc d => acc * 10 + d) 0

def least_number (digits : List Nat) : Nat :=
  digits.toArray.qsort (fun a b => a < b)
    |> Array.foldl (fun acc d => acc * 10 + d) 0

theorem difference_largest_least :
  largest_number digits - least_number digits = 8262 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_largest_least_l865_86538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_on_interval_l865_86581

-- Define the function f
def f (x : ℝ) : ℝ := 8 + 2*x - x^2

-- Define the function g in terms of f
def g (x : ℝ) : ℝ := f (2 - x^2)

-- Theorem statement
theorem g_decreasing_on_interval : 
  ∀ x y, x ∈ Set.Ioo (-1 : ℝ) 0 → y ∈ Set.Ioo (-1 : ℝ) 0 → x < y → g y < g x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_on_interval_l865_86581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_ratio_bounds_l865_86570

open Set Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Ioi 0
  pos : ∀ x ∈ domain, f x > 0
  deriv_bounds : ∀ x ∈ domain, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x

/-- The main theorem stating the bounds for f(1)/f(2) -/
theorem special_function_ratio_bounds (φ : SpecialFunction) :
  1/8 < φ.f 1 / φ.f 2 ∧ φ.f 1 / φ.f 2 < 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_ratio_bounds_l865_86570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_13_l865_86502

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola with equation y^2 = 4x -/
def Parabola : Type := Unit

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

theorem hyperbola_eccentricity_sqrt_13 (h : Hyperbola) (p : Parabola) :
  triangle_area 1 ((2 * h.b) / h.a) = 2 * Real.sqrt 3 →
  eccentricity h = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_13_l865_86502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_equation_l865_86579

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem two_solutions_equation :
  ∃ (s : Finset ℝ), s.card = 2 ∧ ∀ x : ℝ, x ∈ s ↔ x^2 = 2 * (floor x) + 1 :=
by
  -- The proof will be skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_equation_l865_86579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_x_l865_86572

/-- The definite integral of √(1-(x-1)^2)-x from 0 to 1 is equal to π/4 - 1/2 -/
theorem integral_sqrt_minus_x : 
  ∫ x in Set.Icc 0 1, (Real.sqrt (1 - (x - 1)^2) - x) = π/4 - 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_x_l865_86572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_for_given_spinner_l865_86558

/-- Represents a spinner with two sectors -/
structure Spinner where
  white_angle : ℝ
  red_angle : ℝ

/-- Calculates the probability of landing on a sector given its central angle -/
noncomputable def sector_probability (angle : ℝ) : ℝ :=
  angle / 360

/-- Calculates the probability of landing on white once and red once in two spins -/
noncomputable def probability_white_once_red_once (s : Spinner) : ℝ :=
  2 * (sector_probability s.white_angle * sector_probability s.red_angle)

/-- Theorem stating the probability of landing on white once and red once for the given spinner -/
theorem probability_for_given_spinner :
  let s : Spinner := ⟨120, 240⟩
  probability_white_once_red_once s = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_for_given_spinner_l865_86558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l865_86578

noncomputable def m (α : Real) : Fin 2 → Real := λ i => match i with
  | 0 => Real.sin α
  | 1 => Real.cos α

noncomputable def n (α : Real) : Fin 2 → Real := λ i => match i with
  | 0 => Real.cos α
  | 1 => Real.cos α

def collinear (u v : Fin 2 → Real) : Prop :=
  ∃ (k : Real), ∀ (i : Fin 2), u i = k * v i

def perpendicular (u v : Fin 2 → Real) : Prop :=
  (u 0 * v 0 + u 1 * v 1) = 0

theorem vector_properties (α : Real) (h : 0 ≤ α ∧ α ≤ Real.pi) :
  (collinear (m α) (n α) ↔ α = Real.pi / 2 ∨ α = Real.pi / 4) ∧
  ¬∃ (α : Real), 0 ≤ α ∧ α ≤ Real.pi ∧ perpendicular (m α) (λ i => m α i + n α i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l865_86578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_is_six_l865_86548

/-- Represents the fuel efficiency of a car in different driving conditions -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℚ
  city_miles_per_tankful : ℚ
  city_miles_per_gallon : ℚ

/-- Calculates the difference in miles per gallon between highway and city driving -/
def mpg_difference (car : CarFuelEfficiency) : ℚ :=
  let tankful_gallons := car.city_miles_per_tankful / car.city_miles_per_gallon
  let highway_mpg := car.highway_miles_per_tankful / tankful_gallons
  highway_mpg - car.city_miles_per_gallon

/-- Theorem stating the difference in miles per gallon between highway and city driving is 6 -/
theorem mpg_difference_is_six (car : CarFuelEfficiency)
    (h1 : car.highway_miles_per_tankful = 420)
    (h2 : car.city_miles_per_tankful = 336)
    (h3 : car.city_miles_per_gallon = 24) :
    mpg_difference car = 6 := by
  sorry

#eval mpg_difference { highway_miles_per_tankful := 420, city_miles_per_tankful := 336, city_miles_per_gallon := 24 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_is_six_l865_86548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_calculations_l865_86563

/-- Represents the paint bucket types -/
inductive BucketType
| Large
| Small

/-- Defines the properties of a paint bucket -/
structure Bucket where
  type : BucketType
  volume : ℝ
  price : ℝ

/-- Defines the promotion rules -/
structure Promotion where
  discount_threshold : ℝ
  discount_amount : ℝ
  buy_quantity : ℕ
  free_quantity : ℕ

/-- Defines the problem parameters -/
structure PaintProblem where
  large_bucket : Bucket
  small_bucket : Bucket
  large_bucket_remainder : ℝ
  small_bucket_extra : ℕ
  small_bucket_remainder : ℝ
  promotion : Promotion
  profit_margin : ℝ

theorem paint_calculations 
  (p : PaintProblem) 
  (h_large : p.large_bucket.type = BucketType.Large ∧ p.large_bucket.volume = 18 ∧ p.large_bucket.price = 225)
  (h_small : p.small_bucket.type = BucketType.Small ∧ p.small_bucket.volume = 5 ∧ p.small_bucket.price = 90)
  (h_large_remainder : p.large_bucket_remainder = 2)
  (h_small_extra : p.small_bucket_extra = 11)
  (h_small_remainder : p.small_bucket_remainder = 1)
  (h_promotion : p.promotion.discount_threshold = 1000 ∧ p.promotion.discount_amount = 120 ∧ 
                 p.promotion.buy_quantity = 4 ∧ p.promotion.free_quantity = 1)
  (h_profit : p.profit_margin = 0.25)
  : 
  (∃ (total_volume : ℝ), total_volume = 74) ∧ 
  (∃ (savings : ℝ), savings = 390) ∧
  (∃ (cost_per_small : ℝ), cost_per_small = 51.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_calculations_l865_86563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_majors_consecutive_probability_l865_86557

/-- The number of people sitting around the table -/
def total_people : ℕ := 10

/-- The number of physics majors -/
def physics_majors : ℕ := 3

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of biology majors -/
def biology_majors : ℕ := 2

/-- The probability of all physics majors sitting in consecutive seats -/
def consecutive_physics_probability : ℚ := 1 / 12

theorem physics_majors_consecutive_probability :
  (total_people * Nat.factorial physics_majors * Nat.factorial (total_people - physics_majors) : ℚ) / 
  Nat.factorial total_people = consecutive_physics_probability := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_majors_consecutive_probability_l865_86557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l865_86524

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_properties (a₁ d : ℝ) :
  /- 1. The sequence {(1/2)^(a_n)} is a geometric sequence -/
  (∃ r : ℝ, ∀ n : ℕ, (1/2)^(arithmetic_sequence a₁ d (n+1)) = r * (1/2)^(arithmetic_sequence a₁ d n)) ∧
  /- 2. If a₁₀ = 3 and S₇ = -7, then S₁₃ = 13 -/
  (arithmetic_sequence a₁ d 10 = 3 ∧ arithmetic_sum a₁ d 7 = -7 → arithmetic_sum a₁ d 13 = 13) ∧
  /- 3. S_n = na_n - (n(n-1)/2)d -/
  (∀ n : ℕ, arithmetic_sum a₁ d n = n * arithmetic_sequence a₁ d n - (n * (n - 1) : ℝ) / 2 * d) ∧
  /- 4. If d > 0, S_n does not always have a maximum value -/
  (d > 0 → ¬∃ M : ℝ, ∀ n : ℕ, arithmetic_sum a₁ d n ≤ M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l865_86524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_19_is_valid_l865_86580

def candidates : List Nat := [16, 18, 19, 21]

def is_valid (n : Nat) : Prop :=
  n < 20 ∧ ¬(n % 2 = 0)

theorem only_19_is_valid : 
  ∃! x, x ∈ candidates ∧ is_valid x ∧ x = 19 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_19_is_valid_l865_86580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_l865_86511

/-- The slope of a line PQ intersecting an ellipse under specific conditions -/
theorem ellipse_intersection_slope :
  ∀ (Γ : Set (ℝ × ℝ)) (F₁ F₂ P Q : ℝ × ℝ),
  (∀ (x y : ℝ), (x, y) ∈ Γ ↔ x^2/2 + y^2 = 1) →  -- Ellipse equation
  (∃ (a : ℝ), F₁ = (-a, 0) ∧ F₂ = (a, 0) ∧ a > 0) →  -- Foci positions
  (1, 0) ∈ Γ →  -- Point (1,0) is on the ellipse
  P ∈ Γ ∧ Q ∈ Γ →  -- P and Q are on the ellipse
  (∃ (k b b' : ℝ), P.2 = k * P.1 + b ∧ Q.2 = -1/k * Q.1 + b') →  -- Perpendicular lines through (1,0)
  ((P.1 - F₁.1, P.2 - F₁.2).1 + (P.1 - F₂.1, P.2 - F₂.2).1) * ((Q.1 - F₁.1, Q.2 - F₁.2).1 + (Q.1 - F₂.1, Q.2 - F₂.2).1) +
  ((P.1 - F₁.1, P.2 - F₁.2).2 + (P.1 - F₂.1, P.2 - F₂.2).2) * ((Q.1 - F₁.1, Q.2 - F₁.2).2 + (Q.1 - F₂.1, Q.2 - F₂.2).2) = 0 →  -- Perpendicular condition
  ∃ (k : ℝ), k^2 = (-5 + 2 * Real.sqrt 10) / 10 ∧ 
  (Q.2 - P.2) / (Q.1 - P.1) = k ∨ (Q.2 - P.2) / (Q.1 - P.1) = -k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_l865_86511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l865_86523

theorem cos_difference (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = -4/5)
  (h2 : Real.sin α + Real.sin β = 1/3) : 
  Real.cos (α - β) = -28/225 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l865_86523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_fixed_point_5_f_cycle_8_main_result_l865_86540

def f : ℕ → ℕ
  | 0 => 9  -- Adding case for 0
  | 1 => 9
  | 2 => 8
  | 3 => 7
  | 4 => 6
  | 5 => 5
  | 6 => 1
  | 7 => 2
  | 8 => 3
  | n + 9 => f n

def iterate_f (n : ℕ) : ℕ → ℕ
  | 0 => n
  | m + 1 => f (iterate_f n m)

theorem f_fixed_point_5 (m : ℕ) : iterate_f 5 m = 5 := by
  sorry

theorem f_cycle_8 (m : ℕ) : 
  iterate_f 8 m = 7 ∨ iterate_f 8 m = 2 ∨ iterate_f 8 m = 8 ∨ iterate_f 8 m = 3 := by
  sorry

theorem main_result (m : ℕ) : 
  (5 * (iterate_f 5 m) + 2 * (iterate_f 8 m)) ∈ ({39, 29, 41, 31} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_fixed_point_5_f_cycle_8_main_result_l865_86540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_activity3_l865_86531

def total_time : ℝ := 5
def time_activity1 : ℝ := 2
def time_activity2 : ℝ := 1.5

theorem time_activity3 : total_time - time_activity1 - time_activity2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_activity3_l865_86531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_nine_equals_two_six_three_three_times_210_l865_86577

theorem factorial_nine_equals_two_six_three_three_times_210 : 
  2^6 * 3^3 * 210 = Nat.factorial 9 := by
  sorry

#check factorial_nine_equals_two_six_three_three_times_210

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_nine_equals_two_six_three_three_times_210_l865_86577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sin_has_property_T_l865_86569

-- Define Property T
def has_property_T (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (deriv f x₁) * (deriv f x₂) = -1

-- Define the four functions
noncomputable def f₁ : ℝ → ℝ := λ x => Real.sin x
noncomputable def f₂ : ℝ → ℝ := λ x => Real.log x
noncomputable def f₃ : ℝ → ℝ := λ x => Real.exp x
def f₄ : ℝ → ℝ := λ x => x^3

-- Theorem statement
theorem only_sin_has_property_T :
  has_property_T f₁ ∧
  ¬has_property_T f₂ ∧
  ¬has_property_T f₃ ∧
  ¬has_property_T f₄ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sin_has_property_T_l865_86569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_satisfies_conditions_principal_amount_approximation_l865_86593

/-- The principal amount that satisfies the given conditions -/
def principal_amount : ℝ := 3750

/-- The interest rate per annum -/
def interest_rate : ℝ := 4

/-- The time period in years -/
def time_period : ℝ := 2

/-- The difference between compound interest and simple interest amounts -/
def interest_difference : ℝ := 6.000000000000455

/-- Theorem stating that the principal amount satisfies the given conditions -/
theorem principal_amount_satisfies_conditions :
  let compound_interest := principal_amount * (1 + interest_rate / 100) ^ time_period
  let simple_interest := principal_amount * (1 + interest_rate * time_period / 100)
  abs (compound_interest - simple_interest - interest_difference) < 1e-6 := by sorry

/-- Theorem stating that the principal amount is approximately 3750 -/
theorem principal_amount_approximation :
  abs (principal_amount - 3750) < 1e-6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_satisfies_conditions_principal_amount_approximation_l865_86593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_scalar_l865_86508

/-- Given a line passing through points (-3, 6) and (2, -1) with direction vector (a, -2), prove that a = 10/7 -/
theorem direction_vector_scalar (a : ℚ) : 
  let p1 : ℚ × ℚ := (-3, 6)
  let p2 : ℚ × ℚ := (2, -1)
  let direction : ℚ × ℚ := (p2.1 - p1.1, p2.2 - p1.2)
  (∃ (k : ℚ), (k * direction.1, k * direction.2) = (a, -2)) → a = 10/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_scalar_l865_86508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l865_86519

/-- Represents the score distribution of students -/
structure ScoreDistribution where
  score72 : ℝ
  score82 : ℝ
  score87 : ℝ
  score91 : ℝ
  score96 : ℝ

/-- Calculates the mean score given a score distribution -/
noncomputable def meanScore (d : ScoreDistribution) : ℝ :=
  (72 * d.score72 + 82 * d.score82 + 87 * d.score87 + 91 * d.score91 + 96 * d.score96) /
  (d.score72 + d.score82 + d.score87 + d.score91 + d.score96)

/-- Calculates the median score given a score distribution -/
noncomputable def medianScore (d : ScoreDistribution) : ℝ :=
  if d.score72 + d.score82 > (1/2 : ℝ) then 82
  else if d.score72 + d.score82 + d.score87 > (1/2 : ℝ) then 87
  else if d.score72 + d.score82 + d.score87 + d.score91 > (1/2 : ℝ) then 91
  else 96

/-- Theorem stating the difference between mean and median score -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score72 = 0.12)
  (h2 : d.score82 = 0.30)
  (h3 : d.score87 = 0.18)
  (h4 : d.score91 = 0.10)
  (h5 : d.score96 = 1 - (d.score72 + d.score82 + d.score87 + d.score91)) :
  meanScore d - medianScore d = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l865_86519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_equals_64_l865_86596

def S : ℕ → ℤ
  | 0 => 2
  | n + 1 => 2 * S n - 1

def a : ℕ → ℤ
  | 0 => 2
  | n + 1 => S (n + 1) - S n

theorem a_8_equals_64 : a 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_equals_64_l865_86596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l865_86533

open Real

-- Define the function f(x) = x / (1 - cos x)
noncomputable def f (x : ℝ) : ℝ := x / (1 - cos x)

-- State the theorem about the derivative of f
theorem derivative_of_f (x : ℝ) (h : 1 - cos x ≠ 0) :
  deriv f x = (1 - cos x - x * sin x) / (1 - cos x)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l865_86533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_integers_with_five_or_seven_l865_86506

/-- The set of digits excluding 5 and 7 -/
def digitsExcluding5And7 : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}

/-- The set of non-zero digits excluding 5 and 7 -/
def nonZeroDigitsExcluding5And7 : Finset Nat := {1, 2, 3, 4, 6, 8, 9}

/-- A four-digit positive integer -/
structure FourDigitInteger where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  h1 : d1 ∈ nonZeroDigitsExcluding5And7 ∪ {5, 7}
  h2 : d2 ∈ digitsExcluding5And7 ∪ {5, 7}
  h3 : d3 ∈ digitsExcluding5And7 ∪ {5, 7}
  h4 : d4 ∈ digitsExcluding5And7 ∪ {5, 7}

/-- The set of all four-digit positive integers -/
def allFourDigitIntegers : Finset FourDigitInteger := sorry

/-- The set of four-digit positive integers with at least one 5 or 7 -/
def fourDigitIntegersWithFiveOrSeven : Finset FourDigitInteger :=
  allFourDigitIntegers.filter (fun n => n.d1 = 5 ∨ n.d1 = 7 ∨ n.d2 = 5 ∨ n.d2 = 7 ∨ n.d3 = 5 ∨ n.d3 = 7 ∨ n.d4 = 5 ∨ n.d4 = 7)

theorem count_four_digit_integers_with_five_or_seven :
  Finset.card fourDigitIntegersWithFiveOrSeven = 5416 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_integers_with_five_or_seven_l865_86506
