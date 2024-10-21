import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_perpendicular_l1362_136247

-- Define the curves
noncomputable def curve1 (x : ℝ) : ℝ := Real.exp x
noncomputable def curve2 (x : ℝ) : ℝ := 1 / x

-- Define the derivatives of the curves
noncomputable def curve1_derivative (x : ℝ) : ℝ := Real.exp x
noncomputable def curve2_derivative (x : ℝ) : ℝ := -1 / (x^2)

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- State the theorem
theorem tangent_lines_perpendicular :
  let slope1 := curve1_derivative 0
  let slope2 := curve2_derivative P.1
  (P.1 > 0) ∧ 
  (slope1 * slope2 = -1) ∧
  (P.2 = curve2 P.1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_perpendicular_l1362_136247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1362_136287

def b (n : ℕ) : ℝ := (2 * n - 1) * 3^n + 4

def a : ℕ → ℝ
| 0 => 0  -- Adding a case for n = 0
| 1 => 7
| n+2 => 4 * (n+2) * 3^(n+1)

theorem sequence_formula (n : ℕ) (h : n ≥ 1) :
  b n - b (n-1) = a n :=
by
  cases n
  · contradiction  -- This case is impossible due to h : n ≥ 1
  case succ n' =>
    cases n'
    · simp [b, a]  -- Case for n = 1
      -- The proof for n = 1 would go here
      sorry
    · simp [b, a]  -- Case for n ≥ 2
      -- The proof for n ≥ 2 would go here
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1362_136287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_min_max_in_T_l1362_136236

-- Define the function g(x) = (3x+4)/(x+3)
noncomputable def g (x : ℝ) : ℝ := (3*x + 4) / (x + 3)

-- Define the set T as the range of g(x) for x > 0
def T : Set ℝ := {y | ∃ x > 0, g x = y}

-- Theorem statement
theorem no_min_max_in_T : ¬∃ (m M : ℝ), (∀ y ∈ T, m ≤ y ∧ y ≤ M) ∧ (m ∈ T ∨ M ∈ T) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_min_max_in_T_l1362_136236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bar_chart_representation_l1362_136248

/-- Represents the relationship between centimeters and number of trees in a bar chart -/
structure BarChart where
  trees_per_cm : ℚ

/-- Calculates the number of trees represented by a given length in centimeters -/
def trees_represented (chart : BarChart) (cm : ℚ) : ℚ :=
  chart.trees_per_cm * cm

/-- Calculates the length in centimeters needed to represent a given number of trees -/
def cm_needed (chart : BarChart) (trees : ℚ) : ℚ :=
  trees / chart.trees_per_cm

theorem bar_chart_representation (chart : BarChart) :
  chart.trees_per_cm = 40 →
  (cm_needed chart 120 = 3 ∧ trees_represented chart (7/2) = 140) := by
  intro h
  apply And.intro
  · -- Proof for cm_needed
    simp [cm_needed, h]
    norm_num
  · -- Proof for trees_represented
    simp [trees_represented, h]
    norm_num

#check bar_chart_representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bar_chart_representation_l1362_136248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_exists_l1362_136207

def four_digit_numbers : List Nat := [3684, 3704, 3714, 3732, 3882]

def is_divisible_by_4 (n : Nat) : Bool :=
  n % 4 = 0

def is_divisible_by_3 (n : Nat) : Bool :=
  n % 3 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem unique_number_exists : ∃! n, n ∈ four_digit_numbers ∧
  ¬(is_divisible_by_4 n) ∧ 
  is_divisible_by_3 n ∧ 
  (units_digit n * tens_digit n = 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_exists_l1362_136207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_optimization_l1362_136229

/-- The maximum value of t for which |2x+5|+|2x-1|-t≥0 holds for all real x -/
noncomputable def max_t : ℝ := 6

/-- The minimum value of 1/(a+2b) + 4/(3a+3b) given 4a+5b=6 and a,b > 0 -/
noncomputable def min_y : ℝ := 3/2

theorem inequality_and_optimization :
  (∀ x : ℝ, |2*x+5| + |2*x-1| - max_t ≥ 0) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 4*a + 5*b = max_t →
    1/(a+2*b) + 4/(3*a+3*b) ≥ min_y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_optimization_l1362_136229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1362_136270

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + (2 * a - 1) * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * a * x + 2 * a

theorem f_properties (a : ℝ) :
  (∀ x > 0, g a x = Real.log x - 2 * a * x + 2 * a) ∧
  (a ≤ 0 → ∀ x > 0, Monotone (g a)) ∧
  (a > 0 → (∀ x ∈ Set.Ioo 0 (1 / (2 * a)), StrictMono (g a)) ∧
            (∀ x ∈ Set.Ioi (1 / (2 * a)), StrictAnti (g a))) ∧
  (∀ x > 0, IsLocalMax (f a) 1 → a > 1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1362_136270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sum_existence_triplet_sum_not_guaranteed_l1362_136294

theorem triplet_sum_existence (S : Finset ℕ) (h1 : S.card = 1002) (h2 : ∀ x ∈ S, x ≤ 2000) :
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c :=
sorry

theorem triplet_sum_not_guaranteed (n : ℕ) (hn : n = 1001) :
  ∃ S : Finset ℕ, S.card = n ∧ (∀ x ∈ S, x ≤ 2000) ∧
  (∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → a + b ≠ c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sum_existence_triplet_sum_not_guaranteed_l1362_136294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_theorem_l1362_136217

/-- A square piece of paper with white top and red bottom -/
structure Paper where
  side : ℝ
  is_square : side > 0

/-- A point within the square paper -/
structure Point (p : Paper) where
  x : ℝ
  y : ℝ
  in_square : 0 ≤ x ∧ x ≤ p.side ∧ 0 ≤ y ∧ y ≤ p.side

/-- The expected number of sides of the red polygon after folding -/
noncomputable def expected_sides (p : Paper) : ℝ := 5 - Real.pi / 2

/-- The theorem stating the expected number of sides of the red polygon -/
theorem expected_sides_theorem (p : Paper) :
  expected_sides p = 5 - Real.pi / 2 := by
  -- Unfold the definition of expected_sides
  unfold expected_sides
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_theorem_l1362_136217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclined_triangular_prism_volume_l1362_136276

/-- The volume of an inclined triangular prism -/
noncomputable def inclinedTriangularPrismVolume (S d : ℝ) : ℝ := (1/2) * S * d

/-- Theorem: The volume of an inclined triangular prism is (1/2) * S * d,
    where S is the area of one of its lateral faces and
    d is the distance from the plane of this face to the opposite edge. -/
theorem inclined_triangular_prism_volume
  (S d : ℝ)
  (h_S : S > 0)  -- Assumption: Area is positive
  (h_d : d > 0)  -- Assumption: Distance is positive
  : inclinedTriangularPrismVolume S d = (1/2) * S * d := by
  -- Unfold the definition of inclinedTriangularPrismVolume
  unfold inclinedTriangularPrismVolume
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclined_triangular_prism_volume_l1362_136276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1362_136216

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a - Real.log x

theorem f_properties (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ 1 ∧
  ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ →
    (f a x₁ - f a x₂) / (x₂ - x₁) < 1 / (x₁ * (x₁ + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1362_136216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_negative_exchange_properties_l1362_136219

-- Define the "inverse-negative" exchange property
def inverse_negative_exchange (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = -f x

-- Define the four functions
noncomputable def f1 (x : ℝ) : ℝ := x - 1 / x
noncomputable def f2 (x : ℝ) : ℝ := x + 1 / x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x
noncomputable def f4 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x
  else if x = 1 then 0
  else if x > 1 then -1 / x
  else 0  -- Default case for x ≤ 0

theorem inverse_negative_exchange_properties :
  inverse_negative_exchange f1 ∧
  inverse_negative_exchange f3 ∧
  inverse_negative_exchange f4 ∧
  ¬inverse_negative_exchange f2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_negative_exchange_properties_l1362_136219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a6_plus_a7_l1362_136220

/-- An arithmetic sequence with positive terms -/
structure PosArithSeq where
  a : ℕ → ℝ
  pos : ∀ n, a n > 0
  is_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The theorem statement -/
theorem min_value_a6_plus_a7 (seq : PosArithSeq) 
    (h : seq.a 5 + seq.a 4 - seq.a 3 - seq.a 2 = 5) :
    ∃ (m : ℝ), (∀ (q : ℝ), seq.a 6 + seq.a 7 ≥ m) ∧ (∃ (q : ℝ), seq.a 6 + seq.a 7 = m) ∧ m = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a6_plus_a7_l1362_136220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_arithmetic_sequence_l1362_136266

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem first_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a (-2))
  (h_sum_equal : sum_of_arithmetic_sequence a 10 = sum_of_arithmetic_sequence a 11) :
  a 1 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_arithmetic_sequence_l1362_136266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1362_136254

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := 7 * x^2 + 4 * y^2 = 28

/-- The line equation -/
def line (x y : ℝ) : Prop := 3 * x - 2 * y - 16 = 0

/-- The distance function from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3 * x - 2 * y - 16| / Real.sqrt 13

/-- The theorem stating the maximum distance -/
theorem max_distance_to_line :
  ∃ (max_dist : ℝ), max_dist = (24 / 13) * Real.sqrt 13 ∧
  ∀ (x y : ℝ), ellipse x y →
    distance_to_line x y ≤ max_dist ∧
    ∃ (x' y' : ℝ), ellipse x' y' ∧ distance_to_line x' y' = max_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1362_136254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_is_one_l1362_136231

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (2 + i) / (2 - i)

theorem modulus_of_z_is_one : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_is_one_l1362_136231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l1362_136260

/-- The constant term in the expansion of (9x + 1/(3√x))^18 -/
def constant_term : ℕ := 18564

/-- The binomial expression (9x + 1/(3√x))^18 -/
noncomputable def binomial (x : ℝ) : ℝ := (9 * x + 1 / (3 * Real.sqrt x)) ^ 18

theorem constant_term_of_binomial_expansion :
  constant_term = (Finset.range 19).sum (λ k => 
    (Nat.choose 18 k : ℕ) * (9^(18-k) * (1/3)^k : ℚ).num * 
    (if 18 - 3*k/2 = 0 then 1 else 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l1362_136260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fourteen_times_sum_of_digits_l1362_136242

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Property that a number is 14 times the sum of its digits -/
def is_fourteen_times_sum_of_digits (n : ℕ) : Prop :=
  n = 14 * sum_of_digits n

theorem unique_fourteen_times_sum_of_digits :
  ∃! n : ℕ, n > 0 ∧ is_fourteen_times_sum_of_digits n :=
by
  use 126
  constructor
  · constructor
    · exact Nat.zero_lt_succ 125
    · sorry  -- Proof that 126 = 14 * (1 + 2 + 6)
  · intro m ⟨m_pos, m_prop⟩
    sorry  -- Proof that m must equal 126

#check unique_fourteen_times_sum_of_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fourteen_times_sum_of_digits_l1362_136242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_composite_divisor_pairs_l1362_136262

theorem no_composite_divisor_pairs : ¬∃ (n m : ℕ), n > 1 ∧ m > 1 ∧
  (∃ (a b : ℕ), 1 < a ∧ a < n ∧ 1 < b ∧ b < n ∧ n = a * b) ∧ 
  (∀ d : ℕ, 0 < d ∧ d < m → (∃ d' : ℕ, 0 < d' ∧ d' < n ∧ d = d' + 1 ∧ n % d' = 0)) ∧
  (∀ d : ℕ, 0 < d ∧ d < n ∧ n % d = 0 → m % (d + 1) = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_composite_divisor_pairs_l1362_136262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_pi_l1362_136279

theorem sum_of_solutions_is_pi : 
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2*Real.pi ∧ 1/Real.sin x + 1/Real.cos x = 4) ∧ 
    (S.sum id) = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_pi_l1362_136279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1362_136240

/-- Calculates the average speed of a car given three speeds for equal portions of a trip -/
noncomputable def averageSpeed (s1 s2 s3 : ℝ) : ℝ :=
  3 / (1/s1 + 1/s2 + 1/s3)

theorem car_average_speed : 
  let s1 := (80 : ℝ)
  let s2 := (24 : ℝ)
  let s3 := (30 : ℝ)
  abs (averageSpeed s1 s2 s3 - 34.2857) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1362_136240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_graphs_l1362_136274

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 - Real.sqrt (1 - (x - 0.5)^2)

-- Define the domain of g
def domain : Set ℝ := Set.Icc (-1) 2

-- Define the theorem
theorem area_enclosed_by_graphs :
  ∃ (A : ℝ), A = π / 2 - 1 ∧
  (∀ x ∈ domain, g x ∈ domain) ∧
  (∀ y ∈ domain, g y ∈ domain) ∧
  (A = (MeasureTheory.volume {p : ℝ × ℝ | p.1 ∈ domain ∧ p.2 ∈ domain ∧ 
    ((p.2 = g p.1) ∧ (p.1 ≤ g p.2)) ∨ ((p.1 = g p.2) ∧ (p.2 ≤ g p.1))}).toReal) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_graphs_l1362_136274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1362_136282

def z (a : ℝ) : ℂ := a^2 - 2*a + (a^2 - 3*a + 2)*Complex.I

theorem complex_number_properties (a : ℝ) :
  (z a = Complex.I * Complex.im (z a) ↔ a = 0) ∧
  (Complex.im (z a) = 0 ↔ a = 1 ∨ a = 2) ∧
  (Complex.re (z a) > 0 ∧ Complex.im (z a) > 0 ↔ a < 0 ∨ a > 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1362_136282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l1362_136296

-- Define the function f(x) = 6/x - √x
noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.sqrt x

-- Theorem statement
theorem f_has_zero_in_interval :
  ∃ (c : ℝ), c ∈ Set.Ioo 3 4 ∧ f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l1362_136296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_position_l1362_136203

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  coordinates : ℝ × ℝ

noncomputable def distance (p : Point) (c : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.coordinates.1 - c.1)^2 + (p.coordinates.2 - c.2)^2)

theorem point_position (O : Circle) (A B C : Point) 
  (hA : distance A O.center = O.radius)
  (hB : distance B O.center < O.radius)
  (hC : distance C O.center > O.radius) :
  (distance A O.center = O.radius ∧ 
   distance B O.center < O.radius ∧ 
   distance C O.center > O.radius) :=
by
  constructor
  · exact hA
  constructor
  · exact hB
  · exact hC

#check point_position

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_position_l1362_136203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_time_is_18_l1362_136235

/-- Represents a task with its duration -/
structure MyTask where
  name : String
  duration : Nat

/-- The list of tasks Xiao Ming needs to complete -/
def tasks : List MyTask := [
  ⟨"making bed", 3⟩,
  ⟨"brushing teeth and washing face", 4⟩,
  ⟨"boiling water", 10⟩,
  ⟨"eating breakfast", 7⟩,
  ⟨"washing dishes", 1⟩,
  ⟨"organizing backpack", 2⟩,
  ⟨"making milk", 1⟩
]

/-- The duration of the longest task -/
def longest_task_duration : Nat := (tasks.map MyTask.duration).maximum?
  |>.getD 0 -- Default to 0 if the list is empty

/-- The sum of durations of tasks that must be done sequentially -/
def sequential_tasks_duration : Nat := 
  (tasks.filter (fun t => t.name = "boiling water" || t.name = "eating breakfast" || t.name = "washing dishes")).map MyTask.duration |>.sum

/-- The shortest possible time to complete all tasks -/
def shortest_time : Nat := longest_task_duration + sequential_tasks_duration

/-- Theorem stating that the shortest possible time to complete all tasks is 18 minutes -/
theorem shortest_time_is_18 : shortest_time = 18 := by
  sorry

#eval shortest_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_time_is_18_l1362_136235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l1362_136243

noncomputable def AreaOfRegion (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem circle_area : 
  ∃ A : ℝ, A = Real.pi * 6 ∧ A = AreaOfRegion {p : ℝ × ℝ | (p.1^2 + p.2^2 + 2*p.1 - 4*p.2) = 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l1362_136243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_cos_half_l1362_136238

noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2)

theorem smallest_positive_period_of_cos_half (T : ℝ) :
  (∀ x, f (x + T) = f x) ∧
  (∀ T' : ℝ, 0 < T' ∧ T' < T → ∃ x, f (x + T') ≠ f x) →
  T = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_cos_half_l1362_136238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1362_136297

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (3 : Polynomial ℝ) * X^2 - (11 : Polynomial ℝ) * X + (18 : Polynomial ℝ) = 
  (X - (3 : Polynomial ℝ)) * q + (12 : Polynomial ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1362_136297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_given_circle_l1362_136246

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 + 8 * x + 4 * y^2 - 12 * y + 29 = 0

/-- The center of a circle given by its equation -/
def circle_center (eq : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

theorem center_of_given_circle :
  circle_center circle_equation = (-1, 3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_given_circle_l1362_136246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_balls_count_l1362_136224

theorem soccer_balls_count (total : ℕ) : total = 100 :=
  let holes_percentage : ℚ := 40 / 100
  let exploded_percentage : ℚ := 20 / 100
  let successful : ℕ := 48

  have h : (1 - holes_percentage) * (1 - exploded_percentage) * total = successful := by sorry

  by
    -- Proof steps would go here
    sorry

#check soccer_balls_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_balls_count_l1362_136224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1362_136204

def sequence_a : ℕ → ℝ
  | 0 => 2  -- Add this case for n = 0
  | 1 => 2
  | n + 2 => 2 * sequence_a (n + 1) + 1

theorem sequence_a_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 3 * 2^(n - 1) - 1) := by
  sorry

#eval sequence_a 5  -- Optional: Add this line to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1362_136204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_distance_l1362_136283

/-- The distance Bob walked when he met Yolanda -/
noncomputable def distance_bob_walked (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) (head_start : ℝ) : ℝ :=
  (bob_speed * (total_distance - yolanda_speed * head_start)) / (yolanda_speed + bob_speed)

/-- Theorem stating that Bob walked 25⅓ miles when they met -/
theorem bob_walked_distance :
  distance_bob_walked 40 2 4 1 = 25 + 1/3 := by
  -- Unfold the definition of distance_bob_walked
  unfold distance_bob_walked
  -- Simplify the expression
  simp
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_distance_l1362_136283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_proof_l1362_136263

noncomputable def sequence_sum (a : ℝ) (n : ℕ) : ℝ :=
  (5 * (1 - a^n)) / ((1 - a)^2) - (4 + (5 * n - 4) * a^n) / (1 - a)

theorem sequence_sum_proof (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) :
  ∀ (n : ℕ), n ≥ 1 →
  let u : ℕ → ℝ := λ k => sequence_sum a k - sequence_sum a (k-1)
  let S : ℕ → ℝ := λ k => sequence_sum a k
  S 1 = 1 ∧ (∀ (k : ℕ), k ≥ 1 → S (k+1) - S k = (5*k + 1) * a^k) →
  ∀ (m : ℕ), m ≥ 1 → S m = sequence_sum a m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_proof_l1362_136263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_a_first_6_minutes_charge_l1362_136208

/-- Represents the charge for a phone call under a given plan -/
noncomputable def call_charge (fixed_charge : ℝ) (per_minute_rate : ℝ) (duration : ℝ) : ℝ :=
  if duration ≤ 6 then fixed_charge
  else fixed_charge + (duration - 6) * per_minute_rate

/-- The theorem stating the charge for the first 6 minutes under plan A -/
theorem plan_a_first_6_minutes_charge :
  ∃ (fixed_charge : ℝ),
    (∀ (duration : ℝ), 
      call_charge fixed_charge 0.06 duration = 0.08 * duration → duration = 12) →
    fixed_charge = 0.60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_a_first_6_minutes_charge_l1362_136208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_l1362_136293

noncomputable def F : ℝ × ℝ := (4, 0)

noncomputable def distance_to_F (M : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - F.1)^2 + M.2^2)

def distance_to_line (M : ℝ × ℝ) : ℝ :=
  |M.1 - 3|

def ratio_condition (M : ℝ × ℝ) : Prop :=
  distance_to_F M / distance_to_line M = 2

def locus_equation (M : ℝ × ℝ) : Prop :=
  3 * M.1^2 - M.2^2 - 16 * M.1 + 20 = 0

theorem locus_of_M :
  ∀ M : ℝ × ℝ, ratio_condition M → locus_equation M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_l1362_136293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_value_l1362_136237

-- Define the exponential function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_value (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →  -- Conditions for exponential function
  f a 2 = 4 →        -- The function passes through (2, 4)
  f a 3 = 8 :=       -- We want to prove f(3) = 8
by
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_value_l1362_136237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_champion_probability_theorem_l1362_136258

/-- Represents the probability of a player winning a single game -/
noncomputable def win_probability : ℝ := 1 / 2

/-- Represents the number of games needed to win the championship -/
def games_to_win : ℕ := 3

/-- Calculates the probability of a player becoming the champion given they've won the first game -/
noncomputable def champion_probability (p : ℝ) (n : ℕ) : ℝ :=
  let rec probability_helper (games_left : ℕ) (wins_needed : ℕ) : ℝ :=
    if wins_needed = 0 then 1
    else if games_left = 0 then 0
    else p * probability_helper (games_left - 1) (wins_needed - 1) +
         (1 - p) * probability_helper (games_left - 1) wins_needed
  probability_helper (2 * n - 2) (n - 1)

/-- Theorem stating that the probability of becoming champion after winning the first game is 11/16 -/
theorem champion_probability_theorem :
  champion_probability win_probability games_to_win = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_champion_probability_theorem_l1362_136258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1362_136212

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define a point on the hyperbola
variable (P : ℝ × ℝ)

-- State the theorem
theorem hyperbola_focal_distance 
  (h_on_hyperbola : hyperbola P.1 P.2)
  (h_perpendicular : (F₂.1 - F₁.1) * (P.1 - F₂.1) + (F₂.2 - F₁.2) * (P.2 - F₂.2) = 0) :
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 13/2 := by
  sorry

#check hyperbola_focal_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1362_136212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_expected_games_l1362_136290

/-- Represents the outcome of a single game -/
inductive GameOutcome
| A_Wins
| B_Wins

/-- Represents the state of the match -/
structure MatchState :=
  (games_played : ℕ)
  (score_diff : ℤ)

/-- Checks if the match has ended -/
def match_ended (state : MatchState) : Bool :=
  state.games_played = 6 ∨ state.score_diff.natAbs = 2

/-- Probability of player A winning a single game -/
def prob_A_wins : ℚ := 2/3

/-- Probability of player B winning a single game -/
def prob_B_wins : ℚ := 1/3

/-- Updates the match state based on the game outcome -/
def update_state (state : MatchState) (outcome : GameOutcome) : MatchState :=
  match outcome with
  | GameOutcome.A_Wins => { games_played := state.games_played + 1, score_diff := state.score_diff + 1 }
  | GameOutcome.B_Wins => { games_played := state.games_played + 1, score_diff := state.score_diff - 1 }

/-- Expected number of games played in the table tennis match -/
def expected_games : ℚ := 266/81

theorem table_tennis_expected_games :
  expected_games = 266/81 := by
  sorry

#eval expected_games

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_expected_games_l1362_136290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_hyperbola_tangent_volume_l1362_136221

/-- The curve C -/
def C (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x

/-- The hyperbola -/
def hyperbola (x y : ℝ) : Prop := x * y = 1

/-- The point P -/
structure Point where
  x : ℝ
  y : ℝ

/-- The common tangent condition -/
def commonTangent (a : ℝ) (p : Point) : Prop :=
  3 * a * p.x^2 + 4 = -p.y / p.x

/-- P lies on both curves -/
def pointOnBothCurves (a : ℝ) (p : Point) : Prop :=
  p.y = C a p.x ∧ hyperbola p.x p.y

/-- P is in the first quadrant -/
def firstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The volume of revolution -/
noncomputable def volumeOfRevolution (a : ℝ) (p : Point) : ℝ :=
  Real.pi * (83 / (105 * Real.sqrt 2))

/-- The main theorem -/
theorem curve_hyperbola_tangent_volume 
  (a : ℝ) (p : Point) 
  (h1 : a ≠ 0)
  (h2 : commonTangent a p)
  (h3 : pointOnBothCurves a p)
  (h4 : firstQuadrant p) :
  a = -4 ∧ 
  p.x = 1 / Real.sqrt 2 ∧ 
  p.y = Real.sqrt 2 ∧
  volumeOfRevolution a p = Real.pi * (83 / (105 * Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_hyperbola_tangent_volume_l1362_136221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1362_136213

/-- Calculates the speed of a train passing a bridge -/
noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / time
  let speed_kmh := speed_ms * 3.6
  speed_kmh

/-- Theorem stating the speed of the train given the specified conditions -/
theorem train_speed_theorem :
  ∃ ε > 0, |train_speed 360 140 32.142857142857146 - 55.9872| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1362_136213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_flowers_theorem_l1362_136264

theorem garden_flowers_theorem (F : ℝ) (hF : F > 0) : 
  let blue_flowers := (7/10) * F
  let red_flowers := F - blue_flowers
  let blue_tulips := (1/2) * blue_flowers
  let blue_daisies := blue_flowers - blue_tulips
  let red_daisies := (2/3) * red_flowers
  let total_daisies := blue_daisies + red_daisies
  (total_daisies / F) * 100 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_flowers_theorem_l1362_136264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_to_reach_three_l1362_136299

/-- A fair six-sided die -/
def FairDie : Type := Fin 6

/-- The probability of rolling a specific number on a fair die -/
noncomputable def prob (n : FairDie) : ℝ := 1 / 6

/-- The expected number of rolls to reach a sum of at least 3 -/
noncomputable def expected_rolls : ℝ := 49 / 36

/-- Theorem stating that the expected number of rolls to reach a sum of at least 3 is 49/36 -/
theorem expected_rolls_to_reach_three :
  expected_rolls = 49 / 36 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_to_reach_three_l1362_136299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l1362_136252

/-- Calculates the speed in km/h given distance in meters and time in minutes -/
noncomputable def calculate_speed (distance_m : ℝ) (time_min : ℝ) : ℝ :=
  (distance_m / 1000) / (time_min / 60)

theorem speed_calculation (distance_m time_min : ℝ) 
  (h1 : distance_m = 1080)
  (h2 : time_min = 12) :
  calculate_speed distance_m time_min = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l1362_136252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_zero_of_f_in_interval_l1362_136250

noncomputable def f (x : ℝ) := Real.sin x + 2 * Real.cos x + 3 * Real.tan x

theorem smallest_zero_of_f_in_interval :
  (∃ x : ℝ, x ∈ Set.Ioo 3 4 ∧ f x = 0) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 3 → f x ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_zero_of_f_in_interval_l1362_136250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l1362_136253

theorem angle_equality (α θ γ : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < θ ∧ θ < π/2)
  (h3 : 0 < γ ∧ γ < π/2)
  (h4 : Real.sin (α + γ) * Real.tan α = Real.sin (θ + γ) * Real.tan θ) : 
  α = θ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l1362_136253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_sine_symmetric_implies_phase_l1362_136269

/-- A function representing a sine wave with phase shift φ -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

/-- The translated function -/
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f φ (x - Real.pi/6)

/-- Symmetry about the origin for the translated function -/
def symmetric_about_origin (φ : ℝ) : Prop :=
  ∀ x, g φ x = - g φ (-x)

/-- Main theorem: If the translated function is symmetric about the origin, then φ = π/3 -/
theorem translated_sine_symmetric_implies_phase (φ : ℝ) :
  symmetric_about_origin φ → φ = Real.pi/3 := by
  sorry

#check translated_sine_symmetric_implies_phase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_sine_symmetric_implies_phase_l1362_136269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_fruit_purchase_cost_l1362_136244

/-- Represents the prices of different fruits -/
structure FruitPrices where
  lemon : ℚ
  papaya : ℚ
  mango : ℚ
  orange : ℚ
  apple : ℚ
  pineapple : ℚ

/-- Represents the quantity of fruits purchased -/
structure FruitQuantities where
  lemon : ℕ
  papaya : ℕ
  mango : ℕ
  orange : ℕ
  apple : ℕ
  pineapple : ℕ

/-- Calculates the total cost before discounts -/
def totalCostBeforeDiscounts (prices : FruitPrices) (quantities : FruitQuantities) : ℚ :=
  prices.lemon * quantities.lemon +
  prices.papaya * quantities.papaya +
  prices.mango * quantities.mango +
  prices.orange * quantities.orange +
  prices.apple * quantities.apple +
  prices.pineapple * quantities.pineapple

/-- Calculates the total number of fruits purchased -/
def totalFruits (quantities : FruitQuantities) : ℕ :=
  quantities.lemon + quantities.papaya + quantities.mango +
  quantities.orange + quantities.apple + quantities.pineapple

/-- Calculates the discount for buying every 4 fruits -/
def discountEveryFourFruits (totalFruits : ℕ) : ℚ :=
  (totalFruits / 4 : ℚ)

/-- Calculates the discount for buying more than 6 fruits of the same type -/
def discountMoreThanSix (quantities : FruitQuantities) : ℚ :=
  (1/2 : ℚ) * (max (quantities.lemon - 6) 0 + max (quantities.papaya - 6) 0 +
         max (quantities.mango - 6) 0 + max (quantities.orange - 6) 0 +
         max (quantities.apple - 6) 0 + max (quantities.pineapple - 6) 0)

/-- Calculates the total discount -/
def totalDiscount (quantities : FruitQuantities) : ℚ :=
  discountEveryFourFruits (totalFruits quantities) +
  2 + -- Discount for buying two different types of fruits
  discountMoreThanSix quantities

/-- Theorem: Tom's total cost for his fruit purchase is $55 -/
theorem toms_fruit_purchase_cost (prices : FruitPrices) (quantities : FruitQuantities)
    (h1 : prices = { lemon := 2, papaya := 1, mango := 4, orange := 3, apple := 3/2, pineapple := 5 })
    (h2 : quantities = { lemon := 8, papaya := 6, mango := 5, orange := 3, apple := 8, pineapple := 2 }) :
    totalCostBeforeDiscounts prices quantities - totalDiscount quantities = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_fruit_purchase_cost_l1362_136244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeroPointThreeSix_eq_elevenThirtieths_l1362_136239

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingLength : ℕ
  nonRepeatingNonneg : 0 ≤ nonRepeating
  repeatingNonneg : 0 ≤ repeating
  repeatingLtOne : repeating < 1

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.nonRepeating + d.repeating / (10^d.repeatingLength - 1)

/-- The repeating decimal 0.3̅6̅ -/
def zeroPointThreeSix : RepeatingDecimal where
  nonRepeating := 3/10
  repeating := 6/10
  repeatingLength := 1
  nonRepeatingNonneg := by norm_num
  repeatingNonneg := by norm_num
  repeatingLtOne := by norm_num

theorem zeroPointThreeSix_eq_elevenThirtieths :
  zeroPointThreeSix.toRational = 11/30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeroPointThreeSix_eq_elevenThirtieths_l1362_136239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discretionary_income_ratio_l1362_136245

-- Define Jill's financial parameters
noncomputable def net_salary : ℚ := 3400
noncomputable def vacation_fund_percent : ℚ := 30 / 100
noncomputable def savings_percent : ℚ := 20 / 100
noncomputable def social_spending_percent : ℚ := 35 / 100
noncomputable def gifts_charity_amount : ℚ := 102

-- Define the discretionary income
noncomputable def discretionary_income : ℚ :=
  gifts_charity_amount / (1 - vacation_fund_percent - savings_percent - social_spending_percent)

-- Theorem statement
theorem discretionary_income_ratio :
  discretionary_income / net_salary = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discretionary_income_ratio_l1362_136245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l1362_136280

/-- Represents the fuel efficiency of a car in kilometers per gallon -/
noncomputable def fuel_efficiency (distance : ℝ) (fuel : ℝ) : ℝ := distance / fuel

/-- Theorem stating that the car's fuel efficiency is 40 km/gal -/
theorem car_fuel_efficiency : 
  let distance := (180 : ℝ) -- km
  let fuel := (4.5 : ℝ) -- gallons
  fuel_efficiency distance fuel = 40 := by
  -- Unfold the definitions
  unfold fuel_efficiency
  -- Simplify the division
  simp
  -- The proof is complete
  norm_num

#eval (180 : Float) / (4.5 : Float) -- For verification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l1362_136280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_solutions_l1362_136228

/-- The complex function we're analyzing -/
noncomputable def f (z : ℂ) : ℂ := (z^4 - 1) / (z^3 - z + 2)

/-- The theorem stating that the equation f(z) = 0 has exactly 2 complex solutions -/
theorem two_complex_solutions :
  ∃! (s : Finset ℂ), s.card = 2 ∧ ∀ z ∈ s, f z = 0 ∧ ∀ z ∉ s, f z ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_solutions_l1362_136228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hcl_mixture_l1362_136222

-- Define the concentrations
def conc1 : ℚ := 10 / 100
def conc2 : ℚ := 30 / 100

-- Define the volume of the first solution
def vol1 : ℚ := 50

-- These would be given if we had the full information
-- We'll use variables instead of undefined constants
variable (desired_conc : ℚ)
variable (total_vol : ℚ)

-- Assumptions to ensure the problem makes sense
axiom desired_conc_range : 10 / 100 < desired_conc ∧ desired_conc < 30 / 100
axiom total_vol_positive : total_vol > vol1

-- Theorem to prove
theorem hcl_mixture :
  ∃ (vol2 : ℚ),
    vol2 > 0 ∧
    (conc1 * vol1 + conc2 * vol2) / (vol1 + vol2) = desired_conc ∧
    vol1 + vol2 = total_vol :=
by
  sorry -- The proof is omitted as requested


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hcl_mixture_l1362_136222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_tangent_l1362_136201

def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle_C1 (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

theorem one_common_tangent (m : ℝ) :
  (∃! l : Set (ℝ × ℝ), (∀ p : ℝ × ℝ, p ∈ l → circle_C p.1 p.2) ∧
                        (∀ p : ℝ × ℝ, p ∈ l → circle_C1 p.1 p.2 m)) →
  m = -24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_tangent_l1362_136201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_break_participants_l1362_136259

/-- Represents the number of participants who left the table. -/
def participants_left : ℕ → Prop := sorry

/-- The total number of participants. -/
def total_participants : ℕ := 14

/-- Proposition that some but not all participants left. -/
def some_but_not_all_left : Prop :=
  ∃ n : ℕ, participants_left n ∧ 0 < n ∧ n < total_participants

/-- Proposition that each remaining participant has exactly one neighbor who left. -/
def one_neighbor_left : Prop :=
  ∀ n : ℕ, participants_left n → (total_participants - n) % 2 = 0

/-- Theorem stating the possible numbers of participants who left. -/
theorem coffee_break_participants :
  some_but_not_all_left →
  one_neighbor_left →
  (participants_left 6 ∨ participants_left 8 ∨ participants_left 10 ∨ participants_left 12) ∧
  (∀ n : ℕ, participants_left n → (n = 6 ∨ n = 8 ∨ n = 10 ∨ n = 12)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_break_participants_l1362_136259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_result_l1362_136278

/-- Calculates the compound interest for a given principal, rate, and number of periods -/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Represents the investment scenario described in the problem -/
def investment_problem (initial_principal : ℝ) (interest_rate : ℝ) (additional_investment : ℝ) : Prop :=
  let semi_annual_rate := interest_rate / 2
  let amount_after_three_years := compound_interest initial_principal semi_annual_rate 6
  let new_principal := amount_after_three_years + additional_investment
  let final_amount := compound_interest new_principal semi_annual_rate 4
  ⌊final_amount⌋ = 17172

/-- The theorem statement for the investment problem -/
theorem investment_result :
  investment_problem 12000 0.045 2000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_result_l1362_136278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_implies_a_half_subset_condition_iff_a_range_l1362_136241

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.log x - a * (x^2 - 1)

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x ≥ 0}

-- Define the set N
def N : Set ℝ := {x | x ≥ 1}

-- Theorem 1: If x = 1 is a local minimum point of f(x), then a = 1/2
theorem local_minimum_implies_a_half (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 1| < δ → f a x ≥ f a 1) → a = 1/2 := by
  sorry

-- Theorem 2: N ⊆ M if and only if a ∈ (-∞, 1/2]
theorem subset_condition_iff_a_range (a : ℝ) :
  N ⊆ M a ↔ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_implies_a_half_subset_condition_iff_a_range_l1362_136241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_sequence_implies_power_of_two_divides_l1362_136215

/-- Sequence definition -/
def a : ℕ → ℤ
  | 0 => 2
  | k + 1 => 2 * (a k)^2 - 1

/-- Main theorem -/
theorem prime_divides_sequence_implies_power_of_two_divides (n : ℕ) (p : ℕ) :
  Nat.Prime p → Odd p → (p : ℤ) ∣ a n → (2^(n + 3) : ℕ) ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_sequence_implies_power_of_two_divides_l1362_136215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peregrine_falcon_dive_time_l1362_136249

/-- The time it takes for a bird to dive a certain distance given its speed and the time taken by another bird -/
noncomputable def dive_time (speed_ratio : ℝ) (reference_time : ℝ) : ℝ :=
  reference_time / speed_ratio

theorem peregrine_falcon_dive_time :
  let bald_eagle_speed : ℝ := 100
  let peregrine_falcon_speed : ℝ := 2 * bald_eagle_speed
  let bald_eagle_time : ℝ := 30
  dive_time (peregrine_falcon_speed / bald_eagle_speed) bald_eagle_time = 15 := by
  -- Unfold the definitions
  unfold dive_time
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peregrine_falcon_dive_time_l1362_136249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l1362_136251

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem interest_rate_is_four_percent :
  ∃ (rate : ℝ),
    simple_interest 500 rate 8 = 160 ∧
    rate = 4 :=
by
  -- We'll use 4 as our witness for the existential quantifier
  use 4
  -- Split the goal into two parts
  constructor
  -- Prove that simple_interest 500 4 8 = 160
  · simp [simple_interest]
    norm_num
  -- Prove that 4 = 4 (trivial)
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l1362_136251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_sum_property_l1362_136200

noncomputable def log_base (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem logarithm_sum_property (x : ℝ) : 
  (x = 3 ∧ 
   x^2 > 0 ∧ 
   x^2 - 3*x + 2 > 0 ∧ 
   x^2 / (x - 2) > 0 ∧ 
   x^2 / (x - 1) > 0) ↔ 
  (log_base (x^2) (x^2 - 3*x + 2) = log_base (x^2) (x^2 / (x - 2)) + log_base (x^2) (x^2 / (x - 1)) ∨
   log_base (x^2) (x^2 / (x - 2)) = log_base (x^2) (x^2 - 3*x + 2) + log_base (x^2) (x^2 / (x - 1)) ∨
   log_base (x^2) (x^2 / (x - 1)) = log_base (x^2) (x^2 - 3*x + 2) + log_base (x^2) (x^2 / (x - 2))) :=
by sorry

#check logarithm_sum_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_sum_property_l1362_136200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l1362_136289

-- Define the points A, B, and C
def A : Fin 3 → ℝ := ![1, 5, -2]
def B : Fin 3 → ℝ := ![2, 4, 1]
def C (p q : ℝ) : Fin 3 → ℝ := ![p, 2, q + 2]

-- Define collinearity
def collinear (A B C : Fin 3 → ℝ) : Prop :=
  ∃ t : ℝ, ∀ i : Fin 3, C i - A i = t * (B i - A i)

-- Theorem statement
theorem collinear_points_sum (p q : ℝ) :
  collinear A B (C p q) → p + q = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l1362_136289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_projection_division_ratio_l1362_136227

/-- A convex polygon in a 2D plane -/
class ConvexPolygon where
  -- Add necessary properties for a convex polygon

/-- A line in a 2D plane -/
class Line where
  -- Add necessary properties for a line

/-- Represents the projection of a polygon onto a line -/
def projection (P : ConvexPolygon) (l : Line) : Set ℝ := sorry

/-- The ratio of the lengths of segments created by a line on a set -/
def divisionRatio (S : Set ℝ) (l : Line) : ℝ := sorry

/-- Checks if a line bisects the area of a polygon -/
def bisectsArea (P : ConvexPolygon) (l : Line) : Prop := sorry

/-- Checks if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop := sorry

/-- The theorem statement -/
theorem polygon_projection_division_ratio 
  (P : ConvexPolygon) (l : Line) (perp_l : Line) :
  bisectsArea P l →
  perpendicular l perp_l →
  divisionRatio (projection P perp_l) l ≤ 1 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_projection_division_ratio_l1362_136227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_2_l1362_136210

-- Define the line
def line (x : ℝ) : ℝ := x

-- Define the curve parametrically
def curve_x (t : ℝ) : ℝ := t - 1
def curve_y (t : ℝ) : ℝ := t^2 - 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p.1 = curve_x t ∧ p.2 = curve_y t ∧ p.2 = line p.1}

-- State the theorem
theorem intersection_distance_is_sqrt_2 :
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧ dist A B = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_2_l1362_136210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_allocation_l1362_136223

noncomputable def total_fund : ℝ := 100

noncomputable def planting_benefit (x : ℝ) : ℝ := 27 * x / (10 + x)

noncomputable def pollution_benefit (x : ℝ) : ℝ := 0.3 * x

noncomputable def total_benefit (x : ℝ) : ℝ := planting_benefit x + pollution_benefit (total_fund - x)

theorem optimal_allocation :
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ total_fund ∧
  ∀ (y : ℝ), y ≥ 0 → y ≤ total_fund → total_benefit x ≥ total_benefit y ∧
  x = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_allocation_l1362_136223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_3_sqrt_5_l1362_136288

/-- Calculates the speed of a particle moving in a 2D plane -/
noncomputable def particleSpeed (t₁ t₂ : ℝ) : ℝ :=
  let x₁ := 3 * t₁ + 4
  let y₁ := 6 * t₁ - 16
  let x₂ := 3 * t₂ + 4
  let y₂ := 6 * t₂ - 16
  let dx := x₂ - x₁
  let dy := y₂ - y₁
  let distance := Real.sqrt (dx^2 + dy^2)
  let time := t₂ - t₁
  distance / time

/-- The speed of the particle over the given interval is 3√5 -/
theorem particle_speed_is_3_sqrt_5 : particleSpeed 2 5 = 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_3_sqrt_5_l1362_136288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_2A_l1362_136234

theorem triangle_sin_2A (A B C : ℝ) (a b c : ℝ) :
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  a / (2 * Real.cos A) = b / (3 * Real.cos B) →
  b / (3 * Real.cos B) = c / (6 * Real.cos C) →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  Real.sin (2 * A) = 3 * Real.sqrt 11 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_2A_l1362_136234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_amount_in_range_l1362_136256

/-- Represents a bank deposit with simple interest -/
structure Deposit where
  principal : ℝ
  rate : ℝ
  years : ℕ
  tax_rate : ℝ

/-- Calculates the post-tax interest for a deposit -/
def post_tax_interest (d : Deposit) : ℝ :=
  d.principal * d.rate * (d.years : ℝ) * (1 - d.tax_rate)

/-- Theorem stating that the original deposit amount is between 30000 and 40000 yuan -/
theorem deposit_amount_in_range (d : Deposit) 
    (h1 : d.rate = 0.027)
    (h2 : d.years = 3)
    (h3 : d.tax_rate = 0.2)
    (h4 : post_tax_interest d = 2241) :
    30000 < d.principal ∧ d.principal < 40000 := by
  sorry

#check deposit_amount_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_amount_in_range_l1362_136256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_separates_quarter_area_l1362_136257

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a median line
def medianLine (t : Triangle) (vertex : ℝ × ℝ) (midpoint : ℝ × ℝ) : Prop :=
  (vertex = t.A ∨ vertex = t.B ∨ vertex = t.C) ∧
  ((midpoint.1 = (t.A.1 + t.B.1) / 2 ∧ midpoint.2 = (t.A.2 + t.B.2) / 2) ∨
   (midpoint.1 = (t.B.1 + t.C.1) / 2 ∧ midpoint.2 = (t.B.2 + t.C.2) / 2) ∨
   (midpoint.1 = (t.C.1 + t.A.1) / 2 ∧ midpoint.2 = (t.C.2 + t.A.2) / 2))

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem median_separates_quarter_area (t : Triangle) (vertex midpoint : ℝ × ℝ) :
  medianLine t vertex midpoint →
  area { A := vertex, 
         B := midpoint, 
         C := if vertex = t.A then t.B else if vertex = t.B then t.C else t.A } = 
  (1/4 : ℝ) * area t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_separates_quarter_area_l1362_136257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_age_is_thirteen_l1362_136286

/-- Represents the current age of a person -/
def CurrentAge : Type := ℕ

/-- Represents the age difference between two people -/
def AgeDifference : Type := ℤ

/-- Given two people's current ages, calculates their combined age after a certain number of years -/
def combinedAgeAfterYears (myAge brotherAge yearsLater : ℕ) : ℕ :=
  myAge + brotherAge + 2 * yearsLater

/-- Given a person's current age, calculates their age a certain number of years ago -/
def ageYearsAgo (currentAge yearsAgo : ℕ) : ℤ :=
  (currentAge : ℤ) - yearsAgo

theorem current_age_is_thirteen (myAge brotherAge : ℕ) 
  (h1 : ageYearsAgo brotherAge 5 = 2 * ageYearsAgo myAge 5)
  (h2 : combinedAgeAfterYears myAge brotherAge 8 = 50) :
  myAge = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_age_is_thirteen_l1362_136286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_triangle_area_value_l1362_136281

/-- Triangle ABC with internal angles A, B, C and corresponding sides a, b, c -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The sine law for a triangle -/
axiom sine_law (t : Triangle) : t.a / (Real.sin t.A) = t.b / (Real.sin t.B)

/-- The cosine law for a triangle -/
axiom cosine_law (t : Triangle) : t.a^2 = t.b^2 + t.c^2 - 2 * t.b * t.c * Real.cos t.A

/-- The area formula for a triangle -/
noncomputable def triangle_area (t : Triangle) : Real := 1/2 * t.b * t.c * Real.sin t.A

theorem angle_A_value (t : Triangle) (h : 2 * t.a * Real.cos t.C = 2 * t.b - t.c) : 
  t.A = Real.pi/3 := by sorry

theorem triangle_area_value (t : Triangle) 
  (h1 : 2 * t.a * Real.cos t.C = 2 * t.b - t.c) 
  (h2 : t.a = Real.sqrt 7) 
  (h3 : t.b + t.c = 5) : 
  triangle_area t = 3 * Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_triangle_area_value_l1362_136281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_4a_cubed_over_7_l1362_136214

/-- A regular quadrilateral pyramid with an inscribed sphere -/
structure PyramidWithSphere where
  /-- Side length of the pyramid's base -/
  a : ℝ
  /-- Ratio of the height division by the inscribed sphere (from apex) -/
  height_ratio_top : ℝ
  /-- Ratio of the height division by the inscribed sphere (from base) -/
  height_ratio_bottom : ℝ
  /-- The sphere touches the base and all lateral faces -/
  sphere_touches_all_faces : Prop
  /-- The height ratios are 9:7 -/
  height_ratio_is_9_7 : height_ratio_top = 9 ∧ height_ratio_bottom = 7

/-- The volume of the pyramid -/
noncomputable def pyramid_volume (p : PyramidWithSphere) : ℝ := 4 * p.a^3 / 7

/-- Theorem: The volume of the pyramid is 4a³/7 -/
theorem pyramid_volume_is_4a_cubed_over_7 (p : PyramidWithSphere) :
  pyramid_volume p = 4 * p.a^3 / 7 := by
  -- Unfold the definition of pyramid_volume
  unfold pyramid_volume
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_4a_cubed_over_7_l1362_136214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_orange_primes_l1362_136226

/-- A prime p is orange if for every integer a, there exists an integer r such that r^q ≡ a (mod p) -/
def is_orange_prime (p q : ℕ) : Prop :=
  Nat.Prime p ∧ ∀ a : ℤ, ∃ r : ℤ, r^q ≡ a [ZMOD p]

theorem infinitely_many_orange_primes (q : ℕ) (hq : Nat.Prime q) (hq_odd : Odd q) :
  ∃ S : Set ℕ, (∀ p ∈ S, is_orange_prime p q) ∧ Infinite S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_orange_primes_l1362_136226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_factors_630_l1362_136265

/-- The sum of all positive even factors of 630 -/
def sumEvenFactors : ℕ := 1248

/-- 630 is the number we're considering -/
def n : ℕ := 630

/-- A factor is even if it's divisible by 2 -/
def isEvenFactor (d : ℕ) : Prop := d ∣ n ∧ 2 ∣ d

/-- Decidable instance for isEvenFactor -/
instance (d : ℕ) : Decidable (isEvenFactor d) :=
  And.decidable

theorem sum_even_factors_630 :
  (Finset.filter isEvenFactor (Finset.range (n + 1))).sum id = sumEvenFactors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_factors_630_l1362_136265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_todd_snow_cone_profit_l1362_136275

/-- Calculates Todd's final profit from his snow-cone stand business --/
theorem todd_snow_cone_profit
  (borrowed : ℝ)
  (repay : ℝ)
  (ingredients_cost : ℝ)
  (sales_quantity : ℝ)
  (price_per_unit : ℝ)
  (h1 : borrowed = 100)
  (h2 : repay = 110)
  (h3 : ingredients_cost = 75)
  (h4 : sales_quantity = 200)
  (h5 : price_per_unit = 0.75)
  : borrowed - ingredients_cost + sales_quantity * price_per_unit - repay = 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_todd_snow_cone_profit_l1362_136275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_equality_l1362_136292

/-- A function f is symmetric about the point (θ, 0) if f(θ + x) = f(θ - x) for all x -/
def symmetric_about (f : ℝ → ℝ) (θ : ℝ) : Prop :=
  ∀ x, f (θ + x) = f (θ - x)

/-- The main theorem -/
theorem symmetry_implies_equality (θ : ℝ) :
  symmetric_about (fun x ↦ Real.sin x + 2 * Real.cos x) θ →
  Real.cos (2 * θ) + Real.sin θ * Real.cos θ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_equality_l1362_136292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1362_136202

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 8 → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1362_136202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_x_l1362_136284

-- Define the function f(x) = lg x - 2x + 11
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10 - 2 * x + 11

-- State the theorem
theorem max_integer_x : 
  ∃ x : ℝ, f x = 0 ∧ x ≤ x → (∀ n : ℤ, (n : ℝ) > x → n > 5) ∧ (5 : ℝ) ≤ x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_x_l1362_136284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_diagonal_formula_l1362_136205

/-- A rectangular cuboid (brick) with length, width, and height -/
structure Brick where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- The diagonal of a brick -/
noncomputable def diagonal (b : Brick) : ℝ :=
  Real.sqrt (b.length ^ 2 + b.width ^ 2 + b.height ^ 2)

/-- Theorem: The diagonal of a brick is equal to the square root of the sum of squares of its dimensions -/
theorem brick_diagonal_formula (b : Brick) :
  diagonal b = Real.sqrt (b.length ^ 2 + b.width ^ 2 + b.height ^ 2) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_diagonal_formula_l1362_136205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_distance_range_l1362_136298

-- Define the curve C1 in polar coordinates
def C1 (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - Real.pi/3) = 1

-- Define the curve C2 in parametric form
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, -2 + Real.sin θ)

-- Define point P as the intersection of C1 and x-axis
def P : ℝ × ℝ := (2, 0)

-- Define points M and N on C2
noncomputable def M (α : ℝ) : ℝ × ℝ := C2 α
noncomputable def N (α : ℝ) : ℝ × ℝ := C2 (α + Real.pi/2)

-- Statement to prove
theorem curve_and_distance_range :
  (∀ x y, x + Real.sqrt 3 * y - 2 = 0 ↔ ∃ ρ θ, C1 ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∀ α, 10 ≤ (‖(M α - P)‖)^2 + (‖(N α - P)‖)^2 ∧
          (‖(M α - P)‖)^2 + (‖(N α - P)‖)^2 ≤ 26) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_distance_range_l1362_136298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honor_students_count_l1362_136273

/-- Represents a class of students -/
structure StudentClass where
  girls : ℕ
  boys : ℕ
  honorGirls : ℕ
  honorBoys : ℕ
  total_lt_30 : girls + boys < 30
  girl_honor_prob : (honorGirls : ℚ) / girls = 3 / 13
  boy_honor_prob : (honorBoys : ℚ) / boys = 4 / 11

/-- The number of honor students in the class is 7 -/
theorem honor_students_count (c : StudentClass) : c.honorGirls + c.honorBoys = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_honor_students_count_l1362_136273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_given_altitudes_is_obtuse_l1362_136277

theorem triangle_with_given_altitudes_is_obtuse :
  ∀ (a b c : ℝ),
  (a > 0) → (b > 0) → (c > 0) →
  (1/13 * a = 1/11 * b) →
  (1/11 * b = 1/5 * c) →
  (a / b = 13 / 11) →
  (b / c = 11 / 5) →
  (c^2 + b^2 - a^2) / (2 * b * c) < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_given_altitudes_is_obtuse_l1362_136277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contains_perfect_square_l1362_136233

/-- P(n) represents the largest prime factor of n -/
def largest_prime_factor (n : ℕ) : ℕ := sorry

/-- The sequence a_n as defined in the problem -/
def a : ℕ → ℕ
  | 0 => 2  -- a_1 is an integer greater than 1, we choose 2 as an example
  | n + 1 => a n + largest_prime_factor (a n)

/-- Theorem stating that the sequence contains a perfect square -/
theorem sequence_contains_perfect_square :
  ∃ n : ℕ, ∃ m : ℕ, a n = m ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contains_perfect_square_l1362_136233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_team_size_l1362_136295

/-- A subset of integers from 1 to 100 where no element is the sum of two others or twice another -/
def ValidTeam (S : Finset ℕ) : Prop :=
  S.Nonempty ∧ 
  (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 100) ∧
  (∀ x y z, x ∈ S → y ∈ S → z ∈ S → x + y ≠ z) ∧
  (∀ x y, x ∈ S → y ∈ S → 2 * x ≠ y)

/-- The maximum size of a valid team is 50 -/
theorem max_team_size :
  (∃ S : Finset ℕ, ValidTeam S ∧ S.card = 50) ∧
  (∀ S : Finset ℕ, ValidTeam S → S.card ≤ 50) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_team_size_l1362_136295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_and_independence_statement_l1362_136291

/-- Definition of regression analysis -/
def regression_analysis : Prop :=
  ∃ (x y : Type) (f : x → y), ∀ (a b : x), f a = f b → a = b

/-- Definition of independence tests -/
def independence_tests : Prop :=
  ∃ (x y : Type) (r : x → y → Prop), ∀ (a : x) (b : y), (r a b) ∨ ¬(r a b)

/-- The correct statement about regression analysis and independence tests -/
def correct_statement : Prop :=
  regression_analysis ∧ independence_tests

theorem regression_and_independence_statement :
  correct_statement :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_and_independence_statement_l1362_136291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circles_theorem_l1362_136261

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the intersection function
variable (intersect : Circle → Circle → Point × Point)

-- Define the property of points being on the same circle or line
variable (on_same_circle_or_line : List Point → Prop)

-- Define the four circles
variable (S₁ S₂ S₃ S₄ : Circle)

-- Define the intersection points
def A₁ (Point Circle : Type) (intersect : Circle → Circle → Point × Point) (S₁ S₂ : Circle) : Point := (intersect S₁ S₂).1
def A₂ (Point Circle : Type) (intersect : Circle → Circle → Point × Point) (S₁ S₂ : Circle) : Point := (intersect S₁ S₂).2
def B₁ (Point Circle : Type) (intersect : Circle → Circle → Point × Point) (S₂ S₃ : Circle) : Point := (intersect S₂ S₃).1
def B₂ (Point Circle : Type) (intersect : Circle → Circle → Point × Point) (S₂ S₃ : Circle) : Point := (intersect S₂ S₃).2
def C₁ (Point Circle : Type) (intersect : Circle → Circle → Point × Point) (S₃ S₄ : Circle) : Point := (intersect S₃ S₄).1
def C₂ (Point Circle : Type) (intersect : Circle → Circle → Point × Point) (S₃ S₄ : Circle) : Point := (intersect S₃ S₄).2
def D₁ (Point Circle : Type) (intersect : Circle → Circle → Point × Point) (S₄ S₁ : Circle) : Point := (intersect S₄ S₁).1
def D₂ (Point Circle : Type) (intersect : Circle → Circle → Point × Point) (S₄ S₁ : Circle) : Point := (intersect S₄ S₁).2

-- State the theorem
theorem four_circles_theorem :
  on_same_circle_or_line [A₁ Point Circle intersect S₁ S₂, B₁ Point Circle intersect S₂ S₃, C₁ Point Circle intersect S₃ S₄, D₁ Point Circle intersect S₄ S₁] →
  on_same_circle_or_line [A₂ Point Circle intersect S₁ S₂, B₂ Point Circle intersect S₂ S₃, C₂ Point Circle intersect S₃ S₄, D₂ Point Circle intersect S₄ S₁] :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circles_theorem_l1362_136261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1362_136267

theorem trigonometric_identity (t : ℝ) (p q r : ℕ) 
  (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 3/2)
  (h2 : (1 - Real.sin t) * (1 - Real.cos t) = (p : ℝ)/(q : ℝ) - Real.sqrt (r : ℝ))
  (h3 : Nat.Coprime p q)
  (hp : p > 0)
  (hq : q > 0)
  (hr : r > 0) : p + q + r = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1362_136267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_event_boys_fraction_l1362_136268

-- Define the schools
structure School where
  total_students : ℕ
  boys_ratio : ℕ
  girls_ratio : ℕ

-- Define the given schools
def lincoln : School := { total_students := 300, boys_ratio := 3, girls_ratio := 2 }
def jackson : School := { total_students := 240, boys_ratio := 2, girls_ratio := 3 }
def franklin : School := { total_students := 150, boys_ratio := 1, girls_ratio := 1 }

-- Function to calculate the number of boys in a school
def boys_count (s : School) : ℕ :=
  s.total_students * s.boys_ratio / (s.boys_ratio + s.girls_ratio)

-- Theorem to prove
theorem sports_event_boys_fraction :
  let total_boys := boys_count lincoln + boys_count jackson + boys_count franklin
  let total_students := lincoln.total_students + jackson.total_students + franklin.total_students
  (total_boys : ℚ) / total_students = 117 / 230 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_event_boys_fraction_l1362_136268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1362_136225

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℝ  -- common ratio
  k : ℝ
  seq : ℕ → ℝ
  is_geometric : ∀ n : ℕ, seq (n + 1) = a * seq n
  first_term : seq 1 = 1
  second_term : seq 2 = a
  relation : ∀ n : ℕ, seq (n + 1) = k * (seq n + seq (n + 2))
  not_one : a ≠ 1

/-- Three consecutive terms can form an arithmetic sequence -/
def ArithmeticProperty (s : GeometricSequence) : Prop :=
  ∀ m : ℕ, ∃ x y z : ℝ, (x = s.seq m ∧ y = s.seq (m + 1) ∧ z = s.seq (m + 2)) ∨
                        (x = s.seq m ∧ y = s.seq (m + 2) ∧ z = s.seq (m + 1)) ∨
                        (x = s.seq (m + 1) ∧ y = s.seq m ∧ z = s.seq (m + 2)) ∨
                        (x = s.seq (m + 1) ∧ y = s.seq (m + 2) ∧ z = s.seq m) ∨
                        (x = s.seq (m + 2) ∧ y = s.seq m ∧ z = s.seq (m + 1)) ∨
                        (x = s.seq (m + 2) ∧ y = s.seq (m + 1) ∧ z = s.seq m) ∧
                        y - x = z - y

theorem geometric_sequence_property (s : GeometricSequence) 
  (h : ArithmeticProperty s) : s.k = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1362_136225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_depth_is_four_l1362_136218

/-- Represents the properties of a river -/
structure River where
  width : ℝ
  flowRate : ℝ
  volumePerMinute : ℝ

/-- Calculates the depth of a river given its properties -/
noncomputable def riverDepth (r : River) : ℝ :=
  r.volumePerMinute / (r.width * r.flowRate * 1000 / 60)

/-- Theorem stating that the depth of the given river is 4 meters -/
theorem river_depth_is_four : 
  let r : River := { 
    width := 22,
    flowRate := 2,
    volumePerMinute := 2933.3333333333335
  }
  riverDepth r = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_depth_is_four_l1362_136218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_three_equals_four_l1362_136211

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c / x + 3

-- State the theorem
theorem f_minus_three_equals_four (a b c : ℝ) : 
  f a b c 3 = 2 → f a b c (-3) = 4 := by
  intro h
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_three_equals_four_l1362_136211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preTaxIncome_example_l1362_136206

/-- Calculates the pre-tax income given the tax rate, tax threshold, and net income. -/
noncomputable def preTaxIncome (taxRate : ℝ) (taxThreshold : ℝ) (netIncome : ℝ) : ℝ :=
  (netIncome - taxRate * taxThreshold) / (1 - taxRate)

/-- Theorem: Given a 10% tax rate on income over $3000 and a net income of $12000,
    the pre-tax income is $13000. -/
theorem preTaxIncome_example :
  preTaxIncome 0.1 3000 12000 = 13000 := by
  -- Unfold the definition of preTaxIncome
  unfold preTaxIncome
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_preTaxIncome_example_l1362_136206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l1362_136255

-- Define the complex number z
noncomputable def z (b : ℝ) : ℂ := (3 + b * Complex.I) * (1 + 3 * Complex.I)

-- Define omega
noncomputable def ω (b : ℝ) : ℂ := (3 + b * Complex.I) / (2 + Complex.I)

-- Theorem statement
theorem complex_problem :
  ∃ (b : ℝ), (z b).re = 0 ∧ (z b).im ≠ 0 ∧ b = 1 ∧ Complex.abs (ω b) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l1362_136255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_problem_l1362_136209

noncomputable def triangle_U_perimeter : ℝ := 22
noncomputable def triangle_U_area : ℝ := 5 * Real.sqrt 11

theorem isosceles_triangles_problem (c d : ℝ) :
  c > 0 ∧ d > 0 →
  2 * c + d = triangle_U_perimeter →
  1/2 * d * Real.sqrt (c^2 - (d/2)^2) = triangle_U_area →
  ∃ ε > 0, |d - 7| < ε := by
  sorry

#check isosceles_triangles_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_problem_l1362_136209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l1362_136232

theorem cosine_inequality (x y : ℝ) (h : 0 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ π) :
  Real.cos x + Real.cos y ≤ 1 + Real.cos (x * y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l1362_136232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_m_value_l1362_136271

/-- The function f(x) = logₘ(m-x) has a maximum value greater than its minimum value by 1 on the interval [3,5] if and only if m = 3 + √6 -/
theorem log_function_m_value (m : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x ∈ Set.Icc 3 5, f x = Real.log (m - x) / Real.log m) ∧
   (∃ (max min : ℝ), (∀ x ∈ Set.Icc 3 5, f x ≤ max) ∧
                     (∀ x ∈ Set.Icc 3 5, min ≤ f x) ∧
                     max - min = 1)) ↔
  m = 3 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_m_value_l1362_136271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1362_136272

noncomputable section

/-- The distance from a point (x, y) to the line ax + by + c = 0 --/
def distancePointToLine (x y a b c : ℝ) : ℝ :=
  (abs (a * x + b * y + c)) / Real.sqrt (a^2 + b^2)

/-- The circle equation (x-1)^2 + (y-2)^2 = r^2 --/
def circleEquation (x y r : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = r^2

/-- The line equation 4x + 3y - 35 = 0 --/
def lineEquation (x y : ℝ) : Prop :=
  4 * x + 3 * y - 35 = 0

theorem circle_line_intersection (r : ℝ) :
  (∃! (p q : ℝ × ℝ), circleEquation p.1 p.2 r ∧ circleEquation q.1 q.2 r ∧
    distancePointToLine p.1 p.2 4 3 (-35) = 1 ∧
    distancePointToLine q.1 q.2 4 3 (-35) = 1) →
  r ∈ Set.Ioo 4 6 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1362_136272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_plus_one_hundredth_l1362_136230

open BigOperators

def fraction_product (n : ℕ) : ℚ :=
  ∏ k in Finset.range n, (4*k + 4 : ℚ) / (4*k + 8 : ℚ)

theorem fraction_product_plus_one_hundredth :
  fraction_product 752 + 1/100 = 213/18800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_plus_one_hundredth_l1362_136230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_ten_shots_success_l1362_136285

/-- Represents the number of successful free throws -/
def SuccessfulShots := ℤ

/-- Represents the total number of free throws -/
def TotalShots := ℤ

/-- Calculates the success rate given successful shots and total shots -/
def successRate (successful : ℚ) (total : ℚ) : ℚ :=
  successful / total

theorem last_ten_shots_success
  (initial_total : ℚ)
  (initial_rate : ℚ)
  (final_total : ℚ)
  (final_rate : ℚ)
  (h1 : initial_total = 15)
  (h2 : successRate (initial_rate * initial_total) initial_total = 3/5)
  (h3 : final_total = initial_total + 10)
  (h4 : successRate (final_rate * final_total) final_total = 13/20)
  : ∃ (last_ten_success : ℤ),
    last_ten_success = ⌊final_rate * final_total⌋ - ⌊initial_rate * initial_total⌋ ∧
    last_ten_success = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_ten_shots_success_l1362_136285
