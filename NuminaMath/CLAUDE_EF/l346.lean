import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_graph_coloring_l346_34632

/-- Represents a graph where vertices are cities and edges are flights -/
structure CityGraph (V : Type*) where
  E : V → V → Prop -- Edge relation (flights)

/-- Represents the coloring of vertices -/
def Coloring {V : Type*} (G : CityGraph V) (k : ℕ) := V → Fin (k + 2)

/-- Predicate to check if a coloring is valid -/
def IsValidColoring {V : Type*} (G : CityGraph V) {k : ℕ} (c : Coloring G k) : Prop :=
  ∀ u v : V, G.E u v → c u ≠ c v

/-- Predicate to check if two edges share a common endpoint -/
def SharesEndpoint {V : Type*} (e1 e2 : V × V) : Prop :=
  e1.1 = e2.1 ∨ e1.1 = e2.2 ∨ e1.2 = e2.1 ∨ e1.2 = e2.2

/-- The main theorem -/
theorem city_graph_coloring {V : Type*} (G : CityGraph V) (k : ℕ) 
  (airline_partition : Fin k → Set (V × V))
  (h_partition : ∀ e : V × V, G.E e.1 e.2 → ∃ i, e ∈ airline_partition i)
  (h_share_endpoint : ∀ i : Fin k, ∀ e1 e2, e1 ∈ airline_partition i → e2 ∈ airline_partition i → SharesEndpoint e1 e2) :
  ∃ c : Coloring G k, IsValidColoring G c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_graph_coloring_l346_34632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_60_6_l346_34685

/-- Represents a sequence of bottle numbers in systematic sampling -/
def BottleSequence := List Nat

/-- Checks if a sequence is valid for systematic sampling -/
def isValidSystematicSample (n : Nat) (k : Nat) (seq : BottleSequence) : Prop :=
  seq.length = k ∧
  ∀ i j, i < j → j < seq.length → seq.get! i < seq.get! j ∧ seq.get! j - seq.get! i = (j - i) * (n / k)

/-- The theorem statement -/
theorem systematic_sampling_60_6 :
  let n : Nat := 60  -- Total number of bottles
  let k : Nat := 6   -- Number of bottles to sample
  let seq : BottleSequence := [3, 13, 23, 33, 43, 53]
  isValidSystematicSample n k seq := by
  sorry

#check systematic_sampling_60_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_60_6_l346_34685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concat_primes_plus_three_equals_five_squared_unique_prime_square_concat_plus_three_l346_34687

/-- Predicate to check if a number is a concatenation of two other numbers -/
def is_concatenation (Nr q r : ℕ) : Prop :=
  ∃ (k : ℕ), Nr = q * 10^k + r ∧ k = (Nat.digits 10 r).length

/-- Given two prime numbers, their concatenation plus 3 equals the square of 5 -/
theorem concat_primes_plus_three_equals_five_squared (q r : ℕ) :
  Nat.Prime q →
  Nat.Prime r →
  ∃ (Nr : ℕ), is_concatenation Nr q r ∧ Nr + 3 = 5^2 :=
by sorry

/-- The only prime p whose square equals the concatenation of two primes plus 3 is 5 -/
theorem unique_prime_square_concat_plus_three (p q r : ℕ) :
  Nat.Prime p →
  Nat.Prime q →
  Nat.Prime r →
  (∃ (Nr : ℕ), is_concatenation Nr q r ∧ Nr + 3 = p^2) →
  p = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concat_primes_plus_three_equals_five_squared_unique_prime_square_concat_plus_three_l346_34687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sixth_minus_six_x_squared_equals_675_l346_34697

theorem x_sixth_minus_six_x_squared_equals_675 :
  ∀ x : ℝ, x = 3 → x^6 - 6*x^2 = 675 := by
  intro x h
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sixth_minus_six_x_squared_equals_675_l346_34697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_phi_l346_34663

/-- The function y in terms of x and φ -/
noncomputable def y (x φ : ℝ) : ℝ := 3 * Real.cos (2 * x + φ)

/-- The theorem stating the minimum absolute value of φ -/
theorem min_abs_phi :
  ∃ (φ : ℝ),
    (y (4 * Real.pi / 3) φ = 0) ∧
    (∀ ψ : ℝ, y (4 * Real.pi / 3) ψ = 0 → |φ| ≤ |ψ|) ∧
    (|φ| = Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_phi_l346_34663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_capacity_equation_correct_l346_34675

/-- Represents the freight capacity of a small truck in tons. -/
def small_truck_capacity : ℝ → ℝ := id

/-- Represents the freight capacity of a large truck in tons. -/
def large_truck_capacity (x : ℝ) : ℝ := x + 4

/-- The equation representing the relationship between truck capacities and goods transported. -/
def capacity_equation (x : ℝ) : Prop :=
  80 / (large_truck_capacity x) = 60 / x

/-- Theorem stating that the capacity equation correctly represents the relationship
    between truck capacities and goods transported. -/
theorem capacity_equation_correct (x : ℝ) (h1 : x > 0) :
  capacity_equation x ↔
    (∃ n : ℕ, n * (large_truck_capacity x) = 80 ∧ n * x = 60) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_capacity_equation_correct_l346_34675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_discount_percentage_l346_34657

/-- Given the initial price, second discount percentage, and final price after both discounts,
    calculate the first discount percentage. -/
noncomputable def calculate_first_discount (initial_price : ℝ) (second_discount : ℝ) (final_price : ℝ) : ℝ :=
  (1 - final_price / (initial_price * (1 - second_discount / 100))) * 100

/-- The first discount percentage is approximately 13.99% given the problem conditions. -/
theorem first_discount_percentage :
  let initial_price : ℝ := 390
  let second_discount : ℝ := 15
  let final_price : ℝ := 285.09
  abs (calculate_first_discount initial_price second_discount final_price - 13.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_discount_percentage_l346_34657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_property_l346_34628

-- Define the parabola
def parabola (p : ℝ) : Set (ℝ × ℝ) := {point | point.2^2 = 2*p*point.1}

-- Define the focus point
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix line
def directrix : ℝ → ℝ := λ x ↦ -1

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distance from a point to the directrix
def distToDirectrix (p : ℝ × ℝ) : ℝ :=
  |p.1 - directrix p.1|

-- State the theorem
theorem parabola_focus_directrix_property :
  ∃ (p : ℝ), ∀ (point : ℝ × ℝ),
    point ∈ parabola p →
    distance point focus = distToDirectrix point →
    p = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_property_l346_34628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l346_34696

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x + a * y + 1 = 0
def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ (a - 2) * x + 3 * y + 1 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop := ∀ x y, f x y ↔ g x y

-- Proposition p
def prop_p : Prop := ∃ a : ℝ, a ≠ -1 ∧ parallel (l₁ a) (l₂ a)

-- Proposition q
def prop_q : Prop := ∀ x : ℝ, x > 0 ∧ x < 1 → x^2 - x < 0

theorem problem_statement : (¬prop_p) ∧ prop_q := by
  sorry

#check problem_statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l346_34696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_l346_34642

noncomputable def sequence_recurrence (a : ℕ → ℝ) : Prop :=
  ∀ k, k ≥ 2 → a k = (a (k - 1) + Real.sqrt 3) / (1 - Real.sqrt 3 * a (k - 1))

theorem sequence_periodic (a : ℕ → ℝ) (h : sequence_recurrence a) :
  ∀ a₁ : ℝ, a₁ ≠ 1 / Real.sqrt 3 → a₁ ≠ -1 / Real.sqrt 3 →
  a 1 = a₁ → a 4 = a 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_l346_34642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sum_bound_l346_34661

/-- A chord in a circle --/
structure Chord where
  start : ℝ × ℝ
  finish : ℝ × ℝ

/-- The circle with radius 1 centered at the origin --/
def UnitCircle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

/-- A theorem about the sum of chord lengths in a unit circle --/
theorem chord_sum_bound
  (chords : Finset Chord)
  (k : ℕ)
  (h_k : k > 0)
  (h_diameter : ∀ d : ℝ × ℝ, d ∈ UnitCircle → (chords.filter (λ c => (c.start.1 - c.finish.1) * d.1 + (c.start.2 - c.finish.2) * d.2 ≠ 0)).card ≤ k)
  : (chords.sum (λ c => Real.sqrt ((c.start.1 - c.finish.1)^2 + (c.start.2 - c.finish.2)^2))) < k * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sum_bound_l346_34661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_smallest_sum_is_380_no_smaller_sum_l346_34666

theorem smallest_sum_of_factors (a b : ℕ) (h : 2^3 * 3^7 * 7^2 = a^b) :
  ∀ (x y : ℕ), 2^3 * 3^7 * 7^2 = x^y → a + b ≤ x + y :=
by sorry

theorem smallest_sum_is_380 :
  ∃ (a b : ℕ), 2^3 * 3^7 * 7^2 = a^b ∧ a + b = 380 :=
by sorry

theorem no_smaller_sum (a b : ℕ) (h : 2^3 * 3^7 * 7^2 = a^b) :
  a + b ≥ 380 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_smallest_sum_is_380_no_smaller_sum_l346_34666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l346_34615

/-- The hyperbola equation -2x^2 + 4y^2 - 8x - 24y - 8 = 0 -/
def hyperbola_eq (x y : ℝ) : Prop :=
  -2 * x^2 + 4 * y^2 - 8 * x - 24 * y - 8 = 0

/-- The center of the hyperbola -/
def center : ℝ × ℝ := (-2, 3)

/-- One of the foci of the hyperbola -/
noncomputable def focus : ℝ × ℝ := (-2, 3 + 3 * Real.sqrt 3)

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), hyperbola_eq x y ↔ 
    ((y - center.2) / a)^2 - ((x - center.1) / b)^2 = 1) ∧
  focus.1 = center.1 ∧
  focus.2 = center.2 + Real.sqrt (a^2 + b^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l346_34615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ssn_x_is_eight_l346_34603

/-- Represents a social security number in the format XYZ-WVU-TSRQ --/
structure SSN where
  X : Nat
  Y : Nat
  Z : Nat
  W : Nat
  V : Nat
  U : Nat
  T : Nat
  S : Nat
  R : Nat
  Q : Nat
  x_gt_y : X > Y
  y_gt_z : Y > Z
  w_gt_v : W > V
  v_gt_u : V > U
  t_gt_s : T > S
  s_gt_r : S > R
  r_gt_q : R > Q
  w_even : Even W
  u_even : Even U
  v_odd : Odd V
  t_even : Even T
  s_odd : Odd S
  r_odd : Odd R
  q_odd : Odd Q
  consecutive_srq : S = R + 1 ∧ R = Q + 1
  sum_xyz : X + Y + Z = 13

theorem ssn_x_is_eight (ssn : SSN) : ssn.X = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ssn_x_is_eight_l346_34603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l346_34637

-- Define the geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

-- Define the eccentricity of an ellipse
noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b * b) / (a * a))

-- Define the eccentricity of a hyperbola
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b * b) / (a * a))

-- Theorem statement
theorem conic_section_eccentricity (m : ℝ) :
  is_geometric_sequence 1 m 9 →
  (∃ (x y : ℝ), x * x / m + y * y = 1) →
  (∃ (e : ℝ), e = Real.sqrt 6 / 3 ∨ e = 2) :=
by
  sorry

#check conic_section_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l346_34637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l346_34635

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Theorem: The area of triangle ABC with coordinates A(1, 4), B(3, 7), and C(2, 8) is 5/2 -/
theorem triangle_abc_area :
  triangle_area 1 4 3 7 2 8 = 5/2 := by
  -- Expand the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l346_34635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_methods_valid_l346_34612

/-- Represents a method for randomly distributing tickets -/
inductive DistributionMethod
  | Drawing
  | RandomNumber

/-- Represents a class with students and tickets to distribute -/
structure ClassInfo where
  numStudents : Nat
  numTickets : Nat

/-- Checks if a distribution method is valid for a given class -/
def isValidMethod (c : ClassInfo) (m : DistributionMethod) : Prop :=
  c.numStudents > 0 ∧ c.numTickets > 0 ∧ c.numTickets ≤ c.numStudents

theorem both_methods_valid (c : ClassInfo) 
  (h1 : c.numStudents = 60) 
  (h2 : c.numTickets = 10) : 
  isValidMethod c DistributionMethod.Drawing ∧ 
  isValidMethod c DistributionMethod.RandomNumber := by
  sorry

#check both_methods_valid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_methods_valid_l346_34612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_equation_l346_34689

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem no_solution_for_equation :
  ∀ x : ℝ, (floor x + floor (2 * x) + floor (4 * x) + floor (8 * x) + floor (16 * x) + floor (32 * x)) ≠ 12345 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_equation_l346_34689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l346_34646

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

theorem right_triangle_hypotenuse (a : ℝ) :
  let A : Point := ⟨a, -a^2⟩
  let B : Point := ⟨-a, -a^2⟩
  let O : Point := ⟨0, 0⟩
  (A.y = -A.x^2) ∧ 
  (B.y = -B.x^2) ∧ 
  (distance O A)^2 + (distance O B)^2 = (distance A B)^2 →
  distance A B = 2 * a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l346_34646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_complex_product_l346_34606

theorem symmetric_complex_product (z₁ z₂ : ℂ) : 
  z₁ = 2 + Complex.I → 
  z₂ = -z₁.re + z₁.im * Complex.I → 
  z₁ * z₂ = -5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_complex_product_l346_34606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l346_34680

noncomputable def f (x : ℝ) : ℝ := (Real.cos x ^ 3 + 6 * Real.cos x ^ 2 + Real.cos x + 2 * Real.sin x ^ 2 - 8) / (Real.cos x - 1)

theorem f_range : 
  ∀ x : ℝ, Real.cos x ≠ 1 → 2 ≤ f x ∧ f x < 12 ∧
  ∀ y : ℝ, 2 ≤ y ∧ y < 12 → ∃ x : ℝ, Real.cos x ≠ 1 ∧ f x = y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l346_34680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_neg_thirty_five_l346_34688

def f (x : ℝ) : ℝ := x^5 - 3*x^3 - 6*x^2 + x - 1

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

theorem f_neg_two_equals_neg_thirty_five :
  f (-2) = -35 :=
by
  -- Define the coefficients of f(x) in descending order of degree
  let coeffs : List ℝ := [1, 0, -3, -6, 1, -1]
  -- Assert that f(-2) is equal to the result of Horner's method
  have h : f (-2) = horner_eval coeffs (-2) := by
    -- The proof of this equality goes here
    sorry
  -- Use the equality to prove the main theorem
  calc
    f (-2) = horner_eval coeffs (-2) := h
    _ = -35 := by
      -- The proof that horner_eval coeffs (-2) = -35 goes here
      sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_neg_thirty_five_l346_34688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_motion_scale_problem_solution_l346_34647

/-- Represents the change in y for a given change in x in a linear motion -/
def linear_motion (Δx : ℝ) (Δy : ℝ) : Prop :=
  ∀ (k : ℝ), k * Δy = (k * Δx / Δx) * Δy

theorem linear_motion_scale (Δx₁ Δy₁ Δx₂ : ℝ) (h : linear_motion Δx₁ Δy₁) :
  Δx₂ = 3 * Δx₁ → linear_motion Δx₂ (3 * Δy₁) :=
by sorry

theorem problem_solution :
  linear_motion 4 6 → linear_motion 12 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_motion_scale_problem_solution_l346_34647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_total_spent_l346_34676

noncomputable def appetizer_price : ℝ := 12.35
noncomputable def main_course_price : ℝ := 27.50
noncomputable def dessert_price : ℝ := 9.95

noncomputable def appetizer_tip_percentage : ℝ := 18
noncomputable def main_course_tip_percentage : ℝ := 20
noncomputable def dessert_tip_percentage : ℝ := 15

noncomputable def total_spent : ℝ :=
  (appetizer_price * (1 + appetizer_tip_percentage / 100)) +
  (main_course_price * (1 + main_course_tip_percentage / 100)) +
  (dessert_price * (1 + dessert_tip_percentage / 100))

theorem tim_total_spent :
  total_spent = (appetizer_price * (1 + appetizer_tip_percentage / 100)) +
                (main_course_price * (1 + main_course_tip_percentage / 100)) +
                (dessert_price * (1 + dessert_tip_percentage / 100)) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_total_spent_l346_34676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_water_usage_l346_34627

-- Define the water pricing structure
noncomputable def water_price (usage : ℝ) : ℝ :=
  if usage ≤ 5 then 1.8 * usage
  else 5 * 1.8 + 2 * (usage - 5)

-- Define the minimum bill amount
def min_bill : ℝ := 15

-- Theorem: The minimum monthly water usage is 8 cubic meters
theorem min_water_usage : 
  ∀ x : ℝ, x ≥ 0 → (water_price x ≥ min_bill → x ≥ 8) ∧ 
  (x < 8 → water_price x < min_bill) := by
  sorry

#check min_water_usage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_water_usage_l346_34627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_1_value_expression_2_value_l346_34655

-- Define the angle α
variable (α : Real)

-- Define the condition that the terminal side of angle α passes through point P(-4,3)
def terminal_side_condition (α : Real) : Prop := ∃ r : Real, r > 0 ∧ r * Real.cos α = -4 ∧ r * Real.sin α = 3

-- Theorem for the first expression
theorem expression_1_value {α : Real} (h : terminal_side_condition α) : 
  (Real.sin (π - α) + Real.cos (-α)) / Real.tan (π + α) = 16/15 := by sorry

-- Theorem for the second expression
theorem expression_2_value {α : Real} (h : terminal_side_condition α) : 
  Real.sin α * Real.cos α + Real.cos α ^ 2 - Real.sin α ^ 2 + 1 = 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_1_value_expression_2_value_l346_34655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_cost_l346_34611

/-- The total cost of a vacation -/
noncomputable def total_cost : ℝ := sorry

/-- The cost per person when divided among 3 people -/
noncomputable def cost_per_person_3 : ℝ := total_cost / 3

/-- The cost per person when divided among 4 people -/
noncomputable def cost_per_person_4 : ℝ := total_cost / 4

/-- The difference in cost per person between 3-person and 4-person division -/
def cost_difference : ℝ := 30

theorem vacation_cost :
  cost_per_person_3 - cost_per_person_4 = cost_difference →
  total_cost = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_cost_l346_34611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l346_34659

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 7) 
  (h3 : a + b > b) (h4 : b + b > a) : a + b + b = 17 :=
by
  -- Replace the sides with their given values
  rw [h1, h2]
  -- Simplify the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l346_34659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_points_satisfying_condition_l346_34607

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A circle with center at the origin and radius 2 -/
def circleSet : Set Point :=
  {p : Point | p.x^2 + p.y^2 < 4}

/-- The endpoints of a diameter of the circle -/
def diameterEndpoints : Point × Point :=
  (⟨-2, 0⟩, ⟨2, 0⟩)

/-- The condition for a point P satisfying the problem statement -/
def satisfiesCondition (p : Point) : Prop :=
  p ∈ circleSet ∧
  let (a, b) := diameterEndpoints
  (distance p a)^2 + (distance p b)^2 = 8

theorem infinite_points_satisfying_condition :
  ∃ (S : Set Point), Set.Infinite S ∧ ∀ p ∈ S, satisfiesCondition p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_points_satisfying_condition_l346_34607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l346_34694

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : Real.cos (2 * α) = -4/5) 
  (h2 : α > π/2 ∧ α < π) : 
  Real.tan (α + π/4) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l346_34694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l346_34679

def N : Matrix (Fin 3) (Fin 3) ℝ := !![4, -1, 9; -2, 6, -10; 3, 0, 6]

def i : Fin 3 → ℝ := ![1, 0, 0]
def j : Fin 3 → ℝ := ![0, 1, 0]
def k : Fin 3 → ℝ := ![0, 0, 1]

theorem matrix_N_satisfies_conditions :
  N.mulVec i = ![4, -2, 3] ∧
  N.mulVec j = ![-1, 6, 0] ∧
  N.mulVec k = 2 • (N.mulVec i) - (N.mulVec j) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l346_34679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l346_34602

/-- The interest rate problem -/
theorem interest_rate_problem (principal : ℝ) (b_rate : ℝ) (b_gain : ℝ) (time : ℝ) 
  (h1 : principal = 2000)
  (h2 : b_rate = 11.5)
  (h3 : b_gain = 90)
  (h4 : time = 3) :
  (principal * b_rate * time / 100 - b_gain) / (principal * time) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l346_34602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_time_is_nine_minutes_l346_34629

/-- The time (in minutes) Lucas can see Liam -/
noncomputable def visibleTime (lucasSpeed laimSpeed initialDistance finalDistance : ℝ) : ℝ :=
  (initialDistance + finalDistance) / (lucasSpeed - laimSpeed) * 60

/-- Theorem stating that the visible time is 9 minutes -/
theorem visible_time_is_nine_minutes :
  visibleTime 20 6 1 1 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_time_is_nine_minutes_l346_34629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_range_l346_34639

theorem log_base_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo (-2) (-1) → (x + 1)^2 < Real.log (abs x) / Real.log a) ↔ a ∈ Set.Ioc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_range_l346_34639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_sum_equality_l346_34672

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to get the medial triangle of a given triangle
noncomputable def medialTriangle (t : Triangle) : Triangle := sorry

-- Define a function to calculate the sum of cotangents of angles in a triangle
noncomputable def sumOfCotangents (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem cotangent_sum_equality (t : Triangle) :
  sumOfCotangents t = sumOfCotangents (medialTriangle t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_sum_equality_l346_34672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_categorization_l346_34601

def given_numbers : List ℚ := [-7, 301/100, 2015, -71/500, 1/10, 0, 99, -7/5]

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n

def is_fraction (q : ℚ) : Prop := true  -- All rationals are fractions

def is_negative_rational (q : ℚ) : Prop := q < 0

def integer_set : Set ℚ := {q ∈ given_numbers | is_integer q}
def fraction_set : Set ℚ := {q ∈ given_numbers | is_fraction q}
def negative_rational_set : Set ℚ := {q ∈ given_numbers | is_negative_rational q}

theorem correct_categorization :
  integer_set = {-7, 2015, 0, 99} ∧
  fraction_set = {301/100, -71/500, 1/10, -7/5} ∧
  negative_rational_set = {-7, -71/500, -7/5} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_categorization_l346_34601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_smallest_to_largest_circle_l346_34665

-- Define the side length of the equilateral triangle
variable (s : ℝ)

-- Define the radius of the largest circle (circumscribing the equilateral triangle)
noncomputable def R (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the radius of the circle inscribed in the equilateral triangle
noncomputable def r (s : ℝ) : ℝ := s * Real.sqrt 3 / 6

-- Define the side length of the square inscribed in the middle circle
noncomputable def s' (s : ℝ) : ℝ := r s * Real.sqrt 2

-- Define the radius of the smallest circle (inscribed in the square)
noncomputable def r' (s : ℝ) : ℝ := s' s / 2

-- Theorem statement
theorem area_ratio_smallest_to_largest_circle (s : ℝ) (h : s > 0) :
  (π * (r' s) ^ 2) / (π * (R s) ^ 2) = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_smallest_to_largest_circle_l346_34665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l346_34604

noncomputable section

open Real

variable (a b c A B C : ℝ)

def triangle_ABC (a b c A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_ABC_properties 
  (h_triangle : triangle_ABC a b c A B C)
  (h_sides : 4 * a = sqrt 5 * c)
  (h_cosC : cos C = 3 / 5)
  (h_b : b = 11) :
  sin A = sqrt 5 / 5 ∧ 
  (1 / 2) * a * b * sin C = 22 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l346_34604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2023_irrational_l346_34683

-- Define the numbers
def a : ℚ := -2023
noncomputable def b : ℝ := Real.sqrt 2023
def c : ℚ := 0
def d : ℚ := 1 / 2023

-- State the theorem
theorem sqrt_2023_irrational :
  ¬ (∃ (q : ℚ), (q : ℝ) = b) ∧ (∃ (q : ℚ), (q : ℝ) = a) ∧ (∃ (q : ℚ), (q : ℝ) = c) ∧ (∃ (q : ℚ), (q : ℝ) = d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2023_irrational_l346_34683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_values_l346_34692

/-- Triangle ABC with circumcenter O and orthocenter H -/
structure Triangle :=
  (A B C O H : ℝ × ℝ)

/-- The circumcenter O is equidistant from all vertices -/
def is_circumcenter (t : Triangle) : Prop :=
  dist t.O t.A = dist t.O t.B ∧ dist t.O t.B = dist t.O t.C

/-- The orthocenter H is the intersection of the altitudes -/
def is_orthocenter (t : Triangle) : Prop :=
  (t.H.1 - t.A.1) * (t.B.1 - t.C.1) + (t.H.2 - t.A.2) * (t.B.2 - t.C.2) = 0 ∧
  (t.H.1 - t.B.1) * (t.C.1 - t.A.1) + (t.H.2 - t.B.2) * (t.C.2 - t.A.2) = 0 ∧
  (t.H.1 - t.C.1) * (t.A.1 - t.B.1) + (t.H.2 - t.C.2) * (t.A.2 - t.B.2) = 0

/-- The angle at vertex B -/
noncomputable def angle_B (t : Triangle) : ℝ := 
  Real.arccos ((dist t.A t.C ^ 2 - dist t.A t.B ^ 2 - dist t.B t.C ^ 2) / (-2 * dist t.A t.B * dist t.B t.C))

/-- The main theorem -/
theorem angle_B_values (t : Triangle) 
  (h_circ : is_circumcenter t) 
  (h_orth : is_orthocenter t) 
  (h_eq : dist t.B t.O = dist t.B t.H) : 
  angle_B t = π / 3 ∨ angle_B t = 2 * π / 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_values_l346_34692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l346_34686

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((x + 1) / (x - 1))

-- State the theorem
theorem f_properties :
  -- f(x) is monotonically decreasing on (1, +∞)
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  -- The solution set of the inequality
  (∀ x, f (x^2 + x + 3) + f (-2*x^2 + 4*x - 7) > 0 ↔ x < 1 ∨ x > 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l346_34686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derek_walking_time_difference_l346_34610

/-- The time in minutes it takes Derek to walk a mile alone -/
noncomputable def derek_alone_time : ℝ := 9

/-- The time in minutes it takes Derek to walk a mile with his brother -/
noncomputable def derek_brother_time : ℝ := 12

/-- The total difference in minutes when walking with his brother -/
noncomputable def total_time_difference : ℝ := 60

/-- The number of miles Derek walks -/
noncomputable def miles_walked : ℝ := total_time_difference / (derek_brother_time - derek_alone_time)

theorem derek_walking_time_difference :
  miles_walked * (derek_brother_time - derek_alone_time) = total_time_difference := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derek_walking_time_difference_l346_34610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_2x_l346_34654

noncomputable def f (x : ℝ) := Real.sin (2 * x)

noncomputable def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x + a)

noncomputable def compress_horizontal (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f (k * x)

noncomputable def g (x : ℝ) := Real.sin (4 * x + 2 * Real.pi / 3)

theorem transform_sin_2x :
  (compress_horizontal (translate_left f (Real.pi / 3)) 2) = g := by
  funext x
  simp [compress_horizontal, translate_left, f, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_2x_l346_34654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_hundred_l346_34608

theorem subset_sum_divisible_by_hundred (a : Fin 100 → ℕ+) :
  ∃ (s : Finset (Fin 100)), s.Nonempty ∧ 100 ∣ (s.sum (λ i => (a i).val)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_hundred_l346_34608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_E_independent_of_x_E_has_real_solutions_l346_34651

-- Define the expression E(x)
noncomputable def E (x p : ℝ) : ℝ := (Real.sin x) ^ 6 + (Real.cos x) ^ 6 + p * ((Real.sin x) ^ 4 + (Real.cos x) ^ 4)

-- Theorem for part a
theorem E_independent_of_x (p : ℝ) :
  (∀ x, (deriv (fun x => E x p)) x = 0) ↔ p = -3/2 := by sorry

-- Theorem for part b
theorem E_has_real_solutions (p : ℝ) :
  (∃ x, E x p = 0) ↔ -1 ≤ p ∧ p ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_E_independent_of_x_E_has_real_solutions_l346_34651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_collection_l346_34670

/-- The amount of newspapers (in kilos) collected by each section in two weeks -/
def amount_per_section : ℕ → ℕ := sorry

/-- The number of sections collecting newspapers -/
def num_sections : ℕ := 6

/-- The target amount of newspapers to collect (in kilos) -/
def target : ℕ := 2000

/-- The amount still needed after three weeks (in kilos) -/
def amount_needed : ℕ := 320

theorem newspaper_collection (x : ℕ) :
  amount_per_section x = x →
  (num_sections + 1) * x = target - amount_needed →
  x = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_collection_l346_34670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_faucets_fill_time_l346_34667

/-- The time (in seconds) it takes for a given number of faucets to fill a tub of a given capacity, 
    given that five faucets can fill a 180-gallon tub in 8 minutes and all faucets dispense water 
    at the same rate. -/
def fill_time (num_faucets : ℕ) (tub_capacity : ℕ) : ℕ :=
  let five_faucets_time : ℕ := 8 * 60  -- 8 minutes in seconds
  let five_faucets_capacity : ℕ := 180  -- in gallons
  let single_faucet_rate : ℚ := (five_faucets_capacity : ℚ) / (five_faucets_time * 5 : ℚ)
  let total_rate : ℚ := single_faucet_rate * num_faucets
  ((tub_capacity : ℚ) / total_rate).ceil.toNat

/-- Theorem stating that ten faucets will fill a 90-gallon tub in 120 seconds. -/
theorem ten_faucets_fill_time : fill_time 10 90 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_faucets_fill_time_l346_34667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_line_l346_34669

noncomputable section

-- Define the ellipse C
def ellipse_C (θ : Real) : Real := 
  12 / (3 * Real.cos θ ^ 2 + 4 * Real.sin θ ^ 2)

-- Define the line l
def line_l (t : Real) : Real × Real :=
  (2 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Define the foci of the ellipse
def F1 : Real × Real := (-1, 0)
def F2 : Real × Real := (1, 0)

-- Define the distance function from a point to a line
def distance_point_to_line (p : Real × Real) (l : Real → Real × Real) : Real :=
  sorry -- Definition of distance from point to parametric line

-- Theorem statement
theorem sum_of_distances_to_line :
  distance_point_to_line F1 line_l + distance_point_to_line F2 line_l = 2 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_line_l346_34669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_k_l346_34658

/-- The function f(x) = x ln x + ax + b -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x * Real.log x + a * x + b

/-- The tangent line at (1, f(1)) is 3x - y - 2 = 0 -/
def tangent_line (a b : ℝ) : Prop := 3 * 1 - f a b 1 - 2 = 0

/-- There exists x > 0 such that k > f(x + 1) / x -/
def condition (a b : ℝ) (k : ℤ) : Prop := ∃ x : ℝ, x > 0 ∧ ↑k > (f a b (x + 1)) / x

theorem minimum_k (a b : ℝ) (k : ℤ) (h1 : tangent_line a b) (h2 : condition a b k) :
  k ≥ 5 ∧ ∃ k' : ℤ, k' = 5 ∧ condition a b k' := by
  sorry

#check minimum_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_k_l346_34658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_cos_minus_sin_sum_sin_l346_34660

theorem cos_sum_cos_minus_sin_sum_sin (x y : ℝ) : 
  Real.cos (x + y) * Real.cos y - Real.sin (x + y) * Real.sin y = Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_cos_minus_sin_sum_sin_l346_34660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_difference_l346_34645

/-- Define the distance to the river -/
def distance_to_river : ℚ := 2

/-- Define Susan's speed -/
def susan_speed : ℚ := 12

/-- Define Sam's speed -/
def sam_speed : ℚ := 5

/-- Define the function to calculate travel time in hours -/
def travel_time (distance : ℚ) (speed : ℚ) : ℚ := distance / speed

/-- Define the function to convert hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

/-- Theorem: The difference in arrival times is 14 minutes -/
theorem arrival_time_difference : 
  hours_to_minutes (travel_time distance_to_river sam_speed - travel_time distance_to_river susan_speed) = 14 := by
  -- Expand definitions
  unfold hours_to_minutes travel_time distance_to_river sam_speed susan_speed
  -- Simplify the expression
  simp [div_sub_div]
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_difference_l346_34645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_zero_l346_34668

/-- Area of a triangle given three points in 2D space -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- Given an ellipse with equation x²/4 + y² = 1 and foci F₁ and F₂,
    if P is a point on the ellipse such that the area of triangle F₁PF₂ is 1,
    then the dot product of PF₁ and PF₂ is 0. -/
theorem ellipse_dot_product_zero (F₁ F₂ P : ℝ × ℝ) :
  (∃ x y : ℝ, P = (x, y) ∧ x^2 / 4 + y^2 = 1) →  -- P is on the ellipse
  (∃ c : ℝ, F₁ = (-c, 0) ∧ F₂ = (c, 0) ∧ c^2 = 3) →  -- F₁ and F₂ are foci
  (area_triangle F₁ P F₂ = 1) →  -- Area of triangle F₁PF₂ is 1
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 :=  -- Dot product is 0
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_zero_l346_34668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xr_xu_ratio_l346_34656

-- Define the triangle XYZ
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

-- Define points P and Q
noncomputable def P (t : Triangle) : ℝ × ℝ := 
  ((6 * t.X.1 + 2 * t.Y.1) / 8, (6 * t.X.2 + 2 * t.Y.2) / 8)

noncomputable def Q (t : Triangle) : ℝ × ℝ := 
  ((5 * t.X.1 + 3 * t.Z.1) / 8, (5 * t.X.2 + 3 * t.Z.2) / 8)

-- Define U as the point where the angle bisector intersects YZ
noncomputable def U (t : Triangle) : ℝ × ℝ := 
  ((t.Y.1 + t.Z.1) / 2, (t.Y.2 + t.Z.2) / 2)

-- Define R as the intersection of XU and PQ
noncomputable def R (t : Triangle) : ℝ × ℝ := 
  ((12 * (P t).1 + 8 * (Q t).1) / 17, (12 * (P t).2 + 8 * (Q t).2) / 17)

-- Define the theorem
theorem xr_xu_ratio (t : Triangle) : 
  let xr := Real.sqrt ((R t).1 - t.X.1)^2 + ((R t).2 - t.X.2)^2
  let xu := Real.sqrt ((U t).1 - t.X.1)^2 + ((U t).2 - t.X.2)^2
  xr / xu = 6 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xr_xu_ratio_l346_34656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_arithmetic_geometric_sequence_l346_34621

/-- Predicate indicating that x, a, b, y form an arithmetic sequence -/
def is_arithmetic_sequence (x a b y : ℝ) : Prop :=
  b - a = y - b ∧ a - x = b - a

/-- Predicate indicating that x, a, b, y form a geometric sequence -/
def is_geometric_sequence (x a b y : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ a = x * r ∧ b = a * r ∧ y = b * r

/-- Given positive real numbers x and y, where x, a₁, a₂, y form an arithmetic sequence
    and x, b₁, b₂, y form a geometric sequence, the minimum value of ((a₁ + a₂) / √(b₁b₂))² is 4. -/
theorem min_value_arithmetic_geometric_sequence (x y a₁ a₂ b₁ b₂ : ℝ) 
    (hx : x > 0) (hy : y > 0)
    (ha : is_arithmetic_sequence x a₁ a₂ y)
    (hg : is_geometric_sequence x b₁ b₂ y) :
    ∀ ε > 0, ((a₁ + a₂) / Real.sqrt (b₁ * b₂))^2 ≥ 4 - ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_arithmetic_geometric_sequence_l346_34621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_and_collinear_vector_l346_34662

open BigOperators
open Real

def a : Fin 3 → ℝ := ![3, -6, 2]
def c : Fin 3 → ℝ := ![-1, 2, 0]
def c_new : Fin 3 → ℝ := fun i => c i + ![1, -2, 1] i

theorem orthogonal_and_collinear_vector :
  ∃! b : Fin 3 → ℝ, (∑ i, (a i) * (b i) = 0) ∧ 
  (∃ t : ℝ, ∀ i, b i = t * c_new i) ∧ 
  (∀ i, b i = 0) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_and_collinear_vector_l346_34662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_alpha_l346_34614

-- Define the function f(x) = (a+b)^x - a^x - b
noncomputable def f (a b x : ℝ) : ℝ := (a + b)^x - a^x - b

-- State the theorem
theorem least_alpha (a b : ℝ) (ha : a > 1) (hb : b > 0) :
  (∀ x ≥ 1, f a b x ≥ 0) ∧
  ∀ α < 1, ∃ x ≥ α, f a b x < 0 :=
by
  sorry

#check least_alpha

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_alpha_l346_34614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l346_34652

-- Define the function f
noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) + 1

-- State the theorem
theorem phi_range (ω φ : ℝ) : 
  (ω > 1) →
  (|φ| ≤ π/2) →
  (∀ x₁ x₂, x₁ < x₂ ∧ f ω φ x₁ = -1 ∧ f ω φ x₂ = -1 → x₂ - x₁ = π) →
  (∀ x ∈ Set.Ioo (-π/12) (π/3), f ω φ x > 1) →
  φ ∈ Set.Icc (π/6) (π/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l346_34652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corridor_single_layer_length_l346_34609

/-- Represents a painter's painting range on a corridor -/
structure PainterRange where
  start : ℝ
  length : ℝ

/-- The problem setup for the corridor painting scenario -/
structure CorridorPainting where
  corridor_length : ℝ
  painter1 : PainterRange
  painter2 : PainterRange

/-- Calculates the length of corridor painted with exactly one layer -/
def single_layer_length (cp : CorridorPainting) : ℝ :=
  sorry

/-- The main theorem stating the length of corridor painted with exactly one layer -/
theorem corridor_single_layer_length :
  let cp : CorridorPainting := {
    corridor_length := 15,
    painter1 := { start := 2, length := 9 },
    painter2 := { start := 4, length := 10 }
  }
  single_layer_length cp = 5 := by
  sorry

#check corridor_single_layer_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corridor_single_layer_length_l346_34609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_is_half_dollar_l346_34690

/-- Represents the dimensions of a storage box --/
structure BoxDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Calculates the volume of a box given its dimensions --/
def boxVolume (d : BoxDimensions) : ℚ := d.length * d.width * d.height

/-- Represents the storage problem --/
structure StorageProblem where
  boxDim : BoxDimensions
  totalVolume : ℚ
  totalMonthlyCost : ℚ

/-- Calculates the number of boxes given the total volume and box volume --/
def numberOfBoxes (p : StorageProblem) : ℚ :=
  p.totalVolume / boxVolume p.boxDim

/-- Calculates the cost per box per month --/
def costPerBoxPerMonth (p : StorageProblem) : ℚ :=
  p.totalMonthlyCost / numberOfBoxes p

/-- Theorem: The cost per box per month is $0.50 --/
theorem cost_per_box_is_half_dollar (p : StorageProblem) 
  (h1 : p.boxDim = { length := 15, width := 12, height := 10 })
  (h2 : p.totalVolume = 1080000)
  (h3 : p.totalMonthlyCost = 300) :
  costPerBoxPerMonth p = 1/2 := by
  sorry

#eval costPerBoxPerMonth {
  boxDim := { length := 15, width := 12, height := 10 },
  totalVolume := 1080000,
  totalMonthlyCost := 300
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_is_half_dollar_l346_34690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_angle_measure_l346_34691

/-- The sum of interior angles of a hexagon is 720 degrees -/
def hexagon_angle_sum : ℝ := 720

/-- Given angles in the hexagon -/
def angle1 : ℝ := 140
def angle2 : ℝ := 100
def angle3 : ℝ := 125
def angle4 : ℝ := 130
def angle5 : ℝ := 110

/-- Theorem: The measure of the sixth angle in the hexagon is 115 degrees -/
theorem sixth_angle_measure :
  hexagon_angle_sum - (angle1 + angle2 + angle3 + angle4 + angle5) = 115 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_angle_measure_l346_34691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_black_squares_l346_34681

/-- A floating plus on an n × n board -/
structure FloatingPlus (n : ℕ) where
  m : Fin n × Fin n
  l : Fin n × Fin n
  r : Fin n × Fin n
  a : Fin n × Fin n
  b : Fin n × Fin n
  h_row_l : l.2 = m.2 ∧ l.1 < m.1
  h_row_r : r.2 = m.2 ∧ m.1 < r.1
  h_col_a : a.1 = m.1 ∧ a.2 < m.2
  h_col_b : b.1 = m.1 ∧ m.2 < b.2

/-- A coloring of an n × n board -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- Predicate for a valid coloring without black floating plus -/
def ValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ fp : FloatingPlus n, ¬(c fp.m.1 fp.m.2 ∧ c fp.l.1 fp.l.2 ∧ c fp.r.1 fp.r.2 ∧ c fp.a.1 fp.a.2 ∧ c fp.b.1 fp.b.2)

/-- The number of black squares in a coloring -/
def BlackCount (n : ℕ) (c : Coloring n) : ℕ :=
  (Finset.univ.filter (fun i => (Finset.univ.filter (fun j => c i j)).card > 0)).card

/-- Main theorem: The maximum number of black squares without forming a floating plus is 4n-4 -/
theorem max_black_squares (n : ℕ) (h : n ≥ 3) :
  (∃ c : Coloring n, ValidColoring n c ∧ BlackCount n c = 4*n - 4) ∧
  (∀ c : Coloring n, ValidColoring n c → BlackCount n c ≤ 4*n - 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_black_squares_l346_34681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_abs_x_plus_a_implies_a_in_range_l346_34699

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then x^2 - 2*x + 4
  else (3/2)*x + 1/x

-- State the theorem
theorem f_geq_abs_x_plus_a_implies_a_in_range (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x + a|) → a ∈ Set.Icc (-15/4 : ℝ) (3/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_abs_x_plus_a_implies_a_in_range_l346_34699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_inequality_l346_34626

open BigOperators

def product_of_fractions (n : ℕ) : ℚ :=
  ∏ i in Finset.range n, (2 * (i + 1) : ℚ) / (2 * (i + 1) + 1)

theorem fraction_product_inequality :
  product_of_fractions 60 > (1 : ℚ) / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_inequality_l346_34626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_l346_34617

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O₂ (x y : ℝ) : Prop := 2*y^2 - 6*x - 8*y + 9 = 0

-- Define a function to count tangent lines
def count_tangent_lines : ℕ := 3

-- Theorem statement
theorem three_tangent_lines :
  count_tangent_lines = 3 :=
by
  -- The proof would go here, but we'll use sorry for now
  sorry

#check three_tangent_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_l346_34617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distances_theorem_l346_34619

/-- Represents a convex quadrilateral ABCD with given side lengths and midpoint distance --/
structure ConvexQuadrilateral where
  a : ℝ  -- length of side AB
  b : ℝ  -- length of side BC
  c : ℝ  -- length of side CD
  d : ℝ  -- length of side DA
  e : ℝ  -- distance between midpoints of AB and CD
  convex : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0

/-- The distance between midpoints of BC and DA --/
noncomputable def midpoint_distance_BC_DA (q : ConvexQuadrilateral) : ℝ :=
  Real.sqrt (q.e^2 + (q.a^2 + q.c^2)/2 - (q.b^2 + q.d^2)/2)

/-- The distance between midpoints of AA₀ and DC₀ --/
noncomputable def midpoint_distance_AA0_DC0 (q : ConvexQuadrilateral) : ℝ :=
  Real.sqrt ((q.d^2)/4 + (q.a^2 + q.e^2)/2 - (q.a^2/4 + q.c^2/4))

/-- Theorem stating the distances between midpoints for a specific quadrilateral --/
theorem midpoint_distances_theorem (q : ConvexQuadrilateral) 
  (h1 : q.a = 20) (h2 : q.b = 16) (h3 : q.c = 14) (h4 : q.d = 8) (h5 : q.e = 11) : 
  midpoint_distance_BC_DA q = Real.sqrt 259 ∧ 
  midpoint_distance_AA0_DC0 q = Real.sqrt 82.75 := by
  sorry

#check midpoint_distances_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distances_theorem_l346_34619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_league_equation_l346_34624

/-- Represents the number of teams in the soccer league -/
def x : ℕ := sorry

/-- The total number of matches played in the league -/
def total_matches : ℕ := 50

/-- The number of matches played when each pair of teams plays exactly once -/
def matches_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the equation correctly represents the situation -/
theorem soccer_league_equation : matches_played x = total_matches := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_league_equation_l346_34624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_notebook_cost_l346_34671

/-- Given the cost of pencils and notebooks in two different combinations,
    calculate the cost of another combination. -/
theorem pencil_notebook_cost
  (pencil_cost notebook_cost : ℚ)
  (cost1 : 9 * pencil_cost + 10 * notebook_cost = 5.06)
  (cost2 : 6 * pencil_cost + 4 * notebook_cost = 2.42)
  : 20 * pencil_cost + 14 * notebook_cost = 8.31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_notebook_cost_l346_34671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_length_l346_34634

theorem right_triangle_median_length : 
  ∀ (A B C A₁ P Q R : ℝ × ℝ),
  -- Right triangle ABC
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 10^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 6^2 →
  (C.1 - B.1) * (B.1 - A.1) + (C.2 - B.2) * (B.2 - A.2) = 0 →
  -- A₁ is on BC and is the intersection of angle bisector of A with BC
  (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ A₁ = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) ∧
  (A₁.1 - A.1) * (C.1 - A.1) + (A₁.2 - A.2) * (C.2 - A.2) =
  (A₁.1 - A.1) * (B.1 - A.1) + (A₁.2 - A.2) * (B.2 - A.2)) →
  -- Right triangle PQR
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (A₁.1 - B.1)^2 + (A₁.2 - B.2)^2 →
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = (A₁.1 - C.1)^2 + (A₁.2 - C.2)^2 →
  (R.1 - Q.1) * (Q.1 - P.1) + (R.2 - Q.2) * (Q.2 - P.2) = 0 →
  -- The length of the median from P to QR is 4√7/7
  let M := ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)
  (M.1 - P.1)^2 + (M.2 - P.2)^2 = (4 * Real.sqrt 7 / 7)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_length_l346_34634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sixth_term_l346_34622

/-- Given a geometric sequence with first term 1024 and 9th term 16,
    the 6th term is equal to 4√2 -/
theorem geometric_sequence_sixth_term :
  ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) = a n * (a 1)^(-(1/8 : ℝ))) →  -- Common ratio definition
  a 0 = 1024 →                                   -- First term
  a 8 = 16 →                                     -- 9th term (index 8)
  a 5 = 4 * Real.sqrt 2 := by                    -- 6th term (index 5)
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sixth_term_l346_34622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_fourth_plus_cos_squared_l346_34643

theorem cos_fourth_plus_cos_squared (α : Real) : 
  Real.sin α ^ 2 + Real.sin α = 1 → Real.cos α ^ 4 + Real.cos α ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_fourth_plus_cos_squared_l346_34643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_radius_l346_34600

/-- A circle in the Cartesian coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line x - y - 3 = 0 -/
def tangent_line (x y : ℝ) : Prop := x - y - 3 = 0

/-- The circle C satisfying the given conditions -/
noncomputable def circle_C : Circle :=
  { center := (2, -1),
    radius := Real.sqrt 2 }

theorem circle_C_radius : 
  (∃ (C : Circle), 
    (C.center.1 - 1)^2 + C.center.2^2 = C.radius^2 ∧ 
    (C.center.1 - 3)^2 + C.center.2^2 = C.radius^2 ∧
    (∃ (x y : ℝ), tangent_line x y ∧ 
      ((x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2))) →
  circle_C.radius = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_radius_l346_34600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l346_34664

/-- Represents the time taken (in hours) for a single machine to complete the job -/
structure MachineTime where
  r : ℚ
  b : ℚ
  c : ℚ

/-- Represents the number of machines of each type -/
structure MachineCount where
  r : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the time taken to complete the job when all machines work together -/
noncomputable def totalTime (t : MachineTime) (n : MachineCount) : ℚ :=
  315 / ((n.r : ℚ) * 63 / t.r + (n.b : ℚ) * 45 / t.b + (n.c : ℚ) * 35 / t.c)

/-- Theorem stating that given the conditions, the job will be completed in 315/457 hours -/
theorem job_completion_time :
  ∀ (t : MachineTime) (n : MachineCount),
    t.r = 5 ∧ t.b = 7 ∧ t.c = 9 ∧
    n.r = 4 ∧ n.b = 3 ∧ n.c = 2 →
    totalTime t n = 315 / 457 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l346_34664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_gain_is_25_percent_l346_34674

-- Define the initial quantities and prices
noncomputable def initial_milk_volume : ℝ := 1  -- 1 liter of milk
noncomputable def water_percentage : ℝ := 20    -- 20% water
noncomputable def cost_price : ℝ := 12          -- 12 rs
noncomputable def selling_price : ℝ := 15       -- 15 rs

-- Calculate the total volume after mixing
noncomputable def total_volume : ℝ := initial_milk_volume * (1 + water_percentage / 100)

-- Calculate the profit
noncomputable def profit : ℝ := selling_price - cost_price

-- Calculate the percentage gain
noncomputable def percentage_gain : ℝ := (profit / cost_price) * 100

-- Theorem to prove
theorem percentage_gain_is_25_percent : percentage_gain = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_gain_is_25_percent_l346_34674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_girls_avg_is_86_l346_34636

/-- Represents a school with its average scores -/
structure School where
  boys_avg : ℚ
  girls_avg : ℚ
  combined_avg : ℚ

/-- Calculates the overall average score for girls across multiple schools -/
def overall_girls_avg (schools : List School) : ℚ :=
  let total_score := schools.foldl (fun acc s => acc + s.girls_avg) 0
  total_score / schools.length

/-- The given data for the three schools -/
def adams : School := ⟨74, 81, 77⟩
def baker : School := ⟨83, 92, 86⟩
def carter : School := ⟨78, 85, 80⟩

/-- The list of all schools -/
def all_schools : List School := [adams, baker, carter]

/-- Theorem: The overall average score for girls across all three schools is 86 -/
theorem overall_girls_avg_is_86 : overall_girls_avg all_schools = 86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_girls_avg_is_86_l346_34636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l346_34638

theorem triangle_side_length (A B : ℝ) (a b : ℝ) :
  A = 30 * Real.pi / 180 →
  B = 45 * Real.pi / 180 →
  a = 2 →
  Real.sin A * b = Real.sin B * a →
  b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l346_34638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_distance_from_ceiling_l346_34640

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem fly_distance_from_ceiling (fly : Point3D) (P : Point3D) :
  P.x = 0 ∧ P.y = 0 ∧ P.z = 0 →  -- P is at the origin
  fly.x = 1 →                    -- fly is 1 meter from one wall
  fly.y = 8 →                    -- fly is 8 meters from the other wall
  distance fly P = 9 →           -- fly is 9 meters from point P
  fly.z = 4 :=                   -- fly is 4 meters from the ceiling
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_distance_from_ceiling_l346_34640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_distance_l346_34644

/-- A rectangle with a point inside it -/
structure RectangleWithPoint where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  P : ℝ × ℝ
  is_rectangle : A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2
  P_inside : P.1 > min A.1 B.1 ∧ P.1 < max A.1 B.1 ∧ P.2 > min A.2 D.2 ∧ P.2 < max A.2 D.2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem to be proved -/
theorem rectangle_point_distance (r : RectangleWithPoint) 
  (h1 : distance r.P r.A = 5)
  (h2 : distance r.P r.B = 3)
  (h3 : distance r.P r.C = 4) :
  distance r.P r.D = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_distance_l346_34644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_year_2012_facts_l346_34641

def year : Nat := 2012
def prc_established : Nat := 1949

def is_leap_year (y : Nat) : Bool :=
  y % 4 = 0 && (y % 100 ≠ 0 || y % 400 = 0)

def days_in_first_quarter (y : Nat) : Nat :=
  if is_leap_year y then 31 + 29 + 31 else 31 + 28 + 31

def years_since_prc_establishment (current_year : Nat) : Nat :=
  current_year - prc_established

theorem year_2012_facts :
  is_leap_year year = true ∧
  days_in_first_quarter year = 91 ∧
  years_since_prc_establishment year = 63 := by
  sorry

#eval is_leap_year year
#eval days_in_first_quarter year
#eval years_since_prc_establishment year

end NUMINAMATH_CALUDE_ERRORFEEDBACK_year_2012_facts_l346_34641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_specific_triangle_l346_34649

/-- Definition of a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Definition of a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Definition of the centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- Theorem: The centroid of triangle ABC with given coordinates is (3, 1) -/
theorem centroid_of_specific_triangle :
  let t : Triangle := { A := { x := 5, y := 5 },
                        B := { x := 8, y := -3 },
                        C := { x := -4, y := 1 } }
  centroid t = { x := 3, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_specific_triangle_l346_34649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l346_34693

/-- The area of a triangle with base 4 kilometers and height 2 kilometers is 4 square kilometers -/
theorem triangle_area (base height : Real) 
  (h1 : base = 4)
  (h2 : height = 2) :
  (1 / 2) * base * height = 4 := by
  -- Substitute the known values
  rw [h1, h2]
  -- Simplify the expression
  ring
  -- The proof is complete
  done

#check triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l346_34693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_is_integer_l346_34620

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Add this case to cover all natural numbers
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | (n + 4) => sequence_a (n + 3) - sequence_a (n + 2) + (sequence_a (n + 3))^2 / sequence_a (n + 1)

theorem sequence_a_is_integer : ∀ n : ℕ, ∃ k : ℤ, sequence_a n = k := by
  intro n
  induction n with
  | zero => 
    use 1
    rfl
  | succ n ih =>
    cases n with
    | zero => 
      use 1
      rfl
    | succ n =>
      cases n with
      | zero => 
        use 2
        rfl
      | succ n =>
        cases n with
        | zero => 
          use 3
          rfl
        | succ n =>
          sorry  -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_is_integer_l346_34620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l346_34650

theorem function_property (f g : ℤ → ℤ) 
  (h1 : ∀ m n : ℤ, f (m + f (f n)) = -f (f (m + 1) - n))
  (h2 : ∃ p : Polynomial ℤ, ∀ n : ℤ, g n = p.eval n)
  (h3 : ∀ n : ℤ, g n = g (f n)) :
  (∀ n : ℤ, f n = -n - 1) ∧ f 1991 = -1992 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l346_34650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_differences_theorem_l346_34682

theorem pairwise_differences_theorem (S : Finset ℕ) : 
  S.card = 20 → (∀ n ∈ S, n < 70) → 
  ∃ (a b c d e f g h : ℕ) (hab hcd hef hgh : ℕ), 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S ∧
    a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ g ≠ h ∧
    (max a b - min a b) = hab ∧ 
    (max c d - min c d) = hcd ∧ 
    (max e f - min e f) = hef ∧ 
    (max g h - min g h) = hgh ∧
    hab = hcd ∧ hcd = hef ∧ hef = hgh :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_differences_theorem_l346_34682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l346_34625

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the area of a trapezoid given its points -/
noncomputable def trapezoidArea (e f g h : Point) : ℝ :=
  let base1 := distance e f
  let base2 := distance g h
  let height := |g.x - e.x|
  (base1 + base2) * height / 2

/-- Theorem: The area of trapezoid EFGH with given coordinates is 12 + 18√5 -/
theorem trapezoid_area_theorem :
  let e := Point.mk 0 0
  let f := Point.mk 0 4
  let g := Point.mk 6 4
  let h := Point.mk 3 (-2)
  trapezoidArea e f g h = 12 + 18 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l346_34625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_items_is_eleven_l346_34653

/-- Represents the café purchase problem --/
structure CafePurchase where
  budget : ℚ
  sandwichCost : ℚ
  drinkCost : ℚ
  minDrinks : ℕ

/-- Calculates the maximum number of items that can be purchased --/
def maxItems (p : CafePurchase) : ℕ :=
  let maxSandwiches := (Int.floor ((p.budget - p.minDrinks * p.drinkCost) / p.sandwichCost)).toNat
  let remainingMoney := p.budget - maxSandwiches * p.sandwichCost
  let additionalDrinks := (Int.floor (remainingMoney / p.drinkCost)).toNat
  maxSandwiches + p.minDrinks + additionalDrinks

/-- Theorem stating that the maximum number of items is 11 --/
theorem max_items_is_eleven :
  ∀ (p : CafePurchase),
    p.budget = 40 ∧
    p.sandwichCost = 5 ∧
    p.drinkCost = 5/4 ∧
    p.minDrinks = 2 →
    maxItems p = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_items_is_eleven_l346_34653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_proof_l346_34695

-- Define the actual speed of the car
noncomputable def actual_speed : ℝ := 35

-- Define the reduced speed as a fraction of the actual speed
def reduced_speed_fraction : ℚ := 5/7

-- Define the distance covered
noncomputable def distance : ℝ := 42

-- Define the time taken in hours
noncomputable def time : ℝ := 1 + 40/60 + 48/3600

-- Theorem statement
theorem car_speed_proof :
  (actual_speed : ℝ) * (reduced_speed_fraction : ℝ) = distance / time :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_proof_l346_34695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_no_twenty_marked_subgrid_l346_34673

/-- A configuration of marked cells on a checkered plane -/
structure CellConfiguration where
  marked_cells : Finset (ℕ × ℕ)
  total_marked : Nat
  h_total : marked_cells.card = total_marked

/-- A rectangular subgrid on the checkered plane -/
structure RectangularSubgrid where
  top_left : ℕ × ℕ
  bottom_right : ℕ × ℕ
  h_valid : top_left.1 ≤ bottom_right.1 ∧ top_left.2 ≤ bottom_right.2

/-- The number of marked cells in a given subgrid -/
def marked_in_subgrid (config : CellConfiguration) (subgrid : RectangularSubgrid) : Nat :=
  (config.marked_cells.filter (fun cell =>
    subgrid.top_left.1 ≤ cell.1 ∧ cell.1 ≤ subgrid.bottom_right.1 ∧
    subgrid.top_left.2 ≤ cell.2 ∧ cell.2 ≤ subgrid.bottom_right.2
  )).card

/-- The main theorem stating that there exists a configuration where no subgrid contains exactly 20 marked cells -/
theorem exists_no_twenty_marked_subgrid :
  ∃ (config : CellConfiguration),
    config.total_marked = 40 ∧
    ∀ (subgrid : RectangularSubgrid),
      marked_in_subgrid config subgrid ≠ 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_no_twenty_marked_subgrid_l346_34673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_identity_l346_34631

theorem triangle_angle_identity (A B C : Real) 
  (h_triangle : A + B + C = Real.pi)
  (h_obtuse : A > Real.pi / 2)
  (h_eq1 : Real.cos A ^ 2 + Real.cos B ^ 2 + 2 * Real.sin B * Real.sin C * Real.cos A = 18/11)
  (h_eq2 : Real.cos C ^ 2 + Real.cos A ^ 2 + 2 * Real.sin C * Real.sin A * Real.cos B = 16/10) :
  Real.cos A ^ 2 + Real.cos B ^ 2 + 2 * Real.sin A * Real.sin B * Real.cos C = 19/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_identity_l346_34631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_quoted_value_theorem_l346_34633

/-- Represents a stock with a face value, dividend rate, and current yield -/
structure Stock where
  face_value : ℝ
  dividend_rate : ℝ
  current_yield : ℝ

/-- Calculates the quoted value of a stock -/
noncomputable def quoted_value (s : Stock) : ℝ :=
  s.face_value * s.dividend_rate / s.current_yield

/-- Theorem stating that for a stock with 10% dividend rate and 8% current yield, 
    the quoted value is 1.25 times the face value -/
theorem stock_quoted_value_theorem (s : Stock) 
    (h1 : s.dividend_rate = 0.1) 
    (h2 : s.current_yield = 0.08) : 
  quoted_value s = 1.25 * s.face_value := by
  sorry

#check stock_quoted_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_quoted_value_theorem_l346_34633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_triangle_exists_l346_34678

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of points in a unit square -/
def UnitSquarePoints : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- The area of a triangle formed by three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2

theorem small_triangle_exists (points : Set Point) 
    (h1 : points ⊆ UnitSquarePoints)
    (h2 : Fintype points)
    (h3 : Fintype.card points = 53) :
    ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ triangleArea p1 p2 p3 ≤ 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_triangle_exists_l346_34678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l346_34698

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Points A and B are the intersections of the hyperbola with a line 
    passing through its right focus and perpendicular to the x-axis -/
noncomputable def AB_length (h : Hyperbola) : ℝ := 2 * h.b^2 / h.a

/-- Points C and D are the intersections of the hyperbola's asymptotes with a line 
    passing through its right focus and perpendicular to the x-axis -/
noncomputable def CD_length (h : Hyperbola) : ℝ := 2 * h.b * h.a / h.a

/-- The main theorem -/
theorem hyperbola_eccentricity_bound (h : Hyperbola) 
  (h_inequality : AB_length h ≥ 3/5 * CD_length h) : 
  eccentricity h ≥ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l346_34698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_triangle_free_2021_l346_34630

/-- A graph with n vertices, each having degree k, and no triangles. -/
structure TriangleFreeRegularGraph (n : ℕ) (k : ℕ) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  degree_k : ∀ v, v ∈ vertices → (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = k
  no_triangles : ∀ a b c, a ∈ vertices → b ∈ vertices → c ∈ vertices → 
                 (a, b) ∈ edges → (b, c) ∈ edges → (a, c) ∉ edges

/-- The maximum degree in a triangle-free regular graph with 2021 vertices is 808. -/
theorem max_degree_triangle_free_2021 :
  ¬∃ (k : ℕ), k > 808 ∧ Nonempty (TriangleFreeRegularGraph 2021 k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_triangle_free_2021_l346_34630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excess_tax_rate_is_twenty_percent_l346_34677

/-- Represents the tax system of Country X -/
structure TaxSystem where
  baseTaxRate : ℚ  -- Tax rate for the first $40,000
  baseIncome : ℚ   -- The income threshold for the base tax rate
  totalIncome : ℚ  -- Total income of the citizen
  totalTax : ℚ     -- Total tax paid by the citizen

/-- Calculates the tax rate for income over the base income -/
def excessTaxRate (ts : TaxSystem) : ℚ :=
  ((ts.totalTax - ts.baseTaxRate * ts.baseIncome) / (ts.totalIncome - ts.baseIncome)) * 100

/-- Theorem stating that the excess tax rate is 20% given the specified conditions -/
theorem excess_tax_rate_is_twenty_percent (ts : TaxSystem) 
  (h1 : ts.baseTaxRate = 1/10)
  (h2 : ts.baseIncome = 40000)
  (h3 : ts.totalIncome = 60000)
  (h4 : ts.totalTax = 8000) :
  excessTaxRate ts = 20 := by
  sorry

#eval excessTaxRate {
  baseTaxRate := 1/10,
  baseIncome := 40000,
  totalIncome := 60000,
  totalTax := 8000
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excess_tax_rate_is_twenty_percent_l346_34677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_travel_theorem_l346_34618

/-- Calculates the distance traveled by a truck in yards -/
noncomputable def truck_distance (b : ℝ) (t : ℝ) : ℝ :=
  let normal_speed := b / 4  -- feet per t seconds
  let slow_speed := normal_speed / 2
  let total_time := 6 * 60  -- 6 minutes in seconds
  let half_time := total_time / 2
  let normal_distance := normal_speed * (half_time / t)
  let slow_distance := slow_speed * (half_time / t)
  (normal_distance + slow_distance) / 3  -- Convert feet to yards

/-- The truck's travel distance theorem -/
theorem truck_travel_theorem (b t : ℝ) (hb : b > 0) (ht : t > 0) :
  truck_distance b t = 22.5 * b / t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_travel_theorem_l346_34618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_without_circumcircle_existential_l346_34616

-- Define Quadrilateral as an inductive type
inductive Quadrilateral : Type
| mk : Quadrilateral

-- Define HasCircumscribedCircle as a proposition
def HasCircumscribedCircle (q : Quadrilateral) : Prop := sorry

-- Define ExistentialProposition
def ExistentialProposition (p : Prop) : Prop := p

theorem quadrilateral_without_circumcircle_existential :
  (∃ q : Quadrilateral, ¬ HasCircumscribedCircle q) → 
  ExistentialProposition (∃ q : Quadrilateral, ¬ HasCircumscribedCircle q) :=
by
  intro h
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_without_circumcircle_existential_l346_34616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_two_l346_34648

/-- A function f is odd if f(x) = -f(-x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

/-- The function f(x) = 1 + a / (e^(2x) - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 + a / (Real.exp (2 * x) - 1)

/-- If f(x) = 1 + a / (e^(2x) - 1) is an odd function, then a = 2 -/
theorem odd_function_implies_a_equals_two :
  ∃ a : ℝ, IsOdd (f a) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_two_l346_34648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l346_34605

/-- The coefficient of the term containing x in the expansion of (√x + 1/(2⁴√x))^n,
    given that the coefficients of the first three terms form an arithmetic sequence -/
theorem expansion_coefficient (x : ℝ) (n : ℕ) : 
  (∃ a d : ℝ, 
    (n.choose 0 : ℝ) * 1 = a ∧
    (n.choose 1 : ℝ) * (1/2) = a + d ∧
    (n.choose 2 : ℝ) * (1/4) = a + 2*d) →
  (∃ k : ℕ, (n.choose k : ℝ) * (1/2)^k * x^(n/2 - 3*k/4) = x) →
  (n.choose 4 : ℝ) * (1/2)^4 = 35/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l346_34605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l346_34623

theorem sqrt_inequality : Real.sqrt 8 - Real.sqrt 6 < Real.sqrt 5 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l346_34623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l346_34613

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x - a) / (2^x + a)

theorem f_properties :
  (∀ x : ℝ, f 1 (-x) = -(f 1 x)) ∧
  (∀ a : ℝ, a > 0 → ∀ x₁ x₂ : ℝ, x₁ > x₂ → f a x₁ > f a x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l346_34613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_minimum_condition_l346_34684

noncomputable def a (n : ℕ+) (c : ℝ) : ℝ := n + c / n

theorem sequence_minimum_condition (c : ℝ) :
  (∀ n : ℕ+, a n c ≥ a 3 c) ↔ c ∈ Set.Icc 6 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_minimum_condition_l346_34684
