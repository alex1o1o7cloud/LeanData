import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_ellipse_iff_max_chord_length_l393_39333

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 9 = 1
def line (x y m : ℝ) : Prop := y = (3/2) * x + m

-- Theorem for the range of m when the line intersects the ellipse
theorem line_intersects_ellipse_iff (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ m ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
sorry

-- Define the chord length function
noncomputable def chord_length (m : ℝ) : ℝ := (Real.sqrt 13 / 3) * Real.sqrt (-m^2 + 8)

-- Theorem for the maximum chord length
theorem max_chord_length :
  ∃ m : ℝ, ∀ m' : ℝ, chord_length m ≥ chord_length m' ∧ chord_length m = 2 * Real.sqrt 26 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_ellipse_iff_max_chord_length_l393_39333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l393_39300

/-- The first term of the sequence -/
def a₁ : ℝ := -8

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (n : ℕ) : ℝ := a₁ + (n - 1) * 2

/-- The property that a₁, a₃, a₄ form a geometric sequence -/
axiom geometric_property : (arithmetic_seq 3) ^ 2 = arithmetic_seq 1 * arithmetic_seq 4

/-- The sum of the first n terms of the sequence -/
noncomputable def S (n : ℕ) : ℝ := n * (arithmetic_seq 1 + arithmetic_seq n) / 2

theorem arithmetic_sequence_properties :
  (∀ n, arithmetic_seq n = 2 * n - 10) ∧ S 100 = 9100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l393_39300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_AB_equal_distances_condition_l393_39303

-- Define points A and B
def A : ℝ × ℝ := (-3, -4)
def B : ℝ × ℝ := (6, 3)

-- Define the line l: x + my + 1 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := x + m * y + 1 = 0

-- Define the perpendicular bisector of AB
def perp_bisector (x y : ℝ) : Prop := 9 * x + 7 * y - 10 = 0

-- Define the distance from a point to a line
noncomputable def distance_to_line (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  |m * p.2 + p.1 + 1| / Real.sqrt (1 + m^2)

theorem perpendicular_bisector_of_AB :
  perp_bisector = λ x y ↦ (x - A.1) * (B.1 - A.1) + (y - A.2) * (B.2 - A.2) = 
    ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 2 :=
by sorry

theorem equal_distances_condition (m : ℝ) :
  distance_to_line A m = distance_to_line B m ↔ m = 5 ∨ m = -9/7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_AB_equal_distances_condition_l393_39303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_factorial_product_l393_39370

theorem binomial_factorial_product : (Nat.choose 12 4) * Nat.factorial 4 = 11880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_factorial_product_l393_39370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_lines_theorem_l393_39327

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

noncomputable def quadrilateral_area (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : ℝ :=
  (1/2) * abs ((x₁*y₂ + x₂*y₃ + x₃*y₄ + x₄*y₁) - (y₁*x₂ + y₂*x₃ + y₃*x₄ + y₄*x₁))

def parallel_lines (m₁ m₂ : ℝ) : Prop :=
  m₁ = m₂

def line_equation (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + b

theorem ellipse_and_lines_theorem (a b : ℝ) (h₁ : a > b) (h₂ : b > 0)
  (h₃ : eccentricity a b = Real.sqrt 2 / 2)
  (h₄ : ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄, 
    ellipse_equation a b x₁ y₁ ∧ 
    ellipse_equation a b x₂ y₂ ∧
    ellipse_equation a b x₃ y₃ ∧
    ellipse_equation a b x₄ y₄ ∧
    quadrilateral_area x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ = 2 * Real.sqrt 2)
  (h₅ : ∃ m₁ b₁ m₂ b₂ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,
    parallel_lines m₁ m₂ ∧
    line_equation m₁ b₁ x₁ y₁ ∧
    line_equation m₁ b₁ x₂ y₂ ∧
    line_equation m₂ b₂ x₃ y₃ ∧
    line_equation m₂ b₂ x₄ y₄ ∧
    ellipse_equation a b x₁ y₁ ∧
    ellipse_equation a b x₂ y₂ ∧
    ellipse_equation a b x₃ y₃ ∧
    ellipse_equation a b x₄ y₄ ∧
    quadrilateral_area x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ = 8/3) :
  (a = Real.sqrt 2 ∧ b = 1) ∧
  ((∃ c, (∀ x y, line_equation 1 c x y ↔ x - y + c = 0) ∧
         (∀ x y, line_equation 1 (-c) x y ↔ x - y - c = 0)) ∨
   (∃ c, (∀ x y, line_equation (-1) c x y ↔ x + y + c = 0) ∧
         (∀ x y, line_equation (-1) (-c) x y ↔ x + y - c = 0))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_lines_theorem_l393_39327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l393_39379

def S : Finset ℚ := {1/5, 6/5, 11/5, 16/5}

theorem problem_solution :
  (∀ x ∈ S, (x - 1 ∈ S) ∨ (6*x - 1 ∈ S)) ∧
  (Finset.card S = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l393_39379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_with_smaller_diameter_l393_39359

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The diameter of a set of points -/
noncomputable def diameter (s : Set Point) : ℝ :=
  sSup {d | ∃ (p q : Point), p ∈ s ∧ q ∈ s ∧ distance p q = d}

/-- Theorem: For any finite set of points in the plane, there exists a point such that
    when removed, the remaining points can be partitioned into two subsets, each with
    a diameter strictly less than the original set's diameter -/
theorem partition_with_smaller_diameter (s : Set Point) (h : s.Finite) :
  ∃ (p : Point), p ∈ s ∧
    ∃ (s₁ s₂ : Set Point), s \ {p} = s₁ ∪ s₂ ∧
      diameter s₁ < diameter s ∧
      diameter s₂ < diameter s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_with_smaller_diameter_l393_39359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeenth_divisor_value_l393_39353

-- Define the type for positive integers
def PositiveInt := { n : ℕ // n > 0 }

-- Define the divisor function
def divisors (n : PositiveInt) : List PositiveInt := sorry

-- Define the order property for divisors
def divisors_ordered (n : PositiveInt) : Prop :=
  let divs := divisors n
  ∀ i j, i < j → i < divs.length → j < divs.length → 
    (divs.get ⟨i, sorry⟩).val < (divs.get ⟨j, sorry⟩).val

-- Define the Pythagorean triple property
def pythagorean_triple_property (n : PositiveInt) : Prop :=
  let divs := divisors n
  (divs.get ⟨6, sorry⟩).val^2 + (divs.get ⟨14, sorry⟩).val^2 = (divs.get ⟨15, sorry⟩).val^2

-- State the theorem
theorem seventeenth_divisor_value (n : PositiveInt) 
  (h1 : divisors_ordered n) 
  (h2 : pythagorean_triple_property n) : 
  (divisors n).get ⟨16, sorry⟩ = ⟨28, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeenth_divisor_value_l393_39353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_equality_l393_39352

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log (x^2) / Real.log (y^8)) * (Real.log (y^3) / Real.log (x^7)) * 
  (Real.log (x^4) / Real.log (y^5)) * (Real.log (y^5) / Real.log (x^4)) * 
  (Real.log (x^7) / Real.log (y^3)) = (1/4) * (Real.log x / Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_equality_l393_39352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_unique_solution_l393_39365

theorem factorial_equation_unique_solution :
  ∀ a b c : ℕ, 
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (Nat.factorial a * Nat.factorial b = Nat.factorial a + Nat.factorial b + Nat.factorial c) → 
  (a = 3 ∧ b = 3 ∧ c = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_unique_solution_l393_39365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l393_39339

/-- Represents the time taken for the entire work to be completed -/
noncomputable def total_time : ℝ := 5

/-- Represents the rate at which p completes the work -/
noncomputable def p_rate : ℝ := 1 / 10

/-- Represents the rate at which q completes the work -/
noncomputable def q_rate : ℝ := 1 / 6

/-- Represents the time after which q joined p -/
noncomputable def q_join_time : ℝ := 2

theorem work_completion_time :
  ∃ (work : ℝ),
    work > 0 ∧
    q_join_time * p_rate * work + 
    (total_time - q_join_time) * (p_rate + q_rate) * work = work :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l393_39339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_count_l393_39307

/-- The number of fruits in each basket --/
def basket1 : ℕ := 18  -- mangoes
def basket2 : ℕ := 10  -- pears
def basket3 : ℕ := 12  -- pawpaws
def basket4 : ℕ := 9   -- kiwi
def basket5 : ℕ := 9   -- lemons

/-- The total number of fruits in all baskets --/
def total_fruits : ℕ := 58

/-- The number of baskets --/
def num_baskets : ℕ := 5

theorem lemon_count :
  basket4 = basket5 ∧
  basket1 + basket2 + basket3 + basket4 + basket5 = total_fruits →
  basket5 = 9 := by
  intro h
  cases h with
  | intro h1 h2 =>
    rw [h1] at h2
    simp [basket1, basket2, basket3, basket4, basket5, total_fruits] at h2
    rfl

#eval basket5  -- This will output 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_count_l393_39307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_textbook_writing_experts_l393_39334

/-- The number of days it takes the initial group to write the textbook -/
def initial_days : ℕ := 24

/-- The number of days it takes the group with one additional expert to write the textbook -/
def additional_expert_days : ℕ := 18

/-- Represents the work rate of a single expert in terms of textbooks per day -/
noncomputable def expert_work_rate (x : ℝ) : ℝ := 1 / (initial_days * x)

/-- The number of initial experts working on the textbook -/
def initial_experts : ℕ := 3

theorem textbook_writing_experts :
  (expert_work_rate (initial_experts : ℝ)) * initial_experts =
  (expert_work_rate ((initial_experts : ℝ) + 1)) * (initial_experts + 1) ∧
  (expert_work_rate ((initial_experts : ℝ) + 1)) * (initial_experts + 1) * additional_expert_days = 1 :=
by sorry

#eval initial_experts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_textbook_writing_experts_l393_39334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l393_39380

/-- Given a triangle ABC with points M and N such that AM and CN intersect at P,
    prove that x + y = 4/7 where AP = x*AB + y*AC. -/
theorem intersection_point_coordinates (A B C M N P : EuclideanSpace ℝ (Fin 2)) 
  (x y : ℝ) : 
  (M - A) = (3/4 : ℝ) • (B - A) + (1/4 : ℝ) • (C - A) →
  (N - C) = (1/2 : ℝ) • (B - C) + (1/2 : ℝ) • (A - C) →
  (P - A) = x • (B - A) + y • (C - A) →
  x + y = 4/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l393_39380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_topmost_multiple_of_five_l393_39396

/-- Represents a triangular arrangement of numbers -/
structure TriangularArrangement where
  bottom_row : List ℤ
  is_valid : bottom_row.length ≥ 2

/-- Checks if the sum of three numbers is divisible by 5 -/
def sum_divisible_by_five (a b c : ℤ) : Prop :=
  (a + b + c) % 5 = 0

/-- Represents a valid arrangement satisfying the given conditions -/
def ValidArrangement (arr : TriangularArrangement) : Prop :=
  arr.bottom_row.head? = some 12 ∧
  arr.bottom_row.reverse.head? = some 3 ∧
  ∀ (i : ℕ), i + 2 < arr.bottom_row.length →
    ∀ (a b c : ℤ), (arr.bottom_row.get? i = some a) →
                    (arr.bottom_row.get? (i + 1) = some b) →
                    (arr.bottom_row.get? (i + 2) = some c) →
                    sum_divisible_by_five a b c

/-- Helper function to calculate the topmost number -/
def get_topmost_number (arr : TriangularArrangement) : ℤ :=
  -(5 * (1 + arr.bottom_row.foldl (· + ·) 0))

/-- The theorem to be proved -/
theorem topmost_multiple_of_five (arr : TriangularArrangement) (h : ValidArrangement arr) :
  ∃ (k : ℤ), get_topmost_number arr = 5 * k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_topmost_multiple_of_five_l393_39396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_greater_if_sine_greater_l393_39369

-- Define IsTriangle and OppositeAngle as variables
variable (IsTriangle : Real → Real → Real → Prop)
variable (OppositeAngle : Real → Real → Real → Prop)

-- Define Sin as a function
noncomputable def Sin : Real → Real := sorry

theorem angle_greater_if_sine_greater (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  IsTriangle A B C →
  -- a, b, c are sides opposite to angles A, B, C respectively
  OppositeAngle A b c ∧ OppositeAngle B a c ∧ OppositeAngle C a b →
  -- If sine of A is greater than sine of B
  Sin A > Sin B →
  -- Then angle A is greater than angle B
  A > B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_greater_if_sine_greater_l393_39369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l393_39371

def scores : List ℝ := [121, 127, 123, 124, 125]

theorem variance_of_scores : 
  let mean := (List.sum scores) / scores.length
  let squared_diff := scores.map (λ x => (x - mean) ^ (2 : ℕ))
  (List.sum squared_diff) / scores.length = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l393_39371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equal_f_2001_l393_39305

noncomputable def f (x : ℝ) : ℝ :=
  if 2 ≤ x ∧ x ≤ 6 then 2 - |x - 4| else 0  -- Default value for x outside [2, 6]

axiom f_scale (x : ℝ) (h : 0 < x) : f (5 * x) = 5 * f x

theorem smallest_x_equal_f_2001 :
  ∃ (x : ℝ), 0 < x ∧ f x = f 2001 ∧ ∀ (y : ℝ), 0 < y → f y = f 2001 → x ≤ y := by
  sorry

#check smallest_x_equal_f_2001

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equal_f_2001_l393_39305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l393_39394

noncomputable section

-- Define the parabola C: x^2 = 2py (p > 0)
def C (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define points O, A, and B
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (0, -1)

-- Define the distance between two points
def my_dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem parabola_properties (p : ℝ) (h1 : C p A.1 A.2) :
  -- 1. The line AB is tangent to C
  (∃ k b : ℝ, ∀ x y : ℝ, C p x y → y = k*x + b → (x, y) = A) ∧
  -- 2. For any line through B intersecting C at P and Q, |OP| * |OQ| > |OA|^2
  (∀ P Q : ℝ × ℝ, C p P.1 P.2 → C p Q.1 Q.2 → 
    (∃ m : ℝ, P.2 - B.2 = m*(P.1 - B.1) ∧ Q.2 - B.2 = m*(Q.1 - B.1)) →
    my_dist O P * my_dist O Q > my_dist O A * my_dist O A) ∧
  -- 3. For any line through B intersecting C at P and Q, |BP| * |BQ| > |BA|^2
  (∀ P Q : ℝ × ℝ, C p P.1 P.2 → C p Q.1 Q.2 → 
    (∃ m : ℝ, P.2 - B.2 = m*(P.1 - B.1) ∧ Q.2 - B.2 = m*(Q.1 - B.1)) →
    my_dist B P * my_dist B Q > my_dist B A * my_dist B A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l393_39394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ratio_form_l393_39304

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_ratio (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (double_factorial (2*i + 1) : ℚ) / (double_factorial (2*i + 2) : ℚ))

theorem sum_ratio_form : ∃ c : ℕ, sum_ratio 1005 = (c : ℚ) / ((2^(2*1005)) : ℚ) ∧ 
  c % 2^4 = 0 ∧ c % 2^5 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ratio_form_l393_39304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_231_count_l393_39340

theorem multiples_of_231_count : ∃ (count : ℕ), count = 784 ∧
  count = Finset.card (Finset.filter (fun p : ℕ × ℕ => 
    let (i, j) := p
    0 ≤ i ∧ i < j ∧ j ≤ 99 ∧ (8^j - 8^i) % 231 = 0) (Finset.product (Finset.range 100) (Finset.range 100))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_231_count_l393_39340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_distribution_l393_39312

/-- Represents the distribution of students in venues A and B -/
structure VenueDistribution where
  venue_a_grade10 : Rat
  venue_a_grade11 : Rat
  venue_a_grade12 : Rat
  venue_b_grade10 : Rat
  venue_b_grade11 : Rat
  venue_b_grade12 : Rat

/-- The main theorem about the student distribution -/
theorem student_distribution 
  (dist : VenueDistribution)
  (venue_a_percent : Rat)
  (venue_b_percent : Rat)
  (grade11_venue_b : Nat) :
  dist.venue_a_grade10 = 1/2 ∧ 
  dist.venue_a_grade11 = 2/5 ∧ 
  dist.venue_a_grade12 = 1/10 ∧
  dist.venue_b_grade10 = 2/5 ∧
  dist.venue_b_grade11 = 1/2 ∧
  dist.venue_b_grade12 = 1/10 ∧
  venue_a_percent = 1/4 ∧
  venue_b_percent = 3/4 ∧
  grade11_venue_b = 75 →
  ∃ (x y z : Rat) (n : Nat),
    x / (x + y + z) = 17/40 ∧
    y / (x + y + z) = 19/40 ∧
    z / (x + y + z) = 1/10 ∧
    n = 200 ∧
    (n : Rat) * venue_a_percent * dist.venue_a_grade10 = 25 ∧
    (n : Rat) * venue_a_percent * dist.venue_a_grade11 = 20 ∧
    (n : Rat) * venue_a_percent * dist.venue_a_grade12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_distribution_l393_39312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_tripling_time_l393_39391

/-- The number of years required for a population to triple, given an annual growth rate of 1/50 -/
noncomputable def years_to_triple : ℝ := Real.log 3 / Real.log (51/50)

/-- Theorem stating that the number of years for the population to triple is approximately 55 -/
theorem population_tripling_time : 
  ⌊years_to_triple⌋ = 55 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_tripling_time_l393_39391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l393_39362

theorem equation_solution :
  let solutions : List (ℂ) := [
    4 + Complex.I * (4 + 2 * Real.sqrt 2).sqrt,
    4 - Complex.I * (4 + 2 * Real.sqrt 2).sqrt,
    4 + Complex.I * (4 - 2 * Real.sqrt 2).sqrt,
    4 - Complex.I * (4 - 2 * Real.sqrt 2).sqrt
  ]
  ∀ x ∈ solutions, (x - 2)^4 + (x - 6)^4 = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l393_39362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_1985th_row_l393_39302

/-- Triangular array (a_{n,k}) -/
def a : ℕ → ℕ → ℚ
  | n, 0 => 0  -- Add this case to handle k = 0
  | n, 1 => 1 / n
  | n, k+1 => a (n-1) k - a n k

/-- Harmonic mean of a list of rational numbers -/
def harmonicMean (l : List ℚ) : ℚ :=
  l.length / (l.map (λ x => 1/x)).sum

/-- The 1985th row of the triangular array -/
def row1985 : List ℚ := List.range 1985 |>.map (λ k => a 1985 (k+1))

theorem harmonic_mean_1985th_row :
  harmonicMean row1985 = 1 / (2^1984) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_1985th_row_l393_39302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_running_time_l393_39335

def total_laps : ℕ := 4
def lap_distance : ℝ := 600
def first_segment_distance : ℝ := 200
def second_segment_distance : ℝ := 400
def first_segment_speed : ℝ := 3
def second_segment_speed : ℝ := 6

theorem paul_running_time :
  let first_segment_time := first_segment_distance / first_segment_speed
  let second_segment_time := second_segment_distance / second_segment_speed
  let total_time := total_laps * (first_segment_time + second_segment_time)
  ∃ (ε : ℝ), abs (total_time - 533.33) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_running_time_l393_39335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_theorem_l393_39374

theorem tan_sum_theorem (x y : ℝ) 
  (h1 : Real.sin x / Real.cos y + Real.sin y / Real.cos x = 2)
  (h2 : Real.cos x / Real.sin y + Real.cos y / Real.sin x = 3) :
  Real.tan x / Real.tan y + Real.tan y / Real.tan x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_theorem_l393_39374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Br_in_AlBr3_l393_39325

/-- The molar mass of aluminum in g/mol -/
noncomputable def molar_mass_Al : ℝ := 26.98

/-- The molar mass of bromine in g/mol -/
noncomputable def molar_mass_Br : ℝ := 79.90

/-- The number of bromine atoms in aluminum bromide -/
def num_Br_atoms : ℕ := 3

/-- The molar mass of aluminum bromide in g/mol -/
noncomputable def molar_mass_AlBr3 : ℝ := molar_mass_Al + num_Br_atoms * molar_mass_Br

/-- The mass percentage of bromine in aluminum bromide -/
noncomputable def mass_percentage_Br : ℝ := (num_Br_atoms * molar_mass_Br) / molar_mass_AlBr3 * 100

theorem mass_percentage_Br_in_AlBr3 :
  ∃ ε > 0, |mass_percentage_Br - 89.89| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Br_in_AlBr3_l393_39325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l393_39382

noncomputable def f (x : ℝ) : ℝ := Real.sin (2005 * Real.pi / 2 - 2004 * x)

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  simp [f]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l393_39382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_ratio_is_negative_one_l393_39342

/-- Represents a cone with a cross-sectional area parallel to the base -/
structure Cone where
  height : ℝ
  base_area : ℝ
  cross_section_height : ℝ
  cross_section_area : ℝ

/-- The cross-sectional area is half of the base area -/
axiom cross_section_half_base (c : Cone) : c.cross_section_area = c.base_area / 2

/-- The ratio of upper to lower segments of the height divided by the cross-section -/
noncomputable def height_ratio (c : Cone) : ℝ := c.cross_section_height / (c.height - c.cross_section_height)

/-- Theorem stating that the height ratio is -1 -/
theorem height_ratio_is_negative_one (c : Cone) : height_ratio c = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_ratio_is_negative_one_l393_39342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_b_squared_l393_39381

-- Define the circles w3 and w4
def w3 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y - 23 = 0
def w4 (x y : ℝ) : Prop := x^2 + y^2 + 12*x + 4*y - 63 = 0

-- Define a circle with center (h, k) and radius r
def circle_eq (h k r : ℝ) (x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Define external tangency
def externally_tangent (c1 c2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ x y r1 r2, c1 x y ∧ c2 x y ∧ r1 + r2 = ((x + 6)^2 + (y + 2)^2).sqrt

-- Define internal tangency
def internally_tangent (c1 c2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ x y r1 r2, c1 x y ∧ c2 x y ∧ |r1 - r2| = ((x - 4)^2 + (y - 3)^2).sqrt

-- Main theorem
theorem smallest_positive_b_squared :
  ∃ b : ℝ, b > 0 ∧
    (∀ b' : ℝ, b' > 0 → b' < b →
      ¬∃ h k r, circle_eq h k r h (b'*h) ∧
        externally_tangent (circle_eq h k r) w4 ∧
        internally_tangent (circle_eq h k r) w3) ∧
    (∃ h k r, circle_eq h k r h (b*h) ∧
      externally_tangent (circle_eq h k r) w4 ∧
      internally_tangent (circle_eq h k r) w3) ∧
    b^2 = 145/460 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_b_squared_l393_39381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_f_positive_range_l393_39301

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m * x^2 - x) * Real.log x + (1/2) * m * x^2

-- Part I: Tangent line when m = 0
theorem tangent_line_at_one (x : ℝ) :
  (fun x => -x * Real.log x) x - (fun x => -x * Real.log x) 1 = -x + 1 := by sorry

-- Part II: Range of m for f(x) > 0
theorem f_positive_range (m : ℝ) :
  (∀ x > 0, f m x > 0) ↔ (1 / (2 * Real.sqrt (Real.exp 1)) < m ∧ m ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_f_positive_range_l393_39301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_center_of_symmetry_l393_39399

noncomputable def f (x : ℝ) : ℝ := Real.sin (4*x + 7*Real.pi/3) / Real.sin (2*x + 2*Real.pi/3)

theorem f_center_of_symmetry :
  ∀ k : ℤ, ∃ x : ℝ, 
    (Real.sin (2*x + 2*Real.pi/3) ≠ 0) →
    (f (x + (k*Real.pi/2 - Real.pi/12)) = -f (-(x - (k*Real.pi/2 - Real.pi/12)))) ∧
    (k*Real.pi/2 - Real.pi/12, 0) = (k*Real.pi/2 - Real.pi/12, 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_center_of_symmetry_l393_39399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_post_height_is_15_feet_l393_39336

/-- Represents the properties of a cylindrical post and a squirrel's path on it -/
structure PostAndPath where
  postCircumference : ℝ
  squirrelTravelDistance : ℝ
  risePerCircuit : ℝ

/-- Calculates the height of the post based on the given properties -/
noncomputable def calculatePostHeight (props : PostAndPath) : ℝ :=
  (props.squirrelTravelDistance / props.risePerCircuit) * props.risePerCircuit

/-- Theorem stating that under the given conditions, the post height is 15 feet -/
theorem post_height_is_15_feet (props : PostAndPath) 
  (h1 : props.postCircumference = 3)
  (h2 : props.squirrelTravelDistance = 15)
  (h3 : props.risePerCircuit = 5) :
  calculatePostHeight props = 15 := by
  sorry

#check post_height_is_15_feet

end NUMINAMATH_CALUDE_ERRORFEEDBACK_post_height_is_15_feet_l393_39336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l393_39357

noncomputable def curve (x : ℝ) : ℝ := (3/2) * x^2 - Real.log x

def line (x : ℝ) : ℝ := 2 * x - 1

noncomputable def min_distance : ℝ := Real.sqrt 5 / 10

theorem min_distance_theorem :
  ∀ x y : ℝ, x > 0 → y = curve x →
  ∃ d : ℝ, d ≥ 0 ∧
    (∀ x' y' : ℝ, (y' = line x') →
      d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) ∧
    d = min_distance := by
  sorry

#check min_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l393_39357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_trips_to_fill_tank_l393_39348

/-- The number of trips Jill makes to fill the tank -/
def jill_trips (tank_capacity bucket_capacity : ℕ) 
               (jack_buckets_per_trip jill_buckets_per_trip : ℕ)
               (jack_trips_per_jill_trips : ℚ) : ℕ :=
  let total_buckets := tank_capacity / bucket_capacity
  let buckets_per_jill_trips := 
    (jack_buckets_per_trip * (jack_trips_per_jill_trips * 2).num + 
     jill_buckets_per_trip * (jack_trips_per_jill_trips * 2).den)
  let sets_of_jill_trips := total_buckets / buckets_per_jill_trips
  (2 * sets_of_jill_trips).toNat

/-- Theorem stating that Jill makes 30 trips to fill the tank -/
theorem jill_trips_to_fill_tank : 
  jill_trips 600 5 2 1 (3/2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_trips_to_fill_tank_l393_39348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_positivity_implies_a_bound_l393_39320

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + a) / x

/-- The theorem statement -/
theorem function_positivity_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x > 0) → a > -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_positivity_implies_a_bound_l393_39320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l393_39375

theorem coefficient_x_cubed_in_expansion : ∃ c : ℕ, 
  c = Nat.choose 6 3 ∧ 
  c = (Finset.range 7).sum (λ k ↦ Nat.choose 6 k * (1 : ℕ)^(6 - k) * 1^k) ∧ 
  c = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l393_39375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_given_line_l393_39389

/-- A line in the rectangular coordinate system given by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The x-intercept of a parametric line -/
noncomputable def xIntercept (l : ParametricLine) : ℝ :=
  l.x (solve_for_t l.y 0)
where
  solve_for_t (y : ℝ → ℝ) (value : ℝ) : ℝ :=
    (value - y 0) / (y 1 - y 0)

/-- The given line -/
def givenLine : ParametricLine :=
  { x := λ t => 2 + 3 * t
  , y := λ t => -1 + 5 * t }

theorem x_intercept_of_given_line :
  xIntercept givenLine = 2.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_given_line_l393_39389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l393_39323

-- Define the ellipse C
noncomputable def ellipse_C (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

-- Define the focus F
def F : ℝ × ℝ := (-2, 0)

-- Define eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 3

-- Define point T
def T (m : ℝ) : ℝ × ℝ := (-3, m)

-- Define the perpendicular condition
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  (P.1 - F.1) * (Q.1 - P.1) + (P.2 - F.2) * (Q.2 - P.2) = 0

-- Define the parallelogram condition
def is_parallelogram (O P Q T : ℝ × ℝ) : Prop :=
  P.1 + Q.1 = O.1 + T.1 ∧ P.2 + Q.2 = O.2 + T.2

-- Theorem statement
theorem ellipse_properties :
  ∃ (P Q : ℝ × ℝ) (m : ℝ),
    ellipse_C P.1 P.2 ∧
    ellipse_C Q.1 Q.2 ∧
    perpendicular P Q ∧
    is_parallelogram (0, 0) P Q (T m) ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 12 := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l393_39323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindys_speed_is_12_l393_39385

/-- The speed of Lindy the dog given the initial conditions of Jack and Christina's walk --/
noncomputable def lindys_speed (initial_distance : ℝ) (jack_speed : ℝ) (christina_speed : ℝ) (lindy_distance : ℝ) : ℝ :=
  let meeting_time := initial_distance / (jack_speed + christina_speed)
  lindy_distance / meeting_time

theorem lindys_speed_is_12 :
  lindys_speed 360 5 7 360 = 12 := by
  unfold lindys_speed
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindys_speed_is_12_l393_39385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_quadrilateral_volume_l393_39328

/-- A tetrahedron with vertices A, B, C, D -/
structure Tetrahedron (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)

/-- The point of intersection of the medians of a tetrahedron -/
def median_intersection {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (T : Tetrahedron V) : V :=
  (1/4) • (T.A + T.B + T.C + T.D)

/-- The volume of a tetrahedron -/
noncomputable def volume {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (T : Tetrahedron V) : ℝ := sorry

/-- A quadrilateral formed by connecting the median intersection to the vertices -/
def median_quadrilateral {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (T : Tetrahedron V) : List V :=
  [median_intersection T, T.A, T.B, T.C]

theorem median_quadrilateral_volume {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (T : Tetrahedron V) (vol : ℝ) :
  volume T = vol →
  volume (Tetrahedron.mk
    (median_intersection T)
    (median_intersection T + T.A - median_intersection T)
    (median_intersection T + T.B - median_intersection T)
    (median_intersection T + T.C - median_intersection T)) = vol / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_quadrilateral_volume_l393_39328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_quadrilateral_l393_39388

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Calculates the dot product of two 2D vectors -/
def dotProduct (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Calculates the magnitude of a 2D vector -/
noncomputable def magnitude (v : Vector2D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

/-- Represents a convex quadrilateral OABC with diagonals OB and AC -/
structure ConvexQuadrilateral where
  OB : Vector2D
  AC : Vector2D

/-- Calculates the area of a convex quadrilateral given its diagonals -/
noncomputable def quadrilateralArea (q : ConvexQuadrilateral) : ℝ :=
  (1 / 2) * magnitude q.OB * magnitude q.AC

theorem area_of_specific_quadrilateral :
  let q := ConvexQuadrilateral.mk (Vector2D.mk 2 4) (Vector2D.mk (-2) 1)
  quadrilateralArea q = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_quadrilateral_l393_39388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_real_l393_39350

-- Define the function f(x) = log₃(-x)
noncomputable def f (x : ℝ) : ℝ := Real.log (-x) / Real.log 3

-- State the theorem
theorem range_of_f_is_real :
  ∀ y : ℝ, ∃ x : ℝ, x < 0 ∧ f x = y :=
by sorry

-- Note: The condition x < 0 represents the domain of f(x), which is (-∞, 0)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_real_l393_39350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_expression_l393_39376

theorem range_of_expression (A B C : ℝ × ℝ) (lambda mu : ℝ) :
  A ≠ B ∧ A ≠ C ∧ B ≠ C →
  A.1^2 + A.2^2 = 1 →
  B.1^2 + B.2^2 = 1 →
  C.1^2 + C.2^2 = 1 →
  lambda > 0 →
  mu > 0 →
  C = (lambda * A.1 + mu * B.1, lambda * A.2 + mu * B.2) →
  ∀ x : ℝ, x > 2 → ∃ y : ℝ, lambda^2 + (mu - 3)^2 = y ∧ y > 2 ∧ y < x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_expression_l393_39376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_range_l393_39331

-- Define the function y = log_a(2-ax)
noncomputable def y (a x : ℝ) : ℝ := Real.log (2 - a * x) / Real.log a

-- Define the property of y being a decreasing function on [0,1]
def isDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b → f x₂ < f x₁

-- State the theorem
theorem log_function_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  isDecreasing (y a) 0 1 → 1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_range_l393_39331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_conditions_l393_39324

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- Represents a parabola in the form y = kx^2 -/
structure Parabola where
  k : ℝ
  h_positive : 0 < k

/-- The focus of a parabola y = kx^2 is at (0, 1/(4k)) -/
noncomputable def parabola_focus (p : Parabola) : ℝ × ℝ := (0, 1 / (4 * p.k))

/-- The standard equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  y^2 / e.a^2 + x^2 / e.b^2 = 1

/-- The major axis of an ellipse is 2a -/
def major_axis (e : Ellipse) : ℝ := 2 * e.a

/-- Theorem: Given the conditions, the ellipse has the specified equation -/
theorem ellipse_equation_from_conditions (p : Parabola) (e : Ellipse) 
  (h_major_axis : major_axis e = 4)
  (h_focus : ∃ (f : ℝ × ℝ), f = parabola_focus p ∧ f.2 = 1) :
  e.a = 2 ∧ e.b = Real.sqrt 3 ∧ ∀ x y, ellipse_equation e x y ↔ y^2/4 + x^2/3 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_conditions_l393_39324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l393_39363

theorem constant_term_expansion : ∃ c : ℕ, c = 787500 ∧ 
  c = (Finset.range 11).sum (fun k => Nat.choose 10 k * (5^(10 - k)) * (1^k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l393_39363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_approx_l393_39315

/-- The radius of the circular table in feet -/
def table_radius : ℝ := 5

/-- The number of place mats on the table -/
def num_mats : ℕ := 8

/-- The width of each place mat in feet -/
def mat_width : ℝ := 1

/-- The length of each place mat in feet -/
noncomputable def mat_length : ℝ := 2 * table_radius * Real.sin (5 * Real.pi / 8)

/-- Theorem stating that the length of each place mat is approximately 3.83 feet -/
theorem place_mat_length_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |mat_length - 3.83| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_approx_l393_39315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_point_conversion_and_distance_l393_39308

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

noncomputable def distance_from_origin (x y z : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 + z^2)

theorem cylindrical_point_conversion_and_distance :
  let r : ℝ := 7
  let θ : ℝ := 5 * Real.pi / 6
  let z : ℝ := -3
  let (x, y, z') := cylindrical_to_rectangular r θ z
  x = -7 * Real.sqrt 3 / 2 ∧
  y = 7 / 2 ∧
  z' = -3 ∧
  distance_from_origin x y z' = Real.sqrt 58 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_point_conversion_and_distance_l393_39308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_range_l393_39321

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (3*a - 2)*x + 1 else a^x

theorem decreasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ (1/2 ≤ a ∧ a < 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_range_l393_39321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_universal_set_equality_l393_39368

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 7}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {3, 5}

theorem universal_set_equality : U = M ∪ (U \ N) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_universal_set_equality_l393_39368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archer_shots_per_day_l393_39330

/-- Represents the archer's shooting practice scenario -/
structure ArcherPractice where
  days_per_week : ℕ
  recovery_rate : ℚ
  arrow_cost : ℚ
  team_payment_rate : ℚ
  weekly_spending : ℚ

/-- Calculates the number of shots per day for the archer -/
def shots_per_day (practice : ArcherPractice) : ℚ :=
  practice.weekly_spending / 
  ((1 - practice.recovery_rate) * practice.arrow_cost * (1 - practice.team_payment_rate)) /
  practice.days_per_week

/-- Theorem stating that the archer shoots 200 shots per day -/
theorem archer_shots_per_day :
  let practice : ArcherPractice := {
    days_per_week := 4,
    recovery_rate := 1/5,
    arrow_cost := 11/2,
    team_payment_rate := 7/10,
    weekly_spending := 1056
  }
  shots_per_day practice = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archer_shots_per_day_l393_39330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_15_minutes_l393_39310

/-- The distance between Hyosung and Mimi after 15 minutes of walking towards each other -/
theorem distance_after_15_minutes
  (total_distance : ℝ)
  (hyosung_speed : ℝ)
  (mimi_speed_per_hour : ℝ)
  (time : ℝ)
  (h1 : total_distance = 2.5)
  (h2 : hyosung_speed = 0.08)
  (h3 : mimi_speed_per_hour = 2.4)
  (h4 : time = 15) :
  total_distance - (hyosung_speed + mimi_speed_per_hour / 60) * time = 0.7 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_15_minutes_l393_39310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_powers_of_two_l393_39397

theorem sequence_powers_of_two (a : ℕ → ℕ) (h : ∀ n, a (n + 1) = a n + (a n % 10)) :
  (∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, ∃ m, a (f k) = 2^m) ↔ ¬ (5 ∣ a 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_powers_of_two_l393_39397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l393_39373

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

-- Theorem statement
theorem set_operations :
  (M ∪ N = Set.Iic 2) ∧
  (M ∩ N = Set.Icc (-2) 1) ∧
  (Set.compl N = Set.Ioi 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l393_39373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_largest_angle_l393_39314

theorem triangle_largest_angle (a b c : ℝ) 
  (h_ratio : (a, b, c) = (1 * x, 3 * x, 5 * x)) 
  (h_triangle : a + b + c = 180) 
  (x : ℝ) : 
  max a (max b c) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_largest_angle_l393_39314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_min_values_l393_39390

/-- The circle defined by the equation x^2 + y^2 - 4x - 4y + 7 = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

/-- The function we want to maximize and minimize -/
def f (x y : ℝ) : ℝ := (x+1)^2 + (y+2)^2

theorem circle_max_min_values :
  (∀ x y : ℝ, circle_eq x y → f x y ≤ 36) ∧
  (∀ x y : ℝ, circle_eq x y → f x y ≥ 16) ∧
  (∃ x y : ℝ, circle_eq x y ∧ f x y = 36) ∧
  (∃ x y : ℝ, circle_eq x y ∧ f x y = 16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_min_values_l393_39390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l393_39326

open Real

theorem f_value_in_third_quadrant (α : ℝ) : 
  (π < α ∧ α < 3*π/2) →  -- α is in the third quadrant
  cos (α - 3*π/2) = 1/5 →
  (λ x => (cos (π/2 + x) * cos (π - x)) / sin (π + x)) α = 2 * Real.sqrt 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l393_39326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_books_borrowed_l393_39395

theorem average_books_borrowed (total_students : ℕ) 
  (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 40)
  (h2 : zero_books = 2)
  (h3 : one_book = 12)
  (h4 : two_books = 14)
  (h5 : zero_books + one_book + two_books < total_students) :
  let remaining_students := total_students - (zero_books + one_book + two_books)
  let min_books := 0 * zero_books + 1 * one_book + 2 * two_books + 3 * remaining_students
  (min_books : ℚ) / total_students = 19 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_books_borrowed_l393_39395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_OA_OB_l393_39393

noncomputable def angle_between_vectors (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ :=
  let x1 := r1 * Real.cos θ1
  let y1 := r1 * Real.sin θ1
  let x2 := r2 * Real.cos θ2
  let y2 := r2 * Real.sin θ2
  let dot_product := x1 * x2 + y1 * y2
  let magnitude1 := Real.sqrt (x1^2 + y1^2)
  let magnitude2 := Real.sqrt (x2^2 + y2^2)
  Real.arccos (dot_product / (magnitude1 * magnitude2))

theorem angle_between_OA_OB :
  angle_between_vectors 2 (π/6) 6 (-π/6) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_OA_OB_l393_39393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_derivative_l393_39398

theorem indefinite_integral_derivative (x : ℝ) :
  (deriv (fun x => (1/9) * (15*x - 11) * Real.exp (3*x))) x = (5*x - 2) * Real.exp (3*x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_derivative_l393_39398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_increase_l393_39313

/-- Calculates the simple interest rate given principal, final amount, and time -/
noncomputable def calculate_rate (principal : ℝ) (final_amount : ℝ) (time : ℝ) : ℝ :=
  ((final_amount - principal) * 100) / (principal * time)

/-- Calculates the percentage increase between two rates -/
noncomputable def percentage_increase (original_rate : ℝ) (new_rate : ℝ) : ℝ :=
  ((new_rate - original_rate) / original_rate) * 100

theorem interest_rate_increase (principal : ℝ) (initial_final : ℝ) (new_final : ℝ) (time : ℝ) :
  principal = 900 →
  initial_final = 956 →
  new_final = 1064 →
  time = 3 →
  let original_rate := calculate_rate principal initial_final time
  let new_rate := calculate_rate principal new_final time
  ∃ ε > 0, |percentage_increase original_rate new_rate - 192.9| < ε := by
    sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_increase_l393_39313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l393_39355

/-- A sequence with general term a_n = n^2 - (6+2λ)n + 2014 -/
def a (n : ℕ) (lambda : ℝ) : ℝ := n^2 - (6+2*lambda)*n + 2014

/-- The theorem stating the range of λ given the conditions -/
theorem lambda_range (lambda : ℝ) :
  (∀ (n : ℕ), a 6 lambda ≤ a n lambda ∨ a 7 lambda ≤ a n lambda) →
  2.5 < lambda ∧ lambda < 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l393_39355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l393_39344

theorem trigonometric_identities (α β : ℝ) (h : α + β = Real.pi) :
  (Real.sin α = Real.sin β) ∧
  (Real.cos α = -Real.cos β) ∧
  (Real.tan α = -Real.tan β) ∧
  (Real.sin α ≠ -Real.sin β) ∧
  (Real.cos α ≠ Real.cos β) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l393_39344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l393_39383

open Real

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 4)

theorem omega_value (ω : ℝ) (h1 : ω > 0) :
  (∀ x ∈ Set.Ioo (-ω) ω, StrictMono (f ω)) →
  (∀ x : ℝ, f ω (ω + x) = f ω (ω - x)) →
  ω = sqrt π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l393_39383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_correct_l393_39316

/-- Represents the result of a stratified sampling --/
structure StratifiedSample where
  senior : Nat
  middle : Nat
  general : Nat
deriving Repr

/-- Calculates the stratified sample given the total employees, sample size, and staff counts --/
def calculateStratifiedSample (total : Nat) (sample : Nat) (senior : Nat) (middle : Nat) (general : Nat) : StratifiedSample :=
  { senior := (sample * senior) / total,
    middle := (sample * middle) / total,
    general := (sample * general) / total }

/-- Proves that the stratified sampling result is correct for the given company --/
theorem stratified_sampling_correct :
  let total := 150
  let sample := 30
  let senior := 15
  let middle := 45
  let general := 90
  let result := calculateStratifiedSample total sample senior middle general
  result.senior = 3 ∧ result.middle = 9 ∧ result.general = 18 := by
  sorry

#eval calculateStratifiedSample 150 30 15 45 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_correct_l393_39316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_count_ratio_l393_39329

/-- Represents the number of birds seen on a given day -/
def BirdCount := ℕ

/-- The bird count problem -/
theorem bird_count_ratio
  (monday tuesday wednesday : ℕ)
  (h1 : monday = 70)
  (h2 : wednesday = tuesday + 8)
  (h3 : monday + tuesday + wednesday = 148) :
  tuesday * 2 = monday :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_count_ratio_l393_39329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_avoiding_circle_l393_39361

/-- The shortest path from (0,0) to (15,20) avoiding a circle -/
theorem shortest_path_avoiding_circle :
  let start : ℝ × ℝ := (0, 0)
  let end_point : ℝ × ℝ := (15, 20)
  let circle_center : ℝ × ℝ := (9, 12)
  let circle_radius : ℝ := 6
  let circle := {p : ℝ × ℝ | (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2}
  ∃ (path : Set (ℝ × ℝ)), 
    (start ∈ path) ∧ 
    (end_point ∈ path) ∧ 
    (∀ p ∈ path, p ∉ circle) ∧
    (∀ other_path : Set (ℝ × ℝ), 
      (start ∈ other_path) → 
      (end_point ∈ other_path) → 
      (∀ p ∈ other_path, p ∉ circle) → 
      (Real.sqrt 21 * 6 + 2 * Real.pi) ≤ Metric.diam other_path) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_avoiding_circle_l393_39361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l393_39345

def sequence_a (n : ℕ) : ℝ := 2^n - 1

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - n

def sequence_b (n : ℕ) : ℝ := (2*n + 1) * sequence_a n + 2*n + 1

def sum_T (n : ℕ) : ℝ := (2*n - 1) * 2^(n+1) + 2

def sequence_c (n : ℕ) (lambda : ℝ) : ℝ := 3^n + lambda * (sequence_a n + 1)

theorem sequence_properties :
  (∀ n : ℕ, sum_S n = 2 * sequence_a n - n) →
  (∀ n : ℕ, n ≥ 1 → sequence_a (n+1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, sequence_a n = 2^n - 1) ∧
  (∀ n : ℕ, sum_T n = (2*n - 1) * 2^(n+1) + 2) ∧
  (∀ lambda : ℝ, (∀ n : ℕ, n ≥ 1 → sequence_c (n+1) lambda > sequence_c n lambda) → lambda > -3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l393_39345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l393_39317

open Real

-- Define the curves C₁ and C₂
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 * cos φ, 2 * sin φ)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 6 / sqrt (4 + 5 * sin θ ^ 2)
  (ρ * cos θ, ρ * sin θ)

-- Define the square vertices
noncomputable def A : ℝ × ℝ := C₁ (π / 6)
noncomputable def B : ℝ × ℝ := C₁ (2 * π / 3)
noncomputable def C : ℝ × ℝ := C₁ (7 * π / 6)
noncomputable def D : ℝ × ℝ := C₁ (5 * π / 3)

-- Define the distance squared function
def distanceSquared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2

-- Define the sum of squared distances
noncomputable def sumSquaredDistances (p : ℝ × ℝ) : ℝ :=
  distanceSquared p A + distanceSquared p B + distanceSquared p C + distanceSquared p D

-- Theorem statement
theorem max_sum_squared_distances :
  ∃ (M : ℝ), ∀ (θ : ℝ), sumSquaredDistances (C₂ θ) ≤ M ∧ 
  ∃ (θ₀ : ℝ), sumSquaredDistances (C₂ θ₀) = M ∧ M = 52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l393_39317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l393_39386

theorem rectangular_to_polar :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r * Real.cos θ = -1 ∧ r * Real.sin θ = -Real.sqrt 3 ∧
  r = 2 ∧ θ = 4 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l393_39386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_five_digit_number_tens_place_l393_39358

def is_odd (n : ℕ) : Prop := n % 2 = 1

def digits : List ℕ := [1, 2, 5, 6, 7]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  (∀ d : ℕ, d ∈ digits → (n.repr.count d.repr.head = 1)) ∧
  (∀ d : ℕ, d ∉ digits → (n.repr.count d.repr.head = 0))

theorem smallest_odd_five_digit_number_tens_place :
  ∃ n : ℕ,
    is_valid_number n ∧
    is_odd n ∧
    (∀ m : ℕ, is_valid_number m → is_odd m → n ≤ m) ∧
    ((n / 10) % 10 = 7) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_five_digit_number_tens_place_l393_39358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oslash_problem_l393_39309

-- Define the oslash operation
noncomputable def oslash (a b : ℝ) : ℝ := (Real.sqrt (2 * a + b)) ^ 3

-- State the theorem
theorem oslash_problem (y : ℝ) : oslash 9 y = 125 → y = 7 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oslash_problem_l393_39309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l393_39351

-- Define A_x^k as a function
def A (x k : ℕ) : ℕ := x.choose k

-- C_n^k is already defined in Mathlib as Nat.choose

theorem problem_solution :
  -- Part 1
  (∀ x : ℕ, (3 * A x 3 ≤ 2 * A (x + 1) 2 + 6 * A x 2) ↔ (x = 3 ∨ x = 4 ∨ x = 5)) ∧
  -- Part 2
  (∀ n : ℕ, 4 ≤ n ∧ n ≤ 5 → (Nat.choose n (5 - n) + Nat.choose (n + 1) (9 - n) = 5 ∨ 
                              Nat.choose n (5 - n) + Nat.choose (n + 1) (9 - n) = 16)) ∧
  -- Part 3
  (∀ m : ℕ, (1 / Nat.choose 5 m : ℚ) - (1 / Nat.choose 6 m : ℚ) = (7 / (10 * Nat.choose 7 m) : ℚ) → 
            Nat.choose 8 m = 28) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l393_39351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_range_l393_39392

-- Define the function f(x) as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 3) / Real.log (1/2)

-- State the theorem
theorem monotone_increasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y < 1 → f a x < f a y) →
  2 ≤ a ∧ a ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_range_l393_39392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trials_for_probability_l393_39306

/-- Given a probability p and a tolerance ε, find the smallest number of trials n 
    such that the probability of the sample mean being within ε of p exceeds 0.96 -/
theorem min_trials_for_probability (p ε : ℝ) (hp : p = 0.7) (hε : ε = 0.2) :
  let n : ℕ := 132
  let q : ℝ := 1 - p
  ∀ k : ℕ, k ≥ n → Real.exp (-2 * (k : ℝ) * ε^2) < 0.04 ∧
  ∀ m : ℕ, m < n → Real.exp (-2 * (m : ℝ) * ε^2) ≥ 0.04 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trials_for_probability_l393_39306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_YXW_l393_39364

open Real

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)
  (XY : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 5)
  (XZ : Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 7)
  (YZ : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 9)

-- Define point W on YZ
noncomputable def W (t : Triangle) : ℝ × ℝ :=
  (t.Y.1 + s * (t.Z.1 - t.Y.1), t.Y.2 + s * (t.Z.2 - t.Y.2))
where
  s : ℝ := sorry  -- The exact value of s is not given in the problem

-- Define the angle bisector property
def is_angle_bisector (t : Triangle) : Prop :=
  let XW := Real.sqrt ((t.X.1 - (W t).1)^2 + (t.X.2 - (W t).2)^2)
  let angle_YXW := arccos ((5^2 + XW^2 - ((W t).1 - t.Y.1)^2 - ((W t).2 - t.Y.2)^2) / (2 * 5 * XW))
  let angle_ZXW := arccos ((7^2 + XW^2 - ((W t).1 - t.Z.1)^2 - ((W t).2 - t.Z.2)^2) / (2 * 7 * XW))
  angle_YXW = angle_ZXW

-- State the theorem
theorem cos_angle_YXW (t : Triangle) (h : is_angle_bisector t) :
  let XW := Real.sqrt ((t.X.1 - (W t).1)^2 + (t.X.2 - (W t).2)^2)
  let cos_YXW := (5^2 + XW^2 - ((W t).1 - t.Y.1)^2 - ((W t).2 - t.Y.2)^2) / (2 * 5 * XW)
  cos_YXW = sqrt 0.45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_YXW_l393_39364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_f_and_inverse_l393_39347

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(x-2) + x/2

-- Define the domain of f
def domain : Set ℝ := Set.Icc 0 2

-- State the theorem
theorem max_value_of_sum_f_and_inverse :
  ∃ (y : ℝ), y = 4 ∧ 
  ∀ (x : ℝ), x ∈ domain → 
  f x + (Function.invFun f) x ≤ y :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_f_and_inverse_l393_39347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_equation_l393_39346

theorem no_solutions_equation (n : ℕ) (x y : ℕ) 
  (h1 : n ≥ 2) 
  (h2 : Nat.Coprime x (n + 1)) : 
  x^n + 1 ≠ y^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_equation_l393_39346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_negative_l393_39384

open Set

theorem solution_set_of_f_negative
  (f : ℝ → ℝ)
  (f_differentiable : Differentiable ℝ f)
  (h1 : ∀ x, deriv f x + 2 * f x > 0)
  (h2 : f (-1) = 0) :
  {x : ℝ | f x < 0} = Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_negative_l393_39384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_nine_fourths_l393_39337

/-- A parabola C with equation y² = 2px and a point A lying on it. -/
structure Parabola where
  p : ℝ
  A : ℝ × ℝ
  h1 : A.2^2 = 2 * p * A.1
  h2 : A.1 = 1
  h3 : A.2 = Real.sqrt 5

/-- The distance from point A to the directrix of parabola C. -/
noncomputable def distance_to_directrix (C : Parabola) : ℝ :=
  C.A.1 + C.p / 2

/-- Theorem stating that the distance from A to the directrix of C is 9/4. -/
theorem distance_is_nine_fourths (C : Parabola) :
  distance_to_directrix C = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_nine_fourths_l393_39337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_lift_work_l393_39372

/-- Work done to lift a satellite -/
noncomputable def work_done (m : ℝ) (g : ℝ) (R₃ : ℝ) (H : ℝ) : ℝ :=
  m * g * R₃^2 * (1/R₃ - 1/(R₃ + H))

/-- Theorem: The work done to lift a satellite -/
theorem satellite_lift_work 
  (m : ℝ) (g : ℝ) (R₃ : ℝ) (H : ℝ) 
  (hm : m > 0) (hg : g > 0) (hR : R₃ > 0) (hH : H > 0) :
  ∃ (W : ℝ), W = work_done m g R₃ H ∧ W > 0 :=
by
  sorry

#check satellite_lift_work

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_lift_work_l393_39372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_area_is_18pi_l393_39387

-- Define the shape of the room
structure Room where
  inner_radius : ℝ
  outer_radius : ℝ
  h_positive : 0 < inner_radius
  h_order : inner_radius < outer_radius

-- Define the property of the farthest distance
def farthest_distance (r : Room) : ℝ := r.outer_radius - r.inner_radius

-- Define the area of the room
noncomputable def room_area (r : Room) : ℝ := Real.pi * (r.outer_radius^2 - r.inner_radius^2) / 2

-- Theorem statement
theorem room_area_is_18pi :
  ∃ r : Room, farthest_distance r = 6 ∧ room_area r = 18 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_area_is_18pi_l393_39387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l393_39349

/-- Calculates the speed of a train in km/hr given its length in meters and time in seconds to cross a pole -/
noncomputable def trainSpeed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a train with length 200 meters crossing a pole in 24 seconds has a speed of 30 km/hr -/
theorem train_speed_theorem :
  trainSpeed 200 24 = 30 := by
  -- Unfold the definition of trainSpeed
  unfold trainSpeed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l393_39349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l393_39322

/-- The problem setup and solution --/
theorem speed_calculation (total_distance x_distance_1 y_distance_2 : ℝ)
  (h_total : total_distance = 300)
  (h_x_1 : x_distance_1 = 140)
  (h_y_2 : y_distance_2 = 180)
  : ∃ (speed_x speed_y : ℝ),
    speed_x / speed_y = x_distance_1 / (total_distance - x_distance_1) ∧
    speed_x / (speed_y + 1) = (total_distance - y_distance_2) / y_distance_2 ∧
    speed_y = 16/5 := by
  sorry

#check speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l393_39322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_car_mileage_l393_39343

-- Define the given conditions
noncomputable def miles_per_day : ℝ := 75
noncomputable def gas_price_per_gallon : ℝ := 3
def days : ℕ := 10
noncomputable def total_spent : ℝ := 45

-- Define the function to calculate miles per gallon
noncomputable def miles_per_gallon : ℝ :=
  (miles_per_day * (days : ℝ)) / (total_spent / gas_price_per_gallon)

-- Theorem to prove
theorem toms_car_mileage :
  miles_per_gallon = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_car_mileage_l393_39343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_inequality_condition_l393_39319

-- Define the statement
theorem cos_inequality_condition :
  (∀ α β : ℝ, α ≠ β → Real.cos α ≠ Real.cos β) ∧
  ¬(∀ α β : ℝ, Real.cos α ≠ Real.cos β → α ≠ β) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_inequality_condition_l393_39319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l393_39332

-- Define the parabola function
def f (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Theorem statement
theorem parabola_properties :
  (∃ a b c : ℝ, a > 0 ∧ (∀ x : ℝ, f x = a * x^2 + b * x + c)) ∧ 
  (∀ x : ℝ, f x = f (4 - x)) ∧
  (∀ x : ℝ, f 2 ≤ f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 2 → f x₁ > f x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l393_39332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_minimum_l393_39341

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (1, -Real.sqrt 3 * Real.sin (x / 2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, 2 * Real.sin (x / 2))

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + Real.sqrt 3

-- Define the set of integers
def Z : Set ℤ := Set.univ

-- Theorem for the intervals where f(x) is increasing
theorem f_increasing (k : ℤ) :
  ∀ x, x ∈ Set.Icc (2 * k * Real.pi - 5 * Real.pi / 6) (2 * k * Real.pi + Real.pi / 6) →
  Monotone f :=
sorry

-- Theorem for the minimum value of f(x) in [0, 2π/3]
theorem f_minimum :
  ∃ x₀ ∈ Set.Icc 0 (2 * Real.pi / 3), ∀ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x₀ ≤ f x ∧ f x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_minimum_l393_39341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_equivalence_l393_39356

-- Define the equivalence of inequalities
def inequalities_equivalent (f g : ℝ → Prop) : Prop :=
  ∀ x, f x ↔ g x

-- Define the six pairs of inequalities
def pair1 (x : ℝ) : Prop × Prop :=
  (x^2 + 5*x < 4, x^2 + 5*x + 3*x < 4 + 3*x)

noncomputable def pair2 (x : ℝ) : Prop × Prop :=
  (x^2 + 5*x < 4, x^2 + 5*x + 1/x < 4 + 1/x)

def pair3 (x : ℝ) : Prop × Prop :=
  (x ≥ 3, x*(x+5)^2 ≥ 3*(x+5)^2)

def pair4 (x : ℝ) : Prop × Prop :=
  (x ≥ 3, x*(x-5)^2 ≥ 3*(x-5)^2)

noncomputable def pair5 (x : ℝ) : Prop × Prop :=
  (x + 3 > 0, (x+3)*(x+1)/(x+1) > 0)

noncomputable def pair6 (x : ℝ) : Prop × Prop :=
  (x - 3 > 0, (x+2)*(x-3)/(x+2) > 0)

-- State the theorem
theorem inequalities_equivalence :
  inequalities_equivalent (λ x ↦ (pair1 x).1) (λ x ↦ (pair1 x).2) ∧
  ¬(inequalities_equivalent (λ x ↦ (pair2 x).1) (λ x ↦ (pair2 x).2)) ∧
  inequalities_equivalent (λ x ↦ (pair3 x).1) (λ x ↦ (pair3 x).2) ∧
  inequalities_equivalent (λ x ↦ (pair4 x).1) (λ x ↦ (pair4 x).2) ∧
  ¬(inequalities_equivalent (λ x ↦ (pair5 x).1) (λ x ↦ (pair5 x).2)) ∧
  inequalities_equivalent (λ x ↦ (pair6 x).1) (λ x ↦ (pair6 x).2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_equivalence_l393_39356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l393_39338

open Set

def A : Set ℝ := {x : ℝ | |x - 1| > 3}

theorem complement_of_A : (Aᶜ : Set ℝ) = Icc (-2) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l393_39338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l393_39367

def has_two_distinct_positive_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + 2*m*x₁ + 1 = 0 ∧ x₂^2 + 2*m*x₂ + 1 = 0

def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x, x^2 + 2*(m-2)*x - 3*m + 10 ≠ 0

def exactly_one_condition_true (m : ℝ) : Prop :=
  (has_two_distinct_positive_roots m ∧ ¬has_no_real_roots m) ∨
  (¬has_two_distinct_positive_roots m ∧ has_no_real_roots m)

theorem range_of_m :
  {m : ℝ | exactly_one_condition_true m} = Set.Iic (-2) ∪ Set.Icc (-1) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l393_39367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l393_39366

-- Define the interval [-2, 2]
def I : Set ℝ := Set.Icc (-2) 2

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 ∧ x ≤ 2 then 2^x - 1
  else if x < 0 ∧ x ≥ -2 then -(2^(-x) - 1)
  else 0  -- This covers the case x = 0

-- Define the function g
def g (m : ℝ) : ℝ → ℝ := fun x => x^2 - 2*x + m

-- State the properties of f
axiom f_odd : ∀ x ∈ I, f (-x) = -f x
axiom f_def_pos : ∀ x ∈ Set.Ioo 0 2, f x = 2^x - 1

-- State the main condition
axiom condition : ∀ m : ℝ, (∀ x₁ ∈ I, ∃ x₂ ∈ I, f x₁ ≤ g m x₂) → m ≥ -5

-- Theorem statement
theorem min_m_value : 
  ∃ m : ℝ, (∀ x₁ ∈ I, ∃ x₂ ∈ I, f x₁ ≤ g m x₂) ∧ 
            (∀ m' < m, ∃ x₁ ∈ I, ∀ x₂ ∈ I, f x₁ > g m' x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l393_39366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthic_triangle_convergence_l393_39318

/-- A type representing a triangle -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthic transformation that maps a triangle to its orthic triangle -/
noncomputable def h (t : Triangle) : Triangle :=
  sorry

/-- Predicate to check if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop :=
  sorry

/-- Function to calculate the perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  sorry

/-- The main theorem -/
theorem orthic_triangle_convergence (ABC : Triangle) :
  ∃ n : ℕ, (isRightTriangle (Nat.rec ABC (fun _ => h) n)) ∨
    (perimeter (Nat.rec ABC (fun _ => h) n) < perimeter ABC) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthic_triangle_convergence_l393_39318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l393_39354

/-- A power function of the form y = (m^2 - 5m - 5)x^(2m+1) -/
noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ := (m^2 - 5*m - 5) * (x^(2*m + 1))

/-- The derivative of the power function with respect to x -/
noncomputable def power_function_derivative (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - 5*m - 5) * (2*m + 1) * (x^(2*m))

theorem decreasing_power_function (m : ℝ) :
  (∀ x > 0, (power_function_derivative m x < 0)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l393_39354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l393_39311

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - 4*x else ((-x)^2 - 4*(-x))

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) : 
  (∀ y : ℝ, f y = f (-y)) →  -- f is even
  (f (x + 2) < 5 ↔ -7 < x ∧ x < 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l393_39311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_hexagon_ratio_l393_39378

/-- The maximum ratio of the area of a square inscribed in a regular hexagon
    with coinciding centers to the area of the hexagon. -/
theorem max_square_hexagon_ratio :
  let r : ℝ := 1  -- radius of the circle circumscribing the hexagon
  let hexagon_area : ℝ := 3 * Real.sqrt 3 / 2 * r^2
  let max_square_side : ℝ := Real.sqrt 3 * r / (Real.sqrt 2 * Real.cos (15 * π / 180))
  let max_square_area : ℝ := max_square_side^2
  max_square_area / hexagon_area = 1 / (Real.sqrt 3 * (Real.cos (15 * π / 180))^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_hexagon_ratio_l393_39378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_areas_sum_l393_39360

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def is_on_parabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- Calculates the area of a triangle with vertices at origin, focus, and a point on parabola -/
noncomputable def triangle_area (p : Point) : ℝ :=
  (1/2) * abs p.y

/-- States that three points satisfy the centroid condition -/
def centroid_condition (a b c : Point) : Prop :=
  (a.x - 1) + (b.x - 1) + (c.x - 1) = 0 ∧ a.y + b.y + c.y = 0

theorem parabola_triangle_areas_sum (a b c : Point) :
  is_on_parabola a ∧ is_on_parabola b ∧ is_on_parabola c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  centroid_condition a b c →
  (triangle_area a)^2 + (triangle_area b)^2 + (triangle_area c)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_areas_sum_l393_39360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_distances_l393_39377

/-- Represents the scale of the map in miles per inch -/
noncomputable def map_scale : ℝ := 24 / 1.5

/-- Conversion factor from inches to centimeters -/
noncomputable def inches_to_cm : ℝ := 2.54

/-- Length of Route X on the map in centimeters -/
noncomputable def route_x_cm : ℝ := 44

/-- Length of Route Y on the map in centimeters -/
noncomputable def route_y_cm : ℝ := 62

/-- Distance between two intermediate cities on Route Y in miles -/
noncomputable def intermediate_distance_miles : ℝ := 40

/-- Length between two intermediate cities on Route Y on the map in inches -/
noncomputable def intermediate_distance_inches : ℝ := 2.5

/-- Theorem stating the actual distances for Route X and Route Y -/
theorem route_distances :
  let route_x_miles := (route_x_cm / inches_to_cm) * map_scale
  let route_y_miles := (route_y_cm / inches_to_cm) * map_scale
  (abs (route_x_miles - 277.12) < 0.01) ∧ (abs (route_y_miles - 390.56) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_distances_l393_39377
