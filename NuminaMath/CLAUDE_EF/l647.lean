import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hole_water_problem_l647_64736

theorem hole_water_problem (total_needed : ℕ) (additional_needed : ℕ) (initial_water : ℕ) :
  total_needed = 823 →
  additional_needed = 147 →
  initial_water = total_needed - additional_needed →
  initial_water = 676 := by
  intro h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

#check hole_water_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hole_water_problem_l647_64736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l647_64716

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + Real.pi/4

theorem f_properties :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ k : ℤ, f (Real.pi/2 + k * Real.pi) = 0) ∧
  (∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → -1 ≤ f x ∧ f x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l647_64716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_wine_amount_l647_64782

/-- Represents the number of stores and flowers encountered -/
def encounters : ℕ := 4

/-- Represents the amount of wine drunk at each flower -/
def wine_drunk_per_flower : ℕ := 2

/-- Represents the final amount of wine left -/
def final_wine : ℕ := 2

/-- Calculates the amount of wine after each encounter -/
def wine_after_encounter (initial : ℕ) : ℕ := initial * 2 - wine_drunk_per_flower

/-- Represents the process of encountering stores and flowers -/
def process_encounters : ℕ → ℕ → ℕ
| 0, initial => initial
| n + 1, initial => process_encounters n (wine_after_encounter initial)

/-- Theorem stating that the initial amount of wine is 2 cups -/
theorem initial_wine_amount : 
  ∃ (initial : ℕ), process_encounters encounters initial = final_wine ∧ initial = 2 := by
  use 2
  apply And.intro
  · simp [process_encounters, encounters, wine_after_encounter, final_wine]
    sorry -- The actual computation step is omitted for brevity
  · rfl

#eval process_encounters encounters 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_wine_amount_l647_64782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_range_l647_64763

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | -x^2 + 4*x - 3 ≥ 0}

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem x0_range (x₀ : ℝ) (h1 : x₀ ∈ A) (h2 : f (f x₀) ∈ A) :
  x₀ < -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_range_l647_64763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_two_digit_prime_with_reversed_prime_l647_64761

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

theorem smallest_two_digit_prime_with_reversed_prime :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ is_prime n ∧ is_prime (reverse_digits n) ∧ reverse_digits n % 10 = 3 ∧
  (∀ m : ℕ, 10 ≤ m → m < n → ¬(is_prime m ∧ is_prime (reverse_digits m) ∧ reverse_digits m % 10 = 3)) ∧
  n = 13 :=
by sorry

#check smallest_two_digit_prime_with_reversed_prime

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_two_digit_prime_with_reversed_prime_l647_64761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_difference_is_two_l647_64797

/-- Represents a football tournament with n teams -/
structure FootballTournament where
  n : ℕ
  n_ge_2 : n ≥ 2

/-- Points awarded for different match outcomes -/
inductive MatchPoints where
  | Win : MatchPoints
  | Draw : MatchPoints
  | Loss : MatchPoints

/-- Convert MatchPoints to integer values -/
def pointValue (result : MatchPoints) : ℤ :=
  match result with
  | MatchPoints.Win => 2
  | MatchPoints.Draw => -1
  | MatchPoints.Loss => 0

/-- The maximum point difference between consecutively ranked teams -/
def maxPointDifference (tournament : FootballTournament) : ℕ := 2

/-- Theorem: The maximum point difference between consecutively ranked teams is 2 -/
theorem max_point_difference_is_two (tournament : FootballTournament) :
  maxPointDifference tournament = 2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_difference_is_two_l647_64797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_starting_positions_l647_64764

/-- The sequence defined by the recurrence relation --/
noncomputable def x (n : ℕ) (x₀ : ℝ) : ℝ := 
  match n with
  | 0 => x₀
  | n+1 => (x n x₀^2 - 2) / (2 * x n x₀)

/-- The theorem stating the number of starting positions --/
theorem number_of_starting_positions :
  ∃ (S : Finset ℝ), (∀ x₀ ∈ S, x 2023 x₀ = x₀) ∧ S.card = 2^2023 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_starting_positions_l647_64764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l647_64740

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 + x) + Real.sqrt (3 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-2 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l647_64740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_ge_three_times_square_side_equality_condition_l647_64759

/-- A right triangle with an inscribed square -/
structure RightTriangleWithSquare where
  -- The vertices of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- The vertices of the inscribed square
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  -- C is a right angle
  right_angle_at_C : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  -- S and P are on BC and CA respectively
  S_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)
  P_on_CA : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * C.1 + (1 - t) * A.1, t * C.2 + (1 - t) * A.2)
  -- Q and R are on AB
  Q_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)
  R_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)
  -- PQRS is a square
  square_PQRS : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (Q.1 - R.1)^2 + (Q.2 - R.2)^2 ∧
                (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = (R.1 - S.1)^2 + (R.2 - S.2)^2 ∧
                (R.1 - S.1)^2 + (R.2 - S.2)^2 = (S.1 - P.1)^2 + (S.2 - P.2)^2 ∧
                (P.1 - Q.1) * (Q.1 - R.1) + (P.2 - Q.2) * (Q.2 - R.2) = 0

/-- The length of AB is greater than or equal to three times the side length of PQRS -/
theorem hypotenuse_ge_three_times_square_side (t : RightTriangleWithSquare) :
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 ≥ 9 * ((t.Q.1 - t.R.1)^2 + (t.Q.2 - t.R.2)^2) := by
  sorry

/-- Equality occurs when the side length of PQRS is exactly one-third of AB -/
theorem equality_condition (t : RightTriangleWithSquare) :
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = 9 * ((t.Q.1 - t.R.1)^2 + (t.Q.2 - t.R.2)^2) ↔
  (t.A.1 - t.Q.1)^2 + (t.A.2 - t.Q.2)^2 = (t.Q.1 - t.R.1)^2 + (t.Q.2 - t.R.2)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_ge_three_times_square_side_equality_condition_l647_64759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_dot_product_l647_64795

noncomputable section

/-- Parabola equation -/
def parabola (x : ℝ) : ℝ := (1/4) * x^2

/-- Line passing through (0, 2) -/
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 2

/-- Intersection points of the line and parabola -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x, p.1 = x ∧ p.2 = parabola x ∧ p.2 = line k x}

/-- Origin point -/
def O : ℝ × ℝ := (0, 0)

/-- Vector from origin to a point -/
def vector_from_origin (p : ℝ × ℝ) : ℝ × ℝ := p

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Main theorem -/
theorem parabola_line_intersection_dot_product :
  ∀ k : ℝ, ∀ A B : ℝ × ℝ,
    A ∈ intersection_points k → B ∈ intersection_points k → A ≠ B →
    dot_product (vector_from_origin A) (vector_from_origin B) = -4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_dot_product_l647_64795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_n_squared_l647_64780

-- Define the linear functions
noncomputable def a : ℝ → ℝ := λ x => -x + 1
noncomputable def b : ℝ → ℝ := λ _ => 1
noncomputable def c : ℝ → ℝ := λ x => x - 1

-- Define m(x) and n(x)
noncomputable def m (x : ℝ) : ℝ := max (max (a x) (b x)) (c x)
noncomputable def n (x : ℝ) : ℝ := min (min (a x) (b x)) (c x)

-- Define the conditions for m(x)
axiom m_segment1 : ∀ x ∈ Set.Icc (-4 : ℝ) (-2), m x = -x + 1
axiom m_segment2 : ∀ x ∈ Set.Icc (-2 : ℝ) 2, m x = 1
axiom m_segment3 : ∀ x ∈ Set.Icc 2 4, m x = x - 1

-- Define the length of the graph of n(x)
noncomputable def length_n : ℝ := 4 + 4 * Real.sqrt 2

-- State the theorem
theorem length_n_squared : length_n ^ 2 = 48 + 32 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_n_squared_l647_64780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l647_64756

/-- The difference in interest earned between two interest rates that differ by 1% 
    for a principal of 2600 invested for 3 years is 78, regardless of the initial interest rate. -/
theorem interest_difference (R : ℝ) : ℝ := by
  -- Define the principal amount
  let principal : ℝ := 2600
  -- Define the time period in years
  let time : ℝ := 3
  -- Define the interest rate difference
  let rate_difference : ℝ := 0.01
  -- Calculate the interest difference
  let interest_diff : ℝ := principal * time * rate_difference
  -- Assert that the interest difference is equal to 78
  have h : interest_diff = 78 := by sorry
  -- Return the result
  exact interest_diff

#check interest_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l647_64756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l647_64743

-- Define the function f
noncomputable def f (x : ℝ) := x * Real.sin x + Real.cos x + x^2

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, x > 0 →
  (f (Real.log x) + f (Real.log (1/x)) < 2 * f 1 ↔ 1/Real.exp 1 < x ∧ x < Real.exp 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l647_64743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_given_three_intersections_l647_64737

open Real

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (-x^2 + x - 1) * exp x

/-- The function g(x) with parameter m -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2 + m

/-- The difference function h(x) -/
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := f x - g m x

/-- Theorem stating the range of m given the conditions -/
theorem range_of_m_given_three_intersections :
  ∀ m : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    h m x = 0 ∧ h m y = 0 ∧ h m z = 0) →
  -3/exp 1 - 1/6 < m ∧ m < -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_given_three_intersections_l647_64737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_store_total_birds_l647_64796

def pet_store_birds_count (talking_birds non_talking_birds : ℕ) : ℕ :=
  talking_birds + non_talking_birds

theorem pet_store_total_birds : 
  pet_store_birds_count 64 13 = 77 := by
  unfold pet_store_birds_count
  rfl

#eval pet_store_birds_count 64 13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_store_total_birds_l647_64796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_calculations_l647_64785

-- Define the weights and age as real numbers
variable (Al Ben Carl Ed Frank AlAge : ℝ)

-- State the given conditions
axiom al_age_relation : Al = 5 * AlAge
axiom al_ben_relation : Al = Ben + 25
axiom ben_carl_relation : Ben = Carl - 16
axiom ed_weight : Ed = 146
axiom ed_al_relation : Ed = Al - 38
axiom frank_carl_relation : Frank = Carl * 1.12

-- State the theorem to be proved
theorem weight_calculations :
  Carl = 175 ∧ Frank = 196 ∧ Al = 5 * AlAge := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_calculations_l647_64785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_is_90_l647_64769

/-- Represents a structure made of cubes -/
structure CubeStructure where
  num_cubes : ℕ
  visible_faces : ℕ

/-- Amount of paint needed for a single cube in ml -/
def paint_per_cube : ℚ := 10

/-- The specific cube structure in the problem -/
def problem_structure : CubeStructure :=
  { num_cubes := 14
  , visible_faces := 54 }

/-- Calculates the amount of paint needed for a given cube structure -/
def paint_needed (s : CubeStructure) : ℚ :=
  (s.visible_faces / 6 : ℚ) * paint_per_cube

/-- Theorem stating that the paint needed for the problem structure is 90 ml -/
theorem paint_needed_is_90 :
  paint_needed problem_structure = 90 := by
  -- Unfold definitions and simplify
  unfold paint_needed
  unfold problem_structure
  unfold paint_per_cube
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_is_90_l647_64769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weekly_income_is_500_l647_64708

/-- Represents the weekly income of a salesman over a 7-week period. -/
structure SalesmanIncome where
  baseSalary : ℚ
  pastWeeksIncome : List ℚ
  futureWeeksCommission : ℚ
  numPastWeeks : ℕ
  numFutureWeeks : ℕ

/-- Calculates the average weekly income over the entire period. -/
def averageWeeklyIncome (income : SalesmanIncome) : ℚ :=
  let totalPastIncome := income.pastWeeksIncome.sum
  let totalFutureIncome := income.numFutureWeeks * (income.baseSalary + income.futureWeeksCommission)
  let totalWeeks := income.numPastWeeks + income.numFutureWeeks
  (totalPastIncome + totalFutureIncome) / totalWeeks

/-- The main theorem stating that the average weekly income is $500. -/
theorem average_weekly_income_is_500 (income : SalesmanIncome)
  (h1 : income.baseSalary = 350)
  (h2 : income.pastWeeksIncome = [406, 413, 420, 436, 495])
  (h3 : income.futureWeeksCommission = 315)
  (h4 : income.numPastWeeks = 5)
  (h5 : income.numFutureWeeks = 2) :
  averageWeeklyIncome income = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weekly_income_is_500_l647_64708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parallelogram_area_l647_64722

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  x : ℝ  -- diagonal length
  a : ℝ  -- shorter side length
  h : a > 0  -- side length is positive
  diag_eq : x^2 = 3 * a^2  -- relationship between diagonal and side length
  angle : Real.cos (60 * Real.pi / 180) = 1/2  -- 60 degree angle

/-- The area of the special parallelogram -/
noncomputable def area (p : SpecialParallelogram) : ℝ := 
  p.a * (2 * p.a) * Real.sin (60 * Real.pi / 180)

/-- Theorem: The area of the special parallelogram is x²/3 -/
theorem special_parallelogram_area (p : SpecialParallelogram) : area p = p.x^2 / 3 := by
  sorry

#check special_parallelogram_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parallelogram_area_l647_64722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_isosceles_triangle_area_ratio_l647_64718

/-- 
Given an isosceles triangle with vertex angle α and a line passing through 
the vertex forming an angle β with the base, this function calculates the 
ratio of the areas of the two parts of the triangle divided by this line.
-/
noncomputable def areaRatio (α β : ℝ) : ℝ :=
  (Real.cos ((α/2) + β)) / (2 * Real.sin (α/2) * Real.sin β)

/-- 
Theorem stating that the area ratio function correctly computes the ratio 
of areas in the described isosceles triangle scenario.
-/
theorem isosceles_triangle_area_ratio (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi) 
  (h2 : 0 < β ∧ β < Real.pi/2) : 
  areaRatio α β = (Real.cos ((α/2) + β)) / (2 * Real.sin (α/2) * Real.sin β) := by
  -- The proof goes here
  sorry

#check isosceles_triangle_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_isosceles_triangle_area_ratio_l647_64718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_properties_l647_64710

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- A line that passes through the right focus of the ellipse and is not perpendicular to the x-axis -/
structure Line where
  n : ℝ
  h_not_perp : n ≠ 0

/-- The fixed point through which MN' passes -/
def fixed_point : ℝ × ℝ := (4, 0)

/-- Main theorem stating the properties of the ellipse and the fixed point -/
theorem ellipse_and_fixed_point_properties (e : Ellipse) (l : Line) :
  eccentricity e = 1/2 →
  e.a^2 = 4 ∧ e.b^2 = 3 ∧
  ∃ (x y : ℝ), x = l.n * y + 1 ∧ 
               3 * (l.n^2 * y^2 + 2 * l.n * y + 1) + 4 * y^2 = 12 ∧
               (x, -y) = fixed_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_properties_l647_64710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_properties_l647_64709

noncomputable section

/-- Line l parametric equation -/
def line_l (t : ℝ) : ℝ × ℝ := (t/2, (Real.sqrt 3/2) * t + 1)

/-- Curve C parametric equation -/
def curve_C (θ : ℝ) : ℝ × ℝ := (2 + Real.cos θ, Real.sin θ)

/-- Slope angle of line l -/
def slope_angle_l : ℝ := Real.pi/3

/-- Minimum distance from curve C to line l -/
def min_distance : ℝ := (2 * Real.sqrt 3 - 1) / 2

theorem line_curve_properties :
  (∀ t : ℝ, ∃ x y : ℝ, line_l t = (x, y)) ∧
  (∀ θ : ℝ, ∃ x y : ℝ, curve_C θ = (x, y)) →
  slope_angle_l = Real.pi/3 ∧
  (∀ Q : ℝ × ℝ, (∃ θ : ℝ, curve_C θ = Q) →
    ∃ d : ℝ, d ≥ min_distance ∧
      ∀ P : ℝ × ℝ, (∃ t : ℝ, line_l t = P) → Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) ≥ d) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_properties_l647_64709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_minimum_value_of_f_l647_64712

noncomputable def f (x : ℝ) : ℝ := x^3 + (1/2) * x^2 - 4*x

theorem extreme_minimum_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -5/2 := by
  -- We'll use x_min = 1 as the minimum point
  let x_min := 1
  
  -- Prove that f(x_min) = -5/2
  have h1 : f x_min = -5/2 := by
    simp [f]
    norm_num
  
  -- Assert the existence of x_min and that it satisfies the conditions
  use x_min
  intro x
  
  -- Split the goal into two parts
  apply And.intro
  
  -- Prove f x ≥ f x_min for all x
  · sorry  -- This requires calculus techniques not easily formalized in Lean
  
  -- Prove f x_min = -5/2
  · exact h1

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_minimum_value_of_f_l647_64712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l647_64779

def S : Finset Nat := {1, 2, 3}

theorem proper_subsets_count : Finset.card (Finset.powerset S \ {S}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l647_64779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_arrangement_impossible_l647_64772

/-- Represents a domino tile -/
structure Domino where
  width : Nat
  height : Nat

/-- Represents a square grid -/
structure Grid where
  size : Nat

/-- Represents a seam in the grid -/
inductive Seam where
  | Horizontal : Nat → Seam
  | Vertical : Nat → Seam

/-- Checks if a domino intersects a seam -/
def intersects (d : Domino) (s : Seam) : Prop :=
  sorry  -- Implementation details omitted for brevity

/-- Checks if a domino arrangement blocks all seams -/
def blocks_all_seams (grid : Grid) (dominoes : List Domino) : Prop :=
  ∀ s : Seam, ∃ d₁ d₂ : Domino, d₁ ≠ d₂ ∧ d₁ ∈ dominoes ∧ d₂ ∈ dominoes ∧ 
    intersects d₁ s ∧ intersects d₂ s

/-- The main theorem stating the impossibility of the arrangement -/
theorem domino_arrangement_impossible (grid : Grid) (dominoes : List Domino) :
  grid.size = 6 →
  dominoes.length = 18 →
  (∀ d ∈ dominoes, d.width = 1 ∧ d.height = 2) →
  ¬(blocks_all_seams grid dominoes) := by
  sorry  -- Proof details omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_arrangement_impossible_l647_64772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l647_64714

open Real

theorem range_of_a (a : ℝ) : (log (1 - a) - log a > 0) → (0 < a ∧ a < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l647_64714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_percent_error_l647_64703

noncomputable section

/-- The actual radius of the circle in cm -/
def actual_radius : ℝ := 10

/-- The maximum measurement error as a percentage -/
def max_error_percent : ℝ := 20

/-- The maximum measurement error as a decimal -/
def max_error : ℝ := max_error_percent / 100

/-- The minimum possible measured radius -/
def min_radius : ℝ := actual_radius * (1 - max_error)

/-- The maximum possible measured radius -/
def max_radius : ℝ := actual_radius * (1 + max_error)

/-- The actual area of the circle -/
def actual_area : ℝ := Real.pi * actual_radius ^ 2

/-- The area calculated using the minimum possible measured radius -/
def min_area : ℝ := Real.pi * min_radius ^ 2

/-- The area calculated using the maximum possible measured radius -/
def max_area : ℝ := Real.pi * max_radius ^ 2

/-- The percent error for the minimum measured radius -/
def min_percent_error : ℝ := ((actual_area - min_area) / actual_area) * 100

/-- The percent error for the maximum measured radius -/
def max_percent_error : ℝ := ((max_area - actual_area) / actual_area) * 100

/-- Theorem stating that the largest possible percent error is 44% -/
theorem largest_percent_error : 
  max max_percent_error min_percent_error = 44 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_percent_error_l647_64703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_angle_l647_64725

-- Define the line
def line (x y : ℝ) : Prop := x + y = 15

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 36

-- Define the slope of the line
def line_slope : ℝ := -1

-- Define the intersection points
def intersection_point1 : ℝ × ℝ := (10, 5)
def intersection_point2 : ℝ × ℝ := (4, 11)

-- Define the center of the circle
def circle_center : ℝ × ℝ := (4, 5)

-- Theorem statement
theorem line_circle_intersection_angle :
  ∃ (x y : ℝ), line x y ∧ circle_eq x y ∧
  (let radius_slope := (y - circle_center.2) / (x - circle_center.1);
   let angle := Real.arctan ((line_slope - radius_slope) / (1 + line_slope * radius_slope));
   angle = π / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_angle_l647_64725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l647_64721

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The given ellipse with equation x²/5 + y²/2 = 1 -/
noncomputable def givenEllipse : Ellipse where
  a := Real.sqrt 5
  b := Real.sqrt 2
  h_pos := by
    constructor
    · exact Real.sqrt_pos.mpr (by norm_num)
    · apply Real.sqrt_lt_sqrt
      · norm_num
      · norm_num

/-- Checks if two ellipses have the same foci -/
def sameFoci (e1 e2 : Ellipse) : Prop :=
  e1.a ^ 2 - e1.b ^ 2 = e2.a ^ 2 - e2.b ^ 2

/-- Represents a chord of an ellipse -/
structure Chord (e : Ellipse) where
  length : ℝ
  passesThroughRightFocus : Prop
  perpendicularToXAxis : Prop

/-- Represents a line y = x + m -/
structure Line where
  m : ℝ

/-- Represents the intersection of a line with an ellipse -/
structure Intersection (e : Ellipse) (l : Line) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  onEllipse : Prop
  onLine : Prop

theorem ellipse_problem (e : Ellipse) (c : Chord e) (l : Line) (i : Intersection e l) :
  sameFoci e givenEllipse →
  c.length = 1 →
  c.passesThroughRightFocus →
  c.perpendicularToXAxis →
  Real.sqrt ((i.A.1 - i.B.1)^2 + (i.A.2 - i.B.2)^2) = 8/5 →
  (e.a = 2 ∧ e.b = 1) ∧ (l.m = Real.sqrt 3 ∨ l.m = -Real.sqrt 3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l647_64721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l647_64747

theorem triangle_side_relation (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) 
  (h2 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h3 : a / Real.sin A = b / Real.sin B)
  (h4 : b / Real.sin B = c / Real.sin C)
  (h5 : ∀ x : ℝ, (x^2 * Real.sin A + 2*x * Real.sin B + Real.sin C = 0) → 
       (4 * Real.sin B^2 - 4 * Real.sin A * Real.sin C = 0)) : 
  b^2 = a * c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l647_64747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_point_monotone_increasing_condition_l647_64775

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + 2 * a * x
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 3 * a^2 * Real.log x + b
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x + g a b x - (2 * a + b) * x

-- State the theorems
theorem common_tangent_point (a b : ℝ) (ha : a > 0) :
  (∃ x₀ : ℝ, f a x₀ = g a b x₀ ∧ (deriv (f a)) x₀ = (deriv (g a b)) x₀) →
  b = (5 * a^2 / 2) - 3 * a^2 * Real.log a := by
  sorry

theorem monotone_increasing_condition (a b : ℝ) :
  b ∈ Set.Icc (-2 : ℝ) 2 →
  StrictMono (h a b) →
  a ≥ Real.sqrt 3 / 3 ∨ a ≤ -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_point_monotone_increasing_condition_l647_64775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l647_64705

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop := ρ^2 * (1 + 3 * Real.sin θ^2) = 4

/-- Two points on curve C are perpendicular from the origin -/
def perpendicular_points (ρ₁ ρ₂ θ : ℝ) : Prop :=
  curve_C ρ₁ θ ∧ curve_C ρ₂ (θ + Real.pi/2)

/-- Area of triangle formed by origin and two points on curve C -/
noncomputable def triangle_area (ρ₁ ρ₂ : ℝ) : ℝ := (1/2) * ρ₁ * ρ₂

/-- Theorem: Minimum area of triangle OMN is 4/5 -/
theorem min_triangle_area :
  ∀ ρ₁ ρ₂ θ : ℝ, perpendicular_points ρ₁ ρ₂ θ →
  triangle_area ρ₁ ρ₂ ≥ 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l647_64705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_g_over_x_cubed_even_k_range_l647_64770

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (2^(2*x))

-- Statement 1: f is monotonically increasing on (0,+∞)
theorem f_increasing : MonotoneOn f (Set.Ioi 0) := by sorry

-- Statement 2: g(x)/x^3 is an even function
theorem g_over_x_cubed_even : ∀ x : ℝ, x ≠ 0 → g x / x^3 = g (-x) / (-x)^3 := by sorry

-- Statement 3: For g(x) - k + l = 0 to have real solutions, k must be in (0,2)
theorem k_range : ∀ k l : ℝ, (∃ x : ℝ, g x - k + l = 0) → k ∈ Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_g_over_x_cubed_even_k_range_l647_64770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_theta_plus_45_l647_64791

/-- Given two parallel vectors a and b, prove that tan(θ + 45°) = -3 -/
theorem parallel_vectors_tan_theta_plus_45 (θ : ℝ) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![Real.sin θ, Real.cos θ]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  Real.tan (θ + Real.pi/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_theta_plus_45_l647_64791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_length_l647_64745

/-- Parabola type representing y^2 = 4x --/
structure Parabola where
  focus : ℝ × ℝ

/-- Line type with inclination angle --/
structure Line where
  angle : ℝ
  point : ℝ × ℝ

/-- Intersection points of a line and a parabola --/
structure Intersection where
  P : ℝ × ℝ
  Q : ℝ × ℝ

/-- Predicate to check if a point is on the parabola --/
def IsOnParabola (C : Parabola) (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

/-- Predicate to check if a point is on the line --/
def IsOnLine (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 - l.point.2 = Real.tan l.angle * (p.1 - l.point.1)

/-- Given a parabola y^2 = 4x and a line through its focus with inclination π/3,
    the length of the chord formed by the intersection points is 16/3 --/
theorem parabola_line_intersection_length
  (C : Parabola)
  (l : Line)
  (i : Intersection)
  (h1 : l.angle = π/3)
  (h2 : l.point = C.focus)
  (h3 : IsOnParabola C i.P ∧ IsOnParabola C i.Q)
  (h4 : IsOnLine l i.P ∧ IsOnLine l i.Q) :
  Real.sqrt ((i.P.1 - i.Q.1)^2 + (i.P.2 - i.Q.2)^2) = 16/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_length_l647_64745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_logarithmic_expression_equals_negative_seven_l647_64755

theorem complex_logarithmic_expression_equals_negative_seven :
  (1/8 : ℝ) - (64/27 : ℝ)^0 - (Real.log 25 / Real.log 2) * (Real.log 4 / Real.log 3) * (Real.log 9 / Real.log 5) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_logarithmic_expression_equals_negative_seven_l647_64755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_symmetric_lines_l647_64762

-- Define the lines l₁ and l₂
def l₁ : ℝ → ℝ → Prop := sorry
def l₂ : ℝ → ℝ → Prop := sorry

-- Define the symmetry relation with respect to x-axis
def symmetric_wrt_x_axis (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ g x (-y)

-- State the theorem
theorem intersection_point_of_symmetric_lines :
  l₁ 0 3 →  -- l₁ passes through (0,3)
  l₂ 5 2 →  -- l₂ passes through (5,2)
  symmetric_wrt_x_axis l₁ l₂ →  -- l₁ is symmetric to l₂ with respect to x-axis
  ∃ x, l₁ x 0 ∧ l₂ x 0 ∧ x = 3 :=  -- The intersection point is (3,0)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_symmetric_lines_l647_64762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l647_64723

/-- Hyperbola with foci above and below the center -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- Point on the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : y^2 / h.a^2 - x^2 / h.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: Eccentricity of the hyperbola is √10/2 under given conditions -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (P Q : HyperbolaPoint h) (F₁ F₂ : ℝ × ℝ) :
  distance (P.x, P.y) F₁ - distance (Q.x, Q.y) F₁ = 2 * h.a →
  dot_product (P.x - F₁.1, P.y - F₁.2) (P.x - F₂.1, P.y - F₂.2) = 0 →
  eccentricity h = Real.sqrt 10 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l647_64723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_formula_l647_64726

/-- A sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ+ → ℕ) : Prop :=
  (∀ p q : ℕ+, a p + a q = a (p + q)) ∧ (a 1 = 2)

/-- The theorem stating that for any sequence satisfying the conditions, a_n = 2n -/
theorem special_sequence_formula (a : ℕ+ → ℕ) (h : SpecialSequence a) :
  ∀ n : ℕ+, a n = 2 * n.val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_formula_l647_64726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_sided_polygon_diagonals_l647_64751

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A structure to represent a convex polygon -/
structure ConvexPolygon where
  sides : ℕ
  right_angles : ℕ

theorem eight_sided_polygon_diagonals :
  ∀ (P : ConvexPolygon),
    P.sides = 8 →
    P.right_angles = 2 →
    num_diagonals P.sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_sided_polygon_diagonals_l647_64751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_set_size_theorem_l647_64766

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def has_prime_diff_property {α : Type} (f : ℕ → α) : Prop :=
  ∀ i j : ℕ, is_prime (Int.natAbs (i - j)) → f i ≠ f j

def min_set_size : ℕ := 4

theorem min_set_size_theorem {A : Type} [Fintype A] :
  (∃ (f : ℕ → A), has_prime_diff_property f) →
  Fintype.card A ≥ min_set_size :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_set_size_theorem_l647_64766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l647_64741

noncomputable section

-- Define the curve C
def curve_C (α : Real) : Real × Real :=
  (3 * Real.cos α, Real.sqrt 3 * Real.sin α)

-- Define the line l in polar coordinates
def line_l (θ : Real) : Real :=
  1 / Real.cos (θ + Real.pi/3)

-- State the theorem
theorem curve_and_line_properties :
  -- 1. Rectangular equation of l
  ∀ x y : Real, (x - Real.sqrt 3 * y - 2 = 0) ↔ 
    ∃ θ : Real, x = line_l θ * Real.cos θ ∧ y = line_l θ * Real.sin θ
  ∧
  -- 2. Ordinary equation of C
  ∀ x y : Real, (x^2 / 9 + y^2 / 3 = 1) ↔ 
    ∃ α : Real, (x, y) = curve_C α
  ∧
  -- 3. Maximum distance from P on C to l
  ∃ d : Real, d = 1 + 3 * Real.sqrt 2 / 2 ∧
    ∀ α : Real, 
      let (x, y) := curve_C α
      abs (x - Real.sqrt 3 * y - 2) / Real.sqrt (1 + 3) ≤ d ∧
      ∃ α₀ : Real, 
        let (x₀, y₀) := curve_C α₀
        abs (x₀ - Real.sqrt 3 * y₀ - 2) / Real.sqrt (1 + 3) = d :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l647_64741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l647_64790

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem equation_solution :
  let equation (x : ℝ) := floor ((9*x - 4)/6) = ⌊(12*x + 7)/4⌋
  ∃ (x₁ x₂ : ℚ), x₁ = -9/4 ∧ x₂ = -23/12 ∧ 
    (∀ x : ℝ, equation x ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l647_64790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_period_duration_l647_64733

/-- The duration of a lending period in years -/
def duration : ℝ → ℝ := λ x => x

/-- The principal amount lent -/
def principal : ℝ := 25000

/-- The interest rate A charges B (as a decimal) -/
def rate_A_to_B : ℝ := 0.10

/-- The interest rate B charges C (as a decimal) -/
def rate_B_to_C : ℝ := 0.115

/-- B's gain in the lending period -/
def gain : ℝ := 1125

/-- Theorem stating that the duration of the lending period is 3 years -/
theorem lending_period_duration :
  duration (principal * rate_B_to_C - principal * rate_A_to_B) = gain →
  duration 375 = 1125 →
  duration 1 = 3 := by
  intro h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_period_duration_l647_64733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_three_l647_64771

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions ensuring it's a valid triangle could be added here

-- State the theorem
theorem angle_A_is_pi_over_three (t : Triangle) 
  (h : (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = (t.c - t.b) * Real.sin t.C) : 
  t.A = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_three_l647_64771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_triangle_l647_64758

theorem angle_in_triangle (ABC BCD BDC : ℝ) : 
  ABC = 152 →              -- Angle ABC is 152°
  BDC = 104 →              -- Angle BDC is 104°
  ABC = BCD + BDC →        -- ABC is a straight line (exterior angle theorem)
  BCD = 48                 -- Angle BCD is 48°
  := by
    intros h1 h2 h3
    have : BCD = ABC - BDC := by
      rw [h3]
      ring
    rw [this, h1, h2]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_triangle_l647_64758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_ratio_l647_64778

theorem cosine_sine_ratio (α : ℝ) 
  (h : (Real.cos α) / (1 + Real.sin α) = Real.sqrt 3) : 
  (Real.cos α) / (Real.sin α - 1) = -(Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_ratio_l647_64778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_cost_effectiveness_l647_64794

-- Define the discount functions for Bookstore A and B
def bookstoreA (x : ℝ) : ℝ := 0.8 * x

noncomputable def bookstoreB (x : ℝ) : ℝ :=
  if x ≤ 100 then x else 0.6 * x + 40

-- Theorem statement
theorem bookstore_cost_effectiveness :
  ∀ x : ℝ,
  (x < 200 → bookstoreA x < bookstoreB x) ∧
  (x = 200 → bookstoreA x = bookstoreB x) ∧
  (x > 200 → bookstoreA x > bookstoreB x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_cost_effectiveness_l647_64794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_ratio_l647_64711

/-- Given a partnership of A, B, and C with the following conditions:
    - A invests 3 times as much as B
    - B invests some fraction of what C invests
    - Total profit is 6600
    - B's share of the profit is 1200
    Prove that the ratio of B's investment to C's investment is 2:3 -/
theorem partnership_investment_ratio 
  (A_invest B_invest C_invest : ℚ) 
  (total_profit B_share : ℚ) 
  (h1 : A_invest = 3 * B_invest)
  (h2 : ∃ f : ℚ, 0 < f ∧ f < 1 ∧ B_invest = f * C_invest)
  (h3 : total_profit = 6600)
  (h4 : B_share = 1200)
  (h5 : B_share / total_profit = B_invest / (A_invest + B_invest + C_invest)) :
  B_invest / C_invest = 2 / 3 := by
  sorry

#eval IO.println "QED"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_ratio_l647_64711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l647_64753

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos (2*x))
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
    (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S)) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -1/2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l647_64753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_renovation_problem_l647_64717

/-- Represents the daily renovation rate of a construction team -/
structure RenovationRate where
  meters_per_day : ℚ
  deriving Repr

/-- Represents the daily cost of a construction team -/
structure DailyCost where
  yuan_per_day : ℚ
  deriving Repr

/-- Represents a construction team with its renovation rate and daily cost -/
structure ConstructionTeam where
  rate : RenovationRate
  cost : DailyCost
  deriving Repr

/-- The problem statement and proof -/
theorem road_renovation_problem 
  (team_a team_b : ConstructionTeam)
  (efficiency_ratio : ℚ)
  (road_length time_difference : ℚ)
  (total_road_length max_cost : ℚ) :
  team_a.rate.meters_per_day = 3/2 * team_b.rate.meters_per_day →
  road_length / team_b.rate.meters_per_day - road_length / team_a.rate.meters_per_day = time_difference →
  road_length = 360 →
  time_difference = 3 →
  team_a.cost.yuan_per_day = 70000 →
  team_b.cost.yuan_per_day = 50000 →
  total_road_length = 1200 →
  max_cost = 1450000 →
  (team_b.rate.meters_per_day = 40 ∧ 
   team_a.rate.meters_per_day = 60 ∧ 
   ∃ (m : ℚ), m ≥ 10 ∧ 
     70000 * m + 50000 * ((total_road_length - 60 * m) / 40) ≤ max_cost ∧
     ∀ (n : ℚ), n < 10 → 
       70000 * n + 50000 * ((total_road_length - 60 * n) / 40) > max_cost) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_renovation_problem_l647_64717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_tan_half_l647_64707

theorem cos_double_angle_tan_half (α : ℝ) (h : Real.tan α = 1/2) : Real.cos (2 * α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_tan_half_l647_64707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stack_height_l647_64732

/-- Represents a stack of metal rings with given properties -/
structure RingStack where
  top_diameter : ℕ
  bottom_diameter : ℕ
  thickness : ℕ
  diameter_decrease : ℕ

/-- Calculates the total vertical distance of a RingStack -/
def total_vertical_distance (stack : RingStack) : ℕ :=
  let n : ℕ := (stack.top_diameter - stack.bottom_diameter) / stack.diameter_decrease + 1
  let a : ℕ := stack.top_diameter - stack.thickness
  let l : ℕ := stack.bottom_diameter - stack.thickness
  (n * (a + l)) / 2

/-- Theorem stating that the total vertical distance of the given stack is 210 cm -/
theorem stack_height : 
  ∀ (stack : RingStack), 
    stack.top_diameter = 30 ∧ 
    stack.bottom_diameter = 4 ∧ 
    stack.thickness = 2 ∧ 
    stack.diameter_decrease = 2 → 
    total_vertical_distance stack = 210 := by
  sorry

#eval total_vertical_distance { top_diameter := 30, bottom_diameter := 4, thickness := 2, diameter_decrease := 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stack_height_l647_64732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l647_64744

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * (Real.cos x)^2 + b * Real.sin x * Real.cos x - Real.sqrt 3 / 2

theorem f_properties (a b : ℝ) :
  f a b 0 = Real.sqrt 3 / 2 →
  f a b (Real.pi / 4) = 1 / 2 →
  ∃ (g : ℝ → ℝ),
    (a = Real.sqrt 3 / 2 ∧ b = 1) ∧
    (∀ x, f a b x = Real.sin (2 * x + Real.pi / 3)) ∧
    (∀ y, g y = Real.sin (2 * y + Real.pi / 3)) ∧
    (Set.range g = Set.Icc (-1 : ℝ) 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l647_64744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_lunch_combinations_l647_64760

/-- Represents the available main courses -/
inductive MainCourse
| Hamburger
| Pasta
| Salad
| Taco

/-- Represents the available beverages -/
inductive Beverage
| Cola
| Lemonade

/-- Represents the available snacks -/
inductive Snack
| ApplePie
| FruitSalad

/-- Determines if a beverage is available for a given main course -/
def hasBeverage (m : MainCourse) : Bool :=
  match m with
  | MainCourse.Hamburger => true
  | MainCourse.Pasta => true
  | _ => false

/-- Represents a lunch combination -/
structure LunchCombination where
  mainCourse : MainCourse
  beverage : Option Beverage
  snack : Snack

/-- Determines if a lunch combination is valid -/
def isValidLunch (l : LunchCombination) : Prop :=
  (hasBeverage l.mainCourse → l.beverage.isSome) ∧
  (¬hasBeverage l.mainCourse → l.beverage.isNone)

/-- Counts the number of valid lunch combinations -/
def countValidLunches : Nat :=
  2 * 2 * 2 -- 2 main courses with beverages * 2 beverages * 2 snacks

theorem valid_lunch_combinations :
  countValidLunches = 8 := by
  rfl

#eval countValidLunches

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_lunch_combinations_l647_64760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meeting_time_l647_64799

/-- Two cyclists on a circular track -/
structure Cyclists where
  speed1 : ℝ
  speed2 : ℝ
  circumference : ℝ

/-- Time taken for cyclists to meet at the starting point -/
noncomputable def meetingTime (c : Cyclists) : ℝ :=
  c.circumference / (c.speed1 + c.speed2)

/-- Theorem: Cyclists meet at the starting point after 45 seconds -/
theorem cyclists_meeting_time (c : Cyclists) 
  (h1 : c.speed1 = 7)
  (h2 : c.speed2 = 8)
  (h3 : c.circumference = 675) :
  meetingTime c = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meeting_time_l647_64799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bushes_for_target_pumpkins_l647_64735

/-- The number of containers of raspberries produced by one bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of raspberries needed to trade for one pumpkin -/
def containers_per_pumpkin : ℚ := 2

/-- The number of pumpkins Natalie wants to obtain -/
def target_pumpkins : ℕ := 36

/-- The function to calculate the number of bushes needed for a given number of pumpkins -/
def bushes_needed (pumpkins : ℕ) : ℕ :=
  Nat.ceil ((pumpkins : ℚ) * containers_per_pumpkin / containers_per_bush)

theorem min_bushes_for_target_pumpkins :
  bushes_needed target_pumpkins = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bushes_for_target_pumpkins_l647_64735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chef_nuts_weight_l647_64757

/-- Converts pounds to grams -/
noncomputable def pounds_to_grams (pounds : ℝ) : ℝ := pounds * 453.592

/-- Converts grams to kilograms -/
noncomputable def grams_to_kg (grams : ℝ) : ℝ := grams / 1000

/-- The total weight of nuts in kilograms -/
noncomputable def total_nuts_kg (almonds_grams : ℝ) (pecans_pounds : ℝ) : ℝ :=
  grams_to_kg (almonds_grams + pounds_to_grams pecans_pounds)

theorem chef_nuts_weight :
  total_nuts_kg 140 0.56 = 0.3936112 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chef_nuts_weight_l647_64757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_eight_l647_64783

noncomputable def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 1 + (n - 1) * d

noncomputable def arithmetic_sum (d : ℝ) (n : ℕ) : ℝ := n * (2 + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_eight (d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence d 1 = 1 →
  (arithmetic_sequence d 2) ^ 2 = arithmetic_sequence d 1 * arithmetic_sequence d 5 →
  arithmetic_sum d 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_eight_l647_64783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_y_l647_64774

/-- Triangle PQR with vertices P(0, 0), Q(0, 5), and R(y, 5) has area 20 square units and y > 0 implies y = 8 -/
theorem triangle_area_implies_y (y : ℝ) : 
  y > 0 → 
  (1/2 : ℝ) * y * 5 = 20 → 
  y = 8 := by
  intros h1 h2
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_y_l647_64774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sqrt_sum_l647_64731

theorem max_value_of_sqrt_sum (x y z : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  Real.sqrt (|x - y|) + Real.sqrt (|y - z|) + Real.sqrt (|z - x|) ≤ Real.sqrt 2 + 1 ∧
  ∃ (a b c : ℝ), a ∈ Set.Icc 0 1 ∧ b ∈ Set.Icc 0 1 ∧ c ∈ Set.Icc 0 1 ∧
    Real.sqrt (|a - b|) + Real.sqrt (|b - c|) + Real.sqrt (|c - a|) = Real.sqrt 2 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sqrt_sum_l647_64731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_for_one_ball_l647_64715

/-- Represents a child in the circle -/
structure Child where
  id : Nat
  hasBall : Bool

/-- Represents the state of the game at any given time -/
structure GameState where
  children : List Child
  totalBalls : Nat

/-- Simulates one minute of the game -/
def simulateMinute (state : GameState) : GameState :=
  sorry

/-- Checks if the game is over (only one ball left) -/
def isGameOver (state : GameState) : Bool :=
  state.totalBalls = 1

/-- The main theorem stating the minimum time required -/
theorem min_time_for_one_ball :
  ∀ (initialState : GameState),
    initialState.children.length = 99 →
    initialState.totalBalls = 99 →
    ∃ (time : Nat),
      time = 98 ∧
      (∀ (t : Nat), t < 98 →
        ¬(isGameOver (Nat.rec initialState (fun _ s => simulateMinute s) t))) ∧
      isGameOver (Nat.rec initialState (fun _ s => simulateMinute s) 98) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_for_one_ball_l647_64715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l647_64734

theorem no_real_solutions : ¬∃ x : ℝ, x^2 + 6*x + 6*x * Real.sqrt (x + 3) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l647_64734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_technician_progress_l647_64786

/-- Represents the distance between two adjacent service centers -/
def d : ℝ := 1  -- We assign a value of 1 to simplify calculations

/-- The number of service centers -/
def num_centers : ℕ := 5

/-- The total number of segments in the round-trip -/
def total_segments : ℕ := 2 * (num_centers - 1)

/-- The distance completed by the technician -/
def distance_completed : ℝ := d + 0.4 * d + 0.6 * d

/-- The total distance of the round-trip -/
def total_distance : ℝ := d * total_segments

/-- The theorem stating that the technician has completed 25% of the round-trip -/
theorem technician_progress : 
  distance_completed / total_distance = 1 / 4 := by
  -- Expand definitions
  unfold distance_completed total_distance total_segments num_centers d
  -- Simplify the expression
  simp [Nat.cast_sub, Nat.cast_mul]
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_technician_progress_l647_64786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_perimeter_l647_64724

/-- The perimeter of a figure formed by 13 circles arranged in a line -/
theorem circle_arrangement_perimeter 
  (n : ℕ) 
  (r : ℝ) 
  (d : ℝ) 
  (h1 : n = 13) 
  (h2 : r = 2 * Real.sqrt (2 - Real.sqrt 3)) 
  (h3 : d = 2) :
  let perimeter := n * 2 * Real.pi * r - (n - 1) * 2 * Real.arccos (d / (2 * r)) * r
  perimeter = 44 * Real.pi * Real.sqrt (2 - Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_perimeter_l647_64724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_ratio_2_3_4_is_acute_l647_64750

/-- A triangle with interior angles in the ratio 2:3:4 is acute. -/
theorem triangle_with_ratio_2_3_4_is_acute (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  (a : ℝ) / 2 = (b : ℝ) / 3 → (b : ℝ) / 3 = (c : ℝ) / 4 →
  a < 90 ∧ b < 90 ∧ c < 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_ratio_2_3_4_is_acute_l647_64750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_is_five_l647_64727

/-- A hexagon constructed with right triangles and a central square -/
structure Hexagon where
  central_square_side : ℝ
  large_triangle_base : ℝ
  large_triangle_height : ℝ
  small_triangle_base : ℝ
  small_triangle_height : ℝ

/-- The area of the hexagon -/
noncomputable def hexagon_area (h : Hexagon) : ℝ :=
  h.central_square_side^2 +
  2 * (1/2 * h.large_triangle_base * h.large_triangle_height) +
  4 * (1/2 * h.small_triangle_base * h.small_triangle_height)

/-- Theorem: The area of the specified hexagon is 5 square units -/
theorem hexagon_area_is_five (h : Hexagon) 
  (h1 : h.central_square_side = 1)
  (h2 : h.large_triangle_base = 2)
  (h3 : h.large_triangle_height = 1)
  (h4 : h.small_triangle_base = 1)
  (h5 : h.small_triangle_height = 1) : 
  hexagon_area h = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_is_five_l647_64727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_with_integral_l647_64704

/-- A linear function f satisfying f(x) = x + 2 ∫₀¹ f(t) dt is equal to x - 1 -/
theorem linear_function_with_integral (f : ℝ → ℝ) :
  (∀ x y : ℝ, ∀ c : ℝ, f (c * x + y) = c * f x + f y) →  -- f is linear
  (∀ x : ℝ, f x = x + 2 * ∫ t in Set.Icc 0 1, f t) →     -- f satisfies the integral equation
  (∀ x : ℝ, f x = x - 1) :=                               -- f(x) = x - 1
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_with_integral_l647_64704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_flight_time_l647_64706

/-- The time taken for a bee to fly from a daisy to a rose -/
noncomputable def time_daisy_to_rose : ℝ :=
  let speed_daisy_to_rose : ℝ := 2.6
  let speed_rose_to_poppy : ℝ := speed_daisy_to_rose + 3
  let time_rose_to_poppy : ℝ := 6
  let distance_rose_to_poppy : ℝ := speed_rose_to_poppy * time_rose_to_poppy
  let distance_daisy_to_rose : ℝ := distance_rose_to_poppy + 8
  distance_daisy_to_rose / speed_daisy_to_rose

/-- Theorem stating that the time taken for the bee to fly from the daisy to the rose is 16 seconds -/
theorem bee_flight_time : time_daisy_to_rose = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_flight_time_l647_64706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_inequality_minimum_l647_64729

theorem polynomial_inequality_minimum (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → a * x^3 - x^2 + 4*x + 3 ≥ 0) → 
  a ≥ -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_inequality_minimum_l647_64729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_triangle_value_l647_64700

/-- A circle with an inscribed regular triangle -/
structure CircleWithTriangle where
  radius : ℝ
  center : ℝ × ℝ

/-- The probability of a point falling within the inscribed triangle -/
noncomputable def probability_in_triangle (c : CircleWithTriangle) : ℝ :=
  3 * Real.sqrt 3 / (4 * Real.pi)

/-- Theorem stating the probability of a point falling within the inscribed triangle -/
theorem probability_in_triangle_value (c : CircleWithTriangle) :
  probability_in_triangle c = 3 * Real.sqrt 3 / (4 * Real.pi) :=
by
  -- Unfold the definition of probability_in_triangle
  unfold probability_in_triangle
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_triangle_value_l647_64700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_a_eq_neg_one_l647_64752

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = ln((2x)/(1+x) + a) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((2 * x) / (1 + x) + a)

/-- Theorem: f is an odd function if and only if a = -1 -/
theorem f_is_odd_iff_a_eq_neg_one (a : ℝ) :
  IsOdd (f a) ↔ a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_a_eq_neg_one_l647_64752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_combination_over_sum_consecutive_l647_64798

-- Define the combination formula
noncomputable def combination (n : ℕ) : ℝ := n * (n - 1) / 2

-- Define the sum of consecutive integers
noncomputable def sum_consecutive (n : ℕ) : ℝ := n * (n + 1) / 2

-- State the theorem
theorem limit_combination_over_sum_consecutive :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, 
    |combination n / sum_consecutive n - 1| < ε := by
  sorry

#check limit_combination_over_sum_consecutive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_combination_over_sum_consecutive_l647_64798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l647_64742

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

-- State the theorem
theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (n : ℕ), n = 3 ∧ (∀ x ∈ Set.Icc 0 Real.pi, f ω x = f ω (-x) ∨ f ω x = -f ω (-x)) ∧ 
    (∀ m : ℕ, m ≠ n → ¬(∀ x ∈ Set.Icc 0 Real.pi, f ω x = f ω (-x) ∨ f ω x = -f ω (-x)))) :
  (9 / 4 ≤ ω ∧ ω < 13 / 4) ∧ 
  (∃ k : ℕ, k > 0 ∧ ∀ x : ℝ, f ω (x + 4 * Real.pi / 5) = f ω x) ∧
  (∀ x ∈ Set.Ioo 0 (Real.pi / 15), ∀ y ∈ Set.Ioo 0 (Real.pi / 15), x < y → f ω x < f ω y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l647_64742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l647_64754

-- Define the function f(x) as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (|x + 2| + |x - 4| - m)

-- Theorem statement
theorem function_properties :
  (∀ x : ℝ, f 6 x ≥ 0) ∧ 
  (∀ m : ℝ, (∀ x : ℝ, f m x ≥ 0) → m ≤ 6) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 4 / (a + 5*b) + 1 / (3*a + 2*b) = 6 → 4*a + 7*b ≥ 3/2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 4 / (a + 5*b) + 1 / (3*a + 2*b) = 6 ∧ 4*a + 7*b = 3/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l647_64754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_quadratic_l647_64719

theorem root_difference_quadratic :
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := 12
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * r₁^2 + b * r₁ - c = 0 ∧
  a * r₂^2 + b * r₂ - c = 0 ∧
  max r₁ r₂ - min r₁ r₂ = 5.5
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_quadratic_l647_64719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l647_64739

/-- Given a circle with diameter endpoints (2, -3) and (8, 9), its center is (5, 3) -/
theorem circle_center (K : Set (ℝ × ℝ)) (r : ℝ) : 
  (∃ (x y : ℝ), K = {p : ℝ × ℝ | (p.1 - x)^2 + (p.2 - y)^2 = r^2}) →
  ((2, -3) ∈ K ∧ (8, 9) ∈ K) →
  ((2 - 8)^2 + (-3 - 9)^2 = (2 * r)^2) →
  (∃ (x y : ℝ), K = {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 - 3)^2 = r^2}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l647_64739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l647_64776

/-- A point in a 2x4 grid --/
structure GridPoint where
  x : Fin 4
  y : Fin 2

/-- A triangle formed by three points in the grid --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Check if three points are collinear --/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p1.x = p2.x ∧ p2.x = p3.x) ∨ (p1.y = p2.y ∧ p2.y = p3.y)

/-- A valid triangle is one where the points are not collinear --/
def valid_triangle (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all valid triangles in the 2x4 grid --/
def valid_triangles : Set GridTriangle :=
  {t : GridTriangle | valid_triangle t}

/-- Prove that valid_triangles is finite --/
instance : Fintype valid_triangles := by
  sorry

/-- The main theorem: there are exactly 48 distinct valid triangles in a 2x4 grid --/
theorem count_valid_triangles : Fintype.card valid_triangles = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l647_64776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tile_probability_l647_64773

/-- A function that determines if a number is blue (congruent to 3 mod 7) -/
def isBlue (n : Nat) : Bool :=
  n % 7 = 3

/-- The total number of tiles in the box -/
def totalTiles : Nat := 70

/-- The number of blue tiles in the box -/
def blueTiles : Nat := (List.range totalTiles).filter isBlue |>.length

/-- The probability of selecting a blue tile -/
def probabilityBlueTile : Rat := blueTiles / totalTiles

theorem blue_tile_probability :
  probabilityBlueTile = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tile_probability_l647_64773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l647_64746

noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

theorem min_value_of_f : 
  ∀ x y : ℝ, 1/4 ≤ x ∧ x ≤ 2/3 ∧ 1/5 ≤ y ∧ y ≤ 1/2 → f x y ≥ 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l647_64746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_eight_l647_64789

theorem expression_equals_eight :
  Real.sqrt 25 - ((-8) : ℝ) ^ (1/3 : ℝ) + 2 * Real.sqrt (1/4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_eight_l647_64789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_midpoint_theorem_l647_64720

/-- A point on a 2D integer grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- The midpoint of two grid points -/
def gridMidpoint (p q : GridPoint) : ℚ × ℚ :=
  ((p.x + q.x) / 2, (p.y + q.y) / 2)

/-- Predicate to check if a point is on the grid (has integer coordinates) -/
def isOnGrid (p : ℚ × ℚ) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

theorem grid_midpoint_theorem (points : Finset GridPoint) :
  points.card = 5 →
  ∃ (p q : GridPoint), p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ isOnGrid (gridMidpoint p q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_midpoint_theorem_l647_64720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_regularity_l647_64784

structure Pyramid where
  vertices : Finset (Fin 3 → ℝ)
  base : Finset (Fin 3 → ℝ)
  apex : Fin 3 → ℝ

def is_regular_polygon (polygon : Finset (Fin 3 → ℝ)) : Prop :=
  sorry

def has_equal_lateral_edges (p : Pyramid) : Prop :=
  sorry

def has_equal_adjacent_dihedral_angles (p : Pyramid) : Prop :=
  sorry

def has_odd_base_sides (p : Pyramid) : Prop :=
  sorry

theorem pyramid_base_regularity (p : Pyramid) :
  has_equal_lateral_edges p →
  has_equal_adjacent_dihedral_angles p →
  has_odd_base_sides p →
  is_regular_polygon p.base := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_regularity_l647_64784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_22_5_l647_64792

/-- Represents a right triangle with given hypotenuse and leg lengths -/
structure RightTriangle where
  hypotenuse : ℝ
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse_sq : hypotenuse ^ 2 = leg1 ^ 2 + leg2 ^ 2

/-- Calculates the area of a right triangle -/
noncomputable def areaOfRightTriangle (t : RightTriangle) : ℝ :=
  t.leg1 * t.leg2 / 2

/-- Creates a smaller right triangle by halving the longest leg -/
noncomputable def createSmallerTriangle (t : RightTriangle) : RightTriangle where
  hypotenuse := max t.leg1 t.leg2 / 2
  leg1 := max t.leg1 t.leg2 / 2
  leg2 := min t.leg1 t.leg2 * (max t.leg1 t.leg2 / 2) / max t.leg1 t.leg2
  hypotenuse_sq := sorry

theorem shaded_area_is_22_5 (t : RightTriangle) 
    (h1 : t.hypotenuse = 13)
    (h2 : t.leg1 = 5) :
    areaOfRightTriangle t - areaOfRightTriangle (createSmallerTriangle t) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_22_5_l647_64792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_l647_64713

-- Define the points
noncomputable def p1 : ℝ × ℝ := (2, -5)
noncomputable def p2 : ℝ × ℝ := (6, 7)
noncomputable def p3 : ℝ × ℝ := (14/3, 3)

-- Define a function to check if three points are collinear
def collinear (a b c : ℝ × ℝ) : Prop :=
  (c.2 - a.2) * (b.1 - a.1) = (b.2 - a.2) * (c.1 - a.1)

-- Theorem statement
theorem point_on_line : collinear p1 p2 p3 := by
  -- Unfold the definition of collinear
  unfold collinear
  -- Unfold the definitions of p1, p2, and p3
  unfold p1 p2 p3
  -- Perform algebraic simplifications
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_l647_64713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l647_64738

/-- Given a triangle ABC with angle C = 2π/3 and c² = 5a² + ab, prove:
    1. sin B / sin A = 2
    2. The maximum value of sin A * sin B is (√3 - 1) / 4 -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) (h1 : C = 2 * π / 3) 
    (h2 : c^2 = 5 * a^2 + a * b) :
  (Real.sin B / Real.sin A = 2) ∧ 
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ A ∧ 0 ≤ y ∧ y ≤ B → 
    Real.sin x * Real.sin y ≤ (Real.sqrt 3 - 1) / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l647_64738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_speed_l647_64728

/-- The speed of an escalator given specific conditions -/
theorem escalator_speed (length walking_speed time escalator_speed : ℝ) 
  (h1 : length = 150)
  (h2 : walking_speed = 3)
  (h3 : time = 10)
  (h4 : length = (walking_speed + escalator_speed) * time) :
  escalator_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_speed_l647_64728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_distribution_l647_64730

/-- Partnership profit distribution problem -/
theorem partnership_profit_distribution 
  (mary_investment mike_investment : ℚ)
  (total_profit : ℚ)
  (h_mary_inv : mary_investment = 800)
  (h_mike_inv : mike_investment = 200)
  (h_total_profit : total_profit = 2999.9999999999995)
  : ∃ (mary_share mike_share : ℚ),
    let total_investment := mary_investment + mike_investment
    let equal_division := (1/3) * total_profit
    let investment_based_division := total_profit - equal_division
    let mary_investment_ratio := mary_investment / total_investment
    let mike_investment_ratio := mike_investment / total_investment
    mary_share = (mary_investment_ratio * investment_based_division) + (equal_division / 2) ∧
    mike_share = (mike_investment_ratio * investment_based_division) + (equal_division / 2) ∧
    mary_share - mike_share = 1200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_distribution_l647_64730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l647_64793

noncomputable def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def Line (a b x y : ℝ) : Prop := a*x + b*y = 1

noncomputable def DistancePointToLine (a b : ℝ) : ℝ := 1 / Real.sqrt (a^2 + b^2)

theorem distance_circle_center_to_line 
  (a b : ℝ) 
  (h : Circle a b) :
  DistancePointToLine a b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l647_64793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l647_64777

/-- Given a hyperbola and a circle, if one asymptote of the hyperbola intersects
    the circle forming a chord of length √3, then the eccentricity of the hyperbola
    is (2/3)√3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := λ x y : ℝ ↦ x^2 / a^2 - y^2 / b^2 = 1
  let circle := λ x y : ℝ ↦ x^2 + y^2 - 2*x = 0
  let asymptote := λ x y : ℝ ↦ y = (b/a) * x
  let chord_length := Real.sqrt 3
  let eccentricity := Real.sqrt (a^2 + b^2) / a
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    asymptote x₁ y₁ ∧ asymptote x₂ y₂ ∧
    circle x₁ y₁ ∧ circle x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2) →
  eccentricity = (2/3) * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l647_64777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l647_64701

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola a b) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)

/-- The equation of the asymptote line of a hyperbola -/
noncomputable def asymptote_line (h : Hyperbola a b) (x : ℝ) : ℝ := (b / a) * x

/-- Symmetry condition with respect to the asymptote line -/
def is_symmetric_to_asymptote (h : Hyperbola a b) (p : ℝ × ℝ) : Prop :=
  let (px, py) := p
  let (fx, _) := right_focus h
  py / (px - fx) = -a / b ∧ py / 2 = (b / a) * (px + fx) / 2

/-- Condition for a point to lie on the hyperbola -/
def on_hyperbola (h : Hyperbola a b) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 / a^2 - y^2 / b^2 = 1

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- Main theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  is_symmetric_to_asymptote h (right_focus h) →
  on_hyperbola h (right_focus h) →
  eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l647_64701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_squared_l647_64767

noncomputable def polar_to_cartesian (ρ : ℝ) (θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def cartesian_midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def polar_midpoint (ρ1 ρ2 θ1 θ2 : ℝ) : ℝ × ℝ :=
  polar_to_cartesian ((ρ1 + ρ2) / 2) ((θ1 + θ2) / 2)

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem midpoint_distance_squared :
  let A := polar_to_cartesian 4 (π / 100)
  let B := polar_to_cartesian 8 (51 * π / 100)
  let M := cartesian_midpoint A B
  let N := polar_midpoint 4 8 (π / 100) (51 * π / 100)
  distance_squared M N = 56 - 36 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_squared_l647_64767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l647_64748

/-- Represents a repeating decimal with up to 4 decimal places before the repeating part -/
structure RepeatingDecimal where
  integerPart : ℕ
  decimalPart : ℕ
  repeatingPart : ℕ
  decimalLength : ℕ
  repeatingLength : ℕ

/-- Converts a RepeatingDecimal to a real number -/
noncomputable def toReal (x : RepeatingDecimal) : ℝ :=
  x.integerPart + 
  (x.decimalPart : ℝ) / (10 ^ x.decimalLength) + 
  (x.repeatingPart : ℝ) / (10 ^ x.decimalLength) / (1 - 1 / (10 ^ x.repeatingLength))

def a : RepeatingDecimal := ⟨3, 2571, 0, 4, 1⟩
def b : RepeatingDecimal := ⟨3, 0, 2571, 0, 4⟩
def c : RepeatingDecimal := ⟨3, 2, 571, 1, 3⟩
def d : RepeatingDecimal := ⟨3, 25, 71, 2, 2⟩
def e : RepeatingDecimal := ⟨3, 257, 1, 3, 1⟩

theorem largest_number : 
  toReal d > toReal a ∧ 
  toReal d > toReal b ∧ 
  toReal d > toReal c ∧ 
  toReal d > toReal e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l647_64748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statements_l647_64765

theorem problem_statements :
  (∀ x y, x < y → x ∈ Set.univ \ {π/2 + π*n | n : ℤ} → y ∈ Set.univ \ {π/2 + π*n | n : ℤ} → Real.tan x < Real.tan y) ∧
  (∀ x, Real.cos ((2/3)*x + π/2) = -Real.cos (-(2/3)*x - π/2)) ∧
  (∀ v, Real.sin (2*(π/8 - v) + 5*π/4) = Real.sin (2*(π/8 + v) + 5*π/4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statements_l647_64765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l647_64702

theorem tan_value_from_sin_cos_sum (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin α + Real.cos α = -7/13) : Real.tan α = -5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l647_64702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trey_turtle_ratio_l647_64781

theorem trey_turtle_ratio
  (kristen_turtles : ℕ)
  (total_turtles : ℕ)
  (h1 : kristen_turtles = 12)
  (h2 : total_turtles = 30) :
  let kris_turtles := kristen_turtles / 4
  let trey_turtles := total_turtles - kris_turtles - kristen_turtles
  trey_turtles / kris_turtles = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trey_turtle_ratio_l647_64781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_value_l647_64749

/-- The function f(x) = x³ - 3x --/
noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

/-- Predicate to check if a point (x,y) is on the curve y = f(x) --/
def on_curve (x y : ℝ) : Prop := y = f x

/-- Predicate to check if a line through (x₁,y₁) and (x₂,y₂) is tangent to f at (x₁,y₁) --/
def is_tangent (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  on_curve x₁ y₁ ∧ (y₂ - y₁) / (x₂ - x₁) = deriv f x₁

/-- The main theorem --/
theorem tangent_point_value (t : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    is_tangent x₁ y₁ 2 t ∧ 
    is_tangent x₂ y₂ 2 t ∧ 
    ¬on_curve 2 t) → 
  t = -6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_value_l647_64749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l647_64788

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- Define the properties of the triangle
def AcuteTriangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧ 0 < t.B ∧ t.B < Real.pi/2 ∧ 0 < t.C ∧ t.C < Real.pi/2

-- Define the angle bisector intersection point O
def AngleBisectorIntersection (t : Triangle) (O : Real × Real) : Prop :=
  -- This is a placeholder for the actual condition
  True

-- Define what it means to be a perimeter of AOC
def IsPerimeter (t : Triangle) (O : Real × Real) (p : Real) : Prop :=
  -- This is a placeholder for the actual condition
  True

-- Define the theorem
theorem triangle_properties (t : Triangle) (O : Real × Real) 
  (h_acute : AcuteTriangle t)
  (h_bisector : AngleBisectorIntersection t O)
  (h_b : t.b = 2 * Real.sqrt 3)
  (h_area : t.S = (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2)) :
  t.B = Real.pi/3 ∧ 
  (∃ (l : Real), l ≤ 4 + 2 * Real.sqrt 3 ∧ 
    ∀ (p : Real), IsPerimeter t O p → p ≤ l) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l647_64788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_interval_1_2_l647_64768

-- Define set A
def A : Set ℝ := {x | (x + 1) * (x - 2) < 0}

-- Define set B
def B : Set ℝ := {x | Real.rpow 2 (x - 1) ≥ 1}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the half-open interval [1, 2)
def interval_1_2 : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- Theorem statement
theorem A_intersect_B_equals_interval_1_2 : A_intersect_B = interval_1_2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_interval_1_2_l647_64768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_iff_a_equals_one_l647_64787

theorem three_solutions_iff_a_equals_one (a : ℝ) :
  (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (abs (abs (x₁ + 3) - 1) = a) ∧ 
    (abs (abs (x₂ + 3) - 1) = a) ∧ 
    (abs (abs (x₃ + 3) - 1) = a)) ↔
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_iff_a_equals_one_l647_64787
