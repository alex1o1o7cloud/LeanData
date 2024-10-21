import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_in_still_water_l945_94531

/-- Represents the speed of the man in still water -/
def man_speed : ℝ := 9

/-- Represents the speed of the stream -/
def stream_speed : ℝ := 3

/-- The distance traveled downstream -/
def downstream_distance : ℝ := 36

/-- The distance traveled upstream -/
def upstream_distance : ℝ := 18

/-- The time taken for both downstream and upstream journeys -/
def journey_time : ℝ := 3

/-- Theorem stating that given the conditions, the man's speed in still water is 9 km/h -/
theorem man_speed_in_still_water :
  (man_speed + stream_speed) * journey_time = downstream_distance ∧
  (man_speed - stream_speed) * journey_time = upstream_distance →
  man_speed = 9 := by
  intro h
  -- The proof goes here
  sorry

#check man_speed_in_still_water

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_in_still_water_l945_94531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_degree_formula_l945_94552

theorem tan_22_5_degree_formula : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_degree_formula_l945_94552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_same_domain_range_l945_94550

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2 / x
noncomputable def g (x : ℝ) : ℝ := x^2
noncomputable def h (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def k (x : ℝ) : ℝ := 2^x

-- Define the domains and ranges
def dom_f : Set ℝ := {x | x ≠ 0}
def range_f : Set ℝ := {y | y ≠ 0}
def dom_g : Set ℝ := Set.univ
def range_g : Set ℝ := {y | y ≥ 0}
def dom_h : Set ℝ := {x | x > 0}
def range_h : Set ℝ := Set.univ
def dom_k : Set ℝ := Set.univ
def range_k : Set ℝ := {y | y > 0}

theorem only_f_same_domain_range :
  (dom_f = range_f) ∧
  (dom_g ≠ range_g) ∧
  (dom_h ≠ range_h) ∧
  (dom_k ≠ range_k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_same_domain_range_l945_94550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ABCD_l945_94560

/-- Represents a cell in the 4x4 grid --/
structure Cell where
  row : Fin 4
  col : Fin 4
  value : Nat

/-- Represents an arrow in the grid --/
structure Arrow where
  fromCell : Cell
  toCell : Cell
  count : Nat

/-- The grid configuration --/
def grid : List Cell := sorry

/-- The arrow configuration --/
def arrows : List Arrow := sorry

/-- Checks if the grid satisfies all arrow constraints --/
def satisfies_constraints (g : List Cell) (a : List Arrow) : Prop := sorry

/-- Extracts the four-digit number ABCD from the grid --/
def extract_ABCD (g : List Cell) : Nat := sorry

/-- Main theorem --/
theorem unique_ABCD : 
  ∀ g : List Cell, 
    satisfies_constraints g arrows → extract_ABCD g = 2112 :=
by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ABCD_l945_94560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_problem_1_distance_problem_2_l945_94578

/-- Distance between two points in a plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Distance between two points on a line parallel to the y-axis -/
def distance_y_parallel (y₁ y₂ : ℝ) : ℝ :=
  |y₂ - y₁|

theorem distance_problem_1 :
  distance 2 4 (-3) (-8) = 13 := by sorry

theorem distance_problem_2 :
  distance_y_parallel 5 (-1) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_problem_1_distance_problem_2_l945_94578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_tax_calculation_l945_94509

noncomputable def calculate_tax (total_value : ℝ) (tax_free_threshold : ℝ) (tax_rate : ℝ) : ℝ :=
  max 0 ((total_value - tax_free_threshold) * tax_rate)

theorem tourist_tax_calculation :
  let total_value : ℝ := 1720
  let tax_free_threshold : ℝ := 600
  let tax_rate : ℝ := 0.12
  calculate_tax total_value tax_free_threshold tax_rate = 134.40 := by
  -- Unfold the definitions
  unfold calculate_tax
  -- Simplify the expression
  simp [max]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_tax_calculation_l945_94509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l945_94534

/-- Geometric sequence with first term a and common ratio q -/
noncomputable def geometricSequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometricSum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_problem (a q : ℝ) :
  let a_n := geometricSequence a q
  let S_n := geometricSum a q
  (S_n 3 = a_n 2 + 10 * a_n 1) → (a_n 5 = 9) → (a_n 1 = 1/9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l945_94534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_monomials_l945_94507

-- Define a structure for algebraic terms
structure AlgebraicTerm where
  constant : ℚ
  vars : List Char
  powers : List ℕ

-- Define what a monomial is
def isMonomial (term : AlgebraicTerm) : Bool :=
  term.constant ≠ 0 || term.vars.length > 0

-- Define the list of terms in the expression
def expressionTerms : List AlgebraicTerm :=
  [
    { constant := 1/3, vars := ['a', 'b'], powers := [1, 1] },
    { constant := -2/3, vars := ['a', 'b', 'c'], powers := [1, 1, 1] },
    { constant := 0, vars := [], powers := [] },
    { constant := -5, vars := [], powers := [] },
    { constant := 1, vars := ['x'], powers := [1] },
    { constant := -1, vars := ['y'], powers := [1] },
    { constant := 2, vars := ['x'], powers := [1] }  -- Changed -1 to 1 for consistency
  ]

-- Theorem to prove
theorem number_of_monomials : 
  (expressionTerms.filter isMonomial).length = 4 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_monomials_l945_94507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_result_l945_94593

theorem expression_result (x y : ℝ) 
  (h1 : x ∈ ({-Real.sqrt 2, Real.sqrt 3, Real.sqrt 6} : Set ℝ))
  (h2 : y ∈ ({-Real.sqrt 2, Real.sqrt 3, Real.sqrt 6} : Set ℝ))
  (h3 : x ≠ y) :
  (x + y)^2 / Real.sqrt 2 ∈ ({5 * Real.sqrt 2 / 2 - 2 * Real.sqrt 3,
                             4 * Real.sqrt 2 - 2 * Real.sqrt 6,
                             9 * Real.sqrt 2 / 2 + 6} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_result_l945_94593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_theorem_l945_94514

/-- Represents a parallelogram ABCD with projections P, Q, R, S -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  area : ℝ
  pq_length : ℝ
  rs_length : ℝ
  is_parallelogram : Prop
  p_q_on_bd : Prop
  r_s_on_ac : Prop

/-- The squared length of the longer diagonal of the parallelogram -/
noncomputable def diagonal_length_squared (p : Parallelogram) : ℝ := 
  let d := Real.sqrt ((p.B.1 - p.D.1)^2 + (p.B.2 - p.D.2)^2)
  d^2

/-- Theorem stating the properties of the specific parallelogram and its diagonal length -/
theorem parallelogram_diagonal_theorem (p : Parallelogram) 
  (h1 : p.area = 24)
  (h2 : p.pq_length = 8)
  (h3 : p.rs_length = 10)
  (h4 : p.is_parallelogram)
  (h5 : p.p_q_on_bd)
  (h6 : p.r_s_on_ac) :
  diagonal_length_squared p = 104 + 22 * Real.sqrt 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_theorem_l945_94514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_eccentricity_l945_94591

-- Define the curve
noncomputable def curve (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define eccentricity
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

-- Theorem statement
theorem curve_eccentricity :
  ∃ (a c : ℝ), a > 0 ∧ c > 0 ∧
  (∀ x y : ℝ, curve x y ↔ x^2 / a^2 + y^2 = 1) ∧
  eccentricity a c = 2 * Real.sqrt 2 / 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_eccentricity_l945_94591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_300_l945_94511

/-- Fixed cost for producing the electronic instrument -/
noncomputable def fixed_cost : ℝ := 20000

/-- Variable cost per unit produced -/
noncomputable def variable_cost : ℝ := 100

/-- Total revenue function -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 500 then 400 * x - (1/2) * x^2
  else 75000

/-- Profit function -/
noncomputable def f (x : ℝ) : ℝ :=
  R x - (fixed_cost + variable_cost * x)

/-- Theorem stating the maximum profit and corresponding production -/
theorem max_profit_at_300 :
  (∃ (x : ℝ), f x = 25000) ∧
  (∀ (x : ℝ), f x ≤ 25000) ∧
  (f 300 = 25000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_300_l945_94511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_value_on_square_root_curve_l945_94536

theorem tangent_value_on_square_root_curve (a : ℝ) : 
  a = Real.sqrt 4 → Real.tan (a * π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_value_on_square_root_curve_l945_94536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_portion_is_two_l945_94554

/-- Represents the distribution of loaves in an arithmetic sequence -/
structure BreadDistribution where
  a : ℕ
  d : ℕ
  h_d_pos : d > 0

/-- The problem of dividing bread loaves -/
def bread_problem (dist : BreadDistribution) : Prop :=
  let portions := [dist.a - 2*dist.d, dist.a - dist.d, dist.a, dist.a + dist.d, dist.a + 2*dist.d]
  (portions.sum = 120) ∧
  (portions[2]! + portions[3]! + portions[4]! = 7 * (portions[0]! + portions[1]!))

/-- The theorem to be proved -/
theorem smallest_portion_is_two :
  ∃ (dist : BreadDistribution), bread_problem dist ∧ (dist.a - 2*dist.d = 2) := by
  -- Construct the BreadDistribution
  let dist : BreadDistribution := {
    a := 24
    d := 11
    h_d_pos := by norm_num
  }
  
  -- Prove that this distribution satisfies the problem conditions
  have h_problem : bread_problem dist := by
    simp [bread_problem]
    norm_num
    rfl

  -- Prove that the smallest portion is 2
  have h_smallest : dist.a - 2*dist.d = 2 := by
    norm_num

  -- Combine the proofs
  exact ⟨dist, h_problem, h_smallest⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_portion_is_two_l945_94554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_lambda_l945_94543

theorem greatest_lambda : ∃ (l : ℝ), l > 0 ∧ 
  (∀ (a b : ℝ), l * a^2 * b^2 * (a + b)^2 ≤ (a^2 + a*b + b^2)^3) ∧ 
  (∀ (m : ℝ), m > l → ∃ (a b : ℝ), m * a^2 * b^2 * (a + b)^2 > (a^2 + a*b + b^2)^3) :=
by
  use 27/4
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_lambda_l945_94543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_years_to_future_value_example_l945_94590

/-- The number of years for a present value to grow to a future value at a given interest rate -/
noncomputable def years_to_future_value (present_value : ℝ) (future_value : ℝ) (interest_rate : ℝ) : ℝ :=
  Real.log (future_value / present_value) / Real.log (1 + interest_rate)

/-- Theorem stating that the number of years for $156.25 to grow to $169 at 4% interest is approximately 2 -/
theorem years_to_future_value_example : 
  ∃ (n : ℝ), abs (n - years_to_future_value 156.25 169 0.04) < 0.1 ∧ Int.floor n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_years_to_future_value_example_l945_94590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_fraction_is_one_third_l945_94569

/-- A cement mixture with sand, water, and gravel -/
structure CementMixture where
  total_weight : ℚ
  water_fraction : ℚ
  gravel_weight : ℚ

/-- The fraction of sand in the cement mixture -/
def sand_fraction (m : CementMixture) : ℚ :=
  1 - m.water_fraction - m.gravel_weight / m.total_weight

/-- Theorem stating that for a specific cement mixture, the sand fraction is 1/3 -/
theorem sand_fraction_is_one_third :
  let m : CementMixture := {
    total_weight := 48,
    water_fraction := 1/2,
    gravel_weight := 8
  }
  sand_fraction m = 1/3 := by
  -- Proof goes here
  sorry

#eval sand_fraction {
  total_weight := 48,
  water_fraction := 1/2,
  gravel_weight := 8
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_fraction_is_one_third_l945_94569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l945_94542

/-- The coefficient of x in the simplified expression of 2(x - 5) + 5(8 - 3x^2 + 4x) - 9(3x - 2) is -5 -/
theorem coefficient_of_x (x : ℝ) : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 4 * x) - 9 * (3 * x - 2)
  let simplified := -15 * x^2 - 5 * x + 48
  expr = simplified ∧ 
  ∃ (a b c : ℝ), simplified = a * x^2 + b * x + c ∧ b = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l945_94542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l945_94506

/-- Given an arithmetic sequence with a non-zero common difference,
    if its 4th, 7th, and 16th terms are the 4th, 6th, and 8th terms
    of a geometric sequence respectively, then the common ratio
    of the geometric sequence is ±√3. -/
theorem arithmetic_geometric_sequence_ratio
  (a₁ d q : ℝ) -- a₁ is the first term, d is the common difference, q is the common ratio
  (h₁ : d ≠ 0)
  (h₂ : a₁ + 3*d = (a₁ + 6*d) / q) -- 4th term of arithmetic = 4th term of geometric
  (h₃ : a₁ + 6*d = (a₁ + 3*d) * q) -- 7th term of arithmetic = 6th term of geometric
  (h₄ : a₁ + 15*d = (a₁ + 3*d) * q^2) -- 16th term of arithmetic = 8th term of geometric
  : q = Real.sqrt 3 ∨ q = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l945_94506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_arithmetic_sequence_properties_l945_94556

/-- An increasing arithmetic sequence with specific properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℚ  -- Changed ℝ to ℚ for computability
  increasing : ∀ n, a n < a (n + 1)
  sum_first_three : a 1 + a 2 + a 3 = -3
  product_first_three : a 1 * a 2 * a 3 = 8

/-- The general term of the special arithmetic sequence -/
def general_term (n : ℕ) : ℚ := 3 * n - 7

/-- The sum of the first n terms of the special arithmetic sequence -/
def sum_n_terms (n : ℕ) : ℚ := (3 / 2) * n^2 - (11 / 2) * n

/-- Theorem stating the properties of the special arithmetic sequence -/
theorem special_arithmetic_sequence_properties (seq : SpecialArithmeticSequence) :
  (∀ n, seq.a n = general_term n) ∧
  (∀ n, (Finset.range n).sum seq.a = sum_n_terms n) := by
  sorry

#check special_arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_arithmetic_sequence_properties_l945_94556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_from_return_time_ratio_l945_94533

/-- An ellipse with a left focus -/
structure Ellipse where
  a : ℝ -- Semi-major axis
  c : ℝ -- Distance from center to focus
  h_positive : 0 < a
  h_c_less_a : c < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- The ratio of longest to shortest return time for a ball in the ellipse -/
noncomputable def return_time_ratio (e : Ellipse) : ℝ := (4 * e.a) / (2 * (e.a - e.c))

/-- Theorem: If the return time ratio is 5, then the eccentricity is 3/5 -/
theorem eccentricity_from_return_time_ratio (e : Ellipse) :
  return_time_ratio e = 5 → eccentricity e = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_from_return_time_ratio_l945_94533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_probability_l945_94594

open BigOperators

def probability_of_event (p : ℚ) : ℚ :=
  let success_probability := p^3 * (1-p)^3  -- Probability of 3 successes in 6 trials
  let combinations := Nat.choose 5 2  -- Number of ways to choose 2 successes from first 5 shots
  ↑combinations * success_probability

theorem basketball_probability :
  probability_of_event (1/2) = 5/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_probability_l945_94594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_theorem_l945_94504

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an inversion circle
structure InversionCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the image of a circle under inversion
noncomputable def imageCircle (c : Circle) (i : InversionCircle) : Circle := 
  sorry

-- Define a function to check if points are collinear
def areCollinear (p1 p2 p3 : ℝ × ℝ) : Prop := 
  sorry

-- Main theorem
theorem inversion_theorem (c1 c2 c3 : Circle) : 
  ∃ (i : InversionCircle), 
    (let i1 := imageCircle c1 i
     let i2 := imageCircle c2 i
     let i3 := imageCircle c3 i
     areCollinear i1.center i2.center i3.center ∧ 
     (i1.radius = i2.radius ∨ i1.radius = i3.radius ∨ i2.radius = i3.radius)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_theorem_l945_94504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sales_approx_l945_94505

/-- Represents the monthly growth rate for type Q electric vehicles -/
noncomputable def q_growth_rate : ℝ := 1.1

/-- Represents the monthly increase in units for type R electric vehicles -/
def r_monthly_increase : ℕ := 20

/-- Represents the initial sales (January) for both Q and R types -/
def initial_sales : ℕ := 50

/-- Calculates the sum of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Calculates the sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * a + (n * (n - 1) / 2) * d

/-- The main theorem stating the total sales of both types of electric vehicles -/
theorem total_sales_approx (months : ℕ) (months_eq_12 : months = 12) :
  let q_sales := geometric_sum (initial_sales : ℝ) q_growth_rate months
  let r_sales := arithmetic_sum initial_sales r_monthly_increase months
  ∃ (total : ℕ), (total ≥ 2965 ∧ total ≤ 2975) ∧ total = ⌊q_sales⌋ + r_sales := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sales_approx_l945_94505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_intersecting_line_equation_trajectory_equation_Q_l945_94576

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (2, 1)

-- Define point D
def point_D : ℝ × ℝ := (1, 2)

-- Define the distance |AB|
noncomputable def distance_AB : ℝ := 2 * Real.sqrt 3

-- Define the moving point M on circle C
def point_M (x₀ y₀ : ℝ) : Prop := circle_C x₀ y₀

-- Define vector ON
def vector_ON (y₀ : ℝ) : ℝ × ℝ := (0, y₀)

-- Define vector OQ as OM + ON
def vector_OQ (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, 2 * y₀)

-- Theorem 1: Tangent line equation
theorem tangent_line_equation :
  ∀ x y : ℝ, (x = 2 ∨ 3*x + 4*y - 10 = 0) ↔ 
  (∃ k : ℝ, y - point_P.2 = k * (x - point_P.1) ∧ 
   (∀ x' y' : ℝ, circle_C x' y' → (y' - point_P.2 = k * (x' - point_P.1) → x' = x ∧ y' = y))) :=
by sorry

-- Theorem 2: Line equation passing through D and intersecting C
theorem intersecting_line_equation :
  ∀ x y : ℝ, (3*x - 4*y + 5 = 0 ∨ x = 1) ↔ 
  (y - point_D.2 = (x - point_D.1) * (y - point_D.2) / (x - point_D.1) ∧
   ∃ A B : ℝ × ℝ, circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
   (A.2 - point_D.2) = (A.1 - point_D.1) * (y - point_D.2) / (x - point_D.1) ∧
   (B.2 - point_D.2) = (B.1 - point_D.1) * (y - point_D.2) / (x - point_D.1) ∧
   (A.1 - B.1)^2 + (A.2 - B.2)^2 = distance_AB^2) :=
by sorry

-- Theorem 3: Trajectory equation of point Q
theorem trajectory_equation_Q :
  ∀ x y : ℝ, x^2/4 + y^2/16 = 1 ↔ 
  (∃ x₀ y₀ : ℝ, point_M x₀ y₀ ∧ vector_OQ x₀ y₀ = (x, y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_intersecting_line_equation_trajectory_equation_Q_l945_94576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_app_user_growth_l945_94547

-- Define the user count function
noncomputable def A (k : ℝ) (t : ℝ) : ℝ := 500 * Real.exp (k * t)

-- State the theorem
theorem app_user_growth (k : ℝ) : 
  A k 10 = 2000 → -- Condition: After 10 days, there are 2000 users
  Real.log 2 / Real.log 10 = 0.30 → -- Given: lg 2 = 0.30
  (∃ t : ℝ, t ≥ 34 ∧ A k t > 50000) ∧ -- The user count exceeds 50000 after at least 34 days
  (∀ t : ℝ, t < 34 → A k t ≤ 50000) -- The user count does not exceed 50000 before 34 days
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_app_user_growth_l945_94547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_745_possible_l945_94503

/-- Represents the coin-changing machine that increases the total by 124 cents each use --/
def machine_increase (n : ℕ) : ℕ := n + 124

/-- Represents the possible amounts after using the machine, starting from 1 cent --/
def possible_amount (k : ℕ) : ℕ := 1 + 124 * k

/-- The list of amounts to check --/
def amount_list : List ℕ := [363, 513, 630, 745, 907]

/-- Theorem stating that 745 is the only amount in the list that can be reached --/
theorem only_745_possible : ∃ (k : ℕ), possible_amount k = 745 ∧ 
  ∀ (n : ℕ), n ∈ amount_list ∧ n ≠ 745 → ¬∃ (k : ℕ), possible_amount k = n := by
  sorry

#eval possible_amount 6  -- Should output 745

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_745_possible_l945_94503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l945_94559

noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 2*x + 5) / (x^2 - 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 2 ∧ x ≠ 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l945_94559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l945_94541

-- Define the circle
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the point A
def point_A : ℝ × ℝ := (1, 0)

-- Define a line passing through point A
def line_through_A (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - point_A.1) + point_A.2

-- Define tangency condition
def is_tangent (k : ℝ) : Prop :=
  ∃ x y, line_through_A k x y ∧ circle_C x y ∧
  ∀ x' y', line_through_A k x' y' → circle_C x' y' → (x', y') = (x, y)

-- Theorem statement
theorem tangent_line_equation :
  (∃ k, is_tangent k ∧ (∀ x y, line_through_A k x y ↔ 3*x - 4*y - 3 = 0)) ∨
  (is_tangent 0 ∧ (∀ x y, line_through_A 0 x y ↔ x = 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l945_94541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l945_94516

-- Define the function representing the inequality
noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x^2 - x - 6)

-- Define the solution set
def solution_set : Set ℝ := Set.Ioc (-2) 1 ∪ Set.Ioi 3

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l945_94516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l945_94570

/-- Predicate to check if n^k is the greatest power of n dividing m -/
def is_greatest_power_of_n_dividing (n k m : ℕ) : Prop :=
  n^k ∣ m ∧ ∀ l, l > k → ¬(n^l ∣ m)

theorem power_difference (a b : ℕ) : 
  is_greatest_power_of_n_dividing 2 a 180 →
  is_greatest_power_of_n_dividing 5 b 180 →
  (1/3 : ℚ)^(b - a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l945_94570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_difference_l945_94523

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ := a^3 / b

-- State the theorem
theorem otimes_difference : 
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = 32/9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_difference_l945_94523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l945_94501

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 / 2
def l (k : ℝ) (x : ℝ) : ℝ := (k - 3) * x - k + 2

theorem problem_solution :
  (∃ k : ℝ, (deriv f) (Real.exp 1) = k - 3 ∧ k = 5) ∧
  (∀ a : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f x₀ < g a x₀) ↔ a > 0) ∧
  (∃! k : ℤ, k = 5 ∧ ∀ x : ℝ, x > 1 → f x > l k x ∧
    ∀ m : ℤ, (∀ x : ℝ, x > 1 → f x > l m x) → m ≤ k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l945_94501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l945_94561

/-- The volume of the solid formed by rotating the region bounded by y = arcsin(x/5), 
    y = arcsin(x), and y = π/2 around the y-axis -/
noncomputable def rotationVolume : ℝ := 6 * Real.pi ^ 2

/-- The lower bound of the region -/
def lowerBound : ℝ := 0

/-- The upper bound of the region -/
noncomputable def upperBound : ℝ := Real.pi / 2

/-- The outer function defining the region -/
noncomputable def outerFunction (y : ℝ) : ℝ := 5 * Real.sin y

/-- The inner function defining the region -/
noncomputable def innerFunction (y : ℝ) : ℝ := Real.sin y

theorem volume_of_rotation : 
  Real.pi * ∫ y in lowerBound..upperBound, (outerFunction y)^2 - (innerFunction y)^2 = rotationVolume := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l945_94561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stevens_height_l945_94555

/-- Converts feet to centimeters -/
noncomputable def feet_to_cm (x : ℝ) : ℝ := 30.48 * x

/-- Converts inches to feet -/
noncomputable def inches_to_feet (x : ℝ) : ℝ := x / 12

theorem stevens_height (pole_height : ℝ) (pole_shadow : ℝ) (steven_shadow_inches : ℝ) :
  pole_height = 60 →
  pole_shadow = 20 →
  steven_shadow_inches = 25 →
  feet_to_cm ((pole_height / pole_shadow) * inches_to_feet steven_shadow_inches) = 190.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stevens_height_l945_94555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_evaluation_l945_94548

noncomputable def diamond (a b : ℝ) : ℝ := (a^2 + b^2) / (a + b)

theorem diamond_evaluation (w x y z : ℝ) 
  (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : w + x + y + z = 10) :
  diamond w (diamond x (diamond y z)) = (w^2 + x^2 + y^2 + z^2) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_evaluation_l945_94548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l945_94513

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + a*x + 2

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*x + a

theorem tangent_line_intersection (a : ℝ) : 
  (f a 0 = 2) ∧ 
  (f_deriv a 0 * (-2) + f a 0 = 0) → 
  a = 1 := by
  sorry

#check tangent_line_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l945_94513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l945_94587

/-- Represents a triangle in the sequence --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Generates the next triangle in the sequence --/
noncomputable def nextTriangle (t : Triangle) : Triangle :=
  { a := (t.b + t.c - t.a) / 2,
    b := (t.a + t.c - t.b) / 2,
    c := (t.a + t.b - t.c) / 2 }

/-- The initial triangle in the sequence --/
def T₁ : Triangle :=
  { a := 1001, b := 1002, c := 1003 }

/-- The sequence of triangles --/
noncomputable def triangleSequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => nextTriangle (triangleSequence n)

/-- The perimeter of a triangle --/
noncomputable def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Predicate to check if a triangle's circumcircle can be defined --/
def hasCircumcircle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧ t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The main theorem to be proved --/
theorem last_triangle_perimeter :
  ∃ n : ℕ, hasCircumcircle (triangleSequence n) ∧
    ¬hasCircumcircle (triangleSequence (n + 1)) ∧
    perimeter (triangleSequence n) = 1503 / 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l945_94587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_distance_theorem_l945_94575

/-- Represents a material point with mass and distance from a reference line -/
structure MaterialPoint where
  mass : ℝ
  distance : ℝ

/-- Calculates the distance of the center of mass from a reference line -/
noncomputable def centerOfMassDistance (points : List MaterialPoint) : ℝ :=
  let totalMass := points.foldl (λ sum p => sum + p.mass) 0
  let weightedSum := points.foldl (λ sum p => sum + p.mass * p.distance) 0
  weightedSum / totalMass

/-- Theorem: The distance of the center of mass from a reference line
    is the weighted average of the distances of individual points -/
theorem center_of_mass_distance_theorem (points : List MaterialPoint) :
  centerOfMassDistance points =
  (points.foldl (λ sum p => sum + p.mass * p.distance) 0) /
  (points.foldl (λ sum p => sum + p.mass) 0) := by
  sorry

#check center_of_mass_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_distance_theorem_l945_94575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l945_94584

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The vector a -/
noncomputable def a : ℝ × ℝ := (2 * Real.sin (4 * Real.pi / 3), Real.cos (5 * Real.pi / 6))

/-- The vector b -/
def b (k : ℝ) : ℝ × ℝ := (k, 1)

theorem parallel_vectors_k_value :
  ∀ k : ℝ, are_parallel a (b k) → k = 2 := by
  intro k h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l945_94584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l945_94518

/-- Given a triangle inscribed in a circle where the vertices divide the circle into three arcs of lengths 6, 8, and 10, the area of the triangle is 36(√3 + 2)/π² -/
theorem inscribed_triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let r := (a + b + c) / (2 * Real.pi)
  let A := c / (a + b + c) * 360
  let B := b / (a + b + c) * 360
  let C := a / (a + b + c) * 360
  let area := 1/2 * r^2 * (Real.sin (A * Real.pi / 180) + Real.sin (B * Real.pi / 180) + Real.sin (C * Real.pi / 180))
  area = 36 * (Real.sqrt 3 + 2) / Real.pi^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l945_94518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_divides_l945_94544

theorem prime_sum_divides (p : ℕ) (h_prime : Prime p) : 
  let k := (p - 1) * p / 2
  k ∣ ((Nat.factorial (p - 1)) - (p - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_divides_l945_94544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_18_seconds_l945_94553

/-- Represents the properties of a train and its movement across a platform and signal pole. -/
structure TrainCrossing where
  trainLength : ℝ
  platformLength : ℝ
  timeToCrossPlatform : ℝ
  
/-- Calculates the time taken for the train to cross a signal pole. -/
noncomputable def timeToCrossSignalPole (tc : TrainCrossing) : ℝ :=
  let totalDistance := tc.trainLength + tc.platformLength
  let speed := totalDistance / tc.timeToCrossPlatform
  tc.trainLength / speed

/-- Theorem stating that the time taken for the train to cross the signal pole is approximately 18 seconds. -/
theorem train_crossing_time_approx_18_seconds (tc : TrainCrossing) 
  (h1 : tc.trainLength = 300)
  (h2 : tc.platformLength = 400)
  (h3 : tc.timeToCrossPlatform = 42) :
  ∃ ε > 0, |timeToCrossSignalPole tc - 18| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_18_seconds_l945_94553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l945_94535

-- Define the square PQRS
def PQRS_area : ℝ := 16

-- Define the side length of smaller squares
def small_square_side : ℝ := 2

-- Define the triangle DEF
structure Triangle_DEF where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define the center point T of square PQRS
def T : ℝ × ℝ := (2, 2)

-- Define a function to calculate the area of a triangle
def area_triangle (t : Triangle_DEF) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_DEF (DEF : Triangle_DEF) :
  -- Conditions
  PQRS_area = 16 →
  small_square_side = 2 →
  DEF.E.1 = DEF.F.1 →  -- E and F have same x-coordinate (EF is vertical)
  DEF.D.1 = DEF.D.2 →  -- DE = DF (isosceles)
  DEF.D = T →  -- D aligns with T when folded
  -- Conclusion
  area_triangle DEF = 2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l945_94535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_terms_l945_94558

noncomputable def arithmeticSequenceSum (a₁ : ℝ) (aₙ : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a₁ + aₙ)

theorem arithmetic_sequence_sum_10_terms :
  let a₁ : ℝ := -3
  let aₙ : ℝ := 60
  let n : ℕ := 10
  arithmeticSequenceSum a₁ aₙ n = 285 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_terms_l945_94558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_theorem_l945_94580

noncomputable def is_arithmetic_sequence (s : List ℝ) : Prop :=
  s.length > 1 ∧ ∀ i, i < s.length - 1 → s.get! (i+1) - s.get! i = s.get! 1 - s.get! 0

noncomputable def is_geometric_sequence (s : List ℝ) : Prop :=
  s.length > 1 ∧ ∀ i, i < s.length - 1 → s.get! (i+1) / s.get! i = s.get! 1 / s.get! 0

theorem sequence_ratio_theorem (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  (is_arithmetic_sequence [-7, a₁, a₂, -1]) →
  (is_geometric_sequence [-4, b₁, b₂, b₃, -1]) →
  (a₂ - a₁) / b₂ = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_theorem_l945_94580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_distance_from_start_l945_94598

/-- Conversion factor from meters to feet -/
def meters_to_feet : ℝ := 3.28084

/-- Alice's northward distance in meters -/
def north_distance : ℝ := 30

/-- Alice's eastward distance in feet -/
def east_distance : ℝ := 40

/-- Alice's southward distance in meters -/
def south_distance_meters : ℝ := 15

/-- Alice's southward distance in feet -/
def south_distance_feet : ℝ := 50

/-- Theorem stating Alice's approximate distance from starting point -/
theorem alice_distance_from_start : 
  let north_feet := north_distance * meters_to_feet
  let south_feet := south_distance_meters * meters_to_feet + south_distance_feet
  let net_south := south_feet - north_feet
  let distance := Real.sqrt (east_distance^2 + net_south^2)
  ∃ ε > 0, |distance - 40| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_distance_from_start_l945_94598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_in_zero_one_l945_94574

open Real

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (x + 1) + a * (x^2 - x)

/-- The theorem statement -/
theorem f_nonnegative_implies_a_in_zero_one :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x ≥ 0) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_in_zero_one_l945_94574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_negative_two_implies_expression_l945_94519

theorem tan_negative_two_implies_expression (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_negative_two_implies_expression_l945_94519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_implies_k_value_l945_94522

/-- Given a function f(x) = sin(2x) + k*cos(2x) with an axis of symmetry at x = π/6,
    prove that k = √3/3 -/
theorem symmetry_axis_implies_k_value (k : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x : ℝ, f x = Real.sin (2 * x) + k * Real.cos (2 * x)) →
  (∃ f : ℝ → ℝ, ∀ x : ℝ, f (π / 6 + x) = f (π / 6 - x)) →
  k = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_implies_k_value_l945_94522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_destruction_probability_l945_94500

/-- The probability of at least two events occurring out of three independent events --/
def prob_at_least_two (p1 p2 p3 : ℝ) : ℝ :=
  1 - ((1 - p1) * (1 - p2) * (1 - p3) + 
       p1 * (1 - p2) * (1 - p3) + 
       (1 - p1) * p2 * (1 - p3) + 
       (1 - p1) * (1 - p2) * p3)

theorem target_destruction_probability 
  (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.9) 
  (h2 : p2 = 0.9) 
  (h3 : p3 = 0.8) : 
  prob_at_least_two p1 p2 p3 = 0.954 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_destruction_probability_l945_94500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_l945_94515

/-- Represents the grid with obstacles -/
def Grid : Type := Fin 5 → Fin 10 → Bool

/-- Check if a cell is an obstacle -/
def is_obstacle (g : Grid) (i j : ℕ) : Bool :=
  if h1 : i < 5 ∧ j < 10 then
    g ⟨i, h1.left⟩ ⟨j, h1.right⟩
  else
    true

/-- Count the number of paths in the grid -/
def count_paths (g : Grid) : ℕ :=
  let rec aux (i j : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then 0
    else if i = 4 ∧ j = 9 then 1
    else if i ≥ 5 ∨ j ≥ 10 ∨ is_obstacle g i j then 0
    else aux (i+1) j (fuel-1) + aux i (j+1) (fuel-1) + aux (i+1) (j+1) (fuel-1)
  aux 0 0 100  -- Set an initial fuel value that's large enough

/-- The theorem to be proved -/
theorem path_count (g : Grid) : count_paths g = 38 := by
  sorry

#eval count_paths (λ _ _ => false)  -- Test with no obstacles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_l945_94515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l945_94597

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := x^2 + y^2 + z^2 + 2*y = 0

def equation2 (a x y z : ℝ) : Prop := x + a*y + a*z - a = 0

-- Define the condition for a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! (p : ℝ × ℝ × ℝ), equation1 p.1 p.2.1 p.2.2 ∧ equation2 a p.1 p.2.1 p.2.2

-- State the theorem
theorem unique_solution_condition :
  ∀ a : ℝ, has_unique_solution a ↔ a = Real.sqrt 2 / 2 ∨ a = -Real.sqrt 2 / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l945_94597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l945_94532

/-- Proves that a man traveling 425.034 meters in 30 seconds has a speed of approximately 51 km/h -/
theorem man_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 425.034 ∧ 
  time = 30 ∧ 
  speed = (distance / 1000) / (time / 3600) →
  ‖speed - 51‖ < 0.1 := by
  sorry

#eval (425.034 / 1000) / (30 / 3600)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l945_94532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_correct_l945_94528

/-- The plane we're looking for -/
def target_plane (x y z : ℝ) : Prop := 3*x + y - 2*z = 0

/-- A point that the plane passes through -/
def point1 : ℝ × ℝ × ℝ := (0, 0, 0)

/-- Another point that the plane passes through -/
def point2 : ℝ × ℝ × ℝ := (2, 2, 4)

/-- The plane that our target plane is perpendicular to -/
def perp_plane (x y z : ℝ) : Prop := x - y + z = 7

theorem plane_equation_correct : 
  (∀ x y z, target_plane x y z ↔ 
    -- The plane passes through point1
    target_plane point1.1 point1.2.1 point1.2.2 ∧ 
    -- The plane passes through point2
    target_plane point2.1 point2.2.1 point2.2.2 ∧ 
    -- The plane is perpendicular to perp_plane
    (∀ a b c d e f, target_plane a b c ∧ target_plane d e f →
      perp_plane a b c → perp_plane d e f → 
      (d - a) * 1 + (e - b) * (-1) + (f - c) * 1 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_correct_l945_94528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_fraction_difference_l945_94595

theorem floor_fraction_difference (n : ℝ) (h : n = 2009) : 
  ⌊(n + 1)^2 / ((n - 1) * n) - (n - 1)^2 / (n * (n + 1))⌋ = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_fraction_difference_l945_94595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l945_94573

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x) ^ 2

-- State the theorem
theorem f_properties :
  -- The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- The minimum value in [-π/3, π/3] is 0
  (∀ (x : ℝ), -Real.pi/3 ≤ x ∧ x ≤ Real.pi/3 → f x ≥ 0) ∧
  (∃ (x : ℝ), -Real.pi/3 ≤ x ∧ x ≤ Real.pi/3 ∧ f x = 0) ∧
  -- The maximum value in [-π/3, π/3] is 2 + √3
  (∀ (x : ℝ), -Real.pi/3 ≤ x ∧ x ≤ Real.pi/3 → f x ≤ 2 + Real.sqrt 3) ∧
  (∃ (x : ℝ), -Real.pi/3 ≤ x ∧ x ≤ Real.pi/3 ∧ f x = 2 + Real.sqrt 3) :=
by
  sorry  -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l945_94573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l945_94579

theorem size_relationship : ∀ (a b c : ℝ), 
  a = (6/10)^2 → b = Real.log (6/10) → c = 2^(6/10) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l945_94579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_probability_l945_94585

/-- The probability that at least 8 out of 9 people stay for an entire concert, 
    given that 5 have a 3/7 chance of staying and 4 are certain to stay. -/
theorem concert_probability : ℚ := by
  let totalAttendees : ℕ := 9
  let uncertainAttendees : ℕ := 5
  let certainAttendees : ℕ := 4
  let stayProbability : ℚ := 3/7

  have h1 : totalAttendees = uncertainAttendees + certainAttendees := by rfl
  have h2 : certainAttendees > 0 := by norm_num
  have h3 : uncertainAttendees > 0 := by norm_num
  have h4 : stayProbability > 0 ∧ stayProbability < 1 := by
    constructor
    · norm_num
    · norm_num

  exact 2511/16807

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_probability_l945_94585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l945_94565

/-- The inclination angle of a line --/
noncomputable def inclination_angle (a b c : ℝ) : ℝ :=
  Real.arctan (-a / b)

/-- Converts degrees to radians --/
noncomputable def degrees_to_radians (degrees : ℝ) : ℝ :=
  degrees * (Real.pi / 180)

theorem line_inclination_angle :
  inclination_angle 1 (Real.sqrt 3) (-5) = degrees_to_radians 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l945_94565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exclusive_p_q_implies_a_range_l945_94582

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2*x + a) / Real.log 0.5

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (-(5-2*a))^x₁ > (-(5-2*a))^x₂

-- State the theorem
theorem exclusive_p_q_implies_a_range :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → 1 < a ∧ a < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exclusive_p_q_implies_a_range_l945_94582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_value_at_negative_two_final_result_l945_94581

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f x - f (f y + f (-x)) + x

/-- The theorem stating that f(-2) = 2 and it's the only possible value -/
theorem unique_value_at_negative_two :
  ∀ f : ℝ → ℝ, FunctionalEquation f → f (-2) = 2 ∧ ∀ y : ℝ, f (-2) = y → y = 2 := by
  sorry

/-- The final result: the product of the number of possible values and their sum -/
theorem final_result :
  ∃ n s : ℕ, n = 1 ∧ s = 2 ∧ n * s = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_value_at_negative_two_final_result_l945_94581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l945_94520

-- Define a 2D vector type
def Vector2D := ℝ × ℝ

-- Define dot product for Vector2D
def dot_product (a b : Vector2D) : ℝ := a.1 * b.1 + a.2 * b.2

-- Define vector norm (magnitude) for Vector2D
noncomputable def vector_norm (a : Vector2D) : ℝ := Real.sqrt (a.1 * a.1 + a.2 * a.2)

-- Define vector addition for Vector2D
def vector_add (a b : Vector2D) : Vector2D := (a.1 + b.1, a.2 + b.2)

-- Define vector subtraction for Vector2D
def vector_sub (a b : Vector2D) : Vector2D := (a.1 - b.1, a.2 - b.2)

-- Define scalar multiplication for Vector2D
def scalar_mult (r : ℝ) (a : Vector2D) : Vector2D := (r * a.1, r * a.2)

theorem vector_properties (a b : Vector2D) (h1 : a ≠ (0, 0)) (h2 : b ≠ (0, 0)) :
  (dot_product a b = 0 → vector_norm (vector_add a b) = vector_norm (vector_sub a b)) ∧
  (vector_norm (vector_add a b) = vector_norm a - vector_norm b → ∃ r : ℝ, a = scalar_mult r b) :=
by
  sorry

#check vector_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l945_94520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l945_94588

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := otimes (Real.sin x) (Real.cos x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), M = Real.sqrt 2 / 2 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l945_94588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_fraction_proof_l945_94545

noncomputable def total_journey : ℝ := 130
noncomputable def rail_fraction : ℝ := 3/5
noncomputable def foot_distance : ℝ := 6.5

theorem bus_fraction_proof :
  let rail_distance : ℝ := rail_fraction * total_journey
  let bus_distance : ℝ := total_journey - (rail_distance + foot_distance)
  bus_distance / total_journey = 45.5 / 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_fraction_proof_l945_94545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_eq_face_area_l945_94517

/-- The area of an equilateral triangle with side length 1 -/
noncomputable def equilateralTriangleArea : ℝ := Real.sqrt 3 / 4

/-- A tetrahedron with two adjacent equilateral triangular faces -/
structure Tetrahedron where
  faceArea : ℝ
  dihedralAngle : ℝ

/-- The maximum projection area of the rotating tetrahedron -/
def maxProjectionArea (t : Tetrahedron) : ℝ := t.faceArea

/-- Theorem: The maximum projection area of the rotating tetrahedron is equal to the area of one face -/
theorem max_projection_area_eq_face_area (t : Tetrahedron) 
  (h1 : t.faceArea = equilateralTriangleArea) 
  (h2 : t.dihedralAngle = π/4) : 
  maxProjectionArea t = t.faceArea := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_eq_face_area_l945_94517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_l945_94539

def b (n : ℕ) : ℤ := n - 35

def a (n : ℕ) : ℕ := 2^n

def f (n : ℕ) : ℚ := (b n : ℚ) / (a n)

theorem max_value_of_sequence :
  ∃ (k : ℕ), ∀ (n : ℕ), n ≥ 1 → f n ≤ f k ∧ f k = 1 / 2^36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_l945_94539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l945_94586

/-- Line l with parametric equation x = t, y = t - 1 -/
def line_l (t : ℝ) : ℝ × ℝ := (t, t - 1)

/-- Curve C with polar equation ρsin²θ - 4cosθ = 0 -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := 
  let ρ := 4 * Real.cos θ / (Real.sin θ)^2
  (ρ * Real.cos θ, ρ * Real.sin θ)

/-- Theorem: If line l intersects curve C at two points A and B, then |AB| = 8 -/
theorem intersection_distance : 
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
  (∃ (θ₁ θ₂ : ℝ), 0 ≤ θ₁ ∧ θ₁ < 2*Real.pi ∧ 0 ≤ θ₂ ∧ θ₂ < 2*Real.pi ∧
  line_l t₁ = curve_C θ₁ ∧ line_l t₂ = curve_C θ₂) →
  Real.sqrt ((t₁ - t₂)^2) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l945_94586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_theorem_l945_94567

-- Define the polynomial type
def IntPolynomial (b₂ b₁ : ℤ) : ℤ → ℤ := λ x => x^3 + b₂*x^2 + b₁*x - 13

-- Define the set of possible integer roots
def PossibleRoots : Set ℤ := {-13, -1, 1, 13}

-- Theorem statement
theorem integer_root_theorem (b₂ b₁ : ℤ) :
  ∀ x : ℤ, (IntPolynomial b₂ b₁ x = 0) → x ∈ PossibleRoots :=
by
  sorry

#check integer_root_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_theorem_l945_94567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_weight_theorem_l945_94502

/-- Represents the weight of a side of beef before and after processing -/
structure BeefWeight where
  initial : ℝ
  after_initial : ℝ
  rib : ℝ
  short_loin : ℝ
  sirloin : ℝ

/-- Calculates the weight of a side of beef after processing -/
def process_beef (w : BeefWeight) : Prop :=
  w.after_initial = 0.75 * w.initial ∧
  w.rib = 120 ∧
  w.short_loin = 144 ∧
  w.sirloin = 180 ∧
  w.after_initial = (w.rib / 0.85) + (w.short_loin / 0.80) + (w.sirloin / 0.75)

/-- Theorem stating that the initial weight of the side of beef is approximately 748.235 pounds -/
theorem beef_weight_theorem (w : BeefWeight) (h : process_beef w) : 
  ‖w.initial - 748.235‖ < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_weight_theorem_l945_94502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_remaining_scores_l945_94527

noncomputable def scores : List ℝ := [90, 89, 90, 95, 93, 94, 93]

noncomputable def remaining_scores : List ℝ := [90, 90, 93, 94, 93]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

theorem variance_of_remaining_scores :
  variance remaining_scores = 2.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_remaining_scores_l945_94527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_side_length_l945_94540

theorem triangle_DEF_side_length (D E F : ℝ) 
  (h1 : Real.cos (2*D - E) + Real.sin (D + E) = 2) 
  (h2 : 0 < D) (h3 : D < π) (h4 : 0 < E) (h5 : E < π) 
  (h6 : 0 < F) (h7 : F < π) (h8 : D + E + F = π) 
  (h9 : (6 : ℝ) = 6) : 
  ∃ (EF : ℝ), EF = 3 ∧ EF = (6 : ℝ) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_side_length_l945_94540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_speed_l945_94566

/-- Calculates the average speed of a two-segment trip -/
noncomputable def average_speed (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) : ℝ :=
  total_distance / ((distance1 / speed1) + (distance2 / speed2))

theorem trip_average_speed :
  let total_distance : ℝ := 50
  let distance1 : ℝ := 25
  let speed1 : ℝ := 60
  let distance2 : ℝ := 25
  let speed2 : ℝ := 30
  average_speed total_distance distance1 speed1 distance2 speed2 = 40 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval average_speed 50 25 60 25 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_speed_l945_94566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l945_94568

noncomputable def reward_function (x a : ℝ) : ℝ := (10 * x - 3 * a) / (x + 2)

theorem minimum_a_value :
  ∃ (a : ℕ), (∀ x : ℝ, 10 ≤ x ∧ x ≤ 1000 →
    (reward_function x (a : ℝ) ≤ 9 ∧
     reward_function x (a : ℝ) ≤ x / 5 ∧
     reward_function x (a : ℝ) > 0)) ∧
  (∀ b : ℕ, b < a →
    ∃ x : ℝ, 10 ≤ x ∧ x ≤ 1000 ∧
      (reward_function x (b : ℝ) > 9 ∨
       reward_function x (b : ℝ) > x / 5 ∨
       reward_function x (b : ℝ) ≤ 0)) ∧
  a = 328 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l945_94568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_cubes_of_roots_special_l945_94589

/-- A monic polynomial of degree n with real coefficients -/
structure MonicPolynomial (n : ℕ) where
  coeffs : Fin n → ℝ
  monic : coeffs ⟨n - 1, sorry⟩ = 1

/-- The roots of a monic polynomial -/
noncomputable def roots (p : MonicPolynomial n) : Finset ℝ := sorry

/-- The sum of the cubes of the roots of a monic polynomial -/
noncomputable def sumCubesOfRoots (p : MonicPolynomial n) : ℝ :=
  (roots p).sum (fun r => r^3)

/-- Theorem: Sum of cubes of roots for a specific type of monic polynomial -/
theorem sum_cubes_of_roots_special (n : ℕ) (a : ℝ) :
  ∀ (p : MonicPolynomial n),
  (∀ i : Fin n, i.val = n - 1 → p.coeffs i = -a) →
  (∀ i : Fin n, i.val = n - 2 → p.coeffs i = -a) →
  sumCubesOfRoots p = 4 * a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_cubes_of_roots_special_l945_94589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triominoes_6x6_grid_l945_94563

/-- Represents an L-shaped triomino --/
structure Triomino :=
  (x : Fin 6)
  (y : Fin 6)

/-- Check if two triominoes overlap --/
def overlap (t1 t2 : Triomino) : Prop :=
  sorry

/-- Check if a triomino can be placed in the grid --/
def can_place (grid : List Triomino) (t : Triomino) : Prop :=
  sorry

/-- The main theorem --/
theorem min_triominoes_6x6_grid :
  ∃ (grid : List Triomino),
    grid.length = 6 ∧
    (∀ t1 t2, t1 ∈ grid → t2 ∈ grid → t1 ≠ t2 → ¬overlap t1 t2) ∧
    (∀ t : Triomino, ¬can_place grid t) ∧
    (∀ grid' : List Triomino,
      grid'.length < 6 →
      (∀ t1 t2, t1 ∈ grid' → t2 ∈ grid' → t1 ≠ t2 → ¬overlap t1 t2) →
      ∃ t : Triomino, can_place grid' t) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triominoes_6x6_grid_l945_94563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_waiting_time_l945_94592

/-- Proves that the waiting time for the cyclist is 18.75 minutes -/
theorem cyclist_waiting_time
  (hiker_speed : ℝ)
  (cyclist_speed : ℝ)
  (waiting_start_time : ℝ)
  (h1 : hiker_speed = 4)
  (h2 : cyclist_speed = 15)
  (h3 : waiting_start_time = 5)
  : ℝ :=
by
  -- Convert speeds to miles per minute
  let hiker_speed_per_minute := hiker_speed / 60
  let cyclist_speed_per_minute := cyclist_speed / 60

  -- Calculate distance traveled by cyclist before stopping
  let cyclist_distance := cyclist_speed_per_minute * waiting_start_time

  -- Calculate time for hiker to cover the distance
  let waiting_time := cyclist_distance / hiker_speed_per_minute

  -- Prove that the waiting time is 18.75 minutes
  sorry

-- Example usage (commented out to avoid evaluation errors)
-- #eval cyclist_waiting_time 4 15 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_waiting_time_l945_94592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cab_driver_average_income_l945_94546

noncomputable def daily_incomes : List ℝ := [250, 400, 750, 400, 500]

def num_days : ℕ := 5

noncomputable def average_income : ℝ := (daily_incomes.sum) / num_days

theorem cab_driver_average_income :
  average_income = 460 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cab_driver_average_income_l945_94546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_turn_off_all_lights_l945_94529

/-- Represents the state of a light (on or off) -/
inductive LightState
| On
| Off

/-- Represents a 5x5 grid of lights -/
def Grid := Fin 5 → Fin 5 → LightState

/-- Initial state of the grid with all lights on -/
def initialGrid : Grid := λ _ _ ↦ LightState.On

/-- Toggles a light state -/
def toggleLight : LightState → LightState
| LightState.On => LightState.Off
| LightState.Off => LightState.On

/-- Applies an operation to the grid, toggling lights in the specified row and column -/
def applyOperation (g : Grid) (row col : Fin 5) : Grid :=
  λ i j ↦ if i = row ∨ j = col then toggleLight (g i j) else g i j

/-- Checks if all lights in the grid are off -/
def allLightsOff (g : Grid) : Prop :=
  ∀ i j, g i j = LightState.Off

/-- The main theorem stating it's impossible to turn off all lights -/
theorem impossible_to_turn_off_all_lights :
  ¬∃ (operations : List (Fin 5 × Fin 5)),
    allLightsOff (operations.foldl (λ g (row, col) ↦ applyOperation g row col) initialGrid) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_turn_off_all_lights_l945_94529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_line_proof_l945_94549

/-- Circle C with center (1, 1) and radius 4 -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 16

/-- Line l parameterized by m -/
def line_l (x y m : ℝ) : Prop := (2*m - 1)*x + (m - 1)*y - 3*m + 1 = 0

/-- The equation of line l when it intercepts the minimum chord length on circle C -/
def min_chord_line (x y : ℝ) : Prop := x - 2*y - 4 = 0

theorem min_chord_line_proof :
  ∀ x y : ℝ, circle_C x y → (∃ m : ℝ, line_l x y m) →
  (∀ m : ℝ, line_l x y m → (∃ x' y' : ℝ, circle_C x' y' ∧ line_l x' y' m)) →
  min_chord_line x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_line_proof_l945_94549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_points_l945_94571

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_line_parallel_points :
  ∃ (a b : ℝ), 
    (a = 1 ∧ b = 0 ∨ a = -1 ∧ b = -4) ∧
    f a = b ∧
    (∀ x : ℝ, x ≠ a → (f x - f a) / (x - a) ≠ 4) ∧
    (∃ δ : ℝ, δ > 0 ∧ 
      ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < δ → 
        ∀ k : ℝ, k ≠ 4 → 
          |(f x - f a) / (x - a) - 4| < |(f x - f a) / (x - a) - k|) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_points_l945_94571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_value_l945_94525

noncomputable section

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the distance between foci
def foci_distance (c : ℝ) : Prop := c = 1

-- Define the perimeter of the triangle
def triangle_perimeter (a c : ℝ) : Prop := 2*a + 2*c = 6

-- Define point P on the ellipse
def point_p : ℝ × ℝ := (2, 1)

-- Define the line l
def line_l (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Define the midpoint condition
def midpoint_condition (k m : ℝ) : Prop :=
  -4 * k * m / (3 + 4 * k^2) = -2 ∧ 3 * m / (3 + 4 * k^2) = 1/2

-- Define the expression to be maximized
def max_expression (m : ℝ) : ℝ :=
  -(3/4) * (m + 4/3)^2 + 52/3

-- Main theorem
theorem ellipse_max_value
  (a b c : ℝ)
  (h_ellipse : ellipse a b 2 1)
  (h_foci : foci_distance c)
  (h_perimeter : triangle_perimeter a c)
  (k m : ℝ)
  (h_midpoint : midpoint_condition k m) :
  ∃ (max_val : ℝ), max_val = 52/3 ∧
  ∀ (m : ℝ), max_expression m ≤ max_val :=
by
  -- The proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_value_l945_94525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_max_speed_l945_94521

-- Define the problem parameters
noncomputable def distance : ℝ := 130
noncomputable def gas_price : ℝ := 2
noncomputable def driver_wage : ℝ := 14

-- Define the cost function
noncomputable def cost_function (x : ℝ) : ℝ :=
  (1820 * x^2 + 520 * x + 260) / x^2

-- State the theorem
theorem min_cost_at_max_speed :
  ∀ x ∈ Set.Icc 50 100,
    cost_function x ≥ cost_function 100 ∧
    cost_function 100 = 1872.26 := by
  sorry

-- Note: Set.Icc 50 100 represents the closed interval [50, 100]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_max_speed_l945_94521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_theorem_l945_94572

structure WaterPricing where
  tier1_limit : ℝ
  tier2_limit : ℝ
  tier1_water_price : ℝ
  tier2_water_price : ℝ
  tier3_water_price : ℝ
  sewage_price : ℝ

noncomputable def calculate_bill (pricing : WaterPricing) (usage : ℝ) : ℝ :=
  if usage ≤ pricing.tier1_limit then
    usage * (pricing.tier1_water_price + pricing.sewage_price)
  else if usage ≤ pricing.tier2_limit then
    pricing.tier1_limit * (pricing.tier1_water_price + pricing.sewage_price) +
    (usage - pricing.tier1_limit) * (pricing.tier2_water_price + pricing.sewage_price)
  else
    pricing.tier1_limit * (pricing.tier1_water_price + pricing.sewage_price) +
    (pricing.tier2_limit - pricing.tier1_limit) * (pricing.tier2_water_price + pricing.sewage_price) +
    (usage - pricing.tier2_limit) * (pricing.tier3_water_price + pricing.sewage_price)

theorem water_pricing_theorem (pricing : WaterPricing) 
  (h1 : pricing.tier1_limit = 17)
  (h2 : pricing.tier2_limit = 30)
  (h3 : pricing.tier3_water_price = 6)
  (h4 : pricing.sewage_price = 0.8)
  (h5 : calculate_bill pricing 15 = 45)
  (h6 : calculate_bill pricing 25 = 91) :
  pricing.tier1_water_price = 2.2 ∧ 
  pricing.tier2_water_price = 4.2 ∧
  calculate_bill pricing 35 = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_theorem_l945_94572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_distribution_theorem_l945_94557

/-- Represents the fair distribution of compensation for road construction -/
noncomputable def fair_distribution (total_length : ℝ) (anatoly_length : ℝ) (vladimir_length : ℝ) (boris_contribution : ℝ) :
  (ℝ × ℝ) :=
  let fair_share := total_length / 3
  let anatoly_excess := anatoly_length - fair_share
  let vladimir_excess := vladimir_length - fair_share
  let anatoly_ratio := anatoly_excess / (anatoly_excess + vladimir_excess)
  let vladimir_ratio := vladimir_excess / (anatoly_excess + vladimir_excess)
  (anatoly_ratio * boris_contribution, vladimir_ratio * boris_contribution)

theorem fair_distribution_theorem (total_length anatoly_length vladimir_length boris_contribution : ℝ) :
  total_length = 16 ∧ 
  anatoly_length = 6 ∧ 
  vladimir_length = 10 ∧ 
  boris_contribution = 16 →
  fair_distribution total_length anatoly_length vladimir_length boris_contribution = (2, 14) :=
by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_distribution_theorem_l945_94557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_dilution_l945_94583

theorem sugar_solution_dilution (initial_volume : ℝ) (added_water : ℝ) (final_percentage : ℝ) :
  initial_volume = 3 →
  added_water = 1 →
  final_percentage = 30.000000000000004 →
  let final_volume := initial_volume + added_water
  let initial_percentage := final_percentage * (final_volume / initial_volume)
  initial_percentage = 40.000000000000005 := by
  intro h1 h2 h3
  simp [h1, h2, h3]
  norm_num
  sorry

#check sugar_solution_dilution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_dilution_l945_94583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_discount_rate_approximately_14_55_l945_94596

-- Define the items and their prices
noncomputable def bag_marked : ℝ := 80
noncomputable def bag_sold : ℝ := 68
noncomputable def shoes_marked : ℝ := 120
noncomputable def shoes_sold : ℝ := 96
noncomputable def jacket_marked : ℝ := 150
noncomputable def jacket_sold : ℝ := 135
noncomputable def hat_marked : ℝ := 40
noncomputable def hat_sold : ℝ := 32
noncomputable def scarf_marked : ℝ := 50
noncomputable def scarf_sold : ℝ := 45

-- Define the total marked price and total discount
noncomputable def total_marked : ℝ := bag_marked + shoes_marked + jacket_marked + hat_marked + scarf_marked
noncomputable def total_discount : ℝ := (bag_marked - bag_sold) + (shoes_marked - shoes_sold) + (jacket_marked - jacket_sold) + (hat_marked - hat_sold) + (scarf_marked - scarf_sold)

-- Define the average discount rate
noncomputable def average_discount_rate : ℝ := (total_discount / total_marked) * 100

-- Theorem statement
theorem average_discount_rate_approximately_14_55 :
  ∃ ε > 0, |average_discount_rate - 14.55| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_discount_rate_approximately_14_55_l945_94596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_value_l945_94537

theorem unique_function_value (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > 0 → y > 0 → f x * f y = f (x * y) + 2023 * (1 / x + 1 / y + 2022)) :
  ∃! v : ℝ, f 2 = v ∧ v = 4047 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_value_l945_94537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_plus_DE_equals_120_l945_94564

-- Define the triangle and points
variable (A B C D E : EuclideanSpace ℝ 2)

-- Define the conditions
axiom triangle_ABC : ¬Collinear A B C
axiom D_on_BC : Collinear B C D
axiom E_on_AC : Collinear A C E
axiom AD_length : dist A D = 60
axiom BD_length : dist B D = 189
axiom CD_length : dist C D = 36
axiom AE_length : dist A E = 40
axiom CE_length : dist C E = 50

-- State the theorem
theorem AB_plus_DE_equals_120 :
  dist A B + dist D E = 120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_plus_DE_equals_120_l945_94564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_midpoint_distances_l945_94562

/-- Given a square with side length 4, the sum of distances from one vertex
    to the midpoints of all sides is 4 + 4√5 -/
theorem square_midpoint_distances (S : Set (ℝ × ℝ)) :
  (∀ (a b : ℝ × ℝ), a ∈ S ∧ b ∈ S → dist a b = 4 ∨ dist a b = 4 * Real.sqrt 2) →
  (∃ (v m₁ m₂ m₃ m₄ : ℝ × ℝ),
    v ∈ S ∧ m₁ ∈ S ∧ m₂ ∈ S ∧ m₃ ∈ S ∧ m₄ ∈ S ∧
    (∀ (x : ℝ × ℝ), x ∈ S → dist x m₁ ≤ 2 ∧ dist x m₂ ≤ 2 ∧ dist x m₃ ≤ 2 ∧ dist x m₄ ≤ 2)) →
  ∃ (v m₁ m₂ m₃ m₄ : ℝ × ℝ),
    v ∈ S ∧ m₁ ∈ S ∧ m₂ ∈ S ∧ m₃ ∈ S ∧ m₄ ∈ S ∧
    dist v m₁ + dist v m₂ + dist v m₃ + dist v m₄ = 4 + 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_midpoint_distances_l945_94562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixture_l945_94551

/-- Represents the volume of a 40% salt solution in liters -/
def x : ℝ := 1

/-- The concentration of salt in the original solution -/
def original_concentration : ℝ := 0.40

/-- The concentration of salt in the final mixture -/
def final_concentration : ℝ := 0.20

/-- The volume of pure water added in liters -/
def pure_water_volume : ℝ := 1

/-- Theorem stating that x equals 1 when mixing 1 liter of pure water with x liters of 40% salt solution to create a 20% salt solution -/
theorem salt_solution_mixture : x = 1 := by
  -- The proof goes here
  sorry

#eval x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixture_l945_94551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_spade_club_diamond_l945_94599

/-- The probability of drawing a spade, then a club, then a diamond from a standard 52-card deck -/
theorem probability_spade_club_diamond : ∃ (p : ℚ), p = 2197 / 132600 := by
  let deck_size : ℕ := 52
  let suit_size : ℕ := 13
  let prob : ℚ := (suit_size : ℚ) / deck_size * 
                  (suit_size : ℚ) / (deck_size - 1) * 
                  (suit_size : ℚ) / (deck_size - 2)
  use prob
  sorry  -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_spade_club_diamond_l945_94599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l945_94508

/-- The number of intersection points between a line and a circle --/
noncomputable def intersectionPoints (a b c d e f : ℝ) : ℕ :=
  let discriminant := (2*a*d + 2*b*e)^2 - 4*(a^2 + b^2)*((d^2 + e^2) - f*(a^2 + b^2))
  if discriminant > 0 then 2
  else if discriminant = 0 then 1
  else 0

/-- Theorem: The line 3x + 4y = 6 and the circle x^2 + y^2 = 9 intersect at 2 points --/
theorem line_circle_intersection :
  intersectionPoints 3 4 (-6) 1 1 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l945_94508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_formula_l945_94524

/-- A function satisfying the given property -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = x * f y + y * f x

/-- The sequence defined by the special function -/
def SpecialSequence (f : ℝ → ℝ) (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n = f ((3 : ℝ) ^ (n : ℝ))

theorem special_sequence_formula 
  (f : ℝ → ℝ) 
  (a : ℕ+ → ℝ) 
  (hf : SpecialFunction f) 
  (ha : SpecialSequence f a) 
  (h1 : a 1 = 3) 
  (h_nonzero : ∃ x, f x ≠ 0) :
  ∀ n : ℕ+, a n = n * (3 : ℝ) ^ (n : ℝ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_formula_l945_94524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l945_94577

theorem train_speed_problem (distance : ℝ) (outbound_speed : ℝ) (total_time : ℝ) : 
  distance = 120 →
  outbound_speed = 40 →
  total_time = 5.4 →
  let outbound_time := distance / outbound_speed
  let return_time := total_time - outbound_time
  let return_speed := distance / return_time
  return_speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l945_94577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_zeros_in_five_over_two_thousand_five_hundred_twenty_l945_94512

noncomputable def decimalExpansion (q : ℚ) : ℕ → ℕ :=
  sorry

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : d ≠ 0) :
  let f := n / d
  let decimal_expansion := decimalExpansion (n / d)
  (∃ k : ℕ, decimal_expansion k ≠ 0 ∧
    (∀ j < k, decimal_expansion j = 0)) →
  (∃ k : ℕ, decimal_expansion k ≠ 0 ∧
    (∀ j < k, decimal_expansion j = 0) ∧
    k = 2) :=
by
  sorry

theorem zeros_in_five_over_two_thousand_five_hundred_twenty :
  let f := 5 / 2520
  let decimal_expansion := decimalExpansion (5 / 2520)
  ∃ k : ℕ, decimal_expansion k ≠ 0 ∧
    (∀ j < k, decimal_expansion j = 0) ∧
    k = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_zeros_in_five_over_two_thousand_five_hundred_twenty_l945_94512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_binomial_expansion_l945_94538

theorem fifth_term_binomial_expansion 
  (a x : ℝ) (h : a ≠ 0 ∧ x > 0) : 
  let expression := (a / Real.sqrt x - Real.sqrt x / a^2)^8
  let fifth_term := (Nat.choose 8 4) * (a / Real.sqrt x)^4 * (-Real.sqrt x / a^2)^4
  fifth_term = 70 / a^4 := by
  sorry

#check fifth_term_binomial_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_binomial_expansion_l945_94538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_triangle_calculation_l945_94510

noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

noncomputable def triangle (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

theorem nabla_triangle_calculation (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ((nabla a b) |> triangle c |> nabla d) = 1 := by
  have h1 : a = 2 := by sorry
  have h2 : b = 3 := by sorry
  have h3 : c = 4 := by sorry
  have h4 : d = 1 := by sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_triangle_calculation_l945_94510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_count_l945_94526

/-- The number of ways to arrange 4 people in a row with 2 specific people together -/
def arrangementCount : ℕ := 12

/-- The number of people to be arranged -/
def totalPeople : ℕ := 4

/-- The number of people that must be together -/
def togetherPeople : ℕ := 2

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_arrangement_count :
  arrangementCount = (totalPeople - togetherPeople + 1) * Nat.factorial togetherPeople := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_count_l945_94526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_sweater_markup_theorem_l945_94530

/-- Represents the markup percentage of a sweater's retail price from its wholesale cost -/
noncomputable def sweater_markup (discount_percent : ℝ) (profit_percent : ℝ) : ℝ :=
  let sale_price := (1 + profit_percent / 100) -- Sale price as a multiple of wholesale cost
  let original_price := sale_price / (1 - discount_percent / 100) -- Original price as a multiple of wholesale cost
  (original_price - 1) * 100 -- Markup percentage

/-- 
Theorem stating that if a sweater sold at a 50% discount yields a 40% profit on the wholesale cost, 
then the normal retail price is marked up by 180% from the wholesale cost.
-/
theorem sweater_markup_theorem : sweater_markup 50 40 = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_sweater_markup_theorem_l945_94530
