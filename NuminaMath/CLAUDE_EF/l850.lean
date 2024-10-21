import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l850_85012

theorem sufficient_not_necessary :
  (∀ x : ℝ, (x - 5) / (2 - x) > 0 → |x - 1| < 4) ∧
  (∃ x : ℝ, |x - 1| < 4 ∧ (x - 5) / (2 - x) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l850_85012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l850_85024

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 6*x + 13)

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 2 ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ f c) ∧
  f c = 1/4 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l850_85024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_range_on_interval_l850_85018

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

-- Theorem for monotonicity
theorem f_monotone_increasing : 
  StrictMonoOn f (Set.Ioi (-1 : ℝ)) := by sorry

-- Theorem for range
theorem f_range_on_interval : 
  Set.image f (Set.Icc (0 : ℝ) 2) = Set.Icc 1 (5/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_range_on_interval_l850_85018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_nine_values_with_three_two_digit_multiples_l850_85077

theorem exactly_nine_values_with_three_two_digit_multiples :
  ∃! (S : Finset ℕ), 
    (∀ x ∈ S, x > 0 ∧ 
      (∃! (M : Finset ℕ), 
        M.card = 3 ∧ 
        (∀ m ∈ M, 10 ≤ m ∧ m < 100 ∧ ∃ k : ℕ, m = k * x) ∧
        x ∈ M ∧ 2*x ∈ M ∧ 3*x ∈ M)) ∧
    (∀ x ∈ S, 3*x < 100 ∧ 4*x ≥ 100) ∧
    S.card = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_nine_values_with_three_two_digit_multiples_l850_85077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l850_85087

-- Define the circle C in polar coordinates
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := ρ * Real.cos θ - 2 * ρ * Real.sin θ + 7 = 0

-- Define the shortest distance function
noncomputable def shortest_distance : ℝ := (8 * Real.sqrt 5) / 5 - 1

-- Theorem statement
theorem shortest_distance_circle_to_line :
  ∀ (θ : ℝ), ∃ (d : ℝ), d ≥ 0 ∧
  (∀ (ρ' θ' : ℝ), line_l ρ' θ' →
    d ≤ Real.sqrt ((circle_C θ * Real.cos θ - ρ' * Real.cos θ')^2 +
                   (circle_C θ * Real.sin θ - ρ' * Real.sin θ')^2)) ∧
  d = shortest_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l850_85087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l850_85060

/-- Represents a jump vector in 2D space -/
structure Jump where
  x : ℝ
  y : ℝ

/-- Generates a random jump of given length -/
noncomputable def randomJump (length : ℝ) : Jump :=
  sorry

/-- Calculates the final position after a sequence of jumps -/
def finalPosition (jumps : List Jump) : Jump :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Jump) : ℝ :=
  sorry

/-- Probability of the last jump being 1 meter -/
noncomputable def probLastJump1m : ℝ := 1/2

/-- Probability of the last jump being 2 meters -/
noncomputable def probLastJump2m : ℝ := 1/2

/-- The frog's jumping scenario -/
theorem frog_jump_probability :
  let firstThreeJumps := [randomJump 1, randomJump 1, randomJump 1]
  let lastJump1m := randomJump 1
  let lastJump2m := randomJump 2
  let finalPos1m := finalPosition (firstThreeJumps ++ [lastJump1m])
  let finalPos2m := finalPosition (firstThreeJumps ++ [lastJump2m])
  let prob1m := probLastJump1m * (if distance finalPos1m (Jump.mk 0 0) ≤ 1.5 then 1 else 0)
  let prob2m := probLastJump2m * (if distance finalPos2m (Jump.mk 0 0) ≤ 1.5 then 1 else 0)
  prob1m + prob2m = 1/6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l850_85060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_problem_y_intercept_l850_85050

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis (i.e., where x = 0) -/
noncomputable def y_intercept (a b c : ℝ) : ℝ := -c / b

/-- The line equation is in the form ax + by + c = 0 -/
theorem y_intercept_of_line (a b c : ℝ) (hb : b ≠ 0) :
  y_intercept a b c = -c / b := by
  sorry

theorem problem_y_intercept :
  y_intercept 1 (-2) (-3) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_problem_y_intercept_l850_85050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_diff_C_B_C_superset_A_intersect_B_l850_85008

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- Theorem for part (1)
theorem intersection_A_diff_C_B :
  ∃ a > 0, A ∩ (C a \ B) = Set.Ioc (-3) 2 := by sorry

-- Theorem for part (2)
theorem C_superset_A_intersect_B (a : ℝ) :
  a > 0 → (C a ⊇ (A ∩ B) ↔ 4/3 ≤ a ∧ a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_diff_C_B_C_superset_A_intersect_B_l850_85008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_factors_l850_85094

theorem min_sum_of_factors (a b c : ℕ) : 
  a * b * c = 396 → 
  Odd a → 
  (∀ a' b' c' : ℕ, a' * b' * c' = 396 → Odd a' → a + b + c ≤ a' + b' + c') →
  a + b + c = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_factors_l850_85094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_specific_l850_85042

/-- The radius of a sphere inscribed in a right cone --/
noncomputable def inscribed_sphere_radius (base_radius height : ℝ) : ℝ :=
  let cone_side := Real.sqrt (base_radius^2 + height^2)
  8 * (Real.sqrt 5 - 1)

/-- Theorem: The radius of a sphere inscribed in a right cone with base radius 16 cm and height 32 cm is 8√5 - 8 cm --/
theorem inscribed_sphere_radius_specific : 
  inscribed_sphere_radius 16 32 = 8 * (Real.sqrt 5 - 1) := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_specific_l850_85042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_acute_iff_lambda_range_l850_85090

/-- Given vectors a and b in ℝ², and c = a + λb, prove that the angle between a and c is acute
    if and only if λ > -5/2 and λ ≠ 0. -/
theorem angle_acute_iff_lambda_range (a b : ℝ × ℝ) (l : ℝ) :
  a = (1, 3) →
  b = (1, 1) →
  let c := a + l • b
  (0 < a.1 * c.1 + a.2 * c.2 ∧ a ≠ c) ↔ l > -5/2 ∧ l ≠ 0 := by
  sorry

#check angle_acute_iff_lambda_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_acute_iff_lambda_range_l850_85090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_diff_eq_series_ln_2_eq_series_l850_85056

/-- The series representation of ln(n+1) - ln(n) for natural numbers n -/
noncomputable def ln_series (n : ℕ) : ℝ := 2 * ∑' k, (1 : ℝ) / ((2 * k + 1) * (2 * n + 1) ^ (2 * k + 1))

/-- The series representation of ln(2) -/
noncomputable def ln_2_series : ℝ := 2 / 3 * ∑' k, (1 : ℝ) / ((2 * k + 1) * 9 ^ k)

/-- Theorem stating the equality of ln(n+1) - ln(n) and its series representation -/
theorem ln_diff_eq_series (n : ℕ) : Real.log (n + 1) - Real.log n = ln_series n := by sorry

/-- Theorem stating the equality of ln(2) and its series representation -/
theorem ln_2_eq_series : Real.log 2 = ln_2_series := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_diff_eq_series_ln_2_eq_series_l850_85056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_m_1_f_extrema_general_l850_85049

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := 2 * (Real.sin x)^2 + m * Real.cos x + 1

-- Theorem for m = 1
theorem f_extrema_m_1 :
  (∀ x, f x 1 ≤ 25/8) ∧ (∃ x, f x 1 = 25/8) ∧
  (∀ x, f x 1 ≥ 0) ∧ (∃ x, f x 1 = 0) := by sorry

-- Helper function for the maximum value of f
noncomputable def f_max (m : ℝ) : ℝ :=
  if -4 ≤ m ∧ m ≤ 4 then m^2/8 + 3
  else if m < -4 then 1 - m
  else 1 + m

-- Helper function for the minimum value of f
noncomputable def f_min (m : ℝ) : ℝ :=
  if m < 0 then 1 + m else 1 - m

-- Theorem for any real m
theorem f_extrema_general (m : ℝ) :
  (∀ x, f x m ≤ f_max m) ∧ (∃ x, f x m = f_max m) ∧
  (∀ x, f x m ≥ f_min m) ∧ (∃ x, f x m = f_min m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_m_1_f_extrema_general_l850_85049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l850_85030

theorem solve_exponential_equation (y : ℝ) (h : (4 : ℝ) ^ y = 128) : y = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l850_85030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circumcenters_l850_85084

noncomputable section

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 5 ∧ BC = Real.sqrt 17 ∧ AC = 4

-- Define point M on AC such that CM = 1
def point_M (A C M : ℝ × ℝ) : Prop :=
  let CM := Real.sqrt ((C.1 - M.1)^2 + (C.2 - M.2)^2)
  CM = 1 ∧ ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)

-- Define the center of a circumcircle
noncomputable def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let a := B.1 - A.1
  let b := B.2 - A.2
  let c := C.1 - A.1
  let d := C.2 - A.2
  let e := a * (A.1 + B.1) + b * (A.2 + B.2)
  let f := c * (A.1 + C.1) + d * (A.2 + C.2)
  let g := 2 * (a * d - b * c)
  ((d * e - b * f) / g, (a * f - c * e) / g)

-- Theorem statement
theorem distance_between_circumcenters 
  (A B C M : ℝ × ℝ) 
  (h_triangle : triangle_ABC A B C) 
  (h_point_M : point_M A C M) : 
  let center_ABM := circumcenter A B M
  let center_BCM := circumcenter B C M
  Real.sqrt ((center_ABM.1 - center_BCM.1)^2 + (center_ABM.2 - center_BCM.2)^2) = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circumcenters_l850_85084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_borrowed_l850_85000

theorem max_books_borrowed (total_students : Nat) (no_books : Nat) (one_book : Nat) (two_books : Nat) (avg_books : Nat) :
  total_students = 20 ∧
  no_books = 3 ∧
  one_book = 9 ∧
  two_books = 4 ∧
  avg_books = 2 →
  ∃ (max_books : Nat),
    max_books = 14 ∧
    max_books = total_students * avg_books - (one_book * 1 + two_books * 2) - (total_students - no_books - one_book - two_books - 1) * 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_borrowed_l850_85000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_two_to_four_l850_85031

-- Define the function f(x) = log_a(ax^2-x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x) / Real.log a

-- Define the property of being increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Theorem statement
theorem f_increasing_on_two_to_four (a : ℝ) (h : a > 1) :
  IncreasingOn (f a) 2 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_two_to_four_l850_85031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_part1_calculation_part2_l850_85083

-- Part 1
theorem calculation_part1 : 
  (1 : ℝ) * (2.25 : ℝ)^(1/2 : ℝ) - (-9.6 : ℝ)^(0 : ℝ) - (27/8 : ℝ)^(-(2/3) : ℝ) + (1.5 : ℝ)^(-2 : ℝ) = 1/2 := by sorry

-- Part 2
theorem calculation_part2 : 
  (1/2 : ℝ) * (Real.log 25 / Real.log 10) + (Real.log 2 / Real.log 10) - 
  (Real.log (Real.sqrt 0.1) / Real.log 10) - 
  (Real.log 9 / Real.log 2) * (Real.log 2 / Real.log 3) = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_part1_calculation_part2_l850_85083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quotient_inequality_l850_85068

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- The quotient function -/
noncomputable def q {α : Type*} (X Y : Set α) : ℝ := sorry

/-- A partition of a set -/
def IsPartition {α : Type*} (S : Set (Set α)) (X : Set α) :=
  (∀ A ∈ S, A ⊆ X) ∧ (∀ x ∈ X, ∃ A ∈ S, x ∈ A) ∧ (∀ A B, A ∈ S → B ∈ S → A ≠ B → A ∩ B = ∅)

/-- Refinement of a partition -/
def IsRefinement {α : Type*} (P P' : Set (Set α)) :=
  ∀ A' ∈ P', ∃ A ∈ P, A' ⊆ A

theorem quotient_inequality (C D : Set V) (C_part D_part : Set (Set V)) (P P' : Set (Set V)) 
  (hCD : Disjoint C D)
  (hC : IsPartition C_part C)
  (hD : IsPartition D_part D)
  (hP : IsPartition P (⋃₀ P))
  (hP' : IsPartition P' (⋃₀ P'))
  (hRef : IsRefinement P P') :
  (q C_part D_part ≥ q C D) ∧ (q P' ≥ q P) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quotient_inequality_l850_85068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_radius_l850_85085

noncomputable section

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem locus_radius (C : Circle) (S O : Point) (R : ℝ) :
  C.center = (O.x, O.y) →
  C.radius = R →
  distance O S < R →
  ∃ (M N' M' N : Point),
    let r := Real.sqrt (2 * R^2 - (distance O S)^2)
    (∀ A : Point, distance O A = R →
      ∃ (A' B B' : Point),
        distance S A = distance S A' ∧ 
        distance S B = distance S B' ∧
        (distance O M = r ∧ distance O N' = r ∧ distance O M' = r ∧ distance O N = r)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_radius_l850_85085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l850_85091

theorem problem_solution (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + 2*b = 3) :
  (1 < a ∧ a < 3) ∧
  (let c := -3*a + 2*b; c = -4*a + 3 ∧ -9 < c ∧ c < -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l850_85091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_measure_l850_85017

/-- The measure of angle E in a quadrilateral EFGH where E = 3F = 4G = 6H -/
noncomputable def angle_E : ℝ :=
  let x : ℝ := 360 / (1 + 1/3 + 1/4 + 1/6)
  ⌊x + 0.5⌋

theorem quadrilateral_angle_measure :
  angle_E = 206 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_measure_l850_85017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_tomato_yield_l850_85052

noncomputable def step_length : ℝ := 2.5
noncomputable def yield_per_sqft : ℝ := 3/4

noncomputable def area_to_yield (length : ℝ) (width : ℝ) : ℝ :=
  length * width * yield_per_sqft

noncomputable def total_yield (area1_length : ℝ) (area1_width : ℝ) (area2_length : ℝ) (area2_width : ℝ) : ℝ :=
  area_to_yield (area1_length * step_length) (area1_width * step_length) +
  area_to_yield (area2_length * step_length) (area2_width * step_length)

theorem expected_tomato_yield :
  total_yield 15 15 15 5 = 1406.25 := by
  -- Unfold definitions
  unfold total_yield
  unfold area_to_yield
  unfold step_length
  unfold yield_per_sqft
  -- Simplify the expression
  simp
  -- The proof itself would require more steps, so we use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_tomato_yield_l850_85052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomials_root_existence_l850_85013

/-- A polynomial of degree 10 with leading coefficient 1 -/
def Polynomial10 : Type := {p : Polynomial ℝ // p.degree = 10 ∧ p.leadingCoeff = 1}

/-- The statement to prove -/
theorem polynomials_root_existence (P Q : Polynomial10) 
  (h : ∀ x : ℝ, (P.val).eval x ≠ (Q.val).eval x) :
  ∃ x : ℝ, ((P.val).comp (Polynomial.X + 1)).eval x = ((Q.val).comp (Polynomial.X - 1)).eval x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomials_root_existence_l850_85013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_eq_cos_n_minus_sin_n_l850_85039

theorem cos_2x_eq_cos_n_minus_sin_n (n : ℤ) : 
  (∀ x : ℝ, Real.cos (2 * x) = (Real.cos x) ^ n - (Real.sin x) ^ n) ↔ n = 2 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_eq_cos_n_minus_sin_n_l850_85039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_D_l850_85062

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A square with side length 1 -/
structure UnitSquare where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The condition that P satisfies u^2 + v^2 = w^2 -/
def satisfiesCondition (P : Point) (square : UnitSquare) : Prop :=
  (distance P square.A)^2 + (distance P square.B)^2 = (distance P square.C)^2

theorem max_distance_to_D (square : UnitSquare) :
  ∃ (P : Point), satisfiesCondition P square ∧
    ∀ (Q : Point), satisfiesCondition Q square →
      distance Q square.D ≤ 2 + Real.sqrt 2 := by
  sorry

#check max_distance_to_D

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_D_l850_85062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_approximation_l850_85086

-- Define the loan parameters
noncomputable def loan_amount : ℚ := 120
noncomputable def received_amount : ℚ := 114
def num_installments : ℕ := 12
noncomputable def installment_amount : ℚ := 10
noncomputable def interest_charged : ℚ := 6

-- Define the interest rate calculation function
noncomputable def calculate_interest_rate (principal : ℚ) (interest : ℚ) : ℚ :=
  (interest / principal) * 100

-- Theorem statement
theorem loan_interest_rate_approximation :
  let total_repayment := (num_installments : ℚ) * installment_amount
  let interest_rate := calculate_interest_rate received_amount interest_charged
  (total_repayment = loan_amount) ∧ 
  (interest_rate ≥ 5) ∧ 
  (interest_rate < 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_approximation_l850_85086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_divisibility_l850_85038

theorem unique_k_divisibility : ∃! k : ℕ, 
  k > 0 ∧ ∃ n : ℕ, (2^n + 11) % (2^k - 1) = 0 ∧ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_divisibility_l850_85038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_cannot_tile_with_triangle_l850_85093

/-- Represents the internal angle of a regular polygon with n sides -/
noncomputable def internal_angle (n : ℕ) : ℝ := (n - 2) * 180 / n

/-- Checks if a regular polygon with n sides can tile with an equilateral triangle -/
def can_tile_with_triangle (n : ℕ) : Prop :=
  ∃ (k m : ℕ), k * internal_angle n + m * 60 = 360 ∧ k > 0 ∧ m > 0

theorem octagon_cannot_tile_with_triangle :
  can_tile_with_triangle 4 ∧ 
  ¬(can_tile_with_triangle 8) ∧ 
  can_tile_with_triangle 12 ∧ 
  can_tile_with_triangle 6 := by
  sorry

#check octagon_cannot_tile_with_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_cannot_tile_with_triangle_l850_85093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_seventh_powers_of_roots_l850_85072

theorem sum_of_seventh_powers_of_roots (x₁ x₂ : ℂ) : 
  x₁^2 + 3 * x₁ + 1 = 0 → 
  x₂^2 + 3 * x₂ + 1 = 0 → 
  x₁^7 + x₂^7 = -843 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_seventh_powers_of_roots_l850_85072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l850_85057

/-- Given positive real numbers a, b, and c, define the function f(x) -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  (a^x / (b^x + c^x)) + (b^x / (a^x + c^x)) + (c^x / (a^x + b^x))

/-- Theorem stating that f(x) is nondecreasing on [0,∞) and nonincreasing on (-∞,0] -/
theorem f_monotonicity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y → f a b c x ≤ f a b c y) ∧
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f a b c x ≥ f a b c y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l850_85057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_problem_l850_85003

theorem energy_problem (E₁ E₂ E₃ : ℤ) 
  (eq1 : E₁^2 - E₂^2 - E₃^2 + E₁*E₂ = 5040)
  (eq2 : E₁^2 + 2*E₂^2 + 2*E₃^2 - 2*E₁*E₂ - E₁*E₃ - E₂*E₃ = -4968)
  (h1 : E₁ ≥ E₂)
  (h2 : E₂ ≥ E₃)
  (h3 : E₁ > 0)
  (h4 : E₂ > 0)
  (h5 : E₃ > 0) :
  E₁ = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_problem_l850_85003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equations_l850_85065

theorem solution_satisfies_equations :
  let x : ℚ := 99 / 47
  let y : ℚ := 137 / 47
  (7 * x - 3 * y = 6) ∧ (4 * x + 5 * y = 23) := by
  -- Unfold the let bindings
  simp_all
  -- Split the conjunction into two goals
  constructor
  -- Prove the first equation
  · norm_num
  -- Prove the second equation
  · norm_num

#check solution_satisfies_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equations_l850_85065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l850_85075

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (4 * x + Real.pi / 3) - 1

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi / 2) ∧
  f (Real.pi / 3) = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l850_85075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_pbq_l850_85051

/-- A square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

/-- A point on a line segment -/
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

/-- The angle between three points -/
noncomputable def Angle (P D Q : ℝ × ℝ) : ℝ := sorry

/-- The distance between two points -/
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- The perimeter of a triangle -/
noncomputable def TrianglePerimeter (P B Q : ℝ × ℝ) : ℝ :=
  Distance P B + Distance B Q + Distance P Q

theorem perimeter_of_triangle_pbq (square : UnitSquare)
  (P : ℝ × ℝ) (hP : PointOnSegment P square.A square.B)
  (Q : ℝ × ℝ) (hQ : PointOnSegment Q square.B square.C)
  (h_angle : Angle P square.D Q = π/4) :
  TrianglePerimeter P square.B Q = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_pbq_l850_85051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l850_85099

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

/-- Function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

/-- The main theorem encapsulating the problem -/
theorem triangle_properties (t : Triangle) (h1 : t.area = 1/2) (h2 : t.AB = 1) (h3 : t.BC = Real.sqrt 2) 
    (h4 : f t.B = -Real.sqrt 3) : 
  (t.AC = 1 ∨ t.AC = Real.sqrt 5) ∧ Real.sin t.A = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l850_85099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_sine_not_always_four_fifths_l850_85046

theorem terminal_side_sine_not_always_four_fifths :
  ∃ (α : ℝ) (k : ℝ),
    k ≠ 0 ∧
    (∃ (x y : ℝ), x = 3*k ∧ y = 4*k ∧ x = y * Real.tan α) →
    Real.sin α ≠ 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_sine_not_always_four_fifths_l850_85046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_50_cents_l850_85005

def box : Finset (ℕ × ℕ) := {(1, 2), (5, 4), (10, 6)}

def total_coins : ℕ := (box.sum (λ x => x.2))

def draw_size : ℕ := 6

def favorable_outcomes : ℕ := 127

def total_outcomes : ℕ := 924

theorem probability_at_least_50_cents :
  (favorable_outcomes : ℚ) / total_outcomes =
  (Finset.filter (λ s => s.sum (λ x => x.1 * x.2) ≥ 50)
    (Finset.powersetCard draw_size box)).card / (total_outcomes : ℚ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_50_cents_l850_85005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_y_relationship_l850_85088

noncomputable section

-- Define the inverse proportion function
def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- Define the theorem
theorem inverse_proportion_y_relationship 
  (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h_k : k < 0)
  (h_y₁ : y₁ = inverse_proportion k (-4))
  (h_y₂ : y₂ = inverse_proportion k (-2))
  (h_y₃ : y₃ = inverse_proportion k 3) :
  y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_y_relationship_l850_85088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_moving_receiver_l850_85082

/-- The speed of a moving receiver given the distance from a stationary sound source,
    the speed of sound, and the time taken for the sound to reach the receiver. -/
theorem speed_of_moving_receiver 
  (distance : ℝ) 
  (speed_of_sound : ℝ) 
  (time : ℝ) 
  (h1 : distance = 1200)
  (h2 : speed_of_sound = 330)
  (h3 : time = 3.9669421487603307)
  : ∃ (speed : ℝ), (abs (speed - 27.5) < 0.1) ∧ distance = (speed_of_sound - speed) * time :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_moving_receiver_l850_85082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_areas_l850_85055

/-- Represents a frustum of a cone --/
structure Frustum where
  r₁ : ℝ  -- lower base radius
  r₂ : ℝ  -- upper base radius
  h : ℝ   -- height

/-- Calculate the lateral surface area of a frustum --/
noncomputable def lateralSurfaceArea (f : Frustum) : ℝ :=
  Real.pi * (f.r₁ + f.r₂) * Real.sqrt (f.h^2 + (f.r₁ - f.r₂)^2)

/-- Calculate the total surface area of a frustum --/
noncomputable def totalSurfaceArea (f : Frustum) : ℝ :=
  lateralSurfaceArea f + Real.pi * (f.r₁^2 + f.r₂^2)

/-- Theorem about the surface areas of a specific frustum --/
theorem frustum_surface_areas :
  let f : Frustum := { r₁ := 8, r₂ := 4, h := 5 }
  lateralSurfaceArea f = 12 * Real.pi * Real.sqrt 41 ∧
  totalSurfaceArea f = (80 + 12 * Real.sqrt 41) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_areas_l850_85055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_avg_is_70_l850_85063

/-- Represents a charitable association with male and female members selling raffle tickets. -/
structure Association where
  overall_avg : ℚ
  male_female_ratio : ℚ
  male_avg : ℚ

/-- Calculates the average number of tickets sold by female members. -/
noncomputable def female_avg (a : Association) : ℚ :=
  (3 * a.overall_avg - a.male_avg) / 2

/-- Theorem stating that given the specific conditions, the average number of tickets
    sold by female members is 70. -/
theorem female_avg_is_70 (a : Association)
  (h1 : a.overall_avg = 66)
  (h2 : a.male_female_ratio = 1 / 2)
  (h3 : a.male_avg = 58) :
  female_avg a = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_avg_is_70_l850_85063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_alternative_form_l850_85021

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic polynomial at a given x -/
noncomputable def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Represents the alternative form of the quadratic polynomial -/
structure AlternativeForm where
  d : ℝ
  e : ℝ
  f : ℝ

/-- Evaluates the alternative form at a given x -/
noncomputable def AlternativeForm.eval (alt : AlternativeForm) (x : ℝ) : ℝ :=
  (alt.d / 2) * x * (x - 1) + alt.e * x + alt.f

theorem quadratic_alternative_form (p : QuadraticPolynomial) :
  ∃ (alt : AlternativeForm),
    (∀ x, p.eval x = alt.eval x) ∧
    alt.d = 2 * p.a ∧
    alt.e = p.a + p.b ∧
    alt.f = p.c ∧
    (∀ n : ℤ, ∃ m : ℤ, p.eval (n : ℝ) = m) ↔ (∃ d' e' f' : ℤ, alt.d = d' ∧ alt.e = e' ∧ alt.f = f') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_alternative_form_l850_85021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_solution_l850_85028

/-- Given a function f : ℝ → ℝ satisfying the equation 3f(x-1) + 2f(1-x) = 2x for all x,
    prove that f(x) = (2/5)x + 2/5 -/
theorem function_solution (f : ℝ → ℝ) (h : ∀ x, 3 * f (x - 1) + 2 * f (1 - x) = 2 * x) :
  f = fun x ↦ (2/5) * x + 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_solution_l850_85028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biangle_congruence_and_area_biangle_area_formula_l850_85032

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a spherical biangle on a sphere -/
structure SphericalBiangle where
  sphere : Sphere
  angle : ℝ

/-- Two spherical biangles are congruent if they have the same sphere and angle -/
def congruent (b1 b2 : SphericalBiangle) : Prop :=
  b1.sphere = b2.sphere ∧ b1.angle = b2.angle

/-- The area of a spherical biangle -/
def area (b : SphericalBiangle) : ℝ :=
  2 * b.sphere.radius ^ 2 * b.angle

theorem biangle_congruence_and_area 
  (b1 b2 : SphericalBiangle) (h : b1.sphere = b2.sphere) :
  b1.angle = b2.angle → congruent b1 b2 ∧ area b1 = area b2 := by
  sorry

theorem biangle_area_formula (b : SphericalBiangle) :
  area b = 2 * b.sphere.radius ^ 2 * b.angle := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biangle_congruence_and_area_biangle_area_formula_l850_85032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l850_85076

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 2 / x

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 2 ∧ f x = y) ↔ y ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l850_85076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_payment_l850_85059

/-- Given a 10% deposit of $130, prove that the remaining amount to be paid is $1170 -/
theorem remaining_payment (deposit : ℚ) (deposit_percentage : ℚ) : 
  deposit = 130 → deposit_percentage = 1/10 → 
  (deposit / deposit_percentage - deposit : ℚ) = 1170 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_payment_l850_85059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_theorem_l850_85022

theorem intersection_point_theorem (a b : ℕ) (h : a ≠ b) :
  ∃ c : ℕ, c ≠ a ∧ c ≠ b ∧
  ∀ x : ℝ, Real.sin (a * x) = Real.sin (b * x) → Real.sin (c * x) = Real.sin (a * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_theorem_l850_85022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_product_minus_main_theorem_l850_85080

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.whole + d.repeating / (10^(Nat.digits 10 d.repeating).length - 1)

/-- The repeating decimal 0.overline{6} -/
def zero_point_six : RepeatingDecimal := ⟨0, 6⟩

/-- The repeating decimal 0.overline{2} -/
def zero_point_two : RepeatingDecimal := ⟨0, 2⟩

/-- The repeating decimal 0.overline{4} -/
def zero_point_four : RepeatingDecimal := ⟨0, 4⟩

theorem repeating_decimal_product_minus (d1 d2 d3 : RepeatingDecimal) :
  (toRational d1 * toRational d2) - toRational d3 = -8/27 :=
by
  sorry

/-- The main theorem proving the given equation. -/
theorem main_theorem : 
  (toRational zero_point_six * toRational zero_point_two) - toRational zero_point_four = -8/27 :=
by
  apply repeating_decimal_product_minus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_product_minus_main_theorem_l850_85080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_shortest_chord_m_value_shortest_chord_length_l850_85092

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line
def line_eq (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Statement 1: The line always passes through (3,1)
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_eq m 3 1 := by sorry

-- Statement 2: The chord is shortest when m = -3/4
theorem shortest_chord_m_value :
  ∃ m : ℝ, m = -3/4 ∧
  ∀ m' : ℝ, m' ≠ m →
    ∃ x y x' y' : ℝ,
      circle_eq x y ∧ circle_eq x' y' ∧
      line_eq m x y ∧ line_eq m x' y' ∧
      line_eq m' x y ∧ line_eq m' x' y' →
      (x - x')^2 + (y - y')^2 < (x - x')^2 + (y - y')^2 := by sorry

-- Statement 3: The length of the shortest chord is 4√5
theorem shortest_chord_length :
  ∃ x y x' y' : ℝ,
    circle_eq x y ∧ circle_eq x' y' ∧
    line_eq (-3/4) x y ∧ line_eq (-3/4) x' y' ∧
    (x - x')^2 + (y - y')^2 = 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_shortest_chord_m_value_shortest_chord_length_l850_85092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_special_triangle_l850_85078

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem min_perimeter_special_triangle :
  ∀ l m n : ℕ,
    l > m ∧ m > n →
    fractional_part (3^l / 10000 : ℝ) = fractional_part (3^m / 10000 : ℝ) ∧
    fractional_part (3^m / 10000 : ℝ) = fractional_part (3^n / 10000 : ℝ) →
    (∀ l' m' n' : ℕ,
      l' > m' ∧ m' > n' →
      fractional_part (3^l' / 10000 : ℝ) = fractional_part (3^m' / 10000 : ℝ) ∧
      fractional_part (3^m' / 10000 : ℝ) = fractional_part (3^n' / 10000 : ℝ) →
      l' + m' + n' ≥ l + m + n) →
    l + m + n = 3003 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_special_triangle_l850_85078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l850_85011

-- Define a function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(2x-3)
def domain_f_2x_minus_3 : Set ℝ := Set.Icc 1 3

-- Define the domain of f(1-3x)
def domain_f_1_minus_3x : Set ℝ := Set.Ico (-2/3) (2/3)

-- Theorem statement
theorem domain_transformation (h : ∀ x, f (2*x - 3) ∈ domain_f_2x_minus_3 ↔ x ∈ Set.Icc 2 3) :
  ∀ x, f (1 - 3*x) ∈ domain_f_1_minus_3x ↔ x ∈ Set.Ico (-2/3) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l850_85011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l850_85054

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression (√x - 2/x)^n
noncomputable def expression (x : ℝ) (n : ℕ) : ℝ := (Real.sqrt x - 2 / x) ^ n

-- Define the condition that the 4th and 9th terms have equal binomial coefficients
def equal_coefficients (n : ℕ) : Prop := binomial n 3 = binomial n 8

-- Define the coefficient of x in the expansion
def x_coefficient (n : ℕ) : ℤ := (-2)^3 * (binomial n 3 : ℤ)

-- State the theorem
theorem expansion_properties :
  ∃ n : ℕ, equal_coefficients n ∧ n = 11 ∧ x_coefficient n = -1320 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l850_85054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_l850_85073

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  equation : ℝ → ℝ → Prop := fun x y => 
    (x - center.1)^2 + (y - center.2)^2 = radius^2

/-- Theorem statement -/
theorem parabola_circle_tangency 
  (para : Parabola) 
  (P₁ P₂ : ℝ × ℝ) 
  (h₁ : para.equation P₁.1 P₁.2)
  (h₂ : para.equation P₂.1 P₂.2)
  (h₃ : |P₁.2 - P₂.2| = 4 * para.p) :
  ∃! Q : ℝ × ℝ, 
    Q ≠ P₁ ∧ Q ≠ P₂ ∧
    para.equation Q.1 Q.2 ∧
    (let c : Circle := {
      center := ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2),
      radius := (((P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2) / 4).sqrt
    }
    c.equation Q.1 Q.2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_l850_85073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l850_85033

theorem simplify_expression (k : ℤ) :
  (2 : ℝ)^(-3*k) - (2 : ℝ)^(-(3*k-2)) + (2 : ℝ)^(-(3*k+2)) = -(11/4) * (2 : ℝ)^(-3*k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l850_85033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_cubes_volume_and_surface_l850_85014

/-- Represents the configuration of two unit cubes with a common space diagonal -/
structure TwoCubesConfig where
  rotation_angle : ℝ
  rotation_axis : Fin 3 → ℝ

/-- Calculates the volume occupied by two unit cubes in the given configuration -/
noncomputable def volume_occupied (config : TwoCubesConfig) : ℝ :=
  sorry

/-- Calculates the surface area of the solid formed by the union of two unit cubes in the given configuration -/
noncomputable def surface_area (config : TwoCubesConfig) : ℝ :=
  sorry

/-- The specific configuration with 60° rotation -/
noncomputable def specific_config : TwoCubesConfig :=
  { rotation_angle := Real.pi / 3,  -- 60° in radians
    rotation_axis := λ _ ↦ 1 }  -- Placeholder for the common space diagonal

theorem two_cubes_volume_and_surface (config : TwoCubesConfig) 
  (h : config = specific_config) : 
  volume_occupied config = 5/4 ∧ surface_area config = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_cubes_volume_and_surface_l850_85014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l850_85045

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (h1 : train_length = 500) 
  (h2 : train_speed_kmh = 180) : 
  train_length / (train_speed_kmh * (1000 / 3600)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l850_85045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_positive_f_inequality_negative_min_value_condition_l850_85041

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := (x^2 + 3) / (x - a)

theorem f_inequality_positive (ha : a > 0) :
  ∀ x, f a x < x ↔ -3/a < x ∧ x < a :=
sorry

theorem f_inequality_negative (ha : a < 0) :
  ∀ x, f a x < x ↔ x < a ∨ x > -3/a :=
sorry

theorem min_value_condition (hmin : ∀ x > a, f a x ≥ 6) :
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_positive_f_inequality_negative_min_value_condition_l850_85041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l850_85096

/-- The function f(x) = |x-2| + 2|x-1| -/
def f (x : ℝ) : ℝ := |x - 2| + 2 * |x - 1|

/-- The solution set of f(x) > 4 -/
def solution_set : Set ℝ := {x | f x > 4}

/-- The range of m for which f(x) > 2m^2 - 7m + 4 holds for all x ∈ ℝ -/
def m_range : Set ℝ := {m | ∀ x, f x > 2 * m^2 - 7 * m + 4}

theorem f_properties :
  (solution_set = Set.Iio 0 ∪ Set.Ioi 0) ∧
  (m_range = Set.Ioo (1/2) 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l850_85096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_approx_l850_85089

/-- Calculates the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The initial deposit amount -/
def initial_deposit : ℝ := 12000

/-- Cedric's interest rate -/
def cedric_rate : ℝ := 0.06

/-- Daniel's interest rate -/
def daniel_rate : ℝ := 0.08

/-- The time period in years -/
def time_period : ℕ := 20

/-- Theorem stating the difference between Cedric's and Daniel's account balances -/
theorem balance_difference_approx : 
  ∃ (diff : ℝ), 
    abs (compound_interest initial_deposit cedric_rate time_period - 
         simple_interest initial_deposit daniel_rate time_period - diff) < 1 ∧
    abs (diff - 7286) < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_approx_l850_85089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_critical_temp_eq_max_liquid_temp_l850_85095

/-- The critical temperature of a substance -/
def critical_temperature (substance : Type) : ℝ := sorry

/-- The maximum temperature at which a substance can exist as a liquid -/
def max_liquid_temperature (substance : Type) : ℝ := sorry

/-- Water as a substance -/
def water : Type := sorry

/-- Theorem: The critical temperature of water is equal to the maximum temperature at which liquid water can exist -/
theorem water_critical_temp_eq_max_liquid_temp :
  critical_temperature water = max_liquid_temperature water := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_critical_temp_eq_max_liquid_temp_l850_85095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_wave_amplitude_is_3_sqrt_5_l850_85037

/-- The amplitude of a combined wave -/
noncomputable def combined_wave_amplitude (t : ℝ) : ℝ :=
  let y₁ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
  let y₂ := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
  let y := y₁ + y₂
  3 * Real.sqrt 5

/-- The amplitude of the combined wave is 3√5 -/
theorem combined_wave_amplitude_is_3_sqrt_5 :
  ∀ t : ℝ, combined_wave_amplitude t = 3 * Real.sqrt 5 := by
  sorry

#check combined_wave_amplitude_is_3_sqrt_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_wave_amplitude_is_3_sqrt_5_l850_85037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_three_thirty_l850_85047

/-- The angle between the hour and minute hands of a clock at a given time -/
noncomputable def clockAngle (hour : ℕ) (minute : ℕ) : ℝ :=
  |60 * (hour : ℝ) - 11 * (minute : ℝ)| / 2

theorem angle_at_three_thirty :
  clockAngle 3 30 = 75 := by
  -- Unfold the definition of clockAngle
  unfold clockAngle
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_three_thirty_l850_85047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l850_85071

/-- Given two trains traveling in opposite directions on a 240-mile route,
    where Train B travels at 109.071 mph and they pass each other after 1.25542053973 hours,
    prove that Train A's speed is approximately 82.07 mph. -/
theorem train_speed_problem (total_distance : ℝ) (speed_B : ℝ) (meeting_time : ℝ) :
  total_distance = 240 →
  speed_B = 109.071 →
  meeting_time = 1.25542053973 →
  ∃ speed_A : ℝ, (speed_A + speed_B) * meeting_time = total_distance ∧ 
  abs (speed_A - 82.07) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l850_85071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l850_85023

/-- The distance between two parallel planes in 3D space -/
noncomputable def distance_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  abs (d₂ - d₁) / Real.sqrt (a₁^2 + b₁^2 + c₁^2)

/-- Theorem stating the distance between two specific planes -/
theorem distance_between_specific_planes :
  distance_between_planes 3 (-1) 1 (-3) 6 (-2) 2 4 = 5 * Real.sqrt 11 / 11 := by
  sorry

#check distance_between_specific_planes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l850_85023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l850_85066

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the lines
def line1 (x y : ℝ) : Prop := 4*x - 3*y + 6 = 0
def line2 (x : ℝ) : Prop := x = -1

-- Define the distance function from a point to a line
noncomputable def distToLine1 (x y : ℝ) : ℝ := |4*x - 3*y + 6| / Real.sqrt (4^2 + 3^2)
def distToLine2 (x : ℝ) : ℝ := |x + 1|

-- Theorem statement
theorem min_sum_distances :
  ∀ x y : ℝ, parabola x y →
  (∃ m : ℝ, ∀ a b : ℝ, parabola a b → distToLine1 a b + distToLine2 a ≥ m) ∧
  (∃ x₀ y₀ : ℝ, parabola x₀ y₀ ∧ distToLine1 x₀ y₀ + distToLine2 x₀ = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l850_85066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_divisibility_l850_85006

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℕ  -- The non-repeating part as a natural number
  repeating : ℕ     -- The repeating part as a natural number
  nonRepeatingDigits : ℕ  -- Number of digits in the non-repeating part
  repeatingDigits : ℕ     -- Number of digits in the repeating part

/-- Converts a RepeatingDecimal to a rational number -/
def toRational (d : RepeatingDecimal) : ℚ :=
  (d.nonRepeating : ℚ) / (10 ^ d.nonRepeatingDigits) + 
  ((d.repeating : ℚ) / ((10 ^ d.repeatingDigits - 1) * 10 ^ d.nonRepeatingDigits))

/-- The main theorem: For any repeating decimal equal to m/n where m and n are coprime,
    n is divisible by 2 or 5 (or both) -/
theorem repeating_decimal_divisibility 
  (d : RepeatingDecimal) 
  (m n : ℕ) 
  (h1 : toRational d = m / n) 
  (h2 : Nat.Coprime m n) 
  (h3 : d.nonRepeatingDigits > 0) : 
  (2 ∣ n) ∨ (5 ∣ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_divisibility_l850_85006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_2023_eq_2_l850_85020

noncomputable section

/-- A function g satisfying the given conditions -/
def g : ℝ → ℝ := sorry

/-- g is positive for positive inputs -/
axiom g_pos (x : ℝ) (hx : x > 0) : g x > 0

/-- The functional equation for g -/
axiom g_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  g (x - y) = Real.sqrt (g (x * y) + 2)

/-- The main theorem: g(2023) = 2 -/
theorem g_2023_eq_2 : g 2023 = 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_2023_eq_2_l850_85020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_currant_weight_approx_l850_85015

/-- The density of water in kg/m³ -/
def water_density : ℝ := 1000

/-- The volume of the bucket in m³ -/
def bucket_volume : ℝ := 0.01

/-- The packing efficiency of currants -/
def packing_efficiency : ℝ := 0.74

/-- The weight of currants in the bucket -/
def currant_weight : ℝ := water_density * bucket_volume * packing_efficiency

theorem currant_weight_approx :
  ∃ ε > 0, |currant_weight - 7.4| < ε := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_currant_weight_approx_l850_85015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_remainder_is_32_l850_85074

/-- The polynomial f(x) = x^4 - 8x^3 + 15x^2 + 22x - 24 -/
def f (x : ℝ) : ℝ := x^4 - 8*x^3 + 15*x^2 + 22*x - 24

/-- The remainder when f(x) is divided by (x-2) is equal to f(2) -/
theorem remainder_theorem (x : ℝ) : 
  ∃ q : ℝ → ℝ, f x = (x - 2) * q x + f 2 := by sorry

/-- The remainder when x^4 - 8x^3 + 15x^2 + 22x - 24 is divided by x-2 is 32 -/
theorem remainder_is_32 : f 2 = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_remainder_is_32_l850_85074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_card_draw_probability_l850_85010

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the event of drawing two cards -/
def TwoCardDraw (d : Deck) : Finset (Fin 52 × Fin 52) :=
  d.cards.product d.cards

/-- Predicate for a card being a spade -/
def IsSpade (card : Fin 52) : Bool := sorry

/-- Predicate for a card being an Ace -/
def IsAce (card : Fin 52) : Bool := sorry

/-- The event where the first card is either a spade or an Ace, and the second is an Ace -/
def TargetEvent (d : Deck) : Finset (Fin 52 × Fin 52) :=
  (TwoCardDraw d).filter (fun (c1, c2) => (IsSpade c1 || IsAce c1) && IsAce c2)

/-- The probability of the target event -/
noncomputable def EventProbability (d : Deck) : ℚ :=
  (TargetEvent d).card / (TwoCardDraw d).card

theorem two_card_draw_probability (d : Deck) :
  EventProbability d = 5 / 221 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_card_draw_probability_l850_85010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_marked_sides_ge_one_l850_85004

/-- Represents a rectangle within a unit square -/
structure Rectangle where
  width : ℝ
  height : ℝ
  marked_side : ℝ
  h_width : 0 < width ∧ width ≤ 1
  h_height : 0 < height ∧ height ≤ 1
  h_marked : marked_side = width ∨ marked_side = height

/-- Represents a partition of the unit square into rectangles -/
def UnitSquarePartition := List Rectangle

/-- The sum of areas of rectangles in a partition equals 1 -/
def valid_partition (p : UnitSquarePartition) : Prop :=
  (p.map (fun r => r.width * r.height)).sum = 1

/-- The sum of marked sides in a valid partition is at least 1 -/
theorem sum_marked_sides_ge_one (p : UnitSquarePartition) (h : valid_partition p) :
  (p.map (fun r => r.marked_side)).sum ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_marked_sides_ge_one_l850_85004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l850_85048

-- Define the circle and points
def Point : Type := ℝ × ℝ

-- Define the circle with radius 3 and center at the origin
def circleEq (p : Point) : Prop := p.1^2 + p.2^2 = 9

-- Define the points
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry
noncomputable def E : Point := sorry
def O : Point := (0, 0)  -- Center of the circle

-- State the conditions
axiom AB_diameter : A.1 = -B.1 ∧ A.2 = -B.2
axiom B_on_circle : circleEq B
axiom BD_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 4
axiom ED_length : Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = 8
axiom ED_perpendicular_AD : (E.1 - D.1) * (A.1 - D.1) + (E.2 - D.2) * (A.2 - D.2) = 0
axiom C_on_AE : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = (t * A.1 + (1 - t) * E.1, t * A.2 + (1 - t) * E.2)
axiom C_on_circle : circleEq C

-- State the theorem
theorem area_of_triangle_ABC :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) *
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) / 2 = 784 / 113 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l850_85048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l850_85001

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) - 1

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  (∀ x, f x ≤ 1) ∧
  (∀ x, f x = 0 ↔ ∃ k : ℤ, x = k * Real.pi ∨ x = Real.pi / 3 + k * Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l850_85001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buckets_for_original_capacity_l850_85034

/-- Represents the number of buckets needed to fill a tank -/
def buckets_needed (bucket_capacity : ℚ) : ℕ := sorry

/-- The capacity of the tank in liters -/
def tank_capacity : ℚ := sorry

/-- Original bucket capacity in liters -/
def original_capacity : ℚ := sorry

/-- Reduced bucket capacity is 2/5 of the original capacity -/
axiom reduced_capacity : buckets_needed (2/5 * original_capacity) = 105

/-- The tank capacity remains constant regardless of bucket size -/
axiom constant_tank_capacity :
  buckets_needed original_capacity * original_capacity =
  buckets_needed (2/5 * original_capacity) * (2/5 * original_capacity)

theorem buckets_for_original_capacity :
  buckets_needed original_capacity = 42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buckets_for_original_capacity_l850_85034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_test_all_residents_l850_85064

-- Define the function for residents tested
noncomputable def residents_tested (x : ℝ) : ℝ :=
  if x ≤ 10 then 0 else 30 * x - 300

-- Theorem statement
theorem time_to_test_all_residents :
  ∃ x : ℝ, x > 10 ∧ residents_tested x = 6000 ∧ x = 210 := by
  -- Provide the existence of x
  use 210
  -- Split the goal into three parts
  constructor
  · -- Prove x > 10
    linarith
  constructor
  · -- Prove residents_tested x = 6000
    simp [residents_tested]
    norm_num
  · -- Prove x = 210
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_test_all_residents_l850_85064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l850_85026

noncomputable def f (x : ℝ) : ℝ := (3*x^2 - 2*x + 1) / ((x+1)*(x-3)) - (5 + 2*x) / ((x+1)*(x-3))

noncomputable def g (x : ℝ) : ℝ := 3*(x-2) / (x-3)

theorem problem_solution :
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 3 → f x = g x) ∧
  f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l850_85026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l850_85067

-- Define the points M and N
def M : ℝ × ℝ := (0, -2)
def N : ℝ × ℝ := (0, 2)

-- Define the moving point P
def P : ℝ × ℝ → Prop
  | (x, y) => Real.sqrt (x^2 + y^2 + 4*y + 4) + Real.sqrt (x^2 + y^2 - 4*y + 4) = 10

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_property (x y : ℝ) :
  P (x, y) → distance (x, y) M = 7 → distance (x, y) N = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l850_85067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l850_85070

/-- Parabola C: y² = x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

/-- Point M on the parabola -/
def M : ℝ × ℝ := (1, -1)

/-- Condition that M is on the parabola -/
axiom M_on_parabola : parabola M

/-- Vector from M to a point P -/
def vector_MP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Distance from a point to a line defined by two points -/
noncomputable def distance_point_to_line (p A B : ℝ × ℝ) : ℝ :=
  let a := B.2 - A.2
  let b := A.1 - B.1
  let c := B.1 * A.2 - A.1 * B.2
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

/-- The theorem to be proved -/
theorem max_distance_to_line (A B : ℝ × ℝ) 
  (h1 : parabola A) (h2 : parabola B)
  (h3 : dot_product (vector_MP A) (vector_MP B) = 0) :
  ∃ (max_dist : ℝ), 
    (∀ (P Q : ℝ × ℝ), parabola P → parabola Q → 
      dot_product (vector_MP P) (vector_MP Q) = 0 → 
      distance_point_to_line M P Q ≤ max_dist) ∧
    max_dist = Real.sqrt 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l850_85070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_is_106_l850_85019

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem median_is_106 : 
  let N := sequence_sum 150
  let median_position := (N + 1) / 2
  ∃ n : ℕ, n = 106 ∧ 
    sequence_sum (n - 1) < median_position ∧ 
    sequence_sum n ≥ median_position := by
  sorry -- Proof to be filled in

#eval sequence_sum 150 -- Should output 11325
#eval (sequence_sum 150 + 1) / 2 -- Should output 5663
#eval sequence_sum 105 -- Should output 5565
#eval sequence_sum 106 -- Should output 5671

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_is_106_l850_85019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_comparison_l850_85007

/-- Given a positive real number representing the common perimeter, 
    calculates the area of a circle with that perimeter. -/
noncomputable def circle_area (perimeter : ℝ) : ℝ := perimeter^2 / (4 * Real.pi)

/-- Given a positive real number representing the common perimeter, 
    calculates the area of a square with that perimeter. -/
noncomputable def square_area (perimeter : ℝ) : ℝ := perimeter^2 / 16

/-- Given a positive real number representing the common perimeter, 
    calculates the area of an equilateral triangle with that perimeter. -/
noncomputable def triangle_area (perimeter : ℝ) : ℝ := (Real.sqrt 3 * perimeter^2) / 36

/-- Theorem stating that for shapes with equal perimeters, 
    the circle has the largest area and the equilateral triangle has the smallest. -/
theorem area_comparison (perimeter : ℝ) (h : perimeter > 0) :
  circle_area perimeter > square_area perimeter ∧
  square_area perimeter > triangle_area perimeter :=
by sorry

#check area_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_comparison_l850_85007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_properties_l850_85036

/-- Geometric series sum -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem for geometric series properties -/
theorem geometric_series_properties
  (a₁ : ℝ) (q : ℝ) (h_pos : a₁ > 0 ∧ q > 0) :
  let S := geometric_sum a₁ q
  (∀ n : ℕ, (Real.log (S n) + Real.log (S (n + 2))) / 2 < Real.log (S (n + 1))) ∧
  (¬ ∃ c : ℝ, c > 0 ∧ ∀ n : ℕ, (Real.log (S n - c) + Real.log (S (n + 2) - c)) / 2 = Real.log (S (n + 1) - c)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_properties_l850_85036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_l850_85016

theorem max_value_of_sequence (x : Fin 1996 → ℝ) 
  (h_positive : ∀ i, x i > 0)
  (h_equal_ends : x 0 = x 1995)
  (h_recurrence : ∀ i : Fin 1995, x i.val + 2 / x i.val = 2 * x (i.val + 1) + 1 / x (i.val + 1)) :
  x 0 ≤ 2^997 ∧ ∃ x' : Fin 1996 → ℝ, 
    x' 0 = 2^997 ∧
    (∀ i, x' i > 0) ∧
    x' 0 = x' 1995 ∧
    (∀ i : Fin 1995, x' i.val + 2 / x' i.val = 2 * x' (i.val + 1) + 1 / x' (i.val + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_l850_85016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_number_l850_85079

theorem unique_four_digit_number : 
  ∃! N : ℕ, 
    1000 ≤ N ∧ N < 10000 ∧ 
    (N % 1000 = N / 7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_number_l850_85079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_cyclic_l850_85040

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Four circles, each touching externally two out of the three others --/
structure FourTouchingCircles where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  c4 : Circle
  touch_12 : dist c1.center c2.center = c1.radius + c2.radius
  touch_23 : dist c2.center c3.center = c2.radius + c3.radius
  touch_34 : dist c3.center c4.center = c3.radius + c4.radius
  touch_41 : dist c4.center c1.center = c4.radius + c1.radius

/-- The point of tangency between two circles --/
noncomputable def tangencyPoint (c1 c2 : Circle) : ℝ × ℝ :=
  sorry

/-- Theorem: The points of tangency of four externally touching circles form a cyclic quadrilateral --/
theorem tangency_points_cyclic (fc : FourTouchingCircles) :
  ∃ (c : Circle), 
    dist c.center (tangencyPoint fc.c1 fc.c2) = c.radius ∧
    dist c.center (tangencyPoint fc.c2 fc.c3) = c.radius ∧
    dist c.center (tangencyPoint fc.c3 fc.c4) = c.radius ∧
    dist c.center (tangencyPoint fc.c4 fc.c1) = c.radius :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_cyclic_l850_85040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoid_highest_point_l850_85058

/-- Given a sinusoidal function with a specific initial phase, 
    prove the coordinates of its highest point. -/
theorem sinusoid_highest_point (θ : ℝ) (k : ℤ) :
  (∀ x, Real.sin (1/4 * x - θ) = Real.sin (1/4 * x - π/6)) →
  (∃ x, Real.sin (1/4 * x - π/6) = 1 ∧ x = 8*π/3 + 8*π*(k:ℝ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoid_highest_point_l850_85058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l850_85081

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points in the space
variable (E F G H : V)

-- Define the property of points being coplanar
def coplanar (p q r s : V) : Prop := ∃ (a b c d : ℝ), a • p + b • q + c • r + d • s = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)

-- Define the property of lines intersecting
def lines_intersect (p1 p2 q1 q2 : V) : Prop := ∃ (t s : ℝ), p1 + t • (p2 - p1) = q1 + s • (q2 - q1)

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ E F G H : V, ¬ coplanar E F G H → ¬ lines_intersect E F G H) ∧
  (∃ E F G H : V, ¬ lines_intersect E F G H ∧ coplanar E F G H) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l850_85081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_calculation_l850_85009

noncomputable def partnership_profit (a_investment b_investment : ℝ) (a_period b_period : ℝ) (total_profit : ℝ) : ℝ :=
  let ratio_b := b_investment * b_period
  let ratio_a := a_investment * a_period
  let total_ratio := ratio_b + ratio_a
  (ratio_b / total_ratio) * total_profit

theorem b_profit_calculation (b_investment : ℝ) (b_period : ℝ) (total_profit : ℝ) :
  partnership_profit (3 * b_investment) b_investment (2 * b_period) b_period total_profit = 7000 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_calculation_l850_85009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bijective_f_inverse_composition_f_inverse_triple_composition_l850_85002

def f : Fin 6 → Fin 6
  | 1 => 4
  | 2 => 1
  | 3 => 6
  | 4 => 3
  | 5 => 2
  | 6 => 5

theorem f_bijective : Function.Bijective f := by sorry

theorem f_inverse_composition (x : Fin 6) : Function.invFun f (f x) = x := by sorry

theorem f_inverse_triple_composition :
  Function.invFun f (Function.invFun f (Function.invFun f 1)) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bijective_f_inverse_composition_f_inverse_triple_composition_l850_85002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_a_range_l850_85043

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x - Real.log x

-- Part 1
theorem monotonicity_intervals (x : ℝ) :
  let f₂ := f 2
  (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f₂ x₁ < f₂ x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f₂ x₁ > f₂ x₂) := by
  sorry

-- Part 2
theorem a_range (a : ℝ) :
  a ≤ -1/4 →
  (∀ x, x ∈ Set.Icc 2 (Real.exp 1) → f a x ≥ -Real.log 2) →
  a ∈ Set.Icc (-4) (-1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_a_range_l850_85043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l850_85061

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 2^x + x^2 + (-1) else -(2^(-x) + (-x)^2 + (-1))

theorem odd_function_value (h : ∀ x, f (-x) = -f x) : f 2 = -13/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l850_85061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l850_85025

/-- The function g(x) defined by (x^2 - 3x + c) / (x^2 - 4x + 3) -/
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + c) / (x^2 - 4*x + 3)

/-- The proposition that g(x) has exactly one vertical asymptote -/
def has_one_vertical_asymptote (c : ℝ) : Prop :=
  ∃! x : ℝ, (x^2 - 4*x + 3 = 0) ∧ (x^2 - 3*x + c ≠ 0)

/-- Theorem stating that g(x) has exactly one vertical asymptote iff c = 0 or c = 2 -/
theorem g_one_vertical_asymptote (c : ℝ) :
  has_one_vertical_asymptote c ↔ (c = 0 ∨ c = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l850_85025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_inequality_l850_85035

/-- Predicate to assert that x, y, z are the lengths of angle bisectors of a triangle -/
def IsAngleBisectorLengths (x y z : ℝ) : Prop := sorry

/-- Function to calculate the perimeter of a triangle given its angle bisector lengths -/
def TrianglePerimeter (x y z : ℝ) : ℝ := sorry

/-- Given a triangle with perimeter 6 and angle bisectors of lengths x, y, and z,
    the sum of the reciprocals of their squares is greater than or equal to 1. -/
theorem angle_bisector_inequality (x y z : ℝ) 
    (h_bisectors : IsAngleBisectorLengths x y z)
    (h_perimeter : TrianglePerimeter x y z = 6) :
    1 / x^2 + 1 / y^2 + 1 / z^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_inequality_l850_85035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_coins_collected_l850_85053

def coin_collection (n : ℕ) : ℕ :=
  12 + 10 * (n - 1)

def total_coins (days : ℕ) : ℚ :=
  let regular_days := (Finset.range (days - 2)).sum (λ i => coin_collection (i + 1))
  let day_7_coins := (coin_collection 7) / 2
  let day_8_coins := coin_collection 8
  ↑regular_days + 12 + ↑day_7_coins + ↑day_8_coins

theorem average_coins_collected (days : ℕ) (h : days = 8) :
  (total_coins days) / days = 85/2 := by
  sorry

#eval (total_coins 8) / 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_coins_collected_l850_85053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_72_112_l850_85069

theorem least_multiple_72_112 : ∃! n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → 72 * m % 112 = 0 → n ≤ m) ∧ 72 * n % 112 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_72_112_l850_85069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_l850_85029

/-- Given a function f(x) = x^3 - f'(1)x^2 + x + 5, prove that f'(1) = 2 -/
theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - (deriv f 1) * x^2 + x + 5) : 
  deriv f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_l850_85029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l850_85044

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1)^2

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (f x - 2*x) / x

-- Theorem statement
theorem function_properties :
  (∀ x y : ℝ, f (x + y) - f y = (x + 2*y - 2) * x) ∧
  (f 1 = 0) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → g (2^x) - k * 2^x ≤ 0) ↔ k ≥ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l850_85044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_calculation_l850_85098

/-- The initially calculated average height of boys in a class -/
noncomputable def initialAverage (n : ℕ) (actualAverage : ℝ) (incorrectHeight correctHeight : ℝ) : ℝ :=
  (n * actualAverage + incorrectHeight - correctHeight) / n

theorem initial_average_calculation (n : ℕ) (actualAverage incorrectHeight correctHeight : ℝ)
    (hn : n = 35)
    (hactual : actualAverage = 180)
    (hincorrect : incorrectHeight = 166)
    (hcorrect : correctHeight = 106) :
    initialAverage n actualAverage incorrectHeight correctHeight = 6360 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_calculation_l850_85098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l850_85027

noncomputable def f (x : ℝ) := 2 * Real.sin (x + Real.pi / 4)

theorem phase_shift_of_f :
  ∃ (shift : ℝ), ∀ (x : ℝ),
    f x = 2 * Real.sin (x - shift) ∧ shift = -Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l850_85027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_tangent_lines_l850_85097

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the condition for point M
def condition_M (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  ((x + 1)^2 + y^2) / ((x - 2)^2 + y^2) = 1/4

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- Define point P
def P : ℝ × ℝ := (0, 2)

-- Define the maximum value of |MP|
noncomputable def max_MP : ℝ := 2 + 2 * Real.sqrt 2

-- Theorem statement
theorem curve_and_tangent_lines :
  (∀ M : ℝ × ℝ, condition_M M ↔ curve_C M.1 M.2) ∧
  (∀ x y : ℝ, curve_C x y → (y = 2 ∨ x = 0) ↔ 
    ((x - P.1)^2 + (y - P.2)^2 ≤ max_MP^2 ∧ 
     ∃ t : ℝ, curve_C (P.1 + t) (P.2 + t) ∧ 
              (x - P.1)^2 + (y - P.2)^2 = ((P.1 + t) - P.1)^2 + ((P.2 + t) - P.2)^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_tangent_lines_l850_85097
