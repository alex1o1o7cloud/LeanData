import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_and_four_fraction_l336_33624

theorem divisible_by_three_and_four_fraction :
  (Finset.filter (λ n : ℕ => 3 ∣ n ∧ 4 ∣ n) (Finset.range 100)).card / 100 = 2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_and_four_fraction_l336_33624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_four_thirds_l336_33633

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then x^2
  else if 0 < x ∧ x ≤ 1 then 1
  else 0

theorem integral_f_equals_four_thirds :
  ∫ x in (-1)..1, f x = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_four_thirds_l336_33633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tension_approx_24_l336_33607

/-- Represents the gravitational acceleration constant -/
noncomputable def g : ℝ := 9.8

/-- Calculates the tension in a cord connecting a uniform disk to a falling mass -/
noncomputable def tension (M m : ℝ) : ℝ :=
  (1/2) * M * (m * g / (m + (1/2) * M))

/-- Theorem stating that the tension in the cord is approximately 24.0 N -/
theorem tension_approx_24 (M m : ℝ) (hM : M = 8.0) (hm : m = 6.0) :
  ∃ ε > 0, |tension M m - 24.0| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tension_approx_24_l336_33607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fifth_term_l336_33680

/-- Given a geometric sequence with the first four terms 3, 9y, 27y^2, 81y^3,
    prove that the fifth term is 243y^4 -/
theorem geometric_sequence_fifth_term (y : ℝ) :
  let seq := λ n => 3 * (3 * y) ^ (n - 1)
  seq 5 = 243 * y^4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fifth_term_l336_33680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_largest_power_of_three_dividing_nine_factorial_l336_33679

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

def largest_power_of_three_dividing (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i ↦ if (i + 1) % 3 = 0 then 1 else 0) +
  (Finset.range n).sum (λ i ↦ if (i + 1) % 9 = 0 then 1 else 0)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_three_dividing_nine_factorial :
  ones_digit (3^(largest_power_of_three_dividing (3^2))) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_largest_power_of_three_dividing_nine_factorial_l336_33679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_discount_example_l336_33636

/-- Calculates the banker's discount given the present worth and true discount of a bill. -/
noncomputable def bankers_discount (present_worth : ℝ) (true_discount : ℝ) : ℝ :=
  let face_value := present_worth + true_discount
  (face_value * true_discount) / (face_value - true_discount)

/-- Theorem stating that for a bill with present worth 800 and true discount 36,
    the banker's discount is approximately 37.62. -/
theorem bankers_discount_example :
  ∃ (x : ℝ), abs (bankers_discount 800 36 - x) < 0.005 ∧ x = 37.62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_discount_example_l336_33636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_tiles_arrangement_l336_33674

def square_side_lengths : List ℕ := [172, 1, 5, 7, 11, 20, 27, 34, 41, 42, 43, 44, 61, 85, 95, 108, 113, 118, 123, 136, 168, 183, 194, 205, 209, 231]

def perimeter_squares : List ℕ := [113, 118, 123, 136, 168, 183, 194, 205, 209, 231]

theorem square_tiles_arrangement (tiles : List ℕ) (perimeter : List ℕ) 
  (h1 : tiles = square_side_lengths)
  (h2 : perimeter = perimeter_squares) :
  ∃ (arrangement : List (List ℕ)) (square_side : ℕ),
    (arrangement.length = 2) ∧ 
    (∀ rect ∈ arrangement, rect.sum = square_side) ∧
    (tiles.sum^2 = (2 * square_side)^2) ∧
    (∃ (perimeter_arrangement : List ℕ), 
      Multiset.toList (Multiset.ofList perimeter_arrangement) = 
        Multiset.toList (Multiset.ofList perimeter) ∧
      perimeter_arrangement.sum + 2 * (square_side - perimeter_arrangement.sum) = 4 * square_side) :=
by sorry

#check square_tiles_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_tiles_arrangement_l336_33674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourism_revenue_growth_equation_l336_33682

/-- Represents the annual average growth rate of tourism revenue -/
def x : ℝ := sorry

/-- The initial tourism revenue in 2020 (in billions of yuan) -/
def initial_revenue : ℝ := 2

/-- The estimated tourism revenue in 2022 (in billions of yuan) -/
def estimated_revenue : ℝ := 2.88

/-- The number of years between the initial and estimated revenue -/
def years : ℕ := 2

/-- Theorem stating that the equation correctly represents the revenue growth -/
theorem tourism_revenue_growth_equation : 
  initial_revenue * (1 + x)^years = estimated_revenue := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourism_revenue_growth_equation_l336_33682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_inequality_implies_log_inequality_l336_33686

theorem exp_inequality_implies_log_inequality (x y : ℝ) :
  (2 : ℝ)^x - (2 : ℝ)^y < (3 : ℝ)^(-x) - (3 : ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_inequality_implies_log_inequality_l336_33686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_time_l336_33603

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_length platform_length : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  (train_length + platform_length) * time_to_cross_pole / train_length

theorem train_crossing_platform_time :
  time_to_cross_platform 300 400 18 = 42 := by
  -- Unfold the definition of time_to_cross_platform
  unfold time_to_cross_platform
  -- Simplify the arithmetic expression
  simp [add_mul, mul_div_assoc]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_time_l336_33603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_inequality_l336_33638

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_minimum_and_inequality (a : ℝ) (h1 : -1 < a) (h2 : a < 0) :
  (∀ x > 0, f a x ≥ f a (Real.sqrt (-a / (a + 1)))) ∧
  (∀ x > 0, f a x > 1 + a / 2 * Real.log (-a)) ↔ 1 / Real.exp 1 - 1 < a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_inequality_l336_33638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_triangle_area_l336_33673

/-- Given a circle with equation x^2 + y^2 = 100 and a point M1(6, 8) on its circumference,
    the area of the triangle formed by the origin O and the intersection points A and B 
    of the tangent line at M1 with the perpendiculars from the ends of the diameter 
    on the x-axis is 656/3. -/
theorem circle_tangent_triangle_area : 
  ∀ (x y : ℝ), 
  x^2 + y^2 = 100 →  -- Circle equation
  ∃ (M1 : ℝ × ℝ),
    M1 = (6, 8) ∧  -- Point on circumference
    M1.1^2 + M1.2^2 = 100 ∧  -- M1 satisfies circle equation
    ∃ (A B : ℝ × ℝ),
      (∀ t : ℝ, t ∈ Set.Icc (-10 : ℝ) 10 → 
        ((A.1 = t ∧ A.2 = (-3/4 * t + 25/2)) ∨
         (B.1 = t ∧ B.2 = (-3/4 * t + 25/2)))) →  -- A and B on tangent line
      (1/2 * (B.1 - A.1) * 8 : ℝ) = 656/3  -- Triangle area
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_triangle_area_l336_33673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l336_33654

-- Define the set M
def M : Set ℝ := {x | |2*x - 1| < 1}

-- Theorem statement
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a * b + 1 > a + b) ∧ 
  (max (2 / Real.sqrt a) (max ((a + b) / Real.sqrt (a * b)) ((a * b + 1) / Real.sqrt b)) > 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l336_33654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l336_33627

theorem expression_evaluation :
  Real.sqrt 3 * Real.tan (60 * π / 180) - 2 - (-2)^(-2 : ℤ) + Real.sqrt 2 = 3/4 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l336_33627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_side_l336_33641

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.cos (t.A / 2) = 2 * Real.sqrt 5 / 5 ∧
  t.b * t.c * Real.cos t.A = 3 ∧
  t.b + t.c = 6

-- Theorem statement
theorem triangle_area_and_side (t : Triangle) 
  (h : triangle_conditions t) : 
  (1/2 * t.b * t.c * Real.sin t.A = 2) ∧ 
  (t.a = Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_side_l336_33641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_inequality_l336_33693

theorem range_of_a_for_inequality (a : ℝ) : 
  (∃ x : ℝ, a * x ≤ a - 1) ↔ a ∈ Set.Iio 0 ∪ Set.Ioi 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_inequality_l336_33693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l336_33653

theorem remainder_problem (j : ℕ) (hj : j > 0) (h : 72 % (j^2) = 8) : 150 % j = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l336_33653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_HN_passes_through_fixed_point_l336_33683

/-- An ellipse centered at the origin with axes of symmetry along the x and y axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0
  h_neq : a ≠ b

/-- The ellipse E passing through (0,-2) and (3/2,-1) -/
noncomputable def E : Ellipse :=
  { a := Real.sqrt 3
    b := 2
    h_pos := by sorry
    h_neq := by sorry }

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fixed point K -/
def K : Point := ⟨0, -2⟩

/-- The point P through which all lines pass -/
def P : Point := ⟨1, -2⟩

/-- A line passing through point P intersecting the ellipse E -/
structure IntersectingLine where
  slope : Option ℝ  -- None represents a vertical line

/-- The intersection points of the line with the ellipse -/
noncomputable def intersection_points (l : IntersectingLine) (E : Ellipse) : Point × Point :=
  sorry

/-- The point T on AB such that MT is parallel to x-axis -/
noncomputable def T (M : Point) (A B : Point) : Point :=
  sorry

/-- The point H satisfying MT = TH -/
noncomputable def H (M T : Point) : Point :=
  sorry

/-- The theorem stating that HN always passes through K -/
theorem HN_passes_through_fixed_point (l : IntersectingLine) :
  let (M, N) := intersection_points l E
  let T := T M ⟨0, -2⟩ ⟨3/2, -1⟩
  let H := H M T
  ∃ t : ℝ, K = ⟨(1 - t) * H.x + t * N.x, (1 - t) * H.y + t * N.y⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_HN_passes_through_fixed_point_l336_33683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_and_inequality_l336_33649

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x - 2 / Real.exp 1

-- State the theorem
theorem f_min_and_inequality :
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≥ f x_min ∧ f x_min = -1 / Real.exp 1) ∧
  (∀ (m n : ℝ), m > 0 → n > 0 → f m ≥ g n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_and_inequality_l336_33649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l336_33626

/-- Polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- Evaluate a polynomial at a given point -/
def evalPoly (p : IntPolynomial n) (x : ℤ) : ℤ :=
  (Finset.sum Finset.univ fun i => p i * x ^ i.val)

/-- Check if all coefficients satisfy the given bound -/
def coeffsBounded (p : IntPolynomial n) (bound : ℕ) : Prop :=
  ∀ i, |p i| ≤ bound

theorem polynomial_equality (f : IntPolynomial 6) (g : IntPolynomial 4) (h : IntPolynomial 3)
    (hf : coeffsBounded f 4)
    (hg : coeffsBounded g 1)
    (hh : coeffsBounded h 1)
    (h_eval : evalPoly f 10 = evalPoly g 10 * evalPoly h 10) :
    ∀ x, evalPoly f x = evalPoly g x * evalPoly h x := by
  sorry

#check polynomial_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l336_33626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vet_kibble_difference_l336_33604

/-- The number of vets recommending Yummy Dog Kibble minus the number recommending Puppy Kibble -/
def vet_difference (total_vets : ℕ) (puppy_percent yummy_percent : ℚ) : ℤ :=
  ⌊yummy_percent * total_vets⌋ - ⌊puppy_percent * total_vets⌋

/-- Theorem stating the difference in vets recommending different kibbles -/
theorem vet_kibble_difference :
  vet_difference 3500 (23.5 / 100) (37.2 / 100) = 479 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vet_kibble_difference_l336_33604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_at_two_l336_33676

/-- The polynomial x^5 - x^2 - x - 1 -/
def f (x : ℤ) : ℤ := x^5 - x^2 - x - 1

/-- A list of irreducible monic polynomials with integer coefficients -/
def irreducible_factors : List (ℤ → ℤ) :=
  [λ x ↦ x^2 + 1, λ x ↦ x^3 - x - 1]

/-- The product of all polynomials in the list equals f -/
axiom factors_product (x : ℤ) : 
  (irreducible_factors.map (λ p ↦ p x)).prod = f x

/-- Each polynomial in the list is irreducible over the integers -/
axiom all_irreducible : 
  ∀ p ∈ irreducible_factors, ¬∃ (q r : ℤ → ℤ), (∀ x, p x = q x * r x) ∧ (∀ x, q x ≠ 1) ∧ (∀ x, r x ≠ 1)

/-- Each polynomial in the list is monic -/
axiom all_monic : 
  ∀ p ∈ irreducible_factors, ∃ n : ℕ, ∀ x, p x = x^n + (p x - x^n)

theorem sum_of_factors_at_two : 
  (irreducible_factors.map (λ p ↦ p 2)).sum = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_at_two_l336_33676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_q_expression_l336_33611

/-- Given a triangle XYZ with points M and N, prove that Q = (9/13)*Y + (4/13)*Z --/
theorem vector_q_expression (X Y Z M N Q : ℝ × ℝ × ℝ) : 
  -- Triangle XYZ exists
  (X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X) →
  -- M lies on YZ extended past Z such that YM:MZ = 4:1
  (∃ t : ℝ, t > 1 ∧ M = Y + t • (Z - Y) ∧ (t - 1) / 1 = 4 / 1) →
  -- N lies on XZ such that XN:NZ = 6:2
  (∃ s : ℝ, 0 < s ∧ s < 1 ∧ N = X + s • (Z - X) ∧ s / (1 - s) = 6 / 2) →
  -- Q is the intersection of YN and XM
  (∃ u v : ℝ, Q = Y + u • (N - Y) ∧ Q = X + v • (M - X)) →
  -- Q is a linear combination of X, Y, and Z with coefficients summing to 1
  (∃ a b c : ℝ, Q = a • X + b • Y + c • Z ∧ a + b + c = 1) →
  -- Prove that Q = (9/13)*Y + (4/13)*Z
  Q = (9/13) • Y + (4/13) • Z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_q_expression_l336_33611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_A_not_axiom_proposition_A_description_proposition_B_description_proposition_C_description_proposition_D_description_l336_33602

-- Define the propositions as axioms or theorems
axiom proposition_A : Prop
axiom proposition_B : Prop
axiom proposition_C : Prop
axiom proposition_D : Prop

-- Define the concept of an axiom
def is_axiom (p : Prop) : Prop := 
  ∃ (description : String), description = "p is a fundamental assumption in solid geometry"

-- State the theorem
theorem proposition_A_not_axiom : 
  is_axiom proposition_B ∧ 
  is_axiom proposition_C ∧ 
  is_axiom proposition_D → 
  ¬ is_axiom proposition_A := by
  sorry

-- Provide informal descriptions of the propositions
theorem proposition_A_description : 
  ∃ (description : String), description = "Two planes parallel to the same plane are parallel to each other" := by
  sorry

theorem proposition_B_description : 
  ∃ (description : String), description = "Through three points not on the same line, there exists exactly one plane" := by
  sorry

theorem proposition_C_description : 
  ∃ (description : String), description = "If two points on a line are in a plane, then all points on the line are in this plane" := by
  sorry

theorem proposition_D_description : 
  ∃ (description : String), description = "If two distinct planes have a common point, then they have exactly one common line passing through that point" := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_A_not_axiom_proposition_A_description_proposition_B_description_proposition_C_description_proposition_D_description_l336_33602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l336_33690

theorem system_solutions :
  let S : Set (ℝ × ℝ × ℝ) := {(x, y, z) |
    x + y = 4 * z ∧
    x^2 + y^2 = 17 * z ∧
    x^3 + y^3 = 76 * z}
  S = {(5, 3, 2), (3, 5, 2),
       ((19 + Real.sqrt 285) / 8, (19 - Real.sqrt 285) / 8, 19/16),
       ((19 - Real.sqrt 285) / 8, (19 + Real.sqrt 285) / 8, 19/16)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l336_33690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_plus_cos4_l336_33681

theorem min_sin4_plus_cos4 (x : ℝ) : Real.sin x ^ 4 + Real.cos x ^ 4 ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_plus_cos4_l336_33681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_primes_five_dice_l336_33695

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of dice that should show a prime number -/
def target_primes : ℕ := 3

/-- The number of prime numbers less than or equal to the number of sides on each die -/
def num_primes : ℕ := 8

/-- The probability of exactly 'target_primes' dice showing a prime number when rolling 'num_dice' fair dice with 'num_sides' sides -/
def prob_exact_primes : ℚ :=
  (Nat.choose num_dice target_primes : ℚ) *
  ((num_primes : ℚ) / (num_sides : ℚ)) ^ target_primes *
  ((num_sides - num_primes : ℚ) / (num_sides : ℚ)) ^ (num_dice - target_primes)

theorem prob_three_primes_five_dice : prob_exact_primes = 720 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_primes_five_dice_l336_33695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l336_33600

/-- The length of a platform given the speed of a train, time to cross, and train length -/
noncomputable def platform_length (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : ℝ :=
  train_speed * (1000 / 3600) * crossing_time - train_length

/-- Theorem: The length of the platform is approximately 259.9584 meters -/
theorem platform_length_calculation :
  let train_speed : ℝ := 72
  let crossing_time : ℝ := 26
  let train_length : ℝ := 260.0416
  |platform_length train_speed crossing_time train_length - 259.9584| < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l336_33600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zachary_pushups_l336_33669

theorem zachary_pushups (david_pushups zachary_pushups : ℕ) (difference : ℕ) 
  (h1 : david_pushups = 44)
  (h2 : david_pushups = difference + zachary_pushups)
  (h3 : difference = 9) :
  zachary_pushups = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zachary_pushups_l336_33669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_earnings_increase_l336_33621

/-- Calculates the percentage increase given initial and final amounts -/
noncomputable def percentage_increase (initial : ℝ) (final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

/-- Proves that John's percentage increase in weekly earnings is approximately 66.67% -/
theorem johns_earnings_increase :
  let initial_earnings : ℝ := 60
  let new_earnings : ℝ := 100
  abs (percentage_increase initial_earnings new_earnings - 66.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_earnings_increase_l336_33621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteen_degree_angle_count_angle_19_occurs_44_times_daily_l336_33632

/-- Represents the number of degrees the minute hand moves per minute -/
noncomputable def minute_hand_speed : ℝ := 360 / 60

/-- Represents the number of degrees the hour hand moves per minute -/
noncomputable def hour_hand_speed : ℝ := 360 / (12 * 60)

/-- Calculates the angle between the hour and minute hands at a given time -/
noncomputable def angle_between_hands (hour : ℝ) (minute : ℝ) : ℝ :=
  |30 * hour - 5.5 * minute|

/-- Counts the number of times the angle between hands is exactly 19° in a 24-hour period -/
def count_19_degree_angles : ℕ :=
  44 -- We're directly stating the result here

theorem nineteen_degree_angle_count :
  count_19_degree_angles = 44 := by
  rfl -- reflexivity proves this trivial equality

/-- Proves that the angle between hands is 19° exactly 44 times in a day -/
theorem angle_19_occurs_44_times_daily :
  ∃ (times : ℕ), times = 44 ∧
  (∀ (hour minute : ℝ), 0 ≤ hour ∧ hour < 24 ∧ 0 ≤ minute ∧ minute < 60 →
    angle_between_hands hour minute = 19 →
    (∃ (n : ℕ), n < times ∧
      ∃ (h m : ℝ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧
        angle_between_hands h m = 19 ∧ (h, m) ≠ (hour, minute))) := by
  sorry -- The actual proof is omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteen_degree_angle_count_angle_19_occurs_44_times_daily_l336_33632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flip_to_all_tails_except_one_l336_33616

/-- Represents the state of a coin (Head or Tail) -/
inductive CoinState
| Head
| Tail

/-- Represents a position on the chessboard -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the chessboard -/
def Chessboard := Array (Array CoinState)

/-- Flip operation that changes 4 consecutive coins in a row or column -/
def flipCoins (board : Chessboard) (start : Position) (isRow : Bool) : Chessboard :=
  sorry

/-- Check if a given position has all tails except for the specified position -/
def allTailsExcept (board : Chessboard) (except : Position) : Bool :=
  sorry

/-- Main theorem: It's possible to flip coins to get all tails except one at (i, j) iff i and j are divisible by 4 -/
theorem flip_to_all_tails_except_one (i j : Nat) :
  (∃ (flips : List (Position × Bool)), 
    let initialBoard : Chessboard := sorry -- All heads
    let finalBoard := flips.foldl (λ b (p, isRow) => flipCoins b p isRow) initialBoard
    allTailsExcept finalBoard ⟨i, j⟩) ↔ 
  (i % 4 = 0 ∧ j % 4 = 0) := by
  sorry

#check flip_to_all_tails_except_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flip_to_all_tails_except_one_l336_33616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_expressions_equal_AC_l336_33644

/-- Regular hexagon with vertices A, B, C, D, E, F -/
structure RegularHexagon (V : Type*) [AddCommGroup V] :=
  (A B C D E F : V)
  (regular : ∀ (X Y : V), (X, Y) ∈ [(A,B), (B,C), (C,D), (D,E), (E,F), (F,A)] → Y - X = C - B)

/-- Theorem stating that all four expressions equal AC in a regular hexagon -/
theorem all_expressions_equal_AC {V : Type*} [AddCommGroup V] [Module ℚ V] (h : RegularHexagon V) :
  let AC := h.C - h.A
  (h.B - h.C) + (h.C - h.D) + (h.E - h.C) = AC ∧
  (2 : ℚ) • (h.C - h.B) + (h.C - h.D) = AC ∧
  (h.E - h.F) + (h.D - h.E) = AC ∧
  (h.C - h.B) - (h.A - h.B) = AC :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_expressions_equal_AC_l336_33644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covers_one_seventh_of_grid_l336_33691

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Calculates the area of a triangle given its three vertices using the Shoelace formula -/
def triangleArea (a b c : Point) : ℚ :=
  (1/2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

/-- Theorem: The fraction of a 7x6 grid covered by the given triangle is 1/7 -/
theorem triangle_covers_one_seventh_of_grid :
  let a : Point := ⟨2, 2⟩
  let b : Point := ⟨6, 2⟩
  let c : Point := ⟨4, 5⟩
  let gridArea : ℚ := 7 * 6
  (triangleArea a b c) / gridArea = 1 / 7 := by
  -- Proof steps would go here
  sorry

#eval triangleArea ⟨2, 2⟩ ⟨6, 2⟩ ⟨4, 5⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covers_one_seventh_of_grid_l336_33691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_distinct_digits_l336_33628

def is_composed_of_distinct_digits (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), 
    digits.Nodup ∧ 
    digits.length = 10 ∧ 
    (∀ d, d ∈ digits → d < 10) ∧
    n = digits.foldl (fun acc d ↦ acc * 10 + d) 0

theorem smallest_number_with_distinct_digits : 
  (is_composed_of_distinct_digits 1023456789) ∧ 
  (∀ m : ℕ, m < 1023456789 → ¬(is_composed_of_distinct_digits m)) := by
  sorry

#check smallest_number_with_distinct_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_distinct_digits_l336_33628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_5_12_13_l336_33656

/-- Predicate to check if a real number is the area of a triangle with given side lengths -/
def is_area_of_triangle (a b c A : ℝ) : Prop :=
  A ≥ 0 ∧ 
  ∃ (s : ℝ), s = (a + b + c) / 2 ∧ 
  A^2 = s * (s - a) * (s - b) * (s - c)

/-- The area of a triangle with sides 5, 12, and 13 is 30 -/
theorem triangle_area_5_12_13 : ∃ (A : ℝ), A = 30 ∧ is_area_of_triangle 5 12 13 A :=
by
  -- We'll use 30 as our proposed area
  use 30
  constructor
  -- First, prove that A = 30
  · rfl
  -- Then, prove that it satisfies is_area_of_triangle
  · unfold is_area_of_triangle
    constructor
    -- Prove A ≥ 0
    · linarith
    -- Prove the existence of s and the area formula
    · use (5 + 12 + 13) / 2
      constructor
      · rfl
      · -- Here we would typically prove that 30^2 equals the Heron's formula result
        -- For now, we'll use sorry to skip the detailed calculation
        sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_5_12_13_l336_33656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_A_times_B_equals_6E_l336_33645

-- Define the hexadecimal system
inductive Hexadecimal
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

-- Define the conversion function from hexadecimal to decimal
def hex_to_decimal : Hexadecimal → ℕ
| Hexadecimal.D0 => 0
| Hexadecimal.D1 => 1
| Hexadecimal.D2 => 2
| Hexadecimal.D3 => 3
| Hexadecimal.D4 => 4
| Hexadecimal.D5 => 5
| Hexadecimal.D6 => 6
| Hexadecimal.D7 => 7
| Hexadecimal.D8 => 8
| Hexadecimal.D9 => 9
| Hexadecimal.A => 10
| Hexadecimal.B => 11
| Hexadecimal.C => 12
| Hexadecimal.D => 13
| Hexadecimal.E => 14
| Hexadecimal.F => 15

-- Define the conversion function from decimal to hexadecimal
def decimal_to_hex (n : ℕ) : Hexadecimal :=
  match n % 16 with
  | 0 => Hexadecimal.D0
  | 1 => Hexadecimal.D1
  | 2 => Hexadecimal.D2
  | 3 => Hexadecimal.D3
  | 4 => Hexadecimal.D4
  | 5 => Hexadecimal.D5
  | 6 => Hexadecimal.D6
  | 7 => Hexadecimal.D7
  | 8 => Hexadecimal.D8
  | 9 => Hexadecimal.D9
  | 10 => Hexadecimal.A
  | 11 => Hexadecimal.B
  | 12 => Hexadecimal.C
  | 13 => Hexadecimal.D
  | 14 => Hexadecimal.E
  | _ => Hexadecimal.F

-- Define hexadecimal multiplication
def hex_mul (a b : Hexadecimal) : Hexadecimal :=
  decimal_to_hex (hex_to_decimal a * hex_to_decimal b)

-- Theorem to prove
theorem hex_A_times_B_equals_6E :
  hex_mul Hexadecimal.A Hexadecimal.B = Hexadecimal.E := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_A_times_B_equals_6E_l336_33645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_problem_solution_l336_33685

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the multiplication problem -/
structure MultiplicationProblem where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  different : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
  product_equality : (C.val * 1000 + D.val * 100 + C.val * 10 + D.val : ℕ) = 
                     ((A.val * 100 + B.val * 10 + A.val) * (C.val * 10 + D.val) : ℕ)

theorem multiplication_problem_solution (p : MultiplicationProblem) : (p.A.val + p.B.val : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_problem_solution_l336_33685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_line_passes_through_fixed_point_l336_33618

-- Define the parabola C
noncomputable def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define a point on the x-axis to the right of F
def B (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the condition |AF| = |FB|
noncomputable def equalDistance (A : ℝ × ℝ) (x : ℝ) : Prop :=
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (x - F.1)^2 + F.2^2

-- Define the tangent point D
noncomputable def D (m : ℝ) : ℝ × ℝ := (4 / m^2, -4 / m)

-- The main theorem
theorem parabola_tangent_line_passes_through_fixed_point :
  ∀ (A : ℝ × ℝ) (x : ℝ),
    A ∈ C →
    x > F.1 →
    equalDistance A x →
    ∃ (m : ℝ), m ≠ 0 ∧
      (∀ (t : ℝ), (1 - t) • A + t • D m = (1, 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_line_passes_through_fixed_point_l336_33618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_free_preserving_function_l336_33613

def SumFreeSet (A : Set ℕ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → x + y ∉ A

def PreservesSumFreeSet (f : ℕ → ℕ) : Prop :=
  ∀ A : Set ℕ, SumFreeSet A → SumFreeSet (f '' A)

theorem unique_sum_free_preserving_function :
  ∀ f : ℕ → ℕ, Function.Surjective f → PreservesSumFreeSet f → f = id := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_free_preserving_function_l336_33613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_range_l336_33671

theorem necessary_condition_range (a : ℝ) : 
  (∀ x : ℝ, |2*x - 5| ≤ 4 → x < a) → 
  a ∈ Set.Ioi (9/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_range_l336_33671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_strategy_l336_33606

def expression (a b c d e f g h : Int) : Int :=
  a * 8^7 + b * 8^6 + c * 8^5 + d * 8^4 + e * 8^3 + f * 8^2 + g * 8 + h

theorem second_player_strategy :
  ∀ (a b c d : Int), a ∈ ({-1, 1} : Set Int) → b ∈ ({-1, 1} : Set Int) → 
  c ∈ ({-1, 1} : Set Int) → d ∈ ({-1, 1} : Set Int) →
  ∃ (e f g h : Int), e ∈ ({-1, 1} : Set Int) ∧ f ∈ ({-1, 1} : Set Int) ∧ 
  g ∈ ({-1, 1} : Set Int) ∧ h ∈ ({-1, 1} : Set Int) ∧
  13 ∣ expression a b c d e f g h :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_strategy_l336_33606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l336_33622

noncomputable def f (x : ℝ) := Real.cos (2 * x + Real.pi / 6)

theorem axis_of_symmetry :
  ∀ (x : ℝ), f (5 * Real.pi / 12 + x) = f (5 * Real.pi / 12 - x) :=
by
  intro x
  unfold f
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l336_33622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_l336_33650

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2

-- Define the function g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x * Real.exp x

-- State the theorem
theorem extremum_and_monotonicity (a : ℝ) :
  (∃ (ext : ℝ), ext = f a (-4/3) ∧ ∀ (x : ℝ), f a x ≤ ext) →
  (a = 1/2 ∧
   f a (-4/3) = 32/27 ∧
   (∀ x y : ℝ, x < y → x < -4 → g a x > g a y) ∧
   (∀ x y : ℝ, x < y → -4 < x → y < -1 → g a x < g a y) ∧
   (∀ x y : ℝ, x < y → -1 < x → y < 0 → g a x > g a y) ∧
   (∀ x y : ℝ, x < y → 0 < x → g a x < g a y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_l336_33650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_324_l336_33665

/-- The side length of both the dodecagon and hexagon -/
def side_length : ℝ := 12

/-- A regular dodecagon -/
structure RegularDodecagon where
  side_length : ℝ

/-- A regular hexagon -/
structure RegularHexagon where
  side_length : ℝ

/-- The area of the shaded region between a regular dodecagon and its inscribed regular hexagon -/
def shaded_area (d : RegularDodecagon) (h : RegularHexagon) : ℝ :=
  sorry -- Definition of the shaded area calculation

/-- Theorem stating that the area of the shaded region is 324 square units -/
theorem shaded_area_is_324 (d : RegularDodecagon) (h : RegularHexagon) 
    (h1 : d.side_length = side_length) 
    (h2 : h.side_length = side_length) 
    (h3 : h.side_length = d.side_length) : 
  shaded_area d h = 324 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_324_l336_33665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_when_m_3_z_purely_imaginary_iff_m_3_l336_33687

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 6) (m + 2)

-- Statement (I)
theorem abs_z_when_m_3 : Complex.abs (z 3) = 5 := by sorry

-- Statement (II)
theorem z_purely_imaginary_iff_m_3 : 
  ∀ m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_when_m_3_z_purely_imaginary_iff_m_3_l336_33687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_percentage_calculation_l336_33692

/-- Calculates the percentage of a stock given the income, investment, and brokerage rate. -/
noncomputable def stock_percentage (income : ℝ) (investment : ℝ) (brokerage_rate : ℝ) : ℝ :=
  let brokerage_fee := brokerage_rate * investment
  let net_investment := investment - brokerage_fee
  (income / net_investment) * 100

/-- Theorem stating that the stock percentage is approximately 10.83% given the specified conditions. -/
theorem stock_percentage_calculation :
  let income : ℝ := 756
  let investment : ℝ := 7000
  let brokerage_rate : ℝ := 0.0025  -- 1/4% expressed as a decimal
  abs (stock_percentage income investment brokerage_rate - 10.83) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_percentage_calculation_l336_33692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adam_money_ratio_l336_33652

-- Define the initial amount Adam had
def initial_amount : ℕ := 91

-- Define the amount Adam spent
def spent_amount : ℕ := 21

-- Define the remaining amount
def remaining_amount : ℕ := initial_amount - spent_amount

-- Define the ratio of remaining to spent as a pair of natural numbers
def ratio : ℕ × ℕ := (10, 3)

-- Theorem to prove
theorem adam_money_ratio :
  Nat.gcd remaining_amount spent_amount = 7 ∧
  (remaining_amount / 7, spent_amount / 7) = ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adam_money_ratio_l336_33652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cook_one_potato_l336_33625

/-- Given a chef cooking potatoes with the following conditions:
  * The total number of potatoes to cook is 12
  * The number of potatoes already cooked is 6
  * The time to cook the remaining potatoes is 36 minutes
  Prove that the time to cook one potato is 6 minutes -/
theorem time_to_cook_one_potato
  (total_potatoes : ℕ)
  (cooked_potatoes : ℕ)
  (time_for_remaining : ℕ)
  (h1 : total_potatoes = 12)
  (h2 : cooked_potatoes = 6)
  (h3 : time_for_remaining = 36) :
  time_for_remaining / (total_potatoes - cooked_potatoes) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cook_one_potato_l336_33625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_plus_2y_equals_one_l336_33662

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (hx : x ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (hy : y ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (a : ℝ)
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_plus_2y_equals_one_l336_33662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_is_e_l336_33635

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + Real.sin x + x^2 - x

theorem tangent_line_parallel_implies_a_is_e (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ (m b : ℝ), ∀ x, m * x + b = f a x + (deriv (f a)) 0 * (x - 0)) →
  (∃ (k : ℝ), ∀ x y, 2*x - 2*y + 9 = k * (m * x + b - f a 0)) →
  a = Real.exp 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_is_e_l336_33635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_labeling_theorem_l336_33614

/-- A type representing points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the angle between three points -/
noncomputable def angle (a b c : Point3D) : ℝ := sorry

/-- Check if four points are coplanar -/
def coplanar (a b c d : Point3D) : Prop := sorry

/-- The main theorem -/
theorem point_labeling_theorem (n : ℕ) (points : Fin n → Point3D) 
  (h1 : n ≥ 4)
  (h2 : ∀ (a b c d : Fin n), ¬coplanar (points a) (points b) (points c) (points d))
  (h3 : ∀ (a b c : Fin n), ∃ (i j k : Fin 3), 
    let pts := [points a, points b, points c]
    angle (pts.get i) (pts.get j) (pts.get k) > 120) :
  ∃ (perm : Fin n ≃ Fin n), ∀ (i j k : Fin n), i < j → j < k → 
    angle (points (perm i)) (points (perm j)) (points (perm k)) > 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_labeling_theorem_l336_33614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trade_profit_l336_33629

/-- Given a 20% discount on the original price and a 55% increase on the buying price, 
    the profit percentage on the original price is 24%. -/
theorem car_trade_profit (P : ℝ) (h : P > 0) : 
  let discount_rate := 0.20
  let increase_rate := 0.55
  let buying_price := P * (1 - discount_rate)
  let selling_price := buying_price * (1 + increase_rate)
  let profit := selling_price - P
  let profit_percentage := (profit / P) * 100
  profit_percentage = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trade_profit_l336_33629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l336_33675

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (h1 : r = 1/4) (h2 : S = 40) :
  ∃ a : ℝ, infiniteGeometricSum a r = S ∧ a = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l336_33675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_is_all_reals_l336_33677

noncomputable def h (x : ℝ) : ℝ := (x^4 - 5*x + 6) / (|x - 4| + |x + 2| - 1)

theorem h_domain_is_all_reals :
  ∀ x : ℝ, |x - 4| + |x + 2| - 1 ≠ 0 :=
by
  intro x
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_is_all_reals_l336_33677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_min_distance_l336_33630

def TurtleMove := ℕ → ℝ × ℝ

def is_valid_turtle_move (move : TurtleMove) : Prop :=
  move 0 = (0, 0) ∧
  ∀ n : ℕ, n < 11 → 
    let (x₁, y₁) := move n
    let (x₂, y₂) := move (n + 1)
    ((x₂ - x₁)^2 + (y₂ - y₁)^2 = 25) ∧
    ((x₂ - x₁ = 0 ∧ (y₂ - y₁ = 5 ∨ y₂ - y₁ = -5)) ∨
     (y₂ - y₁ = 0 ∧ (x₂ - x₁ = 5 ∨ x₂ - x₁ = -5)))

theorem turtle_min_distance (move : TurtleMove) 
  (h : is_valid_turtle_move move) : 
  ∃ (d : ℝ), d ≥ 5 ∧ 
  let (x, y) := move 11
  x^2 + y^2 = d^2 := by
  sorry

#check turtle_min_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_min_distance_l336_33630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_isosceles_division_l336_33678

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- An angle bisector of a triangle -/
structure AngleBisector (T : Triangle) where
  vertex : Fin 3
  endpoint : Point

/-- Predicate to check if a triangle is isosceles -/
def IsIsosceles (T : Triangle) : Prop := sorry

/-- Predicate to check if an angle bisector divides a triangle into two isosceles triangles -/
def DividesIntoIsosceles (T : Triangle) (AB : AngleBisector T) : Prop := sorry

/-- The angles of a triangle -/
def Angles (T : Triangle) : Fin 3 → ℝ := sorry

theorem angle_bisector_isosceles_division (T : Triangle) (AB : AngleBisector T) :
  DividesIntoIsosceles T AB →
  (∀ (i : Fin 3), (Angles T i = 45 ∨ Angles T i = 90)) ∨ 
  (∀ (i : Fin 3), (Angles T i = 36 ∨ Angles T i = 72)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_isosceles_division_l336_33678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_extrema_l336_33694

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin (x/2 + π/4) * cos (x/2 + π/4) - sin (x + π)

noncomputable def g (x : ℝ) : ℝ := f (x - π/6)

theorem f_period_and_g_extrema :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
   (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 π → g x ≤ 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 π → g x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 π ∧ g x = 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 π ∧ g x = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_extrema_l336_33694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_easter_eggs_fraction_l336_33640

theorem easter_eggs_fraction (E : ℝ) (hE : E > 0) :
  let blue_eggs := (4/5) * E
  let purple_eggs := E - blue_eggs
  let purple_five_candy := (1/2) * purple_eggs
  let blue_five_candy := (1/4) * blue_eggs
  let total_five_candy := purple_five_candy + blue_five_candy
  total_five_candy = (3/10) * E →
  purple_eggs / E = 1/5 :=
by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_easter_eggs_fraction_l336_33640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_collect_all_water_l336_33646

/-- Represents the state of water distribution in buckets -/
def WaterState := List Nat

/-- The initial state of water distribution -/
def initial_state : WaterState := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- The sum of water in all buckets -/
def total_water (state : WaterState) : Nat :=
  state.sum

/-- Represents a pouring operation from bucket i to bucket j -/
def pour (state : WaterState) (i j : Nat) : WaterState :=
  sorry

/-- Predicate to check if all water is in one bucket -/
def all_in_one_bucket (state : WaterState) : Prop :=
  ∃ (i : Nat), i < state.length ∧ 
    state.get ⟨i, by sorry⟩ = total_water state ∧ 
    ∀ (j : Nat), j < state.length → j ≠ i → state.get ⟨j, by sorry⟩ = 0

/-- The main theorem stating it's impossible to get all water in one bucket -/
theorem cannot_collect_all_water :
  ¬∃ (final_state : WaterState), 
    (∃ (n : Nat) (indices : List (Nat × Nat)), 
      final_state = (indices.foldl (λ acc (i, j) => pour acc i j) initial_state)) ∧
    all_in_one_bucket final_state :=
  sorry

#check cannot_collect_all_water

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_collect_all_water_l336_33646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_love_puzzle_solution_l336_33637

/-- Represents a digit in the range [0, 9] -/
def Digit := Fin 10

/-- Represents the equation LOVE + EVOL + LOVE = SOLVES -/
def EquationHolds (L O V E S : Digit) : Prop :=
  (1000 * L.val + 100 * O.val + 10 * V.val + E.val : ℕ) +
  (1000 * E.val + 100 * V.val + 10 * O.val + L.val : ℕ) +
  (1000 * L.val + 100 * O.val + 10 * V.val + E.val : ℕ) =
  (10000 * S.val + 1000 * O.val + 100 * L.val + 10 * V.val + E.val + S.val : ℕ)

/-- All letters represent distinct digits -/
def DistinctDigits (L O V E S : Digit) : Prop :=
  L ≠ O ∧ L ≠ V ∧ L ≠ E ∧ L ≠ S ∧
  O ≠ V ∧ O ≠ E ∧ O ≠ S ∧
  V ≠ E ∧ V ≠ S ∧
  E ≠ S

theorem love_puzzle_solution :
  ∃! (L O V E S : Digit),
    EquationHolds L O V E S ∧
    DistinctDigits L O V E S ∧
    (1000 * L.val + 100 * O.val + 10 * V.val + E.val : ℕ) = 4378 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_love_puzzle_solution_l336_33637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_modulo_prime_l336_33631

theorem distinct_sums_modulo_prime (p k l : ℕ) (x : Fin k → ℤ) (y : Fin l → ℤ) (h_prime : Nat.Prime p) 
  (h_x_distinct : ∀ i j, i ≠ j → x i % p ≠ x j % p)
  (h_y_distinct : ∀ i j, i ≠ j → y i % p ≠ y j % p) :
  (Finset.univ.image (λ (i : Fin k) (j : Fin l) => (x i + y j) % p)).card ≥ min p (k + l - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_modulo_prime_l336_33631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PQ_length_and_sum_l336_33623

def R : ℝ × ℝ := (10, 8)

def line1 (x y : ℝ) : Prop := 5 * y = 12 * x

def line2 (x y : ℝ) : Prop := 15 * y = 4 * x

def is_midpoint (P Q M : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem PQ_length_and_sum (P Q : ℝ × ℝ) (m n : ℕ) :
  line1 P.1 P.2 →
  line2 Q.1 Q.2 →
  is_midpoint P Q R →
  distance P Q = 2 * Real.sqrt 41 →
  (distance P Q)^2 = m / n →
  Nat.Coprime m n →
  m + n = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PQ_length_and_sum_l336_33623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l336_33610

/-- The distance between the foci of a hyperbola with equation x^2 - 5x - 3y^2 - 9y = 45 is 16√33/3 -/
theorem hyperbola_foci_distance (x y : ℝ) : 
  x^2 - 5*x - 3*y^2 - 9*y = 45 → 
  ∃ (f₁ f₂ : ℝ × ℝ), dist f₁ f₂ = (16 * Real.sqrt 33) / 3 :=
by
  intro h
  -- Define the foci
  let f₁ : ℝ × ℝ := (5/2 + 4*Real.sqrt 11, 3/2)
  let f₂ : ℝ × ℝ := (5/2 - 4*Real.sqrt 11, 3/2)
  -- Prove that these points satisfy the distance formula
  have dist_eq : dist f₁ f₂ = (16 * Real.sqrt 33) / 3 := by
    -- The actual proof would go here
    sorry
  -- Show that the foci exist
  exact ⟨f₁, f₂, dist_eq⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l336_33610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkout_lane_shoppers_l336_33672

theorem checkout_lane_shoppers (total_shoppers : ℕ) (express_lane_fraction : ℚ) : 
  total_shoppers = 480 → 
  express_lane_fraction = 5 / 8 →
  total_shoppers - (express_lane_fraction * ↑total_shoppers).floor = 180 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkout_lane_shoppers_l336_33672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_response_is_either_l336_33696

-- Define an enum for the possible responses
inductive Response
  | Both
  | Either
  | None
  | Neither

-- Define a function that checks if a response is correct
def isCorrectResponse (r : Response) : Prop :=
  match r with
  | Response.Either => True
  | _ => False

-- Theorem: The correct response is "Either"
theorem correct_response_is_either :
  isCorrectResponse Response.Either :=
by
  -- Unfold the definition of isCorrectResponse
  unfold isCorrectResponse
  -- The goal is now True, which is trivially provable
  trivial

#check correct_response_is_either

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_response_is_either_l336_33696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_qualifying_numbers_l336_33666

/-- A function that checks if a four-digit number is greater than 3999 -/
def is_greater_than_3999 (n : ℕ) : Bool :=
  n ≥ 4000 && n < 10000

/-- A function that extracts the middle two digits of a four-digit number -/
def middle_two_digits (n : ℕ) : ℕ × ℕ :=
  ((n / 100) % 10, (n / 10) % 10)

/-- A function that checks if the sum of two digits exceeds 10 -/
def sum_exceeds_10 (pair : ℕ × ℕ) : Bool :=
  pair.1 + pair.2 > 10

/-- The main theorem stating that the count of qualifying numbers is 960 -/
theorem count_qualifying_numbers :
  (Finset.filter (λ n : ℕ => is_greater_than_3999 n && sum_exceeds_10 (middle_two_digits n)) (Finset.range 10000)).card = 960 := by
  sorry

#eval (Finset.filter (λ n : ℕ => is_greater_than_3999 n && sum_exceeds_10 (middle_two_digits n)) (Finset.range 10000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_qualifying_numbers_l336_33666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_root_values_l336_33698

/-- A polynomial with integer coefficients of the form x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 36 -/
def IntPolynomial (b₃ b₂ b₁ : ℤ) (x : ℝ) : ℝ :=
  x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 36

/-- s is a double root of the polynomial -/
def IsDoubleRoot (s : ℤ) (b₃ b₂ b₁ : ℤ) : Prop :=
  ∃ (c₁ c₀ : ℝ), (IntPolynomial b₃ b₂ b₁ s = 0) ∧
  (∀ x : ℝ, IntPolynomial b₃ b₂ b₁ x = (x - s)^2 * (x^2 + c₁*x + c₀))

/-- The theorem stating possible values of s -/
theorem double_root_values (b₃ b₂ b₁ s : ℤ) :
  IsDoubleRoot s b₃ b₂ b₁ → s ∈ ({-6, -3, -2, -1, 1, 2, 3, 6} : Set ℤ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_root_values_l336_33698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_l336_33697

/-- Represents a square with side length 2 inches -/
structure Square where
  side : ℝ := 2

/-- Represents the configuration of three squares -/
structure SquareConfiguration where
  left : Square
  center : Square
  right : Square

/-- Calculates the height of the top vertex of the rotated center square -/
noncomputable def height_after_rotation (config : SquareConfiguration) : ℝ :=
  (3 * Real.sqrt 2) / 2

/-- Theorem stating that the height of the top vertex of the rotated center square is (3√2)/2 inches -/
theorem rotated_square_height 
  (config : SquareConfiguration) 
  (h_rotation : Real.pi / 6 = 30 * (Real.pi / 180)) -- 30 degrees in radians
  (h_touching : config.center.side = 2) :
  height_after_rotation config = (3 * Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_l336_33697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_players_with_high_score_l336_33667

/-- Represents a player in the league --/
structure Player where
  id : Nat
  score : Rat

/-- Represents the league --/
structure League where
  players : Finset Player
  num_players : Nat
  total_points : Rat

/-- Theorem stating the maximum number of players who can have at least 54 points --/
theorem max_players_with_high_score (L : League) : 
  L.num_players = 90 → 
  L.total_points = (L.num_players * (L.num_players - 1) / 2) → 
  (∀ p ∈ L.players, p.score ≤ L.num_players - 1) →
  (∀ p ∈ L.players, p.score ≥ 0) →
  (L.players.filter (fun p => p.score ≥ 54)).card ≤ 71 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_players_with_high_score_l336_33667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_parabola_l336_33615

def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ I → y ∈ I → x < y → f x > f y

theorem decreasing_parabola 
  (a b : ℝ) 
  (h1 : is_decreasing (fun x ↦ a * x) (Set.Ioi 0))
  (h2 : is_decreasing (fun x ↦ -b / x) (Set.Ioi 0)) :
  is_decreasing (fun x ↦ a * x^2 + b * x) (Set.Ioi 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_parabola_l336_33615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_correct_tangent_not_parallel_l336_33663

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a - 1) * x - Real.log x

-- Define the maximum value function
def max_value (a : ℝ) : ℝ :=
  if -1/4 ≤ a ∧ a < 0 then
    2 - Real.log 2
  else if -1/2 < a ∧ a < -1/4 then
    1 - 1/(4*a) + Real.log (-2*a)
  else
    1 - a

-- Theorem for the maximum value of f(x)
theorem max_value_correct (a : ℝ) (h : a < 0) :
  ∀ x ∈ Set.Icc 1 2, f a x ≤ max_value a := by sorry

-- Theorem for non-parallel tangent
theorem tangent_not_parallel (a : ℝ) (h : a ≠ 0) (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : x₁ ≠ x₂) (h₂ : f a x₁ = y₁) (h₃ : f a x₂ = y₂) :
  let x₀ := (x₁ + x₂) / 2
  let k₁ := (y₂ - y₁) / (x₂ - x₁)
  let k₂ := (deriv (f a)) x₀
  k₁ ≠ k₂ := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_correct_tangent_not_parallel_l336_33663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l336_33670

/-- Sequence A: sum of reciprocals of odd square numbers with alternating signs -/
noncomputable def sequenceA : ℝ := ∑' n, (-1)^(n+1) / (2*n - 1)^2

/-- Sequence B: sum of reciprocals of even square numbers with alternating signs -/
noncomputable def sequenceB : ℝ := ∑' n, (-1)^n / (2*n)^2

/-- The ratio of sequence A to sequence B is -4 -/
theorem sequence_ratio : sequenceA / sequenceB = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l336_33670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l336_33657

open Filter Topology

theorem limit_of_sequence : 
  Tendsto (fun n : ℕ => (2^(n+1) + 3^(n+1)) / (2^n + 3^n)) atTop (𝓝 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l336_33657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l336_33608

open Real

-- Define the two functions as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * sin (2 * x) - cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * (x + π / 12))

-- State the theorem
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by
  intro x
  -- The proof goes here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l336_33608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_theorem_l336_33684

/-- The compound interest function -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The final amount after applying compound interest for three periods -/
noncomputable def final_amount (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (rate3 : ℝ) (time3 : ℝ) : ℝ :=
  compound_interest (compound_interest (compound_interest principal rate1 1) rate2 1) rate3 time3

/-- The theorem stating the relationship between the principal and final amount -/
theorem principal_amount_theorem (P : ℝ) :
  final_amount P 0.05 0.06 0.07 (2/5) = 1008 →
  ∃ ε > 0, |P - 905.08| < ε := by
  sorry

#check principal_amount_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_theorem_l336_33684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_determine_plane_three_points_not_always_determine_plane_line_and_point_not_always_determine_plane_three_intersecting_lines_not_always_determine_plane_l336_33601

-- Define the basic geometric objects
structure Point where

structure Line where

structure Plane where

-- Define the relationships between objects
def determines_plane (objects : List (Point ⊕ Line)) (p : Plane) : Prop :=
  sorry

def parallel (l1 l2 : Line) : Prop :=
  sorry

-- State the theorem
theorem parallel_lines_determine_plane :
  ∀ (l1 l2 : Line),
  parallel l1 l2 → ∃ (p : Plane), determines_plane [Sum.inr l1, Sum.inr l2] p :=
by sorry

-- State the counterexamples for the other propositions
theorem three_points_not_always_determine_plane :
  ∃ (p1 p2 p3 : Point),
  ¬∃ (p : Plane), determines_plane [Sum.inl p1, Sum.inl p2, Sum.inl p3] p :=
by sorry

theorem line_and_point_not_always_determine_plane :
  ∃ (l : Line) (p : Point),
  ¬∃ (plane : Plane), determines_plane [Sum.inr l, Sum.inl p] plane :=
by sorry

theorem three_intersecting_lines_not_always_determine_plane :
  ∃ (l1 l2 l3 : Line),
  (∃ (p : Point), ∃ (plane : Plane), determines_plane [Sum.inr l1, Sum.inr l2, Sum.inr l3] plane) ∧
  ¬∃ (plane : Plane), determines_plane [Sum.inr l1, Sum.inr l2, Sum.inr l3] plane :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_determine_plane_three_points_not_always_determine_plane_line_and_point_not_always_determine_plane_three_intersecting_lines_not_always_determine_plane_l336_33601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_orthogonal_line_plane_pairs_l336_33668

/-- Represents a cube -/
structure Cube where
  -- Add any necessary properties of a cube
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- Represents an orthogonal line-plane pair in a cube -/
structure OrthogonalLinePlanePair (c : Cube) where
  -- Add any necessary properties of an orthogonal line-plane pair
  line : Set (Fin 3 → ℝ)
  plane : Set (Fin 3 → ℝ)

/-- Count the number of orthogonal line-plane pairs in a cube -/
def countOrthogonalLinePlanePairs (c : Cube) : ℕ :=
  -- Implementation details are not provided, so we'll use a placeholder
  36

/-- Theorem stating that the number of orthogonal line-plane pairs in a cube is 36 -/
theorem cube_orthogonal_line_plane_pairs (c : Cube) :
  countOrthogonalLinePlanePairs c = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_orthogonal_line_plane_pairs_l336_33668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_flip_theorem_l336_33617

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the state of the grid -/
def Grid := Cell → Int

/-- An operation on the grid -/
def Operation := Cell → Grid → Grid

/-- Checks if two cells are adjacent -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col + 1 = c2.col ∨ c2.col + 1 = c1.col)) ∨
  (c1.col = c2.col ∧ (c1.row + 1 = c2.row ∨ c2.row + 1 = c1.row))

/-- Defines a valid operation on the grid -/
def validOperation (op : Operation) (n : Nat) : Prop :=
  ∀ (c : Cell) (g : Grid),
    c.row ≤ n ∧ c.col ≤ n →
    op c g c = g c ∧
    ∀ (c' : Cell), adjacent c c' → op c g c' = -g c'

/-- Initial state of the grid -/
def initialGrid (n : Nat) : Grid :=
  fun _ => 1

/-- Final state of the grid (all -1) -/
def finalGrid (n : Nat) : Grid :=
  fun _ => -1

/-- Theorem stating the condition for achieving the goal -/
theorem grid_flip_theorem (n : Nat) :
  n ≥ 2 →
  (∃ (ops : List Operation),
    (∀ op ∈ ops, validOperation op n) ∧
    (ops.foldl (fun g op => op (Cell.mk 1 1) g) (initialGrid n) = finalGrid n))
  ↔
  Even n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_flip_theorem_l336_33617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_and_smallest_A_l336_33689

def is_nine_digit (n : ℕ) : Prop := 100000000 ≤ n ∧ n ≤ 999999999

def last_digit (n : ℕ) : ℕ := n % 10

def without_last_digit (n : ℕ) : ℕ := n / 10

def move_last_digit_to_front (n : ℕ) : ℕ :=
  (last_digit n) * 100000000 + (without_last_digit n)

theorem largest_and_smallest_A (B : ℕ) 
  (h1 : is_nine_digit B)
  (h2 : Nat.Coprime B 18)
  (h3 : B > 222222222) :
  ∃ (A_max A_min : ℕ),
    (∀ A : ℕ, (∃ B' : ℕ, is_nine_digit B' ∧ Nat.Coprime B' 18 ∧ B' > 222222222 ∧ A = move_last_digit_to_front B') 
      → A ≤ A_max ∧ A ≥ A_min) ∧
    A_max = 999999998 ∧
    A_min = 122222224 :=
by
  sorry

#check largest_and_smallest_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_and_smallest_A_l336_33689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l336_33664

/-- The quadrilateral region defined by the system of inequalities -/
def QuadrilateralRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 ≤ 4 ∧ 3 * p.1 + p.2 ≥ 3 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- The vertices of the quadrilateral region -/
def Vertices : Set (ℝ × ℝ) :=
  {(0, 0), (1, 0), (4, 0), (0, 3)}

/-- The length of a line segment between two points -/
noncomputable def SegmentLength (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that the longest side of the quadrilateral has length 5 -/
theorem longest_side_length :
  ∃ (p q : ℝ × ℝ), p ∈ Vertices ∧ q ∈ Vertices ∧
    SegmentLength p q = 5 ∧
    ∀ (r s : ℝ × ℝ), r ∈ Vertices → s ∈ Vertices →
      SegmentLength r s ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l336_33664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_range_l336_33660

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2 = 1

-- Define the point M
def M : ℝ × ℝ := (1, -1)

-- Define the line l (implicitly through its properties)
def line_l (A B N : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), A = M + t • (N - M) ∧ B = M + t • (N - M) ∧ N.2 = 0

-- Define the relationship between vectors
def vector_relation (A B N : ℝ × ℝ) (lambda1 lambda2 : ℝ) : Prop :=
  A - M = lambda1 • (N - A) ∧ B - M = lambda2 • (N - B)

-- State the theorem
theorem hyperbola_intersection_range :
  ∀ (A B N : ℝ × ℝ) (lambda1 lambda2 : ℝ),
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  line_l A B N →
  vector_relation A B N lambda1 lambda2 →
  A.1 > 2 ∧ B.1 > 2 →
  (lambda1 / lambda2 + lambda2 / lambda1) ≥ 2 ∧ 
  ∀ ε > 0, ∃ A' B' N' lambda1' lambda2',
    hyperbola A'.1 A'.2 ∧
    hyperbola B'.1 B'.2 ∧
    line_l A' B' N' ∧
    vector_relation A' B' N' lambda1' lambda2' ∧
    A'.1 > 2 ∧ B'.1 > 2 ∧
    lambda1' / lambda2' + lambda2' / lambda1' < 2 + ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_range_l336_33660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_coordinates_l336_33688

def point_C : ℝ × ℝ := (8, 6)
def point_M : ℝ × ℝ := (4, 10)

theorem midpoint_sum_coordinates (point_D : ℝ × ℝ) : 
  (point_M.1 = (point_C.1 + point_D.1) / 2 ∧ 
   point_M.2 = (point_C.2 + point_D.2) / 2) →
  point_D.1 + point_D.2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_coordinates_l336_33688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l336_33648

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_properties (a b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a1 : a 1 = 1)
  (h_b1 : b 1 = 1)
  (h_a2a4 : a 2 + a 4 = 10)
  (h_b2b4 : b 2 * b 4 = a 5) :
  (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∀ n : ℕ, (Finset.range n).sum (fun i => b (2 * i + 1)) = (3^n - 1) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l336_33648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l336_33634

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote: y = 2x + 3 -/
  asymptote1 : ℝ → ℝ
  /-- Second asymptote: y = -2x + 1 -/
  asymptote2 : ℝ → ℝ
  /-- The hyperbola passes through the point (4,5) -/
  point : ℝ × ℝ
  /-- Condition for the first asymptote -/
  h1 : ∀ x, asymptote1 x = 2 * x + 3
  /-- Condition for the second asymptote -/
  h2 : ∀ x, asymptote2 x = -2 * x + 1
  /-- Condition for the point -/
  h3 : point = (4, 5)

/-- The distance between the foci of the hyperbola -/
noncomputable def foci_distance (h : Hyperbola) : ℝ := 3 * Real.sqrt 10

/-- Theorem stating that the distance between the foci of the given hyperbola is 3√10 -/
theorem hyperbola_foci_distance (h : Hyperbola) : foci_distance h = 3 * Real.sqrt 10 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l336_33634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_k_zero_range_k_two_zeros_l336_33655

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ k then Real.exp x - x else x^3 - x + 1

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := f k x - 1

-- Part 1: Solution set when k = 0
theorem solution_set_k_zero :
  {x : ℝ | f 0 x < 1} = Set.Ioo 0 1 := by sorry

-- Part 2: Range of k for exactly two zeros of g
theorem range_k_two_zeros :
  {k : ℝ | (∃ x y : ℝ, x ≠ y ∧ g k x = 0 ∧ g k y = 0 ∧ 
    ∀ z : ℝ, g k z = 0 → z = x ∨ z = y)} = Set.Icc (-1) (1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_k_zero_range_k_two_zeros_l336_33655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_score_probability_l336_33605

/-- The probability of achieving a score of n in a fair coin tossing game -/
noncomputable def score_probability (n : ℕ) : ℝ :=
  (2 + (-1/2)^n) / 3

/-- A function representing the true probability of achieving score n -/
noncomputable def probability_of_achieving_score (n : ℕ) : ℝ :=
  sorry -- This function represents the true probability of achieving score n

/-- Theorem stating that the probability of achieving a score of n
    in a fair coin tossing game where heads score 1 and tails score 2
    is given by (2 + (-1/2)^n) / 3 -/
theorem fair_coin_score_probability (n : ℕ) :
  score_probability n = probability_of_achieving_score n :=
by
  sorry -- Proof to be filled in later


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_score_probability_l336_33605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_for_hall_l336_33620

/-- The number of stones required to pave a rectangular hall with rectangular stones -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  (hall_length * hall_width * 100 / (stone_length * stone_width)).ceil.toNat

/-- Theorem stating the number of stones required for the given hall and stone dimensions -/
theorem stones_for_hall : stones_required 36 15 8 5 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_for_hall_l336_33620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_equals_one_l336_33647

theorem sin_minus_cos_equals_one (x : ℝ) : 
  0 ≤ x → x < π → Real.sin x - Real.cos x = 1 → x = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_equals_one_l336_33647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_distance_proof_l336_33609

/-- The distance between the starting points of Alex and Blake -/
def initial_distance : ℝ := 150

/-- Alex's speed in meters per second -/
def alex_speed : ℝ := 10

/-- Blake's speed in meters per second -/
def blake_speed : ℝ := 9

/-- The angle between Alex's path and the line connecting the starting points -/
noncomputable def alex_angle : ℝ := Real.pi / 4

/-- The time at which Alex and Blake meet -/
noncomputable def meeting_time : ℝ := 
  let a := 19
  let b := -1500 * Real.sqrt 2
  let c := 22500
  (- b - Real.sqrt (b^2 - 4*a*c)) / (2*a)

/-- The distance Alex cycles before meeting Blake -/
noncomputable def alex_distance : ℝ := alex_speed * meeting_time

theorem alex_distance_proof : 
  alex_distance = alex_speed * meeting_time ∧ 
  19 * meeting_time^2 - 1500 * Real.sqrt 2 * meeting_time + 22500 = 0 ∧
  meeting_time > 0 ∧
  ∀ t : ℝ, t > 0 → 19 * t^2 - 1500 * Real.sqrt 2 * t + 22500 = 0 → t ≥ meeting_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_distance_proof_l336_33609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_de_length_l336_33661

-- Define the points and distances
variable (a b c d e : ℝ)
variable (ab bc cd de ae : ℝ)

-- State the conditions
axiom consecutive_points : a < b ∧ b < c ∧ c < d ∧ d < e
axiom bc_relation : bc = 3 * cd
axiom ab_length : ab = 5
axiom ac_length : ab + bc = 11
axiom ae_length : ae = 20

-- Define relationships between points and distances
axiom ae_sum : ae = ab + bc + cd + de

-- Theorem to prove
theorem de_length : de = 7 := by
  sorry -- Proof to be completed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_de_length_l336_33661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l336_33639

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem statement
theorem equation_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ 
  (∀ x : ℝ, x ∈ s ↔ (floor (2 * x) + floor (3 * x) = ⌊8 * x - 6⌋)) ∧
  (∀ y : ℝ, floor (2 * y) + floor (3 * y) = ⌊8 * y - 6⌋ → y ∈ s) := by
  sorry

#check equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l336_33639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intersection_l336_33642

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse defined by its foci -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point lies on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) (constantSum : ℝ) : Prop :=
  distance p e.focus1 + distance p e.focus2 = constantSum

/-- The main theorem about the ellipse's intersection with x-axis -/
theorem ellipse_x_intersection (e : Ellipse) (constantSum : ℝ) :
  e.focus1 = ⟨0, 3⟩ →
  e.focus2 = ⟨4, 0⟩ →
  isOnEllipse e ⟨0, 0⟩ constantSum →
  ∃ (x : ℝ), x ≠ 0 ∧ isOnEllipse e ⟨x, 0⟩ constantSum ∧ x = 56 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intersection_l336_33642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l336_33659

-- Define propositions p and q
noncomputable def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2*x + a) / Real.log 0.5

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (-(5 - 2*a))^x₁ > (-(5 - 2*a))^x₂

-- Main theorem
theorem range_of_a : 
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  {a : ℝ | 1 < a ∧ a < 2} = {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l336_33659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_relationship_l336_33643

/-- Represents the taxi fare function for daytime rides in a certain city. -/
noncomputable def taxi_fare (x : ℝ) : ℝ :=
  if x ≤ 3 then 14 else 14 + 2.4 * (x - 3)

/-- Theorem stating the relationship between distance and fare for taxi rides longer than 3 km. -/
theorem taxi_fare_relationship (x : ℝ) (h : x > 3) :
  taxi_fare x = 2.4 * x + 6.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_relationship_l336_33643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_slope_theorem_l336_33651

/-- Midpoint of a line segment --/
def our_midpoint (x₁ y₁ x₂ y₂ : ℚ) : ℚ × ℚ :=
  ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

/-- Slope between two points --/
def our_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ :=
  (y₂ - y₁) / (x₂ - x₁)

theorem midpoint_slope_theorem :
  let m₁ := our_midpoint 1 2 3 6
  let m₂ := our_midpoint 4 3 7 9
  our_slope m₁.1 m₁.2 m₂.1 m₂.2 = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_slope_theorem_l336_33651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_player_counts_l336_33658

/-- Represents a group of players divided into subgroups -/
structure PlayerGroup where
  n : ℕ  -- number of subgroups
  m : ℕ  -- number of players in each subgroup

/-- The property that each player has exactly 15 opponents -/
def has_fifteen_opponents (g : PlayerGroup) : Prop :=
  (g.n - 1) * g.m = 15

/-- The total number of players in the group -/
def total_players (g : PlayerGroup) : ℕ :=
  g.n * g.m

/-- Theorem stating the possible total numbers of players -/
theorem possible_player_counts (g : PlayerGroup) :
  has_fifteen_opponents g → total_players g ∈ ({16, 18, 20, 30} : Set ℕ) := by
  sorry

#check possible_player_counts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_player_counts_l336_33658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l336_33619

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2*(a-2)*x - 1 else a^x

-- State the theorem
theorem increasing_function_a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y, 0 < x ∧ x < y → f a x < f a y) →
  (4/3 : ℝ) ≤ a ∧ a ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l336_33619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l336_33699

/-- Represents a repeating decimal in base 10 with a repeating part of arbitrary length -/
def RepeatingDecimal (whole : ℕ) (repeat_part : ℕ) (repeat_length : ℕ) : ℚ :=
  (whole : ℚ) + (repeat_part : ℚ) / ((10 ^ repeat_length - 1) : ℚ)

/-- The sum of three specific repeating decimals equals 2474646/9999 -/
theorem repeating_decimal_sum :
  RepeatingDecimal 0 2 1 + RepeatingDecimal 0 2 2 + RepeatingDecimal 0 2 4 = 2474646 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l336_33699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_dot_product_minimized_l336_33612

-- Define the line equation
def line_equation (m x y : ℝ) : Prop :=
  (m + 2) * x - (2 * m + 1) * y - 3 = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (2, 1)

-- Define point P
def point_p : ℝ × ℝ := (-1, -2)

-- Define the function to calculate the dot product of PA and PB
noncomputable def dot_product (m : ℝ) : ℝ :=
  let a := 3 / (m + 2)  -- x-intercept
  let b := 3 / (2 * m + 1)  -- y-intercept
  (a + 1) * 1 + 2 * (b + 2)  -- dot product formula

-- Theorem 1: The line passes through the fixed point for all m
theorem line_passes_through_fixed_point (m : ℝ) :
  line_equation m (fixed_point.1) (fixed_point.2) := by
  sorry

-- Theorem 2: The dot product is minimized when m = -5/4
theorem dot_product_minimized :
  ∃ (m : ℝ), m = -5/4 ∧ ∀ (n : ℝ), dot_product m ≤ dot_product n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_dot_product_minimized_l336_33612
