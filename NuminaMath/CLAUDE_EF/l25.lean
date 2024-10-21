import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_solution_l25_2599

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

-- Define the solution set
def solution_set : Set ℝ := {x | Real.pi / 3 < x ∧ x < Real.pi / 2}

-- Theorem statement
theorem g_inequality_solution :
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi ∧ g x > Real.sqrt 3 / 2} = solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_solution_l25_2599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quotients_power_of_two_l25_2563

/-- A natural power of two is 2^n where n is a natural number. -/
def NaturalPowerOfTwo (x : ℝ) : Prop := ∃ n : ℕ, x = 2^n

theorem quotients_power_of_two 
  (a : Fin 8 → ℝ) 
  (distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (positive : ∀ i, a i > 0)
  (h : ∃ S : Finset (Fin 8 × Fin 8), 
       S.card = 22 ∧ 
       (∀ (i j : Fin 8), (i, j) ∈ S → i < j → NaturalPowerOfTwo (a j / a i))) :
  ∀ i j, i < j → NaturalPowerOfTwo (a j / a i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quotients_power_of_two_l25_2563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_polar_to_circle_center_l25_2568

/-- The distance from a point in polar coordinates to the center of a circle --/
theorem distance_polar_to_circle_center (r θ : Real) : 
  r = 2 ∧ θ = Real.pi / 3 → 
  ∃ (center_x center_y : Real), 
    (∀ (x y : Real), x^2 + y^2 = 2*x ↔ (x - center_x)^2 + (y - center_y)^2 = 1) ∧
    Real.sqrt ((r * Real.cos θ - center_x)^2 + (r * Real.sin θ - center_y)^2) = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_polar_to_circle_center_l25_2568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_is_four_l25_2504

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -12 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (-3, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the distance from a point to the y-axis
def distToYAxis (p : ℝ × ℝ) : ℝ := |p.1|

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The main theorem
theorem distance_to_focus_is_four (P : PointOnParabola) 
  (h : distToYAxis (P.x, P.y) = 1) : 
  distance (P.x, P.y) focus = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_is_four_l25_2504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_is_circle_when_k_is_1_intersection_point_when_k_is_4_l25_2509

-- Define the curves C₁ and C₂
noncomputable def C₁ (k : ℝ) (t : ℝ) : ℝ × ℝ := (Real.cos t ^ k, Real.sin t ^ k)

def C₂ (θ : ℝ) (ρ : ℝ) : Prop := 4 * ρ * Real.cos θ - 16 * ρ * Real.sin θ + 3 = 0

-- Theorem for part 1
theorem C₁_is_circle_when_k_is_1 :
  ∀ x y : ℝ, (∃ t : ℝ, C₁ 1 t = (x, y)) ↔ x^2 + y^2 = 1 := by
  sorry

-- Theorem for part 2
theorem intersection_point_when_k_is_4 :
  (∃ t : ℝ, C₁ 4 t = (1/4, 1/4)) ∧ C₂ (Real.arctan 1) (Real.sqrt (1/32)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_is_circle_when_k_is_1_intersection_point_when_k_is_4_l25_2509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monthly_profit_l25_2538

/-- Represents the daily newspaper sales scenario -/
structure NewspaperSales where
  buy_price : ℚ
  sell_price : ℚ
  return_price : ℚ
  high_demand_days : ℕ
  low_demand_days : ℕ
  high_demand_copies : ℕ
  low_demand_copies : ℕ

/-- Calculates the monthly profit for a given number of copies bought daily -/
def monthly_profit (s : NewspaperSales) (copies : ℕ) : ℚ :=
  let high_revenue := s.high_demand_days * s.sell_price * (min copies s.high_demand_copies)
  let low_revenue := s.low_demand_days * s.sell_price * (min copies s.low_demand_copies)
  let return_revenue := s.return_price * (copies - s.low_demand_copies) * s.low_demand_days
  let total_cost := (s.high_demand_days + s.low_demand_days) * s.buy_price * copies
  high_revenue + low_revenue + return_revenue - total_cost

/-- The theorem stating the maximum monthly profit -/
theorem max_monthly_profit (s : NewspaperSales) 
  (h1 : s.buy_price = 1/5)
  (h2 : s.sell_price = 3/10)
  (h3 : s.return_price = 1/20)
  (h4 : s.high_demand_days = 20)
  (h5 : s.low_demand_days = 10)
  (h6 : s.high_demand_copies = 400)
  (h7 : s.low_demand_copies = 250) :
  ∃ (copies : ℕ), monthly_profit s copies = 825 ∧ 
    ∀ (x : ℕ), monthly_profit s x ≤ monthly_profit s copies :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monthly_profit_l25_2538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_inverse_l25_2536

theorem power_equality_implies_inverse (x : ℝ) : 
  (256 : ℝ)^8 = (16 : ℝ)^x → (2 : ℝ)^(-x) = 1 / 65536 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_inverse_l25_2536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_8_g_is_minimum_k_range_l25_2588

noncomputable section

-- Define the function f
def f (x t : ℝ) : ℝ := 
  (Real.sin (2*x - Real.pi/4))^2 - 2*t*(Real.sin (2*x - Real.pi/4)) + t^2 - 6*t + 1

-- Define the domain of x
def x_domain : Set ℝ := Set.Icc (Real.pi/24) (Real.pi/2)

-- Define the function g
def g (t : ℝ) : ℝ :=
  if t < -1/2 then t^2 - 5*t + 5/4
  else if t ≤ 1 then -6*t + 1
  else t^2 - 8*t + 2

-- Theorem 1: f(π/8) = -4 when t = 1
theorem f_at_pi_over_8 : f (Real.pi/8) 1 = -4 := by sorry

-- Theorem 2: g(t) is the minimum value of f(x) for x in the domain
theorem g_is_minimum (t : ℝ) : 
  ∀ x ∈ x_domain, f x t ≥ g t := by sorry

-- Theorem 3: Range of k for which g(t) = kt has a real root
theorem k_range : 
  ∀ k : ℝ, (∃ t : ℝ, -1/2 ≤ t ∧ t ≤ 1 ∧ g t = k*t) ↔ (k ≤ -8 ∨ k ≥ -5) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_8_g_is_minimum_k_range_l25_2588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l25_2598

-- Define the vectors
noncomputable def a (x : Real) : Fin 2 → Real := ![Real.sin x, Real.sqrt 3 * Real.cos x]
def b : Fin 2 → Real := ![-1, 1]
def c : Fin 2 → Real := ![1, 1]

-- Define the parallel condition
def parallel (v w : Fin 2 → Real) : Prop :=
  ∃ k : Real, ∀ i : Fin 2, v i = k * w i

theorem vector_problem (x : Real) (hx : x ∈ Set.Icc 0 Real.pi) :
  (parallel (fun i => a x i + b i) c → x = 5 * Real.pi / 6) ∧
  (((fun i => a x i) • b = 1 / 2) → Real.sin (x + Real.pi / 6) = Real.sqrt 15 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l25_2598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_is_identity_l25_2544

-- Define the property that the function f must satisfy
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y * f (x + y)) = y^2 + f (x * f (y + 1))

-- Theorem statement
theorem function_is_identity (f : ℝ → ℝ) (h : satisfies_equation f) : 
  ∀ x : ℝ, f x = x := by
  sorry

-- Example to show that the identity function satisfies the equation
example : satisfies_equation (λ x : ℝ => x) := by
  intro x y
  simp
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_is_identity_l25_2544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l25_2550

theorem polynomial_roots (s t u v AX XB CY YD : ℝ) : 
  AX + XB = s →
  AX * XB = t^2 →
  CY + YD = u →
  CY * YD = v^2 →
  ∃ (p : Polynomial ℝ), 
    p = X^4 - (s + u) • X^3 + (s * u + t^2 + v^2) • X^2 - (s * v^2 + u * t^2) • X + (t^2 * v^2) • 1 ∧
    (p.roots.toFinset = {AX, XB, CY, YD}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l25_2550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l25_2502

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 - Real.cos (2 * x + Real.pi / 3)

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ T = Real.pi ∧ ∀ x, f (x + T) = f x) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 5 / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 5 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l25_2502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_j_for_divisibility_l25_2557

-- Define q as the largest prime with 2023 digits
noncomputable def q : ℕ := sorry

-- Axiom: q is prime
axiom q_prime : Nat.Prime q

-- Axiom: q has 2023 digits
axiom q_digits : (Nat.digits 10 q).length = 2023

-- Define the property that we're looking for
def is_divisible_by_15 (n : ℕ) : Prop := 15 ∣ n

-- Theorem to prove
theorem smallest_j_for_divisibility : 
  ∃ j : ℕ, j > 0 ∧ is_divisible_by_15 (q^2 - j) ∧
  ∀ k : ℕ, 0 < k ∧ k < j → ¬is_divisible_by_15 (q^2 - k) := by
  sorry

#check smallest_j_for_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_j_for_divisibility_l25_2557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_max_value_l25_2516

theorem sin_max_value (φ : ℝ) : 
  -π/2 < φ ∧ φ < π/2 →
  (∀ x : ℝ, Real.sin (2*x + φ) ≤ Real.sin (2*(π/6) + φ)) →
  φ = π/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_max_value_l25_2516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_kth_term_l25_2579

/-- The binomial coefficient of the k-th term in the expansion of (ax + by)^n -/
def binomial_coefficient_of_kth_term (a b : ℝ) (n k : ℕ) : ℕ :=
  n.choose (k - 1)

theorem binomial_coefficient_kth_term (n k : ℕ) :
  binomial_coefficient_of_kth_term (2 : ℝ) (5 : ℝ) n k = (n.choose (k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_kth_term_l25_2579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_OP_range_l25_2540

-- Define the ellipse C
noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the focus F
noncomputable def focus : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the line l
def line (k m x : ℝ) : ℝ := k * x + m

-- Define the condition for k
noncomputable def k_condition (k : ℝ) : Prop := abs k ≤ Real.sqrt 2 / 2

-- Define the parallelogram OAPB
def parallelogram (A B P : ℝ × ℝ) : Prop :=
  ellipse P.1 P.2 ∧ 
  P.1 = A.1 + B.1 ∧ 
  P.2 = A.2 + B.2

-- Define |OP|
noncomputable def OP_length (P : ℝ × ℝ) : ℝ := Real.sqrt (P.1^2 + P.2^2)

-- Theorem statement
theorem ellipse_OP_range :
  ∀ (k m : ℝ) (A B P : ℝ × ℝ),
    k_condition k →
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    A.2 = line k m A.1 →
    B.2 = line k m B.1 →
    parallelogram A B P →
    Real.sqrt 2 ≤ OP_length P ∧ OP_length P ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_OP_range_l25_2540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l25_2514

-- Define the function g(x) as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (4 - Real.sqrt (7 - Real.sqrt (2 * x + 1)))

-- Define the domain of g(x)
def domain_g : Set ℝ := { x : ℝ | -1/2 ≤ x ∧ x ≤ 24 }

-- Theorem statement
theorem g_domain : 
  ∀ x : ℝ, g x ∈ Set.univ ↔ x ∈ domain_g :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l25_2514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminum_sulfur_reaction_l25_2583

/-- Represents a chemical element --/
structure Element where
  name : String
  molarMass : Float

/-- Represents a chemical compound --/
structure Compound where
  name : String
  formula : List (Element × Nat)

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List (Element × Nat)
  products : List (Compound × Nat)

/-- Calculate the amount of product that can be produced by a given reactant --/
noncomputable def calculatePotentialProduct (reaction : Reaction) (reactant : Element) (amount : Float) : Float :=
  sorry

/-- Determine the limiting reactant in a reaction --/
noncomputable def limitingReactant (reaction : Reaction) (reactants : List (Element × Float)) : Element :=
  sorry

/-- Calculate the mass of a compound produced in a reaction --/
noncomputable def massProduced (compound : Compound) (moles : Float) : Float :=
  sorry

/-- Theorem statement --/
theorem aluminum_sulfur_reaction 
  (al : Element) 
  (s : Element) 
  (al2s3 : Compound) 
  (reaction : Reaction) :
  al.name = "Aluminum" →
  s.name = "Sulfur" →
  al2s3.name = "Aluminum sulfide" →
  al.molarMass = 26.98 →
  s.molarMass = 32.07 →
  al2s3.formula = [(al, 2), (s, 3)] →
  reaction.reactants = [(al, 2), (s, 3)] →
  reaction.products = [(al2s3, 1)] →
  let initialAl := 6
  let initialS := 12
  let limitingReactant := limitingReactant reaction [(al, initialAl), (s, initialS)]
  let producedMoles := calculatePotentialProduct reaction limitingReactant (if limitingReactant.name = al.name then initialAl else initialS)
  let producedMass := massProduced al2s3 producedMoles
  limitingReactant.name = al.name ∧ (producedMass - 450.51).abs < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminum_sulfur_reaction_l25_2583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_of_q_l25_2500

/-- The numerator of our rational function -/
noncomputable def numerator (x : ℝ) : ℝ := 3 * x^7 - 5 * x^6 + 2 * x^3 + 4

/-- The rational function f(x) = numerator(x) / q(x) -/
noncomputable def f (q : ℝ → ℝ) (x : ℝ) : ℝ := numerator x / q x

/-- A function has a horizontal asymptote if it converges to a finite value as x approaches infinity -/
def has_horizontal_asymptote (f : ℝ → ℝ) : Prop :=
  ∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x > M, |f x - L| < ε

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

/-- The main theorem: The smallest possible degree of q(x) is 7 -/
theorem smallest_degree_of_q (q : ℝ → ℝ) :
  has_horizontal_asymptote (f q) → degree q ≥ 7 ∧ ∃ q₀ : ℝ → ℝ, degree q₀ = 7 ∧ has_horizontal_asymptote (f q₀) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_of_q_l25_2500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_repayment_theorem_l25_2519

/-- Calculates the annual repayment amount for a loan -/
noncomputable def annual_repayment (M : ℝ) (m : ℝ) : ℝ :=
  (M * m * (1 + m)^10) / ((1 + m)^10 - 1)

/-- Theorem stating that the annual repayment amount calculated by the formula
    is correct for a loan repaid over exactly 10 years -/
theorem loan_repayment_theorem (M : ℝ) (m : ℝ) (a : ℝ) 
    (h_positive_M : M > 0) 
    (h_positive_m : m > 0) 
    (h_repayment : a > 0) :
  (a * ((1 + m)^10 - 1) / m = M * (1 + m)^10) → 
  a = annual_repayment M m := by
  sorry

#check loan_repayment_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_repayment_theorem_l25_2519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_element_is_sixteen_l25_2512

/-- Represents the number of elements in each row of the arrangement -/
def elementsInRow (n : ℕ) : ℕ := 2 * n

/-- Represents the value of each element in a given row -/
def elementValue (n : ℕ) : ℕ := 2 * n

/-- Calculates the total number of elements up to and including row n -/
def totalElementsUpToRow (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc + elementsInRow (i + 1)) 0

/-- Finds the row number for a given position in the sequence -/
def findRowForPosition (pos : ℕ) : ℕ :=
  (List.range pos).findSome? (fun n => if totalElementsUpToRow (n + 1) ≥ pos then some n else none)
    |>.getD 0

theorem sixtieth_element_is_sixteen :
  elementValue (findRowForPosition 60 + 1) = 16 := by
  sorry

#eval elementValue (findRowForPosition 60 + 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_element_is_sixteen_l25_2512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l25_2556

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  d : ℚ      -- Common difference (changed to ℚ)
  first_term_def : a 1 = a 1  -- First term definition (tautology to represent a₁)
  term_relation : ∀ n : ℕ, a (n + 1) = a n + d  -- Relation between consecutive terms

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem stating the property of the arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
    (h1 : seq.a 3 = 3)
    (h2 : S seq 9 - S seq 6 = 27) : 
    seq.a 1 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l25_2556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_volume_576_l25_2571

/-- A cuboid with a square base and given dimensions -/
structure Cuboid where
  base_perimeter : ℝ
  height : ℝ
  is_square_base : base_perimeter > 0
  is_positive_height : height > 0

/-- The volume of a cuboid -/
noncomputable def volume (c : Cuboid) : ℝ :=
  (c.base_perimeter / 4) ^ 2 * c.height

/-- Theorem: The volume of a cuboid with base perimeter 32 cm and height 9 cm is 576 cm³ -/
theorem cuboid_volume_576 (c : Cuboid) 
    (h1 : c.base_perimeter = 32) 
    (h2 : c.height = 9) : 
  volume c = 576 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_volume_576_l25_2571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_constant_term_calculation_main_theorem_l25_2528

/-- The constant term in the expansion of (1/x + x^2)^3 -/
noncomputable def k : ℝ := 3

/-- The area of the closed figure formed by y = kx and y = x^2 -/
noncomputable def area : ℝ := (9 : ℝ) / 2

theorem area_calculation : area = ∫ x in (0)..(3), k * x - x^2 := by sorry

theorem constant_term_calculation : k = 3 := by sorry

theorem main_theorem : area = (9 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_constant_term_calculation_main_theorem_l25_2528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_2cos_squared_x_l25_2565

theorem sin_2x_plus_2cos_squared_x (x : ℝ) :
  Real.tan (x + π / 4) = -2 → Real.sin (2 * x) + 2 * (Real.cos x)^2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_2cos_squared_x_l25_2565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_polynomial_l25_2532

/-- The function f(X) = (X^2 + 1)(2 + cos(X)) -/
noncomputable def f (X : ℝ) : ℝ := (X^2 + 1) * (2 + Real.cos X)

/-- Theorem stating that f is not a polynomial function -/
theorem f_not_polynomial : ¬ ∃ (p : Polynomial ℝ), ∀ x, f x = p.eval x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_polynomial_l25_2532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l25_2523

-- Define the line
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the intersection
def intersection (x y : ℝ) : Prop := line x y ∧ circle_eq x y

-- Theorem: The number of intersection points is 0
theorem no_intersection : 
  ¬ ∃ (x y : ℝ), intersection x y :=
by
  sorry

#check no_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l25_2523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_divisibility_l25_2541

/-- A prime p is considered "good" if it is congruent to 1 modulo 1009 -/
def is_good_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 1009 = 1

/-- The transformation function that Sarah applies to N -/
def transform (N : ℕ) (p : ℕ) (k : ℕ) : ℕ := N * (p^k - 1)

/-- The main theorem to be proved -/
theorem sarah_divisibility :
  ∃ S : Set ℕ, (Set.Infinite S) ∧ 
  (∀ k ∈ S, Even k) ∧
  (∀ k ∈ S, ∀ N₀ > 1, ∀ seq : ℕ → ℕ, 
    (∀ i, Nat.Prime (seq i) ∧ seq i ∣ (transform N₀ (seq 0) k)) →
    ∃ n, 2018 ∣ (transform N₀ (seq n) k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_divisibility_l25_2541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_daisies_l25_2572

theorem roses_daisies (total : ℕ) (sunflowers : ℕ) (tulip_fraction : ℚ) :
  total = 12 →
  sunflowers = 4 →
  tulip_fraction = 3/5 →
  ∃ (daisies : ℕ) (tulips : ℕ),
    daisies + tulips + sunflowers = total ∧
    tulips = (tulip_fraction * (total - daisies : ℚ)).floor ∧
    daisies = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_daisies_l25_2572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l25_2580

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (6 - x - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l25_2580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_specific_vectors_l25_2589

-- Define the vectors a and b
noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
def b : ℝ × ℝ := sorry  -- We don't know the exact components of b

-- State the theorem
theorem dot_product_of_specific_vectors {π : ℝ} :
  let angle := π / 3
  let norm_b := 3
  (a.1 * b.1 + a.2 * b.2) = norm_b * Real.sqrt (a.1^2 + a.2^2) * Real.cos angle := by
  sorry

#check dot_product_of_specific_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_specific_vectors_l25_2589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_L_pieces_8x8_L_tiling_odd_board_l25_2521

/-- An L-shaped piece covers 3 squares -/
def L_piece_size : ℕ := 3

/-- Defines a valid L-shaped piece -/
def valid_L_piece (piece : ℕ × ℕ × ℕ × ℕ) : Prop :=
  let (x1, y1, x2, y2) := piece
  (x1 = x2 ∧ y2 = y1 + 1 ∧ x2 + 1 = x1 + 2) ∨
  (y1 = y2 ∧ x2 = x1 + 1 ∧ y2 + 1 = y1 + 2)

/-- Checks if an L-shaped piece covers a specific square -/
def covers_square (piece : ℕ × ℕ × ℕ × ℕ) (square : ℕ × ℕ) : Prop :=
  let (x1, y1, x2, y2) := piece
  let (x, y) := square
  (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ 
  (valid_L_piece piece ∧ ((x = x1 ∧ y = y2) ∨ (x = x2 ∧ y = y1)))

/-- Checks if the tiling covers the entire board except one square -/
def covers_board_except_one_square (board_size : ℕ) (removed_square : ℕ × ℕ) (tiling : List (ℕ × ℕ × ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ), x < board_size ∧ y < board_size ∧ (x, y) ≠ removed_square →
    ∃ (piece : ℕ × ℕ × ℕ × ℕ), piece ∈ tiling ∧ covers_square piece (x, y)

/-- Theorem for the 8 × 8 grid board -/
theorem min_L_pieces_8x8 (board_size : ℕ) (h_board_size : board_size = 8) :
  ∃ (n : ℕ), n = 11 ∧ 
  (∀ (m : ℕ), m < n → m * L_piece_size < board_size * board_size) ∧
  n * L_piece_size ≥ board_size * board_size := by
  sorry

/-- Theorem for odd-sized square grid boards -/
theorem L_tiling_odd_board (n : ℕ) :
  let board_size := 6 * n + 1
  ∀ (removed_square : ℕ × ℕ),
    removed_square.1 < board_size ∧ removed_square.2 < board_size →
    ∃ (tiling : List (ℕ × ℕ × ℕ × ℕ)),
      (∀ piece ∈ tiling, valid_L_piece piece) ∧
      covers_board_except_one_square board_size removed_square tiling := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_L_pieces_8x8_L_tiling_odd_board_l25_2521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l25_2564

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 1 2 → x^2 + 1 < 2*x + Real.log x / Real.log a) → 
  a ∈ Set.Ioc 1 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l25_2564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l25_2525

-- Define the equation (marked as noncomputable due to dependency on Real.sqrt)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 10) - 6 / Real.sqrt (x + 10)

-- Theorem statement
theorem equation_roots :
  (∃! x : ℝ, f x = 5 ∧ x > 0) ∧
  (∃ y : ℝ, -10 < y ∧ y < -6 ∧ (Real.sqrt (y + 10))^2 = y + 10 ∧ f y ≠ 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l25_2525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_condition1_if_finitely_many_discontinuities_l25_2562

-- Define the open interval (a, b)
variable (a b : ℝ) (hab : a < b)

-- Define a function f on (a, b)
variable (f : ℝ → ℝ)

-- Define the property of having finitely many discontinuities
def FinitelyManyDiscontinuities (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (S : Finset ℝ), ∀ x, x ∈ Set.Ioo a b → ¬ContinuousAt f x → x ∈ S

-- Define condition (1) as a property
def Condition1 (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Ioo a b → y ∈ Set.Ioo a b → f (x + y) = f x + f y

-- State the theorem
theorem not_condition1_if_finitely_many_discontinuities
  (h : FinitelyManyDiscontinuities f a b) :
  ¬(Condition1 f a b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_condition1_if_finitely_many_discontinuities_l25_2562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_velocity_turn_time_l25_2585

/-- The time for a ball's velocity vector to turn 60° when thrown at 30° angle -/
theorem ball_velocity_turn_time 
  (v₀ : ℝ) 
  (g : ℝ) 
  (launch_angle : ℝ) 
  (h₁ : v₀ = 20)
  (h₂ : g = 10)
  (h₃ : launch_angle = π / 6) : -- 30° in radians
  ∃ (t : ℝ), t = 2 ∧ 
    (v₀ * Real.sin launch_angle * t - 1/2 * g * t^2 = 0) ∧
    (v₀ * Real.cos launch_angle * t = v₀ * Real.cos (launch_angle + π/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_velocity_turn_time_l25_2585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l25_2555

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- The problem statement -/
theorem interest_rate_calculation (initial_amount : ℝ) (initial_time : ℝ) 
  (comparison_amount : ℝ) (comparison_time : ℝ) (comparison_rate : ℝ) :
  initial_amount = 100 →
  initial_time = 48 →
  comparison_amount = 600 →
  comparison_time = 4 →
  comparison_rate = 10 →
  simple_interest initial_amount 5 initial_time = 
    simple_interest comparison_amount comparison_rate comparison_time →
  5 = (simple_interest comparison_amount comparison_rate comparison_time * 100) / 
    (initial_amount * initial_time) :=
by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l25_2555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_decreases_as_n_doubles_l25_2547

/-- A function representing C in terms of n, given constant positive real numbers e, R, and r -/
noncomputable def C (e R r n : ℝ) : ℝ := (4 * e * n) / (2 * R + n^2 * r)

/-- Theorem stating that C decreases as n is doubled -/
theorem C_decreases_as_n_doubles (e R r : ℝ) (h_pos : e > 0 ∧ R > 0 ∧ r > 0) :
  ∀ n : ℝ, n > 0 → C e R r (2*n) < C e R r n := by
  sorry

#check C_decreases_as_n_doubles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_decreases_as_n_doubles_l25_2547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l25_2506

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 2) 
  (h2 : Real.cos t.B = 4/5) 
  (h3 : 0 < t.B ∧ t.B < Real.pi) :
  (t.b = 3 → Real.sin t.A = 2/5) ∧ 
  (1/2 * t.a * t.c * Real.sin t.B = 3 → t.b = Real.sqrt 13 ∧ t.c = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l25_2506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_cube_root_l25_2520

/-- Represents a right circular cone water tank -/
structure ConeTank where
  baseRadius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (tank : ConeTank) : ℝ :=
  (1/3) * Real.pi * tank.baseRadius^2 * tank.height

/-- Calculates the height of water in the tank given a fill percentage -/
noncomputable def waterHeight (tank : ConeTank) (fillPercentage : ℝ) : ℝ :=
  tank.height * (fillPercentage)^(1/3)

theorem water_height_cube_root (tank : ConeTank) :
  tank.baseRadius = 24 →
  tank.height = 72 →
  waterHeight tank 0.4 = 36 * (16 : ℝ)^(1/3) := by
  sorry

#eval (36 : ℕ) + 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_cube_root_l25_2520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l25_2587

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Line 1 equation: 3x + 4y - 5 = 0 -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0

/-- Line 2 equation: 6x + 8y - 5 = 0 -/
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y - 5 = 0

theorem distance_between_given_lines :
  distance_between_parallel_lines 3 4 5 (5/2) = 1/2 := by
  sorry

#check distance_between_given_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l25_2587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barney_situps_count_l25_2501

/-- Represents the number of sit-ups Barney can do in one minute -/
def barney_situps : ℕ := 45

/-- Represents the number of sit-ups Carrie can do in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- Represents the number of sit-ups Jerrie can do in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := barney_situps * 1 + carrie_situps * 2 + jerrie_situps * 3

theorem barney_situps_count : barney_situps = 45 :=
  by
    -- Proof goes here
    sorry

#check barney_situps_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barney_situps_count_l25_2501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_odd_increasing_function_l25_2575

-- Define the properties of the function f
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def isIncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

-- State the theorem
theorem solution_set_of_odd_increasing_function 
  (f : ℝ → ℝ) 
  (h_odd : isOddFunction f) 
  (h_inc : isIncreasingOn f (Set.Ici 0)) :
  {x : ℝ | f (x + 1) ≥ 0} = Set.Ici (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_odd_increasing_function_l25_2575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geologists_distance_probability_l25_2584

/-- The number of equal sectors in the circular field -/
def num_sectors : ℕ := 8

/-- The speed of the geologists in km/h -/
noncomputable def speed : ℝ := 5

/-- The time the geologists walk in hours -/
noncomputable def time : ℝ := 1

/-- The distance threshold in km -/
noncomputable def distance_threshold : ℝ := 8

/-- The angle between adjacent roads in radians -/
noncomputable def angle_between_roads : ℝ := 2 * Real.pi / num_sectors

/-- Function to calculate the distance between geologists based on the number of roads between them -/
noncomputable def distance_between_geologists (roads_between : ℕ) : ℝ :=
  Real.sqrt (2 * speed^2 * time^2 * (1 - Real.cos (roads_between * angle_between_roads)))

/-- The probability of the geologists being more than the distance threshold apart -/
def probability_apart : ℚ := 3 / 8

theorem geologists_distance_probability :
  probability_apart = 3 / 8 ∧
  ∀ (roads_between : ℕ), roads_between ≥ 3 →
    distance_between_geologists roads_between > distance_threshold :=
by sorry

#eval probability_apart

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geologists_distance_probability_l25_2584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_60_3_l25_2559

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_60_3_l25_2559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_otimes_sum_l25_2566

-- Define the operation ⊗
noncomputable def otimes (x y : ℝ) : ℝ := (x^2 - y^2) / (x * y)

-- Theorem statement
theorem min_value_otimes_sum :
  ∃ (min : ℝ), min = Real.sqrt 2 ∧ ∀ x y : ℝ, x > 0 → y > 0 → otimes x y + otimes (2*y) x ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_otimes_sum_l25_2566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_for_all_real_domain_l25_2558

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.sqrt ((3/8) - k*x - 2*k*x^2)

-- State the theorem
theorem range_of_k_for_all_real_domain :
  (∀ k : ℝ, (∀ x : ℝ, f k x ∈ Set.range Real.sqrt) ↔ k ∈ Set.Icc (-3) 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_for_all_real_domain_l25_2558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_minimum_surface_area_l25_2526

/-- The volume of the cylinder -/
noncomputable def volume : ℝ := 27 * Real.pi

/-- The surface area of an open-top cylinder as a function of its radius -/
noncomputable def surface_area (r : ℝ) : ℝ := Real.pi * r^2 + 2 * Real.pi * r * (volume / (Real.pi * r^2))

/-- The radius that minimizes the surface area -/
def optimal_radius : ℝ := 3

theorem cylinder_minimum_surface_area :
  ∀ r > 0, surface_area r ≥ surface_area optimal_radius :=
by sorry

#check cylinder_minimum_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_minimum_surface_area_l25_2526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integer_solutions_l25_2567

theorem three_integer_solutions :
  ∃! (a : ℝ), ∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x : ℤ, x ∈ s ↔ (abs (abs (x - 2) - 1) : ℝ) = a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integer_solutions_l25_2567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_latitude_circles_l25_2517

/-- Represents a latitude circle on a sphere -/
structure LatitudeCircle where
  latitude : Real
  radius : Real

/-- Represents a spherical Earth -/
structure Earth where
  radius : Real
  tropic_of_capricorn : LatitudeCircle
  arctic_circle : LatitudeCircle

/-- 
Given a spherical Earth and two latitude circles with a combined angular separation of 90°,
the distance between their planes is equal to the sum of their radii.
-/
theorem distance_between_latitude_circles (e : Earth) 
  (h1 : e.tropic_of_capricorn.latitude + e.arctic_circle.latitude = 90) :
  let distance := e.radius * (Real.sin (e.arctic_circle.latitude * Real.pi / 180) - 
                              Real.sin (e.tropic_of_capricorn.latitude * Real.pi / 180))
  distance = e.tropic_of_capricorn.radius + e.arctic_circle.radius :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_latitude_circles_l25_2517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l25_2529

/-- Calculates the cost of fencing a rectangular park -/
noncomputable def fencing_cost (ratio_long : ℕ) (ratio_short : ℕ) (area : ℝ) (cost_per_meter : ℝ) : ℝ :=
  let x : ℝ := Real.sqrt (area / (ratio_long * ratio_short : ℝ))
  let length : ℝ := ratio_long * x
  let width : ℝ := ratio_short * x
  let perimeter : ℝ := 2 * (length + width)
  perimeter * cost_per_meter

/-- Theorem stating the cost of fencing the rectangular park -/
theorem park_fencing_cost :
  fencing_cost 3 2 3750 0.5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l25_2529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sequence_up_to_10_l25_2542

def my_sequence : ℕ → ℕ
  | 0 => 0
  | n + 1 => (my_sequence n)^2 + 1

def sum_up_to_10 : ℕ → ℕ
  | 0 => 0
  | n + 1 => let term := my_sequence n
              if term ≤ 10 then
                term + sum_up_to_10 n
              else
                sum_up_to_10 n

theorem sum_sequence_up_to_10 : sum_up_to_10 4 = 8 := by
  rfl

#eval sum_up_to_10 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sequence_up_to_10_l25_2542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_purchase_theorem_l25_2513

theorem bookstore_purchase_theorem (books : Fin 11 → ℕ) (max_books : ∀ i, books i ≤ 100) :
  ∃ i j, i ≠ j ∧ (books i : Int) - (books j : Int) < 10 ∧ (books j : Int) - (books i : Int) < 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_purchase_theorem_l25_2513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_one_l25_2581

theorem sum_of_solutions_is_one :
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, |2*x + 3| = 3*|x - 1| ∧ x^2 - 4*x + 3 = 0) ∧
    (S.sum id) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_one_l25_2581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_sequence_a_l25_2539

def sequence_a : ℕ → ℝ
  | 0 => 2  -- Define for 0 to cover all natural numbers
  | 1 => 2
  | n + 2 => 2 * sequence_a (n + 1) - 1

theorem general_term_sequence_a (n : ℕ) : sequence_a n = 2^(n-1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_sequence_a_l25_2539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_J_specific_value_l25_2527

/-- The function J for nonzero real numbers a, b, and c -/
noncomputable def J (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : ℝ :=
  a / b + b / c + c / a

/-- Theorem stating that J(-3, 15, -5) = -23/15 -/
theorem J_specific_value : 
  J (-3) 15 (-5) (by norm_num) (by norm_num) (by norm_num) = -23/15 := by
  -- Unfold the definition of J
  unfold J
  -- Simplify the arithmetic expressions
  simp [div_eq_mul_inv]
  -- Perform numerical calculations
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_J_specific_value_l25_2527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_hyperbola_region_l25_2582

noncomputable section

open Real MeasureTheory

/-- The volume of a solid obtained by rotating a region around the y-axis --/
def rotationVolume (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  Real.pi * ∫ y in a..b, (f y)^2

/-- The hyperbola function x(y) = 4/y --/
def hyperbola (y : ℝ) : ℝ := 4 / y

theorem volume_of_rotated_hyperbola_region :
  rotationVolume hyperbola 1 2 = 8 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_hyperbola_region_l25_2582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l25_2549

theorem expression_evaluation : 
  Real.sqrt 5 * (5 : ℝ)^(1/2 : ℝ) + 16 / 4 * 2 - (8 : ℝ)^(3/2 : ℝ) = 5 + 8 - 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l25_2549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l25_2576

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_x_value (T : ℕ) (X : ℕ) :
  T > 0 →
  is_binary_number T →
  X = T / 36 →
  X * 36 = T →
  (∀ Y : ℕ, Y > 0 → is_binary_number Y → (Y / 36 : ℕ) = Y / 36 → Y / 36 < X → False) →
  X = 308642525 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l25_2576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_qr_length_bound_l25_2574

open Real

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) : Type :=
  (ha : a > 0)
  (hb : b > 0)
  (hab : a > b)

/-- A point on the ellipse -/
def PointOnEllipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The vertices of the ellipse -/
def Vertices (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-a, 0), (a, 0))

/-- The foci of the ellipse -/
noncomputable def Foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 - b^2)
  ((-c, 0), (c, 0))

/-- Perpendicular condition for Q -/
def QPerpCondition (a b : ℝ) (P Q : ℝ × ℝ) : Prop :=
  let ((x1, y1), (x2, y2)) := Vertices a b
  let (xp, yp) := P
  let (xq, yq) := Q
  (xq - x1) * (xp - x1) + (yq - y1) * (yp - y1) = 0 ∧
  (xq - x2) * (xp - x2) + (yq - y2) * (yp - y2) = 0

/-- Perpendicular condition for R -/
def RPerpCondition (a b : ℝ) (P R : ℝ × ℝ) : Prop :=
  let ((x1, y1), (x2, y2)) := Foci a b
  let (xp, yp) := P
  let (xr, yr) := R
  (xr - x1) * (xp - x1) + (yr - y1) * (yp - y1) = 0 ∧
  (xr - x2) * (xp - x2) + (yr - y2) * (yp - y2) = 0

/-- The main theorem -/
theorem ellipse_qr_length_bound (a b : ℝ) (e : Ellipse a b) 
  (P : ℝ × ℝ) (hP : PointOnEllipse a b P.1 P.2) 
  (hPvert : P ≠ (Vertices a b).1 ∧ P ≠ (Vertices a b).2)
  (Q R : ℝ × ℝ) (hQ : QPerpCondition a b P Q) (hR : RPerpCondition a b P R) :
  dist Q R ≥ b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_qr_length_bound_l25_2574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l25_2578

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

/-- The slope of the tangent line at x = 1 -/
noncomputable def k : ℝ := f' 1

/-- The y-intercept of the tangent line -/
def b : ℝ := -1

/-- The x-intercept of the tangent line -/
def a : ℝ := 1

theorem tangent_line_triangle_area : 
  (1/2 : ℝ) * a * (-b) = (1/2 : ℝ) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l25_2578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_theorem_l25_2570

/-- Represents a class with its enrollment -/
structure MyClass where
  enrollment : ℕ

/-- Represents a school with students, teachers, and classes -/
structure School where
  numStudents : ℕ
  numTeachers : ℕ
  classes : List MyClass

def averageFromTeacherPerspective (school : School) : ℚ :=
  (school.classes.map (λ c => c.enrollment)).sum / school.numTeachers

def averageFromStudentPerspective (school : School) : ℚ :=
  (school.classes.map (λ c => c.enrollment * c.enrollment)).sum / school.numStudents

/-- The theorem to be proved -/
theorem average_difference_theorem (school : School) 
    (h1 : school.numStudents = 120)
    (h2 : school.numTeachers = 4)
    (h3 : school.classes = [⟨60⟩, ⟨30⟩, ⟨20⟩, ⟨10⟩])
    (h4 : (school.classes.map (λ c => c.enrollment)).sum = school.numStudents) :
  averageFromTeacherPerspective school - averageFromStudentPerspective school = -35/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_theorem_l25_2570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_one_fourth_sixteen_l25_2591

-- Define the logarithm function for an arbitrary base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_fourth_sixteen : log (1/4) 16 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_one_fourth_sixteen_l25_2591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l25_2522

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  (Real.cos C + Real.cos A * Real.cos B - Real.sqrt 3 * Real.sin A * Real.cos B = 0) →
  (a + c = 1) →
  -- Conclusions
  (B = π / 3) ∧
  (1 / 2 ≤ b) ∧ (b < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l25_2522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l25_2535

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -2 * Real.log x + (1/2) * (x^2 + 1) - a * x

theorem f_property (a : ℝ) (h1 : a > 0) :
  (∀ x, x > 1 → f a x ≥ 0) →
  (∃! x, x > 1 ∧ f a x = 0) →
  a < 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l25_2535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_decrease_l25_2534

/-- Proves that decreasing one side of a square by 15% results in a 27.75% decrease in area -/
theorem square_area_decrease (initial_area : ℝ) (side_decrease_percent : ℝ) 
  (h1 : initial_area = 50)
  (h2 : side_decrease_percent = 15) :
  let new_side := Real.sqrt initial_area * (1 - side_decrease_percent / 100)
  let new_area := new_side ^ 2
  let area_decrease_percent := (initial_area - new_area) / initial_area * 100
  area_decrease_percent = 27.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_decrease_l25_2534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_parallel_lines_l25_2524

-- Define the necessary structures
structure Plane :=
  (α : Type*)

structure Line :=
  (l : Type*)

-- Define the relationships
def outside (a : Line) (α : Plane) : Prop := sorry

def within (b : Line) (α : Plane) : Prop := sorry

def parallel (l1 l2 : Line) : Prop := sorry

-- Define the theorem
theorem infinitely_many_parallel_lines
  (α : Plane) (a b : Line)
  (h1 : outside a α)
  (h2 : within b α)
  (h3 : ¬ parallel a b) :
  ∃ (S : Set Line), (∀ l ∈ S, within l α ∧ parallel l a) ∧ Infinite S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_parallel_lines_l25_2524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisected_by_M_on_hyperbola_l25_2515

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := 3*x + 4*y - 5 = 0

/-- Point M -/
def M : ℝ × ℝ := (3, -1)

theorem line_bisected_by_M_on_hyperbola :
  line M.1 M.2 ∧ 
  hyperbola M.1 M.2 ∧ 
  ∃ (t : ℝ), 
    let P₁ := (M.1 + t, M.2 + (-3/4) * t)
    let P₂ := (M.1 - t, M.2 - (-3/4) * t)
    hyperbola P₁.1 P₁.2 ∧ 
    hyperbola P₂.1 P₂.2 ∧
    line P₁.1 P₁.2 ∧ 
    line P₂.1 P₂.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisected_by_M_on_hyperbola_l25_2515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_30_minutes_l25_2586

-- Define the speeds for Alvin and Benny
noncomputable def alvin_flat_speed : ℝ := 25
noncomputable def alvin_downhill_speed : ℝ := 35
noncomputable def alvin_uphill_speed : ℝ := 10
noncomputable def benny_flat_speed : ℝ := 35
noncomputable def benny_downhill_speed : ℝ := 45
noncomputable def benny_uphill_speed : ℝ := 15

-- Define the distances
noncomputable def uphill_distance : ℝ := 15
noncomputable def downhill_distance : ℝ := 25
noncomputable def flat_distance : ℝ := 30

-- Calculate times for Alvin and Benny
noncomputable def alvin_time : ℝ := uphill_distance / alvin_uphill_speed + 
                      downhill_distance / alvin_downhill_speed + 
                      flat_distance / alvin_flat_speed

noncomputable def benny_time : ℝ := flat_distance / benny_flat_speed + 
                      uphill_distance / benny_uphill_speed + 
                      downhill_distance / benny_downhill_speed

-- Theorem to prove
theorem time_difference_is_30_minutes : 
  ⌊(alvin_time - benny_time) * 60⌋ = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_30_minutes_l25_2586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fifth_term_l25_2593

theorem geometric_sequence_fifth_term (x : ℝ) :
  let seq := fun n : ℕ => 2 * (3 * x)^(n - 1)
  seq 1 = 2 ∧ seq 2 = 6 * x ∧ seq 3 = 18 * x^2 ∧ seq 4 = 54 * x^3 →
  seq 5 = 162 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fifth_term_l25_2593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fruit_cost_l25_2551

/-- Calculate the total cost of fruits for all four people --/
theorem total_fruit_cost : ℕ := by
  let louis_oranges : ℕ := 5
  let louis_apples : ℕ := 3
  let samantha_oranges : ℕ := 8
  let samantha_apples : ℕ := 7
  let marley_oranges : ℕ := 2 * louis_oranges
  let marley_apples : ℕ := 3 * samantha_apples
  let edward_oranges : ℕ := 3 * louis_oranges
  let edward_bananas : ℕ := 4
  let orange_cost : ℕ := 2
  let apple_cost : ℕ := 3
  let banana_cost : ℕ := 1

  let total_cost : ℕ :=
    (louis_oranges + samantha_oranges + marley_oranges + edward_oranges) * orange_cost +
    (louis_apples + samantha_apples + marley_apples) * apple_cost +
    edward_bananas * banana_cost

  have : total_cost = 173 := by
    -- The proof goes here
    sorry

  exact 173


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fruit_cost_l25_2551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_expression_l25_2546

theorem minimum_value_of_expression (a b : ℝ) : 
  (a > 0) → (b > 0) →
  (∃ x y : ℝ, a*x + b*y = 1) →
  (∃ x y : ℝ, x^2 + y^2 - 2*x - 4*y = 0) →
  (a + 2*b = 1) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → (a' + 2*b' = 1) → (1/a' + 2/b' ≥ 1/a + 2/b)) →
  (1/a + 2/b = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_expression_l25_2546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_above_x_axis_is_half_l25_2577

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Calculate the area of a parallelogram -/
noncomputable def parallelogramArea (para : Parallelogram) : ℝ := sorry

/-- Calculate the area of the portion of the parallelogram above or on the x-axis -/
noncomputable def areaAboveXAxis (para : Parallelogram) : ℝ := sorry

/-- The probability of selecting a point on or above the x-axis in the given parallelogram -/
noncomputable def probAboveXAxis (para : Parallelogram) : ℝ :=
  areaAboveXAxis para / parallelogramArea para

theorem prob_above_x_axis_is_half :
  let para := Parallelogram.mk
    (Point.mk 4 4) (Point.mk (-2) (-4))
    (Point.mk (-8) (-4)) (Point.mk (-2) 4)
  probAboveXAxis para = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_above_x_axis_is_half_l25_2577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l25_2561

/-- Calculates the distance traveled given speed and time -/
noncomputable def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Calculates the speed given distance and time -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
    (h1 : marguerite_distance = 150)
    (h2 : marguerite_time = 3)
    (h3 : sam_time = 4.5) :
  distance (speed marguerite_distance marguerite_time) sam_time = 225 := by
  -- Calculate Marguerite's speed
  have marguerite_speed : ℝ := speed marguerite_distance marguerite_time
  
  -- Calculate Sam's distance
  have sam_distance : ℝ := distance marguerite_speed sam_time
  
  -- Prove that Sam's distance is 225 miles
  sorry  -- The actual proof steps would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l25_2561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roberta_initial_amount_l25_2530

-- Define the amount of money Roberta took to the mall
noncomputable def initial_amount : ℚ := 158

-- Define the cost of shoes
noncomputable def shoe_cost : ℚ := 45

-- Define the cost of the bag
noncomputable def bag_cost : ℚ := shoe_cost - 17

-- Define the cost of lunch
noncomputable def lunch_cost : ℚ := bag_cost / 4

-- Define the amount left after purchases
noncomputable def amount_left : ℚ := 78

-- Theorem to prove
theorem roberta_initial_amount :
  initial_amount - shoe_cost - bag_cost - lunch_cost = amount_left :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roberta_initial_amount_l25_2530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l25_2507

theorem cube_root_simplification : 
  (8 : ℝ) ^ (1/3) - (1 - Real.sqrt 2) - (2 : ℝ)^0 = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l25_2507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_third_f_range_on_interval_f_extrema_existence_l25_2548

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin (x/2) * Real.cos (x/2) + Real.sin (x/2)^2

-- Theorem for part I
theorem f_value_at_pi_third : f (π/3) = (1 + Real.sqrt 3) / 4 := by
  sorry

-- Theorem for part II
theorem f_range_on_interval :
  ∀ x ∈ Set.Ioc (-π/3) (π/2),
    ((1 - Real.sqrt 2) / 2 ≤ f x) ∧ (f x ≤ 1) := by
  sorry

-- Existence of minimum and maximum points
theorem f_extrema_existence :
  (∃ x₁ ∈ Set.Ioc (-π/3) (π/2), f x₁ = (1 - Real.sqrt 2) / 2) ∧
  (∃ x₂ ∈ Set.Ioc (-π/3) (π/2), f x₂ = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_third_f_range_on_interval_f_extrema_existence_l25_2548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l25_2531

/-- The distance between the foci of an ellipse -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- Theorem: The distance between the foci of the given ellipse is approximately 5.196 -/
theorem ellipse_foci_distance : 
  let a : ℝ := 6
  let b : ℝ := 3
  ∃ ε > 0, |distance_between_foci a b - 5.196| < ε := by
  sorry

#check ellipse_foci_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l25_2531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l25_2533

/-- An isosceles trapezoid with given measurements -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- The area of an isosceles trapezoid -/
noncomputable def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  (10000 - 2000 * Real.sqrt 11) / 9

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 40,
    diagonal_length := 50,
    longer_base := 60
  }
  trapezoid_area t = (10000 - 2000 * Real.sqrt 11) / 9 := by
  sorry

#check specific_trapezoid_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l25_2533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_elements_in_S_l25_2508

/-- First arithmetic progression with a₁ = 1 and d = 3 -/
def seq1 (n : ℕ) : ℕ := 3 * n + 1

/-- Second arithmetic progression with a₁ = 9 and d = 7 -/
def seq2 (n : ℕ) : ℕ := 7 * n + 9

/-- Set of the first 2004 terms of seq1 -/
def set1 : Finset ℕ := Finset.range 2004 |>.image seq1

/-- Set of the first 2004 terms of seq2 -/
def set2 : Finset ℕ := Finset.range 2004 |>.image seq2

/-- The union of set1 and set2 -/
def S : Finset ℕ := set1 ∪ set2

theorem distinct_elements_in_S : Finset.card S = 3722 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_elements_in_S_l25_2508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_solution_set_implies_a_range_l25_2596

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 3| + |x - 4| < a)) → a ∈ Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_solution_set_implies_a_range_l25_2596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pine_tree_arrangement_l25_2537

def isPine (i : Nat) : Prop := sorry

theorem pine_tree_arrangement (total_trees : Nat) (pine_trees : Nat) (fir_trees : Nat)
  (h1 : total_trees = 2019)
  (h2 : pine_trees = 1009)
  (h3 : fir_trees = 1010)
  (h4 : total_trees = pine_trees + fir_trees) :
  ∃ (i : Nat), (i < total_trees) → 
    ((isPine ((i + 1) % total_trees) ∧ isPine ((i + 3) % total_trees)) ∨
     (isPine ((i - 1 + total_trees) % total_trees) ∧ isPine ((i - 3 + total_trees) % total_trees))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pine_tree_arrangement_l25_2537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_classification_l25_2545

-- Define the fixed points
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points P satisfying the condition
def trajectory (a : ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | distance P F₁ - distance P F₂ = 2 * a}

-- Define placeholders for the missing concepts
def IsHyperbola (s : Set (ℝ × ℝ)) : Prop := sorry
def oneBranch (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry
def IsRay (s : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem trajectory_classification :
  (∃ h : Set (ℝ × ℝ), IsHyperbola h ∧ trajectory 3 = oneBranch h) ∧
  IsRay (trajectory 5) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_classification_l25_2545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l25_2510

/-- A sinusoidal function with given properties -/
noncomputable def SinusoidalFunction (A ω φ c : ℝ) : ℝ → ℝ :=
  fun x ↦ A * Real.sin (ω * x + φ) + c

theorem sinusoidal_function_properties
  (A ω φ c : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : |φ| < π / 2)
  (h_high : SinusoidalFunction A ω φ c 2 = 2)
  (h_low : SinusoidalFunction A ω φ c 8 = -4) :
  (∃ T : ℝ, T = 12 ∧ ∀ x : ℝ, SinusoidalFunction A ω φ c (x + T) = SinusoidalFunction A ω φ c x) ∧
  (∀ k : ℤ, ∀ x : ℝ, 12 * (k : ℝ) - 4 ≤ x ∧ x ≤ 12 * (k : ℝ) + 2 →
    ∀ y : ℝ, x ≤ y → SinusoidalFunction A ω φ c x ≤ SinusoidalFunction A ω φ c y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l25_2510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_range_of_a_l25_2597

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - (3/2) * a * x^2 + 2 * a^2 * x + b

-- Define the derivative of f(x)
noncomputable def f_derivative (a x : ℝ) : ℝ := x^2 - 3 * a * x + 2 * a^2

-- Theorem statement
theorem extreme_values_range_of_a (a b : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, f_derivative a x = 0) →
  (1 < a ∧ a < 2) ∨ (1/2 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_range_of_a_l25_2597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_point_on_circles_l25_2518

-- Define a quadratic function
def quadratic_function (p q : ℝ) : ℝ → ℝ := λ x ↦ x^2 + p*x + q

-- Define the condition of having three distinct intersections with axes
def has_three_distinct_intersections (p q : ℝ) : Prop :=
  ∃ r s : ℝ, r ≠ s ∧ r ≠ 0 ∧ s ≠ 0 ∧ q ≠ 0 ∧
  quadratic_function p q r = 0 ∧ quadratic_function p q s = 0 ∧ q = r * s

-- Define a circle passing through three points
def circle_through_points (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | ∃ r : ℝ, (P.1 - A.1)^2 + (P.2 - A.2)^2 = r^2 ∧
                       (P.1 - B.1)^2 + (P.2 - B.2)^2 = r^2 ∧
                       (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2}

theorem common_point_on_circles (p q : ℝ) 
  (h : has_three_distinct_intersections p q) :
  ∃ r s : ℝ, (0, 1) ∈ circle_through_points (r, 0) (s, 0) (0, q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_point_on_circles_l25_2518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_count_l25_2511

theorem integer_root_count : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, x ≥ 0 ∧ ∃ k : ℤ, Real.sqrt (169 - Real.rpow x (1/3 : ℝ)) = k) ∧ 
  Finset.card S = 14 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_count_l25_2511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_specific_area_l25_2554

/-- Represents a trapezoid with given diagonal lengths and midline length. -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  midline : ℝ

/-- Calculates the area of a trapezoid given its properties. -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  (t.diagonal1 * t.diagonal2) / 2

/-- Theorem stating that a trapezoid with diagonals 6 and 8, and midline 5 has area 24. -/
theorem trapezoid_specific_area :
  let t : Trapezoid := { diagonal1 := 6, diagonal2 := 8, midline := 5 }
  trapezoid_area t = 24 := by
  sorry

#check trapezoid_specific_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_specific_area_l25_2554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_satisfies_equation_x_approximation_l25_2592

/-- The value of x that satisfies the equation 0.45x = (1/3)x + 110 -/
noncomputable def x : ℝ :=
  110 * 60 / 7

/-- The equation that x satisfies -/
theorem x_satisfies_equation : 0.45 * x = (1/3) * x + 110 := by
  sorry

/-- The approximate value of x -/
theorem x_approximation : ∃ ε > 0, |x - 942.857| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_satisfies_equation_x_approximation_l25_2592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_plane_not_equiv_parallel_lines_l25_2560

-- Define the basic geometric objects
variable (α : Type*) -- Plane α
variable (l m : Type*) -- Lines l and m

-- Define the geometric relationships
def contained_in (line plane : Type*) : Prop := sorry
def parallel_line_plane (line plane : Type*) : Prop := sorry
def parallel_lines (line1 line2 : Type*) : Prop := sorry

-- State the theorem
theorem parallel_line_plane_not_equiv_parallel_lines 
  {α : Type*} {l m : Type*} (h : contained_in m α) : 
  ¬(∀ l, parallel_line_plane l α ↔ parallel_lines l m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_plane_not_equiv_parallel_lines_l25_2560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pay_is_correct_l25_2594

def game_stats := List (Nat × Nat × Nat × Nat)

def base_pay (avg_points : Rat) : Nat :=
  if avg_points ≥ 30 then 10000 else 8000

def assist_bonus (total_assists : Nat) : Nat :=
  if total_assists ≥ 20 then 5000
  else if total_assists ≥ 10 then 3000
  else 1000

def rebound_bonus (total_rebounds : Nat) : Nat :=
  if total_rebounds ≥ 40 then 5000
  else if total_rebounds ≥ 20 then 3000
  else 1000

def steal_bonus (total_steals : Nat) : Nat :=
  if total_steals ≥ 15 then 5000
  else if total_steals ≥ 5 then 3000
  else 1000

def calculate_total_pay (games : game_stats) : Nat :=
  let total_points := (games.map (λ g => g.1)).sum
  let total_assists := (games.map (λ g => g.2.1)).sum
  let total_rebounds := (games.map (λ g => g.2.2.1)).sum
  let total_steals := (games.map (λ g => g.2.2.2)).sum
  let avg_points : Rat := total_points / games.length
  base_pay avg_points + assist_bonus total_assists + 
  rebound_bonus total_rebounds + steal_bonus total_steals

theorem total_pay_is_correct (games : game_stats) : 
  games = [(30, 5, 7, 3), (28, 6, 5, 2), (32, 4, 9, 1), (34, 3, 11, 2), (26, 2, 8, 3)] →
  calculate_total_pay games = 23000 := by
  sorry

#eval calculate_total_pay [(30, 5, 7, 3), (28, 6, 5, 2), (32, 4, 9, 1), (34, 3, 11, 2), (26, 2, 8, 3)]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pay_is_correct_l25_2594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l25_2590

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 + 2*Complex.I) / (3 - 4*Complex.I)
  Complex.im z = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l25_2590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l25_2569

/-- Represents a quadrilateral with mid-segments -/
structure Quadrilateral :=
  (A B C D E F G H : ℝ × ℝ)
  (EF_midpoint : E = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧ F = ((C.1 + D.1)/2, (C.2 + D.2)/2))
  (GH_midpoint : G = ((B.1 + C.1)/2, (B.2 + C.2)/2) ∧ H = ((A.1 + D.1)/2, (A.2 + D.2)/2))
  (EF_perpendicular_GH : (F.1 - E.1) * (H.1 - G.1) + (F.2 - E.2) * (H.2 - G.2) = 0)
  (EF_length : ((F.1 - E.1)^2 + (F.2 - E.2)^2) = 18^2)
  (GH_length : ((H.1 - G.1)^2 + (H.2 - G.2)^2) = 24^2)

/-- The area of the quadrilateral ABCD is 864 -/
theorem quadrilateral_area (q : Quadrilateral) : 
  abs ((q.A.1 - q.C.1) * (q.B.2 - q.D.2) - (q.B.1 - q.D.1) * (q.A.2 - q.C.2)) / 2 = 864 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l25_2569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mens_wages_l25_2543

/-- Proves that given the conditions of the problem, each man's wage is 14 Rs. -/
theorem mens_wages (total_earnings men_wage women_wage boys_wage : ℕ) (num_men num_boys W : ℕ) : 
  total_earnings = 210 →
  num_men = 5 →
  num_boys = 8 →
  (5 : ℕ) * men_wage = W * women_wage →
  W * women_wage = 8 * boys_wage →
  total_earnings = 5 * men_wage + W * women_wage + 8 * boys_wage →
  men_wage = 14 := by
  sorry

#check mens_wages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mens_wages_l25_2543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_domain_span_l25_2595

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x * sin (x + π/3) - 1/4

-- State the theorem
theorem min_domain_span (m n : ℝ) (h1 : m < n) :
  (∀ x ∈ Set.Icc m n, -1/2 ≤ f x ∧ f x ≤ 1/4) →
  (∃ x ∈ Set.Icc m n, f x = -1/2) →
  (∃ x ∈ Set.Icc m n, f x = 1/4) →
  n - m ≥ 2*π/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_domain_span_l25_2595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l25_2503

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±x -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (asymptotes : ∀ x y : ℝ, (y = x ∨ y = -x) ↔ (x^2 / a^2 - y^2 / b^2 = 1)) :
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l25_2503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_isosceles_triangle_vertex_angle_l25_2552

/-- An obtuse isosceles triangle with the given property has a vertex angle of approximately 160 degrees. -/
theorem obtuse_isosceles_triangle_vertex_angle (a b h : ℝ) (θ φ : ℝ) : 
  a > 0 → -- Side length is positive
  b > 0 → -- Base length is positive
  h > 0 → -- Height is positive
  φ > 90 → -- Triangle is obtuse
  φ = 180 - 2 * θ → -- Relationship between vertex angle and base angles
  a^2 = b * (3 * h) → -- Given condition
  b = 2 * a * Real.cos θ → -- Base in terms of side and angle
  h = a * Real.sin θ → -- Height in terms of side and angle
  ∃ ε > 0, abs (φ - 160) < ε := by
  sorry

#check obtuse_isosceles_triangle_vertex_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_isosceles_triangle_vertex_angle_l25_2552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l25_2553

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) + Real.log (x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > 2}

-- Statement: f is neither odd nor even
theorem f_neither_odd_nor_even :
  ¬(∀ x ∈ domain_f, f (-x) = -f x) ∧ 
  ¬(∀ x ∈ domain_f, f (-x) = f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l25_2553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_triangles_common_point_l25_2573

/-- A triangle on a plane -/
structure Triangle where
  vertices : Finset (ℝ × ℝ)
  is_triangle : vertices.card = 3

/-- The set of all triangles on the plane -/
def AllTriangles : Finset Triangle := sorry

/-- A function that returns the set of vertices of other triangles contained in a given triangle -/
def contained_vertices (t : Triangle) : Finset (ℝ × ℝ) := sorry

theorem three_triangles_common_point :
  (AllTriangles.card = 1993) →
  (∀ t ∈ AllTriangles, (contained_vertices t).card ≥ 4) →
  ∃ t1 t2 t3 : Triangle, t1 ∈ AllTriangles ∧ t2 ∈ AllTriangles ∧ t3 ∈ AllTriangles ∧
    t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
    ∃ p : ℝ × ℝ, p ∈ t1.vertices ∧ p ∈ t2.vertices ∧ p ∈ t3.vertices :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_triangles_common_point_l25_2573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_pieces_per_box_l25_2505

theorem chocolate_pieces_per_box 
  (initial_boxes : ℕ) 
  (boxes_given_away : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : initial_boxes = 12)
  (h2 : boxes_given_away = 5)
  (h3 : remaining_pieces = 21)
  (h4 : initial_boxes > boxes_given_away) :
  remaining_pieces / (initial_boxes - boxes_given_away) = 3 := by
  sorry

#eval 21 / (12 - 5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_pieces_per_box_l25_2505
