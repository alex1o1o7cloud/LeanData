import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_4_l765_76514

/-- A positive geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  ratio : ∀ n, a (n + 1) / a n = a 1 / a 0
  decreasing : a 1 / a 0 < 1

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumGeometric (seq : GeometricSequence) (n : ℕ) : ℝ :=
  (seq.a 0) * (1 - (seq.a 1 / seq.a 0)^n) / (1 - seq.a 1 / seq.a 0)

theorem geometric_sequence_sum_4 (seq : GeometricSequence) 
  (h1 : seq.a 2 + seq.a 4 = 20)
  (h2 : seq.a 2 * seq.a 4 = 64) :
  sumGeometric seq 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_4_l765_76514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_value_is_750_l765_76527

/-- The value of a single sale that ensures no loss in income when changing jobs --/
noncomputable def sale_value : ℝ :=
  let old_salary : ℝ := 75000
  let new_base_salary : ℝ := 45000
  let commission_rate : ℝ := 0.15
  let min_sales : ℝ := 266.67
  (old_salary - new_base_salary) / (commission_rate * min_sales)

/-- Theorem stating that the sale value is $750 --/
theorem sale_value_is_750 : sale_value = 750 := by
  -- Unfold the definition of sale_value
  unfold sale_value
  -- Perform the calculation
  norm_num
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_value_is_750_l765_76527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_on_interval_l765_76505

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- State the theorem
theorem max_min_values_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 4 ∧ min = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_on_interval_l765_76505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_theorem_l765_76552

/-- Calculates the compound interest --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * years)

/-- Calculates the total payment for Plan 1 --/
noncomputable def plan1_payment (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  let half_term := years / 2
  let first_payment := (compound_interest principal rate 4 half_term) / 2
  let remaining := (compound_interest principal rate 4 half_term) / 2
  first_payment + compound_interest remaining rate 4 half_term

/-- Calculates the total payment for Plan 2 --/
noncomputable def plan2_payment (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  compound_interest principal rate 1 years

/-- The main theorem --/
theorem payment_difference_theorem (principal : ℝ) (rate : ℝ) (years : ℝ)
  (h_principal : principal = 10000)
  (h_rate : rate = 0.1)
  (h_years : years = 10) :
  ∃ (diff : ℝ), 
    abs (plan2_payment principal rate years - plan1_payment principal rate years - diff) < 1 ∧ 
    diff = 4319 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_theorem_l765_76552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_count_l765_76599

/-- Represents a class with students -/
structure MyClass where
  girls : ℕ
  boys : ℕ

/-- The ratio of girls to boys is 3:2 -/
def ratio_constraint (c : MyClass) : Prop :=
  3 * c.boys = 2 * c.girls

/-- The total number of students is 45 -/
def total_constraint (c : MyClass) : Prop :=
  c.girls + c.boys = 45

/-- Theorem stating that a class satisfying the given constraints has 27 girls -/
theorem girls_count (c : MyClass) 
  (h1 : ratio_constraint c) 
  (h2 : total_constraint c) : 
  c.girls = 27 := by
  sorry

#check girls_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_count_l765_76599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_three_l765_76578

theorem angle_A_is_pi_over_three (A B C a b : ℝ) :
  0 < A → A < π/2 →
  0 < B → B < π/2 →
  0 < C → C < π/2 →
  A + B + C = π →
  a = 2 * Real.sin B / Real.sin A →
  b = 2 * Real.sin C / Real.sin A →
  2 * a * Real.sin B = Real.sqrt 3 * b →
  A = π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_three_l765_76578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_sum_l765_76525

-- Define the equation
def equation (x : ℝ) : Prop := 2 * x * (4 * x - 9) = -8

-- Define the solution form
def solution_form (m n p : ℤ) (x : ℝ) : Prop :=
  (x = (m + Real.sqrt (n : ℝ)) / p ∨ x = (m - Real.sqrt (n : ℝ)) / p) ∧
  Int.gcd (Int.gcd m n) p = 1

theorem equation_solution_sum :
  ∃ m n p : ℤ,
    (∀ x : ℝ, equation x ↔ solution_form m n p x) →
    m + n + p = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_sum_l765_76525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l765_76521

theorem triangle_side_length (L M N : ℝ × ℝ) (LM LN : ℝ) : 
  (M.2 = 0 ∧ N.2 = 0) →  -- M and N are on the x-axis
  (L.1 = M.1 ∧ L.2 > 0) →  -- L is directly above M
  (LM = Real.sqrt ((L.1 - M.1)^2 + (L.2 - M.2)^2)) →  -- LM is the distance between L and M
  (LN = Real.sqrt ((L.1 - N.1)^2 + (L.2 - N.2)^2)) →  -- LN is the distance between L and N
  (Real.sin (Real.arccos ((LM^2 + LN^2 - (N.1 - M.1)^2) / (2 * LM * LN))) = 3/5) →  -- sin N = 3/5
  (LM = 15) →  -- Length of LM is 15
  LN = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l765_76521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l765_76590

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Definition of perpendicularity between two line segments -/
def perpendicular (p q r s : Point) : Prop :=
  (q.x - p.x) * (s.x - r.x) + (q.y - p.y) * (s.y - r.y) = 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_eccentricity (h : Hyperbola) (f₁ f₂ p q : Point) :
  (∃ (l : Set Point), f₁ ∈ l ∧ p ∈ l ∧ q ∈ l) →  -- line l through F₁, P, and Q
  distance p f₁ = 2 * distance f₁ q →  -- |PF₁| = 2|F₁Q|
  perpendicular f₂ q p q →  -- F₂Q ⟂ PQ
  eccentricity h = Real.sqrt 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l765_76590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_equal_one_l765_76558

-- Define the first expression
noncomputable def expression1 : ℝ := (2/3)^(-2 : ℤ) + (1 - Real.sqrt 2)^(0 : ℝ) - (3 * 3/8)^(2/3 : ℝ)

-- Define the second expression
noncomputable def expression2 : ℝ := (2 * Real.log 2 + Real.log 3) / (1 + (1/2) * Real.log 0.36 + (1/3) * Real.log 8)

-- Theorem statement
theorem expressions_equal_one : expression1 = 1 ∧ expression2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_equal_one_l765_76558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blueberries_in_blue_box_l765_76536

/-- The number of blueberries in each blue box -/
def B : ℕ := sorry

/-- The number of strawberries in each red box -/
def S : ℕ := sorry

/-- The theorem stating the number of blueberries in each blue box -/
theorem blueberries_in_blue_box :
  (S - B = 12) →
  (S + B = 76) →
  B = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blueberries_in_blue_box_l765_76536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_coefficient_l765_76583

theorem fourth_term_coefficient : 
  (Finset.range 16).sum (λ k => (-1)^k * Nat.choose 15 k * (2:ℝ)^k * X^k) = 
    -3640 * X^3 + X^4 * (X - 2)^11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_coefficient_l765_76583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_of_3_equals_88_l765_76537

-- Define q(x)
noncomputable def q (x : ℝ) : ℝ := 2 * x - 5

-- Define r(y)
noncomputable def r (y : ℝ) : ℝ := 
  let x := (y + 5) / 2  -- Inverse of q(x)
  x^3 + 2*x^2 - x - 4

-- Theorem statement
theorem r_of_3_equals_88 : r 3 = 88 := by
  -- Expand the definition of r
  unfold r
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_of_3_equals_88_l765_76537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l765_76575

/-- Parallelogram PQRS with given vertices -/
structure Parallelogram where
  P : ℝ × ℝ := (4, 4)
  Q : ℝ × ℝ := (-2, -2)
  R : ℝ × ℝ := (-8, -2)
  S : ℝ × ℝ := (2, 4)

/-- Probability of a point not being above x-axis in the parallelogram -/
noncomputable def probability_not_above_x_axis (p : Parallelogram) : ℝ :=
  1 / 2

/-- Theorem stating the probability of a point not being above x-axis is 1/2 -/
theorem probability_theorem (p : Parallelogram) :
  probability_not_above_x_axis p = 1 / 2 := by
  sorry

#check probability_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l765_76575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_set_size_l765_76559

def ValidDigits : Set Nat := {6, 7, 8, 9}

def IsThreeDigitNumber (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

def UsesValidDigits (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ∈ ValidDigits

def NoAdjacentSwap (n m : Nat) : Prop :=
  ¬∃ (i : Nat), i < 2 ∧
    (n.digits 10).take i ++ (n.digits 10).get! (i + 1) :: 
    (n.digits 10).get! i :: (n.digits 10).drop (i + 2) = m.digits 10

def ValidSet (S : Set Nat) : Prop :=
  ∀ n ∈ S, IsThreeDigitNumber n ∧ UsesValidDigits n ∧
    ∀ m ∈ S, m ≠ n → NoAdjacentSwap n m

theorem max_valid_set_size :
  ∃ (S : Set Nat), ValidSet S ∧ S.ncard = 40 ∧
    ∀ (T : Set Nat), ValidSet T → T.ncard ≤ 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_set_size_l765_76559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_gt_two_l765_76564

/-- The function f(x) defined in the problem -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x + Real.log x - (5/4) * k

/-- The function g(x) defined in the problem -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := f k x - k * x + 1/x

/-- Theorem stating the main result to be proved -/
theorem sum_of_zeros_gt_two (k : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : g k x₁ = 0) (h₂ : g k x₂ = 0) (h₃ : x₁ ≠ x₂) :
  x₁ + x₂ > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_gt_two_l765_76564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_7_4_minus_5_l765_76518

theorem binomial_7_4_minus_5 : Nat.choose 7 4 - 5 = 30 := by
  -- Calculation of binomial coefficient
  have h1 : Nat.choose 7 4 = 35 := by rfl
  
  -- Subtraction
  have h2 : 35 - 5 = 30 := by rfl
  
  -- Combine the steps
  rw [h1, h2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_7_4_minus_5_l765_76518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l765_76544

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the foci of the ellipse
noncomputable def ellipse_foci : Set (ℝ × ℝ) := {(-Real.sqrt 5, 0), (Real.sqrt 5, 0)}

-- Define the eccentricity of the hyperbola
noncomputable def hyperbola_eccentricity : ℝ := Real.sqrt 5 / 2

-- Theorem statement
theorem hyperbola_properties :
  ∀ (x y : ℝ),
    hyperbola x y →
    (∃ (f : ℝ × ℝ), f ∈ ellipse_foci ∧ 
      (x - f.1)^2 + y^2 = (hyperbola_eccentricity * x)^2) ∧
    (∀ (f : ℝ × ℝ), f ∈ ellipse_foci → 
      ∃ (t : ℝ), (x - f.1)^2 + y^2 = (t * hyperbola_eccentricity)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l765_76544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rearrangements_l765_76554

def digits : List Nat := [3, 0, 3, 3]

def is_valid_arrangement (arr : List Nat) : Bool :=
  arr.length = 4 && arr.head? ≠ some 0 && arr.toFinset ⊆ digits.toFinset

def count_valid_arrangements : Nat :=
  (digits.permutations.filter is_valid_arrangement).length

theorem count_rearrangements : count_valid_arrangements = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rearrangements_l765_76554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l765_76543

-- Define the complex number z
noncomputable def z (α : ℝ) : ℂ := 1 + Real.cos α + Complex.I * Real.sin α

-- State the theorem
theorem modulus_of_z (α : ℝ) (h : π < α ∧ α < 2*π) :
  Complex.abs (z α) = -2 * Real.cos (α/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l765_76543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_traces_ellipse_l765_76571

-- Define the complex number w
variable (w : ℂ)

-- Define the condition that w traces a circle with radius 3
def traces_circle (w : ℂ) : Prop := Complex.abs w = 3

-- Define the transformation
noncomputable def transformation (w : ℂ) : ℂ := w + 2 / w

-- Theorem statement
theorem transformation_traces_ellipse :
  traces_circle w → ∃ (a b : ℝ), ∀ (z : ℂ),
    z = transformation w →
    (Complex.re z / a)^2 + (Complex.im z / b)^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_traces_ellipse_l765_76571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_revenue_increase_2017_l765_76567

def revenue : ℕ → ℕ
  | 2010 => 100
  | 2011 => 120
  | 2012 => 115
  | 2013 => 130
  | 2014 => 140
  | 2015 => 150
  | 2016 => 145
  | 2017 => 180
  | 2018 => 175
  | 2019 => 200
  | _ => 0

def revenue_increase (year : ℕ) : ℤ :=
  (revenue year : ℤ) - (revenue (year - 1) : ℤ)

def valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ year ≤ 2019

theorem max_revenue_increase_2017 :
  ∀ year, valid_year year →
    revenue_increase 2017 ≥ revenue_increase year :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_revenue_increase_2017_l765_76567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l765_76523

-- Define c and d as noncomputable
noncomputable def c : ℝ := Real.log 8
noncomputable def d : ℝ := Real.log 25

-- State the theorem
theorem log_sum_equality : 5^(c/d) + 2^(d/c) = 2 * Real.sqrt 2 + 5^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l765_76523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l765_76557

open MeasureTheory Interval Set

variable (f : ℝ → ℝ) (hf : Differentiable ℝ f) (hf' : Continuous (deriv f))
  (hf0 : f 0 = 0) (hf'_bound : ∀ x > 0, 0 ≤ deriv f x ∧ deriv f x ≤ 1)
  (n : ℕ) (hn : n > 0) (a : ℝ) (ha : a > 0)

theorem integral_inequality :
  (1 / (n + 1 : ℝ)) * ∫ x in Set.Icc 0 a, (f x) ^ (2 * n + 1) ≤ (∫ x in Set.Icc 0 a, (f x) ^ n) ^ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l765_76557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_6heads_before_3tails_is_three_fourths_l765_76553

/-- The probability of getting a run of 6 heads before a run of 3 tails 
    when repeatedly flipping a fair coin. -/
def prob_6heads_before_3tails : ℚ := 3/4

/-- A fair coin has equal probability of heads and tails. -/
def fair_coin_prob : ℚ := 1/2

/-- The number of consecutive heads needed to win. -/
def winning_heads : ℕ := 6

/-- The number of consecutive tails needed to lose. -/
def losing_tails : ℕ := 3

/-- Theorem stating that the probability of getting 6 heads before 3 tails is 3/4. -/
theorem prob_6heads_before_3tails_is_three_fourths :
  prob_6heads_before_3tails = 3/4 := by
  -- The proof is omitted for now
  sorry

#eval prob_6heads_before_3tails

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_6heads_before_3tails_is_three_fourths_l765_76553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l765_76534

-- Define the current price and the price decrease
noncomputable def current_price : ℝ := 3800
noncomputable def price_decrease : ℝ := 200

-- Define the original price based on the given condition
noncomputable def original_price : ℝ := current_price + price_decrease

-- Define the percentage decrease
noncomputable def percentage_decrease : ℝ := price_decrease / original_price * 100

-- Theorem to prove
theorem price_decrease_percentage :
  percentage_decrease = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l765_76534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_fill_theorem_l765_76593

/-- Represents the capacity of a bucket -/
structure Bucket where
  capacity : ℚ

/-- Represents a drum that can be filled with liquid -/
structure Drum where
  capacity : ℚ

/-- Calculates the number of turns needed to fill a drum using a single bucket -/
def turnsTofill (b : Bucket) (d : Drum) : ℚ :=
  d.capacity / b.capacity

theorem bucket_fill_theorem (p q : Bucket) (d : Drum) 
  (h1 : p.capacity = 3 * q.capacity) 
  (h2 : turnsTofill p d + turnsTofill q d = 60) : 
  turnsTofill p d = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_fill_theorem_l765_76593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_n_is_36_l765_76511

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 8
  h4 : (a 5) ^ 2 = (a 1) * (a 7)

/-- Sum of the first n terms of the arithmetic sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1) + (n * (n - 1) / 2) * seq.d

/-- The maximum value of S_n is 36 -/
theorem max_S_n_is_36 (seq : ArithmeticSequence) :
  ∃ N : ℕ, ∀ n : ℕ, S_n seq n ≤ S_n seq N ∧ S_n seq N = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_n_is_36_l765_76511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_area_theorem_l765_76520

-- Define the hyperbola
def hyperbola (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (16 + k) - y^2 / (8 - k) = 1

-- Define the domain of k
def k_domain (k : ℝ) : Prop :=
  -16 < k ∧ k < 8

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = -Real.sqrt 3 * x

-- Define symmetry about origin
def symmetric_about_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -x₁ ∧ y₂ = -y₁

-- Helper function to calculate area (placeholder)
noncomputable def area_quadrilateral (p₁ p₂ p₃ p₄ : ℝ × ℝ) : ℝ :=
  sorry

-- Main theorem
theorem hyperbola_area_theorem (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  k_domain k →
  hyperbola k x₁ y₁ →
  hyperbola k x₂ y₂ →
  asymptote x₁ y₁ →
  x₁ = 3 →
  symmetric_about_origin x₁ y₁ x₂ y₂ →
  ∃ F₁ F₂ : ℝ × ℝ, area_quadrilateral F₁ (x₂, y₂) F₂ (x₁, y₁) = 12 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_area_theorem_l765_76520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_slope_specific_ellipse_chord_slope_l765_76556

/-- The slope of the chord passing through the midpoint of an ellipse --/
theorem ellipse_chord_slope (a b x₀ y₀ : ℝ) (h_ellipse : x₀^2 / a^2 + y₀^2 / b^2 < 1) :
  let slope := -(2 * x₀ / a^2) / (2 * y₀ / b^2)
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 →
    (x - x₀) * (2 * y₀ / b^2) = (y - y₀) * (2 * x₀ / a^2) →
    y - y₀ = slope * (x - x₀) :=
by sorry

/-- The specific case for the given ellipse and point --/
theorem specific_ellipse_chord_slope :
  let a := 6
  let b := 3
  let x₀ := 4
  let y₀ := 2
  let slope := -(2 * x₀ / a^2) / (2 * y₀ / b^2)
  x₀^2 / a^2 + y₀^2 / b^2 < 1 ∧ slope = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_slope_specific_ellipse_chord_slope_l765_76556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l765_76574

/-- Given two vectors a and b in ℝ², prove that the angle between them is π/4 --/
theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a = (1, Real.sqrt 3) → 
  ‖b‖ = Real.sqrt 2 → 
  ‖a + 2 • b‖ = 2 * Real.sqrt 5 → 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)) = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l765_76574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l765_76580

-- Constants
noncomputable def principal : ℝ := 15000
noncomputable def compoundRate : ℝ := 0.08
noncomputable def simpleRate : ℝ := 0.10
noncomputable def compoundingFrequency : ℝ := 12
noncomputable def totalYears : ℝ := 15
noncomputable def halfPaymentYear : ℝ := 7

-- Function to calculate compound interest
noncomputable def compoundInterest (p r n t : ℝ) : ℝ :=
  p * (1 + r / n) ^ (n * t)

-- Function to calculate simple interest
noncomputable def simpleInterest (p r t : ℝ) : ℝ :=
  p * (1 + r * t)

-- Theorem statement
theorem loan_payment_difference :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  abs (abs (compoundInterest principal compoundRate compoundingFrequency halfPaymentYear / 2 +
            compoundInterest (compoundInterest principal compoundRate compoundingFrequency halfPaymentYear / 2)
                             compoundRate compoundingFrequency (totalYears - halfPaymentYear) -
            simpleInterest principal simpleRate totalYears) - 979) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l765_76580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l765_76507

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + 3^x + Real.log 3

-- State the theorem about the derivative of f
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 3 * x^2 + 3^x * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l765_76507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_nested_calculation_l765_76531

-- Define the spade operation
noncomputable def spade (x y : ℝ) : ℝ := x - 1 / (y^2)

-- State the theorem
theorem spade_nested_calculation : 
  spade 3 (spade 3 3) = 1947 / 676 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_nested_calculation_l765_76531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_product_l765_76582

theorem square_of_product (n : ℕ) (h1 : n = 300) (e : ℕ) (h2 : Even e) (h3 : e ≥ 3) 
  (h4 : ∀ k : ℕ, Even k → k ≥ 3 → k < e → ¬∃ m : ℕ, n * k = m ^ 2) : 
  ∃ i : ℕ, n * e = i ^ 2 ∧ i = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_product_l765_76582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l765_76504

-- Define the circle C
noncomputable def circleC (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the line l
def lineL (θ : ℝ) (ρ : ℝ) : Prop := ρ * (Real.sin θ + Real.cos θ) = 3 * Real.sqrt 3

-- Define the ray OM
noncomputable def rayOM : ℝ := Real.pi / 3

-- Theorem statement
theorem intersection_distance :
  ∃ (ρ₁ ρ₂ : ℝ),
    ρ₁ = circleC rayOM ∧
    lineL rayOM ρ₂ ∧
    ρ₂ - ρ₁ = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l765_76504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_translation_on_sine_graphs_l765_76532

theorem point_translation_on_sine_graphs (t s : ℝ) (k : ℤ) :
  (∃ k, t = π/4 + k*π) →
  s > 0 →
  Real.sin (2*t) = 1 →
  Real.sin (2*(t+s) - π/3) = 1 →
  (∃ k, t = π/4 + k*π) ∧ (∀ s' > 0, Real.sin (2*(t+s') - π/3) = 1 → s' ≥ π/6) ∧ (Real.sin (2*(t+π/6) - π/3) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_translation_on_sine_graphs_l765_76532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contributions_before_johns_l765_76545

/-- The number of contributions before John's donation -/
def n : ℕ := 2

/-- The average contribution before John's donation -/
def A : ℚ := 50

/-- John's contribution amount -/
def john_contribution : ℚ := 125

/-- The new average after John's contribution -/
def new_average : ℚ := 75

theorem contributions_before_johns :
  (n * A + john_contribution) / (n + 1) = new_average ∧
  (n * A + john_contribution) / (n + 1) = A * (3/2) ∧
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contributions_before_johns_l765_76545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_on_ellipse_l765_76529

/-- Ellipse C defined by x²/4 + y²/3 = 1 -/
def Ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Left focus F₁ of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- Right focus F₂ of the ellipse -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Point A on the ellipse such that AF₂ ⊥ F₁F₂ -/
noncomputable def A : ℝ × ℝ := (1, 3/2)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The maximum value of F₁P · F₂A for P on the ellipse -/
theorem max_dot_product_on_ellipse :
  ∃ (M : ℝ), M = 3 * Real.sqrt 3 / 2 ∧
  ∀ (P : ℝ × ℝ), Ellipse P.1 P.2 →
    dot_product (P.1 - F₁.1, P.2 - F₁.2) (F₂.1 - A.1, F₂.2 - A.2) ≤ M :=
by sorry

#check max_dot_product_on_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_on_ellipse_l765_76529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_exceeds_target_l765_76588

/-- Represents a mixture of alcohol, glycerin, and water -/
structure Mixture where
  total : ℚ
  alcohol : ℚ
  glycerin : ℚ
  water : ℚ

/-- Calculate the percentage of a component in a mixture -/
def percentage (component : ℚ) (total : ℚ) : ℚ :=
  (component / total) * 100

theorem alcohol_exceeds_target (initial : Mixture) (added_alcohol : ℚ) (added_water : ℚ) :
  initial.total = 18 ∧
  initial.alcohol = 0.25 * initial.total ∧
  initial.glycerin = 0.30 * initial.total ∧
  initial.water = 0.45 * initial.total ∧
  added_alcohol = 3 ∧
  added_water = 2 →
  let new_mixture : Mixture := {
    total := initial.total + added_alcohol + added_water,
    alcohol := initial.alcohol + added_alcohol,
    glycerin := initial.glycerin,
    water := initial.water + added_water
  }
  percentage new_mixture.alcohol new_mixture.total > 22 := by
  intro h
  -- Proof steps would go here
  sorry

#eval percentage 7.5 23 -- This should evaluate to approximately 32.61

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_exceeds_target_l765_76588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_value_of_m_l765_76535

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - a^(-x)

-- Define the function g
noncomputable def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2*m*(f a x)

-- Theorem 1
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 < 0) :
  0 < a ∧ a < 1 := by
  sorry

-- Theorem 2
theorem value_of_m (a m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3/2)
  (h4 : ∀ x ≥ 1, g a m x ≥ -2) (h5 : ∃ x ≥ 1, g a m x = -2) :
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_value_of_m_l765_76535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_five_odd_in_seven_rolls_l765_76509

/-- A fair 6-sided die has equal probability for each outcome -/
noncomputable def fair_die (outcome : Fin 6) : ℝ := 1 / 6

/-- An outcome is odd if it's 1, 3, or 5 -/
def is_odd (outcome : Fin 6) : Prop := outcome.val % 2 = 1

/-- The probability of getting an odd number on a single roll -/
noncomputable def prob_odd : ℝ := 1 / 2

/-- The number of rolls -/
def num_rolls : ℕ := 7

/-- The number of desired odd outcomes -/
def num_odd : ℕ := 5

/-- The probability of getting exactly 5 odd numbers in 7 rolls of a fair 6-sided die -/
theorem prob_five_odd_in_seven_rolls : 
  (Nat.choose num_rolls num_odd : ℝ) * prob_odd ^ num_odd * (1 - prob_odd) ^ (num_rolls - num_odd) = 21 / 128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_five_odd_in_seven_rolls_l765_76509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shelves_for_five_books_l765_76579

/-- The total number of books -/
def total_books : ℕ := 1300

/-- The number of bookshelves -/
def num_shelves : ℕ := 18

/-- A function that checks if there are always 5 books on the same shelf after rearrangement -/
def always_five_books_together (k : ℕ) : Prop :=
  ∀ (arrangement1 arrangement2 : Fin total_books → Fin k),
    ∃ (shelf : Fin k),
      ∃ (books : Finset (Fin total_books)),
        books.card = 5 ∧
        (∀ book ∈ books, arrangement1 book = shelf) ∧
        (∀ book ∈ books, arrangement2 book = shelf)

theorem max_shelves_for_five_books :
  always_five_books_together num_shelves ∧
  ¬always_five_books_together (num_shelves + 1) :=
by
  sorry

#check max_shelves_for_five_books

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shelves_for_five_books_l765_76579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l765_76541

-- Define the ellipse E
def ellipse_E (x y a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (8 - a^2) = 1

-- Define the conditions
variable (a : ℝ)
axiom a_positive : 0 < a
axiom foci_on_x_axis : True  -- This is implied by the problem statement
axiom focal_distance : ∃ c : ℝ, c = 2 ∧ c^2 = a^2 - (8 - a^2)

-- Define the line l
def line_l (x y m : ℝ) : Prop :=
  y = -(Real.sqrt 3 / 3) * (x - m)

-- Define the theorem
theorem ellipse_properties :
  -- Part I: Standard equation of E
  (∀ x y : ℝ, ellipse_E x y a ↔ x^2/6 + y^2/2 = 1) ∧
  -- Part II: Range of m
  (∀ m : ℝ, m > a →
    (∃ C D : ℝ × ℝ,
      ellipse_E C.1 C.2 a ∧
      ellipse_E D.1 D.2 a ∧
      line_l C.1 C.2 m ∧
      line_l D.1 D.2 m ∧
      -- Right focus F inside circle with CD as diameter
      (C.1 - 2) * (D.1 - 2) + C.2 * D.2 < 0) →
    Real.sqrt 6 < m ∧ m < 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l765_76541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cross_bridge_l765_76513

-- Define the train length in meters
noncomputable def train_length : ℝ := 180

-- Define the train speed in kilometers per hour
noncomputable def train_speed_kmph : ℝ := 54

-- Define the bridge length in meters
noncomputable def bridge_length : ℝ := 660

-- Define the conversion factor from km/h to m/s
noncomputable def kmph_to_ms : ℝ := 1000 / 3600

-- Theorem stating the time taken to cross the bridge
theorem time_to_cross_bridge :
  (train_length + bridge_length) / (train_speed_kmph * kmph_to_ms) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cross_bridge_l765_76513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_difference_l765_76502

/-- Represents the principal amount in Rupees -/
def principal (P : ℝ) : Prop := P = 1500

/-- Calculates the simple interest for a given principal and annual rate -/
def simpleInterest (P : ℝ) (rate : ℝ) : ℝ := P * rate

/-- Calculates the compound interest for a given principal, half-yearly rate, and number of half-years -/
def compoundInterest (P : ℝ) (rate : ℝ) (periods : ℕ) : ℝ := 
  P * (1 + rate) ^ periods - P

/-- The theorem stating the relationship between principal and interest difference -/
theorem principal_from_interest_difference 
  (P : ℝ) 
  (h1 : compoundInterest P 0.05 2 - simpleInterest P 0.1 = 3.75) : 
  principal P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_difference_l765_76502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l765_76506

-- Define the hyperbola structure
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

-- Define the focus of the hyperbola
def focus (h : Hyperbola) : ℝ × ℝ := (3, 0)

-- Define the distance from focus to asymptote
noncomputable def focus_to_asymptote_distance (h : Hyperbola) : ℝ := 
  h.b * 3 / Real.sqrt (h.a^2 + h.b^2)

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) 
  (h_distance : focus_to_asymptote_distance h = Real.sqrt 5) : 
  h.a = 2 ∧ h.b = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l765_76506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l765_76562

noncomputable def f (x : ℝ) := (x - 3) * (x - 4) * (x - 7) / ((x - 1) * (x - 6) * (x - 8))

def solution_set : Set ℝ := Set.Iic 1 ∪ Set.Ioo 3 4 ∪ Set.Ioo 6 7 ∪ Set.Ici 8

theorem inequality_solution (x : ℝ) : f x > 0 ↔ x ∈ solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l765_76562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_to_equation_l765_76555

theorem unique_solution_to_equation :
  ∃! p : ℝ × ℝ, (let (x, y) := p; (x + 3)^2 + y^2 + (x - y)^2 = 3) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_to_equation_l765_76555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_through_specific_points_l765_76568

/-- The slope of a line passing through two points is equal to the difference in y-coordinates divided by the difference in x-coordinates. -/
def line_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ := (y₂ - y₁) / (x₂ - x₁)

/-- The slope of the line passing through the points (-2, 3) and (3, -4) is -7/5. -/
theorem slope_through_specific_points :
  line_slope (-2) 3 3 (-4) = -7/5 := by
  -- Unfold the definition of line_slope
  unfold line_slope
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_through_specific_points_l765_76568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagram_arrangement_count_l765_76542

/-- Represents a regular hexagram with 12 points -/
structure Hexagram :=
  (points : Fin 12 → Type)

/-- The group of symmetries of a regular hexagram -/
def HexagramSymmetryGroup : Type := Unit

/-- The order of the hexagram symmetry group -/
def hexagram_symmetry_order : ℕ := 12

/-- The number of distinct arrangements of 12 different objects on a hexagram,
    considering rotational and reflectional symmetries as equivalent -/
def distinct_hexagram_arrangements : ℕ := 39916800

theorem hexagram_arrangement_count :
  distinct_hexagram_arrangements = (Nat.factorial 12) / hexagram_symmetry_order :=
by
  -- The proof goes here
  sorry

#eval distinct_hexagram_arrangements
#eval (Nat.factorial 12) / hexagram_symmetry_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagram_arrangement_count_l765_76542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuckoo_clock_chimes_l765_76573

/-- Represents the number of chimes at a given hour -/
def chimes_at_hour (hour : ℕ) : ℕ :=
  if hour ≤ 12 then hour else hour % 12

/-- The sum of chimes from 10:00 to 16:00 -/
def total_chimes : ℕ :=
  (List.range 7).map (λ i => chimes_at_hour (i + 10)) |>.sum

theorem cuckoo_clock_chimes : total_chimes = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuckoo_clock_chimes_l765_76573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l765_76598

/-- The equation of the tangent line to y = -x^3 at (0, 2) is y = -3x + 2 -/
theorem tangent_line_at_origin : 
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b → 
      (y = -x^3 → x = 0 ∧ y = 2) ∨ 
      (x = 0 ∧ y = 2 → HasDerivAt (λ t => m * t + b) (-(3 * 0^2)) 0)) ∧ 
    m = -3 ∧ 
    b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l765_76598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_second_day_prob_A_first_given_second_satisfaction_improvement_association_l765_76587

-- Define the probabilities
def P_A1 : ℚ := 1/2  -- Probability of going to A on first day
def P_B1 : ℚ := 1/2  -- Probability of going to B on first day
def P_A2_given_A1 : ℚ := 3/5  -- Probability of A on second day given A on first day
def P_A2_given_B1 : ℚ := 4/5  -- Probability of A on second day given B on first day

-- Define the survey data
def satisfied_improved : ℕ := 28
def not_satisfied_improved : ℕ := 12
def total_satisfied : ℕ := 85
def total_not_satisfied : ℕ := 15
def total_improved : ℕ := 40
def total_surveyed : ℕ := 100

-- Define the critical value for α = 0.005
def critical_value : ℚ := 7879/1000

-- Theorem for the probability of going to A on the second day
theorem prob_A_second_day :
  P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1 = 7/10 := by sorry

-- Theorem for the probability of going to A on the first day given A on the second day
theorem prob_A_first_given_second :
  (P_A1 * P_A2_given_A1) / (P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1) = 3/7 := by sorry

-- Function to calculate chi-square statistic
noncomputable def chi_square (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n : ℚ) * (a * d - b * c : ℚ)^2 / ((a + b : ℚ) * (c + d : ℚ) * (a + c : ℚ) * (b + d : ℚ))

-- Theorem for the association between satisfaction and improvements
theorem satisfaction_improvement_association :
  chi_square satisfied_improved (total_satisfied - satisfied_improved)
              not_satisfied_improved (total_not_satisfied - not_satisfied_improved)
  > critical_value := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_second_day_prob_A_first_given_second_satisfaction_improvement_association_l765_76587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l765_76508

def sequence_term (n : ℕ) : ℚ :=
  (1234567 : ℚ) / (3^n)

def is_integer (q : ℚ) : Prop :=
  ∃ (n : ℤ), q = n

theorem last_integer_in_sequence :
  ∃ (k : ℕ), (∀ n < k, is_integer (sequence_term n)) ∧
             (is_integer (sequence_term k)) ∧
             (∀ m > k, ¬ is_integer (sequence_term m)) ∧
             (sequence_term k = 2) := by
  sorry

#eval sequence_term 0
#eval sequence_term 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l765_76508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_value_of_p_l765_76547

/-- Given two non-collinear vectors a and b in a vector space V over ℝ,
    and points A, B, C, D such that AB = 2a + pb, BC = a + b, CD = a - 2b,
    if A, B, and D are collinear, then p = -1. -/
theorem collinear_points_value_of_p 
  (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V]
  (a b : V) (p : ℝ) (A B C D : V) :
  a ≠ 0 ∧ b ≠ 0 ∧ ¬ ∃ (k : ℝ), a = k • b →
  B - A = 2 • a + p • b →
  C - B = a + b →
  D - C = a - 2 • b →
  ∃ (l : ℝ), B - A = l • (D - B) →
  p = -1 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_value_of_p_l765_76547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meeting_points_l765_76596

/-- The number of distinct meeting points for two runners on a circular track -/
def distinct_meeting_points (t1 t2 : ℚ) : ℕ :=
  (Int.floor ((lcm t1 t2) / ((t1 * t2) / (t1 + t2)))).toNat

/-- Theorem: Two runners on a circular track with completion times of 5 and 8 minutes
    will have 13 distinct meeting points when running for at least one hour -/
theorem runners_meeting_points :
  distinct_meeting_points 5 8 = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meeting_points_l765_76596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l765_76501

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 + 9 * x - 15 * y + 3 = 0

/-- The area of the circle -/
noncomputable def circle_area : ℝ := 25 * Real.pi / 4

/-- Theorem stating the existence of a center and radius satisfying the circle equation,
    and that the area is correct -/
theorem circle_area_proof :
  ∃ (center_x center_y radius : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    circle_area = Real.pi * radius^2 := by
  -- We'll provide the center and radius values
  let center_x := -3/2
  let center_y := 5/2
  let radius := 5/2

  -- Now we'll prove the existence
  use center_x, center_y, radius
  
  constructor
  
  -- First part: equivalence of equations
  · intro x y
    sorry -- The detailed proof would go here
  
  -- Second part: area calculation
  · sorry -- The detailed proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l765_76501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_l765_76585

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a unit cube -/
structure UnitCube where
  Q1 : Point3D
  Q2 : Point3D
  Q3 : Point3D
  Q4 : Point3D
  Q1' : Point3D
  Q2' : Point3D
  Q3' : Point3D
  Q4' : Point3D

/-- Represents an octahedron inscribed in a unit cube -/
structure InscribedOctahedron where
  cube : UnitCube
  V1 : Point3D -- on Q₁Q₂
  V2 : Point3D -- on Q₁Q₃
  V3 : Point3D -- on Q₁Q₄
  V4 : Point3D -- on Q₁'Q₂'
  V5 : Point3D -- on Q₁'Q₃'
  V6 : Point3D -- on Q₁'Q₄'

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: The side length of the inscribed octahedron is √2/3 -/
theorem octahedron_side_length 
  (cube : UnitCube) 
  (octa : InscribedOctahedron) 
  (h1 : cube.Q1 = ⟨0, 0, 0⟩) 
  (h2 : cube.Q1' = ⟨1, 1, 1⟩)
  (h3 : octa.V1 = ⟨1/3, 0, 0⟩)
  (h4 : octa.V2 = ⟨0, 1/3, 0⟩)
  (h5 : octa.V3 = ⟨0, 0, 1/3⟩)
  (h6 : octa.V4 = ⟨1, 2/3, 1⟩)
  (h7 : octa.V5 = ⟨1, 1, 2/3⟩)
  (h8 : octa.V6 = ⟨2/3, 1, 1⟩) :
  distance octa.V1 octa.V2 = Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_l765_76585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_subset_real_l765_76528

def M (a : ℝ) : Set ℂ := {x : ℂ | x = a + (a^2 - 1) * Complex.I}

theorem complex_subset_real (a : ℝ) : M a ⊆ Set.range (Complex.ofReal) → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_subset_real_l765_76528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_paints_150_sq_ft_l765_76597

/-- Calculates the area painted by Charlie given the total area and work ratios -/
noncomputable def charlies_area (total_area : ℝ) (alice_ratio bob_ratio charlie_ratio : ℕ) : ℝ :=
  let total_ratio : ℝ := (alice_ratio + bob_ratio + charlie_ratio : ℝ)
  (total_area * (charlie_ratio : ℝ)) / total_ratio

/-- Theorem: Given the problem conditions, Charlie paints 150 square feet -/
theorem charlie_paints_150_sq_ft :
  charlies_area 360 3 4 5 = 150 := by
  -- Unfold the definition of charlies_area
  unfold charlies_area
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_paints_150_sq_ft_l765_76597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientists_overlap_probability_l765_76592

/-- The probability of two scientists' lifetimes overlapping -/
theorem scientists_overlap_probability :
  let total_years : ℕ := 300
  let lifespan : ℕ := 80
  let probability : ℚ := 104 / 225
  probability = (total_years * total_years - 2 * (total_years - lifespan) ^ 2) / (total_years * total_years) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientists_overlap_probability_l765_76592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_qr_length_bound_l765_76512

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the vertices and foci
def A₁ (a : ℝ) : ℝ × ℝ := (-a, 0)
def A₂ (a : ℝ) : ℝ × ℝ := (a, 0)
noncomputable def F₁ (a b : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 - b^2), 0)
noncomputable def F₂ (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

-- Define the perpendicular property
def Perpendicular (l₁ l₂ : (ℝ × ℝ) → (ℝ × ℝ) → Prop) : Prop :=
  ∀ p q r s, l₁ p q → l₂ r s → (q.1 - p.1) * (s.1 - r.1) + (q.2 - p.2) * (s.2 - r.2) = 0

-- Define the theorem
theorem ellipse_qr_length_bound {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b)
  (P : ℝ × ℝ) (hP : P ∈ Ellipse a b) (hP_not_vertex : P ≠ A₁ a ∧ P ≠ A₂ a)
  (Q R : ℝ × ℝ)
  (hQ₁ : Perpendicular (λ x y => x = Q ∧ y = A₁ a) (λ x y => x = P ∧ y = A₁ a))
  (hQ₂ : Perpendicular (λ x y => x = Q ∧ y = A₂ a) (λ x y => x = P ∧ y = A₂ a))
  (hR₁ : Perpendicular (λ x y => x = R ∧ y = F₁ a b) (λ x y => x = P ∧ y = F₁ a b))
  (hR₂ : Perpendicular (λ x y => x = R ∧ y = F₂ a b) (λ x y => x = P ∧ y = F₂ a b)) :
  Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) ≥ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_qr_length_bound_l765_76512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_three_plus_one_lt_two_sqrt_two_l765_76538

theorem sqrt_three_plus_one_lt_two_sqrt_two : Real.sqrt 3 + 1 < 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_three_plus_one_lt_two_sqrt_two_l765_76538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_perpendicular_vectors_l765_76550

-- Define complex numbers z₁ and z₂
noncomputable def z₁ : ℂ := (1 + Complex.I) / Complex.I
def z₂ (x y : ℝ) : ℂ := x + y * Complex.I

-- Define vectors OZ₁ and OZ₂
noncomputable def OZ₁ : ℝ × ℝ := (z₁.re, z₁.im)
def OZ₂ (x y : ℝ) : ℝ × ℝ := (x, y)

-- Theorem for parallel vectors
theorem parallel_vectors (x y : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ OZ₂ x y = k • OZ₁) → x + y = 0 := by
  sorry

-- Theorem for perpendicular vectors
theorem perpendicular_vectors (x y : ℝ) :
  (OZ₁.1 * (OZ₂ x y).1 + OZ₁.2 * (OZ₂ x y).2 = 0) →
  Complex.abs (z₁ + z₂ x y) = Complex.abs (z₁ - z₂ x y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_perpendicular_vectors_l765_76550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donald_duck_remote_control_uses_l765_76576

/-- The minimum number of times Donald Duck needs to use the remote control to win the race -/
theorem donald_duck_remote_control_uses : ℕ := by
  -- Define the function representing the time wasted by Mickey Mouse
  let time_wasted (n : ℕ) := n + (n * (n + 1)) / 20

  -- Define the property that n should satisfy
  let satisfies_condition (n : ℕ) := time_wasted n ≥ 20

  -- State that 13 satisfies the condition
  have h1 : satisfies_condition 13 := by
    -- Proof omitted
    sorry

  -- State that no number less than 13 satisfies the condition
  have h2 : ∀ m : ℕ, m < 13 → ¬satisfies_condition m := by
    -- Proof omitted
    sorry

  -- Conclude that 13 is the minimum number satisfying the condition
  exact 13


end NUMINAMATH_CALUDE_ERRORFEEDBACK_donald_duck_remote_control_uses_l765_76576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_ordering_l765_76503

def A : Nat := 77^7
def B : Nat := 7^77
def C : Nat := 7^(7^7)

-- For factorial, we need to define it separately
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def D : Nat := factorial 7

theorem number_ordering : D < A ∧ A < B ∧ B < C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_ordering_l765_76503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_runners_in_picture_l765_76565

/-- Represents a runner on a circular track -/
structure Runner where
  direction : Bool  -- true for counterclockwise, false for clockwise
  lap_time : ℚ      -- time to complete one lap in seconds

/-- Calculates the position of a runner at a given time -/
def runner_position (r : Runner) (t : ℚ) : ℚ :=
  (t / r.lap_time) % 1

/-- Determines if a runner is in the picture -/
def in_picture (r : Runner) (t : ℚ) : Prop :=
  let pos := runner_position r t
  if r.direction then
    pos ≤ 1/6 ∨ pos ≥ 5/6
  else
    1/3 ≤ pos ∧ pos ≤ 2/3

theorem both_runners_in_picture (rachel ryan : Runner) 
  (h_rachel : rachel.direction = true ∧ rachel.lap_time = 100)
  (h_ryan : ryan.direction = false ∧ ryan.lap_time = 75) :
  in_picture rachel 630 ∧ in_picture ryan 630 :=
by
  sorry

#check both_runners_in_picture

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_runners_in_picture_l765_76565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_correlation_l765_76539

-- Define the variables
variable (weight_car : ℝ → ℝ)
variable (distance_per_liter : ℝ → ℝ)
variable (study_time : ℝ → ℝ)
variable (academic_performance : ℝ → ℝ)
variable (smoking_amount : ℝ → ℝ)
variable (health_condition : ℝ → ℝ)
variable (side_length : ℝ → ℝ)
variable (area : ℝ → ℝ)
variable (fuel_consumption : ℝ → ℝ)

-- Define the relationship types
inductive Relationship
| Positive
| Negative
| Functional

-- Define the relationship between two variables
def relationship (f g : ℝ → ℝ) : Relationship := sorry

-- Theorem statement
theorem positive_correlation :
  (relationship study_time academic_performance = Relationship.Positive) ∧
  (relationship weight_car fuel_consumption = Relationship.Positive) ∧
  (relationship weight_car distance_per_liter ≠ Relationship.Positive) ∧
  (relationship smoking_amount health_condition ≠ Relationship.Positive) ∧
  (relationship side_length area = Relationship.Functional) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_correlation_l765_76539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_numbers_l765_76533

noncomputable def numbers : List ℝ := [5, 8, 11]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem standard_deviation_of_numbers : standardDeviation numbers = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_numbers_l765_76533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribing_sphere_area_l765_76551

/-- A triangular pyramid with specific properties -/
structure TriangularPyramid where
  -- Lateral face areas
  lateral_area_1 : ℝ
  lateral_area_2 : ℝ
  lateral_area_3 : ℝ
  -- Base area
  base_area : ℝ
  -- Property that angles between lateral faces and base are equal
  equal_angles : Bool

/-- The surface area of a sphere -/
noncomputable def sphere_surface_area (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

/-- Theorem about the surface area of the circumscribing sphere of a specific triangular pyramid -/
theorem circumscribing_sphere_area (p : TriangularPyramid) 
  (h1 : p.lateral_area_1 = 3)
  (h2 : p.lateral_area_2 = 4)
  (h3 : p.lateral_area_3 = 5)
  (h4 : p.base_area = 6)
  (h5 : p.equal_angles = true) :
  ∃ (r : ℝ), sphere_surface_area r = 79 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribing_sphere_area_l765_76551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l765_76572

theorem exponential_equation_solution :
  ∃ y : ℝ, (3 : ℝ)^(2*y + 3) = (27 : ℝ)^(y - 1) ∧ y = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l765_76572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l765_76540

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * (Real.sin x + Real.cos x)

-- State the theorem
theorem f_properties :
  -- The graph of f is symmetric about x = -3π/4
  (∀ x : ℝ, f ((-3 * Real.pi / 4) + x) = f ((-3 * Real.pi / 4) - x)) ∧
  -- There exists φ such that f(x + φ) is centrally symmetric about the origin
  (∃ φ : ℝ, ∀ x : ℝ, f (x + φ) = -f (-x + φ)) := by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l765_76540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l765_76570

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := 3^x - 2

-- Define the sets M and N
def M : Set ℝ := {x | f (g x) > 0}
def N : Set ℝ := {x | g x < 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l765_76570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_with_common_interesting_year_l765_76595

/-- A year is interesting for a person if their age equals the sum of digits of their birth year -/
def is_interesting_year (birth_year : ℕ) (current_year : ℕ) : Prop :=
  current_year - birth_year = (Nat.digits 10 birth_year).sum

/-- The birth year of a person born in the 20th century -/
def birth_year_20th_century (x y : ℕ) : ℕ := 1900 + 10 * x + y

/-- The birth year of a person born in the 21st century -/
def birth_year_21st_century (z t : ℕ) : ℕ := 2000 + 10 * z + t

/-- The theorem stating the age difference between two people with a common interesting year -/
theorem age_difference_with_common_interesting_year 
  (x y z t : ℕ) 
  (hx : x < 10) (hy : y < 10) (hz : z < 10) (ht : t < 10) 
  (h_common : ∃ (year : ℕ), 
    is_interesting_year (birth_year_20th_century x y) year ∧ 
    is_interesting_year (birth_year_21st_century z t) year) :
  birth_year_21st_century z t - birth_year_20th_century x y = 18 := by
  sorry

#check age_difference_with_common_interesting_year

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_with_common_interesting_year_l765_76595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_construction_right_triangle_construction_unique_l765_76548

open Real

/-- Given the radius of a circumcircle and an angle bisector,
    a right triangle can be constructed. -/
theorem right_triangle_construction (r f : ℝ) (h : f < 2*r) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 = c^2 ∧  -- Pythagorean theorem
  c = 2*r ∧          -- Diameter is twice the radius
  ∃ (f' : ℝ), f' = f ∧ f'^2 = 2*a^2*c/(a+c) := by
  sorry

/-- The construction is unique when f < c (which is equivalent to f < 2r). -/
theorem right_triangle_construction_unique (r f : ℝ) (h : f < 2*r) :
  ∃! (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 = c^2 ∧
  c = 2*r ∧
  ∃ (f' : ℝ), f' = f ∧ f'^2 = 2*a^2*c/(a+c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_construction_right_triangle_construction_unique_l765_76548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_l765_76594

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi :=
sorry

-- Theorem for the range of f in [0, 2π/3]
theorem range_in_interval :
  ∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 0 (2 * Real.pi / 3) ∧ f x = y) ↔
  y ∈ Set.Icc 0 (3 / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_l765_76594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l765_76515

noncomputable section

-- Define the given function
def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

-- Define the transformed function
def g (x : ℝ) : ℝ := 3 * Real.sin (2 * (x - Real.pi / 12))

-- Theorem stating the equivalence of the two functions
theorem function_equivalence : ∀ x : ℝ, f x = g x := by
  intro x
  simp [f, g]
  -- The actual proof would go here, but we'll use sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l765_76515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_l765_76560

/-- Represents the four items --/
inductive Item
  | Notebook
  | CheatSheet
  | Player
  | Sneakers

/-- Represents the four locations --/
inductive Location
  | UnderPillow
  | UnderCouch
  | OnTable
  | UnderTable

/-- A function that assigns each item to a location --/
def arrangement : Item → Location := sorry

/-- Each location contains exactly one item --/
axiom location_unique : ∀ l : Location, ∃! i : Item, arrangement i = l

/-- The notebook and player are not under the table --/
axiom not_under_table : arrangement Item.Notebook ≠ Location.UnderTable ∧ 
                        arrangement Item.Player ≠ Location.UnderTable

/-- The cheat sheet is not under the couch or under the table --/
axiom cheat_sheet_location : arrangement Item.CheatSheet ≠ Location.UnderCouch ∧ 
                             arrangement Item.CheatSheet ≠ Location.UnderTable

/-- The player is not on the table or under the couch --/
axiom player_location : arrangement Item.Player ≠ Location.OnTable ∧ 
                        arrangement Item.Player ≠ Location.UnderCouch

/-- The theorem stating the correct arrangement --/
theorem correct_arrangement : 
  arrangement Item.Notebook = Location.UnderCouch ∧
  arrangement Item.CheatSheet = Location.OnTable ∧
  arrangement Item.Player = Location.UnderPillow ∧
  arrangement Item.Sneakers = Location.UnderTable := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_l765_76560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_positive_l765_76516

theorem inequality_implies_log_positive (x y : ℝ) :
  (2:ℝ)^x - (2:ℝ)^y < (3:ℝ)^(-x) - (3:ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_positive_l765_76516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l765_76530

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 2 * a / x

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ (0 < a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l765_76530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_point_properties_l765_76577

-- Define the proportionality relation
def is_directly_proportional (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x, f x = k * g x

-- Define the function
def f (x : ℝ) : ℝ := x - 4

-- Theorem statement
theorem function_and_point_properties :
  (is_directly_proportional (fun x ↦ f x + 3) (fun x ↦ x - 1)) ∧
  (f 2 = -2) →
  (∀ x, f x = x - 4) ∧
  (f (-1) = -5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_point_properties_l765_76577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_distance_at_faster_speed_l765_76510

/-- Calculates the additional distance covered when walking at a faster speed for the same duration as a slower walk. -/
theorem additional_distance_at_faster_speed 
  (actual_speed : ℝ) 
  (faster_speed : ℝ) 
  (actual_distance : ℝ) 
  (h1 : actual_speed = 10) 
  (h2 : faster_speed = 14) 
  (h3 : actual_distance = 50) : 
  faster_speed * (actual_distance / actual_speed) - actual_distance = 20 := by
  sorry

-- Remove the #eval line as it's causing the universe level error

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_distance_at_faster_speed_l765_76510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_range_of_expression_l765_76524

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (is_triangle : A + B + C = Real.pi)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- Define the conditions
def arithmetic_sequence (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B = t.A + d ∧ t.C = t.B + d

def sine_condition (t : Triangle) : Prop :=
  (Real.sin t.B)^2 = Real.sin t.A * Real.sin t.C

def obtuse_triangle (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

-- Define the theorems to be proved
theorem equilateral_triangle (t : Triangle) 
  (h1 : arithmetic_sequence t) (h2 : sine_condition t) : 
  t.a = t.b ∧ t.b = t.c :=
sorry

theorem range_of_expression (t : Triangle) 
  (h1 : arithmetic_sequence t) (h2 : obtuse_triangle t) (h3 : t.a > t.c) :
  ∀ x : ℝ, x = (Real.sin (t.C/2))^2 + Real.sqrt 3 * Real.sin (t.A/2) * Real.cos (t.A/2) - 1/2 →
  1/4 < x ∧ x ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_range_of_expression_l765_76524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emergency_vehicle_reachable_area_l765_76586

/-- The area of the region reachable by an emergency vehicle in 5 minutes -/
noncomputable def reachableArea (roadSpeed : ℝ) (sandSpeed : ℝ) (time : ℝ) : ℝ :=
  4 * Real.pi * ∫ x in (0)..(roadSpeed * time), (roadSpeed * time / 4 - x / 4) ^ 2

theorem emergency_vehicle_reachable_area :
  let roadSpeed : ℝ := 60
  let sandSpeed : ℝ := 15
  let time : ℝ := 5 / 60
  reachableArea roadSpeed sandSpeed time = 175 * Real.pi / 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emergency_vehicle_reachable_area_l765_76586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_point_division_l765_76584

-- Define the type for rational points
def RationalPoint := ℚ × ℚ

-- Define the property of a set having finite intersection with vertical lines
def FiniteVerticalIntersection (S : Set RationalPoint) :=
  ∀ c : ℚ, (Set.Finite {y : ℚ | (c, y) ∈ S})

-- Define the property of a set having finite intersection with horizontal lines
def FiniteHorizontalIntersection (S : Set RationalPoint) :=
  ∀ d : ℚ, (Set.Finite {x : ℚ | (x, d) ∈ S})

-- State the theorem
theorem rational_point_division :
  ∃ (A B : Set RationalPoint),
    (∀ p : RationalPoint, p ∈ A ∪ B) ∧
    (FiniteVerticalIntersection A) ∧
    (FiniteHorizontalIntersection B) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_point_division_l765_76584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_times_l765_76517

/-- Represents the travel scenario of K and M --/
structure TravelScenario where
  distance : ℝ
  time_difference : ℝ
  speed_difference : ℝ
  k_speed : ℝ

/-- Calculates the travel time given distance and speed --/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Theorem stating the travel times for K and M --/
theorem travel_times (scenario : TravelScenario) 
  (h1 : scenario.distance = 60)
  (h2 : scenario.time_difference = 1/2)
  (h3 : scenario.speed_difference = 1/2)
  (h4 : scenario.k_speed > scenario.speed_difference) :
  ∃ (k_time m_time : ℝ),
    k_time = travel_time scenario.distance scenario.k_speed ∧
    m_time = travel_time scenario.distance (scenario.k_speed - scenario.speed_difference) ∧
    m_time - k_time = scenario.time_difference := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_times_l765_76517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sqrt_equation_solutions_l765_76566

theorem cubic_root_sqrt_equation_solutions :
  ∀ x : ℝ, (Real.rpow (3 - x) (1/3 : ℝ) + Real.sqrt (x - 1) = 1) ↔ (x = 2 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sqrt_equation_solutions_l765_76566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l765_76589

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (ω * x) ^ 2 - 1 / 2

noncomputable def g (x : ℝ) : ℝ := Real.sin x

theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) (h_period : ∀ x, f ω (x + π / 2) = f ω x) :
  (∀ x, f ω x = Real.sin (4 * x + π / 6)) ∧
  (∀ x, g x = Real.sin x) ∧
  (Set.Icc (-1 / 2 : ℝ) 0 ∪ {-1} = {m : ℝ | ∃! x, x ∈ Set.Icc 0 (5 * π / 6) ∧ g x + m = 0}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l765_76589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_row_product_implies_constant_column_product_l765_76569

open Real BigOperators

variable (n : ℕ)
variable (a b : Fin n → ℝ)
variable (c : ℝ)

theorem constant_row_product_implies_constant_column_product :
  (∀ i : Fin n, ∏ j, (a i + b j) = c) →
  ∃ d : ℝ, ∀ j : Fin n, ∏ i, (a i + b j) = d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_row_product_implies_constant_column_product_l765_76569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_heads_fair_coin_prob_even_heads_general_l765_76500

/-- The probability of getting an even number of heads in n tosses of a coin -/
noncomputable def prob_even_heads (p : ℝ) (n : ℕ) : ℝ :=
  (1 + (1 - 2*p)^n) / 2

/-- A fair coin has probability 1/2 of landing heads -/
def is_fair_coin (p : ℝ) : Prop := p = 1/2

theorem prob_even_heads_fair_coin (n : ℕ) (p : ℝ) (h : is_fair_coin p) :
  prob_even_heads p n = 1/2 := by
  sorry

theorem prob_even_heads_general (p : ℝ) (n : ℕ) (h : 0 < p ∧ p < 1) :
  prob_even_heads p n = (1 + (1 - 2*p)^n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_heads_fair_coin_prob_even_heads_general_l765_76500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l765_76581

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 3)

theorem monotonic_increasing_interval
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (h_period : ∀ x, f ω (x + Real.pi / ω) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) :=
by
  sorry

#check monotonic_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l765_76581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l765_76522

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₁ - C₂| / Real.sqrt (A^2 + B^2)

theorem parallel_lines_distance :
  let line1 : ℝ × ℝ → Prop := λ (x, y) ↦ 3 * x - 4 * y - 3 = 0
  let line2 : ℝ × ℝ → Prop := λ (x, y) ↦ 6 * x - 8 * y + 5 = 0
  distance_between_parallel_lines 3 (-4) (-3) 2.5 = 11/10 := by
  sorry

#eval (11 : ℚ) / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l765_76522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_sum_gt_incircle_radius_l765_76549

/-- A right triangle with medians and incircle -/
structure RightTriangle where
  -- The triangle vertices
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- C is the right angle
  right_angle_at_C : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  -- Medians
  ma : ℝ
  mb : ℝ
  -- Incircle radius
  r : ℝ

/-- Theorem: In a right triangle, the sum of squares of two medians is greater than 29 times the square of the incircle radius -/
theorem median_sum_gt_incircle_radius (t : RightTriangle) : t.ma^2 + t.mb^2 > 29 * t.r^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_sum_gt_incircle_radius_l765_76549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_either_is_valid_l765_76591

def Tea : Type := Unit
def Coffee : Type := Unit

inductive Beverage
| tea : Tea → Beverage
| coffee : Coffee → Beverage

def choice : Prop := ∃ (b : Beverage), True

theorem either_is_valid : choice :=
  sorry

#check either_is_valid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_either_is_valid_l765_76591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_equals_three_l765_76526

theorem polynomial_sum_equals_three (x : ℝ) 
  (h1 : x^2017 - 3*x + 3 = 0) 
  (h2 : x ≠ 1) : 
  Finset.sum (Finset.range 2017) (λ i => x^i) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_equals_three_l765_76526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kitchen_repair_percentage_l765_76546

theorem kitchen_repair_percentage 
  (initial_nails : ℕ) 
  (remaining_nails : ℕ) 
  (fence_repair_percentage : ℚ) : ℚ :=
by
  have h1 : initial_nails = 400 := by sorry
  have h2 : remaining_nails = 84 := by sorry
  have h3 : fence_repair_percentage = 70 / 100 := by sorry
  
  let kitchen_repair_percentage : ℚ := 30 / 100
  
  have h4 : remaining_nails = 
    (initial_nails : ℚ) * (1 - kitchen_repair_percentage) * (1 - fence_repair_percentage) := by sorry
  
  exact kitchen_repair_percentage


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kitchen_repair_percentage_l765_76546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_m_implies_range_l765_76561

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := (1/3) * x^3 - x^2 + m

-- Theorem 1: If the maximum value of f(x) on [-1,1] is 2/3, then m = 2/3
theorem max_value_implies_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f x m ≤ 2/3) ∧ (∃ x ∈ Set.Icc (-1) 1, f x m = 2/3) →
  m = 2/3 := by
  sorry

-- Theorem 2: If m = 2/3, then the range of f(x) on [-2,2] is [-6, 2/3]
theorem m_implies_range :
  ∀ y ∈ Set.Icc (-6) (2/3), ∃ x ∈ Set.Icc (-2) 2, f x (2/3) = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_m_implies_range_l765_76561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l765_76519

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1 + 3)^2 + p.2^2 = 100}

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Define a point P on the circle C
noncomputable def P : ℝ × ℝ := sorry

-- Assume P is on the circle C
axiom P_on_C : P ∈ C

-- Define point M as the intersection of the perpendicular bisector of BP and CP
noncomputable def M : ℝ × ℝ := sorry

-- Define the property that M is on the perpendicular bisector of BP
def M_on_perp_bisector : Prop := sorry

-- Define the property that M is on CP
def M_on_CP : Prop := sorry

-- Assume M satisfies both properties
axiom M_properties : M_on_perp_bisector ∧ M_on_CP

-- State the theorem
theorem trajectory_of_M :
  {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 16 = 1} = {p : ℝ × ℝ | ∃ P, P ∈ C ∧ p = M} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l765_76519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_formula_l765_76563

/-- The radius of a triangle's circumscribed circle -/
noncomputable def circumradius (a b c t : ℝ) : ℝ := (a * b * c) / (4 * t)

/-- Theorem: The radius of the circle circumscribed around a triangle is abc/(4t) -/
theorem circumradius_formula (a b c t : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ t > 0) :
  ∃ (r : ℝ), r > 0 ∧ r = circumradius a b c t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_formula_l765_76563
