import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_approximation_l34_3487

/-- Given a square with side length s and a rectangle with length 20 and breadth b,
    prove that if the perimeter of the square equals the perimeter of the rectangle,
    and the circumference of a semicircle with diameter s is 26.70,
    then the breadth of the rectangle is approximately 0.78. -/
theorem rectangle_breadth_approximation (s b : ℝ) : 
  26.70 = Real.pi * s / 2 + s →
  4 * s = 2 * (20 + b) →
  ∃ ε > 0, |b - 0.78| < ε := by
  sorry

#check rectangle_breadth_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_approximation_l34_3487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_problem_l34_3471

-- Define the initial conditions
noncomputable def initial_volume : ℝ := 40
noncomputable def initial_alcohol_percentage : ℝ := 5
noncomputable def added_alcohol : ℝ := 4.5
noncomputable def added_water : ℝ := 5.5

-- Define the function to calculate the final alcohol percentage
noncomputable def final_alcohol_percentage (initial_volume : ℝ) (initial_alcohol_percentage : ℝ) 
                             (added_alcohol : ℝ) (added_water : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * (initial_alcohol_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  (final_alcohol / final_volume) * 100

-- Theorem statement
theorem alcohol_solution_problem :
  final_alcohol_percentage initial_volume initial_alcohol_percentage added_alcohol added_water = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_problem_l34_3471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_pairs_l34_3411

theorem harmonic_mean_pairs : 
  let harmonic_mean (x y : ℕ) := (2 * x * y) / (x + y)
  let valid_pair (p : ℕ × ℕ) := p.1 < p.2 ∧ harmonic_mean p.1 p.2 = (12 : ℝ) ^ 10
  (Finset.filter valid_pair (Finset.range 10000 ×ˢ Finset.range 10000)).card = 220 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_pairs_l34_3411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_and_sin_beta_l34_3447

-- Define the function f
noncomputable def f (x : Real) : Real := 2 * Real.sqrt 3 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

-- Define the theorem
theorem function_range_and_sin_beta 
  (h1 : ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ∈ Set.Icc (-(Real.sqrt 3)) 2)
  (α β : Real)
  (h2 : f (α / 2 - Real.pi / 6) = 8 / 5)
  (h3 : Real.cos (α + β) = -12 / 13)
  (h4 : 0 < α ∧ α < Real.pi / 2)
  (h5 : 0 < β ∧ β < Real.pi / 2) :
  Real.sin β = 63 / 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_and_sin_beta_l34_3447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l34_3468

theorem tan_sum_given_tan_cot_sum (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 15)
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 20) : 
  Real.tan (x + y) = 60 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l34_3468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_estimation_l34_3469

/-- Represents that a value is an estimate of another value -/
def IsEstimateOf (estimate actual : ℝ) : Prop := 
  ∃ ε > 0, |estimate - actual| < ε

theorem integral_estimation 
  (φ : ℝ → ℝ) 
  (a b c : ℝ) 
  (n n₁ : ℕ) 
  (h1 : a < b) 
  (h2 : c > 0) 
  (h3 : ∀ x, a ≤ x ∧ x ≤ b → 0 ≤ φ x ∧ φ x ≤ c) :
  ∃ l₃_star : ℝ, l₃_star = (b - a) * c * ((n₁ : ℝ) / n) ∧ 
  IsEstimateOf l₃_star (∫ x in a..b, φ x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_estimation_l34_3469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_GCD_l34_3489

-- Define the square ABCD
def square_ABCD : Set (ℝ × ℝ) := sorry

-- Define the area of the square
def area_ABCD : ℝ := 144

-- Define point E on BC
def point_E : ℝ × ℝ := sorry

-- Define the ratio BE:EC
def BE_EC_ratio : ℚ := 3/1

-- Define point F as midpoint of AE
def point_F : ℝ × ℝ := sorry

-- Define point G as midpoint of DE
def point_G : ℝ × ℝ := sorry

-- Define quadrilateral BEGF
def quad_BEGF : Set (ℝ × ℝ) := sorry

-- Define the area of quadrilateral BEGF
def area_BEGF : ℝ := 25

-- Define triangle GCD
def triangle_GCD : Set (ℝ × ℝ) := sorry

-- Function to calculate area of a set in ℝ²
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_triangle_GCD : area triangle_GCD = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_GCD_l34_3489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_max_value_l34_3491

theorem cos_max_value :
  ∃ (M : ℝ), M = 3/2 ∧ ∀ x : ℝ, Real.cos (2*x + 2*Real.sin x) ≤ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_max_value_l34_3491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_increasing_function_inequality_l34_3441

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem odd_increasing_function_inequality 
  (h_odd : ∀ x, x ∈ Set.Ioo (-1) 1 → f (-x) = -f x)
  (h_increasing : ∀ x y, x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → x < y → f x < f y)
  (h_domain : ∀ x, f x ≠ 0 → x ∈ Set.Ioo (-1) 1)
  (m : ℝ)
  (h_inequality : f (1 - m) + f (m^2 - 1) < 0) :
  0 < m ∧ m < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_increasing_function_inequality_l34_3441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_properties_l34_3429

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -4 ∧ x ≤ -1 then -x - 1
  else if x > -1 ∧ x ≤ 1 then x^2 - 3
  else if x > 1 ∧ x ≤ 4 then x - 1
  else 0  -- Undefined for other x values

-- Define h(x) = g(|x|) + 1
noncomputable def h (x : ℝ) : ℝ := g (abs x) + 1

-- Theorem statement
theorem h_properties (x : ℝ) :
  ((-1 ≤ x ∧ x ≤ 1) → h x = x^2 - 2) ∧
  ((1 < x ∧ x ≤ 4) ∨ (-4 ≤ x ∧ x < -1) → h x = x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_properties_l34_3429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_value_theorem_l34_3444

/-- Represents the value of the coat in rubles -/
noncomputable def coat_value : ℚ := 24/5

/-- Represents the total annual compensation in rubles -/
noncomputable def annual_compensation (coat_value : ℚ) : ℚ := 12 + coat_value

/-- Represents the pro-rata pay for 7 months of work -/
noncomputable def pro_rata_pay (coat_value : ℚ) : ℚ := (7 / 12) * annual_compensation coat_value

/-- Represents the actual pay given to the worker -/
noncomputable def actual_pay (coat_value : ℚ) : ℚ := 5 + coat_value

theorem coat_value_theorem : 
  pro_rata_pay coat_value = actual_pay coat_value := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_value_theorem_l34_3444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_multiple_of_four_l34_3423

/-- Represents a fair eight-sided die -/
def EightSidedDie : Finset ℕ := Finset.range 8

/-- The probability of an event occurring when rolling two eight-sided dice -/
def probability (event : ℕ → ℕ → Bool) : ℚ :=
  (Finset.filter (fun p => event p.1 p.2) (EightSidedDie.product EightSidedDie)).card /
    (EightSidedDie.card * EightSidedDie.card : ℚ)

/-- Predicate for when the product of two numbers is a multiple of 4 -/
def productMultipleOf4 (x y : ℕ) : Bool := (x * y) % 4 = 0

theorem probability_product_multiple_of_four :
  probability productMultipleOf4 = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_multiple_of_four_l34_3423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_definition_l34_3459

/-- Definition of a solution to an equation -/
def IsSolution (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

/-- Predicate to represent that a value is called the solution of an equation -/
def is_called_solution_of (x : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, IsSolution f y → y = x

/-- The value that makes both sides of an equation equal is called its solution -/
theorem solution_definition (f : ℝ → ℝ) (x : ℝ) :
  IsSolution f x ↔ is_called_solution_of x f :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_definition_l34_3459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l34_3478

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set N
def N : Set ℝ := {x | x > 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioc 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l34_3478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l34_3401

/-- An ellipse with focus at (1, 0) and point (-1, √2/2) on it -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : a^2 - b^2 = 1
  h4 : 1 / a^2 + 1 / (2 * b^2) = 1

/-- Point T defined by the vector OT -/
noncomputable def T (e : Ellipse) : ℝ × ℝ := (2, 0)

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2 / 2 + y^2 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The maximum area of triangle PQT -/
noncomputable def max_area_PQT (e : Ellipse) : ℝ := Real.sqrt 2 / 2

/-- The collinearity of vectors PQ and QT -/
def PQ_QT_collinear (e : Ellipse) (P Q : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, Q.1 - P.1 = k * (Q.1 - (T e).1) ∧ Q.2 - P.2 = k * (Q.2 - (T e).2)

/-- Area of a triangle given three points -/
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

/-- The main theorem combining all parts -/
theorem ellipse_properties (e : Ellipse) :
  ellipse_equation e ∧
  (∀ P Q : ℝ × ℝ, P.1^2 / e.a^2 + P.2^2 / e.b^2 = 1 →
                   Q.1^2 / e.a^2 + Q.2^2 / e.b^2 = 1 →
                   area_triangle P Q (T e) ≤ max_area_PQT e) ∧
  (∀ P Q : ℝ × ℝ, P.1^2 / e.a^2 + P.2^2 / e.b^2 = 1 →
                   Q.1^2 / e.a^2 + Q.2^2 / e.b^2 = 1 →
                   PQ_QT_collinear e P Q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l34_3401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sin_squared_over_one_plus_cos_cubed_l34_3422

theorem limit_sin_squared_over_one_plus_cos_cubed (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - π| ∧ |x - π| < δ →
    |Real.sin x ^ 2 / (1 + Real.cos x ^ 3) - 2/3| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sin_squared_over_one_plus_cos_cubed_l34_3422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_from_medians_l34_3486

/-- Given a triangle with medians of lengths a, b, and c, 
    the perimeter of the triangle is 2(a + b + c) -/
theorem triangle_perimeter_from_medians 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let perimeter := 2 * (a + b + c)
  perimeter = 26 ↔ a = 3 ∧ b = 4 ∧ c = 6 := by
  sorry

#check triangle_perimeter_from_medians

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_from_medians_l34_3486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_meeting_time_l34_3464

def horse_lap_time (k : ℕ) : ℕ := 2 * k

def meets_at_start (t : ℕ) (k : ℕ) : Bool :=
  t % (horse_lap_time k) = 0

def count_horses_at_start (t : ℕ) : ℕ :=
  (List.range 8).filter (meets_at_start t) |>.length

theorem least_meeting_time :
  ∃ (T : ℕ),
    T > 0 ∧
    count_horses_at_start T ≥ 4 ∧
    ∀ (t : ℕ), t > 0 ∧ t < T → count_horses_at_start t < 4 := by
  sorry

#eval count_horses_at_start 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_meeting_time_l34_3464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l34_3495

/-- A geometric sequence with a common ratio not equal to 1 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ ∀ n, a (n + 1) = q * a n

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a 1) * (1 - q^n) / (1 - q)

/-- Arithmetic sequence property for three terms -/
def arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_properties
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_arith : arithmetic_sequence (a 5) (a 3) (a 4)) :
  (q = -2) ∧
  (∀ k : ℕ, k > 0 →
    arithmetic_sequence
      (geometric_sum a q (k + 2))
      (geometric_sum a q k)
      (geometric_sum a q (k + 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l34_3495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dining_table_original_price_l34_3436

noncomputable def original_price (discount_percentage : ℝ) (sale_price : ℝ) : ℝ :=
  sale_price / (1 - discount_percentage / 100)

theorem dining_table_original_price :
  original_price 10 450 = 500 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dining_table_original_price_l34_3436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_indivisible_alternating_sum_l34_3466

theorem subset_with_indivisible_alternating_sum 
  (S : Finset ℤ) 
  (h_card : S.card = 10000) 
  (h_not_div : ∀ x ∈ S, ¬ 47 ∣ x) :
  ∃ Y : Finset ℤ, Y ⊆ S ∧ Y.card = 2015 ∧ 
    ∀ a b c d e, a ∈ Y → b ∈ Y → c ∈ Y → d ∈ Y → e ∈ Y → ¬ 47 ∣ (a - b + c - d + e) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_indivisible_alternating_sum_l34_3466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medium_popcorn_can_be_rational_l34_3435

/-- Represents a popcorn container with its size, weight, and price -/
structure Popcorn where
  size : String
  weight : Nat
  price : Nat

/-- Represents a customer's preferences -/
structure CustomerPreference where
  wantsPopcorn : Bool
  wantsDrink : Bool

/-- Rational choice function -/
def rationalChoice (popcorns : List Popcorn) (budget : Nat) (pref : CustomerPreference) : Option Popcorn :=
  sorry

/-- Theorem stating that choosing medium popcorn can be rational -/
theorem medium_popcorn_can_be_rational :
  ∃ (budget : Nat) (pref : CustomerPreference),
    let popcorns : List Popcorn := [
      { size := "small", weight := 50, price := 200 },
      { size := "medium", weight := 70, price := 400 },
      { size := "large", weight := 130, price := 500 }
    ]
    budget = 500 ∧
    pref.wantsPopcorn ∧ pref.wantsDrink ∧
    (rationalChoice popcorns budget pref).isSome ∧
    (rationalChoice popcorns budget pref).map (λ p => p.size) = some "medium" :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_medium_popcorn_can_be_rational_l34_3435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_tangent_l34_3412

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

noncomputable def g (x : ℝ) : ℝ := x / (x + 1)

theorem one_common_tangent (h : ∀ x, x > -1) :
  ∃! (a : ℝ), a > -1 ∧
    (∃ (m : ℝ), m * (deriv f a) = deriv g a ∧
      f a + m * (deriv f a) * (-a) = g a + (deriv g a) * (-a)) ∧
    (∀ x, f a + (deriv f a) * (x - a) = g a + (deriv g a) * (x - a) → x = a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_tangent_l34_3412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l34_3426

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  q : ℝ
  h_q_gt_one : q > 1
  h_a_1 : a 1 = 2
  h_arithmetic : a 1 - (a 2 - (a 3 - 8)) = 0

/-- The sequence b_n -/
noncomputable def b (n : ℕ) : ℝ := 2 * n - 9

/-- The sum S_n -/
noncomputable def S (n : ℕ) : ℝ := n^2 - 8 * n

/-- The sequence c_n -/
noncomputable def c (as : ArithmeticSequence) (n : ℕ) : ℝ := b n / as.a n

/-- The theorem to be proved -/
theorem min_m_value (as : ArithmeticSequence) : 
  (∃ m : ℝ, ∀ n : ℕ, c as n ≤ m) → 
  (∀ m : ℝ, (∀ n : ℕ, c as n ≤ m) → m ≥ 1/162) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l34_3426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_after_six_passes_l34_3473

/-- Represents the probability of the first person having the ball after n passes -/
def prob_first_person (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (2^(n-1) - (prob_first_person (n-1)).num.toNat) / 2^n

/-- The main theorem stating the probability after 6 passes -/
theorem prob_after_six_passes :
  prob_first_person 6 = 11 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_after_six_passes_l34_3473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l34_3449

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

-- State the theorem
theorem omega_value (ω : ℝ) : 
  ω > 0 ∧ 
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π/3 → f ω x ≤ f ω y) ∧
  (∀ x, 0 ≤ x ∧ x ≤ π/3 → f ω x ≤ Real.sqrt 2) ∧
  (∃ x, 0 ≤ x ∧ x ≤ π/3 ∧ f ω x = Real.sqrt 2) →
  ω = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l34_3449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l34_3437

noncomputable def P : ℝ × ℝ := (1, 2)
noncomputable def Q : ℝ × ℝ := (3, 6)
noncomputable def R : ℝ × ℝ := (6, 3)
noncomputable def S : ℝ × ℝ := (8, 1)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def perimeter : ℝ :=
  distance P Q + distance Q R + distance R S + distance S P

theorem quadrilateral_perimeter :
  ∃ (c d : ℤ), perimeter = c * Real.sqrt 2 + d * Real.sqrt 10 ∧ c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l34_3437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_and_period_range_l34_3432

noncomputable def ω : ℝ := Real.pi / 2
noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos (ω * x), Real.sin (ω * x))
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos (ω * x), Real.sqrt 3 * Real.cos (ω * x))
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem monotonicity_intervals_and_period_range (h1 : ω > 0) (h2 : 0 < ω) (h3 : ω < 2)
  (h4 : ∀ x, f x = f (Real.pi / 3 - x)) :
  (∀ k : ℤ, ∀ x ∈ Set.Icc (-2 * Real.pi / 3 + 2 * k * Real.pi) (Real.pi / 3 + 2 * k * Real.pi),
    (MonotoneOn f (Set.Icc (-2 * Real.pi / 3 + 2 * k * Real.pi) (Real.pi / 3 + 2 * k * Real.pi)))) ∧
  (∃ k : ℤ, k ≠ 0 ∧ ∀ x, f (x + k * Real.pi) = f x) ∧
  Set.range f = Set.Icc (-1/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_and_period_range_l34_3432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l34_3433

/-- A pyramid with triangular faces -/
structure Pyramid :=
  (vertices : Fin 4 → ℝ × ℝ × ℝ)

/-- The length of an edge in the pyramid -/
noncomputable def edge_length (p : Pyramid) (i j : Fin 4) : ℝ :=
  let (xi, yi, zi) := p.vertices i
  let (xj, yj, zj) := p.vertices j
  Real.sqrt ((xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2)

/-- The area of a triangular face of the pyramid -/
noncomputable def face_area (p : Pyramid) (i j k : Fin 4) : ℝ :=
  let a := edge_length p i j
  let b := edge_length p j k
  let c := edge_length p k i
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The total surface area of the pyramid -/
noncomputable def total_surface_area (p : Pyramid) : ℝ :=
  face_area p 0 1 2 + face_area p 0 1 3 + face_area p 0 2 3 + face_area p 1 2 3

/-- Main theorem -/
theorem pyramid_surface_area (p : Pyramid) :
  (∀ i j, i ≠ j → edge_length p i j = 25 ∨ edge_length p i j = 60) →
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i →
    edge_length p i j ≠ edge_length p j k ∨
    edge_length p j k ≠ edge_length p k i ∨
    edge_length p k i ≠ edge_length p i j) →
  total_surface_area p = 3600 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l34_3433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l34_3498

/-- The area of a rhombus with given side length and one diagonal length. -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) : 
  side = 20 → diagonal1 = 16 → 
  ∃ (area : ℝ), area = 64 * Real.sqrt 21 ∧ 
  area = (diagonal1 * Real.sqrt (4 * side^2 - diagonal1^2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l34_3498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expressions_evaluation_l34_3431

noncomputable section

theorem complex_expressions_evaluation :
  -- Expression 1
  let expr1 := (2 * 7/9)^(1/2 : ℝ) - (2 * Real.sqrt 3 - Real.pi)^(0 : ℝ) - (2 * 10/27)^(-(2/3) : ℝ) + 0.25^(-(3/2) : ℝ)
  -- Expression 2
  let expr2 := (Real.log 6.25) / (Real.log 2.5) + Real.log 5 / Real.log 10 + Real.log (Real.sqrt (Real.exp 1)) + 2^(-1 + Real.log 3 / Real.log 2) + (Real.log 2 / Real.log 10)^2 + (Real.log 5 / Real.log 10) * (Real.log 2 / Real.log 10)
  
  expr1 = 389/48 ∧ expr2 = 5 := by
    sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expressions_evaluation_l34_3431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_in_triangle_l34_3434

theorem cosine_value_in_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  a * Real.cos B = (3 * c - b) * Real.cos A →
  Real.cos A = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_in_triangle_l34_3434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relations_l34_3481

def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

theorem set_relations (a : ℝ) :
  (A ⊂ B a ↔ a < -2) ∧
  (A ∩ B a ≠ ∅ ↔ a < 3) ∧
  (A ∩ B a = ∅ ↔ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relations_l34_3481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l34_3490

theorem solution_difference : 
  ∃ s₁ s₂ : ℝ, 
    (s₁^2 - 5*s₁ - 22) / (s₁ + 4) = 3*s₁ + 8 ∧
    (s₂^2 - 5*s₂ - 22) / (s₂ + 4) = 3*s₂ + 8 ∧
    abs (s₁ - s₂) = (3 : ℝ) / 2 ∧ 
    ∀ t₁ t₂ : ℝ, 
      (t₁^2 - 5*t₁ - 22) / (t₁ + 4) = 3*t₁ + 8 →
      (t₂^2 - 5*t₂ - 22) / (t₂ + 4) = 3*t₂ + 8 →
      abs (t₁ - t₂) ≤ abs (s₁ - s₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l34_3490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_intersection_constraint_l34_3463

/-- Given a set of 8 elements, this theorem states that the maximum number of 4-element subsets
    that can be chosen, such that the intersection of any three of these subsets contains at most
    one element, is 8. -/
theorem max_subsets_with_intersection_constraint (S : Finset ℕ) (hS : S.card = 8) :
  (∃ F : Finset (Finset ℕ), 
    (∀ A ∈ F, A ⊆ S ∧ A.card = 4) ∧
    (∀ A B C, A ∈ F → B ∈ F → C ∈ F → (A ∩ B ∩ C).card ≤ 1) ∧
    F.card = 8) ∧
  (∀ G : Finset (Finset ℕ),
    (∀ A ∈ G, A ⊆ S ∧ A.card = 4) →
    (∀ A B C, A ∈ G → B ∈ G → C ∈ G → (A ∩ B ∩ C).card ≤ 1) →
    G.card ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_intersection_constraint_l34_3463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_distance_l34_3454

/-- Represents the distance between two villages connected by a mountain road --/
def distance : ℝ := sorry

/-- Represents the speed of the bus when traveling uphill --/
def uphill_speed : ℝ := 15

/-- Represents the speed of the bus when traveling downhill --/
def downhill_speed : ℝ := 30

/-- Represents the total time taken for the round trip --/
def round_trip_time : ℝ := 4

/-- Theorem stating that the distance between the villages is 40 km --/
theorem village_distance : distance = 40 := by
  sorry

/-- Lemma stating that the time taken for the round trip is the sum of uphill and downhill times --/
lemma round_trip_time_sum : round_trip_time = distance / uphill_speed + distance / downhill_speed := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_distance_l34_3454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l34_3428

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  eccentricity : ℝ
  intersection_distance : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_eccentricity : eccentricity = 2 * Real.sqrt 2 / 3
  h_intersection : intersection_distance = 3 * Real.sqrt 3

/-- The equation and perimeter properties of the ellipse -/
theorem ellipse_properties (e : Ellipse) :
  e.a = 3 ∧ e.b = 1 ∧ 
  (∀ x y : ℝ, x^2 / 9 + y^2 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  (let f₁ := Real.sqrt (e.a^2 - e.b^2)
   let f₂ := -f₁
   ∃ g h : ℝ × ℝ, 
     g.1^2 / e.a^2 + g.2^2 / e.b^2 = 1 ∧
     h.1^2 / e.a^2 + h.2^2 / e.b^2 = 1 ∧
     g.2 - f₁ = g.1 - f₁ ∧
     h.2 - f₁ = h.1 - f₁ ∧
     Real.sqrt ((g.1 - f₂)^2 + g.2^2) +
     Real.sqrt ((h.1 - f₂)^2 + h.2^2) +
     Real.sqrt ((g.1 - h.1)^2 + (g.2 - h.2)^2) = 12) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l34_3428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l34_3407

/-- The ellipse defined by x² + y²/3 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 + y^2/3 = 1

/-- The line y = x - 1 -/
def line (x y : ℝ) : Prop := y = x - 1

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem intersection_distance :
  ∃ x1 y1 x2 y2 : ℝ,
    ellipse x1 y1 ∧ line x1 y1 ∧
    ellipse x2 y2 ∧ line x2 y2 ∧
    distance x1 y1 x2 y2 = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l34_3407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_theorem_l34_3477

/-- The perimeter of an isosceles triangle with side lengths 6 and 10 -/
def isosceles_triangle_perimeter : Set ℝ :=
  {22, 26}

/-- IsIsosceles predicate for a triangle -/
def IsIsosceles (sides : Set ℝ) : Prop :=
  ∃ x y, sides = {x, y, x} ∨ sides = {x, y, y}

/-- Theorem: The perimeter of an isosceles triangle with side lengths 6 and 10 is either 22 or 26 -/
theorem isosceles_triangle_perimeter_theorem (a b : ℝ) 
  (h1 : a = 6 ∨ a = 10) 
  (h2 : b = 6 ∨ b = 10) 
  (h3 : a ≠ b) 
  (h4 : IsIsosceles {a, b, a}) : 
  a + b + a ∈ isosceles_triangle_perimeter := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_theorem_l34_3477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_from_centroid_line_l34_3413

-- Define the points
variable (A B C D M : EuclideanSpace ℝ (Fin 2))

-- Define the quadrilateral
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the diagonals intersection
def diagonals_intersect_at (A B C D M : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the centroid of a triangle
noncomputable def centroid (P Q R : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

-- Define a point lying on a line connecting two points
def lies_on_line (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define parallel lines
def parallel (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Theorem statement
theorem trapezoid_from_centroid_line (A B C D M : EuclideanSpace ℝ (Fin 2)) :
  is_convex_quadrilateral A B C D →
  diagonals_intersect_at A B C D M →
  lies_on_line M (centroid A B M) (centroid C D M) →
  parallel A B C D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_from_centroid_line_l34_3413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_min_l34_3475

theorem sin_cos_sixth_power_min (x : ℝ) : Real.sin x ^ 6 + Real.cos x ^ 6 ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_min_l34_3475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_only_increasing_on_reals_l34_3485

-- Define the functions
noncomputable def f₁ (x : ℝ) := Real.exp (-x)
def f₂ (x : ℝ) := x^3
noncomputable def f₃ (x : ℝ) := Real.log x
def f₄ (x : ℝ) := |x|

-- State the theorem
theorem cubic_only_increasing_on_reals :
  (∀ x y : ℝ, x < y → f₂ x < f₂ y) ∧
  (¬∀ x y : ℝ, x < y → f₁ x < f₁ y) ∧
  (¬∀ x y : ℝ, x < y → f₃ x < f₃ y) ∧
  (¬∀ x y : ℝ, x < y → f₄ x < f₄ y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_only_increasing_on_reals_l34_3485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_distance_100_l34_3483

/-- Represents the beetle's movement pattern -/
def segmentLength (n : ℕ) : ℕ := (n - 1) / 4 + 1

/-- The total distance covered after n segments -/
def totalDistance (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => segmentLength (i + 1))

/-- Theorem stating the total distance after 100 segments -/
theorem beetle_distance_100 : totalDistance 100 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_distance_100_l34_3483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_selection_count_l34_3461

theorem committee_selection_count : 
  let total_members : Nat := 30
  let committee_size : Nat := 5
  let president_included : Nat := 1
  let remaining_members : Nat := total_members - president_included
  let positions_to_fill : Nat := committee_size - president_included
  Nat.choose remaining_members positions_to_fill = 23741 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_selection_count_l34_3461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_inside_circle_square_outside_l34_3496

-- Define the circle centered at the origin
def our_circle (x y : ℝ) : Prop := x^2 + y^2 ≤ 1

-- Define the rhombus
def our_rhombus (x y : ℝ) : Prop := 2 * |x| + 2 * |y| ≤ 2

-- Define the square
def our_square (x y : ℝ) : Prop := max (|x|^2) (|y|^2) ≤ 1

-- State the theorem
theorem rhombus_inside_circle_square_outside : 
  ∀ x y : ℝ, 2 * |x| + 2 * |y| ≥ x^2 + y^2 ∧ x^2 + y^2 ≥ max (|x|^2) (|y|^2) →
  (our_rhombus x y → our_circle x y) ∧ (our_circle x y → our_square x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_inside_circle_square_outside_l34_3496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_150_value_l34_3438

def b : ℕ → ℕ
  | 0 => 3  -- Define b(0) to handle the base case
  | n + 1 => b n + 4 * n

theorem b_150_value : b 150 = 44703 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_150_value_l34_3438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_methods_correct_l34_3497

/-- The number of different hiring methods for 3 units to hire from 4 university graduates -/
def hiringMethods : ℕ := 60

/-- The number of units hiring employees -/
def numUnits : ℕ := 3

/-- The number of university graduates -/
def numGraduates : ℕ := 4

/-- Represents whether a unit hired a graduate -/
def hired (unit : Fin numUnits) (graduate : Fin numGraduates) : Prop := sorry

/-- Each unit must hire at least one person -/
axiom at_least_one_hire (unit : Fin numUnits) : ∃ (graduate : Fin numGraduates), hired unit graduate

/-- The number of different hiring methods is correct -/
theorem hiring_methods_correct : hiringMethods = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_methods_correct_l34_3497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_number_with_ten_unique_ending_divisors_l34_3451

theorem no_number_with_ten_unique_ending_divisors :
  ¬ ∃ (n : ℕ), 
    (∃ (divs : Finset ℕ), 
      (divs.card = 10) ∧ 
      (∀ d, d ∈ divs → d ∣ n) ∧
      (∀ d, d ∈ divs → ∃ i : Fin 10, d % 10 = i.val) ∧
      (∀ d1 d2, d1 ∈ divs → d2 ∈ divs → d1 ≠ d2 → d1 % 10 ≠ d2 % 10)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_number_with_ten_unique_ending_divisors_l34_3451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l34_3448

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.cos x ^ 2)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_properties :
  ∀ x : ℝ,
  (f x = 2 * Real.sin (2 * x + π / 6) + 1) ∧
  (∀ y : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ y : ℝ, 
    k * π - π / 3 ≤ y ∧ y ≤ k * π + π / 6 → 
    ∀ z : ℝ, y < z → f y ≤ f z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l34_3448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_one_third_l34_3417

-- Define the two functions
def f (x : ℝ) : ℝ := x^(1/2)
def g (x : ℝ) : ℝ := x^2

-- Define the area of the enclosed region
noncomputable def enclosed_area : ℝ := ∫ x in Set.Icc 0 1, (f x - g x)

-- Theorem statement
theorem enclosed_area_is_one_third : enclosed_area = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_one_third_l34_3417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_at_height_8_l34_3400

/-- The volume of a regular tetrahedron with side length a and height h -/
noncomputable def tetrahedronVolume (a : ℝ) (h : ℝ) : ℝ := (1/3) * a^2 * h

/-- The relationship between the height of the tetrahedron and the side length of its base,
    given that it's inscribed in a sphere with diameter 12 -/
def heightSideLengthRelation (h : ℝ) (a : ℝ) : Prop :=
  h^2 - 12*h + (1/2)*a^2 = 0

/-- The volume of the tetrahedron as a function of its height -/
noncomputable def volumeFunction (h : ℝ) : ℝ := (2/3) * (12*h^2 - h^3)

/-- The theorem stating that the volume of the inscribed regular tetrahedron
    is maximized when its height is 8 -/
theorem max_volume_at_height_8 :
  ∃ (h : ℝ), h > 0 ∧ h < 12 ∧
  (∀ (h' : ℝ), h' > 0 → h' < 12 → volumeFunction h' ≤ volumeFunction h) ∧
  h = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_at_height_8_l34_3400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_exists_l34_3440

-- Define the triangle type
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angle function
noncomputable def angle (T : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Define the altitude function
noncomputable def altitude (T : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Define the median function
noncomputable def median (T : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_construction_exists (α m k : ℝ) :
  ∃ (T : Triangle),
    angle T T.A = α ∧
    altitude T T.B = m ∧
    median T T.A = k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_exists_l34_3440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l34_3420

theorem calculate_expression : (12 : ℚ) * (1/3 + 1/4 + 1/6)⁻¹ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l34_3420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_point_l34_3465

/-- Given a hyperbola and specific points, prove that x₀ = √2/2 -/
theorem hyperbola_intersection_point (x₀ : ℝ) (t : ℝ) : 
  x₀ > 0 →
  x₀ ≠ Real.sqrt 3 →
  ∃ (A B : ℝ × ℝ),
    (λ (x y : ℝ) => x^2 - y^2/3 = 1) A.1 A.2 ∧
    (λ (x y : ℝ) => x^2 - y^2/3 = 1) B.1 B.2 ∧
    (∃ (k : ℝ), A = (k * x₀, k * 4)) ∧
    (∃ (k : ℝ), B = (k * x₀, k * 4)) ∧
    (x₀, 0) - (0, 4) = (t * (A.1 - x₀), t * A.2) ∧
    (x₀, 0) - (0, 4) = ((2-t) * (B.1 - x₀), (2-t) * B.2) →
  x₀ = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_point_l34_3465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_characteristic_theorem_l34_3450

/-- Represents a dress with a size and a characteristic -/
structure Dress where
  size : ℕ
  characteristic : String

/-- Represents the scenario of daughters choosing dresses -/
structure DressScenario where
  dresses : Finset Dress
  daughters : Finset ℕ
  darkRoom : Bool

/-- The probability of all daughters not choosing their own dress -/
noncomputable def probabilityAllWrong (scenario : DressScenario) : ℝ := sorry

/-- Checks if dresses are indistinguishable in the dark except for size -/
def dressesIndistinguishableInDark (scenario : DressScenario) : Prop := sorry

theorem dress_characteristic_theorem (scenario : DressScenario) :
  scenario.dresses.card = 3 ∧
  scenario.daughters.card = 3 ∧
  scenario.darkRoom = true ∧
  (∀ d₁ d₂, d₁ ∈ scenario.dresses → d₂ ∈ scenario.dresses → d₁ ≠ d₂ → d₁.size ≠ d₂.size) ∧
  probabilityAllWrong scenario = 0.65 →
  dressesIndistinguishableInDark scenario := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_characteristic_theorem_l34_3450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_students_same_school_probability_l34_3453

/-- The number of students -/
def num_students : ℕ := 3

/-- The number of schools -/
def num_schools : ℕ := 4

/-- The probability of exactly two students choosing the same school -/
def probability_two_same_school : ℚ := 9 / 16

/-- Theorem stating that the probability of exactly two students choosing the same school is 9/16 -/
theorem two_students_same_school_probability :
  (Fintype.card {f : Fin num_students → Fin num_schools | 
    ∃ (i j : Fin num_students), i ≠ j ∧ f i = f j ∧ 
    ∀ (k : Fin num_students), k ≠ i → k ≠ j → f k ≠ f i}) /
  (Fintype.card (Fin num_students → Fin num_schools)) = probability_two_same_school :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_students_same_school_probability_l34_3453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_range_l34_3419

-- Define the functions f and g
noncomputable def f (x m : ℝ) : ℝ := x^2 + m
noncomputable def g (x : ℝ) : ℝ := -Real.log (1/x) - 3*x

-- Define the domain
def domain : Set ℝ := Set.Icc (1/2) 2

-- Define the theorem
theorem symmetric_points_range (m : ℝ) :
  (∃ x ∈ domain, f x m = g x) ↔ m ∈ Set.Icc (2 - Real.log 2) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_range_l34_3419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_higher_probability_l34_3462

noncomputable section

/-- The probability that a ball lands in bin k -/
def prob_in_bin (k : ℕ+) : ℝ := (1 : ℝ) / 3^(k : ℝ)

/-- The probability that both balls land in the same bin k -/
def prob_same_bin (k : ℕ+) : ℝ := prob_in_bin k * prob_in_bin k

/-- The sum of probabilities of both balls landing in the same bin for all bins -/
noncomputable def prob_same_bin_total : ℝ := ∑' k, prob_same_bin k

/-- The probability that the red ball is in a higher-numbered bin than the blue ball -/
noncomputable def prob_red_higher : ℝ := (1 - prob_same_bin_total) / 2

theorem red_ball_higher_probability :
  prob_red_higher = 7 / 16 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_higher_probability_l34_3462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_side_product_bound_l34_3458

/-- A convex quadrilateral with sides a, b, c, d and diagonals e, f -/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  convex : True  -- We assume convexity without formally defining it

/-- The maximum of the sides and diagonals is 1 -/
def max_side_is_one (q : ConvexQuadrilateral) : Prop :=
  max q.a (max q.b (max q.c (max q.d (max q.e q.f)))) = 1

/-- The theorem stating the upper bound of the product of sides -/
theorem quadrilateral_side_product_bound (q : ConvexQuadrilateral) 
    (h : max_side_is_one q) : q.a * q.b * q.c * q.d ≤ 2 - Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_side_product_bound_l34_3458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l34_3467

noncomputable def f (x : ℝ) : ℝ := 
  |Real.sin x| / Real.sin x + Real.cos x / |Real.cos x| + |Real.tan x| / Real.tan x

theorem f_range (x : ℝ) (h1 : Real.sin x ≠ 0) (h2 : Real.cos x ≠ 0) : 
  f x = -1 ∨ f x = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l34_3467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_volume_l34_3452

/-- The volume of a cone formed by rotating a right triangle around one of its legs -/
noncomputable def cone_volume (leg1 : ℝ) (leg2 : ℝ) : ℝ :=
  (1/3) * Real.pi * leg2^2 * leg1

/-- Theorem: The volume of the solid formed by rotating a right triangle
    with legs 3 and 4 around the leg of length 3 is equal to 16π -/
theorem right_triangle_rotation_volume :
  cone_volume 3 4 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_volume_l34_3452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarters_count_correct_l34_3406

/-- Calculates the number of quarters given the other coin counts and the final amount after fee --/
def calculate_quarters (dimes nickels pennies : ℕ) (final_amount : ℚ) : ℕ :=
  let dime_value : ℚ := 1/10
  let nickel_value : ℚ := 1/20
  let penny_value : ℚ := 1/100
  let quarter_value : ℚ := 1/4
  let fee_percentage : ℚ := 1/10
  let total_before_fee : ℚ := final_amount / (1 - fee_percentage)
  let other_coins_value : ℚ := dimes * dime_value + nickels * nickel_value + pennies * penny_value
  let quarters_value : ℚ := total_before_fee - other_coins_value
  (quarters_value / quarter_value).floor.toNat

theorem quarters_count_correct (dimes nickels pennies : ℕ) (final_amount : ℚ) :
  dimes = 85 → nickels = 20 → pennies = 150 → final_amount = 27 →
  calculate_quarters dimes nickels pennies final_amount = 76 := by
  sorry

#eval calculate_quarters 85 20 150 27

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarters_count_correct_l34_3406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l34_3414

/-- The volume of a cone formed from a sector of a circle, divided by π -/
noncomputable def cone_volume_over_pi (sector_angle : ℝ) (circle_radius : ℝ) : ℝ :=
  let base_radius := circle_radius * (sector_angle / (2 * Real.pi))
  let cone_height := Real.sqrt (circle_radius ^ 2 - base_radius ^ 2)
  (1 / 3) * base_radius ^ 2 * cone_height

/-- Theorem stating that the volume of a cone formed from a 270-degree sector
    of a circle with radius 20, when divided by π, is equal to 1125 √7 -/
theorem cone_volume_theorem :
  cone_volume_over_pi (3 * Real.pi / 2) 20 = 1125 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l34_3414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_cubic_equation_l34_3492

theorem unique_solution_cubic_equation (x y z : ℕ) : 
  Nat.Prime y → 
  ¬(y ∣ z) → 
  ¬(3 ∣ z) → 
  x^3 - y^3 = z^2 → 
  x = 8 ∧ y = 7 ∧ z = 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_cubic_equation_l34_3492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_f_l34_3439

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x + 1

/-- The symmetric center of a periodic function f is a point (x₀, y₀) such that
    f(x₀ + x) = f(x₀ - x) for all x, and the function is symmetric about the vertical line x = x₀. -/
def symmetric_center (f : ℝ → ℝ) : ℝ × ℝ := sorry

theorem symmetric_center_of_f :
  ∃ (k : ℤ), symmetric_center f = (k * Real.pi + Real.pi / 3, 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_f_l34_3439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_sequences_is_80_l34_3470

/-- Represents a sequence of 10 integers satisfying the given conditions -/
def ValidSequence : Type := 
  {a : Fin 10 → ℤ // 
    (a 9 = 3 * a 0) ∧ 
    (a 1 + a 7 = 2 * a 4) ∧ 
    (∀ i : Fin 9, a (i + 1) = a i + 1 ∨ a (i + 1) = a i + 2)}

/-- The number of valid sequences -/
noncomputable def numValidSequences : ℕ := sorry

/-- The main theorem stating that the number of valid sequences is 80 -/
theorem num_valid_sequences_is_80 : numValidSequences = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_sequences_is_80_l34_3470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_prices_correct_l34_3402

noncomputable def original_price_running : ℝ := 100
noncomputable def original_price_formal : ℝ := 150
noncomputable def original_price_casual : ℝ := 75

def store_wide_discount : ℝ := 0.2
def additional_discount : ℝ := 0.5

noncomputable def price_after_store_discount (price : ℝ) : ℝ := price * (1 - store_wide_discount)

noncomputable def cheapest_after_store_discount : ℝ := min (price_after_store_discount original_price_running) 
                                          (min (price_after_store_discount original_price_formal) 
                                               (price_after_store_discount original_price_casual))

noncomputable def final_price (price : ℝ) : ℝ :=
  if price_after_store_discount price = cheapest_after_store_discount
  then price_after_store_discount price * (1 - additional_discount)
  else price_after_store_discount price

theorem shoe_prices_correct : 
  final_price original_price_running = 80 ∧ 
  final_price original_price_formal = 120 ∧ 
  final_price original_price_casual = 30 ∧
  final_price original_price_running + final_price original_price_formal + final_price original_price_casual = 230 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_prices_correct_l34_3402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compounded_ratio_result_l34_3404

def ratio1 : ℚ := 2 / 3
def ratio2 : ℚ := 6 / 11
def ratio3 : ℚ := 11 / 2

def compounded_ratio (r1 r2 r3 : ℚ) : ℚ :=
  (r1.num * r2.num * r3.num) / (r1.den * r2.den * r3.den)

theorem compounded_ratio_result :
  compounded_ratio ratio1 ratio2 ratio3 = 2 / 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compounded_ratio_result_l34_3404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l34_3424

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ↔ a ∈ Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l34_3424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_given_lcm_l34_3456

theorem greatest_x_given_lcm (x : ℕ) : 
  Nat.lcm x (Nat.lcm 12 15) = 180 → x ≤ 180 ∧ ∃ (y : ℕ), y > 180 → Nat.lcm y (Nat.lcm 12 15) > 180 := by
  sorry

#check greatest_x_given_lcm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_given_lcm_l34_3456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_defective_rate_is_0_007_l34_3403

/-- The defective rate of worker x -/
noncomputable def defective_rate_x : ℝ := 0.005

/-- The defective rate of worker y -/
noncomputable def defective_rate_y : ℝ := 0.008

/-- The proportion of products checked by worker y -/
noncomputable def proportion_y : ℝ := 2/3

/-- The proportion of products checked by worker x -/
noncomputable def proportion_x : ℝ := 1 - proportion_y

/-- The total defective rate of all products -/
noncomputable def total_defective_rate : ℝ := defective_rate_x * proportion_x + defective_rate_y * proportion_y

theorem total_defective_rate_is_0_007 : total_defective_rate = 0.007 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_defective_rate_is_0_007_l34_3403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solar_panel_area_scientific_notation_l34_3416

noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

def is_valid_scientific_notation (a : ℝ) : Prop := 1 ≤ |a| ∧ |a| < 10

theorem solar_panel_area_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_valid_scientific_notation a ∧ scientific_notation a n = 30000 ∧ a = 3 ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solar_panel_area_scientific_notation_l34_3416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_path_count_l34_3479

/-- Represents a point on the square lattice -/
structure Point where
  x : ℤ
  y : ℤ

/-- The number of paths between two points -/
def num_paths (start finish : Point) : ℕ := sorry

theorem bug_path_count (A B C : Point) 
  (h1 : num_paths A B = 6) 
  (h2 : num_paths B C = 6) : 
  num_paths A C = 36 := by sorry

#check bug_path_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_path_count_l34_3479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l34_3425

theorem power_equation_solution (k : ℝ) : (1/2 : ℝ)^23 * (1/81 : ℝ)^k = (1/18 : ℝ)^23 → k = 5.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l34_3425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l34_3484

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, 1 < x ∧ x < 2 → |x - 2| < 1) ∧
  (∃ x : ℝ, |x - 2| < 1 ∧ ¬(1 < x ∧ x < 2)) :=
by
  constructor
  · intro x h
    sorry
  · use 2.5
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l34_3484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l34_3476

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem problem_solution (f g : ℝ → ℝ) 
  (h_odd : odd_function f) (h_even : even_function g)
  (h1 : f (-1) + g 1 = 2) (h2 : f 1 + g (-1) = 4) : 
  g 1 = 3 := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l34_3476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_division_trapezoid_segment_formula_l34_3408

-- Define the trapezoid and its properties
variable (a c : ℝ) -- bases of the trapezoid
variable (p q : ℝ) -- ratio of areas
variable (d : ℝ) -- segment that divides the trapezoid

-- State the theorem
theorem trapezoid_area_division (h_pos : a > 0 ∧ c > 0 ∧ p > 0 ∧ q > 0) :
  d = Real.sqrt ((q * a^2 + p * c^2) / (p + q)) ↔
  ∃ (u v : ℝ), u > 0 ∧ v > 0 ∧
  (a + d) * u / ((c + d) * v) = p / q ∧
  u / v = (a - d) / (d - c) :=
sorry

-- Main proof
theorem trapezoid_segment_formula (h_pos : a > 0 ∧ c > 0 ∧ p > 0 ∧ q > 0) :
  d = Real.sqrt ((q * a^2 + p * c^2) / (p + q)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_division_trapezoid_segment_formula_l34_3408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l34_3405

noncomputable section

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  -3 * x^2 + 4 * y^2 - 12 * x - 24 * y + 6 = 0

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define what it means for a point to be a focus of the hyperbola
def is_focus (f : Point) (a : ℝ) : Prop :=
  ∃ (p1 p2 : Point),
    hyperbola_eq p1.1 p1.2 ∧
    hyperbola_eq p2.1 p2.2 ∧
    |distance f p1 - distance f p2| = 2 * a

-- Theorem statement
theorem hyperbola_focus :
  ∃ (a : ℝ), is_focus (-2, 3 + Real.sqrt 14) a := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l34_3405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_transformation_path_length_l34_3480

/-- A square with side length 4 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ := 4)
  (is_square : A.1 = B.1 ∧ A.2 + side_length = B.2 ∧
               B.1 + side_length = C.1 ∧ B.2 = C.2 ∧
               C.1 = D.1 ∧ C.2 - side_length = D.2 ∧
               D.1 - side_length = A.1 ∧ D.2 = A.2)

/-- Rotation of 180 degrees clockwise about vertex A -/
def rotate180 (s : Square) : ℝ × ℝ :=
  s.A

/-- Reflection across diagonal BD -/
def reflectBD (s : Square) (_ : ℝ × ℝ) : ℝ × ℝ :=
  s.D

/-- The total path length traveled by vertex C -/
noncomputable def pathLength (s : Square) : ℝ :=
  let rotated := rotate180 s
  let reflected := reflectBD s rotated
  Real.sqrt ((s.C.1 - rotated.1)^2 + (s.C.2 - rotated.2)^2) +
  Real.sqrt ((rotated.1 - reflected.1)^2 + (rotated.2 - reflected.2)^2)

theorem square_transformation_path_length (s : Square) :
  pathLength s = 4 * Real.sqrt 2 + 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_transformation_path_length_l34_3480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l34_3494

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.cos x ^ 2 + 1 / 2

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≥ f x) ∧
  f x = -1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l34_3494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AP_is_sqrt_2_l34_3415

/-- Square ABCD with side length 2 -/
def square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- Circle ω with center (1, 0) and radius 1 -/
def omega : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 1}

/-- Point A of the square -/
def A : ℝ × ℝ := (0, 2)

/-- Point M where circle intersects CD -/
def M : ℝ × ℝ := (2, 0)

/-- Point P where AM intersects circle ω (different from M) -/
def P : ℝ × ℝ := (1, 1)

/-- The length of AP is √2 -/
theorem length_AP_is_sqrt_2 : 
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AP_is_sqrt_2_l34_3415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_intersection_points_l34_3474

-- Define the equations
def equation1 (x y : ℝ) : Prop := (x + 2*y - 7)*(3*x - 4*y + 8) = 0
def equation2 (x y : ℝ) : Prop := (x - 2*y - 1)*(4*x + 5*y - 20) = 0

-- Define a solution as a pair of real numbers satisfying both equations
def is_solution (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

-- Define the set of all solutions
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | is_solution p}

-- Theorem statement
theorem num_intersection_points :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 3 ∧ ∀ p, p ∈ solution_set ↔ p ∈ s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_intersection_points_l34_3474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l34_3430

-- Define the ellipse
noncomputable def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_eccentricity_range 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0)
  (A B : ℝ × ℝ)
  (h_A : A ∈ Ellipse a b)
  (h_B : B ∈ Ellipse a b)
  (h_symmetric : A.1 = -B.1 ∧ A.2 = -B.2)
  (F : ℝ × ℝ)
  (h_F : F.1 = Real.sqrt (a^2 - b^2) ∧ F.2 = 0)  -- Right focus
  (h_perpendicular : (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 0)  -- AF ⊥ BF
  (α : ℝ)
  (h_angle : α = Real.arccos ((B.1 - A.1) / (2 * Real.sqrt (a^2 - b^2))))  -- ∠ABF = α
  (h_α_range : π/4 < α ∧ α < π/3) :
  Real.sqrt 2 / 2 < eccentricity a b ∧ eccentricity a b < Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l34_3430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_insurance_cost_l34_3446

def monthly_plan_price : ℚ := 500
def hourly_wage : ℚ := 25
def weekly_hours : ℚ := 30
def weeks_per_month : ℚ := 4

def monthly_income : ℚ := hourly_wage * weekly_hours * weeks_per_month
def annual_income : ℚ := monthly_income * 12

noncomputable def government_subsidy_rate (income : ℚ) : ℚ :=
  if income < 10000 then 9/10
  else if income > 50000 then 1/5
  else 1/2

noncomputable def monthly_insurance_cost : ℚ :=
  monthly_plan_price * (1 - government_subsidy_rate annual_income)

theorem annual_insurance_cost : monthly_insurance_cost * 12 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_insurance_cost_l34_3446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_last_digits_all_digits_possible_l34_3460

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def last_digit (n : ℕ) : ℕ := n % 10

def possible_last_digits : Finset ℕ := Finset.range 10

theorem count_possible_last_digits : Finset.card possible_last_digits = 10 := by
  rw [possible_last_digits]
  exact Finset.card_range 10

theorem all_digits_possible : ∀ d : ℕ, d < 10 → ∃ n : ℕ, is_divisible_by_3 n ∧ last_digit n = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_last_digits_all_digits_possible_l34_3460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l34_3457

def ellipse_E (x y : ℝ) : Prop := x^2/2 + y^2 = 1

def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 3

def chord_length (l : Set (ℝ × ℝ)) (O : Set (ℝ × ℝ)) : ℝ := 3

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  O : ℝ × ℝ
  A_on_ellipse : ellipse_E A.1 A.2
  B_on_ellipse : ellipse_E B.1 B.2
  O_is_origin : O = (0, 0)

def triangle_area (t : Triangle) : ℝ := sorry

theorem max_triangle_area (l : Set (ℝ × ℝ)) (O : Set (ℝ × ℝ)) :
  ∀ (t : Triangle),
  chord_length l O = 3 →
  triangle_area t ≤ Real.sqrt 30 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l34_3457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_with_two_differences_l34_3409

/-- Represents the attributes of a block -/
structure BlockAttributes where
  material : Fin 2
  size : Fin 3
  color : Fin 4
  shape : Fin 4
deriving Fintype, DecidableEq

/-- The total number of blocks in the set -/
def total_blocks : Nat := 96

/-- The reference block (plastic medium red circle) -/
def reference_block : BlockAttributes :=
  { material := 0, size := 1, color := 2, shape := 0 }

/-- Counts the number of differences between two BlockAttributes -/
def count_differences (b1 b2 : BlockAttributes) : Nat :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0)

/-- The set of all possible blocks -/
def all_blocks : Finset BlockAttributes := Finset.univ

/-- The theorem to be proved -/
theorem blocks_with_two_differences :
  (all_blocks.filter (λ b => count_differences b reference_block = 2)).card = 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_with_two_differences_l34_3409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l34_3421

/-- The line defined by the parametric equations x = 1 - t, y = 8 - 2t -/
def line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 1 - t ∧ p.2 = 8 - 2*t}

/-- The curve defined by the parametric equations x = 1 + √5 * cos(θ), y = -2 + √5 * sin(θ) -/
def curve : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ θ : ℝ, p.1 = 1 + Real.sqrt 5 * Real.cos θ ∧ p.2 = -2 + Real.sqrt 5 * Real.sin θ}

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating the range of distances between points on the line and curve -/
theorem distance_range :
  (∀ p ∈ line, ∀ q ∈ curve, distance p q ≥ Real.sqrt 5) ∧
  (∀ r : ℝ, r > Real.sqrt 5 → ∃ p ∈ line, ∃ q ∈ curve, distance p q = r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l34_3421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coins_on_board_l34_3427

/-- Represents a placement of coins on a 10x10 board --/
def CoinPlacement := Fin 10 → Fin 10 → Bool

/-- Checks if four coins form a rectangle --/
def formsRectangle (placement : CoinPlacement) (r1 r2 c1 c2 : Fin 10) : Prop :=
  placement r1 c1 ∧ placement r1 c2 ∧ placement r2 c1 ∧ placement r2 c2

/-- Checks if a placement is valid (no rectangles formed) --/
def isValidPlacement (placement : CoinPlacement) : Prop :=
  ∀ r1 r2 c1 c2, r1 ≠ r2 → c1 ≠ c2 → ¬formsRectangle placement r1 r2 c1 c2

/-- Counts the number of coins in a placement --/
def coinCount (placement : CoinPlacement) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 10)) fun r =>
    Finset.sum (Finset.univ : Finset (Fin 10)) fun c =>
      if placement r c then 1 else 0)

/-- Theorem: The maximum number of coins that can be placed on a 10x10 board
    with no rectangles formed is 34 --/
theorem max_coins_on_board :
  (∃ placement : CoinPlacement, isValidPlacement placement ∧ coinCount placement = 34) ∧
  (∀ placement : CoinPlacement, isValidPlacement placement → coinCount placement ≤ 34) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coins_on_board_l34_3427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_45_eq_sqrt2_div_2_l34_3482

noncomputable def unit_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

noncomputable def angle_45 : ℝ := Real.pi / 4

noncomputable def point_Q (c : Set (ℝ × ℝ)) (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sin α)

def point_E (Q : ℝ × ℝ) : ℝ × ℝ :=
  (Q.1, 0)

theorem sin_45_eq_sqrt2_div_2 :
  let Q := point_Q unit_circle angle_45
  let E := point_E Q
  (Q ∈ unit_circle) →
  (E.1 = Q.1 ∧ E.2 = 0) →
  (Q.1^2 + Q.2^2 = 1) →
  (Q.1 = Q.2) →
  Real.sin angle_45 = Real.sqrt 2 / 2 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_45_eq_sqrt2_div_2_l34_3482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l34_3493

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (oplus 1 x) + (oplus 2 x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 18 ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 3 → f x ≤ M) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-2) 3 ∧ f x = M) := by
  sorry

-- You can add more lemmas or theorems here if needed for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l34_3493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_to_triangle_area_ratio_l34_3472

/-- The ratio of a regular pentagon's area to the area of one of its five central triangles -/
theorem pentagon_to_triangle_area_ratio (s : ℝ) (hs : s > 0) :
  let R := s / (2 * Real.sin (36 * π / 180))
  let q := (5 / 2) * R * s * Real.sin (72 * π / 180)
  let t := (1 / 2) * s * (R * Real.sin (36 * π / 180))
  q / t = 5 * Real.sin (72 * π / 180) / Real.sin (36 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_to_triangle_area_ratio_l34_3472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_quadrilateral_l34_3455

/-- The volume of the solid obtained by rotating quadrilateral MNTS around directrix l -/
theorem volume_of_rotated_quadrilateral (p : ℝ) (M N : ℝ × ℝ) (S T : ℝ × ℝ) : 
  p > 0 →
  (∀ x y, 2*x - 2*y - 1 = 0 ↔ (x, y) = M ∨ (x, y) = N) →
  (∀ x y, y^2 = 2*p*x ↔ (x, y) = M ∨ (x, y) = N) →
  ‖M - N‖ = 4 →
  S.1 = -1/2 ∧ T.1 = -1/2 →
  S.2 = M.2 ∧ T.2 = N.2 →
  9 * Real.sqrt 2 * π = (11*π + 5*π/2) * 2*Real.sqrt 2 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_quadrilateral_l34_3455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l34_3488

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (1/2 * x + Real.pi/6) - 1

theorem f_extrema :
  (∃ (y_max : ℝ), ∀ (x : ℝ), f x ≤ y_max ∧ y_max = 2) ∧
  (∃ (y_min : ℝ), ∀ (x : ℝ), f x ≥ y_min ∧ y_min = -4) ∧
  (∀ (x : ℝ), f x = 2 ↔ ∃ (k : ℤ), x = 4 * k * Real.pi + 2 * Real.pi / 3) ∧
  (∀ (x : ℝ), f x = -4 ↔ ∃ (k : ℤ), x = 4 * k * Real.pi - 4 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l34_3488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_drivers_sufficient_and_alexey_returns_at_2130_l34_3445

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hMinutesValid : minutes < 60

/-- Represents a driver -/
inductive Driver
| A | B | C | D

/-- Represents a trip -/
structure Trip where
  driver : Driver
  departure : Time
  arrival : Time

def one_way_duration : ℕ := 160 -- 2 hours and 40 minutes in minutes

def round_trip_duration : ℕ := 2 * one_way_duration

def rest_duration : ℕ := 60 -- 1 hour in minutes

/-- Function to add minutes to a given time -/
def add_minutes (t : Time) (m : ℕ) : Time :=
  sorry

/-- Function to check if a driver is available at a given time -/
def is_driver_available (d : Driver) (t : Time) (trips : List Trip) : Bool :=
  sorry

theorem four_drivers_sufficient_and_alexey_returns_at_2130 
  (trips : List Trip)
  (hA_return : ∃ t, t ∈ trips ∧ t.driver = Driver.A ∧ t.arrival = ⟨12, 40, sorry⟩)
  (hB_C_en_route : ¬ is_driver_available Driver.B ⟨12, 40, sorry⟩ trips ∧ 
                   ¬ is_driver_available Driver.C ⟨12, 40, sorry⟩ trips)
  (hD_depart : ∃ t, t ∈ trips ∧ t.driver = Driver.D ∧ t.departure = ⟨13, 5, sorry⟩)
  (hB_return_depart : ∃ t1 t2, t1 ∈ trips ∧ t2 ∈ trips ∧
    t1.driver = Driver.B ∧ t1.arrival = ⟨16, 0, sorry⟩ ∧
    t2.driver = Driver.B ∧ t2.departure = ⟨17, 30, sorry⟩)
  (hA_fifth_trip : ∃ t, t ∈ trips ∧ t.driver = Driver.A ∧ t.departure = ⟨16, 10, sorry⟩)
  : (∀ t : Time, ∃ d : Driver, is_driver_available d t trips) ∧
    (∃ t, t ∈ trips ∧ t.driver = Driver.A ∧ t.arrival = ⟨21, 30, sorry⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_drivers_sufficient_and_alexey_returns_at_2130_l34_3445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_range_l34_3442

theorem inequality_implies_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 3 → x^2 - a*x - 3 ≤ 0) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_range_l34_3442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l34_3443

/-- Sequence of partial sums -/
def S : ℕ+ → ℝ := sorry

/-- Sequence of products of partial sums -/
def T : ℕ+ → ℝ := sorry

/-- Original sequence -/
def a : ℕ+ → ℝ := sorry

theorem sequence_properties (h : ∀ n : ℕ+, S n + T n = S n * T n) 
  (h_nonzero : ∀ n : ℕ+, S n ≠ 0) :
  (a 1 = 2) ∧ 
  (∃ d : ℝ, ∀ n : ℕ+, T (n + 1) = T n + d) ∧
  (∀ n : ℕ+, S n = (n + 1 : ℝ) / n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l34_3443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_distance_proof_l34_3418

/-- The length of the track in meters -/
def track_length : ℝ := 300

/-- The time in minutes when A first overtakes B -/
def first_overtake : ℝ := 12

/-- The time in minutes when A second overtakes B -/
def second_overtake : ℝ := 32

/-- The initial distance between A and B in meters -/
def initial_distance : ℝ := 180

theorem initial_distance_proof (v_A v_B : ℝ) (h1 : v_A > v_B) 
  (h2 : v_A * first_overtake = v_B * first_overtake + track_length)
  (h3 : v_A * second_overtake = v_B * second_overtake + 2 * track_length) :
  initial_distance = 180 := by
  sorry

#check initial_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_distance_proof_l34_3418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_from_p_l34_3499

/-- Triangle PQR with specific properties -/
structure TrianglePQR where
  /-- Side PQ is no more than 9 -/
  pq : ℝ
  pq_bound : pq ≤ 9
  /-- Side PR is no more than 12 -/
  pr : ℝ
  pr_bound : pr ≤ 12
  /-- Area of the triangle is at least 54 -/
  area : ℝ
  area_bound : area ≥ 54

/-- Predicate to represent that a length is a median of a triangle -/
def IsMedian (a b m : ℝ) : Prop :=
  ∃ c : ℝ, m^2 = (a^2 + b^2) / 4 + c^2 / 4

/-- The median drawn from vertex P in the given triangle PQR is 7.5 -/
theorem median_from_p (t : TrianglePQR) : ∃ m : ℝ, m = 7.5 ∧ IsMedian t.pq t.pr m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_from_p_l34_3499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l34_3410

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3*θ) = -117/125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l34_3410
