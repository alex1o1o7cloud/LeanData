import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_when_t1_is_1_a_equals_plus_minus_2_l355_35500

-- Define the points and vectors
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (4, 6)

def OA : ℝ × ℝ := A
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define M as a function of t₁ and t₂
def M (t₁ t₂ : ℝ) : ℝ × ℝ :=
  (t₁ * OA.1 + t₂ * AB.1, t₁ * OA.2 + t₂ * AB.2)

-- Define collinearity
def collinear (P Q R : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), (R.1 - P.1, R.2 - P.2) = (t * (Q.1 - P.1), t * (Q.2 - P.2))

-- Define perpendicularity
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define area of a triangle
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (Q.2 - P.2) * (R.1 - P.1))

-- Theorem 1
theorem collinear_when_t1_is_1 (t₂ : ℝ) :
  collinear A B (M 1 t₂) := by
  sorry

-- Theorem 2
theorem a_equals_plus_minus_2 (a : ℝ) :
  (∃ t₂, perpendicular (M (a^2) t₂) AB ∧ triangle_area A B (M (a^2) t₂) = 12) →
  a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_when_t1_is_1_a_equals_plus_minus_2_l355_35500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_l355_35519

/-- Represents the pricing and sales data for shirts -/
structure ShirtData where
  initial_price : ℚ
  cost_price : ℚ
  initial_sales : ℚ
  profit_target : ℚ
  price_reduction_step : ℚ
  sales_increase_step : ℚ

/-- Calculates the monthly profit based on the new price -/
noncomputable def monthly_profit (data : ShirtData) (new_price : ℚ) : ℚ :=
  let price_reduction := data.initial_price - new_price
  let sales_increase := (price_reduction / data.price_reduction_step) * data.sales_increase_step
  (new_price - data.cost_price) * (data.initial_sales + sales_increase)

/-- Theorem stating that a price of 68 yuan results in the target monthly profit -/
theorem optimal_price (data : ShirtData) 
    (h1 : data.initial_price = 80)
    (h2 : data.cost_price = 50)
    (h3 : data.initial_sales = 200)
    (h4 : data.profit_target = 7920)
    (h5 : data.price_reduction_step = 2)
    (h6 : data.sales_increase_step = 40) :
  monthly_profit data 68 = data.profit_target := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_l355_35519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_factor_change_l355_35598

noncomputable def q (w v f z : ℝ) : ℝ := 5 * w / (4 * v * f * z^2)

theorem q_factor_change (w v f z : ℝ) (hv : v ≠ 0) (hf : f ≠ 0) (hz : z ≠ 0) :
  q (4 * w) v (2 * f) (3 * z) = (2 / 9) * q w v f z :=
by
  unfold q
  field_simp [hv, hf, hz]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_factor_change_l355_35598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pairs_theorem_l355_35536

/-- A triangle represented by its three side lengths -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- A pair of triangles -/
structure TrianglePair where
  t1 : Triangle
  t2 : Triangle

def is_valid_pair (p : TrianglePair) : Prop :=
  p.t1.a + p.t1.b + p.t1.c + p.t2.a + p.t2.b + p.t2.c = 20

def valid_pairs : List TrianglePair :=
  [ TrianglePair.mk ⟨5, 5, 5, by simp⟩ ⟨1, 2, 2, by simp⟩,
    TrianglePair.mk ⟨5, 5, 3, by simp⟩ ⟨3, 2, 2, by simp⟩,
    TrianglePair.mk ⟨5, 5, 2, by simp⟩ ⟨3, 3, 2, by simp⟩,
    TrianglePair.mk ⟨5, 5, 1, by simp⟩ ⟨4, 3, 2, by simp⟩,
    TrianglePair.mk ⟨5, 5, 1, by simp⟩ ⟨4, 4, 1, by simp⟩ ]

theorem triangle_pairs_theorem :
  ∀ p : TrianglePair, is_valid_pair p ↔ p ∈ valid_pairs :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pairs_theorem_l355_35536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_f_solution_set_l355_35597

noncomputable def f (x : Real) := Real.sqrt (10 * Real.sin x - 2) - Real.sqrt (5 * Real.cos x - 3)

def is_acute (θ : Real) := 0 < θ ∧ θ < Real.pi / 2

theorem f_equals_one (θ : Real) (h1 : is_acute θ) (h2 : Real.tan (2 * θ) = 24 / 7) : 
  f θ = 1 := by sorry

theorem f_solution_set (x : Real) : 
  f x = 1 ↔ ∃ k : Int, x = Real.arctan (3 / 4) + k * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_f_solution_set_l355_35597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_problem_l355_35535

/-- If 5x = 6y, xy ≠ 0, and there exists a constant k such that 
    (kx) / (1/5 * y) ≈ 1.9999999999999998, then x/y = 6/5 -/
theorem ratio_problem (x y : ℝ) (k : ℝ) 
    (h1 : 5 * x = 6 * y) 
    (h2 : x * y ≠ 0) 
    (h3 : (k * x) / ((1/5) * y) = 1.9999999999999998) : 
  x / y = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_problem_l355_35535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l355_35592

/-- The maximum distance from a point on the ellipse x²/2 + y² = 1 to the line y = x + 1 -/
theorem max_distance_ellipse_to_line :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let distance (p : ℝ × ℝ) := |p.2 - p.1 - 1| / Real.sqrt 2
  ∀ p ∈ ellipse, distance p ≤ (Real.sqrt 6 + Real.sqrt 2) / 2 ∧
  ∃ p ∈ ellipse, distance p = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l355_35592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_54_l355_35567

/-- The sum of all positive integer solutions x ≤ 30 to the congruence 15(5x-3) ≡ 45 (mod 12) -/
def sum_of_solutions : ℕ :=
  (Finset.filter (fun x => x > 0 ∧ x ≤ 30 ∧ (15 * (5 * x - 3)) % 12 = 45 % 12) (Finset.range 31)).sum id

/-- Theorem stating that the sum of solutions is 54 -/
theorem sum_of_solutions_is_54 : sum_of_solutions = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_54_l355_35567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_expression_l355_35599

theorem smallest_value_of_expression (a b c : ℤ) (ω : ℂ) : 
  (∃ n : ℤ, a = n ∧ b = n + 1 ∧ c = n + 2) →  -- consecutive integers
  ω^4 = 1 →
  ω ≠ 1 →
  Real.sqrt 2 ≤ Complex.abs (a + b*ω + c*ω^3) ∧ 
  ∃ a' b' c' : ℤ, (∃ n' : ℤ, a' = n' ∧ b' = n' + 1 ∧ c' = n' + 2) ∧ 
                  Complex.abs (a' + b'*ω + c'*ω^3) = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_expression_l355_35599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l355_35569

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem odd_function_inequality (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_mono : monotone_increasing_on f (Set.Ioi 0))
  (h_f1 : f 1 = 0) :
  {x : ℝ | f x > 0} = Set.Ioo (-1) 0 ∪ Set.Ioi 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l355_35569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l355_35587

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y = 2*x ∧ x = c) →
  c^2 = a^2 * ((Real.sqrt 2 + 1)^2 - 1) ∧ b^2 = a^2 * (Real.sqrt 2 + 1)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l355_35587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_result_l355_35552

/-- P is a function that takes a real number x and returns 3 times the square root of x -/
noncomputable def P (x : ℝ) : ℝ := 3 * Real.sqrt x

/-- Q is a function that takes a real number x and returns x cubed -/
def Q (x : ℝ) : ℝ := x^3

/-- Theorem stating that the composition of P and Q functions applied to 2 results in 846√2 -/
theorem composition_result :
  P (Q (P (Q (P (Q 2))))) = 846 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_result_l355_35552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_even_radius_of_circle_eight_is_greatest_even_radius_l355_35564

theorem greatest_even_radius_of_circle (r : ℕ) : Even r ∧ π * r^2 < 100 * π → r ≤ 8 := by
  sorry

theorem eight_is_greatest_even_radius : ∃ (r : ℕ), Even r ∧ π * r^2 < 100 * π ∧ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_even_radius_of_circle_eight_is_greatest_even_radius_l355_35564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l355_35566

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x

theorem sin_shift_equivalence :
  ∀ x : ℝ, f x = g (x + Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l355_35566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pe_score_analysis_l355_35596

/-- Represents the scores and their frequencies in the class -/
structure ScoreDistribution where
  score_30 : Nat
  score_25 : Nat
  score_20 : Nat
  score_15 : Nat

/-- Calculates the total number of students -/
def total_students (sd : ScoreDistribution) : Nat :=
  sd.score_30 + sd.score_25 + sd.score_20 + sd.score_15

/-- Calculates the total score -/
def total_score (sd : ScoreDistribution) : Nat :=
  30 * sd.score_30 + 25 * sd.score_25 + 20 * sd.score_20 + 15 * sd.score_15

/-- Calculates the average score -/
def average_score (sd : ScoreDistribution) : ℚ :=
  (total_score sd : ℚ) / (total_students sd : ℚ)

/-- Finds the median of the scores -/
noncomputable def median_score (sd : ScoreDistribution) : ℚ := sorry

/-- Finds the mode of the scores -/
def mode_score (sd : ScoreDistribution) : Nat := sorry

theorem pe_score_analysis 
  (sd : ScoreDistribution)
  (h1 : total_students sd = 10)
  (h2 : sd.score_30 = 2)
  (h3 : sd.score_15 = 1)
  (h4 : average_score sd = 23)
  : median_score sd = 22.5 ∧ mode_score sd = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pe_score_analysis_l355_35596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l355_35543

theorem rectangle_area_increase (P : ℝ) : 
  (1 + P / 100)^2 = 1.44 → P = 20 := by
  intro h
  -- Proof steps would go here
  sorry

#eval (1 + 20 / 100)^2  -- This should evaluate to approximately 1.44

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l355_35543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l355_35514

theorem solve_exponential_equation :
  ∃ x : ℝ, (4 : ℝ) ^ (2 * x) = Real.sqrt 64 → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l355_35514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_equation_solution_l355_35584

theorem unique_digit_equation_solution :
  ∀ (A B C D E F G H I : ℕ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧
     E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧
     F ≠ G ∧ F ≠ H ∧ F ≠ I ∧
     G ≠ H ∧ G ≠ I ∧
     H ≠ I) →
    (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧
    (1 ≤ D ∧ D ≤ 9) ∧ (1 ≤ E ∧ E ≤ 9) ∧ (1 ≤ F ∧ F ≤ 9) ∧
    (1 ≤ G ∧ G ≤ 9) ∧ (1 ≤ H ∧ H ≤ 9) ∧ (1 ≤ I ∧ I ≤ 9) →
    (((100 * A + 10 * B + C) - (10 * D + E) : ℚ) + 
     ((F^2 : ℚ) / (10 * G + H : ℚ)) - (2010 : ℚ) / I) = 1219 / 100 →
    A = 3 ∧ B = 4 ∧ C = 1 ∧ D = 7 ∧ E = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_equation_solution_l355_35584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_trick_success_l355_35501

/-- Represents a point on a circle of 12 equally spaced points -/
def CirclePoint := Fin 12

/-- Calculates the arc distance between two points on the circle -/
def arcDistance (p q : CirclePoint) : Nat :=
  min (Int.natAbs (p.val - q.val)) (12 - Int.natAbs (p.val - q.val))

/-- Represents a quadrilateral on the circle -/
structure Quadrilateral where
  v1 : CirclePoint
  v2 : CirclePoint
  v3 : CirclePoint
  v4 : CirclePoint

/-- Checks if the quadrilateral covers the arc between two points -/
def covers (q : Quadrilateral) (p1 p2 : CirclePoint) : Prop :=
  ∃ (v w : CirclePoint), v ∈ ({q.v1, q.v2, q.v3, q.v4} : Set CirclePoint) ∧ 
                         w ∈ ({q.v1, q.v2, q.v3, q.v4} : Set CirclePoint) ∧ 
                         arcDistance v w = arcDistance p1 p2

theorem coin_trick_success :
  ∃ (q : Quadrilateral), ∀ (p1 p2 : CirclePoint), covers q p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_trick_success_l355_35501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_l355_35517

theorem car_speed_problem (x : ℝ) : 
  x > 0 →  -- Speed must be positive
  (x + 45) / 2 = 65 →  -- Average speed equation
  x = 85 := by
  intro h1 h2
  -- Proof steps would go here
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_l355_35517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_equals_one_l355_35530

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 2) / (x - 1)

def is_symmetric (a : ℝ) : Prop :=
  ∀ x y, f a x = y ↔ f a (2 - x) = 2 - y

theorem symmetry_implies_a_equals_one (a : ℝ) (h : is_symmetric a) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_equals_one_l355_35530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_minus_beta_l355_35504

theorem cos_two_alpha_minus_beta 
  (α β : ℝ) 
  (h1 : Real.cos α = Real.sqrt 5 / 5)
  (h2 : Real.sin (α - β) = Real.sqrt 10 / 10)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : 0 < β ∧ β < Real.pi / 2) : 
  Real.cos (2 * α - β) = Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_minus_beta_l355_35504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_divisible_by_n_l355_35561

def u : ℕ → ℕ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 2
  | 3 => 24
  | n + 4 => (6 * (u (n + 3))^2 * u (n + 1) - 8 * u (n + 3) * (u (n + 2))^2) / (u (n + 2) * u (n + 1))

theorem u_divisible_by_n (n : ℕ) : n > 0 → n ∣ u n := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_divisible_by_n_l355_35561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l355_35554

/-- The range of m given the conditions -/
theorem range_of_m : ∃ (S : Set ℝ), S = Set.Iic (-7) ∪ Set.Ici 1 ∧
  ∀ m : ℝ, (∀ x : ℝ, x^2 + 3*x - 4 < 0 → (x - m)*(x - m - 3) > 0) ∧
           (∃ x : ℝ, (x - m)*(x - m - 3) > 0 ∧ x^2 + 3*x - 4 ≥ 0) →
  m ∈ S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l355_35554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_slope_l355_35559

-- Define the slope of a line passing through two points
noncomputable def my_slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

-- Define the slope of a perpendicular line
noncomputable def perpendicular_slope (m : ℝ) : ℝ := -1 / m

-- Theorem statement
theorem perpendicular_line_slope :
  let m := my_slope 3 (-4) (-6) 2
  perpendicular_slope m = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_slope_l355_35559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_eight_l355_35523

theorem cube_root_of_negative_eight : ((-8 : ℝ) ^ (1/3 : ℝ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_eight_l355_35523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_non_constant_coeffs_l355_35594

-- Define the binomial expression
noncomputable def binomial_expr (x : ℝ) : ℝ := (2 / Real.sqrt x - x) ^ 9

-- Define the sum of all coefficients
noncomputable def sum_all_coeffs : ℝ := binomial_expr 1

-- Define the constant term
def constant_term : ℝ := -5376

-- Theorem statement
theorem sum_non_constant_coeffs :
  sum_all_coeffs - constant_term = 5377 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_non_constant_coeffs_l355_35594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l355_35585

open Real

theorem trigonometric_identities 
  (α β γ : ℝ) 
  (h1 : Real.cos α + Real.cos β + Real.cos γ = 0) 
  (h2 : Real.sin α + Real.sin β + Real.sin γ = 0) : 
  (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = (3/2) ∧ 
  Real.cos (3*α) + Real.cos (3*β) + Real.cos (3*γ) = 3 * Real.cos (α + β + γ) ∧
  Real.sin (3*α) + Real.sin (3*β) + Real.sin (3*γ) = 3 * Real.sin (α + β + γ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l355_35585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisible_by_sum_count_l355_35558

theorem factorial_divisible_by_sum_count : 
  (Finset.filter (fun n : ℕ => n ≤ 30 ∧ (n.factorial % (n * (n + 1) / 2) = 0)) (Finset.range 31)).card = 20 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisible_by_sum_count_l355_35558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_correlation_l355_35572

-- Define the chi-square statistic
noncomputable def chi_square (a b c d : ℕ) : ℝ :=
  let n : ℝ := (a + b + c + d : ℝ)
  n * ((a : ℝ) * (d : ℝ) - (b : ℝ) * (c : ℝ))^2 / (((a + c : ℕ) : ℝ) * ((b + d : ℕ) : ℝ) * ((a + b : ℕ) : ℝ) * ((c + d : ℕ) : ℝ))

-- Define a measure of correlation
def correlation_measure (k : ℝ) : ℝ := k

-- State the theorem
theorem chi_square_correlation {a b c d : ℕ} (k1 k2 : ℝ) 
  (h1 : k1 = chi_square a b c d) 
  (h2 : k2 = chi_square a b c d) 
  (h3 : k1 < k2) : 
  correlation_measure k1 < correlation_measure k2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_correlation_l355_35572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_specific_draw_l355_35583

/-- The probability of drawing 1 green, 2 blue, and 3 red balls from a box -/
theorem probability_of_specific_draw (red blue green total drawn : ℕ) 
  (h_red : red = 15)
  (h_blue : blue = 9)
  (h_green : green = 6)
  (h_total : total = red + blue + green)
  (h_drawn : drawn = 6) :
  (Nat.choose red 3 * Nat.choose blue 2 * Nat.choose green 1 : ℚ) / 
  Nat.choose total drawn = 24 / 145 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_specific_draw_l355_35583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_solution_l355_35593

-- Define vector a
def a : ℝ × ℝ := (1, -2)

-- Define the magnitude of vector b
noncomputable def b_magnitude : ℝ := 2 * Real.sqrt 5

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem vector_b_solution :
  ∃ (b : ℝ × ℝ), (b.1 * b.1 + b.2 * b.2 = b_magnitude * b_magnitude) ∧
                 (parallel a b) ∧
                 ((b = (2, -4)) ∨ (b = (-2, 4))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_solution_l355_35593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l355_35538

noncomputable def y : ℕ → ℝ
  | 0 => 0  -- Add a case for 0
  | 1 => 4^(1/4)
  | 2 => (4^(1/4))^(4^(1/4))
  | n+3 => (y (n+2))^(4^(1/4))

theorem smallest_integer_y : ∀ k : ℕ, k < 4 → ¬(∃ m : ℕ, y k = m) ∧ ∃ m : ℕ, y 4 = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l355_35538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l355_35570

theorem expression_value (a : ℝ) (h : a ≠ 0) : 
  (1 / 16) * a^0 + (1 / (16 * a))^0 - 64^(-(1/2 : ℝ)) - (-32 : ℝ)^(-(4/5 : ℝ)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l355_35570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l355_35574

open Real

theorem trigonometric_simplification (θ : ℝ) :
  (Real.tan (2 * π - θ) * Real.sin (-2 * π - θ) * Real.cos (6 * π - θ)) /
  (Real.cos (θ - π) * Real.sin (5 * π + θ)) = Real.tan θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l355_35574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_incorrect_translation_correct_l355_35577

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the original function
def original (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x)

-- Define the claimed result of translation
def claimed_translation (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x - 1)

-- Define the correct result of translation
def correct_translation (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x + 1)

-- Theorem stating that the claimed translation is incorrect
theorem translation_incorrect (f : ℝ → ℝ) : 
  ∀ x : ℝ, original f (x - 1) ≠ claimed_translation f x :=
sorry

-- Theorem stating that the correct translation is correct
theorem translation_correct (f : ℝ → ℝ) : 
  ∀ x : ℝ, original f (x - 1) = correct_translation f x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_incorrect_translation_correct_l355_35577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_satisfying_functions_mx_satisfies_conditions_l355_35534

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ,
    ((a + b + c ≥ 0 → f (a^3) + f (b^3) + f (c^3) ≥ 3 * f (a*b*c)) ∧
     (a + b + c ≤ 0 → f (a^3) + f (b^3) + f (c^3) ≤ 3 * f (a*b*c)))

/-- The main theorem stating that any function satisfying the conditions is of the form f(x) = mx for some m ≥ 0 -/
theorem characterization_of_satisfying_functions (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∃ m : ℝ, m ≥ 0 ∧ ∀ x : ℝ, f x = m * x := by
  sorry

/-- Proof that mx satisfies the conditions for m ≥ 0 -/
theorem mx_satisfies_conditions (m : ℝ) (hm : m ≥ 0) :
  SatisfiesConditions (fun x ↦ m * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_satisfying_functions_mx_satisfies_conditions_l355_35534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l355_35551

noncomputable def f (x : ℝ) : ℝ := (3 * Real.sin x + 1) / (Real.sin x + 2)

theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -2 ≤ y ∧ y ≤ 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l355_35551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_interval_l355_35541

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.cos (2 * x + Real.pi / 4))

theorem f_monotone_decreasing_interval :
  ∀ x₁ x₂, -Real.pi/8 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi/8 → f x₁ > f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_interval_l355_35541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_speed_l355_35513

/-- The total cost function for flying a plane -/
noncomputable def total_cost (a : ℝ) (v : ℝ) : ℝ := (a / v) * (4900 + 0.01 * v^2)

/-- The theorem stating the speed that minimizes the total cost -/
theorem min_cost_speed (a : ℝ) (h_a : a > 0) :
  ∃ v : ℝ, v > 0 ∧ ∀ u : ℝ, u > 0 → total_cost a v ≤ total_cost a u ∧ v = 700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_speed_l355_35513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_scoops_is_ten_l355_35545

/-- The number of scoops in different ice cream orders -/
structure IceCreamOrder where
  single_cone : ℕ
  double_cone : ℕ
  banana_split : ℕ
  waffle_bowl : ℕ

/-- The conditions of the ice cream orders -/
def ice_cream_conditions (order : IceCreamOrder) : Prop :=
  (order.single_cone = 1) ∧ 
  (order.double_cone = 2 * order.single_cone) ∧
  (order.banana_split = 3 * order.single_cone) ∧
  (order.waffle_bowl = order.banana_split + 1)

/-- The theorem stating that the total number of scoops is 10 -/
theorem total_scoops_is_ten (order : IceCreamOrder) 
  (h : ice_cream_conditions order) : 
  order.single_cone + order.double_cone + order.banana_split + order.waffle_bowl = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_scoops_is_ten_l355_35545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_carpet_shampooing_time_l355_35557

/-- Proves that two people working together can complete a task faster than either working alone -/
theorem combined_work_time (time_A time_B : ℝ) (h_A : time_A > 0) (h_B : time_B > 0) :
  (1 / (1 / time_A + 1 / time_B)) = (time_A * time_B) / (time_A + time_B) := by
  sorry

/-- Proves that if one person can complete a task in 3 hours and another in 6 hours,
    they can complete it together in 2 hours -/
theorem carpet_shampooing_time :
  (1 / (1 / 3 + 1 / 6)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_carpet_shampooing_time_l355_35557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dacid_weighted_average_l355_35509

noncomputable def weighted_average (marks : List ℚ) (credits : List ℚ) : ℚ :=
  (List.sum (List.zipWith (· * ·) marks credits)) / (List.sum credits)

theorem dacid_weighted_average :
  let marks := [72, 45, 72, 77, 75]
  let credits := [3, 4, 4, 3, 2]
  weighted_average marks credits = 1065 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dacid_weighted_average_l355_35509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_expression_l355_35571

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - x) :
  ∀ x, f x = x^2 - 3*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_expression_l355_35571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_average_speed_l355_35553

/-- Represents the hiking scenario with Chantal and Jean --/
structure HikingScenario where
  totalDistance : ℝ
  midpointDistance : ℝ
  chantalInitialSpeed : ℝ
  chantalRoughSpeed : ℝ
  chantalDescentSpeed : ℝ

/-- Calculates Jean's average speed given the hiking scenario --/
noncomputable def calculateJeanSpeed (scenario : HikingScenario) : ℝ :=
  let initialTime := scenario.midpointDistance / scenario.chantalInitialSpeed
  let roughTime := scenario.midpointDistance / scenario.chantalRoughSpeed
  let descentTime := scenario.midpointDistance / scenario.chantalDescentSpeed
  let totalTime := initialTime + roughTime + descentTime
  scenario.midpointDistance / totalTime

/-- Theorem stating that Jean's average speed is 4/3 miles per hour --/
theorem jean_average_speed (scenario : HikingScenario)
  (h1 : scenario.totalDistance = 6)
  (h2 : scenario.midpointDistance = 3)
  (h3 : scenario.chantalInitialSpeed = 6)
  (h4 : scenario.chantalRoughSpeed = 3)
  (h5 : scenario.chantalDescentSpeed = 4) :
  calculateJeanSpeed scenario = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_average_speed_l355_35553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_seven_l355_35546

theorem complex_expression_equals_seven :
  |(-2 : ℤ)| - (Real.pi - 1)^(0 : ℕ) + 4^2023 * (1/4 : ℝ)^2022 + (1/2 : ℝ)^(-1 : ℤ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_seven_l355_35546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_12_25_l355_35512

/-- A trapezoid ABCD with a circumscribed circle whose center lies on base AD -/
structure CircumscribedTrapezoid where
  /-- Point A of the trapezoid -/
  A : ℝ × ℝ
  /-- Point B of the trapezoid -/
  B : ℝ × ℝ
  /-- Point C of the trapezoid -/
  C : ℝ × ℝ
  /-- Point D of the trapezoid -/
  D : ℝ × ℝ
  /-- The center of the circumscribed circle -/
  O : ℝ × ℝ
  /-- The circle is circumscribed around the trapezoid -/
  circumscribed : True
  /-- The center of the circle lies on base AD -/
  center_on_base : True
  /-- Length of AB is 3/4 -/
  AB_length : dist A B = 3/4
  /-- Length of AC is 1 -/
  AC_length : dist A C = 1

/-- Calculate the area of a trapezoid given its four vertices -/
def trapezoid_area (A B C D : ℝ × ℝ) : ℝ := sorry

/-- The area of the trapezoid ABCD is 12/25 -/
theorem trapezoid_area_is_12_25 (t : CircumscribedTrapezoid) : 
  trapezoid_area t.A t.B t.C t.D = 12/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_12_25_l355_35512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_pi_over_nine_l355_35539

/-- The side length of the square region -/
def squareSideLength : ℝ := 6

/-- The radius of the circle centered at the origin -/
def circleRadius : ℝ := 2

/-- The area of the square region -/
def squareArea : ℝ := squareSideLength ^ 2

/-- The area of the circle centered at the origin -/
noncomputable def circleArea : ℝ := Real.pi * circleRadius ^ 2

/-- The probability that a randomly selected point Q from the square region
    is within two units of the origin -/
noncomputable def probability : ℝ := circleArea / squareArea

/-- Theorem: The probability that a randomly selected point Q from a square region
    with vertices at (±3, ±3) is within two units of the origin is π/9 -/
theorem probability_is_pi_over_nine : probability = Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_pi_over_nine_l355_35539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_circle_problem_l355_35588

open Complex

theorem complex_circle_problem (z₁ z₂ : ℂ) :
  z₁ ≠ 0 →
  z₂ ≠ 0 →
  abs (z₁ - I) = 1 →
  abs (z₂ - I) = 1 →
  (z₁ * z₂).re = 0 →
  arg z₁ = π / 6 →
  z₂ = -Real.sqrt 3 / 2 + (3 / 2) * I :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_circle_problem_l355_35588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l355_35565

theorem tan_value_from_sin_cos_sum (α : Real) 
  (h1 : Real.sin α + Real.cos α = 1/5)
  (h2 : -π/2 ≤ α ∧ α ≤ π/2) : 
  Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l355_35565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_perpendicular_l355_35511

/-- A line is symmetric to another line with respect to a third line -/
def IsSymmetric (l l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

/-- Two lines are perpendicular -/
def IsPerpendicular (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

/-- Given three lines in a plane, if the line symmetric to one line with respect to another is perpendicular to the third line, then the coefficient in the third line equation is 1/2. -/
theorem symmetric_line_perpendicular (l₁ l₂ l₃ : Set (ℝ × ℝ)) (m : ℝ) : 
  (∀ x y, (4*x + y = 1) ↔ (x, y) ∈ l₁) →
  (∀ x y, (x - y = 0) ↔ (x, y) ∈ l₂) →
  (∀ x y, (2*x - m*y = 3) ↔ (x, y) ∈ l₃) →
  (∃ l : Set (ℝ × ℝ), IsSymmetric l l₁ l₂ ∧ IsPerpendicular l l₃) →
  m = 1/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_perpendicular_l355_35511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barge_unloading_time_l355_35506

-- Define the total work as 1 unit
def total_work : ℚ := 1

-- Define the time taken by one higher power crane
noncomputable def x : ℚ := 24

-- Define the time taken by one lower power crane
noncomputable def y : ℚ := 36

-- Define the time taken by one higher power crane and one lower power crane together
noncomputable def z : ℚ := 72/5

-- Theorem statement
theorem barge_unloading_time :
  -- Initial 4 cranes work for 2 hours
  (4 / x) * 2 +
  -- Then 4 higher power and 2 lower power cranes work for 3 hours
  ((4 / x + 2 / y) * 3) = total_work ∧
  -- All 6 cranes working together for 4.5 hours
  (4 / x + 2 / y) * (9/2) = total_work ∧
  -- Relationship between x, y, and z
  1 / x + 1 / y = 1 / z ∧
  -- The result we want to prove
  z = 72/5 := by
  sorry

#eval (72 : ℚ) / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barge_unloading_time_l355_35506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_condition_l355_35547

def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ (x y : ℝ), a * x^2 + b * x * y + c * y^2 = (p * x + q * y)^2

theorem perfect_square_trinomial_condition (m : ℝ) :
  is_perfect_square_trinomial 1 m 4 → m = 4 ∨ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_condition_l355_35547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_outside_circle_l355_35531

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def my_line (a b x y : ℝ) : Prop := a*x + b*y = 1

-- Define the point P
def point_P (a b : ℝ) : ℝ × ℝ := (a, b)

-- Define what it means for a point to be outside the circle
def outside_circle (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 > 1

-- Theorem statement
theorem point_P_outside_circle (a b : ℝ) :
  (∃ x y : ℝ, my_circle x y ∧ my_line a b x y) →
  outside_circle (point_P a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_outside_circle_l355_35531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_perfect_square_product_l355_35548

def tiles : Finset ℕ := Finset.range 15 
def die : Finset ℕ := Finset.range 10 

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- We need to make this function decidable
def is_perfect_square_decidable (n : ℕ) : Bool :=
  match n.sqrt with
  | m => m * m = n

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (tiles.product die).filter (λ p => is_perfect_square_decidable ((p.fst + 1) * (p.snd + 1)))

theorem probability_of_perfect_square_product : 
  (favorable_outcomes.card : ℚ) / ((tiles.card * die.card) : ℚ) = 19 / 150 := by
  sorry

#eval favorable_outcomes
#eval favorable_outcomes.card
#eval tiles.card * die.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_perfect_square_product_l355_35548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l355_35563

def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem min_value_f (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → m ≤ f a x) ∧
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → 
    (if a < -1 then 2*a
     else if a ≤ 1 then -1 - a^2
     else -2*a) ≤ f a x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l355_35563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_combo_latest_l355_35527

/-- Represents the infinite string of digits formed by concatenating all natural numbers --/
def digitString : ℕ → ℕ := sorry

/-- Returns the position of the first occurrence of a four-digit combination in the digit string --/
def firstOccurrence (combo : Fin 10000) : ℕ := sorry

/-- The combination "5678" first appears from the 5th to the 8th position --/
axiom combo_5678_position : firstOccurrence 5678 = 5

/-- The combination "0111" first appears from the 11th to the 14th position --/
axiom combo_0111_position : firstOccurrence 111 = 11

/-- Theorem: "0000" appears later than any other four-digit combination --/
theorem zero_combo_latest : ∀ (combo : Fin 10000), combo ≠ 0 → firstOccurrence 0 > firstOccurrence combo := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_combo_latest_l355_35527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_non_officers_l355_35529

/-- Proves that the number of non-officers is 495 given the specified conditions --/
theorem number_of_non_officers (avg_salary avg_salary_officers avg_salary_non_officers : ℝ) 
  (num_officers : ℕ) 
  (h1 : avg_salary = 120)
  (h2 : avg_salary_officers = 450)
  (h3 : avg_salary_non_officers = 110)
  (h4 : num_officers = 15)
  : ∃ (num_non_officers : ℕ), 
    avg_salary * (num_officers + num_non_officers) = 
    avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers ∧ 
    num_non_officers = 495 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_non_officers_l355_35529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l355_35581

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (α + Real.pi/4) = 3/5) 
  (h2 : α ∈ Set.Ioo (Real.pi/4) (3*Real.pi/4)) : 
  Real.cos α = -Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l355_35581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_photography_l355_35562

theorem star_photography (k : ℕ) (h_k : k > 1) 
  (S : Set (Set ℝ)) (h_finite : S.Finite) 
  (h_intervals : ∀ I ∈ S, ∃ a b : ℝ, a < b ∧ I = Set.Icc a b) 
  (h_intersection : ∀ T ⊆ S, T.Finite → T.ncard = k → 
    ∃ I J, I ∈ T ∧ J ∈ T ∧ I ≠ J ∧ (I ∩ J).Nonempty) :
  ∃ P : Set ℝ, P.Finite ∧ P.ncard = k - 1 ∧ 
    ∀ I ∈ S, ∃ p ∈ P, p ∈ I :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_photography_l355_35562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l355_35556

-- Define the given line
def given_line (x y : ℝ) : Prop := 2*x - y = 0

-- Define the point that the new line passes through
def point : ℝ × ℝ := (0, 1)

-- Define the new line
def new_line (x y : ℝ) : Prop := x + 2*y - 2 = 0

-- Define the slope of the given line
def given_line_slope : ℝ := 2

-- Theorem statement
theorem perpendicular_line_through_point :
  (∀ x y : ℝ, new_line x y → (x, y) ≠ point → 
    (y - point.2) / (x - point.1) = -1 / given_line_slope) ∧
  new_line point.1 point.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l355_35556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l355_35578

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 5) + 1 / (x^2 - 4) + 1 / (x^3 - 27)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = 
    {x | x < -2} ∪ {x | -2 < x ∧ x < 2} ∪ {x | 2 < x ∧ x < 3} ∪ 
    {x | 3 < x ∧ x < 5} ∪ {x | 5 < x} := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l355_35578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_sharks_fin_area_l355_35573

/-- The area of a modified shark's fin falcata -/
theorem modified_sharks_fin_area : 
  let outer_circle_radius : ℝ := 5
  let inner_circle_radius : ℝ := 2
  let inner_circle_center_y : ℝ := 3
  let outer_quarter_circle_area : ℝ := (Real.pi * outer_circle_radius^2) / 4
  let inner_quarter_circle_area : ℝ := (Real.pi * inner_circle_radius^2) / 4
  outer_quarter_circle_area - inner_quarter_circle_area = 21 * Real.pi / 4 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_sharks_fin_area_l355_35573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l355_35595

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

noncomputable def S (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + ((n - 1 : ℕ) : ℝ) * d) / 2

theorem arithmetic_sequence_max_sum
  (a₁ : ℝ)
  (d : ℝ)
  (h₁ : a₁ > 0)
  (h₂ : 3 * (arithmetic_sequence a₁ d 8) = 5 * (arithmetic_sequence a₁ d 13)) :
  ∀ n : ℕ, n > 0 → S a₁ d 20 ≥ S a₁ d n :=
by
  sorry

#check arithmetic_sequence_max_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l355_35595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_change_l355_35579

/-- Represents a cylinder with radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Theorem: If a cylinder's original volume is 10 cubic feet, and its radius is doubled
    and height is tripled, then its new volume is 120 cubic feet -/
theorem cylinder_volume_change (c : Cylinder) :
  volume c = 10 →
  volume { radius := 2 * c.radius, height := 3 * c.height } = 120 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_change_l355_35579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_total_distance_l355_35526

def bug_crawl (start end1 end2 : Int) : Nat :=
  (end1 - start).natAbs + (end2 - end1).natAbs

theorem bug_total_distance :
  bug_crawl (-2) (-6) 5 = 15 := by
  rfl

#eval bug_crawl (-2) (-6) 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_total_distance_l355_35526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l355_35555

-- Define the triangle sides
def a : ℝ := 31
def b : ℝ := 56
def c : ℝ := 40

-- Define the semi-perimeter
noncomputable def s : ℝ := (a + b + c) / 2

-- Define Heron's formula for the area
noncomputable def triangle_area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- State the theorem
theorem triangle_area_approx : 
  |triangle_area - 190.274| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l355_35555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l355_35528

theorem trig_simplification (x : ℝ) (h1 : Real.sin x ≠ 0) (h2 : 1 + Real.cos x ≠ 0) :
  (Real.sin x) / (1 + Real.cos x) + (1 + Real.cos x) / (Real.sin x) = 2 * (1 / Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l355_35528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_week_iced_coffee_cost_l355_35525

/-- Calculates the cost of iced coffee for a given period -/
def icedCoffeeCost (servingsPerBottle : ℕ) (bottlesPerDay : ℚ) (costPerBottle : ℚ) (days : ℕ) : ℚ :=
  (bottlesPerDay * days * costPerBottle).ceil

/-- Theorem stating the cost of iced coffee for 2 weeks under given conditions -/
theorem two_week_iced_coffee_cost :
  let servingsPerBottle : ℕ := 6
  let bottlesPerDay : ℚ := 1/2
  let costPerBottle : ℚ := 3
  let daysInTwoWeeks : ℕ := 14
  icedCoffeeCost servingsPerBottle bottlesPerDay costPerBottle daysInTwoWeeks = 21 := by
  sorry

#eval icedCoffeeCost 6 (1/2) 3 14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_week_iced_coffee_cost_l355_35525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l355_35510

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_gt_b : b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The right focus of an ellipse -/
noncomputable def right_focus (e : Ellipse) : ℝ × ℝ :=
  (e.a * eccentricity e, 0)

/-- An endpoint of the minor axis -/
def minor_axis_endpoint (e : Ellipse) : ℝ × ℝ :=
  (0, e.b)

/-- Predicate to check if a point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : ℝ × ℝ) : Prop :=
  (p.1 / e.a)^2 + (p.2 / e.b)^2 = 1

/-- Predicate to check if a point is the trisection point of a line segment -/
def is_trisection_point (a b c : ℝ × ℝ) : Prop :=
  2 * (c.1 - a.1) = b.1 - a.1 ∧ 2 * (c.2 - a.2) = b.2 - a.2

theorem ellipse_eccentricity_theorem (e : Ellipse) :
  let f := right_focus e
  let a := minor_axis_endpoint e
  ∃ m : ℝ × ℝ, on_ellipse e m ∧ is_trisection_point a m f →
  eccentricity e = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l355_35510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_101_equals_153_l355_35540

def F : ℕ → ℚ
  | 0 => 3  -- Adding the base case for 0
  | 1 => 3
  | (n + 2) => (2 * F (n + 1) + 3) / 2

theorem F_101_equals_153 : F 101 = 153 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_101_equals_153_l355_35540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_specific_line_l355_35518

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ :=
  l.x₁ - l.y₁ * (l.x₁ - l.x₂) / (l.y₁ - l.y₂)

/-- The theorem stating that the x-intercept of the line passing through (10, 3) and (-6, -5) is 4 -/
theorem x_intercept_of_specific_line :
  x_intercept { x₁ := 10, y₁ := 3, x₂ := -6, y₂ := -5 } = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_specific_line_l355_35518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_growth_threshold_l355_35516

/-- The growth rate constant for the player base --/
noncomputable def k : ℝ := Real.log 10 / 5

/-- The number of players at time t --/
noncomputable def R (t : ℝ) : ℝ := 100 * Real.exp (k * t)

/-- The minimum number of days for the player count to exceed 30000 --/
def min_days : ℕ := 13

theorem player_growth_threshold : 
  (∀ t : ℕ, t < min_days → R t ≤ 30000) ∧ 
  R min_days > 30000 := by
  sorry

#check player_growth_threshold

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_growth_threshold_l355_35516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_count_l355_35507

-- Define the set S
def S : Set (ℝ × ℝ × ℝ × ℝ) :=
  {q : ℝ × ℝ × ℝ × ℝ | 
    let (a, b, c, d) := q
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a^2 + b^2 + c^2 + d^2 = 9 ∧
    (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81}

-- State the theorem
theorem quadruple_count : Nat.card S = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_count_l355_35507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_red_given_red_is_one_l355_35580

/-- Represents a card with two sides --/
inductive Card
| GG  -- Green on both sides
| GR  -- Green on one side, Red on the other
| RR  -- Red on both sides
deriving DecidableEq

/-- The box of cards --/
def box : Multiset Card :=
  Multiset.replicate 5 Card.GG +
  Multiset.replicate 2 Card.GR +
  Multiset.replicate 3 Card.RR

/-- The probability of drawing a card from the box --/
noncomputable def probDraw (c : Card) : ℝ :=
  (box.count c : ℝ) / (Multiset.card box : ℝ)

/-- The probability of observing a red side on a given card --/
noncomputable def probRedSide (c : Card) : ℝ :=
  match c with
  | Card.GG => 0
  | Card.GR => 1/2
  | Card.RR => 1

/-- The probability of observing a red side --/
noncomputable def probRedObserved : ℝ :=
  Finset.sum (Multiset.toFinset box) (λ c => probDraw c * probRedSide c)

/-- The probability that both sides are red, given that a red side is observed --/
noncomputable def probBothRedGivenRed : ℝ :=
  (probDraw Card.RR * probRedSide Card.RR) / probRedObserved

theorem prob_both_red_given_red_is_one :
  probBothRedGivenRed = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_red_given_red_is_one_l355_35580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_store_revenue_ratio_l355_35533

theorem toy_store_revenue_ratio : 
  ∀ (december : ℚ), december > 0 →
  let november := (2 : ℚ) / 5 * december
  let january := (1 : ℚ) / 2 * november
  let average := (november + january) / 2
  december / average = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_store_revenue_ratio_l355_35533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_eight_digit_binary_l355_35544

theorem least_eight_digit_binary : ∃ n : ℕ, n > 0 ∧
  (∀ m : ℕ, m < n → Nat.log2 m < 7) ∧ Nat.log2 n = 7 := by
  -- The proof goes here
  sorry

#check least_eight_digit_binary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_eight_digit_binary_l355_35544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l355_35591

-- Define the given parameters
noncomputable def train_length : ℝ := 300
noncomputable def signal_crossing_time : ℝ := 10
noncomputable def platform_length : ℝ := 870

-- Define the train speed
noncomputable def train_speed : ℝ := train_length / signal_crossing_time

-- Define the total distance to cross the platform
noncomputable def total_distance : ℝ := train_length + platform_length

-- Theorem statement
theorem train_platform_crossing_time :
  total_distance / train_speed = 39 := by
  -- Unfold definitions
  unfold total_distance train_speed train_length signal_crossing_time platform_length
  
  -- Perform the calculation
  norm_num
  
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l355_35591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recovery_distance_range_l355_35515

/-- Represents the gas consumption and cost parameters for a car --/
structure CarParameters where
  original_consumption : ℚ  -- Liters per 100 km before modification
  modified_consumption : ℚ  -- Liters per 100 km after modification
  modification_cost : ℚ     -- Cost of modification in dollars
  gas_price : ℚ             -- Price of gas per liter in dollars

/-- Calculates the minimum distance required to recover modification cost --/
def minimum_recovery_distance (params : CarParameters) : ℚ :=
  let gas_saved_per_100km := params.original_consumption - params.modified_consumption
  let equivalent_gas_liters := params.modification_cost / params.gas_price
  (equivalent_gas_liters / gas_saved_per_100km) * 100

/-- Theorem stating that the minimum recovery distance is between 22000 and 26000 km --/
theorem recovery_distance_range (params : CarParameters) 
    (h1 : params.original_consumption = 84/10)
    (h2 : params.modified_consumption = 63/10)
    (h3 : params.modification_cost = 400)
    (h4 : params.gas_price = 4/5) : 
  22000 < minimum_recovery_distance params ∧ minimum_recovery_distance params < 26000 := by
  sorry

#eval minimum_recovery_distance { 
  original_consumption := 84/10, 
  modified_consumption := 63/10, 
  modification_cost := 400, 
  gas_price := 4/5 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recovery_distance_range_l355_35515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_value_l355_35537

theorem greatest_x_value (x : ℤ) : x ≤ 3 ∧ (2.134 * (10 : ℝ)^x < 21000) ↔ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_value_l355_35537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l355_35520

noncomputable def f (A ω α : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + α)

theorem sine_function_properties (A ω α : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : -π/2 < α ∧ α < π/2) 
  (h4 : ∀ x, f A ω α (x + π/ω) = f A ω α x) 
  (h5 : f A ω α (π/6) = 5) 
  (h6 : ∀ x, f A ω α x ≤ 5) :
  (∀ x, f A ω α x = 5 * Real.sin (2*x + π/6)) ∧
  (∀ m : ℝ, m > 0 ∧ (∀ x, f A ω α (x - m) = f A ω α (m - x)) → m ≥ π/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l355_35520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_power_of_two_l355_35582

theorem divisibility_by_power_of_two (n : ℕ) : 
  ∃ m : ℤ, (2018 * m^2 + 20182017 * m + 2017) % (2^n : ℤ) = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_power_of_two_l355_35582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_and_AB_distance_l355_35560

-- Define the parametric equations of curve C₁
noncomputable def C₁ (θ : Real) : Real × Real :=
  (1 + Real.sqrt 3 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the relation between points on C₁ and C₂
def C₂_relation (M P : Real × Real) : Prop :=
  P.1 = 2 * M.1 ∧ P.2 = 2 * M.2

-- Define the equation of curve C₂
def C₂_equation (x y : Real) : Prop :=
  (x - 2)^2 + y^2 = 12

-- Define points A and B
noncomputable def A : Real × Real := C₁ (Real.pi / 3)
noncomputable def B : Real × Real := (4 * Real.cos (Real.pi / 3), 4 * Real.sin (Real.pi / 3))

-- Theorem statement
theorem C₂_and_AB_distance :
  (∀ θ : Real, ∃ P : Real × Real, C₂_relation (C₁ θ) P ∧ C₂_equation P.1 P.2) ∧
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_and_AB_distance_l355_35560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_sum_zero_l355_35590

noncomputable def f (x : ℝ) := 1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5))

noncomputable def g (x A B C D E F : ℝ) := A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)

theorem partial_fraction_sum_zero :
  ∃ A B C D E F : ℝ, (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 → f x = g x A B C D E F) →
  A + B + C + D + E + F = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_sum_zero_l355_35590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_rate_equation_l355_35542

/-- Represents the monthly average growth rate of smart charging piles built -/
def x : Real := sorry

/-- Number of charging piles built in the first month -/
def first_month : Nat := 302

/-- Number of charging piles built in the third month -/
def third_month : Nat := 503

/-- Theorem stating that the equation correctly represents the growth rate -/
theorem growth_rate_equation : 
  (first_month : Real) * (1 + x)^2 = third_month := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_rate_equation_l355_35542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l355_35502

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

-- Theorem statement
theorem f_has_unique_zero (a b : ℝ) :
  ((1/2 < a ∧ a ≤ Real.exp 2 / 2 ∧ b > 2*a) ∨
   (0 < a ∧ a < 1/2 ∧ b ≤ 2*a)) →
  ∃! x : ℝ, f a b x = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l355_35502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_explicit_formula_l355_35522

def a : ℕ → ℤ
  | 0 => 1  -- Add this case to handle n = 0
  | 1 => 1
  | n + 1 => 2 * a n + n^2

theorem a_explicit_formula (n : ℕ) (h : n ≥ 1) :
  a n = 7 * 2^(n - 1) - n^2 - 2*n - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_explicit_formula_l355_35522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_max_in_triangle_l355_35521

open Real

-- Define the convexity of sin x on (0, π)
def sin_convex_on_0_pi : Prop :=
  ∀ x y : ℝ, 0 < x ∧ x < Real.pi ∧ 0 < y ∧ y < Real.pi →
    ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 →
      sin (t * x + (1 - t) * y) ≤ t * sin x + (1 - t) * sin y

-- Define a triangle by its angles
def is_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi

-- State the theorem
theorem sin_sum_max_in_triangle (h_convex : sin_convex_on_0_pi) :
  ∀ A B C : ℝ, is_triangle A B C →
    sin A + sin B + sin C ≤ 3 * sqrt 3 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_max_in_triangle_l355_35521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_m_value_l355_35532

/-- Prove that if a line passing through points A(-2, m) and B(m, 4) is parallel to the line 2x + y + 1 = 0, then m = -8. -/
theorem parallel_line_m_value (m : ℝ) : 
  (((4 - m) / (m + 2)) = -2) → m = -8 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_m_value_l355_35532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_factors_count_l355_35508

/-- The number of even natural-number factors of 2^3 * 3^2 * 5^2 with restricted exponents -/
def count_even_factors : ℕ :=
  Finset.card (Finset.filter
    (fun (a, b, c) => 1 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ a + b + c ≤ 4)
    (Finset.product (Finset.range 4) (Finset.product (Finset.range 3) (Finset.range 3))))

theorem even_factors_count : count_even_factors = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_factors_count_l355_35508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l355_35586

def a : ℕ → ℕ
| 0 => 3
| n + 1 => 2 * a n + 1

theorem sequence_formula : 
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n+1) - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l355_35586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_grid_distances_l355_35575

/-- Complex cube root of unity -/
noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

/-- Vertex in the triangular grid -/
def Vertex : Type := ℤ × ℤ

/-- Convert a vertex to a complex number -/
noncomputable def vertex_to_complex (v : Vertex) : ℂ := v.1 + v.2 * ω

/-- Distance between two vertices -/
noncomputable def vertex_distance (v₁ v₂ : Vertex) : ℝ :=
  Complex.abs (vertex_to_complex v₁ - vertex_to_complex v₂)

theorem equilateral_triangle_grid_distances :
  (∀ (h k : ℝ), (∃ (A B C D : Vertex), vertex_distance A B = h ∧ vertex_distance C D = k) →
    ∃ (E F : Vertex), vertex_distance E F = h * k) ∧
  ∃ (G H : Vertex), vertex_distance G H = Real.sqrt 1981 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_grid_distances_l355_35575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l355_35503

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2/8 = 1

/-- The foci of the hyperbola -/
def foci (F₁ F₂ : ℝ × ℝ) : Prop := sorry

/-- A point on the hyperbola -/
def point_on_hyperbola (P : ℝ × ℝ) : Prop := 
  let (x, y) := P; hyperbola_equation x y

/-- PF₁ is perpendicular to PF₂ -/
def perpendicular (P F₁ F₂ : ℝ × ℝ) : Prop := sorry

/-- The area of triangle PF₁F₂ -/
def triangle_area (P F₁ F₂ : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_triangle_area 
  (F₁ F₂ P : ℝ × ℝ) 
  (h₁ : foci F₁ F₂)
  (h₂ : point_on_hyperbola P)
  (h₃ : perpendicular P F₁ F₂) :
  triangle_area P F₁ F₂ = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l355_35503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_pi_l355_35550

-- Define the region
def region (x y : ℝ) : Prop :=
  y^2 = 2*x ∧ 0 ≤ x ∧ x ≤ 1 ∧ y ≥ 0

-- Define the volume of the solid of revolution
noncomputable def volume_of_revolution : ℝ :=
  Real.pi * ∫ x in (Set.Icc 0 1), 2*x

-- Theorem statement
theorem volume_equals_pi : volume_of_revolution = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_pi_l355_35550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_problem_l355_35549

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^((4*a)/5)

-- State the theorem
theorem function_problem (a : ℝ) :
  (a > 0) → (a ≠ 1) → (f a 2 = 1/4) →
  (a = 1/2) ∧
  (∀ t : ℝ, g (1/2) (2*t - 1) < g (1/2) (t + 1) ↔ 0 < t ∧ t < 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_problem_l355_35549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximation_l355_35524

/-- Calculates the speed of a train crossing a bridge -/
noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  3.6 * speed_ms

theorem train_speed_approximation :
  let train_length : ℝ := 165
  let bridge_length : ℝ := 720
  let crossing_time : ℝ := 58.9952803775698
  abs (train_speed train_length bridge_length crossing_time - 54) < 0.01 := by
  sorry

-- Use #eval only for computable functions
def approximate_train_speed : ℚ :=
  let train_length : ℚ := 165
  let bridge_length : ℚ := 720
  let crossing_time : ℚ := 58.9952803775698
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  3.6 * speed_ms

#eval approximate_train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximation_l355_35524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l355_35589

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + Real.sin x + 1

theorem f_extrema :
  (∃ x : ℝ, f x = -1) ∧ 
  (∃ x : ℝ, f x = 17/8) ∧ 
  (∀ x : ℝ, -1 ≤ f x ∧ f x ≤ 17/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l355_35589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_minus_2alpha_l355_35568

theorem tan_beta_minus_2alpha (α β : ℝ) : 
  0 < α → α < π/2 →  -- α is in the first quadrant
  Real.sin α ^ 2 + Real.sin α * Real.cos α = 3/5 →
  Real.tan (α - β) = -3/2 →
  Real.tan (β - 2*α) = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_minus_2alpha_l355_35568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l355_35576

theorem min_value_f (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : c * y + b * z = a)
  (eq2 : a * z + c * x = b)
  (eq3 : b * x + a * y = c) :
  let f := fun (x y z : ℝ) ↦ x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)
  ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → f x' y' z' ≥ (1 / 2 : ℝ) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ f x₀ y₀ z₀ = (1 / 2 : ℝ) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l355_35576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l355_35505

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

theorem triangle_theorem (t : Triangle) :
  (t.b * t.c = t.a^2 - (t.b - t.c)^2) →
  (t.A = π/3) ∧
  ((t.a = 2 * Real.sqrt 3 ∧ t.area = 2 * Real.sqrt 3) →
    ((t.b = 4 ∧ t.c = 2) ∨ (t.b = 2 ∧ t.c = 4))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l355_35505
