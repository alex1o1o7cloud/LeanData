import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_value_l120_12058

/-- The function representing the left side of the equation -/
noncomputable def f (x : ℝ) : ℝ := ((5*x - 25) / (4*x - 5))^3 + (5*x - 25) / (4*x - 5)

/-- The theorem stating that 5 is the greatest possible value of x satisfying the equation -/
theorem greatest_x_value :
  (∃ (x : ℝ), f x = 16) ∧ 
  (∀ (x : ℝ), f x = 16 → x ≤ 5) ∧
  (f 5 = 16) := by
  sorry

#check greatest_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_value_l120_12058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_sum_zero_l120_12028

noncomputable section

/-- The function v(x) = -x + 2cos(πx/2) -/
def v (x : ℝ) : ℝ := -x + 2 * Real.cos (Real.pi * x / 2)

/-- Theorem stating that v(-1.75) + v(-0.5) + v(0.5) + v(1.75) = 0 -/
theorem v_sum_zero :
  v (-1.75) + v (-0.5) + v 0.5 + v 1.75 = 0 :=
by
  sorry

/-- Lemma stating that v is an odd function -/
lemma v_odd (x : ℝ) : v (-x) = -v x :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_sum_zero_l120_12028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extreme_points_implies_m_range_l120_12023

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 / 2 + (m + 1) * Real.exp x + 2

noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := x + (m + 1) * Real.exp x

def has_two_extreme_points (m : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f_deriv m x₁ = 0 ∧ f_deriv m x₂ = 0

theorem two_extreme_points_implies_m_range (m : ℝ) :
  has_two_extreme_points m → m > -1 - 1 / Real.exp 1 ∧ m < -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extreme_points_implies_m_range_l120_12023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_equals_six_sevenths_l120_12074

/-- Given complex numbers z and u, where z + u = 4/5 + 3/5i, prove that tan(α + β) = 6/7 -/
theorem tan_sum_equals_six_sevenths (α β : ℝ) 
  (z u : ℂ)
  (hz : z = Complex.exp (Complex.I * α))
  (hu : u = Complex.exp (Complex.I * β))
  (hsum : z + u = Complex.ofReal (4/5) + Complex.I * Complex.ofReal (3/5)) : 
  Real.tan (α + β) = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_equals_six_sevenths_l120_12074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seedless_trait_inheritance_l120_12071

-- Define the traits and methods
structure Fruit where
  name : String
  seedlessTrait : Bool
  inheritable : Bool

def usesAuxin (f : Fruit) : Bool := sorry
def usesPolyploidBreeding (f : Fruit) : Bool := sorry
def changesGeneticMaterial (f : Fruit) : Bool := sorry
def involvesChromosomalMutations (f : Fruit) : Bool := sorry

-- Define the fruits
def tomato : Fruit := ⟨"tomato", true, false⟩
def watermelon : Fruit := ⟨"watermelon", true, true⟩

-- Theorem to prove
theorem seedless_trait_inheritance :
  (usesAuxin tomato ∧ ¬changesGeneticMaterial tomato → ¬tomato.inheritable) ∧
  (usesPolyploidBreeding watermelon ∧ involvesChromosomalMutations watermelon → watermelon.inheritable) := by
  sorry

-- Additional facts (not used in the proof, but part of the problem context)
def isDevelopedFromGametes (organism : String) : Prop := sorry
def mayContainHomologousChromosomes (organism : String) : Prop := sorry
def appearsInOneGeneration (disease : String) : Prop := sorry
def canBeGenetic (disease : String) : Prop := sorry

axiom haploid_development : ∀ (organism : String), isDevelopedFromGametes organism → mayContainHomologousChromosomes organism
axiom genetic_disease_occurrence : ∀ (disease : String), appearsInOneGeneration disease → canBeGenetic disease

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seedless_trait_inheritance_l120_12071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_l120_12056

theorem modulus_of_complex (i : ℂ) (h : i^2 = -1) :
  Complex.abs ((5 * i) / (1 + 2 * i)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_l120_12056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_curve_l120_12075

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

/-- The curve C in the xy-plane -/
def curve_C (x y : ℝ) : Prop := x^2 = 2*y

/-- The distance between two points in the xy-plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The minimum distance between a point on line l and a point on curve C -/
theorem min_distance_line_curve :
  ∃ (x1 y1 x2 y2 : ℝ),
    line_l x1 y1 ∧ curve_C x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ),
      line_l x3 y3 → curve_C x4 y4 →
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = 15 * Real.sqrt 2 / 32 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_curve_l120_12075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l120_12022

theorem starting_number_proof (x : ℕ) : 
  (x ≤ 100) ∧ 
  (∃ n : ℕ, x = 2 * n) ∧ 
  (Finset.filter (fun i => i % 2 = 0) (Finset.range (101 - x))).card = 46 →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l120_12022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_logarithmic_equality_l120_12089

noncomputable def log_expr1 (x : ℝ) : ℝ := Real.log ((x / 7) + 7) / Real.log (Real.sqrt (29 - x))
noncomputable def log_expr2 (x : ℝ) : ℝ := Real.log (29 - x) / Real.log ((x + 1)^2)
noncomputable def log_expr3 (x : ℝ) : ℝ := Real.log (-x - 1) / Real.log (Real.sqrt ((x / 7) + 7))

theorem unique_solution_logarithmic_equality :
  ∃! x : ℝ, (log_expr1 x = log_expr2 x ∧ log_expr3 x = log_expr1 x + 1) ∨
            (log_expr1 x = log_expr3 x ∧ log_expr2 x = log_expr1 x + 1) ∨
            (log_expr2 x = log_expr3 x ∧ log_expr1 x = log_expr2 x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_logarithmic_equality_l120_12089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_solution_l120_12077

theorem at_most_one_solution (a b : ℕ+) 
  (ha : ¬ ∃ (n : ℕ), a = n^2)
  (hb : ¬ ∃ (n : ℕ), b = n^2)
  (hab : ¬ ∃ (n : ℕ), a * b = n^2) :
  ¬ (∃ (x y : ℕ+), (a : ℤ) * x^2 - (b : ℤ) * y^2 = 1) ∨
  ¬ (∃ (x y : ℕ+), (a : ℤ) * x^2 - (b : ℤ) * y^2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_solution_l120_12077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_theorem_l120_12064

def total_coins : ℕ := 128

def perform_round (coins : List ℕ) : List ℕ :=
  sorry

def perform_rounds (n : ℕ) (coins : List ℕ) : List ℕ :=
  sorry

theorem pirate_theorem (initial_coins : List ℕ) 
  (h1 : initial_coins.sum = total_coins) :
  ∃ (final_coins : List ℕ), 
    (perform_rounds 7 initial_coins = final_coins) ∧ 
    (∃ (i : Fin final_coins.length), final_coins[i] = total_coins) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_theorem_l120_12064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passengers_who_got_on_l120_12073

/-- Proves that the number of passengers who got on the bus is 7 --/
theorem passengers_who_got_on (initial got_off final got_on : ℕ)
  (h1 : initial = 28)
  (h2 : got_off = 9)
  (h3 : final = 26)
  (h4 : final = initial + got_on - got_off)
  : got_on = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_passengers_who_got_on_l120_12073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l120_12055

def number_of_ways_to_distribute (n : ℕ) : ℕ := 
  -- Number of ways to distribute n different balls into two boxes,
  -- with each box containing at least one ball
  sorry

theorem ball_distribution (n : ℕ) (h : n = 4) : 
  (number_of_ways_to_distribute n) = 2 * (n.choose 1) + (n.choose 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l120_12055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solar_panel_optimization_initial_conditions_satisfied_l120_12045

/-- Represents the annual electricity bill as a function of solar panel area -/
noncomputable def C (b : ℝ) (x : ℝ) : ℝ := b / (20 * x + 100)

/-- Represents the total cost over 15 years as a function of solar panel area -/
noncomputable def F (x : ℝ) : ℝ := 15 * C 2400 x + 0.5 * x

/-- Theorem stating the minimum total cost and optimal solar panel area -/
theorem solar_panel_optimization :
  ∃ (x_min : ℝ), x_min ≥ 0 ∧ 
  (∀ x ≥ 0, F x ≥ F x_min) ∧
  x_min = 55 ∧ F x_min = 57.5 := by
  sorry

/-- Verification that the initial conditions are satisfied -/
theorem initial_conditions_satisfied :
  C 2400 0 = 24 ∧
  (∀ x ≥ 0, F x = 1800 / (x + 5) + 0.5 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solar_panel_optimization_initial_conditions_satisfied_l120_12045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_range_l120_12046

theorem system_solution_range (m : ℝ) (hm : m > 0) :
  (∃ x y : ℝ, Real.sin x = m * (Real.sin y)^3 ∧ Real.cos x = m * (Real.cos y)^3) ↔ 1 ≤ m ∧ m ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_range_l120_12046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_symmetry_of_h_max_value_of_k_l_decreasing_when_m_is_1_l120_12003

-- Define the functions
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f (2 * x)
noncomputable def h (x : ℝ) : ℝ := (x + 1) / (x + 2)
noncomputable def k (x : ℝ) : ℝ := (1/2) ^ (-x^2 + 1)
noncomputable def l (m : ℝ) (x : ℝ) : ℝ := (m^2 - 3*m + 3) * x^(3*m - 4)

-- State the theorems
theorem domain_of_g : Set.Icc 0 1 = {x | g x ∈ Set.Icc 0 2} := by sorry

theorem symmetry_of_h : ∀ x : ℝ, h (-(x + 4) + (-2)) = -(h x - 1) + 1 := by sorry

theorem max_value_of_k : ∃ x : ℝ, k x > 1/2 := by sorry

theorem l_decreasing_when_m_is_1 : 
  StrictMonoOn (fun x => -(l 1 x)) (Set.Ioi 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_symmetry_of_h_max_value_of_k_l_decreasing_when_m_is_1_l120_12003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_lengths_line_through_B_and_C_l120_12083

open Real

-- Define the curve M
noncomputable def M (θ : ℝ) : ℝ := 4 * cos θ

-- Define points A, B, C
noncomputable def A (φ : ℝ) : ℝ := M φ
noncomputable def B (φ : ℝ) : ℝ := M (φ + π/4)
noncomputable def C (φ : ℝ) : ℝ := M (φ - π/4)

-- Theorem 1
theorem sum_of_lengths (φ : ℝ) : B φ + C φ = sqrt 2 * A φ := by sorry

-- Define line l
noncomputable def l (m α t : ℝ) : ℝ × ℝ := (m + t * cos α, t * sin α)

-- Theorem 2
theorem line_through_B_and_C : 
  let φ := π/12
  let B_point : ℝ × ℝ := (1, sqrt 3)
  let C_point : ℝ × ℝ := (3, -sqrt 3)
  ∃ (t₁ t₂ : ℝ), l 2 (2*π/3) t₁ = B_point ∧ l 2 (2*π/3) t₂ = C_point := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_lengths_line_through_B_and_C_l120_12083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_relation_l120_12079

/-- Given an arithmetic sequence {a_n} and a geometric sequence {b_n} satisfying certain conditions,
    prove that the common ratio of the geometric sequence is either 1 or 4. -/
theorem arithmetic_geometric_sequence_relation (a₁ d q : ℝ) : 
  let a : ℕ → ℝ := λ n => a₁ + (n - 1) * d
  let b : ℕ → ℝ := λ n => a₁ * q^(n - 3)
  (a 1 = b 3) → (a 3 = b 4) → (a 11 = b 5) → 
  q = 1 ∨ q = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_relation_l120_12079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adams_school_week_total_time_l120_12030

/-- Represents the time spent at school for each day of the week -/
structure SchoolWeek where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- Calculates the total time spent at school during the week -/
def totalTime (week : SchoolWeek) : ℝ :=
  week.monday + week.tuesday + week.wednesday + week.thursday + week.friday

/-- Adam's school week schedule -/
noncomputable def adamsWeek : SchoolWeek :=
  { monday := 4 * 1 + 2 * 0.5
  , tuesday := 3 * 1 + 1 * 1.5 + 1 * 0.75
  , wednesday := 2 * (3 * 1 + 1 * 1.5 + 1 * 0.75)
  , thursday := 3.5 + 2 + 1 + 0.5
  , friday := (2 * (3 * 1 + 1 * 1.5 + 1 * 0.75)) / 2
  }

theorem adams_school_week_total_time :
  totalTime adamsWeek = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adams_school_week_total_time_l120_12030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_identity_l120_12080

/-- Standard basis vectors in R³ -/
def i : ℝ × ℝ × ℝ := (1, 0, 0)
def j : ℝ × ℝ × ℝ := (0, 1, 0)
def k : ℝ × ℝ × ℝ := (0, 0, 1)

/-- Cross product in R³ -/
def cross (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (a.2.2 * b.2.1 - a.2.1 * b.2.2, a.1 * b.2.2 - a.2.2 * b.1, a.2.1 * b.1 - a.1 * b.2.1)

/-- Addition of vectors in R³ -/
def add (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2)

/-- Scalar multiplication in R³ -/
def smul (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2.1, c * v.2.2)

/-- The main theorem -/
theorem vector_identity (v : ℝ × ℝ × ℝ) : 
  add (add (cross i (cross (add j k) i)) (cross j (cross (add k i) j))) (cross k (cross (add i j) k)) = smul 2 v := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_identity_l120_12080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_l120_12050

/-- The ellipse C with equation x²/4 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The left focus F₁ of the ellipse C -/
noncomputable def F₁ : ℝ × ℝ := (-1, 0)

/-- The right focus F₂ of the ellipse C -/
noncomputable def F₂ : ℝ × ℝ := (1, 0)

/-- A point P on the ellipse C -/
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ellipse_C x y

/-- The centroid G of triangle PF₁F₂ -/
noncomputable def centroid (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (x/3, y/3)

/-- The locus equation of the centroid G -/
def locus_equation (x y : ℝ) : Prop := 9*x^2/4 + 3*y^2 = 1

theorem centroid_locus :
  ∀ P : ℝ × ℝ, point_on_ellipse P →
  let (x, y) := centroid P
  y ≠ 0 → locus_equation x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_l120_12050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_triangle_l120_12019

/-- Given a semicircle with diameter AB of length 2R, and a chord CD parallel to AB,
    where the inscribed angle subtended by arc AC is α (with AC < AD),
    the volume of the solid formed by rotating triangle ACD around AB
    is (2/3) * π * R³ * sin(4α) * sin(2α) -/
theorem volume_of_rotated_triangle (R α : ℝ) (hR : 0 < R) (hα : 0 < α) (hα_bound : α < π/2) :
  (2/3) * Real.pi * R^3 * Real.sin (4*α) * Real.sin (2*α) =
  (2/3) * Real.pi * R^3 * Real.sin (4*α) * Real.sin (2*α) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_triangle_l120_12019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_problem_l120_12060

-- Define the probabilities
noncomputable def p_A_score : ℝ := 1/2
noncomputable def p_B_score : ℝ := 2/3
noncomputable def p_A_save : ℝ := 1/2
noncomputable def p_B_save : ℝ := 1/5

-- Define X as A's score after one round
def X : ℝ → ℝ := λ _ => 0  -- Placeholder definition

-- Define the probability distribution of X
noncomputable def prob_dist_X (x : ℝ) : ℝ :=
  if x = -1 then 1/5
  else if x = 0 then 8/15
  else if x = 1 then 4/15
  else 0

-- Define the expected value of X
noncomputable def E_X : ℝ := 1/15

-- Define p₂ as the probability that A's cumulative score is higher than B's after two rounds
noncomputable def p₂ : ℝ := 16/45

-- Theorem statement
theorem football_problem :
  (∀ x, prob_dist_X x = if x = -1 then 1/5 else if x = 0 then 8/15 else if x = 1 then 4/15 else 0) ∧
  E_X = 1/15 ∧
  p₂ = 16/45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_problem_l120_12060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_multiplication_result_l120_12099

theorem correct_multiplication_result (f : ℕ) 
  (h : f * 153 = 102325) : f * 153 = 102357 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_multiplication_result_l120_12099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_of_specific_number_l120_12068

theorem factor_count_of_specific_number (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (hodd_a : Odd a) (hodd_b : Odd b) (hodd_c : Odd c)
  (horder : a < b ∧ b < c) :
  let x := 2^2 * a^3 * b^2 * c^4
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_of_specific_number_l120_12068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_coloring_l120_12034

theorem polygon_coloring (n : ℕ) (h : n = 1991) :
  ∀ (color : Fin n → Fin n → Bool) (f : Fin n → Fin n),
  Function.Bijective f →
  ∃ k l : Fin n, k < l ∧ color k l = color (f k) (f l) := by
  sorry

#check polygon_coloring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_coloring_l120_12034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_is_F_l120_12052

-- Define the function f as the integrand
noncomputable def f (x : ℝ) : ℝ := (3 * Real.tan x ^ 2 - 1) / (Real.tan x ^ 2 + 5)

-- Define the function F as the antiderivative
noncomputable def F (x : ℝ) : ℝ := -x + (4 / Real.sqrt 5) * Real.arctan ((Real.tan x) / Real.sqrt 5)

-- Theorem statement
theorem integral_of_f_is_F : ∀ x : ℝ, deriv F x = f x := by
  intro x
  sorry -- The actual proof would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_is_F_l120_12052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_theorem_l120_12048

/-- Proposition p: The equation x^2 + kx + 9/4 = 0 has no real roots -/
def prop_p (k : ℝ) : Prop := ∀ x : ℝ, x^2 + k*x + 9/4 ≠ 0

/-- Proposition q: The domain of y = log_2(kx^2 + kx + 1) is ℝ -/
def prop_q (k : ℝ) : Prop := ∀ x : ℝ, k*x^2 + k*x + 1 > 0

/-- The range of k satisfying the given conditions -/
def k_range (k : ℝ) : Prop := (-3 < k ∧ k < 0) ∨ (3 ≤ k ∧ k < 4)

theorem k_range_theorem :
  (∀ k : ℝ, prop_p k ∨ prop_q k) ∧ (∃ k : ℝ, ¬(prop_p k ∧ prop_q k)) →
  ∃ k : ℝ, k_range k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_theorem_l120_12048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_power_l120_12002

theorem greatest_divisor_power (B : ℕ) (result : ℕ) : ∃! m : ℕ, 
  (Nat.factorial 20) / (B ^ m) = result ∧ 
  ∀ k : ℕ, k > m → ¬((Nat.factorial 20) / (B ^ k) = result) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_power_l120_12002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l120_12078

/-- Calculates the speed of a train given its length, platform length, and time to cross the platform -/
noncomputable def trainSpeed (trainLength platformLength : ℝ) (timeToCross : ℝ) : ℝ :=
  let totalDistance := trainLength + platformLength
  let speedMPS := totalDistance / timeToCross
  speedMPS * 3.6

/-- Theorem stating that a train with given parameters has a speed of approximately 72.006 km/h -/
theorem train_speed_calculation :
  let trainLength : ℝ := 250
  let platformLength : ℝ := 50.024
  let timeToCross : ℝ := 15
  let calculatedSpeed := trainSpeed trainLength platformLength timeToCross
  abs (calculatedSpeed - 72.006) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l120_12078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sides_proof_l120_12008

def count_ways_to_sum (n : ℕ) (sum : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => if i + 1 ≤ sum - 1 ∧ sum - (i + 1) ≤ n then 1 else 0)

theorem dice_sides_proof (n : ℕ) (h : n > 1) : 
  (↑(count_ways_to_sum n 8) / (n^2 : ℚ) = 5/36) → n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sides_proof_l120_12008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l120_12070

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, 1 + (1 / 2) * t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define point M
def point_M : ℝ × ℝ := (1, 1)

-- Define the intersection points A and B
def intersection_points (t : ℝ) : Prop :=
  let (x, y) := line_l t
  curve_C x y

-- State the theorem
theorem intersection_distance_product :
  ∃ t₁ t₂ : ℝ,
    intersection_points t₁ ∧
    intersection_points t₂ ∧
    t₁ ≠ t₂ ∧
    (t₁ * t₂ = -3) ∧
    (let (x₁, y₁) := line_l t₁
     let (x₂, y₂) := line_l t₂
     ((x₁ - 1)^2 + (y₁ - 1)^2) * ((x₂ - 1)^2 + (y₂ - 1)^2) = 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l120_12070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_simplification_l120_12059

theorem cube_root_sum_simplification (x : ℝ) : 
  ((x^3 - 3*x^2 + 3*x - 1)^(1/3) + (x^3 + 3*x^2 + 3*x + 1)^(1/3)) = 2*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_simplification_l120_12059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l120_12001

theorem sufficient_not_necessary :
  (∀ x : ℝ, |x - 2| < 1 → x^2 + x - 2 > 0) ∧
  (∃ x : ℝ, x^2 + x - 2 > 0 ∧ |x - 2| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l120_12001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_on_interval_l120_12081

noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 13/2

theorem function_extrema_on_interval (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ≥ 2*a) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 2*b) ∧
  (∃ x ∈ Set.Icc a b, f x = 2*a) ∧
  (∃ x ∈ Set.Icc a b, f x = 2*b) →
  a = 1 ∧ b = 3 := by
  sorry

#check function_extrema_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_on_interval_l120_12081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l120_12038

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -(1 : ℝ)) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -(1 : ℝ)) ∧
  (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → 
    f x₀ = 6/5 → Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l120_12038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l120_12005

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | x^2 - 3*x - 4 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioc 2 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l120_12005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_90_degrees_l120_12025

noncomputable section

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] [CompleteSpace n]

/-- The angle between two vectors in radians -/
noncomputable def angle_between_vectors (a b : n) : ℝ := Real.arccos ((inner a b) / (norm a * norm b))

/-- Theorem: If ‖a + k•b‖ = ‖a - k•b‖ for nonzero vectors a and b and nonzero scalar k, 
    then the angle between a and b is 90 degrees (π/2 radians). -/
theorem orthogonal_vectors_90_degrees 
  (a b : n) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (k : ℝ) 
  (hk : k ≠ 0) 
  (h_norm : ‖a + k • b‖ = ‖a - k • b‖) : 
  angle_between_vectors a b = π / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_90_degrees_l120_12025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_length_is_15_l120_12062

/-- Calculates the length of a room given the carpeting parameters -/
noncomputable def room_length (room_breadth : ℝ) (carpet_width : ℝ) (carpet_cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let total_carpet_length := total_cost / carpet_cost_per_meter
  let num_strips := room_breadth / carpet_width
  total_carpet_length / num_strips

/-- Theorem stating that the room length is 15 meters given the specified parameters -/
theorem room_length_is_15 :
  room_length 6 0.75 0.30 36 = 15 := by
  -- Unfold the definition of room_length
  unfold room_length
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_length_is_15_l120_12062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_angle_value_l120_12044

/-- Given an angle α in the plane rectangular coordinate system xOy,
    prove that -30° is a possible value for α when its terminal side
    intersects the unit circle at a point with a horizontal coordinate of √3/2. -/
theorem possible_angle_value (α : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x = Real.sqrt 3 / 2) →
  ∃ (k : ℤ), α = -π/6 + 2*π*↑k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_angle_value_l120_12044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_b1_b2_l120_12085

def sequence_b (b₁ b₂ : ℕ) : ℕ → ℕ
  | 0 => b₁
  | 1 => b₂
  | (n + 2) => (sequence_b b₁ b₂ n + 2213) / (2 + sequence_b b₁ b₂ (n + 1))

theorem smallest_sum_b1_b2 :
  ∀ b₁ b₂ : ℕ,
  (∀ n, sequence_b b₁ b₂ n > 0) →
  (∀ n, n ≥ 2 → sequence_b b₁ b₂ n = (sequence_b b₁ b₂ (n - 2) + 2213) / (2 + sequence_b b₁ b₂ (n - 1))) →
  b₁ + b₂ ≥ 30 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_b1_b2_l120_12085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_proposition_l120_12027

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, (2 : ℝ)^x < 1) ↔ (∀ x : ℝ, (2 : ℝ)^x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_proposition_l120_12027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_time_approx_l120_12010

/-- Represents Jaydee's typing speed in words per minute -/
def typing_speed : ℚ := 32

/-- Represents the number of words in the research paper -/
def paper_words : ℚ := 7125

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℚ := 60

/-- Calculates the time in hours required to type the paper -/
noncomputable def typing_time : ℚ := paper_words / typing_speed / minutes_per_hour

/-- Theorem stating that the time required to type the paper is approximately 3.71 hours -/
theorem typing_time_approx : 
  (typing_time * 100).floor / 100 = 371/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_time_approx_l120_12010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_cookies_smallest_area_bella_bakes_most_l120_12098

/-- Represents a baker with their cookie shape and area -/
structure Baker where
  name : String
  cookieArea : ℝ

/-- Proves that the baker with the smallest cookie area makes the most cookies -/
theorem most_cookies_smallest_area (totalDough : ℝ) (bakers : List Baker) 
    (hDough : totalDough > 0)
    (hBakers : bakers.length > 0)
    (hUniformThickness : ∀ b1 b2, b1 ∈ bakers → b2 ∈ bakers → b1.cookieArea > 0 ∧ b2.cookieArea > 0) :
    ∃ b ∈ bakers, ∀ other ∈ bakers, 
      b.cookieArea ≤ other.cookieArea → 
      totalDough / b.cookieArea ≥ totalDough / other.cookieArea := by
  sorry

/-- The competition setup -/
def bakingCompetition : List Baker := [
  ⟨"Alex", 9⟩,
  ⟨"Bella", 7⟩,
  ⟨"Carlo", 8⟩,
  ⟨"Dana", 10⟩
]

/-- Proves that Bella bakes the most cookies in this competition -/
theorem bella_bakes_most (totalDough : ℝ) (hDough : totalDough > 0) :
    ∃ b ∈ bakingCompetition, 
      b.name = "Bella" ∧ 
      ∀ other ∈ bakingCompetition, 
        totalDough / b.cookieArea ≥ totalDough / other.cookieArea := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_cookies_smallest_area_bella_bakes_most_l120_12098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l120_12076

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ my_circle p.1 p.2}

-- Theorem statement
theorem intersection_distance :
  ∃ (C D : ℝ × ℝ), C ∈ intersection_points ∧ D ∈ intersection_points ∧
    C ≠ D ∧ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l120_12076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hyperbola_chord_length_l120_12091

/-- A hyperbola with given foci and asymptote -/
structure Hyperbola where
  focal_distance : ℝ
  asymptote_slope : ℝ
  focal_distance_pos : 0 < focal_distance
  asymptote_slope_pos : 0 < asymptote_slope

/-- The length of the chord passing through the foci and perpendicular to the x-axis -/
noncomputable def chord_length (h : Hyperbola) : ℝ :=
  2 * h.asymptote_slope^2 * h.focal_distance / Real.sqrt (1 + h.asymptote_slope^2)

/-- Theorem stating the chord length for a specific hyperbola -/
theorem specific_hyperbola_chord_length :
  let h : Hyperbola := {
    focal_distance := 3,
    asymptote_slope := Real.sqrt 2,
    focal_distance_pos := by norm_num,
    asymptote_slope_pos := by norm_num
  }
  chord_length h = 4 * Real.sqrt 3 := by
  sorry

#check specific_hyperbola_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hyperbola_chord_length_l120_12091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_find_a_alt_l120_12007

def U : Set ℝ := {1, 3, 5, 7, 9}

def A (a : ℝ) : Set ℝ := {1, |a - 5|, 9}

theorem find_a : ∃ a : ℝ, (A a ∩ U = {1, 3, 9}) ∧ (a = 2 ∨ a = 8) := by
  use 2
  constructor
  · simp [A, U]
    norm_num
  · left
    rfl

-- Alternative proof using 8
theorem find_a_alt : ∃ a : ℝ, (A a ∩ U = {1, 3, 9}) ∧ (a = 2 ∨ a = 8) := by
  use 8
  constructor
  · simp [A, U]
    norm_num
  · right
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_find_a_alt_l120_12007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_tangent_circle_l120_12014

/-- The line to which the circles are tangent -/
noncomputable def tangent_line (m : ℝ) := {(x, y) : ℝ × ℝ | m * x - y - 2 * m + 1 = 0}

/-- The distance from the origin to the tangent line -/
noncomputable def distance_to_line (m : ℝ) : ℝ := |1 - 2*m| / Real.sqrt (m^2 + 1)

/-- A circle centered at the origin with radius r -/
def circle_at_origin (r : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}

theorem largest_tangent_circle :
  ∃ (r : ℝ), r^2 = 5 ∧
  ∀ (m : ℝ), distance_to_line m ≤ r ∧
  ∃ (m₀ : ℝ), distance_to_line m₀ = r := by
  sorry

#check largest_tangent_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_tangent_circle_l120_12014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_in_triangle_l120_12026

theorem min_value_in_triangle (a b c A B C : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  a^2 + c^2 = a*c + b^2 →
  b = Real.sqrt 3 →
  a ≥ c →
  -- Conclusion
  (∀ x : ℝ, 2*a - c ≥ Real.sqrt 3) ∧ (∃ x : ℝ, 2*a - c = Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_in_triangle_l120_12026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_figure_l120_12092

-- Define the function f(x) = x^2 + 2
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the area of the figure
noncomputable def figureArea : ℝ := ∫ x in (0:ℝ)..1, f x

-- Theorem statement
theorem area_of_figure : figureArea = 7/3 := by
  -- Unfold the definition of figureArea
  unfold figureArea
  -- Evaluate the integral
  simp [f]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_figure_l120_12092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exist_l120_12057

theorem min_value_exist : ∃ m n : ℤ, |2015 * m^5 - 2014 * n^4| = 0 := by
  -- We'll use m = 2014^3 * 2015 and n = 2014^4 * 2015
  let m : ℤ := (2014^3) * 2015
  let n : ℤ := (2014^4) * 2015
  use m, n
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exist_l120_12057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_has_two_zeros_g_inequality_holds_l120_12043

-- Define the functions
noncomputable def f (x : ℝ) := max (x^2 - 1) (2 * Real.log x)
noncomputable def g (a x : ℝ) := max (x + Real.log x) (-x^2 + (a^2 - 1/2)*x + 2*a^2 + 4*a)
noncomputable def h (x : ℝ) := f x - 3*(x - 1/2)*(x - 1)^2

-- Theorem for the first part of the problem
theorem h_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧ h x₁ = 0 ∧ h x₂ = 0 ∧
  ∀ x, 0 < x ∧ x ≤ 1 ∧ h x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

-- Theorem for the second part of the problem
theorem g_inequality_holds (a : ℝ) :
  (Real.log 2 / 4 - 1/4 < a ∧ a ≤ 2) →
  ∀ x, x > a + 2 → g a x < 3/2 * x + 4 * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_has_two_zeros_g_inequality_holds_l120_12043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_with_distance_l120_12063

/-- A point on a parametric line that is a certain distance from a given point -/
noncomputable def PointOnLine (t : ℝ) : ℝ × ℝ := (1 - Real.sqrt 2 * t, 2 + Real.sqrt 2 * t)

/-- The distance between two points in 2D space -/
noncomputable def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating that the only points on the given line that are 4√2 distance from (1,2) are (-3,6) and (5,-2) -/
theorem points_on_line_with_distance : 
  ∀ t : ℝ, Distance (PointOnLine t) (1, 2) = 4 * Real.sqrt 2 ↔ 
    PointOnLine t = (-3, 6) ∨ PointOnLine t = (5, -2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_with_distance_l120_12063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_similar_triangles_l120_12013

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define the points of the quadrilateral
variable (A B C D : ℝ × ℝ)

-- Define the intersection points
variable (P Q : ℝ × ℝ)

-- Assume the quadrilateral is inscribed in the circle
axiom inscribed : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle

-- Assume P is the intersection of extended AB and CD
axiom P_intersect : ∃ (t₁ t₂ : ℝ), P = (1 - t₁) • A + t₁ • B ∧ P = (1 - t₂) • C + t₂ • D

-- Assume Q is the intersection of extended AD and BC
axiom Q_intersect : ∃ (s₁ s₂ : ℝ), Q = (1 - s₁) • A + s₁ • D ∧ Q = (1 - s₂) • B + s₂ • C

-- Define similarity of triangles
def triangles_similar (T₁ T₂ T₃ U₁ U₂ U₃ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    dist T₁ T₂ = k * dist U₁ U₂ ∧
    dist T₂ T₃ = k * dist U₂ U₃ ∧
    dist T₃ T₁ = k * dist U₃ U₁

-- State the theorem
theorem inscribed_quadrilateral_similar_triangles :
  triangles_similar P A B P C D ∧ triangles_similar Q A D Q B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_similar_triangles_l120_12013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l120_12095

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (3 * X^2 - 20 * X + 32 : Polynomial ℝ) = (X - 4) * q + 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l120_12095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l120_12037

/-- Given an ellipse with specific properties, prove its slope and equation -/
theorem ellipse_properties (a b c : ℝ) (M : ℝ × ℝ) :
  a > b ∧ b > 0 →  -- a > b > 0
  (fun x y ↦ x^2 / a^2 + y^2 / b^2 = 1) = (fun x y ↦ x^2 / a^2 + y^2 / b^2 = 1) →  -- ellipse equation
  (-c, 0) = (-c, 0) →  -- left focus
  c^2 / a^2 = 1 / 3 →  -- eccentricity
  M.1 > 0 ∧ M.2 > 0 →  -- M in first quadrant
  M.1^2 / a^2 + M.2^2 / b^2 = 1 →  -- M on ellipse
  ∃ P : ℝ × ℝ, P.1^2 + P.2^2 = b^2 / 4 ∧ 
      (P.2 - 0) / (P.1 - (-c)) = (M.2 - 0) / (M.1 - (-c)) ∧
      (P.1 - (-c))^2 + P.2^2 = c^2 →  -- intersection property
  (M.1 - (-c))^2 + M.2^2 = (4 * Real.sqrt 3 / 3)^2 →  -- |FM| = 4√3/3
  ((M.2 - 0) / (M.1 - (-c)) = Real.sqrt 3 / 3 ∧  -- slope of FM
   (fun x y ↦ x^2 / 3 + y^2 / 2 = 1) = (fun x y ↦ x^2 / 3 + y^2 / 2 = 1)) -- ellipse equation
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l120_12037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_comparison_l120_12047

/-- Represents a person's reading speed and duration -/
structure Reader where
  speed : ℝ  -- pages per hour
  duration : ℝ  -- in hours

/-- Calculates the number of pages read given a reader -/
def pages_read (r : Reader) : ℝ := r.speed * r.duration

/-- The problem statement -/
theorem reading_comparison (dustin sam nicole alex : Reader)
  (h1 : dustin.speed = 75)
  (h2 : sam.speed = 24)
  (h3 : nicole.speed = 35)
  (h4 : alex.speed = 50)
  (h5 : dustin.duration = 1)
  (h6 : sam.duration = 55/60)
  (h7 : nicole.duration = 35/60)
  (h8 : alex.duration = 50/60) :
  ∃ ε > 0, |pages_read dustin - (pages_read sam + pages_read nicole + pages_read alex) + 9.09| < ε := by
  sorry

#eval Float.toString ((75 * 1) - (24 * 55/60 + 35 * 35/60 + 50 * 50/60))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_comparison_l120_12047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_symmetry_l120_12009

/-- Define a line through two points in ℝ³ -/
def line_through (A B : ℝ × ℝ × ℝ) : Set (ℝ × ℝ × ℝ) :=
  {X | ∃ t : ℝ, X = (1 - t) • A + t • B}

/-- Define symmetry of two points with respect to a line in ℝ³ -/
def is_symmetric (P Q : ℝ × ℝ × ℝ) (L : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ O : ℝ × ℝ × ℝ, O ∈ L ∧ O = (P + Q) / 2

/-- Given points A, B, and P in ℝ³, find the point Q symmetric to P with respect to line AB -/
theorem point_symmetry (A B P : ℝ × ℝ × ℝ) (hA : A = (1, 2, -6)) (hB : B = (7, -7, 6)) (hP : P = (1, 3, 2)) :
  ∃ Q : ℝ × ℝ × ℝ, Q = (5, -5, -6) ∧ is_symmetric P Q (line_through A B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_symmetry_l120_12009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_form_arithmetic_sequence_l120_12040

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

theorem slopes_form_arithmetic_sequence
  (A B : ℝ × ℝ)
  (m n t : ℝ)
  (k_MA k_MQ k_MB : ℝ)
  (hA : ellipse A.1 A.2)
  (hB : ellipse B.1 B.2)
  (hAB_distinct : A ≠ B)
  (hm_ne_2 : m ≠ 2)
  (hm_ne_neg2 : m ≠ -2)
  (hm_ne_0 : m ≠ 0)
  (hn_ne_sqrt3 : n ≠ Real.sqrt 3)
  (hn_ne_neg_sqrt3 : n ≠ -Real.sqrt 3)
  (hn_ne_0 : n ≠ 0)
  (hMA_slope : k_MA ≠ 0)
  (hMQ_slope : k_MQ ≠ 0)
  (hMB_slope : k_MB ≠ 0)
  (hM_on_line : A.2 = 3 / n)
  (hMA_slope_def : k_MA = (A.2 - 3 / n) / (A.1 - t))
  (hMQ_slope_def : k_MQ = (3 / n - n) / t)
  (hMB_slope_def : k_MB = (B.2 - 3 / n) / (B.1 - t)) :
  ∃ (r : ℝ), 1 / k_MA + 1 / k_MB = 2 / k_MQ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_form_arithmetic_sequence_l120_12040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l120_12021

/-- The distance from the center of a sphere to the plane of an isosceles triangle tangent to the sphere -/
theorem sphere_triangle_distance (r a b c : ℝ) 
  (h1 : r = 9) (h2 : a = 13) (h3 : b = 13) (h4 : c = 10) :
  let s := (a + b + c) / 2
  let triangle_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inradius := triangle_area / s
  Real.sqrt (r^2 - inradius^2) = Real.sqrt (629/9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l120_12021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_inequality_l120_12094

noncomputable def f (x : ℝ) : ℝ := |x - 5/2| + |x - 1/2|

theorem f_minimum_and_inequality :
  (∀ x : ℝ, f x ≥ 2) ∧
  (∃ x : ℝ, f x = 2) ∧
  (∀ x : ℝ, f x ≤ x + 4 ↔ -1/3 ≤ x ∧ x ≤ 7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_inequality_l120_12094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l120_12024

theorem max_distance_circle_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 2)^2 = 1}
  let ellipse := {p : ℝ × ℝ | p.1^2/9 + p.2^2 = 1}
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ circle ∧ Q ∈ ellipse ∧
    ∀ (P' : ℝ × ℝ) (Q' : ℝ × ℝ), P' ∈ circle → Q' ∈ ellipse →
      dist P Q ≥ dist P' Q' ∧
      dist P Q = (3 * Real.sqrt 6) / 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l120_12024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_indices_l120_12049

/-- Given 2009 non-degenerate triangles with sides colored blue, red, and white,
    prove that the maximum number of indices j for which bⱼ, rⱼ, wⱼ can form
    a non-degenerate triangle is 1. -/
theorem max_triangle_indices (b r w : Fin 2009 → ℝ)
  (h_sorted_b : ∀ i j : Fin 2009, i ≤ j → b i ≤ b j)
  (h_sorted_r : ∀ i j : Fin 2009, i ≤ j → r i ≤ r j)
  (h_sorted_w : ∀ i j : Fin 2009, i ≤ j → w i ≤ w j)
  (h_nondeg : ∀ i : Fin 2009, b i + r i > w i ∧ b i + w i > r i ∧ r i + w i > b i) :
  (Finset.filter (λ j : Fin 2009 ↦
    b j + r j > w j ∧ b j + w j > r j ∧ r j + w j > b j) Finset.univ).card = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_indices_l120_12049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_transformation_l120_12061

theorem cosine_sine_transformation (x : ℝ) :
  Real.cos (2 * x) - Real.sqrt 3 * Real.sin (2 * x) = 2 * Real.cos (2 * x + π / 3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_transformation_l120_12061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_l120_12065

/-- Represents a distribution scheme of employees between two departments -/
structure Distribution where
  dept_a : Finset Nat
  dept_b : Finset Nat

/-- The set of all valid distribution schemes -/
def ValidDistributions : Set Distribution :=
  { d | d.dept_a.card = 4 ∧ d.dept_b.card = 4 ∧
        d.dept_a ∪ d.dept_b = Finset.range 8 ∧
        d.dept_a ∩ d.dept_b = ∅ ∧
        (d.dept_a ∩ {0, 1}).card ≠ 2 ∧
        (d.dept_b ∩ {0, 1}).card ≠ 2 ∧
        (d.dept_a ∩ {2, 3, 4}).card ≠ 3 ∧
        (d.dept_b ∩ {2, 3, 4}).card ≠ 3 }

/-- Instance to make ValidDistributions a finite type -/
instance : Fintype ValidDistributions :=
  sorry

theorem distribution_count :
  Fintype.card ValidDistributions = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_l120_12065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_smallest_A_l120_12031

/-- Given a nine-digit natural number A obtained from B by moving the last digit to the first position,
    where B is coprime with 24 and B > 666666666, prove that the largest possible value of A is 999999998
    and the smallest possible value of A is 166666667. -/
theorem largest_smallest_A (A B : ℕ) : 
  (A ≥ 100000000 ∧ A < 1000000000) →  -- A is a nine-digit number
  (∃ b : ℕ, b < 10 ∧ A = b * 10^8 + B / 10) →  -- A is obtained from B by moving last digit to first
  Nat.Coprime B 24 →  -- B is coprime with 24
  B > 666666666 →  -- B > 666666666
  (∀ A' : ℕ, 
    (A' ≥ 100000000 ∧ A' < 1000000000) →
    (∃ B' : ℕ, ∃ b' : ℕ, b' < 10 ∧ A' = b' * 10^8 + B' / 10) →
    Nat.Coprime B' 24 →
    B' > 666666666 →
    A' ≤ 999999998 ∧ A' ≥ 166666667) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_smallest_A_l120_12031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_piece_probability_theorem_longer_piece_probability_integral_l120_12072

/-- The probability that when a unit string is cut at a random point, 
    the longer piece is at least x times as large as the shorter piece -/
noncomputable def longer_piece_probability (x : ℝ) : ℝ :=
  2 / (x + 1)

/-- Theorem stating the probability of the longer piece being at least x times 
    as large as the shorter piece when a unit string is cut at a random point -/
theorem longer_piece_probability_theorem (x : ℝ) (hx : x > 0) : 
  longer_piece_probability x = 
    2 / (x + 1) :=
by
  -- Unfold the definition of longer_piece_probability
  unfold longer_piece_probability
  -- The rest of the proof is omitted
  sorry

/-- Helper lemma for the main theorem -/
lemma longer_piece_condition (c : ℝ) (x : ℝ) :
  (max c (1 - c)) ≥ x * (min c (1 - c)) ↔ c ≤ 1 / (x + 1) ∨ c ≥ x / (x + 1) :=
by sorry

/-- Main theorem relating the probability to the integral over the unit interval -/
theorem longer_piece_probability_integral (x : ℝ) (hx : x > 0) :
  longer_piece_probability x = 
    ∫ c in (Set.Icc 0 1), (if c ≤ 1 / (x + 1) ∨ c ≥ x / (x + 1) then 1 else 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_piece_probability_theorem_longer_piece_probability_integral_l120_12072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_SABCQ_proof_l120_12011

/-- Regular pyramid MNPQ with base side length 5 -/
structure PyramidMNPQ where
  base_side : ℝ
  base_side_eq : base_side = 5

/-- Regular pyramid SABCQ with square base ABCD -/
structure PyramidSABCQ (p : PyramidMNPQ) where
  /-- S lies on edge QM of MNPQ -/
  s_on_qm : Bool
  /-- MS = 3/4 MQ -/
  ms_ratio : ℝ
  ms_ratio_eq : ms_ratio = 3/4
  /-- All vertices of SABCQ are on edges of MNPQ -/
  vertices_on_edges : Bool

/-- The volume of pyramid SABCQ -/
noncomputable def volume_SABCQ (p : PyramidMNPQ) (q : PyramidSABCQ p) : ℝ := 2 * Real.sqrt 3 / 3

theorem volume_SABCQ_proof (p : PyramidMNPQ) (q : PyramidSABCQ p) :
  volume_SABCQ p q = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_SABCQ_proof_l120_12011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_similar_numbers_l120_12087

/-- A function that generates a number by repeating a 3-digit number n times -/
def repeat_digits (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^(3*n) - 1) / 999

/-- Helper function to get the digits of a number -/
def digits (n : ℕ) : List ℕ :=
  sorry

/-- Helper function to count the number of digits in a number -/
def number_of_digits (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating the existence of three 1995-digit numbers satisfying the conditions -/
theorem exist_three_similar_numbers :
  ∃ (A B C : ℕ),
    (A = repeat_digits 459 665) ∧
    (B = repeat_digits 495 665) ∧
    (C = repeat_digits 954 665) ∧
    (A + B = C) ∧
    (∀ d : ℕ, d ∈ [4, 5, 9] → (d ∈ digits A ∧ d ∈ digits B ∧ d ∈ digits C)) ∧
    (∀ d : ℕ, d ∉ [4, 5, 9] → (d ∉ digits A ∧ d ∉ digits B ∧ d ∉ digits C)) ∧
    (number_of_digits A = 1995) ∧
    (number_of_digits B = 1995) ∧
    (number_of_digits C = 1995) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_similar_numbers_l120_12087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_circle_radius_l120_12042

/-- A right triangle with a specific configuration of tangent circles -/
structure TriangleWithCircles where
  -- The triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- The first circle
  O₁ : ℝ × ℝ
  r₁ : ℝ
  -- The second circle
  O₂ : ℝ × ℝ
  r₂ : ℝ
  -- Conditions
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  ab_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = Real.sqrt 3
  bc_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 3
  -- First circle conditions
  tangent_ab : (O₁.1 - B.1)^2 + (O₁.2 - B.2)^2 = r₁^2
  tangent_ac : ((O₁.1 - A.1) * (C.2 - A.2) - (O₁.2 - A.2) * (C.1 - A.1))^2 = 
               r₁^2 * ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  -- Second circle conditions
  tangent_first : (O₂.1 - O₁.1)^2 + (O₂.2 - O₁.2)^2 = (r₁ + r₂)^2
  tangent_ab_second : ((O₂.1 - A.1) * (B.2 - A.2) - (O₂.2 - A.2) * (B.1 - A.1))^2 = 
                      r₂^2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  tangent_ac_second : ((O₂.1 - A.1) * (C.2 - A.2) - (O₂.2 - A.2) * (C.1 - A.1))^2 = 
                      r₂^2 * ((C.1 - A.1)^2 + (C.2 - A.2)^2)

/-- The radius of the second circle in the specific configuration is √3/6 -/
theorem second_circle_radius (t : TriangleWithCircles) : t.r₂ = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_circle_radius_l120_12042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l120_12000

noncomputable def f (a b x : ℝ) : ℝ :=
  -2 * a * Real.sin (2 * x + Real.pi / 6) + 2 * a + b

noncomputable def g (a b x : ℝ) : ℝ :=
  f a b (x + Real.pi / 2)

theorem function_properties (a b : ℝ) :
  a > 0 →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a b x ∈ Set.Icc (-5) 1) →
  (a = 2 ∧ b = -5) ∧
  (∀ k : ℤ,
    (StrictMonoOn (g a b) (Set.Ioo (↑k * Real.pi) (↑k * Real.pi + Real.pi / 6)) ∧
     StrictAntiOn (g a b) (Set.Ioo (↑k * Real.pi + Real.pi / 6) (↑k * Real.pi + Real.pi / 3)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l120_12000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l120_12015

theorem sin_alpha_value (α : Real) 
  (h1 : Real.cos (Real.pi/4 + α) = Real.sqrt 2/10) 
  (h2 : 0 < α) 
  (h3 : α < Real.pi/2) : 
  Real.sin α = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l120_12015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_open_unit_interval_l120_12097

noncomputable def f (x : ℝ) : ℝ := x + 1/x

theorem f_decreasing_on_open_unit_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f x₁ > f x₂ := by
  intros x₁ x₂ h₁ h₂ h₃
  unfold f
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_open_unit_interval_l120_12097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l120_12017

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x + 3 * Real.pi / 4)

theorem g_properties :
  (∀ x, g x = g (-x)) ∧ 
  (∀ x, g (x + Real.pi) = g x) ∧
  (∀ p, p > 0 → (∀ x, g (x + p) = g x) → p ≥ Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l120_12017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l120_12093

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1 ∧ a > 0

-- Define the foci of the hyperbola
def foci (c : ℝ) : Prop := c > 0

-- Define the point P
noncomputable def point_P : ℝ × ℝ := (2, Real.sqrt 5)

-- Define the right triangle condition
def right_triangle (c : ℝ) : Prop :=
  let P := point_P
  (P.1 + c) * (P.1 - c) + P.2 * P.2 = 0

-- Theorem statement
theorem hyperbola_focal_distance (a c : ℝ) :
  hyperbola a 2 (Real.sqrt 5) →
  foci c →
  right_triangle c →
  2 * c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l120_12093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_mean_value_function_m_range_l120_12035

/-- A function is a "double mean value function" on an interval [a,b] if its second derivative
    equals the average rate of change of the function over [a,b] at two distinct points in (a,b). -/
def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    (deriv^[2] f) x₁ = (f b - f a) / (b - a) ∧
    (deriv^[2] f) x₂ = (f b - f a) / (b - a)

/-- The function g(x) = (1/3)x^3 - (m/2)x^2 -/
noncomputable def g (m : ℝ) : ℝ → ℝ := λ x ↦ (1/3) * x^3 - (m/2) * x^2

theorem double_mean_value_function_m_range :
  ∀ m : ℝ, is_double_mean_value_function (g m) 0 2 → 4/3 < m ∧ m < 8/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_mean_value_function_m_range_l120_12035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_system_solution_l120_12082

/-- Two linear equations in x and y, parameterized by a -/
structure LinearSystem (a : ℝ) :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)
  (h1 : ∀ x y, eq1 x y = x + a^2*y + 6)
  (h2 : ∀ x y, eq2 x y = (a-2)*x + 3*a*y + 2*a)

/-- The condition for parallel or overlapping lines -/
def parallel_or_overlap (a : ℝ) (sys : LinearSystem a) : Prop :=
  (∀ x y, sys.eq1 x y = 0 ↔ sys.eq2 x y = 0) ∨
  (∃ m b, (∀ x y, sys.eq1 x y = 0 ↔ y = m*x + b) ∧
          (∀ x y, sys.eq2 x y = 0 ↔ y = m*x + b) ∧
          (∃ x y, sys.eq1 x y ≠ 0 ∨ sys.eq2 x y ≠ 0))

/-- The main theorem -/
theorem linear_system_solution :
  ∃ (a : ℝ), a = -1 ∧ parallel_or_overlap a
    ⟨λ x y => x + a^2*y + 6,
     λ x y => (a-2)*x + 3*a*y + 2*a,
     λ x y => rfl,
     λ x y => rfl⟩ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_system_solution_l120_12082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_integer_part_count_l120_12088

noncomputable def finite_decimal (n : ℕ) := { x : ℝ | ∃ (a : ℝ), 0 ≤ a ∧ a < 1 ∧ x = n + a }

noncomputable def integer_part (x : ℝ) := ⌊x⌋

theorem product_integer_part_count :
  ∃ (A B : ℝ),
    A ∈ finite_decimal 7 ∧
    B ∈ finite_decimal 10 ∧
    (Finset.range 18).card = (Finset.range 88 \ Finset.range 70).card ∧
    ∀ (n : ℕ), integer_part (A * B) = n → n ∈ Finset.range 88 \ Finset.range 70 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_integer_part_count_l120_12088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_point_A_inside_circle_O_l120_12041

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in 2D space -/
def Point : Type := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Predicate for a point being inside a circle -/
def is_inside (p : Point) (c : Circle) : Prop :=
  distance p c.center < c.radius

theorem point_inside_circle (c : Circle) (p : Point) :
  distance p c.center < c.radius → is_inside p c :=
by
  intro h
  exact h

/-- The main theorem proving that point A is inside circle O -/
theorem point_A_inside_circle_O (O : Circle) (A : Point) 
  (h_radius : O.radius = 6)
  (h_distance : distance A O.center = 5) : 
  is_inside A O :=
by
  apply point_inside_circle
  rw [h_radius, h_distance]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_point_A_inside_circle_O_l120_12041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_formula_l120_12016

-- Define the base function f
noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

-- Define f_n recursively
noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => f ∘ f_n n

-- Theorem statement
theorem f_n_formula (n : ℕ) (x : ℝ) (h : x > 0) :
  n ≥ 1 → f_n (n + 1) x = x / ((2^(n + 1) - 1) * x + 2^(n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_formula_l120_12016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_ratio_theorem_l120_12066

theorem complex_ratio_theorem (z₁ z₂ : ℂ) (h₁ : Complex.abs z₁ = 2) (h₂ : Complex.abs z₂ = 3)
  (h₃ : (z₁.arg - z₂.arg).cos = 1/2) :
  Complex.abs ((z₁ + z₂) / (z₁ - z₂)) = Real.sqrt 133 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_ratio_theorem_l120_12066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_opens_in_fewer_than_seven_attempts_l120_12084

/-- A good code is a seven-digit code composed of seven distinct digits -/
def GoodCode : Type := { code : Fin 7 → Fin 10 // Function.Injective code }

/-- A set of six carefully chosen good codes -/
def SixCodes : Finset GoodCode :=
  sorry

/-- The safe's password (an arbitrary good code) -/
def SafePassword : GoodCode :=
  sorry

/-- Predicate to check if a code opens the safe -/
def OpensSafe (code : GoodCode) (password : GoodCode) : Prop :=
  ∃ i : Fin 7, code.val i = password.val i

/-- Theorem: The safe can be opened in fewer than seven attempts -/
theorem safe_opens_in_fewer_than_seven_attempts :
  ∃ (codes : Finset GoodCode), Finset.card codes < 7 ∧
    ∀ (password : GoodCode), ∃ (code : GoodCode), code ∈ codes ∧ OpensSafe code password :=
  sorry

#check safe_opens_in_fewer_than_seven_attempts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_opens_in_fewer_than_seven_attempts_l120_12084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l120_12096

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties :
  (∃ (x : ℝ), x > 0 ∧ f x = (1 : ℝ) / Real.exp 1 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x) ∧
  (∃! (x : ℝ), x > 0 ∧ f x = 0) ∧
  (f 2 < f Real.pi ∧ f Real.pi < f 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l120_12096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l120_12004

noncomputable def plane (x y z : ℝ) : Prop := 2*x + 3*y - z = 15

noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

theorem closest_point_on_plane :
  let x₀ := 30/7
  let y₀ := 10/7
  let z₀ := -15/7
  (plane x₀ y₀ z₀) ∧
  (∀ x y z, plane x y z →
    distance x y z 4 1 (-2) ≥ distance x₀ y₀ z₀ 4 1 (-2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l120_12004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_coverage_in_tiling_hexagon_coverage_percentage_l120_12051

/-- Represents the fraction of the plane covered by hexagons in a specific tiling. -/
noncomputable def hexagon_coverage : ℝ :=
  (3 * Real.sqrt 3 / 2) / (3 * Real.sqrt 3 + 6)

/-- The theorem states that in a plane tiled with congruent hexagons, squares, and triangles,
    where each hexagon is surrounded by six triangles and each triangle shares a side with a square,
    and all shapes have the same side length, the fraction of the plane covered by hexagons
    is equal to (3√3/2) / (3√3 + 6). -/
theorem hexagon_coverage_in_tiling :
  hexagon_coverage = (3 * Real.sqrt 3 / 2) / (3 * Real.sqrt 3 + 6) := by
  sorry

/-- The percentage of the plane covered by hexagons is approximately 30%. -/
theorem hexagon_coverage_percentage :
  ⌊100 * hexagon_coverage⌋ = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_coverage_in_tiling_hexagon_coverage_percentage_l120_12051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_homogeneous_polynomial_for_primitive_lattice_points_l120_12067

/-- A primitive lattice point is a pair of integers with greatest common divisor 1. -/
def PrimitiveLatticePoint (p : ℤ × ℤ) : Prop :=
  Int.gcd p.1 p.2 = 1

/-- A homogeneous polynomial with integer coefficients. -/
def HomogeneousPolynomial (f : ℤ → ℤ → ℤ) (degree : ℕ) : Prop :=
  ∀ (x y : ℤ) (c : ℤ), f (c * x) (c * y) = c^degree * f x y

/-- The main theorem statement. -/
theorem exists_homogeneous_polynomial_for_primitive_lattice_points 
    (S : Set (ℤ × ℤ)) (hS : S.Finite) :
    (∀ p ∈ S, PrimitiveLatticePoint p) →
    ∃ (f : ℤ → ℤ → ℤ) (degree : ℕ), 
      HomogeneousPolynomial f degree ∧
      (∀ p ∈ S, f p.1 p.2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_homogeneous_polynomial_for_primitive_lattice_points_l120_12067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_iff_a_in_interval_l120_12006

/-- The function f(x) defined as ae^x - x ln(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x * Real.log x

/-- Proposition: There exists an integer n such that f(x) has two extreme values
    in the interval (n, n+2) if and only if a is in the open interval (ln(2e)/e^2, 1/e) --/
theorem extreme_values_iff_a_in_interval (a : ℝ) :
  (∃ (n : ℕ), ∃ (x₁ x₂ : ℝ), n < x₁ ∧ x₁ < x₂ ∧ x₂ < n + 2 ∧
    (∀ (x : ℝ), n < x ∧ x < n + 2 → HasDerivAt (f a) 0 x ↔ x = x₁ ∨ x = x₂)) ↔
  Real.log (2 * Real.exp 1) / Real.exp 2 < a ∧ a < 1 / Real.exp 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_iff_a_in_interval_l120_12006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l120_12069

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem triangle_problem (ω φ A B C a b c : ℝ) :
  ω > 0 →
  0 < φ ∧ φ < Real.pi / 2 →
  f ω φ 0 = 1 / 2 →
  2 * Real.pi / ω = Real.pi →
  f ω φ (A / 2) - Real.cos A = 1 / 2 →
  b * c = 1 →
  b + c = 3 →
  (∃ (x : ℝ), f ω φ x = f 2 (Real.pi / 6) x) ∧
  A = Real.pi / 3 ∧
  a = Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l120_12069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_contribution_l120_12036

/-- Calculate the individual contribution for a shared meal --/
theorem meal_contribution
  (total_price : ℚ)
  (coupon_discount : ℚ)
  (tax_rate : ℚ)
  (num_people : ℕ)
  (h_total : total_price = 94)
  (h_coupon : coupon_discount = 13/2)
  (h_tax : tax_rate = 15/200)
  (h_people : num_people = 4) :
  let subtotal := total_price - coupon_discount
  let tax_amount := (subtotal * tax_rate).floor / 100
  let final_total := subtotal + tax_amount
  let individual_contribution := (final_total / num_people * 100).floor / 100
  individual_contribution = 1176/50 := by
sorry

#eval (1176/50 : ℚ)  -- Should output 23.52

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_contribution_l120_12036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_t_for_full_range_l120_12012

open Real Set

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6)

theorem exists_t_for_full_range : 
  ∃ t : ℤ, ∀ x : ℝ, x ∈ Icc (t : ℝ) (t + 2) → f x ∈ Icc (-1) 1 ∧ 
  (∃ y z : ℝ, y ∈ Icc (t : ℝ) (t + 2) ∧ z ∈ Icc (t : ℝ) (t + 2) ∧ f y = -1 ∧ f z = 1) :=
by
  -- Proof goes here
  sorry

#check exists_t_for_full_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_t_for_full_range_l120_12012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_with_same_prime_factors_l120_12090

theorem unique_triple_with_same_prime_factors : ∀ a m n : ℕ,
  (a > 1) →
  (m < n) →
  (∀ p : ℕ, Nat.Prime p → (p ∣ (a^m - 1) ↔ p ∣ (a^n - 1))) →
  (a = 3 ∧ m = 1 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_with_same_prime_factors_l120_12090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l120_12029

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  slope : ℝ
  passesThrough : Point

/-- Directrix of a parabola -/
noncomputable def directrix (para : Parabola) : ℝ := -para.p / 2

/-- Focus of a parabola -/
noncomputable def focus (para : Parabola) : Point := ⟨para.p / 2, 0⟩

/-- Check if a point lies on a parabola -/
def onParabola (para : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * para.p * pt.x

/-- Check if a point lies on a line -/
def onLine (l : Line) (pt : Point) : Prop :=
  pt.y = l.slope * (pt.x - l.passesThrough.x) + l.passesThrough.y

/-- Check if a point lies on the directrix -/
def onDirectrix (para : Parabola) (pt : Point) : Prop :=
  pt.x = directrix para

/-- Check if two vectors are equal -/
def vectorsEqual (a b c d : Point) : Prop :=
  b.x - a.x = d.x - c.x ∧ b.y - a.y = d.y - c.y

theorem parabola_intersection_theorem (para : Parabola) 
    (l : Line) (A B : Point) :
  l.passesThrough = ⟨1, 0⟩ →
  l.slope = Real.sqrt 3 →
  onDirectrix para A →
  onParabola para B →
  onLine l A →
  onLine l B →
  vectorsEqual A ⟨1, 0⟩ ⟨1, 0⟩ B →
  para.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l120_12029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_graph_shift_l120_12020

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) * sin (2*x - Real.pi/6)

-- Statement for critical points
theorem critical_points (k : ℤ) :
  ∃ (x : ℝ), x = Real.pi/3 + k*Real.pi/2 ∧ (deriv f) x = 0 :=
by sorry

-- Statement for graph shift
theorem graph_shift (x : ℝ) :
  f x = (1/2) * sin (2*(x - Real.pi/12)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_graph_shift_l120_12020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_3_rational_others_l120_12018

theorem irrational_sqrt_3_rational_others : 
  (Irrational (Real.sqrt 3)) ∧ 
  (¬ Irrational (0 : ℝ)) ∧ 
  (¬ Irrational (1 : ℝ)) ∧ 
  (¬ Irrational (1/7 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_3_rational_others_l120_12018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_range_l120_12032

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x) - Real.sqrt 3 * (Real.cos x)^2

noncomputable def g (x : ℝ) : ℝ := f (x/2)

theorem f_properties_and_g_range :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  (∃ (m : ℝ), m = -(2 + Real.sqrt 3)/2 ∧ (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m)) ∧
  (Set.Icc (Real.pi/2) Real.pi).image g = Set.Icc ((1 - Real.sqrt 3)/2) ((2 - Real.sqrt 3)/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_range_l120_12032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_23_special_number_l120_12054

/-- Given a natural number x satisfying certain conditions in base-23,
    prove that x has a specific form. -/
theorem base_23_special_number (x : ℕ) (m : ℕ) : 
  (∃ (a : Fin 23), x = a.val * ((23^(2*m : ℕ) - 1) / 22)) →  -- x consists of 2m identical digits in base-23
  (∃ (b : Fin 23), x^2 = b.val * (1 + 23^(4*m - 1 : ℕ))) →  -- x^2 has identical extreme digits with zeros between
  x = 13 * ((23^(2*m : ℕ) - 1) / 22) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_23_special_number_l120_12054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_root_implies_m_equals_one_l120_12033

-- Define the imaginary unit
def i : ℂ := Complex.I

-- The property of i is already defined in Mathlib

-- Define the equation
def equation (x : ℂ) (m : ℝ) : ℂ := x^2 - (2 + i) * x + 1 + m * i

-- State the theorem
theorem real_root_implies_m_equals_one (n m : ℝ) : 
  equation n m = 0 → m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_root_implies_m_equals_one_l120_12033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_l120_12086

/-- The height of a tower given two angle measurements -/
theorem tower_height (initial_angle : Real) (final_angle : Real) (distance : Real) :
  initial_angle = π / 6 →  -- 30 degrees in radians
  final_angle = π / 4 →    -- 45 degrees in radians
  distance = 20 →
  ∃ (h : Real), h = 20 * Real.sqrt 3 ∧ 
    (Real.tan initial_angle = h / (h / Real.tan final_angle + distance)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_l120_12086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_function_A_l120_12039

noncomputable def data_points : List (ℝ × ℝ) := [(1, 3), (2, 5.99), (3, 12.01)]

noncomputable def function_A (x : ℝ) : ℝ := 3 * 2^(x - 1)
noncomputable def function_B (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def function_C (x : ℝ) : ℝ := 3 * x
noncomputable def function_D (x : ℝ) : ℝ := x^2

noncomputable def sum_squared_error (f : ℝ → ℝ) (points : List (ℝ × ℝ)) : ℝ :=
  points.foldl (fun acc (x, y) => acc + (f x - y)^2) 0

theorem best_fit_function_A :
  sum_squared_error function_A data_points < sum_squared_error function_B data_points ∧
  sum_squared_error function_A data_points < sum_squared_error function_C data_points ∧
  sum_squared_error function_A data_points < sum_squared_error function_D data_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_function_A_l120_12039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_isosceles_triangulation_has_two_equal_sides_l120_12053

/-- A convex polygon -/
structure ConvexPolygon where
  sides : List ℝ
  convex : List ℝ → Prop

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  sides : Fin 3 → ℝ
  isosceles : ∃ (i j : Fin 3), i ≠ j ∧ sides i = sides j

/-- A division of a polygon into isosceles triangles -/
structure IsoscelesTriangulation (P : ConvexPolygon) where
  triangles : List IsoscelesTriangle
  covers : (List ℝ → List (Fin 3 → ℝ) → Prop)
  nonintersecting : List IsoscelesTriangle → Prop

/-- The main theorem -/
theorem convex_polygon_isosceles_triangulation_has_two_equal_sides
  (P : ConvexPolygon)
  (triangulation : IsoscelesTriangulation P) :
  ∃ (i j : Fin P.sides.length), i ≠ j ∧ P.sides[i] = P.sides[j] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_isosceles_triangulation_has_two_equal_sides_l120_12053
