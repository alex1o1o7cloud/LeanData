import Mathlib

namespace NUMINAMATH_CALUDE_line_equation_from_intercept_and_slope_l3617_361750

/-- A line with x-intercept a and slope m -/
structure Line where
  a : ℝ  -- x-intercept
  m : ℝ  -- slope

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line with x-intercept 2 and slope 1, its equation is x - y - 2 = 0 -/
theorem line_equation_from_intercept_and_slope :
  ∀ (L : Line), L.a = 2 ∧ L.m = 1 →
  ∃ (eq : LineEquation), eq.a = 1 ∧ eq.b = -1 ∧ eq.c = -2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_intercept_and_slope_l3617_361750


namespace NUMINAMATH_CALUDE_tan_negative_210_degrees_l3617_361700

theorem tan_negative_210_degrees : 
  Real.tan (-(210 * π / 180)) = -(Real.sqrt 3 / 3) := by sorry

end NUMINAMATH_CALUDE_tan_negative_210_degrees_l3617_361700


namespace NUMINAMATH_CALUDE_equation_solution_l3617_361704

theorem equation_solution :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁^3 - 3*x₁*y₁^2 = 2007) ∧ (y₁^3 - 3*x₁^2*y₁ = 2006) ∧
    (x₂^3 - 3*x₂*y₂^2 = 2007) ∧ (y₂^3 - 3*x₂^2*y₂ = 2006) ∧
    (x₃^3 - 3*x₃*y₃^2 = 2007) ∧ (y₃^3 - 3*x₃^2*y₃ = 2006) →
    (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1003 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3617_361704


namespace NUMINAMATH_CALUDE_tax_free_limit_correct_l3617_361759

/-- The tax-free total value limit for imported goods in country X. -/
def tax_free_limit : ℝ := 500

/-- The tax rate applied to the value exceeding the tax-free limit. -/
def tax_rate : ℝ := 0.08

/-- The total value of goods imported by a specific tourist. -/
def total_value : ℝ := 730

/-- The tax paid by the tourist. -/
def tax_paid : ℝ := 18.40

/-- Theorem stating that the tax-free limit is correct given the problem conditions. -/
theorem tax_free_limit_correct : 
  tax_rate * (total_value - tax_free_limit) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_tax_free_limit_correct_l3617_361759


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l3617_361740

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss (current_weight weight_lost : ℕ) 
  (h1 : current_weight = 27)
  (h2 : weight_lost = 101) :
  current_weight + weight_lost = 128 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l3617_361740


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3617_361793

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (m-2)*x - m
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (∀ (y : ℝ), f y = 0 → y = x₁ ∨ y = x₂) ∧
  x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ - x₂ = 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3617_361793


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3617_361798

theorem polynomial_evaluation :
  let y : ℤ := -2
  (y^3 - y^2 + 2*y + 2 : ℤ) = -14 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3617_361798


namespace NUMINAMATH_CALUDE_intersection_P_complement_M_l3617_361761

-- Define the universal set U as the set of integers
def U : Set Int := Set.univ

-- Define set M
def M : Set Int := {1, 2}

-- Define set P
def P : Set Int := {-2, -1, 0, 1, 2}

-- Theorem statement
theorem intersection_P_complement_M : 
  P ∩ (U \ M) = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_P_complement_M_l3617_361761


namespace NUMINAMATH_CALUDE_factory_working_days_l3617_361713

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 5505

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 1101

/-- The number of working days per week -/
def working_days : ℕ := toys_per_week / toys_per_day

theorem factory_working_days : working_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_factory_working_days_l3617_361713


namespace NUMINAMATH_CALUDE_expression_simplification_l3617_361754

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (2 / (a + 1) + (a + 2) / (a^2 - 1)) / (a / (a + 1)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3617_361754


namespace NUMINAMATH_CALUDE_baseball_hits_percentage_l3617_361715

theorem baseball_hits_percentage (total_hits : ℕ) (home_runs : ℕ) (triples : ℕ) (doubles : ℕ)
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 10) :
  (total_hits - (home_runs + triples + doubles)) / total_hits * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_baseball_hits_percentage_l3617_361715


namespace NUMINAMATH_CALUDE_tips_fraction_of_income_l3617_361766

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- The fraction of income from tips for a waitress -/
def fractionFromTips (income : WaitressIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: If tips are 11/4 of salary, then 11/15 of income is from tips -/
theorem tips_fraction_of_income 
  (income : WaitressIncome) 
  (h : income.tips = (11 / 4) * income.salary) : 
  fractionFromTips income = 11 / 15 := by
  sorry

#check tips_fraction_of_income

end NUMINAMATH_CALUDE_tips_fraction_of_income_l3617_361766


namespace NUMINAMATH_CALUDE_sum_of_even_integers_2_to_2022_l3617_361743

theorem sum_of_even_integers_2_to_2022 : 
  (Finset.range 1011).sum (fun i => 2 * (i + 1)) = 1023112 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_2_to_2022_l3617_361743


namespace NUMINAMATH_CALUDE_sandys_money_l3617_361763

theorem sandys_money (pie_cost sandwich_cost book_cost remaining_money : ℕ) : 
  pie_cost = 6 →
  sandwich_cost = 3 →
  book_cost = 10 →
  remaining_money = 38 →
  pie_cost + sandwich_cost + book_cost + remaining_money = 57 := by
sorry

end NUMINAMATH_CALUDE_sandys_money_l3617_361763


namespace NUMINAMATH_CALUDE_hypotenuse_segment_ratio_l3617_361787

/-- A right triangle with leg lengths in ratio 3:4 -/
structure RightTriangle where
  a : ℝ  -- length of first leg
  b : ℝ  -- length of second leg
  h : ℝ  -- ratio of legs is 3:4
  leg_ratio : b = (4/3) * a

/-- The segments of the hypotenuse created by the altitude -/
structure HypotenuseSegments where
  x : ℝ  -- length of first segment
  y : ℝ  -- length of second segment

/-- Theorem: The ratio of hypotenuse segments is 21:16 -/
theorem hypotenuse_segment_ratio (t : RightTriangle) (s : HypotenuseSegments) :
  s.y / s.x = 21 / 16 :=
sorry

end NUMINAMATH_CALUDE_hypotenuse_segment_ratio_l3617_361787


namespace NUMINAMATH_CALUDE_smallest_factor_of_36_l3617_361789

theorem smallest_factor_of_36 (a b c : ℤ) (h1 : a * b * c = 36) (h2 : a + b + c = 4) :
  min a (min b c) = -4 :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_of_36_l3617_361789


namespace NUMINAMATH_CALUDE_parametric_to_general_form_l3617_361745

/-- Given parametric equations of a line, prove its general form -/
theorem parametric_to_general_form (t : ℝ) (x y : ℝ) :
  x = 2 - 3 * t ∧ y = 1 + 2 * t → 2 * x + 3 * y - 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_general_form_l3617_361745


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3617_361781

theorem trigonometric_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 2 / Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3617_361781


namespace NUMINAMATH_CALUDE_altitude_polynomial_exists_l3617_361756

/-- Given a triangle whose side lengths are the roots of a cubic polynomial
    with rational coefficients, there exists a polynomial of sixth degree
    with rational coefficients whose roots are the altitudes of this triangle. -/
theorem altitude_polynomial_exists (a b c d : ℚ) (r₁ r₂ r₃ : ℝ) :
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) →
  (r₁ + r₂ > r₃ ∧ r₂ + r₃ > r₁ ∧ r₃ + r₁ > r₂) →
  ∃ (p q s t u v w : ℚ), ∀ x : ℝ,
    p * x^6 + q * x^5 + s * x^4 + t * x^3 + u * x^2 + v * x + w = 0 ↔
    x = (2 * (r₁ * r₂ * r₃ * (r₁ + r₂ + r₃) * (r₁ + r₂ - r₃) * (r₂ + r₃ - r₁) * (r₃ + r₁ - r₂))^(1/2)) / (r₁ * (r₂ + r₃ - r₁))
    ∨ x = (2 * (r₁ * r₂ * r₃ * (r₁ + r₂ + r₃) * (r₁ + r₂ - r₃) * (r₂ + r₃ - r₁) * (r₃ + r₁ - r₂))^(1/2)) / (r₂ * (r₃ + r₁ - r₂))
    ∨ x = (2 * (r₁ * r₂ * r₃ * (r₁ + r₂ + r₃) * (r₁ + r₂ - r₃) * (r₂ + r₃ - r₁) * (r₃ + r₁ - r₂))^(1/2)) / (r₃ * (r₁ + r₂ - r₃)) :=
by
  sorry

end NUMINAMATH_CALUDE_altitude_polynomial_exists_l3617_361756


namespace NUMINAMATH_CALUDE_range_of_t_l3617_361711

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | (1/4 : ℝ) ≤ 2^x ∧ 2^x ≤ (1/2 : ℝ)}
def B (t : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*t*x + 1 ≤ 0}

-- State the theorem
theorem range_of_t (t : ℝ) : A ∩ B t = A → t ≤ -(5/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l3617_361711


namespace NUMINAMATH_CALUDE_sin_two_phi_l3617_361771

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_l3617_361771


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l3617_361765

/-- Proves that Danny found 1 more bottle cap than he threw away. -/
theorem danny_bottle_caps (found : ℕ) (thrown_away : ℕ) (current : ℕ)
  (h1 : found = 36)
  (h2 : thrown_away = 35)
  (h3 : current = 22)
  : found - thrown_away = 1 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l3617_361765


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3617_361709

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3617_361709


namespace NUMINAMATH_CALUDE_brand_a_soap_users_l3617_361734

theorem brand_a_soap_users (total : ℕ) (neither : ℕ) (both : ℕ) :
  total = 260 →
  neither = 80 →
  both = 30 →
  (total - neither) = (3 * both) + both + (total - neither - 3 * both - both) →
  (total - neither - 3 * both - both) = 60 :=
by sorry

end NUMINAMATH_CALUDE_brand_a_soap_users_l3617_361734


namespace NUMINAMATH_CALUDE_number_problem_l3617_361783

theorem number_problem (x : ℝ) : 0.4 * x = 0.2 * 650 + 190 ↔ x = 800 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3617_361783


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l3617_361706

/-- Calculates the total surface area of a cube with holes --/
def totalSurfaceArea (cubeEdge : ℝ) (holeEdge : ℝ) : ℝ :=
  let originalSurface := 6 * cubeEdge^2
  let holeArea := 6 * holeEdge^2
  let internalSurface := 6 * 4 * holeEdge^2
  originalSurface - holeArea + internalSurface

/-- The problem statement --/
theorem cube_with_holes_surface_area :
  totalSurfaceArea 5 2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l3617_361706


namespace NUMINAMATH_CALUDE_min_value_problem_l3617_361702

theorem min_value_problem (m n : ℝ) (hm : m > 0) (hn : n > 0) (heq : 2*m + n = 1) :
  (1/m) + (2/n) ≥ 8 ∧ ((1/m) + (2/n) = 8 ↔ n = 2*m ∧ n = 1/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3617_361702


namespace NUMINAMATH_CALUDE_infinite_pairs_divisibility_l3617_361729

theorem infinite_pairs_divisibility (m : ℕ) (h_m_even : Even m) (h_m_ge_2 : m ≥ 2) :
  ∃ n : ℕ, n = m + 1 ∧ 
    n ≥ 2 ∧ 
    (m^m - 1) % n = 0 ∧ 
    (n^n - 1) % m = 0 := by
  sorry

end NUMINAMATH_CALUDE_infinite_pairs_divisibility_l3617_361729


namespace NUMINAMATH_CALUDE_plant_beds_calculation_l3617_361776

/-- Calculate the number of plant beds required for given vegetable plantings -/
theorem plant_beds_calculation (bean_seedlings pumpkin_seeds radishes : ℕ)
  (bean_per_row pumpkin_per_row radish_per_row : ℕ)
  (rows_per_bed : ℕ)
  (h1 : bean_seedlings = 64)
  (h2 : pumpkin_seeds = 84)
  (h3 : radishes = 48)
  (h4 : bean_per_row = 8)
  (h5 : pumpkin_per_row = 7)
  (h6 : radish_per_row = 6)
  (h7 : rows_per_bed = 2) :
  (bean_seedlings / bean_per_row + pumpkin_seeds / pumpkin_per_row + radishes / radish_per_row) / rows_per_bed = 14 := by
  sorry

end NUMINAMATH_CALUDE_plant_beds_calculation_l3617_361776


namespace NUMINAMATH_CALUDE_distance_from_origin_l3617_361762

theorem distance_from_origin (x y : ℝ) (n : ℝ) : 
  y = 15 →
  (x - 2)^2 + (y - 8)^2 = 13^2 →
  x > 2 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (349 + 8 * Real.sqrt 30) := by
sorry

end NUMINAMATH_CALUDE_distance_from_origin_l3617_361762


namespace NUMINAMATH_CALUDE_bird_count_l3617_361760

/-- The number of birds in a park -/
theorem bird_count (blackbirds_per_tree : ℕ) (num_trees : ℕ) (num_magpies : ℕ) :
  blackbirds_per_tree = 3 →
  num_trees = 7 →
  num_magpies = 13 →
  blackbirds_per_tree * num_trees + num_magpies = 34 := by
sorry


end NUMINAMATH_CALUDE_bird_count_l3617_361760


namespace NUMINAMATH_CALUDE_sphere_radius_is_six_l3617_361799

/-- A truncated cone with horizontal bases of radii 12 and 3, and a sphere tangent to its top, bottom, and lateral surface. -/
structure TruncatedConeWithSphere where
  lower_radius : ℝ
  upper_radius : ℝ
  sphere_radius : ℝ
  lower_radius_eq : lower_radius = 12
  upper_radius_eq : upper_radius = 3
  sphere_tangent : True  -- We can't directly express tangency in this simple structure

/-- The radius of the sphere in the TruncatedConeWithSphere is 6. -/
theorem sphere_radius_is_six (cone : TruncatedConeWithSphere) : cone.sphere_radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_six_l3617_361799


namespace NUMINAMATH_CALUDE_badminton_probabilities_l3617_361753

/-- Represents the state of a badminton game -/
structure BadmintonGame where
  score_a : Nat
  score_b : Nat
  a_serving : Bool

/-- Rules for winning a badminton game -/
def game_won (game : BadmintonGame) : Bool :=
  (game.score_a = 21 && game.score_b < 20) ||
  (game.score_b = 21 && game.score_a < 20) ||
  (game.score_a ≥ 20 && game.score_b ≥ 20 && 
   ((game.score_a = 30) || (game.score_b = 30) || 
    (game.score_a ≥ 22 && game.score_a - game.score_b = 2) ||
    (game.score_b ≥ 22 && game.score_b - game.score_a = 2)))

/-- Probability of player A winning a rally when serving -/
def p_a_serving : ℝ := 0.4

/-- Probability of player A winning a rally when not serving -/
def p_a_not_serving : ℝ := 0.5

/-- The initial game state at 28:28 with A serving -/
def initial_state : BadmintonGame := ⟨28, 28, true⟩

theorem badminton_probabilities :
  let p_game_ends_in_two : ℝ := 0.46
  let p_a_wins : ℝ := 0.4
  (∃ (p_game_ends_in_two' p_a_wins' : ℝ),
    p_game_ends_in_two' = p_game_ends_in_two ∧
    p_a_wins' = p_a_wins ∧
    p_game_ends_in_two' = p_a_serving * p_a_serving + (1 - p_a_serving) * (1 - p_a_not_serving) ∧
    p_a_wins' = p_a_serving * p_a_serving + 
                p_a_serving * (1 - p_a_serving) * p_a_not_serving +
                (1 - p_a_serving) * p_a_not_serving * p_a_serving) :=
by sorry

end NUMINAMATH_CALUDE_badminton_probabilities_l3617_361753


namespace NUMINAMATH_CALUDE_blurred_page_frequency_l3617_361712

theorem blurred_page_frequency 
  (total_pages : ℕ) 
  (crumple_frequency : ℕ) 
  (unaffected_pages : ℕ) 
  (h1 : total_pages = 42)
  (h2 : crumple_frequency = 7)
  (h3 : unaffected_pages = 24) : 
  (total_pages - unaffected_pages - total_pages / crumple_frequency) / total_pages = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_blurred_page_frequency_l3617_361712


namespace NUMINAMATH_CALUDE_min_value_g_l3617_361728

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
def distance (p q : Point3D) : ℝ := sorry

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Function g(X) as defined in the problem -/
def g (tetra : Tetrahedron) (X : Point3D) : ℝ :=
  distance tetra.A X + distance tetra.B X + distance tetra.C X + distance tetra.D X

/-- Theorem stating the minimum value of g(X) for the given tetrahedron -/
theorem min_value_g (tetra : Tetrahedron) 
  (h1 : distance tetra.A tetra.D = 30)
  (h2 : distance tetra.B tetra.C = 30)
  (h3 : distance tetra.A tetra.C = 46)
  (h4 : distance tetra.B tetra.D = 46)
  (h5 : distance tetra.A tetra.B = 50)
  (h6 : distance tetra.C tetra.D = 50) :
  ∃ (min_val : ℝ), min_val = 4 * Real.sqrt 628 ∧ ∀ (X : Point3D), g tetra X ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_g_l3617_361728


namespace NUMINAMATH_CALUDE_max_daily_sales_amount_l3617_361744

def f (t : ℕ) : ℝ := -t + 30

def g (t : ℕ) : ℝ :=
  if t ≤ 10 then 2 * t + 40 else 15

def S (t : ℕ) : ℝ := f t * g t

theorem max_daily_sales_amount (t : ℕ) (h1 : 1 ≤ t) (h2 : t ≤ 20) :
  ∃ (max_t : ℕ) (max_value : ℝ), 
    (∀ t', 1 ≤ t' → t' ≤ 20 → S t' ≤ S max_t) ∧ 
    S max_t = max_value ∧ 
    max_t = 5 ∧ 
    max_value = 1250 :=
  sorry

end NUMINAMATH_CALUDE_max_daily_sales_amount_l3617_361744


namespace NUMINAMATH_CALUDE_cannot_reach_2003_l3617_361779

/-- The set of numbers that can appear on the board -/
def BoardNumbers : Set ℕ :=
  {n : ℕ | ∃ (k : ℕ), n ≡ 5 [ZMOD 5] ∨ n ≡ 7 [ZMOD 5] ∨ n ≡ 9 [ZMOD 5]}

/-- The transformation rule -/
def Transform (a b : ℕ) : ℕ := 5 * a - 4 * b

/-- Theorem stating that 2003 cannot appear on the board -/
theorem cannot_reach_2003 : 2003 ∉ BoardNumbers := by
  sorry

/-- Lemma: The transformation preserves the set of possible remainders modulo 5 -/
lemma transform_preserves_remainders (a b : ℕ) (h : a ∈ BoardNumbers) (h' : b ∈ BoardNumbers) :
  Transform a b ∈ BoardNumbers := by
  sorry

end NUMINAMATH_CALUDE_cannot_reach_2003_l3617_361779


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_minus_1_l3617_361716

theorem infinitely_many_primes_4k_minus_1 : 
  ∃ (S : Set Nat), (∀ n ∈ S, Nat.Prime n ∧ ∃ k, n = 4*k - 1) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_minus_1_l3617_361716


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_l3617_361710

theorem product_of_roots_cubic (a b c : ℂ) : 
  (3 * a^3 - 4 * a^2 + 9 * a - 18 = 0) ∧ 
  (3 * b^3 - 4 * b^2 + 9 * b - 18 = 0) ∧ 
  (3 * c^3 - 4 * c^2 + 9 * c - 18 = 0) → 
  a * b * c = 6 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_l3617_361710


namespace NUMINAMATH_CALUDE_function_properties_l3617_361782

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

def is_symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem function_properties (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_shift : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧ 
  (is_symmetric_about f 1) ∧
  (is_decreasing_on f 0 1) ∧
  (f 2014 = f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3617_361782


namespace NUMINAMATH_CALUDE_candy_problem_l3617_361792

theorem candy_problem (initial_candy : ℕ) (num_bowls : ℕ) (removed_per_bowl : ℕ) (remaining_in_bowl : ℕ)
  (h1 : initial_candy = 100)
  (h2 : num_bowls = 4)
  (h3 : removed_per_bowl = 3)
  (h4 : remaining_in_bowl = 20) :
  initial_candy - (num_bowls * (remaining_in_bowl + removed_per_bowl)) = 8 :=
sorry

end NUMINAMATH_CALUDE_candy_problem_l3617_361792


namespace NUMINAMATH_CALUDE_parallelogram_side_product_l3617_361773

/-- Given a parallelogram EFGH with side lengths as specified, 
    prove that the product of x and y is (53 * ∛6) / 3 --/
theorem parallelogram_side_product (x y : ℝ) : 
  (58 : ℝ) = 3 * x + 5 →   -- EF = GH (opposite sides are equal)
  (4 : ℝ) * y ^ 3 = 24 →   -- FG = HE (opposite sides are equal)
  x * y = 53 * (6 : ℝ) ^ (1/3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_product_l3617_361773


namespace NUMINAMATH_CALUDE_cans_recycling_l3617_361719

theorem cans_recycling (total_cans : ℕ) (saturday_bags : ℕ) (cans_per_bag : ℕ) : 
  total_cans = 42 →
  saturday_bags = 4 →
  cans_per_bag = 6 →
  (total_cans - saturday_bags * cans_per_bag) / cans_per_bag = 3 :=
by sorry

end NUMINAMATH_CALUDE_cans_recycling_l3617_361719


namespace NUMINAMATH_CALUDE_smallest_a_value_l3617_361774

theorem smallest_a_value (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0)
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x + π)) :
  a ≥ 17 ∧ (∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x + π)) → a' ≥ a) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3617_361774


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3617_361726

/-- The solution set of the inequality x + 2/(x+1) > 2 -/
theorem solution_set_inequality (x : ℝ) : x + 2 / (x + 1) > 2 ↔ x ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3617_361726


namespace NUMINAMATH_CALUDE_distance_between_vertices_l3617_361770

/-- The distance between the vertices of two quadratic functions -/
theorem distance_between_vertices (a b c d e f : ℝ) : 
  let f1 := fun x : ℝ => x^2 + a*x + b
  let f2 := fun x : ℝ => x^2 + c*x + d
  let vertex1 := (-a/2, f1 (-a/2))
  let vertex2 := (-c/2, f2 (-c/2))
  (a = -4 ∧ b = 5 ∧ c = 6 ∧ d = 13) →
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l3617_361770


namespace NUMINAMATH_CALUDE_total_treats_eq_155_l3617_361790

/-- The number of chewing gums -/
def chewing_gums : ℕ := 60

/-- The number of chocolate bars -/
def chocolate_bars : ℕ := 55

/-- The number of candies of different flavors -/
def candies : ℕ := 40

/-- The total number of treats -/
def total_treats : ℕ := chewing_gums + chocolate_bars + candies

theorem total_treats_eq_155 : total_treats = 155 := by sorry

end NUMINAMATH_CALUDE_total_treats_eq_155_l3617_361790


namespace NUMINAMATH_CALUDE_semicircle_radius_theorem_l3617_361701

/-- Theorem: Given a rectangle with length 48 cm and width 24 cm, and a semicircle
    attached to one side of the rectangle (with the diameter equal to the length
    of the rectangle), if the perimeter of the combined shape is 144 cm, then the
    radius of the semicircle is 48 / (π + 2) cm. -/
theorem semicircle_radius_theorem (rectangle_length : ℝ) (rectangle_width : ℝ) 
    (combined_perimeter : ℝ) (semicircle_radius : ℝ) :
  rectangle_length = 48 →
  rectangle_width = 24 →
  combined_perimeter = 144 →
  combined_perimeter = 2 * rectangle_width + rectangle_length + π * semicircle_radius →
  semicircle_radius = 48 / (π + 2) :=
by sorry

end NUMINAMATH_CALUDE_semicircle_radius_theorem_l3617_361701


namespace NUMINAMATH_CALUDE_johns_running_time_l3617_361795

theorem johns_running_time (H : ℝ) : 
  H > 0 →
  (12 : ℝ) * (1.75 * H) = 168 →
  H = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_running_time_l3617_361795


namespace NUMINAMATH_CALUDE_odd_prime_congruence_l3617_361723

theorem odd_prime_congruence (p : Nat) (c : Int) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ a : Int, (a^((p+1)/2) + (a+c)^((p+1)/2)) % p = c % p := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_congruence_l3617_361723


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l3617_361777

-- Define arithmetic sequence a_n
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric sequence b_n
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

theorem arithmetic_geometric_sequence_problem
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  d ≠ 0 →
  2 * a 4 - (a 7)^2 + 2 * a 10 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 5 * b 9 = 16 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l3617_361777


namespace NUMINAMATH_CALUDE_intersection_segment_length_l3617_361747

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y + Real.sqrt 3 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  curve_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_segment_length :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32 / 7 := by sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l3617_361747


namespace NUMINAMATH_CALUDE_same_remainder_implies_specific_remainder_l3617_361778

theorem same_remainder_implies_specific_remainder 
  (m : ℕ) 
  (h1 : m ≠ 1) 
  (h2 : ∃ r : ℕ, 69 % m = r ∧ 90 % m = r ∧ 125 % m = r) : 
  86 % m = 2 := by
sorry

end NUMINAMATH_CALUDE_same_remainder_implies_specific_remainder_l3617_361778


namespace NUMINAMATH_CALUDE_hypotenuse_length_of_special_triangle_l3617_361751

theorem hypotenuse_length_of_special_triangle : 
  ∀ (a b c : ℝ), 
  (a^2 - 17*a + 60 = 0) → 
  (b^2 - 17*b + 60 = 0) → 
  (a ≠ b) →
  (c^2 = a^2 + b^2) →
  c = 13 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_length_of_special_triangle_l3617_361751


namespace NUMINAMATH_CALUDE_smallest_nat_greater_than_12_l3617_361735

theorem smallest_nat_greater_than_12 :
  ∀ n : ℕ, n > 12 → n ≥ 13 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_nat_greater_than_12_l3617_361735


namespace NUMINAMATH_CALUDE_investment_split_l3617_361788

/-- Proves that given a total investment of $15,000 split between two rates of 6% and 7.5%, 
    which together yield $1,023 in simple interest after one year, 
    the amount invested at 6% is $6,800. -/
theorem investment_split (x y : ℝ) : 
  x + y = 15000 →
  0.06 * x + 0.075 * y = 1023 →
  x = 6800 := by
sorry

end NUMINAMATH_CALUDE_investment_split_l3617_361788


namespace NUMINAMATH_CALUDE_binomial_expansion_probability_l3617_361755

/-- The number of terms in the binomial expansion -/
def num_terms : ℕ := 9

/-- The exponent of the binomial -/
def n : ℕ := num_terms - 1

/-- The number of rational terms in the expansion -/
def num_rational_terms : ℕ := 3

/-- The number of irrational terms in the expansion -/
def num_irrational_terms : ℕ := num_terms - num_rational_terms

/-- The total number of permutations of all terms -/
def total_permutations : ℕ := (Nat.factorial num_terms)

/-- The number of favorable permutations where rational terms are not adjacent -/
def favorable_permutations : ℕ := 
  (Nat.factorial num_irrational_terms) * (Nat.choose (num_irrational_terms + 1) num_rational_terms)

/-- The probability that all rational terms are not adjacent when rearranged -/
def probability : ℚ := (favorable_permutations : ℚ) / (total_permutations : ℚ)

theorem binomial_expansion_probability : probability = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_probability_l3617_361755


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_for_given_problem_l3617_361725

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

theorem systematic_sampling_interval_for_given_problem :
  let populationSize : ℕ := 1000
  let sampleSize : ℕ := 40
  systematicSamplingInterval populationSize sampleSize = 25 := by
  sorry

#eval systematicSamplingInterval 1000 40

end NUMINAMATH_CALUDE_systematic_sampling_interval_for_given_problem_l3617_361725


namespace NUMINAMATH_CALUDE_num_tangent_circles_bounds_l3617_361722

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of solutions for circles tangent to a line and another circle --/
def num_tangent_circles (r : ℝ) (L : Line) (C : Circle) : ℕ :=
  sorry

/-- Theorem stating the bounds on the number of tangent circles --/
theorem num_tangent_circles_bounds (r : ℝ) (L : Line) (C : Circle) :
  0 ≤ num_tangent_circles r L C ∧ num_tangent_circles r L C ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_num_tangent_circles_bounds_l3617_361722


namespace NUMINAMATH_CALUDE_rent_is_840_l3617_361727

/-- The total rent for a pasture shared by three people --/
def total_rent (a_horses b_horses c_horses : ℕ) (a_months b_months c_months : ℕ) (b_rent : ℕ) : ℕ :=
  let a_horse_months := a_horses * a_months
  let b_horse_months := b_horses * b_months
  let c_horse_months := c_horses * c_months
  let total_horse_months := a_horse_months + b_horse_months + c_horse_months
  (b_rent * total_horse_months) / b_horse_months

/-- Theorem stating that the total rent is 840 given the problem conditions --/
theorem rent_is_840 :
  total_rent 12 16 18 8 9 6 348 = 840 := by
  sorry

end NUMINAMATH_CALUDE_rent_is_840_l3617_361727


namespace NUMINAMATH_CALUDE_power_of_128_equals_32_l3617_361742

theorem power_of_128_equals_32 : (128 : ℝ) ^ (5/7 : ℝ) = 32 := by
  have h : 128 = 2^7 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_128_equals_32_l3617_361742


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3617_361737

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 6 * x + c = 0) →
  a + c = 12 →
  a < c →
  (a, c) = (6 - 3 * Real.sqrt 3, 6 + 3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3617_361737


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3617_361780

theorem complex_number_in_first_quadrant : let z : ℂ := (Complex.I) / (Complex.I + 1)
  (0 < z.re) ∧ (0 < z.im) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3617_361780


namespace NUMINAMATH_CALUDE_inequality_solution_l3617_361721

theorem inequality_solution (x : ℝ) :
  (x + 2) / ((x + 1)^2) < 0 ↔ x < -2 ∧ x ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3617_361721


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3617_361786

theorem no_integer_solutions : 
  ¬∃ (x y z : ℤ), 
    (x^2 - 3*x*y + 2*y^2 - z^2 = 31) ∧ 
    (-x^2 + 6*y*z + 2*z^2 = 44) ∧ 
    (x^2 + x*y + 8*z^2 = 100) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3617_361786


namespace NUMINAMATH_CALUDE_board_sum_always_odd_l3617_361775

theorem board_sum_always_odd (n : ℕ) (h : n = 1966) :
  let initial_sum := n * (n + 1) / 2
  ∀ (operations : ℕ), ∃ (final_sum : ℤ),
    final_sum ≡ initial_sum [ZMOD 2] ∧ final_sum ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_board_sum_always_odd_l3617_361775


namespace NUMINAMATH_CALUDE_fraction_simplification_l3617_361731

theorem fraction_simplification (x y : ℚ) (hx : x = 4/3) (hy : y = 8/6) : 
  (6 * x^2 + 4 * y) / (36 * x * y) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3617_361731


namespace NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_l3617_361794

theorem x_squared_geq_one_necessary_not_sufficient :
  (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_l3617_361794


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_power_10000_l3617_361772

theorem last_three_digits_of_7_power_10000 (h : 7^500 ≡ 1 [ZMOD 1250]) :
  7^10000 ≡ 1 [ZMOD 1250] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_power_10000_l3617_361772


namespace NUMINAMATH_CALUDE_frog_corner_prob_four_hops_l3617_361758

/-- Represents a position on the 4x4 grid -/
inductive Position
| Center
| Edge
| Corner

/-- Represents the state of the frog's movement -/
structure FrogState where
  position : Position
  hops : Nat

/-- The probability of moving to a corner in one hop from a given position -/
def cornerProbFromPosition (pos : Position) : ℚ :=
  match pos with
  | Position.Center => 0
  | Position.Edge => 1/8
  | Position.Corner => 1

/-- The probability of the frog being in a corner after n hops -/
def cornerProbAfterNHops (n : Nat) : ℚ :=
  sorry

theorem frog_corner_prob_four_hops :
  cornerProbAfterNHops 4 = 3/8 := by sorry

end NUMINAMATH_CALUDE_frog_corner_prob_four_hops_l3617_361758


namespace NUMINAMATH_CALUDE_coordinate_problem_l3617_361784

/-- Represents a point in the coordinate system -/
structure Point where
  x : ℕ
  y : ℕ

/-- The problem statement -/
theorem coordinate_problem (A B : Point) : 
  (A.x < A.y) →  -- Angle OA > 45°
  (B.x > B.y) →  -- Angle OB < 45°
  (B.x * B.y - A.x * A.y = 67) →  -- Area difference
  (A.x * 1000 + B.x * 100 + B.y * 10 + A.y = 1985) := by
  sorry

end NUMINAMATH_CALUDE_coordinate_problem_l3617_361784


namespace NUMINAMATH_CALUDE_planting_area_is_2x_l3617_361732

/-- Represents the area of the planting region in a rectangular garden with an internal path. -/
def planting_area (x : ℝ) : ℝ :=
  let garden_length : ℝ := x + 2
  let garden_width : ℝ := 4
  let path_width : ℝ := 1
  let planting_length : ℝ := garden_length - 2 * path_width
  let planting_width : ℝ := garden_width - 2 * path_width
  planting_length * planting_width

/-- Theorem stating that the planting area is equal to 2x square meters. -/
theorem planting_area_is_2x (x : ℝ) : planting_area x = 2 * x := by
  sorry


end NUMINAMATH_CALUDE_planting_area_is_2x_l3617_361732


namespace NUMINAMATH_CALUDE_area_ratio_is_nine_thirtytwo_l3617_361703

/-- Triangle XYZ with points G, H, I on its sides -/
structure TriangleXYZ where
  /-- Length of side XY -/
  xy : ℝ
  /-- Length of side YZ -/
  yz : ℝ
  /-- Length of side ZX -/
  zx : ℝ
  /-- Ratio of XG to XY -/
  s : ℝ
  /-- Ratio of YH to YZ -/
  t : ℝ
  /-- Ratio of ZI to ZX -/
  u : ℝ
  /-- XY length is 14 -/
  xy_eq : xy = 14
  /-- YZ length is 16 -/
  yz_eq : yz = 16
  /-- ZX length is 18 -/
  zx_eq : zx = 18
  /-- s is positive -/
  s_pos : s > 0
  /-- t is positive -/
  t_pos : t > 0
  /-- u is positive -/
  u_pos : u > 0
  /-- Sum of s, t, u is 3/4 -/
  sum_stu : s + t + u = 3/4
  /-- Sum of squares of s, t, u is 3/8 -/
  sum_sq_stu : s^2 + t^2 + u^2 = 3/8

/-- The ratio of the area of triangle GHI to the area of triangle XYZ -/
def areaRatio (T : TriangleXYZ) : ℝ :=
  1 - T.s * (1 - T.u) - T.t * (1 - T.s) - T.u * (1 - T.t)

theorem area_ratio_is_nine_thirtytwo (T : TriangleXYZ) : 
  areaRatio T = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_nine_thirtytwo_l3617_361703


namespace NUMINAMATH_CALUDE_complex_multiplication_l3617_361705

/-- Given that i is the imaginary unit, prove that i(3-4i) = 4 + 3i -/
theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (3 - 4*i) = 4 + 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3617_361705


namespace NUMINAMATH_CALUDE_quadrilateral_bf_length_l3617_361717

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
variable (h1 : A.1 = 0 ∧ A.2 = 0)  -- A is at (0,0)
variable (h2 : C.1 = 10 ∧ C.2 = 0)  -- C is at (10,0)
variable (h3 : E.1 = 3 ∧ E.2 = 0)  -- E is at (3,0)
variable (h4 : F.1 = 7 ∧ F.2 = 0)  -- F is at (7,0)
variable (h5 : D.1 = 3 ∧ D.2 = -5)  -- D is at (3,-5)
variable (h6 : B.1 = 7 ∧ B.2 = 4.2)  -- B is at (7,4.2)

-- Define the geometric conditions
variable (h7 : (B.2 - A.2) * (D.1 - A.1) = (D.2 - A.2) * (B.1 - A.1))  -- ∠BAD is right
variable (h8 : (B.2 - C.2) * (D.1 - C.1) = (D.2 - C.2) * (B.1 - C.1))  -- ∠BCD is right
variable (h9 : (D.2 - E.2) * (C.1 - E.1) = (C.2 - E.2) * (D.1 - E.1))  -- DE ⊥ AC
variable (h10 : (B.2 - F.2) * (A.1 - F.1) = (A.2 - F.2) * (B.1 - F.1))  -- BF ⊥ AC

-- Define the length conditions
variable (h11 : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 3)  -- AE = 3
variable (h12 : Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = 5)  -- DE = 5
variable (h13 : Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2) = 7)  -- CE = 7

-- Theorem statement
theorem quadrilateral_bf_length : 
  Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) = 4.2 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_bf_length_l3617_361717


namespace NUMINAMATH_CALUDE_solution_equation_l3617_361741

theorem solution_equation (x : ℝ) (hx : x ≠ 0) :
  (9 * x)^10 = (18 * x)^5 → x = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_solution_equation_l3617_361741


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3617_361730

/-- Number of ways to arrange books with specific conditions -/
def arrange_books (math_books : ℕ) (history_books : ℕ) : ℕ :=
  let total_books := math_books + history_books
  let middle_slots := total_books - 2
  let unrestricted_arrangements := Nat.factorial middle_slots
  let adjacent_arrangements := Nat.factorial (middle_slots - 1) * 2
  (math_books * (math_books - 1)) * (unrestricted_arrangements - adjacent_arrangements)

/-- Theorem stating the number of ways to arrange books under given conditions -/
theorem book_arrangement_count :
  arrange_books 4 6 = 362880 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3617_361730


namespace NUMINAMATH_CALUDE_actual_speed_is_30_l3617_361757

/-- Given that increasing the speed by 10 miles per hour reduces travel time by 1/4,
    prove that the actual average speed is 30 miles per hour. -/
theorem actual_speed_is_30 (v : ℝ) (h : v / (v + 10) = 3 / 4) : v = 30 := by
  sorry

end NUMINAMATH_CALUDE_actual_speed_is_30_l3617_361757


namespace NUMINAMATH_CALUDE_minkowski_sum_convex_l3617_361714

-- Define a type for points in a 2D space
variable {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Define a convex figure as a set of points
def ConvexFigure (S : Set α) : Prop :=
  ∀ (x y : α), x ∈ S → y ∈ S → ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
    (1 - t) • x + t • y ∈ S

-- Define Minkowski sum of two sets
def MinkowskiSum (S T : Set α) : Set α :=
  {z | ∃ (x y : α), x ∈ S ∧ y ∈ T ∧ z = x + y}

-- Theorem statement
theorem minkowski_sum_convex
  (Φ₁ Φ₂ : Set α) (h1 : ConvexFigure Φ₁) (h2 : ConvexFigure Φ₂) :
  ConvexFigure (MinkowskiSum Φ₁ Φ₂) :=
sorry

end NUMINAMATH_CALUDE_minkowski_sum_convex_l3617_361714


namespace NUMINAMATH_CALUDE_true_proposition_l3617_361749

-- Define proposition p
def p : Prop := ∀ α : ℝ, (Real.sin α)^2 ≤ 1

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀^2 + 1 = 0

-- Theorem statement
theorem true_proposition (hp : p) (hq : ¬q) : (¬p) ∨ (¬q) :=
by
  sorry

end NUMINAMATH_CALUDE_true_proposition_l3617_361749


namespace NUMINAMATH_CALUDE_min_value_theorem_l3617_361733

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a / (4 * b) + 1 / a ≥ 2 ∧
  (a / (4 * b) + 1 / a = 2 ↔ a = 2/3 ∧ b = 1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3617_361733


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3617_361746

/-- Given that i is the imaginary unit, prove that (3 + i) / (1 + 2*i) = 1 - i -/
theorem complex_fraction_simplification :
  (3 + I : ℂ) / (1 + 2*I) = 1 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3617_361746


namespace NUMINAMATH_CALUDE_water_layer_thickness_l3617_361791

/-- Thickness of water layer after removing a sphere from a cylindrical vessel -/
theorem water_layer_thickness (R r : ℝ) (h_R : R = 4) (h_r : r = 3) :
  let V := π * R^2 * (2 * r)
  let V_sphere := (4/3) * π * r^3
  let V_water := V - V_sphere
  V_water / (π * R^2) = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_water_layer_thickness_l3617_361791


namespace NUMINAMATH_CALUDE_units_digit_of_8429_pow_1246_l3617_361720

theorem units_digit_of_8429_pow_1246 :
  (8429^1246) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_8429_pow_1246_l3617_361720


namespace NUMINAMATH_CALUDE_fraction_exceeding_by_20_l3617_361767

theorem fraction_exceeding_by_20 (N : ℚ) (F : ℚ) : 
  N = 32 → N = F * N + 20 → F = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_exceeding_by_20_l3617_361767


namespace NUMINAMATH_CALUDE_smallest_c_value_l3617_361797

theorem smallest_c_value (c d : ℝ) (h_nonneg_c : c ≥ 0) (h_nonneg_d : d ≥ 0)
  (h_cos_eq : ∀ x : ℤ, Real.cos (c * x + d) = Real.cos (17 * x)) :
  c ≥ 17 ∧ ∃ (c' : ℝ), c' ≥ 0 ∧ c' < 17 → ¬(∀ x : ℤ, Real.cos (c' * x + d) = Real.cos (17 * x)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l3617_361797


namespace NUMINAMATH_CALUDE_samantha_coins_value_l3617_361748

theorem samantha_coins_value :
  ∀ (n d : ℕ),
  n + d = 30 →
  5 * n + 10 * d + 120 = 10 * n + 5 * d →
  5 * n + 10 * d = 165 :=
by
  sorry

end NUMINAMATH_CALUDE_samantha_coins_value_l3617_361748


namespace NUMINAMATH_CALUDE_right_triangles_in_18gon_l3617_361768

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- A right-angled triangle formed by three vertices of a regular polygon --/
structure RightTriangle (p : RegularPolygon n) where
  vertices : Fin 3 → Fin n
  is_right_angled : sorry

/-- The number of right-angled triangles in a regular polygon --/
def num_right_triangles (p : RegularPolygon n) : ℕ :=
  sorry

theorem right_triangles_in_18gon :
  ∀ (p : RegularPolygon 18), num_right_triangles p = 144 :=
sorry

end NUMINAMATH_CALUDE_right_triangles_in_18gon_l3617_361768


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3617_361769

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 5 →
  downstream_distance = 34.47 →
  downstream_time = 44 / 60 →
  ∃ (boat_speed : ℝ), abs (boat_speed - 42.01) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3617_361769


namespace NUMINAMATH_CALUDE_expression_equal_to_five_l3617_361736

theorem expression_equal_to_five : 3^2 - 2^2 = 5 := by
  sorry

#check expression_equal_to_five

end NUMINAMATH_CALUDE_expression_equal_to_five_l3617_361736


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3617_361708

theorem solve_linear_equation (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x + 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3617_361708


namespace NUMINAMATH_CALUDE_simplified_expression_l3617_361718

theorem simplified_expression (x : ℝ) :
  Real.sqrt (4 * x^2 - 8 * x + 4) + Real.sqrt (4 * x^2 + 8 * x + 4) + 5 =
  2 * |x - 1| + 2 * |x + 1| + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l3617_361718


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3617_361764

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) * (2 - Complex.I)
  (z.re = 0) → m = -2 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3617_361764


namespace NUMINAMATH_CALUDE_fraction_transformation_l3617_361796

theorem fraction_transformation (a b : ℝ) (h : b ≠ 0) :
  a / b = (a + 2 * a) / (b + 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3617_361796


namespace NUMINAMATH_CALUDE_complex_equation_imaginary_part_l3617_361707

theorem complex_equation_imaginary_part :
  ∀ z : ℂ, (1 + Complex.I) / (3 * Complex.I + z) = Complex.I →
  z.im = -4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_imaginary_part_l3617_361707


namespace NUMINAMATH_CALUDE_stuffed_animals_sales_difference_l3617_361785

/-- Given the sales of stuffed animals by Jake, Thor, and Quincy, prove that Quincy sold 170 more than Jake. -/
theorem stuffed_animals_sales_difference :
  ∀ (jake_sales thor_sales quincy_sales : ℕ),
  jake_sales = thor_sales + 10 →
  quincy_sales = 10 * thor_sales →
  quincy_sales = 200 →
  quincy_sales - jake_sales = 170 := by
sorry

end NUMINAMATH_CALUDE_stuffed_animals_sales_difference_l3617_361785


namespace NUMINAMATH_CALUDE_sandys_puppies_l3617_361724

/-- Given that Sandy initially had 8 puppies and gave away 4 puppies,
    prove that she now has 4 puppies remaining. -/
theorem sandys_puppies (initial_puppies : ℕ) (puppies_given_away : ℕ)
  (h1 : initial_puppies = 8)
  (h2 : puppies_given_away = 4) :
  initial_puppies - puppies_given_away = 4 := by
sorry

end NUMINAMATH_CALUDE_sandys_puppies_l3617_361724


namespace NUMINAMATH_CALUDE_yumi_counting_l3617_361752

def reduce_number (start : ℕ) (amount : ℕ) (times : ℕ) : ℕ :=
  start - amount * times

theorem yumi_counting :
  reduce_number 320 10 4 = 280 := by
  sorry

end NUMINAMATH_CALUDE_yumi_counting_l3617_361752


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3617_361739

/-- An isosceles triangle with side lengths 5 and 2 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  isIsosceles : a = b
  sideLength1 : a = 5
  sideLength2 : b = 5
  base : ℝ
  baseLength : base = 2

/-- The perimeter of the isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.a + t.b + t.base

theorem isosceles_triangle_perimeter :
  ∀ (t : IsoscelesTriangle), perimeter t = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3617_361739


namespace NUMINAMATH_CALUDE_initial_bottle_caps_l3617_361738

/-- Given the number of bottle caps lost and the final number of bottle caps,
    calculate the initial number of bottle caps. -/
theorem initial_bottle_caps (lost final : ℝ) : 
  lost = 18.0 → final = 45 → lost + final = 63.0 := by sorry

end NUMINAMATH_CALUDE_initial_bottle_caps_l3617_361738
