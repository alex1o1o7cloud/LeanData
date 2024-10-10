import Mathlib

namespace weekly_rainfall_sum_l819_81923

def monday_rainfall : ℝ := 0.12962962962962962
def tuesday_rainfall : ℝ := 0.35185185185185186
def wednesday_rainfall : ℝ := 0.09259259259259259
def thursday_rainfall : ℝ := 0.25925925925925924
def friday_rainfall : ℝ := 0.48148148148148145
def saturday_rainfall : ℝ := 0.2222222222222222
def sunday_rainfall : ℝ := 0.4444444444444444

theorem weekly_rainfall_sum :
  monday_rainfall + tuesday_rainfall + wednesday_rainfall + thursday_rainfall +
  friday_rainfall + saturday_rainfall + sunday_rainfall = 1.9814814814814815 := by
  sorry

end weekly_rainfall_sum_l819_81923


namespace power_eleven_mod_hundred_l819_81957

theorem power_eleven_mod_hundred : 11^2023 % 100 = 31 := by
  sorry

end power_eleven_mod_hundred_l819_81957


namespace min_value_expression_min_value_achieved_l819_81962

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (1 / a + 4 / b) ≥ 9 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + b) * (1 / a + 4 / b) < 9 + ε :=
by sorry

end min_value_expression_min_value_achieved_l819_81962


namespace count_pairs_satisfying_inequality_l819_81909

theorem count_pairs_satisfying_inequality : 
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 * p.2 < 30 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 6) (Finset.range 30))).card = 41 := by
  sorry

end count_pairs_satisfying_inequality_l819_81909


namespace expression_simplification_l819_81942

theorem expression_simplification (a b : ℝ) (ha : a = -1) (hb : b = 2) :
  (a + b)^2 + (a^2 * b - 2 * a * b^2 - b^3) / b - (a - b) * (a + b) = 5 := by
  sorry

end expression_simplification_l819_81942


namespace units_digit_factorial_product_15_l819_81968

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def productFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.foldl (·*·) 1

theorem units_digit_factorial_product_15 :
  unitsDigit (productFactorials 15) = 0 := by
  sorry

end units_digit_factorial_product_15_l819_81968


namespace function_value_at_pi_over_12_l819_81991

theorem function_value_at_pi_over_12 (x : Real) (h : x = π / 12) :
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2) - 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) = 
  Real.sqrt 3 / 2 := by
  sorry

end function_value_at_pi_over_12_l819_81991


namespace two_numbers_sum_and_sum_of_squares_l819_81951

theorem two_numbers_sum_and_sum_of_squares (a b : ℝ) :
  (∃ (x y : ℚ), x > 0 ∧ y > 0 ∧ (x : ℝ) + y = a ∧ (x : ℝ)^2 + y^2 = b) ↔
  (∃ (k : ℕ), 2*b - a^2 = (k : ℝ)^2 ∧ k > 0) :=
sorry

end two_numbers_sum_and_sum_of_squares_l819_81951


namespace functional_equation_properties_l819_81932

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (f 0 = 0) ∧ (f 1 = 0) ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end functional_equation_properties_l819_81932


namespace carolyn_piano_practice_time_l819_81903

/-- Given Carolyn's practice schedule, prove she practices piano for 20 minutes daily. -/
theorem carolyn_piano_practice_time :
  ∀ (piano_time : ℕ),
    (∃ (violin_time : ℕ), violin_time = 3 * piano_time) →
    (∃ (weekly_practice : ℕ), weekly_practice = 6 * (piano_time + 3 * piano_time)) →
    (∃ (monthly_practice : ℕ), monthly_practice = 4 * 6 * (piano_time + 3 * piano_time)) →
    4 * 6 * (piano_time + 3 * piano_time) = 1920 →
    piano_time = 20 := by
  sorry

end carolyn_piano_practice_time_l819_81903


namespace factorial_30_prime_factors_l819_81978

theorem factorial_30_prime_factors : 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 10 := by
  sorry

end factorial_30_prime_factors_l819_81978


namespace cuboid_diagonal_count_l819_81949

/-- The number of unit cubes a diagonal passes through in a cuboid -/
def diagonalCubeCount (length width height : ℕ) : ℕ :=
  length + width + height - 2

/-- Theorem: The number of unit cubes a diagonal passes through in a 77 × 81 × 100 cuboid is 256 -/
theorem cuboid_diagonal_count :
  diagonalCubeCount 77 81 100 = 256 := by
  sorry

end cuboid_diagonal_count_l819_81949


namespace simplify_sqrt_fraction_l819_81950

theorem simplify_sqrt_fraction : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end simplify_sqrt_fraction_l819_81950


namespace factor_x4_plus_16_l819_81997

theorem factor_x4_plus_16 (x : ℝ) : x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end factor_x4_plus_16_l819_81997


namespace cube_sum_theorem_l819_81919

theorem cube_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a * b + a * c + b * c = 3)
  (h3 : a * b * c = 5) :
  a^3 + b^3 + c^3 = 15 := by
sorry

end cube_sum_theorem_l819_81919


namespace quadratic_trinomial_minimum_l819_81915

theorem quadratic_trinomial_minimum (a b : ℝ) (h1 : a > b)
  (h2 : ∀ x : ℝ, a * x^2 + 2*x + b ≥ 0)
  (h3 : ∃ x₀ : ℝ, a * x₀^2 + 2*x₀ + b = 0) :
  (∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2) ∧
  (∃ x : ℝ, (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2) :=
sorry

end quadratic_trinomial_minimum_l819_81915


namespace line_slope_proof_l819_81906

theorem line_slope_proof : 
  let A : ℝ := Real.sin (30 * π / 180)
  let B : ℝ := Real.cos (150 * π / 180)
  let slope : ℝ := -A / B
  slope = Real.sqrt 3 / 3 :=
by sorry

end line_slope_proof_l819_81906


namespace sequence_strictly_decreasing_l819_81989

/-- Given real numbers a and b with b > a > 1, prove that the sequence x_n is strictly monotonically decreasing -/
theorem sequence_strictly_decreasing (a b : ℝ) (h1 : a > 1) (h2 : b > a) : 
  ∀ n : ℕ, (2^n * (b^(1/2^n) - a^(1/2^n))) > (2^(n+1) * (b^(1/2^(n+1)) - a^(1/2^(n+1)))) := by
  sorry

#check sequence_strictly_decreasing

end sequence_strictly_decreasing_l819_81989


namespace max_value_of_f_l819_81990

def f (x : ℝ) := x^2 + 2*x

theorem max_value_of_f :
  ∃ (M : ℝ), M = 8 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ M :=
sorry

end max_value_of_f_l819_81990


namespace work_completion_time_l819_81946

/-- Given that A can do a work in 12 days and A and B together can do the work in 8 days,
    prove that B can do the work alone in 24 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 12) (hab : 1 / a + 1 / b = 1 / 8) : b = 24 := by
  sorry

end work_completion_time_l819_81946


namespace wall_length_calculation_l819_81961

/-- Given a square mirror and a rectangular wall, if the mirror's area is half the wall's area,
    prove the length of the wall. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 18 →
  wall_width = 32 →
  (mirror_side ^ 2) * 2 = wall_width * (20.25 : ℝ) := by
  sorry

end wall_length_calculation_l819_81961


namespace tinsel_count_l819_81940

/-- The number of pieces of tinsel in each box of Christmas decorations. -/
def tinsel_per_box : ℕ := 4

/-- The number of Christmas trees in each box. -/
def trees_per_box : ℕ := 1

/-- The number of snow globes in each box. -/
def snow_globes_per_box : ℕ := 5

/-- The total number of boxes distributed. -/
def total_boxes : ℕ := 12

/-- The total number of decorations handed out. -/
def total_decorations : ℕ := 120

/-- Theorem stating that the number of pieces of tinsel in each box is 4. -/
theorem tinsel_count : 
  total_boxes * (tinsel_per_box + trees_per_box + snow_globes_per_box) = total_decorations :=
by sorry

end tinsel_count_l819_81940


namespace max_perimeter_special_triangle_l819_81977

/-- Represents a triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  is_triangle : a + b > c ∧ b + c > a ∧ c + a > b

/-- A triangle satisfying the given conditions -/
def SpecialTriangle (t : IntTriangle) : Prop :=
  (t.a = 4 * t.b ∨ t.b = 4 * t.c ∨ t.c = 4 * t.a) ∧ (t.a = 20 ∨ t.b = 20 ∨ t.c = 20)

/-- The perimeter of a triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Theorem stating the maximum perimeter of the special triangle -/
theorem max_perimeter_special_triangle :
  ∀ t : IntTriangle, SpecialTriangle t → perimeter t ≤ 50 :=
sorry

end max_perimeter_special_triangle_l819_81977


namespace original_plums_count_l819_81995

theorem original_plums_count (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 4 → total = 21 → initial + added = total → initial = 17 := by
sorry

end original_plums_count_l819_81995


namespace ellipse_problem_l819_81929

-- Define the ellipses and points
def C₁ (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1
def C₂ (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
theorem ellipse_problem (a b : ℝ) (A B H P M N : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (∃ x y, C₂ a b x y ∧ x^2 = 5 ∧ y = 0) ∧
  (∃ x₁ y₁ x₂ y₂, C₂ a b x₁ y₁ ∧ C₂ a b x₂ y₂ ∧ y₂ - y₁ = x₂ - x₁) ∧
  H = (2, -1) ∧
  C₂ a b P.1 P.2 ∧
  C₁ M.1 M.2 ∧
  C₁ N.1 N.2 ∧
  P.1 = M.1 + 2 * N.1 ∧
  P.2 = M.2 + 2 * N.2 →
  (a^2 = 10 ∧ b^2 = 5) ∧
  (M.2 / M.1 * N.2 / N.1 = -1/2) :=
by sorry

end ellipse_problem_l819_81929


namespace smallest_lattice_triangle_area_is_half_l819_81937

/-- A lattice triangle is a triangle on a square grid where all vertices are grid points. -/
structure LatticeTriangle where
  vertices : Fin 3 → ℤ × ℤ

/-- The area of a grid square is 1 square unit. -/
def grid_square_area : ℝ := 1

/-- The area of a lattice triangle -/
def lattice_triangle_area (t : LatticeTriangle) : ℝ := sorry

/-- The smallest possible area of a lattice triangle -/
def smallest_lattice_triangle_area : ℝ := sorry

/-- Theorem: The area of the smallest lattice triangle is 1/2 square unit -/
theorem smallest_lattice_triangle_area_is_half :
  smallest_lattice_triangle_area = 1/2 := by sorry

end smallest_lattice_triangle_area_is_half_l819_81937


namespace smallest_base_perfect_square_four_is_solution_four_is_smallest_l819_81976

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 3 → (∃ n : ℕ, 3 * b + 4 = n^2) → b ≥ 4 :=
by sorry

theorem four_is_solution : 
  ∃ n : ℕ, 3 * 4 + 4 = n^2 :=
by sorry

theorem four_is_smallest : 
  ∀ b : ℕ, b > 3 ∧ (∃ n : ℕ, 3 * b + 4 = n^2) → b = 4 :=
by sorry

end smallest_base_perfect_square_four_is_solution_four_is_smallest_l819_81976


namespace age_problem_solution_l819_81963

/-- Represents the problem of finding when Anand's age was one-third of Bala's age -/
def age_problem (x : ℕ) : Prop :=
  let anand_current_age : ℕ := 15
  let bala_current_age : ℕ := anand_current_age + 10
  let anand_past_age : ℕ := anand_current_age - x
  let bala_past_age : ℕ := bala_current_age - x
  anand_past_age = bala_past_age / 3

/-- Theorem stating that 10 years ago, Anand's age was one-third of Bala's age -/
theorem age_problem_solution : age_problem 10 := by
  sorry

#check age_problem_solution

end age_problem_solution_l819_81963


namespace log5_of_125_l819_81980

-- Define the logarithm function for base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log5_of_125 : log5 125 = 3 := by
  sorry

end log5_of_125_l819_81980


namespace computational_not_basic_l819_81996

/-- The set of basic algorithmic statements -/
def BasicAlgorithmicStatements : Set String :=
  {"assignment", "conditional", "loop", "input", "output"}

/-- Proposition: Computational statements are not basic algorithmic statements -/
theorem computational_not_basic : "computational" ∉ BasicAlgorithmicStatements := by
  sorry

end computational_not_basic_l819_81996


namespace baker_remaining_cakes_l819_81956

theorem baker_remaining_cakes (total_cakes sold_cakes : ℕ) 
  (h1 : total_cakes = 155)
  (h2 : sold_cakes = 140) :
  total_cakes - sold_cakes = 15 := by
  sorry

end baker_remaining_cakes_l819_81956


namespace tourist_contact_probability_l819_81921

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 42

theorem tourist_contact_probability (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  contact_probability p =
    1 - (1 - p) ^ (6 * 7) :=
by sorry

end tourist_contact_probability_l819_81921


namespace geometric_sequence_product_l819_81974

def isGeometricSequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

theorem geometric_sequence_product (a b : ℝ) :
  isGeometricSequence 2 a b 16 → a * b = 32 := by
  sorry

end geometric_sequence_product_l819_81974


namespace negation_of_proposition_l819_81933

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x > 2, x^3 - 8 > 0) ↔ (∃ x > 2, x^3 - 8 ≤ 0) := by
  sorry

end negation_of_proposition_l819_81933


namespace trig_problem_l819_81954

theorem trig_problem (α : Real) (h : Real.tan α = 2) : 
  (2 * Real.sin α ^ 2 + 1) / Real.cos (2 * (α - π/4)) = 13/4 := by
  sorry

end trig_problem_l819_81954


namespace min_value_problem_l819_81964

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (8^a * 2^b)) :
  1/a + 2/b ≥ 5 + 2 * Real.sqrt 6 := by
sorry

end min_value_problem_l819_81964


namespace smallest_harmonic_sum_exceeding_10_l819_81981

def harmonic_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 1 / (i + 1 : ℚ))

theorem smallest_harmonic_sum_exceeding_10 :
  (∀ k < 12367, harmonic_sum k ≤ 10) ∧ harmonic_sum 12367 > 10 := by
  sorry

end smallest_harmonic_sum_exceeding_10_l819_81981


namespace sum_of_roots_l819_81959

theorem sum_of_roots (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ + 2 = 0) → (x₂^2 - 3*x₂ + 2 = 0) → x₁ + x₂ = 3 := by
  sorry

end sum_of_roots_l819_81959


namespace prove_z_value_l819_81936

theorem prove_z_value (z : ℝ) : 
  (Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt z / Real.sqrt 0.49 = 2.650793650793651) → 
  z = 1.00 := by
sorry

end prove_z_value_l819_81936


namespace ping_pong_theorem_l819_81992

/-- Represents the number of ping-pong balls in the box -/
def total_balls : ℕ := 7

/-- Represents the number of unused balls initially -/
def initial_unused : ℕ := 5

/-- Represents the number of used balls initially -/
def initial_used : ℕ := 2

/-- Represents the number of balls taken out and used -/
def balls_taken : ℕ := 3

/-- Represents the set of possible values for X (number of used balls after the process) -/
def possible_X : Set ℕ := {3, 4, 5}

/-- Represents the probability of X being 3 -/
def prob_X_3 : ℚ := 1/7

theorem ping_pong_theorem :
  (∀ x : ℕ, x ∈ possible_X ↔ (x ≥ initial_used ∧ x ≤ initial_used + balls_taken)) ∧
  (Nat.choose initial_unused 1 * Nat.choose initial_used 2 : ℚ) / Nat.choose total_balls balls_taken = prob_X_3 :=
by sorry

end ping_pong_theorem_l819_81992


namespace existence_of_index_l819_81945

theorem existence_of_index (n : ℕ) (x : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_x : ∀ i, i ∈ Finset.range (n + 1) → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i ∈ Finset.range n, x 1 * (1 - x (i + 1)) ≥ (1 / 4) * x 1 * (1 - x n) := by
  sorry

end existence_of_index_l819_81945


namespace percent_greater_l819_81960

theorem percent_greater (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x = 1.2 * y) (hyz : y = 1.2 * z) (hwx : w = 0.8 * x) :
  w = 1.152 * z := by
  sorry

end percent_greater_l819_81960


namespace mel_katherine_age_difference_l819_81970

/-- Given that Mel is younger than Katherine, and when Katherine is 24, Mel is 21,
    prove that Mel is 3 years younger than Katherine. -/
theorem mel_katherine_age_difference :
  ∀ (katherine_age mel_age : ℕ),
  katherine_age > mel_age →
  (katherine_age = 24 → mel_age = 21) →
  katherine_age - mel_age = 3 :=
by
  sorry

end mel_katherine_age_difference_l819_81970


namespace least_product_xy_l819_81902

theorem least_product_xy (x y : ℕ+) (h : (x : ℚ)⁻¹ + (3 * y : ℚ)⁻¹ = (6 : ℚ)⁻¹) :
  (∀ a b : ℕ+, (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (6 : ℚ)⁻¹ → x * y ≤ a * b) ∧ x * y = 48 :=
sorry

end least_product_xy_l819_81902


namespace totient_product_inequality_l819_81916

theorem totient_product_inequality (m n : ℕ) (h : m ≠ n) : 
  n * (Nat.totient n) ≠ m * (Nat.totient m) := by
  sorry

end totient_product_inequality_l819_81916


namespace soda_problem_l819_81984

theorem soda_problem (S : ℝ) : 
  (S / 2 + 2000 = S - (S / 2 - 2000)) → 
  ((S / 2 - 2000) / 2 + 2000 = S / 2 - 2000) → 
  S = 12000 := by
  sorry

end soda_problem_l819_81984


namespace f_of_2_equals_3_l819_81938

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - 1

-- State the theorem
theorem f_of_2_equals_3 : f 2 = 3 := by sorry

end f_of_2_equals_3_l819_81938


namespace special_ellipse_property_l819_81931

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  center : ℝ × ℝ := (0, 0)
  focus_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ
  ecc_eq : eccentricity = Real.sqrt (6/3)
  point_eq : passes_through = (Real.sqrt 5, 0)

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  fixed_point : ℝ × ℝ
  point_eq : fixed_point = (-1, 0)

/-- Intersection points of the line with the ellipse -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  midpoint_x : ℝ
  mid_eq : midpoint_x = -1/2

/-- The theorem statement -/
theorem special_ellipse_property
  (e : SpecialEllipse) (l : IntersectingLine) (p : IntersectionPoints) :
  ∃ (M : ℝ × ℝ), M.1 = -7/3 ∧ M.2 = 0 ∧
  (∀ (A B : ℝ × ℝ), 
    ((A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2)) = 4/9) :=
sorry

end special_ellipse_property_l819_81931


namespace complex_equation_solution_l819_81975

theorem complex_equation_solution (z : ℂ) :
  (1 + Complex.I) * z = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l819_81975


namespace complex_math_expression_equals_35_l819_81928

theorem complex_math_expression_equals_35 :
  ((9^2 + (3^3 - 1) * 4^2) % 6 : ℕ) * Real.sqrt 49 + (15 - 3 * 5) = 35 := by
  sorry

end complex_math_expression_equals_35_l819_81928


namespace interval_intersection_l819_81967

-- Define the conditions
def condition1 (x : ℝ) : Prop := 2 < 5 * x ∧ 5 * x < 3
def condition2 (x : ℝ) : Prop := 4 < 7 * x ∧ 7 * x < 6

-- Define the theorem
theorem interval_intersection :
  ∀ x : ℝ, (condition1 x ∧ condition2 x) ↔ (4/7 < x ∧ x < 3/5) :=
sorry

end interval_intersection_l819_81967


namespace derivative_at_negative_one_l819_81947

/-- Given f(x) = (1/3)x³ + 2x + 1, prove that f'(-1) = 3 -/
theorem derivative_at_negative_one (f : ℝ → ℝ) (hf : ∀ x, f x = (1/3) * x^3 + 2*x + 1) :
  (deriv f) (-1) = 3 := by
  sorry

end derivative_at_negative_one_l819_81947


namespace imaginary_part_of_product_l819_81913

def complex_mul (a b c d : ℝ) : ℂ :=
  (a * c - b * d : ℝ) + (a * d + b * c : ℝ) * Complex.I

theorem imaginary_part_of_product :
  let z₁ : ℂ := 1 - Complex.I
  let z₂ : ℂ := 2 + 4 * Complex.I
  Complex.im (z₁ * z₂) = 2 := by sorry

end imaginary_part_of_product_l819_81913


namespace smallest_digit_for_divisibility_l819_81908

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def number_with_d (d : ℕ) : ℕ := 563000 + d * 100 + 4

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by_9 (number_with_d d) ∧
    ∀ (d' : ℕ), d' < d → ¬(is_divisible_by_9 (number_with_d d')) :=
by sorry

end smallest_digit_for_divisibility_l819_81908


namespace newton_county_population_l819_81998

theorem newton_county_population (num_cities : ℕ) (lower_bound upper_bound : ℝ) :
  num_cities = 20 →
  lower_bound = 4500 →
  upper_bound = 5000 →
  let avg_population := (lower_bound + upper_bound) / 2
  num_cities * avg_population = 95000 := by
  sorry

end newton_county_population_l819_81998


namespace graduating_class_boys_count_l819_81918

theorem graduating_class_boys_count (total : ℕ) (difference : ℕ) (boys : ℕ) : 
  total = 345 →
  difference = 69 →
  total = boys + (boys + difference) →
  boys = 138 := by
sorry

end graduating_class_boys_count_l819_81918


namespace specific_parallelepiped_volume_l819_81988

/-- A right parallelepiped with a parallelogram base -/
structure RightParallelepiped where
  /-- Length of one side of the base -/
  sideA : ℝ
  /-- Length of the other side of the base -/
  sideB : ℝ
  /-- Angle between the sides of the base in radians -/
  baseAngle : ℝ
  /-- The smaller diagonal of the parallelepiped -/
  smallerDiagonal : ℝ

/-- The volume of the right parallelepiped -/
def volume (p : RightParallelepiped) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific parallelepiped -/
theorem specific_parallelepiped_volume :
  ∃ (p : RightParallelepiped),
    p.sideA = 3 ∧
    p.sideB = 4 ∧
    p.baseAngle = 2 * π / 3 ∧
    p.smallerDiagonal = Real.sqrt (p.sideA ^ 2 + p.sideB ^ 2 - 2 * p.sideA * p.sideB * Real.cos p.baseAngle) ∧
    volume p = 36 * Real.sqrt 2 :=
  sorry

end specific_parallelepiped_volume_l819_81988


namespace ellipse_standard_equation_parabola_standard_equation_l819_81994

-- Ellipse
def ellipse_equation (x y : ℝ) := x^2 / 25 + y^2 / 9 = 1

theorem ellipse_standard_equation 
  (foci_on_x_axis : Bool) 
  (major_axis_length : ℝ) 
  (eccentricity : ℝ) :
  foci_on_x_axis ∧ 
  major_axis_length = 10 ∧ 
  eccentricity = 4/5 →
  ∀ x y : ℝ, ellipse_equation x y :=
sorry

-- Parabola
def parabola_equation (x y : ℝ) := x^2 = -8*y

theorem parabola_standard_equation 
  (vertex : ℝ × ℝ) 
  (directrix : ℝ → ℝ) :
  vertex = (0, 0) ∧ 
  (∀ x : ℝ, directrix x = 2) →
  ∀ x y : ℝ, parabola_equation x y :=
sorry

end ellipse_standard_equation_parabola_standard_equation_l819_81994


namespace sequence_sum_l819_81925

/-- Given a sequence {a_n} where the sum of its first n terms S_n = n^2,
    and a sequence {b_n} defined as b_n = 2^(a_n),
    prove that the sum of the first n terms of {b_n}, T_n, is (2/3) * (4^n - 1) -/
theorem sequence_sum (n : ℕ) (a b : ℕ → ℕ) (S T : ℕ → ℚ)
  (h_S : ∀ k, S k = k^2)
  (h_b : ∀ k, b k = 2^(a k)) :
  T n = 2/3 * (4^n - 1) := by
  sorry

end sequence_sum_l819_81925


namespace theresas_work_hours_l819_81985

theorem theresas_work_hours : ∀ (final_week_hours : ℕ),
  final_week_hours ≥ 10 →
  (7 + 10 + 8 + 11 + 9 + 7 + final_week_hours) / 7 = 9 →
  final_week_hours = 11 := by
sorry

end theresas_work_hours_l819_81985


namespace trains_crossing_time_l819_81917

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 300)
  (h2 : length2 = 400)
  (h3 : speed1 = 36 * 1000 / 3600)
  (h4 : speed2 = 18 * 1000 / 3600)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0) :
  (length1 + length2) / (speed1 + speed2) = 46.67 := by
  sorry

end trains_crossing_time_l819_81917


namespace possible_values_of_x_l819_81910

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem possible_values_of_x (x : ℝ) :
  A x ∩ B x = B x → x = 0 ∨ x = -2 ∨ x = 2 := by
  sorry

end possible_values_of_x_l819_81910


namespace circle_properties_l819_81948

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 6*y - 3 = 0

/-- The center of the circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- The radius of the circle -/
def CircleRadius : ℝ := 4

/-- Theorem stating that the given equation represents a circle with the specified center and radius -/
theorem circle_properties :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = CircleRadius^2 := by
  sorry

end circle_properties_l819_81948


namespace subtract_two_percent_l819_81966

theorem subtract_two_percent (a : ℝ) : a - (0.02 * a) = 0.98 * a := by
  sorry

end subtract_two_percent_l819_81966


namespace class_size_proof_l819_81934

theorem class_size_proof :
  ∀ n : ℕ,
  20 < n ∧ n < 30 →
  ∃ x : ℕ, n = 3 * x →
  ∃ y : ℕ, n = 4 * y →
  n = 24 := by
sorry

end class_size_proof_l819_81934


namespace diagonals_not_bisect_equiv_not_p_l819_81939

-- Define the proposition "The diagonals of a trapezoid bisect each other"
def diagonals_bisect_each_other : Prop := sorry

-- Define the proposition "The diagonals of a trapezoid do not bisect each other"
def diagonals_do_not_bisect_each_other : Prop := ¬diagonals_bisect_each_other

-- Theorem stating that the given proposition is equivalent to "not p"
theorem diagonals_not_bisect_equiv_not_p : 
  diagonals_do_not_bisect_each_other ↔ ¬diagonals_bisect_each_other :=
sorry

end diagonals_not_bisect_equiv_not_p_l819_81939


namespace remainder_of_M_mod_50_l819_81943

def M : ℕ := sorry -- Definition of M as concatenation of numbers from 1 to 49

theorem remainder_of_M_mod_50 : M % 50 = 49 := by sorry

end remainder_of_M_mod_50_l819_81943


namespace simplify_expression_l819_81920

theorem simplify_expression (x : ℝ) : x + 3 - 5*x + 6 + 7*x - 2 - 9*x + 8 = -6*x + 15 := by
  sorry

end simplify_expression_l819_81920


namespace decreasing_quadratic_function_m_range_l819_81935

/-- A function f(x) = mx^2 + (m-1)x + 1 is decreasing on (-∞, 1] if and only if m ∈ [0, 1/3] -/
theorem decreasing_quadratic_function_m_range (m : ℝ) : 
  (∀ x ≤ 1, ∀ y ≤ 1, x < y → m * x^2 + (m - 1) * x + 1 > m * y^2 + (m - 1) * y + 1) ↔ 
  0 ≤ m ∧ m ≤ 1/3 := by
sorry

end decreasing_quadratic_function_m_range_l819_81935


namespace pony_jeans_discount_rate_l819_81972

theorem pony_jeans_discount_rate 
  (fox_price : ℝ) 
  (pony_price : ℝ) 
  (total_savings : ℝ) 
  (fox_quantity : ℕ) 
  (pony_quantity : ℕ) 
  (total_discount_rate : ℝ) :
  fox_price = 15 →
  pony_price = 18 →
  total_savings = 8.55 →
  fox_quantity = 3 →
  pony_quantity = 2 →
  total_discount_rate = 22 →
  ∃ (fox_discount_rate : ℝ) (pony_discount_rate : ℝ),
    fox_discount_rate + pony_discount_rate = total_discount_rate ∧
    fox_quantity * (fox_price * fox_discount_rate / 100) + 
    pony_quantity * (pony_price * pony_discount_rate / 100) = total_savings ∧
    pony_discount_rate = 15 :=
by sorry

end pony_jeans_discount_rate_l819_81972


namespace cos_difference_of_zeros_l819_81999

open Real

theorem cos_difference_of_zeros (f g : ℝ → ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, f x = sin (2 * x - π / 3)) →
  (∀ x, g x = f x - 1 / 3) →
  g x₁ = 0 →
  g x₂ = 0 →
  x₁ ≠ x₂ →
  0 ≤ x₁ ∧ x₁ ≤ π →
  0 ≤ x₂ ∧ x₂ ≤ π →
  cos (x₁ - x₂) = 1 / 3 := by
sorry

end cos_difference_of_zeros_l819_81999


namespace line_inclination_angle_l819_81955

/-- The angle of inclination of a line passing through (0, 0) and (1, -1) is 135°. -/
theorem line_inclination_angle : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (t, -t)}
  let angle : ℝ := Real.arctan (-1) * (180 / Real.pi)
  angle = 135 := by sorry

end line_inclination_angle_l819_81955


namespace obtuse_triangle_side_range_l819_81930

/-- A triangle with sides a, a+2, and a+4 is obtuse if and only if 2 < a < 6 -/
theorem obtuse_triangle_side_range (a : ℝ) : 
  (∃ (x y z : ℝ), x = a ∧ y = a + 2 ∧ z = a + 4 ∧ 
   x > 0 ∧ y > 0 ∧ z > 0 ∧
   x + y > z ∧ x + z > y ∧ y + z > x ∧
   z^2 > x^2 + y^2) ↔ 
  (2 < a ∧ a < 6) :=
sorry

end obtuse_triangle_side_range_l819_81930


namespace right_triangle_identification_l819_81952

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_identification :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 5 12 13 ∧
  is_right_triangle 6 8 10 ∧
  ¬ is_right_triangle 4 6 8 :=
sorry

end right_triangle_identification_l819_81952


namespace incorrect_statement_l819_81905

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (planesPerp : Plane → Plane → Prop)

-- State the theorem
theorem incorrect_statement 
  (α β : Plane) (m n : Line) : 
  ¬(perpendicular m n ∧ perpendicularToPlane m α ∧ parallelToPlane n β → planesPerp α β) :=
by sorry

end incorrect_statement_l819_81905


namespace boat_round_trip_time_l819_81982

theorem boat_round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 9)
  (h2 : stream_speed = 6)
  (h3 : distance = 170)
  : (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = 68 := by
  sorry

end boat_round_trip_time_l819_81982


namespace contest_probability_l819_81941

theorem contest_probability (n : ℕ) : n = 4 ↔ n = Nat.succ (Nat.floor (Real.log 10 / Real.log 2)) := by sorry

end contest_probability_l819_81941


namespace train_tunnel_time_l819_81986

/-- Proves that a train of given length and speed passing through a tunnel of given length takes 1 minute to completely clear the tunnel. -/
theorem train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length_km : ℝ) :
  train_length = 100 →
  train_speed_kmh = 72 →
  tunnel_length_km = 1.1 →
  (tunnel_length_km * 1000 + train_length) / (train_speed_kmh * 1000 / 60) = 1 := by
  sorry

end train_tunnel_time_l819_81986


namespace pasture_rental_problem_l819_81900

/-- Calculate the rent share for a person based on their oxen usage and total pasture usage -/
def calculate_rent_share (total_rent : ℚ) (person_oxen_months : ℕ) (total_oxen_months : ℕ) : ℚ :=
  (person_oxen_months : ℚ) * total_rent / (total_oxen_months : ℚ)

/-- Represents the pasture rental problem -/
theorem pasture_rental_problem :
  let total_rent : ℚ := 750
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let c_months := 3
  let d_oxen := 18
  let d_months := 6
  let e_oxen := 20
  let e_months := 4
  let f_oxen := 25
  let f_months := 2
  let total_oxen_months := a_oxen * a_months + b_oxen * b_months + c_oxen * c_months +
                           d_oxen * d_months + e_oxen * e_months + f_oxen * f_months
  let c_share := calculate_rent_share total_rent (c_oxen * c_months) total_oxen_months
  c_share = 81.75 := by
    sorry

end pasture_rental_problem_l819_81900


namespace range_of_f_range_of_m_l819_81953

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

-- Theorem for the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-3) 3 :=
sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x, f x + 2 * m - 1 ≥ 0) ↔ m ≥ 2 :=
sorry

end range_of_f_range_of_m_l819_81953


namespace symmetric_complex_quotient_l819_81987

/-- Two complex numbers are symmetric about the y-axis if their real parts are negatives of each other and their imaginary parts are equal -/
def symmetric_about_y_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem symmetric_complex_quotient (z₁ z₂ : ℂ) :
  symmetric_about_y_axis z₁ z₂ → z₁ = 1 + I → z₂ / z₁ = I :=
by
  sorry

#check symmetric_complex_quotient

end symmetric_complex_quotient_l819_81987


namespace sum_of_solutions_eq_sixteen_l819_81924

theorem sum_of_solutions_eq_sixteen : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 8)^2 = 36 ∧ (x₂ - 8)^2 = 36 ∧ x₁ + x₂ = 16 := by
  sorry

end sum_of_solutions_eq_sixteen_l819_81924


namespace marys_max_earnings_l819_81907

/-- Calculates the maximum weekly earnings for Mary given her work conditions --/
def max_weekly_earnings (max_hours : ℕ) (regular_rate : ℚ) (overtime_rate : ℚ) (higher_overtime_rate : ℚ) : ℚ :=
  let regular_pay := regular_rate * 40
  let overtime_pay := overtime_rate * 10
  let higher_overtime_pay := higher_overtime_rate * 10
  regular_pay + overtime_pay + higher_overtime_pay

/-- Theorem stating that Mary's maximum weekly earnings are $675 --/
theorem marys_max_earnings :
  let max_hours : ℕ := 60
  let regular_rate : ℚ := 10
  let overtime_rate : ℚ := regular_rate * (1 + 1/4)
  let higher_overtime_rate : ℚ := regular_rate * (1 + 1/2)
  max_weekly_earnings max_hours regular_rate overtime_rate higher_overtime_rate = 675 := by
  sorry

end marys_max_earnings_l819_81907


namespace ratio_comparison_l819_81901

theorem ratio_comparison : ∀ (a b : ℕ), 
  a = 6 ∧ b = 7 →
  ∃ (x : ℕ), x = 3 ∧
  (a - x : ℚ) / (b - x : ℚ) < 3 / 4 ∧
  ∀ (y : ℕ), y < x →
  (a - y : ℚ) / (b - y : ℚ) ≥ 3 / 4 :=
by sorry

end ratio_comparison_l819_81901


namespace min_value_theorem_l819_81993

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) 
  (heq : a^2 * (b + 4*b^2 + 2*a^2) = 8 - 2*b^3) :
  ∃ (m : ℝ), m = 8 * Real.sqrt 3 ∧ 
  (∀ (x y : ℝ), x > 0 → y > 1 → x^2 * (y + 4*y^2 + 2*x^2) = 8 - 2*y^3 → 
    8*x^2 + 4*y^2 + 3*y ≥ m) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 1 ∧ x^2 * (y + 4*y^2 + 2*x^2) = 8 - 2*y^3 ∧ 
    8*x^2 + 4*y^2 + 3*y = m) :=
by
  sorry

end min_value_theorem_l819_81993


namespace least_positive_integer_with_remainder_one_l819_81926

theorem least_positive_integer_with_remainder_one (n : ℕ) : n = 421 ↔ 
  (n > 1) ∧ 
  (∀ d ∈ ({3, 4, 5, 6, 7, 10, 12} : Set ℕ), n % d = 1) ∧
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({3, 4, 5, 6, 7, 10, 12} : Set ℕ), m % d = 1) → m ≥ n) :=
by sorry

end least_positive_integer_with_remainder_one_l819_81926


namespace exists_circle_with_n_points_l819_81983

/-- A function that counts the number of lattice points strictly inside a circle -/
def count_lattice_points (center : ℝ × ℝ) (radius : ℝ) : ℕ :=
  sorry

/-- Theorem stating that for any non-negative integer, there exists a circle containing exactly that many lattice points -/
theorem exists_circle_with_n_points (n : ℕ) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), count_lattice_points center radius = n :=
sorry

end exists_circle_with_n_points_l819_81983


namespace increase_percentage_theorem_l819_81973

theorem increase_percentage_theorem (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hM : M > 0) (hpq : q < p) :
  M * (1 + p / 100) * (1 + q / 100) > M ↔ (p > 0 ∧ q > 0) :=
by sorry

end increase_percentage_theorem_l819_81973


namespace angle_sum_proof_l819_81912

theorem angle_sum_proof (a b : Real) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2)
  (h1 : 4 * (Real.cos a)^3 - 3 * (Real.cos b)^3 = 2)
  (h2 : 4 * Real.cos (2*a) + 3 * Real.cos (2*b) = 1) :
  2*a + b = π/2 := by sorry

end angle_sum_proof_l819_81912


namespace pushups_total_l819_81904

/-- The number of push-ups Zachary and David did altogether -/
def total_pushups (zachary_pushups : ℕ) (david_extra_pushups : ℕ) : ℕ :=
  zachary_pushups + (zachary_pushups + david_extra_pushups)

/-- Theorem stating that given the conditions, the total number of push-ups is 146 -/
theorem pushups_total : total_pushups 44 58 = 146 := by
  sorry

end pushups_total_l819_81904


namespace sarahs_trip_length_l819_81958

theorem sarahs_trip_length :
  ∀ (x : ℝ),
  (x / 4 : ℝ) + 15 + (x / 3 : ℝ) = x →
  x = 36 := by
sorry

end sarahs_trip_length_l819_81958


namespace mystery_number_l819_81914

theorem mystery_number (x : ℕ) (h : x + 45 = 92) : x = 47 := by
  sorry

end mystery_number_l819_81914


namespace expression_upper_bound_l819_81965

theorem expression_upper_bound (α β γ δ ε : ℝ) : 
  (1 - α) * Real.exp α + 
  (1 - β) * Real.exp (α + β) + 
  (1 - γ) * Real.exp (α + β + γ) + 
  (1 - δ) * Real.exp (α + β + γ + δ) + 
  (1 - ε) * Real.exp (α + β + γ + δ + ε) ≤ Real.exp 4 := by
  sorry

#check expression_upper_bound

end expression_upper_bound_l819_81965


namespace fred_change_theorem_l819_81922

/-- Calculates the change received after a purchase -/
def calculate_change (ticket_price : ℚ) (num_tickets : ℕ) (borrowed_movie_price : ℚ) (paid_amount : ℚ) : ℚ :=
  paid_amount - (ticket_price * num_tickets + borrowed_movie_price)

theorem fred_change_theorem :
  let ticket_price : ℚ := 8.25
  let num_tickets : ℕ := 4
  let borrowed_movie_price : ℚ := 9.50
  let paid_amount : ℚ := 50
  calculate_change ticket_price num_tickets borrowed_movie_price paid_amount = 7.50 := by
  sorry

#eval calculate_change 8.25 4 9.50 50

end fred_change_theorem_l819_81922


namespace total_octopus_legs_l819_81969

/-- The number of octopuses Sawyer saw -/
def num_octopuses : ℕ := 5

/-- The number of legs each octopus has -/
def legs_per_octopus : ℕ := 8

/-- The total number of octopus legs -/
def total_legs : ℕ := num_octopuses * legs_per_octopus

theorem total_octopus_legs : total_legs = 40 := by
  sorry

end total_octopus_legs_l819_81969


namespace parabola_directrix_l819_81979

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y = (x^2 - 4*x + 4) / 8

/-- The directrix equation -/
def directrix_eq (y : ℝ) : Prop := y = -1/4

/-- Theorem: The directrix of the given parabola is y = -1/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_eq x y → ∃ y_d : ℝ, directrix_eq y_d ∧ 
  (∀ x' y' : ℝ, parabola_eq x' y' → 
    (x' - x)^2 + (y' - y)^2 = (y' - y_d)^2) :=
sorry

end parabola_directrix_l819_81979


namespace solve_system_l819_81927

theorem solve_system (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h1 : 1/x + 1/y = 3/2) (h2 : x*y = 9) : y = 6 := by
  sorry

end solve_system_l819_81927


namespace cleaning_time_ratio_with_help_cleaning_time_ratio_l819_81971

/-- Represents the grove of trees -/
structure Grove where
  rows : Nat
  columns : Nat

/-- Represents the time spent cleaning trees -/
structure CleaningTime where
  minutes : Nat

theorem cleaning_time_ratio_with_help (g : Grove) 
  (time_per_tree_without_help : Nat) 
  (total_time_with_help : CleaningTime) : 
  2 * (total_time_with_help.minutes / (g.rows * g.columns)) = time_per_tree_without_help :=
by
  sorry

#check cleaning_time_ratio_with_help

/-- Main theorem that proves the ratio of cleaning time with help to without help is 1:2 -/
theorem cleaning_time_ratio (g : Grove) 
  (time_per_tree_without_help : Nat) 
  (total_time_with_help : CleaningTime) : 
  (total_time_with_help.minutes / (g.rows * g.columns)) / time_per_tree_without_help = 1 / 2 :=
by
  sorry

#check cleaning_time_ratio

end cleaning_time_ratio_with_help_cleaning_time_ratio_l819_81971


namespace discount_percentages_l819_81944

/-- Merchant's markup percentage -/
def markup : ℚ := 75 / 100

/-- Profit percentage for 65 items -/
def profit65 : ℚ := 575 / 1000

/-- Profit percentage for 30 items -/
def profit30 : ℚ := 525 / 1000

/-- Profit percentage for 5 items -/
def profit5 : ℚ := 48 / 100

/-- Calculate discount percentage given profit percentage -/
def calcDiscount (profit : ℚ) : ℚ :=
  (markup - profit) / (1 + markup) * 100

/-- Round to nearest integer -/
def roundToInt (q : ℚ) : ℤ :=
  (q + 1/2).floor

/-- Theorem stating the discount percentages -/
theorem discount_percentages :
  let x := roundToInt (calcDiscount profit5)
  let y := roundToInt (calcDiscount profit30)
  let z := roundToInt (calcDiscount profit65)
  x = 15 ∧ y = 13 ∧ z = 10 ∧
  (5 ≤ x ∧ x ≤ 25) ∧ (5 ≤ y ∧ y ≤ 25) ∧ (5 ≤ z ∧ z ≤ 25) :=
by sorry


end discount_percentages_l819_81944


namespace tank_insulation_cost_l819_81911

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the total cost of insulation for a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

theorem tank_insulation_cost :
  let l : ℝ := 6
  let w : ℝ := 3
  let h : ℝ := 2
  let costPerSqFt : ℝ := 20
  insulationCost l w h costPerSqFt = 1440 := by
  sorry

#eval insulationCost 6 3 2 20

end tank_insulation_cost_l819_81911
