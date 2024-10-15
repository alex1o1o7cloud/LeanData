import Mathlib

namespace NUMINAMATH_CALUDE_variance_or_std_dev_measures_stability_l2528_252809

-- Define a type for exam scores
def ExamScore := ℝ

-- Define a type for a set of exam scores
def ExamScores := List ExamScore

-- Define a function to calculate variance
noncomputable def variance (scores : ExamScores) : ℝ := sorry

-- Define a function to calculate standard deviation
noncomputable def standardDeviation (scores : ExamScores) : ℝ := sorry

-- Define a measure of stability
noncomputable def stabilityMeasure (scores : ExamScores) : ℝ := sorry

-- Theorem stating that variance or standard deviation is the most appropriate measure of stability
theorem variance_or_std_dev_measures_stability (scores : ExamScores) :
  (stabilityMeasure scores = variance scores) ∨ (stabilityMeasure scores = standardDeviation scores) :=
sorry

end NUMINAMATH_CALUDE_variance_or_std_dev_measures_stability_l2528_252809


namespace NUMINAMATH_CALUDE_cylinder_volume_l2528_252858

/-- The volume of a cylinder with base diameter and height both equal to 3 is (27/4)π. -/
theorem cylinder_volume (d h : ℝ) (hd : d = 3) (hh : h = 3) :
  let r := d / 2
  π * r^2 * h = (27 / 4) * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l2528_252858


namespace NUMINAMATH_CALUDE_leaders_photo_theorem_l2528_252895

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := n.factorial

/-- The number of ways to choose k objects from n distinct objects and arrange them. -/
def arrangements (n k : ℕ) : ℕ := 
  if k ≤ n then (permutations n) / (permutations (n - k)) else 0

/-- The number of arrangements for the leaders' photo. -/
def leaders_photo_arrangements : ℕ := 
  (arrangements 2 1) * (arrangements 18 18)

theorem leaders_photo_theorem : 
  leaders_photo_arrangements = (arrangements 2 1) * (arrangements 18 18) := by
  sorry

end NUMINAMATH_CALUDE_leaders_photo_theorem_l2528_252895


namespace NUMINAMATH_CALUDE_perpendicular_line_correct_l2528_252810

/-- The slope of the given line x - 2y + 3 = 0 -/
def m₁ : ℚ := 1 / 2

/-- The point P through which the perpendicular line passes -/
def P : ℚ × ℚ := (-1, 3)

/-- The equation of the perpendicular line in the form ax + by + c = 0 -/
def perpendicular_line (x y : ℚ) : Prop := 2 * x + y - 1 = 0

theorem perpendicular_line_correct :
  /- The line passes through point P -/
  perpendicular_line P.1 P.2 ∧
  /- The line is perpendicular to x - 2y + 3 = 0 -/
  (∃ m₂ : ℚ, m₂ * m₁ = -1 ∧
    ∀ x y : ℚ, perpendicular_line x y ↔ y - P.2 = m₂ * (x - P.1)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_correct_l2528_252810


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2528_252807

theorem complex_equation_solution (z : ℂ) 
  (h : 10 * Complex.normSq z = 2 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 16) + 40) : 
  z + 9 / z = -3 / 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2528_252807


namespace NUMINAMATH_CALUDE_min_value_theorem_l2528_252856

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ (3 * x + 4 * y = 5 ↔ x = 1 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2528_252856


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2528_252834

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2528_252834


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2528_252804

theorem quadratic_equation_properties (a b c : ℝ) (h : a ≠ 0) :
  -- Statement ①
  (a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
  -- Statement ②
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + c = 0 ∧ a*y^2 + c = 0 →
    ∃ u v : ℝ, u ≠ v ∧ a*u^2 + b*u + c = 0 ∧ a*v^2 + b*v + c = 0) ∧
  -- Statement ④
  (∀ x₀ : ℝ, a*x₀^2 + b*x₀ + c = 0 → b^2 - 4*a*c = (2*a*x₀ + b)^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2528_252804


namespace NUMINAMATH_CALUDE_cos_product_from_sum_relations_l2528_252833

theorem cos_product_from_sum_relations (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 0.6) 
  (h2 : Real.cos x + Real.cos y = 0.8) : 
  Real.cos x * Real.cos y = -11/100 := by
sorry

end NUMINAMATH_CALUDE_cos_product_from_sum_relations_l2528_252833


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2528_252872

/-- The modulus of the complex number z = (1+3i)/(1-i) is equal to √5 -/
theorem modulus_of_complex_fraction : 
  let z : ℂ := (1 + 3*I) / (1 - I)
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2528_252872


namespace NUMINAMATH_CALUDE_odd_function_decomposition_l2528_252853

/-- An odd function. -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A periodic function with period T. -/
def PeriodicFunction (φ : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, φ (x + T) = φ x

/-- A linear function. -/
def LinearFunction (g : ℝ → ℝ) : Prop :=
  ∃ k h : ℝ, ∀ x, g x = k * x + h

/-- A function with a center of symmetry at (a, b). -/
def HasCenterOfSymmetry (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) + f (a - x) = 2 * b

theorem odd_function_decomposition (f : ℝ → ℝ) :
  OddFunction f →
  (∃ φ g : ℝ → ℝ, ∃ T : ℝ, T ≠ 0 ∧
    PeriodicFunction φ T ∧
    LinearFunction g ∧
    (∀ x, f x = φ x + g x)) ↔
  (∃ a b : ℝ, (a, b) ≠ (0, 0) ∧ HasCenterOfSymmetry f a b ∧ ∃ k : ℝ, b = k * a) :=
sorry

end NUMINAMATH_CALUDE_odd_function_decomposition_l2528_252853


namespace NUMINAMATH_CALUDE_series_one_over_sqrt_n_diverges_l2528_252873

theorem series_one_over_sqrt_n_diverges :
  ¬ Summable (fun n : ℕ => 1 / Real.sqrt n) := by sorry

end NUMINAMATH_CALUDE_series_one_over_sqrt_n_diverges_l2528_252873


namespace NUMINAMATH_CALUDE_paper_I_maximum_mark_l2528_252867

theorem paper_I_maximum_mark :
  ∃ (M : ℕ),
    (M : ℚ) * (55 : ℚ) / (100 : ℚ) = (65 : ℚ) + (35 : ℚ) ∧
    M = 182 := by
  sorry

end NUMINAMATH_CALUDE_paper_I_maximum_mark_l2528_252867


namespace NUMINAMATH_CALUDE_zeros_of_cosine_minus_one_l2528_252864

theorem zeros_of_cosine_minus_one (ω : ℝ) : 
  (ω > 0) →
  (∃ (x₁ x₂ x₃ : ℝ), 
    (x₁ ∈ Set.Icc 0 (2 * Real.pi)) ∧ 
    (x₂ ∈ Set.Icc 0 (2 * Real.pi)) ∧ 
    (x₃ ∈ Set.Icc 0 (2 * Real.pi)) ∧ 
    (x₁ ≠ x₂) ∧ (x₂ ≠ x₃) ∧ (x₁ ≠ x₃) ∧
    (Real.cos (ω * x₁) = 1) ∧ 
    (Real.cos (ω * x₂) = 1) ∧ 
    (Real.cos (ω * x₃) = 1) ∧
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.cos (ω * x) = 1 → (x = x₁ ∨ x = x₂ ∨ x = x₃))) ↔
  (2 ≤ ω ∧ ω < 3) :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_cosine_minus_one_l2528_252864


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2528_252836

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  Real.sqrt (4 * x + 1) + Real.sqrt (4 * y + 1) + Real.sqrt (4 * z + 1) ≤ 3 * Real.sqrt (35 / 3) ∧
  ∃ x y z, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 8 ∧
    Real.sqrt (4 * x + 1) + Real.sqrt (4 * y + 1) + Real.sqrt (4 * z + 1) = 3 * Real.sqrt (35 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2528_252836


namespace NUMINAMATH_CALUDE_expression_change_l2528_252843

/-- The change in the expression 2x^2 + 5 when x changes by ±b -/
theorem expression_change (x b : ℝ) (h : b > 0) :
  let f : ℝ → ℝ := λ t => 2 * t^2 + 5
  abs (f (x + b) - f x) = 2 * b * (2 * x + b) ∧
  abs (f (x - b) - f x) = 2 * b * (2 * x + b) :=
by sorry

end NUMINAMATH_CALUDE_expression_change_l2528_252843


namespace NUMINAMATH_CALUDE_perimeter_to_hypotenuse_ratio_l2528_252845

/-- Right triangle ABC with altitude CD to hypotenuse AB and circle ω with CD as diameter -/
structure RightTriangleWithCircle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point D on hypotenuse AB -/
  D : ℝ × ℝ
  /-- Center of circle ω -/
  O : ℝ × ℝ
  /-- Point I outside the triangle -/
  I : ℝ × ℝ
  /-- ABC is a right triangle with right angle at C -/
  is_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  /-- AC = 15 -/
  ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 15
  /-- BC = 20 -/
  bc_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 20
  /-- CD is perpendicular to AB -/
  cd_perpendicular : (D.1 - C.1) * (B.1 - A.1) + (D.2 - C.2) * (B.2 - A.2) = 0
  /-- D is on AB -/
  d_on_ab : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  /-- O is the midpoint of CD -/
  o_midpoint : O = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  /-- AI is tangent to circle ω -/
  ai_tangent : Real.sqrt ((I.1 - A.1)^2 + (I.2 - A.2)^2) * Real.sqrt ((I.1 - O.1)^2 + (I.2 - O.2)^2) = (I.1 - A.1) * (I.1 - O.1) + (I.2 - A.2) * (I.2 - O.2)
  /-- BI is tangent to circle ω -/
  bi_tangent : Real.sqrt ((I.1 - B.1)^2 + (I.2 - B.2)^2) * Real.sqrt ((I.1 - O.1)^2 + (I.2 - O.2)^2) = (I.1 - B.1) * (I.1 - O.1) + (I.2 - B.2) * (I.2 - O.2)

/-- The ratio of the perimeter of triangle ABI to the length of AB is 5/2 -/
theorem perimeter_to_hypotenuse_ratio (t : RightTriangleWithCircle) :
  let ab_length := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let ai_length := Real.sqrt ((t.I.1 - t.A.1)^2 + (t.I.2 - t.A.2)^2)
  let bi_length := Real.sqrt ((t.I.1 - t.B.1)^2 + (t.I.2 - t.B.2)^2)
  (ai_length + bi_length + ab_length) / ab_length = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_to_hypotenuse_ratio_l2528_252845


namespace NUMINAMATH_CALUDE_number_of_products_l2528_252802

/-- Prove that the number of products is 20 given the fixed cost, marginal cost, and total cost. -/
theorem number_of_products (fixed_cost marginal_cost total_cost : ℚ)
  (h1 : fixed_cost = 12000)
  (h2 : marginal_cost = 200)
  (h3 : total_cost = 16000)
  (h4 : total_cost = fixed_cost + marginal_cost * n) :
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_products_l2528_252802


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_min_mn_l2528_252868

/-- Given two positive real numbers m and n satisfying 1/m + 2/n = 1,
    the eccentricity of the ellipse x²/m² + y²/n² = 1 is √3/2
    when mn takes its minimum value. -/
theorem ellipse_eccentricity_min_mn (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (h : 1/m + 2/n = 1) :
  let e := Real.sqrt (1 - (min m n)^2 / (max m n)^2)
  ∃ (min_mn : ℝ), (∀ m' n' : ℝ, m' > 0 → n' > 0 → 1/m' + 2/n' = 1 → m' * n' ≥ min_mn) ∧
    (m * n = min_mn → e = Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_min_mn_l2528_252868


namespace NUMINAMATH_CALUDE_experiment_A_not_control_based_l2528_252861

-- Define the type for experiments
inductive Experiment
| A
| B
| C
| D

-- Define a predicate for experiments designed based on the principle of control
def is_control_based (e : Experiment) : Prop :=
  match e with
  | Experiment.A => False
  | _ => True

-- Theorem statement
theorem experiment_A_not_control_based :
  is_control_based Experiment.B ∧
  is_control_based Experiment.C ∧
  is_control_based Experiment.D →
  ¬is_control_based Experiment.A :=
by
  sorry

end NUMINAMATH_CALUDE_experiment_A_not_control_based_l2528_252861


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2528_252852

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x > 0 ∧ (1057 + x) % 23 = 0 ∧ ∀ y : ℕ, y > 0 ∧ (1057 + y) % 23 = 0 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2528_252852


namespace NUMINAMATH_CALUDE_solar_project_profit_l2528_252891

/-- Represents the net profit of a solar power generation project -/
def net_profit (n : ℕ+) : ℤ :=
  n - (4 * n^2 + 20 * n) - 144

/-- Theorem stating the net profit expression and when the project starts to make profit -/
theorem solar_project_profit :
  (∀ n : ℕ+, net_profit n = -4 * n^2 + 80 * n - 144) ∧
  (∀ n : ℕ+, net_profit n > 0 ↔ n ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_solar_project_profit_l2528_252891


namespace NUMINAMATH_CALUDE_roberts_balls_theorem_l2528_252824

/-- Calculates the final number of balls Robert has -/
def robertsFinalBalls (robertsInitial : ℕ) (timsTotal : ℕ) (jennysTotal : ℕ) : ℕ :=
  robertsInitial + timsTotal / 2 + jennysTotal / 3

theorem roberts_balls_theorem :
  robertsFinalBalls 25 40 60 = 65 := by
  sorry

end NUMINAMATH_CALUDE_roberts_balls_theorem_l2528_252824


namespace NUMINAMATH_CALUDE_milk_price_increase_percentage_l2528_252819

def lowest_price : ℝ := 16
def highest_price : ℝ := 22

theorem milk_price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_increase_percentage_l2528_252819


namespace NUMINAMATH_CALUDE_pullups_calculation_l2528_252879

/-- Calculates the number of pull-ups done per visit given the total pull-ups per week and visits per day -/
def pullups_per_visit (total_pullups : ℕ) (visits_per_day : ℕ) : ℚ :=
  total_pullups / (visits_per_day * 7)

/-- Theorem: If a person does 70 pull-ups per week and visits a room 5 times per day, 
    then the number of pull-ups done each visit is 2 -/
theorem pullups_calculation :
  pullups_per_visit 70 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pullups_calculation_l2528_252879


namespace NUMINAMATH_CALUDE_root_implies_h_value_l2528_252841

theorem root_implies_h_value (h : ℝ) : 
  (3 : ℝ)^3 + h * 3 + 5 = 0 → h = -32/3 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_h_value_l2528_252841


namespace NUMINAMATH_CALUDE_arm_wrestling_tournament_rounds_l2528_252869

/-- Represents the rules and structure of the arm wrestling tournament. -/
structure TournamentRules where
  num_athletes : ℕ
  max_point_diff : ℕ

/-- Calculates the minimum number of rounds required to determine a sole leader. -/
def min_rounds_required (rules : TournamentRules) : ℕ :=
  sorry

/-- Theorem stating that for a tournament with 510 athletes and the given rules,
    the minimum number of rounds required is 9. -/
theorem arm_wrestling_tournament_rounds 
  (rules : TournamentRules) 
  (h1 : rules.num_athletes = 510) 
  (h2 : rules.max_point_diff = 1) : 
  min_rounds_required rules = 9 := by
  sorry

end NUMINAMATH_CALUDE_arm_wrestling_tournament_rounds_l2528_252869


namespace NUMINAMATH_CALUDE_clothing_price_problem_l2528_252863

theorem clothing_price_problem (total_spent : ℕ) (num_pieces : ℕ) (price1 : ℕ) (price2 : ℕ) 
  (h1 : total_spent = 610)
  (h2 : num_pieces = 7)
  (h3 : price1 = 49)
  (h4 : price2 = 81)
  : (total_spent - price1 - price2) / (num_pieces - 2) = 96 := by
  sorry

end NUMINAMATH_CALUDE_clothing_price_problem_l2528_252863


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2528_252811

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) * (x + 1) ≥ 0}
def B : Set ℝ := {x | x < -4/5}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2528_252811


namespace NUMINAMATH_CALUDE_problem_statement_l2528_252893

theorem problem_statement :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ∧
  (∀ x : ℝ, 0 < x → x < π / 2 → x > Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2528_252893


namespace NUMINAMATH_CALUDE_min_yellow_fraction_l2528_252876

/-- Represents a cube with its edge length and number of blue and yellow subcubes. -/
structure Cube where
  edge_length : ℕ
  blue_cubes : ℕ
  yellow_cubes : ℕ

/-- Calculates the minimum yellow surface area for a given cube configuration. -/
def min_yellow_surface_area (c : Cube) : ℕ :=
  sorry

/-- Calculates the total surface area of a cube. -/
def total_surface_area (c : Cube) : ℕ :=
  6 * c.edge_length * c.edge_length

/-- The main theorem stating the minimum fraction of yellow surface area. -/
theorem min_yellow_fraction (c : Cube) 
  (h1 : c.edge_length = 4)
  (h2 : c.blue_cubes = 48)
  (h3 : c.yellow_cubes = 16)
  (h4 : c.blue_cubes + c.yellow_cubes = c.edge_length * c.edge_length * c.edge_length) :
  min_yellow_surface_area c / total_surface_area c = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_min_yellow_fraction_l2528_252876


namespace NUMINAMATH_CALUDE_set_operations_l2528_252883

-- Define the sets A and B
def A : Set ℝ := {x | x = 0 ∨ ∃ y, x = |y|}
def B : Set ℝ := {-1, 0, 1}

-- State the theorem
theorem set_operations (h : A ⊆ B) :
  (A ∩ B = {0, 1}) ∧
  (A ∪ B = {-1, 0, 1}) ∧
  (B \ A = {-1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2528_252883


namespace NUMINAMATH_CALUDE_no_k_satisfies_condition_l2528_252860

-- Define a function to get the nth odd prime number
def nthOddPrime (n : ℕ) : ℕ := sorry

-- Define a function to calculate the product of the first k odd primes
def productOfFirstKOddPrimes (k : ℕ) : ℕ := sorry

-- Define a function to check if a number is a perfect power greater than 1
def isPerfectPowerGreaterThanOne (n : ℕ) : Prop := sorry

-- Theorem statement
theorem no_k_satisfies_condition :
  ∀ k : ℕ, k > 0 → ¬(isPerfectPowerGreaterThanOne (productOfFirstKOddPrimes k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_k_satisfies_condition_l2528_252860


namespace NUMINAMATH_CALUDE_divisible_by_thirty_l2528_252806

theorem divisible_by_thirty (n : ℕ+) : ∃ k : ℤ, (n : ℤ)^19 - (n : ℤ)^7 = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_thirty_l2528_252806


namespace NUMINAMATH_CALUDE_allison_wins_probability_l2528_252823

-- Define the faces of each cube
def allison_cube : Finset Nat := {4}
def charlie_cube : Finset Nat := {1, 2, 3, 4, 5, 6}
def eve_cube : Finset Nat := {3, 3, 4, 4, 4, 5}

-- Define the probability of rolling each face
def prob_roll (cube : Finset Nat) (face : Nat) : ℚ :=
  (cube.filter (· = face)).card / cube.card

-- Define the event of rolling less than 4
def roll_less_than_4 (cube : Finset Nat) : ℚ :=
  (cube.filter (· < 4)).card / cube.card

-- Theorem statement
theorem allison_wins_probability :
  prob_roll allison_cube 4 * roll_less_than_4 charlie_cube * roll_less_than_4 eve_cube = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_allison_wins_probability_l2528_252823


namespace NUMINAMATH_CALUDE_part_to_third_ratio_l2528_252866

theorem part_to_third_ratio (N P : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) :
  P / ((1/3) * N) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_third_ratio_l2528_252866


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2528_252848

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (6 * a^3 - 803 * a + 1606 = 0) → 
  (6 * b^3 - 803 * b + 1606 = 0) → 
  (6 * c^3 - 803 * c + 1606 = 0) → 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 803 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2528_252848


namespace NUMINAMATH_CALUDE_contradiction_assumption_l2528_252897

theorem contradiction_assumption (x y : ℝ) (h : x + y > 2) :
  ¬(x ≤ 1 ∧ y ≤ 1) → (x > 1 ∨ y > 1) := by
  sorry

#check contradiction_assumption

end NUMINAMATH_CALUDE_contradiction_assumption_l2528_252897


namespace NUMINAMATH_CALUDE_inverse_of_inverse_nine_l2528_252878

def f (x : ℝ) : ℝ := 5 * x + 7

theorem inverse_of_inverse_nine :
  let f_inv (x : ℝ) := (x - 7) / 5
  f_inv (f_inv 9) = -33 / 25 := by
sorry

end NUMINAMATH_CALUDE_inverse_of_inverse_nine_l2528_252878


namespace NUMINAMATH_CALUDE_triangle_properties_l2528_252820

theorem triangle_properties (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ratio : ∃ (k : ℝ), a = 2*k ∧ b = 5*k ∧ c = 6*k) 
  (h_area : (1/2) * a * c * Real.sqrt (1 - ((a^2 + c^2 - b^2) / (2*a*c))^2) = 3 * Real.sqrt 39 / 4) :
  ((a^2 + c^2 - b^2) / (2*a*c) = 5/8) ∧ (a + b + c = 13) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2528_252820


namespace NUMINAMATH_CALUDE_closest_fraction_l2528_252830

def medals_won : ℚ := 35 / 225

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (x : ℚ), x ∈ options ∧ 
  ∀ (y : ℚ), y ∈ options → |x - medals_won| ≤ |y - medals_won| ∧
  x = 1/6 :=
sorry

end NUMINAMATH_CALUDE_closest_fraction_l2528_252830


namespace NUMINAMATH_CALUDE_johns_remaining_money_l2528_252826

/-- Given John's initial money and his purchases, calculate the remaining amount --/
theorem johns_remaining_money (initial : ℕ) (roast : ℕ) (vegetables : ℕ) :
  initial = 100 ∧ roast = 17 ∧ vegetables = 11 →
  initial - (roast + vegetables) = 72 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l2528_252826


namespace NUMINAMATH_CALUDE_car_trip_duration_l2528_252890

theorem car_trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) : 
  initial_speed = 30 →
  initial_time = 6 →
  additional_speed = 46 →
  average_speed = 34 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed ∧
    total_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_duration_l2528_252890


namespace NUMINAMATH_CALUDE_cookie_difference_l2528_252814

theorem cookie_difference (alyssa_cookies aiyanna_cookies : ℕ) 
  (h1 : alyssa_cookies = 129) (h2 : aiyanna_cookies = 140) : 
  aiyanna_cookies - alyssa_cookies = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l2528_252814


namespace NUMINAMATH_CALUDE_max_a_value_l2528_252862

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 3 -/
def LineEquation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

/-- Predicate to check if a point satisfies the line equation -/
def SatisfiesEquation (m : ℚ) (x y : ℤ) : Prop :=
  LineEquation m x = y

/-- The main theorem -/
theorem max_a_value : 
  ∃ (a : ℚ), a = 101 / 151 ∧ 
  (∀ (m : ℚ), 2/3 < m → m < a → 
    ∀ (x y : ℤ), 0 < x → x ≤ 150 → LatticePoint x y → ¬SatisfiesEquation m x y) ∧
  (∀ (a' : ℚ), a' > a → 
    ∃ (m : ℚ), 2/3 < m ∧ m < a' ∧
      ∃ (x y : ℤ), 0 < x ∧ x ≤ 150 ∧ LatticePoint x y ∧ SatisfiesEquation m x y) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2528_252862


namespace NUMINAMATH_CALUDE_other_x_intercept_l2528_252838

/-- Given a quadratic function with vertex (5, -3) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = -3 + a * (x - 5)^2) →  -- vertex form
  (a * 1^2 + b * 1 + c = 0) →                        -- x-intercept at (1, 0)
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9       -- other x-intercept at 9
  := by sorry

end NUMINAMATH_CALUDE_other_x_intercept_l2528_252838


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2528_252896

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + 2 * (a - 1) * x - 4 < 0) ↔ 
  (-3 < a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2528_252896


namespace NUMINAMATH_CALUDE_initial_stock_calculation_l2528_252825

/-- The number of toys sold in the first week -/
def toys_sold_first_week : ℕ := 38

/-- The number of toys sold in the second week -/
def toys_sold_second_week : ℕ := 26

/-- The number of toys left after two weeks -/
def toys_left : ℕ := 19

/-- The initial number of toys in stock -/
def initial_stock : ℕ := toys_sold_first_week + toys_sold_second_week + toys_left

theorem initial_stock_calculation :
  initial_stock = 83 := by sorry

end NUMINAMATH_CALUDE_initial_stock_calculation_l2528_252825


namespace NUMINAMATH_CALUDE_total_students_is_150_l2528_252813

/-- Represents the number of students in a school with age distribution. -/
structure School where
  total : ℕ
  below_8 : ℕ
  age_8 : ℕ
  above_8 : ℕ

/-- Conditions for the school problem. -/
def school_conditions (s : School) : Prop :=
  s.below_8 = (s.total * 20) / 100 ∧
  s.above_8 = (s.age_8 * 2) / 3 ∧
  s.age_8 = 72 ∧
  s.total = s.below_8 + s.age_8 + s.above_8

/-- Theorem stating that the total number of students is 150. -/
theorem total_students_is_150 :
  ∃ s : School, school_conditions s ∧ s.total = 150 := by
  sorry

#check total_students_is_150

end NUMINAMATH_CALUDE_total_students_is_150_l2528_252813


namespace NUMINAMATH_CALUDE_f_explicit_function_l2528_252829

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 - 1

-- State the theorem
theorem f_explicit_function (x : ℝ) (h : x ≥ 0) : 
  f (Real.sqrt x + 1) = x + 2 * Real.sqrt x ↔ (∀ y ≥ 1, f y = y^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_explicit_function_l2528_252829


namespace NUMINAMATH_CALUDE_min_value_problem_l2528_252887

theorem min_value_problem (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) :
  (9 * z) / (3 * x + y) + (9 * x) / (y + 3 * z) + (4 * y) / (x + z) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2528_252887


namespace NUMINAMATH_CALUDE_daisy_tuesday_toys_l2528_252885

/-- The number of dog toys Daisy had on various days --/
structure DaisyToys where
  monday : ℕ
  tuesday_before : ℕ
  tuesday_after : ℕ
  wednesday_new : ℕ
  total_if_found : ℕ

/-- Theorem stating the number of toys Daisy had on Tuesday before new purchases --/
theorem daisy_tuesday_toys (d : DaisyToys)
  (h1 : d.monday = 5)
  (h2 : d.tuesday_after = d.tuesday_before + 3)
  (h3 : d.wednesday_new = 5)
  (h4 : d.total_if_found = 13)
  (h5 : d.total_if_found = d.tuesday_before + 3 + d.wednesday_new) :
  d.tuesday_before = 5 := by
  sorry

#check daisy_tuesday_toys

end NUMINAMATH_CALUDE_daisy_tuesday_toys_l2528_252885


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_11_l2528_252831

theorem greatest_two_digit_multiple_of_11 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 11 ∣ n → n ≤ 99 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_11_l2528_252831


namespace NUMINAMATH_CALUDE_circle_area_theorem_l2528_252881

theorem circle_area_theorem (r : ℝ) (h : 8 / (2 * Real.pi * r) = (2 * r)^2) :
  π * r^2 = Real.pi^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l2528_252881


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_arithmetic_sequence_sum_l2528_252837

-- Define the geometric sequence {a_n}
def a (n : ℕ) : ℝ := sorry

-- Define the arithmetic sequence {b_n}
def b (n : ℕ) : ℝ := sorry

-- Define the sum of the first n terms of {b_n}
def S (n : ℕ) : ℝ := sorry

theorem geometric_sequence_formula :
  (a 2 = 6) →
  (a 2 + a 3 = 24) →
  ∀ n : ℕ, a n = 2 * 3^(n - 1) := by sorry

theorem arithmetic_sequence_sum :
  (b 1 = a 1) →
  (b 3 = -10) →
  ∀ n : ℕ, S n = -3 * n^2 + 5 * n := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_arithmetic_sequence_sum_l2528_252837


namespace NUMINAMATH_CALUDE_student_count_l2528_252832

theorem student_count (ratio : ℝ) (teachers : ℕ) (h1 : ratio = 27.5) (h2 : teachers = 42) :
  ↑teachers * ratio = 1155 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2528_252832


namespace NUMINAMATH_CALUDE_max_colored_cells_1000_cube_l2528_252827

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  sideLength : n > 0

/-- Represents the maximum number of cells that can be colored on a cube's surface -/
def maxColoredCells (c : Cube n) : ℕ :=
  6 * n^2 - 2 * n^2

theorem max_colored_cells_1000_cube :
  ∀ (c : Cube 1000), maxColoredCells c = 2998000 :=
sorry

end NUMINAMATH_CALUDE_max_colored_cells_1000_cube_l2528_252827


namespace NUMINAMATH_CALUDE_max_snacks_is_11_l2528_252874

/-- Represents the number of snacks in a pack -/
inductive SnackPack
  | Single : SnackPack
  | Pack4 : SnackPack
  | Pack7 : SnackPack

/-- The cost of a snack pack in dollars -/
def cost : SnackPack → ℕ
  | SnackPack.Single => 2
  | SnackPack.Pack4 => 6
  | SnackPack.Pack7 => 9

/-- The number of snacks in a pack -/
def snacks : SnackPack → ℕ
  | SnackPack.Single => 1
  | SnackPack.Pack4 => 4
  | SnackPack.Pack7 => 7

/-- The budget in dollars -/
def budget : ℕ := 15

/-- A purchase is a list of snack packs -/
def Purchase := List SnackPack

/-- The total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.foldl (fun acc pack => acc + cost pack) 0

/-- The total number of snacks in a purchase -/
def totalSnacks (p : Purchase) : ℕ :=
  p.foldl (fun acc pack => acc + snacks pack) 0

/-- A purchase is valid if its total cost is within the budget -/
def isValidPurchase (p : Purchase) : Prop :=
  totalCost p ≤ budget

/-- The theorem stating that 11 is the maximum number of snacks that can be purchased -/
theorem max_snacks_is_11 :
  ∀ p : Purchase, isValidPurchase p → totalSnacks p ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_snacks_is_11_l2528_252874


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l2528_252847

theorem fraction_sum_equals_one (a : ℝ) (h : a ≠ -2) :
  (a + 1) / (a + 2) + 1 / (a + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l2528_252847


namespace NUMINAMATH_CALUDE_exam_score_ratio_l2528_252818

theorem exam_score_ratio (total_questions : ℕ) (lowella_percentage : ℚ) 
  (pamela_additional_percentage : ℚ) (mandy_score : ℕ) : 
  total_questions = 100 →
  lowella_percentage = 35 / 100 →
  pamela_additional_percentage = 20 / 100 →
  mandy_score = 84 →
  ∃ (k : ℚ), k * (lowella_percentage * total_questions + 
    pamela_additional_percentage * (lowella_percentage * total_questions)) = mandy_score ∧ 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_ratio_l2528_252818


namespace NUMINAMATH_CALUDE_office_network_connections_l2528_252859

/-- A network of switches where each switch connects to exactly four others. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ
  connection_count : ℕ

/-- The theorem stating the correct number of connections in the given network. -/
theorem office_network_connections (network : SwitchNetwork)
  (h1 : network.num_switches = 30)
  (h2 : network.connections_per_switch = 4) :
  network.connection_count = 60 := by
  sorry

end NUMINAMATH_CALUDE_office_network_connections_l2528_252859


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2528_252851

theorem complex_equation_solution (z : ℂ) : z * (1 - 2*I) = 2 + I → z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2528_252851


namespace NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l2528_252839

theorem ceiling_squared_negative_fraction :
  ⌈(-7/4)^2⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l2528_252839


namespace NUMINAMATH_CALUDE_square_side_length_l2528_252822

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 9 / 16) (h2 : side * side = area) : side = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2528_252822


namespace NUMINAMATH_CALUDE_team_formation_ways_l2528_252821

/-- Represents the number of people who know a specific pair of subjects -/
structure SubjectKnowledge where
  math_physics : Nat
  physics_chemistry : Nat
  chemistry_math : Nat
  physics_biology : Nat

/-- Calculates the total number of people -/
def total_people (sk : SubjectKnowledge) : Nat :=
  sk.math_physics + sk.physics_chemistry + sk.chemistry_math + sk.physics_biology

/-- Calculates the number of ways to choose 3 people from n people -/
def choose_3_from_n (n : Nat) : Nat :=
  n * (n - 1) * (n - 2) / 6

/-- Calculates the number of invalid selections (all 3 from the same group) -/
def invalid_selections (sk : SubjectKnowledge) : Nat :=
  choose_3_from_n sk.math_physics +
  choose_3_from_n sk.physics_chemistry +
  choose_3_from_n sk.chemistry_math +
  choose_3_from_n sk.physics_biology

/-- The main theorem to prove -/
theorem team_formation_ways (sk : SubjectKnowledge) 
  (h1 : sk.math_physics = 7)
  (h2 : sk.physics_chemistry = 6)
  (h3 : sk.chemistry_math = 3)
  (h4 : sk.physics_biology = 4) :
  choose_3_from_n (total_people sk) - invalid_selections sk = 1080 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_ways_l2528_252821


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2528_252871

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a*b*c) ≥ 216 := by
  sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a*b*c) = 216 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2528_252871


namespace NUMINAMATH_CALUDE_lavinia_son_older_than_daughter_l2528_252815

/-- Given information about the ages of Lavinia's and Katie's children, prove that Lavinia's son is 21 years older than Lavinia's daughter. -/
theorem lavinia_son_older_than_daughter :
  ∀ (lavinia_daughter lavinia_son katie_daughter katie_son : ℕ),
  lavinia_daughter = katie_daughter / 3 →
  lavinia_son = 2 * katie_daughter →
  lavinia_daughter + lavinia_son = 2 * katie_daughter + 5 →
  katie_daughter = 12 →
  katie_son + 3 = lavinia_son →
  lavinia_son - lavinia_daughter = 21 :=
by sorry

end NUMINAMATH_CALUDE_lavinia_son_older_than_daughter_l2528_252815


namespace NUMINAMATH_CALUDE_reflect_F_theorem_l2528_252875

/-- Reflects a point over the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point over the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The composition of two reflections -/
def double_reflect (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_eq_x (reflect_x_axis p)

theorem reflect_F_theorem :
  let F : ℝ × ℝ := (1, 1)
  double_reflect F = (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_reflect_F_theorem_l2528_252875


namespace NUMINAMATH_CALUDE_christina_bank_transfer_l2528_252877

/-- Calculates the remaining balance after a transfer --/
def remaining_balance (initial : ℕ) (transfer : ℕ) : ℕ :=
  initial - transfer

theorem christina_bank_transfer :
  remaining_balance 27004 69 = 26935 := by
  sorry

end NUMINAMATH_CALUDE_christina_bank_transfer_l2528_252877


namespace NUMINAMATH_CALUDE_fraction_equality_l2528_252884

theorem fraction_equality (x y z : ℝ) (h1 : x / 2 = y / 3) (h2 : x / 2 = z / 5) (h3 : 2 * x + y ≠ 0) :
  (x + y - 3 * z) / (2 * x + y) = -10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2528_252884


namespace NUMINAMATH_CALUDE_smallest_multiple_l2528_252894

theorem smallest_multiple : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 5 = 0) ∧ 
  ((a + 1) % 7 = 0) ∧ 
  ((a + 2) % 9 = 0) ∧ 
  ((a + 3) % 11 = 0) ∧ 
  (∀ (b : ℕ), b > 0 ∧ 
    (b % 5 = 0) ∧ 
    ((b + 1) % 7 = 0) ∧ 
    ((b + 2) % 9 = 0) ∧ 
    ((b + 3) % 11 = 0) → 
    a ≤ b) ∧
  a = 720 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2528_252894


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2528_252882

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2528_252882


namespace NUMINAMATH_CALUDE_yellow_candy_probability_l2528_252842

theorem yellow_candy_probability (p_red p_orange p_yellow : ℝ) : 
  p_red = 0.25 →
  p_orange = 0.35 →
  p_red + p_orange + p_yellow = 1 →
  p_yellow = 0.4 := by
sorry

end NUMINAMATH_CALUDE_yellow_candy_probability_l2528_252842


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2528_252800

theorem arithmetic_expression_equality : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2528_252800


namespace NUMINAMATH_CALUDE_janet_freelance_earnings_l2528_252850

/-- Calculates the difference in monthly earnings between freelancing and current job --/
def freelance_earnings_difference (
  hours_per_week : ℕ)
  (current_wage : ℚ)
  (freelance_wage : ℚ)
  (weeks_per_month : ℕ)
  (extra_fica_per_week : ℚ)
  (healthcare_premium_per_month : ℚ) : ℚ :=
  let wage_difference := freelance_wage - current_wage
  let weekly_earnings_difference := wage_difference * hours_per_week
  let monthly_earnings_difference := weekly_earnings_difference * weeks_per_month
  let extra_monthly_expenses := extra_fica_per_week * weeks_per_month + healthcare_premium_per_month
  monthly_earnings_difference - extra_monthly_expenses

/-- Theorem stating the earnings difference for Janet's specific situation --/
theorem janet_freelance_earnings :
  freelance_earnings_difference 40 30 40 4 25 400 = 1100 := by
  sorry

end NUMINAMATH_CALUDE_janet_freelance_earnings_l2528_252850


namespace NUMINAMATH_CALUDE_edge_length_of_total_72_l2528_252892

/-- Represents a rectangular prism with equal edge lengths -/
structure EqualEdgePrism where
  edge_length : ℝ
  total_length : ℝ
  total_length_eq : total_length = 12 * edge_length

/-- Theorem: If the sum of all edge lengths in an equal edge prism is 72 cm, 
    then the length of one edge is 6 cm -/
theorem edge_length_of_total_72 (prism : EqualEdgePrism) 
  (h : prism.total_length = 72) : prism.edge_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_edge_length_of_total_72_l2528_252892


namespace NUMINAMATH_CALUDE_absolute_difference_inequality_l2528_252857

theorem absolute_difference_inequality (x : ℝ) : 
  |x - 1| - |x - 2| > (1/2) ↔ x > (7/4) := by sorry

end NUMINAMATH_CALUDE_absolute_difference_inequality_l2528_252857


namespace NUMINAMATH_CALUDE_postcard_width_is_six_l2528_252846

/-- Represents a rectangular postcard -/
structure Postcard where
  width : ℝ
  height : ℝ

/-- The perimeter of a rectangular postcard -/
def perimeter (p : Postcard) : ℝ := 2 * (p.width + p.height)

theorem postcard_width_is_six :
  ∀ p : Postcard,
  p.height = 4 →
  perimeter p = 20 →
  p.width = 6 := by
sorry

end NUMINAMATH_CALUDE_postcard_width_is_six_l2528_252846


namespace NUMINAMATH_CALUDE_imaginary_part_of_3_minus_4i_l2528_252880

theorem imaginary_part_of_3_minus_4i :
  Complex.im (3 - 4 * Complex.I) = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_3_minus_4i_l2528_252880


namespace NUMINAMATH_CALUDE_pen_ratio_problem_l2528_252828

theorem pen_ratio_problem (blue_pens green_pens : ℕ) : 
  (blue_pens : ℚ) / green_pens = 4 / 3 →
  blue_pens = 16 →
  green_pens = 12 := by
sorry

end NUMINAMATH_CALUDE_pen_ratio_problem_l2528_252828


namespace NUMINAMATH_CALUDE_quadrilateral_pyramid_ratio_l2528_252805

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a quadrilateral pyramid -/
structure QuadrilateralPyramid where
  P : Point3D
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Checks if two line segments are parallel -/
def areParallel (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Checks if two line segments are perpendicular -/
def arePerpendicular (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Checks if a line is perpendicular to a plane -/
def isPerpendicularToPlane (p1 p2 : Point3D) (plane : Plane3D) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Calculates the sine of the angle between a line and a plane -/
def sineAngleLinePlane (p1 p2 : Point3D) (plane : Plane3D) : ℝ := sorry

/-- Main theorem -/
theorem quadrilateral_pyramid_ratio 
  (pyramid : QuadrilateralPyramid) 
  (Q : Point3D)
  (h1 : areParallel pyramid.A pyramid.B pyramid.C pyramid.D)
  (h2 : arePerpendicular pyramid.A pyramid.B pyramid.A pyramid.D)
  (h3 : distance pyramid.A pyramid.B = 4)
  (h4 : distance pyramid.A pyramid.D = 2 * Real.sqrt 2)
  (h5 : distance pyramid.C pyramid.D = 2)
  (h6 : isPerpendicularToPlane pyramid.P pyramid.A (Plane3D.mk 0 0 1 0))  -- Assuming ABCD is on the xy-plane
  (h7 : distance pyramid.P pyramid.A = 4)
  (h8 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q.x = pyramid.P.x + t * (pyramid.B.x - pyramid.P.x) ∧
                              Q.y = pyramid.P.y + t * (pyramid.B.y - pyramid.P.y) ∧
                              Q.z = pyramid.P.z + t * (pyramid.B.z - pyramid.P.z))
  (h9 : sineAngleLinePlane Q pyramid.C (Plane3D.mk 1 0 0 0) = Real.sqrt 3 / 3)  -- Assuming PAC is on the yz-plane
  : ∃ (t : ℝ), distance pyramid.P Q / distance pyramid.P pyramid.B = 7/12 ∧ 
               Q.x = pyramid.P.x + t * (pyramid.B.x - pyramid.P.x) ∧
               Q.y = pyramid.P.y + t * (pyramid.B.y - pyramid.P.y) ∧
               Q.z = pyramid.P.z + t * (pyramid.B.z - pyramid.P.z) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_pyramid_ratio_l2528_252805


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2528_252889

/-- If a 60° arc on circle A has the same length as a 40° arc on circle B,
    then the ratio of the area of circle A to the area of circle B is 4/9 -/
theorem circle_area_ratio (r_A r_B : ℝ) (h : r_A > 0 ∧ r_B > 0) :
  (60 / 360) * (2 * Real.pi * r_A) = (40 / 360) * (2 * Real.pi * r_B) →
  (Real.pi * r_A ^ 2) / (Real.pi * r_B ^ 2) = 4 / 9 := by
sorry


end NUMINAMATH_CALUDE_circle_area_ratio_l2528_252889


namespace NUMINAMATH_CALUDE_donald_oranges_l2528_252844

theorem donald_oranges (initial_oranges found_oranges : ℕ) 
  (h1 : initial_oranges = 4)
  (h2 : found_oranges = 5) :
  initial_oranges + found_oranges = 9 := by
  sorry

end NUMINAMATH_CALUDE_donald_oranges_l2528_252844


namespace NUMINAMATH_CALUDE_correct_equation_l2528_252801

/-- Represents a bookstore's novel purchases -/
structure NovelPurchases where
  first_cost : ℝ
  second_cost : ℝ
  quantity_difference : ℕ
  first_quantity : ℕ

/-- The equation representing equal cost per copy for both purchases -/
def equal_cost_equation (p : NovelPurchases) : Prop :=
  p.first_cost / p.first_quantity = p.second_cost / (p.first_quantity + p.quantity_difference)

/-- Theorem stating that the given equation correctly represents the situation -/
theorem correct_equation (p : NovelPurchases) 
  (h1 : p.first_cost = 2000)
  (h2 : p.second_cost = 3000)
  (h3 : p.quantity_difference = 50) :
  equal_cost_equation p ↔ p.first_cost / p.first_quantity = p.second_cost / (p.first_quantity + p.quantity_difference) :=
sorry

end NUMINAMATH_CALUDE_correct_equation_l2528_252801


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_three_l2528_252808

/-- Given two linear functions f and g defined by real parameters A and B,
    proves that if f(g(x)) - g(f(x)) = 2(B - A) and A ≠ B, then A + B = 3. -/
theorem sum_of_coefficients_is_three
  (A B : ℝ)
  (hne : A ≠ B)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x + B)
  (hg : ∀ x, g x = B * x + A)
  (h : ∀ x, f (g x) - g (f x) = 2 * (B - A)) :
  A + B = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_three_l2528_252808


namespace NUMINAMATH_CALUDE_simplify_expression_log_equation_result_l2528_252803

-- Part 1
theorem simplify_expression (x : ℝ) (h : x > 0) :
  (x - 1) / (x^(2/3) + x^(1/3) + 1) + (x + 1) / (x^(1/3) + 1) - (x - x^(1/3)) / (x^(1/3) - 1) = -x^(1/3) :=
sorry

-- Part 2
theorem log_equation_result (x : ℝ) (h1 : x > 0) (h2 : 3*x - 2 > 0) (h3 : 3*x + 2 > 0)
  (h4 : 2 * Real.log (3*x - 2) = Real.log x + Real.log (3*x + 2)) :
  Real.log (Real.sqrt (2 * Real.sqrt (2 * Real.sqrt 2))) / Real.log (Real.sqrt x) = 7/4 :=
sorry

end NUMINAMATH_CALUDE_simplify_expression_log_equation_result_l2528_252803


namespace NUMINAMATH_CALUDE_range_of_b_l2528_252817

theorem range_of_b (a b c : ℝ) (sum_eq : a + b + c = 9) (prod_eq : a * b + b * c + c * a = 24) :
  1 ≤ b ∧ b ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_b_l2528_252817


namespace NUMINAMATH_CALUDE_arithmetic_seq_fifth_term_l2528_252854

/-- An arithmetic sequence -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_fifth_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) 
  (h_sum : a 3 + a 8 = 22) 
  (h_sixth : a 6 = 8) : 
  a 5 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_seq_fifth_term_l2528_252854


namespace NUMINAMATH_CALUDE_max_real_axis_length_l2528_252849

/-- Represents a hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are of the form 2x ± y = 0 -/
  asymptotes : Unit
  /-- The hyperbola passes through the intersection of two lines -/
  intersection_point : ℝ × ℝ
  /-- The parameter t determines the intersection point -/
  t : ℝ
  /-- The intersection point satisfies the equations of both lines -/
  satisfies_line1 : intersection_point.1 + intersection_point.2 = 3
  satisfies_line2 : 2 * intersection_point.1 - intersection_point.2 = -3 * t
  /-- The parameter t is within the specified range -/
  t_range : -2 ≤ t ∧ t ≤ 5

/-- The length of the real axis of the hyperbola -/
def real_axis_length (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the maximum possible length of the real axis -/
theorem max_real_axis_length (h : Hyperbola) : 
  real_axis_length h ≤ 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_max_real_axis_length_l2528_252849


namespace NUMINAMATH_CALUDE_math_club_teams_count_l2528_252899

-- Define the number of girls and boys in the math club
def num_girls : ℕ := 5
def num_boys : ℕ := 7

-- Define the number of girls and boys required for each team
def girls_per_team : ℕ := 2
def boys_per_team : ℕ := 2

-- Define the theorem
theorem math_club_teams_count :
  (Nat.choose num_girls girls_per_team) *
  (Nat.choose num_boys boys_per_team) *
  boys_per_team = 420 := by
sorry

end NUMINAMATH_CALUDE_math_club_teams_count_l2528_252899


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2528_252855

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Check if two isosceles triangles are noncongruent -/
def noncongruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.leg ≠ t2.leg ∨ t1.base ≠ t2.base

/-- Calculate the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ :=
  2 * t.leg + t.base

/-- Calculate the area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℚ :=
  (t.base : ℚ) * (((t.leg : ℚ) ^ 2 - ((t.base : ℚ) / 2) ^ 2).sqrt) / 2

/-- The theorem statement -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    noncongruent t1 t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    10 * t1.base = 9 * t2.base ∧
    perimeter t1 = 362 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      noncongruent s1 s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      10 * s1.base = 9 * s2.base →
      perimeter s1 ≥ 362) :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2528_252855


namespace NUMINAMATH_CALUDE_root_sum_squares_l2528_252816

theorem root_sum_squares (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) → 
  (b^3 - 15*b^2 + 25*b - 10 = 0) → 
  (c^3 - 15*c^2 + 25*c - 10 = 0) → 
  (a-b)^2 + (b-c)^2 + (c-a)^2 = 125 := by sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2528_252816


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_3_l2528_252898

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := 3 * x + (a - 1) * y = a - 7

-- Define parallel lines
def parallel (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, line1 a x₁ y₁ → line2 a x₂ y₂ → 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) → (a * (x₂ - x₁) = 2 * (y₁ - y₂) ∧ 3 * (x₂ - x₁) = (a - 1) * (y₁ - y₂))

-- Theorem statement
theorem lines_parallel_iff_a_eq_3 : ∀ a : ℝ, parallel a ↔ a = 3 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_3_l2528_252898


namespace NUMINAMATH_CALUDE_candy_mixture_problem_l2528_252840

/-- Candy mixture problem -/
theorem candy_mixture_problem (x : ℝ) :
  (64 * 2 + x * 3 = (64 + x) * 2.2) →
  (64 + x = 80) := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_problem_l2528_252840


namespace NUMINAMATH_CALUDE_equation_solution_l2528_252886

theorem equation_solution (x : ℝ) :
  (∃ (n : ℤ), x = π / 18 + 2 * π * n / 9) ∨ (∃ (s : ℤ), x = 2 * π * s / 3) ↔
  (((1 - (Real.cos (15 * x))^7 * (Real.cos (9 * x))^2)^(1/4) = Real.sin (9 * x)) ∧
   Real.sin (9 * x) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2528_252886


namespace NUMINAMATH_CALUDE_percentage_calculation_l2528_252865

theorem percentage_calculation (part whole : ℝ) (h1 : part = 375.2) (h2 : whole = 12546.8) :
  (part / whole) * 100 = 2.99 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2528_252865


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2528_252835

/-- A function f: ℝ⁺ → ℝ⁺ satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (y * f x) * (x + y) = x^2 * (f x + f y)

/-- The theorem stating that the only function satisfying the equation is f(x) = 1/x -/
theorem functional_equation_solution (f : ℝ → ℝ) :
  FunctionalEquation f → ∀ x, x > 0 → f x = 1 / x := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l2528_252835


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2528_252888

theorem at_least_one_not_less_than_two (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2528_252888


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l2528_252812

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 4}

theorem complement_of_N_in_M : M \ N = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l2528_252812


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2528_252870

theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + (2 : ℚ) / 99 = (8 : ℚ) / 33 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2528_252870
