import Mathlib

namespace range_of_a_theorem_l2891_289121

/-- Proposition p: For any x ∈ ℝ, x² - 2x > a -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

/-- Proposition q: There exists x ∈ ℝ such that x² + 2ax + 2 - a = 0 -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

/-- The range of a given the conditions -/
def range_of_a : Set ℝ := { a : ℝ | (a > -2 ∧ a < -1) ∨ a ≥ 1 }

theorem range_of_a_theorem (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ range_of_a := by sorry

end range_of_a_theorem_l2891_289121


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l2891_289159

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l2891_289159


namespace triangle_properties_l2891_289155

/-- Theorem about a specific triangle ABC -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists with sides a, b, c opposite to angles A, B, C
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  -- Given conditions
  (2 * c - a) * Real.cos B = b * Real.cos A →
  3 * a + b = 2 * c →
  b = 2 →
  1 / Real.sin A + 1 / Real.sin C = 4 * Real.sqrt 3 / 3 →
  -- Conclusions
  Real.cos C = -1/7 ∧ 
  (1/2 * a * c * Real.sin B : ℝ) = Real.sqrt 3 :=
by sorry

end triangle_properties_l2891_289155


namespace fixed_point_power_function_l2891_289167

theorem fixed_point_power_function (f : ℝ → ℝ) :
  (∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) →
  f 2 = Real.sqrt 2 / 2 →
  f 9 = 1 / 3 := by
sorry

end fixed_point_power_function_l2891_289167


namespace sqrt_x_minus_7_real_implies_x_geq_7_l2891_289179

theorem sqrt_x_minus_7_real_implies_x_geq_7 (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 7) → x ≥ 7 := by
sorry

end sqrt_x_minus_7_real_implies_x_geq_7_l2891_289179


namespace functional_equation_solution_l2891_289196

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f (x + y)) = f (y * f x) + x^2) →
  (∀ x : ℝ, f x = x ∨ f x = -x) := by
sorry

end functional_equation_solution_l2891_289196


namespace river_distance_l2891_289153

theorem river_distance (d : ℝ) : 
  (¬(d ≤ 12)) → (¬(d ≥ 15)) → (¬(d ≥ 10)) → (12 < d ∧ d < 15) := by
  sorry

end river_distance_l2891_289153


namespace first_rope_length_l2891_289132

/-- Represents the lengths of ropes Tony found -/
structure Ropes where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ
  fifth : ℝ

/-- Calculates the total length of ropes before tying -/
def total_length (r : Ropes) : ℝ :=
  r.first + r.second + r.third + r.fourth + r.fifth

/-- Calculates the length lost due to knots -/
def knot_loss (num_ropes : ℕ) (loss_per_knot : ℝ) : ℝ :=
  (num_ropes - 1 : ℝ) * loss_per_knot

/-- Theorem stating that given the conditions, the first rope Tony found is 20 feet long -/
theorem first_rope_length
  (r : Ropes)
  (h1 : r.second = 2)
  (h2 : r.third = 2)
  (h3 : r.fourth = 2)
  (h4 : r.fifth = 7)
  (h5 : total_length r - knot_loss 5 1.2 = 35) :
  r.first = 20 := by
  sorry

end first_rope_length_l2891_289132


namespace intersection_of_A_and_B_l2891_289125

def set_A : Set ℝ := {x | |x| < 3}
def set_B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | 1 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l2891_289125


namespace dilation_circle_to_ellipse_l2891_289133

/-- Given a circle A and a dilation transformation, prove the equation of the resulting curve C -/
theorem dilation_circle_to_ellipse :
  let circle_A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let dilation (p : ℝ × ℝ) : ℝ × ℝ := (2 * p.1, 3 * p.2)
  let curve_C : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 / 9 = 1}
  (∀ p ∈ circle_A, dilation p ∈ curve_C) ∧
  (∀ q ∈ curve_C, ∃ p ∈ circle_A, dilation p = q) := by
sorry

end dilation_circle_to_ellipse_l2891_289133


namespace ellipse_properties_l2891_289107

-- Define the ellipse C
def ellipse_C (x y : ℝ) (a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (7 - a^2) = 1 ∧ a > 0

-- Define the focal distance
def focal_distance (a : ℝ) : ℝ := 2

-- Define the standard form of the ellipse
def standard_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define a line passing through (4,0)
def line_through_R (x y : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 4)

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the right focus
def right_focus : (ℝ × ℝ) := (1, 0)

-- Theorem statement
theorem ellipse_properties (a : ℝ) (k : ℝ) (x1 y1 x2 y2 : ℝ) :
  (∀ x y, ellipse_C x y a ↔ standard_ellipse x y) ∧
  (∀ x y, line_through_R x y k ∧ point_on_ellipse x y →
    ∃ xn yn xq yq,
      point_on_ellipse xn yn ∧
      point_on_ellipse xq yq ∧
      xn = x1 ∧ yn = -y1 ∧
      (yn - y2) * (xq - right_focus.1) = (yq - right_focus.2) * (xn - right_focus.1)) :=
by sorry

end ellipse_properties_l2891_289107


namespace linear_independence_exp_trig_l2891_289118

theorem linear_independence_exp_trig (α β : ℝ) (h : β ≠ 0) :
  ∀ (α₁ α₂ : ℝ), (∀ x : ℝ, α₁ * Real.exp (α * x) * Real.sin (β * x) + 
                           α₂ * Real.exp (α * x) * Real.cos (β * x) = 0) →
                 α₁ = 0 ∧ α₂ = 0 := by
  sorry

end linear_independence_exp_trig_l2891_289118


namespace x_value_l2891_289103

theorem x_value : ∃ x : ℝ, 3 * x - 48.2 = 0.25 * (4 * x + 56.8) ∧ x = 31.2 := by
  sorry

end x_value_l2891_289103


namespace trigonometric_equation_solutions_l2891_289160

theorem trigonometric_equation_solutions (x : ℝ) :
  (5.14 * Real.sin (3 * x) + Real.sin (5 * x) = 2 * ((Real.cos (2 * x))^2 - (Real.sin (3 * x))^2)) ↔
  (∃ k : ℤ, x = π / 2 * (2 * k + 1) ∨ x = π / 18 * (4 * k + 1)) :=
by sorry

end trigonometric_equation_solutions_l2891_289160


namespace power_function_sum_l2891_289149

/-- Given a power function f(x) = kx^α that passes through the point (1/2, √2),
    prove that k + α = 1/2 -/
theorem power_function_sum (k α : ℝ) (h : k * (1/2)^α = Real.sqrt 2) : k + α = 1/2 := by
  sorry

end power_function_sum_l2891_289149


namespace expression_equality_l2891_289117

theorem expression_equality : |Real.sqrt 2 - 1| - (π + 1)^0 + Real.sqrt ((-3)^2) = Real.sqrt 2 + 1 := by
  sorry

end expression_equality_l2891_289117


namespace oil_percentage_in_mixtureA_l2891_289186

/-- Represents the composition of a mixture --/
structure Mixture where
  oil : ℝ
  materialB : ℝ

/-- The original mixture A --/
def mixtureA : Mixture := sorry

/-- The weight of the original mixture A in kilograms --/
def originalWeight : ℝ := 8

/-- The weight of oil added to mixture A in kilograms --/
def addedOil : ℝ := 2

/-- The weight of mixture A added to the new mixture in kilograms --/
def addedMixtureA : ℝ := 6

/-- The percentage of material B in the final mixture --/
def finalMaterialBPercentage : ℝ := 70

/-- Theorem stating that the percentage of oil in the original mixture A is 20% --/
theorem oil_percentage_in_mixtureA : mixtureA.oil / (mixtureA.oil + mixtureA.materialB) = 0.2 := by
  sorry

end oil_percentage_in_mixtureA_l2891_289186


namespace parabola_focus_directrix_distance_l2891_289144

/-- The parabola is defined by the equation y^2 = 8x -/
def is_parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The equation of the directrix -/
def directrix_x : ℝ := -2

/-- The distance from a point to a vertical line -/
def distance_to_line (p : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  |p.1 - line_x|

theorem parabola_focus_directrix_distance :
  distance_to_line focus directrix_x = 4 := by sorry

end parabola_focus_directrix_distance_l2891_289144


namespace square_sum_theorem_l2891_289143

theorem square_sum_theorem (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := by
  sorry

end square_sum_theorem_l2891_289143


namespace symmetrical_line_passes_through_point_l2891_289112

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The line y = x -/
def lineYEqX : Line := ⟨1, -1, 0⟩

/-- Get the symmetrical line with respect to y = x -/
def symmetricalLine (l : Line) : Line :=
  ⟨l.b, l.a, l.c⟩

theorem symmetrical_line_passes_through_point :
  let l₁ : Line := ⟨2, 1, -1⟩  -- y = -2x + 1 rewritten as 2x + y - 1 = 0
  let l₂ := symmetricalLine l₁
  let p : Point := ⟨3, -1⟩
  pointOnLine l₂ p := by sorry

end symmetrical_line_passes_through_point_l2891_289112


namespace no_rabbits_perished_l2891_289176

/-- Represents the farm with animals before and after the disease outbreak -/
structure Farm where
  initial_count : ℕ  -- Initial count of each animal type
  surviving_cows : ℕ
  surviving_pigs : ℕ
  surviving_horses : ℕ
  surviving_rabbits : ℕ

/-- The conditions of the farm after the disease outbreak -/
def farm_conditions (f : Farm) : Prop :=
  -- Initially equal number of each animal
  f.initial_count > 0 ∧
  -- One out of every five cows died
  f.surviving_cows = (4 * f.initial_count) / 5 ∧
  -- Number of horses that died equals number of pigs that survived
  f.surviving_horses = f.initial_count - f.surviving_pigs ∧
  -- Proportion of rabbits among survivors is 5/14
  14 * f.surviving_rabbits = 5 * (f.surviving_cows + f.surviving_pigs + f.surviving_horses + f.surviving_rabbits)

/-- The theorem to prove -/
theorem no_rabbits_perished (f : Farm) (h : farm_conditions f) : 
  f.surviving_rabbits = f.initial_count := by
  sorry

end no_rabbits_perished_l2891_289176


namespace ellipse_foci_distance_l2891_289164

def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y - 3)^2) + Real.sqrt ((x + 7)^2 + (y + 2)^2) = 24

def focus1 : ℝ × ℝ := (1, 3)
def focus2 : ℝ × ℝ := (-7, -2)

theorem ellipse_foci_distance :
  let d := Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2)
  d = Real.sqrt 89 := by sorry

end ellipse_foci_distance_l2891_289164


namespace unique_monic_quadratic_l2891_289148

-- Define a monic polynomial of degree 2
def MonicQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, ∀ x, f x = x^2 + b*x + c

theorem unique_monic_quadratic (f : ℝ → ℝ) 
  (monic : MonicQuadratic f) 
  (eval_zero : f 0 = 4)
  (eval_one : f 1 = 10) :
  ∀ x, f x = x^2 + 5*x + 4 := by
sorry

end unique_monic_quadratic_l2891_289148


namespace intersection_point_sum_l2891_289166

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem intersection_point_sum :
  ∃ (a b : ℝ), f a = f (a - 4) ∧ a + b = 6 := by sorry

end intersection_point_sum_l2891_289166


namespace negation_of_existential_proposition_l2891_289127

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x^3 - x + 1 ≤ 0) := by
  sorry

end negation_of_existential_proposition_l2891_289127


namespace system_solution_ratio_l2891_289128

theorem system_solution_ratio (x y a b : ℝ) 
  (eq1 : 8 * x - 6 * y = a)
  (eq2 : 9 * y - 12 * x = b)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (b_nonzero : b ≠ 0) :
  a / b = -2 / 3 := by
sorry

end system_solution_ratio_l2891_289128


namespace tuesday_total_counts_l2891_289147

/-- Represents the number of times Carla counted tiles on Tuesday -/
def tile_counts : Nat := 2

/-- Represents the number of times Carla counted books on Tuesday -/
def book_counts : Nat := 3

/-- Theorem stating that the total number of counts on Tuesday is 5 -/
theorem tuesday_total_counts : tile_counts + book_counts = 5 := by
  sorry

end tuesday_total_counts_l2891_289147


namespace nonagon_diagonal_intersection_probability_l2891_289169

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of vertices in a regular nonagon -/
def num_vertices : ℕ := 9

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := (num_vertices.choose 2) - num_vertices

/-- The number of ways to choose 2 diagonals from all diagonals -/
def num_diagonal_pairs (n : RegularNonagon) : ℕ := (num_diagonals n).choose 2

/-- The number of ways to choose 4 vertices that form a convex quadrilateral -/
def num_intersecting_pairs (n : RegularNonagon) : ℕ := num_vertices.choose 4

/-- The probability that two randomly chosen diagonals intersect -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  (num_intersecting_pairs n : ℚ) / (num_diagonal_pairs n : ℚ)

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry

end nonagon_diagonal_intersection_probability_l2891_289169


namespace function_equality_l2891_289116

theorem function_equality (x : ℝ) (h : x > 0) : 
  (Real.sqrt x)^2 / x = x / (Real.sqrt x)^2 ∧ 
  (Real.sqrt x)^2 / x = 1 ∧ 
  x / (Real.sqrt x)^2 = 1 := by
sorry

end function_equality_l2891_289116


namespace absolute_value_equation_solution_l2891_289145

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 21| + |x - 17| = |2*x - 38| :=
by
  -- The unique solution is x = 19
  use 19
  sorry

end absolute_value_equation_solution_l2891_289145


namespace remainder_of_second_division_l2891_289137

def p (x : ℝ) : ℝ := x^6 - 4*x^5 + 6*x^4 - 4*x^3 + x^2

def s1 (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2

def t1 : ℝ := p 1

def t2 : ℝ := s1 1

theorem remainder_of_second_division (x : ℝ) : t2 = 0 := by
  sorry

end remainder_of_second_division_l2891_289137


namespace stadium_fee_difference_l2891_289113

def stadium_capacity : ℕ := 2000
def entry_fee : ℕ := 20

theorem stadium_fee_difference :
  let full_capacity := stadium_capacity
  let partial_capacity := (3 * stadium_capacity) / 4
  let full_fees := full_capacity * entry_fee
  let partial_fees := partial_capacity * entry_fee
  full_fees - partial_fees = 10000 := by
sorry

end stadium_fee_difference_l2891_289113


namespace cook_selection_l2891_289187

theorem cook_selection (total : ℕ) (vegetarians : ℕ) (cooks : ℕ) :
  total = 10 → vegetarians = 3 → cooks = 2 →
  (Nat.choose vegetarians 1) * (Nat.choose (total - 1) 1) = 27 :=
by sorry

end cook_selection_l2891_289187


namespace reaction_enthalpy_change_l2891_289171

/-- Represents the enthalpy change for a chemical reaction --/
def enthalpy_change (bonds_broken bonds_formed : ℝ) : ℝ :=
  bonds_broken - bonds_formed

/-- Bond dissociation energy for CH3-CH2 (C-C) bond --/
def e_cc : ℝ := 347

/-- Bond dissociation energy for CH3-O (C-O) bond --/
def e_co : ℝ := 358

/-- Bond dissociation energy for CH2-OH (O-H) bond --/
def e_oh_alcohol : ℝ := 463

/-- Bond dissociation energy for C=O (COOH) bond --/
def e_co_double : ℝ := 745

/-- Bond dissociation energy for O-H (COOH) bond --/
def e_oh_acid : ℝ := 467

/-- Bond dissociation energy for O=O (O2) bond --/
def e_oo : ℝ := 498

/-- Bond dissociation energy for O-H (H2O) bond --/
def e_oh_water : ℝ := 467

/-- Total energy of bonds broken in reactants --/
def bonds_broken : ℝ := e_cc + e_co + e_oh_alcohol + 1.5 * e_oo

/-- Total energy of bonds formed in products --/
def bonds_formed : ℝ := e_co_double + e_oh_acid + e_oh_water

/-- Theorem stating the enthalpy change for the given reaction --/
theorem reaction_enthalpy_change :
  enthalpy_change bonds_broken bonds_formed = 236 := by
  sorry

end reaction_enthalpy_change_l2891_289171


namespace pentagon_triangle_side_ratio_l2891_289140

theorem pentagon_triangle_side_ratio :
  ∀ (t p : ℝ),
    t > 0 ∧ p > 0 →
    3 * t = 15 →
    5 * p = 15 →
    t / p = 5 / 3 := by
  sorry

end pentagon_triangle_side_ratio_l2891_289140


namespace profit_percent_calculation_l2891_289135

theorem profit_percent_calculation (selling_price : ℝ) (cost_price : ℝ) (h : cost_price = 0.8 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end profit_percent_calculation_l2891_289135


namespace logarithmic_function_properties_l2891_289181

-- Define the logarithmic function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem logarithmic_function_properties :
  -- f(x) passes through (8,3)
  f 8 = 3 →
  -- f(x) is a logarithmic function (this is implied by its definition)
  -- Prove the following:
  (-- 1. f(x) = log₂(x) (this is true by definition of f)
   -- 2. The domain of f(x) is (0, +∞)
   (∀ x : ℝ, x > 0 ↔ f x ≠ 0) ∧
   -- 3. For f(1-x) > f(1+x), x ∈ (-1, 0)
   (∀ x : ℝ, f (1 - x) > f (1 + x) ↔ -1 < x ∧ x < 0)) :=
by
  sorry

end logarithmic_function_properties_l2891_289181


namespace jasons_pepper_spray_dilemma_l2891_289120

theorem jasons_pepper_spray_dilemma :
  ¬ ∃ (raccoons squirrels opossums : ℕ),
    squirrels = 6 * raccoons ∧
    opossums = 2 * raccoons ∧
    raccoons + squirrels + opossums = 168 :=
by sorry

end jasons_pepper_spray_dilemma_l2891_289120


namespace strictly_decreasing_function_l2891_289119

/-- A function satisfying the given condition -/
noncomputable def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, ∃ (S : Finset ℝ), ∀ y ∈ S, y > 0 ∧ (x + f y) * (y + f x) ≤ 4

/-- The main theorem -/
theorem strictly_decreasing_function 
  (f : ℝ → ℝ) (h : SatisfiesCondition f) :
  ∀ x y, 0 < x ∧ x < y → f x > f y := by
  sorry

end strictly_decreasing_function_l2891_289119


namespace overlapping_triangles_angle_sum_l2891_289138

/-- Given two overlapping triangles ABC and DEF where B and E are the same point,
    prove that the sum of angles A, B, C, D, and F is 290 degrees. -/
theorem overlapping_triangles_angle_sum
  (A B C D F : Real)
  (h1 : A = 40)
  (h2 : C = 70)
  (h3 : D = 50)
  (h4 : F = 60)
  (h5 : A + B + C = 180)  -- Sum of angles in triangle ABC
  (h6 : D + B + F = 180)  -- Sum of angles in triangle DEF (B is used instead of E)
  : A + B + C + D + F = 290 := by
  sorry

end overlapping_triangles_angle_sum_l2891_289138


namespace quadratic_roots_l2891_289198

theorem quadratic_roots (c : ℝ) : 
  (∀ x : ℝ, 2*x^2 + 6*x + c = 0 ↔ x = (-3 + Real.sqrt c) ∨ x = (-3 - Real.sqrt c)) → 
  c = 3 := by
sorry

end quadratic_roots_l2891_289198


namespace prob_not_green_correct_l2891_289101

/-- Given odds for pulling a green marble from a bag -/
def green_marble_odds : ℚ := 5 / 6

/-- The probability of not pulling a green marble -/
def prob_not_green : ℚ := 6 / 11

/-- Theorem stating that given the odds for pulling a green marble,
    the probability of not pulling a green marble is correct -/
theorem prob_not_green_correct :
  green_marble_odds = 5 / 6 →
  prob_not_green = 6 / 11 := by
sorry

end prob_not_green_correct_l2891_289101


namespace max_value_of_expression_l2891_289189

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / ((a + b + c)^4 - 79 * (a * b * c)^(4/3))
  A ≤ 3 ∧ ∃ (x : ℝ), x > 0 ∧ 
    let A' := (3 * x^3 * (2 * x)) / ((3 * x)^4 - 79 * x^4)
    A' = 3 :=
by sorry

end max_value_of_expression_l2891_289189


namespace ellipse_hyperbola_coinciding_foci_l2891_289154

/-- The value of b^2 for an ellipse and hyperbola with coinciding foci -/
theorem ellipse_hyperbola_coinciding_foci (b : ℝ) : 
  (∃ (x y : ℝ), x^2/25 + y^2/b^2 = 1) ∧ 
  (∃ (x y : ℝ), x^2/169 - y^2/144 = 1/36) ∧
  (∀ (x y : ℝ), x^2/25 + y^2/b^2 = 1 ↔ x^2/169 - y^2/144 = 1/36) →
  b^2 = 587/36 := by
sorry

end ellipse_hyperbola_coinciding_foci_l2891_289154


namespace car_distance_l2891_289129

theorem car_distance (time : ℝ) (cyclist_distance : ℝ) (speed_difference : ℝ) :
  time = 8 →
  cyclist_distance = 88 →
  speed_difference = 5 →
  let cyclist_speed := cyclist_distance / time
  let car_speed := cyclist_speed + speed_difference
  car_speed * time = 128 := by
  sorry

end car_distance_l2891_289129


namespace book_arrangement_count_l2891_289175

/-- The number of ways to arrange math and history books on a shelf -/
def arrange_books (num_math_books num_history_books : ℕ) : ℕ :=
  let end_arrangements := num_math_books * (num_math_books - 1)
  let remaining_math_arrangements := 2  -- factorial of (num_math_books - 2)
  let history_distributions := (Nat.choose num_history_books 2) * 
                               (Nat.choose (num_history_books - 2) 2) *
                               2  -- Last 2 is automatic
  let history_permutations := (2 * 2 * 2)  -- 2! for each of the 3 slots
  end_arrangements * remaining_math_arrangements * history_distributions * history_permutations

/-- Theorem stating the number of ways to arrange the books -/
theorem book_arrangement_count :
  arrange_books 4 6 = 17280 :=
by sorry

end book_arrangement_count_l2891_289175


namespace lakota_used_cd_count_l2891_289180

/-- The price of a new CD in dollars -/
def new_cd_price : ℝ := 17.99

/-- The price of a used CD in dollars -/
def used_cd_price : ℝ := 9.99

/-- The number of new CDs Lakota bought -/
def lakota_new_cds : ℕ := 6

/-- The total amount Lakota spent in dollars -/
def lakota_total : ℝ := 127.92

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new_cds : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used_cds : ℕ := 8

/-- The total amount Mackenzie spent in dollars -/
def mackenzie_total : ℝ := 133.89

/-- The number of used CDs Lakota bought -/
def lakota_used_cds : ℕ := 2

theorem lakota_used_cd_count : 
  lakota_new_cds * new_cd_price + lakota_used_cds * used_cd_price = lakota_total ∧
  mackenzie_new_cds * new_cd_price + mackenzie_used_cds * used_cd_price = mackenzie_total :=
by sorry

end lakota_used_cd_count_l2891_289180


namespace cube_roots_of_unity_l2891_289199

theorem cube_roots_of_unity :
  let z₁ : ℂ := 1
  let z₂ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₃ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  (z₁^3 = 1) ∧ (z₂^3 = 1) ∧ (z₃^3 = 1) ∧
  ∀ z : ℂ, z^3 = 1 → (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by
  sorry

end cube_roots_of_unity_l2891_289199


namespace circle_center_l2891_289184

theorem circle_center (c : ℝ × ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    (∀ p : ℝ × ℝ, (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 → 
      (3 * p.1 + 4 * p.2 = 24 ∨ 3 * p.1 + 4 * p.2 = -6))) ∧ 
  c.1 - 3 * c.2 = 0 → 
  c = (27/13, 9/13) := by
sorry


end circle_center_l2891_289184


namespace partial_fraction_decomposition_l2891_289151

theorem partial_fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
    (B * x - 19) / (x^2 - 8*x + 15) = A / (x - 3) + 5 / (x - 5)) →
  A + B = 33/5 := by
sorry

end partial_fraction_decomposition_l2891_289151


namespace partial_fraction_decomposition_l2891_289139

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 10 → x ≠ -5 →
  (8 * x - 3) / (x^2 - 5*x - 50) = (77/15) / (x - 10) + (43/15) / (x + 5) :=
by
  sorry

#check partial_fraction_decomposition

end partial_fraction_decomposition_l2891_289139


namespace remainder_sum_powers_mod_five_l2891_289123

theorem remainder_sum_powers_mod_five :
  (Nat.pow 9 5 + Nat.pow 8 7 + Nat.pow 7 6) % 5 = 1 := by
  sorry

end remainder_sum_powers_mod_five_l2891_289123


namespace door_height_calculation_l2891_289124

/-- Calculates the height of a door in a room given the room dimensions, 
    whitewashing cost, window dimensions, and total cost. -/
theorem door_height_calculation 
  (room_length room_width room_height : ℝ)
  (door_width : ℝ)
  (window_width window_height : ℝ)
  (num_windows : ℕ)
  (whitewash_cost_per_sqft : ℝ)
  (total_cost : ℝ) :
  room_length = 25 ∧ 
  room_width = 15 ∧ 
  room_height = 12 ∧
  door_width = 6 ∧
  window_width = 4 ∧
  window_height = 3 ∧
  num_windows = 3 ∧
  whitewash_cost_per_sqft = 2 ∧
  total_cost = 1812 →
  ∃ (door_height : ℝ),
    door_height = 3 ∧
    total_cost = whitewash_cost_per_sqft * 
      (2 * (room_length + room_width) * room_height - 
       (door_width * door_height + num_windows * window_width * window_height)) :=
by sorry

end door_height_calculation_l2891_289124


namespace hyperbola_equation_l2891_289100

-- Define the ellipse D
def ellipse_D (x y : ℝ) : Prop := x^2 / 50 + y^2 / 25 = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 9

-- Define the hyperbola G
def hyperbola_G (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci of ellipse D
def foci_D : Set (ℝ × ℝ) := {(-5, 0), (5, 0)}

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y : ℝ,
  (∀ x' y' : ℝ, ellipse_D x' y' → foci_D = {(-5, 0), (5, 0)}) →
  (∀ x' y' : ℝ, hyperbola_G x' y' → foci_D = {(-5, 0), (5, 0)}) →
  (∃ a b : ℝ, ∀ x' y' : ℝ, (b * x' = a * y' ∨ b * x' = -a * y') →
    ∃ t : ℝ, circle_M (x' + t) (y' + t)) →
  hyperbola_G x y := by sorry

end hyperbola_equation_l2891_289100


namespace johnsonville_marching_band_max_size_l2891_289161

theorem johnsonville_marching_band_max_size :
  ∀ m : ℕ,
  (∃ k : ℕ, 30 * m = 34 * k + 2) →
  30 * m < 1500 →
  (∀ n : ℕ, (∃ j : ℕ, 30 * n = 34 * j + 2) → 30 * n < 1500 → 30 * n ≤ 30 * m) →
  30 * m = 1260 :=
by sorry

end johnsonville_marching_band_max_size_l2891_289161


namespace distinct_products_count_l2891_289146

def S : Finset ℕ := {2, 3, 5, 7, 13}

def products (s : Finset ℕ) : Finset ℕ :=
  (Finset.powerset s).filter (λ t => t.card ≥ 2) |>.image (λ t => t.prod id)

theorem distinct_products_count : (products S).card = 26 := by
  sorry

end distinct_products_count_l2891_289146


namespace negation_of_universal_proposition_l2891_289178

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := by sorry

end negation_of_universal_proposition_l2891_289178


namespace orange_cost_l2891_289152

/-- Given that three dozen oranges cost $18.00, prove that four dozen oranges at the same rate cost $24.00 -/
theorem orange_cost (cost_three_dozen : ℝ) (h1 : cost_three_dozen = 18) :
  let cost_per_dozen := cost_three_dozen / 3
  cost_per_dozen * 4 = 24 := by
  sorry

end orange_cost_l2891_289152


namespace golden_state_total_points_l2891_289188

/-- The Golden State Team's total points calculation -/
theorem golden_state_total_points :
  let draymond_points : ℕ := 12
  let curry_points : ℕ := 2 * draymond_points
  let kelly_points : ℕ := 9
  let durant_points : ℕ := 2 * kelly_points
  let klay_points : ℕ := draymond_points / 2
  draymond_points + curry_points + kelly_points + durant_points + klay_points = 69 := by
  sorry

end golden_state_total_points_l2891_289188


namespace coffee_shop_sales_teas_sold_l2891_289106

/-- The number of teas sold at a coffee shop -/
def num_teas : ℕ := 6

/-- The number of lattes sold at a coffee shop -/
def num_lattes : ℕ := 32

/-- Theorem stating the relationship between lattes and teas sold -/
theorem coffee_shop_sales : num_lattes = 4 * num_teas + 8 := by
  sorry

/-- Theorem proving the number of teas sold -/
theorem teas_sold : num_teas = 6 := by
  sorry

end coffee_shop_sales_teas_sold_l2891_289106


namespace consecutive_cube_divisible_l2891_289122

theorem consecutive_cube_divisible (k : ℕ+) :
  ∃ n : ℤ, ∀ j : ℕ, j ∈ Finset.range k →
    ∃ m : ℕ, m > 1 ∧ (n + j : ℤ) % (m^3 : ℤ) = 0 :=
sorry

end consecutive_cube_divisible_l2891_289122


namespace solution_difference_l2891_289136

theorem solution_difference (x : ℝ) : 
  (∃ y : ℝ, (7 - y^2 / 4)^(1/3) = -3 ∧ y ≠ x ∧ (7 - x^2 / 4)^(1/3) = -3) → 
  |x - y| = 2 * Real.sqrt 136 := by
sorry

end solution_difference_l2891_289136


namespace triangle_angle_inequality_l2891_289177

-- Define a triangle
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to A
  b : ℝ  -- side opposite to B
  c : ℝ  -- side opposite to C
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  
-- State the theorem
theorem triangle_angle_inequality (t : Triangle) :
  Real.sin t.A * Real.sin t.B > Real.sin t.C ^ 2 → t.C < π / 3 := by
  sorry

end triangle_angle_inequality_l2891_289177


namespace difference_of_squares_factorization_l2891_289163

theorem difference_of_squares_factorization (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end difference_of_squares_factorization_l2891_289163


namespace oliver_new_socks_l2891_289131

theorem oliver_new_socks (initial_socks : ℕ) (thrown_away : ℕ) (final_socks : ℕ)
  (h1 : initial_socks = 11)
  (h2 : thrown_away = 4)
  (h3 : final_socks = 33) :
  final_socks - (initial_socks - thrown_away) = 26 := by
  sorry

end oliver_new_socks_l2891_289131


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l2891_289104

-- Problem 1
theorem problem_1 : -|-2/3 - 3/2| - |-1/5 + (-2/5)| = -83/30 := by sorry

-- Problem 2
theorem problem_2 : (-7.33) * 42.07 + (-2.07) * (-7.33) = -293.2 := by sorry

-- Problem 3
theorem problem_3 : -4 - 28 - (-19) + (-24) = -37 := by sorry

-- Problem 4
theorem problem_4 : -|-2023| - (-2023) + 2023 = 2023 := by sorry

-- Problem 5
theorem problem_5 : 19 * (31/32) * (-4) = -79 - 7/8 := by sorry

-- Problem 6
theorem problem_6 : (1/2 + 5/6 - 7/12) * (-36) = -27 := by sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l2891_289104


namespace length_of_AE_l2891_289110

/-- Given four points A, B, C, D on a 2D plane, and E as the intersection of segments AB and CD,
    prove that the length of AE is 5√5/3. -/
theorem length_of_AE (A B C D E : ℝ × ℝ) : 
  A = (0, 3) →
  B = (6, 0) →
  C = (4, 2) →
  D = (2, 0) →
  E.1 = 10/3 →
  E.2 = 4/3 →
  (E.2 - A.2) / (E.1 - A.1) = (B.2 - A.2) / (B.1 - A.1) →
  (E.2 - C.2) / (E.1 - C.1) = (D.2 - C.2) / (D.1 - C.1) →
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 5 * Real.sqrt 5 / 3 := by
  sorry

end length_of_AE_l2891_289110


namespace sports_equipment_pricing_and_purchasing_l2891_289173

theorem sports_equipment_pricing_and_purchasing (x y a b : ℤ) : 
  (2 * x + y = 330) →
  (5 * x + 2 * y = 780) →
  (120 * a + 90 * b = 810) →
  (x = 120 ∧ y = 90) ∧ (a = 3 ∧ b = 5) :=
by sorry

end sports_equipment_pricing_and_purchasing_l2891_289173


namespace inequality_theorem_l2891_289172

-- Define the inequality and its solution set
def inequality (m : ℝ) (x : ℝ) : Prop := m - |x - 2| ≥ 1
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | inequality m x}

-- Define the theorem
theorem inequality_theorem (m : ℝ) 
  (h1 : solution_set m = Set.Icc 0 4) 
  (a b : ℝ) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : a + b = m) : 
  m = 3 ∧ ∃ (min : ℝ), min = 9/2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = m → a^2 + b^2 ≥ min :=
sorry

end inequality_theorem_l2891_289172


namespace tan_eleven_pi_fourths_l2891_289183

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by
  sorry

end tan_eleven_pi_fourths_l2891_289183


namespace probability_square_or_circle_l2891_289193

/- Define the total number of figures -/
def total_figures : ℕ := 10

/- Define the number of squares -/
def num_squares : ℕ := 4

/- Define the number of circles -/
def num_circles : ℕ := 3

/- Theorem statement -/
theorem probability_square_or_circle :
  (num_squares + num_circles : ℚ) / total_figures = 7 / 10 := by
  sorry

end probability_square_or_circle_l2891_289193


namespace line_passes_through_fixed_point_l2891_289165

/-- Trajectory M in the Cartesian plane -/
def trajectory_M (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ 2 ∧ x ≠ -2

/-- Line l in the Cartesian plane -/
def line_l (k m x y : ℝ) : Prop :=
  y = k * x + m

/-- Intersection points of line l and trajectory M -/
def intersection_points (k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory_M x₁ y₁ ∧ trajectory_M x₂ y₂ ∧
    line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ ∧
    x₁ ≠ x₂

/-- Angle condition for F₂P and F₂Q -/
def angle_condition (k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory_M x₁ y₁ ∧ trajectory_M x₂ y₂ ∧
    line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ ∧
    (y₁ - 0) / (x₁ - 1) + (y₂ - 0) / (x₂ - 1) = 0

theorem line_passes_through_fixed_point :
  ∀ k m : ℝ,
    k ≠ 0 →
    intersection_points k m →
    angle_condition k m →
    ∃ x y : ℝ, x = 4 ∧ y = 0 ∧ line_l k m x y :=
sorry

end line_passes_through_fixed_point_l2891_289165


namespace magician_trick_l2891_289158

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem magician_trick (numbers : Finset ℕ) (a d : ℕ) :
  numbers = Finset.range 16 →
  a ∈ numbers →
  d ∈ numbers →
  is_even a →
  is_even d →
  ∃ (b c : ℕ), b ∈ numbers ∧ c ∈ numbers ∧
    (b < c ∨ (b > c ∧ a < d)) ∧
    (c < d ∨ (c > d ∧ b < a)) →
  a * d = 120 := by
  sorry

end magician_trick_l2891_289158


namespace basic_computer_price_l2891_289174

theorem basic_computer_price 
  (total_price : ℝ) 
  (price_difference : ℝ) 
  (printer_ratio : ℝ) :
  total_price = 2500 →
  price_difference = 500 →
  printer_ratio = 1/6 →
  ∃ (basic_price printer_price : ℝ),
    basic_price + printer_price = total_price ∧
    printer_price = printer_ratio * (basic_price + price_difference + printer_price) ∧
    basic_price = 2000 :=
by sorry

end basic_computer_price_l2891_289174


namespace min_points_to_win_correct_l2891_289105

/-- Represents a chess tournament with 6 players where each player plays 2 games against every other player. -/
structure ChessTournament where
  num_players : ℕ
  games_per_pair : ℕ
  win_points : ℚ
  draw_points : ℚ
  loss_points : ℚ

/-- The minimum number of points needed to guarantee a player has more points than any other player -/
def min_points_to_win (t : ChessTournament) : ℚ := 9.5

/-- Theorem stating that 9.5 points is the minimum required to guarantee winning the tournament -/
theorem min_points_to_win_correct (t : ChessTournament) 
  (h1 : t.num_players = 6)
  (h2 : t.games_per_pair = 2)
  (h3 : t.win_points = 1)
  (h4 : t.draw_points = 0.5)
  (h5 : t.loss_points = 0) :
  ∀ (p : ℚ), p < min_points_to_win t → 
  ∃ (other_player_points : ℚ), other_player_points ≥ p ∧ other_player_points ≤ (t.num_players - 1) * t.games_per_pair * t.win_points :=
sorry

end min_points_to_win_correct_l2891_289105


namespace hyperbola_asymptote_angle_l2891_289142

/-- For a hyperbola x²/a² - y²/b² = 1 with a > b, if the angle between asymptotes is 45°, then a/b = 1 -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (angle_between_asymptotes = Real.pi / 4) → 
  a / b = 1 := by
sorry

end hyperbola_asymptote_angle_l2891_289142


namespace inequality_proof_l2891_289114

theorem inequality_proof (a b c d : ℝ) :
  (a^2 - a + 1) * (b^2 - b + 1) * (c^2 - c + 1) * (d^2 - d + 1) ≥ 
  9/16 * (a - b) * (b - c) * (c - d) * (d - a) := by
  sorry

end inequality_proof_l2891_289114


namespace sqrt_of_squared_negative_l2891_289111

theorem sqrt_of_squared_negative : Real.sqrt ((-5)^2) = 5 := by sorry

end sqrt_of_squared_negative_l2891_289111


namespace pet_store_parrots_l2891_289194

/-- The number of bird cages in the pet store -/
def num_cages : ℝ := 6.0

/-- The number of parakeets in the pet store -/
def num_parakeets : ℝ := 2.0

/-- The average number of birds that can occupy 1 cage -/
def birds_per_cage : ℝ := 1.333333333

/-- The number of parrots in the pet store -/
def num_parrots : ℝ := 6.0

theorem pet_store_parrots :
  num_parrots = num_cages * birds_per_cage - num_parakeets :=
by sorry

end pet_store_parrots_l2891_289194


namespace election_winner_votes_l2891_289185

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  winner_percentage = 62 / 100 →
  vote_difference = 324 →
  (winner_percentage * total_votes).num = (winner_percentage * total_votes).den * 837 :=
by sorry

end election_winner_votes_l2891_289185


namespace a_minus_b_value_l2891_289157

theorem a_minus_b_value (a b : ℚ) 
  (ha : |a| = 5) 
  (hb : |b| = 2) 
  (hab : |a + b| = a + b) : 
  a - b = 3 ∨ a - b = 7 :=
sorry

end a_minus_b_value_l2891_289157


namespace bernie_postcards_final_count_l2891_289156

/-- Calculates the number of postcards Bernie has after his transactions -/
def postcards_after_transactions (initial_postcards : ℕ) (sell_price : ℕ) (buy_price : ℕ) : ℕ :=
  let sold_postcards := initial_postcards / 2
  let remaining_postcards := initial_postcards - sold_postcards
  let money_earned := sold_postcards * sell_price
  let new_postcards := money_earned / buy_price
  remaining_postcards + new_postcards

/-- Theorem stating that Bernie will have 36 postcards after his transactions -/
theorem bernie_postcards_final_count :
  postcards_after_transactions 18 15 5 = 36 := by
  sorry

end bernie_postcards_final_count_l2891_289156


namespace sum_of_children_ages_l2891_289197

/-- Represents the ages of Cynthia's children -/
structure ChildrenAges where
  freddy : ℕ
  matthew : ℕ
  rebecca : ℕ

/-- Theorem stating the sum of Cynthia's children's ages -/
theorem sum_of_children_ages (ages : ChildrenAges) : 
  ages.freddy = 15 → 
  ages.matthew = ages.freddy - 4 → 
  ages.rebecca = ages.matthew - 2 → 
  ages.freddy + ages.matthew + ages.rebecca = 35 := by
  sorry


end sum_of_children_ages_l2891_289197


namespace carpet_width_calculation_l2891_289191

theorem carpet_width_calculation (room_length room_width carpet_cost_per_sqm total_cost : ℝ) 
  (h1 : room_length = 13)
  (h2 : room_width = 9)
  (h3 : carpet_cost_per_sqm = 12)
  (h4 : total_cost = 1872) : 
  (total_cost / carpet_cost_per_sqm / room_length) * 100 = 1200 := by
  sorry

end carpet_width_calculation_l2891_289191


namespace sin_35pi_over_6_l2891_289195

theorem sin_35pi_over_6 : Real.sin (35 * π / 6) = -1/2 := by
  sorry

end sin_35pi_over_6_l2891_289195


namespace complex_real_condition_l2891_289150

theorem complex_real_condition (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / Complex.I
  z.im = 0 → a = 0 := by
  sorry

end complex_real_condition_l2891_289150


namespace dhoni_leftover_earnings_l2891_289141

def rent_percent : ℝ := 20
def dishwasher_percent : ℝ := 15
def bills_percent : ℝ := 10
def car_percent : ℝ := 8
def grocery_percent : ℝ := 12
def tax_percent : ℝ := 5
def savings_percent : ℝ := 40

theorem dhoni_leftover_earnings : 
  let total_expenses := rent_percent + dishwasher_percent + bills_percent + car_percent + grocery_percent + tax_percent
  let remaining_after_expenses := 100 - total_expenses
  let savings := (savings_percent / 100) * remaining_after_expenses
  let leftover := remaining_after_expenses - savings
  leftover = 18 := by sorry

end dhoni_leftover_earnings_l2891_289141


namespace expression_equality_l2891_289109

theorem expression_equality : 
  (5 + 8) * (5^2 + 8^2) * (5^4 + 8^4) * (5^8 + 8^8) * (5^16 + 8^16) * (5^32 + 8^32) = 8^32 - 5^32 := by
  sorry

end expression_equality_l2891_289109


namespace same_color_probability_is_71_288_l2891_289190

/-- Represents a 24-sided die with colored sides -/
structure ColoredDie :=
  (purple : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (sparkly : ℕ)
  (total : ℕ)
  (sum_sides : purple + green + blue + yellow + sparkly = total)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.purple^2 + d.green^2 + d.blue^2 + d.yellow^2 + d.sparkly^2) / d.total^2

/-- Our specific 24-sided die -/
def our_die : ColoredDie :=
  { purple := 5
  , green := 6
  , blue := 8
  , yellow := 4
  , sparkly := 1
  , total := 24
  , sum_sides := by rfl }

theorem same_color_probability_is_71_288 :
  same_color_probability our_die = 71 / 288 := by
  sorry

end same_color_probability_is_71_288_l2891_289190


namespace tower_combinations_l2891_289134

/-- The number of different towers of height 7 that can be built using 3 red cubes, 4 blue cubes, and 2 yellow cubes -/
def num_towers : ℕ := 5040

/-- The height of the tower -/
def tower_height : ℕ := 7

/-- The number of red cubes -/
def red_cubes : ℕ := 3

/-- The number of blue cubes -/
def blue_cubes : ℕ := 4

/-- The number of yellow cubes -/
def yellow_cubes : ℕ := 2

/-- The total number of cubes -/
def total_cubes : ℕ := red_cubes + blue_cubes + yellow_cubes

theorem tower_combinations : num_towers = 5040 := by
  sorry

end tower_combinations_l2891_289134


namespace max_value_trig_expression_l2891_289108

theorem max_value_trig_expression :
  ∀ x y z : ℝ, 
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) * 
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 9 / 2 ∧
  ∃ x y z : ℝ, 
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) * 
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) = 9 / 2 :=
by sorry

end max_value_trig_expression_l2891_289108


namespace equation_solution_l2891_289168

theorem equation_solution : 
  ∃ x : ℝ, x = (8 * Real.sqrt 2) / 3 ∧ 
  Real.sqrt (9 + Real.sqrt (16 + 3*x)) + Real.sqrt (3 + Real.sqrt (4 + x)) = 3 + 3 * Real.sqrt 2 :=
by sorry

end equation_solution_l2891_289168


namespace gum_sharing_l2891_289170

/-- The number of pieces of gum each person will receive when shared equally --/
def gum_per_person (john cole aubrey maria : ℕ) : ℕ :=
  (john + cole + aubrey + maria) / 6

/-- Theorem stating that given the initial gum distribution, each person will receive 34 pieces --/
theorem gum_sharing (john cole aubrey maria : ℕ) 
  (h_john : john = 54)
  (h_cole : cole = 45)
  (h_aubrey : aubrey = 37)
  (h_maria : maria = 70) :
  gum_per_person john cole aubrey maria = 34 := by
  sorry

end gum_sharing_l2891_289170


namespace total_campers_rowing_hiking_l2891_289115

/-- The total number of campers who went rowing and hiking -/
theorem total_campers_rowing_hiking 
  (morning_rowing : ℕ) 
  (morning_hiking : ℕ) 
  (afternoon_rowing : ℕ) 
  (h1 : morning_rowing = 41)
  (h2 : morning_hiking = 4)
  (h3 : afternoon_rowing = 26) :
  morning_rowing + morning_hiking + afternoon_rowing = 71 := by
  sorry

end total_campers_rowing_hiking_l2891_289115


namespace hyperbola_equation_l2891_289182

theorem hyperbola_equation (ellipse : Real → Real → Prop)
  (hyperbola : Real → Real → Prop)
  (h1 : ∀ x y, ellipse x y ↔ x^2/27 + y^2/36 = 1)
  (h2 : ∃ x, hyperbola x 4 ∧ ellipse x 4)
  (h3 : ∀ x y, hyperbola x y → (x = 0 → y^2 = 9) ∧ (y = 0 → x^2 = 9)) :
  ∀ x y, hyperbola x y ↔ y^2/4 - x^2/5 = 1 :=
by sorry

end hyperbola_equation_l2891_289182


namespace smallest_non_prime_non_square_with_large_factors_l2891_289162

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(p ∣ n)

theorem smallest_non_prime_non_square_with_large_factors : 
  (∀ m : ℕ, m < 4087 → 
    is_prime m ∨ 
    is_square m ∨ 
    ¬(has_no_prime_factor_less_than m 60)) ∧ 
  ¬(is_prime 4087) ∧ 
  ¬(is_square 4087) ∧ 
  has_no_prime_factor_less_than 4087 60 :=
by sorry

end smallest_non_prime_non_square_with_large_factors_l2891_289162


namespace negative_sqrt_17_bound_l2891_289192

theorem negative_sqrt_17_bound : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end negative_sqrt_17_bound_l2891_289192


namespace diophantine_equation_solution_l2891_289102

theorem diophantine_equation_solution (x y z : ℤ) : x^2 + y^2 = 3*z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end diophantine_equation_solution_l2891_289102


namespace hotel_room_charge_difference_l2891_289130

theorem hotel_room_charge_difference (G R P : ℝ) : 
  P = G * 0.9 →
  R = G * 1.19999999999999986 →
  (R - P) / R * 100 = 25 := by
sorry

end hotel_room_charge_difference_l2891_289130


namespace arithmetic_sequence_and_max_lambda_l2891_289126

def sequence_a : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 - 1 / sequence_a n

def sequence_b : ℕ → ℚ
  | 0 => 20 * sequence_a 3
  | n + 1 => sequence_a n * sequence_b n

def T (n : ℕ) : ℚ := (List.range n).map sequence_b |>.sum

theorem arithmetic_sequence_and_max_lambda :
  (∀ n : ℕ, 1 / (sequence_a n - 1) = n + 1) ∧
  (∀ n : ℕ+, 2 * T n + 400 ≥ 225 * n) ∧
  (∀ ε > 0, ∃ n : ℕ+, 2 * T n + 400 < (225 + ε) * n) := by
  sorry

end arithmetic_sequence_and_max_lambda_l2891_289126
