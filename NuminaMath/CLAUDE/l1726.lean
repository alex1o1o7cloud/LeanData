import Mathlib

namespace NUMINAMATH_CALUDE_square_area_increase_l1726_172614

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.05 * s
  let original_area := s ^ 2
  let new_area := new_side ^ 2
  (new_area - original_area) / original_area = 0.1025 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l1726_172614


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l1726_172695

theorem complex_arithmetic_equality : 
  -1^10 - (13/14 - 11/12) * (4 - (-2)^2) + 1/2 / 3 = -5/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l1726_172695


namespace NUMINAMATH_CALUDE_vector_addition_l1726_172635

/-- Given two 2D vectors a and b, prove that their sum is equal to (4, 6) -/
theorem vector_addition (a b : ℝ × ℝ) (h1 : a = (6, 2)) (h2 : b = (-2, 4)) :
  a + b = (4, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l1726_172635


namespace NUMINAMATH_CALUDE_two_digit_pairs_count_l1726_172686

/-- Given two natural numbers x and y, returns true if they contain only two different digits --/
def hasTwoDigits (x y : ℕ) : Prop := sorry

/-- The number of pairs (x, y) where x and y are three-digit numbers, 
    x + y = 999, and x and y together contain only two different digits --/
def countTwoDigitPairs : ℕ := sorry

theorem two_digit_pairs_count : countTwoDigitPairs = 40 := by sorry

end NUMINAMATH_CALUDE_two_digit_pairs_count_l1726_172686


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1726_172664

theorem fraction_evaluation (a b c : ℚ) (ha : a = 7) (hb : b = 11) (hc : c = 19) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1726_172664


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1726_172625

-- Define a structure for parallelograms
structure Parallelogram where
  -- Add necessary fields (for illustration purposes)
  vertices : Fin 4 → ℝ × ℝ

-- Define properties for diagonals
def diagonals_are_equal (p : Parallelogram) : Prop :=
  -- Add definition here
  sorry

def diagonals_bisect_each_other (p : Parallelogram) : Prop :=
  -- Add definition here
  sorry

-- The theorem to prove
theorem negation_of_universal_proposition :
  (¬ ∀ p : Parallelogram, diagonals_are_equal p ∧ diagonals_bisect_each_other p) ↔
  (∃ p : Parallelogram, ¬(diagonals_are_equal p) ∨ ¬(diagonals_bisect_each_other p)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1726_172625


namespace NUMINAMATH_CALUDE_overtake_time_l1726_172661

/-- The time it takes for a faster runner to overtake and finish ahead of a slower runner -/
theorem overtake_time (initial_distance steve_speed john_speed final_distance : ℝ) 
  (h1 : initial_distance = 12)
  (h2 : steve_speed = 3.7)
  (h3 : john_speed = 4.2)
  (h4 : final_distance = 2)
  (h5 : john_speed > steve_speed) :
  (initial_distance + final_distance) / (john_speed - steve_speed) = 28 := by
  sorry

#check overtake_time

end NUMINAMATH_CALUDE_overtake_time_l1726_172661


namespace NUMINAMATH_CALUDE_intersection_P_complement_M_l1726_172699

def U : Set Int := Set.univ

def M : Set Int := {1, 2}

def P : Set Int := {-2, -1, 0, 1, 2}

theorem intersection_P_complement_M :
  P ∩ (U \ M) = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_P_complement_M_l1726_172699


namespace NUMINAMATH_CALUDE_moses_esther_difference_l1726_172603

theorem moses_esther_difference (total : ℝ) (moses_percentage : ℝ) : 
  total = 50 →
  moses_percentage = 0.4 →
  let moses_share := moses_percentage * total
  let remainder := total - moses_share
  let esther_share := remainder / 2
  moses_share - esther_share = 5 := by
  sorry

end NUMINAMATH_CALUDE_moses_esther_difference_l1726_172603


namespace NUMINAMATH_CALUDE_abc_equality_l1726_172613

theorem abc_equality (a b c x : ℝ) 
  (h : a * x^2 - b * x - c = b * x^2 - c * x - a ∧ 
       b * x^2 - c * x - a = c * x^2 - a * x - b) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_abc_equality_l1726_172613


namespace NUMINAMATH_CALUDE_no_integer_solution_for_sum_of_cubes_l1726_172658

theorem no_integer_solution_for_sum_of_cubes (n : ℤ) : 
  n % 9 = 4 → ¬∃ (x y z : ℤ), x^3 + y^3 + z^3 = n := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_sum_of_cubes_l1726_172658


namespace NUMINAMATH_CALUDE_johann_oranges_l1726_172656

theorem johann_oranges (x : ℕ) : 
  (x - 10) / 2 + 5 = 30 → x = 60 := by sorry

end NUMINAMATH_CALUDE_johann_oranges_l1726_172656


namespace NUMINAMATH_CALUDE_exterior_angle_triangle_l1726_172682

theorem exterior_angle_triangle (α β γ : ℝ) : 
  0 < α ∧ 0 < β ∧ 0 < γ →  -- angles are positive
  α + β + γ = 180 →  -- sum of angles in a triangle is 180°
  α + β = 148 →  -- exterior angle
  β = 58 →  -- one interior angle
  γ = 90  -- prove that the other interior angle is 90°
  := by sorry

end NUMINAMATH_CALUDE_exterior_angle_triangle_l1726_172682


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l1726_172629

/-- The angle with the same terminal side as α = π/12 + 2kπ (k ∈ ℤ) is equivalent to 25π/12 radians. -/
theorem same_terminal_side_angle (k : ℤ) : ∃ (n : ℤ), (π/12 + 2*k*π) = 25*π/12 + 2*n*π := by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l1726_172629


namespace NUMINAMATH_CALUDE_tan_double_angle_l1726_172621

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l1726_172621


namespace NUMINAMATH_CALUDE_transportation_budget_degrees_l1726_172683

theorem transportation_budget_degrees (salaries research_and_development utilities equipment supplies : ℝ)
  (h1 : salaries = 60)
  (h2 : research_and_development = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : supplies = 2)
  (h6 : salaries + research_and_development + utilities + equipment + supplies < 100) :
  let transportation := 100 - (salaries + research_and_development + utilities + equipment + supplies)
  (transportation / 100) * 360 = 72 := by
sorry

end NUMINAMATH_CALUDE_transportation_budget_degrees_l1726_172683


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l1726_172687

theorem recreation_spending_comparison (wages_last_week : ℝ) : 
  let recreation_last_week := 0.15 * wages_last_week
  let wages_this_week := 0.90 * wages_last_week
  let recreation_this_week := 0.30 * wages_this_week
  recreation_this_week / recreation_last_week = 1.8 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l1726_172687


namespace NUMINAMATH_CALUDE_representations_equivalence_distinct_representations_equivalence_l1726_172623

/-- The number of ways to represent a positive integer as a sum of positive integers -/
def numRepresentations (n m : ℕ+) : ℕ :=
  sorry

/-- The number of ways to represent a positive integer as a sum of distinct positive integers -/
def numDistinctRepresentations (n m : ℕ+) : ℕ :=
  sorry

/-- The number of ways to represent a positive integer as a sum of integers from a given set -/
def numRepresentationsFromSet (n : ℕ) (s : Finset ℕ) : ℕ :=
  sorry

theorem representations_equivalence (n m : ℕ+) :
  numRepresentations n m = numRepresentationsFromSet (n - m) (Finset.range m) :=
sorry

theorem distinct_representations_equivalence (n m : ℕ+) :
  numDistinctRepresentations n m = numRepresentationsFromSet (n - m * (m + 1) / 2) (Finset.range n) :=
sorry

end NUMINAMATH_CALUDE_representations_equivalence_distinct_representations_equivalence_l1726_172623


namespace NUMINAMATH_CALUDE_distribute_four_balls_four_boxes_l1726_172679

/-- The number of ways to distribute indistinguishable objects into distinguishable containers -/
def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 32 ways to distribute 4 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_four_balls_four_boxes : distribute_objects 4 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_balls_four_boxes_l1726_172679


namespace NUMINAMATH_CALUDE_tangent_perpendicular_l1726_172698

-- Define the curve C
def C (x : ℝ) : ℝ := x^2 + x

-- Define the derivative of C
def C' (x : ℝ) : ℝ := 2*x + 1

-- Define the perpendicular line
def perp_line (a x y : ℝ) : Prop := a*x - y + 1 = 0

theorem tangent_perpendicular :
  ∀ a : ℝ, 
  (C' 1 = -1/a) →  -- The slope of the tangent at x=1 is the negative reciprocal of a
  a = -1/3 := by
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_l1726_172698


namespace NUMINAMATH_CALUDE_field_ratio_l1726_172667

theorem field_ratio (field_length field_width pond_side : ℝ) : 
  field_length = 16 →
  field_length = field_width * (field_length / field_width) →
  pond_side = 4 →
  pond_side^2 = (1/8) * (field_length * field_width) →
  field_length / field_width = 2 := by
sorry

end NUMINAMATH_CALUDE_field_ratio_l1726_172667


namespace NUMINAMATH_CALUDE_half_obtuse_angle_in_first_quadrant_l1726_172696

theorem half_obtuse_angle_in_first_quadrant (α : Real) (h : π / 2 < α ∧ α < π) :
  π / 4 < α / 2 ∧ α / 2 < π / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_obtuse_angle_in_first_quadrant_l1726_172696


namespace NUMINAMATH_CALUDE_boxes_theorem_l1726_172637

/-- Represents the operation of adding or removing balls from three consecutive boxes. -/
inductive Operation
  | Add
  | Remove

/-- Represents the state of the boxes after operations. -/
def BoxState (n : ℕ) := Fin n → ℕ

/-- Defines the initial state of the boxes. -/
def initial_state (n : ℕ) : BoxState n :=
  fun i => i.val + 1

/-- Applies an operation to three consecutive boxes. -/
def apply_operation (state : BoxState n) (start : Fin n) (op : Operation) : BoxState n :=
  sorry

/-- Checks if all boxes have exactly k balls. -/
def all_equal (state : BoxState n) (k : ℕ) : Prop :=
  ∀ i : Fin n, state i = k

/-- Main theorem: Characterizes when it's possible to achieve k balls in each box. -/
theorem boxes_theorem (n : ℕ) (h : n ≥ 3) :
  ∀ k : ℕ, k > 0 →
    (∃ (final : BoxState n),
      ∃ (ops : List (Fin n × Operation)),
        all_equal final k ∧
        final = (ops.foldl (fun st (i, op) => apply_operation st i op) (initial_state n))) ↔
    ((n % 3 = 1 ∧ k % 3 = 1) ∨ (n % 3 = 2 ∧ k % 3 = 0)) :=
  sorry

end NUMINAMATH_CALUDE_boxes_theorem_l1726_172637


namespace NUMINAMATH_CALUDE_f_has_one_zero_max_ab_value_l1726_172666

noncomputable def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + Real.exp (x - 1)

theorem f_has_one_zero :
  ∃! x, f (-1) 1 x = 0 :=
sorry

theorem max_ab_value (a b : ℝ) (h : a ≠ 0) :
  (∀ x, f a b x ≤ Real.exp (x - 1) + x + 1) →
  a * b ≤ (1 / 2) * Real.exp 3 :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_max_ab_value_l1726_172666


namespace NUMINAMATH_CALUDE_bacterium_diameter_nanometers_l1726_172649

/-- Conversion factor from meters to nanometers -/
def meters_to_nanometers : ℝ := 10^9

/-- Diameter of the bacterium in meters -/
def bacterium_diameter_meters : ℝ := 0.00000285

/-- Theorem stating the diameter of the bacterium in nanometers -/
theorem bacterium_diameter_nanometers :
  bacterium_diameter_meters * meters_to_nanometers = 2.85 * 10^3 := by
  sorry

#check bacterium_diameter_nanometers

end NUMINAMATH_CALUDE_bacterium_diameter_nanometers_l1726_172649


namespace NUMINAMATH_CALUDE_somu_age_problem_l1726_172655

theorem somu_age_problem (s f : ℕ) : 
  s = f / 4 →
  s - 12 = (f - 12) / 7 →
  s = 24 :=
by sorry

end NUMINAMATH_CALUDE_somu_age_problem_l1726_172655


namespace NUMINAMATH_CALUDE_no_solutions_for_absolute_value_equation_l1726_172615

theorem no_solutions_for_absolute_value_equation :
  ¬ ∃ (x : ℝ), |x - 3| = x^2 + 2*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_absolute_value_equation_l1726_172615


namespace NUMINAMATH_CALUDE_investment_profit_distribution_l1726_172650

/-- Represents the investment and profit distribution problem -/
theorem investment_profit_distribution 
  (total_investment : ℕ) 
  (a_extra : ℕ) 
  (b_extra : ℕ) 
  (profit_ratio_a : ℕ) 
  (profit_ratio_b : ℕ) 
  (profit_ratio_c : ℕ) 
  (total_profit : ℕ) 
  (h1 : total_investment = 120000)
  (h2 : a_extra = 6000)
  (h3 : b_extra = 8000)
  (h4 : profit_ratio_a = 4)
  (h5 : profit_ratio_b = 3)
  (h6 : profit_ratio_c = 2)
  (h7 : total_profit = 50000) :
  (profit_ratio_c : ℚ) / (profit_ratio_a + profit_ratio_b + profit_ratio_c : ℚ) * total_profit = 11111.11 := by
  sorry

end NUMINAMATH_CALUDE_investment_profit_distribution_l1726_172650


namespace NUMINAMATH_CALUDE_ball_difference_l1726_172628

/-- Problem: Difference between basketballs and soccer balls --/
theorem ball_difference (total : ℕ) (soccer : ℕ) (tennis : ℕ) (baseball : ℕ) (volleyball : ℕ) (basketball : ℕ) : 
  total = 145 →
  soccer = 20 →
  tennis = 2 * soccer →
  baseball = soccer + 10 →
  volleyball = 30 →
  basketball > soccer →
  total = soccer + tennis + baseball + volleyball + basketball →
  basketball - soccer = 5 := by
  sorry

#check ball_difference

end NUMINAMATH_CALUDE_ball_difference_l1726_172628


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1726_172605

theorem sqrt_sum_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt a + Real.sqrt b ≥ Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1726_172605


namespace NUMINAMATH_CALUDE_ellipse_equation_l1726_172600

/-- An ellipse with center at the origin, right focus at (1,0), and eccentricity 1/2 -/
structure Ellipse where
  /-- The x-coordinate of a point on the ellipse -/
  x : ℝ
  /-- The y-coordinate of a point on the ellipse -/
  y : ℝ
  /-- The distance from the center to the focus -/
  c : ℝ
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The semi-major axis of the ellipse -/
  a : ℝ
  /-- The semi-minor axis of the ellipse -/
  b : ℝ
  /-- The center is at the origin -/
  center_origin : c = 1
  /-- The eccentricity is 1/2 -/
  eccentricity_half : e = 1/2
  /-- The relation between eccentricity, c, and a -/
  eccentricity_def : e = c / a
  /-- The relation between a, b, and c -/
  axis_relation : b^2 = a^2 - c^2

/-- The equation of the ellipse is x^2/4 + y^2/3 = 1 -/
theorem ellipse_equation (C : Ellipse) : C.x^2 / 4 + C.y^2 / 3 = 1 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1726_172600


namespace NUMINAMATH_CALUDE_shopkeeper_loss_theorem_l1726_172646

/-- Calculates the loss percent for a shopkeeper given profit margin and theft percentage -/
def shopkeeper_loss_percent (profit_margin : ℝ) (theft_percentage : ℝ) : ℝ :=
  let selling_price := 1 + profit_margin
  let remaining_goods := 1 - theft_percentage
  let actual_revenue := selling_price * remaining_goods
  let actual_profit := actual_revenue - remaining_goods
  let net_loss := theft_percentage - actual_profit
  net_loss * 100

/-- Theorem stating that a shopkeeper with 10% profit margin and 20% theft has a 12% loss -/
theorem shopkeeper_loss_theorem :
  shopkeeper_loss_percent 0.1 0.2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_theorem_l1726_172646


namespace NUMINAMATH_CALUDE_hockey_season_length_l1726_172670

theorem hockey_season_length 
  (games_per_month : ℕ) 
  (total_games : ℕ) 
  (h1 : games_per_month = 13) 
  (h2 : total_games = 182) : 
  total_games / games_per_month = 14 := by
sorry

end NUMINAMATH_CALUDE_hockey_season_length_l1726_172670


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l1726_172654

theorem logarithmic_equation_solution :
  ∃ x : ℝ, (Real.log x / Real.log 4) - 3 * (Real.log 8 / Real.log 2) = 1 - (Real.log 2 / Real.log 2) ∧ x = 262144 := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l1726_172654


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l1726_172675

theorem solution_satisfies_equations :
  ∃ (x y : ℝ), 3 * x - 7 * y = 2 ∧ 4 * y - x = 6 ∧ x = 10 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l1726_172675


namespace NUMINAMATH_CALUDE_monochromatic_isosceles_independent_of_coloring_l1726_172617

/-- A regular polygon with 6n+1 sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  is_regular : sides = 6 * n + 1

/-- A coloring of the vertices of a regular polygon -/
structure Coloring (n : ℕ) where
  polygon : RegularPolygon n
  red_vertices : ℕ
  valid_coloring : red_vertices ≤ polygon.sides

/-- An isosceles triangle in a regular polygon -/
structure IsoscelesTriangle (n : ℕ) where
  polygon : RegularPolygon n

/-- A monochromatic isosceles triangle (all vertices same color) -/
structure MonochromaticIsoscelesTriangle (n : ℕ) extends IsoscelesTriangle n where
  coloring : Coloring n

/-- The number of monochromatic isosceles triangles in a colored regular polygon -/
def num_monochromatic_isosceles_triangles (n : ℕ) (c : Coloring n) : ℕ := sorry

/-- The main theorem: the number of monochromatic isosceles triangles is independent of coloring -/
theorem monochromatic_isosceles_independent_of_coloring (n : ℕ) 
  (c1 c2 : Coloring n) (h : c1.red_vertices = c2.red_vertices) :
  num_monochromatic_isosceles_triangles n c1 = num_monochromatic_isosceles_triangles n c2 := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_isosceles_independent_of_coloring_l1726_172617


namespace NUMINAMATH_CALUDE_area_circle_inscribed_square_l1726_172616

/-- The area of a circle inscribed in a square with diagonal 10 meters is 12.5π square meters. -/
theorem area_circle_inscribed_square (d : ℝ) (A : ℝ) :
  d = 10 → A = π * (d / (2 * Real.sqrt 2))^2 → A = 12.5 * π := by
  sorry

end NUMINAMATH_CALUDE_area_circle_inscribed_square_l1726_172616


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l1726_172602

theorem cos_sin_sum_equals_half : 
  Real.cos (25 * π / 180) * Real.cos (85 * π / 180) + 
  Real.sin (25 * π / 180) * Real.sin (85 * π / 180) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l1726_172602


namespace NUMINAMATH_CALUDE_smallest_valid_circular_arrangement_l1726_172630

/-- A function that checks if two natural numbers share at least one digit in their decimal representation -/
def shareDigit (a b : ℕ) : Prop := sorry

/-- A function that checks if a list of natural numbers satisfies the neighboring digit condition -/
def validArrangement (lst : List ℕ) : Prop := sorry

/-- The smallest natural number N ≥ 2 for which a valid circular arrangement exists -/
def smallestValidN : ℕ := 29

theorem smallest_valid_circular_arrangement :
  (smallestValidN ≥ 2) ∧
  (∃ (lst : List ℕ), lst.length = smallestValidN ∧ 
    (∀ n, n ∈ lst ↔ 1 ≤ n ∧ n ≤ smallestValidN) ∧
    validArrangement lst) ∧
  (∀ N < smallestValidN, ¬∃ (lst : List ℕ), lst.length = N ∧
    (∀ n, n ∈ lst ↔ 1 ≤ n ∧ n ≤ N) ∧
    validArrangement lst) := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_circular_arrangement_l1726_172630


namespace NUMINAMATH_CALUDE_unique_solution_abc_l1726_172671

theorem unique_solution_abc (a b c : ℕ+) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_abc_l1726_172671


namespace NUMINAMATH_CALUDE_lines_sum_l1726_172672

-- Define the lines
def l₀ (x y : ℝ) : Prop := x - y + 1 = 0
def l₁ (a x y : ℝ) : Prop := a * x - 2 * y + 1 = 0
def l₂ (b x y : ℝ) : Prop := x + b * y + 3 = 0

-- Define perpendicularity and parallelism
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, f x₁ y₁ → f x₂ y₂ → g x₁ y₁ → g x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (y₂ - y₁) - (y₂ - y₁) * (x₂ - x₁) = 0)

def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ, 
    f x₁ y₁ → f x₂ y₂ → g x₃ y₃ → g x₄ y₄ →
    (x₂ - x₁) * (y₄ - y₃) = (y₂ - y₁) * (x₄ - x₃)

-- Theorem statement
theorem lines_sum (a b : ℝ) : 
  perpendicular (l₀) (l₁ a) → parallel (l₀) (l₂ b) → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_lines_sum_l1726_172672


namespace NUMINAMATH_CALUDE_room_width_calculation_l1726_172606

/-- Given a rectangular room with specified length, paving cost per square meter,
    and total paving cost, prove that the width of the room is as calculated. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : cost_per_sqm = 600)
    (h3 : total_cost = 12375) :
    total_cost / (cost_per_sqm * length) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1726_172606


namespace NUMINAMATH_CALUDE_smallest_average_l1726_172626

-- Define the set of digits
def digits : Finset Nat := Finset.range 9

-- Define the property of a valid selection
def valid_selection (single_digits double_digits : Finset Nat) : Prop :=
  single_digits.card = 3 ∧
  double_digits.card = 6 ∧
  (single_digits ∪ double_digits) = digits ∧
  single_digits ∩ double_digits = ∅

-- Define the average of the resulting set of numbers
def average (single_digits double_digits : Finset Nat) : ℚ :=
  let single_sum := single_digits.sum id
  let double_sum := (double_digits.filter (· ≤ 3)).sum (· * 10) +
                    (double_digits.filter (· > 3)).sum id
  (single_sum + double_sum : ℚ) / 6

-- Theorem statement
theorem smallest_average :
  ∀ single_digits double_digits : Finset Nat,
    valid_selection single_digits double_digits →
    average single_digits double_digits ≥ 33/2 :=
sorry

end NUMINAMATH_CALUDE_smallest_average_l1726_172626


namespace NUMINAMATH_CALUDE_john_needs_72_strings_l1726_172681

/-- The number of strings John needs to restring all instruments -/
def total_strings (num_basses : ℕ) (strings_per_bass : ℕ) (strings_per_guitar : ℕ) (strings_per_8string_guitar : ℕ) : ℕ :=
  let num_guitars := 2 * num_basses
  let num_8string_guitars := num_guitars - 3
  num_basses * strings_per_bass + num_guitars * strings_per_guitar + num_8string_guitars * strings_per_8string_guitar

/-- Theorem stating the total number of strings John needs -/
theorem john_needs_72_strings :
  total_strings 3 4 6 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_72_strings_l1726_172681


namespace NUMINAMATH_CALUDE_average_rate_of_change_x_squared_plus_x_l1726_172645

/-- The average rate of change of f(x) = x^2 + x on [1, 2] is 4 -/
theorem average_rate_of_change_x_squared_plus_x : ∀ (f : ℝ → ℝ),
  (∀ x, f x = x^2 + x) →
  (((f 2) - (f 1)) / (2 - 1) = 4) :=
by sorry

end NUMINAMATH_CALUDE_average_rate_of_change_x_squared_plus_x_l1726_172645


namespace NUMINAMATH_CALUDE_alvin_wood_needed_l1726_172692

/-- The number of wood pieces Alvin needs for his house -/
def total_wood_needed (friend_pieces brother_pieces more_pieces : ℕ) : ℕ :=
  friend_pieces + brother_pieces + more_pieces

/-- Theorem: Alvin needs 376 pieces of wood in total -/
theorem alvin_wood_needed :
  total_wood_needed 123 136 117 = 376 := by
  sorry

end NUMINAMATH_CALUDE_alvin_wood_needed_l1726_172692


namespace NUMINAMATH_CALUDE_mary_ate_seven_slices_l1726_172619

/-- The number of slices in a large pizza -/
def slices_per_pizza : ℕ := 8

/-- The number of pizzas Mary ordered -/
def pizzas_ordered : ℕ := 2

/-- The number of slices Mary has remaining -/
def slices_remaining : ℕ := 9

/-- The number of slices Mary ate -/
def slices_eaten : ℕ := pizzas_ordered * slices_per_pizza - slices_remaining

theorem mary_ate_seven_slices : slices_eaten = 7 := by
  sorry

end NUMINAMATH_CALUDE_mary_ate_seven_slices_l1726_172619


namespace NUMINAMATH_CALUDE_tan_sum_specific_angles_l1726_172697

theorem tan_sum_specific_angles (α β : ℝ) 
  (h1 : 2 * Real.tan α = 1) 
  (h2 : Real.tan β = -2) : 
  Real.tan (α + β) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_specific_angles_l1726_172697


namespace NUMINAMATH_CALUDE_balls_in_boxes_count_l1726_172609

def num_balls : ℕ := 6
def num_boxes : ℕ := 3

theorem balls_in_boxes_count : 
  (num_boxes : ℕ) ^ (num_balls : ℕ) = 729 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_count_l1726_172609


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l1726_172607

theorem trader_gain_percentage (cost : ℝ) (h : cost > 0) :
  let gain := 30 * cost
  let cost_price := 100 * cost
  let gain_percentage := (gain / cost_price) * 100
  gain_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l1726_172607


namespace NUMINAMATH_CALUDE_token_game_1994_token_game_1991_l1726_172640

/-- Represents the state of the token-passing game -/
structure GameState (N : ℕ) where
  tokens : Fin N → ℕ
  total_tokens : ℕ

/-- Defines a single move in the game -/
def move (state : GameState N) (i : Fin N) : GameState N :=
  sorry

/-- Determines if the game has terminated -/
def is_terminated (state : GameState N) : Prop :=
  ∀ i, state.tokens i ≤ 1

/-- Theorem for the token-passing game with 1994 girls -/
theorem token_game_1994 (n : ℕ) :
  (n < 1994 → ∃ (final_state : GameState 1994), is_terminated final_state) ∧
  (n = 1994 → ¬∃ (final_state : GameState 1994), is_terminated final_state) :=
  sorry

/-- Theorem for the token-passing game with 1991 girls -/
theorem token_game_1991 (n : ℕ) :
  n ≤ 1991 → ¬∃ (final_state : GameState 1991), is_terminated final_state :=
  sorry

end NUMINAMATH_CALUDE_token_game_1994_token_game_1991_l1726_172640


namespace NUMINAMATH_CALUDE_square_sum_product_l1726_172620

theorem square_sum_product (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 24) :
  (x^2 + y^2) * (x + y) = 803 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l1726_172620


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l1726_172638

theorem min_sum_reciprocals (w x y z : ℝ) 
  (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : w + x + y + z = 1) :
  1/w + 1/x + 1/y + 1/z ≥ 16 ∧
  (1/w + 1/x + 1/y + 1/z = 16 ↔ w = 1/4 ∧ x = 1/4 ∧ y = 1/4 ∧ z = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l1726_172638


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l1726_172689

/-- The length of a marathon in miles -/
def marathon_miles : ℕ := 28

/-- The additional yards in a marathon beyond the whole miles -/
def marathon_extra_yards : ℕ := 1500

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The number of marathons run -/
def marathons_run : ℕ := 15

/-- The total number of yards run in all marathons -/
def total_yards : ℕ := marathons_run * (marathon_miles * yards_per_mile + marathon_extra_yards)

/-- The remainder of yards after converting total yards to miles -/
def remainder_yards : ℕ := total_yards % yards_per_mile

theorem marathon_remainder_yards : remainder_yards = 1200 := by
  sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l1726_172689


namespace NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l1726_172642

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + m = 0

-- Define what it means for the equation to represent a circle
def is_circle (m : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem stating that m = 0 is sufficient but not necessary
theorem m_zero_sufficient_not_necessary :
  (is_circle 0) ∧ (∃ m : ℝ, m ≠ 0 ∧ is_circle m) :=
sorry

end NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l1726_172642


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l1726_172624

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l1726_172624


namespace NUMINAMATH_CALUDE_cloth_coloring_problem_l1726_172662

/-- Calculates the length of cloth colored by a group of women in a given number of days -/
def clothLength (women : ℕ) (days : ℕ) (rate : ℝ) : ℝ :=
  women * days * rate

theorem cloth_coloring_problem (rate : ℝ) (h1 : rate > 0) :
  clothLength 5 1 rate = 100 →
  clothLength 6 3 rate = 360 := by
  sorry

end NUMINAMATH_CALUDE_cloth_coloring_problem_l1726_172662


namespace NUMINAMATH_CALUDE_small_rectangle_perimeter_l1726_172604

/-- Given a square with perimeter 256 units divided into 16 equal smaller squares,
    each further divided into two rectangles along a diagonal,
    the perimeter of one of these smaller rectangles is 32 + 16√2 units. -/
theorem small_rectangle_perimeter (large_square_perimeter : ℝ) 
  (h1 : large_square_perimeter = 256) 
  (num_divisions : ℕ) 
  (h2 : num_divisions = 16) : ℝ :=
by
  -- Define the perimeter of one small rectangle
  let small_rectangle_perimeter := 32 + 16 * Real.sqrt 2
  
  -- Prove that this is indeed the perimeter
  sorry

#check small_rectangle_perimeter

end NUMINAMATH_CALUDE_small_rectangle_perimeter_l1726_172604


namespace NUMINAMATH_CALUDE_log_8_x_equals_3_5_l1726_172647

theorem log_8_x_equals_3_5 (x : ℝ) : 
  Real.log x / Real.log 8 = 3.5 → x = 181.04 := by
  sorry

end NUMINAMATH_CALUDE_log_8_x_equals_3_5_l1726_172647


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_l1726_172608

theorem greatest_common_divisor_of_differences (a b c : ℕ) (h : a < b ∧ b < c) :
  ∃ d : ℕ, d > 0 ∧ 
    (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
    (∀ k : ℕ, k > d → ¬(∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s)) →
  (Nat.gcd (b - a) (c - b) = 10) →
  (a = 20 ∧ b = 40 ∧ c = 90) →
  (∃ d : ℕ, d = 10 ∧ d > 0 ∧ 
    (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
    (∀ k : ℕ, k > d → ¬(∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_l1726_172608


namespace NUMINAMATH_CALUDE_equal_discriminants_l1726_172694

/-- A monic quadratic polynomial with distinct roots -/
structure MonicQuadratic where
  a : ℝ
  b : ℝ
  distinct_roots : a ≠ b

/-- The value of a monic quadratic polynomial at a given point -/
def evaluate (p : MonicQuadratic) (x : ℝ) : ℝ :=
  (x - p.a) * (x - p.b)

/-- The discriminant of a monic quadratic polynomial -/
def discriminant (p : MonicQuadratic) : ℝ :=
  (p.a - p.b)^2

theorem equal_discriminants (P Q : MonicQuadratic)
  (h : evaluate Q P.a + evaluate Q P.b = evaluate P Q.a + evaluate P Q.b) :
  discriminant P = discriminant Q := by
  sorry

end NUMINAMATH_CALUDE_equal_discriminants_l1726_172694


namespace NUMINAMATH_CALUDE_water_depth_calculation_l1726_172652

def rons_height : ℝ := 13

def water_depth : ℝ := 16 * rons_height

theorem water_depth_calculation : water_depth = 208 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_calculation_l1726_172652


namespace NUMINAMATH_CALUDE_sin_300_degrees_l1726_172653

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l1726_172653


namespace NUMINAMATH_CALUDE_range_of_m_l1726_172680

-- Define the conditions
def condition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m*x + 3/2 > 0

def condition_q (m : ℝ) : Prop :=
  m > 1 ∧ m < 3 -- Simplified condition for foci on x-axis

-- Define the theorem
theorem range_of_m (m : ℝ) :
  condition_p m ∧ condition_q m → 2 < m ∧ m < Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1726_172680


namespace NUMINAMATH_CALUDE_unfair_coin_expected_value_l1726_172690

/-- The expected value of an unfair coin flip -/
theorem unfair_coin_expected_value :
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := 9
  p_heads * gain_heads + p_tails * (-loss_tails) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_unfair_coin_expected_value_l1726_172690


namespace NUMINAMATH_CALUDE_modular_congruence_in_range_l1726_172601

theorem modular_congruence_in_range : ∃ n : ℤ, 5 ≤ n ∧ n ≤ 12 ∧ n ≡ 10569 [ZMOD 7] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_in_range_l1726_172601


namespace NUMINAMATH_CALUDE_shifted_linear_function_equation_l1726_172632

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Shifts a linear function vertically by a given amount -/
def shiftVertically (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { slope := f.slope, yIntercept := f.yIntercept + shift }

theorem shifted_linear_function_equation 
  (f : LinearFunction) 
  (h1 : f.slope = 2) 
  (h2 : f.yIntercept = -3) :
  (shiftVertically f 3).yIntercept = 0 := by
  sorry

#check shifted_linear_function_equation

end NUMINAMATH_CALUDE_shifted_linear_function_equation_l1726_172632


namespace NUMINAMATH_CALUDE_cheryl_m_and_ms_l1726_172643

/-- Cheryl's m&m's problem -/
theorem cheryl_m_and_ms 
  (initial : ℕ) 
  (after_dinner : ℕ) 
  (given_to_sister : ℕ) 
  (h1 : initial = 25) 
  (h2 : after_dinner = 5) 
  (h3 : given_to_sister = 13) :
  initial - (after_dinner + given_to_sister) = 7 :=
by sorry

end NUMINAMATH_CALUDE_cheryl_m_and_ms_l1726_172643


namespace NUMINAMATH_CALUDE_onion_transport_trips_l1726_172660

theorem onion_transport_trips (bags_per_trip : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ) : 
  bags_per_trip = 10 → weight_per_bag = 50 → total_weight = 10000 →
  (total_weight / (bags_per_trip * weight_per_bag) : ℕ) = 20 := by
sorry

end NUMINAMATH_CALUDE_onion_transport_trips_l1726_172660


namespace NUMINAMATH_CALUDE_franks_mower_blades_expenditure_l1726_172684

theorem franks_mower_blades_expenditure 
  (total_earned : ℕ) 
  (games_affordable : ℕ) 
  (game_price : ℕ) 
  (h1 : total_earned = 19)
  (h2 : games_affordable = 4)
  (h3 : game_price = 2) :
  total_earned - games_affordable * game_price = 11 := by
sorry

end NUMINAMATH_CALUDE_franks_mower_blades_expenditure_l1726_172684


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1726_172674

theorem quadratic_factorization (a b : ℕ) : 
  (∀ x, x^2 - 20*x + 96 = (x - a)*(x - b)) →
  a > b →
  4*b - a = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1726_172674


namespace NUMINAMATH_CALUDE_current_speed_l1726_172610

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 22)
  (h2 : speed_against_current = 12) :
  ∃ (current_speed : ℝ), current_speed = 5 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by sorry

end NUMINAMATH_CALUDE_current_speed_l1726_172610


namespace NUMINAMATH_CALUDE_circles_intersect_l1726_172612

/-- Two circles are intersecting if the distance between their centers is less than the sum of their radii
    and greater than the absolute difference of their radii. -/
def circles_intersecting (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : Prop :=
  let distance := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  distance < radius1 + radius2 ∧ distance > |radius1 - radius2|

/-- Given two circles: (x-a)^2+(y-b)^2=4 and (x-a-1)^2+(y-b-2)^2=1 where a, b ∈ ℝ,
    prove that they are intersecting. -/
theorem circles_intersect (a b : ℝ) : 
  circles_intersecting (a, b) (a+1, b+2) 2 1 := by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_l1726_172612


namespace NUMINAMATH_CALUDE_sum_five_probability_l1726_172659

theorem sum_five_probability (n : ℕ) : n ≥ 5 →
  (Nat.choose n 2 : ℚ)⁻¹ * 2 = 1 / 14 ↔ n = 8 := by sorry

end NUMINAMATH_CALUDE_sum_five_probability_l1726_172659


namespace NUMINAMATH_CALUDE_solution_set_f_range_g_a_gt_2_range_g_a_lt_2_range_g_a_eq_2_l1726_172685

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + x

-- Define the function g
def g (a x : ℝ) : ℝ := f x - |a*x - 1| - x

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f (x : ℝ) : 
  f x ≤ 5 ↔ x ∈ Set.Icc (-6) (4/3) :=
sorry

-- Theorem for the range of g(x) when a > 2
theorem range_g_a_gt_2 (a : ℝ) (h : a > 2) :
  Set.range (g a) = Set.Iic (2/a + 1) :=
sorry

-- Theorem for the range of g(x) when 0 < a < 2
theorem range_g_a_lt_2 (a : ℝ) (h1 : a > 0) (h2 : a < 2) :
  Set.range (g a) = Set.Ici (-a/2 - 1) :=
sorry

-- Theorem for the range of g(x) when a = 2
theorem range_g_a_eq_2 :
  Set.range (g 2) = Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_g_a_gt_2_range_g_a_lt_2_range_g_a_eq_2_l1726_172685


namespace NUMINAMATH_CALUDE_initial_blue_pens_l1726_172657

theorem initial_blue_pens (initial_black : ℕ) (initial_red : ℕ) 
  (blue_removed : ℕ) (black_removed : ℕ) (remaining : ℕ) :
  initial_black = 21 →
  initial_red = 6 →
  blue_removed = 4 →
  black_removed = 7 →
  remaining = 25 →
  ∃ initial_blue : ℕ, 
    initial_blue + initial_black + initial_red = 
    remaining + blue_removed + black_removed ∧
    initial_blue = 9 :=
by sorry

end NUMINAMATH_CALUDE_initial_blue_pens_l1726_172657


namespace NUMINAMATH_CALUDE_sam_drew_age_multiple_l1726_172627

/-- Proves that in five years, Sam's age divided by Drew's age equals 3 -/
theorem sam_drew_age_multiple (drew_current_age sam_current_age : ℕ) : 
  drew_current_age = 12 →
  sam_current_age = 46 →
  (sam_current_age + 5) / (drew_current_age + 5) = 3 := by
sorry

end NUMINAMATH_CALUDE_sam_drew_age_multiple_l1726_172627


namespace NUMINAMATH_CALUDE_student_fails_by_10_marks_l1726_172669

/-- Calculates the number of marks a student fails by in a test -/
def marksFailed (maxMarks : ℕ) (passingPercentage : ℚ) (studentScore : ℕ) : ℕ :=
  let passingMark := (maxMarks : ℚ) * passingPercentage
  (passingMark.ceil - studentScore).toNat

/-- Proves that a student who scores 80 marks in a 300-mark test with 30% passing requirement fails by 10 marks -/
theorem student_fails_by_10_marks :
  marksFailed 300 (30 / 100) 80 = 10 := by
  sorry

end NUMINAMATH_CALUDE_student_fails_by_10_marks_l1726_172669


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1726_172633

theorem trigonometric_simplification (α : ℝ) : 
  Real.sin (π / 2 + α) * Real.cos (α - π / 3) + Real.sin (π - α) * Real.sin (α - π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1726_172633


namespace NUMINAMATH_CALUDE_allocation_schemes_l1726_172693

/-- The number of ways to allocate teachers to buses -/
def allocate_teachers (n : ℕ) (m : ℕ) : ℕ :=
  sorry

/-- There are 3 buses -/
def num_buses : ℕ := 3

/-- There are 5 teachers -/
def num_teachers : ℕ := 5

/-- Each bus must have at least one teacher -/
axiom at_least_one_teacher (b : ℕ) : b ≤ num_buses → b > 0

theorem allocation_schemes :
  allocate_teachers num_teachers num_buses = 150 :=
sorry

end NUMINAMATH_CALUDE_allocation_schemes_l1726_172693


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1726_172636

theorem intersection_point_of_lines (x y : ℚ) :
  (5 * x - 3 * y = 20) ∧ (3 * x + 4 * y = 6) ↔ x = 98/29 ∧ y = 87/58 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1726_172636


namespace NUMINAMATH_CALUDE_tanner_savings_l1726_172688

def savings_september : ℕ := 17
def savings_october : ℕ := 48
def savings_november : ℕ := 25
def video_game_cost : ℕ := 49

theorem tanner_savings : 
  savings_september + savings_october + savings_november - video_game_cost = 41 := by
  sorry

end NUMINAMATH_CALUDE_tanner_savings_l1726_172688


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1726_172677

open Set

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem complement_A_intersect_B :
  (𝕌 \ A) ∩ B = Ioc 2 3 :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1726_172677


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l1726_172622

theorem inequality_and_minimum_value :
  (∃ m n : ℝ, (∀ x : ℝ, |x + 1| + |2*x - 1| ≤ 3 ↔ m ≤ x ∧ x ≤ n) ∧
   m = -1 ∧ n = 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 →
   (1/a + 1/b + 1/c ≥ 9/2 ∧ 
    ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 2 ∧ 1/a₀ + 1/b₀ + 1/c₀ = 9/2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l1726_172622


namespace NUMINAMATH_CALUDE_freddy_call_cost_l1726_172676

/-- Calculates the total cost of phone calls in dollars -/
def total_call_cost (local_duration : ℕ) (international_duration : ℕ) 
                    (local_rate : ℚ) (international_rate : ℚ) : ℚ :=
  (local_duration : ℚ) * local_rate + (international_duration : ℚ) * international_rate

/-- Proves that Freddy's total call cost is $10.00 -/
theorem freddy_call_cost : 
  total_call_cost 45 31 (5 / 100) (25 / 100) = 10 := by
  sorry

#eval total_call_cost 45 31 (5 / 100) (25 / 100)

end NUMINAMATH_CALUDE_freddy_call_cost_l1726_172676


namespace NUMINAMATH_CALUDE_decryption_theorem_l1726_172678

/-- Represents an encrypted text --/
def EncryptedText := String

/-- Represents a decrypted message --/
def DecryptedMessage := String

/-- The encryption method used for the word "МОСКВА" --/
def moscowEncryption (s : String) : EncryptedText :=
  sorry

/-- The decryption method for the given encryption --/
def decrypt (s : EncryptedText) : DecryptedMessage :=
  sorry

/-- Checks if two encrypted texts correspond to the same message --/
def sameMessage (t1 t2 : EncryptedText) : Prop :=
  decrypt t1 = decrypt t2

theorem decryption_theorem 
  (text1 text2 text3 : EncryptedText)
  (h1 : moscowEncryption "МОСКВА" = "ЙМЫВОТСЬЛКЪГВЦАЯЯ")
  (h2 : moscowEncryption "МОСКВА" = "УКМАПОЧСРКЩВЗАХ")
  (h3 : moscowEncryption "МОСКВА" = "ШМФЭОГЧСЙЪКФЬВЫЕАКК")
  (h4 : text1 = "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ")
  (h5 : text2 = "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП")
  (h6 : text3 = "РТПАИОМВСВТИЕОБПРОЕННИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК")
  (h7 : sameMessage text1 text3 ∨ sameMessage text1 text2 ∨ sameMessage text2 text3)
  : decrypt text1 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧ 
    decrypt text3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧
    decrypt text2 = "СМОТРИВКОРЕНЬ" :=
  sorry

end NUMINAMATH_CALUDE_decryption_theorem_l1726_172678


namespace NUMINAMATH_CALUDE_beach_towel_laundry_loads_l1726_172611

theorem beach_towel_laundry_loads 
  (num_families : ℕ) 
  (people_per_family : ℕ) 
  (vacation_days : ℕ) 
  (towels_per_person_per_day : ℕ) 
  (towels_per_load : ℕ) 
  (h1 : num_families = 3) 
  (h2 : people_per_family = 4) 
  (h3 : vacation_days = 7) 
  (h4 : towels_per_person_per_day = 1) 
  (h5 : towels_per_load = 14) : 
  (num_families * people_per_family * vacation_days * towels_per_person_per_day + towels_per_load - 1) / towels_per_load = 6 := by
  sorry

end NUMINAMATH_CALUDE_beach_towel_laundry_loads_l1726_172611


namespace NUMINAMATH_CALUDE_vote_percentages_sum_to_100_l1726_172639

theorem vote_percentages_sum_to_100 (candidate1_percent candidate2_percent candidate3_percent : ℝ) 
  (h1 : candidate1_percent = 25)
  (h2 : candidate2_percent = 45)
  (h3 : candidate3_percent = 30) :
  candidate1_percent + candidate2_percent + candidate3_percent = 100 := by
  sorry

end NUMINAMATH_CALUDE_vote_percentages_sum_to_100_l1726_172639


namespace NUMINAMATH_CALUDE_windfall_percentage_increase_l1726_172691

theorem windfall_percentage_increase 
  (initial_balance : ℝ)
  (weekly_investment : ℝ)
  (weeks_in_year : ℕ)
  (final_balance : ℝ)
  (h1 : initial_balance = 250000)
  (h2 : weekly_investment = 2000)
  (h3 : weeks_in_year = 52)
  (h4 : final_balance = 885000) :
  let balance_before_windfall := initial_balance + weekly_investment * weeks_in_year
  let windfall := final_balance - balance_before_windfall
  (windfall / balance_before_windfall) * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_windfall_percentage_increase_l1726_172691


namespace NUMINAMATH_CALUDE_square_or_double_square_l1726_172665

theorem square_or_double_square (p m n : ℕ) : 
  Prime p → 
  m ≠ n → 
  p^2 = (m^2 + n^2) / 2 → 
  ∃ k : ℤ, (2*p - m - n : ℤ) = k^2 ∨ (2*p - m - n : ℤ) = 2*k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_or_double_square_l1726_172665


namespace NUMINAMATH_CALUDE_cobalt_percentage_is_15_percent_l1726_172668

/-- Represents the composition of a mixture -/
structure Mixture where
  cobalt : ℝ
  lead : ℝ
  copper : ℝ

/-- The given mixture satisfies the problem conditions -/
def problem_mixture : Mixture where
  lead := 0.25
  copper := 0.60
  cobalt := 1 - (0.25 + 0.60)

/-- The total weight of the mixture in kg -/
def total_weight : ℝ := 5 + 12

theorem cobalt_percentage_is_15_percent (m : Mixture) 
  (h1 : m.lead = 0.25)
  (h2 : m.copper = 0.60)
  (h3 : m.lead + m.copper + m.cobalt = 1)
  (h4 : m.lead * total_weight = 5)
  (h5 : m.copper * total_weight = 12) :
  m.cobalt = 0.15 := by
  sorry

#check cobalt_percentage_is_15_percent

end NUMINAMATH_CALUDE_cobalt_percentage_is_15_percent_l1726_172668


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1726_172644

/-- Given plane vectors a and b, prove that the angle between a and a+b is π/3 -/
theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (1, 0) →
  b = (-1/2, Real.sqrt 3/2) →
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  Real.arccos ((a.1 * a_plus_b.1 + a.2 * a_plus_b.2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (a_plus_b.1^2 + a_plus_b.2^2))) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1726_172644


namespace NUMINAMATH_CALUDE_fraction_order_l1726_172631

theorem fraction_order (a b m n : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : m > 0) (h4 : n > 0) :
  b / a < (b + m) / (a + m) ∧ 
  (b + m) / (a + m) < (a + n) / (b + n) ∧ 
  (a + n) / (b + n) < a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l1726_172631


namespace NUMINAMATH_CALUDE_units_digit_of_7_451_l1726_172663

theorem units_digit_of_7_451 : (7^451) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_451_l1726_172663


namespace NUMINAMATH_CALUDE_wilsons_theorem_l1726_172651

theorem wilsons_theorem (p : Nat) (hp : Nat.Prime p) : (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l1726_172651


namespace NUMINAMATH_CALUDE_megan_math_problems_l1726_172634

/-- Proves that Megan had 36 math problems given the conditions of the problem -/
theorem megan_math_problems :
  ∀ (total_problems math_problems spelling_problems : ℕ)
    (problems_per_hour hours_taken : ℕ),
  spelling_problems = 28 →
  problems_per_hour = 8 →
  hours_taken = 8 →
  total_problems = math_problems + spelling_problems →
  total_problems = problems_per_hour * hours_taken →
  math_problems = 36 := by
  sorry

end NUMINAMATH_CALUDE_megan_math_problems_l1726_172634


namespace NUMINAMATH_CALUDE_roots_of_unity_cubic_equation_l1726_172641

theorem roots_of_unity_cubic_equation :
  ∃ (c d : ℤ), ∃ (roots : Finset ℂ),
    (∀ z ∈ roots, z^3 = 1) ∧
    (∀ z ∈ roots, z^3 + c*z + d = 0) ∧
    (roots.card = 3) ∧
    (∀ z : ℂ, z^3 = 1 → z^3 + c*z + d = 0 → z ∈ roots) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_unity_cubic_equation_l1726_172641


namespace NUMINAMATH_CALUDE_weight_difference_l1726_172648

/-- Given the weights of four individuals with specific relationships, prove the weight difference between two of them. -/
theorem weight_difference (total_weight : ℝ) (jack_weight : ℝ) (avg_weight : ℝ) : 
  total_weight = 240 ∧ 
  jack_weight = 52 ∧ 
  avg_weight = 60 →
  ∃ (sam_weight lisa_weight daisy_weight : ℝ),
    sam_weight = jack_weight / 0.8 ∧
    lisa_weight = jack_weight * 1.4 ∧
    daisy_weight = (jack_weight + lisa_weight) / 3 ∧
    total_weight = jack_weight + sam_weight + lisa_weight + daisy_weight ∧
    sam_weight - daisy_weight = 23.4 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l1726_172648


namespace NUMINAMATH_CALUDE_line_through_point_l1726_172618

/-- Given a line equation 3bx + (2b-1)y = 5b - 3 that passes through the point (3, -7),
    prove that b = 1. -/
theorem line_through_point (b : ℝ) : 
  (3 * b * 3 + (2 * b - 1) * (-7) = 5 * b - 3) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1726_172618


namespace NUMINAMATH_CALUDE_power_function_through_point_l1726_172673

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x : ℝ, f x = x ^ α) →  -- f is a power function
  f 2 = Real.sqrt 2 →       -- f passes through (2, √2)
  ∀ x : ℝ, f x = x ^ (1/2)  -- f(x) = x^(1/2)
:= by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1726_172673
