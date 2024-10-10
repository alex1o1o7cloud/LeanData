import Mathlib

namespace charles_paint_area_l2509_250948

/-- 
Given a wall that requires 320 square feet to be painted and a work ratio of 2:6 between Allen and Charles,
prove that Charles paints 240 square feet.
-/
theorem charles_paint_area (total_area : ℝ) (allen_ratio charles_ratio : ℕ) : 
  total_area = 320 →
  allen_ratio = 2 →
  charles_ratio = 6 →
  (charles_ratio / (allen_ratio + charles_ratio)) * total_area = 240 := by
  sorry

end charles_paint_area_l2509_250948


namespace constant_term_expansion_l2509_250998

theorem constant_term_expansion (x : ℝ) : 
  (∃ c : ℝ, c = -160 ∧ 
   ∃ f : ℝ → ℝ, f x = (2*x - 1/x)^6 ∧ 
   ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) :=
sorry

end constant_term_expansion_l2509_250998


namespace quadratic_minimum_value_l2509_250972

/-- A quadratic function f(x) = x^2 - 2x + m with a minimum value of 1 on [3, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m

/-- The domain of the function -/
def domain : Set ℝ := {x : ℝ | x ≥ 3}

theorem quadratic_minimum_value (m : ℝ) :
  (∀ x ∈ domain, f m x ≥ 1) ∧ (∃ x ∈ domain, f m x = 1) → m = -2 :=
sorry

end quadratic_minimum_value_l2509_250972


namespace equal_roots_condition_l2509_250978

/-- A quadratic equation of the form x(x+1) + ax = 0 has two equal real roots if and only if a = -1 -/
theorem equal_roots_condition (a : ℝ) : 
  (∃ x : ℝ, x * (x + 1) + a * x = 0 ∧ 
   ∀ y : ℝ, y * (y + 1) + a * y = 0 → y = x) ↔ 
  a = -1 :=
sorry

end equal_roots_condition_l2509_250978


namespace fixed_points_are_corresponding_l2509_250936

/-- A type representing a geometric figure -/
structure Figure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A type representing a point in a geometric figure -/
structure Point where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Predicate to check if three figures are similar -/
def are_similar (f1 f2 f3 : Figure) : Prop :=
  sorry

/-- Predicate to check if a point is fixed (invariant) in a figure -/
def is_fixed_point (p : Point) (f : Figure) : Prop :=
  sorry

/-- Predicate to check if two points are corresponding in two figures -/
def are_corresponding_points (p1 p2 : Point) (f1 f2 : Figure) : Prop :=
  sorry

/-- Theorem stating that fixed points of three similar figures are corresponding points -/
theorem fixed_points_are_corresponding
  (f1 f2 f3 : Figure)
  (h_similar : are_similar f1 f2 f3)
  (p1 : Point)
  (h_fixed1 : is_fixed_point p1 f1)
  (p2 : Point)
  (h_fixed2 : is_fixed_point p2 f2)
  (p3 : Point)
  (h_fixed3 : is_fixed_point p3 f3) :
  are_corresponding_points p1 p2 f1 f2 ∧
  are_corresponding_points p2 p3 f2 f3 ∧
  are_corresponding_points p1 p3 f1 f3 :=
by
  sorry

end fixed_points_are_corresponding_l2509_250936


namespace negation_of_universal_proposition_l2509_250938

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - 3*x > 0) ↔ (∃ x : ℝ, x^3 - 3*x ≤ 0) := by sorry

end negation_of_universal_proposition_l2509_250938


namespace visitors_scientific_notation_l2509_250911

-- Define 1.12 million
def visitors : ℝ := 1.12 * 1000000

-- Define scientific notation
def scientific_notation (x : ℝ) (base : ℝ) (exponent : ℤ) : Prop :=
  x = base * (10 : ℝ) ^ exponent ∧ 1 ≤ base ∧ base < 10

-- Theorem statement
theorem visitors_scientific_notation :
  scientific_notation visitors 1.12 6 := by
  sorry

end visitors_scientific_notation_l2509_250911


namespace problem_statement_l2509_250989

open Real

theorem problem_statement :
  ∃ a : ℝ,
    (∀ x : ℝ, x > 0 → exp x - log x ≥ exp a - log a) ∧
    exp a * log a = -1 ∧
    ∀ x₁ x₂ : ℝ,
      1 < x₁ → x₁ < x₂ →
        (∃ x₀ : ℝ, x₁ < x₀ ∧ x₀ < x₂ ∧
          ((exp x₁ - exp x₂) / (x₁ - x₂)) / ((log x₁ - log x₂) / (x₁ - x₂)) = x₀ * exp x₀) ∧
        (exp x₁ - exp x₂) / (x₁ - x₂) - (log x₁ - log x₂) / (x₁ - x₂) <
          (exp x₁ + exp x₂) / 2 - 1 / sqrt (x₁ * x₂) := by
  sorry

end problem_statement_l2509_250989


namespace johns_remaining_money_l2509_250994

/-- Calculates the remaining money for John after transactions --/
def remaining_money (initial_amount : ℚ) (sister_fraction : ℚ) (groceries_cost : ℚ) (gift_cost : ℚ) : ℚ :=
  initial_amount - (sister_fraction * initial_amount) - groceries_cost - gift_cost

/-- Theorem stating that John's remaining money is $11.67 --/
theorem johns_remaining_money :
  remaining_money 100 (1/3) 40 15 = 35/3 :=
by sorry

end johns_remaining_money_l2509_250994


namespace function_value_at_zero_l2509_250990

theorem function_value_at_zero 
  (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 2) = f (x + 1) - f x) 
  (h2 : f 1 = Real.log (3/2)) 
  (h3 : f 2 = Real.log 15) : 
  f 0 = -1 := by sorry

end function_value_at_zero_l2509_250990


namespace quadratic_root_implies_coefficient_l2509_250969

theorem quadratic_root_implies_coefficient (b : ℝ) : 
  (2^2 + b*2 - 10 = 0) → b = 3 := by
  sorry

end quadratic_root_implies_coefficient_l2509_250969


namespace evaluate_expression_l2509_250960

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 9
  (x - a + 5) = 14 := by sorry

end evaluate_expression_l2509_250960


namespace area_of_three_arc_region_sum_of_coefficients_l2509_250992

/-- The area of a region bounded by three circular arcs -/
theorem area_of_three_arc_region :
  let r : ℝ := 5  -- radius of each circle
  let θ : ℝ := π / 2  -- central angle of each arc (90 degrees in radians)
  let sector_area : ℝ := (θ / (2 * π)) * π * r^2  -- area of one sector
  let triangle_side : ℝ := r * Real.sqrt 2  -- side length of the equilateral triangle
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2  -- area of the equilateral triangle
  let region_area : ℝ := 3 * sector_area - triangle_area  -- area of the bounded region
  region_area = -125 * Real.sqrt 3 / 4 + 75 * π / 4 := by
    sorry

/-- The sum of coefficients in the area expression -/
theorem sum_of_coefficients :
  let a : ℝ := -125 / 4
  let b : ℝ := 3
  let c : ℝ := 75 / 4
  ⌊a + b + c⌋ = -9 := by
    sorry

end area_of_three_arc_region_sum_of_coefficients_l2509_250992


namespace wayne_blocks_total_l2509_250976

theorem wayne_blocks_total (initial_blocks additional_blocks : ℕ) 
  (h1 : initial_blocks = 9)
  (h2 : additional_blocks = 6) :
  initial_blocks + additional_blocks = 15 := by
  sorry

end wayne_blocks_total_l2509_250976


namespace equilateral_triangle_locus_l2509_250973

-- Define an equilateral triangle ABC
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Define the reflection of a point over a line
def ReflectPointOverLine (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the set of points P satisfying PA^2 = PB^2 + PC^2
def SatisfyingPoints (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | dist P A ^ 2 = dist P B ^ 2 + dist P C ^ 2}

-- Theorem statement
theorem equilateral_triangle_locus 
  (A B C : ℝ × ℝ) 
  (h : EquilateralTriangle A B C) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = ReflectPointOverLine A B C ∧ 
    radius = dist A B ∧
    SatisfyingPoints A B C = {P : ℝ × ℝ | dist P center = radius} :=
  sorry

end equilateral_triangle_locus_l2509_250973


namespace unique_solution_tan_cos_equation_l2509_250965

theorem unique_solution_tan_cos_equation : 
  ∃! (n : ℕ), n > 0 ∧ Real.tan (π / (2 * n)) + Real.cos (π / (2 * n)) = n / 4 := by
  sorry

end unique_solution_tan_cos_equation_l2509_250965


namespace total_pies_eq_750_l2509_250983

/-- The number of mini meat pies made by the first team -/
def team1_pies : ℕ := 235

/-- The number of mini meat pies made by the second team -/
def team2_pies : ℕ := 275

/-- The number of mini meat pies made by the third team -/
def team3_pies : ℕ := 240

/-- The total number of teams -/
def num_teams : ℕ := 3

/-- The total number of mini meat pies made by all teams -/
def total_pies : ℕ := team1_pies + team2_pies + team3_pies

theorem total_pies_eq_750 : total_pies = 750 := by
  sorry

end total_pies_eq_750_l2509_250983


namespace z_in_fourth_quadrant_l2509_250924

theorem z_in_fourth_quadrant : 
  ∀ z : ℂ, (1 - I) / (z - 2) = 1 + I → 
  (z.re > 0 ∧ z.im < 0) :=
by sorry

end z_in_fourth_quadrant_l2509_250924


namespace angle_120_in_second_quadrant_l2509_250944

/-- An angle in the Cartesian plane -/
structure CartesianAngle where
  /-- The measure of the angle in degrees -/
  measure : ℝ
  /-- The angle's vertex is at the origin -/
  vertex_at_origin : Bool
  /-- The angle's initial side is along the positive x-axis -/
  initial_side_positive_x : Bool

/-- Definition of the second quadrant -/
def is_in_second_quadrant (angle : CartesianAngle) : Prop :=
  angle.measure > 90 ∧ angle.measure < 180

/-- Theorem: An angle of 120° with vertex at origin and initial side along positive x-axis is in the second quadrant -/
theorem angle_120_in_second_quadrant :
  ∀ (angle : CartesianAngle),
    angle.measure = 120 ∧
    angle.vertex_at_origin = true ∧
    angle.initial_side_positive_x = true →
    is_in_second_quadrant angle :=
by sorry

end angle_120_in_second_quadrant_l2509_250944


namespace more_girls_than_boys_l2509_250959

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boy_ratio = 3)
  (h3 : girl_ratio = 4) :
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    boy_ratio * girls = girl_ratio * boys ∧
    girls - boys = 6 := by
  sorry

end more_girls_than_boys_l2509_250959


namespace quadratic_vertex_form_l2509_250945

theorem quadratic_vertex_form (x : ℝ) : ∃ (a h k : ℝ), 
  x^2 - 6*x + 1 = a*(x - h)^2 + k ∧ k = -8 := by sorry

end quadratic_vertex_form_l2509_250945


namespace hyperbola_line_slope_l2509_250993

/-- Given two points on a hyperbola with a specific midpoint, prove that the slope of the line connecting them is 9/4 -/
theorem hyperbola_line_slope (A B : ℝ × ℝ) : 
  (A.1^2 - A.2^2/9 = 1) →  -- A is on the hyperbola
  (B.1^2 - B.2^2/9 = 1) →  -- B is on the hyperbola
  ((A.1 + B.1)/2 = -1) →   -- x-coordinate of midpoint
  ((A.2 + B.2)/2 = -4) →   -- y-coordinate of midpoint
  (B.2 - A.2)/(B.1 - A.1) = 9/4 :=  -- slope of line AB
by sorry

end hyperbola_line_slope_l2509_250993


namespace team_selection_ways_l2509_250933

def num_boys : ℕ := 10
def num_girls : ℕ := 12
def team_size : ℕ := 8
def boys_in_team : ℕ := 4
def girls_in_team : ℕ := 4

theorem team_selection_ways :
  (Nat.choose num_boys boys_in_team) * (Nat.choose num_girls girls_in_team) = 103950 :=
by sorry

end team_selection_ways_l2509_250933


namespace fraction_equality_l2509_250974

theorem fraction_equality (a b c : ℝ) :
  (|a^2 + b^2|^3 + |b^2 + c^2|^3 + |c^2 + a^2|^3) / (|a + b|^3 + |b + c|^3 + |c + a|^3) = 1 :=
by sorry

end fraction_equality_l2509_250974


namespace tan_alpha_values_l2509_250900

theorem tan_alpha_values (α : ℝ) (h : Real.sin (2 * α) = -Real.sin α) : 
  Real.tan α = 0 ∨ Real.tan α = Real.sqrt 3 ∨ Real.tan α = -Real.sqrt 3 :=
by sorry

end tan_alpha_values_l2509_250900


namespace rhombus_area_l2509_250923

/-- The area of a rhombus given its perimeter and one diagonal -/
theorem rhombus_area (perimeter : ℝ) (diagonal : ℝ) : 
  perimeter > 0 → diagonal > 0 → diagonal < perimeter → 
  (perimeter * diagonal) / 8 = 96 → 
  (perimeter / 4) * (((perimeter / 4)^2 - (diagonal / 2)^2).sqrt) = 96 := by
  sorry

end rhombus_area_l2509_250923


namespace score_change_effect_l2509_250982

/-- Proves that changing one student's score from 86 to 74 in a group of 8 students
    with an initial average of 82.5 decreases the average by 1.5 points -/
theorem score_change_effect (n : ℕ) (initial_avg : ℚ) (old_score new_score : ℚ) :
  n = 8 →
  initial_avg = 82.5 →
  old_score = 86 →
  new_score = 74 →
  initial_avg - (n * initial_avg - old_score + new_score) / n = 1.5 := by
  sorry

end score_change_effect_l2509_250982


namespace solution_set_implies_b_power_a_l2509_250932

theorem solution_set_implies_b_power_a (a b : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 3) ↔ x^2 < a*x + b) → 
  b^a = 81 := by
sorry

end solution_set_implies_b_power_a_l2509_250932


namespace locus_of_N_l2509_250931

/-- The locus of point N in an equilateral triangle with a moving point on the unit circle -/
theorem locus_of_N (M N : ℂ) (t : ℝ) : 
  (∀ t, M = Complex.exp (Complex.I * t)) →  -- M is on the unit circle
  (N - 3 = Complex.exp (Complex.I * (5 * Real.pi / 3)) * (M - 3)) →  -- N forms equilateral triangle with A(3,0) and M
  (Complex.abs (N - (3/2 + Complex.I * (3 * Real.sqrt 3 / 2))) = 1) :=  -- Locus of N is a circle
by sorry

end locus_of_N_l2509_250931


namespace total_stones_l2509_250956

/-- The number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- The conditions for the stone distribution -/
def validDistribution (p : StonePiles) : Prop :=
  p.pile5 = 6 * p.pile3 ∧
  p.pile2 = 2 * (p.pile3 + p.pile5) ∧
  p.pile1 = p.pile5 / 3 ∧
  p.pile1 = p.pile4 - 10 ∧
  p.pile4 = p.pile2 / 2

/-- The theorem stating that the total number of stones is 60 -/
theorem total_stones (p : StonePiles) (h : validDistribution p) : 
  p.pile1 + p.pile2 + p.pile3 + p.pile4 + p.pile5 = 60 := by
  sorry

end total_stones_l2509_250956


namespace textbook_weight_difference_l2509_250951

theorem textbook_weight_difference :
  let chemistry_weight : ℝ := 7.125
  let geometry_weight : ℝ := 0.625
  chemistry_weight - geometry_weight = 6.5 := by
  sorry

end textbook_weight_difference_l2509_250951


namespace no_point_M_exists_line_EF_exists_l2509_250942

-- Define the ellipse C
def C (x y : ℝ) : Prop := (x - 2)^2 + y^2/4 = 1

-- Define the line l
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define point R
def R : ℝ × ℝ := (1, 4)

-- Theorem 1: No point M exists inside C satisfying the given condition
theorem no_point_M_exists : ¬ ∃ M : ℝ × ℝ, 
  C M.1 M.2 ∧ 
  (∀ Q A B : ℝ × ℝ, 
    l Q.1 Q.2 → 
    C A.1 A.2 → 
    C B.1 B.2 → 
    (∃ t : ℝ, M.1 = t * (Q.1 - M.1) + M.1 ∧ M.2 = t * (Q.2 - M.2) + M.2) →
    (A.1 - M.1)^2 + (A.2 - M.2)^2 = (B.1 - M.1)^2 + (B.2 - M.2)^2 →
    (A.1 - M.1)^2 + (A.2 - M.2)^2 = (A.1 - Q.1)^2 + (A.2 - Q.2)^2) :=
sorry

-- Theorem 2: Line EF exists and has the given equations
theorem line_EF_exists : ∃ E F : ℝ × ℝ,
  C E.1 E.2 ∧ 
  C F.1 F.2 ∧
  (R.1 - E.1)^2 + (R.2 - E.2)^2 = (E.1 - F.1)^2 + (E.2 - F.2)^2 ∧
  (2 * E.1 + E.2 - 6 = 0 ∨ 14 * E.1 + E.2 - 18 = 0) ∧
  (2 * F.1 + F.2 - 6 = 0 ∨ 14 * F.1 + F.2 - 18 = 0) :=
sorry

end no_point_M_exists_line_EF_exists_l2509_250942


namespace crab_price_proof_l2509_250920

/-- Proves that the price per crab is $3 given the conditions of John's crab selling business -/
theorem crab_price_proof (baskets_per_week : ℕ) (crabs_per_basket : ℕ) (collection_frequency : ℕ) (total_revenue : ℕ) :
  baskets_per_week = 3 →
  crabs_per_basket = 4 →
  collection_frequency = 2 →
  total_revenue = 72 →
  (total_revenue : ℚ) / (baskets_per_week * crabs_per_basket * collection_frequency) = 3 := by
  sorry

#check crab_price_proof

end crab_price_proof_l2509_250920


namespace rectangle_area_difference_l2509_250957

/-- Given a real number x, this theorem states that the area of a rectangle with 
    dimensions (x+8) and (x+6), minus the area of a rectangle with dimensions (2x-1) 
    and (x-1), plus the area of a rectangle with dimensions (x-3) and (x-5), 
    equals 25x + 62. -/
theorem rectangle_area_difference (x : ℝ) : 
  (x + 8) * (x + 6) - (2*x - 1) * (x - 1) + (x - 3) * (x - 5) = 25*x + 62 := by
  sorry

end rectangle_area_difference_l2509_250957


namespace division_multiplication_result_l2509_250950

theorem division_multiplication_result : (2 : ℚ) / 3 * (-1/3) = -2/9 := by
  sorry

end division_multiplication_result_l2509_250950


namespace parabola_intersection_locus_locus_nature_l2509_250966

/-- Given a parabola and a point in its plane, this theorem describes the locus of 
    intersection points formed by certain lines related to the parabola. -/
theorem parabola_intersection_locus 
  (p : ℝ) -- Parameter of the parabola
  (α β : ℝ) -- Coordinates of point A
  (x y : ℝ) -- Coordinates of the locus point M
  (h_parabola : y^2 = 2*p*x) -- Equation of the parabola
  : 2*p*x^2 - β*x*y + α*y^2 - 2*p*α*x = 0 := by
  sorry

/-- This theorem characterizes the nature of the locus based on the position of point A 
    relative to the parabola. -/
theorem locus_nature 
  (p : ℝ) -- Parameter of the parabola
  (α β : ℝ) -- Coordinates of point A
  : (β^2 = 8*p*α → IsParabola) ∧ 
    (β^2 < 8*p*α → IsEllipse) ∧ 
    (β^2 > 8*p*α → IsHyperbola) := by
  sorry

-- We need to define these predicates
axiom IsParabola : Prop
axiom IsEllipse : Prop
axiom IsHyperbola : Prop

end parabola_intersection_locus_locus_nature_l2509_250966


namespace system_solution_l2509_250934

theorem system_solution (x y k : ℚ) 
  (eq1 : 3 * x + 2 * y = k + 1)
  (eq2 : 2 * x + 3 * y = k)
  (sum_condition : x + y = 2) :
  k = 9 / 2 := by
sorry

end system_solution_l2509_250934


namespace max_notebooks_purchasable_l2509_250961

def available_funds : ℚ := 21.45
def notebook_cost : ℚ := 2.75

theorem max_notebooks_purchasable :
  ∀ n : ℕ, (n : ℚ) * notebook_cost ≤ available_funds ↔ n ≤ 7 :=
by sorry

end max_notebooks_purchasable_l2509_250961


namespace horatio_sonnets_l2509_250970

/-- Represents the number of lines in a sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of sonnets the lady heard before telling Horatio to leave -/
def sonnets_heard : ℕ := 7

/-- Represents the number of romantic lines Horatio wrote that were never heard -/
def unheard_lines : ℕ := 70

/-- Calculates the total number of sonnets Horatio wrote -/
def total_sonnets : ℕ := sonnets_heard + (unheard_lines / lines_per_sonnet)

theorem horatio_sonnets : total_sonnets = 12 := by sorry

end horatio_sonnets_l2509_250970


namespace probability_sum_ten_l2509_250947

/-- Represents an octahedral die with 8 faces -/
def OctahedralDie := Fin 8

/-- The set of possible outcomes when rolling two octahedral dice -/
def DiceOutcomes := OctahedralDie × OctahedralDie

/-- The total number of possible outcomes when rolling two octahedral dice -/
def totalOutcomes : ℕ := 64

/-- Predicate to check if a pair of dice rolls sums to 10 -/
def sumsToTen (roll : DiceOutcomes) : Prop :=
  (roll.1.val + 1) + (roll.2.val + 1) = 10

/-- The number of favorable outcomes (sum of 10) -/
def favorableOutcomes : ℕ := 5

/-- Theorem stating the probability of rolling a sum of 10 -/
theorem probability_sum_ten :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 64 := by
  sorry

end probability_sum_ten_l2509_250947


namespace chicken_pizza_menu_combinations_l2509_250997

theorem chicken_pizza_menu_combinations : 
  let chicken_types : ℕ := 4
  let pizza_types : ℕ := 3
  let same_chicken_diff_pizza := chicken_types * (pizza_types * (pizza_types - 1))
  let same_pizza_diff_chicken := pizza_types * (chicken_types * (chicken_types - 1))
  same_chicken_diff_pizza + same_pizza_diff_chicken = 60 :=
by sorry

end chicken_pizza_menu_combinations_l2509_250997


namespace alyssa_picked_32_limes_l2509_250946

/-- The number of limes Alyssa picked -/
def alyssas_limes (total_limes fred_limes nancy_limes : ℕ) : ℕ :=
  total_limes - (fred_limes + nancy_limes)

/-- Proof that Alyssa picked 32 limes -/
theorem alyssa_picked_32_limes :
  alyssas_limes 103 36 35 = 32 := by
  sorry

end alyssa_picked_32_limes_l2509_250946


namespace mixed_fruit_juice_cost_l2509_250922

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost_per_litre : ℝ := 1399.45

/-- The cost per litre of açaí berry juice -/
def acai_cost_per_litre : ℝ := 3104.35

/-- The volume of mixed fruit juice used -/
def mixed_fruit_volume : ℝ := 32

/-- The volume of açaí berry juice used -/
def acai_volume : ℝ := 21.333333333333332

/-- The cost per litre of mixed fruit juice -/
def mixed_fruit_cost_per_litre : ℝ := 262.8125

theorem mixed_fruit_juice_cost : 
  cocktail_cost_per_litre * (mixed_fruit_volume + acai_volume) = 
  mixed_fruit_cost_per_litre * mixed_fruit_volume + acai_cost_per_litre * acai_volume := by
  sorry

end mixed_fruit_juice_cost_l2509_250922


namespace g_of_2_l2509_250977

/-- Given functions f and g, prove the value of g(2) -/
theorem g_of_2 (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 2 * x^2 + 4 * x - 6)
  (hg : ∀ x, g (f x) = 3 * x^3 + 2 * x - 5) :
  g 2 = 3 * (-1 + Real.sqrt 5)^3 + 2 * (-1 + Real.sqrt 5) - 5 := by
sorry

end g_of_2_l2509_250977


namespace leg_head_difference_l2509_250919

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (g : AnimalGroup) : ℕ := 2 * g.ducks + 4 * g.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (g : AnimalGroup) : ℕ := g.ducks + g.cows

/-- The main theorem -/
theorem leg_head_difference (g : AnimalGroup) 
  (h1 : g.cows = 20)
  (h2 : ∃ k : ℕ, totalLegs g = 2 * totalHeads g + k) :
  ∃ k : ℕ, k = 40 ∧ totalLegs g = 2 * totalHeads g + k := by
  sorry


end leg_head_difference_l2509_250919


namespace circle_and_tangent_lines_l2509_250991

-- Define the circle C
def circle_C (x y : ℝ) := x^2 + (y + 1)^2 = 2

-- Define the parabola
def parabola (x y : ℝ) := y^2 = 4 * x

-- Define the line y = x
def line_y_eq_x (x y : ℝ) := y = x

-- Define the point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define the two tangent lines
def tangent_line_1 (x y : ℝ) := x + y - 1 = 0
def tangent_line_2 (x y : ℝ) := 7 * x - y + 9 = 0

theorem circle_and_tangent_lines :
  -- Circle C is symmetric about the y-axis
  (∀ x y : ℝ, circle_C x y ↔ circle_C (-x) y) →
  -- Circle C passes through the focus of the parabola y^2 = 4x
  (circle_C 1 0) →
  -- Circle C is divided into two arc lengths with a ratio of 1:2 by the line y = x
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, circle_C x y → line_y_eq_x x y → 
    (x^2 + y^2)^(1/2) = r ∧ ((x - 0)^2 + (y - (-1))^2)^(1/2) = 2 * r) →
  -- The center of circle C is below the x-axis
  (∃ a : ℝ, a < 0 ∧ ∀ x y : ℝ, circle_C x y ↔ x^2 + (y - a)^2 = 2) →
  -- The equation of circle C is x^2 + (y + 1)^2 = 2
  (∀ x y : ℝ, circle_C x y ↔ x^2 + (y + 1)^2 = 2) ∧
  -- The equations of the tangent lines passing through P(-1, 2) are x + y - 1 = 0 and 7x - y + 9 = 0
  (∀ x y : ℝ, (tangent_line_1 x y ∨ tangent_line_2 x y) ↔
    (∃ t : ℝ, circle_C (point_P.1 + t * (x - point_P.1)) (point_P.2 + t * (y - point_P.2)) ∧
      (∀ s : ℝ, s ≠ t → ¬ circle_C (point_P.1 + s * (x - point_P.1)) (point_P.2 + s * (y - point_P.2))))) :=
by sorry

end circle_and_tangent_lines_l2509_250991


namespace rectangle_perpendicular_point_theorem_l2509_250995

/-- Given a rectangle ABCD with point E on diagonal BD such that AE is perpendicular to BD -/
structure RectangleWithPerpendicularPoint where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Distance from E to DC -/
  n : ℝ
  /-- Distance from E to BC -/
  EC : ℝ
  /-- Distance from E to AB -/
  x : ℝ
  /-- Length of diagonal BD -/
  d : ℝ
  /-- EC is 1 -/
  h_EC : EC = 1
  /-- ABCD is a rectangle -/
  h_rectangle : AB > 0 ∧ BC > 0
  /-- E is on diagonal BD -/
  h_E_on_BD : d > 0
  /-- AE is perpendicular to BD -/
  h_AE_perp_BD : True

/-- The main theorem about the rectangle with perpendicular point -/
theorem rectangle_perpendicular_point_theorem (r : RectangleWithPerpendicularPoint) :
  /- Part a -/
  (r.d - r.x * Real.sqrt (1 + r.x^2))^2 = r.x^4 * (1 + r.x^2) ∧
  /- Part b -/
  r.n = r.x^3 ∧
  /- Part c -/
  r.d^(2/3) - r.x^(2/3) = 1 := by
  sorry

end rectangle_perpendicular_point_theorem_l2509_250995


namespace swim_meet_cars_l2509_250943

theorem swim_meet_cars (num_vans : ℕ) (people_per_car : ℕ) (people_per_van : ℕ) 
  (max_per_car : ℕ) (max_per_van : ℕ) (extra_capacity : ℕ) :
  num_vans = 3 →
  people_per_car = 5 →
  people_per_van = 3 →
  max_per_car = 6 →
  max_per_van = 8 →
  extra_capacity = 17 →
  ∃ (num_cars : ℕ), 
    num_cars * people_per_car + num_vans * people_per_van + extra_capacity = 
    num_cars * max_per_car + num_vans * max_per_van ∧
    num_cars = 2 :=
by sorry

end swim_meet_cars_l2509_250943


namespace arithmetic_sequence_property_l2509_250941

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_3 + a_11 = 22, prove that a_7 = 11 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h_sum : a 3 + a 11 = 22) : a 7 = 11 := by
  sorry

end arithmetic_sequence_property_l2509_250941


namespace empty_solution_set_iff_a_range_l2509_250908

theorem empty_solution_set_iff_a_range (a : ℝ) :
  (∀ x : ℝ, |x - 1| + |x - 3| > a^2 - 2*a - 1) ↔ (-1 < a ∧ a < 3) :=
by sorry

end empty_solution_set_iff_a_range_l2509_250908


namespace sports_day_participation_l2509_250917

/-- Given that the number of participants in a school sports day this year (m) 
    is a 10% increase from last year, prove that the number of participants 
    last year was m / (1 + 10%). -/
theorem sports_day_participation (m : ℝ) : 
  let last_year := m / (1 + 10 / 100)
  let increase_rate := 10 / 100
  m = last_year * (1 + increase_rate) → 
  last_year = m / (1 + increase_rate) := by
sorry


end sports_day_participation_l2509_250917


namespace mixture_weight_is_3_64_l2509_250953

-- Define the weights of brands in grams per liter
def weight_a : ℚ := 950
def weight_b : ℚ := 850

-- Define the ratio of volumes
def ratio_a : ℚ := 3
def ratio_b : ℚ := 2

-- Define the total volume in liters
def total_volume : ℚ := 4

-- Define the function to calculate the weight of the mixture in kg
def mixture_weight : ℚ :=
  ((ratio_a / (ratio_a + ratio_b)) * total_volume * weight_a +
   (ratio_b / (ratio_a + ratio_b)) * total_volume * weight_b) / 1000

-- Theorem statement
theorem mixture_weight_is_3_64 : mixture_weight = 3.64 := by
  sorry

end mixture_weight_is_3_64_l2509_250953


namespace rick_cheese_servings_l2509_250912

/-- Calculates the number of cheese servings eaten given the remaining calories -/
def servingsEaten (caloriesPerServing : ℕ) (servingsPerBlock : ℕ) (remainingCalories : ℕ) : ℕ :=
  (caloriesPerServing * servingsPerBlock - remainingCalories) / caloriesPerServing

theorem rick_cheese_servings :
  servingsEaten 110 16 1210 = 5 := by
  sorry

end rick_cheese_servings_l2509_250912


namespace base_conversion_sum_l2509_250910

def base_11_to_10 (n : ℕ) : ℕ := 3224

def base_5_to_10 (n : ℕ) : ℕ := 36

def base_7_to_10 (n : ℕ) : ℕ := 1362

def base_8_to_10 (n : ℕ) : ℕ := 3008

theorem base_conversion_sum :
  (base_11_to_10 2471 / base_5_to_10 121) - base_7_to_10 3654 + base_8_to_10 5680 = 1736 := by
  sorry

end base_conversion_sum_l2509_250910


namespace trigonometric_simplification_trigonometric_evaluation_l2509_250949

-- Part 1
theorem trigonometric_simplification (α : ℝ) : 
  (Real.cos (α - π/2)) / (Real.sin (5*π/2 + α)) * Real.sin (α - 2*π) * Real.cos (2*π - α) = Real.sin α ^ 2 := by
  sorry

-- Part 2
theorem trigonometric_evaluation : 
  Real.sin (25*π/6) + Real.cos (25*π/3) + Real.tan (-25*π/4) = 0 := by
  sorry

end trigonometric_simplification_trigonometric_evaluation_l2509_250949


namespace special_polynomial_max_value_l2509_250984

/-- A polynomial with real coefficients satisfying the given condition -/
def SpecialPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, P t = P 1 * t^2 + P (P 1) * t + P (P (P 1))

/-- The theorem stating the maximum value of P(P(P(P(1)))) -/
theorem special_polynomial_max_value (P : ℝ → ℝ) (h : SpecialPolynomial P) :
    ∃ M : ℝ, M = (1 : ℝ) / 9 ∧ P (P (P (P 1))) ≤ M ∧ 
    ∃ P₀ : ℝ → ℝ, SpecialPolynomial P₀ ∧ P₀ (P₀ (P₀ (P₀ 1))) = M :=
by sorry

end special_polynomial_max_value_l2509_250984


namespace square_less_than_triple_l2509_250913

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end square_less_than_triple_l2509_250913


namespace unique_real_solution_l2509_250954

theorem unique_real_solution :
  ∃! x : ℝ, (x^12 + 1) * (x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 12 * x^11 :=
by sorry

end unique_real_solution_l2509_250954


namespace third_year_sample_size_l2509_250967

/-- Calculates the number of students to be sampled from the third year in a stratified sampling -/
theorem third_year_sample_size 
  (total_students : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 900) 
  (h2 : first_year = 240) 
  (h3 : second_year = 260) 
  (h4 : sample_size = 45) :
  (sample_size * (total_students - first_year - second_year)) / total_students = 20 := by
  sorry

#check third_year_sample_size

end third_year_sample_size_l2509_250967


namespace rainy_days_count_l2509_250916

theorem rainy_days_count (n : ℕ) : 
  (∃ (R NR : ℕ), 
    R + NR = 7 ∧ 
    n * R + 4 * NR = 26 ∧ 
    4 * NR - n * R = 14) → 
  (∃ (R : ℕ), R = 2 ∧ 
    (∃ (NR : ℕ), R + NR = 7 ∧ 
      n * R + 4 * NR = 26 ∧ 
      4 * NR - n * R = 14)) :=
by sorry

end rainy_days_count_l2509_250916


namespace lou_fine_shoes_pricing_l2509_250928

/-- Calculates the price of shoes after Lou's Fine Shoes pricing strategy --/
theorem lou_fine_shoes_pricing (initial_price : ℝ) : 
  initial_price = 50 →
  (initial_price * (1 + 0.2)) * (1 - 0.2) = 48 := by
sorry

end lou_fine_shoes_pricing_l2509_250928


namespace p_sufficient_not_necessary_for_q_l2509_250902

-- Define the conditions
def p (a : ℝ) : Prop := (a - 1)^2 ≤ 1

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 ≥ 0

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end p_sufficient_not_necessary_for_q_l2509_250902


namespace S_intersections_empty_l2509_250962

def S (n : ℕ) : Set ℕ :=
  {x | ∃ g : ℕ, g ≥ 2 ∧ x = (g^n - 1) / (g - 1)}

theorem S_intersections_empty :
  (S 3 ∩ S 4 = ∅) ∧ (S 3 ∩ S 5 = ∅) := by
  sorry

end S_intersections_empty_l2509_250962


namespace division_result_l2509_250904

theorem division_result (n : ℕ) (h : n = 2011) : 
  (4 * 10^n - 1) / (4 * ((10^n - 1) / 3) + 1) = 3 := by
  sorry

end division_result_l2509_250904


namespace parallel_lines_bisect_circle_perimeter_l2509_250907

-- Define the lines and circle
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x - 2 * y + 2 = 0
def line_m (a : ℝ) (x y : ℝ) : Prop := x + (a - 3) * y + 1 = 0
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, line_l a x y ↔ line_m a x y) ↔ a = 1 :=
sorry

-- Theorem for bisecting circle's perimeter
theorem bisect_circle_perimeter (a : ℝ) :
  (∃ x y : ℝ, line_l a x y ∧ x = 1 ∧ y = 0) ↔ a = -2 :=
sorry

end parallel_lines_bisect_circle_perimeter_l2509_250907


namespace division_problem_l2509_250925

theorem division_problem (dividend : ℕ) (divisor : ℝ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 13698 →
  divisor = 153.75280898876406 →
  remainder = 14 →
  quotient = 89 →
  (dividend : ℝ) = divisor * quotient + remainder := by
  sorry

end division_problem_l2509_250925


namespace sqrt_equation_solution_l2509_250903

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (3 + 2 * Real.sqrt x) = 4 → x = 169 / 4 := by
  sorry

end sqrt_equation_solution_l2509_250903


namespace completing_square_sum_l2509_250979

theorem completing_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 9 = 0 ↔ (x + b)^2 = c) → b + c = -3 := by
  sorry

end completing_square_sum_l2509_250979


namespace m_value_proof_l2509_250999

theorem m_value_proof (m : ℤ) (h : m < (Real.sqrt 11 - 1) / 2 ∧ (Real.sqrt 11 - 1) / 2 < m + 1) : m = 1 := by
  sorry

end m_value_proof_l2509_250999


namespace geometric_series_property_l2509_250971

theorem geometric_series_property (b₁ q : ℝ) (h_q : |q| < 1) :
  (b₁ / (1 - q)) / (b₁^3 / (1 - q^3)) = 1/12 →
  (b₁^4 / (1 - q^4)) / (b₁^2 / (1 - q^2)) = 36/5 →
  (b₁ = 3 ∨ b₁ = -3) ∧ q = -1/2 := by
sorry

end geometric_series_property_l2509_250971


namespace solution_set_of_inequality_l2509_250929

theorem solution_set_of_inequality (x : ℝ) :
  {x | x^2 - 2*x + 1 ≤ 0} = {1} := by sorry

end solution_set_of_inequality_l2509_250929


namespace gray_squares_33_l2509_250937

/-- The number of squares in the n-th figure of the series -/
def total_squares (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The number of black squares in the n-th figure -/
def black_squares (n : ℕ) : ℕ := n ^ 2

/-- The number of white squares in the n-th figure -/
def white_squares (n : ℕ) : ℕ := (n - 1) ^ 2

/-- The number of gray squares in the n-th figure -/
def gray_squares (n : ℕ) : ℕ := total_squares n - black_squares n - white_squares n

theorem gray_squares_33 : gray_squares 33 = 2112 := by
  sorry

end gray_squares_33_l2509_250937


namespace closest_to_zero_l2509_250906

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem closest_to_zero (a₁ d : ℤ) (h₁ : a₁ = 81) (h₂ : d = -7) :
  ∀ n : ℕ, n ≠ 13 → |arithmetic_sequence a₁ d 13| ≤ |arithmetic_sequence a₁ d n| :=
by sorry

end closest_to_zero_l2509_250906


namespace top_square_is_five_l2509_250930

/-- Represents a square on the grid --/
structure Square :=
  (number : Nat)
  (row : Nat)
  (col : Nat)

/-- Represents the grid of squares --/
def Grid := List Square

/-- Creates the initial 5x5 grid --/
def initialGrid : Grid :=
  sorry

/-- Performs the first diagonal fold --/
def foldDiagonal (g : Grid) : Grid :=
  sorry

/-- Performs the second fold (bottom half up) --/
def foldBottomUp (g : Grid) : Grid :=
  sorry

/-- Performs the third fold (left half behind) --/
def foldLeftBehind (g : Grid) : Grid :=
  sorry

/-- Returns the top square after all folds --/
def topSquareAfterFolds (g : Grid) : Square :=
  sorry

theorem top_square_is_five :
  let finalGrid := foldLeftBehind (foldBottomUp (foldDiagonal initialGrid))
  (topSquareAfterFolds finalGrid).number = 5 := by
  sorry

end top_square_is_five_l2509_250930


namespace last_three_average_l2509_250968

theorem last_three_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 60 →
  (list.take 4).sum / 4 = 55 →
  (list.drop 4).sum / 3 = 200 / 3 := by
  sorry

end last_three_average_l2509_250968


namespace female_officers_count_female_officers_count_proof_l2509_250987

/-- The number of female officers on a police force, given:
  * 10% of female officers were on duty
  * 200 officers were on duty in total
  * Half of the officers on duty were female
-/
theorem female_officers_count : ℕ :=
  let total_on_duty : ℕ := 200
  let female_ratio_on_duty : ℚ := 1/2
  let female_on_duty_ratio : ℚ := 1/10
  1000

/-- Proof that the number of female officers is correct -/
theorem female_officers_count_proof :
  let total_on_duty : ℕ := 200
  let female_ratio_on_duty : ℚ := 1/2
  let female_on_duty_ratio : ℚ := 1/10
  female_officers_count = 1000 := by
  sorry

end female_officers_count_female_officers_count_proof_l2509_250987


namespace soda_discount_percentage_l2509_250927

-- Define the given constants
def full_price : ℚ := 30
def group_size : ℕ := 10
def num_children : ℕ := 4
def soda_price : ℚ := 5
def total_paid : ℚ := 197

-- Define the calculation functions
def adult_price := full_price
def child_price := full_price / 2

def total_price_without_discount : ℚ :=
  (group_size - num_children) * adult_price + num_children * child_price

def price_paid_for_tickets : ℚ := total_paid - soda_price

def discount_amount : ℚ := total_price_without_discount - price_paid_for_tickets

def discount_percentage : ℚ := (discount_amount / total_price_without_discount) * 100

-- State the theorem
theorem soda_discount_percentage : discount_percentage = 20 := by sorry

end soda_discount_percentage_l2509_250927


namespace amount_ratio_l2509_250996

/-- Prove that the ratio of A's amount to B's amount is 1:3 given the conditions -/
theorem amount_ratio (total amount_B amount_C : ℚ) (h1 : total = 1440)
  (h2 : amount_B = 270) (h3 : amount_B = (1/4) * amount_C) :
  ∃ amount_A : ℚ, amount_A + amount_B + amount_C = total ∧ amount_A = (1/3) * amount_B := by
  sorry

end amount_ratio_l2509_250996


namespace no_larger_subdivision_exists_max_subdivision_exists_max_triangles_is_correct_l2509_250939

/-- Represents a triangular subdivision of a triangle T -/
structure TriangularSubdivision where
  numTriangles : ℕ
  numSegmentsPerVertex : ℕ
  verticesDontSplitSides : Bool

/-- The maximum number of triangles in a valid subdivision -/
def maxTriangles : ℕ := 19

/-- Checks if a triangular subdivision is valid according to the problem conditions -/
def isValidSubdivision (s : TriangularSubdivision) : Prop :=
  s.numSegmentsPerVertex > 1 ∧ s.verticesDontSplitSides

/-- States that no valid subdivision can have more than maxTriangles triangles -/
theorem no_larger_subdivision_exists (s : TriangularSubdivision) :
  isValidSubdivision s → s.numTriangles ≤ maxTriangles :=
sorry

/-- States that there exists a valid subdivision with exactly maxTriangles triangles -/
theorem max_subdivision_exists :
  ∃ s : TriangularSubdivision, isValidSubdivision s ∧ s.numTriangles = maxTriangles :=
sorry

/-- The main theorem stating that maxTriangles is indeed the maximum -/
theorem max_triangles_is_correct :
  (∀ s : TriangularSubdivision, isValidSubdivision s → s.numTriangles ≤ maxTriangles) ∧
  (∃ s : TriangularSubdivision, isValidSubdivision s ∧ s.numTriangles = maxTriangles) :=
sorry

end no_larger_subdivision_exists_max_subdivision_exists_max_triangles_is_correct_l2509_250939


namespace tan_330_degrees_l2509_250901

theorem tan_330_degrees : Real.tan (330 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_330_degrees_l2509_250901


namespace average_weight_of_all_boys_l2509_250918

theorem average_weight_of_all_boys (group1_count : ℕ) (group1_avg : ℝ) 
  (group2_count : ℕ) (group2_avg : ℝ) : 
  group1_count = 16 → 
  group1_avg = 50.25 → 
  group2_count = 8 → 
  group2_avg = 45.15 → 
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  let total_count := group1_count + group2_count
  (total_weight / total_count) = 48.55 := by
sorry

end average_weight_of_all_boys_l2509_250918


namespace count_perfect_square_factors_l2509_250952

/-- The number of factors of 12000 that are perfect squares -/
def num_perfect_square_factors : ℕ :=
  sorry

/-- 12000 expressed as its prime factorization -/
def twelve_thousand_factorization : ℕ :=
  2^5 * 3 * 5^3

theorem count_perfect_square_factors :
  num_perfect_square_factors = 6 ∧ twelve_thousand_factorization = 12000 :=
sorry

end count_perfect_square_factors_l2509_250952


namespace vip_tickets_count_l2509_250926

theorem vip_tickets_count (initial_savings : ℕ) (vip_ticket_cost : ℕ) (regular_ticket_cost : ℕ) (regular_tickets_count : ℕ) (remaining_money : ℕ) : 
  initial_savings = 500 →
  vip_ticket_cost = 100 →
  regular_ticket_cost = 50 →
  regular_tickets_count = 3 →
  remaining_money = 150 →
  ∃ vip_tickets_count : ℕ, 
    vip_tickets_count * vip_ticket_cost + regular_tickets_count * regular_ticket_cost = initial_savings - remaining_money ∧
    vip_tickets_count = 2 :=
by sorry

end vip_tickets_count_l2509_250926


namespace range_of_a_l2509_250940

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {0, 1, a}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- State the theorem
theorem range_of_a (a : ℝ) : A a ∩ B = {1, a} → a ∈ Set.Ioo 0 1 ∪ Set.Ioo 1 2 := by
  sorry

end range_of_a_l2509_250940


namespace distance_point_to_line_l2509_250985

/-- The distance between a point and a horizontal line is the absolute difference
    between their y-coordinates. -/
def distance_point_to_horizontal_line (point : ℝ × ℝ) (line_y : ℝ) : ℝ :=
  |point.2 - line_y|

/-- Theorem: The distance between the point (3, 0) and the line y = 1 is 1. -/
theorem distance_point_to_line : distance_point_to_horizontal_line (3, 0) 1 = 1 := by
  sorry

end distance_point_to_line_l2509_250985


namespace polynomial_division_remainder_l2509_250909

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 2*x^3 = (x^2 + 6*x + 2) * q + (22*x^2 + 8*x) := by
  sorry

end polynomial_division_remainder_l2509_250909


namespace johns_journey_distance_l2509_250921

/-- Calculates the total distance traveled given two journey segments -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem: The total distance traveled in John's journey is 255 miles -/
theorem johns_journey_distance :
  total_distance 45 2 55 3 = 255 := by
  sorry

end johns_journey_distance_l2509_250921


namespace total_cookies_is_16000_l2509_250975

/-- The number of church members volunteering to bake cookies. -/
def num_members : ℕ := 100

/-- The number of sheets of cookies each member bakes. -/
def sheets_per_member : ℕ := 10

/-- The number of cookies on each sheet. -/
def cookies_per_sheet : ℕ := 16

/-- The total number of cookies baked by all church members. -/
def total_cookies : ℕ := num_members * sheets_per_member * cookies_per_sheet

/-- Theorem stating that the total number of cookies baked is 16,000. -/
theorem total_cookies_is_16000 : total_cookies = 16000 := by
  sorry

end total_cookies_is_16000_l2509_250975


namespace davids_age_l2509_250980

theorem davids_age (david : ℕ) (yuan : ℕ) : 
  yuan = david + 7 → yuan = 2 * david → david = 7 := by
  sorry

end davids_age_l2509_250980


namespace x_range_theorem_l2509_250914

-- Define the condition from the original problem
def satisfies_equation (x y : ℝ) : Prop :=
  x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y)

-- Define the range of x
def x_range (x : ℝ) : Prop :=
  x ∈ Set.Icc 4 20 ∪ {0}

-- Theorem statement
theorem x_range_theorem :
  ∀ x y : ℝ, satisfies_equation x y → x_range x :=
by
  sorry

end x_range_theorem_l2509_250914


namespace problem_statement_l2509_250955

theorem problem_statement (a b c m n : ℝ) 
  (h1 : a - b = m) 
  (h2 : b - c = n) : 
  a^2 + b^2 + c^2 - a*b - b*c - c*a = m^2 + n^2 + m*n := by
sorry

end problem_statement_l2509_250955


namespace lino_shell_collection_l2509_250905

/-- Theorem: Lino's shell collection
  Given:
  - Lino put 292 shells back in the afternoon
  - She has 32 shells in total at the end
  Prove that Lino picked up 324 shells in the morning
-/
theorem lino_shell_collection (shells_put_back shells_remaining : ℕ) 
  (h1 : shells_put_back = 292)
  (h2 : shells_remaining = 32) :
  shells_put_back + shells_remaining = 324 := by
  sorry

end lino_shell_collection_l2509_250905


namespace simplify_and_rationalize_l2509_250958

theorem simplify_and_rationalize :
  (Real.sqrt 6 / Real.sqrt 10) * (Real.sqrt 5 / Real.sqrt 15) * (Real.sqrt 8 / Real.sqrt 14) = 2 * Real.sqrt 35 / 35 := by
  sorry

end simplify_and_rationalize_l2509_250958


namespace equation_solution_l2509_250981

theorem equation_solution : ∃ x : ℝ, (x - 3) ^ 4 = (1 / 16)⁻¹ ∧ x = 5 := by
  sorry

end equation_solution_l2509_250981


namespace circle_symmetry_line_l2509_250963

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The property of a circle being symmetric with respect to a line -/
def isSymmetric (c : Circle) (l : Line) : Prop := sorry

theorem circle_symmetry_line (m : ℝ) :
  let c : Circle := { equation := fun x y => x^2 + y^2 + 2*x - 4*y = 0 }
  let l : Line := { equation := fun x y => 3*x + y + m = 0 }
  isSymmetric c l → m = 1 := by
  sorry

end circle_symmetry_line_l2509_250963


namespace max_value_sum_of_squares_l2509_250915

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_value_sum_of_squares (u v w : V) 
  (hu : ‖u‖ = 3) (hv : ‖v‖ = 1) (hw : ‖w‖ = 2) : 
  ‖u - 3 • v‖^2 + ‖v - 3 • w‖^2 + ‖w - 3 • u‖^2 ≤ 224 := by sorry

end max_value_sum_of_squares_l2509_250915


namespace income_percentage_increase_l2509_250988

/-- Calculates the percentage increase in monthly income given initial and new weekly incomes -/
theorem income_percentage_increase 
  (initial_job_income initial_freelance_income : ℚ)
  (new_job_income new_freelance_income : ℚ)
  (weeks_per_month : ℕ)
  (h1 : initial_job_income = 60)
  (h2 : initial_freelance_income = 40)
  (h3 : new_job_income = 120)
  (h4 : new_freelance_income = 60)
  (h5 : weeks_per_month = 4) :
  let initial_monthly_income := (initial_job_income + initial_freelance_income) * weeks_per_month
  let new_monthly_income := (new_job_income + new_freelance_income) * weeks_per_month
  (new_monthly_income - initial_monthly_income) / initial_monthly_income * 100 = 80 :=
by sorry


end income_percentage_increase_l2509_250988


namespace pigeon_difference_l2509_250964

theorem pigeon_difference (total_pigeons : ℕ) (black_ratio : ℚ) (male_ratio : ℚ) : 
  total_pigeons = 70 →
  black_ratio = 1/2 →
  male_ratio = 1/5 →
  (black_ratio * total_pigeons : ℚ) * (1 - male_ratio) - (black_ratio * total_pigeons : ℚ) * male_ratio = 21 := by
  sorry

end pigeon_difference_l2509_250964


namespace fraction_to_decimal_plus_two_l2509_250935

theorem fraction_to_decimal_plus_two : (7 : ℚ) / 16 + 2 = (2.4375 : ℚ) := by
  sorry

end fraction_to_decimal_plus_two_l2509_250935


namespace quadratic_inequality_range_l2509_250986

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end quadratic_inequality_range_l2509_250986
