import Mathlib

namespace NUMINAMATH_CALUDE_mixture_weight_l1477_147747

/-- Given substances a and b mixed in a ratio of 9:11, prove that the total weight
    of the mixture is 58 kg when 26.1 kg of substance a is used. -/
theorem mixture_weight (a b : ℝ) (h1 : a / b = 9 / 11) (h2 : a = 26.1) :
  a + b = 58 := by
  sorry

end NUMINAMATH_CALUDE_mixture_weight_l1477_147747


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1477_147709

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔ 
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1477_147709


namespace NUMINAMATH_CALUDE_expression_evaluation_l1477_147766

theorem expression_evaluation (x y : ℚ) (hx : x = 4 / 7) (hy : y = 6 / 8) :
  (7 * x + 8 * y) / (56 * x * y) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1477_147766


namespace NUMINAMATH_CALUDE_unsold_tomatoes_l1477_147732

def total_harvested : ℝ := 245.5
def sold_to_maxwell : ℝ := 125.5
def sold_to_wilson : ℝ := 78

theorem unsold_tomatoes : 
  total_harvested - (sold_to_maxwell + sold_to_wilson) = 42 := by
  sorry

end NUMINAMATH_CALUDE_unsold_tomatoes_l1477_147732


namespace NUMINAMATH_CALUDE_inverse_proportional_solution_l1477_147774

-- Define the inverse proportionality constant
def C : ℝ := 315

-- Define the relationship between x and y
def inverse_proportional (x y : ℝ) : Prop := x * y = C

-- State the theorem
theorem inverse_proportional_solution :
  ∀ x y : ℝ,
  inverse_proportional x y →
  x + y = 36 →
  x - y = 6 →
  x = 7 →
  y = 45 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportional_solution_l1477_147774


namespace NUMINAMATH_CALUDE_trigonometric_product_l1477_147783

theorem trigonometric_product (cos60 sin60 cos30 sin30 : ℝ) : 
  cos60 = 1/2 →
  sin60 = Real.sqrt 3 / 2 →
  cos30 = Real.sqrt 3 / 2 →
  sin30 = 1/2 →
  (1 - 1/cos30) * (1 + 1/sin60) * (1 - 1/sin30) * (1 + 1/cos60) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_l1477_147783


namespace NUMINAMATH_CALUDE_carrots_total_l1477_147744

/-- The number of carrots Sandy grew -/
def sandy_carrots : ℕ := 8

/-- The number of carrots Mary grew -/
def mary_carrots : ℕ := 6

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := sandy_carrots + mary_carrots

theorem carrots_total : total_carrots = 14 := by
  sorry

end NUMINAMATH_CALUDE_carrots_total_l1477_147744


namespace NUMINAMATH_CALUDE_rooks_placement_formula_l1477_147715

/-- The number of ways to place k non-attacking rooks on an n × n chessboard -/
def rooks_placement (n k : ℕ) : ℕ :=
  Nat.choose n k * Nat.descFactorial n k

/-- An n × n chessboard -/
structure Chessboard (n : ℕ) where
  size : ℕ := n

theorem rooks_placement_formula {n k : ℕ} (C : Chessboard n) (h : k ≤ n) :
  rooks_placement n k = Nat.choose n k * Nat.descFactorial n k := by
  sorry

end NUMINAMATH_CALUDE_rooks_placement_formula_l1477_147715


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1477_147764

theorem simplify_and_evaluate (b : ℝ) : 
  (15 * b^5) / (75 * b^3) = b^2 / 5 ∧ 
  (15 * 4^5) / (75 * 4^3) = 16 / 5 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1477_147764


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1477_147724

/-- An arithmetic sequence with a_4 = 5 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d ∧ a 4 = 5

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  2 * (a 1) - (a 5) + (a 11) = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1477_147724


namespace NUMINAMATH_CALUDE_sandbox_capacity_increase_l1477_147753

theorem sandbox_capacity_increase (l w h : ℝ) : 
  l * w * h = 10 → (2 * l) * (2 * w) * (2 * h) = 80 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_capacity_increase_l1477_147753


namespace NUMINAMATH_CALUDE_alvin_marbles_l1477_147700

def marble_game (initial : ℕ) (game1 : ℤ) (game2 : ℤ) (game3 : ℤ) (game4 : ℤ) (give : ℕ) (receive : ℕ) : ℕ :=
  (initial : ℤ) + game1 + game2 + game3 + game4 - give + receive |>.toNat

theorem alvin_marbles : 
  marble_game 57 (-18) 25 (-12) 15 10 8 = 65 := by
  sorry

end NUMINAMATH_CALUDE_alvin_marbles_l1477_147700


namespace NUMINAMATH_CALUDE_linear_systems_solutions_l1477_147772

theorem linear_systems_solutions :
  -- First system
  (∃ x y : ℝ, x + y = 5 ∧ 4*x - 2*y = 2 ∧ x = 2 ∧ y = 3) ∧
  -- Second system
  (∃ x y : ℝ, 3*x - 2*y = 13 ∧ 4*x + 3*y = 6 ∧ x = 3 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_linear_systems_solutions_l1477_147772


namespace NUMINAMATH_CALUDE_df_length_is_six_l1477_147746

/-- Represents a triangle with side lengths and an angle --/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle : ℝ

/-- The perimeter of a triangle --/
def perimeter (t : Triangle) : ℝ := t.side1 + t.side2 + t.side3

/-- Given two triangles ABC and DEF with the specified properties,
    prove that the length of DF is 6 cm --/
theorem df_length_is_six 
  (ABC : Triangle)
  (DEF : Triangle)
  (angle_relation : ABC.angle = 2 * DEF.angle)
  (ab_length : ABC.side1 = 4)
  (ac_length : ABC.side2 = 6)
  (de_length : DEF.side1 = 2)
  (perimeter_relation : perimeter ABC = 2 * perimeter DEF) :
  DEF.side2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_df_length_is_six_l1477_147746


namespace NUMINAMATH_CALUDE_tangent_parabola_hyperbola_l1477_147780

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 5

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 1

/-- Tangency condition -/
def are_tangent (m : ℝ) : Prop := ∃ (x y : ℝ), parabola x y ∧ hyperbola m x y ∧
  ∀ (x' y' : ℝ), parabola x' y' ∧ hyperbola m x' y' → (x' = x ∧ y' = y)

theorem tangent_parabola_hyperbola (m : ℝ) :
  are_tangent m ↔ (m = 10 + 2 * Real.sqrt 6 ∨ m = 10 - 2 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_tangent_parabola_hyperbola_l1477_147780


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_point_l1477_147775

theorem hyperbola_asymptote_point (a : ℝ) (h1 : a > 0) : 
  (∃ (x y : ℝ), x^2/4 - y^2/a = 1 ∧ 
   (y = (Real.sqrt a / 2) * x ∨ y = -(Real.sqrt a / 2) * x) ∧
   x = 2 ∧ y = Real.sqrt 3) → 
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_point_l1477_147775


namespace NUMINAMATH_CALUDE_weight_of_fresh_grapes_l1477_147742

/-- Given that fresh grapes contain 90% water by weight, dried grapes contain 20% water by weight,
    and the weight of dry grapes available is 3.125 kg, prove that the weight of fresh grapes is 78.125 kg. -/
theorem weight_of_fresh_grapes :
  let fresh_water_ratio : ℝ := 0.9
  let dried_water_ratio : ℝ := 0.2
  let dried_grapes_weight : ℝ := 3.125
  fresh_water_ratio * fresh_grapes_weight = dried_water_ratio * dried_grapes_weight + 
    (1 - dried_water_ratio) * dried_grapes_weight →
  fresh_grapes_weight = 78.125
  := by sorry

#check weight_of_fresh_grapes

end NUMINAMATH_CALUDE_weight_of_fresh_grapes_l1477_147742


namespace NUMINAMATH_CALUDE_non_similar_1500_pointed_stars_l1477_147779

/-- The number of non-similar regular n-pointed stars -/
def num_non_similar_stars (n : ℕ) : ℕ := 
  (Nat.totient n - 2) / 2

/-- Properties of regular n-pointed stars -/
axiom regular_star_properties (n : ℕ) : 
  ∃ (prop : ℕ → Prop), prop n ∧ prop 1000

theorem non_similar_1500_pointed_stars : 
  num_non_similar_stars 1500 = 199 := by
  sorry

end NUMINAMATH_CALUDE_non_similar_1500_pointed_stars_l1477_147779


namespace NUMINAMATH_CALUDE_vectors_form_basis_l1477_147743

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : LinearIndependent ℝ ![e₁, e₂] ∧ Submodule.span ℝ {e₁, e₂} = ⊤ := by
  sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l1477_147743


namespace NUMINAMATH_CALUDE_shortest_path_length_l1477_147750

/-- The shortest path length from (0,0) to (12,16) avoiding a circle -/
theorem shortest_path_length (start end_ circle_center : ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  let path_length := 10 * Real.sqrt 3 + 5 * Real.pi / 3
  by
    sorry

#check shortest_path_length (0, 0) (12, 16) (6, 8) 5

end NUMINAMATH_CALUDE_shortest_path_length_l1477_147750


namespace NUMINAMATH_CALUDE_marble_probability_l1477_147730

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) :
  p_white = 1/4 →
  p_green = 2/7 →
  p_white + p_green + (1 - p_white - p_green) = 1 →
  1 - p_white - p_green = 13/28 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1477_147730


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1477_147731

/-- Given a line L1 with equation 2x + y - 1 = 0 and a point P (-1, 2),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation x - 2y + 5 = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  L1 = {(x, y) | 2 * x + y - 1 = 0} →
  P = (-1, 2) →
  ∃ L2 : Set (ℝ × ℝ),
    (P ∈ L2) ∧
    (∀ (p q : ℝ × ℝ), p ∈ L1 → q ∈ L1 → p ≠ q →
      ∀ (r s : ℝ × ℝ), r ∈ L2 → s ∈ L2 → r ≠ s →
        (p.1 - q.1) * (r.1 - s.1) + (p.2 - q.2) * (r.2 - s.2) = 0) ∧
    L2 = {(x, y) | x - 2 * y + 5 = 0} :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1477_147731


namespace NUMINAMATH_CALUDE_problem_statement_l1477_147726

theorem problem_statement (a b c d : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > -a) 
  (h4 : c < d) (h5 : d < 0) : 
  (a / d + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1477_147726


namespace NUMINAMATH_CALUDE_triangle_formation_l1477_147762

/-- Triangle inequality theorem: the sum of the lengths of any two sides 
    of a triangle must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 5 6 10 ∧
  ¬can_form_triangle 2 3 5 ∧
  ¬can_form_triangle 5 6 11 ∧
  ¬can_form_triangle 3 4 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l1477_147762


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l1477_147725

theorem dining_bill_calculation (total : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) 
  (service_charge : ℝ) (dessert_cost : ℝ) (food_price : ℝ) : 
  total = 211.20 →
  tip_rate = 0.20 →
  tax_rate = 0.10 →
  service_charge = 5 →
  dessert_cost = 8 →
  food_price * (1 + tip_rate + tax_rate) + service_charge + dessert_cost = total →
  food_price = 152.46 := by
sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l1477_147725


namespace NUMINAMATH_CALUDE_mom_bought_71_packages_l1477_147718

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The total number of t-shirts Mom has -/
def total_shirts : ℕ := 426

/-- The number of packages Mom bought -/
def num_packages : ℕ := total_shirts / shirts_per_package

theorem mom_bought_71_packages : num_packages = 71 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_71_packages_l1477_147718


namespace NUMINAMATH_CALUDE_fraction_left_handed_l1477_147754

/-- Represents the ratio of red to blue participants -/
def red_blue_ratio : ℚ := 10 / 5

/-- Fraction of left-handed red participants -/
def left_handed_red : ℚ := 1 / 3

/-- Fraction of left-handed blue participants -/
def left_handed_blue : ℚ := 2 / 3

/-- Theorem: The fraction of left-handed participants is 4/9 -/
theorem fraction_left_handed :
  let total_ratio := red_blue_ratio + 1
  let left_handed_ratio := red_blue_ratio * left_handed_red + left_handed_blue
  left_handed_ratio / total_ratio = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_fraction_left_handed_l1477_147754


namespace NUMINAMATH_CALUDE_tent_count_solution_l1477_147799

def total_value : ℕ := 940000
def total_tents : ℕ := 600
def cost_A : ℕ := 1700
def cost_B : ℕ := 1300

theorem tent_count_solution :
  ∃ (x y : ℕ),
    x + y = total_tents ∧
    cost_A * x + cost_B * y = total_value ∧
    x = 400 ∧
    y = 200 := by
  sorry

end NUMINAMATH_CALUDE_tent_count_solution_l1477_147799


namespace NUMINAMATH_CALUDE_garden_perimeter_l1477_147756

/-- 
Given a rectangular garden with one side of length 10 feet and an area of 80 square feet,
prove that the perimeter of the garden is 36 feet.
-/
theorem garden_perimeter : ∀ (width : ℝ), 
  width > 0 →
  10 * width = 80 →
  2 * (10 + width) = 36 := by
  sorry


end NUMINAMATH_CALUDE_garden_perimeter_l1477_147756


namespace NUMINAMATH_CALUDE_line_slope_through_circle_l1477_147776

/-- Given a line passing through (0,√5) and intersecting the circle x^2 + y^2 = 16 at points A and B,
    if a point P on the circle satisfies OP = OA + OB, then the slope of the line is ±1/2. -/
theorem line_slope_through_circle (A B P : ℝ × ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let line := {(x, y) : ℝ × ℝ | ∃ (k : ℝ), y - Real.sqrt 5 = k * x}
  (0, Real.sqrt 5) ∈ line ∧ 
  A ∈ line ∧ 
  B ∈ line ∧
  A.1^2 + A.2^2 = 16 ∧
  B.1^2 + B.2^2 = 16 ∧
  P.1^2 + P.2^2 = 16 ∧
  (P.1 - O.1, P.2 - O.2) = (A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2) →
  ∃ (k : ℝ), k = 1/2 ∨ k = -1/2 ∧ ∀ (x y : ℝ), (x, y) ∈ line ↔ y - Real.sqrt 5 = k * x :=
by sorry

end NUMINAMATH_CALUDE_line_slope_through_circle_l1477_147776


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_function_l1477_147751

open Real

theorem symmetry_of_shifted_function :
  ∃ α : ℝ, 0 < α ∧ α < π / 3 ∧
  ∀ x : ℝ, (sin (x + α) + Real.sqrt 3 * cos (x + α)) =
           (sin (-x + α) + Real.sqrt 3 * cos (-x + α)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_function_l1477_147751


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1477_147722

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), x^2 + 2*x + 4 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1477_147722


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1477_147723

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 36

theorem bowling_ball_weight_proof :
  (8 * bowling_ball_weight = 4 * canoe_weight) ∧
  (3 * canoe_weight = 108) →
  bowling_ball_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1477_147723


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_max_is_six_l1477_147745

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x < 1} = {x : ℝ | x < 1/2} := by sorry

-- Part II
theorem a_value_when_max_is_six :
  (∃ (x : ℝ), f a x = 6) ∧ (∀ (x : ℝ), f a x ≤ 6) → a = 5 ∨ a = -7 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_max_is_six_l1477_147745


namespace NUMINAMATH_CALUDE_ethanol_in_fuel_mix_l1477_147703

/-- Calculates the total ethanol in a fuel tank -/
def total_ethanol (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) : ℝ :=
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let ethanol_a := fuel_a_volume * fuel_a_ethanol_percent
  let ethanol_b := fuel_b_volume * fuel_b_ethanol_percent
  ethanol_a + ethanol_b

/-- Theorem: The total ethanol in the specified fuel mix is 30 gallons -/
theorem ethanol_in_fuel_mix :
  total_ethanol 200 49.99999999999999 0.12 0.16 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ethanol_in_fuel_mix_l1477_147703


namespace NUMINAMATH_CALUDE_cruz_marbles_l1477_147736

/-- 
Given:
- Three times the sum of marbles that Atticus, Jensen, and Cruz have is equal to 60.
- Atticus has half as many marbles as Jensen.
- Atticus has 4 marbles.
Prove that Cruz has 8 marbles.
-/
theorem cruz_marbles (atticus jensen cruz : ℕ) : 
  3 * (atticus + jensen + cruz) = 60 →
  atticus = jensen / 2 →
  atticus = 4 →
  cruz = 8 := by
sorry

end NUMINAMATH_CALUDE_cruz_marbles_l1477_147736


namespace NUMINAMATH_CALUDE_triangle_properties_l1477_147796

/-- Given a triangle ABC with the following properties:
    - Sides a, b, c are opposite to angles A, B, C respectively
    - Vector m = (2 * sin B, -√3)
    - Vector n = (cos(2B), 2 * cos²(B/2) - 1)
    - m is parallel to n
    - B is an acute angle
    - b = 2
    Prove that the measure of angle B is π/3 and the maximum area of the triangle is √3 -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (m : ℝ × ℝ) (n : ℝ × ℝ) :
  m.1 = 2 * Real.sin B ∧ 
  m.2 = -Real.sqrt 3 ∧
  n.1 = Real.cos (2 * B) ∧ 
  n.2 = 2 * (Real.cos (B / 2))^2 - 1 ∧
  ∃ (k : ℝ), m = k • n ∧
  0 < B ∧ B < π / 2 ∧
  b = 2 →
  B = π / 3 ∧ 
  (∀ (S : ℝ), S = 1/2 * a * c * Real.sin B → S ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1477_147796


namespace NUMINAMATH_CALUDE_smallest_with_twelve_odd_eighteen_even_divisors_l1477_147777

def count_odd_divisors (n : ℕ) : ℕ := 
  (Finset.filter (λ d => d % 2 = 1) (Nat.divisors n)).card

def count_even_divisors (n : ℕ) : ℕ := 
  (Finset.filter (λ d => d % 2 = 0) (Nat.divisors n)).card

theorem smallest_with_twelve_odd_eighteen_even_divisors :
  ∀ n : ℕ, n > 0 → 
    (count_odd_divisors n = 12 ∧ count_even_divisors n = 18) → 
    n ≥ 900 :=
sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_odd_eighteen_even_divisors_l1477_147777


namespace NUMINAMATH_CALUDE_math_competition_probabilities_l1477_147719

-- Define the total number of questions and the number of questions each student answers
def total_questions : ℕ := 6
def questions_answered : ℕ := 3

-- Define the number of questions student A can correctly answer
def student_a_correct : ℕ := 4

-- Define the probability of student B correctly answering a question
def student_b_prob : ℚ := 2/3

-- Define the point values for correct answers
def points_a : ℕ := 15
def points_b : ℕ := 10

-- Define the probability that students A and B together correctly answer 3 questions
def prob_three_correct : ℚ := 31/135

-- Define the expected value of the total score
def expected_total_score : ℕ := 50

-- Theorem statement
theorem math_competition_probabilities :
  (prob_three_correct = 31/135) ∧
  (expected_total_score = 50) := by
  sorry

end NUMINAMATH_CALUDE_math_competition_probabilities_l1477_147719


namespace NUMINAMATH_CALUDE_negation_existence_absolute_value_l1477_147713

theorem negation_existence_absolute_value (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, |x| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_absolute_value_l1477_147713


namespace NUMINAMATH_CALUDE_notebook_words_per_page_l1477_147784

theorem notebook_words_per_page :
  ∀ (words_per_page : ℕ),
    words_per_page > 0 →
    words_per_page ≤ 150 →
    (180 * words_per_page) % 221 = 246 % 221 →
    words_per_page = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_words_per_page_l1477_147784


namespace NUMINAMATH_CALUDE_min_prime_no_solution_l1477_147793

theorem min_prime_no_solution : 
  ∀ p : ℕ, Prime p → p > 3 →
    (∀ n : ℕ, n > 0 → ¬(2^n + 3^n) % p = 0) →
    p ≥ 19 :=
by sorry

end NUMINAMATH_CALUDE_min_prime_no_solution_l1477_147793


namespace NUMINAMATH_CALUDE_sum_of_parts_l1477_147794

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 54) (h2 : y = 34) : 10 * x + 22 * y = 948 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_l1477_147794


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1477_147778

/-- The line y - 1 = k(x - 1) is tangent to the circle x^2 + y^2 - 2y = 0 for any real k -/
theorem line_tangent_to_circle (k : ℝ) : 
  ∃! (x y : ℝ), (y - 1 = k * (x - 1)) ∧ (x^2 + y^2 - 2*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1477_147778


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1477_147781

-- Define the original proposition
def original_proposition (m : ℝ) : Prop :=
  m > 0 → ∃ x : ℝ, x^2 + x - m = 0

-- Define the contrapositive
def contrapositive (m : ℝ) : Prop :=
  (¬∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0

-- Theorem stating the equivalence of the contrapositive to the original proposition
theorem contrapositive_equivalence :
  ∀ m : ℝ, (¬original_proposition m) ↔ contrapositive m :=
by
  sorry


end NUMINAMATH_CALUDE_contrapositive_equivalence_l1477_147781


namespace NUMINAMATH_CALUDE_tiles_cut_to_square_and_rectangle_l1477_147707

/-- Represents a rectangular tile with width and height -/
structure Tile where
  width : ℝ
  height : ℝ

/-- Represents a rectangle formed by tiles -/
structure Rectangle where
  width : ℝ
  height : ℝ
  tiles : List Tile

/-- Theorem stating that tiles can be cut to form a square and a rectangle -/
theorem tiles_cut_to_square_and_rectangle 
  (n : ℕ) 
  (original : Rectangle) 
  (h_unequal_sides : original.width ≠ original.height) 
  (h_tile_count : original.tiles.length = n) :
  ∃ (square : Rectangle) (remaining : Rectangle),
    square.width = square.height ∧
    square.tiles.length = n ∧
    remaining.tiles.length = n ∧
    (∀ t ∈ original.tiles, ∃ t1 t2, t1 ∈ square.tiles ∧ t2 ∈ remaining.tiles) :=
sorry

end NUMINAMATH_CALUDE_tiles_cut_to_square_and_rectangle_l1477_147707


namespace NUMINAMATH_CALUDE_area_of_triangle_from_centers_area_is_sqrt_three_l1477_147795

/-- The area of an equilateral triangle formed by connecting the centers of three equilateral
    triangles of side length 2, arranged around a vertex of a square. -/
theorem area_of_triangle_from_centers : ℝ :=
  let side_length : ℝ := 2
  let triangle_centers_distance : ℝ := side_length
  let area_formula (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2
  area_formula triangle_centers_distance

/-- The area of the triangle formed by connecting the centers is √3. -/
theorem area_is_sqrt_three : area_of_triangle_from_centers = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_from_centers_area_is_sqrt_three_l1477_147795


namespace NUMINAMATH_CALUDE_negation_of_proposition_P_l1477_147710

theorem negation_of_proposition_P :
  (¬ (∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_P_l1477_147710


namespace NUMINAMATH_CALUDE_remainder_problem_l1477_147733

theorem remainder_problem (j : ℕ+) (h : 75 % (j^2 : ℕ) = 3) : 
  (130 % (j : ℕ) = 0) ∨ (130 % (j : ℕ) = 1) := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1477_147733


namespace NUMINAMATH_CALUDE_digit_removal_theorem_l1477_147763

def original_number : ℕ := 111123445678

-- Function to check if a number is divisible by 5
def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Function to represent the removal of digits
def remove_digits (n : ℕ) (removed : List ℕ) : ℕ := sorry

-- Function to count valid ways of digit removal
def count_valid_removals (n : ℕ) : ℕ := sorry

theorem digit_removal_theorem :
  count_valid_removals original_number = 60 := by sorry

end NUMINAMATH_CALUDE_digit_removal_theorem_l1477_147763


namespace NUMINAMATH_CALUDE_not_same_size_and_precision_l1477_147720

/-- Represents the precision of a decimal number -/
inductive Precision
| Tenths
| Hundredths

/-- Represents a decimal number with its value and precision -/
structure DecimalNumber where
  value : ℚ
  precision : Precision

/-- Check if two DecimalNumbers have the same size and precision -/
def sameSizeAndPrecision (a b : DecimalNumber) : Prop :=
  a.value = b.value ∧ a.precision = b.precision

theorem not_same_size_and_precision :
  ¬(sameSizeAndPrecision
    { value := 1.2, precision := Precision.Hundredths }
    { value := 1.2, precision := Precision.Tenths }) := by
  sorry

end NUMINAMATH_CALUDE_not_same_size_and_precision_l1477_147720


namespace NUMINAMATH_CALUDE_pool_filling_rates_l1477_147761

theorem pool_filling_rates (r₁ r₂ r₃ : ℝ) 
  (h1 : r₁ + r₂ = 1 / 70)
  (h2 : r₁ + r₃ = 1 / 84)
  (h3 : r₂ + r₃ = 1 / 140) :
  r₁ = 1 / 105 ∧ r₂ = 1 / 210 ∧ r₃ = 1 / 420 ∧ r₁ + r₂ + r₃ = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_rates_l1477_147761


namespace NUMINAMATH_CALUDE_fixed_points_bound_l1477_147755

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial n) : ℕ := n - 1

/-- Evaluation of a polynomial at a point -/
def eval (p : IntPolynomial n) (x : ℤ) : ℤ := sorry

/-- Composition of a polynomial with itself k times -/
def composeK (p : IntPolynomial n) (k : ℕ) : IntPolynomial n := sorry

/-- The number of integer solutions to the equation Q_k(t) = t -/
def numFixedPoints (p : IntPolynomial n) (k : ℕ) : ℕ := sorry

theorem fixed_points_bound (n : ℕ) (p : IntPolynomial n) (k : ℕ) :
  degree p > 1 → numFixedPoints p k ≤ degree p := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_bound_l1477_147755


namespace NUMINAMATH_CALUDE_download_time_proof_l1477_147734

def internet_speed : ℝ := 2
def file1_size : ℝ := 80
def file2_size : ℝ := 90
def file3_size : ℝ := 70
def minutes_per_hour : ℝ := 60

theorem download_time_proof :
  let total_size := file1_size + file2_size + file3_size
  let download_time_minutes := total_size / internet_speed
  let download_time_hours := download_time_minutes / minutes_per_hour
  download_time_hours = 2 := by
sorry

end NUMINAMATH_CALUDE_download_time_proof_l1477_147734


namespace NUMINAMATH_CALUDE_same_solution_k_value_l1477_147748

theorem same_solution_k_value (x k : ℝ) : 
  (2 * x + 4 = 4 * (x - 2) ∧ -x + k = 2 * x - 1) ↔ k = 17 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_k_value_l1477_147748


namespace NUMINAMATH_CALUDE_ball_probability_l1477_147708

/-- Given a bag of 100 balls with specified colors, prove the probability of choosing a ball that is neither red nor purple -/
theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 100)
  (h2 : white = 50)
  (h3 : green = 30)
  (h4 : yellow = 8)
  (h5 : red = 9)
  (h6 : purple = 3)
  (h7 : total = white + green + yellow + red + purple) :
  (white + green + yellow : ℚ) / total = 88 / 100 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l1477_147708


namespace NUMINAMATH_CALUDE_simplified_expression_implies_A_l1477_147739

/-- 
Given that (A - 3 / (a - 1)) * ((2 * a - 2) / (a + 2)) = 2 * a - 4,
prove that A = a + 1
-/
theorem simplified_expression_implies_A (a : ℝ) (A : ℝ) 
  (h : (A - 3 / (a - 1)) * ((2 * a - 2) / (a + 2)) = 2 * a - 4) :
  A = a + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_implies_A_l1477_147739


namespace NUMINAMATH_CALUDE_rhombus_in_rectangle_perimeter_l1477_147752

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Checks if a quadrilateral is a rhombus -/
def is_rhombus (q : Quadrilateral) : Prop := sorry

/-- Checks if a quadrilateral is a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is on a line segment -/
def is_on_segment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 : Point) (p2 : Point) : ℝ := sorry

/-- Calculates the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

theorem rhombus_in_rectangle_perimeter 
  (W X Y Z : Point) 
  (A B C D : Point) :
  let rect := Quadrilateral.mk W X Y Z
  let rhom := Quadrilateral.mk A B C D
  is_rectangle rect →
  is_rhombus rhom →
  is_on_segment A W X →
  is_on_segment B X Y →
  is_on_segment C Y Z →
  is_on_segment D Z W →
  distance W A = 12 →
  distance X B = 9 →
  distance B D = 15 →
  distance A C = distance X Y →
  perimeter rect = 66 := by sorry

end NUMINAMATH_CALUDE_rhombus_in_rectangle_perimeter_l1477_147752


namespace NUMINAMATH_CALUDE_stationery_problem_solution_l1477_147737

/-- Represents a box of stationery --/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Defines the conditions of the stationery problem --/
def stationeryProblem (box : StationeryBox) : Prop :=
  -- Tom's condition: all envelopes used, 100 sheets left
  box.sheets - box.envelopes = 100 ∧
  -- Jerry's condition: all sheets used, 25 envelopes left
  box.envelopes + 25 = box.sheets / 3

/-- The theorem stating the solution to the stationery problem --/
theorem stationery_problem_solution :
  ∃ (box : StationeryBox), stationeryProblem box ∧ box.sheets = 120 :=
sorry

end NUMINAMATH_CALUDE_stationery_problem_solution_l1477_147737


namespace NUMINAMATH_CALUDE_two_valid_numbers_l1477_147771

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000 ∧ n < 1000000) ∧  -- six-digit number
  (n % 72 = 0) ∧  -- divisible by 72
  (∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = a * 100000 + 2016 * 10 + b)  -- formed by adding digits to 2016

theorem two_valid_numbers :
  {n : ℕ | is_valid_number n} = {920160, 120168} := by sorry

end NUMINAMATH_CALUDE_two_valid_numbers_l1477_147771


namespace NUMINAMATH_CALUDE_min_value_on_line_l1477_147721

/-- The minimum value of 2^x + 4^y for points (x, y) on the line through (3, 0) and (1, 1) -/
theorem min_value_on_line : 
  ∀ (x y : ℝ), (x + 2*y = 3) → (2^x + 4^y ≥ 4 * Real.sqrt 2) ∧ 
  ∃ (x₀ y₀ : ℝ), (x₀ + 2*y₀ = 3) ∧ (2^x₀ + 4^y₀ = 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_line_l1477_147721


namespace NUMINAMATH_CALUDE_f_simplification_l1477_147728

def f (x : ℝ) : ℝ := (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1

theorem f_simplification (x : ℝ) : f x = 32 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_f_simplification_l1477_147728


namespace NUMINAMATH_CALUDE_eighth_term_value_l1477_147798

/-- An arithmetic sequence with 30 terms, first term 5, and last term 80 -/
def arithmeticSequence (n : ℕ) : ℚ :=
  let d := (80 - 5) / 29
  5 + (n - 1) * d

/-- The 8th term of the arithmetic sequence -/
def eighthTerm : ℚ := arithmeticSequence 8

theorem eighth_term_value : eighthTerm = 670 / 29 := by sorry

end NUMINAMATH_CALUDE_eighth_term_value_l1477_147798


namespace NUMINAMATH_CALUDE_initial_balls_count_l1477_147758

def process (x : ℕ) : ℕ := x / 2 + 1

def iterate_process (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | m + 1 => process (iterate_process n m)

theorem initial_balls_count (n : ℕ) : iterate_process n 2010 = 2 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_balls_count_l1477_147758


namespace NUMINAMATH_CALUDE_orchid_bushes_planted_l1477_147701

/-- The number of orchid bushes planted in the park -/
theorem orchid_bushes_planted (initial : ℕ) (final : ℕ) (planted : ℕ) : 
  initial = 2 → final = 6 → planted = final - initial → planted = 4 := by
  sorry

end NUMINAMATH_CALUDE_orchid_bushes_planted_l1477_147701


namespace NUMINAMATH_CALUDE_connect_four_shapes_l1477_147790

/-- Represents a Connect Four board configuration --/
def ConnectFourBoard := Fin 7 → Fin 9

/-- The number of unique shapes in a Connect Four board --/
def num_unique_shapes : ℕ :=
  let symmetric_shapes := 9^4
  let total_shapes := 9^7
  symmetric_shapes + (total_shapes - symmetric_shapes) / 2

/-- The sum of the first n natural numbers --/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that the number of unique shapes in a Connect Four board
    is equal to 9 times the sum of the first 729 natural numbers --/
theorem connect_four_shapes :
  num_unique_shapes = 9 * sum_first_n 729 := by
  sorry


end NUMINAMATH_CALUDE_connect_four_shapes_l1477_147790


namespace NUMINAMATH_CALUDE_exam_mean_score_l1477_147716

/-- Given an exam where a score of 42 is 5 standard deviations below the mean
    and a score of 67 is 2.5 standard deviations above the mean,
    prove that the mean score is 440/7.5 -/
theorem exam_mean_score (μ σ : ℝ) 
  (h1 : 42 = μ - 5 * σ)
  (h2 : 67 = μ + 2.5 * σ) : 
  μ = 440 / 7.5 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l1477_147716


namespace NUMINAMATH_CALUDE_projection_of_v_onto_u_l1477_147767

def v : Fin 2 → ℚ := ![5, 7]
def u : Fin 2 → ℚ := ![1, -3]

def projection (v u : Fin 2 → ℚ) : Fin 2 → ℚ :=
  let dot_product := (v 0) * (u 0) + (v 1) * (u 1)
  let magnitude_squared := (u 0)^2 + (u 1)^2
  let scalar := dot_product / magnitude_squared
  ![scalar * (u 0), scalar * (u 1)]

theorem projection_of_v_onto_u :
  projection v u = ![-8/5, 24/5] := by sorry

end NUMINAMATH_CALUDE_projection_of_v_onto_u_l1477_147767


namespace NUMINAMATH_CALUDE_carla_marbles_count_l1477_147787

/-- The number of marbles Carla had before -/
def initial_marbles : ℝ := 187.0

/-- The number of marbles Carla bought -/
def bought_marbles : ℝ := 134.0

/-- The total number of marbles Carla has now -/
def total_marbles : ℝ := initial_marbles + bought_marbles

/-- Theorem: The total number of marbles Carla has now is 321.0 -/
theorem carla_marbles_count : total_marbles = 321.0 := by
  sorry

end NUMINAMATH_CALUDE_carla_marbles_count_l1477_147787


namespace NUMINAMATH_CALUDE_class_size_l1477_147788

theorem class_size (total : ℚ) 
  (h1 : (2 : ℚ) / 3 * total = total * (2 : ℚ) / 3)  -- Two-thirds of the class have brown eyes
  (h2 : (1 : ℚ) / 2 * (total * (2 : ℚ) / 3) = total / 3)  -- Half of the students with brown eyes have black hair
  (h3 : (total / 3 : ℚ) = 6)  -- There are 6 students with brown eyes and black hair
  : total = 18 := by
sorry

end NUMINAMATH_CALUDE_class_size_l1477_147788


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1477_147791

/-- Given a triangle with sides 9, 12, and 15, the shortest altitude has length 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 →
  (a^2 + b^2 = c^2) →
  (h * c = 2 * (a * b / 2)) →
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1477_147791


namespace NUMINAMATH_CALUDE_function_through_point_l1477_147740

theorem function_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (fun x : ℝ ↦ a^x) (-1) = 2 → (fun x : ℝ ↦ a^x) = (fun x : ℝ ↦ (1/2)^x) := by
  sorry

end NUMINAMATH_CALUDE_function_through_point_l1477_147740


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1477_147759

/-- Given a hyperbola with center at (-3, 1), one focus at (-3 + √41, 1), and one vertex at (-7, 1),
    prove that h + k + a + b = 7, where (h, k) is the center, a is the distance from the center to
    the vertex, and b² = c² - a² (c being the distance from the center to the focus). -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 ∧ 
  k = 1 ∧ 
  (h + Real.sqrt 41 - h)^2 = c^2 ∧ 
  (h - 4 - h)^2 = a^2 ∧ 
  b^2 = c^2 - a^2 → 
  h + k + a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1477_147759


namespace NUMINAMATH_CALUDE_point_q_in_third_quadrant_l1477_147717

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def is_in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A is in the second quadrant, then Q is in the third quadrant -/
theorem point_q_in_third_quadrant (A : Point) (h : is_in_second_quadrant A) :
  let Q : Point := ⟨A.x, -A.y⟩
  is_in_third_quadrant Q :=
by
  sorry

end NUMINAMATH_CALUDE_point_q_in_third_quadrant_l1477_147717


namespace NUMINAMATH_CALUDE_abc_inequality_l1477_147760

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a + b + c = 1) :
  min (a - a*b) (min (b - b*c) (c - c*a)) ≤ 1/4 ∧ 
  max (a - a*b) (max (b - b*c) (c - c*a)) ≥ 2/9 := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l1477_147760


namespace NUMINAMATH_CALUDE_jameson_medals_l1477_147704

theorem jameson_medals (total_medals : ℕ) (badminton_medals : ℕ) 
  (h1 : total_medals = 20)
  (h2 : badminton_medals = 5) :
  ∃ track_medals : ℕ, 
    track_medals + 2 * track_medals + badminton_medals = total_medals ∧ 
    track_medals = 5 := by
  sorry

end NUMINAMATH_CALUDE_jameson_medals_l1477_147704


namespace NUMINAMATH_CALUDE_equal_sampling_probability_l1477_147727

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Define a function to represent the probability of an individual being sampled
def samplingProbability (method : SamplingMethod) (individual : ℕ) : ℝ :=
  sorry

-- State the theorem
theorem equal_sampling_probability (method : SamplingMethod) (individual1 individual2 : ℕ) :
  samplingProbability method individual1 = samplingProbability method individual2 :=
sorry

end NUMINAMATH_CALUDE_equal_sampling_probability_l1477_147727


namespace NUMINAMATH_CALUDE_mechanic_days_worked_l1477_147797

/-- Calculates the number of days a mechanic worked on a car given the following conditions:
  * Hourly rate charged by the mechanic
  * Hours worked per day
  * Cost of parts used
  * Total amount paid by the car owner
-/
def days_worked (hourly_rate : ℚ) (hours_per_day : ℚ) (parts_cost : ℚ) (total_paid : ℚ) : ℚ :=
  (total_paid - parts_cost) / (hourly_rate * hours_per_day)

/-- Theorem stating that given the specific conditions in the problem,
    the number of days worked by the mechanic is 14 -/
theorem mechanic_days_worked :
  days_worked 60 8 2500 9220 = 14 := by
  sorry


end NUMINAMATH_CALUDE_mechanic_days_worked_l1477_147797


namespace NUMINAMATH_CALUDE_combined_weight_theorem_l1477_147705

def leo_weight : ℝ := 80
def weight_gain : ℝ := 10

theorem combined_weight_theorem (kendra_weight : ℝ) 
  (h : leo_weight + weight_gain = 1.5 * kendra_weight) :
  leo_weight + kendra_weight = 140 := by
sorry

end NUMINAMATH_CALUDE_combined_weight_theorem_l1477_147705


namespace NUMINAMATH_CALUDE_farmhand_work_hours_l1477_147792

/-- Represents the number of golden delicious apples needed for one pint of cider -/
def golden_delicious_per_pint : ℕ := 20

/-- Represents the number of pink lady apples needed for one pint of cider -/
def pink_lady_per_pint : ℕ := 40

/-- Represents the number of farmhands -/
def num_farmhands : ℕ := 6

/-- Represents the number of apples each farmhand can pick per hour -/
def apples_per_hour_per_farmhand : ℕ := 240

/-- Represents the ratio of golden delicious to pink lady apples -/
def apple_ratio : Rat := 1 / 2

/-- Represents the number of pints of cider Haley can make with the gathered apples -/
def pints_of_cider : ℕ := 120

/-- Theorem stating that the farmhands will work for 5 hours -/
theorem farmhand_work_hours : 
  ∃ (hours : ℕ), 
    hours = 5 ∧ 
    hours * (num_farmhands * apples_per_hour_per_farmhand) = 
      pints_of_cider * (golden_delicious_per_pint + pink_lady_per_pint) ∧
    apple_ratio = (pints_of_cider * golden_delicious_per_pint : ℚ) / 
                  (pints_of_cider * pink_lady_per_pint : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_farmhand_work_hours_l1477_147792


namespace NUMINAMATH_CALUDE_files_remaining_l1477_147769

theorem files_remaining (initial_music : ℕ) (initial_video : ℕ) (deleted : ℕ) : 
  initial_music = 4 → initial_video = 21 → deleted = 23 → 
  initial_music + initial_video - deleted = 2 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l1477_147769


namespace NUMINAMATH_CALUDE_count_pairs_eq_45_l1477_147782

def count_pairs : Nat :=
  (Finset.range 6).sum fun m =>
    (Finset.range ((40 - (m + 1)^2) / 3 + 1)).card

theorem count_pairs_eq_45 : count_pairs = 45 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_45_l1477_147782


namespace NUMINAMATH_CALUDE_rehabilitation_centers_fraction_l1477_147735

theorem rehabilitation_centers_fraction (L J H Ja : ℕ) (f : ℚ) : 
  L = 6 →
  J = L - f * L →
  H = 2 * J - 2 →
  Ja = 2 * H + 6 →
  L + J + H + Ja = 27 →
  f = 1/2 := by sorry

end NUMINAMATH_CALUDE_rehabilitation_centers_fraction_l1477_147735


namespace NUMINAMATH_CALUDE_max_sum_xy_l1477_147757

theorem max_sum_xy (x y a b : ℝ) (hx : x > 0) (hy : y > 0) 
  (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a*x + b*y = 1) :
  x + y ≤ Real.sqrt 5 ∧ ∃ x y, x + y = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_xy_l1477_147757


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1477_147714

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1477_147714


namespace NUMINAMATH_CALUDE_mean_study_hours_thompson_class_l1477_147702

theorem mean_study_hours_thompson_class : 
  let study_hours := [0, 2, 4, 6, 8, 10, 12]
  let student_counts := [3, 6, 8, 5, 4, 2, 2]
  let total_students := 30
  let total_hours := (List.zip study_hours student_counts).map (fun (h, c) => h * c) |>.sum
  (total_hours : ℚ) / total_students = 5 := by
  sorry

end NUMINAMATH_CALUDE_mean_study_hours_thompson_class_l1477_147702


namespace NUMINAMATH_CALUDE_mixing_solutions_l1477_147773

/-- Proves that mixing 28 ounces of a 30% solution with 12 ounces of an 80% solution 
    results in a 45% solution with a total volume of 40 ounces. -/
theorem mixing_solutions (volume_30 volume_80 total_volume : ℝ) 
  (h1 : volume_30 = 28)
  (h2 : volume_80 = 12)
  (h3 : total_volume = volume_30 + volume_80) :
  (0.30 * volume_30 + 0.80 * volume_80) / total_volume = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_mixing_solutions_l1477_147773


namespace NUMINAMATH_CALUDE_golden_ratio_approximation_l1477_147738

theorem golden_ratio_approximation :
  (∃ (S : Set ℚ), Set.Infinite S ∧
    ∀ r ∈ S, ∃ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 ∧ r = q / p ∧
      |r - (Real.sqrt 5 - 1) / 2| < 1 / p^2) ∧
  (∀ p q : ℤ, p > 0 → Int.gcd p q = 1 →
    |(q : ℝ) / p - (Real.sqrt 5 - 1) / 2| > 1 / (Real.sqrt 5 + 1) / p^2) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_approximation_l1477_147738


namespace NUMINAMATH_CALUDE_product_sum_squares_problem_l1477_147786

theorem product_sum_squares_problem :
  ∃ x y : ℝ,
    x * y = 120 ∧
    x^2 + y^2 = 289 ∧
    x + y = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_squares_problem_l1477_147786


namespace NUMINAMATH_CALUDE_opposite_quadratics_solution_l1477_147749

theorem opposite_quadratics_solution (x : ℚ) : 
  (2 * x^2 + 1 = -(4 * x^2 - 2 * x - 5)) → (x = 1 ∨ x = -2/3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_quadratics_solution_l1477_147749


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l1477_147729

theorem parametric_to_cartesian (t : ℝ) :
  let x := 3 * t + 6
  let y := 5 * t - 8
  y = (5/3) * x - 18 := by
sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l1477_147729


namespace NUMINAMATH_CALUDE_soccer_league_games_l1477_147765

theorem soccer_league_games (n : ℕ) (regular_games_per_matchup : ℕ) (promotional_games_per_team : ℕ) : 
  n = 20 → 
  regular_games_per_matchup = 3 → 
  promotional_games_per_team = 3 → 
  (n * (n - 1) * regular_games_per_matchup) / 2 + n * promotional_games_per_team = 1200 := by
sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1477_147765


namespace NUMINAMATH_CALUDE_particle_probability_l1477_147741

def probability (x y : ℕ) : ℚ :=
  sorry

theorem particle_probability :
  let start_x : ℕ := 5
  let start_y : ℕ := 5
  probability start_x start_y = 1 / 243 :=
by
  sorry

axiom probability_recursive (x y : ℕ) :
  x > 0 → y > 0 →
  probability x y = (1/3) * probability (x-1) y + 
                    (1/3) * probability x (y-1) + 
                    (1/3) * probability (x-1) (y-1)

axiom probability_boundary_zero (x y : ℕ) :
  (x = 0 ∧ y > 0) ∨ (x > 0 ∧ y = 0) →
  probability x y = 0

axiom probability_origin :
  probability 0 0 = 1

end NUMINAMATH_CALUDE_particle_probability_l1477_147741


namespace NUMINAMATH_CALUDE_n_times_n_plus_one_is_even_l1477_147785

theorem n_times_n_plus_one_is_even (n : ℤ) : 2 ∣ n * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_n_times_n_plus_one_is_even_l1477_147785


namespace NUMINAMATH_CALUDE_min_lcm_x_z_l1477_147706

def problem (x y z : ℕ) : Prop :=
  Nat.lcm x y = 20 ∧ Nat.lcm y z = 28

theorem min_lcm_x_z (x y z : ℕ) (h : problem x y z) :
  Nat.lcm x z ≥ 35 :=
sorry

end NUMINAMATH_CALUDE_min_lcm_x_z_l1477_147706


namespace NUMINAMATH_CALUDE_range_of_f_l1477_147768

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 - 2)^2 + 1

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ f x = y) ↔ y ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1477_147768


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l1477_147789

def product (n : ℕ) : ℕ := (n+2)*(n+4)*(n+6)*(n+8)*(n+10)

theorem largest_divisor_of_product (n : ℕ) (h : Odd n) :
  (∃ (k : ℕ), product n = 15 * k) ∧
  (∀ (m : ℕ), m > 15 → ¬(∀ (n : ℕ), Odd n → ∃ (k : ℕ), product n = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l1477_147789


namespace NUMINAMATH_CALUDE_solution_set_equality_l1477_147770

theorem solution_set_equality : 
  {x : ℝ | x^2 - 2*x ≤ 0} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1477_147770


namespace NUMINAMATH_CALUDE_quadratic_roots_average_l1477_147711

theorem quadratic_roots_average (c : ℝ) 
  (h : ∃ x y : ℝ, x ≠ y ∧ 2 * x^2 - 6 * x + c = 0 ∧ 2 * y^2 - 6 * y + c = 0) :
  ∃ x y : ℝ, x ≠ y ∧ 
    2 * x^2 - 6 * x + c = 0 ∧ 
    2 * y^2 - 6 * y + c = 0 ∧ 
    (x + y) / 2 = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_average_l1477_147711


namespace NUMINAMATH_CALUDE_nancys_payment_is_384_l1477_147712

/-- Nancy's annual payment for her daughter's car insurance -/
def nancys_annual_payment (total_monthly_cost : ℝ) (nancy_share_percent : ℝ) : ℝ :=
  total_monthly_cost * nancy_share_percent * 12

/-- Proof that Nancy's annual payment is $384 -/
theorem nancys_payment_is_384 :
  nancys_annual_payment 80 0.4 = 384 := by
  sorry

end NUMINAMATH_CALUDE_nancys_payment_is_384_l1477_147712
