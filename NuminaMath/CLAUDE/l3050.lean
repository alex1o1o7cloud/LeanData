import Mathlib

namespace charity_race_fundraising_l3050_305056

/-- Proves that 30 students, with 10 raising $20 each and the rest raising $30 each, raise a total of $800 --/
theorem charity_race_fundraising (total_students : Nat) (group1_students : Nat) (group1_amount : Nat) (group2_amount : Nat) :
  total_students = 30 →
  group1_students = 10 →
  group1_amount = 20 →
  group2_amount = 30 →
  group1_students * group1_amount + (total_students - group1_students) * group2_amount = 800 := by
  sorry

#check charity_race_fundraising

end charity_race_fundraising_l3050_305056


namespace game_score_problem_l3050_305068

theorem game_score_problem (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 30 →
  correct_points = 7 →
  incorrect_points = -12 →
  total_score = 77 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_answers * correct_points + (total_questions - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 23 := by
  sorry

end game_score_problem_l3050_305068


namespace optimal_procurement_plan_l3050_305000

/-- Represents a snowflake model type -/
inductive ModelType
| A
| B

/-- Represents the number of pipes needed for a model -/
structure PipeCount where
  long : ℕ
  short : ℕ

/-- Represents the store's inventory -/
structure Inventory where
  long : ℕ
  short : ℕ

/-- Represents a procurement plan -/
structure ProcurementPlan where
  modelA : ℕ
  modelB : ℕ

def pipe_price : ℚ := 1/2

def long_pipe_price : ℚ := 2 * pipe_price

def inventory : Inventory := ⟨267, 2130⟩

def budget : ℚ := 1280

def pipes_per_model (t : ModelType) : PipeCount :=
  match t with
  | ModelType.A => ⟨3, 21⟩
  | ModelType.B => ⟨3, 27⟩

def cost_of_plan (plan : ProcurementPlan) : ℚ :=
  let total_long := plan.modelA * (pipes_per_model ModelType.A).long + 
                    plan.modelB * (pipes_per_model ModelType.B).long
  let total_short := plan.modelA * (pipes_per_model ModelType.A).short + 
                     plan.modelB * (pipes_per_model ModelType.B).short - 
                     (total_long / 3)
  total_long * long_pipe_price + total_short * pipe_price

def is_valid_plan (plan : ProcurementPlan) : Prop :=
  let total_long := plan.modelA * (pipes_per_model ModelType.A).long + 
                    plan.modelB * (pipes_per_model ModelType.B).long
  let total_short := plan.modelA * (pipes_per_model ModelType.A).short + 
                     plan.modelB * (pipes_per_model ModelType.B).short
  total_long ≤ inventory.long ∧ 
  total_short ≤ inventory.short ∧ 
  cost_of_plan plan = budget

theorem optimal_procurement_plan :
  ∀ plan : ProcurementPlan,
    is_valid_plan plan →
    plan.modelA + plan.modelB ≤ 49 ∧
    (plan.modelA + plan.modelB = 49 → plan.modelA = 48 ∧ plan.modelB = 1) :=
sorry

end optimal_procurement_plan_l3050_305000


namespace range_of_f_range_of_m_l3050_305001

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1| - 1
def g (x : ℝ) : ℝ := -|x + 1| - 4

-- Theorem 1: Range of x for which f(x) ≤ 1
theorem range_of_f (x : ℝ) : f x ≤ 1 ↔ x ∈ Set.Icc (-1) 3 := by sorry

-- Theorem 2: Range of m for which f(x) - g(x) ≥ m + 1 holds for all x
theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ∈ Set.Iic 4 := by sorry

end range_of_f_range_of_m_l3050_305001


namespace book_distribution_theorem_l3050_305045

/-- The number of different books to be distributed -/
def num_books : ℕ := 6

/-- The number of students -/
def num_students : ℕ := 6

/-- The number of students who receive books -/
def num_receiving_students : ℕ := num_students - 1

/-- The number of ways to distribute the books -/
def distribution_ways : ℕ := num_students * (num_receiving_students ^ num_books)

theorem book_distribution_theorem : distribution_ways = 93750 := by
  sorry

end book_distribution_theorem_l3050_305045


namespace functional_polynomial_is_constant_l3050_305009

/-- A polynomial satisfying the given functional equation -/
def FunctionalPolynomial (p : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 1) ∧ p (-1) = 2

theorem functional_polynomial_is_constant
    (p : ℝ → ℝ) (hp : FunctionalPolynomial p) :
    ∀ x : ℝ, p x = 2 := by
  sorry

end functional_polynomial_is_constant_l3050_305009


namespace total_coronavirus_cases_l3050_305085

-- Define the number of cases for each state
def new_york_cases : ℕ := 2000
def california_cases : ℕ := new_york_cases / 2
def texas_cases : ℕ := california_cases - 400

-- Theorem to prove
theorem total_coronavirus_cases : 
  new_york_cases + california_cases + texas_cases = 3600 := by
  sorry

end total_coronavirus_cases_l3050_305085


namespace partial_fraction_decomposition_l3050_305019

theorem partial_fraction_decomposition :
  ∃ (a b c : ℤ), 
    (1 : ℚ) / 2015 = a / 5 + b / 13 + c / 31 ∧
    0 ≤ a ∧ a < 5 ∧
    0 ≤ b ∧ b < 13 ∧
    a + b = 14 := by
  sorry

end partial_fraction_decomposition_l3050_305019


namespace felix_lift_problem_l3050_305074

/-- Felix's weight lifting problem -/
theorem felix_lift_problem (felix_weight : ℝ) (felix_brother_weight : ℝ) (felix_brother_lift : ℝ) :
  (felix_brother_weight = 2 * felix_weight) →
  (felix_brother_lift = 3 * felix_brother_weight) →
  (felix_brother_lift = 600) →
  (1.5 * felix_weight = 150) :=
by
  sorry

#check felix_lift_problem

end felix_lift_problem_l3050_305074


namespace pq_length_in_30_60_90_triangle_l3050_305060

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The length of the side opposite to the 30° angle -/
  short_side : ℝ
  /-- The length of the side opposite to the 60° angle -/
  long_side : ℝ
  /-- The hypotenuse is twice the short side -/
  hypotenuse_twice_short : hypotenuse = 2 * short_side
  /-- The long side is √3 times the short side -/
  long_side_sqrt3_short : long_side = Real.sqrt 3 * short_side

/-- Theorem: In a 30-60-90 triangle PQR where PR = 6√3 and angle QPR = 30°, PQ = 6√3 -/
theorem pq_length_in_30_60_90_triangle (t : Triangle30_60_90) 
  (h : t.hypotenuse = 6 * Real.sqrt 3) : t.long_side = 6 * Real.sqrt 3 := by
  sorry

end pq_length_in_30_60_90_triangle_l3050_305060


namespace smallest_divisible_by_15_16_18_l3050_305053

theorem smallest_divisible_by_15_16_18 : 
  ∃ n : ℕ, (n > 0) ∧ 
           (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) ∧ 
           (∀ m : ℕ, (m > 0) ∧ (15 ∣ m) ∧ (16 ∣ m) ∧ (18 ∣ m) → n ≤ m) ∧
           n = 720 := by
  sorry

end smallest_divisible_by_15_16_18_l3050_305053


namespace events_A_B_mutually_exclusive_l3050_305037

/-- Represents the possible outcomes of throwing a fair regular hexahedral die -/
inductive DieOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- Defines event A: "the number is odd" -/
def eventA (outcome : DieOutcome) : Prop :=
  outcome = DieOutcome.one ∨ outcome = DieOutcome.three ∨ outcome = DieOutcome.five

/-- Defines event B: "the number is 4" -/
def eventB (outcome : DieOutcome) : Prop :=
  outcome = DieOutcome.four

/-- Theorem stating that events A and B are mutually exclusive -/
theorem events_A_B_mutually_exclusive :
  ∀ (outcome : DieOutcome), ¬(eventA outcome ∧ eventB outcome) :=
by
  sorry


end events_A_B_mutually_exclusive_l3050_305037


namespace f_always_positive_l3050_305040

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality conditions
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define the function f(x)
def f (t : Triangle) (x : ℝ) : ℝ :=
  t.b^2 * x^2 + (t.b^2 + t.c^2 - t.a^2) * x + t.c^2

-- Theorem statement
theorem f_always_positive (t : Triangle) : ∀ x : ℝ, f t x > 0 := by
  sorry

end f_always_positive_l3050_305040


namespace stock_value_change_l3050_305034

theorem stock_value_change (initial_value : ℝ) (h : initial_value > 0) : 
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.35)
  (day2_value - initial_value) / initial_value * 100 = 1.25 := by
  sorry

end stock_value_change_l3050_305034


namespace circle_diameter_from_viewing_angles_l3050_305094

theorem circle_diameter_from_viewing_angles 
  (r : ℝ) (d α β : ℝ) 
  (h_positive : r > 0 ∧ d > 0)
  (h_angles : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2) :
  2 * r = (d * Real.sin α * Real.sin β) / (Real.sin ((α + β)/2) * Real.cos ((α - β)/2)) :=
by sorry

end circle_diameter_from_viewing_angles_l3050_305094


namespace line_relationships_l3050_305013

-- Define the slopes of the lines
def slope1 : ℚ := 2
def slope2 : ℚ := 3
def slope3 : ℚ := 2
def slope4 : ℚ := 3/2
def slope5 : ℚ := 1/2

-- Define a function to check if two slopes are parallel
def are_parallel (m1 m2 : ℚ) : Prop := m1 = m2

-- Define a function to check if two slopes are perpendicular
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Define the list of all slopes
def slopes : List ℚ := [slope1, slope2, slope3, slope4, slope5]

-- Theorem statement
theorem line_relationships :
  (∃! (i j : Fin 5), i < j ∧ are_parallel (slopes.get i) (slopes.get j)) ∧
  (∀ (i j : Fin 5), i < j → ¬are_perpendicular (slopes.get i) (slopes.get j)) :=
sorry

end line_relationships_l3050_305013


namespace amaya_total_score_l3050_305029

/-- Represents the scores in different subjects -/
structure Scores where
  music : ℕ
  social_studies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total score across all subjects -/
def total_score (s : Scores) : ℕ :=
  s.music + s.social_studies + s.arts + s.maths

/-- Theorem stating the total score given the conditions -/
theorem amaya_total_score :
  ∀ s : Scores,
  s.music = 70 →
  s.social_studies = s.music + 10 →
  s.maths = s.arts - 20 →
  s.maths = (9 * s.arts) / 10 →
  total_score s = 530 := by
  sorry

#check amaya_total_score

end amaya_total_score_l3050_305029


namespace correct_sum_and_digit_change_l3050_305025

theorem correct_sum_and_digit_change : ∃ (d e : ℕ), 
  (d ≤ 9 ∧ e ≤ 9) ∧ 
  (553672 + 637528 = 1511200) ∧ 
  (d + e = 14) ∧
  (953672 + 637528 ≠ 1511200) := by
sorry

end correct_sum_and_digit_change_l3050_305025


namespace range_of_a_l3050_305008

theorem range_of_a (x a : ℝ) :
  (∀ x, (-4 < x - a ∧ x - a < 4) ↔ (1 < x ∧ x < 2)) →
  -2 ≤ a ∧ a ≤ 5 :=
by sorry

end range_of_a_l3050_305008


namespace floor_equation_solution_l3050_305023

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ + 1/3⌋ = ⌊x + 5⌋) ↔ (7/3 ≤ x ∧ x < 3) := by
sorry

end floor_equation_solution_l3050_305023


namespace parallelogram_area_theorem_l3050_305006

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a parallelogram given its vertices -/
def parallelogramArea (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem stating that the area of a parallelogram with specific vertex coordinates is 4ap -/
theorem parallelogram_area_theorem (x a b p : ℝ) :
  let para := Parallelogram.mk
    (Point.mk x p)
    (Point.mk a b)
    (Point.mk x (-p))
    (Point.mk (-a) (-b))
  parallelogramArea para = 4 * a * p := by
  sorry

end parallelogram_area_theorem_l3050_305006


namespace red_balls_count_l3050_305057

/-- Given a jar with white and red balls where the ratio of white to red balls is 3:2,
    and there are 9 white balls, prove that the number of red balls is 6. -/
theorem red_balls_count (white_balls : ℕ) (red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 3 / 2 →
  white_balls = 9 →
  red_balls = 6 := by
sorry

end red_balls_count_l3050_305057


namespace sara_remaining_pears_l3050_305061

-- Define the initial number of pears Sara picked
def initial_pears : ℕ := 35

-- Define the number of pears Sara gave to Dan
def pears_given : ℕ := 28

-- Theorem to prove
theorem sara_remaining_pears :
  initial_pears - pears_given = 7 := by
  sorry

end sara_remaining_pears_l3050_305061


namespace square_area_from_diagonal_l3050_305084

/-- The area of a square with a diagonal of 3.8 meters is 7.22 square meters. -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 3.8) :
  let s := d / Real.sqrt 2
  s ^ 2 = 7.22 := by sorry

end square_area_from_diagonal_l3050_305084


namespace givenEquationIsParabola_l3050_305024

/-- Represents a conic section type -/
inductive ConicType
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines if an equation represents a parabola -/
def isParabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧ 
    ∀ x y : ℝ, f x y ↔ (a * y^2 + b * y + c * x + d = 0 ∨ a * x^2 + b * x + c * y + d = 0)

/-- The given equation -/
def givenEquation (x y : ℝ) : Prop :=
  |x - 3| = Real.sqrt ((y + 4)^2 + x^2)

/-- Theorem stating that the given equation represents a parabola -/
theorem givenEquationIsParabola : isParabola givenEquation := by
  sorry

/-- The conic type of the given equation is a parabola -/
def conicTypeOfGivenEquation : ConicType := ConicType.Parabola

end givenEquationIsParabola_l3050_305024


namespace prime_absolute_value_quadratic_l3050_305031

theorem prime_absolute_value_quadratic (a : ℤ) : 
  Nat.Prime (Int.natAbs (a^2 - 3*a - 6)) ↔ a = -1 ∨ a = 4 := by
sorry

end prime_absolute_value_quadratic_l3050_305031


namespace cos_two_alpha_equals_zero_l3050_305072

theorem cos_two_alpha_equals_zero (α : ℝ) 
  (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : 
  Real.cos (2 * α) = 0 := by
  sorry

end cos_two_alpha_equals_zero_l3050_305072


namespace largest_n_for_trig_inequality_l3050_305007

theorem largest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 2/n)) ∧
  (∀ (m : ℕ), m > 6 → ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m < 2/m) :=
by sorry

end largest_n_for_trig_inequality_l3050_305007


namespace max_m_for_monotonic_f_l3050_305091

/-- Given a function f(x) = x^4 - (1/3)mx^3 + (1/2)x^2 + 1, 
    if f is monotonically increasing on (0,1), 
    then the maximum value of m is 4 -/
theorem max_m_for_monotonic_f (m : ℝ) : 
  let f := fun (x : ℝ) ↦ x^4 - (1/3)*m*x^3 + (1/2)*x^2 + 1
  (∀ x ∈ Set.Ioo 0 1, Monotone f) → m ≤ 4 := by
  sorry

end max_m_for_monotonic_f_l3050_305091


namespace square_minus_product_equals_one_l3050_305086

theorem square_minus_product_equals_one : 2014^2 - 2013 * 2015 = 1 := by
  sorry

end square_minus_product_equals_one_l3050_305086


namespace area_between_concentric_circles_l3050_305077

theorem area_between_concentric_circles 
  (R r : ℝ) 
  (h_positive_R : R > 0) 
  (h_positive_r : r > 0) 
  (h_R_greater_r : R > r) 
  (h_tangent : r^2 + 5^2 = R^2) : 
  π * (R^2 - r^2) = 25 * π := by
sorry

end area_between_concentric_circles_l3050_305077


namespace sum_of_numbers_ge_04_l3050_305080

theorem sum_of_numbers_ge_04 : 
  let numbers := [0.8, 1/2, 0.3]
  let sum_ge_04 := (numbers.filter (λ x => x ≥ 0.4)).sum
  sum_ge_04 = 1.3 := by
sorry

end sum_of_numbers_ge_04_l3050_305080


namespace sixth_root_equation_l3050_305039

theorem sixth_root_equation (x : ℝ) : 
  (x * (x^4)^(1/3))^(1/6) = 2 → x = 2^(18/7) := by
sorry

end sixth_root_equation_l3050_305039


namespace paths_through_point_c_l3050_305075

/-- The number of paths on a grid from (0,0) to (x,y) moving only right or up -/
def gridPaths (x y : ℕ) : ℕ := Nat.choose (x + y) y

/-- The total number of paths from A(0,0) to B(7,6) passing through C(3,2) on a 7x6 grid -/
def totalPaths : ℕ :=
  gridPaths 3 2 * gridPaths 4 3

theorem paths_through_point_c :
  totalPaths = 200 := by sorry

end paths_through_point_c_l3050_305075


namespace correct_operation_l3050_305011

theorem correct_operation (a b : ℝ) : 3*a + 2*b - 2*(a - b) = a + 4*b := by
  sorry

end correct_operation_l3050_305011


namespace unique_p_for_natural_roots_l3050_305078

def cubic_equation (p : ℝ) (x : ℝ) : ℝ :=
  5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1 - 66 * p

theorem unique_p_for_natural_roots :
  ∃! p : ℝ, p = 76 ∧
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  cubic_equation p x = 0 ∧
  cubic_equation p y = 0 ∧
  cubic_equation p z = 0 :=
sorry

end unique_p_for_natural_roots_l3050_305078


namespace sum_of_roots_product_polynomials_l3050_305087

theorem sum_of_roots_product_polynomials :
  let p₁ : Polynomial ℝ := 3 * X^3 - 2 * X^2 + 9 * X - 15
  let p₂ : Polynomial ℝ := 4 * X^3 + 8 * X^2 - 4 * X + 24
  let roots := (p₁.roots.toFinset ∪ p₂.roots.toFinset).toList
  List.sum roots = -4/3 := by
  sorry

end sum_of_roots_product_polynomials_l3050_305087


namespace new_customers_calculation_l3050_305098

theorem new_customers_calculation (initial_customers final_customers : ℕ) 
  (h1 : initial_customers = 3)
  (h2 : final_customers = 8) :
  final_customers - initial_customers = 5 := by
  sorry

end new_customers_calculation_l3050_305098


namespace exponent_calculation_l3050_305032

theorem exponent_calculation : (1 / ((-5^4)^2)) * (-5)^7 = -1/5 := by sorry

end exponent_calculation_l3050_305032


namespace number_puzzle_l3050_305052

theorem number_puzzle : ∃ x : ℤ, (x - 10 = 15) ∧ (x + 5 = 30) := by
  sorry

end number_puzzle_l3050_305052


namespace sphere_ratio_l3050_305088

theorem sphere_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 16 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 64 := by
sorry

end sphere_ratio_l3050_305088


namespace hotel_cost_proof_l3050_305030

theorem hotel_cost_proof (initial_share : ℝ) (final_share : ℝ) : 
  (∃ (total_cost : ℝ),
    (initial_share = total_cost / 4) ∧ 
    (final_share = total_cost / 7) ∧
    (initial_share - 15 = final_share)) →
  ∃ (total_cost : ℝ), total_cost = 140 := by
sorry

end hotel_cost_proof_l3050_305030


namespace isosceles_triangle_perimeter_l3050_305082

/-- An isosceles triangle with two sides measuring 5 and 6 -/
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (is_isosceles : side1 = side2 ∨ side1 = 6 ∨ side2 = 6)
  (has_sides_5_6 : (side1 = 5 ∧ side2 = 6) ∨ (side1 = 6 ∧ side2 = 5))

/-- The perimeter of an isosceles triangle with sides 5 and 6 is either 16 or 17 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) :
  ∃ (p : ℝ), (p = 16 ∨ p = 17) ∧ p = t.side1 + t.side2 + (if t.side1 = t.side2 then 5 else 6) :=
sorry

end isosceles_triangle_perimeter_l3050_305082


namespace black_and_white_cartridge_cost_l3050_305067

/-- The cost of a black-and-white printer cartridge -/
def black_and_white_cost : ℕ := sorry

/-- The cost of a color printer cartridge -/
def color_cost : ℕ := 32

/-- The total cost of printer cartridges -/
def total_cost : ℕ := 123

/-- The number of color cartridges needed -/
def num_color_cartridges : ℕ := 3

/-- The number of black-and-white cartridges needed -/
def num_black_and_white_cartridges : ℕ := 1

theorem black_and_white_cartridge_cost :
  black_and_white_cost = 27 :=
by sorry

end black_and_white_cartridge_cost_l3050_305067


namespace binomial_12_choose_5_l3050_305022

theorem binomial_12_choose_5 : Nat.choose 12 5 = 792 := by
  sorry

end binomial_12_choose_5_l3050_305022


namespace max_value_of_f_l3050_305093

-- Define the function f
def f (x : ℝ) : ℝ := x * (4 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 4 ∧ ∀ x, x ∈ Set.Ioo 0 4 → f x ≤ M :=
by sorry

end max_value_of_f_l3050_305093


namespace special_numbers_l3050_305069

def is_special (n : ℕ+) : Prop :=
  ∃ k : ℕ+, ∀ d : ℕ+, d ∣ n → (d - k : ℤ) ∣ n

theorem special_numbers (n : ℕ+) :
  is_special n ↔ n = 3 ∨ n = 4 ∨ n = 6 ∨ Nat.Prime n.val :=
sorry

end special_numbers_l3050_305069


namespace triangle_side_length_l3050_305046

/-- Given a triangle ABC with specific properties, prove that the length of side b is √7 -/
theorem triangle_side_length (A B C : ℝ) (α l a b c : ℝ) : 
  0 < α → 0 < l → 0 < a → 0 < b → 0 < c →
  B = π / 3 →
  (a * c : ℝ) * Real.cos B = 3 / 2 →
  a + c = 4 →
  b ^ 2 = 7 :=
by sorry

end triangle_side_length_l3050_305046


namespace shoe_pairs_calculation_shoe_pairs_proof_l3050_305028

/-- Given a total number of shoes and the probability of selecting two shoes of the same color
    without replacement, calculate the number of pairs of shoes. -/
theorem shoe_pairs_calculation (total_shoes : ℕ) (probability : ℚ) : ℕ :=
  let pairs := total_shoes / 2
  let calculated_prob := 1 / (total_shoes - 1 : ℚ)
  if total_shoes = 12 ∧ probability = 1/11 ∧ calculated_prob = probability
  then pairs
  else 0

/-- Prove that given 12 shoes in total and a probability of 1/11 for selecting 2 shoes
    of the same color without replacement, the number of pairs of shoes is 6. -/
theorem shoe_pairs_proof :
  shoe_pairs_calculation 12 (1/11) = 6 := by
  sorry

end shoe_pairs_calculation_shoe_pairs_proof_l3050_305028


namespace paul_books_theorem_l3050_305015

/-- The number of books Paul initially had -/
def initial_books : ℕ := 134

/-- The number of books Paul gave to his friend -/
def books_given : ℕ := 39

/-- The number of books Paul sold in the garage sale -/
def books_sold : ℕ := 27

/-- The number of books Paul had left -/
def books_left : ℕ := 68

/-- Theorem stating that the initial number of books equals the sum of books given away, sold, and left -/
theorem paul_books_theorem : initial_books = books_given + books_sold + books_left := by
  sorry

end paul_books_theorem_l3050_305015


namespace memory_sequence_increment_prime_or_one_l3050_305058

/-- Sequence representing the memory cell value after each step -/
def memory_sequence : ℕ → ℕ
  | 0 => 6
  | (n + 1) => memory_sequence n + Nat.gcd (memory_sequence n) (n + 1)

/-- Proposition: The difference between consecutive terms is either 1 or prime -/
theorem memory_sequence_increment_prime_or_one :
  ∀ n : ℕ, (memory_sequence (n + 1) - memory_sequence n = 1) ∨ 
    Nat.Prime (memory_sequence (n + 1) - memory_sequence n) :=
by
  sorry


end memory_sequence_increment_prime_or_one_l3050_305058


namespace sock_order_ratio_l3050_305079

/-- Represents the number of pairs of socks and their prices -/
structure SockOrder where
  grey_pairs : ℕ
  white_pairs : ℕ
  white_price : ℝ

/-- Calculates the total cost of a sock order -/
def total_cost (order : SockOrder) : ℝ :=
  order.grey_pairs * (3 * order.white_price) + order.white_pairs * order.white_price

theorem sock_order_ratio (order : SockOrder) :
  order.grey_pairs = 6 →
  total_cost { grey_pairs := order.white_pairs, white_pairs := order.grey_pairs, white_price := order.white_price } = 1.25 * total_cost order →
  (order.grey_pairs : ℚ) / order.white_pairs = 6 / 10 := by
  sorry

end sock_order_ratio_l3050_305079


namespace reciprocal_of_negative_three_l3050_305038

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by
  sorry

end reciprocal_of_negative_three_l3050_305038


namespace vector_sum_magnitude_l3050_305042

/-- Given two vectors a and b in a plane with an angle of 30° between them,
    |a| = √3, and |b| = 2, prove that |a + 2b| = √31 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  let angle := 30 * π / 180
  (norm a = Real.sqrt 3) →
  (norm b = 2) →
  (a.1 * b.1 + a.2 * b.2 = norm a * norm b * Real.cos angle) →
  norm (a + 2 • b) = Real.sqrt 31 := by
  sorry

end vector_sum_magnitude_l3050_305042


namespace f_3_eq_2488_l3050_305033

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 7x^5 + 12x^4 - 5x^3 - 6x^2 + 3x - 5 -/
def f (x : ℝ) : ℝ := 
  horner_eval [7, 12, -5, -6, 3, -5] x

theorem f_3_eq_2488 : f 3 = 2488 := by
  sorry

end f_3_eq_2488_l3050_305033


namespace range_of_m_l3050_305076

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) ↔ m > 1 := by sorry

end range_of_m_l3050_305076


namespace unique_quadratic_solution_l3050_305005

theorem unique_quadratic_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + 2*(b + 1/b)*x + c = 0) → 
  c = 4 := by sorry

end unique_quadratic_solution_l3050_305005


namespace Only_Statement3_Is_Correct_l3050_305043

-- Define the basic properties of functions
def Monotonic_Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def Odd_Function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def Even_Function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def Symmetric_About_Y_Axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the four statements
def Statement1 : Prop :=
  Monotonic_Increasing (fun x => -1/x)

def Statement2 : Prop :=
  ∀ f : ℝ → ℝ, Odd_Function f → f 0 = 0

def Statement3 : Prop :=
  ∀ f : ℝ → ℝ, Even_Function f → Symmetric_About_Y_Axis f

def Statement4 : Prop :=
  ∀ f : ℝ → ℝ, Odd_Function f → Even_Function f → ∀ x, f x = 0

-- Theorem stating that only Statement3 is correct
theorem Only_Statement3_Is_Correct :
  ¬Statement1 ∧ ¬Statement2 ∧ Statement3 ∧ ¬Statement4 :=
sorry

end Only_Statement3_Is_Correct_l3050_305043


namespace frequency_converges_to_half_l3050_305099

/-- A coin toss experiment -/
structure CoinToss where
  /-- The probability of getting heads in a single toss -/
  probHeads : ℝ
  /-- The coin is fair -/
  isFair : probHeads = 0.5

/-- The frequency of heads after n tosses -/
def frequency (c : CoinToss) (n : ℕ) : ℝ :=
  sorry

/-- The theorem stating that the frequency of heads converges to 0.5 as n approaches infinity -/
theorem frequency_converges_to_half (c : CoinToss) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |frequency c n - 0.5| < ε :=
sorry

end frequency_converges_to_half_l3050_305099


namespace russells_earnings_l3050_305051

/-- Proof of Russell's earnings --/
theorem russells_earnings (vika_earnings breanna_earnings saheed_earnings kayla_earnings russell_earnings : ℕ) : 
  vika_earnings = 84 →
  kayla_earnings = vika_earnings - 30 →
  saheed_earnings = 4 * kayla_earnings →
  breanna_earnings = saheed_earnings + (saheed_earnings / 4) →
  russell_earnings = 2 * (breanna_earnings - kayla_earnings) →
  russell_earnings = 432 := by
  sorry

end russells_earnings_l3050_305051


namespace pillowcase_material_proof_l3050_305071

/-- The amount of material needed for one pillowcase -/
def pillowcase_material : ℝ := 1.25

theorem pillowcase_material_proof :
  let total_material : ℝ := 5000
  let third_bale_ratio : ℝ := 0.22
  let sheet_pillowcase_diff : ℝ := 3.25
  let sheets_sewn : ℕ := 150
  let pillowcases_sewn : ℕ := 240
  ∃ (first_bale second_bale third_bale : ℝ),
    first_bale + second_bale + third_bale = total_material ∧
    3 * first_bale = second_bale ∧
    third_bale = third_bale_ratio * total_material ∧
    sheets_sewn * (pillowcase_material + sheet_pillowcase_diff) + pillowcases_sewn * pillowcase_material = first_bale :=
by sorry

end pillowcase_material_proof_l3050_305071


namespace arithmetic_sequence_common_difference_l3050_305066

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + a 6 = 12) 
  (h_a4 : a 4 = 7) : 
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l3050_305066


namespace magnitude_b_cos_angle_ab_l3050_305027

-- Define the vectors
def a : ℝ × ℝ := (4, 3)
def b : ℝ × ℝ := (-1, 2)

-- Theorem for the magnitude of vector b
theorem magnitude_b : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = Real.sqrt 5 := by sorry

-- Theorem for the cosine of the angle between vectors a and b
theorem cos_angle_ab : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) * Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))) 
  = (2 * Real.sqrt 5) / 25 := by sorry

end magnitude_b_cos_angle_ab_l3050_305027


namespace cafeteria_apples_l3050_305054

theorem cafeteria_apples (initial_apples : ℕ) : 
  (initial_apples - 2 + 23 = 38) → initial_apples = 17 := by
  sorry

end cafeteria_apples_l3050_305054


namespace quadratic_inequality_solution_l3050_305049

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 2

-- State the theorem
theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, -1/3 < x ∧ x < 1 ↔ f a x > 0) → a = -6 := by
  sorry

end quadratic_inequality_solution_l3050_305049


namespace permutation_combination_equality_l3050_305047

theorem permutation_combination_equality (n : ℕ) : 
  (n * (n - 1) * (n - 2) = n * (n - 1) * (n - 2) * (n - 3) / 24) → n = 27 := by
  sorry

end permutation_combination_equality_l3050_305047


namespace circle_op_properties_l3050_305096

-- Define the set A as ordered pairs of real numbers
def A : Type := ℝ × ℝ

-- Define the operation ⊙
def circle_op (α β : A) : A :=
  let (a, b) := α
  let (c, d) := β
  (a * d + b * c, b * d - a * c)

-- Theorem statement
theorem circle_op_properties :
  -- Part 1: Specific calculation
  circle_op (2, 3) (-1, 4) = (5, 14) ∧
  -- Part 2: Commutativity
  (∀ α β : A, circle_op α β = circle_op β α) ∧
  -- Part 3: Identity element
  (∃ I : A, ∀ α : A, circle_op I α = α ∧ circle_op α I = α) ∧
  (∀ I : A, (∀ α : A, circle_op I α = α ∧ circle_op α I = α) → I = (0, 1)) :=
by sorry

end circle_op_properties_l3050_305096


namespace matrix_row_replacement_determinant_l3050_305018

theorem matrix_row_replacement_determinant :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 5; 3, 7]
  let B : Matrix (Fin 2) (Fin 2) ℤ := 
    Matrix.updateRow A 1 (fun j => 2 * A 0 j + A 1 j)
  Matrix.det B = -1 := by
  sorry

end matrix_row_replacement_determinant_l3050_305018


namespace diophantine_equation_solutions_l3050_305089

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+,
  z = Nat.gcd x y →
  x + y^2 + z^3 = x * y * z →
  ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = 6) ∨ (x = 5 ∧ y = 2) ∨ (x = 5 ∧ y = 3)) :=
by sorry

end diophantine_equation_solutions_l3050_305089


namespace chocolate_distribution_l3050_305003

/-- 
Given:
- Ingrid starts with n chocolates
- Jin receives 1/3 of Ingrid's chocolates
- Jin gives 8 chocolates to Brian
- Jin eats half of her remaining chocolates
- Jin ends up with 5 chocolates

Prove: n = 54
-/
theorem chocolate_distribution (n : ℕ) : 
  (n / 3 - 8) / 2 = 5 → n = 54 := by
  sorry

end chocolate_distribution_l3050_305003


namespace custom_operation_value_l3050_305004

/-- Custom operation * for non-zero integers -/
def star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem custom_operation_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 10) (prod_eq : a * b = 24) : 
  star a b = 5 / 12 := by
  sorry

end custom_operation_value_l3050_305004


namespace geometric_series_common_ratio_l3050_305059

theorem geometric_series_common_ratio :
  let a₁ : ℚ := 8 / 10
  let a₂ : ℚ := -6 / 15
  let a₃ : ℚ := 54 / 225
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → (a₁ * r ^ (n - 1) = if n % 2 = 0 then -a₁ * (1 / 2) ^ (n - 1) else a₁ * (1 / 2) ^ (n - 1))) →
  r = -1 / 2 :=
by sorry

end geometric_series_common_ratio_l3050_305059


namespace walking_distance_l3050_305090

/-- 
Given a person who walks at 10 km/hr, if increasing their speed to 16 km/hr 
would allow them to walk 36 km more in the same time, then the actual distance 
traveled is 60 km.
-/
theorem walking_distance (actual_speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) 
  (h1 : actual_speed = 10)
  (h2 : faster_speed = 16)
  (h3 : extra_distance = 36)
  (h4 : (actual_distance / actual_speed) = ((actual_distance + extra_distance) / faster_speed)) :
  actual_distance = 60 :=
by
  sorry

#check walking_distance

end walking_distance_l3050_305090


namespace fred_baseball_cards_l3050_305081

theorem fred_baseball_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 40 → cards_bought = 22 → remaining_cards = initial_cards - cards_bought → 
  remaining_cards = 18 := by
sorry

end fred_baseball_cards_l3050_305081


namespace tetrahedron_volume_specific_l3050_305044

def tetrahedron_volume (AB AC AD BC BD CD : ℝ) : ℝ := sorry

theorem tetrahedron_volume_specific : 
  tetrahedron_volume 2 4 3 (Real.sqrt 17) (Real.sqrt 13) 5 = 6 * Real.sqrt 247 / 64 := by sorry

end tetrahedron_volume_specific_l3050_305044


namespace geometric_area_ratios_l3050_305020

theorem geometric_area_ratios (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let triangle_area := s^2 / 2
  let circle_area := π * (s/2)^2
  let small_square_area := (s/2)^2
  (triangle_area / square_area = 1/2) ∧
  (circle_area / square_area = π/4) ∧
  (small_square_area / square_area = 1/4) :=
by sorry

end geometric_area_ratios_l3050_305020


namespace theater_seats_l3050_305021

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with given properties has 770 seats -/
theorem theater_seats :
  ∀ t : Theater,
    t.first_row_seats = 14 →
    t.seat_increase = 2 →
    t.last_row_seats = 56 →
    total_seats t = 770 := by
  sorry

#eval total_seats { first_row_seats := 14, seat_increase := 2, last_row_seats := 56 }

end theater_seats_l3050_305021


namespace remainder_101_pow_37_mod_100_l3050_305012

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := by
  sorry

end remainder_101_pow_37_mod_100_l3050_305012


namespace probability_of_white_ball_l3050_305035

/-- The probability of drawing a white ball from a bag with red and white balls -/
theorem probability_of_white_ball (red_balls white_balls : ℕ) :
  red_balls = 3 → white_balls = 5 →
  (white_balls : ℚ) / ((red_balls : ℚ) + (white_balls : ℚ)) = 5 / 8 := by
  sorry

end probability_of_white_ball_l3050_305035


namespace min_value_expression_l3050_305050

theorem min_value_expression (a b c : ℤ) (h1 : c > 0) (h2 : a = b + c) :
  (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) ≥ 2 ∧
  ∃ (a b : ℤ), ∃ (c : ℤ), c > 0 ∧ a = b + c ∧
    (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) = 2 :=
by sorry

end min_value_expression_l3050_305050


namespace min_distinct_values_for_given_conditions_l3050_305064

/-- Given a list of positive integers with a unique mode, this function returns the minimum number of distinct values that can occur in the list. -/
def min_distinct_values (list_size : ℕ) (mode_frequency : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of distinct values for the given conditions -/
theorem min_distinct_values_for_given_conditions :
  min_distinct_values 2057 15 = 147 := by
  sorry

end min_distinct_values_for_given_conditions_l3050_305064


namespace maximize_product_l3050_305095

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 50) :
  x^4 * y^3 ≤ (200/7)^4 * (150/7)^3 ∧
  x^4 * y^3 = (200/7)^4 * (150/7)^3 ↔ x = 200/7 ∧ y = 150/7 := by
  sorry

end maximize_product_l3050_305095


namespace space_creature_perimeter_calc_l3050_305055

/-- The perimeter of a space creature, which is a sector of a circle --/
def space_creature_perimeter (r : ℝ) (central_angle : ℝ) : ℝ :=
  r * central_angle + 2 * r

/-- Theorem: The perimeter of the space creature with radius 2 cm and central angle 270° is 3π + 4 cm --/
theorem space_creature_perimeter_calc :
  space_creature_perimeter 2 (3 * π / 2) = 3 * π + 4 := by
  sorry

#check space_creature_perimeter_calc

end space_creature_perimeter_calc_l3050_305055


namespace indeterminate_product_sum_l3050_305070

theorem indeterminate_product_sum (A B : ℝ) 
  (hA : 0 < A ∧ A < 1) (hB : 0 < B ∧ B < 1) : 
  ∃ (x y z : ℝ), x < 1 ∧ y = 1 ∧ z > 1 ∧ 
  (A * B + 0.1 = x ∨ A * B + 0.1 = y ∨ A * B + 0.1 = z) :=
sorry

end indeterminate_product_sum_l3050_305070


namespace set_equals_interval_l3050_305041

-- Define the set S as {x | x > 0 and x ≠ 2}
def S : Set ℝ := {x : ℝ | x > 0 ∧ x ≠ 2}

-- Define the interval representation
def intervalRep : Set ℝ := Set.Ioo 0 2 ∪ Set.Ioi 2

-- Theorem stating the equivalence of the set and the interval representation
theorem set_equals_interval : S = intervalRep := by sorry

end set_equals_interval_l3050_305041


namespace right_triangle_legs_l3050_305097

theorem right_triangle_legs (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  ((a = 16 ∧ b = 63) ∨ (a = 63 ∧ b = 16)) →  -- Possible leg lengths
  ∃ (x y : ℕ), x^2 + y^2 = 65^2 ∧ (x = 16 ∧ y = 63) := by
  sorry

end right_triangle_legs_l3050_305097


namespace sine_cosine_inequality_l3050_305065

theorem sine_cosine_inequality (α : Real) (h1 : 0 < α) (h2 : α < π) :
  2 * Real.sin (2 * α) ≤ Real.cos (α / 2) ∧
  (2 * Real.sin (2 * α) = Real.cos (α / 2) ↔ α = π / 3) := by
  sorry

end sine_cosine_inequality_l3050_305065


namespace ball_box_probability_l3050_305048

/-- The number of ways to place 5 balls in 4 boxes with no box left empty -/
def total_placements : ℕ := 240

/-- The number of ways to place 5 balls in 4 boxes with no box left empty and no ball in a box with the same label -/
def valid_placements : ℕ := 84

/-- The probability of placing 5 balls in 4 boxes with no box left empty and no ball in a box with the same label -/
def probability : ℚ := valid_placements / total_placements

theorem ball_box_probability : probability = 7 / 20 := by
  sorry

end ball_box_probability_l3050_305048


namespace water_intake_increase_l3050_305002

theorem water_intake_increase (current : ℕ) (recommended : ℕ) : 
  current = 15 → recommended = 21 → 
  (((recommended - current) : ℚ) / current) * 100 = 40 := by
  sorry

end water_intake_increase_l3050_305002


namespace sqrt_equation_solution_l3050_305010

theorem sqrt_equation_solution (a b c : ℝ) 
  (h1 : Real.sqrt a = Real.sqrt b + Real.sqrt c)
  (h2 : b = 52 - 30 * Real.sqrt 3)
  (h3 : c = a - 2) : 
  a = 27 := by
sorry

end sqrt_equation_solution_l3050_305010


namespace x3y2z3_coefficient_in_x_plus_y_plus_z_to_8_l3050_305016

def multinomial_coefficient (n : ℕ) (a b c : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial a * Nat.factorial b * Nat.factorial c)

theorem x3y2z3_coefficient_in_x_plus_y_plus_z_to_8 :
  multinomial_coefficient 8 3 2 3 = 560 := by
  sorry

end x3y2z3_coefficient_in_x_plus_y_plus_z_to_8_l3050_305016


namespace password_length_l3050_305062

-- Define the structure of the password
structure PasswordStructure where
  lowercase_letters : Nat
  uppercase_and_numbers : Nat
  digits : Nat
  symbols : Nat

-- Define Pat's password structure
def pats_password : PasswordStructure :=
  { lowercase_letters := 12
  , uppercase_and_numbers := 6
  , digits := 4
  , symbols := 2 }

-- Theorem to prove the total number of characters
theorem password_length :
  (pats_password.lowercase_letters +
   pats_password.uppercase_and_numbers +
   pats_password.digits +
   pats_password.symbols) = 24 := by
  sorry

end password_length_l3050_305062


namespace abigail_report_time_l3050_305083

/-- Given a report length, typing speed, and words already written, 
    calculate the time required to finish the report. -/
def time_to_finish_report (total_words : ℕ) (words_per_half_hour : ℕ) (words_written : ℕ) : ℕ :=
  let words_remaining := total_words - words_written
  let minutes_per_word := 30 / words_per_half_hour
  words_remaining * minutes_per_word

/-- Proof that for the given conditions, the time to finish the report is 80 minutes. -/
theorem abigail_report_time : time_to_finish_report 1000 300 200 = 80 := by
  sorry

end abigail_report_time_l3050_305083


namespace stamp_collection_value_l3050_305063

theorem stamp_collection_value (partial_value : ℚ) (partial_fraction : ℚ) (total_value : ℚ) : 
  partial_fraction = 4/7 ∧ partial_value = 28 → total_value = 49 :=
by sorry

end stamp_collection_value_l3050_305063


namespace inequality_system_solution_set_l3050_305017

def inequality_system (x : ℝ) : Prop :=
  x^2 - 2*x - 3 > 0 ∧ -x^2 - 3*x + 4 ≥ 0

theorem inequality_system_solution_set :
  {x : ℝ | inequality_system x} = {x : ℝ | -4 ≤ x ∧ x < -1} := by sorry

end inequality_system_solution_set_l3050_305017


namespace max_value_expression_max_value_attained_l3050_305073

theorem max_value_expression (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a^2 + b^2 + c^2 = 1) : 
  3 * a * b * Real.sqrt 3 + 9 * b * c ≤ 3 :=
by sorry

theorem max_value_attained (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a^2 + b^2 + c^2 = 1 ∧ 
  3 * a * b * Real.sqrt 3 + 9 * b * c > 3 - ε :=
by sorry

end max_value_expression_max_value_attained_l3050_305073


namespace quadratic_equation_roots_range_l3050_305092

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ y > 0 ∧ 2 * x^2 - (m + 1) * x + m = 0 ∧ 2 * y^2 - (m + 1) * y + m = 0) 
  ↔ 
  (0 < m ∧ m < 3 - 2 * Real.sqrt 2) ∨ (m > 3 + 2 * Real.sqrt 2) :=
sorry

end quadratic_equation_roots_range_l3050_305092


namespace no_real_sqrt_negative_number_l3050_305014

theorem no_real_sqrt_negative_number (x : ℝ) :
  x = -2.5 ∨ x = 0 ∨ x = 2.1 ∨ x = 6 →
  (∃ y : ℝ, y ^ 2 = x) ↔ x ≠ -2.5 :=
by sorry

end no_real_sqrt_negative_number_l3050_305014


namespace ellipse_m_range_l3050_305026

/-- Represents an ellipse with the given equation and foci on the y-axis -/
structure Ellipse where
  m : ℝ
  eq : ∀ (x y : ℝ), x^2 / (25 - m) + y^2 / (m + 9) = 1
  foci_on_y_axis : True  -- This is a placeholder for the foci condition

/-- The range of valid m values for the given ellipse -/
theorem ellipse_m_range (e : Ellipse) : 8 < e.m ∧ e.m < 25 := by
  sorry

#check ellipse_m_range

end ellipse_m_range_l3050_305026


namespace pages_left_to_read_l3050_305036

/-- Given a book with a total number of pages and a number of pages already read,
    calculate the number of pages left to read. -/
theorem pages_left_to_read (total_pages pages_read : ℕ) 
    (h1 : total_pages = 563)
    (h2 : pages_read = 147) :
    total_pages - pages_read = 416 := by
  sorry

end pages_left_to_read_l3050_305036
