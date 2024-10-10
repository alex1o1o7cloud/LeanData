import Mathlib

namespace sqrt_50_minus_sqrt_48_approx_0_14_l317_31710

theorem sqrt_50_minus_sqrt_48_approx_0_14 (ε : ℝ) (h : ε > 0) :
  ∃ δ > 0, |Real.sqrt 50 - Real.sqrt 48 - 0.14| < δ ∧ δ < ε :=
by sorry

end sqrt_50_minus_sqrt_48_approx_0_14_l317_31710


namespace base_7_addition_problem_l317_31762

/-- Convert a base 7 number to base 10 -/
def to_base_10 (a b c : ℕ) : ℕ := a * 7^2 + b * 7 + c

/-- Convert a base 10 number to base 7 -/
def to_base_7 (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 49
  let remainder := n % 49
  let tens := remainder / 7
  let ones := remainder % 7
  (hundreds, tens, ones)

theorem base_7_addition_problem (X Y : ℕ) :
  (to_base_7 (to_base_10 5 X Y + to_base_10 0 5 2) = (6, 4, X)) →
  X + Y = 10 := by
  sorry

end base_7_addition_problem_l317_31762


namespace angle_measure_l317_31739

theorem angle_measure (x : ℝ) : 
  (90 - x = 2/3 * (180 - x) - 40) → x = 30 := by
  sorry

end angle_measure_l317_31739


namespace some_number_value_l317_31745

theorem some_number_value (x y : ℝ) 
  (h1 : x / y = 3 / 2)
  (h2 : (7 * x + y) / (x - y) = 23) :
  y = 1 := by
sorry

end some_number_value_l317_31745


namespace plane_equation_l317_31795

/-- The plane passing through points (0,3,-1), (4,7,1), and (2,5,0) has the equation y - 2z - 5 = 0 -/
theorem plane_equation (p q r : ℝ × ℝ × ℝ) : 
  p = (0, 3, -1) → q = (4, 7, 1) → r = (2, 5, 0) →
  ∃ (A B C D : ℤ), 
    (A > 0) ∧ 
    (Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) ∧
    (∀ (x y z : ℝ), A * x + B * y + C * z + D = 0 ↔ y - 2 * z - 5 = 0) :=
by sorry

end plane_equation_l317_31795


namespace matt_current_age_l317_31767

/-- Matt's current age -/
def matt_age : ℕ := sorry

/-- Kaylee's current age -/
def kaylee_age : ℕ := 8

theorem matt_current_age : matt_age = 5 := by
  have h1 : kaylee_age + 7 = 3 * matt_age := sorry
  sorry

end matt_current_age_l317_31767


namespace race_remaining_distance_l317_31786

/-- The remaining distance in a race with specific lead changes -/
def remaining_distance (total_length initial_even alex_lead1 max_lead alex_lead2 : ℕ) : ℕ :=
  total_length - (initial_even + alex_lead1 + max_lead + alex_lead2)

/-- Theorem stating the remaining distance in the specific race scenario -/
theorem race_remaining_distance :
  remaining_distance 5000 200 300 170 440 = 3890 := by
  sorry

end race_remaining_distance_l317_31786


namespace cat_and_mouse_positions_after_347_moves_l317_31778

/-- Represents the positions around a pentagon -/
inductive PentagonPosition
| Top
| RightUpper
| RightLower
| LeftLower
| LeftUpper

/-- Represents the positions for the mouse, including edges -/
inductive MousePosition
| TopLeftEdge
| LeftUpperVertex
| LeftMiddleEdge
| LeftLowerVertex
| BottomEdge
| RightLowerVertex
| RightMiddleEdge
| RightUpperVertex
| TopRightEdge
| TopVertex

/-- Function to determine the cat's position after a given number of moves -/
def catPosition (moves : ℕ) : PentagonPosition :=
  match moves % 5 with
  | 0 => PentagonPosition.LeftUpper
  | 1 => PentagonPosition.Top
  | 2 => PentagonPosition.RightUpper
  | 3 => PentagonPosition.RightLower
  | _ => PentagonPosition.LeftLower

/-- Function to determine the mouse's position after a given number of moves -/
def mousePosition (moves : ℕ) : MousePosition :=
  match moves % 10 with
  | 0 => MousePosition.TopVertex
  | 1 => MousePosition.TopLeftEdge
  | 2 => MousePosition.LeftUpperVertex
  | 3 => MousePosition.LeftMiddleEdge
  | 4 => MousePosition.LeftLowerVertex
  | 5 => MousePosition.BottomEdge
  | 6 => MousePosition.RightLowerVertex
  | 7 => MousePosition.RightMiddleEdge
  | 8 => MousePosition.RightUpperVertex
  | _ => MousePosition.TopRightEdge

theorem cat_and_mouse_positions_after_347_moves :
  (catPosition 347 = PentagonPosition.RightUpper) ∧
  (mousePosition 347 = MousePosition.RightMiddleEdge) := by
  sorry


end cat_and_mouse_positions_after_347_moves_l317_31778


namespace inequality_holds_iff_one_l317_31765

def is_valid (x : ℕ) : Prop := x > 0 ∧ x < 100

theorem inequality_holds_iff_one (x : ℕ) (h : is_valid x) :
  (2^x : ℚ) / x.factorial > x^2 ↔ x = 1 := by
  sorry

end inequality_holds_iff_one_l317_31765


namespace fourth_degree_polynomial_property_l317_31742

/-- A fourth-degree polynomial with real coefficients -/
def FourthDegreePolynomial : Type := ℝ → ℝ

/-- The property that |f(-2)| = |f(0)| = |f(1)| = |f(3)| = |f(4)| = 16 -/
def HasSpecifiedValues (f : FourthDegreePolynomial) : Prop :=
  |f (-2)| = 16 ∧ |f 0| = 16 ∧ |f 1| = 16 ∧ |f 3| = 16 ∧ |f 4| = 16

/-- The main theorem -/
theorem fourth_degree_polynomial_property (f : FourthDegreePolynomial) 
  (h : HasSpecifiedValues f) : |f 5| = 208 := by
  sorry


end fourth_degree_polynomial_property_l317_31742


namespace line_circle_relationship_l317_31779

theorem line_circle_relationship (m : ℝ) (h : m > 0) :
  let line := {(x, y) : ℝ × ℝ | Real.sqrt 2 * (x + y) + 1 + m = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = m}
  (∃ p, p ∈ line ∩ circle ∧ (∀ q ∈ line ∩ circle, q = p)) ∨
  (line ∩ circle = ∅) :=
by sorry

end line_circle_relationship_l317_31779


namespace largest_number_l317_31738

-- Define the base conversion function
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the numbers in their respective bases
def A : List Nat := [1, 0, 1, 1, 1, 1]
def B : List Nat := [1, 2, 1, 0]
def C : List Nat := [1, 1, 2]
def D : List Nat := [6, 9]

-- Theorem statement
theorem largest_number :
  to_decimal D 12 > to_decimal A 2 ∧
  to_decimal D 12 > to_decimal B 3 ∧
  to_decimal D 12 > to_decimal C 8 :=
by sorry

end largest_number_l317_31738


namespace solve_fish_problem_l317_31776

def fish_problem (initial_fish : ℕ) (yearly_increase : ℕ) (years : ℕ) (final_fish : ℕ) : ℕ → Prop :=
  λ yearly_deaths : ℕ =>
    initial_fish + years * yearly_increase - years * yearly_deaths = final_fish

theorem solve_fish_problem :
  ∃ yearly_deaths : ℕ, fish_problem 2 2 5 7 yearly_deaths :=
by
  sorry

end solve_fish_problem_l317_31776


namespace optimal_tax_theorem_l317_31713

/-- Market model with linear demand and supply functions, and a per-unit tax --/
structure MarketModel where
  -- Demand function: Qd = a - bP
  a : ℝ
  b : ℝ
  -- Supply function: Qs = cP + d
  c : ℝ
  d : ℝ
  -- Elasticity ratio at equilibrium
  elasticity_ratio : ℝ
  -- Tax amount
  tax : ℝ
  -- Producer price after tax
  producer_price : ℝ

/-- Finds the optimal tax rate and resulting revenue for a given market model --/
def optimal_tax_and_revenue (model : MarketModel) : ℝ × ℝ :=
  sorry

/-- The main theorem stating the optimal tax and revenue for the given market conditions --/
theorem optimal_tax_theorem (model : MarketModel) :
  model.a = 688 ∧
  model.b = 4 ∧
  model.elasticity_ratio = 1.5 ∧
  model.tax = 90 ∧
  model.producer_price = 64 →
  optimal_tax_and_revenue model = (54, 10800) :=
sorry

end optimal_tax_theorem_l317_31713


namespace rectangle_measurement_error_l317_31715

theorem rectangle_measurement_error (L W : ℝ) (L_excess W_deficit : ℝ) 
  (h1 : L_excess = 1.20)  -- 20% excess on first side
  (h2 : W_deficit > 0)    -- deficit percentage is positive
  (h3 : W_deficit < 1)    -- deficit percentage is less than 100%
  (h4 : L_excess * (1 - W_deficit) = 1.08)  -- 8% error in area
  : W_deficit = 0.10 :=   -- 10% deficit on second side
by sorry

end rectangle_measurement_error_l317_31715


namespace negation_of_universal_proposition_l317_31716

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 1 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l317_31716


namespace journey_length_l317_31726

theorem journey_length :
  ∀ (total : ℚ),
  (1 / 4 : ℚ) * total +        -- First part (dirt road)
  30 +                         -- Second part (highway)
  (1 / 7 : ℚ) * total =        -- Third part (city street)
  total →                      -- Sum of all parts equals total
  total = 840 / 17 := by
sorry

end journey_length_l317_31726


namespace quadrant_I_solution_range_l317_31737

theorem quadrant_I_solution_range (c : ℝ) :
  (∃ x y : ℝ, x - y = 5 ∧ 2 * c * x + y = 8 ∧ x > 0 ∧ y > 0) ↔ -1/2 < c ∧ c < 4/5 := by
  sorry

end quadrant_I_solution_range_l317_31737


namespace admission_price_is_12_l317_31723

/-- The admission price for the aqua park. -/
def admission_price : ℝ := sorry

/-- The price of the tour. -/
def tour_price : ℝ := 6

/-- The number of people in the first group (who take the tour). -/
def group1_size : ℕ := 10

/-- The number of people in the second group (who only pay admission). -/
def group2_size : ℕ := 5

/-- The total earnings of the aqua park. -/
def total_earnings : ℝ := 240

/-- Theorem stating that the admission price is $12 given the conditions. -/
theorem admission_price_is_12 :
  (group1_size : ℝ) * (admission_price + tour_price) + (group2_size : ℝ) * admission_price = total_earnings →
  admission_price = 12 := by
  sorry

end admission_price_is_12_l317_31723


namespace estimate_red_balls_l317_31780

theorem estimate_red_balls (black_balls : ℕ) (total_draws : ℕ) (black_draws : ℕ) : 
  black_balls = 4 → 
  total_draws = 100 → 
  black_draws = 40 → 
  ∃ red_balls : ℕ, (black_balls : ℚ) / (black_balls + red_balls : ℚ) = 2 / 5 ∧ red_balls = 6 :=
by sorry

end estimate_red_balls_l317_31780


namespace ping_pong_sum_of_products_l317_31732

/-- The sum of products for n ping pong balls -/
def sum_of_products (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The number of ping pong balls -/
def num_balls : ℕ := 10

theorem ping_pong_sum_of_products :
  sum_of_products num_balls = 45 := by
  sorry


end ping_pong_sum_of_products_l317_31732


namespace machine_working_time_l317_31727

theorem machine_working_time 
  (total_shirts : ℕ) 
  (production_rate : ℕ) 
  (num_malfunctions : ℕ) 
  (malfunction_fix_time : ℕ) 
  (h1 : total_shirts = 360)
  (h2 : production_rate = 4)
  (h3 : num_malfunctions = 2)
  (h4 : malfunction_fix_time = 5) :
  (total_shirts / production_rate) + (num_malfunctions * malfunction_fix_time) = 100 := by
  sorry

end machine_working_time_l317_31727


namespace diamond_set_eq_three_lines_l317_31748

/-- Define the ⋄ operation -/
def diamond (a b : ℝ) : ℝ := a^2 * b - a * b^2

/-- The set of points (x, y) where x ⋄ y = y ⋄ x -/
def diamond_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and x = y -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2}

theorem diamond_set_eq_three_lines : diamond_set = three_lines := by
  sorry

end diamond_set_eq_three_lines_l317_31748


namespace triangle_area_is_11_over_2_l317_31799

-- Define the vertices of the triangle
def D : ℝ × ℝ := (2, -3)
def E : ℝ × ℝ := (0, 4)
def F : ℝ × ℝ := (3, -1)

-- Define the function to calculate the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_is_11_over_2 : triangleArea D E F = 11 / 2 := by sorry

end triangle_area_is_11_over_2_l317_31799


namespace license_plate_count_l317_31712

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letter positions in the license plate -/
def num_letter_positions : ℕ := 3

/-- The number of digit positions in the license plate -/
def num_digit_positions : ℕ := 3

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_letters ^ num_letter_positions * num_digits ^ num_digit_positions

theorem license_plate_count : total_license_plates = 17576000 := by
  sorry

end license_plate_count_l317_31712


namespace same_color_marble_probability_same_color_marble_probability_value_l317_31743

/-- The probability of drawing three marbles of the same color from a bag containing
    5 red marbles, 7 white marbles, and 4 green marbles, without replacement. -/
theorem same_color_marble_probability : ℚ :=
  let total_marbles : ℕ := 5 + 7 + 4
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let green_marbles : ℕ := 4
  let prob_all_red : ℚ := (red_marbles * (red_marbles - 1) * (red_marbles - 2)) /
    (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  let prob_all_white : ℚ := (white_marbles * (white_marbles - 1) * (white_marbles - 2)) /
    (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  let prob_all_green : ℚ := (green_marbles * (green_marbles - 1) * (green_marbles - 2)) /
    (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  prob_all_red + prob_all_white + prob_all_green

theorem same_color_marble_probability_value :
  same_color_marble_probability = 43 / 280 := by
  sorry

end same_color_marble_probability_same_color_marble_probability_value_l317_31743


namespace nabla_squared_l317_31761

theorem nabla_squared (odot nabla : ℕ) : 
  odot < 20 ∧ nabla < 20 ∧ 
  odot ≠ nabla ∧ 
  odot > 0 ∧ nabla > 0 ∧
  nabla * nabla * odot = nabla → 
  nabla * nabla = 64 := by
  sorry

end nabla_squared_l317_31761


namespace f_max_value_l317_31730

open Real

-- Define the function f
def f (x : ℝ) := (3 + 2*x)^3 * (4 - x)^4

-- Define the interval
def I : Set ℝ := {x | -3/2 < x ∧ x < 4}

-- State the theorem
theorem f_max_value :
  ∃ (x_max : ℝ), x_max ∈ I ∧
  f x_max = 432 * (11/7)^7 ∧
  x_max = 6/7 ∧
  ∀ (x : ℝ), x ∈ I → f x ≤ f x_max :=
sorry

end f_max_value_l317_31730


namespace gcd_divisibility_and_multiple_l317_31725

theorem gcd_divisibility_and_multiple (a b n : ℕ) (h : a ≠ 0) :
  let d := Nat.gcd a b
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧
  ∀ c : ℕ, c > 0 → Nat.gcd (a * c) (b * c) = c * Nat.gcd a b :=
by sorry

end gcd_divisibility_and_multiple_l317_31725


namespace cuboid_diagonal_l317_31763

theorem cuboid_diagonal (a b : ℤ) :
  (∃ c : ℕ+, ∃ d : ℕ+, d * d = a * a + b * b + c * c) ↔ 2 ∣ (a * b) := by
  sorry

end cuboid_diagonal_l317_31763


namespace R_value_for_S_12_l317_31787

theorem R_value_for_S_12 (g : ℝ) (R S : ℝ → ℝ) :
  (∀ x, R x = g * S x - 3) →
  R 5 = 17 →
  S 12 = 12 →
  R 12 = 45 := by
sorry

end R_value_for_S_12_l317_31787


namespace circle_unique_dual_symmetry_l317_31753

-- Define the shapes
inductive Shape
  | Parallelogram
  | Circle
  | EquilateralTriangle
  | RegularPentagon

-- Define symmetry properties
def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Circle => true
  | Shape.EquilateralTriangle => true
  | Shape.RegularPentagon => true
  | _ => false

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => true
  | Shape.Circle => true
  | _ => false

-- Theorem statement
theorem circle_unique_dual_symmetry :
  ∀ s : Shape, (isAxiallySymmetric s ∧ isCentrallySymmetric s) ↔ s = Shape.Circle :=
by sorry

end circle_unique_dual_symmetry_l317_31753


namespace functional_equation_properties_l317_31733

/-- A function satisfying the given properties -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x, f x > 0) ∧ 
  (∀ a b, f a * f b = f (a + b))

/-- Main theorem stating the properties of f -/
theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (f 0 = 1) ∧
  (∀ a, f (-a) = 1 / f a) ∧
  (∀ a, f a = (f (3 * a)) ^ (1/3)) :=
by sorry

end functional_equation_properties_l317_31733


namespace marble_problem_l317_31788

theorem marble_problem (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 30 →
  red = 9 →
  prob_red_or_white = 5 / 6 →
  ∃ (blue white : ℕ), blue + red + white = total ∧ 
                       (red + white : ℚ) / total = prob_red_or_white ∧
                       blue = 5 := by
  sorry

end marble_problem_l317_31788


namespace only_B_on_x_axis_l317_31711

def point_A : ℝ × ℝ := (-2, -3)
def point_B : ℝ × ℝ := (-3, 0)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (0, 3)

def is_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

theorem only_B_on_x_axis :
  is_on_x_axis point_B ∧
  ¬is_on_x_axis point_A ∧
  ¬is_on_x_axis point_C ∧
  ¬is_on_x_axis point_D :=
by sorry

end only_B_on_x_axis_l317_31711


namespace cos_36_minus_cos_72_eq_half_l317_31760

theorem cos_36_minus_cos_72_eq_half : Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1 / 2 := by
  sorry

end cos_36_minus_cos_72_eq_half_l317_31760


namespace volume_of_specific_pyramid_l317_31741

/-- Regular quadrilateral pyramid with given properties -/
structure RegularQuadPyramid where
  -- Point P is on the height VO
  p_on_height : Bool
  -- P is equidistant from base and apex
  p_midpoint : Bool
  -- Distance from P to any side face
  dist_p_to_side : ℝ
  -- Distance from P to base
  dist_p_to_base : ℝ

/-- Volume of a regular quadrilateral pyramid -/
def volume (pyramid : RegularQuadPyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific pyramid -/
theorem volume_of_specific_pyramid :
  ∀ (pyramid : RegularQuadPyramid),
    pyramid.p_on_height ∧
    pyramid.p_midpoint ∧
    pyramid.dist_p_to_side = 3 ∧
    pyramid.dist_p_to_base = 5 →
    volume pyramid = 750 :=
by
  sorry

end volume_of_specific_pyramid_l317_31741


namespace panda_arrangement_count_l317_31759

/-- Represents the number of pandas -/
def num_pandas : ℕ := 9

/-- Represents the number of shortest pandas that must be at the ends -/
def num_shortest : ℕ := 3

/-- Calculates the number of ways to arrange the pandas -/
def panda_arrangements : ℕ :=
  2 * (num_pandas - num_shortest).factorial

/-- Theorem stating that the number of panda arrangements is 1440 -/
theorem panda_arrangement_count :
  panda_arrangements = 1440 := by
  sorry

end panda_arrangement_count_l317_31759


namespace f_has_unique_zero_in_interval_l317_31754

noncomputable def f (x : ℝ) := x + 3^(x + 2)

theorem f_has_unique_zero_in_interval :
  ∃! x, x ∈ Set.Ioo (-2 : ℝ) (-1) ∧ f x = 0 := by
  sorry

end f_has_unique_zero_in_interval_l317_31754


namespace min_value_complex_ratio_l317_31749

theorem min_value_complex_ratio (z : ℂ) (h : z.re ≠ 0) :
  ∃ (min : ℝ), min = -8 ∧ 
  (∀ (w : ℂ), w.re ≠ 0 → (w.re^4)⁻¹ * (w^4).re ≥ min) ∧
  (∃ (w : ℂ), w.re ≠ 0 ∧ (w.re^4)⁻¹ * (w^4).re = min) :=
sorry

end min_value_complex_ratio_l317_31749


namespace spinner_probability_l317_31703

def spinner_A : Finset ℕ := {1, 2, 3}
def spinner_B : Finset ℕ := {2, 3, 4}

def is_multiple_of_four (n : ℕ) : Bool :=
  n % 4 = 0

def total_outcomes : ℕ :=
  (spinner_A.card) * (spinner_B.card)

def favorable_outcomes : ℕ :=
  (spinner_A.card) * (spinner_B.filter (λ b => is_multiple_of_four (1 + b))).card +
  (spinner_A.card) * (spinner_B.filter (λ b => is_multiple_of_four (2 + b))).card +
  (spinner_A.card) * (spinner_B.filter (λ b => is_multiple_of_four (3 + b))).card

theorem spinner_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 :=
sorry

end spinner_probability_l317_31703


namespace triangle_abc_problem_l317_31717

theorem triangle_abc_problem (a b c : ℝ) (A B C : ℝ) :
  A = π / 6 →
  (1 + Real.sqrt 3) * c = 2 * b →
  b * a * Real.cos C = 1 + Real.sqrt 3 →
  C = π / 4 ∧ a = Real.sqrt 2 ∧ b = 1 + Real.sqrt 3 ∧ c = 2 :=
by sorry

end triangle_abc_problem_l317_31717


namespace least_subtrahend_for_divisibility_l317_31736

theorem least_subtrahend_for_divisibility (n m : ℕ) : 
  ∃ (x : ℕ), x = n % m ∧ 
  (∀ (y : ℕ), (n - y) % m = 0 → y ≥ x) ∧
  (n - x) % m = 0 :=
sorry

#check least_subtrahend_for_divisibility 13602 87

end least_subtrahend_for_divisibility_l317_31736


namespace equation_solution_l317_31718

theorem equation_solution : ∃ x : ℝ, 
  (0.5^3 - x^3 / 0.5^2 + 0.05 + 0.1^2 = 0.4) ∧ 
  (abs (x + 0.378) < 0.001) := by
  sorry

end equation_solution_l317_31718


namespace article_cost_l317_31747

/-- The cost of an article given specific selling prices and gain percentages -/
theorem article_cost : ∃ (cost : ℝ), 
  (895 - cost) = 1.075 * (785 - cost) ∧ 
  cost > 0 ∧ 
  cost < 785 := by
sorry

end article_cost_l317_31747


namespace radical_conjugate_sum_product_l317_31751

theorem radical_conjugate_sum_product (c d : ℝ) : 
  (c + 2 * Real.sqrt d) + (c - 2 * Real.sqrt d) = 6 ∧ 
  (c + 2 * Real.sqrt d) * (c - 2 * Real.sqrt d) = 4 → 
  c + d = 17/4 := by
sorry

end radical_conjugate_sum_product_l317_31751


namespace probability_one_black_one_white_l317_31757

def total_balls : ℕ := 6 + 2
def black_balls : ℕ := 6
def white_balls : ℕ := 2

theorem probability_one_black_one_white :
  let total_ways := Nat.choose total_balls 2
  let favorable_ways := black_balls * white_balls
  (favorable_ways : ℚ) / total_ways = 3 / 7 := by
    sorry

end probability_one_black_one_white_l317_31757


namespace max_expenditure_l317_31746

def linear_regression (x : ℝ) (b a e : ℝ) : ℝ := b * x + a + e

theorem max_expenditure (x : ℝ) (e : ℝ) 
  (h1 : x = 10) 
  (h2 : |e| ≤ 0.5) : 
  linear_regression x 0.8 2 e ≤ 10.5 := by
  sorry

end max_expenditure_l317_31746


namespace solve_equation_l317_31764

theorem solve_equation : ∃ x : ℝ, 2 * x = (26 - x) + 19 ∧ x = 15 := by
  sorry

end solve_equation_l317_31764


namespace min_altitude_inequality_l317_31793

/-- The minimum altitude of a triangle, or zero if the points are collinear -/
noncomputable def min_altitude (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The triangle inequality for minimum altitudes -/
theorem min_altitude_inequality (A B C X : ℝ × ℝ) :
  min_altitude A B C ≤ min_altitude A B X + min_altitude A X C + min_altitude X B C := by
  sorry

end min_altitude_inequality_l317_31793


namespace unique_fixed_point_l317_31774

-- Define the plane
variable (Plane : Type)

-- Define the set of all lines in the plane
variable (S : Set (Set Plane))

-- Define the function f
variable (f : Set Plane → Plane)

-- Define the notion of a point being on a line
variable (on_line : Plane → Set Plane → Prop)

-- Define the notion of a line passing through a point
variable (passes_through : Set Plane → Plane → Prop)

-- Define the notion of points being on the same circle
variable (on_same_circle : Plane → Plane → Plane → Plane → Prop)

-- Main theorem
theorem unique_fixed_point
  (h1 : ∀ l ∈ S, on_line (f l) l)
  (h2 : ∀ (X : Plane) (l₁ l₂ l₃ : Set Plane),
        l₁ ∈ S → l₂ ∈ S → l₃ ∈ S →
        passes_through l₁ X → passes_through l₂ X → passes_through l₃ X →
        on_same_circle (f l₁) (f l₂) (f l₃) X) :
  ∃! P : Plane, ∀ l ∈ S, passes_through l P → f l = P :=
sorry

end unique_fixed_point_l317_31774


namespace age_ratio_is_one_half_l317_31731

/-- The ratio of Pam's age to Rena's age -/
def age_ratio (p r : ℕ) : ℚ := p / r

/-- Pam's current age -/
def pam_age : ℕ := 5

theorem age_ratio_is_one_half :
  ∃ (r : ℕ), 
    r > pam_age ∧ 
    r + 10 = pam_age + 15 ∧ 
    age_ratio pam_age r = 1 / 2 := by
  sorry

end age_ratio_is_one_half_l317_31731


namespace complex_number_relation_l317_31702

theorem complex_number_relation (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (eq1 : p = (q + r) / (s - 3))
  (eq2 : q = (p + r) / (t - 3))
  (eq3 : r = (p + q) / (u - 3))
  (eq4 : s * t + s * u + t * u = 8)
  (eq5 : s + t + u = 4) :
  s * t * u = 10 := by
sorry


end complex_number_relation_l317_31702


namespace solution_system1_solution_system2_l317_31700

-- Define the systems of equations
def system1 (x y : ℚ) : Prop :=
  x - y = 3 ∧ 3 * x - 8 * y = 14

def system2 (x y : ℚ) : Prop :=
  3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33

-- Theorem for the first system
theorem solution_system1 :
  ∃ x y : ℚ, system1 x y ∧ x = 2 ∧ y = -1 :=
by sorry

-- Theorem for the second system
theorem solution_system2 :
  ∃ x y : ℚ, system2 x y ∧ x = 6 ∧ y = -1/2 :=
by sorry

end solution_system1_solution_system2_l317_31700


namespace compute_expression_l317_31756

theorem compute_expression : 12 + 4 * (5 - 10 / 2)^3 = 12 := by
  sorry

end compute_expression_l317_31756


namespace conditions_for_a_and_b_l317_31721

/-- Given a system of equations, prove the conditions for a and b -/
theorem conditions_for_a_and_b (a b x y : ℝ) : 
  (x^2 + x*y + y^2 - y = 0) →
  (a * x^2 + b * x * y + x = 0) →
  ((a + 1)^2 = 4*(b + 1) ∧ b ≠ -1) :=
by sorry

end conditions_for_a_and_b_l317_31721


namespace prime_sum_product_l317_31752

theorem prime_sum_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 101 → p * q = 194 := by
sorry

end prime_sum_product_l317_31752


namespace triangle_side_and_area_l317_31735

/-- Given a triangle ABC with side lengths a and c, and angle B, 
    prove the length of side b and the area of the triangle. -/
theorem triangle_side_and_area 
  (a c : ℝ) 
  (B : ℝ) 
  (ha : a = 3 * Real.sqrt 3) 
  (hc : c = 2) 
  (hB : B = 150 * π / 180) : 
  ∃ (b S : ℝ), 
    b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ 
    b = 7 ∧
    S = (1/2) * a * c * Real.sin B ∧ 
    S = (3/2) * Real.sqrt 3 := by
  sorry


end triangle_side_and_area_l317_31735


namespace bridge_length_calculation_l317_31724

/-- Proves that given a train of length 75 meters, which crosses a bridge in 7.5 seconds
    and a lamp post on the bridge in 2.5 seconds, the length of the bridge is 150 meters. -/
theorem bridge_length_calculation (train_length : ℝ) (bridge_crossing_time : ℝ) (lamppost_crossing_time : ℝ)
  (h1 : train_length = 75)
  (h2 : bridge_crossing_time = 7.5)
  (h3 : lamppost_crossing_time = 2.5) :
  let bridge_length := (train_length * bridge_crossing_time / lamppost_crossing_time) - train_length
  bridge_length = 150 := by
  sorry

end bridge_length_calculation_l317_31724


namespace min_value_quadratic_l317_31794

theorem min_value_quadratic (x : ℝ) : 
  4 * x^2 + 8 * x + 16 ≥ 12 ∧ ∃ y : ℝ, 4 * y^2 + 8 * y + 16 = 12 := by
  sorry

end min_value_quadratic_l317_31794


namespace sum_of_integers_50_to_75_l317_31775

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_of_integers_50_to_75 : sum_of_integers 50 75 = 1625 := by
  sorry

end sum_of_integers_50_to_75_l317_31775


namespace cubes_not_touching_foil_l317_31705

/-- Represents the dimensions of a rectangular prism --/
structure PrismDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of cubes in a rectangular prism given its dimensions --/
def cubesInPrism (d : PrismDimensions) : ℕ := d.width * d.length * d.height

/-- Theorem: The number of cubes not touching tin foil in the given prism is 128 --/
theorem cubes_not_touching_foil (prism_width : ℕ) (inner_prism : PrismDimensions) : 
  prism_width = 10 →
  inner_prism.width = 2 * inner_prism.length →
  inner_prism.width = 2 * inner_prism.height →
  inner_prism.width ≤ prism_width - 2 →
  cubesInPrism inner_prism = 128 := by
  sorry

#check cubes_not_touching_foil

end cubes_not_touching_foil_l317_31705


namespace triangle_iff_positive_f_l317_31773

/-- The polynomial f(x, y, z) = (x + y + z)(-x + y + z)(x - y + z)(x + y - z) -/
def f (x y z : ℝ) : ℝ := (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

/-- A predicate to check if three numbers form a triangle -/
def is_triangle (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y > z ∧ y + z > x ∧ z + x > y

theorem triangle_iff_positive_f :
  ∀ x y z : ℝ, is_triangle (|x|) (|y|) (|z|) ↔ f x y z > 0 := by sorry

end triangle_iff_positive_f_l317_31773


namespace sufficient_not_necessary_l317_31785

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) := by
  sorry

end sufficient_not_necessary_l317_31785


namespace units_digit_of_squares_l317_31719

theorem units_digit_of_squares (n : ℕ) : 
  (n ≥ 10 ∧ n ≤ 99) → 
  (n % 10 = 2 ∨ n % 10 = 7) → 
  (n^2 % 10 ≠ 2 ∧ n^2 % 10 ≠ 6 ∧ n^2 % 10 ≠ 3) :=
by sorry

end units_digit_of_squares_l317_31719


namespace S_is_valid_set_l317_31708

-- Define the set of numbers greater than √2
def S : Set ℝ := {x : ℝ | x > Real.sqrt 2}

-- Theorem stating that S is a valid set
theorem S_is_valid_set : 
  (∀ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y → x ≠ y) ∧  -- Elements are distinct
  (∀ x y, x ∈ S ∧ y ∈ S → y ∈ S ∧ x ∈ S) ∧  -- Elements are unordered
  (∀ x, x ∈ S ↔ x > Real.sqrt 2)  -- Elements are determined
  := by sorry

end S_is_valid_set_l317_31708


namespace arithmetic_equalities_l317_31796

theorem arithmetic_equalities :
  (187 / 12 - 63 / 12 - 52 / 12 = 6) ∧
  (321321 * 123 - 123123 * 321 = 0) := by
  sorry

end arithmetic_equalities_l317_31796


namespace three_percent_difference_l317_31781

theorem three_percent_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) (hy : 3 = 0.05 * y) : y - x = 40 := by
  sorry

end three_percent_difference_l317_31781


namespace max_sum_perpendicular_distances_l317_31734

-- Define a triangle with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a ≥ b
  h_bc : b ≥ c
  h_pos : 0 < c

-- Define the inradius of a triangle
def inradius (t : Triangle) : ℝ := sorry

-- Define the sum of perpendicular distances from a point to the sides of the triangle
def sum_perpendicular_distances (t : Triangle) (P : ℝ × ℝ) : ℝ := sorry

theorem max_sum_perpendicular_distances (t : Triangle) :
  ∀ P, sum_perpendicular_distances t P ≤ 2 * (inradius t) * (t.a + t.b + t.c) :=
sorry

end max_sum_perpendicular_distances_l317_31734


namespace smallest_prime_perimeter_triangle_l317_31740

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

/-- A function that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- The main theorem -/
theorem smallest_prime_perimeter_triangle :
  ∃ (a b c : ℕ),
    a < b ∧ b < c ∧
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    a > 5 ∧ b > 5 ∧ c > 5 ∧
    isValidTriangle a b c ∧
    isPrime (a + b + c) ∧
    (∀ (x y z : ℕ),
      x < y ∧ y < z ∧
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      x > 5 ∧ y > 5 ∧ z > 5 ∧
      isValidTriangle x y z ∧
      isPrime (x + y + z) →
      a + b + c ≤ x + y + z) ∧
    a + b + c = 31 :=
sorry

end smallest_prime_perimeter_triangle_l317_31740


namespace price_decrease_calculation_l317_31720

/-- The original price of an article before a price decrease -/
def original_price : ℝ := 421.05

/-- The percentage of the original price after the decrease -/
def percentage_after_decrease : ℝ := 0.76

/-- The price of the article after the decrease -/
def price_after_decrease : ℝ := 320

/-- Theorem stating that the original price is correct given the conditions -/
theorem price_decrease_calculation :
  price_after_decrease = percentage_after_decrease * original_price := by
  sorry

end price_decrease_calculation_l317_31720


namespace at_most_one_true_l317_31701

theorem at_most_one_true (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end at_most_one_true_l317_31701


namespace log_inequality_l317_31772

theorem log_inequality : 
  Real.log 6 / Real.log 3 > Real.log 10 / Real.log 5 ∧ 
  Real.log 10 / Real.log 5 > Real.log 14 / Real.log 7 := by
  sorry

end log_inequality_l317_31772


namespace bake_sale_donation_l317_31783

/-- The total donation to the homeless shelter given the bake sale earnings and additional personal donation -/
def total_donation_to_shelter (total_earnings : ℕ) (ingredients_cost : ℕ) (personal_donation : ℕ) : ℕ :=
  let remaining_total := total_earnings - ingredients_cost
  let shelter_donation := remaining_total / 2 + personal_donation
  shelter_donation

/-- Theorem stating that the total donation to the homeless shelter is $160 -/
theorem bake_sale_donation :
  total_donation_to_shelter 400 100 10 = 160 := by
  sorry

end bake_sale_donation_l317_31783


namespace max_value_a_l317_31769

theorem max_value_a (a b c e : ℕ+) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 5 * e)
  (h4 : e < 100) :
  a ≤ 2961 ∧ ∃ (a' b' c' e' : ℕ+), 
    a' = 2961 ∧ 
    a' < 2 * b' ∧ 
    b' < 3 * c' ∧ 
    c' < 5 * e' ∧ 
    e' < 100 :=
by sorry

end max_value_a_l317_31769


namespace polynomial_evaluation_l317_31758

theorem polynomial_evaluation :
  let f (x : ℤ) := x^3 + x^2 + x + 1
  f (-2) = -5 := by sorry

end polynomial_evaluation_l317_31758


namespace price_reduction_profit_l317_31784

/-- Represents the daily sales and profit model for a product in a shopping mall -/
structure SalesModel where
  baseItems : ℕ  -- Base number of items sold per day
  baseProfit : ℕ  -- Base profit per item in yuan
  salesIncrease : ℕ  -- Additional items sold per yuan of price reduction
  priceReduction : ℕ  -- Amount of price reduction in yuan

/-- Calculates the daily profit given a SalesModel -/
def dailyProfit (model : SalesModel) : ℕ :=
  let newItems := model.baseItems + model.salesIncrease * model.priceReduction
  let newProfit := model.baseProfit - model.priceReduction
  newItems * newProfit

/-- Theorem stating that a price reduction of 20 yuan results in a daily profit of 2100 yuan -/
theorem price_reduction_profit (model : SalesModel) 
  (h1 : model.baseItems = 30)
  (h2 : model.baseProfit = 50)
  (h3 : model.salesIncrease = 2)
  (h4 : model.priceReduction = 20) :
  dailyProfit model = 2100 := by
  sorry

end price_reduction_profit_l317_31784


namespace intersection_implies_a_value_l317_31789

def A : Set ℝ := {-1, 0, 1}
def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a = {0}) → a = -1 := by sorry

end intersection_implies_a_value_l317_31789


namespace framed_painting_ratio_l317_31714

/-- The width of the painting in inches -/
def painting_width : ℝ := 20

/-- The height of the painting in inches -/
def painting_height : ℝ := 30

/-- The frame width on the sides in inches -/
def frame_side_width : ℝ := 5

/-- The frame width on the top and bottom in inches -/
def frame_top_bottom_width : ℝ := 3 * frame_side_width

/-- The area of the painting in square inches -/
def painting_area : ℝ := painting_width * painting_height

/-- The area of the framed painting in square inches -/
def framed_painting_area : ℝ := (painting_width + 2 * frame_side_width) * (painting_height + 2 * frame_top_bottom_width)

/-- The width of the framed painting in inches -/
def framed_painting_width : ℝ := painting_width + 2 * frame_side_width

/-- The height of the framed painting in inches -/
def framed_painting_height : ℝ := painting_height + 2 * frame_top_bottom_width

theorem framed_painting_ratio :
  framed_painting_area = 2 * painting_area ∧
  framed_painting_width / framed_painting_height = 1 / 2 := by
  sorry

end framed_painting_ratio_l317_31714


namespace deepak_age_l317_31755

/-- Given that the ratio of Rahul's age to Deepak's age is 4:3 and 
    Rahul's age after 6 years will be 34 years, 
    prove that Deepak's present age is 21 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 34 →
  deepak_age = 21 := by
sorry

end deepak_age_l317_31755


namespace chord_equation_l317_31709

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a circle with center (0,0) and radius 3 -/
def isOnCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 = 9

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m p q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

/-- Checks if a line passes through a point -/
def linePassesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem chord_equation (p q : Point) (m : Point) :
  isOnCircle p ∧ isOnCircle q ∧ isMidpoint m p q ∧ m.x = 1 ∧ m.y = 2 →
  ∃ l : Line, l.a = 1 ∧ l.b = 2 ∧ l.c = -5 ∧ linePassesThrough l p ∧ linePassesThrough l q :=
sorry

end chord_equation_l317_31709


namespace fraction_cracked_pots_is_two_fifths_l317_31744

/-- The fraction of cracked pots given the initial number of pots,
    the revenue from selling non-cracked pots, and the price per pot. -/
def fraction_cracked_pots (initial_pots : ℕ) (revenue : ℕ) (price_per_pot : ℕ) : ℚ :=
  1 - (revenue / (initial_pots * price_per_pot) : ℚ)

/-- Theorem stating that the fraction of cracked pots is 2/5 given the problem conditions. -/
theorem fraction_cracked_pots_is_two_fifths :
  fraction_cracked_pots 80 1920 40 = 2 / 5 := by
  sorry

end fraction_cracked_pots_is_two_fifths_l317_31744


namespace female_students_count_l317_31704

/-- Represents the class configuration described in the problem -/
structure ClassConfiguration where
  total_students : Nat
  male_students : Nat
  (total_ge_male : total_students ≥ male_students)

/-- The number of students called by the kth student -/
def students_called (k : Nat) : Nat := k + 2

/-- The theorem statement -/
theorem female_students_count (c : ClassConfiguration) 
  (h1 : c.total_students = 42)
  (h2 : ∀ k, k ≤ c.male_students → students_called k ≤ c.total_students)
  (h3 : students_called c.male_students = c.total_students / 2) :
  c.total_students - c.male_students = 23 := by
  sorry


end female_students_count_l317_31704


namespace company_picnic_attendance_l317_31777

/-- Represents the percentage of men who attended the company picnic -/
def percentage_men_attended : ℝ := 0.2

theorem company_picnic_attendance 
  (percent_women_attended : ℝ) 
  (percent_men_total : ℝ) 
  (percent_total_attended : ℝ) 
  (h1 : percent_women_attended = 0.4)
  (h2 : percent_men_total = 0.45)
  (h3 : percent_total_attended = 0.31000000000000007) :
  percentage_men_attended = 
    (percent_total_attended - (1 - percent_men_total) * percent_women_attended) / percent_men_total :=
by sorry

end company_picnic_attendance_l317_31777


namespace similarity_of_triangles_l317_31791

-- Define the points
variable (A B C D E F G H O : Point)

-- Define the cyclic quadrilateral ABCD
def is_cyclic_quad (A B C D : Point) : Prop := sorry

-- Define the circle centered at O passing through B and D
def circle_O_passes_through (O B D : Point) : Prop := sorry

-- Define that E and F are on lines BA and BC respectively
def E_on_BA (E B A : Point) : Prop := sorry
def F_on_BC (F B C : Point) : Prop := sorry

-- Define that E and F are distinct from A, B, C
def E_F_distinct (E F A B C : Point) : Prop := sorry

-- Define H as the orthocenter of triangle DEF
def H_orthocenter_DEF (H D E F : Point) : Prop := sorry

-- Define that AC, DO, and EF are concurrent
def lines_concurrent (A C D O E F : Point) : Prop := sorry

-- Define similarity of triangles
def triangles_similar (A B C E H F : Point) : Prop := sorry

-- Theorem statement
theorem similarity_of_triangles 
  (h1 : is_cyclic_quad A B C D)
  (h2 : circle_O_passes_through O B D)
  (h3 : E_on_BA E B A)
  (h4 : F_on_BC F B C)
  (h5 : E_F_distinct E F A B C)
  (h6 : H_orthocenter_DEF H D E F)
  (h7 : lines_concurrent A C D O E F) :
  triangles_similar A B C E H F :=
sorry

end similarity_of_triangles_l317_31791


namespace boys_to_girls_ratio_l317_31797

theorem boys_to_girls_ratio (total_students girls : ℕ) 
  (h1 : total_students = 1040)
  (h2 : girls = 400) :
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end boys_to_girls_ratio_l317_31797


namespace pure_imaginary_condition_l317_31770

theorem pure_imaginary_condition (m : ℝ) : 
  (∀ z : ℂ, z = Complex.mk (m^2 - 1) (m + 1) → z.re = 0 ∧ z.im ≠ 0) → m = 1 := by
  sorry

end pure_imaginary_condition_l317_31770


namespace sqrt_sum_equality_l317_31768

theorem sqrt_sum_equality : Real.sqrt (11 + 6 * Real.sqrt 2) + Real.sqrt (11 - 6 * Real.sqrt 2) = 6 := by
  sorry

end sqrt_sum_equality_l317_31768


namespace sqrt_24_simplification_l317_31790

theorem sqrt_24_simplification : Real.sqrt 24 = 2 * Real.sqrt 6 := by sorry

end sqrt_24_simplification_l317_31790


namespace range_of_x_l317_31707

theorem range_of_x (x : ℝ) : (1 + 2*x ≤ 8 + 3*x) → x ≥ -7 := by
  sorry

end range_of_x_l317_31707


namespace zoe_chocolate_sales_l317_31728

/-- Given a box of chocolate bars, calculate the money made from selling a certain number of bars. -/
def money_made (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Prove that Zoe made $42 by selling all but 6 bars from a box of 13 bars, each costing $6. -/
theorem zoe_chocolate_sales : money_made 13 6 6 = 42 := by
  sorry

end zoe_chocolate_sales_l317_31728


namespace target_shooting_problem_l317_31771

theorem target_shooting_problem (p : ℝ) (k₀ : ℕ) (h_p : p = 0.7) (h_k₀ : k₀ = 16) :
  ∃ n : ℕ, (n = 22 ∨ n = 23) ∧ 
    (k₀ : ℝ) ≤ n * p + p ∧ 
    (k₀ : ℝ) ≥ n * p - (1 - p) :=
sorry

end target_shooting_problem_l317_31771


namespace rectangle_perimeter_l317_31750

/-- Perimeter of a rectangle with area equal to a right triangle --/
theorem rectangle_perimeter (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a * b = 108) : 
  let triangle_area := a * b / 2
  let rectangle_length := c / 2
  let rectangle_width := triangle_area / rectangle_length
  2 * (rectangle_length + rectangle_width) = 29.4 :=
by sorry

end rectangle_perimeter_l317_31750


namespace partnership_investment_l317_31798

/-- Partnership investment problem -/
theorem partnership_investment (x : ℝ) (y : ℝ) : 
  x > 0 →  -- Raman's investment is positive
  y > 0 →  -- Lakshmi invests after a positive number of months
  y < 12 → -- Lakshmi invests before the end of the year
  (2 * x * (12 - y)) / (x * 12 + 2 * x * (12 - y) + 3 * x * 4) = 1 / 3 →
  y = 6 := by
sorry

end partnership_investment_l317_31798


namespace exists_triangle_with_large_inner_triangle_l317_31782

-- Define the structure of a triangle
structure Triangle :=
  (A B C : Point)

-- Define the properties of the triangle
def is_acute (t : Triangle) : Prop := sorry

-- Define the line segments
def median (t : Triangle) : Point → Point := sorry
def angle_bisector (t : Triangle) : Point → Point := sorry
def altitude (t : Triangle) : Point → Point := sorry

-- Define the intersection points
def intersection_points (t : Triangle) : Triangle := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- The main theorem
theorem exists_triangle_with_large_inner_triangle :
  ∃ (t : Triangle),
    is_acute t ∧
    area (intersection_points t) > 0.499 * area t :=
sorry

end exists_triangle_with_large_inner_triangle_l317_31782


namespace range_of_x_l317_31729

theorem range_of_x (x : ℝ) : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 9/a + 1/b = 2 ∧ 
    (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' ≥ x^2 + 2*x)) → 
  -4 ≤ x ∧ x ≤ 2 := by
sorry

end range_of_x_l317_31729


namespace odd_periodic_function_sum_l317_31766

-- Define the properties of the function f
def is_odd_periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (f 1 = 2) ∧ 
  (∀ x, f (x + 1) = f (x + 5))

-- State the theorem
theorem odd_periodic_function_sum (f : ℝ → ℝ) 
  (h : is_odd_periodic_function f) : f 12 + f 3 = -2 := by
  sorry

end odd_periodic_function_sum_l317_31766


namespace least_integer_in_ratio_l317_31722

theorem least_integer_in_ratio (a b c : ℕ+) : 
  (a : ℝ) + (b : ℝ) + (c : ℝ) = 90 →
  (b : ℝ) = 3 * (a : ℝ) →
  (c : ℝ) = 5 * (a : ℝ) →
  a = 10 := by
sorry

end least_integer_in_ratio_l317_31722


namespace length_of_AB_l317_31706

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = -12*x

-- Define the line
def line (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧ 
  line A.1 A.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) : 
  intersection_points A B → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 30 :=
sorry

end length_of_AB_l317_31706


namespace max_intersection_area_theorem_l317_31792

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- Represents the maximum possible intersection area of two rectangles --/
def max_intersection_area (r1 r2 : Rectangle) : ℕ :=
  min r1.width r2.width * min r1.height r2.height

theorem max_intersection_area_theorem (r1 r2 : Rectangle) :
  r1.width < r1.height →
  r2.width > r2.height →
  2011 < area r1 →
  area r1 < 2020 →
  2011 < area r2 →
  area r2 < 2020 →
  max_intersection_area r1 r2 ≤ 1764 ∧
  ∃ (r1' r2' : Rectangle),
    r1'.width < r1'.height ∧
    r2'.width > r2'.height ∧
    2011 < area r1' ∧
    area r1' < 2020 ∧
    2011 < area r2' ∧
    area r2' < 2020 ∧
    max_intersection_area r1' r2' = 1764 := by
  sorry

#check max_intersection_area_theorem

end max_intersection_area_theorem_l317_31792
