import Mathlib

namespace tree_planting_equation_l1149_114924

theorem tree_planting_equation (x : ℝ) (h : x > 0) : 
  (180 / x - 180 / (1.5 * x) = 2) ↔ 
  (∃ (planned_trees actual_trees : ℝ),
    planned_trees = 180 / x ∧
    actual_trees = 180 / (1.5 * x) ∧
    planned_trees - actual_trees = 2 ∧
    180 / x > 2) := by sorry

end tree_planting_equation_l1149_114924


namespace unique_integer_function_l1149_114965

def IntegerFunction (f : ℤ → ℚ) : Prop :=
  ∀ (x y z : ℤ), 
    (∀ (c : ℚ), f x < c ∧ c < f y → ∃ (w : ℤ), f w = c) ∧
    (x + y + z = 0 → f x + f y + f z = f x * f y * f z)

theorem unique_integer_function : 
  ∃! (f : ℤ → ℚ), IntegerFunction f ∧ (∀ x : ℤ, f x = 0) :=
sorry

end unique_integer_function_l1149_114965


namespace f_range_l1149_114957

noncomputable def f (x : ℝ) : ℝ :=
  (1/2) * Real.sin (2*x) * Real.tan x + 2 * Real.sin x * Real.tan (x/2)

theorem f_range :
  Set.range f = Set.Icc 0 3 ∪ Set.Ioo 3 4 :=
sorry

end f_range_l1149_114957


namespace line_symmetry_x_axis_symmetric_line_2x_plus_1_l1149_114968

/-- Given a line y = mx + b, its symmetric line with respect to the x-axis is y = -mx - b -/
theorem line_symmetry_x_axis (m b : ℝ) :
  let original_line := fun (x : ℝ) => m * x + b
  let symmetric_line := fun (x : ℝ) => -m * x - b
  ∀ x y : ℝ, y = original_line x ↔ -y = symmetric_line x :=
by sorry

/-- The line symmetric to y = 2x + 1 with respect to the x-axis is y = -2x - 1 -/
theorem symmetric_line_2x_plus_1 :
  let original_line := fun (x : ℝ) => 2 * x + 1
  let symmetric_line := fun (x : ℝ) => -2 * x - 1
  ∀ x y : ℝ, y = original_line x ↔ -y = symmetric_line x :=
by sorry

end line_symmetry_x_axis_symmetric_line_2x_plus_1_l1149_114968


namespace unique_solution_rational_equation_l1149_114975

theorem unique_solution_rational_equation :
  ∃! x : ℚ, x ≠ 4 ∧ x ≠ 1 ∧
  (3 * x^2 - 15 * x + 12) / (2 * x^2 - 10 * x + 8) = x - 4 := by
  sorry

end unique_solution_rational_equation_l1149_114975


namespace thirteen_people_evaluations_l1149_114989

/-- The number of evaluations for a group of people, where each pair is categorized into one of three categories. -/
def num_evaluations (n : ℕ) : ℕ := n.choose 2 * 3

/-- Theorem: For a group of 13 people, where each pair is categorized into one of three categories, the total number of evaluations is 234. -/
theorem thirteen_people_evaluations : num_evaluations 13 = 234 := by
  sorry

end thirteen_people_evaluations_l1149_114989


namespace right_triangle_third_side_l1149_114940

theorem right_triangle_third_side (a b x : ℝ) : 
  a = 3 → b = 4 → (a^2 + b^2 = x^2 ∨ a^2 + x^2 = b^2) → x = 5 ∨ x = Real.sqrt 7 := by
  sorry

end right_triangle_third_side_l1149_114940


namespace eccentricity_product_range_l1149_114954

/-- An ellipse and a hyperbola with common foci -/
structure ConicPair where
  F₁ : ℝ × ℝ  -- Left focus
  F₂ : ℝ × ℝ  -- Right focus
  P : ℝ × ℝ   -- Intersection point
  e₁ : ℝ      -- Eccentricity of ellipse
  e₂ : ℝ      -- Eccentricity of hyperbola

/-- The conditions given in the problem -/
def satisfies_conditions (pair : ConicPair) : Prop :=
  pair.F₁.1 < 0 ∧ pair.F₂.1 > 0 ∧  -- Foci on x-axis, centered at origin
  pair.P.1 > 0 ∧ pair.P.2 > 0 ∧    -- P in first quadrant
  ‖pair.P - pair.F₁‖ = ‖pair.P - pair.F₂‖ ∧  -- Isosceles triangle
  ‖pair.P - pair.F₁‖ = 10 ∧        -- |PF₁| = 10
  pair.e₁ > 0 ∧ pair.e₂ > 0        -- Positive eccentricities

theorem eccentricity_product_range (pair : ConicPair) 
  (h : satisfies_conditions pair) : 
  pair.e₁ * pair.e₂ > 1/3 ∧ 
  ∀ M, ∃ pair', satisfies_conditions pair' ∧ pair'.e₁ * pair'.e₂ > M :=
by sorry

end eccentricity_product_range_l1149_114954


namespace dogwood_trees_planted_today_l1149_114922

/-- The number of dogwood trees planted today -/
def trees_planted_today : ℕ := 41

/-- The initial number of trees in the park -/
def initial_trees : ℕ := 39

/-- The number of trees to be planted tomorrow -/
def trees_planted_tomorrow : ℕ := 20

/-- The final total number of trees -/
def final_total_trees : ℕ := 100

theorem dogwood_trees_planted_today :
  initial_trees + trees_planted_today + trees_planted_tomorrow = final_total_trees :=
by sorry

end dogwood_trees_planted_today_l1149_114922


namespace magnitude_relationship_l1149_114976

theorem magnitude_relationship : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by
  sorry

end magnitude_relationship_l1149_114976


namespace father_son_age_ratio_l1149_114960

def father_son_ages (son_age : ℕ) (age_difference : ℕ) : Prop :=
  ∃ (k : ℕ), (son_age + age_difference + 2) = k * (son_age + 2)

theorem father_son_age_ratio :
  let son_age : ℕ := 22
  let age_difference : ℕ := 24
  father_son_ages son_age age_difference →
  (son_age + age_difference + 2) / (son_age + 2) = 2 := by
sorry

end father_son_age_ratio_l1149_114960


namespace decimal_addition_l1149_114969

theorem decimal_addition : (0.9 : ℝ) + 0.09 = 0.99 := by
  sorry

end decimal_addition_l1149_114969


namespace integer_solutions_of_equation_l1149_114918

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 6*x*y - 4*x + 9*y - 366 = 0 ↔ (x = 3 ∧ y = 14) ∨ (x = -24 ∧ y = -2) := by
  sorry

end integer_solutions_of_equation_l1149_114918


namespace phone_bill_minutes_l1149_114927

def monthly_fee : ℚ := 2
def per_minute_rate : ℚ := 12 / 100
def total_bill : ℚ := 2336 / 100

theorem phone_bill_minutes : 
  ∃ (minutes : ℕ), 
    (monthly_fee + per_minute_rate * minutes) = total_bill ∧ 
    minutes = 178 := by
  sorry

end phone_bill_minutes_l1149_114927


namespace debby_stuffed_animal_tickets_l1149_114907

/-- The number of tickets Debby spent on various items at the arcade -/
structure ArcadeTickets where
  hat : ℕ
  yoyo : ℕ
  stuffed_animal : ℕ
  total : ℕ

/-- Theorem about Debby's ticket spending at the arcade -/
theorem debby_stuffed_animal_tickets (d : ArcadeTickets) 
  (hat_tickets : d.hat = 2)
  (yoyo_tickets : d.yoyo = 2)
  (total_tickets : d.total = 14)
  (sum_correct : d.hat + d.yoyo + d.stuffed_animal = d.total) :
  d.stuffed_animal = 10 := by
  sorry

end debby_stuffed_animal_tickets_l1149_114907


namespace retail_price_decrease_percentage_l1149_114913

/-- Proves that the retail price decrease percentage is equal to 44.000000000000014% 
    given the conditions in the problem. -/
theorem retail_price_decrease_percentage 
  (wholesale_price : ℝ) 
  (retail_price : ℝ) 
  (decrease_percentage : ℝ) : 
  retail_price = wholesale_price * 1.80 →
  retail_price * (1 - decrease_percentage) = 
    (wholesale_price * 1.44000000000000014) * 1.80 →
  decrease_percentage = 0.44000000000000014 := by
sorry

end retail_price_decrease_percentage_l1149_114913


namespace train_length_l1149_114983

/-- Given a bridge and a train, prove the length of the train -/
theorem train_length 
  (bridge_length : ℝ) 
  (train_cross_time : ℝ) 
  (man_cross_time : ℝ) 
  (train_speed : ℝ) 
  (h1 : bridge_length = 180) 
  (h2 : train_cross_time = 20) 
  (h3 : man_cross_time = 8) 
  (h4 : train_speed = 15) : 
  ∃ train_length : ℝ, train_length = 120 := by
  sorry

end train_length_l1149_114983


namespace mans_rowing_speed_in_still_water_l1149_114981

/-- Proves that a man's rowing speed in still water is 15 km/h given the conditions of downstream travel --/
theorem mans_rowing_speed_in_still_water :
  let current_speed : ℝ := 3 -- km/h
  let distance : ℝ := 60 / 1000 -- 60 meters converted to km
  let time : ℝ := 11.999040076793857 / 3600 -- seconds converted to hours
  let downstream_speed : ℝ := distance / time
  downstream_speed = current_speed + 15 := by sorry

end mans_rowing_speed_in_still_water_l1149_114981


namespace smallest_n_for_factorization_l1149_114903

/-- 
Theorem: The smallest value of n for which 5x^2 + nx + 60 can be factored 
as the product of two linear factors with integer coefficients is 56.
-/
theorem smallest_n_for_factorization : 
  (∃ n : ℤ, ∀ m : ℤ, 
    (∃ a b : ℤ, 5 * X^2 + n * X + 60 = (5 * X + a) * (X + b)) ∧ 
    (∀ k : ℤ, k < n → ¬∃ c d : ℤ, 5 * X^2 + k * X + 60 = (5 * X + c) * (X + d))) ∧
  (∀ n : ℤ, 
    (∃ a b : ℤ, 5 * X^2 + n * X + 60 = (5 * X + a) * (X + b)) ∧ 
    (∀ k : ℤ, k < n → ¬∃ c d : ℤ, 5 * X^2 + k * X + 60 = (5 * X + c) * (X + d)) 
    → n = 56) :=
sorry


end smallest_n_for_factorization_l1149_114903


namespace lcm_of_9_12_15_l1149_114979

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end lcm_of_9_12_15_l1149_114979


namespace polynomial_divisibility_l1149_114930

theorem polynomial_divisibility (x : ℝ) : 
  let f : ℝ → ℝ := λ x => -x^4 - x^3 - x + 1
  ∃ (u v : ℝ → ℝ), 
    f x = (x^2 + 1) * (u x) ∧ 
    f x + 1 = (x^3 + x^2 + 1) * (v x) := by
  sorry

end polynomial_divisibility_l1149_114930


namespace triangle_inequality_left_equality_condition_right_equality_condition_l1149_114947

/-- A triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq_ab : a + b > c
  triangle_ineq_bc : b + c > a
  triangle_ineq_ca : c + a > b

theorem triangle_inequality (t : Triangle) :
  3 * (t.a * t.b + t.b * t.c + t.c * t.a) ≤ (t.a + t.b + t.c)^2 ∧
  (t.a + t.b + t.c)^2 < 4 * (t.a * t.b + t.b * t.c + t.c * t.a) := by
  sorry

theorem left_equality_condition (t : Triangle) :
  3 * (t.a * t.b + t.b * t.c + t.c * t.a) = (t.a + t.b + t.c)^2 ↔ t.a = t.b ∧ t.b = t.c := by
  sorry

theorem right_equality_condition (t : Triangle) :
  (t.a + t.b + t.c)^2 = 4 * (t.a * t.b + t.b * t.c + t.c * t.a) ↔
  t.a + t.b = t.c ∨ t.b + t.c = t.a ∨ t.c + t.a = t.b := by
  sorry

end triangle_inequality_left_equality_condition_right_equality_condition_l1149_114947


namespace red_balls_count_l1149_114984

/-- Given a bag of balls with red and yellow colors, prove that the number of red balls is 6 -/
theorem red_balls_count (total_balls : ℕ) (prob_red : ℚ) : 
  total_balls = 15 → prob_red = 2/5 → (prob_red * total_balls : ℚ) = 6 := by
  sorry

end red_balls_count_l1149_114984


namespace dart_board_probability_l1149_114958

/-- The probability of a dart landing in the center square of a regular octagon dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2
  let center_square_area := (s/2)^2
  center_square_area / octagon_area = 1 / (4 + 4 * Real.sqrt 2) := by
  sorry

end dart_board_probability_l1149_114958


namespace angle_AOB_is_right_angle_l1149_114953

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 3*x

-- Define a line passing through (3,0)
def line_through_3_0 (t : ℝ) (x y : ℝ) : Prop := x = t*y + 3

-- Define the intersection points
def intersection_points (t : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_through_3_0 t x₁ y₁ ∧ line_through_3_0 t x₂ y₂

-- Theorem statement
theorem angle_AOB_is_right_angle (t : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  intersection_points t x₁ y₁ x₂ y₂ →
  x₁ * x₂ + y₁ * y₂ = 0 :=
sorry

end angle_AOB_is_right_angle_l1149_114953


namespace solution_k_value_l1149_114974

theorem solution_k_value (x y k : ℝ) 
  (hx : x = -1)
  (hy : y = 2)
  (heq : 2 * x + k * y = 6) :
  k = 4 := by
  sorry

end solution_k_value_l1149_114974


namespace fraction_simplification_l1149_114967

theorem fraction_simplification (a b c : ℝ) (h : a + b + c ≠ 0) :
  (a^2 + 3*a*b + b^2 - c^2) / (a^2 + 3*a*c + c^2 - b^2) = (a + b - c) / (a - b + c) := by
  sorry

end fraction_simplification_l1149_114967


namespace list_price_calculation_l1149_114994

theorem list_price_calculation (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) → 
  list_price = 40 := by
  sorry

end list_price_calculation_l1149_114994


namespace probability_one_of_each_l1149_114973

def num_shirts : ℕ := 6
def num_shorts : ℕ := 8
def num_socks : ℕ := 9
def num_hats : ℕ := 4
def total_items : ℕ := num_shirts + num_shorts + num_socks + num_hats
def items_to_select : ℕ := 4

theorem probability_one_of_each :
  (num_shirts.choose 1 * num_shorts.choose 1 * num_socks.choose 1 * num_hats.choose 1) / total_items.choose items_to_select = 96 / 975 :=
by sorry

end probability_one_of_each_l1149_114973


namespace right_triangle_tangent_midpoint_l1149_114910

theorem right_triangle_tangent_midpoint (n : ℕ) (a h : ℝ) (α : ℝ) :
  n > 1 →
  Odd n →
  0 < a →
  0 < h →
  0 < α →
  α < π / 2 →
  Real.tan α = (4 * n * h) / ((n^2 - 1) * a) :=
by sorry

end right_triangle_tangent_midpoint_l1149_114910


namespace tangent_line_to_exp_plus_x_l1149_114992

/-- A line y = mx + b is tangent to a curve y = f(x) at point (x₀, f(x₀)) if:
    1. The line passes through the point (x₀, f(x₀))
    2. The slope of the line equals the derivative of f at x₀ -/
def is_tangent_line (f : ℝ → ℝ) (f' : ℝ → ℝ) (m b x₀ : ℝ) : Prop :=
  f x₀ = m * x₀ + b ∧ f' x₀ = m

theorem tangent_line_to_exp_plus_x (b : ℝ) :
  (∃ x₀ : ℝ, is_tangent_line (λ x => Real.exp x + x) (λ x => Real.exp x + 1) 2 b x₀) →
  b = 1 :=
sorry

end tangent_line_to_exp_plus_x_l1149_114992


namespace average_equation_solution_l1149_114971

theorem average_equation_solution (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 84 → a = 32 := by
  sorry

end average_equation_solution_l1149_114971


namespace transformations_of_f_l1149_114963

def f (x : ℝ) : ℝ := 3 * x + 4

def shift_left_down (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = f (x + 1) - 2

def reflect_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = f (-x)

def reflect_y_eq_1 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = 2 - f x

def reflect_y_eq_neg_x (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = (x + 4) / 3

def reflect_point (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, g x = f (2 * a - x) + 2 * (b - a)

theorem transformations_of_f :
  (∃ g : ℝ → ℝ, shift_left_down g ∧ (∀ x, g x = 3 * x + 5)) ∧
  (∃ g : ℝ → ℝ, reflect_y_axis g ∧ (∀ x, g x = -3 * x + 4)) ∧
  (∃ g : ℝ → ℝ, reflect_y_eq_1 g ∧ (∀ x, g x = -3 * x - 2)) ∧
  (∃ g : ℝ → ℝ, reflect_y_eq_neg_x g) ∧
  (∀ a b : ℝ, ∃ g : ℝ → ℝ, reflect_point g a b ∧ (∀ x, g x = 3 * x + 2 * b - 6 * a - 4)) :=
sorry

end transformations_of_f_l1149_114963


namespace heptagon_diagonals_l1149_114931

/-- The number of distinct diagonals in a convex heptagon -/
def num_diagonals_heptagon : ℕ := 14

/-- The number of sides in a heptagon -/
def heptagon_sides : ℕ := 7

/-- Theorem: The number of distinct diagonals in a convex heptagon is 14 -/
theorem heptagon_diagonals :
  num_diagonals_heptagon = (heptagon_sides * (heptagon_sides - 3)) / 2 :=
by sorry

end heptagon_diagonals_l1149_114931


namespace smallest_five_digit_multiple_l1149_114990

theorem smallest_five_digit_multiple : ∃ (n : ℕ), 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  (15 ∣ n) ∧ (45 ∣ n) ∧ (54 ∣ n) ∧ 
  (∃ (k : ℕ), n = 2^k * (n / 2^k)) ∧
  (∀ (m : ℕ), m < n → 
    ¬((m ≥ 10000 ∧ m < 100000) ∧ 
      (15 ∣ m) ∧ (45 ∣ m) ∧ (54 ∣ m) ∧ 
      (∃ (j : ℕ), m = 2^j * (m / 2^j)))) ∧
  n = 69120 := by
sorry

end smallest_five_digit_multiple_l1149_114990


namespace papers_per_notepad_l1149_114996

/-- The number of folds applied to the paper -/
def num_folds : ℕ := 3

/-- The number of days a notepad lasts -/
def days_per_notepad : ℕ := 4

/-- The number of notes written per day -/
def notes_per_day : ℕ := 10

/-- The number of smaller pieces obtained from one letter-size paper after folding -/
def pieces_per_paper : ℕ := 2^num_folds

/-- The total number of notes in one notepad -/
def notes_per_notepad : ℕ := days_per_notepad * notes_per_day

/-- Theorem: The number of letter-size papers needed for one notepad is 5 -/
theorem papers_per_notepad : (notes_per_notepad + pieces_per_paper - 1) / pieces_per_paper = 5 := by
  sorry

end papers_per_notepad_l1149_114996


namespace cube_inequality_l1149_114932

theorem cube_inequality (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end cube_inequality_l1149_114932


namespace hexagon_sixth_angle_l1149_114926

/-- A hexagon with given angle measures -/
structure Hexagon where
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  U : ℝ
  V : ℝ
  sum_angles : Q + R + S + T + U + V = 720
  Q_value : Q = 110
  R_value : R = 135
  S_value : S = 140
  T_value : T = 95
  U_value : U = 100

/-- The sixth angle of a hexagon with five known angles measures 140° -/
theorem hexagon_sixth_angle (h : Hexagon) : h.V = 140 := by
  sorry

end hexagon_sixth_angle_l1149_114926


namespace visual_illusion_occurs_l1149_114933

/-- Represents the structure of the cardboard disc -/
structure Disc where
  inner_sectors : Nat
  outer_sectors : Nat
  inner_white : Nat
  outer_white : Nat

/-- Represents the properties of electric lighting -/
structure Lighting where
  flicker_frequency : Real
  flicker_interval : Real

/-- Defines the rotation speeds that create the visual illusion -/
def illusion_speeds (d : Disc) (l : Lighting) : Prop :=
  let inner_speed := 25
  let outer_speed := 20
  inner_speed * l.flicker_interval = 0.25 ∧
  outer_speed * l.flicker_interval = 0.2

theorem visual_illusion_occurs (d : Disc) (l : Lighting) :
  d.inner_sectors = 8 ∧
  d.outer_sectors = 10 ∧
  d.inner_white = 4 ∧
  d.outer_white = 5 ∧
  l.flicker_frequency = 100 ∧
  l.flicker_interval = 0.01 →
  illusion_speeds d l :=
by sorry


end visual_illusion_occurs_l1149_114933


namespace robins_hair_length_l1149_114991

/-- Given Robin's initial hair length and the amount cut, calculate the remaining length -/
theorem robins_hair_length (initial_length cut_length : ℕ) : 
  initial_length = 17 → cut_length = 4 → initial_length - cut_length = 13 := by
  sorry

end robins_hair_length_l1149_114991


namespace figure_area_proof_l1149_114945

theorem figure_area_proof (square_side : ℝ) (gray_area white_area : ℝ) 
  (h1 : gray_area = square_side^2)
  (h2 : white_area = (3/2) * square_side^2)
  (h3 : gray_area = white_area + 0.6)
  (h4 : square_side^2 = 1.2) :
  5 * square_side^2 = 6 := by sorry

end figure_area_proof_l1149_114945


namespace p_amount_l1149_114997

theorem p_amount (p : ℚ) : p = (1/4) * p + 42 → p = 56 := by
  sorry

end p_amount_l1149_114997


namespace sin_120_degrees_l1149_114952

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_120_degrees_l1149_114952


namespace average_sale_calculation_l1149_114938

theorem average_sale_calculation (sale1 sale2 sale3 sale4 : ℕ) :
  sale1 = 2500 →
  sale2 = 4000 →
  sale3 = 3540 →
  sale4 = 1520 →
  (sale1 + sale2 + sale3 + sale4) / 4 = 2890 := by
sorry

end average_sale_calculation_l1149_114938


namespace sqrt_196_equals_14_l1149_114955

theorem sqrt_196_equals_14 : Real.sqrt 196 = 14 := by
  sorry

end sqrt_196_equals_14_l1149_114955


namespace tangent_circles_diametric_intersection_l1149_114998

-- Define the types for our geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given circles and points
variable (c c1 c2 : Circle)
variable (A B P Q : Point)

-- Define the property of internal tangency
def internallyTangent (c1 c2 : Circle) (P : Point) : Prop :=
  -- The distance between centers is the difference of radii
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius - c2.radius)^2
  -- P lies on both circles
  ∧ (P.x - c1.center.1)^2 + (P.y - c1.center.2)^2 = c1.radius^2
  ∧ (P.x - c2.center.1)^2 + (P.y - c2.center.2)^2 = c2.radius^2

-- Define the property of a point lying on a circle
def pointOnCircle (P : Point) (c : Circle) : Prop :=
  (P.x - c.center.1)^2 + (P.y - c.center.2)^2 = c.radius^2

-- Define the property of points being diametrically opposite on a circle
def diametricallyOpposite (P Q : Point) (c : Circle) : Prop :=
  (P.x - c.center.1) = -(Q.x - c.center.1) ∧ (P.y - c.center.2) = -(Q.y - c.center.2)

-- State the theorem
theorem tangent_circles_diametric_intersection :
  internallyTangent c c1 A
  → internallyTangent c c2 B
  → ∃ (M N : Point),
    pointOnCircle M c
    ∧ pointOnCircle N c
    ∧ diametricallyOpposite M N c
    ∧ (∃ (t : ℝ), M = ⟨A.x + t * (P.x - A.x), A.y + t * (P.y - A.y)⟩)
    ∧ (∃ (s : ℝ), N = ⟨B.x + s * (Q.x - B.x), B.y + s * (Q.y - B.y)⟩) :=
by sorry

end tangent_circles_diametric_intersection_l1149_114998


namespace oldest_child_age_l1149_114959

def children_ages (ages : Fin 5 → ℕ) : Prop :=
  -- The average age is 6
  (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 = 6 ∧
  -- Ages are different
  ∀ i j, i ≠ j → ages i ≠ ages j ∧
  -- Difference between consecutive ages is 2
  ∀ i : Fin 4, ages i.succ = ages i + 2

theorem oldest_child_age (ages : Fin 5 → ℕ) (h : children_ages ages) :
  ages 0 = 10 := by
  sorry

end oldest_child_age_l1149_114959


namespace johns_extra_hours_l1149_114993

/-- Given John's work conditions, prove the number of extra hours he works for the bonus -/
theorem johns_extra_hours (regular_wage : ℝ) (regular_hours : ℝ) (bonus : ℝ) (bonus_hourly_rate : ℝ)
  (h1 : regular_wage = 80)
  (h2 : regular_hours = 8)
  (h3 : bonus = 20)
  (h4 : bonus_hourly_rate = 10) :
  (regular_wage + bonus) / bonus_hourly_rate - regular_hours = 2 := by
  sorry

end johns_extra_hours_l1149_114993


namespace monkey_climb_theorem_l1149_114909

/-- Calculates the time taken for a monkey to climb a tree given the tree height,
    distance hopped up per hour, and distance slipped back per hour. -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ :=
  let net_distance := hop_distance - slip_distance
  let full_climb_distance := tree_height - hop_distance
  full_climb_distance / net_distance + 1

theorem monkey_climb_theorem :
  monkey_climb_time 19 3 2 = 17 :=
sorry

end monkey_climb_theorem_l1149_114909


namespace stock_percentage_calculation_l1149_114914

/-- Calculates the percentage of a stock given its yield and quoted price. -/
theorem stock_percentage_calculation (yield : ℝ) (quote : ℝ) :
  yield = 10 →
  quote = 160 →
  let face_value := 100
  let market_price := quote * face_value / 100
  let annual_income := yield * face_value / 100
  let stock_percentage := annual_income / market_price * 100
  stock_percentage = 6.25 := by
  sorry

end stock_percentage_calculation_l1149_114914


namespace ticket_price_reduction_l1149_114956

theorem ticket_price_reduction (original_price : ℚ) 
  (h1 : original_price = 50)
  (h2 : ∃ (x : ℚ), x > 0 ∧ 
    (4/3 * x) * (original_price - 25/2) = (5/4) * (x * original_price)) :
  original_price - 25/2 = 46.875 := by
sorry

end ticket_price_reduction_l1149_114956


namespace base3_addition_theorem_l1149_114928

/-- Convert a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Convert a decimal number to its base 3 representation -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 3) ((m % 3) :: acc)
    go n []

theorem base3_addition_theorem :
  let a := [2]
  let b := [2, 2]
  let c := [2, 0, 2]
  let d := [2, 2, 0, 2]
  let result := [0, 1, 0, 1, 2]
  base3ToDecimal a + base3ToDecimal b + base3ToDecimal c + base3ToDecimal d =
  base3ToDecimal result := by
  sorry

end base3_addition_theorem_l1149_114928


namespace lionel_walked_four_miles_l1149_114937

-- Define the constants from the problem
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287
def total_feet : ℕ := 25332
def feet_per_yard : ℕ := 3
def feet_per_mile : ℕ := 5280

-- Define Lionel's distance in miles
def lionel_miles : ℚ := 4

-- Theorem statement
theorem lionel_walked_four_miles :
  (total_feet - (esther_yards * feet_per_yard + niklaus_feet)) / feet_per_mile = lionel_miles := by
  sorry

end lionel_walked_four_miles_l1149_114937


namespace roots_sum_and_product_l1149_114925

def absolute_value_equation (x : ℝ) : Prop :=
  |x|^3 + |x|^2 - 4*|x| - 12 = 0

theorem roots_sum_and_product :
  ∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, absolute_value_equation x) ∧
    (roots.sum id = 0) ∧
    (roots.prod id = -4) :=
sorry

end roots_sum_and_product_l1149_114925


namespace parabola_intersection_theorem_l1149_114900

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line
def line (x y k : ℝ) : Prop := y = k*x - 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k ∧
    x₁ > 0 ∧ y₁ > 0 ∧ x₂ > 0 ∧ y₂ > 0 ∧
    x₁ ≠ x₂

-- Define the distance ratio condition
def distance_ratio (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ((x₁ - 0)^2 + (y₁ - 1)^2).sqrt = 3 * ((x₂ - 0)^2 + (y₂ - 1)^2).sqrt

-- The theorem statement
theorem parabola_intersection_theorem (k : ℝ) :
  intersection_points k →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k ∧
    distance_ratio x₁ y₁ x₂ y₂) →
  k = (2 * Real.sqrt 3) / 3 :=
sorry

end parabola_intersection_theorem_l1149_114900


namespace sum_of_decimals_l1149_114917

theorem sum_of_decimals : 
  (5.76 : ℝ) + (4.29 : ℝ) = (10.05 : ℝ) := by sorry

end sum_of_decimals_l1149_114917


namespace triangle_inequalities_l1149_114906

/-- 
For any triangle ABC, we define:
- ha, hb, hc as the altitudes
- ra, rb, rc as the exradii
- r as the inradius
-/
theorem triangle_inequalities (A B C : Point) 
  (ha hb hc : ℝ) (ra rb rc : ℝ) (r : ℝ) :
  (ha > 0 ∧ hb > 0 ∧ hc > 0) →
  (ra > 0 ∧ rb > 0 ∧ rc > 0) →
  (r > 0) →
  (ha * hb * hc ≥ 27 * r^3) ∧ (ra * rb * rc ≥ 27 * r^3) := by
  sorry

end triangle_inequalities_l1149_114906


namespace bike_ride_problem_l1149_114999

/-- Bike ride problem -/
theorem bike_ride_problem (total_distance : ℝ) (total_time : ℝ) (rest_time : ℝ) 
  (fast_speed : ℝ) (slow_speed : ℝ) 
  (h1 : total_distance = 142)
  (h2 : total_time = 8)
  (h3 : rest_time = 0.5)
  (h4 : fast_speed = 22)
  (h5 : slow_speed = 15) :
  ∃ energetic_time : ℝ, 
    energetic_time * fast_speed + (total_time - rest_time - energetic_time) * slow_speed = total_distance ∧ 
    energetic_time = 59 / 14 := by
  sorry


end bike_ride_problem_l1149_114999


namespace tank_capacity_is_40_l1149_114934

/-- Represents the total capacity of a water tank in gallons. -/
def tank_capacity : ℝ := sorry

/-- The tank is initially 3/4 full of water. -/
axiom initial_fill : (3 / 4 : ℝ) * tank_capacity = tank_capacity - 5

/-- Adding 5 gallons of water makes the tank 7/8 full. -/
axiom after_adding : (7 / 8 : ℝ) * tank_capacity = tank_capacity

/-- The tank's total capacity is 40 gallons. -/
theorem tank_capacity_is_40 : tank_capacity = 40 := by sorry

end tank_capacity_is_40_l1149_114934


namespace tangent_line_at_point_2_neg6_l1149_114964

-- Define the function f(x) = x³ + x - 16
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_point_2_neg6 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := -6
  let m : ℝ := f' x₀
  (∀ x y, y - y₀ = m * (x - x₀) ↔ 13 * x - y - 32 = 0) ∧
  f x₀ = y₀ ∧
  m = 13 := by sorry

end tangent_line_at_point_2_neg6_l1149_114964


namespace marbles_lost_fraction_l1149_114987

theorem marbles_lost_fraction (initial_marbles : ℕ) (additional_marbles : ℕ) (new_marbles : ℕ) (final_marbles : ℕ)
  (h1 : initial_marbles = 12)
  (h2 : additional_marbles = 10)
  (h3 : new_marbles = 25)
  (h4 : final_marbles = 41) :
  (initial_marbles - (final_marbles - additional_marbles - new_marbles)) / initial_marbles = 1 / 2 := by
  sorry

end marbles_lost_fraction_l1149_114987


namespace bicycle_ride_average_speed_l1149_114986

/-- Prove that given an initial ride of 8 miles at 20 mph, riding an additional 16 miles at 40 mph 
    will result in an average speed of 30 mph for the entire trip. -/
theorem bicycle_ride_average_speed 
  (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_average_speed : ℝ)
  (additional_distance : ℝ) :
  initial_distance = 8 ∧ 
  initial_speed = 20 ∧ 
  second_speed = 40 ∧ 
  target_average_speed = 30 ∧
  additional_distance = 16 →
  (initial_distance + additional_distance) / 
    ((initial_distance / initial_speed) + (additional_distance / second_speed)) = 
  target_average_speed :=
by sorry

end bicycle_ride_average_speed_l1149_114986


namespace total_suitcases_l1149_114929

/-- The number of siblings in Lily's family -/
def num_siblings : Nat := 6

/-- The number of parents in Lily's family -/
def num_parents : Nat := 2

/-- The number of grandparents in Lily's family -/
def num_grandparents : Nat := 2

/-- The number of other relatives in Lily's family -/
def num_other_relatives : Nat := 3

/-- The number of suitcases each parent brings -/
def suitcases_per_parent : Nat := 3

/-- The number of suitcases each grandparent brings -/
def suitcases_per_grandparent : Nat := 2

/-- The total number of suitcases brought by other relatives -/
def suitcases_other_relatives : Nat := 8

/-- The sum of suitcases brought by siblings -/
def siblings_suitcases : Nat := (List.range num_siblings).sum.succ

/-- The total number of suitcases brought by Lily's family -/
theorem total_suitcases : 
  siblings_suitcases + 
  (num_parents * suitcases_per_parent) + 
  (num_grandparents * suitcases_per_grandparent) + 
  suitcases_other_relatives = 39 := by
  sorry

end total_suitcases_l1149_114929


namespace min_distance_between_lines_l1149_114919

/-- The minimum distance between a point on the line 3x+4y-12=0 and a point on the line 6x+8y+5=0 is 29/10 -/
theorem min_distance_between_lines : 
  let line1 := {(x, y) : ℝ × ℝ | 3 * x + 4 * y - 12 = 0}
  let line2 := {(x, y) : ℝ × ℝ | 6 * x + 8 * y + 5 = 0}
  ∃ (d : ℝ), d = 29/10 ∧ 
    ∀ (p q : ℝ × ℝ), p ∈ line1 → q ∈ line2 → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by
  sorry

end min_distance_between_lines_l1149_114919


namespace simplify_and_evaluate_l1149_114941

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) :
  5 * x^2 - 7 * x - (3 * x^2 - 2 * (-x^2 + 4 * x - 1)) = 1 := by
  sorry

end simplify_and_evaluate_l1149_114941


namespace sum_of_zeros_l1149_114911

/-- The parabola after transformations -/
def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 7

/-- The zeros of the transformed parabola -/
def zeros : Set ℝ := {x | transformed_parabola x = 0}

theorem sum_of_zeros : ∃ (a b : ℝ), a ∈ zeros ∧ b ∈ zeros ∧ a + b = 14 := by
  sorry

end sum_of_zeros_l1149_114911


namespace quadratic_function_properties_l1149_114942

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (h1 : f a b c (-1) = -1)
  (h2 : f a b c 0 = -7/4)
  (h3 : f a b c 1 = -2)
  (h4 : f a b c 2 = -7/4) :
  (f a b c 3 = -1) ∧
  (∀ x, f a b c x ≥ -2) ∧
  (f a b c 1 = -2) ∧
  (∀ x₁ x₂, -1 < x₁ → x₁ < 0 → 1 < x₂ → x₂ < 2 → f a b c x₁ > f a b c x₂) ∧
  (∀ x, 0 ≤ x → x ≤ 5 → -2 ≤ f a b c x ∧ f a b c x ≤ 2) :=
by sorry

end quadratic_function_properties_l1149_114942


namespace bad_oranges_l1149_114977

theorem bad_oranges (total_oranges : ℕ) (num_students : ℕ) (reduction : ℕ) : 
  total_oranges = 108 →
  num_students = 12 →
  reduction = 3 →
  (total_oranges / num_students - reduction) * num_students = total_oranges - 36 :=
by sorry

end bad_oranges_l1149_114977


namespace range_of_a_max_value_of_z_l1149_114916

-- Define the variables
variable (a b z : ℝ)

-- Define the conditions
def condition1 : Prop := 2 * a + b = 9
def condition2 : Prop := |9 - b| + |a| < 3
def condition3 : Prop := a > 0 ∧ b > 0
def condition4 : Prop := z = a^2 * b

-- Theorem for part (i)
theorem range_of_a (h1 : condition1 a b) (h2 : condition2 a b) : 
  -1 < a ∧ a < 1 := by sorry

-- Theorem for part (ii)
theorem max_value_of_z (h1 : condition1 a b) (h3 : condition3 a b) (h4 : condition4 a b z) : 
  z ≤ 27 := by sorry

end range_of_a_max_value_of_z_l1149_114916


namespace x_zero_not_necessary_nor_sufficient_l1149_114946

theorem x_zero_not_necessary_nor_sufficient :
  ¬(∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0) :=
by
  sorry

end x_zero_not_necessary_nor_sufficient_l1149_114946


namespace karls_drive_distance_l1149_114972

/-- Represents the problem of calculating Karl's total drive distance --/
theorem karls_drive_distance :
  -- Conditions
  let miles_per_gallon : ℝ := 35
  let tank_capacity : ℝ := 14
  let initial_drive : ℝ := 350
  let gas_bought : ℝ := 8
  let final_tank_fraction : ℝ := 1/2

  -- Definitions derived from conditions
  let initial_gas_used : ℝ := initial_drive / miles_per_gallon
  let remaining_gas_after_initial_drive : ℝ := tank_capacity - initial_gas_used
  let gas_after_refuel : ℝ := remaining_gas_after_initial_drive + gas_bought
  let final_gas : ℝ := tank_capacity * final_tank_fraction
  let gas_used_second_leg : ℝ := gas_after_refuel - final_gas
  let second_leg_distance : ℝ := gas_used_second_leg * miles_per_gallon
  let total_distance : ℝ := initial_drive + second_leg_distance

  -- Theorem statement
  total_distance = 525 := by
  sorry

end karls_drive_distance_l1149_114972


namespace count_multiples_of_12_and_9_l1149_114950

def count_multiples (lower upper divisor : ℕ) : ℕ :=
  (upper / divisor) - ((lower - 1) / divisor)

theorem count_multiples_of_12_and_9 : 
  count_multiples 50 400 (Nat.lcm 12 9) = 10 := by
  sorry

end count_multiples_of_12_and_9_l1149_114950


namespace water_evaporation_proof_l1149_114978

theorem water_evaporation_proof (initial_mass : ℝ) (initial_water_percentage : ℝ) 
  (final_water_percentage : ℝ) (evaporated_water : ℝ) : 
  initial_mass = 500 →
  initial_water_percentage = 0.85 →
  final_water_percentage = 0.75 →
  evaporated_water = 200 →
  (initial_mass * initial_water_percentage - evaporated_water) / (initial_mass - evaporated_water) = final_water_percentage :=
by
  sorry

end water_evaporation_proof_l1149_114978


namespace vector_magnitude_problem_l1149_114920

/-- Given vectors a and b in ℝ², prove that the magnitude of 2a + b equals 4. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (ha : a = (3, -3)) (hb : b = (-2, 6)) : 
  ‖(2 : ℝ) • a + b‖ = 4 := by
  sorry

end vector_magnitude_problem_l1149_114920


namespace urn_probability_theorem_l1149_114982

/-- The number of blue balls in the second urn -/
def M : ℝ := 7.4

/-- The probability of drawing two balls of the same color -/
def same_color_probability : ℝ := 0.65

/-- The number of green balls in the first urn -/
def green_balls_urn1 : ℕ := 3

/-- The number of blue balls in the first urn -/
def blue_balls_urn1 : ℕ := 7

/-- The number of green balls in the second urn -/
def green_balls_urn2 : ℕ := 20

theorem urn_probability_theorem :
  (green_balls_urn1 / (green_balls_urn1 + blue_balls_urn1 : ℝ)) * (green_balls_urn2 / (green_balls_urn2 + M : ℝ)) +
  (blue_balls_urn1 / (green_balls_urn1 + blue_balls_urn1 : ℝ)) * (M / (green_balls_urn2 + M : ℝ)) =
  same_color_probability :=
by sorry

end urn_probability_theorem_l1149_114982


namespace power_17_mod_28_l1149_114944

theorem power_17_mod_28 : 17^2023 % 28 = 17 := by
  sorry

end power_17_mod_28_l1149_114944


namespace find_m_l1149_114995

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | (x : ℝ)^2 + m * x = 0}

theorem find_m :
  ∃ m : ℝ, (U \ A m = {1, 2}) → m = -3 := by
  sorry

end find_m_l1149_114995


namespace unique_number_from_dialogue_l1149_114988

/-- Represents a two-digit natural number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the number of divisors of a natural number -/
def numberOfDivisors (n : ℕ) : ℕ := sorry

/-- Checks if the number satisfies the dialogue conditions -/
def satisfiesDialogueConditions (n : ℕ) : Prop :=
  TwoDigitNumber n ∧
  (∀ m : ℕ, TwoDigitNumber m → sumOfDigits m = sumOfDigits n → m ≠ n) ∧
  (numberOfDivisors n ≠ 2 ∧ numberOfDivisors n ≠ 12) ∧
  (∀ m : ℕ, TwoDigitNumber m → 
    sumOfDigits m = sumOfDigits n → 
    numberOfDivisors m = numberOfDivisors n → 
    m = n)

theorem unique_number_from_dialogue :
  ∃! n : ℕ, satisfiesDialogueConditions n ∧ n = 30 := by
  sorry

end unique_number_from_dialogue_l1149_114988


namespace zigzag_angle_theorem_l1149_114951

/-- A structure representing a zigzag line in a rectangle --/
structure ZigzagRectangle where
  ACB : ℝ
  FEG : ℝ
  DCE : ℝ
  DEC : ℝ

/-- The theorem stating that given specific angle measurements in a zigzag rectangle,
    the angle θ formed by the zigzag line is equal to 11 degrees --/
theorem zigzag_angle_theorem (z : ZigzagRectangle) 
  (h1 : z.ACB = 10)
  (h2 : z.FEG = 26)
  (h3 : z.DCE = 14)
  (h4 : z.DEC = 33) :
  ∃ θ : ℝ, θ = 11 := by
  sorry

end zigzag_angle_theorem_l1149_114951


namespace committee_arrangement_l1149_114939

theorem committee_arrangement (n m : ℕ) (hn : n = 7) (hm : m = 3) :
  (Nat.choose (n + m) m) = 120 := by
  sorry

end committee_arrangement_l1149_114939


namespace f_16_values_l1149_114902

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 3 * f (a^2 + b^2) = 2 * (f a)^2 + 2 * (f b)^2 - f a * f b

theorem f_16_values (f : ℕ → ℕ) (h : is_valid_f f) : 
  {n : ℕ | f 16 = n} = {0, 2} := by
  sorry

end f_16_values_l1149_114902


namespace answer_key_problem_l1149_114923

theorem answer_key_problem (total_ways : ℕ) (tf_questions : ℕ) (mc_questions : ℕ) : 
  total_ways = 384 → 
  tf_questions = 3 → 
  mc_questions = 3 → 
  (∃ (n : ℕ), total_ways = 6 * n^mc_questions) →
  (∃ (n : ℕ), n = 4 ∧ total_ways = 6 * n^mc_questions) := by
sorry

end answer_key_problem_l1149_114923


namespace grass_seed_bags_l1149_114936

theorem grass_seed_bags (lawn_length lawn_width coverage_per_bag extra_coverage : ℕ) 
  (h1 : lawn_length = 22)
  (h2 : lawn_width = 36)
  (h3 : coverage_per_bag = 250)
  (h4 : extra_coverage = 208) :
  (lawn_length * lawn_width + extra_coverage) / coverage_per_bag = 4 := by
  sorry

end grass_seed_bags_l1149_114936


namespace right_triangle_acute_angle_theorem_l1149_114962

theorem right_triangle_acute_angle_theorem :
  ∀ (a b : ℝ), 
  a > 0 ∧ b > 0 →  -- Ensuring positive angles
  a = 2 * b →      -- One acute angle is twice the other
  a + b = 90 →     -- Sum of acute angles in a right triangle is 90°
  a = 60 :=        -- The larger acute angle is 60°
by sorry

end right_triangle_acute_angle_theorem_l1149_114962


namespace construction_work_proof_l1149_114908

/-- Represents the number of men who dropped out -/
def men_dropped_out : ℕ := 1

theorem construction_work_proof :
  let initial_men : ℕ := 5
  let half_job_days : ℕ := 15
  let full_job_days : ℕ := 30
  let completion_days : ℕ := 25
  (initial_men * full_job_days : ℚ) = ((initial_men - men_dropped_out) * completion_days : ℚ) :=
by sorry

end construction_work_proof_l1149_114908


namespace wire_cutting_l1149_114949

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 14 →
  ratio = 2 / 5 →
  shorter_length + ratio * shorter_length = total_length →
  shorter_length = 4 := by
sorry

end wire_cutting_l1149_114949


namespace highway_distance_is_4km_l1149_114935

/-- Represents the travel scenario between two points A and B -/
structure TravelScenario where
  highway_speed : ℝ
  path_speed : ℝ
  time_difference : ℝ
  distance_difference : ℝ

/-- The distance from A to B along the highway given the travel scenario -/
def highway_distance (scenario : TravelScenario) : ℝ :=
  scenario.path_speed * scenario.time_difference

/-- Theorem stating that for the given scenario, the highway distance is 4 km -/
theorem highway_distance_is_4km (scenario : TravelScenario) 
  (h1 : scenario.highway_speed = 5)
  (h2 : scenario.path_speed = 4)
  (h3 : scenario.time_difference = 1)
  (h4 : scenario.distance_difference = 6) :
  highway_distance scenario = 4 := by
  sorry

#eval highway_distance { highway_speed := 5, path_speed := 4, time_difference := 1, distance_difference := 6 }

end highway_distance_is_4km_l1149_114935


namespace math_competition_score_ratio_l1149_114901

theorem math_competition_score_ratio :
  let sammy_score : ℕ := 20
  let gab_score : ℕ := 2 * sammy_score
  let opponent_score : ℕ := 85
  let total_score : ℕ := opponent_score + 55
  let cher_score : ℕ := total_score - (sammy_score + gab_score)
  (cher_score : ℚ) / (gab_score : ℚ) = 2 / 1 :=
by sorry

end math_competition_score_ratio_l1149_114901


namespace distance_between_locations_l1149_114980

theorem distance_between_locations (speed_A speed_B : ℝ) (time : ℝ) (remaining_fraction : ℝ) : 
  speed_A = 60 →
  speed_B = 45 →
  time = 2 →
  remaining_fraction = 2 / 5 →
  (speed_A + speed_B) * time / (1 - remaining_fraction) = 350 :=
by
  sorry

end distance_between_locations_l1149_114980


namespace cyclic_fraction_sum_l1149_114905

theorem cyclic_fraction_sum (x y z w t : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ z ≠ w ∧ w ≠ x)
  (h_eq : x + 1/y = t ∧ y + 1/z = t ∧ z + 1/w = t ∧ w + 1/x = t) :
  t = Real.sqrt 2 := by
sorry

end cyclic_fraction_sum_l1149_114905


namespace alloy_interchange_mass_l1149_114985

theorem alloy_interchange_mass (m₁ m₂ x : ℝ) : 
  m₁ = 6 →
  m₂ = 12 →
  0 < x →
  x < m₁ →
  x < m₂ →
  x / m₁ = (m₂ - x) / m₂ →
  x = 4 := by
  sorry

end alloy_interchange_mass_l1149_114985


namespace average_weight_problem_l1149_114961

/-- Given the average weight of three people and two of them, prove the average weight of two of them. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →  -- The average weight of a, b, and c is 43 kg
  (a + b) / 2 = 40 →      -- The average weight of a and b is 40 kg
  b = 37 →                -- The weight of b is 37 kg
  (b + c) / 2 = 43        -- The average weight of b and c is 43 kg
  := by sorry

end average_weight_problem_l1149_114961


namespace revenue_change_l1149_114904

theorem revenue_change 
  (original_price original_quantity : ℝ) 
  (price_increase : ℝ) 
  (quantity_decrease : ℝ) 
  (h1 : price_increase = 0.75) 
  (h2 : quantity_decrease = 0.45) : 
  let new_price := original_price * (1 + price_increase)
  let new_quantity := original_quantity * (1 - quantity_decrease)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue = -0.0375 := by
sorry

end revenue_change_l1149_114904


namespace max_discarded_grapes_l1149_114915

theorem max_discarded_grapes (n : ℕ) : ∃ (q : ℕ), n = 8 * q + (n % 8) ∧ n % 8 ≤ 7 := by
  sorry

end max_discarded_grapes_l1149_114915


namespace sarah_shirts_l1149_114943

/-- The total number of shirts Sarah owns after buying new shirts -/
theorem sarah_shirts (initial_shirts new_shirts : ℕ) 
  (h1 : initial_shirts = 9)
  (h2 : new_shirts = 8) : 
  initial_shirts + new_shirts = 17 := by
  sorry

end sarah_shirts_l1149_114943


namespace bounds_of_abs_diff_over_sum_l1149_114921

theorem bounds_of_abs_diff_over_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m M : ℝ),
    (∀ z, z = |x - y| / (|x| + |y|) → m ≤ z ∧ z ≤ M) ∧
    m = 0 ∧ M = 1 ∧ M - m = 1 :=
by sorry

end bounds_of_abs_diff_over_sum_l1149_114921


namespace exponent_problem_l1149_114970

theorem exponent_problem (a m n : ℕ) (h1 : a^m = 3) (h2 : a^n = 5) : a^(2*m + n) = 45 := by
  sorry

end exponent_problem_l1149_114970


namespace algebra_test_average_l1149_114912

theorem algebra_test_average (total_average : ℝ) (male_count : ℕ) (female_average : ℝ) (female_count : ℕ) :
  total_average = 90 →
  male_count = 8 →
  female_average = 92 →
  female_count = 32 →
  (total_average * (male_count + female_count) - female_average * female_count) / male_count = 82 := by
  sorry

end algebra_test_average_l1149_114912


namespace cloth_sale_problem_l1149_114966

/-- Proves that the number of metres of cloth sold is 200 given the specified conditions -/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) : 
  total_selling_price = 12000 →
  loss_per_metre = 12 →
  cost_price_per_metre = 72 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 200 :=
by
  sorry

end cloth_sale_problem_l1149_114966


namespace water_purification_equation_l1149_114948

/-- Represents the water purification scenario -/
structure WaterPurification where
  total_area : ℝ
  efficiency_increase : ℝ
  days_saved : ℝ
  daily_rate : ℝ

/-- Theorem stating the correct equation for the water purification scenario -/
theorem water_purification_equation (wp : WaterPurification) 
  (h1 : wp.total_area = 2400)
  (h2 : wp.efficiency_increase = 0.2)
  (h3 : wp.days_saved = 40)
  (h4 : wp.daily_rate > 0) :
  (wp.total_area * (1 + wp.efficiency_increase)) / wp.daily_rate - wp.total_area / wp.daily_rate = wp.days_saved :=
by sorry

end water_purification_equation_l1149_114948
