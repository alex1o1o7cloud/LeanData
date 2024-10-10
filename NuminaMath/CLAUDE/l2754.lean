import Mathlib

namespace farm_water_consumption_l2754_275404

/-- Calculates the total weekly water consumption for Mr. Reyansh's farm animals -/
theorem farm_water_consumption : 
  let num_cows : ℕ := 40
  let num_goats : ℕ := 25
  let num_pigs : ℕ := 30
  let cow_water : ℕ := 80
  let goat_water : ℕ := cow_water / 2
  let pig_water : ℕ := cow_water / 3
  let num_sheep : ℕ := num_cows * 10
  let sheep_water : ℕ := cow_water / 4
  let daily_consumption : ℕ := 
    num_cows * cow_water + 
    num_goats * goat_water + 
    num_pigs * pig_water + 
    num_sheep * sheep_water
  let weekly_consumption : ℕ := daily_consumption * 7
  weekly_consumption = 91000 := by
  sorry

end farm_water_consumption_l2754_275404


namespace perpendicular_line_through_point_perpendicular_line_equation_l2754_275429

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def passes_through (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_line_through_point 
  (P : Point2D)
  (given_line : Line2D)
  (result_line : Line2D) : Prop :=
  P.x = -1 ∧ 
  P.y = 3 ∧ 
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 3 ∧
  result_line.a = 2 ∧ 
  result_line.b = 1 ∧ 
  result_line.c = -1 ∧
  perpendicular given_line result_line ∧
  passes_through result_line P ∧
  ∀ (other_line : Line2D), 
    perpendicular given_line other_line ∧ 
    passes_through other_line P → 
    other_line = result_line

theorem perpendicular_line_equation : 
  perpendicular_line_through_point 
    (Point2D.mk (-1) 3) 
    (Line2D.mk 1 (-2) 3) 
    (Line2D.mk 2 1 (-1)) := by
  sorry

end perpendicular_line_through_point_perpendicular_line_equation_l2754_275429


namespace pyramid_volume_l2754_275435

/-- The volume of a square-based pyramid with given dimensions -/
theorem pyramid_volume (base_side : ℝ) (apex_distance : ℝ) (volume : ℝ) : 
  base_side = 24 →
  apex_distance = Real.sqrt 364 →
  volume = (1 / 3) * base_side^2 * Real.sqrt ((apex_distance^2) - (1/2 * base_side * Real.sqrt 2)^2) →
  volume = 384 * Real.sqrt 19 :=
by sorry

end pyramid_volume_l2754_275435


namespace max_NF_value_slope_AN_l2754_275496

-- Define the ellipse parameters
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
axiom major_minor_ratio : 2*a = (3*Real.sqrt 5 / 5) * (2*b)
axiom point_D_on_ellipse : ellipse a b (-1) (2*Real.sqrt 10 / 3)
axiom vector_relation (x₀ y₀ x₁ y₁ : ℝ) (hx₀ : x₀ > 0) (hy₀ : y₀ > 0) :
  ellipse a b x₀ y₀ → ellipse a b x₁ y₁ → (x₀, y₀) = 2 * (x₁ + 3, y₁)

-- State the theorems to be proved
theorem max_NF_value :
  ∃ (c : ℝ), c^2 = a^2 - b^2 ∧ a + c = 5 :=
sorry

theorem slope_AN :
  ∃ (x₀ y₀ : ℝ), ellipse a b x₀ y₀ ∧ x₀ > 0 ∧ y₀ > 0 ∧ y₀ / x₀ = 5 * Real.sqrt 3 / 3 :=
sorry

end max_NF_value_slope_AN_l2754_275496


namespace sweater_cost_l2754_275494

/-- Given shopping information, prove the cost of a sweater --/
theorem sweater_cost (initial_amount : ℕ) (tshirt_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ)
  (h1 : initial_amount = 91)
  (h2 : tshirt_cost = 6)
  (h3 : shoes_cost = 11)
  (h4 : remaining_amount = 50) :
  initial_amount - remaining_amount - tshirt_cost - shoes_cost = 24 := by
  sorry

end sweater_cost_l2754_275494


namespace tax_rate_calculation_l2754_275401

/-- Represents the tax calculation for a citizen in Country X --/
def TaxCalculation (income : ℝ) (totalTax : ℝ) (baseRate : ℝ) (baseIncome : ℝ) : Prop :=
  income > baseIncome ∧
  totalTax = baseRate * baseIncome + 
    ((income - baseIncome) * (totalTax - baseRate * baseIncome) / (income - baseIncome))

/-- Theorem statement for the tax calculation problem --/
theorem tax_rate_calculation :
  ∀ (income : ℝ) (totalTax : ℝ),
  TaxCalculation income totalTax 0.15 40000 →
  income = 50000 →
  totalTax = 8000 →
  (totalTax - 0.15 * 40000) / (income - 40000) = 0.20 := by
  sorry


end tax_rate_calculation_l2754_275401


namespace differential_equation_classification_l2754_275441

-- Define a type for equations
inductive Equation
| A : Equation  -- y' + 3x = 0
| B : Equation  -- y² + x² = 5
| C : Equation  -- y = e^x
| D : Equation  -- y = ln|x| + C
| E : Equation  -- y'y - x = 0
| F : Equation  -- 2dy + 3xdx = 0

-- Define a predicate for differential equations
def isDifferentialEquation : Equation → Prop
| Equation.A => True
| Equation.B => False
| Equation.C => False
| Equation.D => False
| Equation.E => True
| Equation.F => True

-- Theorem statement
theorem differential_equation_classification :
  (isDifferentialEquation Equation.A ∧
   isDifferentialEquation Equation.E ∧
   isDifferentialEquation Equation.F) ∧
  (¬isDifferentialEquation Equation.B ∧
   ¬isDifferentialEquation Equation.C ∧
   ¬isDifferentialEquation Equation.D) :=
by sorry

end differential_equation_classification_l2754_275441


namespace violet_necklace_problem_l2754_275419

theorem violet_necklace_problem (x : ℝ) 
  (h1 : (1/2 : ℝ) * x + 30 = (3/4 : ℝ) * x) : 
  (1/4 : ℝ) * x = 30 := by
  sorry

end violet_necklace_problem_l2754_275419


namespace jane_max_tickets_l2754_275432

/-- The maximum number of tickets that can be bought with a given budget, 
    given a regular price, discounted price, and discount threshold. -/
def maxTickets (budget : ℕ) (regularPrice discountPrice : ℕ) (discountThreshold : ℕ) : ℕ :=
  let regularTickets := budget / regularPrice
  let discountedTotal := 
    discountThreshold * regularPrice + 
    (budget - discountThreshold * regularPrice) / discountPrice
  max regularTickets discountedTotal

/-- Theorem: Given the specific conditions of the problem, 
    the maximum number of tickets Jane can buy is 11. -/
theorem jane_max_tickets : 
  maxTickets 150 15 12 5 = 11 := by
  sorry

end jane_max_tickets_l2754_275432


namespace tangent_slope_exponential_l2754_275481

theorem tangent_slope_exponential (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.exp x
  (deriv f) 1 = Real.exp 1 := by sorry

end tangent_slope_exponential_l2754_275481


namespace distinct_fraction_equality_l2754_275456

theorem distinct_fraction_equality (a b c : ℝ) (k : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a / (1 + b) = k ∧ b / (1 + c) = k ∧ c / (1 + a) = k →
  k = -1 :=
sorry

end distinct_fraction_equality_l2754_275456


namespace pigeons_flew_in_l2754_275482

theorem pigeons_flew_in (initial_count final_count : ℕ) 
  (h_initial : initial_count = 15)
  (h_final : final_count = 21) :
  final_count - initial_count = 6 := by
  sorry

end pigeons_flew_in_l2754_275482


namespace system_solution_l2754_275464

theorem system_solution (x y b : ℝ) : 
  (4 * x + y = b) → 
  (3 * x + 4 * y = 3 * b) → 
  (x = 3) → 
  (b = 39) := by
sorry

end system_solution_l2754_275464


namespace units_digit_factorial_25_l2754_275428

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem units_digit_factorial_25 : factorial 25 % 10 = 0 := by
  sorry

end units_digit_factorial_25_l2754_275428


namespace cone_slant_height_l2754_275479

theorem cone_slant_height (V : ℝ) (θ : ℝ) (l : ℝ) : 
  V = 9 * Real.pi → θ = Real.pi / 4 → l = 3 * Real.sqrt 2 :=
by sorry

end cone_slant_height_l2754_275479


namespace smallest_n_with_perfect_square_sum_l2754_275450

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def partition_with_perfect_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ),
    (A ∪ B = Finset.range n) →
    (A ∩ B = ∅) →
    (A ≠ ∅) →
    (B ≠ ∅) →
    (∃ (x y : ℕ), (x ∈ A ∧ y ∈ A ∧ is_perfect_square (x + y)) ∨
                  (x ∈ B ∧ y ∈ B ∧ is_perfect_square (x + y)))

theorem smallest_n_with_perfect_square_sum : 
  (∀ k < 15, ¬ partition_with_perfect_square_sum k) ∧ 
  partition_with_perfect_square_sum 15 :=
sorry

end smallest_n_with_perfect_square_sum_l2754_275450


namespace floor_painting_problem_l2754_275402

def is_valid_pair (a b : ℕ) : Prop :=
  b > a ∧ 
  (a - 4) * (b - 4) = 2 * a * b / 3 ∧
  a > 4 ∧ b > 4

theorem floor_painting_problem :
  ∃! (pairs : List (ℕ × ℕ)), 
    pairs.length = 3 ∧ 
    ∀ (p : ℕ × ℕ), p ∈ pairs ↔ is_valid_pair p.1 p.2 :=
by sorry

end floor_painting_problem_l2754_275402


namespace price_conditions_max_basketballs_l2754_275436

/-- Represents the price of a basketball -/
def basketball_price : ℕ := 80

/-- Represents the price of a soccer ball -/
def soccer_price : ℕ := 50

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 60

/-- The maximum allowed total cost -/
def max_cost : ℕ := 4000

/-- Verifies that the prices satisfy the given conditions -/
theorem price_conditions : 
  2 * basketball_price + 3 * soccer_price = 310 ∧
  5 * basketball_price + 2 * soccer_price = 500 := by sorry

/-- Proves that the maximum number of basketballs that can be purchased is 33 -/
theorem max_basketballs :
  ∀ m : ℕ, 
    m ≤ total_balls ∧ 
    m * basketball_price + (total_balls - m) * soccer_price ≤ max_cost →
    m ≤ 33 := by sorry

end price_conditions_max_basketballs_l2754_275436


namespace balance_after_school_days_l2754_275412

/-- Represents the balance after spending money for a certain number of days. -/
def balance (initial_balance : ℝ) (daily_spending : ℝ) (days : ℝ) : ℝ :=
  initial_balance - daily_spending * days

/-- Theorem stating the relationship between balance and days spent at school. -/
theorem balance_after_school_days 
  (initial_balance : ℝ) 
  (daily_spending : ℝ) 
  (days : ℝ) 
  (h1 : initial_balance = 200)
  (h2 : daily_spending = 36)
  (h3 : 0 ≤ days)
  (h4 : days ≤ 5) :
  balance initial_balance daily_spending days = 200 - 36 * days :=
by sorry

end balance_after_school_days_l2754_275412


namespace fixed_point_of_exponential_function_l2754_275452

/-- The fixed point of the function f(x) = 2a^(x+1) - 3, where a > 0 and a ≠ 1, is (-1, -1). -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x + 1) - 3
  f (-1) = -1 ∧ ∀ x : ℝ, f x = x → x = -1 := by
  sorry

end fixed_point_of_exponential_function_l2754_275452


namespace cyclic_sum_nonnegative_l2754_275446

theorem cyclic_sum_nonnegative (r : ℝ) (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x^r * (x - y) * (x - z) + y^r * (y - x) * (y - z) + z^r * (z - x) * (z - y) ≥ 0 :=
by sorry

end cyclic_sum_nonnegative_l2754_275446


namespace num_connected_subsets_2x1_l2754_275414

/-- A rectangle in the Cartesian plane -/
structure Rectangle :=
  (bottomLeft : ℝ × ℝ)
  (topRight : ℝ × ℝ)

/-- An edge of a rectangle -/
inductive Edge
  | BottomLeft
  | BottomRight
  | TopLeft
  | TopRight
  | Left
  | Middle
  | Right

/-- A subset of edges -/
def EdgeSubset := Set Edge

/-- Predicate to determine if a subset of edges is connected -/
def is_connected (s : EdgeSubset) : Prop := sorry

/-- The number of connected subsets of edges in a 2x1 rectangle divided into two unit squares -/
def num_connected_subsets (r : Rectangle) : ℕ := sorry

/-- Theorem stating that the number of connected subsets is 81 -/
theorem num_connected_subsets_2x1 :
  ∀ r : Rectangle,
  r.bottomLeft = (0, 0) ∧ r.topRight = (2, 1) →
  num_connected_subsets r = 81 :=
sorry

end num_connected_subsets_2x1_l2754_275414


namespace value_of_expression_l2754_275497

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem value_of_expression (a b c d : ℝ) :
  f a b c d (-2) = -3 → 10 * a - 5 * b + 3 * c - 2 * d = 12 := by
  sorry

end value_of_expression_l2754_275497


namespace hexagon_percentage_l2754_275466

-- Define the tiling structure
structure Tiling where
  smallSquareArea : ℝ
  largeSquareArea : ℝ
  hexagonArea : ℝ
  smallSquaresPerLarge : ℕ
  hexagonsPerLarge : ℕ
  smallSquaresInHexagons : ℝ

-- Define the tiling conditions
def tilingConditions (t : Tiling) : Prop :=
  t.smallSquaresPerLarge = 16 ∧
  t.hexagonsPerLarge = 3 ∧
  t.largeSquareArea = 16 * t.smallSquareArea ∧
  t.hexagonArea = 2 * t.smallSquareArea ∧
  t.smallSquaresInHexagons = 3 * t.hexagonArea

-- Theorem to prove
theorem hexagon_percentage (t : Tiling) (h : tilingConditions t) :
  (t.smallSquaresInHexagons / t.largeSquareArea) * 100 = 37.5 := by
  sorry

end hexagon_percentage_l2754_275466


namespace g_of_three_equals_fourteen_l2754_275485

-- Define the function g
def g : ℝ → ℝ := fun x ↦ 2 * x + 4

-- State the theorem
theorem g_of_three_equals_fourteen : g 3 = 14 := by
  sorry

end g_of_three_equals_fourteen_l2754_275485


namespace solution_set_of_inequality_l2754_275492

-- Define the inequality function
def f (x : ℝ) := (x - 2) * (x + 1)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x < 0} = Set.Ioo (-1 : ℝ) 2 := by
  sorry

end solution_set_of_inequality_l2754_275492


namespace arithmetic_expression_equality_l2754_275471

theorem arithmetic_expression_equality : 
  8.1 * 1.3 + 8 / 1.3 + 1.9 * 1.3 - 11.9 / 1.3 = 16 := by
  sorry

end arithmetic_expression_equality_l2754_275471


namespace change_calculation_l2754_275473

def egg_cost : ℕ := 3
def pancake_cost : ℕ := 2
def cocoa_cost : ℕ := 2
def tax : ℕ := 1
def initial_order_cost : ℕ := egg_cost + pancake_cost + 2 * cocoa_cost + tax
def additional_order_cost : ℕ := pancake_cost + cocoa_cost
def total_paid : ℕ := 15

theorem change_calculation :
  total_paid - (initial_order_cost + additional_order_cost) = 1 := by
  sorry

end change_calculation_l2754_275473


namespace basketball_game_total_points_l2754_275430

theorem basketball_game_total_points : 
  ∀ (adam_2pt adam_3pt mada_2pt mada_3pt : ℕ),
    adam_2pt + adam_3pt = 10 →
    mada_2pt + mada_3pt = 11 →
    adam_2pt = mada_3pt →
    2 * adam_2pt + 3 * adam_3pt = 3 * mada_3pt + 2 * mada_2pt →
    2 * adam_2pt + 3 * adam_3pt + 3 * mada_3pt + 2 * mada_2pt = 52 :=
by sorry

end basketball_game_total_points_l2754_275430


namespace saddle_value_l2754_275484

theorem saddle_value (total_value : ℝ) (horse_saddle_ratio : ℝ) :
  total_value = 100 →
  horse_saddle_ratio = 7 →
  ∃ (saddle_value : ℝ),
    saddle_value + horse_saddle_ratio * saddle_value = total_value ∧
    saddle_value = 12.5 := by
  sorry

end saddle_value_l2754_275484


namespace hyperbola_circle_eccentricity_l2754_275408

/-- Given a hyperbola with equation x^2 - ny^2 = 1 and eccentricity 2, 
    prove that the eccentricity of the circle x^2 + ny^2 = 1 is √6/3 -/
theorem hyperbola_circle_eccentricity (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_hyperbola_ecc : (m⁻¹ + n⁻¹) / m⁻¹ = 4) :
  Real.sqrt ((n⁻¹ - m⁻¹) / n⁻¹) = Real.sqrt 6 / 3 := by
  sorry

end hyperbola_circle_eccentricity_l2754_275408


namespace dinner_bill_proof_l2754_275475

theorem dinner_bill_proof (total_friends : Nat) (paying_friends : Nat) (extra_payment : Real) (total_bill : Real) : 
  total_friends = 10 →
  paying_friends = 9 →
  extra_payment = 3 →
  paying_friends * (total_bill / total_friends + extra_payment) = total_bill →
  total_bill = 270 := by
sorry

end dinner_bill_proof_l2754_275475


namespace jakes_third_test_score_l2754_275455

/-- Proof that Jake scored 65 marks in the third test given the conditions -/
theorem jakes_third_test_score :
  ∀ (third_test_score : ℕ),
  (∃ (first_test : ℕ) (second_test : ℕ) (fourth_test : ℕ),
    first_test = 80 ∧
    second_test = first_test + 10 ∧
    fourth_test = third_test_score ∧
    (first_test + second_test + third_test_score + fourth_test) / 4 = 75) →
  third_test_score = 65 := by
sorry

end jakes_third_test_score_l2754_275455


namespace quadratic_equation_solution_l2754_275480

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x * (x + 1) = 3 * (x + 1) ↔ x = x₁ ∨ x = x₂) ∧ x₁ = -1 ∧ x₂ = 3 := by
  sorry

end quadratic_equation_solution_l2754_275480


namespace shortest_side_length_l2754_275421

/-- Represents a triangle with angles in the ratio 1:2:3 and longest side of length 6 -/
structure SpecialTriangle where
  /-- The smallest angle of the triangle -/
  smallest_angle : ℝ
  /-- The ratio of angles is 1:2:3 -/
  angle_ratio : smallest_angle > 0 ∧ smallest_angle + 2 * smallest_angle + 3 * smallest_angle = 180
  /-- The length of the longest side is 6 -/
  longest_side : ℝ
  longest_side_eq : longest_side = 6

/-- The length of the shortest side in a SpecialTriangle is 3 -/
theorem shortest_side_length (t : SpecialTriangle) : ∃ (shortest_side : ℝ), shortest_side = 3 := by
  sorry

end shortest_side_length_l2754_275421


namespace tangent_slope_at_zero_l2754_275422

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - 2*x

theorem tangent_slope_at_zero :
  deriv f 0 = -1 := by sorry

end tangent_slope_at_zero_l2754_275422


namespace two_intersection_points_l2754_275474

/-- Represents a line in the 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a * X + b * Y = c

/-- Represents a point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- The three lines given in the problem --/
def line1 : Line := ⟨4, -3, 2, by sorry⟩
def line2 : Line := ⟨1, 3, 3, by sorry⟩
def line3 : Line := ⟨3, -4, 3, by sorry⟩

/-- Theorem stating that there are exactly two distinct intersection points --/
theorem two_intersection_points : 
  ∃! (p1 p2 : Point), 
    (pointOnLine p1 line1 ∧ pointOnLine p1 line2) ∨ 
    (pointOnLine p1 line1 ∧ pointOnLine p1 line3) ∨ 
    (pointOnLine p1 line2 ∧ pointOnLine p1 line3) ∧
    (pointOnLine p2 line1 ∧ pointOnLine p2 line2) ∨ 
    (pointOnLine p2 line1 ∧ pointOnLine p2 line3) ∨ 
    (pointOnLine p2 line2 ∧ pointOnLine p2 line3) ∧
    p1 ≠ p2 :=
  sorry

end two_intersection_points_l2754_275474


namespace money_distribution_l2754_275415

theorem money_distribution (a b c : ℤ) 
  (total : a + b + c = 500)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 340) :
  c = 40 := by
  sorry

end money_distribution_l2754_275415


namespace factor_expression_l2754_275459

theorem factor_expression (x y a b : ℝ) :
  3 * x * (a - b) - 9 * y * (b - a) = 3 * (a - b) * (x + 3 * y) := by
  sorry

end factor_expression_l2754_275459


namespace inequality_proof_l2754_275483

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end inequality_proof_l2754_275483


namespace f_composition_negative_four_l2754_275491

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x else (1/2)^x

theorem f_composition_negative_four : f (f (-4)) = 4 := by
  sorry

end f_composition_negative_four_l2754_275491


namespace slant_base_angle_is_36_degrees_l2754_275431

/-- A regular pentagonal pyramid where the slant height is equal to the base edge -/
structure RegularPentagonalPyramid where
  /-- The base of the pyramid is a regular pentagon -/
  base : RegularPentagon
  /-- The slant height of the pyramid -/
  slant_height : ℝ
  /-- The base edge of the pyramid -/
  base_edge : ℝ
  /-- The slant height is equal to the base edge -/
  slant_height_eq_base_edge : slant_height = base_edge

/-- The angle between a slant height and a non-intersecting, non-perpendicular base edge -/
def slant_base_angle (p : RegularPentagonalPyramid) : Angle := sorry

/-- Theorem: The angle between a slant height and a non-intersecting, non-perpendicular base edge is 36° -/
theorem slant_base_angle_is_36_degrees (p : RegularPentagonalPyramid) :
  slant_base_angle p = 36 * π / 180 := by sorry

end slant_base_angle_is_36_degrees_l2754_275431


namespace alcohol_water_ratio_l2754_275465

theorem alcohol_water_ratio (alcohol_fraction water_fraction : ℚ) 
  (h1 : alcohol_fraction = 3/5)
  (h2 : water_fraction = 2/5)
  (h3 : alcohol_fraction + water_fraction = 1) :
  alcohol_fraction / water_fraction = 3/2 := by
  sorry

end alcohol_water_ratio_l2754_275465


namespace mass_B13N3O12H12_value_l2754_275405

/-- The mass in grams of 12 moles of Trinitride dodecahydroxy tridecaborate (B13N3O12H12) -/
def mass_B13N3O12H12 : ℝ :=
  let atomic_mass_B : ℝ := 10.81
  let atomic_mass_N : ℝ := 14.01
  let atomic_mass_O : ℝ := 16.00
  let atomic_mass_H : ℝ := 1.01
  let molar_mass : ℝ := 13 * atomic_mass_B + 3 * atomic_mass_N + 12 * atomic_mass_O + 12 * atomic_mass_H
  12 * molar_mass

/-- Theorem stating that the mass of 12 moles of B13N3O12H12 is 4640.16 grams -/
theorem mass_B13N3O12H12_value : mass_B13N3O12H12 = 4640.16 := by
  sorry

end mass_B13N3O12H12_value_l2754_275405


namespace volume_formula_correct_l2754_275425

/-- A solid formed by the union of a sphere and a truncated cone -/
structure SphereConeUnion where
  R : ℝ  -- radius of the sphere
  S : ℝ  -- total surface area of the solid

/-- The sphere is tangent to one base of the truncated cone -/
axiom sphere_tangent_base (solid : SphereConeUnion) : True

/-- The sphere is tangent to the lateral surface of the cone along a circle -/
axiom sphere_tangent_lateral (solid : SphereConeUnion) : True

/-- The circle of tangency coincides with the other base of the cone -/
axiom tangency_coincides_base (solid : SphereConeUnion) : True

/-- The volume of the solid formed by the union of the cone and the sphere -/
noncomputable def volume (solid : SphereConeUnion) : ℝ :=
  (1 / 3) * solid.S * solid.R

/-- Theorem stating that the volume formula is correct -/
theorem volume_formula_correct (solid : SphereConeUnion) :
  volume solid = (1 / 3) * solid.S * solid.R := by sorry

end volume_formula_correct_l2754_275425


namespace range_of_m_l2754_275417

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * x + 1

-- Define the condition for two real roots
def has_two_real_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0

-- Theorem statement
theorem range_of_m (m : ℝ) :
  has_two_real_roots m ↔ m ≤ 2 ∧ m ≠ 1 := by
  sorry

end range_of_m_l2754_275417


namespace subset_implies_a_range_l2754_275493

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : M ⊆ N a → a ≤ -1 :=
by sorry

end subset_implies_a_range_l2754_275493


namespace solution1_solution2_a_solution2_b_l2754_275420

-- Part 1
def equation1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y + 13 = 0

theorem solution1 : equation1 2 (-3) := by sorry

-- Part 2
def equation2 (x y : ℝ) : Prop :=
  x*y - 1 = x - y

theorem solution2_a (y : ℝ) : equation2 1 y := by sorry

theorem solution2_b (x : ℝ) (h : x ≠ 1) : equation2 x 1 := by sorry

end solution1_solution2_a_solution2_b_l2754_275420


namespace solution_existence_l2754_275476

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- State the theorem
theorem solution_existence (m : ℝ) :
  (∃ x : ℝ, f x < |m - 2|) ↔ m ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
sorry

end solution_existence_l2754_275476


namespace tangent_line_triangle_area_l2754_275427

/-- A line tangent to the unit circle with intercepts summing to √3 forms a triangle with area 3/2 --/
theorem tangent_line_triangle_area :
  ∀ (a b : ℝ),
  (a > 0 ∧ b > 0) →  -- Positive intercepts
  (a + b = Real.sqrt 3) →  -- Sum of intercepts
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*y + b*x = a*b) →  -- Tangent to unit circle
  (1/2 * a * b = 3/2) :=  -- Area of triangle
by sorry

end tangent_line_triangle_area_l2754_275427


namespace min_value_abc_l2754_275409

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2*a + 4*b + 7*c ≤ 2*a*b*c) : 
  a + b + c ≥ 15/2 := by
  sorry


end min_value_abc_l2754_275409


namespace blue_face_cubes_count_l2754_275440

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Counts the number of cubes with more than one blue face in a painted block -/
def count_multi_blue_face_cubes (b : Block) : Nat :=
  sorry

/-- The main theorem stating that a 5x3x1 block has 10 cubes with more than one blue face -/
theorem blue_face_cubes_count :
  let block := Block.mk 5 3 1
  count_multi_blue_face_cubes block = 10 := by
  sorry

end blue_face_cubes_count_l2754_275440


namespace max_value_sum_product_l2754_275498

theorem max_value_sum_product (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 → 
  a * b + b * c + c * d + d * a ≤ 10000 := by
  sorry

end max_value_sum_product_l2754_275498


namespace unknown_number_proof_l2754_275453

theorem unknown_number_proof (x : ℝ) : 
  (10 + 30 + 50) / 3 = ((x + 40 + 6) / 3) + 8 → x = 20 := by
  sorry

end unknown_number_proof_l2754_275453


namespace max_volume_inscribed_cone_l2754_275410

/-- Given a sphere with volume 36π, the maximum volume of an inscribed cone is 32π/3 -/
theorem max_volume_inscribed_cone (sphere_volume : ℝ) (h_volume : sphere_volume = 36 * Real.pi) :
  ∃ (max_cone_volume : ℝ),
    (∀ (cone_volume : ℝ), cone_volume ≤ max_cone_volume) ∧
    (max_cone_volume = (32 * Real.pi) / 3) :=
sorry

end max_volume_inscribed_cone_l2754_275410


namespace sum_of_series_equals_three_fourths_l2754_275467

/-- The sum of the infinite series ∑(k=1 to ∞) k/3^k is equal to 3/4 -/
theorem sum_of_series_equals_three_fourths :
  (∑' k : ℕ, (k : ℝ) / 3^k) = 3/4 := by sorry

end sum_of_series_equals_three_fourths_l2754_275467


namespace james_berets_l2754_275434

/-- The number of spools required to make one beret -/
def spools_per_beret : ℕ := 3

/-- The number of red yarn spools James has -/
def red_spools : ℕ := 12

/-- The number of black yarn spools James has -/
def black_spools : ℕ := 15

/-- The number of blue yarn spools James has -/
def blue_spools : ℕ := 6

/-- The total number of spools James has -/
def total_spools : ℕ := red_spools + black_spools + blue_spools

/-- The number of berets James can make -/
def berets_made : ℕ := total_spools / spools_per_beret

theorem james_berets :
  berets_made = 11 := by sorry

end james_berets_l2754_275434


namespace tangent_line_constant_l2754_275437

/-- The value of m for which y = -x + m is tangent to y = x^2 - 3ln(x) -/
theorem tangent_line_constant (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 
    x^2 - 3 * Real.log x = -x + m ∧ 
    2 * x - 3 / x = -1) → 
  m = 2 := by
sorry

end tangent_line_constant_l2754_275437


namespace complement_of_union_l2754_275454

def U : Finset ℕ := {1, 3, 5, 9}
def A : Finset ℕ := {1, 3, 9}
def B : Finset ℕ := {1, 9}

theorem complement_of_union :
  (U \ (A ∪ B)) = {5} := by sorry

end complement_of_union_l2754_275454


namespace min_value_of_exponential_sum_l2754_275489

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 3) :
  2^a + 2^b ≥ 4 * Real.sqrt 2 := by
  sorry

end min_value_of_exponential_sum_l2754_275489


namespace right_triangle_ratio_l2754_275407

theorem right_triangle_ratio (a d : ℝ) : 
  (a - d) ^ 2 + a ^ 2 = (a + d) ^ 2 → 
  a = d * (2 + Real.sqrt 3) := by
sorry

end right_triangle_ratio_l2754_275407


namespace calculate_expression_solve_inequality_system_l2754_275470

-- Part 1
theorem calculate_expression : (π - 3) ^ 0 + (-1) ^ 2023 - Real.sqrt 8 = -2 * Real.sqrt 2 := by
  sorry

-- Part 2
theorem solve_inequality_system {x : ℝ} : (4 * x - 3 > 9 ∧ 2 + x ≥ 0) ↔ x > 3 := by
  sorry

end calculate_expression_solve_inequality_system_l2754_275470


namespace rational_equation_solution_no_solution_rational_equation_l2754_275478

-- Problem 1
theorem rational_equation_solution (x : ℝ) :
  x ≠ 2 →
  ((2*x - 5) / (x - 2) = 3 / (2 - x)) ↔ (x = 4) :=
sorry

-- Problem 2
theorem no_solution_rational_equation (x : ℝ) :
  x ≠ 3 →
  x ≠ -3 →
  ¬(12 / (x^2 - 9) - 2 / (x - 3) = 1 / (x + 3)) :=
sorry

end rational_equation_solution_no_solution_rational_equation_l2754_275478


namespace expression_value_at_negative_two_l2754_275448

theorem expression_value_at_negative_two :
  let x : ℝ := -2
  let expr := (x - 5 + 16 / (x + 3)) / ((x - 1) / (x^2 - 9))
  expr = 15 := by sorry

end expression_value_at_negative_two_l2754_275448


namespace valid_tree_arrangement_exists_l2754_275451

/-- Represents a tree type -/
inductive TreeType
| Apple
| Pear
| Plum
| Apricot
| Cherry
| Almond

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertex1 : Point
  vertex2 : Point
  vertex3 : Point

/-- Represents the arrangement of trees -/
structure TreeArrangement where
  triangles : List EquilateralTriangle
  treeAssignment : Point → Option TreeType

/-- The main theorem stating that a valid tree arrangement exists -/
theorem valid_tree_arrangement_exists : ∃ (arrangement : TreeArrangement), 
  (arrangement.triangles.length = 6) ∧ 
  (∀ t ∈ arrangement.triangles, 
    ∃ (t1 t2 t3 : TreeType), t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
    arrangement.treeAssignment t.vertex1 = some t1 ∧
    arrangement.treeAssignment t.vertex2 = some t2 ∧
    arrangement.treeAssignment t.vertex3 = some t3) ∧
  (∀ p : Point, (∃ t ∈ arrangement.triangles, p ∈ [t.vertex1, t.vertex2, t.vertex3]) →
    ∃! treeType : TreeType, arrangement.treeAssignment p = some treeType) :=
by sorry

end valid_tree_arrangement_exists_l2754_275451


namespace correct_passwords_count_l2754_275458

def total_passwords : ℕ := 10000

def invalid_passwords : ℕ := 10

theorem correct_passwords_count :
  total_passwords - invalid_passwords = 9990 :=
by sorry

end correct_passwords_count_l2754_275458


namespace max_value_expression_l2754_275426

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_eq : c^2 = a^2 + b^2) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) ≤ 2 * a^2 + b^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) = 2 * a^2 + b^2) :=
sorry

end max_value_expression_l2754_275426


namespace equation_solution_l2754_275469

theorem equation_solution (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))) :
  x = (3 + Real.sqrt 13) / 2 ∧ y = (3 + Real.sqrt 13) / 2 ∧ z = (3 + Real.sqrt 13) / 2 := by
  sorry

end equation_solution_l2754_275469


namespace rectangle_diagonals_plus_three_l2754_275461

theorem rectangle_diagonals_plus_three (rectangle_diagonals : ℕ) : 
  rectangle_diagonals = 2 → rectangle_diagonals + 3 = 5 := by
  sorry

end rectangle_diagonals_plus_three_l2754_275461


namespace chocolate_ice_cream_orders_l2754_275447

theorem chocolate_ice_cream_orders (total_orders : ℕ) (vanilla_ratio : ℚ) 
  (h1 : total_orders = 220)
  (h2 : vanilla_ratio = 1/5)
  (h3 : vanilla_ratio * total_orders = 2 * (total_orders - vanilla_ratio * total_orders - (total_orders - vanilla_ratio * total_orders))) :
  total_orders - vanilla_ratio * total_orders - (total_orders - vanilla_ratio * total_orders) = 22 := by
  sorry

end chocolate_ice_cream_orders_l2754_275447


namespace bridget_profit_is_40_l2754_275433

/-- Calculates Bridget's profit from baking and selling bread --/
def bridget_profit (
  total_loaves : ℕ)
  (morning_price afternoon_price late_afternoon_price : ℚ)
  (production_cost fixed_cost : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3
  let afternoon_sales := (total_loaves - morning_sales) / 2
  let late_afternoon_sales := total_loaves - morning_sales - afternoon_sales
  let total_revenue := 
    morning_sales * morning_price + 
    afternoon_sales * afternoon_price + 
    late_afternoon_sales * late_afternoon_price
  let total_cost := total_loaves * production_cost + fixed_cost
  total_revenue - total_cost

/-- Theorem stating that Bridget's profit is $40 given the problem conditions --/
theorem bridget_profit_is_40 :
  bridget_profit 60 3 (3/2) 1 1 10 = 40 := by
  sorry

end bridget_profit_is_40_l2754_275433


namespace line_relationship_l2754_275472

-- Define a type for lines in space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a placeholder definition

-- Define parallel relationship between lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Add definition of parallel lines

-- Define intersection relationship between lines
def intersects (l1 l2 : Line3D) : Prop :=
  sorry -- Add definition of intersecting lines

-- Define skew relationship between lines
def skew (l1 l2 : Line3D) : Prop :=
  sorry -- Add definition of skew lines

-- Theorem statement
theorem line_relationship (a b c : Line3D) 
  (h1 : parallel a b) 
  (h2 : intersects a c) :
  skew b c ∨ intersects b c :=
sorry

end line_relationship_l2754_275472


namespace derivative_sin_cos_x_l2754_275418

theorem derivative_sin_cos_x (x : ℝ) : 
  deriv (fun x => Real.sin x * Real.cos x) x = Real.cos x ^ 2 - Real.sin x ^ 2 := by
  sorry

end derivative_sin_cos_x_l2754_275418


namespace train_passenger_count_l2754_275463

/-- Calculates the total number of passengers transported by a train between two stations -/
def total_passengers (num_round_trips : ℕ) (passengers_first_trip : ℕ) (passengers_return_trip : ℕ) : ℕ :=
  num_round_trips * (passengers_first_trip + passengers_return_trip)

/-- Proves that the total number of passengers transported is 640 given the specified conditions -/
theorem train_passenger_count :
  let num_round_trips : ℕ := 4
  let passengers_first_trip : ℕ := 100
  let passengers_return_trip : ℕ := 60
  total_passengers num_round_trips passengers_first_trip passengers_return_trip = 640 :=
by
  sorry


end train_passenger_count_l2754_275463


namespace f_decreasing_area_is_one_l2754_275487

/-- A function that is directly proportional to x-1 and passes through (-1, 4) -/
def f (x : ℝ) : ℝ := -2 * x + 2

/-- The property that f is directly proportional to x-1 -/
axiom f_prop (x : ℝ) : ∃ k : ℝ, f x = k * (x - 1)

/-- The property that f(-1) = 4 -/
axiom f_point : f (-1) = 4

/-- For any two x-values, if x1 > x2, then f(x1) < f(x2) -/
theorem f_decreasing (x1 x2 : ℝ) (h : x1 > x2) : f x1 < f x2 := by sorry

/-- The area of the triangle formed by shifting f down by 4 units -/
def triangle_area : ℝ := 1

/-- The area of the triangle formed by shifting f down by 4 units is 1 -/
theorem area_is_one : triangle_area = 1 := by sorry

end f_decreasing_area_is_one_l2754_275487


namespace probability_same_color_is_31_364_l2754_275442

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def green_plates : ℕ := 3
def total_plates : ℕ := red_plates + blue_plates + green_plates

def probability_same_color : ℚ :=
  (Nat.choose red_plates 3 + Nat.choose blue_plates 3 + Nat.choose green_plates 3) /
  Nat.choose total_plates 3

theorem probability_same_color_is_31_364 :
  probability_same_color = 31 / 364 := by
  sorry

end probability_same_color_is_31_364_l2754_275442


namespace percentage_increase_decrease_l2754_275486

theorem percentage_increase_decrease (x : ℝ) (h : x > 0) :
  x * (1 + 0.25) * (1 - 0.20) = x := by
  sorry

#check percentage_increase_decrease

end percentage_increase_decrease_l2754_275486


namespace candy_sampling_percentage_l2754_275460

theorem candy_sampling_percentage (caught_sampling : Real) (total_sampling : Real) 
  (h1 : caught_sampling = 22)
  (h2 : total_sampling = 25.88235294117647) :
  total_sampling - caught_sampling = 3.88235294117647 := by
sorry

end candy_sampling_percentage_l2754_275460


namespace carter_has_152_cards_l2754_275499

/-- The number of baseball cards Marcus has -/
def marcus_cards : ℕ := 210

/-- The difference between Marcus's and Carter's cards -/
def marcus_carter_diff : ℕ := 58

/-- The difference between Carter's and Jenny's cards -/
def carter_jenny_diff : ℕ := 35

/-- Carter's number of baseball cards -/
def carter_cards : ℕ := marcus_cards - marcus_carter_diff

theorem carter_has_152_cards : carter_cards = 152 := by
  sorry

end carter_has_152_cards_l2754_275499


namespace magnitude_of_a_plus_bi_l2754_275423

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (a b : ℝ) : Prop :=
  a / (1 - i) = 1 - b * i

-- State the theorem
theorem magnitude_of_a_plus_bi (a b : ℝ) :
  given_equation a b → Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end magnitude_of_a_plus_bi_l2754_275423


namespace valid_fractions_l2754_275443

def is_valid_fraction (num den : ℕ) : Prop :=
  10 ≤ num ∧ num < 100 ∧ 10 ≤ den ∧ den < 100 ∧
  (num / 10 : ℕ) = den % 10 ∧
  (num % 10 : ℚ) / (den / 10 : ℚ) = (num : ℚ) / (den : ℚ)

theorem valid_fractions :
  {f : ℚ | ∃ (num den : ℕ), is_valid_fraction num den ∧ f = (num : ℚ) / (den : ℚ)} =
  {64/16, 98/49, 95/19, 65/26} :=
by sorry

end valid_fractions_l2754_275443


namespace platform_length_l2754_275449

/-- Given a train passing a platform, calculate the platform length -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (time_to_pass : ℝ) 
  (h1 : train_length = 360) 
  (h2 : train_speed_kmh = 45) 
  (h3 : time_to_pass = 51.99999999999999) : 
  ∃ platform_length : ℝ, platform_length = 290 := by
  sorry

end platform_length_l2754_275449


namespace butterfly_collection_l2754_275468

/-- Given a collection of butterflies with specific conditions, prove the number of black butterflies. -/
theorem butterfly_collection (total : ℕ) (blue : ℕ) (yellow : ℕ) (black : ℕ)
  (h_total : total = 19)
  (h_blue : blue = 6)
  (h_yellow_ratio : yellow * 2 = blue)
  (h_sum : total = blue + yellow + black) :
  black = 10 := by
  sorry

end butterfly_collection_l2754_275468


namespace congruence_solution_l2754_275490

theorem congruence_solution (n : ℤ) : 
  (13 * n) % 47 = 5 % 47 → n % 47 = 4 :=
by sorry

end congruence_solution_l2754_275490


namespace fiftieth_day_of_previous_year_l2754_275488

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  number : ℤ
  isLeapYear : Bool

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (dayNumber : ℕ) : DayOfWeek :=
  sorry

/-- Determines if two days of the week are equal -/
def dayOfWeekEqual (d1 d2 : DayOfWeek) : Prop :=
  sorry

theorem fiftieth_day_of_previous_year
  (N : Year)
  (h1 : N.isLeapYear = true)
  (h2 : dayOfWeekEqual (dayOfWeek N 250) DayOfWeek.Monday = true)
  (h3 : dayOfWeekEqual (dayOfWeek (Year.mk (N.number + 1) false) 150) DayOfWeek.Tuesday = true) :
  dayOfWeekEqual (dayOfWeek (Year.mk (N.number - 1) false) 50) DayOfWeek.Wednesday = true :=
sorry

end fiftieth_day_of_previous_year_l2754_275488


namespace smallest_n_proof_l2754_275416

/-- The number of boxes -/
def num_boxes : ℕ := 2010

/-- The probability of stopping after drawing exactly n marbles -/
def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

/-- The threshold probability -/
def threshold : ℚ := 1 / num_boxes

/-- The smallest n for which P(n) < 1/2010 -/
def smallest_n : ℕ := 45

theorem smallest_n_proof :
  (∀ k < smallest_n, P k ≥ threshold) ∧
  P smallest_n < threshold :=
sorry

#check smallest_n_proof

end smallest_n_proof_l2754_275416


namespace unique_solution_quadratic_inequality_l2754_275462

theorem unique_solution_quadratic_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 - 6*b*x + 5*b| ≤ 3) ↔ 
  (b = (5 + Real.sqrt 73) / 8 ∨ b = (5 - Real.sqrt 73) / 8) :=
sorry

end unique_solution_quadratic_inequality_l2754_275462


namespace tan_pi_minus_alpha_l2754_275413

open Real

theorem tan_pi_minus_alpha (α : ℝ) :
  tan (π - α) = 3/4 →
  π/2 < α ∧ α < π →
  1 / (sin ((π + α)/2) * sin ((π - α)/2)) = 10 := by
  sorry

end tan_pi_minus_alpha_l2754_275413


namespace polynomial_simplification_l2754_275445

theorem polynomial_simplification :
  ∀ x : ℝ, 3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 :=
by
  sorry

end polynomial_simplification_l2754_275445


namespace base_prime_repr_294_l2754_275424

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of natural numbers is a valid base prime representation -/
def is_valid_base_prime_repr (repr : List ℕ) : Prop :=
  sorry

theorem base_prime_repr_294 :
  let repr := base_prime_repr 294
  is_valid_base_prime_repr repr ∧ repr = [2, 1, 0, 1] :=
sorry

end base_prime_repr_294_l2754_275424


namespace find_x_l2754_275495

theorem find_x : ∃ x : ℚ, x * 9999 = 724827405 ∧ x = 72492.75 := by
  sorry

end find_x_l2754_275495


namespace perpendicular_vectors_vector_sum_magnitude_l2754_275406

def a : ℝ × ℝ := (2, 4)
def b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem perpendicular_vectors (m : ℝ) :
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) → m = 2 := by sorry

theorem vector_sum_magnitude (m : ℝ) :
  ((a.1 + (b m).1)^2 + (a.2 + (b m).2)^2 = 25) → (m = 2 ∨ m = -6) := by sorry

end perpendicular_vectors_vector_sum_magnitude_l2754_275406


namespace counterexample_exists_l2754_275444

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Statement of the theorem
theorem counterexample_exists : ∃ n : ℕ, 
  (sumOfDigits n % 9 = 0) ∧ (n % 9 ≠ 0) :=
sorry

end counterexample_exists_l2754_275444


namespace min_value_of_f_l2754_275411

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - log x

theorem min_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = (1 + log 2) / 2 :=
sorry

end min_value_of_f_l2754_275411


namespace sequence_a_bounds_l2754_275457

def sequence_a : ℕ → ℚ
  | 0     => 1/2
  | (n+1) => sequence_a n + (1 / (n+1)^2) * (sequence_a n)^2

theorem sequence_a_bounds (n : ℕ) : 
  1 - 1 / (2^(n+1)) ≤ sequence_a n ∧ sequence_a n < 7/5 := by
  sorry

end sequence_a_bounds_l2754_275457


namespace increasing_function_condition_l2754_275439

/-- A function f(x) = (1/2)mx^2 + ln x - 2x is increasing on its domain (x > 0) if and only if m ≥ 1 -/
theorem increasing_function_condition (m : ℝ) :
  (∀ x > 0, Monotone (fun x => (1/2) * m * x^2 + Real.log x - 2*x)) ↔ m ≥ 1 := by
  sorry

end increasing_function_condition_l2754_275439


namespace arithmetic_sequence_properties_l2754_275403

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  s : ℕ → ℝ  -- The sum of the first n terms
  second_term : a 2 = 4
  sum_formula : ∀ n : ℕ, s n = n^2 + c * n
  c : ℝ       -- The constant in the sum formula

/-- Theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.c = 1 ∧ ∀ n : ℕ, seq.a n = 2 * n := by
  sorry

end arithmetic_sequence_properties_l2754_275403


namespace certain_number_proof_l2754_275438

/-- The certain number that, when multiplied by the smallest positive integer a
    that makes the product a square, equals 14 -/
def certain_number : ℕ := 14

theorem certain_number_proof (a : ℕ) (h1 : a = 14) 
  (h2 : ∀ k : ℕ, k < a → ¬∃ m : ℕ, k * certain_number = m * m) 
  (h3 : ∃ m : ℕ, a * certain_number = m * m) : 
  certain_number = 14 := by sorry

end certain_number_proof_l2754_275438


namespace positive_t_value_l2754_275400

theorem positive_t_value (a b : ℂ) (t : ℝ) (h1 : a * b = t - 3 * Complex.I) 
  (h2 : Complex.abs a = 3) (h3 : Complex.abs b = 5) : 
  t > 0 → t = 6 * Real.sqrt 6 := by
  sorry

end positive_t_value_l2754_275400


namespace back_seat_capacity_l2754_275477

def bus_capacity := 88
def left_side_seats := 15
def seat_capacity := 3

theorem back_seat_capacity : 
  ∀ (right_side_seats : ℕ) (back_seat_capacity : ℕ),
    right_side_seats = left_side_seats - 3 →
    bus_capacity = left_side_seats * seat_capacity + right_side_seats * seat_capacity + back_seat_capacity →
    back_seat_capacity = 7 := by
  sorry

end back_seat_capacity_l2754_275477
