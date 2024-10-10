import Mathlib

namespace parabola_focus_l113_11349

/-- For a parabola y = ax^2 with focus at (0, 1), a = 1/4 -/
theorem parabola_focus (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Parabola equation
  (0, 1) = (0, 1 / (4 * a)) →  -- Focus at (0, 1)
  a = 1/4 := by
  sorry

end parabola_focus_l113_11349


namespace sequence_formulas_l113_11395

-- Sequence of all positive even numbers
def evenSequence (n : ℕ+) : ℕ := 2 * n

-- Sequence of all positive odd numbers
def oddSequence (n : ℕ+) : ℕ := 2 * n - 1

-- Sequence 1, 4, 9, 16, ...
def squareSequence (n : ℕ+) : ℕ := n ^ 2

-- Sequence -4, -1, 2, 5, ..., 23
def arithmeticSequence (n : ℕ+) : ℤ := 3 * n - 7

theorem sequence_formulas :
  (∀ n : ℕ+, evenSequence n = 2 * n) ∧
  (∀ n : ℕ+, oddSequence n = 2 * n - 1) ∧
  (∀ n : ℕ+, squareSequence n = n ^ 2) ∧
  (∀ n : ℕ+, arithmeticSequence n = 3 * n - 7) := by
  sorry

end sequence_formulas_l113_11395


namespace hamburgers_left_over_l113_11343

/-- Given a restaurant that made hamburgers and served some, 
    calculate the number of hamburgers left over. -/
theorem hamburgers_left_over 
  (total_made : ℕ) 
  (served : ℕ) 
  (h1 : total_made = 25) 
  (h2 : served = 11) : 
  total_made - served = 14 := by
  sorry

end hamburgers_left_over_l113_11343


namespace loop_termination_min_n_value_l113_11391

def s (n : ℕ) : ℕ := 2010 / 2^n + 3 * (2^n - 1) / 2^(n-1)

theorem loop_termination : 
  ∀ k : ℕ, k < 5 → s k ≥ 120 ∧ s 5 < 120 :=
sorry

theorem min_n_value : (∃ n : ℕ, s n < 120) ∧ (∀ k : ℕ, s k < 120 → k ≥ 5) :=
sorry

end loop_termination_min_n_value_l113_11391


namespace derivative_f_at_1_l113_11360

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 0 := by sorry

end derivative_f_at_1_l113_11360


namespace inverse_proportion_order_l113_11345

/-- Given points A, B, C on the graph of y = 3/x, prove y₂ < y₁ < y₃ -/
theorem inverse_proportion_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = 3 / (-2) → y₂ = 3 / (-1) → y₃ = 3 / 1 → y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end inverse_proportion_order_l113_11345


namespace fraction_zero_implies_x_negative_two_l113_11319

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (x^2 - 4) / (2*x - 4) = 0 ∧ 2*x - 4 ≠ 0 → x = -2 := by
  sorry

end fraction_zero_implies_x_negative_two_l113_11319


namespace emily_spent_12_dollars_l113_11368

def flower_cost : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

theorem emily_spent_12_dollars :
  flower_cost * (roses_bought + daisies_bought) = 12 := by
  sorry

end emily_spent_12_dollars_l113_11368


namespace distance_spain_other_proof_l113_11371

/-- The distance between Spain and the other country -/
def distance_spain_other : ℕ := 5404

/-- The total distance between two countries -/
def total_distance : ℕ := 7019

/-- The distance between Spain and Germany -/
def distance_spain_germany : ℕ := 1615

/-- Theorem stating that the distance between Spain and the other country
    is equal to the total distance minus the distance between Spain and Germany -/
theorem distance_spain_other_proof :
  distance_spain_other = total_distance - distance_spain_germany :=
by sorry

end distance_spain_other_proof_l113_11371


namespace equidistant_point_x_value_l113_11353

/-- A point in a 2D rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

/-- Distance from a point to the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- A point is equidistant from x-axis and y-axis -/
def isEquidistant (p : Point) : Prop :=
  distToXAxis p = distToYAxis p

/-- The main theorem -/
theorem equidistant_point_x_value (x : ℝ) :
  let p := Point.mk (-2*x) (x-6)
  isEquidistant p → x = 2 ∨ x = -6 := by
  sorry


end equidistant_point_x_value_l113_11353


namespace ginger_water_usage_l113_11341

def water_usage (hours_worked : ℕ) (cups_per_bottle : ℕ) (bottles_for_plants : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (bottles_for_plants * cups_per_bottle)

theorem ginger_water_usage :
  water_usage 8 2 5 = 26 := by
  sorry

end ginger_water_usage_l113_11341


namespace fair_coin_three_tosses_one_head_l113_11333

/-- A fair coin is a coin with equal probability of landing on either side. -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of getting exactly k successes in n trials
    with probability p for each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For a fair coin tossed 3 times, the probability
    of getting exactly 1 head and 2 tails is 3/8. -/
theorem fair_coin_three_tosses_one_head (p : ℝ) (h : fair_coin p) :
  binomial_probability 3 1 p = 3/8 := by
  sorry

end fair_coin_three_tosses_one_head_l113_11333


namespace circle_radius_l113_11351

/-- The radius of a circle described by the equation x² + y² + 12 = 10x - 6y is √22. -/
theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 + 12 = 10*x - 6*y) → ∃ (center_x center_y : ℝ), 
    ∀ (point_x point_y : ℝ), 
      (point_x - center_x)^2 + (point_y - center_y)^2 = 22 := by
sorry


end circle_radius_l113_11351


namespace complex_modulus_problem_l113_11318

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end complex_modulus_problem_l113_11318


namespace triangle_abc_properties_l113_11335

/-- Triangle ABC with vertices A(-4,0), B(0,2), and C(2,-2) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The equation of a circle in the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure CircleEquation :=
  (D : ℝ)
  (E : ℝ)
  (F : ℝ)

/-- Function to check if a point satisfies a line equation -/
def satisfiesLineEquation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Function to check if a point satisfies a circle equation -/
def satisfiesCircleEquation (p : ℝ × ℝ) (eq : CircleEquation) : Prop :=
  p.1^2 + p.2^2 + eq.D * p.1 + eq.E * p.2 + eq.F = 0

/-- Theorem stating the properties of triangle ABC -/
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.A = (-4, 0))
  (h2 : t.B = (0, 2))
  (h3 : t.C = (2, -2)) :
  ∃ (medianAB : LineEquation) (circumcircle : CircleEquation),
    -- Median equation
    (medianAB = ⟨3, 4, -2⟩) ∧ 
    -- Circumcircle equation
    (circumcircle = ⟨2, 2, -8⟩) ∧
    -- Verify that C and the midpoint of AB satisfy the median equation
    (satisfiesLineEquation t.C medianAB) ∧
    (satisfiesLineEquation ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2) medianAB) ∧
    -- Verify that all vertices satisfy the circumcircle equation
    (satisfiesCircleEquation t.A circumcircle) ∧
    (satisfiesCircleEquation t.B circumcircle) ∧
    (satisfiesCircleEquation t.C circumcircle) :=
by
  sorry

end triangle_abc_properties_l113_11335


namespace probability_of_winning_l113_11379

def total_balls : ℕ := 10
def red_balls : ℕ := 5
def white_balls : ℕ := 5
def drawn_balls : ℕ := 5

def winning_outcomes : ℕ := Nat.choose red_balls 4 * Nat.choose white_balls 1 + Nat.choose red_balls 5

def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

theorem probability_of_winning :
  (winning_outcomes : ℚ) / total_outcomes = 26 / 252 :=
sorry

end probability_of_winning_l113_11379


namespace intersection_product_constant_l113_11398

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  (a_pos : 0 < a)
  (b_pos : 0 < b)

/-- A point (x₀, y₀) on the hyperbola -/
structure PointOnHyperbola (H : Hyperbola a b) where
  x₀ : ℝ
  y₀ : ℝ
  on_hyperbola : x₀^2 / a^2 - y₀^2 / b^2 = 1

/-- The theorem stating that the product of x-coordinates of intersections is constant -/
theorem intersection_product_constant
  (H : Hyperbola a b) (P : PointOnHyperbola H) :
  ∃ (x₁ x₂ : ℝ),
    (x₁ * (b / a) = (P.x₀ * x₁) / a^2 - (P.y₀ * (b / a) * x₁) / b^2) ∧
    (x₂ * (-b / a) = (P.x₀ * x₂) / a^2 - (P.y₀ * (-b / a) * x₂) / b^2) ∧
    x₁ * x₂ = a^4 :=
sorry

end intersection_product_constant_l113_11398


namespace mixture_ratio_weight_l113_11309

theorem mixture_ratio_weight (total_weight : ℝ) (ratio_a ratio_b : ℕ) (weight_b : ℝ) : 
  total_weight = 58.00000000000001 →
  ratio_a = 9 →
  ratio_b = 11 →
  weight_b = (ratio_b : ℝ) / ((ratio_a : ℝ) + (ratio_b : ℝ)) * total_weight →
  weight_b = 31.900000000000006 := by
sorry

end mixture_ratio_weight_l113_11309


namespace f_even_and_periodic_l113_11375

-- Define a non-constant function f on ℝ
variable (f : ℝ → ℝ)

-- Condition: f is non-constant
axiom f_non_constant : ∃ x y, f x ≠ f y

-- Condition: f(10 + x) is an even function
axiom f_10_even : ∀ x, f (10 + x) = f (10 - x)

-- Condition: f(5 - x) = f(5 + x)
axiom f_5_symmetric : ∀ x, f (5 - x) = f (5 + x)

-- Theorem to prove
theorem f_even_and_periodic :
  (∀ x, f x = f (-x)) ∧ (∃ T > 0, ∀ x, f (x + T) = f x) :=
sorry

end f_even_and_periodic_l113_11375


namespace right_triangle_hypotenuse_l113_11332

structure RightTriangle :=
  (O X Y : ℝ × ℝ)
  (is_right : (X.1 - O.1) * (Y.1 - O.1) + (X.2 - O.2) * (Y.2 - O.2) = 0)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (M_midpoint : M = ((X.1 + O.1) / 2, (X.2 + O.2) / 2))
  (N_midpoint : N = ((Y.1 + O.1) / 2, (Y.2 + O.2) / 2))
  (XN_length : Real.sqrt ((X.1 - N.1)^2 + (X.2 - N.2)^2) = 19)
  (YM_length : Real.sqrt ((Y.1 - M.1)^2 + (Y.2 - M.2)^2) = 22)

theorem right_triangle_hypotenuse (t : RightTriangle) :
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 26 :=
by sorry

end right_triangle_hypotenuse_l113_11332


namespace complement_of_supplement_30_l113_11356

/-- The supplement of an angle in degrees -/
def supplement (angle : ℝ) : ℝ := 180 - angle

/-- The complement of an angle in degrees -/
def complement (angle : ℝ) : ℝ := 90 - angle

/-- The degree measure of the complement of the supplement of a 30-degree angle is 60° -/
theorem complement_of_supplement_30 : complement (supplement 30) = 60 := by
  sorry

end complement_of_supplement_30_l113_11356


namespace problem_statement_l113_11367

theorem problem_statement (x y : ℝ) (h : 2 * x - y = 8) : 6 - 2 * x + y = -2 := by
  sorry

end problem_statement_l113_11367


namespace john_climbed_45_feet_l113_11373

/-- Calculates the total distance climbed given the number of steps in three staircases and the height of each step -/
def total_distance_climbed (first_staircase : ℕ) (step_height : ℝ) : ℝ :=
  let second_staircase := 2 * first_staircase
  let third_staircase := second_staircase - 10
  let total_steps := first_staircase + second_staircase + third_staircase
  total_steps * step_height

/-- Theorem stating that John climbed 45 feet given the problem conditions -/
theorem john_climbed_45_feet :
  total_distance_climbed 20 0.5 = 45 := by
  sorry

end john_climbed_45_feet_l113_11373


namespace square_side_length_l113_11355

theorem square_side_length (d : ℝ) (h : d = 24) :
  ∃ s : ℝ, s > 0 ∧ s * s + s * s = d * d ∧ s = 12 * Real.sqrt 2 := by
  sorry

end square_side_length_l113_11355


namespace johns_remaining_money_l113_11365

/-- Calculates the remaining money after John's expenses -/
def remaining_money (initial : ℚ) (sweets : ℚ) (friend_gift : ℚ) (num_friends : ℕ) : ℚ :=
  initial - sweets - (friend_gift * num_friends)

/-- Theorem stating that John will be left with $2.45 -/
theorem johns_remaining_money :
  remaining_money 10.10 3.25 2.20 2 = 2.45 := by
  sorry

end johns_remaining_money_l113_11365


namespace cranberry_juice_unit_cost_l113_11380

/-- Given a 12-ounce can of cranberry juice selling for 84 cents, 
    prove that the unit cost is 7 cents per ounce. -/
theorem cranberry_juice_unit_cost 
  (can_size : ℕ) 
  (total_cost : ℕ) 
  (h1 : can_size = 12)
  (h2 : total_cost = 84) :
  total_cost / can_size = 7 :=
sorry

end cranberry_juice_unit_cost_l113_11380


namespace line_y_coordinate_l113_11377

/-- 
Given a line that:
- passes through a point (3, y)
- has a slope of 2
- has an x-intercept of 1

Prove that the y-coordinate of the point (3, y) is 4.
-/
theorem line_y_coordinate (y : ℝ) : 
  (∃ (m b : ℝ), m = 2 ∧ b = -2 ∧ 
    (∀ x : ℝ, y = m * (3 - x) + (m * x + b)) ∧
    (0 = m * 1 + b)) → 
  y = 4 :=
by sorry

end line_y_coordinate_l113_11377


namespace arctan_sum_equals_pi_over_four_l113_11392

theorem arctan_sum_equals_pi_over_four :
  ∃ (n : ℕ), n > 0 ∧
  Real.arctan (1 / 6) + Real.arctan (1 / 7) + Real.arctan (1 / 5) + Real.arctan (1 / n) = π / 4 ∧
  n = 311 := by
  sorry

end arctan_sum_equals_pi_over_four_l113_11392


namespace exam_average_problem_l113_11336

theorem exam_average_problem (n : ℕ) : 
  (15 : ℝ) * 75 + (10 : ℝ) * 90 = (n : ℝ) * 81 → n = 25 :=
by
  sorry

end exam_average_problem_l113_11336


namespace john_hiking_probability_l113_11310

theorem john_hiking_probability (p_rain : ℝ) (p_hike_given_rain : ℝ) (p_hike_given_sunny : ℝ)
  (h_rain : p_rain = 0.3)
  (h_hike_rain : p_hike_given_rain = 0.1)
  (h_hike_sunny : p_hike_given_sunny = 0.9) :
  p_rain * p_hike_given_rain + (1 - p_rain) * p_hike_given_sunny = 0.66 := by
  sorry

end john_hiking_probability_l113_11310


namespace reading_speed_first_half_l113_11338

/-- Given a book with specific reading conditions, calculate the reading speed for the first half. -/
theorem reading_speed_first_half (total_pages : ℕ) (second_half_speed : ℕ) (total_days : ℕ) : 
  total_pages = 500 → 
  second_half_speed = 5 → 
  total_days = 75 → 
  (total_pages / 2) / (total_days - (total_pages / 2) / second_half_speed) = 10 := by
  sorry

#check reading_speed_first_half

end reading_speed_first_half_l113_11338


namespace cos_2alpha_problem_l113_11328

theorem cos_2alpha_problem (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α - Real.cos α = Real.sqrt 10 / 5) : 
  Real.cos (2 * α) = -4/5 := by
  sorry

end cos_2alpha_problem_l113_11328


namespace probability_sum_15_l113_11330

/-- The number of ways to roll a sum of 15 with five six-sided dice -/
def waysToRoll15 : ℕ := 95

/-- The total number of possible outcomes when rolling five six-sided dice -/
def totalOutcomes : ℕ := 6^5

/-- A fair, standard six-sided die -/
structure Die :=
  (faces : Finset ℕ)
  (fair : faces = {1, 2, 3, 4, 5, 6})

/-- The probability of rolling a sum of 15 with five fair, standard six-sided dice -/
theorem probability_sum_15 (d1 d2 d3 d4 d5 : Die) :
  (waysToRoll15 : ℚ) / totalOutcomes = 95 / 7776 := by
  sorry


end probability_sum_15_l113_11330


namespace distance_between_centers_l113_11369

/-- Given a triangle with sides 6, 8, and 10, the distance between the centers
    of its inscribed and circumscribed circles is √13. -/
theorem distance_between_centers (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inradius := area / s
  let circumradius := (a * b * c) / (4 * area)
  Real.sqrt (circumradius^2 + inradius^2 - 2 * circumradius * inradius * Real.cos (π / 2)) = Real.sqrt 13 := by
  sorry

end distance_between_centers_l113_11369


namespace exponential_inequality_l113_11312

theorem exponential_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) :
  a^(Real.sqrt a) > a^(a^a) ∧ a^(a^a) > a := by
  sorry

end exponential_inequality_l113_11312


namespace leila_spending_difference_l113_11382

theorem leila_spending_difference : 
  ∀ (total_money sweater_cost jewelry_cost remaining : ℕ),
  sweater_cost = 40 →
  4 * sweater_cost = total_money →
  remaining = 20 →
  total_money = sweater_cost + jewelry_cost + remaining →
  jewelry_cost - sweater_cost = 60 := by
sorry

end leila_spending_difference_l113_11382


namespace genetic_material_not_equal_l113_11399

/-- Represents a cell involved in fertilization -/
structure Cell where
  nucleus : Bool
  cytoplasm : Nat

/-- Represents the process of fertilization -/
def fertilization (sperm : Cell) (egg : Cell) : Prop :=
  sperm.nucleus ∧ egg.nucleus ∧ sperm.cytoplasm < egg.cytoplasm

/-- Represents the zygote formed after fertilization -/
def zygote (sperm : Cell) (egg : Cell) : Prop :=
  fertilization sperm egg

/-- Theorem stating that genetic material in the zygote does not come equally from both parents -/
theorem genetic_material_not_equal (sperm egg : Cell) 
  (h_sperm : sperm.nucleus ∧ sperm.cytoplasm = 0)
  (h_egg : egg.nucleus ∧ egg.cytoplasm > 0)
  (h_zygote : zygote sperm egg) :
  ¬(∃ (x : Nat), x > 0 ∧ x = sperm.cytoplasm ∧ x = egg.cytoplasm) := by
  sorry


end genetic_material_not_equal_l113_11399


namespace vector_relation_l113_11320

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variable (P A B C : V)

/-- Given that PA + 2PB + 3PC = 0, prove that AP = (1/3)AB + (1/2)AC -/
theorem vector_relation (h : (A - P) + 2 • (B - P) + 3 • (C - P) = 0) :
  P - A = (1/3) • (B - A) + (1/2) • (C - A) := by sorry

end vector_relation_l113_11320


namespace a_share_is_240_l113_11314

/-- Calculates the share of profit for partner A given the initial investments,
    changes in investment, and total profit. -/
def calculate_share_a (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) 
                      (total_months : ℕ) (change_month : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_a := initial_a * change_month + (initial_a - withdraw_a) * (total_months - change_month)
  let investment_months_b := initial_b * change_month + (initial_b + advance_b) * (total_months - change_month)
  let total_investment_months := investment_months_a + investment_months_b
  (investment_months_a * total_profit) / total_investment_months

/-- Theorem stating that given the problem conditions, A's share of the profit is 240. -/
theorem a_share_is_240 : 
  calculate_share_a 3000 4000 1000 1000 12 8 630 = 240 := by
  sorry

end a_share_is_240_l113_11314


namespace prime_composite_inequality_l113_11347

theorem prime_composite_inequality (n : ℕ) : 
  (Prime (2 * n - 1) → 
    ∀ (a : Fin n → ℕ+), (∀ i j, i ≠ j → a i ≠ a j) → 
      ∃ i j, (a i + a j : ℝ) / (Nat.gcd (a i) (a j)) ≥ 2 * n - 1) ∧
  (¬Prime (2 * n - 1) → 
    ∃ (a : Fin n → ℕ+), (∀ i j, i ≠ j → a i ≠ a j) ∧ 
      ∀ i j, (a i + a j : ℝ) / (Nat.gcd (a i) (a j)) < 2 * n - 1) :=
by sorry

end prime_composite_inequality_l113_11347


namespace negation_equivalence_l113_11313

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (2 : ℝ) ^ x < x ^ 2) ↔ (∀ x : ℝ, (2 : ℝ) ^ x ≥ x ^ 2) :=
by sorry

end negation_equivalence_l113_11313


namespace regular_hexagon_diagonals_l113_11358

/-- Regular hexagon with side length, shortest diagonal, and longest diagonal -/
structure RegularHexagon where
  a : ℝ  -- side length
  b : ℝ  -- shortest diagonal
  d : ℝ  -- longest diagonal

/-- Theorem: In a regular hexagon, the shortest diagonal is √3 times the side length,
    and the longest diagonal is 4/√3 times the side length -/
theorem regular_hexagon_diagonals (h : RegularHexagon) :
  h.b = Real.sqrt 3 * h.a ∧ h.d = (4 * h.a) / Real.sqrt 3 :=
by sorry

end regular_hexagon_diagonals_l113_11358


namespace good_carrots_count_l113_11386

/-- The number of good carrots given the number of carrots picked by Faye and her mom, and the number of bad carrots. -/
def goodCarrots (fayeCarrots momCarrots badCarrots : ℕ) : ℕ :=
  fayeCarrots + momCarrots - badCarrots

/-- Theorem stating that the number of good carrots is 12 given the problem conditions. -/
theorem good_carrots_count : goodCarrots 23 5 16 = 12 := by
  sorry

end good_carrots_count_l113_11386


namespace original_equals_scientific_l113_11389

-- Define the original number
def original_number : ℕ := 150000000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.5 * (10 ^ 11)

-- Theorem to prove the equality
theorem original_equals_scientific : (original_number : ℝ) = scientific_notation := by
  sorry

end original_equals_scientific_l113_11389


namespace root_difference_quadratic_l113_11331

theorem root_difference_quadratic (x : ℝ) : 
  let eq := fun x : ℝ => x^2 + 42*x + 360 + 49
  let roots := {r : ℝ | eq r = 0}
  let diff := fun (a b : ℝ) => |a - b|
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ diff r₁ r₂ = 8 * Real.sqrt 2 :=
by sorry

end root_difference_quadratic_l113_11331


namespace fruit_shop_problem_l113_11366

/-- The price per kilogram of apples in yuan -/
def apple_price : ℝ := 8

/-- The price per kilogram of pears in yuan -/
def pear_price : ℝ := 6

/-- The maximum number of kilograms of apples that can be purchased -/
def max_apples : ℝ := 5

theorem fruit_shop_problem :
  (1 * apple_price + 3 * pear_price = 26) ∧
  (2 * apple_price + 1 * pear_price = 22) ∧
  (∀ x y : ℝ, x + y = 15 → x * apple_price + y * pear_price ≤ 100 → x ≤ max_apples) :=
by sorry

end fruit_shop_problem_l113_11366


namespace fraction_equals_zero_l113_11354

theorem fraction_equals_zero (x : ℝ) :
  (x^2 - 1) / (1 - x) = 0 ∧ 1 - x ≠ 0 → x = -1 := by
  sorry

end fraction_equals_zero_l113_11354


namespace xiaopang_birthday_is_26th_l113_11388

/-- Represents a day in May -/
def MayDay := Fin 31

/-- Xiaopang's birthday -/
def xiaopang_birthday : MayDay := sorry

/-- Xiaoya's birthday -/
def xiaoya_birthday : MayDay := sorry

/-- Days of the week, represented as integers mod 7 -/
def DayOfWeek := Fin 7

/-- Function to determine the day of the week for a given day in May -/
def day_of_week (d : MayDay) : DayOfWeek := sorry

/-- Wednesday, represented as a specific day of the week -/
def wednesday : DayOfWeek := sorry

theorem xiaopang_birthday_is_26th :
  -- Both birthdays are in May (implied by their types)
  -- Both birthdays fall on a Wednesday
  day_of_week xiaopang_birthday = wednesday ∧
  day_of_week xiaoya_birthday = wednesday ∧
  -- Xiaopang's birthday is later than Xiaoya's
  xiaopang_birthday.val > xiaoya_birthday.val ∧
  -- The sum of their birth dates is 38
  xiaopang_birthday.val + xiaoya_birthday.val = 38 →
  -- Conclusion: Xiaopang's birthday is on the 26th
  xiaopang_birthday.val = 26 := by
  sorry

end xiaopang_birthday_is_26th_l113_11388


namespace rocks_difference_l113_11361

/-- Given the number of rocks collected by Joshua, Jose, and Albert, prove that Albert collected 20 more rocks than Jose. -/
theorem rocks_difference (joshua_rocks : ℕ) (jose_rocks : ℕ) (albert_rocks : ℕ)
  (h1 : joshua_rocks = 80)
  (h2 : jose_rocks = joshua_rocks - 14)
  (h3 : albert_rocks = joshua_rocks + 6) :
  albert_rocks - jose_rocks = 20 := by
sorry

end rocks_difference_l113_11361


namespace inverse_proportion_ratio_l113_11334

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (h_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y₁ ≠ 0 ∧ y₂ ≠ 0) 
  (h_inverse : ∃ c : ℝ, c ≠ 0 ∧ x₁ * y₁ = c ∧ x₂ * y₂ = c) 
  (h_ratio : x₁ / x₂ = 3 / 5) : 
  y₁ / y₂ = 5 / 3 := by
  sorry

end inverse_proportion_ratio_l113_11334


namespace necessary_but_not_sufficient_l113_11317

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- State the theorem
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, a = 1 → A ∪ B a = Set.univ) ∧
  (∃ a : ℝ, a ≤ 1 ∧ a ≠ 1 ∧ A ∪ B a = Set.univ) :=
by sorry

end necessary_but_not_sufficient_l113_11317


namespace median_bisects_perimeter_implies_isosceles_l113_11352

/-- A triangle is represented by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- A median of a triangle -/
structure Median (t : Triangle) where
  base : ℝ
  is_median : base = t.a ∨ base = t.b ∨ base = t.c

/-- A triangle is isosceles if at least two of its sides are equal -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The theorem statement -/
theorem median_bisects_perimeter_implies_isosceles (t : Triangle) (m : Median t) :
  (m.base / 2 + (t.perimeter - m.base) / 2 = t.perimeter / 2) → t.isIsosceles :=
by
  sorry

end median_bisects_perimeter_implies_isosceles_l113_11352


namespace circle_properties_l113_11329

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 6*y - 3 = 0

/-- The center of the circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- The radius of the circle -/
def CircleRadius : ℝ := 4

/-- Theorem: The given equation represents a circle with center (-2, 3) and radius 4 -/
theorem circle_properties :
  ∀ (x y : ℝ),
    CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = CircleRadius^2 :=
by sorry

end circle_properties_l113_11329


namespace complex_equation_solution_l113_11393

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : 
  z = -1 + Complex.I := by
sorry

end complex_equation_solution_l113_11393


namespace quadratic_inequality_max_value_l113_11397

theorem quadratic_inequality_max_value (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) →
  (∃ M, M = -4 ∧ ∀ a' b' c', (∀ x, a'*x^2 + b'*x + c' > 0 ↔ -1 < x ∧ x < 2) → b' - c' + 4/a' ≤ M) :=
sorry

end quadratic_inequality_max_value_l113_11397


namespace exponent_calculation_l113_11339

theorem exponent_calculation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end exponent_calculation_l113_11339


namespace sum_mod_eight_l113_11362

theorem sum_mod_eight :
  (7145 + 7146 + 7147 + 7148 + 7149) % 8 = 7 := by
  sorry

end sum_mod_eight_l113_11362


namespace shared_foci_implies_m_equals_one_l113_11372

/-- Given an ellipse and a hyperbola that share the same foci, prove that m = 1 -/
theorem shared_foci_implies_m_equals_one (m : ℝ) :
  (∀ x y : ℝ, x^2/4 + y^2/m^2 = 1 ↔ x^2/m - y^2/2 = 1) →
  (∃ c : ℝ, c^2 = 4 - m^2 ∧ c^2 = m + 2) →
  m = 1 := by
  sorry

end shared_foci_implies_m_equals_one_l113_11372


namespace hcf_problem_l113_11363

theorem hcf_problem (a b : ℕ) (h1 : a = 391) (h2 : ∃ (hcf : ℕ), Nat.lcm a b = hcf * 13 * 17) : Nat.gcd a b = 23 := by
  sorry

end hcf_problem_l113_11363


namespace eggs_in_box_l113_11346

/-- The number of eggs in the box after adding more eggs -/
def total_eggs (initial : Float) (added : Float) : Float :=
  initial + added

/-- Theorem stating that adding 5.0 eggs to 47.0 eggs results in 52.0 eggs -/
theorem eggs_in_box : total_eggs 47.0 5.0 = 52.0 := by
  sorry

end eggs_in_box_l113_11346


namespace contestant_paths_count_l113_11315

/-- Represents the diamond-shaped grid for the word "CONTESTANT" -/
def ContestantGrid : Type := Unit  -- placeholder for the actual grid structure

/-- Represents a valid path in the grid -/
def ValidPath (grid : ContestantGrid) : Type := Unit  -- placeholder for the actual path structure

/-- The number of valid paths in the grid -/
def numValidPaths (grid : ContestantGrid) : ℕ := sorry

/-- The theorem stating that the number of valid paths is 256 -/
theorem contestant_paths_count (grid : ContestantGrid) : numValidPaths grid = 256 := by
  sorry

end contestant_paths_count_l113_11315


namespace factor_tree_problem_l113_11340

theorem factor_tree_problem (H I F G X : ℕ) : 
  H = 7 * 2 →
  I = 11 * 2 →
  F = 7 * H →
  G = 11 * I →
  X = F * G →
  X = 23716 :=
by sorry

end factor_tree_problem_l113_11340


namespace fraction_simplification_l113_11396

theorem fraction_simplification 
  (x y z u : ℝ) 
  (h1 : x + z ≠ 0) 
  (h2 : y + u ≠ 0) : 
  (x * y^2 + 2 * y * z^2 + y * z * u + 2 * x * y * z + 2 * x * z * u + y^2 * z + 2 * z^2 * u + x * y * u) / 
  (x * u^2 + y * z^2 + y * z * u + x * u * z + x * y * u + u * z^2 + z * u^2 + x * y * z) = 
  (y + 2 * z) / (u + z) := by sorry

end fraction_simplification_l113_11396


namespace right_triangle_inradius_l113_11325

/-- The inradius of a right triangle with side lengths 9, 12, and 15 is 3 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →
  a^2 + b^2 = c^2 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 3 := by
  sorry

end right_triangle_inradius_l113_11325


namespace polynomial_real_roots_l113_11387

def polynomial (x : ℝ) : ℝ := x^9 - 37*x^8 - 2*x^7 + 74*x^6 + x^4 - 37*x^3 - 2*x^2 + 74*x

theorem polynomial_real_roots :
  ∃ (s : Finset ℝ), s.card = 5 ∧ (∀ x : ℝ, polynomial x = 0 ↔ x ∈ s) := by
  sorry

end polynomial_real_roots_l113_11387


namespace factory_temporary_workers_percentage_l113_11303

theorem factory_temporary_workers_percentage 
  (total_workers : ℕ) 
  (technician_percentage : ℚ) 
  (non_technician_percentage : ℚ) 
  (permanent_technician_percentage : ℚ) 
  (permanent_non_technician_percentage : ℚ) 
  (h1 : technician_percentage = 80 / 100)
  (h2 : non_technician_percentage = 20 / 100)
  (h3 : permanent_technician_percentage = 80 / 100)
  (h4 : permanent_non_technician_percentage = 20 / 100)
  (h5 : technician_percentage + non_technician_percentage = 1) :
  let permanent_workers := (technician_percentage * permanent_technician_percentage + 
                            non_technician_percentage * permanent_non_technician_percentage) * total_workers
  let temporary_workers := total_workers - permanent_workers
  temporary_workers / total_workers = 32 / 100 := by
sorry

end factory_temporary_workers_percentage_l113_11303


namespace inequality_condition_l113_11381

theorem inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
sorry

end inequality_condition_l113_11381


namespace unique_pairs_from_ten_l113_11376

theorem unique_pairs_from_ten (n : ℕ) (h : n = 10) : n * (n - 1) / 2 = 45 := by
  sorry

end unique_pairs_from_ten_l113_11376


namespace consecutive_odd_integers_sum_l113_11323

theorem consecutive_odd_integers_sum (x : ℤ) :
  (∃ y z : ℤ, y = x + 2 ∧ z = x + 4 ∧ x + z = 150) →
  x + (x + 2) + (x + 4) = 225 := by
  sorry

end consecutive_odd_integers_sum_l113_11323


namespace special_triangle_property_l113_11316

/-- Triangle with given side, inscribed circle radius, and excircle radius -/
structure SpecialTriangle where
  -- Side length
  a : ℝ
  -- Inscribed circle radius
  r : ℝ
  -- Excircle radius
  r_b : ℝ
  -- Assumption that all values are positive
  a_pos : 0 < a
  r_pos : 0 < r
  r_b_pos : 0 < r_b

/-- Theorem stating the relationship between side length, semiperimeter, and tangent length -/
theorem special_triangle_property (t : SpecialTriangle) :
  ∃ (p : ℝ) (tangent_length : ℝ),
    -- Semiperimeter is positive
    0 < p ∧
    -- Tangent length is positive and less than semiperimeter
    0 < tangent_length ∧ tangent_length < p ∧
    -- The given side length equals semiperimeter minus tangent length
    t.a = p - tangent_length :=
  sorry

end special_triangle_property_l113_11316


namespace n_gon_regions_l113_11370

/-- The number of regions into which the diagonals of an n-gon divide it -/
def R (n : ℕ) : ℕ := (n*(n-1)*(n-2)*(n-3))/24 + (n*(n-3))/2 + 1

/-- Theorem stating the number of regions in an n-gon divided by its diagonals -/
theorem n_gon_regions (n : ℕ) (h : n ≥ 3) :
  R n = (n*(n-1)*(n-2)*(n-3))/24 + (n*(n-3))/2 + 1 :=
by sorry


end n_gon_regions_l113_11370


namespace polynomial_intersection_l113_11383

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem polynomial_intersection (a b c d : ℝ) : 
  -- f and g are distinct polynomials
  (∃ x, f a b x ≠ g c d x) →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- Both f and g have the same minimum value
  (f a b (-a/2) = g c d (-c/2)) →
  -- The graphs of f and g intersect at the point (150, -150)
  (f a b 150 = -150 ∧ g c d 150 = -150) →
  -- Conclusion: a + c = -600
  a + c = -600 :=
by sorry

end polynomial_intersection_l113_11383


namespace ten_row_triangle_count_l113_11342

/-- Calculates the sum of the first n natural numbers. -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of rods in a triangle with n rows. -/
def rods_count (n : ℕ) : ℕ := 3 * triangular_number n

/-- Calculates the number of connectors in a triangle with n rows of rods. -/
def connectors_count (n : ℕ) : ℕ := triangular_number (n + 1)

/-- The total number of rods and connectors in a triangle with n rows of rods. -/
def total_count (n : ℕ) : ℕ := rods_count n + connectors_count n

theorem ten_row_triangle_count :
  total_count 10 = 231 := by sorry

end ten_row_triangle_count_l113_11342


namespace triangle_problem_l113_11300

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * Real.sin C = b * Real.sin A ∧ b * Real.sin C = c * Real.sin B ∧ c * Real.sin A = a * Real.sin B →
  (a * (Real.sin C - Real.sin A)) / (Real.sin C + Real.sin B) = c - b →
  Real.tan B / Real.tan A + Real.tan B / Real.tan C = 4 →
  B = π / 3 ∧ Real.sin A / Real.sin C = (3 + Real.sqrt 5) / 2 ∨ Real.sin A / Real.sin C = (3 - Real.sqrt 5) / 2 := by
  sorry


end triangle_problem_l113_11300


namespace jogger_speed_l113_11306

/-- Proves that the jogger's speed is 9 kmph given the conditions of the problem -/
theorem jogger_speed (train_length : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : train_speed = 45)
  (h3 : initial_distance = 240)
  (h4 : passing_time = 36)
  : ∃ (jogger_speed : ℝ), jogger_speed = 9 ∧ 
    (train_speed - jogger_speed) * passing_time * (5/18) = initial_distance + train_length :=
by
  sorry

#check jogger_speed

end jogger_speed_l113_11306


namespace imaginary_part_of_i_over_one_plus_i_l113_11308

theorem imaginary_part_of_i_over_one_plus_i (i : ℂ) :
  i * i = -1 →
  Complex.im (i / (1 + i)) = 1 / 2 := by
sorry

end imaginary_part_of_i_over_one_plus_i_l113_11308


namespace jerry_has_49_feathers_l113_11327

/-- The number of feathers Jerry has left after his adventure -/
def jerrys_remaining_feathers : ℕ :=
  let hawk_feathers : ℕ := 6
  let eagle_feathers : ℕ := 17 * hawk_feathers
  let total_feathers : ℕ := hawk_feathers + eagle_feathers
  let feathers_after_giving : ℕ := total_feathers - 10
  (feathers_after_giving / 2 : ℕ)

/-- Theorem stating that Jerry has 49 feathers left -/
theorem jerry_has_49_feathers : jerrys_remaining_feathers = 49 := by
  sorry

end jerry_has_49_feathers_l113_11327


namespace min_value_xyz_l113_11301

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 27) :
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 27 → x + 3 * y + 6 * z ≤ a + 3 * b + 6 * c ∧
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 27 ∧ x + 3 * y + 6 * z = 27 :=
sorry

end min_value_xyz_l113_11301


namespace cos_squared_pi_fourth_minus_alpha_l113_11302

theorem cos_squared_pi_fourth_minus_alpha (α : Real) 
  (h : Real.tan (α + π/4) = 3/4) : 
  Real.cos (π/4 - α)^2 = 9/25 := by
  sorry

end cos_squared_pi_fourth_minus_alpha_l113_11302


namespace complex_number_coordinates_l113_11344

theorem complex_number_coordinates : (Complex.I + 1)^2 * Complex.I = -2 * Complex.I := by
  sorry

end complex_number_coordinates_l113_11344


namespace smallest_n_multiple_of_five_l113_11321

theorem smallest_n_multiple_of_five (x y : ℤ) 
  (hx : 5 ∣ (x - 2)) 
  (hy : 5 ∣ (y + 4)) : 
  ∃ n : ℕ+, 
    5 ∣ (x^2 + 2*x*y + y^2 + n) ∧ 
    ∀ m : ℕ+, (5 ∣ (x^2 + 2*x*y + y^2 + m) → n ≤ m) ∧
    n = 1 := by
sorry

end smallest_n_multiple_of_five_l113_11321


namespace david_chemistry_marks_l113_11384

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average of a list of natural numbers --/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Theorem: Given David's marks and average, his Chemistry mark must be 67 --/
theorem david_chemistry_marks (m : Marks) (h1 : m.english = 51) (h2 : m.mathematics = 65)
    (h3 : m.physics = 82) (h4 : m.biology = 85)
    (h5 : average [m.english, m.mathematics, m.physics, m.chemistry, m.biology] = 70) :
    m.chemistry = 67 := by
  sorry

#check david_chemistry_marks

end david_chemistry_marks_l113_11384


namespace first_part_multiplier_l113_11378

theorem first_part_multiplier (x : ℝ) : x + 7 * x = 55 → x = 5 → 1 = 1 := by
  sorry

end first_part_multiplier_l113_11378


namespace parabola_properties_l113_11357

/-- A parabola with vertex at the origin, focus on the x-axis, and passing through (2, 2) -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 2*x

theorem parabola_properties :
  (parabola_equation 0 0) ∧ 
  (∃ p : ℝ, p > 0 ∧ parabola_equation p 0) ∧
  (parabola_equation 2 2) := by
  sorry

end parabola_properties_l113_11357


namespace least_five_digit_congruent_to_3_mod_17_l113_11326

theorem least_five_digit_congruent_to_3_mod_17 :
  ∀ n : ℕ, n ≥ 10000 → n ≡ 3 [ZMOD 17] → n ≥ 10004 :=
by sorry

end least_five_digit_congruent_to_3_mod_17_l113_11326


namespace blueberry_pancakes_l113_11311

theorem blueberry_pancakes (total : ℕ) (banana : ℕ) (plain : ℕ)
  (h1 : total = 67)
  (h2 : banana = 24)
  (h3 : plain = 23) :
  total - banana - plain = 20 := by
  sorry

end blueberry_pancakes_l113_11311


namespace students_left_unassigned_l113_11322

/-- The number of students left unassigned to groups in a school with specific classroom distributions -/
theorem students_left_unassigned (total_students : ℕ) (num_classrooms : ℕ) 
  (classroom_A : ℕ) (classroom_B : ℕ) (classroom_C : ℕ) (classroom_D : ℕ) 
  (num_groups : ℕ) : 
  total_students = 128 →
  num_classrooms = 4 →
  classroom_A = 37 →
  classroom_B = 31 →
  classroom_C = 25 →
  classroom_D = 35 →
  num_groups = 9 →
  classroom_A + classroom_B + classroom_C + classroom_D = total_students →
  total_students - (num_groups * (total_students / num_groups)) = 2 := by
  sorry

#eval 128 - (9 * (128 / 9))  -- This should evaluate to 2

end students_left_unassigned_l113_11322


namespace robot_constraint_l113_11305

-- Define the robot's path as a parabola
def robot_path (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through P(-1, 0) with slope k
def line_through_P (k x y : ℝ) : Prop := y = k*(x + 1)

-- Define the condition that the line does not intersect the robot's path
def no_intersection (k : ℝ) : Prop :=
  ∀ x y : ℝ, robot_path x y ∧ line_through_P k x y → False

-- Theorem statement
theorem robot_constraint (k : ℝ) :
  no_intersection k ↔ k < -Real.sqrt 2 ∨ k > Real.sqrt 2 :=
sorry

end robot_constraint_l113_11305


namespace diamond_commutative_eq_four_lines_l113_11359

/-- Diamond operation -/
def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^3 * b - a * b^3

/-- The set of points (x, y) where x ◇ y = y ◇ x -/
def diamond_commutative_set : Set (ℝ × ℝ) :=
  {p | diamond p.1 p.2 = diamond p.2 p.1}

/-- The union of four lines: x = 0, y = 0, y = x, and y = -x -/
def four_lines : Set (ℝ × ℝ) :=
  {p | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

theorem diamond_commutative_eq_four_lines :
  diamond_commutative_set = four_lines := by sorry

end diamond_commutative_eq_four_lines_l113_11359


namespace grandfather_grandson_ages_l113_11394

def isComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem grandfather_grandson_ages :
  ∀ (grandfather grandson : ℕ),
    isComposite grandfather →
    isComposite grandson →
    (grandfather + 1) * (grandson + 1) = 1610 →
    grandfather = 69 ∧ grandson = 22 := by
  sorry

end grandfather_grandson_ages_l113_11394


namespace sum_of_four_consecutive_even_integers_l113_11324

theorem sum_of_four_consecutive_even_integers (n : ℤ) : 
  (∃ k : ℤ, n = 4*k + 12 ∧ k % 2 = 0) ↔ n ∈ ({56, 80, 124, 200} : Set ℤ) := by
  sorry

#check sum_of_four_consecutive_even_integers 34
#check sum_of_four_consecutive_even_integers 56
#check sum_of_four_consecutive_even_integers 80
#check sum_of_four_consecutive_even_integers 124
#check sum_of_four_consecutive_even_integers 200

end sum_of_four_consecutive_even_integers_l113_11324


namespace expression_value_l113_11304

theorem expression_value : (35 + 12)^2 - (12^2 + 35^2 - 2 * 12 * 35) = 1680 := by
  sorry

end expression_value_l113_11304


namespace prism_height_l113_11337

/-- A triangular prism with given dimensions -/
structure TriangularPrism where
  volume : ℝ
  base_side1 : ℝ
  base_side2 : ℝ
  height : ℝ

/-- The volume of a triangular prism is equal to the area of its base times its height -/
axiom volume_formula (p : TriangularPrism) : 
  p.volume = (1/2) * p.base_side1 * p.base_side2 * p.height

/-- Theorem: Given a triangular prism with volume 120 cm³ and base sides 3 cm and 4 cm, 
    its height is 20 cm -/
theorem prism_height (p : TriangularPrism) 
  (h_volume : p.volume = 120)
  (h_base1 : p.base_side1 = 3)
  (h_base2 : p.base_side2 = 4) :
  p.height = 20 := by
  sorry

end prism_height_l113_11337


namespace velocity_maximum_at_lowest_point_l113_11390

/-- Represents a point on the roller coaster track -/
structure TrackPoint where
  height : ℝ
  velocity : ℝ

/-- Represents the roller coaster system -/
structure RollerCoaster where
  points : List TrackPoint
  initial_velocity : ℝ
  g : ℝ  -- Acceleration due to gravity

/-- The total mechanical energy of the system -/
def total_energy (rc : RollerCoaster) (p : TrackPoint) : ℝ :=
  0.5 * p.velocity^2 + rc.g * p.height

/-- The point with minimum height has maximum velocity -/
theorem velocity_maximum_at_lowest_point (rc : RollerCoaster) :
  ∀ p q : TrackPoint,
    p ∈ rc.points →
    q ∈ rc.points →
    p.height < q.height →
    total_energy rc p = total_energy rc q →
    p.velocity > q.velocity :=
sorry

end velocity_maximum_at_lowest_point_l113_11390


namespace greatest_gcd_6Tn_n_minus_1_l113_11350

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem greatest_gcd_6Tn_n_minus_1 :
  (∀ n : ℕ+, Nat.gcd (6 * triangular_number n) (n - 1) ≤ 3) ∧
  (∃ n : ℕ+, Nat.gcd (6 * triangular_number n) (n - 1) = 3) :=
by sorry

end greatest_gcd_6Tn_n_minus_1_l113_11350


namespace imaginary_part_of_complex_number_l113_11348

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_complex_number :
  imaginary_part (1/5 - 2/5 * I) = -2/5 := by
  sorry

end imaginary_part_of_complex_number_l113_11348


namespace maggie_earnings_proof_l113_11374

/-- Calculates Maggie's earnings from magazine subscriptions -/
def maggieEarnings (pricePerSubscription : ℕ) 
                   (parentsSubscriptions : ℕ)
                   (grandfatherSubscriptions : ℕ)
                   (nextDoorNeighborSubscriptions : ℕ) : ℕ :=
  let otherNeighborSubscriptions := 2 * nextDoorNeighborSubscriptions
  let totalSubscriptions := parentsSubscriptions + grandfatherSubscriptions + 
                            nextDoorNeighborSubscriptions + otherNeighborSubscriptions
  pricePerSubscription * totalSubscriptions

/-- Proves that Maggie's earnings are $55.00 -/
theorem maggie_earnings_proof : 
  maggieEarnings 5 4 1 2 = 55 := by
  sorry

end maggie_earnings_proof_l113_11374


namespace benzoic_acid_weight_l113_11307

/-- Represents the molecular formula of a compound -/
structure MolecularFormula where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound -/
def molecularWeight (formula : MolecularFormula) (weights : AtomicWeights) : ℝ :=
  formula.carbon * weights.carbon +
  formula.hydrogen * weights.hydrogen +
  formula.oxygen * weights.oxygen

/-- Theorem: The molecular weight of 4 moles of Benzoic acid is 488.472 grams -/
theorem benzoic_acid_weight :
  let benzoicAcid : MolecularFormula := { carbon := 7, hydrogen := 6, oxygen := 2 }
  let atomicWeights : AtomicWeights := { carbon := 12.01, hydrogen := 1.008, oxygen := 16.00 }
  (4 : ℝ) * molecularWeight benzoicAcid atomicWeights = 488.472 := by
  sorry


end benzoic_acid_weight_l113_11307


namespace polygon_exterior_angles_l113_11364

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 30) → (n * exterior_angle = 360) → n = 12 := by
  sorry

end polygon_exterior_angles_l113_11364


namespace daria_savings_weeks_l113_11385

/-- The number of weeks required for Daria to save enough money for a vacuum cleaner. -/
def weeks_to_save (initial_savings : ℕ) (weekly_contribution : ℕ) (vacuum_cost : ℕ) : ℕ :=
  ((vacuum_cost - initial_savings) + weekly_contribution - 1) / weekly_contribution

/-- Theorem: Daria needs 10 weeks to save for the vacuum cleaner. -/
theorem daria_savings_weeks : weeks_to_save 20 10 120 = 10 := by
  sorry

end daria_savings_weeks_l113_11385
