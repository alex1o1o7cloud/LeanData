import Mathlib

namespace NUMINAMATH_CALUDE_six_eight_ten_pythagorean_l3391_339155

/-- A function to check if three positive integers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ+) : Prop :=
  a * a + b * b = c * c

/-- Theorem stating that 6, 8, and 10 form a Pythagorean triple -/
theorem six_eight_ten_pythagorean :
  isPythagoreanTriple 6 8 10 := by
  sorry

#check six_eight_ten_pythagorean

end NUMINAMATH_CALUDE_six_eight_ten_pythagorean_l3391_339155


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l3391_339177

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (r s : ℝ), 16 * x^2 + 32 * x - 2048 = 0 ↔ (x + r)^2 = s ∧ s = 129 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l3391_339177


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l3391_339169

theorem max_value_of_trigonometric_function :
  let f : ℝ → ℝ := λ x => Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)
  let S : Set ℝ := {x | -5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 3}
  ∃ x₀ ∈ S, ∀ x ∈ S, f x ≤ f x₀ ∧ f x₀ = 11 * Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l3391_339169


namespace NUMINAMATH_CALUDE_simplify_expression_l3391_339186

theorem simplify_expression (x y : ℝ) : 2 - (3 - (2 + (5 - (3*y - x)))) = 6 - 3*y + x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3391_339186


namespace NUMINAMATH_CALUDE_christines_speed_l3391_339185

/-- Given a distance of 20 miles and a time of 5 hours, the speed is 4 miles per hour. -/
theorem christines_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 20)
  (h2 : time = 5)
  (h3 : speed = distance / time) :
  speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_christines_speed_l3391_339185


namespace NUMINAMATH_CALUDE_ratio_equality_l3391_339159

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x - z) = (x + 2*y) / z ∧ (x + 2*y) / z = x / (y + z)) :
  x / (y + z) = (2*y - z) / (y + z) := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l3391_339159


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3391_339167

def A : Set ℕ := {x | (x + 4) * (x - 5) ≤ 0}
def B : Set ℕ := {x | x < 2}
def U : Set ℕ := Set.univ

theorem intersection_complement_equality :
  A ∩ (U \ B) = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3391_339167


namespace NUMINAMATH_CALUDE_function_composition_equality_l3391_339106

/-- Given functions f, g, and h, proves that A = 3B / (1 + C) -/
theorem function_composition_equality (A B C : ℝ) : 
  let f := fun x => A * x - 3 * B^2
  let g := fun x => B * x
  let h := fun x => x + C
  f (g (h 1)) = 0 → A = 3 * B / (1 + C) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3391_339106


namespace NUMINAMATH_CALUDE_intersecting_lines_theorem_l3391_339152

/-- Given two lines l₁ and l₂ that intersect at point P, and a third line l₃ -/
structure IntersectingLines where
  /-- The equation of line l₁ is 3x + 4y - 2 = 0 -/
  l₁ : ℝ → ℝ → Prop
  l₁_eq : ∀ x y, l₁ x y ↔ 3 * x + 4 * y - 2 = 0

  /-- The equation of line l₂ is 2x + y + 2 = 0 -/
  l₂ : ℝ → ℝ → Prop
  l₂_eq : ∀ x y, l₂ x y ↔ 2 * x + y + 2 = 0

  /-- P is the intersection point of l₁ and l₂ -/
  P : ℝ × ℝ
  P_on_l₁ : l₁ P.1 P.2
  P_on_l₂ : l₂ P.1 P.2

  /-- The equation of line l₃ is x - 2y - 1 = 0 -/
  l₃ : ℝ → ℝ → Prop
  l₃_eq : ∀ x y, l₃ x y ↔ x - 2 * y - 1 = 0

/-- The main theorem stating the equations of the two required lines -/
theorem intersecting_lines_theorem (g : IntersectingLines) :
  (∀ x y, x + y = 0 ↔ (∃ t : ℝ, x = t * g.P.1 ∧ y = t * g.P.2)) ∧
  (∀ x y, 2 * x + y + 2 = 0 ↔ (g.l₃ x y → (x - g.P.1) * 1 + (y - g.P.2) * 2 = 0)) :=
sorry

end NUMINAMATH_CALUDE_intersecting_lines_theorem_l3391_339152


namespace NUMINAMATH_CALUDE_treasure_value_l3391_339110

theorem treasure_value (fonzie_investment aunt_bee_investment lapis_investment lapis_share : ℚ)
  (h1 : fonzie_investment = 7000)
  (h2 : aunt_bee_investment = 8000)
  (h3 : lapis_investment = 9000)
  (h4 : lapis_share = 337500) :
  let total_investment := fonzie_investment + aunt_bee_investment + lapis_investment
  let lapis_proportion := lapis_investment / total_investment
  lapis_proportion * (lapis_share / lapis_proportion) = 1125000 := by
sorry

end NUMINAMATH_CALUDE_treasure_value_l3391_339110


namespace NUMINAMATH_CALUDE_probability_is_nine_fourteenths_l3391_339134

-- Define the total number of marbles and the number of each color
def total_marbles : ℕ := 9
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 3

-- Define the number of marbles to be selected
def selected_marbles : ℕ := 4

-- Define the probability function
def probability_one_each_color_plus_one (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) (selected : ℕ) : ℚ :=
  let favorable_outcomes := red * blue * green * (total - red - blue - green)
  let total_outcomes := Nat.choose total selected
  favorable_outcomes / total_outcomes

-- State the theorem
theorem probability_is_nine_fourteenths :
  probability_one_each_color_plus_one total_marbles red_marbles blue_marbles green_marbles selected_marbles = 9 / 14 :=
sorry

end NUMINAMATH_CALUDE_probability_is_nine_fourteenths_l3391_339134


namespace NUMINAMATH_CALUDE_f_composition_value_l3391_339154

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^3
  else if 0 ≤ x ∧ x < Real.pi / 2 then -Real.sin x
  else 0  -- undefined for x ≥ π/2, but we need to cover all cases

theorem f_composition_value : f (f (Real.pi / 6)) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3391_339154


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3391_339144

/-- A polynomial is a perfect square trinomial if it can be written as (px + q)^2 for some real p and q. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + bx + 1 is a perfect square trinomial, then b = ±4. -/
theorem perfect_square_condition (b : ℝ) :
  is_perfect_square_trinomial 4 b 1 → b = 4 ∨ b = -4 := by
  sorry

#check perfect_square_condition

end NUMINAMATH_CALUDE_perfect_square_condition_l3391_339144


namespace NUMINAMATH_CALUDE_baking_difference_l3391_339149

/-- Calculates the difference between remaining flour to be added and total sugar required -/
def flour_sugar_difference (total_flour sugar_required flour_added : ℕ) : ℤ :=
  (total_flour - flour_added : ℤ) - sugar_required

/-- Proves that the difference between remaining flour and total sugar is 1 cup -/
theorem baking_difference (total_flour sugar_required flour_added : ℕ) 
  (h1 : total_flour = 10)
  (h2 : sugar_required = 2)
  (h3 : flour_added = 7) :
  flour_sugar_difference total_flour sugar_required flour_added = 1 := by
  sorry

end NUMINAMATH_CALUDE_baking_difference_l3391_339149


namespace NUMINAMATH_CALUDE_trig_identity_l3391_339118

theorem trig_identity (θ a b : ℝ) (h : 0 < a) (h' : 0 < b) :
  (Real.sin θ)^6 / a^2 + (Real.cos θ)^6 / b^2 = 1 / (a + b) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3391_339118


namespace NUMINAMATH_CALUDE_trapezoid_constructible_l3391_339184

/-- A trapezoid with given bases and diagonals -/
structure Trapezoid where
  a : ℝ  -- length of one base
  c : ℝ  -- length of the other base
  e : ℝ  -- length of one diagonal
  f : ℝ  -- length of the other diagonal

/-- Condition for trapezoid constructibility -/
def is_constructible (t : Trapezoid) : Prop :=
  t.e + t.f > t.a + t.c ∧
  t.e + (t.a + t.c) > t.f ∧
  t.f + (t.a + t.c) > t.e

/-- Theorem stating that a trapezoid with bases 8 and 4, and diagonals 9 and 15 is constructible -/
theorem trapezoid_constructible : 
  is_constructible { a := 8, c := 4, e := 9, f := 15 } := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_constructible_l3391_339184


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l3391_339193

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (-1, 3)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (1, 3)

theorem reflection_across_y_axis :
  reflect_y original_point = reflected_point := by sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l3391_339193


namespace NUMINAMATH_CALUDE_jane_age_proof_l3391_339116

/-- Represents Jane's age when she started babysitting -/
def start_age : ℕ := 20

/-- Represents the number of years since Jane stopped babysitting -/
def years_since_stop : ℕ := 10

/-- Represents the current age of the oldest person Jane could have babysat -/
def oldest_babysat_current_age : ℕ := 22

/-- Represents Jane's current age -/
def jane_current_age : ℕ := 34

theorem jane_age_proof :
  ∀ (jane_age : ℕ),
    jane_age ≥ start_age →
    (oldest_babysat_current_age - years_since_stop) * 2 ≤ jane_age - years_since_stop →
    jane_age = jane_current_age := by
  sorry

end NUMINAMATH_CALUDE_jane_age_proof_l3391_339116


namespace NUMINAMATH_CALUDE_basketball_team_combinations_l3391_339122

theorem basketball_team_combinations (n : ℕ) (k : ℕ) (h : n = 12 ∧ k = 6) :
  n * Nat.choose (n - 1) (k - 1) = 5544 :=
sorry

end NUMINAMATH_CALUDE_basketball_team_combinations_l3391_339122


namespace NUMINAMATH_CALUDE_intersection_condition_l3391_339148

/-- Two curves intersect at exactly two distinct points -/
def HasTwoDistinctIntersections (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ (f x₁ y₁ ∧ g x₁ y₁) ∧ (f x₂ y₂ ∧ g x₂ y₂) ∧
  ∀ (x y : ℝ), (f x y ∧ g x y) → ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))

/-- The circle equation -/
def Circle (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = b^2

/-- The parabola equation -/
def Parabola (b : ℝ) (x y : ℝ) : Prop := y = -x^2 + b

theorem intersection_condition (b : ℝ) :
  HasTwoDistinctIntersections (Circle b) (Parabola b) ↔ b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l3391_339148


namespace NUMINAMATH_CALUDE_tablet_cash_price_l3391_339160

/-- Represents the installment plan for a tablet purchase -/
structure InstallmentPlan where
  downPayment : ℕ
  firstFourMonths : ℕ
  middleFourMonths : ℕ
  lastFourMonths : ℕ
  savings : ℕ

/-- Calculates the cash price of the tablet given the installment plan -/
def cashPrice (plan : InstallmentPlan) : ℕ :=
  plan.downPayment +
  4 * plan.firstFourMonths +
  4 * plan.middleFourMonths +
  4 * plan.lastFourMonths -
  plan.savings

/-- Theorem stating that the cash price of the tablet is 450 -/
theorem tablet_cash_price :
  let plan := InstallmentPlan.mk 100 40 35 30 70
  cashPrice plan = 450 := by
  sorry

end NUMINAMATH_CALUDE_tablet_cash_price_l3391_339160


namespace NUMINAMATH_CALUDE_complex_polynomial_solution_l3391_339153

theorem complex_polynomial_solution (c₀ c₁ c₂ c₃ c₄ a b : ℝ) :
  let z : ℂ := Complex.mk a b
  let i : ℂ := Complex.I
  let f : ℂ → ℂ := λ w => c₄ * w^4 + i * c₃ * w^3 + c₂ * w^2 + i * c₁ * w + c₀
  f z = 0 → f (Complex.mk (-a) b) = 0 := by sorry

end NUMINAMATH_CALUDE_complex_polynomial_solution_l3391_339153


namespace NUMINAMATH_CALUDE_initial_dogs_count_l3391_339101

/-- Proves that the initial number of dogs in a pet center is 36, given the conditions of the problem. -/
theorem initial_dogs_count (initial_cats : ℕ) (adopted_dogs : ℕ) (added_cats : ℕ) (final_total : ℕ) 
  (h1 : initial_cats = 29)
  (h2 : adopted_dogs = 20)
  (h3 : added_cats = 12)
  (h4 : final_total = 57)
  (h5 : final_total = initial_cats + added_cats + (initial_dogs - adopted_dogs)) :
  initial_dogs = 36 := by
  sorry

end NUMINAMATH_CALUDE_initial_dogs_count_l3391_339101


namespace NUMINAMATH_CALUDE_product_of_square_roots_l3391_339150

theorem product_of_square_roots : 
  let P : ℝ := Real.sqrt 2025 + Real.sqrt 2024
  let Q : ℝ := -Real.sqrt 2025 - Real.sqrt 2024
  let R : ℝ := Real.sqrt 2025 - Real.sqrt 2024
  let S : ℝ := Real.sqrt 2024 - Real.sqrt 2025
  P * Q * R * S = -1 := by
sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l3391_339150


namespace NUMINAMATH_CALUDE_min_value_3x_plus_4y_l3391_339138

theorem min_value_3x_plus_4y (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 3 * y = x * y) :
  25 ≤ 3 * x + 4 * y := by
  sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_4y_l3391_339138


namespace NUMINAMATH_CALUDE_tan_sqrt_three_solution_l3391_339176

theorem tan_sqrt_three_solution (x : ℝ) : 
  Real.tan x = Real.sqrt 3 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_sqrt_three_solution_l3391_339176


namespace NUMINAMATH_CALUDE_journey_speed_l3391_339115

theorem journey_speed (d : ℝ) (v : ℝ) :
  d > 0 ∧ v > 0 ∧
  3 * d = 1.5 ∧
  d / 5 + d / 10 + d / v = 11 / 60 →
  v = 15 := by
sorry

end NUMINAMATH_CALUDE_journey_speed_l3391_339115


namespace NUMINAMATH_CALUDE_square_root_pattern_l3391_339146

theorem square_root_pattern (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h2 : Real.sqrt (2 + 2/3) = 2 * Real.sqrt (2/3))
  (h3 : Real.sqrt (3 + 3/8) = 3 * Real.sqrt (3/8))
  (h4 : Real.sqrt (4 + 4/15) = 4 * Real.sqrt (4/15))
  (h7 : Real.sqrt (7 + a/b) = 7 * Real.sqrt (a/b)) :
  a + b = 55 := by sorry

end NUMINAMATH_CALUDE_square_root_pattern_l3391_339146


namespace NUMINAMATH_CALUDE_vector_calculation_l3391_339183

def a : Fin 3 → ℝ := ![(-3 : ℝ), 5, 2]
def b : Fin 3 → ℝ := ![(6 : ℝ), -1, -3]
def c : Fin 3 → ℝ := ![(1 : ℝ), 2, 3]

theorem vector_calculation :
  a - (4 • b) + (2 • c) = ![(-25 : ℝ), 13, 20] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l3391_339183


namespace NUMINAMATH_CALUDE_circumcircle_equation_l3391_339102

/-- The circumcircle of a triangle ABC with vertices A(-√3, 0), B(√3, 0), and C(0, 3) 
    has the equation x² + (y - 1)² = 4 -/
theorem circumcircle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (-Real.sqrt 3, 0)
  let B : ℝ × ℝ := (Real.sqrt 3, 0)
  let C : ℝ × ℝ := (0, 3)
  x^2 + (y - 1)^2 = 4 ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2 :=
by sorry


end NUMINAMATH_CALUDE_circumcircle_equation_l3391_339102


namespace NUMINAMATH_CALUDE_invalid_vote_percentage_l3391_339192

theorem invalid_vote_percentage
  (total_votes : ℕ)
  (candidate_A_share : ℚ)
  (candidate_A_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_A_share = 60 / 100)
  (h3 : candidate_A_votes = 285600) :
  (total_votes - (candidate_A_votes / candidate_A_share : ℚ)) / total_votes = 15 / 100 :=
by sorry

end NUMINAMATH_CALUDE_invalid_vote_percentage_l3391_339192


namespace NUMINAMATH_CALUDE_decimal_difference_value_l3391_339143

/-- The repeating decimal 0.0̅6̅ -/
def repeating_decimal : ℚ := 2 / 33

/-- The terminating decimal 0.06 -/
def terminating_decimal : ℚ := 6 / 100

/-- The difference between the repeating decimal 0.0̅6̅ and the terminating decimal 0.06 -/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_value : decimal_difference = 2 / 3300 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_value_l3391_339143


namespace NUMINAMATH_CALUDE_tan_five_pi_fourths_l3391_339139

theorem tan_five_pi_fourths : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_fourths_l3391_339139


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_squared_plus_x_positive_l3391_339189

theorem x_positive_sufficient_not_necessary_for_x_squared_plus_x_positive :
  (∃ x : ℝ, x > 0 ∧ x^2 + x > 0) ∧
  (∃ x : ℝ, x^2 + x > 0 ∧ ¬(x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_squared_plus_x_positive_l3391_339189


namespace NUMINAMATH_CALUDE_circle_point_bounds_l3391_339133

theorem circle_point_bounds (x y : ℝ) (h : x^2 + (y-1)^2 = 1) :
  (-(Real.sqrt 3) / 3 ≤ (y-1)/(x-2) ∧ (y-1)/(x-2) ≤ (Real.sqrt 3) / 3) ∧
  (1 - Real.sqrt 5 ≤ 2*x + y ∧ 2*x + y ≤ 1 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_circle_point_bounds_l3391_339133


namespace NUMINAMATH_CALUDE_sarah_investment_l3391_339109

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem sarah_investment :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let time : ℕ := 21
  let final_amount := compound_interest principal rate time
  ∃ ε > 0, |final_amount - 3046.28| < ε :=
sorry

end NUMINAMATH_CALUDE_sarah_investment_l3391_339109


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l3391_339119

theorem salt_solution_mixture (x : ℝ) : 
  (0.20 * x + 0.60 * 40 = 0.40 * (x + 40)) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l3391_339119


namespace NUMINAMATH_CALUDE_divisibility_conditions_divisibility_conditions_2_divisibility_conditions_3_l3391_339111

theorem divisibility_conditions (n : ℤ) :
  (∃ k : ℤ, n = 225 * k + 99) ↔ (9 ∣ n ∧ 25 ∣ (n + 1)) :=
sorry

theorem divisibility_conditions_2 (n : ℤ) :
  (∃ k : ℤ, n = 3465 * k + 1649) ↔ (21 ∣ n ∧ 165 ∣ (n + 1)) :=
sorry

theorem divisibility_conditions_3 (n : ℤ) :
  (∃ m : ℤ, n = 900 * m + 774) ↔ (9 ∣ n ∧ 25 ∣ (n + 1) ∧ 4 ∣ (n + 2)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_conditions_divisibility_conditions_2_divisibility_conditions_3_l3391_339111


namespace NUMINAMATH_CALUDE_exists_point_X_l3391_339136

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
def problem_setup (A B : ℝ × ℝ) (circle : Circle) (MN : Line) :=
  ∃ (X : ℝ × ℝ),
    -- X is on the circle
    (X.1 - circle.center.1)^2 + (X.2 - circle.center.2)^2 = circle.radius^2 ∧
    -- Define lines AX and BX
    let AX : Line := ⟨A, X⟩
    let BX : Line := ⟨B, X⟩
    -- C and D are intersections of AX and BX with the circle
    ∃ (C D : ℝ × ℝ),
      -- C and D are on the circle
      (C.1 - circle.center.1)^2 + (C.2 - circle.center.2)^2 = circle.radius^2 ∧
      (D.1 - circle.center.1)^2 + (D.2 - circle.center.2)^2 = circle.radius^2 ∧
      -- C is on AX, D is on BX
      (C.2 - A.2) * (X.1 - A.1) = (C.1 - A.1) * (X.2 - A.2) ∧
      (D.2 - B.2) * (X.1 - B.1) = (D.1 - B.1) * (X.2 - B.2) ∧
      -- CD is parallel to MN
      (C.2 - D.2) * (MN.point2.1 - MN.point1.1) = (C.1 - D.1) * (MN.point2.2 - MN.point1.2)

-- Theorem statement
theorem exists_point_X (A B : ℝ × ℝ) (circle : Circle) (MN : Line) :
  problem_setup A B circle MN :=
sorry

end NUMINAMATH_CALUDE_exists_point_X_l3391_339136


namespace NUMINAMATH_CALUDE_invisible_dots_count_l3391_339181

/-- The sum of numbers on a single die -/
def dieFaceSum : ℕ := 21

/-- The number of dice -/
def numDice : ℕ := 5

/-- The visible numbers on the dice -/
def visibleNumbers : List ℕ := [1, 1, 2, 2, 3, 3, 4, 5, 6, 6]

/-- The theorem stating that the total number of dots not visible is 72 -/
theorem invisible_dots_count : 
  numDice * dieFaceSum - visibleNumbers.sum = 72 := by sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l3391_339181


namespace NUMINAMATH_CALUDE_rosa_peach_apple_difference_l3391_339128

-- Define the number of peaches and apples for Steven
def steven_peaches : ℕ := 17
def steven_apples : ℕ := 16

-- Define Jake's peaches and apples in terms of Steven's
def jake_peaches : ℕ := steven_peaches - 6
def jake_apples : ℕ := steven_apples + 8

-- Define Rosa's peaches and apples
def rosa_peaches : ℕ := 3 * jake_peaches
def rosa_apples : ℕ := steven_apples / 2

-- Theorem to prove
theorem rosa_peach_apple_difference : rosa_peaches - rosa_apples = 25 := by
  sorry

end NUMINAMATH_CALUDE_rosa_peach_apple_difference_l3391_339128


namespace NUMINAMATH_CALUDE_sheila_initial_savings_l3391_339114

/-- Calculates Sheila's initial savings given her monthly savings, savings duration, family contribution, and final total --/
def initial_savings (monthly_savings : ℕ) (savings_duration_months : ℕ) (family_contribution : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (monthly_savings * savings_duration_months + family_contribution)

/-- Proves that Sheila's initial savings were $3,000 --/
theorem sheila_initial_savings :
  initial_savings 276 (4 * 12) 7000 23248 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_sheila_initial_savings_l3391_339114


namespace NUMINAMATH_CALUDE_grid_configurations_l3391_339156

/-- Represents a grid of lightbulbs -/
structure LightbulbGrid where
  rows : Nat
  cols : Nat

/-- Represents the switches for a lightbulb grid -/
structure Switches where
  count : Nat

/-- Calculates the number of distinct configurations for a given lightbulb grid and switches -/
def distinctConfigurations (grid : LightbulbGrid) (switches : Switches) : Nat :=
  2^(switches.count - 1)

/-- Theorem: The number of distinct configurations for a 20x16 grid with 36 switches is 2^35 -/
theorem grid_configurations :
  let grid : LightbulbGrid := ⟨20, 16⟩
  let switches : Switches := ⟨36⟩
  distinctConfigurations grid switches = 2^35 := by
  sorry

#eval distinctConfigurations ⟨20, 16⟩ ⟨36⟩

end NUMINAMATH_CALUDE_grid_configurations_l3391_339156


namespace NUMINAMATH_CALUDE_chocolate_comparison_l3391_339191

theorem chocolate_comparison 
  (robert_chocolates : ℕ)
  (robert_price : ℚ)
  (nickel_chocolates : ℕ)
  (nickel_discount : ℚ)
  (h1 : robert_chocolates = 7)
  (h2 : robert_price = 2)
  (h3 : nickel_chocolates = 5)
  (h4 : nickel_discount = 1.5)
  (h5 : robert_chocolates * robert_price = nickel_chocolates * (robert_price - nickel_discount)) :
  ∃ (n : ℕ), (robert_price * robert_chocolates) / (robert_price - nickel_discount) - robert_chocolates = n ∧ n = 21 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_comparison_l3391_339191


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3391_339179

theorem regular_polygon_sides (n : ℕ) (interior_angle exterior_angle : ℝ) : 
  n > 2 →
  interior_angle = exterior_angle + 60 →
  interior_angle + exterior_angle = 180 →
  n * exterior_angle = 360 →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3391_339179


namespace NUMINAMATH_CALUDE_square_diagonal_l3391_339145

theorem square_diagonal (area : ℝ) (side : ℝ) (diagonal : ℝ) 
  (h1 : area = 4802) 
  (h2 : area = side ^ 2) 
  (h3 : diagonal ^ 2 = 2 * side ^ 2) : 
  diagonal = Real.sqrt (2 * 4802) := by
sorry

end NUMINAMATH_CALUDE_square_diagonal_l3391_339145


namespace NUMINAMATH_CALUDE_unique_solution_l3391_339100

theorem unique_solution : ∃! x : ℝ, x > 12 ∧ (x - 6) / 12 = 5 / (x - 12) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3391_339100


namespace NUMINAMATH_CALUDE_last_three_nonzero_digits_of_80_factorial_l3391_339135

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- Returns the last three nonzero digits of a natural number -/
def lastThreeNonzeroDigits (n : ℕ) : ℕ :=
  n % 1000

theorem last_three_nonzero_digits_of_80_factorial :
  lastThreeNonzeroDigits (factorial 80) = 712 := by
  sorry

end NUMINAMATH_CALUDE_last_three_nonzero_digits_of_80_factorial_l3391_339135


namespace NUMINAMATH_CALUDE_birds_in_tree_l3391_339104

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) (birds_remaining : ℝ) : 
  birds_flew_away = 14.0 → birds_remaining = 7 → initial_birds = birds_flew_away + birds_remaining :=
by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3391_339104


namespace NUMINAMATH_CALUDE_distribution_schemes_eq_60_l3391_339195

/-- Represents the number of girls in the group. -/
def num_girls : ℕ := 5

/-- Represents the number of boys in the group. -/
def num_boys : ℕ := 2

/-- Represents the number of places for volunteer activities. -/
def num_places : ℕ := 2

/-- Calculates the number of ways to distribute girls and boys to two places. -/
def distribution_schemes : ℕ := sorry

/-- Theorem stating that the number of distribution schemes is 60. -/
theorem distribution_schemes_eq_60 : distribution_schemes = 60 := by sorry

end NUMINAMATH_CALUDE_distribution_schemes_eq_60_l3391_339195


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3391_339166

theorem trigonometric_identity (α : Real) 
  (h1 : π < α ∧ α < 2*π) 
  (h2 : Real.cos (α - 7*π) = -3/5) : 
  Real.sin (3*π + α) * Real.tan (α - 7*π/2) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3391_339166


namespace NUMINAMATH_CALUDE_petri_dishes_count_l3391_339147

-- Define the total number of germs
def total_germs : ℝ := 0.037 * 10^5

-- Define the number of germs per dish
def germs_per_dish : ℕ := 25

-- Define the number of petri dishes
def num_petri_dishes : ℕ := 148

-- Theorem statement
theorem petri_dishes_count :
  (total_germs / germs_per_dish : ℝ) = num_petri_dishes := by
  sorry

end NUMINAMATH_CALUDE_petri_dishes_count_l3391_339147


namespace NUMINAMATH_CALUDE_puzzle_arrangement_count_l3391_339190

/-- The number of letters in the word "puzzle" -/
def n : ℕ := 6

/-- The number of times the letter "z" appears in "puzzle" -/
def z_count : ℕ := 2

/-- The number of distinct arrangements of the letters in "puzzle" -/
def puzzle_arrangements : ℕ := n.factorial / z_count.factorial

theorem puzzle_arrangement_count : puzzle_arrangements = 360 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_arrangement_count_l3391_339190


namespace NUMINAMATH_CALUDE_bike_ride_problem_l3391_339188

theorem bike_ride_problem (total_distance : ℝ) (total_time : ℝ) (speed_good : ℝ) (speed_tired : ℝ) :
  total_distance = 122 →
  total_time = 8 →
  speed_good = 20 →
  speed_tired = 12 →
  ∃ time_feeling_good : ℝ,
    time_feeling_good * speed_good + (total_time - time_feeling_good) * speed_tired = total_distance ∧
    time_feeling_good = 13 / 4 :=
by sorry

end NUMINAMATH_CALUDE_bike_ride_problem_l3391_339188


namespace NUMINAMATH_CALUDE_square_side_increase_l3391_339174

theorem square_side_increase (s : ℝ) (h : s > 0) :
  ∃ p : ℝ, (s * (1 + p / 100))^2 = 1.69 * s^2 → p = 30 := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l3391_339174


namespace NUMINAMATH_CALUDE_string_length_problem_l3391_339165

theorem string_length_problem (total_strings : ℕ) (total_avg : ℝ) (subset_strings : ℕ) (subset_avg : ℝ) :
  total_strings = 6 →
  total_avg = 80 →
  subset_strings = 2 →
  subset_avg = 70 →
  let remaining_strings := total_strings - subset_strings
  let total_length := total_strings * total_avg
  let subset_length := subset_strings * subset_avg
  let remaining_length := total_length - subset_length
  (remaining_length / remaining_strings) = 85 := by
sorry

end NUMINAMATH_CALUDE_string_length_problem_l3391_339165


namespace NUMINAMATH_CALUDE_last_ball_is_green_l3391_339108

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents the state of the box with balls -/
structure BoxState where
  red : Nat
  blue : Nat
  green : Nat

/-- Represents an exchange operation -/
inductive Exchange
  | RedBlueToGreen
  | RedGreenToBlue
  | BlueGreenToRed

/-- Applies an exchange operation to a box state -/
def applyExchange (state : BoxState) (ex : Exchange) : BoxState :=
  match ex with
  | Exchange.RedBlueToGreen => 
      { red := state.red - 1, blue := state.blue - 1, green := state.green + 1 }
  | Exchange.RedGreenToBlue => 
      { red := state.red - 1, blue := state.blue + 1, green := state.green - 1 }
  | Exchange.BlueGreenToRed => 
      { red := state.red + 1, blue := state.blue - 1, green := state.green - 1 }

/-- Checks if the box state has only one ball left -/
def isLastBall (state : BoxState) : Bool :=
  state.red + state.blue + state.green = 1

/-- Gets the color of the last ball -/
def getLastBallColor (state : BoxState) : Option Color :=
  if state.red = 1 then some Color.Red
  else if state.blue = 1 then some Color.Blue
  else if state.green = 1 then some Color.Green
  else none

/-- The main theorem to prove -/
theorem last_ball_is_green (exchanges : List Exchange) :
  let initialState : BoxState := { red := 10, blue := 11, green := 12 }
  let finalState := exchanges.foldl applyExchange initialState
  isLastBall finalState → getLastBallColor finalState = some Color.Green :=
by sorry

end NUMINAMATH_CALUDE_last_ball_is_green_l3391_339108


namespace NUMINAMATH_CALUDE_tangent_sphere_radius_l3391_339105

/-- A truncated cone with a sphere tangent to its surfaces -/
structure TruncatedConeWithSphere where
  bottom_radius : ℝ
  top_radius : ℝ
  slant_height : ℝ
  sphere_radius : ℝ

/-- The sphere is tangent to the top, bottom, and lateral surface of the truncated cone -/
def is_tangent_sphere (cone : TruncatedConeWithSphere) : Prop :=
  cone.sphere_radius > 0 ∧
  cone.sphere_radius ≤ cone.bottom_radius ∧
  cone.sphere_radius ≤ cone.top_radius ∧
  cone.sphere_radius ≤ cone.slant_height

/-- The theorem stating the radius of the tangent sphere -/
theorem tangent_sphere_radius (cone : TruncatedConeWithSphere) 
  (h1 : cone.bottom_radius = 20)
  (h2 : cone.top_radius = 5)
  (h3 : cone.slant_height = 25)
  (h4 : is_tangent_sphere cone) :
  cone.sphere_radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sphere_radius_l3391_339105


namespace NUMINAMATH_CALUDE_x_gt_4_sufficient_not_necessary_for_inequality_l3391_339107

theorem x_gt_4_sufficient_not_necessary_for_inequality :
  (∀ x : ℝ, x > 4 → x^2 - 4*x > 0) ∧
  (∃ x : ℝ, x^2 - 4*x > 0 ∧ ¬(x > 4)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_4_sufficient_not_necessary_for_inequality_l3391_339107


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3391_339137

def set_A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3391_339137


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l3391_339141

theorem cubic_root_ratio (a b c d : ℝ) (h : a ≠ 0) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3) → 
  c / d = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l3391_339141


namespace NUMINAMATH_CALUDE_july_husband_age_l3391_339197

def hannah_age_then : ℕ := 6
def years_passed : ℕ := 20

theorem july_husband_age :
  ∀ (july_age_then : ℕ),
  hannah_age_then = 2 * july_age_then →
  (july_age_then + years_passed + 2 = 25) :=
by
  sorry

end NUMINAMATH_CALUDE_july_husband_age_l3391_339197


namespace NUMINAMATH_CALUDE_meet_once_l3391_339164

/-- Represents the movement of Michael and the garbage truck -/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of times Michael and the truck meet -/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- The specific movement scenario described in the problem -/
def problem_scenario : Movement where
  michael_speed := 6
  truck_speed := 12
  pail_distance := 300
  truck_stop_time := 20

/-- Theorem stating that Michael and the truck meet exactly once -/
theorem meet_once : number_of_meetings problem_scenario = 1 := by
  sorry

end NUMINAMATH_CALUDE_meet_once_l3391_339164


namespace NUMINAMATH_CALUDE_expression_equals_95_l3391_339180

theorem expression_equals_95 : 
  let some_number := -5765435
  7 ^ 8 - 6 / 2 + 9 ^ 3 + 3 + some_number = 95 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_95_l3391_339180


namespace NUMINAMATH_CALUDE_function_satisfying_equation_l3391_339121

theorem function_satisfying_equation (f : ℕ → ℕ) :
  (∀ m n : ℕ, f (m * n) + f (m + n) = f m * f n + 1) →
  (∀ n : ℕ, f n = 1 ∨ f n = n + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_equation_l3391_339121


namespace NUMINAMATH_CALUDE_zaras_estimate_l3391_339125

theorem zaras_estimate (x y z : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : z > 0) :
  (x + z) - (y + z) = x - y := by sorry

end NUMINAMATH_CALUDE_zaras_estimate_l3391_339125


namespace NUMINAMATH_CALUDE_reduced_rate_fraction_l3391_339196

def hours_in_week : ℕ := 7 * 24

def weekday_reduced_hours : ℕ := 5 * 12

def weekend_reduced_hours : ℕ := 2 * 24

def total_reduced_hours : ℕ := weekday_reduced_hours + weekend_reduced_hours

theorem reduced_rate_fraction :
  (total_reduced_hours : ℚ) / hours_in_week = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_reduced_rate_fraction_l3391_339196


namespace NUMINAMATH_CALUDE_work_done_by_resistive_force_l3391_339127

def mass : Real := 0.01
def initial_velocity : Real := 400
def final_velocity : Real := 100

def kinetic_energy (m : Real) (v : Real) : Real :=
  0.5 * m * v^2

def work_done (m : Real) (v1 : Real) (v2 : Real) : Real :=
  kinetic_energy m v1 - kinetic_energy m v2

theorem work_done_by_resistive_force :
  work_done mass initial_velocity final_velocity = 750 := by
  sorry

end NUMINAMATH_CALUDE_work_done_by_resistive_force_l3391_339127


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3391_339120

theorem greatest_prime_factor_of_expression :
  ∃ (p : ℕ), p.Prime ∧ p ∣ (3^8 + 6^7) ∧ ∀ (q : ℕ), q.Prime → q ∣ (3^8 + 6^7) → q ≤ p ∧ p = 131 :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3391_339120


namespace NUMINAMATH_CALUDE_chess_match_probability_l3391_339151

/-- Given a chess match between players A and B, this theorem proves
    the probability of player B not losing, given the probabilities
    of a draw and player B winning. -/
theorem chess_match_probability (draw_prob win_prob : ℝ) :
  draw_prob = (1 : ℝ) / 2 →
  win_prob = (1 : ℝ) / 3 →
  draw_prob + win_prob = (5 : ℝ) / 6 :=
by sorry

end NUMINAMATH_CALUDE_chess_match_probability_l3391_339151


namespace NUMINAMATH_CALUDE_race_finish_time_difference_l3391_339194

/-- Calculates the time difference at the finish line between two runners in a race -/
theorem race_finish_time_difference 
  (race_distance : ℝ) 
  (alice_speed : ℝ) 
  (bob_speed : ℝ) 
  (h1 : race_distance = 15) 
  (h2 : alice_speed = 7) 
  (h3 : bob_speed = 9) : 
  bob_speed * race_distance - alice_speed * race_distance = 30 := by
  sorry

#check race_finish_time_difference

end NUMINAMATH_CALUDE_race_finish_time_difference_l3391_339194


namespace NUMINAMATH_CALUDE_prob_more_twos_than_fives_correct_l3391_339162

def num_dice : ℕ := 5
def num_sides : ℕ := 6

def prob_more_twos_than_fives : ℚ := 2721 / 7776

theorem prob_more_twos_than_fives_correct :
  let total_outcomes := num_sides ^ num_dice
  let equal_twos_and_fives := 2334
  (1 / 2) * (1 - equal_twos_and_fives / total_outcomes) = prob_more_twos_than_fives :=
by sorry

end NUMINAMATH_CALUDE_prob_more_twos_than_fives_correct_l3391_339162


namespace NUMINAMATH_CALUDE_min_three_digit_quotient_l3391_339113

theorem min_three_digit_quotient :
  ∃ (a c : ℕ+), 
    a < 5 ∧ 
    c ≤ 9 ∧ 
    a ≠ c ∧ 
    2 * a ≠ c ∧ 
    (120 * a.val + c.val : ℚ) / (3 * a.val + c.val) = 10.75 ∧
    ∀ (x y : ℕ+), 
      x < 5 → 
      y ≤ 9 → 
      x ≠ y → 
      2 * x ≠ y → 
      (120 * x.val + y.val : ℚ) / (3 * x.val + y.val) ≥ 10.75 :=
by sorry

end NUMINAMATH_CALUDE_min_three_digit_quotient_l3391_339113


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l3391_339157

theorem regular_triangular_pyramid_volume 
  (b : ℝ) (β : ℝ) (h : 0 < β ∧ β < π / 2) :
  let volume := (((36 * b^2 * Real.cos β^2) / (1 + 9 * Real.cos β^2))^(3/2) * Real.tan β) / 24
  ∃ (a : ℝ), 
    a > 0 ∧ 
    volume = (a^3 * Real.tan β) / 24 ∧
    a^2 = (36 * b^2 * Real.cos β^2) / (1 + 9 * Real.cos β^2) :=
by sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l3391_339157


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3391_339198

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a^2 + b^2 = 0 → a * b = 0) ∧
  ∃ a b : ℝ, a * b = 0 ∧ a^2 + b^2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3391_339198


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_fourth_l3391_339178

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_fourth : 12 / (1 / 4) = 48 := by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_fourth_l3391_339178


namespace NUMINAMATH_CALUDE_current_rate_calculation_l3391_339130

/-- Given a boat traveling downstream, calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ) -- Speed of the boat in still water (km/hr)
  (distance : ℝ)   -- Distance traveled downstream (km)
  (time : ℝ)       -- Time traveled downstream (minutes)
  (h1 : boat_speed = 42)
  (h2 : distance = 36.67)
  (h3 : time = 44)
  : ∃ (current_rate : ℝ), current_rate = 8 ∧ 
    distance = (boat_speed + current_rate) * (time / 60) :=
by sorry


end NUMINAMATH_CALUDE_current_rate_calculation_l3391_339130


namespace NUMINAMATH_CALUDE_three_lines_intersection_l3391_339140

theorem three_lines_intersection (x : ℝ) : 
  (∀ (a b c d e f : ℝ), a = x ∧ b = x ∧ c = x ∧ d = x ∧ e = x ∧ f = x) →  -- opposite angles are equal
  (a + b + c + d + e + f = 360) →                                       -- sum of angles around a point is 360°
  x = 60 :=                                                            -- prove that x = 60°
by
  sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l3391_339140


namespace NUMINAMATH_CALUDE_target_sectors_degrees_l3391_339142

def circle_degrees : ℝ := 360

def microphotonics_percent : ℝ := 12
def home_electronics_percent : ℝ := 17
def food_additives_percent : ℝ := 9
def genetically_modified_microorganisms_percent : ℝ := 22
def industrial_lubricants_percent : ℝ := 6
def artificial_intelligence_percent : ℝ := 4
def nanotechnology_percent : ℝ := 5

def basic_astrophysics_percent : ℝ :=
  100 - (microphotonics_percent + home_electronics_percent + food_additives_percent +
         genetically_modified_microorganisms_percent + industrial_lubricants_percent +
         artificial_intelligence_percent + nanotechnology_percent)

def target_sectors_percent : ℝ :=
  basic_astrophysics_percent + artificial_intelligence_percent + nanotechnology_percent

theorem target_sectors_degrees :
  target_sectors_percent * (circle_degrees / 100) = 122.4 := by
  sorry

end NUMINAMATH_CALUDE_target_sectors_degrees_l3391_339142


namespace NUMINAMATH_CALUDE_opposite_solutions_value_of_m_l3391_339171

theorem opposite_solutions_value_of_m :
  ∀ (x y m : ℝ),
  (3 * x + 4 * y = 7) →
  (5 * x - 4 * y = m) →
  (x + y = 0) →
  m = -63 := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_value_of_m_l3391_339171


namespace NUMINAMATH_CALUDE_shari_walking_distance_l3391_339131

def walking_problem (walking_rate : ℝ) (first_duration : ℝ) (rest_duration : ℝ) (second_duration : ℝ) : Prop :=
  walking_rate = 4 ∧ 
  first_duration = 2 ∧ 
  rest_duration = 0.5 ∧ 
  second_duration = 1 →
  walking_rate * (first_duration + second_duration) = 12

theorem shari_walking_distance : walking_problem 4 2 0.5 1 := by
  sorry

end NUMINAMATH_CALUDE_shari_walking_distance_l3391_339131


namespace NUMINAMATH_CALUDE_calculate_expression_l3391_339123

theorem calculate_expression : (-3)^2 + 2017^0 - Real.sqrt 18 * Real.sin (π/4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3391_339123


namespace NUMINAMATH_CALUDE_horseshoe_profit_calculation_l3391_339103

/-- Calculates the profit for a horseshoe manufacturing company --/
theorem horseshoe_profit_calculation 
  (initial_outlay : ℕ) 
  (manufacturing_cost_per_set : ℕ) 
  (selling_price_per_set : ℕ) 
  (sets_produced_and_sold : ℕ) :
  initial_outlay = 10000 →
  manufacturing_cost_per_set = 20 →
  selling_price_per_set = 50 →
  sets_produced_and_sold = 500 →
  (selling_price_per_set * sets_produced_and_sold) - 
  (initial_outlay + manufacturing_cost_per_set * sets_produced_and_sold) = 5000 :=
by sorry

end NUMINAMATH_CALUDE_horseshoe_profit_calculation_l3391_339103


namespace NUMINAMATH_CALUDE_elderly_sample_count_l3391_339129

/-- Represents the composition of employees in a unit -/
structure EmployeeComposition where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents a stratified sample of employees -/
structure StratifiedSample where
  youngSampled : ℕ
  elderlySampled : ℕ

/-- Calculates the number of elderly employees in a stratified sample -/
def calculateElderlySampled (comp : EmployeeComposition) (sample : StratifiedSample) : ℚ :=
  (comp.elderly : ℚ) / comp.total * sample.youngSampled

theorem elderly_sample_count (comp : EmployeeComposition) (sample : StratifiedSample) :
  comp.total = 430 →
  comp.young = 160 →
  comp.middleAged = 2 * comp.elderly →
  sample.youngSampled = 32 →
  calculateElderlySampled comp sample = 18 := by
  sorry

end NUMINAMATH_CALUDE_elderly_sample_count_l3391_339129


namespace NUMINAMATH_CALUDE_concert_attendance_l3391_339117

theorem concert_attendance (adults : ℕ) 
  (h1 : 3 * adults = children)
  (h2 : 7 * adults + 3 * children = 6000) :
  adults + children = 1500 :=
by sorry

end NUMINAMATH_CALUDE_concert_attendance_l3391_339117


namespace NUMINAMATH_CALUDE_arithmetic_progression_condition_l3391_339161

theorem arithmetic_progression_condition 
  (a b c : ℝ) (p n k : ℕ+) : 
  (∃ (d : ℝ) (a₁ : ℝ), a = a₁ + (p - 1) * d ∧ b = a₁ + (n - 1) * d ∧ c = a₁ + (k - 1) * d) ↔ 
  (a * (n - k) + b * (k - p) + c * (p - n) = 0) ∧ 
  ((b - a) / (c - b) = (n - p : ℝ) / (k - n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_condition_l3391_339161


namespace NUMINAMATH_CALUDE_cards_remaining_l3391_339175

def initial_cards : Nat := 13
def cards_given_away : Nat := 9

theorem cards_remaining (initial : Nat) (given_away : Nat) : 
  initial = initial_cards → given_away = cards_given_away → 
  initial - given_away = 4 := by
  sorry

end NUMINAMATH_CALUDE_cards_remaining_l3391_339175


namespace NUMINAMATH_CALUDE_triple_sum_of_45_2_and_quarter_l3391_339173

theorem triple_sum_of_45_2_and_quarter (x : ℝ) (h : x = 45.2 + (1 / 4)) :
  3 * x = 136.35 := by
  sorry

end NUMINAMATH_CALUDE_triple_sum_of_45_2_and_quarter_l3391_339173


namespace NUMINAMATH_CALUDE_compound_hydrogen_count_l3391_339126

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight oxygenWeight hydrogenWeight : ℕ) : ℕ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem compound_hydrogen_count :
  ∀ (c : Compound),
    c.carbon = 3 →
    c.oxygen = 1 →
    molecularWeight c 12 16 1 = 58 →
    c.hydrogen = 6 :=
by sorry

end NUMINAMATH_CALUDE_compound_hydrogen_count_l3391_339126


namespace NUMINAMATH_CALUDE_final_output_is_four_l3391_339170

def program_output (initial : ℕ) (increment1 : ℕ) (increment2 : ℕ) : ℕ :=
  initial + increment1 + increment2

theorem final_output_is_four :
  program_output 1 1 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_final_output_is_four_l3391_339170


namespace NUMINAMATH_CALUDE_max_hot_dogs_proof_l3391_339124

-- Define pack sizes and prices
structure PackInfo where
  size : Nat
  price : Rat

-- Define the problem parameters
def budget : Rat := 300
def packInfos : List PackInfo := [
  ⟨8, 155/100⟩,
  ⟨20, 305/100⟩,
  ⟨50, 745/100⟩,
  ⟨100, 1410/100⟩,
  ⟨250, 2295/100⟩
]
def discountThreshold : Nat := 10
def discountRate : Rat := 5/100
def maxPacksPerSize : Nat := 30
def minTotalPacks : Nat := 15

-- Define a function to calculate the total number of hot dogs
def totalHotDogs (purchases : List (PackInfo × Nat)) : Nat :=
  purchases.foldl (fun acc (pack, quantity) => acc + pack.size * quantity) 0

-- Define a function to calculate the total cost
def totalCost (purchases : List (PackInfo × Nat)) : Rat :=
  purchases.foldl (fun acc (pack, quantity) =>
    let basePrice := pack.price * quantity
    let discountedPrice := if quantity > discountThreshold then basePrice * (1 - discountRate) else basePrice
    acc + discountedPrice
  ) 0

-- Theorem statement
theorem max_hot_dogs_proof :
  ∃ (purchases : List (PackInfo × Nat)),
    totalHotDogs purchases = 3250 ∧
    totalCost purchases ≤ budget ∧
    purchases.all (fun (_, quantity) => quantity ≤ maxPacksPerSize) ∧
    purchases.foldl (fun acc (_, quantity) => acc + quantity) 0 ≥ minTotalPacks ∧
    (∀ (otherPurchases : List (PackInfo × Nat)),
      totalCost otherPurchases ≤ budget →
      purchases.all (fun (_, quantity) => quantity ≤ maxPacksPerSize) →
      purchases.foldl (fun acc (_, quantity) => acc + quantity) 0 ≥ minTotalPacks →
      totalHotDogs otherPurchases ≤ totalHotDogs purchases) :=
by
  sorry

end NUMINAMATH_CALUDE_max_hot_dogs_proof_l3391_339124


namespace NUMINAMATH_CALUDE_teaching_position_allocation_l3391_339172

theorem teaching_position_allocation :
  let total_positions : ℕ := 8
  let num_schools : ℕ := 3
  let min_positions_per_school : ℕ := 1
  let min_positions_school_a : ℕ := 2
  let remaining_positions : ℕ := total_positions - (min_positions_school_a + min_positions_per_school * (num_schools - 1))
  (remaining_positions.choose (num_schools - 1)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_teaching_position_allocation_l3391_339172


namespace NUMINAMATH_CALUDE_solve_for_y_l3391_339158

theorem solve_for_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3391_339158


namespace NUMINAMATH_CALUDE_triangle_max_area_l3391_339168

/-- Given a triangle ABC with side lengths a, b, c, where c = 1,
    and area S = (a^2 + b^2 - 1) / 4,
    prove that the maximum value of S is (√2 + 1) / 4 -/
theorem triangle_max_area (a b : ℝ) (h_c : c = 1) 
  (h_area : (a^2 + b^2 - 1) / 4 = (1/2) * a * b * Real.sin C) :
  (∃ (S : ℝ), S = (a^2 + b^2 - 1) / 4 ∧ 
    (∀ (S' : ℝ), S' = (a'^2 + b'^2 - 1) / 4 → S' ≤ S)) →
  (a^2 + b^2 - 1) / 4 ≤ (Real.sqrt 2 + 1) / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3391_339168


namespace NUMINAMATH_CALUDE_polynomial_transformation_l3391_339132

-- Define the original polynomial
def original_poly (b : ℝ) (x : ℝ) : ℝ := x^4 - b*x - 3

-- Define the transformed polynomial
def transformed_poly (b : ℝ) (x : ℝ) : ℝ := 3*x^4 - b*x^3 - 1

theorem polynomial_transformation (b : ℝ) (a c d : ℝ) :
  (original_poly b a = 0 ∧ original_poly b b = 0 ∧ original_poly b c = 0 ∧ original_poly b d = 0) →
  (transformed_poly b ((a + b + c) / d^2) = 0 ∧
   transformed_poly b ((a + b + d) / c^2) = 0 ∧
   transformed_poly b ((a + c + d) / b^2) = 0 ∧
   transformed_poly b ((b + c + d) / a^2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l3391_339132


namespace NUMINAMATH_CALUDE_fixed_point_range_l3391_339182

/-- The problem statement translated to Lean 4 --/
theorem fixed_point_range (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) (hm1 : m ≠ 1) :
  (∃ (x y : ℝ), (2 * a * x - b * y + 14 = 0) ∧ 
                (y = m^(x + 1) + 1) ∧ 
                ((x - a + 1)^2 + (y + b - 2)^2 ≤ 25)) →
  (3 / 4 : ℝ) ≤ b / a ∧ b / a ≤ (4 / 3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_range_l3391_339182


namespace NUMINAMATH_CALUDE_intersection_S_T_l3391_339187

def S : Set ℝ := {x | x + 1 ≥ 2}
def T : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_S_T : S ∩ T = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_S_T_l3391_339187


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3391_339199

theorem sqrt_equation_solution (x : ℝ) (h : x > 0) : 18 / Real.sqrt x = 2 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3391_339199


namespace NUMINAMATH_CALUDE_perfect_square_floor_l3391_339112

theorem perfect_square_floor (a b : ℝ) : 
  (∀ n : ℕ+, ∃ k : ℕ, ⌊a * n + b⌋ = k^2) ↔ 
  (a = 0 ∧ ∃ k : ℕ, ∃ u : ℝ, b = k^2 + u ∧ 0 ≤ u ∧ u < 1) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_floor_l3391_339112


namespace NUMINAMATH_CALUDE_male_teacher_classes_proof_l3391_339163

/-- Represents the number of classes taught by male teachers when only male teachers are teaching. -/
def male_teacher_classes : ℕ := 10

/-- Represents the number of classes taught by female teachers. -/
def female_teacher_classes : ℕ := 15

/-- Represents the average number of tutoring classes per month. -/
def average_classes : ℕ := 6

theorem male_teacher_classes_proof (x y : ℕ) :
  female_teacher_classes * x = average_classes * (x + y) →
  male_teacher_classes * y = average_classes * (x + y) :=
by sorry

end NUMINAMATH_CALUDE_male_teacher_classes_proof_l3391_339163
