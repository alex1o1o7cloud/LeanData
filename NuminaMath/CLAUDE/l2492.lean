import Mathlib

namespace log_equation_solution_l2492_249242

theorem log_equation_solution (x : ℝ) (h : Real.log x / Real.log 3 * Real.log 3 / Real.log 4 = 4) : x = 256 := by
  sorry

end log_equation_solution_l2492_249242


namespace unique_triple_l2492_249247

def is_infinite_repeating_decimal (a b c : ℕ+) : Prop :=
  (a + b / 9 : ℚ)^2 = c + 7/9

def fraction_is_integer (c a : ℕ+) : Prop :=
  ∃ k : ℤ, (c + a : ℚ) / (c - a) = k

theorem unique_triple : 
  ∃! (a b c : ℕ+), 
    b < 10 ∧ 
    is_infinite_repeating_decimal a b c ∧ 
    fraction_is_integer c a ∧
    a = 1 ∧ b = 6 ∧ c = 2 := by sorry

end unique_triple_l2492_249247


namespace tournament_divisibility_l2492_249288

theorem tournament_divisibility (n : ℕ) : 
  let tournament_year := fun i => 1978 + i
  (tournament_year 43 = 2021) →
  (∃! k, k = 3 ∧ 
    (∀ i ∈ Finset.range k, 
      ∃ m > 43, tournament_year m % m = 0 ∧
      ∀ j ∈ Finset.Icc 44 (m - 1), tournament_year j % j ≠ 0)) :=
by sorry

end tournament_divisibility_l2492_249288


namespace overall_pass_rate_l2492_249258

theorem overall_pass_rate (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (1 - a) * (1 - b) = ab - a - b + 1 := by
  sorry

end overall_pass_rate_l2492_249258


namespace rice_mixture_price_l2492_249221

theorem rice_mixture_price (price1 price2 proportion1 proportion2 : ℚ) 
  (h1 : price1 = 31/10)
  (h2 : price2 = 36/10)
  (h3 : proportion1 = 7)
  (h4 : proportion2 = 3)
  : (price1 * proportion1 + price2 * proportion2) / (proportion1 + proportion2) = 13/4 := by
  sorry

end rice_mixture_price_l2492_249221


namespace decimal_fraction_equality_l2492_249287

theorem decimal_fraction_equality : (0.5^4) / (0.05^3) = 500 := by
  sorry

end decimal_fraction_equality_l2492_249287


namespace max_absolute_sum_l2492_249238

theorem max_absolute_sum (x y z : ℝ) :
  (|x + 2*y - 3*z| ≤ 6) →
  (|x - 2*y + 3*z| ≤ 6) →
  (|x - 2*y - 3*z| ≤ 6) →
  (|x + 2*y + 3*z| ≤ 6) →
  |x| + |y| + |z| ≤ 6 := by
  sorry

end max_absolute_sum_l2492_249238


namespace calculation_proof_l2492_249208

theorem calculation_proof :
  (Real.sqrt 8 - abs (-2) + (1/3)⁻¹ - 4 * Real.cos (45 * π / 180)) = 1 ∧
  ∀ x : ℝ, (x - 2)^2 - x*(x - 4) = 4 := by
  sorry

end calculation_proof_l2492_249208


namespace equation_solution_l2492_249257

theorem equation_solution : 
  {x : ℝ | -x^2 = (2*x + 4)/(x + 2)} = {-2, -1} :=
by sorry

end equation_solution_l2492_249257


namespace unique_solution_value_l2492_249265

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant (b^2 - 4ac) must be zero -/
def has_unique_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 3x^2 - 7x + k = 0 -/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  3*x^2 - 7*x + k = 0

theorem unique_solution_value (k : ℝ) :
  (∃! x, quadratic_equation k x) ↔ k = 49/12 :=
sorry

end unique_solution_value_l2492_249265


namespace alpha_value_l2492_249272

/-- A structure representing the relationship between α, β, and γ -/
structure Relationship where
  α : ℝ
  β : ℝ
  γ : ℝ
  k : ℝ
  h1 : α = k * γ / β

/-- The theorem stating the relationship between α, β, and γ -/
theorem alpha_value (r : Relationship) (h2 : r.α = 4) (h3 : r.β = 27) (h4 : r.γ = 3) :
  ∃ (r' : Relationship), r'.β = -81 ∧ r'.γ = 9 ∧ r'.α = -4 :=
sorry

end alpha_value_l2492_249272


namespace complex_equation_solution_l2492_249226

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 2 - Complex.I) → z = -1 - 2*Complex.I := by
  sorry

end complex_equation_solution_l2492_249226


namespace isosceles_triangle_base_angle_l2492_249285

-- Define an isosceles triangle with a vertex angle of 40°
structure IsoscelesTriangle where
  vertex_angle : ℝ
  is_isosceles : Bool
  vertex_angle_value : vertex_angle = 40

-- Define the property we want to prove
def base_angle_is_70 (triangle : IsoscelesTriangle) : Prop :=
  ∃ (base_angle : ℝ), base_angle = 70 ∧ 
    triangle.vertex_angle + 2 * base_angle = 180

-- State the theorem
theorem isosceles_triangle_base_angle 
  (triangle : IsoscelesTriangle) : 
  base_angle_is_70 triangle :=
sorry

end isosceles_triangle_base_angle_l2492_249285


namespace smallest_common_multiple_of_6_and_15_l2492_249298

theorem smallest_common_multiple_of_6_and_15 :
  ∃ b : ℕ+, (∀ n : ℕ+, (6 ∣ n) → (15 ∣ n) → b ≤ n) ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ b = 30 := by
  sorry

end smallest_common_multiple_of_6_and_15_l2492_249298


namespace x_value_proof_l2492_249260

theorem x_value_proof (x : ℝ) : x = 2 * (1/x) * (-x) - 5 → x = -7 := by
  sorry

end x_value_proof_l2492_249260


namespace probability_theorem_l2492_249246

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ k : ℕ, a * b + a + b = 7 * k - 2

def total_pairs : ℕ := Nat.choose 100 2

def valid_pairs : ℕ := 105

theorem probability_theorem :
  (valid_pairs : ℚ) / total_pairs = 7 / 330 := by sorry

end probability_theorem_l2492_249246


namespace wine_without_cork_cost_l2492_249213

def bottle_with_cork : ℝ := 2.10
def cork : ℝ := 2.05

theorem wine_without_cork_cost (bottle_without_cork : ℝ) 
  (h1 : bottle_without_cork > cork) : 
  bottle_without_cork - cork > 0.05 := by
  sorry

end wine_without_cork_cost_l2492_249213


namespace odd_cube_plus_one_not_square_l2492_249228

theorem odd_cube_plus_one_not_square (n : ℤ) (h : Odd n) :
  ¬ ∃ x : ℤ, n^3 + 1 = x^2 := by
sorry

end odd_cube_plus_one_not_square_l2492_249228


namespace repeating_decimal_multiplication_l2492_249207

theorem repeating_decimal_multiplication (x : ℝ) : 
  (∀ n : ℕ, (x * 10^(4 + 2*n)) % 1 = 0.3131) → 
  (10^5 - 10^3) * x = 309.969 := by
sorry

end repeating_decimal_multiplication_l2492_249207


namespace polynomial_division_quotient_l2492_249264

theorem polynomial_division_quotient :
  ∀ x : ℝ, x ≠ 1 →
  (x^6 + 6) = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 7 := by
sorry

end polynomial_division_quotient_l2492_249264


namespace system_solution_l2492_249232

theorem system_solution (x y z b : ℝ) : 
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 - z^2 = b^2) →
  ((b = 0 ∧ ((x = 0 ∧ z = -y) ∨ (y = 0 ∧ z = -x))) ∨
   (b ≠ 0 ∧ z = 0 ∧ 
    ((x = (1 + Real.sqrt (-1/2)) * b ∧ y = (1 - Real.sqrt (-1/2)) * b) ∨
     (x = (1 - Real.sqrt (-1/2)) * b ∧ y = (1 + Real.sqrt (-1/2)) * b)))) := by
  sorry

end system_solution_l2492_249232


namespace arrangement_count_l2492_249280

theorem arrangement_count (n m : ℕ) (hn : n = 6) (hm : m = 4) :
  (Nat.choose n m) * (Nat.factorial m) = 360 :=
sorry

end arrangement_count_l2492_249280


namespace parking_solution_l2492_249251

def parking_problem (first_level second_level third_level fourth_level : ℕ) : Prop :=
  first_level = 4 ∧
  second_level = first_level + 7 ∧
  third_level > second_level ∧
  fourth_level = 14 ∧
  first_level + second_level + third_level + fourth_level = 46

theorem parking_solution :
  ∀ first_level second_level third_level fourth_level : ℕ,
  parking_problem first_level second_level third_level fourth_level →
  third_level - second_level = 6 :=
by
  sorry

#check parking_solution

end parking_solution_l2492_249251


namespace sin_alpha_for_point_one_neg_two_l2492_249222

/-- Given that the terminal side of angle α passes through point P(1,-2),
    prove that sin α = -2√5/5 -/
theorem sin_alpha_for_point_one_neg_two (α : Real) :
  (∃ (P : ℝ × ℝ), P = (1, -2) ∧ P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  Real.sin α = -2 * Real.sqrt 5 / 5 := by
sorry


end sin_alpha_for_point_one_neg_two_l2492_249222


namespace perpendicular_vectors_trig_equality_l2492_249216

/-- Given two perpendicular vectors a and b, prove that 
    (sin³α + cos³α) / (sinα - cosα) = 9/5 -/
theorem perpendicular_vectors_trig_equality 
  (a b : ℝ × ℝ) 
  (h1 : a = (4, -2)) 
  (h2 : ∃ α : ℝ, b = (Real.cos α, Real.sin α)) 
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  ∃ α : ℝ, (Real.sin α)^3 + (Real.cos α)^3 = 9/5 * ((Real.sin α) - (Real.cos α)) :=
sorry

end perpendicular_vectors_trig_equality_l2492_249216


namespace lucas_payment_l2492_249240

/-- Calculates the payment for window cleaning based on given conditions --/
def calculate_payment (windows_per_floor : ℕ) (num_floors : ℕ) (payment_per_window : ℕ) 
  (penalty_per_period : ℕ) (days_per_period : ℕ) (total_days : ℕ) : ℕ :=
  let total_windows := windows_per_floor * num_floors
  let total_earned := total_windows * payment_per_window
  let num_periods := total_days / days_per_period
  let total_penalty := num_periods * penalty_per_period
  total_earned - total_penalty

theorem lucas_payment :
  calculate_payment 5 4 3 2 4 12 = 54 :=
sorry

end lucas_payment_l2492_249240


namespace best_fitting_model_l2492_249271

structure Model where
  id : Nat
  r_squared : Real

def models : List Model := [
  { id := 1, r_squared := 0.98 },
  { id := 2, r_squared := 0.80 },
  { id := 3, r_squared := 0.54 },
  { id := 4, r_squared := 0.35 }
]

theorem best_fitting_model :
  ∃ m ∈ models, ∀ m' ∈ models, m.r_squared ≥ m'.r_squared ∧ m.id = 1 :=
by sorry

end best_fitting_model_l2492_249271


namespace circles_tangent_implies_a_value_l2492_249243

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Definition of circle C₂ -/
def C₂ (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 1

/-- Two circles are tangent if the distance between their centers equals the sum or difference of their radii -/
def are_tangent (a : ℝ) : Prop := |a| = 5 ∨ |a| = 3

/-- Theorem stating that if C₁ and C₂ are tangent, then |a| = 5 or |a| = 3 -/
theorem circles_tangent_implies_a_value (a : ℝ) :
  (∃ x y : ℝ, C₁ x y ∧ C₂ a x y) → are_tangent a :=
sorry

end circles_tangent_implies_a_value_l2492_249243


namespace constant_remainder_iff_b_eq_neg_five_halves_l2492_249286

/-- The dividend polynomial -/
def dividend (b x : ℝ) : ℝ := 12 * x^4 - 5 * x^3 + b * x^2 - 4 * x + 8

/-- The divisor polynomial -/
def divisor (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

/-- Theorem stating that the remainder is constant iff b = -5/2 -/
theorem constant_remainder_iff_b_eq_neg_five_halves :
  ∃ (q : ℝ → ℝ) (r : ℝ), ∀ x, dividend (-5/2) x = q x * divisor x + r ↔ 
  ∀ b, (∃ (q : ℝ → ℝ) (r : ℝ), ∀ x, dividend b x = q x * divisor x + r) → b = -5/2 := by
  sorry

end constant_remainder_iff_b_eq_neg_five_halves_l2492_249286


namespace max_roads_removal_l2492_249233

/-- A graph representing the Empire of Westeros --/
structure WesterosGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  is_connected : Bool
  vertex_count : vertices.card = 1000
  edge_count : edges.card = 2017
  initial_connectivity : is_connected = true

/-- The result of removing roads from the graph --/
structure KingdomFormation where
  removed_roads : Nat
  kingdom_count : Nat

/-- The maximum number of roads that can be removed to form exactly 7 kingdoms --/
def max_removable_roads (g : WesterosGraph) : Nat :=
  993

/-- Theorem stating the maximum number of removable roads --/
theorem max_roads_removal (g : WesterosGraph) :
  ∃ (kf : KingdomFormation),
    kf.removed_roads = max_removable_roads g ∧
    kf.kingdom_count = 7 ∧
    ∀ (kf' : KingdomFormation),
      kf'.kingdom_count = 7 → kf'.removed_roads ≤ kf.removed_roads :=
sorry


end max_roads_removal_l2492_249233


namespace area_between_tangent_circles_l2492_249219

/-- The area of the region between two tangent circles -/
theorem area_between_tangent_circles 
  (r₁ : ℝ) -- radius of the inner circle
  (d : ℝ)  -- distance between the centers of the circles
  (h₁ : r₁ = 5) -- given radius of inner circle
  (h₂ : d = 3)  -- given distance between centers
  : (π * ((r₁ + d)^2 - r₁^2) : ℝ) = 39 * π :=
sorry

end area_between_tangent_circles_l2492_249219


namespace cricket_bat_profit_l2492_249297

/-- Calculates the profit amount for a cricket bat sale -/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 850 →
  profit_percentage = 42.857142857142854 →
  ∃ (cost_price : ℝ) (profit : ℝ),
    cost_price > 0 ∧
    profit > 0 ∧
    selling_price = cost_price + profit ∧
    profit_percentage = (profit / cost_price) * 100 ∧
    profit = 255 := by
  sorry

end cricket_bat_profit_l2492_249297


namespace limit_equals_negative_six_l2492_249270

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem limit_equals_negative_six :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0,
    |Δx| < δ → |((f (1 - 2*Δx) - f 1) / Δx) + 6| < ε :=
by sorry

end limit_equals_negative_six_l2492_249270


namespace percentage_difference_l2492_249256

theorem percentage_difference (X : ℝ) (h : X > 0) : 
  let first_number := 0.70 * X
  let second_number := 0.63 * X
  (first_number - second_number) / first_number * 100 = 10 := by
sorry

end percentage_difference_l2492_249256


namespace perfect_square_condition_l2492_249252

theorem perfect_square_condition (m : ℤ) : 
  (∃ n : ℤ, m^2 + 6*m + 28 = n^2) ↔ (m = 6 ∨ m = -12) :=
by sorry

end perfect_square_condition_l2492_249252


namespace select_students_problem_l2492_249268

/-- The number of ways to select students for a meeting -/
def select_students (num_boys num_girls : ℕ) (total_selected : ℕ) (min_boys min_girls : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to select students for the given conditions -/
theorem select_students_problem : select_students 5 4 4 2 1 = 100 := by
  sorry

end select_students_problem_l2492_249268


namespace smallest_positive_integer_solution_l2492_249292

theorem smallest_positive_integer_solution : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (|5 * (x : ℤ) - 8| = 47) ∧ 
  (∀ (y : ℕ), y > 0 ∧ |5 * (y : ℤ) - 8| = 47 → x ≤ y) ∧
  (x = 11) := by
sorry

end smallest_positive_integer_solution_l2492_249292


namespace time_period_is_12_hours_l2492_249204

/-- The time period in hours for a given population net increase -/
def time_period (birth_rate : ℚ) (death_rate : ℚ) (net_increase : ℕ) : ℚ :=
  let net_rate_per_second : ℚ := (birth_rate - death_rate) / 2
  let seconds : ℚ := net_increase / net_rate_per_second
  seconds / 3600

/-- Theorem stating that given the problem conditions, the time period is 12 hours -/
theorem time_period_is_12_hours :
  time_period 8 6 86400 = 12 := by
  sorry

end time_period_is_12_hours_l2492_249204


namespace quadratic_roots_distance_l2492_249295

theorem quadratic_roots_distance (m : ℝ) : 
  (∃ α β : ℂ, (α^2 - 2 * Real.sqrt 2 * α + m = 0) ∧ 
              (β^2 - 2 * Real.sqrt 2 * β + m = 0) ∧ 
              (Complex.abs (α - β) = 3)) → 
  m = 17/4 := by
sorry

end quadratic_roots_distance_l2492_249295


namespace unique_solution_system_l2492_249237

theorem unique_solution_system (x y z : ℝ) : 
  (x^2 + 25*y + 19*z = -471) ∧
  (y^2 + 23*x + 21*z = -397) ∧
  (z^2 + 21*x + 21*y = -545) ↔
  (x = -22 ∧ y = -23 ∧ z = -20) := by
sorry

end unique_solution_system_l2492_249237


namespace sphere_surface_area_l2492_249230

theorem sphere_surface_area (r : ℝ) (h : r = 2) : 4 * Real.pi * r^2 = 16 * Real.pi := by
  sorry

end sphere_surface_area_l2492_249230


namespace abs_ln_equal_implies_product_one_l2492_249261

theorem abs_ln_equal_implies_product_one (a b : ℝ) (h1 : a ≠ b) (h2 : |Real.log a| = |Real.log b|) : a * b = 1 := by
  sorry

end abs_ln_equal_implies_product_one_l2492_249261


namespace inverse_not_always_true_l2492_249299

theorem inverse_not_always_true :
  ¬(∀ (a b m : ℝ), (a < b → a * m^2 < b * m^2)) :=
sorry

end inverse_not_always_true_l2492_249299


namespace hyperbola_equation_l2492_249212

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (b/a = Real.sqrt 2) →                   -- Slope of asymptote
  (a^2 + b^2 = 3) →                       -- Right focus coincides with parabola focus
  (∀ (x y : ℝ), x^2 - y^2/2 = 1 ↔ x^2/a^2 - y^2/b^2 = 1) :=
by sorry

end hyperbola_equation_l2492_249212


namespace fifteen_solutions_l2492_249235

/-- The system of equations has exactly 15 distinct real solutions -/
theorem fifteen_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ × ℝ)),
    (∀ (u v s t : ℝ), (u, v, s, t) ∈ solutions ↔ 
      (u = s + t + s*u*t ∧
       v = t + u + t*u*v ∧
       s = u + v + u*v*s ∧
       t = v + s + v*s*t)) ∧
    solutions.card = 15 := by
  sorry

end fifteen_solutions_l2492_249235


namespace symmetry_probability_l2492_249231

/-- Represents a point on the grid --/
structure GridPoint where
  x : Fin 11
  y : Fin 11

/-- The center point of the grid --/
def centerPoint : GridPoint :=
  ⟨5, 5⟩

/-- The set of all points on the grid --/
def allPoints : Finset GridPoint :=
  sorry

/-- The set of all points except the center point --/
def nonCenterPoints : Finset GridPoint :=
  sorry

/-- Predicate to check if a line through two points is a line of symmetry --/
def isSymmetryLine (p q : GridPoint) : Prop :=
  sorry

/-- The set of points that form symmetry lines with the center point --/
def symmetryPoints : Finset GridPoint :=
  sorry

theorem symmetry_probability :
    (symmetryPoints.card : ℚ) / (nonCenterPoints.card : ℚ) = 1 / 3 :=
  sorry

end symmetry_probability_l2492_249231


namespace constant_term_expansion_l2492_249215

theorem constant_term_expansion (x : ℝ) : 
  let expression := (x - 4 + 4 / x)^3
  ∃ (a b c : ℝ), expression = a * x^3 + b * x^2 + c * x - 160
  := by sorry

end constant_term_expansion_l2492_249215


namespace five_balls_four_boxes_l2492_249227

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 61 ways to distribute 5 distinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 61 := by
  sorry

end five_balls_four_boxes_l2492_249227


namespace problem_statement_l2492_249273

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 * b^3 / 5 = 1000) 
  (h2 : a * b = 2) : 
  a^3 * b^2 / 3 = 2 / 705 := by
  sorry

end problem_statement_l2492_249273


namespace s_5_l2492_249209

/-- s(n) is a function that attaches the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- Examples of s(n) for n = 1, 2, 3, 4 -/
axiom s_examples : s 1 = 1 ∧ s 2 = 14 ∧ s 3 = 149 ∧ s 4 = 14916

/-- Theorem: s(5) equals 1491625 -/
theorem s_5 : s 5 = 1491625 := sorry

end s_5_l2492_249209


namespace candy_cost_l2492_249275

/-- 
Given Chris's babysitting earnings and expenses, prove the cost of the candy assortment.
-/
theorem candy_cost 
  (video_game_cost : ℕ) 
  (hourly_rate : ℕ) 
  (hours_worked : ℕ) 
  (money_left : ℕ) 
  (h1 : video_game_cost = 60)
  (h2 : hourly_rate = 8)
  (h3 : hours_worked = 9)
  (h4 : money_left = 7) :
  video_game_cost + money_left + 5 = hourly_rate * hours_worked :=
by sorry

end candy_cost_l2492_249275


namespace amy_local_calls_l2492_249262

/-- Proves that Amy made 15 local calls given the conditions of the problem -/
theorem amy_local_calls :
  ∀ (L I : ℕ),
  (L : ℚ) / I = 5 / 2 →
  L / (I + 3) = 5 / 3 →
  L = 15 :=
by
  sorry

end amy_local_calls_l2492_249262


namespace tangent_circles_radius_l2492_249239

theorem tangent_circles_radius (r : ℝ) (R : ℝ) : 
  r > 0 → 
  (∃ (A B C : ℝ × ℝ) (O : ℝ × ℝ),
    -- Three circles with centers A, B, C and radius r are externally tangent to each other
    dist A B = 2 * r ∧ 
    dist B C = 2 * r ∧ 
    dist C A = 2 * r ∧
    -- These three circles are internally tangent to a larger circle with center O and radius R
    dist O A = R - r ∧
    dist O B = R - r ∧
    dist O C = R - r) →
  -- Then the radius of the large circle is 2(√3 + 1) when r = 2
  r = 2 → R = 2 * (Real.sqrt 3 + 1) := by
sorry


end tangent_circles_radius_l2492_249239


namespace dorothy_score_l2492_249255

theorem dorothy_score (tatuya ivanna dorothy : ℚ) 
  (h1 : tatuya = 2 * ivanna)
  (h2 : ivanna = (3/5) * dorothy)
  (h3 : (tatuya + ivanna + dorothy) / 3 = 84) :
  dorothy = 90 := by
  sorry

end dorothy_score_l2492_249255


namespace range_of_m_l2492_249259

def p (x : ℝ) : Prop := |x - 4| ≤ 6

def q (x m : ℝ) : Prop := x^2 - m^2 - 2*x + 1 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) →
  (∃ x, p x ∧ ¬q x m) →
  m ∈ Set.Ici 9 :=
sorry

end range_of_m_l2492_249259


namespace probability_not_hearing_favorite_song_l2492_249217

-- Define the number of songs
def num_songs : ℕ := 8

-- Define the length of the shortest song
def shortest_song_length : ℕ := 1

-- Define the length of the favorite song
def favorite_song_length : ℕ := 5

-- Define the duration we're considering
def considered_duration : ℕ := 7

-- Function to calculate song length based on its position
def song_length (position : ℕ) : ℕ :=
  shortest_song_length + position - 1

-- Theorem stating the probability of not hearing every second of the favorite song
theorem probability_not_hearing_favorite_song :
  let total_arrangements := num_songs.factorial
  let favorable_arrangements := (num_songs - 1).factorial + (num_songs - 2).factorial
  (total_arrangements - favorable_arrangements) / total_arrangements = 6 / 7 :=
sorry

end probability_not_hearing_favorite_song_l2492_249217


namespace sqrt_equation_solution_l2492_249211

theorem sqrt_equation_solution :
  let x : ℝ := 3721 / 256
  Real.sqrt x + Real.sqrt (x + 3) = 8 := by sorry

end sqrt_equation_solution_l2492_249211


namespace intersection_range_l2492_249223

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line
def line (k x : ℝ) (y : ℝ) : Prop := y = k*x - 2

-- Define the condition for intersection
def intersection_condition (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line k x y ∧
    ∃ (x' y' : ℝ), circle_C x' y' ∧
      (x - x')^2 + (y - y')^2 ≤ 4

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, intersection_condition k ↔ 0 ≤ k ∧ k ≤ 4/3 :=
by sorry

end intersection_range_l2492_249223


namespace cosine_inequality_solution_l2492_249284

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 Real.pi) → 
  ((∀ x ∈ Set.Icc 0 Real.pi, Real.cos (x + y) ≥ Real.cos x * Real.cos y) ↔ 
   (y = 0 ∨ y = Real.pi)) := by
  sorry

end cosine_inequality_solution_l2492_249284


namespace limit_of_r_as_m_approaches_zero_l2492_249250

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + 2*x - 6

-- Define L(m) as the smaller root of x^2 + 2x - (m + 6) = 0
noncomputable def L (m : ℝ) : ℝ := -1 - Real.sqrt (m + 7)

-- Define r as a function of m
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- Theorem statement
theorem limit_of_r_as_m_approaches_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ → |r m - 1 / Real.sqrt 7| < ε :=
sorry

end limit_of_r_as_m_approaches_zero_l2492_249250


namespace sum_of_squares_l2492_249229

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end sum_of_squares_l2492_249229


namespace smallest_page_number_l2492_249267

theorem smallest_page_number : ∃ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 13 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 4 ∣ m ∧ 13 ∣ m) → n ≤ m :=
by sorry

end smallest_page_number_l2492_249267


namespace volleyball_team_median_age_l2492_249214

/-- Represents the age distribution of the volleyball team --/
def AgeDistribution : List (Nat × Nat) :=
  [(18, 3), (19, 5), (20, 2), (21, 1), (22, 1)]

/-- The total number of team members --/
def TotalMembers : Nat := 12

/-- Calculates the median age of the team --/
def medianAge (dist : List (Nat × Nat)) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median age of the team is 19 --/
theorem volleyball_team_median_age :
  medianAge AgeDistribution TotalMembers = 19 := by
  sorry

end volleyball_team_median_age_l2492_249214


namespace complex_magnitude_thirteen_l2492_249283

theorem complex_magnitude_thirteen (x : ℝ) : 
  x > 0 → (Complex.abs (3 + x * Complex.I) = 13 ↔ x = 8 * Real.sqrt 5) :=
by sorry

end complex_magnitude_thirteen_l2492_249283


namespace fractional_equation_solution_l2492_249236

theorem fractional_equation_solution : 
  ∃ x : ℝ, (1 / x = 2 / (x + 3)) ∧ (x = 3) := by
  sorry

end fractional_equation_solution_l2492_249236


namespace cyclic_fraction_inequality_l2492_249289

theorem cyclic_fraction_inequality (a b x y z : ℝ) (ha : a > 0) (hb : b > 0) :
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end cyclic_fraction_inequality_l2492_249289


namespace red_star_company_profit_optimization_l2492_249210

/-- Red Star Company's profit optimization problem -/
theorem red_star_company_profit_optimization :
  -- Define the cost per item
  let cost : ℝ := 40
  -- Define the initial sales volume (in thousand items)
  let initial_sales : ℝ := 5
  -- Define the price-sales relationship function
  let sales (x : ℝ) : ℝ :=
    if x ≤ 50 then initial_sales else 10 - 0.1 * x
  -- Define the profit function without donation
  let profit (x : ℝ) : ℝ := (x - cost) * sales x
  -- Define the profit function with donation
  let profit_with_donation (x a : ℝ) : ℝ := (x - cost - a) * sales x
  -- State the conditions and the theorem
  ∀ x a : ℝ,
    cost ≤ x ∧ x ≤ 100 →
    -- Maximum profit occurs at x = 70
    profit 70 = 90 ∧
    -- Maximum profit is 90 million yuan
    (∀ y, cost ≤ y ∧ y ≤ 100 → profit y ≤ 90) ∧
    -- With donation a = 4, maximum profit is 78 million yuan
    (x ≤ 70 → profit_with_donation x 4 ≤ 78) ∧
    profit_with_donation 70 4 = 78 := by
  sorry

end red_star_company_profit_optimization_l2492_249210


namespace sqrt_difference_square_l2492_249245

theorem sqrt_difference_square : (Real.sqrt 7 + Real.sqrt 6) * (Real.sqrt 7 - Real.sqrt 6) = 1 := by
  sorry

end sqrt_difference_square_l2492_249245


namespace pet_store_cats_l2492_249205

theorem pet_store_cats (initial_siamese : ℕ) (cats_sold : ℕ) (cats_remaining : ℕ) 
  (h1 : initial_siamese = 13)
  (h2 : cats_sold = 10)
  (h3 : cats_remaining = 8) :
  ∃ initial_house : ℕ, 
    initial_house = 5 ∧ 
    initial_siamese + initial_house - cats_sold = cats_remaining :=
  sorry

end pet_store_cats_l2492_249205


namespace triple_composition_even_l2492_249282

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem triple_composition_even
  (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by
  sorry

end triple_composition_even_l2492_249282


namespace ellipse_focus_distance_l2492_249248

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_a_ge_b : a ≥ b

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The theorem stating the property of the ellipse -/
theorem ellipse_focus_distance (e : Ellipse) (p : PointOnEllipse e) 
  (h_focus1 : ℝ) (h_on_ellipse : e.a = 5 ∧ e.b = 4) :
  h_focus1 = 8 → ∃ h_focus2 : ℝ, h_focus2 = 2 ∧ h_focus1 + h_focus2 = 2 * e.a := by
  sorry

end ellipse_focus_distance_l2492_249248


namespace unique_three_digit_factorial_sum_l2492_249269

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_of_digit_factorials (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.map factorial |> List.sum

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = sum_of_digit_factorials n :=
by
  use 145
  sorry

end unique_three_digit_factorial_sum_l2492_249269


namespace base_8_to_7_conversion_l2492_249241

def base_8_to_10 (n : ℕ) : ℕ := 
  5 * 8^2 + 3 * 8^1 + 6 * 8^0

def base_10_to_7 (n : ℕ) : ℕ := 
  1 * 7^3 + 0 * 7^2 + 1 * 7^1 + 0 * 7^0

theorem base_8_to_7_conversion : 
  base_10_to_7 (base_8_to_10 536) = 1010 := by
  sorry

end base_8_to_7_conversion_l2492_249241


namespace negation_of_existence_negation_of_proposition_l2492_249277

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x > Real.sin x) ↔ (∀ x : ℝ, x ≤ Real.sin x) := by sorry

end negation_of_existence_negation_of_proposition_l2492_249277


namespace production_improvement_l2492_249281

/-- Represents the production efficiency of a team --/
structure ProductionTeam where
  initial_time : ℕ  -- Initial completion time in hours
  ab_swap_reduction : ℕ  -- Time reduction when swapping A and B
  cd_swap_reduction : ℕ  -- Time reduction when swapping C and D

/-- Calculates the time reduction when swapping both A with B and C with D --/
def time_reduction (team : ProductionTeam) : ℕ :=
  -- Definition to be proved
  108

theorem production_improvement (team : ProductionTeam) 
  (h1 : team.initial_time = 9)
  (h2 : team.ab_swap_reduction = 1)
  (h3 : team.cd_swap_reduction = 1) :
  time_reduction team = 108 := by
  sorry


end production_improvement_l2492_249281


namespace incorrect_operation_correction_l2492_249278

theorem incorrect_operation_correction (x : ℝ) : 
  x - 4.3 = 8.8 → x + 4.3 = 17.4 := by
  sorry

end incorrect_operation_correction_l2492_249278


namespace income_tax_problem_l2492_249279

theorem income_tax_problem (q : ℝ) :
  let tax_rate_low := q / 100
  let tax_rate_high := (q + 3) / 100
  let total_tax_rate := (q + 0.5) / 100
  let income := 36000
  let tax_low := tax_rate_low * 30000
  let tax_high := tax_rate_high * (income - 30000)
  tax_low + tax_high = total_tax_rate * income := by sorry

end income_tax_problem_l2492_249279


namespace intersection_distance_two_points_condition_l2492_249203

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := ρ * (3 * Real.cos θ - 4 * Real.sin θ) = 2

-- Define the curve C in polar coordinates
def curve_C (ρ m : ℝ) : Prop := ρ = m ∧ m > 0

-- Theorem for part (1)
theorem intersection_distance : 
  ∃ (ρ : ℝ), line_l ρ 0 ∧ ρ = 2/3 :=
sorry

-- Theorem for part (2)
theorem two_points_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = m^2 ∧ 
   (|3*x - 4*y - 2| / 5 = 1/5 ∨ |3*x - 4*y - 2| / 5 = 1/5) ∧
   (∀ (x' y' : ℝ), x'^2 + y'^2 = m^2 ∧ |3*x' - 4*y' - 2| / 5 = 1/5 → 
     (x' = x ∧ y' = y) ∨ (x' = -x ∧ y' = -y))) ↔ 
  (1/5 < m ∧ m < 3/5) :=
sorry

end intersection_distance_two_points_condition_l2492_249203


namespace red_balls_count_l2492_249206

theorem red_balls_count (total : ℕ) (prob : ℚ) (red : ℕ) : 
  total = 20 → prob = 1/4 → (red : ℚ)/total = prob → red = 5 := by
  sorry

end red_balls_count_l2492_249206


namespace triangle_inequality_and_side_length_relations_l2492_249293

/-- Given a triangle with side lengths a, b, c, prove the existence of a triangle
    with side lengths √a, √b, √c and the inequality involving these lengths. -/
theorem triangle_inequality_and_side_length_relations
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a + b > c)
  (hbc : b + c > a)
  (hca : c + a > b) :
  (∃ (x y z : ℝ), x = Real.sqrt a ∧ y = Real.sqrt b ∧ z = Real.sqrt c ∧
    x + y > z ∧ y + z > x ∧ z + x > y) ∧
  Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c ∧
  a + b + c < 2 * (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a)) :=
by sorry

end triangle_inequality_and_side_length_relations_l2492_249293


namespace sin_sum_alpha_beta_l2492_249249

theorem sin_sum_alpha_beta (α β : Real) 
  (h1 : 13 * Real.sin α + 5 * Real.cos β = 9)
  (h2 : 13 * Real.cos α + 5 * Real.sin β = 15) : 
  Real.sin (α + β) = 56 / 65 := by
sorry

end sin_sum_alpha_beta_l2492_249249


namespace solution_set_inequality_l2492_249218

theorem solution_set_inequality (x : ℝ) :
  (x * (x + 2) < 3) ↔ (-3 < x ∧ x < 1) := by
  sorry

end solution_set_inequality_l2492_249218


namespace bronze_medals_count_l2492_249254

theorem bronze_medals_count (total : ℕ) (gold : ℕ) (silver : ℕ) (bronze : ℕ)
  (h_total : total = 67)
  (h_gold : gold = 19)
  (h_silver : silver = 32)
  (h_sum : total = gold + silver + bronze) :
  bronze = 16 :=
by sorry

end bronze_medals_count_l2492_249254


namespace concert_ticket_prices_l2492_249220

theorem concert_ticket_prices (x : ℕ) : 
  (∃ a b : ℕ, a * x = 80 ∧ b * x = 100) → 
  (Finset.filter (fun d => d ∣ 80 ∧ d ∣ 100) (Finset.range 101)).card = 6 := by
  sorry

end concert_ticket_prices_l2492_249220


namespace line_always_intersects_ellipse_l2492_249244

/-- The range of m for which the line y = kx + 1 always intersects the ellipse x²/5 + y²/m = 1 -/
theorem line_always_intersects_ellipse (k : ℝ) (m : ℝ) :
  (∀ x y, y = k * x + 1 → x^2 / 5 + y^2 / m = 1) ↔ m ≥ 1 ∧ m ≠ 5 :=
sorry

end line_always_intersects_ellipse_l2492_249244


namespace factory_production_theorem_l2492_249296

/-- Represents a production line with its output and sample size -/
structure ProductionLine where
  output : ℕ
  sample : ℕ

/-- Represents the factory's production data -/
structure FactoryProduction where
  total_output : ℕ
  line_a : ProductionLine
  line_b : ProductionLine
  line_c : ProductionLine

/-- Checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- The main theorem about the factory's production -/
theorem factory_production_theorem (f : FactoryProduction) :
  f.total_output = 16800 ∧
  isArithmeticSequence f.line_a.sample f.line_b.sample f.line_c.sample ∧
  f.line_a.output + f.line_b.output + f.line_c.output = f.total_output →
  f.line_b.output = 5600 := by
  sorry

end factory_production_theorem_l2492_249296


namespace min_m_value_l2492_249200

theorem min_m_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2*y + 2/x + 1/y = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y + 2/x + 1/y = 6 → m ≥ x + 2*y) ∧ 
  (∀ (m' : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y + 2/x + 1/y = 6 → m' ≥ x + 2*y) → m' ≥ m) ∧
  m = 4 :=
sorry

end min_m_value_l2492_249200


namespace min_value_of_sum_of_roots_l2492_249294

theorem min_value_of_sum_of_roots (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 13) + Real.sqrt (x^2 - 10*x + 26) ≥ 5 := by
  sorry

end min_value_of_sum_of_roots_l2492_249294


namespace vector_sum_magnitude_l2492_249202

theorem vector_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : a • b = 0) 
  (h2 : ‖a‖ = 2) 
  (h3 : ‖b‖ = 1) : 
  ‖a + 2 • b‖ = 2 * Real.sqrt 2 := by
  sorry

end vector_sum_magnitude_l2492_249202


namespace expected_groups_formula_l2492_249224

/-- A sequence of k zeros and m ones arranged in random order -/
structure BinarySequence where
  k : ℕ
  m : ℕ

/-- The expected number of alternating groups in a BinarySequence -/
noncomputable def expectedGroups (seq : BinarySequence) : ℝ :=
  1 + (2 * seq.k * seq.m : ℝ) / (seq.k + seq.m)

/-- Theorem stating the expected number of alternating groups -/
theorem expected_groups_formula (seq : BinarySequence) :
    expectedGroups seq = 1 + (2 * seq.k * seq.m : ℝ) / (seq.k + seq.m) := by
  sorry

end expected_groups_formula_l2492_249224


namespace average_chocolate_pieces_per_cookie_l2492_249263

theorem average_chocolate_pieces_per_cookie 
  (total_cookies : ℕ) 
  (chocolate_chips : ℕ) 
  (mms_ratio : ℚ) :
  total_cookies = 48 →
  chocolate_chips = 108 →
  mms_ratio = 1/3 →
  (chocolate_chips + (mms_ratio * chocolate_chips)) / total_cookies = 3 := by
  sorry

end average_chocolate_pieces_per_cookie_l2492_249263


namespace problem_solution_l2492_249225

/-- The distance between two points A and B, given the conditions of the problem -/
def distance_AB : ℝ := 1656

/-- Jun Jun's speed -/
def v_jun : ℝ := 14

/-- Ping's speed -/
def v_ping : ℝ := 9

/-- Distance from C to the point where Jun Jun turns back -/
def d_turn : ℝ := 100

/-- Distance from C to the point where Jun Jun catches up with Ping -/
def d_catchup : ℝ := 360

theorem problem_solution :
  ∃ (d_AC d_BC : ℝ),
    d_AC + d_BC = distance_AB ∧
    d_AC / d_BC = v_jun / v_ping ∧
    d_AC - d_catchup = d_turn + d_catchup ∧
    (d_AC - d_catchup) / (d_BC + d_catchup) = v_ping / v_jun :=
by sorry

end problem_solution_l2492_249225


namespace program_output_l2492_249201

def S : ℕ → ℕ
  | 0 => 1
  | (n + 1) => S n + (2 * (n + 1) - 1)

theorem program_output :
  (S 1 = 2) ∧ (S 2 = 5) ∧ (S 3 = 10) := by
  sorry

end program_output_l2492_249201


namespace polynomial_factorization_l2492_249266

theorem polynomial_factorization (x : ℝ) (h : x^3 ≠ 1) :
  x^12 + x^6 + 1 = (x^6 + x^3 + 1) * (x^6 - x^3 + 1) := by
  sorry

end polynomial_factorization_l2492_249266


namespace traced_path_is_asterisk_l2492_249276

/-- A regular n-gon in the plane -/
structure RegularNGon where
  n : ℕ
  center : ℝ × ℝ
  vertices : Fin n → ℝ × ℝ

/-- Triangle formed by two adjacent vertices and the center of a regular n-gon -/
structure TriangleABO (ngon : RegularNGon) where
  A : Fin ngon.n
  B : Fin ngon.n
  hAdjacent : (A.val + 1) % ngon.n = B.val

/-- The path traced by point O when triangle ABO glides around the n-gon -/
def tracedPath (ngon : RegularNGon) : Set (ℝ × ℝ) := sorry

/-- An asterisk consisting of n segments emanating from the center -/
def asterisk (center : ℝ × ℝ) (n : ℕ) (length : ℝ) : Set (ℝ × ℝ) := sorry

/-- Main theorem: The path traced by O forms an asterisk -/
theorem traced_path_is_asterisk (ngon : RegularNGon) :
  ∃ (length : ℝ), tracedPath ngon = asterisk ngon.center ngon.n length := by sorry

end traced_path_is_asterisk_l2492_249276


namespace mikes_second_job_hours_l2492_249290

/-- Given Mike's total wages, wages from his first job, and hourly rate at his second job,
    calculate the number of hours he worked at his second job. -/
theorem mikes_second_job_hours
  (total_wages : ℕ)
  (first_job_wages : ℕ)
  (second_job_hourly_rate : ℕ)
  (h1 : total_wages = 160)
  (h2 : first_job_wages = 52)
  (h3 : second_job_hourly_rate = 9) :
  (total_wages - first_job_wages) / second_job_hourly_rate = 12 := by
  sorry

end mikes_second_job_hours_l2492_249290


namespace xy_range_l2492_249253

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + x*y = 30) :
  12 < x*y ∧ x*y < 870 := by
  sorry

end xy_range_l2492_249253


namespace jason_gave_nine_cards_l2492_249274

/-- The number of Pokemon cards Jason started with -/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason has left -/
def remaining_cards : ℕ := 4

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem jason_gave_nine_cards : cards_given = 9 := by
  sorry

end jason_gave_nine_cards_l2492_249274


namespace factorization_of_a_squared_minus_4ab_l2492_249234

theorem factorization_of_a_squared_minus_4ab (a b : ℝ) :
  a^2 - 4*a*b = a*(a - 4*b) := by sorry

end factorization_of_a_squared_minus_4ab_l2492_249234


namespace circumradius_inradius_inequality_l2492_249291

/-- A triangle with its circumradius and inradius -/
structure Triangle where
  R : ℝ  -- Circumradius
  r : ℝ  -- Inradius

/-- Predicate to check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  t.R = 2 * t.r

theorem circumradius_inradius_inequality (t : Triangle) :
  t.R ≥ 2 * t.r ∧ (t.R = 2 * t.r ↔ is_equilateral t) :=
sorry

end circumradius_inradius_inequality_l2492_249291
