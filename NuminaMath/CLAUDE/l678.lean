import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_range_l678_67818

theorem quadratic_roots_range (h1 : ∃ x : ℝ, Real.log x < 0)
  (h2 : ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0))
  (m : ℝ) :
  -2 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l678_67818


namespace NUMINAMATH_CALUDE_max_monthly_profit_l678_67870

/-- Represents the monthly profit function for a product with given cost and pricing conditions. -/
def monthly_profit (x : ℕ) : ℚ :=
  -10 * x^2 + 110 * x + 2100

/-- Theorem stating the maximum monthly profit and the optimal selling prices. -/
theorem max_monthly_profit :
  (∀ x : ℕ, 0 < x → x ≤ 15 → monthly_profit x ≤ 2400) ∧
  monthly_profit 5 = 2400 ∧
  monthly_profit 6 = 2400 :=
sorry

end NUMINAMATH_CALUDE_max_monthly_profit_l678_67870


namespace NUMINAMATH_CALUDE_diamond_calculation_l678_67898

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a + 1 / b

-- Theorem statement
theorem diamond_calculation :
  (diamond (diamond 3 4) 5) - (diamond 3 (diamond 4 5)) = 89 / 420 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l678_67898


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l678_67883

/-- Given three square regions A, B, and C, where the perimeter of A is 16 units
    and the perimeter of B is 32 units, prove that the ratio of the area of
    region A to the area of region C is 1/9. -/
theorem area_ratio_of_squares (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) →
  (4 * a = 16) → (4 * b = 32) → (c = 3 * a) →
  (a^2) / (c^2) = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l678_67883


namespace NUMINAMATH_CALUDE_power_and_division_simplification_l678_67847

theorem power_and_division_simplification : 1^567 + 3^5 / 3^3 - 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_and_division_simplification_l678_67847


namespace NUMINAMATH_CALUDE_max_value_of_a_l678_67871

theorem max_value_of_a (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  a ≤ Real.sqrt 6 / 3 ∧ ∃ (a₀ : ℝ), a₀ = Real.sqrt 6 / 3 ∧ 
  ∃ (b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 0 ∧ a₀^2 + b₀^2 + c₀^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l678_67871


namespace NUMINAMATH_CALUDE_ratio_composition_l678_67845

theorem ratio_composition (a b c : ℚ) 
  (h1 : a / b = 11 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 11 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_composition_l678_67845


namespace NUMINAMATH_CALUDE_halfDollarProbabilityIs3_16_l678_67840

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | HalfDollar
  | Quarter

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Dime => 10
  | Coin.HalfDollar => 50
  | Coin.Quarter => 25

/-- The total value of each coin type in cents -/
def totalValue : Coin → ℕ
  | Coin.Dime => 2000
  | Coin.HalfDollar => 3000
  | Coin.Quarter => 1500

/-- The number of coins of each type -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Dime + coinCount Coin.HalfDollar + coinCount Coin.Quarter

/-- The probability of selecting a half-dollar -/
def halfDollarProbability : ℚ := coinCount Coin.HalfDollar / totalCoins

theorem halfDollarProbabilityIs3_16 : halfDollarProbability = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_halfDollarProbabilityIs3_16_l678_67840


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l678_67821

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 6*x₁ + 9*k = 0 ∧ x₂^2 - 6*x₂ + 9*k = 0) ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l678_67821


namespace NUMINAMATH_CALUDE_final_selling_price_approx_1949_l678_67807

/-- Calculate the final selling price of a cycle, helmet, and safety lights --/
def calculate_final_selling_price (cycle_cost helmet_cost safety_light_cost : ℚ) 
  (num_safety_lights : ℕ) (cycle_discount tax_rate cycle_loss helmet_profit transaction_fee : ℚ) : ℚ :=
  let cycle_discounted := cycle_cost * (1 - cycle_discount)
  let total_cost := cycle_discounted + helmet_cost + (safety_light_cost * num_safety_lights)
  let total_with_tax := total_cost * (1 + tax_rate)
  let cycle_selling := cycle_discounted * (1 - cycle_loss)
  let helmet_selling := helmet_cost * (1 + helmet_profit)
  let safety_lights_selling := safety_light_cost * num_safety_lights
  let total_selling := cycle_selling + helmet_selling + safety_lights_selling
  let final_price := total_selling * (1 - transaction_fee)
  final_price

/-- Theorem stating that the final selling price is approximately 1949 --/
theorem final_selling_price_approx_1949 : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_final_selling_price 1400 400 200 2 (10/100) (5/100) (12/100) (25/100) (3/100) - 1949| < ε :=
sorry

end NUMINAMATH_CALUDE_final_selling_price_approx_1949_l678_67807


namespace NUMINAMATH_CALUDE_charlie_prob_different_colors_l678_67878

/-- Represents the number of marbles of each color -/
def num_marbles : ℕ := 3

/-- Represents the total number of marbles -/
def total_marbles : ℕ := 3 * num_marbles

/-- Represents the number of marbles each person takes -/
def marbles_per_person : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the total number of ways the marbles can be drawn -/
def total_ways : ℕ := 
  (choose total_marbles marbles_per_person) * 
  (choose (total_marbles - marbles_per_person) marbles_per_person) * 
  (choose marbles_per_person marbles_per_person)

/-- Calculates the number of favorable outcomes for Charlie -/
def favorable_outcomes : ℕ := 
  (choose num_marbles 2) * (choose num_marbles 2) * (choose num_marbles 2)

/-- The probability of Charlie drawing three different colored marbles -/
theorem charlie_prob_different_colors : 
  (favorable_outcomes : ℚ) / total_ways = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_charlie_prob_different_colors_l678_67878


namespace NUMINAMATH_CALUDE_inequality_solution_l678_67826

def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 ∨ m = -12 then
    {x | x ≠ m / 6}
  else if m < -12 ∨ m > 0 then
    {x | x < (m - Real.sqrt (m^2 + 12*m)) / 6 ∨ x > (m + Real.sqrt (m^2 + 12*m)) / 6}
  else
    Set.univ

theorem inequality_solution (m : ℝ) :
  {x : ℝ | 3 * x^2 - m * x - m > 0} = solution_set m :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l678_67826


namespace NUMINAMATH_CALUDE_bennys_seashells_l678_67808

theorem bennys_seashells (initial_seashells given_away_seashells : ℚ) 
  (h1 : initial_seashells = 66.5)
  (h2 : given_away_seashells = 52.5) :
  initial_seashells - given_away_seashells = 14 :=
by sorry

end NUMINAMATH_CALUDE_bennys_seashells_l678_67808


namespace NUMINAMATH_CALUDE_inequality_proof_l678_67816

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (ha1 : a ≥ 1) (hb1 : b ≥ 1) (hc1 : c ≥ 1)
  (habcd : a * b * c * d = 1) :
  1 / (a^2 - a + 1)^2 + 1 / (b^2 - b + 1)^2 + 
  1 / (c^2 - c + 1)^2 + 1 / (d^2 - d + 1)^2 ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l678_67816


namespace NUMINAMATH_CALUDE_triangle_third_side_l678_67867

theorem triangle_third_side (a b c : ℝ) (θ : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : θ = Real.pi / 3) :
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos θ → c = Real.sqrt 57 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l678_67867


namespace NUMINAMATH_CALUDE_fraction_equality_l678_67823

theorem fraction_equality : (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l678_67823


namespace NUMINAMATH_CALUDE_kwi_wins_l678_67813

/-- Represents a frog in the race -/
structure Frog where
  name : String
  jump_length : ℚ
  jumps_per_time_unit : ℚ

/-- Calculates the time taken to complete the race for a given frog -/
def race_time (f : Frog) (race_distance : ℚ) : ℚ :=
  (race_distance / f.jump_length) / f.jumps_per_time_unit

/-- The race distance in decimeters -/
def total_race_distance : ℚ := 400

/-- Kwa, the first frog -/
def kwa : Frog := ⟨"Kwa", 6, 2⟩

/-- Kwi, the second frog -/
def kwi : Frog := ⟨"Kwi", 4, 3⟩

theorem kwi_wins : race_time kwi total_race_distance < race_time kwa total_race_distance := by
  sorry

end NUMINAMATH_CALUDE_kwi_wins_l678_67813


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l678_67885

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_count : vertices.card = 20
  edge_count : edges.card = 30
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that are endpoints of an edge in a regular dodecahedron -/
def edge_endpoint_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem: The probability of randomly selecting two vertices that are endpoints of an edge in a regular dodecahedron is 3/19 -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_endpoint_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l678_67885


namespace NUMINAMATH_CALUDE_optimal_box_volume_l678_67882

/-- The volume of an open box made from a 48m x 36m sheet by cutting squares from corners -/
def box_volume (x : ℝ) : ℝ := (48 - 2*x) * (36 - 2*x) * x

/-- The derivative of the box volume function -/
def box_volume_derivative (x : ℝ) : ℝ := 1728 - 336*x + 12*x^2

theorem optimal_box_volume :
  ∃ (x : ℝ),
    x = 12 ∧
    (∀ y : ℝ, 0 < y ∧ y < 24 → box_volume y ≤ box_volume x) ∧
    box_volume x = 3456 :=
by sorry

end NUMINAMATH_CALUDE_optimal_box_volume_l678_67882


namespace NUMINAMATH_CALUDE_similar_polygons_area_sum_l678_67843

/-- Given two similar polygons with corresponding sides a and b, 
    we can construct a third similar polygon with side c -/
theorem similar_polygons_area_sum 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (similar : ∃ (k : ℝ), k > 0 ∧ b = k * a) :
  ∃ (c : ℝ), 
    c > 0 ∧ 
    c^2 = a^2 + b^2 ∧ 
    ∃ (k : ℝ), k > 0 ∧ c = k * a ∧
    c^2 / a^2 = (a^2 + b^2) / a^2 :=
by sorry

end NUMINAMATH_CALUDE_similar_polygons_area_sum_l678_67843


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l678_67810

theorem part_to_whole_ratio (N : ℝ) (h1 : (1/3) * (2/5) * N = 17) (h2 : 0.4 * N = 204) : 
  17 / N = 1 / 30 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l678_67810


namespace NUMINAMATH_CALUDE_billion_product_without_zeros_l678_67860

theorem billion_product_without_zeros :
  ∃ (a b : ℕ), 
    a * b = 1000000000 ∧ 
    (∀ d : ℕ, d > 0 → d ≤ 9 → (a / 10^d) % 10 ≠ 0) ∧
    (∀ d : ℕ, d > 0 → d ≤ 9 → (b / 10^d) % 10 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_billion_product_without_zeros_l678_67860


namespace NUMINAMATH_CALUDE_determinant_scaling_l678_67848

theorem determinant_scaling (x y z a b c p q r : ℝ) :
  Matrix.det !![x, y, z; a, b, c; p, q, r] = 2 →
  Matrix.det !![3*x, 3*y, 3*z; 3*a, 3*b, 3*c; 3*p, 3*q, 3*r] = 54 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l678_67848


namespace NUMINAMATH_CALUDE_dartboard_angles_l678_67859

theorem dartboard_angles (p₁ p₂ : ℝ) (θ₁ θ₂ : ℝ) :
  p₁ = 1/8 →
  p₂ = 2 * p₁ →
  p₁ = θ₁ / 360 →
  p₂ = θ₂ / 360 →
  θ₁ = 45 ∧ θ₂ = 90 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_angles_l678_67859


namespace NUMINAMATH_CALUDE_inequality_proof_l678_67829

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) > 8 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l678_67829


namespace NUMINAMATH_CALUDE_inequality_proof_l678_67861

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 3*b^3) / (5*a + b) + (b^3 + 3*c^3) / (5*b + c) + (c^3 + 3*a^3) / (5*c + a) ≥ 2/3 * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l678_67861


namespace NUMINAMATH_CALUDE_polynomial_exists_for_non_squares_l678_67837

-- Define the polynomial P(x,y,z)
def P (x y z : ℕ) : ℤ :=
  (1 - 2013 * (z - 1) * (z - 2)) * ((x + y - 1)^2 + 2*y - 2 + z)

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

-- State the theorem
theorem polynomial_exists_for_non_squares :
  ∀ n : ℕ, n > 0 →
    (¬ is_perfect_square n ↔ ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ P x y z = n) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_exists_for_non_squares_l678_67837


namespace NUMINAMATH_CALUDE_problem_solution_l678_67811

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m|

-- State the theorem
theorem problem_solution :
  -- Given conditions
  (∀ x : ℝ, f x 2 ≤ 3 ↔ x ∈ Set.Icc (-1) 5) →
  -- Part I: m = 2
  (∃ m : ℝ, ∀ x : ℝ, f x m ≤ 3 ↔ x ∈ Set.Icc (-1) 5) ∧ 
  -- Part II: Minimum value of a² + b² + c² is 2/3
  (∀ a b c : ℝ, a - 2*b + c = 2 → a^2 + b^2 + c^2 ≥ 2/3) ∧
  (∃ a b c : ℝ, a - 2*b + c = 2 ∧ a^2 + b^2 + c^2 = 2/3) :=
by sorry


end NUMINAMATH_CALUDE_problem_solution_l678_67811


namespace NUMINAMATH_CALUDE_parabola_equation_l678_67827

/-- Represents a parabola -/
structure Parabola where
  -- The equation of the parabola in the form y² = 2px or x² = 2py
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

/-- Checks if the parabola has a given focus -/
def Parabola.hasFocus (p : Parabola) (fx fy : ℝ) : Prop :=
  ∃ (a : ℝ), (p.equation = fun x y ↦ (y - fy)^2 = 4*a*(x - fx)) ∨
             (p.equation = fun x y ↦ (x - fx)^2 = 4*a*(y - fy))

/-- Checks if the parabola has a given directrix -/
def Parabola.hasDirectrix (p : Parabola) (d : ℝ) : Prop :=
  ∃ (a : ℝ), (p.equation = fun x y ↦ y^2 = 4*a*(x + a)) ∧ d = -a ∨
             (p.equation = fun x y ↦ x^2 = 4*a*(y + a)) ∧ d = -a

theorem parabola_equation (p : Parabola) :
  p.hasFocus (-2) 0 →
  p.hasDirectrix (-1) →
  p.contains 1 2 →
  (p.equation = fun x y ↦ y^2 = 4*x) ∨
  (p.equation = fun x y ↦ x^2 = 1/2*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l678_67827


namespace NUMINAMATH_CALUDE_sqrt_6_simplest_l678_67880

-- Define the criteria for simplest square root form
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℚ, x ≠ y^2 ∧ (∀ z : ℕ, z > 1 → ¬ (∃ w : ℕ, x = z * w^2))

-- Define the set of square roots to compare
def sqrt_set : Set ℝ := {Real.sqrt 0.2, Real.sqrt (1/2), Real.sqrt 6, Real.sqrt 12}

-- Theorem statement
theorem sqrt_6_simplest :
  ∀ x ∈ sqrt_set, x ≠ Real.sqrt 6 → ¬(is_simplest_sqrt x) ∧ is_simplest_sqrt (Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_sqrt_6_simplest_l678_67880


namespace NUMINAMATH_CALUDE_tank_weight_l678_67895

/-- Proves that the weight of a water tank filled to 80% capacity is 1360 pounds. -/
theorem tank_weight (tank_capacity : ℝ) (empty_tank_weight : ℝ) (water_weight_per_gallon : ℝ) :
  tank_capacity = 200 →
  empty_tank_weight = 80 →
  water_weight_per_gallon = 8 →
  empty_tank_weight + 0.8 * tank_capacity * water_weight_per_gallon = 1360 := by
  sorry

end NUMINAMATH_CALUDE_tank_weight_l678_67895


namespace NUMINAMATH_CALUDE_invalid_triangle_1_invalid_triangle_2_l678_67824

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the property that triangle angles sum to 180 degrees
def valid_triangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Theorem: A triangle with angles 90°, 60°, and 60° cannot exist
theorem invalid_triangle_1 : 
  ¬ ∃ (t : Triangle), t.angle1 = 90 ∧ t.angle2 = 60 ∧ t.angle3 = 60 ∧ valid_triangle t :=
sorry

-- Theorem: A triangle with angles 90°, 50°, and 50° cannot exist
theorem invalid_triangle_2 :
  ¬ ∃ (t : Triangle), t.angle1 = 90 ∧ t.angle2 = 50 ∧ t.angle3 = 50 ∧ valid_triangle t :=
sorry

end NUMINAMATH_CALUDE_invalid_triangle_1_invalid_triangle_2_l678_67824


namespace NUMINAMATH_CALUDE_banana_distribution_l678_67889

theorem banana_distribution (total_children : ℕ) 
  (original_bananas_per_child : ℕ) 
  (extra_bananas_per_child : ℕ) : 
  total_children = 720 →
  original_bananas_per_child = 2 →
  extra_bananas_per_child = 2 →
  (total_children - (total_children * original_bananas_per_child) / 
   (original_bananas_per_child + extra_bananas_per_child)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l678_67889


namespace NUMINAMATH_CALUDE_handshakes_eight_couples_l678_67833

/-- Represents the number of handshakes in a group of couples with one injured person --/
def handshakes_in_couples_group (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let shaking_people := total_people - 1
  let handshakes_per_person := total_people - 3
  (shaking_people * handshakes_per_person) / 2

/-- Theorem stating that in a group of 8 married couples where everyone shakes hands
    with each other except their spouse and one person doesn't shake hands at all,
    the total number of handshakes is 90 --/
theorem handshakes_eight_couples :
  handshakes_in_couples_group 8 = 90 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_eight_couples_l678_67833


namespace NUMINAMATH_CALUDE_susan_missed_pay_l678_67884

/-- Calculates the missed pay for Susan's vacation --/
def missed_pay (weeks : ℕ) (work_days_per_week : ℕ) (paid_vacation_days : ℕ) 
                (hourly_rate : ℚ) (hours_per_day : ℕ) : ℚ :=
  let total_work_days := weeks * work_days_per_week
  let unpaid_days := total_work_days - paid_vacation_days
  let daily_pay := hourly_rate * hours_per_day
  unpaid_days * daily_pay

/-- Proves that Susan will miss $480 on her vacation --/
theorem susan_missed_pay : 
  missed_pay 2 5 6 15 8 = 480 := by
  sorry

end NUMINAMATH_CALUDE_susan_missed_pay_l678_67884


namespace NUMINAMATH_CALUDE_calculate_gross_profit_gross_profit_calculation_l678_67892

/-- Given a sales price and a gross profit percentage, calculate the gross profit --/
theorem calculate_gross_profit (sales_price : ℝ) (gross_profit_percentage : ℝ) : ℝ :=
  let cost := sales_price / (1 + gross_profit_percentage)
  let gross_profit := cost * gross_profit_percentage
  gross_profit

/-- Prove that given a sales price of $81 and a gross profit that is 170% of cost, 
    the gross profit is equal to $51 --/
theorem gross_profit_calculation :
  calculate_gross_profit 81 1.7 = 51 := by
  sorry

end NUMINAMATH_CALUDE_calculate_gross_profit_gross_profit_calculation_l678_67892


namespace NUMINAMATH_CALUDE_max_gcd_sum_780_l678_67830

theorem max_gcd_sum_780 :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 780 ∧
  ∀ (a b : ℕ), a > 0 → b > 0 → a + b = 780 → Nat.gcd a b ≤ Nat.gcd x y ∧
  Nat.gcd x y = 390 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_sum_780_l678_67830


namespace NUMINAMATH_CALUDE_machine_times_solution_l678_67888

/-- Represents the time taken by three machines to complete a task individually and together -/
structure MachineTimes where
  first : ℝ
  second : ℝ
  third : ℝ
  together : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (t : MachineTimes) : Prop :=
  t.second = t.first + 2 ∧
  t.third = 2 * t.first ∧
  t.together = 8/3

/-- The theorem statement -/
theorem machine_times_solution (t : MachineTimes) :
  satisfies_conditions t → t.first = 6 ∧ t.second = 8 ∧ t.third = 12 := by
  sorry

end NUMINAMATH_CALUDE_machine_times_solution_l678_67888


namespace NUMINAMATH_CALUDE_average_first_n_odd_numbers_l678_67822

/-- The nth odd number -/
def nthOddNumber (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of the first n odd numbers -/
def sumFirstNOddNumbers (n : ℕ) : ℕ := n ^ 2

/-- The average of the first n odd numbers -/
def averageFirstNOddNumbers (n : ℕ) : ℕ := sumFirstNOddNumbers n / n

theorem average_first_n_odd_numbers (n : ℕ) (h : n > 0) :
  averageFirstNOddNumbers n = nthOddNumber n := by
  sorry

end NUMINAMATH_CALUDE_average_first_n_odd_numbers_l678_67822


namespace NUMINAMATH_CALUDE_skating_minutes_needed_for_average_l678_67853

def skating_schedule (days : Nat) (hours_per_day : Nat) : Nat :=
  days * hours_per_day * 60

def total_minutes_8_days : Nat :=
  skating_schedule 6 1 + skating_schedule 2 2

def average_minutes_per_day : Nat := 100

def total_days : Nat := 10

theorem skating_minutes_needed_for_average :
  skating_schedule 6 1 + skating_schedule 2 2 + 400 = total_days * average_minutes_per_day := by
  sorry

end NUMINAMATH_CALUDE_skating_minutes_needed_for_average_l678_67853


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l678_67876

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (ha : ‖a‖ = 3)
  (hb : ‖b‖ = 4)
  (hab : ‖a + b‖ = 2) :
  ‖a - b‖ = Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l678_67876


namespace NUMINAMATH_CALUDE_non_technicians_percentage_l678_67877

/-- Represents the composition of workers in a factory -/
structure Factory where
  total : ℕ
  technicians : ℕ
  permanent_technicians : ℕ
  permanent_non_technicians : ℕ
  temporary : ℕ

/-- The conditions of the factory as described in the problem -/
def factory_conditions (f : Factory) : Prop :=
  f.technicians = f.total / 2 ∧
  f.permanent_technicians = f.technicians / 2 ∧
  f.permanent_non_technicians = (f.total - f.technicians) / 2 ∧
  f.temporary = f.total / 2

/-- The theorem stating that under the given conditions, 
    non-technicians make up 50% of the workforce -/
theorem non_technicians_percentage (f : Factory) 
  (h : factory_conditions f) : 
  (f.total - f.technicians) * 100 / f.total = 50 := by
  sorry


end NUMINAMATH_CALUDE_non_technicians_percentage_l678_67877


namespace NUMINAMATH_CALUDE_max_value_of_sum_l678_67844

theorem max_value_of_sum (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) :
  x + y ≤ 11 / 4 ∧ ∃ (x₀ y₀ : ℝ), 5 * x₀ + 3 * y₀ ≤ 10 ∧ 3 * x₀ + 6 * y₀ ≤ 12 ∧ x₀ + y₀ = 11 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l678_67844


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l678_67809

/-- Theorem: For a parabola x^2 = 2py (p > 0) with a point A(m, 4) on it,
    if the distance from A to its focus is 17/4, then p = 1/2 and m = ±2. -/
theorem parabola_focus_distance (p m : ℝ) : 
  p > 0 →  -- p is positive
  m^2 = 2*p*4 →  -- A(m, 4) is on the parabola
  (m^2 + (4 - p/2)^2)^(1/2) = 17/4 →  -- Distance from A to focus is 17/4
  p = 1/2 ∧ (m = 2 ∨ m = -2) := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l678_67809


namespace NUMINAMATH_CALUDE_largest_integer_less_than_93_remainder_4_mod_7_l678_67831

theorem largest_integer_less_than_93_remainder_4_mod_7 :
  ∃ n : ℕ, n < 93 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 93 ∧ m % 7 = 4 → m ≤ n :=
by
  use 88
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_93_remainder_4_mod_7_l678_67831


namespace NUMINAMATH_CALUDE_total_teachers_is_210_l678_67873

/-- Represents the number of teachers in each category and the sample size -/
structure TeacherData where
  senior : ℕ
  intermediate : ℕ
  sample_size : ℕ
  other_sampled : ℕ

/-- Calculates the total number of teachers given the data -/
def totalTeachers (data : TeacherData) : ℕ :=
  sorry

/-- Theorem stating that given the conditions, the total number of teachers is 210 -/
theorem total_teachers_is_210 (data : TeacherData) 
  (h1 : data.senior = 104)
  (h2 : data.intermediate = 46)
  (h3 : data.sample_size = 42)
  (h4 : data.other_sampled = 12)
  (h5 : ∀ (category : ℕ), (category : ℚ) / (totalTeachers data : ℚ) = (data.sample_size : ℚ) / (totalTeachers data : ℚ)) :
  totalTeachers data = 210 :=
sorry

end NUMINAMATH_CALUDE_total_teachers_is_210_l678_67873


namespace NUMINAMATH_CALUDE_residue_of_11_power_1234_mod_19_l678_67890

theorem residue_of_11_power_1234_mod_19 :
  (11 : ℤ)^1234 ≡ 16 [ZMOD 19] := by sorry

end NUMINAMATH_CALUDE_residue_of_11_power_1234_mod_19_l678_67890


namespace NUMINAMATH_CALUDE_game_wheel_probability_l678_67869

theorem game_wheel_probability (pX pY pZ pW : ℚ) : 
  pX = 1/4 → pY = 1/3 → pW = 1/6 → pX + pY + pZ + pW = 1 → pZ = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_game_wheel_probability_l678_67869


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_a_range_l678_67839

/-- The piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

/-- The theorem stating the range of a for which f is strictly decreasing -/
theorem f_strictly_decreasing_a_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, (f a x₁ - f a x₂) * (x₁ - x₂) < 0) ↔ 0 < a ∧ a ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_f_strictly_decreasing_a_range_l678_67839


namespace NUMINAMATH_CALUDE_negation_equivalence_l678_67886

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ ≤ 0 ∧ x₀^2 ≥ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l678_67886


namespace NUMINAMATH_CALUDE_min_value_expression_l678_67863

theorem min_value_expression (x y z : ℝ) (h : 2 * x * y + y * z > 0) :
  (x^2 + y^2 + z^2) / (2 * x * y + y * z) ≥ 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), 2 * x₀ * y₀ + y₀ * z₀ > 0 ∧
    (x₀^2 + y₀^2 + z₀^2) / (2 * x₀ * y₀ + y₀ * z₀) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l678_67863


namespace NUMINAMATH_CALUDE_range_of_a_for_two_zeros_l678_67881

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then Real.log (1 - x) else Real.sqrt x - a

-- State the theorem
theorem range_of_a_for_two_zeros (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
   (∀ z : ℝ, f a z = 0 → z = x ∨ z = y)) →
  a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_zeros_l678_67881


namespace NUMINAMATH_CALUDE_sara_initial_quarters_l678_67820

/-- The number of quarters Sara had initially -/
def initial_quarters : ℕ := sorry

/-- The number of quarters Sara's dad gave her -/
def quarters_from_dad : ℕ := 49

/-- The total number of quarters Sara has after receiving quarters from her dad -/
def total_quarters : ℕ := 70

/-- Theorem stating that Sara initially had 21 quarters -/
theorem sara_initial_quarters : initial_quarters = 21 := by
  sorry

end NUMINAMATH_CALUDE_sara_initial_quarters_l678_67820


namespace NUMINAMATH_CALUDE_b_profit_l678_67868

-- Define the basic variables
def total_profit : ℕ := 21000

-- Define the investment ratio
def investment_ratio : ℕ := 3

-- Define the time ratio
def time_ratio : ℕ := 2

-- Define the profit sharing ratio
def profit_sharing_ratio : ℕ := investment_ratio * time_ratio

-- Theorem to prove
theorem b_profit (a_investment b_investment : ℕ) (a_time b_time : ℕ) :
  a_investment = investment_ratio * b_investment →
  a_time = time_ratio * b_time →
  (profit_sharing_ratio * b_investment * b_time + b_investment * b_time) * 3000 = total_profit * b_investment * b_time :=
by sorry


end NUMINAMATH_CALUDE_b_profit_l678_67868


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l678_67836

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧
  (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) ∧
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l678_67836


namespace NUMINAMATH_CALUDE_diagonal_sum_lower_bound_l678_67851

/-- Given a convex quadrilateral ABCD with sides a, b, c, d and diagonals x, y,
    where a is the smallest side, prove that x + y ≥ (1 + √3)a -/
theorem diagonal_sum_lower_bound (a b c d x y : ℝ) :
  a > 0 →
  b ≥ a →
  c ≥ a →
  d ≥ a →
  x ≥ a →
  y ≥ a →
  x + y ≥ (1 + Real.sqrt 3) * a :=
by sorry

end NUMINAMATH_CALUDE_diagonal_sum_lower_bound_l678_67851


namespace NUMINAMATH_CALUDE_cuboid_volume_l678_67856

theorem cuboid_volume (a b c : ℝ) : 
  (a^2 + b^2 + c^2 = 16) →  -- space diagonal length is 4
  (a / 4 = 1/2) →           -- edge a forms 60° angle with diagonal
  (b / 4 = 1/2) →           -- edge b forms 60° angle with diagonal
  (c / 4 = 1/2) →           -- edge c forms 60° angle with diagonal
  (a * b * c = 8) :=        -- volume is 8
by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l678_67856


namespace NUMINAMATH_CALUDE_system_solution_l678_67893

theorem system_solution (x y z : ℝ) : 
  (x + y + z = 1 ∧ x^3 + y^3 + z^3 = 1 ∧ x*y*z = -16) ↔ 
  ((x = 1 ∧ y = 4 ∧ z = -4) ∨
   (x = 1 ∧ y = -4 ∧ z = 4) ∨
   (x = 4 ∧ y = 1 ∧ z = -4) ∨
   (x = 4 ∧ y = -4 ∧ z = 1) ∨
   (x = -4 ∧ y = 1 ∧ z = 4) ∨
   (x = -4 ∧ y = 4 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l678_67893


namespace NUMINAMATH_CALUDE_radio_price_theorem_l678_67891

def original_price : ℕ := 5000
def final_amount : ℕ := 2468

def discount (price : ℕ) : ℚ :=
  let d1 := min price 2000 * (2 / 100)
  let d2 := min (max (price - 2000) 0) 2000 * (5 / 100)
  let d3 := max (price - 4000) 0 * (10 / 100)
  d1 + d2 + d3

def sales_tax (price : ℕ) : ℚ :=
  let t1 := min price 2500 * (4 / 100)
  let t2 := min (max (price - 2500) 0) 2000 * (7 / 100)
  let t3 := max (price - 4500) 0 * (9 / 100)
  t1 + t2 + t3

theorem radio_price_theorem (reduced_price : ℕ) :
  reduced_price - discount reduced_price + sales_tax original_price = final_amount :=
by sorry

end NUMINAMATH_CALUDE_radio_price_theorem_l678_67891


namespace NUMINAMATH_CALUDE_six_eight_ten_pythagorean_l678_67806

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- Theorem: (6, 8, 10) is a Pythagorean triple -/
theorem six_eight_ten_pythagorean : is_pythagorean_triple 6 8 10 := by
  sorry

end NUMINAMATH_CALUDE_six_eight_ten_pythagorean_l678_67806


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l678_67855

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_decreasing : ∀ x y, 0 ≤ x → x < y → f y < f x)

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (2*x - 1) > f (1/3)}

-- Theorem statement
theorem solution_set_is_open_interval :
  solution_set f = Set.Ioo (1/3) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l678_67855


namespace NUMINAMATH_CALUDE_f_has_three_distinct_roots_l678_67841

/-- The polynomial function whose roots we're counting -/
def f (x : ℝ) : ℝ := (x - 8) * (x^2 + 4*x + 3)

/-- The theorem stating that f has exactly 3 distinct real roots -/
theorem f_has_three_distinct_roots : 
  ∃ (r₁ r₂ r₃ : ℝ), (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) ∧ 
  (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃) ∧
  (∀ x : ℝ, f x = 0 → x = r₁ ∨ x = r₂ ∨ x = r₃) :=
sorry

end NUMINAMATH_CALUDE_f_has_three_distinct_roots_l678_67841


namespace NUMINAMATH_CALUDE_line_equation_proof_l678_67897

/-- Given a line that passes through the point (-2, 5) with a slope of -3/4,
    prove that its equation is 3x + 4y - 14 = 0 -/
theorem line_equation_proof (x y : ℝ) : 
  (y - 5 = -(3/4) * (x + 2)) ↔ (3*x + 4*y - 14 = 0) := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l678_67897


namespace NUMINAMATH_CALUDE_sum_of_variables_l678_67825

/-- Given a system of equations, prove that 2x + 2y + 2z = 8 -/
theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 20 - 4*x)
  (eq2 : x + z = -10 - 4*y)
  (eq3 : x + y = 14 - 4*z) :
  2*x + 2*y + 2*z = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_variables_l678_67825


namespace NUMINAMATH_CALUDE_brand_z_fraction_fraction_to_percentage_l678_67817

/-- Represents the state of the fuel tank -/
structure TankState where
  z : ℚ  -- Amount of brand Z gasoline
  y : ℚ  -- Amount of brand Y gasoline

/-- Fills the tank with brand Z gasoline when empty -/
def initial_fill : TankState :=
  { z := 1, y := 0 }

/-- Fills the tank with brand Y when 1/4 empty -/
def first_refill (s : TankState) : TankState :=
  { z := 3/4 * s.z, y := 1/4 + 3/4 * s.y }

/-- Fills the tank with brand Z when half empty -/
def second_refill (s : TankState) : TankState :=
  { z := 1/2 + 1/2 * s.z, y := 1/2 * s.y }

/-- Fills the tank with brand Y when half empty -/
def third_refill (s : TankState) : TankState :=
  { z := 1/2 * s.z, y := 1/2 + 1/2 * s.y }

/-- The final state of the tank after all refills -/
def final_state : TankState :=
  third_refill (second_refill (first_refill initial_fill))

/-- Theorem stating that the fraction of brand Z gasoline in the final state is 7/16 -/
theorem brand_z_fraction :
  final_state.z / (final_state.z + final_state.y) = 7/16 := by
  sorry

/-- Theorem stating that 7/16 is equivalent to 43.75% -/
theorem fraction_to_percentage :
  (7/16 : ℚ) = 43.75/100 := by
  sorry

end NUMINAMATH_CALUDE_brand_z_fraction_fraction_to_percentage_l678_67817


namespace NUMINAMATH_CALUDE_outdoor_section_length_l678_67875

/-- Given a rectangular outdoor section with area 35 square feet and width 7 feet, 
    the length of the section is 5 feet. -/
theorem outdoor_section_length : 
  ∀ (area width length : ℝ), 
    area = 35 → 
    width = 7 → 
    area = width * length → 
    length = 5 := by
sorry

end NUMINAMATH_CALUDE_outdoor_section_length_l678_67875


namespace NUMINAMATH_CALUDE_boundary_slopes_sum_l678_67857

/-- Parabola P with equation y = x^2 + 4x + 4 -/
def P : ℝ → ℝ := λ x => x^2 + 4*x + 4

/-- Point Q -/
def Q : ℝ × ℝ := (10, 16)

/-- Function to determine if a line with slope m through Q intersects P -/
def intersects (m : ℝ) : Prop :=
  ∃ x : ℝ, P x = Q.2 + m * (x - Q.1)

/-- The lower boundary slope -/
noncomputable def r : ℝ := -24 - 16 * Real.sqrt 2

/-- The upper boundary slope -/
noncomputable def s : ℝ := -24 + 16 * Real.sqrt 2

/-- Theorem stating that r + s = -48 -/
theorem boundary_slopes_sum : r + s = -48 := by sorry

end NUMINAMATH_CALUDE_boundary_slopes_sum_l678_67857


namespace NUMINAMATH_CALUDE_violet_hiking_time_l678_67852

/-- Proves that Violet and her dog can spend 4 hours hiking given the conditions --/
theorem violet_hiking_time :
  let violet_water_per_hour : ℚ := 800 / 1000  -- Convert ml to L
  let dog_water_per_hour : ℚ := 400 / 1000     -- Convert ml to L
  let total_water_capacity : ℚ := 4.8          -- In L
  
  (total_water_capacity / (violet_water_per_hour + dog_water_per_hour) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_violet_hiking_time_l678_67852


namespace NUMINAMATH_CALUDE_vincent_sticker_packs_l678_67802

/-- The number of packs Vincent bought yesterday -/
def yesterday_packs : ℕ := sorry

/-- The number of packs Vincent bought today -/
def today_packs : ℕ := yesterday_packs + 10

/-- The total number of packs Vincent has -/
def total_packs : ℕ := 40

theorem vincent_sticker_packs : yesterday_packs = 15 := by
  sorry

end NUMINAMATH_CALUDE_vincent_sticker_packs_l678_67802


namespace NUMINAMATH_CALUDE_second_bottle_capacity_l678_67800

theorem second_bottle_capacity
  (total_milk : ℝ)
  (first_bottle_capacity : ℝ)
  (second_bottle_milk : ℝ)
  (h1 : total_milk = 8)
  (h2 : first_bottle_capacity = 4)
  (h3 : second_bottle_milk = 16 / 3)
  (h4 : ∃ (f : ℝ), f * first_bottle_capacity + second_bottle_milk = total_milk ∧
                   f * first_bottle_capacity ≤ first_bottle_capacity ∧
                   second_bottle_milk ≤ f * (total_milk - first_bottle_capacity * f)) :
  total_milk - first_bottle_capacity * (total_milk - second_bottle_milk) / first_bottle_capacity = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_second_bottle_capacity_l678_67800


namespace NUMINAMATH_CALUDE_box_height_is_nine_l678_67872

/-- A rectangular box with dimensions 6 × 6 × h -/
structure Box (h : ℝ) where
  length : ℝ := 6
  width : ℝ := 6
  height : ℝ := h

/-- A sphere with a given radius -/
structure Sphere (r : ℝ) where
  radius : ℝ := r

/-- Predicate to check if a sphere is tangent to three sides of a box -/
def tangent_to_three_sides (s : Sphere r) (b : Box h) : Prop :=
  sorry

/-- Predicate to check if two spheres are tangent -/
def spheres_tangent (s1 : Sphere r1) (s2 : Sphere r2) : Prop :=
  sorry

/-- The main theorem -/
theorem box_height_is_nine :
  ∀ (h : ℝ) (b : Box h) (large_sphere : Sphere 3) (small_spheres : Fin 8 → Sphere 1.5),
    (∀ i, tangent_to_three_sides (small_spheres i) b) →
    (∀ i, spheres_tangent large_sphere (small_spheres i)) →
    h = 9 :=
by sorry

end NUMINAMATH_CALUDE_box_height_is_nine_l678_67872


namespace NUMINAMATH_CALUDE_smallest_alpha_inequality_half_satisfies_inequality_smallest_alpha_is_half_l678_67805

theorem smallest_alpha_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∀ α : ℝ, α > 0 → α < 1/2 →
    ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + y) / 2 < α * Real.sqrt (x * y) + (1 - α) * Real.sqrt ((x^2 + y^2) / 2) :=
by sorry

theorem half_satisfies_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / 2 ≥ (1/2) * Real.sqrt (x * y) + (1/2) * Real.sqrt ((x^2 + y^2) / 2) :=
by sorry

theorem smallest_alpha_is_half :
  ∀ α : ℝ, α > 0 →
    (∀ x y : ℝ, x > 0 → y > 0 →
      (x + y) / 2 ≥ α * Real.sqrt (x * y) + (1 - α) * Real.sqrt ((x^2 + y^2) / 2)) →
    α ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_alpha_inequality_half_satisfies_inequality_smallest_alpha_is_half_l678_67805


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l678_67842

/-- A regular polygon with perimeter 160 and side length 10 has 16 sides -/
theorem regular_polygon_sides (p : ℕ) (perimeter side_length : ℝ) 
  (h_perimeter : perimeter = 160)
  (h_side_length : side_length = 10)
  (h_regular : p * side_length = perimeter) : 
  p = 16 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l678_67842


namespace NUMINAMATH_CALUDE_roses_per_girl_l678_67862

theorem roses_per_girl (total_students : Nat) (total_plants : Nat) (total_birches : Nat) 
  (h1 : total_students = 24)
  (h2 : total_plants = 24)
  (h3 : total_birches = 6)
  (h4 : total_birches * 3 ≤ total_students) :
  ∃ (roses_per_girl : Nat), 
    roses_per_girl * (total_students - total_birches * 3) = total_plants - total_birches ∧ 
    roses_per_girl = 3 := by
  sorry

end NUMINAMATH_CALUDE_roses_per_girl_l678_67862


namespace NUMINAMATH_CALUDE_product_of_reals_l678_67894

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 8) (cube_sum_eq : a^3 + b^3 = 170) : a * b = 21.375 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reals_l678_67894


namespace NUMINAMATH_CALUDE_simplify_expression_l678_67819

theorem simplify_expression : (81 ^ (1/4) - Real.sqrt 12.25) ^ 2 = 1/4 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l678_67819


namespace NUMINAMATH_CALUDE_sheilas_weekly_earnings_l678_67899

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Sheila's work hours for each day -/
def workHours (d : Day) : ℕ :=
  match d with
  | Day.Monday => 8
  | Day.Tuesday => 6
  | Day.Wednesday => 8
  | Day.Thursday => 6
  | Day.Friday => 8
  | Day.Saturday => 0
  | Day.Sunday => 0

/-- Sheila's hourly wage -/
def hourlyWage : ℚ := 14

/-- Calculates daily earnings -/
def dailyEarnings (d : Day) : ℚ :=
  hourlyWage * (workHours d)

/-- Calculates weekly earnings -/
def weeklyEarnings : ℚ :=
  (dailyEarnings Day.Monday) +
  (dailyEarnings Day.Tuesday) +
  (dailyEarnings Day.Wednesday) +
  (dailyEarnings Day.Thursday) +
  (dailyEarnings Day.Friday) +
  (dailyEarnings Day.Saturday) +
  (dailyEarnings Day.Sunday)

/-- Theorem: Sheila's weekly earnings are $504 -/
theorem sheilas_weekly_earnings : weeklyEarnings = 504 := by
  sorry

end NUMINAMATH_CALUDE_sheilas_weekly_earnings_l678_67899


namespace NUMINAMATH_CALUDE_money_left_after_distributions_l678_67835

/-- Calculates the amount of money left after distributions --/
theorem money_left_after_distributions (income : ℝ) : 
  income = 1000 → 
  income * (1 - 0.2 - 0.2) * (1 - 0.1) = 540 := by
  sorry

#check money_left_after_distributions

end NUMINAMATH_CALUDE_money_left_after_distributions_l678_67835


namespace NUMINAMATH_CALUDE_negation_of_forall_gt_one_negation_of_inequality_l678_67854

theorem negation_of_forall_gt_one (P : ℝ → Prop) :
  (¬ ∀ x > 1, P x) ↔ (∃ x > 1, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_gt_one_negation_of_inequality_l678_67854


namespace NUMINAMATH_CALUDE_geese_ducks_difference_l678_67804

def geese : ℝ := 58.0
def ducks : ℝ := 37.0

theorem geese_ducks_difference : geese - ducks = 21.0 := by
  sorry

end NUMINAMATH_CALUDE_geese_ducks_difference_l678_67804


namespace NUMINAMATH_CALUDE_shells_given_to_brother_l678_67865

def shells_per_day : ℕ := 10
def days_collecting : ℕ := 6
def shells_remaining : ℕ := 58

theorem shells_given_to_brother :
  shells_per_day * days_collecting - shells_remaining = 2 := by
  sorry

end NUMINAMATH_CALUDE_shells_given_to_brother_l678_67865


namespace NUMINAMATH_CALUDE_edward_escape_problem_l678_67803

/-- The problem of Edward escaping from prison and being hit by an arrow. -/
theorem edward_escape_problem (initial_distance : ℝ) (arrow_initial_velocity : ℝ) 
  (edward_acceleration : ℝ) (arrow_deceleration : ℝ) :
  initial_distance = 1875 →
  arrow_initial_velocity = 100 →
  edward_acceleration = 1 →
  arrow_deceleration = 1 →
  ∃ t : ℝ, t > 0 ∧ 
    (-1/2 * arrow_deceleration * t^2 + arrow_initial_velocity * t) = 
    (1/2 * edward_acceleration * t^2 + initial_distance) ∧
    (arrow_initial_velocity - arrow_deceleration * t) = 75 :=
by sorry

end NUMINAMATH_CALUDE_edward_escape_problem_l678_67803


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l678_67832

theorem tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ) 
  (lily_win_prob : ℚ) 
  (h1 : amy_win_prob = 3/8) 
  (h2 : lily_win_prob = 3/10) 
  (h3 : amy_win_prob + lily_win_prob ≤ 1) :
  1 - (amy_win_prob + lily_win_prob) = 13/40 :=
by sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l678_67832


namespace NUMINAMATH_CALUDE_percent_increase_l678_67850

theorem percent_increase (N M P : ℝ) (h : M = N * (1 + P / 100)) :
  (M - N) / N * 100 = P := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_l678_67850


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l678_67874

/-- The range of values for m where the line y = x + m intersects the ellipse x^2/4 + y^2/3 = 1 -/
theorem line_ellipse_intersection_range :
  let line (x m : ℝ) := x + m
  let ellipse (x y : ℝ) := x^2/4 + y^2/3 = 1
  let intersects (m : ℝ) := ∃ x, ellipse x (line x m)
  ∀ m, intersects m ↔ m ∈ Set.Icc (-Real.sqrt 7) (Real.sqrt 7) :=
by sorry


end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l678_67874


namespace NUMINAMATH_CALUDE_arithmetic_progression_fifth_term_l678_67846

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def isArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_fifth_term
  (a : ℕ → ℝ)
  (h_ap : isArithmeticProgression a)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 45) :
  a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_fifth_term_l678_67846


namespace NUMINAMATH_CALUDE_products_produced_is_twenty_l678_67812

/-- Calculates the number of products produced given fixed cost, marginal cost, and total cost. -/
def products_produced (fixed_cost marginal_cost total_cost : ℚ) : ℚ :=
  (total_cost - fixed_cost) / marginal_cost

/-- Theorem stating that the number of products produced is 20 given the specified costs. -/
theorem products_produced_is_twenty :
  products_produced 12000 200 16000 = 20 := by
  sorry

#eval products_produced 12000 200 16000

end NUMINAMATH_CALUDE_products_produced_is_twenty_l678_67812


namespace NUMINAMATH_CALUDE_gcd_lcm_392_count_l678_67849

theorem gcd_lcm_392_count : 
  ∃! n : ℕ, n > 0 ∧ 
  (∃ S : Finset ℕ, S.card = n ∧
    ∀ d ∈ S, d > 0 ∧
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧
      Nat.gcd a b * Nat.lcm a b = 392 ∧
      Nat.gcd a b = d) ∧
    (∀ a b : ℕ, a > 0 → b > 0 →
      Nat.gcd a b * Nat.lcm a b = 392 →
      Nat.gcd a b ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_gcd_lcm_392_count_l678_67849


namespace NUMINAMATH_CALUDE_student_rank_l678_67815

theorem student_rank (total : Nat) (left_rank : Nat) (right_rank : Nat) : 
  total = 20 → left_rank = 8 → right_rank = total - left_rank + 1 → right_rank = 13 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_l678_67815


namespace NUMINAMATH_CALUDE_f_negative_alpha_l678_67887

noncomputable def f (x : ℝ) : ℝ := Real.tan x + 1 / Real.tan x

theorem f_negative_alpha (α : ℝ) (h : f α = 5) : f (-α) = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_alpha_l678_67887


namespace NUMINAMATH_CALUDE_surface_area_of_problem_solid_l678_67801

/-- Represents a solid formed by unit cubes -/
structure CubeSolid where
  base_length : ℕ
  top_cube_position : ℕ
  total_cubes : ℕ

/-- Calculate the surface area of the cube solid -/
def surface_area (solid : CubeSolid) : ℕ :=
  -- Front and back
  2 * solid.base_length +
  -- Left and right sides
  (solid.base_length - 1) + (solid.top_cube_position + 3) +
  -- Top surface
  solid.base_length + 1

/-- The specific cube solid described in the problem -/
def problem_solid : CubeSolid :=
  { base_length := 7
  , top_cube_position := 2
  , total_cubes := 8 }

theorem surface_area_of_problem_solid :
  surface_area problem_solid = 34 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_problem_solid_l678_67801


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l678_67838

theorem absolute_value_inequality (x : ℝ) : 
  |x + 1| > 3 ↔ x ∈ Set.Iio (-4) ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l678_67838


namespace NUMINAMATH_CALUDE_brenda_bracelets_l678_67866

/-- Given the number of bracelets and total number of stones, 
    calculate the number of stones per bracelet -/
def stones_per_bracelet (num_bracelets : ℕ) (total_stones : ℕ) : ℕ :=
  total_stones / num_bracelets

/-- Theorem: Given 3 bracelets and 36 total stones, 
    there will be 12 stones per bracelet -/
theorem brenda_bracelets : stones_per_bracelet 3 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_brenda_bracelets_l678_67866


namespace NUMINAMATH_CALUDE_test_questions_count_l678_67879

theorem test_questions_count (total_questions : ℕ) 
  (correct_answers : ℕ) (final_score : ℚ) :
  correct_answers = 104 →
  final_score = 100 →
  (correct_answers : ℚ) + ((total_questions - correct_answers : ℕ) : ℚ) * (-1/4) = final_score →
  total_questions = 120 :=
by sorry

end NUMINAMATH_CALUDE_test_questions_count_l678_67879


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l678_67896

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Given a line with slope 4 passing through (-2, 5), prove m + b = 17 -/
theorem line_slope_intercept_sum (L : Line) 
  (slope_is_4 : L.m = 4)
  (passes_through : 5 = 4 * (-2) + L.b) : 
  L.m + L.b = 17 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l678_67896


namespace NUMINAMATH_CALUDE_quadratic_range_on_interval_l678_67814

/-- A quadratic function defined on a closed interval -/
def QuadraticFunction (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The range of a quadratic function on a closed interval -/
def QuadraticRange (a b c : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc (-1 : ℝ) 2, y = QuadraticFunction a b c x}

theorem quadratic_range_on_interval
  (a b c : ℝ) (h : a > 0) :
  QuadraticRange a b c =
    Set.Icc (min (a - b + c) (c - b^2 / (4 * a))) (4 * a + 2 * b + c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_on_interval_l678_67814


namespace NUMINAMATH_CALUDE_farm_animals_l678_67834

/-- Given a farm with cows and horses, prove the number of horses -/
theorem farm_animals (cow_count : ℕ) (horse_count : ℕ) : 
  (cow_count : ℚ) / horse_count = 7 / 2 → cow_count = 21 → horse_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l678_67834


namespace NUMINAMATH_CALUDE_circle_triangle_area_l678_67828

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  -- We'll assume a simple representation for this problem
  -- More complex representations might be needed for general use
  point : Point
  direction : Point

def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.x - c2.center.x) ^ 2 + (c1.center.y - c2.center.y) ^ 2 = (c1.radius + c2.radius) ^ 2

def internally_tangent_to_line (c : Circle) (l : Line) : Prop :=
  -- This is a simplification; in reality, we'd need more complex calculations
  c.center.y = l.point.y + c.radius

def externally_tangent_to_line (c : Circle) (l : Line) : Prop :=
  -- This is a simplification; in reality, we'd need more complex calculations
  c.center.y = l.point.y - c.radius

def between_points_on_line (p q r : Point) (l : Line) : Prop :=
  -- This is a simplification; in reality, we'd need more complex calculations
  p.x < q.x ∧ q.x < r.x

def triangle_area (p q r : Point) : ℝ :=
  0.5 * |p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y)|

theorem circle_triangle_area :
  ∀ (P Q R : Circle) (l : Line) (P' Q' R' : Point),
    P.radius = 1 →
    Q.radius = 3 →
    R.radius = 5 →
    internally_tangent_to_line P l →
    externally_tangent_to_line Q l →
    externally_tangent_to_line R l →
    between_points_on_line P' Q' R' l →
    externally_tangent P Q →
    externally_tangent Q R →
    triangle_area P.center Q.center R.center = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_area_l678_67828


namespace NUMINAMATH_CALUDE_edward_pipe_usage_l678_67864

/- Define the problem parameters -/
def total_washers : ℕ := 20
def remaining_washers : ℕ := 4
def washers_per_bolt : ℕ := 2
def feet_per_bolt : ℕ := 5

/- Define the function to calculate feet of pipe used -/
def feet_of_pipe_used (total_washers remaining_washers washers_per_bolt feet_per_bolt : ℕ) : ℕ :=
  let washers_used := total_washers - remaining_washers
  let bolts_used := washers_used / washers_per_bolt
  bolts_used * feet_per_bolt

/- Theorem statement -/
theorem edward_pipe_usage :
  feet_of_pipe_used total_washers remaining_washers washers_per_bolt feet_per_bolt = 40 := by
  sorry

end NUMINAMATH_CALUDE_edward_pipe_usage_l678_67864


namespace NUMINAMATH_CALUDE_infinite_points_in_region_l678_67858

theorem infinite_points_in_region : 
  ∃ (S : Set (ℚ × ℚ)), 
    (∀ (p : ℚ × ℚ), p ∈ S ↔ 
      (0 < p.1 ∧ 0 < p.2) ∧ 
      (p.1^2 + p.2^2 ≤ 16) ∧ 
      (p.1 ≤ 3 ∧ p.2 ≤ 3)) ∧ 
    Set.Infinite S :=
by sorry

end NUMINAMATH_CALUDE_infinite_points_in_region_l678_67858
