import Mathlib

namespace divisibility_property_l569_56991

theorem divisibility_property (a b : ℕ+) : ∃ n : ℕ+, (a : ℕ) ∣ (b : ℕ)^(n : ℕ) - (n : ℕ) := by
  sorry

end divisibility_property_l569_56991


namespace larger_solution_of_quadratic_l569_56961

theorem larger_solution_of_quadratic (x : ℝ) :
  x^2 - 9*x - 22 = 0 →
  (∃ y : ℝ, y ≠ x ∧ y^2 - 9*y - 22 = 0) →
  (x = 11 ∨ x < 11) :=
sorry

end larger_solution_of_quadratic_l569_56961


namespace largest_term_binomial_expansion_largest_term_specific_case_l569_56987

theorem largest_term_binomial_expansion (n : ℕ) (x : ℝ) (h : x > 0) :
  let A : ℕ → ℝ := λ k => (n.choose k) * x^k
  ∃ k : ℕ, k ≤ n ∧ ∀ j : ℕ, j ≤ n → A k ≥ A j :=
by
  sorry

theorem largest_term_specific_case :
  let n : ℕ := 500
  let x : ℝ := 0.3
  let A : ℕ → ℝ := λ k => (n.choose k) * x^k
  ∃ k : ℕ, k = 125 ∧ ∀ j : ℕ, j ≤ n → A k ≥ A j :=
by
  sorry

end largest_term_binomial_expansion_largest_term_specific_case_l569_56987


namespace congruence_solution_l569_56914

theorem congruence_solution : 
  {x : ℤ | 20 ≤ x ∧ x ≤ 50 ∧ (6 * x + 5) % 10 = (-19) % 10} = 
  {21, 26, 31, 36, 41, 46} := by
  sorry

end congruence_solution_l569_56914


namespace unique_x_for_volume_l569_56975

/-- A function representing the volume of the rectangular prism -/
def volume (x : ℕ) : ℕ := (x + 3) * (x - 3) * (x^2 + 9)

/-- The theorem stating that there is exactly one positive integer x satisfying the conditions -/
theorem unique_x_for_volume :
  ∃! x : ℕ, x > 3 ∧ volume x < 500 :=
sorry

end unique_x_for_volume_l569_56975


namespace sum_of_special_integers_l569_56906

theorem sum_of_special_integers (a b c d e : ℤ) : 
  (a + 1 = b) ∧ (c + 1 = d) ∧ (d + 1 = e) ∧ (a * b = 272) ∧ (c * d * e = 336) →
  a + b + c + d + e = 54 := by
sorry

end sum_of_special_integers_l569_56906


namespace cistern_wet_surface_area_l569_56905

/-- Calculates the total wet surface area of a rectangular cistern -/
def cisternWetSurfaceArea (length width depth : Real) : Real :=
  let bottomArea := length * width
  let longerSidesArea := 2 * length * depth
  let shorterSidesArea := 2 * width * depth
  bottomArea + longerSidesArea + shorterSidesArea

/-- Theorem: The wet surface area of a cistern with given dimensions is 68.6 square meters -/
theorem cistern_wet_surface_area :
  cisternWetSurfaceArea 7 5 1.40 = 68.6 := by
  sorry

#eval cisternWetSurfaceArea 7 5 1.40

end cistern_wet_surface_area_l569_56905


namespace binary_110_equals_6_l569_56993

def binary_to_decimal (b₂ b₁ b₀ : Nat) : Nat :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_110_equals_6 : binary_to_decimal 1 1 0 = 6 := by
  sorry

end binary_110_equals_6_l569_56993


namespace reading_reward_pie_chart_l569_56999

theorem reading_reward_pie_chart (agree disagree neutral : ℕ) 
  (h_ratio : (agree : ℚ) / (disagree : ℚ) = 7 / 2 ∧ (agree : ℚ) / (neutral : ℚ) = 7 / 1) :
  (360 : ℚ) * (agree : ℚ) / ((agree : ℚ) + (disagree : ℚ) + (neutral : ℚ)) = 252 := by
  sorry

end reading_reward_pie_chart_l569_56999


namespace cat_grooming_time_l569_56964

/-- Calculates the total grooming time for a cat given specific grooming tasks and the cat's characteristics. -/
theorem cat_grooming_time :
  let clip_time_per_claw : ℕ := 10
  let clean_time_per_ear : ℕ := 90
  let shampoo_time_minutes : ℕ := 5
  let claws_per_foot : ℕ := 4
  let feet : ℕ := 4
  let ears : ℕ := 2
  clip_time_per_claw * claws_per_foot * feet + 
  clean_time_per_ear * ears + 
  shampoo_time_minutes * 60 = 640 := by
sorry


end cat_grooming_time_l569_56964


namespace parallel_planes_line_relations_l569_56982

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the contained relation for lines in planes
variable (contained_in : Line → Plane → Prop)

-- Define the intersect relation for lines
variable (intersect : Line → Line → Prop)

-- Define the coplanar relation for lines
variable (coplanar : Line → Line → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_line_relations 
  (α β : Plane) (a b : Line)
  (h1 : parallel_planes α β)
  (h2 : contained_in a α)
  (h3 : contained_in b β) :
  (¬ intersect a b) ∧ (coplanar a b ∨ skew a b) :=
sorry

end parallel_planes_line_relations_l569_56982


namespace unique_divisor_with_remainder_sum_l569_56996

theorem unique_divisor_with_remainder_sum (a b c : ℕ) : ∃! n : ℕ,
  n > 3 ∧
  ∃ x y z r s t : ℕ,
    63 = n * x + r ∧
    91 = n * y + s ∧
    130 = n * z + t ∧
    r + s + t = 26 := by
  sorry

end unique_divisor_with_remainder_sum_l569_56996


namespace train_length_calculation_l569_56965

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 15 → ∃ (length : ℝ), abs (length - 500) < 1 :=
by
  sorry

end train_length_calculation_l569_56965


namespace perpendicular_tangents_ratio_l569_56988

theorem perpendicular_tangents_ratio (a b : ℝ) : 
  (∃ (x y : ℝ), a*x + b*y - 5 = 0 ∧ y = x^3) →  -- Line and curve equations
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ a*x + b*y - 5 = 0 ∧ y = x^3) →  -- Point P(1, 1) satisfies both equations
  (∀ (m₁ m₂ : ℝ), (m₁ * m₂ = -1) → 
    (m₁ = -a/b ∧ m₂ = 3 * 1^2)) →  -- Perpendicular tangent lines condition
  a/b = 1/3 :=
by sorry

end perpendicular_tangents_ratio_l569_56988


namespace max_objective_value_l569_56926

/-- The system of inequalities and objective function --/
def LinearProgram (x y : ℝ) : Prop :=
  x + 7 * y ≤ 32 ∧
  2 * x + 5 * y ≤ 42 ∧
  3 * x + 4 * y ≤ 62 ∧
  2 * x + y = 34 ∧
  x ≥ 0 ∧ y ≥ 0

/-- The objective function --/
def ObjectiveFunction (x y : ℝ) : ℝ :=
  3 * x + 8 * y

/-- The theorem stating the maximum value of the objective function --/
theorem max_objective_value :
  ∃ (x y : ℝ), LinearProgram x y ∧
  ∀ (x' y' : ℝ), LinearProgram x' y' →
  ObjectiveFunction x y ≥ ObjectiveFunction x' y' ∧
  ObjectiveFunction x y = 64 :=
sorry

end max_objective_value_l569_56926


namespace regular_polygon_area_l569_56922

theorem regular_polygon_area (n : ℕ) (R : ℝ) (h : n > 0) :
  (n * R^2 / 2) * (Real.sin (2 * Real.pi / n) + Real.cos (Real.pi / n)) = 4 * R^2 →
  n = 24 := by
sorry

end regular_polygon_area_l569_56922


namespace paper_fold_crease_length_l569_56932

/-- Given a rectangular paper of width 8 inches, when folded so that the bottom right corner 
    touches the left edge dividing it in a 1:2 ratio, the length of the crease L is equal to 
    16/3 csc θ, where θ is the angle between the crease and the bottom edge. -/
theorem paper_fold_crease_length (width : ℝ) (θ : ℝ) (L : ℝ) :
  width = 8 →
  0 < θ → θ < π / 2 →
  L = (16 / 3) * (1 / Real.sin θ) := by
  sorry

end paper_fold_crease_length_l569_56932


namespace parallel_vectors_x_value_l569_56989

/-- Two planar vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given planar vectors a and b, if they are parallel, then x = -3/2 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (-2, 3)
  are_parallel a b → x = -3/2 := by
sorry

end parallel_vectors_x_value_l569_56989


namespace jump_height_ratio_l569_56979

/-- The jump heights of four people and their ratios -/
theorem jump_height_ratio :
  let mark_height := 6
  let lisa_height := 2 * mark_height
  let jacob_height := 2 * lisa_height
  let james_height := 16
  (james_height : ℚ) / jacob_height = 2 / 3 := by
  sorry

end jump_height_ratio_l569_56979


namespace austin_started_with_80_l569_56960

/-- The amount of money Austin started with, given the conditions of the problem. -/
def austin_starting_amount : ℚ :=
  let num_robots : ℕ := 7
  let robot_cost : ℚ := 875 / 100
  let total_tax : ℚ := 722 / 100
  let change : ℚ := 1153 / 100
  num_robots * robot_cost + total_tax + change

/-- Theorem stating that Austin started with $80. -/
theorem austin_started_with_80 : austin_starting_amount = 80 := by
  sorry

end austin_started_with_80_l569_56960


namespace largest_integer_in_interval_l569_56968

theorem largest_integer_in_interval : 
  ∃ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 3/5 ∧ 
  ∀ (z : ℤ), ((1/4 : ℚ) < (z : ℚ)/7 ∧ (z : ℚ)/7 < 3/5) → z ≤ y :=
by
  -- The proof goes here
  sorry

end largest_integer_in_interval_l569_56968


namespace range_of_m_l569_56929

theorem range_of_m (x m : ℝ) : 
  (∀ x, (x ≥ -2 ∧ x ≤ 10) → (x + m - 1) * (x - m - 1) ≤ 0) →
  m > 0 →
  m ≥ 9 :=
by sorry

end range_of_m_l569_56929


namespace x_value_l569_56970

theorem x_value (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 := by
  sorry

end x_value_l569_56970


namespace pea_patch_fraction_l569_56938

theorem pea_patch_fraction (radish_patch : ℝ) (pea_patch : ℝ) (fraction : ℝ) : 
  radish_patch = 15 →
  pea_patch = 2 * radish_patch →
  fraction * pea_patch = 5 →
  fraction = 1 / 6 := by
  sorry

end pea_patch_fraction_l569_56938


namespace twenty_cows_twenty_days_l569_56956

/-- The number of bags of husk eaten by a group of cows over a period of days -/
def bags_eaten (num_cows : ℕ) (num_days : ℕ) : ℚ :=
  (num_cows : ℚ) * (num_days : ℚ) * (1 / 20 : ℚ)

/-- Theorem stating that 20 cows eat 20 bags of husk in 20 days -/
theorem twenty_cows_twenty_days : bags_eaten 20 20 = 20 := by
  sorry

end twenty_cows_twenty_days_l569_56956


namespace instrument_probability_l569_56913

theorem instrument_probability (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = 3/5 →
  two_or_more = 96 →
  (((at_least_one * total) - two_or_more) / total : ℚ) = 12/25 := by
sorry

end instrument_probability_l569_56913


namespace train_length_l569_56984

/-- Proves the length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (platform_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  time_to_pass = 39.2 →
  platform_length = 130 →
  train_speed * time_to_pass - platform_length = 360 := by
sorry

end train_length_l569_56984


namespace prob_head_fair_coin_l569_56949

/-- A fair coin with two sides. -/
structure FairCoin where
  sides : Fin 2
  prob_head : ℝ
  prob_tail : ℝ
  sum_to_one : prob_head + prob_tail = 1
  equal_prob : prob_head = prob_tail

/-- The probability of getting a head in a fair coin toss is 1/2. -/
theorem prob_head_fair_coin (c : FairCoin) : c.prob_head = 1/2 := by
  sorry

#check prob_head_fair_coin

end prob_head_fair_coin_l569_56949


namespace quadratic_radicals_combination_l569_56969

theorem quadratic_radicals_combination (a : ℝ) : 
  (∃ k : ℝ, k * (1 + a) = 4 - 2*a ∧ k > 0) → a = 1 := by
  sorry

end quadratic_radicals_combination_l569_56969


namespace larger_cuboid_length_l569_56903

/-- Proves that the length of a larger cuboid is 12 meters, given its width, height, and the number and dimensions of smaller cuboids it can be divided into. -/
theorem larger_cuboid_length (width height : ℝ) (num_small_cuboids : ℕ) 
  (small_length small_width small_height : ℝ) : 
  width = 14 →
  height = 10 →
  num_small_cuboids = 56 →
  small_length = 5 →
  small_width = 3 →
  small_height = 2 →
  (width * height * (num_small_cuboids * small_length * small_width * small_height) / (width * height)) = 12 :=
by sorry

end larger_cuboid_length_l569_56903


namespace range_of_a_l569_56967

-- Define the polynomials p and q
def p (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
def q (x a : ℝ) : ℝ := x^2 - (2 * a + 1) * x + a^2 + a

-- Define the condition for p
def p_condition (x : ℝ) : Prop := p x ≤ 0

-- Define the condition for q
def q_condition (x a : ℝ) : Prop := q x a ≤ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p_condition x → q_condition x a) ∧
  (∃ x, q_condition x a ∧ ¬p_condition x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l569_56967


namespace units_digit_17_power_2024_l569_56959

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sequence of units digits for powers of a number -/
def unitsDigitSequence (base : ℕ) : ℕ → ℕ
  | 0 => unitsDigit base
  | n + 1 => unitsDigit (base * unitsDigitSequence base n)

theorem units_digit_17_power_2024 :
  unitsDigit (17^2024) = 1 :=
sorry

end units_digit_17_power_2024_l569_56959


namespace rectangle_area_sum_l569_56986

theorem rectangle_area_sum : 
  let rect1 := 7 * 8
  let rect2 := 5 * 3
  let rect3 := 2 * 8
  let rect4 := 2 * 7
  let rect5 := 4 * 4
  rect1 + rect2 + rect3 + rect4 + rect5 = 117 := by
sorry

end rectangle_area_sum_l569_56986


namespace polynomial_simplification_l569_56923

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 6*x^2 + 9 + 11*x - 12*x^2 - 15 + 17*x + 18*x^2 - 2*x^3 = -2*x^3 + 23*x - 3 := by
  sorry

end polynomial_simplification_l569_56923


namespace x_13_plus_inv_x_13_l569_56985

theorem x_13_plus_inv_x_13 (x : ℝ) (hx : x ≠ 0) :
  let y := x + 1/x
  x^13 + 1/x^13 = y^13 - 13*y^11 + 65*y^9 - 156*y^7 + 182*y^5 - 91*y^3 + 13*y :=
by
  sorry

end x_13_plus_inv_x_13_l569_56985


namespace parabola_properties_l569_56974

/-- Parabola passing through given points with specific properties -/
theorem parabola_properties (a b : ℝ) (m : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + 1) →
  (-2 = a * 1^2 + b * 1 + 1) →
  (13 = a * (-2)^2 + b * (-2) + 1) →
  (∃ y₁ y₂, y₁ = a * 5^2 + b * 5 + 1 ∧ 
            y₂ = a * m^2 + b * m + 1 ∧ 
            y₂ = 12 - y₁) →
  (a = 1 ∧ b = -4 ∧ m = -1) := by
sorry

end parabola_properties_l569_56974


namespace problem_statement_l569_56981

theorem problem_statement (x y : ℝ) (h1 : x - y > -x) (h2 : x + y > y) : x > 0 ∧ y < 2*x := by
  sorry

end problem_statement_l569_56981


namespace supermarket_prices_theorem_l569_56910

/-- Represents the prices and discounts at supermarkets -/
structure SupermarketPrices where
  english_machine : ℕ
  backpack : ℕ
  discount_a : ℚ
  voucher_b : ℕ
  voucher_threshold : ℕ

/-- Theorem stating the correct prices and most cost-effective supermarket -/
theorem supermarket_prices_theorem (prices : SupermarketPrices)
    (h1 : prices.english_machine + prices.backpack = 452)
    (h2 : prices.english_machine = 4 * prices.backpack - 8)
    (h3 : prices.discount_a = 75 / 100)
    (h4 : prices.voucher_b = 30)
    (h5 : prices.voucher_threshold = 100)
    (h6 : 400 ≥ prices.english_machine + prices.backpack) :
    prices.english_machine = 360 ∧ 
    prices.backpack = 92 ∧ 
    (prices.english_machine + prices.backpack) * prices.discount_a < 
      prices.english_machine + prices.backpack - prices.voucher_b := by
  sorry


end supermarket_prices_theorem_l569_56910


namespace theorem_1_theorem_2_l569_56995

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side length opposite to angle A
  b : ℝ  -- side length opposite to angle B
  c : ℝ  -- side length opposite to angle C
  A : ℝ  -- angle A in radians
  B : ℝ  -- angle B in radians
  C : ℝ  -- angle C in radians

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.B = 4/5

-- Theorem 1
theorem theorem_1 (t : Triangle) (h : triangle_conditions t) (h_A : t.A = Real.pi/6) :
  t.a = 5/3 := by sorry

-- Theorem 2
theorem theorem_2 (t : Triangle) (h : triangle_conditions t) 
  (h_area : (1/2) * t.a * t.c * Real.sin t.B = 3) :
  t.a = Real.sqrt 10 ∧ t.c = Real.sqrt 10 := by sorry

end theorem_1_theorem_2_l569_56995


namespace perpendicular_line_equation_distance_line_equations_l569_56963

-- Define the lines
def l₁ (x y : ℝ) : Prop := y = 2 * x
def l₂ (x y : ℝ) : Prop := x + y = 6
def l₀ (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the intersection point P
def P : ℝ × ℝ := (2, 4)

-- Theorem for part (1)
theorem perpendicular_line_equation :
  ∀ x y : ℝ, (x - P.1) = -2 * (y - P.2) ↔ 2 * x + y - 8 = 0 :=
sorry

-- Theorem for part (2)
theorem distance_line_equations :
  ∀ x y : ℝ, 
    (x = P.1 ∨ 3 * x - 4 * y + 10 = 0) ↔
    (∃ k : ℝ, y - P.2 = k * (x - P.1) ∧ 
      |k * P.1 - P.2| / Real.sqrt (k^2 + 1) = 2) ∨
    (x = P.1 ∧ |x| = 2) :=
sorry

end perpendicular_line_equation_distance_line_equations_l569_56963


namespace clown_balloon_count_l569_56930

/-- The number of balloons a clown has after a series of actions -/
def final_balloon_count (initial : ℕ) (additional : ℕ) (given_away : ℕ) : ℕ :=
  initial + additional - given_away

/-- Theorem stating that the clown has 149 balloons at the end -/
theorem clown_balloon_count :
  final_balloon_count 123 53 27 = 149 := by
  sorry

end clown_balloon_count_l569_56930


namespace symmetry_theorem_l569_56980

/-- The line about which the points are symmetrical -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Defines a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry between two points about a line -/
def symmetric_about_line (P Q : Point) : Prop :=
  -- The midpoint of PQ lies on the symmetry line
  symmetry_line ((P.x + Q.x) / 2) ((P.y + Q.y) / 2) ∧
  -- The slope of PQ is perpendicular to the slope of the symmetry line
  (Q.y - P.y) / (Q.x - P.x) = -1

/-- The theorem to be proved -/
theorem symmetry_theorem (a b : ℝ) :
  let P : Point := ⟨3, 4⟩
  let Q : Point := ⟨a, b⟩
  symmetric_about_line P Q → a = 5 ∧ b = 2 := by
  sorry

end symmetry_theorem_l569_56980


namespace intersection_dot_product_l569_56966

/-- Given a line and a parabola that intersect at points A and B, 
    and the focus of the parabola F, prove that the dot product 
    of vectors FA and FB is -11. -/
theorem intersection_dot_product 
  (A B : ℝ × ℝ) 
  (hA : A.2 = 2 * A.1 - 2 ∧ A.2^2 = 8 * A.1) 
  (hB : B.2 = 2 * B.1 - 2 ∧ B.2^2 = 8 * B.1) 
  (hAB_distinct : A ≠ B) : 
  let F : ℝ × ℝ := (2, 0)
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = -11 := by
  sorry

end intersection_dot_product_l569_56966


namespace cube_plus_reciprocal_cube_l569_56907

theorem cube_plus_reciprocal_cube (m : ℝ) (h : m + 1/m = 10) :
  m^3 + 1/m^3 + 6 = 976 := by sorry

end cube_plus_reciprocal_cube_l569_56907


namespace cube_volume_from_surface_area_l569_56958

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 216 →
  volume = (((surface_area / 6) ^ (1/2 : ℝ)) ^ 3) →
  volume = 216 := by
sorry

end cube_volume_from_surface_area_l569_56958


namespace recordedLineLengthApprox_l569_56918

/-- Represents the parameters of a record turntable --/
structure TurntableParams where
  revPerMinute : ℝ
  playTime : ℝ
  initialDiameter : ℝ
  finalDiameter : ℝ

/-- Calculates the length of the recorded line on a turntable --/
def recordedLineLength (params : TurntableParams) : ℝ :=
  sorry

/-- The main theorem stating the length of the recorded line --/
theorem recordedLineLengthApprox (params : TurntableParams) 
  (h1 : params.revPerMinute = 100)
  (h2 : params.playTime = 24.5)
  (h3 : params.initialDiameter = 29)
  (h4 : params.finalDiameter = 11.5) :
  abs (recordedLineLength params - 155862.265789099) < 1e-6 := by
  sorry

end recordedLineLengthApprox_l569_56918


namespace sunglasses_cap_probability_l569_56902

theorem sunglasses_cap_probability (total_sunglasses : ℕ) (total_caps : ℕ) 
  (prob_cap_given_sunglasses : ℚ) :
  total_sunglasses = 80 →
  total_caps = 60 →
  prob_cap_given_sunglasses = 3/8 →
  (prob_cap_given_sunglasses * total_sunglasses : ℚ) / total_caps = 1/2 := by
  sorry

end sunglasses_cap_probability_l569_56902


namespace expected_winnings_is_one_l569_56971

/-- Represents the possible outcomes of the dice roll -/
inductive Outcome
| Star
| Moon
| Sun

/-- The probability of each outcome -/
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Star => 1/4
  | Outcome.Moon => 1/2
  | Outcome.Sun => 1/4

/-- The winnings (or losses) associated with each outcome -/
def winnings (o : Outcome) : ℤ :=
  match o with
  | Outcome.Star => 2
  | Outcome.Moon => 4
  | Outcome.Sun => -6

/-- The expected winnings from rolling the dice once -/
def expected_winnings : ℚ :=
  (probability Outcome.Star * winnings Outcome.Star) +
  (probability Outcome.Moon * winnings Outcome.Moon) +
  (probability Outcome.Sun * winnings Outcome.Sun)

theorem expected_winnings_is_one : expected_winnings = 1 := by
  sorry

end expected_winnings_is_one_l569_56971


namespace experiment_sequences_l569_56972

def num_procedures : ℕ → ℕ
  | n => 4 * Nat.factorial (n - 3)

theorem experiment_sequences (n : ℕ) (h : n ≥ 3) : num_procedures n = 96 := by
  sorry

end experiment_sequences_l569_56972


namespace polynomial_factor_value_theorem_l569_56977

theorem polynomial_factor_value_theorem (h k : ℝ) : 
  (∀ x : ℝ, (x + 2) * (x - 1) * (x + 3) ∣ (3 * x^4 - 2 * h * x^2 + h * x + k)) →
  |3 * h - 2 * k| = 11 := by
sorry

end polynomial_factor_value_theorem_l569_56977


namespace coprime_20172019_l569_56990

theorem coprime_20172019 : 
  (Nat.gcd 20172019 20172017 = 1) ∧ 
  (Nat.gcd 20172019 20172018 = 1) ∧ 
  (Nat.gcd 20172019 20172020 = 1) ∧ 
  (Nat.gcd 20172019 20172021 = 1) := by
  sorry

end coprime_20172019_l569_56990


namespace flower_visitation_l569_56998

theorem flower_visitation (total_flowers : ℕ) (num_bees : ℕ) (flowers_per_bee : ℕ)
  (h_total : total_flowers = 88)
  (h_bees : num_bees = 3)
  (h_flowers_per_bee : flowers_per_bee = 54)
  : ∃ (sweet bitter : ℕ), 
    sweet + bitter ≤ total_flowers ∧ 
    num_bees * flowers_per_bee = 3 * sweet + 2 * (total_flowers - sweet - bitter) + bitter ∧
    bitter = sweet + 14 := by
  sorry

end flower_visitation_l569_56998


namespace interval_intersection_l569_56915

theorem interval_intersection : ∀ x : ℝ, 
  (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) := by sorry

end interval_intersection_l569_56915


namespace evaluate_expression_l569_56973

theorem evaluate_expression : (-2 : ℤ) ^ (3 ^ 2) + 2 ^ (3 ^ 2) = 0 := by
  sorry

end evaluate_expression_l569_56973


namespace tan_theta_value_l569_56916

theorem tan_theta_value (θ : Real) : 
  (Real.sin (π - θ) + Real.cos (θ - 2*π)) / (Real.sin θ + Real.cos (π + θ)) = 1/2 → 
  Real.tan θ = -3 := by
  sorry

end tan_theta_value_l569_56916


namespace xy_value_l569_56943

theorem xy_value (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) : x * y = 1 := by
  sorry

end xy_value_l569_56943


namespace congruence_solution_l569_56928

theorem congruence_solution (p q : Nat) (n : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Odd p ∧ Odd q ∧ n > 0 ∧
  (q^(n+2) : Nat) % (p^n) = (3^(n+2) : Nat) % (p^n) ∧
  (p^(n+2) : Nat) % (q^n) = (3^(n+2) : Nat) % (q^n) →
  p = 3 ∧ q = 3 := by
sorry

end congruence_solution_l569_56928


namespace forty_percent_of_number_l569_56997

theorem forty_percent_of_number (n : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 16 → 
  (40/100 : ℝ) * n = 192 := by
sorry

end forty_percent_of_number_l569_56997


namespace solution_set_m_zero_range_of_m_x_in_2_3_l569_56937

-- Define the inequality
def inequality (x m : ℝ) : Prop := x * abs (x - m) - 2 ≥ m

-- Part 1: Solution set when m = 0
theorem solution_set_m_zero :
  {x : ℝ | inequality x 0} = {x : ℝ | x ≥ Real.sqrt 2} := by sorry

-- Part 2: Range of m when x ∈ [2, 3]
theorem range_of_m_x_in_2_3 :
  {m : ℝ | ∀ x ∈ Set.Icc 2 3, inequality x m} = 
  {m : ℝ | m ≤ 2/3 ∨ m ≥ 6} := by sorry

end solution_set_m_zero_range_of_m_x_in_2_3_l569_56937


namespace quadratic_equation_form_l569_56934

/-- 
Given a quadratic equation ax^2 + bx + c = 0,
if a = 3 and c = 1, then the equation is equivalent to 3x^2 + 1 = 0.
-/
theorem quadratic_equation_form (a b c : ℝ) : 
  a = 3 → c = 1 → (∃ x, a * x^2 + b * x + c = 0) ↔ (∃ x, 3 * x^2 + 1 = 0) := by
  sorry

end quadratic_equation_form_l569_56934


namespace factorization_xy_squared_minus_x_l569_56957

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end factorization_xy_squared_minus_x_l569_56957


namespace line_canonical_form_l569_56947

/-- Given two planes that intersect to form a line, prove that the line can be represented in canonical form. -/
theorem line_canonical_form (x y z : ℝ) : 
  (2*x - y + 3*z = 1) ∧ (5*x + 4*y - z = 7) →
  ∃ (t : ℝ), x = -11*t ∧ y = 17*t + 2 ∧ z = 13*t + 1 :=
by sorry

end line_canonical_form_l569_56947


namespace garden_fence_length_l569_56939

theorem garden_fence_length (side_length : ℝ) (h : side_length = 28) : 
  4 * side_length = 112 := by
  sorry

end garden_fence_length_l569_56939


namespace quadratic_function_value_bound_l569_56931

theorem quadratic_function_value_bound (p q : ℝ) : 
  ¬(∀ x ∈ ({1, 2, 3} : Set ℝ), |x^2 + p*x + q| < (1/2 : ℝ)) := by
  sorry

end quadratic_function_value_bound_l569_56931


namespace baking_time_l569_56978

/-- 
Given:
- It takes 7 minutes to bake 1 pan of cookies
- The total time to bake 4 pans is 28 minutes

Prove that the time to bake 4 pans of cookies is 28 minutes.
-/
theorem baking_time (time_for_one_pan : ℕ) (total_time : ℕ) (num_pans : ℕ) :
  time_for_one_pan = 7 →
  total_time = 28 →
  num_pans = 4 →
  total_time = 28 := by
sorry

end baking_time_l569_56978


namespace point_on_curve_in_third_quadrant_l569_56900

theorem point_on_curve_in_third_quadrant :
  ∀ a : ℝ, a < 0 → 3 * a^2 + (2 * a)^2 = 28 → a = -2 := by
  sorry

end point_on_curve_in_third_quadrant_l569_56900


namespace circle_tangent_to_x_axis_l569_56909

theorem circle_tangent_to_x_axis (x y : ℝ) :
  let center : ℝ × ℝ := (-3, 4)
  let equation := (x + 3)^2 + (y - 4)^2 = 16
  let is_tangent_to_x_axis := ∃ (x₀ : ℝ), (x₀ + 3)^2 + 4^2 = 16 ∧ ∀ (y : ℝ), y ≠ 0 → (x₀ + 3)^2 + (y - 4)^2 > 16
  equation ∧ is_tangent_to_x_axis :=
by
  sorry


end circle_tangent_to_x_axis_l569_56909


namespace incorrect_permutations_of_error_l569_56917

def word : String := "error"

theorem incorrect_permutations_of_error (n : ℕ) :
  (n = word.length) →
  (n.choose 2 * 1 - 1 = 19) :=
by
  sorry

end incorrect_permutations_of_error_l569_56917


namespace karen_grooms_six_rottweilers_l569_56924

/-- Represents the time taken to groom different dog breeds and the total grooming time -/
structure GroomingInfo where
  rottweilerTime : ℕ
  borderCollieTime : ℕ
  chihuahuaTime : ℕ
  totalTime : ℕ
  borderCollieCount : ℕ
  chihuahuaCount : ℕ

/-- Calculates the number of Rottweilers groomed given the grooming information -/
def calculateRottweilers (info : GroomingInfo) : ℕ :=
  (info.totalTime - info.borderCollieTime * info.borderCollieCount - info.chihuahuaTime * info.chihuahuaCount) / info.rottweilerTime

/-- Theorem stating that Karen grooms 6 Rottweilers given the problem conditions -/
theorem karen_grooms_six_rottweilers (info : GroomingInfo)
  (h1 : info.rottweilerTime = 20)
  (h2 : info.borderCollieTime = 10)
  (h3 : info.chihuahuaTime = 45)
  (h4 : info.totalTime = 255)
  (h5 : info.borderCollieCount = 9)
  (h6 : info.chihuahuaCount = 1) :
  calculateRottweilers info = 6 := by
  sorry


end karen_grooms_six_rottweilers_l569_56924


namespace glass_bottles_in_second_scenario_l569_56994

/-- The weight of a glass bottle in grams -/
def glass_weight : ℕ := 200

/-- The weight of a plastic bottle in grams -/
def plastic_weight : ℕ := 50

/-- The number of glass bottles in the first scenario -/
def first_scenario_bottles : ℕ := 3

/-- The number of plastic bottles in the second scenario -/
def second_scenario_plastic : ℕ := 5

/-- The total weight in the first scenario in grams -/
def first_scenario_weight : ℕ := 600

/-- The total weight in the second scenario in grams -/
def second_scenario_weight : ℕ := 1050

/-- The weight difference between a glass and plastic bottle in grams -/
def weight_difference : ℕ := 150

theorem glass_bottles_in_second_scenario :
  ∃ x : ℕ, 
    first_scenario_bottles * glass_weight = first_scenario_weight ∧
    glass_weight = plastic_weight + weight_difference ∧
    x * glass_weight + second_scenario_plastic * plastic_weight = second_scenario_weight ∧
    x = 4 := by
  sorry

end glass_bottles_in_second_scenario_l569_56994


namespace no_polynomial_satisfies_condition_l569_56927

/-- A polynomial function of degree exactly 3 -/
def PolynomialDegree3 (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x + c

/-- The condition that f(x^2) = [f(x)]^2 = f(f(x)) -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x^2) = (f x)^2 ∧ f (x^2) = f (f x)

theorem no_polynomial_satisfies_condition :
  ¬∃ f : ℝ → ℝ, PolynomialDegree3 f ∧ SatisfiesCondition f :=
sorry

end no_polynomial_satisfies_condition_l569_56927


namespace range_of_a_l569_56992

def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) (h : A a ∪ B a = Set.univ) : a ≤ 2 := by
  sorry

end range_of_a_l569_56992


namespace distance_sum_bounded_l569_56936

/-- The sum of squared distances from a point on an ellipse to four fixed points is bounded -/
theorem distance_sum_bounded (x y : ℝ) :
  (x / 2)^2 + (y / 3)^2 = 1 →
  32 ≤ (x - 1)^2 + (y - Real.sqrt 3)^2 +
       (x + Real.sqrt 3)^2 + (y - 1)^2 +
       (x + 1)^2 + (y + Real.sqrt 3)^2 +
       (x - Real.sqrt 3)^2 + (y + 1)^2 ∧
  (x - 1)^2 + (y - Real.sqrt 3)^2 +
  (x + Real.sqrt 3)^2 + (y - 1)^2 +
  (x + 1)^2 + (y + Real.sqrt 3)^2 +
  (x - Real.sqrt 3)^2 + (y + 1)^2 ≤ 52 := by
  sorry


end distance_sum_bounded_l569_56936


namespace initial_ratio_is_four_to_one_l569_56942

-- Define the total initial volume of the mixture
def total_volume : ℚ := 60

-- Define the volume of water added
def added_water : ℚ := 60

-- Define the ratio of milk to water after adding water
def new_ratio : ℚ := 1 / 2

-- Theorem statement
theorem initial_ratio_is_four_to_one :
  ∀ (initial_milk initial_water : ℚ),
    initial_milk + initial_water = total_volume →
    initial_milk / (initial_water + added_water) = new_ratio →
    initial_milk / initial_water = 4 := by
  sorry

end initial_ratio_is_four_to_one_l569_56942


namespace root_sum_cubes_l569_56954

theorem root_sum_cubes (r s t : ℝ) : 
  (6 * r^3 + 1506 * r + 3009 = 0) →
  (6 * s^3 + 1506 * s + 3009 = 0) →
  (6 * t^3 + 1506 * t + 3009 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 1504.5 := by
sorry

end root_sum_cubes_l569_56954


namespace water_tank_capacity_l569_56944

theorem water_tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (added_amount : ℚ) (capacity : ℚ) : 
  initial_fraction = 1/7 →
  final_fraction = 1/5 →
  added_amount = 5 →
  initial_fraction * capacity + added_amount = final_fraction * capacity →
  capacity = 87.5 := by
sorry

end water_tank_capacity_l569_56944


namespace complex_modulus_problem_l569_56919

open Complex

theorem complex_modulus_problem (m : ℝ) : 
  (↑(1 + m * I) * (3 + I) * I).im ≠ 0 →
  (↑(1 + m * I) * (3 + I) * I).re = 0 →
  abs ((m + 3 * I) / (1 - I)) = 3 := by
  sorry

end complex_modulus_problem_l569_56919


namespace triangle_perimeter_l569_56948

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 13*x + 40 = 0 →
  3 + 4 + x > x ∧ 3 + x > 4 ∧ 4 + x > 3 →
  3 + 4 + x = 12 :=
by
  sorry

end triangle_perimeter_l569_56948


namespace largest_solution_floor_equation_l569_56976

theorem largest_solution_floor_equation :
  let f (x : ℝ) := ⌊x⌋ = 10 + 50 * (x - ⌊x⌋)
  ∃ (max_sol : ℝ), f max_sol ∧ max_sol = 59.98 ∧ ∀ y, f y → y ≤ max_sol :=
by sorry

end largest_solution_floor_equation_l569_56976


namespace sum_of_stacks_with_green_3_and_4_l569_56935

/-- Represents a card with a color and a number -/
structure Card where
  color : String
  number : Nat

/-- Represents a stack of cards -/
structure Stack where
  green : Card
  orange : Option Card

/-- Checks if a stack is valid according to the problem rules -/
def isValidStack (s : Stack) : Bool :=
  match s.orange with
  | none => true
  | some o => s.green.number ≤ o.number

/-- The set of all green cards -/
def greenCards : List Card :=
  [1, 2, 3, 4, 5].map (λ n => ⟨"green", n⟩)

/-- The set of all orange cards -/
def orangeCards : List Card :=
  [2, 3, 4, 5].map (λ n => ⟨"orange", n⟩)

/-- Calculates the sum of numbers in a stack -/
def stackSum (s : Stack) : Nat :=
  s.green.number + match s.orange with
  | none => 0
  | some o => o.number

/-- The main theorem to prove -/
theorem sum_of_stacks_with_green_3_and_4 :
  ∃ (s₁ s₂ : Stack),
    s₁.green.number = 3 ∧
    s₂.green.number = 4 ∧
    s₁.green ∈ greenCards ∧
    s₂.green ∈ greenCards ∧
    (∀ o₁ ∈ s₁.orange, o₁ ∈ orangeCards) ∧
    (∀ o₂ ∈ s₂.orange, o₂ ∈ orangeCards) ∧
    isValidStack s₁ ∧
    isValidStack s₂ ∧
    stackSum s₁ + stackSum s₂ = 14 :=
  sorry

end sum_of_stacks_with_green_3_and_4_l569_56935


namespace salary_growth_rate_l569_56908

/-- Proves that the given annual compound interest rate satisfies the salary growth equation -/
theorem salary_growth_rate (initial_salary final_salary total_increase : ℝ) 
  (years : ℕ) (rate : ℝ) 
  (h1 : initial_salary = final_salary - total_increase)
  (h2 : final_salary = 90000)
  (h3 : total_increase = 25000)
  (h4 : years = 3) :
  final_salary = initial_salary * (1 + rate)^years := by
  sorry

end salary_growth_rate_l569_56908


namespace quadratic_inequality_solution_l569_56925

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality_solution 
  (b c : ℝ) 
  (h1 : quadratic_function b c (-1) = 0)
  (h2 : quadratic_function b c 2 = 0) :
  {x : ℝ | quadratic_function b c x < 4} = Set.Ioo (-2) 3 :=
sorry

end quadratic_inequality_solution_l569_56925


namespace hyperbola_sum_l569_56945

-- Define the hyperbola
def Hyperbola (center focus vertex : ℝ × ℝ) : Prop :=
  let (h, k) := center
  let (_, f_y) := focus
  let (_, v_y) := vertex
  let a : ℝ := |k - v_y|
  let c : ℝ := |f_y - k|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  ∀ x y : ℝ, ((y - k)^2 / a^2) - ((x - h)^2 / b^2) = 1

-- State the theorem
theorem hyperbola_sum (center focus vertex : ℝ × ℝ) 
  (h : Hyperbola center focus vertex) 
  (hc : center = (3, 1)) 
  (hf : focus = (3, 9)) 
  (hv : vertex = (3, -2)) : 
  let (h, k) := center
  let (_, f_y) := focus
  let (_, v_y) := vertex
  let a : ℝ := |k - v_y|
  let c : ℝ := |f_y - k|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  h + k + a + b = 7 + Real.sqrt 55 := by
  sorry

end hyperbola_sum_l569_56945


namespace volume_ratio_minimum_l569_56950

noncomputable section

/-- The volume ratio of a cone to its circumscribed cylinder, given the sine of the cone's half-angle -/
def volume_ratio (s : ℝ) : ℝ := (1 + s)^3 / (6 * s * (1 - s^2))

/-- The theorem stating that the volume ratio is minimized when sin(θ) = 1/3 -/
theorem volume_ratio_minimum :
  ∀ s : ℝ, 0 < s → s < 1 →
  volume_ratio s ≥ 4/3 ∧
  (volume_ratio s = 4/3 ↔ s = 1/3) :=
sorry

end

end volume_ratio_minimum_l569_56950


namespace geometric_sequence_formula_l569_56904

/-- Given a geometric sequence {a_n} where the first three terms are a-1, a+1, and a+4 respectively,
    prove that the general formula for the nth term is a_n = 4 · (3/2)^(n-1) -/
theorem geometric_sequence_formula (a : ℝ) (a_n : ℕ → ℝ) :
  a_n 1 = a - 1 ∧ a_n 2 = a + 1 ∧ a_n 3 = a + 4 →
  ∀ n : ℕ, a_n n = 4 * (3/2) ^ (n - 1) := by
sorry

end geometric_sequence_formula_l569_56904


namespace tan_equality_proof_l569_56983

theorem tan_equality_proof (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (345 * π / 180) → n = -15 := by
  sorry

end tan_equality_proof_l569_56983


namespace alex_coin_distribution_l569_56951

def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 60) : 
  min_additional_coins num_friends initial_coins = 60 := by
  sorry

end alex_coin_distribution_l569_56951


namespace cl2_moles_required_l569_56920

/-- Represents the stoichiometric ratio of Cl2 to CH4 in the reaction -/
def cl2_ch4_ratio : ℚ := 4

/-- Represents the number of moles of CH4 given -/
def ch4_moles : ℚ := 3

/-- Represents the number of moles of CCl4 produced -/
def ccl4_moles : ℚ := 3

/-- Theorem stating that the number of moles of Cl2 required is 12 -/
theorem cl2_moles_required : cl2_ch4_ratio * ch4_moles = 12 := by
  sorry

end cl2_moles_required_l569_56920


namespace range_of_a_l569_56901

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 < 0}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- Define the theorem
theorem range_of_a (a : ℝ) : M ⊆ N a ↔ a ∈ Set.Ici 2 := by
  sorry

-- Note: Set.Ici 2 represents the set [2, +∞) in Lean

end range_of_a_l569_56901


namespace mode_of_data_set_l569_56921

def data_set : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_data_set :
  mode data_set = 3 := by
  sorry

end mode_of_data_set_l569_56921


namespace f_range_l569_56962

-- Define the function
def f (x : ℝ) := x^2 - 6*x + 7

-- State the theorem
theorem f_range :
  {y : ℝ | ∃ x ≥ 4, f x = y} = {y : ℝ | y ≥ -1} :=
by sorry

end f_range_l569_56962


namespace marilyn_shared_bottlecaps_l569_56933

/-- 
Given that Marilyn starts with 51 bottle caps and ends up with 15 bottle caps,
prove that she shared 36 bottle caps with Nancy.
-/
theorem marilyn_shared_bottlecaps : 
  let initial_caps : ℕ := 51
  let remaining_caps : ℕ := 15
  let shared_caps : ℕ := initial_caps - remaining_caps
  shared_caps = 36 := by sorry

end marilyn_shared_bottlecaps_l569_56933


namespace unique_solution_l569_56946

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x ^ 3 + f y ^ 3 + 3 * x * y) - 3 * x^2 * y^2 * f x

/-- The theorem stating that there is a unique function satisfying the equation -/
theorem unique_solution :
  ∃! f : ℝ → ℝ, SatisfiesEquation f ∧ ∀ x : ℝ, f x = 0 := by sorry

end unique_solution_l569_56946


namespace min_value_of_expression_l569_56911

theorem min_value_of_expression (x y : ℝ) 
  (hx : |x| ≤ 1) (hy : |y| ≤ 1) : 
  ∃ (min_val : ℝ), min_val = 3 ∧ 
  ∀ (a b : ℝ), |a| ≤ 1 → |b| ≤ 1 → |b + 1| + |2*b - a - 4| ≥ min_val :=
by sorry

end min_value_of_expression_l569_56911


namespace q_divided_by_p_equals_44_l569_56912

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 12

/-- The number of cards with each number -/
def cards_per_number : ℕ := 4

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The probability that all drawn cards bear the same number -/
noncomputable def p : ℚ := 12 / Nat.choose total_cards cards_drawn

/-- The probability that four cards bear one number and the fifth bears a different number -/
noncomputable def q : ℚ := 528 / Nat.choose total_cards cards_drawn

/-- Theorem stating that the ratio of q to p is 44 -/
theorem q_divided_by_p_equals_44 : q / p = 44 := by sorry

end q_divided_by_p_equals_44_l569_56912


namespace sum_of_ace_l569_56953

/-- Given 5 children with player numbers, prove that the sum of numbers for A, C, and E is 24 -/
theorem sum_of_ace (a b c d e : ℕ) : 
  a + b + c + d + e = 35 →
  b + c = 13 →
  a + b + c + e = 31 →
  b + c + e = 21 →
  b = 7 →
  a + c + e = 24 := by
sorry

end sum_of_ace_l569_56953


namespace product_prs_l569_56940

theorem product_prs (p r s : ℕ) : 
  4^p + 4^3 = 272 → 
  3^r + 27 = 81 → 
  2^s + 7^2 = 1024 → 
  p * r * s = 160 := by
sorry

end product_prs_l569_56940


namespace share_distribution_theorem_l569_56952

/-- Represents the share distribution problem among three children -/
def ShareDistribution (anusha_share babu_share esha_share k : ℚ) : Prop :=
  -- Total amount is 378
  anusha_share + babu_share + esha_share = 378 ∧
  -- Anusha's share is 84
  anusha_share = 84 ∧
  -- 12 times Anusha's share equals k times Babu's share
  12 * anusha_share = k * babu_share ∧
  -- k times Babu's share equals 6 times Esha's share
  k * babu_share = 6 * esha_share

/-- The main theorem stating that given the conditions, k equals 4 -/
theorem share_distribution_theorem :
  ∀ (anusha_share babu_share esha_share k : ℚ),
  ShareDistribution anusha_share babu_share esha_share k →
  k = 4 :=
by
  sorry


end share_distribution_theorem_l569_56952


namespace arithmetic_sequence_sum_l569_56955

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 20 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 + a 7 + a 9 + a 11 = 20

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  a 1 + a 13 = 8 := by
  sorry

end arithmetic_sequence_sum_l569_56955


namespace solve_for_n_l569_56941

variable (n : ℝ)

def f (x : ℝ) : ℝ := x^2 - 3*x + n
def g (x : ℝ) : ℝ := x^2 - 3*x + 5*n

theorem solve_for_n : 3 * f 3 = 2 * g 3 → n = 0 := by
  sorry

end solve_for_n_l569_56941
