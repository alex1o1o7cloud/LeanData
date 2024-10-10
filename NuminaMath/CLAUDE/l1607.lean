import Mathlib

namespace garden_perimeter_l1607_160741

/-- The perimeter of a rectangular garden with width 16 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 56 meters. -/
theorem garden_perimeter (garden_width playground_length playground_width : ℝ) :
  garden_width = 16 →
  playground_length = 16 →
  playground_width = 12 →
  garden_width * (playground_length * playground_width / garden_width) + 2 * garden_width = 56 := by
  sorry

end garden_perimeter_l1607_160741


namespace tan_B_in_triangle_l1607_160759

theorem tan_B_in_triangle (A B C : ℝ) (cosC : ℝ) (AC BC : ℝ) 
  (h1 : cosC = 2/3)
  (h2 : AC = 4)
  (h3 : BC = 3)
  (h4 : A + B + C = Real.pi) -- sum of angles in a triangle
  (h5 : 0 < AC ∧ 0 < BC) -- positive side lengths
  : Real.tan B = 4 * Real.sqrt 5 := by
  sorry

end tan_B_in_triangle_l1607_160759


namespace geometric_sequence_ratio_l1607_160773

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  a5_eq_3 : a 5 = 3
  a4_times_a7_eq_45 : a 4 * a 7 = 45

/-- The main theorem about the specific ratio in the geometric sequence -/
theorem geometric_sequence_ratio
  (seq : GeometricSequence) :
  (seq.a 7 - seq.a 9) / (seq.a 5 - seq.a 7) = 25 := by
  sorry

end geometric_sequence_ratio_l1607_160773


namespace max_slope_product_30deg_l1607_160733

/-- The maximum product of slopes for two lines intersecting at 30° with one slope four times the other -/
theorem max_slope_product_30deg (m₁ m₂ : ℝ) : 
  m₁ ≠ 0 → m₂ ≠ 0 →  -- nonhorizontal and nonvertical lines
  m₂ = 4 * m₁ →  -- one slope is 4 times the other
  |((m₂ - m₁) / (1 + m₁ * m₂))| = 1 / Real.sqrt 3 →  -- 30° angle between lines
  m₁ * m₂ ≤ (3 * Real.sqrt 3 + Real.sqrt 11)^2 / 16 :=
by sorry

end max_slope_product_30deg_l1607_160733


namespace propositions_b_and_c_are_true_l1607_160719

theorem propositions_b_and_c_are_true :
  (∀ a b : ℝ, |a| > |b| → a^2 > b^2) ∧
  (∀ a b c : ℝ, (a - b) * c^2 > 0 → a > b) := by
  sorry

end propositions_b_and_c_are_true_l1607_160719


namespace ellipse_axis_endpoint_distance_l1607_160704

/-- Given an ellipse with equation 4(x+2)^2 + 16y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ), 
    (∀ (x y : ℝ), 4 * (x + 2)^2 + 16 * y^2 = 64 ↔ (x + 2)^2 / 16 + y^2 / 4 = 1) ∧
    (C.1 = -2 ∧ C.2 = 4 ∨ C.1 = -2 ∧ C.2 = -4) ∧  -- C is an endpoint of the major axis
    (D.1 = 0 ∧ D.2 = 0 ∨ D.1 = -4 ∧ D.2 = 0) ∧    -- D is an endpoint of the minor axis
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end ellipse_axis_endpoint_distance_l1607_160704


namespace range_of_a_l1607_160775

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 5*x - 6 ≤ 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - 4*a^2 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a ≥ 0) →
  (∀ x, ¬(q x a) → ¬(p x)) →
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ≥ 5/2 :=
by
  sorry

end range_of_a_l1607_160775


namespace perfect_square_trinomial_l1607_160710

theorem perfect_square_trinomial (a : ℚ) : 
  (∃ r s : ℚ, a * x^2 + 20 * x + 9 = (r * x + s)^2) → a = 100 / 9 :=
by sorry

end perfect_square_trinomial_l1607_160710


namespace complex_imaginary_solution_l1607_160702

theorem complex_imaginary_solution (z : ℂ) : 
  (∃ b : ℝ, z = b * I) → 
  (∃ c : ℝ, (z - 3)^2 + 12 * I = c * I) → 
  (z = 3 * I ∨ z = -3 * I) := by
sorry

end complex_imaginary_solution_l1607_160702


namespace U_value_l1607_160735

theorem U_value : 
  let U := 1 / (4 - Real.sqrt 9) + 1 / (Real.sqrt 9 - Real.sqrt 8) - 
           1 / (Real.sqrt 8 - Real.sqrt 7) + 1 / (Real.sqrt 7 - Real.sqrt 6) - 
           1 / (Real.sqrt 6 - 3)
  U = 1 := by sorry

end U_value_l1607_160735


namespace cube_sum_l1607_160783

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where

/-- The number of faces in a cube -/
def Cube.faces (c : Cube) : ℕ := 6

/-- The number of edges in a cube -/
def Cube.edges (c : Cube) : ℕ := 12

/-- The number of vertices in a cube -/
def Cube.vertices (c : Cube) : ℕ := 8

/-- The sum of faces, edges, and vertices in a cube is 26 -/
theorem cube_sum (c : Cube) : c.faces + c.edges + c.vertices = 26 := by
  sorry

end cube_sum_l1607_160783


namespace y_value_l1607_160729

theorem y_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end y_value_l1607_160729


namespace divisibility_criterion_l1607_160756

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem divisibility_criterion (n : ℕ) (h : n > 1) :
  (Nat.factorial (n - 1)) % n = 0 ↔ is_composite n ∧ n ≠ 4 :=
sorry

end divisibility_criterion_l1607_160756


namespace sum_of_digits_of_N_l1607_160793

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_N (N : ℕ) (h : N^2 = 36^50 * 50^36) : sum_of_digits N = 10 := by
  sorry

end sum_of_digits_of_N_l1607_160793


namespace coin_packing_theorem_l1607_160743

/-- A coin is represented by its center and radius -/
structure Coin where
  center : ℝ × ℝ
  radius : ℝ

/-- The configuration of 12 coins forming a regular 12-gon -/
def outer_ring : List Coin := sorry

/-- The configuration of 7 coins inside the outer ring -/
def inner_coins : List Coin := sorry

/-- Two coins are tangent if the distance between their centers equals the sum of their radii -/
def are_tangent (c1 c2 : Coin) : Prop := sorry

/-- All coins in a list are mutually tangent -/
def all_tangent (coins : List Coin) : Prop := sorry

/-- The centers of the outer coins form a regular 12-gon -/
def is_regular_12gon (coins : List Coin) : Prop := sorry

theorem coin_packing_theorem :
  is_regular_12gon outer_ring ∧
  all_tangent outer_ring ∧
  (∀ c ∈ inner_coins, ∀ o ∈ outer_ring, are_tangent c o ∨ c = o) ∧
  all_tangent inner_coins ∧
  (List.length outer_ring = 12) ∧
  (List.length inner_coins = 7) := by
  sorry

end coin_packing_theorem_l1607_160743


namespace problem_statement_l1607_160751

theorem problem_statement (x y M : ℝ) (h : M / ((x * y + y^2) / (x - y)^2) = (x^2 - y^2) / y) :
  M = (x + y)^2 / (x - y) := by sorry

end problem_statement_l1607_160751


namespace college_students_count_l1607_160763

theorem college_students_count (num_girls : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) : 
  num_girls = 120 → ratio_boys = 8 → ratio_girls = 5 →
  (num_girls + (num_girls * ratio_boys) / ratio_girls : ℕ) = 312 := by
sorry

end college_students_count_l1607_160763


namespace a_formula_l1607_160782

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 5
  | n + 1 => ⌊a n⌋ + 1 / (a n - ⌊a n⌋)

theorem a_formula (n : ℕ) : a n = 4 * n + Real.sqrt 5 := by
  sorry

end a_formula_l1607_160782


namespace sqrt_equation_solution_l1607_160737

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (2 - 3 * z) = 9 :=
by
  -- The unique solution is z = -79/3
  use -79/3
  constructor
  · -- Prove that -79/3 satisfies the equation
    sorry
  · -- Prove that any z satisfying the equation must equal -79/3
    sorry

#check sqrt_equation_solution

end sqrt_equation_solution_l1607_160737


namespace solution_set_part1_solution_set_characterization_l1607_160767

-- Part 1
theorem solution_set_part1 (x : ℝ) :
  -5 * x^2 + 3 * x + 2 > 0 ↔ -2/5 < x ∧ x < 1 := by sorry

-- Part 2
def solution_set_part2 (a x : ℝ) : Prop :=
  a * x^2 + 3 * x + 2 > -a * x - 1

theorem solution_set_characterization (a x : ℝ) (ha : a > 0) :
  solution_set_part2 a x ↔
    (0 < a ∧ a < 3 ∧ (x < -3/a ∨ x > -1)) ∨
    (a = 3 ∧ x ≠ -1) ∨
    (a > 3 ∧ (x < -1 ∨ x > -3/a)) := by sorry

end solution_set_part1_solution_set_characterization_l1607_160767


namespace square_perimeter_l1607_160768

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 392) (h2 : side^2 = area) : 
  4 * side = 112 := by
  sorry

end square_perimeter_l1607_160768


namespace cement_bags_ratio_l1607_160786

theorem cement_bags_ratio (bags1 : ℕ) (weight1 : ℚ) (cost1 : ℚ) (cost2 : ℚ) (weight_ratio : ℚ) :
  bags1 = 80 →
  weight1 = 50 →
  cost1 = 6000 →
  cost2 = 10800 →
  weight_ratio = 3 / 5 →
  (cost2 / (cost1 / bags1 * weight_ratio)) / bags1 = 3 / 1 := by
sorry

end cement_bags_ratio_l1607_160786


namespace vector_inequality_not_always_holds_l1607_160750

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_inequality_not_always_holds :
  ∃ (a b : V), ‖a - b‖ > |‖a‖ - ‖b‖| := by sorry

end vector_inequality_not_always_holds_l1607_160750


namespace salt_production_increase_l1607_160705

/-- Proves that given an initial production of 1000 tonnes in January and an average
    monthly production of 1550 tonnes for the year, the constant monthly increase
    in production from February to December is 100 tonnes. -/
theorem salt_production_increase (initial_production : ℕ) (average_production : ℕ) 
  (monthly_increase : ℕ) (h1 : initial_production = 1000) 
  (h2 : average_production = 1550) :
  (monthly_increase = 100 ∧ 
   (12 * initial_production + (monthly_increase * 11 * 12 / 2) = 12 * average_production)) := by
  sorry

end salt_production_increase_l1607_160705


namespace k_satisfies_conditions_l1607_160713

/-- The number of digits in the second factor of (9)(999...9) -/
def k : ℕ := 55

/-- The resulting integer from the multiplication (9)(999...9) -/
def result (n : ℕ) : ℕ := 9 * (10^n - 1)

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- The theorem stating that k satisfies the given conditions -/
theorem k_satisfies_conditions : digit_sum (result k) = 500 := by sorry

end k_satisfies_conditions_l1607_160713


namespace leading_coefficient_of_g_l1607_160776

/-- A polynomial g satisfying g(x + 1) - g(x) = 4x + 6 for all x has a leading coefficient of 2 -/
theorem leading_coefficient_of_g (g : ℝ → ℝ) (hg : ∀ x, g (x + 1) - g x = 4 * x + 6) :
  ∃ (a b c : ℝ), (∀ x, g x = 2 * x^2 + a * x + b) ∧ c = 2 ∧ c ≠ 0 ∧ 
  (∀ d, (∀ x, g x = d * x^2 + a * x + b) → d ≤ c) := by
  sorry

end leading_coefficient_of_g_l1607_160776


namespace hyperbola_eccentricity_l1607_160766

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  hyperbola 3 4 → e = 5/3 := by
  sorry

end hyperbola_eccentricity_l1607_160766


namespace relay_team_permutations_l1607_160791

theorem relay_team_permutations (n : ℕ) (h : n = 3) : Nat.factorial n = 6 := by
  sorry

end relay_team_permutations_l1607_160791


namespace int_tan_triangle_values_l1607_160774

-- Define a triangle with integer tangents
structure IntTanTriangle where
  α : Real
  β : Real
  γ : Real
  tan_α : Int
  tan_β : Int
  tan_γ : Int
  sum_angles : α + β + γ = Real.pi
  tan_α_def : Real.tan α = tan_α
  tan_β_def : Real.tan β = tan_β
  tan_γ_def : Real.tan γ = tan_γ

-- Theorem statement
theorem int_tan_triangle_values (t : IntTanTriangle) :
  (t.tan_α = 1 ∧ t.tan_β = 2 ∧ t.tan_γ = 3) ∨
  (t.tan_α = 1 ∧ t.tan_β = 3 ∧ t.tan_γ = 2) ∨
  (t.tan_α = 2 ∧ t.tan_β = 1 ∧ t.tan_γ = 3) ∨
  (t.tan_α = 2 ∧ t.tan_β = 3 ∧ t.tan_γ = 1) ∨
  (t.tan_α = 3 ∧ t.tan_β = 1 ∧ t.tan_γ = 2) ∨
  (t.tan_α = 3 ∧ t.tan_β = 2 ∧ t.tan_γ = 1) :=
by sorry

end int_tan_triangle_values_l1607_160774


namespace division_and_addition_l1607_160769

theorem division_and_addition : (-75) / (-25) + (1 / 2) = 7 / 2 := by
  sorry

end division_and_addition_l1607_160769


namespace least_even_perimeter_l1607_160716

theorem least_even_perimeter (a b c : ℕ) : 
  a = 24 →
  b = 37 →
  c ≥ a ∧ c ≥ b →
  a + b + c > a + b →
  Even (a + b + c) →
  (∀ x : ℕ, x < c → ¬(Even (a + b + x) ∧ a + b + x > a + b)) →
  a + b + c = 100 := by
  sorry

end least_even_perimeter_l1607_160716


namespace range_of_b_l1607_160757

theorem range_of_b (a : ℝ) (h1 : 0 < a) (h2 : a ≤ 5/4) :
  (∃ (b : ℝ), b > 0 ∧ 
    (∀ (x : ℝ), |x - a| < b → |x - a^2| < 1/2) ∧
    (∀ (c : ℝ), c > b → ∃ (y : ℝ), |y - a| < c ∧ |y - a^2| ≥ 1/2)) ∧
  (∀ (b : ℝ), (∀ (x : ℝ), |x - a| < b → |x - a^2| < 1/2) → b ≤ 3/16) :=
sorry

end range_of_b_l1607_160757


namespace f_properties_l1607_160732

noncomputable def f (x : ℝ) := Real.sin x * (Real.sin x + Real.sqrt 3 * Real.cos x)

theorem f_properties :
  ∃ (T : ℝ) (M : ℝ) (S : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ t, 0 < t → t < T → ¬ (∀ x, f (x + t) = f x)) ∧
    (∀ x, f x ≤ M) ∧
    (∃ x, f x = M) ∧
    T = π ∧
    M = 3/2 ∧
    (∀ A B C a b c : ℝ,
      0 < A ∧ A < π/2 ∧
      0 < B ∧ B < π/2 ∧
      0 < C ∧ C < π/2 ∧
      A + B + C = π ∧
      f (A/2) = 1 ∧
      a = 2 * Real.sqrt 3 ∧
      a = b * Real.sin C ∧
      b = c * Real.sin A ∧
      c = a * Real.sin B →
      1/2 * b * c * Real.sin A ≤ S) ∧
    S = 3 * Real.sqrt 3 :=
by sorry

end f_properties_l1607_160732


namespace some_number_value_l1607_160748

theorem some_number_value (x : ℝ) (some_number : ℝ) 
  (h1 : 5 + 7 / x = some_number - 5 / x)
  (h2 : x = 12) : 
  some_number = 6 := by
sorry

end some_number_value_l1607_160748


namespace percentage_calculation_l1607_160714

theorem percentage_calculation (a : ℝ) (x : ℝ) (h1 : a = 140) (h2 : (x / 100) * a = 70) : x = 50 := by
  sorry

end percentage_calculation_l1607_160714


namespace solve_for_b_l1607_160712

theorem solve_for_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 315 * b) : b = 7 := by
  sorry

end solve_for_b_l1607_160712


namespace expression_simplification_l1607_160721

theorem expression_simplification (a b : ℝ) 
  (h : |a - 1| + b^2 - 6*b + 9 = 0) : 
  ((3*a + 2*b)*(3*a - 2*b) + (3*a - b)^2 - b*(2*a - 3*b)) / (2*a) = -3 :=
by sorry

end expression_simplification_l1607_160721


namespace total_peanuts_l1607_160728

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := 4

/-- The number of peanuts Mary adds to the box -/
def added_peanuts : ℕ := 4

/-- Theorem: The total number of peanuts in the box is 8 -/
theorem total_peanuts : initial_peanuts + added_peanuts = 8 := by
  sorry

end total_peanuts_l1607_160728


namespace first_platform_length_l1607_160778

/-- The length of a train in meters. -/
def train_length : ℝ := 350

/-- The time taken to cross the first platform in seconds. -/
def time_first : ℝ := 15

/-- The length of the second platform in meters. -/
def length_second : ℝ := 250

/-- The time taken to cross the second platform in seconds. -/
def time_second : ℝ := 20

/-- The length of the first platform in meters. -/
def length_first : ℝ := 100

theorem first_platform_length :
  (train_length + length_first) / time_first = (train_length + length_second) / time_second :=
by sorry

end first_platform_length_l1607_160778


namespace mans_age_puzzle_l1607_160738

theorem mans_age_puzzle (A : ℕ) (h : A = 72) :
  ∃ N : ℕ, (A + 6) * N - (A - 6) * N = A ∧ N = 6 := by
  sorry

end mans_age_puzzle_l1607_160738


namespace cube_root_of_216_l1607_160727

theorem cube_root_of_216 (y : ℝ) : (Real.sqrt y)^3 = 216 → y = 36 := by
  sorry

end cube_root_of_216_l1607_160727


namespace youngest_child_age_l1607_160779

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The set of ages of the six children -/
def childrenAges (x : ℕ) : Finset ℕ :=
  {x, x + 2, x + 6, x + 8, x + 12, x + 14}

/-- Theorem stating that the youngest child's age is 5 -/
theorem youngest_child_age :
  ∃ (x : ℕ), x = 5 ∧ 
    (∀ y ∈ childrenAges x, isPrime y) ∧
    (childrenAges x).card = 6 :=
  sorry

end youngest_child_age_l1607_160779


namespace max_integer_difference_l1607_160794

theorem max_integer_difference (x y : ℤ) (hx : 4 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  (∃ (a b : ℤ), 4 < a ∧ a < 6 ∧ 6 < b ∧ b < 10 ∧ b - a ≤ y - x) ∧ y - x ≤ 4 :=
by sorry

end max_integer_difference_l1607_160794


namespace division_by_fraction_twelve_divided_by_three_fifths_l1607_160790

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b := by sorry

theorem twelve_divided_by_three_fifths :
  12 / (3 / 5) = 20 := by sorry

end division_by_fraction_twelve_divided_by_three_fifths_l1607_160790


namespace triangle_similarity_l1607_160724

-- Define the types for points and triangles
variable (Point : Type) (Triangle : Type)

-- Define the necessary relations and properties
variable (is_scalene : Triangle → Prop)
variable (point_on_segment : Point → Point → Point → Prop)
variable (similar_triangles : Triangle → Triangle → Prop)
variable (point_on_line : Point → Point → Point → Prop)
variable (equal_distance : Point → Point → Point → Point → Prop)

-- State the theorem
theorem triangle_similarity 
  (A B C A₁ B₁ C₁ A₂ B₂ C₂ : Point) 
  (ABC A₁B₁C₁ A₂B₂C₂ : Triangle) :
  is_scalene ABC →
  point_on_segment A₁ B C →
  point_on_segment B₁ C A →
  point_on_segment C₁ A B →
  similar_triangles A₁B₁C₁ ABC →
  point_on_line A₂ B₁ C₁ →
  equal_distance A A₂ A₁ A₂ →
  point_on_line B₂ C₁ A₁ →
  equal_distance B B₂ B₁ B₂ →
  point_on_line C₂ A₁ B₁ →
  equal_distance C C₂ C₁ C₂ →
  similar_triangles A₂B₂C₂ ABC :=
by sorry

end triangle_similarity_l1607_160724


namespace double_roll_probability_l1607_160799

def die_roll : Finset (Nat × Nat) := Finset.product (Finset.range 6) (Finset.range 6)

def favorable_outcomes : Finset (Nat × Nat) :=
  {(0, 1), (1, 3), (2, 5)}

theorem double_roll_probability :
  (favorable_outcomes.card : ℚ) / die_roll.card = 1 / 12 := by
  sorry

end double_roll_probability_l1607_160799


namespace necessary_but_not_sufficient_l1607_160701

-- Define the solution sets
def solution_set_1 : Set ℝ := {x : ℝ | (x + 3) * (x - 1) = 0}
def solution_set_2 : Set ℝ := {x : ℝ | x - 1 = 0}

-- State the theorem
theorem necessary_but_not_sufficient :
  (solution_set_2 ⊆ solution_set_1) ∧ (solution_set_2 ≠ solution_set_1) :=
by sorry

end necessary_but_not_sufficient_l1607_160701


namespace p_start_time_correct_l1607_160758

/-- The time when J starts walking (in hours after midnight) -/
def j_start_time : ℝ := 12

/-- J's walking speed in km/h -/
def j_speed : ℝ := 6

/-- P's cycling speed in km/h -/
def p_speed : ℝ := 8

/-- The time when J is 3 km behind P (in hours after midnight) -/
def final_time : ℝ := 19.3

/-- The distance J is behind P at the final time (in km) -/
def distance_behind : ℝ := 3

/-- The time when P starts following J (in hours after midnight) -/
def p_start_time : ℝ := j_start_time + 1.45

theorem p_start_time_correct :
  j_speed * (final_time - j_start_time) + distance_behind =
  p_speed * (final_time - p_start_time) := by sorry

end p_start_time_correct_l1607_160758


namespace star_computation_l1607_160787

/-- Operation ⭐ defined as (5a + b) / (a - b) -/
def star (a b : ℚ) : ℚ := (5 * a + b) / (a - b)

theorem star_computation :
  star (star 7 (star 2 5)) 3 = -31 := by
  sorry

end star_computation_l1607_160787


namespace p_and_q_true_p_and_not_q_false_l1607_160740

-- Define proposition p
def p : Prop := ∀ m : ℝ, ∃ x : ℝ, x^2 - m*x - 1 = 0

-- Define proposition q
def q : Prop := ∃ x₀ : ℕ, x₀^2 - 2*x₀ - 1 ≤ 0

-- Theorem stating that p and q are true
theorem p_and_q_true : p ∧ q := by sorry

-- Theorem stating that p ∧ (¬q) is false
theorem p_and_not_q_false : ¬(p ∧ ¬q) := by sorry

end p_and_q_true_p_and_not_q_false_l1607_160740


namespace quadratic_sequence_exists_l1607_160706

theorem quadratic_sequence_exists (b c : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ 
  ∀ i : ℕ, i ≥ 1 → i ≤ n → |a i - a (i-1)| = i^2 := by
  sorry

end quadratic_sequence_exists_l1607_160706


namespace function_composition_multiplication_l1607_160722

-- Define the composition operation
def compose (f g : ℝ → ℝ) : ℝ → ℝ := λ x => f (g x)

-- Define the multiplication operation
def multiply (f g : ℝ → ℝ) : ℝ → ℝ := λ x => f x * g x

-- State the theorem
theorem function_composition_multiplication (f g h : ℝ → ℝ) :
  compose (multiply f g) h = multiply (compose f h) (compose g h) := by
  sorry

end function_composition_multiplication_l1607_160722


namespace parabola_properties_l1607_160795

/-- A parabola with given properties -/
structure Parabola where
  vertex : ℝ × ℝ
  axis_vertical : Bool
  passing_point : ℝ × ℝ

/-- Shift vector -/
def shift_vector : ℝ × ℝ := (2, 3)

/-- Our specific parabola -/
def our_parabola : Parabola := {
  vertex := (3, -2),
  axis_vertical := true,
  passing_point := (5, 6)
}

/-- The equation of our parabola -/
def parabola_equation (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 16

/-- The new vertex after shifting -/
def new_vertex : ℝ × ℝ := (5, 1)

theorem parabola_properties :
  (∀ x, parabola_equation x = 2 * (x - our_parabola.vertex.1)^2 + our_parabola.vertex.2) ∧
  parabola_equation our_parabola.passing_point.1 = our_parabola.passing_point.2 ∧
  new_vertex = (our_parabola.vertex.1 + shift_vector.1, our_parabola.vertex.2 + shift_vector.2) :=
by sorry

end parabola_properties_l1607_160795


namespace d_bounds_l1607_160755

-- Define the circle
def Circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function
def d (P : ℝ × ℝ) : ℝ :=
  let (px, py) := P
  (px - A.1)^2 + (py - A.2)^2 + (px - B.1)^2 + (py - B.2)^2

-- Theorem statement
theorem d_bounds :
  ∀ P : ℝ × ℝ, Circle P.1 P.2 → 
  66 - 16 * Real.sqrt 2 ≤ d P ∧ d P ≤ 66 + 16 * Real.sqrt 2 :=
by sorry

end d_bounds_l1607_160755


namespace lucas_numbers_l1607_160784

theorem lucas_numbers (a b : ℤ) : 
  (3 * a + 4 * b = 161) → 
  ((a = 17 ∨ b = 17) → (a = 31 ∨ b = 31)) :=
by sorry

end lucas_numbers_l1607_160784


namespace perpendicular_line_equation_l1607_160703

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem perpendicular_line_equation :
  ∃ (l : Line),
    perpendicular l (Line.mk 2 (-3) 4) ∧
    point_on_line (-1) 2 l ∧
    l = Line.mk 3 2 (-1) := by
  sorry

end perpendicular_line_equation_l1607_160703


namespace games_that_didnt_work_l1607_160734

/-- The number of games that didn't work given Ned's game purchases and good games. -/
theorem games_that_didnt_work (friend_games garage_sale_games good_games : ℕ) : 
  friend_games = 50 → garage_sale_games = 27 → good_games = 3 → 
  friend_games + garage_sale_games - good_games = 74 := by
  sorry

end games_that_didnt_work_l1607_160734


namespace centroid_maximizes_dist_product_l1607_160715

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- Distance from a point to a line defined by two points --/
def distToLine (P : Point) (A B : Point) : ℝ := sorry

/-- The centroid of a triangle --/
def centroid (t : Triangle) : Point := sorry

/-- Product of distances from a point to the sides of a triangle --/
def distProduct (P : Point) (t : Triangle) : ℝ := 
  distToLine P t.A t.B * distToLine P t.B t.C * distToLine P t.C t.A

/-- Predicate to check if a point is inside a triangle --/
def isInside (P : Point) (t : Triangle) : Prop := sorry

theorem centroid_maximizes_dist_product (t : Triangle) :
  ∀ P, isInside P t → distProduct P t ≤ distProduct (centroid t) t :=
sorry

end centroid_maximizes_dist_product_l1607_160715


namespace game_points_theorem_l1607_160771

theorem game_points_theorem (eric : ℕ) (mark : ℕ) (samanta : ℕ) : 
  mark = eric + eric / 2 →
  samanta = mark + 8 →
  eric + mark + samanta = 32 →
  eric = 6 := by
sorry

end game_points_theorem_l1607_160771


namespace library_book_sale_l1607_160709

theorem library_book_sale (initial_books : ℕ) (remaining_fraction : ℚ) : 
  initial_books = 9900 →
  remaining_fraction = 4/6 →
  initial_books * (1 - remaining_fraction) = 3300 :=
by sorry

end library_book_sale_l1607_160709


namespace modular_exponentiation_16_cube_mod_7_l1607_160723

theorem modular_exponentiation_16_cube_mod_7 :
  ∃ m : ℕ, 16^3 ≡ m [ZMOD 7] ∧ 0 ≤ m ∧ m < 7 → m = 1 := by
  sorry

end modular_exponentiation_16_cube_mod_7_l1607_160723


namespace arc_length_calculation_l1607_160744

theorem arc_length_calculation (circumference : ℝ) (central_angle : ℝ) 
  (h1 : circumference = 72) 
  (h2 : central_angle = 45) : 
  (central_angle / 360) * circumference = 9 := by
  sorry

end arc_length_calculation_l1607_160744


namespace ticket_cost_l1607_160777

/-- The cost of a single ticket at the fair, given the initial number of tickets,
    remaining tickets, and total amount spent on the ferris wheel. -/
theorem ticket_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (total_spent : ℕ) :
  initial_tickets > remaining_tickets →
  total_spent % (initial_tickets - remaining_tickets) = 0 →
  total_spent / (initial_tickets - remaining_tickets) = 9 :=
by
  intro h_tickets h_divisible
  sorry

#check ticket_cost 13 4 81

end ticket_cost_l1607_160777


namespace algebraic_expression_value_l1607_160765

/-- Given x = √5 + 2 and y = √5 - 2, prove that x^2 - y + xy = 12 + 3√5 -/
theorem algebraic_expression_value :
  let x : ℝ := Real.sqrt 5 + 2
  let y : ℝ := Real.sqrt 5 - 2
  x^2 - y + x*y = 12 + 3 * Real.sqrt 5 := by
sorry

end algebraic_expression_value_l1607_160765


namespace phi_value_l1607_160789

theorem phi_value (φ : Real) (a : Real) :
  φ ∈ Set.Icc 0 (2 * Real.pi) →
  (∃ x₁ x₂ x₃ : Real,
    x₁ ∈ Set.Icc 0 Real.pi ∧
    x₂ ∈ Set.Icc 0 Real.pi ∧
    x₃ ∈ Set.Icc 0 Real.pi ∧
    Real.sin (2 * x₁ + φ) = a ∧
    Real.sin (2 * x₂ + φ) = a ∧
    Real.sin (2 * x₃ + φ) = a ∧
    x₁ + x₂ + x₃ = 7 * Real.pi / 6) →
  φ = Real.pi / 3 ∨ φ = 4 * Real.pi / 3 :=
by sorry

end phi_value_l1607_160789


namespace abs_equation_solution_l1607_160726

theorem abs_equation_solution : ∃! x : ℝ, |x - 3| = 5 - x := by
  sorry

end abs_equation_solution_l1607_160726


namespace gcd_12345_67890_l1607_160745

theorem gcd_12345_67890 : Nat.gcd 12345 67890 = 15 := by
  sorry

end gcd_12345_67890_l1607_160745


namespace sallys_onions_l1607_160736

theorem sallys_onions (fred_onions : ℕ) (given_to_sara : ℕ) (remaining_onions : ℕ) : ℕ :=
  sorry

end sallys_onions_l1607_160736


namespace derivative_of_periodic_is_periodic_l1607_160764

/-- A function f is periodic with period T if f(x + T) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- The main theorem: If f is differentiable and periodic with period T,
    then its derivative f' is also periodic with period T -/
theorem derivative_of_periodic_is_periodic
    (f : ℝ → ℝ) (T : ℝ) (hT : T > 0) (hf : Differentiable ℝ f) (hper : IsPeriodic f T) :
    IsPeriodic (deriv f) T := by
  sorry

end derivative_of_periodic_is_periodic_l1607_160764


namespace max_students_distribution_l1607_160718

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1204) (h2 : pencils = 840) :
  Nat.gcd pens pencils = 16 := by
  sorry

end max_students_distribution_l1607_160718


namespace first_discount_percentage_l1607_160788

/-- Proves that the first discount percentage is 25% given the original price, final price, and second discount percentage. -/
theorem first_discount_percentage 
  (original_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) 
  (h1 : original_price = 33.78)
  (h2 : final_price = 19)
  (h3 : second_discount = 25) :
  ∃ (first_discount : ℝ), 
    first_discount = 25 ∧ 
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) := by
  sorry


end first_discount_percentage_l1607_160788


namespace cauchy_inequality_2d_l1607_160785

theorem cauchy_inequality_2d (a b c d : ℝ) : 
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) ∧ 
  ((a * c + b * d)^2 = (a^2 + b^2) * (c^2 + d^2) ↔ a * d = b * c) :=
sorry

end cauchy_inequality_2d_l1607_160785


namespace min_value_xy_l1607_160780

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 / x) + (8 / y) = 1) :
  xy ≥ 64 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (2 / x₀) + (8 / y₀) = 1 ∧ x₀ * y₀ = 64 :=
by sorry

end min_value_xy_l1607_160780


namespace pr_qs_ratio_l1607_160742

/-- Given four points P, Q, R, and S on a number line, prove that the ratio of lengths PR:QS is 7:12 -/
theorem pr_qs_ratio (P Q R S : ℝ) (hP : P = 3) (hQ : Q = 5) (hR : R = 10) (hS : S = 17) :
  (R - P) / (S - Q) = 7 / 12 := by
  sorry

end pr_qs_ratio_l1607_160742


namespace expression_values_l1607_160708

theorem expression_values (x y z : ℝ) (h : x * y * z ≠ 0) :
  let expr := |x| / x + y / |y| + |z| / z
  expr = 1 ∨ expr = -1 ∨ expr = 3 ∨ expr = -3 :=
by sorry

end expression_values_l1607_160708


namespace cos_sixty_degrees_l1607_160730

theorem cos_sixty_degrees : Real.cos (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end cos_sixty_degrees_l1607_160730


namespace sum_of_coefficients_10_11_l1607_160711

/-- Given that (x-1)^21 = a + a₁x + a₂x² + ... + a₂₁x²¹, prove that a₁₀ + a₁₁ = 0 -/
theorem sum_of_coefficients_10_11 (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ a₁₅ a₁₆ a₁₇ a₁₈ a₁₉ a₂₀ a₂₁ : ℝ) :
  (∀ x : ℝ, (x - 1)^21 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + 
             a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12 + a₁₃*x^13 + a₁₄*x^14 + a₁₅*x^15 + a₁₆*x^16 + a₁₇*x^17 + a₁₈*x^18 + 
             a₁₉*x^19 + a₂₀*x^20 + a₂₁*x^21) →
  a₁₀ + a₁₁ = 0 := by
  sorry

end sum_of_coefficients_10_11_l1607_160711


namespace machine_time_calculation_l1607_160731

/-- Given a machine that can make a certain number of shirts per minute
    and has made a total number of shirts, calculate the time it worked. -/
def machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) : ℚ :=
  total_shirts / shirts_per_minute

/-- Theorem stating that for a machine making 3 shirts per minute
    and having made 6 shirts in total, it worked for 2 minutes. -/
theorem machine_time_calculation :
  machine_working_time 3 6 = 2 := by
  sorry

end machine_time_calculation_l1607_160731


namespace reading_time_per_disc_l1607_160796

/-- Proves that given a total reading time of 480 minutes, disc capacity of 60 minutes,
    and the conditions of using the smallest possible number of discs with equal reading time on each disc,
    the reading time per disc is 60 minutes. -/
theorem reading_time_per_disc (total_time : ℕ) (disc_capacity : ℕ) (reading_per_disc : ℕ) :
  total_time = 480 →
  disc_capacity = 60 →
  reading_per_disc * (total_time / disc_capacity) = total_time →
  reading_per_disc ≤ disc_capacity →
  reading_per_disc = 60 := by
  sorry

end reading_time_per_disc_l1607_160796


namespace roots_in_interval_l1607_160747

theorem roots_in_interval (m : ℝ) :
  (∀ x, 4 * x^2 - (3 * m + 1) * x - m - 2 = 0 → -1 < x ∧ x < 2) ↔ -1 < m ∧ m < 12/7 := by
  sorry

end roots_in_interval_l1607_160747


namespace exponent_multiplication_l1607_160798

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l1607_160798


namespace max_at_2_implies_c_6_l1607_160707

/-- The function f(x) = x(x-c)² has a maximum value at x=2 -/
def has_max_at_2 (c : ℝ) : Prop :=
  ∀ x : ℝ, x * (x - c)^2 ≤ 2 * (2 - c)^2

/-- Theorem: If f(x) = x(x-c)² has a maximum value at x=2, then c = 6 -/
theorem max_at_2_implies_c_6 :
  ∀ c : ℝ, has_max_at_2 c → c = 6 := by
  sorry

end max_at_2_implies_c_6_l1607_160707


namespace point_C_in_fourth_quadrant_l1607_160717

/-- A point in the 2D Cartesian coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point we want to prove is in the fourth quadrant -/
def point_C : Point :=
  { x := 1, y := -2 }

/-- Theorem: point_C is in the fourth quadrant -/
theorem point_C_in_fourth_quadrant : is_in_fourth_quadrant point_C := by
  sorry

end point_C_in_fourth_quadrant_l1607_160717


namespace solve_a_b_l1607_160725

def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}

def A (b : ℝ) : Set ℝ := {b, 2}

def complement_U_A (a b : ℝ) : Set ℝ := U a \ A b

theorem solve_a_b (a b : ℝ) : 
  complement_U_A a b = {5} →
  ((a = 2 ∨ a = -4) ∧ b = 3) :=
by sorry

end solve_a_b_l1607_160725


namespace complement_A_in_U_l1607_160797

def U : Set ℤ := {-3, -1, 0, 1, 3}

def A : Set ℤ := {x | x^2 - 2*x - 3 = 0}

theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {-3, 0, 1} := by sorry

end complement_A_in_U_l1607_160797


namespace cathy_remaining_money_l1607_160762

/-- Calculates the remaining money in Cathy's wallet after all expenditures --/
def remaining_money (initial_amount dad_sent mom_sent_multiplier book_cost cab_ride_percent dinner_percent : ℝ) : ℝ :=
  let total_from_parents := dad_sent + mom_sent_multiplier * dad_sent
  let total_initial := total_from_parents + initial_amount
  let food_budget := 0.4 * total_initial
  let after_book := total_initial - book_cost
  let cab_cost := cab_ride_percent * after_book
  let after_cab := after_book - cab_cost
  let dinner_cost := dinner_percent * food_budget
  after_cab - dinner_cost

/-- Theorem stating that Cathy's remaining money is $52.44 --/
theorem cathy_remaining_money :
  remaining_money 12 25 2 15 0.03 0.5 = 52.44 := by
  sorry

end cathy_remaining_money_l1607_160762


namespace export_volume_equation_l1607_160746

def export_volume_2023 : ℝ := 107
def export_volume_2013 : ℝ → ℝ := λ x => x

theorem export_volume_equation (x : ℝ) : 
  export_volume_2023 = 4 * (export_volume_2013 x) + 3 ↔ 4 * x + 3 = 107 :=
by sorry

end export_volume_equation_l1607_160746


namespace rhombus_count_in_triangle_l1607_160739

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ

/-- Represents a rhombus composed of smaller equilateral triangles -/
structure Rhombus where
  num_triangles : ℕ

/-- The number of rhombuses in one direction -/
def rhombuses_in_one_direction (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The main theorem -/
theorem rhombus_count_in_triangle (big_triangle : EquilateralTriangle) 
  (small_triangle : EquilateralTriangle) (rhombus : Rhombus) : 
  big_triangle.side_length = 10 →
  small_triangle.side_length = 1 →
  rhombus.num_triangles = 8 →
  (rhombuses_in_one_direction 7) * 3 = 84 := by
  sorry

#check rhombus_count_in_triangle

end rhombus_count_in_triangle_l1607_160739


namespace adrian_cards_l1607_160749

theorem adrian_cards (n : ℕ) : 
  (∃ k : ℕ, 
    k ≥ 1 ∧ 
    k + n - 1 ≤ 2 * n ∧ 
    (2 * n * (2 * n + 1)) / 2 - (n * k + (n * (n - 1)) / 2) = 1615) →
  (n = 34 ∨ n = 38) :=
by sorry

end adrian_cards_l1607_160749


namespace reciprocal_equation_solution_l1607_160772

theorem reciprocal_equation_solution (x : ℝ) : 
  (2 - 1 / (3 - 2 * x) = 1 / (3 - 2 * x)) → x = 1 := by
  sorry

end reciprocal_equation_solution_l1607_160772


namespace probability_two_red_two_blue_l1607_160753

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

theorem probability_two_red_two_blue :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles selected_marbles = 56 / 147 :=
by sorry

end probability_two_red_two_blue_l1607_160753


namespace construction_problem_l1607_160700

/-- Represents the construction plan for a quarter --/
structure ConstructionPlan where
  ordinary : ℝ
  elevated : ℝ
  tunnel : ℝ

/-- Represents the cost per kilometer for each type of construction --/
structure CostPerKm where
  ordinary : ℝ
  elevated : ℝ
  tunnel : ℝ

/-- Calculates the total cost of a construction plan given the cost per kilometer --/
def totalCost (plan : ConstructionPlan) (cost : CostPerKm) : ℝ :=
  plan.ordinary * cost.ordinary + plan.elevated * cost.elevated + plan.tunnel * cost.tunnel

theorem construction_problem (a : ℝ) :
  let q1_plan : ConstructionPlan := { ordinary := 32, elevated := 21, tunnel := 3 }
  let q1_cost : CostPerKm := { ordinary := 1, elevated := 2, tunnel := 4 }
  let q2_plan : ConstructionPlan := { ordinary := 32 - 9*a, elevated := 21 - 2*a, tunnel := 3 + a }
  let q2_cost : CostPerKm := { ordinary := 1, elevated := 2 + 0.5*a, tunnel := 4 }
  
  (∀ x, x ≤ 3 → 56 - 32 - x ≥ 7*x) ∧ 
  (totalCost q1_plan q1_cost = totalCost q2_plan q2_cost) →
  a = 3/2 := by sorry

end construction_problem_l1607_160700


namespace geometry_textbook_weight_l1607_160781

/-- The weight of Kelly's chemistry textbook in pounds -/
def chemistry_weight : ℝ := 7.125

/-- The weight difference between the chemistry and geometry textbooks in pounds -/
def weight_difference : ℝ := 6.5

/-- The weight of Kelly's geometry textbook in pounds -/
def geometry_weight : ℝ := chemistry_weight - weight_difference

theorem geometry_textbook_weight :
  geometry_weight = 0.625 := by sorry

end geometry_textbook_weight_l1607_160781


namespace set_operations_l1607_160752

def A : Set ℝ := {x | 1 < 2*x - 1 ∧ 2*x - 1 < 7}
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem set_operations :
  (A ∩ B = {x | 1 < x ∧ x < 3}) ∧
  (Set.compl (A ∪ B) = {x | x ≤ -1 ∨ x ≥ 4}) := by
  sorry

end set_operations_l1607_160752


namespace f_inequality_l1607_160792

noncomputable def f (x : ℝ) := x^2 - Real.cos x

theorem f_inequality : f 0 < f (-0.5) ∧ f (-0.5) < f 0.6 := by
  sorry

end f_inequality_l1607_160792


namespace sum_of_bn_l1607_160754

theorem sum_of_bn (m : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (∀ n ∈ Finset.range (2 * m + 1), (a n) * (a (n + 1)) = b n) →
  (∀ n ∈ Finset.range (2 * m), (a n) + (a (n + 1)) = -4 * n) →
  a 1 = 0 →
  (Finset.range (2 * m)).sum b = (8 * m / 3) * (4 * m^2 + 3 * m - 1) :=
by sorry

end sum_of_bn_l1607_160754


namespace minimum_point_of_translated_abs_function_l1607_160761

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + 7

-- State the theorem
theorem minimum_point_of_translated_abs_function :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x ≥ f x₀) ∧ f x₀ = 7 ∧ x₀ = 4 :=
sorry

end minimum_point_of_translated_abs_function_l1607_160761


namespace sin_405_degrees_l1607_160760

theorem sin_405_degrees : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_degrees_l1607_160760


namespace tan_cos_expression_equals_negative_one_l1607_160720

theorem tan_cos_expression_equals_negative_one :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end tan_cos_expression_equals_negative_one_l1607_160720


namespace eight_percent_difference_l1607_160770

theorem eight_percent_difference (x y : ℝ) 
  (hx : 8 = 0.25 * x) 
  (hy : 8 = 0.5 * y) : 
  x - y = 16 := by
  sorry

end eight_percent_difference_l1607_160770
