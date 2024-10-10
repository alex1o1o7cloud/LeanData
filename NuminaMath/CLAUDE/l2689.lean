import Mathlib

namespace parabola_translation_l2689_268975

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * dx + p.b
    c := p.a * dx^2 - p.b * dx + p.c - dy }

theorem parabola_translation :
  let original := Parabola.mk 1 0 1  -- y = x^2 + 1
  let translated := translate original 3 (-2)  -- 3 units right, 2 units down
  translated = Parabola.mk 1 (-6) (-1)  -- y = (x - 3)^2 - 1
  := by sorry

end parabola_translation_l2689_268975


namespace ball_transfer_probability_l2689_268939

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  red : ℕ

/-- The probability of drawing a red ball from a bag -/
def redProbability (bag : Bag) : ℚ :=
  bag.red / (bag.white + bag.red)

/-- The probability of drawing a white ball from a bag -/
def whiteProbability (bag : Bag) : ℚ :=
  bag.white / (bag.white + bag.red)

/-- The probability of drawing a red ball from the second bag
    after transferring a ball from the first bag -/
def transferAndDrawRed (bagA bagB : Bag) : ℚ :=
  (redProbability bagA) * (redProbability (Bag.mk bagB.white (bagB.red + 1))) +
  (whiteProbability bagA) * (redProbability (Bag.mk (bagB.white + 1) bagB.red))

theorem ball_transfer_probability :
  let bagA : Bag := ⟨2, 3⟩
  let bagB : Bag := ⟨1, 2⟩
  transferAndDrawRed bagA bagB = 13 / 20 := by
  sorry

end ball_transfer_probability_l2689_268939


namespace bicycle_sales_theorem_l2689_268923

/-- Represents the sales and pricing data for bicycle types A and B -/
structure BicycleSales where
  lastYearTotalSalesA : ℕ
  priceIncreaseA : ℕ
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceB : ℕ
  totalPurchase : ℕ

/-- Calculates the selling price of type A bicycles this year -/
def sellingPriceA (data : BicycleSales) : ℕ :=
  sorry

/-- Calculates the optimal purchase plan to maximize profit -/
def optimalPurchasePlan (data : BicycleSales) : ℕ × ℕ :=
  sorry

/-- Main theorem stating the selling price of type A bicycles and the optimal purchase plan -/
theorem bicycle_sales_theorem (data : BicycleSales) 
  (h1 : data.lastYearTotalSalesA = 32000)
  (h2 : data.priceIncreaseA = 400)
  (h3 : data.purchasePriceA = 1100)
  (h4 : data.purchasePriceB = 1400)
  (h5 : data.sellingPriceB = 2400)
  (h6 : data.totalPurchase = 50)
  (h7 : ∀ (x y : ℕ), x + y = data.totalPurchase → y ≤ 2 * x) :
  sellingPriceA data = 2000 ∧ optimalPurchasePlan data = (17, 33) :=
sorry

end bicycle_sales_theorem_l2689_268923


namespace complex_fraction_simplification_l2689_268993

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I := by
  sorry

end complex_fraction_simplification_l2689_268993


namespace max_renovation_days_l2689_268956

def turnkey_cost : ℕ := 50000
def materials_cost : ℕ := 20000
def husband_wage : ℕ := 2000
def wife_wage : ℕ := 1500

theorem max_renovation_days : 
  ∃ n : ℕ, n = 8 ∧ 
  n * (husband_wage + wife_wage) + materials_cost ≤ turnkey_cost ∧
  (n + 1) * (husband_wage + wife_wage) + materials_cost > turnkey_cost :=
sorry

end max_renovation_days_l2689_268956


namespace equation_solution_l2689_268901

theorem equation_solution (x : ℚ) : 1 / (x + 1/5) = 5/3 → x = 2/5 := by
  sorry

end equation_solution_l2689_268901


namespace wall_volume_theorem_l2689_268945

/-- Calculates the volume of a rectangular wall given its width and height-to-width and length-to-height ratios -/
def wall_volume (width : ℝ) (height_ratio : ℝ) (length_ratio : ℝ) : ℝ :=
  width * (height_ratio * width) * (length_ratio * height_ratio * width)

/-- Theorem: The volume of a wall with width 4m, height 6 times its width, and length 7 times its height is 16128 cubic meters -/
theorem wall_volume_theorem :
  wall_volume 4 6 7 = 16128 := by
  sorry

end wall_volume_theorem_l2689_268945


namespace three_bushes_same_flowers_l2689_268943

theorem three_bushes_same_flowers (garden : Finset ℕ) (flower_count : ℕ → ℕ) :
  garden.card = 201 →
  (∀ bush ∈ garden, 1 ≤ flower_count bush ∧ flower_count bush ≤ 100) →
  ∃ n : ℕ, ∃ bush₁ bush₂ bush₃ : ℕ,
    bush₁ ∈ garden ∧ bush₂ ∈ garden ∧ bush₃ ∈ garden ∧
    bush₁ ≠ bush₂ ∧ bush₁ ≠ bush₃ ∧ bush₂ ≠ bush₃ ∧
    flower_count bush₁ = n ∧ flower_count bush₂ = n ∧ flower_count bush₃ = n :=
by sorry

end three_bushes_same_flowers_l2689_268943


namespace triangle_inequality_l2689_268905

theorem triangle_inequality (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ x^3 + y^3 + z^3 + 6*x*y*z := by
  sorry

end triangle_inequality_l2689_268905


namespace fair_die_probability_at_least_one_six_l2689_268928

theorem fair_die_probability_at_least_one_six (n : ℕ) (p : ℚ) : 
  n = 3 → p = 1/6 → (1 : ℚ) - (1 - p)^n = 91/216 := by
  sorry

end fair_die_probability_at_least_one_six_l2689_268928


namespace square_figure_perimeter_l2689_268951

/-- A figure composed of two rows of three consecutive unit squares, with the top row directly above the bottom row -/
structure SquareFigure where
  /-- The side length of each square -/
  side_length : ℝ
  /-- The number of squares in each row -/
  squares_per_row : ℕ
  /-- The number of rows -/
  rows : ℕ
  /-- The side length is 1 -/
  unit_side : side_length = 1
  /-- There are three squares in each row -/
  three_squares : squares_per_row = 3
  /-- There are two rows -/
  two_rows : rows = 2

/-- The perimeter of the SquareFigure -/
def perimeter (fig : SquareFigure) : ℝ :=
  2 * fig.side_length * fig.squares_per_row + 2 * fig.side_length * fig.rows

/-- Theorem stating that the perimeter of the SquareFigure is 16 -/
theorem square_figure_perimeter (fig : SquareFigure) : perimeter fig = 16 := by
  sorry

end square_figure_perimeter_l2689_268951


namespace u_2008_eq_4008_l2689_268982

/-- Defines the sequence u_n as described in the problem -/
def u : ℕ → ℕ :=
  sorry

/-- Theorem stating that the 2008th term of the sequence is 4008 -/
theorem u_2008_eq_4008 : u 2008 = 4008 := by
  sorry

end u_2008_eq_4008_l2689_268982


namespace percentage_of_number_l2689_268925

theorem percentage_of_number (x : ℚ) (y : ℕ) (z : ℕ) :
  (x / 100) * y = z → x = 33 + 1/3 → y = 210 → z = 70 := by
  sorry

end percentage_of_number_l2689_268925


namespace min_v_value_l2689_268955

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the translated function g
def g (x u v : ℝ) : ℝ := (x-u)^3 - 3*(x-u) - v

-- Theorem statement
theorem min_v_value (u : ℝ) (h : u > 0) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = g x₁ u v → f x₂ ≠ g x₂ u v) →
  v ≥ 4 :=
sorry

end min_v_value_l2689_268955


namespace product_from_hcf_lcm_l2689_268957

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 205) :
  a * b = 2460 := by
  sorry

end product_from_hcf_lcm_l2689_268957


namespace max_bisections_for_zero_approximation_l2689_268992

/-- Theorem: Maximum number of bisections for approximating a zero --/
theorem max_bisections_for_zero_approximation 
  (f : ℝ → ℝ) 
  (zero_exists : ∃ x, x ∈ (Set.Ioo 0 1) ∧ f x = 0) 
  (accuracy : ℝ := 0.01) :
  (∃ n : ℕ, n ≤ 7 ∧ 
    (1 : ℝ) / (2 ^ n) < accuracy ∧ 
    ∀ m : ℕ, m < n → (1 : ℝ) / (2 ^ m) ≥ accuracy) :=
sorry

end max_bisections_for_zero_approximation_l2689_268992


namespace solve_for_y_l2689_268988

theorem solve_for_y (x y : ℤ) (h1 : x - y = 12) (h2 : x + y = 6) : y = -3 := by
  sorry

end solve_for_y_l2689_268988


namespace log_xy_equals_three_fourths_l2689_268994

-- Define x and y as positive real numbers
variable (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Define the given conditions
def condition1 : Prop := Real.log (x^2 * y^4) = 2
def condition2 : Prop := Real.log (x^3 * y^2) = 2

-- State the theorem
theorem log_xy_equals_three_fourths 
  (h1 : condition1 x y) (h2 : condition2 x y) : 
  Real.log (x * y) = 3/4 := by
  sorry

end log_xy_equals_three_fourths_l2689_268994


namespace geometric_sequence_sum_property_l2689_268921

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Sum of three consecutive terms in a sequence -/
def SumOfThree (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n + a (n + 1) + a (n + 2)

theorem geometric_sequence_sum_property
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_sum1 : SumOfThree a 1 = 8)
  (h_sum2 : SumOfThree a 4 = -4) :
  SumOfThree a 7 = 2 := by
sorry

end geometric_sequence_sum_property_l2689_268921


namespace geometric_sequence_product_l2689_268906

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 4 * a 5 * a 6 = 27 →
  a 1 * a 9 = 9 := by
  sorry

end geometric_sequence_product_l2689_268906


namespace inverse_half_plus_sqrt_four_log_sum_minus_power_inverse_sum_with_sqrt_three_l2689_268946

-- Part 1
theorem inverse_half_plus_sqrt_four (x y : ℝ) (h1 : x = 0.5) (h2 : y = 4) :
  x⁻¹ + y^(1/2) = 4 := by sorry

-- Part 2
theorem log_sum_minus_power (x y z : ℝ) (h1 : x = 2) (h2 : y = 5) (h3 : z = π / 23) :
  Real.log x / Real.log 10 + Real.log y / Real.log 10 - z^0 = 0 := by sorry

-- Part 3
theorem inverse_sum_with_sqrt_three (x : ℝ) (h : x = 3) :
  (2 - Real.sqrt x)⁻¹ + (2 + Real.sqrt x)⁻¹ = 4 := by sorry

end inverse_half_plus_sqrt_four_log_sum_minus_power_inverse_sum_with_sqrt_three_l2689_268946


namespace stellas_clocks_l2689_268938

/-- Stella's antique shop inventory problem -/
theorem stellas_clocks :
  ∀ (num_clocks : ℕ),
    (3 * 5 + num_clocks * 15 + 5 * 4 = 40 + 25) →
    num_clocks = 2 :=
by
  sorry

end stellas_clocks_l2689_268938


namespace right_rectangular_prism_volume_l2689_268961

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h_side : a * b = 20) 
  (h_front : b * c = 12) 
  (h_bottom : a * c = 15) : 
  a * b * c = 60 := by
sorry

end right_rectangular_prism_volume_l2689_268961


namespace product_of_square_roots_l2689_268912

theorem product_of_square_roots (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p) * Real.sqrt (10 * p^3) * Real.sqrt (14 * p^5) = 10 * p^4 * Real.sqrt (21 * p) :=
by sorry

end product_of_square_roots_l2689_268912


namespace quadratic_equation_solution_l2689_268922

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x + 5
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 5 := by
sorry

end quadratic_equation_solution_l2689_268922


namespace x_plus_inv_x_eq_five_l2689_268909

theorem x_plus_inv_x_eq_five (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end x_plus_inv_x_eq_five_l2689_268909


namespace meat_for_hamburgers_l2689_268960

/-- Given that 5 pounds of meat make 10 hamburgers, prove that 15 pounds of meat are needed for 30 hamburgers -/
theorem meat_for_hamburgers (meat_per_10 : ℕ) (hamburgers_per_5 : ℕ) 
  (h1 : meat_per_10 = 5) 
  (h2 : hamburgers_per_5 = 10) :
  (meat_per_10 * 3 : ℕ) = 15 ∧ (hamburgers_per_5 * 3 : ℕ) = 30 := by
  sorry

end meat_for_hamburgers_l2689_268960


namespace difference_from_sum_and_difference_of_squares_l2689_268908

theorem difference_from_sum_and_difference_of_squares
  (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  x - y = 4 := by
sorry

end difference_from_sum_and_difference_of_squares_l2689_268908


namespace irrational_in_set_l2689_268942

-- Define the set of numbers
def numbers : Set ℝ := {0, -2, Real.sqrt 3, 1/2}

-- Define a predicate for rational numbers
def isRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Theorem statement
theorem irrational_in_set :
  ∃ (x : ℝ), x ∈ numbers ∧ ¬(isRational x) ∧
  ∀ (y : ℝ), y ∈ numbers ∧ y ≠ x → isRational y :=
sorry

end irrational_in_set_l2689_268942


namespace inverse_of_A_l2689_268952

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem inverse_of_A :
  let inv_A : Matrix (Fin 2) (Fin 2) ℝ := !![-1, -3; -2, -5]
  Matrix.det A ≠ 0 → A * inv_A = 1 ∧ inv_A * A = 1 := by
  sorry

end inverse_of_A_l2689_268952


namespace sum_after_operations_l2689_268979

/-- Given two numbers x and y whose sum is T, prove that if 5 is added to each number
    and then each resulting number is tripled, the sum of the final two numbers is 3T + 30. -/
theorem sum_after_operations (x y T : ℝ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 := by
  sorry

end sum_after_operations_l2689_268979


namespace mean_height_is_70_74_l2689_268966

def player_heights : List ℕ := [58, 59, 60, 61, 62, 63, 65, 65, 68, 70, 71, 74, 76, 78, 79, 81, 83, 85, 86]

def mean_height (heights : List ℕ) : ℚ :=
  (heights.sum : ℚ) / heights.length

theorem mean_height_is_70_74 :
  mean_height player_heights = 70.74 := by
  sorry

end mean_height_is_70_74_l2689_268966


namespace quadratic_function_properties_l2689_268949

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define g(x) in terms of f(x) and m
def g (m : ℝ) (x : ℝ) : ℝ := m * f x + 1

-- Theorem statement
theorem quadratic_function_properties :
  (∀ x : ℝ, f x ≥ -4) ∧
  (f (-2) = -3) ∧
  (f 0 = -3) ∧
  (∀ m : ℝ, m < 0 → ∃! x : ℝ, x ≤ 1 ∧ g m x = 0) ∧
  (∀ m : ℝ, m > 0 →
    (m ≤ 8/7 → (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3/2 → |g m x| ≤ 9*m/4 + 1)) ∧
    (m > 8/7 → (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3/2 → |g m x| ≤ 4*m - 1))) :=
by sorry

end quadratic_function_properties_l2689_268949


namespace trail_mix_peanuts_weight_l2689_268986

theorem trail_mix_peanuts_weight (total_weight chocolate_weight raisin_weight : ℚ)
  (h1 : total_weight = 0.4166666666666667)
  (h2 : chocolate_weight = 0.16666666666666666)
  (h3 : raisin_weight = 0.08333333333333333) :
  total_weight - (chocolate_weight + raisin_weight) = 0.1666666666666667 := by
  sorry

end trail_mix_peanuts_weight_l2689_268986


namespace cos_2B_gt_cos_2A_necessary_not_sufficient_l2689_268999

-- Define a structure for a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the main theorem
theorem cos_2B_gt_cos_2A_necessary_not_sufficient (t : Triangle) :
  (∀ t : Triangle, t.A > t.B → Real.cos (2 * t.B) > Real.cos (2 * t.A)) ∧
  ¬(∀ t : Triangle, Real.cos (2 * t.B) > Real.cos (2 * t.A) → t.A > t.B) := by
  sorry

end cos_2B_gt_cos_2A_necessary_not_sufficient_l2689_268999


namespace xixi_apples_count_l2689_268910

/-- The number of students in Teacher Xixi's class -/
def xixi_students : ℕ := 12

/-- The number of students in Teacher Shanshan's class -/
def shanshan_students : ℕ := xixi_students

/-- The number of apples Teacher Xixi prepared -/
def xixi_apples : ℕ := 72

/-- The number of oranges Teacher Shanshan prepared -/
def shanshan_oranges : ℕ := 60

theorem xixi_apples_count : xixi_apples = 72 := by
  have h1 : xixi_apples = shanshan_students * 6 := sorry
  have h2 : shanshan_oranges = xixi_students * 3 + 12 := sorry
  have h3 : shanshan_oranges = shanshan_students * 5 := sorry
  sorry

end xixi_apples_count_l2689_268910


namespace delta_y_over_delta_x_l2689_268918

/-- The function f(x) = 2x^2 + 5 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 5

/-- Theorem stating that for the given function and points, Δy / Δx = 2Δx + 4 -/
theorem delta_y_over_delta_x (Δx : ℝ) (Δy : ℝ) :
  f 1 = 7 →
  f (1 + Δx) = 7 + Δy →
  Δy / Δx = 2 * Δx + 4 :=
by
  sorry

end delta_y_over_delta_x_l2689_268918


namespace tetrahedron_dihedral_angle_l2689_268950

/-- Regular tetrahedron with given dimensions -/
structure RegularTetrahedron where
  base_side_length : ℝ
  side_edge_length : ℝ

/-- Plane that divides the tetrahedron's volume equally -/
structure DividingPlane where
  tetrahedron : RegularTetrahedron
  passes_through_AB : Bool
  divides_volume_equally : Bool

/-- The cosine of the dihedral angle between the dividing plane and the base -/
def dihedral_angle_cosine (plane : DividingPlane) : ℝ :=
  sorry

theorem tetrahedron_dihedral_angle 
  (t : RegularTetrahedron) 
  (p : DividingPlane) 
  (h1 : t.base_side_length = 1) 
  (h2 : t.side_edge_length = 2) 
  (h3 : p.tetrahedron = t) 
  (h4 : p.passes_through_AB = true) 
  (h5 : p.divides_volume_equally = true) : 
  dihedral_angle_cosine p = 2 * Real.sqrt 15 / 15 :=
sorry

end tetrahedron_dihedral_angle_l2689_268950


namespace giraffe_count_l2689_268976

/-- The number of giraffes in the zoo -/
def num_giraffes : ℕ := 5

/-- The number of penguins in the zoo -/
def num_penguins : ℕ := 10

/-- The number of elephants in the zoo -/
def num_elephants : ℕ := 2

/-- The total number of animals in the zoo -/
def total_animals : ℕ := 50

theorem giraffe_count :
  (num_penguins = 2 * num_giraffes) ∧
  (num_penguins = (20 : ℕ) * total_animals / 100) ∧
  (num_elephants = (4 : ℕ) * total_animals / 100) ∧
  (num_elephants = 2) →
  num_giraffes = 5 := by
sorry

end giraffe_count_l2689_268976


namespace rectangle_perimeter_l2689_268947

/-- The perimeter of a rectangle with length 6 cm and width 4 cm is 20 cm. -/
theorem rectangle_perimeter : 
  let length : ℝ := 6
  let width : ℝ := 4
  let perimeter := 2 * (length + width)
  perimeter = 20 := by
  sorry

end rectangle_perimeter_l2689_268947


namespace trees_in_park_l2689_268934

/-- The number of trees after n years, given an initial number and annual growth rate. -/
def trees_after_years (initial : ℕ) (growth_rate : ℚ) (years : ℕ) : ℚ :=
  initial * (1 + growth_rate) ^ years

/-- Theorem stating that given 5000 trees initially and 30% annual growth,
    the number of trees after 3 years is 10985. -/
theorem trees_in_park (initial : ℕ) (growth_rate : ℚ) (years : ℕ) 
  (h_initial : initial = 5000)
  (h_growth : growth_rate = 3/10)
  (h_years : years = 3) :
  trees_after_years initial growth_rate years = 10985 := by
  sorry

#eval trees_after_years 5000 (3/10) 3

end trees_in_park_l2689_268934


namespace rational_inequality_equivalence_l2689_268996

theorem rational_inequality_equivalence (x : ℝ) :
  (2 * x - 1) / (x + 1) > 1 ↔ x < -1 ∨ x > 2 := by
  sorry

end rational_inequality_equivalence_l2689_268996


namespace sin_cos_equation_solution_l2689_268920

theorem sin_cos_equation_solution (x y : ℝ) : 
  (Real.sin (x + y))^2 - (Real.cos (x - y))^2 = 1 ↔ 
  (∃ (k l : ℤ), x = Real.pi / 2 * (2 * k + l + 1) ∧ y = Real.pi / 2 * (2 * k - l)) ∨
  (∃ (m n : ℤ), x = Real.pi / 2 * (2 * m + n) ∧ y = Real.pi / 2 * (2 * m - n - 1)) :=
by sorry

end sin_cos_equation_solution_l2689_268920


namespace inequality_proof_l2689_268948

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (3*x^2 - x)/(1 + x^2) + (3*y^2 - y)/(1 + y^2) + (3*z^2 - z)/(1 + z^2) ≥ 0 := by
  sorry

end inequality_proof_l2689_268948


namespace eliza_says_500_l2689_268968

-- Define the upper bound of the counting range
def upper_bound : ℕ := 500

-- Define the skipping pattern for each student
def alice_skip (n : ℕ) : Bool := n % 4 = 0
def barbara_skip (n : ℕ) : Bool := n % 12 = 4
def candice_skip (n : ℕ) : Bool := n % 16 = 0
def debbie_skip (n : ℕ) : Bool := n % 64 = 0

-- Define a function to check if a number is said by any of the first four students
def is_said_by_first_four (n : ℕ) : Bool :=
  ¬(alice_skip n) ∨ ¬(barbara_skip n) ∨ ¬(candice_skip n) ∨ ¬(debbie_skip n)

-- Theorem statement
theorem eliza_says_500 : 
  ∀ n : ℕ, n ≤ upper_bound → (n ≠ upper_bound → is_said_by_first_four n) ∧ ¬(is_said_by_first_four upper_bound) :=
by sorry

end eliza_says_500_l2689_268968


namespace intersection_of_A_and_B_l2689_268980

-- Define set A
def A : Set ℝ := {x | 2 * x ≤ 4}

-- Define set B (domain of lg(x-1))
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 2 := by
  sorry

end intersection_of_A_and_B_l2689_268980


namespace part_one_part_two_l2689_268902

-- Part 1
theorem part_one (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x > 0, f x = x - a * Real.log x)
  (h1 : ∀ x > 0, f x ≥ 1) : a = 1 := by
  sorry

-- Part 2
theorem part_two (x₁ x₂ : ℝ) 
  (h1 : x₁ > 0)
  (h2 : x₂ > 0)
  (h3 : Real.exp x₁ + Real.log x₂ > x₁ + x₂) :
  Real.exp x₁ + x₂ > 2 := by
  sorry

end part_one_part_two_l2689_268902


namespace first_player_wins_n_9_first_player_wins_n_10_l2689_268915

/-- Represents the state of the game board -/
inductive BoardState
  | Minuses (n : ℕ)
  | Pluses (n : ℕ)

/-- Represents a move in the game -/
inductive Move
  | ChangeOne
  | ChangeTwo

/-- Defines the game rules and winning condition -/
def gameRules (n : ℕ) (player : ℕ) (board : BoardState) (move : Move) : Prop :=
  match board with
  | BoardState.Minuses m =>
      (move = Move.ChangeOne ∧ m > 0) ∨
      (move = Move.ChangeTwo ∧ m > 1)
  | BoardState.Pluses _ => false

/-- Defines the winning condition -/
def isWinningState (board : BoardState) : Prop :=
  match board with
  | BoardState.Minuses 0 => true
  | _ => false

/-- Theorem: The first player has a winning strategy for n = 9 -/
theorem first_player_wins_n_9 :
  ∃ (strategy : ℕ → BoardState → Move),
    ∀ (opponent_strategy : ℕ → BoardState → Move),
      isWinningState (BoardState.Minuses 0) ∧
      (∀ (t : ℕ),
        gameRules 9 (t % 2) (BoardState.Minuses (9 - t)) (strategy t (BoardState.Minuses (9 - t)))) :=
sorry

/-- Theorem: The first player has a winning strategy for n = 10 -/
theorem first_player_wins_n_10 :
  ∃ (strategy : ℕ → BoardState → Move),
    ∀ (opponent_strategy : ℕ → BoardState → Move),
      isWinningState (BoardState.Minuses 0) ∧
      (∀ (t : ℕ),
        gameRules 10 (t % 2) (BoardState.Minuses (10 - t)) (strategy t (BoardState.Minuses (10 - t)))) :=
sorry

end first_player_wins_n_9_first_player_wins_n_10_l2689_268915


namespace rationalize_denominator_l2689_268936

theorem rationalize_denominator :
  ∀ (x : ℝ), x > 0 → (5 / (x^(1/3) + (27 * x)^(1/3))) = (5 * (9 * x)^(1/3)) / 12 :=
by sorry

end rationalize_denominator_l2689_268936


namespace range_of_a_l2689_268964

theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - 8*x - 20 ≤ 0 → x^2 - 2*x + 1 - a^2 ≤ 0) ∧ 
  (∃ x, x^2 - 8*x - 20 ≤ 0 ∧ x^2 - 2*x + 1 - a^2 > 0) ∧
  a > 0 → 
  a ≥ 9 := by sorry

end range_of_a_l2689_268964


namespace product_remainder_by_10_l2689_268944

theorem product_remainder_by_10 : (2583 * 7462 * 93215) % 10 = 0 := by
  sorry

end product_remainder_by_10_l2689_268944


namespace july_birth_percentage_l2689_268998

theorem july_birth_percentage (total : ℕ) (july_births : ℕ) 
  (h1 : total = 120) (h2 : july_births = 16) : 
  (july_births : ℝ) / total * 100 = 13.33 := by
  sorry

end july_birth_percentage_l2689_268998


namespace expression_evaluation_l2689_268963

theorem expression_evaluation : (36 + 12) / (6 - (2 + 1)) = 16 := by
  sorry

end expression_evaluation_l2689_268963


namespace number_puzzle_solution_l2689_268981

theorem number_puzzle_solution :
  ∃ (A B C D E : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    A > 5 ∧
    A % B = 0 ∧
    C + A = D ∧
    B + C + E = A ∧
    B + C < E ∧
    C + E < B + 5 ∧
    A = 8 ∧ B = 2 ∧ C = 1 ∧ D = 9 ∧ E = 5 :=
by sorry

end number_puzzle_solution_l2689_268981


namespace center_coordinates_sum_l2689_268959

/-- Given two points as the endpoints of a diameter of a circle, 
    prove that the sum of the coordinates of the center is 0. -/
theorem center_coordinates_sum (x₁ y₁ x₂ y₂ : ℝ) 
  (h : x₁ = 9 ∧ y₁ = -5 ∧ x₂ = -3 ∧ y₂ = -1) : 
  ((x₁ + x₂) / 2) + ((y₁ + y₂) / 2) = 0 :=
by sorry

end center_coordinates_sum_l2689_268959


namespace hall_length_breadth_difference_l2689_268919

/-- Represents a rectangular hall -/
structure RectangularHall where
  length : ℝ
  breadth : ℝ
  area : ℝ

/-- Theorem: For a rectangular hall with area 750 m² and length 30 m, 
    the difference between length and breadth is 5 m -/
theorem hall_length_breadth_difference 
  (hall : RectangularHall) 
  (h1 : hall.area = 750) 
  (h2 : hall.length = 30) : 
  hall.length - hall.breadth = 5 := by
  sorry


end hall_length_breadth_difference_l2689_268919


namespace parallel_to_plane_not_always_parallel_l2689_268935

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_to_plane_not_always_parallel :
  ∃ (m n : Line) (α : Plane),
    parallelLinePlane m α ∧ parallelLinePlane n α ∧ ¬ parallelLine m n := by
  sorry

end parallel_to_plane_not_always_parallel_l2689_268935


namespace discriminant_of_polynomial_l2689_268969

/-- The discriminant of a quadratic polynomial ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic polynomial 5x^2 + (5 + 1/5)x + 1/5 -/
def polynomial (x : ℚ) : ℚ := 5*x^2 + (5 + 1/5)*x + 1/5

theorem discriminant_of_polynomial :
  discriminant 5 (5 + 1/5) (1/5) = 576/25 := by
  sorry

end discriminant_of_polynomial_l2689_268969


namespace arithmetic_mean_of_S_l2689_268926

def S : Finset ℕ := {8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888}

def arithmetic_mean (s : Finset ℕ) : ℚ :=
  (s.sum id) / s.card

def digits (n : ℕ) : Finset ℕ :=
  sorry

theorem arithmetic_mean_of_S :
  arithmetic_mean S = 109728268 ∧
  ∀ d : ℕ, d < 10 → (d ∉ digits 109728268 ↔ d = 4) := by
  sorry

end arithmetic_mean_of_S_l2689_268926


namespace solve_for_z_l2689_268973

theorem solve_for_z (x z : ℝ) 
  (h1 : x = 102) 
  (h2 : x^4*z - 3*x^3*z + 2*x^2*z = 1075648000) : 
  z = 1.024 := by
  sorry

end solve_for_z_l2689_268973


namespace tire_circumference_l2689_268978

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference (rotation_speed : ℝ) (car_velocity : ℝ) : 
  rotation_speed = 400 ∧ car_velocity = 96 → 
  (car_velocity * 1000 / 60) / rotation_speed = 4 := by
  sorry

end tire_circumference_l2689_268978


namespace complex_intersection_l2689_268940

theorem complex_intersection (z : ℂ) (k : ℝ) : 
  k > 0 → 
  Complex.abs (z - 4) = 3 * Complex.abs (z + 4) →
  Complex.abs z = k →
  (∃! z', Complex.abs (z' - 4) = 3 * Complex.abs (z' + 4) ∧ Complex.abs z' = k) →
  k = 4 ∨ k = 14 := by
sorry

end complex_intersection_l2689_268940


namespace fraction_difference_equals_specific_fraction_l2689_268990

theorem fraction_difference_equals_specific_fraction : 
  (3^2 + 5^2 + 7^2) / (2^2 + 4^2 + 6^2) - (2^2 + 4^2 + 6^2) / (3^2 + 5^2 + 7^2) = 3753 / 4656 := by
  sorry

end fraction_difference_equals_specific_fraction_l2689_268990


namespace quadratic_inequality_range_l2689_268932

def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  (a - 1) * x^2 + 2 * (a - 1) * x - 4 ≥ 0

def empty_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(quadratic_inequality a x)

theorem quadratic_inequality_range :
  ∀ a : ℝ, empty_solution_set a ↔ -3 < a ∧ a ≤ 1 :=
by sorry

end quadratic_inequality_range_l2689_268932


namespace digit_sum_eleven_l2689_268916

/-- Represents a digit in base 10 -/
def Digit := Fin 10

/-- Defines a two-digit number -/
def TwoDigitNumber (a b : Digit) : ℕ := 10 * a.val + b.val

/-- Defines a three-digit number -/
def ThreeDigitNumber (c d e : Digit) : ℕ := 100 * c.val + 10 * d.val + e.val

/-- Checks if three digits are consecutive and increasing -/
def ConsecutiveIncreasing (c d e : Digit) : Prop :=
  d.val = c.val + 1 ∧ e.val = d.val + 1

theorem digit_sum_eleven 
  (a b c d e : Digit) 
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h2 : TwoDigitNumber a b * TwoDigitNumber c b = ThreeDigitNumber c d e)
  (h3 : ConsecutiveIncreasing c d e) :
  a.val + b.val + c.val + d.val + e.val = 11 := by
  sorry

end digit_sum_eleven_l2689_268916


namespace uncoverable_3x7_and_7x3_other_boards_coverable_l2689_268991

/-- A board configuration -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)
  (removed : ℕ)

/-- Checks if a board can be completely covered by dominoes -/
def can_cover (b : Board) : Prop :=
  (b.rows * b.cols - b.removed) % 2 = 0

/-- Theorem: A 3x7 or 7x3 board cannot be completely covered by dominoes -/
theorem uncoverable_3x7_and_7x3 :
  ¬(can_cover ⟨3, 7, 0⟩) ∧ ¬(can_cover ⟨7, 3, 0⟩) :=
sorry

/-- Theorem: All other given board configurations can be covered by dominoes -/
theorem other_boards_coverable :
  can_cover ⟨2, 3, 0⟩ ∧
  can_cover ⟨4, 4, 4⟩ ∧
  can_cover ⟨5, 5, 1⟩ :=
sorry

end uncoverable_3x7_and_7x3_other_boards_coverable_l2689_268991


namespace six_year_olds_count_l2689_268984

/-- Represents the number of children in each age group -/
structure AgeGroups where
  three_year_olds : ℕ
  four_year_olds : ℕ
  five_year_olds : ℕ
  six_year_olds : ℕ

/-- Represents the Sunday school with its age groups and class information -/
structure SundaySchool where
  ages : AgeGroups
  avg_class_size : ℕ
  num_classes : ℕ

def SundaySchool.total_children (s : SundaySchool) : ℕ :=
  s.ages.three_year_olds + s.ages.four_year_olds + s.ages.five_year_olds + s.ages.six_year_olds

theorem six_year_olds_count (s : SundaySchool) 
  (h1 : s.ages.three_year_olds = 13)
  (h2 : s.ages.four_year_olds = 20)
  (h3 : s.ages.five_year_olds = 15)
  (h4 : s.avg_class_size = 35)
  (h5 : s.num_classes = 2)
  : s.ages.six_year_olds = 22 := by
  sorry

#check six_year_olds_count

end six_year_olds_count_l2689_268984


namespace arithmetic_sequence_common_difference_l2689_268931

/-- Given an arithmetic sequence {a_n} where a_2 = 10 and a_4 = 18, 
    the common difference d equals 4. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- a is a sequence of real numbers indexed by natural numbers
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- a is arithmetic
  (h_a2 : a 2 = 10) -- a_2 = 10
  (h_a4 : a 4 = 18) -- a_4 = 18
  : a 3 - a 2 = 4 := by
sorry

end arithmetic_sequence_common_difference_l2689_268931


namespace ellipse_locus_and_intercept_range_l2689_268913

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point B
def B : ℝ × ℝ := (0, 1)

-- Define the perpendicularity condition
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  (P.2 - B.2) * (Q.2 - B.2) = -(P.1 - B.1) * (Q.1 - B.1)

-- Define the projection M
def M (P Q : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the perpendicular bisector l
def l (P Q : ℝ × ℝ) : ℝ → ℝ := sorry

-- Define the x-intercept of l
def x_intercept (P Q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_locus_and_intercept_range :
  ∀ (P Q : ℝ × ℝ),
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  P ≠ B →
  Q ≠ B →
  perpendicular P Q →
  (∀ (x y : ℝ), 
    (x, y) = M P Q →
    y ≠ 1 →
    x^2 + (y - 1/5)^2 = (4/5)^2) ∧
  (-9/20 ≤ x_intercept P Q ∧ x_intercept P Q ≤ 9/20) :=
sorry

end ellipse_locus_and_intercept_range_l2689_268913


namespace axis_of_symmetry_l2689_268907

-- Define the parabola
def parabola (x : ℝ) : ℝ := (2 - x) * x

-- State the theorem
theorem axis_of_symmetry :
  (∀ x : ℝ, parabola (1 + x) = parabola (1 - x)) ∧
  (∀ a : ℝ, a ≠ 1 → ∃ x : ℝ, parabola (a + x) ≠ parabola (a - x)) :=
by sorry

end axis_of_symmetry_l2689_268907


namespace fourth_root_of_1250000_l2689_268933

theorem fourth_root_of_1250000 : (1250000 : ℝ) ^ (1/4 : ℝ) = 100 := by
  sorry

end fourth_root_of_1250000_l2689_268933


namespace common_tangent_sum_l2689_268941

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = 2 * x^2 + 125 / 100

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = 2 * y^2 + 65 / 4

/-- Common tangent line L -/
def L (x y a b c : ℝ) : Prop := a * x + b * y = c

/-- The slope of L is rational -/
def rational_slope (a b : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ a / b = p / q

theorem common_tangent_sum (a b c : ℕ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧
    L x₁ y₁ a b c ∧ L x₂ y₂ a b c ∧
    rational_slope a b ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.gcd a (Nat.gcd b c) = 1) →
  a + b + c = 289 := by
  sorry

end common_tangent_sum_l2689_268941


namespace first_player_winning_strategy_l2689_268958

/-- Represents the state of the game -/
structure GameState where
  score : ℕ
  remainingCards : List ℕ

/-- Defines a valid move in the game -/
def validMove (state : GameState) (card : ℕ) : Prop :=
  card ∈ state.remainingCards ∧ card ≥ 1 ∧ card ≤ 4

/-- Defines the winning condition -/
def isWinningMove (state : GameState) (card : ℕ) : Prop :=
  validMove state card ∧ (state.score + card = 22 ∨ state.score + card > 22)

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_winning_strategy :
  ∃ (initialCards : List ℕ),
    (initialCards.length = 16) ∧
    (∀ c ∈ initialCards, c ≥ 1 ∧ c ≤ 4) ∧
    (∃ (strategy : GameState → ℕ),
      ∀ (opponentStrategy : GameState → ℕ),
        let initialState : GameState := { score := 0, remainingCards := initialCards }
        let firstMove := strategy initialState
        validMove initialState firstMove ∧ firstMove = 1 →
        ∃ (finalState : GameState),
          isWinningMove finalState (strategy finalState)) :=
sorry

end first_player_winning_strategy_l2689_268958


namespace hulk_jump_exceeds_20000_l2689_268914

def hulk_jump (n : ℕ) : ℝ := 3 * (3 ^ (n - 1))

theorem hulk_jump_exceeds_20000 :
  (∀ m : ℕ, m < 10 → hulk_jump m ≤ 20000) ∧
  hulk_jump 10 > 20000 := by
sorry

end hulk_jump_exceeds_20000_l2689_268914


namespace bearing_and_ring_problem_l2689_268971

theorem bearing_and_ring_problem :
  ∃ (x y : ℕ),
    (x = 25 ∧ y = 16) ∨ (x = 16 ∧ y = 25) ∧
    (x : ℤ) + 2 = y ∧
    (y : ℤ) = x + 2 ∧
    x * ((y : ℤ) - 2) + y * (x + 2) - 800 = 2 * (y - x) ∧
    x * x + y * y = 881 :=
by sorry

end bearing_and_ring_problem_l2689_268971


namespace permutation_solution_l2689_268937

def is_valid_permutation (a : Fin 9 → ℕ) : Prop :=
  (∀ i j : Fin 9, i ≠ j → a i ≠ a j) ∧
  (∀ i : Fin 9, a i ∈ (Set.range (fun i : Fin 9 => i.val + 1)))

def satisfies_conditions (a : Fin 9 → ℕ) : Prop :=
  (a 0 + a 1 + a 2 + a 3 = a 3 + a 4 + a 5 + a 6) ∧
  (a 3 + a 4 + a 5 + a 6 = a 6 + a 7 + a 8 + a 0) ∧
  (a 0^2 + a 1^2 + a 2^2 + a 3^2 = a 3^2 + a 4^2 + a 5^2 + a 6^2) ∧
  (a 3^2 + a 4^2 + a 5^2 + a 6^2 = a 6^2 + a 7^2 + a 8^2 + a 0^2)

def solution : Fin 9 → ℕ := fun i =>
  match i with
  | ⟨0, _⟩ => 2
  | ⟨1, _⟩ => 4
  | ⟨2, _⟩ => 9
  | ⟨3, _⟩ => 5
  | ⟨4, _⟩ => 1
  | ⟨5, _⟩ => 6
  | ⟨6, _⟩ => 8
  | ⟨7, _⟩ => 3
  | ⟨8, _⟩ => 7

theorem permutation_solution :
  is_valid_permutation solution ∧ satisfies_conditions solution :=
by sorry

end permutation_solution_l2689_268937


namespace distance_between_runners_l2689_268903

-- Define the race length in kilometers
def race_length_km : ℝ := 1

-- Define Arianna's position in meters when Ethan finished
def arianna_position : ℝ := 184

-- Theorem to prove the distance between Ethan and Arianna
theorem distance_between_runners : 
  (race_length_km * 1000) - arianna_position = 816 := by
  sorry

end distance_between_runners_l2689_268903


namespace fraction_equality_l2689_268974

theorem fraction_equality (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a / b + b / a = 4) :
  (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end fraction_equality_l2689_268974


namespace red_ball_probability_l2689_268967

-- Define the containers and their contents
structure Container where
  red : ℕ
  green : ℕ

def containerA : Container := ⟨10, 5⟩
def containerB : Container := ⟨6, 6⟩
def containerC : Container := ⟨3, 9⟩
def containerD : Container := ⟨4, 8⟩

-- Define the list of containers
def containers : List Container := [containerA, containerB, containerC, containerD]

-- Function to calculate the probability of selecting a red ball from a container
def redProbability (c : Container) : ℚ :=
  c.red / (c.red + c.green)

-- Theorem stating the probability of selecting a red ball
theorem red_ball_probability :
  (1 / (containers.length : ℚ)) * (containers.map redProbability).sum = 25 / 48 := by
  sorry

end red_ball_probability_l2689_268967


namespace coin_sequence_count_l2689_268900

/-- Represents a coin toss sequence -/
def CoinSequence := List Bool

/-- Counts the number of specific subsequences in a coin sequence -/
def countSubsequences (seq : CoinSequence) : Nat × Nat × Nat × Nat :=
  sorry

/-- Checks if a coin sequence has the required number of subsequences -/
def hasRequiredSubsequences (seq : CoinSequence) : Bool :=
  let (hh, ht, th, tt) := countSubsequences seq
  hh = 3 ∧ ht = 2 ∧ th = 5 ∧ tt = 6

/-- Generates all possible 17-toss coin sequences -/
def allSequences : List CoinSequence :=
  sorry

/-- Counts the number of sequences with required subsequences -/
def countValidSequences : Nat :=
  (allSequences.filter hasRequiredSubsequences).length

theorem coin_sequence_count : countValidSequences = 840 := by
  sorry

end coin_sequence_count_l2689_268900


namespace geometric_sequence_problem_l2689_268917

/-- Given a geometric sequence {a_n} with positive terms, prove that if a_6 + a_5 = 4 
    and a_4 + a_3 - a_2 - a_1 = 1, then a_1 = √2 - 1 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence definition
  a 6 + a 5 = 4 →  -- First given equation
  a 4 + a 3 - a 2 - a 1 = 1 →  -- Second given equation
  a 1 = Real.sqrt 2 - 1 := by
sorry


end geometric_sequence_problem_l2689_268917


namespace count_sevens_to_2017_l2689_268987

/-- Count of occurrences of a digit in a range of natural numbers -/
def countDigitOccurrences (digit : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The main theorem stating that the count of 7's from 1 to 2017 is 602 -/
theorem count_sevens_to_2017 : countDigitOccurrences 7 1 2017 = 602 := by
  sorry

end count_sevens_to_2017_l2689_268987


namespace intersecting_chords_theorem_chord_intersection_equality_l2689_268929

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane. -/
def Point := ℝ × ℝ

/-- The distance between two points. -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a point lies on a circle. -/
def onCircle (c : Circle) (p : Point) : Prop := 
  distance c.center p = c.radius

/-- Represents a chord of a circle. -/
structure Chord (c : Circle) where
  p1 : Point
  p2 : Point
  h1 : onCircle c p1
  h2 : onCircle c p2

/-- Theorem: For two intersecting chords and a line through their intersection point,
    the product of the distances from the intersection point to the endpoints of one chord
    is equal to the product of the distances from the intersection point to the endpoints of the other chord. -/
theorem intersecting_chords_theorem (c : Circle) (ab cd : Chord c) (e f g h i : Point) : 
  onCircle c f ∧ onCircle c g ∧ onCircle c h ∧ onCircle c i →
  distance e f * distance e g = distance e h * distance e i :=
sorry

/-- Main theorem to prove -/
theorem chord_intersection_equality (c : Circle) (ab cd : Chord c) (e f g h i : Point) : 
  onCircle c f ∧ onCircle c g ∧ onCircle c h ∧ onCircle c i →
  distance f g = distance h i :=
sorry

end intersecting_chords_theorem_chord_intersection_equality_l2689_268929


namespace p_div_q_eq_7371_l2689_268985

/-- The number of balls -/
def n : ℕ := 30

/-- The number of bins -/
def k : ℕ := 10

/-- The probability that two bins have 2 balls each and eight bins have 3 balls each -/
noncomputable def p : ℝ := (Nat.choose k 2) * (Nat.choose n 2) * (Nat.choose (n - 2) 2) * 
  (Nat.choose (n - 4) 3) * (Nat.choose (n - 7) 3) * (Nat.choose (n - 10) 3) * 
  (Nat.choose (n - 13) 3) * (Nat.choose (n - 16) 3) * (Nat.choose (n - 19) 3) * 
  (Nat.choose (n - 22) 3) * (Nat.choose (n - 25) 3) / (k ^ n)

/-- The probability that every bin has 3 balls -/
noncomputable def q : ℝ := (Nat.choose n 3) * (Nat.choose (n - 3) 3) * (Nat.choose (n - 6) 3) * 
  (Nat.choose (n - 9) 3) * (Nat.choose (n - 12) 3) * (Nat.choose (n - 15) 3) * 
  (Nat.choose (n - 18) 3) * (Nat.choose (n - 21) 3) * (Nat.choose (n - 24) 3) * 
  (Nat.choose (n - 27) 3) / (k ^ n)

/-- The theorem stating that the ratio of p to q is 7371 -/
theorem p_div_q_eq_7371 : p / q = 7371 := by
  sorry

end p_div_q_eq_7371_l2689_268985


namespace jackson_keeps_120_lollipops_l2689_268953

/-- The number of lollipops Jackson keeps for himself -/
def lollipops_kept (apple banana cherry dragon_fruit : ℕ) (friends : ℕ) : ℕ :=
  cherry

/-- Theorem stating that Jackson keeps 120 lollipops for himself -/
theorem jackson_keeps_120_lollipops :
  lollipops_kept 53 62 120 15 13 = 120 := by
  sorry

#eval lollipops_kept 53 62 120 15 13

end jackson_keeps_120_lollipops_l2689_268953


namespace sum_O_eq_1000_l2689_268911

/-- O(n) is the sum of the odd digits of n -/
def O (n : ℕ) : ℕ := sorry

/-- The sum of O(n) for n from 1 to 200 -/
def sum_O : ℕ := (Finset.range 200).sum (fun i => O (i + 1))

/-- Theorem: The sum of O(n) for n from 1 to 200 is equal to 1000 -/
theorem sum_O_eq_1000 : sum_O = 1000 := by sorry

end sum_O_eq_1000_l2689_268911


namespace expression_simplification_and_evaluation_l2689_268930

theorem expression_simplification_and_evaluation :
  ∀ a : ℤ, -1 < a ∧ a < Real.sqrt 5 ∧ a ≠ 0 ∧ a ≠ 1 →
  let expr := ((a + 1) / (2 * a - 2) - 5 / (2 * a^2 - 2) - (a + 3) / (2 * a + 2)) / (a^2 / (a^2 - 1))
  expr = -1 / (2 * a^2) ∧
  (a = 2 → expr = -1/8) :=
by sorry

end expression_simplification_and_evaluation_l2689_268930


namespace jims_paycheck_l2689_268977

def gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def tax_deduction : ℝ := 100

def retirement_deduction : ℝ := gross_pay * retirement_rate

def net_pay : ℝ := gross_pay - retirement_deduction - tax_deduction

theorem jims_paycheck : net_pay = 740 := by
  sorry

end jims_paycheck_l2689_268977


namespace thomas_score_l2689_268983

theorem thomas_score (n : ℕ) (avg_without_thomas avg_with_thomas thomas_score : ℚ) :
  n = 20 →
  avg_without_thomas = 78 →
  avg_with_thomas = 80 →
  (n - 1) * avg_without_thomas + thomas_score = n * avg_with_thomas →
  thomas_score = 118 := by
sorry

end thomas_score_l2689_268983


namespace optimal_triangle_count_l2689_268970

/-- A configuration of points in space --/
structure PointConfiguration where
  total_points : Nat
  num_groups : Nat
  group_sizes : Fin num_groups → Nat
  non_collinear : Bool
  different_sizes : ∀ i j, i ≠ j → group_sizes i ≠ group_sizes j

/-- The number of triangles formed by selecting one point from each of three different groups --/
def num_triangles (config : PointConfiguration) : Nat :=
  sorry

/-- The optimal configuration for maximizing the number of triangles --/
def optimal_config : PointConfiguration where
  total_points := 1989
  num_groups := 30
  group_sizes := fun i => 
    if i.val < 6 then 51 + i.val
    else if i.val = 6 then 58
    else 59 + i.val - 7
  non_collinear := true
  different_sizes := sorry

theorem optimal_triangle_count (config : PointConfiguration) :
  config.total_points = 1989 →
  config.num_groups = 30 →
  config.non_collinear = true →
  num_triangles config ≤ num_triangles optimal_config :=
sorry

end optimal_triangle_count_l2689_268970


namespace no_eulerian_path_in_problem_graph_l2689_268997

/-- A region in the planar graph --/
structure Region where
  edges : ℕ

/-- A planar graph representation --/
structure PlanarGraph where
  regions : List Region
  total_edges : ℕ

/-- Check if a planar graph has an Eulerian path --/
def has_eulerian_path (g : PlanarGraph) : Prop :=
  (g.regions.filter (λ r => r.edges % 2 = 1)).length ≤ 2

/-- The specific planar graph from the problem --/
def problem_graph : PlanarGraph :=
  { regions := [
      { edges := 5 },
      { edges := 5 },
      { edges := 4 },
      { edges := 5 },
      { edges := 4 },
      { edges := 4 },
      { edges := 4 }
    ],
    total_edges := 16
  }

theorem no_eulerian_path_in_problem_graph :
  ¬ (has_eulerian_path problem_graph) :=
by sorry

end no_eulerian_path_in_problem_graph_l2689_268997


namespace inequality_proof_l2689_268927

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / 2 + (a + b) / 4 ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end inequality_proof_l2689_268927


namespace opposite_of_2023_l2689_268904

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end opposite_of_2023_l2689_268904


namespace max_value_of_e_l2689_268924

def b (n : ℕ) : ℤ := (10^n - 1) / 7

def e (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 2)))

theorem max_value_of_e : ∀ n : ℕ, e n ≤ 99 ∧ ∃ m : ℕ, e m = 99 :=
sorry

end max_value_of_e_l2689_268924


namespace checkerboard_achievable_l2689_268995

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Color

/-- Initial grid configuration -/
def initial_grid : Grid :=
  λ i j => if j.val < 2 then Color.Black else Color.White

/-- Checkerboard pattern grid -/
def checkerboard : Grid :=
  λ i j => if (i.val + j.val) % 2 = 0 then Color.White else Color.Black

/-- Represents a rectangular subgrid -/
structure Rectangle where
  top_left : Fin 4 × Fin 4
  bottom_right : Fin 4 × Fin 4

/-- Toggles the color of cells within a rectangle -/
def toggle_rectangle (g : Grid) (r : Rectangle) : Grid :=
  λ i j => if i.val ≥ r.top_left.1.val && i.val ≤ r.bottom_right.1.val &&
             j.val ≥ r.top_left.2.val && j.val ≤ r.bottom_right.2.val
           then
             match g i j with
             | Color.Black => Color.White
             | Color.White => Color.Black
           else g i j

/-- Theorem stating that the checkerboard pattern is achievable in three operations -/
theorem checkerboard_achievable :
  ∃ (r1 r2 r3 : Rectangle),
    toggle_rectangle (toggle_rectangle (toggle_rectangle initial_grid r1) r2) r3 = checkerboard :=
  sorry

end checkerboard_achievable_l2689_268995


namespace tiffany_album_distribution_l2689_268989

/-- Calculates the number of pictures in each album given the total number of pictures and the number of albums. -/
def pictures_per_album (phone_pics camera_pics num_albums : ℕ) : ℕ :=
  (phone_pics + camera_pics) / num_albums

/-- Proves that given the conditions in the problem, the number of pictures in each album is 4. -/
theorem tiffany_album_distribution :
  let phone_pics := 7
  let camera_pics := 13
  let num_albums := 5
  pictures_per_album phone_pics camera_pics num_albums = 4 := by
sorry

#eval pictures_per_album 7 13 5

end tiffany_album_distribution_l2689_268989


namespace reflection_of_D_l2689_268962

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)  -- Translate down by 1
  let p'' := (p'.2, p'.1)   -- Reflect over y = x
  (p''.1, p''.2 + 1)        -- Translate up by 1

def D : ℝ × ℝ := (4, 1)

theorem reflection_of_D : 
  reflect_y_eq_x_minus_1 (reflect_x D) = (-2, 5) := by sorry

end reflection_of_D_l2689_268962


namespace book_sale_revenue_l2689_268972

theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) : 
  total_books > 0 ∧ 
  price_per_book > 0 ∧
  (total_books : ℚ) / 3 = 30 ∧ 
  price_per_book = 17/4 → 
  (2 : ℚ) / 3 * (total_books : ℚ) * price_per_book = 255 := by
sorry

end book_sale_revenue_l2689_268972


namespace total_drawing_time_l2689_268965

/-- Given Bianca's and Lucas's drawing times, prove their total drawing time is 86 minutes -/
theorem total_drawing_time 
  (bianca_school : Nat) 
  (bianca_home : Nat)
  (lucas_school : Nat)
  (lucas_home : Nat)
  (h1 : bianca_school = 22)
  (h2 : bianca_home = 19)
  (h3 : lucas_school = 10)
  (h4 : lucas_home = 35) :
  bianca_school + bianca_home + lucas_school + lucas_home = 86 := by
  sorry

#check total_drawing_time

end total_drawing_time_l2689_268965


namespace doctors_who_quit_correct_number_of_doctors_quit_l2689_268954

theorem doctors_who_quit (initial_doctors : ℕ) (initial_nurses : ℕ) 
  (nurses_quit : ℕ) (final_total : ℕ) : ℕ :=
  let doctors_quit := initial_doctors + initial_nurses - nurses_quit - final_total
  doctors_quit

theorem correct_number_of_doctors_quit : 
  doctors_who_quit 11 18 2 22 = 5 := by sorry

end doctors_who_quit_correct_number_of_doctors_quit_l2689_268954
