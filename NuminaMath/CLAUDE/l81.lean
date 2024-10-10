import Mathlib

namespace regular_triangle_on_hyperbola_coordinates_l81_8197

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define a point on the hyperbola
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

-- Define the branches of the hyperbola
def on_positive_branch (p : PointOnHyperbola) : Prop := p.x > 0
def on_negative_branch (p : PointOnHyperbola) : Prop := p.x < 0

-- Define a regular triangle on the hyperbola
structure RegularTriangleOnHyperbola where
  P : PointOnHyperbola
  Q : PointOnHyperbola
  R : PointOnHyperbola
  is_regular : True  -- We assume this property without proving it

-- Theorem statement
theorem regular_triangle_on_hyperbola_coordinates 
  (t : RegularTriangleOnHyperbola)
  (h_P : t.P.x = -1 ∧ t.P.y = 1)
  (h_P_branch : on_negative_branch t.P)
  (h_Q_branch : on_positive_branch t.Q)
  (h_R_branch : on_positive_branch t.R) :
  ((t.Q.x = 2 - Real.sqrt 3 ∧ t.Q.y = 2 + Real.sqrt 3) ∧
   (t.R.x = 2 + Real.sqrt 3 ∧ t.R.y = 2 - Real.sqrt 3)) ∨
  ((t.Q.x = 2 + Real.sqrt 3 ∧ t.Q.y = 2 - Real.sqrt 3) ∧
   (t.R.x = 2 - Real.sqrt 3 ∧ t.R.y = 2 + Real.sqrt 3)) :=
by sorry

end regular_triangle_on_hyperbola_coordinates_l81_8197


namespace sum_of_xyz_equals_sqrt_13_l81_8180

theorem sum_of_xyz_equals_sqrt_13 (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : x^2 + y^2 + x*y = 3)
  (h_eq2 : y^2 + z^2 + y*z = 4)
  (h_eq3 : z^2 + x^2 + z*x = 7) :
  x + y + z = Real.sqrt 13 := by
sorry

end sum_of_xyz_equals_sqrt_13_l81_8180


namespace problem_solution_l81_8100

theorem problem_solution :
  45 / (8 - 3/7) = 315/53 := by
  sorry

end problem_solution_l81_8100


namespace joan_balloons_l81_8163

theorem joan_balloons (total sally jessica : ℕ) (h1 : total = 16) (h2 : sally = 5) (h3 : jessica = 2) :
  ∃ joan : ℕ, joan + sally + jessica = total ∧ joan = 9 := by
  sorry

end joan_balloons_l81_8163


namespace problem_solution_l81_8144

theorem problem_solution (a b c d : ℤ) 
  (h1 : a = d)
  (h2 : b = c)
  (h3 : d + d = c * d)
  (h4 : b = d)
  (h5 : d + d = d * d)
  (h6 : c = 3) :
  a * b = 4 := by
  sorry

end problem_solution_l81_8144


namespace marble_weight_sum_l81_8142

theorem marble_weight_sum : 
  let piece1 : ℝ := 0.33
  let piece2 : ℝ := 0.33
  let piece3 : ℝ := 0.08
  piece1 + piece2 + piece3 = 0.74 := by
sorry

end marble_weight_sum_l81_8142


namespace line_quadrants_l81_8164

/-- A line passing through the second and fourth quadrants has a negative slope -/
def passes_through_second_and_fourth_quadrants (k : ℝ) : Prop :=
  k < 0

/-- A line y = kx + b passes through the first and third quadrants if k > 0 -/
def passes_through_first_and_third_quadrants (k : ℝ) : Prop :=
  k > 0

/-- A line y = kx + b passes through the fourth quadrant if k > 0 and b < 0 -/
def passes_through_fourth_quadrant (k b : ℝ) : Prop :=
  k > 0 ∧ b < 0

theorem line_quadrants (k : ℝ) :
  passes_through_second_and_fourth_quadrants k →
  passes_through_first_and_third_quadrants (-k) ∧
  passes_through_fourth_quadrant (-k) (-1) :=
by sorry

end line_quadrants_l81_8164


namespace chapters_read_l81_8160

/-- Represents the number of pages in each chapter of the book --/
def pages_per_chapter : ℕ := 8

/-- Represents the total number of pages Tom read --/
def total_pages_read : ℕ := 24

/-- Theorem stating that the number of chapters Tom read is 3 --/
theorem chapters_read : (total_pages_read / pages_per_chapter : ℕ) = 3 := by
  sorry

end chapters_read_l81_8160


namespace inscribed_circle_radius_when_area_equals_twice_perimeter_l81_8104

/-- Given a triangle with area A, perimeter p, semiperimeter s, and inradius r -/
theorem inscribed_circle_radius_when_area_equals_twice_perimeter 
  (A : ℝ) (p : ℝ) (s : ℝ) (r : ℝ) 
  (h1 : A = 2 * p)  -- Area is twice the perimeter
  (h2 : p = 2 * s)  -- Perimeter is twice the semiperimeter
  (h3 : A = r * s)  -- Area formula for a triangle
  (h4 : s ≠ 0)      -- Semiperimeter is non-zero
  : r = 4 :=
by sorry

end inscribed_circle_radius_when_area_equals_twice_perimeter_l81_8104


namespace initial_number_of_persons_l81_8130

/-- Given that when a person weighing 87 kg replaces a person weighing 67 kg,
    the average weight increases by 2.5 kg, prove that the number of persons
    initially is 8. -/
theorem initial_number_of_persons : ℕ :=
  let old_weight := 67
  let new_weight := 87
  let average_increase := 2.5
  let n := (new_weight - old_weight) / average_increase
  8

#check initial_number_of_persons

end initial_number_of_persons_l81_8130


namespace sum_consecutive_triangular_numbers_l81_8179

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_consecutive_triangular_numbers (n : ℕ) :
  triangular_number n + triangular_number (n + 1) = (n + 1)^2 := by
  sorry

end sum_consecutive_triangular_numbers_l81_8179


namespace arithmetic_sequence_problem_l81_8181

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 20)
  (h_a4 : a 4 = 2) :
  a 12 = 18 := by
  sorry

end arithmetic_sequence_problem_l81_8181


namespace arithmetic_calculation_l81_8128

theorem arithmetic_calculation : 2^2 + 3 * 4 - 5 + (6 - 1) = 16 := by
  sorry

end arithmetic_calculation_l81_8128


namespace value_of_x_l81_8194

theorem value_of_x (z y x : ℝ) (hz : z = 90) (hy : y = 1/3 * z) (hx : x = 1/2 * y) :
  x = 15 := by
  sorry

end value_of_x_l81_8194


namespace line_plane_parallelism_l81_8154

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallelLine m n) 
  (h3 : parallelLinePlane m α) : 
  parallelLinePlane n α :=
sorry

end line_plane_parallelism_l81_8154


namespace geometric_sequence_fifth_term_l81_8141

/-- Given a sequence aₙ where {1 + aₙ} is geometric with ratio 2 and a₁ = 1, prove a₅ = 31 -/
theorem geometric_sequence_fifth_term (a : ℕ → ℝ) 
  (h1 : ∀ n, (1 + a (n + 1)) = 2 * (1 + a n))  -- {1 + aₙ} is geometric with ratio 2
  (h2 : a 1 = 1)  -- a₁ = 1
  : a 5 = 31 := by
sorry

end geometric_sequence_fifth_term_l81_8141


namespace angle_at_point_l81_8105

theorem angle_at_point (x : ℝ) : 
  x > 0 ∧ x + x + 140 = 360 → x = 110 := by
  sorry

end angle_at_point_l81_8105


namespace remaining_segments_length_is_23_l81_8139

/-- Represents a polygon with perpendicular adjacent sides -/
structure Polygon where
  vertical_height : ℕ
  top_horizontal : ℕ
  first_descent : ℕ
  middle_horizontal : ℕ
  final_descent : ℕ

/-- Calculates the length of segments in the new figure after removing four sides -/
def remaining_segments_length (p : Polygon) : ℕ :=
  p.vertical_height + (p.top_horizontal + p.middle_horizontal) + 
  (p.first_descent + p.final_descent) + p.middle_horizontal

/-- The original polygon described in the problem -/
def original_polygon : Polygon :=
  { vertical_height := 7
  , top_horizontal := 3
  , first_descent := 2
  , middle_horizontal := 4
  , final_descent := 3 }

theorem remaining_segments_length_is_23 :
  remaining_segments_length original_polygon = 23 := by
  sorry

#eval remaining_segments_length original_polygon

end remaining_segments_length_is_23_l81_8139


namespace C_not_necessary_nor_sufficient_for_A_l81_8115

-- Define the propositions
variable (A B C : Prop)

-- Define the given conditions
axiom C_sufficient_for_B : C → B
axiom B_necessary_for_A : A → B

-- Theorem to prove
theorem C_not_necessary_nor_sufficient_for_A :
  ¬(∀ (h : A), C) ∧ ¬(∀ (h : C), A) :=
by sorry

end C_not_necessary_nor_sufficient_for_A_l81_8115


namespace platform_length_l81_8161

/-- Given a train of length 300 m that crosses a platform in 39 seconds
    and a signal pole in 36 seconds, the length of the platform is 25 m. -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 36) :
  let speed := train_length / time_pole
  let platform_length := speed * time_platform - train_length
  platform_length = 25 := by
  sorry

#check platform_length

end platform_length_l81_8161


namespace range_implies_m_value_subset_implies_m_range_l81_8124

-- Define the function f(x)
def f (x m : ℝ) : ℝ := |x - m| - |x - 2|

-- Define the solution set M
def M (m : ℝ) : Set ℝ := {x | f x m ≥ |x - 4|}

-- Theorem for part (1)
theorem range_implies_m_value (m : ℝ) :
  (∀ y ∈ Set.Icc (-4) 4, ∃ x, f x m = y) →
  (∀ x, f x m ∈ Set.Icc (-4) 4) →
  m = -2 ∨ m = 6 := by sorry

-- Theorem for part (2)
theorem subset_implies_m_range (m : ℝ) :
  Set.Icc 2 4 ⊆ M m →
  m ∈ Set.Iic 0 ∪ Set.Ici 6 := by sorry

end range_implies_m_value_subset_implies_m_range_l81_8124


namespace smallest_b_for_composite_l81_8159

theorem smallest_b_for_composite (b : ℕ) (h : b = 8) :
  (∀ x : ℤ, ∃ y z : ℤ, y ≠ 1 ∧ z ≠ 1 ∧ y * z = x^4 + b^4) ∧
  (∀ b' : ℕ, 0 < b' ∧ b' < b →
    ∃ x : ℤ, ∀ y z : ℤ, (y * z = x^4 + b'^4) → (y = 1 ∨ z = 1)) :=
sorry

end smallest_b_for_composite_l81_8159


namespace max_students_before_new_year_l81_8175

/-- The maximum number of students before New Year given the conditions -/
theorem max_students_before_new_year
  (N : ℕ) -- Total number of students before New Year
  (M : ℕ) -- Number of boys before New Year
  (k : ℕ) -- Percentage of boys before New Year
  (ℓ : ℕ) -- Percentage of boys after New Year
  (h1 : M = k * N / 100) -- Condition relating M, k, and N
  (h2 : ℓ < 100) -- ℓ is less than 100
  (h3 : 100 * (M + 1) = ℓ * (N + 3)) -- Condition after New Year
  : N ≤ 197 := by
  sorry

end max_students_before_new_year_l81_8175


namespace selling_price_calculation_l81_8120

def cost_price : ℝ := 1500
def loss_percentage : ℝ := 14.000000000000002

theorem selling_price_calculation (cost_price : ℝ) (loss_percentage : ℝ) :
  let loss_amount := (loss_percentage / 100) * cost_price
  let selling_price := cost_price - loss_amount
  selling_price = 1290 := by sorry

end selling_price_calculation_l81_8120


namespace sqrt_equation_l81_8119

theorem sqrt_equation (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end sqrt_equation_l81_8119


namespace largest_square_side_length_l81_8188

-- Define the lengths of sticks for each side
def side1 : List ℕ := [4, 4, 2, 3]
def side2 : List ℕ := [4, 4, 3, 1, 1]
def side3 : List ℕ := [4, 3, 3, 2, 1]
def side4 : List ℕ := [3, 3, 3, 2, 2]

-- Theorem statement
theorem largest_square_side_length :
  List.sum side1 = List.sum side2 ∧
  List.sum side2 = List.sum side3 ∧
  List.sum side3 = List.sum side4 ∧
  List.sum side4 = 13 := by
  sorry

end largest_square_side_length_l81_8188


namespace mary_marbles_count_l81_8150

/-- The number of yellow marbles Mary and Joan have in total -/
def total_marbles : ℕ := 12

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := total_marbles - joan_marbles

theorem mary_marbles_count : mary_marbles = 9 := by
  sorry

end mary_marbles_count_l81_8150


namespace helen_washing_time_l81_8166

/-- The time it takes Helen to wash pillowcases each time -/
def washing_time (weeks_between_washes : ℕ) (minutes_per_year : ℕ) (weeks_per_year : ℕ) : ℕ :=
  minutes_per_year / (weeks_per_year / weeks_between_washes)

/-- Theorem stating that Helen's pillowcase washing time is 30 minutes -/
theorem helen_washing_time :
  washing_time 4 390 52 = 30 := by
  sorry

end helen_washing_time_l81_8166


namespace discount_comparison_l81_8198

theorem discount_comparison (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_price : x = 2 * y) :
  x + y = (3/2) * (0.6 * x + 0.8 * y) :=
by sorry

#check discount_comparison

end discount_comparison_l81_8198


namespace power_product_squared_l81_8108

theorem power_product_squared (m n : ℝ) : (m * n)^2 = m^2 * n^2 := by
  sorry

end power_product_squared_l81_8108


namespace f_increasing_iff_a_ge_four_l81_8101

/-- The function f(x) = ax - x^3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

/-- The theorem statement -/
theorem f_increasing_iff_a_ge_four (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 → f a x2 - f a x1 > x2 - x1) ↔ a ≥ 4 := by
  sorry

end f_increasing_iff_a_ge_four_l81_8101


namespace rectangle_area_change_l81_8152

theorem rectangle_area_change (original_area : ℝ) (length_decrease : ℝ) (width_increase : ℝ) :
  original_area = 600 →
  length_decrease = 0.2 →
  width_increase = 0.05 →
  original_area * (1 - length_decrease) * (1 + width_increase) = 504 := by
  sorry

end rectangle_area_change_l81_8152


namespace solution_set_min_value_min_value_expression_l81_8158

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Theorem stating the solution set for f(x) ≤ 4
theorem solution_set : {x : ℝ | f x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Theorem stating the minimum value of f(x)
theorem min_value : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ m = 3 := by sorry

-- Theorem for the minimum value of 1/(a-1) + 2/b
theorem min_value_expression :
  ∀ (a b : ℝ), a > 1 → b > 0 → a + 2*b = 3 →
  ∀ (y : ℝ), y = 1/(a-1) + 2/b → y ≥ 9/2 := by sorry

end solution_set_min_value_min_value_expression_l81_8158


namespace sphere_surface_area_from_cube_l81_8178

/-- Given a cube with surface area 6a^2 and all its vertices on a sphere,
    prove that the surface area of the sphere is 3πa^2 -/
theorem sphere_surface_area_from_cube (a : ℝ) (h : a > 0) :
  let cube_surface_area := 6 * a^2
  let cube_diagonal := a * Real.sqrt 3
  let sphere_radius := cube_diagonal / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  sphere_surface_area = 3 * Real.pi * a^2 := by
sorry

end sphere_surface_area_from_cube_l81_8178


namespace division_problem_l81_8182

theorem division_problem (dividend : ℕ) (remainder : ℕ) (quotient : ℕ) (divisor : ℕ) (n : ℕ) :
  dividend = 251 →
  remainder = 8 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + n →
  dividend = divisor * quotient + remainder →
  n = 3 :=
by sorry

end division_problem_l81_8182


namespace rounding_and_multiplication_l81_8146

/-- Round a number to the nearest significant figure -/
def roundToSignificantFigure (x : ℝ) : ℝ := sorry

/-- Round a number up to the nearest hundred -/
def roundUpToHundred (x : ℝ) : ℕ := sorry

/-- The main theorem -/
theorem rounding_and_multiplication :
  let a := 0.000025
  let b := 6546300
  let rounded_a := roundToSignificantFigure a
  let rounded_b := roundToSignificantFigure b
  let product := rounded_a * rounded_b
  roundUpToHundred product = 200 := by sorry

end rounding_and_multiplication_l81_8146


namespace joe_test_scores_l81_8132

theorem joe_test_scores (initial_avg : ℚ) (lowest_score : ℚ) (new_avg : ℚ) 
  (h1 : initial_avg = 90)
  (h2 : lowest_score = 75)
  (h3 : new_avg = 85) :
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_avg - lowest_score = (n - 1 : ℚ) * new_avg ∧
    n = 13 := by
sorry

end joe_test_scores_l81_8132


namespace inequality_proof_l81_8170

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  b * c^2 + c * a^2 + a * b^2 < b^2 * c + c^2 * a + a^2 * b := by
sorry

end inequality_proof_l81_8170


namespace spherical_coordinate_transformation_l81_8102

/-- Given a point with rectangular coordinates (-3, -4, 5) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, -θ, φ) has rectangular coordinates (-3, 4, 5) -/
theorem spherical_coordinate_transformation (ρ θ φ : Real) 
  (h1 : -3 = ρ * Real.sin φ * Real.cos θ)
  (h2 : -4 = ρ * Real.sin φ * Real.sin θ)
  (h3 : 5 = ρ * Real.cos φ) :
  (-3 = ρ * Real.sin φ * Real.cos (-θ)) ∧ 
  (4 = ρ * Real.sin φ * Real.sin (-θ)) ∧ 
  (5 = ρ * Real.cos φ) :=
by sorry

end spherical_coordinate_transformation_l81_8102


namespace specific_profit_calculation_l81_8183

/-- Given an item cost, markup percentage, discount percentage, and number of items sold,
    calculates the total profit. -/
def totalProfit (a : ℝ) (markup discount : ℝ) (m : ℝ) : ℝ :=
  let sellingPrice := a * (1 + markup)
  let discountedPrice := sellingPrice * (1 - discount)
  m * (discountedPrice - a)

/-- Theorem stating that under specific conditions, the total profit is 0.08am -/
theorem specific_profit_calculation (a m : ℝ) :
  totalProfit a 0.2 0.1 m = 0.08 * a * m :=
by sorry

end specific_profit_calculation_l81_8183


namespace nonnegative_solutions_system_l81_8196

theorem nonnegative_solutions_system (x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  Real.sqrt (x + y) + Real.sqrt z = 7 →
  Real.sqrt (x + z) + Real.sqrt y = 7 →
  Real.sqrt (y + z) + Real.sqrt x = 5 →
  ((x = 1 ∧ y = 4 ∧ z = 4) ∨ (x = 1 ∧ y = 9 ∧ z = 9)) :=
by sorry

end nonnegative_solutions_system_l81_8196


namespace modulo_equivalence_in_range_l81_8136

theorem modulo_equivalence_in_range : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [MOD 11] ∧ n = 3 := by
  sorry

end modulo_equivalence_in_range_l81_8136


namespace power_equation_solution_l81_8113

theorem power_equation_solution (n : Real) : 
  10^n = 10^4 * Real.sqrt (10^155 / 0.0001) → n = 83.5 := by
  sorry

end power_equation_solution_l81_8113


namespace tan_theta_equals_five_twelfths_l81_8177

/-- Given a dilation matrix D and a rotation matrix R, prove that tan θ = 5/12 -/
theorem tan_theta_equals_five_twelfths 
  (k : ℝ) 
  (θ : ℝ) 
  (hk : k > 0) 
  (D : Matrix (Fin 2) (Fin 2) ℝ) 
  (R : Matrix (Fin 2) (Fin 2) ℝ) 
  (hD : D = ![![k, 0], ![0, k]]) 
  (hR : R = ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]) 
  (h_prod : R * D = ![![12, -5], ![5, 12]]) : 
  Real.tan θ = 5/12 := by
sorry

end tan_theta_equals_five_twelfths_l81_8177


namespace trapezoidal_dam_pressure_l81_8145

/-- 
Represents a vertical trapezoidal dam with water pressure.
-/
structure TrapezoidalDam where
  ρ : ℝ  -- density of water
  g : ℝ  -- acceleration due to gravity
  h : ℝ  -- height of the dam
  a : ℝ  -- top width of the dam
  b : ℝ  -- bottom width of the dam
  h_pos : h > 0
  a_pos : a > 0
  b_pos : b > 0
  a_ge_b : a ≥ b

/-- 
The total water pressure on a vertical trapezoidal dam is ρg(h^2(2a + b))/6.
-/
theorem trapezoidal_dam_pressure (dam : TrapezoidalDam) :
  ∃ P : ℝ, P = dam.ρ * dam.g * (dam.h^2 * (2 * dam.a + dam.b)) / 6 :=
by
  sorry

end trapezoidal_dam_pressure_l81_8145


namespace total_coins_l81_8187

/-- Given 5 piles of quarters, 5 piles of dimes, and 3 coins in each pile, 
    the total number of coins is 30. -/
theorem total_coins (piles_quarters piles_dimes coins_per_pile : ℕ) 
  (h1 : piles_quarters = 5)
  (h2 : piles_dimes = 5)
  (h3 : coins_per_pile = 3) :
  piles_quarters * coins_per_pile + piles_dimes * coins_per_pile = 30 :=
by sorry

end total_coins_l81_8187


namespace motorboat_speed_l81_8184

/-- Prove that the maximum speed of a motorboat in still water is 40 km/h given the specified conditions -/
theorem motorboat_speed (flood_rate : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  flood_rate = 10 →
  downstream_distance = 2 →
  upstream_distance = 1.2 →
  (downstream_distance / (v + flood_rate) = upstream_distance / (v - flood_rate)) →
  v = 40 :=
by
  sorry

#check motorboat_speed

end motorboat_speed_l81_8184


namespace only_two_valid_plans_l81_8127

/-- Represents a deployment plan for trucks -/
structure DeploymentPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a deployment plan is valid according to the given conditions -/
def isValidPlan (p : DeploymentPlan) : Prop :=
  p.typeA + p.typeB = 70 ∧
  p.typeB ≤ 3 * p.typeA ∧
  25 * p.typeA + 15 * p.typeB ≤ 1245

/-- The set of all valid deployment plans -/
def validPlans : Set DeploymentPlan :=
  {p | isValidPlan p}

/-- The theorem stating that there are only two valid deployment plans -/
theorem only_two_valid_plans :
  validPlans = {DeploymentPlan.mk 18 52, DeploymentPlan.mk 19 51} :=
by
  sorry

end only_two_valid_plans_l81_8127


namespace soccer_season_games_l81_8165

/-- The number of months in the soccer season -/
def season_length : ℕ := 3

/-- The number of soccer games played per month -/
def games_per_month : ℕ := 9

/-- The total number of soccer games played during the season -/
def total_games : ℕ := season_length * games_per_month

theorem soccer_season_games : total_games = 27 := by
  sorry

end soccer_season_games_l81_8165


namespace subset_P_l81_8167

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by
  sorry

end subset_P_l81_8167


namespace solution_set_for_a_eq_2_range_of_a_for_x_in_1_to_3_l81_8129

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |2*x - 1|

-- Statement for part 1
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x + 3 ≥ 0} = {x : ℝ | -4 ≤ x ∧ x ≤ 2} := by sorry

-- Statement for part 2
theorem range_of_a_for_x_in_1_to_3 :
  (∀ x ∈ Set.Icc 1 3, f a x ≤ 3) → a ∈ Set.Icc (-3) 5 := by sorry

end solution_set_for_a_eq_2_range_of_a_for_x_in_1_to_3_l81_8129


namespace sufficient_not_necessary_l81_8112

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x ≠ 5 ∧ x^2 - 4*x - 5 = 0) ∧ 
  (∀ x : ℝ, x = 5 → x^2 - 4*x - 5 = 0) := by
  sorry

#check sufficient_not_necessary

end sufficient_not_necessary_l81_8112


namespace quadratic_cubic_inequalities_l81_8107

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem quadratic_cubic_inequalities 
  (m n a b : ℝ) 
  (h1 : n = 0)
  (h2 : -2 * m + n = -2)
  (h3 : m * 1^2 + n * 1 = a * 1^3 + b * 1 - 3)
  (h4 : 2 * m * 1 + n = 3 * a * 1^2 + b) :
  ∃ (k p : ℝ), k = 2 ∧ p = -1 ∧ 
  (∀ x > 0, f m n x ≥ k * x + p ∧ g a b x ≤ k * x + p) := by
sorry

end quadratic_cubic_inequalities_l81_8107


namespace lowest_price_for_electronic_component_l81_8185

/-- Calculates the lowest price per component to break even -/
def lowest_break_even_price (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (units_sold : ℕ) : ℚ :=
  (production_cost + shipping_cost + (fixed_costs / units_sold))

theorem lowest_price_for_electronic_component :
  let production_cost : ℚ := 80
  let shipping_cost : ℚ := 3
  let fixed_costs : ℚ := 16500
  let units_sold : ℕ := 150
  lowest_break_even_price production_cost shipping_cost fixed_costs units_sold = 193 := by
sorry

#eval lowest_break_even_price 80 3 16500 150

end lowest_price_for_electronic_component_l81_8185


namespace no_alpha_sequence_exists_l81_8121

theorem no_alpha_sequence_exists : ¬ ∃ (α : ℝ) (a : ℕ → ℝ),
  (0 < α ∧ α < 1) ∧
  (∀ n, 0 < a n) ∧
  (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end no_alpha_sequence_exists_l81_8121


namespace unique_n_for_prime_sequence_l81_8193

theorem unique_n_for_prime_sequence : ∃! (n : ℕ), 
  n > 0 ∧ 
  Nat.Prime (n + 1) ∧ 
  Nat.Prime (n + 3) ∧ 
  Nat.Prime (n + 7) ∧ 
  Nat.Prime (n + 9) ∧ 
  Nat.Prime (n + 13) ∧ 
  Nat.Prime (n + 15) :=
by sorry

end unique_n_for_prime_sequence_l81_8193


namespace achieve_target_average_l81_8147

/-- Represents Gage's skating schedule and target average -/
structure SkatingSchedule where
  days_with_80_min : Nat
  days_with_100_min : Nat
  target_average : Nat
  total_days : Nat

/-- Calculates the total skating time for the given schedule -/
def total_skating_time (schedule : SkatingSchedule) (last_day_minutes : Nat) : Nat :=
  schedule.days_with_80_min * 80 + 
  schedule.days_with_100_min * 100 + 
  last_day_minutes

/-- Theorem stating that skating 140 minutes on the 8th day achieves the target average -/
theorem achieve_target_average (schedule : SkatingSchedule) 
    (h1 : schedule.days_with_80_min = 4)
    (h2 : schedule.days_with_100_min = 3)
    (h3 : schedule.target_average = 95)
    (h4 : schedule.total_days = 8) :
  total_skating_time schedule 140 / schedule.total_days = schedule.target_average := by
  sorry

#eval total_skating_time { days_with_80_min := 4, days_with_100_min := 3, target_average := 95, total_days := 8 } 140

end achieve_target_average_l81_8147


namespace midpoint_distance_to_origin_l81_8199

theorem midpoint_distance_to_origin : 
  let p1 : ℝ × ℝ := (-6, 8)
  let p2 : ℝ × ℝ := (6, -8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1^2 + midpoint.2^2).sqrt = 0 := by
  sorry

end midpoint_distance_to_origin_l81_8199


namespace gcd_factorial_problem_l81_8172

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem gcd_factorial_problem : Nat.gcd (factorial 7) ((factorial 10) / (factorial 4)) = 5040 := by
  sorry

end gcd_factorial_problem_l81_8172


namespace hyperbola_center_l81_8143

theorem hyperbola_center (x y : ℝ) :
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 71 = 0 →
  ∃ a b : ℝ, a = 3 ∧ b = 4 ∧
  ∀ x' y' : ℝ, 9 * (x' - a)^2 - 16 * (y' - b)^2 = 9 * x'^2 - 54 * x' - 16 * y'^2 + 128 * y' - 71 :=
by sorry

end hyperbola_center_l81_8143


namespace train_speed_l81_8135

/-- The speed of a train given the time it takes to pass a pole and cross a stationary train -/
theorem train_speed
  (pole_pass_time : ℝ)
  (stationary_train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : pole_pass_time = 5)
  (h2 : stationary_train_length = 360)
  (h3 : crossing_time = 25) :
  let train_speed := stationary_train_length / (crossing_time - pole_pass_time)
  train_speed = 18 := by
  sorry

end train_speed_l81_8135


namespace quadratic_equation_solution_inequality_system_solution_l81_8116

-- Part 1: Quadratic equation
theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by sorry

-- Part 2: System of inequalities
theorem inequality_system_solution :
  ∀ x : ℝ, (x + 3 > 0 ∧ 2*(x + 1) < 4) ↔ (-3 < x ∧ x < 1) := by sorry

end quadratic_equation_solution_inequality_system_solution_l81_8116


namespace f_equals_g_l81_8174

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (t : ℝ) : ℝ := t^2 - 2*t

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end f_equals_g_l81_8174


namespace smallest_blue_chips_l81_8157

theorem smallest_blue_chips (total : ℕ) (h_total : total = 49) :
  ∃ (blue red prime : ℕ),
    blue + red = total ∧
    red = blue + prime ∧
    Nat.Prime prime ∧
    ∀ (b r p : ℕ), b + r = total → r = b + p → Nat.Prime p → blue ≤ b :=
by sorry

end smallest_blue_chips_l81_8157


namespace solution_value_l81_8123

theorem solution_value (a : ℝ) (h : a^2 - 5*a - 1 = 0) : 3*a^2 - 15*a = 3 := by
  sorry

end solution_value_l81_8123


namespace division_of_fractions_l81_8114

theorem division_of_fractions : (3 + 1/2) / 7 / (5/3) = 3/10 := by
  sorry

end division_of_fractions_l81_8114


namespace smaller_two_digit_factor_l81_8133

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 4536 →
  min a b = 54 := by
sorry

end smaller_two_digit_factor_l81_8133


namespace tourist_distribution_l81_8111

theorem tourist_distribution (n m : ℕ) (hn : n = 8) (hm : m = 3) :
  (m ^ n : ℕ) - m * ((m - 1) ^ n) + (m.choose 2) * (1 ^ n) = 5796 :=
sorry

end tourist_distribution_l81_8111


namespace no_real_solution_cos_sin_l81_8106

theorem no_real_solution_cos_sin : ¬∃ (x : ℝ), (Real.cos x = 1/2) ∧ (Real.sin x = 3/4) := by
  sorry

end no_real_solution_cos_sin_l81_8106


namespace logarithm_sum_approximation_l81_8140

theorem logarithm_sum_approximation : 
  let expr := (1 / (Real.log 3 / Real.log 8 + 1)) + 
              (1 / (Real.log 2 / Real.log 12 + 1)) + 
              (1 / (Real.log 4 / Real.log 9 + 1))
  ∃ ε > 0, |expr - 3| < ε := by
  sorry

end logarithm_sum_approximation_l81_8140


namespace opposite_of_negative_six_l81_8103

theorem opposite_of_negative_six : -((-6 : ℝ)) = (6 : ℝ) := by
  sorry

end opposite_of_negative_six_l81_8103


namespace geometric_sequence_fifth_term_l81_8162

/-- A geometric sequence with third term 16 and seventh term 2 has fifth term 8 -/
theorem geometric_sequence_fifth_term (a : ℝ) (r : ℝ) 
  (h1 : a * r^2 = 16)  -- third term is 16
  (h2 : a * r^6 = 2)   -- seventh term is 2
  : a * r^4 = 8 :=     -- fifth term is 8
by sorry

end geometric_sequence_fifth_term_l81_8162


namespace q_factor_change_l81_8176

theorem q_factor_change (w m z : ℝ) (hw : w ≠ 0) (hm : m ≠ 0) (hz : z ≠ 0) :
  let q := 5 * w / (4 * m * z^2)
  let q_new := 5 * (4*w) / (4 * (2*m) * (3*z)^2)
  q_new = (2/9) * q := by
sorry

end q_factor_change_l81_8176


namespace largest_non_sum_of_composites_l81_8156

/-- A natural number is composite if it has a proper factor -/
def IsComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A natural number can be represented as the sum of two composite numbers -/
def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

/-- 11 is the largest natural number that cannot be represented as the sum of two composite numbers -/
theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end largest_non_sum_of_composites_l81_8156


namespace fourth_root_cube_root_equality_l81_8134

theorem fourth_root_cube_root_equality : 
  (0.000008 : ℝ)^((1/3) * (1/4)) = (2 : ℝ)^(1/4) / (10 : ℝ)^(1/2) :=
sorry

end fourth_root_cube_root_equality_l81_8134


namespace odd_function_value_l81_8191

def f (a : ℝ) (x : ℝ) : ℝ := a * x + a + 3

theorem odd_function_value (a : ℝ) :
  (∀ x : ℝ, f a x = -(f a (-x))) → a = -3 := by
  sorry

end odd_function_value_l81_8191


namespace function_square_evaluation_l81_8138

theorem function_square_evaluation (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2
  f (a + 1) = a^2 + 2*a + 1 := by sorry

end function_square_evaluation_l81_8138


namespace no_real_solution_exists_sum_equals_square_plus_twenty_l81_8110

/-- Given three numbers with a difference of 4 between each,
    where their sum is 20 more than the square of the first number,
    prove that no real solution exists for the middle number. -/
theorem no_real_solution_exists (x : ℝ) : ¬ ∃ x : ℝ, x^2 - 3*x + 8 = 0 := by
  sorry

/-- Define the relationship between the three numbers -/
def second_number (x : ℝ) : ℝ := x + 4

/-- Define the relationship between the three numbers -/
def third_number (x : ℝ) : ℝ := x + 8

/-- Define the sum of the three numbers -/
def sum_of_numbers (x : ℝ) : ℝ := x + second_number x + third_number x

/-- Define the relationship between the sum and the square of the first number -/
theorem sum_equals_square_plus_twenty (x : ℝ) : sum_of_numbers x = x^2 + 20 := by
  sorry

end no_real_solution_exists_sum_equals_square_plus_twenty_l81_8110


namespace balance_four_hearts_l81_8109

/-- Represents the weight of a symbol in the balance game -/
structure Weight (α : Type) where
  value : ℚ

/-- The balance game with three symbols -/
structure BalanceGame where
  star : Weight ℚ
  heart : Weight ℚ
  circle : Weight ℚ

/-- Defines the balance equations for the game -/
def balance_equations (game : BalanceGame) : Prop :=
  4 * game.star.value + 3 * game.heart.value = 12 * game.circle.value ∧
  2 * game.star.value = game.heart.value + 3 * game.circle.value

/-- The main theorem to prove -/
theorem balance_four_hearts (game : BalanceGame) :
  balance_equations game →
  4 * game.heart.value = 5 * game.circle.value :=
by sorry

end balance_four_hearts_l81_8109


namespace limit_special_function_l81_8131

/-- The limit of (5 - 4/cos(x))^(1/sin^2(3x)) as x approaches 0 is e^(-2/9) -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
      |(5 - 4 / Real.cos x) ^ (1 / Real.sin (3 * x) ^ 2) - Real.exp (-2/9)| < ε := by
  sorry

end limit_special_function_l81_8131


namespace cos_alpha_plus_seven_pi_twelfths_l81_8126

theorem cos_alpha_plus_seven_pi_twelfths (α : ℝ) 
  (h : Real.sin (α + π / 12) = 1 / 3) : 
  Real.cos (α + 7 * π / 12) = -1 / 3 := by
  sorry

end cos_alpha_plus_seven_pi_twelfths_l81_8126


namespace sin_75_cos_45_minus_cos_75_sin_45_l81_8117

theorem sin_75_cos_45_minus_cos_75_sin_45 :
  Real.sin (75 * π / 180) * Real.cos (45 * π / 180) -
  Real.cos (75 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end sin_75_cos_45_minus_cos_75_sin_45_l81_8117


namespace y_values_l81_8149

theorem y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  let y := ((x - 3)^2 * (x + 2)) / (2 * x - 4)
  y = 9 ∨ y = 3.6 :=
by sorry

end y_values_l81_8149


namespace jerrys_action_figures_l81_8190

theorem jerrys_action_figures (initial : ℕ) : 
  (initial + 2 - 7 = 10) → initial = 15 := by
  sorry

end jerrys_action_figures_l81_8190


namespace smaller_number_in_ratio_l81_8137

theorem smaller_number_in_ratio (a b : ℝ) : 
  a / b = 3 / 4 → a + b = 420 → a = 180 := by sorry

end smaller_number_in_ratio_l81_8137


namespace weight_of_b_l81_8186

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 30)
  (h2 : (a + b) / 2 = 25)
  (h3 : (b + c) / 2 = 28) : 
  b = 16 := by sorry

end weight_of_b_l81_8186


namespace circle_tangent_intersection_ratio_l81_8148

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are externally tangent at a point -/
def ExternallyTangent (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Two circles intersect at a point -/
def Intersect (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Distance between two points -/
def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem circle_tangent_intersection_ratio
  (Γ₁ Γ₂ Γ₃ Γ₄ : Circle)
  (P A B C D : ℝ × ℝ)
  (h1 : Γ₁ ≠ Γ₂ ∧ Γ₁ ≠ Γ₃ ∧ Γ₁ ≠ Γ₄ ∧ Γ₂ ≠ Γ₃ ∧ Γ₂ ≠ Γ₄ ∧ Γ₃ ≠ Γ₄)
  (h2 : ExternallyTangent Γ₁ Γ₃ P)
  (h3 : ExternallyTangent Γ₂ Γ₄ P)
  (h4 : Intersect Γ₁ Γ₂ A)
  (h5 : Intersect Γ₂ Γ₃ B)
  (h6 : Intersect Γ₃ Γ₄ C)
  (h7 : Intersect Γ₄ Γ₁ D)
  (h8 : A ≠ P ∧ B ≠ P ∧ C ≠ P ∧ D ≠ P) :
  (Distance A B * Distance B C) / (Distance A D * Distance D C) = 
  (Distance P B)^2 / (Distance P D)^2 :=
sorry

end circle_tangent_intersection_ratio_l81_8148


namespace cheese_cost_proof_l81_8125

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 2

/-- The number of sandwiches Ted makes -/
def num_sandwiches : ℕ := 10

/-- The cost of bread in dollars -/
def bread_cost : ℚ := 4

/-- The cost of one pack of sandwich meat in dollars -/
def meat_cost : ℚ := 5

/-- The number of packs of sandwich meat needed -/
def num_meat_packs : ℕ := 2

/-- The number of packs of sliced cheese needed -/
def num_cheese_packs : ℕ := 2

/-- The discount on one pack of cheese in dollars -/
def cheese_discount : ℚ := 1

/-- The discount on one pack of meat in dollars -/
def meat_discount : ℚ := 1

/-- The cost of one pack of sliced cheese without the coupon -/
def cheese_cost : ℚ := 4.5

theorem cheese_cost_proof :
  cheese_cost * num_cheese_packs + bread_cost + meat_cost * num_meat_packs - 
  cheese_discount - meat_discount = sandwich_cost * num_sandwiches := by
  sorry

end cheese_cost_proof_l81_8125


namespace yara_ahead_of_theon_l81_8189

/-- Proves that Yara will be 3 hours ahead of Theon given their ship speeds and destination distance -/
theorem yara_ahead_of_theon (theon_speed yara_speed distance : ℝ) 
  (h1 : theon_speed = 15)
  (h2 : yara_speed = 30)
  (h3 : distance = 90) :
  yara_speed / distance - theon_speed / distance = 3 := by
  sorry

end yara_ahead_of_theon_l81_8189


namespace lucille_earnings_l81_8169

/-- Represents the number of weeds in different areas of the garden -/
structure GardenWeeds where
  flower_bed : Nat
  vegetable_patch : Nat
  grass : Nat

/-- Represents Lucille's weeding and earnings -/
def LucilleWeeding (garden : GardenWeeds) (soda_cost : Nat) (money_left : Nat) : Prop :=
  let weeds_pulled := garden.flower_bed + garden.vegetable_patch + garden.grass / 2
  let total_earnings := soda_cost + money_left
  total_earnings / weeds_pulled = 6

/-- Theorem: Given the garden conditions and Lucille's spending, she earns 6 cents per weed -/
theorem lucille_earnings (garden : GardenWeeds) 
  (h1 : garden.flower_bed = 11)
  (h2 : garden.vegetable_patch = 14)
  (h3 : garden.grass = 32)
  (h4 : LucilleWeeding garden 99 147) : 
  ∃ (earnings_per_weed : Nat), earnings_per_weed = 6 := by
  sorry

end lucille_earnings_l81_8169


namespace quadratic_inequality_range_l81_8173

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 + (a - 1) * x - 1 < 0) ↔ -3/5 < a ∧ a ≤ 1 :=
by sorry

end quadratic_inequality_range_l81_8173


namespace carls_membership_number_l81_8151

/-- A predicate to check if a number is a two-digit prime -/
def isTwoDigitPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

/-- The main theorem -/
theorem carls_membership_number
  (a b c d : ℕ)
  (ha : isTwoDigitPrime a)
  (hb : isTwoDigitPrime b)
  (hc : isTwoDigitPrime c)
  (hd : isTwoDigitPrime d)
  (sum_all : a + b + c + d = 100)
  (sum_no_ben : b + c + d = 30)
  (sum_no_carl : a + b + d = 29)
  (sum_no_david : a + b + c = 23) :
  c = 23 := by
  sorry


end carls_membership_number_l81_8151


namespace smallest_sum_of_squares_l81_8122

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 145 → ∃ (a b : ℕ), a^2 - b^2 = 145 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 433 :=
by sorry

end smallest_sum_of_squares_l81_8122


namespace integral_inequality_l81_8171

theorem integral_inequality (n : ℕ) (hn : n ≥ 2) :
  (1 : ℝ) / n < ∫ x in (0 : ℝ)..(π / 2), 1 / (1 + Real.cos x) ^ n ∧
  ∫ x in (0 : ℝ)..(π / 2), 1 / (1 + Real.cos x) ^ n < (n + 5 : ℝ) / (n * (n + 1)) :=
by sorry

end integral_inequality_l81_8171


namespace dagger_example_l81_8155

-- Define the ⋄ operation
def dagger (m n p q : ℚ) : ℚ := m^2 * p * (q / n)

-- Theorem statement
theorem dagger_example : dagger 5 9 4 6 = 200 / 3 := by
  sorry

end dagger_example_l81_8155


namespace water_polo_team_selection_result_l81_8118

/-- The number of ways to choose a starting team in water polo -/
def water_polo_team_selection (total_players : ℕ) (team_size : ℕ) (goalie_count : ℕ) : ℕ :=
  Nat.choose total_players goalie_count * Nat.choose (total_players - goalie_count) (team_size - goalie_count)

/-- Theorem: The number of ways to choose a starting team of 9 players (including 2 goalies) from a team of 20 members is 6,046,560 -/
theorem water_polo_team_selection_result :
  water_polo_team_selection 20 9 2 = 6046560 := by
  sorry

end water_polo_team_selection_result_l81_8118


namespace specific_prism_volume_l81_8195

/-- A rectangular prism with given edge length sum and proportions -/
structure RectangularPrism where
  edgeSum : ℝ
  width : ℝ
  height : ℝ
  length : ℝ
  edgeSum_eq : edgeSum = 4 * (width + height + length)
  height_prop : height = 2 * width
  length_prop : length = 4 * width

/-- The volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℝ := p.width * p.height * p.length

/-- Theorem: The volume of the specific rectangular prism is 85184/343 -/
theorem specific_prism_volume :
  ∃ (p : RectangularPrism), p.edgeSum = 88 ∧ volume p = 85184 / 343 := by
  sorry

end specific_prism_volume_l81_8195


namespace smartphone_charge_time_proof_l81_8153

/-- The time in minutes to fully charge a smartphone -/
def smartphone_charge_time : ℝ := 26

/-- The time in minutes to fully charge a tablet -/
def tablet_charge_time : ℝ := 53

/-- The total time in minutes for Ana to charge her devices -/
def ana_charge_time : ℝ := 66

theorem smartphone_charge_time_proof :
  smartphone_charge_time = 26 :=
by
  have h1 : tablet_charge_time = 53 := rfl
  have h2 : tablet_charge_time + (1/2 * smartphone_charge_time) = ana_charge_time :=
    by sorry
  sorry

end smartphone_charge_time_proof_l81_8153


namespace gift_payment_l81_8192

theorem gift_payment (a b c d : ℝ) 
  (h1 : a + b + c + d = 84)
  (h2 : a = (1/3) * (b + c + d))
  (h3 : b = (1/4) * (a + c + d))
  (h4 : c = (1/5) * (a + b + d))
  (h5 : a ≥ 0) (h6 : b ≥ 0) (h7 : c ≥ 0) (h8 : d ≥ 0) : 
  d = 40 := by
  sorry

end gift_payment_l81_8192


namespace no_solution_to_system_l81_8168

theorem no_solution_to_system :
  ¬∃ (x y : ℝ), 
    (80 * x + 15 * y - 7) / (78 * x + 12 * y) = 1 ∧
    (2 * x^2 + 3 * y^2 - 11) / (y^2 - x^2 + 3) = 1 ∧
    78 * x + 12 * y ≠ 0 ∧
    y^2 - x^2 + 3 ≠ 0 := by
  sorry

end no_solution_to_system_l81_8168
