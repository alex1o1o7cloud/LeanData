import Mathlib

namespace condition_analysis_l1761_176124

theorem condition_analysis (a b c : ℝ) (h : a > b ∧ b > c) :
  (∀ a b c, a + b + c = 0 → a * b > a * c) ∧
  (∃ a b c, a * b > a * c ∧ a + b + c ≠ 0) := by
  sorry

end condition_analysis_l1761_176124


namespace vegetable_pieces_count_l1761_176185

/-- Calculates the total number of vegetable pieces after cutting -/
def total_vegetable_pieces (bell_peppers onions zucchinis : ℕ) : ℕ :=
  let bell_pepper_thin := (bell_peppers / 4) * 20
  let bell_pepper_large := (bell_peppers - bell_peppers / 4) * 10
  let bell_pepper_small := (bell_pepper_large / 2) * 3
  let onion_thin := (onions / 2) * 18
  let onion_chunk := (onions - onions / 2) * 8
  let zucchini_thin := (zucchinis * 3 / 10) * 15
  let zucchini_chunk := (zucchinis - zucchinis * 3 / 10) * 8
  bell_pepper_thin + bell_pepper_large + bell_pepper_small + onion_thin + onion_chunk + zucchini_thin + zucchini_chunk

/-- Theorem stating that given the conditions, the total number of vegetable pieces is 441 -/
theorem vegetable_pieces_count : total_vegetable_pieces 10 7 15 = 441 := by
  sorry

end vegetable_pieces_count_l1761_176185


namespace tetrahedron_volume_l1761_176140

/-- The volume of a tetrahedron with vertices on coordinate axes -/
theorem tetrahedron_volume (d e f : ℝ) : 
  d > 0 → e > 0 → f > 0 →  -- Positive coordinates
  d^2 + e^2 = 49 →         -- DE = 7
  e^2 + f^2 = 64 →         -- EF = 8
  f^2 + d^2 = 81 →         -- FD = 9
  (1/6 : ℝ) * d * e * f = 4 * Real.sqrt 11 := by
  sorry


end tetrahedron_volume_l1761_176140


namespace composite_cubes_surface_area_l1761_176106

/-- Represents a composite shape formed by two cubes -/
structure CompositeCubes where
  large_cube_edge : ℝ
  small_cube_edge : ℝ

/-- Calculate the surface area of the composite shape -/
def surface_area (shape : CompositeCubes) : ℝ :=
  let large_cube_area := 6 * shape.large_cube_edge ^ 2
  let small_cube_area := 6 * shape.small_cube_edge ^ 2
  let covered_area := shape.small_cube_edge ^ 2
  let exposed_small_cube_area := 4 * shape.small_cube_edge ^ 2
  large_cube_area - covered_area + exposed_small_cube_area

/-- Theorem stating that the surface area of the specific composite shape is 49 -/
theorem composite_cubes_surface_area : 
  let shape := CompositeCubes.mk 3 1
  surface_area shape = 49 := by
  sorry

end composite_cubes_surface_area_l1761_176106


namespace abs_x_squared_lt_x_solution_set_l1761_176146

theorem abs_x_squared_lt_x_solution_set :
  {x : ℝ | |x| * |x| < x} = {x : ℝ | (0 < x ∧ x < 1) ∨ x < -1} := by sorry

end abs_x_squared_lt_x_solution_set_l1761_176146


namespace thirteen_binary_l1761_176170

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Checks if a list of booleans represents a given natural number in binary -/
def is_binary_rep (n : ℕ) (bits : List Bool) : Prop :=
  to_binary n = bits

theorem thirteen_binary :
  is_binary_rep 13 [true, false, true, true] := by sorry

end thirteen_binary_l1761_176170


namespace car_speed_problem_l1761_176163

theorem car_speed_problem (average_speed : ℝ) (first_hour_speed : ℝ) (total_time : ℝ) :
  average_speed = 65 →
  first_hour_speed = 100 →
  total_time = 2 →
  (average_speed * total_time - first_hour_speed) = 30 :=
by
  sorry

end car_speed_problem_l1761_176163


namespace alcohol_mixture_concentration_l1761_176136

theorem alcohol_mixture_concentration
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid_poured : ℝ)
  (final_vessel_capacity : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 25)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percentage = 50)
  (h5 : total_liquid_poured = 8)
  (h6 : final_vessel_capacity = 10)
  : (vessel1_capacity * vessel1_alcohol_percentage / 100 +
     vessel2_capacity * vessel2_alcohol_percentage / 100) /
    final_vessel_capacity * 100 = 35 := by
  sorry

end alcohol_mixture_concentration_l1761_176136


namespace closest_point_on_line_l1761_176173

/-- The point on the line y = 3x - 1 that is closest to (1,4) is (-3/5, -4/5) -/
theorem closest_point_on_line (x y : ℝ) : 
  y = 3 * x - 1 → 
  (x - (-3/5))^2 + (y - (-4/5))^2 ≤ (x - 1)^2 + (y - 4)^2 :=
by sorry

end closest_point_on_line_l1761_176173


namespace cats_vasyas_equality_l1761_176119

variable {α : Type*}
variable (C V : Set α)

theorem cats_vasyas_equality : C ∩ V = V ∩ C := by
  sorry

end cats_vasyas_equality_l1761_176119


namespace transformation_correctness_l1761_176152

theorem transformation_correctness (a b : ℝ) (h : a > b) : 1 + 2*a > 1 + 2*b := by
  sorry

end transformation_correctness_l1761_176152


namespace helen_cookies_proof_l1761_176160

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 31

/-- The number of cookies Helen baked this morning -/
def cookies_this_morning : ℕ := 270

/-- The total number of cookies Helen baked till last night -/
def total_cookies : ℕ := 450

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_before_yesterday : ℕ := total_cookies - (cookies_yesterday + cookies_this_morning)

theorem helen_cookies_proof : 
  cookies_before_yesterday = 149 := by sorry

end helen_cookies_proof_l1761_176160


namespace sqrt_225_range_l1761_176130

theorem sqrt_225_range : 15 < Real.sqrt 225 ∧ Real.sqrt 225 < 16 := by
  sorry

end sqrt_225_range_l1761_176130


namespace olivias_wallet_problem_l1761_176178

/-- The initial amount of money in Olivia's wallet -/
def initial_money : ℕ := 100

/-- The amount of money Olivia collected from the ATM -/
def atm_money : ℕ := 148

/-- The amount of money Olivia spent at the supermarket -/
def spent_money : ℕ := 89

/-- The amount of money left after visiting the supermarket -/
def remaining_money : ℕ := 159

theorem olivias_wallet_problem :
  initial_money + atm_money = remaining_money + spent_money :=
by sorry

end olivias_wallet_problem_l1761_176178


namespace rainwater_farm_problem_l1761_176166

/-- Mr. Rainwater's farm animals problem -/
theorem rainwater_farm_problem (goats cows chickens : ℕ) : 
  goats = 4 * cows →
  goats = 2 * chickens →
  chickens = 18 →
  cows = 9 := by
sorry

end rainwater_farm_problem_l1761_176166


namespace root_product_expression_l1761_176108

theorem root_product_expression (p q : ℝ) 
  (hα : ∃ α : ℝ, α^2 + p*α + 2 = 0)
  (hβ : ∃ β : ℝ, β^2 + p*β + 2 = 0)
  (hγ : ∃ γ : ℝ, γ^2 + q*γ - 3 = 0)
  (hδ : ∃ δ : ℝ, δ^2 + q*δ - 3 = 0)
  (hαβ_distinct : ∀ α β : ℝ, α^2 + p*α + 2 = 0 → β^2 + p*β + 2 = 0 → α ≠ β)
  (hγδ_distinct : ∀ γ δ : ℝ, γ^2 + q*γ - 3 = 0 → δ^2 + q*δ - 3 = 0 → γ ≠ δ) :
  ∃ (α β γ δ : ℝ), (α - γ)*(β - γ)*(α + δ)*(β + δ) = 3*(q^2 - p^2) + 15 :=
by sorry

end root_product_expression_l1761_176108


namespace construction_materials_cost_l1761_176162

/-- Calculates the total amount paid for construction materials --/
def total_amount_paid (cement_bags : ℕ) (cement_price : ℚ) (cement_discount : ℚ)
                      (sand_lorries : ℕ) (sand_tons_per_lorry : ℕ) (sand_price_per_ton : ℚ)
                      (sand_tax : ℚ) : ℚ :=
  let cement_cost := cement_bags * cement_price
  let cement_discount_amount := cement_cost * cement_discount
  let cement_total := cement_cost - cement_discount_amount
  let sand_tons := sand_lorries * sand_tons_per_lorry
  let sand_cost := sand_tons * sand_price_per_ton
  let sand_tax_amount := sand_cost * sand_tax
  let sand_total := sand_cost + sand_tax_amount
  cement_total + sand_total

/-- The total amount paid for construction materials is $13,310 --/
theorem construction_materials_cost :
  total_amount_paid 500 10 (5/100) 20 10 40 (7/100) = 13310 := by
  sorry

end construction_materials_cost_l1761_176162


namespace john_popcorn_profit_l1761_176158

/-- Calculates the profit for John's popcorn business --/
theorem john_popcorn_profit :
  let regular_price : ℚ := 4
  let discount_rate : ℚ := 0.1
  let adult_price : ℚ := 8
  let child_price : ℚ := 6
  let packaging_cost : ℚ := 0.5
  let transport_fee : ℚ := 10
  let adult_bags : ℕ := 20
  let child_bags : ℕ := 10
  let total_bags : ℕ := adult_bags + child_bags
  let discounted_price : ℚ := regular_price * (1 - discount_rate)
  let total_cost : ℚ := discounted_price * total_bags + packaging_cost * total_bags + transport_fee
  let total_revenue : ℚ := adult_price * adult_bags + child_price * child_bags
  let profit : ℚ := total_revenue - total_cost
  profit = 87 := by
    sorry

end john_popcorn_profit_l1761_176158


namespace num_routes_eq_expected_l1761_176117

/-- Represents the number of southern cities -/
def num_southern_cities : ℕ := 4

/-- Represents the number of northern cities -/
def num_northern_cities : ℕ := 5

/-- Calculates the number of different routes for a traveler -/
def num_routes : ℕ := (Nat.factorial (num_southern_cities - 1)) * (num_northern_cities ^ num_southern_cities)

/-- Theorem stating that the number of routes is equal to 3! × 5^4 -/
theorem num_routes_eq_expected : num_routes = 3750 := by
  sorry

end num_routes_eq_expected_l1761_176117


namespace vector_magnitude_problem_l1761_176100

theorem vector_magnitude_problem (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 5) →
  ((a.1 - b.1)^2 + (a.2 - b.2)^2 = 20) →
  (b.1^2 + b.2^2 = 25) :=
by sorry

end vector_magnitude_problem_l1761_176100


namespace injective_function_equation_l1761_176150

theorem injective_function_equation (f : ℝ → ℝ) (h_inj : Function.Injective f) :
  (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y)) →
  ∀ x : ℝ, f x = x := by
  sorry

end injective_function_equation_l1761_176150


namespace solution_set_inequality_l1761_176154

theorem solution_set_inequality (x : ℝ) : 
  (2 * x - 3) / (x + 2) ≤ 1 ↔ -2 < x ∧ x ≤ 5 := by
  sorry

end solution_set_inequality_l1761_176154


namespace unpainted_cubes_in_6x6x6_l1761_176159

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_unit_cubes : Nat
  painted_grid_size : Nat

/-- Calculates the number of unpainted unit cubes in the painted cube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem unpainted_cubes_in_6x6x6 :
  let cube : PaintedCube := {
    size := 6,
    total_unit_cubes := 216,
    painted_grid_size := 4
  }
  unpainted_cubes cube = 176 := by
  sorry

end unpainted_cubes_in_6x6x6_l1761_176159


namespace count_not_divisible_1200_l1761_176155

def count_not_divisible (n : ℕ) : ℕ :=
  (n - 1) - ((n - 1) / 6 + (n - 1) / 8 - (n - 1) / 24)

theorem count_not_divisible_1200 :
  count_not_divisible 1200 = 900 := by
  sorry

end count_not_divisible_1200_l1761_176155


namespace unique_solution_cube_equation_l1761_176148

theorem unique_solution_cube_equation :
  ∀ (x y z : ℤ), x^3 + 2*y^3 = 4*z^3 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end unique_solution_cube_equation_l1761_176148


namespace parabola_shift_left_2_l1761_176120

/-- Represents a parabola in the form y = (x - h)^2 + k, where (h, k) is the vertex -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The equation of a parabola given x -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (shift : ℝ) : Parabola :=
  { h := p.h + shift, k := p.k }

theorem parabola_shift_left_2 :
  let original := Parabola.mk 0 0
  let shifted := shift_parabola original (-2)
  ∀ x, parabola_equation shifted x = (x + 2)^2 := by
  sorry

end parabola_shift_left_2_l1761_176120


namespace set_operations_l1761_176121

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {1,2,5}

theorem set_operations :
  (A ∩ B = {2,5}) ∧ (A ∪ (U \ B) = {2,3,4,5,6}) := by sorry

end set_operations_l1761_176121


namespace stock_price_increase_probability_l1761_176118

/-- Probability of stock price increase given interest rate conditions -/
theorem stock_price_increase_probability
  (p_increase_when_lowered : ℝ)
  (p_increase_when_unchanged : ℝ)
  (p_increase_when_raised : ℝ)
  (p_rate_reduction : ℝ)
  (p_rate_unchanged : ℝ)
  (h1 : p_increase_when_lowered = 0.7)
  (h2 : p_increase_when_unchanged = 0.2)
  (h3 : p_increase_when_raised = 0.1)
  (h4 : p_rate_reduction = 0.6)
  (h5 : p_rate_unchanged = 0.3)
  (h6 : p_rate_reduction + p_rate_unchanged + (1 - p_rate_reduction - p_rate_unchanged) = 1) :
  p_rate_reduction * p_increase_when_lowered +
  p_rate_unchanged * p_increase_when_unchanged +
  (1 - p_rate_reduction - p_rate_unchanged) * p_increase_when_raised = 0.49 := by
  sorry


end stock_price_increase_probability_l1761_176118


namespace largest_three_digit_base7_decimal_l1761_176187

/-- The largest three-digit number in base 7 -/
def largest_base7 : ℕ := 666

/-- Conversion from base 7 to decimal -/
def base7_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- Theorem stating that the largest decimal number represented by a three-digit base-7 number is 342 -/
theorem largest_three_digit_base7_decimal :
  base7_to_decimal largest_base7 = 342 := by sorry

end largest_three_digit_base7_decimal_l1761_176187


namespace range_of_m_l1761_176180

/-- Given conditions p and q, prove that m ∈ [4, +∞) -/
theorem range_of_m (x m : ℝ) : 
  (∀ x, x^2 - 3*x - 4 ≤ 0 → |x - 3| ≤ m) ∧ 
  (∃ x, |x - 3| ≤ m ∧ x^2 - 3*x - 4 > 0) →
  m ≥ 4 := by
sorry

end range_of_m_l1761_176180


namespace least_reducible_fraction_l1761_176149

def is_reducible (n : ℕ) : Prop :=
  (n > 15) ∧ (Nat.gcd (n - 15) (3 * n + 4) > 1)

theorem least_reducible_fraction :
  ∀ k : ℕ, k < 22 → ¬(is_reducible k) ∧ is_reducible 22 :=
sorry

end least_reducible_fraction_l1761_176149


namespace simplify_fraction_l1761_176139

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end simplify_fraction_l1761_176139


namespace friends_seinfeld_relationship_l1761_176157

-- Define the variables
variable (x y z : ℚ)

-- Define the conditions
def friends_episodes : ℚ := 50
def seinfeld_episodes : ℚ := 75

-- State the theorem
theorem friends_seinfeld_relationship 
  (h1 : x * z = friends_episodes) 
  (h2 : y * z = seinfeld_episodes) :
  y = 1.5 * x := by
  sorry

end friends_seinfeld_relationship_l1761_176157


namespace equation_solution_l1761_176183

theorem equation_solution : 
  ∃! (x : ℝ), x > 0 ∧ (1/2) * (3*x^2 - 1) = (x^2 - 50*x - 10) * (x^2 + 25*x + 5) ∧ x = 25 + 2 * Real.sqrt 159 := by
  sorry

end equation_solution_l1761_176183


namespace math_problem_l1761_176177

theorem math_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ (h : a * b - a - 2 * b = 0), a + 2 * b ≥ 8) ∧
  (a^2 / b + b^2 / a ≥ a + b) ∧
  (∀ (h : 1 / (a + 1) + 1 / (b + 2) = 1 / 3), a * b + a + b ≥ 14 + 6 * Real.sqrt 6) :=
by sorry

end math_problem_l1761_176177


namespace building_E_floors_l1761_176189

/-- The number of floors in Building A -/
def floors_A : ℕ := 4

/-- The number of floors in Building B -/
def floors_B : ℕ := floors_A + 9

/-- The number of floors in Building C -/
def floors_C : ℕ := 5 * floors_B - 6

/-- The number of floors in Building D -/
def floors_D : ℕ := 2 * floors_C - (floors_A + floors_B)

/-- The number of floors in Building E -/
def floors_E : ℕ := 3 * (floors_B + floors_C + floors_D) - 10

/-- Theorem stating that Building E has 509 floors -/
theorem building_E_floors : floors_E = 509 := by
  sorry

end building_E_floors_l1761_176189


namespace cube_surface_area_equal_volume_l1761_176101

/-- The surface area of a cube with the same volume as a rectangular prism of dimensions 10 inches by 5 inches by 20 inches is 600 square inches. -/
theorem cube_surface_area_equal_volume (prism_length prism_width prism_height : ℝ)
  (h1 : prism_length = 10)
  (h2 : prism_width = 5)
  (h3 : prism_height = 20) :
  (6 : ℝ) * ((prism_length * prism_width * prism_height) ^ (1/3 : ℝ))^2 = 600 := by
  sorry

end cube_surface_area_equal_volume_l1761_176101


namespace power_three_nineteen_mod_ten_l1761_176138

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by
  sorry

end power_three_nineteen_mod_ten_l1761_176138


namespace f_properties_l1761_176175

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 3 * 2^(-x)
  else if x < 0 then 3 * 2^x - 2^(-x)
  else 0

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x < 0, f x = 3 * 2^x - 2^(-x)) ∧  -- f(x) for x < 0
  f 1 = 1/2 := by sorry

end f_properties_l1761_176175


namespace symmetry_axes_count_cube_symmetry_axes_count_tetrahedron_symmetry_axes_count_l1761_176196

/-- The number of axes of symmetry in a cube -/
def cube_symmetry_axes : ℕ := 13

/-- The number of axes of symmetry in a regular tetrahedron -/
def tetrahedron_symmetry_axes : ℕ := 7

/-- Theorem stating the number of axes of symmetry for a cube and a regular tetrahedron -/
theorem symmetry_axes_count :
  (cube_symmetry_axes = 13) ∧ (tetrahedron_symmetry_axes = 7) := by
  sorry

/-- Theorem for the number of axes of symmetry in a cube -/
theorem cube_symmetry_axes_count : cube_symmetry_axes = 13 := by
  sorry

/-- Theorem for the number of axes of symmetry in a regular tetrahedron -/
theorem tetrahedron_symmetry_axes_count : tetrahedron_symmetry_axes = 7 := by
  sorry

end symmetry_axes_count_cube_symmetry_axes_count_tetrahedron_symmetry_axes_count_l1761_176196


namespace x_value_l1761_176199

theorem x_value (x y : ℝ) (h : x / (x - 1) = (y^2 + 3*y - 5) / (y^2 + 3*y - 7)) :
  x = (y^2 + 3*y - 5) / 2 := by
  sorry

end x_value_l1761_176199


namespace g_502_solutions_l1761_176191

-- Define g₁
def g₁ (x : ℚ) : ℚ := 1/2 - 4/(4*x + 2)

-- Define gₙ recursively
def g (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | 1 => g₁ x
  | n+1 => g₁ (g n x)

-- Theorem statement
theorem g_502_solutions (x : ℚ) : 
  g 502 x = x - 2 ↔ x = 115/64 ∨ x = 51/64 := by sorry

end g_502_solutions_l1761_176191


namespace julia_tag_monday_l1761_176109

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 14

/-- The additional number of kids Julia played tag with on Monday compared to Tuesday -/
def additional_monday_kids : ℕ := 8

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := tuesday_kids + additional_monday_kids

theorem julia_tag_monday : monday_kids = 22 := by
  sorry

end julia_tag_monday_l1761_176109


namespace total_flour_used_l1761_176131

-- Define the amount of wheat flour used
def wheat_flour : ℝ := 0.2

-- Define the amount of white flour used
def white_flour : ℝ := 0.1

-- Theorem stating the total amount of flour used
theorem total_flour_used : wheat_flour + white_flour = 0.3 := by
  sorry

end total_flour_used_l1761_176131


namespace can_achieve_any_coloring_can_achieve_checkerboard_l1761_176182

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a square -/
inductive Color
  | White
  | Black

/-- Represents the state of the chessboard -/
def Board := Square → Color

/-- Represents a move that changes the color of squares in a row and column -/
structure Move where
  row : Fin 8
  col : Fin 8

/-- Applies a move to a board, changing colors in the specified row and column -/
def applyMove (b : Board) (m : Move) : Board :=
  fun s => if s.row = m.row || s.col = m.col then
             match b s with
             | Color.White => Color.Black
             | Color.Black => Color.White
           else b s

/-- The initial all-white board -/
def initialBoard : Board := fun _ => Color.White

/-- The standard checkerboard pattern -/
def checkerboardPattern : Board :=
  fun s => if (s.row.val + s.col.val) % 2 = 0 then Color.White else Color.Black

/-- Theorem stating that any desired board coloring can be achieved -/
theorem can_achieve_any_coloring :
  ∀ (targetBoard : Board), ∃ (moves : List Move),
    (moves.foldl applyMove initialBoard) = targetBoard :=
  sorry

/-- Corollary stating that the standard checkerboard pattern can be achieved -/
theorem can_achieve_checkerboard :
  ∃ (moves : List Move),
    (moves.foldl applyMove initialBoard) = checkerboardPattern :=
  sorry

end can_achieve_any_coloring_can_achieve_checkerboard_l1761_176182


namespace smarties_remainder_l1761_176151

theorem smarties_remainder (n : ℕ) (h : n % 11 = 8) : (2 * n) % 11 = 5 := by
  sorry

end smarties_remainder_l1761_176151


namespace longer_base_length_l1761_176167

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithCircle where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Length of the shorter base -/
  short_base : ℝ
  /-- Length of the longer base -/
  long_base : ℝ
  /-- The circle is inscribed in the trapezoid -/
  inscribed : r > 0
  /-- The trapezoid is a right trapezoid -/
  right_angled : True
  /-- The shorter base is positive -/
  short_base_positive : short_base > 0
  /-- The longer base is longer than the shorter base -/
  base_inequality : long_base > short_base

/-- Theorem: The longer base of the trapezoid is 12 units -/
theorem longer_base_length (t : RightTrapezoidWithCircle) 
  (h1 : t.r = 3) 
  (h2 : t.short_base = 4) : 
  t.long_base = 12 := by
  sorry

end longer_base_length_l1761_176167


namespace rectangle_perimeter_area_inequality_l1761_176110

theorem rectangle_perimeter_area_inequality (l S : ℝ) (hl : l > 0) (hS : S > 0) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ l = 2 * (a + b) ∧ S = a * b) → l^2 ≥ 16 * S :=
by sorry

#check rectangle_perimeter_area_inequality

end rectangle_perimeter_area_inequality_l1761_176110


namespace min_value_theorem_l1761_176147

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  (x + 6) / Real.sqrt (x - 2) ≥ 4 * Real.sqrt 2 ∧
  ((x + 6) / Real.sqrt (x - 2) = 4 * Real.sqrt 2 ↔ x = 10) := by
  sorry

end min_value_theorem_l1761_176147


namespace difference_divisible_by_nine_l1761_176174

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem difference_divisible_by_nine (N : ℕ) :
  ∃ k : ℤ, N - (sum_of_digits N) = 9 * k := by
  sorry

end difference_divisible_by_nine_l1761_176174


namespace sal_and_phil_combined_money_l1761_176112

/-- Given that Kim has 40% more money than Sal, Sal has 20% less money than Phil,
    and Kim has $1.12, prove that Sal and Phil have a combined total of $1.80. -/
theorem sal_and_phil_combined_money :
  ∀ (kim sal phil : ℝ),
  kim = 1.4 * sal →
  sal = 0.8 * phil →
  kim = 1.12 →
  sal + phil = 1.80 :=
by
  sorry

end sal_and_phil_combined_money_l1761_176112


namespace stratified_sample_size_l1761_176176

-- Define the total number of male and female athletes
def total_male : ℕ := 42
def total_female : ℕ := 30

-- Define the number of female athletes in the sample
def sampled_female : ℕ := 5

-- Theorem statement
theorem stratified_sample_size :
  ∃ (n : ℕ), 
    -- The sample size is the sum of sampled males and females
    n = (total_male * sampled_female / total_female) + sampled_female ∧
    -- The sample maintains the same ratio as the population
    n * total_female = (total_male + total_female) * sampled_female ∧
    -- The sample size is 12
    n = 12 :=
by
  sorry

end stratified_sample_size_l1761_176176


namespace last_digit_of_large_prime_l1761_176195

theorem last_digit_of_large_prime (h : 859433 = 214858 * 4 + 1) :
  (2^859433 - 1) % 10 = 1 := by
  sorry

end last_digit_of_large_prime_l1761_176195


namespace ab_equals_e_cubed_l1761_176145

theorem ab_equals_e_cubed (a b : ℝ) (h1 : Real.exp (2 - a) = a) (h2 : b * (Real.log b - 1) = Real.exp 3) : a * b = Real.exp 3 := by
  sorry

end ab_equals_e_cubed_l1761_176145


namespace longest_all_green_interval_is_20_seconds_l1761_176113

/-- Represents a traffic light with its timing properties -/
structure TrafficLight where
  greenDuration : ℝ
  yellowDuration : ℝ
  redDuration : ℝ
  cycleStart : ℝ

/-- Calculates the longest interval during which all lights are green -/
def longestAllGreenInterval (lights : List TrafficLight) : ℝ :=
  sorry

/-- The main theorem stating the longest interval of all green lights -/
theorem longest_all_green_interval_is_20_seconds :
  let lights : List TrafficLight := List.range 8 |>.map (fun i =>
    { greenDuration := 90  -- 1.5 minutes in seconds
      yellowDuration := 3
      redDuration := 90    -- 1.5 minutes in seconds
      cycleStart := i * 10 -- Each light starts 10 seconds after the previous
    })
  longestAllGreenInterval lights = 20 := by
  sorry

end longest_all_green_interval_is_20_seconds_l1761_176113


namespace max_grandchildren_l1761_176142

/-- The number of grandchildren for a person with given children and grandchildren distribution -/
def grandchildren_count (num_children : ℕ) (num_children_with_same : ℕ) (num_grandchildren_same : ℕ) (num_children_different : ℕ) (num_grandchildren_different : ℕ) : ℕ :=
  (num_children_with_same * num_grandchildren_same) + (num_children_different * num_grandchildren_different)

/-- Theorem stating that Max has 58 grandchildren -/
theorem max_grandchildren :
  grandchildren_count 8 6 8 2 5 = 58 := by
  sorry

end max_grandchildren_l1761_176142


namespace x_power_2048_minus_reciprocal_l1761_176164

theorem x_power_2048_minus_reciprocal (x : ℝ) (h : x - 1/x = Real.sqrt 3) :
  x^2048 - 1/x^2048 = 277526 := by
  sorry

end x_power_2048_minus_reciprocal_l1761_176164


namespace february_first_day_l1761_176111

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a given number of days
def advanceDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => nextDay (advanceDays start n)

-- Theorem statement
theorem february_first_day (january_first : DayOfWeek) 
  (h : january_first = DayOfWeek.Monday) : 
  advanceDays january_first 31 = DayOfWeek.Thursday := by
  sorry


end february_first_day_l1761_176111


namespace triathlete_average_speed_l1761_176135

/-- Calculates the average speed for a triathlete's swimming and running events,
    assuming equal distances for both activities. -/
theorem triathlete_average_speed
  (swim_speed : ℝ)
  (run_speed : ℝ)
  (h1 : swim_speed = 1)
  (h2 : run_speed = 7) :
  (2 * swim_speed * run_speed) / (swim_speed + run_speed) = 1.75 := by
  sorry

#check triathlete_average_speed

end triathlete_average_speed_l1761_176135


namespace nearest_integer_to_three_plus_sqrt_five_fourth_power_l1761_176125

theorem nearest_integer_to_three_plus_sqrt_five_fourth_power :
  ∃ (n : ℤ), ∀ (m : ℤ), |((3 : ℝ) + Real.sqrt 5)^4 - n| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - m| ∧ n = 752 :=
sorry

end nearest_integer_to_three_plus_sqrt_five_fourth_power_l1761_176125


namespace pat_to_kate_ratio_l1761_176188

-- Define the variables
def total_hours : ℕ := 117
def mark_extra_hours : ℕ := 65

-- Define the hours charged by each person as real numbers
variable (pat_hours kate_hours mark_hours : ℝ)

-- Define the conditions
axiom total_hours_sum : pat_hours + kate_hours + mark_hours = total_hours
axiom pat_to_mark_ratio : pat_hours = (1/3) * mark_hours
axiom mark_to_kate_diff : mark_hours = kate_hours + mark_extra_hours

-- Define the theorem
theorem pat_to_kate_ratio :
  (∃ r : ℝ, pat_hours = r * kate_hours) →
  pat_hours / kate_hours = 2 := by sorry

end pat_to_kate_ratio_l1761_176188


namespace cube_root_of_negative_27_l1761_176107

theorem cube_root_of_negative_27 :
  let S : Set ℂ := {z : ℂ | z^3 = -27}
  S = {-3, (3/2 : ℂ) + (3*Complex.I*Real.sqrt 3)/2, (3/2 : ℂ) - (3*Complex.I*Real.sqrt 3)/2} := by
  sorry

end cube_root_of_negative_27_l1761_176107


namespace minimum_correct_answers_l1761_176133

theorem minimum_correct_answers (total_questions : ℕ) 
  (correct_points : ℕ) (incorrect_points : ℤ) (min_score : ℕ) :
  total_questions = 20 →
  correct_points = 10 →
  incorrect_points = -5 →
  min_score = 120 →
  (∃ x : ℕ, x * correct_points + (total_questions - x) * incorrect_points > min_score ∧
    ∀ y : ℕ, y < x → y * correct_points + (total_questions - y) * incorrect_points ≤ min_score) →
  (∃ x : ℕ, x * correct_points + (total_questions - x) * incorrect_points > min_score ∧
    ∀ y : ℕ, y < x → y * correct_points + (total_questions - y) * incorrect_points ≤ min_score) →
  x = 15 :=
by sorry

end minimum_correct_answers_l1761_176133


namespace girl_speed_l1761_176116

/-- Given a girl traveling a distance of 96 meters in 16 seconds,
    prove that her speed is 6 meters per second. -/
theorem girl_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 96) 
    (h2 : time = 16) 
    (h3 : speed = distance / time) : 
  speed = 6 := by
  sorry

end girl_speed_l1761_176116


namespace x_equals_one_l1761_176169

theorem x_equals_one (x y : ℝ) (h1 : x + 3 * y = 10) (h2 : y = 3) : x = 1 := by
  sorry

end x_equals_one_l1761_176169


namespace polynomial_division_theorem_l1761_176179

theorem polynomial_division_theorem (x : ℝ) :
  ∃ (q r : ℝ), 5*x^3 - 4*x^2 + 6*x - 9 = (x - 1) * (5*x^2 + x + 7) + r ∧ r = -2 := by
  sorry

end polynomial_division_theorem_l1761_176179


namespace min_value_x_plus_4y_l1761_176115

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2*x*y) :
  x + 4*y ≥ 9/2 ∧ (x + 4*y = 9/2 ↔ x = 3/2 ∧ y = 3/4) := by
  sorry

end min_value_x_plus_4y_l1761_176115


namespace least_subtraction_for_divisibility_l1761_176102

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(12 ∣ (427398 - y))) ∧
  (12 ∣ (427398 - x)) := by
sorry

end least_subtraction_for_divisibility_l1761_176102


namespace simplify_fraction_l1761_176143

theorem simplify_fraction (a : ℚ) (h : a = -2) : 18 * a^5 / (27 * a^3) = 8/3 := by
  sorry

end simplify_fraction_l1761_176143


namespace johns_allowance_is_150_cents_l1761_176105

def johns_allowance (A : ℚ) : Prop :=
  let arcade_spent : ℚ := 3 / 5 * A
  let remaining_after_arcade : ℚ := A - arcade_spent
  let toy_store_spent : ℚ := 1 / 3 * remaining_after_arcade
  let remaining_after_toy_store : ℚ := remaining_after_arcade - toy_store_spent
  remaining_after_toy_store = 40 / 100

theorem johns_allowance_is_150_cents :
  ∃ A : ℚ, johns_allowance A ∧ A = 150 / 100 :=
sorry

end johns_allowance_is_150_cents_l1761_176105


namespace books_read_per_year_l1761_176134

/-- The total number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  84 * c * s

/-- Theorem: The total number of books read by the entire student body in one year
    is equal to 84 * c * s, given the conditions of the reading program -/
theorem books_read_per_year (c s : ℕ) (books_per_month : ℕ) (months_per_year : ℕ)
    (h1 : books_per_month = 7)
    (h2 : months_per_year = 12)
    (h3 : c > 0)
    (h4 : s > 0) :
    total_books_read c s = books_per_month * months_per_year * c * s :=
  sorry

end books_read_per_year_l1761_176134


namespace left_handed_jazz_lovers_count_l1761_176156

/-- Represents a club with members having different characteristics -/
structure Club where
  total : Nat
  leftHanded : Nat
  jazzLovers : Nat
  rightHandedNonJazz : Nat

/-- The number of left-handed jazz lovers in the club -/
def leftHandedJazzLovers (c : Club) : Nat :=
  c.leftHanded + c.jazzLovers - (c.total - c.rightHandedNonJazz)

/-- Theorem stating the number of left-handed jazz lovers in the specific club -/
theorem left_handed_jazz_lovers_count (c : Club) 
  (h1 : c.total = 30)
  (h2 : c.leftHanded = 12)
  (h3 : c.jazzLovers = 20)
  (h4 : c.rightHandedNonJazz = 3) :
  leftHandedJazzLovers c = 5 := by
  sorry

#check left_handed_jazz_lovers_count

end left_handed_jazz_lovers_count_l1761_176156


namespace logo_shaded_area_l1761_176192

/-- The shaded area of a logo design with a square containing four larger circles and one smaller circle -/
theorem logo_shaded_area (square_side : ℝ) (large_circle_radius : ℝ) (small_circle_radius : ℝ) : 
  square_side = 24 →
  large_circle_radius = 6 →
  small_circle_radius = 3 →
  (square_side ^ 2) - (4 * Real.pi * large_circle_radius ^ 2) - (Real.pi * small_circle_radius ^ 2) = 576 - 153 * Real.pi :=
by sorry

end logo_shaded_area_l1761_176192


namespace press_conference_seating_l1761_176181

/-- Represents the number of ways to seat players from different teams -/
def seating_arrangements (cubs : Nat) (red_sox : Nat) : Nat :=
  2 * 2 * (Nat.factorial cubs) * (Nat.factorial red_sox)

/-- Theorem stating the number of seating arrangements for the given conditions -/
theorem press_conference_seating :
  seating_arrangements 4 3 = 576 :=
by sorry

end press_conference_seating_l1761_176181


namespace sample_volume_calculation_l1761_176153

theorem sample_volume_calculation (m : ℝ) 
  (h1 : m > 0)  -- Ensure m is positive
  (h2 : 8 / m + 0.15 + 0.45 = 1) : m = 20 := by
  sorry

end sample_volume_calculation_l1761_176153


namespace max_value_of_expression_l1761_176194

theorem max_value_of_expression (x y : ℝ) (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : x + y = 8) :
  (∀ a b : ℝ, a ≥ 1 → b ≥ 1 → a + b = 8 → 
    |Real.sqrt (x - 1/y) + Real.sqrt (y - 1/x)| ≥ |Real.sqrt (a - 1/b) + Real.sqrt (b - 1/a)|) ∧
  |Real.sqrt (x - 1/y) + Real.sqrt (y - 1/x)| ≤ Real.sqrt 15 :=
by sorry

end max_value_of_expression_l1761_176194


namespace polynomial_simplification_l1761_176123

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^2 + 6 * x - 4) - (2 * x^2 + 3 * x - 15) = x^2 + 3 * x + 11 := by
  sorry

end polynomial_simplification_l1761_176123


namespace solution_set_inequality_l1761_176186

theorem solution_set_inequality (x : ℝ) : (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 := by
  sorry

end solution_set_inequality_l1761_176186


namespace prob_jack_and_jill_selected_l1761_176184

/-- The probability of Jack being selected for the interview. -/
def prob_jack : ℝ := 0.20

/-- The probability of Jill being selected for the interview. -/
def prob_jill : ℝ := 0.15

/-- The number of workers in the hospital. -/
def num_workers : ℕ := 8

/-- The number of workers to be interviewed. -/
def num_interviewed : ℕ := 2

/-- Assumption that the selection of Jack and Jill are independent events. -/
axiom selection_independent : True

theorem prob_jack_and_jill_selected : 
  prob_jack * prob_jill = 0.03 := by sorry

end prob_jack_and_jill_selected_l1761_176184


namespace nancy_future_games_l1761_176144

/-- The number of games Nancy plans to attend next month -/
def games_next_month (games_this_month games_last_month total_games : ℕ) : ℕ :=
  total_games - (games_this_month + games_last_month)

/-- Proof that Nancy plans to attend 7 games next month -/
theorem nancy_future_games : games_next_month 9 8 24 = 7 := by
  sorry

end nancy_future_games_l1761_176144


namespace gracie_is_56_inches_tall_l1761_176197

/-- Gracie's height in inches -/
def gracies_height : ℕ := 56

/-- Theorem stating Gracie's height is 56 inches -/
theorem gracie_is_56_inches_tall : gracies_height = 56 := by
  sorry

end gracie_is_56_inches_tall_l1761_176197


namespace intersection_of_A_and_B_l1761_176127

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l1761_176127


namespace three_diamonds_balance_two_circles_l1761_176168

/-- Represents the balance of symbols in the problem -/
structure Balance where
  triangle : ℕ  -- Δ
  diamond : ℕ   -- ◊
  circle : ℕ    -- •

/-- First balance equation: 4Δ + 2◊ = 12• -/
def balance_equation1 (b : Balance) : Prop :=
  4 * b.triangle + 2 * b.diamond = 12 * b.circle

/-- Second balance equation: Δ = ◊ + 2• -/
def balance_equation2 (b : Balance) : Prop :=
  b.triangle = b.diamond + 2 * b.circle

/-- Theorem stating that 3◊ balances 2• -/
theorem three_diamonds_balance_two_circles (b : Balance) 
  (h1 : balance_equation1 b) (h2 : balance_equation2 b) : 
  3 * b.diamond = 2 * b.circle :=
sorry

end three_diamonds_balance_two_circles_l1761_176168


namespace square_equation_solution_l1761_176122

theorem square_equation_solution : ∃! (M : ℕ), M > 0 ∧ 14^2 * 35^2 = 70^2 * M^2 := by sorry

end square_equation_solution_l1761_176122


namespace correct_bottles_calculation_l1761_176190

/-- Given that B bottles of water can be purchased for P pennies,
    and 1 euro is worth 100 pennies, this function calculates
    the number of bottles that can be purchased for E euros. -/
def bottles_per_euro (B P E : ℚ) : ℚ :=
  (100 * E * B) / P

/-- Theorem stating that the number of bottles that can be purchased
    for E euros is (100 * E * B) / P, given the conditions. -/
theorem correct_bottles_calculation (B P E : ℚ) (hB : B > 0) (hP : P > 0) (hE : E > 0) :
  bottles_per_euro B P E = (100 * E * B) / P :=
by sorry

end correct_bottles_calculation_l1761_176190


namespace max_prime_factors_l1761_176128

theorem max_prime_factors (a b : ℕ+) 
  (h_gcd : (Finset.card (Nat.primeFactors (Nat.gcd a b))) = 8)
  (h_lcm : (Finset.card (Nat.primeFactors (Nat.lcm a b))) = 30)
  (h_fewer : (Finset.card (Nat.primeFactors a)) < (Finset.card (Nat.primeFactors b))) :
  (Finset.card (Nat.primeFactors a)) ≤ 19 := by
  sorry

end max_prime_factors_l1761_176128


namespace distance_between_complex_points_l1761_176161

theorem distance_between_complex_points :
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 17 := by
sorry

end distance_between_complex_points_l1761_176161


namespace parabola_circle_equation_l1761_176104

/-- The equation of a circle with center at the focus of a parabola and diameter
    equal to the line segment formed by the intersection of the parabola with a
    line perpendicular to the x-axis passing through the focus. -/
theorem parabola_circle_equation (x y : ℝ) : 
  let parabola := {(x, y) | y^2 = 4*x}
  let focus := (1, 0)
  let perpendicular_line := {(x, y) | x = 1}
  let intersection := parabola ∩ perpendicular_line
  true → (x - 1)^2 + y^2 = 4 := by
  sorry

end parabola_circle_equation_l1761_176104


namespace sin_75_cos_15_minus_1_l1761_176114

theorem sin_75_cos_15_minus_1 : 
  2 * Real.sin (75 * π / 180) * Real.cos (15 * π / 180) - 1 = Real.sqrt 3 / 2 := by
  sorry

end sin_75_cos_15_minus_1_l1761_176114


namespace polynomial_remainder_theorem_l1761_176141

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^14 - 1) % (x + 1) = 0 := by
  sorry

end polynomial_remainder_theorem_l1761_176141


namespace half_coverage_days_l1761_176171

/-- Represents the number of days it takes for the lily pad patch to cover the entire lake -/
def full_coverage_days : ℕ := 58

/-- Represents the growth factor of the lily pad patch per day -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that the number of days to cover half the lake is one less than the number of days to cover the full lake -/
theorem half_coverage_days : 
  ∃ (half_days : ℕ), half_days = full_coverage_days - 1 ∧ 
  (daily_growth_factor ^ half_days) * 2 = daily_growth_factor ^ full_coverage_days :=
sorry

end half_coverage_days_l1761_176171


namespace min_length_GH_l1761_176126

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the vertices A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a point P on the ellipse above the x-axis
def P (x y : ℝ) : Prop := ellipse_C x y ∧ y > 0

-- Define the line y = 3
def line_y_3 (x y : ℝ) : Prop := y = 3

-- Define the intersection points G and H
def G (x y : ℝ) : Prop := ∃ (k : ℝ), y = k * (x + 2) ∧ line_y_3 x y
def H (x y : ℝ) : Prop := ∃ (k : ℝ), y = -1/(4*k) * (x - 2) ∧ line_y_3 x y

-- Theorem statement
theorem min_length_GH :
  ∀ (x_p y_p x_g y_g x_h y_h : ℝ),
    P x_p y_p →
    G x_g y_g →
    H x_h y_h →
    ∀ (l : ℝ), l = |x_g - x_h| →
    ∃ (min_l : ℝ), min_l = 8 ∧ l ≥ min_l :=
sorry

end min_length_GH_l1761_176126


namespace vector_difference_magnitude_l1761_176103

theorem vector_difference_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a + b‖ = ‖a - b‖) : 
  ‖a - b‖ = Real.sqrt 5 := by
sorry

end vector_difference_magnitude_l1761_176103


namespace g_zero_at_seven_fifths_l1761_176193

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(7/5) = 0 -/
theorem g_zero_at_seven_fifths : g (7/5) = 0 := by
  sorry

end g_zero_at_seven_fifths_l1761_176193


namespace alpha_value_theorem_l1761_176165

/-- Given a function f(x) = x^α where α is a constant, 
    if the second derivative of f at x = -1 is 4, then α = -4 -/
theorem alpha_value_theorem (α : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^α) 
    (h2 : (deriv^[2] f) (-1) = 4) : 
  α = -4 := by
  sorry

end alpha_value_theorem_l1761_176165


namespace sum_of_coefficients_l1761_176129

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end sum_of_coefficients_l1761_176129


namespace blood_expiration_time_l1761_176198

-- Define the number of seconds in a day
def seconds_per_day : ℕ := 24 * 60 * 60

-- Define the expiration time in seconds (8!)
def expiration_time : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

-- Define the donation time (noon)
def donation_hour : ℕ := 12

-- Theorem statement
theorem blood_expiration_time :
  (expiration_time / seconds_per_day = 0) ∧
  (expiration_time % seconds_per_day / 3600 + donation_hour = 23) :=
sorry

end blood_expiration_time_l1761_176198


namespace car_a_speed_car_a_speed_is_58_l1761_176172

/-- Proves that Car A's speed is 58 mph given the problem conditions -/
theorem car_a_speed : ℝ → Prop :=
  fun (speed_a : ℝ) =>
    let initial_gap : ℝ := 10
    let overtake_distance : ℝ := 8
    let speed_b : ℝ := 50
    let time : ℝ := 2.25
    let distance_b : ℝ := speed_b * time
    let distance_a : ℝ := distance_b + initial_gap + overtake_distance
    speed_a = distance_a / time ∧ speed_a = 58

/-- The theorem is true -/
theorem car_a_speed_is_58 : ∃ (speed_a : ℝ), car_a_speed speed_a :=
sorry

end car_a_speed_car_a_speed_is_58_l1761_176172


namespace total_yellow_balloons_l1761_176132

theorem total_yellow_balloons (fred sam mary : ℕ) 
  (h1 : fred = 5) 
  (h2 : sam = 6) 
  (h3 : mary = 7) : 
  fred + sam + mary = 18 := by
sorry

end total_yellow_balloons_l1761_176132


namespace imaginary_part_of_z_l1761_176137

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = -(1/2 : ℂ) * (1 + Complex.I)) : 
  z.im = (1/2 : ℝ) := by
  sorry

end imaginary_part_of_z_l1761_176137
