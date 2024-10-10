import Mathlib

namespace sufficient_but_not_necessary_l2019_201940

theorem sufficient_but_not_necessary (p q : Prop) :
  (p ∧ q → ¬(¬p)) ∧ ¬(¬(¬p) → p ∧ q) := by sorry

end sufficient_but_not_necessary_l2019_201940


namespace mooncake_sales_properties_l2019_201960

/-- Represents the mooncake sales scenario -/
structure MooncakeSales where
  initial_purchase : ℕ
  purchase_price : ℕ
  initial_selling_price : ℕ
  price_reduction : ℕ
  sales_increase_per_yuan : ℕ

/-- Calculates the profit per box in the second sale -/
def profit_per_box (s : MooncakeSales) : ℤ :=
  s.initial_selling_price - s.purchase_price - s.price_reduction

/-- Calculates the expected sales volume in the second sale -/
def expected_sales_volume (s : MooncakeSales) : ℕ :=
  s.initial_purchase + s.sales_increase_per_yuan * s.price_reduction

/-- Theorem stating the properties of the mooncake sales scenario -/
theorem mooncake_sales_properties (s : MooncakeSales) 
  (h1 : s.initial_purchase = 180)
  (h2 : s.purchase_price = 40)
  (h3 : s.initial_selling_price = 52)
  (h4 : s.sales_increase_per_yuan = 10) :
  (∃ a : ℕ, 
    profit_per_box { initial_purchase := s.initial_purchase,
                     purchase_price := s.purchase_price,
                     initial_selling_price := s.initial_selling_price,
                     price_reduction := a,
                     sales_increase_per_yuan := s.sales_increase_per_yuan } = 12 - a ∧
    expected_sales_volume { initial_purchase := s.initial_purchase,
                            purchase_price := s.purchase_price,
                            initial_selling_price := s.initial_selling_price,
                            price_reduction := a,
                            sales_increase_per_yuan := s.sales_increase_per_yuan } = 180 + 10 * a ∧
    (profit_per_box { initial_purchase := s.initial_purchase,
                      purchase_price := s.purchase_price,
                      initial_selling_price := s.initial_selling_price,
                      price_reduction := a,
                      sales_increase_per_yuan := s.sales_increase_per_yuan } *
     expected_sales_volume { initial_purchase := s.initial_purchase,
                             purchase_price := s.purchase_price,
                             initial_selling_price := s.initial_selling_price,
                             price_reduction := a,
                             sales_increase_per_yuan := s.sales_increase_per_yuan } = 2000 →
     a = 2)) :=
by sorry

end mooncake_sales_properties_l2019_201960


namespace regression_line_equation_l2019_201908

theorem regression_line_equation (slope : ℝ) (center_x center_y : ℝ) :
  slope = 2.03 →
  center_x = 5 →
  center_y = 11 →
  ∀ x y : ℝ, y = slope * x + (center_y - slope * center_x) ↔ y = 2.03 * x + 0.85 :=
by sorry

end regression_line_equation_l2019_201908


namespace star_equation_roots_l2019_201914

-- Define the "★" operation
def star (a b : ℝ) : ℝ := a^2 - b^2

-- Theorem statement
theorem star_equation_roots :
  let x₁ : ℝ := 4
  let x₂ : ℝ := -4
  (star (star 2 3) x₁ = 9) ∧ (star (star 2 3) x₂ = 9) ∧
  (∀ x : ℝ, star (star 2 3) x = 9 → x = x₁ ∨ x = x₂) :=
by sorry

end star_equation_roots_l2019_201914


namespace triangle_problem_l2019_201950

theorem triangle_problem (a b c A B C : Real) (R : Real) :
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- Angles are positive
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Sides are positive
  a + b + c = π ∧  -- Sum of angles in a triangle
  2 * B * Real.cos A = C * Real.cos a + A * Real.cos c ∧  -- Given equation
  A + B + C = 8 ∧  -- Perimeter is 8
  R = Real.sqrt 3 ∧  -- Radius of circumscribed circle is √3
  2 * R * Real.sin (a / 2) = A ∧  -- Relation between side and circumradius
  2 * R * Real.sin (b / 2) = B ∧
  2 * R * Real.sin (c / 2) = C →
  a = π / 3 ∧  -- Angle A is 60°
  A * B * Real.sin c / 2 = 4 * Real.sqrt 3 / 3  -- Area of triangle
  :=
by sorry

end triangle_problem_l2019_201950


namespace room_tiles_count_l2019_201939

/-- Represents the dimensions of a room in centimeters -/
structure RoomDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a room given its dimensions -/
def roomVolume (d : RoomDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Finds the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Calculates the number of cubic tiles needed to fill a room -/
def numTiles (d : RoomDimensions) : ℕ :=
  let tileSize := gcd3 d.length d.width d.height
  roomVolume d / (tileSize * tileSize * tileSize)

/-- The main theorem stating the number of tiles needed for the given room -/
theorem room_tiles_count (room : RoomDimensions) 
    (h1 : room.length = 624)
    (h2 : room.width = 432)
    (h3 : room.height = 356) : 
  numTiles room = 1493952 := by
  sorry

end room_tiles_count_l2019_201939


namespace smallest_sum_on_square_corners_l2019_201901

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_not_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b > 1

theorem smallest_sum_on_square_corners (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 → D > 0 →
  is_relatively_prime A C →
  is_relatively_prime B D →
  is_not_relatively_prime A B →
  is_not_relatively_prime B C →
  is_not_relatively_prime C D →
  is_not_relatively_prime D A →
  A + B + C + D ≥ 60 :=
by sorry

end smallest_sum_on_square_corners_l2019_201901


namespace G_of_4_f_2_l2019_201997

-- Define the functions f and G
def f (a : ℝ) : ℝ := a^2 - 3
def G (a b : ℝ) : ℝ := b^2 - a

-- State the theorem
theorem G_of_4_f_2 : G 4 (f 2) = -3 := by sorry

end G_of_4_f_2_l2019_201997


namespace customer_income_proof_l2019_201969

/-- Proves that given a group of 50 customers with an average income of $45,000, 
    where 10 of these customers have an average income of $55,000, 
    the average income of the remaining 40 customers is $42,500. -/
theorem customer_income_proof (total_customers : Nat) (wealthy_customers : Nat)
  (remaining_customers : Nat) (total_avg_income : ℝ) (wealthy_avg_income : ℝ) :
  total_customers = 50 →
  wealthy_customers = 10 →
  remaining_customers = total_customers - wealthy_customers →
  total_avg_income = 45000 →
  wealthy_avg_income = 55000 →
  (total_customers * total_avg_income - wealthy_customers * wealthy_avg_income) / remaining_customers = 42500 :=
by sorry

end customer_income_proof_l2019_201969


namespace perpendicular_distance_to_adjacent_plane_l2019_201913

/-- A rectangular parallelepiped with dimensions 5 × 5 × 4 -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  length_eq : length = 5
  width_eq : width = 5
  height_eq : height = 4

/-- A vertex of the parallelepiped -/
structure Vertex where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The perpendicular distance from a vertex to a plane -/
def perpendicularDistance (v : Vertex) (plane : Vertex → Vertex → Vertex → Prop) : ℝ :=
  sorry

theorem perpendicular_distance_to_adjacent_plane (p : Parallelepiped) 
  (h v1 v2 v3 : Vertex) 
  (adj1 : h.z = 0)
  (adj2 : v1.z = 0 ∧ v1.y = 0 ∧ v1.x = p.length)
  (adj3 : v2.z = 0 ∧ v2.x = 0 ∧ v2.y = p.width)
  (adj4 : v3.x = 0 ∧ v3.y = 0 ∧ v3.z = p.height) :
  perpendicularDistance h (fun a b c => True) = 4 :=
sorry

end perpendicular_distance_to_adjacent_plane_l2019_201913


namespace book_costs_18_l2019_201975

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage for the CD
def cd_discount_percentage : ℝ := 0.30

-- Define the cost difference between the book and CD
def book_cd_difference : ℝ := 4

-- Calculate the cost of the CD
def cd_cost : ℝ := album_cost * (1 - cd_discount_percentage)

-- Calculate the cost of the book
def book_cost : ℝ := cd_cost + book_cd_difference

-- Theorem to prove
theorem book_costs_18 : book_cost = 18 := by
  sorry

end book_costs_18_l2019_201975


namespace quadratic_square_of_binomial_l2019_201933

theorem quadratic_square_of_binomial (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 116*x + c = (x + a)^2) → c = 3364 := by
  sorry

end quadratic_square_of_binomial_l2019_201933


namespace unique_prime_sum_digits_l2019_201993

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Primality test -/
def isPrime (n : ℕ) : Prop := sorry

theorem unique_prime_sum_digits : 
  ∃! (n : ℕ), isPrime n ∧ n + S n + S (S n) = 3005 :=
sorry

end unique_prime_sum_digits_l2019_201993


namespace repeating_decimal_to_fraction_l2019_201953

/-- Expresses the repeating decimal 0.7̄8̄ as a rational number -/
theorem repeating_decimal_to_fraction : 
  ∃ (n d : ℕ), d ≠ 0 ∧ (0.7 + 0.08 / (1 - 0.1) : ℚ) = n / d ∧ n = 781 ∧ d = 990 := by
  sorry

end repeating_decimal_to_fraction_l2019_201953


namespace increasing_function_condition_l2019_201966

/-- A function f(x) = x - 5/x - a*ln(x) is increasing on [1, +∞) if and only if a ≤ 2√5 -/
theorem increasing_function_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → Monotone (fun x => x - 5 / x - a * Real.log x)) ↔ a ≤ 2 * Real.sqrt 5 := by
  sorry

end increasing_function_condition_l2019_201966


namespace triangle_property_l2019_201961

-- Define a triangle
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi

-- Define the condition
def condition (t : Triangle) : Prop :=
  Real.tan t.β * (Real.sin t.γ)^2 = Real.tan t.γ * (Real.sin t.β)^2

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.α = t.β ∨ t.β = t.γ ∨ t.γ = t.α

-- Define right-angled triangle
def is_right_angled (t : Triangle) : Prop :=
  t.α = Real.pi/2 ∨ t.β = Real.pi/2 ∨ t.γ = Real.pi/2

-- Theorem statement
theorem triangle_property (t : Triangle) :
  condition t → is_isosceles t ∨ is_right_angled t :=
by sorry

end triangle_property_l2019_201961


namespace floor_area_calculation_l2019_201915

/-- The total area of a floor covered by square stone slabs -/
def floor_area (num_slabs : ℕ) (slab_side_length : ℝ) : ℝ :=
  (num_slabs : ℝ) * slab_side_length * slab_side_length

/-- Theorem: The total area of a floor covered by 50 square stone slabs, 
    each with a side length of 140 cm, is 980000 cm² -/
theorem floor_area_calculation :
  floor_area 50 140 = 980000 := by
  sorry

end floor_area_calculation_l2019_201915


namespace equilateral_triangle_perimeter_l2019_201970

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = s → 3 * s = 4 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l2019_201970


namespace three_tenths_of_number_l2019_201926

theorem three_tenths_of_number (n : ℚ) (h : (1/3) * (1/4) * n = 15) : (3/10) * n = 54 := by
  sorry

end three_tenths_of_number_l2019_201926


namespace range_of_m_l2019_201973

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x-1)/3| < 2
def q (x m : ℝ) : Prop := (x-1)^2 < m^2

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (sufficient_not_necessary (p) (q m)) ↔ (-5 < m ∧ m < 5) :=
sorry

end range_of_m_l2019_201973


namespace star_op_result_l2019_201937

/-- The * operation for non-zero integers -/
def star_op (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

/-- Theorem stating the result of the star operation given the conditions -/
theorem star_op_result (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_cond : a + b = 15) (prod_cond : a * b = 36) : 
  star_op a b = 5 / 12 := by
  sorry

end star_op_result_l2019_201937


namespace line_tangent_to_parabola_l2019_201987

theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), 4 * x + 3 * y + 18 = 0 ∧ y^2 = 32 * x := by
  sorry

end line_tangent_to_parabola_l2019_201987


namespace balloon_difference_l2019_201912

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
sorry

end balloon_difference_l2019_201912


namespace polynomial_multiple_power_coefficients_l2019_201971

theorem polynomial_multiple_power_coefficients 
  (p : Polynomial ℝ) (n : ℕ) (hn : n > 0) :
  ∃ q : Polynomial ℝ, q ≠ 0 ∧ 
  ∀ i : ℕ, (p * q).coeff i ≠ 0 → ∃ k : ℕ, i = n * k :=
by sorry

end polynomial_multiple_power_coefficients_l2019_201971


namespace rocket_max_height_l2019_201928

/-- The height function of the rocket -/
def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 36

/-- Theorem stating that the maximum height of the rocket is 144 feet -/
theorem rocket_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 144 := by
  sorry

end rocket_max_height_l2019_201928


namespace average_sleep_time_l2019_201954

def sleep_times : List ℝ := [10, 9, 10, 8, 8]

theorem average_sleep_time : (sleep_times.sum / sleep_times.length) = 9 := by
  sorry

end average_sleep_time_l2019_201954


namespace tangent_implies_one_point_one_point_not_always_tangent_l2019_201904

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a parabola in 2D space
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of a line being tangent to a parabola
def is_tangent (l : Line) (p : Parabola) : Prop := sorry

-- Define the property of a line and a parabola having exactly one common point
def has_one_common_point (l : Line) (p : Parabola) : Prop := sorry

-- Theorem stating the relationship between tangency and having one common point
theorem tangent_implies_one_point (l : Line) (p : Parabola) :
  is_tangent l p → has_one_common_point l p :=
sorry

-- Theorem stating that having one common point doesn't always imply tangency
theorem one_point_not_always_tangent :
  ∃ l : Line, ∃ p : Parabola, has_one_common_point l p ∧ ¬is_tangent l p :=
sorry

end tangent_implies_one_point_one_point_not_always_tangent_l2019_201904


namespace non_congruent_squares_count_l2019_201929

/-- A lattice point on a 2D grid --/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A square on a lattice grid --/
structure LatticeSquare where
  vertices : Finset LatticePoint
  size : ℕ

/-- The size of the grid --/
def gridSize : ℕ := 6

/-- Function to count standard squares of a given size --/
def countStandardSquares (k : ℕ) : ℕ :=
  (gridSize - k + 1) ^ 2

/-- Function to count 45-degree tilted squares with diagonal of a given size --/
def countTiltedSquares (k : ℕ) : ℕ :=
  (gridSize - k) ^ 2

/-- Function to count 45-degree tilted squares with diagonal of a rectangle --/
def countRectangleDiagonalSquares (w h : ℕ) : ℕ :=
  2 * (gridSize - w) * (gridSize - h)

/-- The total number of non-congruent squares on the grid --/
def totalNonCongruentSquares : ℕ :=
  (countStandardSquares 1) + (countStandardSquares 2) + (countStandardSquares 3) +
  (countStandardSquares 4) + (countStandardSquares 5) +
  (countTiltedSquares 1) + (countTiltedSquares 2) +
  (countRectangleDiagonalSquares 1 2) + (countRectangleDiagonalSquares 1 3)

/-- Theorem: The number of non-congruent squares on a 6x6 grid is 201 --/
theorem non_congruent_squares_count :
  totalNonCongruentSquares = 201 := by
  sorry

end non_congruent_squares_count_l2019_201929


namespace remainder_of_n_l2019_201935

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 7 = 1) (h2 : n^3 % 7 = 6) : n % 7 = 6 := by
  sorry

end remainder_of_n_l2019_201935


namespace smallest_of_three_successive_numbers_l2019_201981

theorem smallest_of_three_successive_numbers :
  ∀ n : ℕ, n * (n + 1) * (n + 2) = 1059460 → n = 101 :=
by
  sorry

end smallest_of_three_successive_numbers_l2019_201981


namespace even_function_value_l2019_201943

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

-- Define the property of being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the domain of the function
def domain (b : ℝ) : Set ℝ := Set.Ioo (-b) (2*b - 2)

-- State the theorem
theorem even_function_value (a b : ℝ) :
  is_even_function (f a b) ∧ (∀ x ∈ domain b, f a b x = f a b x) →
  f a b (b/2) = 2 :=
sorry

end even_function_value_l2019_201943


namespace forty_percent_of_three_fifths_of_150_forty_percent_of_three_fifths_of_150_equals_36_l2019_201979

theorem forty_percent_of_three_fifths_of_150 : ℚ :=
  let number : ℚ := 150
  let three_fifths : ℚ := 3 / 5
  let forty_percent : ℚ := 40 / 100
  forty_percent * (three_fifths * number)
  
-- Prove that the above expression equals 36
theorem forty_percent_of_three_fifths_of_150_equals_36 :
  forty_percent_of_three_fifths_of_150 = 36 := by sorry

end forty_percent_of_three_fifths_of_150_forty_percent_of_three_fifths_of_150_equals_36_l2019_201979


namespace some_value_theorem_l2019_201977

theorem some_value_theorem (w x y : ℝ) (h1 : (w + x) / 2 = 0.5) (h2 : w * x = y) :
  ∃ some_value : ℝ, 3 / w + some_value = 3 / y ∧ some_value = 6 := by
  sorry

end some_value_theorem_l2019_201977


namespace range_of_a_l2019_201980

open Set

-- Define sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = ∅ ↔ 1 < a ∧ a < 2 := by sorry

end range_of_a_l2019_201980


namespace least_satisfying_number_l2019_201909

def is_multiple_of_36 (n : ℕ) : Prop := ∃ k : ℕ, n = 36 * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_multiple_of_36 n ∧ is_multiple_of_36 (digit_product n)

theorem least_satisfying_number :
  satisfies_condition 1296 ∧ 
  ∀ m : ℕ, m > 0 ∧ m < 1296 → ¬(satisfies_condition m) :=
by sorry

end least_satisfying_number_l2019_201909


namespace tangent_circles_m_values_l2019_201934

-- Define the equations of the circles
def C₁ (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x + 4*y + m^2 - 5 = 0
def C₂ (x y m : ℝ) : Prop := x^2 + y^2 + 2*x - 2*m*y + m^2 - 3 = 0

-- Define the condition for circles being tangent
def are_tangent (m : ℝ) : Prop :=
  ∃ x y, C₁ x y m ∧ C₂ x y m ∧
  (∀ x' y', C₁ x' y' m ∧ C₂ x' y' m → (x', y') = (x, y))

-- Theorem statement
theorem tangent_circles_m_values :
  {m : ℝ | are_tangent m} = {-5, -2, -1, 2} :=
sorry

end tangent_circles_m_values_l2019_201934


namespace solve_laboratory_budget_l2019_201952

def laboratory_budget_problem (total_budget flask_cost : ℕ) : Prop :=
  let test_tube_cost := (2 * flask_cost) / 3
  let safety_gear_cost := test_tube_cost / 2
  let total_spent := flask_cost + test_tube_cost + safety_gear_cost
  let remaining_budget := total_budget - total_spent
  total_budget = 325 ∧ flask_cost = 150 → remaining_budget = 25

theorem solve_laboratory_budget :
  laboratory_budget_problem 325 150 := by
  sorry

end solve_laboratory_budget_l2019_201952


namespace circle_centers_and_m_l2019_201948

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0
def circle_C2 (x y m : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + m = 0

-- Define external tangency
def externally_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧ 
  ∀ (x' y' : ℝ), (C1 x' y' → (x' - x)^2 + (y' - y)^2 > 0) ∧
                 (C2 x' y' → (x' - x)^2 + (y' - y)^2 > 0)

-- Theorem statement
theorem circle_centers_and_m :
  externally_tangent circle_C1 (circle_C2 · · (-3)) →
  (∃ (x y : ℝ), circle_C1 x y ∧ x = -1 ∧ y = -1) ∧
  (∀ m : ℝ, externally_tangent circle_C1 (circle_C2 · · m) → m = -3) :=
sorry

end circle_centers_and_m_l2019_201948


namespace carpet_dimensions_l2019_201995

/-- A rectangular carpet with integer side lengths. -/
structure Carpet where
  length : ℕ
  width : ℕ

/-- A rectangular room. -/
structure Room where
  length : ℕ
  width : ℕ

/-- Predicate to check if a carpet fits perfectly (diagonally) in a room. -/
def fits_perfectly (c : Carpet) (r : Room) : Prop :=
  (c.length ^ 2 + c.width ^ 2 : ℕ) = r.length ^ 2 + r.width ^ 2

/-- The main theorem about the carpet dimensions. -/
theorem carpet_dimensions :
  ∀ (c : Carpet) (r1 r2 : Room),
    r1.width = 50 →
    r2.width = 38 →
    r1.length = r2.length →
    fits_perfectly c r1 →
    fits_perfectly c r2 →
    c.length = 50 ∧ c.width = 25 := by
  sorry

end carpet_dimensions_l2019_201995


namespace smallest_divisor_after_361_l2019_201946

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem smallest_divisor_after_361 (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m ≤ 9999)  -- m is a 4-digit number
  (h2 : is_even m)             -- m is even
  (h3 : m % 361 = 0)           -- m is divisible by 361
  : (∃ d : ℕ, d ∣ m ∧ 361 < d ∧ ∀ d' : ℕ, d' ∣ m → 361 < d' → d ≤ d') → 
    (∃ d : ℕ, d ∣ m ∧ 361 < d ∧ ∀ d' : ℕ, d' ∣ m → 361 < d' → d ≤ d' ∧ d = 380) :=
by sorry

end smallest_divisor_after_361_l2019_201946


namespace union_of_M_and_N_l2019_201903

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 = x}
def N : Set ℝ := {x : ℝ | Real.log x ≤ 0}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end union_of_M_and_N_l2019_201903


namespace stone_breaking_loss_l2019_201955

/-- Represents the properties of a precious stone -/
structure Stone where
  weight : ℝ
  price : ℝ
  k : ℝ

/-- Calculates the price of a stone given its weight and constant k -/
def calculatePrice (weight : ℝ) (k : ℝ) : ℝ := k * weight^3

/-- Calculates the loss when a stone breaks -/
def calculateLoss (original : Stone) (piece1 : Stone) (piece2 : Stone) : ℝ :=
  original.price - (piece1.price + piece2.price)

theorem stone_breaking_loss (original : Stone) (piece1 : Stone) (piece2 : Stone) :
  original.weight = 28 ∧ 
  original.price = 60000 ∧ 
  original.price = calculatePrice original.weight original.k ∧
  piece1.weight = (17 / 28) * original.weight ∧
  piece2.weight = (11 / 28) * original.weight ∧
  piece1.k = original.k ∧
  piece2.k = original.k ∧
  piece1.price = calculatePrice piece1.weight piece1.k ∧
  piece2.price = calculatePrice piece2.weight piece2.k →
  abs (calculateLoss original piece1 piece2 - 42933.33) < 0.01 := by
  sorry

end stone_breaking_loss_l2019_201955


namespace sufficient_not_necessary_condition_l2019_201989

theorem sufficient_not_necessary_condition (m : ℝ) : m = 1/2 →
  m > 0 ∧
  (∀ x : ℝ, 0 < x ∧ x < m → x * (x - 1) < 0) ∧
  (∃ x : ℝ, x * (x - 1) < 0 ∧ ¬(0 < x ∧ x < m)) := by
  sorry

end sufficient_not_necessary_condition_l2019_201989


namespace sarah_apples_to_teachers_l2019_201910

/-- Calculates the number of apples given to teachers -/
def apples_to_teachers (initial : ℕ) (to_friends : ℕ) (eaten : ℕ) (left : ℕ) : ℕ :=
  initial - to_friends - eaten - left

/-- Theorem stating that Sarah gave 16 apples to teachers -/
theorem sarah_apples_to_teachers :
  apples_to_teachers 25 5 1 3 = 16 := by
  sorry

#eval apples_to_teachers 25 5 1 3

end sarah_apples_to_teachers_l2019_201910


namespace equation_solution_l2019_201907

theorem equation_solution : ∃! x : ℝ, (1 / (x - 1) = 3 / (x - 3)) ∧ x = 0 := by
  sorry

end equation_solution_l2019_201907


namespace angle_of_inclination_of_line_l2019_201978

theorem angle_of_inclination_of_line (x y : ℝ) :
  x + Real.sqrt 3 * y - 1 = 0 →
  ∃ α : ℝ, α = 5 * π / 6 ∧ Real.tan α = -Real.sqrt 3 / 3 := by
  sorry

end angle_of_inclination_of_line_l2019_201978


namespace expand_product_l2019_201900

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9*x + 18 := by
  sorry

end expand_product_l2019_201900


namespace molecular_weight_CCl4_is_152_l2019_201924

/-- The molecular weight of Carbon tetrachloride -/
def molecular_weight_CCl4 : ℝ := 152

/-- The number of moles in the given sample -/
def num_moles : ℝ := 9

/-- The total molecular weight of the given sample -/
def total_weight : ℝ := 1368

/-- Theorem stating that the molecular weight of Carbon tetrachloride is 152 g/mol -/
theorem molecular_weight_CCl4_is_152 :
  molecular_weight_CCl4 = total_weight / num_moles :=
by
  sorry

end molecular_weight_CCl4_is_152_l2019_201924


namespace boxes_with_neither_l2019_201938

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : markers = 8)
  (h3 : erasers = 5)
  (h4 : both = 4)
  : total - (markers + erasers - both) = 6 := by
  sorry

end boxes_with_neither_l2019_201938


namespace jason_final_cards_l2019_201996

def pokemon_card_transactions (initial_cards : ℕ) 
  (benny_trade_out benny_trade_in : ℕ) 
  (sean_trade_out sean_trade_in : ℕ) 
  (given_to_brother : ℕ) : ℕ :=
  initial_cards - benny_trade_out + benny_trade_in - sean_trade_out + sean_trade_in - given_to_brother

theorem jason_final_cards : 
  pokemon_card_transactions 5 2 3 3 4 2 = 5 := by
  sorry

end jason_final_cards_l2019_201996


namespace cd_length_ratio_l2019_201949

/-- Given three CDs, where two have the same length and the total length of all CDs is known,
    this theorem proves the ratio of the length of the third CD to one of the first two. -/
theorem cd_length_ratio (length_first_two : ℝ) (total_length : ℝ) : 
  length_first_two > 0 →
  total_length > 2 * length_first_two →
  (total_length - 2 * length_first_two) / length_first_two = 2 := by
  sorry

#check cd_length_ratio

end cd_length_ratio_l2019_201949


namespace range_of_a_l2019_201957

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∃ a : ℝ, (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(q x a) ∧ p x)) →
  (∀ a : ℝ, (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(q x a) ∧ p x) → a ≥ 1) :=
by sorry

-- The range of a is [1, +∞)

end range_of_a_l2019_201957


namespace book_club_meeting_lcm_l2019_201976

/-- The least common multiple of 5, 6, 8, 9, and 10 is 360 -/
theorem book_club_meeting_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 10))) = 360 := by
  sorry

end book_club_meeting_lcm_l2019_201976


namespace unique_five_digit_square_last_five_l2019_201916

theorem unique_five_digit_square_last_five : ∃! (A : ℕ), 
  10000 ≤ A ∧ A < 100000 ∧ A^2 % 100000 = A :=
by
  use 90625
  sorry

end unique_five_digit_square_last_five_l2019_201916


namespace f_composition_at_one_l2019_201982

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x else Real.log (x - 1)

theorem f_composition_at_one : f (f 1) = 0 := by
  sorry

end f_composition_at_one_l2019_201982


namespace ellipse_eccentricity_m_values_l2019_201919

theorem ellipse_eccentricity_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 / 9 + y^2 / (m + 9) = 1) →
  (∃ c : ℝ, c^2 / (m + 9) = 1/4) →
  (m = -9/4 ∨ m = 3) := by
  sorry

end ellipse_eccentricity_m_values_l2019_201919


namespace balcony_orchestra_difference_l2019_201972

/-- Represents the number of tickets sold for a theater performance. -/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total number of tickets sold. -/
def TicketSales.total (ts : TicketSales) : ℕ :=
  ts.orchestra + ts.balcony

/-- Calculates the total revenue from ticket sales. -/
def TicketSales.revenue (ts : TicketSales) : ℕ :=
  12 * ts.orchestra + 8 * ts.balcony

/-- Theorem stating the difference between balcony and orchestra ticket sales. -/
theorem balcony_orchestra_difference (ts : TicketSales) 
  (h1 : ts.total = 350)
  (h2 : ts.revenue = 3320) :
  ts.balcony - ts.orchestra = 90 := by
  sorry


end balcony_orchestra_difference_l2019_201972


namespace right_triangle_ratio_l2019_201947

theorem right_triangle_ratio (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_right_triangle : a^2 + b^2 = c^2) : 
  (a^2 + b^2) / (a^2 + b^2 + c^2) = 1/2 :=
by sorry

end right_triangle_ratio_l2019_201947


namespace water_containers_capacity_l2019_201999

/-- The problem of calculating the combined capacity of three water containers -/
theorem water_containers_capacity :
  ∀ (A B C : ℝ),
    (0.35 * A + 48 = 0.75 * A) →
    (0.45 * B + 36 = 0.95 * B) →
    (0.20 * C - 24 = 0.10 * C) →
    A + B + C = 432 :=
by sorry

end water_containers_capacity_l2019_201999


namespace eighth_term_of_specific_sequence_l2019_201958

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_diff : ℝ

/-- The nth term of an arithmetic sequence. -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + seq.common_diff * (n - 1 : ℝ)

theorem eighth_term_of_specific_sequence :
  ∃ (seq : ArithmeticSequence),
    nth_term seq 4 = 23 ∧
    nth_term seq 6 = 47 ∧
    nth_term seq 8 = 71 := by
  sorry

end eighth_term_of_specific_sequence_l2019_201958


namespace largest_n_binomial_equality_l2019_201902

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n)) ∧ 
  (∀ m : ℕ, m > 6 → Nat.choose 10 4 + Nat.choose 10 5 ≠ Nat.choose 11 m) :=
by sorry

end largest_n_binomial_equality_l2019_201902


namespace homework_check_probability_l2019_201905

/-- Represents the days of the week when math lessons occur -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- The probability space for the homework checking scenario -/
structure HomeworkProbability where
  /-- The probability that the teacher will not check homework at all during the week -/
  p_no_check : ℝ
  /-- The probability that the teacher will check homework exactly once during the week -/
  p_check_once : ℝ
  /-- The number of math lessons per week -/
  num_lessons : ℕ
  /-- Assumption: probabilities sum to 1 -/
  sum_to_one : p_no_check + p_check_once = 1
  /-- Assumption: probabilities are non-negative -/
  non_negative : 0 ≤ p_no_check ∧ 0 ≤ p_check_once
  /-- Assumption: there are 5 math lessons per week -/
  five_lessons : num_lessons = 5

/-- The main theorem to prove -/
theorem homework_check_probability (hp : HomeworkProbability) :
  hp.p_no_check = 1/2 →
  hp.p_check_once = 1/2 →
  (1/hp.num_lessons : ℝ) * hp.p_check_once / (hp.p_no_check + (1/hp.num_lessons) * hp.p_check_once) = 1/6 :=
by sorry

end homework_check_probability_l2019_201905


namespace bowling_tournament_orderings_l2019_201956

/-- Represents a tournament with a fixed number of participants and rounds --/
structure Tournament where
  participants : Nat
  rounds : Nat

/-- Calculates the number of possible orderings in a tournament --/
def possibleOrderings (t : Tournament) : Nat :=
  2 ^ t.rounds

/-- The specific tournament described in the problem --/
def bowlingTournament : Tournament :=
  { participants := 6, rounds := 5 }

/-- Theorem stating that the number of possible orderings in the bowling tournament is 32 --/
theorem bowling_tournament_orderings :
  possibleOrderings bowlingTournament = 32 := by
  sorry

#eval possibleOrderings bowlingTournament

end bowling_tournament_orderings_l2019_201956


namespace problem_statement_l2019_201994

theorem problem_statement : (2112 - 2021)^2 / 169 = 49 := by
  sorry

end problem_statement_l2019_201994


namespace x_minus_y_equals_twelve_l2019_201974

theorem x_minus_y_equals_twelve (x y : ℕ) : 
  3^x * 4^y = 531441 → x = 12 → x - y = 12 := by
  sorry

end x_minus_y_equals_twelve_l2019_201974


namespace adults_attending_concert_concert_attendance_proof_l2019_201921

/-- The number of adults attending a music festival concert, given ticket prices and total revenue --/
theorem adults_attending_concert (adult_price : ℕ) (child_price : ℕ) (num_children : ℕ) (total_revenue : ℕ) : ℕ :=
  let adults : ℕ := (total_revenue - num_children * child_price) / adult_price
  adults

/-- Proof that 183 adults attended the concert given the specific conditions --/
theorem concert_attendance_proof :
  adults_attending_concert 26 13 28 5122 = 183 := by
  sorry

end adults_attending_concert_concert_attendance_proof_l2019_201921


namespace school_ratio_problem_l2019_201925

/-- Given a school with 300 students, where the ratio of boys to girls is x : y,
    prove that if the number of boys is increased by z such that the number of girls
    becomes x% of the total, then z = 300 - 3x - 300x / (x + y). -/
theorem school_ratio_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  ∃ z : ℝ, z = 300 - 3*x - 300*x / (x + y) := by
  sorry


end school_ratio_problem_l2019_201925


namespace no_solution_absolute_value_equation_l2019_201932

theorem no_solution_absolute_value_equation :
  (∀ x : ℝ, (x - 4)^2 ≠ 0 → ∃ y : ℝ, (y - 4)^2 = 0) ∧
  (∀ x : ℝ, |(-5 : ℝ) * x| + 10 ≠ 0) ∧
  (∀ x : ℝ, Real.sqrt (-x) - 3 ≠ 0 → ∃ y : ℝ, Real.sqrt (-y) - 3 = 0) ∧
  (∀ x : ℝ, Real.sqrt x - 7 ≠ 0 → ∃ y : ℝ, Real.sqrt y - 7 = 0) ∧
  (∀ x : ℝ, |(-5 : ℝ) * x| - 6 ≠ 0 → ∃ y : ℝ, |(-5 : ℝ) * y| - 6 = 0) :=
by sorry


end no_solution_absolute_value_equation_l2019_201932


namespace largest_two_digit_satisfying_property_l2019_201945

/-- Given a two-digit number n = 10a + b, where a and b are single digits,
    we define a function that switches the digits and adds 5. -/
def switchAndAdd5 (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a + 5

/-- The property that switching digits and adding 5 results in 3n -/
def satisfiesProperty (n : ℕ) : Prop :=
  switchAndAdd5 n = 3 * n

theorem largest_two_digit_satisfying_property :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ satisfiesProperty n ∧
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ satisfiesProperty m → m ≤ n :=
by
  use 13
  sorry

end largest_two_digit_satisfying_property_l2019_201945


namespace nancy_coffee_days_l2019_201985

/-- Represents Nancy's coffee buying habits and expenses -/
structure CoffeeExpense where
  double_espresso_price : ℚ
  iced_coffee_price : ℚ
  total_spent : ℚ

/-- Calculates the number of days Nancy has been buying coffee -/
def days_buying_coffee (expense : CoffeeExpense) : ℚ :=
  expense.total_spent / (expense.double_espresso_price + expense.iced_coffee_price)

/-- Theorem stating that Nancy has been buying coffee for 20 days -/
theorem nancy_coffee_days :
  let expense : CoffeeExpense := {
    double_espresso_price := 3,
    iced_coffee_price := 5/2,
    total_spent := 110
  }
  days_buying_coffee expense = 20 := by
  sorry

end nancy_coffee_days_l2019_201985


namespace triangle_properties_l2019_201991

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b^2 + t.c^2 = 3 * t.b * t.c * Real.cos t.A)
  (h2 : t.B = t.C)
  (h3 : t.a = 2) :
  (1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 5) ∧ 
  (Real.tan t.A / Real.tan t.B + Real.tan t.A / Real.tan t.C = 1) := by
  sorry

end triangle_properties_l2019_201991


namespace fraction_subtraction_and_multiplication_l2019_201917

theorem fraction_subtraction_and_multiplication :
  (1 / 2 : ℚ) * ((5 / 6 : ℚ) - (1 / 9 : ℚ)) = 13 / 36 := by
  sorry

end fraction_subtraction_and_multiplication_l2019_201917


namespace fence_painting_ways_l2019_201986

/-- Represents the number of colors available for painting --/
def num_colors : ℕ := 3

/-- Represents the number of boards in the fence --/
def num_boards : ℕ := 10

/-- Calculates the total number of ways to paint the fence with any two adjacent boards having different colors --/
def total_ways : ℕ := num_colors * (2^(num_boards - 1))

/-- Calculates the number of ways to paint the fence using only two colors --/
def two_color_ways : ℕ := num_colors * (num_colors - 1)

/-- Theorem: The number of ways to paint a fence of 10 boards with 3 colors, 
    such that any two adjacent boards are of different colors and all three colors are used, 
    is equal to 1530 --/
theorem fence_painting_ways : 
  total_ways - two_color_ways = 1530 := by sorry

end fence_painting_ways_l2019_201986


namespace range_of_a_range_of_m_l2019_201951

-- Part 1
def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a :
  (∀ x, ¬(q x a) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

-- Part 2
def s (m : ℝ) : Prop :=
  ∃ x y, x^2 + (m - 3) * x + m = 0 ∧
         y^2 + (m - 3) * y + m = 0 ∧
         0 < x ∧ x < 1 ∧ 2 < y ∧ y < 3

def t (m : ℝ) : Prop :=
  ∀ x, m * x^2 - 2 * x + 1 > 0

theorem range_of_m :
  s m ∨ t m →
  (0 < m ∧ m < 2/3) ∨ m > 1 :=
sorry

end range_of_a_range_of_m_l2019_201951


namespace geometric_sequences_exist_and_unique_l2019_201920

/-- Three geometric sequences satisfying the given conditions -/
def geometric_sequences (a q : ℝ) : Fin 3 → ℕ → ℝ
| ⟨0, _⟩ => λ n => a * (q - 2) ^ n
| ⟨1, _⟩ => λ n => 2 * a * (q - 1) ^ n
| ⟨2, _⟩ => λ n => 4 * a * q ^ n

/-- The theorem stating the existence and uniqueness of the geometric sequences -/
theorem geometric_sequences_exist_and_unique :
  ∃ (a q : ℝ),
    (∀ i : Fin 3, geometric_sequences a q i 0 = a * (2 ^ i.val)) ∧
    (geometric_sequences a q 1 1 - geometric_sequences a q 0 1 =
     geometric_sequences a q 2 1 - geometric_sequences a q 1 1) ∧
    (geometric_sequences a q 0 1 + geometric_sequences a q 1 1 + geometric_sequences a q 2 1 = 24) ∧
    (geometric_sequences a q 0 0 + geometric_sequences a q 1 0 + geometric_sequences a q 2 0 = 84) ∧
    ((a = 1 ∧ q = 4) ∨ (a = 192 / 31 ∧ q = 9 / 8)) := by
  sorry


end geometric_sequences_exist_and_unique_l2019_201920


namespace non_indian_percentage_approx_l2019_201984

/-- Represents the number of attendees in a category and the percentage of Indians in that category -/
structure AttendeeCategory where
  total : ℕ
  indianPercentage : ℚ

/-- Calculates the number of non-Indian attendees in a category -/
def nonIndianCount (category : AttendeeCategory) : ℚ :=
  category.total * (1 - category.indianPercentage)

/-- Data for the climate conference -/
def conferenceData : List AttendeeCategory := [
  ⟨1200, 25/100⟩,  -- Male participants
  ⟨800, 40/100⟩,   -- Male volunteers
  ⟨1000, 35/100⟩,  -- Female participants
  ⟨500, 15/100⟩,   -- Female volunteers
  ⟨1800, 10/100⟩,  -- Children
  ⟨500, 45/100⟩,   -- Male scientists
  ⟨250, 30/100⟩,   -- Female scientists
  ⟨350, 55/100⟩,   -- Male government officials
  ⟨150, 50/100⟩    -- Female government officials
]

/-- Total number of attendees -/
def totalAttendees : ℕ := 6550

/-- Theorem stating that the percentage of non-Indian attendees is approximately 72.61% -/
theorem non_indian_percentage_approx :
  abs ((List.sum (List.map nonIndianCount conferenceData) / totalAttendees) - 72.61/100) < 1/100 := by
  sorry

end non_indian_percentage_approx_l2019_201984


namespace sqrt_12_minus_sqrt_3_between_1_and_2_l2019_201959

theorem sqrt_12_minus_sqrt_3_between_1_and_2 : 1 < Real.sqrt 12 - Real.sqrt 3 ∧ Real.sqrt 12 - Real.sqrt 3 < 2 := by
  sorry

end sqrt_12_minus_sqrt_3_between_1_and_2_l2019_201959


namespace example_linear_equation_l2019_201923

/-- Represents a linear equation with two variables -/
structure LinearEquationTwoVars where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ a * x + b * y = c

/-- The equation 5x + y = 2 is a linear equation with two variables -/
theorem example_linear_equation : ∃ e : LinearEquationTwoVars, e.a = 5 ∧ e.b = 1 ∧ e.c = 2 := by
  sorry

end example_linear_equation_l2019_201923


namespace williams_tickets_l2019_201918

theorem williams_tickets (initial_tickets : ℕ) : 
  initial_tickets + 3 = 18 → initial_tickets = 15 := by
  sorry

end williams_tickets_l2019_201918


namespace only_14_satisfies_l2019_201922

def is_multiple_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_conditions (n : ℕ) : Prop :=
  ¬(is_multiple_of_three n) ∧
  ¬(is_perfect_square n) ∧
  is_prime (sum_of_digits n)

theorem only_14_satisfies :
  satisfies_conditions 14 ∧
  ¬(satisfies_conditions 12) ∧
  ¬(satisfies_conditions 16) ∧
  ¬(satisfies_conditions 21) ∧
  ¬(satisfies_conditions 26) :=
sorry

end only_14_satisfies_l2019_201922


namespace trigonometric_identity_l2019_201967

theorem trigonometric_identity (h1 : Real.tan (10 * π / 180) * Real.tan (20 * π / 180) + 
                                     Real.tan (20 * π / 180) * Real.tan (60 * π / 180) + 
                                     Real.tan (60 * π / 180) * Real.tan (10 * π / 180) = 1)
                               (h2 : Real.tan (5 * π / 180) * Real.tan (10 * π / 180) + 
                                     Real.tan (10 * π / 180) * Real.tan (75 * π / 180) + 
                                     Real.tan (75 * π / 180) * Real.tan (5 * π / 180) = 1) :
  Real.tan (8 * π / 180) * Real.tan (12 * π / 180) + 
  Real.tan (12 * π / 180) * Real.tan (70 * π / 180) + 
  Real.tan (70 * π / 180) * Real.tan (8 * π / 180) = 1 := by
  sorry

end trigonometric_identity_l2019_201967


namespace price_reduction_proof_l2019_201968

theorem price_reduction_proof (current_price : ℝ) (reduction_percentage : ℝ) (claimed_reduction : ℝ) : 
  current_price = 45 ∧ reduction_percentage = 0.1 ∧ claimed_reduction = 10 →
  (100 / (100 - reduction_percentage * 100) * current_price) - current_price ≠ claimed_reduction :=
by
  sorry

end price_reduction_proof_l2019_201968


namespace average_of_remaining_numbers_l2019_201927

theorem average_of_remaining_numbers
  (total_count : Nat)
  (total_average : ℝ)
  (first_pair_average : ℝ)
  (second_pair_average : ℝ)
  (h_total_count : total_count = 6)
  (h_total_average : total_average = 4.60)
  (h_first_pair_average : first_pair_average = 3.4)
  (h_second_pair_average : second_pair_average = 3.8) :
  (total_count : ℝ) * total_average - 2 * first_pair_average - 2 * second_pair_average = 2 * 6.6 := by
sorry

end average_of_remaining_numbers_l2019_201927


namespace sum_of_seventh_terms_l2019_201965

/-- First sequence defined by a_n = n^2 + n - 1 -/
def sequence_a (n : ℕ) : ℕ := n^2 + n - 1

/-- Second sequence defined by b_n = n(n+1)/2 -/
def sequence_b (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the 7th terms of both sequences is 83 -/
theorem sum_of_seventh_terms :
  sequence_a 7 + sequence_b 7 = 83 := by sorry

end sum_of_seventh_terms_l2019_201965


namespace sqrt_equation_solution_l2019_201930

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 * x - 6) = 10 → x = 106 / 3 := by
  sorry

end sqrt_equation_solution_l2019_201930


namespace min_boxes_for_muffins_l2019_201964

/-- Represents the number of muffins that can be packed in each box type -/
structure BoxCapacity where
  large : Nat
  medium : Nat
  small : Nat

/-- Represents the number of boxes used for each type -/
structure BoxCount where
  large : Nat
  medium : Nat
  small : Nat

def total_muffins : Nat := 250

def box_capacity : BoxCapacity := ⟨12, 8, 4⟩

def box_count : BoxCount := ⟨20, 1, 1⟩

/-- Calculates the total number of muffins that can be packed in the given boxes -/
def muffins_packed (capacity : BoxCapacity) (count : BoxCount) : Nat :=
  capacity.large * count.large + capacity.medium * count.medium + capacity.small * count.small

/-- Calculates the total number of boxes used -/
def total_boxes (count : BoxCount) : Nat :=
  count.large + count.medium + count.small

theorem min_boxes_for_muffins :
  muffins_packed box_capacity box_count = total_muffins ∧
  total_boxes box_count = 22 ∧
  ∀ (other_count : BoxCount),
    muffins_packed box_capacity other_count ≥ total_muffins →
    total_boxes other_count ≥ total_boxes box_count :=
by
  sorry

end min_boxes_for_muffins_l2019_201964


namespace ratio_calculation_l2019_201936

theorem ratio_calculation (x y a b : ℚ) 
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / (3 * b - y) = 3) :
  a / b = 9 / 2 := by
  sorry

end ratio_calculation_l2019_201936


namespace xyz_sum_max_min_l2019_201990

theorem xyz_sum_max_min (x y z : ℝ) (h : 4 * (x + y + z) = x^2 + y^2 + z^2) :
  let f := fun (a b c : ℝ) => a * b + a * c + b * c
  ∃ (M m : ℝ), (∀ (a b c : ℝ), 4 * (a + b + c) = a^2 + b^2 + c^2 → f a b c ≤ M) ∧
               (∀ (a b c : ℝ), 4 * (a + b + c) = a^2 + b^2 + c^2 → m ≤ f a b c) ∧
               M + 10 * m = 28 :=
by sorry

end xyz_sum_max_min_l2019_201990


namespace monday_miles_proof_l2019_201931

def weekly_miles : ℕ := 30
def wednesday_miles : ℕ := 12

theorem monday_miles_proof (monday_miles : ℕ) 
  (h1 : monday_miles + wednesday_miles + 2 * monday_miles = weekly_miles) : 
  monday_miles = 6 := by
  sorry

end monday_miles_proof_l2019_201931


namespace min_value_2a_plus_b_plus_c_l2019_201963

theorem min_value_2a_plus_b_plus_c (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a * (a + b + c) + b * c = 4) : 
  ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * (x + y + z) + y * z = 4 → 2*a + b + c ≤ 2*x + y + z :=
by sorry

end min_value_2a_plus_b_plus_c_l2019_201963


namespace estate_distribution_valid_l2019_201911

/-- Represents the estate distribution problem with twins --/
structure EstateDistribution :=
  (total : ℚ)
  (son_share : ℚ)
  (daughter_share : ℚ)
  (mother_share : ℚ)

/-- Checks if the distribution is valid according to the will's conditions --/
def is_valid_distribution (d : EstateDistribution) : Prop :=
  d.total = 210 ∧
  d.son_share + d.daughter_share + d.mother_share = d.total ∧
  d.son_share = (2/3) * d.total ∧
  d.daughter_share = (1/2) * d.mother_share

/-- Theorem stating that the given distribution is valid --/
theorem estate_distribution_valid :
  is_valid_distribution ⟨210, 140, 70/3, 140/3⟩ := by
  sorry

#check estate_distribution_valid

end estate_distribution_valid_l2019_201911


namespace percentage_relation_l2019_201992

theorem percentage_relation (a b c P : ℝ) : 
  (P / 100) * a = 12 →
  (12 / 100) * b = 6 →
  c = b / a →
  c = P / 24 :=
by
  sorry

end percentage_relation_l2019_201992


namespace class_size_difference_l2019_201944

theorem class_size_difference (A B : ℕ) (h : A - 4 = B + 4) : A - B = 8 := by
  sorry

end class_size_difference_l2019_201944


namespace gp_special_term_l2019_201941

def geometric_progression (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem gp_special_term (b : ℕ → ℝ) (α : ℝ) :
  geometric_progression b →
  (0 < α) ∧ (α < Real.pi / 2) →
  b 25 = 2 * Real.tan α →
  b 31 = 2 * Real.sin α →
  b 37 = Real.sin (2 * α) :=
by sorry

end gp_special_term_l2019_201941


namespace ellipse_equation_l2019_201988

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line with slope m passing through point p -/
structure Line where
  m : ℝ
  p : Point

/-- The theorem statement -/
theorem ellipse_equation (E : Ellipse) (F : Point) (l : Line) (M : Point) :
  F.x = 3 ∧ F.y = 0 ∧  -- Right focus at (3,0)
  l.m = 1/2 ∧ l.p = F ∧  -- Line with slope 1/2 passing through F
  M.x = 1 ∧ M.y = -1 ∧  -- Midpoint at (1,-1)
  (∃ A B : Point, A ≠ B ∧
    (A.x^2 / E.a^2 + A.y^2 / E.b^2 = 1) ∧
    (B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1) ∧
    (A.y - F.y = l.m * (A.x - F.x)) ∧
    (B.y - F.y = l.m * (B.x - F.x)) ∧
    M.x = (A.x + B.x) / 2 ∧
    M.y = (A.y + B.y) / 2) →
  E.a^2 = 18 ∧ E.b^2 = 9 := by
sorry

end ellipse_equation_l2019_201988


namespace statement_contrapositive_and_negation_l2019_201998

theorem statement_contrapositive_and_negation (x y : ℝ) :
  (((x - 1) * (y + 2) = 0 → x = 1 ∨ y = -2) ↔
   (x ≠ 1 ∧ y ≠ -2 → (x - 1) * (y + 2) ≠ 0)) ∧
  (¬((x - 1) * (y + 2) = 0 → x = 1 ∨ y = -2) ↔
   ((x - 1) * (y + 2) = 0 → x ≠ 1 ∧ y ≠ -2)) :=
by sorry

end statement_contrapositive_and_negation_l2019_201998


namespace min_reciprocal_sum_l2019_201983

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  1/a + 1/b + 1/c ≥ 3 ∧ (1/a + 1/b + 1/c = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end min_reciprocal_sum_l2019_201983


namespace zebra_stripes_l2019_201962

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes is one more than white stripes
  b = w + 7 →      -- White stripes are 7 more than wide black stripes
  n = 8            -- Number of narrow black stripes is 8
:= by sorry

end zebra_stripes_l2019_201962


namespace vector_sum_components_l2019_201942

/-- Given 2D vectors a, b, and c, prove that 3a - 2b + c is equal to
    (3ax - 2bx + cx, 3ay - 2by + cy) where ax, ay, bx, by, cx, and cy
    are the respective x and y components of vectors a, b, and c. -/
theorem vector_sum_components (a b c : ℝ × ℝ) :
  3 • a - 2 • b + c = (3 * a.1 - 2 * b.1 + c.1, 3 * a.2 - 2 * b.2 + c.2) := by
  sorry

end vector_sum_components_l2019_201942


namespace pitcher_juice_distribution_l2019_201906

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) :
  let juice_amount : ℝ := (3 / 4) * C
  let cups : ℕ := 8
  let juice_per_cup : ℝ := juice_amount / cups
  let percent_per_cup : ℝ := (juice_per_cup / C) * 100
  percent_per_cup = 9.375 := by
  sorry

end pitcher_juice_distribution_l2019_201906
