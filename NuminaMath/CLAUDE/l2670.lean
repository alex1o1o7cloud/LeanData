import Mathlib

namespace cone_slant_height_l2670_267059

/-- The slant height of a cone given its base radius and curved surface area -/
theorem cone_slant_height (r : ℝ) (csa : ℝ) (h1 : r = 10) (h2 : csa = 628.3185307179587) :
  csa / (π * r) = 20 := by
  sorry

end cone_slant_height_l2670_267059


namespace quadratic_solution_l2670_267054

theorem quadratic_solution (x : ℚ) : 
  (63 * x^2 - 100 * x + 45 = 0) → 
  (63 * (5/7)^2 - 100 * (5/7) + 45 = 0) → 
  (63 * 1^2 - 100 * 1 + 45 = 0) :=
by
  sorry

end quadratic_solution_l2670_267054


namespace line_circle_intersection_k_l2670_267056

/-- A line intersects a circle -/
structure LineCircleIntersection where
  /-- Slope of the line y = kx + 3 -/
  k : ℝ
  /-- The line intersects the circle (x-1)^2 + (y-2)^2 = 9 at two points -/
  intersects : k > 1
  /-- The distance between the two intersection points is 12√5/5 -/
  distance : ℝ
  distance_eq : distance = 12 * Real.sqrt 5 / 5

/-- The slope k of the line is 2 -/
theorem line_circle_intersection_k (lci : LineCircleIntersection) : lci.k = 2 := by
  sorry

end line_circle_intersection_k_l2670_267056


namespace expression_evaluation_l2670_267087

theorem expression_evaluation (x y : ℚ) 
  (hx : x = 2 / 15) (hy : y = 3 / 2) : 
  (2 * x + y)^2 - (3 * x - y)^2 + 5 * x * (x - y) = 1 := by
  sorry

end expression_evaluation_l2670_267087


namespace lost_card_number_l2670_267004

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ≤ n) : 
  ∃ (x : ℕ), x > 0 ∧ x ≤ n ∧ (n * (n + 1)) / 2 - x = 101 ∧ x = 4 := by
  sorry

#check lost_card_number

end lost_card_number_l2670_267004


namespace expansion_properties_l2670_267038

def polynomial_expansion (x : ℝ) (a : Fin 8 → ℝ) : Prop :=
  (1 - 2*x)^7 = a 0 + a 1*x + a 2*x^2 + a 3*x^3 + a 4*x^4 + a 5*x^5 + a 6*x^6 + a 7*x^7

theorem expansion_properties (a : Fin 8 → ℝ) 
  (h : ∀ x, polynomial_expansion x a) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = -2) ∧
  (a 1 + a 3 + a 5 + a 7 = -1094) ∧
  (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| = 2187) := by
  sorry

end expansion_properties_l2670_267038


namespace triangle_properties_l2670_267031

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  Real.sqrt 3 * (a - c * Real.cos B) = b * Real.sin C →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 / 3 →
  a + b = 4 →
  C = Real.pi / 3 ∧
  Real.sin A * Real.sin B = 1 / 12 ∧
  Real.cos A * Real.cos B = 5 / 12 := by
sorry


end triangle_properties_l2670_267031


namespace sqrt_88200_simplification_l2670_267019

theorem sqrt_88200_simplification : Real.sqrt 88200 = 210 * Real.sqrt 6 := by
  sorry

end sqrt_88200_simplification_l2670_267019


namespace smallest_square_cover_l2670_267052

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Checks if a square can be perfectly covered by rectangles of a given size -/
def canCoverSquare (s : Square) (r : Rectangle) : Prop :=
  ∃ n : ℕ, n * r.width * r.height = s.side * s.side ∧ 
    s.side % r.width = 0 ∧ s.side % r.height = 0

/-- The number of rectangles needed to cover the square -/
def numRectangles (s : Square) (r : Rectangle) : ℕ :=
  (s.side * s.side) / (r.width * r.height)

theorem smallest_square_cover :
  ∃ (s : Square) (r : Rectangle), 
    r.width = 3 ∧ r.height = 4 ∧
    canCoverSquare s r ∧
    (∀ (s' : Square), s'.side < s.side → ¬ canCoverSquare s' r) ∧
    numRectangles s r = 9 := by
  sorry

end smallest_square_cover_l2670_267052


namespace integral_x_squared_minus_x_l2670_267086

theorem integral_x_squared_minus_x : ∫ (x : ℝ) in (0)..(1), (x^2 - x) = -1/6 := by sorry

end integral_x_squared_minus_x_l2670_267086


namespace cone_lateral_surface_area_l2670_267021

/-- The lateral surface area of a cone with an equilateral triangle cross-section --/
theorem cone_lateral_surface_area (r h : Real) : 
  r^2 + h^2 = 1 →  -- Condition for equilateral triangle with side length 2
  r * h = 1/2 →    -- Condition for equilateral triangle with side length 2
  2 * π * r = 2 * π := by
  sorry

end cone_lateral_surface_area_l2670_267021


namespace temperature_range_l2670_267005

theorem temperature_range (highest_temp lowest_temp t : ℝ) 
  (h_highest : highest_temp = 30)
  (h_lowest : lowest_temp = 20)
  (h_range : lowest_temp ≤ t ∧ t ≤ highest_temp) :
  20 ≤ t ∧ t ≤ 30 := by
  sorry

end temperature_range_l2670_267005


namespace existence_of_n_with_s_prime_divisors_l2670_267033

theorem existence_of_n_with_s_prime_divisors (s : ℕ) (hs : s > 0) :
  ∃ n : ℕ, n > 0 ∧ (∃ (P : Finset Nat), P.card ≥ s ∧ 
    (∀ p ∈ P, Nat.Prime p ∧ p ∣ (2^n - 1))) :=
by sorry

end existence_of_n_with_s_prime_divisors_l2670_267033


namespace count_ak_divisible_by_9_l2670_267043

/-- The number obtained by writing the integers 1 to n from left to right -/
def a (n : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The count of a_k divisible by 9 for 1 ≤ k ≤ 100 -/
def countDivisibleBy9 : ℕ := sorry

theorem count_ak_divisible_by_9 : countDivisibleBy9 = 22 := by sorry

end count_ak_divisible_by_9_l2670_267043


namespace sin_seven_pi_sixths_l2670_267006

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -(1 / 2) := by sorry

end sin_seven_pi_sixths_l2670_267006


namespace fifteenth_term_of_arithmetic_sequence_l2670_267077

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifteenth_term_of_arithmetic_sequence 
  (a : ℕ → ℕ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 15)
  (h_third : a 3 = 27) :
  a 15 = 171 :=
sorry

end fifteenth_term_of_arithmetic_sequence_l2670_267077


namespace expansion_has_six_nonzero_terms_l2670_267009

/-- The polynomial resulting from expanding (2x^3 - 4)(3x^2 + 5x - 7) + 5 (x^4 - 3x^3 + 2x^2) -/
def expanded_polynomial (x : ℝ) : ℝ :=
  6*x^5 + 15*x^4 - 29*x^3 - 2*x^2 - 20*x + 28

/-- The coefficients of the expanded polynomial -/
def coefficients : List ℝ := [6, 15, -29, -2, -20, 28]

/-- Theorem stating that the expansion has exactly 6 nonzero terms -/
theorem expansion_has_six_nonzero_terms :
  coefficients.length = 6 ∧ coefficients.all (· ≠ 0) := by sorry

end expansion_has_six_nonzero_terms_l2670_267009


namespace compound_interest_principal_l2670_267093

/-- Proves that given specific compound interest conditions, the principal amount is 1500 --/
theorem compound_interest_principal :
  ∀ (CI R T P : ℝ),
    CI = 315 →
    R = 10 →
    T = 2 →
    CI = P * ((1 + R / 100) ^ T - 1) →
    P = 1500 := by
  sorry

end compound_interest_principal_l2670_267093


namespace greatest_divisor_with_remainders_l2670_267081

theorem greatest_divisor_with_remainders : ∃! d : ℕ,
  d > 0 ∧
  (∀ k : ℕ, k > 0 ∧ (∃ q₁ : ℕ, 13976 = k * q₁ + 23) ∧ (∃ q₂ : ℕ, 20868 = k * q₂ + 37) → k ≤ d) ∧
  (∃ q₁ : ℕ, 13976 = d * q₁ + 23) ∧
  (∃ q₂ : ℕ, 20868 = d * q₂ + 37) ∧
  d = 1 :=
by
  sorry

end greatest_divisor_with_remainders_l2670_267081


namespace lateral_side_is_five_l2670_267012

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ
  lateral : ℝ

/-- The property that the given dimensions form a valid isosceles trapezoid -/
def is_valid_trapezoid (t : IsoscelesTrapezoid) : Prop :=
  t.base1 > 0 ∧ t.base2 > 0 ∧ t.area > 0 ∧ t.lateral > 0 ∧
  t.area = (t.base1 + t.base2) * t.lateral / 2

/-- The theorem stating that the lateral side of the trapezoid is 5 -/
theorem lateral_side_is_five (t : IsoscelesTrapezoid)
  (h1 : t.base1 = 8)
  (h2 : t.base2 = 14)
  (h3 : t.area = 44)
  (h4 : is_valid_trapezoid t) :
  t.lateral = 5 :=
sorry

end lateral_side_is_five_l2670_267012


namespace problem_solution_l2670_267095

theorem problem_solution (x y : ℝ) (hx : x = 2 - Real.sqrt 3) (hy : y = 2 + Real.sqrt 3) :
  (x^2 - y^2 = -8 * Real.sqrt 3) ∧ (x^2 + x*y + y^2 = 15) := by
  sorry

end problem_solution_l2670_267095


namespace shelter_animals_count_l2670_267050

theorem shelter_animals_count (cats : ℕ) (dogs : ℕ) 
  (h1 : cats = 645) (h2 : dogs = 567) : cats + dogs = 1212 := by
  sorry

end shelter_animals_count_l2670_267050


namespace private_teacher_cost_l2670_267061

/-- Calculates the amount each parent must pay for a private teacher --/
theorem private_teacher_cost 
  (former_salary : ℕ) 
  (raise_percentage : ℚ) 
  (num_kids : ℕ) 
  (h1 : former_salary = 45000)
  (h2 : raise_percentage = 1/5)
  (h3 : num_kids = 9) :
  (former_salary + former_salary * raise_percentage) / num_kids = 6000 := by
  sorry

#check private_teacher_cost

end private_teacher_cost_l2670_267061


namespace coin_problem_l2670_267057

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

def total_coins : ℕ := 11
def total_value : ℕ := 118

theorem coin_problem (p n d q : ℕ) : 
  p ≥ 1 → n ≥ 1 → d ≥ 1 → q ≥ 1 →
  p + n + d + q = total_coins →
  p * penny + n * nickel + d * dime + q * quarter = total_value →
  d = 3 := by
sorry

end coin_problem_l2670_267057


namespace imaginary_part_of_z_l2670_267041

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - 3 * Complex.I) : 
  z.im = -5/2 := by
  sorry

end imaginary_part_of_z_l2670_267041


namespace circles_common_internal_tangent_l2670_267030

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O₂ (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 9

-- Define the center of circle O₂
def center_O₂ : ℝ × ℝ := (3, 3)

-- Define the property of being externally tangent
def externally_tangent (O₁ O₂ : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), O₁ x y ∧ O₂ x y ∧
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(O₁ x' y' ∧ O₂ x' y')

-- Define the common internal tangent line
def common_internal_tangent (x y : ℝ) : Prop := 3*x + 4*y - 21 = 0

-- State the theorem
theorem circles_common_internal_tangent :
  externally_tangent circle_O₁ circle_O₂ →
  ∀ (x y : ℝ), common_internal_tangent x y ↔
    (∃ (t : ℝ), circle_O₁ (x + t) (y - (3/4)*t) ∧
               circle_O₂ (x - t) (y + (3/4)*t)) :=
sorry

end circles_common_internal_tangent_l2670_267030


namespace scooter_gain_percent_l2670_267036

-- Define the purchase price, repair cost, and selling price
def purchase_price : ℚ := 900
def repair_cost : ℚ := 300
def selling_price : ℚ := 1260

-- Define the total cost
def total_cost : ℚ := purchase_price + repair_cost

-- Define the gain
def gain : ℚ := selling_price - total_cost

-- Define the gain percent
def gain_percent : ℚ := (gain / total_cost) * 100

-- Theorem to prove
theorem scooter_gain_percent : gain_percent = 5 := by
  sorry

end scooter_gain_percent_l2670_267036


namespace exists_desired_arrangement_l2670_267094

/-- A type representing a 10x10 grid of natural numbers -/
def Grid := Fin 10 → Fin 10 → ℕ

/-- A type representing a domino (1x2 rectangle) in the grid -/
inductive Domino
| horizontal : Fin 10 → Fin 9 → Domino
| vertical : Fin 9 → Fin 10 → Domino

/-- A partition of the grid into dominoes -/
def Partition := List Domino

/-- Function to check if a partition is valid (covers the entire grid without overlaps) -/
def isValidPartition (p : Partition) : Prop := sorry

/-- Function to calculate the sum of numbers in a domino for a given grid -/
def dominoSum (g : Grid) (d : Domino) : ℕ := sorry

/-- Function to count the number of dominoes with even sum in a partition -/
def countEvenSumDominoes (g : Grid) (p : Partition) : ℕ := sorry

/-- The main theorem statement -/
theorem exists_desired_arrangement : 
  ∃ (g : Grid), ∀ (p : Partition), isValidPartition p → countEvenSumDominoes g p = 7 := by sorry

end exists_desired_arrangement_l2670_267094


namespace jims_taxi_additional_charge_l2670_267058

/-- The additional charge for each 2/5 of a mile in Jim's taxi service -/
def additional_charge (initial_fee total_distance total_charge : ℚ) : ℚ :=
  ((total_charge - initial_fee) * 2) / (5 * total_distance)

/-- Theorem stating the additional charge for each 2/5 of a mile in Jim's taxi service -/
theorem jims_taxi_additional_charge :
  additional_charge (5/2) (36/10) (565/100) = 35/100 := by
  sorry

end jims_taxi_additional_charge_l2670_267058


namespace root_exists_in_interval_l2670_267011

def f (x : ℝ) := x^3 - x^2 - x - 1

theorem root_exists_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry

end root_exists_in_interval_l2670_267011


namespace tiling_count_is_96_l2670_267032

/-- Represents a tile with width and height -/
structure Tile :=
  (width : Nat)
  (height : Nat)

/-- Represents a rectangle with width and height -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- Represents a set of tiles -/
def TileSet := List Tile

/-- Counts the number of ways to tile a rectangle with a given set of tiles -/
def tileCount (r : Rectangle) (ts : TileSet) : Nat :=
  sorry

/-- The set of tiles for our problem -/
def problemTiles : TileSet :=
  [⟨1, 1⟩, ⟨1, 2⟩, ⟨1, 3⟩, ⟨1, 4⟩, ⟨1, 5⟩]

/-- The main theorem stating that the number of tilings is 96 -/
theorem tiling_count_is_96 :
  tileCount ⟨5, 3⟩ problemTiles = 96 :=
sorry

end tiling_count_is_96_l2670_267032


namespace no_convex_quadrilateral_with_all_acute_triangles_l2670_267097

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Condition for convexity

-- Define an acute-angled triangle
def is_acute_angled_triangle (a b c : ℝ × ℝ) : Prop :=
  sorry -- Condition for all angles being less than 90 degrees

-- Define a diagonal of a quadrilateral
def diagonal (q : ConvexQuadrilateral) (i j : Fin 4) : Prop :=
  sorry -- Condition for i and j being opposite vertices

-- Theorem statement
theorem no_convex_quadrilateral_with_all_acute_triangles :
  ¬ ∃ (q : ConvexQuadrilateral),
    ∀ (i j : Fin 4), diagonal q i j →
      is_acute_angled_triangle (q.vertices i) (q.vertices j) (q.vertices ((i + 1) % 4)) ∧
      is_acute_angled_triangle (q.vertices i) (q.vertices j) (q.vertices ((j + 1) % 4)) :=
sorry

end no_convex_quadrilateral_with_all_acute_triangles_l2670_267097


namespace quadratic_inequality_condition_l2670_267026

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end quadratic_inequality_condition_l2670_267026


namespace negative_of_negative_greater_than_negative_of_positive_l2670_267067

theorem negative_of_negative_greater_than_negative_of_positive :
  -(-1) > -(2) := by
  sorry

end negative_of_negative_greater_than_negative_of_positive_l2670_267067


namespace total_visitors_three_days_l2670_267051

/-- The number of visitors to Buckingham Palace on the day Rachel visited -/
def visitors_rachel_day : ℕ := 92

/-- The number of visitors to Buckingham Palace on the day before Rachel's visit -/
def visitors_previous_day : ℕ := 419

/-- The number of visitors to Buckingham Palace two days before Rachel's visit -/
def visitors_two_days_before : ℕ := 103

/-- Theorem stating that the total number of visitors over the three known days is 614 -/
theorem total_visitors_three_days : 
  visitors_rachel_day + visitors_previous_day + visitors_two_days_before = 614 := by
  sorry

end total_visitors_three_days_l2670_267051


namespace smallest_number_with_conditions_l2670_267017

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than_60 (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < 60 → ¬(p ∣ n)

theorem smallest_number_with_conditions : 
  (∀ n : ℕ, n < 4087 → 
    is_prime n ∨ 
    is_square n ∨ 
    ¬(has_no_prime_factor_less_than_60 n)) ∧ 
  ¬(is_prime 4087) ∧ 
  ¬(is_square 4087) ∧ 
  has_no_prime_factor_less_than_60 4087 :=
sorry

end smallest_number_with_conditions_l2670_267017


namespace second_bell_interval_l2670_267062

def bell_intervals (x : ℕ) : List ℕ := [5, x, 11, 15]

theorem second_bell_interval (x : ℕ) :
  (∃ (k : ℕ), k > 0 ∧ k * (Nat.lcm (Nat.lcm (Nat.lcm 5 x) 11) 15) = 1320) →
  x = 8 :=
by sorry

end second_bell_interval_l2670_267062


namespace product_purchase_savings_l2670_267048

/-- Proves that under given conditions, the product could have been purchased for 10% less -/
theorem product_purchase_savings (original_selling_price : ℝ) 
  (h1 : original_selling_price = 989.9999999999992)
  (h2 : original_selling_price = 1.1 * original_purchase_price)
  (h3 : 1.3 * reduced_purchase_price = original_selling_price + 63) :
  (original_purchase_price - reduced_purchase_price) / original_purchase_price = 0.1 := by
  sorry

end product_purchase_savings_l2670_267048


namespace min_total_routes_l2670_267073

/-- Represents the number of routes for each airline company -/
structure AirlineRoutes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The minimum number of routes needed to maintain connectivity -/
def min_connectivity : ℕ := 14

/-- The total number of cities in the country -/
def num_cities : ℕ := 15

/-- Predicate to check if the network remains connected after removing any one company's routes -/
def remains_connected (routes : AirlineRoutes) : Prop :=
  routes.a + routes.b ≥ min_connectivity ∧
  routes.b + routes.c ≥ min_connectivity ∧
  routes.c + routes.a ≥ min_connectivity

/-- Theorem stating the minimum number of total routes needed -/
theorem min_total_routes (routes : AirlineRoutes) :
  remains_connected routes → routes.a + routes.b + routes.c ≥ 21 := by
  sorry


end min_total_routes_l2670_267073


namespace volume_to_surface_area_ratio_l2670_267040

/-- A structure composed of unit cubes -/
structure CubeStructure where
  /-- The number of unit cubes in the structure -/
  num_cubes : ℕ
  /-- The number of cubes surrounding the center cube -/
  surrounding_cubes : ℕ
  /-- Assertion that there is one center cube -/
  has_center_cube : num_cubes = surrounding_cubes + 1

/-- Calculate the volume of the structure -/
def volume (s : CubeStructure) : ℕ := s.num_cubes

/-- Calculate the surface area of the structure -/
def surface_area (s : CubeStructure) : ℕ :=
  1 + (s.surrounding_cubes - 1) * 5 + 4

/-- The theorem to be proved -/
theorem volume_to_surface_area_ratio (s : CubeStructure) 
  (h1 : s.num_cubes = 10) 
  (h2 : s.surrounding_cubes = 9) : 
  (volume s : ℚ) / (surface_area s : ℚ) = 2/9 := by
  sorry

end volume_to_surface_area_ratio_l2670_267040


namespace positive_number_squared_plus_self_l2670_267088

theorem positive_number_squared_plus_self (n : ℝ) : n > 0 ∧ n^2 + n = 210 → n = 14 := by
  sorry

end positive_number_squared_plus_self_l2670_267088


namespace smallest_N_proof_l2670_267037

def f (n : ℕ+) : ℕ := sorry

def g (n : ℕ+) : ℕ := sorry

def N : ℕ+ := sorry

theorem smallest_N_proof : N = 44 ∧ (∀ m : ℕ+, m < N → g m < 11) ∧ g N ≥ 11 := by sorry

end smallest_N_proof_l2670_267037


namespace students_speaking_both_languages_l2670_267085

theorem students_speaking_both_languages (total : ℕ) (english : ℕ) (japanese : ℕ) (neither : ℕ) :
  total = 50 →
  english = 36 →
  japanese = 20 →
  neither = 8 →
  ∃ x : ℕ, x = 14 ∧ 
    x = english + japanese - (total - neither) :=
by sorry

end students_speaking_both_languages_l2670_267085


namespace hannah_stocking_stuffers_l2670_267055

/-- The number of candy canes per stocking -/
def candy_canes_per_stocking : ℕ := 4

/-- The number of beanie babies per stocking -/
def beanie_babies_per_stocking : ℕ := 2

/-- The number of books per stocking -/
def books_per_stocking : ℕ := 1

/-- The number of kids Hannah has -/
def number_of_kids : ℕ := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stocking_stuffers : ℕ := 
  (candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking) * number_of_kids

theorem hannah_stocking_stuffers : total_stocking_stuffers = 21 := by
  sorry

end hannah_stocking_stuffers_l2670_267055


namespace sum_of_inserted_numbers_l2670_267078

/-- A sequence of five real numbers -/
structure Sequence :=
  (a b c : ℝ)

/-- Check if the first four terms form a harmonic progression -/
def isHarmonicProgression (s : Sequence) : Prop :=
  ∃ (h : ℝ), 1/4 - 1/s.a = 1/s.a - 1/s.b ∧ 1/s.a - 1/s.b = 1/s.b - 1/s.c

/-- Check if the last four terms form a quadratic sequence -/
def isQuadraticSequence (s : Sequence) : Prop :=
  ∃ (p q : ℝ), 
    s.a = 1^2 + p + q ∧
    s.b = 2^2 + 2*p + q ∧
    s.c = 3^2 + 3*p + q ∧
    16 = 4^2 + 4*p + q

/-- The main theorem -/
theorem sum_of_inserted_numbers (s : Sequence) :
  s.a > 0 ∧ s.b > 0 ∧ s.c > 0 →
  isHarmonicProgression s →
  isQuadraticSequence s →
  s.a + s.b + s.c = 33 :=
sorry

end sum_of_inserted_numbers_l2670_267078


namespace sum_equals_300_l2670_267039

theorem sum_equals_300 : 157 + 43 + 19 + 81 = 300 := by
  sorry

end sum_equals_300_l2670_267039


namespace divisibility_conditions_l2670_267024

theorem divisibility_conditions (a b : ℕ) : 
  (∃ k : ℤ, a^3 * b - 1 = k * (a + 1)) ∧ 
  (∃ m : ℤ, a * b^3 + 1 = m * (b - 1)) → 
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) := by
  sorry

end divisibility_conditions_l2670_267024


namespace exists_valid_a_l2670_267068

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {0, a}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem exists_valid_a : ∃ a : ℝ, A a ⊆ B ∧ a = 1 := by
  sorry

end exists_valid_a_l2670_267068


namespace cades_remaining_marbles_l2670_267002

/-- Proves that Cade has 79 marbles left after giving away 8 marbles from his initial 87 marbles. -/
theorem cades_remaining_marbles (initial_marbles : ℕ) (marbles_given_away : ℕ) 
  (h1 : initial_marbles = 87) 
  (h2 : marbles_given_away = 8) : 
  initial_marbles - marbles_given_away = 79 := by
  sorry

end cades_remaining_marbles_l2670_267002


namespace perpendicular_to_parallel_plane_parallel_line_to_parallel_plane_l2670_267096

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (planesParallel : Plane → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)

-- Theorem for proposition ②
theorem perpendicular_to_parallel_plane 
  (m n : Line) (α : Plane)
  (h1 : perpendicularToPlane m α)
  (h2 : parallelToPlane n α) :
  perpendicular m n :=
sorry

-- Theorem for proposition ③
theorem parallel_line_to_parallel_plane 
  (m : Line) (α β : Plane)
  (h1 : planesParallel α β)
  (h2 : lineInPlane m α) :
  parallelToPlane m β :=
sorry

end perpendicular_to_parallel_plane_parallel_line_to_parallel_plane_l2670_267096


namespace prob_different_fruits_l2670_267074

/-- The number of fruit types available --/
def num_fruits : ℕ := 5

/-- The number of meals over two days --/
def num_meals : ℕ := 6

/-- The probability of choosing a specific fruit for all meals --/
def prob_same_fruit : ℚ := (1 / num_fruits) ^ num_meals

/-- The probability of eating at least two different kinds of fruit over two days --/
theorem prob_different_fruits : 
  1 - num_fruits * prob_same_fruit = 15620 / 15625 := by
  sorry

end prob_different_fruits_l2670_267074


namespace simplify_expression_l2670_267083

theorem simplify_expression : Real.sqrt ((Real.pi - 4) ^ 2) + (Real.pi - 3) = 1 := by
  sorry

end simplify_expression_l2670_267083


namespace pokemon_card_count_l2670_267091

/-- The number of people who have Pokemon cards -/
def num_people : ℕ := 4

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 14

/-- The total number of Pokemon cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem pokemon_card_count : total_cards = 56 := by
  sorry

end pokemon_card_count_l2670_267091


namespace expression_evaluation_l2670_267060

theorem expression_evaluation (b : ℝ) (h : b = 3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / (2 * b) = 5 / 27 := by
  sorry

end expression_evaluation_l2670_267060


namespace range_of_a_l2670_267044

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 := by
  sorry

end range_of_a_l2670_267044


namespace inequality_proof_l2670_267014

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 3) : 
  (a / Real.sqrt (a^3 + 5)) + (b / Real.sqrt (b^3 + 5)) + (c / Real.sqrt (c^3 + 5)) ≤ Real.sqrt 6 / 2 := by
  sorry

end inequality_proof_l2670_267014


namespace exam_score_distribution_l2670_267016

/-- Represents the normal distribution of exam scores -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ

/-- Represents the class and exam information -/
structure ExamInfo where
  totalStudents : ℕ
  scoreDistribution : NormalDistribution
  middleProbability : ℝ

/-- Calculates the number of students with scores above a given threshold -/
def studentsAboveThreshold (info : ExamInfo) (threshold : ℝ) : ℕ :=
  sorry

theorem exam_score_distribution (info : ExamInfo) :
  info.totalStudents = 50 ∧
  info.scoreDistribution = { μ := 110, σ := 10 } ∧
  info.middleProbability = 0.34 →
  studentsAboveThreshold info 120 = 8 :=
sorry

end exam_score_distribution_l2670_267016


namespace actual_distance_is_1542_l2670_267080

/-- Represents a faulty odometer that skips digits 4 and 7 --/
def FaultyOdometer := ℕ → ℕ

/-- The current reading of the odometer --/
def current_reading : ℕ := 2056

/-- The function that calculates the actual distance traveled --/
def actual_distance (o : FaultyOdometer) (reading : ℕ) : ℕ := sorry

/-- Theorem stating that the actual distance traveled is 1542 miles --/
theorem actual_distance_is_1542 (o : FaultyOdometer) :
  actual_distance o current_reading = 1542 := by sorry

end actual_distance_is_1542_l2670_267080


namespace sues_mother_cookies_l2670_267042

/-- The number of cookies Sue's mother made -/
def total_cookies (bags : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  bags * cookies_per_bag

/-- Proof that Sue's mother made 75 cookies -/
theorem sues_mother_cookies : total_cookies 25 3 = 75 := by
  sorry

end sues_mother_cookies_l2670_267042


namespace specific_normal_distribution_two_std_devs_less_l2670_267099

/-- Represents a normal distribution --/
structure NormalDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation

/-- The value that is exactly 2 standard deviations less than the mean --/
def twoStdDevsLessThanMean (nd : NormalDistribution) : ℝ :=
  nd.μ - 2 * nd.σ

/-- Theorem statement for the given problem --/
theorem specific_normal_distribution_two_std_devs_less (nd : NormalDistribution) 
  (h1 : nd.μ = 16.5) (h2 : nd.σ = 1.5) : 
  twoStdDevsLessThanMean nd = 13.5 := by
  sorry

end specific_normal_distribution_two_std_devs_less_l2670_267099


namespace fermat_numbers_coprime_l2670_267035

theorem fermat_numbers_coprime (m n : ℕ) (h : m ≠ n) :
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := by
  sorry

end fermat_numbers_coprime_l2670_267035


namespace square_area_l2670_267072

theorem square_area (side : ℝ) (h : side = 6) : side * side = 36 := by
  sorry

end square_area_l2670_267072


namespace sodium_hydroxide_moles_l2670_267098

/-- Represents the chemical reaction between Sodium hydroxide and Chlorine to produce Water -/
structure ChemicalReaction where
  naoh : ℝ  -- moles of Sodium hydroxide
  cl2 : ℝ   -- moles of Chlorine
  h2o : ℝ   -- moles of Water produced

/-- The stoichiometric ratio of the reaction -/
def stoichiometricRatio : ℝ := 2

theorem sodium_hydroxide_moles (reaction : ChemicalReaction) 
  (h1 : reaction.cl2 = 2)
  (h2 : reaction.h2o = 2)
  (h3 : reaction.naoh = stoichiometricRatio * reaction.h2o) :
  reaction.naoh = 4 := by
  sorry

end sodium_hydroxide_moles_l2670_267098


namespace abs_equation_solution_l2670_267010

theorem abs_equation_solution (x : ℝ) : |-5 + x| = 3 → x = 8 ∨ x = 2 := by
  sorry

end abs_equation_solution_l2670_267010


namespace tims_interest_rate_l2670_267007

/-- Calculates the compound interest after n years -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

theorem tims_interest_rate :
  let tim_principal : ℝ := 600
  let lana_principal : ℝ := 1000
  let lana_rate : ℝ := 0.05
  let years : ℕ := 2
  ∀ tim_rate : ℝ,
    (compoundInterest tim_principal tim_rate years - tim_principal) =
    (compoundInterest lana_principal lana_rate years - lana_principal) + 23.5 →
    tim_rate = 0.1 := by
  sorry

#check tims_interest_rate

end tims_interest_rate_l2670_267007


namespace sanchez_rope_theorem_l2670_267066

/-- The amount of rope in feet bought last week -/
def rope_last_week : ℕ := 6

/-- The difference in feet between last week's and this week's rope purchase -/
def rope_difference : ℕ := 4

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The total amount of rope bought in inches -/
def total_rope_inches : ℕ := 
  (rope_last_week * inches_per_foot) + 
  ((rope_last_week - rope_difference) * inches_per_foot)

theorem sanchez_rope_theorem : total_rope_inches = 96 := by
  sorry

end sanchez_rope_theorem_l2670_267066


namespace fifteenth_student_age_l2670_267023

theorem fifteenth_student_age
  (total_students : Nat)
  (average_age : ℚ)
  (group1_size group2_size : Nat)
  (group1_average group2_average : ℚ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_size = 7)
  (h4 : group2_size = 7)
  (h5 : group1_average = 14)
  (h6 : group2_average = 16) :
  (total_students * average_age - (group1_size * group1_average + group2_size * group2_average)) / (total_students - group1_size - group2_size) = 15 := by
  sorry


end fifteenth_student_age_l2670_267023


namespace rosie_apple_crisps_l2670_267018

/-- The number of apple crisps Rosie can make with a given number of apples -/
def apple_crisps (apples : ℕ) : ℕ :=
  (3 * apples) / 12

theorem rosie_apple_crisps :
  apple_crisps 36 = 9 := by
  sorry

end rosie_apple_crisps_l2670_267018


namespace kids_wearing_socks_and_shoes_l2670_267084

/-- Given a classroom with kids wearing socks, shoes, or barefoot, 
    prove the number of kids wearing both socks and shoes. -/
theorem kids_wearing_socks_and_shoes 
  (total : ℕ) 
  (socks : ℕ) 
  (shoes : ℕ) 
  (barefoot : ℕ) 
  (h1 : total = 22) 
  (h2 : socks = 12) 
  (h3 : shoes = 8) 
  (h4 : barefoot = 8) 
  (h5 : total = socks + barefoot) 
  (h6 : total = shoes + barefoot) :
  shoes = socks + shoes - total := by
sorry

end kids_wearing_socks_and_shoes_l2670_267084


namespace multiplicative_inverse_exists_l2670_267045

theorem multiplicative_inverse_exists : ∃ N : ℕ, 
  N > 0 ∧ 
  N < 1000000 ∧ 
  (123456 * 654321 * N) % 1234567 = 1 := by
sorry

end multiplicative_inverse_exists_l2670_267045


namespace tina_pen_difference_l2670_267070

/-- Prove that Tina has 3 more blue pens than green pens -/
theorem tina_pen_difference : 
  ∀ (pink green blue : ℕ),
  pink = 12 →
  green = pink - 9 →
  blue > green →
  pink + green + blue = 21 →
  blue - green = 3 :=
by
  sorry

end tina_pen_difference_l2670_267070


namespace abs_z_equals_sqrt_two_l2670_267029

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem abs_z_equals_sqrt_two (z : ℂ) (h : z * (1 + i) = 2 * i) : Complex.abs z = Real.sqrt 2 := by
  sorry

end abs_z_equals_sqrt_two_l2670_267029


namespace badminton_racket_purchase_l2670_267027

theorem badminton_racket_purchase
  (total_pairs : ℕ)
  (cost_A : ℕ)
  (cost_B : ℕ)
  (total_cost : ℕ)
  (h1 : total_pairs = 30)
  (h2 : cost_A = 50)
  (h3 : cost_B = 40)
  (h4 : total_cost = 1360) :
  ∃ (pairs_A pairs_B : ℕ),
    pairs_A + pairs_B = total_pairs ∧
    pairs_A * cost_A + pairs_B * cost_B = total_cost ∧
    pairs_A = 16 ∧
    pairs_B = 14 := by
  sorry

end badminton_racket_purchase_l2670_267027


namespace arithmetic_geometric_mean_ratio_existence_l2670_267013

theorem arithmetic_geometric_mean_ratio_existence :
  ∃ (a b : ℝ), 
    (a + b) / 2 = 3 * Real.sqrt (a * b) ∧
    a > b ∧ b > 0 ∧
    round (a / b) = 28 := by
  sorry

end arithmetic_geometric_mean_ratio_existence_l2670_267013


namespace inequality_proof_l2670_267025

theorem inequality_proof (a b : ℝ) (h : a + b = 1) :
  Real.sqrt (1 + 5 * a^2) + 5 * Real.sqrt (2 + b^2) ≥ 9 := by sorry

end inequality_proof_l2670_267025


namespace sams_phone_bill_l2670_267053

-- Define the constants from the problem
def base_cost : ℚ := 25
def text_cost : ℚ := 8 / 100  -- 8 cents in dollars
def extra_minute_cost : ℚ := 15 / 100  -- 15 cents in dollars
def included_hours : ℕ := 25
def texts_sent : ℕ := 150
def hours_talked : ℕ := 26

-- Define the function to calculate the total cost
def calculate_total_cost : ℚ :=
  let text_total := text_cost * texts_sent
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_minutes_total := extra_minute_cost * extra_minutes
  base_cost + text_total + extra_minutes_total

-- State the theorem
theorem sams_phone_bill : calculate_total_cost = 46 := by sorry

end sams_phone_bill_l2670_267053


namespace cube_preserves_order_l2670_267015

theorem cube_preserves_order (a b c : ℝ) (h : b > a) : b^3 > a^3 := by
  sorry

end cube_preserves_order_l2670_267015


namespace min_value_sum_of_distances_min_value_achievable_l2670_267092

theorem min_value_sum_of_distances (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2) ≥ 6 * Real.sqrt 2 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2) = 6 * Real.sqrt 2 := by
  sorry

end min_value_sum_of_distances_min_value_achievable_l2670_267092


namespace organization_members_l2670_267000

/-- The number of committees in the organization -/
def num_committees : ℕ := 5

/-- The number of committees each member belongs to -/
def committees_per_member : ℕ := 2

/-- The number of unique members shared between each pair of committees -/
def shared_members_per_pair : ℕ := 2

/-- The total number of members in the organization -/
def total_members : ℕ := 10

/-- Theorem stating the total number of members in the organization -/
theorem organization_members :
  (num_committees = 5) →
  (committees_per_member = 2) →
  (shared_members_per_pair = 2) →
  (total_members = 10) :=
by sorry

end organization_members_l2670_267000


namespace power_of_power_l2670_267064

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end power_of_power_l2670_267064


namespace aarti_work_completion_l2670_267022

/-- The number of days Aarti needs to complete one piece of work -/
def days_for_one_work : ℕ := 5

/-- The number of times the work is multiplied -/
def work_multiplier : ℕ := 3

/-- Theorem: Aarti will complete three times the work in 15 days -/
theorem aarti_work_completion :
  days_for_one_work * work_multiplier = 15 := by
  sorry

end aarti_work_completion_l2670_267022


namespace simplify_and_rationalize_l2670_267082

theorem simplify_and_rationalize : 
  (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 10 / Real.sqrt 15) * (Real.sqrt 12 / Real.sqrt 20) = 1 / 3 := by
  sorry

end simplify_and_rationalize_l2670_267082


namespace smallest_difference_of_powers_eleven_is_representable_eleven_is_smallest_l2670_267079

theorem smallest_difference_of_powers : 
  ∀ k l : ℕ, 36^k - 5^l > 0 → 36^k - 5^l ≥ 11 :=
by sorry

theorem eleven_is_representable : 
  ∃ k l : ℕ, 36^k - 5^l = 11 :=
by sorry

theorem eleven_is_smallest :
  (∃ k l : ℕ, 36^k - 5^l = 11) ∧
  (∀ m n : ℕ, 36^m - 5^n > 0 → 36^m - 5^n ≥ 11) :=
by sorry

end smallest_difference_of_powers_eleven_is_representable_eleven_is_smallest_l2670_267079


namespace pencil_pen_difference_l2670_267063

/-- Given a ratio of pens to pencils and the total number of pencils,
    calculate the difference between pencils and pens. -/
theorem pencil_pen_difference
  (ratio_pens : ℕ)
  (ratio_pencils : ℕ)
  (total_pencils : ℕ)
  (h_ratio : ratio_pens < ratio_pencils)
  (h_total : total_pencils = 36)
  (h_ratio_pencils : total_pencils % ratio_pencils = 0) :
  total_pencils - (total_pencils / ratio_pencils * ratio_pens) = 6 :=
by sorry

end pencil_pen_difference_l2670_267063


namespace triangle_properties_l2670_267008

theorem triangle_properties (A B C : Real) (a b c : Real) 
  (m_x m_y n_x n_y : Real → Real) :
  (∀ θ, m_x θ = 2 * Real.cos θ ∧ m_y θ = 1) →
  (∀ θ, n_x θ = 1 ∧ n_y θ = Real.sin (θ + Real.pi / 6)) →
  (∃ k : Real, k ≠ 0 ∧ ∀ θ, m_x θ * k = n_x θ ∧ m_y θ * k = n_y θ) →
  a = 2 * Real.sqrt 3 →
  c = 4 →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = Real.pi / 3 ∧ 
  b = 2 ∧ 
  1/2 * b * c * Real.sin A = 2 * Real.sqrt 3 :=
by sorry

end triangle_properties_l2670_267008


namespace tangent_intercept_implies_a_value_l2670_267003

/-- A function f(x) = ax³ + 4x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x + 5

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4

theorem tangent_intercept_implies_a_value (a : ℝ) :
  (f' a 1 * (-3/7 - 1) + f a 1 = 0) → a = 1 := by
  sorry

end tangent_intercept_implies_a_value_l2670_267003


namespace exists_valid_coloring_l2670_267090

/-- A coloring function that assigns a color (represented by a Boolean) to each point with integer coordinates. -/
def Coloring := ℤ × ℤ → Bool

/-- Predicate to check if a rectangle satisfies the required properties. -/
def ValidRectangle (a b c d : ℤ × ℤ) : Prop :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  let (x₃, y₃) := c
  let (x₄, y₄) := d
  (x₁ = x₄ ∧ x₂ = x₃ ∧ y₁ = y₂ ∧ y₃ = y₄) ∧ 
  (∃ k : ℕ, (x₂ - x₁).natAbs * (y₃ - y₁).natAbs = 2^k)

/-- Theorem stating that there exists a coloring such that no valid rectangle has all vertices of the same color. -/
theorem exists_valid_coloring : 
  ∃ (f : Coloring), ∀ (a b c d : ℤ × ℤ), 
    ValidRectangle a b c d → 
    ¬(f a = f b ∧ f b = f c ∧ f c = f d) :=
sorry

end exists_valid_coloring_l2670_267090


namespace minimal_n_for_square_product_set_l2670_267089

theorem minimal_n_for_square_product_set (m : ℕ+) (p : ℕ) (h1 : p.Prime) (h2 : p ∣ m) 
  (h3 : p > Real.sqrt (2 * m) + 1) :
  ∃ (n : ℕ), n = m + p ∧
  (∀ (k : ℕ), k < n → 
    ¬∃ (S : Finset ℕ), 
      (∀ x ∈ S, m ≤ x ∧ x ≤ k) ∧ 
      (∃ y : ℕ, (S.prod id : ℕ) = y * y)) ∧
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, m ≤ x ∧ x ≤ n) ∧ 
    (∃ y : ℕ, (S.prod id : ℕ) = y * y) :=
by sorry

end minimal_n_for_square_product_set_l2670_267089


namespace yannas_cookies_l2670_267020

/-- Yanna's cookie baking problem -/
theorem yannas_cookies
  (morning_butter_cookies : ℕ)
  (morning_biscuits : ℕ)
  (afternoon_butter_cookies : ℕ)
  (afternoon_biscuits : ℕ)
  (h1 : morning_butter_cookies = 20)
  (h2 : morning_biscuits = 40)
  (h3 : afternoon_butter_cookies = 10)
  (h4 : afternoon_biscuits = 20) :
  (morning_biscuits + afternoon_biscuits) - (morning_butter_cookies + afternoon_butter_cookies) = 30 :=
by sorry

end yannas_cookies_l2670_267020


namespace rectangle_problem_l2670_267075

theorem rectangle_problem (l b : ℝ) : 
  l = 2 * b →
  (l - 5) * (b + 5) - l * b = 75 →
  20 < l ∧ l < 50 →
  10 < b ∧ b < 30 →
  l = 40 := by
sorry

end rectangle_problem_l2670_267075


namespace polygon_sides_count_l2670_267069

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by sorry

end polygon_sides_count_l2670_267069


namespace faye_candy_eaten_l2670_267001

/-- Represents the number of candy pieces Faye ate on the first night -/
def candy_eaten (initial : ℕ) (received : ℕ) (final : ℕ) : ℕ :=
  initial + received - final

theorem faye_candy_eaten : 
  candy_eaten 47 40 62 = 25 := by
sorry

end faye_candy_eaten_l2670_267001


namespace triangle_angle_measure_l2670_267071

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  (c - b) / (Real.sqrt 2 * c - a) = Real.sin A / (Real.sin B + Real.sin C) →
  B = π / 4 := by
sorry

end triangle_angle_measure_l2670_267071


namespace simplify_expression_l2670_267076

theorem simplify_expression (a b : ℝ) : a - 4*(2*a - b) - 2*(a + 2*b) = -9*a := by
  sorry

end simplify_expression_l2670_267076


namespace cuboid_third_edge_length_l2670_267047

/-- Given a cuboid with two edges of 4 cm each and a volume of 96 cm³, 
    prove that the length of the third edge is 6 cm. -/
theorem cuboid_third_edge_length 
  (edge1 : ℝ) (edge2 : ℝ) (volume : ℝ) (third_edge : ℝ) 
  (h1 : edge1 = 4) 
  (h2 : edge2 = 4) 
  (h3 : volume = 96) 
  (h4 : volume = edge1 * edge2 * third_edge) : 
  third_edge = 6 :=
sorry

end cuboid_third_edge_length_l2670_267047


namespace tangent_line_at_one_l2670_267028

noncomputable section

variable (f : ℝ → ℝ)

-- Define the function property
def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = 2 * f (2 * x - 1) - 3 * x^2 + 2

-- Define the tangent line equation
def tangent_line_equation (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ 2 * x - 1

-- Theorem statement
theorem tangent_line_at_one (h : function_property f) :
  ∃ (f' : ℝ → ℝ), (∀ x, HasDerivAt f (f' x) x) ∧
  (∀ x, (tangent_line_equation f) x = f 1 + f' 1 * (x - 1)) :=
sorry

end

end tangent_line_at_one_l2670_267028


namespace smallest_prime_with_digit_sum_23_l2670_267049

-- Define a function to calculate the sum of digits
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

-- Define primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

-- Theorem statement
theorem smallest_prime_with_digit_sum_23 :
  ∀ n : ℕ, is_prime n ∧ digit_sum n = 23 → n ≥ 599 :=
sorry

end smallest_prime_with_digit_sum_23_l2670_267049


namespace smallest_integer_with_remainders_l2670_267034

theorem smallest_integer_with_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 3 = 2 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧ 
  ∀ (y : ℕ), y > 0 ∧ y % 3 = 2 ∧ y % 4 = 3 ∧ y % 5 = 4 → x ≤ y :=
by
  use 59
  sorry

end smallest_integer_with_remainders_l2670_267034


namespace max_xy_value_l2670_267065

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  ∃ (max_val : ℝ), max_val = 1/8 ∧ ∀ (z : ℝ), x*y ≤ z ∧ z ≤ max_val :=
sorry

end max_xy_value_l2670_267065


namespace rug_profit_calculation_l2670_267046

/-- Calculate the profit from selling rugs -/
theorem rug_profit_calculation (cost_price selling_price number_of_rugs : ℕ) :
  let profit_per_rug := selling_price - cost_price
  let total_profit := number_of_rugs * profit_per_rug
  total_profit = number_of_rugs * (selling_price - cost_price) :=
by sorry

end rug_profit_calculation_l2670_267046
