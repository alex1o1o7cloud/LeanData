import Mathlib

namespace regular_polygon_interior_angle_sum_l3251_325125

theorem regular_polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) : 
  (360 / n = 45) → (n - 2) * 180 = 1080 := by
  sorry

end regular_polygon_interior_angle_sum_l3251_325125


namespace birthday_money_calculation_l3251_325190

/-- The amount of money Sam spent on baseball gear -/
def amount_spent : ℕ := 64

/-- The amount of money Sam had left over -/
def amount_left : ℕ := 23

/-- The total amount of money Sam received for his birthday -/
def total_amount : ℕ := amount_spent + amount_left

/-- Theorem stating that the total amount Sam received is the sum of what he spent and what he had left -/
theorem birthday_money_calculation : total_amount = 87 := by
  sorry

end birthday_money_calculation_l3251_325190


namespace game_points_total_l3251_325170

theorem game_points_total (layla_points nahima_points : ℕ) : 
  layla_points = 70 → 
  layla_points = nahima_points + 28 → 
  layla_points + nahima_points = 112 := by
sorry

end game_points_total_l3251_325170


namespace coin_probability_theorem_l3251_325195

theorem coin_probability_theorem (p q : ℝ) : 
  p + q = 1 →
  0 ≤ p ∧ p ≤ 1 →
  0 ≤ q ∧ q ≤ 1 →
  (Nat.choose 10 5 : ℝ) * p^5 * q^5 = (Nat.choose 10 6 : ℝ) * p^6 * q^4 →
  p = 6/11 :=
by sorry

end coin_probability_theorem_l3251_325195


namespace triangle_side_value_l3251_325182

noncomputable section

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Conditions for a valid triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem triangle_side_value
  (A B C : ℝ) (a b c : ℝ)
  (h_triangle : triangle A B C a b c)
  (h_angle : A = 2 * C)
  (h_side_c : c = 2)
  (h_side_a : a^2 = 4*b - 4) :
  a = 2 * Real.sqrt 3 :=
sorry

end triangle_side_value_l3251_325182


namespace smallest_total_blocks_smallest_total_blocks_exist_l3251_325118

/-- Represents the dimensions of a cubic block -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cubic pedestal -/
structure Pedestal where
  sideLength : ℕ

/-- Represents a square foundation -/
structure Foundation where
  sideLength : ℕ
  thickness : ℕ

/-- Calculates the volume of a pedestal in terms of blocks -/
def pedestalVolume (p : Pedestal) : ℕ :=
  p.sideLength ^ 3

/-- Calculates the volume of a foundation in terms of blocks -/
def foundationVolume (f : Foundation) : ℕ :=
  f.sideLength ^ 2 * f.thickness

theorem smallest_total_blocks : ℕ × ℕ → Prop
  | (pedestal_side, foundation_side) =>
    let block : Block := ⟨1, 1, 1⟩
    let pedestal : Pedestal := ⟨pedestal_side⟩
    let foundation : Foundation := ⟨foundation_side, 1⟩
    (pedestalVolume pedestal = foundationVolume foundation) ∧
    (foundation_side = pedestal_side ^ (3/2)) ∧
    (pedestalVolume pedestal + foundationVolume foundation = 128) ∧
    ∀ (p : Pedestal) (f : Foundation),
      (pedestalVolume p = foundationVolume f) →
      (f.sideLength = p.sideLength ^ (3/2)) →
      (pedestalVolume p + foundationVolume f ≥ 128)

theorem smallest_total_blocks_exist :
  ∃ (pedestal_side foundation_side : ℕ),
    smallest_total_blocks (pedestal_side, foundation_side) :=
  sorry

end smallest_total_blocks_smallest_total_blocks_exist_l3251_325118


namespace function_properties_l3251_325159

noncomputable def f (ω θ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + θ)

theorem function_properties (ω θ : ℝ) (h_ω : ω > 0) (h_θ : 0 ≤ θ ∧ θ ≤ π/2)
  (h_intersect : f ω θ 0 = Real.sqrt 3)
  (h_period : ∀ x, f ω θ (x + π) = f ω θ x) :
  (θ = π/6 ∧ ω = 2) ∧
  (∃ x₀ ∈ Set.Icc (π/2) π,
    let y₀ := Real.sqrt 3 / 2
    let x₁ := 2 * x₀ - π/2
    let y₁ := f ω θ x₁
    y₀ = (y₁ + 0) / 2 ∧ (x₀ = 2*π/3 ∨ x₀ = 3*π/4)) :=
by sorry

end function_properties_l3251_325159


namespace kishore_rent_expense_l3251_325148

def monthly_salary (savings : ℕ) : ℕ := savings * 10

def total_expenses_excluding_rent : ℕ := 1500 + 4500 + 2500 + 2000 + 5200

def rent_expense (salary savings : ℕ) : ℕ :=
  salary - (total_expenses_excluding_rent + savings)

theorem kishore_rent_expense :
  rent_expense (monthly_salary 2300) 2300 = 5000 := by
  sorry

end kishore_rent_expense_l3251_325148


namespace quadratic_sum_l3251_325169

/-- A quadratic function f(x) = ax^2 + bx + c with a minimum value of 36
    and roots at x = 1 and x = 5 has the property that a + b + c = 0 -/
theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c ≥ 36) ∧ 
  (∃ x₀, ∀ x, a * x^2 + b * x + c ≥ a * x₀^2 + b * x₀ + c ∧ a * x₀^2 + b * x₀ + c = 36) ∧
  (a * 1^2 + b * 1 + c = 0) ∧
  (a * 5^2 + b * 5 + c = 0) →
  a + b + c = 0 := by
  sorry

end quadratic_sum_l3251_325169


namespace full_price_tickets_count_l3251_325141

/-- Represents the number of tickets sold at each price point -/
structure TicketSales where
  full : ℕ
  half : ℕ
  double : ℕ

/-- Represents the price of a full-price ticket -/
def FullPrice : ℕ := 30

/-- The total number of tickets sold -/
def TotalTickets : ℕ := 200

/-- The total revenue from all ticket sales -/
def TotalRevenue : ℕ := 3600

/-- Calculates the total number of tickets sold -/
def totalTicketCount (sales : TicketSales) : ℕ :=
  sales.full + sales.half + sales.double

/-- Calculates the total revenue from all ticket sales -/
def totalRevenue (sales : TicketSales) : ℕ :=
  sales.full * FullPrice + sales.half * (FullPrice / 2) + sales.double * (2 * FullPrice)

/-- Theorem stating that the number of full-price tickets sold is 80 -/
theorem full_price_tickets_count :
  ∃ (sales : TicketSales),
    totalTicketCount sales = TotalTickets ∧
    totalRevenue sales = TotalRevenue ∧
    sales.full = 80 :=
by sorry

end full_price_tickets_count_l3251_325141


namespace bubble_gum_cost_l3251_325189

theorem bubble_gum_cost (total_pieces : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) : 
  total_pieces = 136 → total_cost = 2448 → cost_per_piece = 18 → 
  total_cost = total_pieces * cost_per_piece :=
by sorry

end bubble_gum_cost_l3251_325189


namespace sum_200th_row_l3251_325194

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ := sorry

/-- The triangular array has the following properties:
    1. The sides contain numbers 0, 1, 2, 3, ...
    2. Each interior number is the sum of two adjacent numbers in the previous row -/
axiom array_properties : True

/-- The sum of numbers in the nth row follows the recurrence relation:
    f(n) = 2 * f(n-1) + 2 for n ≥ 2 -/
axiom recurrence_relation (n : ℕ) (h : n ≥ 2) : f n = 2 * f (n-1) + 2

/-- The sum of numbers in the 200th row of the triangular array is 2^200 - 2 -/
theorem sum_200th_row : f 200 = 2^200 - 2 := by sorry

end sum_200th_row_l3251_325194


namespace pascal_triangle_interior_sum_l3251_325177

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The problem statement -/
theorem pascal_triangle_interior_sum :
  interior_sum 6 = 30 →
  interior_sum 8 = 126 := by
sorry

end pascal_triangle_interior_sum_l3251_325177


namespace subset_implies_m_equals_one_l3251_325112

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_equals_one (m : ℝ) :
  B m ⊆ A m → m = 1 :=
by sorry

end subset_implies_m_equals_one_l3251_325112


namespace sqrt_144000_l3251_325162

theorem sqrt_144000 : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end sqrt_144000_l3251_325162


namespace nth_equation_l3251_325130

/-- The product of consecutive integers from n+1 to 2n -/
def leftSide (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i => n + i + 1)

/-- The product of odd numbers from 1 to 2n-1 -/
def oddProduct (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i => 2 * i + 1)

/-- The nth equation in the pattern -/
theorem nth_equation (n : ℕ) : leftSide n = 2^n * oddProduct n := by
  sorry

end nth_equation_l3251_325130


namespace units_digit_of_M_M7_l3251_325105

def M : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | (n + 2) => 2 * M (n + 1) + M n

theorem units_digit_of_M_M7 : M (M 7) % 10 = 3 := by
  sorry

end units_digit_of_M_M7_l3251_325105


namespace oblique_square_area_l3251_325124

/-- The area of an oblique two-dimensional drawing of a unit square -/
theorem oblique_square_area :
  ∀ (S_oblique : ℝ),
  (1 : ℝ) ^ 2 = 1 →  -- Side length of original square is 1
  S_oblique / 1 = Real.sqrt 2 / 4 →  -- Ratio of areas
  S_oblique = Real.sqrt 2 / 4 := by
sorry

end oblique_square_area_l3251_325124


namespace smallest_integer_with_20_divisors_l3251_325187

theorem smallest_integer_with_20_divisors : 
  ∃ n : ℕ+, (n = 240) ∧ 
  (∀ m : ℕ+, m < n → (Finset.card (Nat.divisors m) ≠ 20)) ∧ 
  (Finset.card (Nat.divisors n) = 20) := by
  sorry

end smallest_integer_with_20_divisors_l3251_325187


namespace y1_value_l3251_325145

theorem y1_value (y1 y2 y3 : ℝ) 
  (h_order : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h_eq : (1 - y1)^2 + 2*(y1 - y2)^2 + 3*(y2 - y3)^2 + 4*y3^2 = 1/2) :
  y1 = (3 * Real.sqrt 6 - 6) / 6 := by
  sorry

end y1_value_l3251_325145


namespace remainder_problem_l3251_325144

theorem remainder_problem (N : ℕ) : N % 751 = 53 → N % 29 = 24 := by
  sorry

end remainder_problem_l3251_325144


namespace trapezoid_area_l3251_325172

theorem trapezoid_area (top_base bottom_base height : ℝ) 
  (h1 : top_base = 4)
  (h2 : bottom_base = 8)
  (h3 : height = 3) :
  (top_base + bottom_base) * height / 2 = 18 := by
  sorry

end trapezoid_area_l3251_325172


namespace roberts_chocolates_l3251_325115

theorem roberts_chocolates (nickel_chocolates : ℕ) (robert_extra : ℕ) : 
  nickel_chocolates = 4 → robert_extra = 9 → nickel_chocolates + robert_extra = 13 :=
by
  sorry

end roberts_chocolates_l3251_325115


namespace remaining_fruit_cost_is_eight_l3251_325120

/-- Represents the cost of fruit remaining in Tanya's bag after half fell out --/
def remaining_fruit_cost (pear_count : ℕ) (pear_price : ℚ) 
                         (apple_count : ℕ) (apple_price : ℚ)
                         (pineapple_count : ℕ) (pineapple_price : ℚ) : ℚ :=
  ((pear_count : ℚ) * pear_price + 
   (apple_count : ℚ) * apple_price + 
   (pineapple_count : ℚ) * pineapple_price) / 2

/-- Theorem stating the cost of remaining fruit excluding plums --/
theorem remaining_fruit_cost_is_eight :
  remaining_fruit_cost 6 1.5 4 0.75 2 2 = 8 := by
  sorry

end remaining_fruit_cost_is_eight_l3251_325120


namespace probability_distribution_problem_l3251_325165

theorem probability_distribution_problem (m n : ℝ) 
  (sum_prob : 0.1 + m + n + 0.1 = 1)
  (condition : m + 2 * n = 1.2) : n = 0.4 := by
  sorry

end probability_distribution_problem_l3251_325165


namespace inequality_proof_l3251_325106

/-- If for all real x, 1 - a cos x - b sin x - A cos 2x - B sin 2x ≥ 0, 
    then a² + b² ≤ 2 and A² + B² ≤ 1 -/
theorem inequality_proof (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end inequality_proof_l3251_325106


namespace concert_attendance_l3251_325131

theorem concert_attendance (num_buses : ℕ) (students_per_bus : ℕ) 
  (h1 : num_buses = 8) (h2 : students_per_bus = 45) : 
  num_buses * students_per_bus = 360 := by
  sorry

end concert_attendance_l3251_325131


namespace sphere_to_cube_volume_ratio_l3251_325139

/-- The ratio of the volume of a sphere with diameter 12 inches to the volume of a cube with edge length 6 inches is 4π/3. -/
theorem sphere_to_cube_volume_ratio : 
  let sphere_diameter : ℝ := 12
  let cube_edge : ℝ := 6
  let sphere_volume := (4 / 3) * Real.pi * (sphere_diameter / 2) ^ 3
  let cube_volume := cube_edge ^ 3
  sphere_volume / cube_volume = (4 * Real.pi) / 3 := by
sorry

end sphere_to_cube_volume_ratio_l3251_325139


namespace abs_negative_six_l3251_325114

theorem abs_negative_six : |(-6 : ℤ)| = 6 := by sorry

end abs_negative_six_l3251_325114


namespace circumcircle_area_l3251_325147

/-- An isosceles triangle with two sides of length 6 and a base of length 4 -/
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  is_isosceles : side = 6 ∧ base = 4

/-- A circle passing through the vertices of an isosceles triangle -/
def CircumCircle (t : IsoscelesTriangle) : ℝ → Prop :=
  fun area => area = 16 * Real.pi

/-- The theorem stating that the area of the circumcircle of the given isosceles triangle is 16π -/
theorem circumcircle_area (t : IsoscelesTriangle) : 
  ∃ area, CircumCircle t area :=
sorry

end circumcircle_area_l3251_325147


namespace treasure_hunt_probability_l3251_325185

def num_islands : ℕ := 6
def num_treasure_islands : ℕ := 3

def prob_treasure : ℚ := 1/4
def prob_traps : ℚ := 1/12
def prob_neither : ℚ := 2/3

theorem treasure_hunt_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  5/54 := by sorry

end treasure_hunt_probability_l3251_325185


namespace sufficient_not_necessary_l3251_325116

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end sufficient_not_necessary_l3251_325116


namespace factorization_proof_l3251_325134

theorem factorization_proof (m x y a : ℝ) : 
  (-3 * m^3 + 12 * m = -3 * m * (m + 2) * (m - 2)) ∧ 
  (2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2) ∧ 
  (a^4 + 3 * a^2 - 4 = (a^2 + 4) * (a + 1) * (a - 1)) := by
  sorry

end factorization_proof_l3251_325134


namespace medium_supermarkets_in_sample_l3251_325176

/-- Represents the number of supermarkets in each category -/
structure SupermarketCounts where
  large : ℕ
  medium : ℕ
  small : ℕ

/-- Calculates the total number of supermarkets -/
def total_supermarkets (counts : SupermarketCounts) : ℕ :=
  counts.large + counts.medium + counts.small

/-- Calculates the number of supermarkets of a given category in a stratified sample -/
def stratified_sample_count (counts : SupermarketCounts) (sample_size : ℕ) (category : ℕ) : ℕ :=
  (category * sample_size) / (total_supermarkets counts)

/-- Theorem stating the number of medium-sized supermarkets in the stratified sample -/
theorem medium_supermarkets_in_sample 
  (counts : SupermarketCounts) 
  (sample_size : ℕ) : 
  counts.large = 200 → 
  counts.medium = 400 → 
  counts.small = 1400 → 
  sample_size = 100 → 
  stratified_sample_count counts sample_size counts.medium = 20 := by
  sorry

end medium_supermarkets_in_sample_l3251_325176


namespace fill_three_positions_from_fifteen_l3251_325107

/-- The number of ways to fill positions from a pool of candidates -/
def fill_positions (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if n < k then 0
  else n * fill_positions (n - 1) (k - 1)

/-- Theorem: There are 2730 ways to fill 3 positions from 15 candidates -/
theorem fill_three_positions_from_fifteen :
  fill_positions 15 3 = 2730 := by
  sorry

end fill_three_positions_from_fifteen_l3251_325107


namespace yacht_distance_squared_bounds_l3251_325137

theorem yacht_distance_squared_bounds (θ : Real) 
  (h1 : 30 * Real.pi / 180 ≤ θ) 
  (h2 : θ ≤ 75 * Real.pi / 180) : 
  ∃ (AC : Real), 200 ≤ AC^2 ∧ AC^2 ≤ 656 := by
  sorry

end yacht_distance_squared_bounds_l3251_325137


namespace min_value_theorem_l3251_325152

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / x + 1 / y = 1) :
  3 * x + 4 * y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), 3 * x₀ + 4 * y₀ = 25 ∧ 3 / x₀ + 1 / y₀ = 1 :=
by sorry

end min_value_theorem_l3251_325152


namespace distance_AB_l3251_325133

noncomputable def C₁ (θ : Real) : Real := 2 * Real.sqrt 3 * Real.cos θ + 2 * Real.sin θ

noncomputable def C₂ (θ : Real) : Real := 2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

theorem distance_AB : 
  let θ := Real.pi / 3
  let ρ₁ := C₁ θ
  let ρ₂ := C₂ θ
  abs (ρ₁ - ρ₂) = 4 - 2 * Real.sqrt 3 := by sorry

end distance_AB_l3251_325133


namespace carpet_area_calculation_l3251_325196

/-- Calculates the required carpet area in square yards for a rectangular bedroom and square closet, including wastage. -/
theorem carpet_area_calculation 
  (bedroom_length : ℝ) 
  (bedroom_width : ℝ) 
  (closet_side : ℝ) 
  (wastage_rate : ℝ) 
  (feet_per_yard : ℝ) 
  (h1 : bedroom_length = 15)
  (h2 : bedroom_width = 10)
  (h3 : closet_side = 6)
  (h4 : wastage_rate = 0.1)
  (h5 : feet_per_yard = 3) :
  let bedroom_area := (bedroom_length / feet_per_yard) * (bedroom_width / feet_per_yard)
  let closet_area := (closet_side / feet_per_yard) ^ 2
  let total_area := bedroom_area + closet_area
  let required_area := total_area * (1 + wastage_rate)
  required_area = 22.715 := by
  sorry


end carpet_area_calculation_l3251_325196


namespace paintings_distribution_l3251_325188

theorem paintings_distribution (total_paintings : ℕ) (paintings_per_room : ℕ) (num_rooms : ℕ) :
  total_paintings = 32 →
  paintings_per_room = 8 →
  total_paintings = paintings_per_room * num_rooms →
  num_rooms = 4 := by
sorry

end paintings_distribution_l3251_325188


namespace circle_tangent_to_two_lines_through_point_circle_through_two_points_tangent_to_line_circle_tangent_to_two_lines_and_circle_l3251_325161

-- Define the basic types
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the tangency and passing through relations
def tangent_to_line (c : Circle) (l : Line) : Prop := sorry

def passes_through (c : Circle) (p : Point) : Prop := sorry

def tangent_to_circle (c1 : Circle) (c2 : Circle) : Prop := sorry

-- Part a
theorem circle_tangent_to_two_lines_through_point 
  (l1 l2 : Line) (A : Point) : 
  ∃ (S : Circle), tangent_to_line S l1 ∧ tangent_to_line S l2 ∧ passes_through S A := by
  sorry

-- Part b
theorem circle_through_two_points_tangent_to_line 
  (A B : Point) (l : Line) :
  ∃ (S : Circle), passes_through S A ∧ passes_through S B ∧ tangent_to_line S l := by
  sorry

-- Part c
theorem circle_tangent_to_two_lines_and_circle 
  (l1 l2 : Line) (S_bar : Circle) :
  ∃ (S : Circle), tangent_to_line S l1 ∧ tangent_to_line S l2 ∧ tangent_to_circle S S_bar := by
  sorry

end circle_tangent_to_two_lines_through_point_circle_through_two_points_tangent_to_line_circle_tangent_to_two_lines_and_circle_l3251_325161


namespace common_point_of_alternating_ap_lines_l3251_325150

/-- Represents a line in 2D space with equation ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line --/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Defines an alternating arithmetic progression for a, b, c --/
def is_alternating_ap (a b c : ℝ) : Prop :=
  ∃ d : ℝ, b = a - d ∧ c = a + d

theorem common_point_of_alternating_ap_lines :
  ∀ l : Line, is_alternating_ap l.a l.b l.c → l.contains 1 (-1) :=
sorry

end common_point_of_alternating_ap_lines_l3251_325150


namespace marks_tanks_l3251_325138

/-- The number of tanks Mark has for pregnant fish -/
def num_tanks : ℕ := sorry

/-- The number of pregnant fish in each tank -/
def fish_per_tank : ℕ := 4

/-- The number of young fish each pregnant fish gives birth to -/
def young_per_fish : ℕ := 20

/-- The total number of young fish Mark has -/
def total_young : ℕ := 240

/-- Theorem stating that the number of tanks Mark has is 3 -/
theorem marks_tanks : num_tanks = 3 := by sorry

end marks_tanks_l3251_325138


namespace even_function_implies_a_equals_two_l3251_325166

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = (x+a)(x-2) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 2)

/-- If f(x) = (x+a)(x-2) is an even function, then a = 2 -/
theorem even_function_implies_a_equals_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by
  sorry

end even_function_implies_a_equals_two_l3251_325166


namespace clips_for_huahuas_handkerchiefs_l3251_325143

/-- The number of clips needed to hang handkerchiefs on clotheslines -/
def clips_needed (handkerchiefs : ℕ) (clotheslines : ℕ) : ℕ :=
  -- We define this function without implementation, as the problem doesn't provide the exact formula
  sorry

/-- Theorem stating the number of clips needed for the given scenario -/
theorem clips_for_huahuas_handkerchiefs :
  clips_needed 40 3 = 43 := by
  sorry

end clips_for_huahuas_handkerchiefs_l3251_325143


namespace binomial_coefficient_six_choose_two_l3251_325101

theorem binomial_coefficient_six_choose_two : 
  Nat.choose 6 2 = 15 := by
  sorry

end binomial_coefficient_six_choose_two_l3251_325101


namespace triangle_division_perimeter_l3251_325135

/-- A structure representing a triangle division scenario -/
structure TriangleDivision where
  large_perimeter : ℝ
  num_small_triangles : ℕ
  small_perimeter : ℝ

/-- The theorem statement -/
theorem triangle_division_perimeter 
  (td : TriangleDivision) 
  (h1 : td.large_perimeter = 120)
  (h2 : td.num_small_triangles = 9)
  (h3 : td.small_perimeter * 3 = td.large_perimeter) :
  td.small_perimeter = 40 := by
  sorry

end triangle_division_perimeter_l3251_325135


namespace peanuts_in_box_l3251_325102

/-- The number of peanuts in a box after adding more -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: If a box initially contains 10 peanuts and 8 more peanuts are added,
    the total number of peanuts in the box is 18. -/
theorem peanuts_in_box : total_peanuts 10 8 = 18 := by
  sorry

end peanuts_in_box_l3251_325102


namespace unique_solution_for_exponential_equation_l3251_325173

theorem unique_solution_for_exponential_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (3^x - 2^y = 7 ↔ x = 2 ∧ y = 1) :=
by sorry

end unique_solution_for_exponential_equation_l3251_325173


namespace milk_consumption_l3251_325157

/-- The amount of regular milk consumed in a week -/
def regular_milk : ℝ := 0.5

/-- The amount of soy milk consumed in a week -/
def soy_milk : ℝ := 0.1

/-- The total amount of milk consumed in a week -/
def total_milk : ℝ := regular_milk + soy_milk

theorem milk_consumption : total_milk = 0.6 := by
  sorry

end milk_consumption_l3251_325157


namespace bucket_weight_calculation_l3251_325117

/-- Given an initial weight of shells and an additional weight of shells,
    calculate the total weight of shells in the bucket. -/
def total_weight (initial_weight additional_weight : ℕ) : ℕ :=
  initial_weight + additional_weight

/-- Theorem stating that given 5 pounds of initial weight and 12 pounds of additional weight,
    the total weight of shells in the bucket is 17 pounds. -/
theorem bucket_weight_calculation :
  total_weight 5 12 = 17 := by
  sorry

end bucket_weight_calculation_l3251_325117


namespace distance_difference_l3251_325193

theorem distance_difference (john_distance nina_distance : ℝ) 
  (h1 : john_distance = 0.7)
  (h2 : nina_distance = 0.4) :
  john_distance - nina_distance = 0.3 := by
  sorry

end distance_difference_l3251_325193


namespace possible_average_82_l3251_325158

def test_scores : List Nat := [71, 77, 80, 87]

theorem possible_average_82 (last_score : Nat) 
  (h1 : last_score ≥ 0)
  (h2 : last_score ≤ 100) :
  ∃ (avg : Rat), 
    avg = (test_scores.sum + last_score) / 5 ∧ 
    avg = 82 := by
  sorry

end possible_average_82_l3251_325158


namespace root_equality_condition_l3251_325111

theorem root_equality_condition (m n p : ℕ) 
  (hm : Even m) (hn : Even n) (hp : Even p) 
  (hm_pos : m > 0) (hn_pos : n > 0) (hp_pos : p > 0) :
  (m - p : ℝ) ^ (1 / n) = (n - p : ℝ) ^ (1 / m) ↔ m = n ∧ m ≥ p :=
sorry

end root_equality_condition_l3251_325111


namespace correct_side_for_significant_figures_l3251_325110

/-- Represents the side from which we start counting significant figures -/
inductive Side
  | Left
  | Right
  | Front
  | Back

/-- Definition of significant figures for an approximate number -/
def significantFigures (number : ℕ) (startSide : Side) : ℕ :=
  sorry

/-- Theorem stating that the correct side to start from for significant figures is the left side -/
theorem correct_side_for_significant_figures :
  ∀ (number : ℕ), significantFigures number Side.Left = significantFigures number Side.Left :=
  sorry

end correct_side_for_significant_figures_l3251_325110


namespace two_bedroom_units_count_l3251_325179

theorem two_bedroom_units_count 
  (total_units : ℕ) 
  (one_bedroom_cost two_bedroom_cost : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_units = 12)
  (h2 : one_bedroom_cost = 360)
  (h3 : two_bedroom_cost = 450)
  (h4 : total_cost = 4950) :
  ∃ (one_bedroom_count two_bedroom_count : ℕ),
    one_bedroom_count + two_bedroom_count = total_units ∧
    one_bedroom_count * one_bedroom_cost + two_bedroom_count * two_bedroom_cost = total_cost ∧
    two_bedroom_count = 7 := by
  sorry

end two_bedroom_units_count_l3251_325179


namespace customers_who_left_l3251_325186

/-- Proves that 12 customers left a waiter's section given the initial and final conditions -/
theorem customers_who_left (initial_customers : ℕ) (people_per_table : ℕ) (remaining_tables : ℕ) : 
  initial_customers = 44 → people_per_table = 8 → remaining_tables = 4 →
  initial_customers - (people_per_table * remaining_tables) = 12 :=
by sorry

end customers_who_left_l3251_325186


namespace second_class_average_l3251_325171

theorem second_class_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 12 → 
  n₂ = 28 → 
  avg₁ = 40 → 
  avg_total = 54 → 
  ∃ avg₂ : ℚ, 
    (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = avg_total ∧ 
    avg₂ = 60 :=
by sorry

end second_class_average_l3251_325171


namespace quadratic_equation_solution_exists_l3251_325109

/-- A positive single-digit integer is a natural number between 1 and 9, inclusive. -/
def PositiveSingleDigit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- The quadratic equation x^2 - (2A)x + AB = 0 has positive integer solutions. -/
def HasPositiveIntegerSolutions (A B : ℕ) : Prop :=
  ∃ x : ℕ, x > 0 ∧ x^2 - (2 * A) * x + A * B = 0

theorem quadratic_equation_solution_exists :
  ∃ A B : ℕ, PositiveSingleDigit A ∧ PositiveSingleDigit B ∧ HasPositiveIntegerSolutions A B := by
  sorry

end quadratic_equation_solution_exists_l3251_325109


namespace star_property_l3251_325184

/-- Custom binary operation ※ -/
def star (a b : ℝ) (x y : ℝ) : ℝ := a * x - b * y

theorem star_property (a b : ℝ) (h : star a b 1 2 = 8) :
  star a b (-2) (-4) = -16 := by sorry

end star_property_l3251_325184


namespace certain_number_problem_l3251_325175

theorem certain_number_problem (n m : ℕ+) : 
  Nat.lcm n m = 48 →
  Nat.gcd n m = 8 →
  n = 24 →
  m = 16 := by
sorry

end certain_number_problem_l3251_325175


namespace largest_prime_divisor_of_factorial_sum_l3251_325108

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), is_prime p ∧ 
    (p ∣ (factorial 13 + factorial 14)) ∧
    (∀ q : ℕ, is_prime q → q ∣ (factorial 13 + factorial 14) → q ≤ p) ∧
    p = 5 := by
  sorry

end largest_prime_divisor_of_factorial_sum_l3251_325108


namespace total_highlighters_l3251_325142

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 9)
  (h2 : yellow = 8)
  (h3 : blue = 5) :
  pink + yellow + blue = 22 := by
  sorry

end total_highlighters_l3251_325142


namespace problem_statement_l3251_325146

theorem problem_statement (x y : ℝ) (h : |2*x - y| + Real.sqrt (x + 3*y - 7) = 0) :
  (Real.sqrt ((x - y)^2)) / (y - x) = 1 := by
sorry

end problem_statement_l3251_325146


namespace james_sticker_cost_l3251_325123

theorem james_sticker_cost (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ) : 
  packs = 4 → 
  stickers_per_pack = 30 → 
  cost_per_sticker = 1/10 → 
  (packs * stickers_per_pack * cost_per_sticker) / 2 = 6 := by
  sorry

end james_sticker_cost_l3251_325123


namespace stationery_problem_l3251_325174

theorem stationery_problem (georgia lorene : ℕ) 
  (h1 : lorene = 3 * georgia) 
  (h2 : georgia = lorene - 50) : 
  georgia = 25 := by
sorry

end stationery_problem_l3251_325174


namespace hidden_dots_count_l3251_325149

/-- The sum of numbers on a single die -/
def dieDots : ℕ := 21

/-- The number of dice stacked -/
def numDice : ℕ := 4

/-- The visible numbers on the stacked dice -/
def visibleNumbers : List ℕ := [1, 2, 2, 3, 3, 5, 6]

/-- The number of visible faces -/
def visibleFaces : ℕ := 7

/-- The number of hidden faces -/
def hiddenFaces : ℕ := 17

theorem hidden_dots_count : 
  numDice * dieDots - visibleNumbers.sum = 62 :=
sorry

end hidden_dots_count_l3251_325149


namespace min_dot_product_on_ellipse_l3251_325151

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define the center and focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- State the theorem
theorem min_dot_product_on_ellipse :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
    ∃ min_value : ℝ, min_value = 6 ∧
      ∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 →
        dot_product (Q.1 - O.1, Q.2 - O.2) (Q.1 - F.1, Q.2 - F.2) ≥ min_value :=
by sorry

end min_dot_product_on_ellipse_l3251_325151


namespace ladder_slip_l3251_325178

theorem ladder_slip (initial_length initial_distance slip_down slide_out : ℝ) 
  (h_length : initial_length = 30)
  (h_distance : initial_distance = 9)
  (h_slip : slip_down = 5)
  (h_slide : slide_out = 3) :
  let final_distance := initial_distance + slide_out
  final_distance = 12 := by sorry

end ladder_slip_l3251_325178


namespace complex_magnitude_problem_l3251_325168

theorem complex_magnitude_problem (z : ℂ) : z = (2 + I) / (1 - I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_magnitude_problem_l3251_325168


namespace quadratic_discriminant_perfect_square_l3251_325160

theorem quadratic_discriminant_perfect_square 
  (a b c t : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * t^2 + b * t + c = 0) : 
  b^2 - 4*a*c = (2*a*t + b)^2 := by
  sorry

end quadratic_discriminant_perfect_square_l3251_325160


namespace gcd_plus_binary_sum_l3251_325136

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem gcd_plus_binary_sum : 
  let a := Nat.gcd 98 63
  let b := binary_to_decimal [true, true, false, false, true, true]
  a + b = 58 := by sorry

end gcd_plus_binary_sum_l3251_325136


namespace happy_children_count_is_30_l3251_325121

/-- Represents the number of children in different categories -/
structure ChildrenCount where
  total : Nat
  sad : Nat
  neither : Nat
  boys : Nat
  girls : Nat
  happyBoys : Nat
  sadGirls : Nat
  neitherBoys : Nat

/-- Calculates the number of happy children given the conditions -/
def happyChildrenCount (c : ChildrenCount) : Nat :=
  c.total - c.sad - c.neither

/-- Theorem stating that the number of happy children is 30 -/
theorem happy_children_count_is_30 (c : ChildrenCount) 
  (h1 : c.total = 60)
  (h2 : c.sad = 10)
  (h3 : c.neither = 20)
  (h4 : c.boys = 22)
  (h5 : c.girls = 38)
  (h6 : c.happyBoys = 6)
  (h7 : c.sadGirls = 4)
  (h8 : c.neitherBoys = 10) :
  happyChildrenCount c = 30 := by
  sorry

#check happy_children_count_is_30

end happy_children_count_is_30_l3251_325121


namespace power_of_64_three_fourths_l3251_325126

theorem power_of_64_three_fourths : (64 : ℝ) ^ (3/4) = 16 * Real.sqrt 2 := by
  sorry

end power_of_64_three_fourths_l3251_325126


namespace base_conversion_theorem_l3251_325153

theorem base_conversion_theorem (n : ℕ+) (A B : ℕ) : 
  (A < 8 ∧ B < 5) →
  (8 * A + B = n) →
  (5 * B + A = n) →
  (n : ℕ) = 33 := by
  sorry

end base_conversion_theorem_l3251_325153


namespace two_digit_product_problem_l3251_325104

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def swap_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_product_problem :
  ∃ (x y z : ℕ),
    is_two_digit x ∧
    is_two_digit y ∧
    y = swap_digits x ∧
    x ≠ y ∧
    z = x * y ∧
    100 ≤ z ∧ z < 1000 ∧
    hundreds_digit z = units_digit z ∧
    ((x = 12 ∧ y = 21) ∨ (x = 21 ∧ y = 12)) ∧
    z = 252 :=
by
  sorry

end two_digit_product_problem_l3251_325104


namespace negation_relationship_l3251_325140

theorem negation_relationship (a : ℝ) : 
  ¬(∀ a, a > 0 → a^2 > a) ∧ ¬(∀ a, a^2 ≤ a → a ≤ 0) := by
  sorry

end negation_relationship_l3251_325140


namespace recipe_total_cups_l3251_325199

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of butter -/
def totalIngredients (ratio : RecipeRatio) (butterCups : ℕ) : ℕ :=
  butterCups * (ratio.butter + ratio.flour + ratio.sugar) / ratio.butter

theorem recipe_total_cups (ratio : RecipeRatio) (butterCups : ℕ) 
    (h1 : ratio.butter = 1) 
    (h2 : ratio.flour = 5) 
    (h3 : ratio.sugar = 3) 
    (h4 : butterCups = 9) : 
  totalIngredients ratio butterCups = 81 := by
  sorry

end recipe_total_cups_l3251_325199


namespace max_russian_score_l3251_325127

/-- Represents a chess player -/
structure Player where
  country : String
  score : ℚ

/-- Represents a chess tournament -/
structure Tournament where
  players : Finset Player
  russianPlayers : Finset Player
  winner : Player
  runnerUp : Player

/-- The scoring system for the tournament -/
def scoringSystem : ℚ × ℚ × ℚ := (1, 1/2, 0)

/-- Theorem statement for the maximum score of Russian players -/
theorem max_russian_score (t : Tournament) : 
  t.players.card = 20 ∧ 
  t.russianPlayers.card = 6 ∧
  t.winner.country = "Russia" ∧
  t.runnerUp.country = "Armenia" ∧
  t.winner.score > t.runnerUp.score ∧
  (∀ p ∈ t.players, p ≠ t.winner → p ≠ t.runnerUp → t.runnerUp.score > p.score) →
  (t.russianPlayers.sum (λ p => p.score)) ≤ 96 := by
  sorry

end max_russian_score_l3251_325127


namespace balloons_bought_at_park_l3251_325167

theorem balloons_bought_at_park (allan_balloons jake_initial_balloons : ℕ) 
  (h1 : allan_balloons = 6)
  (h2 : jake_initial_balloons = 3)
  (h3 : ∃ (x : ℕ), jake_initial_balloons + x = allan_balloons + 1) :
  ∃ (x : ℕ), x = 4 ∧ jake_initial_balloons + x = allan_balloons + 1 := by
sorry

end balloons_bought_at_park_l3251_325167


namespace perpendicular_bisector_b_value_l3251_325180

/-- A line that is a perpendicular bisector of a line segment -/
structure PerpendicularBisector where
  a : ℝ
  b : ℝ
  c : ℝ
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  is_perpendicular_bisector : True  -- This is a placeholder for the actual condition

/-- The theorem stating that b = 12 for the given perpendicular bisector -/
theorem perpendicular_bisector_b_value :
  ∀ (pb : PerpendicularBisector), 
  pb.a = 1 ∧ pb.b = 1 ∧ pb.p1 = (2, 4) ∧ pb.p2 = (8, 10) → 
  pb.c = 12 := by
  sorry

end perpendicular_bisector_b_value_l3251_325180


namespace keyboard_printer_cost_l3251_325191

/-- The total cost of keyboards and printers -/
def total_cost (num_keyboards : ℕ) (num_printers : ℕ) (keyboard_price : ℕ) (printer_price : ℕ) : ℕ :=
  num_keyboards * keyboard_price + num_printers * printer_price

/-- Theorem stating that the total cost of 15 keyboards at $20 each and 25 printers at $70 each is $2050 -/
theorem keyboard_printer_cost : total_cost 15 25 20 70 = 2050 := by
  sorry

end keyboard_printer_cost_l3251_325191


namespace f_properties_l3251_325119

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 3

-- State the theorem
theorem f_properties (a : ℝ) (h_a_pos : a > 0) :
  (∀ x ≥ 3, f a x ≥ 0) → a ≥ 1 ∧
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  ∃ s : ℝ, 2 < s ∧ s < 4 ∧ ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁^2 + x₂^2 = s :=
by sorry

end f_properties_l3251_325119


namespace unsold_bars_l3251_325183

theorem unsold_bars (total_bars : ℕ) (price_per_bar : ℕ) (total_amount : ℕ) : 
  total_bars = 8 → price_per_bar = 4 → total_amount = 20 → 
  total_bars - (total_amount / price_per_bar) = 3 :=
by sorry

end unsold_bars_l3251_325183


namespace billy_soda_distribution_l3251_325100

/-- Represents the number of sodas Billy can give to each sibling -/
def sodas_per_sibling (total_sodas : ℕ) (num_sisters : ℕ) : ℕ :=
  total_sodas / (num_sisters + 2 * num_sisters)

/-- Theorem stating that Billy can give 2 sodas to each sibling -/
theorem billy_soda_distribution :
  sodas_per_sibling 12 2 = 2 := by
  sorry

end billy_soda_distribution_l3251_325100


namespace nineteenth_term_is_zero_l3251_325164

/-- A sequence with specific properties -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  a 3 = 2 ∧ 
  a 7 = 1 ∧ 
  ∃ d : ℝ, ∀ n : ℕ, 1 / (a (n + 1) + 1) - 1 / (a n + 1) = d

/-- Theorem stating that for a special sequence, the 19th term is 0 -/
theorem nineteenth_term_is_zero (a : ℕ → ℝ) (h : special_sequence a) : 
  a 19 = 0 := by sorry

end nineteenth_term_is_zero_l3251_325164


namespace a_equals_fibonacci_ratio_l3251_325154

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def a : ℕ → ℕ
  | 0 => 3
  | (n + 1) => (a n)^2 - 2

theorem a_equals_fibonacci_ratio (n : ℕ) :
  a n = fibonacci (2^(n+1)) / fibonacci 4 := by
  sorry

end a_equals_fibonacci_ratio_l3251_325154


namespace thirty_five_million_scientific_notation_l3251_325129

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive integer to scientific notation -/
def to_scientific_notation (n : ℕ+) : ScientificNotation :=
  sorry

theorem thirty_five_million_scientific_notation :
  to_scientific_notation 35000000 = ScientificNotation.mk 3.5 7 (by norm_num) :=
sorry

end thirty_five_million_scientific_notation_l3251_325129


namespace square_area_with_inscribed_triangle_l3251_325198

theorem square_area_with_inscribed_triangle (d : ℝ) (h : d = 16) : 
  let s := d / Real.sqrt 2
  let square_area := s^2
  square_area = 128 := by
  sorry

end square_area_with_inscribed_triangle_l3251_325198


namespace roots_form_parallelogram_l3251_325156

/-- The polynomial whose roots we're investigating -/
def P (b : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 17*b*z^2 - 2*(3*b^2 + 4*b - 4)*z + 9

/-- A function that checks if four complex numbers form a parallelogram -/
def isParallelogram (z₁ z₂ z₃ z₄ : ℂ) : Prop := 
  (z₁ + z₃ = z₂ + z₄) ∧ (z₁ - z₂ = z₄ - z₃)

/-- The main theorem stating the values of b for which the roots form a parallelogram -/
theorem roots_form_parallelogram : 
  ∀ b : ℝ, (∃ z₁ z₂ z₃ z₄ : ℂ, 
    (P b z₁ = 0) ∧ (P b z₂ = 0) ∧ (P b z₃ = 0) ∧ (P b z₄ = 0) ∧ 
    isParallelogram z₁ z₂ z₃ z₄) ↔ 
  (b = 7/3 ∨ b = 2) :=
sorry

end roots_form_parallelogram_l3251_325156


namespace cos_sin_identity_l3251_325163

theorem cos_sin_identity (α : Real) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = -(2 + Real.sqrt 3) / 3 := by
  sorry

end cos_sin_identity_l3251_325163


namespace no_primes_divisible_by_77_l3251_325122

theorem no_primes_divisible_by_77 : ¬∃ p : ℕ, Nat.Prime p ∧ 77 ∣ p := by
  sorry

end no_primes_divisible_by_77_l3251_325122


namespace sum_reciprocals_of_three_integers_l3251_325113

theorem sum_reciprocals_of_three_integers (a b c : ℕ+) :
  a < b ∧ b < c ∧ a + b + c = 11 →
  (1 : ℚ) / a + 1 / b + 1 / c = 31 / 21 := by
  sorry

end sum_reciprocals_of_three_integers_l3251_325113


namespace ruth_school_days_l3251_325103

/-- Ruth's school schedule -/
def school_schedule (days_per_week : ℝ) : Prop :=
  let hours_per_day : ℝ := 8
  let math_class_fraction : ℝ := 0.25
  let math_hours_per_week : ℝ := 10
  (hours_per_day * days_per_week * math_class_fraction = math_hours_per_week)

theorem ruth_school_days : ∃ (d : ℝ), school_schedule d ∧ d = 5 := by
  sorry

end ruth_school_days_l3251_325103


namespace quadratic_solution_range_l3251_325192

/-- The range of t for which the quadratic equation x^2 - 4x + 1 - t = 0 has solutions in (0, 7/2) -/
theorem quadratic_solution_range (t : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x < 7/2 ∧ x^2 - 4*x + 1 - t = 0) ↔ -3 ≤ t ∧ t < 1 := by
  sorry

end quadratic_solution_range_l3251_325192


namespace scientific_notation_130_billion_l3251_325132

theorem scientific_notation_130_billion : 130000000000 = 1.3 * (10 ^ 11) := by
  sorry

end scientific_notation_130_billion_l3251_325132


namespace circle_area_l3251_325197

theorem circle_area (circumference : ℝ) (area : ℝ) : 
  circumference = 36 → area = 324 / Real.pi := by
  sorry

end circle_area_l3251_325197


namespace cylinder_cone_sphere_volume_l3251_325155

/-- Given a cylinder with volume 150π cm³, prove that the sum of the volumes of a cone 
    with the same base radius and height as the cylinder, and a sphere with the same 
    radius as the cylinder, is equal to 50π + (4/3)π * (∛150)² cm³. -/
theorem cylinder_cone_sphere_volume 
  (r h : ℝ) 
  (h_cylinder_volume : π * r^2 * h = 150 * π) :
  (1/3 * π * r^2 * h) + (4/3 * π * r^3) = 50 * π + 4/3 * π * (150^(2/3)) := by
  sorry

end cylinder_cone_sphere_volume_l3251_325155


namespace yard_area_l3251_325128

/-- The area of a rectangular yard with square cutouts -/
theorem yard_area (length width cutout_side : ℕ) (num_cutouts : ℕ) : 
  length = 20 → 
  width = 18 → 
  cutout_side = 4 → 
  num_cutouts = 2 → 
  length * width - num_cutouts * cutout_side * cutout_side = 328 := by
  sorry

end yard_area_l3251_325128


namespace conditions_satisfied_l3251_325181

-- Define the points and lengths
variable (P Q R S : ℝ) -- Representing points as real numbers for simplicity
variable (a b c k : ℝ)

-- State the conditions
axiom distinct_collinear : P < Q ∧ Q < R ∧ R < S
axiom positive_lengths : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0
axiom length_PQ : Q - P = a
axiom length_PR : R - P = b
axiom length_PS : S - P = c
axiom b_relation : b = a + k

-- Triangle inequality conditions
axiom triangle_inequality1 : a + (b - a) > c - b
axiom triangle_inequality2 : (b - a) + (c - b) > a
axiom triangle_inequality3 : a + (c - b) > b - a

-- Theorem to prove
theorem conditions_satisfied :
  a < c / 2 ∧ b < 2 * a + c / 2 :=
sorry

end conditions_satisfied_l3251_325181
