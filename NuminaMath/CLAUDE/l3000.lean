import Mathlib

namespace art_supplies_cost_l3000_300086

def total_spent : ℕ := 50
def num_skirts : ℕ := 2
def skirt_cost : ℕ := 15

theorem art_supplies_cost : total_spent - (num_skirts * skirt_cost) = 20 := by
  sorry

end art_supplies_cost_l3000_300086


namespace basketball_volleyball_cost_total_cost_proof_l3000_300066

/-- The cost of buying basketballs and volleyballs -/
theorem basketball_volleyball_cost (m n : ℝ) : ℝ :=
  3 * m + 7 * n

/-- Proof that the total cost of 3 basketballs and 7 volleyballs is 3m + 7n yuan -/
theorem total_cost_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  basketball_volleyball_cost m n = 3 * m + 7 * n :=
by sorry

end basketball_volleyball_cost_total_cost_proof_l3000_300066


namespace sticker_sharing_l3000_300029

theorem sticker_sharing (total_stickers : ℕ) (andrew_final : ℕ) : 
  total_stickers = 1500 →
  andrew_final = 900 →
  (2 : ℚ) / 3 = (andrew_final - total_stickers / 5) / (3 * total_stickers / 5) :=
by sorry

end sticker_sharing_l3000_300029


namespace line_tangent_to_parabola_l3000_300060

/-- A line is tangent to a parabola if and only if their intersection has exactly one point. -/
def is_tangent_line_to_parabola (a b c : ℝ) : Prop :=
  ∃! x : ℝ, (3 * x + 1)^2 = 12 * x

/-- The line y = 3x + 1 is tangent to the parabola y^2 = 12x. -/
theorem line_tangent_to_parabola : is_tangent_line_to_parabola 3 1 12 := by
  sorry

end line_tangent_to_parabola_l3000_300060


namespace max_value_squared_sum_max_value_squared_sum_achieved_l3000_300003

theorem max_value_squared_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x^2 + y^2 + z^4 ≤ 1 :=
by sorry

theorem max_value_squared_sum_achieved (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ a^2 + b^2 + c^4 = 1 :=
by sorry

end max_value_squared_sum_max_value_squared_sum_achieved_l3000_300003


namespace square_sum_value_l3000_300069

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 9) : a^2 + b^2 = 27 := by
  sorry

end square_sum_value_l3000_300069


namespace cylinder_volume_relation_l3000_300098

/-- Theorem: Volume of cylinder X in terms of cylinder Y's height -/
theorem cylinder_volume_relation (h : ℝ) (h_pos : h > 0) : ∃ (r_x r_y h_x : ℝ),
  r_y = 2 * h_x ∧ 
  h_x = 3 * r_y ∧ 
  h_x = 3 * h ∧
  r_x = 6 * h ∧
  π * r_x^2 * h_x = 3 * (π * r_y^2 * h) ∧
  π * r_x^2 * h_x = 108 * π * h^3 := by
  sorry

end cylinder_volume_relation_l3000_300098


namespace count_with_six_seven_l3000_300024

/-- The number of integers from 1 to 512 in base 8 that don't use digits 6 or 7 -/
def count_without_six_seven : ℕ := 215

/-- The total number of integers we're considering -/
def total_count : ℕ := 512

theorem count_with_six_seven :
  total_count - count_without_six_seven = 297 := by
  sorry

end count_with_six_seven_l3000_300024


namespace three_sevenths_minus_forty_percent_l3000_300087

theorem three_sevenths_minus_forty_percent (x : ℝ) : 
  (0.3 * x = 63.0000000000001) → 
  ((3/7) * x - 0.4 * x = 6.00000000000006) := by
sorry

end three_sevenths_minus_forty_percent_l3000_300087


namespace calculation_proof_inequality_system_solution_l3000_300049

-- Problem 1
theorem calculation_proof :
  Real.sqrt 4 - 2 * Real.sin (45 * π / 180) + (1/3)⁻¹ + |-(Real.sqrt 2)| = 5 := by sorry

-- Problem 2
theorem inequality_system_solution (x : ℝ) :
  (3*x + 1 < 2*x + 3 ∧ 2*x > (3*x - 1)/2) ↔ (-1 < x ∧ x < 2) := by sorry

end calculation_proof_inequality_system_solution_l3000_300049


namespace area_is_40_l3000_300082

/-- Two perpendicular lines intersecting at point B -/
structure PerpendicularLines where
  B : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  perpendicular : Bool
  intersect_at_B : Bool
  y_intercept_product : ℝ

/-- The area of triangle BRS given two perpendicular lines -/
def triangle_area (lines : PerpendicularLines) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle BRS is 40 -/
theorem area_is_40 (lines : PerpendicularLines) 
  (h1 : lines.B = (8, 6))
  (h2 : lines.perpendicular = true)
  (h3 : lines.intersect_at_B = true)
  (h4 : lines.y_intercept_product = -24)
  : triangle_area lines = 40 := by
  sorry

end area_is_40_l3000_300082


namespace largest_space_diagonal_squared_of_box_l3000_300070

/-- The square of the largest possible length of the space diagonal of a smaller box -/
def largest_space_diagonal_squared (a b c : ℕ) : ℕ :=
  max
    (a * a + (b / 2) * (b / 2) + c * c)
    (max
      (a * a + b * b + (c / 2) * (c / 2))
      ((a / 2) * (a / 2) + b * b + c * c))

/-- Theorem stating the largest possible space diagonal squared for the given box -/
theorem largest_space_diagonal_squared_of_box :
  largest_space_diagonal_squared 1 2 16 = 258 := by
  sorry

end largest_space_diagonal_squared_of_box_l3000_300070


namespace expected_balls_in_original_position_l3000_300059

/-- Represents the number of balls in the circle. -/
def n : ℕ := 6

/-- Represents the number of swaps performed. -/
def k : ℕ := 3

/-- The probability of a specific ball being swapped in one swap. -/
def p : ℚ := 1 / 3

/-- The probability of a ball remaining in its original position after k swaps. -/
def prob_original_position (k : ℕ) (p : ℚ) : ℚ :=
  (1 - p)^k + k * p * (1 - p)^(k-1)

/-- The expected number of balls in their original positions after k swaps. -/
def expected_original_positions (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  n * prob_original_position k p

theorem expected_balls_in_original_position :
  expected_original_positions n k p = 84 / 27 :=
by sorry

end expected_balls_in_original_position_l3000_300059


namespace vector_opposite_direction_l3000_300046

/-- Given two vectors a and b in ℝ², where a = (1, -1), |b| = |a|, and b is in the opposite direction of a, prove that b = (-1, 1). -/
theorem vector_opposite_direction (a b : ℝ × ℝ) : 
  a = (1, -1) → 
  ‖b‖ = ‖a‖ → 
  ∃ (k : ℝ), k < 0 ∧ b = k • a → 
  b = (-1, 1) := by
sorry

end vector_opposite_direction_l3000_300046


namespace harvard_mit_puzzle_l3000_300052

/-- Given that the product of letters in "harvard", "mit", and "hmmt" all equal 100,
    prove that the product of letters in "rad" and "trivia" equals 10000. -/
theorem harvard_mit_puzzle (h a r v d m i t : ℕ) : 
  h * a * r * v * a * r * d = 100 →
  m * i * t = 100 →
  h * m * m * t = 100 →
  (r * a * d) * (t * r * i * v * i * a) = 10000 := by
  sorry

end harvard_mit_puzzle_l3000_300052


namespace product_with_miscopied_digit_l3000_300001

theorem product_with_miscopied_digit (x y : ℕ) 
  (h1 : x * y = 4500)
  (h2 : x * (y - 2) = 4380) :
  x = 60 ∧ y = 75 := by
sorry

end product_with_miscopied_digit_l3000_300001


namespace unique_number_with_gcd_l3000_300017

theorem unique_number_with_gcd : ∃! n : ℕ, 90 < n ∧ n < 100 ∧ Nat.gcd 35 n = 7 := by
  sorry

end unique_number_with_gcd_l3000_300017


namespace quadratic_inequality_solution_set_l3000_300038

theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a + 2 > 0) → a > -1 := by
  sorry

end quadratic_inequality_solution_set_l3000_300038


namespace B_subset_A_l3000_300009

-- Define the set A
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- State the theorem
theorem B_subset_A (B : Set ℝ) (h : A ∩ B = B) : B ⊆ A := by
  sorry

end B_subset_A_l3000_300009


namespace optimal_price_l3000_300077

/-- Represents the selling price and corresponding daily sales volume -/
structure PriceSales where
  price : ℝ
  sales : ℝ

/-- The cost price of the fruit in yuan per kilogram -/
def costPrice : ℝ := 22

/-- The initial selling price and sales volume -/
def initialSale : PriceSales :=
  { price := 38, sales := 160 }

/-- The change in sales volume per yuan price reduction -/
def salesIncrease : ℝ := 40

/-- The required daily profit in yuan -/
def requiredProfit : ℝ := 3640

/-- Calculates the daily profit given a selling price -/
def calculateProfit (sellingPrice : ℝ) : ℝ :=
  let priceReduction := initialSale.price - sellingPrice
  let salesVolume := initialSale.sales + salesIncrease * priceReduction
  (sellingPrice - costPrice) * salesVolume

/-- The theorem to be proved -/
theorem optimal_price :
  ∃ (optimalPrice : ℝ),
    calculateProfit optimalPrice = requiredProfit ∧
    optimalPrice = 29 ∧
    ∀ (price : ℝ),
      calculateProfit price = requiredProfit →
      price ≥ optimalPrice :=
sorry

end optimal_price_l3000_300077


namespace cosine_sine_identity_l3000_300099

theorem cosine_sine_identity : 
  (Real.cos (10 * π / 180)) / (2 * Real.sin (10 * π / 180)) - 2 * Real.cos (10 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cosine_sine_identity_l3000_300099


namespace circumcenter_coincidence_l3000_300068

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  distance : ℝ

/-- The circumcenter of a tetrahedron -/
def circumcenter (t : Tetrahedron) : Point3D :=
  sorry

/-- The inscribed sphere of a tetrahedron -/
def inscribedSphere (t : Tetrahedron) : Sphere :=
  sorry

/-- Points where the inscribed sphere touches the faces of the tetrahedron -/
def touchPoints (t : Tetrahedron) (s : Sphere) : (Point3D × Point3D × Point3D × Point3D) :=
  sorry

/-- Plane equidistant from a point and another plane -/
def equidistantPlane (p : Point3D) (pl : Plane) : Plane :=
  sorry

/-- Tetrahedron formed by four planes -/
def tetrahedronFromPlanes (p1 p2 p3 p4 : Plane) : Tetrahedron :=
  sorry

/-- Main theorem statement -/
theorem circumcenter_coincidence (t : Tetrahedron) : 
  let s := inscribedSphere t
  let (A₁, B₁, C₁, D₁) := touchPoints t s
  let p1 := equidistantPlane t.A (Plane.mk B₁ 0)
  let p2 := equidistantPlane t.B (Plane.mk C₁ 0)
  let p3 := equidistantPlane t.C (Plane.mk D₁ 0)
  let p4 := equidistantPlane t.D (Plane.mk A₁ 0)
  let t' := tetrahedronFromPlanes p1 p2 p3 p4
  circumcenter t = circumcenter t' :=
by
  sorry

end circumcenter_coincidence_l3000_300068


namespace perpendicular_planes_l3000_300041

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (a b : Line) 
  (ξ ζ : Plane) 
  (diff_lines : a ≠ b) 
  (diff_planes : ξ ≠ ζ) 
  (h1 : perp_line_line a b) 
  (h2 : perp_line_plane a ξ) 
  (h3 : perp_line_plane b ζ) : 
  perp_plane_plane ξ ζ :=
sorry

end perpendicular_planes_l3000_300041


namespace simplify_square_roots_l3000_300080

theorem simplify_square_roots : 
  (Real.sqrt 338 / Real.sqrt 288) + (Real.sqrt 150 / Real.sqrt 96) = 7 / 3 := by
  sorry

end simplify_square_roots_l3000_300080


namespace sugar_for_frosting_l3000_300005

theorem sugar_for_frosting (total_sugar cake_sugar frosting_sugar : ℚ) : 
  total_sugar = 0.8 →
  cake_sugar = 0.2 →
  total_sugar = cake_sugar + frosting_sugar →
  frosting_sugar = 0.6 := by
sorry

end sugar_for_frosting_l3000_300005


namespace incorrect_calculation_l3000_300071

theorem incorrect_calculation : 3 * Real.sqrt 3 - Real.sqrt 3 ≠ 2 := by
  sorry

end incorrect_calculation_l3000_300071


namespace quadratic_perfect_square_l3000_300016

theorem quadratic_perfect_square (p : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 + 27*x + p = (a*x + b)^2) → p = 81/4 := by
  sorry

end quadratic_perfect_square_l3000_300016


namespace greenhouse_optimization_l3000_300031

/-- Given a rectangle with area 800 m², prove that the maximum area of the inner rectangle
    formed by subtracting a 1 m border on three sides and a 3 m border on one side
    is 648 m², achieved when the original rectangle has dimensions 40 m × 20 m. -/
theorem greenhouse_optimization (a b : ℝ) :
  a > 0 ∧ b > 0 ∧ a * b = 800 →
  (a - 2) * (b - 4) ≤ 648 ∧
  (a - 2) * (b - 4) = 648 ↔ a = 40 ∧ b = 20 :=
by sorry

end greenhouse_optimization_l3000_300031


namespace number_pair_theorem_l3000_300012

theorem number_pair_theorem (S P : ℝ) (x y : ℝ) 
  (h1 : x + y = S) (h2 : x * y = P) :
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) :=
by sorry

end number_pair_theorem_l3000_300012


namespace profit_percentage_example_l3000_300015

/-- Calculates the profit percentage given the selling price and cost price. -/
def profit_percentage (selling_price cost_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that for a selling price of 250 and a cost price of 200, 
    the profit percentage is 25%. -/
theorem profit_percentage_example : 
  profit_percentage 250 200 = 25 := by
  sorry

end profit_percentage_example_l3000_300015


namespace double_age_in_two_years_l3000_300035

/-- The number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  (man_age - 2 * son_age) / (2 - 1)

/-- Theorem: Given the son's age is 35 and the age difference is 37, 
    the number of years until the man's age is twice his son's age is 2 -/
theorem double_age_in_two_years (son_age : ℕ) (age_difference : ℕ) 
  (h1 : son_age = 35) (h2 : age_difference = 37) : 
  years_until_double_age son_age age_difference = 2 := by
  sorry

end double_age_in_two_years_l3000_300035


namespace smallest_divisible_by_8_9_11_l3000_300055

theorem smallest_divisible_by_8_9_11 : ∀ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 9 ∣ n ∧ 11 ∣ n → n ≥ 792 :=
by
  sorry

end smallest_divisible_by_8_9_11_l3000_300055


namespace scientific_notation_138000_l3000_300096

theorem scientific_notation_138000 :
  138000 = 1.38 * (10 ^ 5) := by
  sorry

end scientific_notation_138000_l3000_300096


namespace gumball_sales_total_l3000_300062

theorem gumball_sales_total (price1 price2 price3 price4 price5 : ℕ) 
  (h1 : price1 = 12)
  (h2 : price2 = 15)
  (h3 : price3 = 8)
  (h4 : price4 = 10)
  (h5 : price5 = 20) :
  price1 + price2 + price3 + price4 + price5 = 65 := by
  sorry

end gumball_sales_total_l3000_300062


namespace isabel_homework_l3000_300028

/-- Given the total number of problems, completed problems, and problems per page,
    calculate the number of remaining pages. -/
def remaining_pages (total : ℕ) (completed : ℕ) (per_page : ℕ) : ℕ :=
  (total - completed) / per_page

/-- Theorem stating that given Isabel's homework conditions, 
    the number of remaining pages is 5. -/
theorem isabel_homework : 
  remaining_pages 72 32 8 = 5 := by
  sorry

end isabel_homework_l3000_300028


namespace stones_per_bracelet_l3000_300039

theorem stones_per_bracelet (total_stones : Float) (num_bracelets : Float) 
  (h1 : total_stones = 88.0)
  (h2 : num_bracelets = 8.0) :
  total_stones / num_bracelets = 11.0 := by
  sorry

end stones_per_bracelet_l3000_300039


namespace jim_travels_two_miles_l3000_300078

/-- The distance John travels in miles -/
def john_distance : ℝ := 15

/-- The difference between John's and Jill's travel distances in miles -/
def distance_difference : ℝ := 5

/-- The percentage of Jill's distance that Jim travels -/
def jim_percentage : ℝ := 0.20

/-- Jill's travel distance in miles -/
def jill_distance : ℝ := john_distance - distance_difference

/-- Jim's travel distance in miles -/
def jim_distance : ℝ := jill_distance * jim_percentage

theorem jim_travels_two_miles :
  jim_distance = 2 := by sorry

end jim_travels_two_miles_l3000_300078


namespace power_of_two_plus_one_square_solution_power_of_two_equals_square_plus_one_solution_l3000_300084

theorem power_of_two_plus_one_square_solution (n x : ℕ+) :
  2^(n:ℕ) + 1 = (x:ℕ)^2 ↔ n = 3 ∧ x = 3 :=
sorry

theorem power_of_two_equals_square_plus_one_solution (n x : ℕ+) :
  2^(n:ℕ) = (x:ℕ)^2 + 1 ↔ n = 1 ∧ x = 1 :=
sorry

end power_of_two_plus_one_square_solution_power_of_two_equals_square_plus_one_solution_l3000_300084


namespace power_division_l3000_300000

theorem power_division (a : ℝ) (h : a ≠ 0) : a^6 / a^2 = a^4 := by
  sorry

end power_division_l3000_300000


namespace no_infinite_prime_sequence_l3000_300023

theorem no_infinite_prime_sequence : 
  ¬ ∃ (p : ℕ → ℕ), (∀ n, Prime (p n)) ∧ (∀ n, p (n + 1) = 2 * p n + 1) := by
  sorry

end no_infinite_prime_sequence_l3000_300023


namespace complex_fraction_simplification_l3000_300002

theorem complex_fraction_simplification :
  (5 : ℂ) / (Complex.I - 2) = -2 - Complex.I := by sorry

end complex_fraction_simplification_l3000_300002


namespace rectangular_prism_surface_area_volume_l3000_300044

theorem rectangular_prism_surface_area_volume (x : ℝ) (h : x > 0) :
  let a := Real.log x
  let b := Real.exp (Real.log x)
  let c := x
  let surface_area := 2 * (a * b + b * c + c * a)
  let volume := a * b * c
  surface_area = 3 * volume → x = Real.exp 2 := by
sorry

end rectangular_prism_surface_area_volume_l3000_300044


namespace mars_network_connected_min_tunnels_for_connectivity_l3000_300037

/-- A graph representing the Mars settlement network -/
structure MarsNetwork where
  settlements : Nat
  tunnels : Nat

/-- The property that a MarsNetwork is connected -/
def is_connected (network : MarsNetwork) : Prop :=
  network.tunnels ≥ network.settlements - 1

/-- The Mars settlement network with 2004 settlements -/
def mars_network : MarsNetwork :=
  { settlements := 2004, tunnels := 2003 }

/-- Theorem stating that the Mars network with 2003 tunnels is connected -/
theorem mars_network_connected :
  is_connected mars_network :=
sorry

/-- Theorem stating that 2003 is the minimum number of tunnels required for connectivity -/
theorem min_tunnels_for_connectivity (network : MarsNetwork) :
  network.settlements = 2004 →
  is_connected network →
  network.tunnels ≥ 2003 :=
sorry

end mars_network_connected_min_tunnels_for_connectivity_l3000_300037


namespace max_product_constraint_l3000_300081

theorem max_product_constraint (x y : ℝ) : 
  x > 0 → y > 0 → x + 2 * y = 1 → x * y ≤ 1/8 := by sorry

end max_product_constraint_l3000_300081


namespace necessary_but_not_sufficient_l3000_300032

theorem necessary_but_not_sufficient 
  (a b : ℝ) 
  (ha : a > 0) :
  (a > abs b → a + b > 0) ∧ 
  ¬(∀ a b : ℝ, a > 0 → a + b > 0 → a > abs b) :=
sorry

end necessary_but_not_sufficient_l3000_300032


namespace bug_prob_after_8_meters_l3000_300008

/-- Represents the probability of the bug being at vertex A after n meters -/
def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 - Q n) / 3

/-- The vertices of the tetrahedron -/
inductive Vertex
| A | B | C | D

/-- The probability of the bug being at vertex A after 8 meters -/
def prob_at_A_after_8 : ℚ := Q 8

theorem bug_prob_after_8_meters :
  prob_at_A_after_8 = 547 / 2187 :=
sorry

end bug_prob_after_8_meters_l3000_300008


namespace all_points_on_line_l3000_300013

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def isOnLine (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

theorem all_points_on_line :
  let p1 : Point := ⟨4, 8⟩
  let p2 : Point := ⟨-2, -4⟩
  let points : List Point := [⟨1, 2⟩, ⟨0, 0⟩, ⟨2, 4⟩, ⟨5, 10⟩, ⟨-1, -2⟩]
  ∀ p ∈ points, isOnLine p p1 p2 := by
  sorry

end all_points_on_line_l3000_300013


namespace cycle_gain_percent_l3000_300043

/-- Calculates the gain percent given the cost price and selling price -/
def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The gain percent is 8% when a cycle is bought for Rs. 1000 and sold for Rs. 1080 -/
theorem cycle_gain_percent :
  gain_percent 1000 1080 = 8 := by
  sorry

end cycle_gain_percent_l3000_300043


namespace fraction_zero_solution_l3000_300065

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 16) / (4 - x) = 0 ∧ x ≠ 4 → x = -4 :=
by sorry

end fraction_zero_solution_l3000_300065


namespace function_inequality_l3000_300075

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : f a > Real.exp a * f 0 := by
  sorry

end function_inequality_l3000_300075


namespace sqrt_simplification_l3000_300051

theorem sqrt_simplification :
  Real.sqrt 75 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 6 * Real.sqrt 2 = 3 * Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end sqrt_simplification_l3000_300051


namespace store_a_cheaper_for_15_boxes_store_a_cheaper_for_x_boxes_l3000_300007

/-- Represents the cost of purchasing table tennis equipment from a store. -/
structure StoreCost where
  ballCost : ℝ  -- Cost per box of balls
  racketCost : ℝ  -- Cost per racket
  numRackets : ℕ  -- Number of rackets needed
  discount : ℝ  -- Discount factor (1 for no discount, 0.9 for 10% discount)
  freeBoxes : ℕ  -- Number of free boxes of balls

/-- Calculates the total cost for a given number of ball boxes. -/
def totalCost (s : StoreCost) (x : ℕ) : ℝ :=
  s.discount * (s.ballCost * (x - s.freeBoxes) + s.racketCost * s.numRackets)

/-- Store A's cost structure -/
def storeA : StoreCost :=
  { ballCost := 5
  , racketCost := 30
  , numRackets := 5
  , discount := 1
  , freeBoxes := 5 }

/-- Store B's cost structure -/
def storeB : StoreCost :=
  { ballCost := 5
  , racketCost := 30
  , numRackets := 5
  , discount := 0.9
  , freeBoxes := 0 }

/-- Theorem stating that Store A is cheaper than or equal to Store B for 15 boxes of balls -/
theorem store_a_cheaper_for_15_boxes :
  totalCost storeA 15 ≤ totalCost storeB 15 :=
by
  sorry

/-- Theorem stating that Store A is cheaper than or equal to Store B for any number of boxes ≥ 5 -/
theorem store_a_cheaper_for_x_boxes (x : ℕ) (h : x ≥ 5) :
  totalCost storeA x ≤ totalCost storeB x :=
by
  sorry

end store_a_cheaper_for_15_boxes_store_a_cheaper_for_x_boxes_l3000_300007


namespace different_plant_choice_probability_l3000_300072

theorem different_plant_choice_probability :
  let num_plant_types : ℕ := 4
  let num_employees : ℕ := 2
  let total_combinations : ℕ := num_plant_types ^ num_employees
  let same_choice_combinations : ℕ := num_plant_types
  let different_choice_combinations : ℕ := total_combinations - same_choice_combinations
  (different_choice_combinations : ℚ) / total_combinations = 13 / 16 :=
by sorry

end different_plant_choice_probability_l3000_300072


namespace quadratic_equation_solution_l3000_300004

theorem quadratic_equation_solution (p q : ℝ) : 
  p = 15 * q^2 - 5 → p = 40 → q = Real.sqrt 3 := by
  sorry

end quadratic_equation_solution_l3000_300004


namespace complex_fraction_evaluation_l3000_300097

theorem complex_fraction_evaluation :
  let expr := (0.128 / 3.2 + 0.86) / ((5/6) * 1.2 + 0.8) * ((1 + 32/63 - 13/21) * 3.6) / (0.505 * 2/5 - 0.002)
  expr = 8 := by
  sorry

end complex_fraction_evaluation_l3000_300097


namespace even_divisors_of_factorial_8_l3000_300019

/-- The factorial of 8 -/
def factorial_8 : ℕ := 40320

/-- The prime factorization of 8! -/
axiom factorial_8_factorization : factorial_8 = 2^7 * 3^2 * 5 * 7

/-- A function that counts the number of even divisors of a natural number -/
def count_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating that 8! has 84 even divisors -/
theorem even_divisors_of_factorial_8 :
  count_even_divisors factorial_8 = 84 := by sorry

end even_divisors_of_factorial_8_l3000_300019


namespace cubic_polynomial_roots_l3000_300067

theorem cubic_polynomial_roots : ∃ (r₁ r₂ : ℝ), 
  (∀ x : ℝ, x^3 - 7*x^2 + 8*x + 16 = 0 ↔ x = r₁ ∨ x = r₂) ∧
  r₁ = -1 ∧ r₂ = 4 ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - r₂| < δ → |x^3 - 7*x^2 + 8*x + 16| < ε * |x - r₂|^2) :=
by sorry

end cubic_polynomial_roots_l3000_300067


namespace probability_acute_triangle_in_pentagon_l3000_300033

-- Define a regular pentagon
def RegularPentagon : Type := Unit

-- Define a function to select 3 distinct vertices from 5
def selectThreeVertices (p : RegularPentagon) : ℕ := 10

-- Define a function to count acute triangles in a regular pentagon
def countAcuteTriangles (p : RegularPentagon) : ℕ := 5

-- Define the probability of forming an acute triangle
def probabilityAcuteTriangle (p : RegularPentagon) : ℚ :=
  (countAcuteTriangles p : ℚ) / (selectThreeVertices p : ℚ)

-- Theorem statement
theorem probability_acute_triangle_in_pentagon (p : RegularPentagon) :
  probabilityAcuteTriangle p = 1 / 2 := by sorry

end probability_acute_triangle_in_pentagon_l3000_300033


namespace triangle_division_theorem_l3000_300054

/-- Represents an equilateral triangle with pegs -/
structure TriangleWithPegs where
  sideLength : ℕ
  pegDistance : ℕ

/-- Counts the number of ways to choose pegs that divide the triangle into 9 regions -/
def countValidPegChoices (t : TriangleWithPegs) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem triangle_division_theorem (t : TriangleWithPegs) :
  t.sideLength = 6 ∧ t.pegDistance = 1 → countValidPegChoices t = 456 :=
sorry

end triangle_division_theorem_l3000_300054


namespace max_even_integer_quadratic_inequality_l3000_300088

theorem max_even_integer_quadratic_inequality :
  (∃ a : ℤ, a % 2 = 0 ∧ a^2 - 12*a + 32 ≤ 0) →
  (∀ a : ℤ, a % 2 = 0 ∧ a^2 - 12*a + 32 ≤ 0 → a ≤ 8) ∧
  (∃ a : ℤ, a = 8 ∧ a % 2 = 0 ∧ a^2 - 12*a + 32 ≤ 0) :=
by sorry

end max_even_integer_quadratic_inequality_l3000_300088


namespace corn_kernel_problem_l3000_300076

theorem corn_kernel_problem (ears_per_stalk : ℕ) (num_stalks : ℕ) (total_kernels : ℕ) :
  ears_per_stalk = 4 →
  num_stalks = 108 →
  total_kernels = 237600 →
  ∃ X : ℕ,
    X * (ears_per_stalk * num_stalks / 2) +
    (X + 100) * (ears_per_stalk * num_stalks / 2) = total_kernels ∧
    X = 500 := by
  sorry

#check corn_kernel_problem

end corn_kernel_problem_l3000_300076


namespace max_sum_abc_l3000_300083

theorem max_sum_abc (a b c : ℤ) 
  (h1 : a + b = 2006) 
  (h2 : c - a = 2005) 
  (h3 : a < b) : 
  (∀ x y z : ℤ, x + y = 2006 → z - x = 2005 → x < y → x + y + z ≤ a + b + c) ∧ 
  a + b + c = 5013 :=
sorry

end max_sum_abc_l3000_300083


namespace oil_production_fraction_l3000_300091

/-- Represents the fraction of oil sent for production -/
def x : ℝ := sorry

/-- Initial sulfur concentration -/
def initial_conc : ℝ := 0.015

/-- Sulfur concentration of first replacement oil -/
def first_repl_conc : ℝ := 0.005

/-- Sulfur concentration of second replacement oil -/
def second_repl_conc : ℝ := 0.02

/-- Theorem stating that the fraction of oil sent for production is 1/2 -/
theorem oil_production_fraction :
  (initial_conc - initial_conc * x + first_repl_conc * x - 
   (initial_conc - initial_conc * x + first_repl_conc * x) * x + 
   second_repl_conc * x = initial_conc) → x = 1/2 := by
  sorry

end oil_production_fraction_l3000_300091


namespace article_a_profit_percentage_l3000_300074

/-- Profit percentage calculation for Article A -/
theorem article_a_profit_percentage 
  (x : ℝ) -- selling price of Article A
  (y : ℝ) -- selling price of Article B
  (h1 : 0.5 * x = 0.8 * (x / 1.6)) -- condition for 20% loss at half price
  (h2 : 1.05 * y = 0.9 * x) -- condition for price equality after changes
  : (0.972 * x - (x / 1.6)) / (x / 1.6) * 100 = 55.52 := by sorry

end article_a_profit_percentage_l3000_300074


namespace sons_age_l3000_300058

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 28 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry

end sons_age_l3000_300058


namespace work_completion_time_l3000_300030

theorem work_completion_time (a_time b_time : ℕ) (remaining_fraction : ℚ) : 
  a_time = 15 → b_time = 20 → remaining_fraction = 8/15 → 
  (1 - remaining_fraction) / ((1 / a_time) + (1 / b_time)) = 4 := by
  sorry

end work_completion_time_l3000_300030


namespace john_crab_earnings_l3000_300053

/-- Calculates the weekly earnings from crab sales given the following conditions:
  * Number of baskets reeled in per collection
  * Number of collections per week
  * Number of crabs per basket
  * Price per crab
-/
def weekly_crab_earnings (baskets_per_collection : ℕ) (collections_per_week : ℕ) (crabs_per_basket : ℕ) (price_per_crab : ℕ) : ℕ :=
  baskets_per_collection * collections_per_week * crabs_per_basket * price_per_crab

/-- Theorem stating that under the given conditions, John makes $72 per week from selling crabs -/
theorem john_crab_earnings :
  weekly_crab_earnings 3 2 4 3 = 72 := by
  sorry

end john_crab_earnings_l3000_300053


namespace min_value_sqrt_sum_l3000_300047

theorem min_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧
  (∀ (c d : ℝ), c > 0 → d > 0 → c + d = 1 →
    Real.sqrt (c^2 + 1) + Real.sqrt (d^2 + 4) ≥ Real.sqrt (x^2 + 1) + Real.sqrt (y^2 + 4)) ∧
  Real.sqrt (x^2 + 1) + Real.sqrt (y^2 + 4) = Real.sqrt 10 :=
sorry

end min_value_sqrt_sum_l3000_300047


namespace square_diagonal_point_theorem_l3000_300048

/-- A square with side length 10 -/
structure Square :=
  (E F G H : ℝ × ℝ)
  (is_square : 
    (E.1 - F.1)^2 + (E.2 - F.2)^2 = 100 ∧
    (F.1 - G.1)^2 + (F.2 - G.2)^2 = 100 ∧
    (G.1 - H.1)^2 + (G.2 - H.2)^2 = 100 ∧
    (H.1 - E.1)^2 + (H.2 - E.2)^2 = 100)

/-- Point Q on diagonal EH -/
def Q (s : Square) : ℝ × ℝ := sorry

/-- R1 is the circumcenter of triangle EFQ -/
def R1 (s : Square) : ℝ × ℝ := sorry

/-- R2 is the circumcenter of triangle GHQ -/
def R2 (s : Square) : ℝ × ℝ := sorry

/-- The angle between R1, Q, and R2 is 150° -/
def angle_R1QR2 (s : Square) : ℝ := sorry

theorem square_diagonal_point_theorem (s : Square) 
  (h1 : (Q s).1 > s.E.1 ∧ (Q s).1 < s.H.1)  -- EQ > HQ
  (h2 : angle_R1QR2 s = 150 * π / 180) :
  let EQ := Real.sqrt ((Q s).1 - s.E.1)^2 + ((Q s).2 - s.E.2)^2
  EQ = Real.sqrt 100 + Real.sqrt 150 := by sorry

end square_diagonal_point_theorem_l3000_300048


namespace water_volume_calculation_l3000_300073

/-- Given a volume of water that can be transferred into small hemisphere containers,
    this theorem proves the total volume of water. -/
theorem water_volume_calculation
  (hemisphere_volume : ℝ)
  (num_hemispheres : ℕ)
  (hemisphere_volume_is_4 : hemisphere_volume = 4)
  (num_hemispheres_is_2945 : num_hemispheres = 2945) :
  hemisphere_volume * num_hemispheres = 11780 :=
by sorry

end water_volume_calculation_l3000_300073


namespace complex_equation_solution_l3000_300079

theorem complex_equation_solution (a b : ℝ) : 
  (b : ℂ) + 5*I = 9 - a + a*I → b = 6 := by
sorry

end complex_equation_solution_l3000_300079


namespace trig_equality_proof_l3000_300093

theorem trig_equality_proof (x : ℝ) : 
  (Real.sin x * Real.cos (2 * x) + Real.cos x * Real.cos (4 * x) = 
   Real.sin (π / 4 + 2 * x) * Real.sin (π / 4 - 3 * x)) ↔ 
  (∃ n : ℤ, x = π / 12 * (4 * n - 1)) :=
by sorry

end trig_equality_proof_l3000_300093


namespace bac_is_105_l3000_300056

/-- Represents the encoding of a base-5 digit --/
inductive Encoding
  | A
  | B
  | C
  | D
  | E

/-- Converts an Encoding to its corresponding base-5 digit --/
def encoding_to_digit (e : Encoding) : Nat :=
  match e with
  | Encoding.A => 1
  | Encoding.B => 4
  | Encoding.C => 0
  | Encoding.D => 3
  | Encoding.E => 4

/-- Converts a sequence of Encodings to its base-10 representation --/
def encodings_to_base10 (encodings : List Encoding) : Nat :=
  encodings.enum.foldl (fun acc (i, e) => acc + encoding_to_digit e * (5 ^ (encodings.length - 1 - i))) 0

/-- The main theorem stating that BAC in the given encoding system represents 105 in base 10 --/
theorem bac_is_105 (h1 : encodings_to_base10 [Encoding.A, Encoding.B, Encoding.E] + 1 = encodings_to_base10 [Encoding.A, Encoding.B, Encoding.D])
                   (h2 : encodings_to_base10 [Encoding.A, Encoding.B, Encoding.D] + 1 = encodings_to_base10 [Encoding.A, Encoding.A, Encoding.C]) :
  encodings_to_base10 [Encoding.B, Encoding.A, Encoding.C] = 105 := by
  sorry

end bac_is_105_l3000_300056


namespace intersection_of_intervals_l3000_300036

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 0 < x}

theorem intersection_of_intervals : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

end intersection_of_intervals_l3000_300036


namespace increase_by_percentage_l3000_300026

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 110 → percentage = 50 → result = initial * (1 + percentage / 100) → result = 165 := by
  sorry

end increase_by_percentage_l3000_300026


namespace train_length_calculation_l3000_300095

theorem train_length_calculation (platform_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) :
  platform_length = 400 →
  platform_crossing_time = 42 →
  pole_crossing_time = 18 →
  ∃ train_length : ℝ,
    train_length = 300 ∧
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by sorry

end train_length_calculation_l3000_300095


namespace inequality_proof_l3000_300089

theorem inequality_proof (α β γ : ℝ) : 1 - Real.sin (α / 2) ≥ 2 * Real.sin (β / 2) * Real.sin (γ / 2) := by
  sorry

end inequality_proof_l3000_300089


namespace broken_seashells_l3000_300092

theorem broken_seashells (total : ℕ) (unbroken : ℕ) (broken : ℕ) : 
  total = 7 → unbroken = 3 → broken = total - unbroken → broken = 4 := by
sorry

end broken_seashells_l3000_300092


namespace six_digit_integers_count_l3000_300020

/-- The number of different six-digit integers that can be formed using the digits 2, 2, 2, 5, 5, and 9 -/
def six_digit_integers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of different six-digit integers
    formed using the digits 2, 2, 2, 5, 5, and 9 is equal to 60 -/
theorem six_digit_integers_count : six_digit_integers = 60 := by
  sorry

end six_digit_integers_count_l3000_300020


namespace arrangement_count_l3000_300064

theorem arrangement_count (boys girls : ℕ) (total_selected : ℕ) (girls_selected : ℕ) : 
  boys = 5 → girls = 3 → total_selected = 5 → girls_selected = 2 →
  (Nat.choose girls girls_selected) * (Nat.choose boys (total_selected - girls_selected)) * (Nat.factorial total_selected) = 3600 :=
sorry

end arrangement_count_l3000_300064


namespace one_pair_probability_l3000_300010

/-- The number of colors of socks --/
def num_colors : ℕ := 5

/-- The number of socks per color --/
def socks_per_color : ℕ := 2

/-- The total number of socks --/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn --/
def socks_drawn : ℕ := 5

/-- The probability of drawing exactly one pair of socks of the same color --/
theorem one_pair_probability : 
  (Nat.choose num_colors 4 * 4 * (2^3)) / Nat.choose total_socks socks_drawn = 40 / 63 :=
sorry

end one_pair_probability_l3000_300010


namespace negation_equivalence_l3000_300061

-- Define the set [-1, 3]
def interval : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- Define the original proposition
def original_prop (a : ℝ) : Prop := ∀ x ∈ interval, x^2 - a ≥ 0

-- Define the negation of the proposition
def negation_prop (a : ℝ) : Prop := ∃ x ∈ interval, x^2 - a < 0

-- Theorem stating that the negation of the original proposition is equivalent to negation_prop
theorem negation_equivalence (a : ℝ) : ¬(original_prop a) ↔ negation_prop a := by
  sorry

end negation_equivalence_l3000_300061


namespace jose_land_division_l3000_300022

/-- 
Given that Jose divides his land equally among himself and his four siblings,
and he ends up with 4,000 square meters, prove that the total amount of land
he initially bought was 20,000 square meters.
-/
theorem jose_land_division (jose_share : ℝ) (num_siblings : ℕ) :
  jose_share = 4000 →
  num_siblings = 4 →
  (jose_share * (num_siblings + 1) : ℝ) = 20000 := by
  sorry

end jose_land_division_l3000_300022


namespace simplify_expression_1_simplify_expression_2_l3000_300063

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  (x - y)^2 - (x + y)*(x - y) = -2*x*y + 2*y^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (12*a^2*b - 6*a*b^2) / (-3*a*b) = -4*a + 2*b := by sorry

end simplify_expression_1_simplify_expression_2_l3000_300063


namespace cistern_wet_surface_area_l3000_300090

/-- Calculates the total wet surface area of a rectangular cistern -/
def totalWetSurfaceArea (length width depth : ℝ) : ℝ :=
  let bottomArea := length * width
  let longSideArea := 2 * depth * length
  let shortSideArea := 2 * depth * width
  bottomArea + longSideArea + shortSideArea

/-- Theorem stating that the total wet surface area of a specific cistern is 68.5 m² -/
theorem cistern_wet_surface_area :
  totalWetSurfaceArea 9 4 1.25 = 68.5 := by
  sorry

#eval totalWetSurfaceArea 9 4 1.25

end cistern_wet_surface_area_l3000_300090


namespace equation_solution_l3000_300057

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (1 - x) / (x - 2) = 1 - 3 / (x - 2) ↔ x = 5/2 :=
by sorry

end equation_solution_l3000_300057


namespace fair_distribution_theorem_l3000_300040

/-- Represents the outcome of a chess game -/
inductive GameOutcome
  | A_Win
  | B_Win

/-- Represents the state of the chess competition -/
structure ChessCompetition where
  total_games : Nat
  games_played : Nat
  a_wins : Nat
  prize_money : Nat
  deriving Repr

/-- Calculates the probability of player A winning the competition -/
def probability_a_wins (comp : ChessCompetition) : Rat :=
  sorry

/-- Calculates the fair distribution of prize money -/
def fair_distribution (comp : ChessCompetition) : Nat × Nat :=
  sorry

/-- Theorem stating the fair distribution of prize money -/
theorem fair_distribution_theorem (comp : ChessCompetition) 
  (h1 : comp.total_games = 7)
  (h2 : comp.games_played = 5)
  (h3 : comp.a_wins = 3)
  (h4 : comp.prize_money = 10000) :
  fair_distribution comp = (7500, 2500) :=
sorry

end fair_distribution_theorem_l3000_300040


namespace triangle_trig_max_value_l3000_300006

theorem triangle_trig_max_value (A B C : ℝ) (h_sum : A + B + C = Real.pi) :
  (∀ A' B' C' : ℝ, A' + B' + C' = Real.pi →
    (Real.sin A * Real.cos B + Real.sin B * Real.cos C + Real.sin C * Real.cos A)^2 ≤
    (Real.sin A' * Real.cos B' + Real.sin B' * Real.cos C' + Real.sin C' * Real.cos A')^2) →
  (Real.sin A * Real.cos B + Real.sin B * Real.cos C + Real.sin C * Real.cos A)^2 = 27 / 16 :=
by sorry

end triangle_trig_max_value_l3000_300006


namespace block_rotation_theorem_l3000_300085

/-- Represents a rectangular block with three dimensions -/
structure Block where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Represents a square board -/
structure Board where
  size : ℕ

/-- Represents a face of the block -/
inductive Face
  | X
  | Y
  | Z

/-- Calculates the area of a given face of the block -/
def faceArea (b : Block) (f : Face) : ℕ :=
  match f with
  | Face.X => b.x * b.y
  | Face.Y => b.x * b.z
  | Face.Z => b.y * b.z

/-- Represents a sequence of rotations -/
def Rotations := List Face

/-- Calculates the number of unique squares contacted after a series of rotations -/
def uniqueSquaresContacted (block : Block) (board : Board) (rotations : Rotations) : ℕ :=
  sorry  -- Implementation details omitted

theorem block_rotation_theorem (block : Block) (board : Board) (rotations : Rotations) :
  block.x = 1 ∧ block.y = 2 ∧ block.z = 3 ∧
  board.size = 8 ∧
  rotations = [Face.X, Face.Y, Face.Z, Face.X, Face.Y, Face.Z] →
  uniqueSquaresContacted block board rotations = 19 :=
by sorry

end block_rotation_theorem_l3000_300085


namespace sqrt_205_between_14_and_15_l3000_300018

theorem sqrt_205_between_14_and_15 : 14 < Real.sqrt 205 ∧ Real.sqrt 205 < 15 := by
  sorry

end sqrt_205_between_14_and_15_l3000_300018


namespace workshop_attendees_count_l3000_300050

/-- Calculates the total number of people at a workshop given the number of novelists and the ratio of novelists to poets -/
def total_workshop_attendees (num_novelists : ℕ) (novelist_ratio : ℕ) (poet_ratio : ℕ) : ℕ :=
  num_novelists + (num_novelists * poet_ratio) / novelist_ratio

/-- Theorem stating that for a workshop with 15 novelists and a 5:3 ratio of novelists to poets, there are 24 people in total -/
theorem workshop_attendees_count :
  total_workshop_attendees 15 5 3 = 24 := by
  sorry

end workshop_attendees_count_l3000_300050


namespace units_digit_of_large_power_l3000_300042

/-- The units' digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The result of 3824^428 -/
def large_power : ℕ := 3824^428

theorem units_digit_of_large_power :
  units_digit large_power = 6 := by
  sorry

end units_digit_of_large_power_l3000_300042


namespace point_in_region_implies_a_negative_l3000_300014

theorem point_in_region_implies_a_negative (a : ℝ) :
  (2 * a + 3 < 3) → a < 0 := by
  sorry

end point_in_region_implies_a_negative_l3000_300014


namespace unique_solution_quadratic_l3000_300045

theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃! x, q * x^2 - 8 * x + 2 = 0) ↔ q = 8 := by
  sorry

end unique_solution_quadratic_l3000_300045


namespace earth_hour_seating_l3000_300025

theorem earth_hour_seating (x : ℕ) : 30 * x + 8 = 31 * x - 26 := by
  sorry

end earth_hour_seating_l3000_300025


namespace statement_holds_only_in_specific_cases_l3000_300011

-- Define the basic types
inductive GeometricObject
| Line
| Plane

-- Define the relationships
def perpendicular (a b : GeometricObject) : Prop := sorry
def parallel (a b : GeometricObject) : Prop := sorry

-- Define the statement
def statement (x y z : GeometricObject) : Prop :=
  perpendicular x z → perpendicular y z → parallel x y

-- Theorem to prove
theorem statement_holds_only_in_specific_cases 
  (x y z : GeometricObject) : 
  statement x y z ↔ 
    ((x = GeometricObject.Line ∧ y = GeometricObject.Line ∧ z = GeometricObject.Plane) ∨
     (x = GeometricObject.Plane ∧ y = GeometricObject.Plane ∧ z = GeometricObject.Line)) :=
by sorry

end statement_holds_only_in_specific_cases_l3000_300011


namespace smallest_x_value_l3000_300021

theorem smallest_x_value (x : ℝ) :
  (x^2 - 5*x - 84) / (x - 9) = 4 / (x + 6) →
  x ≥ (-13 - Real.sqrt 17) / 2 ∧
  ∃ y : ℝ, y < (-13 - Real.sqrt 17) / 2 ∧ (y^2 - 5*y - 84) / (y - 9) ≠ 4 / (y + 6) :=
by sorry

end smallest_x_value_l3000_300021


namespace max_candies_bob_l3000_300027

theorem max_candies_bob (total : ℕ) (h1 : total = 30) : ∃ (bob : ℕ), bob ≤ 10 ∧ bob + 2 * bob = total := by
  sorry

end max_candies_bob_l3000_300027


namespace vector_calculation_l3000_300034

/-- Given two plane vectors a and b, prove that (1/2)a - (3/2)b equals (-1, 2) -/
theorem vector_calculation (a b : ℝ × ℝ) : 
  a = (1, 1) → b = (1, -1) → (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by
  sorry

end vector_calculation_l3000_300034


namespace complex_fraction_real_l3000_300094

theorem complex_fraction_real (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (Complex.ofReal a - Complex.I) / (2 + Complex.I) ∈ Set.range Complex.ofReal →
  a = -2 := by
  sorry

end complex_fraction_real_l3000_300094
