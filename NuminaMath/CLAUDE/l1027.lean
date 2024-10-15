import Mathlib

namespace NUMINAMATH_CALUDE_chessboard_square_rectangle_ratio_l1027_102761

/-- The number of rectangles formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_rectangles : ℕ := 2025

/-- The number of squares formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_squares : ℕ := 285

/-- The ratio of squares to rectangles expressed as a fraction with relatively prime positive integers -/
def square_rectangle_ratio : ℚ := 19 / 135

theorem chessboard_square_rectangle_ratio :
  (num_squares : ℚ) / (num_rectangles : ℚ) = square_rectangle_ratio := by
  sorry

end NUMINAMATH_CALUDE_chessboard_square_rectangle_ratio_l1027_102761


namespace NUMINAMATH_CALUDE_veranda_width_l1027_102770

/-- Proves that the width of a veranda surrounding a rectangular room is 2 meters -/
theorem veranda_width (room_length room_width veranda_area : ℝ) : 
  room_length = 18 → 
  room_width = 12 → 
  veranda_area = 136 → 
  ∃ w : ℝ, w = 2 ∧ 
    (room_length + 2 * w) * (room_width + 2 * w) - room_length * room_width = veranda_area :=
by sorry

end NUMINAMATH_CALUDE_veranda_width_l1027_102770


namespace NUMINAMATH_CALUDE_sin_has_property_T_l1027_102702

-- Define property T
def has_property_T (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (deriv f x₁) * (deriv f x₂) = -1

-- State the theorem
theorem sin_has_property_T :
  has_property_T Real.sin :=
sorry

end NUMINAMATH_CALUDE_sin_has_property_T_l1027_102702


namespace NUMINAMATH_CALUDE_cubic_polynomial_theorem_l1027_102701

/-- Represents a cubic polynomial a₃x³ - x² + a₁x - 7 = 0 -/
structure CubicPolynomial where
  a₃ : ℝ
  a₁ : ℝ

/-- Represents the roots of the cubic polynomial -/
structure Roots where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Checks if the given roots satisfy the condition -/
def satisfiesCondition (r : Roots) : Prop :=
  (225 * r.α^2) / (r.α^2 + 7) = (144 * r.β^2) / (r.β^2 + 7) ∧
  (144 * r.β^2) / (r.β^2 + 7) = (100 * r.γ^2) / (r.γ^2 + 7)

/-- Checks if the given roots are positive -/
def arePositive (r : Roots) : Prop :=
  r.α > 0 ∧ r.β > 0 ∧ r.γ > 0

/-- Checks if the given roots are valid for the cubic polynomial -/
def areValidRoots (p : CubicPolynomial) (r : Roots) : Prop :=
  p.a₃ * r.α^3 - r.α^2 + p.a₁ * r.α - 7 = 0 ∧
  p.a₃ * r.β^3 - r.β^2 + p.a₁ * r.β - 7 = 0 ∧
  p.a₃ * r.γ^3 - r.γ^2 + p.a₁ * r.γ - 7 = 0

theorem cubic_polynomial_theorem (p : CubicPolynomial) (r : Roots) 
  (h1 : satisfiesCondition r)
  (h2 : arePositive r)
  (h3 : areValidRoots p r) :
  abs (p.a₁ - 130.6667) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_theorem_l1027_102701


namespace NUMINAMATH_CALUDE_green_tile_probability_l1027_102784

theorem green_tile_probability :
  let tiles := Finset.range 100
  let green_tiles := tiles.filter (fun n => (n + 1) % 7 = 3)
  (green_tiles.card : ℚ) / tiles.card = 7 / 50 := by
  sorry

end NUMINAMATH_CALUDE_green_tile_probability_l1027_102784


namespace NUMINAMATH_CALUDE_popsicle_sticks_count_l1027_102767

theorem popsicle_sticks_count (gino_sticks : ℕ) (total_sticks : ℕ) (my_sticks : ℕ) : 
  gino_sticks = 63 → total_sticks = 113 → total_sticks = gino_sticks + my_sticks → my_sticks = 50 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_sticks_count_l1027_102767


namespace NUMINAMATH_CALUDE_max_value_implies_m_value_l1027_102797

def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem max_value_implies_m_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, f m x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-2) 2, f m x = 3) →
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_m_value_l1027_102797


namespace NUMINAMATH_CALUDE_no_consecutive_perfect_squares_l1027_102738

theorem no_consecutive_perfect_squares (a b : ℤ) : a^2 - b^2 = 1 → (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_perfect_squares_l1027_102738


namespace NUMINAMATH_CALUDE_tangent_line_at_2_minus_6_tangent_lines_slope_4_l1027_102744

-- Define the function f(x) = x³ + x - 16
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem for the tangent line at (2, -6)
theorem tangent_line_at_2_minus_6 :
  ∃ (m b : ℝ), m * 2 - b = f 2 ∧ 
               m = f' 2 ∧
               ∀ x, m * x - b = 13 * x - 32 :=
sorry

-- Theorem for tangent lines with slope 4
theorem tangent_lines_slope_4 :
  ∃ (x₁ x₂ b₁ b₂ : ℝ), 
    x₁ ≠ x₂ ∧
    f' x₁ = 4 ∧ f' x₂ = 4 ∧
    4 * x₁ - b₁ = f x₁ ∧
    4 * x₂ - b₂ = f x₂ ∧
    (∀ x, 4 * x - b₁ = 4 * x - 18) ∧
    (∀ x, 4 * x - b₂ = 4 * x - 14) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_minus_6_tangent_lines_slope_4_l1027_102744


namespace NUMINAMATH_CALUDE_sandra_son_age_l1027_102790

/-- Sandra's current age -/
def sandra_age : ℕ := 36

/-- The ratio of Sandra's age to her son's age 3 years ago -/
def age_ratio : ℕ := 3

/-- Sandra's son's current age -/
def son_age : ℕ := 14

theorem sandra_son_age : 
  sandra_age - 3 = age_ratio * (son_age - 3) :=
sorry

end NUMINAMATH_CALUDE_sandra_son_age_l1027_102790


namespace NUMINAMATH_CALUDE_solar_panel_flat_fee_l1027_102729

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def house_cost : ℕ := 120000
def cow_count : ℕ := 20
def cow_cost_per_unit : ℕ := 1000
def chicken_count : ℕ := 100
def chicken_cost_per_unit : ℕ := 5
def solar_installation_hours : ℕ := 6
def solar_installation_cost_per_hour : ℕ := 100
def total_cost : ℕ := 147700

theorem solar_panel_flat_fee :
  total_cost - (land_acres * land_cost_per_acre + house_cost + 
    cow_count * cow_cost_per_unit + chicken_count * chicken_cost_per_unit + 
    solar_installation_hours * solar_installation_cost_per_hour) = 26000 := by
  sorry

end NUMINAMATH_CALUDE_solar_panel_flat_fee_l1027_102729


namespace NUMINAMATH_CALUDE_salon_customers_count_l1027_102773

/-- Represents the number of customers who made only one visit -/
def single_visit_customers : ℕ := 44

/-- Represents the number of customers who made two visits -/
def double_visit_customers : ℕ := 30

/-- Represents the number of customers who made three visits -/
def triple_visit_customers : ℕ := 10

/-- The cost of the first visit in a calendar month -/
def first_visit_cost : ℕ := 10

/-- The cost of each subsequent visit in the same calendar month -/
def subsequent_visit_cost : ℕ := 8

/-- The total revenue for the calendar month -/
def total_revenue : ℕ := 1240

theorem salon_customers_count :
  single_visit_customers + double_visit_customers + triple_visit_customers = 84 ∧
  first_visit_cost * (single_visit_customers + double_visit_customers + triple_visit_customers) +
  subsequent_visit_cost * (double_visit_customers + 2 * triple_visit_customers) = total_revenue :=
sorry

end NUMINAMATH_CALUDE_salon_customers_count_l1027_102773


namespace NUMINAMATH_CALUDE_amulet_seller_profit_l1027_102769

/-- Calculates the profit for an amulet seller at a Ren Faire --/
theorem amulet_seller_profit
  (days : ℕ)
  (amulets_per_day : ℕ)
  (selling_price : ℕ)
  (cost_price : ℕ)
  (faire_cut_percent : ℕ)
  (h1 : days = 2)
  (h2 : amulets_per_day = 25)
  (h3 : selling_price = 40)
  (h4 : cost_price = 30)
  (h5 : faire_cut_percent = 10)
  : (days * amulets_per_day * selling_price) - 
    (days * amulets_per_day * cost_price) - 
    (days * amulets_per_day * selling_price * faire_cut_percent / 100) = 300 :=
by sorry

end NUMINAMATH_CALUDE_amulet_seller_profit_l1027_102769


namespace NUMINAMATH_CALUDE_fraction_comparison_l1027_102732

theorem fraction_comparison (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  ((1 + y) / x < 2) ∨ ((1 + x) / y < 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1027_102732


namespace NUMINAMATH_CALUDE_expand_polynomial_l1027_102742

theorem expand_polynomial (x : ℝ) : 
  (13 * x^2 + 5 * x + 3) * (3 * x^3) = 39 * x^5 + 15 * x^4 + 9 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1027_102742


namespace NUMINAMATH_CALUDE_sqrt_11_between_3_and_4_l1027_102764

theorem sqrt_11_between_3_and_4 : 3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_11_between_3_and_4_l1027_102764


namespace NUMINAMATH_CALUDE_inequality_proof_l1027_102786

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) :
  x + y / 2 + z / 3 ≤ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1027_102786


namespace NUMINAMATH_CALUDE_daniel_initial_noodles_l1027_102711

/-- The number of noodles Daniel gave to William -/
def noodles_given : ℕ := 12

/-- The number of noodles Daniel has now -/
def noodles_remaining : ℕ := 54

/-- The initial number of noodles Daniel had -/
def initial_noodles : ℕ := noodles_given + noodles_remaining

theorem daniel_initial_noodles :
  initial_noodles = 66 :=
by sorry

end NUMINAMATH_CALUDE_daniel_initial_noodles_l1027_102711


namespace NUMINAMATH_CALUDE_propositions_truth_l1027_102760

-- Proposition 1
def proposition1 : Prop := ∃ a b : ℝ, a ≤ b ∧ a^2 > b^2

-- Proposition 2
def proposition2 : Prop := ∀ x y : ℝ, x = -y → x + y = 0

-- Proposition 3
def proposition3 : Prop := ∀ x : ℝ, (x ≤ -2 ∨ x ≥ 2) → x^2 ≥ 4

theorem propositions_truth : ¬proposition1 ∧ proposition2 ∧ proposition3 := by
  sorry

end NUMINAMATH_CALUDE_propositions_truth_l1027_102760


namespace NUMINAMATH_CALUDE_three_toppings_from_seven_l1027_102779

theorem three_toppings_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_three_toppings_from_seven_l1027_102779


namespace NUMINAMATH_CALUDE_power_of_two_floor_l1027_102724

theorem power_of_two_floor (n : ℕ) (h1 : n ≥ 4) 
  (h2 : ∃ k : ℕ, ⌊(2^n : ℝ) / n⌋ = 2^k) : 
  ∃ m : ℕ, n = 2^m :=
sorry

end NUMINAMATH_CALUDE_power_of_two_floor_l1027_102724


namespace NUMINAMATH_CALUDE_circle_properties_l1027_102731

def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 12*x - 12*y - 88

def line_equation (x y : ℝ) : ℝ := x + 3*y + 16

def point_A : ℝ × ℝ := (-6, 10)
def point_B : ℝ × ℝ := (2, -6)

theorem circle_properties :
  (circle_equation point_A.1 point_A.2 = 0) ∧
  (circle_equation point_B.1 point_B.2 = 0) ∧
  (line_equation point_B.1 point_B.2 = 0) ∧
  (∃ (t : ℝ), t ≠ 0 ∧
    (2 * point_B.1 - 12) * 1 + (2 * point_B.2 - 12) * 3 = t * (1^2 + 3^2)) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l1027_102731


namespace NUMINAMATH_CALUDE_circle_coloring_exists_l1027_102736

/-- A type representing the two colors we can use -/
inductive Color
  | Red
  | Blue

/-- A type representing a region in the plane -/
structure Region

/-- A type representing a circle in the plane -/
structure Circle

/-- A function that determines if two regions are adjacent (separated by an arc of a circle) -/
def adjacent (r1 r2 : Region) : Prop := sorry

/-- A coloring function that assigns a color to each region -/
def coloring (r : Region) : Color := sorry

/-- The main theorem stating that a valid coloring exists for any number of circles -/
theorem circle_coloring_exists (n : ℕ) (h : n ≥ 1) :
  ∃ (circles : Finset Circle) (regions : Finset Region),
    circles.card = n ∧
    (∀ r1 r2 : Region, r1 ∈ regions → r2 ∈ regions → adjacent r1 r2 → coloring r1 ≠ coloring r2) :=
  sorry

end NUMINAMATH_CALUDE_circle_coloring_exists_l1027_102736


namespace NUMINAMATH_CALUDE_specific_tangent_distances_l1027_102781

/-- Two externally tangent circles with radii R and r -/
structure TangentCircles where
  R : ℝ
  r : ℝ
  h_positive_R : R > 0
  h_positive_r : r > 0
  h_external : R > r

/-- The distances from the point of tangency to the common tangents -/
def tangent_distances (c : TangentCircles) : Set ℝ :=
  {0, (c.R + c.r) * c.r / c.R}

/-- Theorem about the distances for specific radii -/
theorem specific_tangent_distances :
  ∃ c : TangentCircles, c.R = 3 ∧ c.r = 1 ∧ tangent_distances c = {0, 7/3} := by
  sorry

end NUMINAMATH_CALUDE_specific_tangent_distances_l1027_102781


namespace NUMINAMATH_CALUDE_ladder_problem_l1027_102704

theorem ladder_problem (c a b : ℝ) : 
  c = 25 → a = 15 → c^2 = a^2 + b^2 → b = 20 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1027_102704


namespace NUMINAMATH_CALUDE_number_of_dimes_l1027_102715

/-- Proves the number of dimes given the number of pennies, nickels, and total value -/
theorem number_of_dimes 
  (num_pennies : ℕ) 
  (num_nickels : ℕ) 
  (total_value : ℚ) 
  (h_num_pennies : num_pennies = 9)
  (h_num_nickels : num_nickels = 4)
  (h_total_value : total_value = 59 / 100)
  (h_penny_value : ∀ n : ℕ, n * (1 / 100 : ℚ) = (n : ℚ) / 100)
  (h_nickel_value : ∀ n : ℕ, n * (5 / 100 : ℚ) = (5 * n : ℚ) / 100)
  (h_dime_value : ∀ n : ℕ, n * (10 / 100 : ℚ) = (10 * n : ℚ) / 100) :
  ∃ num_dimes : ℕ, 
    num_dimes = 3 ∧ 
    total_value = 
      num_pennies * (1 / 100 : ℚ) + 
      num_nickels * (5 / 100 : ℚ) + 
      num_dimes * (10 / 100 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_number_of_dimes_l1027_102715


namespace NUMINAMATH_CALUDE_expression_value_l1027_102756

theorem expression_value (a : ℝ) (h1 : a - 1 ≥ 0) (h2 : 1 - a ≥ 0) :
  a + 2 * Real.sqrt (a - 1) - Real.sqrt (1 - a) + 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1027_102756


namespace NUMINAMATH_CALUDE_one_different_value_l1027_102799

/-- The standard exponentiation result for 2^(2^(2^2)) -/
def standard_result : ℕ := 65536

/-- The set of all possible values obtained by different parenthesizations of 2^2^2^2 -/
def all_results : Set ℕ :=
  {2^(2^(2^2)), 2^((2^2)^2), ((2^2)^2)^2, (2^(2^2))^2, (2^2)^(2^2)}

/-- The theorem stating that there is exactly one value different from the standard result -/
theorem one_different_value :
  ∃! x, x ∈ all_results ∧ x ≠ standard_result :=
sorry

end NUMINAMATH_CALUDE_one_different_value_l1027_102799


namespace NUMINAMATH_CALUDE_janes_farm_chickens_l1027_102791

/-- Represents the farm scenario with chickens and egg production --/
structure Farm where
  chickens : ℕ
  eggs_per_chicken_per_week : ℕ
  price_per_dozen : ℕ
  weeks : ℕ
  total_revenue : ℕ

/-- Calculates the total number of eggs produced by the farm in the given period --/
def total_eggs (f : Farm) : ℕ :=
  f.chickens * f.eggs_per_chicken_per_week * f.weeks

/-- Calculates the revenue generated from selling all eggs --/
def revenue (f : Farm) : ℕ :=
  (total_eggs f / 12) * f.price_per_dozen

/-- Theorem stating that Jane's farm has 10 chickens given the conditions --/
theorem janes_farm_chickens :
  ∃ (f : Farm),
    f.eggs_per_chicken_per_week = 6 ∧
    f.price_per_dozen = 2 ∧
    f.weeks = 2 ∧
    f.total_revenue = 20 ∧
    f.chickens = 10 :=
  sorry

end NUMINAMATH_CALUDE_janes_farm_chickens_l1027_102791


namespace NUMINAMATH_CALUDE_total_cost_is_four_dollars_l1027_102768

/-- The cost of a single tire in dollars -/
def cost_per_tire : ℝ := 0.50

/-- The number of tires -/
def number_of_tires : ℕ := 8

/-- The total cost of all tires -/
def total_cost : ℝ := cost_per_tire * number_of_tires

theorem total_cost_is_four_dollars : total_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_four_dollars_l1027_102768


namespace NUMINAMATH_CALUDE_bisecting_cross_section_dihedral_angle_l1027_102707

/-- Regular tetrahedron with specific dimensions -/
structure RegularTetrahedron where
  -- Base side length
  base_side : ℝ
  -- Side edge length
  side_edge : ℝ
  -- Assumption that base_side = 1 and side_edge = 2
  base_side_eq_one : base_side = 1
  side_edge_eq_two : side_edge = 2

/-- Cross-section that bisects the tetrahedron's volume -/
structure BisectingCrossSection (t : RegularTetrahedron) where
  -- The cross-section passes through edge AB of the base
  passes_through_base_edge : Prop

/-- Dihedral angle between the cross-section and the base -/
def dihedralAngle (t : RegularTetrahedron) (cs : BisectingCrossSection t) : ℝ :=
  sorry -- Definition of dihedral angle

/-- Main theorem -/
theorem bisecting_cross_section_dihedral_angle 
  (t : RegularTetrahedron) (cs : BisectingCrossSection t) : 
  Real.cos (dihedralAngle t cs) = 2 * Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_cross_section_dihedral_angle_l1027_102707


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1027_102733

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (27 - 9*x - x^2 = 0) → 
  (∃ r s : ℝ, (27 - 9*r - r^2 = 0) ∧ (27 - 9*s - s^2 = 0) ∧ (r + s = 9)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1027_102733


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l1027_102709

theorem isosceles_triangle_angles (a b c : ℝ) : 
  -- The triangle is isosceles
  (a = b ∨ b = c ∨ a = c) →
  -- One of the interior angles is 50°
  (a = 50 ∨ b = 50 ∨ c = 50) →
  -- The sum of interior angles in a triangle is 180°
  a + b + c = 180 →
  -- The other two angles are either (65°, 65°) or (80°, 50°)
  ((a = 65 ∧ b = 65 ∧ c = 50) ∨ 
   (a = 65 ∧ c = 65 ∧ b = 50) ∨ 
   (b = 65 ∧ c = 65 ∧ a = 50) ∨
   (a = 80 ∧ b = 50 ∧ c = 50) ∨ 
   (a = 50 ∧ b = 80 ∧ c = 50) ∨ 
   (a = 50 ∧ b = 50 ∧ c = 80)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l1027_102709


namespace NUMINAMATH_CALUDE_birds_in_pet_shop_l1027_102714

/-- The number of birds in a pet shop -/
def number_of_birds (total animals : ℕ) (kittens hamsters : ℕ) : ℕ :=
  total - kittens - hamsters

/-- Theorem: There are 30 birds in the pet shop -/
theorem birds_in_pet_shop :
  let total := 77
  let kittens := 32
  let hamsters := 15
  number_of_birds total kittens hamsters = 30 := by
sorry

end NUMINAMATH_CALUDE_birds_in_pet_shop_l1027_102714


namespace NUMINAMATH_CALUDE_subgroup_equality_l1027_102719

variable {G : Type*} [Group G]

theorem subgroup_equality (S : Set G) (x s : G) (hs : s ∈ Subgroup.closure S) :
  Subgroup.closure (S ∪ {x}) = Subgroup.closure (S ∪ {x * s}) ∧
  Subgroup.closure (S ∪ {x}) = Subgroup.closure (S ∪ {s * x}) := by
  sorry

end NUMINAMATH_CALUDE_subgroup_equality_l1027_102719


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1027_102734

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I)^2 = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1027_102734


namespace NUMINAMATH_CALUDE_percentage_calculation_l1027_102775

theorem percentage_calculation : (0.47 * 1442 - 0.36 * 1412) + 63 = 232.42 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1027_102775


namespace NUMINAMATH_CALUDE_tree_cutting_percentage_l1027_102782

theorem tree_cutting_percentage (initial_trees : ℕ) (final_trees : ℕ) (replant_rate : ℕ) : 
  initial_trees = 400 → 
  final_trees = 720 → 
  replant_rate = 5 → 
  (100 * (final_trees - initial_trees)) / (initial_trees * (replant_rate - 1)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tree_cutting_percentage_l1027_102782


namespace NUMINAMATH_CALUDE_fraction_equality_l1027_102743

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) : 
  (2 * m * r - 5 * n * t) / (5 * n * t - 4 * m * r) = -2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1027_102743


namespace NUMINAMATH_CALUDE_jane_lemonade_glasses_l1027_102722

/-- The number of glasses of lemonade that can be made -/
def glasses_of_lemonade (total_lemons : ℕ) (lemons_per_glass : ℕ) : ℕ :=
  total_lemons / lemons_per_glass

/-- Theorem: Jane can make 9 glasses of lemonade -/
theorem jane_lemonade_glasses : glasses_of_lemonade 18 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jane_lemonade_glasses_l1027_102722


namespace NUMINAMATH_CALUDE_multiples_of_3_or_5_not_6_up_to_200_l1027_102725

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n / m : ℕ)

def count_multiples_of_3_or_5_not_6 (upper_bound : ℕ) : ℕ :=
  count_multiples upper_bound 3 +
  count_multiples upper_bound 5 -
  count_multiples upper_bound 15 -
  count_multiples upper_bound 6

theorem multiples_of_3_or_5_not_6_up_to_200 :
  count_multiples_of_3_or_5_not_6 200 = 60 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_3_or_5_not_6_up_to_200_l1027_102725


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1027_102777

theorem inequality_solution_set 
  (a b : ℝ) (ha : a < 0) : 
  {x : ℝ | a * x + b < 0} = {x : ℝ | x > -b / a} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1027_102777


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1027_102728

theorem algebraic_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ -b) :
  (a^2 + a*b + b^2) / (a + b) - (a^2 - a*b + b^2) / (a - b) + (2*b^2 - b^2 + a^2) / (a^2 - b^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1027_102728


namespace NUMINAMATH_CALUDE_break_even_is_80_weeks_l1027_102798

/-- Represents the chicken and egg problem --/
structure ChickenEggProblem where
  num_chickens : ℕ
  chicken_cost : ℚ
  weekly_feed_cost : ℚ
  eggs_per_chicken : ℕ
  eggs_bought_weekly : ℕ
  egg_cost_per_dozen : ℚ

/-- Calculates the break-even point in weeks --/
def break_even_weeks (problem : ChickenEggProblem) : ℕ :=
  sorry

/-- Theorem stating that the break-even point is 80 weeks for the given problem --/
theorem break_even_is_80_weeks (problem : ChickenEggProblem)
  (h1 : problem.num_chickens = 4)
  (h2 : problem.chicken_cost = 20)
  (h3 : problem.weekly_feed_cost = 1)
  (h4 : problem.eggs_per_chicken = 3)
  (h5 : problem.eggs_bought_weekly = 12)
  (h6 : problem.egg_cost_per_dozen = 2) :
  break_even_weeks problem = 80 :=
sorry

end NUMINAMATH_CALUDE_break_even_is_80_weeks_l1027_102798


namespace NUMINAMATH_CALUDE_painted_faces_count_l1027_102754

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  is_painted : Bool := true

/-- Counts the number of smaller cubes with at least two painted faces when a painted cube is cut into unit cubes -/
def count_painted_faces (c : PaintedCube 4) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 painted cube cut into 1x1x1 cubes has 32 smaller cubes with at least two painted faces -/
theorem painted_faces_count (c : PaintedCube 4) : count_painted_faces c = 32 :=
  sorry

end NUMINAMATH_CALUDE_painted_faces_count_l1027_102754


namespace NUMINAMATH_CALUDE_chord_length_l1027_102774

theorem chord_length (r : ℝ) (h : r = 15) : 
  let chord_length : ℝ := 2 * (r^2 - (r/3)^2).sqrt
  chord_length = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l1027_102774


namespace NUMINAMATH_CALUDE_acrobat_weight_l1027_102752

/-- Given weights of various objects, prove that an acrobat weighs twice as much as a lamb -/
theorem acrobat_weight (barrel dog acrobat lamb coil : ℝ) 
  (h1 : acrobat + dog = 2 * barrel)
  (h2 : dog = 2 * coil)
  (h3 : lamb + coil = barrel) :
  acrobat = 2 * lamb := by
  sorry

end NUMINAMATH_CALUDE_acrobat_weight_l1027_102752


namespace NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l1027_102748

-- Define the set of geometric solids
inductive GeometricSolid
  | Cone
  | Cylinder
  | TriangularPyramid
  | QuadrangularPrism

-- Define a predicate for having a quadrilateral front view
def hasQuadrilateralFrontView (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => True
  | GeometricSolid.QuadrangularPrism => True
  | _ => False

-- Theorem statement
theorem quadrilateral_front_view_solids :
  ∀ (solid : GeometricSolid),
    hasQuadrilateralFrontView solid ↔
      (solid = GeometricSolid.Cylinder ∨ solid = GeometricSolid.QuadrangularPrism) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l1027_102748


namespace NUMINAMATH_CALUDE_bobs_walking_rate_l1027_102700

/-- Proves that Bob's walking rate is 3 miles per hour given the conditions of the problem -/
theorem bobs_walking_rate (total_distance : ℝ) (yolanda_rate : ℝ) (bob_distance : ℝ) :
  total_distance = 17 →
  yolanda_rate = 3 →
  bob_distance = 8 →
  ∃ (bob_rate : ℝ), bob_rate = 3 ∧ bob_rate * (total_distance / (yolanda_rate + bob_rate) - 1) = bob_distance :=
by sorry

end NUMINAMATH_CALUDE_bobs_walking_rate_l1027_102700


namespace NUMINAMATH_CALUDE_jean_sale_savings_l1027_102763

/-- Represents the total savings during a jean sale -/
def total_savings (fox_price pony_price : ℚ) (fox_discount pony_discount : ℚ) (fox_quantity pony_quantity : ℕ) : ℚ :=
  (fox_price * fox_quantity * fox_discount / 100) + (pony_price * pony_quantity * pony_discount / 100)

/-- Theorem stating the total savings during the jean sale -/
theorem jean_sale_savings :
  let fox_price : ℚ := 15
  let pony_price : ℚ := 18
  let fox_quantity : ℕ := 3
  let pony_quantity : ℕ := 2
  let pony_discount : ℚ := 13.999999999999993
  let fox_discount : ℚ := 22 - pony_discount
  total_savings fox_price pony_price fox_discount pony_discount fox_quantity pony_quantity = 864 / 100 :=
by
  sorry


end NUMINAMATH_CALUDE_jean_sale_savings_l1027_102763


namespace NUMINAMATH_CALUDE_initial_red_marbles_l1027_102721

/-- Given a bag of red and green marbles with the following properties:
    1. The initial ratio of red to green marbles is 5:3
    2. After adding 15 red marbles and removing 9 green marbles, the new ratio is 3:1
    This theorem proves that the initial number of red marbles is 52.5 -/
theorem initial_red_marbles (r g : ℚ) : 
  r / g = 5 / 3 →
  (r + 15) / (g - 9) = 3 / 1 →
  r = 52.5 := by
sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l1027_102721


namespace NUMINAMATH_CALUDE_quadratic_function_with_log_range_l1027_102751

/-- A quadratic function -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_with_log_range
  (f : ℝ → ℝ)
  (h1 : QuadraticFunction f)
  (h2 : Set.range (fun x ↦ Real.log (f x)) = Set.Ici 0) :
  ∃ a b : ℝ, f = fun x ↦ x^2 + 2*x + 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_with_log_range_l1027_102751


namespace NUMINAMATH_CALUDE_pages_difference_l1027_102718

theorem pages_difference (total_pages book_length first_day fourth_day : ℕ) : 
  book_length = 354 → 
  first_day = 63 → 
  fourth_day = 29 → 
  total_pages = 4 → 
  (book_length - (first_day + 2 * first_day + fourth_day)) - 2 * first_day = 10 := by
  sorry

end NUMINAMATH_CALUDE_pages_difference_l1027_102718


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l1027_102765

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice being thrown -/
def numDice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def minSum : ℕ := numDice

/-- The maximum possible sum when rolling the dice -/
def maxSum : ℕ := numDice * sides

/-- The number of possible unique sums -/
def uniqueSums : ℕ := maxSum - minSum + 1

/-- The minimum number of throws needed to guarantee a repeated sum -/
def minThrows : ℕ := uniqueSums + 1

theorem min_throws_for_repeated_sum :
  minThrows = 22 := by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l1027_102765


namespace NUMINAMATH_CALUDE_inequality_range_l1027_102792

theorem inequality_range (x : ℝ) : 
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) ↔ (x < -1 ∨ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1027_102792


namespace NUMINAMATH_CALUDE_right_triangle_area_l1027_102771

/-- The area of a right triangle with base 8 and hypotenuse 10 is 24 square units. -/
theorem right_triangle_area : 
  ∀ (base height hypotenuse : ℝ),
  base = 8 →
  hypotenuse = 10 →
  base ^ 2 + height ^ 2 = hypotenuse ^ 2 →
  (1 / 2) * base * height = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1027_102771


namespace NUMINAMATH_CALUDE_ball_max_height_l1027_102785

/-- The height function of the ball's path -/
def f (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- The maximum height reached by the ball -/
def max_height : ℝ := 40

/-- Theorem stating that the maximum value of f is equal to max_height -/
theorem ball_max_height : ∀ t : ℝ, f t ≤ max_height := by sorry

end NUMINAMATH_CALUDE_ball_max_height_l1027_102785


namespace NUMINAMATH_CALUDE_apples_per_pie_l1027_102749

theorem apples_per_pie 
  (total_apples : ℕ) 
  (unripe_apples : ℕ) 
  (num_pies : ℕ) 
  (h1 : total_apples = 34) 
  (h2 : unripe_apples = 6) 
  (h3 : num_pies = 7) 
  (h4 : unripe_apples < total_apples) :
  (total_apples - unripe_apples) / num_pies = 4 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l1027_102749


namespace NUMINAMATH_CALUDE_candy_division_l1027_102717

theorem candy_division (p q r : ℕ) (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0)
  (h_order : p < q ∧ q < r) (h_a : 20 = 3 * r - 2 * p) (h_b : 10 = r - p)
  (h_c : 9 = 3 * q - 3 * p) (h_c_sum : 3 * q = 18) :
  p = 3 ∧ q = 6 ∧ r = 13 := by sorry

end NUMINAMATH_CALUDE_candy_division_l1027_102717


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l1027_102780

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, -1, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l1027_102780


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1027_102783

/-- Given a geometric sequence {a_n} with common ratio q = 2,
    if the arithmetic mean of a_2 and 2a_3 is 5, then a_1 = 1 -/
theorem geometric_sequence_first_term
  (a : ℕ → ℝ)  -- a is the sequence
  (h_geom : ∀ n, a (n + 1) = 2 * a n)  -- geometric sequence with ratio 2
  (h_mean : (a 2 + 2 * a 3) / 2 = 5)  -- arithmetic mean condition
  : a 1 = 1 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1027_102783


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l1027_102723

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l1027_102723


namespace NUMINAMATH_CALUDE_balance_difference_theorem_l1027_102757

def initial_deposit : ℝ := 10000
def jasmine_rate : ℝ := 0.04
def lucas_rate : ℝ := 0.06
def years : ℕ := 20

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

theorem balance_difference_theorem :
  ∃ ε > 0, ε < 1 ∧
  (simple_interest initial_deposit lucas_rate years -
   compound_interest initial_deposit jasmine_rate years) - 89 < ε :=
sorry

end NUMINAMATH_CALUDE_balance_difference_theorem_l1027_102757


namespace NUMINAMATH_CALUDE_reinforcement_size_l1027_102727

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days passed before reinforcement, and remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
  (days_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_duration
  let provisions_left := total_provisions - initial_garrison * days_passed
  let new_total_men := initial_garrison + (provisions_left / remaining_duration - initial_garrison)
  new_total_men - initial_garrison

/-- Theorem stating that given the problem conditions, the reinforcement size is 3000. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 65 15 20 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l1027_102727


namespace NUMINAMATH_CALUDE_flowers_per_pot_l1027_102747

/-- Given 141 pots and 10011 flowers in total, prove that each pot contains 71 flowers. -/
theorem flowers_per_pot (total_pots : ℕ) (total_flowers : ℕ) (h1 : total_pots = 141) (h2 : total_flowers = 10011) :
  total_flowers / total_pots = 71 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_pot_l1027_102747


namespace NUMINAMATH_CALUDE_least_product_of_three_primes_greater_than_10_l1027_102745

theorem least_product_of_three_primes_greater_than_10 :
  ∃ (p q r : ℕ),
    Prime p ∧ Prime q ∧ Prime r ∧
    p > 10 ∧ q > 10 ∧ r > 10 ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    p * q * r = 2431 ∧
    (∀ (a b c : ℕ),
      Prime a ∧ Prime b ∧ Prime c ∧
      a > 10 ∧ b > 10 ∧ c > 10 ∧
      a ≠ b ∧ a ≠ c ∧ b ≠ c →
      a * b * c ≥ 2431) :=
by sorry


end NUMINAMATH_CALUDE_least_product_of_three_primes_greater_than_10_l1027_102745


namespace NUMINAMATH_CALUDE_unique_x_value_l1027_102726

theorem unique_x_value : ∃! x : ℤ, 
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  -1 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_x_value_l1027_102726


namespace NUMINAMATH_CALUDE_union_equals_reals_l1027_102776

-- Define the sets E and F
def E : Set ℝ := {x | x^2 - 5*x - 6 > 0}
def F (a : ℝ) : Set ℝ := {x | x - 5 < a}

-- State the theorem
theorem union_equals_reals (a : ℝ) (h : (11 : ℝ) ∈ F a) : E ∪ F a = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_union_equals_reals_l1027_102776


namespace NUMINAMATH_CALUDE_enrollment_increase_l1027_102758

theorem enrollment_increase (E : ℝ) (E_1992 : ℝ) (E_1993 : ℝ)
  (h1 : E_1993 = 1.26 * E)
  (h2 : E_1993 = 1.05 * E_1992) :
  (E_1992 - E) / E * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_increase_l1027_102758


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1027_102796

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^2022 + y^2 = 2*y + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1027_102796


namespace NUMINAMATH_CALUDE_infinitely_many_a_for_perfect_cube_l1027_102730

theorem infinitely_many_a_for_perfect_cube (n : ℕ) : 
  ∃ (f : ℕ → ℤ), Function.Injective f ∧ ∀ (k : ℕ), ∃ (m : ℕ), (n^6 + 3 * (f k) : ℤ) = m^3 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_a_for_perfect_cube_l1027_102730


namespace NUMINAMATH_CALUDE_integral_x_exp_x_squared_l1027_102741

theorem integral_x_exp_x_squared (x : ℝ) :
  (deriv (fun x => (1/2) * Real.exp (x^2))) x = x * Real.exp (x^2) := by
  sorry

end NUMINAMATH_CALUDE_integral_x_exp_x_squared_l1027_102741


namespace NUMINAMATH_CALUDE_log_function_fixed_point_l1027_102793

theorem log_function_fixed_point (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a + 1
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_function_fixed_point_l1027_102793


namespace NUMINAMATH_CALUDE_dolls_distribution_count_l1027_102735

def distribute_dolls (n_dolls : ℕ) (n_houses : ℕ) : ℕ :=
  let choose_two := n_dolls.choose 2
  let select_house := n_houses
  let arrange_rest := (n_dolls - 2).factorial
  choose_two * select_house * arrange_rest

theorem dolls_distribution_count :
  distribute_dolls 7 6 = 15120 :=
by sorry

end NUMINAMATH_CALUDE_dolls_distribution_count_l1027_102735


namespace NUMINAMATH_CALUDE_student_tickets_sold_l1027_102753

/-- Proves the number of student tickets sold given ticket prices and total sales information -/
theorem student_tickets_sold
  (adult_price : ℝ)
  (student_price : ℝ)
  (total_tickets : ℕ)
  (total_amount : ℝ)
  (h1 : adult_price = 4)
  (h2 : student_price = 2.5)
  (h3 : total_tickets = 59)
  (h4 : total_amount = 222.5) :
  ∃ (student_tickets : ℕ),
    student_tickets = 9 ∧
    (total_tickets - student_tickets) * adult_price + student_tickets * student_price = total_amount :=
by
  sorry

#check student_tickets_sold

end NUMINAMATH_CALUDE_student_tickets_sold_l1027_102753


namespace NUMINAMATH_CALUDE_apple_distribution_l1027_102706

theorem apple_distribution (x : ℕ) (h : x > 0) :
  (1430 / x : ℚ) - (1430 / (x + 45) : ℚ) = 9 → 1430 / x = 22 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1027_102706


namespace NUMINAMATH_CALUDE_binary_sum_equals_318_l1027_102737

/-- Convert a binary number represented as a string to its decimal equivalent -/
def binary_to_decimal (s : String) : ℕ :=
  s.foldl (fun acc c => 2 * acc + c.toString.toNat!) 0

/-- The sum of 11111111₂ and 111111₂ in base 10 -/
theorem binary_sum_equals_318 :
  binary_to_decimal "11111111" + binary_to_decimal "111111" = 318 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_318_l1027_102737


namespace NUMINAMATH_CALUDE_distance_circle_C_to_line_l_l1027_102794

/-- Circle C with center (1, 0) and radius 1 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}

/-- Line l with equation x + y + 2√2 - 1 = 0 -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 2 * Real.sqrt 2 - 1 = 0}

/-- Center of circle C -/
def center_C : ℝ × ℝ := (1, 0)

/-- Distance from a point to a line -/
def point_to_line_distance (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem distance_circle_C_to_line_l :
  point_to_line_distance center_C line_l = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_circle_C_to_line_l_l1027_102794


namespace NUMINAMATH_CALUDE_product_selection_theorem_l1027_102716

/-- Represents the outcome of selecting two items from a batch of products -/
inductive Outcome
  | TwoGenuine
  | OneGenuineOneDefective
  | TwoDefective

/-- Represents a batch of products -/
structure Batch where
  genuine : ℕ
  defective : ℕ
  h_genuine : genuine > 2
  h_defective : defective > 2

/-- Probability of an outcome given a batch -/
def prob (b : Batch) (o : Outcome) : ℝ := sorry

/-- Event of having exactly one defective product -/
def exactly_one_defective (o : Outcome) : Prop :=
  o = Outcome.OneGenuineOneDefective

/-- Event of having exactly two defective products -/
def exactly_two_defective (o : Outcome) : Prop :=
  o = Outcome.TwoDefective

/-- Event of having at least one defective product -/
def at_least_one_defective (o : Outcome) : Prop :=
  o = Outcome.OneGenuineOneDefective ∨ o = Outcome.TwoDefective

/-- Event of having all genuine products -/
def all_genuine (o : Outcome) : Prop :=
  o = Outcome.TwoGenuine

theorem product_selection_theorem (b : Batch) :
  -- Statement ②: Exactly one defective and exactly two defective are mutually exclusive
  (∀ o : Outcome, ¬(exactly_one_defective o ∧ exactly_two_defective o)) ∧
  -- Statement ④: At least one defective and all genuine are mutually exclusive and complementary
  (∀ o : Outcome, ¬(at_least_one_defective o ∧ all_genuine o)) ∧
  (∀ o : Outcome, at_least_one_defective o ∨ all_genuine o) ∧
  -- Statements ① and ③ are incorrect (we don't need to prove them, just state that they're not included)
  True := by sorry

end NUMINAMATH_CALUDE_product_selection_theorem_l1027_102716


namespace NUMINAMATH_CALUDE_field_purchase_problem_l1027_102795

theorem field_purchase_problem :
  let good_field_value : ℚ := 300  -- value of 1 acre of good field
  let bad_field_value : ℚ := 500 / 7  -- value of 1 acre of bad field
  let total_area : ℚ := 100  -- total area in acres
  let total_cost : ℚ := 10000  -- total cost in coins
  let good_field_acres : ℚ := 25 / 2  -- solution for good field acres
  let bad_field_acres : ℚ := 175 / 2  -- solution for bad field acres
  (good_field_acres + bad_field_acres = total_area) ∧
  (good_field_value * good_field_acres + bad_field_value * bad_field_acres = total_cost) :=
by sorry


end NUMINAMATH_CALUDE_field_purchase_problem_l1027_102795


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l1027_102712

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 4/y ≥ 1/a + 4/b) → 1/a + 4/b = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l1027_102712


namespace NUMINAMATH_CALUDE_original_light_wattage_l1027_102766

theorem original_light_wattage (W : ℝ) : 
  (W + 0.3 * W = 143) → W = 110 := by
  sorry

end NUMINAMATH_CALUDE_original_light_wattage_l1027_102766


namespace NUMINAMATH_CALUDE_train_length_l1027_102750

/-- The length of a train given its speed and time to pass a point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (h1 : speed_kmh = 63) (h2 : time_s = 16) :
  speed_kmh * 1000 / 3600 * time_s = 280 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1027_102750


namespace NUMINAMATH_CALUDE_boris_clock_theorem_l1027_102705

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_valid_time (h m : ℕ) : Prop :=
  h ≤ 23 ∧ m ≤ 59

def satisfies_clock_conditions (h m : ℕ) : Prop :=
  is_valid_time h m ∧ digit_sum h + digit_sum m = 6 ∧ h + m = 15

def possible_times : Set (ℕ × ℕ) :=
  {(0,15), (1,14), (2,13), (3,12), (4,11), (5,10), (10,5), (11,4), (12,3), (13,2), (14,1), (15,0)}

theorem boris_clock_theorem :
  {(h, m) | satisfies_clock_conditions h m} = possible_times :=
sorry

end NUMINAMATH_CALUDE_boris_clock_theorem_l1027_102705


namespace NUMINAMATH_CALUDE_vector_collinearity_l1027_102746

/-- Given vectors a, b, and c in ℝ², prove that if b - a is collinear with c, then the n-coordinate of b equals -3. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (n : ℝ) :
  a = (1, 2) →
  b = (n, 3) →
  c = (4, -1) →
  ∃ (k : ℝ), (b.1 - a.1, b.2 - a.2) = (k * c.1, k * c.2) →
  n = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1027_102746


namespace NUMINAMATH_CALUDE_increasing_quadratic_function_condition_l1027_102759

/-- A function f(x) = x^2 - 2ax is increasing on [1, +∞) if and only if a ≤ 1 -/
theorem increasing_quadratic_function_condition (a : ℝ) : 
  (∀ x ≥ 1, Monotone (fun x => x^2 - 2*a*x)) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_quadratic_function_condition_l1027_102759


namespace NUMINAMATH_CALUDE_pizza_theorem_l1027_102755

def pizza_problem (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ) : Prop :=
  ∃ (n a b c : ℕ),
    -- Total slices
    total_slices = 24 ∧
    -- Slices with each topping
    pepperoni_slices = 12 ∧
    mushroom_slices = 14 ∧
    olive_slices = 16 ∧
    -- Every slice has at least one topping
    (12 - n) + (14 - n) + (16 - n) + a + b + c + n = total_slices ∧
    -- Venn diagram constraint
    42 - 3*n - 2*(a + b + c) + a + b + c + n = total_slices ∧
    -- Number of slices with all three toppings
    n = 2

theorem pizza_theorem :
  ∀ (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ),
    pizza_problem total_slices pepperoni_slices mushroom_slices olive_slices :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l1027_102755


namespace NUMINAMATH_CALUDE_exam_comparison_l1027_102740

theorem exam_comparison (total_items : ℕ) (lyssa_incorrect_percent : ℚ) (precious_mistakes : ℕ)
  (h1 : total_items = 120)
  (h2 : lyssa_incorrect_percent = 25 / 100)
  (h3 : precious_mistakes = 17) :
  (total_items - (lyssa_incorrect_percent * total_items).num) - (total_items - precious_mistakes) = -13 :=
by sorry

end NUMINAMATH_CALUDE_exam_comparison_l1027_102740


namespace NUMINAMATH_CALUDE_fortieth_day_from_tuesday_is_sunday_l1027_102739

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

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem fortieth_day_from_tuesday_is_sunday :
  advanceDay DayOfWeek.Tuesday 40 = DayOfWeek.Sunday := by
  sorry


end NUMINAMATH_CALUDE_fortieth_day_from_tuesday_is_sunday_l1027_102739


namespace NUMINAMATH_CALUDE_original_price_is_10000_l1027_102710

/-- Calculates the original price of a machine given repair cost, transportation cost, profit percentage, and selling price. -/
def calculate_original_price (repair_cost : ℕ) (transport_cost : ℕ) (profit_percent : ℕ) (selling_price : ℕ) : ℕ :=
  let total_additional_cost := repair_cost + transport_cost
  let total_cost_multiplier := 100 + profit_percent
  ((selling_price * 100) / total_cost_multiplier) - total_additional_cost

/-- Theorem stating that given the specific conditions, the original price of the machine was 10000. -/
theorem original_price_is_10000 :
  calculate_original_price 5000 1000 50 24000 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_original_price_is_10000_l1027_102710


namespace NUMINAMATH_CALUDE_radius_difference_is_zero_l1027_102703

/-- A circle with center C tangent to positive x and y-axes and externally tangent to another circle -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = radius ∧ center.2 = radius
  externally_tangent : (radius - 2)^2 + radius^2 = (radius + 2)^2

/-- The radius difference between the largest and smallest possible radii is 0 -/
theorem radius_difference_is_zero : 
  ∀ (c₁ c₂ : TangentCircle), c₁.radius - c₂.radius = 0 := by
  sorry

end NUMINAMATH_CALUDE_radius_difference_is_zero_l1027_102703


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l1027_102787

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  2 * log10 2 + log10 5 / log10 (Real.sqrt 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l1027_102787


namespace NUMINAMATH_CALUDE_emails_left_theorem_l1027_102788

/-- Given an initial number of emails, calculate the number of emails left in the inbox
    after moving half to trash and 40% of the remainder to a work folder. -/
def emails_left_in_inbox (initial_emails : ℕ) : ℕ :=
  let after_trash := initial_emails / 2
  let to_work_folder := (after_trash * 40) / 100
  after_trash - to_work_folder

/-- Theorem stating that given 400 initial emails, 120 emails are left in the inbox
    after moving half to trash and 40% of the remainder to a work folder. -/
theorem emails_left_theorem : emails_left_in_inbox 400 = 120 := by
  sorry

#eval emails_left_in_inbox 400

end NUMINAMATH_CALUDE_emails_left_theorem_l1027_102788


namespace NUMINAMATH_CALUDE_lesser_fraction_l1027_102713

theorem lesser_fraction (x y : ℚ) : 
  x + y = 8/9 → x * y = 1/8 → min x y = 7/40 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1027_102713


namespace NUMINAMATH_CALUDE_frequency_distribution_best_for_proportions_l1027_102778

-- Define the possible statistical measures
inductive StatisticalMeasure
  | Average
  | Variance
  | Mode
  | FrequencyDistribution

-- Define a function to determine if a measure can calculate proportions within ranges
def canCalculateProportionsInRange (measure : StatisticalMeasure) : Prop :=
  match measure with
  | StatisticalMeasure.FrequencyDistribution => True
  | _ => False

-- Theorem statement
theorem frequency_distribution_best_for_proportions :
  ∀ (measure : StatisticalMeasure),
    canCalculateProportionsInRange measure →
    measure = StatisticalMeasure.FrequencyDistribution :=
by sorry

end NUMINAMATH_CALUDE_frequency_distribution_best_for_proportions_l1027_102778


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l1027_102708

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 12 := by sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l1027_102708


namespace NUMINAMATH_CALUDE_f_g_inequality_l1027_102789

open Set
open Function
open Topology

-- Define the interval [a, b]
variable (a b : ℝ) (hab : a < b)

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
variable (hf : DifferentiableOn ℝ f (Icc a b))
variable (hg : DifferentiableOn ℝ g (Icc a b))
variable (h_deriv : ∀ x ∈ Icc a b, deriv f x > deriv g x)

-- State the theorem
theorem f_g_inequality (x : ℝ) (hx : x ∈ Ioo a b) :
  f x + g a > g x + f a := by sorry

end NUMINAMATH_CALUDE_f_g_inequality_l1027_102789


namespace NUMINAMATH_CALUDE_perfect_squares_between_a_and_2a_l1027_102720

theorem perfect_squares_between_a_and_2a (a : ℕ) : 
  (a > 0) → 
  (∃ x : ℕ, x^2 > a ∧ (x+9)^2 < 2*a ∧ 
    ∀ y : ℕ, (y^2 > a ∧ y^2 < 2*a) → (x ≤ y ∧ y ≤ x+9)) →
  (481 ≤ a ∧ a ≤ 684) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_between_a_and_2a_l1027_102720


namespace NUMINAMATH_CALUDE_total_beignets_l1027_102772

/-- The number of beignets eaten per day -/
def beignets_per_day : ℕ := 3

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weeks we're considering -/
def weeks : ℕ := 16

/-- Theorem: The total number of beignets eaten in 16 weeks -/
theorem total_beignets : beignets_per_day * days_in_week * weeks = 336 := by
  sorry

end NUMINAMATH_CALUDE_total_beignets_l1027_102772


namespace NUMINAMATH_CALUDE_third_chest_coin_difference_l1027_102762

theorem third_chest_coin_difference (total_gold total_silver : ℕ) 
  (x1 y1 x2 y2 x3 y3 : ℕ) : 
  total_gold = 40 →
  total_silver = 40 →
  x1 + x2 + x3 = total_gold →
  y1 + y2 + y3 = total_silver →
  x1 = y1 + 7 →
  y2 = x2 - 15 →
  y3 - x3 = 22 :=
by sorry

end NUMINAMATH_CALUDE_third_chest_coin_difference_l1027_102762
