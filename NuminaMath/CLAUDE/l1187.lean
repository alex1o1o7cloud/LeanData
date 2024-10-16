import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l1187_118768

theorem equation_solution : ∃ (z : ℂ), 
  (z - 4)^6 + (z - 6)^6 = 32 ∧ 
  (z = 5 + Complex.I * Real.sqrt 3 ∨ z = 5 - Complex.I * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1187_118768


namespace NUMINAMATH_CALUDE_min_value_sqrt_expression_l1187_118759

theorem min_value_sqrt_expression (x : ℝ) (hx : x > 0) :
  2 * Real.sqrt x + 3 / Real.sqrt x ≥ 2 * Real.sqrt 6 ∧
  (2 * Real.sqrt x + 3 / Real.sqrt x = 2 * Real.sqrt 6 ↔ x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_expression_l1187_118759


namespace NUMINAMATH_CALUDE_set_union_problem_l1187_118782

theorem set_union_problem (M N : Set ℕ) (x : ℕ) : 
  M = {0, x} → N = {1, 2} → M ∩ N = {2} → M ∪ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l1187_118782


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l1187_118790

theorem baker_cakes_sold (initial_cakes bought_cakes : ℕ) 
  (h : initial_cakes = 8)
  (h' : bought_cakes = 139) :
  initial_cakes + bought_cakes + 6 = 145 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l1187_118790


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l1187_118776

theorem dining_bill_calculation (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (food_price : ℝ) : 
  total = 158.40 ∧ 
  tax_rate = 0.10 ∧ 
  tip_rate = 0.20 ∧
  total = food_price * (1 + tax_rate) * (1 + tip_rate) →
  food_price = 120 := by
sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l1187_118776


namespace NUMINAMATH_CALUDE_simplify_fraction_l1187_118741

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1187_118741


namespace NUMINAMATH_CALUDE_sallys_nickels_l1187_118770

/-- The number of nickels Sally has after receiving some from her parents -/
def total_nickels (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem: Sally's total nickels equals the sum of her initial nickels and those received from parents -/
theorem sallys_nickels (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) :
  total_nickels initial from_dad from_mom = initial + from_dad + from_mom := by
  sorry

end NUMINAMATH_CALUDE_sallys_nickels_l1187_118770


namespace NUMINAMATH_CALUDE_cube_root_problem_l1187_118756

theorem cube_root_problem (a m : ℝ) (h1 : a > 0) 
  (h2 : (m + 7)^2 = a) (h3 : (2*m - 1)^2 = a) : 
  (a - m)^(1/3 : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_root_problem_l1187_118756


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1187_118726

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24 ∧ a.val + b.val ≤ x.val + y.val ∧ a.val + b.val = 100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1187_118726


namespace NUMINAMATH_CALUDE_cos_plus_sin_range_l1187_118781

/-- 
Given a point P(x,1) where x ≥ 1 on the terminal side of angle θ in the Cartesian coordinate system,
the sum of cosine and sine of θ is strictly greater than 1 and less than or equal to √2.
-/
theorem cos_plus_sin_range (x : ℝ) (θ : ℝ) (h1 : x ≥ 1) 
  (h2 : x = Real.cos θ * Real.sqrt (x^2 + 1)) 
  (h3 : 1 = Real.sin θ * Real.sqrt (x^2 + 1)) : 
  1 < Real.cos θ + Real.sin θ ∧ Real.cos θ + Real.sin θ ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cos_plus_sin_range_l1187_118781


namespace NUMINAMATH_CALUDE_unit_circle_point_coordinate_l1187_118702

/-- Theorem: For a point P(x₀, y₀) on the unit circle in the xy-plane, where ∠xOP = α, 
α ∈ (π/4, 3π/4), and cos(α + π/4) = -12/13, the value of x₀ is equal to -7√2/26. -/
theorem unit_circle_point_coordinate (x₀ y₀ α : Real) : 
  x₀^2 + y₀^2 = 1 → -- Point P lies on the unit circle
  x₀ = Real.cos α → -- Definition of cosine
  y₀ = Real.sin α → -- Definition of sine
  π/4 < α → α < 3*π/4 → -- α ∈ (π/4, 3π/4)
  Real.cos (α + π/4) = -12/13 → -- Given condition
  x₀ = -7 * Real.sqrt 2 / 26 := by
sorry

end NUMINAMATH_CALUDE_unit_circle_point_coordinate_l1187_118702


namespace NUMINAMATH_CALUDE_wedge_volume_l1187_118783

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (θ : ℝ) (h : ℝ) : 
  d = 16 → θ = 30 → h = d → (π * d^2 * h) / 8 = 512 * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l1187_118783


namespace NUMINAMATH_CALUDE_all_digits_are_perfect_cube_units_l1187_118706

-- Define the set of possible units digits of perfect cubes modulo 10
def PerfectCubeUnitsDigits : Set ℕ :=
  {d | ∃ n : ℤ, (n^3 : ℤ) % 10 = d}

-- Theorem statement
theorem all_digits_are_perfect_cube_units : PerfectCubeUnitsDigits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

end NUMINAMATH_CALUDE_all_digits_are_perfect_cube_units_l1187_118706


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_power_plus_a_l1187_118740

theorem infinite_primes_dividing_power_plus_a (a : ℕ) (ha : a > 0) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ 2^(2^n) + a} :=
by sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_power_plus_a_l1187_118740


namespace NUMINAMATH_CALUDE_total_fish_count_l1187_118785

/-- The number of fish tanks James has -/
def num_tanks : ℕ := 3

/-- The number of fish in the first tank -/
def fish_in_first_tank : ℕ := 20

/-- The number of fish in each of the other tanks -/
def fish_in_other_tanks : ℕ := 2 * fish_in_first_tank

/-- The total number of fish in all tanks -/
def total_fish : ℕ := fish_in_first_tank + 2 * fish_in_other_tanks

theorem total_fish_count : total_fish = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l1187_118785


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l1187_118799

def total_shoes : ℕ := 10
def red_shoes : ℕ := 4
def green_shoes : ℕ := 6
def drawn_shoes : ℕ := 2

theorem probability_two_red_shoes :
  (Nat.choose red_shoes drawn_shoes : ℚ) / (Nat.choose total_shoes drawn_shoes) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l1187_118799


namespace NUMINAMATH_CALUDE_josh_final_wallet_amount_l1187_118750

-- Define the initial conditions
def initial_wallet_amount : ℝ := 300
def initial_investment : ℝ := 2000
def stock_price_increase : ℝ := 0.30

-- Define the function to calculate the final amount
def final_wallet_amount : ℝ :=
  initial_wallet_amount + initial_investment * (1 + stock_price_increase)

-- Theorem to prove
theorem josh_final_wallet_amount :
  final_wallet_amount = 2900 := by
  sorry

end NUMINAMATH_CALUDE_josh_final_wallet_amount_l1187_118750


namespace NUMINAMATH_CALUDE_decreasing_quadratic_l1187_118708

theorem decreasing_quadratic (m : ℝ) : 
  (∀ x₁ x₂ : ℤ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → 
    x₁^2 + (m-1)*x₁ + 1 > x₂^2 + (m-1)*x₂ + 1) ↔ 
  m ≤ -8 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_l1187_118708


namespace NUMINAMATH_CALUDE_pizza_sales_total_l1187_118751

theorem pizza_sales_total (pepperoni bacon cheese : ℕ) 
  (h1 : pepperoni = 2) 
  (h2 : bacon = 6) 
  (h3 : cheese = 6) : 
  pepperoni + bacon + cheese = 14 := by
  sorry

end NUMINAMATH_CALUDE_pizza_sales_total_l1187_118751


namespace NUMINAMATH_CALUDE_estimate_value_l1187_118786

theorem estimate_value : 5 < (3 * Real.sqrt 15 - Real.sqrt 3) * Real.sqrt (1/3) ∧
                         (3 * Real.sqrt 15 - Real.sqrt 3) * Real.sqrt (1/3) < 6 := by
  sorry

end NUMINAMATH_CALUDE_estimate_value_l1187_118786


namespace NUMINAMATH_CALUDE_trig_ratios_for_point_on_terminal_side_l1187_118704

/-- Given a point P(3m, -2m) where m < 0 lying on the terminal side of angle α,
    prove the trigonometric ratios for α. -/
theorem trig_ratios_for_point_on_terminal_side (m : ℝ) (α : ℝ) 
  (h1 : m < 0) 
  (h2 : ∃ (r : ℝ), r > 0 ∧ 3 * m = r * Real.cos α ∧ -2 * m = r * Real.sin α) :
  Real.sin α = 2 * Real.sqrt 13 / 13 ∧ 
  Real.cos α = -3 * Real.sqrt 13 / 13 ∧ 
  Real.tan α = -2 / 3 := by
sorry


end NUMINAMATH_CALUDE_trig_ratios_for_point_on_terminal_side_l1187_118704


namespace NUMINAMATH_CALUDE_closest_to_sqrt_difference_l1187_118787

theorem closest_to_sqrt_difference : 
  let diff := Real.sqrt 145 - Real.sqrt 141
  ∀ x ∈ ({0.19, 0.20, 0.21, 0.22} : Set ℝ), 
    |diff - 0.18| < |diff - x| := by
  sorry

end NUMINAMATH_CALUDE_closest_to_sqrt_difference_l1187_118787


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l1187_118707

theorem tangent_line_perpendicular (a : ℝ) : 
  let f : ℝ → ℝ := λ x => 2 * Real.sin x - 2 * Real.cos x
  let point : ℝ × ℝ := (Real.pi / 2, 2)
  let tangent_slope : ℝ := (2 * Real.cos (Real.pi / 2) + 2 * Real.sin (Real.pi / 2))
  let perpendicular_line : ℝ → ℝ := λ y => 1 / a * y + 1 / a
  (tangent_slope * (1 / a) = -1) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l1187_118707


namespace NUMINAMATH_CALUDE_green_tea_leaves_needed_l1187_118718

/-- The number of sprigs of mint added to each batch of mud -/
def sprigs_of_mint : ℕ := 3

/-- The number of green tea leaves added per sprig of mint -/
def leaves_per_sprig : ℕ := 2

/-- The factor by which the efficacy of ingredients is reduced in the new mud -/
def efficacy_reduction : ℚ := 1/2

/-- The number of green tea leaves needed for the new batch of mud to maintain the same efficacy -/
def new_leaves_needed : ℕ := 12

/-- Theorem stating that the number of green tea leaves needed for the new batch of mud
    to maintain the same efficacy is equal to 12 -/
theorem green_tea_leaves_needed :
  (sprigs_of_mint * leaves_per_sprig : ℚ) / efficacy_reduction = new_leaves_needed := by
  sorry

end NUMINAMATH_CALUDE_green_tea_leaves_needed_l1187_118718


namespace NUMINAMATH_CALUDE_unique_solution_system_l1187_118797

theorem unique_solution_system (x y : ℝ) : 
  (x + 2*y = 4 ∧ 2*x - y = 3) ↔ (x = 2 ∧ y = 1) := by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1187_118797


namespace NUMINAMATH_CALUDE_special_multiples_count_l1187_118769

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def count_special_multiples (n : ℕ) : ℕ :=
  count_multiples n 3 + count_multiples n 4 - count_multiples n 6

theorem special_multiples_count :
  count_special_multiples 3000 = 1250 := by sorry

end NUMINAMATH_CALUDE_special_multiples_count_l1187_118769


namespace NUMINAMATH_CALUDE_melanie_plums_count_l1187_118723

/-- The number of plums Melanie picked initially -/
def melanie_picked : ℝ := 7.0

/-- The number of plums Sam gave to Melanie -/
def sam_gave : ℝ := 3.0

/-- The total number of plums Melanie has now -/
def total_plums : ℝ := melanie_picked + sam_gave

theorem melanie_plums_count : total_plums = 10.0 := by
  sorry

end NUMINAMATH_CALUDE_melanie_plums_count_l1187_118723


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1187_118709

-- Problem 1
theorem problem_1 : 12 - (-1) + (-7) = 6 := by sorry

-- Problem 2
theorem problem_2 : -3.5 * (-3/4) / (7/8) = 3 := by sorry

-- Problem 3
theorem problem_3 : (1/3 - 1/6 - 1/12) * (-12) = -1 := by sorry

-- Problem 4
theorem problem_4 : (-2)^4 / (-4) * (-1/2)^2 - 1^2 = -2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1187_118709


namespace NUMINAMATH_CALUDE_mango_seller_profit_l1187_118784

/-- Proves that given a fruit seller who loses 15% when selling mangoes at Rs. 6 per kg,
    if they want to sell at Rs. 7.411764705882353 per kg, their desired profit percentage is 5%. -/
theorem mango_seller_profit (loss_price : ℝ) (loss_percentage : ℝ) (desired_price : ℝ) :
  loss_price = 6 →
  loss_percentage = 15 →
  desired_price = 7.411764705882353 →
  let cost_price := loss_price / (1 - loss_percentage / 100)
  let profit_percentage := (desired_price / cost_price - 1) * 100
  profit_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_mango_seller_profit_l1187_118784


namespace NUMINAMATH_CALUDE_factorization_x4_minus_16y4_l1187_118732

theorem factorization_x4_minus_16y4 (x y : ℚ) :
  x^4 - 16*y^4 = (x^2 + 4*y^2) * (x + 2*y) * (x - 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_16y4_l1187_118732


namespace NUMINAMATH_CALUDE_vector_OA_coordinates_l1187_118754

/-- Given that O is the origin, point A is in the second quadrant,
    |OA| = 2, and ∠xOA = 150°, prove that the coordinates of vector OA are (-√3, 1). -/
theorem vector_OA_coordinates (A : ℝ × ℝ) :
  A.1 < 0 ∧ A.2 > 0 →  -- A is in the second quadrant
  A.1^2 + A.2^2 = 4 →  -- |OA| = 2
  Real.cos (150 * π / 180) = A.1 / 2 ∧ Real.sin (150 * π / 180) = A.2 / 2 →  -- ∠xOA = 150°
  A = (-Real.sqrt 3, 1) :=
by sorry

end NUMINAMATH_CALUDE_vector_OA_coordinates_l1187_118754


namespace NUMINAMATH_CALUDE_pentagon_obtuse_angles_dodecagon_diagonals_four_sided_angle_sum_equality_l1187_118701

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of obtuse angles in a polygon -/
def numObtuseAngles (p : Polygon n) : ℕ := sorry

/-- The number of diagonals in a polygon -/
def numDiagonals (p : Polygon n) : ℕ := sorry

/-- The sum of interior angles of a polygon -/
def sumInteriorAngles (p : Polygon n) : ℝ := sorry

/-- The sum of exterior angles of a polygon -/
def sumExteriorAngles (p : Polygon n) : ℝ := sorry

theorem pentagon_obtuse_angles :
  ∀ p : Polygon 5, numObtuseAngles p ≥ 2 := by sorry

theorem dodecagon_diagonals :
  ∀ p : Polygon 12, numDiagonals p = 54 := by sorry

theorem four_sided_angle_sum_equality :
  ∀ n : ℕ, ∀ p : Polygon n,
    (sumInteriorAngles p = sumExteriorAngles p) ↔ n = 4 := by sorry

end NUMINAMATH_CALUDE_pentagon_obtuse_angles_dodecagon_diagonals_four_sided_angle_sum_equality_l1187_118701


namespace NUMINAMATH_CALUDE_opposite_face_of_ten_l1187_118739

/-- Represents a cube with six faces labeled with distinct integers -/
structure Cube where
  faces : Finset ℕ
  distinct : faces.card = 6
  range : ∀ n ∈ faces, 6 ≤ n ∧ n ≤ 11

/-- The sum of all numbers on the cube's faces -/
def Cube.total_sum (c : Cube) : ℕ := c.faces.sum id

/-- Represents a roll of the cube, showing four lateral faces -/
structure Roll (c : Cube) where
  lateral_sum : ℕ
  valid : lateral_sum = c.total_sum - (c.faces.sum id - lateral_sum)

theorem opposite_face_of_ten (c : Cube) 
  (roll1 : Roll c) (roll2 : Roll c)
  (h1 : roll1.lateral_sum = 36)
  (h2 : roll2.lateral_sum = 33)
  : ∃ n ∈ c.faces, n = 8 ∧ (c.faces.sum id - (10 + n) = roll1.lateral_sum ∨ 
                            c.faces.sum id - (10 + n) = roll2.lateral_sum) :=
sorry

end NUMINAMATH_CALUDE_opposite_face_of_ten_l1187_118739


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1187_118792

theorem units_digit_of_expression : 2^2023 * 5^2024 * 11^2025 % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1187_118792


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l1187_118755

theorem factorization_of_cubic (x : ℝ) : 4 * x^3 - x = x * (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l1187_118755


namespace NUMINAMATH_CALUDE_coordinates_of_P_fixed_points_of_N_min_length_AB_l1187_118745

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define point P on line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define tangent PA
def tangent_PA (p : Point_P) (a : ℝ × ℝ) : Prop :=
  circle_M a.1 a.2 ∧ 
  (p.x - a.1)^2 + (p.y - a.2)^2 = ((0 - p.x)^2 + (4 - p.y)^2)

-- Theorem 1
theorem coordinates_of_P : 
  ∃ (p : Point_P), (∃ (a : ℝ × ℝ), tangent_PA p a ∧ (p.x - a.1)^2 + (p.y - a.2)^2 = 12) →
  (p.x = 0 ∧ p.y = 0) ∨ (p.x = 16/5 ∧ p.y = 8/5) :=
sorry

-- Define circle N (circumcircle of triangle PAM)
def circle_N (p : Point_P) (x y : ℝ) : Prop :=
  (2*x + y - 4) * p.y - (x^2 + y^2 - 4*y) = 0

-- Theorem 2
theorem fixed_points_of_N :
  ∀ (p : Point_P), 
    (circle_N p 0 4 ∧ circle_N p (8/5) (4/5)) ∧
    (∀ (x y : ℝ), circle_N p x y → (x = 0 ∧ y = 4) ∨ (x = 8/5 ∧ y = 4/5)) :=
sorry

-- Define chord AB
def chord_AB (p : Point_P) (x y : ℝ) : Prop :=
  2 * p.y * x + (p.y - 4) * y + 12 - 4 * p.y = 0

-- Theorem 3
theorem min_length_AB :
  ∃ (p : Point_P), 
    (∀ (p' : Point_P), 
      (∀ (x y : ℝ), chord_AB p x y → 
        (x - 0)^2 + (y - 4)^2 ≥ (x - 0)^2 + (y - 4)^2)) ∧
    (∃ (a b : ℝ × ℝ), chord_AB p a.1 a.2 ∧ chord_AB p b.1 b.2 ∧ 
      (a.1 - b.1)^2 + (a.2 - b.2)^2 = 11) :=
sorry

end NUMINAMATH_CALUDE_coordinates_of_P_fixed_points_of_N_min_length_AB_l1187_118745


namespace NUMINAMATH_CALUDE_ellipse_properties_l1187_118729

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- Define the line that intersects the ellipse
def intersecting_line (x y : ℝ) : Prop := y = x + 2

-- Define the condition for a point to be on the circle with AB as diameter
def on_circle_AB (x y : ℝ) : Prop := 
  ∃ (x1 y1 x2 y2 : ℝ), 
    ellipse_C x1 y1 ∧ 
    ellipse_C x2 y2 ∧ 
    intersecting_line x1 y1 ∧ 
    intersecting_line x2 y2 ∧ 
    x * (x1 + x2) + y * (y1 + y2) = (x1^2 + y1^2 + x2^2 + y2^2) / 2

theorem ellipse_properties :
  (∀ x y, ellipse_C x y ↔ x^2/9 + y^2 = 1) ∧ 
  ¬(on_circle_AB 0 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1187_118729


namespace NUMINAMATH_CALUDE_books_division_l1187_118727

theorem books_division (total_books : ℕ) (divisions : ℕ) (final_category_size : ℕ) : 
  total_books = 400 → divisions = 4 → final_category_size = total_books / (2^divisions) → 
  final_category_size = 25 := by
sorry

end NUMINAMATH_CALUDE_books_division_l1187_118727


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l1187_118714

/-- Given an article with a marked price and cost price, calculate the discount percentage. -/
theorem discount_percentage_calculation
  (marked_price : ℝ)
  (cost_price : ℝ)
  (h1 : cost_price = 0.64 * marked_price)
  (h2 : (cost_price * 1.375 - cost_price) / cost_price = 0.375) :
  (marked_price - cost_price * 1.375) / marked_price = 0.12 := by
  sorry

#check discount_percentage_calculation

end NUMINAMATH_CALUDE_discount_percentage_calculation_l1187_118714


namespace NUMINAMATH_CALUDE_negation_of_conjunction_l1187_118736

theorem negation_of_conjunction (p q : Prop) : ¬(p ∧ q) ↔ (¬p ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_negation_of_conjunction_l1187_118736


namespace NUMINAMATH_CALUDE_quadratic_inequality_bc_value_l1187_118796

theorem quadratic_inequality_bc_value 
  (b c : ℝ) 
  (h : ∀ x : ℝ, x^2 + b*x + c < 0 ↔ 2 < x ∧ x < 4) : 
  b * c = -48 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_bc_value_l1187_118796


namespace NUMINAMATH_CALUDE_factorial_prime_factors_l1187_118793

theorem factorial_prime_factors (x i k m p : ℕ) : 
  x = (Finset.range 8).prod (λ n => n + 1) →
  x = 2^i * 3^k * 5^m * 7^p →
  i > 0 ∧ k > 0 ∧ m > 0 ∧ p > 0 →
  i + k + m + p = 11 := by
sorry

end NUMINAMATH_CALUDE_factorial_prime_factors_l1187_118793


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1187_118737

/-- The minimum distance from a point on y = e^x + x to the line 2x-y-3=0 -/
theorem min_distance_to_line :
  let f : ℝ → ℝ := fun x ↦ Real.exp x + x
  let P : ℝ × ℝ := (0, f 0)
  let d (x y : ℝ) : ℝ := |2*x - y - 3| / Real.sqrt (2^2 + (-1)^2)
  ∀ x : ℝ, d x (f x) ≥ d P.1 P.2 ∧ d P.1 P.2 = 4 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l1187_118737


namespace NUMINAMATH_CALUDE_power_division_rule_l1187_118712

theorem power_division_rule (a : ℝ) : a^7 / a^3 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1187_118712


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1187_118760

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to (a + mb), then m = 5. -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (2, -1))
  (h2 : b = (1, 3))
  (h3 : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) :
  m = 5 := by
  sorry

#check perpendicular_vector_scalar

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1187_118760


namespace NUMINAMATH_CALUDE_marble_distribution_l1187_118765

theorem marble_distribution (total_marbles : ℕ) (people : ℕ) : 
  total_marbles = 180 →
  (total_marbles / people : ℚ) - (total_marbles / (people + 2) : ℚ) = 1 →
  people = 18 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l1187_118765


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1187_118744

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i^2 = -1 →
  let z := (3 - 2 * i^2) / (1 + i)
  Complex.im z = -5/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1187_118744


namespace NUMINAMATH_CALUDE_problem_solution_l1187_118742

theorem problem_solution (a b : ℚ) 
  (eq1 : 4 + 2*a = 5 - b) 
  (eq2 : 5 + b = 9 + 3*a) : 
  4 - 2*a = 26/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1187_118742


namespace NUMINAMATH_CALUDE_relationship_abc_l1187_118761

theorem relationship_abc (a b c : ℝ) : 
  a = (1.01 : ℝ) ^ (0.5 : ℝ) →
  b = (1.01 : ℝ) ^ (0.6 : ℝ) →
  c = (0.6 : ℝ) ^ (0.5 : ℝ) →
  b > a ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1187_118761


namespace NUMINAMATH_CALUDE_melanie_total_dimes_l1187_118752

def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

theorem melanie_total_dimes : 
  initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end NUMINAMATH_CALUDE_melanie_total_dimes_l1187_118752


namespace NUMINAMATH_CALUDE_east_region_difference_l1187_118748

/-- The difference in square miles between the regions east of two plains -/
def region_difference (total_area plain_B_area : ℕ) : ℕ :=
  plain_B_area - (total_area - plain_B_area)

/-- Theorem stating the difference between regions east of plain B and A -/
theorem east_region_difference :
  ∀ (total_area plain_B_area : ℕ),
  total_area = 350 →
  plain_B_area = 200 →
  region_difference total_area plain_B_area = 50 :=
by
  sorry

#eval region_difference 350 200

end NUMINAMATH_CALUDE_east_region_difference_l1187_118748


namespace NUMINAMATH_CALUDE_circular_permutations_count_l1187_118798

/-- The number of elements of type 'a' -/
def num_a : ℕ := 2

/-- The number of elements of type 'b' -/
def num_b : ℕ := 2

/-- The number of elements of type 'c' -/
def num_c : ℕ := 4

/-- The total number of elements -/
def total_elements : ℕ := num_a + num_b + num_c

/-- First-class circular permutations -/
def first_class_permutations : ℕ := 52

/-- Second-class circular permutations -/
def second_class_permutations : ℕ := 33

theorem circular_permutations_count :
  (first_class_permutations = 52) ∧ (second_class_permutations = 33) := by
  sorry

end NUMINAMATH_CALUDE_circular_permutations_count_l1187_118798


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l1187_118749

def coin_flips : ℕ := 12
def heads_count : ℕ := 9

-- Define the probability of getting exactly k heads in n flips of a fair coin
def probability_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

theorem probability_nine_heads_in_twelve_flips :
  probability_k_heads coin_flips heads_count = 55 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l1187_118749


namespace NUMINAMATH_CALUDE_eleven_triangles_arrangement_l1187_118724

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  center : Point
  sideLength : ℝ

/-- Checks if two triangles overlap -/
def trianglesOverlap (t1 t2 : EquilateralTriangle) : Prop :=
  sorry

/-- Checks if a triangle touches another triangle -/
def trianglesTouch (t1 t2 : EquilateralTriangle) : Prop :=
  sorry

/-- The main theorem stating that 11 equilateral triangles can be placed around a central triangle -/
theorem eleven_triangles_arrangement (centralTriangle : EquilateralTriangle) :
  ∃ (tiles : Fin 11 → EquilateralTriangle),
    (∀ i j, i ≠ j → ¬trianglesOverlap (tiles i) (tiles j)) ∧
    (∀ i, trianglesTouch (tiles i) centralTriangle) :=
  sorry

end NUMINAMATH_CALUDE_eleven_triangles_arrangement_l1187_118724


namespace NUMINAMATH_CALUDE_f_composition_equals_8c_implies_c_equals_1_l1187_118763

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 199^x + 1 else x^2 + 2*c*x

theorem f_composition_equals_8c_implies_c_equals_1 (c : ℝ) :
  f c (f c 0) = 8*c → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_8c_implies_c_equals_1_l1187_118763


namespace NUMINAMATH_CALUDE_cubic_sum_fraction_l1187_118795

theorem cubic_sum_fraction (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z = 0) (hprod : x*y + x*z + y*z ≠ 0) :
  (x^3 + y^3 + z^3) / (x*y*z * (x*y + x*z + y*z)) = -3 / (2*(x^2 + y^2 + x*y)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_fraction_l1187_118795


namespace NUMINAMATH_CALUDE_negative_product_sum_l1187_118773

theorem negative_product_sum (a b : ℚ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_sum_l1187_118773


namespace NUMINAMATH_CALUDE_quadratic_point_ordering_l1187_118794

/-- Given a quadratic function f(x) = x² + 2x + c, prove that for points
    A(-3, y₁), B(-2, y₂), and C(2, y₃) on its graph, y₃ > y₁ > y₂ holds. -/
theorem quadratic_point_ordering (c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x + c
  let y₁ : ℝ := f (-3)
  let y₂ : ℝ := f (-2)
  let y₃ : ℝ := f 2
  y₃ > y₁ ∧ y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_ordering_l1187_118794


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1187_118743

-- Define the quadratic function
def y (x m : ℝ) : ℝ := (x - 1) * (x - m + 1)

-- State the theorem
theorem quadratic_function_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, y x m > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1187_118743


namespace NUMINAMATH_CALUDE_largest_house_number_l1187_118722

def phone_number : List Nat := [4, 3, 1, 7, 8, 2]

def digit_sum (num : List Nat) : Nat :=
  num.sum

def is_distinct (num : List Nat) : Prop :=
  num.length = num.toFinset.card

theorem largest_house_number :
  ∃ (house : List Nat),
    house.length = 5 ∧
    is_distinct house ∧
    digit_sum house = digit_sum phone_number ∧
    (∀ other : List Nat,
      other.length = 5 →
      is_distinct other →
      digit_sum other = digit_sum phone_number →
      house.foldl (fun acc d => acc * 10 + d) 0 ≥
      other.foldl (fun acc d => acc * 10 + d) 0) ∧
    house = [9, 8, 7, 1, 0] :=
sorry

end NUMINAMATH_CALUDE_largest_house_number_l1187_118722


namespace NUMINAMATH_CALUDE_not_or_false_implies_and_or_l1187_118710

theorem not_or_false_implies_and_or (p q : Prop) :
  ¬(¬p ∨ ¬q) → (p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_or_false_implies_and_or_l1187_118710


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l1187_118725

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (2, -3)

/-- First line equation -/
def line1 (x y : ℝ) : Prop := 9 * x - 4 * y = 30

/-- Second line equation -/
def line2 (x y : ℝ) : Prop := 7 * x + y = 11

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y := by
  sorry

#check intersection_point_is_solution

end NUMINAMATH_CALUDE_intersection_point_is_solution_l1187_118725


namespace NUMINAMATH_CALUDE_odd_function_m_zero_l1187_118730

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- The function f(x) = 2x^3 + m -/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ 2 * x^3 + m

theorem odd_function_m_zero :
  ∀ m : ℝ, IsOdd (f m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_m_zero_l1187_118730


namespace NUMINAMATH_CALUDE_student_tickets_sold_l1187_118700

theorem student_tickets_sold (adult_price student_price : ℚ)
  (total_tickets : ℕ) (total_amount : ℚ)
  (h1 : adult_price = 4)
  (h2 : student_price = 5/2)
  (h3 : total_tickets = 59)
  (h4 : total_amount = 445/2) :
  ∃ (student_tickets : ℕ),
    student_tickets = 9 ∧
    student_tickets ≤ total_tickets ∧
    ∃ (adult_tickets : ℕ),
      adult_tickets + student_tickets = total_tickets ∧
      adult_price * adult_tickets + student_price * student_tickets = total_amount :=
by sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l1187_118700


namespace NUMINAMATH_CALUDE_elevator_distribution_ways_l1187_118771

/-- The number of elevators available --/
def num_elevators : ℕ := 4

/-- The number of people taking elevators --/
def num_people : ℕ := 3

/-- The number of people taking the same elevator --/
def same_elevator : ℕ := 2

/-- The number of ways to distribute people among elevators --/
def distribute_ways : ℕ := 36

/-- Theorem stating that the number of ways to distribute people among elevators is 36 --/
theorem elevator_distribution_ways :
  (num_elevators = 4) →
  (num_people = 3) →
  (same_elevator = 2) →
  (distribute_ways = 36) := by
sorry

end NUMINAMATH_CALUDE_elevator_distribution_ways_l1187_118771


namespace NUMINAMATH_CALUDE_frustum_slant_height_is_9_l1187_118774

/-- Represents a cone cut by a plane parallel to its base, forming a frustum -/
structure ConeFrustum where
  -- Ratio of top to bottom surface areas
  area_ratio : ℝ
  -- Slant height of the removed cone
  removed_slant_height : ℝ

/-- Calculates the slant height of the frustum -/
def slant_height_frustum (cf : ConeFrustum) : ℝ :=
  sorry

/-- Theorem stating the slant height of the frustum is 9 given the conditions -/
theorem frustum_slant_height_is_9 (cf : ConeFrustum) 
  (h1 : cf.area_ratio = 1 / 16)
  (h2 : cf.removed_slant_height = 3) : 
  slant_height_frustum cf = 9 :=
sorry

end NUMINAMATH_CALUDE_frustum_slant_height_is_9_l1187_118774


namespace NUMINAMATH_CALUDE_sears_tower_height_calculation_l1187_118717

/-- The height of Burj Khalifa in meters -/
def burj_khalifa_height : ℕ := 830

/-- The difference in height between Burj Khalifa and Sears Tower in meters -/
def height_difference : ℕ := 303

/-- The height of Sears Tower in meters -/
def sears_tower_height : ℕ := burj_khalifa_height - height_difference

theorem sears_tower_height_calculation :
  sears_tower_height = 527 :=
by sorry

end NUMINAMATH_CALUDE_sears_tower_height_calculation_l1187_118717


namespace NUMINAMATH_CALUDE_sum_of_roots_is_36_l1187_118778

def f (x : ℝ) : ℝ := (11 - x)^3 + (13 - x)^3 - (24 - 2*x)^3

theorem sum_of_roots_is_36 :
  ∃ (x₁ x₂ x₃ : ℝ), 
    (∀ x, f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    x₁ + x₂ + x₃ = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_36_l1187_118778


namespace NUMINAMATH_CALUDE_sqrt_less_than_2x_iff_x_greater_than_quarter_l1187_118734

theorem sqrt_less_than_2x_iff_x_greater_than_quarter (x : ℝ) (hx : x > 0) :
  Real.sqrt x < 2 * x ↔ x > (1 / 4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_sqrt_less_than_2x_iff_x_greater_than_quarter_l1187_118734


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1187_118764

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1187_118764


namespace NUMINAMATH_CALUDE_triangle_shape_l1187_118728

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- positive side lengths
  (A > 0 ∧ B > 0 ∧ C > 0) →  -- positive angles
  (A + B + C = 180) →        -- sum of angles in a triangle
  (A = 30 ∨ B = 30 ∨ C = 30) →  -- one angle is 30°
  (a = 2*b ∨ b = 2*c ∨ c = 2*a) →  -- one side is twice another
  (¬(A < 90 ∧ B < 90 ∧ C < 90) ∧ (C = 90 ∨ C > 90 ∨ B > 90)) := by
sorry

end NUMINAMATH_CALUDE_triangle_shape_l1187_118728


namespace NUMINAMATH_CALUDE_perfect_square_triples_l1187_118719

theorem perfect_square_triples :
  ∀ (a b c : ℕ),
    (∃ (x : ℕ), a^2 + 2*b + c = x^2) ∧
    (∃ (y : ℕ), b^2 + 2*c + a = y^2) ∧
    (∃ (z : ℕ), c^2 + 2*a + b = z^2) →
    ((a, b, c) = (0, 0, 0) ∨
     (a, b, c) = (1, 1, 1) ∨
     (a, b, c) = (127, 106, 43) ∨
     (a, b, c) = (106, 43, 127) ∨
     (a, b, c) = (43, 127, 106)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_triples_l1187_118719


namespace NUMINAMATH_CALUDE_volleyball_team_lineups_l1187_118772

def team_size : ℕ := 16
def quadruplet_size : ℕ := 4
def starter_size : ℕ := 6

def valid_lineups : ℕ := Nat.choose team_size starter_size - Nat.choose (team_size - quadruplet_size) (starter_size - quadruplet_size)

theorem volleyball_team_lineups : valid_lineups = 7942 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_lineups_l1187_118772


namespace NUMINAMATH_CALUDE_marble_problem_l1187_118703

theorem marble_problem (A V : ℤ) (x : ℤ) : 
  (A + x = V - x) ∧ (V + 2*x = A - 2*x + 30) → x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l1187_118703


namespace NUMINAMATH_CALUDE_dani_pants_reward_l1187_118733

/-- The number of pairs of pants Dani gets each year -/
def pants_per_year (initial_pants : ℕ) (pants_after_5_years : ℕ) : ℕ :=
  ((pants_after_5_years - initial_pants) / 5) / 2

/-- Theorem stating that Dani gets 4 pairs of pants each year -/
theorem dani_pants_reward (initial_pants : ℕ) (pants_after_5_years : ℕ) 
  (h1 : initial_pants = 50) 
  (h2 : pants_after_5_years = 90) : 
  pants_per_year initial_pants pants_after_5_years = 4 := by
  sorry

end NUMINAMATH_CALUDE_dani_pants_reward_l1187_118733


namespace NUMINAMATH_CALUDE_initial_kids_on_field_l1187_118775

theorem initial_kids_on_field (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  joined = 22 → total = 36 → total = initial + joined → initial = 14 := by
sorry

end NUMINAMATH_CALUDE_initial_kids_on_field_l1187_118775


namespace NUMINAMATH_CALUDE_shaded_area_comparison_l1187_118789

/-- Represents a square divided into smaller squares -/
structure DividedSquare where
  total_divisions : ℕ
  shaded_divisions : ℕ

/-- The three squares described in the problem -/
def square_I : DividedSquare := { total_divisions := 16, shaded_divisions := 4 }
def square_II : DividedSquare := { total_divisions := 64, shaded_divisions := 16 }
def square_III : DividedSquare := { total_divisions := 16, shaded_divisions := 8 }

/-- Calculates the shaded area ratio of a divided square -/
def shaded_area_ratio (s : DividedSquare) : ℚ :=
  s.shaded_divisions / s.total_divisions

/-- Theorem stating the equality of shaded areas for squares I and II, and the difference for square III -/
theorem shaded_area_comparison :
  shaded_area_ratio square_I = shaded_area_ratio square_II ∧
  shaded_area_ratio square_I ≠ shaded_area_ratio square_III ∧
  shaded_area_ratio square_II ≠ shaded_area_ratio square_III := by
  sorry

#eval shaded_area_ratio square_I
#eval shaded_area_ratio square_II
#eval shaded_area_ratio square_III

end NUMINAMATH_CALUDE_shaded_area_comparison_l1187_118789


namespace NUMINAMATH_CALUDE_smallest_square_addition_l1187_118766

theorem smallest_square_addition (n : ℕ) (h : n = 2019) : 
  ∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m^2 ∧ 
  ∀ k : ℕ, k < 1 → ¬∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + k = m^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_addition_l1187_118766


namespace NUMINAMATH_CALUDE_binary_representation_of_2_pow_n_minus_1_binary_to_decimal_ten_ones_l1187_118738

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents a binary number with n ones -/
def all_ones (n : ℕ) : List Bool :=
  List.replicate n true

theorem binary_representation_of_2_pow_n_minus_1 (n : ℕ) :
  binary_to_decimal (all_ones n) = 2^n - 1 := by
  sorry

/-- The main theorem proving that (1111111111)₂ in decimal form is 2^10 - 1 -/
theorem binary_to_decimal_ten_ones :
  binary_to_decimal (all_ones 10) = 2^10 - 1 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_2_pow_n_minus_1_binary_to_decimal_ten_ones_l1187_118738


namespace NUMINAMATH_CALUDE_correct_first_year_caps_l1187_118753

/-- The number of caps Lilith collects per month in the first year -/
def first_year_monthly_caps : ℕ := 3

/-- The number of years Lilith has been collecting caps -/
def total_years : ℕ := 5

/-- The number of caps Lilith collects per month after the first year -/
def later_years_monthly_caps : ℕ := 5

/-- The number of caps Lilith receives each Christmas -/
def christmas_caps : ℕ := 40

/-- The number of caps Lilith loses each year -/
def yearly_lost_caps : ℕ := 15

/-- The total number of caps Lilith has collected after 5 years -/
def total_caps : ℕ := 401

theorem correct_first_year_caps : 
  first_year_monthly_caps * 12 + 
  (total_years - 1) * 12 * later_years_monthly_caps + 
  total_years * christmas_caps - 
  total_years * yearly_lost_caps = total_caps := by
  sorry

end NUMINAMATH_CALUDE_correct_first_year_caps_l1187_118753


namespace NUMINAMATH_CALUDE_sister_share_is_49_50_l1187_118758

/-- Calculates the amount each sister receives after Gina's spending and investments --/
def sister_share (initial_amount : ℚ) : ℚ :=
  let mom_share := initial_amount * (1 / 4)
  let clothes_share := initial_amount * (1 / 8)
  let charity_share := initial_amount * (1 / 5)
  let groceries_share := initial_amount * (15 / 100)
  let remaining_before_stocks := initial_amount - mom_share - clothes_share - charity_share - groceries_share
  let stocks_investment := remaining_before_stocks * (1 / 10)
  let final_remaining := remaining_before_stocks - stocks_investment
  final_remaining / 2

/-- Theorem stating that each sister receives $49.50 --/
theorem sister_share_is_49_50 :
  sister_share 400 = 49.50 := by sorry

end NUMINAMATH_CALUDE_sister_share_is_49_50_l1187_118758


namespace NUMINAMATH_CALUDE_rational_roots_count_l1187_118767

/-- A polynomial with integer coefficients of the form 9x^4 + a₃x³ + a₂x² + a₁x + 15 = 0 -/
def IntPolynomial (a₃ a₂ a₁ : ℤ) (x : ℚ) : ℚ :=
  9 * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + 15

/-- The set of possible rational roots of the polynomial -/
def PossibleRoots : Finset ℚ :=
  {1, -1, 3, -3, 5, -5, 15, -15, 1/3, -1/3, 5/3, -5/3, 1/9, -1/9, 5/9, -5/9}

theorem rational_roots_count (a₃ a₂ a₁ : ℤ) :
  (PossibleRoots.filter (fun x => IntPolynomial a₃ a₂ a₁ x = 0)).card ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_rational_roots_count_l1187_118767


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l1187_118762

theorem x_value_when_y_is_two (x y : ℝ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l1187_118762


namespace NUMINAMATH_CALUDE_function_satisfying_condition_l1187_118705

/-- A function that satisfies f(a f(b)) = a b for all a and b -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a * f b) = a * b

/-- The theorem stating that a function satisfying the condition must be either the identity function or its negation -/
theorem function_satisfying_condition (f : ℝ → ℝ) (h : SatisfiesCondition f) :
    (∀ x, f x = x) ∨ (∀ x, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_condition_l1187_118705


namespace NUMINAMATH_CALUDE_min_point_of_translated_abs_function_l1187_118779

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| - 2

-- State the theorem
theorem min_point_of_translated_abs_function :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (x₀ = 4 ∧ f x₀ = -2) :=
sorry

end NUMINAMATH_CALUDE_min_point_of_translated_abs_function_l1187_118779


namespace NUMINAMATH_CALUDE_right_handed_players_count_l1187_118788

theorem right_handed_players_count (total_players throwers : ℕ) : 
  total_players = 70 →
  throwers = 34 →
  throwers ≤ total_players →
  (total_players - throwers) % 3 = 0 →
  (∃ (right_handed : ℕ), 
    right_handed = throwers + 2 * ((total_players - throwers) / 3) ∧
    right_handed = 58) := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l1187_118788


namespace NUMINAMATH_CALUDE_partner_investment_period_l1187_118731

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where q invests for 14 months, prove that p invests for 7 months. -/
theorem partner_investment_period
  (investment_ratio : ℚ) -- Ratio of p's investment to q's investment
  (profit_ratio : ℚ) -- Ratio of p's profit to q's profit
  (q_period : ℕ) -- Investment period of partner q in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 10)
  (h3 : q_period = 14) :
  ∃ (p_period : ℕ), p_period = 7 ∧ 
    (investment_ratio * p_period) / (q_period : ℚ) = profit_ratio :=
by sorry

end NUMINAMATH_CALUDE_partner_investment_period_l1187_118731


namespace NUMINAMATH_CALUDE_only_parallelogram_centrally_symmetric_l1187_118735

-- Define the shapes
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | RegularPentagon
  | RightTriangle

-- Define central symmetry
def is_centrally_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | _ => False

-- Theorem statement
theorem only_parallelogram_centrally_symmetric :
  ∀ s : Shape, is_centrally_symmetric s ↔ s = Shape.Parallelogram :=
by
  sorry

end NUMINAMATH_CALUDE_only_parallelogram_centrally_symmetric_l1187_118735


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l1187_118721

theorem rectangle_area_ratio : 
  let length_A : ℝ := 48
  let breadth_A : ℝ := 30
  let length_B : ℝ := 60
  let breadth_B : ℝ := 35
  let area_A := length_A * breadth_A
  let area_B := length_B * breadth_B
  (area_A / area_B) = 24 / 35 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l1187_118721


namespace NUMINAMATH_CALUDE_log_ratio_squared_l1187_118716

theorem log_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x = Real.log a ∧ y = Real.log b ∧ 2 * x^2 - 4 * x + 1 = 0 ∧ 2 * y^2 - 4 * y + 1 = 0) →
  (Real.log (a / b))^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l1187_118716


namespace NUMINAMATH_CALUDE_james_living_room_set_price_l1187_118713

/-- The final price James paid for the living room set after discount -/
theorem james_living_room_set_price (coach : ℝ) (sectional : ℝ) (other : ℝ) 
  (h1 : coach = 2500)
  (h2 : sectional = 3500)
  (h3 : other = 2000)
  (discount_rate : ℝ) 
  (h4 : discount_rate = 0.1) : 
  (coach + sectional + other) * (1 - discount_rate) = 7200 := by
  sorry

end NUMINAMATH_CALUDE_james_living_room_set_price_l1187_118713


namespace NUMINAMATH_CALUDE_magic_trick_strategy_exists_l1187_118746

/-- Represents a card in the set of 29 cards -/
def Card := Fin 29

/-- Represents a pair of cards -/
def CardPair := (Card × Card)

/-- The strategy function for the assistant -/
def AssistantStrategy := (CardPair → CardPair)

/-- The deduction function for the magician -/
def MagicianDeduction := (CardPair → CardPair)

/-- Theorem stating the existence of a successful strategy -/
theorem magic_trick_strategy_exists :
  ∃ (strategy : AssistantStrategy) (deduction : MagicianDeduction),
    ∀ (audience_choice : CardPair),
      deduction (strategy audience_choice) = audience_choice :=
by sorry

end NUMINAMATH_CALUDE_magic_trick_strategy_exists_l1187_118746


namespace NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l1187_118757

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + 1

theorem f_monotonicity_and_m_range :
  ∀ (a : ℝ),
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ ∧ a ≤ Real.sqrt 2 → f a x₁ < f a x₂) ∧
  (a > Real.sqrt 2 → 
    ∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x : ℝ, x₁ < x ∧ x < x₂ → f a x > f a x₁ ∧ f a x > f a x₃) ∧
    (∀ x : ℝ, 0 < x ∧ x < x₁ → f a x < f a x₁) ∧
    (∀ x : ℝ, x > x₃ → f a x > f a x₃)) ∧
  (∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ 1 ∧
    (∀ m : ℝ, (∀ a : ℝ, -2 < a ∧ a ≤ 0 → 
      2*m*Real.exp a*(a+1) + f a x₀ > a^2 + 2*a + 4) ↔ 1 < m ∧ m ≤ Real.exp 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l1187_118757


namespace NUMINAMATH_CALUDE_triangle_height_and_median_l1187_118715

-- Define the triangle vertices
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 6)
def C : ℝ × ℝ := (0, 2)

-- Define the height line from A to BC
def height_line (x y : ℝ) : Prop := 3 * x + 2 * y - 12 = 0

-- Define the median line from B to AC
def median_line (x y : ℝ) : Prop := x + 2 * y - 18 = 0

-- Theorem statement
theorem triangle_height_and_median :
  (∀ x y, height_line x y ↔ 
    (y - A.2) = -(3/2) * (x - A.1) ∧ 
    (y - B.2) * (C.1 - B.1) = (C.2 - B.2) * (x - B.1)) ∧
  (∀ x y, median_line x y ↔ 
    (y - B.2) = -(1/2) * (x - B.1) ∧ 
    (x, y) = B ∨ (x, y) = ((A.1 + C.1)/2, (A.2 + C.2)/2)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_height_and_median_l1187_118715


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l1187_118711

theorem bicycle_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (price_C : ℝ) 
  (h1 : profit_A_to_B = 0.20)
  (h2 : profit_B_to_C = 0.25)
  (h3 : price_C = 225) :
  ∃ (cost_price_A : ℝ), 
    cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) = price_C ∧ 
    cost_price_A = 150 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l1187_118711


namespace NUMINAMATH_CALUDE_log_sum_equation_l1187_118747

theorem log_sum_equation (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 →
  y = 3 ^ (10 / 3) := by
sorry

end NUMINAMATH_CALUDE_log_sum_equation_l1187_118747


namespace NUMINAMATH_CALUDE_chocolate_bar_breaks_chocolate_bar_40_pieces_l1187_118777

/-- The minimum number of breaks required to separate a chocolate bar into individual pieces -/
def min_breaks (n : ℕ) : ℕ := n - 1

/-- Theorem stating that the minimum number of breaks for a chocolate bar with n pieces is n - 1 -/
theorem chocolate_bar_breaks (n : ℕ) (h : n > 0) : 
  min_breaks n = n - 1 := by
  sorry

/-- Corollary for the specific case of 40 pieces -/
theorem chocolate_bar_40_pieces : 
  min_breaks 40 = 39 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_breaks_chocolate_bar_40_pieces_l1187_118777


namespace NUMINAMATH_CALUDE_remainder_problem_l1187_118780

theorem remainder_problem (x y : ℤ) 
  (hx : x % 72 = 65) 
  (hy : y % 54 = 22) : 
  (x - y) % 18 = 7 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l1187_118780


namespace NUMINAMATH_CALUDE_special_function_upper_bound_l1187_118791

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  nonneg : ∀ x, x ≥ 0 → f x ≥ 0
  inequality : ∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)
  bounded : ∃ M, M > 0 ∧ ∀ x, 0 ≤ x → x ≤ 1 → |f x| ≤ M

/-- The main theorem to be proved -/
theorem special_function_upper_bound (sf : SpecialFunction) : 
  ∀ x, x ≥ 0 → sf.f x ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_upper_bound_l1187_118791


namespace NUMINAMATH_CALUDE_smallest_max_volume_is_500_l1187_118720

/-- Represents a cuboid with integral side lengths -/
structure Cuboid where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the volume of a cuboid -/
def Cuboid.volume (c : Cuboid) : ℕ := c.length.val * c.width.val * c.height.val

/-- Represents the result of cutting a cube into three cuboids -/
structure CubeCut where
  cuboid1 : Cuboid
  cuboid2 : Cuboid
  cuboid3 : Cuboid

/-- Checks if a CubeCut is valid for a cube with side length 10 -/
def isValidCubeCut (cut : CubeCut) : Prop :=
  cut.cuboid1.length + cut.cuboid2.length + cut.cuboid3.length = 10 ∧
  cut.cuboid1.width = 10 ∧ cut.cuboid2.width = 10 ∧ cut.cuboid3.width = 10 ∧
  cut.cuboid1.height = 10 ∧ cut.cuboid2.height = 10 ∧ cut.cuboid3.height = 10

/-- The main theorem to prove -/
theorem smallest_max_volume_is_500 :
  ∀ (cut : CubeCut),
    isValidCubeCut cut →
    max cut.cuboid1.volume (max cut.cuboid2.volume cut.cuboid3.volume) ≥ 500 :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_volume_is_500_l1187_118720
