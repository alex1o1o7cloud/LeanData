import Mathlib

namespace NUMINAMATH_CALUDE_third_degree_polynomial_specific_value_l1199_119911

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that the absolute value of g at certain points equals 10 -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  |g (-1)| = 10 ∧ |g 0| = 10 ∧ |g 2| = 10 ∧ |g 4| = 10 ∧ |g 5| = 10 ∧ |g 8| = 10

/-- The theorem statement -/
theorem third_degree_polynomial_specific_value (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g 3| = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_specific_value_l1199_119911


namespace NUMINAMATH_CALUDE_store_discount_percentage_l1199_119983

theorem store_discount_percentage (C : ℝ) (C_pos : C > 0) : 
  let initial_price := 1.20 * C
  let new_year_price := 1.25 * initial_price
  let final_price := 1.32 * C
  let discount_percentage := (new_year_price - final_price) / new_year_price * 100
  discount_percentage = 12 := by sorry

end NUMINAMATH_CALUDE_store_discount_percentage_l1199_119983


namespace NUMINAMATH_CALUDE_product_of_roots_l1199_119941

theorem product_of_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ r₄ : ℝ, x^4 - 12*x^3 + 50*x^2 + 48*x - 35 = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄)) →
  (∃ r₁ r₂ r₃ r₄ : ℝ, x^4 - 12*x^3 + 50*x^2 + 48*x - 35 = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) ∧
                       r₁ * r₂ * r₃ * r₄ = 35) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1199_119941


namespace NUMINAMATH_CALUDE_function_properties_and_triangle_l1199_119945

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 + 3

theorem function_properties_and_triangle (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f x ≤ 4) ∧ 
  (∀ ε > 0, ∃ T > 0, T ≤ π ∧ ∀ x, f (x + T) = f x) ∧
  c = Real.sqrt 3 →
  f C = 4 →
  Real.sin A = 2 * Real.sin B →
  a = 2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_and_triangle_l1199_119945


namespace NUMINAMATH_CALUDE_smallest_integers_satisfying_equation_l1199_119950

theorem smallest_integers_satisfying_equation :
  ∃ (a b : ℕ+),
    (7 * a^3 = 11 * b^5) ∧
    (∀ (a' b' : ℕ+), 7 * a'^3 = 11 * b'^5 → a ≤ a' ∧ b ≤ b') ∧
    a = 41503 ∧
    b = 539 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integers_satisfying_equation_l1199_119950


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l1199_119953

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l1199_119953


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_450_l1199_119954

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 450 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 450 → m ≥ n :=
  sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_450_l1199_119954


namespace NUMINAMATH_CALUDE_loss_percentage_tables_l1199_119998

theorem loss_percentage_tables (C S : ℝ) (h : 15 * C = 20 * S) : 
  (C - S) / C * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_tables_l1199_119998


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a10_l1199_119915

/-- An arithmetic sequence with a_2 = 2 and a_3 = 4 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 3 = 4 ∧ ∀ n : ℕ, a (n + 1) - a n = a 3 - a 2

theorem arithmetic_sequence_a10 (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a10_l1199_119915


namespace NUMINAMATH_CALUDE_max_median_value_l1199_119947

theorem max_median_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 16 →
  k < m → m < r → r < s → s < t →
  t = 42 →
  r ≤ 17 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 42) / 5 = 16 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 42 ∧
    r' = 17 :=
by sorry

end NUMINAMATH_CALUDE_max_median_value_l1199_119947


namespace NUMINAMATH_CALUDE_julia_short_amount_l1199_119996

def rock_price : ℝ := 5
def pop_price : ℝ := 10
def dance_price : ℝ := 3
def country_price : ℝ := 7
def discount_rate : ℝ := 0.1
def julia_money : ℝ := 75

def rock_quantity : ℕ := 3
def pop_quantity : ℕ := 4
def dance_quantity : ℕ := 2
def country_quantity : ℕ := 4

def discount_threshold : ℕ := 3

def genre_cost (price : ℝ) (quantity : ℕ) : ℝ := price * quantity

def apply_discount (cost : ℝ) (quantity : ℕ) : ℝ :=
  if quantity ≥ discount_threshold then cost * (1 - discount_rate) else cost

theorem julia_short_amount : 
  let rock_cost := apply_discount (genre_cost rock_price rock_quantity) rock_quantity
  let pop_cost := apply_discount (genre_cost pop_price pop_quantity) pop_quantity
  let dance_cost := apply_discount (genre_cost dance_price dance_quantity) dance_quantity
  let country_cost := apply_discount (genre_cost country_price country_quantity) country_quantity
  let total_cost := rock_cost + pop_cost + dance_cost + country_cost
  total_cost - julia_money = 7.2 := by sorry

end NUMINAMATH_CALUDE_julia_short_amount_l1199_119996


namespace NUMINAMATH_CALUDE_perpendicular_slope_l1199_119914

/-- Given a line with equation 5x - 2y = 10, the slope of a perpendicular line is -2/5 -/
theorem perpendicular_slope (x y : ℝ) :
  (5 * x - 2 * y = 10) → 
  (slope_of_perpendicular_line : ℝ) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l1199_119914


namespace NUMINAMATH_CALUDE_max_digit_count_is_24_l1199_119903

def apartment_numbers : List Nat := 
  (List.range 46).map (· + 90) ++ (List.range 46).map (· + 190)

def digit_count (d : Nat) (n : Nat) : Nat :=
  if n = 0 then
    if d = 0 then 1 else 0
  else
    digit_count d (n / 10) + if n % 10 = d then 1 else 0

def count_digit (d : Nat) (numbers : List Nat) : Nat :=
  numbers.foldl (fun acc n => acc + digit_count d n) 0

theorem max_digit_count_is_24 :
  (List.range 10).foldl (fun acc d => max acc (count_digit d apartment_numbers)) 0 = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_digit_count_is_24_l1199_119903


namespace NUMINAMATH_CALUDE_reflection_of_point_A_l1199_119980

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The initial point A -/
def point_A : ℝ × ℝ := (1, 2)

theorem reflection_of_point_A :
  reflect_y_axis point_A = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_point_A_l1199_119980


namespace NUMINAMATH_CALUDE_min_value_and_exponential_sum_l1199_119913

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - 2*a| + |x + b|

-- State the theorem
theorem min_value_and_exponential_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f x a b ≥ 2) 
  (hmin_exists : ∃ x, f x a b = 2) : 
  (2*a + b = 2) ∧ (∀ a' b', a' > 0 → b' > 0 → 2*a' + b' = 2 → 9^a' + 3^b' ≥ 6) ∧ 
  (∃ a' b', a' > 0 ∧ b' > 0 ∧ 2*a' + b' = 2 ∧ 9^a' + 3^b' = 6) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_exponential_sum_l1199_119913


namespace NUMINAMATH_CALUDE_fraction_sum_l1199_119938

theorem fraction_sum (x y : ℝ) (h : y / x = 3 / 4) : (x + y) / x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1199_119938


namespace NUMINAMATH_CALUDE_line_equation_through_A_and_B_l1199_119949

/-- Two-point form equation of a line passing through two points -/
def two_point_form (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1)

/-- Theorem: The two-point form equation of the line passing through A(1,2) and B(-1,1) -/
theorem line_equation_through_A_and_B :
  two_point_form 1 2 (-1) 1 x y ↔ (x - 1) / (-2) = (y - 2) / (-1) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_A_and_B_l1199_119949


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l1199_119912

theorem pet_store_siamese_cats 
  (house_cats : ℕ) 
  (cats_sold : ℕ) 
  (cats_remaining : ℕ) 
  (h1 : house_cats = 25)
  (h2 : cats_sold = 45)
  (h3 : cats_remaining = 18) :
  ∃ (initial_siamese : ℕ), 
    initial_siamese + house_cats = cats_sold + cats_remaining ∧ 
    initial_siamese = 38 := by
sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l1199_119912


namespace NUMINAMATH_CALUDE_mrs_snyder_pink_cookies_l1199_119919

/-- The total number of cookies Mrs. Snyder made -/
def total_cookies : ℕ := 86

/-- The number of red cookies Mrs. Snyder made -/
def red_cookies : ℕ := 36

/-- The number of pink cookies Mrs. Snyder made -/
def pink_cookies : ℕ := total_cookies - red_cookies

theorem mrs_snyder_pink_cookies : pink_cookies = 50 := by
  sorry

end NUMINAMATH_CALUDE_mrs_snyder_pink_cookies_l1199_119919


namespace NUMINAMATH_CALUDE_min_value_a_plus_8b_l1199_119991

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  ∀ x y, x > 0 → y > 0 → 2 * x * y = x + 2 * y → x + 8 * y ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_8b_l1199_119991


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1199_119905

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2) :
  a / (a^2 + 2*a + 1) / (1 - a / (a + 1)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1199_119905


namespace NUMINAMATH_CALUDE_each_brother_pays_19_80_l1199_119985

/-- The amount each brother pays when buying cakes and splitting the cost -/
def amount_per_person (num_cakes : ℕ) (price_per_cake : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_before_tax := num_cakes * price_per_cake
  let tax_amount := total_before_tax * tax_rate
  let total_after_tax := total_before_tax + tax_amount
  total_after_tax / 2

/-- Theorem stating that each brother pays $19.80 -/
theorem each_brother_pays_19_80 :
  amount_per_person 3 12 (1/10) = 198/10 := by
  sorry

end NUMINAMATH_CALUDE_each_brother_pays_19_80_l1199_119985


namespace NUMINAMATH_CALUDE_floor_of_e_l1199_119930

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_e_l1199_119930


namespace NUMINAMATH_CALUDE_max_profit_at_10_max_profit_value_l1199_119944

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2 * x

/-- Total profit function -/
def totalProfit (x : ℝ) : ℝ := L₁ x + L₂ (15 - x)

/-- The maximum profit is achieved when selling 10 cars in location A -/
theorem max_profit_at_10 :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 15 → totalProfit x ≤ totalProfit 10 :=
sorry

/-- The maximum profit is 45.6 -/
theorem max_profit_value : totalProfit 10 = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_10_max_profit_value_l1199_119944


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1199_119928

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 3*x + 2 - 12
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1199_119928


namespace NUMINAMATH_CALUDE_wendy_recycling_l1199_119972

/-- Given that Wendy earns 5 points per bag recycled, had 11 bags in total, 
    and earned 45 points, prove that she did not recycle 2 bags. -/
theorem wendy_recycling (points_per_bag : ℕ) (total_bags : ℕ) (total_points : ℕ) 
  (h1 : points_per_bag = 5)
  (h2 : total_bags = 11)
  (h3 : total_points = 45) :
  total_bags - (total_points / points_per_bag) = 2 := by
  sorry


end NUMINAMATH_CALUDE_wendy_recycling_l1199_119972


namespace NUMINAMATH_CALUDE_shaded_area_ratio_is_five_ninths_l1199_119981

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a shaded region in the grid -/
structure ShadedRegion :=
  (start_row : ℕ)
  (start_col : ℕ)
  (end_row : ℕ)
  (end_col : ℕ)

/-- Calculates the ratio of shaded area to total area -/
def shaded_area_ratio (g : Grid) (sr : ShadedRegion) : ℚ :=
  sorry

/-- Theorem stating the ratio of shaded area to total area for the given problem -/
theorem shaded_area_ratio_is_five_ninths :
  let g : Grid := ⟨9⟩
  let sr : ShadedRegion := ⟨2, 1, 5, 9⟩
  shaded_area_ratio g sr = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_is_five_ninths_l1199_119981


namespace NUMINAMATH_CALUDE_parabola_ellipse_intersection_l1199_119921

/-- Represents a parabola with equation y² = -4x -/
structure Parabola where
  equation : ∀ x y, y^2 = -4*x

/-- Represents an ellipse with equation x²/4 + y²/b² = 1, where b > 0 -/
structure Ellipse where
  b : ℝ
  b_pos : b > 0
  equation : ∀ x y, x^2/4 + y^2/b^2 = 1

/-- The x-coordinate of the latus rectum for a parabola y² = -4x -/
def latus_rectum_x (p : Parabola) : ℝ := 1

/-- The x-coordinate of the focus for an ellipse x²/4 + y²/b² = 1 -/
def focus_x (e : Ellipse) : ℝ := 1

/-- Theorem stating that if the latus rectum of the parabola passes through
    the focus of the ellipse, then b = √3 -/
theorem parabola_ellipse_intersection
  (p : Parabola) (e : Ellipse)
  (h : latus_rectum_x p = focus_x e) :
  e.b = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_parabola_ellipse_intersection_l1199_119921


namespace NUMINAMATH_CALUDE_power_multiplication_l1199_119918

theorem power_multiplication (a : ℝ) : a^3 * a = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1199_119918


namespace NUMINAMATH_CALUDE_prop1_prop4_l1199_119904

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perp_to_plane : Line → Plane → Prop)

-- Proposition 1
theorem prop1 (a b c : Line) :
  parallel a b → perpendicular b c → perpendicular a c :=
sorry

-- Proposition 4
theorem prop4 (a b : Line) (α : Plane) :
  perp_to_plane a α → contained_in b α → perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_prop1_prop4_l1199_119904


namespace NUMINAMATH_CALUDE_arithmetic_less_than_geometric_l1199_119975

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

/-- Theorem: For arithmetic sequence a and geometric sequence b satisfying given conditions,
    a_n < b_n for all n > 2 -/
theorem arithmetic_less_than_geometric
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_eq1 : a 1 = b 1)
  (h_eq2 : a 2 = b 2)
  (h_neq : a 1 ≠ a 2)
  (h_pos : ∀ i : ℕ, a i > 0) :
  ∀ n : ℕ, n > 2 → a n < b n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_less_than_geometric_l1199_119975


namespace NUMINAMATH_CALUDE_exponent_division_l1199_119970

theorem exponent_division (a : ℝ) (m n : ℕ) (h : a ≠ 0) : a^m / a^n = a^(m - n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1199_119970


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l1199_119966

/-- Represents a box of stationery -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- The scenario described in the problem -/
def stationery_scenario (box : StationeryBox) : Prop :=
  (box.sheets - box.envelopes = 30) ∧ 
  (2 * box.envelopes = box.sheets)

/-- The theorem to prove -/
theorem stationery_box_sheets : 
  ∀ (box : StationeryBox), stationery_scenario box → box.sheets = 60 := by
  sorry

end NUMINAMATH_CALUDE_stationery_box_sheets_l1199_119966


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1199_119995

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 7 → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1199_119995


namespace NUMINAMATH_CALUDE_stating_sum_of_sides_approx_11_2_l1199_119939

/-- Represents a right triangle with angles 40°, 50°, and 90° -/
structure RightTriangle40_50_90 where
  /-- The side opposite to the 50° angle -/
  side_a : ℝ
  /-- The side opposite to the 40° angle -/
  side_b : ℝ
  /-- The hypotenuse -/
  side_c : ℝ
  /-- Constraint that side_a is 8 units long -/
  side_a_eq_8 : side_a = 8

/-- 
Theorem stating that the sum of the two sides (opposite to 40° and 90°) 
in a 40-50-90 right triangle with hypotenuse of 8 units 
is approximately 11.2 units
-/
theorem sum_of_sides_approx_11_2 (t : RightTriangle40_50_90) :
  ∃ ε > 0, abs (t.side_b + t.side_c - 11.2) < ε := by
  sorry


end NUMINAMATH_CALUDE_stating_sum_of_sides_approx_11_2_l1199_119939


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l1199_119908

/-- The area of a circle with diameter endpoints C(-2,3) and D(4,-1) is 13π. -/
theorem circle_area_from_diameter_endpoints :
  let C : ℝ × ℝ := (-2, 3)
  let D : ℝ × ℝ := (4, -1)
  let diameter_squared := (D.1 - C.1)^2 + (D.2 - C.2)^2
  let radius_squared := diameter_squared / 4
  let circle_area := π * radius_squared
  circle_area = 13 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l1199_119908


namespace NUMINAMATH_CALUDE_ideal_gas_pressure_change_l1199_119920

/-- Given an ideal gas at constant temperature, calculate the new pressure when the volume changes. -/
theorem ideal_gas_pressure_change (V1 V2 P1 P2 : ℝ) (hV1 : V1 = 4.56) (hV2 : V2 = 2.28) (hP1 : P1 = 10) :
  V1 * P1 = V2 * P2 → P2 = 20 := by
  sorry

#check ideal_gas_pressure_change

end NUMINAMATH_CALUDE_ideal_gas_pressure_change_l1199_119920


namespace NUMINAMATH_CALUDE_shop_equations_correct_l1199_119901

/-- A shop with rooms and guests satisfying certain conditions -/
structure Shop where
  rooms : ℕ
  guests : ℕ
  seven_per_room_overflow : 7 * rooms + 7 = guests
  nine_per_room_empty : 9 * (rooms - 1) = guests

/-- The theorem stating that the system of equations correctly describes the shop's situation -/
theorem shop_equations_correct (s : Shop) : 
  (7 * s.rooms + 7 = s.guests) ∧ (9 * (s.rooms - 1) = s.guests) := by
  sorry

end NUMINAMATH_CALUDE_shop_equations_correct_l1199_119901


namespace NUMINAMATH_CALUDE_count_odd_sum_numbers_l1199_119958

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A function that checks if a number is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- A function that returns the sum of digits of a three-digit number -/
def digitSum (n : Nat) : Nat :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- The set of all three-digit numbers formed by the given digits without repetition -/
def threeDigitNumbers : Finset Nat :=
  Finset.filter (fun n => n ≥ 100 ∧ n < 1000 ∧ (Finset.card (Finset.filter (fun d => d ∈ digits) (Finset.range 10))) = 3) (Finset.range 1000)

theorem count_odd_sum_numbers :
  Finset.card (Finset.filter (fun n => isOdd (digitSum n)) threeDigitNumbers) = 24 := by sorry

end NUMINAMATH_CALUDE_count_odd_sum_numbers_l1199_119958


namespace NUMINAMATH_CALUDE_farm_animals_count_l1199_119973

theorem farm_animals_count (rabbits chickens : ℕ) : 
  rabbits = chickens + 17 → 
  rabbits = 64 → 
  rabbits + chickens = 111 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_count_l1199_119973


namespace NUMINAMATH_CALUDE_coefficient_sum_after_shift_l1199_119926

def original_function (x : ℝ) : ℝ := 2 * x^2 - x + 7

def shifted_function (x : ℝ) : ℝ := original_function (x - 4)

def quadratic_form (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem coefficient_sum_after_shift :
  ∃ (a b c : ℝ), (∀ x, shifted_function x = quadratic_form a b c x) ∧ a + b + c = 28 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_after_shift_l1199_119926


namespace NUMINAMATH_CALUDE_f_of_three_equals_zero_l1199_119990

theorem f_of_three_equals_zero (f : ℝ → ℝ) (h : ∀ x, f (1 - 2*x) = x^2 + x) : f 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_three_equals_zero_l1199_119990


namespace NUMINAMATH_CALUDE_workshop_percentage_l1199_119989

-- Define the workday duration in minutes
def workday_minutes : ℕ := 8 * 60

-- Define the duration of the first workshop in minutes
def first_workshop_minutes : ℕ := 60

-- Define the duration of the second workshop in minutes
def second_workshop_minutes : ℕ := 2 * first_workshop_minutes

-- Define the total time spent in workshops
def total_workshop_minutes : ℕ := first_workshop_minutes + second_workshop_minutes

-- Theorem statement
theorem workshop_percentage :
  (total_workshop_minutes : ℚ) / (workday_minutes : ℚ) * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_workshop_percentage_l1199_119989


namespace NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_three_l1199_119974

/-- The constant term in the expansion of (x^2 + 2)(1/x^2 - 1)^5 -/
theorem constant_term_expansion : ℤ :=
  let expansion := (fun x : ℚ => (x^2 + 2) * (1/x^2 - 1)^5)
  3

/-- Proof that the constant term in the expansion of (x^2 + 2)(1/x^2 - 1)^5 is 3 -/
theorem constant_term_is_three : constant_term_expansion = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_three_l1199_119974


namespace NUMINAMATH_CALUDE_average_equation_solution_l1199_119957

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 8) + (5*x + 3) + (3*x + 9)) = 5*x - 10 → x = 10 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1199_119957


namespace NUMINAMATH_CALUDE_smallest_n_for_2005_angles_l1199_119956

/-- A function that, given a natural number n, returns the number of angles not exceeding 120° 
    between pairs of points when n points are placed on a circle. -/
def anglesNotExceeding120 (n : ℕ) : ℕ := sorry

/-- The proposition that 91 is the smallest natural number satisfying the condition -/
theorem smallest_n_for_2005_angles : 
  (∀ n : ℕ, n < 91 → anglesNotExceeding120 n < 2005) ∧ 
  (anglesNotExceeding120 91 ≥ 2005) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_2005_angles_l1199_119956


namespace NUMINAMATH_CALUDE_tangent_x_intercept_difference_l1199_119992

/-- 
Given a point (x₀, y₀) on the curve y = e^x, if the tangent line at this point 
intersects the x-axis at (x₁, 0), then x₁ - x₀ = -1.
-/
theorem tangent_x_intercept_difference (x₀ : ℝ) : 
  let y₀ : ℝ := Real.exp x₀
  let f : ℝ → ℝ := λ x => Real.exp x
  let f' : ℝ → ℝ := λ x => Real.exp x
  let tangent_line : ℝ → ℝ := λ x => f' x₀ * (x - x₀) + y₀
  let x₁ : ℝ := x₀ - 1 / f' x₀
  (tangent_line x₁ = 0) → (x₁ - x₀ = -1) := by
  sorry

#check tangent_x_intercept_difference

end NUMINAMATH_CALUDE_tangent_x_intercept_difference_l1199_119992


namespace NUMINAMATH_CALUDE_father_son_age_difference_l1199_119960

/-- Proves that a father is 25 years older than his son given the problem conditions -/
theorem father_son_age_difference :
  ∀ (father_age son_age : ℕ),
    father_age > son_age →
    son_age = 23 →
    father_age + 2 = 2 * (son_age + 2) →
    father_age - son_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_difference_l1199_119960


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l1199_119959

variable (a b c : ℤ)
variable (ω : ℂ)

theorem min_value_complex_expression (h1 : a * b * c = 60)
                                     (h2 : ω ≠ 1)
                                     (h3 : ω^3 = 1) :
  ∃ (min : ℝ), min = Real.sqrt 3 ∧
    ∀ (x y z : ℤ), x * y * z = 60 →
      Complex.abs (↑x + ↑y * ω + ↑z * ω^2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l1199_119959


namespace NUMINAMATH_CALUDE_gcd_max_digits_l1199_119978

theorem gcd_max_digits (a b : ℕ) : 
  10000 ≤ a ∧ a < 100000 ∧ 
  10000 ≤ b ∧ b < 100000 ∧ 
  100000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000 →
  Nat.gcd a b < 100 := by
sorry

end NUMINAMATH_CALUDE_gcd_max_digits_l1199_119978


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l1199_119963

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let original_area := L * W
  let new_area := (1.2 * L) * (1.2 * W)
  (new_area - original_area) / original_area * 100 = 44 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l1199_119963


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1199_119994

def U : Finset Nat := {1, 3, 5, 7}
def M : Finset Nat := {1, 5}

theorem complement_of_M_in_U :
  (U \ M) = {3, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1199_119994


namespace NUMINAMATH_CALUDE_selene_purchase_total_l1199_119925

/-- The price of an instant camera -/
def camera_price : ℝ := 110

/-- The price of a digital photo frame -/
def frame_price : ℝ := 120

/-- The number of cameras purchased -/
def num_cameras : ℕ := 2

/-- The number of frames purchased -/
def num_frames : ℕ := 3

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.05

/-- The total amount Selene pays -/
def total_paid : ℝ := 551

theorem selene_purchase_total :
  (camera_price * num_cameras + frame_price * num_frames) * (1 - discount_rate) = total_paid := by
  sorry

end NUMINAMATH_CALUDE_selene_purchase_total_l1199_119925


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l1199_119936

-- Define the polynomial
def p (x y : ℝ) : ℝ := 3 * x * y^2 - 2 * y - 1

-- State the theorem
theorem polynomial_coefficients :
  (∃ a b c d : ℝ, ∀ x y : ℝ, p x y = a * x * y^2 + b * y + c * x + d) ∧
  (∀ a b c d : ℝ, (∀ x y : ℝ, p x y = a * x * y^2 + b * y + c * x + d) →
    b = -2 ∧ d = -1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l1199_119936


namespace NUMINAMATH_CALUDE_total_fruit_count_l1199_119946

/-- The number of crates containing oranges -/
def num_crates : ℕ := 25

/-- The number of oranges in each crate -/
def oranges_per_crate : ℕ := 270

/-- The number of boxes containing nectarines -/
def num_boxes : ℕ := 38

/-- The number of nectarines in each box -/
def nectarines_per_box : ℕ := 50

/-- The number of baskets containing apples -/
def num_baskets : ℕ := 15

/-- The number of apples in each basket -/
def apples_per_basket : ℕ := 90

/-- Theorem stating that the total number of pieces of fruit is 10,000 -/
theorem total_fruit_count : 
  num_crates * oranges_per_crate + 
  num_boxes * nectarines_per_box + 
  num_baskets * apples_per_basket = 10000 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_count_l1199_119946


namespace NUMINAMATH_CALUDE_rest_worker_salary_l1199_119931

def workshop (total_workers : ℕ) (avg_salary : ℚ) (technicians : ℕ) (avg_technician_salary : ℚ) : Prop :=
  total_workers = 12 ∧
  avg_salary = 9000 ∧
  technicians = 6 ∧
  avg_technician_salary = 12000

theorem rest_worker_salary (total_workers : ℕ) (avg_salary : ℚ) (technicians : ℕ) (avg_technician_salary : ℚ) :
  workshop total_workers avg_salary technicians avg_technician_salary →
  (total_workers * avg_salary - technicians * avg_technician_salary) / (total_workers - technicians) = 6000 :=
by
  sorry

#check rest_worker_salary

end NUMINAMATH_CALUDE_rest_worker_salary_l1199_119931


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l1199_119979

def regular_decagon : ℕ := 10

def total_triangles : ℕ := regular_decagon.choose 3

def favorable_outcomes : ℕ := regular_decagon * (regular_decagon - 4)

def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l1199_119979


namespace NUMINAMATH_CALUDE_similarity_coefficient_bounds_l1199_119952

/-- Two triangles are similar if their corresponding sides are proportional -/
def similar_triangles (x y z p : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ x = k * y ∧ y = k * z ∧ z = k * p

/-- The interval for the similarity coefficient -/
def similarity_coefficient_interval (k : ℝ) : Prop :=
  k > Real.sqrt 5 / 2 - 1 / 2 ∧ k < Real.sqrt 5 / 2 + 1 / 2

/-- Theorem: The similarity coefficient of two similar triangles lies within a specific interval -/
theorem similarity_coefficient_bounds (x y z p : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ p > 0) 
  (h_similar : similar_triangles x y z p) : 
  ∃ k : ℝ, similarity_coefficient_interval k ∧ x = k * y ∧ y = k * z ∧ z = k * p :=
by
  sorry

end NUMINAMATH_CALUDE_similarity_coefficient_bounds_l1199_119952


namespace NUMINAMATH_CALUDE_representations_non_negative_representations_natural_l1199_119961

/-- The number of ways to represent a natural number as a sum of non-negative integers -/
def representationsNonNegative (n m : ℕ) : ℕ := Nat.choose (n + m - 1) n

/-- The number of ways to represent a natural number as a sum of natural numbers -/
def representationsNatural (n m : ℕ) : ℕ := Nat.choose (n - 1) (n - m)

/-- Theorem stating the number of ways to represent n as a sum of m non-negative integers -/
theorem representations_non_negative (n m : ℕ) :
  representationsNonNegative n m = Nat.choose (n + m - 1) n := by sorry

/-- Theorem stating the number of ways to represent n as a sum of m natural numbers -/
theorem representations_natural (n m : ℕ) (h : m ≤ n) :
  representationsNatural n m = Nat.choose (n - 1) (n - m) := by sorry

end NUMINAMATH_CALUDE_representations_non_negative_representations_natural_l1199_119961


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1199_119977

/-- The solution set of the inequality 2 < |2x-5| ≤ 7 -/
def solution_set_1 : Set ℝ :=
  {x | -1 ≤ x ∧ x < 3/2 ∨ 7/2 < x ∧ x ≤ 6}

/-- The solution set of the inequality 1/(x-1) > x + 1 -/
def solution_set_2 : Set ℝ :=
  {x | x < -Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2)}

theorem inequality_solution_1 :
  {x : ℝ | 2 < |2*x - 5| ∧ |2*x - 5| ≤ 7} = solution_set_1 := by sorry

theorem inequality_solution_2 :
  {x : ℝ | 1/(x-1) > x + 1} = solution_set_2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1199_119977


namespace NUMINAMATH_CALUDE_A_equals_B_l1199_119907

-- Define set A
def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4*a + a^2}

-- Define set B
def B : Set ℝ := {y | ∃ b : ℝ, y = 4*b^2 + 4*b + 2}

-- Theorem statement
theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l1199_119907


namespace NUMINAMATH_CALUDE_decagon_area_ratio_l1199_119935

theorem decagon_area_ratio (decagon_area : ℝ) (below_PQ_square_area : ℝ) (triangle_base : ℝ) (XQ QY : ℝ) :
  decagon_area = 12 →
  below_PQ_square_area = 1 →
  triangle_base = 6 →
  XQ + QY = 6 →
  (decagon_area / 2 = below_PQ_square_area + (1/2 * triangle_base * ((decagon_area / 2) - below_PQ_square_area) / triangle_base)) →
  XQ / QY = 2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_area_ratio_l1199_119935


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1199_119982

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := 4 * a + 2 * b

-- State the theorem
theorem diamond_equation_solution :
  ∃ x : ℚ, diamond 3 (diamond x 7) = 5 ∧ x = -35/8 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l1199_119982


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1199_119900

def is_decreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≥ a (n + 1)

theorem contrapositive_equivalence (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (a n + a (n + 2)) / 2 < a (n + 1) → is_decreasing a) ↔
  (¬ is_decreasing a → ∀ n : ℕ+, (a n + a (n + 2)) / 2 ≥ a (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1199_119900


namespace NUMINAMATH_CALUDE_overall_gain_percent_l1199_119997

/-- Calculates the overall gain percent after applying two discounts -/
theorem overall_gain_percent (M : ℝ) (M_pos : M > 0) : 
  let cost_price := 0.64 * M
  let price_after_first_discount := 0.86 * M
  let final_price := 0.9 * price_after_first_discount
  let gain := final_price - cost_price
  let gain_percent := (gain / cost_price) * 100
  ∃ ε > 0, |gain_percent - 20.94| < ε :=
by sorry

end NUMINAMATH_CALUDE_overall_gain_percent_l1199_119997


namespace NUMINAMATH_CALUDE_amy_total_score_l1199_119986

/-- Calculates the total score for Amy's video game performance --/
def total_score (treasure_points enemy_points : ℕ)
                (level1_treasures level1_enemies : ℕ)
                (level2_enemies : ℕ) : ℕ :=
  let level1_score := treasure_points * level1_treasures + enemy_points * level1_enemies
  let level2_score := enemy_points * level2_enemies * 2
  level1_score + level2_score

/-- Theorem stating that Amy's total score is 154 points --/
theorem amy_total_score :
  total_score 4 10 6 3 5 = 154 :=
by sorry

end NUMINAMATH_CALUDE_amy_total_score_l1199_119986


namespace NUMINAMATH_CALUDE_unique_gcd_triplet_l1199_119932

theorem unique_gcd_triplet :
  ∃! (x y z : ℕ),
    (∃ (a b c : ℕ), x = Nat.gcd a b ∧ y = Nat.gcd b c ∧ z = Nat.gcd c a) ∧
    x ∈ ({6, 8, 12, 18, 24} : Set ℕ) ∧
    y ∈ ({14, 20, 28, 44, 56} : Set ℕ) ∧
    z ∈ ({5, 15, 18, 27, 42} : Set ℕ) ∧
    x = 8 ∧ y = 14 ∧ z = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_gcd_triplet_l1199_119932


namespace NUMINAMATH_CALUDE_negative_half_power_times_two_power_l1199_119988

theorem negative_half_power_times_two_power : (-0.5)^2016 * 2^2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_half_power_times_two_power_l1199_119988


namespace NUMINAMATH_CALUDE_f_is_odd_g_sum_one_l1199_119976

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the given conditions
axiom func_property : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_one_zero : f 1 = 0

-- State the theorems to be proved
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem g_sum_one : g 1 + g (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_f_is_odd_g_sum_one_l1199_119976


namespace NUMINAMATH_CALUDE_eggs_per_basket_l1199_119962

theorem eggs_per_basket (purple_eggs teal_eggs min_eggs : ℕ) 
  (h1 : purple_eggs = 30)
  (h2 : teal_eggs = 42)
  (h3 : min_eggs = 5) :
  ∃ (n : ℕ), n ≥ min_eggs ∧ purple_eggs % n = 0 ∧ teal_eggs % n = 0 ∧ 
  ∀ (m : ℕ), m ≥ min_eggs ∧ purple_eggs % m = 0 ∧ teal_eggs % m = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l1199_119962


namespace NUMINAMATH_CALUDE_angle_bisector_coefficient_sum_l1199_119917

/-- Given a triangle ABC with vertices A = (-3, 2), B = (4, -1), and C = (-1, -5),
    the equation of the angle bisector of ∠A in the form dx + 2y + e = 0
    has coefficients d and e such that d + e equals a specific value. -/
theorem angle_bisector_coefficient_sum (d e : ℝ) : 
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (4, -1)
  let C : ℝ × ℝ := (-1, -5)
  ∃ (k : ℝ), d * A.1 + 2 * A.2 + e = k ∧
             d * B.1 + 2 * B.2 + e = 0 ∧
             d * C.1 + 2 * C.2 + e = 0 →
  d + e = sorry -- The exact value would be calculated here
:= by sorry


end NUMINAMATH_CALUDE_angle_bisector_coefficient_sum_l1199_119917


namespace NUMINAMATH_CALUDE_black_and_white_films_count_l1199_119906

-- Define variables
variable (x y : ℚ)
variable (B : ℚ)

-- Define the theorem
theorem black_and_white_films_count :
  (6 * y) / ((y / x) / 100 * B + 6 * y) = 20 / 21 →
  B = 30 * x := by
sorry

end NUMINAMATH_CALUDE_black_and_white_films_count_l1199_119906


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l1199_119984

theorem sum_of_squares_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let s₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let s₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → s₁^2 + s₂^2 = (b^2 - 2*a*c) / a^2 := by
  sorry

theorem sum_of_squares_specific_quadratic :
  let s₁ := (15 + Real.sqrt 201) / 2
  let s₂ := (15 - Real.sqrt 201) / 2
  x^2 - 15*x + 6 = 0 → s₁^2 + s₂^2 = 213 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l1199_119984


namespace NUMINAMATH_CALUDE_probability_not_snow_l1199_119923

theorem probability_not_snow (p : ℚ) (h : p = 2/5) : 1 - p = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snow_l1199_119923


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l1199_119993

/-- The path length of the center of a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 3 / π) :
  let path_length := 3 * (π * r / 2)
  path_length = 9 / 2 := by sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l1199_119993


namespace NUMINAMATH_CALUDE_division_result_l1199_119968

theorem division_result : (35 : ℝ) / 0.07 = 500 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1199_119968


namespace NUMINAMATH_CALUDE_sin_angle_equality_l1199_119964

theorem sin_angle_equality (α : Real) (h : Real.sin (π + α) = -1/2) : 
  Real.sin (4*π - α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_angle_equality_l1199_119964


namespace NUMINAMATH_CALUDE_top_three_probability_correct_l1199_119967

/-- Represents a knockout tournament with 64 teams. -/
structure Tournament :=
  (teams : Fin 64 → ℕ)
  (distinct_skills : ∀ i j, i ≠ j → teams i ≠ teams j)

/-- The probability of the top three teams finishing in order of their skill levels. -/
def top_three_probability (t : Tournament) : ℚ :=
  512 / 1953

/-- Theorem stating the probability of the top three teams finishing in order of their skill levels. -/
theorem top_three_probability_correct (t : Tournament) : 
  top_three_probability t = 512 / 1953 := by
  sorry

end NUMINAMATH_CALUDE_top_three_probability_correct_l1199_119967


namespace NUMINAMATH_CALUDE_lcm_gcd_product_10_15_l1199_119999

theorem lcm_gcd_product_10_15 : Nat.lcm 10 15 * Nat.gcd 10 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_10_15_l1199_119999


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1199_119910

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + a > 0) → (0 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1199_119910


namespace NUMINAMATH_CALUDE_alexis_dresses_l1199_119951

theorem alexis_dresses (isabella_total : ℕ) (alexis_pants : ℕ) 
  (h1 : isabella_total = 13)
  (h2 : alexis_pants = 21) : 
  3 * isabella_total - alexis_pants = 18 := by
  sorry

end NUMINAMATH_CALUDE_alexis_dresses_l1199_119951


namespace NUMINAMATH_CALUDE_prob_consonant_correct_l1199_119969

/-- The word from which letters are selected -/
def word : String := "khantkar"

/-- The number of letters in the word -/
def word_length : Nat := word.length

/-- The number of vowels in the word -/
def vowel_count : Nat := 2

/-- The number of consonants in the word -/
def consonant_count : Nat := word_length - vowel_count

/-- The probability of selecting at least one consonant when randomly choosing two letters -/
def prob_at_least_one_consonant : ℚ := 20 / 21

/-- Theorem stating that the probability of selecting at least one consonant
    when randomly choosing two letters from the word "khantkar" is 20/21 -/
theorem prob_consonant_correct :
  prob_at_least_one_consonant = 1 - (vowel_count / word_length * (vowel_count - 1) / (word_length - 1)) :=
by sorry

end NUMINAMATH_CALUDE_prob_consonant_correct_l1199_119969


namespace NUMINAMATH_CALUDE_transformation_result_l1199_119943

/-- Rotation of 180 degrees counterclockwise around a point -/
def rotate180 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

/-- Reflection about the line y = -x -/
def reflectAboutNegativeXEqualsY (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.2, point.1)

/-- The main theorem -/
theorem transformation_result (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let center : ℝ × ℝ := (1, 5)
  let transformed := reflectAboutNegativeXEqualsY (rotate180 center P)
  transformed = (7, -3) → b - a = -2 := by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l1199_119943


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l1199_119942

theorem least_perimeter_triangle (a b c : ℕ) : 
  a = 45 → b = 53 → c > 0 → 
  (a + b > c) → (a + c > b) → (b + c > a) →
  ∀ x : ℕ, (x > 0 ∧ (a + x > b) ∧ (b + x > a) ∧ (a + b > x)) → (a + b + c ≤ a + b + x) →
  a + b + c = 107 := by
sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l1199_119942


namespace NUMINAMATH_CALUDE_min_additional_teddy_bears_l1199_119965

def teddy_bears : ℕ := 37
def row_size : ℕ := 8

theorem min_additional_teddy_bears :
  let next_multiple := ((teddy_bears + row_size - 1) / row_size) * row_size
  next_multiple - teddy_bears = 3 := by
sorry

end NUMINAMATH_CALUDE_min_additional_teddy_bears_l1199_119965


namespace NUMINAMATH_CALUDE_revenue_change_l1199_119971

theorem revenue_change (revenue_1995 : ℝ) : 
  let revenue_1996 := revenue_1995 * 1.2
  let revenue_1997 := revenue_1996 * 0.8
  (revenue_1995 - revenue_1997) / revenue_1995 * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_revenue_change_l1199_119971


namespace NUMINAMATH_CALUDE_tank_depth_l1199_119934

theorem tank_depth (length width : ℝ) (cost_per_sqm total_cost : ℝ) (d : ℝ) :
  length = 25 →
  width = 12 →
  cost_per_sqm = 0.3 →
  total_cost = 223.2 →
  cost_per_sqm * (length * width + 2 * (length * d) + 2 * (width * d)) = total_cost →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_tank_depth_l1199_119934


namespace NUMINAMATH_CALUDE_pen_pencil_cost_total_cost_is_13_l1199_119987

/-- The total cost of a pen and a pencil, where the pen costs $9 more than the pencil and the pencil costs $2. -/
theorem pen_pencil_cost : ℕ → ℕ → ℕ
  | pencil_cost, pen_extra_cost =>
    let pen_cost := pencil_cost + pen_extra_cost
    pencil_cost + pen_cost

/-- Proof that the total cost of a pen and a pencil is $13, given the conditions. -/
theorem total_cost_is_13 : pen_pencil_cost 2 9 = 13 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_cost_total_cost_is_13_l1199_119987


namespace NUMINAMATH_CALUDE_expression_evaluation_l1199_119924

theorem expression_evaluation : |-3| - 2 * Real.tan (π / 3) + (1 / 2)⁻¹ + Real.sqrt 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1199_119924


namespace NUMINAMATH_CALUDE_tile_arrangements_l1199_119909

/-- The number of distinguishable arrangements of tiles -/
def num_arrangements (brown purple green yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 brown, 1 purple, 2 green, and 3 yellow tiles is 420 -/
theorem tile_arrangements :
  num_arrangements 1 1 2 3 = 420 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l1199_119909


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l1199_119916

theorem multiplication_addition_equality : 15 * 35 + 45 * 15 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l1199_119916


namespace NUMINAMATH_CALUDE_value_of_x_l1199_119955

theorem value_of_x (w y z : ℚ) (h1 : w = 45) (h2 : z = 2 * w) (h3 : y = (1 / 6) * z) : (1 / 3) * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1199_119955


namespace NUMINAMATH_CALUDE_game_winner_l1199_119948

/-- Given a game with three players and three cards, prove who received q marbles in the first round -/
theorem game_winner (p q r : ℕ) (total_rounds : ℕ) : 
  0 < p → p < q → q < r →
  total_rounds > 1 →
  total_rounds * (p + q + r) = 39 →
  2 * p + r = 10 →
  2 * q + p = 9 →
  q = 4 →
  (∃ (x : ℕ), x = total_rounds ∧ x = 3) →
  (∃ (player : String), player = "A" ∧ 
    (∀ (other : String), other ≠ "A" → 
      (other = "B" → (∃ (y : ℕ), y = r ∧ y = 8)) ∧ 
      (other = "C" → (∃ (z : ℕ), z = p ∧ z = 1)))) :=
by sorry

end NUMINAMATH_CALUDE_game_winner_l1199_119948


namespace NUMINAMATH_CALUDE_expression_evaluation_l1199_119940

theorem expression_evaluation :
  (2 * Real.sqrt 2 - Real.pi) ^ 0 - 4 * Real.cos (60 * π / 180) + |Real.sqrt 2 - 2| - Real.sqrt 18 = 1 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1199_119940


namespace NUMINAMATH_CALUDE_total_go_stones_l1199_119927

theorem total_go_stones (white_stones black_stones : ℕ) : 
  white_stones = 954 →
  white_stones = black_stones + 468 →
  white_stones + black_stones = 1440 :=
by
  sorry

end NUMINAMATH_CALUDE_total_go_stones_l1199_119927


namespace NUMINAMATH_CALUDE_scale_division_l1199_119922

/-- Given a scale of length 7 feet and 12 inches divided into 4 equal parts,
    the length of each part is 24 inches. -/
theorem scale_division (scale_length_feet : ℕ) (scale_length_inches : ℕ) (num_parts : ℕ) :
  scale_length_feet = 7 →
  scale_length_inches = 12 →
  num_parts = 4 →
  (scale_length_feet * 12 + scale_length_inches) / num_parts = 24 :=
by sorry

end NUMINAMATH_CALUDE_scale_division_l1199_119922


namespace NUMINAMATH_CALUDE_square_sum_or_product_l1199_119902

theorem square_sum_or_product (a b c : ℕ+) (p : ℕ) :
  a + b = b * (a - c) →
  c + 1 = p^2 →
  Nat.Prime p →
  (∃ k : ℕ, (a + b : ℕ) = k^2) ∨ (∃ k : ℕ, (a * b : ℕ) = k^2) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_or_product_l1199_119902


namespace NUMINAMATH_CALUDE_solution_range_l1199_119937

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x ≤ 1 ∧ 3^x = a^2 + 2*a) → 
  (a ∈ Set.Icc (-3) (-2) ∪ Set.Ioo 0 1) :=
sorry

end NUMINAMATH_CALUDE_solution_range_l1199_119937


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_y_l1199_119933

theorem min_value_of_x_plus_y (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 9 / x + 1 / y = 1) : 
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 9 / x₀ + 1 / y₀ = 1 ∧ x₀ + y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_y_l1199_119933


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_18_l1199_119929

def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem largest_four_digit_sum_18 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 18 → n ≤ 9720 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_18_l1199_119929
