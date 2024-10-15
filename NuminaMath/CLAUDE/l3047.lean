import Mathlib

namespace NUMINAMATH_CALUDE_R_value_when_S_is_5_l3047_304742

/-- Given that R = gS^2 - 6 and R = 15 when S = 3, prove that R = 157/3 when S = 5 -/
theorem R_value_when_S_is_5 (g : ℚ) :
  (∃ R, R = g * 3^2 - 6 ∧ R = 15) →
  g * 5^2 - 6 = 157 / 3 := by
sorry

end NUMINAMATH_CALUDE_R_value_when_S_is_5_l3047_304742


namespace NUMINAMATH_CALUDE_savings_difference_l3047_304748

def original_value : ℝ := 20000

def discount_scheme_1 (x : ℝ) : ℝ :=
  x * (1 - 0.3) * (1 - 0.1) - 800

def discount_scheme_2 (x : ℝ) : ℝ :=
  x * (1 - 0.25) * (1 - 0.2) - 1000

theorem savings_difference :
  discount_scheme_1 original_value - discount_scheme_2 original_value = 800 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l3047_304748


namespace NUMINAMATH_CALUDE_bicycle_discount_proof_l3047_304765

theorem bicycle_discount_proof (original_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) : 
  original_price = 200 →
  discount1 = 0.60 →
  discount2 = 0.20 →
  discount3 = 0.10 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 57.60 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_discount_proof_l3047_304765


namespace NUMINAMATH_CALUDE_smaller_circles_radius_l3047_304766

/-- Given a central circle of radius 2 and 4 identical smaller circles
    touching the central circle and each other, the radius of each smaller circle is 6. -/
theorem smaller_circles_radius (r : ℝ) : r = 6 :=
  by
  -- Define the relationship between the radii
  have h1 : (2 + r)^2 + (2 + r)^2 = (2*r)^2 :=
    sorry
  -- Solve the resulting equation
  have h2 : r^2 - 4*r - 4 = 0 :=
    sorry
  -- Apply the quadratic formula and choose the positive solution
  sorry

end NUMINAMATH_CALUDE_smaller_circles_radius_l3047_304766


namespace NUMINAMATH_CALUDE_rod_pieces_count_l3047_304713

/-- The length of the rod in meters -/
def rod_length : ℝ := 38.25

/-- The length of each piece in centimeters -/
def piece_length : ℝ := 85

/-- The number of pieces that can be cut from the rod -/
def num_pieces : ℕ := 45

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

theorem rod_pieces_count : 
  ⌊(rod_length * meters_to_cm) / piece_length⌋ = num_pieces := by
  sorry

end NUMINAMATH_CALUDE_rod_pieces_count_l3047_304713


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l3047_304790

/-- For any positive real number a not equal to 1, 
    the function f(x) = a^(x-3) + 1 passes through the point (3, 2) -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) + 1
  f 3 = 2 := by sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l3047_304790


namespace NUMINAMATH_CALUDE_range_of_p_l3047_304724

def h (x : ℝ) : ℝ := 2 * x + 1

def p (x : ℝ) : ℝ := h (h (h x))

theorem range_of_p :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → -1 ≤ p x ∧ p x ≤ 31 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l3047_304724


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3047_304743

/-- A cubic function with parameters m and n -/
def f (m n x : ℝ) : ℝ := x^3 + m*x^2 + n*x

/-- The derivative of f with respect to x -/
def f' (m n x : ℝ) : ℝ := 3*x^2 + 2*m*x + n

theorem cubic_function_properties (m n : ℝ) :
  (∀ x, f' m n x ≤ f' m n 1) →
  (f' m n 1 = 0 ∧ ∃! (a b : ℝ), a ≠ b ∧ 
    ∃ (t : ℝ), f m n t = a*t + (1 - a) ∧
    f m n t = b*t + (1 - b)) →
  (m < -3 ∧ m = -3) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3047_304743


namespace NUMINAMATH_CALUDE_fraction_simplification_l3047_304779

theorem fraction_simplification (x : ℝ) : (2*x - 3)/4 + (5 - 4*x)/3 = (-10*x + 11)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3047_304779


namespace NUMINAMATH_CALUDE_drums_filled_per_day_l3047_304771

/-- The number of drums filled per day given the total number of drums and days -/
def drums_per_day (total_drums : ℕ) (total_days : ℕ) : ℕ :=
  total_drums / total_days

/-- Theorem stating that 90 drums filled in 6 days results in 15 drums per day -/
theorem drums_filled_per_day :
  drums_per_day 90 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_drums_filled_per_day_l3047_304771


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3047_304717

theorem geometric_sequence_first_term (a r : ℝ) : 
  (a * r^2 = 3) → (a * r^4 = 27) → (a = Real.sqrt 9 ∨ a = -Real.sqrt 9) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3047_304717


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l3047_304792

theorem quadratic_form_equivalence :
  ∀ x y : ℝ, y = x^2 - 4*x + 5 ↔ y = (x - 2)^2 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l3047_304792


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_power_l3047_304760

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_fraction_power :
  ((1 + i) / (1 - i)) ^ 1002 = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_power_l3047_304760


namespace NUMINAMATH_CALUDE_right_triangle_sides_from_median_perimeters_l3047_304764

/-- Given a right triangle with a median to the hypotenuse dividing it into two triangles
    with perimeters m and n, this theorem states the sides of the original triangle. -/
theorem right_triangle_sides_from_median_perimeters (m n : ℝ) 
  (h₁ : m > 0) (h₂ : n > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    ∃ (x : ℝ), x > 0 ∧
      x^2 = (a/2)^2 + (b/2)^2 ∧
      m = x + (c/2 - x) + b ∧
      n = x + (c/2 - x) + a ∧
      a = Real.sqrt (2*m*n) - m ∧
      b = Real.sqrt (2*m*n) - n ∧
      c = n + m - Real.sqrt (2*m*n) :=
by
  sorry


end NUMINAMATH_CALUDE_right_triangle_sides_from_median_perimeters_l3047_304764


namespace NUMINAMATH_CALUDE_combined_final_selling_price_is_630_45_l3047_304799

/-- Calculate the final selling price for an item given its cost price, profit percentage, and tax or discount percentage -/
def finalSellingPrice (costPrice : ℝ) (profitPercentage : ℝ) (taxOrDiscountPercentage : ℝ) (isTax : Bool) : ℝ :=
  let sellingPriceBeforeTaxOrDiscount := costPrice * (1 + profitPercentage)
  if isTax then
    sellingPriceBeforeTaxOrDiscount * (1 + taxOrDiscountPercentage)
  else
    sellingPriceBeforeTaxOrDiscount * (1 - taxOrDiscountPercentage)

/-- The combined final selling price for all three items -/
def combinedFinalSellingPrice : ℝ :=
  finalSellingPrice 180 0.15 0.05 true +
  finalSellingPrice 220 0.20 0.10 false +
  finalSellingPrice 130 0.25 0.08 true

theorem combined_final_selling_price_is_630_45 :
  combinedFinalSellingPrice = 630.45 := by sorry

end NUMINAMATH_CALUDE_combined_final_selling_price_is_630_45_l3047_304799


namespace NUMINAMATH_CALUDE_tennis_players_count_l3047_304762

/-- Given a sports club with the following properties:
  * There are 42 total members
  * 20 members play badminton
  * 6 members play neither badminton nor tennis
  * 7 members play both badminton and tennis
  Prove that 23 members play tennis -/
theorem tennis_players_count (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 42)
  (h_badminton : badminton = 20)
  (h_neither : neither = 6)
  (h_both : both = 7) :
  ∃ tennis : ℕ, tennis = 23 ∧ 
  tennis = total - neither - (badminton - both) :=
by sorry

end NUMINAMATH_CALUDE_tennis_players_count_l3047_304762


namespace NUMINAMATH_CALUDE_quadratic_real_root_range_l3047_304726

theorem quadratic_real_root_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 4 = 0) ∨ 
  (∃ x : ℝ, x^2 + (a-2)*x + 4 = 0) ∨ 
  (∃ x : ℝ, x^2 + 2*a*x + a^2 + 1 = 0) ↔ 
  a ≥ 4 ∨ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_range_l3047_304726


namespace NUMINAMATH_CALUDE_sector_central_angle_l3047_304707

theorem sector_central_angle (l : ℝ) (S : ℝ) (h1 : l = 6) (h2 : S = 18) : ∃ (r : ℝ) (α : ℝ), 
  S = (1/2) * l * r ∧ l = r * α ∧ α = 1 :=
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3047_304707


namespace NUMINAMATH_CALUDE_double_inequality_l3047_304783

theorem double_inequality (a b : ℝ) (h : a > b) : 2 * a > 2 * b := by
  sorry

end NUMINAMATH_CALUDE_double_inequality_l3047_304783


namespace NUMINAMATH_CALUDE_initial_number_exists_l3047_304786

theorem initial_number_exists : ∃ N : ℝ, ∃ k : ℤ, N + 69.00000000008731 = 330 * (k : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_initial_number_exists_l3047_304786


namespace NUMINAMATH_CALUDE_perimeter_ABCDEFG_l3047_304788

-- Define the points
variable (A B C D E F G : ℝ × ℝ)

-- Define the conditions
def is_equilateral (X Y Z : ℝ × ℝ) : Prop := 
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X

def is_midpoint (M X Y : ℝ × ℝ) : Prop :=
  M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- State the theorem
theorem perimeter_ABCDEFG (h1 : is_equilateral A B C)
                          (h2 : is_equilateral A D E)
                          (h3 : is_equilateral E F G)
                          (h4 : is_midpoint D A C)
                          (h5 : is_midpoint G A E)
                          (h6 : is_midpoint F E G)
                          (h7 : dist A B = 6) :
  dist A B + dist B C + dist C D + dist D E + dist E F + dist F G + dist G A = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ABCDEFG_l3047_304788


namespace NUMINAMATH_CALUDE_equation_solution_l3047_304796

theorem equation_solution (x : ℝ) : 3*x - 5 = 10*x + 9 → 4*(x + 7) = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3047_304796


namespace NUMINAMATH_CALUDE_remaining_trees_correct_l3047_304741

/-- The number of dogwood trees remaining in a park after cutting some down. -/
def remaining_trees (part1 : ℝ) (part2 : ℝ) (cut : ℝ) : ℝ :=
  part1 + part2 - cut

/-- Theorem stating that the number of remaining trees is correct. -/
theorem remaining_trees_correct (part1 : ℝ) (part2 : ℝ) (cut : ℝ) :
  remaining_trees part1 part2 cut = part1 + part2 - cut :=
by sorry

end NUMINAMATH_CALUDE_remaining_trees_correct_l3047_304741


namespace NUMINAMATH_CALUDE_final_price_in_euros_l3047_304731

-- Define the pin prices
def pin_prices : List ℝ := [23, 18, 20, 15, 25, 22, 19, 16, 24, 17]

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the exchange rate (USD to Euro)
def exchange_rate : ℝ := 0.85

-- Theorem statement
theorem final_price_in_euros :
  let original_price := pin_prices.sum
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_tax := discounted_price * (1 + sales_tax_rate)
  let final_price := price_with_tax * exchange_rate
  ∃ ε > 0, |final_price - 155.28| < ε :=
sorry

end NUMINAMATH_CALUDE_final_price_in_euros_l3047_304731


namespace NUMINAMATH_CALUDE_cube_sum_equals_one_l3047_304770

theorem cube_sum_equals_one (x y z : ℝ) 
  (h1 : x * y + y * z + z * x = x * y * z) 
  (h2 : x + y + z = 1) : 
  x^3 + y^3 + z^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_one_l3047_304770


namespace NUMINAMATH_CALUDE_tangent_count_depends_on_position_l3047_304756

/-- Represents the position of a point relative to a circle -/
inductive PointPosition
  | OnCircle
  | OutsideCircle
  | InsideCircle

/-- Represents the number of tangents that can be drawn -/
inductive TangentCount
  | Zero
  | One
  | Two

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines the position of a point relative to a circle -/
def pointPosition (c : Circle) (p : ℝ × ℝ) : PointPosition :=
  sorry

/-- Counts the number of tangents that can be drawn from a point to a circle -/
def tangentCount (c : Circle) (p : ℝ × ℝ) : TangentCount :=
  sorry

/-- Theorem: The number of tangents depends on the point's position relative to the circle -/
theorem tangent_count_depends_on_position (c : Circle) (p : ℝ × ℝ) :
  (pointPosition c p = PointPosition.OnCircle → tangentCount c p = TangentCount.One) ∧
  (pointPosition c p = PointPosition.OutsideCircle → tangentCount c p = TangentCount.Two) ∧
  (pointPosition c p = PointPosition.InsideCircle → tangentCount c p = TangentCount.Zero) :=
  sorry

end NUMINAMATH_CALUDE_tangent_count_depends_on_position_l3047_304756


namespace NUMINAMATH_CALUDE_certain_number_equation_l3047_304722

theorem certain_number_equation (x : ℝ) : 300 + 5 * x = 340 ↔ x = 8 := by sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3047_304722


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3047_304774

theorem simplify_trig_expression (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 4)) :
  Real.sqrt (1 - 2 * Real.sin (π + θ) * Real.sin ((3 * π) / 2 - θ)) = Real.cos θ - Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3047_304774


namespace NUMINAMATH_CALUDE_lemonade_price_ratio_l3047_304734

theorem lemonade_price_ratio :
  -- Define the ratio of small cups sold
  let small_ratio : ℚ := 3/5
  -- Define the ratio of large cups sold
  let large_ratio : ℚ := 1 - small_ratio
  -- Define the fraction of revenue from large cups
  let large_revenue_fraction : ℚ := 357142857142857150 / 1000000000000000000
  -- Define the price ratio of large to small cups
  let price_ratio : ℚ := large_revenue_fraction * (1 / large_ratio)
  -- The theorem
  price_ratio = 892857142857143 / 1000000000000000 :=
by sorry

end NUMINAMATH_CALUDE_lemonade_price_ratio_l3047_304734


namespace NUMINAMATH_CALUDE_sin_increasing_interval_l3047_304750

/-- The function f with given properties has (-π/12, 5π/12) as its strictly increasing interval -/
theorem sin_increasing_interval (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x + π / 6)
  (∀ x, f x > 0) →
  (∀ p, p > 0 → (∀ x, f (x + p) = f x) → p ≥ π) →
  (∃ p, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ p = π) →
  (∀ x ∈ Set.Ioo (-π/12 : ℝ) (5*π/12), StrictMono f) :=
by
  sorry

end NUMINAMATH_CALUDE_sin_increasing_interval_l3047_304750


namespace NUMINAMATH_CALUDE_zero_subset_X_l3047_304793

def X : Set ℝ := {x | x > -1}

theorem zero_subset_X : {0} ⊆ X := by sorry

end NUMINAMATH_CALUDE_zero_subset_X_l3047_304793


namespace NUMINAMATH_CALUDE_inequality_and_function_minimum_l3047_304753

-- Define the set A
def A (a : ℕ+) : Set ℝ := {x : ℝ | |x - 2| < a}

-- State the theorem
theorem inequality_and_function_minimum (a : ℕ+) 
  (h1 : (3/2 : ℝ) ∈ A a) 
  (h2 : (1/2 : ℝ) ∉ A a) :
  (a = 1) ∧ 
  (∀ x : ℝ, |x + a| + |x - 2| ≥ 3) ∧ 
  (∃ x : ℝ, |x + a| + |x - 2| = 3) := by
sorry

end NUMINAMATH_CALUDE_inequality_and_function_minimum_l3047_304753


namespace NUMINAMATH_CALUDE_f_properties_l3047_304738

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 3|

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f x < 7 ↔ -2 < x ∧ x < 5) ∧
  (∀ x : ℝ, f x - |2*x - 7| < x^2 - 2*x + Real.sqrt 26) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l3047_304738


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l3047_304785

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 21) : 
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l3047_304785


namespace NUMINAMATH_CALUDE_shaded_area_of_carpet_l3047_304761

theorem shaded_area_of_carpet (S T : ℝ) : 
  12 / S = 4 →
  S / T = 4 →
  (8 * T^2 + S^2) = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_carpet_l3047_304761


namespace NUMINAMATH_CALUDE_marble_capacity_l3047_304712

/-- 
Given:
- A small bottle with volume 20 ml can hold 40 marbles
- A larger bottle has volume 60 ml
Prove that the larger bottle can hold 120 marbles
-/
theorem marble_capacity (small_volume small_capacity large_volume : ℕ) 
  (h1 : small_volume = 20)
  (h2 : small_capacity = 40)
  (h3 : large_volume = 60) :
  (large_volume * small_capacity) / small_volume = 120 := by
  sorry

end NUMINAMATH_CALUDE_marble_capacity_l3047_304712


namespace NUMINAMATH_CALUDE_tuesday_occurs_five_times_in_august_l3047_304730

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- July of year N -/
def july : Month := sorry

/-- August of year N -/
def august : Month := sorry

/-- Counts the occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat := sorry

theorem tuesday_occurs_five_times_in_august 
  (h1 : july.days = 31)
  (h2 : countDayOccurrences july DayOfWeek.Monday = 5)
  (h3 : august.days = 30) :
  countDayOccurrences august DayOfWeek.Tuesday = 5 := by sorry

end NUMINAMATH_CALUDE_tuesday_occurs_five_times_in_august_l3047_304730


namespace NUMINAMATH_CALUDE_T_forms_three_lines_closed_region_l3047_304703

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    (4 ≤ x - 1 ∧ 4 ≤ y + 3 ∧ y + 3 < x - 1) ∨
    (4 ≤ x - 1 ∧ 4 ≤ y + 3 ∧ x - 1 < y + 3) ∨
    (4 ≤ x - 1 ∧ y + 3 < 4 ∧ y + 3 < x - 1) ∨
    (x - 1 < 4 ∧ 4 ≤ y + 3 ∧ x - 1 < y + 3) ∨
    (x - 1 ≤ y + 3 ∧ 4 < x - 1 ∧ 4 < y + 3)}

-- Define the three lines
def line1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 5 ∧ p.2 ≤ 1}
def line2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 1 ∧ p.1 ≤ 5}
def line3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 - 4 ∧ p.1 ≥ 5}

-- Theorem statement
theorem T_forms_three_lines_closed_region :
  ∃ (point : ℝ × ℝ), 
    point ∈ T ∧
    point ∈ line1 ∧ point ∈ line2 ∧ point ∈ line3 ∧
    T = line1 ∪ line2 ∪ line3 :=
sorry


end NUMINAMATH_CALUDE_T_forms_three_lines_closed_region_l3047_304703


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3047_304784

theorem polar_to_rectangular_conversion :
  let r : ℝ := 10
  let θ : ℝ := 5 * π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 5 ∧ y = -5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3047_304784


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3047_304732

theorem complex_fraction_sum (A B : ℝ) : 
  (Complex.I : ℂ) * (3 + Complex.I) = (1 + 2 * Complex.I) * (A + B * Complex.I) → 
  A + B = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3047_304732


namespace NUMINAMATH_CALUDE_number_equation_l3047_304798

theorem number_equation (x : ℝ) : 3 * x - 4 = 5 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3047_304798


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3047_304767

theorem quadratic_root_problem (a : ℝ) (k : ℝ) :
  (∃ x : ℂ, x^2 + 4*x + k = 0 ∧ x = a + 3*Complex.I) →
  k = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3047_304767


namespace NUMINAMATH_CALUDE_area_inequality_special_quadrilateral_l3047_304791

/-- A point in a 2D plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral := (A B C D : Point)

/-- Check if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculate the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (p : Point) (q : Quadrilateral) : Prop := sorry

/-- Check if a point is on a line segment between two other points -/
def isOnSegment (p : Point) (a b : Point) : Prop := sorry

/-- Check if four points form a parallelogram -/
def isParallelogram (a b c d : Point) : Prop := sorry

/-- Theorem: Area inequality for quadrilaterals with special interior point -/
theorem area_inequality_special_quadrilateral 
  (ABCD : Quadrilateral) 
  (O K L M N : Point) 
  (h_convex : isConvex ABCD)
  (h_inside : isInside O ABCD)
  (h_K : isOnSegment K ABCD.A ABCD.B)
  (h_L : isOnSegment L ABCD.B ABCD.C)
  (h_M : isOnSegment M ABCD.C ABCD.D)
  (h_N : isOnSegment N ABCD.D ABCD.A)
  (h_OKBL : isParallelogram O K ABCD.B L)
  (h_OMDN : isParallelogram O M ABCD.D N)
  (S := area ABCD)
  (S1 := area (Quadrilateral.mk O N ABCD.A K))
  (S2 := area (Quadrilateral.mk O L ABCD.C M)) :
  Real.sqrt S ≥ Real.sqrt S1 + Real.sqrt S2 := by
  sorry

end NUMINAMATH_CALUDE_area_inequality_special_quadrilateral_l3047_304791


namespace NUMINAMATH_CALUDE_connie_markers_count_l3047_304720

/-- The number of red markers Connie has -/
def red_markers : ℕ := 41

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 64

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers

theorem connie_markers_count : total_markers = 105 := by
  sorry

end NUMINAMATH_CALUDE_connie_markers_count_l3047_304720


namespace NUMINAMATH_CALUDE_sum_integer_chord_lengths_equals_40_l3047_304763

/-- A circle with center O and a point P inside it. -/
structure CircleWithPoint where
  O : Point    -- Center of the circle
  P : Point    -- Point inside the circle
  radius : ℝ   -- Radius of the circle
  OP : ℝ       -- Distance between O and P

/-- The sum of all possible integer chord lengths passing through P -/
def sumIntegerChordLengths (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem sum_integer_chord_lengths_equals_40 (c : CircleWithPoint) 
  (h_radius : c.radius = 5)
  (h_OP : c.OP = 4) :
  sumIntegerChordLengths c = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_integer_chord_lengths_equals_40_l3047_304763


namespace NUMINAMATH_CALUDE_cube_root_two_not_rational_plus_sqrt_l3047_304769

theorem cube_root_two_not_rational_plus_sqrt (a b c : ℚ) (hc : c > 0) :
  (a + b * Real.sqrt c) ^ 3 ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_cube_root_two_not_rational_plus_sqrt_l3047_304769


namespace NUMINAMATH_CALUDE_solve_sqrt_equation_l3047_304775

theorem solve_sqrt_equation (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_sqrt_equation_l3047_304775


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3047_304746

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((x + a) / x + (x - a) / x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3047_304746


namespace NUMINAMATH_CALUDE_only_minute_hand_rotates_l3047_304718

-- Define the set of objects
inductive Object
  | MinuteHand
  | Boat
  | Car

-- Define the motion types
inductive Motion
  | Rotation
  | Translation
  | Combined

-- Function to determine the motion type of an object
def motionType (obj : Object) : Motion :=
  match obj with
  | Object.MinuteHand => Motion.Rotation
  | Object.Boat => Motion.Combined
  | Object.Car => Motion.Combined

-- Theorem statement
theorem only_minute_hand_rotates :
  ∀ (obj : Object), motionType obj = Motion.Rotation ↔ obj = Object.MinuteHand :=
by sorry

end NUMINAMATH_CALUDE_only_minute_hand_rotates_l3047_304718


namespace NUMINAMATH_CALUDE_oatmeal_cookies_count_l3047_304754

def cookies_per_bag : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def number_of_baggies : ℕ := 6

def total_cookies : ℕ := cookies_per_bag * number_of_baggies

theorem oatmeal_cookies_count :
  total_cookies - chocolate_chip_cookies = 41 :=
by sorry

end NUMINAMATH_CALUDE_oatmeal_cookies_count_l3047_304754


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l3047_304781

theorem lcm_hcf_problem (A B : ℕ) (h1 : A = 330) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 30) :
  B = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l3047_304781


namespace NUMINAMATH_CALUDE_mans_walking_speed_l3047_304716

theorem mans_walking_speed (woman_speed : ℝ) (passing_wait_time : ℝ) (catch_up_time : ℝ) :
  woman_speed = 25 →
  passing_wait_time = 5 / 60 →
  catch_up_time = 20 / 60 →
  ∃ (man_speed : ℝ),
    woman_speed * passing_wait_time = man_speed * (passing_wait_time + catch_up_time) ∧
    man_speed = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_mans_walking_speed_l3047_304716


namespace NUMINAMATH_CALUDE_race_length_is_90_l3047_304702

/-- The race between Nicky and Cristina -/
structure Race where
  head_start : ℝ
  cristina_speed : ℝ
  nicky_speed : ℝ
  catch_up_time : ℝ

/-- Calculate the length of the race -/
def race_length (r : Race) : ℝ :=
  r.nicky_speed * r.catch_up_time

/-- Theorem stating that the race length is 90 meters -/
theorem race_length_is_90 (r : Race)
  (h1 : r.head_start = 12)
  (h2 : r.cristina_speed = 5)
  (h3 : r.nicky_speed = 3)
  (h4 : r.catch_up_time = 30) :
  race_length r = 90 := by
  sorry

#check race_length_is_90

end NUMINAMATH_CALUDE_race_length_is_90_l3047_304702


namespace NUMINAMATH_CALUDE_repaired_shoes_duration_is_one_year_l3047_304736

/-- The duration for which the repaired shoes last, in years -/
def repaired_shoes_duration : ℝ := 1

/-- The cost to repair the used shoes, in dollars -/
def repair_cost : ℝ := 10.50

/-- The cost of new shoes, in dollars -/
def new_shoes_cost : ℝ := 30.00

/-- The duration for which new shoes last, in years -/
def new_shoes_duration : ℝ := 2

/-- The percentage by which the average cost per year of new shoes 
    is greater than the cost of repairing used shoes -/
def cost_difference_percentage : ℝ := 42.857142857142854

theorem repaired_shoes_duration_is_one_year :
  repaired_shoes_duration = 
    (repair_cost * (1 + cost_difference_percentage / 100)) / 
    (new_shoes_cost / new_shoes_duration) :=
by sorry

end NUMINAMATH_CALUDE_repaired_shoes_duration_is_one_year_l3047_304736


namespace NUMINAMATH_CALUDE_ellen_chairs_count_l3047_304757

/-- The number of chairs Ellen bought at a garage sale -/
def num_chairs : ℕ := 180 / 15

/-- The cost of each chair in dollars -/
def chair_cost : ℕ := 15

/-- The total amount Ellen spent in dollars -/
def total_spent : ℕ := 180

theorem ellen_chairs_count :
  num_chairs = 12 ∧ chair_cost * num_chairs = total_spent :=
sorry

end NUMINAMATH_CALUDE_ellen_chairs_count_l3047_304757


namespace NUMINAMATH_CALUDE_additional_cars_needed_min_additional_cars_l3047_304745

def current_cars : ℕ := 35
def cars_per_row : ℕ := 8

theorem additional_cars_needed : 
  ∃ (n : ℕ), n > 0 ∧ (current_cars + n) % cars_per_row = 0 ∧
  ∀ (m : ℕ), m < n → (current_cars + m) % cars_per_row ≠ 0 := by
  sorry

theorem min_additional_cars : 
  ∃ (n : ℕ), n = 5 ∧ (current_cars + n) % cars_per_row = 0 ∧
  ∀ (m : ℕ), m < n → (current_cars + m) % cars_per_row ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_additional_cars_needed_min_additional_cars_l3047_304745


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3047_304795

theorem geometric_sequence_problem (x : ℝ) : 
  x > 0 → 
  (∃ r : ℝ, r > 0 ∧ x = 40 * r ∧ (10/3) = x * r) → 
  x = (20 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3047_304795


namespace NUMINAMATH_CALUDE_gmat_scores_l3047_304705

theorem gmat_scores (u v w : ℝ) 
  (h_order : u > v ∧ v > w)
  (h_avg : u - w = (u + v + w) / 3)
  (h_diff : u - v = 2 * (v - w)) :
  v / u = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_gmat_scores_l3047_304705


namespace NUMINAMATH_CALUDE_cone_axial_angle_when_max_section_twice_axial_l3047_304700

/-- Represents a right circular cone -/
structure RightCircularCone where
  vertex : Point
  axialAngle : ℝ

/-- Represents a cross-section of a cone -/
structure ConeSection where
  cone : RightCircularCone
  angle : ℝ

/-- The area of a cone section -/
def sectionArea (s : ConeSection) : ℝ := sorry

/-- The maximum area cross-section of a cone -/
def maxSectionArea (c : RightCircularCone) : ℝ := sorry

/-- The axial cross-section of a cone -/
def axialSection (c : RightCircularCone) : ConeSection := sorry

theorem cone_axial_angle_when_max_section_twice_axial 
  (c : RightCircularCone) :
  maxSectionArea c = 2 * sectionArea (axialSection c) →
  c.axialAngle = 120 * π / 180 := by sorry

end NUMINAMATH_CALUDE_cone_axial_angle_when_max_section_twice_axial_l3047_304700


namespace NUMINAMATH_CALUDE_even_iff_period_two_l3047_304759

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define the condition f(1+x) = f(1-x)
def symmetry_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

-- Define an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a function with period 2
def has_period_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

-- Theorem statement
theorem even_iff_period_two (f : ℝ → ℝ) (h : symmetry_condition f) :
  is_even f ↔ has_period_two f :=
sorry

end NUMINAMATH_CALUDE_even_iff_period_two_l3047_304759


namespace NUMINAMATH_CALUDE_cityF_greatest_increase_l3047_304737

/-- Represents a city with population data for 1970 and 1980 --/
structure City where
  name : String
  pop1970 : Nat
  pop1980 : Nat

/-- Calculates the percentage increase in population from 1970 to 1980 --/
def percentageIncrease (city : City) : Rat :=
  (city.pop1980 - city.pop1970 : Rat) / city.pop1970 * 100

/-- The set of cities in the region --/
def cities : Finset City := sorry

/-- City F with its population data --/
def cityF : City := { name := "F", pop1970 := 30000, pop1980 := 45000 }

/-- City G with its population data --/
def cityG : City := { name := "G", pop1970 := 60000, pop1980 := 75000 }

/-- Combined City H (including I) with its population data --/
def cityH : City := { name := "H", pop1970 := 60000, pop1980 := 70000 }

/-- City J with its population data --/
def cityJ : City := { name := "J", pop1970 := 90000, pop1980 := 120000 }

/-- Theorem stating that City F had the greatest percentage increase --/
theorem cityF_greatest_increase : 
  ∀ city ∈ cities, percentageIncrease cityF ≥ percentageIncrease city :=
sorry

end NUMINAMATH_CALUDE_cityF_greatest_increase_l3047_304737


namespace NUMINAMATH_CALUDE_f_3_equals_130_l3047_304744

def f (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 2 * x^2 - x + 7

theorem f_3_equals_130 : f 3 = 130 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_130_l3047_304744


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_area_l3047_304728

/-- Given a rectangle with dimensions 32 cm * 10 cm, if the area of a square is five times
    the area of this rectangle, then the perimeter of the square is 160 cm. -/
theorem square_perimeter_from_rectangle_area : 
  let rectangle_length : ℝ := 32
  let rectangle_width : ℝ := 10
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  square_side * 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_area_l3047_304728


namespace NUMINAMATH_CALUDE_trapezoid_area_l3047_304787

/-- The area of a trapezoid with height 2a, one base 5a, and the other base 4a, is 9a² -/
theorem trapezoid_area (a : ℝ) : 
  let height : ℝ := 2 * a
  let base1 : ℝ := 5 * a
  let base2 : ℝ := 4 * a
  let area : ℝ := (height * (base1 + base2)) / 2
  area = 9 * a^2 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3047_304787


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3047_304733

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I) / z = I) : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3047_304733


namespace NUMINAMATH_CALUDE_smallest_positive_equivalent_angle_proof_l3047_304704

/-- The smallest positive angle (in degrees) with the same terminal side as -2002° -/
def smallest_positive_equivalent_angle : ℝ := 158

theorem smallest_positive_equivalent_angle_proof :
  ∃ (k : ℤ), smallest_positive_equivalent_angle = -2002 + 360 * k ∧
  0 < smallest_positive_equivalent_angle ∧
  smallest_positive_equivalent_angle < 360 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_equivalent_angle_proof_l3047_304704


namespace NUMINAMATH_CALUDE_hunting_season_quarter_year_l3047_304715

/-- Represents the hunting scenario -/
structure HuntingScenario where
  hunts_per_month : ℕ
  deers_per_hunt : ℕ
  deer_weight : ℕ
  kept_fraction : ℚ
  kept_weight : ℕ

/-- Calculates the fraction of the year the hunting season lasts -/
def hunting_season_fraction (scenario : HuntingScenario) : ℚ :=
  let total_catch := scenario.kept_weight / scenario.kept_fraction
  let catch_per_hunt := scenario.deers_per_hunt * scenario.deer_weight
  let hunts_per_year := total_catch / catch_per_hunt
  let months_of_hunting := hunts_per_year / scenario.hunts_per_month
  months_of_hunting / 12

/-- Theorem stating that for the given scenario, the hunting season lasts 1/4 of the year -/
theorem hunting_season_quarter_year (scenario : HuntingScenario) 
  (h1 : scenario.hunts_per_month = 6)
  (h2 : scenario.deers_per_hunt = 2)
  (h3 : scenario.deer_weight = 600)
  (h4 : scenario.kept_fraction = 1/2)
  (h5 : scenario.kept_weight = 10800) :
  hunting_season_fraction scenario = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_hunting_season_quarter_year_l3047_304715


namespace NUMINAMATH_CALUDE_f_has_unique_zero_a_lower_bound_l3047_304773

noncomputable section

def f (x : ℝ) : ℝ := -1/2 * Real.log x + 2/(x+1)

theorem f_has_unique_zero :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

theorem a_lower_bound (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (1/Real.exp 1) 1 →
    ∀ t : ℝ, t ∈ Set.Icc (1/2) 2 →
      f x ≥ t^3 - t^2 - 2*a*t + 2) →
  a ≥ 5/4 :=
sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_a_lower_bound_l3047_304773


namespace NUMINAMATH_CALUDE_min_operations_to_check_square_l3047_304772

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define an operation (either measurement or comparison)
inductive Operation
  | Measure : Point → Point → Operation
  | Compare : ℝ → ℝ → Operation

-- Define a function to check if a quadrilateral is a square
def isSquare (q : Quadrilateral) : Prop := sorry

-- Define a function that returns the list of operations needed to check if a quadrilateral is a square
def operationsToCheckSquare (q : Quadrilateral) : List Operation := sorry

-- Theorem statement
theorem min_operations_to_check_square (q : Quadrilateral) :
  (isSquare q ↔ operationsToCheckSquare q = [
    Operation.Measure q.A q.B,
    Operation.Measure q.B q.C,
    Operation.Measure q.C q.D,
    Operation.Measure q.D q.A,
    Operation.Measure q.A q.C,
    Operation.Measure q.B q.D,
    Operation.Compare (q.A.x - q.B.x) (q.B.x - q.C.x),
    Operation.Compare (q.B.x - q.C.x) (q.C.x - q.D.x),
    Operation.Compare (q.C.x - q.D.x) (q.D.x - q.A.x),
    Operation.Compare (q.A.x - q.C.x) (q.B.x - q.D.x)
  ]) :=
sorry

end NUMINAMATH_CALUDE_min_operations_to_check_square_l3047_304772


namespace NUMINAMATH_CALUDE_sufficient_condition_for_equation_l3047_304739

theorem sufficient_condition_for_equation (a : ℝ) (f g h : ℝ → ℝ) 
  (ha : a > 1)
  (h_sum_nonneg : ∀ x, f x + g x + h x ≥ 0)
  (h_common_root : ∃ x₀, f x₀ = 0 ∧ g x₀ = 0 ∧ h x₀ = 0) :
  ∃ x, a^(f x) + a^(g x) + a^(h x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_equation_l3047_304739


namespace NUMINAMATH_CALUDE_ninth_grade_class_problem_l3047_304797

theorem ninth_grade_class_problem (total : ℕ) (math : ℕ) (foreign : ℕ) (science_only : ℕ) (math_and_foreign : ℕ) :
  total = 120 →
  math = 85 →
  foreign = 75 →
  science_only = 20 →
  math_and_foreign = 40 →
  ∃ (math_only : ℕ), math_only = 45 ∧ math_only = math - math_and_foreign :=
by sorry

end NUMINAMATH_CALUDE_ninth_grade_class_problem_l3047_304797


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3047_304740

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (4^11 + 6^13)) → 
  2 ∣ (4^11 + 6^13) ∧ 
  ∀ p : ℕ, Nat.Prime p → p ∣ (4^11 + 6^13) → p ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3047_304740


namespace NUMINAMATH_CALUDE_odot_inequality_equivalence_l3047_304747

-- Define the operation ⊙
def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem odot_inequality_equivalence :
  ∀ x : ℝ, odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_odot_inequality_equivalence_l3047_304747


namespace NUMINAMATH_CALUDE_train_speed_l3047_304780

/-- Given a train of length 125 metres that takes 7.5 seconds to pass a pole, 
    its speed is 60 km/hr. -/
theorem train_speed (train_length : Real) (time_to_pass : Real) 
  (h1 : train_length = 125) 
  (h2 : time_to_pass = 7.5) : 
  (train_length / time_to_pass) * 3.6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3047_304780


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l3047_304727

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (10*x + 2) * (10*x + 6)^2 * (5*x + 1) = 24 * k) ∧
  (∀ (m : ℤ), m > 24 → ∃ (y : ℤ), Odd y ∧ ¬(∃ (l : ℤ), (10*y + 2) * (10*y + 6)^2 * (5*y + 1) = m * l)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l3047_304727


namespace NUMINAMATH_CALUDE_average_monthly_sales_l3047_304710

def monthly_sales : List ℝ := [150, 120, 80, 100, 90, 130]

theorem average_monthly_sales :
  (List.sum monthly_sales) / (List.length monthly_sales) = 111.67 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l3047_304710


namespace NUMINAMATH_CALUDE_julia_car_rental_cost_l3047_304778

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, 
    number of days, and miles driven. -/
def carRentalCost (dailyRate : ℚ) (mileageRate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  dailyRate * days + mileageRate * miles

/-- Theorem stating that the total cost for Julia's car rental is $215 -/
theorem julia_car_rental_cost :
  carRentalCost 30 0.25 3 500 = 215 := by
  sorry

end NUMINAMATH_CALUDE_julia_car_rental_cost_l3047_304778


namespace NUMINAMATH_CALUDE_min_value_theorem_l3047_304709

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1 = 2*m + n) :
  (1/m + 2/n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 1 = 2*m₀ + n₀ ∧ 1/m₀ + 2/n₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3047_304709


namespace NUMINAMATH_CALUDE_parenthesized_results_l3047_304721

def original_expression : ℚ := 72 / 9 - 3 * 2

def parenthesized_expressions : List ℚ := [
  (72 / 9 - 3) * 2,
  72 / (9 - 3) * 2,
  72 / ((9 - 3) * 2)
]

theorem parenthesized_results :
  original_expression = 2 →
  (parenthesized_expressions.toFinset = {6, 10, 24}) ∧
  (parenthesized_expressions.length = 3) :=
by sorry

end NUMINAMATH_CALUDE_parenthesized_results_l3047_304721


namespace NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l3047_304782

theorem negation_of_exists_lt_is_forall_ge (p : Prop) : 
  (¬ (∃ x : ℝ, x^2 + 2*x < 0)) ↔ (∀ x : ℝ, x^2 + 2*x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l3047_304782


namespace NUMINAMATH_CALUDE_vectors_collinear_l3047_304725

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 0)
def c : ℝ × ℝ := (2, 4)

theorem vectors_collinear : ∃ k : ℝ, k • a = b + c := by sorry

end NUMINAMATH_CALUDE_vectors_collinear_l3047_304725


namespace NUMINAMATH_CALUDE_hyperbola_transformation_l3047_304749

/-- Given a hyperbola with equation x^2/4 - y^2/5 = 1, 
    prove that the standard equation of a hyperbola with the same foci as vertices 
    and perpendicular asymptotes is x^2/9 - y^2/9 = 1 -/
theorem hyperbola_transformation (x y : ℝ) : 
  (∃ (a b : ℝ), x^2/a - y^2/b = 1 ∧ a = 4 ∧ b = 5) →
  (∃ (c : ℝ), x^2/c - y^2/c = 1 ∧ c = 9) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_transformation_l3047_304749


namespace NUMINAMATH_CALUDE_original_recipe_yield_l3047_304751

/-- Represents a cookie recipe -/
structure Recipe where
  butter : ℝ
  cookies : ℝ

/-- Proves that given a recipe that uses 4 pounds of butter, 
    if 1 pound of butter makes 4 dozen cookies, 
    then the original recipe makes 16 dozen cookies. -/
theorem original_recipe_yield 
  (original : Recipe) 
  (h1 : original.butter = 4) 
  (h2 : ∃ (scaled : Recipe), scaled.butter = 1 ∧ scaled.cookies = 4) : 
  original.cookies = 16 := by
sorry

end NUMINAMATH_CALUDE_original_recipe_yield_l3047_304751


namespace NUMINAMATH_CALUDE_field_trip_girls_fraction_l3047_304701

theorem field_trip_girls_fraction (g : ℚ) (h1 : g > 0) : 
  let b := 2 * g
  let girls_on_trip := (4 / 5) * g
  let boys_on_trip := (3 / 4) * b
  let total_on_trip := girls_on_trip + boys_on_trip
  girls_on_trip / total_on_trip = 8 / 23 := by
sorry

end NUMINAMATH_CALUDE_field_trip_girls_fraction_l3047_304701


namespace NUMINAMATH_CALUDE_parabola_p_value_l3047_304723

/-- Represents a parabola with equation y^2 = 2px and directrix x = -2 -/
structure Parabola where
  p : ℝ
  eq : ∀ x y : ℝ, y^2 = 2 * p * x
  directrix : ∀ x : ℝ, x = -2

/-- The value of p for the given parabola is 4 -/
theorem parabola_p_value (par : Parabola) : par.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_p_value_l3047_304723


namespace NUMINAMATH_CALUDE_paper_distribution_l3047_304752

theorem paper_distribution (total_sheets : ℕ) (num_printers : ℕ) 
  (h1 : total_sheets = 221) (h2 : num_printers = 31) :
  (total_sheets / num_printers : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_paper_distribution_l3047_304752


namespace NUMINAMATH_CALUDE_theresa_required_hours_l3047_304789

/-- The average number of hours Theresa needs to work per week over 4 weeks -/
def required_average : ℝ := 12

/-- The total number of weeks -/
def total_weeks : ℕ := 4

/-- The minimum total hours Theresa needs to work -/
def minimum_total_hours : ℝ := 50

/-- The hours Theresa worked in the first week -/
def first_week_hours : ℝ := 15

/-- The hours Theresa worked in the second week -/
def second_week_hours : ℝ := 8

/-- The number of remaining weeks -/
def remaining_weeks : ℕ := 2

theorem theresa_required_hours :
  let total_worked := first_week_hours + second_week_hours
  let remaining_hours := minimum_total_hours - total_worked
  (remaining_hours / remaining_weeks : ℝ) = 13.5 ∧
  remaining_hours ≥ required_average * remaining_weeks := by
  sorry

end NUMINAMATH_CALUDE_theresa_required_hours_l3047_304789


namespace NUMINAMATH_CALUDE_jean_gives_480_l3047_304776

/-- The amount Jean gives away to her grandchildren in a year -/
def total_amount_given (num_grandchildren : ℕ) (cards_per_grandchild : ℕ) (amount_per_card : ℕ) : ℕ :=
  num_grandchildren * cards_per_grandchild * amount_per_card

/-- Proof that Jean gives away $480 to her grandchildren in a year -/
theorem jean_gives_480 :
  total_amount_given 3 2 80 = 480 :=
by sorry

end NUMINAMATH_CALUDE_jean_gives_480_l3047_304776


namespace NUMINAMATH_CALUDE_triangle_ABC_c_value_l3047_304758

/-- Triangle ABC with vertices A(0, 4), B(3, 0), and C(c, 6) has area 7 and 0 < c < 3 -/
def triangle_ABC (c : ℝ) : Prop :=
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ × ℝ := (c, 6)
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  (area = 7) ∧ (0 < c) ∧ (c < 3)

/-- If triangle ABC satisfies the given conditions, then c = 2 -/
theorem triangle_ABC_c_value :
  ∀ c : ℝ, triangle_ABC c → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_c_value_l3047_304758


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3047_304711

theorem min_perimeter_triangle (a b c : ℕ) : 
  a = 52 → b = 76 → c > 0 → 
  (a + b > c) → (a + c > b) → (b + c > a) →
  (∀ x : ℕ, x > 0 → (a + b > x) → (a + x > b) → (b + x > a) → c ≤ x) →
  a + b + c = 153 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3047_304711


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3047_304719

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (1/a + 1/b) ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3047_304719


namespace NUMINAMATH_CALUDE_system_solution_l3047_304729

theorem system_solution (a : ℝ) (h : a > 0) :
  ∃ (x y : ℝ), 
    (a^(7*x) * a^(15*y) = (a^19)^(1/2)) ∧ 
    ((a^(25*y))^(1/3) / (a^(13*x))^(1/2) = a^(1/12)) ∧
    x = 1/2 ∧ y = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3047_304729


namespace NUMINAMATH_CALUDE_total_votes_l3047_304735

/-- Given that Ben and Matt received votes in the ratio 2:3 and Ben got 24 votes,
    prove that the total number of votes cast is 60. -/
theorem total_votes (ben_votes : ℕ) (matt_votes : ℕ) : 
  ben_votes = 24 → 
  ben_votes * 3 = matt_votes * 2 → 
  ben_votes + matt_votes = 60 := by
sorry

end NUMINAMATH_CALUDE_total_votes_l3047_304735


namespace NUMINAMATH_CALUDE_arrangement_exists_l3047_304777

/-- A type representing a 10x10 table of real numbers -/
def Table := Fin 10 → Fin 10 → ℝ

/-- A predicate to check if two cells are adjacent in the table -/
def adjacent (i j k l : Fin 10) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ 
  (i = k ∧ l.val + 1 = j.val) ∨ 
  (j = l ∧ i.val + 1 = k.val) ∨ 
  (j = l ∧ k.val + 1 = i.val)

/-- The main theorem statement -/
theorem arrangement_exists (S : Finset ℝ) (h : S.card = 100) :
  ∃ (f : Table), 
    (∀ x ∈ S, ∃ i j, f i j = x) ∧ 
    (∀ i j k l, adjacent i j k l → |f i j - f k l| ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_exists_l3047_304777


namespace NUMINAMATH_CALUDE_two_integer_tangent_lengths_l3047_304755

def circle_circumference : ℝ := 10

def is_valid_arc_length (x : ℝ) : Prop :=
  0 < x ∧ x < circle_circumference

theorem two_integer_tangent_lengths :
  ∃ (t₁ t₂ : ℕ), t₁ ≠ t₂ ∧
  (∀ m : ℕ, is_valid_arc_length m →
    (∃ n : ℝ, is_valid_arc_length n ∧
      m + n = circle_circumference ∧
      (t₁ : ℝ)^2 = m * n ∨ (t₂ : ℝ)^2 = m * n)) ∧
  (∀ t : ℕ, (∃ m : ℕ, is_valid_arc_length m ∧
    (∃ n : ℝ, is_valid_arc_length n ∧
      m + n = circle_circumference ∧
      (t : ℝ)^2 = m * n)) →
    t = t₁ ∨ t = t₂) :=
by sorry

end NUMINAMATH_CALUDE_two_integer_tangent_lengths_l3047_304755


namespace NUMINAMATH_CALUDE_mary_work_hours_l3047_304794

/-- Mary's weekly work schedule and earnings -/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ
  tue_thu_hours : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Theorem stating Mary's work hours on Monday, Wednesday, and Friday -/
theorem mary_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.tue_thu_hours = 5)
  (h2 : schedule.weekly_earnings = 407)
  (h3 : schedule.hourly_rate = 11)
  (h4 : schedule.hourly_rate * (3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours) = schedule.weekly_earnings) :
  schedule.mon_wed_fri_hours = 9 := by
  sorry


end NUMINAMATH_CALUDE_mary_work_hours_l3047_304794


namespace NUMINAMATH_CALUDE_dice_probability_l3047_304708

def standard_die : Finset ℕ := Finset.range 6
def eight_sided_die : Finset ℕ := Finset.range 8

def prob_not_one (die : Finset ℕ) : ℚ :=
  (die.filter (· ≠ 1)).card / die.card

theorem dice_probability : 
  (prob_not_one standard_die)^2 * (prob_not_one eight_sided_die) = 175/288 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l3047_304708


namespace NUMINAMATH_CALUDE_largest_average_is_17_multiples_l3047_304768

def upper_bound : ℕ := 100810

def average_of_multiples (n : ℕ) : ℚ :=
  let last_multiple := upper_bound - (upper_bound % n)
  (n + last_multiple) / 2

theorem largest_average_is_17_multiples :
  average_of_multiples 17 > average_of_multiples 11 ∧
  average_of_multiples 17 > average_of_multiples 13 ∧
  average_of_multiples 17 > average_of_multiples 19 :=
by sorry

end NUMINAMATH_CALUDE_largest_average_is_17_multiples_l3047_304768


namespace NUMINAMATH_CALUDE_arkansas_game_sales_l3047_304714

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts : ℕ := 172

/-- The number of t-shirts sold during the Texas Tech game -/
def texas_tech_shirts : ℕ := 186 - arkansas_shirts

/-- The revenue per t-shirt in dollars -/
def revenue_per_shirt : ℕ := 78

/-- The total number of t-shirts sold during both games -/
def total_shirts : ℕ := 186

/-- The revenue from the Texas Tech game in dollars -/
def texas_tech_revenue : ℕ := 1092

theorem arkansas_game_sales : 
  arkansas_shirts = 172 ∧ 
  texas_tech_shirts + arkansas_shirts = total_shirts ∧
  texas_tech_shirts * revenue_per_shirt = texas_tech_revenue :=
sorry

end NUMINAMATH_CALUDE_arkansas_game_sales_l3047_304714


namespace NUMINAMATH_CALUDE_coffee_per_donut_l3047_304706

/-- Proves that the number of ounces of coffee needed per donut is 2, given the specified conditions. -/
theorem coffee_per_donut (ounces_per_pot : ℕ) (cost_per_pot : ℕ) (dozen_donuts : ℕ) (total_coffee_cost : ℕ) :
  ounces_per_pot = 12 →
  cost_per_pot = 3 →
  dozen_donuts = 3 →
  total_coffee_cost = 18 →
  (total_coffee_cost / cost_per_pot * ounces_per_pot) / (dozen_donuts * 12) = 2 :=
by sorry

end NUMINAMATH_CALUDE_coffee_per_donut_l3047_304706
