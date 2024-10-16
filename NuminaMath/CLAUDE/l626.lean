import Mathlib

namespace NUMINAMATH_CALUDE_arrangement_schemes_l626_62623

theorem arrangement_schemes (teachers students : ℕ) (h1 : teachers = 2) (h2 : students = 4) : 
  (teachers.choose 1) * (students.choose 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_schemes_l626_62623


namespace NUMINAMATH_CALUDE_triangle_inequality_l626_62644

theorem triangle_inequality (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l626_62644


namespace NUMINAMATH_CALUDE_sum_inequality_l626_62665

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) : a + b + c ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l626_62665


namespace NUMINAMATH_CALUDE_combined_monthly_profit_is_90_l626_62622

/-- Represents a book with its purchase price, sale price, and months held before sale -/
structure Book where
  purchase_price : ℕ
  sale_price : ℕ
  months_held : ℕ

/-- Calculates the monthly profit for a single book -/
def monthly_profit (book : Book) : ℚ :=
  (book.sale_price - book.purchase_price : ℚ) / book.months_held

/-- Calculates the combined monthly rate of profit for a list of books -/
def combined_monthly_profit (books : List Book) : ℚ :=
  books.map monthly_profit |>.sum

theorem combined_monthly_profit_is_90 (books : List Book) : combined_monthly_profit books = 90 :=
  by
  have h1 : books = [
    { purchase_price := 50, sale_price := 90, months_held := 1 },
    { purchase_price := 120, sale_price := 150, months_held := 2 },
    { purchase_price := 75, sale_price := 110, months_held := 0 }
  ] := by sorry
  rw [h1]
  simp [combined_monthly_profit, monthly_profit]
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_combined_monthly_profit_is_90_l626_62622


namespace NUMINAMATH_CALUDE_intersection_point_unique_l626_62678

/-- The first line equation: 2x + 3y - 7 = 0 -/
def line1 (x y : ℝ) : Prop := 2 * x + 3 * y - 7 = 0

/-- The second line equation: 5x - y - 9 = 0 -/
def line2 (x y : ℝ) : Prop := 5 * x - y - 9 = 0

/-- The intersection point (2, 1) -/
def intersection_point : ℝ × ℝ := (2, 1)

/-- Theorem stating that (2, 1) is the unique intersection point of the two lines -/
theorem intersection_point_unique :
  (∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2) ∧
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l626_62678


namespace NUMINAMATH_CALUDE_distance_between_points_l626_62691

/-- The distance between two points in 3D space is the square root of the sum of the squares of the differences of their coordinates. -/
theorem distance_between_points (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) :
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2) = Real.sqrt 185 ↔
  x₁ = -2 ∧ y₁ = 4 ∧ z₁ = 1 ∧ x₂ = 3 ∧ y₂ = -8 ∧ z₂ = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l626_62691


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l626_62641

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  -- We don't need to define any specific properties here, as we're only interested in the general structure

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

theorem rectangular_prism_sum (rp : RectangularPrism) : 
  num_faces rp + num_edges rp + num_vertices rp = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l626_62641


namespace NUMINAMATH_CALUDE_cyclic_number_property_l626_62685

def digit_set (n : ℕ) : Set ℕ :=
  {d | ∃ k, n = d + 10 * k ∨ k = d + 10 * n}

def has_same_digits (a b : ℕ) : Prop :=
  digit_set a = digit_set b

theorem cyclic_number_property (n : ℕ) (h : n = 142857) :
  ∀ k : ℕ, k ≥ 1 → k ≤ 6 → has_same_digits n (n * k) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_number_property_l626_62685


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l626_62687

theorem quadratic_inequality_solution_set (m : ℝ) (hm : m > 1) :
  {x : ℝ | x^2 + (m - 1) * x - m ≥ 0} = {x : ℝ | x ≤ -m ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l626_62687


namespace NUMINAMATH_CALUDE_kate_change_l626_62647

-- Define the prices of items
def gum_price : ℚ := 89 / 100
def chocolate_price : ℚ := 125 / 100
def chips_price : ℚ := 249 / 100

-- Define the sales tax rate
def sales_tax_rate : ℚ := 6 / 100

-- Define the amount Kate gave to the clerk
def amount_given : ℚ := 10

-- Theorem statement
theorem kate_change (gum : ℚ) (chocolate : ℚ) (chips : ℚ) (tax_rate : ℚ) (given : ℚ) :
  gum = gum_price →
  chocolate = chocolate_price →
  chips = chips_price →
  tax_rate = sales_tax_rate →
  given = amount_given →
  ∃ (change : ℚ), change = 509 / 100 ∧ 
    change = given - (gum + chocolate + chips + (gum + chocolate + chips) * tax_rate) :=
by sorry

end NUMINAMATH_CALUDE_kate_change_l626_62647


namespace NUMINAMATH_CALUDE_max_ratio_squared_l626_62683

theorem max_ratio_squared (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b)
  (hx : 0 ≤ x ∧ x < a) (hy : 0 ≤ y ∧ y < b) (hx2 : x ≤ 2*a/3)
  (heq : a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b - y)^2) :
  (∃ ρ : ℝ, ∀ a' b' : ℝ, a' / b' ≤ ρ ∧ ρ^2 = 9/5) :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l626_62683


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l626_62666

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l626_62666


namespace NUMINAMATH_CALUDE_average_sale_is_5500_l626_62672

def sales : List ℕ := [5435, 5927, 5855, 6230, 5562]
def sixth_month_sale : ℕ := 3991
def num_months : ℕ := 6

theorem average_sale_is_5500 :
  (sales.sum + sixth_month_sale) / num_months = 5500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_is_5500_l626_62672


namespace NUMINAMATH_CALUDE_all_equations_have_one_negative_one_positive_root_l626_62668

-- Define the equations
def equation1 (x : ℝ) : Prop := 4 * x^2 - 6 = 34
def equation2 (x : ℝ) : Prop := (3*x-2)^2 = (x+1)^2
def equation3 (x : ℝ) : Prop := (x^2-12).sqrt = (2*x-2).sqrt

-- Define the property of having one negative and one positive root
def has_one_negative_one_positive_root (f : ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x ∧ f y

-- Theorem statement
theorem all_equations_have_one_negative_one_positive_root :
  has_one_negative_one_positive_root equation1 ∧
  has_one_negative_one_positive_root equation2 ∧
  has_one_negative_one_positive_root equation3 :=
sorry

end NUMINAMATH_CALUDE_all_equations_have_one_negative_one_positive_root_l626_62668


namespace NUMINAMATH_CALUDE_cubic_repeated_root_condition_l626_62607

/-- A cubic polynomial with a repeated root -/
def has_repeated_root (b : ℝ) : Prop :=
  ∃ x : ℝ, (b * x^3 + 15 * x^2 + 9 * x + 2 = 0) ∧
           (3 * b * x^2 + 30 * x + 9 = 0)

/-- Theorem stating that if a nonzero b makes the cubic have a repeated root, then b = 100 -/
theorem cubic_repeated_root_condition (b : ℝ) (hb : b ≠ 0) :
  has_repeated_root b → b = 100 := by
  sorry

end NUMINAMATH_CALUDE_cubic_repeated_root_condition_l626_62607


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l626_62659

theorem largest_prime_factor_of_3913 : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ 3913 ∧ ∀ (q : Nat), Nat.Prime q → q ∣ 3913 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l626_62659


namespace NUMINAMATH_CALUDE_reciprocal_lcm_24_221_l626_62618

theorem reciprocal_lcm_24_221 :
  let a : ℕ := 24
  let b : ℕ := 221
  Nat.gcd a b = 1 →
  (1 : ℚ) / (Nat.lcm a b) = 1 / 5304 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_lcm_24_221_l626_62618


namespace NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l626_62626

theorem three_fourths_to_fifth_power : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l626_62626


namespace NUMINAMATH_CALUDE_tan_plus_reciprocal_l626_62643

theorem tan_plus_reciprocal (θ : Real) (h : Real.sin (2 * θ) = 2/3) :
  Real.tan θ + (Real.tan θ)⁻¹ = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_reciprocal_l626_62643


namespace NUMINAMATH_CALUDE_shot_put_surface_area_l626_62653

/-- The surface area of a sphere with diameter 5 inches is 25π square inches. -/
theorem shot_put_surface_area :
  let diameter : ℝ := 5
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shot_put_surface_area_l626_62653


namespace NUMINAMATH_CALUDE_probability_below_curve_probability_is_one_third_l626_62638

/-- The probability that a randomly chosen point in the unit square falls below the curve y = x^2 is 1/3 -/
theorem probability_below_curve : Real → Prop := λ p =>
  let curve := λ x : Real => x^2
  let unit_square_area := 1
  let area_below_curve := ∫ x in (0 : Real)..1, curve x
  p = area_below_curve / unit_square_area ∧ p = 1/3

/-- The main theorem stating the probability is 1/3 -/
theorem probability_is_one_third : ∃ p : Real, probability_below_curve p := by
  sorry

end NUMINAMATH_CALUDE_probability_below_curve_probability_is_one_third_l626_62638


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l626_62680

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}
def B : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l626_62680


namespace NUMINAMATH_CALUDE_stock_price_uniqueness_l626_62654

theorem stock_price_uniqueness : ¬∃ (k m : ℕ), (117/100)^k * (83/100)^m = 1 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_uniqueness_l626_62654


namespace NUMINAMATH_CALUDE_negation_of_exists_equals_sin_l626_62661

theorem negation_of_exists_equals_sin (x : ℝ) : 
  (¬ ∃ x : ℝ, x = Real.sin x) ↔ (∀ x : ℝ, x ≠ Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_equals_sin_l626_62661


namespace NUMINAMATH_CALUDE_fraction_inequality_l626_62616

theorem fraction_inequality (x : ℝ) : 
  x ∈ Set.Icc (-2 : ℝ) 2 →
  (8 * x - 3 > 2 + 5 * x ↔ 5 / 3 < x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l626_62616


namespace NUMINAMATH_CALUDE_cos_180_degrees_l626_62674

theorem cos_180_degrees : Real.cos (π) = -1 := by sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l626_62674


namespace NUMINAMATH_CALUDE_sequence_sum_inequality_l626_62697

theorem sequence_sum_inequality (a : ℕ → ℝ) (S : ℕ → ℝ) (x : ℝ) :
  a 1 = 1 →
  (∀ n, 2 * a (n + 1) = a n) →
  (∀ n : ℕ, ∀ t ∈ Set.Icc (-1 : ℝ) 1, x^2 + t*x + 1 > S n) →
  x ∈ Set.Iic (((-1:ℝ) - Real.sqrt 5) / 2) ∪ Set.Ici ((1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_inequality_l626_62697


namespace NUMINAMATH_CALUDE_sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_proof_l626_62698

/-- The sum of the exterior angles of a pentagon is 360 degrees. -/
theorem sum_exterior_angles_pentagon : ℝ :=
  360

/-- A pentagon has 5 sides. -/
def pentagon_sides : ℕ := 5

/-- The sum of the exterior angles of any polygon with n sides. -/
def sum_exterior_angles (n : ℕ) : ℝ := 360

theorem sum_exterior_angles_pentagon_proof :
  sum_exterior_angles pentagon_sides = sum_exterior_angles_pentagon :=
by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_proof_l626_62698


namespace NUMINAMATH_CALUDE_equation_solution_l626_62606

theorem equation_solution :
  ∃ m : ℝ, (m - 5) ^ 3 = (1 / 16)⁻¹ ∧ m = 5 + 2 ^ (4 / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l626_62606


namespace NUMINAMATH_CALUDE_function_inequality_l626_62694

/-- Given a function f(x) = axe^x where a ≠ 0 and a ≥ 4/e^2, 
    prove that f(x)/(x+1) - (x+1)ln(x) > 0 for x > 0 -/
theorem function_inequality (a : ℝ) (h1 : a ≠ 0) (h2 : a ≥ 4 / Real.exp 2) :
  ∀ x > 0, (a * x * Real.exp x) / (x + 1) - (x + 1) * Real.log x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l626_62694


namespace NUMINAMATH_CALUDE_sum_four_digit_even_distinct_mod_1000_l626_62656

/-- A function that generates all four-digit positive integers with distinct even digits -/
def fourDigitEvenDistinct : List Nat := sorry

/-- The sum of all four-digit positive integers with distinct even digits -/
def sumFourDigitEvenDistinct : Nat := (fourDigitEvenDistinct.map id).sum

/-- Theorem: The sum of all four-digit positive integers with distinct even digits,
    when divided by 1000, leaves a remainder of 560 -/
theorem sum_four_digit_even_distinct_mod_1000 :
  sumFourDigitEvenDistinct % 1000 = 560 := by sorry

end NUMINAMATH_CALUDE_sum_four_digit_even_distinct_mod_1000_l626_62656


namespace NUMINAMATH_CALUDE_pastry_solution_l626_62619

/-- Represents the number of pastries each person has -/
structure Pastries where
  calvin : ℕ
  phoebe : ℕ
  frank : ℕ
  grace : ℕ

/-- The conditions of the pastry problem -/
def pastry_problem (p : Pastries) : Prop :=
  p.grace = 30 ∧
  p.calvin > p.frank ∧
  p.phoebe > p.frank ∧
  p.calvin = p.grace - 5 ∧
  p.phoebe = p.grace - 5 ∧
  p.calvin + p.phoebe + p.frank + p.grace = 97

/-- The theorem stating the solution to the pastry problem -/
theorem pastry_solution (p : Pastries) (h : pastry_problem p) :
  p.calvin - p.frank = 8 ∧ p.phoebe - p.frank = 8 := by
  sorry

end NUMINAMATH_CALUDE_pastry_solution_l626_62619


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l626_62635

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) * (Real.sqrt 9 / Real.sqrt 10) = 
  (3 * Real.sqrt 1050) / 120 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l626_62635


namespace NUMINAMATH_CALUDE_average_string_length_l626_62692

theorem average_string_length :
  let string1 : ℚ := 2
  let string2 : ℚ := 5
  let string3 : ℚ := 3
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_string_length_l626_62692


namespace NUMINAMATH_CALUDE_square_root_division_l626_62614

theorem square_root_division (x : ℝ) : (Real.sqrt 1936) / x = 4 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_root_division_l626_62614


namespace NUMINAMATH_CALUDE_larger_number_problem_l626_62648

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 1395)
  (h2 : L = 6 * S + 15) :
  L = 1671 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l626_62648


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l626_62631

theorem min_sum_with_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - x * y = 0) :
  x + y ≥ 9 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 4 * y₀ - x₀ * y₀ = 0 ∧ x₀ + y₀ = 9 := by
sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l626_62631


namespace NUMINAMATH_CALUDE_kristin_green_beans_count_l626_62621

/-- Represents the number of vegetables a person has -/
structure VegetableCount where
  carrots : ℕ
  cucumbers : ℕ
  bellPeppers : ℕ
  greenBeans : ℕ

/-- The problem statement -/
theorem kristin_green_beans_count 
  (jaylen : VegetableCount)
  (kristin : VegetableCount)
  (h1 : jaylen.carrots = 5)
  (h2 : jaylen.cucumbers = 2)
  (h3 : jaylen.bellPeppers = 2 * kristin.bellPeppers)
  (h4 : jaylen.greenBeans = kristin.greenBeans / 2 - 3)
  (h5 : jaylen.carrots + jaylen.cucumbers + jaylen.bellPeppers + jaylen.greenBeans = 18)
  (h6 : kristin.bellPeppers = 2) :
  kristin.greenBeans = 20 := by
  sorry

end NUMINAMATH_CALUDE_kristin_green_beans_count_l626_62621


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l626_62640

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l626_62640


namespace NUMINAMATH_CALUDE_peppers_required_per_day_l626_62655

/-- Represents the number of jalapeno pepper strips per sandwich -/
def strips_per_sandwich : ℕ := 4

/-- Represents the number of slices one jalapeno pepper can make -/
def slices_per_pepper : ℕ := 8

/-- Represents the time in minutes between serving each sandwich -/
def minutes_per_sandwich : ℕ := 5

/-- Represents the number of hours in a workday -/
def hours_per_day : ℕ := 8

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating the number of jalapeno peppers required for an 8-hour day -/
theorem peppers_required_per_day : 
  (hours_per_day * minutes_per_hour / minutes_per_sandwich) * 
  (strips_per_sandwich : ℚ) / slices_per_pepper = 48 := by
  sorry


end NUMINAMATH_CALUDE_peppers_required_per_day_l626_62655


namespace NUMINAMATH_CALUDE_park_pairings_l626_62667

/-- The number of unique pairings in a group of 12 people where two specific individuals do not interact -/
theorem park_pairings (n : ℕ) (h : n = 12) : 
  (n.choose 2) - 1 = 65 := by
  sorry

end NUMINAMATH_CALUDE_park_pairings_l626_62667


namespace NUMINAMATH_CALUDE_congruence_modulo_nine_l626_62611

theorem congruence_modulo_nine : 
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -4981 [ZMOD 9] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_congruence_modulo_nine_l626_62611


namespace NUMINAMATH_CALUDE_polygon_coloring_l626_62657

/-- Given a regular 103-sided polygon with 79 red vertices and 24 blue vertices,
    A is the number of pairs of adjacent red vertices and
    B is the number of pairs of adjacent blue vertices. -/
theorem polygon_coloring (A B : ℕ) :
  (∀ i : ℕ, 0 ≤ i ∧ i ≤ 23 → (A = 55 + i ∧ B = i)) ∧
  (B = 14 →
    (Nat.choose 23 10 * Nat.choose 78 9) / 14 =
      (Nat.choose 23 9 * Nat.choose 78 9) / 10) :=
by sorry

end NUMINAMATH_CALUDE_polygon_coloring_l626_62657


namespace NUMINAMATH_CALUDE_gasoline_cost_calculation_l626_62662

theorem gasoline_cost_calculation
  (cost_per_litre : ℝ)
  (distance_per_litre : ℝ)
  (distance_to_travel : ℝ)
  (cost_per_litre_positive : 0 < cost_per_litre)
  (distance_per_litre_positive : 0 < distance_per_litre) :
  cost_per_litre * distance_to_travel / distance_per_litre =
  cost_per_litre * (distance_to_travel / distance_per_litre) :=
by sorry

#check gasoline_cost_calculation

end NUMINAMATH_CALUDE_gasoline_cost_calculation_l626_62662


namespace NUMINAMATH_CALUDE_complex_multiplication_l626_62609

theorem complex_multiplication :
  let i : ℂ := Complex.I
  (3 - 4 * i) * (-7 + 6 * i) = 3 + 46 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l626_62609


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l626_62646

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 1638 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 1638 → 
  A + B + C ≤ X + Y + Z → A + B + C ≤ 126 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l626_62646


namespace NUMINAMATH_CALUDE_shaded_fraction_is_one_twelfth_l626_62601

/-- A point in a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A rectangle defined by its top-left and bottom-right corners -/
structure Rectangle where
  topLeft : GridPoint
  bottomRight : GridPoint

/-- The 6x6 grid -/
def gridSize : ℕ := 6

/-- The rectangle in question -/
def shadedRectangle : Rectangle := {
  topLeft := { x := 2, y := 5 }
  bottomRight := { x := 3, y := 2 }
}

/-- Calculate the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ :=
  (r.bottomRight.x - r.topLeft.x) * (r.topLeft.y - r.bottomRight.y)

/-- Calculate the area of the entire grid -/
def gridArea : ℕ := gridSize * gridSize

/-- The fraction of the grid occupied by the shaded rectangle -/
def shadedFraction : ℚ :=
  (rectangleArea shadedRectangle : ℚ) / gridArea

/-- Theorem: The shaded fraction is equal to 1/12 -/
theorem shaded_fraction_is_one_twelfth : shadedFraction = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_is_one_twelfth_l626_62601


namespace NUMINAMATH_CALUDE_solve_star_equation_l626_62624

-- Define the ☆ operator
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem solve_star_equation : 
  ∃! x : ℝ, star 3 x = -9 ∧ x = -3 :=
sorry

end NUMINAMATH_CALUDE_solve_star_equation_l626_62624


namespace NUMINAMATH_CALUDE_root_sum_squares_l626_62682

theorem root_sum_squares (p q r s : ℂ) : 
  (p^4 - 24*p^3 + 50*p^2 - 26*p + 7 = 0) →
  (q^4 - 24*q^3 + 50*q^2 - 26*q + 7 = 0) →
  (r^4 - 24*r^3 + 50*r^2 - 26*r + 7 = 0) →
  (s^4 - 24*s^3 + 50*s^2 - 26*s + 7 = 0) →
  (p+q)^2 + (q+r)^2 + (r+s)^2 + (s+p)^2 + (p+r)^2 + (q+s)^2 = 1052 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_squares_l626_62682


namespace NUMINAMATH_CALUDE_decagon_diagonals_l626_62637

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l626_62637


namespace NUMINAMATH_CALUDE_ab_value_l626_62628

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l626_62628


namespace NUMINAMATH_CALUDE_odd_function_inverse_range_l626_62634

/-- An odd function f defined on ℝ with specific properties -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∃ a b, 0 < a ∧ a < 1 ∧ ∀ x > 0, f x = a^x + b)

/-- Theorem stating the range of b for which f has an inverse -/
theorem odd_function_inverse_range (f : ℝ → ℝ) (h : OddFunction f) 
  (h_inv : Function.Injective f) : 
  ∃ a b, (0 < a ∧ a < 1) ∧ (∀ x > 0, f x = a^x + b) ∧ (b ≤ -1 ∨ b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inverse_range_l626_62634


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l626_62620

theorem circle_line_distance_range (b : ℝ) : 
  (∃! (p q : ℝ × ℝ), (p.1 - 1)^2 + (p.2 - 1)^2 = 4 ∧ 
                      (q.1 - 1)^2 + (q.2 - 1)^2 = 4 ∧ 
                      p ≠ q ∧
                      (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 4 → 
                        (|y - (x + b)| / Real.sqrt 2 = 1 → (x, y) = p ∨ (x, y) = q))) →
  b ∈ Set.union (Set.Ioo (-3 * Real.sqrt 2) (-Real.sqrt 2)) 
                (Set.Ioo (Real.sqrt 2) (3 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l626_62620


namespace NUMINAMATH_CALUDE_max_m_value_l626_62604

theorem max_m_value (b a m : ℝ) (hb : b > 0) :
  (∀ a, (b - (a - 2))^2 + (Real.log b - (a - 1))^2 ≥ m^2 - m) →
  m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l626_62604


namespace NUMINAMATH_CALUDE_sqrt_problem_l626_62673

theorem sqrt_problem (x : ℝ) (h : (Real.sqrt x - 8) / 13 = 6) :
  ⌊(x^2 - 45) / 23⌋ = 2380011 := by sorry

end NUMINAMATH_CALUDE_sqrt_problem_l626_62673


namespace NUMINAMATH_CALUDE_circle_properties_l626_62608

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 4*m = 0

-- Define the range of m for which the equation represents a circle
def is_circle (m : ℝ) : Prop :=
  m < 5/4

-- Define the symmetric circle when m = 1
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + (y + 2)^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Theorem statement
theorem circle_properties :
  (∀ m, is_circle m ↔ ∀ x y, ∃ r > 0, circle_equation x y m ↔ (x - 1)^2 + (y + 2)^2 = r^2) ∧
  (∀ x y, symmetric_circle x y →
    (∃ d, d = 2*Real.sqrt 2 + 1 ∧ ∀ x' y', line x' y' → d ≥ Real.sqrt ((x - x')^2 + (y - y')^2)) ∧
    (∃ d, d = 2*Real.sqrt 2 - 1 ∧ ∀ x' y', line x' y' → d ≤ Real.sqrt ((x - x')^2 + (y - y')^2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l626_62608


namespace NUMINAMATH_CALUDE_equal_probabilities_l626_62679

/-- Represents a box containing colored balls -/
structure Box where
  red_balls : ℕ
  green_balls : ℕ

/-- Represents the state of both boxes -/
structure BoxState where
  red_box : Box
  green_box : Box

/-- Initial state of the boxes -/
def initial_state : BoxState :=
  { red_box := { red_balls := 100, green_balls := 0 },
    green_box := { red_balls := 0, green_balls := 100 } }

/-- State after first transfer (8 red balls from red to green box) -/
def first_transfer (state : BoxState) : BoxState :=
  { red_box := { red_balls := state.red_box.red_balls - 8, green_balls := state.red_box.green_balls },
    green_box := { red_balls := state.green_box.red_balls + 8, green_balls := state.green_box.green_balls } }

/-- Probability of drawing a specific color from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => box.red_balls / (box.red_balls + box.green_balls)
  | "green" => box.green_balls / (box.red_balls + box.green_balls)
  | _ => 0

/-- Theorem stating the equality of probabilities after transfers and mixing -/
theorem equal_probabilities (final_state : BoxState) 
    (h1 : final_state.red_box.green_balls + final_state.green_box.green_balls = 100) 
    (h2 : final_state.red_box.red_balls + final_state.green_box.red_balls = 100) :
    prob_draw final_state.red_box "green" = prob_draw final_state.green_box "red" :=
  sorry


end NUMINAMATH_CALUDE_equal_probabilities_l626_62679


namespace NUMINAMATH_CALUDE_fourth_power_sum_l626_62695

theorem fourth_power_sum (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x^2 + y^2 + z^2 = 5) 
  (h3 : x^3 + y^3 + z^3 = 8) : 
  x^4 + y^4 + z^4 = 113/6 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l626_62695


namespace NUMINAMATH_CALUDE_supplementary_angle_theorem_l626_62684

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define addition for Angle
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let extraDegrees := totalMinutes / 60
  { degrees := a.degrees + b.degrees + extraDegrees,
    minutes := totalMinutes % 60 }

-- Define subtraction for Angle
def Angle.sub (a b : Angle) : Angle :=
  let totalMinutes := (a.degrees * 60 + a.minutes) - (b.degrees * 60 + b.minutes)
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60 }

-- Define the given complementary angle
def complementaryAngle : Angle := { degrees := 54, minutes := 38 }

-- Define 90 degrees
def rightAngle : Angle := { degrees := 90, minutes := 0 }

-- Define 180 degrees
def straightAngle : Angle := { degrees := 180, minutes := 0 }

-- Theorem statement
theorem supplementary_angle_theorem :
  let angle := Angle.sub rightAngle complementaryAngle
  Angle.sub straightAngle angle = { degrees := 144, minutes := 38 } := by sorry

end NUMINAMATH_CALUDE_supplementary_angle_theorem_l626_62684


namespace NUMINAMATH_CALUDE_import_tax_percentage_l626_62660

/-- The import tax percentage calculation problem -/
theorem import_tax_percentage 
  (total_value : ℝ)
  (tax_threshold : ℝ)
  (tax_paid : ℝ)
  (h1 : total_value = 2570)
  (h2 : tax_threshold = 1000)
  (h3 : tax_paid = 109.90) :
  (tax_paid / (total_value - tax_threshold)) * 100 = 7 := by
sorry

end NUMINAMATH_CALUDE_import_tax_percentage_l626_62660


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l626_62651

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i / (1 - i)
  (z.im : ℝ) = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l626_62651


namespace NUMINAMATH_CALUDE_milk_production_l626_62664

/-- Milk production calculation -/
theorem milk_production
  (a b c d e f : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
  (h_efficiency : 0 < f ∧ f ≤ 100)
  (h_initial : b = (a * c) * (b / (a * c)))  -- Initial production rate
  : (d * e) * ((b / (a * c)) * (f / 100)) = b * d * e * f / (100 * a * c) :=
by sorry

#check milk_production

end NUMINAMATH_CALUDE_milk_production_l626_62664


namespace NUMINAMATH_CALUDE_special_numbers_l626_62605

/-- A two-digit number is equal to three times the product of its digits -/
def is_special_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ (a b : ℕ), n = 10 * a + b ∧ n = 3 * a * b

/-- The only two-digit numbers that are equal to three times the product of their digits are 15 and 24 -/
theorem special_numbers : ∀ n : ℕ, is_special_number n ↔ (n = 15 ∨ n = 24) :=
sorry

end NUMINAMATH_CALUDE_special_numbers_l626_62605


namespace NUMINAMATH_CALUDE_cricket_bat_profit_l626_62681

theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 900 ∧ profit_percentage = 50 → 
  ∃ (cost_price : ℝ) (profit : ℝ),
    profit = selling_price - cost_price ∧
    profit_percentage = (profit / cost_price) * 100 ∧
    profit = 300 :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_l626_62681


namespace NUMINAMATH_CALUDE_set_equation_solution_l626_62630

theorem set_equation_solution (p q r : ℝ) : 
  let A : Set ℝ := {x | x^2 - p*x - 2 = 0}
  let B : Set ℝ := {x | x^2 + q*x + r = 0}
  (A ∪ B = {-2, 1, 5} ∧ A ∩ B = {-2}) → p + q + r = -14 := by
sorry

end NUMINAMATH_CALUDE_set_equation_solution_l626_62630


namespace NUMINAMATH_CALUDE_bread_needed_for_field_trip_bread_needed_proof_l626_62610

/-- Calculates the number of pieces of bread needed for a field trip --/
theorem bread_needed_for_field_trip 
  (sandwiches_per_student : ℕ) 
  (students_per_group : ℕ) 
  (number_of_groups : ℕ) 
  (bread_per_sandwich : ℕ) : ℕ :=
  let total_students := students_per_group * number_of_groups
  let total_sandwiches := total_students * sandwiches_per_student
  let total_bread := total_sandwiches * bread_per_sandwich
  total_bread

/-- Proves that 120 pieces of bread are needed for the field trip --/
theorem bread_needed_proof :
  bread_needed_for_field_trip 2 6 5 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_bread_needed_for_field_trip_bread_needed_proof_l626_62610


namespace NUMINAMATH_CALUDE_function_symmetry_l626_62686

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the conditions
def passes_through_point (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Theorem statement
theorem function_symmetry 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : passes_through_point (log a) 2 (-1)) 
  (f : ℝ → ℝ) 
  (h4 : symmetric_wrt_y_eq_x f (log a)) : 
  f = fun x ↦ (1/2)^x := by sorry

end NUMINAMATH_CALUDE_function_symmetry_l626_62686


namespace NUMINAMATH_CALUDE_two_xy_value_l626_62669

theorem two_xy_value (x y : ℝ) : 
  y = Real.sqrt (2 * x - 5) + Real.sqrt (5 - 2 * x) - 3 → 2 * x * y = -15 := by
  sorry

end NUMINAMATH_CALUDE_two_xy_value_l626_62669


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l626_62652

theorem consecutive_integers_sum (x y z : ℤ) : 
  (x = y + 1) → 
  (y = z + 1) → 
  (x > y) → 
  (y > z) → 
  (2 * x + 3 * y + 3 * z = 5 * y + 11) → 
  z = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l626_62652


namespace NUMINAMATH_CALUDE_system_of_inequalities_l626_62613

theorem system_of_inequalities (x : ℝ) : 
  (-3 * x^2 + 7 * x + 6 > 0 ∧ 4 * x - 4 * x^2 > -3) ↔ (-1/2 < x ∧ x < 3/2) := by
sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l626_62613


namespace NUMINAMATH_CALUDE_father_twice_son_age_l626_62639

/-- Represents the ages of a father and son --/
structure Ages where
  sonPast : ℕ
  fatherPast : ℕ
  sonNow : ℕ
  fatherNow : ℕ

/-- The conditions of the problem --/
def ageConditions (a : Ages) : Prop :=
  a.fatherPast = 3 * a.sonPast ∧
  a.sonNow = a.sonPast + 18 ∧
  a.fatherNow = a.fatherPast + 18 ∧
  a.sonNow + a.fatherNow = 108 ∧
  ∃ k : ℕ, a.fatherNow = k * a.sonNow

/-- The theorem to be proved --/
theorem father_twice_son_age (a : Ages) (h : ageConditions a) : a.fatherNow = 2 * a.sonNow := by
  sorry

end NUMINAMATH_CALUDE_father_twice_son_age_l626_62639


namespace NUMINAMATH_CALUDE_max_odd_digits_on_board_l626_62642

/-- A function that counts the number of odd digits in a natural number -/
def countOddDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 10 digits -/
def hasTenDigits (n : ℕ) : Prop := sorry

theorem max_odd_digits_on_board (a b : ℕ) (h1 : hasTenDigits a) (h2 : hasTenDigits b) :
  countOddDigits a + countOddDigits b + countOddDigits (a + b) ≤ 30 ∧
  ∃ (a' b' : ℕ), hasTenDigits a' ∧ hasTenDigits b' ∧
    countOddDigits a' + countOddDigits b' + countOddDigits (a' + b') = 30 :=
sorry

end NUMINAMATH_CALUDE_max_odd_digits_on_board_l626_62642


namespace NUMINAMATH_CALUDE_correlation_coefficient_properties_l626_62612

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

-- Define a positive relationship between two variables
def positive_relationship (x y : ℝ → ℝ) : Prop :=
  ∀ a b, a < b → x a < x b → y a < y b

-- Define a perfect linear relationship between two variables
def perfect_linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ m b, ∀ t, y t = m * x t + b

-- Theorem statement
theorem correlation_coefficient_properties
  (x y : ℝ → ℝ) (r : ℝ) (h : r = correlation_coefficient x y) :
  (r > 0 → positive_relationship x y) ∧
  (r = 1 ∨ r = -1 → perfect_linear_relationship x y) :=
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_properties_l626_62612


namespace NUMINAMATH_CALUDE_april_earnings_correct_l626_62693

/-- Calculate April's earnings from flower sales with tax --/
def april_earnings (rose_price tulip_price daisy_price : ℚ)
  (initial_roses initial_tulips initial_daisies : ℕ)
  (final_roses final_tulips final_daisies : ℕ)
  (tax_rate : ℚ) : ℚ :=
  let roses_sold := initial_roses - final_roses
  let tulips_sold := initial_tulips - final_tulips
  let daisies_sold := initial_daisies - final_daisies
  let revenue := rose_price * roses_sold + tulip_price * tulips_sold + daisy_price * daisies_sold
  let tax := revenue * tax_rate
  revenue + tax

theorem april_earnings_correct :
  april_earnings 4 3 2 13 10 8 4 3 1 (1/10) = 781/10 := by
  sorry

end NUMINAMATH_CALUDE_april_earnings_correct_l626_62693


namespace NUMINAMATH_CALUDE_product_of_complex_numbers_l626_62699

/-- Represents a complex number in polar form -/
structure PolarComplex where
  r : ℝ
  θ : ℝ
  h_r_pos : r > 0
  h_θ_range : 0 ≤ θ ∧ θ < 2 * Real.pi

/-- Multiplies two complex numbers in polar form -/
def polar_multiply (z₁ z₂ : PolarComplex) : PolarComplex :=
  { r := z₁.r * z₂.r,
    θ := z₁.θ + z₂.θ,
    h_r_pos := by sorry,
    h_θ_range := by sorry }

theorem product_of_complex_numbers :
  let z₁ : PolarComplex := ⟨5, 30 * Real.pi / 180, by sorry, by sorry⟩
  let z₂ : PolarComplex := ⟨4, 140 * Real.pi / 180, by sorry, by sorry⟩
  let result := polar_multiply z₁ z₂
  result.r = 20 ∧ result.θ = 170 * Real.pi / 180 := by sorry

end NUMINAMATH_CALUDE_product_of_complex_numbers_l626_62699


namespace NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l626_62696

/-- Given a real-valued function f, prove that the graphs of y = f(x-1) and y = f(1-x) 
    are symmetric about the line x = 1 -/
theorem symmetry_about_x_equals_one (f : ℝ → ℝ) :
  ∀ (x y : ℝ), y = f (x - 1) ∧ y = f (1 - x) →
  (∃ (x' y' : ℝ), y' = f (x' - 1) ∧ y' = f (1 - x') ∧ 
   x' = 2 - x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l626_62696


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l626_62645

theorem square_perimeter_problem (A B C : ℝ) : 
  (A > 0) → (B > 0) → (C > 0) →
  (4 * A = 16) → (4 * B = 32) → (C = A + B - 2) →
  (4 * C = 40) := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l626_62645


namespace NUMINAMATH_CALUDE_factorization_3x2_minus_12_factorization_ax2_4axy_4ay2_l626_62632

-- Statement 1
theorem factorization_3x2_minus_12 (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

-- Statement 2
theorem factorization_ax2_4axy_4ay2 (a x y : ℝ) : a * x^2 - 4 * a * x * y + 4 * a * y^2 = a * (x - 2 * y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x2_minus_12_factorization_ax2_4axy_4ay2_l626_62632


namespace NUMINAMATH_CALUDE_bacteria_eliminated_l626_62677

/-- Number of bacteria on a given day -/
def bacteria_count (day : ℕ) : ℤ :=
  50 - 6 * (day - 1)

/-- The day when bacteria are eliminated -/
def elimination_day : ℕ := 10

/-- Theorem stating that bacteria are eliminated on the 10th day -/
theorem bacteria_eliminated :
  bacteria_count elimination_day ≤ 0 ∧
  ∀ d : ℕ, d < elimination_day → bacteria_count d > 0 :=
sorry

end NUMINAMATH_CALUDE_bacteria_eliminated_l626_62677


namespace NUMINAMATH_CALUDE_marks_remaining_money_l626_62627

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial_amount : ℕ) (num_items : ℕ) (item_cost : ℕ) : ℕ :=
  initial_amount - num_items * item_cost

/-- Proves that Mark has $35 left after buying books -/
theorem marks_remaining_money :
  remaining_money 85 10 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_marks_remaining_money_l626_62627


namespace NUMINAMATH_CALUDE_tip_is_24_dollars_l626_62633

/-- The cost of a woman's haircut in dollars -/
def womens_haircut_cost : ℚ := 48

/-- The cost of a child's haircut in dollars -/
def childrens_haircut_cost : ℚ := 36

/-- The number of women getting haircuts -/
def num_women : ℕ := 1

/-- The number of children getting haircuts -/
def num_children : ℕ := 2

/-- The tip percentage as a decimal -/
def tip_percentage : ℚ := 0.20

/-- The total cost of haircuts before tip -/
def total_cost : ℚ := womens_haircut_cost * num_women + childrens_haircut_cost * num_children

/-- The tip amount in dollars -/
def tip_amount : ℚ := total_cost * tip_percentage

theorem tip_is_24_dollars : tip_amount = 24 := by
  sorry

end NUMINAMATH_CALUDE_tip_is_24_dollars_l626_62633


namespace NUMINAMATH_CALUDE_bus_cyclist_speed_problem_l626_62602

/-- Proves that given the problem conditions, the speeds of the bus and cyclist are 35 km/h and 15 km/h respectively. -/
theorem bus_cyclist_speed_problem (distance : ℝ) (first_meeting_time : ℝ) (bus_stop_time : ℝ) (overtake_time : ℝ)
  (h1 : distance = 70)
  (h2 : first_meeting_time = 7/5)
  (h3 : bus_stop_time = 1/3)
  (h4 : overtake_time = 161/60) :
  ∃ (bus_speed cyclist_speed : ℝ),
    bus_speed = 35 ∧
    cyclist_speed = 15 ∧
    first_meeting_time * (bus_speed + cyclist_speed) = distance ∧
    (first_meeting_time + overtake_time - bus_stop_time) * bus_speed - (first_meeting_time + overtake_time) * cyclist_speed = distance :=
by sorry

end NUMINAMATH_CALUDE_bus_cyclist_speed_problem_l626_62602


namespace NUMINAMATH_CALUDE_fourth_person_height_l626_62670

def height_problem (h₁ h₂ h₃ h₄ : ℝ) : Prop :=
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ ∧
  h₂ - h₁ = 2 ∧
  h₃ - h₂ = 2 ∧
  h₄ - h₃ = 6 ∧
  (h₁ + h₂ + h₃ + h₄) / 4 = 76

theorem fourth_person_height 
  (h₁ h₂ h₃ h₄ : ℝ) 
  (h : height_problem h₁ h₂ h₃ h₄) : 
  h₄ = 82 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l626_62670


namespace NUMINAMATH_CALUDE_litter_collection_weight_l626_62671

theorem litter_collection_weight 
  (gina_bags : ℕ) 
  (neighborhood_multiplier : ℕ) 
  (bag_weight : ℕ) 
  (h1 : gina_bags = 8)
  (h2 : neighborhood_multiplier = 120)
  (h3 : bag_weight = 6) : 
  (gina_bags + gina_bags * neighborhood_multiplier) * bag_weight = 5808 := by
  sorry

end NUMINAMATH_CALUDE_litter_collection_weight_l626_62671


namespace NUMINAMATH_CALUDE_max_missed_problems_l626_62675

/-- Given a test with 50 problems and a passing score of at least 85%,
    the maximum number of problems a student can miss and still pass is 7. -/
theorem max_missed_problems (total_problems : Nat) (passing_percentage : Rat) :
  total_problems = 50 →
  passing_percentage = 85 / 100 →
  (↑(total_problems - 7) : Rat) / total_problems ≥ passing_percentage ∧
  ∀ n : Nat, n > 7 → (↑(total_problems - n) : Rat) / total_problems < passing_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_missed_problems_l626_62675


namespace NUMINAMATH_CALUDE_chichikov_dead_souls_l626_62617

theorem chichikov_dead_souls (x y z : ℕ) (h : x + y + z = 1001) :
  ∃ N : ℕ, N ≤ 1001 ∧
  (∀ w : ℕ, w + min x N + min y N + min z N + min (x + y) N + min (y + z) N + min (x + z) N < N →
   w ≥ 71) :=
sorry

end NUMINAMATH_CALUDE_chichikov_dead_souls_l626_62617


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l626_62629

theorem final_sum_after_transformation (a b c S : ℝ) (h : a + b + c = S) :
  3 * (a - 4) + 3 * (b - 4) + 3 * (c - 4) = 3 * S - 36 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l626_62629


namespace NUMINAMATH_CALUDE_women_average_age_is_23_l626_62600

/-- The average age of two women given the conditions of the problem -/
def average_age_of_women (initial_men_count : ℕ) 
                         (age_increase : ℕ) 
                         (replaced_man1_age : ℕ) 
                         (replaced_man2_age : ℕ) : ℚ :=
  let total_age_increase := initial_men_count * age_increase
  let total_women_age := total_age_increase + replaced_man1_age + replaced_man2_age
  total_women_age / 2

/-- Theorem stating that the average age of the women is 23 years -/
theorem women_average_age_is_23 : 
  average_age_of_women 8 2 20 10 = 23 := by
  sorry

end NUMINAMATH_CALUDE_women_average_age_is_23_l626_62600


namespace NUMINAMATH_CALUDE_event_arrangements_eq_60_l626_62650

/-- The number of ways to select 4 students from 5 for a three-day event --/
def event_arrangements (total_students : ℕ) (selected_students : ℕ) (days : ℕ) 
  (first_day_attendees : ℕ) : ℕ :=
  Nat.choose total_students first_day_attendees * 
  (Nat.factorial (total_students - first_day_attendees) / 
   Nat.factorial (total_students - selected_students))

/-- Proof that the number of arrangements for the given conditions is 60 --/
theorem event_arrangements_eq_60 : 
  event_arrangements 5 4 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_event_arrangements_eq_60_l626_62650


namespace NUMINAMATH_CALUDE_problem_statement_l626_62676

theorem problem_statement (a b : ℝ) (h : 4 * a^2 - a * b + b^2 = 1) :
  (abs a ≤ 2 * Real.sqrt 15 / 15) ∧
  (4 / 5 ≤ 4 * a^2 + b^2 ∧ 4 * a^2 + b^2 ≤ 4 / 3) ∧
  (abs (2 * a - b) ≤ 2 * Real.sqrt 10 / 5) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l626_62676


namespace NUMINAMATH_CALUDE_derivative_of_f_l626_62688

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- State the theorem
theorem derivative_of_f :
  deriv f = fun x => 3 * x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l626_62688


namespace NUMINAMATH_CALUDE_ones_digit_of_14_power_power_of_4_cycle_exponent_even_ones_digit_14_power_14_7_power_7_l626_62658

theorem ones_digit_of_14_power (n : ℕ) : (14^n) % 10 = (4^n) % 10 := by sorry

theorem power_of_4_cycle : ∀ n : ℕ, (4^n) % 10 = (4^(n % 2 + 1)) % 10 := by sorry

theorem exponent_even : (14 * (7^7)) % 2 = 0 := by sorry

theorem ones_digit_14_power_14_7_power_7 : (14^(14 * (7^7))) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_14_power_power_of_4_cycle_exponent_even_ones_digit_14_power_14_7_power_7_l626_62658


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_equal_l626_62603

/-- Represents the investment and profit calculation for two partners over a year -/
structure Investment where
  a_initial : ℕ  -- A's initial investment
  b_initial : ℕ  -- B's initial investment
  months : ℕ     -- Total number of months
  mid_months : ℕ -- Months after which A doubles investment

/-- Calculates the total capital-months for partner A -/
def capital_months_a (i : Investment) : ℕ :=
  i.a_initial * i.mid_months + (2 * i.a_initial) * (i.months - i.mid_months)

/-- Calculates the total capital-months for partner B -/
def capital_months_b (i : Investment) : ℕ :=
  i.b_initial * i.months

/-- Theorem stating that the profit-sharing ratio is 1:1 given the specific investment conditions -/
theorem profit_sharing_ratio_equal (i : Investment) 
  (h1 : i.a_initial = 3000)
  (h2 : i.b_initial = 4500)
  (h3 : i.months = 12)
  (h4 : i.mid_months = 6) :
  capital_months_a i = capital_months_b i := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_ratio_equal_l626_62603


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l626_62636

theorem unique_solution_trigonometric_equation :
  ∃! (n : ℕ), n > 0 ∧ Real.sin (π / (3 * n)) + Real.cos (π / (3 * n)) = Real.sqrt (2 * n) / 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l626_62636


namespace NUMINAMATH_CALUDE_selling_price_equal_profit_loss_l626_62690

/-- Proves that the selling price yielding the same profit as the loss is 54,
    given the cost price and a known selling price that results in a loss. -/
theorem selling_price_equal_profit_loss
  (cost_price : ℝ)
  (loss_price : ℝ)
  (h1 : cost_price = 47)
  (h2 : loss_price = 40)
  : ∃ (selling_price : ℝ),
    selling_price - cost_price = cost_price - loss_price ∧
    selling_price = 54 :=
by
  sorry

#check selling_price_equal_profit_loss

end NUMINAMATH_CALUDE_selling_price_equal_profit_loss_l626_62690


namespace NUMINAMATH_CALUDE_circle_symmetry_trig_identity_l626_62689

/-- Given two circles C₁ and C₂ defined by their equations and a line of symmetry,
    prove that sin θ cos θ = -2/5 --/
theorem circle_symmetry_trig_identity (a θ : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + a*x = 0 → 2*x - y - 1 = 0) →
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + y*Real.tan θ = 0 → 2*x - y - 1 = 0) →
  Real.sin θ * Real.cos θ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_trig_identity_l626_62689


namespace NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l626_62649

theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  (∃ (x y : ℤ), y^k = x^2 + x) ↔ (k = 2 ∧ (∃ x : ℤ, x = 0 ∨ x = -1)) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l626_62649


namespace NUMINAMATH_CALUDE_correct_remaining_money_l626_62625

/-- Calculates the remaining money after shopping --/
def remaining_money (initial_amount : ℕ) (banana_price : ℕ) (banana_quantity : ℕ) 
  (pear_price : ℕ) (asparagus_price : ℕ) (chicken_price : ℕ) : ℕ :=
  initial_amount - (banana_price * banana_quantity + pear_price + asparagus_price + chicken_price)

/-- Proves that the remaining money is correct given the initial amount and purchases --/
theorem correct_remaining_money :
  remaining_money 55 4 2 2 6 11 = 28 := by
  sorry

end NUMINAMATH_CALUDE_correct_remaining_money_l626_62625


namespace NUMINAMATH_CALUDE_sausage_problem_l626_62615

/-- Calculates the total pounds of spicy meat mix used to make sausages -/
def total_meat_mix (initial_links : ℕ) (eaten_links : ℕ) (remaining_ounces : ℕ) : ℚ :=
  let remaining_links := initial_links - eaten_links
  let ounces_per_link := remaining_ounces / remaining_links
  let total_ounces := initial_links * ounces_per_link
  total_ounces / 16

/-- Theorem stating that given the conditions, the total meat mix used was 10 pounds -/
theorem sausage_problem (initial_links : ℕ) (eaten_links : ℕ) (remaining_ounces : ℕ) 
  (h1 : initial_links = 40)
  (h2 : eaten_links = 12)
  (h3 : remaining_ounces = 112) :
  total_meat_mix initial_links eaten_links remaining_ounces = 10 := by
  sorry

#eval total_meat_mix 40 12 112

end NUMINAMATH_CALUDE_sausage_problem_l626_62615


namespace NUMINAMATH_CALUDE_matching_socks_probability_l626_62663

/-- The number of blue-bottomed socks -/
def blue_socks : ℕ := 12

/-- The number of red-bottomed socks -/
def red_socks : ℕ := 10

/-- The number of green-bottomed socks -/
def green_socks : ℕ := 6

/-- The total number of socks -/
def total_socks : ℕ := blue_socks + red_socks + green_socks

/-- The number of ways to choose 2 socks from n socks -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of picking a matching pair of socks -/
theorem matching_socks_probability : 
  (choose_two blue_socks + choose_two red_socks + choose_two green_socks) / choose_two total_socks = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_matching_socks_probability_l626_62663
