import Mathlib

namespace NUMINAMATH_CALUDE_semicircular_cubicle_perimeter_approx_l1617_161767

/-- The perimeter of a semicircular cubicle with radius 14 units is approximately 71.96 units. -/
theorem semicircular_cubicle_perimeter_approx : ∃ (p : ℝ), 
  (abs (p - (28 + π * 14)) < 0.01) ∧ (abs (p - 71.96) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_semicircular_cubicle_perimeter_approx_l1617_161767


namespace NUMINAMATH_CALUDE_dividend_calculation_l1617_161773

theorem dividend_calculation (quotient : ℕ) (k : ℕ) (h1 : quotient = 4) (h2 : k = 14) :
  quotient * k = 56 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1617_161773


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1617_161739

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1617_161739


namespace NUMINAMATH_CALUDE_simplify_expression_l1617_161742

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1617_161742


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1617_161703

theorem fractional_equation_solution :
  ∃! x : ℚ, (2 - x) / (x - 3) + 3 = 2 / (3 - x) ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1617_161703


namespace NUMINAMATH_CALUDE_cube_root_of_sqrt_64_l1617_161736

theorem cube_root_of_sqrt_64 : ∃ (x : ℝ), x^3 = Real.sqrt 64 ∧ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_sqrt_64_l1617_161736


namespace NUMINAMATH_CALUDE_right_triangle_sides_l1617_161775

theorem right_triangle_sides (t k : ℝ) (ht : t = 84) (hk : k = 56) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = k ∧
    (1 / 2) * a * b = t ∧
    c * c = a * a + b * b ∧
    (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 24 ∧ b = 7 ∧ c = 25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l1617_161775


namespace NUMINAMATH_CALUDE_soda_cost_l1617_161785

/-- The cost of items in a fast food restaurant. -/
structure FastFoodCosts where
  burger : ℕ  -- Cost of a burger in cents
  soda : ℕ    -- Cost of a soda in cents

/-- Alice's purchase -/
def alicePurchase (c : FastFoodCosts) : ℕ := 3 * c.burger + 2 * c.soda

/-- Bob's purchase -/
def bobPurchase (c : FastFoodCosts) : ℕ := 2 * c.burger + 4 * c.soda

/-- The theorem stating the cost of a soda given the purchase information -/
theorem soda_cost :
  ∃ (c : FastFoodCosts),
    alicePurchase c = 360 ∧
    bobPurchase c = 480 ∧
    c.soda = 90 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l1617_161785


namespace NUMINAMATH_CALUDE_circle_symmetry_symmetric_circle_correct_l1617_161797

/-- Given two circles in the xy-plane, this theorem states that they are symmetric with respect to the line y = x. -/
theorem circle_symmetry (x y : ℝ) : 
  ((x - 3)^2 + (y + 1)^2 = 2) ↔ ((y + 1)^2 + (x - 3)^2 = 2) := by sorry

/-- The equation of the circle symmetric to (x-3)^2 + (y+1)^2 = 2 with respect to y = x -/
def symmetric_circle_equation (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 3)^2 = 2

theorem symmetric_circle_correct (x y : ℝ) : 
  symmetric_circle_equation x y ↔ ((y - 3)^2 + (x + 1)^2 = 2) := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_symmetric_circle_correct_l1617_161797


namespace NUMINAMATH_CALUDE_must_divide_five_l1617_161720

theorem must_divide_five (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 40)
  (h2 : Nat.gcd b c = 45)
  (h3 : Nat.gcd c d = 75)
  (h4 : 120 < Nat.gcd d a ∧ Nat.gcd d a < 150) :
  5 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_must_divide_five_l1617_161720


namespace NUMINAMATH_CALUDE_stagecoach_encounter_l1617_161730

/-- The number of stagecoaches traveling daily from Bratislava to Brașov -/
def daily_coaches_bratislava_to_brasov : ℕ := 2

/-- The number of stagecoaches traveling daily from Brașov to Bratislava -/
def daily_coaches_brasov_to_bratislava : ℕ := 2

/-- The number of days the journey takes -/
def journey_duration : ℕ := 10

/-- The number of stagecoaches encountered when traveling from Bratislava to Brașov -/
def encountered_coaches : ℕ := daily_coaches_brasov_to_bratislava * journey_duration

theorem stagecoach_encounter :
  encountered_coaches = 20 :=
sorry

end NUMINAMATH_CALUDE_stagecoach_encounter_l1617_161730


namespace NUMINAMATH_CALUDE_february_to_january_ratio_l1617_161759

-- Define the oil bills for January and February
def january_bill : ℚ := 120
def february_bill : ℚ := 180

-- Define the condition that February's bill is more than January's
axiom february_more_than_january : february_bill > january_bill

-- Define the condition about the 5:3 ratio if February's bill was $20 more
axiom ratio_condition : (february_bill + 20) / january_bill = 5 / 3

-- Theorem to prove
theorem february_to_january_ratio :
  february_bill / january_bill = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_february_to_january_ratio_l1617_161759


namespace NUMINAMATH_CALUDE_ratio_equality_l1617_161721

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1617_161721


namespace NUMINAMATH_CALUDE_expression_evaluation_l1617_161770

theorem expression_evaluation (x y : ℝ) 
  (h : (x - 1)^2 + |y + 2| = 0) : 
  (3/2) * x^2 * y - (x^2 * y - 3 * (2 * x * y - x^2 * y) - x * y) = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1617_161770


namespace NUMINAMATH_CALUDE_arc_length_sector_l1617_161768

/-- The length of an arc in a sector with radius 3 and central angle 120° is 2π -/
theorem arc_length_sector (r : ℝ) (θ : ℝ) : 
  r = 3 → θ = 120 → 2 * π * r * (θ / 360) = 2 * π := by sorry

end NUMINAMATH_CALUDE_arc_length_sector_l1617_161768


namespace NUMINAMATH_CALUDE_range_of_a_l1617_161737

/-- Proposition p: The function y=(a-1)x is increasing -/
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a - 1) * x < (a - 1) * y

/-- Proposition q: The inequality -x^2+2x-2≤a holds true for all real numbers x -/
def q (a : ℝ) : Prop := ∀ x : ℝ, -x^2 + 2*x - 2 ≤ a

/-- The main theorem stating the range of a -/
theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : 
  a ∈ Set.Icc (-1 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1617_161737


namespace NUMINAMATH_CALUDE_cycle_price_proof_l1617_161749

/-- Represents the original price of a cycle -/
def original_price : ℝ := 800

/-- Represents the selling price of the cycle -/
def selling_price : ℝ := 680

/-- Represents the loss percentage -/
def loss_percentage : ℝ := 15

theorem cycle_price_proof :
  selling_price = original_price * (1 - loss_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_cycle_price_proof_l1617_161749


namespace NUMINAMATH_CALUDE_solution_set_f_nonnegative_range_of_a_l1617_161705

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x| - 2

-- Theorem 1: Solution set of f(x) ≥ 0
theorem solution_set_f_nonnegative :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} :=
sorry

-- Theorem 2: Range of values for a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≤ |x| + a) → a ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_nonnegative_range_of_a_l1617_161705


namespace NUMINAMATH_CALUDE_binomial_coefficient_1000_1000_l1617_161776

theorem binomial_coefficient_1000_1000 : Nat.choose 1000 1000 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1000_1000_l1617_161776


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1617_161789

/-- A polynomial of degree 4 with coefficients a, b, and c -/
def P (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + b*x + c

/-- The condition for P to be divisible by (x-1)^3 -/
def isDivisibleBy (a b c : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, P a b c x = (x - 1)^3 * q x

/-- The theorem stating the necessary and sufficient conditions for P to be divisible by (x-1)^3 -/
theorem polynomial_divisibility (a b c : ℝ) :
  isDivisibleBy a b c ↔ a = -6 ∧ b = 8 ∧ c = -3 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_divisibility_l1617_161789


namespace NUMINAMATH_CALUDE_c_profit_is_3600_l1617_161707

def initial_home_value : ℝ := 20000
def profit_percentage : ℝ := 0.20
def loss_percentage : ℝ := 0.15

def sale_price : ℝ := initial_home_value * (1 + profit_percentage)
def repurchase_price : ℝ := sale_price * (1 - loss_percentage)

theorem c_profit_is_3600 : sale_price - repurchase_price = 3600 := by sorry

end NUMINAMATH_CALUDE_c_profit_is_3600_l1617_161707


namespace NUMINAMATH_CALUDE_immediate_boarding_probability_l1617_161784

def train_departure_interval : ℝ := 15
def train_stop_duration : ℝ := 2

theorem immediate_boarding_probability :
  (train_stop_duration / train_departure_interval : ℝ) = 2 / 15 := by sorry

end NUMINAMATH_CALUDE_immediate_boarding_probability_l1617_161784


namespace NUMINAMATH_CALUDE_smaug_silver_coins_l1617_161755

/-- Represents the number of coins of each type in Smaug's hoard -/
structure DragonHoard where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Calculates the total value of the hoard in copper coins -/
def hoardValue (h : DragonHoard) : ℕ :=
  h.gold * 3 * 8 + h.silver * 8 + h.copper

/-- Theorem stating that Smaug has 60 silver coins -/
theorem smaug_silver_coins :
  ∃ h : DragonHoard,
    h.gold = 100 ∧
    h.copper = 33 ∧
    hoardValue h = 2913 ∧
    h.silver = 60 := by
  sorry

end NUMINAMATH_CALUDE_smaug_silver_coins_l1617_161755


namespace NUMINAMATH_CALUDE_will_buttons_count_l1617_161780

theorem will_buttons_count (mari_buttons : ℕ) (kendra_buttons : ℕ) (sue_buttons : ℕ) (will_buttons : ℕ) : 
  mari_buttons = 8 →
  kendra_buttons = 5 * mari_buttons + 4 →
  sue_buttons = kendra_buttons / 2 →
  will_buttons = 2 * (kendra_buttons + sue_buttons) →
  will_buttons = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_will_buttons_count_l1617_161780


namespace NUMINAMATH_CALUDE_units_digit_G_2000_l1617_161793

/-- Definition of G_n -/
def G (n : ℕ) : ℕ := 2^(2^n) + 5^(5^n)

/-- Property of units digit for powers of 2 -/
axiom units_digit_power_2 (n : ℕ) : n % 4 = 0 → (2^(2^n)) % 10 = 6

/-- Property of units digit for powers of 5 -/
axiom units_digit_power_5 (n : ℕ) : (5^(5^n)) % 10 = 5

/-- Theorem: The units digit of G_2000 is 1 -/
theorem units_digit_G_2000 : G 2000 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_2000_l1617_161793


namespace NUMINAMATH_CALUDE_selection_theorem_l1617_161704

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of people to choose from -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def select_count : ℕ := 4

/-- The number of ways to select 4 people from 4 boys and 3 girls,
    such that the selection includes at least one boy and one girl -/
def selection_methods : ℕ := 34

theorem selection_theorem :
  (Nat.choose total_people select_count) - (Nat.choose num_boys select_count) = selection_methods :=
sorry

end NUMINAMATH_CALUDE_selection_theorem_l1617_161704


namespace NUMINAMATH_CALUDE_hyperbola_axis_relation_l1617_161769

-- Define the hyperbola equation
def hyperbola_equation (x y b : ℝ) : Prop := x^2 - y^2 / b^2 = 1

-- Define the length of the conjugate axis
def conjugate_axis_length (b : ℝ) : ℝ := 2 * b

-- Define the length of the transverse axis
def transverse_axis_length : ℝ := 2

-- State the theorem
theorem hyperbola_axis_relation (b : ℝ) :
  b > 0 →
  hyperbola_equation x y b →
  conjugate_axis_length b = 2 * transverse_axis_length →
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_axis_relation_l1617_161769


namespace NUMINAMATH_CALUDE_circle_square_area_l1617_161756

theorem circle_square_area (r : ℝ) (s : ℝ) (hr : r = 1) (hs : s = 2) :
  let circle_area := π * r^2
  let square_area := s^2
  let square_diagonal := s * Real.sqrt 2
  circle_area - square_area = 0 := by sorry

end NUMINAMATH_CALUDE_circle_square_area_l1617_161756


namespace NUMINAMATH_CALUDE_garys_to_harrys_book_ratio_l1617_161724

/-- Proves that the ratio of Gary's books to Harry's books is 1:2 given the specified conditions -/
theorem garys_to_harrys_book_ratio :
  ∀ (harry_books flora_books gary_books : ℕ),
    harry_books = 50 →
    flora_books = 2 * harry_books →
    harry_books + flora_books + gary_books = 175 →
    gary_books = (1 : ℚ) / 2 * harry_books := by
  sorry

end NUMINAMATH_CALUDE_garys_to_harrys_book_ratio_l1617_161724


namespace NUMINAMATH_CALUDE_material_left_proof_l1617_161713

theorem material_left_proof (material1 material2 used_material : ℚ) : 
  material1 = 5/11 →
  material2 = 2/3 →
  used_material = 2/3 →
  material1 + material2 - used_material = 5/11 := by
sorry

end NUMINAMATH_CALUDE_material_left_proof_l1617_161713


namespace NUMINAMATH_CALUDE_unique_nonnegative_integer_solution_l1617_161718

theorem unique_nonnegative_integer_solution :
  ∃! (x y z : ℕ), 5 * x + 7 * y + 5 * z = 37 ∧ 6 * x - y - 10 * z = 3 ∧ x = 4 ∧ y = 1 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_nonnegative_integer_solution_l1617_161718


namespace NUMINAMATH_CALUDE_festival_allowance_petty_cash_l1617_161765

theorem festival_allowance_petty_cash (staff_count : ℕ) (days : ℕ) (daily_rate : ℕ) (total_given : ℕ) :
  staff_count = 20 →
  days = 30 →
  daily_rate = 100 →
  total_given = 65000 →
  total_given - (staff_count * days * daily_rate) = 5000 := by
sorry

end NUMINAMATH_CALUDE_festival_allowance_petty_cash_l1617_161765


namespace NUMINAMATH_CALUDE_wedge_volume_l1617_161701

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (α : ℝ) : 
  d = 10 → α = 60 → (π * (d / 2)^2 * (d / 2 * Real.cos (α * π / 180))) = 125 * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l1617_161701


namespace NUMINAMATH_CALUDE_substitution_result_l1617_161716

theorem substitution_result (x y : ℝ) :
  y = 2 * x + 1 ∧ 5 * x - 2 * y = 7 →
  5 * x - 4 * x - 2 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_substitution_result_l1617_161716


namespace NUMINAMATH_CALUDE_inheritance_tax_theorem_inheritance_uniqueness_l1617_161783

/-- The original amount of inheritance --/
def inheritance : ℝ := 41379

/-- The total amount of taxes paid --/
def total_taxes : ℝ := 15000

/-- Theorem stating that the inheritance amount satisfies the tax conditions --/
theorem inheritance_tax_theorem :
  0.25 * inheritance + 0.15 * (0.75 * inheritance) = total_taxes :=
by sorry

/-- Theorem proving that the inheritance amount is unique --/
theorem inheritance_uniqueness (x : ℝ) :
  0.25 * x + 0.15 * (0.75 * x) = total_taxes → x = inheritance :=
by sorry

end NUMINAMATH_CALUDE_inheritance_tax_theorem_inheritance_uniqueness_l1617_161783


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_base_edge_l1617_161758

/-- Represents a hexagonal pyramid -/
structure HexagonalPyramid where
  base_edge : ℝ
  side_edge : ℝ

/-- Calculates the sum of all edge lengths in a hexagonal pyramid -/
def total_edge_length (p : HexagonalPyramid) : ℝ :=
  6 * p.base_edge + 6 * p.side_edge

/-- Theorem stating the length of the base edge in a specific hexagonal pyramid -/
theorem hexagonal_pyramid_base_edge :
  ∃ (p : HexagonalPyramid),
    p.side_edge = 8 ∧
    total_edge_length p = 120 ∧
    p.base_edge = 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_base_edge_l1617_161758


namespace NUMINAMATH_CALUDE_expand_product_l1617_161738

theorem expand_product (x : ℝ) : (x^2 + 3*x + 3) * (x^2 - 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1617_161738


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1617_161779

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 - x / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1617_161779


namespace NUMINAMATH_CALUDE_bake_sale_ratio_l1617_161752

/-- Given a bake sale where 104 items were sold in total, with 48 cookies sold,
    prove that the ratio of brownies to cookies sold is 7:6. -/
theorem bake_sale_ratio : 
  let total_items : ℕ := 104
  let cookies_sold : ℕ := 48
  let brownies_sold : ℕ := total_items - cookies_sold
  (brownies_sold : ℚ) / (cookies_sold : ℚ) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_ratio_l1617_161752


namespace NUMINAMATH_CALUDE_textbooks_on_sale_textbooks_on_sale_is_five_l1617_161740

/-- Proves the number of textbooks bought on sale given the conditions of the problem -/
theorem textbooks_on_sale (sale_price : ℕ) (online_total : ℕ) (bookstore_multiplier : ℕ) (total_spent : ℕ) : ℕ :=
  let sale_count := (total_spent - online_total - (bookstore_multiplier * online_total)) / sale_price
  sale_count

#check textbooks_on_sale 10 40 3 210 = 5

/-- The main theorem that proves the number of textbooks bought on sale is 5 -/
theorem textbooks_on_sale_is_five : textbooks_on_sale 10 40 3 210 = 5 := by
  sorry

end NUMINAMATH_CALUDE_textbooks_on_sale_textbooks_on_sale_is_five_l1617_161740


namespace NUMINAMATH_CALUDE_min_distance_complex_l1617_161766

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z - (1 + 2*I)) = 2) :
  ∃ (min_val : ℝ), min_val = 2*Real.sqrt 2 - 2 ∧
    ∀ (w : ℂ), Complex.abs (w - (1 + 2*I)) = 2 → Complex.abs (w - 3) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_complex_l1617_161766


namespace NUMINAMATH_CALUDE_simplify_fraction_l1617_161772

theorem simplify_fraction : 
  ((2^1010)^2 - (2^1008)^2) / ((2^1009)^2 - (2^1007)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1617_161772


namespace NUMINAMATH_CALUDE_soup_weight_proof_l1617_161782

theorem soup_weight_proof (initial_weight : ℝ) : 
  (((initial_weight / 2) / 2) / 2 = 5) → initial_weight = 40 := by
  sorry

end NUMINAMATH_CALUDE_soup_weight_proof_l1617_161782


namespace NUMINAMATH_CALUDE_eight_stairs_climbs_l1617_161743

-- Define the function for the number of ways to climb n stairs
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | m + 4 => climbStairs m + climbStairs (m + 1) + climbStairs (m + 2) + climbStairs (m + 3)

-- Theorem statement
theorem eight_stairs_climbs : climbStairs 8 = 108 := by
  sorry


end NUMINAMATH_CALUDE_eight_stairs_climbs_l1617_161743


namespace NUMINAMATH_CALUDE_box_length_l1617_161778

/-- The length of a box with given dimensions and cube requirements -/
theorem box_length (width : ℝ) (height : ℝ) (cube_volume : ℝ) (num_cubes : ℕ)
  (h_width : width = 12)
  (h_height : height = 3)
  (h_cube_volume : cube_volume = 3)
  (h_num_cubes : num_cubes = 108) :
  width * height * (num_cubes : ℝ) * cube_volume / (width * height) = 9 := by
  sorry

end NUMINAMATH_CALUDE_box_length_l1617_161778


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1617_161719

theorem imaginary_part_of_z (z : ℂ) : z = (Complex.I : ℂ) / (1 - Complex.I) → z.im = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1617_161719


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l1617_161764

/-- A line y = kx + 2 intersects the left branch of x^2 - y^2 = 4 at two distinct points iff k ∈ (1, √2) -/
theorem line_hyperbola_intersection (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧
    y₁ = k * x₁ + 2 ∧ y₂ = k * x₂ + 2 ∧
    x₁^2 - y₁^2 = 4 ∧ x₂^2 - y₂^2 = 4) ↔ 
  (1 < k ∧ k < Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l1617_161764


namespace NUMINAMATH_CALUDE_only_y_eq_0_is_equation_l1617_161711

-- Define a type for the expressions
inductive Expression
  | Addition : Expression
  | Equation : Expression
  | Inequality : Expression
  | NotEqual : Expression

-- Define a function to check if an expression is an equation
def isEquation (e : Expression) : Prop :=
  match e with
  | Expression.Equation => True
  | _ => False

-- State the theorem
theorem only_y_eq_0_is_equation :
  let x_plus_1_5 := Expression.Addition
  let y_eq_0 := Expression.Equation
  let six_plus_x_lt_5 := Expression.Inequality
  let ab_neq_60 := Expression.NotEqual
  (¬ isEquation x_plus_1_5) ∧
  (isEquation y_eq_0) ∧
  (¬ isEquation six_plus_x_lt_5) ∧
  (¬ isEquation ab_neq_60) :=
by sorry


end NUMINAMATH_CALUDE_only_y_eq_0_is_equation_l1617_161711


namespace NUMINAMATH_CALUDE_max_subsets_with_intersection_condition_l1617_161731

/-- Given a positive integer n ≥ 2, prove that the maximum number of mutually distinct subsets
    that can be selected from an n-element set, satisfying (Aᵢ ∩ Aₖ) ⊆ Aⱼ for all 1 ≤ i < j < k ≤ m,
    is 2n. -/
theorem max_subsets_with_intersection_condition (n : ℕ) (hn : n ≥ 2) :
  (∃ (m : ℕ) (S : Finset (Finset (Fin n))),
    (∀ A ∈ S, A ⊆ Finset.univ) ∧
    (Finset.card S = m) ∧
    (∀ (A B C : Finset (Fin n)), A ∈ S → B ∈ S → C ∈ S →
      (Finset.toList S).indexOf A < (Finset.toList S).indexOf B →
      (Finset.toList S).indexOf B < (Finset.toList S).indexOf C →
      (A ∩ C) ⊆ B) ∧
    (∀ (m' : ℕ) (S' : Finset (Finset (Fin n))),
      (∀ A ∈ S', A ⊆ Finset.univ) →
      (Finset.card S' = m') →
      (∀ (A B C : Finset (Fin n)), A ∈ S' → B ∈ S' → C ∈ S' →
        (Finset.toList S').indexOf A < (Finset.toList S').indexOf B →
        (Finset.toList S').indexOf B < (Finset.toList S').indexOf C →
        (A ∩ C) ⊆ B) →
      m' ≤ m)) ∧
  (m = 2 * n) :=
by sorry

end NUMINAMATH_CALUDE_max_subsets_with_intersection_condition_l1617_161731


namespace NUMINAMATH_CALUDE_swim_meet_car_capacity_l1617_161777

/-- Represents the transportation details for the swimming club's trip --/
structure SwimMeetTransport where
  num_cars : ℕ
  num_vans : ℕ
  people_per_car : ℕ
  people_per_van : ℕ
  max_per_van : ℕ
  additional_capacity : ℕ

/-- Calculates the maximum capacity per car given the transport details --/
def max_capacity_per_car (t : SwimMeetTransport) : ℕ :=
  let total_people := t.num_cars * t.people_per_car + t.num_vans * t.people_per_van
  let total_capacity := total_people + t.additional_capacity
  let van_capacity := t.num_vans * t.max_per_van
  (total_capacity - van_capacity) / t.num_cars

/-- Theorem stating that the maximum capacity per car is 6 for the given scenario --/
theorem swim_meet_car_capacity :
  let t : SwimMeetTransport := {
    num_cars := 2,
    num_vans := 3,
    people_per_car := 5,
    people_per_van := 3,
    max_per_van := 8,
    additional_capacity := 17
  }
  max_capacity_per_car t = 6 := by
  sorry

end NUMINAMATH_CALUDE_swim_meet_car_capacity_l1617_161777


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l1617_161722

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l1617_161722


namespace NUMINAMATH_CALUDE_smallest_digit_sum_of_product_l1617_161757

/-- Given two two-digit positive integers with all digits different and both less than 50,
    the smallest possible sum of digits of their product (a four-digit number) is 20. -/
theorem smallest_digit_sum_of_product (m n : ℕ) : 
  10 ≤ m ∧ m < 50 ∧ 10 ≤ n ∧ n < 50 ∧ 
  (∀ d₁ d₂ d₃ d₄, m = 10 * d₁ + d₂ ∧ n = 10 * d₃ + d₄ → d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄) →
  1000 ≤ m * n ∧ m * n < 10000 →
  20 ≤ (m * n / 1000 + (m * n / 100) % 10 + (m * n / 10) % 10 + m * n % 10) ∧
  ∀ p q : ℕ, 10 ≤ p ∧ p < 50 ∧ 10 ≤ q ∧ q < 50 →
    (∀ e₁ e₂ e₃ e₄, p = 10 * e₁ + e₂ ∧ q = 10 * e₃ + e₄ → e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₁ ≠ e₄ ∧ e₂ ≠ e₃ ∧ e₂ ≠ e₄ ∧ e₃ ≠ e₄) →
    1000 ≤ p * q ∧ p * q < 10000 →
    (p * q / 1000 + (p * q / 100) % 10 + (p * q / 10) % 10 + p * q % 10) ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_sum_of_product_l1617_161757


namespace NUMINAMATH_CALUDE_reciprocal_of_2023_l1617_161798

theorem reciprocal_of_2023 : 
  (∀ x : ℝ, x ≠ 0 → (1 / x) = x⁻¹) → 2023⁻¹ = (1 : ℝ) / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2023_l1617_161798


namespace NUMINAMATH_CALUDE_equation_solution_l1617_161787

theorem equation_solution : 
  ∃ t : ℝ, 3 * 3^t + Real.sqrt (9 * 9^t) = 18 ∧ t = 1 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1617_161787


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_factor_l1617_161754

/-- A police emergency number is a positive integer that ends with 133 in decimal representation. -/
def PoliceEmergencyNumber (n : ℕ+) : Prop :=
  ∃ k : ℕ, n = k * 1000 + 133

/-- Theorem: Every police emergency number has a prime factor greater than 7. -/
theorem police_emergency_number_prime_factor
  (n : ℕ+) (h : PoliceEmergencyNumber n) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n.val :=
by sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_factor_l1617_161754


namespace NUMINAMATH_CALUDE_gain_percent_is_112_5_l1617_161729

/-- Represents the ratio of selling price to cost price -/
def price_ratio : ℚ := 5 / 2

/-- Represents the discount factor applied to the selling price -/
def discount_factor : ℚ := 85 / 100

/-- Calculates the gain percent based on the given conditions -/
def gain_percent : ℚ := (price_ratio * discount_factor - 1) * 100

/-- Theorem stating the gain percent under the given conditions -/
theorem gain_percent_is_112_5 : gain_percent = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_is_112_5_l1617_161729


namespace NUMINAMATH_CALUDE_store_pricing_theorem_l1617_161791

/-- Represents the cost of pencils and notebooks in a store -/
structure StorePricing where
  pencil_price : ℝ
  notebook_price : ℝ
  h1 : 9 * pencil_price + 5 * notebook_price = 3.45
  h2 : 6 * pencil_price + 4 * notebook_price = 2.40

/-- The cost of 18 pencils and 9 notebooks is $6.75 -/
theorem store_pricing_theorem (sp : StorePricing) :
  18 * sp.pencil_price + 9 * sp.notebook_price = 6.75 := by
  sorry


end NUMINAMATH_CALUDE_store_pricing_theorem_l1617_161791


namespace NUMINAMATH_CALUDE_roses_per_day_l1617_161750

theorem roses_per_day (total_roses : ℕ) (days : ℕ) (dozens_per_day : ℕ) 
  (h1 : total_roses = 168) 
  (h2 : days = 7) 
  (h3 : dozens_per_day * 12 * days = total_roses) : 
  dozens_per_day = 2 := by
  sorry

end NUMINAMATH_CALUDE_roses_per_day_l1617_161750


namespace NUMINAMATH_CALUDE_cubic_polynomial_value_at_6_l1617_161761

/-- A cubic polynomial satisfying specific conditions -/
def cubic_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → p n = 1 / (n^2 : ℝ))

/-- Theorem stating that a cubic polynomial satisfying given conditions has p(6) = 0 -/
theorem cubic_polynomial_value_at_6 (p : ℝ → ℝ) (h : cubic_polynomial p) : p 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_value_at_6_l1617_161761


namespace NUMINAMATH_CALUDE_expression_value_approximation_l1617_161727

def x : ℝ := 102
def y : ℝ := 98

theorem expression_value_approximation :
  let expr := (x^2 - y^2) / (x + y)^3 - (x^3 + y^3) * Real.log (x*y)
  ∃ ε > 0, |expr + 18446424.7199| < ε := by
  sorry

end NUMINAMATH_CALUDE_expression_value_approximation_l1617_161727


namespace NUMINAMATH_CALUDE_consecutive_integers_equality_l1617_161786

theorem consecutive_integers_equality (n : ℕ) (h : n > 0) : 
  (n + (n+1) + (n+2) + (n+3) = (n+4) + (n+5) + (n+6)) ↔ n = 9 :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_equality_l1617_161786


namespace NUMINAMATH_CALUDE_identity_implies_equality_l1617_161709

theorem identity_implies_equality (a b c d : ℝ) :
  (∀ x : ℝ, a * x + b = c * x + d) → (a = c ∧ b = d) := by
  sorry

end NUMINAMATH_CALUDE_identity_implies_equality_l1617_161709


namespace NUMINAMATH_CALUDE_ice_cream_picnic_tickets_l1617_161771

theorem ice_cream_picnic_tickets (total_tickets : ℕ) (student_price non_student_price total_collected : ℚ) 
  (h1 : total_tickets = 193)
  (h2 : student_price = 1/2)
  (h3 : non_student_price = 3/2)
  (h4 : total_collected = 825/4) :
  ∃ (student_tickets : ℕ), 
    student_tickets ≤ total_tickets ∧ 
    (student_tickets : ℚ) * student_price + (total_tickets - student_tickets : ℚ) * non_student_price = total_collected ∧
    student_tickets = 83 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_picnic_tickets_l1617_161771


namespace NUMINAMATH_CALUDE_root_implies_sum_l1617_161794

/-- Given that 2 + i is a root of the polynomial x^4 + px^2 + qx + 1 = 0,
    where p and q are real numbers, prove that p + q = 4 -/
theorem root_implies_sum (p q : ℝ) 
  (h : (2 + Complex.I) ^ 4 + p * (2 + Complex.I) ^ 2 + q * (2 + Complex.I) + 1 = 0) : 
  p + q = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_sum_l1617_161794


namespace NUMINAMATH_CALUDE_tangent_line_equations_l1617_161748

/-- Given a cubic curve and a point, prove the equations of tangent lines passing through the point. -/
theorem tangent_line_equations (x y : ℝ → ℝ) (P : ℝ × ℝ) : 
  (∀ t, y t = (1/3) * (x t)^3 + 4/3) →  -- Curve equation
  P = (2, 4) →  -- Point P
  ∃ (A B : ℝ), 
    ((4 * x A - y A - 4 = 0) ∨ (x B - y B + 2 = 0)) ∧ 
    (∀ t, (4 * t - y A - 4 = 0) → (x A, y A) = (2, 4)) ∧
    (∀ t, (t - y B + 2 = 0) → (x B, y B) = (2, 4)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l1617_161748


namespace NUMINAMATH_CALUDE_lucas_numbers_l1617_161781

theorem lucas_numbers (a b : ℤ) : 
  3 * a + 4 * b = 140 → (a = 20 ∨ b = 20) → a = 20 ∧ b = 20 := by
  sorry

end NUMINAMATH_CALUDE_lucas_numbers_l1617_161781


namespace NUMINAMATH_CALUDE_mariela_cards_l1617_161799

/-- The total number of get well cards Mariela received -/
def total_cards (hospital_cards : ℕ) (home_cards : ℕ) : ℕ :=
  hospital_cards + home_cards

/-- Theorem stating the total number of cards Mariela received -/
theorem mariela_cards : 
  total_cards 403 287 = 690 := by
  sorry

end NUMINAMATH_CALUDE_mariela_cards_l1617_161799


namespace NUMINAMATH_CALUDE_inequality_theorem_l1617_161753

/-- The function f(x, y) = ax² + 2bxy + cy² -/
def f (a b c x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

/-- The main theorem -/
theorem inequality_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_nonneg : ∀ (x y : ℝ), 0 ≤ f a b c x y) :
  ∀ (x₁ x₂ y₁ y₂ : ℝ),
    Real.sqrt (f a b c x₁ y₁ * f a b c x₂ y₂) * f a b c (x₁ - x₂) (y₁ - y₂) ≥
    (a * c - b^2) * (x₁ * y₂ - x₂ * y₁)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1617_161753


namespace NUMINAMATH_CALUDE_encyclopedia_interest_percentage_l1617_161734

/-- Calculates the interest percentage given the conditions of an encyclopedia purchase --/
theorem encyclopedia_interest_percentage 
  (down_payment : ℚ)
  (total_cost : ℚ)
  (monthly_payment : ℚ)
  (num_monthly_payments : ℕ)
  (final_payment : ℚ)
  (h1 : down_payment = 300)
  (h2 : total_cost = 750)
  (h3 : monthly_payment = 57)
  (h4 : num_monthly_payments = 9)
  (h5 : final_payment = 21) :
  let total_paid := down_payment + (monthly_payment * num_monthly_payments) + final_payment
  let amount_borrowed := total_cost - down_payment
  let interest_paid := total_paid - total_cost
  interest_paid / amount_borrowed = 8533 / 10000 := by
sorry


end NUMINAMATH_CALUDE_encyclopedia_interest_percentage_l1617_161734


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l1617_161746

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 3*x + 5

-- State the theorem
theorem root_exists_in_interval :
  ∃ x₀ ∈ Set.Ioo 1 2, f x₀ = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l1617_161746


namespace NUMINAMATH_CALUDE_expected_value_is_91_div_6_l1617_161792

/-- The expected value of rolling a fair 6-sided die where the win is n^2 dollars for rolling n -/
def expected_value : ℚ :=
  (1 / 6 : ℚ) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

/-- Theorem stating that the expected value is equal to 91/6 -/
theorem expected_value_is_91_div_6 : expected_value = 91 / 6 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_91_div_6_l1617_161792


namespace NUMINAMATH_CALUDE_product_quotient_calculation_l1617_161774

theorem product_quotient_calculation : 16 * 0.0625 / 4 * 0.5 * 2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_quotient_calculation_l1617_161774


namespace NUMINAMATH_CALUDE_inequality_proof_l1617_161796

/-- The function f(x) defined as |x-m| + |x+3| -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 3|

/-- Theorem stating that given the conditions, 1/(m+n) + 1/t ≥ 2 -/
theorem inequality_proof (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0) 
  (h_min : ∀ x, f m x ≥ 5 - n - t) : 
  1 / (m + n) + 1 / t ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_inequality_proof_l1617_161796


namespace NUMINAMATH_CALUDE_floor_ceiling_difference_l1617_161741

theorem floor_ceiling_difference : ⌊(1.999 : ℝ)⌋ - ⌈(3.001 : ℝ)⌉ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_difference_l1617_161741


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l1617_161744

theorem floor_ceiling_sum (x y : ℝ) (hx : 1 < x ∧ x < 2) (hy : 3 < y ∧ y < 4) :
  ⌊x⌋ + ⌈y⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l1617_161744


namespace NUMINAMATH_CALUDE_min_fraction_sum_l1617_161725

def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_selection (P Q R S : ℕ) : Prop :=
  P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ P < Q ∧ Q < R ∧ R < S

def fraction_sum (P Q R S : ℕ) : ℚ :=
  (P : ℚ) / (R : ℚ) + (Q : ℚ) / (S : ℚ)

theorem min_fraction_sum :
  ∃ (P Q R S : ℕ), is_valid_selection P Q R S ∧
    (∀ (P' Q' R' S' : ℕ), is_valid_selection P' Q' R' S' →
      fraction_sum P Q R S ≤ fraction_sum P' Q' R' S') ∧
    fraction_sum P Q R S = 25 / 72 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l1617_161725


namespace NUMINAMATH_CALUDE_emily_bought_seven_songs_l1617_161700

/-- The number of songs Emily bought later -/
def songs_bought_later (initial_songs total_songs : ℕ) : ℕ :=
  total_songs - initial_songs

/-- Proof that Emily bought 7 songs later -/
theorem emily_bought_seven_songs :
  let initial_songs := 6
  let total_songs := 13
  songs_bought_later initial_songs total_songs = 7 := by
  sorry

end NUMINAMATH_CALUDE_emily_bought_seven_songs_l1617_161700


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_8640_l1617_161788

/-- The number of positive integer factors of 8640 that are perfect squares -/
def num_perfect_square_factors (n : ℕ) : ℕ :=
  (Finset.range 4).card * (Finset.range 2).card * (Finset.range 1).card

/-- The prime factorization of 8640 -/
def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  [(2, 6), (3, 3), (5, 1)]

theorem perfect_square_factors_of_8640 :
  num_perfect_square_factors 8640 = 8 ∧ prime_factorization 8640 = [(2, 6), (3, 3), (5, 1)] := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_8640_l1617_161788


namespace NUMINAMATH_CALUDE_mod_eight_difference_l1617_161728

theorem mod_eight_difference (n : ℕ) : (47^n - 23^n) % 8 = 0 :=
sorry

end NUMINAMATH_CALUDE_mod_eight_difference_l1617_161728


namespace NUMINAMATH_CALUDE_june_score_june_score_correct_l1617_161762

theorem june_score (april_may_avg : ℕ) (april_may_june_avg : ℕ) : ℕ :=
  let april_may_total := april_may_avg * 2
  let april_may_june_total := april_may_june_avg * 3
  april_may_june_total - april_may_total

theorem june_score_correct :
  june_score 89 88 = 86 := by sorry

end NUMINAMATH_CALUDE_june_score_june_score_correct_l1617_161762


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1617_161706

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1617_161706


namespace NUMINAMATH_CALUDE_log_expression_equals_one_l1617_161702

theorem log_expression_equals_one : 
  (((1 - Real.log 3 / Real.log 6) ^ 2 + (Real.log 2 / Real.log 6) * (Real.log 18 / Real.log 6)) / (Real.log 4 / Real.log 6)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_one_l1617_161702


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1617_161708

theorem polynomial_division_theorem (x : ℝ) : 
  8 * x^3 + 4 * x^2 - 6 * x - 9 = (x + 3) * (8 * x^2 - 20 * x + 54) - 171 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1617_161708


namespace NUMINAMATH_CALUDE_non_adjacent_arrangement_count_l1617_161732

/-- Represents the number of ways to arrange balls in a row -/
def arrangement_count : ℕ := 12

/-- Represents the number of white balls -/
def white_ball_count : ℕ := 1

/-- Represents the number of red balls -/
def red_ball_count : ℕ := 1

/-- Represents the number of yellow balls -/
def yellow_ball_count : ℕ := 3

/-- Theorem stating that the number of arrangements where white and red balls are not adjacent is 12 -/
theorem non_adjacent_arrangement_count :
  (white_ball_count = 1) →
  (red_ball_count = 1) →
  (yellow_ball_count = 3) →
  (arrangement_count = 12) := by
  sorry

#check non_adjacent_arrangement_count

end NUMINAMATH_CALUDE_non_adjacent_arrangement_count_l1617_161732


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1617_161747

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The length of the base
  base : ℝ
  -- The length of the median drawn to one of the congruent sides
  median : ℝ
  -- The length of each congruent side
  side : ℝ
  -- Condition that the base is 4√2
  base_eq : base = 4 * Real.sqrt 2
  -- Condition that the median is 5
  median_eq : median = 5

/-- Theorem stating the length of the congruent sides in the specific isosceles triangle -/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) : t.side = Real.sqrt 34 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1617_161747


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l1617_161733

/-- If a side of beef loses 50 percent of its weight in processing and weighs 750 pounds after processing, then it weighed 1500 pounds before processing. -/
theorem beef_weight_before_processing (weight_after : ℝ) (h1 : weight_after = 750) :
  ∃ weight_before : ℝ, weight_before * 0.5 = weight_after ∧ weight_before = 1500 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l1617_161733


namespace NUMINAMATH_CALUDE_stratified_sampling_probability_l1617_161745

/-- Represents the number of students in each year of high school. -/
structure SchoolPopulation where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Represents the number of students selected from each year in the sample. -/
structure SampleSize where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- The probability of a student being selected in a stratified sampling survey. -/
def selectionProbability (population : SchoolPopulation) (sample : SampleSize) : ℚ :=
  sample.third_year / population.third_year

theorem stratified_sampling_probability
  (population : SchoolPopulation)
  (sample : SampleSize)
  (h1 : population.first_year = 800)
  (h2 : population.second_year = 600)
  (h3 : population.third_year = 500)
  (h4 : sample.third_year = 25) :
  selectionProbability population sample = 1 / 20 := by
  sorry

#check stratified_sampling_probability

end NUMINAMATH_CALUDE_stratified_sampling_probability_l1617_161745


namespace NUMINAMATH_CALUDE_angle_greater_if_sine_greater_l1617_161726

theorem angle_greater_if_sine_greater (A B C : Real) (a b c : Real) :
  -- Define triangle ABC
  (A + B + C = Real.pi) →
  (a > 0) → (b > 0) → (c > 0) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Given condition
  (Real.sin B > Real.sin C) →
  -- Conclusion
  B > C := by
  sorry


end NUMINAMATH_CALUDE_angle_greater_if_sine_greater_l1617_161726


namespace NUMINAMATH_CALUDE_jenny_run_distance_l1617_161735

theorem jenny_run_distance (walked : Real) (ran_extra : Real) : 
  walked = 0.4 → ran_extra = 0.2 → walked + ran_extra = 0.6 := by
sorry

end NUMINAMATH_CALUDE_jenny_run_distance_l1617_161735


namespace NUMINAMATH_CALUDE_hand_count_theorem_l1617_161715

def special_deck_size : ℕ := 60
def hand_size : ℕ := 12

def number_of_hands : ℕ := Nat.choose special_deck_size hand_size

theorem hand_count_theorem (C : ℕ) (h : C < 10) :
  ∃ (B : ℕ), number_of_hands = 192 * (10^6) + B * (10^5) + C * (10^4) + 3210 :=
by sorry

end NUMINAMATH_CALUDE_hand_count_theorem_l1617_161715


namespace NUMINAMATH_CALUDE_inequality_proof_l1617_161712

theorem inequality_proof (a : ℝ) (x : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1/2) (h3 : x ≥ 0) :
  let f : ℝ → ℝ := λ y => Real.exp y
  let g : ℝ → ℝ := λ y => a * y + 1
  1 / f x + x / g x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1617_161712


namespace NUMINAMATH_CALUDE_p_cubed_plus_mp_l1617_161723

theorem p_cubed_plus_mp (p m : ℤ) (h_p_odd : Odd p) : 
  Odd (p^3 + m*p) ↔ Even m := by
  sorry

end NUMINAMATH_CALUDE_p_cubed_plus_mp_l1617_161723


namespace NUMINAMATH_CALUDE_parabola_c_range_l1617_161795

/-- The range of c for a parabola with specific properties -/
theorem parabola_c_range (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 → -1 < x ∧ x < 3) →  -- roots within (-1, 3)
  (∃ x, -1 < x ∧ x < 3 ∧ x^2 + b*x + c = 0 ∧ -x^2 - b*x - c = 0) →  -- equal roots exist
  b = -4 →  -- axis of symmetry at x = 2
  (-5 < c ∧ c ≤ 3) ∨ c = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_c_range_l1617_161795


namespace NUMINAMATH_CALUDE_dans_to_barrys_dimes_ratio_l1617_161751

/-- The ratio of Dan's initial dimes to Barry's dimes -/
theorem dans_to_barrys_dimes_ratio :
  let barry_dimes : ℕ := 1000 / 10
  let dan_final_dimes : ℕ := 52
  let dan_initial_dimes : ℕ := dan_final_dimes - 2
  (dan_initial_dimes : ℚ) / barry_dimes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_dans_to_barrys_dimes_ratio_l1617_161751


namespace NUMINAMATH_CALUDE_validSelectionsCount_l1617_161763

/-- Represents the set of available colors --/
inductive Color
| Red
| Blue
| Yellow
| Green

/-- Represents a ball with a color and number --/
structure Ball where
  color : Color
  number : Fin 6

/-- The set of all balls --/
def allBalls : Finset Ball :=
  sorry

/-- Checks if three numbers are non-consecutive --/
def areNonConsecutive (n1 n2 n3 : Fin 6) : Prop :=
  sorry

/-- Checks if three balls have different colors --/
def haveDifferentColors (b1 b2 b3 : Ball) : Prop :=
  sorry

/-- The set of valid selections of 3 balls --/
def validSelections : Finset (Fin 24 × Fin 24 × Fin 24) :=
  sorry

theorem validSelectionsCount :
  Finset.card validSelections = 96 := by
  sorry

end NUMINAMATH_CALUDE_validSelectionsCount_l1617_161763


namespace NUMINAMATH_CALUDE_dodecahedral_die_expected_value_l1617_161714

/-- A fair dodecahedral die with faces numbered from 1 to 12 -/
def DodecahedralDie : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 12

/-- The expected value of rolling the die -/
def expected_value : ℚ := (DodecahedralDie.sum (fun i => prob i * (i + 1)))

/-- Theorem: The expected value of rolling a fair dodecahedral die is 6.5 -/
theorem dodecahedral_die_expected_value :
  expected_value = 13 / 2 := by sorry

end NUMINAMATH_CALUDE_dodecahedral_die_expected_value_l1617_161714


namespace NUMINAMATH_CALUDE_room_length_proof_l1617_161790

/-- Proves that the length of a rectangular room is 5.5 meters given specific conditions -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 3.75 →
  total_cost = 24750 →
  paving_rate = 1200 →
  (total_cost / paving_rate) / width = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l1617_161790


namespace NUMINAMATH_CALUDE_marys_age_l1617_161717

theorem marys_age (mary_age rahul_age : ℕ) : 
  rahul_age = mary_age + 30 →
  rahul_age + 20 = 2 * (mary_age + 20) →
  mary_age = 10 := by
sorry

end NUMINAMATH_CALUDE_marys_age_l1617_161717


namespace NUMINAMATH_CALUDE_cash_realized_proof_l1617_161710

/-- Given an amount before brokerage and a brokerage rate, calculates the cash realized after brokerage. -/
def cash_realized (amount_before_brokerage : ℚ) (brokerage_rate : ℚ) : ℚ :=
  amount_before_brokerage - (amount_before_brokerage * brokerage_rate)

/-- Theorem stating that for the given conditions, the cash realized is 104.7375 -/
theorem cash_realized_proof :
  let amount_before_brokerage : ℚ := 105
  let brokerage_rate : ℚ := 1 / 400
  cash_realized amount_before_brokerage brokerage_rate = 104.7375 := by
  sorry

#eval cash_realized 105 (1/400)

end NUMINAMATH_CALUDE_cash_realized_proof_l1617_161710


namespace NUMINAMATH_CALUDE_tree_spacing_l1617_161760

theorem tree_spacing (road_length : ℝ) (num_trees : ℕ) (space_between : ℝ) :
  road_length = 157 ∧ num_trees = 13 ∧ space_between = 12 →
  (road_length - space_between * (num_trees - 1)) / num_trees = 1 :=
by sorry

end NUMINAMATH_CALUDE_tree_spacing_l1617_161760
