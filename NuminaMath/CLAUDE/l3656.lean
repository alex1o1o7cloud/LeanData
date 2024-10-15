import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_equality_l3656_365608

theorem arithmetic_equality : 4 * 8 + 5 * 11 - 2 * 3 + 7 * 9 = 144 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3656_365608


namespace NUMINAMATH_CALUDE_x_value_l3656_365672

theorem x_value : ∃ x : ℝ, x = 88 * (1 + 0.40) ∧ x = 123.2 :=
by sorry

end NUMINAMATH_CALUDE_x_value_l3656_365672


namespace NUMINAMATH_CALUDE_three_propositions_imply_l3656_365603

theorem three_propositions_imply (p q r : Prop) : 
  (((p ∨ (¬q ∧ r)) → ((p → q) → r)) ∧
   ((¬p ∨ (¬q ∧ r)) → ((p → q) → r)) ∧
   ((p ∨ (¬q ∧ ¬r)) → ¬((p → q) → r)) ∧
   ((¬p ∨ (q ∧ r)) → ((p → q) → r))) := by
  sorry

end NUMINAMATH_CALUDE_three_propositions_imply_l3656_365603


namespace NUMINAMATH_CALUDE_rice_mixture_price_l3656_365673

/-- Proves that the price of the second type of rice is 9.60 Rs/kg -/
theorem rice_mixture_price (price1 : ℝ) (weight1 : ℝ) (weight2 : ℝ) (mixture_price : ℝ) 
  (h1 : price1 = 6.60)
  (h2 : weight1 = 49)
  (h3 : weight2 = 56)
  (h4 : mixture_price = 8.20)
  (h5 : weight1 + weight2 = 105) :
  ∃ (price2 : ℝ), price2 = 9.60 ∧ 
  (price1 * weight1 + price2 * weight2) / (weight1 + weight2) = mixture_price :=
by sorry

end NUMINAMATH_CALUDE_rice_mixture_price_l3656_365673


namespace NUMINAMATH_CALUDE_parallel_vectors_k_equals_two_l3656_365639

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then k = 2 -/
theorem parallel_vectors_k_equals_two (k : ℝ) :
  let a : ℝ × ℝ := (k - 1, k)
  let b : ℝ × ℝ := (1, 2)
  are_parallel a b → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_equals_two_l3656_365639


namespace NUMINAMATH_CALUDE_pages_per_sheet_calculation_l3656_365670

/-- The number of stories John writes per week -/
def stories_per_week : ℕ := 3

/-- The number of pages in each story -/
def pages_per_story : ℕ := 50

/-- The number of weeks John writes -/
def weeks : ℕ := 12

/-- The number of reams of paper John uses over 12 weeks -/
def reams_used : ℕ := 3

/-- The number of sheets in each ream of paper -/
def sheets_per_ream : ℕ := 500

/-- Calculate the number of pages each sheet of paper can hold -/
def pages_per_sheet : ℕ := 1

theorem pages_per_sheet_calculation :
  pages_per_sheet = 1 :=
by sorry

end NUMINAMATH_CALUDE_pages_per_sheet_calculation_l3656_365670


namespace NUMINAMATH_CALUDE_f_equals_g_l3656_365642

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := (x^6)^(1/3)

-- Theorem statement
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l3656_365642


namespace NUMINAMATH_CALUDE_third_number_solution_l3656_365633

theorem third_number_solution (x : ℝ) : 3 + 33 + x + 33.3 = 399.6 → x = 330.3 := by
  sorry

end NUMINAMATH_CALUDE_third_number_solution_l3656_365633


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3656_365648

/-- The line x = my + 2 is tangent to the circle x^2 + 2x + y^2 + 2y = 0 if and only if m = 1 or m = -7 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x = m * y + 2 → x^2 + 2*x + y^2 + 2*y ≠ 0) ∨
  (∃! x y : ℝ, x = m * y + 2 ∧ x^2 + 2*x + y^2 + 2*y = 0) ↔ 
  m = 1 ∨ m = -7 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3656_365648


namespace NUMINAMATH_CALUDE_sound_travel_distance_l3656_365660

/-- The speed of sound in air at 20°C in meters per second -/
def speed_of_sound_at_20C : ℝ := 342

/-- The time of travel in seconds -/
def travel_time : ℝ := 5

/-- The distance traveled by sound in 5 seconds at 20°C -/
def distance_traveled : ℝ := speed_of_sound_at_20C * travel_time

theorem sound_travel_distance : distance_traveled = 1710 := by
  sorry

end NUMINAMATH_CALUDE_sound_travel_distance_l3656_365660


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3656_365617

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of a line passing through a point
def passesThrough (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the property of a line having equal intercepts
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.a = l.b ∨ (l.a = 0 ∧ l.c = 0) ∨ (l.b = 0 ∧ l.c = 0)

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l₁ l₂ : Line2D),
    (passesThrough l₁ ⟨2, 3⟩ ∧ hasEqualIntercepts l₁ ∧ l₁ = ⟨1, 1, -5⟩) ∧
    (passesThrough l₂ ⟨2, 3⟩ ∧ hasEqualIntercepts l₂ ∧ l₂ = ⟨3, -2, 0⟩) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3656_365617


namespace NUMINAMATH_CALUDE_valid_numbers_l3656_365611

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (a + 1) * (b + 2) * (c + 3) * (d + 4) = 234

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n → n = 1109 ∨ n = 2009 :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3656_365611


namespace NUMINAMATH_CALUDE_special_triangle_solution_l3656_365600

/-- Represents a triangle with given properties -/
structure SpecialTriangle where
  a : ℝ
  r : ℝ
  ρ : ℝ
  h_a : a = 6
  h_r : r = 5
  h_ρ : ρ = 2

/-- The other two sides and area of the special triangle -/
def TriangleSolution (t : SpecialTriangle) : ℝ × ℝ × ℝ :=
  (8, 10, 24)

theorem special_triangle_solution (t : SpecialTriangle) :
  let (b, c, area) := TriangleSolution t
  b * c = 10 * area / 3 ∧
  b + c = area - t.a ∧
  area = t.ρ * (t.a + b + c) / 2 ∧
  area^2 = (t.a + b + c) / 2 * ((t.a + b + c) / 2 - t.a) * ((t.a + b + c) / 2 - b) * ((t.a + b + c) / 2 - c) ∧
  t.r = t.a * b * c / (4 * area) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_solution_l3656_365600


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3656_365615

theorem trigonometric_identity (θ φ : Real) 
  (h : (Real.sin θ)^4 / (Real.sin φ)^2 + (Real.cos θ)^4 / (Real.cos φ)^2 = 1) :
  (Real.cos φ)^4 / (Real.cos θ)^2 + (Real.sin φ)^4 / (Real.sin θ)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3656_365615


namespace NUMINAMATH_CALUDE_sum_of_critical_slopes_l3656_365624

/-- Parabola defined by y = 2x^2 -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- Line through Q with slope m -/
def line (m : ℝ) (x : ℝ) : ℝ := m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, line m x ≠ parabola x

/-- Theorem stating the sum of critical slopes -/
theorem sum_of_critical_slopes :
  ∃ r s, (∀ m, r < m ∧ m < s ↔ no_intersection m) ∧ r + s = 40 := by sorry

end NUMINAMATH_CALUDE_sum_of_critical_slopes_l3656_365624


namespace NUMINAMATH_CALUDE_angle_A_measure_l3656_365627

/-- In a geometric configuration with angles of 110°, 100°, and 40°, there exists an angle A that measures 30°. -/
theorem angle_A_measure (α β γ : Real) (h1 : α = 110) (h2 : β = 100) (h3 : γ = 40) :
  ∃ A : Real, A = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_l3656_365627


namespace NUMINAMATH_CALUDE_batsman_innings_l3656_365690

theorem batsman_innings (average : ℝ) (highest_score : ℝ) (score_difference : ℝ) (average_excluding : ℝ) 
  (h1 : average = 61)
  (h2 : score_difference = 150)
  (h3 : average_excluding = 58)
  (h4 : highest_score = 202) :
  ∃ (n : ℕ), n = 46 ∧ 
    average * n = highest_score + (highest_score - score_difference) + average_excluding * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_batsman_innings_l3656_365690


namespace NUMINAMATH_CALUDE_field_day_shirt_cost_l3656_365629

/-- The total cost of shirts for field day -/
def total_cost (kindergarten_count : ℕ) (kindergarten_price : ℚ)
                (first_grade_count : ℕ) (first_grade_price : ℚ)
                (second_grade_count : ℕ) (second_grade_price : ℚ)
                (third_grade_count : ℕ) (third_grade_price : ℚ) : ℚ :=
  kindergarten_count * kindergarten_price +
  first_grade_count * first_grade_price +
  second_grade_count * second_grade_price +
  third_grade_count * third_grade_price

/-- The total cost of shirts for field day is $2317.00 -/
theorem field_day_shirt_cost :
  total_cost 101 (580/100) 113 5 107 (560/100) 108 (525/100) = 2317 := by
  sorry

end NUMINAMATH_CALUDE_field_day_shirt_cost_l3656_365629


namespace NUMINAMATH_CALUDE_distance_between_homes_l3656_365677

/-- Proves the distance between Maxwell's and Brad's homes given their speeds and meeting point -/
theorem distance_between_homes
  (maxwell_speed : ℝ)
  (brad_speed : ℝ)
  (maxwell_distance : ℝ)
  (h1 : maxwell_speed = 2)
  (h2 : brad_speed = 3)
  (h3 : maxwell_distance = 26)
  (h4 : maxwell_distance / maxwell_speed = (total_distance - maxwell_distance) / brad_speed) :
  total_distance = 65 :=
by
  sorry

#check distance_between_homes

end NUMINAMATH_CALUDE_distance_between_homes_l3656_365677


namespace NUMINAMATH_CALUDE_exists_n_factorial_starts_with_2015_l3656_365652

/-- Given a natural number n, returns the first four digits of n! as a natural number -/
def firstFourDigitsOfFactorial (n : ℕ) : ℕ :=
  sorry

/-- Theorem: There exists a positive integer n such that the first four digits of n! are 2015 -/
theorem exists_n_factorial_starts_with_2015 : ∃ n : ℕ+, firstFourDigitsOfFactorial n.val = 2015 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_factorial_starts_with_2015_l3656_365652


namespace NUMINAMATH_CALUDE_fertilizer_growth_rate_l3656_365671

theorem fertilizer_growth_rate 
  (april_output : ℝ) 
  (may_decrease : ℝ) 
  (july_output : ℝ) 
  (h1 : april_output = 500)
  (h2 : may_decrease = 0.2)
  (h3 : july_output = 576) :
  ∃ (x : ℝ), 
    april_output * (1 - may_decrease) * (1 + x)^2 = july_output ∧ 
    x = 0.2 :=
sorry

end NUMINAMATH_CALUDE_fertilizer_growth_rate_l3656_365671


namespace NUMINAMATH_CALUDE_square_sum_inequality_l3656_365637

theorem square_sum_inequality (a b : ℝ) : a^2 + b^2 ≥ 2*(a - b - 1) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l3656_365637


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l3656_365655

/-- Given a point A with coordinates (2, 3), its symmetric point with respect to the x-axis has coordinates (2, -3). -/
theorem symmetric_point_wrt_x_axis :
  let A : ℝ × ℝ := (2, 3)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point A = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l3656_365655


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3656_365626

theorem quadratic_expression_value (x : ℝ) : 2 * x^2 + 3 * x - 1 = 7 → 4 * x^2 + 6 * x + 9 = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3656_365626


namespace NUMINAMATH_CALUDE_log_product_identity_l3656_365602

theorem log_product_identity (b c : ℝ) (hb_pos : b > 0) (hc_pos : c > 0) (hb_ne_one : b ≠ 1) (hc_ne_one : c ≠ 1) :
  Real.log b / Real.log (2 * c) * Real.log (2 * c) / Real.log b = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_identity_l3656_365602


namespace NUMINAMATH_CALUDE_gcd_sum_characterization_l3656_365613

theorem gcd_sum_characterization (M : ℝ) (h_M : M ≥ 1) :
  ∀ n : ℕ, (∃ a b c : ℕ, a > M ∧ b > M ∧ c > M ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = Nat.gcd a b * Nat.gcd b c + Nat.gcd b c * Nat.gcd c a + Nat.gcd c a * Nat.gcd a b) ↔
  (Even (Nat.log 2 n) ∧ ¬∃ k : ℕ, n = 4^k) :=
by sorry

end NUMINAMATH_CALUDE_gcd_sum_characterization_l3656_365613


namespace NUMINAMATH_CALUDE_third_graders_count_l3656_365687

theorem third_graders_count (T : ℚ) 
  (h1 : T + 2 * T + T / 2 = 70) : T = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_graders_count_l3656_365687


namespace NUMINAMATH_CALUDE_median_and_area_of_triangle_l3656_365651

/-- Triangle DEF with given side lengths -/
structure Triangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ

/-- The isosceles triangle DEF with given side lengths -/
def isoscelesTriangle : Triangle where
  DE := 13
  DF := 13
  EF := 14

/-- The length of the median DM in triangle DEF -/
def medianLength (t : Triangle) : ℝ := sorry

/-- The area of triangle DEF -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Theorem stating the length of the median and the area of the triangle -/
theorem median_and_area_of_triangle :
  medianLength isoscelesTriangle = 2 * Real.sqrt 30 ∧
  triangleArea isoscelesTriangle = 84 := by sorry

end NUMINAMATH_CALUDE_median_and_area_of_triangle_l3656_365651


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l3656_365680

theorem polygon_interior_angles (n : ℕ) : 
  (n - 2) * 180 = 540 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l3656_365680


namespace NUMINAMATH_CALUDE_unique_number_property_l3656_365681

theorem unique_number_property : ∃! x : ℝ, x / 2 = x - 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_property_l3656_365681


namespace NUMINAMATH_CALUDE_pizza_cost_per_pizza_l3656_365685

theorem pizza_cost_per_pizza (num_pizzas : ℕ) (num_toppings : ℕ) 
  (cost_per_topping : ℚ) (tip : ℚ) (total_cost : ℚ) :
  num_pizzas = 3 →
  num_toppings = 4 →
  cost_per_topping = 1 →
  tip = 5 →
  total_cost = 39 →
  ∃ (cost_per_pizza : ℚ), 
    cost_per_pizza = 10 ∧ 
    num_pizzas * cost_per_pizza + num_toppings * cost_per_topping + tip = total_cost :=
by sorry

end NUMINAMATH_CALUDE_pizza_cost_per_pizza_l3656_365685


namespace NUMINAMATH_CALUDE_thirteenth_result_l3656_365678

theorem thirteenth_result (total_count : Nat) (total_avg : ℚ) (first_12_avg : ℚ) (last_12_avg : ℚ) 
  (h_total_count : total_count = 25)
  (h_total_avg : total_avg = 20)
  (h_first_12_avg : first_12_avg = 14)
  (h_last_12_avg : last_12_avg = 17) :
  ∃ (thirteenth : ℚ), 
    (total_count : ℚ) * total_avg = 
      12 * first_12_avg + thirteenth + 12 * last_12_avg ∧ 
    thirteenth = 128 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_result_l3656_365678


namespace NUMINAMATH_CALUDE_bits_of_base16_ABCD_l3656_365696

/-- The number of bits in the binary representation of a base-16 number ABCD₁₆ --/
theorem bits_of_base16_ABCD : ∃ (A B C D : ℕ), 
  A < 16 ∧ B < 16 ∧ C < 16 ∧ D < 16 →
  let base16_value := A * 16^3 + B * 16^2 + C * 16^1 + D * 16^0
  let binary_repr := Nat.bits base16_value
  binary_repr.length = 16 := by
  sorry

end NUMINAMATH_CALUDE_bits_of_base16_ABCD_l3656_365696


namespace NUMINAMATH_CALUDE_prime_factor_difference_l3656_365640

theorem prime_factor_difference (n : Nat) (h : n = 278459) :
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Prime r → r ∣ n → p ≥ r ∧ r ≥ q) ∧
  p - q = 254 := by
  sorry

end NUMINAMATH_CALUDE_prime_factor_difference_l3656_365640


namespace NUMINAMATH_CALUDE_v_equation_l3656_365607

/-- Given that V = kZ - 6 and V = 14 when Z = 5, prove that V = 22 when Z = 7 -/
theorem v_equation (k : ℝ) : 
  (∀ Z, (k * Z - 6 = 14) → (Z = 5)) →
  (k * 7 - 6 = 22) :=
by sorry

end NUMINAMATH_CALUDE_v_equation_l3656_365607


namespace NUMINAMATH_CALUDE_initial_amount_equation_l3656_365658

/-- The initial amount Kanul had, in dollars -/
def initial_amount : ℝ := 7058.82

/-- The amount spent on raw materials, in dollars -/
def raw_materials : ℝ := 3000

/-- The amount spent on machinery, in dollars -/
def machinery : ℝ := 2000

/-- The percentage of the initial amount spent as cash -/
def cash_percentage : ℝ := 0.15

/-- The amount spent on labor costs, in dollars -/
def labor_costs : ℝ := 1000

/-- Theorem stating that the initial amount satisfies the equation -/
theorem initial_amount_equation :
  initial_amount = raw_materials + machinery + cash_percentage * initial_amount + labor_costs := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_equation_l3656_365658


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3656_365609

theorem quadratic_equation_solutions :
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3656_365609


namespace NUMINAMATH_CALUDE_frame_diameter_l3656_365638

/-- Given two circular frames X and Y, where X has a diameter of 16 cm and Y covers 0.5625 of X's area, prove that Y's diameter is 12 cm. -/
theorem frame_diameter (dX : ℝ) (coverage : ℝ) (dY : ℝ) : 
  dX = 16 → coverage = 0.5625 → dY = 12 → 
  (π * (dY / 2)^2) = coverage * (π * (dX / 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_frame_diameter_l3656_365638


namespace NUMINAMATH_CALUDE_equation_solutions_l3656_365631

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 - Real.sqrt 2 ∧ x₂ = 2 + Real.sqrt 2 ∧
    x₁^2 - 4*x₁ = 4 ∧ x₂^2 - 4*x₂ = 4) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 2 ∧
    (x₁ + 2)*(x₁ + 1) = 12 ∧ (x₂ + 2)*(x₂ + 1) = 12) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 5/2 ∧ x₂ = 5 ∧
    0.2*x₁^2 + 5/2 = 3/2*x₁ ∧ 0.2*x₂^2 + 5/2 = 3/2*x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3656_365631


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l3656_365692

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 4) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l3656_365692


namespace NUMINAMATH_CALUDE_pirate_coin_division_l3656_365657

theorem pirate_coin_division (n m : ℕ) : 
  n % 10 = 5 → m = 2 * n → m % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_pirate_coin_division_l3656_365657


namespace NUMINAMATH_CALUDE_angle_sum_equality_l3656_365614

theorem angle_sum_equality (α β : Real) : 
  0 < α ∧ α < Real.pi/2 ∧ 
  0 < β ∧ β < Real.pi/2 ∧ 
  Real.cos α = 7/Real.sqrt 50 ∧ 
  Real.tan β = 1/3 → 
  α + 2*β = Real.pi/4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equality_l3656_365614


namespace NUMINAMATH_CALUDE_two_white_socks_cost_45_cents_l3656_365683

/-- The cost of a single brown sock in cents -/
def brown_sock_cost : ℕ := 300 / 15

/-- The cost of two white socks in cents -/
def white_socks_cost : ℕ := brown_sock_cost + 25

theorem two_white_socks_cost_45_cents : white_socks_cost = 45 := by
  sorry

#eval white_socks_cost

end NUMINAMATH_CALUDE_two_white_socks_cost_45_cents_l3656_365683


namespace NUMINAMATH_CALUDE_dans_earnings_difference_l3656_365684

/-- Calculates the difference in earnings between two sets of tasks -/
def earningsDifference (numTasks1 : ℕ) (rate1 : ℚ) (numTasks2 : ℕ) (rate2 : ℚ) : ℚ :=
  numTasks1 * rate1 - numTasks2 * rate2

/-- Proves that the difference in earnings between 400 tasks at $0.25 each and 5 tasks at $2.00 each is $90 -/
theorem dans_earnings_difference :
  earningsDifference 400 (25 / 100) 5 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_dans_earnings_difference_l3656_365684


namespace NUMINAMATH_CALUDE_rahul_mary_age_difference_l3656_365662

/-- 
Given:
- Mary's current age is 10 years
- In 20 years, Rahul will be twice as old as Mary

Prove that Rahul is currently 30 years older than Mary
-/
theorem rahul_mary_age_difference :
  ∀ (rahul_age mary_age : ℕ),
    mary_age = 10 →
    rahul_age + 20 = 2 * (mary_age + 20) →
    rahul_age - mary_age = 30 :=
by sorry

end NUMINAMATH_CALUDE_rahul_mary_age_difference_l3656_365662


namespace NUMINAMATH_CALUDE_seashells_given_l3656_365621

theorem seashells_given (initial : ℕ) (left : ℕ) (given : ℕ) : 
  initial ≥ left → given = initial - left → given = 62 - 13 :=
by sorry

end NUMINAMATH_CALUDE_seashells_given_l3656_365621


namespace NUMINAMATH_CALUDE_worker_savings_percentage_l3656_365632

theorem worker_savings_percentage
  (last_year_salary : ℝ)
  (last_year_savings_percentage : ℝ)
  (this_year_salary_increase : ℝ)
  (this_year_savings_percentage : ℝ)
  (h1 : this_year_salary_increase = 0.20)
  (h2 : this_year_savings_percentage = 0.05)
  (h3 : this_year_savings_percentage * (1 + this_year_salary_increase) * last_year_salary = last_year_savings_percentage * last_year_salary)
  : last_year_savings_percentage = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_worker_savings_percentage_l3656_365632


namespace NUMINAMATH_CALUDE_sandy_initial_fish_count_l3656_365661

theorem sandy_initial_fish_count (initial_fish final_fish bought_fish : ℕ) 
  (h1 : final_fish = initial_fish + bought_fish)
  (h2 : final_fish = 32)
  (h3 : bought_fish = 6) : 
  initial_fish = 26 := by
sorry

end NUMINAMATH_CALUDE_sandy_initial_fish_count_l3656_365661


namespace NUMINAMATH_CALUDE_circle_center_sum_l3656_365654

/-- Given a circle defined by the equation x^2 + y^2 + 6x - 4y - 12 = 0,
    if (a, b) is the center of this circle, then a + b = -1. -/
theorem circle_center_sum (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 6*x - 4*y - 12 = 0 ↔ (x - a)^2 + (y - b)^2 = (a^2 + b^2 + 6*a - 4*b - 12)) →
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3656_365654


namespace NUMINAMATH_CALUDE_complement_of_intersection_equals_expected_l3656_365605

-- Define the sets M and N
def M : Set ℝ := {x | x ≥ 1/3}
def N : Set ℝ := {x | 0 < x ∧ x < 1/2}

-- Define the complement of the intersection
def complementOfIntersection : Set ℝ := {x | x < 1/3 ∨ x ≥ 1/2}

-- Theorem statement
theorem complement_of_intersection_equals_expected :
  complementOfIntersection = (Set.Iic (1/3 : ℝ)).diff {1/3} ∪ Set.Ici (1/2 : ℝ) := by
  sorry

#check complement_of_intersection_equals_expected

end NUMINAMATH_CALUDE_complement_of_intersection_equals_expected_l3656_365605


namespace NUMINAMATH_CALUDE_neglart_hands_count_l3656_365676

/-- Represents a race on planet Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of toes on each hand for a given race -/
def toes_per_hand (race : Race) : ℕ :=
  match race with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of hands for Hoopits -/
def hoopit_hands : ℕ := 4

/-- Number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating the number of hands each Neglart has -/
theorem neglart_hands_count :
  ∃ (neglart_hands : ℕ),
    neglart_hands * neglart_students * toes_per_hand Race.Neglart +
    hoopit_hands * hoopit_students * toes_per_hand Race.Hoopit = total_toes ∧
    neglart_hands = 5 := by
  sorry

end NUMINAMATH_CALUDE_neglart_hands_count_l3656_365676


namespace NUMINAMATH_CALUDE_exp_greater_than_log_squared_l3656_365693

open Real

theorem exp_greater_than_log_squared (x : ℝ) (h : x > 0) : exp x - exp 2 * log x > 0 := by
  sorry

end NUMINAMATH_CALUDE_exp_greater_than_log_squared_l3656_365693


namespace NUMINAMATH_CALUDE_circle_symmetry_l3656_365601

/-- The original circle -/
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y - 4 = 0

/-- The line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  x - y - 1 = 0

/-- The symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 2)^2 = 9

/-- Theorem stating that the symmetric circle is indeed symmetric to the original circle
    with respect to the given line of symmetry -/
theorem circle_symmetry :
  ∀ (x y : ℝ), original_circle x y ↔ 
  ∃ (x' y' : ℝ), symmetric_circle x' y' ∧ 
  ((x + x')/2 = (y + y')/2 + 1) ∧
  (y' - y)/(x' - x) = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3656_365601


namespace NUMINAMATH_CALUDE_reciprocal_difference_problem_l3656_365653

theorem reciprocal_difference_problem (m : ℚ) (hm : m ≠ 1) (h : 1 / (m - 1) = m) :
  m^4 + 1 / m^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_difference_problem_l3656_365653


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3656_365618

theorem negation_of_proposition (n : ℕ) :
  ¬(2^n > 1000) ↔ (2^n ≤ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3656_365618


namespace NUMINAMATH_CALUDE_ellipse_sum_l3656_365649

/-- Represents an ellipse with center (h, k), semi-major axis a, and semi-minor axis c -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  c : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.c^2 = 1

theorem ellipse_sum (e : Ellipse) 
    (center_h : e.h = 3)
    (center_k : e.k = -5)
    (major_axis : e.a = 7)
    (minor_axis : e.c = 4) :
  e.h + e.k + e.a + e.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l3656_365649


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3656_365668

theorem trigonometric_identity (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.cos (α + Real.pi / 6) = 4 / 5) : 
  Real.sin (2 * α + Real.pi / 3) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3656_365668


namespace NUMINAMATH_CALUDE_cos_angle_relation_l3656_365650

theorem cos_angle_relation (α : ℝ) (h : Real.cos (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 - α) = -(Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_relation_l3656_365650


namespace NUMINAMATH_CALUDE_mobile_wire_left_l3656_365699

/-- The amount of wire left after making mobiles -/
def wire_left (total_wire : ℚ) (wire_per_mobile : ℚ) : ℚ :=
  total_wire - wire_per_mobile * ⌊total_wire / wire_per_mobile⌋

/-- Converts millimeters to centimeters -/
def mm_to_cm (mm : ℚ) : ℚ :=
  mm / 10

theorem mobile_wire_left : 
  mm_to_cm (wire_left 117.6 4) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_mobile_wire_left_l3656_365699


namespace NUMINAMATH_CALUDE_part_one_part_two_l3656_365606

-- Define propositions P and Q
def P (a : ℝ) : Prop := ∀ x ∈ (Set.Icc 1 2), x^2 - a ≥ 0
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem for part (1)
theorem part_one (a : ℝ) : P a → a ≤ 1 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a > 1 ∨ (-2 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3656_365606


namespace NUMINAMATH_CALUDE_mortgage_food_ratio_is_three_to_one_l3656_365636

/-- Esperanza's monthly finances -/
structure MonthlyFinances where
  rent : ℕ
  food_ratio : ℚ
  savings : ℕ
  tax_ratio : ℚ
  gross_salary : ℕ

/-- Calculate the ratio of mortgage bill to food expenses -/
def mortgage_to_food_ratio (finances : MonthlyFinances) : ℚ :=
  let food_expense := finances.food_ratio * finances.rent
  let taxes := finances.tax_ratio * finances.savings
  let total_expenses := finances.rent + food_expense + finances.savings + taxes
  let mortgage := finances.gross_salary - total_expenses
  mortgage / food_expense

/-- Theorem stating the ratio of mortgage bill to food expenses -/
theorem mortgage_food_ratio_is_three_to_one :
  let esperanza_finances : MonthlyFinances := {
    rent := 600,
    food_ratio := 3/5,
    savings := 2000,
    tax_ratio := 2/5,
    gross_salary := 4840
  }
  mortgage_to_food_ratio esperanza_finances = 3 := by
  sorry


end NUMINAMATH_CALUDE_mortgage_food_ratio_is_three_to_one_l3656_365636


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3656_365679

theorem sufficient_not_necessary (p q : Prop) :
  (∃ (h : p ∧ q), ¬p = False) ∧
  (∃ (h : ¬p = False), ¬(p ∧ q = True)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3656_365679


namespace NUMINAMATH_CALUDE_apple_profit_percentage_l3656_365686

/-- Calculates the total profit percentage for a shopkeeper selling apples -/
theorem apple_profit_percentage
  (total_apples : ℝ)
  (percent_sold_at_low_profit : ℝ)
  (percent_sold_at_high_profit : ℝ)
  (low_profit_rate : ℝ)
  (high_profit_rate : ℝ)
  (h1 : total_apples = 280)
  (h2 : percent_sold_at_low_profit = 0.4)
  (h3 : percent_sold_at_high_profit = 0.6)
  (h4 : low_profit_rate = 0.1)
  (h5 : high_profit_rate = 0.3)
  (h6 : percent_sold_at_low_profit + percent_sold_at_high_profit = 1) :
  let cost_price := 1
  let low_profit_quantity := percent_sold_at_low_profit * total_apples
  let high_profit_quantity := percent_sold_at_high_profit * total_apples
  let total_cost := total_apples * cost_price
  let low_profit_revenue := low_profit_quantity * cost_price * (1 + low_profit_rate)
  let high_profit_revenue := high_profit_quantity * cost_price * (1 + high_profit_rate)
  let total_revenue := low_profit_revenue + high_profit_revenue
  let total_profit := total_revenue - total_cost
  let profit_percentage := (total_profit / total_cost) * 100
  profit_percentage = 22 := by sorry

end NUMINAMATH_CALUDE_apple_profit_percentage_l3656_365686


namespace NUMINAMATH_CALUDE_shaded_to_large_square_ratio_l3656_365647

theorem shaded_to_large_square_ratio :
  let large_square_side : ℕ := 5
  let unit_squares_count : ℕ := large_square_side ^ 2
  let half_squares_in_shaded : ℕ := 5
  let shaded_area : ℚ := (half_squares_in_shaded : ℚ) / 2
  let large_square_area : ℕ := unit_squares_count
  (shaded_area : ℚ) / (large_square_area : ℚ) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_large_square_ratio_l3656_365647


namespace NUMINAMATH_CALUDE_dans_marbles_l3656_365688

/-- The number of violet marbles Dan has -/
def violet_marbles : ℕ := 64

/-- The number of red marbles Mary gave to Dan -/
def red_marbles : ℕ := 14

/-- The total number of marbles Dan has now -/
def total_marbles : ℕ := violet_marbles + red_marbles

theorem dans_marbles : total_marbles = 78 := by
  sorry

end NUMINAMATH_CALUDE_dans_marbles_l3656_365688


namespace NUMINAMATH_CALUDE_sally_found_two_balloons_l3656_365665

/-- The number of additional orange balloons Sally found -/
def additional_balloons (initial final : ℝ) : ℝ := final - initial

/-- Theorem stating that Sally found 2.0 more orange balloons -/
theorem sally_found_two_balloons (initial final : ℝ) 
  (h1 : initial = 9.0) 
  (h2 : final = 11) : 
  additional_balloons initial final = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_sally_found_two_balloons_l3656_365665


namespace NUMINAMATH_CALUDE_total_turtles_count_l3656_365694

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The number of turtles Marion received -/
def marion_turtles : ℕ := martha_turtles + 20

/-- The total number of turtles received by Marion and Martha -/
def total_turtles : ℕ := marion_turtles + martha_turtles

theorem total_turtles_count : total_turtles = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_turtles_count_l3656_365694


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3656_365695

theorem algebraic_simplification (x y : ℝ) (h : y ≠ 0) :
  (25 * x^3 * y) * (8 * x * y) * (1 / (5 * x * y^2)^2) = 8 * x^2 / y^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3656_365695


namespace NUMINAMATH_CALUDE_vector_properties_l3656_365619

def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-2, 1)

theorem vector_properties : 
  (a.1 * b.1 + a.2 * b.2 = 0) ∧ 
  (((a.1 + b.1)^2 + (a.2 + b.2)^2).sqrt = 5) ∧
  (((a.1 - b.1)^2 + (a.2 - b.2)^2).sqrt = 5) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l3656_365619


namespace NUMINAMATH_CALUDE_range_of_f_l3656_365644

-- Define the function f
def f (x : ℝ) : ℝ := |1 - x| - |x - 3|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y ∈ Set.range f, -2 ≤ y ∧ y ≤ 2 ∧
  ∃ x₁ x₂ : ℝ, f x₁ = -2 ∧ f x₂ = 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l3656_365644


namespace NUMINAMATH_CALUDE_inequality_problem_l3656_365674

theorem inequality_problem (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  ¬(1 / (a - b) > 1 / a) := by
sorry

end NUMINAMATH_CALUDE_inequality_problem_l3656_365674


namespace NUMINAMATH_CALUDE_function_properties_l3656_365663

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
def odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc a b → f (-x) = -f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x > f y

-- State the theorem
theorem function_properties :
  odd_on_interval f (-1) 1 ∧ decreasing_on_interval f (-1) 1 →
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → 
    (f x₁ + f x₂) * (x₁ + x₂) ≤ 0) ∧
  (∀ a, f (1 - a) + f ((1 - a)^2) < 0 → a ∈ Set.Ico 0 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3656_365663


namespace NUMINAMATH_CALUDE_divisible_by_eight_l3656_365669

theorem divisible_by_eight (n : ℕ) : 
  8 ∣ (5^n + 2 * 3^(n-1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_l3656_365669


namespace NUMINAMATH_CALUDE_product_reciprocal_sum_l3656_365667

theorem product_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 12 → (1 / x) = 3 * (1 / y) → x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_reciprocal_sum_l3656_365667


namespace NUMINAMATH_CALUDE_root_product_expression_l3656_365666

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) 
  (h1 : α^2 + p*α + 2 = 0) 
  (h2 : β^2 + p*β + 2 = 0)
  (h3 : γ^2 + q*γ + 2 = 0)
  (h4 : δ^2 + q*δ + 2 = 0) :
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = -2*(p^2 - q^2) + 4 := by
  sorry

end NUMINAMATH_CALUDE_root_product_expression_l3656_365666


namespace NUMINAMATH_CALUDE_exists_solution_l3656_365612

open Complex

/-- The equation that z must satisfy -/
def equation (z : ℂ) : Prop :=
  z * (z + I) * (z - 2 + I) * (z + 3*I) = 2018 * I

/-- The condition that b should be maximized -/
def b_maximized (z : ℂ) : Prop :=
  ∀ w : ℂ, equation w → z.im ≥ w.im

/-- The main theorem stating the existence of z satisfying the conditions -/
theorem exists_solution :
  ∃ z : ℂ, equation z ∧ b_maximized z :=
sorry

/-- Helper lemma to extract the real part of the solution -/
lemma solution_real_part (z : ℂ) (h : equation z ∧ b_maximized z) :
  ∃ a : ℝ, z.re = a :=
sorry

end NUMINAMATH_CALUDE_exists_solution_l3656_365612


namespace NUMINAMATH_CALUDE_set_A_equals_singleton_l3656_365698

-- Define the set A
def A : Set (ℕ × ℕ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.2 = 6 / (p.1 + 3)}

-- State the theorem
theorem set_A_equals_singleton : A = {(3, 1)} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_singleton_l3656_365698


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l3656_365625

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The absolute difference of distances from a point on the hyperbola to the foci -/
  vertex_distance : ℝ
  /-- The eccentricity of the hyperbola -/
  eccentricity : ℝ

/-- Calculates the length of the focal distance of a hyperbola -/
def focal_distance (h : Hyperbola) : ℝ :=
  h.vertex_distance * h.eccentricity

/-- Theorem stating that for a hyperbola with given properties, the focal distance is 10 -/
theorem hyperbola_focal_distance :
  ∀ h : Hyperbola, h.vertex_distance = 6 ∧ h.eccentricity = 5/3 → focal_distance h = 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l3656_365625


namespace NUMINAMATH_CALUDE_count_multiples_of_30_l3656_365620

def smallest_square_multiple_of_30 : ℕ := 900
def smallest_cube_multiple_of_30 : ℕ := 27000

theorem count_multiples_of_30 :
  (Finset.range ((smallest_cube_multiple_of_30 - smallest_square_multiple_of_30) / 30 + 1)).card = 871 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_30_l3656_365620


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3656_365645

/-- A structure representing a 3D geometric space with lines and planes. -/
structure GeometricSpace where
  Line : Type
  Plane : Type
  parallelLinePlane : Line → Plane → Prop
  perpendicularLinePlane : Line → Plane → Prop
  parallelPlanes : Plane → Plane → Prop
  perpendicularLines : Line → Line → Prop

/-- Theorem stating the relationship between parallel planes and perpendicular lines. -/
theorem perpendicular_lines_from_parallel_planes 
  (S : GeometricSpace) 
  (α β : S.Plane) 
  (m n : S.Line) :
  S.parallelPlanes α β →
  S.perpendicularLinePlane m α →
  S.parallelLinePlane n β →
  S.perpendicularLines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3656_365645


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_l3656_365664

theorem sqrt_expression_equals_three : 
  (Real.sqrt 3 - 2) * Real.sqrt 3 + Real.sqrt 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_l3656_365664


namespace NUMINAMATH_CALUDE_average_of_x_and_y_l3656_365616

theorem average_of_x_and_y (x y : ℝ) : 
  (4 + 6.5 + 8 + x + y) / 5 = 18 → (x + y) / 2 = 35.75 := by
  sorry

end NUMINAMATH_CALUDE_average_of_x_and_y_l3656_365616


namespace NUMINAMATH_CALUDE_mac_loss_calculation_l3656_365635

-- Define exchange rates
def canadian_dime_usd : ℝ := 0.075
def canadian_penny_usd : ℝ := 0.0075
def mexican_centavo_usd : ℝ := 0.0045
def cuban_centavo_usd : ℝ := 0.0036
def euro_cent_usd : ℝ := 0.011
def uk_pence_usd : ℝ := 0.013
def canadian_nickel_usd : ℝ := 0.038
def us_half_dollar_usd : ℝ := 0.5
def brazilian_centavo_usd : ℝ := 0.0019
def australian_cent_usd : ℝ := 0.0072
def indian_paisa_usd : ℝ := 0.0013
def mexican_peso_usd : ℝ := 0.045
def japanese_yen_usd : ℝ := 0.0089

-- Define daily trades
def day1_trade : ℝ := 6 * canadian_dime_usd + 2 * canadian_penny_usd
def day2_trade : ℝ := 10 * mexican_centavo_usd + 5 * cuban_centavo_usd
def day3_trade : ℝ := 4 * 0.1 + 1 * euro_cent_usd
def day4_trade : ℝ := 7 * uk_pence_usd + 5 * canadian_nickel_usd
def day5_trade : ℝ := 3 * us_half_dollar_usd + 2 * brazilian_centavo_usd
def day6_trade : ℝ := 12 * australian_cent_usd + 3 * indian_paisa_usd
def day7_trade : ℝ := 8 * mexican_peso_usd + 6 * japanese_yen_usd

-- Define quarter value
def quarter_value : ℝ := 0.25

-- Theorem statement
theorem mac_loss_calculation :
  (day1_trade - quarter_value) +
  (quarter_value - day2_trade) +
  (day3_trade - quarter_value) +
  (day4_trade - quarter_value) +
  (day5_trade - quarter_value) +
  (quarter_value - day6_trade) +
  (day7_trade - quarter_value) = 2.1619 :=
by sorry

end NUMINAMATH_CALUDE_mac_loss_calculation_l3656_365635


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_squares_is_integer_l3656_365610

/-- An arithmetic progression containing the squares of its first three terms consists of integers. -/
theorem arithmetic_progression_with_squares_is_integer (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic progression condition
  (∃ k l m : ℕ, a k = (a 1)^2 ∧ a l = (a 2)^2 ∧ a m = (a 3)^2) →  -- squares condition
  ∀ n, ∃ z : ℤ, a n = z :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_squares_is_integer_l3656_365610


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l3656_365691

theorem polynomial_product_sum (k j : ℚ) : 
  (∀ d, (8*d^2 - 4*d + k) * (4*d^2 + j*d - 10) = 32*d^4 - 56*d^3 - 68*d^2 + 28*d - 90) →
  k + j = 23/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l3656_365691


namespace NUMINAMATH_CALUDE_min_value_h_l3656_365682

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

noncomputable def g (x : ℝ) : ℝ := x * Real.exp x

noncomputable def h (x : ℝ) : ℝ := f x / g x

theorem min_value_h :
  ∃ (min : ℝ), min = 2 / Real.pi ∧
  ∀ x ∈ Set.Ioo 0 (Real.pi / 2), h x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_h_l3656_365682


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3656_365641

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > a → x > 2) ∧ (∃ x, x > 2 ∧ x ≤ a) ↔ a > 2 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3656_365641


namespace NUMINAMATH_CALUDE_expand_expression_l3656_365675

theorem expand_expression (y z : ℝ) : 
  -2 * (5 * y^3 - 3 * y^2 * z + 4 * y * z^2 - z^3) = 
  -10 * y^3 + 6 * y^2 * z - 8 * y * z^2 + 2 * z^3 := by
sorry

end NUMINAMATH_CALUDE_expand_expression_l3656_365675


namespace NUMINAMATH_CALUDE_smallest_y_divisible_by_11_l3656_365646

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def number_with_y (y : ℕ) : ℕ :=
  7000000 + y * 100000 + 86038

theorem smallest_y_divisible_by_11 :
  ∀ y : ℕ, y < 14 → ¬(is_divisible_by_11 (number_with_y y)) ∧
  is_divisible_by_11 (number_with_y 14) := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_divisible_by_11_l3656_365646


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l3656_365659

theorem maximum_marks_calculation (percentage : ℝ) (obtained_marks : ℝ) (max_marks : ℝ) : 
  percentage = 95 → obtained_marks = 285 → 
  (obtained_marks / max_marks) * 100 = percentage → 
  max_marks = 300 := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l3656_365659


namespace NUMINAMATH_CALUDE_larger_number_problem_l3656_365643

theorem larger_number_problem (x y : ℝ) (h1 : y > x) (h2 : 4 * y = 5 * x) (h3 : y - x = 10) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3656_365643


namespace NUMINAMATH_CALUDE_min_draw_count_correct_l3656_365630

/-- Represents a box of colored balls -/
structure ColoredBallBox where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat

/-- The setup of the two boxes -/
def box1 : ColoredBallBox := ⟨40, 30, 25, 15⟩
def box2 : ColoredBallBox := ⟨35, 25, 20, 0⟩

/-- The target number of balls of a single color -/
def targetCount : Nat := 20

/-- The minimum number of balls to draw -/
def minDrawCount : Nat := 73

/-- Theorem stating the minimum number of balls to draw -/
theorem min_draw_count_correct : 
  ∀ (draw : Nat), draw < minDrawCount → 
  ∃ (redCount greenCount yellowCount blueCount : Nat),
    redCount < targetCount ∧
    greenCount < targetCount ∧
    yellowCount < targetCount ∧
    blueCount < targetCount ∧
    redCount + greenCount + yellowCount + blueCount = draw ∧
    redCount ≤ box1.red + box2.red ∧
    greenCount ≤ box1.green + box2.green ∧
    yellowCount ≤ box1.yellow + box2.yellow ∧
    blueCount ≤ box1.blue + box2.blue :=
by sorry

#check min_draw_count_correct

end NUMINAMATH_CALUDE_min_draw_count_correct_l3656_365630


namespace NUMINAMATH_CALUDE_ratio_equality_l3656_365656

theorem ratio_equality : ∃ x : ℚ, (3/4) / (1/2) = x / (2/6) ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3656_365656


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3656_365689

/-- Given two adjacent points (2,1) and (2,7) on a square in a Cartesian coordinate plane,
    the area of the square is 36. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (2, 7)
  let square_side := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  square_side^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3656_365689


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3656_365623

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  3 * x^2 * y + 12 * x^2 * y^2 + 12 * x * y^3 = 3 * x * y * (x + 4 * x * y + 4 * y^2) := by
  sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  2 * a^5 * b - 2 * a * b^5 = 2 * a * b * (a^2 + b^2) * (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3656_365623


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l3656_365604

/-- Represents the possible weather outcomes for a day -/
inductive Weather
  | Sun
  | Rain2Inches
  | Rain8Inches

/-- The probability of each weather outcome -/
def weather_probability : Weather → ℝ
  | Weather.Sun => 0.35
  | Weather.Rain2Inches => 0.40
  | Weather.Rain8Inches => 0.25

/-- The amount of rainfall for each weather outcome -/
def rainfall_amount : Weather → ℝ
  | Weather.Sun => 0
  | Weather.Rain2Inches => 2
  | Weather.Rain8Inches => 8

/-- The number of days in the week -/
def days_in_week : ℕ := 7

/-- The expected rainfall for a single day -/
def expected_daily_rainfall : ℝ :=
  (weather_probability Weather.Sun * rainfall_amount Weather.Sun) +
  (weather_probability Weather.Rain2Inches * rainfall_amount Weather.Rain2Inches) +
  (weather_probability Weather.Rain8Inches * rainfall_amount Weather.Rain8Inches)

/-- Theorem: The expected total rainfall for the week is 19.6 inches -/
theorem expected_weekly_rainfall :
  (days_in_week : ℝ) * expected_daily_rainfall = 19.6 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l3656_365604


namespace NUMINAMATH_CALUDE_marker_cost_l3656_365697

theorem marker_cost (total_students : Nat) (buyers : Nat) (markers_per_student : Nat) (total_cost : Nat) :
  total_students = 40 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  markers_per_student % 2 = 0 →
  markers_per_student > 2 →
  total_cost = 3185 →
  ∃ (cost_per_marker : Nat),
    cost_per_marker > markers_per_student ∧
    buyers * markers_per_student * cost_per_marker = total_cost ∧
    cost_per_marker = 13 :=
by sorry

end NUMINAMATH_CALUDE_marker_cost_l3656_365697


namespace NUMINAMATH_CALUDE_clothes_to_total_ratio_l3656_365628

def weekly_allowance_1 : ℕ := 5
def weeks_1 : ℕ := 8
def weekly_allowance_2 : ℕ := 6
def weeks_2 : ℕ := 6
def video_game_cost : ℕ := 35
def money_left : ℕ := 3

def total_saved : ℕ := weekly_allowance_1 * weeks_1 + weekly_allowance_2 * weeks_2
def spent_on_video_game_and_left : ℕ := video_game_cost + money_left
def spent_on_clothes : ℕ := total_saved - spent_on_video_game_and_left

theorem clothes_to_total_ratio :
  (spent_on_clothes : ℚ) / total_saved = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_clothes_to_total_ratio_l3656_365628


namespace NUMINAMATH_CALUDE_tan_theta_plus_pi_over_8_minus_reciprocal_l3656_365634

theorem tan_theta_plus_pi_over_8_minus_reciprocal (θ : Real) 
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) : 
  Real.tan (θ + π/8) - (1 / Real.tan (θ + π/8)) = -14 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_plus_pi_over_8_minus_reciprocal_l3656_365634


namespace NUMINAMATH_CALUDE_license_plate_count_l3656_365622

/-- The number of vowels (excluding Y) -/
def num_vowels : Nat := 5

/-- The number of digits between 1 and 5 -/
def num_digits : Nat := 5

/-- The number of consonants (including Y) -/
def num_consonants : Nat := 26 - num_vowels

/-- The total number of license plates meeting the specified criteria -/
def total_plates : Nat := num_vowels * num_digits * num_consonants * num_consonants * num_vowels

theorem license_plate_count : total_plates = 55125 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3656_365622
