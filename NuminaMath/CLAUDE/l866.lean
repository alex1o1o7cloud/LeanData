import Mathlib

namespace NUMINAMATH_CALUDE_sum_exterior_angles_regular_decagon_l866_86627

/-- A regular decagon is a polygon with 10 sides -/
def RegularDecagon : Type := Unit

/-- The sum of exterior angles of a polygon -/
def SumExteriorAngles (p : Type) : ℝ := sorry

/-- Theorem: The sum of exterior angles of a regular decagon is 360° -/
theorem sum_exterior_angles_regular_decagon :
  SumExteriorAngles RegularDecagon = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_regular_decagon_l866_86627


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_seven_l866_86694

theorem smallest_positive_multiple_of_seven (x : ℕ) : 
  (∃ k : ℕ, x = 7 * k) → -- x is a positive multiple of 7
  x^2 > 144 →            -- x^2 > 144
  x < 25 →               -- x < 25
  x = 14 :=              -- x = 14 is the smallest value satisfying all conditions
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_seven_l866_86694


namespace NUMINAMATH_CALUDE_fraction_equality_implies_value_l866_86662

theorem fraction_equality_implies_value (a : ℝ) (x : ℝ) :
  (a - 2) / x = 1 / (2 * a + 7) → x = 2 * a^2 + 3 * a - 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_value_l866_86662


namespace NUMINAMATH_CALUDE_cross_product_example_l866_86605

/-- The cross product of two 3D vectors -/
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

theorem cross_product_example : 
  cross_product (3, 2, -1) (-2, 4, 6) = (16, -16, 16) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_example_l866_86605


namespace NUMINAMATH_CALUDE_n_fourth_plus_four_composite_l866_86667

theorem n_fourth_plus_four_composite (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
by sorry

end NUMINAMATH_CALUDE_n_fourth_plus_four_composite_l866_86667


namespace NUMINAMATH_CALUDE_planted_field_fraction_l866_86601

theorem planted_field_fraction (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_leg1 : a = 5) (h_leg2 : b = 12) (s : ℝ) (h_distance : 3 / 5 = s / (s + 3)) :
  (a * b / 2 - s^2) / (a * b / 2) = 13 / 40 := by
  sorry

end NUMINAMATH_CALUDE_planted_field_fraction_l866_86601


namespace NUMINAMATH_CALUDE_pet_store_parakeets_l866_86670

/-- Calculates the number of parakeets in a pet store given the number of cages, parrots, and average birds per cage. -/
theorem pet_store_parakeets 
  (num_cages : ℝ) 
  (num_parrots : ℝ) 
  (avg_birds_per_cage : ℝ) 
  (h1 : num_cages = 6)
  (h2 : num_parrots = 6)
  (h3 : avg_birds_per_cage = 1.333333333) :
  num_cages * avg_birds_per_cage - num_parrots = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_parakeets_l866_86670


namespace NUMINAMATH_CALUDE_expected_rain_total_l866_86629

/-- The number of days in the weather forecast. -/
def num_days : ℕ := 5

/-- The probability of a sunny day with no rain. -/
def prob_sun : ℝ := 0.4

/-- The probability of a day with 4 inches of rain. -/
def prob_rain_4 : ℝ := 0.25

/-- The probability of a day with 10 inches of rain. -/
def prob_rain_10 : ℝ := 0.35

/-- The amount of rain on a sunny day. -/
def rain_sun : ℝ := 0

/-- The amount of rain on a day with 4 inches of rain. -/
def rain_4 : ℝ := 4

/-- The amount of rain on a day with 10 inches of rain. -/
def rain_10 : ℝ := 10

/-- The expected value of rain for a single day. -/
def expected_rain_day : ℝ :=
  prob_sun * rain_sun + prob_rain_4 * rain_4 + prob_rain_10 * rain_10

/-- Theorem: The expected value of the total number of inches of rain for 5 days is 22.5 inches. -/
theorem expected_rain_total : num_days * expected_rain_day = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_rain_total_l866_86629


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l866_86651

theorem geometric_sequence_common_ratio
  (a : ℝ)
  (seq : ℕ → ℝ)
  (h_seq : ∀ n : ℕ, seq n = a + Real.log 3 / Real.log (2^(2^n)))
  : (∃ q : ℝ, ∀ n : ℕ, seq (n + 1) = q * seq n) ∧
    (∀ q : ℝ, (∀ n : ℕ, seq (n + 1) = q * seq n) → q = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l866_86651


namespace NUMINAMATH_CALUDE_intersection_M_N_l866_86641

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l866_86641


namespace NUMINAMATH_CALUDE_sum_of_roots_l866_86615

theorem sum_of_roots (a β : ℝ) (ha : a^2 - 2*a = 1) (hβ : β^2 - 2*β - 1 = 0) (hneq : a ≠ β) :
  a + β = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l866_86615


namespace NUMINAMATH_CALUDE_three_cubes_sum_equals_three_to_fourth_l866_86660

theorem three_cubes_sum_equals_three_to_fourth : 3^3 + 3^3 + 3^3 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_three_cubes_sum_equals_three_to_fourth_l866_86660


namespace NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l866_86636

-- Define the angle α
def α : Real := sorry

-- Define the conditions
axiom cos_α : Real.cos α = -1/5
axiom sin_α : Real.sin α = 2 * Real.sqrt 6 / 5

-- Define the second quadrant
def second_quadrant (θ : Real) : Prop :=
  Real.cos θ < 0 ∧ Real.sin θ > 0

-- Theorem to prove
theorem terminal_side_in_second_quadrant : second_quadrant α := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l866_86636


namespace NUMINAMATH_CALUDE_line_equation_l866_86639

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + (y-1)^2 = 5

-- Define the line l
def Line (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

-- Define the condition that P(1,1) satisfies 2⃗AP = ⃗PB
def PointCondition (xa ya xb yb : ℝ) : Prop :=
  2*(1 - xa, 1 - ya) = (xb - 1, yb - 1)

theorem line_equation :
  ∀ (m : ℝ) (xa ya xb yb : ℝ),
    Circle xa ya → Circle xb yb →  -- A and B are on the circle
    Line m xa ya → Line m xb yb →  -- A and B are on the line
    PointCondition xa ya xb yb →   -- P(1,1) satisfies 2⃗AP = ⃗PB
    (m = 1 ∨ m = -1) :=             -- The slope of the line is either 1 or -1
by sorry

end NUMINAMATH_CALUDE_line_equation_l866_86639


namespace NUMINAMATH_CALUDE_marble_redistribution_l866_86686

/-- Given Tyrone's initial marbles -/
def tyrone_initial : ℕ := 150

/-- Given Eric's initial marbles -/
def eric_initial : ℕ := 30

/-- The number of marbles Tyrone gives to Eric -/
def marbles_given : ℕ := 15

theorem marble_redistribution :
  (tyrone_initial - marbles_given = 3 * (eric_initial + marbles_given)) ∧
  (0 < marbles_given) ∧ (marbles_given < tyrone_initial) := by
  sorry

end NUMINAMATH_CALUDE_marble_redistribution_l866_86686


namespace NUMINAMATH_CALUDE_exists_empty_subsquare_l866_86677

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- Checks if a point is inside a square -/
def isInside (p : Point) (s : Square) : Prop :=
  s.bottomLeft.x ≤ p.x ∧ p.x < s.bottomLeft.x + s.sideLength ∧
  s.bottomLeft.y ≤ p.y ∧ p.y < s.bottomLeft.y + s.sideLength

theorem exists_empty_subsquare (bigSquare : Square) (points : Finset Point) :
  bigSquare.sideLength = 4 →
  points.card = 15 →
  (∀ p ∈ points, isInside p bigSquare) →
  ∃ (smallSquare : Square),
    smallSquare.sideLength = 1 ∧
    isInside smallSquare.bottomLeft bigSquare ∧
    (∀ p ∈ points, ¬isInside p smallSquare) :=
by sorry

end NUMINAMATH_CALUDE_exists_empty_subsquare_l866_86677


namespace NUMINAMATH_CALUDE_logarithm_identity_l866_86671

theorem logarithm_identity (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  Real.log x / Real.log (a * b) = (Real.log x / Real.log a * Real.log x / Real.log b) /
    (Real.log x / Real.log a + Real.log x / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_identity_l866_86671


namespace NUMINAMATH_CALUDE_campaign_fund_family_contribution_percentage_l866_86688

/-- Calculates the percentage of family contribution in a campaign fund scenario -/
theorem campaign_fund_family_contribution_percentage 
  (total_funds : ℝ) 
  (friends_percentage : ℝ) 
  (president_savings : ℝ) : 
  total_funds = 10000 →
  friends_percentage = 40 →
  president_savings = 4200 →
  let friends_contribution := (friends_percentage / 100) * total_funds
  let remaining_after_friends := total_funds - friends_contribution
  let family_contribution := remaining_after_friends - president_savings
  (family_contribution / remaining_after_friends) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_campaign_fund_family_contribution_percentage_l866_86688


namespace NUMINAMATH_CALUDE_monomial_sum_implies_mn_value_l866_86690

theorem monomial_sum_implies_mn_value 
  (m n : ℤ) 
  (h : ∃ (a : ℚ), 3 * X^(m+6) * Y^(2*n+1) + X * Y^7 = a * X^(m+6) * Y^(2*n+1)) : 
  m * n = -15 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_mn_value_l866_86690


namespace NUMINAMATH_CALUDE_smallest_odd_integer_triangle_perimeter_l866_86619

/-- A function that checks if three numbers form a valid triangle --/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that generates three consecutive odd integers --/
def consecutive_odd_integers (n : ℕ) : (ℕ × ℕ × ℕ) :=
  (2*n + 1, 2*n + 3, 2*n + 5)

/-- The theorem stating the smallest possible perimeter of a triangle with consecutive odd integer side lengths --/
theorem smallest_odd_integer_triangle_perimeter :
  ∃ (n : ℕ), 
    let (a, b, c) := consecutive_odd_integers n
    is_valid_triangle a b c ∧
    ∀ (m : ℕ), m < n → ¬(is_valid_triangle (2*m + 1) (2*m + 3) (2*m + 5)) ∧
    a + b + c = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_integer_triangle_perimeter_l866_86619


namespace NUMINAMATH_CALUDE_john_payment_john_payment_is_8400_l866_86676

/-- Calculates John's payment for lawyer fees --/
theorem john_payment (upfront_fee : ℕ) (hourly_rate : ℕ) (court_hours : ℕ) 
  (prep_time_multiplier : ℕ) (paperwork_fee : ℕ) (transport_costs : ℕ) : ℕ :=
  let total_hours := court_hours + prep_time_multiplier * court_hours
  let total_fee := upfront_fee + hourly_rate * total_hours + paperwork_fee + transport_costs
  total_fee / 2

/-- Proves that John's payment is $8400 given the specified conditions --/
theorem john_payment_is_8400 : 
  john_payment 1000 100 50 2 500 300 = 8400 := by
  sorry

end NUMINAMATH_CALUDE_john_payment_john_payment_is_8400_l866_86676


namespace NUMINAMATH_CALUDE_multiply_decimals_l866_86630

theorem multiply_decimals : (0.25 : ℝ) * 0.08 = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l866_86630


namespace NUMINAMATH_CALUDE_intersection_points_sum_greater_than_two_l866_86612

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + (2*a - 1) * x

theorem intersection_points_sum_greater_than_two (a t x₁ x₂ : ℝ) 
  (ha : a ≤ 0) (ht : -1 < t ∧ t < 0) (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hf₁ : f a x₁ = t) (hf₂ : f a x₂ = t) : 
  x₁ + x₂ > 2 := by sorry

end NUMINAMATH_CALUDE_intersection_points_sum_greater_than_two_l866_86612


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_c_eq_five_halves_l866_86628

theorem infinite_solutions_iff_c_eq_five_halves (c : ℚ) :
  (∀ y : ℚ, 3 * (5 + 2 * c * y) = 15 * y + 15) ↔ c = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_c_eq_five_halves_l866_86628


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l866_86663

theorem contrapositive_equivalence (a : ℝ) :
  (¬(a > 1) → ¬(a > 0)) ↔ (a ≤ 1 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l866_86663


namespace NUMINAMATH_CALUDE_paolo_sevilla_birthday_friends_l866_86673

theorem paolo_sevilla_birthday_friends :
  ∀ (n : ℕ) (total_bill : ℝ),
    (total_bill / (n + 2 : ℝ) = 12) →
    (total_bill / n = 16) →
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_paolo_sevilla_birthday_friends_l866_86673


namespace NUMINAMATH_CALUDE_system_solution_l866_86665

theorem system_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + y + x*y = 8)
  (eq2 : y + z + y*z = 15)
  (eq3 : z + x + z*x = 35) :
  x + y + z + x*y = 15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l866_86665


namespace NUMINAMATH_CALUDE_three_to_six_minus_one_prime_factors_l866_86685

theorem three_to_six_minus_one_prime_factors :
  let n := 3^6 - 1
  ∃ (p q r : Nat), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    n % p = 0 ∧ n % q = 0 ∧ n % r = 0 ∧
    (∀ (s : Nat), Nat.Prime s → n % s = 0 → s = p ∨ s = q ∨ s = r) ∧
    p + q + r = 22 :=
by sorry

end NUMINAMATH_CALUDE_three_to_six_minus_one_prime_factors_l866_86685


namespace NUMINAMATH_CALUDE_pencils_per_child_l866_86680

theorem pencils_per_child (num_children : ℕ) (total_pencils : ℕ) 
  (h1 : num_children = 11) 
  (h2 : total_pencils = 22) : 
  total_pencils / num_children = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_child_l866_86680


namespace NUMINAMATH_CALUDE_vector_dot_product_equation_l866_86632

/-- Given vectors a, b, c, and a dot product equation, prove that x = 1 -/
theorem vector_dot_product_equation (a b c : ℝ × ℝ) (x : ℝ) :
  a = (1, 1) →
  b = (-1, 3) →
  c = (2, x) →
  (3 • a + b) • c = 10 →
  x = 1 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_equation_l866_86632


namespace NUMINAMATH_CALUDE_simplify_expression_l866_86623

theorem simplify_expression (a b : ℝ) : a - 4*(2*a - b) - 2*(a + 2*b) = -9*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l866_86623


namespace NUMINAMATH_CALUDE_trapezoid_y_property_l866_86607

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  -- Length of the shorter base
  c : ℝ
  -- Height of the trapezoid
  k : ℝ
  -- The segment joining midpoints divides the trapezoid into regions with area ratio 3:4
  midpoint_ratio : (c + 75) / (c + 150) = 3 / 4
  -- Length of the segment that divides the trapezoid into two equal areas
  y : ℝ
  -- The segment y divides the trapezoid into two equal areas
  equal_areas : y^2 = 65250

/-- The main theorem stating the property of y -/
theorem trapezoid_y_property (t : Trapezoid) : ⌊t.y^2 / 150⌋ = 435 := by
  sorry

#check trapezoid_y_property

end NUMINAMATH_CALUDE_trapezoid_y_property_l866_86607


namespace NUMINAMATH_CALUDE_f_f_eq_f_solutions_l866_86681

def f (x : ℝ) := x^2 - 2*x

theorem f_f_eq_f_solutions :
  {x : ℝ | f (f x) = f x} = {0, 2, -1, 3} := by sorry

end NUMINAMATH_CALUDE_f_f_eq_f_solutions_l866_86681


namespace NUMINAMATH_CALUDE_a₃_value_l866_86609

/-- The function f(x) = x^6 -/
def f (x : ℝ) : ℝ := x^6

/-- The expansion of f(x) in terms of (1+x) -/
def f_expansion (x a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : ℝ := 
  a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5 + a₆*(1+x)^6

/-- Theorem: If f(x) = x^6 can be expressed as the expansion, then a₃ = -20 -/
theorem a₃_value (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, f x = f_expansion x a₀ a₁ a₂ a₃ a₄ a₅ a₆) → a₃ = -20 := by
  sorry

end NUMINAMATH_CALUDE_a₃_value_l866_86609


namespace NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_unique_digits_mod_1000_l866_86633

/-- A function that checks if a natural number has all unique digits -/
def hasUniqueDigits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 9 with all unique digits -/
def M : ℕ := sorry

theorem greatest_multiple_of_nine_with_unique_digits_mod_1000 :
  M % 1000 = 981 := by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_unique_digits_mod_1000_l866_86633


namespace NUMINAMATH_CALUDE_triangle_side_difference_l866_86622

theorem triangle_side_difference (x : ℤ) : 
  (∀ y : ℤ, 3 ≤ y ∧ y ≤ 17 → (y + 8 > 10 ∧ y + 10 > 8 ∧ 8 + 10 > y)) →
  (∀ z : ℤ, z < 3 ∨ z > 17 → ¬(z + 8 > 10 ∧ z + 10 > 8 ∧ 8 + 10 > z)) →
  (17 - 3 : ℤ) = 14 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l866_86622


namespace NUMINAMATH_CALUDE_probability_three_one_is_five_ninths_l866_86626

def total_balls : ℕ := 18
def blue_balls : ℕ := 10
def red_balls : ℕ := 8
def drawn_balls : ℕ := 4

def probability_three_one : ℚ :=
  let favorable_outcomes := Nat.choose blue_balls 3 * Nat.choose red_balls 1 +
                            Nat.choose blue_balls 1 * Nat.choose red_balls 3
  let total_outcomes := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_three_one_is_five_ninths :
  probability_three_one = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_one_is_five_ninths_l866_86626


namespace NUMINAMATH_CALUDE_yearly_pet_feeding_cost_l866_86602

-- Define the number of each type of pet
def num_geckos : ℕ := 3
def num_iguanas : ℕ := 2
def num_snakes : ℕ := 4

-- Define the monthly feeding cost for each type of pet
def gecko_cost : ℕ := 15
def iguana_cost : ℕ := 5
def snake_cost : ℕ := 10

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Theorem statement
theorem yearly_pet_feeding_cost :
  (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * snake_cost) * months_per_year = 1140 :=
by sorry

end NUMINAMATH_CALUDE_yearly_pet_feeding_cost_l866_86602


namespace NUMINAMATH_CALUDE_b_speed_is_20_l866_86613

/-- The speed of person A in km/h -/
def speed_a : ℝ := 10

/-- The head start time of person A in hours -/
def head_start : ℝ := 5

/-- The total distance traveled when B catches up with A in km -/
def total_distance : ℝ := 100

/-- The speed of person B in km/h -/
def speed_b : ℝ := 20

theorem b_speed_is_20 :
  speed_b = (total_distance - speed_a * head_start) / (total_distance / speed_a - head_start) :=
by sorry

end NUMINAMATH_CALUDE_b_speed_is_20_l866_86613


namespace NUMINAMATH_CALUDE_hyperbola_equation_l866_86682

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  h_focus : ∃ c : ℝ, c = 5 -- Right focus coincides with focus of y^2 = 20x
  h_asymptote : ∀ x y : ℝ, y = 4/3 * x ∨ y = -4/3 * x

/-- The theorem stating that the hyperbola with given properties has the equation x^2/9 - y^2/16 = 1 -/
theorem hyperbola_equation (C : Hyperbola) : C.a^2 = 9 ∧ C.b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l866_86682


namespace NUMINAMATH_CALUDE_fishing_and_camping_l866_86679

/-- Represents the fishing and camping problem -/
theorem fishing_and_camping
  (total_fish_weight : ℝ)
  (wastage_percentage : ℝ)
  (adult_consumption : ℝ)
  (child_consumption : ℝ)
  (adult_child_ratio : ℚ)
  (max_campers : ℕ)
  (h1 : total_fish_weight = 44)
  (h2 : wastage_percentage = 0.2)
  (h3 : adult_consumption = 3)
  (h4 : child_consumption = 1)
  (h5 : adult_child_ratio = 2 / 5)
  (h6 : max_campers = 12) :
  ∃ (adult_campers child_campers : ℕ),
    adult_campers = 2 ∧
    child_campers = 5 ∧
    adult_campers + child_campers ≤ max_campers ∧
    (adult_campers : ℚ) / (child_campers : ℚ) = adult_child_ratio ∧
    (adult_campers : ℝ) * adult_consumption + (child_campers : ℝ) * child_consumption ≤
      total_fish_weight * (1 - wastage_percentage) :=
by sorry

end NUMINAMATH_CALUDE_fishing_and_camping_l866_86679


namespace NUMINAMATH_CALUDE_volume_sin_squared_rotation_l866_86652

theorem volume_sin_squared_rotation (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x ^ 2) :
  ∫ x in (0)..(Real.pi / 2), π * (f x)^2 = (3 * Real.pi^2) / 16 := by
  sorry

end NUMINAMATH_CALUDE_volume_sin_squared_rotation_l866_86652


namespace NUMINAMATH_CALUDE_parallel_vectors_a_value_l866_86664

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_a_value :
  let m : ℝ × ℝ := (2, 1)
  let n : ℝ × ℝ := (4, a)
  ∀ a : ℝ, are_parallel m n → a = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_a_value_l866_86664


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l866_86656

def M : Set ℝ := {x | x + 2 ≥ 0}
def N : Set ℝ := {x | x - 1 < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l866_86656


namespace NUMINAMATH_CALUDE_revenue_change_l866_86608

theorem revenue_change 
  (P : ℝ) 
  (N : ℝ) 
  (price_decrease : ℝ) 
  (sales_increase : ℝ) 
  (h1 : price_decrease = 0.2) 
  (h2 : sales_increase = 0.6) 
  : (1 - price_decrease) * (1 + sales_increase) * (P * N) = 1.28 * (P * N) := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l866_86608


namespace NUMINAMATH_CALUDE_fixed_point_exponential_l866_86603

/-- The function f(x) = a^(x+1) - 2 passes through the point (-1, -1) for all a > 0 and a ≠ 1 -/
theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) - 2
  f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_l866_86603


namespace NUMINAMATH_CALUDE_total_cars_in_group_l866_86644

/-- Given a group of cars with specific properties, we prove that the total number of cars is 137. -/
theorem total_cars_in_group (total : ℕ) 
  (no_ac : ℕ) 
  (with_stripes : ℕ) 
  (ac_no_stripes : ℕ) 
  (h1 : no_ac = 37)
  (h2 : with_stripes ≥ 51)
  (h3 : ac_no_stripes = 49)
  (h4 : total = no_ac + with_stripes + ac_no_stripes) :
  total = 137 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_in_group_l866_86644


namespace NUMINAMATH_CALUDE_smallest_n_square_cube_l866_86691

theorem smallest_n_square_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 3 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 3 * x = y^2) → (∃ (z : ℕ), 5 * x = z^3) → x ≥ n) ∧
  n = 675 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_cube_l866_86691


namespace NUMINAMATH_CALUDE_square_sum_plus_sum_squares_l866_86600

theorem square_sum_plus_sum_squares : (6 + 10)^2 + (6^2 + 10^2) = 392 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_plus_sum_squares_l866_86600


namespace NUMINAMATH_CALUDE_jimmy_garden_servings_l866_86689

/-- Represents the number of plants in each plot -/
def plants_per_plot : ℕ := 9

/-- Represents the number of servings produced by each carrot plant -/
def carrot_servings : ℕ := 4

/-- Represents the number of servings produced by each corn plant -/
def corn_servings : ℕ := 5 * carrot_servings

/-- Represents the number of servings produced by each green bean plant -/
def green_bean_servings : ℕ := corn_servings / 2

/-- Calculates the total number of servings from all three plots -/
def total_servings : ℕ := 
  plants_per_plot * carrot_servings +
  plants_per_plot * corn_servings +
  plants_per_plot * green_bean_servings

/-- Theorem stating that the total number of servings is 306 -/
theorem jimmy_garden_servings : total_servings = 306 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_garden_servings_l866_86689


namespace NUMINAMATH_CALUDE_wipes_used_correct_l866_86668

/-- Calculates the number of wipes used before refilling -/
def wipes_used (initial : ℕ) (refill : ℕ) (final : ℕ) : ℕ :=
  initial + refill - final

theorem wipes_used_correct (initial refill final : ℕ) 
  (h_initial : initial = 70)
  (h_refill : refill = 10)
  (h_final : final = 60) :
  wipes_used initial refill final = 20 := by
  sorry

#eval wipes_used 70 10 60

end NUMINAMATH_CALUDE_wipes_used_correct_l866_86668


namespace NUMINAMATH_CALUDE_min_value_implies_a_l866_86659

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |2*x + a|

/-- The theorem stating the relationship between the minimum value of f and the value of a -/
theorem min_value_implies_a (a : ℝ) : (∀ x : ℝ, f a x ≥ 3) ∧ (∃ x : ℝ, f a x = 3) → a = -4 ∨ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l866_86659


namespace NUMINAMATH_CALUDE_binary_representation_of_106_l866_86666

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else go (m / 2) ((m % 2 = 1) :: acc)
    go n []

/-- Converts a list of bits to a string representation of a binary number -/
def binaryToString (bits : List Bool) : String :=
  bits.map (fun b => if b then '1' else '0') |> String.mk

theorem binary_representation_of_106 :
  binaryToString (toBinary 106) = "1101010" := by
  sorry

#eval binaryToString (toBinary 106)

end NUMINAMATH_CALUDE_binary_representation_of_106_l866_86666


namespace NUMINAMATH_CALUDE_no_common_root_for_quadratics_l866_86631

/-- Two quadratic polynomials with coefficients satisfying certain inequalities cannot have a common root -/
theorem no_common_root_for_quadratics (k m n l : ℝ) 
  (h1 : k > m) (h2 : m > n) (h3 : n > l) (h4 : l > 0) :
  ¬∃ x : ℝ, x^2 + m*x + n = 0 ∧ x^2 + k*x + l = 0 := by
  sorry


end NUMINAMATH_CALUDE_no_common_root_for_quadratics_l866_86631


namespace NUMINAMATH_CALUDE_marble_box_capacity_l866_86698

theorem marble_box_capacity (jack_capacity : ℕ) (lucy_scale : ℕ) : 
  jack_capacity = 50 → lucy_scale = 3 → 
  (lucy_scale ^ 3) * jack_capacity = 1350 := by
  sorry

end NUMINAMATH_CALUDE_marble_box_capacity_l866_86698


namespace NUMINAMATH_CALUDE_line_characteristics_l866_86648

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The line y = -x - 3 -/
def line : Line := { slope := -1, y_intercept := -3 }

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on the line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

/-- Check if the line passes through a quadrant -/
def Line.passes_through_quadrant (l : Line) (q : ℕ) : Prop :=
  ∃ (p : Point), l.contains p ∧
    match q with
    | 1 => p.x > 0 ∧ p.y > 0
    | 2 => p.x < 0 ∧ p.y > 0
    | 3 => p.x < 0 ∧ p.y < 0
    | 4 => p.x > 0 ∧ p.y < 0
    | _ => False

theorem line_characteristics :
  (line.passes_through_quadrant 2 ∧
   line.passes_through_quadrant 3 ∧
   line.passes_through_quadrant 4) ∧
  line.slope < 0 ∧
  line.contains { x := 0, y := -3 } ∧
  ¬ line.contains { x := 3, y := 0 } := by sorry

end NUMINAMATH_CALUDE_line_characteristics_l866_86648


namespace NUMINAMATH_CALUDE_sqrt_difference_simplification_l866_86674

theorem sqrt_difference_simplification :
  3 * Real.sqrt 2 - |Real.sqrt 2 - Real.sqrt 3| = 4 * Real.sqrt 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_simplification_l866_86674


namespace NUMINAMATH_CALUDE_probability_at_least_three_same_l866_86611

def num_dice : ℕ := 5
def num_sides : ℕ := 8

def total_outcomes : ℕ := num_sides ^ num_dice

def favorable_outcomes : ℕ :=
  -- Exactly 3 dice showing the same number
  (num_sides * (num_dice.choose 3) * (num_sides - 1)^2) +
  -- Exactly 4 dice showing the same number
  (num_sides * (num_dice.choose 4) * (num_sides - 1)) +
  -- All 5 dice showing the same number
  num_sides

theorem probability_at_least_three_same (h : favorable_outcomes = 4208) :
  (favorable_outcomes : ℚ) / total_outcomes = 1052 / 8192 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_three_same_l866_86611


namespace NUMINAMATH_CALUDE_lotus_flower_problem_l866_86657

theorem lotus_flower_problem (x : ℚ) : 
  (x / 3 + x / 5 + x / 6 + x / 4 + 6 = x) → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_lotus_flower_problem_l866_86657


namespace NUMINAMATH_CALUDE_jerrys_average_increase_l866_86661

theorem jerrys_average_increase :
  ∀ (initial_average : ℝ) (fourth_test_score : ℝ),
    initial_average = 90 →
    fourth_test_score = 98 →
    (3 * initial_average + fourth_test_score) / 4 = initial_average + 2 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_average_increase_l866_86661


namespace NUMINAMATH_CALUDE_no_solution_factorial_equation_l866_86647

theorem no_solution_factorial_equation :
  ∀ (k m : ℕ+), k.val.factorial + 48 ≠ 48 * (k.val + 1) ^ m.val := by
  sorry

end NUMINAMATH_CALUDE_no_solution_factorial_equation_l866_86647


namespace NUMINAMATH_CALUDE_function_inequality_l866_86653

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

theorem function_inequality (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 →
    |f a x₁ - f a x₂| ≤ a - 1) →
  a ≥ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l866_86653


namespace NUMINAMATH_CALUDE_percent_calculation_l866_86617

theorem percent_calculation (x y : ℝ) (h : x = 120.5 ∧ y = 80.75) :
  (x / y) * 100 = 149.26 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l866_86617


namespace NUMINAMATH_CALUDE_problem_solution_l866_86649

theorem problem_solution (x y : ℝ) 
  (eq1 : |x| + x + y = 15)
  (eq2 : x + |y| - y = 9)
  (eq3 : y = 3*x - 7) : 
  x + y = 53/5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l866_86649


namespace NUMINAMATH_CALUDE_fraction_sum_l866_86650

theorem fraction_sum : (2 : ℚ) / 3 + 5 / 18 - 1 / 6 = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l866_86650


namespace NUMINAMATH_CALUDE_exponent_product_simplification_l866_86692

theorem exponent_product_simplification :
  (10 ^ 0.5) * (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.05) * (10 ^ 1.05) = 100 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_simplification_l866_86692


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l866_86614

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem sum_of_coefficients :
  (∀ x, f (x + 2) = 2 * x^3 + 5 * x^2 + 3 * x + 6) →
  (∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  (∃ a b c d : ℝ, (∀ x, f x = a * x^3 + b * x^2 + c * x + d) ∧ a + b + c + d = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l866_86614


namespace NUMINAMATH_CALUDE_max_F_value_l866_86693

def is_eternal_number (M : ℕ) : Prop :=
  M ≥ 1000 ∧ M < 10000 ∧
  (M / 100 % 10 + M / 10 % 10 + M % 10 = 12)

def N (M : ℕ) : ℕ :=
  (M / 1000) * 100 + (M / 100 % 10) * 1000 + (M / 10 % 10) + (M % 10) * 10

def F (M : ℕ) : ℚ :=
  (M - N M) / 9

theorem max_F_value (M : ℕ) :
  is_eternal_number M →
  (M / 100 % 10 - M % 10 = M / 1000) →
  (F M / 9).isInt →
  F M ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_F_value_l866_86693


namespace NUMINAMATH_CALUDE_polynomial_root_coefficients_l866_86678

theorem polynomial_root_coefficients :
  ∀ (a b c : ℝ),
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - Complex.I : ℂ) ^ 4 + a * (2 - Complex.I : ℂ) ^ 3 + b * (2 - Complex.I : ℂ) ^ 2 - 2 * (2 - Complex.I : ℂ) + c = 0 →
  a = 2 + 2 * Real.sqrt 1.5 ∧
  b = 10 + 2 * Real.sqrt 1.5 ∧
  c = 10 - 8 * Real.sqrt 1.5 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_coefficients_l866_86678


namespace NUMINAMATH_CALUDE_remainder_equality_l866_86646

theorem remainder_equality (P P' D R R' r r' : ℕ) 
  (h1 : P > P') 
  (h2 : R = P % D) 
  (h3 : R' = P' % D) 
  (h4 : r = (P * P') % D) 
  (h5 : r' = (R * R') % D) : 
  r = r' := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l866_86646


namespace NUMINAMATH_CALUDE_problem_solution_l866_86620

open Set

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x ≤ a + 6}

theorem problem_solution (a : ℝ) (h : A ⊆ M a) :
  ((𝒰 \ B) ∩ A = {x | -3 ≤ x ∧ x ≤ 2}) ∧ (a ≥ -3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l866_86620


namespace NUMINAMATH_CALUDE_smallest_seven_digit_binary_l866_86683

theorem smallest_seven_digit_binary : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m.digits 2 = [1, 0, 0, 0, 0, 0, 0] → m ≥ n) ∧
  n.digits 2 = [1, 0, 0, 0, 0, 0, 0] ∧
  n = 64 := by
  sorry

end NUMINAMATH_CALUDE_smallest_seven_digit_binary_l866_86683


namespace NUMINAMATH_CALUDE_units_digit_of_7_62_l866_86645

theorem units_digit_of_7_62 : ∃ n : ℕ, 7^62 ≡ 9 [MOD 10] :=
by
  -- We'll use n = 9 to prove the existence
  use 9
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_62_l866_86645


namespace NUMINAMATH_CALUDE_average_books_borrowed_l866_86624

theorem average_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ)
  (h1 : total_students = 40)
  (h2 : zero_books = 2)
  (h3 : one_book = 12)
  (h4 : two_books = 12)
  (h5 : zero_books + one_book + two_books < total_students) :
  let remaining_students := total_students - (zero_books + one_book + two_books)
  let total_books := one_book * 1 + two_books * 2 + remaining_students * 3
  (total_books : ℚ) / total_students = 39/20 := by
sorry

end NUMINAMATH_CALUDE_average_books_borrowed_l866_86624


namespace NUMINAMATH_CALUDE_count_distinct_keys_l866_86695

/-- Represents a rotational stencil cipher key of size n × n -/
structure StencilKey (n : ℕ) where
  size : n % 2 = 0  -- n is even

/-- The number of distinct rotational stencil cipher keys for a given even size n -/
def num_distinct_keys (n : ℕ) : ℕ := 4^(n^2/4)

/-- Theorem stating the number of distinct rotational stencil cipher keys -/
theorem count_distinct_keys (n : ℕ) (key : StencilKey n) :
  num_distinct_keys n = 4^(n^2/4) := by
  sorry

#check count_distinct_keys

end NUMINAMATH_CALUDE_count_distinct_keys_l866_86695


namespace NUMINAMATH_CALUDE_f_symmetric_about_origin_l866_86638

/-- The function f(x) = 2sin(x)cos(x) is symmetric about the origin -/
theorem f_symmetric_about_origin :
  ∀ x : ℝ, (2 * Real.sin x * Real.cos x) = -(2 * Real.sin (-x) * Real.cos (-x)) := by
  sorry

end NUMINAMATH_CALUDE_f_symmetric_about_origin_l866_86638


namespace NUMINAMATH_CALUDE_hundred_chickens_problem_l866_86640

theorem hundred_chickens_problem :
  ∀ x y z : ℕ,
  x + y + z = 100 →
  5 * x + 3 * y + (z / 3 : ℚ) = 100 →
  z = 81 →
  x = 8 ∧ y = 11 := by
sorry

end NUMINAMATH_CALUDE_hundred_chickens_problem_l866_86640


namespace NUMINAMATH_CALUDE_prime_odd_sum_product_l866_86610

theorem prime_odd_sum_product (p q : ℕ) : 
  Prime p → 
  Odd q → 
  q > 0 → 
  p^2 + q = 125 → 
  p * q = 242 := by
sorry

end NUMINAMATH_CALUDE_prime_odd_sum_product_l866_86610


namespace NUMINAMATH_CALUDE_inequality_of_four_variables_l866_86621

theorem inequality_of_four_variables (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_four_variables_l866_86621


namespace NUMINAMATH_CALUDE_parabola_vertex_l866_86616

/-- The vertex of a parabola given by y^2 - 4y + 2x + 7 = 0 is (-3/2, 2) -/
theorem parabola_vertex :
  let f : ℝ → ℝ → ℝ := λ x y => y^2 - 4*y + 2*x + 7
  ∃! (vx vy : ℝ), (∀ x y, f x y = 0 → (x - vx)^2 ≤ (x + 3/2)^2 ∧ y = vy) ∧ vx = -3/2 ∧ vy = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l866_86616


namespace NUMINAMATH_CALUDE_simons_flower_purchase_l866_86618

def flower_purchase (pansy_price petunia_price hydrangea_price : ℝ)
                    (pansy_count petunia_count : ℕ)
                    (discount_rate : ℝ)
                    (change_received : ℝ) : Prop :=
  let total_before_discount := pansy_price * (pansy_count : ℝ) +
                               petunia_price * (petunia_count : ℝ) +
                               hydrangea_price
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let amount_paid := total_after_discount + change_received
  amount_paid = 50

theorem simons_flower_purchase :
  flower_purchase 2.5 1 12.5 5 5 0.1 23 := by
  sorry

end NUMINAMATH_CALUDE_simons_flower_purchase_l866_86618


namespace NUMINAMATH_CALUDE_floor_sum_equality_implies_integer_difference_l866_86655

theorem floor_sum_equality_implies_integer_difference (a b c d : ℝ) 
  (h : ∀ (n : ℕ+), ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) : 
  (∃ (z : ℤ), a + b = z) ∨ (∃ (z : ℤ), a - c = z) ∨ (∃ (z : ℤ), a - d = z) := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_equality_implies_integer_difference_l866_86655


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l866_86625

theorem power_zero_eq_one (x : ℝ) : x ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l866_86625


namespace NUMINAMATH_CALUDE_product_of_numbers_l866_86642

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l866_86642


namespace NUMINAMATH_CALUDE_math_problems_sum_l866_86669

/-- The sum of four math problem answers given specific conditions -/
theorem math_problems_sum : 
  let answer1 : ℝ := 600
  let answer2 : ℝ := 2 * answer1
  let answer3 : ℝ := answer1 + answer2 - 400
  let answer4 : ℝ := (answer1 + answer2 + answer3) / 3
  (answer1 + answer2 + answer3 + answer4) = 4266.67 := by
  sorry

end NUMINAMATH_CALUDE_math_problems_sum_l866_86669


namespace NUMINAMATH_CALUDE_same_remainder_for_282_l866_86606

theorem same_remainder_for_282 : ∃ r : ℕ, r < 9 ∧ r < 31 ∧ 282 % 31 = r ∧ 282 % 9 = r ∧ r = 3 := by
  sorry

end NUMINAMATH_CALUDE_same_remainder_for_282_l866_86606


namespace NUMINAMATH_CALUDE_total_toys_count_l866_86696

def bill_toys : ℕ := 60

def hana_toys : ℕ := (5 * bill_toys) / 6

def hash_toys : ℕ := hana_toys / 2 + 9

def total_toys : ℕ := bill_toys + hana_toys + hash_toys

theorem total_toys_count : total_toys = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_count_l866_86696


namespace NUMINAMATH_CALUDE_exists_irrational_less_than_neg_two_l866_86684

theorem exists_irrational_less_than_neg_two : ∃ x : ℝ, Irrational x ∧ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_exists_irrational_less_than_neg_two_l866_86684


namespace NUMINAMATH_CALUDE_smallest_positive_period_monotonically_increasing_intervals_l866_86654

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x * Real.cos x - 5 * Real.sqrt 3 * (Real.cos x)^2 + 5 * Real.sqrt 3 / 2

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧ 
  T = Real.pi :=
sorry

-- Theorem for monotonically increasing intervals
theorem monotonically_increasing_intervals :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (- Real.pi / 12 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi)) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_monotonically_increasing_intervals_l866_86654


namespace NUMINAMATH_CALUDE_total_marbles_l866_86699

theorem total_marbles (jungkook_marbles : ℕ) (jimin_extra_marbles : ℕ) : 
  jungkook_marbles = 3 → 
  jimin_extra_marbles = 4 → 
  jungkook_marbles + (jungkook_marbles + jimin_extra_marbles) = 10 := by
sorry

end NUMINAMATH_CALUDE_total_marbles_l866_86699


namespace NUMINAMATH_CALUDE_dimes_in_shorts_l866_86675

/-- Given a total amount of money and the number of dimes in a jacket, 
    calculate the number of dimes in the shorts. -/
theorem dimes_in_shorts 
  (total : ℚ) 
  (jacket_dimes : ℕ) 
  (dime_value : ℚ) 
  (h1 : total = 19/10) 
  (h2 : jacket_dimes = 15) 
  (h3 : dime_value = 1/10) : 
  ↑jacket_dimes * dime_value + 4 * dime_value = total :=
sorry

end NUMINAMATH_CALUDE_dimes_in_shorts_l866_86675


namespace NUMINAMATH_CALUDE_rectangle_area_l866_86634

/-- The area of a rectangular region bounded by y = a, y = a-2b, x = -2c, and x = d -/
theorem rectangle_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a - (a - 2 * b)) * (d - (-2 * c)) = 2 * b * d + 4 * b * c := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l866_86634


namespace NUMINAMATH_CALUDE_apples_bought_is_three_l866_86637

/-- Calculates the number of apples bought given the total cost, number of oranges,
    price difference between oranges and apples, and the cost of each fruit. -/
def apples_bought (total_cost orange_count price_diff fruit_cost : ℚ) : ℚ :=
  (total_cost - orange_count * (fruit_cost + price_diff)) / fruit_cost

/-- Theorem stating that under the given conditions, the number of apples bought is 3. -/
theorem apples_bought_is_three :
  let total_cost : ℚ := 456/100
  let orange_count : ℚ := 7
  let price_diff : ℚ := 28/100
  let fruit_cost : ℚ := 26/100
  apples_bought total_cost orange_count price_diff fruit_cost = 3 := by
  sorry

#eval apples_bought (456/100) 7 (28/100) (26/100)

end NUMINAMATH_CALUDE_apples_bought_is_three_l866_86637


namespace NUMINAMATH_CALUDE_expression_simplification_l866_86672

theorem expression_simplification :
  ∀ x : ℝ, ((3*x^2 + 2*x - 1) + x^2*2)*4 + (5 - 2/2)*(3*x^2 + 6*x - 8) = 32*x^2 + 32*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l866_86672


namespace NUMINAMATH_CALUDE_negation_equivalence_l866_86658

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 1 > 0) := by
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l866_86658


namespace NUMINAMATH_CALUDE_vector_u_satisfies_equation_l866_86635

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 0, 2]

theorem vector_u_satisfies_equation :
  let u : Matrix (Fin 2) (Fin 1) ℝ := !![5/273; 8/21]
  (B^5 + B^3 + B) * u = !![5; 16] := by
  sorry

end NUMINAMATH_CALUDE_vector_u_satisfies_equation_l866_86635


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_450_l866_86697

theorem least_integer_greater_than_sqrt_450 : 
  (∀ n : ℤ, n ≤ ⌊Real.sqrt 450⌋ → n < 22) ∧ 22 > Real.sqrt 450 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_450_l866_86697


namespace NUMINAMATH_CALUDE_unique_valid_number_l866_86643

def is_valid_number (n : ℕ) : Prop :=
  n % 25 = 0 ∧ n % 35 = 0 ∧
  (∃ (a b c : ℕ), a * n ≤ 1050 ∧ b * n ≤ 1050 ∧ c * n ≤ 1050 ∧
   a < b ∧ b < c ∧
   ∀ (x : ℕ), x * n ≤ 1050 → x = a ∨ x = b ∨ x = c)

theorem unique_valid_number : 
  is_valid_number 350 ∧ ∀ (m : ℕ), is_valid_number m → m = 350 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l866_86643


namespace NUMINAMATH_CALUDE_estimation_correct_l866_86687

/-- Represents a school population --/
structure School where
  total_students : ℕ
  sample_size : ℕ
  sample_enthusiasts : ℕ

/-- Calculates the estimated number of enthusiasts in the entire school population --/
def estimate_enthusiasts (s : School) : ℕ :=
  (s.total_students * s.sample_enthusiasts) / s.sample_size

/-- Theorem stating that the estimation method in statement D is correct --/
theorem estimation_correct (s : School) 
  (h1 : s.total_students = 3200)
  (h2 : s.sample_size = 200)
  (h3 : s.sample_enthusiasts = 85) :
  estimate_enthusiasts s = 1360 := by
  sorry

#eval estimate_enthusiasts { total_students := 3200, sample_size := 200, sample_enthusiasts := 85 }

end NUMINAMATH_CALUDE_estimation_correct_l866_86687


namespace NUMINAMATH_CALUDE_jacks_remaining_money_l866_86604

/-- Calculates the remaining money after currency conversion, fees, and spending --/
def calculate_remaining_money (
  initial_dollars : ℝ)
  (initial_euros : ℝ)
  (initial_yen : ℝ)
  (initial_rubles : ℝ)
  (euro_to_dollar : ℝ)
  (yen_to_dollar : ℝ)
  (ruble_to_dollar : ℝ)
  (transaction_fee : ℝ)
  (spending_percentage : ℝ) : ℝ :=
  let converted_euros := initial_euros * euro_to_dollar
  let converted_yen := initial_yen * yen_to_dollar
  let converted_rubles := initial_rubles * ruble_to_dollar
  let total_before_fees := initial_dollars + converted_euros + converted_yen + converted_rubles
  let fees := (converted_euros + converted_yen + converted_rubles) * transaction_fee
  let total_after_fees := total_before_fees - fees
  let amount_spent := total_after_fees * spending_percentage
  total_after_fees - amount_spent

/-- Theorem stating that Jack's remaining money is approximately $132.85 --/
theorem jacks_remaining_money :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |calculate_remaining_money 45 36 1350 1500 2 0.009 0.013 0.01 0.1 - 132.85| < ε :=
sorry

end NUMINAMATH_CALUDE_jacks_remaining_money_l866_86604
