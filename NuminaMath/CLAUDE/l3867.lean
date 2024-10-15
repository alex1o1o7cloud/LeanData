import Mathlib

namespace NUMINAMATH_CALUDE_number_problem_l3867_386796

theorem number_problem : 
  ∃ x : ℚ, (30 / 100 : ℚ) * x = (25 / 100 : ℚ) * 40 ∧ x = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3867_386796


namespace NUMINAMATH_CALUDE_roots_in_arithmetic_progression_l3867_386710

theorem roots_in_arithmetic_progression (m : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ x : ℝ, x^4 - (3*m + 2)*x^2 + m^2 = 0 ↔ x = a ∨ x = b ∨ x = -b ∨ x = -a) ∧
    (b - (-b) = -b - (-a) ∧ a - b = b - (-b)))
  ↔ m = 6 ∨ m = -6/19 := by sorry

end NUMINAMATH_CALUDE_roots_in_arithmetic_progression_l3867_386710


namespace NUMINAMATH_CALUDE_dawsons_friends_l3867_386738

def total_cost : ℕ := 13500
def cost_per_person : ℕ := 900

theorem dawsons_friends :
  (total_cost / cost_per_person) - 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_dawsons_friends_l3867_386738


namespace NUMINAMATH_CALUDE_grace_garden_seeds_l3867_386712

/-- Represents the number of large beds in Grace's garden -/
def num_large_beds : Nat := 2

/-- Represents the number of medium beds in Grace's garden -/
def num_medium_beds : Nat := 2

/-- Represents the number of rows in a large bed -/
def rows_large_bed : Nat := 4

/-- Represents the number of rows in a medium bed -/
def rows_medium_bed : Nat := 3

/-- Represents the number of seeds per row in a large bed -/
def seeds_per_row_large : Nat := 25

/-- Represents the number of seeds per row in a medium bed -/
def seeds_per_row_medium : Nat := 20

/-- Calculates the total number of seeds Grace can plant in her raised bed garden -/
def total_seeds : Nat :=
  num_large_beds * rows_large_bed * seeds_per_row_large +
  num_medium_beds * rows_medium_bed * seeds_per_row_medium

/-- Proves that the total number of seeds Grace can plant is 320 -/
theorem grace_garden_seeds : total_seeds = 320 := by
  sorry

end NUMINAMATH_CALUDE_grace_garden_seeds_l3867_386712


namespace NUMINAMATH_CALUDE_gcd_property_l3867_386797

theorem gcd_property (a : ℕ) (h : ∀ n : ℤ, (Int.gcd (a * n + 1) (2 * n + 1) = 1)) :
  (∀ n : ℤ, Int.gcd (a - 2) (2 * n + 1) = 1) ∧
  (a = 1 ∨ ∃ m : ℕ, a = 2 + 2^m) := by
  sorry

end NUMINAMATH_CALUDE_gcd_property_l3867_386797


namespace NUMINAMATH_CALUDE_consecutive_integers_prime_divisor_ratio_l3867_386764

theorem consecutive_integers_prime_divisor_ratio :
  ∃ a : ℕ, ∀ i ∈ Finset.range 2009,
    let n := a + i + 1
    ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ∣ n ∧ q ∣ n ∧
      (∀ r : ℕ, Prime r → r ∣ n → p ≤ r) ∧
      (∀ r : ℕ, Prime r → r ∣ n → r ≤ q) ∧
      q > 20 * p :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_prime_divisor_ratio_l3867_386764


namespace NUMINAMATH_CALUDE_semicircle_overlap_width_l3867_386778

/-- Given a rectangle with two semicircles drawn inside, where each semicircle
    has a radius of 5 cm and the rectangle height is 8 cm, the width of the
    overlap between the semicircles is 6 cm. -/
theorem semicircle_overlap_width (r : ℝ) (h : ℝ) (w : ℝ) :
  r = 5 →
  h = 8 →
  w = 2 * Real.sqrt (r^2 - (h/2)^2) →
  w = 6 := by
  sorry

#check semicircle_overlap_width

end NUMINAMATH_CALUDE_semicircle_overlap_width_l3867_386778


namespace NUMINAMATH_CALUDE_square_side_length_l3867_386727

theorem square_side_length
  (a : ℝ) -- side length of the square
  (x : ℝ) -- one leg of the right triangle
  (b : ℝ) -- hypotenuse of the right triangle
  (h1 : 4 * a + 2 * x = 58) -- perimeter of rectangle
  (h2 : 2 * a + 2 * b + 2 * x = 60) -- perimeter of trapezoid
  (h3 : a^2 + x^2 = b^2) -- Pythagorean theorem
  : a = 12 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3867_386727


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l3867_386786

/-- Calculates the final amount after two years of compound interest with different rates each year. -/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, the final amount after two years is as calculated. -/
theorem compound_interest_calculation :
  final_amount 5460 0.04 0.05 = 5962.32 := by
  sorry

#eval final_amount 5460 0.04 0.05

end NUMINAMATH_CALUDE_compound_interest_calculation_l3867_386786


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3867_386700

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3867_386700


namespace NUMINAMATH_CALUDE_july_husband_age_l3867_386754

def hannah_age_then : ℕ := 6
def years_passed : ℕ := 20

theorem july_husband_age :
  ∀ (july_age_then : ℕ),
  hannah_age_then = 2 * july_age_then →
  (july_age_then + years_passed + 2 = 25) :=
by
  sorry

end NUMINAMATH_CALUDE_july_husband_age_l3867_386754


namespace NUMINAMATH_CALUDE_sum_squares_product_l3867_386783

theorem sum_squares_product (m n : ℝ) (h : m + n = -2) : 5*m^2 + 5*n^2 + 10*m*n = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_product_l3867_386783


namespace NUMINAMATH_CALUDE_permutations_without_patterns_l3867_386733

/-- The total number of permutations of 4 x's, 3 y's, and 2 z's -/
def total_permutations : ℕ := 1260

/-- The set of permutations where the pattern xxxx appears -/
def A₁ : Finset (List Char) := sorry

/-- The set of permutations where the pattern yyy appears -/
def A₂ : Finset (List Char) := sorry

/-- The set of permutations where the pattern zz appears -/
def A₃ : Finset (List Char) := sorry

/-- The theorem to be proved -/
theorem permutations_without_patterns (h₁ : Finset.card A₁ = 60) 
  (h₂ : Finset.card A₂ = 105) (h₃ : Finset.card A₃ = 280)
  (h₄ : Finset.card (A₁ ∩ A₂) = 12) (h₅ : Finset.card (A₁ ∩ A₃) = 20)
  (h₆ : Finset.card (A₂ ∩ A₃) = 30) (h₇ : Finset.card (A₁ ∩ A₂ ∩ A₃) = 6) :
  total_permutations - Finset.card (A₁ ∪ A₂ ∪ A₃) = 871 := by
  sorry

end NUMINAMATH_CALUDE_permutations_without_patterns_l3867_386733


namespace NUMINAMATH_CALUDE_solution_in_interval_l3867_386723

theorem solution_in_interval :
  ∃ x₀ : ℝ, (Real.log x₀ + x₀ = 4) ∧ (2 < x₀) ∧ (x₀ < 3) := by
  sorry

#check solution_in_interval

end NUMINAMATH_CALUDE_solution_in_interval_l3867_386723


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l3867_386719

/-- The probability of selecting at least one woman when choosing 4 people at random
    from a group of 8 men and 4 women is equal to 85/99. -/
theorem prob_at_least_one_woman (total : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total = men + women →
  men = 8 →
  women = 4 →
  selected = 4 →
  (1 : ℚ) - (men.choose selected : ℚ) / (total.choose selected : ℚ) = 85 / 99 := by
  sorry

#check prob_at_least_one_woman

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l3867_386719


namespace NUMINAMATH_CALUDE_log_xy_value_l3867_386758

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 3) :
  Real.log (x * y) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l3867_386758


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l3867_386736

theorem least_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), 3 * |y| - 2 * y + 8 < 23 → x ≤ y) ∧ (3 * |x| - 2 * x + 8 < 23) :=
by sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l3867_386736


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l3867_386734

theorem coefficient_x4_in_expansion : 
  (Finset.range 8).sum (fun k => Nat.choose 7 k * (1^(7-k) * x^(2*k))) = 
  21 * x^4 + (Finset.range 8).sum (fun k => if k ≠ 2 then Nat.choose 7 k * (1^(7-k) * x^(2*k)) else 0) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l3867_386734


namespace NUMINAMATH_CALUDE_otimes_self_otimes_self_l3867_386788

/-- Definition of the ⊗ operation -/
def otimes (x y : ℝ) : ℝ := x^3 + x^2 - y

/-- Theorem: For any real number a, a ⊗ (a ⊗ a) = a -/
theorem otimes_self_otimes_self (a : ℝ) : otimes a (otimes a a) = a := by
  sorry

end NUMINAMATH_CALUDE_otimes_self_otimes_self_l3867_386788


namespace NUMINAMATH_CALUDE_orthogonal_to_pencil_l3867_386702

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A pencil of circles -/
structure PencilOfCircles where
  circles : Set Circle

/-- Two circles are orthogonal if the square of the distance between their centers
    is equal to the sum of the squares of their radii -/
def orthogonal (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = c1.radius^2 + c2.radius^2

theorem orthogonal_to_pencil
  (S : Circle) (P : PencilOfCircles) (S1 S2 : Circle)
  (h1 : S1 ∈ P.circles) (h2 : S2 ∈ P.circles) (h3 : S1 ≠ S2)
  (orth1 : orthogonal S S1) (orth2 : orthogonal S S2) :
  ∀ C ∈ P.circles, orthogonal S C :=
sorry

end NUMINAMATH_CALUDE_orthogonal_to_pencil_l3867_386702


namespace NUMINAMATH_CALUDE_quadratic_root_and_m_l3867_386701

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 + 2*x + m = 0

-- Theorem statement
theorem quadratic_root_and_m :
  ∀ m : ℝ, quadratic_equation (-2) m → m = 0 ∧ quadratic_equation 0 m :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_and_m_l3867_386701


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_l3867_386750

/-- Pentagon formed by cutting a triangular corner from a rectangle -/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {14, 21, 22, 28, 35}

/-- The area of the pentagon -/
def pentagon_area (p : CornerCutPentagon) : ℕ :=
  1421

/-- Theorem stating that the area of the specified pentagon is 1421 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : 
  pentagon_area p = 1421 := by sorry

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_l3867_386750


namespace NUMINAMATH_CALUDE_min_value_expression_l3867_386784

theorem min_value_expression (a b m n : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : m > 0) (h4 : n > 0) 
  (h5 : a + b = 1) (h6 : m * n = 2) :
  (a * m + b * n) * (b * m + a * n) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3867_386784


namespace NUMINAMATH_CALUDE_dz_dt_formula_l3867_386728

noncomputable def z (t : ℝ) := Real.arcsin ((2*t)^2 + (4*t^2)^2 + t^2)

theorem dz_dt_formula (t : ℝ) :
  deriv z t = (2*t*(1 + 4*t + 32*t^2)) / Real.sqrt (1 - ((2*t)^2 + (4*t^2)^2 + t^2)^2) :=
sorry

end NUMINAMATH_CALUDE_dz_dt_formula_l3867_386728


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l3867_386748

def I : Finset Nat := {1,2,3,4,5,6,7,8}
def M : Finset Nat := {3,4,5}
def N : Finset Nat := {1,3,6}

theorem complement_intersection_equals_set : 
  (I \ M) ∩ (I \ N) = {2,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l3867_386748


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l3867_386706

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : Nat := 52

/-- Number of 5's in a standard deck -/
def NumberOfFives : Nat := 4

/-- Number of hearts in a standard deck -/
def NumberOfHearts : Nat := 13

/-- Number of Aces in a standard deck -/
def NumberOfAces : Nat := 4

/-- Probability of drawing a 5 as the first card, a heart as the second card, 
    and an Ace as the third card from a standard 52-card deck -/
def probabilityOfSpecificDraw : ℚ :=
  (NumberOfFives * NumberOfHearts * NumberOfAces) / 
  (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2))

theorem probability_of_specific_draw :
  probabilityOfSpecificDraw = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l3867_386706


namespace NUMINAMATH_CALUDE_cube_root_difference_l3867_386771

theorem cube_root_difference : (8 : ℝ) ^ (1/3) - (343 : ℝ) ^ (1/3) = -5 := by sorry

end NUMINAMATH_CALUDE_cube_root_difference_l3867_386771


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l3867_386714

theorem twenty_five_percent_less_than_80 (x : ℝ) : x + (1/4) * x = 60 → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l3867_386714


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3867_386735

/-- A quadratic function is always positive if and only if its coefficient of x^2 is positive and its discriminant is negative -/
theorem quadratic_always_positive (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (a > 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3867_386735


namespace NUMINAMATH_CALUDE_remaining_space_is_7200_mb_l3867_386763

/-- Conversion factor from GB to MB -/
def gb_to_mb : ℕ := 1024

/-- Total hard drive capacity in GB -/
def total_capacity_gb : ℕ := 300

/-- Used storage space in MB -/
def used_space_mb : ℕ := 300000

/-- Theorem: The remaining empty space on the hard drive is 7200 MB -/
theorem remaining_space_is_7200_mb :
  total_capacity_gb * gb_to_mb - used_space_mb = 7200 := by
  sorry

end NUMINAMATH_CALUDE_remaining_space_is_7200_mb_l3867_386763


namespace NUMINAMATH_CALUDE_a_in_range_l3867_386708

/-- Proposition p: for any real number x, ax^2 + ax + 1 > 0 always holds -/
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: the equation x^2 - x + a = 0 has real roots with respect to x -/
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- The set representing the range of a: (-∞, 0) ∪ (1/4, 4) -/
def range_a : Set ℝ := {a | a < 0 ∨ (1/4 < a ∧ a < 4)}

/-- Main theorem: If only one of prop_p and prop_q is true, then a is in range_a -/
theorem a_in_range (a : ℝ) : 
  (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a) → a ∈ range_a := by sorry

end NUMINAMATH_CALUDE_a_in_range_l3867_386708


namespace NUMINAMATH_CALUDE_unique_base_conversion_l3867_386746

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base b to base 10 -/
def baseBToBase10 (n : Nat) (b : Nat) : Nat :=
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem unique_base_conversion :
  ∃! (b : Nat), b > 0 ∧ base6ToBase10 125 = baseBToBase10 221 b :=
by sorry

end NUMINAMATH_CALUDE_unique_base_conversion_l3867_386746


namespace NUMINAMATH_CALUDE_line_through_point_l3867_386757

/-- A line contains a point if the point's coordinates satisfy the line's equation. -/
def line_contains_point (m : ℚ) (x y : ℚ) : Prop :=
  2 - m * x = -4 * y

/-- The theorem states that the line 2 - mx = -4y contains the point (5, -2) when m = -6/5. -/
theorem line_through_point (m : ℚ) :
  line_contains_point m 5 (-2) ↔ m = -6/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3867_386757


namespace NUMINAMATH_CALUDE_overall_loss_is_184_76_l3867_386709

-- Define the structure for an item
structure Item where
  name : String
  price : ℝ
  currency : String
  tax_rate : ℝ
  discount_rate : ℝ
  profit_loss_rate : ℝ

-- Define the currency conversion rates
def conversion_rates : List (String × ℝ) :=
  [("USD", 75), ("EUR", 80), ("GBP", 100), ("JPY", 0.7)]

-- Define the items
def items : List Item :=
  [{ name := "grinder", price := 150, currency := "USD", tax_rate := 0.1, discount_rate := 0, profit_loss_rate := -0.04 },
   { name := "mobile_phone", price := 100, currency := "EUR", tax_rate := 0.15, discount_rate := 0.05, profit_loss_rate := 0.1 },
   { name := "laptop", price := 200, currency := "GBP", tax_rate := 0.08, discount_rate := 0, profit_loss_rate := -0.08 },
   { name := "camera", price := 12000, currency := "JPY", tax_rate := 0.05, discount_rate := 0.12, profit_loss_rate := 0.15 }]

-- Function to calculate the final price of an item in INR
def calculate_final_price (item : Item) (conversion_rates : List (String × ℝ)) : ℝ :=
  sorry

-- Function to calculate the overall profit or loss
def calculate_overall_profit_loss (items : List Item) (conversion_rates : List (String × ℝ)) : ℝ :=
  sorry

-- Theorem statement
theorem overall_loss_is_184_76 :
  calculate_overall_profit_loss items conversion_rates = -184.76 := by sorry

end NUMINAMATH_CALUDE_overall_loss_is_184_76_l3867_386709


namespace NUMINAMATH_CALUDE_rectangular_to_spherical_conversion_l3867_386743

/-- Conversion from rectangular to spherical coordinates -/
theorem rectangular_to_spherical_conversion
  (x y z : ℝ)
  (h_x : x = 3 * Real.sqrt 2)
  (h_y : y = -3)
  (h_z : z = 5)
  (h_rho_pos : 0 < Real.sqrt (x^2 + y^2 + z^2))
  (h_theta_range : 0 ≤ 2 * Real.pi - Real.arctan (1 / Real.sqrt 2) ∧ 
                   2 * Real.pi - Real.arctan (1 / Real.sqrt 2) < 2 * Real.pi)
  (h_phi_range : 0 ≤ Real.arccos (z / Real.sqrt (x^2 + y^2 + z^2)) ∧ 
                 Real.arccos (z / Real.sqrt (x^2 + y^2 + z^2)) ≤ Real.pi) :
  (Real.sqrt (x^2 + y^2 + z^2),
   2 * Real.pi - Real.arctan (1 / Real.sqrt 2),
   Real.arccos (z / Real.sqrt (x^2 + y^2 + z^2))) =
  (Real.sqrt 52, 2 * Real.pi - Real.arctan (1 / Real.sqrt 2), Real.arccos (5 / Real.sqrt 52)) := by
  sorry

#check rectangular_to_spherical_conversion

end NUMINAMATH_CALUDE_rectangular_to_spherical_conversion_l3867_386743


namespace NUMINAMATH_CALUDE_boys_from_pine_l3867_386744

theorem boys_from_pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_girls : ℕ)
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : maple_students = 50)
  (h5 : pine_students = 100)
  (h6 : maple_girls = 30)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = maple_students + pine_students)
  (h9 : total_girls = maple_girls + (total_girls - maple_girls)) :
  pine_students - (total_girls - maple_girls) = 70 := by
  sorry

end NUMINAMATH_CALUDE_boys_from_pine_l3867_386744


namespace NUMINAMATH_CALUDE_smallest_n_value_l3867_386741

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2020 →
  c > a → c > b →
  c > a + 100 →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  (∀ k, k < n → ∃ l, a.factorial * b.factorial * c.factorial = l * (10 ^ k) ∧ 10 ∣ l) →
  n = 499 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l3867_386741


namespace NUMINAMATH_CALUDE_cos_105_degrees_l3867_386759

theorem cos_105_degrees : Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l3867_386759


namespace NUMINAMATH_CALUDE_book_price_increase_l3867_386730

theorem book_price_increase (initial_price : ℝ) : 
  let decrease_rate : ℝ := 0.20
  let net_change_rate : ℝ := 0.11999999999999986
  let price_after_decrease : ℝ := initial_price * (1 - decrease_rate)
  let final_price : ℝ := initial_price * (1 + net_change_rate)
  ∃ (increase_rate : ℝ), 
    price_after_decrease * (1 + increase_rate) = final_price ∧ 
    abs (increase_rate - 0.4) < 0.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_book_price_increase_l3867_386730


namespace NUMINAMATH_CALUDE_number_of_candidates_l3867_386762

/-- The number of ways to select a president and vice president -/
def selection_ways : ℕ := 30

/-- Theorem: Given 30 ways to select a president and vice president, 
    where the same person cannot be both, there are 6 candidates. -/
theorem number_of_candidates : 
  ∃ (n : ℕ), n > 0 ∧ n * (n - 1) = selection_ways := by
  sorry

end NUMINAMATH_CALUDE_number_of_candidates_l3867_386762


namespace NUMINAMATH_CALUDE_pen_problem_l3867_386785

theorem pen_problem (marked_price : ℝ) (num_pens : ℕ) : 
  marked_price > 0 →
  num_pens * marked_price = 46 * marked_price →
  (num_pens * marked_price * 0.99 - 46 * marked_price) / (46 * marked_price) * 100 = 29.130434782608695 →
  num_pens = 60 := by
sorry

end NUMINAMATH_CALUDE_pen_problem_l3867_386785


namespace NUMINAMATH_CALUDE_grocer_coffee_solution_l3867_386732

/-- Represents the grocer's coffee inventory --/
structure CoffeeInventory where
  initial : ℝ
  decafRatio : ℝ
  newPurchase : ℝ
  newDecafRatio : ℝ
  finalDecafRatio : ℝ

/-- The grocer's coffee inventory problem --/
def grocerProblem : CoffeeInventory where
  initial := 400  -- This is what we want to prove
  decafRatio := 0.2
  newPurchase := 100
  newDecafRatio := 0.5
  finalDecafRatio := 0.26

/-- Theorem stating the solution to the grocer's coffee inventory problem --/
theorem grocer_coffee_solution (inv : CoffeeInventory) : 
  inv.initial = 400 ∧ 
  inv.decafRatio = 0.2 ∧ 
  inv.newPurchase = 100 ∧ 
  inv.newDecafRatio = 0.5 ∧ 
  inv.finalDecafRatio = 0.26 →
  inv.finalDecafRatio * (inv.initial + inv.newPurchase) = 
    inv.decafRatio * inv.initial + inv.newDecafRatio * inv.newPurchase := by
  sorry

#check grocer_coffee_solution grocerProblem

end NUMINAMATH_CALUDE_grocer_coffee_solution_l3867_386732


namespace NUMINAMATH_CALUDE_sqrt_3_squared_4_fourth_l3867_386779

theorem sqrt_3_squared_4_fourth : Real.sqrt (3^2 * 4^4) = 48 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_squared_4_fourth_l3867_386779


namespace NUMINAMATH_CALUDE_compare_values_l3867_386756

theorem compare_values : 
  let a := (4 : ℝ) ^ (1/4 : ℝ)
  let b := (27 : ℝ) ^ (1/3 : ℝ)
  let c := (16 : ℝ) ^ (1/8 : ℝ)
  let d := (81 : ℝ) ^ (1/2 : ℝ)
  (d > a ∧ d > b ∧ d > c) ∧ 
  (b > a ∧ b > c) :=
by sorry

end NUMINAMATH_CALUDE_compare_values_l3867_386756


namespace NUMINAMATH_CALUDE_system_of_equations_product_l3867_386721

theorem system_of_equations_product (a b c d : ℚ) : 
  3*a + 4*b + 6*c + 8*d = 48 →
  4*(d+c) = b →
  4*b + 2*c = a →
  c - 2 = d →
  a * b * c * d = -1032192 / 1874161 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_product_l3867_386721


namespace NUMINAMATH_CALUDE_bank_depositors_bound_l3867_386747

theorem bank_depositors_bound (total_deposits : ℝ) (probability_less_100 : ℝ) 
  (h_total : total_deposits = 20000)
  (h_prob : probability_less_100 = 0.8) :
  ∃ n : ℕ, n ≤ 1000 ∧ (total_deposits / n : ℝ) ≤ 100 / (1 - probability_less_100) := by
  sorry

end NUMINAMATH_CALUDE_bank_depositors_bound_l3867_386747


namespace NUMINAMATH_CALUDE_number_minus_division_equals_l3867_386715

theorem number_minus_division_equals (x : ℝ) : x - (104 / 20.8) = 545 ↔ x = 550 := by
  sorry

end NUMINAMATH_CALUDE_number_minus_division_equals_l3867_386715


namespace NUMINAMATH_CALUDE_monotonicity_undetermined_l3867_386749

-- Define the real numbers a, b, and c
variable (a b c : ℝ)

-- Assume a < b < c
variable (h1 : a < b) (h2 : b < c)

-- Define a function f from reals to reals
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an open interval
def IncreasingOn (f : ℝ → ℝ) (l r : ℝ) : Prop :=
  ∀ x y, l < x ∧ x < y ∧ y < r → f x < f y

-- State the theorem
theorem monotonicity_undetermined
  (h_ab : IncreasingOn f a b)
  (h_bc : IncreasingOn f b c) :
  ¬ (IncreasingOn f a c ∨ (∀ x y, a < x ∧ x < y ∧ y < c → f x > f y)) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_undetermined_l3867_386749


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3867_386729

/-- The number of ways to place n distinguishable objects into k distinguishable containers -/
def ways_to_place (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^5 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : ways_to_place 5 3 = 243 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3867_386729


namespace NUMINAMATH_CALUDE_pythagorean_numbers_l3867_386795

theorem pythagorean_numbers (m : ℕ) (a b c : ℝ) : 
  m % 2 = 1 → 
  m > 1 → 
  a = (1/2 : ℝ) * m^2 - (1/2 : ℝ) → 
  c = (1/2 : ℝ) * m^2 + (1/2 : ℝ) → 
  a < c → 
  b < c → 
  a^2 + b^2 = c^2 → 
  b = m := by sorry

end NUMINAMATH_CALUDE_pythagorean_numbers_l3867_386795


namespace NUMINAMATH_CALUDE_virginia_friends_l3867_386765

/-- The number of friends Virginia gave Sweettarts to -/
def num_friends (total : ℕ) (per_person : ℕ) : ℕ :=
  (total / per_person) - 1

/-- Proof that Virginia gave Sweettarts to 3 friends -/
theorem virginia_friends :
  num_friends 13 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_virginia_friends_l3867_386765


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3867_386792

/-- For an infinite geometric series with common ratio 1/4 and sum 80, the first term is 60. -/
theorem infinite_geometric_series_first_term : 
  ∀ (a : ℝ), 
  (∑' n, a * (1/4)^n) = 80 → 
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3867_386792


namespace NUMINAMATH_CALUDE_distance_to_center_squared_l3867_386776

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the square of the distance between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: The square of the distance from B to the center of the circle is 50 -/
theorem distance_to_center_squared (O A B C : Point) : 
  O.x = 0 ∧ O.y = 0 →  -- Center at origin
  distanceSquared O A = 100 →  -- A is on the circle
  distanceSquared O C = 100 →  -- C is on the circle
  distanceSquared A B = 64 →  -- AB = 8
  distanceSquared B C = 9 →  -- BC = 3
  (B.x - A.x) * (C.y - B.y) = (B.y - A.y) * (C.x - B.x) →  -- ABC is a right angle
  distanceSquared O B = 50 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_center_squared_l3867_386776


namespace NUMINAMATH_CALUDE_third_discount_percentage_l3867_386780

/-- Given a car with an initial price and three successive discounts, 
    calculate the third discount percentage. -/
theorem third_discount_percentage 
  (initial_price : ℝ) 
  (first_discount second_discount : ℝ)
  (final_price : ℝ) :
  initial_price = 12000 →
  first_discount = 0.20 →
  second_discount = 0.15 →
  final_price = 7752 →
  ∃ (third_discount : ℝ),
    final_price = initial_price * 
      (1 - first_discount) * 
      (1 - second_discount) * 
      (1 - third_discount) ∧
    third_discount = 0.05 := by
  sorry


end NUMINAMATH_CALUDE_third_discount_percentage_l3867_386780


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3867_386794

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x = -2 ∧ x = 2) →
  (∃ y : ℝ, 3 * y^2 + m * y = -2 ∧ y = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3867_386794


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l3867_386716

/-- The value of k that makes the line 4x + 6y + k = 0 tangent to the parabola y^2 = 32x -/
def tangent_k : ℝ := 72

/-- The line equation -/
def line (x y k : ℝ) : Prop := 4 * x + 6 * y + k = 0

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 32 * x

/-- The tangency condition -/
def is_tangent (k : ℝ) : Prop :=
  ∃! (x y : ℝ), line x y k ∧ parabola x y

theorem tangent_line_to_parabola :
  is_tangent tangent_k :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l3867_386716


namespace NUMINAMATH_CALUDE_valid_tiling_characterization_l3867_386717

/-- A tetromino type -/
inductive Tetromino
| T
| Square

/-- Represents a tiling of an n × n field -/
structure Tiling (n : ℕ) where
  pieces : List (Tetromino × ℕ × ℕ)  -- List of (type, row, col) for each piece
  no_gaps : Sorry
  no_overlaps : Sorry
  covers_field : Sorry
  odd_squares : Sorry  -- The number of square tetrominoes is odd

/-- Main theorem: Characterization of valid n for tiling -/
theorem valid_tiling_characterization (n : ℕ) :
  (∃ (t : Tiling n), True) ↔ (∃ (k : ℕ), n = 2 * k ∧ k % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_valid_tiling_characterization_l3867_386717


namespace NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l3867_386793

theorem x_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, x ^ 2 - 3 * x + 2 = 0 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x ^ 2 - 3 * x + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l3867_386793


namespace NUMINAMATH_CALUDE_banana_preference_percentage_l3867_386703

/-- Represents the preference count for each fruit in the survey. -/
structure FruitPreferences where
  apple : ℕ
  banana : ℕ
  cherry : ℕ
  dragonfruit : ℕ

/-- Calculates the percentage of people who preferred a specific fruit. -/
def fruitPercentage (prefs : FruitPreferences) (fruitCount : ℕ) : ℚ :=
  (fruitCount : ℚ) / (prefs.apple + prefs.banana + prefs.cherry + prefs.dragonfruit : ℚ) * 100

/-- Theorem stating that the percentage of people who preferred Banana is 37.5%. -/
theorem banana_preference_percentage
  (prefs : FruitPreferences)
  (h1 : prefs.apple = 45)
  (h2 : prefs.banana = 75)
  (h3 : prefs.cherry = 30)
  (h4 : prefs.dragonfruit = 50) :
  fruitPercentage prefs prefs.banana = 37.5 := by
  sorry

#eval fruitPercentage ⟨45, 75, 30, 50⟩ 75

end NUMINAMATH_CALUDE_banana_preference_percentage_l3867_386703


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_2x_l3867_386768

theorem integral_sqrt_minus_2x : 
  ∫ x in (0:ℝ)..1, (Real.sqrt (1 - (x - 1)^2) - 2*x) = π/4 - 1 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_2x_l3867_386768


namespace NUMINAMATH_CALUDE_bakery_chairs_l3867_386753

/-- The number of chairs in a bakery -/
def total_chairs (indoor_tables outdoor_tables chairs_per_indoor_table chairs_per_outdoor_table : ℕ) : ℕ :=
  indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table

/-- Proof that the total number of chairs in the bakery is 60 -/
theorem bakery_chairs :
  total_chairs 8 12 3 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bakery_chairs_l3867_386753


namespace NUMINAMATH_CALUDE_function_equality_l3867_386774

theorem function_equality (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = x + f (f y)) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_equality_l3867_386774


namespace NUMINAMATH_CALUDE_archies_sod_area_l3867_386789

/-- Calculates the area of sod needed for a rectangular backyard with a rectangular shed. -/
def area_of_sod_needed (backyard_length backyard_width shed_length shed_width : ℝ) : ℝ :=
  backyard_length * backyard_width - shed_length * shed_width

/-- Theorem: The area of sod needed for Archie's backyard is 245 square yards. -/
theorem archies_sod_area :
  area_of_sod_needed 20 13 3 5 = 245 := by
  sorry

end NUMINAMATH_CALUDE_archies_sod_area_l3867_386789


namespace NUMINAMATH_CALUDE_base8_subtraction_and_conversion_l3867_386720

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Subtracts two numbers in base 8 -/
def subtractBase8 (a b : ℕ) : ℕ := sorry

theorem base8_subtraction_and_conversion :
  let a := 7463
  let b := 3254
  let result_base8 := subtractBase8 a b
  let result_base10 := base8ToBase10 result_base8
  result_base8 = 4207 ∧ result_base10 = 2183 := by sorry

end NUMINAMATH_CALUDE_base8_subtraction_and_conversion_l3867_386720


namespace NUMINAMATH_CALUDE_chocolate_bar_sales_l3867_386740

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Proves that selling 4 out of 11 bars at $4 each yields $16 -/
theorem chocolate_bar_sales : money_made 11 4 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_sales_l3867_386740


namespace NUMINAMATH_CALUDE_max_absolute_value_of_z_l3867_386761

theorem max_absolute_value_of_z (a b c z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs b = Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ (1 + Real.sqrt 5) / 2 := by sorry

end NUMINAMATH_CALUDE_max_absolute_value_of_z_l3867_386761


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l3867_386737

/-- Given a right pyramid with a square base, if the area of one lateral face
    is 120 square meters and the slant height is 40 meters, then the length
    of the side of its base is 6 meters. -/
theorem pyramid_base_side_length
  (area : ℝ) (slant_height : ℝ) (base_side : ℝ) :
  area = 120 →
  slant_height = 40 →
  area = (1/2) * base_side * slant_height →
  base_side = 6 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l3867_386737


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l3867_386781

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 ∧ h₁ > 0 ∧ r₂ > 0 ∧ h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l3867_386781


namespace NUMINAMATH_CALUDE_rook_placements_corners_removed_8x8_l3867_386731

/-- Represents a chessboard with corners removed -/
def CornersRemovedChessboard : Type := Unit

/-- The number of ways to place non-attacking rooks on a corners-removed chessboard -/
def num_rook_placements (board : CornersRemovedChessboard) : ℕ := 21600

/-- The theorem stating the number of ways to place eight non-attacking rooks
    on an 8x8 chessboard with its four corners removed -/
theorem rook_placements_corners_removed_8x8 (board : CornersRemovedChessboard) :
  num_rook_placements board = 21600 := by sorry

end NUMINAMATH_CALUDE_rook_placements_corners_removed_8x8_l3867_386731


namespace NUMINAMATH_CALUDE_discount_percentage_decrease_l3867_386725

theorem discount_percentage_decrease (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * (1 + 0.25)
  let decrease_percentage := (increased_price - original_price) / increased_price
  decrease_percentage = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_decrease_l3867_386725


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3867_386755

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a^2 + b^2 = 0 → a * b = 0) ∧
  ∃ a b : ℝ, a * b = 0 ∧ a^2 + b^2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3867_386755


namespace NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l3867_386769

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 4, 6, 8}

-- Define set M
def M : Set ℕ := {0, 4, 6}

-- Define set N
def N : Set ℕ := {0, 1, 6}

-- Theorem statement
theorem intersection_of_M_and_complement_of_N :
  M ∩ (U \ N) = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l3867_386769


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l3867_386739

theorem point_on_unit_circle (s : ℝ) : 
  let x := (3 - s^2) / (3 + s^2)
  let y := 4*s / (3 + s^2)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l3867_386739


namespace NUMINAMATH_CALUDE_shortest_line_on_square_pyramid_l3867_386770

/-- The shortest line on the lateral faces of a square pyramid -/
theorem shortest_line_on_square_pyramid (a m : ℝ) (ha : a > 0) (hm : m > 0) (h_eq : a = m) :
  let x := Real.sqrt (2 * a^2)
  let m₁ := Real.sqrt (x^2 - (a/2)^2)
  2 * a * m₁ / x = 80 * Real.sqrt (5/6) :=
by sorry

end NUMINAMATH_CALUDE_shortest_line_on_square_pyramid_l3867_386770


namespace NUMINAMATH_CALUDE_exactly_one_statement_correct_l3867_386722

-- Define rational and irrational numbers
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- Define the four statements
def Statement1 : Prop :=
  ∀ (r i : ℝ), IsRational r → IsIrrational i → IsIrrational (r + i)

def Statement2 : Prop :=
  ∀ (r i : ℝ), IsRational r → IsIrrational i → IsIrrational (r * i)

def Statement3 : Prop :=
  ∀ (i₁ i₂ : ℝ), IsIrrational i₁ → IsIrrational i₂ → IsIrrational (i₁ + i₂)

def Statement4 : Prop :=
  ∀ (i₁ i₂ : ℝ), IsIrrational i₁ → IsIrrational i₂ → IsIrrational (i₁ * i₂)

-- The main theorem
theorem exactly_one_statement_correct :
  (Statement1 ∧ ¬Statement2 ∧ ¬Statement3 ∧ ¬Statement4) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_statement_correct_l3867_386722


namespace NUMINAMATH_CALUDE_unique_solution_range_l3867_386777

theorem unique_solution_range (a : ℝ) : 
  (∃! x : ℝ, 1 < x ∧ x < 3 ∧ Real.log (x - 1) + Real.log (3 - x) = Real.log (x - a)) ↔ 
  (3/4 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_range_l3867_386777


namespace NUMINAMATH_CALUDE_count_valid_parallelograms_l3867_386742

/-- A point in the coordinate plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A parallelogram defined by four points in the coordinate plane -/
structure Parallelogram where
  O : IntPoint
  A : IntPoint
  B : IntPoint
  C : IntPoint

/-- The center of a parallelogram -/
def center (p : Parallelogram) : IntPoint :=
  { x := (p.O.x + p.B.x) / 2,
    y := (p.O.y + p.B.y) / 2 }

/-- Check if a parallelogram satisfies the given conditions -/
def validParallelogram (p : Parallelogram) : Prop :=
  p.O = { x := 0, y := 0 } ∧
  center p = { x := 19, y := 15 } ∧
  p.A.x > 0 ∧ p.A.y > 0 ∧
  p.B.x > 0 ∧ p.B.y > 0 ∧
  p.C.x > 0 ∧ p.C.y > 0

/-- Two parallelograms are considered equivalent if they have the same set of vertices -/
def equivalentParallelograms (p1 p2 : Parallelogram) : Prop :=
  (p1.O = p2.O ∧ p1.A = p2.A ∧ p1.B = p2.B ∧ p1.C = p2.C) ∨
  (p1.O = p2.O ∧ p1.A = p2.C ∧ p1.B = p2.B ∧ p1.C = p2.A)

theorem count_valid_parallelograms :
  ∃ (s : Finset Parallelogram),
    (∀ p ∈ s, validParallelogram p) ∧
    (∀ p, validParallelogram p → ∃ q ∈ s, equivalentParallelograms p q) ∧
    s.card = 126 :=
sorry

end NUMINAMATH_CALUDE_count_valid_parallelograms_l3867_386742


namespace NUMINAMATH_CALUDE_max_value_cube_sum_ratio_l3867_386745

theorem max_value_cube_sum_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^3 / (x^3 + y^3 + z^3) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cube_sum_ratio_l3867_386745


namespace NUMINAMATH_CALUDE_sports_conference_games_l3867_386766

/-- Calculates the number of games in a sports conference season -/
def conference_games (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * inter_division_games
  (total_teams * games_per_team) / 2

/-- Theorem: The number of games in the described sports conference is 232 -/
theorem sports_conference_games : 
  conference_games 16 8 3 1 = 232 := by sorry

end NUMINAMATH_CALUDE_sports_conference_games_l3867_386766


namespace NUMINAMATH_CALUDE_odd_function_implies_a_equals_negative_one_l3867_386726

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (x+a+1)(x^2+a-1) -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (x + a + 1) * (x^2 + a - 1)

theorem odd_function_implies_a_equals_negative_one :
  ∀ a : ℝ, IsOdd (f a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_equals_negative_one_l3867_386726


namespace NUMINAMATH_CALUDE_matinee_customers_count_l3867_386707

/-- Represents the revenue calculation for a movie theater. -/
def theater_revenue (matinee_customers : ℕ) : ℕ :=
  let matinee_price : ℕ := 5
  let evening_price : ℕ := 7
  let opening_night_price : ℕ := 10
  let popcorn_price : ℕ := 10
  let evening_customers : ℕ := 40
  let opening_night_customers : ℕ := 58
  let total_customers : ℕ := matinee_customers + evening_customers + opening_night_customers
  let popcorn_customers : ℕ := total_customers / 2

  matinee_price * matinee_customers +
  evening_price * evening_customers +
  opening_night_price * opening_night_customers +
  popcorn_price * popcorn_customers

/-- Theorem stating that the number of matinee customers is 32. -/
theorem matinee_customers_count : ∃ (n : ℕ), theater_revenue n = 1670 ∧ n = 32 :=
sorry

end NUMINAMATH_CALUDE_matinee_customers_count_l3867_386707


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3867_386718

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = Complex.abs (4 + 3*I)) :
  z.im = 4/5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3867_386718


namespace NUMINAMATH_CALUDE_alicia_miles_run_l3867_386773

/-- Represents the step counter's maximum value before reset --/
def max_steps : Nat := 99999

/-- Represents the number of times the counter reset --/
def reset_count : Nat := 50

/-- Represents the steps shown on the counter on the last day --/
def final_steps : Nat := 25000

/-- Represents the number of steps Alicia takes per mile --/
def steps_per_mile : Nat := 1500

/-- Calculates the total number of steps Alicia took over the year --/
def total_steps : Nat := (max_steps + 1) * reset_count + final_steps

/-- Calculates the approximate number of miles Alicia ran --/
def miles_run : Nat := total_steps / steps_per_mile

/-- Theorem stating that Alicia ran approximately 3350 miles --/
theorem alicia_miles_run : miles_run = 3350 := by
  sorry

end NUMINAMATH_CALUDE_alicia_miles_run_l3867_386773


namespace NUMINAMATH_CALUDE_max_radius_of_circle_max_radius_achieved_l3867_386705

/-- A circle in a rectangular coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def pointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem max_radius_of_circle (c : Circle) :
  pointOnCircle c (8, 0) → pointOnCircle c (-8, 0) → c.radius ≤ 8 := by
  sorry

theorem max_radius_achieved (r : ℝ) :
  r ≤ 8 →
  ∃ c : Circle, pointOnCircle c (8, 0) ∧ pointOnCircle c (-8, 0) ∧ c.radius = r := by
  sorry

end NUMINAMATH_CALUDE_max_radius_of_circle_max_radius_achieved_l3867_386705


namespace NUMINAMATH_CALUDE_sector_central_angle_l3867_386787

/-- Theorem: Given a circular sector with arc length 3 and radius 2, the central angle is 3/2 radians. -/
theorem sector_central_angle (l : ℝ) (r : ℝ) (θ : ℝ) 
  (hl : l = 3) (hr : r = 2) (hθ : l = r * θ) : θ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3867_386787


namespace NUMINAMATH_CALUDE_lcm_of_15_25_35_l3867_386752

theorem lcm_of_15_25_35 : Nat.lcm 15 (Nat.lcm 25 35) = 525 := by sorry

end NUMINAMATH_CALUDE_lcm_of_15_25_35_l3867_386752


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3867_386790

theorem regular_polygon_sides (n : ℕ) (interior_angle exterior_angle : ℝ) : 
  n > 2 →
  interior_angle = exterior_angle + 60 →
  interior_angle + exterior_angle = 180 →
  n * exterior_angle = 360 →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3867_386790


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a20_l3867_386760

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a20 (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 1 + a 3 + a 5 = 105 →
  a 2 + a 4 + a 6 = 99 →
  a 20 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a20_l3867_386760


namespace NUMINAMATH_CALUDE_xyz_value_l3867_386751

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 18)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 6) : 
  x * y * z = 4 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3867_386751


namespace NUMINAMATH_CALUDE_system_solution_l3867_386791

theorem system_solution : 
  ∃! (x y : ℝ), x + y = 8 ∧ 2*x - y = 7 ∧ x = 5 ∧ y = 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3867_386791


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3867_386775

def S : Finset Int := {-7, -5, -3, -2, 2, 4, 6, 13}

theorem min_sum_of_squares (a b c d e f g h : Int) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
                c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
                d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
                e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
                f ≠ g ∧ f ≠ h ∧
                g ≠ h)
  (h_in_S : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 34 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : Int), 
    (a' ∈ S ∧ b' ∈ S ∧ c' ∈ S ∧ d' ∈ S ∧ e' ∈ S ∧ f' ∈ S ∧ g' ∈ S ∧ h' ∈ S) ∧
    (a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ a' ≠ e' ∧ a' ≠ f' ∧ a' ≠ g' ∧ a' ≠ h' ∧
     b' ≠ c' ∧ b' ≠ d' ∧ b' ≠ e' ∧ b' ≠ f' ∧ b' ≠ g' ∧ b' ≠ h' ∧
     c' ≠ d' ∧ c' ≠ e' ∧ c' ≠ f' ∧ c' ≠ g' ∧ c' ≠ h' ∧
     d' ≠ e' ∧ d' ≠ f' ∧ d' ≠ g' ∧ d' ≠ h' ∧
     e' ≠ f' ∧ e' ≠ g' ∧ e' ≠ h' ∧
     f' ≠ g' ∧ f' ≠ h' ∧
     g' ≠ h') ∧
    (a' + b' + c' + d')^2 + (e' + f' + g' + h')^2 = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3867_386775


namespace NUMINAMATH_CALUDE_sin_810_degrees_l3867_386711

theorem sin_810_degrees : Real.sin (810 * π / 180) = 1 := by sorry

end NUMINAMATH_CALUDE_sin_810_degrees_l3867_386711


namespace NUMINAMATH_CALUDE_gate_width_scientific_notation_l3867_386724

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  prop : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem gate_width_scientific_notation :
  toScientificNotation 0.000000007 = ScientificNotation.mk 7 (-9) sorry := by
  sorry

end NUMINAMATH_CALUDE_gate_width_scientific_notation_l3867_386724


namespace NUMINAMATH_CALUDE_nested_bracket_value_l3867_386799

def bracket (a b c : ℚ) : ℚ :=
  if c ≠ 0 then (a + b) / c else 0

theorem nested_bracket_value :
  bracket (bracket 30 45 75) (bracket 4 2 6) (bracket 12 18 30) = 2 :=
by sorry

end NUMINAMATH_CALUDE_nested_bracket_value_l3867_386799


namespace NUMINAMATH_CALUDE_complement_of_union_l3867_386704

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x ∈ U | x^2 - 4*x + 3 = 0}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3867_386704


namespace NUMINAMATH_CALUDE_closest_to_95_l3867_386782

def options : List ℝ := [90, 92, 95, 98, 100]

theorem closest_to_95 :
  let product := 2.1 * (45.5 - 0.25)
  ∀ x ∈ options, |product - 95| ≤ |product - x| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_to_95_l3867_386782


namespace NUMINAMATH_CALUDE_problem_statement_l3867_386772

def B : Set ℝ := {m | m < 2}

def A (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem problem_statement :
  (∀ m : ℝ, m ∈ B ↔ ∀ x : ℝ, x ≥ 2 → x^2 - x - m > 0) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B) → A a ⊂ B → a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3867_386772


namespace NUMINAMATH_CALUDE_count_subsets_correct_l3867_386798

/-- Given a natural number n, this function returns the number of two-tuples (X, Y) 
    of subsets of {1, 2, ..., n} such that max X > min Y -/
def count_subsets (n : ℕ) : ℕ := 
  2^(2*n) - (n+1) * 2^n

/-- Theorem stating that count_subsets gives the correct number of two-tuples -/
theorem count_subsets_correct (n : ℕ) : 
  count_subsets n = (Finset.powerset (Finset.range n)).card * 
                    (Finset.powerset (Finset.range n)).card - 
                    (Finset.filter (fun p : Finset ℕ × Finset ℕ => 
                      p.1.max ≤ p.2.min) 
                      ((Finset.powerset (Finset.range n)).product 
                       (Finset.powerset (Finset.range n)))).card :=
  sorry

#eval count_subsets 3  -- Example usage

end NUMINAMATH_CALUDE_count_subsets_correct_l3867_386798


namespace NUMINAMATH_CALUDE_tangent_line_intersection_three_distinct_solutions_l3867_386767

/-- The function f(x) = x³ - 9x -/
def f (x : ℝ) : ℝ := x^3 - 9*x

/-- The function g(x) = 3x² + a -/
def g (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 9

/-- The derivative of g -/
def g' (x : ℝ) : ℝ := 6*x

theorem tangent_line_intersection (a : ℝ) :
  (∃ m : ℝ, f' 0 = g' m ∧ f 0 + f' 0 * m = g a m) → a = 27/4 :=
sorry

theorem three_distinct_solutions (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = g a x ∧ f y = g a y ∧ f z = g a z) ↔ 
  -27 < a ∧ a < 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_three_distinct_solutions_l3867_386767


namespace NUMINAMATH_CALUDE_sock_order_ratio_l3867_386713

def sock_order_problem (black_socks green_socks : ℕ) (price_green : ℝ) : Prop :=
  let price_black := 3 * price_green
  let original_cost := black_socks * price_black + green_socks * price_green
  let interchanged_cost := green_socks * price_black + black_socks * price_green
  black_socks = 5 ∧
  interchanged_cost = 1.8 * original_cost ∧
  (black_socks : ℝ) / green_socks = 3 / 11

theorem sock_order_ratio :
  ∃ (green_socks : ℕ) (price_green : ℝ),
    sock_order_problem 5 green_socks price_green :=
by sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l3867_386713
