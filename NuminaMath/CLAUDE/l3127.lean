import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l3127_312799

theorem range_of_a (a : ℝ) : (∀ x > 0, x^2 + a*x + 1 ≥ 0) → a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3127_312799


namespace NUMINAMATH_CALUDE_path_area_is_675_l3127_312713

/-- Calculates the area of a path surrounding a rectangular field. -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Theorem: The area of the path surrounding the given rectangular field is 675 sq m. -/
theorem path_area_is_675 (field_length field_width path_width cost_per_sqm total_cost : ℝ) :
  field_length = 75 →
  field_width = 55 →
  path_width = 2.5 →
  cost_per_sqm = 10 →
  total_cost = 6750 →
  path_area field_length field_width path_width = 675 :=
by
  sorry

#eval path_area 75 55 2.5

end NUMINAMATH_CALUDE_path_area_is_675_l3127_312713


namespace NUMINAMATH_CALUDE_average_calculation_l3127_312740

theorem average_calculation (math_score history_score third_score : ℚ)
  (h1 : math_score = 74/100)
  (h2 : history_score = 81/100)
  (h3 : third_score = 70/100) :
  (math_score + history_score + third_score) / 3 = 75/100 := by
  sorry

end NUMINAMATH_CALUDE_average_calculation_l3127_312740


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_2310_l3127_312730

theorem sum_of_prime_factors_2310 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (2310 + 1))) id) = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_2310_l3127_312730


namespace NUMINAMATH_CALUDE_at_least_one_root_exists_l3127_312707

theorem at_least_one_root_exists (c m a n : ℝ) : 
  (m^2 + 4*a*c ≥ 0) ∨ (n^2 - 4*a*c ≥ 0) := by sorry

end NUMINAMATH_CALUDE_at_least_one_root_exists_l3127_312707


namespace NUMINAMATH_CALUDE_total_material_needed_l3127_312742

-- Define the dimensions of the tablecloth
def tablecloth_length : ℕ := 102
def tablecloth_width : ℕ := 54

-- Define the dimensions of a napkin
def napkin_length : ℕ := 6
def napkin_width : ℕ := 7

-- Define the number of napkins
def num_napkins : ℕ := 8

-- Theorem to prove
theorem total_material_needed :
  tablecloth_length * tablecloth_width + num_napkins * napkin_length * napkin_width = 5844 :=
by sorry

end NUMINAMATH_CALUDE_total_material_needed_l3127_312742


namespace NUMINAMATH_CALUDE_triangle_area_prime_l3127_312720

/-- The area of a triangle formed by the line y = 10x - a and the coordinate axes -/
def triangleArea (a : ℤ) : ℚ := a^2 / 20

/-- Predicate to check if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

theorem triangle_area_prime :
  ∀ a : ℤ,
  (∃ n : ℕ, (triangleArea a).num = n ∧ (triangleArea a).den = 1 ∧ isPrime n) →
  triangleArea a = 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_prime_l3127_312720


namespace NUMINAMATH_CALUDE_order_of_expressions_l3127_312771

theorem order_of_expressions : 
  let a : ℝ := (3/5)^(2/5)
  let b : ℝ := (2/5)^(3/5)
  let c : ℝ := (2/5)^(2/5)
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l3127_312771


namespace NUMINAMATH_CALUDE_expression_simplification_l3127_312715

theorem expression_simplification (a b x y : ℝ) :
  (2*a - (4*a + 5*b) + 2*(3*a - 4*b) = 4*a - 13*b) ∧
  (5*x^2 - 2*(3*y^2 - 5*x^2) + (-4*y^2 + 7*x*y) = 15*x^2 - 10*y^2 + 7*x*y) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3127_312715


namespace NUMINAMATH_CALUDE_painting_difference_l3127_312756

/-- Represents a 5x5x5 cube -/
structure Cube :=
  (size : Nat)
  (h_size : size = 5)

/-- Counts the number of unit cubes with at least one painted face when two opposite faces and one additional face are painted -/
def count_painted_opposite_plus_one (c : Cube) : Nat :=
  c.size * c.size + (c.size - 2) * c.size + c.size * c.size

/-- Counts the number of unit cubes with at least one painted face when three adjacent faces sharing one vertex are painted -/
def count_painted_adjacent (c : Cube) : Nat :=
  (c.size - 1) * 9 + c.size * c.size

/-- The difference between the two painting configurations is 4 -/
theorem painting_difference (c : Cube) : 
  count_painted_opposite_plus_one c - count_painted_adjacent c = 4 := by
  sorry


end NUMINAMATH_CALUDE_painting_difference_l3127_312756


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3127_312743

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_of_M_and_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3127_312743


namespace NUMINAMATH_CALUDE_prob_not_shaded_is_500_1001_l3127_312776

/-- Represents a 2 by 1001 rectangle with middle squares shaded -/
structure ShadedRectangle where
  width : ℕ := 2
  length : ℕ := 1001
  middle_shaded : ℕ := (length + 1) / 2

/-- Calculates the total number of rectangles in the figure -/
def total_rectangles (r : ShadedRectangle) : ℕ :=
  r.width * (r.length * (r.length + 1)) / 2

/-- Calculates the number of rectangles that include a shaded square -/
def shaded_rectangles (r : ShadedRectangle) : ℕ :=
  r.width * r.middle_shaded * (r.length - r.middle_shaded + 1)

/-- The probability of choosing a rectangle that doesn't include a shaded square -/
def prob_not_shaded (r : ShadedRectangle) : ℚ :=
  1 - (shaded_rectangles r : ℚ) / (total_rectangles r : ℚ)

theorem prob_not_shaded_is_500_1001 (r : ShadedRectangle) :
  prob_not_shaded r = 500 / 1001 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_shaded_is_500_1001_l3127_312776


namespace NUMINAMATH_CALUDE_total_books_l3127_312727

theorem total_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 8)
  (h2 : mystery_shelves = 12)
  (h3 : picture_shelves = 9) :
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 168 :=
by
  sorry

end NUMINAMATH_CALUDE_total_books_l3127_312727


namespace NUMINAMATH_CALUDE_quadratic_root_n_value_l3127_312772

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 8 * x - 5 = 0

-- Define the root form
def root_form (x m n p : ℝ) : Prop := 
  (x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) ∧ 
  m > 0 ∧ n > 0 ∧ p > 0 ∧ Int.gcd ⌊m⌋ (Int.gcd ⌊n⌋ ⌊p⌋) = 1

-- Theorem statement
theorem quadratic_root_n_value : 
  ∃ (x m p : ℝ), quadratic_equation x ∧ root_form x m 31 p := by sorry

end NUMINAMATH_CALUDE_quadratic_root_n_value_l3127_312772


namespace NUMINAMATH_CALUDE_equation_result_l3127_312797

theorem equation_result : 300 * 2 + (12 + 4) * 1 / 8 = 602 := by
  sorry

end NUMINAMATH_CALUDE_equation_result_l3127_312797


namespace NUMINAMATH_CALUDE_gcd_of_128_144_480_l3127_312738

theorem gcd_of_128_144_480 : Nat.gcd 128 (Nat.gcd 144 480) = 16 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_128_144_480_l3127_312738


namespace NUMINAMATH_CALUDE_left_handed_sci_fi_fans_count_l3127_312758

/-- Represents the book club with its member properties -/
structure BookClub where
  total_members : ℕ
  left_handed : ℕ
  sci_fi_fans : ℕ
  right_handed_non_sci_fi : ℕ

/-- The number of left-handed members who like sci-fi books in the book club -/
def left_handed_sci_fi_fans (club : BookClub) : ℕ :=
  club.total_members - (club.left_handed + club.sci_fi_fans + club.right_handed_non_sci_fi) + club.left_handed + club.sci_fi_fans - club.total_members

/-- Theorem stating that the number of left-handed sci-fi fans is 4 for the given book club -/
theorem left_handed_sci_fi_fans_count (club : BookClub) 
  (h1 : club.total_members = 30)
  (h2 : club.left_handed = 12)
  (h3 : club.sci_fi_fans = 18)
  (h4 : club.right_handed_non_sci_fi = 4) :
  left_handed_sci_fi_fans club = 4 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_sci_fi_fans_count_l3127_312758


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3127_312762

noncomputable def f (x : ℝ) := Real.exp x * (x^2 + x + 1)

theorem f_monotonicity_and_extrema :
  (∀ x y, x < y ∧ y < -2 → f x < f y) ∧
  (∀ x y, -2 < x ∧ x < y ∧ y < -1 → f x > f y) ∧
  (∀ x y, -1 < x ∧ x < y → f x < f y) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-2)| < δ ∧ x ≠ -2 → f x < f (-2)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-1)| < δ ∧ x ≠ -1 → f x > f (-1)) ∧
  f (-2) = 3 / Real.exp 2 ∧
  f (-1) = 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3127_312762


namespace NUMINAMATH_CALUDE_billy_already_ahead_l3127_312774

def billy_miles : List ℝ := [2, 3, 0, 4, 1, 0]
def tiffany_miles : List ℝ := [1.5, 0, 2.5, 2.5, 3, 0]

theorem billy_already_ahead : 
  (billy_miles.sum > tiffany_miles.sum) ∧ 
  (billy_miles.length = tiffany_miles.length) := by
  sorry

end NUMINAMATH_CALUDE_billy_already_ahead_l3127_312774


namespace NUMINAMATH_CALUDE_cubic_factorization_l3127_312769

theorem cubic_factorization (x : ℝ) : 
  x^3 + x^2 - 2*x - 2 = (x + 1) * (x - Real.sqrt 2) * (x + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3127_312769


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l3127_312725

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (p₁ p₂ p₃ p₄ : Nat), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    210 % p₁ = 0 ∧ 210 % p₂ = 0 ∧ 210 % p₃ = 0 ∧ 210 % p₄ = 0 ∧
    ∀ n : Nat, n > 0 ∧ n < 210 → 
      ¬∃ (q₁ q₂ q₃ q₄ : Nat), 
        Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
        q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
        n % q₁ = 0 ∧ n % q₂ = 0 ∧ n % q₃ = 0 ∧ n % q₄ = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l3127_312725


namespace NUMINAMATH_CALUDE_five_pow_minus_two_pow_div_by_three_l3127_312732

theorem five_pow_minus_two_pow_div_by_three (n : ℕ) :
  ∃ k : ℤ, 5^n - 2^n = 3 * k :=
sorry

end NUMINAMATH_CALUDE_five_pow_minus_two_pow_div_by_three_l3127_312732


namespace NUMINAMATH_CALUDE_shortest_distance_circle_to_origin_l3127_312701

/-- The shortest distance between any point on the circle (x-2)^2+(y+m-4)^2=1 and the origin (0,0) is 1, where m is a real number. -/
theorem shortest_distance_circle_to_origin :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), (x - 2)^2 + (y + m - 4)^2 = 1) →
  (∃ (d : ℝ), d = 1 ∧ 
    ∀ (x y : ℝ), (x - 2)^2 + (y + m - 4)^2 = 1 → 
      Real.sqrt (x^2 + y^2) ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_circle_to_origin_l3127_312701


namespace NUMINAMATH_CALUDE_set_operation_result_l3127_312757

open Set

def U : Set Int := univ
def A : Set Int := {-2, -1, 0, 1, 2}
def B : Set Int := {-1, 0, 1, 2, 3}

theorem set_operation_result : A ∩ (U \ B) = {-2} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l3127_312757


namespace NUMINAMATH_CALUDE_correct_expression_proof_l3127_312780

theorem correct_expression_proof (x a b : ℝ) : 
  ((2*x - a) * (3*x + b) = 6*x^2 - 13*x + 6) →
  ((2*x + a) * (x + b) = 2*x^2 - x - 6) →
  (a = 3 ∧ b = -2 ∧ (2*x + a) * (3*x + b) = 6*x^2 + 5*x - 6) := by
  sorry

end NUMINAMATH_CALUDE_correct_expression_proof_l3127_312780


namespace NUMINAMATH_CALUDE_product_of_solutions_l3127_312722

theorem product_of_solutions (x : ℝ) : 
  (|5 * x| + 7 = 47) → (∃ y : ℝ, |5 * y| + 7 = 47 ∧ x * y = -64) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3127_312722


namespace NUMINAMATH_CALUDE_zoo_visitors_l3127_312746

theorem zoo_visitors (sandwiches_per_person : ℝ) (total_sandwiches : ℕ) :
  sandwiches_per_person = 3.0 →
  total_sandwiches = 657 →
  ↑total_sandwiches / sandwiches_per_person = 219 := by
sorry

end NUMINAMATH_CALUDE_zoo_visitors_l3127_312746


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3127_312721

theorem unique_solution_equation (x : ℝ) : 
  x > 12 ∧ (x - 5) / 12 = 5 / (x - 12) ↔ x = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3127_312721


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3127_312777

theorem circle_y_axis_intersection_sum : 
  ∀ (x y : ℝ), 
  ((x + 8)^2 + (y - 5)^2 = 13^2) →  -- Circle equation
  (x = 0) →                        -- Points on y-axis
  ∃ (y1 y2 : ℝ),
    ((0 + 8)^2 + (y1 - 5)^2 = 13^2) ∧
    ((0 + 8)^2 + (y2 - 5)^2 = 13^2) ∧
    y1 + y2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3127_312777


namespace NUMINAMATH_CALUDE_computer_table_price_l3127_312787

/-- Calculates the selling price given the cost price and markup percentage -/
def selling_price (cost_price : ℚ) (markup_percent : ℚ) : ℚ :=
  cost_price * (1 + markup_percent / 100)

/-- Proves that the selling price of a computer table with cost price 4480 and 25% markup is 5600 -/
theorem computer_table_price : selling_price 4480 25 = 5600 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_price_l3127_312787


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l3127_312765

theorem right_triangle_ratio (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_right_triangle : a^2 + b^2 = c^2) : 
  (a^2 + b^2) / (a^2 + b^2 + c^2) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l3127_312765


namespace NUMINAMATH_CALUDE_coefficient_of_3_squared_x_squared_l3127_312789

/-- Definition of a coefficient in an algebraic term -/
def is_coefficient (c : ℝ) (term : ℝ → ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, term x = c * f x

/-- The coefficient of 3^2 * x^2 is 3^2 -/
theorem coefficient_of_3_squared_x_squared :
  is_coefficient (3^2) (λ x => 3^2 * x^2) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_3_squared_x_squared_l3127_312789


namespace NUMINAMATH_CALUDE_like_terms_imply_sum_l3127_312729

-- Define the concept of "like terms" for our specific case
def are_like_terms (m n : ℤ) : Prop :=
  m + 3 = 4 ∧ n + 3 = 1

-- State the theorem
theorem like_terms_imply_sum (m n : ℤ) :
  are_like_terms m n → m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_sum_l3127_312729


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3127_312741

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement that a_3 and a_10 are roots of x^2 - 3x - 5 = 0 -/
def roots_condition (a : ℕ → ℝ) : Prop :=
  a 3 ^ 2 - 3 * a 3 - 5 = 0 ∧ a 10 ^ 2 - 3 * a 10 - 5 = 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : roots_condition a) : 
  a 5 + a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3127_312741


namespace NUMINAMATH_CALUDE_max_knights_and_courtiers_l3127_312790

/-- Represents the number of people at each table -/
structure TableCounts where
  king : ℕ
  courtiers : ℕ
  knights : ℕ

/-- Checks if the table counts are valid according to the problem constraints -/
def is_valid_table_counts (tc : TableCounts) : Prop :=
  tc.king = 7 ∧ 
  12 ≤ tc.courtiers ∧ tc.courtiers ≤ 18 ∧
  10 ≤ tc.knights ∧ tc.knights ≤ 20

/-- The rule that the sum of a knight's portion and a courtier's portion equals the king's portion -/
def satisfies_portion_rule (tc : TableCounts) : Prop :=
  (1 : ℚ) / tc.courtiers + (1 : ℚ) / tc.knights = (1 : ℚ) / tc.king

/-- The main theorem stating the maximum number of knights and corresponding courtiers -/
theorem max_knights_and_courtiers :
  ∃ (tc : TableCounts), 
    is_valid_table_counts tc ∧ 
    satisfies_portion_rule tc ∧
    tc.knights = 14 ∧ 
    tc.courtiers = 14 ∧
    (∀ (tc' : TableCounts), 
      is_valid_table_counts tc' ∧ 
      satisfies_portion_rule tc' → 
      tc'.knights ≤ tc.knights) :=
by sorry

end NUMINAMATH_CALUDE_max_knights_and_courtiers_l3127_312790


namespace NUMINAMATH_CALUDE_sin_equality_proof_l3127_312705

theorem sin_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 → 
  Real.sin (n * π / 180) = Real.sin (670 * π / 180) → 
  n = -50 := by sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l3127_312705


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_angles_l3127_312733

theorem isosceles_trapezoid_angles (a d : ℝ) : 
  -- The trapezoid is isosceles and angles form an arithmetic sequence
  a > 0 ∧ d > 0 ∧ 
  -- The sum of angles in a quadrilateral is 360°
  a + (a + d) + (a + 2*d) + 140 = 360 ∧ 
  -- The largest angle is 140°
  a + 3*d = 140 → 
  -- The smallest angle is 40°
  a = 40 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_angles_l3127_312733


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l3127_312788

theorem largest_multiple_of_8_under_100 : ∃ n : ℕ, n * 8 = 96 ∧ 
  (∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l3127_312788


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3127_312783

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem tenth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 5/6) (h₃ : a₃ = 7/6) :
  arithmetic_sequence a₁ (a₂ - a₁) 10 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3127_312783


namespace NUMINAMATH_CALUDE_average_speed_theorem_l3127_312714

def speed_1 : ℝ := 100
def speed_2 : ℝ := 80
def speed_3_4 : ℝ := 90
def speed_5 : ℝ := 60
def speed_6 : ℝ := 70

def duration_1 : ℝ := 1
def duration_2 : ℝ := 1
def duration_3_4 : ℝ := 2
def duration_5 : ℝ := 1
def duration_6 : ℝ := 1

def total_distance : ℝ := 
  speed_1 * duration_1 + 
  speed_2 * duration_2 + 
  speed_3_4 * duration_3_4 + 
  speed_5 * duration_5 + 
  speed_6 * duration_6

def total_time : ℝ := 
  duration_1 + duration_2 + duration_3_4 + duration_5 + duration_6

theorem average_speed_theorem : 
  total_distance / total_time = 490 / 6 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_theorem_l3127_312714


namespace NUMINAMATH_CALUDE_joe_age_l3127_312784

/-- Given that Joe has a daughter Jane, and their ages satisfy certain conditions,
    prove that Joe's age is 38. -/
theorem joe_age (joe_age jane_age : ℕ) 
  (sum_ages : joe_age + jane_age = 54)
  (diff_ages : joe_age - jane_age = 22) : 
  joe_age = 38 := by
sorry

end NUMINAMATH_CALUDE_joe_age_l3127_312784


namespace NUMINAMATH_CALUDE_function_properties_l3127_312748

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 3 * Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : φ ∈ Set.Ioo (-π) 0)
  (h2 : ∀ x, f x φ = f (π/4 - x) φ) :
  φ = -3*π/4 ∧
  (∀ k : ℤ, ∀ x : ℝ, 5*π/8 + k*π ≤ x ∧ x ≤ 9*π/8 + k*π → 
    ∀ y : ℝ, x < y → f y φ < f x φ) ∧
  Set.range (fun x => f x φ) = Set.Icc (-3) (3*Real.sqrt 2/2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3127_312748


namespace NUMINAMATH_CALUDE_price_change_equivalence_l3127_312761

theorem price_change_equivalence :
  let initial_increase := 0.40
  let subsequent_decrease := 0.15
  let equivalent_single_increase := 0.19
  ∀ (original_price : ℝ),
    original_price > 0 →
    original_price * (1 + initial_increase) * (1 - subsequent_decrease) =
    original_price * (1 + equivalent_single_increase) := by
  sorry

end NUMINAMATH_CALUDE_price_change_equivalence_l3127_312761


namespace NUMINAMATH_CALUDE_train_length_proof_l3127_312728

/-- Proves that given a train moving at 55 km/hr and a man moving at 7 km/hr in the opposite direction,
    if it takes 10.45077684107852 seconds for the train to pass the man, then the length of the train is 180 meters. -/
theorem train_length_proof (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 55 →
  man_speed = 7 →
  passing_time = 10.45077684107852 →
  (train_speed + man_speed) * (5 / 18) * passing_time = 180 := by
sorry

end NUMINAMATH_CALUDE_train_length_proof_l3127_312728


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3127_312775

theorem simplify_and_evaluate_expression (x y : ℚ) 
  (hx : x = -1) (hy : y = 1/5) : 
  2 * (x^2 * y - 2 * x * y) - 3 * (x^2 * y - 3 * x * y) + x^2 * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3127_312775


namespace NUMINAMATH_CALUDE_television_regular_price_l3127_312752

theorem television_regular_price (sale_price : ℝ) (discount_rate : ℝ) (regular_price : ℝ) :
  sale_price = regular_price * (1 - discount_rate) →
  discount_rate = 0.2 →
  sale_price = 480 →
  regular_price = 600 := by
sorry

end NUMINAMATH_CALUDE_television_regular_price_l3127_312752


namespace NUMINAMATH_CALUDE_number_ratio_l3127_312702

theorem number_ratio (x : ℚ) (h : 3 * (2 * x + 9) = 81) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l3127_312702


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3127_312745

/-- Represents a repeating decimal with a single digit repeating. -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- The sum of specific repeating decimals equals 2/3 -/
theorem repeating_decimal_sum :
  RepeatingDecimal 7 + RepeatingDecimal 5 - RepeatingDecimal 6 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3127_312745


namespace NUMINAMATH_CALUDE_christopher_age_l3127_312794

theorem christopher_age (c g : ℕ) : 
  c = 2 * g ∧ 
  c - 9 = 5 * (g - 9) → 
  c = 24 := by
sorry

end NUMINAMATH_CALUDE_christopher_age_l3127_312794


namespace NUMINAMATH_CALUDE_first_pipe_rate_correct_l3127_312718

/-- The rate at which the first pipe pumps water (in gallons per hour) -/
def first_pipe_rate : ℝ := 48

/-- The rate at which the second pipe pumps water (in gallons per hour) -/
def second_pipe_rate : ℝ := 192

/-- The capacity of the well in gallons -/
def well_capacity : ℝ := 1200

/-- The time it takes to fill the well in hours -/
def fill_time : ℝ := 5

theorem first_pipe_rate_correct : 
  first_pipe_rate * fill_time + second_pipe_rate * fill_time = well_capacity := by
  sorry

end NUMINAMATH_CALUDE_first_pipe_rate_correct_l3127_312718


namespace NUMINAMATH_CALUDE_meeting_point_coordinates_l3127_312792

/-- The point that divides a line segment in a given ratio -/
def dividing_point (x₁ y₁ x₂ y₂ : ℚ) (m n : ℚ) : ℚ × ℚ :=
  ((m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n))

/-- Proof that the point dividing the line segment from (2, 5) to (10, 1) 
    in the ratio 1:3 starting from (2, 5) has coordinates (4, 4) -/
theorem meeting_point_coordinates : 
  dividing_point 2 5 10 1 1 3 = (4, 4) := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_coordinates_l3127_312792


namespace NUMINAMATH_CALUDE_managers_salary_l3127_312751

/-- Proves that the manager's salary is 11500 given the conditions of the problem -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) :
  num_employees = 24 →
  avg_salary = 1500 →
  salary_increase = 400 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase : ℕ) = 11500 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l3127_312751


namespace NUMINAMATH_CALUDE_vacation_pictures_remaining_l3127_312763

def zoo_pictures : ℕ := 15
def museum_pictures : ℕ := 18
def deleted_pictures : ℕ := 31

theorem vacation_pictures_remaining :
  zoo_pictures + museum_pictures - deleted_pictures = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_remaining_l3127_312763


namespace NUMINAMATH_CALUDE_infinitely_many_fixed_points_l3127_312760

def is_cyclic (f : ℕ → ℕ) : Prop :=
  ∀ n, ∃ k, k > 0 ∧ (f^[k] n = n)

theorem infinitely_many_fixed_points
  (f : ℕ → ℕ)
  (h1 : ∀ n, f n - n < 2021)
  (h2 : is_cyclic f) :
  ∀ m, ∃ n > m, f n = n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_fixed_points_l3127_312760


namespace NUMINAMATH_CALUDE_pet_store_animals_l3127_312759

/-- Calculates the total number of animals in a pet store given the number of dogs and ratios for other animals. -/
def total_animals (num_dogs : ℕ) : ℕ :=
  let num_cats := num_dogs / 2
  let num_birds := num_dogs * 2
  let num_fish := num_dogs * 3
  num_dogs + num_cats + num_birds + num_fish

/-- Theorem stating that a pet store with 6 dogs and specified ratios of other animals has 39 animals in total. -/
theorem pet_store_animals : total_animals 6 = 39 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_animals_l3127_312759


namespace NUMINAMATH_CALUDE_fault_line_current_movement_l3127_312708

/-- Represents the movement of a fault line over two years -/
structure FaultLineMovement where
  total : ℝ  -- Total movement over two years
  previous : ℝ  -- Movement in the previous year
  current : ℝ  -- Movement in the current year

/-- Theorem stating the movement of the fault line in the current year -/
theorem fault_line_current_movement (f : FaultLineMovement)
  (h1 : f.total = 6.5)
  (h2 : f.previous = 5.25)
  (h3 : f.total = f.previous + f.current) :
  f.current = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_current_movement_l3127_312708


namespace NUMINAMATH_CALUDE_fraction_simplifiable_l3127_312785

theorem fraction_simplifiable (e : ℤ) : 
  (∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ (16 * e - 10) * b = (10 * e - 3) * a) ↔ 
  (∃ (k : ℤ), e = 13 * k + 12) := by
sorry

end NUMINAMATH_CALUDE_fraction_simplifiable_l3127_312785


namespace NUMINAMATH_CALUDE_sum_243_81_base3_l3127_312768

/-- Converts a natural number to its base 3 representation as a list of digits -/
def toBase3 (n : ℕ) : List ℕ := sorry

/-- Adds two numbers represented in base 3 -/
def addBase3 (a b : List ℕ) : List ℕ := sorry

/-- Checks if a list of digits is a valid base 3 representation -/
def isValidBase3 (l : List ℕ) : Prop := sorry

theorem sum_243_81_base3 :
  let a := toBase3 243
  let b := toBase3 81
  let sum := addBase3 a b
  isValidBase3 a ∧ isValidBase3 b ∧ isValidBase3 sum ∧ sum = [0, 0, 0, 0, 1, 1] := by sorry

end NUMINAMATH_CALUDE_sum_243_81_base3_l3127_312768


namespace NUMINAMATH_CALUDE_martin_family_ice_cream_l3127_312747

/-- The cost of ice cream for the Martin family at the mall --/
def ice_cream_cost (double_scoop_price : ℕ) : Prop :=
  let kiddie_scoop_price : ℕ := 3
  let regular_scoop_price : ℕ := 4
  let num_regular_scoops : ℕ := 2  -- Mr. and Mrs. Martin
  let num_kiddie_scoops : ℕ := 2   -- Two children
  let num_double_scoops : ℕ := 3   -- Three teenage children
  let total_cost : ℕ := 32
  (num_regular_scoops * regular_scoop_price +
   num_kiddie_scoops * kiddie_scoop_price +
   num_double_scoops * double_scoop_price) = total_cost

theorem martin_family_ice_cream : ice_cream_cost 6 := by
  sorry

end NUMINAMATH_CALUDE_martin_family_ice_cream_l3127_312747


namespace NUMINAMATH_CALUDE_circle_centers_and_m_l3127_312766

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0
def circle_C2 (x y m : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + m = 0

-- Define external tangency
def externally_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧ 
  ∀ (x' y' : ℝ), (C1 x' y' → (x' - x)^2 + (y' - y)^2 > 0) ∧
                 (C2 x' y' → (x' - x)^2 + (y' - y)^2 > 0)

-- Theorem statement
theorem circle_centers_and_m :
  externally_tangent circle_C1 (circle_C2 · · (-3)) →
  (∃ (x y : ℝ), circle_C1 x y ∧ x = -1 ∧ y = -1) ∧
  (∀ m : ℝ, externally_tangent circle_C1 (circle_C2 · · m) → m = -3) :=
sorry

end NUMINAMATH_CALUDE_circle_centers_and_m_l3127_312766


namespace NUMINAMATH_CALUDE_square_tile_area_l3127_312786

theorem square_tile_area (side_length : ℝ) (h : side_length = 7) :
  side_length * side_length = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_tile_area_l3127_312786


namespace NUMINAMATH_CALUDE_gina_payment_is_90_l3127_312750

/-- Calculates the total payment for Gina's order given her painting rates and order details. -/
def total_payment (rose_rate : ℕ) (lily_rate : ℕ) (rose_order : ℕ) (lily_order : ℕ) (hourly_rate : ℕ) : ℕ :=
  let rose_time := rose_order / rose_rate
  let lily_time := lily_order / lily_rate
  let total_time := rose_time + lily_time
  total_time * hourly_rate

/-- Proves that Gina's total payment for the given order is $90. -/
theorem gina_payment_is_90 : total_payment 6 7 6 14 30 = 90 := by
  sorry

#eval total_payment 6 7 6 14 30

end NUMINAMATH_CALUDE_gina_payment_is_90_l3127_312750


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l3127_312798

-- Part 1
def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a :
  (∀ x, ¬(q x a) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

-- Part 2
def s (m : ℝ) : Prop :=
  ∃ x y, x^2 + (m - 3) * x + m = 0 ∧
         y^2 + (m - 3) * y + m = 0 ∧
         0 < x ∧ x < 1 ∧ 2 < y ∧ y < 3

def t (m : ℝ) : Prop :=
  ∀ x, m * x^2 - 2 * x + 1 > 0

theorem range_of_m :
  s m ∨ t m →
  (0 < m ∧ m < 2/3) ∨ m > 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l3127_312798


namespace NUMINAMATH_CALUDE_john_drive_distance_l3127_312767

-- Define the constants
def speed : ℝ := 55
def time_before_lunch : ℝ := 2
def time_after_lunch : ℝ := 3

-- Define the total distance function
def total_distance (s t1 t2 : ℝ) : ℝ := s * (t1 + t2)

-- Theorem statement
theorem john_drive_distance :
  total_distance speed time_before_lunch time_after_lunch = 275 := by
  sorry

end NUMINAMATH_CALUDE_john_drive_distance_l3127_312767


namespace NUMINAMATH_CALUDE_balloon_count_is_22_l3127_312764

/-- The number of balloons each person brought to the park -/
structure BalloonCount where
  allan : ℕ
  jake : ℕ
  maria : ℕ
  tom_initial : ℕ
  tom_lost : ℕ

/-- The total number of balloons in the park -/
def total_balloons (bc : BalloonCount) : ℕ :=
  bc.allan + bc.jake + bc.maria + (bc.tom_initial - bc.tom_lost)

/-- Theorem: The total number of balloons in the park is 22 -/
theorem balloon_count_is_22 (bc : BalloonCount) 
    (h1 : bc.allan = 5)
    (h2 : bc.jake = 7)
    (h3 : bc.maria = 3)
    (h4 : bc.tom_initial = 9)
    (h5 : bc.tom_lost = 2) : 
  total_balloons bc = 22 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_is_22_l3127_312764


namespace NUMINAMATH_CALUDE_rhombus_area_l3127_312731

/-- The area of a rhombus with vertices at (0, 3.5), (11, 0), (0, -3.5), and (-11, 0) is 77 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (11, 0), (0, -3.5), (-11, 0)]
  let vertical_diagonal : ℝ := |3.5 - (-3.5)|
  let horizontal_diagonal : ℝ := |11 - (-11)|
  let area : ℝ := (vertical_diagonal * horizontal_diagonal) / 2
  area = 77 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l3127_312731


namespace NUMINAMATH_CALUDE_max_value_of_f_l3127_312723

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-1) 4 → f x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3127_312723


namespace NUMINAMATH_CALUDE_smallest_consecutive_number_l3127_312793

theorem smallest_consecutive_number (a b c d : ℕ) : 
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ a * b * c * d = 4574880 → a = 43 :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_number_l3127_312793


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3127_312709

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) :
  ∀ a b : ℝ, a > 0 → b > 0 → (a + 1) * (b + 1) = 9 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + 1) * (y + 1) = 9 ∧ x + y = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3127_312709


namespace NUMINAMATH_CALUDE_adjacent_chair_subsets_theorem_l3127_312719

/-- The number of subsets containing at least three adjacent chairs in a circular arrangement of 12 chairs -/
def adjacent_chair_subsets : ℕ := 1634

/-- The number of chairs arranged in a circle -/
def num_chairs : ℕ := 12

/-- A function that calculates the number of subsets containing at least three adjacent chairs -/
def calculate_subsets (n : ℕ) : ℕ := sorry

theorem adjacent_chair_subsets_theorem :
  calculate_subsets num_chairs = adjacent_chair_subsets :=
by sorry

end NUMINAMATH_CALUDE_adjacent_chair_subsets_theorem_l3127_312719


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l3127_312755

def f (x : ℝ) := 1 + x - x^2

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2) 4, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2) 4, f x = max) ∧
    (∀ x ∈ Set.Icc (-2) 4, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2) 4, f x = min) ∧
    max = 5/4 ∧ min = -11 :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l3127_312755


namespace NUMINAMATH_CALUDE_odot_examples_l3127_312773

def odot (a b : ℚ) : ℚ := a * (a + b) - 1

theorem odot_examples :
  (odot 3 (-2) = 2) ∧ (odot (-2) (odot 3 5) = -43) := by
  sorry

end NUMINAMATH_CALUDE_odot_examples_l3127_312773


namespace NUMINAMATH_CALUDE_football_yardage_l3127_312734

theorem football_yardage (total_yardage running_yardage : ℕ) 
  (h1 : total_yardage = 150)
  (h2 : running_yardage = 90) :
  total_yardage - running_yardage = 60 := by
  sorry

end NUMINAMATH_CALUDE_football_yardage_l3127_312734


namespace NUMINAMATH_CALUDE_license_plate_theorem_l3127_312716

def letter_count : Nat := 26
def digit_count : Nat := 10
def letter_positions : Nat := 5
def digit_positions : Nat := 3

def license_plate_combinations : Nat :=
  letter_count * (Nat.choose (letter_count - 1) (letter_positions - 2)) *
  (Nat.choose letter_positions 2) * (Nat.factorial (letter_positions - 2)) *
  digit_count * (digit_count - 1) * (digit_count - 2)

theorem license_plate_theorem :
  license_plate_combinations = 2594880000 := by sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l3127_312716


namespace NUMINAMATH_CALUDE_reflection_x_axis_l3127_312736

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The reflection of (-2, -3) across the x-axis is (-2, 3) -/
theorem reflection_x_axis : reflect_x (-2, -3) = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_reflection_x_axis_l3127_312736


namespace NUMINAMATH_CALUDE_fold_point_area_l3127_312749

/-- Definition of a triangle ABC --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Definition of a fold point --/
def FoldPoint (t : Triangle) (P : ℝ × ℝ) : Prop :=
  sorry  -- Definition of fold point

/-- Set of all fold points of a triangle --/
def FoldPointSet (t : Triangle) : Set (ℝ × ℝ) :=
  {P | FoldPoint t P}

/-- Area of a set in ℝ² --/
noncomputable def Area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry  -- Definition of area

theorem fold_point_area (t : Triangle) : 
  t.A.1 = 0 ∧ t.A.2 = 0 ∧
  t.B.1 = 36 ∧ t.B.2 = 0 ∧
  t.C.1 = 0 ∧ t.C.2 = 72 →
  Area (FoldPointSet t) = 270 * Real.pi - 324 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_fold_point_area_l3127_312749


namespace NUMINAMATH_CALUDE_path_area_calculation_l3127_312704

/-- Calculates the area of a path surrounding a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

theorem path_area_calculation (field_length field_width path_width : ℝ)
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 3.5) :
  path_area field_length field_width path_width = 959 := by
  sorry

#eval path_area 75 55 3.5

end NUMINAMATH_CALUDE_path_area_calculation_l3127_312704


namespace NUMINAMATH_CALUDE_solution_set_length_l3127_312737

theorem solution_set_length (a : ℝ) (h1 : a > 0) : 
  (∃ x1 x2 : ℝ, x1 < x2 ∧ 
    (∀ x : ℝ, x1 ≤ x ∧ x ≤ x2 ↔ Real.sqrt (x + a) + Real.sqrt (x - a) ≤ Real.sqrt (2 * (x + 1))) ∧
    x2 - x1 = 1/2) →
  a = 3/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_length_l3127_312737


namespace NUMINAMATH_CALUDE_constant_function_proof_l3127_312703

theorem constant_function_proof (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 + x) = 2 - f x) 
  (h2 : ∀ x, f (x + 3) ≥ f x) : 
  ∀ x, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l3127_312703


namespace NUMINAMATH_CALUDE_jiAnWinningCases_l3127_312778

/-- Represents the possible moves in rock-paper-scissors -/
inductive Move
  | Rock
  | Paper
  | Scissors

/-- Determines if the first move wins against the second move -/
def wins (m1 m2 : Move) : Bool :=
  match m1, m2 with
  | Move.Rock, Move.Scissors => true
  | Move.Paper, Move.Rock => true
  | Move.Scissors, Move.Paper => true
  | _, _ => false

/-- Counts the number of winning cases for the first player -/
def countWinningCases : Nat :=
  List.length (List.filter
    (fun (m1, m2) => wins m1 m2)
    [(Move.Rock, Move.Paper), (Move.Rock, Move.Scissors), (Move.Rock, Move.Rock),
     (Move.Paper, Move.Rock), (Move.Paper, Move.Scissors), (Move.Paper, Move.Paper),
     (Move.Scissors, Move.Rock), (Move.Scissors, Move.Paper), (Move.Scissors, Move.Scissors)])

theorem jiAnWinningCases :
  countWinningCases = 3 := by sorry

end NUMINAMATH_CALUDE_jiAnWinningCases_l3127_312778


namespace NUMINAMATH_CALUDE_min_triple_sum_bound_l3127_312724

def circle_arrangement (n : ℕ) := Fin n → ℕ

theorem min_triple_sum_bound (arr : circle_arrangement 10) :
  ∀ i : Fin 10, arr i ∈ Finset.range 11 →
  (∀ i j : Fin 10, i ≠ j → arr i ≠ arr j) →
  ∃ i : Fin 10, arr i + arr ((i + 1) % 10) + arr ((i + 2) % 10) ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_min_triple_sum_bound_l3127_312724


namespace NUMINAMATH_CALUDE_ashley_receives_22_50_l3127_312744

/-- The amount Ashley receives for her state quarters -/
def ashley_receives (num_quarters : ℕ) (face_value : ℚ) (collector_percentage : ℕ) : ℚ :=
  (num_quarters : ℚ) * face_value * (collector_percentage : ℚ) / 100

/-- Proof that Ashley receives $22.50 for her six state quarters -/
theorem ashley_receives_22_50 :
  ashley_receives 6 0.25 1500 = 22.50 := by
  sorry

end NUMINAMATH_CALUDE_ashley_receives_22_50_l3127_312744


namespace NUMINAMATH_CALUDE_student_marks_calculation_l3127_312735

theorem student_marks_calculation 
  (max_marks : ℕ) 
  (passing_percentage : ℚ) 
  (fail_margin : ℕ) 
  (h1 : max_marks = 400)
  (h2 : passing_percentage = 36 / 100)
  (h3 : fail_margin = 14) :
  ∃ (student_marks : ℕ), 
    student_marks = max_marks * passing_percentage - fail_margin ∧
    student_marks = 130 :=
by sorry

end NUMINAMATH_CALUDE_student_marks_calculation_l3127_312735


namespace NUMINAMATH_CALUDE_vip_seat_cost_l3127_312781

theorem vip_seat_cost (total_tickets : ℕ) (total_revenue : ℕ) (general_price : ℕ) (ticket_difference : ℕ) :
  total_tickets = 320 →
  total_revenue = 7500 →
  general_price = 20 →
  ticket_difference = 276 →
  ∃ vip_price : ℕ,
    vip_price = 70 ∧
    (total_tickets - ticket_difference) * general_price + ticket_difference * vip_price = total_revenue :=
by
  sorry

#check vip_seat_cost

end NUMINAMATH_CALUDE_vip_seat_cost_l3127_312781


namespace NUMINAMATH_CALUDE_optimal_vegetable_transport_plan_l3127_312754

/-- Represents the capacity and rental cost of a truck type -/
structure TruckType where
  capacity : ℕ
  rentalCost : ℕ

/-- The problem setup -/
def vegetableTransportProblem (typeA typeB : TruckType) : Prop :=
  -- Conditions
  2 * typeA.capacity + typeB.capacity = 10 ∧
  typeA.capacity + 2 * typeB.capacity = 11 ∧
  -- Define a function to calculate the total capacity of a plan
  (λ (x y : ℕ) => x * typeA.capacity + y * typeB.capacity) = 
    (λ (x y : ℕ) => 31) ∧
  -- Define a function to calculate the total cost of a plan
  (λ (x y : ℕ) => x * typeA.rentalCost + y * typeB.rentalCost) = 
    (λ (x y : ℕ) => 940) ∧
  -- The optimal plan
  (1 : ℕ) * typeA.capacity + (7 : ℕ) * typeB.capacity = 31

/-- The theorem to prove -/
theorem optimal_vegetable_transport_plan :
  ∃ (typeA typeB : TruckType),
    vegetableTransportProblem typeA typeB ∧
    typeA.rentalCost = 100 ∧
    typeB.rentalCost = 120 :=
  sorry


end NUMINAMATH_CALUDE_optimal_vegetable_transport_plan_l3127_312754


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3127_312753

theorem rectangle_area_diagonal (l w d : ℝ) (h_ratio : l / w = 5 / 2) (h_diag : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3127_312753


namespace NUMINAMATH_CALUDE_first_loan_amount_l3127_312706

/-- Represents a student loan -/
structure Loan where
  amount : ℝ
  rate : ℝ

/-- Calculates the interest paid on a loan -/
def interest_paid (loan : Loan) : ℝ :=
  loan.amount * loan.rate

theorem first_loan_amount
  (loan1 loan2 : Loan)
  (h1 : loan2.rate = 0.09)
  (h2 : loan1.amount = loan2.amount + 1500)
  (h3 : interest_paid loan1 + interest_paid loan2 = 617)
  (h4 : loan2.amount = 4700) :
  loan1.amount = 6200 := by
  sorry

end NUMINAMATH_CALUDE_first_loan_amount_l3127_312706


namespace NUMINAMATH_CALUDE_sum_of_coordinates_l3127_312710

/-- Given a function g where g(4) = 8, and h defined as h(x) = (g(x))^2 + 1,
    prove that the sum of coordinates of the point (4, h(4)) is 69. -/
theorem sum_of_coordinates (g : ℝ → ℝ) (h : ℝ → ℝ) : 
  g 4 = 8 → 
  (∀ x, h x = (g x)^2 + 1) → 
  4 + h 4 = 69 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_l3127_312710


namespace NUMINAMATH_CALUDE_average_age_combined_l3127_312770

/-- The average age of a combined group of fifth-graders and parents -/
theorem average_age_combined (num_fifth_graders : ℕ) (num_parents : ℕ) 
  (avg_age_fifth_graders : ℚ) (avg_age_parents : ℚ) :
  num_fifth_graders = 40 →
  num_parents = 50 →
  avg_age_fifth_graders = 10 →
  avg_age_parents = 35 →
  (num_fifth_graders * avg_age_fifth_graders + num_parents * avg_age_parents) / 
  (num_fifth_graders + num_parents : ℚ) = 215 / 9 := by
sorry

end NUMINAMATH_CALUDE_average_age_combined_l3127_312770


namespace NUMINAMATH_CALUDE_students_not_playing_sports_l3127_312782

theorem students_not_playing_sports (total_students football_players cricket_players both_players : ℕ) 
  (h1 : total_students = 420)
  (h2 : football_players = 325)
  (h3 : cricket_players = 175)
  (h4 : both_players = 130)
  (h5 : both_players ≤ football_players)
  (h6 : both_players ≤ cricket_players)
  (h7 : football_players ≤ total_students)
  (h8 : cricket_players ≤ total_students) :
  total_students - (football_players + cricket_players - both_players) = 50 := by
sorry

end NUMINAMATH_CALUDE_students_not_playing_sports_l3127_312782


namespace NUMINAMATH_CALUDE_estate_distribution_valid_l3127_312791

/-- Represents the estate distribution problem with twins --/
structure EstateDistribution :=
  (total : ℚ)
  (son_share : ℚ)
  (daughter_share : ℚ)
  (mother_share : ℚ)

/-- Checks if the distribution is valid according to the will's conditions --/
def is_valid_distribution (d : EstateDistribution) : Prop :=
  d.total = 210 ∧
  d.son_share + d.daughter_share + d.mother_share = d.total ∧
  d.son_share = (2/3) * d.total ∧
  d.daughter_share = (1/2) * d.mother_share

/-- Theorem stating that the given distribution is valid --/
theorem estate_distribution_valid :
  is_valid_distribution ⟨210, 140, 70/3, 140/3⟩ := by
  sorry

#check estate_distribution_valid

end NUMINAMATH_CALUDE_estate_distribution_valid_l3127_312791


namespace NUMINAMATH_CALUDE_sector_area_l3127_312739

theorem sector_area (r : ℝ) (θ : ℝ) (h : r = 2) (h' : θ = π / 4) :
  (1 / 2) * r^2 * θ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3127_312739


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l3127_312711

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ≥ 2, Monotone (f a)) → a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l3127_312711


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l3127_312717

theorem orange_juice_fraction : 
  let pitcher_capacity : ℚ := 600
  let pitcher1_fraction : ℚ := 1/3
  let pitcher2_fraction : ℚ := 2/5
  let orange_juice1 : ℚ := pitcher_capacity * pitcher1_fraction
  let orange_juice2 : ℚ := pitcher_capacity * pitcher2_fraction
  let total_orange_juice : ℚ := orange_juice1 + orange_juice2
  let total_mixture : ℚ := pitcher_capacity * 2
  total_orange_juice / total_mixture = 11/30 := by
sorry


end NUMINAMATH_CALUDE_orange_juice_fraction_l3127_312717


namespace NUMINAMATH_CALUDE_cube_less_than_triple_l3127_312796

theorem cube_less_than_triple : ∃! x : ℤ, x^3 < 3*x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_less_than_triple_l3127_312796


namespace NUMINAMATH_CALUDE_right_triangle_sets_l3127_312726

/-- A function that checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (6, 8, 11) cannot form a right-angled triangle --/
theorem right_triangle_sets :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 8 15 17 ∧
  is_right_triangle 7 24 25 ∧
  ¬(is_right_triangle 6 8 11) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l3127_312726


namespace NUMINAMATH_CALUDE_mary_promised_cards_l3127_312700

/-- The number of baseball cards Mary promised to give Fred -/
def promised_cards (initial : ℝ) (bought : ℝ) (left : ℝ) : ℝ :=
  initial + bought - left

theorem mary_promised_cards :
  promised_cards 18.0 40.0 32.0 = 26.0 := by
  sorry

end NUMINAMATH_CALUDE_mary_promised_cards_l3127_312700


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3127_312795

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 1 → (a - 1) * (b - 1) > 0) ∧
  ¬(∀ a b : ℝ, (a - 1) * (b - 1) > 0 → a > 1 ∧ b > 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3127_312795


namespace NUMINAMATH_CALUDE_dividend_calculation_l3127_312712

theorem dividend_calculation (quotient divisor remainder : ℕ) : 
  quotient = 15000 → 
  divisor = 82675 → 
  remainder = 57801 → 
  quotient * divisor + remainder = 1240182801 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3127_312712


namespace NUMINAMATH_CALUDE_trajectory_of_moving_point_l3127_312779

theorem trajectory_of_moving_point (x y : ℝ) :
  let segment_length : ℝ := 3
  let point_A : ℝ × ℝ := (3 * x, 0)
  let point_B : ℝ × ℝ := (0, 3 * y / 2)
  let point_C : ℝ × ℝ := (x, y)
  (point_A.1 - point_C.1)^2 + (point_A.2 - point_C.2)^2 = 4 * ((point_C.1 - point_B.1)^2 + (point_C.2 - point_B.2)^2) →
  x^2 + y^2 / 4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_point_l3127_312779
