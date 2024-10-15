import Mathlib

namespace NUMINAMATH_CALUDE_different_terminal_sides_l44_4492

-- Define a function to check if two angles have the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

-- Theorem statement
theorem different_terminal_sides :
  ¬ same_terminal_side 1050 (-300) :=
by
  sorry

end NUMINAMATH_CALUDE_different_terminal_sides_l44_4492


namespace NUMINAMATH_CALUDE_product_digits_sum_base7_l44_4485

/-- Converts a base-7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Sums the digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

theorem product_digits_sum_base7 :
  let a := 35
  let b := 42
  let product := toBase7 (toDecimal a * toDecimal b)
  sumDigitsBase7 product = 15
  := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_base7_l44_4485


namespace NUMINAMATH_CALUDE_regular_polygon_with_740_diagonals_has_40_sides_l44_4445

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 740 diagonals has 40 sides -/
theorem regular_polygon_with_740_diagonals_has_40_sides :
  ∃ (n : ℕ), n > 3 ∧ num_diagonals n = 740 ∧ n = 40 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_740_diagonals_has_40_sides_l44_4445


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l44_4420

theorem quadratic_rewrite_ratio : ∃ (c r s : ℚ),
  (∀ k, 8 * k^2 - 12 * k + 20 = c * (k + r)^2 + s) ∧
  s / r = -62 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l44_4420


namespace NUMINAMATH_CALUDE_small_semicircle_radius_l44_4448

/-- Given a large semicircle with radius 12, a circle with radius 6 inside it, 
    and a smaller semicircle, all pairwise tangent to each other, 
    the radius of the smaller semicircle is 4. -/
theorem small_semicircle_radius (r : ℝ) 
  (h1 : r > 0) -- radius of smaller semicircle is positive
  (h2 : 12 > 0) -- radius of larger semicircle is positive
  (h3 : 6 > 0)  -- radius of circle is positive
  (h4 : r < 12) -- radius of smaller semicircle is less than larger semicircle
  (h5 : r + 6 < 12) -- sum of radii of smaller semicircle and circle is less than larger semicircle
  : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_semicircle_radius_l44_4448


namespace NUMINAMATH_CALUDE_number_division_problem_l44_4497

theorem number_division_problem (x : ℝ) : x / 5 = 30 + x / 6 ↔ x = 900 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l44_4497


namespace NUMINAMATH_CALUDE_complex_quadrant_l44_4460

theorem complex_quadrant (z : ℂ) (h : (1 + Complex.I) * z = Complex.abs (1 + Complex.I)) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l44_4460


namespace NUMINAMATH_CALUDE_unique_prime_p_l44_4427

theorem unique_prime_p (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (5 * p^2 - 2)) : p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_p_l44_4427


namespace NUMINAMATH_CALUDE_mrs_susnas_grade_distribution_l44_4402

/-- Represents the fraction of students getting each grade in Mrs. Susna's class -/
structure GradeDistribution where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  f : ℚ
  passingGrade : ℚ

/-- The actual grade distribution in Mrs. Susna's class -/
def mrsSusnasClass : GradeDistribution where
  b := 1/2
  c := 1/8
  d := 1/12
  f := 1/24
  passingGrade := 7/8
  a := 0  -- We'll prove this value

theorem mrs_susnas_grade_distribution :
  let g := mrsSusnasClass
  g.a + g.b + g.c + g.d + g.f = 1 ∧
  g.a + g.b + g.c = g.passingGrade ∧
  g.a = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_mrs_susnas_grade_distribution_l44_4402


namespace NUMINAMATH_CALUDE_cookies_per_bag_l44_4464

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 2173) (h2 : num_bags = 53) :
  total_cookies / num_bags = 41 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l44_4464


namespace NUMINAMATH_CALUDE_equidistant_function_property_l44_4404

open Complex

theorem equidistant_function_property (a b : ℝ) :
  (∀ z : ℂ, abs ((a + b * I) * z - z) = abs ((a + b * I) * z - I)) →
  abs (a + b * I) = 10 →
  b^2 = (1/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_equidistant_function_property_l44_4404


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l44_4403

theorem polynomial_multiplication (x y : ℝ) :
  (3 * x^4 - 4 * y^3) * (9 * x^8 + 12 * x^4 * y^3 + 16 * y^6) = 27 * x^12 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l44_4403


namespace NUMINAMATH_CALUDE_reflection_changes_color_l44_4453

/-- Determines if a number is red (can be expressed as 81x + 100y for positive integers x and y) -/
def isRed (n : ℤ) : Prop :=
  ∃ x y : ℕ+, n = 81 * x + 100 * y

/-- The point P -/
def P : ℤ := 3960

/-- Reflects a point T with respect to P -/
def reflect (T : ℤ) : ℤ := 2 * P - T

theorem reflection_changes_color :
  ∀ T : ℤ, isRed T ≠ isRed (reflect T) :=
sorry

end NUMINAMATH_CALUDE_reflection_changes_color_l44_4453


namespace NUMINAMATH_CALUDE_inequality_holds_l44_4410

theorem inequality_holds (x y z : ℝ) : x^2 + y^2 + z^2 - x*y - x*z - y*z ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l44_4410


namespace NUMINAMATH_CALUDE_base4_10201_to_decimal_l44_4487

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_10201_to_decimal :
  base4_to_decimal [1, 0, 2, 0, 1] = 289 := by
  sorry

end NUMINAMATH_CALUDE_base4_10201_to_decimal_l44_4487


namespace NUMINAMATH_CALUDE_smallest_possible_d_value_l44_4465

theorem smallest_possible_d_value (d : ℝ) : 
  (2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2 = (4 * d) ^ 2 → 
  d ≥ (1 + 2 * Real.sqrt 10) / 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_value_l44_4465


namespace NUMINAMATH_CALUDE_triangle_area_l44_4498

/-- Given a triangle with perimeter 40 and inradius 2.5, prove its area is 50 -/
theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) : 
  P = 40 → r = 2.5 → A = r * (P / 2) → A = 50 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l44_4498


namespace NUMINAMATH_CALUDE_customer_difference_l44_4411

theorem customer_difference (X Y Z : ℕ) 
  (h1 : X - Y = 10) 
  (h2 : 10 - Z = 4) : 
  X - 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_customer_difference_l44_4411


namespace NUMINAMATH_CALUDE_runner_speed_ratio_l44_4483

theorem runner_speed_ratio :
  ∀ (v1 v2 : ℝ),
    v1 > v2 →
    v1 - v2 = 4 →
    v1 + v2 = 20 →
    v1 / v2 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_ratio_l44_4483


namespace NUMINAMATH_CALUDE_tournament_games_l44_4441

theorem tournament_games (x : ℕ) : 
  (3 / 4 : ℚ) * x + (1 / 4 : ℚ) * x = x ∧ 
  (2 / 3 : ℚ) * (x + 10) + (1 / 3 : ℚ) * (x + 10) = x + 10 ∧
  (2 / 3 : ℚ) * (x + 10) = (3 / 4 : ℚ) * x + 5 ∧
  (1 / 3 : ℚ) * (x + 10) = (1 / 4 : ℚ) * x + 5 →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_tournament_games_l44_4441


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l44_4437

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 * a 5 = a 3 * a 6 →
  a 9 * a 10 = -8 →
  a 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l44_4437


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_11_l44_4414

def reverse_digits (n : ℕ) : ℕ :=
  sorry

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_and_11 :
  ∃ m : ℕ,
    is_four_digit m ∧
    is_four_digit (reverse_digits m) ∧
    63 ∣ m ∧
    63 ∣ (reverse_digits m) ∧
    11 ∣ m ∧
    ∀ k : ℕ, (is_four_digit k ∧
              is_four_digit (reverse_digits k) ∧
              63 ∣ k ∧
              63 ∣ (reverse_digits k) ∧
              11 ∣ k) →
              k ≤ m ∧
    m = 9696 :=
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_11_l44_4414


namespace NUMINAMATH_CALUDE_circle_ratio_l44_4444

/-- Two circles touching externally -/
structure ExternallyTouchingCircles where
  R₁ : ℝ  -- Radius of the first circle
  R₂ : ℝ  -- Radius of the second circle
  h₁ : R₁ > 0
  h₂ : R₂ > 0

/-- Point of tangency between the circles -/
def pointOfTangency (c : ExternallyTouchingCircles) : ℝ := c.R₁ + c.R₂

/-- Distance from point of tangency to center of second circle -/
def tangentDistance (c : ExternallyTouchingCircles) : ℝ := 3 * c.R₂

theorem circle_ratio (c : ExternallyTouchingCircles) 
  (h : tangentDistance c = pointOfTangency c - c.R₁) : 
  c.R₁ = 4 * c.R₂ := by
  sorry

#check circle_ratio

end NUMINAMATH_CALUDE_circle_ratio_l44_4444


namespace NUMINAMATH_CALUDE_tan_420_degrees_l44_4431

theorem tan_420_degrees : Real.tan (420 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_420_degrees_l44_4431


namespace NUMINAMATH_CALUDE_tens_digit_of_23_pow_1987_l44_4447

theorem tens_digit_of_23_pow_1987 :
  (23^1987 / 10) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_pow_1987_l44_4447


namespace NUMINAMATH_CALUDE_factorization_of_x4_plus_256_l44_4429

theorem factorization_of_x4_plus_256 (x : ℝ) : 
  x^4 + 256 = (x^2 - 8*x + 32) * (x^2 + 8*x + 32) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x4_plus_256_l44_4429


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_is_417_l44_4478

/-- The sum of areas of six rectangles with width 3 and lengths (2², 3², 4², 5², 6², 7²) -/
def sum_of_rectangle_areas : ℕ :=
  let width := 3
  let lengths := [2, 3, 4, 5, 6, 7].map (λ x => x^2)
  (lengths.map (λ l => width * l)).sum

/-- Theorem stating that the sum of the areas is 417 -/
theorem sum_of_rectangle_areas_is_417 : sum_of_rectangle_areas = 417 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_is_417_l44_4478


namespace NUMINAMATH_CALUDE_sum_zero_inequality_l44_4493

theorem sum_zero_inequality (a b c d : ℝ) (h : a + b + c + d = 0) :
  (a*b + a*c + a*d + b*c + b*d + c*d)^2 + 12 ≥ 6*(a*b*c + a*b*d + a*c*d + b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_inequality_l44_4493


namespace NUMINAMATH_CALUDE_drummer_drum_stick_usage_l44_4446

/-- Calculates the total number of drum stick sets used by a drummer over multiple shows. -/
def total_drum_stick_sets (sets_per_show : ℕ) (tossed_sets : ℕ) (num_shows : ℕ) : ℕ :=
  (sets_per_show + tossed_sets) * num_shows

/-- Theorem stating that a drummer using 5 sets per show, tossing 6 sets, for 30 shows uses 330 sets in total. -/
theorem drummer_drum_stick_usage :
  total_drum_stick_sets 5 6 30 = 330 := by
  sorry

end NUMINAMATH_CALUDE_drummer_drum_stick_usage_l44_4446


namespace NUMINAMATH_CALUDE_merchant_salt_price_l44_4400

/-- Represents the price per pound of the unknown salt in cents -/
def unknown_price : ℝ := 50

/-- The weight of the unknown salt in pounds -/
def unknown_weight : ℝ := 20

/-- The weight of the known salt in pounds -/
def known_weight : ℝ := 40

/-- The price per pound of the known salt in cents -/
def known_price : ℝ := 35

/-- The selling price per pound of the mixture in cents -/
def selling_price : ℝ := 48

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.2

theorem merchant_salt_price :
  unknown_price = 50 ∧
  (unknown_price * unknown_weight + known_price * known_weight) * (1 + profit_percentage) =
    selling_price * (unknown_weight + known_weight) :=
by sorry

end NUMINAMATH_CALUDE_merchant_salt_price_l44_4400


namespace NUMINAMATH_CALUDE_c_share_is_40_l44_4418

/-- Represents the share distribution among three parties -/
structure ShareDistribution where
  total : ℝ
  b_share : ℝ
  c_share : ℝ
  d_share : ℝ

/-- The condition for the share distribution -/
def valid_distribution (s : ShareDistribution) : Prop :=
  s.total = 80 ∧
  s.c_share = 1.5 * s.b_share ∧
  s.d_share = 0.5 * s.b_share ∧
  s.total = s.b_share + s.c_share + s.d_share

/-- Theorem stating that under the given conditions, c's share is 40 rupees -/
theorem c_share_is_40 (s : ShareDistribution) (h : valid_distribution s) : s.c_share = 40 := by
  sorry

end NUMINAMATH_CALUDE_c_share_is_40_l44_4418


namespace NUMINAMATH_CALUDE_cookies_calculation_l44_4496

/-- The number of people Brenda's mother made cookies for -/
def num_people : ℕ := 25

/-- The number of cookies each person had -/
def cookies_per_person : ℕ := 45

/-- The total number of cookies Brenda's mother prepared -/
def total_cookies : ℕ := num_people * cookies_per_person

theorem cookies_calculation :
  total_cookies = 1125 :=
by sorry

end NUMINAMATH_CALUDE_cookies_calculation_l44_4496


namespace NUMINAMATH_CALUDE_layer_sum_2014_implies_digit_sum_13_l44_4442

/-- Represents a four-digit positive integer --/
structure FourDigitInt where
  w : Nat
  x : Nat
  y : Nat
  z : Nat
  w_nonzero : w ≠ 0
  w_upper_bound : w < 10
  x_upper_bound : x < 10
  y_upper_bound : y < 10
  z_upper_bound : z < 10

/-- Calculates the layer sum of a four-digit integer --/
def layerSum (n : FourDigitInt) : Nat :=
  1000 * n.w + 100 * n.x + 10 * n.y + n.z +
  100 * n.x + 10 * n.y + n.z +
  10 * n.y + n.z +
  n.z

/-- Main theorem --/
theorem layer_sum_2014_implies_digit_sum_13 (n : FourDigitInt) :
  layerSum n = 2014 → n.w + n.x + n.y + n.z = 13 := by
  sorry

end NUMINAMATH_CALUDE_layer_sum_2014_implies_digit_sum_13_l44_4442


namespace NUMINAMATH_CALUDE_grass_seed_min_cost_l44_4432

/-- Represents a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  price : Rat

/-- Finds the minimum cost to buy grass seed given the constraints -/
def minCostGrassSeed (bags : List GrassSeedBag) (minWeight maxWeight : Nat) : Rat :=
  sorry

/-- Theorem stating the minimum cost for the given problem -/
theorem grass_seed_min_cost :
  let bags : List GrassSeedBag := [
    { weight := 5, price := 138/10 },
    { weight := 10, price := 2043/100 },
    { weight := 25, price := 3225/100 }
  ]
  minCostGrassSeed bags 65 80 = 9675/100 := by sorry

end NUMINAMATH_CALUDE_grass_seed_min_cost_l44_4432


namespace NUMINAMATH_CALUDE_sum_bounds_l44_4451

def A : Set ℕ := {n | n ≤ 2018}

theorem sum_bounds (x y z : ℕ) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A) 
  (h : x^2 + y^2 - z^2 = 2019^2) : 
  2181 ≤ x + y + z ∧ x + y + z ≤ 5781 := by
  sorry

end NUMINAMATH_CALUDE_sum_bounds_l44_4451


namespace NUMINAMATH_CALUDE_quadratic_shift_properties_l44_4488

/-- Represents a quadratic function of the form y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Shifts a quadratic function up by a given amount -/
def shift_up (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { f with c := f.c + shift }

theorem quadratic_shift_properties (f : QuadraticFunction) :
  let f_shifted := shift_up f 3
  (f.a = f_shifted.a) ∧ 
  (-f.b / (2 * f.a) = -f_shifted.b / (2 * f_shifted.a)) ∧
  (f.c ≠ f_shifted.c) := by sorry

end NUMINAMATH_CALUDE_quadratic_shift_properties_l44_4488


namespace NUMINAMATH_CALUDE_max_triangles_correct_l44_4422

/-- The maximum number of triangles formed by drawing non-intersecting diagonals in a convex n-gon -/
def max_triangles (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 * n - 4 else 2 * n - 5

theorem max_triangles_correct (n : ℕ) (h : n ≥ 3) :
  max_triangles n = 
    (if n % 2 = 0 then 2 * n - 4 else 2 * n - 5) ∧
  ∀ k : ℕ, k ≤ max_triangles n :=
by sorry

end NUMINAMATH_CALUDE_max_triangles_correct_l44_4422


namespace NUMINAMATH_CALUDE_triangle_radii_inequality_l44_4479

/-- Given a triangle ABC with circumradius R, inradius r, distance from circumcenter to centroid e,
    and distance from incenter to centroid f, prove that R² - e² ≥ 4(r² - f²),
    with equality if and only if the triangle is equilateral. -/
theorem triangle_radii_inequality (R r e f : ℝ) (hR : R > 0) (hr : r > 0) (he : e ≥ 0) (hf : f ≥ 0) :
  R^2 - e^2 ≥ 4*(r^2 - f^2) ∧
  (R^2 - e^2 = 4*(r^2 - f^2) ↔ ∃ (s : ℝ), R = s ∧ r = s/3 ∧ e = s/3 ∧ f = s/6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_radii_inequality_l44_4479


namespace NUMINAMATH_CALUDE_inequality_not_true_l44_4415

theorem inequality_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / (a - b) > 1 / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_true_l44_4415


namespace NUMINAMATH_CALUDE_f_5_equals_207_l44_4480

def f (n : ℕ) : ℕ := n^3 + 2*n^2 + 3*n + 17

theorem f_5_equals_207 : f 5 = 207 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_207_l44_4480


namespace NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_for_beta_l44_4467

-- Define the propositions
def α (x : ℝ) : Prop := |x - 1| ≤ 2
def β (x : ℝ) : Prop := (x - 3) / (x + 1) ≤ 0

-- Theorem statement
theorem alpha_necessary_not_sufficient_for_beta :
  (∀ x, β x → α x) ∧ ¬(∀ x, α x → β x) := by sorry

end NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_for_beta_l44_4467


namespace NUMINAMATH_CALUDE_friends_bill_calculation_l44_4419

/-- Represents a restaurant order --/
structure Order where
  tacos : ℕ
  enchiladas : ℕ

/-- Represents the cost of an order --/
def cost (o : Order) (taco_price enchilada_price : ℚ) : ℚ :=
  o.tacos * taco_price + o.enchiladas * enchilada_price

theorem friends_bill_calculation (enchilada_price : ℚ) 
  (your_order friend_order : Order) (your_bill : ℚ) 
  (h1 : enchilada_price = 2)
  (h2 : your_order = ⟨2, 3⟩)
  (h3 : friend_order = ⟨3, 5⟩)
  (h4 : your_bill = 39/5) : 
  ∃ (taco_price : ℚ), cost friend_order taco_price enchilada_price = 127/10 := by
  sorry

#eval 127/10  -- Should output 12.7

end NUMINAMATH_CALUDE_friends_bill_calculation_l44_4419


namespace NUMINAMATH_CALUDE_downstream_speed_l44_4405

theorem downstream_speed (upstream_speed : ℝ) (average_speed : ℝ) (downstream_speed : ℝ) :
  upstream_speed = 6 →
  average_speed = 60 / 11 →
  (1 / upstream_speed + 1 / downstream_speed) / 2 = 1 / average_speed →
  downstream_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_downstream_speed_l44_4405


namespace NUMINAMATH_CALUDE_range_of_f_l44_4462

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l44_4462


namespace NUMINAMATH_CALUDE_integer_divisibility_l44_4473

theorem integer_divisibility (n : ℕ) (h : ∃ m : ℤ, (2^n - 2 : ℤ) = n * m) :
  ∃ k : ℤ, (2^(2^n - 1) - 2 : ℤ) = (2^n - 1) * k :=
sorry

end NUMINAMATH_CALUDE_integer_divisibility_l44_4473


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l44_4499

/-- Given a square and a circle intersecting such that each side of the square contains
    a chord of the circle with length equal to half the radius of the circle,
    the ratio of the area of the square to the area of the circle is 3/π. -/
theorem square_circle_area_ratio (r : ℝ) (h : r > 0) :
  let s := r * Real.sqrt 3
  (s^2) / (π * r^2) = 3 / π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l44_4499


namespace NUMINAMATH_CALUDE_multiple_of_17_l44_4430

theorem multiple_of_17 (x y : ℤ) : (2 * x + 3 * y) % 17 = 0 → (9 * x + 5 * y) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_17_l44_4430


namespace NUMINAMATH_CALUDE_max_volume_rect_prism_l44_4416

/-- A right prism with rectangular bases -/
structure RectPrism where
  a : ℝ  -- length of base
  b : ℝ  -- width of base
  h : ℝ  -- height of prism
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

/-- The sum of areas of three mutually adjacent faces is 48 -/
def adjacent_faces_area (p : RectPrism) : ℝ :=
  p.a * p.h + p.b * p.h + p.a * p.b

/-- The volume of the prism -/
def volume (p : RectPrism) : ℝ :=
  p.a * p.b * p.h

/-- The theorem stating the maximum volume of the prism -/
theorem max_volume_rect_prism :
  ∃ (p : RectPrism),
    adjacent_faces_area p = 48 ∧
    p.a = p.b ∧  -- two lateral faces are congruent
    ∀ (q : RectPrism),
      adjacent_faces_area q = 48 →
      q.a = q.b →
      volume q ≤ volume p ∧
      volume p = 64 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_rect_prism_l44_4416


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l44_4489

def arithmetic_sequence_sum (a l d : ℤ) : ℤ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_specific :
  arithmetic_sequence_sum (-45) 3 4 = -273 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l44_4489


namespace NUMINAMATH_CALUDE_treasure_chest_rubies_l44_4468

theorem treasure_chest_rubies (total_gems diamonds : ℕ) 
  (h1 : total_gems = 5155)
  (h2 : diamonds = 45) :
  total_gems - diamonds = 5110 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_rubies_l44_4468


namespace NUMINAMATH_CALUDE_rationalize_denominator_l44_4435

theorem rationalize_denominator : 35 / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l44_4435


namespace NUMINAMATH_CALUDE_age_difference_l44_4449

/-- Represents the ages of a mother and daughter pair -/
structure AgesPair where
  mother : ℕ
  daughter : ℕ

/-- Checks if the digits of the daughter's age are the reverse of the mother's age digits -/
def AgesPair.isReverse (ages : AgesPair) : Prop :=
  ages.daughter = ages.mother % 10 * 10 + ages.mother / 10

/-- Checks if in 13 years, the mother will be twice as old as the daughter -/
def AgesPair.futureCondition (ages : AgesPair) : Prop :=
  ages.mother + 13 = 2 * (ages.daughter + 13)

/-- The main theorem stating the age difference -/
theorem age_difference (ages : AgesPair) 
  (h1 : ages.isReverse)
  (h2 : ages.futureCondition) :
  ages.mother - ages.daughter = 40 := by
  sorry

/-- Example usage of the theorem -/
example : ∃ (ages : AgesPair), ages.isReverse ∧ ages.futureCondition ∧ ages.mother - ages.daughter = 40 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l44_4449


namespace NUMINAMATH_CALUDE_reflection_of_A_wrt_BC_l44_4491

/-- Reflection of a point with respect to a horizontal line -/
def reflect_point (p : ℝ × ℝ) (y : ℝ) : ℝ × ℝ :=
  (p.1, 2 * y - p.2)

theorem reflection_of_A_wrt_BC :
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (0, 1)
  let C : ℝ × ℝ := (3, 1)
  reflect_point A B.2 = (2, -1) := by
sorry

end NUMINAMATH_CALUDE_reflection_of_A_wrt_BC_l44_4491


namespace NUMINAMATH_CALUDE_triangle_area_l44_4470

/-- The area of a triangle ABC with given side lengths and angle -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 2 →
  c = 2 * Real.sqrt 2 →
  C = π / 4 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l44_4470


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l44_4455

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fourth_term_of_sequence (a : ℕ → ℝ) 
    (h1 : a 1 = 2)
    (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n) :
  a 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l44_4455


namespace NUMINAMATH_CALUDE_limit_of_sequence_l44_4486

def a (n : ℕ) : ℚ := (2 - 3 * n^2) / (4 + 5 * n^2)

theorem limit_of_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-3/5)| < ε := by sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l44_4486


namespace NUMINAMATH_CALUDE_cubic_function_max_l44_4439

/-- Given a cubic function with specific properties, prove its maximum value on [-3, 3] -/
theorem cubic_function_max (a b c : ℝ) : 
  (∀ x, (∃ y, y = a * x^3 + b * x + c)) →  -- f(x) = ax³ + bx + c
  (∃ y, y = 8 * a + 2 * b + c ∧ y = c - 16) →  -- f(2) = c - 16
  (3 * a * 2^2 + b = 0) →  -- f'(2) = 0 (extremum condition)
  (a = 1 ∧ b = -12) →  -- Values of a and b
  (∃ x, ∀ y, a * x^3 + b * x + c ≥ y ∧ a * x^3 + b * x + c = 28) →  -- Maximum value is 28
  (∃ x, x ∈ Set.Icc (-3) 3 ∧ 
    ∀ y ∈ Set.Icc (-3) 3, a * x^3 + b * x + c ≥ a * y^3 + b * y + c ∧ 
    a * x^3 + b * x + c = 28) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_max_l44_4439


namespace NUMINAMATH_CALUDE_clinton_meal_days_l44_4433

/-- The number of days Clinton buys a meal, given the base cost, up-size cost, and total spent. -/
def days_buying_meal (base_cost : ℚ) (upsize_cost : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent / (base_cost + upsize_cost)

/-- Theorem stating that Clinton buys the meal for 5 days. -/
theorem clinton_meal_days :
  let base_cost : ℚ := 6
  let upsize_cost : ℚ := 1
  let total_spent : ℚ := 35
  days_buying_meal base_cost upsize_cost total_spent = 5 := by
  sorry

end NUMINAMATH_CALUDE_clinton_meal_days_l44_4433


namespace NUMINAMATH_CALUDE_parallelepiped_arrangement_exists_l44_4421

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  -- Define the parallelepiped structure (simplified for this example)
  dummy : Unit

/-- Represents a point in 3D space -/
structure Point where
  -- Define the point structure (simplified for this example)
  dummy : Unit

/-- Checks if two parallelepipeds intersect -/
def intersects (p1 p2 : Parallelepiped) : Prop :=
  sorry

/-- Checks if a point is inside a parallelepiped -/
def isInside (point : Point) (p : Parallelepiped) : Prop :=
  sorry

/-- Checks if a vertex of a parallelepiped is visible from a point -/
def isVertexVisible (point : Point) (p : Parallelepiped) : Prop :=
  sorry

/-- Theorem stating the existence of the required arrangement -/
theorem parallelepiped_arrangement_exists : 
  ∃ (parallelepipeds : Fin 6 → Parallelepiped) (observationPoint : Point),
    (∀ i j : Fin 6, i ≠ j → ¬intersects (parallelepipeds i) (parallelepipeds j)) ∧
    (∀ i : Fin 6, ¬isInside observationPoint (parallelepipeds i)) ∧
    (∀ i : Fin 6, ¬isVertexVisible observationPoint (parallelepipeds i)) :=
  sorry

end NUMINAMATH_CALUDE_parallelepiped_arrangement_exists_l44_4421


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l44_4409

theorem fraction_inequality_solution_set (x : ℝ) :
  (x ≠ -1) → ((2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l44_4409


namespace NUMINAMATH_CALUDE_square_clock_area_l44_4459

-- Define the side length of the square clock
def clock_side_length : ℝ := 30

-- Define the area of the square clock
def clock_area : ℝ := clock_side_length * clock_side_length

-- Theorem to prove
theorem square_clock_area : clock_area = 900 := by
  sorry

end NUMINAMATH_CALUDE_square_clock_area_l44_4459


namespace NUMINAMATH_CALUDE_quarter_percentage_approx_l44_4425

/-- Represents the number and value of coins -/
structure Coins where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ
  dime_value : ℕ
  quarter_value : ℕ
  nickel_value : ℕ

/-- Calculates the percentage of quarters in the total value -/
def quarter_percentage (c : Coins) : ℚ :=
  let total_value := c.dimes * c.dime_value + c.quarters * c.quarter_value + c.nickels * c.nickel_value
  let quarter_value := c.quarters * c.quarter_value
  (quarter_value : ℚ) / (total_value : ℚ) * 100

/-- Theorem stating that the percentage of quarters is approximately 51.28% -/
theorem quarter_percentage_approx (c : Coins) 
  (h1 : c.dimes = 80) (h2 : c.quarters = 40) (h3 : c.nickels = 30)
  (h4 : c.dime_value = 10) (h5 : c.quarter_value = 25) (h6 : c.nickel_value = 5) : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 100 ∧ |quarter_percentage c - (5128 : ℚ) / 100| < ε := by
  sorry

end NUMINAMATH_CALUDE_quarter_percentage_approx_l44_4425


namespace NUMINAMATH_CALUDE_machine_operation_l44_4475

theorem machine_operation (x : ℤ) : 26 + x - 6 = 35 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_machine_operation_l44_4475


namespace NUMINAMATH_CALUDE_independent_x_implies_result_l44_4481

theorem independent_x_implies_result (m n : ℝ) :
  (∀ x y : ℝ, ∃ k : ℝ, (m*x^2 + 3*x - y) - (4*x^2 - (2*n + 3)*x + 3*y - 2) = k) →
  (m - n) + |m*n| = 19 := by
sorry

end NUMINAMATH_CALUDE_independent_x_implies_result_l44_4481


namespace NUMINAMATH_CALUDE_smallest_good_sequence_index_is_60_l44_4482

-- Define a good sequence
def GoodSequence (a : ℕ → ℝ) : Prop :=
  (∃ k : ℕ+, a 0 = k) ∧
  (∀ i : ℕ, (a (i + 1) = 2 * a i + 1) ∨ (a (i + 1) = a i / (a i + 2))) ∧
  (∃ k : ℕ+, a k = 2014)

-- Define the property we want to prove
def SmallestGoodSequenceIndex : Prop :=
  ∃ n : ℕ+, 
    (∃ a : ℕ → ℝ, GoodSequence a ∧ a n = 2014) ∧
    (∀ m : ℕ+, m < n → ¬∃ a : ℕ → ℝ, GoodSequence a ∧ a m = 2014)

-- The theorem to prove
theorem smallest_good_sequence_index_is_60 : 
  SmallestGoodSequenceIndex ∧ (∃ n : ℕ+, n = 60 ∧ ∃ a : ℕ → ℝ, GoodSequence a ∧ a n = 2014) :=
sorry

end NUMINAMATH_CALUDE_smallest_good_sequence_index_is_60_l44_4482


namespace NUMINAMATH_CALUDE_race_distance_proof_l44_4456

/-- Represents the total distance of a race in meters. -/
def race_distance : ℝ := 88

/-- Represents the time taken by Runner A to complete the race in seconds. -/
def time_A : ℝ := 20

/-- Represents the time taken by Runner B to complete the race in seconds. -/
def time_B : ℝ := 25

/-- Represents the distance by which Runner A beats Runner B in meters. -/
def beating_distance : ℝ := 22

theorem race_distance_proof : 
  race_distance = 88 ∧ 
  (race_distance / time_A) * time_B = race_distance + beating_distance :=
sorry

end NUMINAMATH_CALUDE_race_distance_proof_l44_4456


namespace NUMINAMATH_CALUDE_prime_divisor_of_mersenne_number_l44_4490

theorem prime_divisor_of_mersenne_number (p q : ℕ) : 
  Prime p → Prime q → q ∣ (2^p - 1) → p ∣ (q - 1) := by sorry

end NUMINAMATH_CALUDE_prime_divisor_of_mersenne_number_l44_4490


namespace NUMINAMATH_CALUDE_largest_common_value_less_than_1000_l44_4408

theorem largest_common_value_less_than_1000 :
  let seq1 := {a : ℕ | ∃ n : ℕ, a = 2 + 3 * n}
  let seq2 := {a : ℕ | ∃ m : ℕ, a = 4 + 8 * m}
  let common_values := seq1 ∩ seq2
  (∃ x ∈ common_values, x < 1000 ∧ ∀ y ∈ common_values, y < 1000 → y ≤ x) →
  (∃ x ∈ common_values, x = 980 ∧ ∀ y ∈ common_values, y < 1000 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_value_less_than_1000_l44_4408


namespace NUMINAMATH_CALUDE_sum_of_distances_constant_l44_4466

/-- A regular pyramid -/
structure RegularPyramid where
  base : Set (Fin 3 → ℝ)  -- Base of the pyramid as a set of points in ℝ³
  apex : Fin 3 → ℝ        -- Apex of the pyramid as a point in ℝ³
  is_regular : Bool       -- Property ensuring the pyramid is regular

/-- A point on the base of the pyramid -/
def BasePoint (pyramid : RegularPyramid) := { p : Fin 3 → ℝ // p ∈ pyramid.base }

/-- The perpendicular line from a point on the base to the base plane -/
def Perpendicular (pyramid : RegularPyramid) (p : BasePoint pyramid) : Set (Fin 3 → ℝ) :=
  sorry

/-- The intersection points of the perpendicular with the face planes -/
def IntersectionPoints (pyramid : RegularPyramid) (p : BasePoint pyramid) : Set (Fin 3 → ℝ) :=
  sorry

/-- The sum of distances from a base point to the intersection points -/
def SumOfDistances (pyramid : RegularPyramid) (p : BasePoint pyramid) : ℝ :=
  sorry

/-- Theorem: The sum of distances is constant for all points on the base -/
theorem sum_of_distances_constant (pyramid : RegularPyramid) :
  ∀ p q : BasePoint pyramid, SumOfDistances pyramid p = SumOfDistances pyramid q :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_constant_l44_4466


namespace NUMINAMATH_CALUDE_soccer_league_games_l44_4458

/-- Calculate the number of games in a soccer league --/
theorem soccer_league_games (n : ℕ) (h : n = 11) : n * (n - 1) / 2 = 55 := by
  sorry

#check soccer_league_games

end NUMINAMATH_CALUDE_soccer_league_games_l44_4458


namespace NUMINAMATH_CALUDE_equals_2022_l44_4423

theorem equals_2022 : 1 - (-2021) = 2022 := by
  sorry

end NUMINAMATH_CALUDE_equals_2022_l44_4423


namespace NUMINAMATH_CALUDE_halloween_goodie_bags_minimum_cost_l44_4440

/-- Represents the theme of a Halloween goodie bag -/
inductive Theme
| Vampire
| Pumpkin

/-- Represents the purchase options available -/
inductive PurchaseOption
| Package
| Individual

theorem halloween_goodie_bags_minimum_cost 
  (total_students : ℕ)
  (vampire_requests : ℕ)
  (pumpkin_requests : ℕ)
  (package_price : ℕ)
  (package_size : ℕ)
  (individual_price : ℕ)
  (discount_buy : ℕ)
  (discount_free : ℕ)
  (h1 : total_students = 25)
  (h2 : vampire_requests = 11)
  (h3 : pumpkin_requests = 14)
  (h4 : vampire_requests + pumpkin_requests = total_students)
  (h5 : package_price = 3)
  (h6 : package_size = 5)
  (h7 : individual_price = 1)
  (h8 : discount_buy = 3)
  (h9 : discount_free = 1) :
  (∃ (vampire_packages vampire_individuals pumpkin_packages : ℕ),
    vampire_packages * package_size + vampire_individuals ≥ vampire_requests ∧
    pumpkin_packages * package_size ≥ pumpkin_requests ∧
    (vampire_packages * package_price + vampire_individuals * individual_price +
     (pumpkin_packages / discount_buy * (discount_buy - discount_free) + pumpkin_packages % discount_buy) * package_price = 13)) :=
by sorry


end NUMINAMATH_CALUDE_halloween_goodie_bags_minimum_cost_l44_4440


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l44_4406

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m β) : 
  perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l44_4406


namespace NUMINAMATH_CALUDE_sphere_surface_area_l44_4413

theorem sphere_surface_area (r h : ℝ) (h1 : r = 1) (h2 : h = Real.sqrt 3) : 
  let R := (2 * Real.sqrt 3) / 3
  4 * π * R^2 = (16 * π) / 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l44_4413


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l44_4471

theorem complex_subtraction_simplification :
  (-5 - 3 * Complex.I) - (2 - 5 * Complex.I) = -7 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l44_4471


namespace NUMINAMATH_CALUDE_magic_king_seasons_l44_4436

theorem magic_king_seasons (total_episodes : ℕ) 
  (episodes_first_half : ℕ) (episodes_second_half : ℕ) :
  total_episodes = 225 ∧ 
  episodes_first_half = 20 ∧ 
  episodes_second_half = 25 →
  ∃ (seasons : ℕ), 
    seasons = 10 ∧
    total_episodes = (seasons / 2) * episodes_first_half + 
                     (seasons / 2) * episodes_second_half :=
by sorry

end NUMINAMATH_CALUDE_magic_king_seasons_l44_4436


namespace NUMINAMATH_CALUDE_sin_2theta_plus_pi_third_l44_4452

theorem sin_2theta_plus_pi_third (θ : Real) 
  (h1 : θ > π / 2) (h2 : θ < π) 
  (h3 : 1 / Real.sin θ + 1 / Real.cos θ = 2 * Real.sqrt 2) : 
  Real.sin (2 * θ + π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_plus_pi_third_l44_4452


namespace NUMINAMATH_CALUDE_wall_width_is_three_l44_4476

/-- Proves that a rectangular wall with given proportions and volume has a width of 3 meters -/
theorem wall_width_is_three (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  l = 7 * h →
  volume = w * h * l →
  volume = 6804 →
  w = 3 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_is_three_l44_4476


namespace NUMINAMATH_CALUDE_gcd_of_abcd_plus_dcba_l44_4461

theorem gcd_of_abcd_plus_dcba :
  ∃ (g : ℕ), g > 1 ∧ 
  (∀ (a : ℕ), 0 ≤ a → a ≤ 3 → 
    g ∣ (1000 * a + 100 * (a + 2) + 10 * (a + 4) + (a + 6)) + 
        (1000 * (a + 6) + 100 * (a + 4) + 10 * (a + 2) + a)) ∧
  (∀ (d : ℕ), d > g → 
    ∃ (a : ℕ), 0 ≤ a ∧ a ≤ 3 ∧ 
      ¬(d ∣ (1000 * a + 100 * (a + 2) + 10 * (a + 4) + (a + 6)) + 
           (1000 * (a + 6) + 100 * (a + 4) + 10 * (a + 2) + a))) ∧
  g = 2 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_abcd_plus_dcba_l44_4461


namespace NUMINAMATH_CALUDE_time_for_c_is_48_l44_4463

/-- The time it takes for worker c to complete the work alone -/
def time_for_c (time_ab time_bc time_ca : ℚ) : ℚ :=
  let a := (1 / time_ab + 1 / time_ca - 1 / time_bc) / 2
  let b := (1 / time_ab + 1 / time_bc - 1 / time_ca) / 2
  let c := (1 / time_bc + 1 / time_ca - 1 / time_ab) / 2
  1 / c

/-- Theorem stating that given the conditions, c will take 48 days to do the work alone -/
theorem time_for_c_is_48 :
  time_for_c 6 8 12 = 48 := by sorry

end NUMINAMATH_CALUDE_time_for_c_is_48_l44_4463


namespace NUMINAMATH_CALUDE_line_equation_forms_l44_4457

/-- Given a line with equation (3x-2)/4 - (2y-1)/2 = 1, prove its various forms -/
theorem line_equation_forms (x y : ℝ) :
  (3*x - 2)/4 - (2*y - 1)/2 = 1 →
  (3*x - 8*y - 2 = 0) ∧
  (y = (3/8)*x - 1/4) ∧
  (x/(2/3) + y/(-1/4) = 1) ∧
  ((3/Real.sqrt 73)*x - (8/Real.sqrt 73)*y - (2/Real.sqrt 73) = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_forms_l44_4457


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l44_4426

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 16) :
  ∃ (max : ℝ), (∀ a b : ℝ, a^2 + b^2 = 16 → |a| + |b| ≤ max) ∧ max = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l44_4426


namespace NUMINAMATH_CALUDE_total_leaves_on_our_farm_l44_4484

/-- Represents a farm with trees -/
structure Farm :=
  (num_trees : ℕ)
  (branches_per_tree : ℕ)
  (sub_branches_per_branch : ℕ)
  (leaves_per_sub_branch : ℕ)

/-- Calculates the total number of leaves on all trees in the farm -/
def total_leaves (f : Farm) : ℕ :=
  f.num_trees * f.branches_per_tree * f.sub_branches_per_branch * f.leaves_per_sub_branch

/-- The farm described in the problem -/
def our_farm : Farm :=
  { num_trees := 4
  , branches_per_tree := 10
  , sub_branches_per_branch := 40
  , leaves_per_sub_branch := 60 }

/-- Theorem stating that the total number of leaves on all trees in our farm is 96,000 -/
theorem total_leaves_on_our_farm : total_leaves our_farm = 96000 := by
  sorry

end NUMINAMATH_CALUDE_total_leaves_on_our_farm_l44_4484


namespace NUMINAMATH_CALUDE_student_pairs_count_l44_4428

def number_of_students : ℕ := 15

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem student_pairs_count :
  choose number_of_students 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_count_l44_4428


namespace NUMINAMATH_CALUDE_x_squared_minus_one_necessary_not_sufficient_l44_4401

theorem x_squared_minus_one_necessary_not_sufficient :
  (∀ x : ℝ, x - 1 = 0 → x^2 - 1 = 0) ∧
  ¬(∀ x : ℝ, x^2 - 1 = 0 → x - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_minus_one_necessary_not_sufficient_l44_4401


namespace NUMINAMATH_CALUDE_unique_starting_digit_l44_4443

def starts_with (x : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, d * 10^k ≤ x ∧ x < (d + 1) * 10^k

theorem unique_starting_digit :
  ∃! a : ℕ, a < 10 ∧ 
    (∃ n : ℕ, starts_with (2^n) a ∧ starts_with (5^n) a) ∧
    (a^2 < 10 ∧ 10 < (a+1)^2) :=
by sorry

end NUMINAMATH_CALUDE_unique_starting_digit_l44_4443


namespace NUMINAMATH_CALUDE_jimmy_snow_shoveling_charge_l44_4450

/-- The amount Jimmy charges per driveway for snow shoveling -/
def jimmy_charge_per_driveway : ℝ := 1.50

theorem jimmy_snow_shoveling_charge :
  let candy_bar_price : ℝ := 0.75
  let candy_bar_count : ℕ := 2
  let lollipop_price : ℝ := 0.25
  let lollipop_count : ℕ := 4
  let driveways_shoveled : ℕ := 10
  let candy_store_spend : ℝ := candy_bar_price * candy_bar_count + lollipop_price * lollipop_count
  let snow_shoveling_earnings : ℝ := candy_store_spend * 6
  jimmy_charge_per_driveway = snow_shoveling_earnings / driveways_shoveled :=
by
  sorry

#check jimmy_snow_shoveling_charge

end NUMINAMATH_CALUDE_jimmy_snow_shoveling_charge_l44_4450


namespace NUMINAMATH_CALUDE_juan_tricycles_l44_4424

theorem juan_tricycles (cars bicycles pickups : ℕ) (total_tires : ℕ) : 
  cars = 15 → 
  bicycles = 3 → 
  pickups = 8 → 
  total_tires = 101 → 
  ∃ (tricycles : ℕ), 
    cars * 4 + bicycles * 2 + pickups * 4 + tricycles * 3 = total_tires ∧ 
    tricycles = 1 := by
  sorry

end NUMINAMATH_CALUDE_juan_tricycles_l44_4424


namespace NUMINAMATH_CALUDE_alice_weekly_distance_l44_4469

/-- The distance Alice walks to school each day -/
def distance_to_school : ℕ := 10

/-- The distance Alice walks back home each day -/
def distance_from_school : ℕ := 12

/-- The number of days Alice walks to and from school in a week -/
def days_per_week : ℕ := 5

/-- Theorem: Alice's total walking distance for the week is 110 miles -/
theorem alice_weekly_distance :
  (distance_to_school + distance_from_school) * days_per_week = 110 := by
  sorry

end NUMINAMATH_CALUDE_alice_weekly_distance_l44_4469


namespace NUMINAMATH_CALUDE_cubic_root_increasing_l44_4417

theorem cubic_root_increasing : 
  ∀ (x y : ℝ), x < y → (x ^ (1/3 : ℝ)) < (y ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_increasing_l44_4417


namespace NUMINAMATH_CALUDE_cheaper_feed_cost_l44_4407

/-- Proves that the cost of the cheaper feed is $0.18 per pound given the problem conditions --/
theorem cheaper_feed_cost (total_mix : ℝ) (mix_price : ℝ) (expensive_price : ℝ) (cheaper_amount : ℝ) 
  (h1 : total_mix = 35)
  (h2 : mix_price = 0.36)
  (h3 : expensive_price = 0.53)
  (h4 : cheaper_amount = 17) :
  ∃ (cheaper_price : ℝ), 
    cheaper_price * cheaper_amount + expensive_price * (total_mix - cheaper_amount) = mix_price * total_mix ∧ 
    cheaper_price = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_cheaper_feed_cost_l44_4407


namespace NUMINAMATH_CALUDE_distance_point_to_line_polar_example_l44_4454

/-- The distance from a point to a line in polar coordinates -/
def distance_point_to_line_polar (ρ₀ : ℝ) (θ₀ : ℝ) (f : ℝ → ℝ → ℝ) : ℝ :=
  sorry

theorem distance_point_to_line_polar_example :
  distance_point_to_line_polar 2 (π/3) (fun ρ θ ↦ ρ * Real.cos (θ + π/3) - 2) = 3 :=
sorry

end NUMINAMATH_CALUDE_distance_point_to_line_polar_example_l44_4454


namespace NUMINAMATH_CALUDE_line_through_point_l44_4495

/-- If the line ax + 3y - 5 = 0 passes through the point (2, 1), then a = 1 -/
theorem line_through_point (a : ℝ) : 
  (a * 2 + 3 * 1 - 5 = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l44_4495


namespace NUMINAMATH_CALUDE_inequality_proof_l44_4474

def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

def A : Set ℝ := {x | f x ≤ 6}

theorem inequality_proof (m n : ℝ) (hm : m ∈ A) (hn : n ∈ A) :
  |1/3 * m - 1/2 * n| ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l44_4474


namespace NUMINAMATH_CALUDE_exactly_six_numbers_l44_4412

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧
  ∃ k : ℕ, n - reverse_digits n = k^3 ∧ k > 0

theorem exactly_six_numbers :
  ∃! (s : Finset ℕ), s.card = 6 ∧ ∀ n, n ∈ s ↔ satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_exactly_six_numbers_l44_4412


namespace NUMINAMATH_CALUDE_invisible_dots_count_l44_4434

def dice_faces : List Nat := [1, 2, 3, 4, 5, 6]

def visible_faces : List Nat := [6, 5, 3, 1, 4, 2, 1]

def total_faces : Nat := 3 * 6

def visible_faces_count : Nat := 7

def hidden_faces_count : Nat := total_faces - visible_faces_count

theorem invisible_dots_count :
  (3 * (dice_faces.sum)) - (visible_faces.sum) = 41 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l44_4434


namespace NUMINAMATH_CALUDE_inequality_proof_l44_4494

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_condition : a + b + c = 1) : 
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l44_4494


namespace NUMINAMATH_CALUDE_petya_wins_against_sasha_l44_4472

/-- Represents a player in the elimination tennis game -/
inductive Player : Type
| Petya : Player
| Sasha : Player
| Misha : Player

/-- The number of matches played by each player -/
def matches_played (p : Player) : ℕ :=
  match p with
  | Player.Petya => 12
  | Player.Sasha => 7
  | Player.Misha => 11

/-- The total number of matches played -/
def total_matches : ℕ := 15

/-- The number of wins by one player against another -/
def wins_against (winner loser : Player) : ℕ := sorry

theorem petya_wins_against_sasha :
  wins_against Player.Petya Player.Sasha = 4 :=
by sorry

end NUMINAMATH_CALUDE_petya_wins_against_sasha_l44_4472


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l44_4438

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 3 →
  a 1 + a 6 = 12 →
  a 7 + a 8 + a 9 = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l44_4438


namespace NUMINAMATH_CALUDE_construction_labor_cost_l44_4477

def worker_salary : ℕ := 100
def electrician_salary : ℕ := 2 * worker_salary
def plumber_salary : ℕ := (5 * worker_salary) / 2
def architect_salary : ℕ := (7 * worker_salary) / 2

def project_cost : ℕ := 2 * worker_salary + electrician_salary + plumber_salary + architect_salary

def total_cost : ℕ := 3 * project_cost

theorem construction_labor_cost : total_cost = 3000 := by
  sorry

end NUMINAMATH_CALUDE_construction_labor_cost_l44_4477
