import Mathlib

namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l2855_285535

/-- A chessboard is represented as a function from (Fin 8 × Fin 8) to Option (Fin 2),
    where Some 0 represents a white piece, Some 1 represents a black piece,
    and None represents an empty square. -/
def Chessboard := Fin 8 → Fin 8 → Option (Fin 2)

/-- Count the number of neighbors of a given color for a piece at position (i, j) -/
def countNeighbors (board : Chessboard) (i j : Fin 8) (color : Fin 2) : Nat :=
  sorry

/-- Check if a given arrangement satisfies the condition that each piece
    has an equal number of white and black neighbors -/
def isValidArrangement (board : Chessboard) : Prop :=
  sorry

/-- Count the total number of pieces of a given color on the board -/
def countPieces (board : Chessboard) (color : Fin 2) : Nat :=
  sorry

/-- The main theorem stating that a valid arrangement exists -/
theorem valid_arrangement_exists : ∃ (board : Chessboard),
  (countPieces board 0 = 16) ∧
  (countPieces board 1 = 16) ∧
  isValidArrangement board :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_exists_l2855_285535


namespace NUMINAMATH_CALUDE_rational_difference_l2855_285534

theorem rational_difference (x y : ℚ) (h : (1 + y) / (x - y) = x) : y = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_difference_l2855_285534


namespace NUMINAMATH_CALUDE_boots_sold_on_monday_l2855_285597

/-- Represents the sales data for a shoe store on a given day -/
structure DailySales where
  shoes : ℕ
  boots : ℕ
  total : ℚ

/-- Represents the pricing structure of the shoe store -/
structure Pricing where
  shoe_price : ℚ
  boot_price : ℚ

def monday_sales : DailySales :=
  { shoes := 22, boots := 24, total := 460 }

def tuesday_sales : DailySales :=
  { shoes := 8, boots := 32, total := 560 }

def store_pricing : Pricing :=
  { shoe_price := 2, boot_price := 17 }

theorem boots_sold_on_monday :
  ∃ (x : ℕ), 
    x = monday_sales.boots ∧
    store_pricing.boot_price = store_pricing.shoe_price + 15 ∧
    monday_sales.shoes * store_pricing.shoe_price + x * store_pricing.boot_price = monday_sales.total ∧
    tuesday_sales.shoes * store_pricing.shoe_price + tuesday_sales.boots * store_pricing.boot_price = tuesday_sales.total :=
by sorry

end NUMINAMATH_CALUDE_boots_sold_on_monday_l2855_285597


namespace NUMINAMATH_CALUDE_ad_ratio_proof_l2855_285583

theorem ad_ratio_proof (page1_ads page2_ads page3_ads page4_ads total_ads : ℕ) : 
  page1_ads = 12 →
  page3_ads = page2_ads + 24 →
  page4_ads = (3 * page2_ads) / 4 →
  total_ads = page1_ads + page2_ads + page3_ads + page4_ads →
  (2 * total_ads) / 3 = 68 →
  page2_ads / page1_ads = 2 := by
sorry

end NUMINAMATH_CALUDE_ad_ratio_proof_l2855_285583


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2855_285520

/-- Given a hyperbola with the general equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    one asymptote passing through the point (2, √3),
    and one focus lying on the directrix of the parabola y² = 4√7x,
    prove that the specific equation of the hyperbola is x²/4 - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote_condition : b / a = Real.sqrt 3 / 2)
  (focus_condition : ∃ (x y : ℝ), x = -Real.sqrt 7 ∧ x^2 / a^2 - y^2 / b^2 = 1) :
  a^2 = 4 ∧ b^2 = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2855_285520


namespace NUMINAMATH_CALUDE_expand_expression_l2855_285507

theorem expand_expression (x : ℝ) : (7*x + 11) * (3*x^2 + 2*x) = 21*x^3 + 47*x^2 + 22*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2855_285507


namespace NUMINAMATH_CALUDE_perfect_square_octal_rep_c_is_one_l2855_285529

/-- Octal representation of a number -/
structure OctalRep where
  a : ℕ
  b : ℕ
  c : ℕ
  h_a_nonzero : a ≠ 0

/-- Perfect square with specific octal representation -/
def is_perfect_square_with_octal_rep (n : ℕ) (rep : OctalRep) : Prop :=
  ∃ k : ℕ, n = k^2 ∧ n = 8^3 * rep.a + 8^2 * rep.b + 8 * 3 + rep.c

theorem perfect_square_octal_rep_c_is_one (n : ℕ) (rep : OctalRep) :
  is_perfect_square_with_octal_rep n rep → rep.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_octal_rep_c_is_one_l2855_285529


namespace NUMINAMATH_CALUDE_contribution_is_180_l2855_285572

/-- Calculates the individual contribution for painting a wall --/
def calculate_contribution (paint_cost_per_gallon : ℚ) (coverage_per_gallon : ℚ) (total_area : ℚ) (num_coats : ℕ) : ℚ :=
  let total_gallons := (total_area / coverage_per_gallon) * num_coats
  let total_cost := total_gallons * paint_cost_per_gallon
  total_cost / 2

/-- Proves that each person's contribution is $180 --/
theorem contribution_is_180 :
  calculate_contribution 45 400 1600 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_contribution_is_180_l2855_285572


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l2855_285533

theorem quadratic_equation_root_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 6 * x + k^2 - 1 = 0) ∧ 
  ((k - 1) * 0^2 + 6 * 0 + k^2 - 1 = 0) → 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l2855_285533


namespace NUMINAMATH_CALUDE_polygon_internal_external_angles_equal_l2855_285569

theorem polygon_internal_external_angles_equal (n : ℕ) : 
  (n : ℝ) ≥ 3 → ((n - 2) * 180 = 360) → n = 4 := by sorry

end NUMINAMATH_CALUDE_polygon_internal_external_angles_equal_l2855_285569


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2855_285508

/-- Given a polynomial q(x) = Dx^4 + Ex^2 + Fx + 6, if the remainder when q(x) is divided by (x - 2) is 14, 
    then the remainder when q(x) is divided by (x + 2) is also 14 -/
theorem polynomial_remainder (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^4 + E * x^2 + F * x + 6
  (q 2 = 14) → (q (-2) = 14) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2855_285508


namespace NUMINAMATH_CALUDE_incenter_is_angle_bisectors_intersection_l2855_285514

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- An angle bisector of a triangle --/
def angle_bisector (t : Triangle) (vertex : Fin 3) : Set (ℝ × ℝ) := sorry

/-- The intersection point of the angle bisectors --/
def angle_bisectors_intersection (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem: The incenter of a triangle is the intersection point of its angle bisectors --/
theorem incenter_is_angle_bisectors_intersection (t : Triangle) :
  incenter t = angle_bisectors_intersection t := by sorry

end NUMINAMATH_CALUDE_incenter_is_angle_bisectors_intersection_l2855_285514


namespace NUMINAMATH_CALUDE_marks_candies_l2855_285500

-- Define the number of people
def num_people : ℕ := 3

-- Define the number of candies each person will have after sharing
def shared_candies : ℕ := 30

-- Define Peter's candies
def peter_candies : ℕ := 25

-- Define John's candies
def john_candies : ℕ := 35

-- Theorem to prove Mark's candies
theorem marks_candies :
  shared_candies * num_people - (peter_candies + john_candies) = 30 := by
  sorry


end NUMINAMATH_CALUDE_marks_candies_l2855_285500


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2855_285565

open Real

theorem trigonometric_identity :
  sin (12 * π / 180) * cos (36 * π / 180) * sin (48 * π / 180) * cos (72 * π / 180) * tan (18 * π / 180) =
  1/2 * (sin (12 * π / 180)^2 + sin (12 * π / 180) * cos (6 * π / 180)) * sin (18 * π / 180)^2 / cos (18 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2855_285565


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2855_285519

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  (∀ m : ℕ, m > 0 ∧ 
    (3815 % m = 31 ∧ 4521 % m = 33) → 
    m ≤ n) ∧
  3815 % n = 31 ∧ 
  4521 % n = 33 ∧
  n = 64 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2855_285519


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l2855_285547

/-- The lateral area of a cylinder with base diameter and height both 4 cm is 16π cm² -/
theorem cylinder_lateral_area (π : ℝ) : 
  let base_diameter : ℝ := 4
  let height : ℝ := 4
  let lateral_area : ℝ := π * base_diameter * height
  lateral_area = 16 * π := by
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l2855_285547


namespace NUMINAMATH_CALUDE_max_m_value_l2855_285575

-- Define the circle M
def circle_M (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 4*p.2 + 5 = m}

-- Define points A and B
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (2, 0)

-- Define the property of right angle APB
def is_right_angle (P : ℝ × ℝ) : Prop :=
  let AP := (P.1 - point_A.1, P.2 - point_A.2)
  let BP := (P.1 - point_B.1, P.2 - point_B.2)
  AP.1 * BP.1 + AP.2 * BP.2 = 0

-- Theorem statement
theorem max_m_value :
  ∃ (m : ℝ), ∀ (m' : ℝ),
    (∃ (P : ℝ × ℝ), P ∈ circle_M m' ∧ is_right_angle P) →
    m' ≤ m ∧
    m = 45 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l2855_285575


namespace NUMINAMATH_CALUDE_discount_calculation_l2855_285546

def list_price : ℝ := 70
def final_price : ℝ := 61.11
def first_discount : ℝ := 10

theorem discount_calculation (x : ℝ) :
  list_price * (1 - first_discount / 100) * (1 - x / 100) = final_price →
  x = 3 := by sorry

end NUMINAMATH_CALUDE_discount_calculation_l2855_285546


namespace NUMINAMATH_CALUDE_books_taken_out_on_friday_l2855_285511

theorem books_taken_out_on_friday 
  (initial_books : ℕ) 
  (taken_out_tuesday : ℕ) 
  (brought_back_thursday : ℕ) 
  (final_books : ℕ) 
  (h1 : initial_books = 235)
  (h2 : taken_out_tuesday = 227)
  (h3 : brought_back_thursday = 56)
  (h4 : final_books = 29) :
  initial_books - taken_out_tuesday + brought_back_thursday - final_books = 35 :=
by sorry

end NUMINAMATH_CALUDE_books_taken_out_on_friday_l2855_285511


namespace NUMINAMATH_CALUDE_ford_younger_than_christopher_l2855_285566

/-- Proves that Ford is 2 years younger than Christopher given the conditions of the problem -/
theorem ford_younger_than_christopher :
  ∀ (george christopher ford : ℕ),
    george = christopher + 8 →
    george + christopher + ford = 60 →
    christopher = 18 →
    ∃ (y : ℕ), ford = christopher - y ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_ford_younger_than_christopher_l2855_285566


namespace NUMINAMATH_CALUDE_tens_digit_of_difference_l2855_285503

/-- Given a single digit t, prove that the tens digit of (6t5 - 5t6) is 9 -/
theorem tens_digit_of_difference (t : ℕ) (h : t < 10) : 
  (6 * 100 + t * 10 + 5) - (5 * 100 + t * 10 + 6) = 94 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_difference_l2855_285503


namespace NUMINAMATH_CALUDE_sean_patch_profit_l2855_285596

/-- Sean's patch business profit calculation -/
theorem sean_patch_profit :
  let order_size : ℕ := 100
  let cost_per_patch : ℚ := 125 / 100
  let selling_price : ℚ := 12
  let total_cost : ℚ := order_size * cost_per_patch
  let total_revenue : ℚ := order_size * selling_price
  let net_profit : ℚ := total_revenue - total_cost
  net_profit = 1075 := by sorry

end NUMINAMATH_CALUDE_sean_patch_profit_l2855_285596


namespace NUMINAMATH_CALUDE_third_roll_wraps_four_gifts_l2855_285527

/-- Represents the number of gifts wrapped with the third roll of paper. -/
def gifts_wrapped_third_roll (total_rolls : ℕ) (total_gifts : ℕ) (gifts_first_roll : ℕ) (gifts_second_roll : ℕ) : ℕ :=
  total_gifts - (gifts_first_roll + gifts_second_roll)

/-- Proves that given 3 rolls of wrapping paper and 12 gifts, if 1 roll wraps 3 gifts
    and 1 roll wraps 5 gifts, then the number of gifts wrapped with the third roll is 4. -/
theorem third_roll_wraps_four_gifts :
  gifts_wrapped_third_roll 3 12 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_third_roll_wraps_four_gifts_l2855_285527


namespace NUMINAMATH_CALUDE_equation_linear_iff_a_eq_neg_two_l2855_285544

/-- The equation (a-2)x^(|a|^(-1)+3) = 0 is linear in x if and only if a = -2 -/
theorem equation_linear_iff_a_eq_neg_two (a : ℝ) :
  (∀ x, ∃ b c : ℝ, (a - 2) * x^(|a|⁻¹ + 3) = b * x + c) ↔ a = -2 :=
sorry

end NUMINAMATH_CALUDE_equation_linear_iff_a_eq_neg_two_l2855_285544


namespace NUMINAMATH_CALUDE_sum_of_alan_and_bob_ages_l2855_285570

-- Define the set of possible ages
def Ages : Set ℕ := {3, 8, 12, 14}

-- Define the cousins' ages as natural numbers
variables (alan_age bob_age carl_age dan_age : ℕ)

-- Define the conditions
def conditions (alan_age bob_age carl_age dan_age : ℕ) : Prop :=
  alan_age ∈ Ages ∧ bob_age ∈ Ages ∧ carl_age ∈ Ages ∧ dan_age ∈ Ages ∧
  alan_age ≠ bob_age ∧ alan_age ≠ carl_age ∧ alan_age ≠ dan_age ∧
  bob_age ≠ carl_age ∧ bob_age ≠ dan_age ∧ carl_age ≠ dan_age ∧
  alan_age < carl_age ∧
  (alan_age + dan_age) % 5 = 0 ∧
  (carl_age + dan_age) % 5 = 0

-- Theorem statement
theorem sum_of_alan_and_bob_ages 
  (h : conditions alan_age bob_age carl_age dan_age) :
  alan_age + bob_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_alan_and_bob_ages_l2855_285570


namespace NUMINAMATH_CALUDE_acrobat_count_range_l2855_285552

/-- Represents the count of animals in the zoo --/
structure AnimalCount where
  elephants : ℕ
  monkeys : ℕ
  acrobats : ℕ

/-- Checks if the animal count satisfies the given conditions --/
def isValidCount (count : AnimalCount) : Prop :=
  count.elephants * 4 + count.monkeys * 2 + count.acrobats * 2 = 50 ∧
  count.elephants + count.monkeys + count.acrobats = 18

/-- The main theorem stating the range of possible acrobat counts --/
theorem acrobat_count_range :
  ∀ n : ℕ, 0 ≤ n ∧ n ≤ 11 →
  ∃ (count : AnimalCount), isValidCount count ∧ count.acrobats = n :=
by sorry

end NUMINAMATH_CALUDE_acrobat_count_range_l2855_285552


namespace NUMINAMATH_CALUDE_inequality_proof_l2855_285589

theorem inequality_proof (a b c : ℝ) :
  Real.sqrt (a^2 + (1 - b)^2) + Real.sqrt (b^2 + (1 - c)^2) + Real.sqrt (c^2 + (1 - a)^2) ≥ 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2855_285589


namespace NUMINAMATH_CALUDE_clothing_price_comparison_l2855_285564

theorem clothing_price_comparison (original_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  original_price = 120 →
  increase_rate = 0.2 →
  discount_rate = 0.2 →
  original_price * (1 + increase_rate) * (1 - discount_rate) < original_price :=
by sorry

end NUMINAMATH_CALUDE_clothing_price_comparison_l2855_285564


namespace NUMINAMATH_CALUDE_rational_operations_l2855_285513

-- Define the new operation
def star (x y : ℚ) : ℚ := x + y - x * y

-- Theorem statement
theorem rational_operations :
  -- Unit elements
  (∀ a : ℚ, a + 0 = a) ∧
  (∀ a : ℚ, a * 1 = a) ∧
  -- Inverse element of 3 under addition
  (3 + (-3) = 0) ∧
  -- 0 has no multiplicative inverse
  (∀ x : ℚ, x ≠ 0 → ∃ y : ℚ, x * y = 1) ∧
  -- Properties of the new operation
  (∀ x : ℚ, star x 0 = x) ∧
  (∀ m : ℚ, m ≠ 1 → star m (m / (m - 1)) = 0) :=
by sorry

end NUMINAMATH_CALUDE_rational_operations_l2855_285513


namespace NUMINAMATH_CALUDE_prob_white_then_yellow_is_two_thirds_l2855_285553

/-- The probability of drawing a white ball first, followed by a yellow ball, 
    from a bag containing 6 yellow and 4 white ping pong balls, 
    when drawing two balls without replacement. -/
def prob_white_then_yellow : ℚ :=
  let total_balls : ℕ := 10
  let yellow_balls : ℕ := 6
  let white_balls : ℕ := 4
  let prob_white_first : ℚ := white_balls / total_balls
  let prob_yellow_second : ℚ := yellow_balls / (total_balls - 1)
  prob_white_first * prob_yellow_second

theorem prob_white_then_yellow_is_two_thirds :
  prob_white_then_yellow = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_then_yellow_is_two_thirds_l2855_285553


namespace NUMINAMATH_CALUDE_angle_ratio_l2855_285539

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- BP and BQ trisect ∠ABC
axiom trisect : angle A B P = angle B P Q ∧ angle B P Q = angle P B Q

-- BM bisects ∠ABP
axiom bisect : angle A B M = (1/2) * angle A B P

-- Theorem statement
theorem angle_ratio : 
  (angle M B Q) / (angle A B Q) = 3/4 := by sorry

end NUMINAMATH_CALUDE_angle_ratio_l2855_285539


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_equals_243_l2855_285532

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 24*x^2 + 143*x - 210

-- Define the roots of the polynomial
variables (p q r : ℝ)

-- State that p, q, r are the roots of f
axiom roots_of_f : f p = 0 ∧ f q = 0 ∧ f r = 0

-- Define A, B, C as real numbers
variables (A B C : ℝ)

-- State the partial fraction decomposition
axiom partial_fraction_decomposition :
  ∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    1 / (s^3 - 24*s^2 + 143*s - 210) = A / (s - p) + B / (s - q) + C / (s - r)

-- The theorem to prove
theorem sum_of_reciprocals_equals_243 :
  1 / A + 1 / B + 1 / C = 243 :=
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_equals_243_l2855_285532


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l2855_285586

theorem power_fraction_simplification : (25 ^ 40) / (125 ^ 20) = 5 ^ 20 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l2855_285586


namespace NUMINAMATH_CALUDE_fraction_before_lunch_l2855_285590

/-- Proves that the fraction of distance driven before lunch is 1/4 given the problem conditions --/
theorem fraction_before_lunch (total_distance : ℝ) (total_time : ℝ) (lunch_time : ℝ) 
  (h1 : total_distance = 200)
  (h2 : total_time = 5)
  (h3 : lunch_time = 1)
  (h4 : total_time ≥ lunch_time) :
  let f := (total_time - lunch_time) / 4 / (total_time - lunch_time)
  f = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_before_lunch_l2855_285590


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l2855_285526

theorem logarithmic_equation_solution (x : ℝ) :
  (x > 0) →
  (5 * (Real.log x / Real.log (x / 9)) + 
   (Real.log (x^3) / Real.log (9 / x)) + 
   8 * (Real.log (x^2) / Real.log (9 * x^2)) = 2) ↔ 
  (x = 3 ∨ x = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l2855_285526


namespace NUMINAMATH_CALUDE_expression_evaluation_l2855_285562

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 15) - 2 = -x^4 + 3*x^3 - 5*x^2 + 15*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2855_285562


namespace NUMINAMATH_CALUDE_ellipse_condition_l2855_285576

/-- The equation represents an ellipse with foci on the x-axis -/
def is_ellipse_on_x_axis (k : ℝ) : Prop :=
  (∀ x y : ℝ, x^2 / (2 - k) + y^2 / (2*k - 1) = 1) ∧
  (2 - k > 0) ∧ (2*k - 1 > 0) ∧ (2 - k > 2*k - 1)

theorem ellipse_condition (k : ℝ) :
  is_ellipse_on_x_axis k ↔ 1/2 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2855_285576


namespace NUMINAMATH_CALUDE_line_parallel_transitive_plane_parallel_transitive_l2855_285509

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallelLine : Line → Line → Prop)

-- Define the parallel relation for planes
variable (parallelPlane : Plane → Plane → Prop)

-- Theorem for lines
theorem line_parallel_transitive (a b c : Line) :
  parallelLine a b → parallelLine a c → parallelLine b c := by sorry

-- Theorem for planes
theorem plane_parallel_transitive (α β γ : Plane) :
  parallelPlane α β → parallelPlane α γ → parallelPlane β γ := by sorry

end NUMINAMATH_CALUDE_line_parallel_transitive_plane_parallel_transitive_l2855_285509


namespace NUMINAMATH_CALUDE_fans_with_all_items_l2855_285559

def arena_capacity : ℕ := 5000
def tshirt_interval : ℕ := 100
def cap_interval : ℕ := 40
def brochure_interval : ℕ := 60

theorem fans_with_all_items : 
  (arena_capacity / (Nat.lcm (Nat.lcm tshirt_interval cap_interval) brochure_interval) : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l2855_285559


namespace NUMINAMATH_CALUDE_angle_subtraction_theorem_polynomial_simplification_theorem_l2855_285521

-- Define angle type
def Angle := ℕ × ℕ -- (degrees, minutes)

-- Define angle subtraction
def angle_sub (a b : Angle) : Angle := sorry

-- Define polynomial expression
def poly_expr (m : ℝ) := 5*m^2 - (m^2 - 6*m) - 2*(-m + 3*m^2)

theorem angle_subtraction_theorem :
  angle_sub (34, 26) (25, 33) = (8, 53) := by sorry

theorem polynomial_simplification_theorem (m : ℝ) :
  poly_expr m = -2*m^2 + 8*m := by sorry

end NUMINAMATH_CALUDE_angle_subtraction_theorem_polynomial_simplification_theorem_l2855_285521


namespace NUMINAMATH_CALUDE_hyperbola_proof_1_hyperbola_proof_2_l2855_285545

-- Part 1
def hyperbola_equation_1 (x y : ℝ) : Prop :=
  x^2 / 5 - y^2 = 1

theorem hyperbola_proof_1 (c : ℝ) (h1 : c = Real.sqrt 6) :
  hyperbola_equation_1 (-5) 2 ∧
  ∃ a b : ℝ, c^2 = a^2 + b^2 ∧ hyperbola_equation_1 x y ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
sorry

-- Part 2
def hyperbola_equation_2 (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

theorem hyperbola_proof_2 :
  hyperbola_equation_2 3 (-4 * Real.sqrt 2) ∧
  hyperbola_equation_2 (9/4) 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_proof_1_hyperbola_proof_2_l2855_285545


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l2855_285593

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem intersection_equality_implies_a_range (a : ℝ) :
  A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l2855_285593


namespace NUMINAMATH_CALUDE_complex_parts_of_i_squared_plus_i_l2855_285505

theorem complex_parts_of_i_squared_plus_i :
  let i : ℂ := Complex.I
  let z : ℂ := i^2 + i
  (z.re = -1) ∧ (z.im = 1) := by sorry

end NUMINAMATH_CALUDE_complex_parts_of_i_squared_plus_i_l2855_285505


namespace NUMINAMATH_CALUDE_dresses_total_l2855_285550

/-- The total number of dresses for Emily, Melissa, and Debora -/
def total_dresses (emily_dresses melissa_dresses debora_dresses : ℕ) : ℕ :=
  emily_dresses + melissa_dresses + debora_dresses

/-- Theorem stating the total number of dresses given the conditions -/
theorem dresses_total (emily_dresses : ℕ) 
  (h1 : emily_dresses = 16)
  (h2 : ∃ (melissa_dresses : ℕ), melissa_dresses = emily_dresses / 2)
  (h3 : ∃ (debora_dresses : ℕ), debora_dresses = emily_dresses / 2 + 12) :
  ∃ (total : ℕ), total = total_dresses emily_dresses (emily_dresses / 2) (emily_dresses / 2 + 12) ∧ total = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_dresses_total_l2855_285550


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2855_285574

theorem shaded_area_calculation (carpet_side : ℝ) (S T : ℝ) 
  (h1 : carpet_side = 12)
  (h2 : carpet_side / S = 4)
  (h3 : S / T = 4)
  (h4 : carpet_side > 0) : 
  S^2 + 16 * T^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2855_285574


namespace NUMINAMATH_CALUDE_particular_number_plus_eight_l2855_285560

theorem particular_number_plus_eight (n : ℝ) : n * 6 = 72 → n + 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_plus_eight_l2855_285560


namespace NUMINAMATH_CALUDE_min_value_theorem_l2855_285577

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 24 / 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 24 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2855_285577


namespace NUMINAMATH_CALUDE_sum_first_102_remainder_l2855_285540

theorem sum_first_102_remainder (n : Nat) (h : n = 102) : 
  (n * (n + 1) / 2) % 5250 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_102_remainder_l2855_285540


namespace NUMINAMATH_CALUDE_seven_consecutive_integers_product_divisible_by_ten_l2855_285556

theorem seven_consecutive_integers_product_divisible_by_ten (n : ℕ+) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) : ℕ) = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_seven_consecutive_integers_product_divisible_by_ten_l2855_285556


namespace NUMINAMATH_CALUDE_license_plate_count_l2855_285568

/-- The number of possible letters in each letter position of the license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in each digit position of the license plate. -/
def num_digits : ℕ := 10

/-- The number of letter positions in the license plate. -/
def num_letter_positions : ℕ := 3

/-- The number of digit positions in the license plate. -/
def num_digit_positions : ℕ := 4

/-- The total number of possible license plates in Eldorado. -/
def total_license_plates : ℕ := num_letters ^ num_letter_positions * num_digits ^ num_digit_positions

theorem license_plate_count :
  total_license_plates = 175760000 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l2855_285568


namespace NUMINAMATH_CALUDE_fraction_equality_l2855_285581

theorem fraction_equality (a b c : ℝ) :
  (|a^2 + b^2|^3 + |b^2 + c^2|^3 + |c^2 + a^2|^3) / (|a + b|^3 + |b + c|^3 + |c + a|^3) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2855_285581


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_45_l2855_285537

/-- Reverses a three-digit number -/
def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem three_digit_number_divisible_by_45 (n : ℕ) :
  is_three_digit n →
  n % 45 = 0 →
  n - reverse_number n = 297 →
  n = 360 ∨ n = 855 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_45_l2855_285537


namespace NUMINAMATH_CALUDE_red_then_blue_probability_l2855_285573

/-- The probability of drawing a red marble first and a blue marble second from a jar -/
theorem red_then_blue_probability (red green white blue : ℕ) :
  red = 4 →
  green = 3 →
  white = 10 →
  blue = 2 →
  let total := red + green + white + blue
  let prob_red := red / total
  let prob_blue_after_red := blue / (total - 1)
  prob_red * prob_blue_after_red = 4 / 171 := by
  sorry

end NUMINAMATH_CALUDE_red_then_blue_probability_l2855_285573


namespace NUMINAMATH_CALUDE_division_in_third_quadrant_l2855_285561

/-- Given two complex numbers z₁ and z₂, prove that z₁/z₂ is in the third quadrant -/
theorem division_in_third_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 1 - 2 * Complex.I) 
  (h₂ : z₂ = 2 + 3 * Complex.I) : 
  (z₁ / z₂).re < 0 ∧ (z₁ / z₂).im < 0 := by
  sorry

end NUMINAMATH_CALUDE_division_in_third_quadrant_l2855_285561


namespace NUMINAMATH_CALUDE_julia_tag_game_l2855_285501

theorem julia_tag_game (monday tuesday : ℕ) 
  (h1 : monday = 12) 
  (h2 : tuesday = 7) : 
  monday + tuesday = 19 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l2855_285501


namespace NUMINAMATH_CALUDE_mallory_journey_expenses_l2855_285531

/-- Calculates the total expenses for Mallory's journey --/
def journey_expenses (fuel_cost : ℚ) (tank_range : ℚ) (journey_distance : ℚ) 
  (hotel_nights : ℕ) (hotel_cost : ℚ) (fuel_increase : ℚ) 
  (maintenance_cost : ℚ) (activity_cost : ℚ) : ℚ :=
  let num_refills := (journey_distance / tank_range).ceil
  let total_fuel_cost := (num_refills * (num_refills - 1) / 2 * fuel_increase) + (num_refills * fuel_cost)
  let food_cost := (3 / 5) * total_fuel_cost
  let hotel_total := hotel_nights * hotel_cost
  let extra_expenses := maintenance_cost + activity_cost
  total_fuel_cost + food_cost + hotel_total + extra_expenses

/-- Theorem stating that Mallory's journey expenses equal $746 --/
theorem mallory_journey_expenses : 
  journey_expenses 45 500 2000 3 80 5 120 50 = 746 := by
  sorry

end NUMINAMATH_CALUDE_mallory_journey_expenses_l2855_285531


namespace NUMINAMATH_CALUDE_smallest_cube_ending_584_l2855_285548

theorem smallest_cube_ending_584 :
  ∃ n : ℕ+, (n : ℤ)^3 ≡ 584 [ZMOD 1000] ∧
  ∀ m : ℕ+, (m : ℤ)^3 ≡ 584 [ZMOD 1000] → n ≤ m ∧ n = 34 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_584_l2855_285548


namespace NUMINAMATH_CALUDE_chimney_bricks_count_l2855_285591

-- Define the number of bricks in the chimney
def chimney_bricks : ℕ := 148

-- Define Brenda's time to build the chimney alone
def brenda_time : ℕ := 7

-- Define Brandon's time to build the chimney alone
def brandon_time : ℕ := 8

-- Define the time they take to build the chimney together
def combined_time : ℕ := 6

-- Define the productivity drop when working together
def productivity_drop : ℕ := 15

-- Theorem statement
theorem chimney_bricks_count :
  -- Individual rates
  let brenda_rate := chimney_bricks / brenda_time
  let brandon_rate := chimney_bricks / brandon_time
  -- Combined rate without drop
  let combined_rate := brenda_rate + brandon_rate
  -- Actual combined rate with productivity drop
  let actual_combined_rate := combined_rate - productivity_drop
  -- The work completed matches the number of bricks
  actual_combined_rate * combined_time = chimney_bricks := by
  sorry

end NUMINAMATH_CALUDE_chimney_bricks_count_l2855_285591


namespace NUMINAMATH_CALUDE_marble_bag_problem_l2855_285584

theorem marble_bag_problem (r b : ℕ) : 
  (r - 1 : ℚ) / (r + b - 2 : ℚ) = 1/8 →
  (r : ℚ) / (r + b - 3 : ℚ) = 1/4 →
  r + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l2855_285584


namespace NUMINAMATH_CALUDE_smallest_even_abundant_l2855_285525

/-- A number is abundant if the sum of its proper divisors is greater than the number itself. -/
def is_abundant (n : ℕ) : Prop :=
  (Finset.filter (· < n) (Finset.range n)).sum (λ i => if n % i = 0 then i else 0) > n

/-- A number is even if it's divisible by 2. -/
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem smallest_even_abundant : ∀ n : ℕ, is_even n → is_abundant n → n ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_abundant_l2855_285525


namespace NUMINAMATH_CALUDE_luke_played_two_rounds_l2855_285594

/-- The number of rounds Luke played in a trivia game -/
def rounds_played (total_points : ℕ) (points_per_round : ℕ) : ℕ :=
  total_points / points_per_round

/-- Theorem stating that Luke played 2 rounds -/
theorem luke_played_two_rounds :
  rounds_played 84 42 = 2 := by
  sorry

end NUMINAMATH_CALUDE_luke_played_two_rounds_l2855_285594


namespace NUMINAMATH_CALUDE_orbius_5_stay_duration_l2855_285530

/-- Calculates the number of days an astronaut stays on a planet given the total days per year, 
    number of seasons per year, and number of seasons stayed. -/
def days_stayed (total_days_per_year : ℕ) (seasons_per_year : ℕ) (seasons_stayed : ℕ) : ℕ :=
  (total_days_per_year / seasons_per_year) * seasons_stayed

/-- Theorem: An astronaut staying on Orbius-5 for 3 seasons will spend 150 days on the planet. -/
theorem orbius_5_stay_duration : 
  days_stayed 250 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_orbius_5_stay_duration_l2855_285530


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2855_285599

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_3 + a_11 = 22, prove that a_7 = 11 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h_sum : a 3 + a 11 = 22) : a 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2855_285599


namespace NUMINAMATH_CALUDE_tan_difference_implies_ratio_l2855_285542

theorem tan_difference_implies_ratio (α : Real) 
  (h : Real.tan (α - π/4) = 1/2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_implies_ratio_l2855_285542


namespace NUMINAMATH_CALUDE_fixed_point_exponential_l2855_285554

/-- The fixed point of the function f(x) = a^(x-2) + 1 -/
theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) + 1
  f 2 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_l2855_285554


namespace NUMINAMATH_CALUDE_complex_exponent_calculation_l2855_285506

theorem complex_exponent_calculation : 
  ((-8 : ℂ) ^ (2/3 : ℂ)) * ((1 / Real.sqrt 2) ^ (-2 : ℂ)) * ((27 : ℂ) ^ (-1/3 : ℂ)) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponent_calculation_l2855_285506


namespace NUMINAMATH_CALUDE_hillary_stop_distance_l2855_285512

/-- Proves that Hillary stops 2900 feet short of the summit given the climbing conditions --/
theorem hillary_stop_distance (summit_distance : ℝ) (hillary_rate : ℝ) (eddy_rate : ℝ) 
  (hillary_descent_rate : ℝ) (climb_time : ℝ) 
  (h1 : summit_distance = 4700)
  (h2 : hillary_rate = 800)
  (h3 : eddy_rate = 500)
  (h4 : hillary_descent_rate = 1000)
  (h5 : climb_time = 6) :
  ∃ x : ℝ, x = 2900 ∧ 
  (summit_distance - x) + (eddy_rate * climb_time) + x = 
  summit_distance + (hillary_rate * climb_time - (summit_distance - x)) :=
by sorry

end NUMINAMATH_CALUDE_hillary_stop_distance_l2855_285512


namespace NUMINAMATH_CALUDE_unique_solution_k_values_l2855_285578

/-- The set of values for k that satisfy the given conditions -/
def k_values : Set ℝ := {1 + Real.sqrt 2, (1 - Real.sqrt 5) / 2}

/-- The system of inequalities -/
def system (k x : ℝ) : Prop :=
  1 ≤ k * x^2 + 2 ∧ x + k ≤ 2

/-- The main theorem stating that k_values is the correct set of values for k -/
theorem unique_solution_k_values :
  ∀ k : ℝ, (∃! x : ℝ, system k x) ↔ k ∈ k_values := by
  sorry

#check unique_solution_k_values

end NUMINAMATH_CALUDE_unique_solution_k_values_l2855_285578


namespace NUMINAMATH_CALUDE_unique_intersection_points_l2855_285551

/-- The set of values k for which |z - 2| = 3|z + 2| intersects |z| = k in exactly one point -/
def intersection_points : Set ℝ :=
  {1.5, 4.5, 5.5}

/-- Predicate to check if a complex number satisfies |z - 2| = 3|z + 2| -/
def satisfies_equation (z : ℂ) : Prop :=
  Complex.abs (z - 2) = 3 * Complex.abs (z + 2)

/-- Predicate to check if a complex number has magnitude k -/
def has_magnitude (z : ℂ) (k : ℝ) : Prop :=
  Complex.abs z = k

/-- The theorem stating that the intersection_points set contains all values of k
    for which |z - 2| = 3|z + 2| intersects |z| = k in exactly one point -/
theorem unique_intersection_points :
  ∀ k : ℝ, (∃! z : ℂ, satisfies_equation z ∧ has_magnitude z k) ↔ k ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_points_l2855_285551


namespace NUMINAMATH_CALUDE_dans_balloons_l2855_285598

theorem dans_balloons (dans_balloons : ℕ) (tims_balloons : ℕ) : 
  tims_balloons = 203 → tims_balloons = 7 * dans_balloons → dans_balloons = 29 := by
  sorry

end NUMINAMATH_CALUDE_dans_balloons_l2855_285598


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2855_285523

/-- Given a quadratic inequality ax^2 - bx + c > 0 with solution set (-1/2, 2), 
    prove properties about its coefficients -/
theorem quadratic_inequality_properties (a b c : ℝ) 
  (h : ∀ x, -1/2 < x ∧ x < 2 ↔ a * x^2 - b * x + c > 0) : 
  b < 0 ∧ c > 0 ∧ a - b + c > 0 ∧ a ≤ 0 ∧ a + b + c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2855_285523


namespace NUMINAMATH_CALUDE_composite_polynomial_l2855_285502

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 6*n^2 + 12*n + 7 = a * b :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_composite_polynomial_l2855_285502


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2855_285557

/-- A quadratic function f(x) with leading coefficient a -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The condition that f(x) > -2x for x ∈ (1,3) -/
def condition_solution_set (a b c : ℝ) : Prop :=
  ∀ x, 1 < x ∧ x < 3 → f a b c x > -2 * x

/-- The condition that f(x) + 6a = 0 has two equal roots -/
def condition_equal_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, f a b c x + 6 * a = 0 ∧
    ∀ y : ℝ, f a b c y + 6 * a = 0 → y = x

/-- The condition that the maximum value of f(x) is positive -/
def condition_positive_max (a b c : ℝ) : Prop :=
  ∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, f a b c x ≤ m

theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a < 0)
  (hf : condition_solution_set a b c) :
  (condition_equal_roots a b c →
    f a b c = fun x ↦ -x^2 - x - 3/5) ∧
  (condition_positive_max a b c →
    a < -2 - Real.sqrt 5 ∨ (-2 + Real.sqrt 5 < a ∧ a < 0)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2855_285557


namespace NUMINAMATH_CALUDE_edward_money_left_l2855_285524

/-- The amount of money Edward initially had to spend --/
def initial_amount : ℚ := 1780 / 100

/-- The cost of one toy car before discount --/
def toy_car_cost : ℚ := 95 / 100

/-- The number of toy cars Edward bought --/
def num_toy_cars : ℕ := 4

/-- The discount rate on toy cars --/
def toy_car_discount_rate : ℚ := 15 / 100

/-- The cost of the race track before tax --/
def race_track_cost : ℚ := 600 / 100

/-- The tax rate on the race track --/
def race_track_tax_rate : ℚ := 8 / 100

/-- The theorem stating how much money Edward has left --/
theorem edward_money_left : 
  initial_amount - 
  (num_toy_cars * toy_car_cost * (1 - toy_car_discount_rate) + 
   race_track_cost * (1 + race_track_tax_rate)) = 809 / 100 := by
  sorry

end NUMINAMATH_CALUDE_edward_money_left_l2855_285524


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l2855_285563

theorem quadratic_root_sum (m n : ℝ) : 
  m^2 + 2*m - 5 = 0 → n^2 + 2*n - 5 = 0 → m^2 + m*n + 2*m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l2855_285563


namespace NUMINAMATH_CALUDE_intersection_sum_l2855_285585

/-- Given two lines that intersect at (3,3), prove that a + b = 4 -/
theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 3 + a) → -- First line passes through (3,3)
  (3 = (1/3) * 3 + b) → -- Second line passes through (3,3)
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l2855_285585


namespace NUMINAMATH_CALUDE_parabola_intersection_l2855_285592

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 6 * x + 2
  let g (x : ℝ) := 9 * x^2 - 4 * x - 5
  (f (-7/3) = g (-7/3) ∧ f (-7/3) = 9) ∧
  (f (1/2) = g (1/2) ∧ f (1/2) = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2855_285592


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2855_285543

theorem regular_polygon_sides (n : ℕ) (h_regular : n ≥ 3) 
  (h_interior_angle : (n - 2) * 180 / n = 140) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2855_285543


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2855_285536

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}
def Q : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2855_285536


namespace NUMINAMATH_CALUDE_complete_factorization_l2855_285516

theorem complete_factorization (x : ℝ) : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_complete_factorization_l2855_285516


namespace NUMINAMATH_CALUDE_graph_shift_l2855_285595

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the shift transformation
def shift (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x - a)

-- Theorem statement
theorem graph_shift (a : ℝ) :
  ∀ x : ℝ, (shift g a) x = g (x - a) :=
by sorry

end NUMINAMATH_CALUDE_graph_shift_l2855_285595


namespace NUMINAMATH_CALUDE_problem_solution_l2855_285538

theorem problem_solution :
  (∀ a : ℝ, 2*a + 3*a - 4*a = a) ∧
  (-1^2022 + 27/4 * (-1/3 - 1) / (-3)^2 + |-1| = -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2855_285538


namespace NUMINAMATH_CALUDE_problem_statement_l2855_285580

open Real

theorem problem_statement :
  ∃ a : ℝ,
    (∀ x : ℝ, x > 0 → exp x - log x ≥ exp a - log a) ∧
    exp a * log a = -1 ∧
    ∀ x₁ x₂ : ℝ,
      1 < x₁ → x₁ < x₂ →
        (∃ x₀ : ℝ, x₁ < x₀ ∧ x₀ < x₂ ∧
          ((exp x₁ - exp x₂) / (x₁ - x₂)) / ((log x₁ - log x₂) / (x₁ - x₂)) = x₀ * exp x₀) ∧
        (exp x₁ - exp x₂) / (x₁ - x₂) - (log x₁ - log x₂) / (x₁ - x₂) <
          (exp x₁ + exp x₂) / 2 - 1 / sqrt (x₁ * x₂) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2855_285580


namespace NUMINAMATH_CALUDE_circle_area_irrational_when_radius_rational_l2855_285579

/-- The area of a circle is irrational when its radius is rational -/
theorem circle_area_irrational_when_radius_rational :
  ∀ r : ℚ, ∃ A : ℝ, A = π * r^2 ∧ Irrational A :=
sorry

end NUMINAMATH_CALUDE_circle_area_irrational_when_radius_rational_l2855_285579


namespace NUMINAMATH_CALUDE_total_guests_calculation_l2855_285587

/-- Given the number of guests in different age groups, calculate the total number of guests served. -/
theorem total_guests_calculation (adults : ℕ) (h1 : adults = 58) : ∃ (children seniors teenagers toddlers : ℕ),
  children = adults - 35 ∧
  seniors = 2 * children ∧
  teenagers = seniors - 15 ∧
  toddlers = teenagers / 2 ∧
  adults + children + seniors + teenagers + toddlers = 173 := by
  sorry


end NUMINAMATH_CALUDE_total_guests_calculation_l2855_285587


namespace NUMINAMATH_CALUDE_product_sum_relation_l2855_285567

theorem product_sum_relation (a b c : ℚ) 
  (h1 : a * b * c = 2 * (a + b + c) + 14)
  (h2 : b = 8)
  (h3 : c = 5) :
  (c - a)^2 + b = 8513 / 361 := by sorry

end NUMINAMATH_CALUDE_product_sum_relation_l2855_285567


namespace NUMINAMATH_CALUDE_min_absolute_value_complex_l2855_285555

open Complex

theorem min_absolute_value_complex (z : ℂ) :
  (abs (z + I) + abs (z - 2 - I) = 2 * Real.sqrt 2) →
  (∃ (w : ℂ), abs w ≤ abs z ∧ abs (w + I) + abs (w - 2 - I) = 2 * Real.sqrt 2) →
  abs z ≥ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_absolute_value_complex_l2855_285555


namespace NUMINAMATH_CALUDE_meeting_time_prove_meeting_time_l2855_285571

/-- The time it takes for Petya and Vasya to meet under the given conditions -/
theorem meeting_time : ℝ → ℝ → ℝ → Prop :=
  fun (x : ℝ) (v_g : ℝ) (t : ℝ) =>
    x > 0 ∧ v_g > 0 ∧  -- Positive distance and speed
    x = 3 * v_g ∧  -- Petya reaches the bridge in 1 hour
    t = 1 + (2 * x - 2 * v_g) / (2 * v_g) ∧  -- Total time calculation
    t = 2  -- The meeting time is 2 hours

/-- Proof of the meeting time theorem -/
theorem prove_meeting_time : ∃ (x v_g : ℝ), meeting_time x v_g 2 := by
  sorry


end NUMINAMATH_CALUDE_meeting_time_prove_meeting_time_l2855_285571


namespace NUMINAMATH_CALUDE_isosceles_triangle_circumscribed_circle_l2855_285528

/-- Given a circle with radius 3 and an isosceles triangle circumscribed around it with a base angle of 30°, 
    this theorem proves the lengths of the sides of the triangle. -/
theorem isosceles_triangle_circumscribed_circle 
  (r : ℝ) 
  (base_angle : ℝ) 
  (h_r : r = 3) 
  (h_angle : base_angle = 30 * π / 180) : 
  ∃ (equal_side base_side : ℝ),
    equal_side = 4 * Real.sqrt 3 + 6 ∧ 
    base_side = 6 * Real.sqrt 3 + 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_circumscribed_circle_l2855_285528


namespace NUMINAMATH_CALUDE_two_color_theorem_l2855_285510

/-- A type representing a region in the plane --/
def Region : Type := ℕ

/-- A type representing a color (either 0 or 1) --/
def Color : Type := Fin 2

/-- A function that determines if two regions are adjacent --/
def adjacent (n : ℕ) (r1 r2 : Region) : Prop := sorry

/-- A coloring of regions --/
def Coloring (n : ℕ) : Type := Region → Color

/-- A predicate that determines if a coloring is valid --/
def is_valid_coloring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ r1 r2 : Region, adjacent n r1 r2 → c r1 ≠ c r2

/-- The main theorem: there exists a valid two-coloring for any number of circles --/
theorem two_color_theorem (n : ℕ) (h : n ≥ 1) :
  ∃ c : Coloring n, is_valid_coloring n c :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l2855_285510


namespace NUMINAMATH_CALUDE_distance_to_plane_l2855_285541

/-- The distance from a point to a plane defined by three points -/
def distancePointToPlane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  -- Implementation details omitted
  sorry

theorem distance_to_plane :
  let M₀ : ℝ × ℝ × ℝ := (1, -6, -5)
  let M₁ : ℝ × ℝ × ℝ := (-1, 2, -3)
  let M₂ : ℝ × ℝ × ℝ := (4, -1, 0)
  let M₃ : ℝ × ℝ × ℝ := (2, 1, -2)
  distancePointToPlane M₀ M₁ M₂ M₃ = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_plane_l2855_285541


namespace NUMINAMATH_CALUDE_complex_arithmetic_l2855_285517

theorem complex_arithmetic : ((2 : ℂ) + 5*I + (3 : ℂ) - 2*I) - ((1 : ℂ) - 3*I) = (4 : ℂ) + 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l2855_285517


namespace NUMINAMATH_CALUDE_cut_scene_is_six_minutes_l2855_285518

/-- The length of a cut scene from a movie --/
def cut_scene_length (original_length final_length : ℕ) : ℕ :=
  original_length - final_length

/-- Theorem: The length of the cut scene is 6 minutes --/
theorem cut_scene_is_six_minutes :
  let original_length : ℕ := 60  -- One hour in minutes
  let final_length : ℕ := 54
  cut_scene_length original_length final_length = 6 := by
  sorry

#eval cut_scene_length 60 54  -- This should output 6

end NUMINAMATH_CALUDE_cut_scene_is_six_minutes_l2855_285518


namespace NUMINAMATH_CALUDE_students_not_playing_l2855_285522

theorem students_not_playing (total : ℕ) (basketball : ℕ) (volleyball : ℕ) (both : ℕ) : 
  total = 20 ∧ 
  basketball = total / 2 ∧ 
  volleyball = total * 2 / 5 ∧ 
  both = total / 10 → 
  total - (basketball + volleyball - both) = 4 := by
sorry

end NUMINAMATH_CALUDE_students_not_playing_l2855_285522


namespace NUMINAMATH_CALUDE_function_lower_bound_l2855_285558

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 3/4) * Real.exp x - (b * Real.exp x) / (Real.exp x + 1)

theorem function_lower_bound (a : ℝ) :
  (∀ x ∈ Set.Ici (-2 : ℝ), f a 1 x ≥ -5/4) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l2855_285558


namespace NUMINAMATH_CALUDE_complex_sum_on_real_axis_l2855_285588

theorem complex_sum_on_real_axis (a : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := 3 + a * I
  (z₁ + z₂).im = 0 → a = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_on_real_axis_l2855_285588


namespace NUMINAMATH_CALUDE_amy_total_crumbs_l2855_285582

/-- Theorem: Amy's total crumbs given Arthur's total crumbs -/
theorem amy_total_crumbs (c : ℝ) : ℝ := by
  -- Define Arthur's trips and crumbs per trip
  let arthur_trips : ℝ := c / (c / c)
  let arthur_crumbs_per_trip : ℝ := c / arthur_trips

  -- Define Amy's trips and crumbs per trip
  let amy_trips : ℝ := 2 * arthur_trips
  let amy_crumbs_per_trip : ℝ := 1.5 * arthur_crumbs_per_trip

  -- Calculate Amy's total crumbs
  let amy_total : ℝ := amy_trips * amy_crumbs_per_trip

  -- Prove that Amy's total crumbs equals 3c
  sorry

end NUMINAMATH_CALUDE_amy_total_crumbs_l2855_285582


namespace NUMINAMATH_CALUDE_ships_required_equals_round_trip_duration_moscow_astrakhan_ships_required_l2855_285515

/-- Represents the duration of travel and stay in days -/
structure TravelDuration :=
  (moscow_to_astrakhan : ℕ)
  (stay_in_astrakhan : ℕ)
  (astrakhan_to_moscow : ℕ)
  (stay_in_moscow : ℕ)

/-- Calculates the total round trip duration -/
def round_trip_duration (t : TravelDuration) : ℕ :=
  t.moscow_to_astrakhan + t.stay_in_astrakhan + t.astrakhan_to_moscow + t.stay_in_moscow

/-- The number of ships required for continuous daily departures -/
def ships_required (t : TravelDuration) : ℕ :=
  round_trip_duration t

/-- Theorem stating that the number of ships required is equal to the round trip duration -/
theorem ships_required_equals_round_trip_duration (t : TravelDuration) :
  ships_required t = round_trip_duration t := by
  sorry

/-- The specific travel durations given in the problem -/
def moscow_astrakhan_route : TravelDuration :=
  { moscow_to_astrakhan := 4
  , stay_in_astrakhan := 2
  , astrakhan_to_moscow := 5
  , stay_in_moscow := 2 }

/-- Theorem proving that 13 ships are required for the Moscow-Astrakhan route -/
theorem moscow_astrakhan_ships_required :
  ships_required moscow_astrakhan_route = 13 := by
  sorry

end NUMINAMATH_CALUDE_ships_required_equals_round_trip_duration_moscow_astrakhan_ships_required_l2855_285515


namespace NUMINAMATH_CALUDE_interest_rate_difference_l2855_285504

/-- Proves that the difference in interest rates is 5% given the specified conditions -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_difference : ℝ)
  (h1 : principal = 800)
  (h2 : time = 10)
  (h3 : interest_difference = 400)
  : ∃ (r1 r2 : ℝ), r2 - r1 = 5 ∧ 
    principal * r2 * time / 100 = principal * r1 * time / 100 + interest_difference :=
sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l2855_285504


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l2855_285549

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l2855_285549
