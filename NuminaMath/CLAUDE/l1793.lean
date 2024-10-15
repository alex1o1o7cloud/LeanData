import Mathlib

namespace NUMINAMATH_CALUDE_number_of_black_marbles_l1793_179375

/-- Given a bag of marbles with white and black marbles, prove the number of black marbles. -/
theorem number_of_black_marbles
  (total_marbles : ℕ)
  (white_marbles : ℕ)
  (h1 : total_marbles = 37)
  (h2 : white_marbles = 19) :
  total_marbles - white_marbles = 18 :=
by sorry

end NUMINAMATH_CALUDE_number_of_black_marbles_l1793_179375


namespace NUMINAMATH_CALUDE_smallest_q_property_l1793_179310

theorem smallest_q_property : ∃ (q : ℕ), q > 0 ∧ q = 2015 ∧
  (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 →
    ∃ (n : ℤ), (m : ℚ) / 1007 * q < n ∧ n < (m + 1 : ℚ) / 1008 * q) ∧
  (∀ (q' : ℕ), 0 < q' ∧ q' < q →
    ¬(∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 →
      ∃ (n : ℤ), (m : ℚ) / 1007 * q' < n ∧ n < (m + 1 : ℚ) / 1008 * q')) :=
by sorry

end NUMINAMATH_CALUDE_smallest_q_property_l1793_179310


namespace NUMINAMATH_CALUDE_common_intersection_point_l1793_179391

-- Define the circle S
def S : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the tangent line L at A
def L : Set (ℝ × ℝ) := {p | p.1 = 1}

-- Define the function for points X and Y on L
def X (p : ℝ) : ℝ × ℝ := (1, 2*p)
def Y (q : ℝ) : ℝ × ℝ := (1, -2*q)

-- Define the condition for X and Y
def XYCondition (p q c : ℝ) : Prop := p * q = c / 4

-- Define the theorem
theorem common_intersection_point (c : ℝ) (h : c > 0) :
  ∀ (p q : ℝ), p > 0 → q > 0 → XYCondition p q c →
  ∃ (R : ℝ × ℝ), R.1 = (4 - c) / (4 + c) ∧ R.2 = 0 ∧
  (∀ (P Q : ℝ × ℝ), P ∈ S → Q ∈ S →
   (∃ (t : ℝ), P = (1 - t) • B + t • X p) →
   (∃ (s : ℝ), Q = (1 - s) • B + s • Y q) →
   ∃ (k : ℝ), R = (1 - k) • P + k • Q) :=
sorry

end NUMINAMATH_CALUDE_common_intersection_point_l1793_179391


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_800_l1793_179389

theorem greatest_multiple_of_5_and_7_under_800 : 
  ∀ n : ℕ, n % 5 = 0 ∧ n % 7 = 0 ∧ n < 800 → n ≤ 770 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_800_l1793_179389


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1793_179377

theorem solve_exponential_equation :
  ∃ y : ℝ, 4^(3*y) = (64 : ℝ)^(1/3) ∧ y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1793_179377


namespace NUMINAMATH_CALUDE_count_parallel_edges_l1793_179361

structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  distinct : length ≠ width ∧ width ≠ height ∧ length ≠ height

def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 6

theorem count_parallel_edges (prism : RectangularPrism) :
  parallel_edge_pairs prism = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_parallel_edges_l1793_179361


namespace NUMINAMATH_CALUDE_fraction_equality_l1793_179348

theorem fraction_equality (a b : ℚ) (h : (a - b) / b = 2 / 3) : a / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1793_179348


namespace NUMINAMATH_CALUDE_density_of_S_l1793_179363

def S : Set ℚ := {q : ℚ | ∃ (m n : ℕ), q = (m * n : ℚ) / ((m^2 + n^2) : ℚ)}

theorem density_of_S (x y : ℚ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) :
  ∃ z : ℚ, z ∈ S ∧ x < z ∧ z < y := by
  sorry

end NUMINAMATH_CALUDE_density_of_S_l1793_179363


namespace NUMINAMATH_CALUDE_circular_pond_area_l1793_179304

/-- Given a circular pond with a diameter of 20 feet and a line from the midpoint 
    of this diameter to the circumference of 18 feet, prove that the area of the 
    pond is 224π square feet. -/
theorem circular_pond_area (diameter : ℝ) (midpoint_to_circle : ℝ) : 
  diameter = 20 → midpoint_to_circle = 18 → 
  ∃ (radius : ℝ), radius^2 * π = 224 * π := by sorry

end NUMINAMATH_CALUDE_circular_pond_area_l1793_179304


namespace NUMINAMATH_CALUDE_female_officers_count_l1793_179332

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_duty_percentage : ℚ) :
  total_on_duty = 360 →
  female_on_duty_ratio = 1/2 →
  female_duty_percentage = 3/5 →
  (↑total_on_duty * female_on_duty_ratio / female_duty_percentage : ℚ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l1793_179332


namespace NUMINAMATH_CALUDE_min_n_for_sqrt_27n_integer_l1793_179352

theorem min_n_for_sqrt_27n_integer (n : ℕ+) (h : ∃ k : ℕ, k^2 = 27 * n) :
  ∀ m : ℕ+, (∃ j : ℕ, j^2 = 27 * m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_sqrt_27n_integer_l1793_179352


namespace NUMINAMATH_CALUDE_right_triangle_rotation_forms_cone_l1793_179318

/-- A right-angled triangle -/
structure RightTriangle where
  /-- One of the right-angled edges of the triangle -/
  edge : ℝ
  /-- The other right-angled edge of the triangle -/
  base : ℝ
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- Condition for a right-angled triangle -/
  right_angle : edge^2 + base^2 = hypotenuse^2

/-- A solid formed by rotating a plane figure -/
inductive RotatedSolid
  | Cone
  | Cylinder
  | Sphere

/-- Function to determine the solid formed by rotating a right-angled triangle -/
def solidFormedByRotation (triangle : RightTriangle) (rotationAxis : ℝ) : RotatedSolid :=
  sorry

/-- Theorem stating that rotating a right-angled triangle about one of its right-angled edges forms a cone -/
theorem right_triangle_rotation_forms_cone (triangle : RightTriangle) :
  solidFormedByRotation triangle triangle.edge = RotatedSolid.Cone := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_rotation_forms_cone_l1793_179318


namespace NUMINAMATH_CALUDE_monomial_difference_implies_m_pow_n_eq_nine_l1793_179379

-- Define the variables
variable (a b m n : ℕ)

-- Define the condition that the difference is a monomial
def is_monomial_difference : Prop :=
  ∃ (k : ℕ) (c : ℤ), 2 * a * b^(2*m+n) - a^(m-n) * b^8 = c * a^k * b^k

-- State the theorem
theorem monomial_difference_implies_m_pow_n_eq_nine
  (h : is_monomial_difference a b m n) : m^n = 9 := by
  sorry

end NUMINAMATH_CALUDE_monomial_difference_implies_m_pow_n_eq_nine_l1793_179379


namespace NUMINAMATH_CALUDE_final_expression_l1793_179369

theorem final_expression (b : ℚ) : ((3 * b + 6) - 5 * b) / 3 = -2/3 * b + 2 := by
  sorry

end NUMINAMATH_CALUDE_final_expression_l1793_179369


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_36_l1793_179397

theorem ceiling_neg_sqrt_36 : ⌈-Real.sqrt 36⌉ = -6 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_36_l1793_179397


namespace NUMINAMATH_CALUDE_min_odd_integers_l1793_179330

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 28)
  (sum2 : a + b + c + d = 46)
  (sum3 : a + b + c + d + e + f = 66) :
  ∃ (a' b' c' d' e' f' : ℤ), 
    (a' + b' = 28) ∧ 
    (a' + b' + c' + d' = 46) ∧ 
    (a' + b' + c' + d' + e' + f' = 66) ∧
    (Even a') ∧ (Even b') ∧ (Even c') ∧ (Even d') ∧ (Even e') ∧ (Even f') :=
by
  sorry

end NUMINAMATH_CALUDE_min_odd_integers_l1793_179330


namespace NUMINAMATH_CALUDE_max_fourth_root_sum_max_fourth_root_sum_achievable_l1793_179388

theorem max_fourth_root_sum (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a + b + c + d = 1) : 
  (abcd : ℝ)^(1/4) + ((1-a)*(1-b)*(1-c)*(1-d) : ℝ)^(1/4) ≤ 1 :=
by sorry

theorem max_fourth_root_sum_achievable : 
  ∃ (a b c d : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a + b + c + d = 1 ∧ 
    (abcd : ℝ)^(1/4) + ((1-a)*(1-b)*(1-c)*(1-d) : ℝ)^(1/4) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_fourth_root_sum_max_fourth_root_sum_achievable_l1793_179388


namespace NUMINAMATH_CALUDE_triangle_third_height_bound_l1793_179337

theorem triangle_third_height_bound (a b c : ℝ) (ha hb : ℝ) (h : ℝ) : 
  ha = 12 → hb = 20 → 
  a * ha = b * hb → 
  c * h = a * ha → 
  c > a - b → 
  h < 30 := by sorry

end NUMINAMATH_CALUDE_triangle_third_height_bound_l1793_179337


namespace NUMINAMATH_CALUDE_circle_travel_in_triangle_l1793_179365

/-- The distance traveled by the center of a circle rolling inside a triangle -/
def circle_travel_distance (a b c r : ℝ) : ℝ :=
  (a - 2 * r) + (b - 2 * r) + (c - 2 * r)

/-- Theorem: The distance traveled by the center of a circle with radius 2
    rolling inside a 6-8-10 triangle is 8 -/
theorem circle_travel_in_triangle :
  circle_travel_distance 6 8 10 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_travel_in_triangle_l1793_179365


namespace NUMINAMATH_CALUDE_test_scores_l1793_179366

theorem test_scores (scores : Finset ℕ) (petya_score : ℕ) : 
  scores.card = 7317 →
  (∀ (x y : ℕ), x ∈ scores → y ∈ scores → x ≠ y) →
  (∀ (x y z : ℕ), x ∈ scores → y ∈ scores → z ∈ scores → x < y + z) →
  petya_score ∈ scores →
  petya_score > 15 :=
by sorry

end NUMINAMATH_CALUDE_test_scores_l1793_179366


namespace NUMINAMATH_CALUDE_final_area_fraction_l1793_179301

/-- The fraction of area remaining after one iteration -/
def remaining_fraction : ℚ := 8 / 9

/-- The number of iterations -/
def num_iterations : ℕ := 6

/-- The theorem stating the final fraction of area remaining -/
theorem final_area_fraction :
  remaining_fraction ^ num_iterations = 262144 / 531441 := by
  sorry

end NUMINAMATH_CALUDE_final_area_fraction_l1793_179301


namespace NUMINAMATH_CALUDE_average_price_per_book_l1793_179333

theorem average_price_per_book (books_shop1 : ℕ) (price_shop1 : ℕ) 
  (books_shop2 : ℕ) (price_shop2 : ℕ) 
  (h1 : books_shop1 = 65) (h2 : price_shop1 = 1380) 
  (h3 : books_shop2 = 55) (h4 : price_shop2 = 900) : 
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_price_per_book_l1793_179333


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l1793_179322

def off_rack_suit_price : ℝ := 300
def tailored_suit_price (off_rack_price : ℝ) : ℝ := 3 * off_rack_price + 200
def dress_shirt_price : ℝ := 80
def shoes_price : ℝ := 120
def tie_price : ℝ := 40
def discount_rate : ℝ := 0.1
def sales_tax_rate : ℝ := 0.08
def shipping_fee : ℝ := 25

def total_cost : ℝ :=
  let discounted_suit_price := off_rack_suit_price * (1 - discount_rate)
  let suits_cost := off_rack_suit_price + discounted_suit_price
  let tailored_suit_cost := tailored_suit_price off_rack_suit_price
  let accessories_cost := dress_shirt_price + shoes_price + tie_price
  let subtotal := suits_cost + tailored_suit_cost + accessories_cost
  let tax := subtotal * sales_tax_rate
  subtotal + tax + shipping_fee

theorem total_cost_is_correct : total_cost = 2087.80 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l1793_179322


namespace NUMINAMATH_CALUDE_triangle_5_7_14_not_exists_l1793_179321

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if a triangle with given side lengths can exist. -/
def triangle_exists (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that a triangle with side lengths 5, 7, and 14 cannot exist. -/
theorem triangle_5_7_14_not_exists : ¬ triangle_exists 5 7 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_5_7_14_not_exists_l1793_179321


namespace NUMINAMATH_CALUDE_probability_of_square_l1793_179358

/-- The probability of selecting a square from a set of figures -/
theorem probability_of_square (total_figures : ℕ) (square_count : ℕ) 
  (h1 : total_figures = 10) (h2 : square_count = 3) : 
  (square_count : ℚ) / total_figures = 3 / 10 := by
  sorry

#check probability_of_square

end NUMINAMATH_CALUDE_probability_of_square_l1793_179358


namespace NUMINAMATH_CALUDE_quadratic_function_a_range_l1793_179367

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_a_range 
  (a b c : ℝ) 
  (h1 : ∀ x, f a b c x < 0 ↔ x < 1 ∨ x > 3)
  (h2 : ∀ x, f a b c x < 2) :
  -2 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_a_range_l1793_179367


namespace NUMINAMATH_CALUDE_greatest_b_for_no_minus_six_l1793_179376

theorem greatest_b_for_no_minus_six (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -6) ↔ b ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_b_for_no_minus_six_l1793_179376


namespace NUMINAMATH_CALUDE_freshman_class_size_l1793_179329

theorem freshman_class_size :
  ∃! n : ℕ, 0 < n ∧ n < 450 ∧ n % 19 = 18 ∧ n % 17 = 10 ∧ n = 265 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l1793_179329


namespace NUMINAMATH_CALUDE_gcd_18_30_l1793_179383

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l1793_179383


namespace NUMINAMATH_CALUDE_a_profit_share_is_3750_l1793_179343

/-- Calculates the share of profit for an investor in a partnership business -/
def calculate_profit_share (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  (investment_a / (investment_a + investment_b + investment_c)) * total_profit

/-- Theorem: Given the investments and total profit, A's share of the profit is 3750 -/
theorem a_profit_share_is_3750 
  (investment_a : ℚ) 
  (investment_b : ℚ) 
  (investment_c : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_a = 6300)
  (h2 : investment_b = 4200)
  (h3 : investment_c = 10500)
  (h4 : total_profit = 12500) :
  calculate_profit_share investment_a investment_b investment_c total_profit = 3750 := by
  sorry

#eval calculate_profit_share 6300 4200 10500 12500

end NUMINAMATH_CALUDE_a_profit_share_is_3750_l1793_179343


namespace NUMINAMATH_CALUDE_platform_length_l1793_179393

/-- Given a train with speed 54 km/hr passing a platform in 32 seconds
    and passing a man standing on the platform in 20 seconds,
    prove that the length of the platform is 180 meters. -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) :
  train_speed = 54 →
  platform_time = 32 →
  man_time = 20 →
  (train_speed * 1000 / 3600) * platform_time - (train_speed * 1000 / 3600) * man_time = 180 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1793_179393


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1793_179345

theorem right_triangle_hypotenuse (QR RS QS : ℝ) (cos_R : ℝ) : 
  cos_R = 3/5 →  -- Given condition
  RS = 10 →     -- Given condition
  QR = RS * cos_R →  -- Definition of cosine in right triangle
  QS^2 = RS^2 - QR^2 →  -- Pythagorean theorem
  QS = 8 :=  -- Conclusion to prove
by sorry  -- Proof omitted

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1793_179345


namespace NUMINAMATH_CALUDE_sniper_B_wins_l1793_179390

/-- Represents the probabilities of scoring 1, 2, and 3 points for a sniper -/
structure SniperProbabilities where
  one : Real
  two : Real
  three : Real
  sum_to_one : one + two + three = 1
  non_negative : one ≥ 0 ∧ two ≥ 0 ∧ three ≥ 0

/-- Calculates the expected score for a sniper given their probabilities -/
def expectedScore (p : SniperProbabilities) : Real :=
  1 * p.one + 2 * p.two + 3 * p.three

/-- Sniper A's probabilities -/
def sniperA : SniperProbabilities where
  one := 0.4
  two := 0.1
  three := 0.5
  sum_to_one := by sorry
  non_negative := by sorry

/-- Sniper B's probabilities -/
def sniperB : SniperProbabilities where
  one := 0.1
  two := 0.6
  three := 0.3
  sum_to_one := by sorry
  non_negative := by sorry

/-- Theorem stating that Sniper B has a higher expected score than Sniper A -/
theorem sniper_B_wins : expectedScore sniperB > expectedScore sniperA := by
  sorry

end NUMINAMATH_CALUDE_sniper_B_wins_l1793_179390


namespace NUMINAMATH_CALUDE_lcm_of_36_and_12_l1793_179382

theorem lcm_of_36_and_12 (a b : ℕ+) (h1 : a = 36) (h2 : b = 12) (h3 : Nat.gcd a b = 8) :
  Nat.lcm a b = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_12_l1793_179382


namespace NUMINAMATH_CALUDE_quadratic_roots_l1793_179360

/-- A quadratic function f(x) = x^2 - px + q -/
def f (p q x : ℝ) : ℝ := x^2 - p*x + q

/-- Theorem: If f(p + q) = 0 and f(p - q) = 0, then either q = 0 (and p can be any value) or (p, q) = (0, -1) -/
theorem quadratic_roots (p q : ℝ) : 
  f p q (p + q) = 0 ∧ f p q (p - q) = 0 → 
  (q = 0 ∨ (p = 0 ∧ q = -1)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1793_179360


namespace NUMINAMATH_CALUDE_decimalRep_periodic_first_seven_digits_digit_150_l1793_179387

/-- The decimal representation of 17/70 as a sequence of digits after the decimal point -/
def decimalRep : ℕ → ℕ := sorry

/-- The decimal representation of 17/70 is periodic with period 7 -/
theorem decimalRep_periodic : ∀ n : ℕ, decimalRep (n + 7) = decimalRep n := sorry

/-- The first 7 digits of the decimal representation of 17/70 -/
theorem first_seven_digits : 
  (decimalRep 0, decimalRep 1, decimalRep 2, decimalRep 3, decimalRep 4, decimalRep 5, decimalRep 6) 
  = (2, 4, 2, 8, 5, 7, 1) := sorry

/-- The 150th digit after the decimal point in the decimal representation of 17/70 is 2 -/
theorem digit_150 : decimalRep 149 = 2 := sorry

end NUMINAMATH_CALUDE_decimalRep_periodic_first_seven_digits_digit_150_l1793_179387


namespace NUMINAMATH_CALUDE_missing_sale_is_1000_l1793_179338

/-- Calculates the missing sale amount given the sales for 5 months and the average sale for 6 months -/
def calculate_missing_sale (sale1 sale2 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Theorem stating that given the specific sales and average, the missing sale must be 1000 -/
theorem missing_sale_is_1000 :
  calculate_missing_sale 800 900 700 800 900 850 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_missing_sale_is_1000_l1793_179338


namespace NUMINAMATH_CALUDE_point_A_distance_theorem_l1793_179398

-- Define the point A on the number line
def A : ℝ → ℝ := λ a ↦ 2 * a + 1

-- Define the distance function from a point to the origin
def distance_to_origin (x : ℝ) : ℝ := |x|

-- Theorem statement
theorem point_A_distance_theorem :
  ∀ a : ℝ, distance_to_origin (A a) = 3 → a = -2 ∨ a = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_point_A_distance_theorem_l1793_179398


namespace NUMINAMATH_CALUDE_third_quadrant_angle_tangent_l1793_179354

theorem third_quadrant_angle_tangent (α β : Real) : 
  (2 * Real.pi - Real.pi < α) ∧ (α < 2 * Real.pi - Real.pi/2) →
  (Real.sin (α + β) * Real.cos β - Real.sin β * Real.cos (α + β) = -12/13) →
  Real.tan (α/2) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_third_quadrant_angle_tangent_l1793_179354


namespace NUMINAMATH_CALUDE_books_read_proof_l1793_179385

def total_books (megan_books kelcie_books greg_books : ℕ) : ℕ :=
  megan_books + kelcie_books + greg_books

theorem books_read_proof (megan_books : ℕ) 
  (h1 : megan_books = 32)
  (h2 : ∃ kelcie_books : ℕ, kelcie_books = megan_books / 4)
  (h3 : ∃ greg_books : ℕ, greg_books = 2 * (megan_books / 4) + 9) :
  ∃ total : ℕ, total_books megan_books (megan_books / 4) (2 * (megan_books / 4) + 9) = 65 := by
  sorry

end NUMINAMATH_CALUDE_books_read_proof_l1793_179385


namespace NUMINAMATH_CALUDE_ticket_sales_revenue_ticket_sales_problem_l1793_179356

theorem ticket_sales_revenue 
  (student_price : ℕ) 
  (general_price : ℕ) 
  (total_tickets : ℕ) 
  (general_tickets : ℕ) : ℕ :=
  let student_tickets := total_tickets - general_tickets
  let student_revenue := student_tickets * student_price
  let general_revenue := general_tickets * general_price
  student_revenue + general_revenue

theorem ticket_sales_problem : 
  ticket_sales_revenue 4 6 525 388 = 2876 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_revenue_ticket_sales_problem_l1793_179356


namespace NUMINAMATH_CALUDE_circle_area_square_gt_ngon_areas_product_l1793_179306

/-- Given a circle and two regular n-gons, one inscribed and one circumscribed,
    prove that the square of the circle's area is greater than the product of the n-gons' areas. -/
theorem circle_area_square_gt_ngon_areas_product (n : ℕ) (S S₁ S₂ : ℝ) 
    (h_n : n ≥ 3)
    (h_S : S > 0)
    (h_S₁ : S₁ > 0)
    (h_S₂ : S₂ > 0)
    (h_inscribed : S₁ = (n / 2) * S * Real.sin (2 * π / n))
    (h_circumscribed : S₂ = (n / 2) * S * Real.tan (π / n)) :
  S^2 > S₁ * S₂ := by
  sorry

end NUMINAMATH_CALUDE_circle_area_square_gt_ngon_areas_product_l1793_179306


namespace NUMINAMATH_CALUDE_min_value_of_f_range_of_t_l1793_179373

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 4|

-- Theorem for the minimum value of f
theorem min_value_of_f : ∀ x : ℝ, f x ≥ 6 ∧ ∃ x₀ : ℝ, f x₀ = 6 := by sorry

-- Define the set A
def A (t : ℝ) : Set ℝ := {x | f x ≤ t^2 - t}

-- Define the set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}

-- Theorem for the range of t
theorem range_of_t : ∀ t : ℝ, (A t ∩ B).Nonempty ↔ t ≤ -2 ∨ t ≥ 3 := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_range_of_t_l1793_179373


namespace NUMINAMATH_CALUDE_min_disks_is_twelve_l1793_179335

/-- Represents the number of files of each size --/
structure FileCount where
  large : Nat  -- 0.85 MB files
  medium : Nat -- 0.65 MB files
  small : Nat  -- 0.5 MB files

/-- Represents the constraints of the problem --/
structure DiskProblem where
  totalFiles : Nat
  diskCapacity : Float
  maxFilesPerDisk : Nat
  fileSizes : FileCount
  largeSizeMB : Float
  mediumSizeMB : Float
  smallSizeMB : Float

def problem : DiskProblem := {
  totalFiles := 35,
  diskCapacity := 1.44,
  maxFilesPerDisk := 4,
  fileSizes := { large := 5, medium := 15, small := 15 },
  largeSizeMB := 0.85,
  mediumSizeMB := 0.65,
  smallSizeMB := 0.5
}

/-- Calculates the minimum number of disks required --/
def minDisksRequired (p : DiskProblem) : Nat :=
  sorry -- Proof goes here

theorem min_disks_is_twelve : minDisksRequired problem = 12 := by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_min_disks_is_twelve_l1793_179335


namespace NUMINAMATH_CALUDE_inverse_contrapositive_relation_l1793_179314

theorem inverse_contrapositive_relation (p q r : Prop) :
  (¬p ↔ q) →  -- inverse of p is q
  ((¬p ↔ r) ↔ p) →  -- contrapositive of p is r
  (q ↔ ¬r) :=  -- q and r are negations of each other
by sorry

end NUMINAMATH_CALUDE_inverse_contrapositive_relation_l1793_179314


namespace NUMINAMATH_CALUDE_square_property_l1793_179326

theorem square_property (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_property_l1793_179326


namespace NUMINAMATH_CALUDE_sector_area_with_diameter_4_and_angle_90_l1793_179334

theorem sector_area_with_diameter_4_and_angle_90 (π : Real) :
  let diameter : Real := 4
  let centralAngle : Real := 90
  let radius : Real := diameter / 2
  let sectorArea : Real := (centralAngle / 360) * π * radius^2
  sectorArea = π := by sorry

end NUMINAMATH_CALUDE_sector_area_with_diameter_4_and_angle_90_l1793_179334


namespace NUMINAMATH_CALUDE_congruence_problem_l1793_179386

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 127 ∧ (126 * n) % 127 = 103 % 127 → n % 127 = 24 % 127 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1793_179386


namespace NUMINAMATH_CALUDE_lily_catches_mary_l1793_179350

/-- Mary's walking speed in miles per hour -/
def mary_speed : ℝ := 4

/-- Lily's walking speed in miles per hour -/
def lily_speed : ℝ := 6

/-- Initial distance between Mary and Lily in miles -/
def initial_distance : ℝ := 2

/-- Time in minutes for Lily to catch up to Mary -/
def catch_up_time : ℝ := 60

theorem lily_catches_mary : 
  (lily_speed - mary_speed) * catch_up_time / 60 = initial_distance := by
  sorry

end NUMINAMATH_CALUDE_lily_catches_mary_l1793_179350


namespace NUMINAMATH_CALUDE_students_allowance_l1793_179368

theorem students_allowance (A : ℚ) : 
  (A > 0) →
  (3 / 5 * A + 1 / 3 * (2 / 5 * A) + 0.4 = A) →
  A = 1.5 := by
sorry

end NUMINAMATH_CALUDE_students_allowance_l1793_179368


namespace NUMINAMATH_CALUDE_b_95_mod_49_l1793_179303

def b (n : ℕ) : ℕ := 5^n + 7^n

theorem b_95_mod_49 : b 95 ≡ 42 [ZMOD 49] := by sorry

end NUMINAMATH_CALUDE_b_95_mod_49_l1793_179303


namespace NUMINAMATH_CALUDE_eggs_per_group_l1793_179394

/-- Given 9 eggs split into 3 groups, prove that there are 3 eggs in each group. -/
theorem eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) (h1 : total_eggs = 9) (h2 : num_groups = 3) :
  total_eggs / num_groups = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_group_l1793_179394


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l1793_179370

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℂ, x^2 + 4*x + k = 0 ∧ x = -2 + 3*I) → k = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l1793_179370


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l1793_179396

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_sum_factorial (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  factorial hundreds + factorial tens + factorial ones

theorem unique_three_digit_factorial_sum : ∀ n : ℕ, 
  100 ≤ n ∧ n ≤ 999 → (n = digit_sum_factorial n ↔ n = 145) :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l1793_179396


namespace NUMINAMATH_CALUDE_games_in_division_l1793_179344

/-- Represents a baseball league with the given conditions -/
structure BaseballLeague where
  P : ℕ  -- Number of games played against each team in own division
  Q : ℕ  -- Number of games played against each team in other divisions
  p_gt_3q : P > 3 * Q
  q_gt_3 : Q > 3
  total_games : 2 * P + 6 * Q = 78

/-- Theorem stating that each team plays 54 games within its own division -/
theorem games_in_division (league : BaseballLeague) : 2 * league.P = 54 := by
  sorry

end NUMINAMATH_CALUDE_games_in_division_l1793_179344


namespace NUMINAMATH_CALUDE_mn_value_l1793_179319

theorem mn_value (m n : ℕ+) (h : m.val^4 - n.val^4 = 3439) : m.val * n.val = 90 := by
  sorry

end NUMINAMATH_CALUDE_mn_value_l1793_179319


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l1793_179316

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 2) :
  ∃ (m : ℝ), m = 24 / 11 ∧ ∀ (a b c : ℝ), a + b + c = 2 → 2 * a^2 + 3 * b^2 + c^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l1793_179316


namespace NUMINAMATH_CALUDE_sequence_sum_proof_l1793_179381

def sequence_sum (n : ℕ) : ℚ := -(n + 1 : ℚ) / (n + 2 : ℚ)

theorem sequence_sum_proof (n : ℕ) :
  let a : ℕ → ℚ := λ k => if k = 1 then -2/3 else sequence_sum k - sequence_sum (k-1)
  let S : ℕ → ℚ := sequence_sum
  (∀ k : ℕ, k ≥ 2 → S k + 1 / S k + 2 = a k) →
  S n = sequence_sum n :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_proof_l1793_179381


namespace NUMINAMATH_CALUDE_function_arithmetic_l1793_179339

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x < y → f x < f y

theorem function_arithmetic (f : ℕ → ℕ) 
  (h_increasing : StrictlyIncreasing f)
  (h_two : f 2 = 2)
  (h_coprime : ∀ m n : ℕ, Nat.Coprime m n → f (m * n) = f m * f n) :
  ∀ n : ℕ, f n = n :=
sorry

end NUMINAMATH_CALUDE_function_arithmetic_l1793_179339


namespace NUMINAMATH_CALUDE_no_solution_exists_l1793_179355

theorem no_solution_exists : ¬ ∃ (a : ℝ), 
  ({0, 1} : Set ℝ) ∩ ({11 - a, Real.log a, 2^a, a} : Set ℝ) = {1} := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1793_179355


namespace NUMINAMATH_CALUDE_bird_count_theorem_l1793_179359

theorem bird_count_theorem (initial_parrots : ℕ) (remaining_parrots : ℕ) 
  (remaining_crows : ℕ) (remaining_pigeons : ℕ) : 
  initial_parrots = 15 →
  remaining_parrots = 5 →
  remaining_crows = 3 →
  remaining_pigeons = 2 →
  ∃ (flew_away : ℕ), 
    flew_away = initial_parrots - remaining_parrots ∧
    initial_parrots + (flew_away + remaining_crows) + (flew_away + remaining_pigeons) = 40 :=
by sorry

end NUMINAMATH_CALUDE_bird_count_theorem_l1793_179359


namespace NUMINAMATH_CALUDE_quadratic_roots_l1793_179364

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) 
  (h1 : a + b + c = 0) (h2 : 4*a - 2*b + c = 0) : 
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ 
  (∀ z : ℝ, a*z^2 + b*z + c = 0 ↔ z = x ∨ z = y) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1793_179364


namespace NUMINAMATH_CALUDE_min_type_c_cards_l1793_179340

/-- Represents the number of cards sold of each type -/
structure CardSales where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total number of cards sold -/
def total_cards (sales : CardSales) : ℕ :=
  sales.a + sales.b + sales.c

/-- Calculates the total income from card sales -/
def total_income (sales : CardSales) : ℚ :=
  0.5 * sales.a + 1 * sales.b + 2.5 * sales.c

/-- Theorem stating the minimum number of type C cards sold -/
theorem min_type_c_cards (sales : CardSales) 
  (h1 : total_cards sales = 150)
  (h2 : total_income sales = 180) :
  sales.c ≥ 20 := by
  sorry

#check min_type_c_cards

end NUMINAMATH_CALUDE_min_type_c_cards_l1793_179340


namespace NUMINAMATH_CALUDE_highest_power_of_two_and_three_l1793_179374

def n : ℤ := 15^4 - 11^4

theorem highest_power_of_two_and_three (n : ℤ) (h : n = 15^4 - 11^4) :
  (∃ m : ℕ, 2^4 * m = n ∧ ¬(∃ k : ℕ, 2^5 * k = n)) ∧
  (∃ m : ℕ, 3^0 * m = n ∧ ¬(∃ k : ℕ, 3^1 * k = n)) :=
sorry

end NUMINAMATH_CALUDE_highest_power_of_two_and_three_l1793_179374


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l1793_179305

theorem crazy_silly_school_books (movies : ℕ) (books : ℕ) 
  (h1 : movies = 14) 
  (h2 : books = movies + 1) : 
  books = 15 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l1793_179305


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1793_179317

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def isPerpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -2)
  isPerpendicular a b → x = 4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1793_179317


namespace NUMINAMATH_CALUDE_problem_solution_l1793_179351

theorem problem_solution : 
  (∃ x : ℝ, x^2 = 6) ∧ (∃ y : ℝ, y^2 = 2) ∧ (∃ z : ℝ, z^2 = 27) ∧ (∃ w : ℝ, w^2 = 9) ∧ (∃ v : ℝ, v^2 = 1/3) →
  (∃ a b : ℝ, 
    (a^2 = 6 ∧ b^2 = 2 ∧ a * b + Real.sqrt 27 / Real.sqrt 9 - Real.sqrt (1/3) = 8 * Real.sqrt 3 / 3) ∧
    ((Real.sqrt 5 - 1)^2 - (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 5 - 2 * Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1793_179351


namespace NUMINAMATH_CALUDE_range_of_a_l1793_179372

theorem range_of_a (a : ℝ) : 
  (a > 0) → 
  (∀ x : ℝ, ((x - 3*a) * (x - a) < 0) → 
    ¬(x^2 - 3*x ≤ 0 ∧ x^2 - x - 2 > 0)) ∧ 
  (∃ x : ℝ, ¬(x^2 - 3*x ≤ 0 ∧ x^2 - x - 2 > 0) ∧ 
    ¬((x - 3*a) * (x - a) < 0)) ↔ 
  (0 < a ∧ a ≤ 2/3) ∨ a ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1793_179372


namespace NUMINAMATH_CALUDE_ball_count_theorem_l1793_179302

/-- Represents the count of balls of each color in a jar. -/
structure BallCount where
  white : ℕ
  red : ℕ
  blue : ℕ

/-- Checks if the given ball count satisfies the 4:3:2 ratio. -/
def satisfiesRatio (bc : BallCount) : Prop :=
  3 * bc.white = 4 * bc.red ∧ 2 * bc.white = 4 * bc.blue

theorem ball_count_theorem (bc : BallCount) 
    (h_ratio : satisfiesRatio bc) (h_white : bc.white = 20) : 
    bc.red = 15 ∧ bc.blue = 10 := by
  sorry

#check ball_count_theorem

end NUMINAMATH_CALUDE_ball_count_theorem_l1793_179302


namespace NUMINAMATH_CALUDE_quadratic_polynomial_from_sum_and_product_l1793_179308

theorem quadratic_polynomial_from_sum_and_product (s r : ℝ) :
  ∃ (a b : ℝ), a + b = s ∧ a * b = r^3 →
  ∀ x : ℝ, (x - a) * (x - b) = x^2 - s*x + r^3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_from_sum_and_product_l1793_179308


namespace NUMINAMATH_CALUDE_triangle_kp_r3_bound_l1793_179346

/-- For any triangle with circumradius R, perimeter P, and area K, KP/R³ ≤ 27/4 -/
theorem triangle_kp_r3_bound (R P K : ℝ) (hR : R > 0) (hP : P > 0) (hK : K > 0) :
  K * P / R^3 ≤ 27 / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_kp_r3_bound_l1793_179346


namespace NUMINAMATH_CALUDE_best_play_win_probability_correct_l1793_179399

/-- The probability of the best play winning in a contest where 2m jurors are randomly selected from 2n moms. -/
def best_play_win_probability (n m : ℕ) : ℚ :=
  let C := fun (n k : ℕ) => Nat.choose n k
  1 / (C (2*n) n * C (2*n) (2*m)) *
  (Finset.sum (Finset.range (2*m + 1)) (fun q =>
    C n q * C n (2*m - q) *
    (Finset.sum (Finset.range (min q (m-1) + 1)) (fun t =>
      C q t * C (2*n - q) (n - t)))))

/-- Theorem stating the probability of the best play winning. -/
theorem best_play_win_probability_correct (n m : ℕ) (h : 2*m ≤ n) :
  best_play_win_probability n m = 
  (1 / (Nat.choose (2*n) n * Nat.choose (2*n) (2*m))) *
  (Finset.sum (Finset.range (2*m + 1)) (fun q =>
    Nat.choose n q * Nat.choose n (2*m - q) *
    (Finset.sum (Finset.range (min q (m-1) + 1)) (fun t =>
      Nat.choose q t * Nat.choose (2*n - q) (n - t))))) :=
by sorry

end NUMINAMATH_CALUDE_best_play_win_probability_correct_l1793_179399


namespace NUMINAMATH_CALUDE_sequence_general_term_l1793_179347

/-- Given a sequence a_n with sum S_n satisfying S_n = 3 - 2a_n,
    prove that the general term of a_n is (2/3)^(n-1) -/
theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ)
    (h : ∀ n, S n = 3 - 2 * a n) :
  ∀ n, a n = (2/3)^(n-1) := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1793_179347


namespace NUMINAMATH_CALUDE_correct_pages_per_booklet_l1793_179328

/-- The number of booklets in Jack's short story section -/
def num_booklets : ℕ := 49

/-- The total number of pages in all booklets -/
def total_pages : ℕ := 441

/-- The number of pages per booklet -/
def pages_per_booklet : ℕ := total_pages / num_booklets

theorem correct_pages_per_booklet : pages_per_booklet = 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_pages_per_booklet_l1793_179328


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l1793_179384

theorem polar_to_rectangular (x y ρ : ℝ) :
  ρ = 2 → x^2 + y^2 = ρ^2 → x^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l1793_179384


namespace NUMINAMATH_CALUDE_jennifer_apples_l1793_179331

def initial_apples : ℕ := 7
def hours : ℕ := 3
def multiply_factor : ℕ := 3
def additional_apples : ℕ := 74

def apples_after_tripling (start : ℕ) (hours : ℕ) (factor : ℕ) : ℕ :=
  start * (factor ^ hours)

theorem jennifer_apples : 
  apples_after_tripling initial_apples hours multiply_factor + additional_apples = 263 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_apples_l1793_179331


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l1793_179378

theorem largest_solution_of_equation (a b c d : ℤ) (x : ℝ) :
  (4 * x / 5 - 2 = 5 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (∀ y, (4 * y / 5 - 2 = 5 / y) → y ≤ x) →
  (x = (5 + 5 * Real.sqrt 5) / 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l1793_179378


namespace NUMINAMATH_CALUDE_orange_segments_total_l1793_179371

/-- Represents the number of orange segments each animal received -/
structure OrangeDistribution where
  siskin : ℕ
  hedgehog : ℕ
  beaver : ℕ

/-- Defines the conditions of the orange distribution problem -/
def validDistribution (d : OrangeDistribution) : Prop :=
  d.hedgehog = 2 * d.siskin ∧
  d.beaver = 5 * d.siskin ∧
  d.beaver = d.siskin + 8

/-- The theorem stating that the total number of orange segments is 16 -/
theorem orange_segments_total (d : OrangeDistribution) 
  (h : validDistribution d) : d.siskin + d.hedgehog + d.beaver = 16 := by
  sorry

#check orange_segments_total

end NUMINAMATH_CALUDE_orange_segments_total_l1793_179371


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l1793_179313

/-- The probability of getting exactly k successes in n independent trials,
    where each trial has a success probability of p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 positive answers out of 8 questions
    with the Magic 8 Ball, where each question has a 2/5 chance of a positive answer -/
theorem magic_8_ball_probability : 
  binomial_probability 8 3 (2/5) = 108864/390625 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l1793_179313


namespace NUMINAMATH_CALUDE_unique_divisor_sum_product_l1793_179395

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

def product_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem unique_divisor_sum_product :
  ∃! P : ℕ, P > 0 ∧ sum_of_divisors P = 2 * P ∧ product_of_divisors P = P ^ 2 ∧ P = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_sum_product_l1793_179395


namespace NUMINAMATH_CALUDE_increasing_function_conditions_l1793_179312

-- Define the piecewise function f
noncomputable def f (a b : ℝ) : ℝ → ℝ := fun x =>
  if x ≥ 0 then x^2 + 3 else a*x + b

-- State the theorem
theorem increasing_function_conditions (a b : ℝ) :
  (∀ x y : ℝ, x < y → f a b x < f a b y) →
  (a > 0 ∧ b ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_conditions_l1793_179312


namespace NUMINAMATH_CALUDE_sales_tax_reduction_difference_l1793_179307

/-- The difference in sales tax between two rates for a given market price -/
def sales_tax_difference (original_rate new_rate market_price : ℝ) : ℝ :=
  market_price * original_rate - market_price * new_rate

/-- Theorem stating the difference in sales tax for the given problem -/
theorem sales_tax_reduction_difference :
  let original_rate : ℝ := 3.5 / 100
  let new_rate : ℝ := 10 / 3 / 100
  let market_price : ℝ := 7800
  abs (sales_tax_difference original_rate new_rate market_price - 13.26) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_sales_tax_reduction_difference_l1793_179307


namespace NUMINAMATH_CALUDE_sphere_volume_condition_l1793_179336

theorem sphere_volume_condition (R V : ℝ) : 
  (V = (4 / 3) * π * R^3) → (R > Real.sqrt 10 → V > 36 * π) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_condition_l1793_179336


namespace NUMINAMATH_CALUDE_polar_curve_is_circle_l1793_179315

/-- The curve defined by the polar equation r = 1 / (sin θ + cos θ) is a circle. -/
theorem polar_curve_is_circle :
  ∀ θ : ℝ, ∃ r : ℝ, r = 1 / (Real.sin θ + Real.cos θ) → ∃ c x₀ y₀ : ℝ, 
    (r * Real.cos θ - x₀)^2 + (r * Real.sin θ - y₀)^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_polar_curve_is_circle_l1793_179315


namespace NUMINAMATH_CALUDE_cost_per_shot_l1793_179349

def number_of_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def shots_per_puppy : ℕ := 2
def total_cost : ℕ := 120

theorem cost_per_shot :
  (total_cost : ℚ) / (number_of_dogs * puppies_per_dog * shots_per_puppy) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_shot_l1793_179349


namespace NUMINAMATH_CALUDE_arithmetic_expression_proof_l1793_179325

theorem arithmetic_expression_proof : (6 + 6 * 3 - 3) / 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_proof_l1793_179325


namespace NUMINAMATH_CALUDE_slope_condition_implies_coefficient_bound_l1793_179353

/-- Given two distinct points on a linear function, if the slope between them is negative, then the coefficient of x in the function is less than 1. -/
theorem slope_condition_implies_coefficient_bound
  (x₁ x₂ y₁ y₂ a : ℝ)
  (h_distinct : x₁ ≠ x₂)
  (h_on_graph₁ : y₁ = (a - 1) * x₁ + 1)
  (h_on_graph₂ : y₂ = (a - 1) * x₂ + 1)
  (h_slope_neg : (y₁ - y₂) / (x₁ - x₂) < 0) :
  a < 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_condition_implies_coefficient_bound_l1793_179353


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_l1793_179309

theorem simplify_sqrt_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (Real.sqrt (3 * a)) / (Real.sqrt (12 * a * b)) = (Real.sqrt b) / (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_l1793_179309


namespace NUMINAMATH_CALUDE_staff_dress_price_l1793_179362

/-- The final price of a dress for staff members after discounts -/
theorem staff_dress_price (d : ℝ) : 
  let initial_discount : ℝ := 0.65
  let staff_discount : ℝ := 0.60
  let price_after_initial_discount : ℝ := d * (1 - initial_discount)
  let final_price : ℝ := price_after_initial_discount * (1 - staff_discount)
  final_price = d * 0.14 := by
sorry

end NUMINAMATH_CALUDE_staff_dress_price_l1793_179362


namespace NUMINAMATH_CALUDE_line_segment_parameterization_l1793_179341

theorem line_segment_parameterization (m n p q : ℝ) : 
  (∃ (t : ℝ), -1 ≤ t ∧ t ≤ 1 ∧ 
    1 = m * (-1) + n ∧ 
    -3 = p * (-1) + q ∧
    6 = m * 1 + n ∧ 
    5 = p * 1 + q) →
  m^2 + n^2 + p^2 + q^2 = 99 := by
sorry

end NUMINAMATH_CALUDE_line_segment_parameterization_l1793_179341


namespace NUMINAMATH_CALUDE_green_shirt_pairs_green_green_pairs_count_l1793_179320

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) : ℕ :=
  let green_green_pairs := 
    have _ : total_students = 144 := by sorry
    have _ : red_students = 63 := by sorry
    have _ : green_students = 81 := by sorry
    have _ : total_pairs = 72 := by sorry
    have _ : red_red_pairs = 27 := by sorry
    have _ : total_students = red_students + green_students := by sorry
    have _ : red_students * 2 ≥ red_red_pairs * 2 := by sorry
    let red_in_mixed_pairs := red_students - (red_red_pairs * 2)
    let remaining_green := green_students - red_in_mixed_pairs
    remaining_green / 2
  green_green_pairs

theorem green_green_pairs_count : 
  green_shirt_pairs 144 63 81 72 27 = 36 := by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_green_green_pairs_count_l1793_179320


namespace NUMINAMATH_CALUDE_banana_groups_count_l1793_179311

def total_bananas : ℕ := 290
def bananas_per_group : ℕ := 145

theorem banana_groups_count : total_bananas / bananas_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_banana_groups_count_l1793_179311


namespace NUMINAMATH_CALUDE_brianne_alex_yard_ratio_l1793_179342

/-- Proves that Brianne's yard is 6 times larger than Alex's yard given the conditions -/
theorem brianne_alex_yard_ratio :
  ∀ (derrick_yard alex_yard brianne_yard : ℝ),
  derrick_yard = 10 →
  alex_yard = derrick_yard / 2 →
  brianne_yard = 30 →
  brianne_yard / alex_yard = 6 := by
sorry

end NUMINAMATH_CALUDE_brianne_alex_yard_ratio_l1793_179342


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l1793_179300

/-- Represents a 9x9 grid filled with numbers 1 to 81 in row-wise order -/
def Grid := Fin 9 → Fin 9 → Fin 81

/-- The value at position (i, j) in the grid -/
def gridValue (i j : Fin 9) : Fin 81 :=
  ⟨i.val * 9 + j.val + 1, by sorry⟩

/-- The sum of the corner values in the grid -/
def cornerSum (g : Grid) : ℕ :=
  (g 0 0).val + (g 0 8).val + (g 8 0).val + (g 8 8).val

/-- Theorem stating that the sum of corner values in the defined grid is 164 -/
theorem corner_sum_is_164 :
  ∃ (g : Grid), cornerSum g = 164 :=
by sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l1793_179300


namespace NUMINAMATH_CALUDE_store_sales_growth_rate_l1793_179357

theorem store_sales_growth_rate 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (months : ℕ) 
  (h1 : initial_sales = 20000)
  (h2 : final_sales = 45000)
  (h3 : months = 2) :
  ∃ (growth_rate : ℝ), 
    growth_rate = 0.5 ∧ 
    final_sales = initial_sales * (1 + growth_rate) ^ months :=
sorry

end NUMINAMATH_CALUDE_store_sales_growth_rate_l1793_179357


namespace NUMINAMATH_CALUDE_rugby_banquet_min_guests_l1793_179327

/-- The minimum number of guests at a banquet given the total food consumed and maximum individual consumption --/
def min_guests (total_food : ℕ) (max_individual_consumption : ℕ) : ℕ :=
  (total_food + max_individual_consumption - 1) / max_individual_consumption

/-- Theorem stating the minimum number of guests at the rugby banquet --/
theorem rugby_banquet_min_guests :
  min_guests 4875 3 = 1625 := by
  sorry

end NUMINAMATH_CALUDE_rugby_banquet_min_guests_l1793_179327


namespace NUMINAMATH_CALUDE_average_speed_two_walks_l1793_179324

theorem average_speed_two_walks 
  (v₁ v₂ t₁ t₂ : ℝ) 
  (h₁ : t₁ > 0) 
  (h₂ : t₂ > 0) :
  let d₁ := v₁ * t₁
  let d₂ := v₂ * t₂
  let total_distance := d₁ + d₂
  let total_time := t₁ + t₂
  (total_distance / total_time) = (v₁ * t₁ + v₂ * t₂) / (t₁ + t₂) := by
sorry

end NUMINAMATH_CALUDE_average_speed_two_walks_l1793_179324


namespace NUMINAMATH_CALUDE_certain_number_power_l1793_179392

theorem certain_number_power (k : ℕ) (h : k = 11) :
  (1/2)^22 * (1/81)^k = (1/354294)^22 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_power_l1793_179392


namespace NUMINAMATH_CALUDE_no_integer_roots_l1793_179323

/-- Polynomial P(x) = x^2019 + 2x^2018 + 3x^2017 + ... + 2019x + 2020 -/
def P (x : ℤ) : ℤ := 
  (Finset.range 2020).sum (fun i => (i + 1) * x^(2019 - i))

theorem no_integer_roots : ∀ x : ℤ, P x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l1793_179323


namespace NUMINAMATH_CALUDE_algebraic_expression_evaluation_l1793_179380

theorem algebraic_expression_evaluation (a b : ℤ) (h1 : a = 1) (h2 : b = -1) :
  a + 2*b + 2*(a + 2*b) + 1 = -2 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_evaluation_l1793_179380
