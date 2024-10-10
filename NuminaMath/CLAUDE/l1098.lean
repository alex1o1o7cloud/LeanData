import Mathlib

namespace intersection_triangle_is_right_angled_l1098_109846

/-- An ellipse with equation x²/m + y² = 1, where m > 1 -/
structure Ellipse where
  m : ℝ
  h_m : m > 1

/-- A hyperbola with equation x²/n - y² = 1, where n > 0 -/
structure Hyperbola where
  n : ℝ
  h_n : n > 0

/-- Two points representing the foci -/
structure Foci where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point representing the intersection of the ellipse and hyperbola -/
def IntersectionPoint (e : Ellipse) (h : Hyperbola) := ℝ × ℝ

/-- Theorem stating that the triangle formed by the foci and intersection point is right-angled -/
theorem intersection_triangle_is_right_angled
  (e : Ellipse) (h : Hyperbola) (f : Foci) (P : IntersectionPoint e h)
  (h_same_foci : e.m - 1 = h.n + 1) :
  -- The triangle F₁PF₂ is right-angled
  ∃ (x y : ℝ), x^2 + y^2 = (f.F₁.1 - f.F₂.1)^2 + (f.F₁.2 - f.F₂.2)^2 :=
sorry

end intersection_triangle_is_right_angled_l1098_109846


namespace quadrilateral_area_main_theorem_l1098_109804

/-- A line with slope -1 intersecting positive x and y axes -/
structure Line1 where
  slope : ℝ
  xIntercept : ℝ
  yIntercept : ℝ

/-- A line passing through (10,0) and intersecting y-axis -/
structure Line2 where
  xIntercept : ℝ
  yIntercept : ℝ

/-- The intersection point of the two lines -/
def intersectionPoint : ℝ × ℝ := (5, 5)

/-- The theorem stating the area of the quadrilateral -/
theorem quadrilateral_area 
  (l1 : Line1) 
  (l2 : Line2) : ℝ :=
  let o := (0, 0)
  let b := (0, l1.yIntercept)
  let e := intersectionPoint
  let c := (l2.xIntercept, 0)
  87.5

/-- Main theorem to prove -/
theorem main_theorem 
  (l1 : Line1) 
  (l2 : Line2) 
  (h1 : l1.slope = -1)
  (h2 : l1.xIntercept > 0)
  (h3 : l1.yIntercept > 0)
  (h4 : l2.xIntercept = 10)
  (h5 : l2.yIntercept > 0) :
  quadrilateral_area l1 l2 = 87.5 := by
  sorry

end quadrilateral_area_main_theorem_l1098_109804


namespace decimal_equivalent_one_fourth_power_one_l1098_109892

theorem decimal_equivalent_one_fourth_power_one : (1 / 4 : ℚ) ^ 1 = 0.25 := by
  sorry

end decimal_equivalent_one_fourth_power_one_l1098_109892


namespace sara_picked_six_pears_l1098_109825

/-- The number of pears picked by Tim -/
def tim_pears : ℕ := 5

/-- The total number of pears picked by Sara and Tim -/
def total_pears : ℕ := 11

/-- The number of pears picked by Sara -/
def sara_pears : ℕ := total_pears - tim_pears

theorem sara_picked_six_pears : sara_pears = 6 := by
  sorry

end sara_picked_six_pears_l1098_109825


namespace maddie_tshirt_cost_l1098_109874

/-- Calculates the total cost of T-shirts bought by Maddie -/
def total_cost (white_packs blue_packs white_per_pack blue_per_pack cost_per_shirt : ℕ) : ℕ :=
  ((white_packs * white_per_pack + blue_packs * blue_per_pack) * cost_per_shirt)

/-- Proves that Maddie spent $66 on T-shirts -/
theorem maddie_tshirt_cost :
  total_cost 2 4 5 3 3 = 66 := by
  sorry

end maddie_tshirt_cost_l1098_109874


namespace gunny_bag_fill_proof_l1098_109828

/-- Conversion factor from tons to pounds -/
def tons_to_pounds : ℝ := 2200

/-- Conversion factor from pounds to ounces -/
def pounds_to_ounces : ℝ := 16

/-- Conversion factor from grams to ounces -/
def grams_to_ounces : ℝ := 0.035274

/-- Capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℝ := 13.5

/-- Weight of a packet in pounds -/
def packet_weight_pounds : ℝ := 16

/-- Weight of a packet in additional ounces -/
def packet_weight_extra_ounces : ℝ := 4

/-- Weight of a packet in additional grams -/
def packet_weight_extra_grams : ℝ := 350

/-- The number of packets needed to fill the gunny bag -/
def packets_needed : ℕ := 1745

theorem gunny_bag_fill_proof : 
  ⌈(gunny_bag_capacity * tons_to_pounds * pounds_to_ounces) / 
   (packet_weight_pounds * pounds_to_ounces + packet_weight_extra_ounces + 
    packet_weight_extra_grams * grams_to_ounces)⌉ = packets_needed := by
  sorry

end gunny_bag_fill_proof_l1098_109828


namespace square_side_length_l1098_109856

theorem square_side_length (area : ℝ) (h : area = 9/16) :
  ∃ (side : ℝ), side > 0 ∧ side^2 = area ∧ side = 3/4 := by
  sorry

end square_side_length_l1098_109856


namespace city_mileage_problem_l1098_109845

theorem city_mileage_problem (n : ℕ) : n * (n - 1) / 2 = 15 → n = 6 := by
  sorry

end city_mileage_problem_l1098_109845


namespace smallest_integer_gcd_lcm_relation_l1098_109883

theorem smallest_integer_gcd_lcm_relation (m : ℕ) (h : m > 0) :
  (Nat.gcd 60 m * 20 = Nat.lcm 60 m) →
  (∀ k : ℕ, k > 0 ∧ k < m → Nat.gcd 60 k * 20 ≠ Nat.lcm 60 k) →
  m = 3 :=
sorry

end smallest_integer_gcd_lcm_relation_l1098_109883


namespace B_subset_A_l1098_109889

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem B_subset_A : B ⊆ A := by sorry

end B_subset_A_l1098_109889


namespace isosceles_triangle_solution_l1098_109868

-- Define the triangle properties
def isIsoscelesTriangle (x : ℝ) : Prop :=
  ∃ (side1 side2 side3 : ℝ),
    side1 = Real.tan x ∧ 
    side2 = Real.tan x ∧ 
    side3 = Real.tan (5 * x) ∧
    side1 = side2

-- Define the vertex angle condition
def hasVertexAngle4x (x : ℝ) : Prop :=
  ∃ (vertexAngle : ℝ),
    vertexAngle = 4 * x

-- Define the theorem
theorem isosceles_triangle_solution :
  ∀ x : ℝ,
    isIsoscelesTriangle x →
    hasVertexAngle4x x →
    0 < x →
    x < 90 →
    x = 20 := by sorry

end isosceles_triangle_solution_l1098_109868


namespace trailing_zeroes_500_factorial_l1098_109810

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial : trailingZeroes 500 = 124 := by sorry

end trailing_zeroes_500_factorial_l1098_109810


namespace greatest_difference_l1098_109815

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_difference (x y : ℕ) 
  (hx1 : 1 < x) (hx2 : x < 20) 
  (hy1 : 20 < y) (hy2 : y < 50) 
  (hxp : is_prime x) 
  (hym : ∃ k : ℕ, y = 7 * k) : 
  (∀ a b : ℕ, 1 < a → a < 20 → 20 < b → b < 50 → is_prime a → (∃ m : ℕ, b = 7 * m) → b - a ≤ y - x) ∧ y - x = 30 := by
  sorry

end greatest_difference_l1098_109815


namespace length_of_AB_prime_l1098_109860

/-- Given points A, B, C, and the conditions that A' and B' lie on y = x,
    prove that the length of A'B' is 3√2/28 -/
theorem length_of_AB_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 7) →
  B = (0, 10) →
  C = (3, 6) →
  (∃ t : ℝ, A' = (t, t)) →
  (∃ s : ℝ, B' = (s, s)) →
  (∃ k : ℝ, A'.1 = k * (C.1 - A.1) + A.1 ∧ A'.2 = k * (C.2 - A.2) + A.2) →
  (∃ m : ℝ, B'.1 = m * (C.1 - B.1) + B.1 ∧ B'.2 = m * (C.2 - B.2) + B.2) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 3 * Real.sqrt 2 / 28 := by
sorry

end length_of_AB_prime_l1098_109860


namespace tenth_term_of_sequence_l1098_109816

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem tenth_term_of_sequence (a : ℤ) (d : ℤ) :
  arithmetic_sequence a d 4 = 23 →
  arithmetic_sequence a d 8 = 55 →
  arithmetic_sequence a d 10 = 71 := by
sorry

end tenth_term_of_sequence_l1098_109816


namespace trigonometric_identity_l1098_109881

theorem trigonometric_identity : 
  let sin30 : ℝ := 1/2
  let cos45 : ℝ := Real.sqrt 2 / 2
  let cos60 : ℝ := 1/2
  2 * sin30 - cos45^2 + cos60 = 1 := by sorry

end trigonometric_identity_l1098_109881


namespace derivative_at_negative_one_l1098_109862

/-- Given a function f(x) = ax^4 + bx^2 + c, if f'(1) = 2, then f'(-1) = -2 -/
theorem derivative_at_negative_one
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^4 + b * x^2 + c)
  (h2 : deriv f 1 = 2) :
  deriv f (-1) = -2 := by
  sorry

end derivative_at_negative_one_l1098_109862


namespace ticket_price_possibilities_l1098_109884

theorem ticket_price_possibilities : ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ x ∈ S, x > 0 ∧ 72 % x = 0 ∧ 90 % x = 0 ∧ 150 % x = 0)) :=
by sorry

end ticket_price_possibilities_l1098_109884


namespace knight_statements_count_l1098_109895

/-- Represents the type of islanders -/
inductive IslanderType
| Knight
| Liar

/-- The total number of islanders -/
def total_islanders : ℕ := 28

/-- The number of times "You are a liar!" was said -/
def liar_statements : ℕ := 230

/-- Function to calculate the number of "You are a knight!" statements -/
def knight_statements (knights : ℕ) (liars : ℕ) : ℕ :=
  knights * (knights - 1) / 2 + liars * (liars - 1) / 2

theorem knight_statements_count :
  ∃ (knights liars : ℕ),
    knights ≥ 2 ∧
    liars ≥ 2 ∧
    knights + liars = total_islanders ∧
    knights * liars = liar_statements / 2 ∧
    knight_statements knights liars + liar_statements = total_islanders * (total_islanders - 1) ∧
    knight_statements knights liars = 526 :=
by
  sorry

end knight_statements_count_l1098_109895


namespace solution_in_interval_l1098_109801

theorem solution_in_interval (x₀ : ℝ) (h : Real.exp x₀ + x₀ = 2) : 0 < x₀ ∧ x₀ < 1 := by
  sorry

end solution_in_interval_l1098_109801


namespace regular_polygon_sides_l1098_109865

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 165 → (n : ℝ) * interior_angle = (n - 2 : ℝ) * 180 → n = 24 := by
  sorry

end regular_polygon_sides_l1098_109865


namespace inequality_solution_l1098_109867

theorem inequality_solution (x : ℝ) :
  x ∈ Set.Icc (-3 : ℝ) 3 ∧ 
  x ≠ -5/3 ∧ 
  (4*x^2 + 2) / (5 + 3*x) ≥ 1 ↔ 
  x ∈ Set.Icc (-3 : ℝ) (-3/4) ∪ Set.Icc 1 3 := by
sorry

end inequality_solution_l1098_109867


namespace very_spicy_peppers_l1098_109837

/-- The number of peppers needed for very spicy curries -/
def V : ℕ := sorry

/-- The number of peppers needed for spicy curries -/
def spicy_peppers : ℕ := 2

/-- The number of peppers needed for mild curries -/
def mild_peppers : ℕ := 1

/-- The number of spicy curries after adjustment -/
def spicy_curries : ℕ := 15

/-- The number of mild curries after adjustment -/
def mild_curries : ℕ := 90

/-- The reduction in the number of peppers bought after adjustment -/
def pepper_reduction : ℕ := 40

theorem very_spicy_peppers : 
  V = pepper_reduction := by sorry

end very_spicy_peppers_l1098_109837


namespace system_solution_l1098_109894

theorem system_solution :
  ∃ (x y : ℝ),
    (10 * x^2 + 5 * y^2 - 2 * x * y - 38 * x - 6 * y + 41 = 0) ∧
    (3 * x^2 - 2 * y^2 + 5 * x * y - 17 * x - 6 * y + 20 = 0) ∧
    (x = 2) ∧ (y = 1) := by
  sorry

end system_solution_l1098_109894


namespace mountain_height_l1098_109873

/-- Given a mountain where a person makes 10 round trips, reaching 3/4 of the height each time,
    and covering a total distance of 600,000 feet, the height of the mountain is 80,000 feet. -/
theorem mountain_height (trips : ℕ) (fraction_reached : ℚ) (total_distance : ℕ) 
    (h1 : trips = 10)
    (h2 : fraction_reached = 3/4)
    (h3 : total_distance = 600000) :
  (total_distance : ℚ) / (2 * trips * fraction_reached) = 80000 := by
  sorry

end mountain_height_l1098_109873


namespace max_roots_of_abs_sum_eq_abs_l1098_109869

/-- A quadratic polynomial of the form ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a given point x -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The number of roots of the equation |p₁(x)| + |p₂(x)| = |p₃(x)| -/
def numRoots (p₁ p₂ p₃ : QuadraticPolynomial) : ℕ :=
  sorry

/-- Theorem: The equation |p₁(x)| + |p₂(x)| = |p₃(x)| has at most 8 roots -/
theorem max_roots_of_abs_sum_eq_abs (p₁ p₂ p₃ : QuadraticPolynomial) :
  numRoots p₁ p₂ p₃ ≤ 8 := by
  sorry

end max_roots_of_abs_sum_eq_abs_l1098_109869


namespace trapezoid_area_l1098_109805

/-- The area of a trapezoid bounded by y = ax, y = bx, x = c, and x = d in the first quadrant -/
theorem trapezoid_area (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hcd : c < d) :
  let area := 0.5 * ((a * c + a * d + b * c + b * d) * (d - c))
  ∃ (trapezoid_area : ℝ), trapezoid_area = area := by
  sorry

end trapezoid_area_l1098_109805


namespace geometric_sequence_sum_first_six_l1098_109891

/-- A geometric sequence with positive terms satisfying a_{n+2} + 2a_{n+1} = 8a_n -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (a 1 = 1) ∧
  (∀ n, a (n + 2) + 2 * a (n + 1) = 8 * a n)

/-- The sum of the first 6 terms of the geometric sequence -/
def SumFirstSixTerms (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem geometric_sequence_sum_first_six (a : ℕ → ℝ) 
  (h : GeometricSequence a) : SumFirstSixTerms a = 63 := by
  sorry

end geometric_sequence_sum_first_six_l1098_109891


namespace max_value_quadratic_l1098_109857

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  ∃ (M : ℝ), M = 30 + 20*Real.sqrt 3 ∧ 
  ∀ (z w : ℝ), z > 0 → w > 0 → z^2 - 2*z*w + 3*w^2 = 10 → 
  z^2 + 2*z*w + 3*w^2 ≤ M := by
sorry

end max_value_quadratic_l1098_109857


namespace mod_37_5_l1098_109880

theorem mod_37_5 : 37 % 5 = 2 := by
  sorry

end mod_37_5_l1098_109880


namespace negation_equivalence_l1098_109818

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (|x| + |x - 1| < 2)) ↔ (∀ x : ℝ, |x| + |x - 1| ≥ 2) := by
  sorry

end negation_equivalence_l1098_109818


namespace correct_statements_l1098_109850

-- Define the proof methods
inductive ProofMethod
| Synthetic
| Analytic
| Contradiction

-- Define the characteristics of proof methods
def isCauseAndEffect (m : ProofMethod) : Prop := sorry
def isResultToCause (m : ProofMethod) : Prop := sorry
def isDirectProof (m : ProofMethod) : Prop := sorry

-- Define the statements
def statement1 : Prop := isCauseAndEffect ProofMethod.Synthetic
def statement2 : Prop := ¬(isDirectProof ProofMethod.Analytic)
def statement3 : Prop := isResultToCause ProofMethod.Analytic
def statement4 : Prop := isDirectProof ProofMethod.Contradiction

-- Theorem stating which statements are correct
theorem correct_statements :
  statement1 ∧ statement3 ∧ ¬statement2 ∧ ¬statement4 := by sorry

end correct_statements_l1098_109850


namespace max_value_of_e_l1098_109807

def b (n : ℕ) : ℤ := (5^n - 1) / 4

def e (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1)))

theorem max_value_of_e (n : ℕ) : e n = 1 := by
  sorry

end max_value_of_e_l1098_109807


namespace bus_speed_problem_l1098_109886

theorem bus_speed_problem (bus_length : ℝ) (fast_bus_speed : ℝ) (passing_time : ℝ) :
  bus_length = 3125 →
  fast_bus_speed = 40 →
  passing_time = 50/3600 →
  ∃ (slow_bus_speed : ℝ),
    slow_bus_speed = (2 * bus_length / 1000) / passing_time - fast_bus_speed ∧
    slow_bus_speed = 410 :=
by sorry

end bus_speed_problem_l1098_109886


namespace expression_evaluation_l1098_109812

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/3
  ((2*x + 3*y)^2 - (2*x + 3*y)*(2*x - 3*y)) / (3*y) = -6 :=
by sorry

end expression_evaluation_l1098_109812


namespace max_candy_leftover_l1098_109877

theorem max_candy_leftover (x : ℕ+) : ∃ (q r : ℕ), x = 7 * q + r ∧ r ≤ 6 ∧ ∀ (r' : ℕ), x = 7 * q + r' → r' ≤ r :=
sorry

end max_candy_leftover_l1098_109877


namespace z_minus_two_purely_imaginary_l1098_109888

def z : ℂ := Complex.mk 2 (-1)

theorem z_minus_two_purely_imaginary :
  Complex.im (z - 2) = Complex.im z ∧ Complex.re (z - 2) = 0 :=
sorry

end z_minus_two_purely_imaginary_l1098_109888


namespace tennis_tournament_matches_l1098_109851

theorem tennis_tournament_matches (n : ℕ) (byes : ℕ) (wildcard : ℕ) : 
  n = 128 → byes = 36 → wildcard = 1 →
  ∃ (total_matches : ℕ), 
    total_matches = n - 1 + wildcard ∧ 
    total_matches = 128 ∧
    total_matches % 2 = 0 :=
by sorry

end tennis_tournament_matches_l1098_109851


namespace divide_by_fraction_twelve_divided_by_one_sixth_l1098_109803

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / (1 / b) = a * b := by sorry

theorem twelve_divided_by_one_sixth : 12 / (1 / 6) = 72 := by sorry

end divide_by_fraction_twelve_divided_by_one_sixth_l1098_109803


namespace intersection_equality_implies_a_range_l1098_109864

-- Define sets A and B
def A : Set ℝ := {x | |x + 1| < 4}
def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - 2*a) < 0}

-- Theorem statement
theorem intersection_equality_implies_a_range (a : ℝ) :
  A ∩ B a = B a → a ∈ Set.Icc (-2.5) 1.5 := by
  sorry

end intersection_equality_implies_a_range_l1098_109864


namespace sqrt_inequality_l1098_109899

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d)
  (h5 : a + d = b + c) : 
  Real.sqrt a + Real.sqrt d < Real.sqrt b + Real.sqrt c := by
sorry

end sqrt_inequality_l1098_109899


namespace milk_discount_l1098_109808

/-- Calculates the discount on milk given grocery prices and remaining money --/
theorem milk_discount (initial_money : ℝ) (milk_price bread_price detergent_price banana_price_per_pound : ℝ)
  (banana_pounds : ℝ) (detergent_coupon : ℝ) (money_left : ℝ) :
  initial_money = 20 ∧
  milk_price = 4 ∧
  bread_price = 3.5 ∧
  detergent_price = 10.25 ∧
  banana_price_per_pound = 0.75 ∧
  banana_pounds = 2 ∧
  detergent_coupon = 1.25 ∧
  money_left = 4 →
  initial_money - (bread_price + (detergent_price - detergent_coupon) + 
    (banana_price_per_pound * banana_pounds) + money_left) = 2 :=
by sorry

end milk_discount_l1098_109808


namespace isosceles_triangle_properties_l1098_109890

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ

/-- Properties of the isosceles triangle -/
def IsoscelesTriangle.properties (t : IsoscelesTriangle) : Prop :=
  t.base = 16 ∧ t.side = 10

/-- Inradius of the triangle -/
def inradius (t : IsoscelesTriangle) : ℝ := sorry

/-- Circumradius of the triangle -/
def circumradius (t : IsoscelesTriangle) : ℝ := sorry

/-- Distance between the centers of inscribed and circumscribed circles -/
def centerDistance (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem about the properties of the isosceles triangle -/
theorem isosceles_triangle_properties (t : IsoscelesTriangle) 
  (h : t.properties) : 
  inradius t = 8/3 ∧ 
  circumradius t = 25/3 ∧ 
  centerDistance t = 5 := by sorry

end isosceles_triangle_properties_l1098_109890


namespace walter_exceptional_days_l1098_109800

/-- Represents the number of days Walter performed his chores in each category -/
structure ChorePerformance where
  poor : ℕ
  adequate : ℕ
  exceptional : ℕ

/-- Theorem stating that given the conditions, Walter performed exceptionally well for 6 days -/
theorem walter_exceptional_days :
  ∃ (perf : ChorePerformance),
    perf.poor + perf.adequate + perf.exceptional = 15 ∧
    2 * perf.poor + 4 * perf.adequate + 7 * perf.exceptional = 70 ∧
    perf.exceptional = 6 := by
  sorry


end walter_exceptional_days_l1098_109800


namespace tan_theta_for_pure_imaginary_l1098_109871

theorem tan_theta_for_pure_imaginary (θ : Real) :
  let z : ℂ := Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)
  (z.re = 0 ∧ z.im ≠ 0) → Real.tan θ = -3/4 := by
  sorry

end tan_theta_for_pure_imaginary_l1098_109871


namespace algebraic_expression_value_l1098_109855

theorem algebraic_expression_value : ∀ x : ℝ, x^2 - 4*x = 5 → 2*x^2 - 8*x - 6 = 4 := by
  sorry

end algebraic_expression_value_l1098_109855


namespace root_range_implies_m_range_l1098_109841

theorem root_range_implies_m_range :
  ∀ m : ℝ,
  (∀ x : ℝ, x^2 - 2*m*x + m^2 - 1 = 0 → x > -2) →
  m > -1 :=
by sorry

end root_range_implies_m_range_l1098_109841


namespace ant_ratio_is_two_to_one_l1098_109820

/-- The number of ants Abe finds -/
def abe_ants : ℕ := 4

/-- The number of ants Beth sees -/
def beth_ants : ℕ := (3 * abe_ants) / 2

/-- The number of ants Duke discovers -/
def duke_ants : ℕ := abe_ants / 2

/-- The total number of ants found by all four children -/
def total_ants : ℕ := 20

/-- The number of ants CeCe watches -/
def cece_ants : ℕ := total_ants - (abe_ants + beth_ants + duke_ants)

/-- The ratio of ants CeCe watches to ants Abe finds -/
def ant_ratio : ℚ := cece_ants / abe_ants

theorem ant_ratio_is_two_to_one : ant_ratio = 2 := by
  sorry

end ant_ratio_is_two_to_one_l1098_109820


namespace square_area_ratio_l1098_109833

theorem square_area_ratio (y : ℝ) (y_pos : y > 0) : 
  (3 * y)^2 / (9 * y)^2 = 1 / 9 := by
  sorry

end square_area_ratio_l1098_109833


namespace parts_probability_theorem_l1098_109839

/-- Represents the outcome of drawing a part -/
inductive DrawOutcome
| Standard
| NonStandard

/-- Represents the type of part that was lost -/
inductive LostPart
| Standard
| NonStandard

/-- The probability model for the parts problem -/
structure PartsModel where
  initialStandard : ℕ
  initialNonStandard : ℕ
  lostPart : LostPart
  drawnPart : DrawOutcome

def PartsModel.totalInitial (m : PartsModel) : ℕ :=
  m.initialStandard + m.initialNonStandard

def PartsModel.remainingTotal (m : PartsModel) : ℕ :=
  m.totalInitial - 1

def PartsModel.remainingStandard (m : PartsModel) : ℕ :=
  match m.lostPart with
  | LostPart.Standard => m.initialStandard - 1
  | LostPart.NonStandard => m.initialStandard

def PartsModel.probability (m : PartsModel) (event : PartsModel → Prop) : ℚ :=
  sorry

theorem parts_probability_theorem (m : PartsModel) 
  (h1 : m.initialStandard = 21)
  (h2 : m.initialNonStandard = 10)
  (h3 : m.drawnPart = DrawOutcome.Standard) :
  (m.probability (fun model => model.lostPart = LostPart.Standard) = 2/3) ∧
  (m.probability (fun model => model.lostPart = LostPart.NonStandard) = 1/3) :=
sorry

end parts_probability_theorem_l1098_109839


namespace line_passes_through_P_and_forms_triangle_circle_passes_through_M_and_N_with_center_on_y_axis_l1098_109859

-- Define the points
def P : ℝ × ℝ := (-1, 2)
def M : ℝ × ℝ := (-2, 3)
def N : ℝ × ℝ := (2, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y = 1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 5

-- Theorem for the line
theorem line_passes_through_P_and_forms_triangle :
  line_equation P.1 P.2 ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (1/2 : ℝ) * a * b = 1/2) :=
sorry

-- Theorem for the circle
theorem circle_passes_through_M_and_N_with_center_on_y_axis :
  circle_equation M.1 M.2 ∧
  circle_equation N.1 N.2 ∧
  (∃ y : ℝ, circle_equation 0 y) :=
sorry

end line_passes_through_P_and_forms_triangle_circle_passes_through_M_and_N_with_center_on_y_axis_l1098_109859


namespace cookie_count_l1098_109878

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) :
  bags * cookies_per_bag = 703 := by
  sorry

end cookie_count_l1098_109878


namespace total_cupcakes_eq_768_l1098_109819

/-- The number of cupcakes ordered for each event -/
def cupcakes_per_event : ℝ := 96.0

/-- The number of different children's events -/
def number_of_events : ℝ := 8.0

/-- The total number of cupcakes needed -/
def total_cupcakes : ℝ := cupcakes_per_event * number_of_events

/-- Theorem stating that the total number of cupcakes is 768.0 -/
theorem total_cupcakes_eq_768 : total_cupcakes = 768.0 := by
  sorry

end total_cupcakes_eq_768_l1098_109819


namespace chocolate_bar_difference_l1098_109826

theorem chocolate_bar_difference :
  let first_friend_portion : ℚ := 5 / 6
  let second_friend_portion : ℚ := 2 / 3
  first_friend_portion - second_friend_portion = 1 / 6 := by
sorry

end chocolate_bar_difference_l1098_109826


namespace gcd_of_45_and_75_l1098_109829

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_of_45_and_75_l1098_109829


namespace ellipse_m_value_l1098_109831

/-- Given an ellipse with equation x²/25 + y²/m² = 1 (m > 0) and left focus point at (-4, 0), 
    prove that m = 3 -/
theorem ellipse_m_value (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2/25 + y^2/m^2 = 1) → 
  (∃ x y : ℝ, x = -4 ∧ y = 0 ∧ (x + 5)^2/25 + y^2/m^2 < 1) → 
  m = 3 :=
by sorry

end ellipse_m_value_l1098_109831


namespace quadratic_solution_set_l1098_109870

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2*x - 3

theorem quadratic_solution_set (m : ℝ) : 
  (∀ x : ℝ, f m x ≤ 0 ↔ -1 < x ∧ x < 3) → m = 1 := by
  sorry

end quadratic_solution_set_l1098_109870


namespace cubes_form_name_l1098_109887

/-- Represents a cube with letters on its faces -/
structure Cube where
  faces : Fin 6 → Char

/-- Represents the visible face of a cube -/
inductive VisibleFace
  | front
  | right

/-- Returns the letter on the visible face of a cube -/
def visibleLetter (c : Cube) (f : VisibleFace) : Char :=
  match f with
  | VisibleFace.front => c.faces 0
  | VisibleFace.right => c.faces 1

/-- Represents the arrangement of four cubes -/
structure CubeArrangement where
  cubes : Fin 4 → Cube
  visibleFaces : Fin 4 → VisibleFace

/-- The name formed by the visible letters in the cube arrangement -/
def formName (arr : CubeArrangement) : String :=
  String.mk (List.ofFn fun i => visibleLetter (arr.cubes i) (arr.visibleFaces i))

/-- The theorem stating that the given cube arrangement forms the name "Ника" -/
theorem cubes_form_name (arr : CubeArrangement) 
  (h1 : visibleLetter (arr.cubes 0) (arr.visibleFaces 0) = 'Н')
  (h2 : visibleLetter (arr.cubes 1) (arr.visibleFaces 1) = 'И')
  (h3 : visibleLetter (arr.cubes 2) (arr.visibleFaces 2) = 'К')
  (h4 : visibleLetter (arr.cubes 3) (arr.visibleFaces 3) = 'А') :
  formName arr = "Ника" := by
  sorry


end cubes_form_name_l1098_109887


namespace water_needed_for_punch_l1098_109849

/-- Represents the recipe ratios and calculates the required amount of water -/
def water_needed (lemon_juice : ℝ) : ℝ :=
  let sugar := 3 * lemon_juice
  let water := 3 * sugar
  water

/-- Proves that 36 cups of water are needed given the recipe ratios and 4 cups of lemon juice -/
theorem water_needed_for_punch : water_needed 4 = 36 := by
  sorry

end water_needed_for_punch_l1098_109849


namespace balls_satisfy_conditions_l1098_109893

/-- Represents a word in the Russian language -/
structure RussianWord where
  word : String

/-- Represents a festive dance event -/
structure FestiveDanceEvent where
  name : String

/-- Represents a sporting event -/
inductive SportingEvent
| FigureSkating
| RhythmicGymnastics
| Other

/-- Represents the Russian pension system -/
structure RussianPensionSystem where
  calculationMethod : String
  yearIntroduced : Nat

/-- Checks if a word sounds similar to a festive dance event -/
def soundsSimilarTo (w : RussianWord) (e : FestiveDanceEvent) : Prop :=
  sorry

/-- Checks if a word is used in a sporting event -/
def usedInSportingEvent (w : RussianWord) (e : SportingEvent) : Prop :=
  sorry

/-- Checks if a word is used in the Russian pension system -/
def usedInPensionSystem (w : RussianWord) (p : RussianPensionSystem) : Prop :=
  sorry

/-- The main theorem stating that "баллы" satisfies all conditions -/
theorem balls_satisfy_conditions :
  ∃ (w : RussianWord) (e : FestiveDanceEvent) (p : RussianPensionSystem),
    w.word = "баллы" ∧
    soundsSimilarTo w e ∧
    usedInSportingEvent w SportingEvent.FigureSkating ∧
    usedInSportingEvent w SportingEvent.RhythmicGymnastics ∧
    usedInPensionSystem w p ∧
    p.yearIntroduced = 2015 :=
  sorry


end balls_satisfy_conditions_l1098_109893


namespace sum_product_ratio_l1098_109896

theorem sum_product_ratio (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : x ≠ z) (h4 : x + y + z = 1) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = (x * y + y * z + z * x) / (1 - 2 * (x * y + y * z + z * x)) :=
by sorry

end sum_product_ratio_l1098_109896


namespace square_division_exists_l1098_109853

/-- Represents a trapezoid with a given height -/
structure Trapezoid where
  height : ℝ

/-- Represents a square with a given side length -/
structure Square where
  side_length : ℝ

/-- Represents a division of a square into trapezoids -/
structure SquareDivision where
  square : Square
  trapezoids : List Trapezoid

/-- Checks if a list of trapezoids has the required heights -/
def has_required_heights (trapezoids : List Trapezoid) : Prop :=
  trapezoids.length = 4 ∧
  (∃ (h₁ h₂ h₃ h₄ : Trapezoid),
    trapezoids = [h₁, h₂, h₃, h₄] ∧
    h₁.height = 1 ∧ h₂.height = 2 ∧ h₃.height = 3 ∧ h₄.height = 4)

/-- Checks if a square division is valid -/
def is_valid_division (div : SquareDivision) : Prop :=
  div.square.side_length = 4 ∧
  has_required_heights div.trapezoids

/-- Theorem: A square with side length 4 can be divided into four trapezoids with heights 1, 2, 3, and 4 -/
theorem square_division_exists : ∃ (div : SquareDivision), is_valid_division div := by
  sorry

end square_division_exists_l1098_109853


namespace traffic_sampling_is_systematic_l1098_109876

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Quota

/-- Represents the characteristics of the sampling process --/
structure SamplingProcess where
  interval : ℕ  -- Time interval between samples
  continuous_stream : Bool  -- Whether there's a continuous stream of units to sample

/-- Determines if a sampling process is systematic --/
def is_systematic (process : SamplingProcess) : Prop :=
  process.interval > 0 ∧ process.continuous_stream

/-- The traffic police sampling process --/
def traffic_sampling : SamplingProcess :=
  { interval := 3,  -- 3 minutes interval
    continuous_stream := true }  -- Continuous stream of passing cars

/-- Theorem stating that the traffic sampling method is systematic --/
theorem traffic_sampling_is_systematic :
  is_systematic traffic_sampling ↔ SamplingMethod.Systematic = 
    (match traffic_sampling with
     | { interval := 3, continuous_stream := true } => SamplingMethod.Systematic
     | _ => SamplingMethod.SimpleRandom) :=
sorry

end traffic_sampling_is_systematic_l1098_109876


namespace quadratic_inequality_always_positive_l1098_109863

theorem quadratic_inequality_always_positive (r : ℝ) :
  (∀ x : ℝ, (r^2 - 1) * x^2 + 2 * (r - 1) * x + 1 > 0) ↔ r > 1 := by
  sorry

end quadratic_inequality_always_positive_l1098_109863


namespace min_value_expression_l1098_109811

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → 42 + b^2 + 1/(a*b) ≤ 42 + y^2 + 1/(x*y) ∧
  ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 42 + b₀^2 + 1/(a₀*b₀) = 17/2 :=
sorry

end min_value_expression_l1098_109811


namespace complex_expression_simplification_l1098_109827

theorem complex_expression_simplification :
  3 * (4 - 2 * Complex.I) - 2 * (2 * Complex.I - 3) = 18 - 10 * Complex.I :=
by sorry

end complex_expression_simplification_l1098_109827


namespace intersection_point_correct_l1098_109830

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The y-intercept of the first line -/
def b₁ : ℚ := -1

/-- The x-coordinate of the given point -/
def x₀ : ℚ := 4

/-- The y-coordinate of the given point -/
def y₀ : ℚ := 2

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := 13 / 10

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := 29 / 10

/-- Theorem stating that the intersection point is correct -/
theorem intersection_point_correct : 
  (m₁ * x_intersect + b₁ = y_intersect) ∧ 
  (m₂ * (x_intersect - x₀) = y_intersect - y₀) := by
  sorry

end intersection_point_correct_l1098_109830


namespace theater_seats_count_l1098_109802

/-- Represents the theater ticket sales scenario -/
structure TheaterSales where
  adultTicketPrice : ℕ
  childTicketPrice : ℕ
  totalRevenue : ℕ
  childTicketsSold : ℕ

/-- Calculates the total number of seats in the theater -/
def totalSeats (sales : TheaterSales) : ℕ :=
  let adultTicketsSold := (sales.totalRevenue - sales.childTicketPrice * sales.childTicketsSold) / sales.adultTicketPrice
  adultTicketsSold + sales.childTicketsSold

/-- Theorem stating that given the specific conditions, the theater has 80 seats -/
theorem theater_seats_count (sales : TheaterSales) 
  (h1 : sales.adultTicketPrice = 12)
  (h2 : sales.childTicketPrice = 5)
  (h3 : sales.totalRevenue = 519)
  (h4 : sales.childTicketsSold = 63) :
  totalSeats sales = 80 := by
  sorry

end theater_seats_count_l1098_109802


namespace quadratic_factorization_l1098_109834

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) 
  (h2 : ∀ x, x^2 - 18*x + 72 = (x - a)*(x - b)) : 
  2*b - a = 0 := by
  sorry

end quadratic_factorization_l1098_109834


namespace ed_doug_marble_difference_l1098_109842

theorem ed_doug_marble_difference (ed_initial : ℕ) (doug_initial : ℕ) (ed_lost : ℕ) (ed_final : ℕ) :
  ed_initial = doug_initial + 30 →
  ed_initial = ed_final + ed_lost →
  ed_lost = 21 →
  ed_final = 91 →
  ed_final - doug_initial = 9 :=
by sorry

end ed_doug_marble_difference_l1098_109842


namespace simplify_fraction_l1098_109821

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (1 - x / (x - 1)) / (1 / (x^2 - x)) = -x := by
  sorry

end simplify_fraction_l1098_109821


namespace brandon_skittles_count_l1098_109898

/-- Given Brandon's initial Skittles count and the number of Skittles he loses,
    prove that his final Skittles count is the difference between the initial count and the number lost. -/
theorem brandon_skittles_count (initial_count lost_count : ℕ) :
  initial_count - lost_count = initial_count - lost_count :=
by sorry

end brandon_skittles_count_l1098_109898


namespace probability_three_same_color_l1098_109838

def total_marbles : ℕ := 23
def red_marbles : ℕ := 6
def white_marbles : ℕ := 8
def blue_marbles : ℕ := 9

def probability_same_color : ℚ := 160 / 1771

theorem probability_three_same_color :
  probability_same_color = (Nat.choose red_marbles 3 + Nat.choose white_marbles 3 + Nat.choose blue_marbles 3) / Nat.choose total_marbles 3 :=
by sorry

end probability_three_same_color_l1098_109838


namespace binomial_square_coefficient_l1098_109824

theorem binomial_square_coefficient (x : ℝ) : ∃ b : ℝ, ∃ t u : ℝ, 
  b * x^2 + 20 * x + 1 = (t * x + u)^2 ∧ b = 100 := by
  sorry

end binomial_square_coefficient_l1098_109824


namespace subset_divisibility_subset_1000_500_divisibility_l1098_109847

theorem subset_divisibility (n : ℕ) (k : ℕ) (p : ℕ) : Prop :=
  p ∣ Nat.choose n k

theorem subset_1000_500_divisibility :
  subset_divisibility 1000 500 3 ∧
  subset_divisibility 1000 500 5 ∧
  ¬(subset_divisibility 1000 500 11) ∧
  subset_divisibility 1000 500 13 ∧
  subset_divisibility 1000 500 17 :=
by sorry

end subset_divisibility_subset_1000_500_divisibility_l1098_109847


namespace expression_evaluation_l1098_109822

theorem expression_evaluation : 200 * (200 - 8) - (200 * 200 + 8) = -1608 := by
  sorry

end expression_evaluation_l1098_109822


namespace min_sum_dimensions_l1098_109840

theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 2310 → 
  ∀ a b c : ℕ+, a * b * c = 2310 → l + w + h ≤ a + b + c → 
  l + w + h = 52 := by
sorry

end min_sum_dimensions_l1098_109840


namespace root_sum_l1098_109848

theorem root_sum (n m : ℝ) (h1 : n ≠ 0) (h2 : n^2 + m*n + 3*n = 0) : m + n = -3 := by
  sorry

end root_sum_l1098_109848


namespace train_length_calculation_l1098_109813

/-- Calculates the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length_calculation (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  speed = 72 → platform_length = 250 → crossing_time = 30 →
  (speed * (5/18) * crossing_time) - platform_length = 350 := by
  sorry

end train_length_calculation_l1098_109813


namespace three_squares_balance_l1098_109844

/-- A balance system with three symbols: triangle, square, and circle. -/
structure BalanceSystem where
  triangle : ℚ
  square : ℚ
  circle : ℚ

/-- The balance rules for the system. -/
def balance_rules (s : BalanceSystem) : Prop :=
  5 * s.triangle + 2 * s.square = 21 * s.circle ∧
  2 * s.triangle = s.square + 3 * s.circle

/-- The theorem to prove. -/
theorem three_squares_balance (s : BalanceSystem) :
  balance_rules s → 3 * s.square = 9 * s.circle :=
by
  sorry

end three_squares_balance_l1098_109844


namespace sine_product_inequality_l1098_109843

theorem sine_product_inequality :
  1/8 < Real.sin (20 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) ∧
  Real.sin (20 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) < 1/4 := by
  sorry

end sine_product_inequality_l1098_109843


namespace scatter_diagram_placement_l1098_109897

/-- Represents a variable in a scatter diagram -/
inductive ScatterVariable
| Explanatory
| Predictor

/-- Represents an axis in a scatter diagram -/
inductive Axis
| X
| Y

/-- Determines the correct axis for a given scatter variable -/
def correct_axis_placement (v : ScatterVariable) : Axis :=
  match v with
  | ScatterVariable.Explanatory => Axis.X
  | ScatterVariable.Predictor => Axis.Y

/-- Theorem stating the correct placement of variables in a scatter diagram -/
theorem scatter_diagram_placement :
  (correct_axis_placement ScatterVariable.Explanatory = Axis.X) ∧
  (correct_axis_placement ScatterVariable.Predictor = Axis.Y) :=
by sorry

end scatter_diagram_placement_l1098_109897


namespace line_not_in_fourth_quadrant_l1098_109814

/-- A line in the 2D plane represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The fourth quadrant of the 2D plane -/
def fourth_quadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0}

/-- Theorem: A line Ax + By + C = 0 where AB < 0 and BC < 0 does not pass through the fourth quadrant -/
theorem line_not_in_fourth_quadrant (l : Line) 
    (h1 : l.A * l.B < 0) 
    (h2 : l.B * l.C < 0) : 
    ∀ p ∈ fourth_quadrant, l.A * p.1 + l.B * p.2 + l.C ≠ 0 :=
by
  sorry

end line_not_in_fourth_quadrant_l1098_109814


namespace corrected_mean_l1098_109879

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (wrong1 wrong2 correct1 correct2 : ℝ) 
  (h1 : n = 100)
  (h2 : original_mean = 56)
  (h3 : wrong1 = 38)
  (h4 : wrong2 = 27)
  (h5 : correct1 = 89)
  (h6 : correct2 = 73) :
  let incorrect_sum := n * original_mean
  let difference := (correct1 + correct2) - (wrong1 + wrong2)
  let corrected_sum := incorrect_sum + difference
  corrected_sum / n = 56.97 := by sorry

end corrected_mean_l1098_109879


namespace inequality_problem_l1098_109866

theorem inequality_problem (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpr : p * r > q * r) :
  ¬(-p > -q) ∧ ¬(-p > q) ∧ ¬(1 > -q/p) ∧ ¬(1 < q/p) :=
by sorry

end inequality_problem_l1098_109866


namespace shortest_altitude_of_special_triangle_l1098_109858

theorem shortest_altitude_of_special_triangle :
  ∀ (a b c h : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = (1/2) * c * h →
  h = 7.2 :=
by
  sorry

end shortest_altitude_of_special_triangle_l1098_109858


namespace min_value_reciprocal_sum_l1098_109852

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_geometric_mean : 4 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 1 := by
  sorry

end min_value_reciprocal_sum_l1098_109852


namespace computer_price_l1098_109882

theorem computer_price (P : ℝ) 
  (h1 : 1.30 * P = 351)
  (h2 : 2 * P = 540) :
  P = 270 := by
sorry

end computer_price_l1098_109882


namespace least_possible_third_side_length_l1098_109832

theorem least_possible_third_side_length (a b c : ℝ) : 
  a = 8 → b = 15 → c > 0 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c ≥ Real.sqrt 161 := by
  sorry

end least_possible_third_side_length_l1098_109832


namespace distance_X_to_Y_l1098_109836

/-- The distance between points X and Y -/
def D : ℝ := sorry

/-- Yolanda's walking rate in miles per hour -/
def yolanda_rate : ℝ := 3

/-- Bob's walking rate in miles per hour -/
def bob_rate : ℝ := 4

/-- Time difference between Yolanda and Bob's start in hours -/
def time_difference : ℝ := 1

/-- Distance Bob walked when they met -/
def bob_distance : ℝ := 4

/-- Theorem stating the distance between X and Y -/
theorem distance_X_to_Y : D = 10 := by sorry

end distance_X_to_Y_l1098_109836


namespace train_length_calculation_l1098_109854

/-- Given a train that crosses a platform and a post, calculate its length. -/
theorem train_length_calculation (platform_length : ℝ) (platform_time : ℝ) (post_time : ℝ) 
  (h1 : platform_length = 350)
  (h2 : platform_time = 39)
  (h3 : post_time = 18) :
  ∃ (train_length : ℝ), train_length = 300 ∧ 
    (train_length + platform_length) / platform_time = train_length / post_time := by
  sorry


end train_length_calculation_l1098_109854


namespace min_value_reciprocal_sum_l1098_109809

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_mean : (a + b) / 2 = 1 / 2) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + y) / 2 = 1 / 2 → 1 / x + 1 / y ≥ 4 :=
by sorry

end min_value_reciprocal_sum_l1098_109809


namespace quadratic_roots_conditions_l1098_109817

theorem quadratic_roots_conditions (k : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * x^2 - 4 * k * x + x - (1 - 2 * k^2)
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ↔ k ≤ 9/8 ∧
  (∀ x : ℝ, f x ≠ 0) ↔ k > 9/8 :=
by sorry

end quadratic_roots_conditions_l1098_109817


namespace pen_purchase_ratio_l1098_109835

/-- The ratio of fountain pens to ballpoint pens in a purchase scenario --/
theorem pen_purchase_ratio (x y : ℕ) (h1 : (2 * x + y) * 3 = 3 * (2 * y + x)) :
  y = 4 * x := by
  sorry

#check pen_purchase_ratio

end pen_purchase_ratio_l1098_109835


namespace quadratic_inequality_l1098_109872

theorem quadratic_inequality (x : ℝ) :
  9 * x^2 - 6 * x + 1 > 0 ↔ x < 1/3 ∨ x > 1/3 := by
  sorry

end quadratic_inequality_l1098_109872


namespace maintain_ratio_theorem_l1098_109875

/-- Represents the ingredients in a cake recipe -/
structure Recipe where
  flour : Float
  sugar : Float
  oil : Float

/-- Calculates the new amounts of ingredients while maintaining the ratio -/
def calculate_new_amounts (original : Recipe) (new_flour : Float) : Recipe :=
  let scale_factor := new_flour / original.flour
  { flour := new_flour,
    sugar := original.sugar * scale_factor,
    oil := original.oil * scale_factor }

/-- Rounds a float to two decimal places -/
def round_to_two_decimals (x : Float) : Float :=
  (x * 100).round / 100

theorem maintain_ratio_theorem (original : Recipe) (extra_flour : Float) :
  let new_recipe := calculate_new_amounts original (original.flour + extra_flour)
  round_to_two_decimals new_recipe.sugar = 3.86 ∧
  round_to_two_decimals new_recipe.oil = 2.57 :=
by sorry

end maintain_ratio_theorem_l1098_109875


namespace sine_cosine_inequality_l1098_109823

theorem sine_cosine_inequality (x y : Real) (h1 : 0 ≤ x) (h2 : x ≤ y) (h3 : y ≤ Real.pi / 2) :
  (Real.sin (x / 2))^2 * Real.cos y ≤ 1 / 8 := by
  sorry

end sine_cosine_inequality_l1098_109823


namespace seven_lines_angle_l1098_109861

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a function to check if two lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define a function to measure the angle between two lines
def angle_between (l1 l2 : Line) : ℝ := sorry

-- The main theorem
theorem seven_lines_angle (lines : Fin 7 → Line) :
  (∀ i j, i ≠ j → ¬ parallel (lines i) (lines j)) →
  ∃ i j, i ≠ j ∧ angle_between (lines i) (lines j) < 26 * π / 180 :=
sorry

end seven_lines_angle_l1098_109861


namespace max_x5_value_l1098_109806

theorem max_x5_value (x₁ x₂ x₃ x₄ x₅ : ℕ) 
  (h : x₁ + x₂ + x₃ + x₄ + x₅ = x₁ * x₂ * x₃ * x₄ * x₅) :
  x₅ ≤ 5 ∧ ∃ y₁ y₂ y₃ y₄ : ℕ, y₁ + y₂ + y₃ + y₄ + 5 = y₁ * y₂ * y₃ * y₄ * 5 :=
by sorry

end max_x5_value_l1098_109806


namespace athlete_A_second_day_prob_l1098_109885

-- Define the probabilities
def prob_A_first_day : ℝ := 0.5
def prob_B_first_day : ℝ := 0.5
def prob_A_second_day_given_A_first : ℝ := 0.6
def prob_A_second_day_given_B_first : ℝ := 0.5

-- State the theorem
theorem athlete_A_second_day_prob :
  prob_A_first_day * prob_A_second_day_given_A_first +
  prob_B_first_day * prob_A_second_day_given_B_first = 0.55 := by
  sorry

end athlete_A_second_day_prob_l1098_109885
