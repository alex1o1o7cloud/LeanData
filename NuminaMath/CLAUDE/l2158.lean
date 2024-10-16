import Mathlib

namespace NUMINAMATH_CALUDE_circle_C_equation_l2158_215897

-- Define the circles and points
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def point_A : ℝ × ℝ := (1, 0)

-- Define the properties of circle C
structure Circle_C where
  center : ℝ × ℝ
  tangent_to_x_axis : center.2 > 0
  tangent_at_A : (center.1 - point_A.1)^2 + (center.2 - point_A.2)^2 = center.2^2
  intersects_O : ∃ P Q : ℝ × ℝ, P ∈ circle_O ∧ Q ∈ circle_O ∧
    (P.1 - center.1)^2 + (P.2 - center.2)^2 = center.2^2 ∧
    (Q.1 - center.1)^2 + (Q.2 - center.2)^2 = center.2^2
  PQ_length : ∃ P Q : ℝ × ℝ, P ∈ circle_O ∧ Q ∈ circle_O ∧
    (P.1 - center.1)^2 + (P.2 - center.2)^2 = center.2^2 ∧
    (Q.1 - center.1)^2 + (Q.2 - center.2)^2 = center.2^2 ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 14/4

-- Theorem stating the standard equation of circle C
theorem circle_C_equation (c : Circle_C) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.center.2^2 ↔
  (x - 1)^2 + (y - 1)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_l2158_215897


namespace NUMINAMATH_CALUDE_probability_ray_in_angle_l2158_215880

/-- The probability of a randomly drawn ray falling within a 60-degree angle in a circular region is 1/6. -/
theorem probability_ray_in_angle (angle : ℝ) (total_angle : ℝ) : 
  angle = 60 → total_angle = 360 → angle / total_angle = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_ray_in_angle_l2158_215880


namespace NUMINAMATH_CALUDE_point_parameters_l2158_215881

/-- Parametric equation of a line -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The given line -/
def givenLine : ParametricLine :=
  { x := λ t => 1 + 2 * t,
    y := λ t => 2 - 3 * t }

/-- Point A -/
def pointA : Point :=
  { x := 1,
    y := 2 }

/-- Point B -/
def pointB : Point :=
  { x := -1,
    y := 5 }

/-- Theorem stating that the parameters for points A and B are 0 and -1 respectively -/
theorem point_parameters : 
  (∃ t : ℝ, givenLine.x t = pointA.x ∧ givenLine.y t = pointA.y ∧ t = 0) ∧
  (∃ t : ℝ, givenLine.x t = pointB.x ∧ givenLine.y t = pointB.y ∧ t = -1) :=
by sorry

end NUMINAMATH_CALUDE_point_parameters_l2158_215881


namespace NUMINAMATH_CALUDE_total_marbles_is_193_l2158_215844

/-- The number of marbles in the jar when Ben, Leo, and Tim combine their marbles. -/
def totalMarbles : ℕ :=
  let benMarbles : ℕ := 56
  let leoMarbles : ℕ := benMarbles + 20
  let timMarbles : ℕ := leoMarbles - 15
  benMarbles + leoMarbles + timMarbles

/-- Theorem stating that the total number of marbles in the jar is 193. -/
theorem total_marbles_is_193 : totalMarbles = 193 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_193_l2158_215844


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2158_215824

theorem imaginary_part_of_complex_fraction : Complex.im ((1 + 3*Complex.I) / (3 - Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2158_215824


namespace NUMINAMATH_CALUDE_product_sum_equals_30_l2158_215846

theorem product_sum_equals_30 (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 3)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 17) : 
  a * b + c * d = 30 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equals_30_l2158_215846


namespace NUMINAMATH_CALUDE_cereal_eating_time_l2158_215890

def mr_fat_rate : ℚ := 1 / 20
def mr_thin_rate : ℚ := 1 / 25
def total_cereal : ℚ := 4

def combined_rate : ℚ := mr_fat_rate + mr_thin_rate

theorem cereal_eating_time :
  (total_cereal / combined_rate) = 400 / 9 := by sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l2158_215890


namespace NUMINAMATH_CALUDE_point_on_segment_with_vector_relation_l2158_215842

/-- Given two points M and N in ℝ², and a point P on the line segment MN
    such that vector PN = -2 * vector PM, prove that P has coordinates (2,4) -/
theorem point_on_segment_with_vector_relation (M N P : ℝ × ℝ) :
  M = (-2, 7) →
  N = (10, -2) →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • M + t • N →
  (N.1 - P.1, N.2 - P.2) = (-2 * (M.1 - P.1), -2 * (M.2 - P.2)) →
  P = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_on_segment_with_vector_relation_l2158_215842


namespace NUMINAMATH_CALUDE_marble_202_is_white_l2158_215820

/-- Represents the colors of marbles -/
inductive Color
  | Gray
  | White
  | Black
  | Red

/-- Returns the color of the nth marble in the repeating pattern -/
def marbleColor (n : ℕ) : Color :=
  match n % 15 with
  | 0 | 1 | 2 | 3 | 4 | 5 => Color.Gray
  | 6 | 7 | 8 => Color.White
  | 9 | 10 | 11 | 12 => Color.Black
  | _ => Color.Red

theorem marble_202_is_white :
  marbleColor 202 = Color.White := by
  sorry

end NUMINAMATH_CALUDE_marble_202_is_white_l2158_215820


namespace NUMINAMATH_CALUDE_D_96_l2158_215801

/-- D(n) is the number of ways of writing n as a product of integers greater than 1, where the order matters -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem: D(96) = 112 -/
theorem D_96 : D 96 = 112 := by sorry

end NUMINAMATH_CALUDE_D_96_l2158_215801


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2158_215868

/-- Given real numbers a, b, and c such that the infinite geometric series
    a/b + a/b^2 + a/b^3 + ... equals 3, prove that the sum of the series
    ca/(a+b) + ca/(a+b)^2 + ca/(a+b)^3 + ... equals 3c/4 -/
theorem geometric_series_sum (a b c : ℝ) 
  (h : ∑' n, a / b^n = 3) : 
  ∑' n, c * a / (a + b)^n = 3/4 * c := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2158_215868


namespace NUMINAMATH_CALUDE_sine_inequality_l2158_215872

theorem sine_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π/4) :
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l2158_215872


namespace NUMINAMATH_CALUDE_arevalo_dinner_bill_l2158_215896

/-- The Arevalo family's dinner bill problem -/
theorem arevalo_dinner_bill (salmon_price black_burger_price chicken_katsu_price : ℝ)
  (service_charge_rate : ℝ) (paid_amount change_received : ℝ) :
  salmon_price = 40 ∧
  black_burger_price = 15 ∧
  chicken_katsu_price = 25 ∧
  service_charge_rate = 0.1 ∧
  paid_amount = 100 ∧
  change_received = 8 →
  let total_food_cost := salmon_price + black_burger_price + chicken_katsu_price
  let service_charge := service_charge_rate * total_food_cost
  let subtotal := total_food_cost + service_charge
  let amount_paid := paid_amount - change_received
  let tip := amount_paid - subtotal
  tip / total_food_cost = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_arevalo_dinner_bill_l2158_215896


namespace NUMINAMATH_CALUDE_find_divisor_l2158_215818

theorem find_divisor (x d : ℚ) : 
  x = 55 → 
  x / d + 10 = 21 → 
  d = 5 := by sorry

end NUMINAMATH_CALUDE_find_divisor_l2158_215818


namespace NUMINAMATH_CALUDE_nori_initial_boxes_l2158_215895

/-- The number of crayons in each box -/
def crayons_per_box : ℕ := 8

/-- The number of crayons Nori gave to Mae -/
def crayons_to_mae : ℕ := 5

/-- The additional number of crayons Nori gave to Lea compared to Mae -/
def additional_crayons_to_lea : ℕ := 7

/-- The number of crayons Nori has left -/
def crayons_left : ℕ := 15

/-- The number of boxes Nori had initially -/
def initial_boxes : ℕ := 4

theorem nori_initial_boxes : 
  crayons_per_box * initial_boxes = 
    crayons_left + crayons_to_mae + (crayons_to_mae + additional_crayons_to_lea) :=
by sorry

end NUMINAMATH_CALUDE_nori_initial_boxes_l2158_215895


namespace NUMINAMATH_CALUDE_angle_terminal_side_formula_l2158_215825

/-- Given a point P(-4,3) on the terminal side of angle α, prove that 2sin α + cos α = 2/5 -/
theorem angle_terminal_side_formula (α : Real) (P : ℝ × ℝ) : 
  P = (-4, 3) → 2 * Real.sin α + Real.cos α = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_formula_l2158_215825


namespace NUMINAMATH_CALUDE_shipping_cost_invariant_l2158_215875

/-- Represents a settlement with its distance from the city and required goods weight -/
structure Settlement where
  distance : ℝ
  weight : ℝ
  distance_eq_weight : distance = weight

/-- Calculates the shipping cost for a given delivery order -/
def shipping_cost (settlements : List Settlement) : ℝ :=
  settlements.enum.foldl
    (fun acc (i, s) =>
      acc + s.weight * (settlements.take i).foldl (fun sum t => sum + t.distance) 0)
    0

/-- Theorem stating that the shipping cost is invariant under different delivery orders -/
theorem shipping_cost_invariant (settlements : List Settlement) :
  ∀ (perm : List Settlement), settlements.Perm perm →
    shipping_cost settlements = shipping_cost perm :=
  sorry

end NUMINAMATH_CALUDE_shipping_cost_invariant_l2158_215875


namespace NUMINAMATH_CALUDE_expression_evaluation_l2158_215816

theorem expression_evaluation : 2 * 0 + 1 - 9 = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2158_215816


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2158_215886

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def M : Finset Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  (U \ M) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2158_215886


namespace NUMINAMATH_CALUDE_total_shells_is_61_l2158_215806

def bucket_a_initial : ℕ := 5
def bucket_a_additional : ℕ := 12

def bucket_b_initial : ℕ := 8
def bucket_b_additional : ℕ := 15

def bucket_c_initial : ℕ := 3
def bucket_c_additional : ℕ := 18

def total_shells : ℕ := 
  (bucket_a_initial + bucket_a_additional) + 
  (bucket_b_initial + bucket_b_additional) + 
  (bucket_c_initial + bucket_c_additional)

theorem total_shells_is_61 : total_shells = 61 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_is_61_l2158_215806


namespace NUMINAMATH_CALUDE_soccer_stars_games_l2158_215831

theorem soccer_stars_games (wins losses draws : ℕ) 
  (h1 : wins = 14)
  (h2 : losses = 2)
  (h3 : 3 * wins + draws = 46) :
  wins + losses + draws = 20 := by
sorry

end NUMINAMATH_CALUDE_soccer_stars_games_l2158_215831


namespace NUMINAMATH_CALUDE_max_expenditure_max_expected_expenditure_l2158_215810

-- Define the linear regression model
def linear_regression (x : ℝ) (b a e : ℝ) : ℝ := b * x + a + e

-- State the theorem
theorem max_expenditure (x : ℝ) (e : ℝ) :
  x = 10 →
  0.8 * x + 2 + e ≤ 10.5 :=
by
  sorry

-- Define the constraint on e
def e_constraint (e : ℝ) : Prop := abs e ≤ 0.5

-- State the main theorem
theorem max_expected_expenditure (x : ℝ) :
  x = 10 →
  ∀ e, e_constraint e →
  linear_regression x 0.8 2 e ≤ 10.5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_expenditure_max_expected_expenditure_l2158_215810


namespace NUMINAMATH_CALUDE_tv_cost_l2158_215813

theorem tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 1800 →
  furniture_fraction = 3/4 →
  tv_cost = savings * (1 - furniture_fraction) →
  tv_cost = 450 := by
sorry

end NUMINAMATH_CALUDE_tv_cost_l2158_215813


namespace NUMINAMATH_CALUDE_remainder_seventeen_power_sixtythree_mod_seven_l2158_215879

theorem remainder_seventeen_power_sixtythree_mod_seven :
  17^63 % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_seventeen_power_sixtythree_mod_seven_l2158_215879


namespace NUMINAMATH_CALUDE_rental_distance_theorem_l2158_215863

/-- Calculates the distance driven given the rental parameters and total cost -/
def distance_driven (daily_rate : ℚ) (per_mile_rate : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - daily_rate) / per_mile_rate

/-- Proves that given the specified rental conditions, the distance driven is 214 miles -/
theorem rental_distance_theorem :
  let daily_rate : ℚ := 29
  let per_mile_rate : ℚ := 8 / 100
  let total_cost : ℚ := 4612 / 100
  distance_driven daily_rate per_mile_rate total_cost = 214 := by
  sorry

end NUMINAMATH_CALUDE_rental_distance_theorem_l2158_215863


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2158_215809

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-1, 3}

theorem union_of_A_and_B : A ∪ B = {-1, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2158_215809


namespace NUMINAMATH_CALUDE_exists_rectangle_six_pieces_l2158_215845

/-- A rectangle inscribed in an isosceles right triangle --/
structure InscribedRectangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  h_positive : side1 > 0 ∧ side2 > 0 ∧ hypotenuse > 0
  h_inscribed : side1 + side2 < hypotenuse

/-- Two straight lines that divide a rectangle --/
structure DividingLines where
  line1 : ℝ × ℝ → ℝ × ℝ → Prop
  line2 : ℝ × ℝ → ℝ × ℝ → Prop

/-- The number of pieces a rectangle is divided into by two straight lines --/
def numPieces (r : InscribedRectangle) (d : DividingLines) : ℕ :=
  sorry

/-- Theorem stating the existence of a rectangle that can be divided into 6 pieces --/
theorem exists_rectangle_six_pieces :
  ∃ (r : InscribedRectangle) (d : DividingLines), numPieces r d = 6 :=
sorry

end NUMINAMATH_CALUDE_exists_rectangle_six_pieces_l2158_215845


namespace NUMINAMATH_CALUDE_students_have_two_hands_l2158_215891

/-- Given a class with the following properties:
  * There are 11 students including Peter
  * The total number of hands excluding Peter's is 20
  * Every student has the same number of hands
  Prove that each student has 2 hands. -/
theorem students_have_two_hands
  (total_students : ℕ)
  (hands_excluding_peter : ℕ)
  (h_total_students : total_students = 11)
  (h_hands_excluding_peter : hands_excluding_peter = 20) :
  hands_excluding_peter + 2 = total_students * 2 :=
sorry

end NUMINAMATH_CALUDE_students_have_two_hands_l2158_215891


namespace NUMINAMATH_CALUDE_not_necessarily_p_or_q_l2158_215860

theorem not_necessarily_p_or_q (h1 : ¬p) (h2 : ¬(p ∧ q)) : 
  ¬∀ (p q : Prop), p ∨ q := by
sorry

end NUMINAMATH_CALUDE_not_necessarily_p_or_q_l2158_215860


namespace NUMINAMATH_CALUDE_peanut_cost_per_pound_l2158_215894

/-- The cost per pound of peanuts at Peanut Emporium -/
def cost_per_pound : ℝ := 3

/-- The minimum purchase amount in pounds -/
def minimum_purchase : ℝ := 15

/-- The amount purchased over the minimum in pounds -/
def over_minimum : ℝ := 20

/-- The total cost of the purchase in dollars -/
def total_cost : ℝ := 105

/-- Proof that the cost per pound of peanuts is $3 -/
theorem peanut_cost_per_pound :
  cost_per_pound = total_cost / (minimum_purchase + over_minimum) := by
  sorry

end NUMINAMATH_CALUDE_peanut_cost_per_pound_l2158_215894


namespace NUMINAMATH_CALUDE_norbs_age_l2158_215889

def guesses : List Nat := [25, 29, 33, 35, 37, 39, 42, 45, 48, 50]

def is_prime (n : Nat) : Prop := Nat.Prime n

def half_guesses_too_low (age : Nat) : Prop :=
  (guesses.filter (· < age)).length = guesses.length / 2

def two_guesses_off_by_one (age : Nat) : Prop :=
  (guesses.filter (λ g => g = age - 1 ∨ g = age + 1)).length = 2

theorem norbs_age : 
  ∃ (age : Nat), age = 47 ∧ 
    is_prime age ∧ 
    half_guesses_too_low age ∧ 
    two_guesses_off_by_one age ∧
    ∀ (n : Nat), n ≠ 47 → 
      ¬(is_prime n ∧ half_guesses_too_low n ∧ two_guesses_off_by_one n) :=
by sorry

end NUMINAMATH_CALUDE_norbs_age_l2158_215889


namespace NUMINAMATH_CALUDE_range_of_m_l2158_215899

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0 → -2 ≤ x ∧ x ≤ 10) ∧
  (∀ x : ℝ, x^2 - 2*x + (1 - m^2) ≤ 0 → 1 - m ≤ x ∧ x ≤ 1 + m) ∧
  (m > 0) ∧
  (∀ x : ℝ, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 10 ∧ (x < 1 - m ∨ x > 1 + m)) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2158_215899


namespace NUMINAMATH_CALUDE_triangle_side_difference_l2158_215823

theorem triangle_side_difference (y : ℕ) : 
  (y > 0 ∧ y + 7 > 9 ∧ y + 9 > 7 ∧ 7 + 9 > y) →
  (∃ (max min : ℕ), 
    (∀ z : ℕ, (z > 0 ∧ z + 7 > 9 ∧ z + 9 > 7 ∧ 7 + 9 > z) → z ≤ max ∧ z ≥ min) ∧
    max - min = 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l2158_215823


namespace NUMINAMATH_CALUDE_loom_weaving_rate_l2158_215836

/-- The rate at which an industrial loom weaves cloth, given the total amount of cloth woven and the time taken. -/
theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) (h : total_cloth = 26 ∧ total_time = 203.125) :
  total_cloth / total_time = 0.128 := by
sorry

end NUMINAMATH_CALUDE_loom_weaving_rate_l2158_215836


namespace NUMINAMATH_CALUDE_banana_pies_count_l2158_215833

def total_pies : ℕ := 30
def ratio_sum : ℕ := 2 + 5 + 3

theorem banana_pies_count :
  let banana_ratio : ℕ := 3
  (banana_ratio * total_pies) / ratio_sum = 9 :=
by sorry

end NUMINAMATH_CALUDE_banana_pies_count_l2158_215833


namespace NUMINAMATH_CALUDE_nell_baseball_cards_count_l2158_215805

/-- Represents the number of cards Nell has --/
structure NellCards where
  initialBaseball : Nat
  initialAce : Nat
  currentAce : Nat
  baseballDifference : Nat

/-- Calculates the current number of baseball cards Nell has --/
def currentBaseballCards (cards : NellCards) : Nat :=
  cards.currentAce + cards.baseballDifference

/-- Theorem stating that Nell's current baseball cards equal 178 --/
theorem nell_baseball_cards_count (cards : NellCards) 
  (h1 : cards.initialBaseball = 438)
  (h2 : cards.initialAce = 18)
  (h3 : cards.currentAce = 55)
  (h4 : cards.baseballDifference = 123) :
  currentBaseballCards cards = 178 := by
  sorry


end NUMINAMATH_CALUDE_nell_baseball_cards_count_l2158_215805


namespace NUMINAMATH_CALUDE_fraction_equality_l2158_215847

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (4*a - b) / (a + 4*b) = 3) : 
  (a - 4*b) / (4*a + b) = 9 / 53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2158_215847


namespace NUMINAMATH_CALUDE_gumdrop_cost_l2158_215887

/-- Given a total amount of 224 cents and the ability to buy 28 gumdrops,
    prove that the cost of each gumdrop is 8 cents. -/
theorem gumdrop_cost (total : ℕ) (quantity : ℕ) (h1 : total = 224) (h2 : quantity = 28) :
  total / quantity = 8 := by
  sorry

end NUMINAMATH_CALUDE_gumdrop_cost_l2158_215887


namespace NUMINAMATH_CALUDE_expression_value_l2158_215851

theorem expression_value (a b : ℝ) (h : (a - 3)^2 + |b + 2| = 0) :
  (-a^2 + 3*a*b - 3*b^2) - 2*(-1/2*a^2 + 4*a*b - 3/2*b^2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2158_215851


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_for_1000_l2158_215828

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k ≤ d - 1 ∧ (n - k) % d = 0 ∧
  ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem least_subtraction_for_1000 :
  ∃ (k : Nat), k = 398 ∧ 
  (427398 - k) % 1000 = 0 ∧
  ∀ (m : Nat), m < k → (427398 - m) % 1000 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_for_1000_l2158_215828


namespace NUMINAMATH_CALUDE_range_of_a_l2158_215870

def p (x : ℝ) : Prop := |4 - x| ≤ 6

def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≥ 0

def sufficient_not_necessary (P Q : ℝ → Prop) : Prop :=
  (∀ x, ¬(P x) → Q x) ∧ ∃ x, Q x ∧ P x

theorem range_of_a :
  ∀ a : ℝ, 
    (a > 0 ∧ 
     sufficient_not_necessary p (q · a)) →
    (0 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2158_215870


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l2158_215804

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 + a 13 = 10) : 
  a 3 + a 5 + a 7 + a 9 + a 11 = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l2158_215804


namespace NUMINAMATH_CALUDE_max_profit_is_4900_l2158_215848

/-- A transportation problem with two types of trucks --/
structure TransportProblem where
  driversAvailable : ℕ
  workersAvailable : ℕ
  typeATrucks : ℕ
  typeBTrucks : ℕ
  typeATruckCapacity : ℕ
  typeBTruckCapacity : ℕ
  minTonsToTransport : ℕ
  typeAWorkersRequired : ℕ
  typeBWorkersRequired : ℕ
  typeAProfit : ℕ
  typeBProfit : ℕ

/-- The solution to the transportation problem --/
structure TransportSolution where
  typeATrucksUsed : ℕ
  typeBTrucksUsed : ℕ

/-- Calculate the profit for a given solution --/
def calculateProfit (p : TransportProblem) (s : TransportSolution) : ℕ :=
  p.typeAProfit * s.typeATrucksUsed + p.typeBProfit * s.typeBTrucksUsed

/-- Check if a solution is valid for a given problem --/
def isValidSolution (p : TransportProblem) (s : TransportSolution) : Prop :=
  s.typeATrucksUsed ≤ p.typeATrucks ∧
  s.typeBTrucksUsed ≤ p.typeBTrucks ∧
  s.typeATrucksUsed * p.typeAWorkersRequired + s.typeBTrucksUsed * p.typeBWorkersRequired ≤ p.workersAvailable ∧
  s.typeATrucksUsed * p.typeATruckCapacity + s.typeBTrucksUsed * p.typeBTruckCapacity ≥ p.minTonsToTransport

/-- The main theorem stating that the maximum profit is 4900 yuan --/
theorem max_profit_is_4900 (p : TransportProblem)
  (h1 : p.driversAvailable = 12)
  (h2 : p.workersAvailable = 19)
  (h3 : p.typeATrucks = 8)
  (h4 : p.typeBTrucks = 7)
  (h5 : p.typeATruckCapacity = 10)
  (h6 : p.typeBTruckCapacity = 6)
  (h7 : p.minTonsToTransport = 72)
  (h8 : p.typeAWorkersRequired = 2)
  (h9 : p.typeBWorkersRequired = 1)
  (h10 : p.typeAProfit = 450)
  (h11 : p.typeBProfit = 350) :
  ∃ (s : TransportSolution), isValidSolution p s ∧ 
  calculateProfit p s = 4900 ∧ 
  ∀ (s' : TransportSolution), isValidSolution p s' → calculateProfit p s' ≤ 4900 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_is_4900_l2158_215848


namespace NUMINAMATH_CALUDE_painted_cubes_equal_iff_n_eq_4_l2158_215839

/-- A cube with edge length n and two opposite faces painted black -/
structure PaintedCube where
  n : ℕ
  h_n_gt_3 : n > 3

/-- The number of unit cubes with exactly one face painted black -/
def one_face_painted (c : PaintedCube) : ℕ := 2 * (c.n - 2)^2

/-- The number of unit cubes with exactly two faces painted black -/
def two_faces_painted (c : PaintedCube) : ℕ := 4 * (c.n - 2)

/-- The theorem stating that the number of unit cubes with one face painted
    equals the number of unit cubes with two faces painted if and only if n = 4 -/
theorem painted_cubes_equal_iff_n_eq_4 (c : PaintedCube) :
  one_face_painted c = two_faces_painted c ↔ c.n = 4 :=
sorry

end NUMINAMATH_CALUDE_painted_cubes_equal_iff_n_eq_4_l2158_215839


namespace NUMINAMATH_CALUDE_original_list_size_l2158_215882

theorem original_list_size (n : ℕ) (m : ℚ) : 
  (m + 3) * (n + 1) = m * n + 20 →
  (m + 1) * (n + 2) = m * n + 21 →
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_original_list_size_l2158_215882


namespace NUMINAMATH_CALUDE_painting_time_l2158_215892

theorem painting_time (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : 
  total_rooms = 10 → time_per_room = 8 → painted_rooms = 8 → 
  (total_rooms - painted_rooms) * time_per_room = 16 := by
sorry

end NUMINAMATH_CALUDE_painting_time_l2158_215892


namespace NUMINAMATH_CALUDE_animal_path_distance_l2158_215859

theorem animal_path_distance : 
  let outer_radius : ℝ := 25
  let middle_radius : ℝ := 15
  let inner_radius : ℝ := 5
  let outer_arc : ℝ := (1/4) * 2 * Real.pi * outer_radius
  let middle_to_outer : ℝ := outer_radius - middle_radius
  let middle_arc : ℝ := (1/4) * 2 * Real.pi * middle_radius
  let to_center_and_back : ℝ := 2 * middle_radius
  let middle_to_inner : ℝ := middle_radius - inner_radius
  outer_arc + middle_to_outer + middle_arc + to_center_and_back + middle_arc + middle_to_inner = 27.5 * Real.pi + 50 := by
  sorry

end NUMINAMATH_CALUDE_animal_path_distance_l2158_215859


namespace NUMINAMATH_CALUDE_periodic_decimal_to_fraction_l2158_215838

theorem periodic_decimal_to_fraction :
  (∃ (x : ℚ), x = 2 + (3 * (2 / 99))) → (2 + (3 * (2 / 99)) = 68 / 33) := by
  sorry

end NUMINAMATH_CALUDE_periodic_decimal_to_fraction_l2158_215838


namespace NUMINAMATH_CALUDE_friends_at_reception_l2158_215850

def wedding_reception (total_guests : ℕ) (bride_couples : ℕ) (groom_couples : ℕ) : ℕ :=
  total_guests - 2 * (bride_couples + groom_couples)

theorem friends_at_reception :
  wedding_reception 300 30 30 = 180 := by
  sorry

end NUMINAMATH_CALUDE_friends_at_reception_l2158_215850


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l2158_215811

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 11*n + 30 ≤ 0 ∧ 
  (∀ (m : ℤ), m^2 - 11*m + 30 ≤ 0 → m ≤ n) ∧
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l2158_215811


namespace NUMINAMATH_CALUDE_james_muffins_count_l2158_215865

def arthur_muffins : ℕ := 115
def james_multiplier : ℚ := 12.5

theorem james_muffins_count :
  ⌈(arthur_muffins : ℚ) * james_multiplier⌉ = 1438 := by
  sorry

end NUMINAMATH_CALUDE_james_muffins_count_l2158_215865


namespace NUMINAMATH_CALUDE_total_wheels_on_floor_l2158_215854

theorem total_wheels_on_floor (num_people : ℕ) (wheels_per_skate : ℕ) (skates_per_person : ℕ) : 
  num_people = 40 → 
  wheels_per_skate = 2 → 
  skates_per_person = 2 → 
  num_people * wheels_per_skate * skates_per_person = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_on_floor_l2158_215854


namespace NUMINAMATH_CALUDE_tax_discount_commute_mathville_problem_l2158_215826

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_price : 0 < price) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) := by
  sorry

/-- Calculates Bob's method: tax first, then discount --/
def bob_method (price tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate)

/-- Calculates Alice's method: discount first, then tax --/
def alice_method (price tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

theorem mathville_problem (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h_tax : tax_rate = 0.08) (h_discount : discount_rate = 0.25) (h_price : price = 120) :
  bob_method price tax_rate discount_rate - alice_method price tax_rate discount_rate = 0 := by
  sorry

end NUMINAMATH_CALUDE_tax_discount_commute_mathville_problem_l2158_215826


namespace NUMINAMATH_CALUDE_first_movie_duration_l2158_215832

/-- Represents the duration of a movie marathon with three movies --/
structure MovieMarathon where
  first_movie : ℝ
  second_movie : ℝ
  third_movie : ℝ

/-- Defines the conditions of the movie marathon --/
def valid_marathon (m : MovieMarathon) : Prop :=
  m.second_movie = 1.5 * m.first_movie ∧
  m.third_movie = m.first_movie + m.second_movie - 1 ∧
  m.first_movie + m.second_movie + m.third_movie = 9

theorem first_movie_duration :
  ∀ m : MovieMarathon, valid_marathon m → m.first_movie = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_movie_duration_l2158_215832


namespace NUMINAMATH_CALUDE_triangle_area_l2158_215864

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  (1/2) * a * b = 54 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2158_215864


namespace NUMINAMATH_CALUDE_square_sum_from_linear_equations_l2158_215849

theorem square_sum_from_linear_equations (x y : ℝ) 
  (eq1 : x + y = 12) 
  (eq2 : 3 * x + y = 20) : 
  x^2 + y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_linear_equations_l2158_215849


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2158_215855

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℤ, x^2021 + 1 = (x^12 - x^9 + x^6 - x^3 + 1) * q + (-x^4 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2158_215855


namespace NUMINAMATH_CALUDE_angle_through_point_neg_pi_fourth_l2158_215876

/-- If the terminal side of angle α passes through the point (1, -1), 
    then α = -π/4 + 2kπ for some k ∈ ℤ, and specifically α = -π/4 when k = 0. -/
theorem angle_through_point_neg_pi_fourth (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 1 ∧ r * Real.sin α = -1) →
  (∃ (k : ℤ), α = -π/4 + 2 * k * π) ∧ 
  (α = -π/4 ∨ α = -π/4 + 2 * π ∨ α = -π/4 - 2 * π) :=
sorry

end NUMINAMATH_CALUDE_angle_through_point_neg_pi_fourth_l2158_215876


namespace NUMINAMATH_CALUDE_apple_distribution_l2158_215817

/-- The number of apples to be distributed -/
def total_apples : ℕ := 30

/-- The number of people receiving apples -/
def num_people : ℕ := 3

/-- The minimum number of apples each person must receive -/
def min_apples : ℕ := 3

/-- The number of ways to distribute the apples -/
def distribution_ways : ℕ := (total_apples - num_people * min_apples + num_people - 1).choose (num_people - 1)

theorem apple_distribution :
  distribution_ways = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l2158_215817


namespace NUMINAMATH_CALUDE_stating_max_marked_segments_correct_l2158_215867

/-- Represents an equilateral triangle divided into smaller equilateral triangles -/
structure DividedEquilateralTriangle where
  n : ℕ  -- number of parts each side is divided into

/-- 
  Given a divided equilateral triangle, returns the maximum number of unit-length segments 
  that can be marked without forming a triangle with all sides marked
-/
def max_marked_segments (t : DividedEquilateralTriangle) : ℕ :=
  t.n * (t.n + 1)

/-- 
  Theorem stating that the maximum number of marked segments in a divided equilateral triangle
  is equal to n(n+1), where n is the number of parts each side is divided into
-/
theorem max_marked_segments_correct (t : DividedEquilateralTriangle) :
  max_marked_segments t = t.n * (t.n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_stating_max_marked_segments_correct_l2158_215867


namespace NUMINAMATH_CALUDE_h_constant_l2158_215852

-- Define h as a function from ℝ to ℝ
def h : ℝ → ℝ := fun x => 5

-- State the theorem
theorem h_constant (x : ℝ) : h (3 * x + 7) = 5 := by
  sorry

end NUMINAMATH_CALUDE_h_constant_l2158_215852


namespace NUMINAMATH_CALUDE_min_width_rectangle_l2158_215822

theorem min_width_rectangle (w : ℝ) : w > 0 →
  w * (w + 20) ≥ 150 →
  ∀ x > 0, x * (x + 20) ≥ 150 → w ≤ x →
  w = 10 := by
sorry

end NUMINAMATH_CALUDE_min_width_rectangle_l2158_215822


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2158_215812

theorem inverse_sum_reciprocal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (a * b + a * c + b * c) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2158_215812


namespace NUMINAMATH_CALUDE_max_intersection_difference_l2158_215869

/-- The first function in the problem -/
def f (x : ℝ) : ℝ := 4 - x^2 + x^3

/-- The second function in the problem -/
def g (x : ℝ) : ℝ := 2 + 2*x^2 + x^3

/-- The difference between the y-coordinates of the intersection points -/
def intersection_difference (x : ℝ) : ℝ := |f x - g x|

/-- The theorem stating the maximum difference between y-coordinates of intersection points -/
theorem max_intersection_difference : 
  ∃ (x : ℝ), f x = g x ∧ 
  ∀ (y : ℝ), f y = g y → intersection_difference x ≥ intersection_difference y ∧
  intersection_difference x = 2 * (2/3)^(3/2) :=
sorry

end NUMINAMATH_CALUDE_max_intersection_difference_l2158_215869


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l2158_215878

/-- In a triangle ABC, if angle A is 2π/3 and side a is √3 times side c, then the ratio of side a to side b is √3. -/
theorem triangle_side_ratio (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  A + B + C = π →  -- angle sum property
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- valid angle measures
  A = 2 * π / 3 →  -- given angle A
  a = Real.sqrt 3 * c →  -- given side relation
  a / b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l2158_215878


namespace NUMINAMATH_CALUDE_sandwich_fraction_l2158_215814

theorem sandwich_fraction (total : ℝ) (ticket_fraction : ℝ) (book_fraction : ℝ) (leftover : ℝ) 
  (h1 : total = 90)
  (h2 : ticket_fraction = 1/6)
  (h3 : book_fraction = 1/2)
  (h4 : leftover = 12) :
  ∃ (sandwich_fraction : ℝ), 
    sandwich_fraction * total + ticket_fraction * total + book_fraction * total + leftover = total ∧ 
    sandwich_fraction = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_fraction_l2158_215814


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2158_215807

-- Define propositions p and q
def p (x y : ℝ) : Prop := x + y ≠ -2
def q (x y : ℝ) : Prop := ¬(x = -1 ∧ y = -1)

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2158_215807


namespace NUMINAMATH_CALUDE_pqr_value_l2158_215843

theorem pqr_value (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 29)
  (h3 : 1 / p + 1 / q + 1 / r + 392 / (p * q * r) = 1) :
  p * q * r = 630 := by
sorry

end NUMINAMATH_CALUDE_pqr_value_l2158_215843


namespace NUMINAMATH_CALUDE_tuesday_is_only_valid_start_day_l2158_215841

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

def advance_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => next_day (advance_days d m)

def voucher_days (start : DayOfWeek) : List DayOfWeek :=
  List.map (fun i => advance_days start (i * 7)) [0, 1, 2, 3, 4]

theorem tuesday_is_only_valid_start_day :
  ∀ (start : DayOfWeek),
    (∀ (d : DayOfWeek), d ∈ voucher_days start → d ≠ DayOfWeek.Monday) ↔
    start = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_tuesday_is_only_valid_start_day_l2158_215841


namespace NUMINAMATH_CALUDE_line_circle_separation_l2158_215819

/-- Given a point P(x₀, y₀) inside a circle C: x² + y² = r², 
    the line xx₀ + yy₀ = r² is separated from the circle C. -/
theorem line_circle_separation 
  (x₀ y₀ r : ℝ) 
  (h_inside : x₀^2 + y₀^2 < r^2) : 
  let d := r^2 / Real.sqrt (x₀^2 + y₀^2)
  d > r := by
sorry

end NUMINAMATH_CALUDE_line_circle_separation_l2158_215819


namespace NUMINAMATH_CALUDE_power_multiplication_l2158_215885

theorem power_multiplication (x : ℝ) (h : x = 5) : x^3 * x^4 = 78125 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2158_215885


namespace NUMINAMATH_CALUDE_beth_friends_count_l2158_215827

theorem beth_friends_count (initial_packs : ℝ) (additional_packs : ℝ) (final_packs : ℝ) :
  initial_packs = 4 →
  additional_packs = 6 →
  final_packs = 6.4 →
  ∃ (num_friends : ℝ),
    num_friends > 0 ∧
    final_packs = additional_packs + initial_packs / num_friends ∧
    num_friends = 10 := by
  sorry

end NUMINAMATH_CALUDE_beth_friends_count_l2158_215827


namespace NUMINAMATH_CALUDE_no_injective_function_exists_l2158_215837

theorem no_injective_function_exists : ¬∃ f : ℝ → ℝ, 
  Function.Injective f ∧ ∀ x : ℝ, f (x^2) - (f x)^2 ≥ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_no_injective_function_exists_l2158_215837


namespace NUMINAMATH_CALUDE_horner_method_v1_l2158_215840

def f (x : ℝ) : ℝ := 4 * x^5 - 12 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

def horner_v1 (a₅ a₄ : ℝ) (x : ℝ) : ℝ := a₅ * x + a₄

theorem horner_method_v1 :
  let x := 5
  let a₅ := 4
  let a₄ := -12
  horner_v1 a₅ a₄ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v1_l2158_215840


namespace NUMINAMATH_CALUDE_aldens_nephews_l2158_215835

theorem aldens_nephews (alden_now alden_past vihaan : ℕ) : 
  alden_now = 2 * alden_past →
  vihaan = alden_now + 60 →
  alden_now + vihaan = 260 →
  alden_past = 50 := by
sorry

end NUMINAMATH_CALUDE_aldens_nephews_l2158_215835


namespace NUMINAMATH_CALUDE_river_flow_speed_l2158_215871

/-- Proves that the speed of river flow is 2 km/hr given the conditions of the boat journey --/
theorem river_flow_speed 
  (distance : ℝ) 
  (boat_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : distance = 32) 
  (h2 : boat_speed = 6) 
  (h3 : total_time = 12) : 
  ∃ (v : ℝ), v = 2 ∧ 
    (distance / (boat_speed - v) + distance / (boat_speed + v) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_river_flow_speed_l2158_215871


namespace NUMINAMATH_CALUDE_number_difference_l2158_215857

theorem number_difference (S L : ℕ) (h1 : S = 270) (h2 : L = 6 * S + 15) :
  L - S = 1365 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2158_215857


namespace NUMINAMATH_CALUDE_power_of_four_l2158_215884

theorem power_of_four (x : ℕ) 
  (h1 : 2 * x + 5 + 2 = 29) : x = 11 := by
  sorry

#check power_of_four

end NUMINAMATH_CALUDE_power_of_four_l2158_215884


namespace NUMINAMATH_CALUDE_billys_age_l2158_215830

theorem billys_age (billy brenda joe : ℚ) 
  (h1 : billy = 3 * brenda)
  (h2 : billy = 2 * joe)
  (h3 : billy + brenda + joe = 72) :
  billy = 432 / 11 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l2158_215830


namespace NUMINAMATH_CALUDE_projectile_max_height_l2158_215800

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- Theorem stating that the maximum height of the projectile is 161 meters -/
theorem projectile_max_height :
  ∃ t : ℝ, h t = 161 ∧ ∀ s : ℝ, h s ≤ h t :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2158_215800


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l2158_215862

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) :
  a + b ≥ 16 + (4/3) * Real.rpow 3 (1/3) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l2158_215862


namespace NUMINAMATH_CALUDE_andrea_wins_stick_game_l2158_215866

-- Define the game setup
def num_sticks : Nat := 98
def stick_lengths : List Nat := List.range num_sticks

-- Define a function to check if three sticks can form a triangle
def can_form_triangle (a b c : Nat) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the game state
structure GameState where
  remaining_sticks : List Nat
  current_player : Bool  -- true for Andrea, false for Béla

-- Define the winning condition for Andrea
def andrea_wins (final_sticks : List Nat) : Prop :=
  final_sticks.length = 3 ∧
  ∃ (a b c : Nat), a ∈ final_sticks ∧ b ∈ final_sticks ∧ c ∈ final_sticks ∧
    can_form_triangle a b c

-- Define a winning strategy for Andrea
def andrea_has_winning_strategy : Prop :=
  ∃ (strategy : GameState → Nat),
    ∀ (game : GameState),
      game.current_player = true →
      game.remaining_sticks.length > 3 →
      ∃ (next_game : GameState),
        next_game.remaining_sticks = game.remaining_sticks.erase (strategy game) ∧
        next_game.current_player = false ∧
        (next_game.remaining_sticks.length = 3 → andrea_wins next_game.remaining_sticks)

-- Theorem statement
theorem andrea_wins_stick_game :
  andrea_has_winning_strategy :=
sorry

end NUMINAMATH_CALUDE_andrea_wins_stick_game_l2158_215866


namespace NUMINAMATH_CALUDE_weight_increase_percentage_l2158_215858

-- Define the initial weight James can lift for 20 meters
def initial_weight : ℝ := 300

-- Define the weight increase for 20 meters
def weight_increase : ℝ := 50

-- Define the strap increase percentage
def strap_increase : ℝ := 0.20

-- Define the final weight James can lift with straps for 10 meters
def final_weight : ℝ := 546

-- Define the function to calculate the weight James can lift for 10 meters with straps
def weight_with_straps (p : ℝ) : ℝ :=
  (initial_weight + weight_increase) * (1 + p) * (1 + strap_increase)

-- Theorem to prove
theorem weight_increase_percentage :
  ∃ p : ℝ, weight_with_straps p = final_weight ∧ p = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_weight_increase_percentage_l2158_215858


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2158_215883

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}

theorem intersection_with_complement : S ∩ (U \ T) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2158_215883


namespace NUMINAMATH_CALUDE_no_rectangular_prism_with_diagonals_7_8_11_l2158_215821

theorem no_rectangular_prism_with_diagonals_7_8_11 :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    ({7^2, 8^2, 11^2} : Finset ℝ) = {a^2 + b^2, b^2 + c^2, a^2 + c^2} :=
by sorry

end NUMINAMATH_CALUDE_no_rectangular_prism_with_diagonals_7_8_11_l2158_215821


namespace NUMINAMATH_CALUDE_bank_queue_properties_l2158_215802

/-- Represents a bank queue with simple and long operations -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum possible total wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum possible total wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected value of wasted person-minutes assuming random order -/
def expected_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Theorem stating the properties of the bank queue problem -/
theorem bank_queue_properties (q : BankQueue) 
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧ 
  max_wasted_time q = 100 ∧ 
  expected_wasted_time q = 84 :=
by sorry

end NUMINAMATH_CALUDE_bank_queue_properties_l2158_215802


namespace NUMINAMATH_CALUDE_triangle_otimes_calculation_l2158_215877

def triangle (a b : ℝ) : ℝ := a + b + a * b - 1

def otimes (a b : ℝ) : ℝ := a^2 - a * b + b^2

theorem triangle_otimes_calculation : triangle 3 (otimes 2 4) = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_otimes_calculation_l2158_215877


namespace NUMINAMATH_CALUDE_seven_solutions_condition_l2158_215834

-- Define the function f
def f (x : ℝ) : ℝ := |x^2 - 1| - 1

-- State the theorem
theorem seven_solutions_condition (b c : ℝ) :
  (∃! (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, f x ^ 2 - b * f x + c = 0) ↔ 
  (c ≤ 0 ∧ 0 < b ∧ b < 1) :=
sorry

end NUMINAMATH_CALUDE_seven_solutions_condition_l2158_215834


namespace NUMINAMATH_CALUDE_circular_fields_area_difference_l2158_215888

theorem circular_fields_area_difference (r₁ r₂ : ℝ) (h : r₁ / r₂ = 3 / 10) :
  1 - (π * r₁^2) / (π * r₂^2) = 91 / 100 := by
  sorry

end NUMINAMATH_CALUDE_circular_fields_area_difference_l2158_215888


namespace NUMINAMATH_CALUDE_dilation_transforms_line_l2158_215829

-- Define the original line
def original_line (x y : ℝ) : Prop := x + y = 1

-- Define the transformed line
def transformed_line (x y : ℝ) : Prop := 2*x + 3*y = 6

-- Define the dilation transformation
def dilation (x y : ℝ) : ℝ × ℝ := (2*x, 3*y)

-- Theorem statement
theorem dilation_transforms_line :
  ∀ x y : ℝ, original_line x y → transformed_line (dilation x y).1 (dilation x y).2 := by
  sorry

end NUMINAMATH_CALUDE_dilation_transforms_line_l2158_215829


namespace NUMINAMATH_CALUDE_qr_length_l2158_215873

-- Define a right triangle
structure RightTriangle where
  QP : ℝ
  QR : ℝ
  cosQ : ℝ
  right_angle : cosQ = QP / QR

-- Theorem statement
theorem qr_length (t : RightTriangle) (h1 : t.cosQ = 0.5) (h2 : t.QP = 10) : t.QR = 20 := by
  sorry

end NUMINAMATH_CALUDE_qr_length_l2158_215873


namespace NUMINAMATH_CALUDE_joan_balloons_l2158_215898

def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2

theorem joan_balloons : initial_balloons - lost_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l2158_215898


namespace NUMINAMATH_CALUDE_inequality_proof_l2158_215893

theorem inequality_proof (x a : ℝ) (hx : x > 0) (ha : a > 0) :
  (1 / Real.sqrt (x + 1)) + (1 / Real.sqrt (a + 1)) + Real.sqrt (a * x / (a * x + 8)) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2158_215893


namespace NUMINAMATH_CALUDE_series_sum_theorem_l2158_215874

/-- The sum of the infinite series (2n+1)x^n from n=0 to infinity -/
noncomputable def S (x : ℝ) : ℝ := ∑' n, (2 * n + 1) * x^n

/-- Theorem stating that if S(x) = 16, then x = (4 - √2) / 4 -/
theorem series_sum_theorem (x : ℝ) (hx : S x = 16) : x = (4 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_theorem_l2158_215874


namespace NUMINAMATH_CALUDE_student_failed_marks_l2158_215815

def total_marks : ℕ := 500
def passing_percentage : ℚ := 33 / 100
def student_marks : ℕ := 125

theorem student_failed_marks :
  (total_marks * passing_percentage).floor - student_marks = 40 :=
by sorry

end NUMINAMATH_CALUDE_student_failed_marks_l2158_215815


namespace NUMINAMATH_CALUDE_black_balls_count_l2158_215861

theorem black_balls_count (red white : ℕ) (p : ℚ) (black : ℕ) : 
  red = 3 → 
  white = 5 → 
  p = 1/4 → 
  (white : ℚ) / ((red : ℚ) + (white : ℚ) + (black : ℚ)) = p → 
  black = 12 := by
sorry

end NUMINAMATH_CALUDE_black_balls_count_l2158_215861


namespace NUMINAMATH_CALUDE_rank_difference_bound_l2158_215803

variable (n : ℕ) 
variable (hn : n ≥ 2)

theorem rank_difference_bound 
  (X Y : Matrix (Fin n) (Fin n) ℂ) : 
  Matrix.rank (X * Y) - Matrix.rank (Y * X) ≤ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_rank_difference_bound_l2158_215803


namespace NUMINAMATH_CALUDE_jack_evening_emails_l2158_215853

/-- The number of emails Jack received in a day -/
def total_emails : ℕ := 10

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := total_emails - (afternoon_emails + morning_emails)

theorem jack_evening_emails : evening_emails = 1 := by
  sorry

end NUMINAMATH_CALUDE_jack_evening_emails_l2158_215853


namespace NUMINAMATH_CALUDE_berry_tuesday_temperature_l2158_215856

def berry_temperatures : List Float := [99.1, 98.2, 99.3, 99.8, 99, 98.9]
def average_temperature : Float := 99
def days_in_week : Nat := 7

theorem berry_tuesday_temperature :
  let total_sum : Float := average_temperature * days_in_week.toFloat
  let known_sum : Float := berry_temperatures.sum
  let tuesday_temp : Float := total_sum - known_sum
  tuesday_temp = 98.7 := by sorry

end NUMINAMATH_CALUDE_berry_tuesday_temperature_l2158_215856


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l2158_215808

theorem units_digit_of_quotient (n : ℕ) : (4^1993 + 5^1993) % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l2158_215808
