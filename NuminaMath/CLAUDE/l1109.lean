import Mathlib

namespace NUMINAMATH_CALUDE_min_value_problem_l1109_110979

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 3) + 1 / (b + 3) = 1 / 4 → 
  x + 3 * y ≤ a + 3 * b ∧ x + 3 * y = 19 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l1109_110979


namespace NUMINAMATH_CALUDE_calculation_proof_l1109_110931

theorem calculation_proof : (-49 : ℚ) * (4/7) - (4/7) / (-8/7) = -55/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1109_110931


namespace NUMINAMATH_CALUDE_probability_third_draw_defective_10_3_l1109_110910

/-- Given a set of products with some defective ones, this function calculates
    the probability of drawing a defective product on the third draw, given
    that the first draw was defective. -/
def probability_third_draw_defective (total_products : ℕ) (defective_products : ℕ) : ℚ :=
  if total_products < 3 ∨ defective_products < 1 ∨ defective_products > total_products then 0
  else
    let remaining_after_first := total_products - 1
    let defective_after_first := defective_products - 1
    let numerator := (remaining_after_first - defective_after_first) * defective_after_first +
                     defective_after_first * (defective_after_first - 1)
    let denominator := remaining_after_first * (remaining_after_first - 1)
    ↑numerator / ↑denominator

/-- Theorem stating that for 10 products with 3 defective ones, the probability
    of drawing a defective product on the third draw, given that the first
    draw was defective, is 2/9. -/
theorem probability_third_draw_defective_10_3 :
  probability_third_draw_defective 10 3 = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_third_draw_defective_10_3_l1109_110910


namespace NUMINAMATH_CALUDE_log_weight_after_cutting_l1109_110992

/-- Given a log of length 20 feet that is cut in half, where each linear foot weighs 150 pounds,
    prove that each cut piece weighs 1500 pounds. -/
theorem log_weight_after_cutting (original_length : ℝ) (weight_per_foot : ℝ) :
  original_length = 20 →
  weight_per_foot = 150 →
  (original_length / 2) * weight_per_foot = 1500 := by
  sorry

#check log_weight_after_cutting

end NUMINAMATH_CALUDE_log_weight_after_cutting_l1109_110992


namespace NUMINAMATH_CALUDE_class_A_student_count_l1109_110916

/-- The number of students who like social studies -/
def social_studies_count : ℕ := 25

/-- The number of students who like music -/
def music_count : ℕ := 32

/-- The number of students who like both social studies and music -/
def both_count : ℕ := 27

/-- The total number of students in class (A) -/
def total_students : ℕ := social_studies_count + music_count - both_count

theorem class_A_student_count :
  total_students = 30 :=
sorry

end NUMINAMATH_CALUDE_class_A_student_count_l1109_110916


namespace NUMINAMATH_CALUDE_sum_of_absolute_b_values_l1109_110972

-- Define the polynomials p and q
def p (a b x : ℝ) : ℝ := x^3 + a*x + b
def q (a b x : ℝ) : ℝ := x^3 + a*x + b + 240

-- Define the theorem
theorem sum_of_absolute_b_values (a b r s : ℝ) : 
  (p a b r = 0) → 
  (p a b s = 0) → 
  (q a b (r + 4) = 0) → 
  (q a b (s - 3) = 0) → 
  (∃ b₁ b₂ : ℝ, (b = b₁ ∨ b = b₂) ∧ (|b₁| + |b₂| = 62)) := by
sorry


end NUMINAMATH_CALUDE_sum_of_absolute_b_values_l1109_110972


namespace NUMINAMATH_CALUDE_minimum_value_implies_ratio_l1109_110958

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sin x * Real.cos x

theorem minimum_value_implies_ratio (θ : ℝ) 
  (h : ∀ x, f x ≥ f θ) : 
  (Real.sin (2 * θ) + 2 * Real.cos θ) / (Real.sin (2 * θ) - 2 * Real.cos (2 * θ)) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_ratio_l1109_110958


namespace NUMINAMATH_CALUDE_ratio_sum_over_y_l1109_110974

theorem ratio_sum_over_y (x y : ℚ) (h : x / y = 1 / 2) : (x + y) / y = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_over_y_l1109_110974


namespace NUMINAMATH_CALUDE_sum_interior_angles_num_diagonals_l1109_110959

/-- A regular polygon with exterior angles measuring 20° -/
structure RegularPolygon20 where
  n : ℕ
  exterior_angle : ℝ
  h_exterior : exterior_angle = 20

/-- The sum of interior angles of a regular polygon with 20° exterior angles is 2880° -/
theorem sum_interior_angles (p : RegularPolygon20) : 
  (p.n - 2) * 180 = 2880 := by sorry

/-- The number of diagonals in a regular polygon with 20° exterior angles is 135 -/
theorem num_diagonals (p : RegularPolygon20) : 
  p.n * (p.n - 3) / 2 = 135 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_num_diagonals_l1109_110959


namespace NUMINAMATH_CALUDE_largest_six_digit_divisible_by_88_l1109_110954

theorem largest_six_digit_divisible_by_88 : ∃ n : ℕ, 
  n ≤ 999999 ∧ 
  n ≥ 100000 ∧
  n % 88 = 0 ∧
  ∀ m : ℕ, m ≤ 999999 ∧ m ≥ 100000 ∧ m % 88 = 0 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_six_digit_divisible_by_88_l1109_110954


namespace NUMINAMATH_CALUDE_beverage_total_cups_l1109_110942

/-- Represents the ratio of ingredients in the beverage -/
structure BeverageRatio where
  milk : ℕ
  coffee : ℕ
  sugar : ℕ

/-- Calculates the total number of cups given a ratio and the number of coffee cups -/
def totalCups (ratio : BeverageRatio) (coffeeCups : ℕ) : ℕ :=
  let partSize := coffeeCups / ratio.coffee
  partSize * (ratio.milk + ratio.coffee + ratio.sugar)

/-- Theorem stating that for the given ratio and coffee amount, the total is 18 cups -/
theorem beverage_total_cups :
  let ratio : BeverageRatio := { milk := 3, coffee := 2, sugar := 1 }
  let coffeeCups : ℕ := 6
  totalCups ratio coffeeCups = 18 := by
  sorry


end NUMINAMATH_CALUDE_beverage_total_cups_l1109_110942


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l1109_110940

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisibility : 
  (∀ n : ℕ, n < 6303 → ¬(is_divisible (n + 3) 18 ∧ is_divisible (n + 3) 1051 ∧ is_divisible (n + 3) 100 ∧ is_divisible (n + 3) 21)) ∧
  (is_divisible (6303 + 3) 18 ∧ is_divisible (6303 + 3) 1051 ∧ is_divisible (6303 + 3) 100 ∧ is_divisible (6303 + 3) 21) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l1109_110940


namespace NUMINAMATH_CALUDE_integral_tan_sin_l1109_110983

open Real MeasureTheory

theorem integral_tan_sin : ∫ (x : ℝ) in Real.arcsin (2 / Real.sqrt 5)..Real.arcsin (3 / Real.sqrt 10), 
  (2 * Real.tan x + 5) / ((5 - Real.tan x) * Real.sin (2 * x)) = 2 * Real.log (3 / 2) := by sorry

end NUMINAMATH_CALUDE_integral_tan_sin_l1109_110983


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1109_110973

theorem binomial_expansion_sum (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (a - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ = 80 →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1109_110973


namespace NUMINAMATH_CALUDE_jerry_grocery_shopping_l1109_110927

/-- The amount of money Jerry has left after grocery shopping -/
def money_left (mustard_oil_price mustard_oil_quantity pasta_price pasta_quantity sauce_price sauce_quantity total_money : ℕ) : ℕ :=
  total_money - (mustard_oil_price * mustard_oil_quantity + pasta_price * pasta_quantity + sauce_price * sauce_quantity)

/-- Theorem stating that Jerry will have $7 left after grocery shopping -/
theorem jerry_grocery_shopping :
  money_left 13 2 4 3 5 1 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_grocery_shopping_l1109_110927


namespace NUMINAMATH_CALUDE_average_marks_math_chem_l1109_110933

theorem average_marks_math_chem (M P C B : ℕ) : 
  M + P = 80 →
  C + B = 120 →
  C = P + 20 →
  B = M - 15 →
  (M + C) / 2 = 50 := by
sorry

end NUMINAMATH_CALUDE_average_marks_math_chem_l1109_110933


namespace NUMINAMATH_CALUDE_triangle_inequality_condition_l1109_110994

theorem triangle_inequality_condition (m : ℝ) :
  (m > 0) →
  (∀ (x y : ℝ), x > 0 → y > 0 →
    (x + y + m * Real.sqrt (x * y) > Real.sqrt (x^2 + y^2 + x * y) ∧
     x + y + Real.sqrt (x^2 + y^2 + x * y) > m * Real.sqrt (x * y) ∧
     m * Real.sqrt (x * y) + Real.sqrt (x^2 + y^2 + x * y) > x + y)) ↔
  (m > 2 - Real.sqrt 3 ∧ m < 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_condition_l1109_110994


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l1109_110985

/-- The time when a ball hits the ground given its height equation -/
theorem ball_hit_ground_time (t : ℝ) : 
  let y : ℝ → ℝ := λ t => -4.9 * t^2 + 4 * t + 6
  y t = 0 → t = 78 / 49 := by
sorry

end NUMINAMATH_CALUDE_ball_hit_ground_time_l1109_110985


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1109_110984

theorem simplify_fraction_product : 5 * (12 / 7) * (49 / -60) = -7 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1109_110984


namespace NUMINAMATH_CALUDE_problem_statement_l1109_110946

theorem problem_statement (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 5*t + 6) 
  (h3 : x = 1) : 
  y = 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1109_110946


namespace NUMINAMATH_CALUDE_smaller_special_integer_l1109_110909

/-- Two positive three-digit integers satisfying the given condition -/
def SpecialIntegers (m n : ℕ) : Prop :=
  100 ≤ m ∧ m < 1000 ∧
  100 ≤ n ∧ n < 1000 ∧
  (m + n) / 2 = m + n / 200

/-- The smaller of two integers satisfying the condition is 891 -/
theorem smaller_special_integer (m n : ℕ) (h : SpecialIntegers m n) : 
  min m n = 891 := by
  sorry

#check smaller_special_integer

end NUMINAMATH_CALUDE_smaller_special_integer_l1109_110909


namespace NUMINAMATH_CALUDE_five_player_tournament_games_l1109_110960

/-- The number of games in a tournament where each player plays every other player once -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 5 players, where each player plays against every other player
    exactly once, the total number of games is 10 -/
theorem five_player_tournament_games :
  tournament_games 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_five_player_tournament_games_l1109_110960


namespace NUMINAMATH_CALUDE_X_is_greatest_l1109_110990

def X : ℚ := 2010/2009 + 2010/2011
def Y : ℚ := 2010/2011 + 2012/2011
def Z : ℚ := 2011/2010 + 2011/2012

theorem X_is_greatest : X > Y ∧ X > Z := by
  sorry

end NUMINAMATH_CALUDE_X_is_greatest_l1109_110990


namespace NUMINAMATH_CALUDE_only_rectangle_area_certain_l1109_110982

-- Define the events
inductive Event
  | WaterFreeze : Event
  | ExamScore : Event
  | CoinToss : Event
  | RectangleArea : Event

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.WaterFreeze => False
  | Event.ExamScore => False
  | Event.CoinToss => False
  | Event.RectangleArea => True

-- Theorem stating that only RectangleArea is a certain event
theorem only_rectangle_area_certain :
  ∀ (e : Event), is_certain e ↔ e = Event.RectangleArea :=
by sorry

end NUMINAMATH_CALUDE_only_rectangle_area_certain_l1109_110982


namespace NUMINAMATH_CALUDE_mn_inequality_l1109_110998

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Define the set P
def P : Set ℝ := {x | f x > 4}

-- State the theorem
theorem mn_inequality (m n : ℝ) (hm : m ∈ P) (hn : n ∈ P) :
  |m * n + 4| > 2 * |m + n| := by
  sorry

end NUMINAMATH_CALUDE_mn_inequality_l1109_110998


namespace NUMINAMATH_CALUDE_orange_profit_maximization_l1109_110901

/-- Represents the cost and selling prices of oranges --/
structure OrangePrices where
  cost_a : ℝ
  sell_a : ℝ
  cost_b : ℝ
  sell_b : ℝ

/-- Represents a purchasing plan for oranges --/
structure PurchasePlan where
  kg_a : ℕ
  kg_b : ℕ

/-- Calculates the total cost of a purchase plan --/
def total_cost (prices : OrangePrices) (plan : PurchasePlan) : ℝ :=
  prices.cost_a * plan.kg_a + prices.cost_b * plan.kg_b

/-- Calculates the profit of a purchase plan --/
def profit (prices : OrangePrices) (plan : PurchasePlan) : ℝ :=
  (prices.sell_a - prices.cost_a) * plan.kg_a + (prices.sell_b - prices.cost_b) * plan.kg_b

/-- The main theorem to prove --/
theorem orange_profit_maximization (prices : OrangePrices) 
    (h1 : prices.sell_a = 16)
    (h2 : prices.sell_b = 24)
    (h3 : total_cost prices {kg_a := 15, kg_b := 20} = 430)
    (h4 : total_cost prices {kg_a := 10, kg_b := 8} = 212)
    (h5 : ∀ plan : PurchasePlan, plan.kg_a + plan.kg_b = 100 → 
      1160 ≤ total_cost prices plan ∧ total_cost prices plan ≤ 1168) :
  prices.cost_a = 10 ∧ 
  prices.cost_b = 14 ∧
  (∀ plan : PurchasePlan, plan.kg_a + plan.kg_b = 100 → 
    profit prices plan ≤ profit prices {kg_a := 58, kg_b := 42}) ∧
  profit prices {kg_a := 58, kg_b := 42} = 768 := by
  sorry


end NUMINAMATH_CALUDE_orange_profit_maximization_l1109_110901


namespace NUMINAMATH_CALUDE_flower_shop_purchase_l1109_110937

theorem flower_shop_purchase 
  (total_flowers : ℕ) 
  (total_cost : ℚ) 
  (carnation_price : ℚ) 
  (rose_price : ℚ) 
  (h1 : total_flowers = 400)
  (h2 : total_cost = 1020)
  (h3 : carnation_price = 6/5)  -- $1.2 as a rational number
  (h4 : rose_price = 3) :
  ∃ (carnations roses : ℕ),
    carnations + roses = total_flowers ∧
    carnation_price * carnations + rose_price * roses = total_cost ∧
    carnations = 100 ∧
    roses = 300 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_purchase_l1109_110937


namespace NUMINAMATH_CALUDE_angle_bisector_sum_geq_nine_times_inradius_l1109_110935

/-- A triangle with an incircle -/
structure TriangleWithIncircle where
  /-- The radius of the incircle -/
  r : ℝ
  /-- The first angle bisector -/
  f_a : ℝ
  /-- The second angle bisector -/
  f_b : ℝ
  /-- The third angle bisector -/
  f_c : ℝ
  /-- Assumption that r is positive -/
  r_pos : r > 0
  /-- Assumption that angle bisectors are positive -/
  f_a_pos : f_a > 0
  f_b_pos : f_b > 0
  f_c_pos : f_c > 0

/-- The sum of angle bisectors is greater than or equal to 9 times the incircle radius -/
theorem angle_bisector_sum_geq_nine_times_inradius (t : TriangleWithIncircle) :
  t.f_a + t.f_b + t.f_c ≥ 9 * t.r :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_geq_nine_times_inradius_l1109_110935


namespace NUMINAMATH_CALUDE_fraction_1790_1799_l1109_110904

/-- The number of states that joined the union from 1790 to 1799 -/
def states_1790_1799 : ℕ := 10

/-- The total number of states in Sophie's collection -/
def total_states : ℕ := 25

/-- The fraction of states that joined from 1790 to 1799 among the first 25 states -/
theorem fraction_1790_1799 : 
  (states_1790_1799 : ℚ) / total_states = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_1790_1799_l1109_110904


namespace NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l1109_110991

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_integer_prime_quadratic :
  ∃ (x : ℤ), (∀ y : ℤ, is_prime (Int.natAbs (5 * y^2 - 42 * y + 8)) → y ≤ x) ∧
             is_prime (Int.natAbs (5 * x^2 - 42 * x + 8)) ∧
             x = 5 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l1109_110991


namespace NUMINAMATH_CALUDE_negation_at_most_one_obtuse_angle_l1109_110924

/-- Definition of a triangle -/
def Triangle : Type := Unit

/-- Definition of an obtuse angle in a triangle -/
def HasObtuseAngle (t : Triangle) : Prop := sorry

/-- Statement: There is at most one obtuse angle in a triangle -/
def AtMostOneObtuseAngle : Prop :=
  ∀ t : Triangle, ∃! a : ℕ, a ≤ 3 ∧ HasObtuseAngle t

/-- Theorem: The negation of "There is at most one obtuse angle in a triangle"
    is equivalent to "There are at least two obtuse angles." -/
theorem negation_at_most_one_obtuse_angle :
  ¬AtMostOneObtuseAngle ↔ ∃ t : Triangle, ∃ a b : ℕ, a ≠ b ∧ a ≤ 3 ∧ b ≤ 3 ∧ HasObtuseAngle t ∧ HasObtuseAngle t :=
by sorry

end NUMINAMATH_CALUDE_negation_at_most_one_obtuse_angle_l1109_110924


namespace NUMINAMATH_CALUDE_coin_game_probability_l1109_110912

/-- Represents the possible outcomes of a coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing three coins -/
structure ThreeCoinToss :=
  (first second third : CoinOutcome)

/-- Defines a winning outcome in the Coin Game -/
def is_winning_toss (toss : ThreeCoinToss) : Prop :=
  (toss.first = CoinOutcome.Heads ∧ toss.second = CoinOutcome.Heads ∧ toss.third = CoinOutcome.Heads) ∨
  (toss.first = CoinOutcome.Tails ∧ toss.second = CoinOutcome.Tails ∧ toss.third = CoinOutcome.Tails)

/-- The set of all possible outcomes when tossing three coins -/
def all_outcomes : Finset ThreeCoinToss := sorry

/-- The set of winning outcomes in the Coin Game -/
def winning_outcomes : Finset ThreeCoinToss := sorry

/-- Theorem stating that the probability of winning the Coin Game is 1/4 -/
theorem coin_game_probability : 
  (Finset.card winning_outcomes : ℚ) / (Finset.card all_outcomes : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_game_probability_l1109_110912


namespace NUMINAMATH_CALUDE_trapezoid_cd_length_l1109_110988

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of segment AB
  ab : ℝ
  -- Length of segment CD
  cd : ℝ
  -- The ratio of the area of triangle ABC to the area of triangle ADC is 5:3
  area_ratio : ab / cd = 5 / 3
  -- The sum of AB and CD is 192 cm
  sum_sides : ab + cd = 192

/-- 
Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC to the area of triangle ADC 
is 5:3, and AB + CD = 192 cm, then the length of segment CD is 72 cm.
-/
theorem trapezoid_cd_length (t : Trapezoid) : t.cd = 72 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_cd_length_l1109_110988


namespace NUMINAMATH_CALUDE_price_per_kg_correct_l1109_110975

/-- The price per kilogram of rooster -/
def price_per_kg : ℝ := 0.5

/-- The weight of the first rooster in kilograms -/
def weight1 : ℝ := 30

/-- The weight of the second rooster in kilograms -/
def weight2 : ℝ := 40

/-- The total earnings from selling both roosters -/
def total_earnings : ℝ := 35

/-- Theorem stating that the price per kilogram is correct -/
theorem price_per_kg_correct : 
  price_per_kg * (weight1 + weight2) = total_earnings := by sorry

end NUMINAMATH_CALUDE_price_per_kg_correct_l1109_110975


namespace NUMINAMATH_CALUDE_curve_is_semicircle_l1109_110944

-- Define the curve
def curve (x y : ℝ) : Prop := y - 1 = Real.sqrt (1 - x^2)

-- Define a semicircle
def is_semicircle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (c : ℝ × ℝ) (r : ℝ),
    r > 0 ∧
    S = {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 ∧ p.2 ≥ c.2}

-- Theorem statement
theorem curve_is_semicircle :
  is_semicircle {p : ℝ × ℝ | curve p.1 p.2} :=
sorry

end NUMINAMATH_CALUDE_curve_is_semicircle_l1109_110944


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1109_110921

theorem cubic_roots_sum (a b c : ℝ) (h1 : a < b) (h2 : b < c)
  (ha : a^3 - 3*a + 1 = 0) (hb : b^3 - 3*b + 1 = 0) (hc : c^3 - 3*c + 1 = 0) :
  1 / (a^2 + b) + 1 / (b^2 + c) + 1 / (c^2 + a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1109_110921


namespace NUMINAMATH_CALUDE_no_profit_after_ten_requests_l1109_110971

def genie_operation (x : ℕ) : ℕ := (x + 1000) / 2

def iterate_genie (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | m + 1 => genie_operation (iterate_genie m x)

theorem no_profit_after_ten_requests (x : ℕ) : iterate_genie 10 x ≤ x := by
  sorry


end NUMINAMATH_CALUDE_no_profit_after_ten_requests_l1109_110971


namespace NUMINAMATH_CALUDE_A_power_93_l1109_110929

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_93 : A ^ 93 = A := by sorry

end NUMINAMATH_CALUDE_A_power_93_l1109_110929


namespace NUMINAMATH_CALUDE_janelle_has_72_marbles_l1109_110923

/-- The number of marbles Janelle has after buying and gifting some marbles -/
def janelles_marbles : ℕ :=
  let initial_green := 26
  let blue_bags := 6
  let marbles_per_bag := 10
  let gifted_green := 6
  let gifted_blue := 8
  
  let total_blue := blue_bags * marbles_per_bag
  let total_before_gift := initial_green + total_blue
  let total_gifted := gifted_green + gifted_blue
  
  total_before_gift - total_gifted

/-- Theorem stating that Janelle has 72 marbles after the transactions -/
theorem janelle_has_72_marbles : janelles_marbles = 72 := by
  sorry

end NUMINAMATH_CALUDE_janelle_has_72_marbles_l1109_110923


namespace NUMINAMATH_CALUDE_crayon_eraser_difference_l1109_110922

def prove_crayon_eraser_difference 
  (initial_erasers : ℕ) 
  (initial_crayons : ℕ) 
  (remaining_crayons : ℕ) : Prop :=
  initial_erasers = 457 ∧ 
  initial_crayons = 617 ∧ 
  remaining_crayons = 523 → 
  remaining_crayons - initial_erasers = 66

theorem crayon_eraser_difference : 
  prove_crayon_eraser_difference 457 617 523 :=
by sorry

end NUMINAMATH_CALUDE_crayon_eraser_difference_l1109_110922


namespace NUMINAMATH_CALUDE_prob_A_wins_first_is_half_l1109_110956

/-- A game series between Team A and Team B -/
structure GameSeries where
  /-- The probability of Team A winning any single game -/
  prob_A_win : ℝ
  /-- The total number of games played in the series -/
  total_games : ℕ

/-- The event that Team A wins the series -/
def team_A_wins (s : GameSeries) : Prop :=
  s.total_games = 5 ∧ s.prob_A_win = 1/2

/-- The probability that Team A wins the first game given that Team A wins the series -/
def prob_A_wins_first_given_series_win (s : GameSeries) : ℝ :=
  sorry

/-- Theorem stating that the probability of Team A winning the first game
    is 1/2 given that Team A wins the series in exactly 5 games -/
theorem prob_A_wins_first_is_half
  (s : GameSeries)
  (h : team_A_wins s) :
  prob_A_wins_first_given_series_win s = 1/2 :=
sorry

end NUMINAMATH_CALUDE_prob_A_wins_first_is_half_l1109_110956


namespace NUMINAMATH_CALUDE_lcm_of_given_numbers_l1109_110993

/-- The least common multiple of 360, 450, 560, 900, and 1176 is 176400. -/
theorem lcm_of_given_numbers : Nat.lcm 360 (Nat.lcm 450 (Nat.lcm 560 (Nat.lcm 900 1176))) = 176400 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_given_numbers_l1109_110993


namespace NUMINAMATH_CALUDE_smallest_s_value_l1109_110966

theorem smallest_s_value : ∃ s : ℚ, s = 4/7 ∧ 
  (∀ t : ℚ, (15*t^2 - 40*t + 18) / (4*t - 3) + 7*t = 9*t - 2 → s ≤ t) ∧ 
  (15*s^2 - 40*s + 18) / (4*s - 3) + 7*s = 9*s - 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_s_value_l1109_110966


namespace NUMINAMATH_CALUDE_fraction_equality_l1109_110995

theorem fraction_equality : (8 : ℝ) / (5 * 42) = 0.8 / (2.1 * 10) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1109_110995


namespace NUMINAMATH_CALUDE_total_shuttlecocks_is_456_l1109_110918

/-- The number of students in Yunsu's class -/
def num_students : ℕ := 24

/-- The number of shuttlecocks each student received -/
def shuttlecocks_per_student : ℕ := 19

/-- The total number of shuttlecocks distributed -/
def total_shuttlecocks : ℕ := num_students * shuttlecocks_per_student

/-- Theorem stating that the total number of shuttlecocks is 456 -/
theorem total_shuttlecocks_is_456 : total_shuttlecocks = 456 := by
  sorry

end NUMINAMATH_CALUDE_total_shuttlecocks_is_456_l1109_110918


namespace NUMINAMATH_CALUDE_vector_parallel_if_negative_l1109_110908

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

def parallel (a b : n) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_parallel_if_negative (a b : n) : a = -b → parallel a b := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_if_negative_l1109_110908


namespace NUMINAMATH_CALUDE_convention_delegates_l1109_110947

theorem convention_delegates (total : ℕ) 
  (h1 : 16 ≤ total) 
  (h2 : (total - 16) % 2 = 0) 
  (h3 : 10 ≤ total - 16 - (total - 16) / 2) : 
  total = 36 := by
  sorry

end NUMINAMATH_CALUDE_convention_delegates_l1109_110947


namespace NUMINAMATH_CALUDE_arithmetic_sequence_range_l1109_110977

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_range (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 ≤ 7)
  (h_a6 : a 6 ≥ 9) :
  (a 10 > 11 ∧ ∀ M : ℝ, ∃ N : ℝ, a 10 > N ∧ N > M) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_range_l1109_110977


namespace NUMINAMATH_CALUDE_tenth_largest_four_digit_odd_l1109_110976

/-- The set of odd digits -/
def OddDigits : Set Nat := {1, 3, 5, 7, 9}

/-- A four-digit number composed of only odd digits -/
def FourDigitOddNumber (a b c d : Nat) : Prop :=
  a ∈ OddDigits ∧ b ∈ OddDigits ∧ c ∈ OddDigits ∧ d ∈ OddDigits ∧
  1000 ≤ a * 1000 + b * 100 + c * 10 + d ∧ a * 1000 + b * 100 + c * 10 + d ≤ 9999

/-- The theorem stating that 9971 is the tenth largest four-digit number composed of only odd digits -/
theorem tenth_largest_four_digit_odd : 
  (∃ (n : Nat), n = 9 ∧ 
    (∃ (a b c d : Nat), FourDigitOddNumber a b c d ∧ 
      a * 1000 + b * 100 + c * 10 + d > 9971)) := by sorry

end NUMINAMATH_CALUDE_tenth_largest_four_digit_odd_l1109_110976


namespace NUMINAMATH_CALUDE_min_extractions_to_reverse_l1109_110962

/-- Represents a stack of cards -/
def CardStack := List Nat

/-- Represents an extraction operation on a card stack -/
def Extraction := CardStack → CardStack

/-- Checks if a card stack is in reverse order -/
def is_reversed (stack : CardStack) : Prop :=
  stack = List.reverse (List.range stack.length)

/-- Theorem: Minimum number of extractions to reverse a card stack -/
theorem min_extractions_to_reverse (n : Nat) :
  ∃ (k : Nat) (extractions : List Extraction),
    k = n / 2 + 1 ∧
    extractions.length = k ∧
    is_reversed (extractions.foldl (λ acc f => f acc) (List.range n)) :=
  sorry

end NUMINAMATH_CALUDE_min_extractions_to_reverse_l1109_110962


namespace NUMINAMATH_CALUDE_volume_formula_l1109_110999

/-- A right rectangular prism with edge lengths 2, 3, and 5 -/
structure Prism where
  length : ℝ := 2
  width : ℝ := 3
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : Prism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume (B : Prism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume polynomial -/
structure VolumeCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem volume_formula (B : Prism) (coeffs : VolumeCoeffs) :
  (∀ r : ℝ, volume B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d) →
  coeffs.a > 0 ∧ coeffs.b > 0 ∧ coeffs.c > 0 ∧ coeffs.d > 0 →
  coeffs.b * coeffs.c / (coeffs.a * coeffs.d) = 20.67 := by
  sorry

end NUMINAMATH_CALUDE_volume_formula_l1109_110999


namespace NUMINAMATH_CALUDE_m_range_l1109_110986

-- Define the condition on x
def X := { x : ℝ | x ≤ -1 }

-- Define the inequality condition
def inequality (m : ℝ) : Prop :=
  ∀ x ∈ X, (m^2 - m) * 4^x - 2^x < 0

-- Theorem statement
theorem m_range :
  ∀ m : ℝ, inequality m ↔ -1 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_m_range_l1109_110986


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_sum_l1109_110906

theorem consecutive_odd_squares_sum (k : ℤ) (n : ℕ) :
  (2 * k - 1)^2 + (2 * k + 1)^2 = n * (n + 1) / 2 ↔ k = 1 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_sum_l1109_110906


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_l1109_110967

/-- The function f(x) = 3x^2 - 18x + 2023 has a minimum value of 1996. -/
theorem min_value_of_quadratic :
  ∃ (m : ℝ), m = 1996 ∧ ∀ x : ℝ, 3 * x^2 - 18 * x + 2023 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_l1109_110967


namespace NUMINAMATH_CALUDE_distribution_ways_4_5_l1109_110914

/-- The number of ways to distribute men and women into groups -/
def distribution_ways (num_men num_women : ℕ) : ℕ :=
  let scenario1 := num_men.choose 1 * num_women.choose 2 * (3 * 2)
  let scenario2 := num_men.choose 2 * num_women.choose 1 * (num_women.choose 2 * 2)
  scenario1 + scenario2

/-- Theorem stating the number of ways to distribute 4 men and 5 women -/
theorem distribution_ways_4_5 :
  distribution_ways 4 5 = 600 := by
  sorry

#eval distribution_ways 4 5

end NUMINAMATH_CALUDE_distribution_ways_4_5_l1109_110914


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l1109_110948

/-- A rectangular prism with side lengths a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A quadrilateral formed by the intersection of a plane with a rectangular prism -/
structure IntersectionQuadrilateral where
  prism : RectangularPrism
  -- Assume A and C are diagonally opposite vertices
  -- Assume B and D are midpoints of opposite edges not containing A or C

/-- The area of the quadrilateral formed by the intersection -/
noncomputable def intersection_area (quad : IntersectionQuadrilateral) : ℝ := sorry

/-- Theorem stating the area of the specific intersection quadrilateral -/
theorem intersection_area_theorem (quad : IntersectionQuadrilateral) 
  (h1 : quad.prism.a = 2)
  (h2 : quad.prism.b = 3)
  (h3 : quad.prism.c = 5) :
  intersection_area quad = 7 * Real.sqrt 26 / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l1109_110948


namespace NUMINAMATH_CALUDE_justice_palms_l1109_110905

/-- The number of palms Justice has -/
def num_palms : ℕ := sorry

/-- The total number of plants Justice wants -/
def total_plants : ℕ := 24

/-- The number of ferns Justice has -/
def num_ferns : ℕ := 3

/-- The number of succulent plants Justice has -/
def num_succulents : ℕ := 7

/-- The number of additional plants Justice needs -/
def additional_plants : ℕ := 9

theorem justice_palms : num_palms = 5 := by sorry

end NUMINAMATH_CALUDE_justice_palms_l1109_110905


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1109_110965

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes :
  distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1109_110965


namespace NUMINAMATH_CALUDE_initial_sum_calculation_l1109_110963

/-- The initial sum that earns a specific total simple interest over 4 years with varying interest rates -/
def initial_sum (total_interest : ℚ) (rate1 rate2 rate3 rate4 : ℚ) : ℚ :=
  total_interest / (rate1 + rate2 + rate3 + rate4)

/-- Theorem stating that given the specified conditions, the initial sum is 5000/9 -/
theorem initial_sum_calculation :
  initial_sum 100 (3/100) (5/100) (4/100) (6/100) = 5000/9 := by
  sorry

#eval initial_sum 100 (3/100) (5/100) (4/100) (6/100)

end NUMINAMATH_CALUDE_initial_sum_calculation_l1109_110963


namespace NUMINAMATH_CALUDE_only_5_12_13_is_right_triangle_l1109_110952

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def number_sets : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (4, 5, 6), (5, 12, 13), (5, 6, 7)]

/-- Theorem stating that only (5, 12, 13) forms a right triangle --/
theorem only_5_12_13_is_right_triangle :
  ∃! (a b c : ℕ), (a, b, c) ∈ number_sets ∧ is_right_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_only_5_12_13_is_right_triangle_l1109_110952


namespace NUMINAMATH_CALUDE_cupcakes_remaining_l1109_110920

/-- The number of cupcake packages Maggi had -/
def packages : ℝ := 3.5

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 7

/-- The number of cupcakes Maggi ate -/
def eaten_cupcakes : ℝ := 5.75

/-- The number of cupcakes left after Maggi ate some -/
def cupcakes_left : ℝ := packages * cupcakes_per_package - eaten_cupcakes

theorem cupcakes_remaining : cupcakes_left = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_remaining_l1109_110920


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l1109_110911

theorem max_value_cos_sin (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l1109_110911


namespace NUMINAMATH_CALUDE_new_xanadu_license_plates_l1109_110945

/-- Represents the number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- Represents the number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- Represents the number of possible first digits (1-9) -/
def num_first_digits : ℕ := 9

/-- Calculates the total number of valid license plates in New Xanadu -/
def num_valid_plates : ℕ :=
  num_letters ^ 3 * num_first_digits * num_digits ^ 2

/-- Theorem stating the total number of valid license plates in New Xanadu -/
theorem new_xanadu_license_plates :
  num_valid_plates = 15818400 := by
  sorry

end NUMINAMATH_CALUDE_new_xanadu_license_plates_l1109_110945


namespace NUMINAMATH_CALUDE_passenger_speed_on_train_l1109_110961

/-- The speed of a passenger relative to the railway track when moving on a train -/
def passenger_speed_relative_to_track (train_speed passenger_speed : ℝ) : ℝ × ℝ :=
  (train_speed + passenger_speed, |train_speed - passenger_speed|)

/-- Theorem: The speed of a passenger relative to the railway track
    when the train moves at 60 km/h and the passenger moves at 3 km/h relative to the train -/
theorem passenger_speed_on_train :
  let train_speed := 60
  let passenger_speed := 3
  passenger_speed_relative_to_track train_speed passenger_speed = (63, 57) := by
  sorry

end NUMINAMATH_CALUDE_passenger_speed_on_train_l1109_110961


namespace NUMINAMATH_CALUDE_correct_factorization_l1109_110907

theorem correct_factorization (a : ℝ) :
  a^2 - a + (1/4 : ℝ) = (a - 1/2)^2 := by sorry

end NUMINAMATH_CALUDE_correct_factorization_l1109_110907


namespace NUMINAMATH_CALUDE_rakesh_distance_l1109_110996

/-- Represents the walking problem with four people: Hiro, Rakesh, Sanjay, and Charu -/
structure WalkingProblem where
  hiro_distance : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The conditions of the walking problem -/
def walking_conditions (wp : WalkingProblem) : Prop :=
  wp.total_distance = 85 ∧
  wp.total_time = 20 ∧
  ∃ (rakesh_time sanjay_time charu_time : ℝ),
    rakesh_time = wp.total_time - (wp.total_time - 2) - sanjay_time - charu_time ∧
    charu_time = wp.total_time - (wp.total_time - 2) ∧
    wp.total_distance = wp.hiro_distance + (4 * wp.hiro_distance - 10) + (2 * wp.hiro_distance + 3) +
      ((4 * wp.hiro_distance - 10) + (2 * wp.hiro_distance + 3)) / 2

/-- The theorem stating Rakesh's walking distance -/
theorem rakesh_distance (wp : WalkingProblem) (h : walking_conditions wp) :
    4 * wp.hiro_distance - 10 = 28.2 := by
  sorry


end NUMINAMATH_CALUDE_rakesh_distance_l1109_110996


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l1109_110938

def father_son_ages (son_age : ℕ) (age_difference : ℕ) : Prop :=
  let father_age : ℕ := son_age + age_difference
  let son_age_in_two_years : ℕ := son_age + 2
  let father_age_in_two_years : ℕ := father_age + 2
  (father_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2

theorem father_son_age_ratio :
  father_son_ages 33 35 := by sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l1109_110938


namespace NUMINAMATH_CALUDE_min_m_value_l1109_110903

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := Real.exp x - Real.exp (-x)
def g (m : ℝ) (x : ℝ) := Real.log (m * x^2 - x + 1/4)

-- State the theorem
theorem min_m_value :
  (∀ x1 ∈ Set.Iic (0 : ℝ), ∃ x2 : ℝ, f x1 = g m x2) →
  (∀ m' : ℝ, m' < -1/3 → ¬(∀ x1 ∈ Set.Iic (0 : ℝ), ∃ x2 : ℝ, f x1 = g m' x2)) →
  (∃ x1 ∈ Set.Iic (0 : ℝ), ∃ x2 : ℝ, f x1 = g (-1/3) x2) →
  m = -1/3 :=
sorry

end

end NUMINAMATH_CALUDE_min_m_value_l1109_110903


namespace NUMINAMATH_CALUDE_three_not_in_range_of_g_l1109_110941

/-- The function g(x) defined as x^2 + 2x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c

/-- Theorem stating that 3 is not in the range of g(x) if and only if c > 4 -/
theorem three_not_in_range_of_g (c : ℝ) :
  (∀ x, g c x ≠ 3) ↔ c > 4 := by
  sorry

end NUMINAMATH_CALUDE_three_not_in_range_of_g_l1109_110941


namespace NUMINAMATH_CALUDE_range_of_m_l1109_110915

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, ¬(p x) → ¬(q x m)) ∧  -- "not p" is sufficient for "not q"
  (∃ x, ¬(q x m) ∧ p x) ∧     -- "not p" is not necessary for "not q"
  (m > 0) →                   -- given condition m > 0
  m ≤ 3 :=                    -- prove that m ≤ 3
sorry

end NUMINAMATH_CALUDE_range_of_m_l1109_110915


namespace NUMINAMATH_CALUDE_ginger_water_usage_l1109_110919

/-- Calculates the total cups of water used by Ginger in her garden -/
def total_water_used (hours_worked : ℕ) (cups_per_bottle : ℕ) (bottles_for_plants : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (bottles_for_plants * cups_per_bottle)

theorem ginger_water_usage :
  total_water_used 8 2 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ginger_water_usage_l1109_110919


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1109_110964

theorem quadratic_inequality_range (x : ℝ) : x^2 + 3*x - 10 < 0 ↔ -5 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1109_110964


namespace NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l1109_110951

theorem max_y_coordinate_sin_3theta :
  let r : ℝ → ℝ := fun θ => Real.sin (3 * θ)
  let y : ℝ → ℝ := fun θ => r θ * Real.sin θ
  ∃ (max_y : ℝ), (∀ θ, y θ ≤ max_y) ∧ (∃ θ, y θ = max_y) ∧ max_y = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l1109_110951


namespace NUMINAMATH_CALUDE_identity_is_unique_strictly_increasing_double_application_less_than_successor_l1109_110953

-- Define a strictly increasing function from ℕ to ℕ
def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem identity_is_unique_strictly_increasing_double_application_less_than_successor
  (f : ℕ → ℕ)
  (h_increasing : StrictlyIncreasing f)
  (h_condition : ∀ n, f (f n) < n + 1) :
  ∀ n, f n = n :=
by sorry

end NUMINAMATH_CALUDE_identity_is_unique_strictly_increasing_double_application_less_than_successor_l1109_110953


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_solution_l1109_110926

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (1/8, -3/4)

/-- First line equation: y = -6x -/
def line1 (x y : ℚ) : Prop := y = -6 * x

/-- Second line equation: y + 3 = 18x -/
def line2 (x y : ℚ) : Prop := y + 3 = 18 * x

/-- Theorem stating that the intersection_point satisfies both line equations
    and is the unique solution -/
theorem intersection_point_is_unique_solution :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ (x' y' : ℚ), line1 x' y' → line2 x' y' → (x', y') = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_solution_l1109_110926


namespace NUMINAMATH_CALUDE_arrange_four_men_five_women_l1109_110925

/-- The number of ways to arrange people into groups -/
def arrange_groups (num_men : ℕ) (num_women : ℕ) : ℕ :=
  let three_person_group := Nat.choose num_men 2 * Nat.choose num_women 1
  let first_two_person_group := Nat.choose (num_men - 2) 1 * Nat.choose (num_women - 1) 1
  three_person_group * first_two_person_group * 1

/-- Theorem stating the number of ways to arrange 4 men and 5 women into specific groups -/
theorem arrange_four_men_five_women :
  arrange_groups 4 5 = 240 := by
  sorry


end NUMINAMATH_CALUDE_arrange_four_men_five_women_l1109_110925


namespace NUMINAMATH_CALUDE_student_line_count_l1109_110981

/-- Given a line of students, if a student is 7th from the left and 5th from the right,
    then the total number of students in the line is 11. -/
theorem student_line_count (n : ℕ) 
  (left_position : ℕ) 
  (right_position : ℕ) 
  (h1 : left_position = 7) 
  (h2 : right_position = 5) : 
  n = left_position + right_position - 1 := by
  sorry

end NUMINAMATH_CALUDE_student_line_count_l1109_110981


namespace NUMINAMATH_CALUDE_circles_intersect_l1109_110950

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_O₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define the center and radius of each circle
def center_O₁ : ℝ × ℝ := (0, 0)
def center_O₂ : ℝ × ℝ := (3, 0)
def radius_O₁ : ℝ := 2
def radius_O₂ : ℝ := 2

-- Define the distance between centers
def distance_between_centers : ℝ := 3

-- Theorem stating that the circles intersect
theorem circles_intersect :
  distance_between_centers > abs (radius_O₁ - radius_O₂) ∧
  distance_between_centers < radius_O₁ + radius_O₂ :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l1109_110950


namespace NUMINAMATH_CALUDE_intersection_M_N_l1109_110900

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N : M ∩ N = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1109_110900


namespace NUMINAMATH_CALUDE_slips_with_three_count_l1109_110997

/-- Given a bag of slips with either 3 or 8 written on them, 
    this function calculates the expected value of a randomly drawn slip. -/
def expected_value (total_slips : ℕ) (slips_with_three : ℕ) : ℚ :=
  (3 * slips_with_three + 8 * (total_slips - slips_with_three)) / total_slips

/-- Theorem stating that given the conditions of the problem, 
    the number of slips with 3 written on them is 8. -/
theorem slips_with_three_count : 
  ∃ (x : ℕ), x ≤ 15 ∧ expected_value 15 x = 5.4 ∧ x = 8 := by
  sorry


end NUMINAMATH_CALUDE_slips_with_three_count_l1109_110997


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l1109_110943

/-- The number of ways to arrange 6 out of 10 distinguishable rings on 4 fingers -/
def ring_arrangements : ℕ :=
  (Nat.choose 10 6) * (Nat.factorial 6) * (Nat.choose 9 3)

/-- Theorem stating the number of ring arrangements -/
theorem ring_arrangements_count : ring_arrangements = 12672000 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l1109_110943


namespace NUMINAMATH_CALUDE_monotonicity_and_range_l1109_110913

noncomputable def f (a x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + a^2

theorem monotonicity_and_range :
  (∀ x > 0, ∀ y > 0, (2-Real.sqrt 2)/2 < x → x < y → y < (2+Real.sqrt 2)/2 → f 2 y < f 2 x) ∧
  (∀ x > 0, ∀ y > 0, 0 < x → x < y → y < (2-Real.sqrt 2)/2 → f 2 x < f 2 y) ∧
  (∀ x > 0, ∀ y > 0, (2+Real.sqrt 2)/2 < x → x < y → f 2 x < f 2 y) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, x < y → f a x ≥ f a y) → a ≥ 19/6) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_range_l1109_110913


namespace NUMINAMATH_CALUDE_three_Y_two_equals_one_l1109_110989

-- Define the Y operation
def Y (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem three_Y_two_equals_one : Y 3 2 = 1 := by sorry

end NUMINAMATH_CALUDE_three_Y_two_equals_one_l1109_110989


namespace NUMINAMATH_CALUDE_betty_orange_boxes_l1109_110957

theorem betty_orange_boxes (oranges_per_box : ℕ) (total_oranges : ℕ) (h1 : oranges_per_box = 24) (h2 : total_oranges = 72) :
  total_oranges / oranges_per_box = 3 :=
by sorry

end NUMINAMATH_CALUDE_betty_orange_boxes_l1109_110957


namespace NUMINAMATH_CALUDE_solve_equation_l1109_110969

theorem solve_equation : ∃ x : ℝ, (0.5^3 - 0.1^3 / 0.5^2 + x + 0.1^2 = 0.4) ∧ (x = 0.269) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1109_110969


namespace NUMINAMATH_CALUDE_percent_of_percent_l1109_110978

theorem percent_of_percent (y : ℝ) (hy : y ≠ 0) :
  (18 / 100) * y = (30 / 100) * ((60 / 100) * y) :=
by sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1109_110978


namespace NUMINAMATH_CALUDE_lemonade_pitchers_l1109_110928

theorem lemonade_pitchers (glasses_per_pitcher : ℕ) (total_glasses : ℕ) (h1 : glasses_per_pitcher = 5) (h2 : total_glasses = 30) :
  total_glasses / glasses_per_pitcher = 6 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_pitchers_l1109_110928


namespace NUMINAMATH_CALUDE_max_k_value_l1109_110932

open Real

noncomputable def f (x : ℝ) : ℝ := exp x - x - 2

theorem max_k_value (k : ℤ) :
  (∀ x : ℝ, x > 0 → (k - x) / (x + 1) * (exp x - 1) < 1) →
  k ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1109_110932


namespace NUMINAMATH_CALUDE_N_subset_M_l1109_110939

def M : Set Nat := {1, 2, 3}
def N : Set Nat := {1}

theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l1109_110939


namespace NUMINAMATH_CALUDE_probability_two_red_cards_value_l1109_110936

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ := 54)
  (red_cards : ℕ := 27)
  (jokers : ℕ := 2)

/-- The probability of drawing two red cards from a standard deck -/
def probability_two_red_cards (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards * (d.red_cards - 1) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing two red cards from a standard deck -/
theorem probability_two_red_cards_value (d : Deck) :
  probability_two_red_cards d = 13 / 53 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_cards_value_l1109_110936


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l1109_110987

theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → 
  c > 0 → 
  c^2 = 2 * a^2 →  -- isosceles right triangle condition
  a^2 + a^2 + c^2 = 1452 →  -- sum of squares condition
  c = Real.sqrt 726 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l1109_110987


namespace NUMINAMATH_CALUDE_positive_integers_sum_product_l1109_110934

theorem positive_integers_sum_product (P Q : ℕ+) (h : P + Q + P * Q = 90) : P + Q = 18 := by
  sorry

end NUMINAMATH_CALUDE_positive_integers_sum_product_l1109_110934


namespace NUMINAMATH_CALUDE_dandelion_seed_survival_l1109_110970

theorem dandelion_seed_survival (seeds_per_dandelion : ℕ) 
  (insect_eaten_fraction : ℚ) (sprout_eaten_fraction : ℚ) 
  (surviving_dandelions : ℕ) : ℚ :=
  let total_seeds := seeds_per_dandelion * surviving_dandelions
  let water_death_fraction := 1 - (surviving_dandelions : ℚ) / 
    (total_seeds * (1 - insect_eaten_fraction) * (1 - sprout_eaten_fraction))
  have h1 : seeds_per_dandelion = 300 := by sorry
  have h2 : insect_eaten_fraction = 1/6 := by sorry
  have h3 : sprout_eaten_fraction = 1/2 := by sorry
  have h4 : surviving_dandelions = 75 := by sorry
  have h5 : water_death_fraction = 124/125 := by sorry
  water_death_fraction

end NUMINAMATH_CALUDE_dandelion_seed_survival_l1109_110970


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1109_110949

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 115) : 
  A + B + C = 180 → C = 65 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1109_110949


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1109_110968

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x : ℝ, x ≥ 0 → x^2 - x ≥ 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 - x < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1109_110968


namespace NUMINAMATH_CALUDE_unique_solution_iff_b_less_than_two_l1109_110930

/-- The equation has exactly one real solution -/
def has_unique_real_solution (b : ℝ) : Prop :=
  ∃! x : ℝ, x^3 - b*x^2 - 3*b*x + b^2 - 4 = 0

/-- The main theorem -/
theorem unique_solution_iff_b_less_than_two :
  ∀ b : ℝ, has_unique_real_solution b ↔ b < 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_b_less_than_two_l1109_110930


namespace NUMINAMATH_CALUDE_f_equals_g_l1109_110902

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := (x^2 - 1)^(1/3)

-- State the theorem
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l1109_110902


namespace NUMINAMATH_CALUDE_school_event_handshakes_l1109_110917

/-- Represents the number of handshakes in a group of children -/
def handshakes (n : ℕ) : ℕ := 
  (n * (n - 1)) / 2

/-- The problem statement -/
theorem school_event_handshakes : 
  handshakes 8 = 36 := by sorry

end NUMINAMATH_CALUDE_school_event_handshakes_l1109_110917


namespace NUMINAMATH_CALUDE_smallest_divisor_property_solution_set_l1109_110980

def smallest_divisor (n : ℕ) : ℕ :=
  (Nat.factors n).head!

theorem smallest_divisor_property (n : ℕ) : 
  n > 1 → smallest_divisor n > 1 ∧ n % smallest_divisor n = 0 := by sorry

theorem solution_set : 
  {n : ℕ | n + smallest_divisor n = 30} = {25, 27, 28} := by sorry

end NUMINAMATH_CALUDE_smallest_divisor_property_solution_set_l1109_110980


namespace NUMINAMATH_CALUDE_angle_halving_l1109_110955

-- Define the third quadrant
def third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

-- Define the second quadrant
def second_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 + 90 < α ∧ α < n * 360 + 180

-- Define the fourth quadrant
def fourth_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 + 270 < α ∧ α < n * 360 + 360

-- Theorem statement
theorem angle_halving (α : Real) :
  third_quadrant α → second_quadrant (α/2) ∨ fourth_quadrant (α/2) :=
by sorry

end NUMINAMATH_CALUDE_angle_halving_l1109_110955
