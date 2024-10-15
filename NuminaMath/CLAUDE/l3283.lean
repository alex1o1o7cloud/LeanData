import Mathlib

namespace NUMINAMATH_CALUDE_sector_max_area_l3283_328340

/-- Given a sector of a circle with perimeter 30 cm, prove that the maximum area is 225/4 cm² and the corresponding central angle is 2 radians. -/
theorem sector_max_area (l R : ℝ) (h1 : l + 2*R = 30) (h2 : l > 0) (h3 : R > 0) :
  ∃ (S : ℝ), S ≤ 225/4 ∧
  (S = 225/4 → l = 15 ∧ R = 15/2 ∧ l / R = 2) :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l3283_328340


namespace NUMINAMATH_CALUDE_winnings_proof_l3283_328349

theorem winnings_proof (total : ℝ) 
  (h1 : total > 0)
  (h2 : total / 4 + total / 7 + 17 = total) : 
  total = 28 := by
sorry

end NUMINAMATH_CALUDE_winnings_proof_l3283_328349


namespace NUMINAMATH_CALUDE_power_function_through_point_l3283_328364

/-- Given a power function f(x) = x^n that passes through (2, √2), prove f(9) = 3 -/
theorem power_function_through_point (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 2 = Real.sqrt 2 →
  f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3283_328364


namespace NUMINAMATH_CALUDE_green_peaches_count_l3283_328320

theorem green_peaches_count (red : ℕ) (yellow : ℕ) (total : ℕ) (green : ℕ) : 
  red = 7 → yellow = 15 → total = 30 → green = total - (red + yellow) → green = 8 := by
sorry

end NUMINAMATH_CALUDE_green_peaches_count_l3283_328320


namespace NUMINAMATH_CALUDE_problem_statement_l3283_328374

theorem problem_statement (a b : ℚ) 
  (h1 : 3 * a + 4 * b = 0) 
  (h2 : a = 2 * b - 3) : 
  5 * b = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3283_328374


namespace NUMINAMATH_CALUDE_max_value_of_f_l3283_328305

/-- The quadratic function f(x) = -5x^2 + 25x - 7 -/
def f (x : ℝ) : ℝ := -5 * x^2 + 25 * x - 7

/-- The maximum value of f(x) is 53/4 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 53/4 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3283_328305


namespace NUMINAMATH_CALUDE_committee_probability_l3283_328377

def science_club_size : ℕ := 24
def num_boys : ℕ := 12
def num_girls : ℕ := 12
def committee_size : ℕ := 5

def probability_at_least_two_of_each : ℚ :=
  4704 / 7084

theorem committee_probability :
  let total_committees := Nat.choose science_club_size committee_size
  let valid_committees := total_committees - (
    Nat.choose num_boys 0 * Nat.choose num_girls 5 +
    Nat.choose num_boys 1 * Nat.choose num_girls 4 +
    Nat.choose num_boys 4 * Nat.choose num_girls 1 +
    Nat.choose num_boys 5 * Nat.choose num_girls 0
  )
  (valid_committees : ℚ) / total_committees = probability_at_least_two_of_each :=
by sorry

end NUMINAMATH_CALUDE_committee_probability_l3283_328377


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l3283_328336

def total_shoes : ℕ := 10
def red_shoes : ℕ := 4
def green_shoes : ℕ := 6
def drawn_shoes : ℕ := 2

theorem probability_two_red_shoes :
  (Nat.choose red_shoes drawn_shoes : ℚ) / (Nat.choose total_shoes drawn_shoes) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l3283_328336


namespace NUMINAMATH_CALUDE_teachers_count_l3283_328390

/-- Given a school with girls, boys, and teachers, calculates the number of teachers. -/
def calculate_teachers (girls boys total : ℕ) : ℕ :=
  total - (girls + boys)

/-- Proves that there are 772 teachers in a school with 315 girls, 309 boys, and 1396 people in total. -/
theorem teachers_count : calculate_teachers 315 309 1396 = 772 := by
  sorry

end NUMINAMATH_CALUDE_teachers_count_l3283_328390


namespace NUMINAMATH_CALUDE_max_digit_sum_l3283_328397

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a * 100000 + a * 10000 + b * 1000 + b * 100 + c * 10 + c) / 1000000

theorem max_digit_sum (a b c y : ℕ) :
  is_digit a → is_digit b → is_digit c →
  decimal_to_fraction a b c = 1 / y →
  y > 0 → y ≤ 16 →
  a + b + c ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l3283_328397


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3283_328329

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3283_328329


namespace NUMINAMATH_CALUDE_loss_percent_calculation_l3283_328315

def cost_price : ℝ := 600
def selling_price : ℝ := 550

theorem loss_percent_calculation :
  let loss := cost_price - selling_price
  let loss_percent := (loss / cost_price) * 100
  ∃ ε > 0, abs (loss_percent - 8.33) < ε :=
by sorry

end NUMINAMATH_CALUDE_loss_percent_calculation_l3283_328315


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3283_328325

theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 16) :
  let square_side := square_perimeter / 4
  let rectangle_side := 2 * square_side
  rectangle_side * 4 = 32 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3283_328325


namespace NUMINAMATH_CALUDE_magic_trick_strategy_exists_l3283_328306

/-- Represents a card in the set of 29 cards -/
def Card := Fin 29

/-- Represents a pair of cards -/
def CardPair := (Card × Card)

/-- The strategy function for the assistant -/
def AssistantStrategy := (CardPair → CardPair)

/-- The deduction function for the magician -/
def MagicianDeduction := (CardPair → CardPair)

/-- Theorem stating the existence of a successful strategy -/
theorem magic_trick_strategy_exists :
  ∃ (strategy : AssistantStrategy) (deduction : MagicianDeduction),
    ∀ (audience_choice : CardPair),
      deduction (strategy audience_choice) = audience_choice :=
by sorry

end NUMINAMATH_CALUDE_magic_trick_strategy_exists_l3283_328306


namespace NUMINAMATH_CALUDE_min_rectangles_correct_l3283_328346

/-- The minimum number of rectangles needed to cover a board -/
def min_rectangles (n : ℕ) : ℕ := 2 * n

/-- A rectangle with integer side lengths and area equal to a power of 2 -/
structure PowerRect where
  width : ℕ
  height : ℕ
  is_power_of_two : ∃ k : ℕ, width * height = 2^k

/-- A covering of the board with rectangles -/
structure BoardCovering (n : ℕ) where
  rectangles : List PowerRect
  covers_board : (List.sum (rectangles.map (λ r => r.width * r.height))) = (2^n - 1) * (2^n + 1)

theorem min_rectangles_correct (n : ℕ) :
  ∀ (cover : BoardCovering n), cover.rectangles.length ≥ min_rectangles n ∧
  ∃ (optimal_cover : BoardCovering n), optimal_cover.rectangles.length = min_rectangles n :=
sorry

end NUMINAMATH_CALUDE_min_rectangles_correct_l3283_328346


namespace NUMINAMATH_CALUDE_circular_permutations_count_l3283_328373

/-- The number of elements of type 'a' -/
def num_a : ℕ := 2

/-- The number of elements of type 'b' -/
def num_b : ℕ := 2

/-- The number of elements of type 'c' -/
def num_c : ℕ := 4

/-- The total number of elements -/
def total_elements : ℕ := num_a + num_b + num_c

/-- First-class circular permutations -/
def first_class_permutations : ℕ := 52

/-- Second-class circular permutations -/
def second_class_permutations : ℕ := 33

theorem circular_permutations_count :
  (first_class_permutations = 52) ∧ (second_class_permutations = 33) := by
  sorry

end NUMINAMATH_CALUDE_circular_permutations_count_l3283_328373


namespace NUMINAMATH_CALUDE_smallest_with_12_divisors_l3283_328375

/-- The number of positive integer divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- n has exactly 12 positive integer divisors -/
def has12Divisors (n : ℕ) : Prop :=
  numDivisors n = 12

/-- 72 is the smallest positive integer with exactly 12 positive integer divisors -/
theorem smallest_with_12_divisors :
  has12Divisors 72 ∧ ∀ m : ℕ, 0 < m → m < 72 → ¬has12Divisors m := by sorry

end NUMINAMATH_CALUDE_smallest_with_12_divisors_l3283_328375


namespace NUMINAMATH_CALUDE_chicken_nuggets_order_l3283_328363

/-- The number of chicken nuggets ordered by Alyssa, Keely, and Kendall -/
theorem chicken_nuggets_order (alyssa keely kendall : ℕ) 
  (h1 : alyssa = 20)
  (h2 : keely = 2 * alyssa)
  (h3 : kendall = 2 * alyssa) :
  alyssa + keely + kendall = 100 := by
  sorry

end NUMINAMATH_CALUDE_chicken_nuggets_order_l3283_328363


namespace NUMINAMATH_CALUDE_new_person_weight_l3283_328328

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 80 kg -/
theorem new_person_weight :
  weight_of_new_person 6 2.5 65 = 80 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3283_328328


namespace NUMINAMATH_CALUDE_max_value_nonnegative_inequality_condition_l3283_328301

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + a

theorem max_value_nonnegative (a : ℝ) :
  ∀ x₀ : ℝ, (∀ x : ℝ, f a x ≤ f a x₀) → f a x₀ ≥ 0 := by sorry

theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x + Real.exp (x - 1) ≥ 1) ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_max_value_nonnegative_inequality_condition_l3283_328301


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_seven_sixths_l3283_328358

theorem negative_sixty_four_to_seven_sixths (x : ℝ) : x = (-64)^(7/6) → x = -16384 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_seven_sixths_l3283_328358


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3283_328399

theorem modulus_of_complex_fraction (z : ℂ) :
  z = (5 : ℂ) / (1 - 2 * Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3283_328399


namespace NUMINAMATH_CALUDE_roots_transformation_l3283_328311

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 9 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 9 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 9 = 0) →
  ((3*r₁)^3 - 12*(3*r₁)^2 + 243 = 0) ∧
  ((3*r₂)^3 - 12*(3*r₂)^2 + 243 = 0) ∧
  ((3*r₃)^3 - 12*(3*r₃)^2 + 243 = 0) := by
sorry

end NUMINAMATH_CALUDE_roots_transformation_l3283_328311


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3283_328316

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * I
  let z₂ : ℂ := 4 - 7 * I
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3283_328316


namespace NUMINAMATH_CALUDE_snack_distribution_l3283_328380

theorem snack_distribution (candies jellies students : ℕ) 
  (h1 : candies = 72)
  (h2 : jellies = 56)
  (h3 : students = 4) :
  (candies + jellies) / students = 32 :=
by sorry

end NUMINAMATH_CALUDE_snack_distribution_l3283_328380


namespace NUMINAMATH_CALUDE_angle_relation_l3283_328310

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define an angle between three points
def Angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- State the angle bisector theorem
axiom angle_bisector_theorem (c : Circle) (A X Y C S : ℝ × ℝ) 
  (hA : PointOnCircle c A) (hX : PointOnCircle c X) (hY : PointOnCircle c Y) 
  (hC : PointOnCircle c C) (hS : PointOnCircle c S) :
  Angle A X C - Angle A Y C = Angle A S C

-- State the theorem to be proved
theorem angle_relation (c : Circle) (B X Y D S : ℝ × ℝ) 
  (hB : PointOnCircle c B) (hX : PointOnCircle c X) (hY : PointOnCircle c Y) 
  (hD : PointOnCircle c D) (hS : PointOnCircle c S) :
  Angle B X D - Angle B Y D = Angle B S D := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l3283_328310


namespace NUMINAMATH_CALUDE_no_arithmetic_progression_with_product_l3283_328312

theorem no_arithmetic_progression_with_product : ¬∃ (a b : ℝ), 
  (b - a = a - 5) ∧ (a * b - b = b - a) := by
  sorry

end NUMINAMATH_CALUDE_no_arithmetic_progression_with_product_l3283_328312


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3283_328323

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (h : z * Complex.I = -1 - Complex.I) :
  is_in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3283_328323


namespace NUMINAMATH_CALUDE_percentage_difference_l3283_328314

theorem percentage_difference : (60 / 100 * 50) - (40 / 100 * 30) = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3283_328314


namespace NUMINAMATH_CALUDE_simplified_fraction_ratio_l3283_328303

theorem simplified_fraction_ratio (m : ℝ) : 
  let expression := (6 * m + 12) / 3
  ∃ (c d : ℤ), expression = c * m + d ∧ (c : ℚ) / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_ratio_l3283_328303


namespace NUMINAMATH_CALUDE_statement_C_is_false_l3283_328318

-- Define the concept of lines in space
variable (Line : Type)

-- Define the perpendicular relationship between lines
variable (perpendicular : Line → Line → Prop)

-- Define the parallel relationship between lines
variable (parallel : Line → Line → Prop)

-- State the theorem to be proven false
theorem statement_C_is_false :
  ¬(∀ (a b c : Line), perpendicular a c → perpendicular b c → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_statement_C_is_false_l3283_328318


namespace NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l3283_328391

noncomputable section

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := y^2 / 3 + x^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x - y = 4

-- Define the distance function between a point (x, y) and the line C₂
def distance_to_C₂ (x y : ℝ) : ℝ := |x - y - 4| / Real.sqrt 2

-- State the theorem
theorem min_distance_ellipse_to_line :
  ∃ (α : ℝ), 
    let x := Real.sin α
    let y := Real.sqrt 3 * Real.cos α
    C₁ x y ∧ 
    (∀ β : ℝ, distance_to_C₂ (Real.sin β) (Real.sqrt 3 * Real.cos β) ≥ Real.sqrt 2) ∧
    distance_to_C₂ x y = Real.sqrt 2 ∧
    x = 1/2 ∧ y = -3/2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l3283_328391


namespace NUMINAMATH_CALUDE_students_in_line_l3283_328348

theorem students_in_line (front : ℕ) (behind : ℕ) (taehyung : ℕ) 
  (h1 : front = 9) 
  (h2 : behind = 16) 
  (h3 : taehyung = 1) : 
  front + behind + taehyung = 26 := by
  sorry

end NUMINAMATH_CALUDE_students_in_line_l3283_328348


namespace NUMINAMATH_CALUDE_total_items_in_jar_l3283_328382

/-- The total number of items in a jar with candy and secret eggs -/
theorem total_items_in_jar (candy : ℝ) (secret_eggs : ℝ) (h1 : candy = 3409.0) (h2 : secret_eggs = 145.0) :
  candy + secret_eggs = 3554.0 := by
  sorry

end NUMINAMATH_CALUDE_total_items_in_jar_l3283_328382


namespace NUMINAMATH_CALUDE_evaluate_expression_l3283_328383

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  Real.sqrt ((x - 1)^2) + Real.sqrt (x^2 + 4*x + 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3283_328383


namespace NUMINAMATH_CALUDE_largest_rank_3_less_than_quarter_proof_l3283_328330

def rank (q : ℚ) : ℕ :=
  sorry

def largest_rank_3_less_than_quarter : ℚ :=
  sorry

theorem largest_rank_3_less_than_quarter_proof :
  rank largest_rank_3_less_than_quarter = 3 ∧
  largest_rank_3_less_than_quarter < 1/4 ∧
  largest_rank_3_less_than_quarter = 1/5 + 1/21 + 1/421 ∧
  ∀ q : ℚ, rank q = 3 → q < 1/4 → q ≤ largest_rank_3_less_than_quarter :=
by sorry

end NUMINAMATH_CALUDE_largest_rank_3_less_than_quarter_proof_l3283_328330


namespace NUMINAMATH_CALUDE_square_area_l3283_328341

theorem square_area (x : ℝ) : 
  (5 * x - 10 = 3 * (x + 4)) → 
  (5 * x - 10)^2 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l3283_328341


namespace NUMINAMATH_CALUDE_min_value_expression_l3283_328368

theorem min_value_expression (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) ≥ 3 ∧ (x + 4 / (x + 1) = 3 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3283_328368


namespace NUMINAMATH_CALUDE_maple_grove_elementary_difference_l3283_328395

theorem maple_grove_elementary_difference : 
  let classrooms : ℕ := 5
  let students_per_classroom : ℕ := 22
  let hamsters_per_classroom : ℕ := 3
  let total_students : ℕ := classrooms * students_per_classroom
  let total_hamsters : ℕ := classrooms * hamsters_per_classroom
  total_students - total_hamsters = 95 := by
  sorry

end NUMINAMATH_CALUDE_maple_grove_elementary_difference_l3283_328395


namespace NUMINAMATH_CALUDE_total_weight_loss_l3283_328392

theorem total_weight_loss (first_person_loss second_person_loss third_person_loss fourth_person_loss : ℕ) :
  first_person_loss = 27 →
  second_person_loss = first_person_loss - 7 →
  third_person_loss = 28 →
  fourth_person_loss = 28 →
  first_person_loss + second_person_loss + third_person_loss + fourth_person_loss = 103 :=
by sorry

end NUMINAMATH_CALUDE_total_weight_loss_l3283_328392


namespace NUMINAMATH_CALUDE_distance_to_midpoint_zero_l3283_328350

theorem distance_to_midpoint_zero (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁ = 10 ∧ y₁ = 20) 
  (h2 : x₂ = -10 ∧ y₂ = -20) : 
  Real.sqrt (((x₁ + x₂) / 2)^2 + ((y₁ + y₂) / 2)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_midpoint_zero_l3283_328350


namespace NUMINAMATH_CALUDE_sock_ratio_l3283_328337

/-- The ratio of black socks to blue socks in an order satisfying certain conditions -/
theorem sock_ratio :
  ∀ (b : ℕ) (x : ℝ),
  x > 0 →
  (18 * x + b * x) * 1.6 = 3 * b * x + 6 * x →
  (6 : ℝ) / b = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_sock_ratio_l3283_328337


namespace NUMINAMATH_CALUDE_store_discount_problem_l3283_328384

/-- Represents the store discount problem --/
theorem store_discount_problem (shirt_price : ℝ) (shirt_count : ℕ)
                                (pants_price : ℝ) (pants_count : ℕ)
                                (suit_price : ℝ)
                                (sweater_price : ℝ) (sweater_count : ℕ)
                                (coupon_discount : ℝ)
                                (final_price : ℝ) :
  shirt_price = 15 →
  shirt_count = 4 →
  pants_price = 40 →
  pants_count = 2 →
  suit_price = 150 →
  sweater_price = 30 →
  sweater_count = 2 →
  coupon_discount = 0.1 →
  final_price = 252 →
  ∃ (store_discount : ℝ),
    store_discount = 0.2 ∧
    final_price = (shirt_price * shirt_count +
                   pants_price * pants_count +
                   suit_price +
                   sweater_price * sweater_count) *
                  (1 - store_discount) *
                  (1 - coupon_discount) := by
  sorry

end NUMINAMATH_CALUDE_store_discount_problem_l3283_328384


namespace NUMINAMATH_CALUDE_ticket_price_is_28_l3283_328326

/-- The price of a single ticket given the total money, number of tickets, and remaining money -/
def ticket_price (total_money : ℕ) (num_tickets : ℕ) (remaining_money : ℕ) : ℕ :=
  (total_money - remaining_money) / num_tickets

/-- Theorem stating that the ticket price is $28 given the problem conditions -/
theorem ticket_price_is_28 :
  ticket_price 251 6 83 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_is_28_l3283_328326


namespace NUMINAMATH_CALUDE_sum_of_21st_set_l3283_328355

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The first element of the nth set -/
def first_element (n : ℕ) : ℕ := 1 + sum_first_n (n - 1)

/-- The last element of the nth set -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- The theorem to prove -/
theorem sum_of_21st_set : S 21 = 4641 := by sorry

end NUMINAMATH_CALUDE_sum_of_21st_set_l3283_328355


namespace NUMINAMATH_CALUDE_equation_solution_difference_l3283_328376

theorem equation_solution_difference : ∃ (s₁ s₂ : ℝ),
  s₁ ≠ s₂ ∧
  s₁ ≠ -6 ∧
  s₂ ≠ -6 ∧
  (s₁^2 - 5*s₁ - 24) / (s₁ + 6) = 3*s₁ + 10 ∧
  (s₂^2 - 5*s₂ - 24) / (s₂ + 6) = 3*s₂ + 10 ∧
  |s₁ - s₂| = 6.5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l3283_328376


namespace NUMINAMATH_CALUDE_greg_harvest_l3283_328317

theorem greg_harvest (sharon_harvest : Real) (greg_additional : Real) : 
  sharon_harvest = 0.1 →
  greg_additional = 0.3 →
  sharon_harvest + greg_additional = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_greg_harvest_l3283_328317


namespace NUMINAMATH_CALUDE_remainder_problem_l3283_328388

theorem remainder_problem (h1 : Nat.Prime 73) (h2 : ¬(73 ∣ 57)) :
  (57^35 + 47) % 73 = 55 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3283_328388


namespace NUMINAMATH_CALUDE_inequality_one_l3283_328343

theorem inequality_one (x y : ℝ) : (x + 1) * (x - 2*y + 1) + y^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_one_l3283_328343


namespace NUMINAMATH_CALUDE_sqrt_225_equals_15_l3283_328338

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_225_equals_15_l3283_328338


namespace NUMINAMATH_CALUDE_flower_bed_path_area_l3283_328302

/-- The area of a circular ring around a flower bed -/
theorem flower_bed_path_area (circumference : Real) (path_width : Real) : 
  circumference = 314 → path_width = 2 →
  let inner_radius := circumference / (2 * Real.pi)
  let outer_radius := inner_radius + path_width
  abs (Real.pi * (outer_radius^2 - inner_radius^2) - 640.56) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_flower_bed_path_area_l3283_328302


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l3283_328381

/-- Given a geometric sequence where the 5th term is 8 and the 8th term is 64,
    prove that the 11th term is 512. -/
theorem geometric_sequence_11th_term (a : ℝ) (r : ℝ) :
  a * r^4 = 8 → a * r^7 = 64 → a * r^10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l3283_328381


namespace NUMINAMATH_CALUDE_geometric_series_relation_l3283_328369

/-- Given real numbers c and d satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 3/5. -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (c/d) / (1 - 1/d) = 3) :
    (c/(c+2*d)) / (1 - 1/(c+2*d)) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l3283_328369


namespace NUMINAMATH_CALUDE_unpainted_area_specific_case_l3283_328308

/-- Represents the configuration of two crossed boards -/
structure CrossedBoards where
  width1 : ℝ
  width2 : ℝ
  angle : ℝ

/-- Calculates the area of the unpainted region on the first board -/
def unpainted_area (boards : CrossedBoards) : ℝ :=
  boards.width1 * boards.width2

/-- Theorem stating the area of the unpainted region for specific board widths and angle -/
theorem unpainted_area_specific_case :
  let boards : CrossedBoards := ⟨5, 7, 45⟩
  unpainted_area boards = 35 := by sorry

end NUMINAMATH_CALUDE_unpainted_area_specific_case_l3283_328308


namespace NUMINAMATH_CALUDE_payment_difference_equation_l3283_328319

/-- Represents the payment structure for two artists painting murals. -/
structure MuralPayment where
  diego : ℝ  -- Diego's payment
  celina : ℝ  -- Celina's payment
  total : ℝ   -- Total payment
  h1 : celina > 4 * diego  -- Celina's payment is more than 4 times Diego's
  h2 : celina + diego = total  -- Sum of payments equals total

/-- The difference between Celina's payment and 4 times Diego's payment. -/
def payment_difference (p : MuralPayment) : ℝ := p.celina - 4 * p.diego

/-- Theorem stating the relationship between the payment difference and Diego's payment. -/
theorem payment_difference_equation (p : MuralPayment) (h3 : p.total = 50000) :
  payment_difference p = 50000 - 5 * p.diego := by
  sorry


end NUMINAMATH_CALUDE_payment_difference_equation_l3283_328319


namespace NUMINAMATH_CALUDE_marias_school_students_l3283_328386

theorem marias_school_students (m d : ℕ) : 
  m = 4 * d → 
  m - d = 1800 → 
  m = 2400 := by
sorry

end NUMINAMATH_CALUDE_marias_school_students_l3283_328386


namespace NUMINAMATH_CALUDE_police_officer_ratio_l3283_328324

/-- Proves that the ratio of female officers to male officers on duty is 1:1 -/
theorem police_officer_ratio (total_on_duty : ℕ) (female_officers : ℕ) (female_on_duty_percent : ℚ) :
  total_on_duty = 204 →
  female_officers = 600 →
  female_on_duty_percent = 17 / 100 →
  ∃ (female_on_duty male_on_duty : ℕ),
    female_on_duty = female_on_duty_percent * female_officers ∧
    male_on_duty = total_on_duty - female_on_duty ∧
    female_on_duty = male_on_duty :=
by sorry

end NUMINAMATH_CALUDE_police_officer_ratio_l3283_328324


namespace NUMINAMATH_CALUDE_square_plus_minus_one_divisible_by_five_l3283_328347

theorem square_plus_minus_one_divisible_by_five (n : ℤ) (h : ¬ 5 ∣ n) : 
  5 ∣ (n^2 + 1) ∨ 5 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_square_plus_minus_one_divisible_by_five_l3283_328347


namespace NUMINAMATH_CALUDE_complex_expression_equality_combinatorial_equality_l3283_328322

-- Part I
theorem complex_expression_equality : 
  (((Complex.abs (1 - Complex.I)) / Real.sqrt 2) ^ 16 + 
   ((1 + 2 * Complex.I) ^ 2) / (1 - Complex.I)) = 
  (-5 / 2 : ℂ) + (1 / 2 : ℂ) * Complex.I := by sorry

-- Part II
theorem combinatorial_equality (m : ℕ) : 
  (1 / Nat.choose 5 m : ℚ) - (1 / Nat.choose 6 m : ℚ) = 
  (7 : ℚ) / (10 * Nat.choose 7 m) → 
  Nat.choose 8 m = 28 := by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_combinatorial_equality_l3283_328322


namespace NUMINAMATH_CALUDE_repeating_decimal_eq_l3283_328365

/-- The repeating decimal 0.565656... expressed as a rational number -/
def repeating_decimal : ℚ := 56 / 99

/-- The theorem stating that the repeating decimal 0.565656... equals 56/99 -/
theorem repeating_decimal_eq : repeating_decimal = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_eq_l3283_328365


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3283_328333

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i^2 = -1 →
  let z := (3 - 2 * i^2) / (1 + i)
  Complex.im z = -5/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3283_328333


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3283_328379

theorem simple_interest_problem (principal rate time : ℝ) : 
  principal = 2100 →
  principal * (rate + 1) * time / 100 = principal * rate * time / 100 + 63 →
  time = 3 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3283_328379


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3283_328370

/-- A line is tangent to a circle if and only if the discriminant of the resulting quadratic equation is zero -/
theorem line_tangent_to_circle (a b : ℤ) : 
  (∃ x y : ℝ, y - 1 = (4 - a*x - b) / b ∧ 
              b^2*(x-1)^2 + (a*x+b-4)^2 - b^2 = 0 ∧ 
              (a*b - 4*a - b^2)^2 = (a^2 + b^2)*(b - 4)^2) ↔ 
  ((a = 12 ∧ b = 5) ∨ (a = -4 ∧ b = 3) ∨ (a = 8 ∧ b = 6) ∨ 
   (a = 0 ∧ b = 2) ∨ (a = 6 ∧ b = 8) ∨ (a = 2 ∧ b = 0) ∨ 
   (a = 5 ∧ b = 12) ∨ (a = 3 ∧ b = -4)) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3283_328370


namespace NUMINAMATH_CALUDE_bob_raised_beds_l3283_328359

/-- Represents the dimensions of a raised bed -/
structure BedDimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the number of planks needed for one raised bed -/
def planksPerBed (dims : BedDimensions) (plankWidth : ℕ) : ℕ :=
  2 * dims.height * (dims.length / plankWidth) + 1

/-- Calculates the number of raised beds that can be constructed -/
def numberOfBeds (dims : BedDimensions) (plankWidth : ℕ) (totalPlanks : ℕ) : ℕ :=
  totalPlanks / planksPerBed dims plankWidth

/-- Theorem: Bob can construct 10 raised beds -/
theorem bob_raised_beds :
  let dims : BedDimensions := { height := 2, width := 2, length := 8 }
  let plankWidth := 1
  let totalPlanks := 50
  numberOfBeds dims plankWidth totalPlanks = 10 := by
  sorry

end NUMINAMATH_CALUDE_bob_raised_beds_l3283_328359


namespace NUMINAMATH_CALUDE_simplify_fraction_l3283_328394

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 147) = 5 * Real.sqrt 3 / 72 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3283_328394


namespace NUMINAMATH_CALUDE_expected_expenditure_2017_l3283_328366

/-- Represents the average income over five years -/
def average_income : ℝ := 10

/-- Represents the average expenditure over five years -/
def average_expenditure : ℝ := 8

/-- The slope of the regression line -/
def b_hat : ℝ := 0.76

/-- The y-intercept of the regression line -/
def a_hat : ℝ := average_expenditure - b_hat * average_income

/-- The regression function -/
def regression_function (x : ℝ) : ℝ := b_hat * x + a_hat

/-- The income in 10,000 yuan for which we want to predict the expenditure -/
def income_2017 : ℝ := 15

theorem expected_expenditure_2017 : 
  regression_function income_2017 = 11.8 := by sorry

end NUMINAMATH_CALUDE_expected_expenditure_2017_l3283_328366


namespace NUMINAMATH_CALUDE_number_exceeding_80_percent_l3283_328362

theorem number_exceeding_80_percent : ∃ x : ℝ, x = 0.8 * x + 120 ∧ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_80_percent_l3283_328362


namespace NUMINAMATH_CALUDE_range_of_a_for_max_and_min_l3283_328361

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a + 2)*x + 1

/-- The theorem stating the range of a for which f has both a maximum and a minimum -/
theorem range_of_a_for_max_and_min (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) → (a > 2 ∨ a < -1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_max_and_min_l3283_328361


namespace NUMINAMATH_CALUDE_oranges_count_l3283_328342

theorem oranges_count (joan_initial : ℕ) (tom_initial : ℕ) (sara_sold : ℕ) (christine_gave : ℕ)
  (h1 : joan_initial = 75)
  (h2 : tom_initial = 42)
  (h3 : sara_sold = 40)
  (h4 : christine_gave = 15) :
  joan_initial + tom_initial - sara_sold + christine_gave = 92 :=
by sorry

end NUMINAMATH_CALUDE_oranges_count_l3283_328342


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_g_l3283_328327

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 5| + |x - 3| - |3*x - 15|

-- Define the domain
def domain : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }

-- Theorem statement
theorem sum_of_max_and_min_g :
  ∃ (max_g min_g : ℝ),
    (∀ x ∈ domain, g x ≤ max_g) ∧
    (∃ x ∈ domain, g x = max_g) ∧
    (∀ x ∈ domain, min_g ≤ g x) ∧
    (∃ x ∈ domain, g x = min_g) ∧
    (max_g + min_g = -2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_g_l3283_328327


namespace NUMINAMATH_CALUDE_green_blue_difference_l3283_328351

/-- Represents the color of a disk -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  color_sum : blue + yellow + green = total
  ratio : ∃ (k : ℕ), blue = 3 * k ∧ yellow = 7 * k ∧ green = 8 * k

/-- The main theorem to prove -/
theorem green_blue_difference (bag : DiskBag) 
  (h_total : bag.total = 108) :
  bag.green - bag.blue = 30 := by
  sorry

#check green_blue_difference

end NUMINAMATH_CALUDE_green_blue_difference_l3283_328351


namespace NUMINAMATH_CALUDE_unit_circle_point_movement_l3283_328339

theorem unit_circle_point_movement (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) :
  let P₀ : ℝ × ℝ := (1, 0)
  let P₁ : ℝ × ℝ := (Real.cos α, Real.sin α)
  let P₂ : ℝ × ℝ := (Real.cos (α + Real.pi / 4), Real.sin (α + Real.pi / 4))
  P₂.1 = -3/5 → P₁.2 = 7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_point_movement_l3283_328339


namespace NUMINAMATH_CALUDE_expression_simplification_l3283_328309

theorem expression_simplification (a b : ℝ) 
  (h : |2*a - 1| + (b + 4)^2 = 0) : 
  a^3*b - a^2*b^3 - 1/2*(4*a*b - 6*a^2*b^3 - 1) + 2*(a*b - a^2*b^3) = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3283_328309


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_43_l3283_328398

theorem smallest_four_digit_divisible_by_43 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 → n ≥ 1032 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_43_l3283_328398


namespace NUMINAMATH_CALUDE_john_pays_21_l3283_328354

/-- The amount John pays for candy bars -/
def john_payment (total_bars : ℕ) (dave_bars : ℕ) (price_per_bar : ℚ) : ℚ :=
  (total_bars - dave_bars : ℚ) * price_per_bar

/-- Theorem: John pays $21 for candy bars -/
theorem john_pays_21 :
  john_payment 20 6 (3/2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_john_pays_21_l3283_328354


namespace NUMINAMATH_CALUDE_unique_prime_sum_10003_l3283_328393

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n

theorem unique_prime_sum_10003 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10003 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_10003_l3283_328393


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3283_328356

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_two : a + b + c + d = 2) :
  (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 
   1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 9 ∧
  ∃ (a' b' c' d' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧
    a' + b' + c' + d' = 2 ∧
    (1 / (a' + b') + 1 / (a' + c') + 1 / (a' + d') + 
     1 / (b' + c') + 1 / (b' + d') + 1 / (c' + d')) = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3283_328356


namespace NUMINAMATH_CALUDE_coordinates_of_P_fixed_points_of_N_min_length_AB_l3283_328334

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define point P on line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define tangent PA
def tangent_PA (p : Point_P) (a : ℝ × ℝ) : Prop :=
  circle_M a.1 a.2 ∧ 
  (p.x - a.1)^2 + (p.y - a.2)^2 = ((0 - p.x)^2 + (4 - p.y)^2)

-- Theorem 1
theorem coordinates_of_P : 
  ∃ (p : Point_P), (∃ (a : ℝ × ℝ), tangent_PA p a ∧ (p.x - a.1)^2 + (p.y - a.2)^2 = 12) →
  (p.x = 0 ∧ p.y = 0) ∨ (p.x = 16/5 ∧ p.y = 8/5) :=
sorry

-- Define circle N (circumcircle of triangle PAM)
def circle_N (p : Point_P) (x y : ℝ) : Prop :=
  (2*x + y - 4) * p.y - (x^2 + y^2 - 4*y) = 0

-- Theorem 2
theorem fixed_points_of_N :
  ∀ (p : Point_P), 
    (circle_N p 0 4 ∧ circle_N p (8/5) (4/5)) ∧
    (∀ (x y : ℝ), circle_N p x y → (x = 0 ∧ y = 4) ∨ (x = 8/5 ∧ y = 4/5)) :=
sorry

-- Define chord AB
def chord_AB (p : Point_P) (x y : ℝ) : Prop :=
  2 * p.y * x + (p.y - 4) * y + 12 - 4 * p.y = 0

-- Theorem 3
theorem min_length_AB :
  ∃ (p : Point_P), 
    (∀ (p' : Point_P), 
      (∀ (x y : ℝ), chord_AB p x y → 
        (x - 0)^2 + (y - 4)^2 ≥ (x - 0)^2 + (y - 4)^2)) ∧
    (∃ (a b : ℝ × ℝ), chord_AB p a.1 a.2 ∧ chord_AB p b.1 b.2 ∧ 
      (a.1 - b.1)^2 + (a.2 - b.2)^2 = 11) :=
sorry

end NUMINAMATH_CALUDE_coordinates_of_P_fixed_points_of_N_min_length_AB_l3283_328334


namespace NUMINAMATH_CALUDE_garden_area_is_400_l3283_328335

/-- A rectangular garden with specific walking distances -/
structure Garden where
  length : ℝ
  width : ℝ
  length_total : ℝ
  perimeter_total : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ

/-- The garden satisfies the given conditions -/
def garden_satisfies_conditions (g : Garden) : Prop :=
  g.length_total = 2000 ∧
  g.perimeter_total = 2000 ∧
  g.length_walks = 50 ∧
  g.perimeter_walks = 20 ∧
  g.length_total = g.length * g.length_walks ∧
  g.perimeter_total = g.perimeter_walks * (2 * g.length + 2 * g.width)

/-- The theorem stating that a garden satisfying the conditions has an area of 400 square meters -/
theorem garden_area_is_400 (g : Garden) (h : garden_satisfies_conditions g) : 
  g.length * g.width = 400 := by
  sorry

#check garden_area_is_400

end NUMINAMATH_CALUDE_garden_area_is_400_l3283_328335


namespace NUMINAMATH_CALUDE_log_sum_equation_l3283_328307

theorem log_sum_equation (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 →
  y = 3 ^ (10 / 3) := by
sorry

end NUMINAMATH_CALUDE_log_sum_equation_l3283_328307


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3283_328345

theorem arithmetic_mean_problem (x : ℝ) : 
  (10 + 20 + 60) / 3 = (10 + 40 + x) / 3 + 5 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3283_328345


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l3283_328385

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∃ (k : ℕ), (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) = 15 * k) ∧
  (∀ (d : ℕ), d > 15 → ∃ (m : ℕ), Even m ∧ m > 0 ∧
    ¬(∃ (k : ℕ), (m + 3) * (m + 5) * (m + 7) * (m + 9) * (m + 11) = d * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l3283_328385


namespace NUMINAMATH_CALUDE_calculate_fifth_subject_score_l3283_328357

/-- Given a student's scores in 4 subjects and the average of all 5 subjects,
    calculate the score in the 5th subject. -/
theorem calculate_fifth_subject_score
  (math_score science_score english_score biology_score : ℕ)
  (average_score : ℚ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 62)
  (h4 : biology_score = 85)
  (h5 : average_score = 74)
  : ∃ (social_studies_score : ℕ),
    (math_score + science_score + english_score + biology_score + social_studies_score : ℚ) / 5 = average_score ∧
    social_studies_score = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_calculate_fifth_subject_score_l3283_328357


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3283_328332

-- Define the quadratic function
def y (x m : ℝ) : ℝ := (x - 1) * (x - m + 1)

-- State the theorem
theorem quadratic_function_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, y x m > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3283_328332


namespace NUMINAMATH_CALUDE_light_wattage_increase_l3283_328331

theorem light_wattage_increase (original_wattage new_wattage : ℝ) 
  (h1 : original_wattage = 80)
  (h2 : new_wattage = 100) :
  (new_wattage - original_wattage) / original_wattage * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_light_wattage_increase_l3283_328331


namespace NUMINAMATH_CALUDE_horner_method_example_l3283_328372

def f (x : ℝ) : ℝ := 9 + 15*x - 8*x^2 - 20*x^3 + 6*x^4 + 3*x^5

theorem horner_method_example : f 4 = 3269 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_example_l3283_328372


namespace NUMINAMATH_CALUDE_sum_convergence_implies_k_value_l3283_328313

/-- Given a real number k > 1 such that the sum of (7n-3)/k^n from n=1 to infinity equals 20/3,
    prove that k = 1.9125 -/
theorem sum_convergence_implies_k_value (k : ℝ) 
  (h1 : k > 1)
  (h2 : ∑' n, (7 * n - 3) / k^n = 20/3) : 
  k = 1.9125 := by
  sorry

end NUMINAMATH_CALUDE_sum_convergence_implies_k_value_l3283_328313


namespace NUMINAMATH_CALUDE_sin_B_in_triangle_ABC_l3283_328367

theorem sin_B_in_triangle_ABC (a b : ℝ) (sin_A : ℝ) :
  a = 15 →
  b = 10 →
  sin_A = (Real.sqrt 3) / 2 →
  (b * sin_A) / a = (Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_sin_B_in_triangle_ABC_l3283_328367


namespace NUMINAMATH_CALUDE_chess_club_mixed_groups_l3283_328389

/-- Represents the chess club scenario -/
structure ChessClub where
  total_children : ℕ
  num_groups : ℕ
  children_per_group : ℕ
  boy_vs_boy_games : ℕ
  girl_vs_girl_games : ℕ

/-- The number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ :=
  (club.total_children - club.boy_vs_boy_games - club.girl_vs_girl_games) / 2

/-- The theorem stating the number of mixed groups in the given scenario -/
theorem chess_club_mixed_groups :
  let club := ChessClub.mk 90 30 3 30 14
  mixed_groups club = 23 := by sorry

end NUMINAMATH_CALUDE_chess_club_mixed_groups_l3283_328389


namespace NUMINAMATH_CALUDE_yujin_wire_length_l3283_328353

/-- The length of Yujin's wire given Junhoe's wire length and the ratio --/
theorem yujin_wire_length (junhoe_length : ℝ) (ratio : ℝ) (h1 : junhoe_length = 134.5) (h2 : ratio = 1.06) :
  junhoe_length * ratio = 142.57 := by
  sorry

end NUMINAMATH_CALUDE_yujin_wire_length_l3283_328353


namespace NUMINAMATH_CALUDE_rational_roots_count_l3283_328304

/-- A polynomial with integer coefficients of the form 9x^4 + a₃x³ + a₂x² + a₁x + 15 = 0 -/
def IntPolynomial (a₃ a₂ a₁ : ℤ) (x : ℚ) : ℚ :=
  9 * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + 15

/-- The set of possible rational roots of the polynomial -/
def PossibleRoots : Finset ℚ :=
  {1, -1, 3, -3, 5, -5, 15, -15, 1/3, -1/3, 5/3, -5/3, 1/9, -1/9, 5/9, -5/9}

theorem rational_roots_count (a₃ a₂ a₁ : ℤ) :
  (PossibleRoots.filter (fun x => IntPolynomial a₃ a₂ a₁ x = 0)).card ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_rational_roots_count_l3283_328304


namespace NUMINAMATH_CALUDE_leos_laundry_bill_l3283_328396

/-- The total bill amount for Leo's laundry -/
def total_bill_amount (trousers_count : ℕ) (initial_shirts_count : ℕ) (missing_shirts_count : ℕ) 
  (trouser_price : ℕ) (shirt_price : ℕ) : ℕ :=
  trousers_count * trouser_price + (initial_shirts_count + missing_shirts_count) * shirt_price

/-- Theorem stating that Leo's total bill amount is $140 -/
theorem leos_laundry_bill : 
  total_bill_amount 10 2 8 9 5 = 140 := by
  sorry

#eval total_bill_amount 10 2 8 9 5

end NUMINAMATH_CALUDE_leos_laundry_bill_l3283_328396


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l3283_328344

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 7) (h_cylinder : r_cylinder = 4) :
  let v_sphere := (4 / 3) * Real.pi * r_sphere ^ 3
  let h_cylinder := 2 * r_sphere
  let v_cylinder := Real.pi * r_cylinder ^ 2 * h_cylinder
  v_sphere - v_cylinder = (700 / 3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l3283_328344


namespace NUMINAMATH_CALUDE_total_fish_count_l3283_328378

/-- The number of fish tanks James has -/
def num_tanks : ℕ := 3

/-- The number of fish in the first tank -/
def fish_in_first_tank : ℕ := 20

/-- The number of fish in each of the other tanks -/
def fish_in_other_tanks : ℕ := 2 * fish_in_first_tank

/-- The total number of fish in all tanks -/
def total_fish : ℕ := fish_in_first_tank + 2 * fish_in_other_tanks

theorem total_fish_count : total_fish = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l3283_328378


namespace NUMINAMATH_CALUDE_circle_and_locus_equations_l3283_328321

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a : ℝ), (x - a)^2 + (y - (2 - a))^2 = (a - 4)^2 + (2 - a)^2 ∧
              (x - a)^2 + (y - (2 - a))^2 = (a - 2)^2 + (2 - a - 2)^2

-- Define the locus of midpoint M
def locus_M (x y : ℝ) : Prop :=
  ∃ (x₁ y₁ : ℝ), circle_C x₁ y₁ ∧ x = (x₁ + 5) / 2 ∧ y = y₁ / 2

theorem circle_and_locus_equations :
  (∀ x y, circle_C x y ↔ (x - 2)^2 + y^2 = 4) ∧
  (∀ x y, locus_M x y ↔ x^2 - 7*x + y^2 + 45/4 = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_and_locus_equations_l3283_328321


namespace NUMINAMATH_CALUDE_quadratic_root_k_l3283_328387

theorem quadratic_root_k (k : ℝ) : (1 : ℝ)^2 + k * 1 - 3 = 0 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_l3283_328387


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3283_328371

def z : ℂ := (1 + Complex.I) * (1 - 2 * Complex.I)

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3283_328371


namespace NUMINAMATH_CALUDE_product_purely_imaginary_l3283_328300

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the product function
def product (x : ℝ) : ℂ := (x - i) * ((x + 2) - i) * ((x + 4) - i)

-- Define the property of being purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

-- State the theorem
theorem product_purely_imaginary (x : ℝ) : 
  isPurelyImaginary (product x) ↔ 
  (x = -3 ∨ x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) :=
sorry

end NUMINAMATH_CALUDE_product_purely_imaginary_l3283_328300


namespace NUMINAMATH_CALUDE_bug_return_probability_l3283_328360

/-- Probability of returning to the starting vertex after n steps -/
def Q (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1/4 : ℚ) + (1/2 : ℚ) * Q (n-1)

/-- Regular tetrahedron with bug movement rules -/
theorem bug_return_probability :
  Q 6 = 354/729 := by sorry

end NUMINAMATH_CALUDE_bug_return_probability_l3283_328360


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3283_328352

theorem sum_of_roots_quadratic (x : ℝ) (h : x^2 - 3*x = 12) : 
  ∃ y : ℝ, y^2 - 3*y = 12 ∧ x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3283_328352
